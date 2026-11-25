/**
 *  @file include/usearch/basic/vram_apple_index.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Similarity Search over variable-length values in the unified memory of one Apple GPU.
 *
 *  The host graph the Apple basic faces share, the twin of @c vram_nvidia_index.hpp run as the
 *  dense Apple index runs: every column is a shared buffer the host reads and writes in place, the
 *  kernels compile from the embedded MSL at the first @c reserve, once per face's kits on each
 *  device, and a batch's claim is one compare-exchange that refuses a second batch with
 *  @c contended_k. A modality's face owns one over its kits, which name their MSL sources and kit
 *  type, the rung a launch's longest value needs, the lanes that rung scores a node with and the
 *  span one value prepares into, so dispatch never names a kit.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::snprintf` - the source's prelude
#include <cstring> // `std::memcpy`, `std::memset`

#include <array>      // `std::array` - a health check's entries per depth
#include <bit>        // `std::bit_ceil`, `std::has_single_bit`, `std::popcount`
#include <concepts>   // `std::same_as` - the sharded index takes 32-bit shards
#include <functional> // `std::hash` - the key map's default hash
#include <limits>     // `std::numeric_limits` - the node-ID type's sentinel
#include <span>       // `std::span` - the caller's node offsets, a snapshot's columns
#include <utility>    // `std::exchange` - a moved-from plan forgets its index

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the published words

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the records hold the kernel kind
#include <numkong/tensor.hpp>     // `nk::matrix_span` - the outputs of a search

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the source and the plans' host columns
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator_t` - the host keys' placement
#include "usearch/utilities/ram_sorts.hpp"         // `order_by_neighborhood`, `split_wide_neighborhoods`
#include "usearch/utilities/ram_tables.hpp"        // `keyed_nodes` - every node's key and every key's nodes
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_saved_columns` - a load's saved columns
#include "usearch/utilities/vram_apple_device.hpp" // `metal_traversal_context`, `metal_launch_t`
#include "usearch/utilities/vram_device.hpp"       // `vram_kernel_t` - the kernels a rung names
#include "usearch/utilities/vram_walks.hpp"        // `gpu_arena_shape_t`, `gpu_candidate_slots`
#include "usearch/utilities/vram_apple_shared.hpp" // `metal_counter_t`, `metal_extent_t` - read first
#include "usearch/basic/vram_apple_index.metal"    // `metal_basic_arguments_t` & the launch words alone, on a host
#include "usearch/basic/ram_index.hpp"             // `columns`, `columns_extents_t`, `refine_report_t`

namespace unum::usearch::basic {
using namespace unum::usearch::utilities;

template <typename index_type_, typename, typename, typename, typename>
    requires std::same_as<typename index_type_::node_id_t, u32_t>
struct vram_apple_indexes;

#pragma region Kernel Records

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

#if USEARCH_ARCH_METAL_

/** The launch words, then the MSL walks, insert, search, round and graph kernels over any face's
 *  kit, spliced last. */
inline constexpr char metal_basic_index_source_k[] = {
#embed "vram_apple_index.metal" suffix(, 0)
};

#else
/* A unit without Metal compiles no pipeline, and NVCC's front end knows no `#embed`. */
inline constexpr char metal_basic_index_source_k[] = {0};
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/** Bytes the prelude naming a face's kit takes, its terminator included. */
inline constexpr std::size_t metal_basic_prelude_capacity_k = 512;

/**
 *  @brief The MSL source of the kits @p kits_type_, assembled at its first ask, once per process.
 *
 *  It splices the contexts' vocabulary and queue, every piece of the kits' own sources in order, a
 *  prelude naming the kit type and its rungs, then the launch words and the kernels.
 *  Its address keys the library built from it, so every face's kits compile their own.
 */
template <typename kits_type_>
struct metal_basic_source {
    growing_buffer<char> text {global_allocator};

    metal_basic_source() noexcept {
        std::size_t bytes = sizeof(metal_context_source_k) + metal_basic_prelude_capacity_k +
                            sizeof(metal_basic_index_source_k);
        for (char const* const piece : kits_type_::metal_sources_k) bytes += std::strlen(piece) + 1;
        if (failed(text.resize_uninitialized(bytes))) return;
        splice_(metal_context_source_k);
        for (char const* const piece : kits_type_::metal_sources_k) splice_(piece);
        length_ += static_cast<std::size_t>(std::snprintf(             //
            text.data() + length_, metal_basic_prelude_capacity_k,     //
            "#define USEARCH_METAL_KIT_RUNGS %zu\n"                    //
            "namespace unum { namespace usearch { namespace basic {\n" //
            "using metal_kit_t = %s;\n"                                //
            "constant constexpr uint metal_kit_rungs_k = %zu;\n"       //
            "} } }\n",                                                 //
            kits_type_::rungs_k, kits_type_::metal_kit_name_k, kits_type_::rungs_k));
        std::memcpy(text.data() + length_, metal_basic_index_source_k, sizeof(metal_basic_index_source_k));
    }

  private:
    /** Appends @p piece and a line break past what the source holds so far. */
    void splice_(char const* piece) noexcept {
        std::size_t const piece_bytes = std::strlen(piece);
        std::memcpy(text.data() + length_, piece, piece_bytes);
        length_ += piece_bytes;
        text[length_++] = '\n';
    }

    std::size_t length_ = 0;
};

/** The MSL source of the kits @p kits_type_, null when the host could not hold it. */
template <typename kits_type_>
char const* metal_basic_source_of() noexcept {
    static metal_basic_source<kits_type_> const source;
    return source.text.empty() ? nullptr : source.text.data();
}

/** One exported kit kernel: the kernel and its register rung, zero where it keeps no ladder or
 *  walks the arena frontier. */
struct metal_basic_rung_t {
    vram_kernel_t kernel;
    u32_t slots;
};

/** Every instantiation each rung of a face's kits exports, which the host builds a pipeline for at
 *  the first @c reserve. */
inline constexpr metal_basic_rung_t metal_basic_rungs[] = {
    {vram_kernel_t::insert_k, 0},
    {vram_kernel_t::insert_k, 2},
    {vram_kernel_t::insert_k, 4},
    {vram_kernel_t::insert_k, 8},
    {vram_kernel_t::insert_k, 16},
    {vram_kernel_t::refine_k, 0},
    {vram_kernel_t::refine_k, 2},
    {vram_kernel_t::refine_k, 4},
    {vram_kernel_t::refine_k, 8},
    {vram_kernel_t::refine_k, 16},
    {vram_kernel_t::back_link_k, 0},
    {vram_kernel_t::search_k, 0},
    {vram_kernel_t::search_k, 2},
    {vram_kernel_t::search_k, 4},
    {vram_kernel_t::search_k, 8},
    {vram_kernel_t::search_k, 16},
    {vram_kernel_t::gather_neighborhoods_k, 0},
    {vram_kernel_t::restitch_k, 0},
};

/** The kernels a load queues, which each face's kits export once, on their first rung. */
inline constexpr vram_kernel_t metal_basic_load_kernels[] = {vram_kernel_t::load_values_k, vram_kernel_t::size_spans_k};

/** Every kernel over the neighbors matrix alone, which the contexts' source holds for every Apple
 *  design: the health check, the heal and a cut's partition. */
inline constexpr vram_kernel_t metal_graph_kernels[] = {
    vram_kernel_t::health_expand_k,   vram_kernel_t::health_recheck_k,    vram_kernel_t::order_islands_k,
    vram_kernel_t::health_islands_k,  vram_kernel_t::island_components_k, vram_kernel_t::health_count_links_k,
    vram_kernel_t::label_islands_k,   vram_kernel_t::heal_islands_k,      vram_kernel_t::partition_propose_k,
    vram_kernel_t::partition_count_k, vram_kernel_t::partition_accept_k,  vram_kernel_t::partition_fill_k,
};

/** Spells the @c host_name a kit kernel exports @p instantiated on kit rung @p rung under: the
 *  kernel's name, the kit rung, then a walking kernel's candidate slots, none for a load's. */
inline void metal_basic_kernel_symbol(vram_kernel_symbol_t& symbol, std::size_t rung,
                                      metal_basic_rung_t instantiated) noexcept {
    symbol = {};
    bool walks = true;
    switch (instantiated.kernel) {
    case vram_kernel_t::insert_k: symbol.append("usearch_basic_insert_kernel_"); break;
    case vram_kernel_t::refine_k: symbol.append("metal_refine_kernel_"); break;
    case vram_kernel_t::search_k: symbol.append("metal_search_kernel_"); break;
    case vram_kernel_t::back_link_k: symbol.append("metal_back_link_kernel_"), walks = false; break;
    case vram_kernel_t::gather_neighborhoods_k:
        symbol.append("metal_gather_neighborhoods_kernel_"), walks = false;
        break;
    case vram_kernel_t::restitch_k: symbol.append("metal_restitch_kernel_"), walks = false; break;
    case vram_kernel_t::load_values_k: symbol.append("metal_load_values_kernel"); return;
    case vram_kernel_t::size_spans_k: symbol.append("metal_size_spans_kernel"); return;
    default: return;
    }
    symbol.append_value(rung);
    if (!walks) return;
    symbol.append("_");
    symbol.append_value(instantiated.slots);
}

/** Spells the @c host_name a graph kernel exports under, as every Apple design spells it. */
inline void metal_graph_kernel_symbol(vram_kernel_symbol_t& symbol, vram_kernel_t kernel) noexcept {
    symbol = {};
    switch (kernel) {
    case vram_kernel_t::health_expand_k: symbol.append("metal_health_expand_kernel"); return;
    case vram_kernel_t::health_recheck_k: symbol.append("metal_health_recheck_kernel"); return;
    case vram_kernel_t::order_islands_k: symbol.append("metal_order_islands_kernel"); return;
    case vram_kernel_t::health_islands_k: symbol.append("metal_health_islands_kernel"); return;
    case vram_kernel_t::island_components_k: symbol.append("metal_island_components_kernel"); return;
    case vram_kernel_t::health_count_links_k: symbol.append("metal_health_count_links_kernel"); return;
    case vram_kernel_t::label_islands_k: symbol.append("metal_label_islands_kernel"); return;
    case vram_kernel_t::heal_islands_k: symbol.append("metal_heal_islands_kernel"); return;
    case vram_kernel_t::partition_propose_k: symbol.append("metal_partition_propose_kernel"); return;
    case vram_kernel_t::partition_count_k: symbol.append("metal_partition_count_kernel"); return;
    case vram_kernel_t::partition_accept_k: symbol.append("metal_partition_accept_kernel"); return;
    case vram_kernel_t::partition_fill_k: symbol.append("metal_partition_fill_kernel"); return;
    default: return;
    }
}

/** Threads per threadgroup: four simdgroups, one walk each. */
inline constexpr u32_t metal_threadgroup_threads_k = metal_simdgroups_k * metal_simdgroup_width_k;

/** Bytes of one launch's counter words. */
inline constexpr std::size_t metal_counters_bytes_k = metal_counter_words_k * sizeof(u32_t);

/** The device storage of a graph as host pointers into its shared buffers - the audit surface for
 *  tests, and what a load's hooks write into. */
template <typename node_id_type_>
struct metal_ragged_storage {
    byte_t* values = nullptr;
    u64_t* offsets = nullptr;
    node_id_type_* neighbors = nullptr;
    u32_t connectivity = 0;
    nk_kernel_kind_t kernel = nk_kernel_unknown_k;
};

/**
 *  @brief One run of a load as the kits' hook stores it: the run's saved values in the caller's
 *      memory, their element sums, the planned tape starts their stored values take, and the
 *      widest span the hook raises.
 *
 *  Memory is unified, so a hook storing values as they are copies them on the host; one converting
 *  them launches the kits' load kernels, which read the run staged into the context's staging.
 */
template <typename node_id_type_>
struct metal_load_step {
    using traversal_context_t = metal_traversal_context<metal_shared_allocator_t>;

    /** The scalars the kits' load kernels read: @c first_id and @c count name the run. */
    metal_basic_arguments_t arguments {};

    /** The run's saved values, and their element sums, @b [count+1], null for fixed widths. */
    bytes_view_t saved;
    u64_t const* saved_offsets = nullptr;

    /** Where the run's first stored value goes in the tape, and every node's planned start,
     *  @b [count+1], absolute in the tape. */
    byte_t* stored = nullptr;
    u64_t const* stored_offsets = nullptr;

    /** The widest span the run's values prepare into, which the hook raises. */
    std::size_t* span_bytes = nullptr;

    /** Copies the run's saved values into the tape as they are, for kits storing them so. */
    status_t copy_values() const noexcept {
        if (!saved.empty()) std::memcpy(stored, saved.data(), saved.size());
        return status_code_t::success_k;
    }

    /**
     *  @brief Commits the kits' load @p kernel over the run the plan staged, on the context's
     *      stream; the widest span it counts folds in once the run drains on that stream.
     *
     *  @par Statuses:
     *  - @c device_k The launch could not be encoded.
     */
    status_t launch_kits(vram_kernel_t kernel) const noexcept {
        metal_pipeline_state_t const state = kernel == vram_kernel_t::load_values_k ? pipelines_[0] : pipelines_[1];
        metal_launch_t launch {context_->api(), context_->stream(), state.pipeline};
        if (!launch) return status_code_t::device_k;
        launch.bind_bytes(&arguments, sizeof(arguments), metal_basic_arguments_buffer_k);
        launch.bind(*values_, metal_basic_values_buffer_k);
        launch.bind(*offsets_, metal_basic_offsets_buffer_k);
        launch.bind(*parameters_, metal_basic_parameters_buffer_k);
        launch.bind(context_->counters_, metal_basic_counters_buffer_k);
        launch.bind(context_->staging_, metal_basic_saved_buffer_k);
        launch.bind(context_->staging_, metal_basic_saved_offsets_buffer_k, saved_offsets_at_);
        launch.bind(*offsets_, metal_basic_stored_offsets_buffer_k, stored_offsets_at_);
        launch.bind(context_->staging_, metal_basic_span_words_buffer_k, span_word_at_);
        std::size_t const groups = divide_round_up<std::size_t>(arguments.count, metal_threadgroup_threads_k);
        return context_->submit_(launch, max_of<std::size_t>(groups, 1), metal_threadgroup_threads_k);
    }

  private:
    template <typename, typename, typename, typename>
    friend struct vram_apple_graph;

    /** Stages the run's saved values, their element sums and a zeroed span word, which the kits'
     *  load kernels read. */
    status_t stage_() noexcept {
        std::size_t const count = arguments.count;
        std::size_t const sums_bytes = saved_offsets ? (count + 1) * sizeof(u64_t) : 0;
        saved_offsets_at_ = round_up_to_multiple<std::size_t>(saved.size() + sizeof(u64_t), sizeof(u64_t));
        span_word_at_ = saved_offsets_at_ + round_up_to_multiple<std::size_t>(sums_bytes, sizeof(u64_t));
        bool fresh = false;
        if (status_t const grown = context_->ensure_(context_->staging_, span_word_at_ + sizeof(u64_t), fresh);
            failed(grown))
            return grown;
        byte_t* const staging = context_->staging_.host;
        if (!saved.empty()) std::memcpy(staging, saved.data(), saved.size());
        if (sums_bytes) std::memcpy(staging + saved_offsets_at_, saved_offsets, sums_bytes);
        std::memset(staging + span_word_at_, 0, sizeof(u64_t));
        return status_code_t::success_k;
    }

    /** The context, the pipelines of the two load kernels, the columns and the parameters a launch
     *  binds, and where the staging and the offsets column place the run. */
    traversal_context_t* context_ = nullptr;
    metal_pipeline_state_t pipelines_[2] = {};
    metal_buffer_t const* values_ = nullptr;
    metal_buffer_t const* offsets_ = nullptr;
    metal_buffer_t const* parameters_ = nullptr;
    std::size_t stored_offsets_at_ = 0, saved_offsets_at_ = 0, span_word_at_ = 0;
};

/** Kits whose selector needs a breadth of at least @c build_breadth_k candidates per build walk. */
template <typename kits_type_>
concept a_kit_set_with_build_breadth = requires { kits_type_::build_breadth_k; };

#pragma endregion Kernel Records

/**
 *  @brief Similarity Search over variable-length values in the unified memory of one Apple GPU.
 *
 *  Speaks the NVIDIA basic graph's verbs: a batch's IDs are one contiguous range, and its nodes
 *  turn visible to searches a whole batch at a time. Values sit back to back in one tape beside a
 *  column of value starts, and the neighbors matrix holds @c connectivity IDs per node; all three
 *  are shared buffers the first @c reserve allocates at the ceiling, which Metal commits lazily, so
 *  no growth ever moves them, no walk shares a hold with a growth, and the host reads and writes
 *  them in place. The kit rung a launch scores on is the narrowest whose nodes cover the longest
 *  one the launch meets, and the spans its walks prepare nodes into are as wide as the widest of
 *  those nodes needs.
 *
 *  A batch inserts in admission waves, each no wider than a quarter of the nodes already below it.
 *  Every wave publishes forward nodes alone, threading each pick onto its target's chain, and a
 *  back-link launch then merges every chain into the node it names; its nodes then walk again,
 *  re-select their own rows and back-link those picks too. Each launch counts into its own words,
 *  since nothing on a stream zeroes one between two command buffers. One batch holds the claim word
 *  from its claim until its drain, and another answers @c contended_k meanwhile, while searches on
 *  other streams keep running throughout.
 *
 *  A refine runs the dense Apple round, every barrier on the host over unified memory. A reshard
 *  moves entries on the host - a gather appends other graphs' nodes, a cut partitions on the device
 *  and scatters into its targets - then runs the same round on every target. Saves and loads cross
 *  the snapshot columns of the basic RAM index in place, values through the kits' codec. Plans
 *  close the claim word, or claim their nodes, for their whole lives.
 *
 *  A modality's face owns one as a member and forwards its verbs. Its MSL is written over 32-bit
 *  node IDs, so a graph over any other width answers @c unsupported_k at its first @c reserve.
 *
 *  @tparam node_id_type_ The node IDs of the neighbors matrix, the walks and the matches.
 *  @tparam kits_type_ The face's kits, held as a member: @c rungs_k kit rungs, narrowest first;
 *      @c metal_sources_k, the MSL pieces to splice, and @c metal_kit_name_k, the MSL kit type the
 *      walks instantiate over; @c rung_serving_ nodes of a length, @c lanes_per_entry_ of a rung,
 *      the @c span_bytes_ one value prepares into and the bound any loaded value of a length takes;
 *      @c upload_parameters_ and @c parameters_, the buffer its kernels read; and its codec,
 *      @c saved_bytes_, @c save_value_, @c stored_bytes_ and @c queue_load_values_.
 *  @tparam key_type_ The caller-visible identity of every node, trivially copyable.
 *  @tparam key_hash_type_ The key map's hash of a key.
 */
template <typename node_id_type_, typename kits_type_, typename key_type_ = uuid128_t,
          typename key_hash_type_ = std::hash<key_type_>>
struct vram_apple_graph {
    using traversal_context_t = metal_traversal_context<metal_shared_allocator_t>;
    using node_id_t = node_id_type_;
    using key_t = key_type_;
    using distance_t = distance_t;
    using candidates_count_t = u32_t;

    /** The node-ID space, less the sentinel. */
    inline static constexpr std::size_t max_node_count_k {std::size_t {(std::numeric_limits<node_id_t>::max)()} - 1u};

    /** The nodes the host keys span: the node-ID space, no further than the address space holds
     *  keys. */
    inline static constexpr std::size_t max_keyed_node_count_k = min_of<std::size_t>(
        max_node_count_k, (std::size_t {1} << address_bits_k) / sizeof(key_t));

    /** Every node's key on the host, and every key's nodes. */
    using keyed_t = keyed_nodes<key_t, node_id_t, chunk_schedule_spanning<max_keyed_node_count_k>,
                                max_keyed_node_count_k, forkunion_allocator_t, fu::standard_atomic_ref, key_hash_type_>;

    /** The strongest @c synchronization_t contract this design honors; nothing routes on it. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;

    /** Pins the graph to the storage allocator's device, its kits scoring under @p kernel where
     *  they cover several metrics, and keeps its host keys and key map on @p keys_allocator; the
     *  first @c reserve freezes the rest. */
    explicit vram_apple_graph(nk_kernel_kind_t kernel, metal_shared_allocator_t const& allocator = {},
                              forkunion_allocator_t const& keys_allocator = {}) noexcept
        : allocator_ {allocator}, kernel_ {kernel}, keyed_(keys_allocator) {
        device_.ordinal = allocator.device();
    }
    vram_apple_graph(vram_apple_graph const&) = delete;
    vram_apple_graph& operator=(vram_apple_graph const&) = delete;

#pragma region Measures

    /** Nodes visible to searches: the watermark every walk samples below and filters neighbors
     *  by. */
    std::size_t size() const noexcept {
        return fu::standard_atomic_ref<std::size_t const>(last_visible_id_).load(std::memory_order_acquire);
    }

    /** The first failure that left claimed nodes unpublished, kept for the graph's lifetime. */
    status_t status() const noexcept {
        status_code_t const code =
            fu::standard_atomic_ref<status_code_t const>(status_.code).load(std::memory_order_acquire);
        return {code, fu::standard_atomic_ref<char const* const>(status_.note).load(std::memory_order_acquire)};
    }

    /** Neighbors per node, zero until the first @c reserve freezes it. */
    std::size_t connectivity() const noexcept {
        return fu::standard_atomic_ref<std::size_t const>(connectivity_).load(std::memory_order_acquire);
    }

    /** Nodes mapped, which every claim within them takes without growing. */
    std::size_t capacity() const noexcept {
        return fu::standard_atomic_ref<std::size_t const>(mapped_nodes_).load(std::memory_order_acquire);
    }

    /** Value bytes mapped in the tape, less the margin past the last. */
    std::size_t bytes_capacity() const noexcept {
        std::size_t const mapped =
            fu::standard_atomic_ref<std::size_t const>(mapped_bytes_).load(std::memory_order_acquire);
        return mapped > tape_margin_k ? mapped - tape_margin_k : 0;
    }

    /** Bytes this graph occupies on the device and the host: what its columns map, which Metal
     *  commits as they are touched, its keys and key map; traversal scratch is the contexts'. */
    std::size_t memory_usage() const noexcept {
        return sizeof(vram_apple_graph) + bytes_capacity() + tape_margin_k + (capacity() + 1) * sizeof(u64_t) +
               capacity() * connectivity() * sizeof(node_id_t) + keyed_.memory_usage();
    }

    /** The pinned device's identity and limits, queried once at the first @c reserve. */
    metal_device_specs_t const& device_specs() const noexcept { return device_; }

    /** The context of the pinned device every launch goes through, the storage allocator's. */
    metal_context_t* context() const noexcept { return allocator_.context(); }

    /** The metric the kits score under where they cover several, unknown otherwise. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Host views of the shared storage, the audit surface for tests. */
    metal_ragged_storage<node_id_t> storage() const noexcept {
        return {values_tape_.host, reinterpret_cast<u64_t*>(offsets_column_.host),
                reinterpret_cast<node_id_t*>(neighbors_matrix_.host), static_cast<u32_t>(connectivity()), kernel_};
    }

    /** The face's kits, which a face configures before the first @c reserve freezes them. */
    kits_type_& kits() noexcept { return kits_; }

    /** Every node's key on the host, and every key's nodes; readers pass the watermark they
     *  acquired. */
    keyed_t const& keys() const noexcept { return keyed_; }

    /**
     *  @brief The key of the visible @p node.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p node is not visible.
     */
    expected<key_t> key_of(node_id_t node) const noexcept {
        if (static_cast<std::size_t>(node) >= size()) return {{}, status_code_t::config_mismatch_k};
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

    /**
     *  @brief Copies the neighbor rows of the nodes from @p first on, @c connectivity IDs each,
     *      into the host @p rows, raw as stored with sentinels and self-loops left in.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p rows hold no whole number of rows, or the rows pass @c size.
     */
    status_t neighbors(node_id_t first, std::span<node_id_t> rows) const noexcept {
        std::size_t const row_ids = connectivity();
        if (!row_ids || rows.size() % row_ids || std::size_t {first} + rows.size() / row_ids > size())
            return status_code_t::config_mismatch_k;
        std::memcpy(rows.data(), neighbor_rows_() + std::size_t {first} * row_ids, rows.size_bytes());
        return status_code_t::success_k;
    }

#pragma endregion Measures

#pragma region Reservation

    /** Requested geometry & capacity for a @c reserve() call; the connectivity and the ceiling are
     *  frozen by the first success. */
    struct reserve_capacity_t {
        std::size_t capacity = 0;
        connectivity_t connectivity = connectivity_32_k;

        /** Value bytes across every node to host. */
        std::size_t bytes = 0;

        /** Nodes the graph may ever host, clamped to the ID space and the device's working set;
         *  past it every claim answers @c overflow_k. */
        std::size_t ceiling = max_node_count_k;
    };

    /**
     *  @brief Ensures the graph can host at least @p config.capacity nodes of @p config.bytes
     *      value bytes in all, and answers the node capacity now in effect.
     *
     *  The first success builds the pipelines, queries the device and allocates the columns at
     *  the ceiling; later ones only map.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The face's kits take parameters, and none were set.
     *  - @c config_mismatch_k The connectivity is not a power of two, or differs from the frozen
     *    one.
     *  - @c unsupported_k The node IDs are not 32-bit, or a pipeline's simdgroups or threadgroups
     *    are narrower than the kernels need.
     *  - @c overflow_k The capacity passes the ID space or the ceiling, or the value bytes pass the
     *    tape.
     *  - @c device_k The device is not listed, or the source would not compile.
     *  - @c out_of_memory_k A column could not be allocated.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (!std::has_single_bit(static_cast<u32_t>(config.connectivity))) return {0, status_code_t::config_mismatch_k};
        if (config.capacity > max_node_count_k) return {capacity(), status_code_t::overflow_k};
        std::size_t const frozen = connectivity();
        if (frozen && frozen != config.connectivity) return {capacity(), status_code_t::config_mismatch_k};
        if (!frozen)
            if (status_t const bound = bind_(config); failed(bound)) return {0, bound};
        if (status_t const mapped = map_nodes_(config.capacity); failed(mapped)) return {capacity(), mapped};
        status_t const mapped_bytes = map_bytes_(config.bytes + tape_margin_k);
        return {capacity(), mapped_bytes};
    }

#pragma endregion Reservation

#pragma region Insertion

    /** Optional tuning knobs; zeros mean "derive from connectivity". */
    struct insert_config_t {

        /** How many candidates a construction walk keeps, rounded up to the register rungs of 64 to
         *  512 ranks and walked exactly past them; zero takes twice the connectivity. */
        candidates_count_t expansion_factor = 0;
    };

    /** Telemetry of one @c insert() batch, whose IDs are @b [first_id,new_size). */
    struct [[nodiscard]] insert_result_t {
        std::size_t new_size = 0;
        node_id_t first_id = (std::numeric_limits<node_id_t>::max)();
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        std::size_t tail_compactions = 0;
        std::size_t tail_dropped = 0;
        std::size_t visited_saturations = 0;

        /** Folds another shard's batch in: the spend adds, while @c new_size and @c first_id,
         *  which name one shard's IDs, stay as they were. */
        insert_result_t& operator+=(insert_result_t const& other) noexcept {
            visited_members += other.visited_members;
            computed_distances += other.computed_distances;
            restarts += other.restarts;
            tail_compactions += other.tail_compactions;
            tail_dropped += other.tail_dropped;
            visited_saturations += other.visited_saturations;
            return *this;
        }
    };

    /** The bytes an executor's run must hold before a context can insert the nodes of a tape, or
     *  search or refine any: none, as every Metal context grows its own buffers at its first verb,
     *  unified memory carving nothing from the executor's run. Kept so every vendor's face and
     *  cell sizes its executor alike. */
    template <typename... usage_types_>
    std::size_t traversal_memory_usage(usage_types_ const&...) const noexcept {
        return 0;
    }

    /**
     *  @brief One batch's nodes claimed ahead of their launch, as the IDs
     *      @b [first_id,first_id+nodes), so a caller files whatever else it keeps by them before
     *      they turn visible.
     *
     *  The claim holds until the drain, since a wave's merges rewrite nodes another batch's walks
     *  would read. A plan that dies before @ref insert gives its nodes back, and @ref insert spends
     *  it whatever it answers.
     */
    struct [[nodiscard]] insert_plan_t {

        /** The first claimed ID. */
        node_id_t first_id = (std::numeric_limits<node_id_t>::max)();

        /** Nodes claimed. */
        std::size_t entries = 0;

        insert_plan_t() noexcept = default;
        insert_plan_t(insert_plan_t&& other) noexcept
            : first_id(other.first_id), entries(other.entries), index_(std::exchange(other.index_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;
        ~insert_plan_t() noexcept {
            if (index_ && entries) index_->unclaim_(first_id, entries);
        }

        /**
         *  @brief Launches the claimed nodes of @p tape under @p keys, one node a claimed ID, as
         *      the one-shot @ref vram_apple_graph::insert describes.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent, or the graph was never reserved.
         *  - @c config_mismatch_k The @p keys or @p tape hold other than the claimed nodes, or
         *    @p context rides another device or cannot walk.
         *  - @c unsupported_k A value past every rung of the face's kits.
         *  - @c overflow_k The nodes pass the reserved range or the host keys, or their values pass
         *    the tape mapped so far and its ceiling.
         */
        expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape,
                                         traversal_context_t& context, insert_config_t config = {}) noexcept {
            if (!index_) return {{}, status_code_t::uninitialized_k};
            vram_apple_graph& index = *index_;
            auto const [launched, launched_status] = index.launch_(*this, keys, tape, context, config);
            if (succeeded(launched_status)) {
                index.publish_keys_({launched.first_id, static_cast<node_id_t>(launched.new_size)});
                index.publish_(launched.new_size);
            }
            return {launched, launched_status};
        }

      private:
        friend vram_apple_graph;

        /** The graph whose claim word holds the nodes; null once spent. */
        vram_apple_graph* index_ = nullptr;
    };

    /**
     *  @brief Claims @p nodes for one batch, exposing their first ID before any of them launches.
     *
     *  A kept failure refuses, and so do nodes past the reserved ceiling and another batch in
     *  flight, the last with @c contended_k.
     *
     *  @par Statuses:
     *  - @c overflow_k The nodes pass the ceiling.
     *  - @c contended_k Another batch or load is in flight, or the claim word is closed.
     */
    expected<insert_plan_t> plan_insert(std::size_t nodes) noexcept {
        insert_plan_t plan;
        if (status_t const kept = status(); failed(kept)) return {std::move(plan), kept};
        if (!nodes) {
            plan.first_id = static_cast<node_id_t>(claimed_());
            return {std::move(plan), status_code_t::success_k};
        }
        auto const [first_id, claim_status] = claim_(nodes);
        if (failed(claim_status)) return {std::move(plan), claim_status};
        plan.first_id = first_id;
        plan.entries = nodes;
        plan.index_ = this;
        return {std::move(plan), status_code_t::success_k};
    }

    /**
     *  @brief Inserts the nodes of @p tape under @p keys as the contiguous IDs
     *      @b [first_id,first_id+nodes): @ref plan_insert, then its plan's @c insert.
     *
     *  The admission waves commit on the context's stream, which then drains; the keys file and
     *  the watermark turns over the range. Another batch in flight answers @c contended_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The graph was never reserved.
     *  - @c config_mismatch_k The @p keys are not one per node, or the @p context rides another
     *    device or cannot walk.
     *  - @c unsupported_k A value past every rung of the face's kits.
     *  - @c overflow_k The nodes pass the ceiling, the reserved range or the host keys, or their
     *    values the reserved tape.
     *  - @c contended_k Another batch or load is in flight, or the claim word is closed.
     */
    expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        std::size_t const entries = tape.size();
        if (keys.size() != entries) return {{}, status_code_t::config_mismatch_k};
        if (!entries) return {{claimed_(), static_cast<node_id_t>(claimed_())}, status_code_t::success_k};
        auto [plan, plan_status] = plan_insert(entries);
        if (failed(plan_status)) return {{}, plan_status};
        return plan.insert(keys, tape, context, config);
    }

    /** Nodes claimed by batches, visible or not: where the next claim starts, and what a modality
     *  face's shard set measures its bound against. */
    std::size_t claimed_() const noexcept { return claims_.claimed(size()); }

    /** Closes the claim word at the watermark, so every claim refuses until @ref reopen_; refused
     *  while a batch or load is in flight. */
    status_t close_() noexcept { return claims_.close(size()); }

    /** Reopens a closed claim word at the watermark, past every node a load published meanwhile; an
     *  open one stays as it is. */
    void reopen_() noexcept { claims_.reopen(size()); }

    /** Drops the watermark to zero and touches no node, key or claim: what a sharded index makes of
     *  a shard it retires with its claims closed for good. */
    void retire_() noexcept {
        fu::standard_atomic_ref<std::size_t>(last_visible_id_).store(0, std::memory_order_release);
    }

    /**
     *  @brief Copies the values of the visible @p listed nodes back to back into @p outputs, their
     *      offsets into @p output_offsets, on the host over unified memory.
     *
     *  Each offset is one past its listed node from the base the first one holds, which the
     *  previous gather ended at.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p output_offsets are not one longer than @p listed, or a listed
     *    node is not visible.
     */
    status_t gather_tape_(std::span<node_id_t const> listed, byte_t* outputs,
                          std::span<u64_t> output_offsets) const noexcept {
        if (output_offsets.size() != listed.size() + 1) return status_code_t::config_mismatch_k;
        std::size_t const visible = size();
        u64_t const* const starts = value_starts_();
        u64_t cursor = output_offsets[0];
        for (std::size_t position = 0; position != listed.size(); ++position) {
            std::size_t const node = listed[position];
            if (node >= visible) return status_code_t::config_mismatch_k;
            u64_t const bytes = starts[node + 1] - starts[node];
            std::memcpy(outputs + cursor, values_tape_.host + starts[node], bytes);
            cursor += bytes;
            output_offsets[position + 1] = cursor;
        }
        return status_code_t::success_k;
    }

  private:

    /** A sharded index walks its shards below one snapshot of their watermarks, and publishes a
     *  batch on every shard inside one commit. */
    template <typename index_type_, typename, typename, typename, typename>
        requires std::same_as<typename index_type_::node_id_t, u32_t>
    friend struct vram_apple_indexes;

    /** Assigns @p keys and launches and drains the nodes @p plan claimed, spending it unless the
     *  arguments mismatch, and leaves them for @c publish_keys_ and @c publish_ to turn visible. */
    expected<insert_result_t> launch_(insert_plan_t& plan, std::span<key_t const> keys, tape_view<byte_t const> tape,
                                      traversal_context_t& context, insert_config_t config) noexcept {
        if (plan.index_ != this) return {{}, status_code_t::uninitialized_k};
        if (tape.offsets.size() != plan.entries + 1 || keys.size() != plan.entries)
            return {{}, status_code_t::config_mismatch_k};
        plan.index_ = nullptr;
        return launch_claimed_(plan.first_id, keys, tape, context, config);
    }

    /** Files every drained node of @p range under its key ahead of its watermark: the claim holder,
     *  or the closer of the word, is the map's one writer. */
    void publish_keys_(iota_view<node_id_t> range) noexcept { keyed_.publish(range); }

    /** Turns every node below @p nodes visible, the end of a batch @c launch_ drained, which also
     *  frees the claim word for the next batch. */
    void publish_(std::size_t nodes) noexcept {
        fu::standard_atomic_ref<std::size_t>(last_visible_id_).store(nodes, std::memory_order_release);
    }

    /** Empties a graph whose nodes moved out under its closed claim word: the tape's extent, the
     *  fills and the key map restart, and the watermark drops to zero. Nothing may search it. */
    void forget_() noexcept {
        tape_bytes_ = 0, longest_value_ = 0, span_bytes_ = 0, filled_nodes_ = 0;
        fu::standard_atomic_ref<std::size_t>(last_visible_id_).store(0, std::memory_order_release);
        keyed_.forget();
    }

    /** Launches the nodes claimed from @p first_id under @p keys, as @ref insert describes, without
     *  publishing them; every refusal before the first commit gives the claim back. */
    expected<insert_result_t> launch_claimed_(node_id_t first_id, std::span<key_t const> keys,
                                              tape_view<byte_t const> tape, traversal_context_t& context,
                                              insert_config_t config) noexcept {
        std::size_t const entries = tape.size();
        auto const refuse = [&](status_t failure) noexcept -> expected<insert_result_t> {
            unclaim_(first_id, entries);
            return {{}, failure};
        };
        if (!connectivity()) return refuse(status_code_t::uninitialized_k);
        if (context.device() != device_.ordinal || !context) return refuse(status_code_t::config_mismatch_k);
        if (status_t const kept = status(); failed(kept)) return refuse(kept);

        // Walks prepare stored nodes as much as the batch's, so the rung and spans cover both.
        std::size_t longest = longest_value_;
        u32_t const span_bytes = static_cast<u32_t>(max_of(span_bytes_, span_bytes_of_(tape, longest)));
        std::size_t const rung = kits_.rung_serving_(longest);
        if (rung == kits_type_::rungs_k) return refuse(status_code_t::unsupported_k);
        std::size_t const batch_bytes = tape.offsets[entries] - tape.offsets[0];
        std::size_t const total_nodes = first_id + entries;

        // Every carve and growth precedes the first commit, so a refusal publishes nothing.
        build_shape_t const shape = build_shape_(config.expansion_factor, span_bytes);
        std::size_t const waves = insert_waves_(first_id, total_nodes);
        std::size_t const links_bytes = (total_nodes + 2 * entries * connectivity()) * sizeof(node_id_t);
        std::size_t const touched_at = round_up_to_multiple<std::size_t>(links_bytes, sizeof(u64_t));
        std::size_t const counters_at = round_up_to_multiple<std::size_t>(touched_at + 2 * waves * sizeof(u32_t),
                                                                          default_alignment_k);
        std::size_t const launches = 4 * waves;
        bool fresh = false;
        if (status_t const staged = context.ensure_(context.staging_, counters_at + launches * metal_counters_bytes_k,
                                                    fresh);
            failed(staged))
            return refuse(staged);
        metal_basic_arguments_t arguments = arguments_();
        u32_t const blocks = device_.multiprocessor_count * resident_blocks_(rung, vram_kernel_t::insert_k, 0);
        if (status_t const sized = shape_arena_(context, arguments, shape.arena_shape, blocks, 2 * entries);
            failed(sized))
            return refuse(sized);
        if (status_t const grown = grow_nodes_(total_nodes); failed(grown)) return refuse(grown);
        if (status_t const mapped = map_bytes_(tape_bytes_ + batch_bytes + tape_margin_k); failed(mapped))
            return refuse(mapped);
        for (std::size_t offset = 0; offset != entries; ++offset)
            keyed_.assign(static_cast<node_id_t>(first_id + offset), keys[offset]);

        // The values and their starts cross in place, so every kernel reads the batch as stored.
        std::memcpy(values_tape_.host + tape_bytes_, tape.elements + tape.offsets[0], batch_bytes);
        u64_t* const starts = value_starts_();
        for (std::size_t offset = 0; offset != entries; ++offset)
            starts[first_id + offset + 1] = tape_bytes_ + (tape.offsets[offset + 1] - tape.offsets[0]);
        // Every chain starts empty, and each back-link launch empties the ones it merged.
        byte_t* const staging = context.staging_.host;
        std::memset(staging, 0xFF, total_nodes * sizeof(node_id_t));
        std::memset(staging + touched_at, 0, counters_at - touched_at + launches * metal_counters_bytes_k);
        std::memset(context.counters_.host, 0, context.counters_.bytes);

        arguments.span_bytes = span_bytes;
        arguments.expansion = shape.expansion;
        status_t const launched = launch_insert_(arguments, shape, first_id, total_nodes, rung, blocks, touched_at,
                                                 counters_at, context);
        // A failure after a commit is kept, as the committed nodes stay claimed.
        if (failed(launched)) return failed(status()) ? expected<insert_result_t> {{}, launched} : refuse(launched);

        tape_bytes_ += batch_bytes;
        longest_value_ = longest;
        span_bytes_ = span_bytes;
        if (status_t const drained = context.wait(); failed(drained)) return {{}, fail_(drained)};
        fold_counters_(context, staging + counters_at, launches);

        vram_telemetry_block_t const spent = context.spent();
        insert_result_t result;
        result.new_size = total_nodes;
        result.first_id = first_id;
        result.visited_members = spent.hops;
        result.computed_distances = spent.scored;
        result.restarts = spent.restarts;
        result.tail_compactions = spent.tail_compactions;
        result.tail_dropped = spent.tail_dropped;
        result.visited_saturations = spent.visited_saturations;
        return {result, status_code_t::success_k};
    }

  public:

#pragma endregion Insertion

#pragma region Search

    /** Optional tuning knobs of @c search(); zeros mean "derive from connectivity". */
    struct search_config_t {

        /** How many candidates a walk keeps, walked as given when it covers @c count and widened to
         *  @c count otherwise; zero takes twice the connectivity. */
        candidates_count_t expansion_factor = 0;

        /** Matches reported per query. */
        candidates_count_t count = 10;
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

        /** Folds another batch or shard in, every count adding. */
        search_result_t& operator+=(search_result_t const& other) noexcept {
            count += other.count;
            visited_members += other.visited_members;
            computed_distances += other.computed_distances;
            restarts += other.restarts;
            tail_compactions += other.tail_compactions;
            tail_dropped += other.tail_dropped;
            visited_saturations += other.visited_saturations;
            return *this;
        }
    };

    /**
     *  @brief Finds the nearest published neighbors of every query, ascending by distance with
     *      sentinel IDs marking absent matches. Safe to overlap with insertions on other streams.
     *
     *  @param[in] queries The queries, one tape entry of bytes each.
     *  @param[out] result_ids IDs of the matches, @b [queries,count].
     *  @param[out] result_distances Distances of the same matches.
     *  @param[in] config The search breadth and traversal.
     *  @param[in,out] context The stream the launch rides; drained before returning.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The graph was never reserved.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk, or the outputs
     *    hold other than one row per query or fewer than @c count columns.
     *  - @c unsupported_k A query past every rung of the face's kits.
     */
    expected<search_result_t> search(tape_view<byte_t const> queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, search_config_t config,
                                     traversal_context_t& context) noexcept {
        return search_below_(queries, result_ids, result_distances, context, config, size());
    }

  private:

    /** The walk behind @ref search, seeded and bounded below @p visible, the watermark the caller
     *  took: the graph's own, or one a sharded index snapshot consistent across its shards. */
    expected<search_result_t> search_below_(tape_view<byte_t const> queries, nk::matrix_span<node_id_t> result_ids,
                                            nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                            search_config_t config, std::size_t visible) noexcept {
        std::size_t const queries_count = queries.size();
        if (!queries_count) return {{}, status_code_t::success_k};
        if (!connectivity()) return {{}, status_code_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_code_t::config_mismatch_k};
        if (result_ids.extent(0) != queries_count || result_ids.extent(1) < config.count ||
            result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)
            return {{}, status_code_t::config_mismatch_k};
        std::size_t longest = 0;
        u32_t const span_bytes = static_cast<u32_t>(span_bytes_of_(queries, longest));
        std::size_t const rung = kits_.rung_serving_(longest);
        if (rung == kits_type_::rungs_k) return {{}, status_code_t::unsupported_k};

        // The staging holds the queries' bytes from a word boundary, their offsets, the outputs.
        std::size_t const batch_bytes = queries.offsets[queries_count] - queries.offsets[0];
        std::size_t const results_count = queries_count * config.count;
        std::size_t const offsets_at = round_up_to_multiple<std::size_t>(batch_bytes + tape_margin_k, sizeof(u64_t));
        std::size_t const ids_at = offsets_at + (queries_count + 1) * sizeof(u64_t);
        std::size_t const distances_at = round_up_to_multiple<std::size_t>(ids_at + results_count * sizeof(node_id_t),
                                                                           sizeof(u64_t));
        bool fresh = false;
        if (status_t const staged = context.ensure_(context.staging_, distances_at + results_count * sizeof(distance_t),
                                                    fresh);
            failed(staged))
            return {{}, staged};
        u32_t const slots = walk_slots_(config);
        gpu_arena_shape_t const arena_shape = walk_arena_shape_(config, span_bytes);
        u32_t const blocks = walk_blocks_(queries_count, rung, slots);
        metal_basic_arguments_t arguments = arguments_();
        if (status_t const sized = shape_arena_(context, arguments, arena_shape, blocks, queries_count); failed(sized))
            return {{}, sized};
        byte_t* const staging = context.staging_.host;
        std::memcpy(staging, queries.elements + queries.offsets[0], batch_bytes);
        std::memset(staging + batch_bytes, 0, offsets_at - batch_bytes);
        u64_t* const staged_offsets = reinterpret_cast<u64_t*>(staging + offsets_at);
        for (std::size_t query = 0; query <= queries_count; ++query)
            staged_offsets[query] = queries.offsets[query] - queries.offsets[0];
        std::memset(context.counters_.host, 0, context.counters_.bytes);

        node_id_t* const ids = reinterpret_cast<node_id_t*>(staging + ids_at);
        distance_t* const distances = reinterpret_cast<distance_t*>(staging + distances_at);
        if (visible) {
            arguments.queries_count = static_cast<u32_t>(queries_count);
            arguments.sampling_bound = static_cast<u32_t>(visible);
            arguments.span_bytes = span_bytes;
            arguments.expansion = walk_expansion_(config);
            arguments.selection_nodes = walk_selection_nodes_(arguments.expansion, slots);
            arguments.concurrent_walks = device_.multiprocessor_count *
                                         resident_blocks_(rung, vram_kernel_t::search_k, slots) *
                                         walkers_per_block_(rung);
            arguments.filling_nodes = filling_nodes<u32_t>(arguments.expansion, static_cast<u32_t>(connectivity()));
            arguments.count = config.count;
            metal_launch_t launch {context.api(), context.stream(),
                                   kits_pipeline_(rung, vram_kernel_t::search_k, slots).pipeline};
            if (!launch) return {{}, status_code_t::device_k};
            bind_kit_storage_(launch, arguments, context);
            launch.bind(context.staging_, metal_basic_staging_buffer_k);
            launch.bind(context.staging_, metal_basic_query_offsets_buffer_k, offsets_at);
            launch.bind(context.staging_, metal_basic_ids_buffer_k, ids_at);
            launch.bind(context.staging_, metal_basic_distances_buffer_k, distances_at);
            if (status_t const submitted = context.submit_(launch, blocks, metal_threadgroup_threads_k);
                failed(submitted))
                return {{}, submitted};
            if (status_t const drained = context.wait(); failed(drained)) return {{}, drained};
        }
        // An empty graph matches nothing: every rank is a sentinel.
        else
            for (std::size_t slot = 0; slot != results_count; ++slot)
                ids[slot] = (std::numeric_limits<node_id_t>::max)();
        for (std::size_t query = 0; query != queries_count; ++query)
            for (std::size_t rank = 0; rank != config.count; ++rank) {
                result_ids(query, rank) = ids[query * config.count + rank];
                result_distances(query, rank) = distances[query * config.count + rank];
            }
        vram_telemetry_block_t const spent = context.spent();
        search_result_t result;
        result.count = config.count;
        result.visited_members = spent.hops;
        result.computed_distances = spent.scored;
        result.restarts = spent.restarts;
        result.tail_compactions = spent.tail_compactions;
        result.tail_dropped = spent.tail_dropped;
        result.visited_saturations = spent.visited_saturations;
        return {result, status_code_t::success_k};
    }

  public:

#pragma endregion Search

#pragma region Refinement

    /** A round's knobs, as the RAM tower names them. */
    struct refine_config_t {

        /** Bytes of threadgroup memory one GPU core keeps, half for a draw's hubs and half for a
         *  neighborhood's entries, each the mean stored value; zero takes the device's, and no more
         *  than the device's is ever taken. */
        std::size_t cache_bytes = 0;
    };

    /** A reshard's knobs, as the RAM tower names them: its rounds', the most entries a sharded
     *  target's shard holds, which no device graph honours, and the seed a cut grows parts by. */
    struct reshard_config_t {
        refine_config_t refine {};
        std::size_t shard_capacity = 0;
        u64_t seed = 0;
    };

    using refine_report_t = basic::refine_report_t;
    using reachability_t = utilities::reachability_t;

    struct refine_plan_t;
    struct reshard_plan_t;

  private:

    /** Who reopens the claim word a refine plan's graph holds closed. */
    enum class claim_t : u8_t {

        /** The caller that closed it, such as the reshard plan running its rounds, or nobody for
         *  an unbound plan. */
        caller_reopens_k,

        /** The plan at its death, since @ref plan_refine closed it. */
        plan_reopens_k,
    };

    /** How a round over @c entries entries sizes its columns: the hubs one gather draws at most and
     *  the most entries a neighborhood holds, as RAM names them, then the room for the hubs of
     *  every gather's runs, the runs, and the threadgroups a restitch runs. */
    struct round_shape_t {
        std::size_t entries = 0, hubs_per_draw = 0, capacity = 0;
        std::size_t hubs = 0, runs = 0, groups = 0;
    };

    /** Where a health check and a round keep each column in a context's round buffer, in bytes, as
     *  dense Apple lays them out: the counters; the depths both ways, the in-link and shallower
     *  in-link counts, and the reached, reaching-root, two-ways and changed bits; then the islands'
     *  columns, which the round's alias: each entry's neighborhood, the entries in neighborhood
     *  order, their starts with room for splits, a gather's hubs, its runs, the entries before each
     *  run, and one gather's or restitch's scratch slice per simdgroup. */
    struct round_layout_t {
        std::size_t counters = 0, depths_from_root = 0, depths_to_root = 0, in_links = 0, shallower_in_links = 0;
        std::size_t reached = 0, reaches_root = 0, two_ways_to_root = 0, changed = 0;
        std::size_t island_nodes = 0, island_components = 0, island_labels = 0, island_parents = 0;
        std::size_t island_entries = 0, island_roles = 0;
        std::size_t entry_neighborhoods = 0, neighborhood_entries = 0, neighborhood_offsets = 0, hub_nodes = 0;
        std::size_t runs = 0, run_items = 0, stitch = 0, bytes = 0;
    };

    /** One part of a cut as the host assigns it: its nodes, those given slots so far, its value
     *  bytes scattered so far and its longest value. */
    struct split_part_t {
        std::size_t entries = 0, assigned_nodes = 0, bytes = 0, longest = 0;
    };

    /** A cut's moves, empty for a gather. */
    struct split_t {

        /** The graph the cut empties, while it still holds its nodes and its claim word. */
        vram_apple_graph* source = nullptr;

        /** The source's nodes at planning, and the most nodes one part takes: those over the
         *  targets, rounded up. */
        std::size_t entries = 0, part_capacity = 0;

        /** The partition's shared buffer, laid out as @c split_layout_ places it, which the host
         *  reads in place between the batches. */
        metal_buffer_t device_bytes;

        /** The growth round the next batch commits from. */
        u32_t next_round = 0;

        /** The parts as the host assigns them. */
        growing_buffer<split_part_t> parts {global_allocator};
    };

  public:

    /**
     *  @brief One health check on the device over the neighbors matrix alone, dense Apple's: one
     *      root, exact depths both ways, in-link counts, and the islands' components, roles,
     *      entries and parents.
     *
     *  Its columns live in a context's round buffer, which its planner sized, so no pass
     *  allocates. @ref queue commits the current pass, the caller waits on the context, and
     *  @ref advance reads the counters over unified memory and picks the next pass. It only ever
     *  reads the graph.
     */
    struct health_plan_t {

        /** The watermark the check covers; IDs strictly below it are swept. */
        node_id_t visible = 0;

        /** Where the check's columns, and the round's beside them, sit in the round buffer. */
        round_layout_t layout {};

        /** Whether another pass follows. */
        bool running() const noexcept { return index_ && step_ != step_t::done_k; }

        /**
         *  @brief Commits the current pass on @p context 's stream, none once the check converged.
         *
         *  @par Statuses:
         *  - @c device_k The pass could not be encoded.
         */
        status_t queue(traversal_context_t& context) noexcept {
            if (!running()) return status_code_t::success_k;
            metal_graph_arguments_t arguments = index_->graph_arguments_();
            arguments.end_id = visible;
            arguments.depth = static_cast<u32_t>(depth_);
            arguments.islands = static_cast<u32_t>(islands_);
            arguments.pass = pass_;
            u32_t* const words = counters();
            vram_apple_graph& index = *index_;
            switch (step_) {
            case step_t::expand_k:
                words[metal_round_reached_k] = words[metal_round_marked_k] = 0;
                return index.launch_check_(vram_kernel_t::health_expand_k, arguments, layout, visible, context);
            case step_t::recheck_k:
                words[metal_round_marked_k] = 0;
                return index.launch_check_(vram_kernel_t::health_recheck_k, arguments, layout, visible, context);
            case step_t::order_islands_k:
                return index.launch_check_(vram_kernel_t::order_islands_k, arguments, layout, 0, context);
            case step_t::islands_k:
                return index.launch_check_(vram_kernel_t::health_islands_k, arguments, layout, islands_, context);
            case step_t::components_k:
                words[metal_round_relabeled_k] = 0;
                return index.launch_check_(vram_kernel_t::island_components_k, arguments, layout, islands_, context);
            case step_t::count_links_k:
                words[metal_round_unlinked_k] = words[metal_round_underlinked_k] = 0;
                return index.launch_check_(vram_kernel_t::health_count_links_k, arguments, layout, visible, context);
            case step_t::labels_k:
                words[metal_round_unreached_islands_k] = words[metal_round_stranded_islands_k] = 0;
                words[metal_round_components_k] = 0;
                return index.launch_check_(vram_kernel_t::label_islands_k, arguments, layout, islands_ + 1, context);
            default: return status_code_t::success_k;
            }
        }

        /** Sleeps until the committed pass finished on @p context; a no-op unless @ref running. */
        status_t wait(traversal_context_t& context) noexcept {
            if (!running()) return status_code_t::success_k;
            return context.wait();
        }

        /** The barrier between passes, once the last one finished, as dense Apple's: files the
         *  entries a depth reached, turns to the rechecks once a depth reaches nothing, orders and
         *  labels the islands once a recheck marks nothing, re-roots once to the largest island
         *  when it outgrows the drawn root's component, then counts the links and labels the
         *  islands' roles; answers whether another pass follows. */
        bool advance() noexcept {
            if (!running()) return false;
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
         *  direction took, the in-link counts and the islands' components; and @c internal_k while
         *  any pass remains to run. */
        expected<reachability_t> finish() const noexcept {
            if (!index_ || !visible) return {reachability_t {}, status_code_t::success_k};
            if (step_ != step_t::done_k) return {{}, status_code_t::internal_k};
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
            return {result, status_code_t::success_k};
        }

        /** Entries the sweeps reached plus entries they marked a way to the root for, of twice the
         *  checked ones: what a caller's progress counts. */
        std::size_t swept() const noexcept { return swept_; }

        /** Entries outside the root's component, the islands' nodes. */
        std::size_t islands() const noexcept { return islands_; }

        /** Turns the check onto @p index 's IDs below @p visible, its columns at @p layout in the
         *  round buffer whose host bytes start at @p columns, and restarts it from a fresh draw. */
        void rebind(vram_apple_graph& index, node_id_t visible, round_layout_t const& layout,
                    byte_t* columns) noexcept {
            index_ = &index;
            this->visible = visible;
            this->layout = layout;
            columns_ = columns;
            root_ = unknown_k;
            restart_sweep();
        }

        /** Clears every column and starts from the root an earlier sweep kept, or else from a
         *  seeded draw, marked both ways at depth zero. */
        void restart_sweep() noexcept {
            // Opening and closing checks share their root, so a fresh draw cannot read as progress.
            bool const kept = root_ < visible;
            node_id_t const root = kept      ? root_
                                   : visible ? static_cast<node_id_t>(seeded_bijection(0, visible, 0))
                                             : unknown_k;
            drawn_root_ = kept ? unknown_k : root;
            start_from_(root);
        }

        /** The counter words the passes count into, read over unified memory once a pass
         *  finished. */
        u32_t* counters() const noexcept { return reinterpret_cast<u32_t*>(columns_ + layout.counters); }

        /** The column at @p offset of the round buffer as @p value_type_ values. */
        template <typename value_type_>
        value_type_* column(std::size_t offset) const noexcept {
            return reinterpret_cast<value_type_*>(columns_ + offset);
        }

      private:
        friend struct vram_apple_graph;
        friend refine_plan_t;

        inline static constexpr node_id_t unknown_k = (std::numeric_limits<node_id_t>::max)();

        /** The passes, as the RAM audit names its own, plus those labeling the islands on the
         *  device. */
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
                drawn_root_ = unknown_k;
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
            swept_ = root == unknown_k ? 0 : 2;
            component_ = islands_ = unsettled_ = 0;
            depth_ = depth_from_root_ = depth_to_root_ = 0;
            unlinked_ = underlinked_ = unreached_islands_ = stranded_islands_ = components_ = 0;
            pass_ = 0;
            root_ = root;
            step_ = root == unknown_k ? step_t::done_k : step_t::expand_k;
            if (root == unknown_k) return;
            u32_t* const reached = column<u32_t>(layout.reached);
            u32_t* const reaches_root = column<u32_t>(layout.reaches_root);
            reached[root / 32] |= 1u << (root % 32), reaches_root[root / 32] |= 1u << (root % 32);
            u8_t* const depths_from_root = column<u8_t>(layout.depths_from_root);
            u8_t* const depths_to_root = column<u8_t>(layout.depths_to_root);
            depths_from_root[root] = depths_to_root[root] = 0;
            depth_entries_[0] = 1;
        }

        /** The checked graph; null on an unbound plan. */
        vram_apple_graph* index_ = nullptr;

        /** The round buffer's host bytes, which planning sized and no pass moves. */
        byte_t* columns_ = nullptr;

        /** The node the sweeps start from, a seeded draw or the first entry of a larger island, and
         *  the root this check drew, which may give way once. */
        node_id_t root_ = unknown_k, drawn_root_ = unknown_k;

        /** Entries each depth reached; the fullest is the depth the deep stratum draws past. */
        std::array<std::size_t, metal_unknown_depth_k> depth_entries_ {};

        /** Entries the sweeps reached and marked so far, of the root's component, outside it, and
         *  outside every settled island component. */
        std::size_t swept_ = 0, component_ = 0, islands_ = 0, unsettled_ = 0;

        /** The depth the next expansion reaches from, and the depths each direction took. */
        std::size_t depth_ = 0, depth_from_root_ = 0, depth_to_root_ = 0;

        /** The population counts the report holds, as @c reachability_t names them. */
        std::size_t unlinked_ = 0, underlinked_ = 0, unreached_islands_ = 0, stranded_islands_ = 0, components_ = 0;

        /** The pass within the components or labels the next commit takes. */
        u32_t pass_ = 0;
        step_t step_ = step_t::done_k;
    };

    /**
     *  @brief One refinement round in flight on this graph's device - dense Apple's round over
     *      values of any length: @ref queue a phase, @ref wait for it, @ref advance past it, and
     *      @ref finish once no phase follows.
     *
     *  The round opens with a health check, heals the islands it named with no distance, draws one
     *  cache-sized set of hubs by seeded bijection, gathers every entry into the neighborhood of
     *  its nearest hub, and again inside every neighborhood over its capacity among hubs drawn from
     *  its own entries, then restitches each neighborhood through the kits; the closing health
     *  check from the same root decides the verdict. The hubs are drawn and the entries ordered by
     *  neighborhood on the host over unified memory at the barriers. Planning made every
     *  allocation, so no phase allocates. A plan @ref plan_refine opened holds the claim word
     *  closed for its life, so batches answer @c contended_k meanwhile while searches run.
     */
    struct refine_plan_t {

        /** The check the round opens and closes with, whose columns the reach rule reads. */
        health_plan_t health;

        refine_plan_t() noexcept = default;
        refine_plan_t(refine_plan_t&& other) noexcept
            : health(other.health), index_(std::exchange(other.index_, nullptr)),
              claim_(std::exchange(other.claim_, claim_t::caller_reopens_k)), shape_(other.shape_),
              neighborhoods_(other.neighborhoods_), runs_(other.runs_), gathered_(other.gathered_),
              round_seed_(other.round_seed_), swept_after_(other.swept_after_), phase_(other.phase_),
              status_(other.status_), tally_(other.tally_) {}
        refine_plan_t& operator=(refine_plan_t&& other) noexcept {
            if (this == &other) return *this;
            if (claim_ == claim_t::plan_reopens_k) index_->reopen_();
            health = other.health, index_ = std::exchange(other.index_, nullptr);
            claim_ = std::exchange(other.claim_, claim_t::caller_reopens_k), shape_ = other.shape_;
            neighborhoods_ = other.neighborhoods_, runs_ = other.runs_, gathered_ = other.gathered_;
            round_seed_ = other.round_seed_, swept_after_ = other.swept_after_, phase_ = other.phase_;
            status_ = other.status_, tally_ = other.tally_;
            return *this;
        }
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** Reopens the claim word @ref plan_refine closed, whether the round finished or not. */
        ~refine_plan_t() noexcept {
            if (claim_ == claim_t::plan_reopens_k) index_->reopen_();
        }

        /**
         *  @brief Commits the current phase on @p context 's stream; a hubs draw commits nothing,
         *      as the barrier closing it draws them on the host. Never waits, never allocates.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers.
         *  - @c config_mismatch_k The @p context rides another device.
         *  - @c device_k A phase could not be encoded, or the plan keeps it.
         */
        status_t queue(traversal_context_t& context) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            if (failed(status_)) return status_;
            if (context.device() != index_->device_.ordinal) return status_ = status_code_t::config_mismatch_k;
            switch (phase_) {
            case phase_t::health_k: status_ = health.queue(context); break;
            case phase_t::heal_k: status_ = index_->queue_heal_(health, context); break;
            case phase_t::gather_k: status_ = index_->queue_gather_(*this, context); break;
            case phase_t::restitch_k: status_ = index_->queue_restitch_(*this, context); break;
            default: break;
            }
            return status_;
        }

        /** Sleeps until the committed phase finished on @p context. */
        status_t wait(traversal_context_t& context) noexcept {
            if (index_ && succeeded(status_) && phase_ != phase_t::done_k) status_ = context.wait();
            return status_;
        }

        /** The barrier after a phase finished: takes a converged check in, draws the hubs, orders
         *  the entries by neighborhood or gathers the wide ones again, or reads the restitch's
         *  counts; answers whether a phase follows. */
        bool advance() noexcept {
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
            case phase_t::hubs_k: draw_hubs_(), phase_ = phase_t::gather_k; break;
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
            return succeeded(status_) && phase_ != phase_t::done_k;
        }

        /** Skips what is left of the round's phase in hand: an opening health check, or a heal not
         *  yet run, ends the round with nothing written; a hubs draw, a gather or a restitch sends
         *  the round to the closing health check; the closing check runs on, and its verdict reads
         *  stopped, whatever it measures. */
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

        /** Whether the plan runs a round, whose every barrier a caller's progress hears. */
        bool rounding() const noexcept {
            if (phase_ == phase_t::health_k)
                return swept_after_ == phase_t::none_k || swept_after_ == phase_t::restitch_k;
            return phase_ >= phase_t::heal_k && phase_ <= phase_t::restitch_k;
        }

        /** The phase the plan is in. */
        phase_t phase() const noexcept { return phase_; }

        /** The position of the round's phase in hand among its six; zero outside a round's
         *  phases. */
        std::size_t ordinal() const noexcept { return rounding() ? round_ordinal(phase_, swept_after_) : 0; }

        /** Units of the round's phase in hand done so far, at a barrier before it advances. */
        std::size_t completed() const noexcept { return phase_ == phase_t::health_k ? health.swept() : total(); }

        /** Units of the round's phase in hand in all: islands healed, hubs drawn, entries one
         *  gather files, or entries re-stitched. */
        std::size_t total() const noexcept {
            switch (phase_) {
            case phase_t::health_k: return 2 * std::size_t {health.visible};
            case phase_t::heal_k: return health.islands();
            case phase_t::hubs_k: return round_first_hubs(health.visible, shape_.hubs_per_draw, shape_.capacity);
            case phase_t::gather_k: return gathered_;
            case phase_t::restitch_k: return health.visible;
            default: return 0;
            }
        }

        /** Opens the next round once one ended, from the root the closing check kept; a no-op
         *  mid-round. */
        void next_round() noexcept {
            if (!index_ || phase_ != phase_t::done_k || failed(status_)) return;
            if (swept_after_ != phase_t::none_k && swept_after_ != phase_t::restitch_k) return;
            tally_ = {};
            verify_(phase_t::none_k);
        }

        /**
         *  @brief The round, complete once @ref advance answered false: its report, the walks'
         *      telemetry read off @p context 's counters; a failure a phase kept answers instead.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound.
         *  - @c config_mismatch_k, @c device_k or @c internal_k A phase kept it.
         *  - @c cancelled_k A phase still follows, or the round was stopped.
         */
        expected<refine_report_t> finish(traversal_context_t const& context) const noexcept {
            if (!index_) return {{}, status_code_t::uninitialized_k};
            refine_report_t result = tally_;
            vram_telemetry_block_t const spent = context.spent();
            result.traversal_stats = {.computed_distances = spent.scored,
                                      .visited_members = spent.hops,
                                      .restarts = spent.restarts,
                                      .saturated_walks = spent.visited_saturations};
            if (failed(status_)) return {result, status_};
            bool const done = phase_ == phase_t::done_k && result.verdict != refine_verdict_t::stopped_k;
            return {result, done ? status_code_t::success_k : status_code_t::cancelled_k};
        }

      private:
        friend struct vram_apple_graph;
        friend reshard_plan_t;

        /** Opens the health check following @p phase from the root the last one kept. */
        void verify_(phase_t phase) noexcept {
            health.restart_sweep();
            swept_after_ = phase;
            phase_ = phase_t::health_k;
        }

        /** Turns the plan onto a fresh round over @p index 's published entries, which the round
         *  buffer at @p columns laid out as @p layout must span; with no entry, the plan ends at
         *  once, as nothing needs a round. */
        void bind_(vram_apple_graph& index, round_layout_t const& layout, byte_t* columns) noexcept {
            index_ = &index;
            tally_ = {};
            neighborhoods_ = runs_ = gathered_ = 0;
            health.rebind(index, static_cast<node_id_t>(index.size()), layout, columns);
            swept_after_ = phase_t::none_k;
            phase_ = health.visible ? phase_t::health_k : phase_t::done_k;
        }

        /** Takes in a converged check: opens the round after the opening one, concludes it after
         *  the closing one; answers whether a phase follows. */
        bool converge_() noexcept {
            auto const [reachability, status] = health.finish();
            if (failed(status)) {
                status_ = status;
                phase_ = phase_t::done_k;
                return false;
            }
            tally_.reachability = reachability;
            if (swept_after_ == phase_t::none_k) open_round_(reachability);
            else phase_ = phase_t::done_k, tally_.verdict = concluded_verdict(tally_);
            return phase_ != phase_t::done_k;
        }

        /** Takes in the opening check: seeds the round's draws from what it measured, starts the
         *  changed marks over, and opens the heal. */
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

        /** Draws the first gather's hubs by seeded bijection over the visible entries, ascending,
         *  into its one run over every ID. */
        void draw_hubs_() noexcept {
            std::size_t const visible = health.visible;
            std::size_t const hubs = round_first_hubs(visible, shape_.hubs_per_draw, shape_.capacity);
            std::span<u32_t> const hub_nodes {health.template column<u32_t>(health.layout.hub_nodes), hubs};
            for (std::size_t rank = 0; rank != hubs; ++rank)
                hub_nodes[rank] = static_cast<u32_t>(seeded_bijection(rank, visible, round_seed_));
            partial_sort(hub_nodes, hubs);
            vram_neighborhood_run_t* const runs = health.template column<vram_neighborhood_run_t>(health.layout.runs);
            u32_t* const run_items = health.template column<u32_t>(health.layout.run_items);
            runs[0] = {0, static_cast<u32_t>(visible), 0, static_cast<u32_t>(hubs)};
            run_items[0] = 0, run_items[1] = static_cast<u32_t>(visible);
            runs_ = 1, neighborhoods_ = 0, gathered_ = visible, hubs_ = hubs;
        }

        /** Sorts the entries by the neighborhood the gather gave each, after the first gather by
         *  @ref order_by_neighborhood and after a later one by @ref split_wide_neighborhoods; then
         *  writes, for every neighborhood still over capacity, its run and its hubs, drawn among
         *  its own entries as RAM draws them, and answers whether a gather runs once more over
         *  those neighborhoods. */
        bool order_neighborhoods_() noexcept {
            round_layout_t const& layout = health.layout;
            std::size_t const visible = health.visible;
            std::span<u16_t const> const labels {health.template column<u16_t>(layout.entry_neighborhoods), visible};
            std::span<u32_t> const entries {health.template column<u32_t>(layout.neighborhood_entries), visible};
            std::span<u32_t> const offsets {health.template column<u32_t>(layout.neighborhood_offsets),
                                            split_room(visible, shape_.hubs_per_draw)};
            neighborhood_order_t const order = neighborhoods_
                                                   ? split_wide_neighborhoods(labels, entries, offsets, neighborhoods_,
                                                                              shape_.capacity, shape_.hubs_per_draw)
                                                   : order_by_neighborhood(labels, hubs_, entries, offsets);
            neighborhoods_ = order.neighborhoods;
            tally_.neighborhoods = order.neighborhoods, tally_.widest_neighborhood = order.widest;
            if (order.widest <= shape_.capacity) return false;
            u32_t* const hub_nodes = health.template column<u32_t>(layout.hub_nodes);
            vram_neighborhood_run_t* const runs = health.template column<vram_neighborhood_run_t>(layout.runs);
            u32_t* const run_items = health.template column<u32_t>(layout.run_items);
            std::size_t hubs = 0;
            runs_ = gathered_ = 0;
            run_items[0] = 0;
            for (std::size_t neighborhood = 0; neighborhood != order.neighborhoods; ++neighborhood) {
                u32_t const first = offsets[neighborhood], width = offsets[neighborhood + 1] - first;
                if (width <= shape_.capacity) continue;
                std::size_t const drawn = round_hubs(width, shape_.hubs_per_draw, shape_.capacity);
                u64_t const seed = mix_bits(round_seed_ ^ first);
                for (std::size_t rank = 0; rank != drawn; ++rank)
                    hub_nodes[hubs + rank] = entries[first + seeded_bijection(rank, width, seed)];
                runs[runs_] = {first, width, static_cast<u32_t>(hubs), static_cast<u32_t>(drawn)};
                run_items[runs_ + 1] = run_items[runs_] + width;
                ++runs_, hubs += drawn, gathered_ += width;
            }
            hubs_ = hubs;
            return true;
        }

        vram_apple_graph* index_ = nullptr;

        /** Who reopens @c index_ 's claim word: the plan at its death, or its caller. */
        claim_t claim_ = claim_t::caller_reopens_k;

        /** The round's sizes, as RAM names them. */
        round_shape_t shape_ {};

        /** Neighborhoods the order holds, zero before the first gather's; the runs, the hubs and
         *  the entries the next gather takes. */
        std::size_t neighborhoods_ = 0, runs_ = 0, hubs_ = 0, gathered_ = 0;

        /** The seed of the round's draws. */
        u64_t round_seed_ = 0;

        /** The phase the current health check follows, which decides the one after it. */
        phase_t swept_after_ = phase_t::none_k;
        phase_t phase_ = phase_t::done_k;
        status_t status_ = status_code_t::success_k;
        refine_report_t tally_;
    };

    /**
     *  @brief One reshard in flight on this graph's device, as @ref plan_reshard opens it: the
     *      moves first, then one round on every target in turn, through the same @ref queue,
     *      @ref wait, @ref advance and @ref finish as a refine.
     *
     *  A gather copies every other source's nodes past this graph's watermark on the host at
     *  planning, so its plan opens on the one target's round. A cut grows its partition on the
     *  device, then the barrier after its last batch places the nodes no batch reached, scatters
     *  every node into its target on the host over unified memory, the links inside its part
     *  rebased and those across dropped, files the keys, publishes every target at once and
     *  empties the source; every target then runs one round, and the rounds' reports fold.
     *
     *  The plan copies its target list and holds every claim word it closed for its life, its
     *  death reopening every one still closed; nothing may search the cut's source meanwhile.
     */
    struct reshard_plan_t {

        reshard_plan_t() noexcept = default;
        reshard_plan_t(reshard_plan_t&& other) noexcept
            : rounds_(std::move(other.rounds_)), split_(std::move(other.split_)), targets_(std::move(other.targets_)),
              closed_targets_(std::exchange(other.closed_targets_, 0)), round_layout_(other.round_layout_),
              round_columns_(std::exchange(other.round_columns_, nullptr)), moves_(other.moves_),
              refined_targets_(other.refined_targets_), folded_targets_(other.folded_targets_),
              targets_tally_(other.targets_tally_) {
            other.split_.source = nullptr;
        }
        reshard_plan_t& operator=(reshard_plan_t&&) = delete;
        reshard_plan_t(reshard_plan_t const&) = delete;
        reshard_plan_t& operator=(reshard_plan_t const&) = delete;

        /** Reopens every claim word the plan closed, whether it finished or not. */
        ~reshard_plan_t() noexcept {
            for (std::size_t target = 0; target != closed_targets_; ++target) targets_[target]->reopen_();
            if (split_.source) split_.source->reopen_();
        }

        /**
         *  @brief Commits the current phase on @p context 's stream: a growth batch of the
         *      partition, then the bound target's round.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers.
         *  - Those of @ref refine_plan_t::queue.
         */
        status_t queue(traversal_context_t& context) noexcept {
            if (!round_columns_) return status_code_t::uninitialized_k;
            // Targets left empty bind no round, which ends with nothing to queue.
            if (!moving()) return rounds_.index_ ? rounds_.queue(context) : rounds_.status_;
            if (failed(rounds_.status_)) return rounds_.status_;
            return rounds_.status_ = split_.source->queue_rounds_(split_, targets_.size(), context);
        }

        /** Sleeps until the committed phase finished on @p context. */
        status_t wait(traversal_context_t& context) noexcept {
            if (!moving()) return rounds_.index_ ? rounds_.wait(context) : rounds_.status_;
            if (succeeded(rounds_.status_)) rounds_.status_ = context.wait();
            return rounds_.status_;
        }

        /** The barrier after a phase finished: grows the parts on while a batch still placed
         *  nodes, then places the rest, scatters, publishes the parts and binds the first target,
         *  or binds the next target once a round ended; answers whether a phase follows. */
        bool advance() noexcept {
            if (!round_columns_ || failed(rounds_.status_)) return false;
            if (moving()) {
                vram_apple_graph& source = *split_.source;
                if (source.placing_(split_, targets_.size())) return true;
                source.place_islands_(split_, targets_.size());
                assign_slots_();
                source.scatter_(split_, to_span(targets_));
                publish_parts_();
                moves_ = phase_t::done_k;
                return bind_target_();
            }
            if (rounds_.advance()) return true;
            // A plan whose last target's round ended settles nothing more.
            if (failed(rounds_.status_) || refined_targets_ == targets_.size()) return false;
            return settle_target_();
        }

        /** Stops the bound target's round as @ref refine_plan_t::stop does, past the moves; every
         *  later target heals and verifies alone. */
        void stop() noexcept {
            if (!moving()) rounds_.stop();
        }

        /** Whether the plan still moves a cut's nodes. */
        bool moving() const noexcept { return moves_ != phase_t::done_k; }

        /** Whether the bound target is inside a round. */
        bool rounding() const noexcept { return !moving() && rounds_.rounding(); }

        /** The phase the plan is in: a move's, then the bound target's round's. */
        phase_t phase() const noexcept { return moving() ? moves_ : rounds_.phase(); }

        /** Where the round's phase sits among its six, and its units done and in all; none in the
         *  moves. */
        std::size_t ordinal() const noexcept { return moving() ? 0 : rounds_.ordinal(); }
        std::size_t completed() const noexcept { return moving() ? 0 : rounds_.completed(); }
        std::size_t total() const noexcept { return moving() ? 0 : rounds_.total(); }

        /** Runs another round on every target once every target's ended, from the first; a no-op
         *  in the moves and mid-round. */
        void next_round() noexcept {
            if (moving() || rounds_.phase_ != phase_t::done_k || failed(rounds_.status_)) return;
            refined_targets_ = folded_targets_ = 0;
            targets_tally_ = {};
            bind_target_();
        }

        /**
         *  @brief The rounds so far, folded over the targets: complete once @ref advance answered
         *      false, else cancelled.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound.
         *  - @c cancelled_k The moves or a round stopped before @ref advance answered false.
         *  - Those of @ref refine_plan_t::finish.
         */
        expected<refine_report_t> finish(traversal_context_t const& context) const noexcept {
            if (!round_columns_) return {{}, status_code_t::uninitialized_k};
            if (moving()) return {{}, failed(rounds_.status_) ? rounds_.status_ : status_code_t::cancelled_k};
            auto [report, finish_status] = rounds_.finish(context);
            if (finish_status == status_code_t::uninitialized_k) return {targets_tally_, status_code_t::success_k};
            // Past the last target the rounds hold the fold already; short of it, the bound round
            // adds to the targets settled before it.
            if (refined_targets_ != targets_.size() && folded_targets_) {
                refine_report_t folded = targets_tally_;
                folded += report;
                report = folded;
            }
            return {report, finish_status};
        }

      private:
        friend struct vram_apple_graph;

        /** Binds the round to the next target holding entries; a target after a stop heals and
         *  verifies alone. Past the last, reports the folded rounds and answers false. */
        bool bind_target_() noexcept {
            while (refined_targets_ != targets_.size() && !targets_[refined_targets_]->size()) ++refined_targets_;
            if (refined_targets_ == targets_.size()) {
                rounds_.tally_ = targets_tally_;
                rounds_.phase_ = phase_t::done_k;
                return false;
            }
            vram_apple_graph& target = *targets_[refined_targets_];
            round_layout_t const layout = target.round_layout_(target.size(), rounds_.shape_);
            rounds_.bind_(target, layout, round_columns_);
            if (folded_targets_ && targets_tally_.verdict == refine_verdict_t::stopped_k)
                rounds_.tally_.verdict = refine_verdict_t::stopped_k;
            return true;
        }

        /** Folds the bound target's ended round into the targets' report, the first taken whole,
         *  then binds the next one, answering whether a phase follows. */
        bool settle_target_() noexcept {
            if (folded_targets_++) targets_tally_ += rounds_.tally_;
            else targets_tally_ = rounds_.tally_;
            ++refined_targets_;
            return bind_target_();
        }

        /** Turns every source node's part into its slot in place, in ID order within each part. */
        void assign_slots_() noexcept {
            u32_t* const slots = reinterpret_cast<u32_t*>(split_.device_bytes.host);
            for (split_part_t& part : split_.parts) part = {};
            for (std::size_t node = 0; node != split_.entries; ++node) {
                std::size_t const part = slots[node];
                ++split_.parts[part].entries;
                slots[node] = static_cast<u32_t>(part * split_.part_capacity + split_.parts[part].assigned_nodes++);
            }
        }

        /** Moves every source key to its slot and publishes every target's nodes at once, its keys
         *  filed under the closed claim words, its tape's extent and longest value its part's and
         *  its spans the source's, then empties the source and reopens its claim word at zero. */
        void publish_parts_() noexcept {
            vram_apple_graph& source = *std::exchange(split_.source, nullptr);
            u32_t const* const slots = reinterpret_cast<u32_t const*>(split_.device_bytes.host);
            for (std::size_t node = 0; node != split_.entries; ++node)
                targets_[slots[node] / split_.part_capacity]->keyed_.assign(
                    static_cast<node_id_t>(slots[node] % split_.part_capacity),
                    source.keyed_[static_cast<node_id_t>(node)]);
            for (std::size_t index = 0; index != targets_.size(); ++index) {
                split_part_t const& part = split_.parts[index];
                vram_apple_graph& target = *targets_[index];
                target.tape_bytes_ = part.bytes, target.longest_value_ = part.longest;
                target.span_bytes_ = source.span_bytes_;
                target.publish_keys_({0, static_cast<node_id_t>(part.entries)});
                target.publish_(part.entries);
            }
            source.forget_();
            source.reopen_();
        }

        /** The rounds over the bound target. */
        refine_plan_t rounds_;

        /** A cut's moves; empty for a gather. */
        split_t split_;

        /** The targets, as the caller named them: this graph alone for a gather, the cut's parts'
         *  graphs otherwise; and how many of them, from the first, the plan closed. */
        growing_buffer<vram_apple_graph*> targets_ {global_allocator};
        std::size_t closed_targets_ = 0;

        /** The round buffer every target's round lays its columns over, sized for the widest; null
         *  until planning bound the plan, so a refused one runs no phase. */
        round_layout_t round_layout_ {};
        byte_t* round_columns_ = nullptr;

        /** The phase of the moves still pending; done once the parts published. */
        phase_t moves_ = phase_t::done_k;

        /** Targets whose rounds ended so far, the bound one the next, and those folded. */
        std::size_t refined_targets_ = 0, folded_targets_ = 0;

        /** The reports of the targets' rounds ended so far, folded. */
        refine_report_t targets_tally_ {};
    };

    /**
     *  @brief Opens one refinement round over the published nodes on @p context 's stream, as the
     *      RAM @c plan_refine does.
     *
     *  Closes the claim word, grows the context's round buffer to the check's columns, the round's
     *  and the restitch's scratch, and sizes the hubs and the neighborhood capacity from
     *  @c cache_bytes and the mean stored value, never past what one GPU core's threadgroup keeps.
     *  No phase allocates.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet.
     *  - @c config_mismatch_k The @p context rides another device or cannot launch.
     *  - @c contended_k A batch, a load or another plan in flight.
     *  - @c out_of_memory_k The round buffer cannot grow.
     */
    expected<refine_plan_t> plan_refine(refine_config_t config, traversal_context_t& context) noexcept {
        refine_plan_t plan;
        if (!connectivity()) return {std::move(plan), status_code_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {std::move(plan), status_code_t::config_mismatch_k};
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        plan.index_ = this, plan.claim_ = claim_t::plan_reopens_k;
        std::size_t const entries = size();
        plan.shape_ = round_shape_(entries, mean_value_bytes_(), config.cache_bytes);
        round_layout_t const layout = round_layout_(entries, plan.shape_);
        // A refused plan keeps the claim word to reopen it, and its status to refuse every phase.
        if (status_t const carved = carve_round_(context, layout.bytes); failed(carved)) {
            plan.status_ = carved;
            return {std::move(plan), carved};
        }
        plan.bind_(*this, layout, context.round_.host);
        return {std::move(plan), status_code_t::success_k};
    }

    /**
     *  @brief One @ref plan_refine driven to its end on the calling thread, waiting once per phase.
     *
     *  @par Statuses:
     *  - Those of @ref plan_refine and @ref refine_plan_t::finish.
     */
    expected<refine_report_t> refine(refine_config_t config, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_refine(config, context);
        if (failed(status)) return {{}, status};
        do
            if (failed(plan.queue(context)) || failed(plan.wait(context))) break;
        while (plan.advance());
        return plan.finish(context);
    }

    /**
     *  @brief Opens a reshard of @p sources into @p targets, as @ref reshard_plan_t describes: a
     *      gather of every other source into this graph, its one target, or a cut of this graph,
     *      its one source, into at least two empty targets.
     *
     *  Every check happens before anything moves: a gather vets every source's device,
     *  connectivity, metric and kept status, and the ceiling, before it closes one; a cut vets and
     *  closes every target, and reserves each for a full part, before its partition commits. A
     *  gather's moves end before the plan returns; a cut's partition grows on the device with one
     *  root per part through @ref seeded_bijection under the config's seed, each part capped at
     *  the nodes over the targets rounded up, so equal seeds over equal graphs split alike, as on
     *  NVIDIA and AMD. Every listed graph must outlive both the call and the plan's finish.
     *
     *  @param[in] sources The graphs giving their nodes up: this graph alone for a cut, any others
     *      for a gather, which keeps this graph's own nodes when it is among them.
     *  @param[in] targets This graph alone for a gather, or at least two empty graphs for a cut.
     *  @param[in] graph The member of each listed owner holding its graph.
     *  @param[in] config The rounds' knobs and the seed picking a cut's part roots.
     *  @param[in,out] context The driving context, whose round buffer every target's round reuses.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k An empty list, a null or a repeated graph within one.
     *  - @c unsupported_k Several sources and several targets, or a nonzero
     *    @c reshard_config_t::shard_capacity, as a graph holds no shards.
     *  - @c config_mismatch_k Neither a gather into this graph nor a cut of it, a cut target that
     *    is this graph or not empty, or a graph of another device, connectivity or metric.
     *  - @c contended_k A batch, a load or a plan in flight on a listed graph.
     *  - @c overflow_k Nodes past the reserved ceiling, or part slots past the 32-bit IDs.
     *  - @c out_of_memory_k The round buffer or the partition's buffer cannot be placed.
     */
    template <typename owner_type_>
    expected<reshard_plan_t> plan_reshard(std::span<owner_type_* const> sources, std::span<owner_type_* const> targets,
                                          vram_apple_graph owner_type_::* graph, reshard_config_t config,
                                          traversal_context_t& context) noexcept {
        reshard_plan_t plan;
        if (sources.empty() || targets.empty()) return {std::move(plan), status_code_t::invalid_argument_k};
        for (std::span<owner_type_* const> const listed : {sources, targets})
            if (status_t const refused = listed_refusal(listed); failed(refused)) return {std::move(plan), refused};
        if (sources.size() > 1 && targets.size() > 1) return {std::move(plan), status_code_t::unsupported_k};
        if (config.shard_capacity) return {std::move(plan), status_code_t::unsupported_k};
        bool const gathering = targets.size() == 1 && &(targets[0]->*graph) == this;
        bool const cutting = sources.size() == 1 && &(sources[0]->*graph) == this && targets.size() > 1;
        if (!gathering && !cutting) return {std::move(plan), status_code_t::config_mismatch_k};
        if (!connectivity()) return {std::move(plan), status_code_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {std::move(plan), status_code_t::config_mismatch_k};
        status_t const opened = gathering ? open_gather_(plan, sources, graph, config, context)
                                          : open_cut_(plan, targets, graph, config, context);
        return {std::move(plan), opened};
    }

    /**
     *  @brief One @ref plan_reshard driven to its end on the calling thread, one wait a phase.
     *
     *  @par Statuses:
     *  - Those of @ref plan_reshard and @ref reshard_plan_t::finish.
     */
    template <typename owner_type_>
    expected<refine_report_t> reshard(std::span<owner_type_* const> sources, std::span<owner_type_* const> targets,
                                      vram_apple_graph owner_type_::* graph, reshard_config_t config,
                                      traversal_context_t& context) noexcept {
        auto [plan, status] = plan_reshard(sources, targets, graph, config, context);
        if (failed(status)) return {{}, status};
        do
            if (failed(plan.queue(context)) || failed(plan.wait(context))) break;
        while (plan.advance());
        return plan.finish(context);
    }

    /**
     *  @brief Checks the health of the published nodes from one drawn root, as a round's opening
     *      check does: the root's strong component, the depths each way, and the outer islands.
     *
     *  Read-only on the graph, so concurrent insertions may leave fresh nodes uncounted. Waits on
     *  the context once per pass.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet.
     *  - @c config_mismatch_k The @p context rides another device or cannot launch.
     *  - @c out_of_memory_k The round buffer cannot grow.
     */
    expected<reachability_t> health(traversal_context_t& context) noexcept {
        std::size_t const entries = size();
        if (!entries) return {{}, status_code_t::success_k};
        if (!connectivity()) return {{}, status_code_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_code_t::config_mismatch_k};
        round_layout_t const layout = round_layout_(entries, {});
        if (status_t const carved = carve_round_(context, layout.bytes); failed(carved)) return {{}, carved};
        health_plan_t plan;
        plan.rebind(*this, static_cast<node_id_t>(entries), layout, context.round_.host);
        do
            if (status_t const queued = plan.queue(context); failed(queued) || failed(plan.wait(context)))
                return {{}, failed(queued) ? queued : status_code_t::device_k};
        while (plan.advance());
        return plan.finish();
    }

#pragma endregion Refinement

#pragma region Checkpoints

    /**
     *  @brief One save in flight: the watermark it took, and the nodes it hands the caller's
     *      columns, read in place from the shared storage.
     *
     *  Neighbors travel as ragged lists, every sentinel and self-loop dropped, and values through
     *  the face's codec, which may unpack them. Planning sized every node, so a call writes the
     *  slices the prefix sums name, for any nodes in any order. The claim word stays closed from
     *  planning to @c finish or the plan's death, so batches answer @c contended_k meanwhile, while
     *  searches run throughout. The keys cross at planning.
     */
    struct [[nodiscard]] save_plan_t {

        /** The nodes under the watermark, the neighbor IDs they keep, and the bytes their values
         *  save into. */
        columns_extents_t extents {};

        save_plan_t() noexcept = default;
        save_plan_t(save_plan_t&& other) noexcept
            : extents(other.extents), neighbor_offsets_(other.neighbor_offsets_), neighbor_ids_(other.neighbor_ids_),
              value_offsets_(other.value_offsets_), values_(other.values_), covered_(other.covered_),
              index_(std::exchange(other.index_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;

        /** A save owns nothing, so a dying plan only reopens the claim word. */
        ~save_plan_t() noexcept {
            if (index_) index_->reopen_();
        }

        /**
         *  @brief Copies the neighbor IDs and the values of @p nodes into the column slices that
         *      the plan's prefix sums name.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The @p nodes pass the plan, or a node's kept neighbors or its
         *    saved value disagree with the offsets the plan measured.
         */
        status_t save(iota_view<node_id_t> nodes) noexcept {
            if (!index_ || neighbor_offsets_.empty()) return status_code_t::uninitialized_k;
            if (nodes.bound < nodes.first || nodes.bound > extents.nodes) return status_code_t::config_mismatch_k;
            node_id_t const count = static_cast<node_id_t>(extents.nodes);
            std::size_t const connectivity = index_->connectivity();
            for (node_id_t node = nodes.first; node != nodes.bound; ++node) {
                u64_t cursor = neighbor_offsets_[node];
                u64_t const bound = neighbor_offsets_[node + 1];
                for (node_id_t const neighbor : index_->row_(node, connectivity)) {
                    if (!kept_link_(neighbor, node, count)) continue;
                    if (cursor == bound) return status_code_t::config_mismatch_k;
                    neighbor_ids_[cursor++] = neighbor;
                }
                if (cursor != bound) return status_code_t::config_mismatch_k;
                if (status_t const saved = index_->kits_.save_value_(
                        index_->value_of_(node),
                        values_.subspan(value_offsets_[node], value_offsets_[node + 1] - value_offsets_[node]));
                    failed(saved))
                    return saved;
            }
            covered_ += nodes.size();
            return status_code_t::success_k;
        }

        /**
         *  @brief Pronounces the snapshot complete once every node was copied: a success spends the
         *      plan and reopens the claim word; a refusal leaves it whole.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k Some nodes were never saved.
         */
        status_t finish() noexcept {
            if (!index_ || neighbor_offsets_.empty()) return status_code_t::uninitialized_k;
            if (covered_ != extents.nodes) return status_code_t::config_mismatch_k;
            std::exchange(index_, nullptr)->reopen_();
            return status_code_t::success_k;
        }

      private:
        friend struct vram_apple_graph;

        /** Sizes every node under the watermark, writing both prefix sums when they are given,
         *  @b [nodes+1] each. */
        status_t size_(std::span<u64_t> neighbor_sums, std::span<u64_t> value_sums) noexcept {
            node_id_t const nodes = static_cast<node_id_t>(extents.nodes);
            std::size_t const connectivity = index_->connectivity();
            extents.neighbors = 0, extents.value_bytes = 0;
            if (!neighbor_sums.empty()) neighbor_sums[0] = 0, value_sums[0] = 0;
            for (node_id_t node = 0; node != nodes; ++node) {
                auto const [saved_bytes, saved_status] = index_->kits_.saved_bytes_(index_->value_of_(node));
                if (failed(saved_status)) return saved_status;
                for (node_id_t const neighbor : index_->row_(node, connectivity))
                    if (kept_link_(neighbor, node, nodes)) ++extents.neighbors;
                extents.value_bytes += saved_bytes;
                if (!neighbor_sums.empty())
                    neighbor_sums[node + 1] = extents.neighbors, value_sums[node + 1] = extents.value_bytes;
            }
            return status_code_t::success_k;
        }

        /** The columns the calls write but the keys, empty in a plan that only sizes. */
        std::span<u64_t> neighbor_offsets_;
        std::span<node_id_t> neighbor_ids_;
        std::span<u64_t> value_offsets_;
        bytes_span_t values_;

        /** Nodes whose neighbors and values were copied. */
        std::size_t covered_ = 0;

        /** The graph being snapshotted, whose claim word the plan holds closed; null in a refused
         *  plan, and once @c finish succeeded. */
        vram_apple_graph* index_ = nullptr;
    };

    /**
     *  @brief Node & element counts a save needs right now, through one sizing pass over every
     *      node under the watermark with the claim word closed.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p context rides another device or cannot walk.
     *  - @c contended_k A batch, a load or a plan in flight.
     */
    expected<columns_extents_t> columns_extents(traversal_context_t& context) noexcept {
        auto [plan, status] = open_save_(context);
        if (succeeded(status)) status = plan.size_({}, {});
        return {plan.extents, status};
    }

    /**
     *  @brief Opens a save into @p columns of every node under the watermark: closes the claim
     *      word, sizes every node and writes both prefix sums.
     *
     *  The key column's length names the node count, and planning fills it from the host keys. A
     *  column shorter than its count answers @c overflow_k, and a longer key or sums column
     *  @c config_mismatch_k, so no call ever writes past them.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p context rides another device or cannot walk, or the key or a
     *    sums column is longer than its count.
     *  - @c overflow_k A column is shorter than its count.
     *  - @c contended_k A batch, a load or a plan in flight.
     */
    expected<save_plan_t> plan_save(columns<key_t, node_id_t, byte_t, u64_t> columns,
                                    traversal_context_t& context) noexcept {
        auto [plan, status] = open_save_(context);
        if (failed(status)) return {std::move(plan), status};
        std::size_t const nodes = plan.extents.nodes;
        for (status_t const sized :
             {sized_exactly_(columns.keys.size(), nodes), sized_exactly_(columns.neighbor_offsets.size(), nodes + 1),
              sized_exactly_(columns.value_offsets.size(), nodes + 1)})
            if (failed(sized)) return {std::move(plan), sized};
        if (status_t const sized = plan.size_(columns.neighbor_offsets, columns.value_offsets); failed(sized))
            return {std::move(plan), sized};
        if (columns.neighbor_ids.size() < plan.extents.neighbors || columns.values.size() < plan.extents.value_bytes)
            return {std::move(plan), status_code_t::overflow_k};
        plan.neighbor_offsets_ = columns.neighbor_offsets, plan.neighbor_ids_ = columns.neighbor_ids;
        plan.value_offsets_ = columns.value_offsets, plan.values_ = columns.values;
        std::size_t saved = 0;
        keyed_.column().for_each_span({0, nodes}, [&](std::span<key_t const> run) noexcept {
            for (key_t const& key : run) columns.keys[saved++] = key;
        });
        return {std::move(plan), status_code_t::success_k};
    }

    /**
     *  @brief Serializes every node under the watermark into @p columns on the calling thread -
     *      the one-shot over @ref plan_save.
     *
     *  @par Statuses:
     *  - Those of @ref plan_save and @ref save_plan_t::save.
     */
    status_t save(columns<key_t, node_id_t, byte_t, u64_t> columns, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_save(columns, context);
        if (failed(status)) return status;
        if (status_t const saved = plan.save({0, static_cast<node_id_t>(plan.extents.nodes)}); failed(saved))
            return saved;
        return plan.finish();
    }

    /**
     *  @brief One load in flight: the nodes it appends past every claimed one, the storage grown to
     *      hold them, and the context the kits' load kernels ride.
     *
     *  Planning sizes every stored value from its saved length alone. A call widens a run of nodes'
     *  saved neighbor rows into the matrix on the host, writes their planned tape starts, and hands
     *  the kits the run to store its values, on the host or through their load kernels. Keys go
     *  into the host key column. @c finish refuses a malformed neighbor row, files the keys and
     *  turns the watermark over the nodes. Holds the claim for its whole life; a plan that dies
     *  before @c finish gives its claim back and leaves the storage grown, its nodes unfilled.
     */
    struct [[nodiscard]] load_plan_t {

        /** The first appended ID, where node zero of the columns goes. */
        node_id_t base = 0;

        /** Nodes the plan appends. */
        std::size_t entries = 0;

        load_plan_t() noexcept = default;

        /** A plan whose kits' load kernels ride a context of its own over @p stream. */
        explicit load_plan_t(metal_stream_t&& stream) noexcept : context_(std::move(stream)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : base(other.base), entries(std::exchange(other.entries, 0)), saved_(other.saved_),
              stored_offsets_(std::move(other.stored_offsets_)), context_(std::move(other.context_)),
              longest_(other.longest_), span_bytes_(other.span_bytes_), covered_(other.covered_),
              widened_(other.widened_), index_(std::exchange(other.index_, nullptr)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** An unfinished plan lets the next grow fill its nodes again, and gives its claim back. */
        ~load_plan_t() noexcept {
            if (!index_) return;
            index_->filled_nodes_ = min_of(index_->filled_nodes_, static_cast<std::size_t>(base));
            index_->unclaim_(base, entries);
        }

        /**
         *  @brief Widens and stores the saved nodes @p range, counted from the columns' first node,
         *      one run at a time, waiting on the context only after a run the kits launched on.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The @p range passes the plan.
         *  - @c device_k A run the kits launched on failed.
         */
        status_t load(iota_view<node_id_t> range) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            if (range.bound < range.first || range.bound > entries) return status_code_t::config_mismatch_k;
            vram_apple_graph& index = *index_;
            std::size_t const key_bytes = dtype_bytes(saved_.key_dtype), connectivity = index.connectivity();
            std::size_t const run_nodes = power_of_two<std::size_t>(crossing_nodes_log2_k);
            u64_t* const starts = index.value_starts_();
            for (node_id_t first = range.first; first != range.bound;) {
                node_id_t const bound = static_cast<node_id_t>(min_of<std::size_t>(range.bound, first + run_nodes));
                for (node_id_t node = first; node != bound; ++node) {
                    widen_row_(node, connectivity);
                    starts[base + node + 1] = stored_offsets_[node + 1];
                }
                std::size_t const values_first = saved_.value_first(first);
                metal_load_step<node_id_t> step;
                step.arguments = index.arguments_();
                step.arguments.first_id = static_cast<u32_t>(base + first);
                step.arguments.count = static_cast<u32_t>(bound - first);
                step.saved = saved_.values.subspan(values_first, saved_.value_first(bound) - values_first);
                step.saved_offsets = saved_.value_width ? nullptr : saved_.value_offsets.data() + first;
                step.stored = index.values_tape_.host + stored_offsets_[first];
                step.stored_offsets = starts + base + first;
                step.span_bytes = &span_bytes_;
                step.context_ = &context_;
                step.pipelines_[0] = index.kits_pipeline_(0, vram_kernel_t::load_values_k, 0);
                step.pipelines_[1] = index.kits_pipeline_(0, vram_kernel_t::size_spans_k, 0);
                step.values_ = &index.values_tape_, step.offsets_ = &index.offsets_column_;
                step.parameters_ = index.parameters_buffer_();
                step.stored_offsets_at_ = (base + first) * sizeof(u64_t);
                if (status_t const staged = step.stage_(); failed(staged)) return staged;
                if (status_t const stored = index.kits_.queue_load_values_(step); failed(stored)) return stored;
                // A run drains before its staging takes the next one; a run nothing launched on
                // finishes at once.
                if (failed(context_.wait())) return status_code_t::device_k;
                byte_t const* const span_word = context_.staging_.host + step.span_word_at_;
                span_bytes_ = max_of(span_bytes_, static_cast<std::size_t>(*reinterpret_cast<u64_t const*>(span_word)));
                for (node_id_t node = first; node != bound; ++node)
                    index.keyed_.assign(
                        static_cast<node_id_t>(base + node),
                        safegraphs_widened_key<key_t>(saved_.keys.data() + node * key_bytes, saved_.key_dtype));
                covered_ += bound - first, first = bound;
            }
            return status_code_t::success_k;
        }

        /**
         *  @brief Publishes the appended nodes once every one was copied: the tape's extent, the
         *      longest value and the widest span grow over them, their keys file under the claim,
         *      and the watermark turns, which lifts the claim.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k Some nodes were never copied.
         *  - @c malformed_k A saved neighbor named no node of the columns, or a row kept more
         *    neighbors than the connectivity.
         */
        status_t finish() noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            if (covered_ != entries) return status_code_t::config_mismatch_k;
            if (failed(widened_)) return widened_;
            vram_apple_graph& index = *std::exchange(index_, nullptr);
            index.tape_bytes_ = stored_offsets_[entries];
            index.longest_value_ = max_of(index.longest_value_, longest_);
            index.span_bytes_ = max_of(index.span_bytes_, span_bytes_);
            index.publish_keys_({base, static_cast<node_id_t>(base + entries)});
            index.publish_(base + entries);
            return status_code_t::success_k;
        }

      private:
        friend struct vram_apple_graph;

        /** Widens node @p node 's saved neighbors into its row past @c base, every ID shifted, the
         *  rest sentinels, and a row keeping none pointing at its own node; an ID past the columns'
         *  nodes marks the plan malformed. */
        void widen_row_(node_id_t node, std::size_t connectivity) noexcept {
            node_id_t* const row = index_->neighbor_rows_() + (base + std::size_t {node}) * connectivity;
            std::span<node_id_t const> const saved =
                saved_.degree
                    ? saved_.neighbor_ids.subspan(node * saved_.degree, saved_.degree)
                    : saved_.neighbor_ids.subspan(saved_.neighbor_offsets[node],
                                                  saved_.neighbor_offsets[node + 1] - saved_.neighbor_offsets[node]);
            std::size_t kept = 0;
            for (node_id_t const neighbor : saved) {
                if (neighbor == (std::numeric_limits<node_id_t>::max)()) continue;
                if (neighbor >= entries || kept == connectivity) {
                    widened_ = status_code_t::malformed_k;
                    continue;
                }
                row[kept++] = static_cast<node_id_t>(neighbor + base);
            }
            for (std::size_t slot = kept; slot != connectivity; ++slot)
                row[slot] = (std::numeric_limits<node_id_t>::max)();
            if (!kept) row[0] = static_cast<node_id_t>(base + node);
        }

        /** The saved columns, read in place from the mapping or the caller's buffer. */
        safegraphs_saved_columns<node_id_t> saved_;

        /** Each node's start in the tape, from the plan's first, @b [nodes+1], absolute. */
        growing_buffer<u64_t> stored_offsets_ {global_allocator};

        /** The context the kits' load kernels ride, its own over the planning context's stream, so
         *  the plan outlives the context it was planned on. */
        traversal_context_t context_;

        /** The longest appended node and the widest span the run hooks counted. */
        std::size_t longest_ = 0;
        std::size_t span_bytes_ = 0;

        /** Nodes copied, and @c malformed_k once a widened row was. */
        std::size_t covered_ = 0;
        status_t widened_ = status_code_t::success_k;

        /** The graph appended to, whose claim the plan holds; null in a refused plan, and once
         *  @c finish published. */
        vram_apple_graph* index_ = nullptr;
    };

    /**
     *  @brief Opens a load of the @p saved columns past every claimed node: vets their offsets
     *      columns, sizes every stored value from its saved length, claims the nodes and grows
     *      the storage to hold them.
     *
     *  The keys name the node count and go at @c base on, beside the nodes. A load whose neighbor
     *  names no node of the columns, or whose fixed row keeps more neighbors than its connectivity
     *  allows, finishes with @c malformed_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk.
     *  - @c malformed_k An offsets column is not one longer than the nodes, its sums pass its
     *    column, a ragged node keeps more neighbors than the connectivity, or a saved value does
     *    not parse.
     *  - @c unsupported_k A value past every rung of the face's kits.
     *  - @c overflow_k The nodes pass the ceiling or the reserved range, or their values the tape.
     *  - @c contended_k A batch, a load or a plan in flight.
     */
    expected<load_plan_t> plan_load(safegraphs_saved_columns<node_id_t> const& saved,
                                    traversal_context_t& context) noexcept {
        load_plan_t plan {context.stream_.share()};
        std::size_t const connectivity = this->connectivity(), entries = saved.nodes();
        if (!connectivity) return {std::move(plan), status_code_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {std::move(plan), status_code_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {std::move(plan), kept};
        bool const neighbors_vetted = saved.degree ? saved.neighbor_ids.size() == entries * saved.degree
                                                   : saved.neighbor_offsets.size() == entries + 1 &&
                                                         succeeded(vetted_sums_(saved.neighbor_offsets,
                                                                                saved.neighbor_ids.size()));
        bool const values_vetted = saved.value_width
                                       ? saved.values.size() == entries * saved.value_width
                                       : saved.element_bytes && saved.value_offsets.size() == entries + 1 &&
                                             succeeded(vetted_sums_(saved.value_offsets,
                                                                    saved.values.size() / saved.element_bytes));
        if (!neighbors_vetted || !values_vetted) return {std::move(plan), status_code_t::malformed_k};
        for (std::size_t node = 0; !saved.degree && node != entries; ++node)
            if (saved.neighbor_offsets[node + 1] - saved.neighbor_offsets[node] > connectivity)
                return {std::move(plan), status_code_t::malformed_k};

        // Every stored value's size up front, from its saved length alone, so the tape is known
        // before the claim.
        if (failed(plan.stored_offsets_.resize_uninitialized(entries + 1)))
            return {std::move(plan), status_code_t::out_of_memory_k};
        plan.stored_offsets_[0] = 0;
        for (std::size_t node = 0; node != entries; ++node) {
            auto const [stored_bytes,
                        stored_status] = kits_.stored_bytes_(saved.value_first(node + 1) - saved.value_first(node));
            if (failed(stored_status)) return {std::move(plan), stored_status};
            plan.stored_offsets_[node + 1] = plan.stored_offsets_[node] + stored_bytes;
            plan.longest_ = max_of<std::size_t>(plan.longest_, stored_bytes);
        }
        if (entries && kits_.rung_serving_(max_of(plan.longest_, longest_value_)) == kits_type_::rungs_k)
            return {std::move(plan), status_code_t::unsupported_k};

        auto const [base, claim_status] = claim_(entries);
        if (failed(claim_status)) return {std::move(plan), claim_status};
        plan.index_ = this, plan.base = base, plan.entries = entries;
        std::size_t const stored_total = plan.stored_offsets_[entries];
        if (status_t const grown = grow_nodes_(base + entries); failed(grown)) return {std::move(plan), grown};
        if (status_t const mapped = map_bytes_(tape_bytes_ + stored_total + tape_margin_k); failed(mapped))
            return {std::move(plan), mapped};
        for (u64_t& offset : plan.stored_offsets_) offset += tape_bytes_;
        plan.saved_ = saved;
        plan.span_bytes_ = kits_.span_bytes_(plan.longest_);
        return {std::move(plan), status_code_t::success_k};
    }

    /**
     *  @brief Loads the @p saved columns past every claimed node on the calling thread - the
     *      one-shot over @ref plan_load.
     *
     *  @par Statuses:
     *  - Those of @ref plan_load and @ref load_plan_t::finish.
     */
    status_t load(safegraphs_saved_columns<node_id_t> const& saved, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_load(saved, context);
        if (failed(status)) return status;
        if (status_t const loaded = plan.load({0, static_cast<node_id_t>(plan.entries)}); failed(loaded)) return loaded;
        return plan.finish();
    }

#pragma endregion Checkpoints

    /**
     *  @brief The pipeline of the kits' kernel @p host_name, outside the rungs the host builds at
     *      the first @c reserve: built in the kits' own library on first use, then kept by the
     *      device's registry. What a face's own launches encode with.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve bound the device yet.
     *  - @c unsupported_k The kits' library exports no such kernel.
     */
    expected<metal_pipeline_state_t> kits_pipeline_named_(char const* host_name) const noexcept {
        metal_context_t* const device = context();
        char const* const source = metal_basic_source_of<kits_type_>();
        if (!connectivity() || !device || !source) return {{}, status_code_t::uninitialized_k};
        auto const [state, status] = device->pipeline(source, host_name);
        return {state, failed(status) ? status_code_t::unsupported_k : status_code_t::success_k};
    }

    /** The shared buffers a face's own launches bind: the values tape, the offsets column, the
     *  neighbors matrix, and the kits' parameters or the tape standing in for none. */
    metal_buffer_t const& values_buffer_() const noexcept { return values_tape_; }
    metal_buffer_t const& offsets_buffer_() const noexcept { return offsets_column_; }
    metal_buffer_t const& neighbors_buffer_() const noexcept { return neighbors_matrix_; }
    metal_buffer_t const* parameters_buffer_() const noexcept {
        metal_buffer_t const& parameters = kits_.parameters_();
        return parameters ? &parameters : &values_tape_;
    }

  private:

    /** Bytes of zeroed margin past the tape's last value, so a word load ending there never
     *  faults. */
    inline static constexpr std::size_t tape_margin_k = sizeof(u64_t);

    /** Lanes per simdgroup, the width every kernel's collectives span. */
    static constexpr u32_t wave_lanes() noexcept { return metal_simdgroup_width_k; }

    /** Row zero of the neighbors matrix, as the IDs its rows hold. */
    node_id_t* neighbor_rows_() const noexcept { return reinterpret_cast<node_id_t*>(neighbors_matrix_.host); }

    /** The offsets column, one start past every node. */
    u64_t* value_starts_() const noexcept { return reinterpret_cast<u64_t*>(offsets_column_.host); }

    /** The neighbor row of @p node, @p connectivity IDs. */
    std::span<node_id_t const> row_(node_id_t node, std::size_t connectivity) const noexcept {
        return {neighbor_rows_() + std::size_t {node} * connectivity, connectivity};
    }

    /** The stored value of @p node. */
    bytes_view_t value_of_(node_id_t node) const noexcept {
        u64_t const* const starts = value_starts_();
        return {values_tape_.host + starts[node], static_cast<std::size_t>(starts[node + 1] - starts[node])};
    }

    /** The scalars every kit launch shares. */
    metal_basic_arguments_t arguments_() const noexcept {
        metal_basic_arguments_t arguments {};
        arguments.connectivity = static_cast<u32_t>(connectivity());
        arguments.metric = static_cast<u32_t>(kernel_);
        return arguments;
    }

    /** The scalars every graph launch shares, in the words the graph kernels read. */
    metal_graph_arguments_t graph_arguments_() const noexcept {
        metal_graph_arguments_t arguments {};
        arguments.connectivity = static_cast<u32_t>(connectivity());
        return arguments;
    }

    /** The pipeline of the kits' @p kernel on kit rung @p rung and register rung @p slots, built at
     *  the first @c reserve. */
    metal_pipeline_state_t kits_pipeline_(std::size_t rung, vram_kernel_t kernel, u32_t slots) const noexcept {
        return kits_pipelines_[rung][static_cast<std::size_t>(kernel)][gpu_rung_index(slots)];
    }

    /** The pipeline of the graph's @p kernel, built at the first @c reserve. */
    metal_pipeline_state_t graph_pipeline_(vram_kernel_t kernel) const noexcept {
        return graph_pipelines_[static_cast<std::size_t>(kernel)];
    }

    /** Walkers one threadgroup hosts on the kit rung @p rung: a simdgroup each when a lane scores
     *  a node alone, one walking the whole threadgroup otherwise. */
    static u32_t walkers_per_block_(std::size_t rung) noexcept {
        return rung < kits_type_::rungs_k && kits_type_::lanes_per_entry_(rung) > 1 ? 1u : metal_simdgroups_k;
    }

    /** Threadgroups of the kits' @p kernel resident per GPU core on kit rung @p rung and register
     *  rung @p slots: as many as one threadgroup of its pipeline holds threads, which its register
     *  use bounds, since Metal reports no occupancy; one at least. */
    u32_t resident_blocks_(std::size_t rung, vram_kernel_t kernel, u32_t slots) const noexcept {
        return max_of<u32_t>(kits_pipeline_(rung, kernel, slots).max_threads / metal_threadgroup_threads_k, 1u);
    }

    /** Threadgroups a walk over @p queries launches on the kit rung @p rung and the register rung
     *  @p slots: a walker per query up to the grid the device runs at once. */
    u32_t walk_blocks_(std::size_t queries, std::size_t rung, u32_t slots) const noexcept {
        return static_cast<u32_t>(max_of<std::size_t>(
            min_of<std::size_t>(
                divide_round_up<std::size_t>(queries, walkers_per_block_(rung)),
                std::size_t {device_.multiprocessor_count} * resident_blocks_(rung, vram_kernel_t::search_k, slots)),
            1));
    }

    /** Which frontier a walk keeps: a register rung, or the arena's past the ladder. */
    enum class frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

    /** The breadth an insert or a refine walks: as given, or twice the connectivity, as on the
     *  CPU, and never under the @c build_breadth_k a modality's selector needs. */
    u32_t build_expansion_(u32_t expansion_factor) const noexcept {
        if (expansion_factor) return expansion_factor;
        u32_t const twice = static_cast<u32_t>(default_expansion_factor(connectivity()));
        if constexpr (a_kit_set_with_build_breadth<kits_type_>) return max_of(twice, kits_type_::build_breadth_k);
        else return twice;
    }

    /** Candidates a walk's selection caches: all of its candidates, @p candidate_slots ranks per
     *  lane on the register ladder, or the arena frontier's @p expansion past it. */
    static u32_t walk_selection_nodes_(u32_t expansion, u32_t candidate_slots) noexcept {
        return candidate_slots ? candidate_slots * wave_lanes() : expansion;
    }

    /** Candidates a back-link round's re-selection caches: the node's live row and a row's worth of
     *  appended nodes. */
    u32_t reselection_nodes_() const noexcept { return 2 * static_cast<u32_t>(connectivity()); }

    /** Bytes of one walker's scratch over @p selection_nodes candidates, as the walks carve it: the
     *  query span and the pivot span of @p span_bytes each, a simdgroup's three ID spans, then per
     *  candidate its score, its selected and candidate IDs, its kept position and its distance, in
     *  whole 16-byte words. */
    static u32_t build_scratch_bytes_(u32_t span_bytes, u32_t selection_nodes) noexcept {
        std::size_t const ids_bytes = round_up_to_multiple<std::size_t>(3 * metal_simdgroup_width_k * sizeof(u32_t),
                                                                        16);
        std::size_t const candidate_bytes = gpu_scored_node_bytes<u32_t> + 3 * sizeof(u32_t) + sizeof(f32_t);
        return static_cast<u32_t>(
            round_up_to_multiple<std::size_t>(2 * span_bytes + ids_bytes + selection_nodes * candidate_bytes, 16));
    }

    /** Everything one building pass derives from its requested expansion, once: the walk's breadth
     *  and register rung, the candidates its selection and a back-link round hold, the scratch both
     *  keep in the arena slice, and that slice. */
    struct build_shape_t {
        u32_t expansion = 0, candidate_slots = 0, walk_selection_nodes = 0, reselection_nodes = 0;
        u32_t scratch_bytes = 0;
        gpu_arena_shape_t arena_shape {};
    };

    /** The shape of a building pass requesting @p expansion_factor over nodes preparing into
     *  @p span_bytes; the scratch always rides the arena, since threadgroup memory would bound how
     *  wide a value may grow. */
    build_shape_t build_shape_(u32_t expansion_factor, u32_t span_bytes) const noexcept {
        build_shape_t shape;
        shape.expansion = build_expansion_(expansion_factor);
        shape.candidate_slots = gpu_candidate_slots(shape.expansion, wave_lanes());
        shape.walk_selection_nodes = walk_selection_nodes_(shape.expansion, shape.candidate_slots);
        shape.reselection_nodes = reselection_nodes_();
        shape.scratch_bytes = build_scratch_bytes_(span_bytes,
                                                   max_of(shape.walk_selection_nodes, shape.reselection_nodes));
        shape.arena_shape = arena_shape_for_(
            shape.walk_selection_nodes,
            shape.candidate_slots ? frontier_kind_t::register_ladder_k : frontier_kind_t::wide_arena_k,
            shape.scratch_bytes);
        return shape;
    }

    /** One walker's arena slice for a walk keeping @p candidate_nodes candidates: the exact visited
     *  set over every ID it claims, plus the spill tail of a wide frontier, then its scratch. */
    gpu_arena_shape_t arena_shape_for_(u32_t candidate_nodes, frontier_kind_t frontier_kind,
                                       u32_t scratch_bytes) const noexcept {
        u32_t const connectivity = static_cast<u32_t>(this->connectivity());
        u32_t const tail = frontier_kind == frontier_kind_t::wide_arena_k
                               ? gpu_walk_tail_entries(candidate_nodes, connectivity, wave_lanes())
                               : 0;
        return gpu_arena_shape(gpu_walk_visits(candidate_nodes, connectivity, wave_lanes()), tail, scratch_bytes);
    }

    /** A walk's breadth: as given, or twice the connectivity, and never under the count. */
    u32_t walk_expansion_(search_config_t config) const noexcept {
        u32_t const expansion = config.expansion_factor ? config.expansion_factor
                                                        : static_cast<u32_t>(default_expansion_factor(connectivity()));
        return gpu_walk_breadth(expansion, config.count);
    }

    /** The register rung a walk keeps, covering both its breadth and its count. */
    u32_t walk_slots_(search_config_t config) const noexcept {
        return gpu_candidate_slots(walk_expansion_(config), wave_lanes());
    }

    /** One walker's slice: the register rung's visited set, or the arena frontier's with its tail
     *  past the ladder, then its scratch. */
    gpu_arena_shape_t walk_arena_shape_(search_config_t config, u32_t span_bytes) const noexcept {
        u32_t const expansion = walk_expansion_(config), slots = walk_slots_(config);
        u32_t const selection_nodes = walk_selection_nodes_(expansion, slots);
        return arena_shape_for_(selection_nodes,
                                slots ? frontier_kind_t::register_ladder_k : frontier_kind_t::wide_arena_k,
                                build_scratch_bytes_(span_bytes, selection_nodes));
    }

    /** The widest span any node of @p tape prepares into, raising @p longest to the longest value
     *  among them. */
    std::size_t span_bytes_of_(tape_view<byte_t const> tape, std::size_t& longest) const noexcept {
        std::size_t widest = 0;
        for (std::size_t node = 0; node != tape.size(); ++node) {
            auto const entry = tape[node];
            longest = max_of(longest, entry.size());
            widest = max_of(widest, kits_.span_bytes_(entry.data(), entry.size()));
        }
        return widest;
    }

    /** Sizes @p context 's arena for @p blocks threadgroups of slices shaped @p shape, taking
     *  epochs for @p work walks and zeroing it when fresh or when the epochs wrap, and fills the
     *  slice shape and the epoch base into @p arguments, as dense Apple does. */
    status_t shape_arena_(traversal_context_t& context, metal_basic_arguments_t& arguments, gpu_arena_shape_t shape,
                          std::size_t blocks, std::size_t work) const noexcept {
        std::size_t const slices = blocks * metal_simdgroups_k;
        bool fresh = false;
        if (status_t const sized = context.ensure_(context.arena_, slices * shape.template slice_bytes<u32_t>(), fresh);
            failed(sized))
            return sized;
        if (fresh || context.epoch_floor_ + work > 0xFFFFFFFEull) {
            if (!fresh) std::memset(context.arena_.host, 0, context.arena_.bytes);
            context.epoch_floor_ = 0;
        }
        arguments.visited_slots = shape.visited_slots;
        arguments.hash_shift = shape.visited_hash_shift;
        arguments.tail_entries = shape.tail_entries;
        arguments.scratch_bytes_per_warp = shape.scratch_bytes;
        arguments.slice_bytes = static_cast<u32_t>(shape.template slice_bytes<u32_t>());
        arguments.epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += work;
        return status_code_t::success_k;
    }

    /** Binds the scalars, the three columns, the context's counters and the kits' parameters every
     *  kit kernel reads. */
    void bind_kit_storage_(metal_launch_t& launch, metal_basic_arguments_t const& arguments,
                           traversal_context_t const& context) const noexcept {
        launch.bind_bytes(&arguments, sizeof(arguments), metal_basic_arguments_buffer_k);
        launch.bind(values_tape_, metal_basic_values_buffer_k);
        launch.bind(offsets_column_, metal_basic_offsets_buffer_k);
        launch.bind(neighbors_matrix_, metal_basic_neighbors_buffer_k);
        launch.bind(context.counters_, metal_basic_counters_buffer_k);
        launch.bind(*parameters_buffer_(), metal_basic_parameters_buffer_k);
    }

    /** Waves an insert of the IDs from @p first_id up to @p end_id launches in. */
    static std::size_t insert_waves_(std::size_t first_id, std::size_t end_id) noexcept {
        std::size_t waves = 0;
        for (std::size_t wave_first = first_id; wave_first != end_id; ++waves)
            wave_first = static_cast<std::size_t>(
                vram_admission_wave_end(wave_first, end_id, vram_basic_admission_share_log2_k));
        return waves;
    }

    /**
     *  @brief Commits the batch @p arguments names from @p first_id to @p end_id in waves, each the
     *      share @c vram_basic_admission_share_log2_k of the IDs below it, every launch sized by
     *      @p shape on the kit rung @p rung over @p blocks threadgroups.
     *
     *  Each wave commits behind the last on the context's stream: its insert, the back-link over
     *  every node the insert's picks touched, its re-selection and that one's back-link. The chains
     *  sit at the staging's start; each walk threads them under its own touched word from
     *  @p touched_at, and each launch counts into its own block from @p counters_at. A failure
     *  after any commit is kept, as the committed nodes stay claimed.
     */
    status_t launch_insert_(metal_basic_arguments_t const& arguments, build_shape_t const& shape, std::size_t first_id,
                            std::size_t end_id, std::size_t rung, u32_t blocks, std::size_t touched_at,
                            std::size_t counters_at, traversal_context_t& context) noexcept {
        std::size_t const entries = end_id - first_id, row_ids = connectivity();
        std::size_t const next_at = end_id * sizeof(node_id_t);
        std::size_t const touched_ids_at = next_at + entries * row_ids * sizeof(node_id_t);
        bool committed = false;
        std::size_t launch_index = 0;
        // Every wave commits its insert, the back-link over its picks, its re-selection, and that
        // one's back-link, each walk under its own touched word.
        struct step_t {
            vram_kernel_t kernel, walk;
        };
        static constexpr step_t steps_k[] = {{vram_kernel_t::insert_k, vram_kernel_t::insert_k},
                                             {vram_kernel_t::back_link_k, vram_kernel_t::insert_k},
                                             {vram_kernel_t::refine_k, vram_kernel_t::refine_k},
                                             {vram_kernel_t::back_link_k, vram_kernel_t::refine_k}};
        auto const commit = [&](vram_kernel_t kernel, metal_basic_arguments_t const& wave,
                                std::size_t walk) noexcept -> status_t {
            bool const walks = kernel != vram_kernel_t::back_link_k;
            metal_basic_arguments_t launched = wave;
            launched.selection_nodes = walks ? shape.walk_selection_nodes : shape.reselection_nodes;
            metal_launch_t launch {context.api(), context.stream(),
                                   kits_pipeline_(rung, kernel, walks ? shape.candidate_slots : 0).pipeline};
            if (!launch) return status_code_t::device_k;
            bind_kit_storage_(launch, launched, context);
            launch.bind(context.staging_, metal_basic_counters_buffer_k,
                        counters_at + launch_index++ * metal_counters_bytes_k);
            launch.bind(context.arena_, metal_basic_arena_buffer_k);
            launch.bind(context.staging_, metal_basic_link_heads_buffer_k);
            launch.bind(context.staging_, metal_basic_link_next_buffer_k, next_at);
            launch.bind(context.staging_, metal_basic_touched_ids_buffer_k, touched_ids_at);
            launch.bind(context.staging_, metal_basic_touched_count_buffer_k, touched_at + walk * sizeof(u32_t));
            return context.submit_(launch, blocks, metal_threadgroup_threads_k);
        };
        std::size_t wave = 0;
        for (std::size_t wave_first = first_id; wave_first != end_id; ++wave) {
            std::size_t const wave_end = static_cast<std::size_t>(
                vram_admission_wave_end(wave_first, end_id, vram_basic_admission_share_log2_k));
            metal_basic_arguments_t launched = arguments;
            launched.first_id = static_cast<u32_t>(wave_first);
            launched.end_id = launched.sampling_bound = static_cast<u32_t>(wave_end);
            for (step_t const step : steps_k) {
                // The re-selections take the batch's second run of epochs.
                bool const reselects = step.walk == vram_kernel_t::refine_k;
                launched.epoch_base = arguments.epoch_base + static_cast<u32_t>(wave_first - first_id) +
                                      (reselects ? static_cast<u32_t>(entries) : 0u);
                status_t const submitted = commit(step.kernel, launched, 2 * wave + reselects);
                if (failed(submitted)) return committed ? fail_(submitted) : submitted;
                committed = true;
            }
            wave_first = wave_end;
        }
        return status_code_t::success_k;
    }

    /** Folds the telemetry of @p launches counter blocks at @p blocks into @p context 's counters,
     *  which @c spent reads: every tally its low word carrying into its high one. */
    static void fold_counters_(traversal_context_t& context, byte_t const* blocks, std::size_t launches) noexcept {
        u32_t* const into = metal_words(context.counters_);
        for (std::size_t launch = 0; launch != launches; ++launch) {
            u32_t const* const words = reinterpret_cast<u32_t const*>(blocks + launch * metal_counters_bytes_k);
            for (u32_t const tally : {metal_hops_k, metal_scored_k, metal_restarts_k, metal_lock_attempts_k,
                                      metal_lock_skips_k, metal_reselections_k, metal_locked_drops_k,
                                      metal_saturations_k, metal_tail_compactions_k, metal_tail_dropped_k}) {
                u64_t const sum = metal_tally(into, tally) + metal_tally(words, tally);
                into[tally] = static_cast<u32_t>(sum), into[tally + 1] = static_cast<u32_t>(sum >> 32);
            }
            into[metal_changed_k] += words[metal_changed_k];
        }
    }

    /** Keeps @p failure as the status unless one is kept already; returns it for the caller's own
     *  report. */
    status_t fail_(status_t failure) noexcept {
        keep_first_failure<fu::standard_atomic_ref>(status_, failure);
        return failure;
    }

    /** Claims @p nodes from the watermark: one compare-exchange that succeeds only on an open word
     *  with no batch or load in flight, so its holder alone grows, launches and publishes them. */
    expected<node_id_t> claim_(std::size_t nodes) noexcept {
        auto const [first_id, claim_status] = claims_.claim(size(), nodes, ceiling_nodes_);
        return {static_cast<node_id_t>(first_id), claim_status};
    }

    /** Gives back the claim over @p nodes from @p first, which launched nothing; a word that moved
     *  on, closed by a caller around a load, stays as it is. */
    void unclaim_(std::size_t first, std::size_t nodes) noexcept { claims_.unclaim(first, nodes); }

    /**
     *  @brief Binds the graph at its first @c reserve: checks the node-ID width, uploads the kits'
     *      parameters, builds every pipeline so no launch waits on the device's registry, queries
     *      the device, and allocates the three columns at the ceiling, which Metal commits lazily.
     *
     *  No column outgrows the device's longest buffer, and none the working set Metal recommends,
     *  whatever the first capacity asks.
     */
    status_t bind_(reserve_capacity_t const& config) noexcept {
        if constexpr (!std::is_same_v<node_id_t, u32_t>) return status_code_t::unsupported_k;
        metal_context_t* const device = context();
        if (!device) return status_code_t::device_k;
        if (status_t const uploaded = kits_.upload_parameters_(*device); failed(uploaded)) return uploaded;
        char const* const source = metal_basic_source_of<kits_type_>();
        if (!source) return status_code_t::out_of_memory_k;
        vram_kernel_symbol_t symbol;
        auto const build = [&](metal_pipeline_state_t& into) noexcept -> status_t {
            auto const [state, status] = device->pipeline(source, symbol.text);
            if (failed(status)) return status;
            if (state.max_threads < metal_threadgroup_threads_k) return status_code_t::unsupported_k;
            into = state;
            return status_code_t::success_k;
        };
        for (std::size_t rung = 0; rung != kits_type_::rungs_k; ++rung)
            for (metal_basic_rung_t const instantiated : metal_basic_rungs) {
                metal_basic_kernel_symbol(symbol, rung, instantiated);
                if (status_t const built = build(kits_pipelines_[rung][static_cast<std::size_t>(instantiated.kernel)]
                                                                [gpu_rung_index(instantiated.slots)]);
                    failed(built))
                    return built;
            }
        for (vram_kernel_t const kernel : metal_basic_load_kernels) {
            metal_basic_kernel_symbol(symbol, 0, {kernel, 0});
            if (status_t const built = build(kits_pipelines_[0][static_cast<std::size_t>(kernel)][0]); failed(built))
                return built;
        }
        for (vram_kernel_t const kernel : metal_graph_kernels) {
            metal_graph_kernel_symbol(symbol, kernel);
            if (status_t const built = build(graph_pipelines_[static_cast<std::size_t>(kernel)]); failed(built))
                return built;
        }
        // The pipelines report the simdgroup width the specification names, so they build first.
        auto const [specs, status] = metal_device_specs_t::query(*device, device_.ordinal);
        if (failed(status)) return status;
        device_ = specs;
        std::size_t const row_bytes = config.connectivity * sizeof(node_id_t);
        std::size_t const budget_bytes = min_of(device_.working_set_bytes, device_.max_buffer_bytes);
        std::size_t const ceiling_nodes = min_of(
            min_of(min_of(config.ceiling, max_node_count_k),
                   max_of<std::size_t>(config.capacity, budget_bytes / row_bytes)),
            min_of(device_.max_buffer_bytes / row_bytes, device_.max_buffer_bytes / sizeof(u64_t) - 1));
        std::size_t const tape_bytes = min_of(max_of(config.bytes + tape_margin_k, budget_bytes),
                                              device_.max_buffer_bytes);
        if (config.capacity > ceiling_nodes || config.bytes + tape_margin_k > tape_bytes)
            return status_code_t::overflow_k;
        // The offsets column holds one start past the last node.
        offsets_column_ = allocator_.allocate((ceiling_nodes + 1) * sizeof(u64_t));
        neighbors_matrix_ = allocator_.allocate(ceiling_nodes * row_bytes);
        values_tape_ = allocator_.allocate(tape_bytes);
        if (!offsets_column_ || !neighbors_matrix_ || !values_tape_) return status_code_t::out_of_memory_k;
        ceiling_nodes_ = ceiling_nodes;
        fu::standard_atomic_ref<std::size_t>(connectivity_).store(config.connectivity, std::memory_order_release);
        return status_code_t::success_k;
    }

    /**
     *  @brief Covers the host keys and maps the columns up to @p nodes, host work alone, leaving
     *      the fresh neighbor rows for @c grow_nodes_ to fill.
     *
     *  @par Statuses:
     *  - @c overflow_k The @p nodes pass the ceiling or the host keys.
     */
    status_t map_nodes_(std::size_t nodes) noexcept {
        if (nodes > ceiling_nodes_) return status_code_t::overflow_k;
        if (status_t const covered = keyed_.cover(nodes); failed(covered)) return covered;
        std::size_t mapped = capacity();
        while (mapped < nodes &&
               !fu::standard_atomic_ref<std::size_t>(mapped_nodes_)
                    .compare_exchange_weak(mapped, nodes, std::memory_order_acq_rel, std::memory_order_acquire)) {}
        return status_code_t::success_k;
    }

    /**
     *  @brief Maps the tape up to @p bytes, its margin included.
     *
     *  @par Statuses:
     *  - @c overflow_k The @p bytes pass the tape.
     */
    status_t map_bytes_(std::size_t bytes) noexcept {
        if (bytes > values_tape_.bytes) return status_code_t::overflow_k;
        std::size_t mapped = fu::standard_atomic_ref<std::size_t>(mapped_bytes_).load(std::memory_order_acquire);
        while (mapped < bytes &&
               !fu::standard_atomic_ref<std::size_t>(mapped_bytes_)
                    .compare_exchange_weak(mapped, bytes, std::memory_order_acq_rel, std::memory_order_acquire)) {}
        return status_code_t::success_k;
    }

    /** Maps up to @p nodes and sentinel-fills every mapped neighbor row no fill reached yet, on the
     *  host; callers hold the claim or the closed word. */
    status_t grow_nodes_(std::size_t nodes) noexcept {
        if (status_t const mapped = map_nodes_(nodes); failed(mapped)) return mapped;
        std::size_t const mapped_nodes = capacity(), row_bytes = connectivity() * sizeof(node_id_t);
        if (mapped_nodes <= filled_nodes_) return status_code_t::success_k;
        // The sentinel is all ones at any width.
        std::memset(neighbors_matrix_.host + filled_nodes_ * row_bytes, 0xFF,
                    (mapped_nodes - filled_nodes_) * row_bytes);
        filled_nodes_ = mapped_nodes;
        return status_code_t::success_k;
    }

    /** Whether slot value @p neighbor of node @p node is a link a snapshot keeps: below @p nodes,
     *  and not the node's own ID, which marks a published node with none. */
    static bool kept_link_(node_id_t neighbor, node_id_t node, node_id_t nodes) noexcept {
        return neighbor < nodes && neighbor != node;
    }

    /** A column of @p given entries where @p needed belong: @c overflow_k short,
     *  @c config_mismatch_k long. */
    static status_t sized_exactly_(std::size_t given, std::size_t needed) noexcept {
        return given < needed   ? status_code_t::overflow_k
               : given > needed ? status_code_t::config_mismatch_k
                                : status_code_t::success_k;
    }

    /** Prefix sums from zero, never falling, ending within the @p elements they index;
     *  @c malformed_k otherwise. */
    static status_t vetted_sums_(std::span<u64_t const> sums, std::size_t elements) noexcept {
        if (sums.empty() || sums[0] != 0 || sums.back() > elements) return status_code_t::malformed_k;
        for (std::size_t position = 1; position != sums.size(); ++position)
            if (sums[position] < sums[position - 1]) return status_code_t::malformed_k;
        return status_code_t::success_k;
    }

    /** Opens a save of every node under the watermark with the claim word closed, sizing nothing
     *  yet. */
    expected<save_plan_t> open_save_(traversal_context_t& context) noexcept {
        save_plan_t plan;
        if (context.device() != device_.ordinal || !context) return {std::move(plan), status_code_t::config_mismatch_k};
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        plan.index_ = this;
        plan.extents.nodes = size();
        return {std::move(plan), status_code_t::success_k};
    }

    /** The mean bytes of a published value, one at least, which sizes a round's hubs and
     *  neighborhoods as a dense row would. */
    std::size_t mean_value_bytes_() const noexcept {
        std::size_t const entries = size();
        return entries ? max_of<std::size_t>(divide_round_up(tape_bytes_, entries), 1) : 1;
    }

    /** The kit rung a round's gather and restitch score on: the one serving the longest stored
     *  value. */
    std::size_t round_rung_() const noexcept {
        return min_of(kits_.rung_serving_(longest_value_), kits_type_::rungs_k - 1);
    }

    /** Candidates a restitch's selection caches: a live row and a row of mates, twice over. */
    u32_t round_selection_nodes_() const noexcept { return 2 * reselection_nodes_(); }

    /** Bytes of one gather's or restitch's scratch slice: its spans and its selection, no visited
     *  set. */
    std::size_t round_slice_bytes_() const noexcept {
        u32_t const scratch = build_scratch_bytes_(static_cast<u32_t>(span_bytes_), round_selection_nodes_());
        return gpu_arena_shape(0, 0, scratch).template slice_bytes<u32_t>();
    }

    /**
     *  @brief A round's shape over @p entries entries of @p value_bytes each on average under
     *      @p cache_bytes, zero for this device's figure, never more than it.
     *
     *  Half the budget holds a draw's hubs, never below one, and the other half a neighborhood's
     *  entries, never below a row of mates and the entry itself. A gather inside the neighborhoods
     *  over capacity draws at most two hubs per capacity plus one each; a restitch runs as many
     *  threadgroups as the device keeps resident.
     */
    round_shape_t round_shape_(std::size_t entries, std::size_t value_bytes, std::size_t cache_bytes) const noexcept {
        std::size_t const device_bytes = device_.cache_bytes_per_core();
        std::size_t const budget = cache_bytes ? min_of(cache_bytes, device_bytes) : device_bytes;
        std::size_t const entry_bytes = max_of<std::size_t>(value_bytes, 1);
        round_shape_t shape;
        shape.entries = entries;
        shape.hubs_per_draw = max_of<std::size_t>(budget / (2 * entry_bytes), 1);
        std::size_t const rest = budget - min_of(budget, shape.hubs_per_draw * entry_bytes);
        shape.capacity = max_of(rest / entry_bytes, connectivity() + 1);
        std::size_t const wide = divide_round_up(entries, shape.capacity);
        shape.hubs = max_of(round_first_hubs(entries, shape.hubs_per_draw, shape.capacity), 3 * wide + 1);
        shape.runs = max_of<std::size_t>(wide, 1);
        shape.groups = std::size_t {device_.multiprocessor_count} *
                       resident_blocks_(round_rung_(), vram_kernel_t::restitch_k, 0);
        return shape;
    }

    /** The round buffer's layout over @p entries entries for a round of @p shape, or for a lone
     *  check's zeros, every column on a 16-byte boundary, as dense Apple carves it. */
    round_layout_t round_layout_(std::size_t entries, round_shape_t const& shape) const noexcept {
        std::size_t offset = 0;
        auto carve = [&](std::size_t bytes) noexcept {
            std::size_t const at = offset;
            offset = round_up_to_multiple<std::size_t>(offset + bytes, 16);
            return at;
        };
        std::size_t const ids = entries * sizeof(u32_t);
        std::size_t const words = divide_round_up<std::size_t>(entries, 32) * sizeof(u32_t);
        round_layout_t layout;
        layout.counters = carve(metal_round_counter_words_k * sizeof(u32_t));
        layout.depths_from_root = carve(entries);
        layout.depths_to_root = carve(entries);
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
        layout.entry_neighborhoods = carve(entries * sizeof(u16_t));
        layout.neighborhood_entries = carve(ids);
        layout.neighborhood_offsets = carve(split_room(entries, max_of<std::size_t>(shape.hubs_per_draw, 1)) *
                                            sizeof(u32_t));
        layout.hub_nodes = carve(shape.hubs * sizeof(u32_t));
        layout.runs = carve(shape.runs * sizeof(vram_neighborhood_run_t));
        layout.run_items = carve((shape.runs + 1) * sizeof(u32_t));
        layout.stitch = carve(shape.groups * metal_simdgroups_k * round_slice_bytes_());
        layout.bytes = max_of(offset, islands_bytes);
        return layout;
    }

    /** Grows @p context 's round buffer to @p bytes and zeroes its counters, so every phase of the
     *  plan adds its telemetry to them. */
    static status_t carve_round_(traversal_context_t& context, std::size_t bytes) noexcept {
        bool fresh = false;
        if (status_t const grown = context.ensure_(context.round_, bytes, fresh); failed(grown)) return grown;
        std::memset(context.counters_.host, 0, context.counters_.bytes);
        return status_code_t::success_k;
    }

    /** One column of the round buffer and the slot it binds at. */
    struct round_column_t {
        unsigned int slot;
        std::size_t offset;
    };

    /** Binds the health check's columns of @p layout, at the dense slots every round kernel
     *  reads. */
    static void bind_health_(metal_launch_t& launch, round_layout_t const& layout,
                             traversal_context_t const& context) noexcept {
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

    /** Threadgroups a graph pass of @p kernel over @p items spreads over, a simdgroup an item up to
     *  the grid the device runs at once, never none. */
    u32_t spread_(vram_kernel_t kernel, std::size_t items) const noexcept {
        u32_t const resident = max_of<u32_t>(graph_pipeline_(kernel).max_threads / metal_threadgroup_threads_k, 1u);
        return static_cast<u32_t>(
            max_of<std::size_t>(min_of<std::size_t>(divide_round_up<std::size_t>(items, metal_simdgroups_k),
                                                    std::size_t {device_.multiprocessor_count} * resident),
                                1));
    }

    /** Commits one pass of a health check's or the heal's @p kernel over @p items on @p context 's
     *  stream, the check's columns and the islands' bound, as dense Apple's @c launch_round_. */
    status_t launch_check_(vram_kernel_t kernel, metal_graph_arguments_t const& arguments, round_layout_t const& layout,
                           std::size_t items, traversal_context_t& context) noexcept {
        metal_launch_t launch {context.api(), context.stream(), graph_pipeline_(kernel).pipeline};
        if (!launch) return status_code_t::device_k;
        launch.bind_bytes(&arguments, sizeof(arguments), metal_graph_arguments_buffer_k);
        launch.bind(neighbors_matrix_, metal_graph_neighbors_buffer_k);
        bind_health_(launch, layout, context);
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
        return context.submit_(launch, alone ? 1u : spread_(kernel, items), metal_threadgroup_threads_k);
    }

    /** Commits the heal of the islands @p health named, none when it named none. */
    status_t queue_heal_(health_plan_t& health, traversal_context_t& context) noexcept {
        if (!health.islands()) return status_code_t::success_k;
        metal_graph_arguments_t arguments = graph_arguments_();
        arguments.end_id = health.visible;
        arguments.islands = static_cast<u32_t>(health.islands());
        return launch_check_(vram_kernel_t::heal_islands_k, arguments, health.layout, 1, context);
    }

    /** Commits a gather's or a restitch's kits @p kernel of @p plan on @p context 's stream, over
     *  the round's threadgroups on the kit rung its longest value needs: the check's columns and
     *  the round's bound, the scratch slices in the round buffer's stitch region, and the claim
     *  word zeroed first. */
    status_t launch_neighborhoods_(vram_kernel_t kernel, metal_basic_arguments_t const& arguments,
                                   refine_plan_t const& plan, traversal_context_t& context) noexcept {
        round_layout_t const& layout = plan.health.layout;
        u32_t* const words = plan.health.counters();
        words[metal_round_claimed_k] = 0;
        metal_launch_t launch {context.api(), context.stream(), kits_pipeline_(round_rung_(), kernel, 0).pipeline};
        if (!launch) return status_code_t::device_k;
        bind_kit_storage_(launch, arguments, context);
        bind_health_(launch, layout, context);
        round_column_t const columns[] = {
            {metal_basic_entry_neighborhoods_buffer_k, layout.entry_neighborhoods},
            {metal_basic_neighborhood_entries_buffer_k, layout.neighborhood_entries},
            {metal_basic_neighborhood_offsets_buffer_k, layout.neighborhood_offsets},
            {metal_basic_hub_nodes_buffer_k, layout.hub_nodes},
            {metal_basic_runs_buffer_k, layout.runs},
            {metal_basic_run_items_buffer_k, layout.run_items},
            {metal_basic_arena_buffer_k, layout.stitch},
        };
        for (auto const& [slot, offset] : columns) launch.bind(context.round_, slot, offset);
        return context.submit_(launch, max_of<std::size_t>(plan.shape_.groups, 1), metal_threadgroup_threads_k);
    }

    /** The scalars a gather or a restitch of @p plan reads. */
    metal_basic_arguments_t round_arguments_(refine_plan_t const& plan) const noexcept {
        metal_basic_arguments_t arguments = arguments_();
        arguments.end_id = plan.health.visible;
        arguments.span_bytes = static_cast<u32_t>(span_bytes_);
        arguments.selection_nodes = round_selection_nodes_();
        arguments.scratch_bytes_per_warp = build_scratch_bytes_(arguments.span_bytes, arguments.selection_nodes);
        arguments.slice_bytes = static_cast<u32_t>(round_slice_bytes_());
        return arguments;
    }

    /** Commits one gather of @p plan over the runs in hand, a first one's over IDs. */
    status_t queue_gather_(refine_plan_t const& plan, traversal_context_t& context) noexcept {
        metal_basic_arguments_t arguments = round_arguments_(plan);
        arguments.runs_count = static_cast<u32_t>(plan.runs_);
        arguments.pass = plan.neighborhoods_ ? 1u : 0u;
        return launch_neighborhoods_(vram_kernel_t::gather_neighborhoods_k, arguments, plan, context);
    }

    /** Commits the restitch of every neighborhood @p plan 's order holds, its link tallies
     *  zeroed. */
    status_t queue_restitch_(refine_plan_t const& plan, traversal_context_t& context) noexcept {
        metal_basic_arguments_t arguments = round_arguments_(plan);
        arguments.neighborhoods = static_cast<u32_t>(plan.neighborhoods_);
        u32_t* const words = plan.health.counters();
        for (u32_t const tally : {metal_round_added_k, metal_round_dropped_k}) words[tally] = words[tally + 1] = 0;
        return launch_neighborhoods_(vram_kernel_t::restitch_k, arguments, plan, context);
    }

    /**
     *  @brief Binds @p plan to a gather of every source of @p sources other than this graph: copies
     *      their nodes past the watermark on the host, files their keys, publishes them, and
     *      empties those sources, then opens the round over every node.
     *
     *  Every check happens first: each source's device, connectivity, metric and kept status, and
     *  the ceiling. Then each source's values copy past this tape, its starts shift by this tape's
     *  bytes, its neighbor rows rebase onto the IDs past the watermark, and its keys move. The
     *  sources' claim words close for the copies alone; this graph's stays closed for the plan's
     *  whole life, whatever it answers.
     */
    template <typename owner_type_>
    status_t open_gather_(reshard_plan_t& plan, std::span<owner_type_* const> sources,
                          vram_apple_graph owner_type_::* graph, reshard_config_t const& config,
                          traversal_context_t& context) noexcept {
        std::size_t moved_nodes = 0, moved_bytes = 0, longest = longest_value_, span_bytes = span_bytes_;
        for (owner_type_* const source : sources) {
            vram_apple_graph const& given = source->*graph;
            if (&given == this || !given.size()) continue;
            if (given.device_.ordinal != device_.ordinal || given.connectivity() != connectivity() ||
                given.kernel_ != kernel_)
                return status_code_t::config_mismatch_k;
            if (status_t const kept = given.status(); failed(kept)) return kept;
            moved_nodes += given.size(), moved_bytes += given.tape_bytes_;
            longest = max_of(longest, given.longest_value_), span_bytes = max_of(span_bytes, given.span_bytes_);
        }
        if (failed(plan.targets_.reserve_exactly(1))) return status_code_t::out_of_memory_k;
        std::size_t const base = size();
        if (moved_nodes > ceiling_nodes_ - base) return status_code_t::overflow_k;
        if (status_t const closed = close_(); failed(closed)) return closed;
        plan.targets_.push_back_reserved(this);
        plan.closed_targets_ = 1;
        // Sources close one by one, and every one closed reopens once the copies end.
        std::size_t closed_sources = 0;
        auto const reopen_sources = [&]() noexcept {
            for (std::size_t source = 0; source != closed_sources; ++source)
                if (&(sources[source]->*graph) != this) (sources[source]->*graph).reopen_();
        };
        for (; closed_sources != sources.size(); ++closed_sources) {
            vram_apple_graph& given = sources[closed_sources]->*graph;
            if (&given == this) continue;
            if (status_t const closed = given.close_(); failed(closed)) {
                reopen_sources();
                return closed;
            }
        }
        std::size_t const entries = base + moved_nodes;
        round_shape_t const shape = round_shape_(
            entries, entries ? max_of<std::size_t>(divide_round_up(tape_bytes_ + moved_bytes, entries), 1) : 1,
            config.refine.cache_bytes);
        span_bytes_ = span_bytes;
        round_layout_t const layout = round_layout_(entries, shape);
        status_t packed = carve_round_(context, layout.bytes);
        if (succeeded(packed)) packed = grow_nodes_(entries);
        if (succeeded(packed)) packed = map_bytes_(tape_bytes_ + moved_bytes + tape_margin_k);
        if (failed(packed)) {
            reopen_sources();
            return packed;
        }

        // The closed claim words make this call every column's and every map's one writer.
        std::size_t const row_ids = connectivity();
        std::size_t into = base, tape_bytes = tape_bytes_;
        u64_t* const starts = value_starts_();
        for (owner_type_* const source : sources) {
            vram_apple_graph& given = source->*graph;
            std::size_t const count = given.size();
            if (&given == this || !count) continue;
            u64_t const* const given_starts = given.value_starts_();
            std::memcpy(values_tape_.host + tape_bytes, given.values_tape_.host, given.tape_bytes_);
            for (std::size_t node = 0; node != count; ++node) {
                starts[into + node + 1] = tape_bytes + given_starts[node + 1];
                rebase_row_(given.neighbor_rows_() + node * row_ids, neighbor_rows_() + (into + node) * row_ids,
                            row_ids, count, into, into + node);
                keyed_.assign(static_cast<node_id_t>(into + node), given.keyed_[static_cast<node_id_t>(node)]);
            }
            into += count, tape_bytes += given.tape_bytes_;
            given.forget_();
        }
        reopen_sources();
        tape_bytes_ = tape_bytes, longest_value_ = longest;
        publish_keys_({static_cast<node_id_t>(base), static_cast<node_id_t>(entries)});
        publish_(entries);
        plan.rounds_.shape_ = shape;
        plan.round_layout_ = layout, plan.round_columns_ = context.round_.host;
        plan.rounds_.bind_(*this, layout, context.round_.host);
        plan.refined_targets_ = 0;
        return status_code_t::success_k;
    }

    /** Copies the row @p from of @p row_ids IDs into @p into: its IDs below @p count shifted by
     *  @p shift and packed to the front, the rest dropped, and a row keeping none pointing at the
     *  node @p self. */
    static void rebase_row_(node_id_t const* from, node_id_t* into, std::size_t row_ids, std::size_t count,
                            std::size_t shift, std::size_t self) noexcept {
        std::size_t kept = 0;
        for (std::size_t slot = 0; slot != row_ids; ++slot)
            if (from[slot] < count) into[kept++] = static_cast<node_id_t>(from[slot] + shift);
        for (std::size_t slot = kept; slot != row_ids; ++slot) into[slot] = (std::numeric_limits<node_id_t>::max)();
        if (!kept) into[0] = static_cast<node_id_t>(self);
    }

    /** Binds @p plan to a cut of this graph into @p targets: vets and closes this graph and every
     *  target, reserves and grows every target for a full part and maps its tape for the most its
     *  part can take, sizes the round buffer over a full part, then seeds the partition. Nothing
     *  moves before every target answered. */
    template <typename owner_type_>
    status_t open_cut_(reshard_plan_t& plan, std::span<owner_type_* const> targets,
                       vram_apple_graph owner_type_::* graph, reshard_config_t const& config,
                       traversal_context_t& context) noexcept {
        std::size_t const parts = targets.size();
        split_t& split = plan.split_;
        if (failed(plan.targets_.reserve_exactly(parts)) || failed(split.parts.resize(parts)))
            return status_code_t::out_of_memory_k;
        for (owner_type_* const listed : targets) {
            vram_apple_graph const& part = listed->*graph;
            if (&part == this || part.device_.ordinal != device_.ordinal || part.kernel_ != kernel_ ||
                (part.connectivity() && part.connectivity() != connectivity()) || part.size())
                return status_code_t::config_mismatch_k;
        }
        if (status_t const closed = close_(); failed(closed)) return closed;
        split.source = this;
        for (owner_type_* const listed : targets) plan.targets_.push_back_reserved(&(listed->*graph));
        split.entries = size();
        split.part_capacity = divide_round_up<std::size_t>(split.entries, parts);
        if (parts * split.part_capacity >= (std::numeric_limits<u32_t>::max)()) return status_code_t::overflow_k;
        for (vram_apple_graph* const part : plan.targets_) {
            if (status_t const closed = part->close_(); failed(closed)) return closed;
            ++plan.closed_targets_;
            if (part->size()) return status_code_t::config_mismatch_k;
            if (status_t const kept = part->status(); failed(kept)) return kept;
        }
        // Every target holds a full part's nodes, sentinel-filled, and the most value bytes they
        // can take, before the partition commits.
        std::size_t const part_bytes = min_of(tape_bytes_, split.part_capacity * longest_value_);
        for (vram_apple_graph* const part : plan.targets_) {
            if (status_t const reserved = part->reserve({split.part_capacity,
                                                         static_cast<connectivity_t>(connectivity()), part_bytes})
                                              .status;
                failed(reserved))
                return reserved;
            if (status_t const grown = part->grow_nodes_(split.part_capacity); failed(grown)) return grown;
        }
        round_shape_t const shape = round_shape_(split.part_capacity, mean_value_bytes_(), config.refine.cache_bytes);
        round_layout_t const layout = round_layout_(split.part_capacity, shape);
        if (status_t const carved = carve_round_(context, layout.bytes); failed(carved)) return carved;
        plan.rounds_.shape_ = shape;
        plan.round_layout_ = layout, plan.round_columns_ = context.round_.host;
        // An empty source leaves every target empty, so no round binds.
        if (!split.entries) {
            split.source = nullptr;
            forget_();
            reopen_();
            plan.refined_targets_ = parts;
            return status_code_t::success_k;
        }
        metal_context_t* const device = this->context();
        if (!device) return status_code_t::device_k;
        split.device_bytes = metal_allocate(*device, split_layout_(split.entries, parts).device_bytes);
        if (!split.device_bytes) return status_code_t::out_of_memory_k;
        seed_partition_(split, parts, config.seed);
        plan.moves_ = phase_t::partition_k;
        return status_code_t::success_k;
    }

    /** Where a cut's shared buffer keeps each array of @p nodes nodes into @p parts parts, in
     *  bytes, as dense Apple lays it out: each node's part, which the host turns into its slot, the
     *  round's proposals, the two frontiers, one bit per node each, then per part its size and its
     *  round count, and the nodes the last two rounds placed, by the round number's parity. */
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

    /** Starts @p split 's partition into @p parts parts on the host, the buffer still idle: every
     *  node unplaced and unproposed, the bitsets and counters empty, and the roots @p seed picks
     *  placed alone into the frontier of round one. */
    static void seed_partition_(split_t& split, std::size_t parts, u64_t seed) noexcept {
        std::size_t const entries = split.entries;
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

    /** Commits @p split 's growth rounds into @p parts parts from its next one up to twice it on
     *  @p context 's stream, four passes each, so the batches double until one ends past
     *  convergence, as dense Apple's; a round queued past it leaves at entry. */
    status_t queue_rounds_(split_t& split, std::size_t parts, traversal_context_t& context) noexcept {
        metal_graph_arguments_t arguments = graph_arguments_();
        arguments.end_id = static_cast<u32_t>(split.entries);
        arguments.part_capacity = static_cast<u32_t>(split.part_capacity);
        split_layout_t const layout = split_layout_(split.entries, parts);
        u32_t const first_round = split.next_round;
        for (u32_t round = first_round; round != 2 * first_round; ++round) {
            arguments.round = round;
            bool const odd = round & 1u;
            for (vram_kernel_t const pass : {vram_kernel_t::partition_propose_k, vram_kernel_t::partition_count_k,
                                             vram_kernel_t::partition_accept_k, vram_kernel_t::partition_fill_k}) {
                metal_launch_t launch {context.api(), context.stream(), graph_pipeline_(pass).pipeline};
                if (!launch) return status_code_t::device_k;
                launch.bind_bytes(&arguments, sizeof(arguments), metal_graph_arguments_buffer_k);
                launch.bind(neighbors_matrix_, metal_graph_neighbors_buffer_k);
                launch.bind(split.device_bytes, metal_node_parts_buffer_k, layout.node_parts);
                launch.bind(split.device_bytes, metal_proposals_buffer_k, layout.proposals);
                launch.bind(split.device_bytes, metal_frontier_buffer_k, odd ? layout.frontier : layout.next_frontier);
                launch.bind(split.device_bytes, metal_next_frontier_buffer_k,
                            odd ? layout.next_frontier : layout.frontier);
                launch.bind(split.device_bytes, metal_part_sizes_buffer_k, layout.part_sizes);
                launch.bind(split.device_bytes, metal_round_counts_buffer_k, layout.round_counts);
                launch.bind(split.device_bytes, metal_placed_buffer_k, layout.placed);
                std::size_t const groups =
                    pass == vram_kernel_t::partition_fill_k
                        ? parts
                        : std::size_t {device_.multiprocessor_count} *
                              max_of<u32_t>(graph_pipeline_(pass).max_threads / metal_threadgroup_threads_k, 1u);
                if (status_t const submitted = context.submit_(launch, groups, metal_threadgroup_threads_k);
                    failed(submitted))
                    return submitted;
            }
        }
        split.next_round = 2 * first_round;
        return status_code_t::success_k;
    }

    /** Whether the last growth round of @p split placed anything, so more rounds follow; read once
     *  its batch finished. */
    static bool placing_(split_t const& split, std::size_t parts) noexcept {
        split_layout_t const layout = split_layout_(split.entries, parts);
        u32_t const* const placed = reinterpret_cast<u32_t const*>(split.device_bytes.host + layout.placed);
        return placed[(split.next_round - 1) & 1u] != 0;
    }

    /** Places every node no growth round reached on the host, in ID order, each entering the part
     *  smallest at its turn, the lowest-indexed among equals, as the CUDA pass does on a block. */
    static void place_islands_(split_t& split, std::size_t parts) noexcept {
        split_layout_t const layout = split_layout_(split.entries, parts);
        u32_t* const node_parts = reinterpret_cast<u32_t*>(split.device_bytes.host + layout.node_parts);
        u32_t* const part_sizes = reinterpret_cast<u32_t*>(split.device_bytes.host + layout.part_sizes);
        for (std::size_t node = 0; node != split.entries; ++node) {
            if (node_parts[node] != (std::numeric_limits<u32_t>::max)()) continue;
            std::size_t smallest = 0;
            for (std::size_t part = 1; part != parts; ++part)
                if (part_sizes[part] < part_sizes[smallest]) smallest = part;
            node_parts[node] = static_cast<u32_t>(smallest);
            ++part_sizes[smallest];
        }
    }

    /** Moves every node of @p split into its target at the slot it was assigned, on the host over
     *  unified memory, in ID order: its value appended to its target's tape with its start, and its
     *  neighbors with the links inside its part rebased through their own slots and packed to the
     *  front, the links across dropped, and a node nothing survived in pointing at itself. */
    void scatter_(split_t& split, std::span<vram_apple_graph* const> targets) noexcept {
        u32_t const* const slots = reinterpret_cast<u32_t const*>(split.device_bytes.host);
        std::size_t const capacity = split.part_capacity, row_ids = connectivity();
        u64_t const* const starts = value_starts_();
        for (std::size_t node = 0; node != split.entries; ++node) {
            std::size_t const part = slots[node] / capacity, position = slots[node] % capacity;
            vram_apple_graph& target = *targets[part];
            split_part_t& filled = split.parts[part];
            std::size_t const bytes = starts[node + 1] - starts[node];
            std::memcpy(target.values_tape_.host + filled.bytes, values_tape_.host + starts[node], bytes);
            filled.bytes += bytes, filled.longest = max_of(filled.longest, bytes);
            u64_t* const target_starts = target.value_starts_();
            target_starts[position + 1] = filled.bytes;
            node_id_t const* const from = neighbor_rows_() + node * row_ids;
            node_id_t* const into = target.neighbor_rows_() + position * row_ids;
            std::size_t kept = 0;
            for (std::size_t slot = 0; slot != row_ids; ++slot) {
                node_id_t const id = from[slot];
                if (id < split.entries && id != node && slots[id] / capacity == part)
                    into[kept++] = static_cast<node_id_t>(slots[id] % capacity);
            }
            for (std::size_t slot = kept; slot != row_ids; ++slot) into[slot] = (std::numeric_limits<node_id_t>::max)();
            if (!kept) into[0] = static_cast<node_id_t>(position);
        }
    }

    /** The storage backend and the pinned device: its ordinal from the allocator at construction,
     *  the remaining specs at the first @c reserve. */
    metal_shared_allocator_t allocator_;
    metal_device_specs_t device_;

    /** The metric the kits score under, fixed at construction. */
    nk_kernel_kind_t kernel_;

    /** Neighbors per node, frozen by the first @c reserve; zero until then. */
    std::size_t connectivity_ = 0;

    /** The values tape, the offsets column and the neighbors matrix, each one shared buffer the
     *  first @c reserve allocates at the ceiling, never moved after. */
    metal_buffer_t values_tape_, offsets_column_, neighbors_matrix_;

    /** Nodes and tape bytes mapped so far, raised by reserves and claims alike. */
    std::size_t mapped_nodes_ = 0, mapped_bytes_ = 0;

    /** Held by the one batch claiming, growing and linking at a time, until its drain. */
    claim_word_t claims_;

    /** Nodes the columns span, the most any claim reaches; fixed by the first @c reserve. */
    std::size_t ceiling_nodes_ = max_node_count_k;

    /** The value bytes every published batch holds in the tape. */
    std::size_t tape_bytes_ = 0;

    /** The longest stored node and the widest span a stored node prepares into, which every later
     *  batch's kit rung and spans cover. */
    std::size_t longest_value_ = 0;
    std::size_t span_bytes_ = 0;

    /** Mapped nodes whose neighbors a grow sentinel-filled, read and advanced by the claim's
     *  holder. */
    std::size_t filled_nodes_ = 0;

    /** Nodes strictly below it are published and visible, turned once per batch after its drain. */
    std::size_t last_visible_id_ = 0;

    /** The first post-claim device failure. */
    status_t status_ = status_code_t::success_k;

    /** The face's kits: their hooks, their sources and their parameters. */
    kits_type_ kits_;

    /** Every kits' kernel's pipeline per kit rung and candidate rung, and every graph kernel's,
     *  owned by the device's registry. */
    metal_pipeline_state_t kits_pipelines_[kits_type_::rungs_k][static_cast<std::size_t>(vram_kernel_t::count_k)]
                                          [gpu_rungs_per_kernel_k] = {};
    metal_pipeline_state_t graph_pipelines_[static_cast<std::size_t>(vram_kernel_t::count_k)] = {};

    /** Every node's key, assigned by its claim's holder, and the key map that holder files them in
     *  just before the watermark moves. */
    keyed_t keyed_;
};

} // namespace unum::usearch::basic
