/**
 *  @file include/usearch/basic/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over variable-length values in a single uniform latency memory space on
 *      NVIDIA GPUs.
 *
 *  The host graph the NVIDIA basic faces share, compiled by any C++ compiler: it fills plain
 *  argument records and launches each modality's kernels by name, as the dense host face does,
 *  through the dense traversal contexts. A modality's face owns one over its kits, which name their
 *  rungs and read the kit for the rung a launch's longest value needs, the lanes that rung scores a
 *  node with and the span one value's preparation takes, in its vendor metric, which only its
 *  NVIDIA unit compiles, so dispatch never names a kit.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <bit>              // `std::bit_ceil`, `std::bit_width`, `std::has_single_bit`
#include <concepts>         // `std::same_as` - the sharded index takes 32-bit shards
#include <functional>       // `std::hash` - the key map's default hash
#include <initializer_list> // the two walks of every wave
#include <limits>           // `std::numeric_limits` - the node-ID type's sentinel
#include <memory>           // `std::unique_ptr` - an audit's pinned counters
#include <span>             // `std::span` - the caller's node offsets, a snapshot's columns
#include <utility>          // `std::exchange` - a moved-from plan forgets its index

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the published words

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the records carry the kernel kind
#include <numkong/tensor.hpp>     // `nk::matrix_span` - the outputs of a search

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"        // `growing_buffer` - a bounce and its host offsets
#include "usearch/utilities/ram_device.hpp"         // `forkunion_allocator_t` - the host keys' placement
#include "usearch/utilities/ram_tables.hpp"         // `keyed_nodes` - every node's key and every key's nodes
#include "usearch/utilities/safegraphs_format.hpp"  // `safegraphs_saved_columns` - a load's saved columns
#include "usearch/utilities/vram_device.hpp"        // `vram_kernel_t` - the kernels a rung names
#include "usearch/utilities/vram_nvidia_device.hpp" // `cuda_traversal_context` & the rungs
#include "usearch/utilities/vram_walks.hpp"         // `gpu_arena_shape_t`, `gpu_candidate_slots`, `vram_bitset_t`
#include "usearch/basic/ram_index.hpp"              // `columns`, `columns_extents_t`, `refine_report_t`

namespace unum::usearch::basic {
using namespace unum::usearch::utilities;

template <typename index_type_, typename, typename, typename, typename>
    requires std::same_as<typename index_type_::node_id_t, u32_t>
struct vram_nvidia_indexes;

#pragma region Kernel Records

/** The ragged storage a kernel reads, mirroring the shared ragged view field by field. */
template <typename node_id_type_>
struct vram_ragged_storage {
    byte_t const* values = nullptr;
    u64_t* offsets = nullptr;
    node_id_type_* neighbors = nullptr;
    u32_t connectivity = 0;
    nk_kernel_kind_t kernel = nk_kernel_unknown_k;
    byte_t const* parameters = nullptr;
};

/** One insertion wave and its back-link passes, mirroring the shared build view. */
template <typename node_id_type_>
struct vram_build {
    vram_ragged_storage<node_id_type_> storage;
    u64_t const* staged_offsets = nullptr;
    u64_t tape_first = 0;
    node_id_type_ end_id = 0;
    node_id_type_ sampling_bound = 0;
    u32_t scratch_bytes_per_warp = 0;
    u32_t span_bytes = 0;
    node_id_type_ first_id = 0;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    u32_t expansion = 0;
    u32_t selection_nodes = 0;
    u32_t epoch_base = 0;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_nodes = nullptr;
    node_id_type_* link_heads = nullptr;
    node_id_type_* link_next = nullptr;
    node_id_type_* touched_ids = nullptr;
    u32_t* touched_count = nullptr;
    vram_bitset_t changed {};
    u32_t const* incoming_offsets = nullptr;
    node_id_type_ const* back_links = nullptr;
    u32_t tape_size = 0;
    node_id_type_ walked_first = 0;
    node_id_type_ walked_bound = 0;
};

/** One walking launch and where it writes its outputs, mirroring the shared search view. */
template <typename node_id_type_>
struct vram_search {
    vram_ragged_storage<node_id_type_> storage;
    byte_t const* queries = nullptr;
    u64_t const* query_offsets = nullptr;
    u32_t queries_count = 0;
    node_id_type_ sampling_bound = 0;
    u32_t scratch_bytes_per_warp = 0;
    u32_t expansion = 0;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    u32_t epoch_base = 0;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_queries = nullptr;
    u32_t* live_walks = nullptr;
    u32_t concurrent_walks = 0;
    u32_t filling_nodes = 0;
    node_id_type_* result_ids = nullptr;
    distance_t* result_distances = nullptr;
    u32_t count = 0;
};

/** The record the two load kernels read: one run of a load's saved values at @c saved, their
 *  element prefix sums at @c saved_offsets, their planned tape starts at @c stored_offsets, one
 *  past each of @c count nodes, the @c tape they go into, and the plan's @c span_bytes word. */
template <typename node_id_type_>
struct vram_load {
    byte_t const* saved = nullptr;
    u64_t const* saved_offsets = nullptr;
    u64_t const* stored_offsets = nullptr;
    byte_t* tape = nullptr;
    u64_t* span_bytes = nullptr;
    u32_t count = 0;
};

/** The one argument every NVIDIA basic kernel takes, each reading the record its kernel names. */
template <typename node_id_type_>
struct vram_arguments {
    vram_build<node_id_type_> build;
    vram_search<node_id_type_> search;
    vram_load<node_id_type_> load;
};

/** One launch over the neighbors matrix alone, reading no value: the rebase of nodes a donor gave
 *  up, the reachability audit and the reconnection healing it, a sweep tile's count and scatter,
 *  and a split's partition and the scatter moving its nodes, whose values pass as opaque bytes. */
template <typename node_id_type_>
struct vram_graph {
    node_id_type_* neighbors = nullptr;
    u32_t connectivity = 0;
    u32_t nodes_count = 0;
    node_id_type_ first_id = 0;
    node_id_type_ source_first = 0;
    node_id_type_ source_bound = 0;
    node_id_type_ shift = 0;
    vram_bitset_t reached {};
    vram_bitset_t frontier {};
    vram_bitset_t next_frontier {};
    u32_t* counter = nullptr;
    u32_t const* previous_counter = nullptr;
    u32_t* levels = nullptr;
    u32_t seed_count = 0;
    u32_t* island_ids = nullptr;
    u32_t islands_count = 0;
    u32_t* in_links = nullptr;
    u32_t* reconnected = nullptr;

    /** The sweep tile a count, an offset or a scatter launch works on. */
    vram_sweep_t sweep {};

    /** A split's part of each node, then its slot once the parts are ordered, the round's
     *  proposals, and per part its size, its round count and its root. */
    u32_t* node_parts = nullptr;
    u32_t* proposals = nullptr;
    u32_t* part_sizes = nullptr;
    u32_t* round_counts = nullptr;
    u32_t const* roots = nullptr;
    u32_t parts_count = 0;
    u32_t part_capacity = 0;
    vram_bitset_t picked {};

    /** The receivers' neighbors matrices, tapes and offsets columns, by part, and the source's tape
     *  and offsets. */
    u32_t* const* part_neighbors = nullptr;
    byte_t* const* part_values = nullptr;
    u64_t const* const* part_offsets = nullptr;
    byte_t const* values = nullptr;
    u64_t const* value_offsets = nullptr;

    /** A gather's listed nodes, and the tape their values go to, back to back, with its offsets,
     *  one past each listed node from the base at its first. */
    node_id_type_ const* listed = nullptr;
    u32_t listed_count = 0;
    byte_t* gathered_values = nullptr;
    u64_t* gathered_offsets = nullptr;

    /** A load's saved neighbors: fixed rows of @c degree IDs, or a run of IDs @c saved_offsets
     *  index; and the word an ID past the nodes or an overfull row sets. */
    node_id_type_ const* saved_neighbors = nullptr;
    u64_t const* saved_offsets = nullptr;
    u32_t degree = 0;
    u64_t* malformed = nullptr;
};

/** Blocks of @p kernel on its @p slots rung resident per multiprocessor - the register budget its
 *  launch bounds set and its grid's share of every multiprocessor: two for the builders and the
 *  back-link riding their grid and arena, the most whose share of the register file holds the walk
 *  and its selection without spilling, since three spill the Levenshtein and shape kits' walks at
 *  every rung; a walk's by its rung; eight for the passes striding the neighbors matrix and a
 *  split's nodes, oversubscribed so enough nodes stay in flight; and two for the rest, whose grids
 *  follow their work. */
constexpr u32_t cuda_resident_blocks(vram_kernel_t kernel, u32_t slots) noexcept {
    switch (kernel) {
    case vram_kernel_t::search_k: return slots <= 8 ? 4 : 3;
    case vram_kernel_t::rebase_links_k:
    case vram_kernel_t::reachability_sweep_k:
    case vram_kernel_t::reached_in_links_k:
    case vram_kernel_t::count_back_links_k:
    case vram_kernel_t::scatter_back_links_k:
    case vram_kernel_t::partition_propose_k:
    case vram_kernel_t::partition_count_k:
    case vram_kernel_t::partition_accept_k:
    case vram_kernel_t::partition_picked_k:
    case vram_kernel_t::split_links_k:
    case vram_kernel_t::split_tape_k:
    case vram_kernel_t::widen_links_k: return 8;
    default: return 2;
    }
}

/** One instantiated kernel: the kernel and its register rung, zero where it keeps no ladder or
 *  walks the arena frontier. */
struct cuda_rung_t {
    vram_kernel_t kernel;
    u32_t slots;
};

/** Every instantiation each rung of a modality's kits exports. */
inline constexpr cuda_rung_t cuda_rungs[] = {
    {vram_kernel_t::insert_k, 0},  {vram_kernel_t::insert_k, 2},           {vram_kernel_t::insert_k, 4},
    {vram_kernel_t::insert_k, 8},  {vram_kernel_t::insert_k, 16},          {vram_kernel_t::refine_k, 0},
    {vram_kernel_t::refine_k, 2},  {vram_kernel_t::refine_k, 4},           {vram_kernel_t::refine_k, 8},
    {vram_kernel_t::refine_k, 16}, {vram_kernel_t::back_link_k, 0},        {vram_kernel_t::search_k, 0},
    {vram_kernel_t::search_k, 2},  {vram_kernel_t::search_k, 4},           {vram_kernel_t::search_k, 8},
    {vram_kernel_t::search_k, 16}, {vram_kernel_t::merge_back_links_k, 0},
};

/** The kernels a load queues, which each kit exports once, on its first rung. */
inline constexpr vram_kernel_t cuda_load_kernels[] = {vram_kernel_t::load_values_k, vram_kernel_t::size_spans_k};

/** Every kernel of the unit all NVIDIA basic faces share, over the neighbors matrix alone, so one
 *  instantiation per node-ID type serves every modality. */
inline constexpr vram_kernel_t cuda_graph_kernels[] = {
    vram_kernel_t::rebase_links_k,         vram_kernel_t::mark_landmarks_k,    vram_kernel_t::reachability_sweep_k,
    vram_kernel_t::reachability_islands_k, vram_kernel_t::reached_in_links_k,  vram_kernel_t::reconnect_island_k,
    vram_kernel_t::partition_roots_k,      vram_kernel_t::partition_propose_k, vram_kernel_t::partition_count_k,
    vram_kernel_t::partition_accept_k,     vram_kernel_t::partition_fill_k,    vram_kernel_t::partition_islands_k,
    vram_kernel_t::partition_picked_k,     vram_kernel_t::split_links_k,       vram_kernel_t::split_tape_k,
    vram_kernel_t::gather_tape_k,          vram_kernel_t::count_back_links_k,  vram_kernel_t::offset_back_links_k,
    vram_kernel_t::scatter_back_links_k,   vram_kernel_t::widen_links_k,
};

/** The fatbin of @ref cuda_graph_kernels over 32-bit node IDs, defined beside them in
 *  @c src/basic/vram_variants_nvidia.cu, which every face loads at its first @c reserve. */
struct cuda_graph_fatbin_t {
    static bytes_view_t bytes() noexcept;
};

/** Spells every kernel of @ref cuda_graph_fatbin_t for a @c cuda_kernel_table_t, in
 *  @ref vram_kernel_t order, through the graph's own spelling of its node-ID type; none is
 *  required, since the fatbin covers 32-bit IDs alone, and a verb over any other answers
 *  @c unsupported_k. */
struct cuda_graph_kernel_speller_t {
    using symbol_t = vram_kernel_symbol_t;

    /** Spells one shared kernel over the graph's node-ID type. */
    void (*spell_kernel)(vram_kernel_symbol_t&, vram_kernel_t) noexcept;

    static constexpr std::size_t count() noexcept { return static_cast<std::size_t>(vram_kernel_t::count_k); }
    static constexpr std::size_t index(vram_kernel_t kernel) noexcept { return static_cast<std::size_t>(kernel); }
    void spell(symbol_t& symbol, std::size_t position) const noexcept {
        spell_kernel(symbol, static_cast<vram_kernel_t>(position));
    }
    static constexpr bool required(std::size_t) noexcept { return false; }
};

/** Spells every kernel of a modality's kits fatbin for a @c cuda_kernel_table_t: by kit rung, then
 *  in @ref vram_kernel_t order, then by register rung; every walk of @ref cuda_rungs is required
 *  on every kit rung. */
struct cuda_kits_kernel_speller_t {
    using symbol_t = vram_kernel_symbol_t;

    /** The kits' rungs, and their spelling of one kernel instantiated on one of them. */
    std::size_t rungs;
    void (*spell_kernel)(vram_kernel_symbol_t&, std::size_t, cuda_rung_t) noexcept;

    /** Positions one kit rung takes. */
    inline static constexpr std::size_t positions_per_kit_rung_k = static_cast<std::size_t>(vram_kernel_t::count_k) *
                                                                   gpu_rungs_per_kernel_k;

    std::size_t count() const noexcept { return rungs * positions_per_kit_rung_k; }

    /** The position of @p kernel on kit rung @p rung and register rung @p slots. */
    static constexpr std::size_t index(std::size_t rung, vram_kernel_t kernel, u32_t slots) noexcept {
        return rung * positions_per_kit_rung_k + static_cast<std::size_t>(kernel) * gpu_rungs_per_kernel_k +
               gpu_rung_index(slots);
    }

    /** The slots of register rung @p rung, inverting @ref gpu_rung_index: none for the arena
     *  frontier, then 2, 4, 8 and 16. */
    static constexpr u32_t slots_of(std::size_t rung) noexcept { return rung ? 1u << rung : 0u; }

    void spell(symbol_t& symbol, std::size_t position) const noexcept {
        std::size_t const within = position % positions_per_kit_rung_k;
        spell_kernel(symbol, position / positions_per_kit_rung_k,
                     cuda_rung_t {static_cast<vram_kernel_t>(within / gpu_rungs_per_kernel_k),
                                  slots_of(within % gpu_rungs_per_kernel_k)});
    }

    /** Whether the kits instantiate the kernel at @p position on its kit rung. */
    static constexpr bool required(std::size_t position) noexcept {
        for (cuda_rung_t const rung : cuda_rungs)
            if (index(0, rung.kernel, rung.slots) == position % positions_per_kit_rung_k) return true;
        for (vram_kernel_t const kernel : cuda_load_kernels)
            if (index(0, kernel, 0) == position) return true;
        return false;
    }
};

/** One run of a load as the kits' hooks queue its work: the run's saved values on the device, where
 *  its first stored value goes in the tape, and the record the kits' load kernels read. */
template <typename node_id_type_>
struct vram_load_step {
    vram_arguments<node_id_type_> arguments {};
    bytes_view_t saved;
    byte_t* stored = nullptr;
    cuda_api_t const* api = nullptr;
    cuda_kernel_table_t const* kits = nullptr;
    cuda_stream_handle_t stream = nullptr;
    u32_t multiprocessors = 0;

    /** Copies the run's saved values into the tape as they are, for kits storing them so. */
    status_t copy_values() const noexcept {
        if (saved.empty()) return status_t::success_k;
        return cuda_copy_async(*api, stored, saved.data(), saved.size(), stream) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Launches the kits' @p kernel over the run's nodes on the stream, a grid-striding pass. */
    status_t launch_kits(vram_kernel_t kernel) const noexcept {
        return cuda_launch(*api, kits->functions[cuda_kits_kernel_speller_t::index(0, kernel, 0)],
                           multiprocessors * cuda_resident_blocks(kernel, 0), vram_warps_per_block_k * 32, 0, stream,
                           cuda_overlap_t::serialized_k, arguments);
    }
};

#pragma endregion Kernel Records

/**
 *  @brief Similarity Search over variable-length values in a single uniform latency memory space on
 *      NVIDIA GPUs.
 *
 *  Speaks the dense VRAM index's verbs over values of any length: a batch's IDs are one contiguous
 *  range, and its nodes turn visible to searches a whole batch at a time. Values sit back to back
 *  in one tape beside a column of value starts, as the caller's own tape lays them out, and the
 *  neighbors matrix holds @c connectivity IDs per node; all three reserve their address range once
 *  and map on demand. The kit rung a launch scores on is the narrowest whose nodes cover the
 *  longest one the launch meets, and the spans its walks prepare nodes into are as wide as the
 *  widest of them.
 *
 *  A batch inserts in admission waves, each no wider than a quarter of the nodes already below it.
 *  Every wave publishes forward nodes alone, threading each pick onto its target's chain, and a
 *  back-link launch then merges every chain into the node it names, so a node never waits on a lock
 *  across a distance. Its nodes then walk again over the linked graph, where they reach one
 *  another, re-select their own neighbor rows, and back-link those picks too. The merges rewrite
 *  live neighbor rows, so one batch holds the claim word from its claim until its drain, and
 *  another answers @c contended_k meanwhile, while searches on other streams keep running
 *  throughout.
 *
 *  A refinement re-selects nodes over the whole graph, each merging its picks into its row at once,
 *  then sweeps tiles of target nodes sized from its limit, merging the nodes linking to each one,
 *  and an absorb appends other indexes' nodes on the device before refining. Both, and the
 *  reachability audit, launch kernels that read the neighbors matrix alone, which one unit shares
 *  across every modality. Saves and loads cross the snapshot columns of the basic RAM index through
 *  a pinned bounce, values through the kits' codec. Plans close the claim word, or claim their
 *  nodes, for their whole lives.
 *
 *  A modality's face owns one as a member and forwards its verbs, so no modality derives from it.
 *  It stands alone as the RAM index does, under either node-ID width and any hashable key, which it
 *  keeps on the host beside every node and files under the key map while it holds the claim word,
 *  just before the watermark moves; only the sharded index the compiled library builds narrows it
 *  to 32-bit IDs, whose nodes a shard table locates in 8 bytes.
 *
 *  @tparam node_id_type_ The node IDs of the neighbors matrix, the walks and the matches, @c u32_t
 *      or @c u64_t; the all-ones ID is the sentinel. The shared kernels cover @c u32_t alone, so
 *      the refinement, the absorb and the audit answer @c unsupported_k over any other.
 *  @tparam kits_type_ The modality's kits, held as a member, whose @c rungs_k kit rungs its kernels
 *      cover, narrowest first, and whose members read them: @c rung_serving_ nodes of a length, or
 *      @c rungs_k past the widest, @c lanes_per_entry_ of a rung, the @c span_bytes_ one value
 *      prepares into on every rung alike, and the bound on it any loaded value of a length takes,
 *      the @c fatbin_ holding its kernels and the @c spell_kits_kernel_ naming each, and the
 *      @c parameters_ its kits read on the device, which @c upload_parameters_ places there; its
 *      codec, @c saved_bytes_ and @c save_value_ turning a stored node into the bytes the RAM codec
 *      saves, @c stored_bytes_ sizing a stored node from its saved length, and
 *      @c queue_load_values_ storing and sizing a load's values on the device, never the host.
 *  @tparam key_type_ The caller-visible identity of every node, trivially copyable.
 *  @tparam key_hash_type_ The key map's hash of a key.
 */
template <typename node_id_type_, typename kits_type_, typename key_type_ = uuid_t,
          typename key_hash_type_ = std::hash<key_type_>>
struct vram_ragged_graph {
    using traversal_context_t = cuda_traversal_context<cuda_arena_allocator_t>;

    /** The graph's and the kits' kernels every index on one device launches, which its executor
     *  stream loads at the first ask and keeps, one table per fatbin. */
    using kernel_table_t = cuda_vendor_t::kernel_table_t;
    using node_id_t = node_id_type_;
    using key_t = key_type_;
    using distance_t = distance_t;
    using candidates_count_t = u32_t;

    /** The node-ID space, less the sentinel. */
    inline static constexpr std::size_t max_node_count_k = std::size_t {std::numeric_limits<node_id_t>::max()} - 1u;

    /** The nodes the host keys span: the node-ID space, no further than the address space holds
     *  keys. */
    inline static constexpr std::size_t max_keyed_node_count_k = min_of<std::size_t>(
        max_node_count_k, (std::size_t {1} << address_bits_k) / sizeof(key_t));

    /** Every node's key on the host, and every key's nodes. */
    using keyed_t = keyed_nodes<key_t, node_id_t, chunk_schedule_spanning<max_keyed_node_count_k>,
                                max_keyed_node_count_k, forkunion_allocator_t, fu::standard_atomic_ref, key_hash_type_>;

    /** The strongest @c synchronization_t contract this design honors; nothing routes on it. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;

    /** Pins the index to the storage allocator's device, its kits scoring under @p kernel where
     *  they cover several metrics, and keeps its host keys and key map on @p keys_allocator; the
     *  first @c reserve freezes the rest. */
    explicit vram_ragged_graph(nk_kernel_kind_t kernel, cuda_virtual_memory_allocator<byte_t> const& allocator = {},
                               forkunion_allocator_t const& keys_allocator = {}) noexcept
        : device_ {allocator.device()}, kernel_ {kernel}, values_tape_(allocator), offsets_column_(allocator),
          neighbors_matrix_(allocator),
          grown_((cuda_set_device(allocator.api(), allocator.device()), cuda_event_t {allocator.api()})),
          keyed_(keys_allocator) {}
    vram_ragged_graph(vram_ragged_graph const&) = delete;
    vram_ragged_graph& operator=(vram_ragged_graph const&) = delete;

#pragma region Measures

    /** Nodes visible to searches: the watermark every walk samples below and filters neighbors
     *  by. */
    std::size_t size() const noexcept {
        return fu::standard_atomic_ref<std::size_t const>(last_visible_id_).load(std::memory_order_acquire);
    }

    /** The first failure that left claimed nodes unpublished, kept for the index's lifetime. */
    status_t status() const noexcept {
        return fu::standard_atomic_ref<status_t const>(status_).load(std::memory_order_acquire);
    }

    /** Neighbors per node, zero until the first @c reserve freezes it. */
    std::size_t connectivity() const noexcept {
        return fu::standard_atomic_ref<std::size_t const>(connectivity_).load(std::memory_order_acquire);
    }

    /** Nodes mapped across the offsets column and the neighbors matrix, the smaller of their
     *  counts. */
    std::size_t capacity() const noexcept {
        std::size_t const row_bytes = connectivity() * sizeof(node_id_t);
        std::size_t const offsets_count = offsets_column_.committed_bytes() / sizeof(u64_t);
        if (!row_bytes || !offsets_count) return 0;
        return min_of(offsets_count - 1, neighbors_matrix_.committed_bytes() / row_bytes);
    }

    /** Value bytes mapped in the tape, less the margin past the last. */
    std::size_t bytes_capacity() const noexcept {
        std::size_t const committed = values_tape_.committed_bytes();
        return committed > tape_margin_k ? committed - tape_margin_k : 0;
    }

    /** Bytes this index occupies on the device and the host, its keys and key map among them;
     *  traversal scratch is the executor's. */
    std::size_t memory_usage() const noexcept {
        return sizeof(vram_ragged_graph) + values_tape_.committed_bytes() + offsets_column_.committed_bytes() +
               neighbors_matrix_.committed_bytes() + keyed_.memory_usage();
    }

    /** The pinned device's identity and limits, queried once at the first @c reserve. */
    cuda_device_specs_t const& device_specs() const noexcept { return device_; }

    /** The driver table every call on this index goes through, the storage allocator's. */
    cuda_api_t const& api() const noexcept { return neighbors_matrix_.api(); }

    /** The metric the kits score under where they cover several, unknown otherwise. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Device-side storage views, the audit surface for tests. */
    vram_ragged_storage<node_id_t> storage() const noexcept {
        return {values_tape_.base(),
                reinterpret_cast<u64_t*>(offsets_column_.base()),
                reinterpret_cast<node_id_t*>(neighbors_matrix_.base()),
                static_cast<u32_t>(connectivity()),
                kernel_,
                kits_.parameters_()};
    }

    /** The modality's kits, which a face configures before the first @c reserve freezes them. */
    kits_type_& kits() noexcept { return kits_; }

    /** Every node's key on the host, and every key's nodes; readers pass the watermark they
     *  acquired. */
    keyed_t const& keys() const noexcept { return keyed_; }

    /**
     *  The key of the visible @p node.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p node is not visible.
     */
    expected<key_t> key_of(node_id_t node) const noexcept {
        if (static_cast<std::size_t>(node) >= size()) return {{}, status_t::config_mismatch_k};
        return {keyed_[node], status_t::success_k};
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
     *      into the host @p rows, raw as stored with sentinels and self-loops left in, and waits
     *      for the device copy to finish.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p rows hold no whole number of rows, or the rows pass @c size.
     *  - @c device_error_k The device could not be selected, or the copy failed.
     */
    status_t neighbors(node_id_t first, std::span<node_id_t> rows) const noexcept {
        std::size_t const row_ids = connectivity();
        if (!row_ids || rows.size() % row_ids || std::size_t {first} + rows.size() / row_ids > size())
            return status_t::config_mismatch_k;
        if (failed(cuda_vendor_t::select(api(), device_.ordinal)) ||
            cuda_copy(api(), rows.data(), neighbors_matrix_.base() + std::size_t {first} * row_ids * sizeof(node_id_t),
                      rows.size_bytes()) != cuda_success_k)
            return status_t::device_error_k;
        return status_t::success_k;
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

        /** Nodes the index may ever host, clamped to the ID space and the device's memory; past it
         *  every claim answers @c overflow_k. */
        std::size_t ceiling = max_node_count_k;
    };

    /**
     *  Ensures the index can host at least @p config.capacity nodes of @p config.bytes value bytes
     *  in all, and answers the node capacity now in effect; the first success binds the device,
     *  loads the kernels and reserves the address ranges, later ones only map.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The face's kits take parameters, and none were set.
     *  - @c config_mismatch_k The connectivity is not a power of two, or differs from the frozen
     *    one.
     *  - @c overflow_k The capacity passes the ID space or the reserved range, or the value bytes
     *    pass the reserved tape.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (!std::has_single_bit(static_cast<u32_t>(config.connectivity))) return {0, status_t::config_mismatch_k};
        if (config.capacity > max_node_count_k) return {capacity(), status_t::overflow_k};
        std::size_t const frozen = connectivity();
        if (frozen && frozen != config.connectivity) return {capacity(), status_t::config_mismatch_k};
        if (!frozen) {
            if (cuda_set_device(api(), device_.ordinal) != cuda_success_k) return {0, status_t::device_error_k};
            auto const [specs, status] = cuda_device_specs_t::query(api(), device_.ordinal);
            if (failed(status)) return {0, status};
            device_ = specs;
            if (status_t const uploaded = kits_.upload_parameters_(api()); failed(uploaded)) return {0, uploaded};
            // No column maps more than the device's memory, whatever the first capacity.
            std::size_t const budget_bytes = device_.vram_bytes;
            std::size_t const row_bytes = config.connectivity * sizeof(node_id_t) + sizeof(u64_t);
            std::size_t const ceiling_nodes = min_of(min_of(config.ceiling, max_node_count_k),
                                                     max_of<std::size_t>(config.capacity, budget_bytes / row_bytes));
            // The offsets column holds one start past the last node.
            if (offsets_column_.reserve_range((ceiling_nodes + 1) * sizeof(u64_t)) != status_t::success_k ||
                neighbors_matrix_.reserve_range(ceiling_nodes * config.connectivity * sizeof(node_id_t)) !=
                    status_t::success_k ||
                values_tape_.reserve_range(max_of(config.bytes, budget_bytes) + tape_margin_k) != status_t::success_k)
                return {0, status_t::bad_alloc_k};
            ceiling_nodes_ = ceiling_nodes;
            fu::standard_atomic_ref<std::size_t>(connectivity_).store(config.connectivity, std::memory_order_release);
        }
        if (status_t const mapped = map_nodes_(config.capacity); failed(mapped)) return {capacity(), mapped};
        status_t const mapped_bytes = values_tape_.commit(config.bytes + tape_margin_k);
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
        node_id_t first_id = std::numeric_limits<node_id_t>::max();
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

    /** The bytes an executor's run must hold before a context carved over it can insert the nodes
     *  of @p tape, one aligned term per carve the verb makes; zero before @c reserve freezes the
     *  geometry. */
    std::size_t traversal_memory_usage(tape_view<byte_t const> tape, insert_config_t config = {}) const noexcept {
        if (!connectivity() || !device_.multiprocessor_count || tape.offsets.empty()) return 0;
        std::size_t const entries = tape.size();
        std::size_t longest = longest_value_;
        u32_t const span_bytes = static_cast<u32_t>(max_of(span_bytes_, span_bytes_of_(tape, longest)));
        std::size_t const rung = kits_.rung_serving_(longest);
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   cuda_resident_blocks(vram_kernel_t::insert_k, 0) * vram_warps_per_block_k;
        return aligned_(sizeof(vram_telemetry_block_t)) + 2 * aligned_(sizeof(u32_t)) +
               aligned_(slices * build_shape_(config.expansion_factor, span_bytes, rung)
                                     .arena_shape.template slice_bytes<node_id_t>()) +
               aligned_(build_staging_bytes_(entries, claimed_() + entries));
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
        node_id_t first_id = std::numeric_limits<node_id_t>::max();

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
         *  Launches the claimed nodes of @p tape under @p keys, one node a claimed ID, as the
         *  one-shot @ref vram_ragged_graph::insert describes.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent, or the index was never reserved.
         *  - @c config_mismatch_k The @p keys or @p tape hold other than the claimed nodes, or
         *    @p context rides another device or cannot walk.
         *  - @c unsupported_k A value past every rung of the face's kits.
         *  - @c overflow_k The nodes pass the reserved range or the host keys, or their values the
         *    reserved tape.
         */
        expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape,
                                         traversal_context_t& context, insert_config_t config = {}) noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            vram_ragged_graph& index = *index_;
            expected<insert_result_t> const launched = index.launch_(*this, keys, tape, context, config);
            if (succeeded(launched.status)) {
                index.publish_keys_({launched.value.first_id, static_cast<node_id_t>(launched.value.new_size)});
                index.publish_(launched.value.new_size);
            }
            return launched;
        }

      private:
        friend vram_ragged_graph;

        /** The index whose claim word holds the nodes; null once spent. */
        vram_ragged_graph* index_ = nullptr;
    };

    /**
     *  Claims @p nodes for one batch, exposing their first ID before any of them launches; refuses
     *  a kept failure, nodes past the reserved ceiling, and another batch in flight with
     *  @c contended_k.
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
            return {std::move(plan), status_t::success_k};
        }
        auto const [first_id, claim_status] = claim_(nodes);
        if (failed(claim_status)) return {std::move(plan), claim_status};
        plan.first_id = first_id;
        plan.entries = nodes;
        plan.index_ = this;
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  Inserts the nodes of @p tape under @p keys as the contiguous IDs
     *  @b [first_id,first_id+nodes), in admission waves riding the context's stream, then drains
     *  it, files the keys and turns the watermark over the range: @ref plan_insert, then its plan's
     *  @c insert. Another batch in flight answers @c contended_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index was never reserved.
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
        if (keys.size() != entries) return {{}, status_t::config_mismatch_k};
        if (!entries) return {{claimed_(), static_cast<node_id_t>(claimed_())}, status_t::success_k};
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
     *  a shard it retires with its claims closed for good, so a reader still walking it reads nodes
     *  nothing frees, and every walk below the watermark finds none. */
    void retire_() noexcept {
        fu::standard_atomic_ref<std::size_t>(last_visible_id_).store(0, std::memory_order_release);
    }

    /**
     *  @brief The kits' kernels on @p context 's stream, loaded there at the first ask and kept
     *      until the stream dies; every verb walking the kits resolves them once per call, and the
     *      walks outside the rungs look their symbols up in the table's module.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The @p context rides no executor stream.
     *  - @c device_error_k The fatbin would not load, or lacks a kit's kernel.
     */
    static expected<kernel_table_t const*> kits_kernels_(traversal_context_t const& context) noexcept {
        return context.table_of(kits_type_::fatbin_(),
                                cuda_kits_kernel_speller_t {kits_type_::rungs_k, &kits_type_::spell_kits_kernel_});
    }

    /** The kernel the kits' fatbin of @p kits exports as @p symbol, or null when it has none. */
    static cuda_function_handle_t kits_symbol_function_(kernel_table_t const& kits, char const* symbol) noexcept {
        return kits.module.function(symbol);
    }

    /**
     *  @brief Copies on @p context 's stream the values of the @p listed nodes, device-resident,
     *      back to back into @p outputs, their offsets into @p output_offsets.
     *
     *  Each offset is one past its listed node from the base the first one holds, the end of the
     *  previous gather: a lookup straight into device outputs.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p output_offsets are not one longer than @p listed.
     *  - @c unsupported_k The shared kernels lack this node-ID type.
     */
    status_t launch_gather_tape_(std::span<node_id_t const> listed, byte_t* outputs, std::span<u64_t> output_offsets,
                                 traversal_context_t const& context) noexcept {
        if (output_offsets.size() != listed.size() + 1) return status_t::config_mismatch_k;
        auto const [graph_kernels, kernels_status] = graph_kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        if (listed.empty()) return status_t::success_k;
        vram_graph<node_id_t> graph;
        graph.values = values_tape_.base();
        graph.value_offsets = reinterpret_cast<u64_t const*>(offsets_column_.base());
        graph.listed = listed.data();
        graph.listed_count = static_cast<u32_t>(listed.size());
        graph.gathered_values = outputs;
        graph.gathered_offsets = output_offsets.data();
        return launch_graph_(*graph_kernels, vram_kernel_t::gather_tape_k, 1, graph, context.stream());
    }

  private:

    /** A sharded index walks its shards below one snapshot of their watermarks, and publishes a
     *  batch on every shard inside one commit. */
    template <typename index_type_, typename, typename, typename, typename>
        requires std::same_as<typename index_type_::node_id_t, u32_t>
    friend struct vram_nvidia_indexes;

    /** Assigns @p keys and launches and drains the nodes @p plan claimed, spending it unless the
     *  arguments mismatch, and leaves them for @c publish_keys_ and @c publish_ to turn visible;
     *  the claim word holds until then. */
    expected<insert_result_t> launch_(insert_plan_t& plan, std::span<key_t const> keys, tape_view<byte_t const> tape,
                                      traversal_context_t& context, insert_config_t config) noexcept {
        if (plan.index_ != this) return {{}, status_t::uninitialized_k};
        if (tape.offsets.size() != plan.entries + 1 || keys.size() != plan.entries)
            return {{}, status_t::config_mismatch_k};
        plan.index_ = nullptr;
        return launch_claimed_(plan.first_id, keys, tape, context, config);
    }

    /** Files every drained node of @p range under its key ahead of its watermark: the claim holder,
     *  or the closer of the word, is the map's one writer. Plain stores and one release byte per
     *  node, no read-modify-write. */
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
     *  publishing them; every refusal before the launch gives the claim back. */
    expected<insert_result_t> launch_claimed_(node_id_t first_id, std::span<key_t const> keys,
                                              tape_view<byte_t const> tape, traversal_context_t& context,
                                              insert_config_t config) noexcept {
        std::size_t const entries = tape.size();
        auto const refuse = [&](status_t failure) noexcept -> expected<insert_result_t> {
            unclaim_(first_id, entries);
            return {{}, failure};
        };
        if (!connectivity()) return refuse(status_t::uninitialized_k);
        if (context.device() != device_.ordinal || !context) return refuse(status_t::config_mismatch_k);
        if (status_t const kept = status(); failed(kept)) return refuse(kept);
        cuda_stream_handle_t const stream = context.stream();

        // Walks prepare stored nodes as much as the batch's, so the rung and spans cover both.
        std::size_t longest = longest_value_;
        u32_t const span_bytes = static_cast<u32_t>(max_of(span_bytes_, span_bytes_of_(tape, longest)));
        std::size_t const rung = kits_.rung_serving_(longest);
        if (rung == kits_type_::rungs_k) return refuse(status_t::unsupported_k);
        std::size_t const batch_bytes = tape.offsets[entries] - tape.offsets[0];
        std::size_t const total_nodes = first_id + entries;

        // Every carve and growth precedes the first launch, so a refusal publishes nothing.
        build_shape_t const shape = build_shape_(config.expansion_factor, span_bytes, rung);
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   cuda_resident_blocks(vram_kernel_t::insert_k, 0) * vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(slices * shape.arena_shape.template slice_bytes<node_id_t>());
        byte_t* const staging = context.ensure_staging(build_staging_bytes_(entries, total_nodes));
        if (!arena || !staging) return refuse(status_t::bad_alloc_k);
        if (status_t const grown = grow_nodes_(total_nodes, stream); failed(grown)) return refuse(grown);
        if (status_t const mapped = values_tape_.commit(tape_bytes_ + batch_bytes + tape_margin_k); failed(mapped))
            return refuse(mapped);
        for (std::size_t offset = 0; offset != entries; ++offset)
            keyed_.assign(static_cast<node_id_t>(first_id + offset), keys[offset]);
        if (cuda_copy_async(api(), values_tape_.base() + tape_bytes_, tape.elements + tape.offsets[0], batch_bytes,
                            stream) != cuda_success_k ||
            cuda_copy_async(api(), staging, tape.offsets.data(), (entries + 1) * sizeof(u64_t), stream) !=
                cuda_success_k ||
            cuda_fill_async(api(), context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream) !=
                cuda_success_k)
            return refuse(status_t::device_error_k);
        // Every node walks twice, to insert and to re-select, each walk under an epoch of its own.
        auto const [epoch_base, epochs_status] = context.take_epochs(2 * entries, arena_bytes_before);
        if (failed(epochs_status)) return refuse(epochs_status);

        vram_build<node_id_t> build;
        build.storage = storage();
        build.staged_offsets = reinterpret_cast<u64_t const*>(staging);
        build.tape_first = tape_bytes_;
        build.span_bytes = span_bytes;
        build.arena = arena;
        build.arena_shape = shape.arena_shape;
        build.expansion = shape.expansion;
        build.epoch_base = epoch_base;
        build.telemetry = context.telemetry_.get();
        build.assigned_nodes = context.assigned_nodes_.get();
        // Every chain starts empty, and each back-link launch empties the ones it merged.
        build.link_heads = reinterpret_cast<node_id_t*>(staging + offsets_bytes_(entries));
        build.link_next = build.link_heads + total_nodes;
        build.touched_ids = build.link_next + entries * connectivity();
        build.touched_count = reinterpret_cast<u32_t*>(build.touched_ids + entries * connectivity());
        if (cuda_fill_async(api(), build.link_heads, 0xFF, total_nodes * sizeof(node_id_t), stream) != cuda_success_k)
            return refuse(status_t::device_error_k);
        auto const [kits, kits_status] = kits_kernels_(context);
        if (failed(kits_status)) return refuse(kits_status);
        status_t const launched = launch_insert_(*kits, build, shape, tape.offsets, first_id, rung, stream);
        if (failed(launched)) return refuse(launched);

        tape_bytes_ += batch_bytes;
        longest_value_ = longest;
        span_bytes_ = span_bytes;
        if (status_t const drained = context.wait(); failed(drained)) return {{}, fail_(drained)};

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
        return {result, status_t::success_k};
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

    /** The bytes an executor's run must hold before a context carved over it can search the
     *  @p queries under @p config, one aligned term per carve. */
    std::size_t traversal_memory_usage(tape_view<byte_t const> queries, search_config_t config) const noexcept {
        if (!connectivity() || !device_.multiprocessor_count || queries.offsets.empty()) return 0;
        std::size_t const queries_count = queries.size();
        std::size_t longest = 0;
        u32_t const span_bytes = static_cast<u32_t>(span_bytes_of_(queries, longest));
        std::size_t const rung = kits_.rung_serving_(longest);
        gpu_arena_shape_t const arena_shape = walk_arena_shape_(config, span_bytes, rung);
        u32_t const slots = arena_shape.tail_entries ? 0u : walk_slots_(config);
        std::size_t const slices = std::size_t {walk_blocks_(queries_count, rung, slots)} * vram_warps_per_block_k;
        return aligned_(sizeof(vram_telemetry_block_t)) + 2 * aligned_(sizeof(u32_t)) +
               aligned_(slices * arena_shape.template slice_bytes<node_id_t>()) +
               aligned_(search_staging_bytes_(queries_count, queries.offsets[queries_count] - queries.offsets[0],
                                              config.count));
    }

    /** The bytes @ref traversal_memory_usage of @p queries under @p config holds, and
     *  @p extra_bytes a caller's own launches carve from the same context's allocator. */
    std::size_t traversal_memory_usage(tape_view<byte_t const> queries, search_config_t config,
                                       std::size_t extra_bytes) const noexcept {
        return traversal_memory_usage(queries, config) + aligned_(extra_bytes);
    }

    /**
     *  @brief Finds the nearest published neighbors of every query, ascending by distance with
     *      sentinel IDs marking absent matches. Safe to overlap with insertions on other streams.
     *
     *  @param[in] queries The queries, one tape entry of bytes each.
     *  @param[out] result_ids IDs of the matches, @b [queries,count], in host or device memory.
     *  @param[out] result_distances Distances of the same matches, in host or device memory.
     *  @param[in] config The search breadth and traversal.
     *  @param[in,out] context The stream every transfer and the launch ride; drained before
     *      returning.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index was never reserved.
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
     *  took: the index's own, or one a sharded index snapshot consistent across its shards. */
    expected<search_result_t> search_below_(tape_view<byte_t const> queries, nk::matrix_span<node_id_t> result_ids,
                                            nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                            search_config_t config, std::size_t visible) noexcept {
        std::size_t const queries_count = queries.size();
        if (!queries_count) return {{}, status_t::success_k};
        if (!connectivity()) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (result_ids.extent(0) != queries_count || result_ids.extent(1) < config.count ||
            result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)
            return {{}, status_t::config_mismatch_k};
        std::size_t longest = 0;
        u32_t const span_bytes = static_cast<u32_t>(span_bytes_of_(queries, longest));
        std::size_t const rung = kits_.rung_serving_(longest);
        if (rung == kits_type_::rungs_k) return {{}, status_t::unsupported_k};
        auto const [kits, kits_status] = kits_kernels_(context);
        if (failed(kits_status)) return {{}, kits_status};
        cuda_stream_handle_t const stream = context.stream();

        // The staging holds the queries' bytes from a word boundary, their offsets, the outputs.
        std::size_t const batch_bytes = queries.offsets[queries_count] - queries.offsets[0];
        std::size_t const results_count = queries_count * config.count;
        gpu_arena_shape_t const arena_shape = walk_arena_shape_(config, span_bytes, rung);
        u32_t const slots = arena_shape.tail_entries ? 0u : walk_slots_(config);
        u32_t const blocks = walk_blocks_(queries_count, rung, slots);
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(std::size_t {blocks} * vram_warps_per_block_k *
                                                   arena_shape.template slice_bytes<node_id_t>());
        byte_t* const staging = context.ensure_staging(search_staging_bytes_(queries_count, batch_bytes, config.count));
        if (!arena || !staging) return {{}, status_t::bad_alloc_k};
        std::size_t const tape_bytes = round_up_to_multiple<std::size_t>(batch_bytes + tape_margin_k, sizeof(u64_t));
        u64_t* const staged_offsets = reinterpret_cast<u64_t*>(staging + tape_bytes);
        node_id_t* const device_ids = reinterpret_cast<node_id_t*>(staged_offsets + queries_count + 1);
        distance_t* const device_distances = reinterpret_cast<distance_t*>(device_ids + results_count);
        auto const [epoch_base, epochs_status] = context.take_epochs(queries_count, arena_bytes_before);
        if (failed(epochs_status)) return {{}, epochs_status};
        if (cuda_copy_async(api(), staging, queries.elements + queries.offsets[0], batch_bytes, stream) !=
                cuda_success_k ||
            cuda_copy_async(api(), staged_offsets, queries.offsets.data(), (queries_count + 1) * sizeof(u64_t),
                            stream) != cuda_success_k ||
            cuda_fill_async(api(), context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream) !=
                cuda_success_k ||
            cuda_fill_async(api(), context.assigned_nodes_.get(), 0, 2 * sizeof(u32_t), stream) != cuda_success_k)
            return {{}, status_t::device_error_k};

        vram_arguments<node_id_t> arguments {};
        vram_search<node_id_t>& search = arguments.search;
        search.storage = storage();
        search.queries = staging;
        search.query_offsets = staged_offsets;
        search.queries_count = static_cast<u32_t>(queries_count);
        search.sampling_bound = static_cast<node_id_t>(visible);
        search.scratch_bytes_per_warp = arena_shape.scratch_bytes ? 0u : span_bytes;
        search.expansion = walk_expansion_(config);
        search.arena = arena;
        search.arena_shape = arena_shape;
        search.epoch_base = epoch_base;
        search.telemetry = context.telemetry_.get();
        search.assigned_queries = context.assigned_nodes_.get();
        search.live_walks = context.assigned_nodes_.get() + 1;
        search.concurrent_walks = device_.multiprocessor_count * cuda_resident_blocks(vram_kernel_t::search_k, slots) *
                                  walkers_per_block_(rung);
        search.filling_nodes = filling_nodes<u32_t>(search.expansion, static_cast<u32_t>(connectivity()));
        search.result_ids = device_ids;
        search.result_distances = device_distances;
        search.count = config.count;
        if (status_t const launched = cuda_launch(api(), kits_function_at_(*kits, rung, vram_kernel_t::search_k, slots),
                                                  blocks, block_threads_k,
                                                  walkers_per_block_(rung) * search.scratch_bytes_per_warp, stream,
                                                  cuda_overlap_t::serialized_k, arguments);
            failed(launched))
            return {{}, launched};
        std::size_t const row_bytes = config.count * sizeof(node_id_t);
        if (cuda_copy_2d_async(api(), result_ids.data(), result_ids.stride_bytes(0), device_ids, row_bytes, row_bytes,
                               queries_count, stream) != cuda_success_k ||
            cuda_copy_2d_async(api(), result_distances.data(), result_distances.stride_bytes(0), device_distances,
                               config.count * sizeof(distance_t), config.count * sizeof(distance_t), queries_count,
                               stream) != cuda_success_k)
            return {{}, status_t::device_error_k};
        if (status_t const drained = context.wait(); failed(drained)) return {{}, drained};
        vram_telemetry_block_t const spent = context.spent();
        search_result_t result;
        result.count = config.count;
        result.visited_members = spent.hops;
        result.computed_distances = spent.scored;
        result.restarts = spent.restarts;
        result.tail_compactions = spent.tail_compactions;
        result.tail_dropped = spent.tail_dropped;
        result.visited_saturations = spent.visited_saturations;
        return {result, status_t::success_k};
    }

  public:

#pragma endregion Search

#pragma region Refinement

    /** Optional tuning knobs of @c refine(); zeros mean "derive from connectivity". */
    struct refine_config_t {

        /** How many candidates a re-selection walk keeps, as an insertion's. */
        candidates_count_t expansion_factor = 0;

        /** Bytes the whole pass may hold on the device beyond the index: the audit, the heal's
         *  buffers, a bit per node and the walks' arena, then tiles of nodes at 4 B per neighbor
         *  slot plus 4 B each, as many as fit. Zero takes every walked node in one tile. */
        std::size_t memory_limit = 0;
    };

    using refine_report_t = basic::refine_report_t;

    /** A split's knobs: its refine's, and the seed that orders the parts' growth. */
    struct split_config_t {
        refine_config_t refine {};
        u64_t seed = 0;
    };
    using reachability_t = utilities::reachability_t;

  private:

    /** Device scratch an audit carves over the nodes it judges: the reached set and two frontiers,
     *  then the levels that expanded anything and the nodes the last two sweeps expanded, by the
     *  sweep number's parity, then the unreached and reconnected counters an audit copies together,
     *  the nodes and links a sweep's tile kept, and for a refine the unreached nodes' IDs and their
     *  in-link counts, @b [nodes] each, and the changed bits. Sweeps run until one expands nothing:
     *  every node enters one frontier at most, so the nodes bound them. */
    struct audit_scratch_t {
        typename traversal_context_t::bytes_ptr_t bytes;
        vram_bitset_t reached, frontier, next_frontier;
        u32_t* levels = nullptr;
        u32_t* expanded = nullptr;
        u32_t* unreached = nullptr;
        u32_t* reconnected = nullptr;
        u32_t* swept = nullptr;
        u32_t* island_ids = nullptr;
        u32_t* in_links = nullptr;
        vram_bitset_t changed {};
        std::size_t words = 0;
    };

    /** Which share of the scratch a carve covers: the audit's alone, or a refine's with its
     *  heal. */
    enum class audit_use_t : u8_t { health_k, refine_k };

    /** The counters an audit copies into page-locked memory, in the device's order: the levels that
     *  expanded, the last sweep's count, nonzero only for a sweep short of convergence, then the
     *  unreached and reconnected counters, and the nodes and links a sweep's tile kept. */
    struct audit_counters_t {
        u32_t levels;
        u32_t last_expanded;
        u32_t unreached;
        u32_t reconnected;
        u32_t swept[2];
    };

    /** Gives an audit's counters back to the stream's pinned slots they came from. */
    struct audit_counters_return_t {
        cuda_pinned_slots_t* pinned_slots = nullptr;
        void operator()(audit_counters_t* counters) const noexcept { pinned_slots->give_back(counters); }
    };
    using audit_counters_ptr_t = std::unique_ptr<audit_counters_t, audit_counters_return_t>;
    static_assert(sizeof(audit_counters_t) <= cuda_pinned_slots_t::slot_bytes_k, "An audit's counters fit one slot");

    /** One part of a split as the host assigns it: its nodes and the picked ones leading them, its
     *  value bytes and longest value, and where its receiver's offsets column starts inside the
     *  pinned host byte span. */
    struct split_part_t {
        std::size_t entries = 0, picked_count = 0, bytes = 0, longest = 0, offsets_first = 0;
    };

    /** A split's state beside the refinement it turns into, empty for a refine or an absorb. */
    struct split_t {

        /** The index the split empties, while it still holds its nodes. */
        vram_ragged_graph* donor = nullptr;

        /** The receivers' graphs, one part each, the ones whose claim words the plan closed,
         *  counted from the first, and the one the refinement is bound to. */
        growing_buffer<vram_ragged_graph*> receivers {global_allocator};
        std::size_t closed_receivers = 0;
        std::size_t bound_receiver = 0;

        /** The donor's nodes at planning, and the most nodes one part takes: those over the
         *  receivers, rounded up. */
        std::size_t entries = 0, part_capacity = 0;

        /** The device scratch, laid out as @c split_scratch_ reads it. */
        typename traversal_context_t::bytes_ptr_t device_bytes;

        /** Page-locked host bytes borrowed from the plan's context, laid out as @c split_layout_
         *  places them: the node parts, assigned into slots in place, the picked bits, the
         *  receivers' offsets columns back to back, the donor's value starts, the bases and the
         *  roots planning uploads, and the last round's placed count. */
        byte_t* host_bytes = nullptr;

        /** The growth round the next batch queues from. */
        u32_t next_round = 0;

        /** The parts as the host assigns them. */
        growing_buffer<split_part_t> parts {global_allocator};

        /** The audits and islands of the receivers healed so far, summed. */
        reachability_t reachability {};
        std::size_t islands = 0;

        /** Rounds of re-selection each receiver runs, none once a stop cut the pass. */
        u32_t rounds = 1;

        explicit split_t(typename traversal_context_t::bytes_ptr_t&& empty) noexcept : device_bytes(std::move(empty)) {}
        split_t(split_t&& other) noexcept
            : donor(std::exchange(other.donor, nullptr)), receivers(std::move(other.receivers)),
              closed_receivers(std::exchange(other.closed_receivers, 0)), bound_receiver(other.bound_receiver),
              entries(other.entries), part_capacity(other.part_capacity), device_bytes(std::move(other.device_bytes)),
              host_bytes(std::exchange(other.host_bytes, nullptr)), next_round(other.next_round),
              parts(std::move(other.parts)), reachability(other.reachability), islands(other.islands),
              rounds(other.rounds) {}
        split_t& operator=(split_t&&) = delete;
    };

  public:

    /**
     *  @brief One refinement in flight on this index's device: @ref queue a phase, @ref wait for
     *      it, @ref advance past it, and @ref finish once no phase follows.
     *
     *  The phases re-select every node of the planned range in one walk launch, each merging its
     *  picks into its row, then sweep tiles of target nodes, sized from @c refine_config_t, each by
     *  a count, a scatter onto a tape and a merge of every target's run into its row, then audit
     *  reachability, then heal round by round while the islands shrink, reconnecting each unreached
     *  node into a reached pick or the landmark covering it.
     *  Planning made every allocation and closed the claim word, which the plan's death reopens, so
     *  batches answer @c contended_k meanwhile while searches on other streams run throughout. A
     *  phase only queues, and only a device fault fails one; a stopped sequence of phases leaves
     *  the graph correct and merely less improved.
     *
     *  A split's plan first partitions and scatters, as @ref plan_split describes, then binds the
     *  receivers one after another, each refined over its picked nodes and healed, its tallies
     *  summing into the plan's; it closed every receiver's claim word and the donor's, and its
     *  death reopens every one still closed.
     */
    struct refine_plan_t {

        /** An unbound plan over @p scratch - what a refused plan answers with. */
        explicit refine_plan_t(audit_scratch_t&& scratch) noexcept
            : scratch_(std::move(scratch)),
              split_(typename traversal_context_t::bytes_ptr_t {nullptr, scratch_.bytes.get_deleter()}) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : scratch_(std::move(other.scratch_)), split_(std::move(other.split_)),
              index_(std::exchange(other.index_, nullptr)), counters_(std::move(other.counters_)),
              config_(other.config_), nodes_(other.nodes_), islands_(other.islands_), next_sweep_(other.next_sweep_),
              sweep_(other.sweep_), tile_rows_(other.tile_rows_), swept_links_(other.swept_links_),
              phase_(other.phase_), status_(other.status_), tally_(other.tally_) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** Reopens the claim words the plan closed, whether it finished or not. */
        ~refine_plan_t() noexcept {
            if (index_ && split_.receivers.empty()) index_->reopen_();
            for (std::size_t receiver = 0; receiver != split_.closed_receivers; ++receiver)
                split_.receivers[receiver]->reopen_();
            if (split_.donor) split_.donor->reopen_();
        }

        /**
         *  @brief Queues the current phase on the stream of @p context, the one the plan was opened
         *      on, then the copy of its counters and telemetry behind its event, without waiting or
         *      allocating anything.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers.
         */
        status_t queue(traversal_context_t& context) noexcept {
            if (!counters_) return status_t::uninitialized_k;
            if (failed(status_)) return status_;
            cuda_stream_handle_t const stream = context.stream();
            // A split's donor holds the graph until its scatter published the parts.
            vram_ragged_graph const& graph = split_.donor ? *split_.donor : *index_;
            auto const [graph_kernels, kernels_status] = graph.graph_kernels_(context);
            if (failed(kernels_status)) return kernels_status;
            switch (phase_) {
            case phase_t::partition_k: status_ = split_.donor->queue_rounds_(*graph_kernels, split_, stream); break;
            case phase_t::picking_k: status_ = split_.donor->queue_picked_(*graph_kernels, split_, stream); break;
            case phase_t::scatter_k: status_ = split_.donor->queue_scatter_(*graph_kernels, split_, stream); break;
            case phase_t::reselect_k:
                status_ = index_->reselect_(context, scratch_, config_, nodes_, index_->size());
                break;
            case phase_t::count_back_links_k:
                status_ = index_->queue_count_back_links_(*graph_kernels, context, scratch_, counters_.get(), nodes_,
                                                          sweep_, tile_rows_);
                break;
            case phase_t::scatter_back_links_k:
                status_ = index_->queue_scatter_back_links_(*graph_kernels, context, scratch_, nodes_, sweep_,
                                                            tile_rows_, swept_links_);
                break;
            case phase_t::merge_back_links_k:
                status_ = index_->queue_merge_back_links_(context, scratch_, config_, nodes_, sweep_, tile_rows_,
                                                          swept_links_);
                break;
            case phase_t::audit_k: status_ = queue_sweeps_(*graph_kernels, stream); break;
            case phase_t::heal_k:
                // Only island nodes gain links, and no reached node points at one, so the audit
                // before stands.
                if (!next_sweep_)
                    status_ = index_->queue_reconnect_(*graph_kernels, scratch_, islands_, index_->size(), stream);
                if (succeeded(status_)) status_ = queue_sweeps_(*graph_kernels, stream);
                break;
            default: return status_t::success_k;
            }
            if (succeeded(status_)) status_ = context.record();
            return status_;
        }

        /** Sleeps until the queued phase finished, on the event of @p context. */
        status_t wait(traversal_context_t& context) noexcept {
            if (!counters_ || phase_ == phase_t::done_k) return status_;
            if (succeeded(status_)) status_ = context.event_.wait();
            return status_;
        }

        /** The barrier after a phase finished: reads its counters, picks the next phase, and
         *  answers whether one follows. */
        bool advance() noexcept {
            if (!counters_ || failed(status_) || phase_ == phase_t::done_k) return false;
            if (phase_ == phase_t::partition_k) {
                // A last round that still placed nodes leaves others unplaced, so the phase grows
                // on.
                if (!placing_(split_)) phase_ = phase_t::picking_k;
                return true;
            }
            if (phase_ == phase_t::picking_k) {
                assign_slots_();
                phase_ = phase_t::scatter_k;
                return true;
            }
            if (phase_ == phase_t::scatter_k) {
                publish_parts_();
                bind_receiver_();
                return phase_ != phase_t::done_k;
            }
            audit_counters_t const& counters = *counters_;
            if (phase_ == phase_t::reselect_k) {
                tally_.reselected_nodes += nodes_.size();
                open_sweep_(0);
                return true;
            }
            if (phase_ == phase_t::count_back_links_k) {
                sweep_.bound = static_cast<node_id_t>(sweep_.first + counters.swept[0]);
                swept_links_ = counters.swept[1];
                tally_.swept_links += swept_links_;
                phase_ = phase_t::scatter_back_links_k;
                return true;
            }
            if (phase_ == phase_t::scatter_back_links_k) {
                phase_ = phase_t::merge_back_links_k;
                return true;
            }
            if (phase_ == phase_t::merge_back_links_k) {
                if (sweep_.bound < index_->size()) open_sweep_(sweep_.bound);
                else phase_ = phase_t::audit_k;
                return true;
            }
            // A last sweep that still expanded leaves nodes unjudged, so the phase sweeps on.
            if (counters.last_expanded) return true;
            next_sweep_ = 0;
            tally_.reachability = audited_(counters, index_->size());
            tally_.islands_reconnected = counters.reconnected;
            u32_t const left = counters.unreached;
            if (phase_ == phase_t::audit_k) tally_.islands_found = left;
            // A round that frees no island frees none on the next either, since the picks it needs
            // are unreached.
            bool const healing = left && (phase_ == phase_t::audit_k || left < islands_);
            islands_ = left;
            phase_ = healing ? phase_t::heal_k : phase_t::done_k;
            if (!healing && !split_.receivers.empty()) return settle_receiver_();
            return healing;
        }

        /** Skips the re-selection when it has not been queued yet, so the next phase audits and
         *  heals, and a split's later receivers skip theirs too; a sweep under way finishes, and it
         *  is a no-op in any other phase, a split's partition and scatter included. */
        void stop() noexcept {
            if (phase_ == phase_t::reselect_k) phase_ = phase_t::audit_k, split_.rounds = 0;
            else if (sweeping_()) split_.rounds = 0;
        }

        /** Whether the current phase re-selects the planned range - the phase a caller's
         *  progress counts. */
        bool reselecting() const noexcept { return phase_ == phase_t::reselect_k; }

        /** Whether the current phase moves a split's nodes - the phases no stop skips. */
        bool moving() const noexcept {
            return phase_ == phase_t::partition_k || phase_ == phase_t::picking_k || phase_ == phase_t::scatter_k;
        }

        /** Where every node of a split's donor goes, by donor ID: its node in its part's receiver,
         *  plus its part times @ref part_capacity; empty until the partition finished, and for a
         *  refine or an absorb. The keys follow it once the scatter finished. */
        std::span<u32_t const> slots() const noexcept {
            if (!split_.part_capacity || phase_ == phase_t::partition_k || phase_ == phase_t::picking_k ||
                failed(status_))
                return {};
            return {reinterpret_cast<u32_t const*>(split_.host_bytes), split_.entries};
        }

        /** The most nodes one part of a split takes, the stride of @ref slots; zero otherwise. */
        std::size_t part_capacity() const noexcept { return split_.part_capacity; }

        /** Nodes re-selected so far. */
        std::size_t visited() const noexcept { return tally_.reselected_nodes; }

        /**
         *  @brief The pass: its telemetry read from the last copy on @p context, complete once the
         *      @ref advance answered false, otherwise @c cancelled_k; a failure that a phase kept
         *      takes its place.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers.
         *  - @c cancelled_k A phase still follows.
         */
        expected<refine_report_t> finish(traversal_context_t const& context) const noexcept {
            if (!counters_) return {{}, status_t::uninitialized_k};
            refine_report_t result = tally_;
            vram_telemetry_block_t const spent = context.spent();
            result.changed_nodes = spent.changed_nodes;
            result.traversal_stats = {.computed_distances = spent.scored,
                                      .visited_members = spent.hops,
                                      .restarts = spent.restarts,
                                      .saturated_walks = spent.visited_saturations};
            if (failed(status_)) return {result, status_};
            return {result, phase_ == phase_t::done_k ? status_t::success_k : status_t::cancelled_k};
        }

      private:
        friend struct vram_ragged_graph;
        enum class phase_t : u8_t {
            partition_k,
            picking_k,
            scatter_k,
            reselect_k,
            audit_k,
            count_back_links_k,
            scatter_back_links_k,
            merge_back_links_k,
            heal_k,
            done_k
        };

        /** Whether the current phase runs a kernel of a sweep's tile. */
        bool sweeping_() const noexcept {
            return phase_ == phase_t::count_back_links_k || phase_ == phase_t::scatter_back_links_k ||
                   phase_ == phase_t::merge_back_links_k;
        }

        /** Opens the sweep's tile of target nodes from @p first, up to @c tile_rows_ of them. */
        void open_sweep_(node_id_t first) noexcept {
            sweep_ = {first, static_cast<node_id_t>(min_of<std::size_t>(first + tile_rows_, index_->size()))};
            ++tally_.swept_tiles;
            phase_ = phase_t::count_back_links_k;
        }

        /** Assigns every donor node its slot in place in the pinned host bytes, each part's picked
         *  nodes first and both runs in ID order, and lays out the receivers' offsets columns. */
        void assign_slots_() noexcept {
            std::size_t const entries = split_.entries, parts = split_.receivers.size();
            split_layout_t const layout = split_layout_(entries, parts);
            u32_t* const slots = reinterpret_cast<u32_t*>(split_.host_bytes);
            u32_t const* const picked = slots + entries;
            u64_t* const offsets = reinterpret_cast<u64_t*>(split_.host_bytes + layout.host_offsets);
            u64_t const* const starts = reinterpret_cast<u64_t const*>(split_.host_bytes + layout.host_starts);
            auto const picked_of = [&](std::size_t node) noexcept { return (picked[node / 32] >> (node % 32)) & 1u; };
            for (std::size_t part = 0; part != parts; ++part) split_.parts[part] = {};
            for (std::size_t node = 0; node != entries; ++node) {
                split_part_t& part = split_.parts[slots[node]];
                ++part.entries, part.picked_count += picked_of(node);
            }
            // Each column's first entry counts its part's assigned nodes until the sums below zero
            // it: the picked nodes are assigned in a first pass, so the rest continue past them in
            // a second pass over the nodes.
            for (std::size_t part = 0, first = 0; part != parts; ++part) {
                split_.parts[part].offsets_first = first;
                offsets[first] = 0;
                first += split_.parts[part].entries + 1;
            }
            for (u32_t const picked_pass : {1u, 0u})
                for (std::size_t node = 0; node != entries; ++node) {
                    if (picked_of(node) != picked_pass) continue;
                    std::size_t const index = slots[node];
                    split_part_t& part = split_.parts[index];
                    std::size_t const position = offsets[part.offsets_first]++;
                    std::size_t const bytes = starts[node + 1] - starts[node];
                    offsets[part.offsets_first + position + 1] = bytes;
                    part.bytes += bytes, part.longest = max_of(part.longest, bytes);
                    slots[node] = static_cast<u32_t>(index * split_.part_capacity + position);
                }
            for (split_part_t const& part : split_.parts) {
                u64_t* const column = offsets + part.offsets_first;
                column[0] = 0;
                for (std::size_t position = 0; position != part.entries; ++position)
                    column[position + 1] += column[position];
            }
        }

        /** Moves every donor key to its slot and publishes every receiver's nodes at once, its keys
         *  filed under the closed claim words, its tape's extent and longest value its part's and
         *  its spans the donor's, then empties the donor and reopens its claim word at zero. */
        void publish_parts_() noexcept {
            vram_ragged_graph& donor = *std::exchange(split_.donor, nullptr);
            u32_t const* const slots = reinterpret_cast<u32_t const*>(split_.host_bytes);
            for (std::size_t node = 0; node != split_.entries; ++node)
                split_.receivers[slots[node] / split_.part_capacity]->keyed_.assign(
                    static_cast<node_id_t>(slots[node] % split_.part_capacity),
                    donor.keyed_[static_cast<node_id_t>(node)]);
            for (std::size_t index = 0; index != split_.receivers.size(); ++index) {
                split_part_t const& part = split_.parts[index];
                vram_ragged_graph& receiver = *split_.receivers[index];
                receiver.tape_bytes_ = part.bytes, receiver.longest_value_ = part.longest;
                receiver.span_bytes_ = donor.span_bytes_;
                receiver.publish_keys_({0, static_cast<node_id_t>(part.entries)});
                fu::standard_atomic_ref<std::size_t>(receiver.last_visible_id_)
                    .store(part.entries, std::memory_order_release);
            }
            donor.forget_();
            donor.reopen_();
        }

        /** Turns the pass to the next receiver holding nodes: its picked nodes as the range, or
         *  straight the audit when it has none or a stop cut the pass; past the last receiver, it
         *  reports the sums over every part. */
        void bind_receiver_() noexcept {
            while (split_.bound_receiver != split_.receivers.size() && !split_.parts[split_.bound_receiver].entries)
                ++split_.bound_receiver;
            if (split_.bound_receiver == split_.receivers.size()) {
                tally_.reachability = split_.reachability, tally_.islands_found = split_.islands;
                phase_ = phase_t::done_k;
                return;
            }
            split_part_t const& part = split_.parts[split_.bound_receiver];
            index_ = split_.receivers[split_.bound_receiver];
            nodes_ = {0, static_cast<node_id_t>(part.picked_count)};
            islands_ = 0, next_sweep_ = 0;
            phase_ = !split_.rounds || nodes_.empty() ? phase_t::audit_k : phase_t::reselect_k;
        }

        /** Folds a healed receiver's audit and islands into the split's, then binds the next one,
         *  answering whether a phase follows. */
        bool settle_receiver_() noexcept {
            split_.reachability += tally_.reachability;
            split_.islands += tally_.islands_found;
            ++split_.bound_receiver;
            bind_receiver_();
            return phase_ != phase_t::done_k;
        }

        audit_scratch_t scratch_;
        split_t split_;
        vram_ragged_graph* index_ = nullptr;

        /** The page-locked counters every phase copies into, held once the plan is bound. */
        audit_counters_ptr_t counters_ {nullptr, audit_counters_return_t {}};
        refine_config_t config_ {};

        /** The IDs the re-selection walks, over every node the bound index publishes. */
        iota_view<node_id_t> nodes_;

        /** The islands the last audit left. */
        u32_t islands_ = 0;

        /** The audit sweep the next batch queues from, zero before an audit starts. */
        u32_t next_sweep_ = 0;

        /** The open sweep tile's targets, the most nodes a tile takes, and the links it kept. */
        iota_view<node_id_t> sweep_;
        std::size_t tile_rows_ = 0;
        u32_t swept_links_ = 0;

        /** Queues the audit's next batch of sweeps through @p graph_kernels, its seeds first when
         *  it starts. */
        status_t queue_sweeps_(kernel_table_t const& graph_kernels, cuda_stream_handle_t stream) noexcept {
            std::size_t const entries = index_->size();
            status_t const queued =
                next_sweep_
                    ? index_->queue_sweeps_(graph_kernels, scratch_, counters_.get(), entries, next_sweep_, stream)
                    : index_->queue_audit_(graph_kernels, scratch_, counters_.get(), entries, stream);
            next_sweep_ = next_sweep_ ? 2 * next_sweep_ : 2;
            return queued;
        }
        phase_t phase_ = phase_t::done_k;
        status_t status_ = status_t::success_k;
        refine_report_t tally_;
    };

    /**
     *  @brief Opens a refinement of the nodes in @p nodes, clamped to the published IDs, on
     *      @p context 's stream: closes the claim word, carves the scratch, grows the context's
     *      arena and staging for the walk, and takes page-locked counters.
     *
     *  No phase allocates. An empty range heals alone. Each re-selection walks from the node's own
     *  neighbors and from landmarks over every published node, so a range refines against the whole
     *  graph.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet.
     *  - @c bad_alloc_k The scratch, the arena or a tile cannot be placed, or the limit holds not
     *    even one tile row past the carried state.
     *  - @c unsupported_k The shared kernels lack this node-ID type.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk.
     *  - @c contended_k A batch, a load or another plan in flight.
     */
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, refine_config_t config,
                                        traversal_context_t& context) noexcept {
        refine_plan_t plan = unbound_plan_(context);
        if (status_t const refused = graph_kernels_(context).status; failed(refused)) return {std::move(plan), refused};
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        plan.index_ = this;
        std::size_t const entries = size();
        node_id_t const bound = static_cast<node_id_t>(min_of<std::size_t>(nodes.bound, entries));
        status_t const opened = open_refine_(plan, {min_of(nodes.first, bound), bound}, entries, longest_value_,
                                             span_bytes_, context, config);
        return {std::move(plan), opened};
    }

    /**
     *  One @ref refine_plan_t driven to its end on the calling thread, waiting once per phase.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet.
     *  - @c unsupported_k The shared kernels lack this node-ID type.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk.
     *  - @c contended_k A batch, a load or another plan in flight.
     */
    expected<refine_report_t> refine(iota_view<node_id_t> nodes, refine_config_t config,
                                     traversal_context_t& context) noexcept {
        auto [plan, status] = plan_refine(nodes, config, context);
        if (failed(status)) return {{}, status};
        do
            if (status_t const queued = plan.queue(context); failed(queued) || failed(plan.wait(context))) break;
        while (plan.advance());
        return plan.finish(context);
    }

    /**
     *  @brief Opens an absorb of @p donors: appends every node of each past the watermark on this
     *      device, one donor after another, empties them, and answers the refinement re-selecting
     *      every node and healing.
     *
     *  Every allocation and every copy happens here, behind one wait: each donor's values copy
     *  past this tape, its offsets shift by this tape's bytes on the host, its neighbor rows copy
     *  and rebase onto the IDs past the watermark on the device, and its keys move on the host.
     *  The appended keys file and the watermark turns over them before the plan returns, and the
     *  claim word stays closed until the plan dies, so searches see the appended parts at once and
     *  walk across them as the re-selection links them. Donors must share this index's device,
     *  connectivity and metric, and nothing may search them meanwhile.
     *
     *  @param[in,out] donors The indexes giving their nodes up, each emptied once absorbed.
     *  @param[in] graph The member of each donor holding its graph.
     *  @param[in] config The refinement budget the plan runs under.
     *  @param[in,out] context The traversal context the plan's walks draw their buffers from.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A null donor, this index itself, or one of another device,
     *    connectivity or metric.
     *  - @c contended_k A batch, a load or a plan in flight on this index or on a donor.
     *  - @c overflow_k Nodes past the reserved ceiling.
     */
    template <typename owner_type_>
    expected<refine_plan_t> plan_absorb(std::span<owner_type_*> donors, vram_ragged_graph owner_type_::* graph,
                                        refine_config_t config, traversal_context_t& context) noexcept {
        refine_plan_t plan = unbound_plan_(context);
        auto const [graph_kernels, kernels_status] = graph_kernels_(context);
        if (failed(kernels_status)) return {std::move(plan), kernels_status};
        std::size_t donor_nodes = 0, donor_bytes = 0, longest = longest_value_, span_bytes = span_bytes_;
        for (owner_type_* const donor : donors) {
            if (!donor || &(donor->*graph) == this) return {std::move(plan), status_t::config_mismatch_k};
            vram_ragged_graph const& donor_graph = donor->*graph;
            if (!donor_graph.size()) continue;
            if (donor_graph.device_.ordinal != device_.ordinal || donor_graph.connectivity() != connectivity() ||
                donor_graph.kernel_ != kernel_)
                return {std::move(plan), status_t::config_mismatch_k};
            if (status_t const kept = donor_graph.status(); failed(kept)) return {std::move(plan), kept};
            donor_nodes += donor_graph.size(), donor_bytes += donor_graph.tape_bytes_;
            longest = max_of(longest, donor_graph.longest_value_),
            span_bytes = max_of(span_bytes, donor_graph.span_bytes_);
        }
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        plan.index_ = this;
        std::size_t const base = size();
        if (donor_nodes > ceiling_nodes_ - base) return {std::move(plan), status_t::overflow_k};
        // Donors close one by one, and a refusal reopens the ones closed before it.
        std::size_t closed_donors = 0;
        auto const reopen_donors = [&]() noexcept {
            for (std::size_t donor = 0; donor != closed_donors; ++donor) (donors[donor]->*graph).reopen_();
        };
        for (; closed_donors != donors.size(); ++closed_donors)
            if (status_t const closed = (donors[closed_donors]->*graph).close_(); failed(closed)) {
                reopen_donors();
                return {std::move(plan), closed};
            }
        std::size_t const entries = base + donor_nodes;
        cuda_stream_handle_t const stream = context.stream();
        growing_buffer<u64_t> shifted_offsets {global_allocator};
        status_t absorbed = open_refine_(plan, {0, static_cast<node_id_t>(entries)}, entries, longest, span_bytes,
                                         context, config);
        if (succeeded(absorbed)) absorbed = grow_nodes_(entries, stream);
        if (succeeded(absorbed)) absorbed = values_tape_.commit(tape_bytes_ + donor_bytes + tape_margin_k);
        if (succeeded(absorbed) && failed(shifted_offsets.resize_uninitialized(entries - base + 1)))
            absorbed = status_t::bad_alloc_k;

        // The appended nodes publish only once every donor's copies drained, so a refusal publishes
        // none.
        std::size_t const row_bytes = connectivity() * sizeof(node_id_t);
        std::size_t into = base, tape_bytes = tape_bytes_;
        for (owner_type_* const donor : donors) {
            vram_ragged_graph& donor_graph = donor->*graph;
            std::size_t const count = donor_graph.size();
            if (failed(absorbed) || !count) continue;
            if (cuda_copy(api(), shifted_offsets.data(), donor_graph.offsets_column_.base(),
                          (count + 1) * sizeof(u64_t)) != cuda_success_k) {
                absorbed = status_t::device_error_k;
                continue;
            }
            for (std::size_t node = 0; node <= count; ++node) shifted_offsets[node] += tape_bytes;
            vram_graph<node_id_t> graph {};
            graph.neighbors = reinterpret_cast<node_id_t*>(neighbors_matrix_.base());
            graph.connectivity = static_cast<u32_t>(connectivity());
            graph.first_id = static_cast<node_id_t>(into);
            graph.nodes_count = static_cast<u32_t>(count);
            graph.source_bound = static_cast<node_id_t>(count);
            graph.shift = static_cast<node_id_t>(into);
            if (cuda_copy(api(), offsets_column_.base() + into * sizeof(u64_t), shifted_offsets.data(),
                          (count + 1) * sizeof(u64_t)) != cuda_success_k ||
                (donor_graph.tape_bytes_ &&
                 cuda_copy_async(api(), values_tape_.base() + tape_bytes, donor_graph.values_tape_.base(),
                                 donor_graph.tape_bytes_, stream) != cuda_success_k) ||
                cuda_copy_async(api(), neighbors_matrix_.base() + into * row_bytes,
                                donor_graph.neighbors_matrix_.base(), count * row_bytes, stream) != cuda_success_k)
                absorbed = status_t::device_error_k;
            else
                absorbed = launch_graph_(*graph_kernels, vram_kernel_t::rebase_links_k,
                                         graph_blocks_(vram_kernel_t::rebase_links_k), graph, stream);
            into += count, tape_bytes += donor_graph.tape_bytes_;
        }
        if (succeeded(absorbed)) absorbed = context.wait();
        if (failed(absorbed)) {
            // The nodes past the watermark hold whatever was copied, so the next grow refills them.
            filled_nodes_ = min_of(filled_nodes_, base);
            reopen_donors();
            return {std::move(plan), absorbed};
        }
        // The closed claim words make this call every map's one writer.
        into = base;
        for (owner_type_* const donor : donors) {
            vram_ragged_graph& donor_graph = donor->*graph;
            for (std::size_t node = 0, count = donor_graph.size(); node != count; ++node)
                keyed_.assign(static_cast<node_id_t>(into++), donor_graph.keyed_[static_cast<node_id_t>(node)]);
            donor_graph.forget_();
        }
        reopen_donors();
        tape_bytes_ = tape_bytes, longest_value_ = longest, span_bytes_ = span_bytes;
        publish_keys_({static_cast<node_id_t>(base), static_cast<node_id_t>(entries)});
        fu::standard_atomic_ref<std::size_t>(last_visible_id_).store(entries, std::memory_order_release);
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Opens a split of this index into @p receivers - the inverse of @c plan_absorb: the
     *      plan's first phases move every node into one graph-local part per receiver, then refine
     *      every receiver in turn over the nodes that lost a link, and heal.
     *
     *  Planning queues the partition behind every allocation: one root per part through @ref
     *  seeded_bijection under @p seed, then rounds of growth over the links, each part capped at
     *  the nodes over the receivers rounded up, and each node a round reaches kept by its lowest-ID
     *  proposer, so equal seeds over equal graphs split alike; nodes no round reached enter the
     *  smallest parts in ID order, and the nodes linking across are picked. Planning maps each
     *  receiver's tape for the most its part can take, its nodes' capacity at the longest value,
     *  never past this index's tape. The barrier after the partition assigns every node its slot,
     *  each part's picked nodes first. The scatter moves every node's value and neighbors on the
     *  device, the links inside its part rebased and the links across dropped, and the barrier
     *  after it publishes every receiver at once and empties this index, whose claim word reopens
     *  at zero. Each receiver then re-selects its picked nodes, audits and heals as @ref
     *  plan_refine does, and the tallies sum over the receivers.
     *
     *  The keys move with the nodes, by the plan's @c slots, at the barrier after the scatter.
     *  Receivers must share this index's device and metric, and either its connectivity or none
     *  yet, which planning then reserves; every claim word stays closed for the plan's life, and
     *  nothing may search this index meanwhile.
     *
     *  @param[in,out] receivers The empty indexes taking the parts, at least two, all distinct.
     *  @param[in] graph The member of each receiver holding its graph.
     *  @param[in] config The refinement knobs, zeros meaning "derive from connectivity", and the
     *      seed picking the parts' roots, so equal seeds over equal graphs split alike.
     *  @param[in,out] context The traversal context the plan's walks draw their buffers from.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *    index itself, or one of another device, connectivity or metric.
     *  - @c contended_k A batch, a load or a plan in flight on this index or on a receiver.
     *  - @c overflow_k Part slots past the 32-bit IDs.
     */
    template <typename owner_type_>
    expected<refine_plan_t> plan_split(std::span<owner_type_*> receivers, vram_ragged_graph owner_type_::* graph,
                                       split_config_t config, traversal_context_t& context) noexcept {
        using phase_t = typename refine_plan_t::phase_t;
        refine_plan_t plan = unbound_plan_(context);
        auto const [graph_kernels, kernels_status] = graph_kernels_(context);
        if (failed(kernels_status)) return {std::move(plan), kernels_status};
        std::size_t const parts = receivers.size();
        if (parts < 2) return {std::move(plan), status_t::config_mismatch_k};
        split_t& split = plan.split_;
        if (failed(split.receivers.reserve_exactly(parts))) return {std::move(plan), status_t::bad_alloc_k};
        for (std::size_t receiver = 0; receiver != parts; ++receiver) {
            owner_type_* const into = receivers[receiver];
            if (!into || &(into->*graph) == this) return {std::move(plan), status_t::config_mismatch_k};
            for (std::size_t earlier = 0; earlier != receiver; ++earlier)
                if (receivers[earlier] == into) return {std::move(plan), status_t::config_mismatch_k};
            vram_ragged_graph const& part = into->*graph;
            if (part.device_.ordinal != device_.ordinal || part.kernel_ != kernel_ ||
                (part.connectivity() && part.connectivity() != connectivity()))
                return {std::move(plan), status_t::config_mismatch_k};
        }
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        for (owner_type_* const into : receivers) split.receivers.push_back_reserved(&(into->*graph));
        split.donor = this;
        split.entries = size();
        split.part_capacity = divide_round_up<std::size_t>(split.entries, parts);
        if (parts * split.part_capacity >= std::numeric_limits<u32_t>::max())
            return {std::move(plan), status_t::overflow_k};
        for (vram_ragged_graph* const into : split.receivers) {
            vram_ragged_graph& part = *into;
            if (status_t const closed = part.close_(); failed(closed)) return {std::move(plan), closed};
            ++split.closed_receivers;
            if (part.size()) return {std::move(plan), status_t::config_mismatch_k};
            if (status_t const kept = part.status(); failed(kept)) return {std::move(plan), kept};
        }
        // Every receiver holds a full part's nodes, sentinel-filled, and the most value bytes they
        // can take, before the partition queues.
        cuda_stream_handle_t const stream = context.stream();
        std::size_t const part_bytes = min_of(tape_bytes_, split.part_capacity * longest_value_);
        for (vram_ragged_graph* const into : split.receivers) {
            vram_ragged_graph& part = *into;
            if (status_t const reserved =
                    part.reserve({split.part_capacity, static_cast<connectivity_t>(connectivity()), 0}).status;
                failed(reserved))
                return {std::move(plan), reserved};
            if (status_t const grown = part.grow_nodes_(split.part_capacity, stream); failed(grown))
                return {std::move(plan), grown};
            if (status_t const mapped = part.values_tape_.commit(part_bytes + tape_margin_k); failed(mapped))
                return {std::move(plan), mapped};
        }
        node_id_t const capacity = static_cast<node_id_t>(split.part_capacity);
        status_t opened = open_refine_(plan, {0, capacity}, split.part_capacity, longest_value_, span_bytes_, context,
                                       config.refine);
        if (succeeded(opened) && split.entries) opened = carve_split_(split, config.seed, context);
        if (succeeded(opened) && split.entries) opened = queue_partition_(*graph_kernels, split, stream);
        if (failed(opened)) return {std::move(plan), opened};
        plan.phase_ = split.entries ? phase_t::partition_k : phase_t::done_k;
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  Measures forward reachability from as many stride-spread landmarks as one search draws -
     *  the connectivity; any shortfall against @c size() is the population no search can
     *  discover. Read-only on the graph, so concurrent insertions may leave fresh nodes
     *  uncounted. Waits on the context's event once per batch of sweeps, the batches doubling
     *  until one expands nothing.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet.
     *  - @c unsupported_k The shared kernels lack this node-ID type.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk.
     */
    expected<reachability_t> health(traversal_context_t& context) noexcept {
        std::size_t const entries = size();
        if (!entries) return {{}, status_t::success_k};
        auto const [graph_kernels, kernels_status] = graph_kernels_(context);
        if (failed(kernels_status)) return {{}, kernels_status};
        audit_scratch_t scratch = unbound_scratch_(context);
        if (status_t const carved = carve_audit_(scratch, context, audit_use_t::health_k, entries); failed(carved))
            return {{}, carved};
        audit_counters_ptr_t const counters = take_audit_counters_(context);
        if (!counters) return {{}, status_t::bad_alloc_k};
        if (status_t const queued = queue_audit_(*graph_kernels, scratch, counters.get(), entries, context.stream());
            failed(queued))
            return {{}, queued};
        for (u32_t first_sweep = 2;; first_sweep *= 2) {
            if (status_t const drained = context.wait(); failed(drained)) return {{}, drained};
            if (!counters->last_expanded) break;
            if (status_t const queued = queue_sweeps_(*graph_kernels, scratch, counters.get(), entries, first_sweep,
                                                      context.stream());
                failed(queued))
                return {{}, queued};
        }
        return {audited_(*counters, entries), status_t::success_k};
    }

    /** The bytes an executor's run must hold before a context carved over it can refine @p nodes
     *  nodes under @p config - @c size() for a refine, past the donors' nodes for an absorb: the
     *  context's counters, the re-selection's arena and staging, and the audit's scratch. */
    std::size_t traversal_memory_usage(refine_config_t config, std::size_t nodes) const noexcept {
        if (!connectivity() || !device_.multiprocessor_count) return 0;
        std::size_t const rung = kits_.rung_serving_(longest_value_);
        std::size_t const arena_bytes = refine_arena_bytes_(config, static_cast<u32_t>(span_bytes_), rung);
        return aligned_(sizeof(vram_telemetry_block_t)) + 2 * aligned_(sizeof(u32_t)) + aligned_(arena_bytes) +
               aligned_(refine_tile_bytes_(refine_tile_rows_(nodes, nodes, config, arena_bytes))) +
               aligned_(audit_bytes_(audit_use_t::refine_k, nodes));
    }

    /** The bytes an executor's run must hold before a context carved over it can split @p nodes
     *  nodes into @p receivers receivers under @p config: a refine's over the widest part, and the
     *  partition's scratch. */
    std::size_t traversal_memory_usage(refine_config_t config, std::size_t nodes,
                                       std::size_t receivers) const noexcept {
        if (!connectivity() || !device_.multiprocessor_count || receivers < 2) return 0;
        return traversal_memory_usage(config, divide_round_up(nodes, receivers)) +
               aligned_(split_layout_(nodes, receivers).device_bytes);
    }

#pragma endregion Refinement

#pragma region Checkpoints

    /**
     *  @brief One save in flight: the watermark it took, the stream its copies ride, and the nodes
     *      it hands the caller's columns through a double-buffered pinned bounce, since the device
     *      storage cannot be read in place.
     *
     *  Neighbors travel as ragged lists, every sentinel and self-loop dropped, and values through
     *  the face's codec, which may unpack them. Planning sized every node, so a call writes the
     *  slices the prefix sums name, for any nodes in any order. A call crosses half a bounce at a
     *  time, copying each half to the host while the next one copies off the device, and the last
     *  one finishes with the next call or in @c finish. The claim word stays closed from planning
     *  to @c finish or the plan's death, so batches answer @c contended_k meanwhile, while searches
     *  run throughout. The keys cross from the host at planning.
     */
    struct [[nodiscard]] save_plan_t {

        /** The nodes under the watermark, the neighbor IDs they keep, and the bytes their values
         *  save into. */
        columns_extents_t extents {};

        save_plan_t() noexcept = default;

        /** Pins its bounce through @p api and births its events on the current device. */
        explicit save_plan_t(cuda_api_t const& api) noexcept
            : bounce_ {cuda_pinned_allocator_t {api}}, crossed_ {cuda_event_t {api}, cuda_event_t {api}} {}
        save_plan_t(save_plan_t&& other) noexcept
            : extents(other.extents), neighbor_offsets_(other.neighbor_offsets_), neighbor_ids_(other.neighbor_ids_),
              value_offsets_(other.value_offsets_), values_(other.values_),
              stored_offsets_(std::move(other.stored_offsets_)), bounce_(std::move(other.bounce_)),
              crossed_ {std::move(other.crossed_[0]), std::move(other.crossed_[1])}, stream_(other.stream_),
              half_nodes_(other.half_nodes_), half_bytes_(other.half_bytes_), neighbors_bytes_(other.neighbors_bytes_),
              pending_(std::exchange(other.pending_, {})), pending_half_(other.pending_half_), covered_(other.covered_),
              index_(std::exchange(other.index_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;

        /** A save owns nothing but its bounce, so a dying plan awaits the copies into its pending
         *  half and reopens the claim word. */
        ~save_plan_t() noexcept {
            if (!pending_.empty()) [[maybe_unused]]
                status_t const drained = crossed_[pending_half_].wait();
            if (index_) index_->reopen_();
        }

        /**
         *  Copies the neighbor IDs and the values of @p nodes off the device into the slices the
         *  prefix sums name, half a bounce at a time; the last half is still copying when it
         *  returns.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The @p nodes pass the plan, or a node's kept neighbors or its
         *    saved value disagree with the offsets the plan measured.
         */
        status_t save(iota_view<node_id_t> nodes) noexcept {
            if (!index_ || neighbor_offsets_.empty()) return status_t::uninitialized_k;
            if (nodes.bound < nodes.first || nodes.bound > extents.nodes) return status_t::config_mismatch_k;
            for (node_id_t first = nodes.first; first != nodes.bound;) {
                node_id_t const bound = crossing_end_(to_span(stored_offsets_), first, nodes.bound, half_nodes_,
                                                      half_bytes_);
                // Every half but the pending one has finished, so the other one is free.
                std::size_t const half = pending_.empty() ? 0 : pending_half_ ^ 1;
                if (status_t const fetched = fetch_({first, bound}, half); failed(fetched)) return fetched;
                if (status_t const copied = copy_pending_(); failed(copied)) return copied;
                pending_ = {first, bound}, pending_half_ = half, first = bound;
            }
            return status_t::success_k;
        }

        /**
         *  @brief Pronounces the snapshot complete once the last half finished and every node did:
         *      a success spends the plan and reopens the claim word; a refusal leaves it whole, so
         *      the missing nodes can still be copied.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k Some nodes were never saved, or a node's kept neighbors or its
         *    saved value disagree with the offsets the plan measured.
         */
        status_t finish() noexcept {
            if (!index_ || neighbor_offsets_.empty()) return status_t::uninitialized_k;
            if (status_t const copied = copy_pending_(); failed(copied)) return copied;
            if (covered_ != extents.nodes) return status_t::config_mismatch_k;
            std::exchange(index_, nullptr)->reopen_();
            return status_t::success_k;
        }

      private:
        friend struct vram_ragged_graph;

        /** Queues the copies of @p nodes ' neighbor rows and stored values into half @p half of the
         *  bounce, recorded on its event. */
        status_t fetch_(iota_view<node_id_t> nodes, std::size_t half) noexcept {
            std::size_t const row_bytes = index_->connectivity() * sizeof(node_id_t);
            byte_t* const neighbors = bounce_.data() + half * (neighbors_bytes_ + half_bytes_);
            u64_t const first_byte = stored_offsets_[nodes.first];
            u64_t const bytes = stored_offsets_[nodes.bound] - first_byte;
            if (cuda_copy_async(index_->api(), neighbors, index_->neighbors_matrix_.base() + nodes.first * row_bytes,
                                nodes.size() * row_bytes, stream_) != cuda_success_k ||
                (bytes && cuda_copy_async(index_->api(), neighbors + neighbors_bytes_,
                                          index_->values_tape_.base() + first_byte, bytes, stream_) != cuda_success_k))
                return status_t::device_error_k;
            return crossed_[half].record(stream_);
        }

        /** Awaits half @p half, then hands @p visit each of @p nodes with its neighbor row and its
         *  stored value there, until one refuses. */
        template <typename visit_type_>
        status_t visit_(iota_view<node_id_t> nodes, std::size_t half, visit_type_&& visit) noexcept {
            if (status_t const drained = crossed_[half].wait(); failed(drained)) return drained;
            std::size_t const connectivity = index_->connectivity();
            byte_t const* const staged = bounce_.data() + half * (neighbors_bytes_ + half_bytes_);
            node_id_t const* const neighbors = reinterpret_cast<node_id_t const*>(staged);
            for (node_id_t node = nodes.first; node != nodes.bound; ++node) {
                std::span<node_id_t const> const slots {neighbors + (node - nodes.first) * connectivity, connectivity};
                bytes_view_t const stored = bytes_of(
                    staged + neighbors_bytes_ + stored_offsets_[node] - stored_offsets_[nodes.first],
                    stored_offsets_[node + 1] - stored_offsets_[node]);
                if (status_t const visited = visit(node, slots, stored); failed(visited)) return visited;
            }
            return status_t::success_k;
        }

        /** Sizes every node under the watermark, half a bounce at a time, writing both prefix sums
         *  when they are given, @b [nodes+1] each. */
        status_t size_(std::span<u64_t> neighbor_sums, std::span<u64_t> value_sums) noexcept {
            node_id_t const nodes = static_cast<node_id_t>(extents.nodes);
            extents.neighbors = 0, extents.value_bytes = 0;
            if (!neighbor_sums.empty()) neighbor_sums[0] = 0, value_sums[0] = 0;
            auto const sized = [&](node_id_t node, std::span<node_id_t const> slots, bytes_view_t stored) noexcept {
                auto const [saved_bytes, saved_status] = index_->kits_.saved_bytes_(stored);
                if (failed(saved_status)) return saved_status;
                for (node_id_t const neighbor : slots)
                    if (kept_link_(neighbor, node, nodes)) ++extents.neighbors;
                extents.value_bytes += saved_bytes;
                if (!neighbor_sums.empty())
                    neighbor_sums[node + 1] = extents.neighbors, value_sums[node + 1] = extents.value_bytes;
                return status_t::success_k;
            };
            for (node_id_t first = 0; first != nodes;) {
                node_id_t const bound = crossing_end_(to_span(stored_offsets_), first, nodes, half_nodes_, half_bytes_);
                if (status_t const fetched = fetch_({first, bound}, 0); failed(fetched)) return fetched;
                if (status_t const visited = visit_({first, bound}, 0, sized); failed(visited)) return visited;
                first = bound;
            }
            return status_t::success_k;
        }

        /**
         *  @brief Awaits the pending half's copies, then writes its nodes to the columns: every
         *      neighbor under the watermark but the node's own ID, and the value through the codec
         *      of the face.
         *
         *  @par Statuses:
         *  - @c config_mismatch_k A node's kept neighbors or its saved value disagree with the
         *    offsets the plan measured.
         */
        status_t copy_pending_() noexcept {
            if (pending_.empty()) return status_t::success_k;
            iota_view<node_id_t> const pending = std::exchange(pending_, {});
            node_id_t const nodes = static_cast<node_id_t>(extents.nodes);
            auto const on_staged = [&](node_id_t node, std::span<node_id_t const> slots, bytes_view_t stored) noexcept {
                u64_t cursor = neighbor_offsets_[node];
                u64_t const bound = neighbor_offsets_[node + 1];
                for (node_id_t const neighbor : slots) {
                    if (!kept_link_(neighbor, node, nodes)) continue;
                    if (cursor == bound) return status_t::config_mismatch_k;
                    neighbor_ids_[cursor++] = neighbor;
                }
                if (cursor != bound) return status_t::config_mismatch_k;
                return index_->kits_.save_value_(
                    stored, values_.subspan(value_offsets_[node], value_offsets_[node + 1] - value_offsets_[node]));
            };
            if (status_t const visited = visit_(pending, pending_half_, on_staged); failed(visited)) return visited;
            covered_ += pending.size();
            return status_t::success_k;
        }

        /** The columns the calls write but the keys, empty in a plan that only sizes. */
        std::span<u64_t> neighbor_offsets_;
        std::span<node_id_t> neighbor_ids_;
        std::span<u64_t> value_offsets_;
        bytes_span_t values_;

        /** The device's value starts into the tape, copied once at planning, @b [nodes+1]. */
        growing_buffer<u64_t> stored_offsets_ {global_allocator};

        /** Page-locked host bytes in two halves: @c half_nodes_ neighbor rows, then @c half_bytes_
         *  of stored values. */
        growing_buffer<byte_t, cuda_pinned_allocator_t> bounce_ {cuda_pinned_allocator_t {}};

        /** Marks the last copies into each half of the bounce, born with the plan on its device. */
        cuda_event_t crossed_[2];
        cuda_stream_handle_t stream_ = nullptr;
        std::size_t half_nodes_ = 0, half_bytes_ = 0, neighbors_bytes_ = 0;

        /** The nodes whose copies sit in a half not yet on the host, and that half. */
        iota_view<node_id_t> pending_ {};
        std::size_t pending_half_ = 0;

        /** Nodes whose neighbors and values were copied. */
        std::size_t covered_ = 0;

        /** The index being snapshotted, whose claim word the plan holds closed; null in a refused
         *  plan, and once @c finish succeeded. */
        vram_ragged_graph* index_ = nullptr;
    };

    /**
     *  Node & element counts a save needs right now, through one sizing pass over every node under
     *  the watermark with the claim word closed; @c contended_k while a batch, a load or a plan is
     *  in flight.
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
     *  @brief Opens a save into @p columns of every node under the watermark on @p context 's
     *      stream: closes the claim word, copies the device's value starts, pins the bounce, sizes
     *      every node and writes both prefix sums.
     *
     *  The key column's length names the node count, and planning fills it from the host keys. A
     *  column shorter than its count answers @c overflow_k, and a longer key or sums column
     *  @c config_mismatch_k, so no call ever writes past them.
     *
     *  @param[out] columns Where the nodes go, sized from @ref columns_extents.
     *  @param[in,out] context The stream every copy queues on.
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
            return {std::move(plan), status_t::overflow_k};
        plan.neighbor_offsets_ = columns.neighbor_offsets, plan.neighbor_ids_ = columns.neighbor_ids;
        plan.value_offsets_ = columns.value_offsets, plan.values_ = columns.values;
        std::size_t saved = 0;
        keyed_.column().for_each_span({0, nodes}, [&](std::span<key_t const> run) noexcept {
            for (key_t const& key : run) columns.keys[saved++] = key;
        });
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  Serializes every node under the watermark into @p columns on the calling thread - the
     *  one-shot over @ref plan_save.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p context rides another device or cannot walk, the key or a
     *    sums column is longer than its count, or a node's kept neighbors or its saved value
     *    disagree with the offsets the plan measured.
     *  - @c overflow_k A column is shorter than its count.
     *  - @c contended_k A batch, a load or a plan in flight.
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
     *      hold them, and the device staging slots their saved columns are read into.
     *
     *  Planning sizes every stored value from its saved length alone. A call reads a run of nodes'
     *  saved neighbors, offsets and values into a slot, through cuFile or a driver copy, while the
     *  other slot's work runs, copies their planned tape starts beside them, widens their neighbor
     *  rows on the device, and hands the kits the run to store its values. Keys go into the host
     *  key column. @c finish is the one full wait: it reads back whether a neighbor row was
     *  malformed and the widest span, files the keys and turns the watermark over the nodes. Holds
     *  the claim for its whole life; a plan that dies before @c finish gives its claim back and
     *  leaves the storage grown, its nodes unfilled.
     */
    struct [[nodiscard]] load_plan_t {

        /** The first appended ID, where node zero of the columns goes. */
        node_id_t base = 0;

        /** Nodes the plan appends. */
        std::size_t entries = 0;

        load_plan_t() noexcept = default;

        /** Births its events through @p api on the current device. */
        explicit load_plan_t(cuda_api_t const& api) noexcept : drained_ {cuda_event_t {api}, cuda_event_t {api}} {}
        load_plan_t(load_plan_t&& other) noexcept
            : base(other.base), entries(std::exchange(other.entries, 0)), saved_(other.saved_),
              stored_offsets_(std::move(other.stored_offsets_)),
              drained_ {std::move(other.drained_[0]), std::move(other.drained_[1])},
              device_words_(std::move(other.device_words_)), staging_(std::move(other.staging_)),
              slot_bytes_(other.slot_bytes_), holder_(std::exchange(other.holder_, nullptr)), stream_(other.stream_),
              graph_kernels_(other.graph_kernels_), kits_kernels_(other.kits_kernels_), longest_(other.longest_),
              span_bytes_(other.span_bytes_), chunks_(other.chunks_), covered_(other.covered_),
              index_(std::exchange(other.index_, nullptr)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** An unfinished plan awaits every slot's work, so the next grow fills its nodes again, and
         *  gives its claim back. */
        ~load_plan_t() noexcept {
            if (holder_) [[maybe_unused]]
                cufile_error_t const deregistered = holder_->file.api().buffer_deregister(staging_.get());
            if (!index_) return;
            if (chunks_) {
                [[maybe_unused]] status_t const first_drained = drained_[0].wait();
                [[maybe_unused]] status_t const second_drained = drained_[1].wait();
            }
            index_->filled_nodes_ = min_of(index_->filled_nodes_, static_cast<std::size_t>(base));
            index_->unclaim_(base, entries);
        }

        /**
         *  @brief Reads, widens and stores the saved nodes @p range, counted from the columns'
         *      first node, one staging slot at a time, and returns while the last slot's work runs.
         *
         *  Driver copies fill a slot in stream order; cuFile, writing from the host, refills one
         *  only once the work out of it finished.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The @p range passes the plan.
         *  - @c io_k A read through cuFile ended short, which never restarts on the host.
         */
        status_t load(iota_view<node_id_t> range) noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (range.bound < range.first || range.bound > entries) return status_t::config_mismatch_k;
            cuda_api_t const& api = index_->api();
            std::size_t const key_bytes = dtype_bytes(saved_.key_dtype);
            for (node_id_t first = range.first; first != range.bound;) {
                node_id_t const bound = staged_end_(first, range.bound);
                std::size_t const slot = chunks_++ % 2;
                if (holder_ && chunks_ > 2 && failed(drained_[slot].wait())) return status_t::device_error_k;
                std::size_t into = slot * slot_bytes_;
                auto const read = [&](bytes_view_t extent) noexcept -> expected<bytes_view_t> {
                    if (holder_)
                        return holder_->file.read(static_cast<std::size_t>(extent.data() - holder_->part.data()),
                                                  extent.size(), staging_.get(), into);
                    byte_t* const staged = staging_.get() + into;
                    into += staged_bytes_(extent.size());
                    if (cuda_copy_async(api, staged, extent.data(), extent.size(), stream_) != cuda_success_k)
                        return {{}, status_t::device_error_k};
                    return {bytes_view_t {staged, extent.size()}, status_t::success_k};
                };
                auto const [neighbors, neighbors_status] = read(neighbor_extent_(first, bound));
                if (failed(neighbors_status)) return neighbors_status;
                bytes_view_t neighbor_offsets, value_offsets;
                if (!saved_.degree) {
                    auto const [offsets, status] = read(
                        std::as_bytes(saved_.neighbor_offsets.subspan(first, bound - first + 1)));
                    if (failed(status)) return status;
                    neighbor_offsets = offsets;
                }
                if (!saved_.value_width) {
                    auto const [offsets,
                                status] = read(std::as_bytes(saved_.value_offsets.subspan(first, bound - first + 1)));
                    if (failed(status)) return status;
                    value_offsets = offsets;
                }
                std::size_t const values_first = saved_.value_first(first);
                auto const [values, values_status] = read(
                    saved_.values.subspan(values_first, saved_.value_first(bound) - values_first));
                if (failed(values_status)) return values_status;

                vram_graph<node_id_t> graph;
                graph.neighbors = reinterpret_cast<node_id_t*>(index_->neighbors_matrix_.base());
                graph.connectivity = static_cast<u32_t>(index_->connectivity());
                graph.first_id = static_cast<node_id_t>(base + first);
                graph.nodes_count = static_cast<u32_t>(bound - first);
                graph.shift = base;
                graph.source_bound = static_cast<node_id_t>(entries);
                graph.saved_neighbors = reinterpret_cast<node_id_t const*>(neighbors.data());
                graph.saved_offsets = reinterpret_cast<u64_t const*>(neighbor_offsets.data());
                graph.degree = static_cast<u32_t>(saved_.degree);
                graph.malformed = device_words_.get();
                vram_load_step<node_id_t> step;
                step.arguments.load = {values.data(),
                                       reinterpret_cast<u64_t const*>(value_offsets.data()),
                                       reinterpret_cast<u64_t const*>(index_->offsets_column_.base()) + base + first,
                                       index_->values_tape_.base(),
                                       device_words_.get() + 1,
                                       static_cast<u32_t>(bound - first)};
                step.saved = values;
                step.stored = index_->values_tape_.base() + stored_offsets_[first];
                step.api = &api, step.kits = kits_kernels_, step.stream = stream_;
                step.multiprocessors = index_->device_.multiprocessor_count;
                // The planned tape starts cross per run, 8 bytes a node, from the host's plan.
                if (cuda_copy_async(api, index_->offsets_column_.base() + (base + first) * sizeof(u64_t),
                                    stored_offsets_.data() + first, (bound - first + 1) * sizeof(u64_t),
                                    stream_) != cuda_success_k ||
                    failed(index_->launch_graph_(*graph_kernels_, vram_kernel_t::widen_links_k,
                                                 index_->graph_blocks_(vram_kernel_t::widen_links_k), graph,
                                                 stream_)) ||
                    failed(index_->kits_.queue_load_values_(step)) || failed(drained_[slot].record(stream_)))
                    return status_t::device_error_k;
                for (node_id_t node = first; node != bound; ++node)
                    index_->keyed_.assign(
                        static_cast<node_id_t>(base + node),
                        safegraphs_widened_key<key_t>(saved_.keys.data() + node * key_bytes, saved_.key_dtype));
                covered_ += bound - first, first = bound;
            }
            return status_t::success_k;
        }

        /**
         *  @brief Publishes the appended nodes once every one was copied: the tape's extent, the
         *      longest value and the widest span grow over them, their keys file under the claim,
         *      and the watermark turns, which lifts the claim.
         *
         *  A success spends the plan; a refusal leaves it whole, so the missing nodes can still be
         *  copied over later, or its death gives the claim back.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k Some nodes were never copied.
         *  - @c malformed_k A saved neighbor named no node of the columns, or a row kept more
         *    neighbors than the connectivity.
         */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (covered_ != entries) return status_t::config_mismatch_k;
            if (failed(drained_[0].wait()) || failed(drained_[1].wait())) return status_t::device_error_k;
            u64_t words[2] = {0, 0};
            if (entries && cuda_copy(index_->api(), words, device_words_.get(), sizeof(words)) != cuda_success_k)
                return status_t::device_error_k;
            if (words[0]) return status_t::malformed_k;
            vram_ragged_graph& index = *std::exchange(index_, nullptr);
            index.tape_bytes_ = stored_offsets_[entries];
            index.longest_value_ = max_of(index.longest_value_, longest_);
            index.span_bytes_ = max_of(index.span_bytes_, max_of(span_bytes_, static_cast<std::size_t>(words[1])));
            index.publish_keys_({base, static_cast<node_id_t>(base + entries)});
            fu::standard_atomic_ref<std::size_t>(index.last_visible_id_)
                .store(base + entries, std::memory_order_release);
            return status_t::success_k;
        }

      private:
        friend struct vram_ragged_graph;

        /** Bytes a read of @p bytes takes in a slot: whole blocks, and one more for its lead. */
        static std::size_t staged_bytes_(std::size_t bytes) noexcept {
            return round_up_to_multiple(bytes + linux_cufile_file_t::block_bytes_k, linux_cufile_file_t::block_bytes_k);
        }

        /** The saved neighbor IDs of nodes @b [first,bound): fixed rows, or their run of IDs. */
        bytes_view_t neighbor_extent_(std::size_t first, std::size_t bound) const noexcept {
            std::span<node_id_t const> const ids =
                saved_.degree
                    ? saved_.neighbor_ids.subspan(first * saved_.degree, (bound - first) * saved_.degree)
                    : saved_.neighbor_ids.subspan(saved_.neighbor_offsets[first],
                                                  saved_.neighbor_offsets[bound] - saved_.neighbor_offsets[first]);
            return std::as_bytes(ids);
        }

        /** Slot bytes the saved columns of nodes @b [first,bound) take. */
        std::size_t run_bytes_(std::size_t first, std::size_t bound) const noexcept {
            std::size_t const sums_bytes = (bound - first + 1) * sizeof(u64_t);
            return staged_bytes_(neighbor_extent_(first, bound).size()) +
                   (saved_.degree ? 0 : staged_bytes_(sums_bytes)) +
                   (saved_.value_width ? 0 : staged_bytes_(sums_bytes)) +
                   staged_bytes_(saved_.value_first(bound) - saved_.value_first(first));
        }

        /** The end of the run from @p first toward @p bound whose saved columns fit one slot, one
         *  node at least, which planning sized every slot for. */
        node_id_t staged_end_(node_id_t first, node_id_t bound) const noexcept {
            node_id_t end = static_cast<node_id_t>(first + 1);
            while (end != bound && run_bytes_(first, end + 1) <= slot_bytes_) ++end;
            return end;
        }

        /** The saved columns, read in place from the mapping or the lent buffer. */
        safegraphs_saved_columns<node_id_t> saved_;

        /** Each node's start in the tape, from the plan's first, @b [nodes+1]. */
        growing_buffer<u64_t> stored_offsets_ {global_allocator};

        /** Marks the last work out of each staging slot, born with the plan on its device. */
        cuda_event_t drained_[2];

        /** On the device: the malformed word, nonzero once a neighbor row is, then the widest span
         *  the kits counted. */
        cuda_unique_ptr<u64_t> device_words_;

        /** The slots runs cross, alternating, back to back, and the bytes each holds. */
        cuda_unique_ptr<byte_t> staging_;
        std::size_t slot_bytes_ = 0;

        /** The registered part the columns are read from by cuFile, whose entries registered the
         *  staging; null where they cross by driver copies. */
        cuda_part_source_t const* holder_ = nullptr;

        cuda_stream_handle_t stream_ = nullptr;

        /** The shared and the kits' kernels on the plan's stream, which outlive the plan there. */
        kernel_table_t const* graph_kernels_ = nullptr;
        kernel_table_t const* kits_kernels_ = nullptr;

        /** The longest appended node and the widest span its length bounds. */
        std::size_t longest_ = 0;
        std::size_t span_bytes_ = 0;

        /** Runs staged so far, and nodes copied. */
        std::size_t chunks_ = 0;
        std::size_t covered_ = 0;

        /** The index appended to, whose claim the plan holds; null in a refused plan, and once
         *  @c finish published. */
        vram_ragged_graph* index_ = nullptr;
    };

    /**
     *  @brief Opens a load of the @p saved columns past every claimed node on @p context 's stream:
     *      vets their offsets columns, sizes every stored value from its saved length, claims the
     *      nodes, grows the storage and opens the staging slots on @p context 's stream.
     *
     *  The keys name the node count and go at @c base on, beside the nodes. Only the offsets
     *  columns are read on the host: the device checks every neighbor ID, and @c finish refuses a
     *  load where one names no node of the columns, or where a fixed row keeps more neighbors than
     *  its connectivity allows.
     *
     *  @param[in] saved The columns as the file holds them, saved by this face or any residency.
     *  @param[in,out] context The stream every copy and kernel queues on.
     *  @param[in] sources The mapped parts the columns are read from, empty for a lent buffer.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk.
     *  - @c malformed_k An offsets column is not one longer than the nodes, its sums pass its
     *    column, a ragged node keeps more neighbors than the connectivity, or a saved value does
     *    not parse.
     *  - @c unsupported_k A value past every rung of the face's kits.
     *  - @c overflow_k The nodes pass the ceiling or the reserved range, or their values the
     *    reserved tape.
     *  - @c contended_k A batch, a load or a plan in flight.
     */
    expected<load_plan_t> plan_load(safegraphs_saved_columns<node_id_t> const& saved, traversal_context_t& context,
                                    std::span<cuda_part_source_t const> sources = {}) noexcept {
        if (cuda_set_device(api(), device_.ordinal) != cuda_success_k) return {{}, status_t::device_error_k};
        load_plan_t plan {api()};
        std::size_t const connectivity = this->connectivity(), entries = saved.nodes();
        if (!connectivity) return {std::move(plan), status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {std::move(plan), status_t::config_mismatch_k};
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
        if (!neighbors_vetted || !values_vetted) return {std::move(plan), status_t::malformed_k};
        for (std::size_t node = 0; !saved.degree && node != entries; ++node)
            if (saved.neighbor_offsets[node + 1] - saved.neighbor_offsets[node] > connectivity)
                return {std::move(plan), status_t::malformed_k};

        // Every stored value's size up front, from its saved length alone, so the tape and the
        // runs are known before the claim.
        if (failed(plan.stored_offsets_.resize_uninitialized(entries + 1)))
            return {std::move(plan), status_t::bad_alloc_k};
        plan.stored_offsets_[0] = 0;
        for (std::size_t node = 0; node != entries; ++node) {
            auto const [stored_bytes,
                        stored_status] = kits_.stored_bytes_(saved.value_first(node + 1) - saved.value_first(node));
            if (failed(stored_status)) return {std::move(plan), stored_status};
            plan.stored_offsets_[node + 1] = plan.stored_offsets_[node] + stored_bytes;
            plan.longest_ = max_of<std::size_t>(plan.longest_, stored_bytes);
        }
        if (entries && kits_.rung_serving_(max_of(plan.longest_, longest_value_)) == kits_type_::rungs_k)
            return {std::move(plan), status_t::unsupported_k};
        auto const [graph_kernels, graph_status] = graph_kernels_(context);
        if (failed(graph_status)) return {std::move(plan), graph_status};
        auto const [kits_kernels, kits_status] = kits_kernels_(context);
        if (failed(kits_status)) return {std::move(plan), kits_status};

        auto const [base, claim_status] = claim_(entries);
        if (failed(claim_status)) return {std::move(plan), claim_status};
        plan.index_ = this, plan.base = base, plan.entries = entries;
        cuda_stream_handle_t const stream = context.stream();
        std::size_t const stored_total = plan.stored_offsets_[entries];
        if (status_t const grown = grow_nodes_(base + entries, stream); failed(grown)) return {std::move(plan), grown};
        if (status_t const mapped = values_tape_.commit(tape_bytes_ + stored_total + tape_margin_k); failed(mapped))
            return {std::move(plan), mapped};
        for (u64_t& offset : plan.stored_offsets_) offset += tape_bytes_;
        plan.saved_ = saved, plan.stream_ = stream;
        plan.graph_kernels_ = graph_kernels, plan.kits_kernels_ = kits_kernels;
        plan.span_bytes_ = kits_.span_bytes_(plan.longest_);
        if (!entries) return {std::move(plan), status_t::success_k};

        // A slot fits the run a save's crossing carries, and at least the widest node; a second
        // slot opens only for a second run.
        plan.slot_bytes_ = plan.run_bytes_(0, min_of(entries, power_of_two<std::size_t>(crossing_nodes_log2_k)));
        for (std::size_t node = 0; node != entries; ++node)
            plan.slot_bytes_ = max_of(plan.slot_bytes_, plan.run_bytes_(node, node + 1));
        std::size_t const staging_bytes = (plan.run_bytes_(0, entries) > plan.slot_bytes_ ? 2 : 1) * plan.slot_bytes_;
        plan.staging_ = cuda_make_unique(cuda_device_allocator_t {context.allocator()}, staging_bytes);
        if (!plan.staging_) return {std::move(plan), status_t::bad_alloc_k};
        plan.holder_ = cufile_holder(sources, std::as_bytes(saved.neighbor_ids));
        // cuFile writes from the host, so the stream's allocation must finish first.
        if (plan.holder_ && (failed(plan.drained_[0].record(stream)) || failed(plan.drained_[0].wait())))
            return {std::move(plan), status_t::device_error_k};
        if (plan.holder_) [[maybe_unused]]
            cufile_error_t const registered = plan.holder_->file.api().buffer_register(plan.staging_.get(),
                                                                                       staging_bytes, 0);
        plan.device_words_ = cuda_make_unique(cuda_device_allocator<u64_t> {api()}, 2);
        if (!plan.device_words_) return {std::move(plan), status_t::bad_alloc_k};
        if (cuda_fill_async(api(), plan.device_words_.get(), 0, 2 * sizeof(u64_t), stream) != cuda_success_k)
            return {std::move(plan), status_t::device_error_k};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Loads the @p saved columns past every claimed node on the calling thread - the
     *      one-shot over @ref plan_load.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk.
     *  - @c malformed_k An offsets column is out of shape, a saved value does not parse, a neighbor
     *    names no node of the columns, or a row keeps more neighbors than the connectivity.
     *  - @c unsupported_k A value past every rung of the face's kits.
     *  - @c overflow_k The nodes pass the ceiling or the reserved range, or their values the
     *    reserved tape.
     *  - @c contended_k A batch, a load or a plan in flight.
     */
    status_t load(safegraphs_saved_columns<node_id_t> const& saved, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_load(saved, context);
        if (failed(status)) return status;
        if (status_t const loaded = plan.load({0, static_cast<node_id_t>(plan.entries)}); failed(loaded)) return loaded;
        return plan.finish();
    }

#pragma endregion Checkpoints

  private:

    /** Bytes of zeroed margin past the tape's last value, so a word load ending there never
     *  faults. */
    inline static constexpr std::size_t tape_margin_k = sizeof(u64_t);

    /** Threads per block, one warp's worth per resident warp. */
    inline static constexpr u32_t block_threads_k = vram_warps_per_block_k * 32;

    /** Shared bytes a kernel keeps beside its scratch: four warps' batches of three lane-widths of
     *  IDs, a round of two scored together and its carry, three per thread. */
    inline static constexpr std::size_t static_shared_bytes_k = 3 * block_threads_k * sizeof(node_id_t);

    /** Lanes per warp, the width every kernel's collectives span. */
    static constexpr u32_t wave_lanes() noexcept { return 32; }

    static std::size_t aligned_(std::size_t bytes) noexcept {
        return divide_round_up<std::size_t>(bytes, default_alignment_k) * default_alignment_k;
    }

    /** Bytes of a batch's staged offsets, whole words of the chains' alignment. */
    static std::size_t offsets_bytes_(std::size_t entries) noexcept { return (entries + 1) * sizeof(u64_t); }

    /** A batch's insertion staging: its offsets, then a chain head over every node the batch leaves
     *  behind, a next link per pick, the touched list as long, and its count. */
    std::size_t build_staging_bytes_(std::size_t entries, std::size_t total_nodes) const noexcept {
        return offsets_bytes_(entries) + (total_nodes + 2 * entries * connectivity()) * sizeof(node_id_t) +
               sizeof(u32_t);
    }

    /** A search's staging: the queries' bytes and margin in whole words, their offsets, then the
     *  matched IDs and distances. */
    static std::size_t search_staging_bytes_(std::size_t queries, std::size_t bytes, std::size_t count) noexcept {
        return round_up_to_multiple<std::size_t>(bytes + tape_margin_k, sizeof(u64_t)) + offsets_bytes_(queries) +
               queries * count * (sizeof(node_id_t) + sizeof(distance_t));
    }

    /** Walkers one block hosts on the kit rung @p rung: a warp each when a lane scores a node
     *  alone, one walking the whole block otherwise. */
    static u32_t walkers_per_block_(std::size_t rung) noexcept {
        return rung < kits_type_::rungs_k && kits_type_::lanes_per_entry_(rung) > 1 ? 1u : vram_warps_per_block_k;
    }

    /** Blocks a walk over @p queries launches on the kit rung @p rung and the register rung
     *  @p slots: a walker per query up to the grid the device runs at once. */
    u32_t walk_blocks_(std::size_t queries, std::size_t rung, u32_t slots) const noexcept {
        return static_cast<u32_t>(min_of<std::size_t>(
            divide_round_up<std::size_t>(queries, walkers_per_block_(rung)),
            std::size_t {device_.multiprocessor_count} * cuda_resident_blocks(vram_kernel_t::search_k, slots)));
    }

    /** Which frontier a walk keeps: a register rung, or the arena's past the ladder. */
    enum class frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

    /** The breadth an insert or a refine walks: as given, or twice the connectivity, as on the
     *  CPU, and never under the @c build_breadth_k a modality's selector needs. */
    u32_t build_expansion_(u32_t expansion_factor) const noexcept {
        if (expansion_factor) return expansion_factor;
        u32_t const twice = static_cast<u32_t>(default_expansion_factor(connectivity()));
        if constexpr (requires { kits_type_::build_breadth_k; }) return max_of(twice, kits_type_::build_breadth_k);
        else return twice;
    }

    /** Candidates a walk's selection caches: all of its candidates, @p candidate_slots ranks per
     *  lane on the register ladder, or the arena frontier's @p expansion past it. */
    static u32_t walk_selection_nodes_(u32_t expansion, u32_t candidate_slots) noexcept {
        return candidate_slots ? candidate_slots * wave_lanes() : expansion;
    }

    /** Candidates a back-link round's re-selection caches: the node's live row and a row's worth of
     *  appended nodes; the node itself prepares into the query span. */
    u32_t reselection_nodes_() const noexcept { return 2 * static_cast<u32_t>(connectivity()); }

    /** Bytes of one building warp's scratch over @p selection_nodes candidates: the query span and
     *  the pivot span of @p span_bytes each, then per candidate its score, its two node-ID slots,
     *  its kept position and its distance, in whole 16-byte words. */
    static u32_t build_scratch_bytes_(u32_t span_bytes, u32_t selection_nodes) noexcept {
        std::size_t const candidate_bytes = gpu_scored_node_bytes<node_id_t> + 2 * sizeof(node_id_t) + sizeof(u32_t) +
                                            sizeof(f32_t);
        return static_cast<u32_t>(
            round_up_to_multiple<std::size_t>(2 * span_bytes + selection_nodes * candidate_bytes, 16));
    }

    /** Everything one building pass derives from its requested expansion, once, for its records:
     *  the walk's breadth and register rung, the candidates its selection and a back-link round
     *  hold, each launch's per-warp shared scratch, zero when the arena slice keeps both, and that
     *  slice. */
    struct build_shape_t {
        u32_t expansion = 0, candidate_slots = 0, walk_selection_nodes = 0, reselection_nodes = 0;
        u32_t walk_scratch_bytes = 0, back_link_scratch_bytes = 0;
        gpu_arena_shape_t arena_shape {};
    };

    /** The shape of a building pass requesting @p expansion_factor over nodes preparing into
     *  @p span_bytes on the kit rung @p rung. */
    build_shape_t build_shape_(u32_t expansion_factor, u32_t span_bytes, std::size_t rung) const noexcept {
        build_shape_t shape;
        shape.expansion = build_expansion_(expansion_factor);
        shape.candidate_slots = gpu_candidate_slots(shape.expansion, 2);
        shape.walk_selection_nodes = walk_selection_nodes_(shape.expansion, shape.candidate_slots);
        shape.reselection_nodes = reselection_nodes_();
        u32_t const walk_bytes = build_scratch_bytes_(span_bytes, shape.walk_selection_nodes);
        u32_t const back_link_bytes = build_scratch_bytes_(span_bytes, shape.reselection_nodes);
        // Every resident block of a builder takes its own scratch, so the most any builder keeps
        // resident must fit on one multiprocessor, or the launch keeps fewer blocks resident than
        // its grid and arena were sized for.
        std::size_t const block_bytes = walkers_per_block_(rung) * max_of(walk_bytes, back_link_bytes) +
                                        static_shared_bytes_k;
        bool const fits = block_bytes <= device_.shared_memory_per_block_bytes &&
                          cuda_resident_blocks(vram_kernel_t::insert_k, 0) *
                                  (block_bytes + device_.reserved_shared_memory_per_block_bytes) <=
                              device_.shared_memory_per_multiprocessor_bytes;
        shape.walk_scratch_bytes = fits ? walk_bytes : 0;
        shape.back_link_scratch_bytes = fits ? back_link_bytes : 0;
        shape.arena_shape = arena_shape_for_(
            shape.walk_selection_nodes,
            shape.candidate_slots ? frontier_kind_t::register_ladder_k : frontier_kind_t::wide_arena_k,
            fits ? 0 : max_of(walk_bytes, back_link_bytes));
        return shape;
    }

    /** One warp's arena slice for a walk keeping @p candidate_nodes candidates: the exact visited
     *  set over every ID it claims, plus the spill tail of a wide frontier, what its expansions
     *  stage, then any scratch shared memory cannot fit. */
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
        u32_t const count_slots = divide_round_up<u32_t>(config.count, wave_lanes());
        return gpu_candidate_slots(walk_expansion_(config), count_slots > 2 ? count_slots : 2);
    }

    /** One walker's slice: the register rung's visited set, or the arena frontier's with its tail
     *  past the ladder, then the query span when shared memory cannot fit it. */
    gpu_arena_shape_t walk_arena_shape_(search_config_t config, u32_t span_bytes, std::size_t rung) const noexcept {
        u32_t const expansion = walk_expansion_(config);
        bool const fits = walkers_per_block_(rung) * span_bytes + static_shared_bytes_k <=
                          device_.shared_memory_per_block_bytes;
        u32_t const scratch_bytes = fits ? 0 : span_bytes;
        u32_t const slots = walk_slots_(config);
        return arena_shape_for_(walk_selection_nodes_(expansion, slots),
                                slots ? frontier_kind_t::register_ladder_k : frontier_kind_t::wide_arena_k,
                                scratch_bytes);
    }

    /** The widest span any node of @p tape prepares into, raising @p longest to the longest value
     *  among them. */
    std::size_t span_bytes_of_(tape_view<byte_t const> tape, std::size_t& longest) const noexcept {
        std::size_t widest = 0;
        for (std::size_t node = 0; node != tape.size(); ++node) {
            bytes_view_t const entry = tape[node];
            longest = max_of(longest, entry.size());
            widest = max_of(widest, kits_.span_bytes_(entry.data(), entry.size()));
        }
        return widest;
    }

    /** Keeps @p failure as the status unless one is kept already; returns it for the caller's own
     *  report. */
    status_t fail_(status_t failure) noexcept {
        status_t expected = status_t::success_k;
        fu::standard_atomic_ref<status_t>(status_).compare_exchange_strong(expected, failure, std::memory_order_acq_rel,
                                                                           std::memory_order_acquire);
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
     *  Covers the host keys and maps the offsets column and the neighbors matrix up to @p nodes,
     *  host work alone, so it rides no stream, and leaves the fresh neighbor rows for
     *  @c grow_nodes_ to fill.
     *
     *  @par Statuses:
     *  - @c overflow_k The @p nodes pass the reserved range or the host keys.
     */
    status_t map_nodes_(std::size_t nodes) noexcept {
        if (status_t const covered = keyed_.cover(nodes); failed(covered)) return covered;
        if (nodes <= capacity()) return status_t::success_k;
        if (status_t const mapped = offsets_column_.commit((nodes + 1) * sizeof(u64_t)); failed(mapped)) return mapped;
        return neighbors_matrix_.commit(nodes * connectivity() * sizeof(node_id_t));
    }

    /** Maps up to @p nodes and sentinel-fills on @p stream every mapped neighbor row no fill
     *  reached yet, @p stream first following the last fill, whichever stream queued it; callers
     *  hold the claim. */
    status_t grow_nodes_(std::size_t nodes, cuda_stream_handle_t stream) noexcept {
        if (status_t const mapped = map_nodes_(nodes); failed(mapped)) return mapped;
        cuda_api_t const& driver = api();
        if (driver.stream_wait_event(stream, grown_, 0) != cuda_success_k) return status_t::device_error_k;
        if (nodes <= filled_nodes_) return status_t::success_k;
        // The sentinel is all ones at any width, so 32-bit words fill IDs of either.
        std::size_t const mapped_nodes = capacity(), row_words = connectivity() * sizeof(node_id_t) / sizeof(u32_t);
        if (driver.fill_words_async(cuda_address(neighbors_matrix_.base() + filled_nodes_ * row_words * sizeof(u32_t)),
                                    0xFFFFFFFFu, (mapped_nodes - filled_nodes_) * row_words,
                                    stream) != cuda_success_k ||
            failed(grown_.record(stream)))
            return status_t::device_error_k;
        filled_nodes_ = mapped_nodes;
        return status_t::success_k;
    }

    /**
     *  @brief Launches the batch @p build stages, whose @p offsets the caller spelled, from
     *      @p first_id on in waves, each the share @c vram_basic_admission_share_log2_k of the IDs
     *      below it, every launch sized by @p shape.
     *
     *  Each wave queues behind the last and admits at least @c vram_admission_ramp_floor_k nodes.
     *  Every wave walks twice, inserting and then re-selecting, and each walk is followed by its
     *  back-link launch over every node its picks touched. A failure after any kernel launched
     *  keeps the failure, since those nodes can no longer be un-claimed.
     */
    status_t launch_insert_(kernel_table_t const& kits, vram_build<node_id_t> const& build, build_shape_t const& shape,
                            std::span<std::size_t const> offsets, node_id_t first_id, std::size_t rung,
                            cuda_stream_handle_t stream) noexcept {
        std::size_t const entries = offsets.size() - 1;
        node_id_t const end_id = static_cast<node_id_t>(first_id + entries);
        // Every launch carves its warps' slices from the one arena, sized for the insert's grid,
        // which the re-selection's and the back-link's resident counts match.
        static_assert(
            cuda_resident_blocks(vram_kernel_t::refine_k, 0) == cuda_resident_blocks(vram_kernel_t::insert_k, 0) &&
            cuda_resident_blocks(vram_kernel_t::back_link_k, 0) == cuda_resident_blocks(vram_kernel_t::insert_k, 0));
        u32_t const blocks = device_.multiprocessor_count *
                             cuda_resident_blocks(vram_kernel_t::insert_k, shape.candidate_slots);
        vram_arguments<node_id_t> arguments {};
        status_t launched = status_t::success_k;
        bool published = false;
        auto const queue = [&](vram_kernel_t kernel, u32_t slots, node_id_t first) noexcept {
            bool const walks = kernel != vram_kernel_t::back_link_k;
            arguments.build.first_id = first;
            arguments.build.selection_nodes = walks ? shape.walk_selection_nodes : shape.reselection_nodes;
            arguments.build.scratch_bytes_per_warp = walks ? shape.walk_scratch_bytes : shape.back_link_scratch_bytes;
            if (succeeded(launched))
                launched = cuda_fill_async(api(), build.assigned_nodes, 0, sizeof(u32_t), stream) == cuda_success_k
                               ? cuda_launch(api(), kits_function_at_(kits, rung, kernel, slots), blocks,
                                             block_threads_k,
                                             walkers_per_block_(rung) * arguments.build.scratch_bytes_per_warp, stream,
                                             cuda_overlap_t::serialized_k, arguments)
                               : status_t::device_error_k;
            published = published || succeeded(launched);
        };
        auto const zero_touched = [&]() noexcept {
            if (succeeded(launched) &&
                cuda_fill_async(api(), build.touched_count, 0, sizeof(u32_t), stream) != cuda_success_k)
                launched = status_t::device_error_k;
        };
        for (node_id_t wave_first = first_id; wave_first != end_id;) {
            node_id_t const wave_end = static_cast<node_id_t>(
                vram_admission_wave_end(wave_first, end_id, vram_basic_admission_share_log2_k));
            // The wave reads its own nodes' starts, from where its first byte went.
            std::size_t const offset = wave_first - first_id;
            arguments.build = build;
            arguments.build.end_id = wave_end;
            arguments.build.sampling_bound = wave_end;
            arguments.build.staged_offsets = build.staged_offsets + offset;
            arguments.build.tape_first = build.tape_first + (offsets[offset] - offsets[0]);
            for (vram_kernel_t const walk : {vram_kernel_t::insert_k, vram_kernel_t::refine_k}) {
                // The re-selections take the batch's second run of epochs.
                arguments.build.epoch_base = build.epoch_base + static_cast<u32_t>(offset) +
                                             (walk == vram_kernel_t::refine_k ? static_cast<u32_t>(entries) : 0u);
                zero_touched();
                queue(walk, shape.candidate_slots, wave_first);
                queue(vram_kernel_t::back_link_k, 0, wave_first);
            }
            if (failed(launched)) return published ? fail_(launched) : launched;
            wave_first = wave_end;
        }
        return status_t::success_k;
    }

    /** Nodes one half of a save's or a load's bounce carries at most, so its two halves pin a few
     *  hundred megabytes at the widest nodes. */
    inline static constexpr u32_t crossing_nodes_log2_k = 19;

    /** Bytes of stored values one half of a bounce holds: a half's share of @p total over @p nodes,
     *  and never less than the @p longest value, so every crossing carries at least one. */
    static std::size_t crossing_bytes_(std::size_t total, std::size_t longest, std::size_t nodes,
                                       std::size_t half_nodes) noexcept {
        if (!nodes) return 0;
        return aligned_(max_of(longest, divide_round_up(total * half_nodes, nodes)));
    }

    /** The end of the crossing from @p first toward @p bound: the nodes one half holds,
     *  @p half_nodes at most and @p half_bytes of the values @p offsets lay out at most, and one at
     *  least. */
    static node_id_t crossing_end_(std::span<u64_t const> offsets, node_id_t first, node_id_t bound,
                                   std::size_t half_nodes, std::size_t half_bytes) noexcept {
        node_id_t end = static_cast<node_id_t>(first + 1);
        while (end != bound && static_cast<std::size_t>(end - first) < half_nodes &&
               offsets[end + 1] - offsets[first] <= half_bytes)
            ++end;
        return end;
    }

    /** Whether slot value @p neighbor of node @p node is a link a snapshot keeps: below @p nodes,
     *  and not the node's own ID, which marks a published node with none. */
    static bool kept_link_(node_id_t neighbor, node_id_t node, node_id_t nodes) noexcept {
        return neighbor < nodes && neighbor != node;
    }

    /** A column of @p given entries where @p needed belong: @c overflow_k short,
     *  @c config_mismatch_k long. */
    static status_t sized_exactly_(std::size_t given, std::size_t needed) noexcept {
        return given < needed   ? status_t::overflow_k
               : given > needed ? status_t::config_mismatch_k
                                : status_t::success_k;
    }

    /** Prefix sums from zero, never falling, ending within the @p elements they index;
     *  @c malformed_k otherwise. */
    static status_t vetted_sums_(std::span<u64_t const> sums, std::size_t elements) noexcept {
        if (sums.empty() || sums[0] != 0 || sums.back() > elements) return status_t::malformed_k;
        for (std::size_t position = 1; position != sums.size(); ++position)
            if (sums[position] < sums[position - 1]) return status_t::malformed_k;
        return status_t::success_k;
    }

    /** Opens a save of every node under the watermark with the claim word closed, the index's
     *  device selected first, where the plan's events are born: copies the device's value starts on
     *  the host and pins the bounce, sizing nothing yet. */
    expected<save_plan_t> open_save_(traversal_context_t& context) noexcept {
        if (cuda_set_device(api(), device_.ordinal) != cuda_success_k) return {{}, status_t::device_error_k};
        save_plan_t plan {api()};
        if (context.device() != device_.ordinal || !context) return {std::move(plan), status_t::config_mismatch_k};
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        plan.index_ = this;
        std::size_t const entries = size();
        plan.extents.nodes = entries;
        plan.stream_ = context.stream();
        if (failed(plan.stored_offsets_.resize_uninitialized(entries + 1)))
            return {std::move(plan), status_t::bad_alloc_k};
        plan.stored_offsets_[0] = 0;
        // Every batch drained before publishing, so the column holds every start under the
        // watermark.
        if (entries && cuda_copy(api(), plan.stored_offsets_.data(), offsets_column_.base(),
                                 (entries + 1) * sizeof(u64_t)) != cuda_success_k)
            return {std::move(plan), status_t::device_error_k};
        plan.half_nodes_ = min_of(entries, power_of_two<std::size_t>(crossing_nodes_log2_k));
        plan.half_bytes_ = crossing_bytes_(tape_bytes_, longest_value_, entries, plan.half_nodes_);
        plan.neighbors_bytes_ = aligned_(plan.half_nodes_ * connectivity() * sizeof(node_id_t));
        if (entries && failed(plan.bounce_.resize_uninitialized(2 * (plan.neighbors_bytes_ + plan.half_bytes_))))
            return {std::move(plan), status_t::bad_alloc_k};
        return {std::move(plan), status_t::success_k};
    }

    /** The kits' loaded @p kernel of @p kits on kit rung @p rung and register rung @p slots. */
    static cuda_function_handle_t kits_function_at_(kernel_table_t const& kits, std::size_t rung, vram_kernel_t kernel,
                                                    u32_t slots) noexcept {
        return kits.functions[cuda_kits_kernel_speller_t::index(rung, kernel, slots)];
    }

    /** Spells the shared @p kernel over this graph's node-ID type, as @ref cuda_graph_fatbin_t
     *  exports it. */
    static void spell_graph_kernel_(vram_kernel_symbol_t& symbol, vram_kernel_t kernel) noexcept {
        vram_kernel_symbol(symbol, "basic", "cuda_kernel", {"vram_graph", 1}, kernel, itanium_type<node_id_type_> {});
    }

    /**
     *  @brief The shared kernels on @p context 's stream, loaded there at the first ask and kept
     *      until the stream dies, for a refinement, an absorb, an audit or a gather over the
     *      neighbors matrix; every such verb resolves them once per call.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet, or the @p context rides no
     *    executor stream.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk.
     *  - @c unsupported_k The shared kernels lack this node-ID type.
     *  - @c device_error_k The fatbin would not load.
     */
    expected<kernel_table_t const*> graph_kernels_(traversal_context_t const& context) const noexcept {
        if (!connectivity()) return {nullptr, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {nullptr, status_t::config_mismatch_k};
        auto const [kernels, load_status] = context.table_of(cuda_graph_fatbin_t::bytes(),
                                                             cuda_graph_kernel_speller_t {&spell_graph_kernel_});
        if (failed(load_status)) return {nullptr, load_status};
        for (vram_kernel_t const kernel : cuda_graph_kernels)
            if (!kernels->functions[cuda_graph_kernel_speller_t::index(kernel)])
                return {nullptr, status_t::unsupported_k};
        return {kernels, status()};
    }

    /** Blocks one grid-striding pass of @p kernel over the neighbors matrix launches. */
    u32_t graph_blocks_(vram_kernel_t kernel) const noexcept {
        return device_.multiprocessor_count * cuda_resident_blocks(kernel, 0);
    }

    /** Launches the shared @p kernel of @p graph_kernels over @p blocks blocks on @p stream. */
    status_t launch_graph_(kernel_table_t const& graph_kernels, vram_kernel_t kernel, u32_t blocks,
                           vram_graph<node_id_t> const& graph, cuda_stream_handle_t stream) noexcept {
        return cuda_launch(api(), graph_kernels.functions[cuda_graph_kernel_speller_t::index(kernel)], blocks,
                           block_threads_k, 0, stream, cuda_overlap_t::serialized_k, graph);
    }

    /** An unbound refinement over @p context 's allocator, which a refused plan answers with. */
    static refine_plan_t unbound_plan_(traversal_context_t& context) noexcept {
        return refine_plan_t {unbound_scratch_(context)};
    }

    /** Bytes of an audit's scratch over @p nodes nodes, its heal's buffers and the changed bits too
     *  for a refine. */
    static std::size_t audit_bytes_(audit_use_t use, std::size_t nodes) noexcept {
        std::size_t const words = divide_round_up<std::size_t>(nodes, 32);
        std::size_t const heal_words = use == audit_use_t::refine_k ? 2 * nodes + words : 0;
        return (words * 3 + 7 + heal_words) * sizeof(u32_t);
    }

    /** Bytes of a sweep tile of @p tile_rows rows: a run offset per row, and a row's worth of links
     *  on its tape per row. */
    std::size_t refine_tile_bytes_(std::size_t tile_rows) const noexcept {
        return tile_rows * (connectivity() + 1) * sizeof(node_id_t);
    }

    /** Bytes of the arena a refine walk preparing into @p span_bytes on the kit rung @p rung takes
     *  under @p config. */
    std::size_t refine_arena_bytes_(refine_config_t config, u32_t span_bytes, std::size_t rung) const noexcept {
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   cuda_resident_blocks(vram_kernel_t::refine_k, 0) * vram_warps_per_block_k;
        return slices *
               build_shape_(config.expansion_factor, span_bytes, rung).arena_shape.template slice_bytes<node_id_t>();
    }

    /** Rows one sweep tile of a pass over @p nodes nodes re-selecting @p forward_rows of them takes
     *  under @p config: as many as its limit holds past the audit's scratch and the @p arena_bytes
     *  arena, every one without a limit; never more than the tape's offsets count, and zero when
     *  the limit holds not one. */
    std::size_t refine_tile_rows_(std::size_t forward_rows, std::size_t nodes, refine_config_t config,
                                  std::size_t arena_bytes) const noexcept {
        std::size_t const row_bytes = refine_tile_bytes_(1);
        std::size_t const capped = min_of<std::size_t>(forward_rows, node_id_limits<u32_t>::unknown_k / connectivity());
        if (!config.memory_limit) return capped;
        std::size_t const carried = audit_bytes_(audit_use_t::refine_k, nodes) + arena_bytes;
        return config.memory_limit > carried ? min_of((config.memory_limit - carried) / row_bytes, capped) : 0;
    }

    /** An empty audit scratch over @p context 's allocator. */
    static audit_scratch_t unbound_scratch_(traversal_context_t& context) noexcept {
        using bytes_ptr_t = typename traversal_context_t::bytes_ptr_t;
        return audit_scratch_t {bytes_ptr_t {nullptr, typename bytes_ptr_t::deleter_type {context.allocator_, 0}}};
    }

    /** Carves @p scratch over @p nodes nodes from @p context 's allocator, which it was made over,
     *  the reconnected counter zeroed. */
    static status_t carve_audit_(audit_scratch_t& scratch, traversal_context_t& context, audit_use_t use,
                                 std::size_t nodes) noexcept {
        std::size_t const bytes = audit_bytes_(use, nodes);
        scratch.bytes.reset(context.allocator_.allocate(bytes));
        scratch.bytes.get_deleter().count = bytes;
        if (!scratch.bytes) return status_t::bad_alloc_k;
        scratch.words = divide_round_up<std::size_t>(nodes, 32);
        scratch.reached.words = reinterpret_cast<u32_t*>(scratch.bytes.get());
        scratch.frontier.words = scratch.reached.words + scratch.words;
        scratch.next_frontier.words = scratch.frontier.words + scratch.words;
        scratch.levels = scratch.next_frontier.words + scratch.words;
        scratch.expanded = scratch.levels + 1;
        scratch.unreached = scratch.expanded + 2;
        scratch.reconnected = scratch.unreached + 1;
        scratch.swept = scratch.reconnected + 1;
        if (use == audit_use_t::refine_k) {
            scratch.island_ids = scratch.swept + 2;
            scratch.in_links = scratch.island_ids + nodes;
            scratch.changed.words = scratch.in_links + nodes;
        }
        return cuda_fill_async(context.api(), scratch.reconnected, 0, sizeof(u32_t), context.stream()) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Zeroed audit counters in a slot of @p context 's stream, or null when it cannot pin one. */
    static audit_counters_ptr_t take_audit_counters_(traversal_context_t& context) noexcept {
        cuda_pinned_slots_t* const pinned_slots = context.allocator_.pinned_slots();
        void* const slot = pinned_slots ? pinned_slots->take() : nullptr;
        return {slot ? new (slot) audit_counters_t {} : nullptr, audit_counters_return_t {pinned_slots}};
    }

    /** The converged audit @p counters reports over @p nodes nodes. */
    static reachability_t audited_(audit_counters_t const& counters, std::size_t nodes) noexcept {
        // A sweep that expands nothing ends the levels, and it counts as one, as on the CPU.
        return {nodes, nodes - counters.unreached, std::size_t {counters.levels} + 1};
    }

    /** Binds @p plan, whose claim word the caller closed, to a refinement of @p nodes over
     *  @p entries nodes of @p longest bytes at most, preparing into @p span_bytes: carves the
     *  audit's scratch, grows the context's arena for the walk and its staging for a sweep tile, so
     *  no phase allocates, and takes the counters last, which marks the plan bound. */
    status_t open_refine_(refine_plan_t& plan, iota_view<node_id_t> nodes, std::size_t entries, std::size_t longest,
                          std::size_t span_bytes, traversal_context_t& context, refine_config_t config) noexcept {
        using phase_t = typename refine_plan_t::phase_t;
        if (status_t const carved = carve_audit_(plan.scratch_, context, audit_use_t::refine_k, entries);
            failed(carved))
            return carved;
        std::size_t const arena_bytes = refine_arena_bytes_(config, static_cast<u32_t>(span_bytes),
                                                            kits_.rung_serving_(longest));
        if (!context.ensure_arena(arena_bytes)) return status_t::bad_alloc_k;
        plan.tile_rows_ = refine_tile_rows_(nodes.size(), entries, config, arena_bytes);
        if ((!nodes.empty() && !plan.tile_rows_) ||
            (plan.tile_rows_ && !context.ensure_staging(refine_tile_bytes_(plan.tile_rows_))))
            return status_t::bad_alloc_k;
        // Zeroed once, so every phase adds up, and copied with every phase.
        if (cuda_fill_async(api(), context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), context.stream()) !=
            cuda_success_k)
            return status_t::device_error_k;
        plan.counters_ = take_audit_counters_(context);
        if (!plan.counters_) return status_t::bad_alloc_k;
        plan.config_ = config;
        plan.nodes_ = nodes;
        plan.phase_ = !entries ? phase_t::done_k : nodes.empty() ? phase_t::audit_k : phase_t::reselect_k;
        return status_t::success_k;
    }

    /** Queues one re-selection of every node in @p nodes over the @p entries published nodes, each
     *  merging its picks into its row, on the arena the plan grew; a launch cache holds a walk's
     *  candidates or a merge round's, whichever are more. */
    status_t reselect_(traversal_context_t& context, audit_scratch_t const& scratch, refine_config_t config,
                       iota_view<node_id_t> nodes, std::size_t entries) noexcept {
        auto const [kits, kits_status] = kits_kernels_(context);
        if (failed(kits_status)) return kits_status;
        cuda_stream_handle_t const stream = context.stream();
        std::size_t const walked = nodes.size(), rung = kits_.rung_serving_(longest_value_);
        u32_t const span_bytes = static_cast<u32_t>(span_bytes_);
        build_shape_t const shape = build_shape_(config.expansion_factor, span_bytes, rung);
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   cuda_resident_blocks(vram_kernel_t::refine_k, 0) * vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(slices * shape.arena_shape.template slice_bytes<node_id_t>());
        if (!arena) return status_t::bad_alloc_k;
        auto const [epoch_base, epochs_status] = context.take_epochs(walked, arena_bytes_before);
        if (failed(epochs_status)) return epochs_status;

        vram_arguments<node_id_t> arguments {};
        vram_build<node_id_t>& build = arguments.build;
        build.storage = storage();
        build.first_id = nodes.first;
        build.end_id = nodes.bound;
        build.sampling_bound = static_cast<node_id_t>(entries);
        build.span_bytes = span_bytes;
        build.arena = arena;
        build.arena_shape = shape.arena_shape;
        build.expansion = shape.expansion;
        build.epoch_base = epoch_base;
        build.telemetry = context.telemetry_.get();
        build.assigned_nodes = context.assigned_nodes_.get();
        build.changed = scratch.changed;
        build.selection_nodes = max_of(shape.walk_selection_nodes, shape.reselection_nodes);
        build.scratch_bytes_per_warp = max_of(shape.walk_scratch_bytes, shape.back_link_scratch_bytes);
        if (cuda_fill_async(api(), scratch.changed.words, 0, scratch.words * sizeof(u32_t), stream) != cuda_success_k ||
            cuda_fill_async(api(), build.assigned_nodes, 0, sizeof(u32_t), stream) != cuda_success_k)
            return status_t::device_error_k;
        u32_t const blocks = device_.multiprocessor_count *
                             cuda_resident_blocks(vram_kernel_t::refine_k, shape.candidate_slots);
        return cuda_launch(api(), kits_function_at_(*kits, rung, vram_kernel_t::refine_k, shape.candidate_slots),
                           blocks, block_threads_k, walkers_per_block_(rung) * build.scratch_bytes_per_warp, stream,
                           cuda_overlap_t::serialized_k, arguments);
    }

    /** The launch record of a sweep's tile of @p targets over the links of @p sources: its run
     *  offsets lead the staging the plan grew for @p tile_rows rows, and the tape follows them,
     *  holding @p tape_capacity links. */
    vram_sweep_t sweep_record_(traversal_context_t& context, audit_scratch_t const& scratch,
                               iota_view<node_id_t> sources, iota_view<node_id_t> targets, std::size_t tile_rows,
                               std::size_t tape_capacity) const noexcept {
        u32_t* const offsets = reinterpret_cast<u32_t*>(context.staging_.get());
        return {reinterpret_cast<u32_t const*>(neighbors_matrix_.base()),
                static_cast<u32_t>(connectivity()),
                static_cast<u32_t>(size()),
                static_cast<u32_t>(sources.first),
                static_cast<u32_t>(sources.bound),
                static_cast<u32_t>(targets.first),
                static_cast<u32_t>(targets.bound),
                offsets,
                offsets + tile_rows,
                static_cast<u32_t>(tape_capacity),
                scratch.swept};
    }

    /** Queues the count of every link @p sources hold into the tile of @p targets, none counted
     *  yet, the offset kernel, which keeps as many rows as a tile's tape of @p tile_rows rows of
     *  links holds, and the copy of what it kept into @p counters. */
    status_t queue_count_back_links_(kernel_table_t const& graph_kernels, traversal_context_t& context,
                                     audit_scratch_t const& scratch, audit_counters_t* counters,
                                     iota_view<node_id_t> sources, iota_view<node_id_t> targets,
                                     std::size_t tile_rows) noexcept {
        cuda_stream_handle_t const stream = context.stream();
        vram_graph<node_id_t> graph {};
        graph.sweep = sweep_record_(context, scratch, sources, targets, tile_rows, tile_rows * connectivity());
        if (cuda_fill_async(api(), graph.sweep.incoming_offsets, 0, targets.size() * sizeof(u32_t), stream) !=
            cuda_success_k)
            return status_t::device_error_k;
        if (status_t const counted = launch_graph_(graph_kernels, vram_kernel_t::count_back_links_k,
                                                   graph_blocks_(vram_kernel_t::count_back_links_k), graph, stream);
            failed(counted))
            return counted;
        if (status_t const offset = launch_graph_(graph_kernels, vram_kernel_t::offset_back_links_k, 1, graph, stream);
            failed(offset))
            return offset;
        return cuda_copy_async(api(), counters->swept, scratch.swept, sizeof(counters->swept), stream) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Queues the scatter of every link @p sources hold into the tile of @p targets onto the first
     *  @p tape_size slots of its tape. */
    status_t queue_scatter_back_links_(kernel_table_t const& graph_kernels, traversal_context_t& context,
                                       audit_scratch_t const& scratch, iota_view<node_id_t> sources,
                                       iota_view<node_id_t> targets, std::size_t tile_rows, u32_t tape_size) noexcept {
        vram_graph<node_id_t> graph {};
        graph.sweep = sweep_record_(context, scratch, sources, targets, tile_rows, tape_size);
        return launch_graph_(graph_kernels, vram_kernel_t::scatter_back_links_k,
                             graph_blocks_(vram_kernel_t::scatter_back_links_k), graph, context.stream());
    }

    /** Queues the merge of each node of the tile of @p targets with its run of the @p tape_size
     *  links on its tape, on the walk's grid and arena, and counts a node of @p sources that gained
     *  a member as changed. */
    status_t queue_merge_back_links_(traversal_context_t& context, audit_scratch_t const& scratch,
                                     refine_config_t config, iota_view<node_id_t> sources, iota_view<node_id_t> targets,
                                     std::size_t tile_rows, u32_t tape_size) noexcept {
        auto const [kits, kits_status] = kits_kernels_(context);
        if (failed(kits_status)) return kits_status;
        cuda_stream_handle_t const stream = context.stream();
        std::size_t const rung = kits_.rung_serving_(longest_value_);
        u32_t const span_bytes = static_cast<u32_t>(span_bytes_);
        build_shape_t const shape = build_shape_(config.expansion_factor, span_bytes, rung);
        byte_t* const arena = context.ensure_arena(refine_arena_bytes_(config, span_bytes, rung));
        if (!arena) return status_t::bad_alloc_k;
        vram_sweep_t const sweep = sweep_record_(context, scratch, sources, targets, tile_rows, tape_size);
        vram_arguments<node_id_t> arguments {};
        vram_build<node_id_t>& build = arguments.build;
        build.storage = storage();
        build.first_id = targets.first;
        build.end_id = targets.bound;
        build.sampling_bound = static_cast<node_id_t>(size());
        build.span_bytes = span_bytes;
        build.arena = arena;
        build.arena_shape = shape.arena_shape;
        build.expansion = shape.expansion;
        build.telemetry = context.telemetry_.get();
        build.assigned_nodes = context.assigned_nodes_.get();
        build.selection_nodes = shape.reselection_nodes;
        build.scratch_bytes_per_warp = shape.back_link_scratch_bytes;
        build.changed = scratch.changed;
        build.incoming_offsets = sweep.incoming_offsets;
        build.back_links = reinterpret_cast<node_id_t const*>(sweep.back_links);
        build.tape_size = tape_size;
        build.walked_first = sources.first;
        build.walked_bound = sources.bound;
        if (cuda_fill_async(api(), build.assigned_nodes, 0, sizeof(u32_t), stream) != cuda_success_k)
            return status_t::device_error_k;
        static_assert(cuda_resident_blocks(vram_kernel_t::merge_back_links_k, 0) ==
                      cuda_resident_blocks(vram_kernel_t::refine_k, 0));
        return cuda_launch(api(), kits_function_at_(*kits, rung, vram_kernel_t::merge_back_links_k, 0),
                           device_.multiprocessor_count * cuda_resident_blocks(vram_kernel_t::refine_k, 0),
                           block_threads_k, walkers_per_block_(rung) * build.scratch_bytes_per_warp, stream,
                           cuda_overlap_t::serialized_k, arguments);
    }

    /** Queues a reachability audit over @p nodes nodes from as many stride-spread landmarks as one
     *  search draws - the connectivity - and its first sweep, through @c queue_sweeps_. Nothing
     *  waits; the caller reads @p counters once its event fired, and queues further sweeps from
     *  sweep 2 while the last one still expanded. */
    status_t queue_audit_(kernel_table_t const& graph_kernels, audit_scratch_t const& scratch,
                          audit_counters_t* counters, std::size_t nodes, cuda_stream_handle_t stream) noexcept {
        // The bitsets, the levels, the sweeps' expanded counters and the unreached counter lie back
        // to back.
        if (cuda_fill_async(api(), scratch.reached.words, 0, (scratch.words * 3 + 4) * sizeof(u32_t), stream) !=
            cuda_success_k)
            return status_t::device_error_k;
        vram_graph<node_id_t> graph {};
        graph.neighbors = reinterpret_cast<node_id_t*>(neighbors_matrix_.base());
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.nodes_count = static_cast<u32_t>(nodes);
        graph.seed_count = static_cast<u32_t>(min_of<std::size_t>(connectivity(), nodes));
        graph.reached = scratch.reached;
        graph.frontier = scratch.frontier;
        if (status_t const launched = launch_graph_(graph_kernels, vram_kernel_t::mark_landmarks_k,
                                                    divide_round_up<u32_t>(graph.seed_count, block_threads_k), graph,
                                                    stream);
            failed(launched))
            return launched;
        return queue_sweeps_(graph_kernels, scratch, counters, nodes, 1, stream);
    }

    /** Queues the audit's sweeps from @p first_sweep up to twice it over @p nodes nodes, so the
     *  batches double until one ends past convergence, then a count of the nodes no sweep reached
     *  so far, collecting their IDs when @p scratch has room for them, then the copy of every
     *  counter, the last sweep's expanded count among them. */
    status_t queue_sweeps_(kernel_table_t const& graph_kernels, audit_scratch_t const& scratch,
                           audit_counters_t* counters, std::size_t nodes, u32_t first_sweep,
                           cuda_stream_handle_t stream) noexcept {
        vram_graph<node_id_t> graph {};
        graph.neighbors = reinterpret_cast<node_id_t*>(neighbors_matrix_.base());
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.nodes_count = static_cast<u32_t>(nodes);
        graph.reached = scratch.reached;
        graph.levels = scratch.levels;
        // A sweep queued past convergence finds the one before it expanded nothing and leaves at
        // entry; odd sweeps grow the first frontier, even ones the second.
        for (u32_t sweep_number = first_sweep; sweep_number != 2 * first_sweep; ++sweep_number) {
            bool const odd = sweep_number & 1u;
            graph.frontier = odd ? scratch.frontier : scratch.next_frontier;
            graph.next_frontier = odd ? scratch.next_frontier : scratch.frontier;
            graph.counter = scratch.expanded + odd;
            graph.previous_counter = sweep_number == 1 ? nullptr : scratch.expanded + !odd;
            if (cuda_fill_async(api(), graph.next_frontier.words, 0, scratch.words * sizeof(u32_t), stream) !=
                    cuda_success_k ||
                cuda_fill_async(api(), graph.counter, 0, sizeof(u32_t), stream) != cuda_success_k)
                return status_t::device_error_k;
            if (status_t const launched = launch_graph_(graph_kernels, vram_kernel_t::reachability_sweep_k,
                                                        graph_blocks_(vram_kernel_t::reachability_sweep_k), graph,
                                                        stream);
                failed(launched))
                return launched;
        }
        graph.island_ids = scratch.island_ids;
        graph.counter = scratch.unreached;
        if (cuda_fill_async(api(), scratch.unreached, 0, sizeof(u32_t), stream) != cuda_success_k)
            return status_t::device_error_k;
        if (status_t const launched = launch_graph_(graph_kernels, vram_kernel_t::reachability_islands_k,
                                                    divide_round_up<u32_t>(graph.nodes_count, block_threads_k), graph,
                                                    stream);
            failed(launched))
            return launched;
        u32_t const* const last_expanded = scratch.expanded + ((2 * first_sweep - 1) & 1u);
        return cuda_copy_async(api(), &counters->levels, scratch.levels, sizeof(u32_t), stream) == cuda_success_k &&
                       cuda_copy_async(api(), &counters->last_expanded, last_expanded, sizeof(u32_t), stream) ==
                           cuda_success_k &&
                       cuda_copy_async(api(), &counters->unreached, scratch.unreached, 2 * sizeof(u32_t), stream) ==
                           cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Queues the reconnection of the @p islands islands the last audit in @p scratch collected
     *  over @p nodes nodes: the reached in-links first, so no reconnection takes the last link a
     *  reached node is found through, then each island into a reached pick or its landmark. */
    status_t queue_reconnect_(kernel_table_t const& graph_kernels, audit_scratch_t const& scratch, u32_t islands,
                              std::size_t nodes, cuda_stream_handle_t stream) noexcept {
        if (cuda_fill_async(api(), scratch.in_links, 0, nodes * sizeof(u32_t), stream) != cuda_success_k)
            return status_t::device_error_k;
        vram_graph<node_id_t> graph {};
        graph.neighbors = reinterpret_cast<node_id_t*>(neighbors_matrix_.base());
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.nodes_count = static_cast<u32_t>(nodes);
        graph.seed_count = static_cast<u32_t>(min_of<std::size_t>(connectivity(), nodes));
        graph.reached = scratch.reached;
        graph.in_links = scratch.in_links;
        graph.island_ids = scratch.island_ids;
        graph.islands_count = islands;
        graph.reconnected = scratch.reconnected;
        if (status_t const counted = launch_graph_(graph_kernels, vram_kernel_t::reached_in_links_k,
                                                   graph_blocks_(vram_kernel_t::reached_in_links_k), graph, stream);
            failed(counted))
            return counted;
        return launch_graph_(graph_kernels, vram_kernel_t::reconnect_island_k,
                             divide_round_up<u32_t>(islands, block_threads_k), graph, stream);
    }

    /** Bytes a split of @p nodes nodes into @p parts parts carves on the device and pins on the
     *  host, and where the pinned bytes place the offsets columns, the donor's value starts, the
     *  bases and the roots, as @c split_t lays them out. */
    struct split_layout_t {
        std::size_t device_bytes = 0, device_bases = 0;
        std::size_t host_bytes = 0, host_offsets = 0, host_starts = 0, host_bases = 0, host_roots = 0,
                    host_assigned = 0;
    };
    static split_layout_t split_layout_(std::size_t nodes, std::size_t parts) noexcept {
        std::size_t const words = divide_round_up<std::size_t>(nodes, 32);
        split_layout_t layout;
        layout.device_bases = aligned_((2 * nodes + 3 * words + 3 * parts + 2) * sizeof(u32_t));
        layout.device_bytes = layout.device_bases + 3 * parts * sizeof(byte_t*);
        layout.host_offsets = aligned_((nodes + words) * sizeof(u32_t));
        layout.host_starts = layout.host_offsets + aligned_((nodes + parts) * sizeof(u64_t));
        layout.host_bases = layout.host_starts + aligned_((nodes + 1) * sizeof(u64_t));
        layout.host_roots = layout.host_bases + 3 * parts * sizeof(byte_t*);
        layout.host_assigned = layout.host_roots + parts * sizeof(u32_t);
        layout.host_bytes = layout.host_assigned + sizeof(u32_t);
        return layout;
    }

    /** Where a split's device scratch keeps each array: each node's part, which the scatter reads
     *  back as its slot, the round's proposals, @b [nodes] each, the two frontiers and the picked
     *  bits, one bit per node each, then per part its size and its round count, the nodes the last
     *  two rounds placed, by round parity, the parts' roots, and the receivers' bases. */
    struct split_scratch_t {
        u32_t* node_parts = nullptr;
        u32_t* proposals = nullptr;
        vram_bitset_t frontier {}, next_frontier {}, picked {};
        u32_t* part_sizes = nullptr;
        u32_t* round_counts = nullptr;
        u32_t* placed = nullptr;
        u32_t* roots = nullptr;
        byte_t** part_bases = nullptr;
    };

    /** The arrays of @p split 's device scratch, read off its bytes. */
    static split_scratch_t split_scratch_(split_t const& split) noexcept {
        std::size_t const entries = split.entries, parts = split.receivers.size();
        std::size_t const words = divide_round_up<std::size_t>(entries, 32);
        split_scratch_t carved;
        carved.node_parts = reinterpret_cast<u32_t*>(split.device_bytes.get());
        carved.proposals = carved.node_parts + entries;
        carved.frontier.words = carved.proposals + entries;
        carved.next_frontier.words = carved.frontier.words + words;
        carved.picked.words = carved.next_frontier.words + words;
        carved.part_sizes = carved.picked.words + words;
        carved.round_counts = carved.part_sizes + parts;
        carved.placed = carved.round_counts + parts;
        carved.roots = carved.placed + 2;
        carved.part_bases = reinterpret_cast<byte_t**>(split.device_bytes.get() +
                                                       split_layout_(entries, parts).device_bases);
        return carved;
    }

    /** Carves @p split 's device scratch from @p context 's allocator, borrows its pinned results
     * as the host bytes, copies this index's value starts into it, and queues the upload of the
     * receivers' bases and the roots
     *  @p seed picks. */
    status_t carve_split_(split_t& split, u64_t seed, traversal_context_t& context) noexcept {
        std::size_t const entries = split.entries, parts = split.receivers.size();
        split_layout_t const layout = split_layout_(entries, parts);
        split.device_bytes.reset(context.allocator_.allocate(layout.device_bytes));
        split.device_bytes.get_deleter().count = layout.device_bytes;
        if (!split.device_bytes) return status_t::bad_alloc_k;
        growing_buffer<scored_node_t, cuda_pinned_allocator_t>& pinned = context.results.entries_;
        if (failed(pinned.resize_uninitialized(divide_round_up(layout.host_bytes, sizeof(scored_node_t)))) ||
            failed(split.parts.resize(parts)))
            return status_t::bad_alloc_k;
        split.host_bytes = reinterpret_cast<byte_t*>(pinned.data());
        split_scratch_t const carved = split_scratch_(split);
        byte_t** const bases = reinterpret_cast<byte_t**>(split.host_bytes + layout.host_bases);
        u32_t* const roots = reinterpret_cast<u32_t*>(split.host_bytes + layout.host_roots);
        for (std::size_t part = 0; part != parts; ++part) {
            vram_ragged_graph& receiver = *split.receivers[part];
            bases[part] = receiver.neighbors_matrix_.base();
            bases[parts + part] = receiver.values_tape_.base();
            bases[2 * parts + part] = receiver.offsets_column_.base();
            roots[part] = static_cast<u32_t>(seeded_bijection(part, entries, seed));
        }
        cuda_stream_handle_t const stream = context.stream();
        return cuda_copy_async(api(), split.host_bytes + layout.host_starts, offsets_column_.base(),
                               (entries + 1) * sizeof(u64_t), stream) == cuda_success_k &&
                       cuda_copy_async(api(), carved.part_bases, bases, 3 * parts * sizeof(byte_t*), stream) ==
                           cuda_success_k &&
                       cuda_copy_async(api(), carved.roots, roots, parts * sizeof(u32_t), stream) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** The record every launch of @p split reads: this index's neighbors and nodes, and the split's
     *  scratch. */
    vram_graph<node_id_t> split_graph_(split_t const& split) const noexcept {
        std::size_t const parts = split.receivers.size();
        split_scratch_t const carved = split_scratch_(split);
        vram_graph<node_id_t> graph {};
        graph.neighbors = reinterpret_cast<node_id_t*>(neighbors_matrix_.base());
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.nodes_count = static_cast<u32_t>(split.entries);
        graph.seed_count = static_cast<u32_t>(min_of(parts, split.entries));
        graph.frontier = carved.frontier;
        graph.next_frontier = carved.next_frontier;
        graph.node_parts = carved.node_parts;
        graph.proposals = carved.proposals;
        graph.part_sizes = carved.part_sizes;
        graph.round_counts = carved.round_counts;
        graph.roots = carved.roots;
        graph.parts_count = static_cast<u32_t>(parts);
        graph.part_capacity = static_cast<u32_t>(split.part_capacity);
        graph.picked = carved.picked;
        graph.part_neighbors = reinterpret_cast<u32_t* const*>(carved.part_bases);
        graph.part_values = carved.part_bases + parts;
        graph.part_offsets = reinterpret_cast<u64_t const* const*>(carved.part_bases + 2 * parts);
        return graph;
    }

    /** Queues the start of @p split 's partition through @p graph_kernels: the node parts and
     *  proposals unknown, the bitsets, the parts' counters and the rounds' counts empty, then the
     *  roots filed in. */
    status_t queue_partition_(kernel_table_t const& graph_kernels, split_t& split,
                              cuda_stream_handle_t stream) noexcept {
        std::size_t const entries = split.entries, parts = split.receivers.size();
        std::size_t const words = divide_round_up<std::size_t>(entries, 32);
        split_scratch_t const carved = split_scratch_(split);
        // Each run lies back to back.
        if (cuda_fill_async(api(), carved.node_parts, 0xFF, 2 * entries * sizeof(u32_t), stream) != cuda_success_k ||
            cuda_fill_async(api(), carved.frontier.words, 0, (3 * words + 2 * parts + 2) * sizeof(u32_t), stream) !=
                cuda_success_k)
            return status_t::device_error_k;
        vram_graph<node_id_t> graph = split_graph_(split);
        split.next_round = 1;
        return launch_graph_(graph_kernels, vram_kernel_t::partition_roots_k,
                             divide_round_up<u32_t>(graph.seed_count, block_threads_k), graph, stream);
    }

    /**
     *  @brief Queues @p split 's growth rounds from its next one up to twice it, so the batches
     *      double until one ends past convergence, then copies the last round's placed count back
     *      to the host.
     *
     *  A round queued past convergence finds the one before placed nothing and leaves at entry; odd
     *  rounds grow the first frontier, even ones the second. Every node is placed once at most, so
     *  the nodes bound the rounds.
     */
    status_t queue_rounds_(kernel_table_t const& graph_kernels, split_t& split, cuda_stream_handle_t stream) noexcept {
        std::size_t const words = divide_round_up<std::size_t>(split.entries, 32), parts = split.receivers.size();
        split_scratch_t const carved = split_scratch_(split);
        vram_graph<node_id_t> graph = split_graph_(split);
        u32_t const first_round = split.next_round;
        for (u32_t round = first_round; round != 2 * first_round; ++round) {
            bool const odd = round & 1u;
            graph.frontier = odd ? carved.frontier : carved.next_frontier;
            graph.next_frontier = odd ? carved.next_frontier : carved.frontier;
            graph.counter = carved.placed + odd;
            graph.previous_counter = round == 1 ? nullptr : carved.placed + !odd;
            if (cuda_fill_async(api(), graph.next_frontier.words, 0, words * sizeof(u32_t), stream) != cuda_success_k ||
                cuda_fill_async(api(), graph.counter, 0, sizeof(u32_t), stream) != cuda_success_k)
                return status_t::device_error_k;
            for (vram_kernel_t const pass : {vram_kernel_t::partition_propose_k, vram_kernel_t::partition_count_k,
                                             vram_kernel_t::partition_accept_k})
                if (status_t const launched = launch_graph_(graph_kernels, pass, graph_blocks_(pass), graph, stream);
                    failed(launched))
                    return launched;
            if (status_t const launched = launch_graph_(graph_kernels, vram_kernel_t::partition_fill_k,
                                                        static_cast<u32_t>(parts), graph, stream);
                failed(launched))
                return launched;
        }
        split.next_round = 2 * first_round;
        return cuda_copy_async(api(), split.host_bytes + split_layout_(split.entries, parts).host_assigned,
                               carved.placed + ((2 * first_round - 1) & 1u), sizeof(u32_t), stream) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** The last growth round of @p split placed something, so more rounds follow. */
    static bool placing_(split_t const& split) noexcept {
        std::size_t const host_assigned = split_layout_(split.entries, split.receivers.size()).host_assigned;
        return *reinterpret_cast<u32_t const*>(split.host_bytes + host_assigned) != 0;
    }

    /** Queues the end of @p split 's partition once no round places anything: the nodes no round
     *  reached, the picked nodes, and the copy of the node parts and the picked bits. */
    status_t queue_picked_(kernel_table_t const& graph_kernels, split_t& split, cuda_stream_handle_t stream) noexcept {
        std::size_t const entries = split.entries, words = divide_round_up<std::size_t>(entries, 32);
        vram_graph<node_id_t> graph = split_graph_(split);
        if (status_t const launched = launch_graph_(graph_kernels, vram_kernel_t::partition_islands_k, 1, graph,
                                                    stream);
            failed(launched))
            return launched;
        if (status_t const launched = launch_graph_(graph_kernels, vram_kernel_t::partition_picked_k,
                                                    graph_blocks_(vram_kernel_t::partition_picked_k), graph, stream);
            failed(launched))
            return launched;
        byte_t* const host_bytes = split.host_bytes;
        split_scratch_t const carved = split_scratch_(split);
        return cuda_copy_async(api(), host_bytes, carved.node_parts, entries * sizeof(u32_t), stream) ==
                           cuda_success_k &&
                       cuda_copy_async(api(), host_bytes + entries * sizeof(u32_t), carved.picked.words,
                                       words * sizeof(u32_t), stream) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Queues @p split 's scatter: the assigned slots and the receivers' offsets columns up, then
     *  every node's neighbors and value into its receiver. */
    status_t queue_scatter_(kernel_table_t const& graph_kernels, split_t const& split,
                            cuda_stream_handle_t stream) noexcept {
        byte_t const* const host_bytes = split.host_bytes;
        u64_t const* const offsets = reinterpret_cast<u64_t const*>(
            host_bytes + split_layout_(split.entries, split.receivers.size()).host_offsets);
        if (cuda_copy_async(api(), split_scratch_(split).node_parts, host_bytes, split.entries * sizeof(u32_t),
                            stream) != cuda_success_k)
            return status_t::device_error_k;
        for (std::size_t index = 0; index != split.receivers.size(); ++index) {
            split_part_t const& part = split.parts[index];
            vram_ragged_graph const& receiver = *split.receivers[index];
            if (part.entries && cuda_copy_async(api(), receiver.offsets_column_.base(), offsets + part.offsets_first,
                                                (part.entries + 1) * sizeof(u64_t), stream) != cuda_success_k)
                return status_t::device_error_k;
        }
        vram_graph<node_id_t> graph = split_graph_(split);
        graph.values = values_tape_.base();
        graph.value_offsets = reinterpret_cast<u64_t const*>(offsets_column_.base());
        for (vram_kernel_t const pass : {vram_kernel_t::split_links_k, vram_kernel_t::split_tape_k})
            if (status_t const launched = launch_graph_(graph_kernels, pass, graph_blocks_(pass), graph, stream);
                failed(launched))
                return launched;
        return status_t::success_k;
    }

    /** Hardware identity: the ordinal comes from the storage allocator at construction, the
     *  remaining specs fill at the first @c reserve. */
    cuda_device_specs_t device_;

    /** The metric the kits score under, fixed at construction. */
    nk_kernel_kind_t kernel_;

    /** Neighbors per node, frozen by the first @c reserve; zero until then. */
    std::size_t connectivity_ = 0;

    /** The values tape, the offsets column and the neighbors matrix, each allocator its own
     *  storage, so capacities are derived, never cached. */
    cuda_virtual_memory_allocator<byte_t> values_tape_, offsets_column_, neighbors_matrix_;

    /** Held by the one batch claiming, growing and linking at a time, until its drain. */
    claim_word_t claims_;

    /** Nodes the reserved ranges span, the most any claim reaches; fixed by the first
     *  @c reserve. */
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

    /** Marks the last sentinel fill, which every grow's stream follows; recorded and followed by
     *  the claim's holder alone. */
    cuda_event_t grown_;

    /** Nodes strictly below it are published and visible, turned once per batch after its drain. */
    std::size_t last_visible_id_ = 0;

    /** The first post-claim device failure. */
    status_t status_ = status_t::success_k;

    /** The modality's kits: their hooks, their fatbin's kernels and their device parameters. */
    kits_type_ kits_;

    /** Every node's key, assigned by its claim's holder, and the key map that holder files them in
     *  just before the watermark moves. */
    keyed_t keyed_;
};

} // namespace unum::usearch::basic
