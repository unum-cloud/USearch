/**
 *  @file include/usearch/basic/vram_amd_index.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Similarity Search over variable-length values in a single uniform latency memory space on
 *      AMD GPUs: the host graph.
 *
 *  The host graph the AMD basic faces share, compiled by any C++ compiler: it fills plain argument
 *  records and launches each modality's kernels by name, as the dense host face does, through the
 *  dense traversal contexts. A modality's face owns one over its kits, which name their rungs and
 *  read the kit for the rung a launch's longest value needs, the lanes that rung scores a node with
 *  and the span one value's preparation takes, in its vendor metric, which only its AMD unit
 *  compiles, so dispatch never names a kit.
 */
#pragma once

#include <cstddef> // `std::size_t`, `offsetof` - where a round's counters follow the check's

#include <array>            // `std::array` - a round's column offsets
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
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - a bounce and its host offsets
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator_t` - the host keys' placement
#include "usearch/utilities/ram_tables.hpp"        // `keyed_nodes` - every node's key and every key's nodes
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_saved_columns` - a load's saved columns
#include "usearch/utilities/vram_device.hpp"       // `vram_kernel_t` - the kernels a rung names
#include "usearch/utilities/vram_amd_device.hpp"   // `hip_traversal_context` & the rungs
#include "usearch/utilities/vram_walks.hpp"        // `gpu_arena_shape_t`, `gpu_candidate_slots`, `vram_bitset_t`
#include "usearch/basic/ram_index.hpp"             // `columns`, `columns_extents_t`, `refine_report_t`

namespace unum::usearch::basic {
using namespace unum::usearch::utilities;

template <typename index_type_, typename, typename, typename, typename>
    requires std::same_as<typename index_type_::node_id_t, u32_t>
struct vram_amd_indexes;

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

/** One gather or restitch launch of a round, mirroring the shared round view: the warps' spans and
 *  selection scratch as a build's, the runs a gather files and the neighborhoods a restitch reads,
 *  and the health check's columns the reach rule reads and the counters take the round's counts. */
template <typename node_id_type_>
struct vram_round {
    vram_ragged_storage<node_id_type_> storage;
    u32_t span_bytes = 0;
    u32_t scratch_bytes_per_warp = 0;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    u32_t selection_nodes = 0;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned = nullptr;
    u32_t const* entries = nullptr;
    u32_t const* hub_nodes = nullptr;
    vram_neighborhood_run_t const* runs = nullptr;
    u32_t const* run_entries = nullptr;
    u32_t runs_count = 0;
    u16_t* entry_neighborhoods = nullptr;
    u32_t const* neighborhood_offsets = nullptr;
    u32_t neighborhoods_count = 0;
    vram_health_t health {};
};

/** The one argument every AMD basic kernel takes, each reading the record its kernel names. */
template <typename node_id_type_>
struct vram_arguments {
    vram_build<node_id_type_> build;
    vram_search<node_id_type_> search;
    vram_load<node_id_type_> load;
    vram_round<node_id_type_> round;
};

/** One launch over the neighbors matrix alone, reading no value: the rebase of nodes a source gave
 *  up, a pass of the health check or its heal, and a cut's partition and the scatter moving its
 *  nodes, whose values pass as opaque bytes. */
template <typename node_id_type_>
struct vram_graph {
    node_id_type_* neighbors = nullptr;
    u32_t connectivity = 0;
    u32_t nodes_count = 0;
    node_id_type_ first_id = 0;
    node_id_type_ source_first = 0;
    node_id_type_ source_bound = 0;
    node_id_type_ shift = 0;
    vram_bitset_t frontier {};
    vram_bitset_t next_frontier {};
    u32_t* counter = nullptr;
    u32_t const* previous_counter = nullptr;
    u32_t seed_count = 0;

    /** The health check's columns and the pass a health launch runs. */
    vram_health_t health {};

    /** A cut's part of each node, then its slot once the parts are ordered, the round's
     *  proposals, and per part its size, its round count and its root. */
    u32_t* node_parts = nullptr;
    u32_t* proposals = nullptr;
    u32_t* part_sizes = nullptr;
    u32_t* round_counts = nullptr;
    u32_t const* roots = nullptr;
    u32_t parts_count = 0;
    u32_t part_capacity = 0;

    /** The targets' neighbors matrices, tapes and offsets columns, by part, and the source's tape
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
constexpr u32_t hip_resident_blocks(vram_kernel_t kernel, u32_t slots) noexcept {
    switch (kernel) {
    case vram_kernel_t::search_k: return slots <= 8 ? 4 : 3;
    case vram_kernel_t::rebase_links_k:
    case vram_kernel_t::health_expand_k:
    case vram_kernel_t::health_recheck_k:
    case vram_kernel_t::health_islands_k:
    case vram_kernel_t::island_components_k:
    case vram_kernel_t::health_count_links_k:
    case vram_kernel_t::label_islands_k:
    case vram_kernel_t::partition_propose_k:
    case vram_kernel_t::partition_count_k:
    case vram_kernel_t::partition_accept_k:
    case vram_kernel_t::split_links_k:
    case vram_kernel_t::split_tape_k:
    case vram_kernel_t::widen_links_k: return 8;
    default: return 2;
    }
}

/** One instantiated kernel: the kernel and its register rung, zero where it keeps no ladder or
 *  walks the arena frontier. */
struct amd_rung_t {
    vram_kernel_t kernel;
    u32_t slots;
};

/** Every instantiation each rung of a modality's kits exports. */
inline constexpr amd_rung_t amd_rungs[] = {
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

/** The kernels a load queues, which each kit exports once, on its first rung. */
inline constexpr vram_kernel_t amd_load_kernels[] = {vram_kernel_t::load_values_k, vram_kernel_t::size_spans_k};

/** Every kernel of the unit all AMD basic faces share, over the neighbors matrix alone, so one
 *  instantiation per node-ID type serves every modality. */
inline constexpr vram_kernel_t amd_graph_kernels[] = {
    vram_kernel_t::rebase_links_k,       vram_kernel_t::health_expand_k,     vram_kernel_t::health_recheck_k,
    vram_kernel_t::order_islands_k,      vram_kernel_t::health_islands_k,    vram_kernel_t::island_components_k,
    vram_kernel_t::health_count_links_k, vram_kernel_t::label_islands_k,     vram_kernel_t::heal_islands_k,
    vram_kernel_t::partition_roots_k,    vram_kernel_t::partition_propose_k, vram_kernel_t::partition_count_k,
    vram_kernel_t::partition_accept_k,   vram_kernel_t::partition_fill_k,    vram_kernel_t::partition_islands_k,
    vram_kernel_t::split_links_k,        vram_kernel_t::split_tape_k,        vram_kernel_t::gather_tape_k,
    vram_kernel_t::widen_links_k,
};

/** The code object of @ref amd_graph_kernels over 32-bit node IDs for the AMD class whose NumKong
 *  capabilities are @p capability_, defined beside them in that class's
 *  @c src/basic/vram_variants_amd.hip, which every face of the class loads at its first
 *  @c reserve. One per class, since each runs its own wavefront. */
template <nk_capability_t capability_>
struct hip_graph_code_object {
    static bytes_view_t bytes() noexcept;
};

/** Spells every kernel of @ref hip_graph_code_object for a @c hip_kernel_table_t, in
 *  @ref vram_kernel_t order, through the graph's own spelling of its node-ID type; none is
 *  required, since the code object covers 32-bit IDs alone, and a verb over any other node-ID type
 *  answers @c unsupported_k. */
struct hip_graph_kernel_speller_t {
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

/** Spells every kernel of a modality's kits code object for a @c hip_kernel_table_t: by kit rung,
 *  then in @ref vram_kernel_t order, then by register rung; every walk of @ref amd_rungs is
 *  required on every kit rung. */
struct hip_kits_kernel_speller_t {
    using symbol_t = vram_kernel_symbol_t;

    /** The kits' rungs, and their spelling of one kernel instantiated on one of them. */
    std::size_t rungs;
    void (*spell_kernel)(vram_kernel_symbol_t&, std::size_t, amd_rung_t) noexcept;

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
                     amd_rung_t {static_cast<vram_kernel_t>(within / gpu_rungs_per_kernel_k),
                                 slots_of(within % gpu_rungs_per_kernel_k)});
    }

    /** Whether the kits instantiate the kernel at @p position on its kit rung. */
    static constexpr bool required(std::size_t position) noexcept {
        for (amd_rung_t const rung : amd_rungs)
            if (index(0, rung.kernel, rung.slots) == position % positions_per_kit_rung_k) return true;
        for (vram_kernel_t const kernel : amd_load_kernels)
            if (index(0, kernel, 0) == position) return true;
        return false;
    }
};

/** One run of a load as the kits' hooks queue its work: the run's saved values on the device, where
 *  its first stored value goes in the tape, and the record the kits' load kernels read. */
template <typename node_id_type_>
struct vram_amd_load_step {
    vram_arguments<node_id_type_> arguments {};
    bytes_view_t saved;
    byte_t* stored = nullptr;
    hip_api_t const* api = nullptr;
    hip_kernel_table_t const* kits = nullptr;
    hip_stream_handle_t stream = nullptr;
    u32_t multiprocessors = 0;

    /** Lanes per wavefront of the device, which the kits' blocks are four of. */
    u32_t warp_lanes = 0;

    /** Copies the run's saved values into the tape as they are, for kits storing them so. */
    status_t copy_values() const noexcept {
        if (saved.empty()) return status_code_t::success_k;
        return hip_vendor_t::copy(*api, stored, saved.data(), saved.size(), stream);
    }

    /** Launches the kits' @p kernel over the run's nodes on the stream, a grid-striding pass. */
    status_t launch_kits(vram_kernel_t kernel) const noexcept {
        return hip_launch(*api, kits->functions[hip_kits_kernel_speller_t::index(0, kernel, 0)],
                          multiprocessors * hip_resident_blocks(kernel, 0), vram_warps_per_block_k * warp_lanes, 0,
                          stream, arguments);
    }
};

#pragma endregion Kernel Records

/**
 *  @brief Similarity Search over variable-length values in a single uniform latency memory space on
 *      AMD GPUs, a modality's face owning one.
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
 *  another answers @c contended_k meanwhile, while searches on other streams keep running.
 *
 *  A refine runs the dense device round: a health check over the neighbors matrix, a heal with no
 *  distance, a seeded draw of hubs, a gather filing every entry under its nearest hub, and a
 *  restitch of each neighborhood through the kits, then the closing check. A reshard moves entries
 *  first - a gather appends other graphs' nodes on the device, a cut partitions and scatters them
 *  into its targets - then runs the same round on every target. The health check, the heal and the
 *  moves launch kernels that read the neighbors matrix alone, which one unit shares across every
 *  modality. Saves and loads cross the snapshot columns of the basic RAM index through a pinned
 *  bounce, values through the kits' codec. Plans close the claim word, or claim their nodes, for
 *  their whole lives.
 *
 *  A modality's face owns one as a member and forwards its verbs, so no modality derives from it.
 *  It stands alone as the RAM index does, under either node-ID width and any hashable key, which it
 *  keeps on the host beside every node and files under the key map while it holds the claim word,
 *  just before the watermark moves; only the sharded index the compiled library builds narrows it
 *  to 32-bit IDs, whose nodes a shard table locates in 8 bytes.
 *
 *  @tparam node_id_type_ The node IDs of the neighbors matrix, the walks and the matches, @c u32_t
 *      or @c u64_t; the all-ones ID is the sentinel. The shared kernels cover @c u32_t alone, so a
 *      refine, a reshard and a health check answer @c unsupported_k over any other.
 *  @tparam kits_type_ The modality's kits, held as a member, whose @c capability_k names the AMD
 *      class whose shared code object it loads, whose @c rungs_k kit rungs its kernels cover,
 *      narrowest first, and whose members read them: @c rung_serving_ nodes of a length, or
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
template <typename node_id_type_, typename kits_type_, typename key_type_ = uuid128_t,
          typename key_hash_type_ = std::hash<key_type_>>
struct vram_amd_ragged_graph {
    using traversal_context_t = hip_traversal_context<hip_arena_allocator_t>;

    /** The graph's and the kits' kernels every index on one device launches, which its executor
     *  stream loads at the first ask and keeps, one table per code object. */
    using kernel_table_t = hip_vendor_t::kernel_table_t;
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

    /** Pins the index to the storage allocator's device, its kits scoring under @p kernel where
     *  they cover several metrics, and keeps its host keys and key map on @p keys_allocator; the
     *  first @c reserve freezes the rest. */
    explicit vram_amd_ragged_graph(nk_kernel_kind_t kernel, hip_virtual_memory_allocator<byte_t> const& allocator = {},
                                   forkunion_allocator_t const& keys_allocator = {}) noexcept
        : device_ {allocator.device()}, kernel_ {kernel}, values_tape_(allocator), offsets_column_(allocator),
          neighbors_matrix_(allocator),
          grown_((hip_set_device(allocator.api(), allocator.device()), hip_event_t {allocator.api()})),
          keyed_(keys_allocator) {}
    vram_amd_ragged_graph(vram_amd_ragged_graph const&) = delete;
    vram_amd_ragged_graph& operator=(vram_amd_ragged_graph const&) = delete;

#pragma region Measures

    /** Nodes visible to searches: the watermark every walk samples below and filters neighbors
     *  by. */
    std::size_t size() const noexcept {
        return fu::standard_atomic_ref<std::size_t const>(last_visible_id_).load(std::memory_order_acquire);
    }

    /** The first failure that left claimed nodes unpublished, kept for the index's lifetime. */
    status_t status() const noexcept {
        status_code_t const code =
            fu::standard_atomic_ref<status_code_t const>(status_.code).load(std::memory_order_acquire);
        return {code, fu::standard_atomic_ref<char const* const>(status_.note).load(std::memory_order_acquire)};
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
        return sizeof(vram_amd_ragged_graph) + values_tape_.committed_bytes() + offsets_column_.committed_bytes() +
               neighbors_matrix_.committed_bytes() + keyed_.memory_usage();
    }

    /** The pinned device's identity and limits, queried once at the first @c reserve. */
    hip_device_specs_t const& device_specs() const noexcept { return device_; }

    /** The driver table every call on this index goes through, the storage allocator's. */
    hip_api_t const& api() const noexcept { return neighbors_matrix_.api(); }

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
     *      into the host @p rows, raw as stored with sentinels and self-loops left in, and waits
     *      for the device copy to finish.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p rows hold no whole number of rows, or the rows pass @c size.
     *  - @c device_k The device could not be selected, or the copy failed.
     */
    status_t neighbors(node_id_t first, std::span<node_id_t> rows) const noexcept {
        std::size_t const row_ids = connectivity();
        if (!row_ids || rows.size() % row_ids || std::size_t {first} + rows.size() / row_ids > size())
            return status_code_t::config_mismatch_k;
        if (failed(hip_vendor_t::select(api(), device_.ordinal)) ||
            api().copy(rows.data(), neighbors_matrix_.base() + std::size_t {first} * row_ids * sizeof(node_id_t),
                       rows.size_bytes(), hip_copy_default_k) != hip_success_k)
            return status_code_t::device_k;
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

        /** Nodes the index may ever host, clamped to the ID space and the device's memory; past it
         *  every claim answers @c overflow_k. */
        std::size_t ceiling = max_node_count_k;
    };

    /**
     *  @brief Ensures the index can host at least @p config.capacity nodes of @p config.bytes value
     *      bytes in all, and answers the node capacity now in effect; the first success binds the
     *      device, loads the kernels and reserves the address ranges, later ones only map.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The face's kits take parameters, and none were set.
     *  - @c config_mismatch_k The connectivity is not a power of two, or differs from the frozen
     *    one.
     *  - @c overflow_k The capacity passes the ID space or the reserved range, or the value bytes
     *    pass the reserved tape.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (!std::has_single_bit(static_cast<u32_t>(config.connectivity))) return {0, status_code_t::config_mismatch_k};
        if (config.capacity > max_node_count_k) return {capacity(), status_code_t::overflow_k};
        std::size_t const frozen = connectivity();
        if (frozen && frozen != config.connectivity) return {capacity(), status_code_t::config_mismatch_k};
        if (!frozen) {
            if (hip_set_device(api(), device_.ordinal) != hip_success_k) return {0, status_code_t::device_k};
            auto const [specs, status] = hip_device_specs_t::query(api(), device_.ordinal);
            if (failed(status)) return {0, status};
            device_ = specs;
            if (status_t const uploaded = kits_.upload_parameters_(api()); failed(uploaded)) return {0, uploaded};
            // No column maps more than the device's memory, whatever the first capacity.
            std::size_t const budget_bytes = device_.vram_bytes;
            std::size_t const row_bytes = config.connectivity * sizeof(node_id_t) + sizeof(u64_t);
            std::size_t const ceiling_nodes = min_of(min_of(config.ceiling, max_node_count_k),
                                                     max_of<std::size_t>(config.capacity, budget_bytes / row_bytes));
            // The offsets column holds one start past the last node.
            if (offsets_column_.reserve_range((ceiling_nodes + 1) * sizeof(u64_t)) != status_code_t::success_k ||
                neighbors_matrix_.reserve_range(ceiling_nodes * config.connectivity * sizeof(node_id_t)) !=
                    status_code_t::success_k ||
                values_tape_.reserve_range(max_of(config.bytes, budget_bytes) + tape_margin_k) !=
                    status_code_t::success_k)
                return {0, status_code_t::out_of_memory_k};
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

    /** The bytes an executor's run must hold before a context carved over it can insert the nodes
     *  of @p tape, one aligned term per carve the verb makes, or zero before a @c reserve freezes
     *  the geometry of the index. */
    std::size_t traversal_memory_usage(tape_view<byte_t const> tape, insert_config_t config = {}) const noexcept {
        if (!connectivity() || !device_.multiprocessor_count || tape.offsets.empty()) return 0;
        std::size_t const entries = tape.size();
        std::size_t longest = longest_value_;
        u32_t const span_bytes = static_cast<u32_t>(max_of(span_bytes_, span_bytes_of_(tape, longest)));
        std::size_t const rung = kits_.rung_serving_(longest);
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   hip_resident_blocks(vram_kernel_t::insert_k, 0) * vram_warps_per_block_k;
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
         *      the one-shot @ref vram_amd_ragged_graph::insert describes.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent, or the index was never reserved.
         *  - @c config_mismatch_k The @p keys or @p tape hold other than the claimed nodes, or
         *    @p context rides another device or cannot walk.
         *  - @c unsupported_k A value past every rung of the face's kits.
         *  - @c overflow_k The nodes pass the reserved range or the host keys, or their values pass
         *    the tape it reserved.
         */
        expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape,
                                         traversal_context_t& context, insert_config_t config = {}) noexcept {
            if (!index_) return {{}, status_code_t::uninitialized_k};
            vram_amd_ragged_graph& index = *index_;
            auto const [launched, launched_status] = index.launch_(*this, keys, tape, context, config);
            if (succeeded(launched_status)) {
                index.publish_keys_({launched.first_id, static_cast<node_id_t>(launched.new_size)});
                index.publish_(launched.new_size);
            }
            return {launched, launched_status};
        }

      private:
        friend vram_amd_ragged_graph;

        /** The index whose claim word holds the nodes; null once spent. */
        vram_amd_ragged_graph* index_ = nullptr;
    };

    /**
     *  @brief Claims @p nodes for one batch, exposing their first ID before any of them launches;
     *      refuses a kept failure, nodes past the reserved ceiling, and another batch in flight
     *      with @c contended_k.
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
     *      @b [first_id,first_id+nodes), in admission waves riding the context's stream, then
     *      drains it, files the keys and turns the watermark over the range: @ref plan_insert, then
     *      its plan's @c insert. Another batch in flight answers @c contended_k.
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
     *  - @c device_k The code object would not load, or lacks a kit's kernel.
     */
    static expected<kernel_table_t const*> kits_kernels_(traversal_context_t const& context) noexcept {
        return context.table_of(kits_type_::fatbin_(),
                                hip_kits_kernel_speller_t {kits_type_::rungs_k, &kits_type_::spell_kits_kernel_});
    }

    /** The kernel the kits' code object of @p kits exports as @p symbol, or null when it has
     *  none. */
    static hip_function_handle_t kits_symbol_function_(kernel_table_t const& kits, char const* symbol) noexcept {
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
        if (output_offsets.size() != listed.size() + 1) return status_code_t::config_mismatch_k;
        auto const [graph_kernels, kernels_status] = graph_kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        if (listed.empty()) return status_code_t::success_k;
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
    friend struct vram_amd_indexes;

    /** Assigns @p keys and launches and drains the nodes @p plan claimed, spending it unless the
     *  arguments mismatch, and leaves them for @c publish_keys_ and @c publish_ to turn visible;
     *  the claim word holds until then. */
    expected<insert_result_t> launch_(insert_plan_t& plan, std::span<key_t const> keys, tape_view<byte_t const> tape,
                                      traversal_context_t& context, insert_config_t config) noexcept {
        if (plan.index_ != this) return {{}, status_code_t::uninitialized_k};
        if (tape.offsets.size() != plan.entries + 1 || keys.size() != plan.entries)
            return {{}, status_code_t::config_mismatch_k};
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
        if (!connectivity()) return refuse(status_code_t::uninitialized_k);
        if (context.device() != device_.ordinal || !context) return refuse(status_code_t::config_mismatch_k);
        if (status_t const kept = status(); failed(kept)) return refuse(kept);
        hip_stream_handle_t const stream = context.stream();

        // Walks prepare stored nodes as much as the batch's, so the rung and spans cover both.
        std::size_t longest = longest_value_;
        u32_t const span_bytes = static_cast<u32_t>(max_of(span_bytes_, span_bytes_of_(tape, longest)));
        std::size_t const rung = kits_.rung_serving_(longest);
        if (rung == kits_type_::rungs_k) return refuse(status_code_t::unsupported_k);
        std::size_t const batch_bytes = tape.offsets[entries] - tape.offsets[0];
        std::size_t const total_nodes = first_id + entries;

        // Every carve and growth precedes the first launch, so a refusal publishes nothing.
        build_shape_t const shape = build_shape_(config.expansion_factor, span_bytes, rung);
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   hip_resident_blocks(vram_kernel_t::insert_k, 0) * vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(slices * shape.arena_shape.template slice_bytes<node_id_t>());
        byte_t* const staging = context.ensure_staging(build_staging_bytes_(entries, total_nodes));
        if (!arena || !staging) return refuse(status_code_t::out_of_memory_k);
        if (status_t const grown = grow_nodes_(total_nodes, stream); failed(grown)) return refuse(grown);
        if (status_t const mapped = values_tape_.commit(tape_bytes_ + batch_bytes + tape_margin_k); failed(mapped))
            return refuse(mapped);
        for (std::size_t offset = 0; offset != entries; ++offset)
            keyed_.assign(static_cast<node_id_t>(first_id + offset), keys[offset]);
        if (failed(hip_vendor_t::copy(api(), values_tape_.base() + tape_bytes_, tape.elements + tape.offsets[0],
                                      batch_bytes, stream)) ||
            failed(hip_vendor_t::copy(api(), staging, tape.offsets.data(), (entries + 1) * sizeof(u64_t), stream)) ||
            failed(hip_vendor_t::fill(api(), context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream)))
            return refuse(status_code_t::device_k);
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
        if (failed(hip_vendor_t::fill(api(), build.link_heads, 0xFF, total_nodes * sizeof(node_id_t), stream)))
            return refuse(status_code_t::device_k);
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
     *  @param[in,out] context The stream every transfer and the launch ride, which the call drains
     *      before it returns.
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
        auto const [kits, kits_status] = kits_kernels_(context);
        if (failed(kits_status)) return {{}, kits_status};
        hip_stream_handle_t const stream = context.stream();

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
        if (!arena || !staging) return {{}, status_code_t::out_of_memory_k};
        std::size_t const tape_bytes = round_up_to_multiple<std::size_t>(batch_bytes + tape_margin_k, sizeof(u64_t));
        u64_t* const staged_offsets = reinterpret_cast<u64_t*>(staging + tape_bytes);
        node_id_t* const device_ids = reinterpret_cast<node_id_t*>(staged_offsets + queries_count + 1);
        distance_t* const device_distances = reinterpret_cast<distance_t*>(device_ids + results_count);
        auto const [epoch_base, epochs_status] = context.take_epochs(queries_count, arena_bytes_before);
        if (failed(epochs_status)) return {{}, epochs_status};
        if (failed(hip_vendor_t::copy(api(), staging, queries.elements + queries.offsets[0], batch_bytes, stream)) ||
            failed(hip_vendor_t::copy(api(), staged_offsets, queries.offsets.data(),
                                      (queries_count + 1) * sizeof(u64_t), stream)) ||
            failed(hip_vendor_t::fill(api(), context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream)) ||
            failed(hip_vendor_t::fill(api(), context.assigned_nodes_.get(), 0, 2 * sizeof(u32_t), stream)))
            return {{}, status_code_t::device_k};

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
        search.concurrent_walks = device_.multiprocessor_count * hip_resident_blocks(vram_kernel_t::search_k, slots) *
                                  walkers_per_block_(rung);
        search.filling_nodes = filling_nodes<u32_t>(search.expansion, static_cast<u32_t>(connectivity()));
        search.result_ids = device_ids;
        search.result_distances = device_distances;
        search.count = config.count;
        if (status_t const launched = hip_launch(
                api(), kits_function_at_(*kits, rung, vram_kernel_t::search_k, slots), blocks, block_threads(),
                walkers_per_block_(rung) * search.scratch_bytes_per_warp, stream, arguments);
            failed(launched))
            return {{}, launched};
        std::size_t const row_bytes = config.count * sizeof(node_id_t);
        if (failed(hip_vendor_t::copy_2d(api(), result_ids.data(), result_ids.stride_bytes(0), device_ids, row_bytes,
                                         row_bytes, queries_count, stream)) ||
            failed(hip_vendor_t::copy_2d(api(), result_distances.data(), result_distances.stride_bytes(0),
                                         device_distances, config.count * sizeof(distance_t),
                                         config.count * sizeof(distance_t), queries_count, stream)))
            return {{}, status_code_t::device_k};
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
        return {result, status_code_t::success_k};
    }

  public:

#pragma endregion Search

#pragma region Refinement

    /** A round's knobs, as the RAM tower names them. */
    struct refine_config_t {

        /** Bytes of cache one multiprocessor keeps, half for a draw's hubs and half for a
         *  neighborhood's entries, each the mean stored value; zero takes the device's, and no
         *  more than the device's is ever taken. */
        std::size_t cache_bytes = 0;
    };

    /** A reshard's knobs, as the RAM tower names them: its rounds', the most entries a sharded
     *  target's shard holds, which no device index honours, and the seed a cut grows parts by. */
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
    using round_scratch_t = vram_round_scratch<traversal_context_t>;
    using round_counters_t = vram_round_counters_t;

    /** Gives a round's counters back to the stream's pinned slots they came from. */
    struct round_counters_return_t {
        hip_pinned_slots_t* pinned_slots = nullptr;
        void operator()(round_counters_t* counters) const noexcept { pinned_slots->give_back(counters); }
    };
    using round_counters_ptr_t = std::unique_ptr<round_counters_t, round_counters_return_t>;
    static_assert(sizeof(round_counters_t) <= hip_pinned_slots_t::slot_bytes_k, "A round's counters fit one slot");

    /** Who holds the claim word of a refine plan's graph: the plan, as @ref plan_refine closes it,
     *  or the reshard plan running its rounds. */
    enum class claim_t : u8_t { borrowed_k, held_k };

    /** How a round over @c entries entries sizes its columns: the hubs one gather draws at most and
     *  the most entries a neighborhood holds, as RAM names them, then the room for the starts with
     *  splits, for the hubs of every gather's runs and for the runs. */
    struct round_shape_t {
        std::size_t entries = 0, hubs_per_draw = 0, capacity = 0;
        std::size_t offsets = 0, hubs = 0, runs = 0;
    };

    /** The page-locked host bytes a round's barriers order on, borrowed from its context's results:
     *  the entries by neighborhood, the starts with split room, the hubs, runs and entries before
     *  each run of the next gather, and each entry's neighborhood. */
    struct round_mirrors_t {
        u32_t* neighborhood_entries = nullptr;
        u32_t* neighborhood_offsets = nullptr;
        u32_t* hub_nodes = nullptr;
        vram_neighborhood_run_t* runs = nullptr;
        u32_t* run_entries = nullptr;
        u16_t* entry_neighborhoods = nullptr;
    };

    /** How a round's gather and restitch launch on the kit rung its longest value needs: the span
     *  one value prepares into, the selection's rows, its two halves of candidates, and each warp's
     *  dynamic-shared scratch, zero when the arena slice carries it. */
    struct round_launch_t {
        std::size_t rung = 0;
        u32_t span_bytes = 0, selection_nodes = 0, scratch_bytes_per_warp = 0;
        gpu_arena_shape_t arena_shape {};
    };

    /** One part of a cut as the host assigns it: its nodes, its value bytes and longest value,
     *  and where its target's offsets column starts inside the pinned host bytes. */
    struct split_part_t {
        std::size_t entries = 0, bytes = 0, longest = 0, offsets_first = 0;
    };

    /** A cut's moves, empty for a gather. */
    struct split_t {

        /** The index the cut empties, while it still holds its nodes and its claim word. */
        vram_amd_ragged_graph* source = nullptr;

        /** The source's nodes at planning, and the most nodes one part takes: those over the
         *  targets, rounded up. */
        std::size_t entries = 0, part_capacity = 0;

        /** The device scratch, laid out as @c split_scratch_ reads it. */
        typename traversal_context_t::bytes_ptr_t device_bytes;

        /** Page-locked host bytes borrowed from the plan's context, laid out as @c split_layout_
         *  places them: the node parts, assigned into slots in place, the targets' offsets columns
         *  back to back, the source's value starts, the bases and the roots planning uploads, and
         *  the last growth's placed count. */
        byte_t* host_bytes = nullptr;

        /** The growth round the next batch queues from, zero once the growth converged and the
         *  unplaced nodes are placed next. */
        u32_t next_round = 0;

        /** The parts as the host assigns them. */
        growing_buffer<split_part_t> parts {global_allocator};

        explicit split_t(typename traversal_context_t::bytes_ptr_t&& empty) noexcept : device_bytes(std::move(empty)) {}
        split_t(split_t&& other) noexcept
            : source(std::exchange(other.source, nullptr)), entries(other.entries), part_capacity(other.part_capacity),
              device_bytes(std::move(other.device_bytes)), host_bytes(std::exchange(other.host_bytes, nullptr)),
              next_round(other.next_round), parts(std::move(other.parts)) {}
        split_t& operator=(split_t&&) = delete;
    };

  public:

    /**
     *  @brief One health check in flight on this graph's device, the dense check over the
     *      neighbors matrix alone: @ref queue a batch of passes, @ref wait for it, @ref advance
     *      past it, and @ref finish once none follows.
     *
     *  It draws one root by seeded bijection, or keeps the one an earlier check kept, sweeps the
     *  depths from it and to it, names the islands outside its strong component and their
     *  components, and counts the unlinked and underlinked entries. Read-only on the graph.
     */
    struct health_plan_t {

        /** An unbound plan over @p scratch - what an empty graph or a refused plan answers with. */
        explicit health_plan_t(round_scratch_t&& scratch) noexcept : columns_(std::move(scratch)) {}
        health_plan_t(health_plan_t&&) noexcept = default;
        health_plan_t& operator=(health_plan_t&&) noexcept = default;

        /** The watermark the check covers; IDs strictly below it are swept. */
        node_id_t visible = 0;

        /** Whether another batch of passes follows. */
        bool running() const noexcept { return index_ && succeeded(status_) && step_ != step_t::done_k; }

        /** Queues the next batch of the current step on @p context, then the copy of the counters;
         *  a no-op unless @ref running. */
        status_t queue(traversal_context_t& context) noexcept {
            if (!running()) return status_;
            status_ = index_->queue_health_(context, *this);
            return status_;
        }

        /** Sleeps until the queued batch finished on @p context; a no-op unless @ref running. */
        status_t wait(traversal_context_t& context) noexcept {
            if (!running()) return status_;
            status_ = context.wait();
            return status_;
        }

        /** The barrier after a batch finished: reads the counters, moves through the passes as the
         *  RAM @c advance_depth does, and answers whether another batch follows. */
        bool advance() noexcept {
            if (!running()) return false;
            round_counters_t const& counters = *counters_;
            u32_t const parity = last_pass_ & 1u;
            switch (step_) {
            case step_t::expand_k:
                if (counters.reached[parity]) return true;
                return next_step_(step_t::recheck_k);
            case step_t::recheck_k:
                if (counters.marked[parity]) return true;
                return next_step_(step_t::islands_k);
            case step_t::islands_k:
                if (!counters.islands) return next_step_(step_t::count_links_k);
                if (counters.marked[parity]) return true;
                island_pass_ = vram_island_pass_t::forward_k;
                return next_step_(step_t::components_k);
            case step_t::components_k: return advance_components_(counters);
            case step_t::count_links_k:
                if (!counters.islands || !counters.relabeled[0]) return next_step_(step_t::done_k);
                step_ = step_t::labels_k, next_pass_ = 1;
                return true;
            case step_t::labels_k:
                if (counters.relabeled[parity]) return true;
                return next_step_(step_t::done_k);
            default: return false;
            }
        }

        /** Entries the sweeps reached plus entries they marked a way to the root for, of twice the
         *  checked ones: what a caller's progress counts. */
        std::size_t swept() const noexcept { return counters_ ? static_cast<std::size_t>(counters_->swept) : 0; }

        /** Entries outside the root's component, as the last batch counted them. */
        std::size_t islands() const noexcept { return counters_ ? counters_->islands : 0; }

        /**
         *  @brief The report: the entries of the root's component, the depths each direction took,
         *      the in-link counts and the islands' components; @c internal_k while passes remain.
         *
         *  @par Statuses:
         *  - @c config_mismatch_k, @c device_k A batch failed, as it kept.
         */
        expected<reachability_t> finish() const noexcept {
            if (failed(status_)) return {{}, status_};
            if (!index_ || !visible) return {reachability_t {}, status_code_t::success_k};
            if (step_ != step_t::done_k) return {{}, status_code_t::internal_k};
            round_counters_t const& counters = *counters_;
            reachability_t result;
            result.audited_nodes = visible;
            result.reachable_nodes = visible - counters.islands;
            result.depth_from_root = counters.depth_from_root;
            result.depth_to_root = counters.depth_to_root;
            result.unlinked = counters.unlinked;
            result.underlinked = counters.underlinked;
            result.beyond_most_populated_depth = counters.beyond_most_populated_depth;
            result.unreached_islands = counters.islands ? counters.unreached_islands : 0;
            result.stranded_islands = counters.islands ? counters.stranded_islands : 0;
            result.components = counters.islands ? counters.components : 1;
            return {result, status_code_t::success_k};
        }

        /** Turns the check onto @p index 's IDs below @p visible, which its columns must span, and
         *  restarts it from a fresh draw. */
        void rebind(vram_amd_ragged_graph& index, node_id_t visible) noexcept {
            index_ = &index;
            this->visible = visible;
            root_ = unknown_k;
            restart_sweep();
        }

        /** Starts over from the root an earlier sweep kept, or else from a seeded draw; the next
         *  batch clears the columns first. */
        void restart_sweep() noexcept {
            // Opening and closing checks share their root, so a fresh draw cannot read as progress.
            bool const kept = root_ < visible;
            node_id_t const root = kept      ? root_
                                   : visible ? static_cast<node_id_t>(seeded_bijection(0, visible, 0))
                                             : unknown_k;
            drawn_root_ = kept ? unknown_k : root;
            start_from_(root);
        }

      private:
        friend struct vram_amd_ragged_graph;
        friend refine_plan_t;

        inline static constexpr node_id_t unknown_k = node_id_limits<node_id_t>::unknown_k;

        /** The check's steps, as the RAM check names its passes, plus the passes that label its
         *  islands on the device. */
        enum class step_t : u8_t { expand_k, recheck_k, islands_k, components_k, count_links_k, labels_k, done_k };

        /** Clears what one root's sweep fills, the next batch queuing the fills, and marks
         *  @p root both ways at depth zero; with no root the check is over. */
        void start_from_(node_id_t root) noexcept {
            root_ = root;
            next_pass_ = 0;
            step_ = root == unknown_k ? step_t::done_k : step_t::expand_k;
        }

        /** Turns to @p step from its first pass, answering whether a batch follows. */
        bool next_step_(step_t step) noexcept {
            step_ = step, next_pass_ = 0;
            return step != step_t::done_k;
        }

        /** The barrier inside the island components: colors forward until none lowers, marks back
         *  until none spreads, a settle; then again while islands stay unclaimed, else the re-root
         *  or the count. */
        bool advance_components_(round_counters_t const& counters) noexcept {
            u32_t const parity = last_pass_ & 1u;
            if (island_pass_ != vram_island_pass_t::settle_k) {
                if (counters.relabeled[parity]) return true;
                island_pass_ = island_pass_ == vram_island_pass_t::forward_k ? vram_island_pass_t::backward_k
                                                                             : vram_island_pass_t::settle_k;
                next_pass_ = 0;
                return true;
            }
            if (counters.unassigned) {
                island_pass_ = vram_island_pass_t::forward_k, next_pass_ = 0;
                return true;
            }
            std::size_t const component = visible - counters.islands;
            std::size_t const largest = static_cast<std::size_t>(counters.largest_island >> 32);
            if (root_ == drawn_root_ && largest > component) {
                drawn_root_ = unknown_k;
                start_from_(static_cast<node_id_t>(~static_cast<u32_t>(counters.largest_island)));
                return true;
            }
            return next_step_(step_t::count_links_k);
        }

        /** The check's and the round's device columns, carved once at planning. */
        round_scratch_t columns_;
        vram_amd_ragged_graph* index_ = nullptr;
        round_counters_ptr_t counters_ {nullptr, round_counters_return_t {}};

        /** The root the sweeps start from, which the closing check keeps; the draw it began with,
         *  which may give way once to the largest island. */
        node_id_t root_ = unknown_k, drawn_root_ = unknown_k;

        /** The depth, recheck or pass the next batch queues from, doubling as the batches do, and
         *  the last one the queued batch holds. */
        u32_t next_pass_ = 0, last_pass_ = 0;
        vram_island_pass_t island_pass_ = vram_island_pass_t::forward_k;
        step_t step_ = step_t::done_k;
        status_t status_ = status_code_t::success_k;
    };

    /**
     *  @brief One refinement round in flight on this graph's device - the dense device round over
     *      values of any length: @ref queue a phase, @ref wait for it, @ref advance past it, and
     *      @ref finish once no phase follows.
     *
     *  The round opens with a health check, heals the islands it named with no distance, draws one
     *  cache-sized set of hubs by seeded bijection, gathers every entry into the neighborhood of
     *  its nearest hub, and again inside every neighborhood over its capacity among hubs drawn from
     *  its own entries, then restitches each neighborhood: every entry keeps the links the kit's
     *  selector keeps among its own and its nearest neighborhood-mates and those reachability
     *  needs, never refilled. The closing health check from the same root decides the verdict. The
     *  hubs are drawn and the entries ordered by neighborhood on the host, at the barriers, over
     *  page-locked mirrors.
     *
     *  A phase queues its launches and their copies behind the context's event and never waits;
     *  planning made every allocation, so no phase allocates, and only a device fault fails one.
     *  Every phase takes the context the plan was opened on back from its caller. A plan
     *  @ref plan_refine opened holds the claim word closed for its life, so batches answer
     *  @c contended_k meanwhile while searches on other streams run throughout; a stopped round
     *  leaves the graph correct and merely less improved.
     */
    struct refine_plan_t {

        /** The check the round opens and closes with, which owns the device scratch. */
        health_plan_t health;

        /** An unbound plan over @p health - what a refused plan answers with. */
        explicit refine_plan_t(health_plan_t&& health) noexcept : health(std::move(health)) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : health(std::move(other.health)), index_(std::exchange(other.index_, nullptr)),
              claim_(std::exchange(other.claim_, claim_t::borrowed_k)), mirrors_(other.mirrors_), shape_(other.shape_),
              neighborhoods_(other.neighborhoods_), runs_(other.runs_), hubs_(other.hubs_),
              round_seed_(other.round_seed_), swept_after_(other.swept_after_), phase_(other.phase_),
              status_(other.status_), tally_(other.tally_) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** Reopens the claim word @ref plan_refine closed, whether the round finished or not. */
        ~refine_plan_t() noexcept {
            if (claim_ == claim_t::held_k) index_->reopen_();
        }

        /**
         *  @brief Queues the current phase on @p context 's stream, then its copies behind the
         *      context's event; never waits and never allocates anything at all.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers.
         *  - @c config_mismatch_k The @p context rides another device, or the plan keeps it.
         *  - @c device_k A launch or a copy failed, or the plan keeps it.
         */
        status_t queue(traversal_context_t& context) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            if (failed(status_)) return status_;
            switch (phase_) {
            case phase_t::health_k: status_ = health.queue(context); break;
            case phase_t::heal_k: status_ = index_->queue_heal_(context, health); break;
            case phase_t::gather_k: status_ = index_->queue_gather_(context, *this); break;
            case phase_t::restitch_k: status_ = index_->queue_restitch_(context, *this); break;
            // The barrier closing the draw draws the hubs, which the gather uploads with its runs.
            default: return status_code_t::success_k;
            }
            if (succeeded(status_)) status_ = context.record();
            return status_;
        }

        /** Sleeps until the queued phase finished, on @p context 's event. */
        status_t wait(traversal_context_t& context) noexcept {
            if (!index_ || phase_ == phase_t::done_k) return status_;
            if (succeeded(status_)) status_ = context.event_.wait();
            return status_;
        }

        /** The barrier after a phase finished: reads its counters or mirrors, picks the next phase,
         *  and answers whether one follows. */
        bool advance() noexcept {
            if (!index_ || failed(status_) || phase_ == phase_t::done_k) return false;
            switch (phase_) {
            case phase_t::health_k:
                if (health.advance()) return true;
                converge_();
                break;
            // A round after a stop heals and verifies alone.
            case phase_t::heal_k:
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
                verify_(phase_t::restitch_k);
                break;
            default: break;
            }
            return succeeded(status_) && phase_ != phase_t::done_k;
        }

        /** Skips what is left of the round's phase in hand: an opening health check, or a heal not
         *  yet run, ends the round with nothing written; a hubs draw, a gather or a restitch sends
         *  the round to the closing health check, the heal's links and the neighborhoods already
         *  re-stitched standing; the closing check runs on, and the verdict reads stopped. */
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

        /** The position of the round's phase in hand among its six, from the opening health check
         *  to the closing one; zero outside a round's phases. */
        std::size_t ordinal() const noexcept { return rounding() ? round_ordinal(phase_, swept_after_) : 0; }

        /** Units of the round's phase in hand done so far, at a barrier before it advances: entries
         *  swept and marked by a health check, and otherwise all of the phase's. */
        std::size_t completed() const noexcept {
            if (phase_ == phase_t::health_k) return health.swept();
            return total();
        }

        /** Units of the round's phase in hand in all, which @ref completed counts toward: islands
         *  healed, hubs drawn, entries one gather files, or entries re-stitched. */
        std::size_t total() const noexcept {
            switch (phase_) {
            case phase_t::health_k: return 2 * std::size_t {health.visible};
            case phase_t::heal_k: return health.islands();
            case phase_t::hubs_k: return round_first_hubs(health.visible, shape_.hubs_per_draw, shape_.capacity);
            case phase_t::gather_k: return runs_ ? mirrors_.run_entries[runs_] : 0;
            case phase_t::restitch_k: return health.visible;
            default: return 0;
            }
        }

        /** Opens the next round once one ended, from the root the closing health check kept: the
         *  report starts over, and the opening health check sweeps again; a no-op mid-round. */
        void next_round() noexcept {
            if (!index_ || phase_ != phase_t::done_k || failed(status_)) return;
            if (swept_after_ != phase_t::none_k && swept_after_ != phase_t::restitch_k) return;
            tally_ = {};
            verify_(phase_t::none_k);
        }

        /**
         *  @brief The round, complete once @ref advance answered false: its report, the walks'
         *      telemetry read off @p context 's last copy; a failure a phase kept answers instead.
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
        friend struct vram_amd_ragged_graph;
        friend reshard_plan_t;

        /** Opens the health check following @p phase from the root the last one kept. */
        void verify_(phase_t phase) noexcept {
            health.restart_sweep();
            swept_after_ = phase;
            phase_ = phase_t::health_k;
        }

        /** Turns the plan onto a fresh round over @p index 's published entries, which the columns
         *  must span; with none, the plan ends at once. */
        void bind_(vram_amd_ragged_graph& index) noexcept {
            index_ = &index;
            tally_ = {};
            neighborhoods_ = runs_ = hubs_ = 0;
            health.rebind(index, static_cast<node_id_t>(index.size()));
            swept_after_ = phase_t::none_k;
            phase_ = health.visible ? phase_t::health_k : phase_t::done_k;
        }

        /** Takes in a converged check: opens the round after the opening one, concludes it after
         *  the closing one. */
        void converge_() noexcept {
            auto const [reachability, status] = health.finish();
            // A check that failed fails the plan, never reads as no islands.
            if (failed(status)) {
                status_ = status;
                phase_ = phase_t::done_k;
                return;
            }
            tally_.reachability = reachability;
            if (swept_after_ == phase_t::none_k) open_round_(reachability);
            else close_round_();
        }

        /** Takes in the opening health check and opens the heal, the round's draws seeded from what
         *  it measured, so a changed graph draws anew while the ID order decides nothing. */
        void open_round_(reachability_t const& opening) noexcept {
            tally_.opening = opening;
            tally_.islands_found = opening.audited_nodes - opening.reachable_nodes;
            round_seed_ = mix_bits(
                opening.audited_nodes ^
                mix_bits(opening.reachable_nodes ^
                         mix_bits(opening.unlinked ^ mix_bits(opening.underlinked ^ opening.depth_from_root))));
            phase_ = health.visible ? phase_t::heal_k : phase_t::done_k;
        }

        /** Concludes the round once its closing health check converged, from the counts the heal
         *  and the restitch left: stopped as the caller left it, converged when it changed no
         *  entry's link set, improved otherwise. */
        void close_round_() noexcept {
            round_counters_t const& counters = *health.counters_;
            tally_.islands_reconnected = counters.healed;
            tally_.changed_nodes = counters.changed;
            tally_.added_links = counters.added_links;
            tally_.dropped_links = counters.dropped_links;
            tally_.verdict = concluded_verdict(tally_);
            phase_ = phase_t::done_k;
        }

        /** Draws one set of hubs by seeded bijection over the visible entries, ascending, as the
         *  first gather's one run over every ID. */
        void draw_hubs_() noexcept {
            node_id_t const visible = health.visible;
            std::size_t const hubs = round_first_hubs(visible, shape_.hubs_per_draw, shape_.capacity);
            for (std::size_t rank = 0; rank != hubs; ++rank)
                mirrors_.hub_nodes[rank] = static_cast<u32_t>(seeded_bijection(rank, visible, round_seed_));
            partial_sort(std::span<u32_t> {mirrors_.hub_nodes, hubs}, hubs);
            mirrors_.runs[0] = {0, static_cast<u32_t>(visible), 0, static_cast<u32_t>(hubs)};
            mirrors_.run_entries[0] = 0;
            mirrors_.run_entries[1] = static_cast<u32_t>(visible);
            runs_ = 1, hubs_ = hubs, neighborhoods_ = 0;
        }

        /**
         *  @brief Sorts the entries by neighborhood, the labels the gather copied down; answers
         *      whether a neighborhood over capacity was given hubs of its own, so the gather runs
         *      once more over those neighborhoods alone.
         *
         *  The first pass counts the entries by hub. Every later one splits each neighborhood over
         *  capacity by the labels its last gather gave, or cuts by ID one that gather left whole,
         *  so no neighborhood stays over capacity. Each still too wide is then one run of the next
         *  gather, its hubs drawn among its own entries as RAM draws them.
         */
        bool order_neighborhoods_() noexcept {
            node_id_t const visible = health.visible;
            std::span<u16_t const> const labels {mirrors_.entry_neighborhoods, visible};
            std::span<u32_t> const entries {mirrors_.neighborhood_entries, visible};
            std::span<u32_t> const offsets {mirrors_.neighborhood_offsets, shape_.offsets};
            neighborhood_order_t const order = neighborhoods_
                                                   ? split_wide_neighborhoods(labels, entries, offsets, neighborhoods_,
                                                                              shape_.capacity, shape_.hubs_per_draw)
                                                   : order_by_neighborhood(labels, hubs_, entries, offsets);
            std::size_t const count = order.neighborhoods;
            neighborhoods_ = count;
            tally_.neighborhoods = count;
            tally_.widest_neighborhood = order.widest;
            if (order.widest <= shape_.capacity) return false;
            std::size_t runs = 0, hubs = 0;
            mirrors_.run_entries[0] = 0;
            for (std::size_t neighborhood = 0; neighborhood != count; ++neighborhood) {
                u32_t const first = offsets[neighborhood], width = offsets[neighborhood + 1] - first;
                if (width <= shape_.capacity) continue;
                std::size_t const drawn = round_hubs(width, shape_.hubs_per_draw, shape_.capacity);
                u64_t const seed = mix_bits(round_seed_ ^ first);
                for (std::size_t rank = 0; rank != drawn; ++rank)
                    mirrors_.hub_nodes[hubs + rank] = entries[first + seeded_bijection(rank, width, seed)];
                mirrors_.runs[runs] = {first, width, static_cast<u32_t>(hubs), static_cast<u32_t>(drawn)};
                mirrors_.run_entries[runs + 1] = mirrors_.run_entries[runs] + width;
                ++runs, hubs += drawn;
            }
            runs_ = runs, hubs_ = hubs;
            return true;
        }

        vram_amd_ragged_graph* index_ = nullptr;

        /** Who holds @c index_ 's claim word, which the plan reopens at its death when it does. */
        claim_t claim_ = claim_t::borrowed_k;

        /** The page-locked mirrors the barriers order on. */
        round_mirrors_t mirrors_;

        /** The round's sizes, as RAM names them. */
        round_shape_t shape_;

        /** Neighborhoods the order holds, zero before the first gather's; the runs and hubs the
         *  next gather takes. */
        std::size_t neighborhoods_ = 0, runs_ = 0, hubs_ = 0;

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
     *  A gather copies every other source's nodes past this graph's watermark at planning, so its
     *  plan opens on the one target's round. A cut partitions on the device, then scatters every
     *  node into its target, the links inside its part rebased and the links across dropped, and
     *  the barrier after the scatter files the keys, publishes every target at once and empties
     *  the source; every target then runs one round, its opening heal linking what the cut left
     *  apart, and the rounds' reports fold. @ref next_round runs another round on every target,
     *  from the first.
     *
     *  The plan copies its target list and holds every claim word it closed for its life, its
     *  death reopening every one still closed; nothing may search the cut's source meanwhile.
     */
    struct reshard_plan_t {

        /** An unbound plan over @p rounds - what a refused plan answers with. */
        explicit reshard_plan_t(refine_plan_t&& rounds) noexcept
            : rounds_(std::move(rounds)),
              split_(typename traversal_context_t::bytes_ptr_t {nullptr, rounds_.health.columns_.bytes.get_deleter()}) {
        }
        reshard_plan_t(reshard_plan_t&& other) noexcept
            : rounds_(std::move(other.rounds_)), split_(std::move(other.split_)), targets_(std::move(other.targets_)),
              closed_targets_(std::exchange(other.closed_targets_, 0)), moves_(other.moves_),
              refined_targets_(other.refined_targets_), folded_targets_(other.folded_targets_),
              targets_tally_(other.targets_tally_) {}
        reshard_plan_t& operator=(reshard_plan_t&&) = delete;
        reshard_plan_t(reshard_plan_t const&) = delete;
        reshard_plan_t& operator=(reshard_plan_t const&) = delete;

        /** Reopens every claim word the plan closed, whether it finished or not. */
        ~reshard_plan_t() noexcept {
            for (std::size_t target = 0; target != closed_targets_; ++target) targets_[target]->reopen_();
            if (split_.source) split_.source->reopen_();
        }

        /**
         *  @brief Queues the current phase on @p context 's stream: a growth batch, the end of the
         *      partition or the scatter, then the bound target's round.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers.
         *  - Those of @ref refine_plan_t::queue.
         */
        status_t queue(traversal_context_t& context) noexcept {
            if (!rounds_.health.columns_.bytes) return status_code_t::uninitialized_k;
            // Targets left empty bind no round, which ends with nothing to queue.
            if (!moving()) return rounds_.index_ ? rounds_.queue(context) : rounds_.status_;
            if (failed(rounds_.status_)) return rounds_.status_;
            vram_amd_ragged_graph& source = *split_.source;
            auto const [graph_kernels, kernels_status] = source.graph_kernels_(context);
            if (failed(kernels_status)) return kernels_status;
            hip_stream_handle_t const stream = context.stream();
            status_t& status = rounds_.status_;
            if (moves_ == phase_t::scatter_k)
                status = source.queue_scatter_(*graph_kernels, split_, to_span(targets_), stream);
            else if (split_.next_round) status = source.queue_rounds_(*graph_kernels, split_, targets_.size(), stream);
            else status = source.queue_partition_end_(*graph_kernels, split_, targets_.size(), stream);
            if (succeeded(status)) status = context.record();
            return status;
        }

        /** Sleeps until the queued phase finished, on @p context 's event. */
        status_t wait(traversal_context_t& context) noexcept {
            if (!moving()) return rounds_.index_ ? rounds_.wait(context) : rounds_.status_;
            if (succeeded(rounds_.status_)) rounds_.status_ = context.event_.wait();
            return rounds_.status_;
        }

        /** The barrier after a phase finished: grows the parts on, assigns the slots, publishes the
         *  parts and binds the first target, or binds the next target once a round ended; answers
         *  whether a phase follows, never once the last target's round ended. */
        bool advance() noexcept {
            if (!rounds_.health.columns_.bytes || failed(rounds_.status_)) return false;
            if (moves_ == phase_t::partition_k) {
                // A growth batch that still placed nodes leaves others unplaced, so it grows on;
                // once one placed none, the rest are placed, and the host assigns the slots.
                if (!split_.next_round) assign_slots_(), moves_ = phase_t::scatter_k;
                else if (!placing_(split_, targets_.size())) split_.next_round = 0;
                return true;
            }
            if (moves_ == phase_t::scatter_k) {
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

        /** Whether the plan still moves a cut's nodes: its partition or its scatter. */
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
            if (!rounds_.health.columns_.bytes) return {{}, status_code_t::uninitialized_k};
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
        friend struct vram_amd_ragged_graph;

        /** Binds the round to the next target holding entries; a target after a stop heals and
         *  verifies alone. Past the last, reports the folded rounds and answers false. */
        bool bind_target_() noexcept {
            while (refined_targets_ != targets_.size() && !targets_[refined_targets_]->size()) ++refined_targets_;
            if (refined_targets_ == targets_.size()) {
                rounds_.tally_ = targets_tally_;
                rounds_.phase_ = phase_t::done_k;
                return false;
            }
            rounds_.bind_(*targets_[refined_targets_]);
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

        /** Assigns every source node its slot in place in the pinned host bytes, in ID order within
         *  each part, and lays out the targets' offsets columns. */
        void assign_slots_() noexcept {
            std::size_t const entries = split_.entries, parts = targets_.size();
            split_layout_t const layout = split_layout_(entries, parts);
            u32_t* const slots = reinterpret_cast<u32_t*>(split_.host_bytes);
            u64_t* const offsets = reinterpret_cast<u64_t*>(split_.host_bytes + layout.host_offsets);
            u64_t const* const starts = reinterpret_cast<u64_t const*>(split_.host_bytes + layout.host_starts);
            for (std::size_t part = 0; part != parts; ++part) split_.parts[part] = {};
            for (std::size_t node = 0; node != entries; ++node) ++split_.parts[slots[node]].entries;
            for (std::size_t part = 0, first = 0; part != parts; ++part) {
                split_.parts[part].offsets_first = first;
                offsets[first] = 0;
                first += split_.parts[part].entries + 1;
            }
            // A column's first entry counts its part's assigned nodes until the sums zero it.
            for (std::size_t node = 0; node != entries; ++node) {
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

        /** Moves every source key to its slot and publishes every target's nodes at once, its keys
         *  filed under the closed claim words, its tape's extent and longest value its part's and
         *  its spans the source's, then empties the source and reopens its claim word at zero. */
        void publish_parts_() noexcept {
            vram_amd_ragged_graph& source = *std::exchange(split_.source, nullptr);
            u32_t const* const slots = reinterpret_cast<u32_t const*>(split_.host_bytes);
            for (std::size_t node = 0; node != split_.entries; ++node)
                targets_[slots[node] / split_.part_capacity]->keyed_.assign(
                    static_cast<node_id_t>(slots[node] % split_.part_capacity),
                    source.keyed_[static_cast<node_id_t>(node)]);
            for (std::size_t index = 0; index != targets_.size(); ++index) {
                split_part_t const& part = split_.parts[index];
                vram_amd_ragged_graph& target = *targets_[index];
                target.tape_bytes_ = part.bytes, target.longest_value_ = part.longest;
                target.span_bytes_ = source.span_bytes_;
                target.publish_keys_({0, static_cast<node_id_t>(part.entries)});
                fu::standard_atomic_ref<std::size_t>(target.last_visible_id_)
                    .store(part.entries, std::memory_order_release);
            }
            source.forget_();
            source.reopen_();
        }

        /** The rounds over the bound target, which own the device columns. */
        refine_plan_t rounds_;

        /** A cut's moves; empty for a gather. */
        split_t split_;

        /** The targets, as the caller named them: this graph alone for a gather, the cut's parts'
         *  graphs otherwise; and how many of them, from the first, the plan closed. */
        growing_buffer<vram_amd_ragged_graph*> targets_ {global_allocator};
        std::size_t closed_targets_ = 0;

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
     *  Closes the claim word, carves the check's columns and the round's from the context, takes
     *  page-locked counters and mirrors, grows the context's arena for the restitch's scratch, and
     *  sizes the hubs and the neighborhood capacity from @c cache_bytes and the mean stored value,
     *  never past what this device's multiprocessor keeps. No phase allocates.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet.
     *  - @c unsupported_k The shared kernels lack this node-ID type.
     *  - @c config_mismatch_k The @p context rides another device or cannot launch.
     *  - @c contended_k A batch, a load or another plan in flight.
     *  - @c out_of_memory_k A column, a mirror, the arena or the counters cannot be placed.
     */
    expected<refine_plan_t> plan_refine(refine_config_t config, traversal_context_t& context) noexcept {
        refine_plan_t plan {health_plan_t {empty_round_scratch_(context)}};
        if (status_t const refused = graph_kernels_(context).status; failed(refused)) return {std::move(plan), refused};
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        plan.index_ = this, plan.claim_ = claim_t::held_k;
        round_shape_t const shape = round_shape_(size(), mean_value_bytes_(), config.cache_bytes);
        std::size_t const arena_bytes = round_arena_bytes_(round_launch_(longest_value_, span_bytes_));
        // A refused plan keeps the claim word to reopen it, and its status to refuse every phase.
        if (status_t const carved = carve_round_(plan, context, shape, arena_bytes, 0); failed(carved)) {
            plan.status_ = carved;
            return {std::move(plan), carved};
        }
        plan.bind_(*this);
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
     *  A gather copies every other source's values, offsets and neighbor rows past this graph's
     *  watermark on the device behind one wait, its offsets shifted on the host and its links
     *  rebased on the device, files their keys, turns the watermark over them and empties those
     *  sources, all before the plan returns; their claim words close for the copies alone. A cut
     *  queues its partition behind every allocation: one root per part through
     *  @ref seeded_bijection under the config's seed, then rounds of growth over the links, each
     *  part capped at the nodes over the targets rounded up, each node a round reaches kept by its
     *  lowest-ID proposer, so equal seeds over equal graphs split alike, and nodes no round reached
     *  entering the smallest parts in ID order. Planning maps each target's tape for the most its
     *  part can take, never past this graph's tape. Every graph must share this graph's device and
     *  metric, and its connectivity or, for a cut's target, none yet, which planning reserves.
     *  Every listed graph must outlive both the call and the plan's finish.
     *
     *  @param[in] sources The graphs giving their nodes up: this graph alone for a cut, any others
     *      for a gather, which keeps this graph's own nodes when it is among them.
     *  @param[in] targets This graph alone for a gather, or at least two empty graphs for a cut.
     *  @param[in] graph The member of each listed owner holding its graph.
     *  @param[in] config The rounds' knobs and the seed picking a cut's part roots.
     *  @param[in,out] context The driving context, which every phase takes back; the moves' scratch
     *      and the rounds' are carved from it.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k An empty list, a null or a repeated graph within one.
     *  - @c unsupported_k Several sources and several targets, or a nonzero
     *    @c reshard_config_t::shard_capacity, as a graph holds no shards.
     *  - @c config_mismatch_k Neither a gather into this graph nor a cut of it, a cut target that
     *    is this graph or not empty, or a graph of another device, connectivity or metric.
     *  - @c contended_k A batch, a load or a plan in flight on a listed graph.
     *  - @c overflow_k Nodes past the reserved ceiling, or part slots past the 32-bit IDs.
     *  - @c out_of_memory_k A column, a mirror, the arena or the counters cannot be placed.
     */
    template <typename owner_type_>
    expected<reshard_plan_t> plan_reshard(std::span<owner_type_* const> sources, std::span<owner_type_* const> targets,
                                          vram_amd_ragged_graph owner_type_::* graph, reshard_config_t config,
                                          traversal_context_t& context) noexcept {
        reshard_plan_t plan {refine_plan_t {health_plan_t {empty_round_scratch_(context)}}};
        if (sources.empty() || targets.empty()) return {std::move(plan), status_code_t::invalid_argument_k};
        for (std::span<owner_type_* const> const listed : {sources, targets})
            if (status_t const refused = listed_refusal(listed); failed(refused)) return {std::move(plan), refused};
        if (sources.size() > 1 && targets.size() > 1) return {std::move(plan), status_code_t::unsupported_k};
        if (config.shard_capacity) return {std::move(plan), status_code_t::unsupported_k};
        bool const gathering = targets.size() == 1 && &(targets[0]->*graph) == this;
        bool const cutting = sources.size() == 1 && &(sources[0]->*graph) == this && targets.size() > 1;
        if (!gathering && !cutting) return {std::move(plan), status_code_t::config_mismatch_k};
        if (status_t const refused = graph_kernels_(context).status; failed(refused)) return {std::move(plan), refused};
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
                                      vram_amd_ragged_graph owner_type_::* graph, reshard_config_t config,
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
     *  the context once per batch of passes, and every island's component is named.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet.
     *  - @c unsupported_k The shared kernels lack this node-ID type.
     *  - @c config_mismatch_k The @p context rides another device or cannot launch.
     *  - @c out_of_memory_k The columns or the counters cannot be placed.
     */
    expected<reachability_t> health(traversal_context_t& context) noexcept {
        std::size_t const entries = size();
        if (!entries) return {{}, status_code_t::success_k};
        if (status_t const refused = graph_kernels_(context).status; failed(refused)) return {{}, refused};
        health_plan_t plan {empty_round_scratch_(context)};
        if (status_t const carved = carve_health_(plan, context, {entries}); failed(carved)) return {{}, carved};
        plan.rebind(*this, static_cast<node_id_t>(entries));
        do
            if (failed(plan.queue(context)) || failed(plan.wait(context))) break;
        while (plan.advance());
        return plan.finish();
    }

    /** The bytes an executor's run must hold before a context carved over it can run rounds over
     *  @p nodes nodes under @p config: the context's counters, the restitch arena, the columns. */
    std::size_t traversal_memory_usage(refine_config_t config, std::size_t nodes) const noexcept {
        if (!connectivity() || !device_.multiprocessor_count) return 0;
        round_shape_t const shape = round_shape_(nodes, mean_value_bytes_(), config.cache_bytes);
        return aligned_(sizeof(vram_telemetry_block_t)) + 2 * aligned_(sizeof(u32_t)) +
               aligned_(round_arena_bytes_(round_launch_(longest_value_, span_bytes_))) +
               aligned_(lay_out_round_(shape).device_bytes);
    }

    /** The bytes an executor's run must hold before a context carved over it can cut @p nodes
     *  nodes into @p targets targets under @p config: rounds over the widest part, and the
     *  partition's scratch. */
    std::size_t traversal_memory_usage(refine_config_t config, std::size_t nodes, std::size_t targets) const noexcept {
        if (!connectivity() || !device_.multiprocessor_count || targets < 2) return 0;
        return traversal_memory_usage(config, divide_round_up(nodes, targets)) +
               aligned_(split_layout_(nodes, targets).device_bytes);
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
        explicit save_plan_t(hip_api_t const& api) noexcept
            : bounce_ {hip_pinned_allocator_t {api}}, crossed_ {hip_event_t {api}, hip_event_t {api}} {}
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
         *  @brief Copies the neighbor IDs and the values of @p nodes off the device into the slices
         *      the prefix sums name, half a bounce at a time; the last half is still copying when
         *      the call returns to its caller.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The @p nodes pass the plan, or a node's kept neighbors or its
         *    saved value disagree with the offsets the plan measured.
         */
        status_t save(iota_view<node_id_t> nodes) noexcept {
            if (!index_ || neighbor_offsets_.empty()) return status_code_t::uninitialized_k;
            if (nodes.bound < nodes.first || nodes.bound > extents.nodes) return status_code_t::config_mismatch_k;
            for (node_id_t first = nodes.first; first != nodes.bound;) {
                node_id_t const bound = crossing_end_(to_span(stored_offsets_), first, nodes.bound, half_nodes_,
                                                      half_bytes_);
                // Every half but the pending one has finished, so the other one is free.
                std::size_t const half = pending_.empty() ? 0 : pending_half_ ^ 1;
                if (status_t const fetched = fetch_({first, bound}, half); failed(fetched)) return fetched;
                if (status_t const copied = copy_pending_(); failed(copied)) return copied;
                pending_ = {first, bound}, pending_half_ = half, first = bound;
            }
            return status_code_t::success_k;
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
            if (!index_ || neighbor_offsets_.empty()) return status_code_t::uninitialized_k;
            if (status_t const copied = copy_pending_(); failed(copied)) return copied;
            if (covered_ != extents.nodes) return status_code_t::config_mismatch_k;
            std::exchange(index_, nullptr)->reopen_();
            return status_code_t::success_k;
        }

      private:
        friend struct vram_amd_ragged_graph;

        /** Queues the copies of @p nodes ' neighbor rows and stored values into half @p half of the
         *  bounce, recorded on its event. */
        status_t fetch_(iota_view<node_id_t> nodes, std::size_t half) noexcept {
            std::size_t const row_bytes = index_->connectivity() * sizeof(node_id_t);
            byte_t* const neighbors = bounce_.data() + half * (neighbors_bytes_ + half_bytes_);
            u64_t const first_byte = stored_offsets_[nodes.first];
            u64_t const bytes = stored_offsets_[nodes.bound] - first_byte;
            if (failed(hip_vendor_t::copy(index_->api(), neighbors,
                                          index_->neighbors_matrix_.base() + nodes.first * row_bytes,
                                          nodes.size() * row_bytes, stream_)) ||
                (bytes && failed(hip_vendor_t::copy(index_->api(), neighbors + neighbors_bytes_,
                                                    index_->values_tape_.base() + first_byte, bytes, stream_))))
                return status_code_t::device_k;
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
            return status_code_t::success_k;
        }

        /** Sizes every node under the watermark, half a bounce at a time, writing both prefix sums
         *  when they are given, @b [nodes+1] each. */
        status_t size_(std::span<u64_t> neighbor_sums, std::span<u64_t> value_sums) noexcept {
            node_id_t const nodes = static_cast<node_id_t>(extents.nodes);
            extents.neighbors = 0, extents.value_bytes = 0;
            if (!neighbor_sums.empty()) neighbor_sums[0] = 0, value_sums[0] = 0;
            auto const sized = [&](node_id_t node, std::span<node_id_t const> slots,
                                   bytes_view_t stored) noexcept -> status_t {
                auto const [saved_bytes, saved_status] = index_->kits_.saved_bytes_(stored);
                if (failed(saved_status)) return saved_status;
                for (node_id_t const neighbor : slots)
                    if (kept_link_(neighbor, node, nodes)) ++extents.neighbors;
                extents.value_bytes += saved_bytes;
                if (!neighbor_sums.empty())
                    neighbor_sums[node + 1] = extents.neighbors, value_sums[node + 1] = extents.value_bytes;
                return status_code_t::success_k;
            };
            for (node_id_t first = 0; first != nodes;) {
                node_id_t const bound = crossing_end_(to_span(stored_offsets_), first, nodes, half_nodes_, half_bytes_);
                if (status_t const fetched = fetch_({first, bound}, 0); failed(fetched)) return fetched;
                if (status_t const visited = visit_({first, bound}, 0, sized); failed(visited)) return visited;
                first = bound;
            }
            return status_code_t::success_k;
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
            if (pending_.empty()) return status_code_t::success_k;
            iota_view<node_id_t> const pending = std::exchange(pending_, {});
            node_id_t const nodes = static_cast<node_id_t>(extents.nodes);
            auto const on_staged = [&](node_id_t node, std::span<node_id_t const> slots,
                                       bytes_view_t stored) noexcept -> status_t {
                u64_t cursor = neighbor_offsets_[node];
                u64_t const bound = neighbor_offsets_[node + 1];
                for (node_id_t const neighbor : slots) {
                    if (!kept_link_(neighbor, node, nodes)) continue;
                    if (cursor == bound) return status_code_t::config_mismatch_k;
                    neighbor_ids_[cursor++] = neighbor;
                }
                if (cursor != bound) return status_code_t::config_mismatch_k;
                return index_->kits_.save_value_(
                    stored, values_.subspan(value_offsets_[node], value_offsets_[node + 1] - value_offsets_[node]));
            };
            if (status_t const visited = visit_(pending, pending_half_, on_staged); failed(visited)) return visited;
            covered_ += pending.size();
            return status_code_t::success_k;
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
        growing_buffer<byte_t, hip_pinned_allocator_t> bounce_ {hip_pinned_allocator_t {}};

        /** Marks the last copies into each half of the bounce, born with the plan on its device. */
        hip_event_t crossed_[2];
        hip_stream_handle_t stream_ = nullptr;
        std::size_t half_nodes_ = 0, half_bytes_ = 0, neighbors_bytes_ = 0;

        /** The nodes whose copies sit in a half not yet on the host, and that half. */
        iota_view<node_id_t> pending_ {};
        std::size_t pending_half_ = 0;

        /** Nodes whose neighbors and values were copied. */
        std::size_t covered_ = 0;

        /** The index being snapshotted, whose claim word the plan holds closed; null in a refused
         *  plan, and once @c finish succeeded. */
        vram_amd_ragged_graph* index_ = nullptr;
    };

    /**
     *  @brief Node & element counts a save needs right now, through one sizing pass over every node
     *      under the watermark with the claim word closed; @c contended_k while a batch, a load or
     *      a plan is in flight.
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
     *  @brief Serializes every node under the watermark into @p columns on the calling thread - the
     *      one-shot over @ref plan_save.
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
     *  saved neighbors, offsets and values into a slot through a driver copy, while the other
     *  slot's work runs, copies their planned tape starts beside them, widens their neighbor rows
     *  on the device, and hands the kits the run to store its values. Keys go into the host key
     *  column. @c finish is the one full wait: it reads back whether a neighbor row was malformed
     *  and the widest span, files the keys and turns the watermark over the nodes. Holds the claim
     *  for its whole life; a plan that dies before @c finish gives its claim back and leaves the
     *  storage grown, its nodes unfilled.
     */
    struct [[nodiscard]] load_plan_t {

        /** The first appended ID, where node zero of the columns goes. */
        node_id_t base = 0;

        /** Nodes the plan appends. */
        std::size_t entries = 0;

        load_plan_t() noexcept = default;

        /** Births its events through @p api on the current device. */
        explicit load_plan_t(hip_api_t const& api) noexcept : drained_ {hip_event_t {api}, hip_event_t {api}} {}
        load_plan_t(load_plan_t&& other) noexcept
            : base(other.base), entries(std::exchange(other.entries, 0)), saved_(other.saved_),
              stored_offsets_(std::move(other.stored_offsets_)),
              drained_ {std::move(other.drained_[0]), std::move(other.drained_[1])},
              device_words_(std::move(other.device_words_)), staging_(std::move(other.staging_)),
              slot_bytes_(other.slot_bytes_), stream_(other.stream_), graph_kernels_(other.graph_kernels_),
              kits_kernels_(other.kits_kernels_), longest_(other.longest_), span_bytes_(other.span_bytes_),
              chunks_(other.chunks_), covered_(other.covered_), index_(std::exchange(other.index_, nullptr)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** An unfinished plan awaits every slot's work, so the next grow fills its nodes again, and
         *  gives its claim back. */
        ~load_plan_t() noexcept {
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
         *  Driver copies fill a slot in stream order, behind the work out of it.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The @p range passes the plan.
         */
        status_t load(iota_view<node_id_t> range) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            if (range.bound < range.first || range.bound > entries) return status_code_t::config_mismatch_k;
            hip_api_t const& api = index_->api();
            std::size_t const key_bytes = dtype_bytes(saved_.key_dtype);
            for (node_id_t first = range.first; first != range.bound;) {
                node_id_t const bound = staged_end_(first, range.bound);
                std::size_t const slot = chunks_++ % 2;
                std::size_t into = slot * slot_bytes_;
                auto const read = [&](bytes_view_t extent) noexcept -> expected<bytes_view_t> {
                    byte_t* const staged = staging_.get() + into;
                    into += staged_bytes_(extent.size());
                    if (failed(hip_vendor_t::copy(api, staged, extent.data(), extent.size(), stream_)))
                        return {{}, status_code_t::device_k};
                    return {bytes_view_t {staged, extent.size()}, status_code_t::success_k};
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
                vram_amd_load_step<node_id_t> step;
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
                step.warp_lanes = index_->device_.warp_lanes;
                // The planned tape starts cross per run, 8 bytes a node, from the host's plan.
                if (failed(hip_vendor_t::copy(api, index_->offsets_column_.base() + (base + first) * sizeof(u64_t),
                                              stored_offsets_.data() + first, (bound - first + 1) * sizeof(u64_t),
                                              stream_)) ||
                    failed(index_->launch_graph_(*graph_kernels_, vram_kernel_t::widen_links_k,
                                                 index_->graph_blocks_(vram_kernel_t::widen_links_k), graph,
                                                 stream_)) ||
                    failed(index_->kits_.queue_load_values_(step)) || failed(drained_[slot].record(stream_)))
                    return status_code_t::device_k;
                for (node_id_t node = first; node != bound; ++node)
                    index_->keyed_.assign(
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
            if (!index_) return status_code_t::uninitialized_k;
            if (covered_ != entries) return status_code_t::config_mismatch_k;
            if (failed(drained_[0].wait()) || failed(drained_[1].wait())) return status_code_t::device_k;
            u64_t words[2] = {0, 0};
            if (entries &&
                index_->api().copy(words, device_words_.get(), sizeof(words), hip_copy_default_k) != hip_success_k)
                return status_code_t::device_k;
            if (words[0]) return status_code_t::malformed_k;
            vram_amd_ragged_graph& index = *std::exchange(index_, nullptr);
            index.tape_bytes_ = stored_offsets_[entries];
            index.longest_value_ = max_of(index.longest_value_, longest_);
            index.span_bytes_ = max_of(index.span_bytes_, max_of(span_bytes_, static_cast<std::size_t>(words[1])));
            index.publish_keys_({base, static_cast<node_id_t>(base + entries)});
            fu::standard_atomic_ref<std::size_t>(index.last_visible_id_)
                .store(base + entries, std::memory_order_release);
            return status_code_t::success_k;
        }

      private:
        friend struct vram_amd_ragged_graph;

        /** Bytes a read of @p bytes takes in a slot, whole alignment lines. */
        static std::size_t staged_bytes_(std::size_t bytes) noexcept {
            return round_up_to_multiple(bytes, default_alignment_k);
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
        hip_event_t drained_[2];

        /** On the device: the malformed word, nonzero once a neighbor row is, then the widest span
         *  the kits counted. */
        hip_unique_ptr<u64_t> device_words_;

        /** The slots runs cross, alternating, back to back, and the bytes each holds. */
        hip_unique_ptr<byte_t> staging_;
        std::size_t slot_bytes_ = 0;

        hip_stream_handle_t stream_ = nullptr;

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
        vram_amd_ragged_graph* index_ = nullptr;
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
    expected<load_plan_t> plan_load(safegraphs_saved_columns<node_id_t> const& saved,
                                    traversal_context_t& context) noexcept {
        if (hip_set_device(api(), device_.ordinal) != hip_success_k) return {{}, status_code_t::device_k};
        load_plan_t plan {api()};
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

        // Every stored value's size up front, from its saved length alone, so the tape and the
        // runs are known before the claim.
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
        auto const [graph_kernels, graph_status] = graph_kernels_(context);
        if (failed(graph_status)) return {std::move(plan), graph_status};
        auto const [kits_kernels, kits_status] = kits_kernels_(context);
        if (failed(kits_status)) return {std::move(plan), kits_status};

        auto const [base, claim_status] = claim_(entries);
        if (failed(claim_status)) return {std::move(plan), claim_status};
        plan.index_ = this, plan.base = base, plan.entries = entries;
        hip_stream_handle_t const stream = context.stream();
        std::size_t const stored_total = plan.stored_offsets_[entries];
        if (status_t const grown = grow_nodes_(base + entries, stream); failed(grown)) return {std::move(plan), grown};
        if (status_t const mapped = values_tape_.commit(tape_bytes_ + stored_total + tape_margin_k); failed(mapped))
            return {std::move(plan), mapped};
        for (u64_t& offset : plan.stored_offsets_) offset += tape_bytes_;
        plan.saved_ = saved, plan.stream_ = stream;
        plan.graph_kernels_ = graph_kernels, plan.kits_kernels_ = kits_kernels;
        plan.span_bytes_ = kits_.span_bytes_(plan.longest_);
        if (!entries) return {std::move(plan), status_code_t::success_k};

        // A slot fits the run a save's crossing carries, and at least the widest node; a second
        // slot opens only for a second run.
        plan.slot_bytes_ = plan.run_bytes_(0, min_of(entries, power_of_two<std::size_t>(crossing_nodes_log2_k)));
        for (std::size_t node = 0; node != entries; ++node)
            plan.slot_bytes_ = max_of(plan.slot_bytes_, plan.run_bytes_(node, node + 1));
        std::size_t const staging_bytes = (plan.run_bytes_(0, entries) > plan.slot_bytes_ ? 2 : 1) * plan.slot_bytes_;
        plan.staging_ = hip_make_unique(hip_device_allocator_t {context.allocator()}, staging_bytes);
        if (!plan.staging_) return {std::move(plan), status_code_t::out_of_memory_k};
        plan.device_words_ = hip_make_unique(hip_device_allocator<u64_t> {api()}, 2);
        if (!plan.device_words_) return {std::move(plan), status_code_t::out_of_memory_k};
        if (failed(hip_vendor_t::fill(api(), plan.device_words_.get(), 0, 2 * sizeof(u64_t), stream)))
            return {std::move(plan), status_code_t::device_k};
        return {std::move(plan), status_code_t::success_k};
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

    /** Threads per block, one wavefront's worth per resident wavefront. */
    u32_t block_threads() const noexcept { return vram_warps_per_block_k * wave_lanes(); }

    /** Shared bytes a kernel keeps beside its scratch: four wavefronts' batches of three
     *  lane-widths of IDs, a round of two scored together and its carry, three per thread. */
    std::size_t static_shared_bytes() const noexcept { return 3 * std::size_t {block_threads()} * sizeof(node_id_t); }

    /** Lanes per wavefront, the width every kernel's collectives span: 64 on CDNA3 and CDNA4, 32 on
     *  CDNA5, as the runtime reports them, which the class's kits are compiled for. */
    u32_t wave_lanes() const noexcept { return device_.warp_lanes; }

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
            std::size_t {device_.multiprocessor_count} * hip_resident_blocks(vram_kernel_t::search_k, slots)));
    }

    /** Which frontier a walk keeps: a register rung, or the arena's past the ladder. */
    enum class frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

    /** The breadth an insert or a refine walks: as given, or twice the connectivity, as on the
     *  CPU, and never under the @c build_breadth_k a modality's selector needs. */
    u32_t build_expansion_(u32_t expansion_factor) const noexcept {
        if (expansion_factor) return expansion_factor;
        u32_t const twice = static_cast<u32_t>(default_expansion_factor(connectivity()));
        constexpr bool widened = requires { kits_type_::build_breadth_k; };
        if constexpr (widened) return max_of(twice, kits_type_::build_breadth_k);
        else return twice;
    }

    /** Candidates a walk's selection caches: all of its candidates, @p candidate_slots ranks per
     *  lane on the register ladder, or the arena frontier's @p expansion past it. */
    u32_t walk_selection_nodes_(u32_t expansion, u32_t candidate_slots) const noexcept {
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
     *  hold, each launch's per-warp shared scratch, zero when the arena slice keeps both, and the
     *  arena slice itself. */
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
                                        static_shared_bytes();
        bool const fits = block_bytes <= device_.shared_memory_per_block_bytes &&
                          // HIP keeps no per-block reserve of the local data share.
                          hip_resident_blocks(vram_kernel_t::insert_k, 0) * block_bytes <=
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
        bool const fits = walkers_per_block_(rung) * span_bytes + static_shared_bytes() <=
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
     *  @brief Covers the host keys and maps the offsets column and the neighbors matrix up to
     *      @p nodes, host work alone, so it rides no stream, and leaves the fresh neighbor rows for
     *      @c grow_nodes_ to fill.
     *
     *  @par Statuses:
     *  - @c overflow_k The @p nodes pass the reserved range or the host keys.
     */
    status_t map_nodes_(std::size_t nodes) noexcept {
        if (status_t const covered = keyed_.cover(nodes); failed(covered)) return covered;
        if (nodes <= capacity()) return status_code_t::success_k;
        if (status_t const mapped = offsets_column_.commit((nodes + 1) * sizeof(u64_t)); failed(mapped)) return mapped;
        return neighbors_matrix_.commit(nodes * connectivity() * sizeof(node_id_t));
    }

    /** Maps up to @p nodes and sentinel-fills on @p stream every mapped neighbor row no fill
     *  reached yet, @p stream first following the last fill, whichever stream queued it; callers
     *  hold the claim. */
    status_t grow_nodes_(std::size_t nodes, hip_stream_handle_t stream) noexcept {
        if (status_t const mapped = map_nodes_(nodes); failed(mapped)) return mapped;
        hip_api_t const& driver = api();
        if (driver.stream_wait_event(stream, grown_, 0) != hip_success_k) return status_code_t::device_k;
        if (nodes <= filled_nodes_) return status_code_t::success_k;
        // The sentinel is all ones at any width, so the runtime's 4-byte words fill IDs of either.
        constexpr std::size_t word_bytes = 4;
        std::size_t const mapped_nodes = capacity(), row_words = connectivity() * sizeof(node_id_t) / word_bytes;
        if (driver.fill_words_async(neighbors_matrix_.base() + filled_nodes_ * row_words * word_bytes, ~0,
                                    (mapped_nodes - filled_nodes_) * row_words, stream) != hip_success_k ||
            failed(grown_.record(stream)))
            return status_code_t::device_k;
        filled_nodes_ = mapped_nodes;
        return status_code_t::success_k;
    }

    /**
     *  @brief Launches the batch @p build stages, whose @p offsets the caller spelled, from
     *      @p first_id on in waves, each the share @c vram_basic_admission_share_log2_k of the IDs
     *      below it, every launch sized by @p shape.
     *
     *  Each wave queues behind the last and admits at least @c vram_admission_ramp_floor_k nodes.
     *  Every wave walks twice, inserting and then re-selecting, and each walk is followed by its
     *  back-link launch over every node its picks touched. A failure after any kernel launched
     *  keeps the failure, since those nodes stay claimed.
     */
    status_t launch_insert_(kernel_table_t const& kits, vram_build<node_id_t> const& build, build_shape_t const& shape,
                            std::span<std::size_t const> offsets, node_id_t first_id, std::size_t rung,
                            hip_stream_handle_t stream) noexcept {
        std::size_t const entries = offsets.size() - 1;
        node_id_t const end_id = static_cast<node_id_t>(first_id + entries);
        // Every launch carves its warps' slices from the one arena, sized for the insert's grid,
        // which the re-selection's and the back-link's resident counts match.
        static_assert(
            hip_resident_blocks(vram_kernel_t::refine_k, 0) == hip_resident_blocks(vram_kernel_t::insert_k, 0) &&
            hip_resident_blocks(vram_kernel_t::back_link_k, 0) == hip_resident_blocks(vram_kernel_t::insert_k, 0));
        u32_t const blocks = device_.multiprocessor_count *
                             hip_resident_blocks(vram_kernel_t::insert_k, shape.candidate_slots);
        vram_arguments<node_id_t> arguments {};
        status_t launched = status_code_t::success_k;
        bool published = false;
        auto const queue = [&](vram_kernel_t kernel, u32_t slots, node_id_t first) noexcept {
            bool const walks = kernel != vram_kernel_t::back_link_k;
            arguments.build.first_id = first;
            arguments.build.selection_nodes = walks ? shape.walk_selection_nodes : shape.reselection_nodes;
            arguments.build.scratch_bytes_per_warp = walks ? shape.walk_scratch_bytes : shape.back_link_scratch_bytes;
            if (succeeded(launched))
                launched = succeeded(hip_vendor_t::fill(api(), build.assigned_nodes, 0, sizeof(u32_t), stream))
                               ? hip_launch(api(), kits_function_at_(kits, rung, kernel, slots), blocks,
                                            block_threads(),
                                            walkers_per_block_(rung) * arguments.build.scratch_bytes_per_warp, stream,
                                            arguments)
                               : status_code_t::device_k;
            published = published || succeeded(launched);
        };
        auto const zero_touched = [&]() noexcept {
            if (succeeded(launched) && failed(hip_vendor_t::fill(api(), build.touched_count, 0, sizeof(u32_t), stream)))
                launched = status_code_t::device_k;
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
        return status_code_t::success_k;
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

    /** The end of the crossing from @p first toward @p bound: the nodes one half holds, at least
     *  one, @p half_nodes at most and @p half_bytes at most of the values @p offsets lay out. */
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

    /** Opens a save of every node under the watermark with the claim word closed, the index's
     *  device selected first, where the plan's events are born: copies the device's value starts on
     *  the host and pins the bounce, sizing nothing yet. */
    expected<save_plan_t> open_save_(traversal_context_t& context) noexcept {
        if (hip_set_device(api(), device_.ordinal) != hip_success_k) return {{}, status_code_t::device_k};
        save_plan_t plan {api()};
        if (context.device() != device_.ordinal || !context) return {std::move(plan), status_code_t::config_mismatch_k};
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        plan.index_ = this;
        std::size_t const entries = size();
        plan.extents.nodes = entries;
        plan.stream_ = context.stream();
        if (failed(plan.stored_offsets_.resize_uninitialized(entries + 1)))
            return {std::move(plan), status_code_t::out_of_memory_k};
        plan.stored_offsets_[0] = 0;
        // Every batch drained before publishing, so the column holds every start below the mark.
        if (entries && api().copy(plan.stored_offsets_.data(), offsets_column_.base(), (entries + 1) * sizeof(u64_t),
                                  hip_copy_default_k) != hip_success_k)
            return {std::move(plan), status_code_t::device_k};
        plan.half_nodes_ = min_of(entries, power_of_two<std::size_t>(crossing_nodes_log2_k));
        plan.half_bytes_ = crossing_bytes_(tape_bytes_, longest_value_, entries, plan.half_nodes_);
        plan.neighbors_bytes_ = aligned_(plan.half_nodes_ * connectivity() * sizeof(node_id_t));
        if (entries && failed(plan.bounce_.resize_uninitialized(2 * (plan.neighbors_bytes_ + plan.half_bytes_))))
            return {std::move(plan), status_code_t::out_of_memory_k};
        return {std::move(plan), status_code_t::success_k};
    }

    /** The kits' loaded @p kernel of @p kits on kit rung @p rung and register rung @p slots. */
    static hip_function_handle_t kits_function_at_(kernel_table_t const& kits, std::size_t rung, vram_kernel_t kernel,
                                                   u32_t slots) noexcept {
        return kits.functions[hip_kits_kernel_speller_t::index(rung, kernel, slots)];
    }

    /** Spells the shared @p kernel over this graph's node-ID type, as @ref hip_graph_code_object
     *  exports it. */
    static void spell_graph_kernel_(vram_kernel_symbol_t& symbol, vram_kernel_t kernel) noexcept {
        vram_kernel_symbol(symbol, "basic", "amd_kernel", {"vram_graph", 1}, kernel, itanium_type<node_id_type_> {});
    }

    /**
     *  @brief The shared kernels on @p context 's stream, loaded there at the first ask and kept
     *      until the stream dies, for a round, a reshard's moves, a health check or a gather;
     *      every such verb resolves them once per call.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No @c reserve froze the connectivity yet, or the @p context rides no
     *    executor stream.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk.
     *  - @c unsupported_k The shared kernels lack this node-ID type.
     *  - @c device_k The code object would not load.
     */
    expected<kernel_table_t const*> graph_kernels_(traversal_context_t const& context) const noexcept {
        if (!connectivity()) return {nullptr, status_code_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {nullptr, status_code_t::config_mismatch_k};
        auto const [kernels, load_status] = context.table_of(hip_graph_code_object<kits_type_::capability_k>::bytes(),
                                                             hip_graph_kernel_speller_t {&spell_graph_kernel_});
        if (failed(load_status)) return {nullptr, load_status};
        for (vram_kernel_t const kernel : amd_graph_kernels)
            if (!kernels->functions[hip_graph_kernel_speller_t::index(kernel)])
                return {nullptr, status_code_t::unsupported_k};
        return {kernels, status()};
    }

    /** Blocks one grid-striding pass of @p kernel over the neighbors matrix launches. */
    u32_t graph_blocks_(vram_kernel_t kernel) const noexcept {
        return device_.multiprocessor_count * hip_resident_blocks(kernel, 0);
    }

    /** Launches the shared @p kernel of @p graph_kernels over @p blocks blocks on @p stream. */
    status_t launch_graph_(kernel_table_t const& graph_kernels, vram_kernel_t kernel, u32_t blocks,
                           vram_graph<node_id_t> const& graph, hip_stream_handle_t stream) noexcept {
        return hip_launch(api(), graph_kernels.functions[hip_graph_kernel_speller_t::index(kernel)], blocks,
                          block_threads(), 0, stream, graph);
    }

    /** Scratch holding nothing yet over @p context 's allocator - what a refused plan keeps. */
    static round_scratch_t empty_round_scratch_(traversal_context_t& context) noexcept {
        using bytes_ptr_t = typename traversal_context_t::bytes_ptr_t;
        return round_scratch_t {bytes_ptr_t {nullptr, typename bytes_ptr_t::deleter_type {context.allocator_, 0}}};
    }

    /** Zeroed round counters in a slot of @p context 's stream, or null when it cannot pin one. */
    static round_counters_ptr_t take_round_counters_(traversal_context_t& context) noexcept {
        hip_pinned_slots_t* const pinned_slots = context.allocator_.pinned_slots();
        void* const slot = pinned_slots ? pinned_slots->take() : nullptr;
        return {slot ? new (slot) round_counters_t {} : nullptr, round_counters_return_t {pinned_slots}};
    }

    /** The mean bytes of a published value, one at least, which sizes a round's hubs and
     *  neighborhoods as a dense row would. */
    std::size_t mean_value_bytes_() const noexcept {
        std::size_t const entries = size();
        return entries ? max_of<std::size_t>(divide_round_up(tape_bytes_, entries), 1) : 1;
    }

    /**
     *  @brief A round's shape over @p entries entries of @p value_bytes each on average under
     *      @p cache_bytes, zero for this device's figure, never more than it.
     *
     *  Half the budget holds a draw's hubs, never below one, and the other half a neighborhood's
     *  entries, never below a row of mates and the entry itself. A neighborhood over capacity
     *  holds more entries than one, so they number fewer than the entries over the capacity, the
     *  hubs of a gather inside them at most twice that and one each, and a cut's starts the room
     *  @c split_room counts.
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
        shape.offsets = split_room(entries, shape.hubs_per_draw);
        shape.hubs = max_of(shape.hubs_per_draw, 3 * wide + 1);
        shape.runs = max_of<std::size_t>(wide, 1);
        return shape;
    }

    /** How a round's gather and restitch launch over values of @p longest bytes at most, preparing
     *  into @p span_bytes: on the kit rung serving them, each warp's spans and its selection over
     *  a live row and a row of mates, twice over, in shared memory where a resident block's warps
     *  fit, in an arena slice otherwise. */
    round_launch_t round_launch_(std::size_t longest, std::size_t span_bytes) const noexcept {
        round_launch_t launch;
        launch.rung = kits_.rung_serving_(longest);
        launch.span_bytes = static_cast<u32_t>(span_bytes);
        launch.selection_nodes = 2 * reselection_nodes_();
        u32_t const bytes = build_scratch_bytes_(launch.span_bytes, launch.selection_nodes);
        std::size_t const block_bytes = walkers_per_block_(launch.rung) * bytes + static_shared_bytes();
        bool const fits = block_bytes <= device_.shared_memory_per_block_bytes &&
                          // HIP keeps no per-block reserve of the local data share.
                          hip_resident_blocks(vram_kernel_t::restitch_k, 0) * block_bytes <=
                              device_.shared_memory_per_multiprocessor_bytes;
        launch.scratch_bytes_per_warp = fits ? bytes : 0;
        launch.arena_shape = gpu_arena_shape(0, 0, fits ? 0 : bytes);
        return launch;
    }

    /** Bytes the arena of a round's @p launch takes: every resident warp's scratch slice, where
     *  shared memory cannot fit it, and none otherwise. */
    std::size_t round_arena_bytes_(round_launch_t const& launch) const noexcept {
        if (!launch.arena_shape.scratch_bytes) return 0;
        return std::size_t {device_.multiprocessor_count} * hip_resident_blocks(vram_kernel_t::restitch_k, 0) *
               vram_warps_per_block_k * launch.arena_shape.template slice_bytes<node_id_t>();
    }

    /** Bytes a round's device scratch and its page-locked mirrors take. */
    struct round_layout_t {
        std::size_t device_bytes = 0, host_bytes = 0;
    };

    /**
     *  @brief Lays a round of @p shape out, answering the bytes of its columns and its mirrors and
     *      placing them from @p device into @p scratch and from @p host into @p mirrors.
     *
     *  The device holds the counters, the depths' tally, the bitsets, the depths both ways and the
     *  in-link counts, then one region the islands' columns and the round's share, since only the
     *  checks and the heal read the islands'. The mirrors take the round's columns in that order.
     */
    static round_layout_t lay_out_round_(round_shape_t const& shape, byte_t* device = nullptr,
                                         round_scratch_t* scratch = nullptr, byte_t* host = nullptr,
                                         round_mirrors_t* mirrors = nullptr) noexcept {
        std::size_t const entries = shape.entries, words = divide_round_up<std::size_t>(entries, 32);
        std::size_t cursor = 0;
        auto take = [&](std::size_t bytes) noexcept {
            std::size_t const offset = cursor;
            cursor = round_up_to_multiple(cursor + bytes, default_alignment_k);
            return offset;
        };
        std::size_t const counters = take(sizeof(round_counters_t));
        std::size_t const depth_entries = take(256 * sizeof(u32_t));
        std::size_t const bitsets = take(4 * words * sizeof(u32_t));
        std::size_t const depths = take(2 * entries);
        std::size_t const links = take(2 * entries * sizeof(u32_t));
        std::size_t const region = cursor;
        std::size_t const island_entries = take(entries * sizeof(u64_t));
        std::size_t const island_columns = take(4 * entries * sizeof(u32_t));
        std::size_t const island_labels = take((entries + 1) * sizeof(u32_t));
        std::size_t const islands_end = cursor;
        cursor = region;
        auto take_round_columns = [&]() noexcept {
            std::size_t const neighborhood_entries = take(entries * sizeof(u32_t));
            std::size_t const neighborhood_offsets = take(shape.offsets * sizeof(u32_t));
            std::size_t const hub_nodes = take(shape.hubs * sizeof(u32_t));
            std::size_t const runs = take(shape.runs * sizeof(vram_neighborhood_run_t));
            std::size_t const run_entries = take((shape.runs + 1) * sizeof(u32_t));
            std::size_t const entry_neighborhoods = take(entries * sizeof(u16_t));
            return std::array<std::size_t, 6> {neighborhood_entries, neighborhood_offsets, hub_nodes, runs,
                                               run_entries,          entry_neighborhoods};
        };
        std::array<std::size_t, 6> const round_columns = take_round_columns();
        cursor = max_of(cursor, islands_end);
        round_layout_t layout;
        layout.device_bytes = cursor;
        if (scratch) {
            vram_health_t& health = scratch->health;
            scratch->words = words;
            health.counters = reinterpret_cast<round_counters_t*>(device + counters);
            health.depth_entries = reinterpret_cast<u32_t*>(device + depth_entries);
            health.reached.words = reinterpret_cast<u32_t*>(device + bitsets);
            health.reaches_root.words = health.reached.words + words;
            health.two_ways_to_root.words = health.reaches_root.words + words;
            health.changed.words = health.two_ways_to_root.words + words;
            health.depths_from_root = reinterpret_cast<u8_t*>(device + depths);
            health.depths_to_root = health.depths_from_root + entries;
            health.in_links = reinterpret_cast<u32_t*>(device + links);
            health.shallower_in_links = health.in_links + entries;
            health.island_entries = reinterpret_cast<u64_t*>(device + island_entries);
            health.island_nodes = reinterpret_cast<u32_t*>(device + island_columns);
            health.island_components = health.island_nodes + entries;
            health.island_parents = health.island_components + entries;
            health.island_roles = health.island_parents + entries;
            health.island_labels = reinterpret_cast<u32_t*>(device + island_labels);
            scratch->neighborhood_entries = reinterpret_cast<u32_t*>(device + round_columns[0]);
            scratch->neighborhood_offsets = reinterpret_cast<u32_t*>(device + round_columns[1]);
            scratch->hub_nodes = reinterpret_cast<u32_t*>(device + round_columns[2]);
            scratch->runs = reinterpret_cast<vram_neighborhood_run_t*>(device + round_columns[3]);
            scratch->run_items = reinterpret_cast<u32_t*>(device + round_columns[4]);
            scratch->entry_neighborhoods = reinterpret_cast<u16_t*>(device + round_columns[5]);
        }
        cursor = 0;
        std::array<std::size_t, 6> const mirrored = take_round_columns();
        layout.host_bytes = cursor;
        if (mirrors) {
            mirrors->neighborhood_entries = reinterpret_cast<u32_t*>(host + mirrored[0]);
            mirrors->neighborhood_offsets = reinterpret_cast<u32_t*>(host + mirrored[1]);
            mirrors->hub_nodes = reinterpret_cast<u32_t*>(host + mirrored[2]);
            mirrors->runs = reinterpret_cast<vram_neighborhood_run_t*>(host + mirrored[3]);
            mirrors->run_entries = reinterpret_cast<u32_t*>(host + mirrored[4]);
            mirrors->entry_neighborhoods = reinterpret_cast<u16_t*>(host + mirrored[5]);
        }
        return layout;
    }

    /** Carves @p plan 's columns for a check of @p shape from @p context 's allocator and takes its
     *  page-locked counters. */
    status_t carve_health_(health_plan_t& plan, traversal_context_t& context, round_shape_t const& shape) noexcept {
        if (context.device() != device_.ordinal || !context) return status_code_t::config_mismatch_k;
        std::size_t const bytes = lay_out_round_(shape).device_bytes;
        round_scratch_t& scratch = plan.columns_;
        scratch.bytes.reset(context.allocator_.allocate(bytes));
        scratch.bytes.get_deleter().count = bytes;
        if (!scratch.bytes) return status_code_t::out_of_memory_k;
        lay_out_round_(shape, scratch.bytes.get(), &scratch);
        plan.counters_ = take_round_counters_(context);
        return plan.counters_ ? status_code_t::success_k : status_code_t::out_of_memory_k;
    }

    /** Carves everything @p plan 's rounds of @p shape keep: the check's and the round's columns,
     *  the page-locked mirrors, holding at least @p pinned_floor bytes for a cut's own, and the
     *  @p arena_bytes restitch arena; zeroes the telemetry once, so every phase adds to it. */
    status_t carve_round_(refine_plan_t& plan, traversal_context_t& context, round_shape_t const& shape,
                          std::size_t arena_bytes, std::size_t pinned_floor) noexcept {
        if (status_t const carved = carve_health_(plan.health, context, shape); failed(carved)) return carved;
        plan.shape_ = shape;
        growing_buffer<scored_node_t, hip_pinned_allocator_t>& pinned = context.results.entries_;
        std::size_t const host_bytes = max_of(lay_out_round_(shape).host_bytes, pinned_floor);
        if (failed(pinned.resize_uninitialized(divide_round_up(host_bytes, sizeof(scored_node_t)))))
            return status_code_t::out_of_memory_k;
        lay_out_round_(shape, nullptr, nullptr, bytes_of(pinned.data(), pinned.size()).data(), &plan.mirrors_);
        if (arena_bytes && !context.ensure_arena(arena_bytes)) return status_code_t::out_of_memory_k;
        return succeeded(hip_vendor_t::fill(api(), context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t),
                                            context.stream()))
                   ? status_code_t::success_k
                   : status_code_t::device_k;
    }

    /** The record every launch of @p plan 's check reads, over this graph's neighbors. */
    vram_health_t health_record_(health_plan_t const& plan) const noexcept {
        vram_health_t health = plan.columns_.health;
        health.neighbors = reinterpret_cast<u32_t*>(neighbors_matrix_.base());
        health.rows_count = plan.visible;
        health.connectivity = static_cast<u32_t>(connectivity());
        return health;
    }

    /** Queues @p count words of @p value from @p words on @p stream. */
    status_t fill_words_(u32_t* words, u32_t value, std::size_t count, hip_stream_handle_t stream) const noexcept {
        return api().fill_words_async(words, value, count, stream) == hip_success_k ? status_code_t::success_k
                                                                                    : status_code_t::device_k;
    }

    /** Queues the clearing of what one root's sweep of @p plan fills, then @p health 's root marked
     *  both ways at depth zero, where the depths' tally starts and both directions count it. */
    status_t queue_health_start_(health_plan_t const& plan, vram_health_t const& health,
                                 hip_stream_handle_t stream) noexcept {
        std::size_t const visible = plan.visible, words = plan.columns_.words;
        u32_t const root = static_cast<u32_t>(plan.root_);
        u32_t const bit = 1u << (root & 31u);
        hip_api_t const& driver = api();
        bool const filled =
            succeeded(hip_vendor_t::fill(driver, health.reached.words, 0, 3 * words * sizeof(u32_t), stream)) &&
            succeeded(hip_vendor_t::fill(driver, health.depths_from_root, 0xFF, visible, stream)) &&
            succeeded(hip_vendor_t::fill(driver, health.depths_to_root, 0xFF, visible, stream)) &&
            succeeded(hip_vendor_t::fill(driver, health.in_links, 0, visible * sizeof(u32_t), stream)) &&
            succeeded(hip_vendor_t::fill(driver, health.shallower_in_links, 0, visible * sizeof(u32_t), stream)) &&
            succeeded(hip_vendor_t::fill(driver, health.depth_entries, 0, 256 * sizeof(u32_t), stream)) &&
            succeeded(hip_vendor_t::fill(driver, health.counters, 0, offsetof(round_counters_t, healed), stream)) &&
            succeeded(fill_words_(health.reached.words + root / 32, bit, 1, stream)) &&
            succeeded(fill_words_(health.reaches_root.words + root / 32, bit, 1, stream)) &&
            succeeded(hip_vendor_t::fill(driver, health.depths_from_root + root, 0, 1, stream)) &&
            succeeded(hip_vendor_t::fill(driver, health.depths_to_root + root, 0, 1, stream)) &&
            succeeded(fill_words_(health.depth_entries, 1, 1, stream)) &&
            succeeded(fill_words_(reinterpret_cast<u32_t*>(&health.counters->swept), 2, 1, stream));
        return filled ? status_code_t::success_k : status_code_t::device_k;
    }

    /**
     *  @brief Queues @p plan 's next batch on @p context 's stream, then the copy of its counters.
     *
     *  A fixpoint phase queues its passes from the next one up to twice it, each zeroing its
     *  parity's counter first, so the batches double until one ends past convergence; the island
     *  steps that run once, the order, a settle with the largest island's tally, and the count
     *  pass with the roles and entries, queue whole.
     */
    status_t queue_health_(traversal_context_t& context, health_plan_t& plan) noexcept {
        using step_t = typename health_plan_t::step_t;
        auto const [graph_kernels, kernels_status] = graph_kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        hip_stream_handle_t const stream = context.stream();
        vram_graph<node_id_t> graph {};
        vram_health_t& health = graph.health;
        health = health_record_(plan);
        round_counters_t* const counters = health.counters;
        u32_t const striding = graph_blocks_(vram_kernel_t::health_expand_k);
        u32_t const first = plan.next_pass_, bound = first ? 2 * first : 1;
        status_t queued = status_code_t::success_k;
        auto zeroed = [&](u32_t& word) noexcept {
            if (succeeded(queued) && failed(hip_vendor_t::fill(api(), &word, 0, sizeof(u32_t), stream)))
                queued = status_code_t::device_k;
        };
        auto launch = [&](vram_kernel_t kernel, u32_t blocks) noexcept {
            if (succeeded(queued)) queued = launch_graph_(*graph_kernels, kernel, blocks, graph, stream);
        };
        auto island_step = [&](vram_kernel_t kernel, vram_island_pass_t pass) noexcept {
            health.pass = static_cast<u32_t>(pass);
            launch(kernel, striding);
        };
        // Each pass of a fixpoint zeroes the counters of its parity, which the one after it reads.
        auto passes = [&](vram_kernel_t kernel, u32_t* parity_counters, u32_t* second_counters) noexcept {
            for (u32_t pass = first; pass != bound; ++pass) {
                health.depth = pass;
                zeroed(parity_counters[pass & 1u]);
                if (second_counters) zeroed(second_counters[pass & 1u]);
                launch(kernel, striding);
            }
        };
        plan.last_pass_ = bound - 1;
        switch (plan.step_) {
        case step_t::expand_k:
            if (!first) queued = queue_health_start_(plan, health, stream);
            passes(vram_kernel_t::health_expand_k, counters->reached, counters->marked);
            break;
        case step_t::recheck_k: passes(vram_kernel_t::health_recheck_k, counters->marked, nullptr); break;
        case step_t::islands_k:
            if (!first) launch(vram_kernel_t::order_islands_k, 1);
            passes(vram_kernel_t::health_islands_k, counters->marked, nullptr);
            break;
        case step_t::components_k:
            health.pass = static_cast<u32_t>(plan.island_pass_);
            if (plan.island_pass_ != vram_island_pass_t::settle_k) {
                passes(vram_kernel_t::island_components_k, counters->relabeled, nullptr);
                break;
            }
            zeroed(counters->unassigned);
            island_step(vram_kernel_t::island_components_k, vram_island_pass_t::settle_k);
            island_step(vram_kernel_t::island_components_k, vram_island_pass_t::largest_k);
            break;
        case step_t::count_links_k:
            launch(vram_kernel_t::health_count_links_k, striding);
            if (!plan.counters_->islands) break;
            for (vram_island_pass_t const pass : {vram_island_pass_t::roles_start_k, vram_island_pass_t::roles_k,
                                                  vram_island_pass_t::roles_spread_k, vram_island_pass_t::entries_k})
                island_step(vram_kernel_t::label_islands_k, pass);
            [[fallthrough]];
        case step_t::labels_k:
            health.pass = static_cast<u32_t>(vram_island_pass_t::weak_k);
            passes(vram_kernel_t::label_islands_k, counters->relabeled, nullptr);
            zeroed(counters->components);
            island_step(vram_kernel_t::label_islands_k, vram_island_pass_t::weak_count_k);
            break;
        default: break;
        }
        plan.next_pass_ = bound;
        if (succeeded(queued) &&
            failed(hip_vendor_t::copy(api(), plan.counters_.get(), counters, sizeof(round_counters_t), stream)))
            queued = status_code_t::device_k;
        return queued;
    }

    /** Queues the round's start on @p context 's stream, its counters and changed bits zeroed, then
     *  the distance-free heal of every island component @p plan named, on one thread. */
    status_t queue_heal_(traversal_context_t& context, health_plan_t const& plan) noexcept {
        auto const [graph_kernels, kernels_status] = graph_kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        hip_stream_handle_t const stream = context.stream();
        vram_graph<node_id_t> graph {};
        graph.health = health_record_(plan);
        std::size_t const round_offset = offsetof(round_counters_t, healed);
        if (failed(hip_vendor_t::fill(api(), reinterpret_cast<byte_t*>(graph.health.counters) + round_offset, 0,
                                      sizeof(round_counters_t) - round_offset, stream)) ||
            failed(
                hip_vendor_t::fill(api(), graph.health.changed.words, 0, plan.columns_.words * sizeof(u32_t), stream)))
            return status_code_t::device_k;
        if (!plan.counters_->islands) return status_code_t::success_k;
        return launch_graph_(*graph_kernels, vram_kernel_t::heal_islands_k, 1, graph, stream);
    }

    /** The record a gather or a restitch of @p plan launches with, its warps' spans as @p launch
     *  shapes them over @p arena, and the claim counter zeroed behind it. */
    vram_round<node_id_t> round_record_(refine_plan_t const& plan, round_launch_t const& launch, byte_t* arena,
                                        traversal_context_t& context) const noexcept {
        round_scratch_t const& scratch = plan.health.columns_;
        vram_round<node_id_t> round {};
        round.storage = storage();
        round.span_bytes = launch.span_bytes;
        round.scratch_bytes_per_warp = launch.scratch_bytes_per_warp;
        round.arena = arena;
        round.arena_shape = launch.arena_shape;
        round.selection_nodes = launch.selection_nodes;
        round.telemetry = context.telemetry_.get();
        round.assigned = &scratch.health.counters->claimed;
        round.health = health_record_(plan.health);
        return round;
    }

    /** Launches the kits' round @p kernel of @p plan, a gather or a restitch, over @p round on
     *  @p context 's stream, on the rung @p launch serves, its claim counter zeroed first. */
    status_t launch_round_(traversal_context_t& context, vram_kernel_t kernel, round_launch_t const& launch,
                           vram_round<node_id_t> const& round) noexcept {
        auto const [kits, kits_status] = kits_kernels_(context);
        if (failed(kits_status)) return kits_status;
        hip_stream_handle_t const stream = context.stream();
        if (failed(hip_vendor_t::fill(api(), round.assigned, 0, sizeof(u32_t), stream))) return status_code_t::device_k;
        vram_arguments<node_id_t> arguments {};
        arguments.round = round;
        return hip_launch(api(), kits_function_at_(*kits, launch.rung, kernel, 0),
                          device_.multiprocessor_count * hip_resident_blocks(kernel, 0), block_threads(),
                          walkers_per_block_(launch.rung) * launch.scratch_bytes_per_warp, stream, arguments);
    }

    /** The arena a round's @p launch carves its warps' scratch from, grown at planning; null where
     *  shared memory carries it, and on a refused growth. */
    byte_t* round_arena_(traversal_context_t& context, round_launch_t const& launch) const noexcept {
        return launch.arena_shape.scratch_bytes ? context.ensure_arena(round_arena_bytes_(launch)) : nullptr;
    }

    /** Queues one gather of @p plan on @p context 's stream: its hubs, runs and the entries before
     *  each run uploaded from the mirrors, the entries in neighborhood order too past the first
     *  gather, then the launch, then the labels copied down for the barrier to order. */
    status_t queue_gather_(traversal_context_t& context, refine_plan_t const& plan) noexcept {
        hip_stream_handle_t const stream = context.stream();
        round_scratch_t const& scratch = plan.health.columns_;
        round_mirrors_t const& mirrors = plan.mirrors_;
        std::size_t const visible = plan.health.visible;
        bool const nested = plan.neighborhoods_ != 0;
        if (failed(
                hip_vendor_t::copy(api(), scratch.hub_nodes, mirrors.hub_nodes, plan.hubs_ * sizeof(u32_t), stream)) ||
            failed(hip_vendor_t::copy(api(), scratch.runs, mirrors.runs, plan.runs_ * sizeof(vram_neighborhood_run_t),
                                      stream)) ||
            failed(hip_vendor_t::copy(api(), scratch.run_items, mirrors.run_entries, (plan.runs_ + 1) * sizeof(u32_t),
                                      stream)) ||
            (nested && failed(hip_vendor_t::copy(api(), scratch.neighborhood_entries, mirrors.neighborhood_entries,
                                                 visible * sizeof(u32_t), stream))))
            return status_code_t::device_k;
        round_launch_t const launch = round_launch_(longest_value_, span_bytes_);
        byte_t* const arena = round_arena_(context, launch);
        if (launch.arena_shape.scratch_bytes && !arena) return status_code_t::out_of_memory_k;
        vram_round<node_id_t> round = round_record_(plan, launch, arena, context);
        round.entries = nested ? scratch.neighborhood_entries : nullptr;
        round.hub_nodes = scratch.hub_nodes;
        round.runs = scratch.runs;
        round.run_entries = scratch.run_items;
        round.runs_count = static_cast<u32_t>(plan.runs_);
        round.entry_neighborhoods = scratch.entry_neighborhoods;
        if (status_t const launched = launch_round_(context, vram_kernel_t::gather_neighborhoods_k, launch, round);
            failed(launched))
            return launched;
        return succeeded(hip_vendor_t::copy(api(), mirrors.entry_neighborhoods, scratch.entry_neighborhoods,
                                            visible * sizeof(u16_t), stream))
                   ? status_code_t::success_k
                   : status_code_t::device_k;
    }

    /** Queues @p plan 's restitch on @p context 's stream: the entries in neighborhood order and
     *  the starts uploaded from the mirrors, then the launch, warps claiming entries in turn. */
    status_t queue_restitch_(traversal_context_t& context, refine_plan_t const& plan) noexcept {
        hip_stream_handle_t const stream = context.stream();
        round_scratch_t const& scratch = plan.health.columns_;
        round_mirrors_t const& mirrors = plan.mirrors_;
        if (failed(hip_vendor_t::copy(api(), scratch.neighborhood_entries, mirrors.neighborhood_entries,
                                      std::size_t {plan.health.visible} * sizeof(u32_t), stream)) ||
            failed(hip_vendor_t::copy(api(), scratch.neighborhood_offsets, mirrors.neighborhood_offsets,
                                      (plan.neighborhoods_ + 1) * sizeof(u32_t), stream)))
            return status_code_t::device_k;
        round_launch_t const launch = round_launch_(longest_value_, span_bytes_);
        byte_t* const arena = round_arena_(context, launch);
        if (launch.arena_shape.scratch_bytes && !arena) return status_code_t::out_of_memory_k;
        vram_round<node_id_t> round = round_record_(plan, launch, arena, context);
        round.entries = scratch.neighborhood_entries;
        round.neighborhood_offsets = scratch.neighborhood_offsets;
        round.neighborhoods_count = static_cast<u32_t>(plan.neighborhoods_);
        return launch_round_(context, vram_kernel_t::restitch_k, launch, round);
    }

    /**
     *  @brief Binds @p plan to a gather of every source of @p sources other than this graph: copies
     *      their nodes past the watermark behind one wait, files their keys, publishes them, and
     *      empties those sources, then opens the round over every node.
     *
     *  Every allocation and every check happens first: each source's values copy past this tape,
     *  its offsets shift by this tape's bytes on the host, its neighbor rows copy and rebase onto
     *  the IDs past the watermark on the device, and its keys move on the host. The sources' claim
     *  words close for the copies alone; this graph's stays closed for the plan's life.
     */
    template <typename owner_type_>
    status_t open_gather_(reshard_plan_t& plan, std::span<owner_type_* const> sources,
                          vram_amd_ragged_graph owner_type_::* graph, reshard_config_t const& config,
                          traversal_context_t& context) noexcept {
        auto const [graph_kernels, kernels_status] = graph_kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        std::size_t moved_nodes = 0, moved_bytes = 0, longest = longest_value_, span_bytes = span_bytes_;
        for (owner_type_* const source : sources) {
            vram_amd_ragged_graph const& given = source->*graph;
            if (&given == this || !given.size()) continue;
            if (given.device_.ordinal != device_.ordinal || given.connectivity() != connectivity() ||
                given.kernel_ != kernel_)
                return status_code_t::config_mismatch_k;
            if (status_t const kept = given.status(); failed(kept)) return kept;
            moved_nodes += given.size(), moved_bytes += given.tape_bytes_;
            longest = max_of(longest, given.longest_value_), span_bytes = max_of(span_bytes, given.span_bytes_);
        }
        if (failed(plan.targets_.reserve_exactly(1))) return status_code_t::out_of_memory_k;
        if (status_t const closed = close_(); failed(closed)) return closed;
        plan.targets_.push_back_reserved(this);
        plan.closed_targets_ = 1;
        std::size_t const base = size();
        if (moved_nodes > ceiling_nodes_ - base) return status_code_t::overflow_k;
        // Sources close one by one, and every one closed reopens once the copies end.
        std::size_t closed_sources = 0;
        auto const reopen_sources = [&]() noexcept {
            for (std::size_t source = 0; source != closed_sources; ++source)
                if (&(sources[source]->*graph) != this) (sources[source]->*graph).reopen_();
        };
        for (; closed_sources != sources.size(); ++closed_sources) {
            vram_amd_ragged_graph& given = sources[closed_sources]->*graph;
            if (&given == this) continue;
            if (status_t const closed = given.close_(); failed(closed)) {
                reopen_sources();
                return closed;
            }
        }
        std::size_t const entries = base + moved_nodes;
        hip_stream_handle_t const stream = context.stream();
        growing_buffer<u64_t> shifted_offsets {global_allocator};
        round_shape_t const shape = round_shape_(
            entries, entries ? max_of<std::size_t>(divide_round_up(tape_bytes_ + moved_bytes, entries), 1) : 1,
            config.refine.cache_bytes);
        status_t packed = carve_round_(plan.rounds_, context, shape,
                                       round_arena_bytes_(round_launch_(longest, span_bytes)), 0);
        if (succeeded(packed)) packed = grow_nodes_(entries, stream);
        if (succeeded(packed)) packed = values_tape_.commit(tape_bytes_ + moved_bytes + tape_margin_k);
        if (succeeded(packed) && failed(shifted_offsets.resize_uninitialized(moved_nodes + 1)))
            packed = status_code_t::out_of_memory_k;

        // The appended nodes publish once every source's copies drained, so a refusal shows none.
        std::size_t const row_bytes = connectivity() * sizeof(node_id_t);
        std::size_t into = base, tape_bytes = tape_bytes_;
        for (owner_type_* const source : sources) {
            vram_amd_ragged_graph& given = source->*graph;
            std::size_t const count = given.size();
            if (failed(packed) || &given == this || !count) continue;
            if (api().copy(shifted_offsets.data(), given.offsets_column_.base(), (count + 1) * sizeof(u64_t),
                           hip_copy_default_k) != hip_success_k) {
                packed = status_code_t::device_k;
                continue;
            }
            for (std::size_t node = 0; node <= count; ++node) shifted_offsets[node] += tape_bytes;
            vram_graph<node_id_t> rebased {};
            rebased.neighbors = reinterpret_cast<node_id_t*>(neighbors_matrix_.base());
            rebased.connectivity = static_cast<u32_t>(connectivity());
            rebased.first_id = static_cast<node_id_t>(into);
            rebased.nodes_count = static_cast<u32_t>(count);
            rebased.source_bound = static_cast<node_id_t>(count);
            rebased.shift = static_cast<node_id_t>(into);
            if (api().copy(offsets_column_.base() + into * sizeof(u64_t), shifted_offsets.data(),
                           (count + 1) * sizeof(u64_t), hip_copy_default_k) != hip_success_k ||
                (given.tape_bytes_ &&
                 failed(hip_vendor_t::copy(api(), values_tape_.base() + tape_bytes, given.values_tape_.base(),
                                           given.tape_bytes_, stream))) ||
                failed(hip_vendor_t::copy(api(), neighbors_matrix_.base() + into * row_bytes,
                                          given.neighbors_matrix_.base(), count * row_bytes, stream)))
                packed = status_code_t::device_k;
            else
                packed = launch_graph_(*graph_kernels, vram_kernel_t::rebase_links_k,
                                       graph_blocks_(vram_kernel_t::rebase_links_k), rebased, stream);
            into += count, tape_bytes += given.tape_bytes_;
        }
        if (succeeded(packed)) packed = context.wait();
        if (failed(packed)) {
            // The nodes past the watermark hold whatever was copied, so the next grow refills them.
            filled_nodes_ = min_of(filled_nodes_, base);
            reopen_sources();
            return packed;
        }
        // The closed claim words make this call every map's one writer.
        into = base;
        for (owner_type_* const source : sources) {
            vram_amd_ragged_graph& given = source->*graph;
            if (&given == this) continue;
            for (std::size_t node = 0, count = given.size(); node != count; ++node)
                keyed_.assign(static_cast<node_id_t>(into++), given.keyed_[static_cast<node_id_t>(node)]);
            given.forget_();
        }
        reopen_sources();
        tape_bytes_ = tape_bytes, longest_value_ = longest, span_bytes_ = span_bytes;
        publish_keys_({static_cast<node_id_t>(base), static_cast<node_id_t>(entries)});
        fu::standard_atomic_ref<std::size_t>(last_visible_id_).store(entries, std::memory_order_release);
        plan.rounds_.bind_(*this);
        return status_code_t::success_k;
    }

    /** Binds @p plan to a cut of this graph into @p targets: closes this graph and every target,
     *  reserves and grows every target for a full part and maps its tape for the most its part can
     *  take, carves the rounds over a full part and the partition, then queues the partition. */
    template <typename owner_type_>
    status_t open_cut_(reshard_plan_t& plan, std::span<owner_type_* const> targets,
                       vram_amd_ragged_graph owner_type_::* graph, reshard_config_t const& config,
                       traversal_context_t& context) noexcept {
        std::size_t const parts = targets.size();
        split_t& split = plan.split_;
        if (failed(plan.targets_.reserve_exactly(parts)) || failed(split.parts.resize(parts)))
            return status_code_t::out_of_memory_k;
        for (owner_type_* const listed : targets) {
            vram_amd_ragged_graph const& part = listed->*graph;
            if (&part == this || part.device_.ordinal != device_.ordinal || part.kernel_ != kernel_ ||
                (part.connectivity() && part.connectivity() != connectivity()))
                return status_code_t::config_mismatch_k;
        }
        if (status_t const closed = close_(); failed(closed)) return closed;
        split.source = this;
        for (owner_type_* const listed : targets) plan.targets_.push_back_reserved(&(listed->*graph));
        split.entries = size();
        split.part_capacity = divide_round_up<std::size_t>(split.entries, parts);
        if (parts * split.part_capacity >= (std::numeric_limits<u32_t>::max)()) return status_code_t::overflow_k;
        for (vram_amd_ragged_graph* const part : plan.targets_) {
            if (status_t const closed = part->close_(); failed(closed)) return closed;
            ++plan.closed_targets_;
            if (part->size()) return status_code_t::config_mismatch_k;
            if (status_t const kept = part->status(); failed(kept)) return kept;
        }
        // Every target holds a full part's nodes, sentinel-filled, and the most value bytes they
        // can take, before the partition queues.
        hip_stream_handle_t const stream = context.stream();
        std::size_t const part_bytes = min_of(tape_bytes_, split.part_capacity * longest_value_);
        for (vram_amd_ragged_graph* const part : plan.targets_) {
            if (status_t const reserved =
                    part->reserve({split.part_capacity, static_cast<connectivity_t>(connectivity()), 0}).status;
                failed(reserved))
                return reserved;
            if (status_t const grown = part->grow_nodes_(split.part_capacity, stream); failed(grown)) return grown;
            if (status_t const mapped = part->values_tape_.commit(part_bytes + tape_margin_k); failed(mapped))
                return mapped;
        }
        // The rounds' mirrors take over the cut's pinned bytes once the parts publish.
        round_shape_t const shape = round_shape_(split.part_capacity, mean_value_bytes_(), config.refine.cache_bytes);
        if (status_t const carved = carve_round_(plan.rounds_, context, shape,
                                                 round_arena_bytes_(round_launch_(longest_value_, span_bytes_)),
                                                 split_layout_(split.entries, parts).host_bytes);
            failed(carved))
            return carved;
        // An empty source leaves every target empty, so no round binds.
        if (!split.entries) {
            plan.refined_targets_ = parts;
            return status_code_t::success_k;
        }
        auto const [graph_kernels, kernels_status] = graph_kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        if (status_t const carved = carve_split_(split, to_span(plan.targets_), config.seed, context); failed(carved))
            return carved;
        if (status_t const queued = queue_partition_(*graph_kernels, split, parts, stream); failed(queued))
            return queued;
        plan.moves_ = phase_t::partition_k;
        return status_code_t::success_k;
    }

    /** Bytes a cut of @p nodes nodes into @p parts parts carves on the device and pins on the
     *  host, and where the pinned bytes place the offsets columns, the source's value starts, the
     *  bases and the roots, as @c split_t lays them out. */
    struct split_layout_t {
        std::size_t device_bytes = 0, device_bases = 0;
        std::size_t host_bytes = 0, host_offsets = 0, host_starts = 0, host_bases = 0, host_roots = 0,
                    host_assigned = 0;
    };
    static split_layout_t split_layout_(std::size_t nodes, std::size_t parts) noexcept {
        std::size_t const words = divide_round_up<std::size_t>(nodes, 32);
        split_layout_t layout;
        layout.device_bases = aligned_((2 * nodes + 2 * words + 3 * parts + 2) * sizeof(u32_t));
        layout.device_bytes = layout.device_bases + 3 * parts * sizeof(byte_t*);
        layout.host_offsets = aligned_(nodes * sizeof(u32_t));
        layout.host_starts = layout.host_offsets + aligned_((nodes + parts) * sizeof(u64_t));
        layout.host_bases = layout.host_starts + aligned_((nodes + 1) * sizeof(u64_t));
        layout.host_roots = layout.host_bases + 3 * parts * sizeof(byte_t*);
        layout.host_assigned = layout.host_roots + parts * sizeof(u32_t);
        layout.host_bytes = layout.host_assigned + sizeof(u32_t);
        return layout;
    }

    /** Where a cut's device scratch keeps each array: each node's part, which the scatter reads
     *  back as its slot, the round's proposals, @b [nodes] each, the two frontiers, one bit per
     *  node each, then per part its size and its round count, the nodes the last two rounds
     *  placed, by round parity, the parts' roots, and the targets' bases. */
    struct split_scratch_t {
        u32_t* node_parts = nullptr;
        u32_t* proposals = nullptr;
        vram_bitset_t frontier {}, next_frontier {};
        u32_t* part_sizes = nullptr;
        u32_t* round_counts = nullptr;
        u32_t* placed = nullptr;
        u32_t* roots = nullptr;
        byte_t** part_bases = nullptr;
    };

    /** The arrays of @p split 's device scratch over @p parts parts, read off its bytes. */
    static split_scratch_t split_scratch_(split_t const& split, std::size_t parts) noexcept {
        std::size_t const entries = split.entries;
        std::size_t const words = divide_round_up<std::size_t>(entries, 32);
        split_scratch_t carved;
        carved.node_parts = reinterpret_cast<u32_t*>(split.device_bytes.get());
        carved.proposals = carved.node_parts + entries;
        carved.frontier.words = carved.proposals + entries;
        carved.next_frontier.words = carved.frontier.words + words;
        carved.part_sizes = carved.next_frontier.words + words;
        carved.round_counts = carved.part_sizes + parts;
        carved.placed = carved.round_counts + parts;
        carved.roots = carved.placed + 2;
        carved.part_bases = reinterpret_cast<byte_t**>(split.device_bytes.get() +
                                                       split_layout_(entries, parts).device_bases);
        return carved;
    }

    /** Carves @p split 's device scratch from @p context 's allocator, borrows its pinned results
     *  as the host bytes, copies this graph's value starts into it, and queues the upload of the
     *  @p targets ' bases and the roots @p seed picks. */
    status_t carve_split_(split_t& split, std::span<vram_amd_ragged_graph* const> targets, u64_t seed,
                          traversal_context_t& context) noexcept {
        std::size_t const entries = split.entries, parts = targets.size();
        split_layout_t const layout = split_layout_(entries, parts);
        split.device_bytes.reset(context.allocator_.allocate(layout.device_bytes));
        split.device_bytes.get_deleter().count = layout.device_bytes;
        if (!split.device_bytes) return status_code_t::out_of_memory_k;
        growing_buffer<scored_node_t, hip_pinned_allocator_t>& pinned = context.results.entries_;
        if (failed(pinned.resize_uninitialized(divide_round_up(layout.host_bytes, sizeof(scored_node_t)))))
            return status_code_t::out_of_memory_k;
        split.host_bytes = bytes_of(pinned.data(), pinned.size()).data();
        split_scratch_t const carved = split_scratch_(split, parts);
        byte_t** const bases = reinterpret_cast<byte_t**>(split.host_bytes + layout.host_bases);
        u32_t* const roots = reinterpret_cast<u32_t*>(split.host_bytes + layout.host_roots);
        for (std::size_t part = 0; part != parts; ++part) {
            vram_amd_ragged_graph& target = *targets[part];
            bases[part] = target.neighbors_matrix_.base();
            bases[parts + part] = target.values_tape_.base();
            bases[2 * parts + part] = target.offsets_column_.base();
            roots[part] = static_cast<u32_t>(seeded_bijection(part, entries, seed));
        }
        hip_stream_handle_t const stream = context.stream();
        return succeeded(hip_vendor_t::copy(api(), split.host_bytes + layout.host_starts, offsets_column_.base(),
                                            (entries + 1) * sizeof(u64_t), stream)) &&
                       succeeded(
                           hip_vendor_t::copy(api(), carved.part_bases, bases, 3 * parts * sizeof(byte_t*), stream)) &&
                       succeeded(hip_vendor_t::copy(api(), carved.roots, roots, parts * sizeof(u32_t), stream))
                   ? status_code_t::success_k
                   : status_code_t::device_k;
    }

    /** The record every launch of @p split into @p parts parts reads: this graph's neighbors and
     *  nodes, and the split's scratch. */
    vram_graph<node_id_t> split_graph_(split_t const& split, std::size_t parts) const noexcept {
        split_scratch_t const carved = split_scratch_(split, parts);
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
        graph.part_neighbors = reinterpret_cast<u32_t* const*>(carved.part_bases);
        graph.part_values = carved.part_bases + parts;
        graph.part_offsets = reinterpret_cast<u64_t const* const*>(carved.part_bases + 2 * parts);
        return graph;
    }

    /** Queues the start of @p split 's partition into @p parts parts through @p graph_kernels: the
     *  node parts and proposals unknown, the bitsets, the parts' counters and the rounds' counts
     *  empty, then the roots filed in. */
    status_t queue_partition_(kernel_table_t const& graph_kernels, split_t& split, std::size_t parts,
                              hip_stream_handle_t stream) noexcept {
        std::size_t const entries = split.entries, words = divide_round_up<std::size_t>(entries, 32);
        split_scratch_t const carved = split_scratch_(split, parts);
        // Each run lies back to back.
        if (failed(hip_vendor_t::fill(api(), carved.node_parts, 0xFF, 2 * entries * sizeof(u32_t), stream)) ||
            failed(hip_vendor_t::fill(api(), carved.frontier.words, 0, (2 * words + 2 * parts + 2) * sizeof(u32_t),
                                      stream)))
            return status_code_t::device_k;
        vram_graph<node_id_t> graph = split_graph_(split, parts);
        split.next_round = 1;
        return launch_graph_(graph_kernels, vram_kernel_t::partition_roots_k,
                             divide_round_up<u32_t>(graph.seed_count, block_threads()), graph, stream);
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
    status_t queue_rounds_(kernel_table_t const& graph_kernels, split_t& split, std::size_t parts,
                           hip_stream_handle_t stream) noexcept {
        std::size_t const words = divide_round_up<std::size_t>(split.entries, 32);
        split_scratch_t const carved = split_scratch_(split, parts);
        vram_graph<node_id_t> graph = split_graph_(split, parts);
        u32_t const first_round = split.next_round;
        for (u32_t round = first_round; round != 2 * first_round; ++round) {
            bool const odd = round & 1u;
            graph.frontier = odd ? carved.frontier : carved.next_frontier;
            graph.next_frontier = odd ? carved.next_frontier : carved.frontier;
            graph.counter = carved.placed + odd;
            graph.previous_counter = round == 1 ? nullptr : carved.placed + !odd;
            if (failed(hip_vendor_t::fill(api(), graph.next_frontier.words, 0, words * sizeof(u32_t), stream)) ||
                failed(hip_vendor_t::fill(api(), graph.counter, 0, sizeof(u32_t), stream)))
                return status_code_t::device_k;
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
        return succeeded(hip_vendor_t::copy(api(), split.host_bytes + split_layout_(split.entries, parts).host_assigned,
                                            carved.placed + ((2 * first_round - 1) & 1u), sizeof(u32_t), stream))
                   ? status_code_t::success_k
                   : status_code_t::device_k;
    }

    /** The last growth round of @p split into @p parts parts placed something, so more rounds
     *  follow. */
    static bool placing_(split_t const& split, std::size_t parts) noexcept {
        std::size_t const host_assigned = split_layout_(split.entries, parts).host_assigned;
        return *reinterpret_cast<u32_t const*>(split.host_bytes + host_assigned) != 0;
    }

    /** Queues the end of @p split 's partition once no round places anything: the nodes no round
     *  reached, then the copy of the node parts. */
    status_t queue_partition_end_(kernel_table_t const& graph_kernels, split_t& split, std::size_t parts,
                                  hip_stream_handle_t stream) noexcept {
        vram_graph<node_id_t> graph = split_graph_(split, parts);
        if (status_t const launched = launch_graph_(graph_kernels, vram_kernel_t::partition_islands_k, 1, graph,
                                                    stream);
            failed(launched))
            return launched;
        return succeeded(
                   hip_vendor_t::copy(api(), split.host_bytes, graph.node_parts, split.entries * sizeof(u32_t), stream))
                   ? status_code_t::success_k
                   : status_code_t::device_k;
    }

    /** Queues @p split 's scatter into @p targets: the assigned slots and the targets' offsets
     *  columns up, then every node's neighbors and value into its target. */
    status_t queue_scatter_(kernel_table_t const& graph_kernels, split_t const& split,
                            std::span<vram_amd_ragged_graph* const> targets, hip_stream_handle_t stream) noexcept {
        byte_t const* const host_bytes = split.host_bytes;
        std::size_t const parts = targets.size();
        u64_t const* const offsets = reinterpret_cast<u64_t const*>(host_bytes +
                                                                    split_layout_(split.entries, parts).host_offsets);
        if (failed(hip_vendor_t::copy(api(), split_scratch_(split, parts).node_parts, host_bytes,
                                      split.entries * sizeof(u32_t), stream)))
            return status_code_t::device_k;
        for (std::size_t index = 0; index != parts; ++index) {
            split_part_t const& part = split.parts[index];
            vram_amd_ragged_graph const& target = *targets[index];
            if (part.entries &&
                failed(hip_vendor_t::copy(api(), target.offsets_column_.base(), offsets + part.offsets_first,
                                          (part.entries + 1) * sizeof(u64_t), stream)))
                return status_code_t::device_k;
        }
        vram_graph<node_id_t> graph = split_graph_(split, parts);
        graph.values = values_tape_.base();
        graph.value_offsets = reinterpret_cast<u64_t const*>(offsets_column_.base());
        for (vram_kernel_t const pass : {vram_kernel_t::split_links_k, vram_kernel_t::split_tape_k})
            if (status_t const launched = launch_graph_(graph_kernels, pass, graph_blocks_(pass), graph, stream);
                failed(launched))
                return launched;
        return status_code_t::success_k;
    }

    /** Hardware identity: the ordinal comes from the storage allocator at construction, the
     *  remaining specs fill at the first @c reserve. */
    hip_device_specs_t device_;

    /** The metric the kits score under, fixed at construction. */
    nk_kernel_kind_t kernel_;

    /** Neighbors per node, frozen by the first @c reserve; zero until then. */
    std::size_t connectivity_ = 0;

    /** The values tape, the offsets column and the neighbors matrix, each allocator its own
     *  storage, so capacities are derived, never cached. */
    hip_virtual_memory_allocator<byte_t> values_tape_, offsets_column_, neighbors_matrix_;

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
    hip_event_t grown_;

    /** Nodes strictly below it are published and visible, turned once per batch after its drain. */
    std::size_t last_visible_id_ = 0;

    /** The first post-claim device failure. */
    status_t status_ = status_code_t::success_k;

    /** The modality's kits: their hooks, their code object's kernels and their device
     *  parameters. */
    kits_type_ kits_;

    /** Every node's key, assigned by its claim's holder, and the key map that holder files them in
     *  just before the watermark moves. */
    keyed_t keyed_;
};

} // namespace unum::usearch::basic
