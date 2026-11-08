/**
 *  @file include/usearch/dense/vram_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The shard core the NVIDIA and AMD dense indexes share: the kernels' argument records, and
 *      the host machinery that fills them, claims rows, publishes them and moves checkpoints.
 *
 *  @c vram_nvidia_index and @c vram_amd_index each own one, compiled by any C++ compiler. Each
 *  names its runtime through a vendor policy, the types it declares with and the copies, fills and
 *  waits it queues, and its kernels through a member the core holds: loading them, the launches,
 *  the walks it stages, and the passes of a refinement.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <bit>         // `std::bit_width` - a refine's rounds, `std::has_single_bit` - the connectivity
#include <bitset>      // `std::bitset` - the devices a sharded index shares the storage with
#include <functional>  // `std::hash` - the default hash of a key
#include <limits>      // `std::numeric_limits` - the unit-RMS import's epsilon
#include <memory>      // `std::unique_ptr` - an audit's pinned counters
#include <span>        // `std::span` - the parts' bases, the counts of a graph product
#include <type_traits> // `std::is_same_v`
#include <utility>     // `std::exchange` - a moved-from plan forgets its index, `std::forward`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the published words
#include <forkunion/types.hpp>   // `fu::wait_capped_t` - the watermark's turn

#include <numkong/cast/serial.h> // `nk_cast_plan_serial_` - the device casts' plans
#include <numkong/each/serial.h> // `nk_each_rmsnorm_arguments_t` - the unit-RMS import's record
#include <numkong/tensor.hpp>    // `nk::matrix_span` - the batches' views
#include <numkong/types.hpp>     // `nk::type_for` - the stored scalar of a dtype
#include <numkong/vector.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_device.hpp"  // `forkunion_allocator_t` - the host key column's placement
#include "usearch/utilities/ram_tables.hpp"  // `keyed_nodes` - every node's key and every key's nodes
#include "usearch/utilities/vram_device.hpp" // `vram_kernel_t` - the kernels the core launches and sizes grids for
#include "usearch/utilities/vram_walks.hpp"  // `gpu_arena_shape_t`, `gpu_candidate_slots`, `vram_bitset_t`
#include "usearch/dense/types.hpp"

namespace unum::usearch::dense {

#pragma region Kernel Records

/** Bytes of a value lookup an attend carries to its kernels. */
inline constexpr std::size_t vram_value_lookup_capacity_k = 64;

/** Bytes of one walking warp's attend scratch over @p heads_count heads of @p columns columns and
 *  @p lanes lanes: every head's batch dots, its running maximum, partition and latest rescale, then
 *  its accumulator row, in whole fragments so the next warp's query span stays aligned. */
constexpr std::size_t vram_attend_scratch_bytes(std::size_t heads_count, std::size_t columns, u32_t lanes) noexcept {
    return round_up_to_multiple<std::size_t>(heads_count * (lanes + 3 + columns) * sizeof(f32_t), 16);
}

/** The flat storage a kernel reads, mirroring the shared flat view field by field. */
struct vram_flat_storage_t {
    byte_t* vectors = nullptr;
    byte_t* norms = nullptr;
    u32_t* neighbors = nullptr;
    u32_t connectivity = 0;
    u32_t row_fragments = 0;
    u32_t lanes_per_row = 0;
    u32_t steps_per_row = 0;
    u32_t row_fragments_magic = 0;
    nk_kernel_kind_t kernel = nk_kernel_dot_k;
};

/** One insertion launch, mirroring the shared build view. */
struct vram_build_t {
    vram_flat_storage_t storage;
    byte_t const* staging = nullptr;
    u32_t end_id = 0;
    u32_t scratch_bytes_per_warp = 0;
    u32_t selection_nodes = 0;
    node_id_t first_id = 0;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    u32_t expansion = 0;
    u32_t epoch_base = 0;
    u32_t* published_rows = nullptr;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_nodes = nullptr;
    u32_t* link_heads = nullptr;
    u32_t* link_next = nullptr;
    u32_t* touched_ids = nullptr;
    u32_t* touched_count = nullptr;
};

/** One walking launch, mirroring the shared search view. */
struct vram_search_t {
    vram_flat_storage_t storage;
    byte_t const* queries = nullptr;
    u32_t queries_count = 0;
    u32_t sampling_bound = 0;
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
};

/** One refinement launch, mirroring the shared refine view. */
struct vram_refine_t {
    vram_flat_storage_t storage;
    u32_t end_id = 0;
    u32_t scratch_bytes_per_warp = 0;
    u32_t selection_nodes = 0;
    node_id_t first_id = 0;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    u32_t expansion = 0;
    u32_t epoch_base = 0;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_nodes = nullptr;
    u32_t const* node_ids = nullptr;
    u32_t sampling_bound = 0;
    u32_t const* part_bases = nullptr;
    u32_t parts = 0;
    u32_t round = 0;
    u32_t reselection_nodes = 0;
    vram_bitset_t changed;
    u32_t const* incoming_offsets = nullptr;
    u32_t const* back_links = nullptr;
    u32_t tape_size = 0;
    node_id_t walked_first = 0;
    node_id_t walked_bound = 0;
};

/** One shard of a fold bin as the bin's storage reads it on every device of the bin: its vectors
 *  and norms, its fold rows, and the first row and the rows it takes in the bin's dense ID
 *  space, which the target shard writes it to. */
struct vram_bin_shard_t {
    byte_t* vectors = nullptr;
    byte_t* norms = nullptr;
    u32_t* neighbors = nullptr;
    u32_t first = 0;
    u32_t rows = 0;
};
static_assert(sizeof(vram_bin_shard_t) == 32, "Every vector a fold scores reads its shard's entry");

/** One shard of a fold bin as its fold reads it: its live neighbors, the picks its rows froze for
 *  the merge, the chain links beside them, @b [rows,connectivity] each, and the heads of the
 *  chains its picks thread onto every other shard's rows, one run per other shard in bin order. */
struct vram_fold_shard_t {
    u32_t const* live = nullptr;
    u32_t* picks = nullptr;
    u32_t* links = nullptr;
    u32_t* heads = nullptr;
};
static_assert(sizeof(vram_fold_shard_t) == 32, "Every row a fold re-selects or merges reads its shard's entry");

/** Which phase of a fold a launch runs: the re-selection of each row, or the merge of the picks
 *  the bin's other shards made of it. */
enum class vram_fold_phase_t : u32_t { reselect_k, merge_k };

/** One fold launch over the rows of one shard of a bin, mirroring the shared fold view: the
 *  geometry of the shards' storage, the bin's tables on the launching device, the launching
 *  shard's place, the bits a bin ID keeps its row in, and the cross results each row holds from
 *  every other shard. */
struct vram_fold_t {
    vram_flat_storage_t storage;
    vram_bin_shard_t const* bin_shards = nullptr;
    vram_fold_shard_t const* fold_shards = nullptr;
    u32_t shards_count = 0;
    u32_t shard = 0;
    u32_t row_bits = 0;
    u32_t cross_count = 0;
    u32_t scratch_bytes_per_warp = 0;
    u32_t selection_nodes = 0;
    vram_fold_phase_t phase = vram_fold_phase_t::reselect_k;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_nodes = nullptr;
};

/** Where a search writes its outputs, mirroring the shared search sink. */
struct vram_search_sink_t {
    node_id_t* result_ids = nullptr;
    distance_t* result_distances = nullptr;
    u32_t count = 0;
    nk_kernel_kind_t kernel = nk_kernel_dot_k;
};

/** Where a graph product writes its outputs, mirroring the shared multiply sink; the products are
 *  the kit's dots. */
struct vram_multiply_sink_t {
    node_id_t* ids = nullptr;
    byte_t* products = nullptr;
    u32_t* counts = nullptr;
    u32_t wanted = 0;
};

/** The heads a walk rescores by: an indexer's with their weights, or an attend's with none; null
 *  heads score by the query alone. */
struct vram_rescoring_t {
    byte_t const* heads = nullptr;
    f32_t const* weights = nullptr;
    u32_t heads_count = 0;
    u32_t row_fragments = 0;
};

/** What the graph kernels read: the norms, the rebase and the audits. */
struct vram_graph_t {
    vram_flat_storage_t storage;
    u32_t* neighbors = nullptr;
    u32_t rows_count = 0;
    u32_t connectivity = 0;
    u32_t first_id = 0;
    u32_t source_first = 0;
    u32_t source_bound = 0;
    u32_t shift = 0;
    vram_bitset_t reached, frontier, next_frontier, reaches_back, back_settled;
    u32_t* counter = nullptr;

    /** The count of the sweep before; zero sends a sweep queued past convergence home at entry. */
    u32_t const* previous_counter = nullptr;

    /** The levels that expanded anything, each counted by its first expansion; a back sweep's
     *  last level that reached a row, stamped as @c sweep_number. */
    u32_t* levels = nullptr;
    u32_t sweep_number = 0;
    u32_t* island_ids = nullptr;
    u32_t islands_count = 0;
    u32_t seed_count = 0;
    u32_t* in_links = nullptr;
    u32_t* reconnected = nullptr;

    /** A split's part of every row, read back as its slot once assigned_nodes, the growth round's
     *  proposals, the parts' sizes and round counts, their roots, and the rows linking across. */
    u32_t* node_parts = nullptr;
    u32_t* proposals = nullptr;
    u32_t* part_sizes = nullptr;
    u32_t* round_counts = nullptr;
    u32_t const* roots = nullptr;
    vram_bitset_t picked;
    u32_t parts_count = 0;
    u32_t part_capacity = 0;

    /** The receivers' neighbors, vectors, norms and keys, @c parts_count bases each, the donor's
     *  keys, and the words of a key, of a vector row and of a norms entry, zero where none are
     *  kept at all. */
    u32_t* const* part_columns = nullptr;
    u32_t const* keys = nullptr;
    u32_t vector_words = 0;
    u32_t norm_words = 0;
    u32_t key_words = 0;
};

/** Where an attend writes its outputs, and the value lookup its walks resolve rows through, carried
 *  as the lookup's own bytes; the sharded gather reads the outputs as @p shards blocks, @p rows
 *  rows in each. */
struct vram_attend_t {
    f32_t* outputs = nullptr;
    softmax_state_t* states = nullptr;
    u32_t* missed = nullptr;
    f32_t logit_scale = 0;
    u32_t heads_count = 0;
    u32_t columns = 0;
    u32_t rows = 0;
    u32_t shards = 0;
    alignas(16) byte_t value_lookup[vram_value_lookup_capacity_k] = {};
};

/** A ranking of keys on the device, mirroring the shared ranking view. */
struct vram_ranking_t {
    byte_t* keys = nullptr;
    byte_t* scores = nullptr;
    std::size_t* counts = nullptr;
    std::size_t keys_stride = 0;
    std::size_t scores_stride = 0;
    u32_t width = 0;
    u32_t key_bytes = 0;
    u32_t score_bytes = 0;
};

/** One gather of stored rows into a matrix of the caller's scalar: each of the @c count entries of
 *  @c rows copies its node's row, @c source_stride bytes apart from @c source, into its output row,
 *  @c output_stride bytes apart from @c outputs, through @c cast, the plan of one row's cast. */
struct vram_gather_t {
    byte_t const* source = nullptr;
    byte_t* outputs = nullptr;
    vram_gathered_row_t const* rows = nullptr;
    std::size_t source_stride = 0;
    std::size_t output_stride = 0;
    u32_t count = 0;
    nk_cast_plan_t cast {};
};

/** A batch's @c queries_count counts summed into the one device word at @c total, which the host
 *  copies back alone. */
struct vram_sum_counts_t {
    std::size_t const* counts = nullptr;
    std::size_t* total = nullptr;
    u32_t queries_count = 0;
};

/** Keys and scores packed back to back on the device, each query's run from its offset, one more
 *  offset than queries: what a call keeping no fixed width per query publishes. */
struct vram_products_tape_t {
    std::size_t* offsets = nullptr;
    byte_t* keys = nullptr;
    byte_t* scores = nullptr;
};

/** A batch's graph products on the device: one shard's walked @c ids and @c products,
 *  @c ids_stride values apart, the first @c scored of each row resolving into @c ranking through
 *  the shard's device @c keys; and @c shards_count rankings, @c block_bytes apart from @c ranking,
 *  packing into @c tape, each query keeping at most the ranking's width, its pairs scored across
 *  the shards summed into @c totals. */
struct vram_multiply_t {
    node_id_t const* ids = nullptr;
    byte_t const* products = nullptr;
    u32_t const* scored = nullptr;
    byte_t const* keys = nullptr;
    vram_ranking_t ranking;
    vram_products_tape_t tape;
    std::size_t* totals = nullptr;
    std::size_t block_bytes = 0;
    u32_t ids_stride = 0;
    u32_t queries_count = 0;
    u32_t shards_count = 0;
};

/** One MaxSim window's exact scores on the device, @c queries_count queries over their token rows
 *  @c query_offsets apart: the distinct keys every shard ranked for a query, @c nominees; each
 *  shard's best dot of every nominee per token row, rescored over the window's staged @c queries
 *  and the shard's @c entries of each nominee, placed by the union tape's @c nominee_offsets, into
 *  @c partials, @c rows_stride values per nominee; their fold into @c scores, the largest across
 *  @c shards_count per row summed over the rows; and the @c wanted best of every query appended to
 *  the @c tape, its pairs scored into @c scored. */
struct vram_maxsim_t {
    vram_flat_storage_t storage;
    vram_ranking_t gathered;
    vram_ranking_t nominees;
    vram_products_tape_t nominated;
    vram_products_tape_t tape;
    byte_t const* queries = nullptr;
    u32_t const* query_offsets = nullptr;
    std::size_t const* nominee_offsets = nullptr;
    vram_gathered_row_t const* entries = nullptr;
    byte_t* partials = nullptr;
    f64_t* totals = nullptr;
    std::size_t* scored = nullptr;
    std::size_t block_bytes = 0;
    std::size_t wanted = 0;
    u32_t queries_count = 0;
    u32_t nominees_count = 0;
    u32_t entries_count = 0;
    u32_t rows_stride = 0;
    u32_t shards_count = 0;
    u32_t key_bytes = 0;
};

/** A key-level value lookup seen through one shard's device key column, so the shard's walks
 *  resolve node IDs: the kernels read a node's row through the key at @c keys[node_id]. */
template <typename key_type_, typename value_lookup_type_>
struct vram_keyed_lookup {

    /** The shard's keys on its device, by node ID. */
    key_type_ const* keys;

    /** Resolves a key to its value row. */
    value_lookup_type_ value_lookup;
};

/** Spells a keyed lookup into a dense kernel's name, @c S1_ being @c unum::usearch::dense. */
template <typename key_type_, typename value_lookup_type_>
void append_itanium_argument(vram_kernel_symbol_t& symbol,
                             itanium_type<vram_keyed_lookup<key_type_, value_lookup_type_>>) noexcept {
    symbol.append("NS1_17vram_keyed_lookupI");
    append_itanium_argument(symbol, itanium_type<key_type_> {});
    append_itanium_argument(symbol, itanium_type<value_lookup_type_> {});
    symbol.append("EE");
}

/** The one argument every dense kernel takes, each reading the records its kernel names: the casts
 *  read NumKong's own plans, and @c matches holds a sharded search's rankings - a shard's
 *  @c walked IDs and distances per query resolve through its device @c keys into @c ranking, and
 *  the first shard merges @c shards rankings, gathered @c block_bytes apart from @c gathered, into
 *  it. */
struct vram_arguments_t {
    vram_build_t build;
    vram_search_t search;
    vram_refine_t refine;
    vram_fold_t fold;
    vram_search_sink_t search_sink;
    vram_multiply_sink_t multiply_sink;
    vram_rescoring_t rescoring;
    vram_graph_t graph;
    vram_sweep_t sweep;
    vram_attend_t attend;
    struct matches_t {
        node_id_t const* ids = nullptr;
        distance_t const* distances = nullptr;
        byte_t const* keys = nullptr;
        vram_ranking_t gathered;
        vram_ranking_t ranking;
        std::size_t block_bytes = 0;
        u32_t walked = 0;
        u32_t queries_count = 0;
        u32_t shards = 0;
    };
    matches_t matches;
    nk_cast_arguments_t cast;
    nk_each_rmsnorm_arguments_t rmscast;
    vram_gather_t gather;
    vram_sum_counts_t counts;
    vram_multiply_t products;
    vram_maxsim_t maxsim;
};

#pragma endregion Kernel Records

#pragma region Shard Types

/** Which frontier a warp's arena slice serves: the register ladder needs only the exact visited
 *  set, the wide arena frontier adds its spill tail. */
enum class vram_frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

/** Optional tuning knobs of a shard's @c search; zeros mean "derive from connectivity". */
struct vram_search_config_t {

    /** How many candidates a walk keeps, walked as given when it covers @c count and widened to
     *  @c count otherwise; a single-query walk takes twice the connectivity for zero. */
    u32_t expansion_factor = 0;

    /** Matches reported per query; a walk past the widest register rung ranks every candidate its
     *  arena frontier kept. */
    u32_t count = 10;

    /** The caller's mutation contract; every level shares one code path, since the walk is
     *  insert-safe by protocol, so the field only documents intent. */
    synchronization_t synchronization = synchronization_t::concurrent_k;

    /** How entries are reached; @c scan_k scores every visible row exactly on the kits' tile steps,
     *  keeping each query's @c count nearest, and only the NVIDIA @c search implements it. */
    traversal_t traversal = traversal_t::walk_k;
};

/** Telemetry of one shard's @c search batch. */
struct [[nodiscard]] vram_search_result_t {
    std::size_t count = 0;
    std::size_t visited_members = 0;
    std::size_t computed_distances = 0;
    std::size_t restarts = 0;
    std::size_t tail_compactions = 0;
    std::size_t tail_dropped = 0;
    std::size_t visited_saturations = 0;
};

/** The grouped-query heads an attend stages beside the rows its walks navigate by: every head of
 *  every token, @c head_count consecutive rows per token. */
struct vram_attend_heads_t {
    rows_view_t rows;
    std::size_t head_count;
};

/** The counters an audit copies into page-locked memory, in the order of the device's: the levels
 *  that expanded, the last sweep's count both ways, which only a sweep short of convergence leaves
 *  nonzero, the levels back, then the unreached, island, reconnected and claim counters, and the
 *  rows and links a sweep's tile kept. */
struct vram_audit_counters_t {
    u32_t levels;
    u32_t last_expanded;
    u32_t back_levels;
    u32_t unreached;
    u32_t islands;
    u32_t reconnected;
    u32_t claims[2];
    u32_t swept[2];
};

/** Device scratch one refine plan carves once from a @p traversal_context_type_ and its phases
 *  reuse, or the audit share one @c health carves: the reached set and two frontiers, the rows
 *  reaching back and those that did before the current level, the levels counter beside the last
 *  two sweeps' expanded counters, by the sweep number's parity, the levels back, the unreached,
 *  island, reconnected, claim and tile counters an audit copies together, then for a refine the
 *  islands' IDs, the parts' bases, the reached in-links and the changed bits. Sweeps run until one
 *  marks nothing either way: every row enters one frontier and reaches back once at most, so the
 *  rows bound them. */
template <typename traversal_context_type_>
struct vram_refine_scratch {
    typename traversal_context_type_::bytes_ptr_t bytes;

    /** The reached set, the two frontiers, the rows reaching a landmark back and the rows that did
     *  before the current level, one bit per row each. */
    vram_bitset_t reached, frontier, next_frontier, reaches_back, back_settled;

    /** The levels that expanded anything, then the rows the last two sweeps reached either way. */
    u32_t* levels = nullptr;
    u32_t* expanded = nullptr;
    u32_t* back_levels = nullptr;
    u32_t* unreached = nullptr;

    /** The rows not reached both ways, the unreached among them. */
    u32_t* islands = nullptr;
    u32_t* reconnected = nullptr;

    /** The walk's and the merge's claim counters, the last pass's completion oracle. */
    u32_t* claims = nullptr;

    /** The rows and links the open sweep tile kept, read once its count finished. */
    u32_t* swept = nullptr;

    /** The rows the last audit did not reach both ways, which the heal walks. */
    u32_t* island_ids = nullptr;

    /** The ascending first rows of the parts the walks merge, uploaded once by the plan. */
    u32_t* part_bases = nullptr;

    /** The rows counted as changed so far, carried across a sweep's tiles. */
    vram_bitset_t changed;

    /** The reached rows linking to each row, which a reconnection spends. */
    u32_t* in_links = nullptr;
    std::size_t words = 0;
};

#pragma endregion Shard Types

/**
 *  @brief The core every dense index over a single uniform latency memory space on one GPU owns,
 *      and forwards its verbs to.
 *
 *  Speaks the RAM-resident dense index's verbs, with two divergences: a batch's IDs are one
 *  contiguous range, and its rows publish unordered on the device, turning visible to searches a
 *  whole batch at a time. The geometry is fixed at construction, and the first growth, a
 *  @c reserve or a claim, binds the device and the virtual-address ceiling; physical slabs map on
 *  demand, and fresh neighbors are sentinel-filled before any ID covering them can be assigned.
 *  The storage grows only under the one claim the claim word admits at a time, and the claim word,
 *  the filled extent, the watermark and the kept status are the only mutable host state; the
 *  published-row count is device memory no host thread dereferences.
 *
 *  Every node's key lives twice: in a host @c keyed_nodes, which also files every key's nodes, and
 *  on the device, where a walk resolves its nodes to keys. The claim holder writes both columns
 *  ahead of its launch and files the keys after its drain, so the map has one writer and a reader
 *  filtering by the watermark it acquired finds a key exactly when its node is visible.
 *
 *  @tparam vendor_type_ The vendor's runtime types and calls, @c cuda_vendor_t or @c hip_vendor_t.
 *  @tparam kernels_type_ The vendor's kernels, held as a member: they load and launch the kernels,
 *      stage the walks and queue a refinement's passes over the core they are handed.
 *  @tparam vector_dtype_ The stored scalar.
 *  @tparam cache_hints_type_ The host's line hints, which demote a filled bounce half before its
 *      copy.
 *  @tparam wait_type_ What a batch waits through while an earlier one turns the watermark.
 *  @tparam vectors_allocator_type_ The vectors matrix and norms column backend, on the device.
 *  @tparam neighbors_allocator_type_ The neighbors matrix backend, on the same device.
 *  @tparam traversals_allocator_type_ The traversal contexts' scratch backend.
 *  @tparam key_type_ The caller-visible identity of every node.
 *  @tparam key_hash_type_ The hash the key map files keys by.
 */
template <                                         //
    typename vendor_type_,                         //
    typename kernels_type_,                        //
    nk_dtype_t vector_dtype_,                      //
    typename cache_hints_type_,                    //
    typename wait_type_,                           //
    typename vectors_allocator_type_,              //
    typename neighbors_allocator_type_,            //
    typename traversals_allocator_type_,           //
    typename key_type_ = uuid_t,                   //
    typename key_hash_type_ = std::hash<key_type_> //
    >
struct vram_shard_core {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    using vendor_t = vendor_type_;
    using wait_t = wait_type_;
    using vectors_allocator_t = vectors_allocator_type_;
    using neighbors_allocator_t = neighbors_allocator_type_;
    using traversals_allocator_t = traversals_allocator_type_;
    using stream_handle_t = typename vendor_t::stream_handle_t;
    using event_t = typename vendor_t::event_t;
    using executor_t = typename vendor_t::executor_t;
    using kernel_table_t = typename vendor_t::kernel_table_t;
    using device_specs_t = typename vendor_t::device_specs_t;
    using api_t = typename vendor_t::api_t;
    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using dot_result_t = typename scalar_t::dot_result_t;

    /** Every kit accumulates the scalar's own dot type, which the norms column stores; the device
     *  writes that plain accumulator into these bytes, so the two must share a representation. */
    using dot_t = typename dot_result_t::raw_t;
    static_assert(sizeof(dot_result_t) == sizeof(dot_t));
    using norms_t = vector_norms<dot_t>;
    using vectors_view_t = nk::matrix_view<scalar_t>;
    using vector_view_t = nk::vector_view<scalar_t>;
    using node_id_t = dense::node_id_t;
    using node_ids_span_t = nk::vector_span<node_id_t>;
    using distance_t = dense::distance_t;
    using candidates_count_t = u32_t;

    /** The page-locked host bytes a save moves through, one batch at a time. */
    using bounce_t = growing_buffer<byte_t, typename vendor_t::pinned_allocator_t>;

    /** Rows one checkpoint batch carries between a matrix and the caller: one slab, the row
     *  granularity the RAM residency grows by. */
    inline static constexpr std::size_t batch_rows_log2_k = 20;
    using traversal_context_t = typename vendor_t::template traversal_context<traversals_allocator_t>;

    /** The strongest @c synchronization_t contract this design honors; nothing routes on it.
     *  Batches claim disjoint ID ranges and the publication protocol makes overlapping searches
     *  safe. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;

    /** The scored node's 32-bit ID space, less the sentinel, which marks a closed claim word. */
    inline static constexpr std::size_t max_node_count_k = (std::size_t {1} << 32) - 2u;

    /** Default visited-set slot count; the scratch slice must also fit the candidate cache. */
    inline static constexpr u32_t default_visited_capacity_k = 2048;

    /** Source rows one import window carries at most, so rows in another scalar cross through
     *  bounded staging. */
    inline static constexpr std::size_t import_window_rows_k = 4096;

    /** Bytes every kit loads at once, the stride the vector rows pad to. */
    inline static constexpr std::size_t fragment_bytes_k = 16;

    /** Whether f32 and f64 rows narrow into the stored scalar at unit RMS on an angular index: the
     *  narrow floats spend their range on direction, the one thing such an index reads. */
    inline static constexpr bool unit_rms_scalar_k = vector_dtype_ == nk_f16_k || vector_dtype_ == nk_e4m3_k ||
                                                     vector_dtype_ == nk_e5m2_k || vector_dtype_ == nk_e2m3_k ||
                                                     vector_dtype_ == nk_e3m2_k;

    /** The caller-visible identity of every node. */
    using key_t = key_type_;

    /** The keys of a batch, one per row. */
    using keys_view_t = nk::vector_view<key_t>;

    /** Every node's key on the host, and every key's nodes, growing in the RAM residency's
     *  chunks. */
    using keyed_t = keyed_nodes<key_t, node_id_t, chunk_schedule<10, 20>, max_node_count_k, forkunion_allocator_t,
                                fu::standard_atomic_ref, key_hash_type_>;

    /** The dimensions and connectivity every row keeps, fixed at construction. */
    using geometry_t = dense::geometry_t;

    /** The @p kernel every walk scores by and the @p geometry every row keeps, both checked at the
     *  first growth, and one allocator per storage concern, each carrying its placement; the index
     *  pins itself to the vectors allocator's device, demands the neighbors allocator agree, and
     *  keeps its host key column and key map on @p keys_allocator. */
    explicit vram_shard_core(nk_kernel_kind_t kernel, geometry_t geometry,
                             vectors_allocator_t const& vectors_allocator = {},
                             neighbors_allocator_t const& neighbors_allocator = {},
                             forkunion_allocator_t const& keys_allocator = {}) noexcept
        : device_ {vectors_allocator.device()}, kernel_ {kernel}, geometry_ {geometry},
          vectors_matrix_(vectors_allocator), norms_column_(vectors_allocator), keys_column_(vectors_allocator),
          neighbors_matrix_(neighbors_allocator),
          grown_((static_cast<void>(vendor_t::select(vectors_allocator.api(), vectors_allocator.device())),
                  event_t {vectors_allocator.api()})),
          keyed_(keys_allocator), keys_staging_(typename vendor_t::pinned_allocator_t {vectors_allocator.api()}),
          keys_crossed_ {event_t {vectors_allocator.api()}, event_t {vectors_allocator.api()}} {}
    vram_shard_core(vram_shard_core const&) = delete;
    vram_shard_core& operator=(vram_shard_core const&) = delete;

#pragma region Measures

    /** Rows visible to searches: the watermark every walk samples below and filters neighbors
     *  by. */
    std::size_t size() const noexcept {
        return fu::standard_atomic_ref<std::size_t const>(last_visible_id_).load(std::memory_order_acquire);
    }

    /** The first failure that left claimed rows unpublished, kept for the index's lifetime; it
     *  refuses every later mutation while searches keep running. */
    status_t status() const noexcept {
        return fu::standard_atomic_ref<status_t const>(status_).load(std::memory_order_acquire);
    }

    /** Rows mapped across the flat storage, the smallest of their counts, since slabs are a fixed
     *  byte size and each matrix maps a different row count per slab; dot's empty norms column
     *  bounds nothing. */
    std::size_t capacity() const noexcept {
        std::size_t const neighbors_row_bytes = connectivity() * sizeof(u32_t);
        std::size_t const vector_row_bytes = row_bytes();
        if (!neighbors_row_bytes || !vector_row_bytes) return 0;
        // The vectors matrix's first row holds the published-row count, ahead of row zero.
        std::size_t const vector_rows = vectors_matrix_.committed_bytes() / vector_row_bytes;
        std::size_t const by_vectors = vector_rows ? vector_rows - 1 : 0;
        std::size_t const by_neighbors = neighbors_matrix_.committed_bytes() / neighbors_row_bytes;
        std::size_t const by_keys = keys_column_.committed_bytes() / sizeof(key_t);
        std::size_t const by_rows = min_of(by_vectors, min_of(by_keys, by_neighbors));
        if (kernel_ == nk_kernel_dot_k) return by_rows;
        return min_of(by_rows, norms_column_.committed_bytes() / sizeof(norms_t));
    }

    /** The dimensions & connectivity fixed at construction. */
    geometry_t geometry() const noexcept { return geometry_; }
    std::size_t connectivity() const noexcept { return geometry_.connectivity; }
    std::size_t dimensions() const noexcept { return geometry_.dimensions; }

    /** The walk breadth @p config asks for, or twice the connectivity when it asks for none. */
    template <typename config_type_>
    candidates_count_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor
                                       : static_cast<candidates_count_t>(default_expansion_factor(connectivity()));
    }

    /** The pinned device's identity and limits, queried once at the first growth. */
    device_specs_t const& device_specs() const noexcept { return device_; }

    /** The driver table every call on the pinned device goes through, the one its storage
     *  carries. */
    api_t const& api() const noexcept { return vectors_matrix_.api(); }

    /** The metric every walk scores by, fixed at construction. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** The vendor's kernels, through which its tower reaches the hooks it serves verbs with. */
    kernels_type_& kernels() noexcept { return kernels_; }
    kernels_type_ const& kernels() const noexcept { return kernels_; }

    /** Whether rows keep norms: every metric but dot finalizes through them, so dot leaves the
     *  norms column empty and checkpoints carry none. */
    bool stores_norms() const noexcept { return kernel_ != nk_kernel_dot_k; }

    /** Device-side flat storage views - the audit surface for invariant checkers & tests. */
    vram_flat_storage_t storage() const noexcept {
        u32_t const fragments = row_fragments();
        return {vectors_rows_(),  norms_column_.base(),
                neighbor_rows_(), static_cast<u32_t>(connectivity()),
                fragments,        kernels_.lanes_per_row(*this),
                steps_per_row(),  static_cast<u32_t>(((u64_t {1} << 31) + fragments - 1) / fragments),
                kernel_};
    }

    /** Bytes of one packed logical row, bit-exact for sub-byte scalars. */
    std::size_t packed_row_bytes() const noexcept {
        return divide_round_up<std::size_t>(dimensions() * scalar_t::bits_per_dimension(), 8u);
    }

    /** Fragments per vector row, derived from the geometry like every row measure. */
    u32_t row_fragments() const noexcept {
        return divide_round_up<u32_t>(static_cast<u32_t>(packed_row_bytes()), fragment_bytes_k);
    }

    /** The vector row's byte length, rounded up to whole fragments; the tail bytes stay zero so
     *  every dot-derived metric is unaffected. */
    std::size_t row_bytes() const noexcept { return row_fragments() * fragment_bytes_k; }

    /** Steps the lanes cooperating on one row take across it. */
    u32_t steps_per_row() const noexcept { return divide_round_up(row_fragments(), kernels_.lanes_per_row(*this)); }

    /** Bytes of one builder warp's slice over @p selection_nodes cached candidates: the query span
     *  while walking, then the candidate cache, and per cached candidate its score, three ID spans,
     *  its distance and its norms. */
    u32_t build_scratch_bytes_per_warp(u32_t selection_nodes) const noexcept {
        u32_t const candidate_bytes = static_cast<u32_t>(row_bytes() + gpu_scored_node_bytes<node_id_t> +
                                                         3 * sizeof(u32_t) + sizeof(f32_t) + sizeof(norms_t));
        return max_of(static_cast<u32_t>(row_bytes()), selection_nodes * candidate_bytes);
    }

    /** The serialized extents: published nodes x logical dimensions & connectivity - the fragment
     *  padding is a storage detail that never reaches a file. */
    struct saved_shape_t {
        std::size_t nodes = 0;
        std::size_t dimensions = 0;
        std::size_t connectivity = 0;
    };
    saved_shape_t saved_shape() const noexcept { return {size(), dimensions(), connectivity()}; }

    /** Where a save writes: row-major host columns the caller sized from @c saved_shape, handed
     *  over at planning so no step names a destination of its own. */
    struct columns_t {

        /** The vectors matrix, one packed row per node. */
        bytes_span_t vectors;

        /** The neighbors matrix, @c connectivity IDs per node. */
        bytes_span_t neighbors;

        /** The norms column, one @c norms_t per node, or empty to write none. */
        bytes_span_t norms;
    };

    /** Where a load reads from - the read-only twin of @ref columns_t. */
    struct columns_view_t {

        /** The vectors matrix, one packed row per node. */
        bytes_view_t vectors;

        /** The neighbors matrix, @c connectivity IDs per node. */
        bytes_view_t neighbors;

        /** The norms column, one @c norms_t per node, or empty to derive them from the vectors. */
        bytes_view_t norms;
    };

    /** Bytes this index occupies - the committed device storage, the published-row count among
     *  it, the host keys and key map, the key staging, and the host struct; traversal scratch
     *  belongs to executors and contexts. */
    std::size_t memory_usage() const noexcept {
        return sizeof(vram_shard_core) + vectors_matrix_.committed_bytes() + norms_column_.committed_bytes() +
               keys_column_.committed_bytes() + neighbors_matrix_.committed_bytes() + keyed_.memory_usage() +
               keys_staging_.size() * sizeof(key_t);
    }

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
        if (node >= size()) return {{}, status_t::config_mismatch_k};
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
     *  Queues the copy of the stored row of @p node into @p destination with the fragment padding
     *  stripped on the caller's @p stream, whose device must be current; the caller waits once,
     *  after its last row. The bulk export is @c save.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p node is not visible, or @p destination is not as wide as a
     *    row.
     *  - @c unsupported_layout_k The @p destination is not contiguous.
     */
    status_t vector(node_id_t node, nk::vector_span<scalar_t> destination, stream_handle_t stream) const noexcept {
        if (node >= size()) return status_t::config_mismatch_k;
        if (destination.size() != dimensions()) return status_t::config_mismatch_k;
        if (!destination.is_contiguous()) return status_t::unsupported_layout_k;
        return succeeded(vendor_t::copy(api(), destination.byte_data(), vectors_rows_() + node * row_bytes(),
                                        packed_row_bytes(), stream))
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

#pragma endregion Measures

#pragma region Reservation

    using frontier_kind_t = vram_frontier_kind_t;

    /** One warp's arena slice for @p walks walks sharing one visited set over @p candidate_nodes
     *  candidates each: the exact visited set over every ID they claim, plus the spill tail when
     *  @p frontier_kind outgrows registers, what their expansions stage. */
    gpu_arena_shape_t arena_shape_for(u32_t candidate_nodes, frontier_kind_t frontier_kind,
                                      u32_t walks = 1) const noexcept {
        u32_t const connectivity = static_cast<u32_t>(this->connectivity()), lanes = kernels_type_::wave_lanes();
        u32_t const visits = walks * gpu_walk_visits(candidate_nodes, connectivity, lanes);
        u32_t const tail = frontier_kind == frontier_kind_t::wide_arena_k
                               ? walks * gpu_walk_tail_entries(candidate_nodes, connectivity, lanes)
                               : 0;
        return gpu_arena_shape(visits, tail, 0);
    }

    /** Requested capacity for a @c reserve() call. */
    struct reserve_capacity_t {
        std::size_t capacity = 0;

        /** Rows the index may ever host, clamped to the ID space; bound with the device by the
         *  first growth, past it every claim answers @c overflow_k. */
        std::size_t ceiling = max_node_count_k;

        /** A request for @p entries; the ceiling is set by name only, so no positional field binds
         *  to it. */
        reserve_capacity_t(std::size_t entries = 0) noexcept : capacity(entries) {}
    };

    /**
     *  Ensures the index can host at least @p config.capacity entries and answers the capacity now
     *  in effect - a hint, since every claim grows the storage it needs. Holds the claim word for
     *  the call, or runs under the word its caller closed; the first growth binds the device and
     *  the virtual-address ceiling, later ones only map slabs.
     *
     *  @par Statuses:
     *  - @c unknown_kind_k The metric is not one the kits score for this scalar.
     *  - @c config_mismatch_k The neighbors live on another device, the dimensions are zero, the
     *    connectivity is not a power of two, or no walk fits the geometry.
     *  - @c contended_k A batch or load is in flight.
     *  - @c overflow_k The capacity passes the ID space or the reserved range.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (config.capacity > max_node_count_k) return {capacity(), status_t::overflow_k};
        // A closed word is the caller's, who grows around its window; an open one is closed here,
        // so the growth has one writer.
        bool const closing = !claims_.closed();
        if (closing)
            if (status_t const closed = close_(); failed(closed)) return {capacity(), closed};
        status_t const mapped = map_nodes_(config.capacity, config.ceiling);
        if (closing) reopen_();
        return {capacity(), mapped};
    }

    /** Maps the storage for those of @p devices that can reach this one, as the sharded index
     *  holding the shard names its devices, so their copies read the rows directly; precedes
     *  @ref reserve. */
    void share_with_(std::bitset<gpu_devices_capacity_k> devices) noexcept {
        vectors_matrix_.share_with(devices), norms_column_.share_with(devices), neighbors_matrix_.share_with(devices);
    }

    /** Maps the storage, rows committed already included, for @p devices too, as a sharded index
     *  absorbing this shard names its own; @c config_mismatch_k when one cannot reach this one. */
    status_t grant_(std::bitset<gpu_devices_capacity_k> devices) noexcept {
        if (status_t const granted = vectors_matrix_.grant(devices); failed(granted)) return granted;
        if (status_t const granted = norms_column_.grant(devices); failed(granted)) return granted;
        return neighbors_matrix_.grant(devices);
    }

#pragma endregion Reservation

#pragma region Insertion

    /** Optional tuning knobs; zeros mean "derive from connectivity". */
    struct insert_config_t {

        /** How many candidates a construction walk keeps; rounded up to the register rungs of 2 to
         *  16 warps' worth of ranks and walked exactly past them. More candidates trade build speed
         *  for graph quality, and zero takes the measured breadth of @ref vram_insert_expansion. */
        candidates_count_t expansion_factor = 0;
    };

    /** The bytes an executor's run must hold before a context carved over it can insert @p rows
     *  rows, one aligned term per carve the verb makes; zero before the first growth binds the
     *  device. */
    std::size_t traversal_memory_usage(std::size_t rows, insert_config_t config = {}) const noexcept {
        if (!dimensions() || !device_.multiprocessor_count) return 0;
        auto aligned = [](std::size_t bytes) noexcept {
            return divide_round_up<std::size_t>(bytes, default_alignment_k) * default_alignment_k;
        };
        u32_t const expansion = static_cast<u32_t>(vram_insert_expansion(config.expansion_factor));
        u32_t const candidate_slots = candidate_slots_for_(expansion, 0);
        gpu_arena_shape_t const arena_shape = build_arena_shape_(expansion, candidate_slots, 1);
        std::size_t const arena_warps = device_.multiprocessor_count *
                                        kernels_type_::resident_blocks_(vram_kernel_t::insert_k, candidate_slots) *
                                        vram_warps_per_block_k;
        std::size_t const results_bytes = // A default single query's device outputs.
            non_zero_expansion_factor(search_config_t {}) * (sizeof(node_id_t) + sizeof(distance_t));
        // The staging also carries the back-link chains: a head per row after the batch, two words
        // per pick and a count.
        std::size_t const links_bytes = (size() + rows + 2 * rows * connectivity() + 1) * sizeof(u32_t);
        return aligned(sizeof(vram_telemetry_block_t)) + 2 * aligned(sizeof(u32_t)) +
               aligned(arena_warps * arena_shape.slice_bytes<node_id_t>()) +
               aligned(round_up_to_multiple<std::size_t>(rows * row_bytes() + import_bytes_(rows, nk_f64_k), 16) +
                       links_bytes + results_bytes);
    }

    /** Telemetry of one @c insert() batch. IDs are the contiguous range
     *  @b [first_id,first_id+count) - the assigned order of the incoming rows - and @c new_size is
     *  the range's end. */
    struct [[nodiscard]] insert_result_t {
        std::size_t new_size = 0;
        node_id_t first_id = node_id_unknown_k;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;

        /** Appended nodes no back-link reached, past what one re-selection weighs - GPU-only
         *  telemetry. */
        std::size_t unlinked_neighbors = 0;

        /** Frontier tails compacted to what can still rank - GPU-only telemetry. */
        std::size_t tail_compactions = 0;

        /** Entries tied at that radius a compacted tail still could not fit - GPU-only
         *  telemetry. */
        std::size_t tail_dropped = 0;

        /** Visited-set claims that ran out of probes and re-scored - GPU-only telemetry. */
        std::size_t visited_saturations = 0;

        /** Refine-side distance work, folded into @c computed_distances here, so always zero. */
        std::size_t computed_distances_in_refines = 0;

        /** Reverse-refine distance work, folded the same way, so always zero. */
        std::size_t computed_distances_in_reverse_refines = 0;

        /** Folds another shard's or launch's spend in; the IDs and the size stay this batch's. */
        insert_result_t& operator+=(insert_result_t const& other) noexcept {
            visited_members += other.visited_members;
            computed_distances += other.computed_distances;
            restarts += other.restarts;
            unlinked_neighbors += other.unlinked_neighbors;
            tail_compactions += other.tail_compactions;
            tail_dropped += other.tail_dropped;
            visited_saturations += other.visited_saturations;
            computed_distances_in_refines += other.computed_distances_in_refines;
            computed_distances_in_reverse_refines += other.computed_distances_in_reverse_refines;
            return *this;
        }
    };

    /**
     *  @brief Inserts @p new_rows under @p keys, growing the storage on demand, as the contiguous
     *      ID range @b [first_id,first_id+rows) claimed from the watermark.
     *
     *  The keys go into the host column and on the device ahead of the launch, every transfer and
     *  the launch riding the context's stream, then the stream drains, the keys file into the map,
     *  the watermark turns over the range, and the telemetry is reported. One batch or load holds
     *  the claim word at a time.
     *
     *  @par Statuses:
     *  - @c cancelled_k The index keeps it, as @ref status says.
     *  - @c overflow_k The rows pass the ceiling or the reserved range, or @ref status keeps it.
     *  - @c contended_k Another batch or load is in flight, or the claim word is closed.
     *  - @c uninitialized_k The @ref status keeps it.
     *  - @c config_mismatch_k The @p keys are not one per row, the rows are not as wide as the
     *    index, @p assigned_ids are not one per row, @p context rides another device or cannot
     *    walk, or @ref status keeps it.
     *  - @c unsupported_k The rows' scalar has no device cast into the stored one, or @ref status
     *    keeps it.
     *  - @c unsupported_layout_k Rows sit closer than one packed row apart, or @ref status keeps
     *    it.
     */
    expected<insert_result_t> insert(keys_view_t keys, rows_view_t new_rows, node_ids_span_t assigned_ids,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        // Zeroed per call rather than per launch, so the launches of a sharded plan on one context
        // add up.
        if (context && context.device() == device_.ordinal &&
            failed(
                vendor_t::fill(api(), context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), context.stream())))
            return {{}, status_t::device_error_k};
        std::size_t const rows = new_rows.rows;
        if (keys.size() != rows) return {{}, status_t::config_mismatch_k};
        if (!rows) return {{claimed_(), static_cast<node_id_t>(claimed_())}, status_t::success_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (!context || context.device() != device_.ordinal) return {{}, status_t::config_mismatch_k};
        auto const [first_id, claim_status] = claim_(rows);
        if (failed(claim_status)) return {{}, claim_status};
        expected<insert_result_t> inserted = submit_insert_(keys, new_rows, first_id, assigned_ids, context, config);
        if (!inserted) {
            unclaim_(first_id, rows);
            return inserted;
        }
        if (status_t const drained = context.wait(); failed(drained))
            return {{inserted.value.new_size, inserted.value.first_id}, fail_(drained)};
        publish_keys_({first_id, static_cast<node_id_t>(first_id + rows)});
        if (status_t const published = publish_(inserted.value.first_id, rows); failed(published))
            return {{inserted.value.new_size, inserted.value.first_id}, published};
        vram_telemetry_block_t const spent = context.spent();
        inserted.value.visited_members = spent.hops;
        inserted.value.computed_distances = spent.scored;
        inserted.value.restarts = spent.restarts;
        inserted.value.unlinked_neighbors = spent.lock_skips;
        inserted.value.tail_compactions = spent.tail_compactions;
        inserted.value.tail_dropped = spent.tail_dropped;
        inserted.value.visited_saturations = spent.visited_saturations;
        return inserted;
    }

#pragma endregion Insertion

#pragma region Search

    using search_config_t = vram_search_config_t;
    using search_result_t = vram_search_result_t;

    /** The bytes a search under the indexer @p heads needs: the plain walk's, plus the staging of
     *  the heads and their weights. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config,
                                       indexer_heads<scalar_t> const& heads) const noexcept {
        // The heads share the staging carve, and aligning their bytes apart can only overestimate
        // it.
        std::size_t const heads_bytes = heads.heads.extent(0) * (row_bytes() + sizeof(f32_t));
        return kernels_.traversal_memory_usage(*this, rows, config) +
               divide_round_up<std::size_t>(heads_bytes, default_alignment_k) * default_alignment_k;
    }

    /** The bytes an attend over @p rows rows needs, one walk per @p head_count of them: the walks'
     *  own, plus every row staged beside its output and carry, and beside those of every further
     *  one of @p shards, which the first shard gathers, then the window importing the heads and
     *  the outputs cast into the widest scalar. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config, std::size_t head_count,
                                       std::size_t shards = 1) const noexcept {
        std::size_t const rows_bytes = rows * (row_bytes() +
                                               shards * (dimensions() * sizeof(f32_t) + sizeof(softmax_state_t)) +
                                               dimensions() * sizeof(f64_t)) +
                                       import_bytes_(rows, nk_f64_k);
        return kernels_.traversal_memory_usage(*this, head_count ? rows / head_count : rows, config) +
               divide_round_up<std::size_t>(rows_bytes, default_alignment_k) * default_alignment_k;
    }

    /**
     *  @brief Finds the nearest published neighbors of every query row, ascending by distance with
     *      sentinel IDs marking absent matches. Safe to overlap with insertions on other streams.
     *
     *  @param[in] queries The query rows, in any scalar NumKong casts, imported on the device; f32
     *      and f64 rows narrow at unit RMS where @c unit_rms_scalar_k says.
     *  @param[out] result_ids IDs of the matches, @b [queries,count] with sentinel fills, in host
     *      or device memory.
     *  @param[out] result_distances Distances of the same matches, in host or device memory; an
     *      empty span skips them.
     *  @param[in] config The per-call breadth and match count.
     *  @param[in,out] context The stream every transfer and the launch ride.
     *  @param[in] completion Blocking drains the context's stream and reports the telemetry;
     *      deferred returns once queued, and the caller drains before reading either output.
     *  @return The match count per query, and the traversal telemetry this call spent.
     *
     *  @par Statuses:
     *  - @c unsupported_k A scan is asked of the vendor that has none, or the queries' scalar has
     *    no device cast into the stored one.
     *  - @c config_mismatch_k The outputs hold other than one row per query or fewer than
     *    @c count columns, the queries are not as wide as the index, or @p context rides another
     *    device or cannot walk.
     *  - @c unsupported_layout_k An output's rows stride closer than @c count values, or query
     *    rows sit closer than one packed row apart.
     *  - @c uninitialized_k The index never grew.
     */
    expected<search_result_t> search(rows_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, search_config_t config,
                                     traversal_context_t& context,
                                     completion_t completion = completion_t::blocking_k) noexcept {
        return settle_(kernels_.submit_search_(*this, queries, result_ids, result_distances, context, config, size()),
                       queries.rows, context, completion);
    }

    /** The batched @c search ranked by a lightning indexer: a node's distance is 1 − Σₕ wₕ ·
     *  ReLU(qₕ · k) over the @p heads of its query, which must match the queries in count and the
     *  index in dimensions, while the walk navigates by @p queries. A dot verb.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the outputs hold other
     *    than one row per query or fewer than @c count columns, the heads or weights disagree with
     *    the queries, rows are not as wide as the index, or @p context rides another device or
     *    cannot walk.
     *  - @c unsupported_k A scan is asked, or the queries' or heads' scalar has no device cast into
     *    the stored one.
     *  - @c unsupported_layout_k An output's rows stride closer than @c count values, or query or
     *    head rows sit closer than one packed row apart.
     *  - @c uninitialized_k The index never grew.
     */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads,
                                     nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, search_config_t config,
                                     traversal_context_t& context,
                                     completion_t completion = completion_t::blocking_k) noexcept
        requires floating_scalar<scalar_t>
    {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return settle_(
            kernels_.submit_search_(*this, queries, result_ids, result_distances, context, config, size(), heads),
            queries.rows, context, completion);
    }

    /**
     *  One query in, the nearest matches ascending in @c context.results - the correctness path; a
     *  one-query grid never fills the machine, the batched overload does.
     *
     *  @par Statuses:
     *  - @c unsupported_k A scan is asked of the vendor that has none.
     *  - @c config_mismatch_k The query is not as wide as the index, or @p context rides another
     *    device or cannot walk.
     *  - @c uninitialized_k The index never grew.
     */
    expected<search_result_t> search(vector_view_t query_vector, search_config_t config,
                                     traversal_context_t& context) noexcept {
        return search_(query_vector, context, config, size());
    }

    /**
     *  Filtered search is refused: a host predicate cannot run inside the walk, and a
     *  device-resident admission bitmap is a design of its own.
     *
     *  @par Statuses:
     *  - @c device_error_k Every call.
     */
    template <typename predicate_type_>
    expected<search_result_t> search_if(vector_view_t, search_config_t, predicate_type_&&,
                                        traversal_context_t&) noexcept {
        return {{}, status_t::device_error_k};
    }

    /**
     *  @brief Graph products of every left-hand row: each row keeps the pairs its walk scored, node
     *      IDs beside native dots, until its output row fills.
     *
     *  @param[in] left The left-hand rows, in any scalar NumKong casts, as in @c search.
     *  @param[out] node_ids The node ID of each kept pair, @b [rows,wanted] in walk order, sentinel
     *      past the kept pairs, in host or device memory.
     *  @param[out] products The device's native dot of each kept pair, parallel to @p node_ids and
     *      unspecified past them, in host or device memory.
     *  @param[out] counts Pairs scored per row, above @c wanted when the row was truncated, in host
     *      memory.
     *  @param[in] config The per-call breadth; @c count plays no part, the outputs' width does.
     *  @param[in,out] context The stream every transfer and the launch ride.
     *  @param[in] completion Blocking drains the context's stream and reports the telemetry;
     *      deferred returns once queued, and the caller drains before reading any output.
     *  @return The pairs scored over every row once blocking - zero when deferred, where @p counts
     *      carry them after the drain - and the traversal telemetry this call spent.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the outputs or
     *    @p counts disagree with the rows, the rows are not as wide as the index, or @p context
     *    rides another device or cannot walk.
     *  - @c unsupported_k A scan is asked, or the rows' scalar has no device cast into the stored
     *    one.
     *  - @c unsupported_layout_k An output's rows stride closer than its width, or rows sit closer
     *    than one packed row apart.
     *  - @c uninitialized_k The index never grew.
     */
    expected<search_result_t> multiply(rows_view_t left, nk::matrix_span<node_id_t> node_ids,
                                       nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                       search_config_t config, traversal_context_t& context,
                                       completion_t completion = completion_t::blocking_k) noexcept {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        expected<search_result_t> settled = settle_(
            kernels_.submit_multiply_(*this, left, node_ids, products, counts, context, config, size()), left.rows,
            context, completion);
        if (settled && completion == completion_t::blocking_k)
            for (u32_t const scored : counts) settled.value.count += scored;
        return settled;
    }

    /**
     *  @brief Grouped-query attention of every token over the published nodes,
     *      FlashAttention-style.
     *
     *  One walk per token ranks a node by the best of its heads, and every node it scores is folded
     *  into each head's running softmax, with the value row read once for all of them.
     *
     *  @param[in] queries The query heads, in any scalar NumKong casts as in @c search; row t ·
     *      @p head_count + h is head h of token t.
     *  @param[in] head_count The query heads sharing this key-value head; it must divide the rows,
     *      and their per-warp softmax states and accumulator rows must fit in shared memory beside
     *      the query span.
     *  @param[in] value_lookup A trivially copyable record whose <tt>vram_value_row(value_lookup,
     *      node_id)</tt> device overload points at the node's value row on a 16-byte boundary, as
     *      wide as the keys and stored like them, in memory this device reads; a null row refuses
     *      the call with @c shape_mismatch_k. The vendor's code object instantiates its attend
     *      kernels over the record's type, which spells itself into their names.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys, in host or
     *      device memory; rows in any scalar but f32 cast on the device first, and must fill whole
     *      bytes.
     *  @param[out] carry Each row's softmax state, one per row, in host or device memory; an empty
     *      span skips them.
     *  @param[in] config The per-call breadth; @c count plays no part.
     *  @param[in,out] context The stream every transfer and the launch ride.
     *  @param[in] completion As in @c multiply; a deferred call leaves a missing row to
     *      @c context.missed().
     *  @return The query rows attended, and the traversal telemetry this call spent.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, @p outputs or
     *    @p carry disagree with the rows, the queries are not as wide as the index, the heads'
     *    scratch fits no walk, or @p context rides another device or cannot walk.
     *  - @c unsupported_k A scan is asked, or the queries' or outputs' scalar has no device cast.
     *  - @c unsupported_layout_k Output or query rows sit closer than one packed row apart.
     *  - @c shape_mismatch_k The @p head_count is zero or does not divide the rows, or a walk
     *    visited a node with no value row.
     *  - @c uninitialized_k The index never grew.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     search_config_t config, traversal_context_t& context,
                                     completion_t completion = completion_t::blocking_k) noexcept {
        std::size_t const rows_count = queries.rows;
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = attend_refusal_(rows_count, outputs, carry); failed(refusal)) return {{}, refusal};
        expected<f32_t*> const submitted = kernels_.submit_attend_(*this, queries, head_count, value_lookup,
                                                                   score_scale, context, config, 1,
                                                                   exported_bytes_(outputs), size());
        if (!submitted || !rows_count) return {{}, submitted.status};
        auto const [kernels, kernels_status] = kernels_.kernels_(context);
        if (failed(kernels_status)) return {{}, kernels_status};
        if (status_t const copied = copy_attended_(*kernels, submitted.value, 1, outputs, carry, context.stream());
            failed(copied))
            return {{}, copied};
        expected<search_result_t> const attended = settle_({{rows_count}, status_t::success_k}, rows_count, context,
                                                           completion);
        if (attended && completion == completion_t::blocking_k && context.missed())
            return {{}, status_t::shape_mismatch_k};
        return attended;
    }

#pragma endregion Search

#pragma region Refinement

    /** The dense refine config; on a device the walk's breadth rounds up to the register rung of 2
     *  to 16 warps' worth of ranks and walks exactly past it. */
    using refine_config_t = dense::refine_config_t;

    /** A split's knobs: its refine's, and the seed that orders the parts' growth. */
    struct split_config_t {
        refine_config_t refine {};
        u64_t seed = 0;
    };

    using refine_report_t = dense::refine_report_t;
    using reachability_t = dense::reachability_t;

  private:
    using refine_scratch_t = vram_refine_scratch<traversal_context_t>;
    using audit_counters_t = vram_audit_counters_t;

    /** Gives an audit's counters back to the stream's pinned slots they came from. */
    struct audit_counters_return_t {
        typename vendor_t::pinned_slots_t* pinned_slots = nullptr;
        void operator()(audit_counters_t* counters) const noexcept { pinned_slots->give_back(counters); }
    };
    using audit_counters_ptr_t = std::unique_ptr<audit_counters_t, audit_counters_return_t>;
    static_assert(sizeof(audit_counters_t) <= vendor_t::pinned_slots_t::slot_bytes_k);

    /** One receiver's part of a split as the host assigns it: its nodes, the picked ones leading
     *  them, and the nodes assigned_nodes so far. */
    struct split_part_t {
        std::size_t entries = 0, picked_count = 0, assigned_nodes = 0;
    };

    /** A split's state beside the refinement it turns into, empty for a refine. */
    struct split_t {

        /** The index the split empties, until the scatter finished. */
        vram_shard_core* donor = nullptr;

        /** The receivers, one part each, and the one the refinement is bound to. */
        growing_buffer<vram_shard_core*> receivers {global_allocator};
        growing_buffer<split_part_t> parts {global_allocator};
        std::size_t bound_receiver = 0;

        /** The donor's nodes at planning, and the most nodes one part takes: those over the
         *  receivers, rounded up. */
        std::size_t entries = 0, part_capacity = 0;

        /** The device scratch, laid out as @c split_scratch_ reads it. */
        typename traversal_context_t::bytes_ptr_t device_bytes;

        /** Page-locked host bytes borrowed from the plan's context, laid out as @c split_layout_
         *  places them: the node parts, assigned into slots in place, the picked bitset, the
         *  receivers' columns, the roots planning uploads and the count of nodes that the last
         *  growth round placed. */
        byte_t* host_bytes = nullptr;

        /** The growth round the next batch queues from, zero once the growth converged and the
         *  picked pass queues next. */
        u32_t next_round = 0;

        /** The audits and islands of the receivers healed so far, summed. */
        reachability_t reachability {};
        std::size_t islands = 0;

        /** Rounds of re-selection each receiver runs, none once a stop cut the pass. */
        u32_t rounds = 1;

        explicit split_t(typename traversal_context_t::bytes_ptr_t&& empty) noexcept : device_bytes(std::move(empty)) {}
    };

  public:

    /**
     *  @brief One refinement in flight on this index's device - the CPU's @c refine_plan_t in the
     *      shape one host thread drives over many devices at once: @ref queue a phase, @ref wait
     *      for it, @ref advance past it, and @ref finish once no phase follows.
     *
     *  The phases re-select every node of the planned range by a walk, once per round of walks
     *  across while the shard holds several parts, then audit reachability both ways, then heal
     *  round by round while the islands shrink: re-select them, link whatever stays unreached into
     *  a reached pick or the landmark covering it, and link whatever reaches no landmark back to
     *  its nearest pick that does or to that landmark. A re-selection merges its picks into its
     *  live row at once, then tiles of target rows, sized from the limit in @c refine_config_t,
     *  collect the links the walked rows hold into each by a count, a scatter onto a tape, and a
     *  merge of each target's run into its live row, re-judging whatever overflows it. The walk
     *  runs once, as the CPU's refine does per call.
     *
     *  A phase queues its launches and the copy of its counters behind the context's event and
     *  never waits; @ref plan_refine made every allocation, so no phase allocates, and only a
     *  device fault fails one. Every phase takes the context the plan was opened on back from its
     *  caller. Demands an exclusive caller: the merge writes live neighbors with plain stores. A
     *  stopped sequence of phases leaves the graph correct and merely less improved.
     *
     *  A split's plan first partitions and scatters, as @ref plan_split describes, then binds the
     *  receivers one after another, each re-selecting its picked nodes, audited and healed, its
     *  tallies summing into the plan's.
     */
    struct refine_plan_t {

        /** An unbound plan over @p scratch - what a refused plan answers with. */
        explicit refine_plan_t(refine_scratch_t&& scratch) noexcept
            : scratch_(std::move(scratch)),
              split_(typename traversal_context_t::bytes_ptr_t {nullptr, scratch_.bytes.get_deleter()}) {}
        refine_plan_t(refine_plan_t&&) noexcept = default;
        refine_plan_t& operator=(refine_plan_t&&) noexcept = default;

        /**
         *  @brief Queues the current phase on @p context 's stream under @p config, then the copy
         *      of its counters and telemetry behind the context's event; never waits and never
         *      allocates anything at all.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers, or @ref status
         *    keeps it.
         *  - @c config_mismatch_k The @p context rides another device, or @ref status keeps it.
         *  - @c cancelled_k, @c unsupported_k, @c unsupported_layout_k or @c overflow_k The index
         *    keeps it, as @ref status says.
         */
        status_t queue(traversal_context_t& context, refine_config_t config) noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (failed(status_)) return status_;
            if (context.device() != index_->device_.ordinal) return status_t::config_mismatch_k;
            stream_handle_t const stream = context.stream();
            auto const [kernels, kernels_status] = index_->kernels_.kernels_(context);
            if (failed(kernels_status)) return kernels_status;
            switch (phase_) {
            case phase_t::partition_k:
                status_ = split_.next_round ? index_->queue_rounds_(*kernels, split_, stream)
                                            : index_->queue_picked_(*kernels, split_, stream);
                break;
            case phase_t::scatter_k: status_ = index_->queue_scatter_(*kernels, split_, stream); break;
            case phase_t::reselect_k: {
                iota_view<node_id_t> const range = round_nodes_();
                if (!range.empty())
                    status_ = index_->kernels_.refine_pass_(*index_, context, scratch_, config, vram_kernel_t::refine_k,
                                                            range, {}, parts_, round_);
                if (succeeded(status_)) status_ = index_->copy_counters_(scratch_, counters_.get(), 0, stream);
                break;
            }
            case phase_t::count_back_links_k:
            case phase_t::scatter_back_links_k:
            case phase_t::merge_back_links_k: status_ = queue_back_links_(context, config); break;
            case phase_t::audit_k: status_ = queue_sweeps_(context); break;
            case phase_t::heal_k:
                // The island IDs replace the range: offsets below their count index them. Only
                // island rows change, so the audit before stands for the reconnection, and the
                // fresh one settles what they alone reached. A continued audit sweeps on alone.
                if (!next_sweep_ && islands_)
                    status_ = index_->kernels_.refine_pass_(*index_, context, scratch_, config, vram_kernel_t::refine_k,
                                                            {0, islands_}, {scratch_.island_ids, islands_});
                if (succeeded(status_) && !next_sweep_)
                    status_ = index_->kernels_.queue_reconnect_(*index_, context, scratch_, islands_);
                if (succeeded(status_)) status_ = queue_sweeps_(context);
                break;
            default: return status_t::success_k;
            }
            if (succeeded(status_)) status_ = context.record();
            return status_;
        }

        /**
         *  @brief Sleeps until the queued phase finished, on @p context 's event.
         *
         *  @par Statuses:
         *  - @c uninitialized_k, @c config_mismatch_k, @c cancelled_k, @c unsupported_k,
         *    @c unsupported_layout_k or @c overflow_k A pass found the index keeping it, as @ref
         *    status says.
         */
        status_t wait(traversal_context_t& context) noexcept {
            if (!index_ || phase_ == phase_t::done_k) return status_;
            if (succeeded(status_)) status_ = context.event_.wait();
            return status_;
        }

        /** The barrier after a phase finished: reads its counters, picks the next phase, and
         *  answers whether one follows. */
        bool advance() noexcept {
            if (!index_ || failed(status_) || phase_ == phase_t::done_k) return false;
            if (phase_ == phase_t::partition_k) {
                // A growth batch that still placed nodes leaves others unplaced, so it grows on;
                // once one placed none, the picked pass queues, and the host assigns the slots they
                // placed, in turn.
                if (!split_.next_round) assign_slots_(), phase_ = phase_t::scatter_k;
                else if (!index_->placing_(split_)) split_.next_round = 0;
                return true;
            }
            if (phase_ == phase_t::scatter_k) {
                publish_parts_();
                bind_receiver_();
                return phase_ != phase_t::done_k;
            }
            audit_counters_t const& counters = *counters_;
            std::size_t const walked = walked_(), rows = index_->size();
            // Claim counts are the completion oracle: every offset claimed means every row walked.
            if (walked && counters.claims[0] < walked) {
                status_ = status_t::device_error_k;
                return false;
            }
            tally_.reselected_nodes += walked;
            if (phase_ == phase_t::reselect_k && walked) {
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
                if (counters.claims[1] < sweep_.size()) {
                    status_ = status_t::device_error_k;
                    return false;
                }
                if (sweep_.bound < rows) open_sweep_(sweep_.bound);
                else phase_ = ++round_ < rounds_ ? phase_t::reselect_k : phase_t::audit_k;
                return true;
            }
            if (phase_ == phase_t::reselect_k && ++round_ < rounds_) return true;
            if (phase_ == phase_t::reselect_k) {
                phase_ = phase_t::audit_k;
                return true;
            }
            // A last sweep that still expanded leaves rows unjudged, so the phase sweeps on.
            if (counters.last_expanded) return true;
            next_sweep_ = 0;
            tally_.reachability = audited_(counters, rows);
            // The device counter is zeroed once per plan and every heal adds to it,
            // so it reads back summed.
            tally_.islands_reconnected = counters.reconnected;
            u32_t const left = counters.islands;
            if (phase_ == phase_t::audit_k) tally_.islands_found = left;
            // A round that frees no island frees none on the next either, since the picks it needs
            // are islands too.
            bool const healing = left && (phase_ == phase_t::audit_k || left < islands_);
            islands_ = left;
            phase_ = healing ? phase_t::heal_k : phase_t::done_k;
            if (!healing && split_.receivers.size()) return settle_receiver_();
            return healing;
        }

        /** Stops the re-selection after the round that finished last, so the next phase audits and
         *  heals, and a split's later receivers skip theirs; a sweep under way finishes its round,
         *  and it is a no-op in any other phase, a split's partition and scatter included. */
        void stop() noexcept {
            if (phase_ == phase_t::reselect_k) rounds_ = 0, split_.rounds = 0, phase_ = phase_t::audit_k;
            else if (sweeping_()) rounds_ = 0, split_.rounds = 0;
        }

        /** Whether the current phase re-selects - the phase a caller's progress counts. */
        bool reselecting() const noexcept { return phase_ == phase_t::reselect_k; }

        /** Nodes re-selected so far. */
        std::size_t visited() const noexcept { return tally_.reselected_nodes; }

        /**
         *  @brief The pass, complete once @ref advance answered false: its telemetry read off
         *      @p context 's last copy; a failure a phase kept answers instead.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, or a pass found @ref status keeping it.
         *  - @c config_mismatch_k, @c unsupported_k, @c unsupported_layout_k or @c overflow_k
         *    A pass found the index keeping it, as @ref status says.
         *  - @c cancelled_k A phase still follows, or a pass found @ref status keeping it.
         */
        expected<refine_report_t> finish(traversal_context_t const& context) const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            refine_report_t result = tally_;
            vram_telemetry_block_t const spent = context.spent();
            result.changed_nodes = spent.changed_nodes;
            result.traversal_stats.visited_members = spent.hops;
            result.traversal_stats.computed_distances = spent.scored;
            result.traversal_stats.restarts = spent.restarts;
            result.traversal_stats.saturated_walks = spent.visited_saturations;
            if (failed(status_)) return {result, status_};
            return {result, phase_ == phase_t::done_k ? status_t::success_k : status_t::cancelled_k};
        }

      private:
        friend struct vram_shard_core;
        enum class phase_t : u8_t {
            partition_k,
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

        /** Opens the sweep's tile of target rows from @p first, up to @c tile_rows_ of them. */
        void open_sweep_(node_id_t first) noexcept {
            sweep_ = {first, static_cast<node_id_t>(min_of<std::size_t>(first + tile_rows_, index_->size()))};
            ++tally_.swept_tiles;
            phase_ = phase_t::count_back_links_k;
        }

        /** Queues the open tile's current kernels on @p context, then a copy of their counters. */
        status_t queue_back_links_(traversal_context_t& context, refine_config_t config) noexcept {
            iota_view<node_id_t> const sources = round_nodes_();
            status_t queued = status_t::success_k;
            if (phase_ == phase_t::count_back_links_k)
                queued = index_->queue_count_back_links_(context, scratch_, sources, sweep_, tile_rows_);
            else if (phase_ == phase_t::scatter_back_links_k)
                queued = index_->queue_scatter_back_links_(context, scratch_, sources, sweep_, tile_rows_,
                                                           swept_links_);
            else
                queued = index_->queue_merge_back_links_(context, scratch_, config, sources, sweep_, tile_rows_,
                                                         swept_links_);
            return succeeded(queued) ? index_->copy_counters_(scratch_, counters_.get(), 0, context.stream()) : queued;
        }

        /** Assigns every donor node its slot in place in the pinned host bytes, each part's picked
         *  nodes first and both runs in ID order. */
        void assign_slots_() noexcept {
            u32_t* const slots = reinterpret_cast<u32_t*>(split_.host_bytes);
            u32_t const* const picked = slots + split_.entries;
            auto const is_picked = [&](std::size_t node) noexcept { return (picked[node / 32] >> (node % 32)) & 1u; };
            for (split_part_t& part : split_.parts) part = {};
            for (std::size_t node = 0; node != split_.entries; ++node) {
                split_part_t& part = split_.parts[slots[node]];
                ++part.entries, part.picked_count += is_picked(node);
            }
            for (u32_t const picked_pass : {1u, 0u})
                for (std::size_t node = 0; node != split_.entries; ++node) {
                    if (is_picked(node) != picked_pass) continue;
                    std::size_t const part = slots[node];
                    slots[node] = static_cast<u32_t>(part * split_.part_capacity + split_.parts[part].assigned_nodes++);
                }
        }

        /** Files every donor key under its slot in its receiver, as a load's steps do, publishes
         *  every receiver's nodes at once under the closed claim words, and empties the donor. */
        void publish_parts_() noexcept {
            vram_shard_core& donor = *std::exchange(split_.donor, nullptr);
            u32_t const* const slots = reinterpret_cast<u32_t const*>(split_.host_bytes);
            for (std::size_t node = 0; node != split_.entries; ++node)
                split_.receivers[slots[node] / split_.part_capacity]->keyed_.assign(
                    static_cast<node_id_t>(slots[node] % split_.part_capacity),
                    donor.keyed_[static_cast<node_id_t>(node)]);
            for (std::size_t index = 0; index != split_.receivers.size(); ++index) {
                vram_shard_core& receiver = *split_.receivers[index];
                std::size_t const entries = split_.parts[index].entries;
                receiver.publish_keys_({0, static_cast<node_id_t>(entries)});
                if (status_t const published = receiver.publish_(0, entries); failed(published)) status_ = published;
            }
            donor.forget_();
        }

        /** Turns the pass to the next receiver holding nodes: its picked nodes as the range, or
         *  straight the audit when it has none or a stop cut the pass; past the last, reports
         *  the sums over all receivers. */
        void bind_receiver_() noexcept {
            while (split_.bound_receiver != split_.receivers.size() && !split_.parts[split_.bound_receiver].entries)
                ++split_.bound_receiver;
            if (split_.bound_receiver == split_.receivers.size()) {
                tally_.reachability = split_.reachability, tally_.islands_found = split_.islands;
                phase_ = phase_t::done_k;
                return;
            }
            index_ = split_.receivers[split_.bound_receiver];
            nodes_ = {0, static_cast<node_id_t>(split_.parts[split_.bound_receiver].picked_count)};
            round_ = 0, rounds_ = split_.rounds, islands_ = 0, next_sweep_ = 0;
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

        /** Queues the audit's next batch of sweeps on @p context, its seeds first when it
         *  starts. */
        status_t queue_sweeps_(traversal_context_t& context) noexcept {
            status_t const queued = next_sweep_
                                        ? index_->kernels_.queue_sweeps_(*index_, context, scratch_, counters_.get(),
                                                                         next_sweep_)
                                        : index_->kernels_.queue_audit_(*index_, context, scratch_, counters_.get());
            next_sweep_ = next_sweep_ ? 2 * next_sweep_ : 2;
            return queued;
        }

        /** The nodes the current round re-selects: the planned range first, every node below its
         *  bound in a later round. */
        iota_view<node_id_t> round_nodes_() const noexcept {
            return round_ ? iota_view<node_id_t> {0, nodes_.bound} : nodes_;
        }

        /** Rows the queued phase claimed: the round's nodes while re-selecting, and the islands
         *  when a heal round's queue began a fresh audit, which leaves the next sweep at two. */
        std::size_t walked_() const noexcept {
            if (phase_ == phase_t::reselect_k) return round_nodes_().size();
            return phase_ == phase_t::heal_k && next_sweep_ == 2 ? islands_ : 0;
        }

        refine_scratch_t scratch_;
        split_t split_;
        vram_shard_core* index_ = nullptr;
        audit_counters_ptr_t counters_ {nullptr, audit_counters_return_t {}};

        /** The IDs the first round re-selects. */
        iota_view<node_id_t> nodes_;

        /** Parts the rounds merge, the rounds, and the round queued or, once they
         *  ended, finished. */
        u32_t parts_ = 0;
        u32_t rounds_ = 1;
        u32_t round_ = 0;

        /** The islands the last audit left. */
        u32_t islands_ = 0;

        /** The audit sweep the next batch queues from, zero before an audit starts. */
        u32_t next_sweep_ = 0;

        /** The open sweep tile's target rows, the most rows a tile takes, and the links it kept. */
        iota_view<node_id_t> sweep_;
        std::size_t tile_rows_ = 0;
        u32_t swept_links_ = 0;
        phase_t phase_ = phase_t::done_k;
        status_t status_ = status_t::success_k;
        refine_report_t tally_;
    };

    /**
     *  @brief Opens a refinement of the nodes in @p nodes, clamped to the published IDs, on
     *      @p context 's stream: carves the scratch, grows the context's arena for the widest walk
     *      and its staging for the passes, takes page-locked counters and queues the upload of
     *      @p part_bases.
     *
     *  No phase allocates. An empty range heals alone.
     *
     *  Two or more @p part_bases below the published IDs split the rows into parts, as a shard
     *  that absorbed others' rows holds them: each node then first walks across the neighboring
     *  group of parts alone, round after round as @c part_across names them, from empty candidates
     *  and landmarks drawn there, before its own neighbors and the whole graph, and re-selects
     *  from the union. On kernels that merge no row until a round's walks end, a walk across never
     *  leaves its group; on kernels that sweep, it may follow a link a row of the group merged
     *  this round. The heal's passes walk no part first.
     *
     *  @param[in] nodes The ID range to walk; IDs at or past @c size() are skipped.
     *  @param[in] part_bases The ascending first rows of the parts; need not outlive the plan.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] context The driving context, which every phase takes back; its arena and
     *      staging grow to the pass's shape.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk.
     *  - @c bad_alloc_k The scratch, the arena or a tile cannot be placed, or the limit holds not
     *    even one tile row past the carried state.
     */
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, std::span<node_id_t const> part_bases,
                                        refine_config_t config, traversal_context_t& context) noexcept {
        std::size_t const rows = size();
        std::size_t parts = 0;
        while (parts != part_bases.size() && part_bases[parts] < rows) ++parts;
        if (parts < 2) parts = 0;
        auto [scratch, carve_status] = carve_scratch_(context, scratch_use_t::refine_k, parts);
        refine_plan_t plan {std::move(scratch)};
        if (failed(carve_status)) return {std::move(plan), carve_status};
        node_id_t const bound = static_cast<node_id_t>(min_of<std::size_t>(nodes.bound, rows));
        iota_view<node_id_t> const planned {min_of(nodes.first, bound), bound};
        // A later round walks every row below the planned bound.
        std::size_t const forward_rows = parts ? bound : planned.size();
        if (status_t const opened = open_refine_(plan, context, config, parts ? 2 : 1, forward_rows); failed(opened))
            return {std::move(plan), opened};
        // A pageable upload stages on the host at once and waits on nothing.
        if (parts && failed(vendor_t::copy(api(), plan.scratch_.part_bases, part_bases.data(),
                                           parts * sizeof(node_id_t), context.stream())))
            return {std::move(plan), status_t::device_error_k};
        plan.index_ = this;
        plan.nodes_ = planned;
        plan.parts_ = static_cast<u32_t>(parts);
        plan.rounds_ = parts ? static_cast<u32_t>(std::bit_width(parts - 1)) : 1u;
        plan.phase_ = plan.nodes_.empty() && plan.rounds_ == 1 ? refine_plan_t::phase_t::audit_k
                                                               : refine_plan_t::phase_t::reselect_k;
        if (!rows) plan.phase_ = refine_plan_t::phase_t::done_k;
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  One @ref refine_plan_t driven to its end on the calling thread, waiting once per phase.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew, or @ref status keeps it.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk, or @ref status
     *    keeps it.
     *  - @c cancelled_k, @c unsupported_k, @c unsupported_layout_k or @c overflow_k The index
     *    keeps it, as @ref status says.
     */
    expected<refine_report_t> refine(iota_view<node_id_t> nodes, std::span<node_id_t const> part_bases,
                                     refine_config_t config, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_refine(nodes, part_bases, config, context);
        if (failed(status)) return {{}, status};
        do
            if (status_t const queued = plan.queue(context, config); failed(queued) || failed(plan.wait(context)))
                break;
        while (plan.advance());
        return plan.finish(context);
    }

    /**
     *  @brief Opens a split of this index into @p receivers - the inverse of an absorb: the plan's
     *      first phases move every node into one graph-local part per receiver, then refine every
     *      receiver in turn over the nodes that lost a link, and heal.
     *
     *  Planning queues the partition behind every allocation: one root per part through @ref
     *  seeded_bijection under @p seed, then rounds of growth over the links, each part capped at
     *  the nodes over the receivers rounded up, and each node a round reaches kept by its lowest-ID
     *  proposer, so equal seeds over equal graphs split alike; nodes no round reached go to the
     *  smallest parts in ID order, and the nodes linking across are picked. The barrier after the
     *  partition assigns every node its slot, each part's picked nodes first. The scatter moves
     *  every node's vector, norms, key and neighbors on the device, the links inside its part
     *  rebased and the links across dropped, and the barrier after it files the keys into the
     *  receivers, publishes every receiver at once and empties this index. Each receiver then
     *  re-selects its picked nodes, audits and heals as @ref plan_refine does, and the tallies sum
     *  over every one of them.
     *
     *  The caller holds the claim words of this index and of every receiver closed for the plan's
     *  life, and nothing searches this index meanwhile. Receivers must be empty, distinct, and
     *  share this index's device, metric and geometry.
     *
     *  @param[in] receivers The empty indexes taking the parts, at least two.
     *  @param[in] core The member of each receiver holding its core.
     *  @param[in] config The refinement knobs, zeros meaning "derive from connectivity", and the
     *      seed picking the parts' roots, so equal seeds over equal graphs split alike.
     *  @param[in,out] context The driving context, which every phase takes back; its arena and
     *      staging grow to the pass's shape, and the partition's scratch is carved from it.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew.
     *  - @c config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *    index itself, one of another device, metric or geometry, or a @p context of another
     *    device or one that cannot walk.
     *  - @c contended_k This index's or a receiver's claim word is open.
     *  - @c overflow_k Part slots past the 32-bit IDs.
     */
    template <typename owner_type_>
    expected<refine_plan_t> plan_split(std::span<owner_type_*> receivers, vram_shard_core owner_type_::* core,
                                       split_config_t config, traversal_context_t& context) noexcept {
        using phase_t = typename refine_plan_t::phase_t;
        auto [scratch, carve_status] = carve_scratch_(context, scratch_use_t::refine_k);
        refine_plan_t plan {std::move(scratch)};
        if (failed(carve_status)) return {std::move(plan), carve_status};
        std::size_t const parts = receivers.size();
        if (parts < 2) return {std::move(plan), status_t::config_mismatch_k};
        if (!claims_.closed()) return {std::move(plan), status_t::contended_k};
        split_t& split = plan.split_;
        if (failed(split.receivers.reserve_exactly(parts)) || failed(split.parts.resize(parts)))
            return {std::move(plan), status_t::bad_alloc_k};
        for (std::size_t receiver = 0; receiver != parts; ++receiver) {
            owner_type_* const into = receivers[receiver];
            if (!into || &(into->*core) == this) return {std::move(plan), status_t::config_mismatch_k};
            for (std::size_t earlier = 0; earlier != receiver; ++earlier)
                if (receivers[earlier] == into) return {std::move(plan), status_t::config_mismatch_k};
            vram_shard_core& part = into->*core;
            if (part.device_.ordinal != device_.ordinal || part.kernel_ != kernel_ || !(part.geometry_ == geometry_) ||
                part.size())
                return {std::move(plan), status_t::config_mismatch_k};
            if (!part.claims_.closed()) return {std::move(plan), status_t::contended_k};
            split.receivers.push_back_reserved(&part);
        }
        split.donor = this;
        split.entries = size();
        split.part_capacity = divide_round_up<std::size_t>(split.entries, parts);
        if (parts * split.part_capacity >= std::numeric_limits<u32_t>::max())
            return {std::move(plan), status_t::overflow_k};
        // Every receiver holds a full part's nodes, sentinel-filled, before the partition queues.
        stream_handle_t const stream = context.stream();
        for (vram_shard_core* const part : split.receivers)
            if (status_t const grown = part->grow_nodes_(split.part_capacity, stream); failed(grown))
                return {std::move(plan), grown};
        auto const [kernels, kernels_status] = kernels_.kernels_(context);
        if (failed(kernels_status)) return {std::move(plan), kernels_status};
        status_t opened = open_refine_(plan, context, config.refine, 1, split.part_capacity);
        if (succeeded(opened) && split.entries) opened = carve_split_(split, config.seed, context);
        if (succeeded(opened) && split.entries) opened = queue_partition_(*kernels, split, stream);
        if (failed(opened)) return {std::move(plan), opened};
        plan.index_ = this;
        plan.phase_ = split.entries ? phase_t::partition_k : phase_t::done_k;
        return {std::move(plan), status_t::success_k};
    }

    /** One reachability audit in flight on this index's device, in the shape one host thread drives
     *  over many devices at once: @ref queue a batch of sweeps, @ref wait for it, and @ref finish
     *  once no batch follows, so every shard's batch is queued before any waits. */
    struct health_plan_t {

        /** An unbound plan over @p scratch - what an empty index or a refused plan answers with. */
        explicit health_plan_t(refine_scratch_t&& scratch) noexcept : scratch_(std::move(scratch)) {}
        health_plan_t(health_plan_t&&) noexcept = default;
        health_plan_t& operator=(health_plan_t&&) noexcept = default;

        /** Whether another batch of sweeps follows, the last one still expanding. */
        bool running() const noexcept { return phase_ == phase_t::audit_k; }

        /** Queues the next batch of sweeps on @p context, the landmarks first, doubling the batch
         *  each time; a no-op unless @ref running. */
        status_t queue(traversal_context_t& context) noexcept {
            if (phase_ != phase_t::audit_k) return status_;
            status_ = next_sweep_
                          ? index_->kernels_.queue_sweeps_(*index_, context, scratch_, counters_.get(), next_sweep_)
                          : index_->kernels_.queue_audit_(*index_, context, scratch_, counters_.get());
            next_sweep_ = next_sweep_ ? 2 * next_sweep_ : 2;
            phase_ = succeeded(status_) ? phase_t::copy_k : phase_t::done_k;
            return status_;
        }

        /** Sleeps until the queued batch finished on @p context 's event, then reads whether it
         *  still expanded; a no-op unless a batch was queued. */
        status_t wait(traversal_context_t& context) noexcept {
            if (phase_ != phase_t::copy_k) return status_;
            status_ = context.wait();
            phase_ = succeeded(status_) && counters_->last_expanded ? phase_t::audit_k : phase_t::done_k;
            return status_;
        }

        /** The audit over the rows published when it was planned; a failure a batch kept answers
         *  instead. */
        expected<reachability_t> finish() const noexcept {
            if (failed(status_)) return {{}, status_};
            return {index_ ? audited_(*counters_, rows_) : reachability_t {}, status_t::success_k};
        }

      private:
        friend struct vram_shard_core;
        enum class phase_t : u8_t { audit_k, copy_k, done_k };

        refine_scratch_t scratch_;
        vram_shard_core* index_ = nullptr;
        audit_counters_ptr_t counters_ {nullptr, audit_counters_return_t {}};
        std::size_t rows_ = 0;

        /** The sweep the next batch queues from, zero before the landmarks. */
        u32_t next_sweep_ = 0;
        phase_t phase_ = phase_t::done_k;
        status_t status_ = status_t::success_k;
    };

    /**
     *  @brief Opens a reachability audit of the rows published now on @p context 's stream:
     *      carves the audit share of the scratch and takes page-locked counters; queues nothing.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p context rides another device.
     */
    expected<health_plan_t> plan_health(traversal_context_t& context) noexcept {
        using bytes_ptr_t = typename traversal_context_t::bytes_ptr_t;
        std::size_t const rows = size();
        if (!rows)
            return {health_plan_t {refine_scratch_t {
                        bytes_ptr_t {nullptr, typename bytes_ptr_t::deleter_type {context.allocator_, 0}}}},
                    status_t::success_k};
        auto [scratch, carve_status] = carve_scratch_(context, scratch_use_t::audit_k);
        health_plan_t plan {std::move(scratch)};
        if (failed(carve_status)) return {std::move(plan), carve_status};
        plan.counters_ = take_audit_counters_(context);
        if (!plan.counters_) return {std::move(plan), status_t::bad_alloc_k};
        plan.index_ = this;
        plan.rows_ = rows;
        plan.phase_ = health_plan_t::phase_t::audit_k;
        return {std::move(plan), status_t::success_k};
    }

    /** Measures forward reachability from as many stride-spread landmarks as one search draws -
     *  the connectivity; any shortfall against @c size() is the population no search can
     *  discover. Read-only on the graph, so concurrent insertions may leave fresh nodes
     *  uncounted. One @ref health_plan_t driven on the calling thread, waiting once per batch of
     *  sweeps, the batches doubling until one expands nothing.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p context rides another device.
     */
    expected<reachability_t> health(traversal_context_t& context) noexcept {
        auto [plan, status] = plan_health(context);
        if (failed(status)) return {{}, status};
        while (plan.running())
            if (failed(plan.queue(context)) || failed(plan.wait(context))) break;
        return plan.finish();
    }

    /** The bytes an executor's run must hold before a context carved over it can refine the rows
     *  published now under @p config, @p walks walks of a row sharing one visited set: the
     *  context's counters, the walks' arena, the passes' staging, and the scratch the refine
     *  carves once, then a split's partition scratch when @p receivers names two or more. */
    std::size_t traversal_memory_usage(refine_config_t config, u32_t walks = 1,
                                       std::size_t receivers = 0) const noexcept {
        if (!dimensions() || !device_.multiprocessor_count) return 0;
        auto aligned = [](std::size_t bytes) noexcept {
            return divide_round_up<std::size_t>(bytes, default_alignment_k) * default_alignment_k;
        };
        return aligned(sizeof(vram_telemetry_block_t)) + 2 * aligned(sizeof(u32_t)) +
               aligned(refine_arena_bytes_(config, walks)) +
               aligned(refine_staging_bytes_(refine_tile_rows_(size(), config, walks))) +
               aligned(scratch_bytes_(scratch_use_t::refine_k)) +
               (receivers > 1 ? aligned(split_layout_(size(), receivers).device_bytes) : 0);
    }

#pragma endregion Refinement

#pragma region Checkpoints

    /**
     *  @brief One save in flight: the watermark it took, the stream its copies ride, and the
     *      batches it hands out through a double-buffered pinned bounce, since a device matrix
     *      cannot be read in place.
     *
     *  Vectors travel as logical rows with the fragment padding stripped, neighbors verbatim under
     *  the positional sentinel convention, and norms verbatim when the columns carry theirs.
     *
     *  A batch crosses half a batch at a time: each half reaches the host while the next one is
     *  still copying off the device, so the host awaits each half's event once, and the last one is
     *  copied in @c finish. One host thread drives a device, so batches are taken one after
     *  another, though the tallies keep the shape a threaded caller expects.
     */
    struct [[nodiscard]] save_plan_t {

        /** Where the rows go, vetted against the watermark at planning. */
        columns_t columns {};

        /** The stream every device-to-host copy queues on. */
        stream_handle_t stream = nullptr;

        /** Page-locked host bytes in two halves of @ref half_rows rows each: packed vector rows,
         *  then their neighbor IDs, then their norms. */
        bounce_t bounce {typename vendor_t::pinned_allocator_t {}};

        /** Rows under the watermark at planning. */
        node_id_t visible = 0;

        /** Rows whose vectors and neighbors arrived, tallied through @c atomic_ref. */
        std::size_t covered = 0;

        save_plan_t() noexcept = default;

        /** The index snapshotted into @p columns on @p stream under the watermark at @p visible;
         *  only @ref plan_save calls it, with the index's device current, where the events are
         *  born, and sizes the bounce afterwards. */
        save_plan_t(vram_shard_core const& index, columns_t columns, typename executor_t::stream_t const& stream,
                    node_id_t visible) noexcept
            : columns(columns), stream(stream.handle), bounce(typename vendor_t::pinned_allocator_t {stream}),
              visible(visible), crossed_ {event_t {index.api()}, event_t {index.api()}}, index_(&index) {}
        save_plan_t(save_plan_t&& other) noexcept
            : columns(other.columns), stream(other.stream), bounce(std::move(other.bounce)), visible(other.visible),
              covered(other.covered), crossed_ {std::move(other.crossed_[0]), std::move(other.crossed_[1])},
              pending_(std::exchange(other.pending_, {})), pending_half_(other.pending_half_),
              index_(std::exchange(other.index_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        // The tally is this plan's own: a copy would tally into a twin `finish` never reads.
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;

        /** A save owns nothing but its bounce, so a dying plan leaves the index as it found it,
         *  once the copies into its pending half finished. */
        ~save_plan_t() noexcept {
            if (!pending_.index_nodes.empty()) [[maybe_unused]]
                status_t const drained = crossed_[pending_half_].wait();
        }

        /** One step of the plan: one bounce-sized batch of the rows under the watermark. */
        struct step_t {

            /** The step's position in the plan - what a fan-out hands to a stream. */
            std::size_t position = 0;

            /** Where the rows sit in the index, in its own ID space. */
            iota_view<node_id_t> index_nodes;

            /** Where they sit in the caller's buffer, counted from its first row. */
            iota_view<node_id_t> source_nodes;
        };

        /** Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = position * batch_rows();
            std::size_t const bound = min_of(first + batch_rows(), static_cast<std::size_t>(visible));
            step.position = position;
            step.index_nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_nodes = step.index_nodes;
            return step;
        }

        /** The steps the save spans - one bounce-sized batch each. */
        indexed_range<save_plan_t> steps() const noexcept {
            return {this, index_ && visible ? divide_round_up<std::size_t>(visible, batch_rows()) : 0};
        }

        /** Rows one batch carries - a bounce's worth, or every row when fewer. */
        std::size_t batch_rows() const noexcept {
            return min_of(static_cast<std::size_t>(visible), power_of_two<std::size_t>(batch_rows_log2_k));
        }

        /** Rows one half of the bounce carries: half a batch, so two halves pin what one batch
         *  would. */
        std::size_t half_rows() const noexcept { return divide_round_up<std::size_t>(batch_rows(), 2); }

        /**
         *  @brief Copies step @p step 's rows off the device into the plan's own columns.
         *
         *  The vectors go with the fragment padding stripped, then the neighbors with every ID past
         *  the watermark folded to the sentinel, then the norms where the columns take them. The
         *  step's last half is still copying when it returns, and finishes with the next step or in
         *  @c finish.
         *
         *  @param[in] step The step to copy, as @ref steps hands them out.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The step is empty, or the columns cannot hold its rows.
         */
        status_t save(step_t const& step) noexcept {
            if (!index_) return status_t::uninitialized_k;
            // Shards' steps interleave on one thread, so another shard's device may be current.
            if (failed(vendor_t::select(index_->api(), index_->device_specs().ordinal)))
                return status_t::device_error_k;
            iota_view<node_id_t> const rows = step.index_nodes;
            if (rows.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(node_id_t);
            std::size_t const first = step.source_nodes.first;
            if (flat_rows(columns.vectors, first, rows.size(), scalar_row_bytes).empty() ||
                flat_rows(columns.neighbors, first, rows.size(), neighbors_row_bytes).empty() ||
                flat_rows(columns.norms, first, rows.size(), sizeof(norms_t)).empty() != columns.norms.empty())
                return status_t::config_mismatch_k;
            for (std::size_t chunk = rows.first; chunk < rows.bound; chunk += half_rows()) {
                std::size_t const bound = min_of<std::size_t>(chunk + half_rows(), rows.bound);
                // Every half but the pending one has finished, so the other one is free.
                std::size_t const half = pending_.index_nodes.empty() ? 0 : pending_half_ ^ 1;
                byte_t* const bounce_vectors =
                    bounce.data() + half * half_rows() * (scalar_row_bytes + neighbors_row_bytes + sizeof(norms_t));
                byte_t* const bounce_neighbors = bounce_vectors + half_rows() * scalar_row_bytes;
                byte_t* const bounce_norms = bounce_neighbors + half_rows() * neighbors_row_bytes;
                std::size_t const count = bound - chunk;
                if (failed(vendor_t::copy_2d(index_->api(), bounce_vectors, scalar_row_bytes,
                                             index_->vectors_rows_() + chunk * index_->row_bytes(), index_->row_bytes(),
                                             scalar_row_bytes, count, stream)) ||
                    (!columns.norms.empty() &&
                     failed(vendor_t::copy(index_->api(), bounce_norms,
                                           index_->norms_column_.base() + chunk * sizeof(norms_t),
                                           count * sizeof(norms_t), stream))) ||
                    failed(vendor_t::copy(index_->api(), bounce_neighbors,
                                          index_->neighbors_matrix_.base() + chunk * neighbors_row_bytes,
                                          count * neighbors_row_bytes, stream)) ||
                    failed(crossed_[half].record(stream)))
                    return status_t::device_error_k;
                // The previous half reaches the host while this one copies off the device.
                if (status_t const copied = copy_pending_(); failed(copied)) return copied;
                pending_.index_nodes = {static_cast<node_id_t>(chunk), static_cast<node_id_t>(bound)};
                pending_.source_nodes = {static_cast<node_id_t>(first + chunk - rows.first),
                                         static_cast<node_id_t>(first + bound - rows.first)};
                pending_half_ = half;
            }
            return status_t::success_k;
        }

        /**
         *  @brief Pronounces the snapshot complete once the last half finished: every batch's
         *      vectors and neighbors in the columns. A success spends the plan; a refusal leaves
         *      it whole, so the missing steps can still arrive.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k Some step's rows never arrived.
         */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (failed(vendor_t::select(index_->api(), index_->device_specs().ordinal)))
                return status_t::device_error_k;
            if (status_t const copied = copy_pending_(); failed(copied)) return copied;
            std::size_t const nodes = static_cast<std::size_t>(visible);
            if (fu::standard_atomic_ref<std::size_t> {covered}.load(std::memory_order_acquire) != nodes)
                return status_t::config_mismatch_k;
            index_ = nullptr;
            return status_t::success_k;
        }

      private:

        /** Awaits the pending half's copies, then moves its rows into the columns, shifting every
         *  neighbor under the watermark and folding every other onto the sentinel. */
        status_t copy_pending_() noexcept {
            iota_view<node_id_t> const rows = pending_.source_nodes;
            if (pending_.index_nodes.empty()) return status_t::success_k;
            if (status_t const drained = crossed_[pending_half_].wait(); failed(drained)) return drained;
            pending_ = {};
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const connectivity = index_->connectivity();
            std::size_t const neighbors_row_bytes = connectivity * sizeof(node_id_t);
            byte_t* const bounce_vectors = bounce.data() +
                                           pending_half_ * half_rows() *
                                               (scalar_row_bytes + neighbors_row_bytes + sizeof(norms_t));
            byte_t* const bounce_neighbors = bounce_vectors + half_rows() * scalar_row_bytes;
            byte_t* const bounce_norms = bounce_neighbors + half_rows() * neighbors_row_bytes;
            bytes_span_t const vectors_rows = flat_rows(columns.vectors, rows.first, rows.size(), scalar_row_bytes);
            bytes_span_t const neighbors_rows = flat_rows(columns.neighbors, rows.first, rows.size(),
                                                          neighbors_row_bytes);
            bytes_span_t const norms_rows = flat_rows(columns.norms, rows.first, rows.size(), sizeof(norms_t));
            [[maybe_unused]] status_t const copied_vectors = copy_bytes(vectors_rows,
                                                                        bytes_of(bounce_vectors, vectors_rows.size()));
            if (!norms_rows.empty()) [[maybe_unused]]
                status_t const copied_norms = copy_bytes(norms_rows, bytes_of(bounce_norms, norms_rows.size()));

            std::span<node_id_t> const neighbors_view {reinterpret_cast<node_id_t*>(bounce_neighbors),
                                                       rows.size() * connectivity};
            for (node_id_t& neighbor : neighbors_view) neighbor = neighbor < visible ? neighbor : node_id_unknown_k;
            [[maybe_unused]] status_t const copied_neighbors = copy_bytes(neighbors_rows,
                                                                          std::as_bytes(neighbors_view));
            fu::standard_atomic_ref<std::size_t> {covered}.add(rows.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Marks the last copies into each half of the bounce, born with the plan. */
        event_t crossed_[2];

        /** The rows whose copies sit in a half not yet on the host; empty when none do. */
        step_t pending_ {};

        /** The half @c pending_ sits in. */
        std::size_t pending_half_ = 0;

        /** The index being snapshotted; null in the empty plan a refused @ref plan_save returns,
         *  and once @c finish succeeded. */
        vram_shard_core const* index_ = nullptr;
    };

    /**
     *  @brief Opens a save into @p columns under @p synchronization on @p stream: takes the
     *      watermark and pins the double-buffered bounce for it.
     *
     *  @c concurrent_k answers @c config_mismatch_k, since the device storage cannot be probed for
     *  settlement; @c exclusive_k reads like @c immutable_k. Columns too short for the rows under
     *  it are refused here, so no step ever writes past them.
     *
     *  @param[out] columns Where the rows go, sized from @c saved_shape.
     *  @param[in] synchronization The contract the caller certifies.
     *  @param[in] stream The stream every copy queues on.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p synchronization is @c concurrent_k, or @p columns are too
     *    short for the rows.
     *  - @c uninitialized_k The index never grew.
     */
    expected<save_plan_t> plan_save(columns_t columns, synchronization_t synchronization,
                                    typename executor_t::stream_t const& stream) const noexcept {
        if (synchronization == synchronization_t::concurrent_k) return {{}, status_t::config_mismatch_k};
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        std::size_t const rows = size();
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return {{}, status_t::config_mismatch_k};
        if (failed(vendor_t::select(api(), device_.ordinal))) return {{}, status_t::device_error_k};
        save_plan_t plan {*this, columns, stream, static_cast<node_id_t>(rows)};
        std::size_t const bounce_bytes = 2 * plan.half_rows() *
                                         (packed_row_bytes() + connectivity() * sizeof(node_id_t) + sizeof(norms_t));
        if (bounce_bytes && failed(plan.bounce.resize_uninitialized(bounce_bytes))) return {{}, status_t::bad_alloc_k};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  Serializes every row under the watermark into @p columns on the calling thread - the
     *  one-shot over @ref plan_save for a caller with no streams to fan the steps out over.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p synchronization is @c concurrent_k, or @p columns are too
     *    short for the rows.
     *  - @c uninitialized_k The index never grew.
     */
    status_t save(columns_t columns, synchronization_t synchronization,
                  typename executor_t::stream_t const& stream) const noexcept {
        auto [plan, status] = plan_save(columns, synchronization, stream);
        if (failed(status)) return status;
        for (typename save_plan_t::step_t const step : plan.steps())
            if (status_t const saved = plan.save(step); failed(saved)) return saved;
        return plan.finish();
    }

    /**
     *  @brief One load in flight: the rows it appends past every claimed one, the storage grown to
     *      hold them, and the steps they span.
     *
     *  Each step copies on the stream from host or device columns, or from another index's
     *  storage, and @ref finish derives the norms no step carried before it publishes the rows.
     *
     *  Holds the claim for its whole life, so nothing else claims rows while it runs. A source's
     *  IDs are local to the rows it delivers, so every batch shifts them onto the appended range
     *  and folds every other ID onto the sentinel, on the device. Repeated plans append, which is
     *  how several parts go into one shard. A step of a part the vendor's reader registered reads
     *  into one of its two staging slots while the other's copies run, and a slot refills only
     *  once its event says the copies out of it finished; @c finish is the one full wait. A plan
     *  that dies before @c finish claims nothing and leaves the storage grown for the next one,
     *  its rows unfilled, so the next claim sentinel-fills over whatever its steps wrote.
     */
    struct [[nodiscard]] load_plan_t {

        /** The stream every copy into the index and every kernel queue on, whose device every step
         *  and @c finish select first. */
        stream_handle_t stream = nullptr;

        /** The first appended ID - the row the claim started at. */
        node_id_t base = 0;

        /** Rows the plan appends. */
        std::size_t rows = 0;

        /** Rows whose vectors, neighbors and keys arrived, tallied through @c atomic_ref. */
        std::size_t covered = 0;

        /** Rows whose norms arrived from the columns, tallied through @c atomic_ref; @c finish
         *  derives every row's when it falls short. */
        std::size_t covered_norms = 0;

        load_plan_t() noexcept = default;

        /** The index appended to at @p base on @p stream, launching through @p kernels, under
         *  the claim over its rows or a claim word its caller closed; only @ref plan_load calls
         *  it, with the index's device current, where the events are born. */
        load_plan_t(vram_shard_core& index, typename executor_t::stream_t const& stream, kernel_table_t const& kernels,
                    node_id_t base, std::size_t rows) noexcept
            : stream(stream.handle), base(base), rows(rows), drained_ {event_t {index.api()}, event_t {index.api()}},
              index_(&index), kernels_(&kernels) {}
        load_plan_t(load_plan_t&& other) noexcept
            : stream(other.stream), base(other.base), rows(std::exchange(other.rows, 0)), covered(other.covered),
              covered_norms(other.covered_norms), drained_ {std::move(other.drained_[0]), std::move(other.drained_[1])},
              reader_(std::move(other.reader_)), chunks_(other.chunks_), index_(std::exchange(other.index_, nullptr)),
              kernels_(other.kernels_) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        // The tallies are this plan's own: a copy would tally into a twin `finish` never reads.
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** An unfinished plan gives its claim back, marking its stream so the next grow follows
         *  every copy it queued, and awaiting that mark before its staging frees. */
        ~load_plan_t() noexcept {
            if (!index_ || !rows) return;
            if (succeeded(vendor_t::select(index_->api(), index_->device_specs().ordinal)) &&
                succeeded(index_->grown_.record(stream)) && chunks_) [[maybe_unused]]
                status_t const drained = index_->grown_.wait();
            // Under the claim or the closed word, which every reader of the filled extent holds.
            index_->filled_rows_ = min_of(index_->filled_rows_, static_cast<std::size_t>(base));
            index_->unclaim_(base, rows);
        }

        /** One step of the plan: one batch of the appended nodes. */
        struct step_t {

            /** The step's position in the plan - what a fan-out hands to a stream. */
            std::size_t position = 0;

            /** Where the rows go in the index, in its own ID space, from @c base. */
            iota_view<node_id_t> index_nodes;

            /** Where they come from in the caller's buffer, from its first row. */
            iota_view<node_id_t> source_nodes;
        };

        /** Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = static_cast<std::size_t>(base) + position * batch_rows();
            std::size_t const bound = min_of(first + batch_rows(), static_cast<std::size_t>(base) + rows);
            step.position = position;
            step.index_nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_nodes = {static_cast<node_id_t>(first - base), static_cast<node_id_t>(bound - base)};
            return step;
        }

        /** The steps the append spans - one batch each. */
        indexed_range<load_plan_t> steps() const noexcept {
            return {this, index_ && rows ? divide_round_up(rows, batch_rows()) : 0};
        }

        /** Rows one batch carries - a slab, or every row when fewer. */
        std::size_t batch_rows() const noexcept { return min_of(rows, power_of_two<std::size_t>(batch_rows_log2_k)); }

        /**
         *  @brief Copies step @p step 's rows out of @p columns on the plan's stream, through
         *      the vendor reader's staging slot where it registered their part, and their keys
         *      into the host's.
         *
         *  The vectors go into the fragment-padded matrix rows, whose fresh tails arrive zeroed,
         *  and their norms where the columns carry them, then the neighbors, which
         *  @c launch_rebase_links_ folds on the plan's stream, every ID inside @p source_range onto
         *  the appended range and every other onto the sentinel. Columns stay alive until
         *  @c finish, whose one wait covers every copy. A sharded plan drives it one part's slice
         *  at a time, its step naming the rows in that part's row and ID spaces.
         *
         *  @param[in] step The step to fill; its source nodes address @p columns.
         *  @param[in] columns The rows' columns, one part's under a sharded plan.
         *  @param[in] source_range The ID space those rows speak.
         *  @param[in] key_of Row @c r of the step's source nodes takes the key <tt>key_of(r)</tt>.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The step is empty, @p columns cannot hold its rows, or the
         *    rebased IDs pass the plan.
         *  - @c io_k A read into a staging slot ended short, which never restarts on the host.
         */
        template <typename key_of_type_>
        status_t load(step_t const& step, columns_view_t columns, iota_view<node_id_t> source_range,
                      key_of_type_&& key_of) noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (failed(vendor_t::select(index_->api(), index_->device_specs().ordinal)))
                return status_t::device_error_k;
            iota_view<node_id_t> const range = step.index_nodes;
            if (range.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(node_id_t);
            std::size_t const first = step.source_nodes.first;
            bytes_view_t vectors_rows = flat_rows(columns.vectors, first, range.size(), scalar_row_bytes);
            bytes_view_t neighbors_rows = flat_rows(columns.neighbors, first, range.size(), neighbors_row_bytes);
            bytes_view_t norms_rows = flat_rows(columns.norms, first, range.size(), sizeof(norms_t));
            if (vectors_rows.empty() || neighbors_rows.empty() || norms_rows.empty() != columns.norms.empty())
                return status_t::config_mismatch_k;
            // The part's rows go to this step's own index rows, so an ID's shift follows from
            // the step rather than from the plan's base - the two agree for a one-part plan.
            node_id_t const shift = static_cast<node_id_t>(range.first - step.source_nodes.first + source_range.first);
            // The device indexes its chain heads by neighbor ID unchecked, so every rebased ID must
            // arrive in this plan.
            if (static_cast<std::size_t>(shift) + source_range.size() > static_cast<std::size_t>(base) + rows)
                return status_t::config_mismatch_k;
            // Dot reserves no norms column, so a part's column has nowhere to go.
            bool const normed = index_->stores_norms() && !norms_rows.empty();
            std::size_t const slot = chunks_++ % 2;
            // A slot refills only once the copies out of it finished, while the other's run.
            if (reader_ && chunks_ > 2 && failed(drained_[slot].wait())) return index_->fail_(status_t::device_error_k);
            bytes_view_t* const extents[] = {&vectors_rows, &neighbors_rows, &norms_rows};
            if (status_t const read = reader_.read(slot, std::span(extents).first(normed ? 3 : 2)); failed(read))
                return index_->fail_(read);
            if (failed(vendor_t::copy_2d(index_->api(), index_->vectors_rows_() + range.first * index_->row_bytes(),
                                         index_->row_bytes(), vectors_rows.data(), scalar_row_bytes, scalar_row_bytes,
                                         range.size(), stream)) ||
                (normed &&
                 failed(vendor_t::copy(index_->api(), index_->norms_column_.base() + range.first * sizeof(norms_t),
                                       norms_rows.data(), norms_rows.size(), stream))) ||
                failed(vendor_t::copy(index_->api(),
                                      index_->neighbors_matrix_.base() + range.first * neighbors_row_bytes,
                                      neighbors_rows.data(), neighbors_rows.size(), stream)) ||
                failed(index_->launch_rebase_links_(*kernels_, range, source_range, shift, stream)) ||
                failed(drained_[slot].record(stream)))
                return index_->fail_(status_t::device_error_k);
            for (node_id_t node = range.first; node != range.bound; ++node)
                index_->keyed_.assign(node, key_of(static_cast<node_id_t>(first + node - range.first)));
            if (normed)
                fu::standard_atomic_ref<std::size_t> {covered_norms}.add(range.size(), std::memory_order_acq_rel);
            fu::standard_atomic_ref<std::size_t> {covered}.add(range.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Files every appended row under the key its step copied, which @ref finish then uploads
         *  and makes visible - the one writer of the key map, once every step finished and before
         *  the finish; with a step missing it files nothing. */
        void publish_keys() noexcept {
            if (!index_ || fu::standard_atomic_ref<std::size_t> {covered}.load(std::memory_order_acquire) != rows)
                return;
            index_->publish_keys_({base, static_cast<node_id_t>(base + rows)});
        }

        /**
         *  @brief The same copy out of @p source_shard 's own storage rather than host columns, as
         *      an absorb drives it, one donor shard at a time, its step naming the rows in that
         *      shard's row and ID spaces.
         *
         *  Both matrices keep fragment-padded rows of one geometry, so every column moves as one
         *  copy on the plan's stream, from another device's storage as well once its sharded index
         *  mapped it for this one; storage this device cannot reach is refused. The neighbors then
         *  rebase on the plan's stream, and the donor's keys go into the host key column. Nothing
         *  waits, so @p source_shard must stay quiescent until the caller drains that stream.
         *
         *  @param[in] step The step to fill; its source nodes address @p source_shard.
         *  @param[in] source_shard The shard the rows come from, read through its core.
         *  @param[in] source_range The ID space the source's rows speak.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k Another geometry, rows outside the plan, or storage on a
         *    device this one cannot reach.
         */
        template <typename shard_type_>
        status_t load(step_t const& step, shard_type_ const& source_shard, iota_view<node_id_t> source_range) noexcept {
            vram_shard_core const& source = source_shard.core();
            if (!index_) return status_t::uninitialized_k;
            int const device = index_->device_specs().ordinal;
            if (failed(vendor_t::select(index_->api(), device))) return status_t::device_error_k;
            iota_view<node_id_t> const range = step.index_nodes;
            bool const normed = index_->stores_norms() && source.stores_norms();
            if (range.empty() || range.size() != step.source_nodes.size() || range.first < base ||
                range.bound > base + rows || step.source_nodes.bound > source.size() ||
                source.dimensions() != index_->dimensions() || source.connectivity() != index_->connectivity() ||
                !source.vectors_matrix_.shared_with(device) || !source.neighbors_matrix_.shared_with(device) ||
                (normed && !source.norms_column_.shared_with(device)))
                return status_t::config_mismatch_k;
            node_id_t const shift = range.first - step.source_nodes.first + source_range.first;
            if (shift + source_range.size() > base + rows) return status_t::config_mismatch_k;
            std::size_t const row_bytes = index_->row_bytes();
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(node_id_t);
            std::size_t const count = range.size(), into = range.first, from = step.source_nodes.first;
            if (failed(vendor_t::copy(index_->api(), index_->vectors_rows_() + into * row_bytes,
                                      source.vectors_rows_() + from * row_bytes, count * row_bytes, stream)) ||
                failed(vendor_t::copy(index_->api(), index_->neighbors_matrix_.base() + into * neighbors_row_bytes,
                                      source.neighbors_matrix_.base() + from * neighbors_row_bytes,
                                      count * neighbors_row_bytes, stream)) ||
                (normed && failed(vendor_t::copy(index_->api(), index_->norms_column_.base() + into * sizeof(norms_t),
                                                 source.norms_column_.base() + from * sizeof(norms_t),
                                                 count * sizeof(norms_t), stream))))
                return index_->fail_(status_t::device_error_k);
            if (failed(index_->launch_rebase_links_(*kernels_, range, source_range, shift, stream)))
                return index_->fail_(status_t::device_error_k);
            for (std::size_t row = 0; row != count; ++row)
                index_->keyed_.assign(static_cast<node_id_t>(into + row),
                                      source.keyed_[static_cast<node_id_t>(from + row)]);
            if (normed) fu::standard_atomic_ref<std::size_t> {covered_norms}.add(count, std::memory_order_acq_rel);
            fu::standard_atomic_ref<std::size_t> {covered}.add(count, std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /**
         *  @brief Publishes the appended nodes once every batch's vectors and neighbors arrived.
         *
         *  The norms derive on the device unless the metric is dot or every batch carried them, the
         *  row count goes beside the keys, uploaded on the plan's stream, the claim lifts, and the
         *  watermark turns over the rows once every earlier batch has, carrying along the keys that
         *  @ref publish_keys filed. A success spends the plan, so @c uninitialized_k answers a
         *  second call instead of publishing twice, and also a turn refused by a failure an earlier
         *  batch kept; any other refusal leaves it whole, so the missing steps can still arrive.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent, or @ref status keeps it.
         *  - @c config_mismatch_k Some step's rows never arrived, or @ref status keeps it.
         *  - @c cancelled_k, @c unsupported_k, @c unsupported_layout_k or @c overflow_k The index
         *    keeps it, as @ref status says.
         */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (failed(vendor_t::select(index_->api(), index_->device_specs().ordinal)))
                return status_t::device_error_k;
            if (fu::standard_atomic_ref<std::size_t> {covered}.load(std::memory_order_acquire) != rows)
                return status_t::config_mismatch_k;
            if (rows) {
                if (failed(index_->upload_keys_({base, base + rows}, stream)))
                    return index_->fail_(status_t::device_error_k);
                // One launch derives every row, so a load mixing carried and missing norms
                // re-derives the carried.
                if (index_->stores_norms() &&
                    fu::standard_atomic_ref<std::size_t> {covered_norms}.load(std::memory_order_acquire) != rows &&
                    failed(index_->launch_derive_norms_(*kernels_, base, rows, stream)))
                    return index_->fail_(status_t::device_error_k);
                // A fill carries the count by value, so no host word must outlive a copy; the one
                // wait finishes every step, the norms and the count before the watermark turns.
                if (failed(vendor_t::fill_words(index_->api(), index_->published_rows_(),
                                                static_cast<u32_t>(base + rows), 1, stream)) ||
                    failed(drained_[0].record(stream)) || failed(drained_[0].wait()))
                    return index_->fail_(status_t::device_error_k);
            }
            vram_shard_core* const index = std::exchange(index_, nullptr);
            return rows ? index->publish_(base, rows) : status_t::success_k;
        }

      private:
        friend struct vram_shard_core;

        /** Marks the last copies out of each staging slot, born with the plan. */
        event_t drained_[2];

        /** Reads steps of a registered part into its staging slots; dies after the destructor's
         *  wait, so no copy still reads a slot it frees. */
        typename kernels_type_::part_reader_t reader_;

        /** Steps queued so far, alternating between the slots, whose sources a dying plan
         *  outwaits. */
        std::size_t chunks_ = 0;

        /** The index appended to; null in the empty plan a refused @ref plan_load returns, and once
         *  @c finish published. */
        vram_shard_core* index_ = nullptr;

        /** The kernels the plan's stream loaded for the index, which outlive the plan there; null
         *  in the empty plan. */
        kernel_table_t const* kernels_ = nullptr;
    };

    /**
     *  @brief Opens a load of @p rows at the watermark on @p stream: claims them, or places them
     *      under the claim word its caller closed, and grows the storage to hold them - the fresh
     *      neighbor rows arrive sentinel-filled.
     *
     *  Its steps source the rows from host or device columns, as a sharded plan drives its parts,
     *  or from other indexes, as an absorb does; a plan reading a part the vendor's reader
     *  registers opens its staging slots first. Refuses an unreserved index, a kept failure, a
     *  batch or load in flight, and rows past the ID space.
     *
     *  @param[in] rows Rows to append.
     *  @param[in] stream The stream every copy and the norms kernel queue on.
     *  @param[in] sources The mapped parts host columns are read from, empty for a lent buffer.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew, or @ref status keeps it.
     *  - @c overflow_k The rows pass the ceiling or the reserved range, or @ref status keeps it.
     *  - @c contended_k A batch or load is in flight.
     *  - @c cancelled_k, @c config_mismatch_k, @c unsupported_k or @c unsupported_layout_k
     *    The index keeps it, as @ref status says.
     */
    expected<load_plan_t> plan_load(std::size_t rows, typename executor_t::stream_t const& stream,
                                    std::span<typename kernels_type_::part_source_t const> sources = {}) noexcept {
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (failed(vendor_t::select(api(), device_.ordinal))) return {{}, status_t::device_error_k};
        auto const [kernels, kernels_status] = kernels_.kernels_(stream);
        if (failed(kernels_status)) return {{}, kernels_status};
        if (!rows)
            return {load_plan_t {*this, stream, *kernels, static_cast<node_id_t>(claimed_()), 0}, status_t::success_k};
        // A closed word is the caller's, who refines or absorbs around the load; an open one is
        // claimed like a batch's, so no batch goes beside it.
        bool const claiming = !claims_.closed();
        node_id_t base = static_cast<node_id_t>(size());
        if (claiming) {
            auto const [first_id, claim_status] = claim_(rows);
            if (failed(claim_status)) return {{}, claim_status};
            base = first_id;
        }
        else if (rows > ceiling_nodes_ - base) return {{}, status_t::overflow_k};
        load_plan_t plan {*this, stream, *kernels, base, rows};
        if (status_t const grown = grow_nodes_(base + rows, stream.handle); failed(grown)) return {{}, grown};
        std::size_t const column_row_bytes[] = {packed_row_bytes(), connectivity() * sizeof(node_id_t),
                                                stores_norms() ? sizeof(norms_t) : 0};
        if (status_t const opened = plan.reader_.open(stream, sources, plan.batch_rows(), plan.steps().size(),
                                                      column_row_bytes);
            failed(opened))
            return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  Appends the whole source on the calling thread - the one-shot over @ref plan_load for a
     *  caller with no streams to fan the steps out over.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew, or @ref status keeps it.
     *  - @c config_mismatch_k The @p columns or @p keys are shorter than @p rows, refused before a
     *    row is claimed, or @ref status keeps it.
     *  - @c overflow_k The rows pass the ceiling or the reserved range, or @ref status keeps it.
     *  - @c contended_k A batch or load is in flight.
     *  - @c cancelled_k, @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref
     *    status says.
     */
    status_t load(columns_view_t columns, keys_view_t keys, std::size_t rows, iota_view<node_id_t> source_range,
                  typename executor_t::stream_t const& stream) noexcept {
        if (keys.size() < rows || columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return status_t::config_mismatch_k;
        auto [plan, status] = plan_load(rows, stream);
        if (failed(status)) return status;
        for (typename load_plan_t::step_t const step : plan.steps())
            if (status_t const loaded = plan.load(step, columns, source_range,
                                                  [&](node_id_t row) noexcept { return keys[row]; });
                failed(loaded))
                return loaded;
        plan.publish_keys();
        return plan.finish();
    }

#pragma endregion Checkpoints

#pragma region Insertion

    /** The submission half of @c insert: grows, writes @p keys to the host column and uploads them,
     *  then stages and launches @p new_rows at @p first_id, all on the context's stream, and
     *  returns with their IDs, the rows invisible until the caller drains and publishes them. The
     *  caller holds the claim over them, and gives it back on a refusal, since nothing launched.
     *  Back-to-back submissions on one context are serialized by its stream and add into its
     *  telemetry, which the caller zeroes; across devices they overlap freely. */
    expected<insert_result_t> submit_insert_(keys_view_t keys, rows_view_t new_rows, node_id_t first_id,
                                             node_ids_span_t assigned_ids, traversal_context_t& context,
                                             insert_config_t config) noexcept {
        std::size_t const rows = new_rows.rows;
        if (status_t const refusal = rows_refusal_(new_rows); failed(refusal)) return {{}, refusal};
        if (assigned_ids.size() && assigned_ids.size() != rows) return {{}, status_t::config_mismatch_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        stream_handle_t const stream = context.stream();
        // The first growth binds the device every size below reads. The claim admits no other
        // grower, and a rejected grid publishes nothing, so the caller un-claims it exactly.
        std::size_t const total_nodes = first_id + rows, picks = rows * connectivity();
        if (status_t const grown = grow_nodes_(total_nodes, stream); failed(grown)) return {{}, grown};
        // The keys ride the launch's stream ahead of the rows they name.
        for (std::size_t row = 0; row != rows; ++row) keyed_.assign(first_id + static_cast<node_id_t>(row), keys[row]);
        if (status_t const uploaded = upload_keys_({first_id, total_nodes}, stream); failed(uploaded))
            return {{}, uploaded};
        std::size_t const import_bytes = import_bytes_(rows, new_rows.dtype);
        // Past the rows and their import window: the chain heads over every row, then the batch's
        // pick links, the touched nodes and their count.
        std::size_t const links_offset = round_up_to_multiple<std::size_t>(rows * row_bytes() + import_bytes, 16);
        std::size_t const links_bytes = (total_nodes + 2 * picks + 1) * sizeof(u32_t);
        byte_t* const staging = context.ensure_staging(links_offset + links_bytes);
        if (!staging) return {{}, status_t::bad_alloc_k};
        u32_t* const link_heads = reinterpret_cast<u32_t*>(staging + links_offset);
        if (failed(vendor_t::fill(api(), link_heads, 0xFF, total_nodes * sizeof(u32_t), stream)))
            return {{}, status_t::device_error_k};
        if ((import_bytes || packed_row_bytes() != row_bytes()) &&
            failed(vendor_t::fill(api(), staging, 0, rows * row_bytes() + import_bytes, stream)))
            return {{}, status_t::device_error_k};
        auto const [kernels, kernels_status] = kernels_.kernels_(context);
        if (failed(kernels_status)) return {{}, kernels_status};
        if (status_t const imported = import_rows_(*kernels, staging, new_rows, staging + rows * row_bytes(), stream);
            failed(imported))
            return {{}, imported};

        // Every inserting warp owns an arena slice for its exact visited set.
        u32_t const expansion = static_cast<u32_t>(vram_insert_expansion(config.expansion_factor));
        u32_t const candidate_slots = candidate_slots_for_(expansion, 0);
        u32_t const selection_nodes = walk_selection_nodes_(expansion, candidate_slots);
        gpu_arena_shape_t const arena_shape = build_arena_shape_(expansion, candidate_slots, 1);
        std::size_t const arena_warps = device_.multiprocessor_count *
                                        kernels_type_::resident_blocks_(vram_kernel_t::insert_k, candidate_slots) *
                                        vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes<node_id_t>());
        if (!arena) return {{}, status_t::bad_alloc_k};

        // Monotone, so an un-claim below only forfeits epoch space.
        auto const [epoch_base, epochs_status] = context.take_epochs(rows, arena_bytes_before);
        if (failed(epochs_status)) return {{}, epochs_status};
        vram_build_t const build {storage(),
                                  staging,
                                  static_cast<u32_t>(total_nodes),
                                  arena_shape.scratch_bytes ? 0u : build_scratch_bytes_per_warp(selection_nodes),
                                  selection_nodes,
                                  first_id,
                                  arena,
                                  arena_shape,
                                  expansion,
                                  epoch_base,
                                  published_rows_(),
                                  context.telemetry_.get(),
                                  context.assigned_nodes_.get(),
                                  link_heads,
                                  link_heads + total_nodes,
                                  link_heads + total_nodes + picks,
                                  link_heads + total_nodes + 2 * picks};
        // The launch is the last fallible step; reading its verdict here keeps the claim
        // reversible.
        if (failed(kernels_.launch_insert_(*this, build, candidate_slots, context)))
            return {{}, status_t::device_error_k};

        insert_result_t result;
        result.first_id = first_id;
        for (std::size_t row = 0; row != assigned_ids.size(); ++row)
            assigned_ids[row] = static_cast<node_id_t>(first_id + row);
        result.new_size = total_nodes;
        return {result, status_t::success_k};
    }

#pragma endregion Insertion

#pragma region Search

    using attend_heads_t = vram_attend_heads_t;

    /** Whether the device casts carry @p from values into @p to, as NumKong's plan of one value
     *  says. */
    static bool casts_(nk_dtype_t from, nk_dtype_t to) noexcept {
        nk_cast_plan_t plan;
        return nk_cast_plan_serial_(from, to, 1, 0, &plan) != 0;
    }

    /** Why @p rows cannot cross into the staging: a width other than the index's, a scalar the
     *  device casts cannot carry into the stored one, or rows closer than one packed row apart, a
     *  zero stride repeating the first. */
    status_t rows_refusal_(rows_view_t rows) const noexcept {
        if (rows.columns != dimensions()) return status_t::config_mismatch_k;
        if (rows.dtype != vector_dtype_ && !casts_(rows.dtype, vector_dtype_)) return status_t::unsupported_k;
        if (rows.stride && rows.stride < packed_bytes(rows.dtype, rows.columns)) return status_t::unsupported_layout_k;
        return status_t::success_k;
    }

    /** Values one staged row holds at the vectors matrix pitch, its pad included. */
    std::size_t pitch_values_() const noexcept { return row_bytes() * 8 / nk_dtype_bits(vector_dtype_); }

    /** Whether rows in @p dtype narrow into the stored scalar at unit RMS rather than cast. */
    bool narrows_at_unit_rms_(nk_dtype_t dtype) const noexcept {
        return unit_rms_scalar_k && kernel_ == nk_kernel_angular_k && (dtype == nk_f32_k || dtype == nk_f64_k);
    }

    /** Bytes of the window @p rows rows in @p dtype import through, in whole fragments: up to
     *  @c import_window_rows_k of them at the stored pitch, then the f32 rows f64 ones widen
     *  through on their way to unit RMS; none for rows already in the stored scalar. */
    std::size_t import_bytes_(std::size_t rows, nk_dtype_t dtype) const noexcept {
        if (dtype == vector_dtype_ || !rows) return 0;
        std::size_t const widened_bits = dtype == nk_f64_k && narrows_at_unit_rms_(dtype) ? 32 : 0;
        return round_up_to_multiple<std::size_t>(
            min_of(rows, import_window_rows_k) * pitch_values_() * (nk_dtype_bits(dtype) + widened_bits) / 8,
            fragment_bytes_k);
    }

    /** Imports @p rows into @p staging at the vectors matrix pitch in the stored scalar on
     *  @p stream, over @p staging and @p window already zeroed, casting through @p kernels. Rows in
     *  the stored scalar take one pitched copy. Others cross a window at a time: a pitched copy
     *  pads them to the stored pitch, and one flat cast writes the window in place, its zeroed pad
     *  values casting to zero pad bytes, or a unit-RMS narrowing writes the columns alone.
     *  Repeated rows import once and double on the device. */
    status_t import_rows_(kernel_table_t const& kernels, byte_t* staging, rows_view_t rows, byte_t* window,
                          stream_handle_t stream) const noexcept {
        std::size_t const distinct = rows.stride ? rows.rows : 1;
        std::size_t const source_row_bytes = packed_bytes(rows.dtype, rows.columns);
        std::size_t const source_pitch = rows.stride ? rows.stride : source_row_bytes;
        if (rows.dtype == vector_dtype_) {
            if (failed(vendor_t::copy_2d(api(), staging, row_bytes(), rows.data, source_pitch, source_row_bytes,
                                         distinct, stream)))
                return status_t::device_error_k;
        }
        else {
            std::size_t const pitch_values = pitch_values_();
            std::size_t const window_rows = min_of(distinct, import_window_rows_k);
            std::size_t const window_row_bytes = pitch_values * nk_dtype_bits(rows.dtype) / 8;
            f32_t* const widened = reinterpret_cast<f32_t*>(window + window_rows * window_row_bytes);
            for (std::size_t first = 0; first < distinct; first += window_rows) {
                std::size_t const count = min_of(window_rows, distinct - first);
                byte_t* const window_staging = staging + first * row_bytes();
                if (failed(vendor_t::copy_2d(api(), window, window_row_bytes, rows.row(first), source_pitch,
                                             source_row_bytes, count, stream)))
                    return status_t::device_error_k;
                status_t cast = status_t::success_k;
                if (!narrows_at_unit_rms_(rows.dtype))
                    cast = kernels_.launch_cast_(kernels, *this, window, rows.dtype, window_staging, vector_dtype_,
                                                 count * pitch_values, stream);
                else if (rows.dtype == nk_f32_k)
                    cast = launch_rmscast_(kernels, reinterpret_cast<f32_t const*>(window), window_row_bytes,
                                           window_staging, count, stream);
                else {
                    cast = kernels_.launch_cast_(kernels, *this, window, nk_f64_k, widened, nk_f32_k,
                                                 count * pitch_values, stream);
                    if (succeeded(cast))
                        cast = launch_rmscast_(kernels, widened, pitch_values * sizeof(f32_t), window_staging, count,
                                               stream);
                }
                if (failed(cast)) return cast;
            }
        }
        for (std::size_t copied = distinct; copied < rows.rows; copied *= 2)
            if (failed(vendor_t::copy(api(), staging + copied * row_bytes(), staging,
                                      min_of(copied, rows.rows - copied) * row_bytes(), stream)))
                return status_t::device_error_k;
        return status_t::success_k;
    }

    /** Why an attend over @p rows rows cannot be written to @p outputs and @p carry: shapes other
     *  than the rows', a scalar the device casts cannot carry f32 into, or rows closer than one row
     *  apart. */
    status_t attend_refusal_(std::size_t rows, rows_span_t outputs,
                             std::span<softmax_state_t const> carry) const noexcept {
        if (outputs.rows != rows || outputs.columns != dimensions() || (!carry.empty() && carry.size() != rows))
            return status_t::config_mismatch_k;
        if (outputs.dtype != nk_f32_k && !casts_(nk_f32_k, outputs.dtype)) return status_t::unsupported_k;
        if (outputs.stride < packed_bytes(outputs.dtype, dimensions())) return status_t::unsupported_layout_k;
        return status_t::success_k;
    }

    /** Bytes the attended rows take past the f32 blocks once cast into the scalar of @p outputs;
     *  none when they stay f32. */
    std::size_t exported_bytes_(rows_span_t outputs) const noexcept {
        return outputs.dtype == nk_f32_k ? 0 : outputs.rows * packed_bytes(outputs.dtype, dimensions());
    }

    /** Queues the copy of the attended rows of the first of @p blocks blocks at @p attended into
     *  @p outputs on @p stream, cast through @p kernels past the blocks first unless they stay
     *  f32, then of the carries past the rows into @p carry unless it is empty. */
    status_t copy_attended_(kernel_table_t const& kernels, f32_t* attended, std::size_t blocks, rows_span_t outputs,
                            std::span<softmax_state_t> carry, stream_handle_t stream) const noexcept {
        std::size_t const values = outputs.rows * dimensions();
        std::size_t const exported_row_bytes = packed_bytes(outputs.dtype, dimensions());
        byte_t* exported = reinterpret_cast<byte_t*>(attended);
        if (outputs.dtype != nk_f32_k) {
            exported += blocks * (values * sizeof(f32_t) + outputs.rows * sizeof(softmax_state_t));
            if (status_t const cast = kernels_.launch_cast_(kernels, *this, attended, nk_f32_k, exported, outputs.dtype,
                                                            values, stream);
                failed(cast))
                return cast;
        }
        if (failed(vendor_t::copy_2d(api(), outputs.data, outputs.stride, exported, exported_row_bytes,
                                     exported_row_bytes, outputs.rows, stream)) ||
            (!carry.empty() &&
             failed(vendor_t::copy(api(), carry.data(), attended + values, carry.size_bytes(), stream))))
            return status_t::device_error_k;
        return status_t::success_k;
    }

    /** Blocking drains the context's stream and adds the telemetry the walks spent; deferred
     *  records the submission for @c context.finished and returns it as queued. */
    expected<search_result_t> settle_(expected<search_result_t> submitted, std::size_t rows,
                                      traversal_context_t& context, completion_t completion) noexcept {
        if (!submitted || !rows) return submitted;
        if (completion == completion_t::deferred_k) {
            if (status_t const recorded = context.record(); failed(recorded)) return {{}, recorded};
            return submitted;
        }
        if (status_t const drained = context.wait(); failed(drained)) return {{}, drained};
        vram_telemetry_block_t const spent = context.spent();
        submitted.value.visited_members = spent.hops;
        submitted.value.computed_distances = spent.scored;
        submitted.value.restarts = spent.restarts;
        submitted.value.tail_compactions = spent.tail_compactions;
        submitted.value.tail_dropped = spent.tail_dropped;
        submitted.value.visited_saturations = spent.visited_saturations;
        return submitted;
    }

    /** The single-query @c search over the @p visible rows, which a sharded caller snapshots across
     *  its shards. */
    expected<search_result_t> search_(vector_view_t query_vector, traversal_context_t& context, search_config_t config,
                                      std::size_t visible) noexcept {
        context.results.count_ = 0;
        if (!visible) return {{}, status_t::success_k}; // An empty index matches nothing, touches nothing.
        // Up to the expansion factor, defaulting to twice the connectivity, clipped to the graph.
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const count = min_of<u32_t>(expansion, visible);
        if (failed(context.results.entries_.reserve(std::size_t {count} * 2))) return {{}, status_t::bad_alloc_k};
        context.results.entries_.extend_uninitialized(std::size_t {count} * 2);
        node_id_t* const staged_ids = reinterpret_cast<node_id_t*>(context.results.entries_.data() + count);
        distance_t* const staged_distances = reinterpret_cast<distance_t*>(staged_ids + count);
        vectors_view_t const one_query {query_vector.data(), {1, query_vector.size()}};
        config.expansion_factor = expansion;
        config.count = count;
        expected<search_result_t> result = settle_(
            kernels_.submit_search_(*this, one_query, nk::matrix_span<node_id_t> {staged_ids, {1, count}},
                                    nk::matrix_span<distance_t> {staged_distances, {1, count}}, context, config,
                                    visible),
            1, context, completion_t::blocking_k);
        if (!result) return result;
        std::size_t live = 0;
        for (; live != count && staged_ids[live] != node_id_unknown_k; ++live)
            context.results.entries_[live] = scored_node_t {staged_ids[live], staged_distances[live]};
        context.results.count_ = live;
        result.value.count = live;
        return result;
    }

#pragma endregion Search

#pragma region Refinement

    /** Zeroed audit counters in a slot of @p context 's stream, or null when none can pin. */
    static audit_counters_ptr_t take_audit_counters_(traversal_context_t& context) noexcept {
        typename vendor_t::pinned_slots_t* const pinned_slots = context.allocator_.pinned_slots();
        void* const slot = pinned_slots ? pinned_slots->take() : nullptr;
        return {slot ? new (slot) audit_counters_t {} : nullptr, audit_counters_return_t {pinned_slots}};
    }

    /** Which share of the scratch a carve covers: the audit's alone, or the whole of a refine's. */
    enum class scratch_use_t : bool { audit_k, refine_k };

    /** Bytes of the scratch over the rows published now: the audit share, and for @c refine_k the
     *  islands' IDs, @p parts bases, the reached in-links and the changed bits too. */
    std::size_t scratch_bytes_(scratch_use_t use, std::size_t parts = 0) const noexcept {
        std::size_t const rows = size(), words = divide_round_up<std::size_t>(rows, 32);
        std::size_t const refine_words = use == scratch_use_t::refine_k ? rows + parts + rows + words : 0;
        return (words * 5 + 11 + refine_words) * sizeof(u32_t);
    }

    /** Bytes of the staging a sweep tile of @p tile_rows rows carves: its run offset per row and a
     *  row's worth of links on its tape per row. */
    std::size_t refine_staging_bytes_(std::size_t tile_rows) const noexcept {
        return tile_rows * (connectivity() + 1) * sizeof(u32_t);
    }

    /** Bytes of the arena @p walks walks of a row sharing one visited set take under @p config. */
    std::size_t refine_arena_bytes_(refine_config_t config, u32_t walks) const noexcept {
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const candidate_slots = candidate_slots_for_(expansion, 4);
        std::size_t const arena_warps = device_.multiprocessor_count *
                                        kernels_type_::resident_blocks_(vram_kernel_t::refine_k, candidate_slots) *
                                        vram_warps_per_block_k;
        return arena_warps * build_arena_shape_(expansion, candidate_slots, walks).template slice_bytes<node_id_t>();
    }

    /** Rows one sweep tile of a pass re-selecting @p forward_rows rows takes under @p config: as
     *  many as its limit holds past the scratch and the arena, every one without a limit; never
     *  more than the tape's offsets count, and zero when none fits. */
    std::size_t refine_tile_rows_(std::size_t forward_rows, refine_config_t config, u32_t walks) const noexcept {
        std::size_t const row_bytes = refine_staging_bytes_(1);
        std::size_t const capped = min_of<std::size_t>(forward_rows, node_id_unknown_k / connectivity());
        if (!config.memory_limit) return capped;
        std::size_t const carried = scratch_bytes_(scratch_use_t::refine_k) + refine_arena_bytes_(config, walks);
        return config.memory_limit > carried ? min_of((config.memory_limit - carried) / row_bytes, capped) : 0;
    }

    /** Carves the scratch over the rows published now, for @c refine_k the whole of a refine's
     *  with room for @p parts bases, otherwise the audit share alone, the reconnected counter
     *  zeroed. */
    expected<refine_scratch_t> carve_scratch_(traversal_context_t& context, scratch_use_t use,
                                              std::size_t parts = 0) noexcept {
        using bytes_ptr_t = typename traversal_context_t::bytes_ptr_t;
        refine_scratch_t scratch {bytes_ptr_t {nullptr, typename bytes_ptr_t::deleter_type {context.allocator_, 0}}};
        if (!vectors_matrix_.reserved_bytes()) return {std::move(scratch), status_t::uninitialized_k};
        if (context.device() != device_.ordinal) return {std::move(scratch), status_t::config_mismatch_k};
        std::size_t const rows = size(), bytes = scratch_bytes_(use, parts);
        scratch.bytes.reset(context.allocator_.allocate(bytes));
        scratch.bytes.get_deleter().count = bytes;
        if (!scratch.bytes) return {std::move(scratch), status_t::bad_alloc_k};
        scratch.words = divide_round_up<std::size_t>(rows, 32);
        scratch.reached.words = reinterpret_cast<u32_t*>(scratch.bytes.get());
        scratch.frontier.words = scratch.reached.words + scratch.words;
        scratch.next_frontier.words = scratch.frontier.words + scratch.words;
        scratch.reaches_back.words = scratch.next_frontier.words + scratch.words;
        scratch.back_settled.words = scratch.reaches_back.words + scratch.words;
        scratch.levels = scratch.back_settled.words + scratch.words;
        scratch.expanded = scratch.levels + 1;
        scratch.back_levels = scratch.expanded + 2;
        scratch.unreached = scratch.back_levels + 1;
        scratch.islands = scratch.unreached + 1;
        scratch.reconnected = scratch.islands + 1;
        scratch.claims = scratch.reconnected + 1;
        scratch.swept = scratch.claims + 2;
        if (use == scratch_use_t::refine_k) {
            scratch.island_ids = scratch.swept + 2;
            scratch.part_bases = scratch.island_ids + rows;
            scratch.in_links = scratch.part_bases + parts;
            scratch.changed.words = scratch.in_links + rows;
        }
        if (failed(vendor_t::fill(api(), scratch.reconnected, 0, sizeof(u32_t), context.stream())))
            return {std::move(scratch), status_t::device_error_k};
        if (use == scratch_use_t::refine_k &&
            failed(vendor_t::fill(api(), scratch.changed.words, 0, scratch.words * sizeof(u32_t), context.stream())))
            return {std::move(scratch), status_t::device_error_k};
        return {std::move(scratch), status_t::success_k};
    }

    /** The launch record of a sweep's tile of @p targets over the links of @p sources: its run
     *  offsets lead the staging the plan grew for @p tile_rows rows, and the tape follows them,
     *  holding @p tape_capacity links. */
    vram_sweep_t sweep_record_(traversal_context_t& context, refine_scratch_t const& scratch,
                               iota_view<node_id_t> sources, iota_view<node_id_t> targets, std::size_t tile_rows,
                               std::size_t tape_capacity) const noexcept {
        u32_t* const offsets = reinterpret_cast<u32_t*>(context.staging_.get());
        return {neighbor_rows_(),
                static_cast<u32_t>(connectivity()),
                static_cast<u32_t>(size()),
                sources.first,
                sources.bound,
                targets.first,
                targets.bound,
                offsets,
                offsets + tile_rows,
                static_cast<u32_t>(tape_capacity),
                scratch.swept};
    }

    /** Queues the count of every link @p sources hold into the tile of @p targets, none counted
     *  yet, then the offset kernel, which keeps as many rows as a tile's tape of @p tile_rows rows
     *  of links holds. */
    status_t queue_count_back_links_(traversal_context_t& context, refine_scratch_t const& scratch,
                                     iota_view<node_id_t> sources, iota_view<node_id_t> targets,
                                     std::size_t tile_rows) noexcept {
        auto const [kernels, kernels_status] = kernels_.kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        stream_handle_t const stream = context.stream();
        vram_arguments_t arguments {};
        arguments.sweep = sweep_record_(context, scratch, sources, targets, tile_rows, tile_rows * connectivity());
        if (failed(vendor_t::fill(api(), arguments.sweep.incoming_offsets, 0, targets.size() * sizeof(u32_t), stream)))
            return status_t::device_error_k;
        u32_t const blocks = device_.multiprocessor_count *
                             kernels_type_::resident_blocks_(vram_kernel_t::count_back_links_k, 0);
        if (status_t const counted = kernels_.launch_(*kernels, vram_kernel_t::count_back_links_k, 0, blocks, 0, stream,
                                                      arguments);
            failed(counted))
            return counted;
        return kernels_.launch_(*kernels, vram_kernel_t::offset_back_links_k, 0, 1, 0, stream, arguments);
    }

    /** Queues the scatter of every link @p sources hold into the tile of @p targets onto the first
     *  @p tape_size slots of its tape. */
    status_t queue_scatter_back_links_(traversal_context_t& context, refine_scratch_t const& scratch,
                                       iota_view<node_id_t> sources, iota_view<node_id_t> targets,
                                       std::size_t tile_rows, u32_t tape_size) noexcept {
        auto const [kernels, kernels_status] = kernels_.kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        vram_arguments_t arguments {};
        arguments.sweep = sweep_record_(context, scratch, sources, targets, tile_rows, tape_size);
        u32_t const blocks = device_.multiprocessor_count *
                             kernels_type_::resident_blocks_(vram_kernel_t::scatter_back_links_k, 0);
        return kernels_.launch_(*kernels, vram_kernel_t::scatter_back_links_k, 0, blocks, 0, context.stream(),
                                arguments);
    }

    /** Queues the merge of each row of the tile of @p targets with its run of the @p tape_size
     *  links on its tape, over the grid and the arena of the walks of @p sources, and counts a row
     *  of @p sources that gained a member as changed. */
    status_t queue_merge_back_links_(traversal_context_t& context, refine_scratch_t const& scratch,
                                     refine_config_t config, iota_view<node_id_t> sources, iota_view<node_id_t> targets,
                                     std::size_t tile_rows, u32_t tape_size) noexcept {
        auto const [kernels, kernels_status] = kernels_.kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        stream_handle_t const stream = context.stream();
        // The merge carves its slices from the arena sized for the grid of the refine it launches.
        static_assert(kernels_type_::resident_blocks_(vram_dense_kernel_t::merge_back_links_k, 0) ==
                      kernels_type_::resident_blocks_(vram_kernel_t::refine_k, 0));
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const candidate_slots = candidate_slots_for_(expansion, 4);
        gpu_arena_shape_t const arena_shape = build_arena_shape_(expansion, candidate_slots, 1);
        u32_t const blocks = device_.multiprocessor_count *
                             kernels_type_::resident_blocks_(vram_kernel_t::refine_k, candidate_slots);
        byte_t* const arena = context.ensure_arena(refine_arena_bytes_(config, 1));
        if (!arena) return status_t::bad_alloc_k;
        if (failed(vendor_t::fill(api(), scratch.claims + 1, 0, sizeof(u32_t), stream)))
            return status_t::device_error_k;
        vram_sweep_t const sweep = sweep_record_(context, scratch, sources, targets, tile_rows, tape_size);
        u32_t const selection_nodes = reselection_nodes_();
        vram_arguments_t arguments {};
        vram_refine_t& refine = arguments.refine;
        refine.storage = storage();
        refine.end_id = targets.bound;
        refine.first_id = targets.first;
        refine.scratch_bytes_per_warp = arena_shape.scratch_bytes ? 0u : build_scratch_bytes_per_warp(selection_nodes);
        refine.selection_nodes = selection_nodes;
        refine.arena = arena;
        refine.arena_shape = arena_shape;
        refine.expansion = expansion;
        refine.telemetry = context.telemetry_.get();
        refine.assigned_nodes = scratch.claims + 1;
        refine.sampling_bound = static_cast<u32_t>(size());
        refine.reselection_nodes = selection_nodes;
        refine.changed = scratch.changed;
        refine.incoming_offsets = sweep.incoming_offsets;
        refine.back_links = sweep.back_links;
        refine.tape_size = tape_size;
        refine.walked_first = sources.first;
        refine.walked_bound = sources.bound;
        return kernels_.launch_(*kernels, vram_dense_kernel_t::merge_back_links_k, 0, blocks,
                                vram_warps_per_block_k * refine.scratch_bytes_per_warp, stream, arguments);
    }

    /** Queues the copy of @p scratch 's counters into @p counters: the levels, the count of sweep
     *  @p last_sweep, then the levels back, the unreached, island, reconnected, claim and tile
     *  counters, which lie back to back. */
    status_t copy_counters_(refine_scratch_t const& scratch, audit_counters_t* counters, u32_t last_sweep,
                            stream_handle_t stream) noexcept {
        return succeeded(vendor_t::copy(api(), &counters->levels, scratch.levels, sizeof(u32_t), stream)) &&
                       succeeded(vendor_t::copy(api(), &counters->last_expanded, scratch.expanded + (last_sweep & 1u),
                                                sizeof(u32_t), stream)) &&
                       succeeded(vendor_t::copy(api(), &counters->back_levels, scratch.back_levels,
                                                sizeof(audit_counters_t) - 2 * sizeof(u32_t), stream))
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** The converged audit @p counters reports over @p rows rows: the rows reached both ways, and
     *  the levels each direction took. */
    static reachability_t audited_(audit_counters_t const& counters, std::size_t rows) noexcept {
        // A sweep that expands nothing ends the levels, and it counts as one, as on the CPU.
        return {rows, rows - counters.islands, std::size_t {counters.levels} + 1, counters.back_levels};
    }

    /** Takes @p plan 's counters, grows @p context 's arena for the widest of @p walks walks under
     *  @p config and its staging for the passes over the rows published now, re-selecting
     *  @p forward_rows of them, so no pass grows them, and zeroes the telemetry once, so every pass
     *  adds to it. */
    status_t open_refine_(refine_plan_t& plan, traversal_context_t& context, refine_config_t config, u32_t walks,
                          std::size_t forward_rows) noexcept {
        if (!context) return status_t::config_mismatch_k;
        plan.counters_ = take_audit_counters_(context);
        if (!plan.counters_) return status_t::bad_alloc_k;
        if (!context.ensure_arena(refine_arena_bytes_(config, walks))) return status_t::bad_alloc_k;
        plan.tile_rows_ = refine_tile_rows_(forward_rows, config, walks);
        if (forward_rows && !plan.tile_rows_) return status_t::bad_alloc_k;
        if (std::size_t const staging_bytes = refine_staging_bytes_(plan.tile_rows_);
            staging_bytes && !context.ensure_staging(staging_bytes))
            return status_t::bad_alloc_k;
        return failed(
                   vendor_t::fill(api(), context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), context.stream()))
                   ? status_t::device_error_k
                   : status_t::success_k;
    }

    /** Bytes a split of @p nodes nodes into @p parts parts carves on the device and pins on the
     *  host, and where each places the receivers' columns, then where the pinned bytes place the
     *  roots and the placed count, as @c split_t lays them out. */
    struct split_layout_t {
        std::size_t device_bytes = 0, device_columns = 0;
        std::size_t host_bytes = 0, host_columns = 0, host_roots = 0, host_assigned = 0;
    };
    static split_layout_t split_layout_(std::size_t nodes, std::size_t parts) noexcept {
        auto aligned = [](std::size_t bytes) noexcept { return round_up_to_multiple(bytes, default_alignment_k); };
        std::size_t const words = divide_round_up<std::size_t>(nodes, 32);
        split_layout_t bytes;
        bytes.device_columns = aligned((2 * nodes + 3 * words + 3 * parts + 2) * sizeof(u32_t));
        bytes.device_bytes = bytes.device_columns + 4 * parts * sizeof(u32_t*);
        bytes.host_columns = aligned((nodes + words) * sizeof(u32_t));
        bytes.host_roots = bytes.host_columns + 4 * parts * sizeof(u32_t*);
        bytes.host_assigned = bytes.host_roots + parts * sizeof(u32_t);
        bytes.host_bytes = bytes.host_assigned + sizeof(u32_t);
        return bytes;
    }

    /** Where a split's device scratch keeps each array: each node's part, which the scatter reads
     *  back as its slot, the round's proposals, @b [nodes] each, the two frontiers and the picked
     *  bitset, one bit per node each, then per part its size and its round count, the nodes the
     *  last two rounds placed, by the round number's parity, the parts' roots, and the receivers'
     *  neighbors, vectors, norms and keys. */
    struct split_scratch_t {
        u32_t* node_parts = nullptr;
        u32_t* proposals = nullptr;
        vram_bitset_t frontier {}, next_frontier {}, picked {};
        u32_t* part_sizes = nullptr;
        u32_t* round_counts = nullptr;
        u32_t* placed = nullptr;
        u32_t* roots = nullptr;
        u32_t** part_columns = nullptr;
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
        carved.part_columns = reinterpret_cast<u32_t**>(split.device_bytes.get() +
                                                        split_layout_(entries, parts).device_columns);
        return carved;
    }

    /** Carves @p split 's device scratch from @p context 's allocator, borrows its pinned results
     * as the host bytes, and queues the upload of the receivers' columns and of the roots that
     * @p seed picks. */
    status_t carve_split_(split_t& split, u64_t seed, traversal_context_t& context) noexcept {
        static_assert(sizeof(norms_t) % sizeof(u32_t) == 0 && sizeof(key_t) % sizeof(u32_t) == 0);
        std::size_t const entries = split.entries, parts = split.receivers.size();
        split_layout_t const bytes = split_layout_(entries, parts);
        split.device_bytes.reset(context.allocator_.allocate(bytes.device_bytes));
        split.device_bytes.get_deleter().count = bytes.device_bytes;
        if (!split.device_bytes) return status_t::bad_alloc_k;
        growing_buffer<scored_node_t, typename vendor_t::pinned_allocator_t>& pinned = context.results.entries_;
        if (failed(pinned.resize_uninitialized(divide_round_up(bytes.host_bytes, sizeof(scored_node_t)))))
            return status_t::bad_alloc_k;
        split.host_bytes = reinterpret_cast<byte_t*>(pinned.data());
        u32_t** const columns = reinterpret_cast<u32_t**>(split.host_bytes + bytes.host_columns);
        u32_t* const roots = reinterpret_cast<u32_t*>(split.host_bytes + bytes.host_roots);
        for (std::size_t part = 0; part != parts; ++part) {
            vram_shard_core const& receiver = *split.receivers[part];
            columns[part] = receiver.neighbor_rows_();
            columns[parts + part] = reinterpret_cast<u32_t*>(receiver.vectors_rows_());
            columns[2 * parts + part] = reinterpret_cast<u32_t*>(receiver.norms_column_.base());
            columns[3 * parts + part] = reinterpret_cast<u32_t*>(receiver.device_keys_());
            roots[part] = static_cast<u32_t>(seeded_bijection(part, entries, seed));
        }
        split_scratch_t const carved = split_scratch_(split);
        stream_handle_t const stream = context.stream();
        return succeeded(vendor_t::copy(api(), carved.part_columns, columns, 4 * parts * sizeof(u32_t*), stream)) &&
                       succeeded(vendor_t::copy(api(), carved.roots, roots, parts * sizeof(u32_t), stream))
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** The record every launch of @p split reads: this index's columns and the split's scratch. */
    vram_arguments_t split_arguments_(split_t const& split) const noexcept {
        std::size_t const parts = split.receivers.size();
        split_scratch_t const carved = split_scratch_(split);
        vram_arguments_t arguments {};
        vram_graph_t& graph = arguments.graph;
        graph.storage = storage();
        graph.neighbors = neighbor_rows_();
        graph.rows_count = static_cast<u32_t>(split.entries);
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.seed_count = static_cast<u32_t>(min_of(parts, split.entries));
        graph.frontier = carved.frontier;
        graph.next_frontier = carved.next_frontier;
        graph.node_parts = carved.node_parts;
        graph.proposals = carved.proposals;
        graph.part_sizes = carved.part_sizes;
        graph.round_counts = carved.round_counts;
        graph.roots = carved.roots;
        graph.picked = carved.picked;
        graph.parts_count = static_cast<u32_t>(parts);
        graph.part_capacity = static_cast<u32_t>(split.part_capacity);
        graph.part_columns = carved.part_columns;
        graph.keys = reinterpret_cast<u32_t const*>(device_keys_());
        graph.vector_words = static_cast<u32_t>(row_bytes() / sizeof(u32_t));
        graph.norm_words = stores_norms() ? static_cast<u32_t>(sizeof(norms_t) / sizeof(u32_t)) : 0u;
        graph.key_words = static_cast<u32_t>(sizeof(key_t) / sizeof(u32_t));
        return arguments;
    }

    /** Blocks of a split pass striding the rows, as many as stay resident. */
    u32_t split_blocks_(vram_kernel_t kernel) const noexcept {
        return device_.multiprocessor_count * kernels_type_::resident_blocks_(kernel, 0);
    }

    /** Queues the start of @p split 's partition through @p kernels: the node parts and proposals
     *  unknown, the bitsets, the parts' counters and the rounds' counts empty, then the roots. */
    status_t queue_partition_(kernel_table_t const& kernels, split_t& split, stream_handle_t stream) noexcept {
        std::size_t const entries = split.entries, parts = split.receivers.size();
        std::size_t const words = divide_round_up<std::size_t>(entries, 32);
        split_scratch_t const carved = split_scratch_(split);
        // Each run lies back to back.
        if (failed(vendor_t::fill(api(), carved.node_parts, 0xFF, 2 * entries * sizeof(u32_t), stream)) ||
            failed(
                vendor_t::fill(api(), carved.frontier.words, 0, (3 * words + 2 * parts + 2) * sizeof(u32_t), stream)))
            return status_t::device_error_k;
        split.next_round = 1;
        vram_arguments_t const arguments = split_arguments_(split);
        u32_t const block_threads = vram_warps_per_block_k * kernels_type_::wave_lanes();
        return kernels_.launch_(kernels, vram_kernel_t::partition_roots_k, 0,
                                divide_round_up<u32_t>(arguments.graph.seed_count, block_threads), 0, stream,
                                arguments);
    }

    /**
     *  @brief Queues @p split 's growth rounds from its next one up to twice it, so the batches
     *      double until one ends past convergence, then copies the last round's placed count to
     *      the host before returning.
     *
     *  A round queued past convergence finds the one before placed nothing and leaves at entry; odd
     *  rounds grow the first frontier, even ones the second. Every node is placed once at most, so
     *  the nodes bound the rounds.
     */
    status_t queue_rounds_(kernel_table_t const& kernels, split_t& split, stream_handle_t stream) noexcept {
        std::size_t const words = divide_round_up<std::size_t>(split.entries, 32);
        split_scratch_t const carved = split_scratch_(split);
        vram_arguments_t arguments = split_arguments_(split);
        vram_graph_t& graph = arguments.graph;
        u32_t const first_round = split.next_round;
        for (u32_t round = first_round; round != 2 * first_round; ++round) {
            u32_t const odd = round & 1u;
            graph.frontier = odd ? carved.frontier : carved.next_frontier;
            graph.next_frontier = odd ? carved.next_frontier : carved.frontier;
            graph.counter = carved.placed + odd;
            graph.previous_counter = round == 1 ? nullptr : carved.placed + (odd ^ 1u);
            if (failed(vendor_t::fill(api(), graph.next_frontier.words, 0, words * sizeof(u32_t), stream)) ||
                failed(vendor_t::fill(api(), graph.counter, 0, sizeof(u32_t), stream)))
                return status_t::device_error_k;
            for (vram_kernel_t const pass : {vram_kernel_t::partition_propose_k, vram_kernel_t::partition_count_k,
                                             vram_kernel_t::partition_accept_k})
                if (status_t const launched = kernels_.launch_(kernels, pass, 0, split_blocks_(pass), 0, stream,
                                                               arguments);
                    failed(launched))
                    return launched;
            if (status_t const launched = kernels_.launch_(kernels, vram_kernel_t::partition_fill_k, 0,
                                                           graph.parts_count, 0, stream, arguments);
                failed(launched))
                return launched;
        }
        split.next_round = 2 * first_round;
        u32_t* const assigned_count = reinterpret_cast<u32_t*>(
            split.host_bytes + split_layout_(split.entries, split.receivers.size()).host_assigned);
        return failed(vendor_t::copy(api(), assigned_count, carved.placed + ((2 * first_round - 1) & 1u), sizeof(u32_t),
                                     stream))
                   ? status_t::device_error_k
                   : status_t::success_k;
    }

    /** Whether the last growth round of @p split placed anything, so more rounds follow. */
    static bool placing_(split_t const& split) noexcept {
        std::size_t const host_assigned = split_layout_(split.entries, split.receivers.size()).host_assigned;
        return *reinterpret_cast<u32_t const*>(split.host_bytes + host_assigned) != 0;
    }

    /** Queues the end of @p split 's partition once no round places anything: the nodes no round
     *  reached, the picked bitset, and the copy of the node parts and the picked bitset. */
    status_t queue_picked_(kernel_table_t const& kernels, split_t& split, stream_handle_t stream) noexcept {
        std::size_t const entries = split.entries, words = divide_round_up<std::size_t>(entries, 32);
        vram_arguments_t const arguments = split_arguments_(split);
        if (status_t const launched = kernels_.launch_(kernels, vram_kernel_t::partition_islands_k, 0, 1, 0, stream,
                                                       arguments);
            failed(launched))
            return launched;
        if (status_t const launched = kernels_.launch_(kernels, vram_kernel_t::partition_picked_k, 0,
                                                       split_blocks_(vram_kernel_t::partition_picked_k), 0, stream,
                                                       arguments);
            failed(launched))
            return launched;
        split_scratch_t const carved = split_scratch_(split);
        byte_t* const host_bytes = split.host_bytes;
        return succeeded(vendor_t::copy(api(), host_bytes, carved.node_parts, entries * sizeof(u32_t), stream)) &&
                       succeeded(vendor_t::copy(api(), host_bytes + entries * sizeof(u32_t), carved.picked.words,
                                                words * sizeof(u32_t), stream))
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Queues @p split 's scatter: the assigned_nodes slots and the receivers' published counts up,
     *  then every node's neighbors, vector, norms and key into its receiver. */
    status_t queue_scatter_(kernel_table_t const& kernels, split_t const& split, stream_handle_t stream) noexcept {
        if (failed(vendor_t::copy(api(), split_scratch_(split).node_parts, split.host_bytes,
                                  split.entries * sizeof(u32_t), stream)))
            return status_t::device_error_k;
        for (std::size_t index = 0; index != split.receivers.size(); ++index)
            if (failed(vendor_t::fill_words(api(), split.receivers[index]->published_rows_(),
                                            static_cast<u32_t>(split.parts[index].entries), 1, stream)))
                return status_t::device_error_k;
        vram_arguments_t const arguments = split_arguments_(split);
        for (vram_kernel_t const pass : {vram_kernel_t::split_links_k, vram_kernel_t::split_nodes_k})
            if (status_t const launched = kernels_.launch_(kernels, pass, 0, split_blocks_(pass), 0, stream, arguments);
                failed(launched))
                return launched;
        return status_t::success_k;
    }

#pragma endregion Refinement

#pragma region Launches

    /** Blocks a walk over @p queries launches at @p resident_blocks blocks per multiprocessor: a
     *  warp per query up to the grid the device runs at once, so the arena carves a slice only for
     *  a warp that can take a query. */
    u32_t walk_blocks_(std::size_t queries, u32_t resident_blocks) const noexcept {
        return static_cast<u32_t>(min_of<std::size_t>(divide_round_up<std::size_t>(queries, vram_warps_per_block_k),
                                                      std::size_t {device_.multiprocessor_count} * resident_blocks));
    }

    /** Keeps @p failure as the status unless one is kept already; returns it for the caller's own
     *  report. */
    status_t fail_(status_t failure) noexcept {
        status_t expected = status_t::success_k;
        fu::standard_atomic_ref<status_t>(status_).compare_exchange_strong(expected, failure, std::memory_order_acq_rel,
                                                                           std::memory_order_acquire);
        return failure;
    }

    /** Rows claimed by batches and loads, visible or not; a closed word holds none past the
     *  watermark. */
    std::size_t claimed_() const noexcept { return claims_.claimed(size()); }

    /** Claims @p rows from the watermark: one compare-exchange that succeeds only on an open word
     *  with no batch or load in flight, so its holder alone grows, launches and publishes them. */
    expected<node_id_t> claim_(std::size_t rows) noexcept {
        auto const [first_id, claim_status] = claims_.claim(size(), rows, ceiling_nodes_);
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

    /** Forgets every row under the closed claim word: the watermark drops to zero, a kept failure
     *  with it, the key map empties, and the next claim sentinel-fills every mapped row again and
     *  zeroes the published count, so IDs restart from zero on the storage already mapped, which
     *  nothing frees. A walk still reaching the index reads that retained storage, and its caller's
     *  epoch discards it. */
    void forget_() noexcept {
        fu::standard_atomic_ref<std::size_t>(last_visible_id_).store(0, std::memory_order_release);
        fu::standard_atomic_ref<status_t>(status_).store(status_t::success_k, std::memory_order_release);
        keyed_.forget();
        filled_rows_ = 0;
    }

    /** Files every drained row of @p range under its key ahead of its watermark: the claim
     *  holder, or the closer of the word, is the map's one writer. Plain stores and one release
     *  byte per row, no read-modify-write. */
    void publish_keys_(iota_view<node_id_t> range) noexcept { keyed_.publish(range); }

    /** The keys on the device, indexed by node, as a walk resolves them. */
    key_t* device_keys_() const noexcept { return reinterpret_cast<key_t*>(keys_column_.base()); }

    /** Copies @p rows of the host key column into the device keys on @p stream, through the
     *  staging half by half, since a pageable source would copy synchronously; the host refills a
     *  half only once the copy out of it finished. Callers hold the claim over the rows, or the
     *  closed claim word. */
    status_t upload_keys_(iota_view<std::size_t> rows, stream_handle_t stream) noexcept {
        std::size_t const half_keys = keys_staging_.size() / 2;
        if (rows.empty()) return status_t::success_k;
        if (!half_keys) return status_t::uninitialized_k;
        for (std::size_t first = rows.first; first < rows.bound; first += half_keys) {
            std::size_t const bound = min_of(first + half_keys, rows.bound);
            std::size_t const half = staged_halves_++ % 2;
            key_t* const staged = keys_staging_.data() + half * half_keys;
            if (staged_halves_ > 2 && failed(keys_crossed_[half].wait())) return status_t::device_error_k;
            std::size_t written = 0;
            keyed_.column().for_each_span({first, bound}, [&](std::span<key_t const> run) noexcept {
                for (key_t const& key : run) staged[written++] = key;
            });
            if (failed(
                    vendor_t::copy(api(), device_keys_() + first, staged, (bound - first) * sizeof(key_t), stream)) ||
                failed(keys_crossed_[half].record(stream)))
                return status_t::device_error_k;
        }
        return status_t::success_k;
    }

    /** Turns the watermark from @p first over @p rows drained rows once every batch claimed before
     *  has turned it, so the visible rows are always whole batches; a failure kept meanwhile
     *  answers instead, the rows invisible. */
    status_t publish_(std::size_t first, std::size_t rows) noexcept {
        for (std::size_t watermark = size(); watermark != first; watermark = size()) {
            // A batch that failed before this one never turns the watermark, so its kept status
            // ends the wait.
            if (status_t const kept = status(); failed(kept)) return kept;
            wait_t {}(&last_visible_id_, watermark, 0u, fu::wait_capped_t {});
        }
        fu::standard_atomic_ref<std::size_t>(last_visible_id_).store(first + rows, std::memory_order_release);
        return status_t::success_k;
    }

    /** Row zero of the vectors matrix, one row past its base. */
    byte_t* vectors_rows_() const noexcept { return vectors_matrix_.base() + row_bytes(); }

    /** The device-resident published-row count every inserting warp advances: the vectors matrix's
     *  first word, so it lives as long as the storage, and an allocation of its own would map a
     *  whole slab for one word. */
    u32_t* published_rows_() const noexcept { return reinterpret_cast<u32_t*>(vectors_matrix_.base()); }

    /** Row zero of the neighbors matrix, as the IDs its rows hold. */
    node_id_t* neighbor_rows_() const noexcept { return reinterpret_cast<node_id_t*>(neighbors_matrix_.base()); }

    /**
     *  @brief Copies the neighbor rows of the nodes from @p first on, @c connectivity IDs each,
     *      into the host @p rows, raw as stored, sentinels and self-loops left in. The copy rides
     *      no stream, so pageable @p rows are filled when it returns.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p rows hold no whole number of rows, or the rows pass @c size.
     *  - @c device_error_k The device could not be selected, or the copy failed.
     */
    status_t neighbors(node_id_t first, std::span<node_id_t> rows) const noexcept {
        std::size_t const row_ids = connectivity();
        if (!row_ids || rows.size() % row_ids || std::size_t {first} + rows.size() / row_ids > size())
            return status_t::config_mismatch_k;
        if (failed(vendor_t::select(api(), device_.ordinal)) ||
            failed(vendor_t::copy(api(), rows.data(), neighbor_rows_() + std::size_t {first} * row_ids,
                                  rows.size_bytes(), stream_handle_t {})))
            return status_t::device_error_k;
        return status_t::success_k;
    }

    /**
     *  Binds the index at its first growth, ahead of @p rows rows: checks the metric and the
     *  geometry, selects the device and queries its specs, loads the kernels, and reserves every
     *  device column up to the ceiling, @p ceiling clamped to what the device's memory holds.
     *  Later calls answer success untouched. Callers hold the claim or the closed claim word.
     *
     *  @par Statuses:
     *  - @c unknown_kind_k The metric is not one the kits score for this scalar.
     *  - @c config_mismatch_k The neighbors live on another device, the dimensions are zero, the
     *    connectivity is not a power of two, or no walk fits the geometry.
     */
    status_t bind_(std::size_t rows, std::size_t ceiling) noexcept {
        if (vectors_matrix_.reserved_bytes()) return status_t::success_k;
        // The kits' epilogues fall through to the last metric of each family, so no other kind may
        // reach them.
        bool const scored = vector_dtype_ == nk_u1_k
                                ? kernel_ == nk_kernel_dot_k || kernel_ == nk_kernel_hamming_k ||
                                      kernel_ == nk_kernel_jaccard_k
                                : kernel_ == nk_kernel_dot_k || kernel_ == nk_kernel_angular_k ||
                                      kernel_ == nk_kernel_sqeuclidean_k || kernel_ == nk_kernel_euclidean_k;
        if (!scored) return status_t::unknown_kind_k;
        if (neighbors_matrix_.device() != device_.ordinal) return status_t::config_mismatch_k;
        if (!dimensions() || !std::has_single_bit(static_cast<u32_t>(connectivity())))
            return status_t::config_mismatch_k;
        // The device must be current before any mapping, since the storage lives on it.
        if (failed(vendor_t::select(api(), device_.ordinal))) return status_t::device_error_k;
        auto const [specs, status] = device_specs_t::query(api(), device_.ordinal);
        if (failed(status)) return status;
        device_ = specs;
        // Geometry no walk can fit is refused here; a builder whose scratch shared memory cannot
        // fit keeps it in its arena slice.
        if (!kernels_.walk_scratch_fits(*this)) return status_t::config_mismatch_k;
        // Device storage maps at most the device's memory, which bounds the ceiling independent of
        // the first growth; the reservation itself is the ceiling, no cached row count shadows it.
        bool const normed = kernel_ != nk_kernel_dot_k;
        std::size_t const bytes_per_row = row_bytes() + (normed ? sizeof(norms_t) : 0) + sizeof(key_t) +
                                          connectivity() * sizeof(u32_t);
        // Device storage budgets against VRAM; managed storage oversubscribes it and budgets
        // against host memory through their own backend, which keeps the topology symbols out of
        // device instantiations.
        std::size_t budget_bytes = device_.vram_bytes;
        if constexpr (vectors_allocator_t::address_space_k == vendor_t::address_space_t::managed_k)
            budget_bytes = vectors_allocator_t::budget_bytes();
        std::size_t const ceiling_rows = min_of(min_of(ceiling, max_node_count_k),
                                                max_of<std::size_t>(rows, budget_bytes / bytes_per_row));
        // The vectors matrix reserves last, as its range is what marks the index bound; one more
        // vectors row ahead of row zero holds the published-row count.
        if ((normed && norms_column_.reserve_range(ceiling_rows * sizeof(norms_t)) != status_t::success_k) ||
            keys_column_.reserve_range(ceiling_rows * sizeof(key_t)) != status_t::success_k ||
            neighbors_matrix_.reserve_range(ceiling_rows * connectivity() * sizeof(u32_t)) != status_t::success_k ||
            vectors_matrix_.reserve_range((ceiling_rows + 1) * row_bytes()) != status_t::success_k)
            return status_t::bad_alloc_k;
        ceiling_nodes_ = ceiling_rows;
        return status_t::success_k;
    }

    /** Binds the index at its first growth under @p ceiling, then maps every matrix and column up
     *  to @p rows, covers the host keys, and widens the key staging to a power of two of rows, up
     *  to a slab per half - host work alone, so it rides no stream, and leaves the fresh neighbor
     *  rows for @ref grow_nodes_ to fill. Callers hold the claim or the closed claim word. */
    status_t map_nodes_(std::size_t rows, std::size_t ceiling = max_node_count_k) noexcept {
        if (status_t const bound = bind_(rows, ceiling); failed(bound)) return bound;
        if (rows > ceiling_nodes_) return status_t::overflow_k;
        if (status_t const covered = keyed_.cover(rows); failed(covered)) return covered;
        std::size_t const half_keys = min_of(power_of_two<std::size_t>(batch_rows_log2_k),
                                             std::bit_ceil(max_of<std::size_t>(rows, 1)));
        if (keys_staging_.size() < 2 * half_keys) {
            // Pinning needs a current device, and the copies out of the old halves must finish.
            if (failed(vendor_t::select(api(), device_.ordinal)) || failed(keys_crossed_[0].wait()) ||
                failed(keys_crossed_[1].wait()))
                return status_t::device_error_k;
            keys_staging_.reset();
            if (failed(keys_staging_.resize_uninitialized(2 * half_keys))) return status_t::bad_alloc_k;
        }
        if (rows <= capacity()) return status_t::success_k;
        if (status_t status = vectors_matrix_.commit((rows + 1) * row_bytes()); status != status_t::success_k)
            return status;
        if (status_t status = kernel_ == nk_kernel_dot_k ? status_t::success_k
                                                         : norms_column_.commit(rows * sizeof(norms_t));
            status != status_t::success_k)
            return status;
        if (status_t status = keys_column_.commit(rows * sizeof(key_t)); status != status_t::success_k) return status;
        return neighbors_matrix_.commit(rows * connectivity() * sizeof(u32_t));
    }

    /** Maps up to @p rows and sentinel-fills on @p stream every mapped neighbor row no fill reached
     *  yet - always ahead of any launch that could assign new_node_ids into them - zeroing the
     *  published count with the first fill. @p stream first follows the last fill, whichever stream
     *  queued it, so a claim never launches into rows still filling. A refused fill advances
     *  nothing, so the next grow retries it. Callers hold the claim over the rows, or the closed
     *  claim word. */
    status_t grow_nodes_(std::size_t rows, stream_handle_t stream) noexcept {
        if (status_t const mapped = map_nodes_(rows); failed(mapped)) return mapped;
        if (failed(vendor_t::follow(api(), stream, grown_))) return status_t::device_error_k;
        if (rows <= filled_rows_) return status_t::success_k;
        std::size_t const mapped_rows = capacity(), row_words = connectivity();
        if (!filled_rows_ && failed(vendor_t::fill_words(api(), published_rows_(), 0, 1, stream)))
            return status_t::device_error_k;
        if (failed(vendor_t::fill_words(api(), neighbors_matrix_.base() + filled_rows_ * row_words * sizeof(u32_t),
                                        node_id_unknown_k, (mapped_rows - filled_rows_) * row_words, stream)) ||
            failed(grown_.record(stream)))
            return status_t::device_error_k;
        filled_rows_ = mapped_rows;
        return status_t::success_k;
    }

    /** Rounds the requested expansion to the register ladder of 2 to 16 slots of the vendor's warp
     *  width, or zero past it, where the arena frontier's rung walks the expansion itself. */
    u32_t candidate_slots_for_(u32_t expansion_factor, u32_t default_slots) const noexcept {
        return gpu_candidate_slots(expansion_factor, default_slots, kernels_type_::wave_lanes());
    }

    /** Candidates a walk's diversity selection caches: all of the walk's candidates, as the CPU's
     *  selection judges, @p candidate_slots ranks per lane on the register ladder, or the arena
     *  frontier's @p expansion past it. */
    u32_t walk_selection_nodes_(u32_t expansion, u32_t candidate_slots) const noexcept {
        return candidate_slots ? candidate_slots * kernels_type_::wave_lanes() : expansion;
    }

    /** Candidates a re-selection caches, a back-link round's or a refinement merge's: a row of the
     *  node's own neighbors or forward picks, a row's worth of incoming appended nodes or reverse
     *  proposals, and the node's own row last. */
    u32_t reselection_nodes_() const noexcept { return 2 * static_cast<u32_t>(connectivity()) + 1; }

    /** Bytes of one builder warp's slice, its walk's or its re-selection's, whichever caches more:
     *  an insert and its back-link, or a refine and its merge, share one arena shape. */
    u32_t builder_scratch_bytes_per_warp_(u32_t expansion, u32_t candidate_slots) const noexcept {
        return build_scratch_bytes_per_warp(
            max_of(walk_selection_nodes_(expansion, candidate_slots), reselection_nodes_()));
    }

    /** One building warp's arena slice for @p walks walks on the rung of @p candidate_slots, the
     *  arena frontier's over @p expansion at zero, then the builders' scratch when shared memory
     *  cannot fit it. */
    gpu_arena_shape_t build_arena_shape_(u32_t expansion, u32_t candidate_slots, u32_t walks) const noexcept {
        u32_t const scratch_bytes = builder_scratch_bytes_per_warp_(expansion, candidate_slots);
        gpu_arena_shape_t shape = arena_shape_for(
            walk_selection_nodes_(expansion, candidate_slots),
            candidate_slots ? frontier_kind_t::register_ladder_k : frontier_kind_t::wide_arena_k, walks);
        if (!kernels_.build_scratch_fits(*this, scratch_bytes)) shape.scratch_bytes = scratch_bytes;
        return shape;
    }

    /** Rebuilds the norms of the @p rows rows from @p first_row on from their vectors, through the
     *  @p kernels given. */
    status_t launch_derive_norms_(kernel_table_t const& kernels, std::size_t first_row, std::size_t rows,
                                  stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.graph.storage = storage();
        arguments.graph.first_id = static_cast<u32_t>(first_row);
        arguments.graph.rows_count = static_cast<u32_t>(rows);
        return kernels_.launch_(
            kernels, vram_dense_kernel_t::derive_norms_k, 0,
            device_.multiprocessor_count * kernels_type_::resident_blocks_(vram_dense_kernel_t::derive_norms_k, 0), 0,
            stream, arguments);
    }

    /** Rebases rows @p rows of the neighbors matrix from @p source_range onto the IDs from @p shift
     *  and compacts them through @p kernels, as a load through the bounce does on the host. */
    status_t launch_rebase_links_(kernel_table_t const& kernels, iota_view<node_id_t> rows,
                                  iota_view<node_id_t> source_range, node_id_t shift, stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        vram_graph_t& graph = arguments.graph;
        graph.neighbors = neighbor_rows_();
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.first_id = rows.first;
        graph.rows_count = static_cast<u32_t>(rows.size());
        graph.source_first = source_range.first;
        graph.source_bound = source_range.bound;
        graph.shift = shift;
        return kernels_.launch_(
            kernels, vram_kernel_t::rebase_links_k, 0,
            device_.multiprocessor_count * kernels_type_::resident_blocks_(vram_kernel_t::rebase_links_k, 0), 0, stream,
            arguments);
    }

    /** Merges the attended rows and carries of @p shards walks into the first's on @p stream
     *  through @p kernels - the sharded attend's gather. */
    status_t launch_merge_attention_(kernel_table_t const& kernels, byte_t* blocks, u32_t rows, u32_t columns,
                                     u32_t shards, stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.attend.outputs = reinterpret_cast<f32_t*>(blocks);
        arguments.attend.rows = rows;
        arguments.attend.columns = columns;
        arguments.attend.shards = shards;
        return kernels_.launch_(kernels, vram_dense_kernel_t::merge_attention_k, 0,
                                divide_round_up<u32_t>(rows, vram_warps_per_block_k), 0, stream, arguments);
    }

    /** Resolves the @p walked matches per query of @p queries_count at @p ids and @p distances
     *  through the device @p keys column into @p ranking on @p stream through @p kernels - a
     *  sharded search's own ranking, gathered or copied as is. */
    status_t launch_resolve_matches_(kernel_table_t const& kernels, node_id_t const* ids, distance_t const* distances,
                                     u32_t walked, byte_t const* keys, vram_ranking_t const& ranking,
                                     u32_t queries_count, stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.matches.ids = ids;
        arguments.matches.distances = distances;
        arguments.matches.keys = keys;
        arguments.matches.ranking = ranking;
        arguments.matches.walked = walked;
        arguments.matches.queries_count = queries_count;
        return kernels_.launch_(kernels, vram_dense_kernel_t::resolve_matches_k, 0,
                                divide_round_up<u32_t>(queries_count, vram_warps_per_block_k), 0, stream, arguments);
    }

    /** Merges @p shards rankings of @p queries_count queries, gathered @p block_bytes apart from
     *  @p gathered, into @p ranking on @p stream through @p kernels - the device gather that serves
     *  the sharded search. */
    status_t launch_merge_matches_(kernel_table_t const& kernels, vram_ranking_t const& gathered,
                                   std::size_t block_bytes, u32_t shards, vram_ranking_t const& ranking,
                                   u32_t queries_count, stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.matches.gathered = gathered;
        arguments.matches.ranking = ranking;
        arguments.matches.block_bytes = block_bytes;
        arguments.matches.queries_count = queries_count;
        arguments.matches.shards = shards;
        return kernels_.launch_(kernels, vram_dense_kernel_t::merge_matches_k, 0,
                                divide_round_up<u32_t>(queries_count, vram_warps_per_block_k), 0, stream, arguments);
    }

    /**
     *  @brief Copies the stored row of every entry of @p rows, device-resident, into its row of
     *      @p outputs, cast into their scalar on @p stream: a lookup's gather straight into
     *      device outputs, with no host bounce.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p outputs are narrower than the index.
     *  - @c unsupported_layout_k Rows of @p outputs sit closer than one packed row apart.
     *  - @c unsupported_k No cast reaches the scalar of @p outputs from the stored one.
     */
    status_t launch_gather_vectors_(kernel_table_t const& kernels, std::span<vram_gathered_row_t const> rows,
                                    rows_span_t outputs, stream_handle_t stream) const noexcept {
        if (outputs.columns != dimensions()) return status_t::config_mismatch_k;
        if (outputs.stride < divide_round_up<std::size_t>(dimensions() * nk_dtype_bits(outputs.dtype), 8u))
            return status_t::unsupported_layout_k;
        if (rows.empty()) return status_t::success_k;
        vram_arguments_t arguments {};
        vram_gather_t& gather = arguments.gather;
        gather.source = vectors_rows_();
        gather.outputs = outputs.data;
        gather.rows = rows.data();
        gather.source_stride = row_bytes();
        gather.output_stride = outputs.stride;
        gather.count = static_cast<u32_t>(rows.size());
        // One plan serves every row, so it copies in words no wider than every row's start aligns.
        std::size_t const addresses = reinterpret_cast<std::uintptr_t>(gather.source) | gather.source_stride |
                                      reinterpret_cast<std::uintptr_t>(gather.outputs) | gather.output_stride;
        if (!nk_cast_plan_serial_(vector_dtype_, outputs.dtype, dimensions(), addresses, &gather.cast))
            return status_t::unsupported_k;
        return kernels_.launch_(kernels, vram_kernel_t::gather_vectors_k, 0,
                                divide_round_up<u32_t>(gather.count, vram_warps_per_block_k), 0, stream, arguments);
    }

    /** Sums the per-query @p counts of a batch, device-resident, into the one device word at
     *  @p total on @p stream through @p kernels, which the host copies back alone. */
    status_t launch_sum_counts_(kernel_table_t const& kernels, std::span<std::size_t const> counts, std::size_t* total,
                                stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.counts = {counts.data(), total, static_cast<u32_t>(counts.size())};
        return kernels_.launch_(kernels, vram_kernel_t::sum_counts_k, 0, 1, 0, stream, arguments);
    }

    /**
     *  @brief Resolves this shard's walked products into @p ranking on @p stream, one row per
     *      query: the first @p scored of each row of @p ids, no more than the ranking's width,
     *      through the device key column, and their @p products beside them.
     *
     *  The products stay as they are, or widen to f64 when the ranking keeps f64 scores; the
     *  ranking's counts, where it keeps any, take the whole scored counts.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p ids, @p products and @p scored disagree in rows or stride, or
     *    the ranking holds other keys or scores than the index's.
     *  - @c unsupported_layout_k A row of @p ids or @p products is not contiguous.
     */
    status_t launch_resolve_products_(kernel_table_t const& kernels, nk::matrix_view<node_id_t> ids,
                                      nk::matrix_view<dot_result_t> products, std::span<u32_t const> scored,
                                      vram_ranking_t const& ranking, stream_handle_t stream) const noexcept {
        static_assert(sizeof(dot_result_t) == sizeof(node_id_t), "Products lie as many values apart as the IDs");
        if (products.extent(0) != ids.extent(0) || products.stride_bytes(0) != ids.stride_bytes(0) ||
            scored.size() != ids.extent(0) ||
            (ranking.score_bytes != sizeof(dot_result_t) && ranking.score_bytes != sizeof(f64_t)) ||
            ranking.key_bytes != sizeof(key_t))
            return status_t::config_mismatch_k;
        if (ids.stride_bytes(1) != sizeof(node_id_t) || products.stride_bytes(1) != sizeof(dot_result_t))
            return status_t::unsupported_layout_k;
        if (scored.empty()) return status_t::success_k;
        vram_arguments_t arguments {};
        vram_multiply_t& resolving = arguments.products;
        resolving.ids = ids.data();
        resolving.products = reinterpret_cast<byte_t const*>(products.data());
        resolving.scored = scored.data();
        resolving.keys = reinterpret_cast<byte_t const*>(device_keys_());
        resolving.ranking = ranking;
        resolving.ids_stride = static_cast<u32_t>(ids.stride_bytes(0) / sizeof(node_id_t));
        resolving.queries_count = static_cast<u32_t>(scored.size());
        return kernels_.launch_(kernels, vram_kernel_t::resolve_products_k, 0,
                                divide_round_up<u32_t>(resolving.queries_count, vram_warps_per_block_k), 0, stream,
                                arguments);
    }

    /** The query rows the last search, multiply or attend on @p context staged, @p rows of them in
     *  the stored scalar at the vectors matrix pitch; valid until the context's next submission. */
    rows_view_t staged_queries_(traversal_context_t const& context, std::size_t rows) const noexcept {
        return {context.staging_.get(), rows, dimensions(), row_bytes(), vector_dtype_};
    }

    /** Packs the @p shards rankings @p block_bytes apart from @p gathered into @p tape on
     *  @p stream, one block scanning the offsets past the tape's base and copying: each of the
     *  @p queries_count queries keeps the first shard's pairs, then the next's, up to the width,
     *  and its counts across the shards sum into @p scored unless it is empty. A product's outputs
     *  compacted on the device, which the tape must not overlap. */
    status_t launch_compact_products_(kernel_table_t const& kernels, vram_ranking_t const& gathered,
                                      std::size_t block_bytes, u32_t shards, u32_t queries_count,
                                      vram_products_tape_t const& tape, std::span<std::size_t> scored,
                                      stream_handle_t stream) const noexcept {
        if (!scored.empty() && scored.size() != queries_count) return status_t::config_mismatch_k;
        vram_arguments_t arguments {};
        vram_multiply_t& compacting = arguments.products;
        compacting.ranking = gathered;
        compacting.tape = tape;
        compacting.totals = scored.data();
        compacting.block_bytes = block_bytes;
        compacting.queries_count = queries_count;
        compacting.shards_count = shards;
        return kernels_.launch_(kernels, vram_kernel_t::compact_products_k, 0, 1, 0, stream, arguments);
    }

    /** Nominates on @p stream every query's distinct keys across the @p shards rankings
     *  @p block_bytes apart from @p gathered, over its token rows @p query_offsets apart, one more
     *  offset than queries, into its row of @p nominees, whose counts take how many it holds; keys
     *  past the nominees' width drop. */
    status_t launch_nominate_keys_(kernel_table_t const& kernels, vram_ranking_t const& gathered,
                                   std::size_t block_bytes, u32_t shards, std::span<u32_t const> query_offsets,
                                   vram_ranking_t const& nominees, stream_handle_t stream) const noexcept {
        if (query_offsets.size() < 2) return status_t::success_k;
        if (nominees.key_bytes != gathered.key_bytes || !nominees.counts) return status_t::config_mismatch_k;
        vram_arguments_t arguments {};
        vram_maxsim_t& maxsim = arguments.maxsim;
        maxsim.gathered = gathered;
        maxsim.nominees = nominees;
        maxsim.block_bytes = block_bytes;
        maxsim.shards_count = shards;
        maxsim.query_offsets = query_offsets.data();
        maxsim.queries_count = static_cast<u32_t>(query_offsets.size() - 1);
        return kernels_.launch_(kernels, vram_kernel_t::nominate_keys_k, 0,
                                divide_round_up<u32_t>(maxsim.queries_count, vram_warps_per_block_k), 0, stream,
                                arguments);
    }

    /**
     *  @brief Rescores on @p stream every nominee of the union tape, whose offsets are
     *      @p nominee_offsets, against this shard: for each token row of its query in
     *      @p queries, as the window's search staged them @p query_offsets apart, the largest
     *      dot with this shard's @p entries of it, listed by ascending output row, into its row
     *      of @p partials, the dot's lowest value where the shard holds none.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p queries are not staged rows of the stored scalar, the offsets
     *    disagree in queries, or @p partials holds other than one row per nominee or fewer columns
     *    than the widest query's rows.
     *  - @c unsupported_layout_k The rows of @p partials are not contiguous.
     */
    status_t launch_maxsim_rescore_(kernel_table_t const& kernels, rows_view_t queries,
                                    std::span<u32_t const> query_offsets, std::span<std::size_t const> nominee_offsets,
                                    std::span<vram_gathered_row_t const> entries,
                                    nk::matrix_span<dot_result_t> partials, stream_handle_t stream) const noexcept {
        if (queries.dtype != vector_dtype_ || queries.stride != row_bytes() || query_offsets.size() < 2 ||
            nominee_offsets.size() != query_offsets.size())
            return status_t::config_mismatch_k;
        if (partials.stride_bytes(1) != sizeof(dot_result_t)) return status_t::unsupported_layout_k;
        if (!partials.extent(0)) return status_t::success_k;
        vram_arguments_t arguments {};
        vram_maxsim_t& maxsim = arguments.maxsim;
        maxsim.storage = storage();
        maxsim.queries = queries.data;
        maxsim.query_offsets = query_offsets.data();
        maxsim.queries_count = static_cast<u32_t>(query_offsets.size() - 1);
        maxsim.nominee_offsets = nominee_offsets.data();
        maxsim.nominees_count = static_cast<u32_t>(partials.extent(0));
        maxsim.entries = entries.data();
        maxsim.entries_count = static_cast<u32_t>(entries.size());
        maxsim.partials = reinterpret_cast<byte_t*>(partials.data());
        maxsim.rows_stride = static_cast<u32_t>(partials.stride_bytes(0) / sizeof(dot_result_t));
        if (partials.extent(1) > maxsim.rows_stride) return status_t::config_mismatch_k;
        return kernels_.launch_(kernels, vram_kernel_t::maxsim_rescore_k, 0,
                                divide_round_up<u32_t>(maxsim.nominees_count, vram_warps_per_block_k), 0, stream,
                                arguments);
    }

    /** Folds on @p stream every nominee's products into @p sums, widened past the dot result type:
     *  per token row of its query, the largest of the @p shards partials @p block_bytes apart from
     *  @p partials, summed over the rows, with the union tape's @p nominee_offsets and
     *  @p query_offsets placing them. */
    status_t launch_maxsim_fold_(kernel_table_t const& kernels, nk::matrix_span<dot_result_t> partials,
                                 std::size_t block_bytes, u32_t shards, std::span<u32_t const> query_offsets,
                                 std::span<std::size_t const> nominee_offsets, std::span<f64_t> sums,
                                 stream_handle_t stream) const noexcept {
        if (query_offsets.size() < 2 || nominee_offsets.size() != query_offsets.size() ||
            sums.size() != partials.extent(0))
            return status_t::config_mismatch_k;
        if (partials.stride_bytes(1) != sizeof(dot_result_t)) return status_t::unsupported_layout_k;
        if (sums.empty()) return status_t::success_k;
        vram_arguments_t arguments {};
        vram_maxsim_t& maxsim = arguments.maxsim;
        maxsim.partials = reinterpret_cast<byte_t*>(partials.data());
        maxsim.block_bytes = block_bytes;
        maxsim.shards_count = shards;
        maxsim.rows_stride = static_cast<u32_t>(partials.stride_bytes(0) / sizeof(dot_result_t));
        maxsim.query_offsets = query_offsets.data();
        maxsim.queries_count = static_cast<u32_t>(query_offsets.size() - 1);
        maxsim.nominee_offsets = nominee_offsets.data();
        maxsim.nominees_count = static_cast<u32_t>(sums.size());
        maxsim.totals = sums.data();
        return kernels_.launch_(kernels, vram_kernel_t::maxsim_fold_k, 0,
                                divide_round_up<u32_t>(maxsim.nominees_count, vram_warps_per_block_k), 0, stream,
                                arguments);
    }

    /** Appends on @p stream every query's @p wanted best nominees of the union tape @p nominated,
     *  by their @p sums, largest first, to @p tape past its base, keys and f64 sums, and each
     *  query's nominee count to @p scored, which sets how many queries there are. */
    status_t launch_top_k_(kernel_table_t const& kernels, std::span<f64_t> sums, vram_products_tape_t const& nominated,
                           std::size_t wanted, vram_products_tape_t const& tape, std::span<std::size_t> scored,
                           stream_handle_t stream) const noexcept {
        if (scored.empty()) return status_t::success_k;
        vram_arguments_t arguments {};
        vram_maxsim_t& maxsim = arguments.maxsim;
        maxsim.totals = sums.data();
        maxsim.nominated = nominated;
        maxsim.tape = tape;
        maxsim.scored = scored.data();
        maxsim.wanted = wanted;
        maxsim.queries_count = static_cast<u32_t>(scored.size());
        maxsim.key_bytes = sizeof(key_t);
        return kernels_.launch_(kernels, vram_kernel_t::top_k_k, 0,
                                divide_round_up<u32_t>(maxsim.queries_count, vram_warps_per_block_k), 0, stream,
                                arguments);
    }

    /** Narrows @p rows rows of the index's columns at @p from, @p from_stride bytes apart, into
     *  @p to at the vectors matrix pitch at unit RMS on @p stream through @p kernels, one block
     *  per row narrowed. */
    status_t launch_rmscast_(kernel_table_t const& kernels, f32_t const* from, std::size_t from_stride, byte_t* to,
                             std::size_t rows, stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        nk_each_rmsnorm_arguments_t& rmscast = arguments.rmscast;
        rmscast.x = reinterpret_cast<unsigned char const*>(from);
        rmscast.y = reinterpret_cast<unsigned char*>(to);
        rmscast.groups = 1;
        rmscast.columns = dimensions();
        rmscast.vectors = rows;
        rmscast.x_stride = from_stride;
        rmscast.y_stride = row_bytes();
        rmscast.epsilon = std::numeric_limits<f32_t>::min();
        u32_t const blocks = static_cast<u32_t>(min_of<std::size_t>(
            rows, device_.multiprocessor_count * kernels_type_::resident_blocks_(vram_dense_kernel_t::rmscast_k, 0)));
        return kernels_.launch_(kernels, vram_dense_kernel_t::rmscast_k, 0, blocks, 0, stream, arguments);
    }

#pragma endregion Launches

  private:
    friend kernels_type_;

    /** The vectors matrix and norms column backend, rebound to bytes. */
    using vectors_matrix_allocator_t = typename vectors_allocator_t::template rebind<byte_t>::other;

    /** The neighbors matrix backend, rebound to bytes. */
    using neighbors_matrix_allocator_t = typename neighbors_allocator_t::template rebind<byte_t>::other;

    /** Hardware identity: the ordinal comes from the vectors allocator at construction, the
     *  remaining specs fill at the first growth. */
    device_specs_t device_;

    /** The metric every walk scores by; under dot, the one metric that reads no norms, the norms
     *  column stays empty. */
    nk_kernel_kind_t kernel_;

    /** The dimensions and connectivity, fixed at construction. */
    geometry_t geometry_;

    /** The vectors matrix, its first row the published-row count, the norms column and the keys
     *  column; each allocator is its storage, owning the base and the committed extent, so
     *  capacities are derived, never cached. */
    vectors_matrix_allocator_t vectors_matrix_, norms_column_, keys_column_;

    /** The neighbors matrix, on the same terms. */
    neighbors_matrix_allocator_t neighbors_matrix_;

    /** The claim word, closed while a refine, an absorb, a load or a save of the sharded index
     *  holds it. */
    claim_word_t claims_;

    /** Mapped rows whose neighbors a grow sentinel-filled, read and advanced by the claim's holder
     *  alone; a reservation maps past it and the next claim fills the rest. */
    std::size_t filled_rows_ = 0;

    /** Rows the reserved ranges span, the most any claim reaches; fixed by the first growth. */
    std::size_t ceiling_nodes_ = max_node_count_k;

    /** Marks the last sentinel fill, or the copies of a load that died over filled rows, which
     *  every grow's stream follows; recorded and followed by the claim's holder alone. */
    event_t grown_;

    /** Every node's key on the host and every key's nodes: assigned by the claim's holder ahead of
     *  its launch, and filed by it after its drain, before the watermark turns. */
    keyed_t keyed_;

    /** The page-locked ring the host keys cross into the device keys through: two halves, each
     *  refilled only once the copy out of it finished, grown with the storage. */
    growing_buffer<key_t, typename vendor_t::pinned_allocator_t> keys_staging_;

    /** Marks the last copy out of each staging half, born on the index's device. */
    event_t keys_crossed_[2];

    /** Halves the uploads filled so far, alternating between the two; the claim's holder alone
     *  advances it. */
    std::size_t staged_halves_ = 0;

    /** Rows strictly below it are published and visible: turned once per batch, in claim order,
     *  after its drain. */
    std::size_t last_visible_id_ = 0;

    /** The first post-claim device failure: IDs past it may be unpublished or unfilled. */
    status_t status_ = status_t::success_k;

    /** The vendor's kernels: the hooks every launch goes through, each resolving its table on the
     *  stream it launches on. */
    kernels_type_ kernels_;

    /** Demotes a filled bounce half toward the last-level cache, where the copy engine reads it. */
    object_prefetcher<cache_hints_type_> prefetcher_;
};

} // namespace unum::usearch::dense
