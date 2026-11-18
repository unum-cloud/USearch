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

#include <cstddef> // `std::size_t`, `offsetof` - where a round's counters follow the check's

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
#include "usearch/utilities/ram_sorts.hpp"   // `order_by_neighborhood` - a round's barriers order its mirrors
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

/** Which step of a fold a launch runs: the re-selection of each row, or the merge of the picks
 *  the bin's other shards made of it. */
enum class vram_fold_step_t : u32_t { reselect_k, merge_k };

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
    vram_fold_step_t step = vram_fold_step_t::reselect_k;
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
    vram_bitset_t reached, frontier, next_frontier, reaches_root, back_settled;
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

    /** The targets' neighbors, vectors, norms and keys, @c parts_count bases each, the source's
     *  keys, and the words of a key, of a vector row and of a norms entry, zero where none are
     *  kept at all. */
    u32_t* const* part_columns = nullptr;
    u32_t const* keys = nullptr;
    u32_t vector_words = 0;
    u32_t norm_words = 0;
    u32_t key_words = 0;
};

/** One gather launch. @c entries is null in the first gather, whose one run spans IDs. Every run
 *  splits into items of as many entries as its hubs, never fewer than a block's warps, so staging
 *  a run's hubs costs no more than scoring them; @c run_items holds the items before each run.
 *  @c staged_bytes is the shared memory the hub rows take, or zero to read them in place. */
struct vram_gather_neighborhoods_t {
    vram_flat_storage_t storage;
    u32_t const* entries = nullptr;
    u32_t const* hub_nodes = nullptr;
    vram_neighborhood_run_t const* runs = nullptr;
    u32_t const* run_items = nullptr;
    u32_t runs_count = 0;
    u32_t staged_bytes = 0;
    u16_t* entry_neighborhoods = nullptr;
    u32_t* assigned_items = nullptr;
    vram_telemetry_block_t* telemetry = nullptr;
};

/** One restitch launch: the neighborhoods in order, each block's slice of a neighborhood capacity
 *  of mates, @c connectivity each, the shared memory its mate scoring may take, zero for none, the
 *  selection caches the builders shape, the check's columns the reach rule reads and takes from,
 *  and the claim counter. */
struct vram_restitch_t {
    vram_flat_storage_t storage;
    u32_t const* neighborhood_entries = nullptr;
    u32_t const* neighborhood_offsets = nullptr;
    u32_t neighborhoods_count = 0;
    u32_t capacity = 0;
    u32_t staged_bytes = 0;
    byte_t* mates = nullptr;
    u32_t selection_nodes = 0;
    u32_t scratch_bytes_per_warp = 0;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    vram_health_t health;
    u32_t* assigned = nullptr;
    vram_telemetry_block_t* telemetry = nullptr;
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
    vram_fold_t fold;
    vram_search_sink_t search_sink;
    vram_multiply_sink_t multiply_sink;
    vram_rescoring_t rescoring;
    vram_graph_t graph;
    vram_health_t health;
    vram_gather_neighborhoods_t gather_neighborhoods;
    vram_restitch_t restitch;
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

    /** How entries are reached; @c scan_k scores every visible row exactly on the kits' tile tasks,
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
    typename key_type_ = uuid128_t,                //
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

    /** Tasks the lanes cooperating on one row take across it. */
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
     *  over at planning so no task names a destination of its own. */
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

    /** Maps the storage, rows committed already included, for @p devices too, as a sharded
     *  index packing this shard into its own names them; @c config_mismatch_k when one of those
     *  devices cannot reach this storage. */
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

    /** The dense refine config; a round reads its @c cache_bytes, never more than this device
     *  grants. */
    using refine_config_t = dense::refine_config_t;

    /** A reshard's knobs: its rounds', a shard capacity a mono-index refuses, and the seed that
     *  orders the parts' growth. */
    using reshard_config_t = dense::reshard_config_t;

    using refine_report_t = dense::refine_report_t;
    using reachability_t = dense::reachability_t;

    struct refine_plan_t;
    struct reshard_plan_t;

  private:
    using round_scratch_t = vram_round_scratch<traversal_context_t>;
    using round_counters_t = vram_round_counters_t;

    /** Gives a round's counters back to the stream's pinned slots they came from. */
    struct round_counters_return_t {
        typename vendor_t::pinned_slots_t* pinned_slots = nullptr;
        void operator()(round_counters_t* counters) const noexcept { pinned_slots->give_back(counters); }
    };
    using round_counters_ptr_t = std::unique_ptr<round_counters_t, round_counters_return_t>;
    static_assert(sizeof(round_counters_t) <= vendor_t::pinned_slots_t::slot_bytes_k);

    /** How a round over @c entries entries sizes its columns: the hubs one gather draws at most and
     *  the most entries a neighborhood holds, as RAM names them, then the room for the starts with
     *  splits, for the hubs of every gather's runs and for the runs. */
    struct round_shape_t {
        std::size_t entries = 0, hubs_per_draw = 0, capacity = 0;
        std::size_t offsets = 0, hubs = 0, runs = 0;
    };

    /** The page-locked host bytes a round's barriers order on, borrowed from its context's results:
     *  the entries by neighborhood, the starts with split room, the hubs, runs and items of the
     *  next gather, and each entry's neighborhood. */
    struct round_mirrors_t {
        u32_t* neighborhood_entries = nullptr;
        u32_t* neighborhood_offsets = nullptr;
        u32_t* hub_nodes = nullptr;
        vram_neighborhood_run_t* runs = nullptr;
        u32_t* run_items = nullptr;
        u16_t* entry_neighborhoods = nullptr;
    };

    /** One target's part of a split as the host assigns it: its nodes, and those assigned so
     *  far. */
    struct split_part_t {
        std::size_t entries = 0, assigned_nodes = 0;
    };

    /** A reshard's moves beside the rounds they turn into. */
    struct split_t {

        /** The index the split empties, until the scatter finished. */
        vram_shard_core* source = nullptr;

        /** The targets, one part each, and the one the round is bound to. */
        growing_buffer<vram_shard_core*> targets {global_allocator};
        growing_buffer<split_part_t> parts {global_allocator};
        std::size_t bound_target = 0;

        /** The source's nodes at planning, and the most nodes one part takes: those over the
         *  targets, rounded up. */
        std::size_t entries = 0, part_capacity = 0;

        /** The device scratch, laid out as @c split_scratch_ reads it. */
        typename traversal_context_t::bytes_ptr_t device_bytes;

        /** Page-locked host bytes borrowed from the plan's context, laid out as @c split_layout_
         *  places them, which the round's mirrors take over once the parts publish. */
        byte_t* host_bytes = nullptr;

        /** The growth round the next batch queues from, zero once the growth converged and the
         *  unplaced nodes are placed next. */
        u32_t next_round = 0;

        /** The reports of the targets whose rounds closed, folded, and how many closed. */
        refine_report_t settled {};
        std::size_t settled_targets = 0;

        explicit split_t(typename traversal_context_t::bytes_ptr_t&& empty) noexcept : device_bytes(std::move(empty)) {}
    };

  public:

    /**
     *  @brief One health check in flight on this index's device, the RAM check's twin in the
     *      shape one host thread drives over many devices at once: @ref queue a batch of passes,
     *      @ref wait for it, @ref advance past it, and @ref finish once none follows.
     *
     *  It starts from one root, a seeded draw or the one an earlier check kept, and expands depth
     *  by depth, every depth counting in-links and shallower in-links and marking ways to the root
     *  in the same read; rechecks then mark the entries reached but not yet reaching the root. The
     *  entries outside the root's component, the islands, are written in ascending order and their
     *  ways to the root closed; their strong components are named by their least rank as colors run
     *  forward along the islands' links, marks run back to each color's least rank and the marked
     *  members settle, until every island belongs to one. A drawn root whose component is smaller
     *  than the largest island gives way once to that island's least entry. A count pass then reads
     *  every row once more, and the islands' roles, entries and weak components follow.
     *
     *  Each step queues its passes in batches that double, a pass queued past convergence leaving
     *  at entry, then the copy of the counters into a page-locked slot. Nothing waits and nothing
     *  allocates: planning carved every column. Read-only on the graph.
     */
    struct health_plan_t {

        /** An unbound plan over @p scratch - what an empty index or a refused plan answers with. */
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

        /** Sleeps until the queued batch finished on @p context 's event; a no-op unless
         *  @ref running. */
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
         *  - @c config_mismatch_k, @c device_error_k A batch failed, as it kept.
         */
        expected<reachability_t> finish() const noexcept {
            if (failed(status_)) return {{}, status_};
            if (!index_ || !visible) return {reachability_t {}, status_t::success_k};
            if (step_ != step_t::done_k) return {{}, status_t::internal_k};
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
            return {result, status_t::success_k};
        }

        /** Turns the check onto @p index 's IDs below @p visible, which its columns must span, and
         *  restarts it from a fresh draw. */
        void rebind(vram_shard_core& index, node_id_t visible) noexcept {
            index_ = &index;
            this->visible = visible;
            root_ = node_id_unknown_k;
            restart_sweep();
        }

        /** Starts over from the root an earlier sweep kept, or else from a seeded draw; the next
         *  batch clears the columns first. */
        void restart_sweep() noexcept {
            // Opening and closing checks share their root, so a fresh draw cannot read as progress.
            bool const kept = root_ < visible;
            node_id_t const root = kept      ? root_
                                   : visible ? static_cast<node_id_t>(seeded_bijection(0, visible, 0))
                                             : node_id_unknown_k;
            drawn_root_ = kept ? node_id_unknown_k : root;
            start_from_(root);
        }

      private:
        friend struct vram_shard_core;
        friend refine_plan_t;
        friend reshard_plan_t;

        /** The check's steps, as the RAM check names its passes, plus the passes that label its
         *  islands on the device, which RAM runs on one thread once its rechecks end. */
        enum class step_t : u8_t { expand_k, recheck_k, islands_k, components_k, count_links_k, labels_k, done_k };

        /** Clears what one root's sweep fills, the next batch queuing the fills, and marks
         *  @p root both ways at depth zero; with no root the check is over. */
        void start_from_(node_id_t root) noexcept {
            root_ = root;
            next_pass_ = 0;
            step_ = root == node_id_unknown_k ? step_t::done_k : step_t::expand_k;
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
                drawn_root_ = node_id_unknown_k;
                start_from_(static_cast<node_id_t>(~static_cast<u32_t>(counters.largest_island)));
                return true;
            }
            return next_step_(step_t::count_links_k);
        }

        /** The check's and the round's device columns, carved once at planning. */
        round_scratch_t columns_;
        vram_shard_core* index_ = nullptr;
        round_counters_ptr_t counters_ {nullptr, round_counters_return_t {}};

        /** The root the sweeps start from, which the closing check keeps; the draw it began with,
         *  which may give way once to the largest island. */
        node_id_t root_ = node_id_unknown_k, drawn_root_ = node_id_unknown_k;

        /** The depth, recheck or pass the next batch queues from, doubling as the batches do, and
         *  the last one the queued batch holds. */
        u32_t next_pass_ = 0, last_pass_ = 0;
        vram_island_pass_t island_pass_ = vram_island_pass_t::forward_k;
        step_t step_ = step_t::done_k;
        status_t status_ = status_t::success_k;
    };

    /**
     *  @brief One refinement round in flight on this index's device - the CPU's round in the shape
     *      one host thread drives over many devices at once: @ref queue a phase, @ref wait for it,
     *      @ref advance past it, and @ref finish once no phase follows.
     *
     *  The round opens with a health check, heals the islands it named with no distance, draws one
     *  cache-sized set of hubs by seeded bijection, gathers every entry into the neighborhood of
     *  its nearest hub, and again inside every neighborhood over its capacity among hubs drawn from
     *  its own entries, then restitches each neighborhood inside one multiprocessor: every entry
     *  keeps the diverse links among its own and its nearest neighborhood-mates and those
     *  reachability needs, never refilled. The closing health check from the same root decides the
     *  verdict. The hubs are drawn and the entries ordered by neighborhood on the host, at the
     *  barriers, over page-locked mirrors.
     *
     *  A phase queues its launches and their copies behind the context's event and never waits;
     *  planning made every allocation, so no phase allocates, and only a device fault fails one.
     *  Every phase takes the context the plan was opened on back from its caller. Demands an
     *  exclusive caller: the heal and the restitch write live neighbors with plain stores. A
     *  stopped round leaves the graph correct and merely less improved.
     */
    struct refine_plan_t {

        /** The check the round opens and closes with, which owns the device scratch. */
        health_plan_t health;

        /** An unbound plan over @p health - what a refused plan answers with. */
        explicit refine_plan_t(health_plan_t&& health) noexcept : health(std::move(health)) {}
        refine_plan_t(refine_plan_t&&) noexcept = default;
        refine_plan_t& operator=(refine_plan_t&&) noexcept = default;

        /**
         *  @brief Queues the current phase on @p context 's stream, then its copies behind the
         *      context's event; never waits and never allocates anything at all.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers, or @ref status
         *    keeps it.
         *  - @c config_mismatch_k The @p context rides another device, or the plan keeps it.
         *  - @c device_error_k A launch or a copy failed, or the plan keeps it.
         */
        status_t queue(traversal_context_t& context, refine_config_t) noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (failed(status_)) return status_;
            if (context.device() != index_->device_.ordinal) return status_t::config_mismatch_k;
            auto const [kernels, kernels_status] = index_->kernels_.kernels_(context);
            if (failed(kernels_status)) return kernels_status;
            switch (phase_) {
            case phase_t::health_k: status_ = health.queue(context); break;
            case phase_t::heal_k: status_ = index_->queue_heal_(context, health); break;
            case phase_t::gather_k: status_ = index_->queue_gather_(context, *this); break;
            case phase_t::restitch_k: status_ = index_->queue_restitch_(context, *this); break;
            // The barrier closing the draw draws the hubs, which the gather uploads with its runs.
            default: return status_t::success_k;
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
            case phase_t::heal_k: phase_ = phase_t::hubs_k; break;
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
         *  re-stitched standing; the closing check runs on, and the verdict reads stopped. A no-op
         *  outside a round. */
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
            case phase_t::gather_k: {
                std::size_t entries = 0;
                for (std::size_t run = 0; run != runs_; ++run) entries += mirrors_.runs[run].width;
                return entries;
            }
            case phase_t::restitch_k: return health.visible;
            default: return 0;
            }
        }

        /** Opens the next round once one ended, from the root the closing health check kept: the
         *  report starts over, and the opening health check sweeps again; a no-op mid-round. */
        void next_round() noexcept {
            if (phase_ != phase_t::done_k || failed(status_)) return;
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
         *  - @c config_mismatch_k, @c device_error_k or @c internal_k A phase kept it.
         *  - @c cancelled_k A phase still follows, or the round was stopped.
         */
        expected<refine_report_t> finish(traversal_context_t const& context) const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            refine_report_t result = tally_;
            vram_telemetry_block_t const spent = context.spent();
            result.traversal_stats.visited_members = spent.hops;
            result.traversal_stats.computed_distances = spent.scored;
            result.traversal_stats.restarts = spent.restarts;
            result.traversal_stats.saturated_walks = spent.visited_saturations;
            if (failed(status_)) return {result, status_};
            bool const done = phase_ == phase_t::done_k && result.verdict != refine_verdict_t::stopped_k;
            return {result, done ? status_t::success_k : status_t::cancelled_k};
        }

      private:
        friend struct vram_shard_core;
        friend reshard_plan_t;

        /** Opens the health check following @p phase from the root the last one kept. */
        void verify_(phase_t phase) noexcept {
            health.restart_sweep();
            swept_after_ = phase;
            phase_ = phase_t::health_k;
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

        /** Items one run of @p hubs hubs over @p width entries splits into: as many entries an item
         *  as hubs, so staging a run's hubs costs no more than scoring them, and never fewer than a
         *  block's warps, so every warp of the block takes an entry. */
        static u32_t run_items_(std::size_t width, std::size_t hubs) noexcept {
            return static_cast<u32_t>(
                divide_round_up<std::size_t>(width, max_of<std::size_t>(hubs, vram_warps_per_block_k)));
        }

        /** Draws one set of hubs by seeded bijection over the visible entries, ascending, as the
         *  first gather's one run over every ID. */
        void draw_hubs_() noexcept {
            node_id_t const visible = health.visible;
            std::size_t const hubs = round_first_hubs(visible, shape_.hubs_per_draw, shape_.capacity);
            for (std::size_t rank = 0; rank != hubs; ++rank)
                mirrors_.hub_nodes[rank] = static_cast<u32_t>(seeded_bijection(rank, visible, round_seed_));
            partial_sort(std::span<u32_t> {mirrors_.hub_nodes, hubs}, hubs);
            mirrors_.runs[0] = {0, visible, 0, static_cast<u32_t>(hubs)};
            mirrors_.run_items[0] = 0;
            mirrors_.run_items[1] = run_items_(visible, hubs);
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
            mirrors_.run_items[0] = 0;
            for (std::size_t neighborhood = 0; neighborhood != count; ++neighborhood) {
                u32_t const first = offsets[neighborhood], width = offsets[neighborhood + 1] - first;
                if (width <= shape_.capacity) continue;
                std::size_t const drawn = round_hubs(width, shape_.hubs_per_draw, shape_.capacity);
                u64_t const seed = mix_bits(round_seed_ ^ first);
                for (std::size_t rank = 0; rank != drawn; ++rank)
                    mirrors_.hub_nodes[hubs + rank] = entries[first + seeded_bijection(rank, width, seed)];
                mirrors_.runs[runs] = {first, width, static_cast<u32_t>(hubs), static_cast<u32_t>(drawn)};
                mirrors_.run_items[runs + 1] = mirrors_.run_items[runs] + run_items_(width, drawn);
                ++runs, hubs += drawn;
            }
            runs_ = runs, hubs_ = hubs;
            return true;
        }

        vram_shard_core* index_ = nullptr;

        /** The page-locked mirrors the barriers order on. */
        round_mirrors_t mirrors_;

        /** The round's sizes, as RAM names them, and the shared memory a gather's hubs, a
         *  restitch's mate scoring and its whole block take, zero where none is granted. */
        round_shape_t shape_;
        u32_t gather_staged_bytes_ = 0, restitch_staged_bytes_ = 0, restitch_shared_bytes_ = 0;

        /** The restitch warps' arena slice, its scratch there when shared memory cannot fit it. */
        gpu_arena_shape_t restitch_arena_shape_ {};

        /** Neighborhoods the order holds, zero before the first gather's; the runs and hubs the
         *  next gather takes. */
        std::size_t neighborhoods_ = 0, runs_ = 0, hubs_ = 0;

        /** The seed of the round's draws. */
        u64_t round_seed_ = 0;

        /** The phase the current health check follows, which decides the one after it. */
        phase_t swept_after_ = phase_t::none_k;
        phase_t phase_ = phase_t::done_k;
        status_t status_ = status_t::success_k;
        refine_report_t tally_;
    };

    /**
     *  @brief Opens one refinement round over the published nodes on @p context 's stream, as
     *      the RAM @c plan_refine does.
     *
     *  Carves the check's columns and the round's from the context, takes page-locked counters and
     *  mirrors, and sizes the hubs and the neighborhood capacity from the shared memory a block
     *  keeps, clamping @c cache_bytes to what this device grants. No phase allocates.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew.
     *  - @c config_mismatch_k The @p context rides another device or cannot launch.
     *  - @c bad_alloc_k A column, a mirror, the arena or the counters cannot be placed.
     */
    expected<refine_plan_t> plan_refine(refine_config_t config, traversal_context_t& context) noexcept {
        refine_plan_t plan {health_plan_t {empty_round_scratch_(context)}};
        std::size_t const rows = size();
        if (status_t const carved = carve_round_(plan, context, rows, config.cache_bytes, 0); failed(carved))
            return {std::move(plan), carved};
        plan.index_ = this;
        plan.health.rebind(*this, static_cast<node_id_t>(rows));
        plan.swept_after_ = phase_t::none_k;
        plan.phase_ = rows ? phase_t::health_k : phase_t::done_k;
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief One @ref plan_refine driven to its end on the calling thread, waiting once per phase.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew.
     *  - @c config_mismatch_k The @p context rides another device or cannot launch.
     *  - @c bad_alloc_k A column, a mirror, the arena or the counters cannot be placed.
     *  - @c device_error_k A launch or a copy failed.
     */
    expected<refine_report_t> refine(refine_config_t config, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_refine(config, context);
        if (failed(status)) return {{}, status};
        do
            if (failed(plan.queue(context, config)) || failed(plan.wait(context))) break;
        while (plan.advance());
        return plan.finish(context);
    }

    /**
     *  @brief One reshard in flight from this index into its targets, on one device, in the shape
     *      of @ref refine_plan_t: @ref queue a phase, @ref wait for it, @ref advance past it, and
     *      @ref finish once no phase follows.
     *
     *  The moves come first: the partition grows one graph-local part per target on the device,
     *  the barrier after it assigns every source node its slot, the scatter moves every node's
     *  vector, norms, key and neighbors into its target, the links inside its part rebased and the
     *  links across dropped, and the barrier after it files the keys, publishes every target at
     *  once and empties the source. Then every target holding nodes runs one round as
     *  @ref refine_plan_t does, its opening heal linking what the cut left apart, the targets'
     *  reports folding into the plan's; @ref next_round runs another round on every target, the
     *  first target first.
     */
    struct reshard_plan_t {

        /** An unbound plan over @p rounds - what a refused plan answers with. */
        explicit reshard_plan_t(refine_plan_t&& rounds) noexcept
            : rounds_(std::move(rounds)),
              split_(typename traversal_context_t::bytes_ptr_t {nullptr, rounds_.health.columns_.bytes.get_deleter()}) {
        }
        reshard_plan_t(reshard_plan_t&&) noexcept = default;
        reshard_plan_t& operator=(reshard_plan_t&&) noexcept = default;

        /**
         *  @brief Queues the current phase on @p context 's stream: a move's launches, or the bound
         *      target's round's, as @ref refine_plan_t::queue does; never waits or allocates.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers.
         *  - @c config_mismatch_k The @p context rides another device, or the plan keeps it.
         *  - @c device_error_k A launch or a copy failed, or the plan keeps it.
         */
        status_t queue(traversal_context_t& context, refine_config_t config) noexcept {
            if (!rounds_.index_) return status_t::uninitialized_k;
            if (!moving()) return rounds_.queue(context, config);
            status_t& status = rounds_.status_;
            if (failed(status)) return status;
            vram_shard_core& source = *split_.source;
            if (context.device() != source.device_.ordinal) return status_t::config_mismatch_k;
            stream_handle_t const stream = context.stream();
            auto const [kernels, kernels_status] = source.kernels_.kernels_(context);
            if (failed(kernels_status)) return kernels_status;
            if (moves_ == phase_t::scatter_k) status = source.queue_scatter_(*kernels, split_, stream);
            else
                status = split_.next_round ? source.queue_rounds_(*kernels, split_, stream)
                                           : source.queue_partition_islands_(*kernels, split_, stream);
            if (succeeded(status)) status = context.record();
            return status;
        }

        /** Sleeps until the queued phase finished, on @p context 's event. */
        status_t wait(traversal_context_t& context) noexcept {
            if (!moving()) return rounds_.wait(context);
            if (succeeded(rounds_.status_)) rounds_.status_ = context.event_.wait();
            return rounds_.status_;
        }

        /** The barrier after a phase finished: grows the parts on, assigns the slots, publishes
         *  the parts, or advances the bound target's round and binds the next target once it
         *  ended; answers whether a phase follows. */
        bool advance() noexcept {
            if (!rounds_.index_ || failed(rounds_.status_)) return false;
            if (moves_ == phase_t::partition_k) {
                // A growth batch that still placed nodes leaves others unplaced, so it grows on;
                // once one placed none, the rest are placed, and the host assigns the slots.
                if (!split_.next_round) assign_slots_(), moves_ = phase_t::scatter_k;
                else if (!vram_shard_core::placing_(split_)) split_.next_round = 0;
                return true;
            }
            if (moves_ == phase_t::scatter_k) {
                publish_parts_();
                moves_ = phase_t::done_k;
                if (failed(rounds_.status_)) return false;
                bind_target_();
                return rounds_.phase_ != phase_t::done_k;
            }
            if (rounds_.advance()) return true;
            // A plan whose last target's round ended settles nothing more.
            if (failed(rounds_.status_) || split_.bound_target == split_.targets.size()) return false;
            return settle_target_();
        }

        /** Stops the bound target's round as @ref refine_plan_t::stop does; the moves run on. */
        void stop() noexcept {
            if (!moving()) rounds_.stop();
        }

        /** Whether the plan still moves the source's nodes: its partition or its scatter. */
        bool moving() const noexcept { return moves_ != phase_t::done_k; }

        /** Whether the bound target runs a round, whose every barrier a caller's progress hears. */
        bool rounding() const noexcept { return !moving() && rounds_.rounding(); }

        /** The phase the plan is in: a move's, then the bound target's round's. */
        phase_t phase() const noexcept { return moving() ? moves_ : rounds_.phase(); }

        /** Where the round's phase sits among its six, and its units done and in all; none in the
         *  moves, which report nothing. */
        std::size_t ordinal() const noexcept { return moving() ? 0 : rounds_.ordinal(); }
        std::size_t completed() const noexcept { return moving() ? 0 : rounds_.completed(); }
        std::size_t total() const noexcept { return moving() ? 0 : rounds_.total(); }

        /** Runs another round on every target once every target's ended, from the first; a no-op
         *  in the moves and mid-round. */
        void next_round() noexcept {
            if (moving() || !rounds_.index_ || rounds_.phase_ != phase_t::done_k || failed(rounds_.status_)) return;
            split_.bound_target = 0, split_.settled_targets = 0;
            bind_target_();
        }

        /**
         *  @brief The rounds so far, folded over the targets: complete once @ref advance answered
         *      false, else cancelled; a failure a phase kept answers instead.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound.
         *  - @c config_mismatch_k, @c device_error_k or @c internal_k A phase kept it.
         *  - @c cancelled_k The moves or a round still follow, or a round was stopped.
         */
        expected<refine_report_t> finish(traversal_context_t const& context) const noexcept {
            if (!rounds_.index_) return {{}, status_t::uninitialized_k};
            if (failed(rounds_.status_)) return {{}, rounds_.status_};
            if (moving()) return {{}, status_t::cancelled_k};
            return rounds_.finish(context);
        }

      private:
        friend struct vram_shard_core;

        /** Assigns every source node its slot in place in the pinned host bytes, in ID order
         *  within each part. */
        void assign_slots_() noexcept {
            u32_t* const slots = reinterpret_cast<u32_t*>(split_.host_bytes);
            for (split_part_t& part : split_.parts) part = {};
            for (std::size_t node = 0; node != split_.entries; ++node) ++split_.parts[slots[node]].entries;
            for (std::size_t node = 0; node != split_.entries; ++node) {
                std::size_t const part = slots[node];
                slots[node] = static_cast<u32_t>(part * split_.part_capacity + split_.parts[part].assigned_nodes++);
            }
        }

        /** Files every source key under its slot in its target, as a load's tasks do, publishes
         *  every target's nodes at once under the closed claim words, and empties the source. */
        void publish_parts_() noexcept {
            vram_shard_core& source = *std::exchange(split_.source, nullptr);
            u32_t const* const slots = reinterpret_cast<u32_t const*>(split_.host_bytes);
            for (std::size_t node = 0; node != split_.entries; ++node)
                split_.targets[slots[node] / split_.part_capacity]->keyed_.assign(
                    static_cast<node_id_t>(slots[node] % split_.part_capacity),
                    source.keyed_[static_cast<node_id_t>(node)]);
            for (std::size_t index = 0; index != split_.targets.size(); ++index) {
                vram_shard_core& target = *split_.targets[index];
                std::size_t const entries = split_.parts[index].entries;
                target.publish_keys_({0, static_cast<node_id_t>(entries)});
                if (status_t const published = target.publish_(0, entries); failed(published))
                    rounds_.status_ = published;
            }
            source.forget_();
        }

        /** Turns the rounds to the bound target or the next one holding nodes, its round opening
         *  with a check rebound onto its part; past the last, reports the folded rounds. */
        void bind_target_() noexcept {
            while (split_.bound_target != split_.targets.size() && !split_.parts[split_.bound_target].entries)
                ++split_.bound_target;
            if (split_.bound_target == split_.targets.size()) {
                rounds_.tally_ = split_.settled;
                rounds_.phase_ = phase_t::done_k;
                return;
            }
            rounds_.index_ = split_.targets[split_.bound_target];
            rounds_.tally_ = {};
            rounds_.health.rebind(*rounds_.index_, static_cast<node_id_t>(split_.parts[split_.bound_target].entries));
            rounds_.swept_after_ = phase_t::none_k;
            rounds_.phase_ = phase_t::health_k;
        }

        /** Folds the bound target's ended round into the plan's, the first one taken whole, then
         *  binds the next target, answering whether a phase follows. */
        bool settle_target_() noexcept {
            if (split_.settled_targets++) split_.settled += rounds_.tally_;
            else split_.settled = rounds_.tally_;
            ++split_.bound_target;
            bind_target_();
            return rounds_.phase_ != phase_t::done_k;
        }

        /** The rounds, bound to the source through the moves, then to each target in turn. */
        refine_plan_t rounds_;
        split_t split_;

        /** The move still pending; done once the parts published. */
        phase_t moves_ = phase_t::done_k;
    };

    /**
     *  @brief Opens a reshard of @p sources into @p targets: this index, the one source, moves into
     *      one graph-local part per target, then every target runs one round, as
     *      @ref reshard_plan_t describes; this index ends empty.
     *
     *  Planning queues the partition behind every allocation: one root per part through @ref
     *  seeded_bijection under the config's seed, then rounds of growth over the links, each part
     *  capped at the nodes over the targets rounded up, and each node a round reaches kept by its
     *  lowest-ID proposer, so equal seeds over equal graphs split alike; nodes no round reached go
     *  to the smallest parts in ID order.
     *
     *  The caller holds the claim words of this index and of every target closed for the plan's
     *  life, and nothing searches this index meanwhile. Every index either list names must
     *  outlive the plan's finish.
     *
     *  @param[in] sources This index alone: a mono-index moves one source into its targets.
     *  @param[in] targets The empty indexes taking the parts, at least two, all distinct.
     *  @param[in] core The member of each listed index holding its core.
     *  @param[in] config The rounds' knobs and the seed picking the parts' roots; a mono-index
     *      places no shards, so the shard capacity must be zero.
     *  @param[in,out] context The driving context, which every phase takes back; the partition's
     *      scratch and the rounds' are carved from it.
     *  @param[in] rows The rows this index holds once the appends planned ahead of the split land,
     *      as a gather into it before its cut plans them; the rows it holds now otherwise.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k An empty list, a null index, or an index listed twice.
     *  - @c unsupported_k A nonzero shard capacity, or any shape but this index alone into two or
     *    more targets.
     *  - @c uninitialized_k The index never grew.
     *  - @c config_mismatch_k This index among the targets, a non-empty target, one of another
     *    device, metric or geometry, or a @p context of another device or one that cannot launch.
     *  - @c contended_k This index's or a target's claim word is open.
     *  - @c overflow_k Part slots past the 32-bit IDs.
     *  - @c bad_alloc_k A column, a mirror, the arena or the counters cannot be placed.
     */
    template <typename owner_type_>
    expected<reshard_plan_t> plan_reshard(std::span<owner_type_* const> sources, std::span<owner_type_* const> targets,
                                          vram_shard_core owner_type_::* core, reshard_config_t config,
                                          traversal_context_t& context, std::size_t rows) noexcept {
        reshard_plan_t plan {refine_plan_t {health_plan_t {empty_round_scratch_(context)}}};
        for (std::span<owner_type_* const> const listed : {sources, targets}) {
            if (listed.empty()) return {std::move(plan), status_t::invalid_argument_k};
            for (std::size_t position = 0; position != listed.size(); ++position) {
                if (!listed[position]) return {std::move(plan), status_t::invalid_argument_k};
                for (std::size_t earlier = 0; earlier != position; ++earlier)
                    if (listed[earlier] == listed[position]) return {std::move(plan), status_t::invalid_argument_k};
            }
        }
        std::size_t const parts = targets.size();
        if (config.shard_capacity || sources.size() != 1 || &(sources[0]->*core) != this || parts < 2)
            return {std::move(plan), status_t::unsupported_k};
        for (owner_type_* const into : targets)
            if (&(into->*core) == this) return {std::move(plan), status_t::config_mismatch_k};
        if (!claims_.closed()) return {std::move(plan), status_t::contended_k};
        split_t& split = plan.split_;
        if (failed(split.targets.reserve_exactly(parts)) || failed(split.parts.resize(parts)))
            return {std::move(plan), status_t::bad_alloc_k};
        for (owner_type_* const into : targets) {
            vram_shard_core& part = into->*core;
            if (part.device_.ordinal != device_.ordinal || part.kernel_ != kernel_ || !(part.geometry_ == geometry_) ||
                part.size())
                return {std::move(plan), status_t::config_mismatch_k};
            if (!part.claims_.closed()) return {std::move(plan), status_t::contended_k};
            split.targets.push_back_reserved(&part);
        }
        split.source = this;
        split.entries = rows;
        split.part_capacity = divide_round_up<std::size_t>(split.entries, parts);
        if (parts * split.part_capacity >= (std::numeric_limits<u32_t>::max)())
            return {std::move(plan), status_t::overflow_k};
        // Every target holds a full part's nodes, sentinel-filled, before the partition queues.
        stream_handle_t const stream = context.stream();
        for (vram_shard_core* const part : split.targets)
            if (status_t const grown = part->grow_nodes_(split.part_capacity, stream); failed(grown))
                return {std::move(plan), grown};
        auto const [kernels, kernels_status] = kernels_.kernels_(context);
        if (failed(kernels_status)) return {std::move(plan), kernels_status};
        // The rounds' mirrors take over the split's pinned bytes once the parts publish.
        status_t opened = carve_round_(plan.rounds_, context, split.part_capacity, config.refine.cache_bytes,
                                       split_layout_(split.entries, parts).host_bytes);
        if (succeeded(opened) && split.entries) opened = carve_split_(split, config.seed, context);
        if (succeeded(opened) && split.entries) opened = queue_partition_(*kernels, split, stream);
        if (failed(opened)) return {std::move(plan), opened};
        plan.rounds_.index_ = this;
        plan.moves_ = split.entries ? phase_t::partition_k : phase_t::done_k;
        if (!split.entries) plan.bind_target_();
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief One @ref plan_reshard driven to its end on the calling thread, one wait a phase.
     *
     *  @par Statuses:
     *  - Those of @ref plan_reshard, then @c device_error_k when a launch or a copy failed.
     */
    template <typename owner_type_>
    expected<refine_report_t> reshard(std::span<owner_type_* const> sources, std::span<owner_type_* const> targets,
                                      vram_shard_core owner_type_::* core, reshard_config_t config,
                                      traversal_context_t& context) noexcept {
        auto [plan, status] = plan_reshard(sources, targets, core, config, context, size());
        if (failed(status)) return {{}, status};
        do
            if (failed(plan.queue(context, config.refine)) || failed(plan.wait(context))) break;
        while (plan.advance());
        return plan.finish(context);
    }

    /**
     *  @brief Opens a health check of the rows published now on @p context 's stream: carves the
     *      columns and takes page-locked counters; queues nothing.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p context rides another device or cannot launch.
     *  - @c bad_alloc_k A column or the counters cannot be placed.
     */
    expected<health_plan_t> plan_health(traversal_context_t& context) noexcept {
        std::size_t const rows = size();
        health_plan_t plan {empty_round_scratch_(context)};
        if (!rows) return {std::move(plan), status_t::success_k};
        if (status_t const carved = carve_health_(plan, context, rows_shape_(rows)); failed(carved))
            return {std::move(plan), carved};
        plan.rebind(*this, static_cast<node_id_t>(rows));
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Measures reachability both ways from one seeded root, as the RAM health check does.
     *
     *  Reports the entries of the root's strong component, the depths, the in-link counts and the
     *  islands' components. Read-only on the graph, so concurrent insertions may leave fresh nodes
     *  uncounted. One @ref health_plan_t driven on the calling thread, waiting once per batch.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p context rides another device or cannot launch.
     *  - @c bad_alloc_k A column or the counters cannot be placed.
     *  - @c device_error_k A launch or a copy failed.
     */
    expected<reachability_t> health(traversal_context_t& context) noexcept {
        auto [plan, status] = plan_health(context);
        if (failed(status)) return {{}, status};
        while (plan.running())
            if (failed(plan.queue(context)) || failed(plan.wait(context)) || !plan.advance()) break;
        return plan.finish();
    }

    /** The bytes an executor's run must hold before a context carved over it can refine the rows
     *  published now under @p config: the context's counters, the round's columns and the restitch
     *  arena, then a split's partition scratch and its targets' rounds past two @p targets. */
    std::size_t traversal_memory_usage(refine_config_t config, std::size_t targets = 0) const noexcept {
        if (!dimensions() || !device_.multiprocessor_count) return 0;
        auto aligned = [](std::size_t bytes) noexcept { return round_up_to_multiple(bytes, default_alignment_k); };
        std::size_t const entries = targets > 1 ? divide_round_up<std::size_t>(size(), targets) : size();
        round_shape_t const shape = round_shape_(entries, config.cache_bytes);
        return aligned(sizeof(vram_telemetry_block_t)) + 2 * aligned(sizeof(u32_t)) +
               aligned(lay_out_round_(shape).device_bytes) + aligned(restitch_arena_bytes_()) +
               (targets > 1 ? aligned(split_layout_(size(), targets).device_bytes) : 0);
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

        /** One task of the plan: one bounce-sized batch of the rows under the watermark. */
        struct task_t {

            /** The task's position in the plan - what a fan-out hands to a stream. */
            std::size_t position = 0;

            /** Where the rows sit in the index, in its own ID space. */
            iota_view<node_id_t> index_nodes;

            /** Where they sit in the caller's buffer, counted from its first row. */
            iota_view<node_id_t> source_nodes;
        };

        /** Task @p position of the plan; an empty one past the last. */
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

        /** The tasks the save spans - one bounce-sized batch each. */
        indexed_range<save_plan_t> tasks() const noexcept {
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
         *  @brief Copies task @p task 's rows off the device into the plan's own columns.
         *
         *  The vectors go with the fragment padding stripped, then the neighbors with every ID past
         *  the watermark folded to the sentinel, then the norms where the columns take them. The
         *  task's last half is still copying when it returns, and finishes with the next task or in
         *  @c finish.
         *
         *  @param[in] task The task to copy, as @ref tasks hands them out.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The task is empty, or the columns cannot hold its rows.
         */
        status_t save(task_t const& task) noexcept {
            if (!index_) return status_t::uninitialized_k;
            // Shards' tasks interleave on one thread, so another shard's device may be current.
            if (failed(vendor_t::select(index_->api(), index_->device_specs().ordinal)))
                return status_t::device_error_k;
            iota_view<node_id_t> const rows = task.index_nodes;
            if (rows.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(node_id_t);
            std::size_t const first = task.source_nodes.first;
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
         *      it whole, so the missing tasks can still arrive.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k Some task's rows never arrived.
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
            fu::standard_atomic_ref<std::size_t> {covered}.store_add(rows.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Marks the last copies into each half of the bounce, born with the plan. */
        event_t crossed_[2];

        /** The rows whose copies sit in a half not yet on the host; empty when none do. */
        task_t pending_ {};

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
     *  it are refused here, so no task ever writes past them.
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
     *  @brief Serializes every row under the watermark into @p columns on the calling thread - the
     *      one-shot over @ref plan_save for a caller with no streams to fan the tasks out over.
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
        for (typename save_plan_t::task_t const task : plan.tasks())
            if (status_t const saved = plan.save(task); failed(saved)) return saved;
        return plan.finish();
    }

    /**
     *  @brief One load in flight: the rows it appends past every claimed one, the storage grown to
     *      hold them, and the tasks they span.
     *
     *  Each task copies on the stream from host or device columns, or from another index's
     *  storage, and @ref finish derives the norms no task carried before it publishes the rows.
     *
     *  Holds the claim for its whole life, so nothing else claims rows while it runs. A source's
     *  IDs are local to the rows it delivers, so every batch shifts them onto the appended range
     *  and folds every other ID onto the sentinel, on the device. Repeated plans append, which is
     *  how several parts go into one shard. A task of a part the vendor's reader registered reads
     *  into one of its two staging slots while the other's copies run, and a slot refills only
     *  once its event says the copies out of it finished; @c finish is the one full wait. A plan
     *  that dies before @c finish claims nothing and leaves the storage grown for the next one,
     *  its rows unfilled, so the next claim sentinel-fills over whatever its tasks wrote.
     */
    struct [[nodiscard]] load_plan_t {

        /** The stream every copy into the index and every kernel queue on, whose device every task
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

        /** One task of the plan: one batch of the appended nodes. */
        struct task_t {

            /** The task's position in the plan - what a fan-out hands to a stream. */
            std::size_t position = 0;

            /** Where the rows go in the index, in its own ID space, from @c base. */
            iota_view<node_id_t> index_nodes;

            /** Where they come from in the caller's buffer, from its first row. */
            iota_view<node_id_t> source_nodes;
        };

        /** Task @p position of the plan; an empty one past the last. */
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

        /** The tasks the append spans - one batch each. */
        indexed_range<load_plan_t> tasks() const noexcept {
            return {this, index_ && rows ? divide_round_up(rows, batch_rows()) : 0};
        }

        /** Rows one batch carries - a slab, or every row when fewer. */
        std::size_t batch_rows() const noexcept { return min_of(rows, power_of_two<std::size_t>(batch_rows_log2_k)); }

        /**
         *  @brief Copies task @p task 's rows out of @p columns on the plan's stream, through
         *      the vendor reader's staging slot where it registered their part, and their keys
         *      into the host's.
         *
         *  The vectors go into the fragment-padded matrix rows, whose fresh tails arrive zeroed,
         *  and their norms where the columns carry them, then the neighbors, which
         *  @c launch_rebase_links_ folds on the plan's stream, every ID inside @p source_range onto
         *  the appended range and every other onto the sentinel. Columns stay alive until
         *  @c finish, whose one wait covers every copy. A sharded plan drives it one part's slice
         *  at a time, its task naming the rows in that part's row and ID spaces.
         *
         *  @param[in] task The task to fill; its source nodes address @p columns.
         *  @param[in] columns The rows' columns, one part's under a sharded plan.
         *  @param[in] source_range The ID space those rows speak.
         *  @param[in] key_of Row @c r of the task's source nodes takes the key <tt>key_of(r)</tt>.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The task is empty, @p columns cannot hold its rows, or the
         *    rebased IDs pass the plan.
         *  - @c io_k A read into a staging slot ended short, which never restarts on the host.
         */
        template <typename key_of_type_>
        status_t load(task_t const& task, columns_view_t columns, iota_view<node_id_t> source_range,
                      key_of_type_&& key_of) noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (failed(vendor_t::select(index_->api(), index_->device_specs().ordinal)))
                return status_t::device_error_k;
            iota_view<node_id_t> const range = task.index_nodes;
            if (range.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(node_id_t);
            std::size_t const first = task.source_nodes.first;
            bytes_view_t vectors_rows = flat_rows(columns.vectors, first, range.size(), scalar_row_bytes);
            bytes_view_t neighbors_rows = flat_rows(columns.neighbors, first, range.size(), neighbors_row_bytes);
            bytes_view_t norms_rows = flat_rows(columns.norms, first, range.size(), sizeof(norms_t));
            if (vectors_rows.empty() || neighbors_rows.empty() || norms_rows.empty() != columns.norms.empty())
                return status_t::config_mismatch_k;
            // The part's rows go to this task's own index rows, so an ID's shift follows from
            // the task rather than from the plan's base - the two agree for a one-part plan.
            node_id_t const shift = static_cast<node_id_t>(range.first - task.source_nodes.first + source_range.first);
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
                fu::standard_atomic_ref<std::size_t> {covered_norms}.store_add(range.size(), std::memory_order_acq_rel);
            fu::standard_atomic_ref<std::size_t> {covered}.store_add(range.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Files every appended row under the key its task copied, which @ref finish then uploads
         *  and makes visible - the one writer of the key map, once every task finished and before
         *  the finish; with a task missing it files nothing. */
        void publish_keys() noexcept {
            if (!index_ || fu::standard_atomic_ref<std::size_t> {covered}.load(std::memory_order_acquire) != rows)
                return;
            index_->publish_keys_({base, static_cast<node_id_t>(base + rows)});
        }

        /**
         *  @brief The same copy out of @p source_shard 's own storage rather than host columns, as
         *      a pack into one target drives it, one source shard at a time, its task naming the
         *      rows in that shard's row and ID spaces.
         *
         *  Both matrices keep fragment-padded rows of one geometry, so every column moves as one
         *  copy on the plan's stream, from another device's storage as well once its sharded index
         *  mapped it for this one; storage this device cannot reach is refused. The neighbors then
         *  rebase on the plan's stream, and the source's keys go into the host key column. Nothing
         *  waits, so @p source_shard must stay quiescent until the caller drains that stream.
         *
         *  @param[in] task The task to fill; its source nodes address @p source_shard.
         *  @param[in] source_shard The shard the rows come from, read through its core.
         *  @param[in] source_range The ID space the source's rows speak.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k Another geometry, rows outside the plan, or storage on a
         *    device this one cannot reach.
         */
        template <typename shard_type_>
        status_t load(task_t const& task, shard_type_ const& source_shard, iota_view<node_id_t> source_range) noexcept {
            vram_shard_core const& source = source_shard.core();
            if (!index_) return status_t::uninitialized_k;
            int const device = index_->device_specs().ordinal;
            if (failed(vendor_t::select(index_->api(), device))) return status_t::device_error_k;
            iota_view<node_id_t> const range = task.index_nodes;
            bool const normed = index_->stores_norms() && source.stores_norms();
            if (range.empty() || range.size() != task.source_nodes.size() || range.first < base ||
                range.bound > base + rows || task.source_nodes.bound > source.size() ||
                source.dimensions() != index_->dimensions() || source.connectivity() != index_->connectivity() ||
                !source.vectors_matrix_.shared_with(device) || !source.neighbors_matrix_.shared_with(device) ||
                (normed && !source.norms_column_.shared_with(device)))
                return status_t::config_mismatch_k;
            node_id_t const shift = range.first - task.source_nodes.first + source_range.first;
            if (shift + source_range.size() > base + rows) return status_t::config_mismatch_k;
            std::size_t const row_bytes = index_->row_bytes();
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(node_id_t);
            std::size_t const count = range.size(), into = range.first, from = task.source_nodes.first;
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
            if (normed)
                fu::standard_atomic_ref<std::size_t> {covered_norms}.store_add(count, std::memory_order_acq_rel);
            fu::standard_atomic_ref<std::size_t> {covered}.store_add(count, std::memory_order_acq_rel);
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
         *  batch kept; any other refusal leaves it whole, so the missing tasks can still arrive.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent, or @ref status keeps it.
         *  - @c config_mismatch_k Some task's rows never arrived, or @ref status keeps it.
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
                // wait finishes every task, the norms and the count before the watermark turns.
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

        /** Reads tasks of a registered part into its staging slots; dies after the destructor's
         *  wait, so no copy still reads a slot it frees. */
        typename kernels_type_::part_reader_t reader_;

        /** Tasks queued so far, alternating between the slots, whose sources a dying plan
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
     *  Its tasks source the rows from host or device columns, as a sharded plan drives its parts,
     *  or from other indexes, as a pack does; a plan reading a part the vendor's reader registers
     *  opens its staging slots first. Refuses an unreserved index, a kept failure, a batch or load
     *  in flight, and rows past the ID space.
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
        // A closed word is the caller's, who refines or reshards around the load; an open one is
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
        if (status_t const opened = plan.reader_.open(stream, sources, plan.batch_rows(), plan.tasks().size(),
                                                      column_row_bytes);
            failed(opened))
            return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Appends the whole source on the calling thread - the one-shot over @ref plan_load for
     *      a caller with no streams to fan the tasks out over.
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
        for (typename load_plan_t::task_t const task : plan.tasks())
            if (status_t const loaded = plan.load(task, columns, source_range,
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
        byte_t* exported = bytes_of(attended, values).data();
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

    /** Zeroed round counters in a slot of @p context 's stream, or null when none can pin. */
    static round_counters_ptr_t take_round_counters_(traversal_context_t& context) noexcept {
        typename vendor_t::pinned_slots_t* const pinned_slots = context.allocator_.pinned_slots();
        void* const slot = pinned_slots ? pinned_slots->take() : nullptr;
        return {slot ? new (slot) round_counters_t {} : nullptr, round_counters_return_t {pinned_slots}};
    }

    /** Scratch holding nothing yet over @p context 's allocator - what a refused plan keeps. */
    static round_scratch_t empty_round_scratch_(traversal_context_t& context) noexcept {
        using bytes_ptr_t = typename traversal_context_t::bytes_ptr_t;
        return round_scratch_t {bytes_ptr_t {nullptr, typename bytes_ptr_t::deleter_type {context.allocator_, 0}}};
    }

    /** Bytes one row takes staged in shared memory: its vector, and its norms where the metric
     *  keeps them. */
    std::size_t staged_row_bytes_() const noexcept { return row_bytes() + (stores_norms() ? sizeof(norms_t) : 0); }

    /** Hubs one gather draws at most under @p cache_bytes: half of it in staged rows, as the RAM
     *  scorer gives one pack half a core's cache, never below one. */
    std::size_t round_hubs_per_draw_(std::size_t cache_bytes) const noexcept {
        return max_of<std::size_t>(cache_bytes / (2 * staged_row_bytes_()), 1);
    }

    /** The most entries a neighborhood holds: the rows the other half keeps, never below one row of
     *  mates and the entry itself, which a restitch needs to fill a row from mates alone; past what
     *  shared memory holds, the restitch reads its rows in place. */
    std::size_t round_capacity_(std::size_t cache_bytes, std::size_t hubs_per_draw) const noexcept {
        std::size_t const rest = cache_bytes - min_of(cache_bytes, hubs_per_draw * staged_row_bytes_());
        return max_of(rest / staged_row_bytes_(), connectivity() + 1);
    }

    /** A round's shape over @p entries entries under @p cache_bytes, zero for this device's figure,
     *  never more than it: a neighborhood over capacity holds more entries than one, so they number
     *  fewer than the entries over the capacity, the hubs of a gather inside them at most twice
     *  that and one each, and a split's starts the room @c split_room counts. */
    round_shape_t round_shape_(std::size_t entries, std::size_t cache_bytes) const noexcept {
        std::size_t const device_bytes = device_.cache_bytes_per_core();
        std::size_t const budget = cache_bytes ? min_of(cache_bytes, device_bytes) : device_bytes;
        round_shape_t shape;
        shape.entries = entries;
        shape.hubs_per_draw = round_hubs_per_draw_(budget);
        shape.capacity = round_capacity_(budget, shape.hubs_per_draw);
        std::size_t const wide = divide_round_up(entries, shape.capacity);
        shape.offsets = split_room(entries, shape.hubs_per_draw);
        shape.hubs = max_of(shape.hubs_per_draw, 3 * wide + 1);
        shape.runs = max_of<std::size_t>(wide, 1);
        return shape;
    }

    /** The shape of a health check alone over @p entries entries: no round's columns at all. */
    static round_shape_t rows_shape_(std::size_t entries) noexcept { return {entries, 0, 0, 0, 0, 0}; }

    /** Blocks of a restitch, one resident per multiprocessor, each taking the whole tile budget. */
    u32_t restitch_blocks_() const noexcept {
        return device_.multiprocessor_count * kernels_type_::resident_blocks_(vram_kernel_t::restitch_k, 0);
    }

    /** Bytes the restitch warps' arena takes: their selection scratch, where shared memory cannot
     *  fit it, and none otherwise. */
    std::size_t restitch_arena_bytes_() const noexcept {
        u32_t const scratch_bytes = build_scratch_bytes_per_warp(reselection_nodes_());
        if (kernels_.build_scratch_fits(*this, scratch_bytes)) return 0;
        return std::size_t {restitch_blocks_()} * vram_warps_per_block_k *
               gpu_arena_shape(0, 0, scratch_bytes).template slice_bytes<node_id_t>();
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
     *  checks and the heal read the islands', then every restitch block's slice of mates. The
     *  mirrors follow the round's columns in the same order.
     */
    round_layout_t lay_out_round_(round_shape_t const& shape, byte_t* device = nullptr,
                                  round_scratch_t* scratch = nullptr, byte_t* host = nullptr,
                                  round_mirrors_t* mirrors = nullptr) const noexcept {
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
            std::size_t const run_items = take((shape.runs + 1) * sizeof(u32_t));
            std::size_t const entry_neighborhoods = take(entries * sizeof(u16_t));
            return std::array<std::size_t, 6> {neighborhood_entries, neighborhood_offsets, hub_nodes, runs, run_items,
                                               entry_neighborhoods};
        };
        std::array<std::size_t, 6> const round_columns = take_round_columns();
        cursor = max_of(cursor, islands_end);
        std::size_t const mates = take(std::size_t {restitch_blocks_()} * shape.capacity * connectivity() *
                                       gpu_scored_node_bytes<node_id_t>);
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
            scratch->mates = device + mates;
        }
        cursor = 0;
        std::array<std::size_t, 6> const mirrored = take_round_columns();
        layout.host_bytes = cursor;
        if (mirrors) {
            mirrors->neighborhood_entries = reinterpret_cast<u32_t*>(host + mirrored[0]);
            mirrors->neighborhood_offsets = reinterpret_cast<u32_t*>(host + mirrored[1]);
            mirrors->hub_nodes = reinterpret_cast<u32_t*>(host + mirrored[2]);
            mirrors->runs = reinterpret_cast<vram_neighborhood_run_t*>(host + mirrored[3]);
            mirrors->run_items = reinterpret_cast<u32_t*>(host + mirrored[4]);
            mirrors->entry_neighborhoods = reinterpret_cast<u16_t*>(host + mirrored[5]);
        }
        return layout;
    }

    /** Carves @p plan 's columns for a check of @p shape from @p context 's allocator and takes its
     *  page-locked counters. */
    status_t carve_health_(health_plan_t& plan, traversal_context_t& context, round_shape_t const& shape) noexcept {
        if (!vectors_matrix_.reserved_bytes()) return status_t::uninitialized_k;
        if (context.device() != device_.ordinal || !context) return status_t::config_mismatch_k;
        std::size_t const bytes = lay_out_round_(shape).device_bytes;
        round_scratch_t& scratch = plan.columns_;
        scratch.bytes.reset(context.allocator_.allocate(bytes));
        scratch.bytes.get_deleter().count = bytes;
        if (!scratch.bytes) return status_t::bad_alloc_k;
        lay_out_round_(shape, scratch.bytes.get(), &scratch);
        plan.counters_ = take_round_counters_(context);
        return plan.counters_ ? status_t::success_k : status_t::bad_alloc_k;
    }

    /** Carves everything @p plan 's rounds over @p entries entries keep under @p cache_bytes: the
     *  check's and the round's columns, the page-locked mirrors, holding at least @p pinned_floor
     *  bytes for a split's own, and the restitch arena where shared memory cannot fit its warps'
     *  scratch; picks what each launch stages in shared memory and zeroes the telemetry once, so
     *  every phase adds to it. */
    status_t carve_round_(refine_plan_t& plan, traversal_context_t& context, std::size_t entries,
                          std::size_t cache_bytes, std::size_t pinned_floor) noexcept {
        round_shape_t const shape = round_shape_(entries, cache_bytes);
        if (status_t const carved = carve_health_(plan.health, context, shape); failed(carved)) return carved;
        plan.shape_ = shape;
        growing_buffer<scored_node_t, typename vendor_t::pinned_allocator_t>& pinned = context.results.entries_;
        std::size_t const host_bytes = max_of(lay_out_round_(shape).host_bytes, pinned_floor);
        if (failed(pinned.resize_uninitialized(divide_round_up(host_bytes, sizeof(scored_node_t)))))
            return status_t::bad_alloc_k;
        lay_out_round_(shape, nullptr, nullptr, bytes_of(pinned.data(), pinned.size()).data(), &plan.mirrors_);
        // A restitch warp keeps its selection scratch in the block's shared memory when it fits,
        // aliasing the neighborhood's staged rows, which the scoring is done with by then.
        std::size_t const staged_row = staged_row_bytes_();
        u32_t const scratch_bytes = build_scratch_bytes_per_warp(reselection_nodes_());
        bool const scratch_shared = kernels_.build_scratch_fits(*this, scratch_bytes);
        std::size_t const warps_scratch = scratch_shared ? std::size_t {vram_warps_per_block_k} * scratch_bytes : 0;
        std::size_t const mates_bytes = kernels_type_::mates_shared_bytes_(shape.capacity * staged_row);
        bool const staged = kernels_.round_scratch_fits(*this, max_of(mates_bytes, warps_scratch));
        plan.restitch_staged_bytes_ = staged ? static_cast<u32_t>(mates_bytes) : 0u;
        plan.restitch_shared_bytes_ = static_cast<u32_t>(
            max_of<std::size_t>(plan.restitch_staged_bytes_, warps_scratch));
        plan.restitch_arena_shape_ = gpu_arena_shape(0, 0, scratch_shared ? 0u : scratch_bytes);
        std::size_t const gather_staged = kernels_type_::gather_shared_bytes_(shape.hubs_per_draw * staged_row);
        plan.gather_staged_bytes_ = kernels_.round_scratch_fits(*this, gather_staged)
                                        ? static_cast<u32_t>(gather_staged)
                                        : 0u;
        if (!scratch_shared && !context.ensure_arena(restitch_arena_bytes_())) return status_t::bad_alloc_k;
        return failed(
                   vendor_t::fill(api(), context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), context.stream()))
                   ? status_t::device_error_k
                   : status_t::success_k;
    }

    /** The record every launch of @p plan 's check reads, over this index's neighbors. */
    vram_health_t health_record_(health_plan_t const& plan) const noexcept {
        vram_health_t health = plan.columns_.health;
        health.neighbors = neighbor_rows_();
        health.rows_count = plan.visible;
        health.connectivity = static_cast<u32_t>(connectivity());
        return health;
    }

    /** Queues the clearing of what one root's sweep of @p plan fills, then @p health 's root marked
     *  both ways at depth zero, where the depths' tally starts and both directions count it. */
    status_t queue_health_start_(health_plan_t const& plan, vram_health_t const& health,
                                 stream_handle_t stream) noexcept {
        std::size_t const visible = plan.visible, words = plan.columns_.words;
        node_id_t const root = plan.root_;
        u32_t const bit = 1u << (root & 31u);
        bool const filled =
            succeeded(vendor_t::fill(api(), health.reached.words, 0, 3 * words * sizeof(u32_t), stream)) &&
            succeeded(vendor_t::fill(api(), health.depths_from_root, 0xFF, visible, stream)) &&
            succeeded(vendor_t::fill(api(), health.depths_to_root, 0xFF, visible, stream)) &&
            succeeded(vendor_t::fill(api(), health.in_links, 0, visible * sizeof(u32_t), stream)) &&
            succeeded(vendor_t::fill(api(), health.shallower_in_links, 0, visible * sizeof(u32_t), stream)) &&
            succeeded(vendor_t::fill(api(), health.depth_entries, 0, 256 * sizeof(u32_t), stream)) &&
            succeeded(vendor_t::fill(api(), health.counters, 0, offsetof(round_counters_t, healed), stream)) &&
            succeeded(vendor_t::fill_words(api(), health.reached.words + root / 32, bit, 1, stream)) &&
            succeeded(vendor_t::fill_words(api(), health.reaches_root.words + root / 32, bit, 1, stream)) &&
            succeeded(vendor_t::fill(api(), health.depths_from_root + root, 0, 1, stream)) &&
            succeeded(vendor_t::fill(api(), health.depths_to_root + root, 0, 1, stream)) &&
            succeeded(vendor_t::fill_words(api(), health.depth_entries, 1, 1, stream)) &&
            succeeded(vendor_t::fill_words(api(), &health.counters->swept, 2, 1, stream));
        return filled ? status_t::success_k : status_t::device_error_k;
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
        auto const [kernels, kernels_status] = kernels_.kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        stream_handle_t const stream = context.stream();
        vram_arguments_t arguments {};
        vram_health_t& health = arguments.health;
        health = health_record_(plan);
        round_counters_t* const counters = health.counters;
        u32_t const striding = device_.multiprocessor_count *
                               kernels_type_::resident_blocks_(vram_kernel_t::health_expand_k, 0);
        u32_t const first = plan.next_pass_, bound = first ? 2 * first : 1;
        status_t queued = status_t::success_k;
        auto zeroed = [&](u32_t& word) noexcept {
            if (succeeded(queued) && failed(vendor_t::fill(api(), &word, 0, sizeof(u32_t), stream)))
                queued = status_t::device_error_k;
        };
        auto launch = [&](vram_kernel_t kernel, u32_t blocks) noexcept {
            if (succeeded(queued)) queued = kernels_.launch_(*kernels, kernel, 0, blocks, 0, stream, arguments);
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
            failed(vendor_t::copy(api(), plan.counters_.get(), counters, sizeof(round_counters_t), stream)))
            queued = status_t::device_error_k;
        return queued;
    }

    /** Queues the round's start on @p context 's stream, its counters and changed bits zeroed, then
     *  the distance-free heal of every island component @p plan named, on one thread. */
    status_t queue_heal_(traversal_context_t& context, health_plan_t const& plan) noexcept {
        auto const [kernels, kernels_status] = kernels_.kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        stream_handle_t const stream = context.stream();
        vram_arguments_t arguments {};
        arguments.health = health_record_(plan);
        std::size_t const round_offset = offsetof(round_counters_t, healed);
        if (failed(vendor_t::fill(api(), reinterpret_cast<byte_t*>(arguments.health.counters) + round_offset, 0,
                                  sizeof(round_counters_t) - round_offset, stream)) ||
            failed(
                vendor_t::fill(api(), arguments.health.changed.words, 0, plan.columns_.words * sizeof(u32_t), stream)))
            return status_t::device_error_k;
        if (!plan.counters_->islands) return status_t::success_k;
        return kernels_.launch_(*kernels, vram_kernel_t::heal_islands_k, 0, 1, 0, stream, arguments);
    }

    /** Queues one gather of @p plan on @p context 's stream: its hubs, runs and items uploaded from
     *  the mirrors, the entries in neighborhood order too past the first gather, then the launch
     *  over a block a multiprocessor, then the labels copied down for the barrier to order. */
    status_t queue_gather_(traversal_context_t& context, refine_plan_t const& plan) noexcept {
        auto const [kernels, kernels_status] = kernels_.kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        stream_handle_t const stream = context.stream();
        round_scratch_t const& scratch = plan.health.columns_;
        round_mirrors_t const& mirrors = plan.mirrors_;
        std::size_t const visible = plan.health.visible;
        bool const nested = plan.neighborhoods_ != 0;
        if (failed(vendor_t::copy(api(), scratch.hub_nodes, mirrors.hub_nodes, plan.hubs_ * sizeof(u32_t), stream)) ||
            failed(vendor_t::copy(api(), scratch.runs, mirrors.runs, plan.runs_ * sizeof(vram_neighborhood_run_t),
                                  stream)) ||
            failed(vendor_t::copy(api(), scratch.run_items, mirrors.run_items, (plan.runs_ + 1) * sizeof(u32_t),
                                  stream)) ||
            (nested && failed(vendor_t::copy(api(), scratch.neighborhood_entries, mirrors.neighborhood_entries,
                                             visible * sizeof(u32_t), stream))) ||
            failed(vendor_t::fill(api(), &scratch.health.counters->claimed, 0, sizeof(u32_t), stream)))
            return status_t::device_error_k;
        vram_arguments_t arguments {};
        vram_gather_neighborhoods_t& gather = arguments.gather_neighborhoods;
        gather.storage = storage();
        gather.entries = nested ? scratch.neighborhood_entries : nullptr;
        gather.hub_nodes = scratch.hub_nodes;
        gather.runs = scratch.runs;
        gather.run_items = scratch.run_items;
        gather.runs_count = static_cast<u32_t>(plan.runs_);
        gather.staged_bytes = plan.gather_staged_bytes_;
        gather.entry_neighborhoods = scratch.entry_neighborhoods;
        gather.assigned_items = &scratch.health.counters->claimed;
        gather.telemetry = context.telemetry_.get();
        u32_t const blocks = device_.multiprocessor_count *
                             kernels_type_::resident_blocks_(vram_kernel_t::gather_neighborhoods_k, 0);
        if (status_t const launched = kernels_.launch_(*kernels, vram_kernel_t::gather_neighborhoods_k, 0, blocks,
                                                       gather.staged_bytes, stream, arguments);
            failed(launched))
            return launched;
        return failed(vendor_t::copy(api(), mirrors.entry_neighborhoods, scratch.entry_neighborhoods,
                                     visible * sizeof(u16_t), stream))
                   ? status_t::device_error_k
                   : status_t::success_k;
    }

    /** Queues @p plan 's restitch on @p context 's stream: the entries in neighborhood order and
     *  the starts uploaded from the mirrors, then the launch over a block a multiprocessor, each
     *  claiming neighborhoods in turn. */
    status_t queue_restitch_(traversal_context_t& context, refine_plan_t const& plan) noexcept {
        auto const [kernels, kernels_status] = kernels_.kernels_(context);
        if (failed(kernels_status)) return kernels_status;
        stream_handle_t const stream = context.stream();
        round_scratch_t const& scratch = plan.health.columns_;
        round_mirrors_t const& mirrors = plan.mirrors_;
        if (failed(vendor_t::copy(api(), scratch.neighborhood_entries, mirrors.neighborhood_entries,
                                  std::size_t {plan.health.visible} * sizeof(u32_t), stream)) ||
            failed(vendor_t::copy(api(), scratch.neighborhood_offsets, mirrors.neighborhood_offsets,
                                  (plan.neighborhoods_ + 1) * sizeof(u32_t), stream)) ||
            failed(vendor_t::fill(api(), &scratch.health.counters->claimed, 0, sizeof(u32_t), stream)))
            return status_t::device_error_k;
        gpu_arena_shape_t const arena_shape = plan.restitch_arena_shape_;
        byte_t* const arena = arena_shape.scratch_bytes ? context.ensure_arena(restitch_arena_bytes_()) : nullptr;
        if (arena_shape.scratch_bytes && !arena) return status_t::bad_alloc_k;
        u32_t const selection_nodes = reselection_nodes_();
        vram_arguments_t arguments {};
        vram_restitch_t& restitch = arguments.restitch;
        restitch.storage = storage();
        restitch.neighborhood_entries = scratch.neighborhood_entries;
        restitch.neighborhood_offsets = scratch.neighborhood_offsets;
        restitch.neighborhoods_count = static_cast<u32_t>(plan.neighborhoods_);
        restitch.capacity = static_cast<u32_t>(plan.shape_.capacity);
        restitch.staged_bytes = plan.restitch_staged_bytes_;
        restitch.mates = scratch.mates;
        restitch.selection_nodes = selection_nodes;
        restitch.scratch_bytes_per_warp = arena_shape.scratch_bytes ? 0u
                                                                    : build_scratch_bytes_per_warp(selection_nodes);
        restitch.arena = arena;
        restitch.arena_shape = arena_shape;
        restitch.health = health_record_(plan.health);
        restitch.assigned = &scratch.health.counters->claimed;
        restitch.telemetry = context.telemetry_.get();
        return kernels_.launch_(*kernels, vram_kernel_t::restitch_k, 0, restitch_blocks_(), plan.restitch_shared_bytes_,
                                stream, arguments);
    }

    /** Bytes a split of @p nodes nodes into @p parts parts carves on the device and pins on the
     *  host, and where each places the targets' columns, then where the pinned bytes place the
     *  roots and the placed count, as @c split_t lays them out. */
    struct split_layout_t {
        std::size_t device_bytes = 0, device_columns = 0;
        std::size_t host_bytes = 0, host_columns = 0, host_roots = 0, host_assigned = 0;
    };
    static split_layout_t split_layout_(std::size_t nodes, std::size_t parts) noexcept {
        auto aligned = [](std::size_t bytes) noexcept { return round_up_to_multiple(bytes, default_alignment_k); };
        std::size_t const words = divide_round_up<std::size_t>(nodes, 32);
        split_layout_t bytes;
        bytes.device_columns = aligned((2 * nodes + 2 * words + 3 * parts + 2) * sizeof(u32_t));
        bytes.device_bytes = bytes.device_columns + 4 * parts * sizeof(u32_t*);
        bytes.host_columns = aligned(nodes * sizeof(u32_t));
        bytes.host_roots = bytes.host_columns + 4 * parts * sizeof(u32_t*);
        bytes.host_assigned = bytes.host_roots + parts * sizeof(u32_t);
        bytes.host_bytes = bytes.host_assigned + sizeof(u32_t);
        return bytes;
    }

    /** Where a split's device scratch keeps each array: each node's part, which the scatter reads
     *  back as its slot, the round's proposals, @b [nodes] each, the two frontiers, one bit per
     *  node each, then per part its size and its round count, the nodes the last two rounds
     *  placed, by the round number's parity, the parts' roots, and the targets' neighbors,
     *  vectors, norms and keys. */
    struct split_scratch_t {
        u32_t* node_parts = nullptr;
        u32_t* proposals = nullptr;
        vram_bitset_t frontier {}, next_frontier {};
        u32_t* part_sizes = nullptr;
        u32_t* round_counts = nullptr;
        u32_t* placed = nullptr;
        u32_t* roots = nullptr;
        u32_t** part_columns = nullptr;
    };

    /** The arrays of @p split 's device scratch, read off its bytes. */
    static split_scratch_t split_scratch_(split_t const& split) noexcept {
        std::size_t const entries = split.entries, parts = split.targets.size();
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
        carved.part_columns = reinterpret_cast<u32_t**>(split.device_bytes.get() +
                                                        split_layout_(entries, parts).device_columns);
        return carved;
    }

    /** Carves @p split 's device scratch from @p context 's allocator, borrows its pinned results,
     *  which the rounds' mirrors already sized, as the host bytes, and queues the upload of the
     *  targets' columns and of the roots that @p seed picks. */
    status_t carve_split_(split_t& split, u64_t seed, traversal_context_t& context) noexcept {
        static_assert(sizeof(norms_t) % sizeof(u32_t) == 0 && sizeof(key_t) % sizeof(u32_t) == 0);
        std::size_t const entries = split.entries, parts = split.targets.size();
        split_layout_t const bytes = split_layout_(entries, parts);
        split.device_bytes.reset(context.allocator_.allocate(bytes.device_bytes));
        split.device_bytes.get_deleter().count = bytes.device_bytes;
        if (!split.device_bytes) return status_t::bad_alloc_k;
        growing_buffer<scored_node_t, typename vendor_t::pinned_allocator_t>& pinned = context.results.entries_;
        if (pinned.size() * sizeof(scored_node_t) < bytes.host_bytes) return status_t::bad_alloc_k;
        split.host_bytes = bytes_of(pinned.data(), pinned.size()).data();
        u32_t** const columns = reinterpret_cast<u32_t**>(split.host_bytes + bytes.host_columns);
        u32_t* const roots = reinterpret_cast<u32_t*>(split.host_bytes + bytes.host_roots);
        for (std::size_t part = 0; part != parts; ++part) {
            vram_shard_core const& target = *split.targets[part];
            columns[part] = target.neighbor_rows_();
            columns[parts + part] = reinterpret_cast<u32_t*>(target.vectors_rows_());
            columns[2 * parts + part] = reinterpret_cast<u32_t*>(target.norms_column_.base());
            columns[3 * parts + part] = reinterpret_cast<u32_t*>(target.device_keys_());
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
        std::size_t const parts = split.targets.size();
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
     *  unknown, the frontiers, the parts' counters and the rounds' counts empty, then the roots. */
    status_t queue_partition_(kernel_table_t const& kernels, split_t& split, stream_handle_t stream) noexcept {
        std::size_t const entries = split.entries, parts = split.targets.size();
        std::size_t const words = divide_round_up<std::size_t>(entries, 32);
        split_scratch_t const carved = split_scratch_(split);
        // Each run lies back to back.
        if (failed(vendor_t::fill(api(), carved.node_parts, 0xFF, 2 * entries * sizeof(u32_t), stream)) ||
            failed(
                vendor_t::fill(api(), carved.frontier.words, 0, (2 * words + 2 * parts + 2) * sizeof(u32_t), stream)))
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
            split.host_bytes + split_layout_(split.entries, split.targets.size()).host_assigned);
        return failed(vendor_t::copy(api(), assigned_count, carved.placed + ((2 * first_round - 1) & 1u), sizeof(u32_t),
                                     stream))
                   ? status_t::device_error_k
                   : status_t::success_k;
    }

    /** Whether the last growth round of @p split placed anything, so more rounds follow. */
    static bool placing_(split_t const& split) noexcept {
        std::size_t const host_assigned = split_layout_(split.entries, split.targets.size()).host_assigned;
        return *reinterpret_cast<u32_t const*>(split.host_bytes + host_assigned) != 0;
    }

    /** Queues the end of @p split 's partition once no round places anything: the nodes no round
     *  reached placed in the smallest parts, and the copy of the node parts. */
    status_t queue_partition_islands_(kernel_table_t const& kernels, split_t& split, stream_handle_t stream) noexcept {
        vram_arguments_t const arguments = split_arguments_(split);
        if (status_t const launched = kernels_.launch_(kernels, vram_kernel_t::partition_islands_k, 0, 1, 0, stream,
                                                       arguments);
            failed(launched))
            return launched;
        return failed(vendor_t::copy(api(), split.host_bytes, split_scratch_(split).node_parts,
                                     split.entries * sizeof(u32_t), stream))
                   ? status_t::device_error_k
                   : status_t::success_k;
    }

    /** Queues @p split 's scatter: the assigned slots and the targets' published counts up, then
     *  every node's neighbors, vector, norms and key into its target. */
    status_t queue_scatter_(kernel_table_t const& kernels, split_t const& split, stream_handle_t stream) noexcept {
        if (failed(vendor_t::copy(api(), split_scratch_(split).node_parts, split.host_bytes,
                                  split.entries * sizeof(u32_t), stream)))
            return status_t::device_error_k;
        for (std::size_t index = 0; index != split.targets.size(); ++index)
            if (failed(vendor_t::fill_words(api(), split.targets[index]->published_rows_(),
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
        rmscast.epsilon = (std::numeric_limits<f32_t>::min)();
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

    /** The claim word, closed while a refine, a reshard, a load or a save of the sharded index
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
