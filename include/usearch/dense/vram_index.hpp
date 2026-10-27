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
#include <bitset>      // `std::bitset` - the devices a plural shares the storage with
#include <limits>      // `std::numeric_limits` - the unit-RMS import's epsilon
#include <memory>      // `std::unique_ptr` - an audit's landing
#include <span>        // `std::span` - the parts' bases, the counts of a graph product
#include <type_traits> // `std::is_same_v`
#include <utility>     // `std::exchange` - a moved-from plan forgets its index

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the published words
#include <forkunion/types.hpp>   // `fu::standard_yield_t` - the watermark's turn

#include <numkong/cast/serial.h> // `nk_cast_plan_serial_` - the device casts' plans
#include <numkong/each/serial.h> // `nk_each_rmsnorm_arguments_t` - the unit-RMS import's record
#include <numkong/tensor.hpp>    // `nk::matrix_span` - the batches' views
#include <numkong/types.hpp>     // `nk::type_for` - the stored scalar of a dtype
#include <numkong/vector.hpp>

#include "usearch/utilities/types.hpp"
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
    u32_t selection_rows = 0;
    node_id_t first_id = 0;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    u32_t expansion = 0;
    u32_t epoch_base = 0;
    u32_t* published_rows = nullptr;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_rows = nullptr;
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
    u32_t filling_rows = 0;
};

/** One refinement launch, mirroring the shared refine view. */
struct vram_refine_t {
    vram_flat_storage_t storage;
    u32_t* shadow_neighbors = nullptr;
    u32_t end_id = 0;
    u32_t scratch_bytes_per_warp = 0;
    u32_t selection_rows = 0;
    node_id_t first_id = 0;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    u32_t expansion = 0;
    u32_t epoch_base = 0;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_rows = nullptr;
    u32_t const* node_ids = nullptr;
    u32_t sampling_bound = 0;
    u32_t const* part_bases = nullptr;
    u32_t parts = 0;
    u32_t round = 0;
    u32_t* link_heads = nullptr;
    u32_t* link_next = nullptr;
    vram_bitset_t walked;
};

/** One shard of a fold bin as the bin's storage reads it on every device of the bin: its vectors
 *  and norms, its fold rows, and the first row and the rows it takes in the bin's dense ID
 *  space, which the target shard lands it at. */
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
    u32_t selection_rows = 0;
    vram_fold_phase_t phase = vram_fold_phase_t::reselect_k;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_rows = nullptr;
};

/** A search's landing, mirroring the shared search sink. */
struct vram_search_sink_t {
    node_id_t* result_ids = nullptr;
    distance_t* result_distances = nullptr;
    u32_t count = 0;
    nk_kernel_kind_t kernel = nk_kernel_dot_k;
};

/** A graph product's landing, mirroring the shared multiply sink; the products are the kit's
 *  dots. */
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
    vram_bitset_t reached, frontier, next_frontier;
    u32_t* counter = nullptr;

    /** The count of the sweep before; zero sends a sweep queued past convergence home at entry. */
    u32_t const* previous_counter = nullptr;

    /** The levels that expanded anything, each counted by its first expansion. */
    u32_t* levels = nullptr;
    u32_t* island_ids = nullptr;
    u32_t islands_count = 0;
    u32_t seed_count = 0;
    u32_t* in_links = nullptr;
    u32_t* reconnected = nullptr;
};

/** An attend's landing and the value lookup its walks resolve rows through, carried as the lookup's
 *  own bytes; the sharded gather reads the landing as @p shards blocks of @p rows rows. */
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
    vram_attend_t attend;
    struct {
        node_id_t const* ids = nullptr;
        distance_t const* distances = nullptr;
        byte_t const* keys = nullptr;
        vram_ranking_t gathered;
        vram_ranking_t ranking;
        std::size_t block_bytes = 0;
        u32_t walked = 0;
        u32_t queries_count = 0;
        u32_t shards = 0;
    } matches;
    nk_cast_arguments_t cast;
    nk_each_rmsnorm_arguments_t rmscast;
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

    /** How rows are reached; @c scan_k scores every visible row exactly on the kits' tile steps,
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

/** Optional tuning knobs of a shard's @c refine; zeros mean "derive from connectivity". */
struct vram_refine_config_t {

    /** How many candidates a re-selection walk keeps, rounded up to the register rungs of 2 to 16
     *  warps' worth of ranks and walked exactly past them; zero takes twice the connectivity, like
     *  the CPU's. */
    u32_t expansion_factor = 0;
};

/** The grouped-query heads an attend stages beside the rows its walks navigate by: every head of
 *  every token, @c head_count consecutive rows per token. */
struct vram_attend_heads_t {
    rows_view_t rows;
    std::size_t head_count;
};

/** The counters an audit lands in page-locked memory, in the order of the device's: the levels that
 *  expanded, the last sweep's count, which only a sweep short of convergence leaves nonzero, then
 *  the unreached, reconnected and claim counters. */
struct vram_audit_landing_t {
    u32_t levels;
    u32_t last_expanded;
    u32_t unreached;
    u32_t reconnected;
    u32_t claims[2];
};

/** Device scratch one refine plan carves once from a @p traversal_context_type_ and its phases
 *  reuse, or the audit share one @c health carves: the reached set and two frontiers, the levels
 *  counter beside the last two sweeps' expanded counters, by the sweep number's parity, the
 *  unreached, reconnected and claim counters an audit lands together, then for a refine the
 *  unreached rows' IDs and the parts' bases. Sweeps run until one expands nothing: every row joins
 *  one frontier at most, so the rows bound them. */
template <typename traversal_context_type_>
struct vram_refine_scratch {
    typename traversal_context_type_::bytes_ptr_t bytes;

    /** The reached set and the two frontiers, one bit per row each. */
    vram_bitset_t reached, frontier, next_frontier;

    /** The levels that expanded anything, then the rows the last two sweeps expanded. */
    u32_t* levels = nullptr;
    u32_t* expanded = nullptr;
    u32_t* unreached = nullptr;
    u32_t* reconnected = nullptr;

    /** The walk's and the merge's claim counters, the last pass's completion oracle. */
    u32_t* claims = nullptr;

    /** The rows the last audit left unreached, which the heal walks. */
    u32_t* island_ids = nullptr;

    /** The ascending first rows of the parts the walks merge, uploaded once by the plan. */
    u32_t* part_bases = nullptr;
    std::size_t words = 0;
};

/** The buffers one refinement pass uses up, carved from its context's staging when it queues: the
 *  heads of the walkers' chains, then the bitset of rows the pass walked, the shadow neighbors the
 *  walks land and the chains' links beside them, @b [rows,connectivity] each. A reconnection counts
 *  its in-links over the chain heads, since it queues after its pass. */
struct vram_refine_passes_t {
    u32_t* link_heads = nullptr;
    vram_bitset_t walked;
    u32_t* shadow_neighbors = nullptr;
    u32_t* link_next = nullptr;
};

#pragma endregion Shard Types

/**
 *  @brief The core every dense index over a single uniform latency memory space on one GPU owns,
 *      and forwards its verbs to.
 *
 *  Speaks the RAM-resident dense index's verbs, with two divergences: a batch's IDs are one
 *  contiguous range, and its rows publish unordered on the device, turning visible to searches a
 *  whole batch at a time. The first successful @c reserve freezes the geometry and the
 *  virtual-address ceiling; physical slabs map on demand, and fresh neighbors are sentinel-filled
 *  before any ID covering them can be assigned. The storage grows only under the one claim the
 *  claim word admits at a time, and the claim word, the filled extent, the watermark and the kept
 *  status are the only mutable host state; the published-row count is device memory no host
 *  thread dereferences.
 *
 *  @tparam vendor_type_ The vendor's runtime types and calls, @c cuda_vendor_t or @c hip_vendor_t.
 *  @tparam kernels_type_ The vendor's kernels, held as a member: they load and launch the kernels,
 *      stage the walks and queue a refinement's passes over the core they are handed.
 *  @tparam vector_dtype_ The stored scalar.
 *  @tparam cache_hints_type_ The host's line hints, which demote a filled bounce half before its
 *      copy.
 *  @tparam vectors_allocator_type_ The vectors matrix and norms column backend, on the device.
 *  @tparam neighbors_allocator_type_ The neighbors matrix backend, on the same device.
 *  @tparam traversals_allocator_type_ The traversal contexts' scratch backend.
 */
template <                               //
    typename vendor_type_,               //
    typename kernels_type_,              //
    nk_dtype_t vector_dtype_,            //
    typename cache_hints_type_,          //
    typename vectors_allocator_type_,    //
    typename neighbors_allocator_type_,  //
    typename traversals_allocator_type_> //
struct vram_shard_core {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    using vendor_t = vendor_type_;
    using vectors_allocator_t = vectors_allocator_type_;
    using neighbors_allocator_t = neighbors_allocator_type_;
    using traversals_allocator_t = traversals_allocator_type_;
    using stream_handle_t = typename vendor_t::stream_handle_t;
    using event_t = typename vendor_t::event_t;
    using executor_t = typename vendor_t::executor_t;
    using device_specs_t = typename vendor_t::device_specs_t;
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
    using small_size_t = u32_t;

    /** The page-locked host bytes a checkpoint moves through, one batch at a time. */
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

    /** The once-frozen geometry word: dimensions in the high half, connectivity in the low, zero
     *  meaning not yet reserved. */
    using packed_geometry_id_t = u64_t;
    inline static constexpr packed_geometry_id_t pack_geometry(std::size_t dimensions,
                                                               std::size_t connectivity) noexcept {
        return (static_cast<packed_geometry_id_t>(dimensions) << 32u) | static_cast<packed_geometry_id_t>(connectivity);
    }
    inline static constexpr u32_t geometry_dimensions(packed_geometry_id_t geometry) noexcept {
        return static_cast<u32_t>(geometry >> 32u);
    }
    inline static constexpr u32_t geometry_connectivity(packed_geometry_id_t geometry) noexcept {
        return static_cast<u32_t>(geometry);
    }

    /** The @p kernel every walk scores by, checked against the dtype at the first @c reserve, and
     *  one allocator per storage concern, each carrying its device placement; the index pins itself
     *  to the vectors allocator's device and demands the neighbors allocator agree. */
    explicit vram_shard_core(nk_kernel_kind_t kernel, vectors_allocator_t const& vectors_allocator = {},
                             neighbors_allocator_t const& neighbors_allocator = {}) noexcept
        : device_ {vectors_allocator.device()}, kernel_ {kernel}, vectors_matrix_(vectors_allocator),
          norms_column_(vectors_allocator), neighbors_matrix_(neighbors_allocator),
          grown_((static_cast<void>(vendor_t::select(vectors_allocator.device())), event_t {})) {}
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
        if (kernel_ == nk_kernel_dot_k) return min_of(by_vectors, by_neighbors);
        std::size_t const by_norms = norms_column_.committed_bytes() / sizeof(norms_t);
        return min_of(by_vectors, min_of(by_norms, by_neighbors));
    }
    std::size_t connectivity() const noexcept {
        return geometry_connectivity(
            fu::standard_atomic_ref<packed_geometry_id_t const>(geometry_).load(std::memory_order_acquire));
    }
    std::size_t dimensions() const noexcept {
        return geometry_dimensions(
            fu::standard_atomic_ref<packed_geometry_id_t const>(geometry_).load(std::memory_order_acquire));
    }

    /** The walk breadth @p config asks for, or twice the connectivity when it asks for none. */
    template <typename config_type_>
    small_size_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor
                                       : static_cast<small_size_t>(default_expansion_factor(connectivity()));
    }

    /** The pinned device's identity and limits, queried once at the first @c reserve. */
    device_specs_t const& device_specs() const noexcept { return device_; }

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

    /** Fragments per vector row, derived from the frozen geometry like every row measure. */
    u32_t row_fragments() const noexcept {
        return divide_round_up<u32_t>(static_cast<u32_t>(packed_row_bytes()), fragment_bytes_k);
    }

    /** The vector row's byte length, rounded up to whole fragments; the tail bytes stay zero so
     *  every dot-derived metric is unaffected. */
    std::size_t row_bytes() const noexcept { return row_fragments() * fragment_bytes_k; }

    /** Steps the lanes cooperating on one row take across it. */
    u32_t steps_per_row() const noexcept { return divide_round_up(row_fragments(), kernels_.lanes_per_row(*this)); }

    /** Bytes of one builder warp's slice over @p selection_rows cached candidates: the query span
     *  while walking, then the candidate cache, and per cached candidate its score, three ID spans,
     *  its distance and its norms. */
    u32_t build_scratch_bytes_per_warp(u32_t selection_rows) const noexcept {
        u32_t const candidate_bytes = static_cast<u32_t>(row_bytes() + gpu_scored_node_bytes<node_id_t> +
                                                         3 * sizeof(u32_t) + sizeof(f32_t) + sizeof(norms_t));
        return max_of(static_cast<u32_t>(row_bytes()), selection_rows * candidate_bytes);
    }

    /** The serialized extents: published nodes x logical dimensions & connectivity - the fragment
     *  padding is a storage detail that never reaches a file. */
    struct saved_shape_t {
        std::size_t nodes = 0;
        std::size_t dimensions = 0;
        std::size_t connectivity = 0;
    };
    saved_shape_t saved_shape() const noexcept { return {size(), dimensions(), connectivity()}; }

    /** Where a save lands: row-major host columns the caller sized from @ref saved_shape, handed
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
     *  it, and the host struct; traversal scratch belongs to executors and contexts. */
    std::size_t memory_usage() const noexcept {
        return sizeof(vram_shard_core) + vectors_matrix_.committed_bytes() + norms_column_.committed_bytes() +
               neighbors_matrix_.committed_bytes();
    }

    /** Queues the copy of the stored row of @p node into @p destination with the fragment padding
     *  stripped on the caller's @p stream, whose device must be current; the caller waits once,
     *  after its last row. The bulk export is @c save. */
    status_t vector(node_id_t node, nk::vector_span<scalar_t> destination, stream_handle_t stream) const noexcept {
        if (node >= size()) return status_t::config_mismatch_k;
        if (destination.size() != dimensions()) return status_t::config_mismatch_k;
        if (!destination.is_contiguous()) return status_t::unsupported_layout_k;
        return succeeded(vendor_t::copy(destination.byte_data(), vectors_rows_() + node * row_bytes(),
                                        packed_row_bytes(), stream))
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

#pragma endregion Measures

#pragma region Reservation

    using frontier_kind_t = vram_frontier_kind_t;

    /** One warp's arena slice for @p walks walks sharing one visited set over @p candidate_rows
     *  candidates each: the exact visited set over every ID they claim, plus the spill tail when
     *  @p frontier_kind outgrows registers, what their expansions stage. */
    gpu_arena_shape_t arena_shape_for(u32_t candidate_rows, frontier_kind_t frontier_kind,
                                      u32_t walks = 1) const noexcept {
        u32_t const connectivity = static_cast<u32_t>(this->connectivity()), lanes = kernels_type_::wave_lanes();
        u32_t const visits = walks * gpu_walk_visits(candidate_rows, connectivity, lanes);
        u32_t const tail = frontier_kind == frontier_kind_t::wide_arena_k
                               ? walks * gpu_walk_tail_entries(candidate_rows, connectivity, lanes)
                               : 0;
        return gpu_arena_shape(visits, tail, 0);
    }

    /** Requested geometry & capacity for a @c reserve() call; frozen by the first success. */
    struct reserve_capacity_t {
        std::size_t capacity = 0;
        connectivity_t connectivity = connectivity_32_k;
        std::size_t dimensions = 0;

        /** Rows the index may ever host, clamped to the ID space; frozen with the geometry, past it
         *  every claim answers @c overflow_k. */
        std::size_t ceiling = max_node_count_k;
    };

    /** Ensures the index can host at least @p config.capacity entries and answers the capacity now
     *  in effect; the first successful call freezes the geometry and the virtual-address ceiling,
     *  later ones only map slabs. */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        // The kits' epilogues fall through to the last metric of each family, so no other kind may
        // reach them.
        bool const scored = vector_dtype_ == nk_u1_k
                                ? kernel_ == nk_kernel_dot_k || kernel_ == nk_kernel_hamming_k ||
                                      kernel_ == nk_kernel_jaccard_k
                                : kernel_ == nk_kernel_dot_k || kernel_ == nk_kernel_angular_k ||
                                      kernel_ == nk_kernel_sqeuclidean_k || kernel_ == nk_kernel_euclidean_k;
        if (!scored) return {0, status_t::unknown_kind_k};
        if (neighbors_matrix_.device() != device_.ordinal) return {0, status_t::config_mismatch_k};
        if (!config.dimensions || !std::has_single_bit(static_cast<u32_t>(config.connectivity)))
            return {0, status_t::config_mismatch_k};
        // The first success freezes both axes at once; later calls must agree with what they
        // observe.
        packed_geometry_id_t const requested = pack_geometry(config.dimensions,
                                                             static_cast<std::size_t>(config.connectivity));
        packed_geometry_id_t observed = 0;
        if (!fu::standard_atomic_ref<packed_geometry_id_t>(geometry_).compare_exchange_strong(
                observed, requested, std::memory_order_acq_rel, std::memory_order_acquire) &&
            observed != requested)
            return {capacity(), status_t::config_mismatch_k};
        connectivity_t const connectivity = config.connectivity;
        if (config.capacity > max_node_count_k) return {capacity(), status_t::overflow_k};
        if (!vectors_matrix_.reserved_bytes()) { // First reservation: bind the device, size the ceiling.
            // The device must be current before any mapping, since the storage lives on it.
            if (failed(vendor_t::select(device_.ordinal))) return {0, status_t::device_error_k};
            auto const [specs, status] = device_specs_t::query(device_.ordinal);
            if (failed(status)) return {0, status};
            device_ = specs;
            // Geometry no walk can fit is refused here; a builder whose scratch shared memory
            // cannot fit keeps it in its arena slice.
            if (!kernels_.walk_scratch_fits(*this)) return {0, status_t::config_mismatch_k};
            if (status_t const loaded = kernels_.load_kernels_(*this); failed(loaded)) return {0, loaded};
            // Device storage maps at most the device's memory, which bounds the ceiling independent
            // of the first requested capacity; the reservation itself is the ceiling, no cached row
            // count shadows it.
            bool const normed = kernel_ != nk_kernel_dot_k;
            std::size_t const bytes_per_row = row_bytes() + (normed ? sizeof(norms_t) : 0) +
                                              connectivity * sizeof(u32_t);
            // Device storage budgets against VRAM; managed storage oversubscribes it and budgets
            // against host memory through their own backend, which keeps the topology symbols out
            // of device instantiations.
            std::size_t budget_bytes = device_.vram_bytes;
            if constexpr (vectors_allocator_t::address_space_k == vendor_t::address_space_t::managed_k)
                budget_bytes = vectors_allocator_t::budget_bytes();
            std::size_t const ceiling_rows = min_of(min_of(config.ceiling, max_node_count_k),
                                                    max_of<std::size_t>(config.capacity, budget_bytes / bytes_per_row));
            // One more vectors row ahead of row zero holds the published-row count.
            if (vectors_matrix_.reserve_range((ceiling_rows + 1) * row_bytes()) != status_t::success_k ||
                (normed && norms_column_.reserve_range(ceiling_rows * sizeof(norms_t)) != status_t::success_k) ||
                neighbors_matrix_.reserve_range(ceiling_rows * connectivity * sizeof(u32_t)) != status_t::success_k)
                return {0, status_t::bad_alloc_k};
            ceiling_rows_ = ceiling_rows;
        }
        status_t const mapped = map_rows_(config.capacity);
        return {capacity(), mapped};
    }

    /** Maps the storage for those of @p devices that can reach this one, as the plural holding the
     *  shard names its devices, so their copies read the rows directly; precedes @ref reserve. */
    void share_with_(std::bitset<gpu_devices_capacity_k> devices) noexcept {
        vectors_matrix_.share_with(devices), norms_column_.share_with(devices), neighbors_matrix_.share_with(devices);
    }

    /** Maps the storage, the rows committed already included, for @p devices too, as a plural
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
        small_size_t expansion_factor = 0;
    };

    /** The bytes an executor's run must hold before a context carved over it can insert @p rows
     *  rows, one aligned term per carve the verb makes; zero before @c reserve freezes the
     *  geometry. */
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
        std::size_t const results_bytes = // A default single query's device landing.
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

        /** Newcomers no back-link reached, past what one re-selection weighs - GPU-only
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

    /** Inserts @p new_rows under the contiguous ID range @b [first_id,first_id+rows) claimed from
     *  the watermark, every transfer and the launch riding the context's stream, then drains that
     *  stream, turns the watermark over the range, and reports telemetry. One batch or load holds
     *  the claim word at a time, so another one in flight, or a closed word, answers
     *  @c contended_k. */
    expected<insert_result_t> insert(rows_view_t new_rows, node_ids_span_t assigned_ids, traversal_context_t& context,
                                     insert_config_t config = {}) noexcept {
        // Zeroed per call rather than per launch, so the launches of a sharded plan on one context
        // add up.
        if (context && context.device() == device_.ordinal &&
            failed(vendor_t::fill(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), context.stream())))
            return {{}, status_t::device_error_k};
        if (!new_rows.rows) return {{claimed_(), static_cast<node_id_t>(claimed_())}, status_t::success_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        auto const [first_id, claim_status] = claim_(new_rows.rows);
        if (failed(claim_status)) return {{}, claim_status};
        expected<insert_result_t> inserted = submit_insert_(new_rows, first_id, assigned_ids, context, config);
        if (!inserted) {
            unclaim_(first_id, new_rows.rows);
            return inserted;
        }
        if (status_t const drained = context.wait(); failed(drained))
            return {{inserted.value.new_size, inserted.value.first_id}, fail_(drained)};
        if (status_t const published = publish_(inserted.value.first_id, new_rows.rows); failed(published))
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
     *  @param[inout] context The stream every transfer and the launch ride.
     *  @param[in] config The per-call breadth and match count.
     *  @param[in] completion Blocking drains the context's stream and reports the telemetry;
     *      deferred returns once queued, and the caller drains before reading either output.
     *  @return The match count per query, and the traversal telemetry this call spent.
     */
    expected<search_result_t> search(rows_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                     search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept {
        return settle_(kernels_.submit_search_(*this, queries, result_ids, result_distances, context, config, size()),
                       queries.rows, context, completion);
    }

    /** The batched @c search ranked by a lightning indexer: a node's distance is 1 − Σₕ wₕ ·
     *  ReLU(qₕ · k) over the @p heads of its query, which must match the queries in count and the
     *  index in dimensions, while the walk navigates by @p queries. A dot verb: any other metric
     *  refuses with @c config_mismatch_k. */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads,
                                     nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                     search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept
        requires floating_scalar<scalar_t>
    {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return settle_(
            kernels_.submit_search_(*this, queries, result_ids, result_distances, context, config, size(), heads),
            queries.rows, context, completion);
    }

    /** One query in, the nearest matches ascending in @c context.results - the correctness path; a
     *  one-query grid never fills the machine, the batched overload does. */
    expected<search_result_t> search(vector_view_t query_vector, traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        return search_(query_vector, context, config, size());
    }

    /** Filtered search answers @c device_error_k: a host predicate cannot run inside the walk, and
     *  a device-resident admission bitmap is a design of its own. */
    template <typename predicate_type_>
    expected<search_result_t> search_if(vector_view_t, predicate_type_&&, traversal_context_t&,
                                        search_config_t = {}) noexcept {
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
     *  @param[inout] context The stream every transfer and the launch ride.
     *  @param[in] config The per-call breadth; @c count plays no part, the outputs' width does.
     *  @param[in] completion Blocking drains the context's stream and reports the telemetry;
     *      deferred returns once queued, and the caller drains before reading any output.
     *  @return The pairs scored over every row once blocking - zero when deferred, where @p counts
     *      carry them after the drain - and the traversal telemetry this call spent;
     *      @c config_mismatch_k unless the index scores by dot.
     */
    expected<search_result_t> multiply(rows_view_t left, nk::matrix_span<node_id_t> node_ids,
                                       nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                       traversal_context_t& context, search_config_t config = {},
                                       completion_t completion = completion_t::blocking_k) noexcept {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        expected<search_result_t> landed = settle_(
            kernels_.submit_multiply_(*this, left, node_ids, products, counts, context, config, size()), left.rows,
            context, completion);
        if (landed && completion == completion_t::blocking_k)
            for (u32_t const scored : counts) landed.value.count += scored;
        return landed;
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
     *  @param[in] value_lookup A trivially copyable record whose `vram_value_row(value_lookup,
     *      node_id)` device overload points at the node's value row on a 16-byte boundary, as wide
     *      as the keys and stored like them, in memory this device reads; a null row refuses the
     *      call with @c shape_mismatch_k. The vendor's code object instantiates its attend kernels
     *      over the record's type, which spells itself into their names.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys, in host or
     *      device memory; rows in any scalar but f32 cast on the device first, and must fill whole
     *      bytes.
     *  @param[out] carry Each row's softmax state, one per row, in host or device memory; an empty
     *      span skips them.
     *  @param[inout] context The stream every transfer and the launch ride.
     *  @param[in] config The per-call breadth; @c count plays no part.
     *  @param[in] completion As in @c multiply; a deferred call leaves a missing row to
     *      @c context.missed().
     *  @return The query rows attended, and the traversal telemetry this call spent;
     *      @c config_mismatch_k unless the index scores by dot.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     traversal_context_t& context, search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept {
        std::size_t const rows_count = queries.rows;
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = landing_refusal_(rows_count, outputs, carry); failed(refusal))
            return {{}, refusal};
        expected<f32_t*> const landed = kernels_.submit_attend_(*this, queries, head_count, value_lookup, score_scale,
                                                                context, config, 1, exported_bytes_(outputs), size());
        if (!landed || !rows_count) return {{}, landed.status};
        if (status_t const copied = land_attended_(landed.value, 1, outputs, carry, context.stream()); failed(copied))
            return {{}, copied};
        expected<search_result_t> const attended = settle_({{rows_count}, status_t::success_k}, rows_count, context,
                                                           completion);
        if (attended && completion == completion_t::blocking_k && context.missed())
            return {{}, status_t::shape_mismatch_k};
        return attended;
    }

#pragma endregion Search

#pragma region Refinement

    using refine_config_t = vram_refine_config_t;

    using refine_result_t = dense::refine_result_t;
    using health_result_t = dense::health_result_t;

  private:
    using refine_scratch_t = vram_refine_scratch<traversal_context_t>;
    using audit_landing_t = vram_audit_landing_t;

    /** Gives a landing back to the stream's slots it came from. */
    struct landing_return_t {
        typename vendor_t::landings_t* landings = nullptr;
        void operator()(audit_landing_t* landing) const noexcept { landings->give_back(landing); }
    };
    using audit_landing_ptr_t = std::unique_ptr<audit_landing_t, landing_return_t>;
    static_assert(sizeof(audit_landing_t) <= vendor_t::landings_t::slot_bytes_k);

  public:

    /**
     *  @brief One refinement in flight on this index's device - the CPU's @c refine_plan_t in the
     *      shape one host thread drives over many devices at once: @ref queue a phase, @ref wait
     *      for it, @ref advance past it, and @ref finish once no phase follows.
     *
     *  The phases re-select every node of the planned range by a walk, once per round of walks
     *  across while the shard holds several parts, then audit reachability, then heal round by
     *  round while the islands shrink: re-select them, and reconnect whatever stays unreached into
     *  a reached pick or the landmark covering it. A re-selection lands shadow neighbors and chains
     *  itself onto every pick, as an insertion does, and the merge folds each re-selected row's
     *  forward picks, then every row chained onto it, into its live row, re-judging whatever
     *  overflows it. The walk runs once, as the CPU's refine does per call.
     *
     *  A phase queues its launches and the landing of its counters behind the context's event and
     *  never waits; @ref plan_refine made every allocation, so no phase allocates, and only a
     *  device fault fails one. Every phase takes the context the plan was opened on back from its
     *  caller. Demands an exclusive caller: the merge writes live neighbors with plain stores. A
     *  stopped sequence of phases leaves the graph correct and merely less improved.
     */
    struct refine_plan_t {

        /** An unbound plan over @p scratch - what a refused plan answers with. */
        explicit refine_plan_t(refine_scratch_t&& scratch) noexcept : scratch_(std::move(scratch)) {}
        refine_plan_t(refine_plan_t&&) noexcept = default;
        refine_plan_t& operator=(refine_plan_t&&) noexcept = default;

        /** Queues the current phase on @p context 's stream under @p config, then the landing of
         *  its counters and telemetry behind the context's event; never waits and never
         *  allocates. */
        status_t queue(traversal_context_t& context, refine_config_t config) noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (failed(status_)) return status_;
            if (context.device() != index_->device_.ordinal) return status_t::config_mismatch_k;
            stream_handle_t const stream = context.stream();
            switch (phase_) {
            case phase_t::reselect_k: {
                iota_view<node_id_t> const range = round_nodes_();
                if (!range.empty())
                    status_ = index_->kernels_.refine_pass_(*index_, context, scratch_, config, vram_kernel_t::refine_k,
                                                            range, {}, parts_, round_);
                if (succeeded(status_)) status_ = land_counters_(scratch_, landing_.get(), 0, stream);
                break;
            }
            case phase_t::audit_k: status_ = queue_sweeps_(context); break;
            case phase_t::heal_k:
                // The island IDs replace the range: offsets below their count index them. Only
                // island rows change, and no reached row points at one, so the audit before stands.
                // A continued audit sweeps on alone.
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

        /** Sleeps until the queued phase landed, on @p context 's event. */
        status_t wait(traversal_context_t& context) noexcept {
            if (!index_ || phase_ == phase_t::done_k) return status_;
            if (succeeded(status_)) status_ = context.event_.wait();
            return status_;
        }

        /** The barrier after a phase landed: reads its counters, picks the next phase, and answers
         *  whether one follows. */
        bool advance() noexcept {
            if (!index_ || failed(status_) || phase_ == phase_t::done_k) return false;
            audit_landing_t const& landed = *landing_;
            std::size_t const walked = walked_(), rows = index_->size();
            // The claim counts are the completion oracle: every offset claimed means every row
            // landed.
            if (walked && (landed.claims[0] < walked || landed.claims[1] < rows)) {
                status_ = status_t::device_error_k;
                return false;
            }
            tally_.reselected_nodes += walked;
            if (phase_ == phase_t::reselect_k && ++round_ < rounds_) return true;
            if (phase_ == phase_t::reselect_k) {
                phase_ = phase_t::audit_k;
                return true;
            }
            // A last sweep that still expanded leaves rows unjudged, so the phase sweeps on.
            if (landed.last_expanded) return true;
            next_sweep_ = 0;
            tally_.health = audited_(landed, rows);
            // The device counter is zeroed once per plan and every heal adds to it,
            // so it lands summed.
            tally_.islands_reconnected = landed.reconnected;
            u32_t const left = landed.unreached;
            if (phase_ == phase_t::audit_k) tally_.islands_found = left;
            // A round that frees no island frees none on the next either, since the picks it needs
            // are unreached.
            bool const healing = left && (phase_ == phase_t::audit_k || left < islands_);
            islands_ = left;
            phase_ = healing ? phase_t::heal_k : phase_t::done_k;
            return healing;
        }

        /** Stops the re-selection after the round that landed last, so the next phase audits and
         *  heals; a no-op in any other phase. */
        void stop() noexcept {
            if (phase_ == phase_t::reselect_k) rounds_ = 0, phase_ = phase_t::audit_k;
        }

        /** Whether the current phase re-selects - the phase a caller's progress counts. */
        bool reselecting() const noexcept { return phase_ == phase_t::reselect_k; }

        /** Nodes re-selected so far. */
        std::size_t visited() const noexcept { return tally_.reselected_nodes; }

        /** The pass: its telemetry read from @p context 's last landing, complete once @ref advance
         *  answered false, otherwise @c cancelled_k; a failure a phase kept answers instead. */
        expected<refine_result_t> finish(traversal_context_t const& context) const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            refine_result_t result = tally_;
            vram_telemetry_block_t const spent = context.spent();
            result.changed_nodes = spent.changed_nodes;
            result.telemetry.visited_members = spent.hops;
            result.telemetry.computed_distances = spent.scored;
            result.telemetry.restarts = spent.restarts;
            result.telemetry.saturated_walks = spent.visited_saturations;
            if (failed(status_)) return {result, status_};
            return {result, phase_ == phase_t::done_k ? status_t::success_k : status_t::cancelled_k};
        }

      private:
        friend struct vram_shard_core;
        enum class phase_t : u8_t { reselect_k, audit_k, heal_k, done_k };

        /** Queues the audit's next batch of sweeps on @p context, its seeds first when it
         *  starts. */
        status_t queue_sweeps_(traversal_context_t& context) noexcept {
            status_t const queued = next_sweep_
                                        ? index_->kernels_.queue_sweeps_(*index_, context, scratch_, landing_.get(),
                                                                         next_sweep_)
                                        : index_->kernels_.queue_audit_(*index_, context, scratch_, landing_.get());
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
        vram_shard_core* index_ = nullptr;
        audit_landing_ptr_t landing_ {nullptr, landing_return_t {}};

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
        phase_t phase_ = phase_t::done_k;
        status_t status_ = status_t::success_k;
        refine_result_t tally_;
    };

    /**
     *  @brief Opens a refinement of the nodes in @p nodes, clamped to the published IDs, on
     *      @p context 's stream: carves the scratch, grows the context's arena for the widest walk
     *      and its staging for the passes, takes a page-locked landing and queues the upload of
     *      @p part_bases.
     *
     *  No phase allocates. An empty range heals alone.
     *
     *  Two or more @p part_bases below the published IDs split the rows into parts, as a shard
     *  that absorbed others' rows holds them: each node then first walks across the neighboring
     *  group of parts alone, round after round as @c part_across names them, from empty candidates
     *  and landmarks drawn there, before its own neighbors and the whole graph, and re-selects
     *  from the union. The live graph stays as it was through a round's walks, so a walk across
     *  never leaves its group, and the heal's passes walk no part first.
     *
     *  @param[in] nodes The ID range to walk; IDs at or past @c size() are skipped.
     *  @param[inout] context The driving context, which every phase takes back; its arena and
     *      staging grow to the pass's shape.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in] part_bases The ascending first rows of the parts; need not outlive the plan.
     */
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                        refine_config_t config = {},
                                        std::span<node_id_t const> part_bases = {}) noexcept {
        std::size_t const rows = size();
        std::size_t parts = 0;
        while (parts != part_bases.size() && part_bases[parts] < rows) ++parts;
        if (parts < 2) parts = 0;
        auto [scratch, carve_status] = carve_scratch_(context, scratch_use_t::refine_k, parts);
        refine_plan_t plan {std::move(scratch)};
        if (failed(carve_status)) return {std::move(plan), carve_status};
        if (!context) return {std::move(plan), status_t::config_mismatch_k};
        plan.landing_ = take_landing_(context);
        if (!plan.landing_) return {std::move(plan), status_t::bad_alloc_k};
        // The widest walk's arena now, so no pass grows it; a fresh run is zeroed by the pass.
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const candidate_slots = candidate_slots_for_(expansion, 4);
        u32_t const walks = parts ? 2 : 1;
        std::size_t const arena_warps = device_.multiprocessor_count *
                                        kernels_type_::resident_blocks_(vram_kernel_t::refine_k, candidate_slots) *
                                        vram_warps_per_block_k;
        if (!context.ensure_arena(
                arena_warps * build_arena_shape_(expansion, candidate_slots, walks).template slice_bytes<node_id_t>()))
            return {std::move(plan), status_t::bad_alloc_k};
        if (std::size_t const staging_bytes = refine_staging_bytes_();
            staging_bytes && !context.ensure_staging(staging_bytes))
            return {std::move(plan), status_t::bad_alloc_k};
        stream_handle_t const stream = context.stream();
        // Zeroed once, so every pass adds up, and landed with every phase. A pageable upload stages
        // on the host at once and waits on nothing.
        if (failed(vendor_t::fill(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream)) ||
            (parts &&
             failed(vendor_t::copy(plan.scratch_.part_bases, part_bases.data(), parts * sizeof(node_id_t), stream))))
            return {std::move(plan), status_t::device_error_k};
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
        while (plan.advance());
        return plan.finish(context);
    }

    /** A split of this index into @p receivers, the inverse of an absorb, which no dense GPU face
     *  serves yet: it answers @c unsupported_k with an unbound plan. */
    template <typename owner_type_>
    expected<refine_plan_t> plan_split(std::span<owner_type_*>, traversal_context_t& context, refine_config_t = {},
                                       u64_t = 0) noexcept {
        using bytes_ptr_t = typename traversal_context_t::bytes_ptr_t;
        return {refine_plan_t {refine_scratch_t {
                    bytes_ptr_t {nullptr, typename bytes_ptr_t::deleter_type {context.allocator_, 0}}}},
                status_t::unsupported_k};
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
        bool running() const noexcept { return phase_ == phase_t::sweeping_k; }

        /** Queues the next batch of sweeps on @p context, the landmarks first, doubling the batch
         *  each time; a no-op unless @ref running. */
        status_t queue(traversal_context_t& context) noexcept {
            if (phase_ != phase_t::sweeping_k) return status_;
            status_ = next_sweep_
                          ? index_->kernels_.queue_sweeps_(*index_, context, scratch_, landing_.get(), next_sweep_)
                          : index_->kernels_.queue_audit_(*index_, context, scratch_, landing_.get());
            next_sweep_ = next_sweep_ ? 2 * next_sweep_ : 2;
            phase_ = succeeded(status_) ? phase_t::landing_k : phase_t::done_k;
            return status_;
        }

        /** Sleeps until the queued batch landed on @p context 's event, then reads whether it still
         *  expanded; a no-op unless a batch was queued. */
        status_t wait(traversal_context_t& context) noexcept {
            if (phase_ != phase_t::landing_k) return status_;
            status_ = context.wait();
            phase_ = succeeded(status_) && landing_->last_expanded ? phase_t::sweeping_k : phase_t::done_k;
            return status_;
        }

        /** The audit over the rows published when it was planned; a failure a batch kept answers
         *  instead. */
        expected<health_result_t> finish() const noexcept {
            if (failed(status_)) return {{}, status_};
            return {index_ ? audited_(*landing_, rows_) : health_result_t {}, status_t::success_k};
        }

      private:
        friend struct vram_shard_core;
        enum class phase_t : u8_t { sweeping_k, landing_k, done_k };

        refine_scratch_t scratch_;
        vram_shard_core* index_ = nullptr;
        audit_landing_ptr_t landing_ {nullptr, landing_return_t {}};
        std::size_t rows_ = 0;

        /** The sweep the next batch queues from, zero before the landmarks. */
        u32_t next_sweep_ = 0;
        phase_t phase_ = phase_t::done_k;
        status_t status_ = status_t::success_k;
    };

    /** Opens a reachability audit of the rows published now on @p context 's stream: carves the
     *  audit share of the scratch and takes a page-locked landing; queues nothing. */
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
        plan.landing_ = take_landing_(context);
        if (!plan.landing_) return {std::move(plan), status_t::bad_alloc_k};
        plan.index_ = this;
        plan.rows_ = rows;
        plan.phase_ = health_plan_t::phase_t::sweeping_k;
        return {std::move(plan), status_t::success_k};
    }

    /** Measures forward reachability from as many stride-spread landmarks as one search draws -
     *  the connectivity; any shortfall against @c size() is the population no search can
     *  discover. Read-only on the graph, so concurrent insertions may leave fresh nodes
     *  uncounted. One @ref health_plan_t driven on the calling thread, waiting once per batch of
     *  sweeps, the batches doubling until one expands nothing. */
    expected<health_result_t> health(traversal_context_t& context) noexcept {
        auto [plan, status] = plan_health(context);
        if (failed(status)) return {{}, status};
        while (plan.running())
            if (failed(plan.queue(context)) || failed(plan.wait(context))) break;
        return plan.finish();
    }

    /** The bytes an executor's run must hold before a context carved over it can refine the rows
     *  published now under @p config, @p walks walks of a row sharing one visited set: the
     *  context's counters, the walks' arena, the passes' staging, and the scratch the refine
     *  carves once. */
    std::size_t traversal_memory_usage(refine_config_t config, u32_t walks = 1) const noexcept {
        if (!dimensions() || !device_.multiprocessor_count) return 0;
        auto aligned = [](std::size_t bytes) noexcept {
            return divide_round_up<std::size_t>(bytes, default_alignment_k) * default_alignment_k;
        };
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const candidate_slots = candidate_slots_for_(expansion, 4);
        gpu_arena_shape_t const arena_shape = build_arena_shape_(expansion, candidate_slots, walks);
        std::size_t const arena_warps = device_.multiprocessor_count *
                                        kernels_type_::resident_blocks_(vram_kernel_t::refine_k, candidate_slots) *
                                        vram_warps_per_block_k;
        return aligned(sizeof(vram_telemetry_block_t)) + 2 * aligned(sizeof(u32_t)) +
               aligned(arena_warps * arena_shape.slice_bytes<node_id_t>()) + aligned(refine_staging_bytes_()) +
               aligned(scratch_bytes_(scratch_use_t::refine_k));
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
     *  A batch crosses half a batch at a time: each half lands on the host while the next one is
     *  still copying off the device, so the host awaits each half's event once, and the last one
     *  lands in @c finish. One host thread drives a device, so batches are taken one after another,
     *  though the tallies keep the shape a threaded caller expects.
     */
    struct [[nodiscard]] save_plan_t {

        /** Where the rows land, vetted against the watermark at planning. */
        columns_t columns {};

        /** The stream every device-to-host copy queues on. */
        stream_handle_t stream = nullptr;

        /** Page-locked host bytes in two halves of @ref half_rows rows each: packed vector rows,
         *  then their neighbor IDs, then their norms. */
        bounce_t bounce {typename vendor_t::pinned_allocator_t {}};

        /** Rows under the watermark at planning. */
        node_id_t visible = 0;

        /** Rows whose vectors and neighbors landed, tallied through @c std::atomic_ref. */
        std::size_t covered = 0;

        save_plan_t() noexcept = default;

        /** The index snapshotted into @p columns on @p stream under the watermark at @p visible;
         *  only @ref plan_save calls it, with the index's device current, where the events are
         *  born, and sizes the bounce afterwards. */
        save_plan_t(vram_shard_core const& index, columns_t columns, typename executor_t::stream_t const& stream,
                    node_id_t visible) noexcept
            : columns(columns), stream(stream.handle), bounce(typename vendor_t::pinned_allocator_t {stream}),
              visible(visible), index_(&index) {}
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
         *  once the copies into its pending half landed. */
        ~save_plan_t() noexcept {
            if (!pending_.index_rows.empty()) [[maybe_unused]]
                status_t const drained = crossed_[pending_half_].wait();
        }

        /** One step of the plan: one bounce-sized batch of the rows under the watermark. */
        struct step_t {

            /** The step's position in the plan - what a fan-out hands to a stream. */
            std::size_t position = 0;

            /** Where the rows sit in the index, in its own ID space. */
            iota_view<node_id_t> index_rows;

            /** Where they sit in the caller's buffer, counted from its first row. */
            iota_view<node_id_t> source_rows;
        };

        /** Step @p position of the plan; an empty one past the last. */
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
         *  step's last half is still copying when it returns, and lands with the next step or in
         *  @c finish.
         *
         *  @param[in] step The step to copy, as @ref steps hands them out.
         *  @param[inout] scratch Unused here, since the bounce is already host memory; it rides
         *      in the signature so a caller drives a device plan exactly as it drives a host one.
         */
        status_t save(step_t const& step, nk::matrix_span<node_id_t>) noexcept {
            if (!index_) return status_t::uninitialized_k;
            // Shards' steps interleave on one thread, so another shard's device may be current.
            if (failed(vendor_t::select(index_->device_specs().ordinal))) return status_t::device_error_k;
            iota_view<node_id_t> const rows = step.index_rows;
            if (rows.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(node_id_t);
            std::size_t const first = step.source_rows.first;
            if (flat_rows(columns.vectors, first, rows.size(), scalar_row_bytes).empty() ||
                flat_rows(columns.neighbors, first, rows.size(), neighbors_row_bytes).empty() ||
                flat_rows(columns.norms, first, rows.size(), sizeof(norms_t)).empty() != columns.norms.empty())
                return status_t::config_mismatch_k;
            for (std::size_t chunk = rows.first; chunk < rows.bound; chunk += half_rows()) {
                std::size_t const bound = min_of<std::size_t>(chunk + half_rows(), rows.bound);
                // Every half but the pending one has landed, so the other one is free.
                std::size_t const half = pending_.index_rows.empty() ? 0 : pending_half_ ^ 1;
                byte_t* const bounce_vectors =
                    bounce.data() + half * half_rows() * (scalar_row_bytes + neighbors_row_bytes + sizeof(norms_t));
                byte_t* const bounce_neighbors = bounce_vectors + half_rows() * scalar_row_bytes;
                byte_t* const bounce_norms = bounce_neighbors + half_rows() * neighbors_row_bytes;
                std::size_t const count = bound - chunk;
                if (failed(vendor_t::copy_2d(bounce_vectors, scalar_row_bytes,
                                             index_->vectors_rows_() + chunk * index_->row_bytes(), index_->row_bytes(),
                                             scalar_row_bytes, count, stream)) ||
                    (!columns.norms.empty() &&
                     failed(vendor_t::copy(bounce_norms, index_->norms_column_.base() + chunk * sizeof(norms_t),
                                           count * sizeof(norms_t), stream))) ||
                    failed(vendor_t::copy(bounce_neighbors,
                                          index_->neighbors_matrix_.base() + chunk * neighbors_row_bytes,
                                          count * neighbors_row_bytes, stream)) ||
                    failed(crossed_[half].record(stream)))
                    return status_t::device_error_k;
                // The previous half lands on the host while this one copies off the device.
                if (status_t const landed = land_pending_(); failed(landed)) return landed;
                pending_.index_rows = {static_cast<node_id_t>(chunk), static_cast<node_id_t>(bound)};
                pending_.source_rows = {static_cast<node_id_t>(first + chunk - rows.first),
                                        static_cast<node_id_t>(first + bound - rows.first)};
                pending_half_ = half;
            }
            return status_t::success_k;
        }

        /** Pronounces the snapshot complete once the last half landed: every batch's vectors and
         *  neighbors in the columns. A success spends the plan; a refusal leaves it whole, so the
         *  missing steps can still land. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (failed(vendor_t::select(index_->device_specs().ordinal))) return status_t::device_error_k;
            if (status_t const landed = land_pending_(); failed(landed)) return landed;
            std::size_t const nodes = static_cast<std::size_t>(visible);
            if (fu::standard_atomic_ref<std::size_t> {covered}.load(std::memory_order_acquire) != nodes)
                return status_t::config_mismatch_k;
            index_ = nullptr;
            return status_t::success_k;
        }

      private:

        /** Awaits the pending half's copies, then moves its rows into the columns, shifting every
         *  neighbor under the watermark and folding every other onto the sentinel. */
        status_t land_pending_() noexcept {
            iota_view<node_id_t> const rows = pending_.source_rows;
            if (pending_.index_rows.empty()) return status_t::success_k;
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

            std::span<node_id_t> const landed {reinterpret_cast<node_id_t*>(bounce_neighbors),
                                               rows.size() * connectivity};
            for (node_id_t& neighbor : landed) neighbor = neighbor < visible ? neighbor : node_id_unknown_k;
            [[maybe_unused]] status_t const copied_neighbors = copy_bytes(neighbors_rows, std::as_bytes(landed));
            fu::standard_atomic_ref<std::size_t> {covered}.add(rows.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Marks the last copies into each half of the bounce, born with the plan. */
        event_t crossed_[2];

        /** The rows whose copies sit in a half not yet landed on the host; empty when none do. */
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
     *  @param[out] columns Where the rows land, sized from @ref saved_shape.
     *  @param[in] synchronization The contract the caller certifies.
     *  @param[in] stream The stream every copy queues on.
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
        if (failed(vendor_t::select(device_.ordinal))) return {{}, status_t::device_error_k};
        save_plan_t plan {*this, columns, stream, static_cast<node_id_t>(rows)};
        std::size_t const bounce_bytes = 2 * plan.half_rows() *
                                         (packed_row_bytes() + connectivity() * sizeof(node_id_t) + sizeof(norms_t));
        if (bounce_bytes && failed(plan.bounce.resize_uninitialized(bounce_bytes))) return {{}, status_t::bad_alloc_k};
        return {std::move(plan), status_t::success_k};
    }

    /** Serializes every row under the watermark into @p columns on the calling thread - the
     *  one-shot over @ref plan_save for a caller with no streams to fan the steps out over. */
    status_t save(columns_t columns, synchronization_t synchronization, nk::matrix_span<node_id_t> scratch,
                  typename executor_t::stream_t const& stream) const noexcept {
        auto [plan, status] = plan_save(columns, synchronization, stream);
        if (failed(status)) return status;
        for (typename save_plan_t::step_t const step : plan.steps())
            if (status_t const saved = plan.save(step, scratch); failed(saved)) return saved;
        return plan.finish();
    }

    /**
     *  @brief One load in flight: the rows it appends past every claimed one, the storage grown to
     *      hold them, and the steps they span.
     *
     *  Each step fills from host columns through a double-buffered pinned bounce or from another
     *  index's storage on the stream, and @ref finish derives the norms no step carried and
     *  publishes.
     *
     *  Holds the claim for its whole life, so nothing else claims rows while it runs. A source's
     *  IDs are local to the rows it delivers, so every batch shifts them onto the appended range
     *  and folds every other ID onto the sentinel. Repeated plans append, which is how several
     *  parts land in one shard. A batch crosses half a batch at a time, the host staging one half
     *  while the other copies to the device, and refilling a half only once its event says the
     *  copies out of it landed; @c finish is the one full wait. A plan that dies before @c finish
     *  claims nothing and leaves the storage grown for the next one, its rows unfilled, so the
     *  next claim sentinel-fills over whatever its steps wrote.
     */
    struct [[nodiscard]] load_plan_t {

        /** The stream every copy into the index and every kernel queue on, whose device every step
         *  and @c finish select first. */
        stream_handle_t stream = nullptr;

        /** Page-locked host bytes in two halves of @ref half_rows rows each: packed vector rows,
         *  then their neighbor IDs, then their norms, pinned at planning by @ref pin_bounce or
         *  never. */
        bounce_t bounce {typename vendor_t::pinned_allocator_t {}};

        /** The first appended ID - the row the claim landed at. */
        node_id_t base = 0;

        /** Rows the plan appends. */
        std::size_t rows = 0;

        /** Rows whose vectors and neighbors landed, tallied through @c std::atomic_ref. */
        std::size_t covered = 0;

        /** Rows whose norms landed from the columns, tallied through @c std::atomic_ref; @c finish
         *  derives every row's when it falls short. */
        std::size_t covered_norms = 0;

        load_plan_t() noexcept = default;

        /** The index appended to at @p base on @p stream, under the claim over its rows or a claim
         *  word its caller closed; only @ref plan_load calls it, with the index's device current,
         *  where the events are born. */
        load_plan_t(vram_shard_core& index, typename executor_t::stream_t const& stream, node_id_t base,
                    std::size_t rows) noexcept
            : stream(stream.handle), bounce(typename vendor_t::pinned_allocator_t {stream}), base(base), rows(rows),
              index_(&index) {}
        load_plan_t(load_plan_t&& other) noexcept
            : stream(other.stream), bounce(std::move(other.bounce)), base(other.base),
              rows(std::exchange(other.rows, 0)), covered(other.covered), covered_norms(other.covered_norms),
              crossed_ {std::move(other.crossed_[0]), std::move(other.crossed_[1])}, chunks_(other.chunks_),
              index_(std::exchange(other.index_, nullptr)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        // The tallies are this plan's own: a copy would tally into a twin `finish` never reads.
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** An unfinished plan gives its claim back, marking its stream so the next grow follows
         *  every copy it queued, and awaiting that mark before its bounce unpins. */
        ~load_plan_t() noexcept {
            if (!index_ || !rows) return;
            if (succeeded(vendor_t::select(index_->device_specs().ordinal)) &&
                succeeded(index_->grown_.record(stream)) && chunks_) [[maybe_unused]]
                status_t const drained = index_->grown_.wait();
            // Under the claim or the closed word, which every reader of the filled extent holds.
            index_->filled_rows_ = min_of(index_->filled_rows_, static_cast<std::size_t>(base));
            index_->unclaim_(base, rows);
        }

        /** One step of the plan: one bounce-sized batch of the appended rows. */
        struct step_t {

            /** The step's position in the plan - what a fan-out hands to a stream. */
            std::size_t position = 0;

            /** Where the rows land in the index, in its own ID space, from @c base. */
            iota_view<node_id_t> index_rows;

            /** Where they come from in the caller's buffer, from its first row. */
            iota_view<node_id_t> source_rows;
        };

        /** Step @p position of the plan; an empty one past the last. */
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

        /** The steps the append spans - one bounce-sized batch each. */
        indexed_range<load_plan_t> steps() const noexcept {
            return {this, index_ && rows ? divide_round_up(rows, batch_rows()) : 0};
        }

        /** Rows one batch carries - a bounce's worth, or every row when fewer. */
        std::size_t batch_rows() const noexcept { return min_of(rows, power_of_two<std::size_t>(batch_rows_log2_k)); }

        /** Rows one half of the bounce carries: half a batch, so two halves pin what one batch
         *  would. */
        std::size_t half_rows() const noexcept { return divide_round_up<std::size_t>(batch_rows(), 2); }

        /** Pins the bounce at planning, for a plan whose steps cross host memory; a plan whose rows
         *  move between device matrices never pins. */
        status_t pin_bounce() noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const row_bytes = index_->packed_row_bytes() + index_->connectivity() * sizeof(node_id_t) +
                                          sizeof(norms_t);
            return bounce.empty() ? bounce.resize_uninitialized(2 * half_rows() * row_bytes) : status_t::success_k;
        }

        /**
         *  @brief Copies step @p step 's rows out of @p columns through the bounce.
         *
         *  The vectors go into the fragment-padded matrix rows, whose fresh tails arrive zeroed,
         *  and their norms where the columns carry them, then the neighbors with every ID inside
         *  @p source_range folded onto the appended range and every other onto the sentinel. A
         *  sharded plan drives it one part's slice at a time, its step naming the rows in that
         *  part's row and ID spaces.
         *
         *  @param[in] step The step to fill; its source rows address @p columns.
         *  @param[in] columns The rows' columns, one part's under a sharded plan.
         *  @param[in] source_range The ID space those rows speak.
         */
        status_t load(step_t const& step, columns_view_t columns, iota_view<node_id_t> source_range) noexcept {
            if (!index_ || bounce.empty()) return status_t::uninitialized_k;
            if (failed(vendor_t::select(index_->device_specs().ordinal))) return status_t::device_error_k;
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
            // The part's rows land at this step's own index rows, so an ID's shift follows from
            // the step rather than from the plan's base - the two agree for a one-part plan.
            node_id_t const shift = static_cast<node_id_t>(range.first - step.source_rows.first + source_range.first);
            // The device indexes its chain heads by neighbor ID unchecked, so every rebased ID must
            // land in this plan.
            if (static_cast<std::size_t>(shift) + source_range.size() > static_cast<std::size_t>(base) + rows)
                return status_t::config_mismatch_k;
            // Dot reserves no norms column, so a part's column has nowhere to land.
            bool const normed = index_->stores_norms() && !norms_rows.empty();
            std::size_t const half_bytes = half_rows() * (scalar_row_bytes + neighbors_row_bytes + sizeof(norms_t));
            for (std::size_t done = 0; done < range.size(); done += half_rows()) {
                std::size_t const count = min_of(half_rows(), range.size() - done), into = range.first + done;
                std::size_t const half = chunks_++ % 2;
                byte_t* const bounce_vectors = bounce.data() + half * half_bytes;
                byte_t* const bounce_neighbors = bounce_vectors + half_rows() * scalar_row_bytes;
                byte_t* const bounce_norms = bounce_neighbors + half_rows() * neighbors_row_bytes;
                // The host refills a half only once the copies out of it landed, while the other
                // half's copies run.
                if (chunks_ > 2 && failed(crossed_[half].wait())) return index_->fail_(status_t::device_error_k);
                [[maybe_unused]] status_t const staged_vectors = copy_bytes(
                    bytes_of(bounce_vectors, count * scalar_row_bytes),
                    vectors_rows.subspan(done * scalar_row_bytes, count * scalar_row_bytes));
                if (normed) [[maybe_unused]]
                    status_t const staged_norms = copy_bytes(
                        bytes_of(bounce_norms, count * sizeof(norms_t)),
                        norms_rows.subspan(done * sizeof(norms_t), count * sizeof(norms_t)));
                [[maybe_unused]] status_t const staged_neighbors = copy_bytes(
                    bytes_of(bounce_neighbors, count * neighbors_row_bytes),
                    neighbors_rows.subspan(done * neighbors_row_bytes, count * neighbors_row_bytes));
                std::span<node_id_t> const landed {reinterpret_cast<node_id_t*>(bounce_neighbors),
                                                   count * connectivity};
                rebase_neighbors(landed, source_range, shift, node_id_unknown_k);
                compact_neighbors(landed, connectivity, static_cast<node_id_t>(into), node_id_unknown_k);
                // The copy engine then reads the half from the last-level cache, not this core.
                index_->prefetcher_.demote(bounce_vectors, half_bytes);
                if (failed(vendor_t::copy_2d(index_->vectors_rows_() + into * index_->row_bytes(), index_->row_bytes(),
                                             bounce_vectors, scalar_row_bytes, scalar_row_bytes, count, stream)) ||
                    (normed && failed(vendor_t::copy(index_->norms_column_.base() + into * sizeof(norms_t),
                                                     bounce_norms, count * sizeof(norms_t), stream))) ||
                    failed(vendor_t::copy(index_->neighbors_matrix_.base() + into * neighbors_row_bytes,
                                          bounce_neighbors, count * neighbors_row_bytes, stream)) ||
                    failed(crossed_[half].record(stream)))
                    return index_->fail_(status_t::device_error_k);
            }
            if (normed)
                fu::standard_atomic_ref<std::size_t> {covered_norms}.add(range.size(), std::memory_order_acq_rel);
            fu::standard_atomic_ref<std::size_t> {covered}.add(range.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /**
         *  @brief The same copy out of @p source_shard 's own storage rather than host columns, as
         *      an absorb drives it, one donor shard at a time, its step naming the rows in that
         *      shard's row and ID spaces.
         *
         *  Both matrices keep fragment-padded rows of one geometry, so every column moves as one
         *  copy on the plan's stream, from another device's storage as well once its plural mapped
         *  it for this one; storage this device cannot reach is refused. The neighbors then rebase
         *  on the plan's stream. Nothing waits, so @p source_shard must stay quiescent until the
         *  caller drains that stream.
         *
         *  @param[in] step The step to fill; its source rows address @p source_shard.
         *  @param[in] source_shard The shard the rows come from, read through its core.
         *  @param[in] source_range The ID space the source's rows speak.
         *  @retval config_mismatch_k Another geometry, rows outside the plan, or storage on a
         *      device this one cannot reach.
         */
        template <typename shard_type_>
        status_t load(step_t const& step, shard_type_ const& source_shard, iota_view<node_id_t> source_range) noexcept {
            vram_shard_core const& source = source_shard.core();
            if (!index_) return status_t::uninitialized_k;
            int const device = index_->device_specs().ordinal;
            if (failed(vendor_t::select(device))) return status_t::device_error_k;
            iota_view<node_id_t> const range = step.index_rows;
            bool const normed = index_->stores_norms() && source.stores_norms();
            if (range.empty() || range.size() != step.source_rows.size() || range.first < base ||
                range.bound > base + rows || step.source_rows.bound > source.size() ||
                source.dimensions() != index_->dimensions() || source.connectivity() != index_->connectivity() ||
                !source.vectors_matrix_.shared_with(device) || !source.neighbors_matrix_.shared_with(device) ||
                (normed && !source.norms_column_.shared_with(device)))
                return status_t::config_mismatch_k;
            node_id_t const shift = range.first - step.source_rows.first + source_range.first;
            if (shift + source_range.size() > base + rows) return status_t::config_mismatch_k;
            std::size_t const row_bytes = index_->row_bytes();
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(node_id_t);
            std::size_t const count = range.size(), into = range.first, from = step.source_rows.first;
            if (failed(vendor_t::copy(index_->vectors_rows_() + into * row_bytes,
                                      source.vectors_rows_() + from * row_bytes, count * row_bytes, stream)) ||
                failed(vendor_t::copy(index_->neighbors_matrix_.base() + into * neighbors_row_bytes,
                                      source.neighbors_matrix_.base() + from * neighbors_row_bytes,
                                      count * neighbors_row_bytes, stream)) ||
                (normed && failed(vendor_t::copy(index_->norms_column_.base() + into * sizeof(norms_t),
                                                 source.norms_column_.base() + from * sizeof(norms_t),
                                                 count * sizeof(norms_t), stream))))
                return index_->fail_(status_t::device_error_k);
            if (failed(index_->launch_rebase_links_(range, source_range, shift, stream)))
                return index_->fail_(status_t::device_error_k);
            if (normed) fu::standard_atomic_ref<std::size_t> {covered_norms}.add(count, std::memory_order_acq_rel);
            fu::standard_atomic_ref<std::size_t> {covered}.add(count, std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Publishes the appended rows once every batch's vectors and neighbors landed: the norms
         *  derive on the device unless the metric is dot or every batch carried them, the row count
         *  lands, the claim lifts, and the watermark turns over the rows once every earlier batch
         *  has. A success spends the plan, so a second call answers @c uninitialized_k instead of
         *  publishing twice, and so does a turn refused by a failure an earlier batch kept; any
         *  other refusal leaves it whole, so the missing steps can still land. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (failed(vendor_t::select(index_->device_specs().ordinal))) return status_t::device_error_k;
            if (fu::standard_atomic_ref<std::size_t> {covered}.load(std::memory_order_acquire) != rows)
                return status_t::config_mismatch_k;
            if (rows) {
                // One launch derives every row, so a load mixing carried and missing norms
                // re-derives the carried.
                if (index_->stores_norms() &&
                    fu::standard_atomic_ref<std::size_t> {covered_norms}.load(std::memory_order_acquire) != rows &&
                    failed(index_->launch_derive_norms_(base, rows, stream)))
                    return index_->fail_(status_t::device_error_k);
                // A fill carries the count by value, so no host word must outlive a copy; the one
                // wait lands every step, the norms and the count before the watermark turns.
                if (failed(
                        vendor_t::fill_words(index_->published_rows_(), static_cast<u32_t>(base + rows), 1, stream)) ||
                    failed(crossed_[0].record(stream)) || failed(crossed_[0].wait()))
                    return index_->fail_(status_t::device_error_k);
            }
            vram_shard_core* const index = std::exchange(index_, nullptr);
            return rows ? index->publish_(base, rows) : status_t::success_k;
        }

      private:

        /** Marks the last copies out of each half of the bounce, born with the plan. */
        event_t crossed_[2];

        /** Halves of rows the steps moved through the bounce so far, alternating between them. */
        std::size_t chunks_ = 0;

        /** The index appended to; null in the empty plan a refused @ref plan_load returns, and once
         *  @c finish published. */
        vram_shard_core* index_ = nullptr;
    };

    /**
     *  @brief Opens a load of @p rows at the watermark on @p stream: claims them, or lands them
     *      under the claim word its caller closed, and grows the storage to hold them - the fresh
     *      neighbor rows arrive sentinel-filled.
     *
     *  Its steps source the rows from host columns, as a sharded plan drives its parts, or from
     *  other indexes, as an absorb does; a plan whose steps cross host memory pins its bounce
     *  first. Refuses an unreserved index, a kept failure, a batch or load in flight, and rows past
     *  the ID space.
     *
     *  @param[in] rows Rows to append.
     *  @param[in] stream The stream every copy and the norms kernel queue on.
     */
    expected<load_plan_t> plan_load(std::size_t rows, typename executor_t::stream_t const& stream) noexcept {
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (failed(vendor_t::select(device_.ordinal))) return {{}, status_t::device_error_k};
        if (!rows) return {load_plan_t {*this, stream, static_cast<node_id_t>(claimed_()), 0}, status_t::success_k};
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
        load_plan_t plan {*this, stream, base, rows};
        if (status_t const grown = grow_rows_(base + rows, stream.handle); failed(grown)) return {{}, grown};
        return {std::move(plan), status_t::success_k};
    }

    /** Appends the whole source on the calling thread - the one-shot over @ref plan_load for a
     *  caller with no streams to fan the steps out over; columns shorter than @p rows are refused
     *  before a row is claimed. */
    status_t load(columns_view_t columns, std::size_t rows, iota_view<node_id_t> source_range,
                  typename executor_t::stream_t const& stream) noexcept {
        if (!vectors_matrix_.reserved_bytes()) return status_t::uninitialized_k;
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return status_t::config_mismatch_k;
        auto [plan, status] = plan_load(rows, stream);
        if (failed(status)) return status;
        if (status_t const pinned = plan.pin_bounce(); failed(pinned)) return pinned;
        for (typename load_plan_t::step_t const step : plan.steps())
            if (status_t const loaded = plan.load(step, columns, source_range); failed(loaded)) return loaded;
        return plan.finish();
    }

#pragma endregion Checkpoints

#pragma region Insertion

    /** The submission half of @c insert: stages, grows and launches @p new_rows at @p first_id, and
     *  returns with their IDs, the rows invisible until the caller drains and publishes them. The
     *  caller holds the claim over them, and gives it back on a refusal, since nothing launched.
     *  Back-to-back submissions on one context are serialized by its stream and add into its
     *  telemetry, which the caller zeroes; across devices they overlap freely. */
    expected<insert_result_t> submit_insert_(rows_view_t new_rows, node_id_t first_id, node_ids_span_t assigned_ids,
                                             traversal_context_t& context, insert_config_t config) noexcept {
        std::size_t const rows = new_rows.rows;
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (status_t const refusal = rows_refusal_(new_rows); failed(refusal)) return {{}, refusal};
        if (assigned_ids.size() && assigned_ids.size() != rows) return {{}, status_t::config_mismatch_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        stream_handle_t const stream = context.stream();
        std::size_t const import_bytes = import_bytes_(rows, new_rows.dtype);
        // Past the rows and their import window: the chain heads over every row, then the batch's
        // pick links, the touched nodes and their count.
        std::size_t const total_rows = first_id + rows, picks = rows * connectivity();
        std::size_t const links_offset = round_up_to_multiple<std::size_t>(rows * row_bytes() + import_bytes, 16);
        std::size_t const links_bytes = (total_rows + 2 * picks + 1) * sizeof(u32_t);
        byte_t* const staging = context.ensure_staging(links_offset + links_bytes);
        if (!staging) return {{}, status_t::bad_alloc_k};
        u32_t* const link_heads = reinterpret_cast<u32_t*>(staging + links_offset);
        if (failed(vendor_t::fill(link_heads, 0xFF, total_rows * sizeof(u32_t), stream)))
            return {{}, status_t::device_error_k};
        if ((import_bytes || packed_row_bytes() != row_bytes()) &&
            failed(vendor_t::fill(staging, 0, rows * row_bytes() + import_bytes, stream)))
            return {{}, status_t::device_error_k};
        if (status_t const imported = import_rows_(staging, new_rows, staging + rows * row_bytes(), stream);
            failed(imported))
            return {{}, imported};

        // Every inserting warp owns an arena slice for its exact visited set.
        u32_t const expansion = static_cast<u32_t>(vram_insert_expansion(config.expansion_factor));
        u32_t const candidate_slots = candidate_slots_for_(expansion, 0);
        u32_t const selection_rows = walk_selection_rows_(expansion, candidate_slots);
        gpu_arena_shape_t const arena_shape = build_arena_shape_(expansion, candidate_slots, 1);
        std::size_t const arena_warps = device_.multiprocessor_count *
                                        kernels_type_::resident_blocks_(vram_kernel_t::insert_k, candidate_slots) *
                                        vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes<node_id_t>());
        if (!arena) return {{}, status_t::bad_alloc_k};

        // The claim admits no other grower, and a rejected grid publishes nothing, so the caller
        // un-claims it exactly.
        if (status_t const grown = grow_rows_(total_rows, stream); failed(grown)) return {{}, grown};
        // Monotone, so an un-claim below only forfeits epoch space.
        auto const [epoch_base, epochs_status] = context.take_epochs(rows, arena_bytes_before);
        if (failed(epochs_status)) return {{}, epochs_status};
        vram_build_t const build {storage(),
                                  staging,
                                  static_cast<u32_t>(total_rows),
                                  arena_shape.scratch_bytes ? 0u : build_scratch_bytes_per_warp(selection_rows),
                                  selection_rows,
                                  first_id,
                                  arena,
                                  arena_shape,
                                  expansion,
                                  epoch_base,
                                  published_rows_(),
                                  context.telemetry_.get(),
                                  context.assigned_rows_.get(),
                                  link_heads,
                                  link_heads + total_rows,
                                  link_heads + total_rows + picks,
                                  link_heads + total_rows + 2 * picks};
        // The launch is the last fallible step; reading its verdict here keeps the claim
        // reversible.
        if (failed(kernels_.launch_insert_(*this, build, candidate_slots, context)))
            return {{}, status_t::device_error_k};

        insert_result_t result;
        result.first_id = first_id;
        for (std::size_t row = 0; row != assigned_ids.size(); ++row)
            assigned_ids[row] = static_cast<node_id_t>(first_id + row);
        result.new_size = total_rows;
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
     *  @p stream, over @p staging and @p window already zeroed. Rows in the stored scalar take one
     *  pitched copy. Others cross a window at a time: a pitched copy pads them to the stored pitch,
     *  and one flat cast lands the window in place, its zeroed pad values casting to zero pad
     *  bytes, or a unit-RMS narrowing writes the columns alone. Repeated rows import once and
     *  double on the device. */
    status_t import_rows_(byte_t* staging, rows_view_t rows, byte_t* window, stream_handle_t stream) const noexcept {
        std::size_t const distinct = rows.stride ? rows.rows : 1;
        std::size_t const source_row_bytes = packed_bytes(rows.dtype, rows.columns);
        std::size_t const source_pitch = rows.stride ? rows.stride : source_row_bytes;
        if (rows.dtype == vector_dtype_) {
            if (failed(vendor_t::copy_2d(staging, row_bytes(), rows.data, source_pitch, source_row_bytes, distinct,
                                         stream)))
                return status_t::device_error_k;
        }
        else {
            std::size_t const pitch_values = pitch_values_();
            std::size_t const window_rows = min_of(distinct, import_window_rows_k);
            std::size_t const window_row_bytes = pitch_values * nk_dtype_bits(rows.dtype) / 8;
            f32_t* const widened = reinterpret_cast<f32_t*>(window + window_rows * window_row_bytes);
            for (std::size_t first = 0; first < distinct; first += window_rows) {
                std::size_t const count = min_of(window_rows, distinct - first);
                byte_t* const landing = staging + first * row_bytes();
                if (failed(vendor_t::copy_2d(window, window_row_bytes, rows.row(first), source_pitch, source_row_bytes,
                                             count, stream)))
                    return status_t::device_error_k;
                status_t landed = status_t::success_k;
                if (!narrows_at_unit_rms_(rows.dtype))
                    landed = kernels_.launch_cast_(*this, window, rows.dtype, landing, vector_dtype_,
                                                   count * pitch_values, stream);
                else if (rows.dtype == nk_f32_k)
                    landed = launch_rmscast_(reinterpret_cast<f32_t const*>(window), window_row_bytes, landing, count,
                                             stream);
                else {
                    landed = kernels_.launch_cast_(*this, window, nk_f64_k, widened, nk_f32_k, count * pitch_values,
                                                   stream);
                    if (succeeded(landed))
                        landed = launch_rmscast_(widened, pitch_values * sizeof(f32_t), landing, count, stream);
                }
                if (failed(landed)) return landed;
            }
        }
        for (std::size_t copied = distinct; copied < rows.rows; copied *= 2)
            if (failed(vendor_t::copy(staging + copied * row_bytes(), staging,
                                      min_of(copied, rows.rows - copied) * row_bytes(), stream)))
                return status_t::device_error_k;
        return status_t::success_k;
    }

    /** Why an attend over @p rows rows cannot land in @p outputs and @p carry: shapes other than
     *  the rows', a scalar the device casts cannot carry f32 into, or rows closer than one row
     *  apart. */
    status_t landing_refusal_(std::size_t rows, rows_span_t outputs,
                              std::span<softmax_state_t const> carry) const noexcept {
        if (outputs.rows != rows || outputs.columns != dimensions() || (!carry.empty() && carry.size() != rows))
            return status_t::config_mismatch_k;
        if (outputs.dtype != nk_f32_k && !casts_(nk_f32_k, outputs.dtype)) return status_t::unsupported_k;
        if (outputs.stride < packed_bytes(outputs.dtype, dimensions())) return status_t::unsupported_layout_k;
        return status_t::success_k;
    }

    /** Bytes the attended rows take past the f32 blocks once cast into the scalar of @p outputs;
     *  none when they land as f32. */
    std::size_t exported_bytes_(rows_span_t outputs) const noexcept {
        return outputs.dtype == nk_f32_k ? 0 : outputs.rows * packed_bytes(outputs.dtype, dimensions());
    }

    /** Queues the copy of the attended rows of the first of @p blocks blocks at @p landed into
     *  @p outputs on @p stream, cast past the blocks first unless they land as f32, then of the
     *  carries past the rows into @p carry unless it is empty. */
    status_t land_attended_(f32_t* landed, std::size_t blocks, rows_span_t outputs, std::span<softmax_state_t> carry,
                            stream_handle_t stream) const noexcept {
        std::size_t const values = outputs.rows * dimensions();
        std::size_t const exported_row_bytes = packed_bytes(outputs.dtype, dimensions());
        byte_t* exported = reinterpret_cast<byte_t*>(landed);
        if (outputs.dtype != nk_f32_k) {
            exported += blocks * (values * sizeof(f32_t) + outputs.rows * sizeof(softmax_state_t));
            if (status_t const cast = kernels_.launch_cast_(*this, landed, nk_f32_k, exported, outputs.dtype, values,
                                                            stream);
                failed(cast))
                return cast;
        }
        if (failed(vendor_t::copy_2d(outputs.data, outputs.stride, exported, exported_row_bytes, exported_row_bytes,
                                     outputs.rows, stream)) ||
            (!carry.empty() && failed(vendor_t::copy(carry.data(), landed + values, carry.size_bytes(), stream))))
            return status_t::device_error_k;
        return status_t::success_k;
    }

    /** Blocking drains the context's stream and adds the telemetry the walks spent; deferred
     *  records the submission for @c context.landed and returns it as queued. */
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

    /** A zeroed landing from @p context 's stream slots, or null when none can pin. */
    static audit_landing_ptr_t take_landing_(traversal_context_t& context) noexcept {
        typename vendor_t::landings_t* const landings = context.allocator_.landings();
        void* const slot = landings ? landings->take() : nullptr;
        return {slot ? new (slot) audit_landing_t {} : nullptr, landing_return_t {landings}};
    }

    /** Which share of the scratch a carve covers: the audit's alone, or the whole of a refine's. */
    enum class scratch_use_t : bool { audit_k, refine_k };

    /** Bytes of the scratch over the rows published now: the audit share, and for @c refine_k the
     *  islands' IDs and @p parts bases too. */
    std::size_t scratch_bytes_(scratch_use_t use, std::size_t parts = 0) const noexcept {
        std::size_t const rows = size(), words = divide_round_up<std::size_t>(rows, 32);
        std::size_t const refine_words = use == scratch_use_t::refine_k ? rows + parts : 0;
        return (words * 3 + 7 + refine_words) * sizeof(u32_t);
    }

    /** Bytes of the staging one refinement pass carves over the rows published now. */
    std::size_t refine_staging_bytes_() const noexcept {
        std::size_t const rows = size(), words = divide_round_up<std::size_t>(rows, 32);
        return (rows + words + 2 * rows * connectivity()) * sizeof(u32_t);
    }

    /** The buffers one refinement pass uses up, from @p context 's staging, which @ref plan_refine
     *  grew for them; all null when it could not. */
    vram_refine_passes_t refine_passes_(traversal_context_t& context) const noexcept {
        vram_refine_passes_t passes;
        std::size_t const rows = size(), words = divide_round_up<std::size_t>(rows, 32);
        byte_t* const staging = context.ensure_staging(refine_staging_bytes_());
        if (!staging) return passes;
        passes.link_heads = reinterpret_cast<u32_t*>(staging);
        passes.walked.words = passes.link_heads + rows;
        passes.shadow_neighbors = passes.walked.words + words;
        passes.link_next = passes.shadow_neighbors + rows * connectivity();
        return passes;
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
        scratch.levels = scratch.next_frontier.words + scratch.words;
        scratch.expanded = scratch.levels + 1;
        scratch.unreached = scratch.expanded + 2;
        scratch.reconnected = scratch.unreached + 1;
        scratch.claims = scratch.reconnected + 1;
        if (use == scratch_use_t::refine_k) {
            scratch.island_ids = scratch.claims + 2;
            scratch.part_bases = scratch.island_ids + rows;
        }
        if (failed(vendor_t::fill(scratch.reconnected, 0, sizeof(u32_t), context.stream())))
            return {std::move(scratch), status_t::device_error_k};
        return {std::move(scratch), status_t::success_k};
    }

    /** Queues the copy of @p scratch 's counters into @p landing: the levels, the count of sweep
     *  @p last_sweep, then the unreached, reconnected and claim counters, which lie back to
     *  back. */
    static status_t land_counters_(refine_scratch_t const& scratch, audit_landing_t* landing, u32_t last_sweep,
                                   stream_handle_t stream) noexcept {
        return succeeded(vendor_t::copy(&landing->levels, scratch.levels, sizeof(u32_t), stream)) &&
                       succeeded(vendor_t::copy(&landing->last_expanded, scratch.expanded + (last_sweep & 1u),
                                                sizeof(u32_t), stream)) &&
                       succeeded(vendor_t::copy(&landing->unreached, scratch.unreached,
                                                sizeof(audit_landing_t) - 2 * sizeof(u32_t), stream))
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** The converged audit @p landing reports over @p rows rows. */
    static health_result_t audited_(audit_landing_t const& landing, std::size_t rows) noexcept {
        // A sweep that expands nothing ends the levels, and it counts as one, as on the CPU.
        return {rows, rows - landing.unreached, std::size_t {landing.levels} + 1};
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

    /** Forgets every row under the closed claim word: the watermark drops to zero, a kept failure
     *  with it, and the next claim sentinel-fills every mapped row again and zeroes the published
     *  count, so IDs restart from zero on the storage already mapped, which nothing frees. A walk
     *  still reaching the index reads that retained storage, and its caller's epoch discards it. */
    void forget_() noexcept {
        fu::standard_atomic_ref<std::size_t>(last_visible_id_).store(0, std::memory_order_release);
        fu::standard_atomic_ref<status_t>(status_).store(status_t::success_k, std::memory_order_release);
        filled_rows_ = 0;
    }

    /** Turns the watermark from @p first over @p rows drained rows once every batch claimed before
     *  has turned it, so the visible rows are always whole batches; a failure kept meanwhile
     *  answers instead, the rows invisible. */
    status_t publish_(std::size_t first, std::size_t rows) noexcept {
        for (std::size_t watermark = size(); watermark != first; watermark = size()) {
            // A batch that failed before this one never turns the watermark, so its kept status
            // ends the wait.
            if (status_t const kept = status(); failed(kept)) return kept;
            fu::standard_yield_t {}(&last_visible_id_, watermark, 0u, fu::wait_capped_t {});
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

    /** Maps every matrix and column up to @p rows - host work alone, so it rides no stream, and
     *  leaves the fresh neighbor rows for @ref grow_rows_ to fill. */
    status_t map_rows_(std::size_t rows) noexcept {
        if (rows <= capacity()) return status_t::success_k;
        if (status_t status = vectors_matrix_.commit((rows + 1) * row_bytes()); status != status_t::success_k)
            return status;
        if (status_t status = kernel_ == nk_kernel_dot_k ? status_t::success_k
                                                         : norms_column_.commit(rows * sizeof(norms_t));
            status != status_t::success_k)
            return status;
        return neighbors_matrix_.commit(rows * connectivity() * sizeof(u32_t));
    }

    /** Maps up to @p rows and sentinel-fills on @p stream every mapped neighbor row no fill reached
     *  yet - always ahead of any launch that could assign new_node_ids into them - zeroing the
     *  published count with the first fill. @p stream first follows the last fill, whichever stream
     *  queued it, so a claim never launches into rows still filling. A refused fill advances
     *  nothing, so the next grow retries it. Callers hold the claim over the rows, or the closed
     *  claim word. */
    status_t grow_rows_(std::size_t rows, stream_handle_t stream) noexcept {
        if (status_t const mapped = map_rows_(rows); failed(mapped)) return mapped;
        if (failed(vendor_t::follow(stream, grown_))) return status_t::device_error_k;
        if (rows <= filled_rows_) return status_t::success_k;
        std::size_t const mapped_rows = capacity(), row_words = connectivity();
        if (!filled_rows_ && failed(vendor_t::fill_words(published_rows_(), 0, 1, stream)))
            return status_t::device_error_k;
        if (failed(vendor_t::fill_words(neighbors_matrix_.base() + filled_rows_ * row_words * sizeof(u32_t),
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
    u32_t walk_selection_rows_(u32_t expansion, u32_t candidate_slots) const noexcept {
        return candidate_slots ? candidate_slots * kernels_type_::wave_lanes() : expansion;
    }

    /** Candidates a re-selection caches, a back-link round's or a refinement merge's: a row of the
     *  node's own neighbors or forward picks, a row's worth of incoming newcomers or reverse
     *  proposals, and the node's own row last. */
    u32_t reselection_rows_() const noexcept { return 2 * static_cast<u32_t>(connectivity()) + 1; }

    /** Bytes of one builder warp's slice, its walk's or its re-selection's, whichever caches more:
     *  an insert and its back-link, or a refine and its merge, share one arena shape. */
    u32_t builder_scratch_bytes_per_warp_(u32_t expansion, u32_t candidate_slots) const noexcept {
        return build_scratch_bytes_per_warp(
            max_of(walk_selection_rows_(expansion, candidate_slots), reselection_rows_()));
    }

    /** One building warp's arena slice for @p walks walks on the rung of @p candidate_slots, the
     *  arena frontier's over @p expansion at zero, then the builders' scratch when shared memory
     *  cannot fit it. */
    gpu_arena_shape_t build_arena_shape_(u32_t expansion, u32_t candidate_slots, u32_t walks) const noexcept {
        u32_t const scratch_bytes = builder_scratch_bytes_per_warp_(expansion, candidate_slots);
        gpu_arena_shape_t shape = arena_shape_for(
            walk_selection_rows_(expansion, candidate_slots),
            candidate_slots ? frontier_kind_t::register_ladder_k : frontier_kind_t::wide_arena_k, walks);
        if (!kernels_.build_scratch_fits(*this, scratch_bytes)) shape.scratch_bytes = scratch_bytes;
        return shape;
    }

    /** Rebuilds the norms of rows @b [first_row,first_row+rows) from their vectors. */
    status_t launch_derive_norms_(std::size_t first_row, std::size_t rows, stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.graph.storage = storage();
        arguments.graph.first_id = static_cast<u32_t>(first_row);
        arguments.graph.rows_count = static_cast<u32_t>(rows);
        return kernels_.launch_(
            vram_dense_kernel_t::derive_norms_k, 0,
            device_.multiprocessor_count * kernels_type_::resident_blocks_(vram_dense_kernel_t::derive_norms_k, 0), 0,
            stream, arguments);
    }

    /** Rebases rows @p rows of the neighbors matrix from @p source_range onto the IDs from @p shift
     *  and compacts them, as a load through the bounce does on the host. */
    status_t launch_rebase_links_(iota_view<node_id_t> rows, iota_view<node_id_t> source_range, node_id_t shift,
                                  stream_handle_t stream) noexcept {
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
            vram_kernel_t::rebase_links_k, 0,
            device_.multiprocessor_count * kernels_type_::resident_blocks_(vram_kernel_t::rebase_links_k, 0), 0, stream,
            arguments);
    }

    /** Merges the attended rows and carries of @p shards walks into the first's on @p stream - the
     *  sharded attend's gather. */
    status_t launch_merge_attention_(byte_t* blocks, u32_t rows, u32_t columns, u32_t shards,
                                     stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.attend.outputs = reinterpret_cast<f32_t*>(blocks);
        arguments.attend.rows = rows;
        arguments.attend.columns = columns;
        arguments.attend.shards = shards;
        return kernels_.launch_(vram_dense_kernel_t::merge_attention_k, 0,
                                divide_round_up<u32_t>(rows, vram_warps_per_block_k), 0, stream, arguments);
    }

    /** Resolves the @p walked matches per query of @p queries_count at @p ids and @p distances
     *  through the device @p keys column into @p ranking on @p stream - a sharded search's own
     *  ranking, gathered or landed as is. */
    status_t launch_resolve_matches_(node_id_t const* ids, distance_t const* distances, u32_t walked,
                                     byte_t const* keys, vram_ranking_t const& ranking, u32_t queries_count,
                                     stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.matches.ids = ids;
        arguments.matches.distances = distances;
        arguments.matches.keys = keys;
        arguments.matches.ranking = ranking;
        arguments.matches.walked = walked;
        arguments.matches.queries_count = queries_count;
        return kernels_.launch_(vram_dense_kernel_t::resolve_matches_k, 0,
                                divide_round_up<u32_t>(queries_count, vram_warps_per_block_k), 0, stream, arguments);
    }

    /** Merges @p shards rankings of @p queries_count queries, gathered @p block_bytes apart from
     *  @p gathered, into @p ranking on @p stream - the sharded search's device gather. */
    status_t launch_merge_matches_(vram_ranking_t const& gathered, std::size_t block_bytes, u32_t shards,
                                   vram_ranking_t const& ranking, u32_t queries_count,
                                   stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.matches.gathered = gathered;
        arguments.matches.ranking = ranking;
        arguments.matches.block_bytes = block_bytes;
        arguments.matches.queries_count = queries_count;
        arguments.matches.shards = shards;
        return kernels_.launch_(vram_dense_kernel_t::merge_matches_k, 0,
                                divide_round_up<u32_t>(queries_count, vram_warps_per_block_k), 0, stream, arguments);
    }

    /** Narrows @p rows rows of the index's columns at @p from, @p from_stride bytes apart, into
     *  @p to at the vectors matrix pitch at unit RMS on @p stream, one block per row. */
    status_t launch_rmscast_(f32_t const* from, std::size_t from_stride, byte_t* to, std::size_t rows,
                             stream_handle_t stream) const noexcept {
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
        return kernels_.launch_(vram_dense_kernel_t::rmscast_k, 0, blocks, 0, stream, arguments);
    }

#pragma endregion Launches

  private:
    friend kernels_type_;

    /** The vectors matrix and norms column backend, rebound to bytes. */
    using vectors_matrix_allocator_t = typename vectors_allocator_t::template rebind<byte_t>::other;

    /** The neighbors matrix backend, rebound to bytes. */
    using neighbors_matrix_allocator_t = typename neighbors_allocator_t::template rebind<byte_t>::other;

    /** Hardware identity: the ordinal comes from the vectors allocator at construction, the
     *  remaining specs fill at the first @c reserve. */
    device_specs_t device_;

    /** The metric every walk scores by; under dot, the one metric that reads no norms, the norms
     *  column stays empty. */
    nk_kernel_kind_t kernel_;

    /** The frozen geometry word: dimensions high, connectivity low, zero when unreserved. */
    packed_geometry_id_t geometry_ = 0;

    /** The vectors matrix, its first row the published-row count, and the norms column; each
     *  allocator is its storage, owning the base and the committed extent, so capacities are
     *  derived, never cached. */
    vectors_matrix_allocator_t vectors_matrix_, norms_column_;

    /** The neighbors matrix, on the same terms. */
    neighbors_matrix_allocator_t neighbors_matrix_;

    /** The claim word, closed while a refine, an absorb, a load or a save of the sharded index
     *  holds it. */
    claim_word_t claims_;

    /** Mapped rows whose neighbors a grow sentinel-filled, read and advanced by the claim's holder
     *  alone; a reservation maps past it and the next claim fills the rest. */
    std::size_t filled_rows_ = 0;

    /** Rows the reserved ranges span, the most any claim reaches; fixed by the first @c reserve. */
    std::size_t ceiling_rows_ = max_node_count_k;

    /** Marks the last sentinel fill, or the copies of a load that died over filled rows, which
     *  every grow's stream follows; recorded and followed by the claim's holder alone. */
    event_t grown_;

    /** Rows strictly below it are published and visible: turned once per batch, in claim order,
     *  after its drain. */
    std::size_t last_visible_id_ = 0;

    /** The first post-claim device failure: IDs past it may be unpublished or unfilled. */
    status_t status_ = status_t::success_k;

    /** The vendor's kernels, loaded at the first @c reserve. */
    kernels_type_ kernels_;

    /** Demotes a filled bounce half toward the last-level cache, where the copy engine reads it. */
    object_prefetcher<cache_hints_type_> prefetcher_;
};

} // namespace unum::usearch::dense
