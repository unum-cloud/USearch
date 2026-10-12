/**
 *  @file include/usearch/dense/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *
 *  The host face of the NVIDIA family, compiled by any C++ compiler: it fills plain argument
 *  records and launches the kernels of @c vram_nvidia_index.cuh by name through the driver it opens
 *  at first use, so it never sees a kit, a shared view or the CUDA runtime. Every kernel is one
 *  instantiation of a single template over five integers - the kernel, the capability ceiling, the
 *  dtype, the register rung and the attend value lookup - so its mangled name is spelled here from
 *  those integers alone, and one fatbin per ceiling carries every dtype.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::snprintf` - a kernel's mangled name
#include <cstring> // `std::memcpy` - an attend's value lookup travels as bytes

#include <atomic>      // `std::atomic` row claiming across concurrent inserting contexts
#include <mutex>       // `std::unique_lock` while one context grows the storage and claims
#include <bit>         // `std::bit_width` for hash-prefix widths, `std::has_single_bit` for connectivity
#include <span>        // `std::span` - the run a context maps its buffers onto
#include <type_traits> // `std::is_same_v` - which heads a walk stages
#include <utility>     // `std::exchange` - a moved-from plan forgets its index
#include <numbers>     // `std::numbers::log2e_v` - the base-2 softmax

#include <forkunion/types.hpp> // `fu::round_up_to_multiple` & friends

#include <numkong/capabilities.h> // `nk_kernel_kind_t` & the kernel-kind constants
#include <numkong/tensor.hpp>     // `nk::matrix_span`, `nk::range` & `nk::slice` - the batches' views
#include <numkong/types.hpp>      // `nk::type_for` - the stored scalar of a dtype
#include <numkong/vector.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_nvidia_device.hpp"
#include "usearch/dense/types.hpp"

namespace unum::usearch::dense {

/** Bytes of one packed traversal candidate on the device, the kernels' scored node. */
inline constexpr std::size_t cuda_scored_node_bytes_k = sizeof(u64_t);

/** Layout of one warp's slice of the traversal arena, computed on the host: the exact visited set,
 *  then the wide frontier's spill tail when it has one. The kernels carry it as their own
 *  @c vram_arena_shape_t. */
struct cuda_arena_shape_t {
    u32_t visited_slots;
    u32_t visited_hash_shift;
    u32_t tail_entries;

    constexpr std::size_t slice_bytes() const noexcept {
        return static_cast<std::size_t>(visited_slots) * sizeof(u64_t) +
               static_cast<std::size_t>(tail_entries) * cuda_scored_node_bytes_k;
    }
};

/** Bytes of one walking warp's attend scratch over @p heads_count heads of @p columns columns:
 *  every head's batch dots, its running maximum, partition and latest rescale, then its accumulator
 *  row, in whole fragments so the next warp's query span stays aligned. */
inline constexpr std::size_t cuda_attend_scratch_bytes(std::size_t heads_count, std::size_t columns) noexcept {
    return round_up_to_multiple<std::size_t>(heads_count * (32 + 3 + columns) * sizeof(f32_t), 16);
}

/** Sixteen bytes every NVIDIA kit loads at once - the host's spelling of the kernels' @c uint4,
 *  which only the CUDA SDK declares. */
struct alignas(16) cuda_fragment_t {
    u32_t x, y, z, w;
};

/**
 *  @brief The host-visible half of the quads kit @c quads_chain picks for @p vector_dtype_
 *      under @p capability_: the types and record sizes the host sizes launches by.
 *
 *  Every NVIDIA kit loads 16-byte fragments and keeps the CPU's norms over its dot type, and the
 *  pick follows the chains' tiers; the kernel side checks both against the kit itself.
 */
template <nk_dtype_t vector_dtype_, nk_capability_t capability_>
struct cuda_kit_traits {
    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using dot_t = typename scalar_t::dot_result_t::raw_t;
    using fragment_t = cuda_fragment_t;
    using norms_t = vector_norms<dot_t>;

    /** Bytes of one warp's staged batch, the kernels' @c vram_gathered: a cache of candidate
     *  IDs, then a warp of scored nodes and a warp of native dots. */
    inline static constexpr std::size_t gathered_bytes_k = vram_cache_rows_k * sizeof(u32_t) +
                                                           32 * (cuda_scored_node_bytes_k + sizeof(dot_t));

    /** Bytes of one warp's static schedule state: Hopper's and Blackwell's barrier stages, or the
     *  Ampere schedule's one flag. */
    inline static constexpr std::size_t schedule_bytes_k =
        (capability_ & (nk_cap_hopper_k | nk_cap_blackwell_k | nk_cap_blackwellrtx_k)) != 0 ? 16 : 1;

    /** The capability of the kit the chain picks: Blackwell's for the half floats and minifloats
     *  where that tier is allowed, Ada's for the fp8 pair, and the Ampere floor otherwise. */
    inline static constexpr nk_capability_t picked_k =
        (vector_dtype_ == nk_f16_k || vector_dtype_ == nk_bf16_k || vector_dtype_ == nk_e5m2_k ||
         vector_dtype_ == nk_e4m3_k || vector_dtype_ == nk_e3m2_k || vector_dtype_ == nk_e2m3_k) &&
                (capability_ & (nk_cap_blackwellrtx_k | nk_cap_blackwell_k)) != 0
            ? nk_cap_blackwell_k
        : (vector_dtype_ == nk_e5m2_k || vector_dtype_ == nk_e4m3_k) && (capability_ & nk_cap_ada_k) != 0
            ? nk_cap_ada_k
            : nk_cap_ampere_k;
};

/** Sets the keyed twin of a value lookup apart from the bare one in an attend kernel's name. */
inline constexpr u32_t cuda_keyed_lookup_k = 1u << 16;

/** A key-level value lookup seen through one shard's device key column, so the shard's walks
 *  resolve node IDs: the kernels read a node's row as @c value_lookup(keys[node_id]). */
template <typename key_type_, typename value_lookup_type_>
struct cuda_keyed_lookup {

    /** Names this lookup to the attend kernels, apart from the bare key-level one. */
    inline static constexpr u32_t device_lookup_k = value_lookup_type_::device_lookup_k | cuda_keyed_lookup_k;

    /** The shard's keys on its device, by node ID. */
    key_type_ const* keys;

    /** Resolves a key to its value row. */
    value_lookup_type_ value_lookup;
};

/** The device value lookup an attend kernel resolves rows through, by the @c device_lookup_k a
 *  lookup type names itself with: whichever unit compiles a fatbin specializes it with @c type for
 *  every lookup that fatbin's attend kernels take. */
template <u32_t lookup_>
struct cuda_value_lookup_of;

#pragma region Kernel Records

/** Bytes of a value lookup an attend carries to its kernels. */
inline constexpr std::size_t cuda_value_lookup_capacity_k = 64;

/** Default dynamic-shared opt-in threshold: launches asking for more must raise the kernel's
 *  attribute before the first launch. */
inline constexpr u32_t cuda_static_scratch_limit_k = 48 * 1024;

/** One bit per row over device-resident words, marked concurrently by a whole grid. */
struct cuda_bitset_t {
    u32_t* words = nullptr;
};

/** The flat storage a kernel reads, mirroring the shared flat view field by field. */
struct cuda_flat_storage_t {
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
struct cuda_build_t {
    cuda_flat_storage_t storage;
    byte_t const* staging = nullptr;
    u32_t end_id = 0;
    u32_t scratch_bytes_per_warp = 0;
    node_id_t first_id = 0;
    byte_t* arena = nullptr;
    cuda_arena_shape_t arena_shape {};
    u32_t epoch_base = 0;
    u32_t* published_rows = nullptr;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_rows = nullptr;
};

/** One walking launch, mirroring the shared search view. */
struct cuda_search_t {
    cuda_flat_storage_t storage;
    byte_t const* queries = nullptr;
    u32_t queries_count = 0;
    u32_t sampling_bound = 0;
    u32_t scratch_bytes_per_warp = 0;
    u32_t expansion = 0;
    byte_t* arena = nullptr;
    cuda_arena_shape_t arena_shape {};
    u32_t epoch_base = 0;
    vram_telemetry_block_t* telemetry = nullptr;
};

/** One refinement launch, mirroring the shared refine view. */
struct cuda_refine_t {
    cuda_flat_storage_t storage;
    u32_t* shadow_neighbors = nullptr;
    u32_t end_id = 0;
    u32_t scratch_bytes_per_warp = 0;
    node_id_t first_id = 0;
    byte_t* arena = nullptr;
    cuda_arena_shape_t arena_shape {};
    u32_t epoch_base = 0;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_rows = nullptr;
    u32_t const* node_ids = nullptr;
    u32_t sampling_bound = 0;
};

/** A search's landing, mirroring the shared search sink. */
struct cuda_search_sink_t {
    node_id_t* result_ids = nullptr;
    distance_t* result_distances = nullptr;
    u32_t count = 0;
    nk_kernel_kind_t kernel = nk_kernel_dot_k;
};

/** A graph product's landing, mirroring the shared multiply sink; the products are the kit's
 *  dots. */
struct cuda_multiply_sink_t {
    node_id_t* ids = nullptr;
    byte_t* products = nullptr;
    u32_t* counts = nullptr;
    u32_t wanted = 0;
};

/** The heads a walk rescores by: an indexer's with their weights, or an attend's with none. */
struct cuda_rescoring_t {
    byte_t const* heads = nullptr;
    f32_t const* weights = nullptr;
    u32_t heads_count = 0;
    u32_t row_fragments = 0;
};

/** What the graph kernels - the scatter, the merge's buckets, the norms and the audits - read. */
struct cuda_graph_t {
    cuda_flat_storage_t storage;
    u32_t* neighbors = nullptr;
    u32_t rows_count = 0;
    u32_t connectivity = 0;
    u32_t first_id = 0;
    u32_t walk_count = 0;
    u32_t const* node_ids = nullptr;
    u32_t bucket_width = 0;
    u32_t* bucket_ids = nullptr;
    u32_t* bucket_counts = nullptr;
    cuda_bitset_t touched, reached, frontier, next_frontier;
    u32_t* counter = nullptr;
    u32_t* island_ids = nullptr;
    u32_t islands_count = 0;
    u32_t seed_count = 0;
    u32_t* in_links = nullptr;
    u32_t* reconnected = nullptr;
};

/** An attend's landing and the value lookup its walks resolve rows through, carried as the lookup's
 *  own bytes; the sharded gather reads the landing as @p shards blocks of @p rows rows. */
struct cuda_attend_t {
    f32_t* outputs = nullptr;
    softmax_state_t* states = nullptr;
    u32_t* missed = nullptr;
    f32_t logit_scale = 0;
    u32_t heads_count = 0;
    u32_t columns = 0;
    u32_t rows = 0;
    u32_t shards = 0;
    alignas(16) byte_t value_lookup[cuda_value_lookup_capacity_k] = {};
};

/** The one argument every NVIDIA kernel takes, each reading the records its kernel names. */
struct cuda_arguments_t {
    cuda_build_t build;
    cuda_search_t search;
    cuda_refine_t refine;
    cuda_search_sink_t search_sink;
    cuda_multiply_sink_t multiply_sink;
    cuda_rescoring_t rescoring;
    cuda_graph_t graph;
    cuda_attend_t attend;
};

#pragma endregion Kernel Records

#pragma region Kernel Names

/** The NVIDIA kernels, the first template argument of @c cuda_kernel. */
enum class cuda_kernel_t : u32_t {
    insert_k,
    refine_k,
    search_k,
    search_indexer_k,
    multiply_k,
    attend_k,
    merge_links_k,
    scatter_links_k,
    derive_norms_k,
    mark_seeds_k,
    reachability_sweep_k,
    reachability_islands_k,
    reached_in_links_k,
    reconnect_island_k,
    merge_attention_k,
    count_k,
};

/** One instantiated kernel: the kernel and its register rung, zero where it keeps no ladder or
 *  walks the arena frontier. */
struct cuda_rung_t {
    cuda_kernel_t kernel;
    u32_t slots;
};

/** Every instantiation one dtype's code exports bar the attends, which the device pass instantiates
 *  and the host looks up as it loads the fatbin. */
inline constexpr cuda_rung_t cuda_rungs[] = {
    {cuda_kernel_t::insert_k, 2},
    {cuda_kernel_t::insert_k, 4},
    {cuda_kernel_t::insert_k, 8},
    {cuda_kernel_t::insert_k, 16},
    {cuda_kernel_t::refine_k, 2},
    {cuda_kernel_t::refine_k, 4},
    {cuda_kernel_t::refine_k, 8},
    {cuda_kernel_t::refine_k, 16},
    {cuda_kernel_t::search_k, 0},
    {cuda_kernel_t::search_k, 2},
    {cuda_kernel_t::search_k, 4},
    {cuda_kernel_t::search_k, 8},
    {cuda_kernel_t::search_k, 16},
    {cuda_kernel_t::search_indexer_k, 0},
    {cuda_kernel_t::search_indexer_k, 2},
    {cuda_kernel_t::search_indexer_k, 4},
    {cuda_kernel_t::search_indexer_k, 8},
    {cuda_kernel_t::search_indexer_k, 16},
    {cuda_kernel_t::multiply_k, 0},
    {cuda_kernel_t::multiply_k, 2},
    {cuda_kernel_t::multiply_k, 4},
    {cuda_kernel_t::multiply_k, 8},
    {cuda_kernel_t::multiply_k, 16},
    {cuda_kernel_t::merge_links_k, 0},
    {cuda_kernel_t::scatter_links_k, 0},
    {cuda_kernel_t::derive_norms_k, 0},
    {cuda_kernel_t::mark_seeds_k, 0},
    {cuda_kernel_t::reachability_sweep_k, 0},
    {cuda_kernel_t::reachability_islands_k, 0},
    {cuda_kernel_t::reached_in_links_k, 0},
    {cuda_kernel_t::reconnect_island_k, 0},
    {cuda_kernel_t::merge_attention_k, 0},
};

/** The rungs every walk family, attends included, is instantiated over. */
inline constexpr u32_t cuda_walk_rungs[] = {0, 2, 4, 8, 16};

/** Rungs per kernel: the arena frontier's zero, then 2, 4, 8 and 16 slots. */
inline constexpr std::size_t cuda_rungs_per_kernel_k = 5;

constexpr std::size_t cuda_rung_index(u32_t slots) noexcept {
    return slots == 0 ? 0 : slots == 2 ? 1 : slots == 4 ? 2 : slots == 8 ? 3 : 4;
}

/** The NVIDIA bits of a capability ceiling as one template integer, counted from @c nk_cap_cuda_k,
 *  the lowest of them. */
constexpr u32_t cuda_capability_bits(nk_capability_t capability) noexcept {
    return static_cast<u32_t>(capability / nk_cap_cuda_k);
}

inline constexpr std::size_t cuda_kernel_name_capacity_k = 160;

/** Spells the Itanium mangled name of @c cuda_kernel<kernel,capabilities,dtype,slots,lookup>:
 *  five unsigned template arguments, then the one @ref cuda_arguments_t parameter, as NVCC
 *  emits it; @p lookup is zero for every kernel but an attend. */
inline void cuda_kernel_name(char (&name)[cuda_kernel_name_capacity_k], cuda_rung_t rung, nk_capability_t capability,
                             nk_dtype_t dtype, u32_t lookup = 0) noexcept {
    std::snprintf(name, sizeof(name),
                  "_ZN4unum7usearch5dense11cuda_kernelILj%uELj%uELj%uELj%uELj%uEEEvNS1_16cuda_arguments_tE",
                  static_cast<unsigned>(rung.kernel), static_cast<unsigned>(cuda_capability_bits(capability)),
                  static_cast<unsigned>(dtype), static_cast<unsigned>(rung.slots), static_cast<unsigned>(lookup));
}

/** One capability ceiling's fatbin, carrying every compiled dtype's kernels, defined by the unit
 *  that embeds it. */
template <nk_capability_t capability_>
bytes_view_t cuda_fatbin() noexcept;

#pragma endregion Kernel Names

template <nk_dtype_t, nk_capability_t, typename, typename, typename>
struct vram_nvidia_index;

/** One stream's device-side scratch for driving a CUDA index: the traversal arena, the batch
 *  staging, the telemetry counters and the attend miss word, riding the stream its allocator names
 *  and never opening one. Two contexts on one index share only the storage and the published
 *  counter, never scratch. */
template <cuda_allocator allocator_type_ = cuda_device_allocator_t>
struct cuda_traversal_context {
    using allocator_t = allocator_type_;

    /** Matches the single-query verb lands at most, in inline storage; the batched verbs land any
     *  count into the caller's rows. */
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

    /** Binds this context to @p allocator's stream and device, a null stream naming the legacy
     *  default one, and allocates only the fixed-size counters; sized buffers wait for the first
     *  verb. */
    cuda_traversal_context(allocator_t allocator = {}) noexcept
        : allocator_ {allocator}, //
          telemetry_ {cuda_make_unique(telemetry_allocator_t {allocator_}, 1)},
          assigned_rows_ {cuda_make_unique(counter_allocator_t {allocator_}, 1)},
          missed_ {cuda_make_unique(counter_allocator_t {allocator_}, 1)},
          arena_ {nullptr, cuda_deleter<allocator_t> {allocator_, 0}},
          staging_ {nullptr, cuda_deleter<allocator_t> {allocator_, 0}} {
        if (telemetry_) cuda_fill_async(telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream());
    }

    cuda_traversal_context(cuda_traversal_context const&) = delete;
    cuda_traversal_context& operator=(cuda_traversal_context const&) = delete;

    /** Every owned resource frees in stream order, so a move is a member-wise steal and destruction
     *  needs no drain. */
    cuda_traversal_context(cuda_traversal_context&&) noexcept = default;
    ~cuda_traversal_context() noexcept = default;

    cuda_stream_handle_t stream() const noexcept { return allocator_.stream(); }
    int device() const noexcept { return allocator_.device(); }

    /** Whether this context can walk at all: every device-side word claimed. */
    explicit operator bool() const noexcept { return telemetry_ && assigned_rows_ && missed_; }

    /** Waits for every transfer and launch queued on this context's stream - one stream's
     *  completion, the executor's @c drain being the fleet's. Selects the context's device first,
     *  since a null stream is the current device's. */
    status_t drain() noexcept {
        return cuda_set_device(device()) == cuda_success_k && cuda_api().stream_synchronize(stream()) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Polls whether everything queued on this context's stream finished - @c success_k once it
     *  has, @c deferred_k while any of it runs - without blocking, so one host thread awaits
     *  whichever of several streams lands first. */
    status_t landed() noexcept {
        if (cuda_set_device(device()) != cuda_success_k) return status_t::device_error_k;
        cuda_result_t const queried = cuda_api().stream_query(stream());
        return queried == cuda_success_k           ? status_t::success_k
               : queried == cuda_error_not_ready_k ? status_t::deferred_k
                                                   : status_t::device_error_k;
    }

    /** The telemetry of the last submission, zeroed when it was queued; read once it landed. */
    vram_telemetry_block_t spent() const noexcept {
        vram_telemetry_block_t snapshot = {};
        cuda_copy(&snapshot, telemetry_.get(), sizeof(snapshot));
        return snapshot;
    }

    /** Whether the last attend met a node its value lookup had no row for, which refuses it; read
     *  once it landed. */
    bool missed() const noexcept {
        u32_t word = 0;
        cuda_copy(&word, missed_.get(), sizeof(word));
        return word != 0;
    }

  private:

    /** Only the index's verbs reach the raw scratch - callers observe a context through @c results,
     *  @ref spent and @ref missed, never through its buffers. */
    template <nk_dtype_t, nk_capability_t, typename, typename, typename>
    friend struct vram_nvidia_index;

    using telemetry_allocator_t = typename allocator_t::template rebind<vram_telemetry_block_t>::other;
    using counter_allocator_t = typename allocator_t::template rebind<u32_t>::other;
    using bytes_ptr_t = cuda_unique_ptr<byte_t, allocator_t>;

    allocator_t allocator_;

    /** Monotone epoch floor over the arena's visited slots, so the arena is zeroed only when
     *  freshly allocated or when the 32-bit epoch space wraps, never per launch. */
    u64_t epoch_floor_ = 0;

    /** The device-side counters of the last submission, zeroed when it was queued. */
    cuda_unique_ptr<vram_telemetry_block_t, telemetry_allocator_t> telemetry_;

    /** Rows of the current launch's batch warps have assigned themselves so far. */
    cuda_unique_ptr<u32_t, counter_allocator_t> assigned_rows_;

    /** Raised by the last attend's walks at a node their value lookup had no row for, zeroed when
     *  it was queued. */
    cuda_unique_ptr<u32_t, counter_allocator_t> missed_;
    bytes_ptr_t arena_;
    std::size_t arena_bytes_ = 0;
    bytes_ptr_t staging_;
    std::size_t staging_bytes_ = 0;

    /** Grow-only traversal arena for @p bytes: the exact visited sets and spill tails of
     *  every resident warp, kept across calls. @return The base, or null on exhaustion. */
    byte_t* ensure_arena(std::size_t bytes) noexcept { return ensure_(arena_, arena_bytes_, bytes); }

    /** Grow-only staging for one batch's device transport: an insert's padded vectors, or a
     *  search's queries and result landing. */
    byte_t* ensure_staging(std::size_t bytes) noexcept { return ensure_(staging_, staging_bytes_, bytes); }

    /** Grows @p buffer to @p bytes when @p capacity falls short; @return the base, or null. */
    byte_t* ensure_(bytes_ptr_t& buffer, std::size_t& capacity, std::size_t bytes) noexcept {
        if (bytes <= capacity) return buffer.get();
        // The deleter already carries this context's allocator, so only the run and its count
        // change.
        buffer.reset(allocator_.allocate(bytes));
        buffer.get_deleter().count = bytes;
        capacity = buffer ? bytes : 0;
        return buffer.get();
    }
};

/**
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *
 *  Speaks the RAM-resident dense index's verbs, with two divergences: a batch's IDs are one
 *  contiguous range, and its rows publish unordered on the device, turning visible to searches a
 *  whole batch at a time; see @ref dense_vram_nvidia_index_concurrency. The first successful
 *  @c reserve freezes the geometry and the virtual-address ceiling; physical slabs map on demand,
 *  and fresh neighbors are sentinel-filled before any ID covering them can be assigned. The storage
 *  grows only while @c claiming_ is held, and the claim counter, the watermark and the kept status
 *  are the only mutable host state; the published-row count is device memory no host thread
 *  dereferences.
 */
template <                                                                      //
    nk_dtype_t vector_dtype_ = nk_i8_k,                                         //
    nk_capability_t capability_ = nk_cap_cuda_k | nk_cap_ampere_k,              //
    typename vectors_allocator_type_ = cuda_virtual_memory_allocator<byte_t>,   //
    typename neighbors_allocator_type_ = cuda_virtual_memory_allocator<byte_t>, //
    typename traversals_allocator_type_ = cuda_arena_allocator_t>               //
struct vram_nvidia_index {
    static_assert(capability_ % nk_cap_cuda_k == 0, "The kernels' names carry the NVIDIA capability bits alone");

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    /** The host-visible half of the quads kit the chain picks under @c capability_. */
    using kit_traits_t = cuda_kit_traits<vector_dtype_, capability_>;
    using vectors_allocator_t = vectors_allocator_type_;
    using neighbors_allocator_t = neighbors_allocator_type_;
    using traversals_allocator_t = traversals_allocator_type_;
    using scalar_t = typename kit_traits_t::scalar_t;
    using dot_t = typename kit_traits_t::dot_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    // The device writes the kit's plain accumulator into these bytes, so the two must share a
    // representation.
    static_assert(sizeof(dot_result_t) == sizeof(dot_t));
    using fragment_t = typename kit_traits_t::fragment_t;
    using norms_t = typename kit_traits_t::norms_t;
    using vectors_view_t = nk::matrix_view<scalar_t>;
    using vector_view_t = nk::vector_view<scalar_t>;
    using node_id_t = dense::node_id_t;
    using node_ids_span_t = nk::vector_span<node_id_t>;
    using distance_t = dense::distance_t;
    using small_size_t = u32_t;

    /** The page-locked host bytes a checkpoint moves through, one batch at a time. */
    using bounce_t = growing_buffer<byte_t, cuda_pinned_allocator_t>;

    /** Rows one checkpoint batch carries between a matrix and the caller: one slab, the row
     *  granularity the RAM residency grows by. */
    inline static constexpr std::size_t bounce_rows_log2_k = 20;
    using traversal_context_t = cuda_traversal_context<traversals_allocator_t>;

    /** The strongest @c synchronization_t contract this design honors; nothing routes on it.
     *  Batches claim disjoint ID ranges and the publication protocol makes overlapping searches
     *  safe. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;

    /** The capability of the quads kit the chain picked under @c capability_, surfaced in the bench
     *  reports. */
    inline static constexpr nk_capability_t picked_quads_capability_k = kit_traits_t::picked_k;

    /** The scored node's 32-bit ID space, less the sentinel. */
    inline static constexpr std::size_t max_node_count_k = (std::size_t {1} << 32) - 1u;

    /** Default visited-set slot count; the scratch slice must also fit the candidate cache. */
    inline static constexpr u32_t default_visited_capacity_k = 2048;

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
    explicit vram_nvidia_index(nk_kernel_kind_t kernel, vectors_allocator_t const& vectors_allocator = {},
                               neighbors_allocator_t const& neighbors_allocator = {}) noexcept
        : device_ {vectors_allocator.device()}, kernel_ {kernel}, vectors_matrix_(vectors_allocator),
          norms_column_(vectors_allocator), neighbors_matrix_(neighbors_allocator) {}
    vram_nvidia_index(vram_nvidia_index const&) = delete;
    vram_nvidia_index& operator=(vram_nvidia_index const&) = delete;

#pragma region Measures

    /** Rows visible to searches: the watermark every walk samples below and filters neighbors
     *  by. */
    std::size_t size() const noexcept {
        return standard_atomic_ref<std::size_t const>(last_visible_id_).load(std::memory_order_acquire);
    }

    /** The first failure that left claimed rows unpublished, kept for the index's lifetime; it
     *  refuses every later mutation while searches keep running. */
    status_t status() const noexcept {
        return standard_atomic_ref<status_t const>(status_).load(std::memory_order_acquire);
    }

    /** Rows mapped across the flat storage, the smallest of their counts, since slabs are a fixed
     *  byte size and each matrix maps a different row count per slab; dot's empty norms column
     *  bounds nothing. */
    std::size_t capacity() const noexcept {
        std::size_t const neighbors_row_bytes = connectivity() * sizeof(u32_t);
        std::size_t const vector_row_bytes = row_bytes();
        if (!neighbors_row_bytes || !vector_row_bytes) return 0;
        std::size_t const by_vectors = vectors_matrix_.committed_bytes() / vector_row_bytes;
        std::size_t const by_neighbors = neighbors_matrix_.committed_bytes() / neighbors_row_bytes;
        if (kernel_ == nk_kernel_dot_k) return min_of(by_vectors, by_neighbors);
        std::size_t const by_norms = norms_column_.committed_bytes() / sizeof(norms_t);
        return min_of(by_vectors, min_of(by_norms, by_neighbors));
    }
    std::size_t connectivity() const noexcept {
        return geometry_connectivity(
            standard_atomic_ref<packed_geometry_id_t const>(geometry_).load(std::memory_order_acquire));
    }
    std::size_t dimensions() const noexcept {
        return geometry_dimensions(
            standard_atomic_ref<packed_geometry_id_t const>(geometry_).load(std::memory_order_acquire));
    }

    /** The walk breadth @p config asks for, or twice the connectivity when it asks for none. */
    template <typename config_type_>
    std::size_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor : connectivity() * 2;
    }

    /** The pinned device's identity and limits, queried once at the first @c reserve. */
    cuda_device_specs_t const& device_specs() const noexcept { return device_; }

    /** The metric every walk scores by, fixed at construction. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Whether rows keep norms: every metric but dot finalizes through them, so dot leaves the
     *  norms column empty and checkpoints carry none. */
    bool stores_norms() const noexcept { return kernel_ != nk_kernel_dot_k; }

    /** Device-side flat storage views - the audit surface for invariant checkers & tests. */
    cuda_flat_storage_t storage() const noexcept {
        u32_t const fragments = row_fragments();
        return {reinterpret_cast<byte_t*>(vectors_matrix_.base()),
                reinterpret_cast<byte_t*>(norms_column_.base()),
                reinterpret_cast<u32_t*>(neighbors_matrix_.base()),
                static_cast<u32_t>(connectivity()),
                fragments,
                lanes_per_row(),
                steps_per_row(),
                static_cast<u32_t>(((u64_t {1} << 31) + fragments - 1) / fragments),
                kernel_};
    }

    /** Bytes of one packed logical row, bit-exact for sub-byte scalars. */
    std::size_t packed_row_bytes() const noexcept {
        return divide_round_up<std::size_t>(dimensions() * scalar_t::bits_per_dimension(), 8u);
    }

    /** Fragments per vector row, derived from the frozen geometry like every row measure. */
    u32_t row_fragments() const noexcept {
        return divide_round_up<u32_t>(static_cast<u32_t>(packed_row_bytes()), sizeof(fragment_t));
    }

    /** The vector row's byte length, rounded up to whole fragments; the tail bytes stay zero so
     *  every dot-derived metric is unaffected. */
    std::size_t row_bytes() const noexcept { return static_cast<std::size_t>(row_fragments()) * sizeof(fragment_t); }

    /** Lanes cooperating on one row: the smallest power of two covering its fragments, at most a
     *  warp. */
    u32_t lanes_per_row() const noexcept { return min_of(std::bit_ceil(row_fragments()), 32u); }

    /** Steps those lanes take across one row. */
    u32_t steps_per_row() const noexcept { return divide_round_up(row_fragments(), lanes_per_row()); }

    /** Bytes of one warp's dynamic-shared slice on the insertion path: the query span while
     *  traversing, then the candidate cache & re-selection spans. */
    u32_t insert_scratch_bytes_per_warp() const noexcept {
        u32_t const selection_slots = min_of(static_cast<u32_t>(connectivity()), //
                                             static_cast<u32_t>(vram_cache_rows_k));
        u32_t const selection_bytes = selection_slots * static_cast<u32_t>(cuda_scored_node_bytes_k + //
                                                                           2 * sizeof(u32_t));
        u32_t const cache_bytes = static_cast<u32_t>(row_bytes()) * vram_cache_rows_k;
        return max_of(static_cast<u32_t>(row_bytes()), cache_bytes + selection_bytes);
    }

    /** Bytes of one warp's dynamic-shared slice on the refinement merge path: the candidate
     *  cache - whose last row hosts the node's own vector - then the scored span and the
     *  kept-row indices. */
    u32_t merge_scratch_bytes_per_warp() const noexcept {
        u32_t const cache_bytes = static_cast<u32_t>(row_bytes()) * vram_cache_rows_k;
        return cache_bytes + vram_cache_rows_k * static_cast<u32_t>(cuda_scored_node_bytes_k + sizeof(u32_t));
    }

    /** Whether a search, multiply or attend launch seats on this device: per warp, the query span
     *  and the @p fold_bytes an attend keeps past it, beside the static batch buffers. */
    bool walk_scratch_fits(std::size_t fold_bytes = 0) const noexcept {
        return std::size_t {vram_warps_per_block_k} * (row_bytes() + fold_bytes + kit_traits_t::gathered_bytes_k) <=
               device_.shared_memory_per_block_bytes;
    }

    /** Whether an insert or refine launch seats on this device: its per-warp candidate cache beside
     *  the static norms, batch buffers and schedule state - wide rows can serve walks yet refuse
     *  building. */
    bool build_scratch_fits() const noexcept {
        std::size_t const statics = std::size_t {vram_warps_per_block_k} *
                                    (vram_cache_rows_k * sizeof(norms_t) + kit_traits_t::gathered_bytes_k +
                                     kit_traits_t::schedule_bytes_k);
        return statics + std::size_t {vram_warps_per_block_k} *
                             max_of(insert_scratch_bytes_per_warp(), merge_scratch_bytes_per_warp()) <=
               device_.shared_memory_per_block_bytes;
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

    /** Bytes this index occupies - the committed device storage, the published-row count, and the
     *  host struct; traversal scratch belongs to executors and contexts. */
    std::size_t memory_usage() const noexcept {
        return sizeof(vram_nvidia_index) + vectors_matrix_.committed_bytes() + norms_column_.committed_bytes() +
               neighbors_matrix_.committed_bytes() + (published_rows_ ? sizeof(u32_t) : 0);
    }

    /** Copies the stored row of @p node into @p destination with the fragment padding stripped,
     *  through one blocking copy on no stream, so it needs no context; the bulk export is
     *  @c save. */
    status_t vector(node_id_t node, nk::vector_span<scalar_t> destination) const noexcept {
        if (node >= size()) return status_t::config_mismatch_k;
        if (destination.size() != dimensions()) return status_t::config_mismatch_k;
        if (!destination.is_contiguous()) return status_t::unsupported_layout_k;
        std::size_t const scalar_row_bytes = packed_row_bytes();
        return cuda_copy(destination.byte_data(), vectors_matrix_.base() + static_cast<std::size_t>(node) * row_bytes(),
                         scalar_row_bytes) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Reports whether every claimed ID is published - a derived health check, never cached state.
     *  Call it only with no batch in flight. */
    status_t validate() const noexcept {
        u32_t published = 0;
        if (cuda_copy(&published, published_rows_.get(), sizeof(u32_t)) != cuda_success_k)
            return status_t::device_error_k;
        return published == claimed_() ? status_t::success_k : status_t::device_error_k;
    }

#pragma endregion Measures

#pragma region Reservation

    /** Which frontier a warp's arena slice serves: the register ladder needs only the exact visited
     *  set, the wide arena frontier adds its spill tail. */
    enum class frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

    /** One warp's arena slice for a walk expanding @p expansion candidates: the exact visited set,
     *  sized well past the expected visit count so probing stays short, plus the spill tail when
     *  @p frontier_kind outgrows registers. */
    static cuda_arena_shape_t arena_shape_for(u32_t expansion, frontier_kind_t frontier_kind) noexcept {
        u32_t const visited_slots = std::bit_ceil(64 * expansion);
        return {visited_slots, 32 - static_cast<u32_t>(std::bit_width(visited_slots - 1)),
                frontier_kind == frontier_kind_t::wide_arena_k ? 32 * expansion : 0};
    }

    /** Requested geometry & capacity for a @c reserve() call; frozen by the first success. */
    struct reserve_capacity_t {
        std::size_t capacity = 0;
        connectivity_t connectivity = connectivity_32_k;
        std::size_t dimensions = 0;
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
        if (!standard_atomic_ref<packed_geometry_id_t>(geometry_).compare_exchange_strong(
                observed, requested, std::memory_order_acq_rel, std::memory_order_acquire) &&
            observed != requested)
            return {capacity(), status_t::config_mismatch_k};
        connectivity_t const connectivity = config.connectivity;
        if (config.capacity > max_node_count_k) return {capacity(), status_t::overflow_k};
        if (!vectors_matrix_.reserved_bytes()) { // First reservation: bind the device, size the ceiling.
            // The primary context must be current before any mapping, since the storage lives
            // in it.
            if (cuda_set_device(device_.ordinal) != cuda_success_k) return {0, status_t::device_error_k};
            auto const [specs, status] = cuda_device_specs_t::query(device_.ordinal);
            if (failed(status)) return {0, status};
            device_ = specs;
            // Geometry no walk can seat is refused here; building checks its own wider scratch per
            // call.
            if (!walk_scratch_fits()) return {0, status_t::config_mismatch_k};
            if (status_t const loaded = load_kernels_(); failed(loaded)) return {0, loaded};
            // Virtual address space is free, so the ceiling is generous and independent of the
            // first requested capacity; the reservation itself is the ceiling, no cached row count
            // shadows it.
            bool const normed = kernel_ != nk_kernel_dot_k;
            std::size_t const bytes_per_row = row_bytes() + (normed ? sizeof(norms_t) : 0) +
                                              connectivity * sizeof(u32_t);
            // Device storage budgets against VRAM; managed storage oversubscribes it and budgets
            // against host memory through their own backend, which keeps the topology symbols out
            // of device instantiations.
            std::size_t budget_bytes = 4 * device_.vram_bytes;
            if constexpr (vectors_allocator_t::address_space_k == cuda_address_space_t::managed_k)
                budget_bytes = vectors_allocator_t::budget_bytes();
            std::size_t const ceiling_rows = min_of(max_node_count_k,
                                                    max_of<std::size_t>(config.capacity, budget_bytes / bytes_per_row));
            if (vectors_matrix_.reserve_range(ceiling_rows * row_bytes(), device_.ordinal) != status_t::success_k ||
                (normed &&
                 norms_column_.reserve_range(ceiling_rows * sizeof(norms_t), device_.ordinal) != status_t::success_k) ||
                neighbors_matrix_.reserve_range(ceiling_rows * connectivity * sizeof(u32_t), device_.ordinal) !=
                    status_t::success_k)
                return {0, status_t::bad_alloc_k};
            counter_allocator_t counter_allocator {nullptr, device_.ordinal};
            published_rows_ = cuda_make_unique(counter_allocator, 1);
            if (!published_rows_ ||
                cuda_api().fill(cuda_address(published_rows_.get()), 0, sizeof(u32_t)) != cuda_success_k)
                return {0, status_t::device_error_k};
        }
        status_t const grown = grow_rows_(config.capacity, nullptr);
        return {capacity(), grown};
    }

#pragma endregion Reservation

#pragma region Insertion

    /** Optional tuning knobs; zeros mean "derive from connectivity". */
    struct insert_config_t {

        /** Breadth of the construction candidate pool; rounded up to a multiple of 32, clamped to
         *  the widest register pool. Larger pools trade build speed for graph quality. */
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
        u32_t const pool_slots = pool_slots_for_(config.expansion_factor, 4);
        cuda_arena_shape_t const arena_shape = arena_shape_for(pool_slots * 32, frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.multiprocessor_count) *
                                        max_of(insert_blocks_per_multiprocessor_k, search_blocks_per_multiprocessor_k) *
                                        vram_warps_per_block_k;
        std::size_t const results_bytes = // The single-query result landing.
            traversal_context_t::results_capacity_k * (sizeof(node_id_t) + sizeof(distance_t));
        return aligned(sizeof(vram_telemetry_block_t)) + 2 * aligned(sizeof(u32_t)) +
               aligned(arena_warps * arena_shape.slice_bytes()) + aligned(rows * row_bytes() + results_bytes);
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

        /** Neighbors the back-linker abandoned after the bounded lock budget - GPU-only
         *  telemetry. */
        std::size_t unlinked_neighbors = 0;

        /** Frontier tail entries shed past their arena slice - GPU-only telemetry. */
        std::size_t tail_drops = 0;

        /** Visited-set claims that ran out of probes and starved a walk - GPU-only telemetry. */
        std::size_t visited_saturations = 0;

        /** Refine-side distance work, folded into @c computed_distances here, so always zero. */
        std::size_t computed_distances_in_refines = 0;

        /** Reverse-refine distance work, folded the same way, so always zero. */
        std::size_t computed_distances_in_reverse_refines = 0;
    };

    /** Inserts @p new_vectors under the contiguous ID range @b [first_id,first_id+rows) after every
     *  row claimed before, every transfer and the launch riding the context's stream, then drains
     *  that stream, turns the watermark over the range once every earlier batch has, and reports
     *  telemetry. */
    expected<insert_result_t> insert(vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        // Zeroed per call rather than per launch, so the launches of a sharded plan on one context
        // add up.
        if (context && context.device() == device_.ordinal &&
            cuda_fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), context.stream()) !=
                cuda_success_k)
            return {{}, status_t::device_error_k};
        expected<insert_result_t> inserted = submit_insert_(new_vectors, assigned_ids, context, config);
        if (!inserted || !new_vectors.extent(0)) return inserted;
        if (status_t const drained = context.drain(); failed(drained))
            return {{inserted.value.new_size, inserted.value.first_id}, fail_(drained)};
        if (status_t const published = publish_(inserted.value.first_id, new_vectors.extent(0)); failed(published))
            return {{inserted.value.new_size, inserted.value.first_id}, published};
        vram_telemetry_block_t const spent = context.spent();
        inserted.value.visited_members = spent.hops;
        inserted.value.computed_distances = spent.scored;
        inserted.value.restarts = spent.restarts;
        inserted.value.unlinked_neighbors = spent.lock_skips;
        inserted.value.tail_drops = spent.tail_drops;
        inserted.value.visited_saturations = spent.visited_saturations;
        return inserted;
    }

#pragma endregion Insertion

#pragma region Search

    /** Optional tuning knobs of @c search(); zeros mean "derive from connectivity". */
    struct search_config_t {

        /** Breadth of the candidate pool; defaults to twice the connectivity, and is raised
         *  automatically when @c count needs a wider pool. */
        small_size_t expansion_factor = 0;

        /** Matches reported per query; the walk expands at least twice as many, and a walk past the
         *  widest register pool ranks every candidate its arena frontier kept. */
        small_size_t count = 10;

        /** The caller's mutation contract; every level shares one code path, since the walk is
         *  insert-safe by protocol, so the field only documents intent. */
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

    /** The bytes an executor's run must hold before a context carved over it can search or multiply
     *  @p rows rows under @p config, landing @c config.count columns per row, one aligned term per
     *  carve the verb makes. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config) const noexcept {
        static_assert(sizeof(dot_t) == sizeof(distance_t));
        if (!dimensions() || !device_.multiprocessor_count) return 0;
        auto aligned = [](std::size_t bytes) noexcept {
            return divide_round_up<std::size_t>(bytes, default_alignment_k) * default_alignment_k;
        };
        // Search widens the pool to cover its count while multiply walks the bare factor, so the
        // wider one sizes.
        u32_t const expansion = max_of<u32_t>(
            config.expansion_factor > config.count ? config.expansion_factor : config.count * 2,
            static_cast<u32_t>(non_zero_expansion_factor(config)));
        bool const wide = expansion > vram_widest_pool_slots_k * 32;
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        u32_t const pool_slots = pool_slots_for_(expansion, count_slots > 2 ? count_slots : 2);
        cuda_arena_shape_t const arena_shape = arena_shape_for(
            wide ? expansion : pool_slots * 32,
            wide ? frontier_kind_t::wide_arena_k : frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.multiprocessor_count) *
                                        search_blocks_per_multiprocessor_k * vram_warps_per_block_k;
        std::size_t const staging_bytes = rows * row_bytes() +
                                          rows * config.count * (sizeof(node_id_t) + sizeof(distance_t)) +
                                          rows * sizeof(u32_t);
        return aligned(sizeof(vram_telemetry_block_t)) + 2 * aligned(sizeof(u32_t)) +
               aligned(arena_warps * arena_shape.slice_bytes()) + aligned(staging_bytes);
    }

    /** The bytes a search under the indexer @p heads needs: the plain walk's, plus the staging of
     *  the heads and their weights. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config,
                                       indexer_heads<scalar_t> const& heads) const noexcept {
        // The heads share the staging carve, and aligning their bytes apart can only overestimate
        // it.
        std::size_t const heads_bytes = heads.heads.extent(0) * (row_bytes() + sizeof(f32_t));
        return traversal_memory_usage(rows, config) +
               divide_round_up<std::size_t>(heads_bytes, default_alignment_k) * default_alignment_k;
    }

    /** The bytes an attend over @p rows rows needs, one walk per @p head_count of them: the walks'
     *  own, plus every row staged beside its output and carry, and beside those of every further
     *  one of @p shards, which the first shard gathers. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config, std::size_t head_count,
                                       std::size_t shards = 1) const noexcept {
        std::size_t const rows_bytes = rows * (row_bytes() +
                                               shards * (dimensions() * sizeof(f32_t) + sizeof(softmax_state_t)));
        return traversal_memory_usage(head_count ? rows / head_count : rows, config) +
               divide_round_up<std::size_t>(rows_bytes, default_alignment_k) * default_alignment_k;
    }

    /**
     *  @brief Finds the nearest published neighbors of every query row, ascending by distance with
     *      sentinel IDs marking absent matches. Safe to overlap with insertions on other streams.
     *
     *  @param[in] queries The query rows, in the index's stored scalar type.
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
    expected<search_result_t> search(vectors_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                     search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept {
        return settle_(submit_search_(queries, result_ids, result_distances, context, config, size()),
                       queries.extent(0), context, completion);
    }

    /** The batched @c search ranked by a lightning indexer: a node's distance is 1 − Σₕ
     *  wₕ · ReLU(qₕ · k) over the @p heads of its query, which must match the queries in
     *  count and the index in dimensions, while the walk navigates by @p queries. A dot
     *  verb: any other metric refuses with @c config_mismatch_k. */
    expected<search_result_t> search(vectors_view_t queries, indexer_heads<scalar_t> const& heads,
                                     nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                     search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept
        requires floating_scalar<scalar_t>
    {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return settle_(submit_search_(queries, result_ids, result_distances, context, config, size(), heads),
                       queries.extent(0), context, completion);
    }

    /** One query in, the nearest matches ascending in @c context.results - the correctness path; a
     *  one-query grid never fills the machine, the batched overload does. */
    expected<search_result_t> search(vector_view_t query_vector, traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        return search_(query_vector, context, config, size());
    }

    /** Filtered search is banned on the device index: a host predicate cannot run inside the walk,
     *  and a device-resident admission bitmap is a design of its own. */
    template <typename predicate_type_>
    expected<search_result_t> search_if(vector_view_t, predicate_type_&&, traversal_context_t&,
                                        search_config_t = {}) noexcept = delete;

    /**
     *  @brief Graph products of every left-hand row: each row keeps the pairs its walk scored, node
     *      IDs beside native dots, until its output row fills.
     *
     *  @param[in] left The left-hand rows, in the index's stored scalar type.
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
    expected<search_result_t> multiply(vectors_view_t left, nk::matrix_span<node_id_t> node_ids,
                                       nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                       traversal_context_t& context, search_config_t config = {},
                                       completion_t completion = completion_t::blocking_k) noexcept {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        expected<search_result_t> landed = settle_(
            submit_multiply_(left, node_ids, products, counts, context, config, size()), left.extent(0), context,
            completion);
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
     *  @param[in] queries The query heads, in the index's stored scalar type; row t ·
     *      @p head_count + h is head h of token t.
     *  @param[in] head_count The query heads sharing this key-value head; it must divide the rows,
     *      and their per-warp softmax states and accumulator rows must seat in shared memory beside
     *      the query span.
     *  @param[in] value_lookup Device-callable: `value_lookup(node_id)` points at the node's value
     *      row on a 16-byte boundary, as wide as the keys and stored like them, in memory this
     *      device reads; a null row refuses the call with @c shape_mismatch_k. Trivially copyable,
     *      it names itself with a @c device_lookup_k the fatbin of @c capability_ instantiates its
     *      attend kernels over, through @ref cuda_value_lookup_of.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys and packed, in
     *      host or device memory.
     *  @param[out] carry Each row's softmax state, one per row, in host or device memory.
     *  @param[inout] context The stream every transfer and the launch ride.
     *  @param[in] config The per-call breadth; @c count plays no part.
     *  @param[in] completion As in @c multiply; a deferred call leaves a missing row to
     *      @c context.missed().
     *  @return The query rows attended, and the traversal telemetry this call spent;
     *      @c config_mismatch_k unless the index scores by dot.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(vectors_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, nk::matrix_span<f32_t> outputs,
                                     std::span<softmax_state_t> carry, traversal_context_t& context,
                                     search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept {
        std::size_t const rows_count = queries.extent(0);
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (outputs.extent(0) != rows_count || outputs.extent(1) != dimensions() || carry.size() != rows_count)
            return {{}, status_t::config_mismatch_k};
        if (!outputs.is_contiguous()) return {{}, status_t::unsupported_layout_k};
        expected<f32_t*> const landed = submit_attend_(queries, head_count, value_lookup, score_scale, context, config,
                                                       1, size());
        if (!landed || !rows_count) return {{}, landed.status};
        cuda_stream_handle_t const stream = context.stream();
        if (cuda_copy_async(outputs.data(), landed.value, outputs.numel() * sizeof(f32_t), stream) != cuda_success_k ||
            cuda_copy_async(carry.data(), landed.value + outputs.numel(), carry.size_bytes(), stream) != cuda_success_k)
            return {{}, status_t::device_error_k};
        expected<search_result_t> const attended = settle_({{rows_count}, status_t::success_k}, rows_count, context,
                                                           completion);
        if (attended && completion == completion_t::blocking_k && context.missed())
            return {{}, status_t::shape_mismatch_k};
        return attended;
    }

#pragma endregion Search

#pragma region Refinement

    /** Optional tuning knobs of @c refine(); zeros mean "derive from connectivity". */
    struct refine_config_t {

        /** Breadth of the re-selection pool, rounded up to a multiple of 32 and clamped to the
         *  widest register pool, 512 ranks; zero takes twice the connectivity, like the CPU's. */
        small_size_t expansion_factor = 0;
    };

    using refine_result_t = dense::refine_result_t;
    using health_result_t = dense::health_result_t;

    /**
     *  @brief Re-selects the neighbors of every node in @p nodes from a fresh walk seeded by the
     *      ones it holds, then heals whatever no search can reach - the same pass the CPU's
     *      @c refine_plan_t runs. An empty range heals.
     *
     *  The walk lands shadow neighbors, the scatter buckets every pick under its target, and the
     *  merge re-judges forward and reverse candidates together, backfilling what diversity leaves
     *  empty. The heal then sweeps reachability and, round by round while the islands shrink,
     *  re-selects them and reconnects whatever stays unreached into a reached pick or the landmark
     *  covering it.
     *
     *  Demands an exclusive caller: the merge writes live neighbors with plain stores. A stopped
     *  sequence of passes leaves the graph correct and merely less improved.
     *
     *  @param[in] nodes The ID range to walk; IDs at or past @c size() are skipped.
     *  @param[inout] context The driving context; its arena grows to the pass's shape.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    expected<refine_result_t> refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                     refine_config_t config = {}) noexcept {
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

        auto [audit, reserve_status] = reserve_audit_(context, rows);
        if (failed(reserve_status)) return {result, reserve_status};
        u32_t islands = 0;
        auto survey = [&]() noexcept {
            auto const [unreached, audit_status] = audit_(context, audit);
            islands = unreached;
            result.sweeps += audit.sweeps;
            return audit_status;
        };
        if (status_t const audited = survey(); failed(audited)) return {result, audited};
        result.islands_found = islands;
        for (u32_t islands_before = islands; islands;) {
            // The island IDs replace the range: offsets below their count index them.
            auto const [healed, healed_status] = refine_pass_(context, config, {0, islands},
                                                              {audit.island_ids, islands});
            if (failed(healed_status)) return {result, healed_status};
            add_walks(healed);
            if (status_t const audited = survey(); failed(audited)) return {result, audited};
            if (!islands) break;
            // The reached in-links first, so no reconnection takes the last link a reached node is
            // found through.
            if (cuda_fill_async(audit.in_links, 0, rows * sizeof(u32_t), context.stream()) != cuda_success_k)
                return {result, status_t::device_error_k};
            if (status_t const launched = launch_reconnect_islands_(audit, islands, rows, context.stream());
                failed(launched))
                return {result, launched};
            if (status_t const audited = survey(); failed(audited)) return {result, audited};
            // A round that frees no island frees none on the next either, since the picks it needs
            // are unreached.
            if (islands >= islands_before) break;
            islands_before = islands;
        }
        auto const [reconnected, read_status] = read_counter_(audit.reconnected, context.stream());
        if (failed(read_status)) return {result, read_status};
        result.islands_reconnected = reconnected;
        result.reachable_nodes = rows - islands;
        return {result, status_t::success_k};
    }

    /** Measures forward reachability from as many stride-spread landmarks as one search draws -
     *  the connectivity; any shortfall against @c size() is the population no search can discover.
     *  Read-only on the graph, so concurrent insertions may leave fresh nodes uncounted. */
    expected<health_result_t> health(traversal_context_t& context) noexcept {
        std::size_t const rows = size();
        if (!rows) return {{}, status_t::success_k};
        auto [audit, reserve_status] = reserve_audit_(context, 0);
        if (failed(reserve_status)) return {{}, reserve_status};
        auto const [unreached, audit_status] = audit_(context, audit);
        if (failed(audit_status)) return {{}, audit_status};
        return {{rows, rows - unreached, audit.sweeps}, status_t::success_k};
    }

#pragma endregion Refinement

#pragma region Checkpoints

    /**
     *  @brief One save in flight: the watermark it took, the stream its copies ride, and the
     *      batches it hands out - one pinned bounce of rows at a time, since a device matrix cannot
     *      be read in place.
     *
     *  Vectors travel as logical rows with the fragment padding stripped, neighbors verbatim under
     *  the positional sentinel convention, and norms verbatim when the columns carry theirs.
     *
     *  One host thread drives a device, so batches are taken one after another, though the tallies
     *  keep the shape a threaded caller expects.
     */
    struct [[nodiscard]] save_plan_t {

        /** Where the rows land, vetted against the watermark at planning. */
        columns_t columns {};

        /** What an ID under the watermark moves by - the row this shard starts at. */
        node_id_t id_shift = 0;

        /** The stream every device-to-host copy queues on, drained before its bytes move. */
        cuda_stream_handle_t stream = nullptr;

        /** Page-locked host bytes: one batch of packed vector rows, then its neighbor IDs, then its
         *  norms. */
        bounce_t bounce {cuda_pinned_allocator_t {}};

        /** Rows under the watermark at planning. */
        node_id_t visible = 0;

        /** Rows one batch carries - a bounce's worth, or every row when fewer. */
        std::size_t batch_rows = 0;

        /** Rows whose vectors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_vectors = 0;

        /** Rows whose neighbors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_neighbors = 0;

        save_plan_t() noexcept = default;

        /** The index snapshotted into @p columns on @p stream under the watermark at @p visible,
         *  its IDs shifted by @p id_shift; only @ref plan_save calls it, and sizes the bounce
         *  afterwards. */
        save_plan_t(vram_nvidia_index const& index, columns_t columns, node_id_t id_shift, cuda_stream_handle_t stream,
                    node_id_t visible) noexcept
            : columns(columns), id_shift(id_shift), stream(stream),
              bounce(cuda_pinned_allocator_t {stream, index.device_.ordinal}), visible(visible),
              batch_rows(min_of(static_cast<std::size_t>(visible), power_of_two<std::size_t>(bounce_rows_log2_k))),
              index_(&index) {}
        save_plan_t(save_plan_t&& other) noexcept
            : columns(other.columns), id_shift(other.id_shift), stream(other.stream), bounce(std::move(other.bounce)),
              visible(other.visible), batch_rows(other.batch_rows), covered_vectors(other.covered_vectors),
              covered_neighbors(other.covered_neighbors), index_(std::exchange(other.index_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        // The tallies are this plan's own: a copy would tally into a twin `finish` never reads.
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;

        /** A save owns nothing but its bounce, so a dying plan leaves the index as it found it. */
        ~save_plan_t() noexcept = default;

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
            std::size_t const first = position * batch_rows;
            std::size_t const bound = min_of(first + batch_rows, static_cast<std::size_t>(visible));
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = step.index_rows;
            return step;
        }

        /** The steps the save spans - one bounce-sized batch each. */
        indexed_range<save_plan_t> steps() const noexcept {
            return {this, index_ && batch_rows ? divide_round_up<std::size_t>(visible, batch_rows) : 0};
        }

        /**
         *  @brief Copies step @p step 's rows off the device into the plan's own columns.
         *
         *  The vectors go with the fragment padding stripped, then the neighbors with every ID
         *  under the watermark shifted by @c id_shift and every other folded to the sentinel, then
         *  the norms where the columns take them.
         *
         *  @param[in] step The step to copy, as @ref steps hands them out.
         *  @param[inout] scratch Unused here, since the bounce is already host memory; it rides
         *      in the signature so a caller drives a device plan exactly as it drives a host one.
         */
        status_t save(step_t const& step, nk::matrix_span<node_id_t>) noexcept {
            if (!index_) return status_t::uninitialized_k;
            iota_view<node_id_t> const rows = step.index_rows;
            if (rows.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const connectivity = index_->connectivity();
            std::size_t const neighbors_row_bytes = connectivity * sizeof(node_id_t);
            std::size_t const step_slots = rows.size() * connectivity;
            std::size_t const first = step.source_rows.first;
            bytes_span_t const vectors_rows = flat_rows(columns.vectors, first, rows.size(), scalar_row_bytes);
            bytes_span_t const neighbors_rows = flat_rows(columns.neighbors, first, rows.size(), neighbors_row_bytes);
            bytes_span_t const norms_rows = flat_rows(columns.norms, first, rows.size(), sizeof(norms_t));
            if (vectors_rows.empty() || neighbors_rows.empty() || norms_rows.empty() != columns.norms.empty())
                return status_t::config_mismatch_k;
            cuda_api_t const& api = cuda_api();
            if (cuda_copy_2d_async(
                    bounce.data(), scalar_row_bytes,
                    index_->vectors_matrix_.base() + static_cast<std::size_t>(rows.first) * index_->row_bytes(),
                    index_->row_bytes(), scalar_row_bytes, rows.size(), stream) != cuda_success_k)
                return status_t::device_error_k;
            if (api.stream_synchronize(stream) != cuda_success_k) return status_t::device_error_k;
            [[maybe_unused]] status_t const copied_vectors = copy_bytes(vectors_rows,
                                                                        bytes_of(bounce.data(), vectors_rows.size()));
            // The norms count with the vectors, so a refused copy of either leaves the step
            // uncovered.
            if (!norms_rows.empty()) {
                bytes_span_t const bounce_norms = to_writable_span(bounce).subspan(
                    batch_rows * (scalar_row_bytes + neighbors_row_bytes), norms_rows.size());
                if (cuda_copy_async(
                        bounce_norms.data(),
                        index_->norms_column_.base() + static_cast<std::size_t>(rows.first) * sizeof(norms_t),
                        bounce_norms.size(), stream) != cuda_success_k)
                    return status_t::device_error_k;
                if (api.stream_synchronize(stream) != cuda_success_k) return status_t::device_error_k;
                [[maybe_unused]] status_t const copied_norms = copy_bytes(norms_rows, bounce_norms);
            }
            standard_atomic_ref<std::size_t> {covered_vectors}.fetch_add(rows.size(), std::memory_order_acq_rel);

            bytes_span_t const bounce_neighbors = to_writable_span(bounce).subspan(batch_rows * scalar_row_bytes);
            if (cuda_copy_async(
                    bounce_neighbors.data(),
                    index_->neighbors_matrix_.base() + static_cast<std::size_t>(rows.first) * neighbors_row_bytes,
                    rows.size() * neighbors_row_bytes, stream) != cuda_success_k)
                return status_t::device_error_k;
            if (api.stream_synchronize(stream) != cuda_success_k) return status_t::device_error_k;
            std::span<node_id_t> const landed {reinterpret_cast<node_id_t*>(bounce_neighbors.data()), step_slots};
            for (node_id_t& neighbor : landed)
                neighbor = neighbor < visible ? static_cast<node_id_t>(neighbor + id_shift) : node_id_unknown_k;
            [[maybe_unused]] status_t const copied_neighbors = copy_bytes(neighbors_rows, std::as_bytes(landed));
            standard_atomic_ref<std::size_t> {covered_neighbors}.fetch_add(rows.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Pronounces the snapshot complete: every batch's vectors and neighbors landed. A success
         *  spends the plan; a refusal leaves it whole, so the missing steps can still land. */
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

        /** The index being snapshotted; null in the empty plan a refused @ref plan_save returns,
         *  and once @c finish succeeded. */
        vram_nvidia_index const* index_ = nullptr;
    };

    /**
     *  @brief Opens a save into @p columns under @p synchronization on @p stream: takes the
     *      watermark and sizes one bounce for it.
     *
     *  @c concurrent_k answers @c config_mismatch_k, since the device storage cannot be probed for
     *  settlement; @c exclusive_k reads like @c immutable_k. Columns too short for the rows under
     *  it are refused here, so no step ever writes past them.
     *
     *  @param[out] columns Where the rows land, sized from @ref saved_shape.
     *  @param[in] synchronization The contract the caller certifies.
     *  @param[in] stream The stream every copy queues on.
     *  @param[in] id_shift What an ID under the watermark moves by - the row this shard starts at.
     */
    expected<save_plan_t> plan_save(columns_t columns, synchronization_t synchronization,
                                    cuda_stream_handle_t stream = nullptr, node_id_t id_shift = 0) const noexcept {
        if (synchronization == synchronization_t::concurrent_k) return {{}, status_t::config_mismatch_k};
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        std::size_t const rows = size();
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return {{}, status_t::config_mismatch_k};
        save_plan_t plan {*this, columns, id_shift, stream, static_cast<node_id_t>(rows)};
        std::size_t const batch_bytes = plan.batch_rows *
                                        (packed_row_bytes() + connectivity() * sizeof(node_id_t) + sizeof(norms_t));
        if (batch_bytes && failed(plan.bounce.resize_uninitialized(batch_bytes))) return {{}, status_t::bad_alloc_k};
        return {std::move(plan), status_t::success_k};
    }

    /** Serializes every row under the watermark into @p columns on the calling thread - the
     *  one-shot over @ref plan_save for a caller with no streams to fan the steps out over. */
    status_t save(columns_t columns, synchronization_t synchronization, nk::matrix_span<node_id_t> scratch,
                  cuda_stream_handle_t stream = nullptr) const noexcept {
        auto [plan, status] = plan_save(columns, synchronization, stream);
        if (failed(status)) return status;
        for (typename save_plan_t::step_t const step : plan.steps())
            if (status_t const saved = plan.save(step, scratch); failed(saved)) return saved;
        return plan.finish();
    }

    /**
     *  @brief One load in flight: the rows it appends past every claimed one, the storage grown to
     *      hold them, and the batches they span, each filled through one pinned bounce and
     *      uploaded on the stream; @ref finish derives the norms no batch carried and publishes.
     *
     *  Holds the claim for its whole life, so nothing else claims rows while it runs. A source's
     *  IDs are local to the rows it delivers, so every batch shifts them onto the appended range
     *  and folds every other ID onto the sentinel. Repeated plans append, which is how several
     *  parts land in one shard. A plan that dies before @c finish claims nothing, drops the mutex,
     *  and leaves the storage grown for the next one.
     */
    struct [[nodiscard]] load_plan_t {

        /** Where the rows come from, vetted against them at planning. */
        columns_view_t columns {};

        /** The ID space the source's rows speak. */
        iota_view<node_id_t> source_range {};

        /** The stream every host-to-device copy and the norms kernel queue on. */
        cuda_stream_handle_t stream = nullptr;

        /** Page-locked host bytes: one batch of packed vector rows, then its neighbor IDs, then its
         *  norms. */
        bounce_t bounce {cuda_pinned_allocator_t {}};

        /** The first appended ID - the row the claim landed at. */
        node_id_t base = 0;

        /** Rows the plan appends. */
        std::size_t rows = 0;

        /** Rows one batch carries - a bounce's worth, or every row when fewer. */
        std::size_t batch_rows = 0;

        /** Rows whose vectors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_vectors = 0;

        /** Rows whose neighbors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_neighbors = 0;

        /** Rows whose norms landed from the columns, tallied through @c std::atomic_ref; @c finish
         *  derives every row's when it falls short. */
        std::size_t covered_norms = 0;

        load_plan_t() noexcept = default;

        /** The index appended to at @p base on @p stream, and the claim held meanwhile - unowned
         *  for a plan of no rows; only @ref plan_load calls it, and sizes the bounce afterwards. */
        load_plan_t(vram_nvidia_index& index, columns_view_t columns, iota_view<node_id_t> source_range,
                    cuda_stream_handle_t stream, std::unique_lock<fu::spin_mutex_t>&& claiming, node_id_t base,
                    std::size_t rows) noexcept
            : columns(columns), source_range(source_range), stream(stream),
              bounce(cuda_pinned_allocator_t {stream, index.device_.ordinal}), base(base), rows(rows),
              batch_rows(min_of(rows, power_of_two<std::size_t>(bounce_rows_log2_k))), index_(&index),
              claiming_(std::move(claiming)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : columns(other.columns), source_range(other.source_range), stream(other.stream),
              bounce(std::move(other.bounce)), base(other.base), rows(std::exchange(other.rows, 0)),
              batch_rows(other.batch_rows), covered_vectors(other.covered_vectors),
              covered_neighbors(other.covered_neighbors), covered_norms(other.covered_norms),
              index_(std::exchange(other.index_, nullptr)), claiming_(std::move(other.claiming_)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        // The tallies are this plan's own: a copy would tally into a twin `finish` never reads.
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** The lock's own death drops the claim, unless @c finish already did. */
        ~load_plan_t() noexcept = default;

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
            std::size_t const first = static_cast<std::size_t>(base) + position * batch_rows;
            std::size_t const bound = min_of(first + batch_rows, static_cast<std::size_t>(base) + rows);
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = {static_cast<node_id_t>(first - base), static_cast<node_id_t>(bound - base)};
            return step;
        }

        /** The steps the append spans - one bounce-sized batch each. */
        indexed_range<load_plan_t> steps() const noexcept {
            return {this, index_ && batch_rows ? divide_round_up(rows, batch_rows) : 0};
        }

        /**
         *  @brief Copies step @p step 's rows out of the plan's own columns through the bounce.
         *
         *  The vectors go into the fragment-padded matrix rows, whose fresh tails arrive zeroed,
         *  and their norms where the columns carry them, then the neighbors with every ID inside
         *  @c source_range folded onto the appended range and every other onto the sentinel.
         *
         *  @param[in] step The step to fill, as @ref steps hands them out.
         */
        status_t load(step_t const& step) noexcept { return load(step, columns, source_range); }

        /** @brief The same copy out of @p columns under @p source_range rather than the plan's
         *      own - what a sharded plan drives, one part's slice at a time, its step naming the
         *      rows in that part's row and ID spaces.
         *
         *  @param[in] step The step to fill; its source rows address @p columns.
         *  @param[in] columns The part's own columns.
         *  @param[in] source_range The ID space the part's rows speak.
         */
        status_t load(step_t const& step, columns_view_t columns, iota_view<node_id_t> source_range) noexcept {
            if (!index_) return status_t::uninitialized_k;
            iota_view<node_id_t> const range = step.index_rows;
            if (range.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const connectivity = index_->connectivity();
            std::size_t const neighbors_row_bytes = connectivity * sizeof(node_id_t);
            std::size_t const step_slots = range.size() * connectivity;
            std::size_t const first = step.source_rows.first;
            bytes_view_t const vectors_rows = flat_rows(columns.vectors, first, range.size(), scalar_row_bytes);
            bytes_view_t const neighbors_rows = flat_rows(columns.neighbors, first, range.size(), neighbors_row_bytes);
            bytes_view_t const norms_rows = flat_rows(columns.norms, first, range.size(), sizeof(norms_t));
            if (vectors_rows.empty() || neighbors_rows.empty() || norms_rows.empty() != columns.norms.empty())
                return status_t::config_mismatch_k;
            // The part's rows land at this step's own index rows, so an ID's shift follows from
            // the step rather than from the plan's base - the two agree for a one-part plan.
            node_id_t const shift = static_cast<node_id_t>(range.first - step.source_rows.first + source_range.first);
            // The device indexes its buckets by neighbor ID unchecked, so every rebased ID must
            // land in this plan.
            if (static_cast<std::size_t>(shift) + source_range.size() > static_cast<std::size_t>(base) + rows)
                return status_t::config_mismatch_k;
            [[maybe_unused]] status_t const staged_vectors = copy_bytes(bytes_of(bounce.data(), vectors_rows.size()),
                                                                        vectors_rows);
            cuda_api_t const& api = cuda_api();
            if (cuda_copy_2d_async(
                    index_->vectors_matrix_.base() + static_cast<std::size_t>(range.first) * index_->row_bytes(),
                    index_->row_bytes(), bounce.data(), scalar_row_bytes, scalar_row_bytes, range.size(),
                    stream) != cuda_success_k)
                return index_->fail_(status_t::device_error_k);
            // Dot reserves no norms column, so a part's column has nowhere to land.
            if (index_->stores_norms() && !norms_rows.empty()) {
                bytes_span_t const bounce_norms = to_writable_span(bounce).subspan(
                    batch_rows * (scalar_row_bytes + neighbors_row_bytes), norms_rows.size());
                [[maybe_unused]] status_t const staged_norms = copy_bytes(bounce_norms, norms_rows);
                if (cuda_copy_async(
                        index_->norms_column_.base() + static_cast<std::size_t>(range.first) * sizeof(norms_t),
                        bounce_norms.data(), bounce_norms.size(), stream) != cuda_success_k)
                    return index_->fail_(status_t::device_error_k);
                // Tallied ahead of the drain: a failed one keeps the vectors short, which `finish`
                // refuses.
                standard_atomic_ref<std::size_t> {covered_norms}.fetch_add(range.size(), std::memory_order_acq_rel);
            }
            if (api.stream_synchronize(stream) != cuda_success_k) return index_->fail_(status_t::device_error_k);
            standard_atomic_ref<std::size_t> {covered_vectors}.fetch_add(range.size(), std::memory_order_acq_rel);

            bytes_span_t const bounce_neighbors = to_writable_span(bounce).subspan(batch_rows * scalar_row_bytes);
            [[maybe_unused]] status_t const staged_neighbors = copy_bytes(bounce_neighbors.first(neighbors_rows.size()),
                                                                          neighbors_rows);
            std::span<node_id_t> const landed {reinterpret_cast<node_id_t*>(bounce_neighbors.data()), step_slots};
            rebase_neighbors(landed, source_range, shift, node_id_unknown_k);
            compact_neighbors(landed, connectivity, range.first, node_id_unknown_k);
            if (cuda_copy_async(
                    index_->neighbors_matrix_.base() + static_cast<std::size_t>(range.first) * neighbors_row_bytes,
                    bounce_neighbors.data(), range.size() * neighbors_row_bytes, stream) != cuda_success_k)
                return index_->fail_(status_t::device_error_k);
            if (api.stream_synchronize(stream) != cuda_success_k) return index_->fail_(status_t::device_error_k);
            standard_atomic_ref<std::size_t> {covered_neighbors}.fetch_add(range.size(), std::memory_order_acq_rel);
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
            if (standard_atomic_ref<std::size_t> {covered_vectors}.load(std::memory_order_acquire) != rows ||
                standard_atomic_ref<std::size_t> {covered_neighbors}.load(std::memory_order_acquire) != rows)
                return status_t::config_mismatch_k;
            if (rows) {
                // One launch derives every row, so a load mixing carried and missing norms
                // re-derives the carried.
                if (index_->stores_norms() &&
                    standard_atomic_ref<std::size_t> {covered_norms}.load(std::memory_order_acquire) != rows &&
                    failed(index_->launch_derive_norms_(base, rows, stream)))
                    return index_->fail_(status_t::device_error_k);
                u32_t const restored = static_cast<u32_t>(base + rows);
                if (cuda_copy_async(index_->published_rows_.get(), &restored, sizeof(restored), stream) !=
                        cuda_success_k ||
                    cuda_api().stream_synchronize(stream) != cuda_success_k)
                    return index_->fail_(status_t::device_error_k);
                standard_atomic_ref<std::size_t>(index_->claimed_rows_).store(base + rows, std::memory_order_release);
                claiming_.unlock();
            }
            vram_nvidia_index* const index = std::exchange(index_, nullptr);
            return rows ? index->publish_(base, rows) : status_t::success_k;
        }

      private:

        /** The index appended to; null in the empty plan a refused @ref plan_load returns, and once
         *  @c finish published. */
        vram_nvidia_index* index_ = nullptr;

        /** The claim held from planning to publication, dropped by its own death otherwise. */
        std::unique_lock<fu::spin_mutex_t> claiming_;
    };

    /**
     *  @brief Opens a load of @p rows from @p columns past every claimed one on @p stream: takes
     *      the claim, grows the storage to hold them - the fresh neighbor rows arrive
     *      sentinel-filled - and sizes one bounce.
     *
     *  Refuses an unreserved index, a kept failure, columns shorter than the rows, a claim already
     *  held, and rows past the ID space.
     *
     *  @param[in] columns Where the rows come from, every column but an empty norms one sized for
     *      @p rows.
     *  @param[in] rows Rows to append.
     *  @param[in] source_range The ID space the source's rows speak.
     *  @param[in] stream The stream every copy and the norms kernel queue on.
     */
    expected<load_plan_t> plan_load(columns_view_t columns, std::size_t rows, iota_view<node_id_t> source_range,
                                    cuda_stream_handle_t stream = nullptr) noexcept {
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return {{}, status_t::config_mismatch_k};
        auto [plan, status] = plan_load(rows, stream);
        if (failed(status)) return {{}, status};
        plan.columns = columns;
        plan.source_range = source_range;
        return {std::move(plan), status};
    }

    /** Opens a load of @p rows the caller sources per step - what a sharded plan drives, its parts
     *  arriving one slice at a time rather than as one pair of columns. */
    expected<load_plan_t> plan_load(std::size_t rows, cuda_stream_handle_t stream = nullptr) noexcept {
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (!rows)
            return {load_plan_t {*this, {}, {}, stream, {}, static_cast<node_id_t>(claimed_()), 0},
                    status_t::success_k};
        if (!claiming_.lock_if_free()) return {{}, status_t::contended_k};
        std::unique_lock<fu::spin_mutex_t> claiming {claiming_, std::adopt_lock};
        // Read under the claim: a batch claiming between an earlier read and the claim would
        // overlap the loaded rows, and `finish` would then lower the claim beneath it.
        node_id_t const base = static_cast<node_id_t>(claimed_());
        if (rows > max_node_count_k - base) return {{}, status_t::overflow_k};
        load_plan_t plan {*this, {}, {}, stream, std::move(claiming), base, rows};
        std::size_t const batch_bytes = plan.batch_rows *
                                        (packed_row_bytes() + connectivity() * sizeof(node_id_t) + sizeof(norms_t));
        if (failed(plan.bounce.resize_uninitialized(batch_bytes))) return {{}, status_t::bad_alloc_k};
        if (status_t const grown = grow_rows_(base + rows, stream); failed(grown)) return {{}, grown};
        return {std::move(plan), status_t::success_k};
    }

    /** Appends the whole source on the calling thread - the one-shot over @ref plan_load for a
     *  caller with no streams to fan the steps out over. */
    status_t load(columns_view_t columns, std::size_t rows, iota_view<node_id_t> source_range,
                  cuda_stream_handle_t stream = nullptr) noexcept {
        auto [plan, status] = plan_load(columns, rows, source_range, stream);
        if (failed(status)) return status;
        for (typename load_plan_t::step_t const step : plan.steps())
            if (status_t const loaded = plan.load(step); failed(loaded)) return loaded;
        return plan.finish();
    }

#pragma endregion Checkpoints

  private:

    /** The sharded index walks its shards under one snapshot of their watermarks, merges their
     *  attends on its first device, and publishes or fails their batches together. */
    template <typename, nk_dtype_t, nk_capability_t, typename>
    friend struct vram_nvidia_indexes;

#pragma region Insertion

    /** The submission half of @c insert: claims, stages, launches, and returns with the IDs, which
     *  are dense and known at claim time, the rows invisible until the caller drains and publishes
     *  them. Back-to-back submissions on one context are serialized by its stream and add into its
     *  telemetry, which the caller zeroes; across devices they overlap freely. */
    expected<insert_result_t> submit_insert_(vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                                             traversal_context_t& context, insert_config_t config) noexcept {
        std::size_t const rows = new_vectors.extent(0);
        if (!rows) return {{claimed_(), static_cast<node_id_t>(claimed_())}, status_t::success_k};
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (new_vectors.extent(1) != dimensions()) return {{}, status_t::config_mismatch_k};
        if (assigned_ids.size() && assigned_ids.size() != rows) return {{}, status_t::config_mismatch_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (!build_scratch_fits()) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (new_vectors.stride_bytes(0) < static_cast<std::ptrdiff_t>(scalar_row_bytes))
            return {{}, status_t::unsupported_layout_k};
        cuda_stream_handle_t const stream = context.stream();
        // Every fallible preparation precedes the claim: a claimed range that never publishes would
        // starve every later batch's admission ramp.
        byte_t* const staging = context.ensure_staging(rows * row_bytes());
        if (!staging) return {{}, status_t::bad_alloc_k};
        if (scalar_row_bytes != row_bytes() &&
            cuda_fill_async(staging, 0, rows * row_bytes(), stream) != cuda_success_k)
            return {{}, status_t::device_error_k};
        // The source pitch is the caller's stride, so a strided view uploads without being gathered
        // first.
        if (cuda_copy_2d_async(staging, row_bytes(), new_vectors.byte_data(), new_vectors.stride_bytes(0),
                               scalar_row_bytes, rows, stream) != cuda_success_k)
            return {{}, status_t::device_error_k};

        // Every inserting warp owns an arena slice for its exact visited set.
        u32_t const pool_slots = pool_slots_for_(config.expansion_factor, 4);
        cuda_arena_shape_t const arena_shape = arena_shape_for(pool_slots * 32, frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.multiprocessor_count) *
                                        insert_blocks_per_multiprocessor_k * vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {{}, status_t::bad_alloc_k};

        // Grow, claim and launch while holding the claiming mutex: the claim is stored only once
        // the launch is accepted, so a rejected grid, which publishes nothing, is un-claimed
        // exactly.
        std::unique_lock claiming {claiming_};
        std::size_t const first_id = standard_atomic_ref<std::size_t>(claimed_rows_).load(std::memory_order_relaxed);
        std::size_t const total_rows = first_id + rows;
        status_t claimed = grow_rows_(total_rows, stream);
        if (claimed == status_t::success_k) {
            // Only a fresh allocation or a wrapped 32-bit epoch space warrants zeroing, never a
            // warm arena.
            if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + rows > 0xFFFFFFFEull) {
                if (cuda_fill_async(arena, 0, context.arena_bytes_, stream) != cuda_success_k) {
                    claimed = status_t::device_error_k;
                    context.arena_bytes_ = 0; // Spent: a retry must re-zero, never walk epoch garbage.
                }
                context.epoch_floor_ = 0;
            }
            u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
            context.epoch_floor_ += rows; // Monotone; an un-claim below only forfeits epoch space.
            cuda_build_t const build {storage(),
                                      staging,
                                      static_cast<u32_t>(total_rows),
                                      insert_scratch_bytes_per_warp(),
                                      static_cast<node_id_t>(first_id),
                                      arena,
                                      arena_shape,
                                      epoch_base,
                                      published_rows_.get(),
                                      context.telemetry_.get(),
                                      context.assigned_rows_.get()};
            // The launch is the last fallible step; reading its verdict here keeps the claim
            // reversible.
            if (claimed == status_t::success_k && failed(launch_insert_(build, pool_slots, context)))
                claimed = status_t::device_error_k;
            if (claimed == status_t::success_k)
                standard_atomic_ref<std::size_t>(claimed_rows_).store(total_rows, std::memory_order_release);
        }
        claiming.unlock();
        if (claimed != status_t::success_k) return {{}, claimed};

        insert_result_t result;
        result.first_id = static_cast<node_id_t>(first_id);
        for (std::size_t row = 0; row != assigned_ids.size(); ++row)
            assigned_ids[row] = static_cast<node_id_t>(first_id + row);
        result.new_size = total_rows;
        return {result, status_t::success_k};
    }

#pragma endregion Insertion

#pragma region Search

    /** What every walking verb stages ahead of its sink: the queries, the walk's breadth and
     *  arena, the frontier the walks keep, the staging past the uploads, and any heads the walks
     *  rescore by - which the kernel side turns into its search view and rescoring. */
    struct walk_launch_t {

        /** The staged query rows, @b [queries,row_fragments], device-resident. */
        fragment_t const* queries = nullptr;
        u32_t queries_count = 0;

        /** The watermark snapshotted at staging: landmarks are drawn below it and no neighbor at or
         *  past it is followed. */
        u32_t visible = 0;

        /** Bytes of the per-warp dynamic-shared slice: the query span, and an attend's fold past
         *  it. */
        u32_t scratch_bytes_per_warp = 0;

        /** Requested expansion; past the register ladder it is the wide walk's budget. */
        u32_t expansion = 0;

        /** The traversal arena every warp carves its slice from, and the slice's shape. */
        byte_t* arena = nullptr;
        cuda_arena_shape_t arena_shape {};

        /** The context's epoch floor at launch. */
        u32_t epoch_base = 0;

        /** The register ladder's width, when the ladder serves. */
        u32_t pool_slots = 0;

        /** Which frontier the walks keep. */
        frontier_kind_t frontier = frontier_kind_t::register_ladder_k;

        /** The staging bytes past the queries and any heads, where the sink writes its outputs. */
        byte_t* outputs = nullptr;

        /** The heads staged beside the queries, their weights under an indexer, and the heads per
         *  query - null and zero for a plain walk. */
        fragment_t const* heads = nullptr;
        f32_t const* weights = nullptr;
        u32_t heads_count = 0;
    };

    /** The grouped-query heads an attend stages beside the rows its walks navigate by: every head
     *  of every token, @c head_count consecutive rows per token. */
    struct attend_heads_t {
        vectors_view_t rows;
        std::size_t head_count;
    };

    /** Validates and uploads @p queries, then any indexer or attend @p heads and the indexer's
     *  weights, sizes the arena for @p expansion, and reserves @p outputs_bytes of staging past
     *  them: everything a walking verb stages before its sink, the walks bound below the @p visible
     *  rows. Until the context drains, the queries, the heads and the context stay live. */
    template <typename heads_type_>
    expected<walk_launch_t> stage_walk_(vectors_view_t queries, [[maybe_unused]] heads_type_ const& heads,
                                        traversal_context_t& context, u32_t expansion, u32_t minimum_slots,
                                        std::size_t outputs_bytes, std::size_t visible) noexcept {
        constexpr bool indexer_k = std::is_same_v<heads_type_, indexer_heads<scalar_t>>;
        constexpr bool attend_k = std::is_same_v<heads_type_, attend_heads_t>;
        std::size_t const queries_count = queries.extent(0);
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (queries.extent(1) != dimensions()) return {{}, status_t::config_mismatch_k};
        // Validated with the rest: past here a refusal would already have burned epoch space.
        if (queries.stride_bytes(0) < static_cast<std::ptrdiff_t>(packed_row_bytes()))
            return {{}, status_t::unsupported_layout_k};
        std::size_t heads_rows = 0;
        if constexpr (indexer_k) {
            heads_rows = heads.heads.extent(0);
            if (heads.weights.extent(0) != queries_count || heads_rows != queries_count * heads.weights.extent(1) ||
                heads.heads.extent(1) != dimensions())
                return {{}, status_t::config_mismatch_k};
            if (heads.heads.stride_bytes(0) < static_cast<std::ptrdiff_t>(packed_row_bytes()))
                return {{}, status_t::unsupported_layout_k};
        }
        if constexpr (attend_k) {
            heads_rows = heads.rows.extent(0);
            if (heads.rows.stride_bytes(0) < static_cast<std::ptrdiff_t>(packed_row_bytes()))
                return {{}, status_t::unsupported_layout_k};
        }
        cuda_stream_handle_t const stream = context.stream();

        // Past the register ladder's widest pool the arena-backed walk serves.
        frontier_kind_t const frontier = expansion > vram_widest_pool_slots_k * 32 ? frontier_kind_t::wide_arena_k
                                                                                   : frontier_kind_t::register_ladder_k;
        u32_t const pool_slots = pool_slots_for_(expansion, minimum_slots);
        cuda_arena_shape_t const arena_shape = arena_shape_for(
            frontier == frontier_kind_t::wide_arena_k ? expansion : pool_slots * 32, frontier);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.multiprocessor_count) *
                                        search_blocks_per_multiprocessor_k * vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {{}, status_t::bad_alloc_k};
        // The staging holds the query rows and any head rows first, since they carry the widest
        // alignment, then the heads' weights, then the outputs; stream order serializes its reuse
        // across calls.
        std::size_t const rows_bytes = (queries_count + heads_rows) * row_bytes();
        std::size_t const weights_bytes = indexer_k ? heads_rows * sizeof(f32_t) : 0;
        byte_t* const staging = context.ensure_staging(rows_bytes + weights_bytes + outputs_bytes);
        if (!staging) return {{}, status_t::bad_alloc_k};
        // Only a fresh allocation or a wrapped 32-bit epoch space warrants zeroing, never a warm
        // arena.
        if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + queries_count > 0xFFFFFFFEull) {
            if (cuda_fill_async(arena, 0, context.arena_bytes_, stream) != cuda_success_k) {
                context.arena_bytes_ = 0; // Spent: a retry must re-zero, never walk epoch garbage.
                return {{}, status_t::device_error_k};
            }
            context.epoch_floor_ = 0;
        }
        u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += queries_count;
        // Pitched uploads: device rows pad to whole fragments, and the pad bytes must be zero for
        // the zero-tail distance identity to hold.
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (scalar_row_bytes != row_bytes() && cuda_fill_async(staging, 0, rows_bytes, stream) != cuda_success_k)
            return {{}, status_t::device_error_k};
        auto upload_rows = [&](byte_t* destination, vectors_view_t rows) noexcept {
            return cuda_copy_2d_async(destination, row_bytes(), rows.byte_data(), rows.stride_bytes(0),
                                      scalar_row_bytes, rows.extent(0), stream) == cuda_success_k;
        };
        if (!upload_rows(staging, queries)) return {{}, status_t::device_error_k};
        if (cuda_fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream) != cuda_success_k)
            return {{}, status_t::device_error_k};

        walk_launch_t walk;
        walk.queries = reinterpret_cast<fragment_t const*>(staging);
        walk.queries_count = static_cast<u32_t>(queries_count);
        walk.visible = static_cast<u32_t>(visible);
        walk.scratch_bytes_per_warp = static_cast<u32_t>(row_bytes());
        walk.expansion = expansion;
        walk.arena = arena;
        walk.arena_shape = arena_shape;
        walk.epoch_base = epoch_base;
        walk.pool_slots = pool_slots;
        walk.frontier = frontier;
        walk.outputs = staging + rows_bytes + weights_bytes;
        if constexpr (indexer_k) {
            byte_t* const staged_heads = staging + queries_count * row_bytes();
            byte_t* const staged_weights = staging + rows_bytes;
            std::size_t const weights_row_bytes = heads.weights.extent(1) * sizeof(f32_t);
            if (!upload_rows(staged_heads, heads.heads) ||
                cuda_copy_2d_async(staged_weights, weights_row_bytes, heads.weights.byte_data(),
                                   heads.weights.stride_bytes(0), weights_row_bytes, queries_count,
                                   stream) != cuda_success_k)
                return {{}, status_t::device_error_k};
            walk.heads = reinterpret_cast<fragment_t const*>(staged_heads);
            walk.weights = reinterpret_cast<f32_t const*>(staged_weights);
            walk.heads_count = static_cast<u32_t>(heads.weights.extent(1));
        }
        if constexpr (attend_k) {
            byte_t* const staged_heads = staging + queries_count * row_bytes();
            if (!upload_rows(staged_heads, heads.rows)) return {{}, status_t::device_error_k};
            walk.heads = reinterpret_cast<fragment_t const*>(staged_heads);
            walk.heads_count = static_cast<u32_t>(heads.head_count);
        }
        return {walk, status_t::success_k};
    }

    /** Blocking drains the context's stream and adds the telemetry the walks spent; deferred
     *  returns the submission as queued. */
    expected<search_result_t> settle_(expected<search_result_t> submitted, std::size_t rows,
                                      traversal_context_t& context, completion_t completion) noexcept {
        if (completion == completion_t::deferred_k || !submitted || !rows) return submitted;
        if (status_t const drained = context.drain(); failed(drained)) return {{}, drained};
        vram_telemetry_block_t const spent = context.spent();
        submitted.value.visited_members = spent.hops;
        submitted.value.computed_distances = spent.scored;
        submitted.value.restarts = spent.restarts;
        submitted.value.tail_drops = spent.tail_drops;
        submitted.value.visited_saturations = spent.visited_saturations;
        return submitted;
    }

    /** The single-query @c search over the @p visible rows, which a sharded caller snapshots across
     *  its shards. */
    expected<search_result_t> search_(vector_view_t query_vector, traversal_context_t& context, search_config_t config,
                                      std::size_t visible) noexcept {
        context.results.count_ = 0;
        if (!visible) return {{}, status_t::success_k}; // An empty index matches nothing, touches nothing.
        // Up to the expansion factor, defaulting to twice the connectivity, clipped to the landing
        // and the graph.
        u32_t const expansion = static_cast<u32_t>(non_zero_expansion_factor(config));
        u32_t const count = min_of<u32_t>(min_of<u32_t>(traversal_context_t::results_capacity_k, expansion),
                                          static_cast<u32_t>(visible));
        node_id_t staged_ids[traversal_context_t::results_capacity_k];
        distance_t staged_distances[traversal_context_t::results_capacity_k];
        vectors_view_t const one_query {query_vector.data(), {1, query_vector.size()}};
        config.count = count;
        expected<search_result_t> result = settle_(
            submit_search_(one_query, nk::matrix_span<node_id_t> {staged_ids, {1, count}},
                           nk::matrix_span<distance_t> {staged_distances, {1, count}}, context, config, visible),
            1, context, completion_t::blocking_k);
        if (!result) return result;
        std::size_t live = 0;
        for (; live != count && staged_ids[live] != node_id_unknown_k; ++live)
            context.results.entries_[live] = scored_node_t {staged_ids[live], staged_distances[live]};
        context.results.count_ = live;
        result.value.count = live;
        return result;
    }

    /** The submission half of @c search: stages the queries and @p heads, launches walks over the
     *  @p visible rows, and queues the pitched writeback. Until the context drains, the queries,
     *  the heads, both outputs and the context stay live; across contexts they must not share a
     *  buffer at all. */
    template <typename heads_type_ = dummy_heads_t>
    expected<search_result_t> submit_search_(vectors_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                             nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                             search_config_t config, std::size_t visible,
                                             heads_type_ const& heads = {}) noexcept {
        std::size_t const queries_count = queries.extent(0);
        if (!queries_count) return {{}, status_t::success_k};
        bool const wants_distances = result_distances.extent(0) != 0;
        if (result_ids.extent(0) != queries_count || result_ids.extent(1) < config.count ||
            (wants_distances &&
             (result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)))
            return {{}, status_t::config_mismatch_k};
        if (result_ids.stride_bytes(0) < static_cast<std::ptrdiff_t>(config.count * sizeof(node_id_t)) ||
            (wants_distances &&
             result_distances.stride_bytes(0) < static_cast<std::ptrdiff_t>(config.count * sizeof(distance_t))))
            return {{}, status_t::unsupported_layout_k};

        // The expansion covers the count, so the ladder's pool holds every rank, and past it the
        // arena frontier ranks everything it saw.
        u32_t const expansion = config.expansion_factor > config.count ? config.expansion_factor : config.count * 2;
        // The pool must cover the requested count, whatever the expansion asked for.
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        std::size_t const results_count = queries_count * config.count;
        std::size_t const ids_bytes = results_count * sizeof(node_id_t);
        std::size_t const distances_bytes = wants_distances ? results_count * sizeof(distance_t) : 0;
        auto const staged = stage_walk_(queries, heads, context, expansion, count_slots > 2 ? count_slots : 2,
                                        ids_bytes + distances_bytes, visible);
        if (!staged) return {{}, staged.status};
        node_id_t* const device_ids = reinterpret_cast<node_id_t*>(staged.value.outputs);
        distance_t* const device_distances = wants_distances
                                                 ? reinterpret_cast<distance_t*>(staged.value.outputs + ids_bytes)
                                                 : nullptr;
        cuda_stream_handle_t const stream = context.stream();
        if (status_t const launched = launch_search_(staged.value, device_ids, device_distances, config.count, context);
            failed(launched))
            return {{}, launched};
        // The landing may be wider than the count, so a multi-index can hand every shard its own
        // columns of one results matrix.
        if (cuda_copy_2d_async(result_ids.data(), result_ids.stride_bytes(0), device_ids,
                               config.count * sizeof(node_id_t), config.count * sizeof(node_id_t), queries_count,
                               stream) != cuda_success_k)
            return {{}, status_t::device_error_k};
        if (wants_distances &&
            cuda_copy_2d_async(result_distances.data(), result_distances.stride_bytes(0), device_distances,
                               config.count * sizeof(distance_t), config.count * sizeof(distance_t), queries_count,
                               stream) != cuda_success_k)
            return {{}, status_t::device_error_k};
        return {{config.count}, status_t::success_k};
    }

    /** The submission half of @c multiply: stages the rows, launches walks over the @p visible
     *  rows, and queues the writeback of the kept pairs and the scored counts. */
    expected<search_result_t> submit_multiply_(vectors_view_t left, nk::matrix_span<node_id_t> node_ids,
                                               nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                               traversal_context_t& context, search_config_t config,
                                               std::size_t visible) noexcept {
        std::size_t const rows_count = left.extent(0);
        std::size_t const wanted = node_ids.extent(1);
        if (node_ids.extent(0) != rows_count || products.extent(0) != rows_count || products.extent(1) != wanted ||
            counts.size() != rows_count)
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};
        if (node_ids.stride_bytes(0) < static_cast<std::ptrdiff_t>(wanted * sizeof(node_id_t)) ||
            products.stride_bytes(0) < static_cast<std::ptrdiff_t>(wanted * sizeof(dot_result_t)))
            return {{}, status_t::unsupported_layout_k};
        std::size_t const ids_bytes = rows_count * wanted * sizeof(node_id_t);
        std::size_t const products_bytes = rows_count * wanted * sizeof(dot_t);
        auto const staged = stage_walk_(left, dummy_heads_t {}, context,
                                        static_cast<u32_t>(non_zero_expansion_factor(config)), 2,
                                        ids_bytes + products_bytes + rows_count * sizeof(u32_t), visible);
        if (!staged) return {{}, staged.status};
        node_id_t* const device_ids = reinterpret_cast<node_id_t*>(staged.value.outputs);
        dot_t* const device_products = reinterpret_cast<dot_t*>(staged.value.outputs + ids_bytes);
        u32_t* const device_counts = reinterpret_cast<u32_t*>(staged.value.outputs + ids_bytes + products_bytes);
        cuda_stream_handle_t const stream = context.stream();
        if (status_t const launched = launch_multiply_(staged.value, device_ids, device_products, device_counts,
                                                       static_cast<u32_t>(wanted), context);
            failed(launched))
            return {{}, launched};
        if (wanted &&
            (cuda_copy_2d_async(node_ids.data(), node_ids.stride_bytes(0), device_ids, wanted * sizeof(node_id_t),
                                wanted * sizeof(node_id_t), rows_count, stream) != cuda_success_k ||
             cuda_copy_2d_async(products.data(), products.stride_bytes(0), device_products, wanted * sizeof(dot_t),
                                wanted * sizeof(dot_t), rows_count, stream) != cuda_success_k))
            return {{}, status_t::device_error_k};
        if (cuda_copy_async(counts.data(), device_counts, rows_count * sizeof(u32_t), stream) != cuda_success_k)
            return {{}, status_t::device_error_k};
        return {{}, status_t::success_k};
    }

    /** The submission half of @c attend: stages each token's first head to navigate by, then every
     *  head, and launches one walk per token over the @p visible rows, which lands the normalized
     *  rows, then the carries, in the first of @p blocks such blocks of staging, returned
     *  device-resident; the sharded index gathers its other shards' into the rest. */
    template <typename value_lookup_type_>
    expected<f32_t*> submit_attend_(vectors_view_t queries, std::size_t head_count,
                                    value_lookup_type_ const& value_lookup, f32_t score_scale,
                                    traversal_context_t& context, search_config_t config, std::size_t blocks,
                                    std::size_t visible) noexcept {
        std::size_t const rows_count = queries.extent(0);
        if (!head_count || rows_count % head_count) return {nullptr, status_t::shape_mismatch_k};
        if (!rows_count) return {nullptr, status_t::success_k};
        std::size_t const scratch_bytes = cuda_attend_scratch_bytes(head_count, dimensions());
        if (!walk_scratch_fits(scratch_bytes)) return {nullptr, status_t::config_mismatch_k};
        std::size_t const outputs_bytes = rows_count * dimensions() * sizeof(f32_t);
        auto staged = stage_walk_(queries(nk::range(std::size_t {0}, rows_count, head_count), nk::slice),
                                  attend_heads_t {queries, head_count}, context,
                                  static_cast<u32_t>(non_zero_expansion_factor(config)), 2,
                                  blocks * (outputs_bytes + rows_count * sizeof(softmax_state_t)), visible);
        if (!staged) return {nullptr, staged.status};
        staged.value.scratch_bytes_per_warp += static_cast<u32_t>(scratch_bytes);
        if (cuda_fill_async(context.missed_.get(), 0, sizeof(u32_t), context.stream()) != cuda_success_k)
            return {nullptr, status_t::device_error_k};
        f32_t* const device_outputs = reinterpret_cast<f32_t*>(staged.value.outputs);
        if (status_t const launched = launch_attend_(
                staged.value, value_lookup, device_outputs,
                reinterpret_cast<softmax_state_t*>(staged.value.outputs + outputs_bytes),
                score_scale * std::numbers::log2e_v<f32_t>, context);
            failed(launched))
            return {nullptr, launched};
        return {device_outputs, status_t::success_k};
    }

#pragma endregion Search

#pragma region Refinement

    /** What one refinement pass hands its launches: the walked rows, the pass-local buffers, and
     *  the walk's arena, which the kernel side turns into its refine views. */
    struct refine_launch_t {

        /** The walked rows: an ID range, or under @c node_ids the offsets into that array. */
        iota_view<node_id_t> nodes;

        /** Optional device-resident IDs to walk instead of the range. */
        u32_t const* node_ids = nullptr;

        /** Rows published when the pass began: the walks' sampling bound and the merge's sweep. */
        u32_t nodes_count = 0;

        /** The shadow neighbors the walks land, @b [rows,connectivity]. */
        u32_t* shadow_neighbors = nullptr;

        /** The reverse buckets @b [rows,bucket_width], their fill counters, and the bitset of rows
         *  the merge re-selects. */
        u32_t* buckets = nullptr;
        u32_t* bucket_counts = nullptr;
        u32_t* touched = nullptr;
        u32_t bucket_width = 0;

        /** The traversal arena, the shape of one warp's slice, and the epoch floor at launch. */
        byte_t* arena = nullptr;
        cuda_arena_shape_t arena_shape {};
        u32_t epoch_base = 0;

        /** The register ladder's width the walks keep. */
        u32_t pool_slots = 0;
    };

    /** The three-pass refinement engine behind @c refine(): the walk covers @p nodes, or under a
     *  device-resident @p node_ids the listed IDs with @p nodes spanning the array's offsets; the
     *  reverse passes re-select the walked rows and the rows they pick, and leave every other row
     *  as it was. */
    expected<refine_result_t> refine_pass_(traversal_context_t& context, refine_config_t config,
                                           iota_view<node_id_t> nodes, std::span<u32_t const> node_ids) noexcept {
        std::size_t const nodes_count = size();
        if (!nodes_count) return {{}, status_t::success_k};
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (!build_scratch_fits()) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        // IDs at or past the published count are unwalkable; an ID array is bounded by its own
        // length.
        if (node_ids.empty()) nodes.bound = static_cast<node_id_t>(min_of<std::size_t>(nodes.bound, nodes_count));
        if (nodes.first >= nodes.bound) return {{}, status_t::success_k};
        std::size_t const walk_count = nodes.size();
        cuda_stream_handle_t const stream = context.stream();
        cuda_api_t const& api = cuda_api();
        // Pass-local buffers no reader observes: the shadow neighbors, the reverse buckets, their
        // counters, and the bitset of rows the merge re-selects; the bucket width is the
        // connectivity.
        std::size_t const shadow_bytes = nodes_count * connectivity() * sizeof(u32_t);
        std::size_t const counters_bytes = (nodes_count + divide_round_up<std::size_t>(nodes_count, 32)) *
                                           sizeof(u32_t);
        u32_t const bucket_width = static_cast<u32_t>(connectivity());
        auto shadow = cuda_make_unique(context.allocator_, shadow_bytes * 2 + counters_bytes);
        if (!shadow) return {{}, status_t::bad_alloc_k};
        nk::matrix_span<u32_t> const shadow_neighbors {reinterpret_cast<u32_t*>(shadow.get()), nodes_count,
                                                       connectivity()};
        nk::matrix_span<u32_t> const buckets {reinterpret_cast<u32_t*>(shadow.get() + shadow_bytes), nodes_count,
                                              bucket_width};
        std::span<u32_t> const counters {reinterpret_cast<u32_t*>(shadow.get() + shadow_bytes * 2),
                                         counters_bytes / sizeof(u32_t)};
        std::span<u32_t> const bucket_counts = counters.first(nodes_count);
        if (cuda_fill_async(counters.data(), 0, counters.size_bytes(), stream) != cuda_success_k)
            return {{}, status_t::device_error_k};

        // Twice the connectivity by default, like every CPU walk, rounded up to the pool sizes the
        // kernel carries.
        u32_t const pool_slots = pool_slots_for_(static_cast<u32_t>(non_zero_expansion_factor(config)), 4);
        cuda_arena_shape_t const arena_shape = arena_shape_for(pool_slots * 32, frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.multiprocessor_count) *
                                        insert_blocks_per_multiprocessor_k * vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {{}, status_t::bad_alloc_k};
        // Only a fresh allocation or a wrapped 32-bit epoch space warrants zeroing, never a warm
        // arena.
        if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + walk_count > 0xFFFFFFFEull) {
            cuda_fill_async(arena, 0, context.arena_bytes_, stream);
            context.epoch_floor_ = 0;
        }
        u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += walk_count;
        cuda_fill_async(context.assigned_rows_.get(), 0, sizeof(u32_t), stream);
        cuda_fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream);

        refine_launch_t launch;
        launch.nodes = nodes;
        launch.node_ids = node_ids.data();
        launch.nodes_count = static_cast<u32_t>(nodes_count);
        launch.shadow_neighbors = shadow_neighbors.data();
        launch.buckets = buckets.data();
        launch.bucket_counts = bucket_counts.data();
        launch.touched = counters.subspan(nodes_count).data();
        launch.bucket_width = bucket_width;
        launch.arena = arena;
        launch.arena_shape = arena_shape;
        launch.epoch_base = epoch_base;
        launch.pool_slots = pool_slots;
        // A walk short of the whole graph starts from the live neighbors copied into the shadow
        // verbatim, so a row it never rewrites but one of its picks lands in merges its own
        // neighbors with the new reverse links.
        if (!node_ids.empty() || walk_count != nodes_count)
            if (cuda_copy_async(shadow_neighbors.data(), neighbors_matrix_.base(), shadow_bytes, stream) !=
                cuda_success_k)
                return {{}, status_t::device_error_k};
        if (status_t const launched = launch_refine_(launch, context); failed(launched)) return {{}, launched};
        if (api.stream_synchronize(stream) != cuda_success_k) return {{}, status_t::device_error_k};
        // The assigned count is the completion oracle: every offset assigned means every shadow row
        // landed.
        u32_t assigned = 0;
        if (cuda_copy(&assigned, context.assigned_rows_.get(), sizeof(u32_t)) != cuda_success_k ||
            assigned < walk_count)
            return {{}, status_t::device_error_k};

        // Scatter the walked rows' shadow neighbors in reverse, then merge each touched node's
        // forward & reverse candidates into its final live row. A heal round over island IDs
        // re-selects the islands' own rows alone: re-judging the rows they pick would drop the
        // links that reconnected earlier islands.
        cuda_fill_async(context.assigned_rows_.get(), 0, sizeof(u32_t), stream);
        if (status_t const launched = launch_merge_(launch, context); failed(launched)) return {{}, launched};
        if (api.stream_synchronize(stream) != cuda_success_k) return {{}, status_t::device_error_k};
        if (cuda_copy(&assigned, context.assigned_rows_.get(), sizeof(u32_t)) != cuda_success_k ||
            assigned < nodes_count)
            return {{}, status_t::device_error_k};

        vram_telemetry_block_t const spent = context.spent();
        refine_result_t result;
        result.visited_nodes = walk_count;
        result.changed_nodes = spent.changed_nodes;
        result.visited_members = spent.hops;
        result.computed_distances = spent.scored;
        result.restarts = spent.restarts;
        return {result, status_t::success_k};
    }

    /** Device scratch one reachability audit sweeps in, reused across a heal's rounds - one
     *  allocation carved into the reached set and two frontiers, the expanded, unreached and
     *  reconnected counters, and the unreached rows' IDs and their in-link counts when a caller
     *  collects them. */
    struct audit_t {
        typename traversal_context_t::bytes_ptr_t bytes;

        /** The reached set and the two frontiers, one bit per row each. */
        u32_t* reached = nullptr;
        u32_t* frontier = nullptr;
        u32_t* next_frontier = nullptr;
        u32_t* expanded = nullptr;
        u32_t* unreached = nullptr;
        u32_t* reconnected = nullptr;
        u32_t* island_ids = nullptr;
        u32_t* in_links = nullptr;
        std::size_t words = 0;

        /** Levels the last audit swept to convergence. */
        std::size_t sweeps = 0;
    };

    /** Allocates an audit over the rows published now, with room for @p island_slots unreached IDs
     *  and as many in-link counts, the reconnected counter zeroed. */
    expected<audit_t> reserve_audit_(traversal_context_t& context, std::size_t island_slots) noexcept {
        audit_t audit {typename traversal_context_t::bytes_ptr_t {
            nullptr, cuda_deleter<typename traversal_context_t::allocator_t> {context.allocator_, 0}}};
        audit.words = divide_round_up<std::size_t>(size(), 32);
        std::size_t const bytes = (audit.words * 3 + 3 + island_slots * 2) * sizeof(u32_t);
        audit.bytes.reset(context.allocator_.allocate(bytes));
        audit.bytes.get_deleter().count = bytes;
        if (!audit.bytes) return {std::move(audit), status_t::bad_alloc_k};
        audit.reached = reinterpret_cast<u32_t*>(audit.bytes.get());
        audit.frontier = audit.reached + audit.words;
        audit.next_frontier = audit.frontier + audit.words;
        audit.expanded = audit.next_frontier + audit.words;
        audit.unreached = audit.expanded + 1;
        audit.reconnected = audit.unreached + 1;
        audit.island_ids = island_slots ? audit.reconnected + 1 : nullptr;
        audit.in_links = island_slots ? audit.island_ids + island_slots : nullptr;
        if (cuda_fill_async(audit.reconnected, 0, sizeof(u32_t), context.stream()) != cuda_success_k)
            return {std::move(audit), status_t::device_error_k};
        return {std::move(audit), status_t::success_k};
    }

    /** Sweeps reachability to convergence from as many stride-spread landmarks as one search
     *  draws - the connectivity - then counts the rows it never reached, collecting their IDs
     *  when @p audit has room for them. */
    expected<u32_t> audit_(traversal_context_t& context, audit_t& audit) noexcept {
        std::size_t const rows = size();
        if (!vectors_matrix_.reserved_bytes()) return {0, status_t::uninitialized_k};
        if (context.device() != device_.ordinal) return {0, status_t::config_mismatch_k};
        cuda_stream_handle_t const stream = context.stream();
        std::size_t const seed_count = min_of<std::size_t>(connectivity(), rows);
        // The bitsets, the expanded counter and the unreached counter lie back to back.
        if (cuda_fill_async(audit.reached, 0, (audit.words * 3 + 2) * sizeof(u32_t), stream) != cuda_success_k)
            return {0, status_t::device_error_k};
        u32_t* frontier = audit.frontier;
        u32_t* next_frontier = audit.next_frontier;
        if (status_t const launched = launch_audit_seeds_(audit.reached, frontier, rows, seed_count, stream);
            failed(launched))
            return {0, launched};

        // Every row joins one frontier at most, so the levels are bounded by the landmarks'
        // eccentricity, capped defensively.
        constexpr std::size_t sweeps_cap_k = 256;
        for (audit.sweeps = 1; audit.sweeps != sweeps_cap_k; ++audit.sweeps) {
            if (cuda_fill_async(audit.expanded, 0, sizeof(u32_t), stream) != cuda_success_k ||
                cuda_fill_async(next_frontier, 0, audit.words * sizeof(u32_t), stream) != cuda_success_k)
                return {0, status_t::device_error_k};
            if (status_t const launched = launch_audit_sweep_(audit.reached, frontier, next_frontier, audit.expanded,
                                                              rows, stream);
                failed(launched))
                return {0, launched};
            auto const [expanded_rows, read_status] = read_counter_(audit.expanded, stream);
            if (failed(read_status)) return {0, read_status};
            if (!expanded_rows) break;
            std::swap(frontier, next_frontier);
        }
        if (status_t const launched = launch_audit_islands_(audit.reached, rows, audit.island_ids, audit.unreached,
                                                            stream);
            failed(launched))
            return {0, launched};
        return read_counter_(audit.unreached, stream);
    }

    /** One device counter read back once the stream drains. */
    static expected<u32_t> read_counter_(u32_t const* counter, cuda_stream_handle_t stream) noexcept {
        u32_t value = 0;
        if (cuda_copy_async(&value, counter, sizeof(u32_t), stream) != cuda_success_k ||
            cuda_api().stream_synchronize(stream) != cuda_success_k)
            return {0, status_t::device_error_k};
        return {value, status_t::success_k};
    }

#pragma endregion Refinement

#pragma region Launches

    /** Insertion blocks per multiprocessor: the resident count saturates the path at its
     *  registers. */
    inline static constexpr u32_t insert_blocks_per_multiprocessor_k = vram_resident_blocks_per_multiprocessor_k;

    /** Search blocks per multiprocessor: oversubscribed, so enough queries stay in flight to reach
     *  the memory-level-parallelism knee. */
    inline static constexpr u32_t search_blocks_per_multiprocessor_k = 8;

    /** Threads per block, one warp's worth per resident warp. */
    inline static constexpr u32_t block_threads_k = vram_warps_per_block_k * 32;

    /** Keeps @p failure as the status unless one is kept already; returns it for the caller's own
     *  report. */
    status_t fail_(status_t failure) noexcept {
        status_t expected = status_t::success_k;
        standard_atomic_ref<status_t>(status_).compare_exchange_strong(expected, failure, std::memory_order_acq_rel,
                                                                       std::memory_order_acquire);
        return failure;
    }

    /** Rows claimed by batches and loads, visible or not: where the next claim starts. */
    std::size_t claimed_() const noexcept {
        return standard_atomic_ref<std::size_t const>(claimed_rows_).load(std::memory_order_acquire);
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
        standard_atomic_ref<std::size_t>(last_visible_id_).store(first + rows, std::memory_order_release);
        return status_t::success_k;
    }

    /** Maps every matrix and column up to @p rows and sentinel-fills the fresh neighbor rows -
     *  always ahead of any launch that could assign new_node_ids into them. Callers hold
     *  @c claiming_, or run exclusively like @c reserve and @c load. */
    status_t grow_rows_(std::size_t rows, cuda_stream_handle_t stream) noexcept {
        if (rows <= capacity()) return status_t::success_k;
        std::size_t const neighbors_before = neighbors_matrix_.committed_bytes();
        if (status_t status = vectors_matrix_.commit(rows * row_bytes()); status != status_t::success_k) return status;
        if (status_t status = kernel_ == nk_kernel_dot_k ? status_t::success_k
                                                         : norms_column_.commit(rows * sizeof(norms_t));
            status != status_t::success_k)
            return status;
        if (status_t status = neighbors_matrix_.commit(rows * connectivity() * sizeof(u32_t));
            status != status_t::success_k)
            return status;
        // Past a successful commit a failure must be kept: `capacity()` has advanced, so a retry
        // would early-return and assign IDs into rows never sentinel-filled.
        std::size_t const fresh_bytes = neighbors_matrix_.committed_bytes() - neighbors_before;
        cuda_api_t const& api = cuda_api();
        if (fresh_bytes && api.fill_words_async(cuda_address(neighbors_matrix_.base() + neighbors_before),
                                                node_id_unknown_k, fresh_bytes / 4, stream) != cuda_success_k)
            return fail_(status_t::device_error_k);
        if (api.stream_synchronize(stream) != cuda_success_k) return fail_(status_t::device_error_k);
        return status_t::success_k;
    }

    /** Rounds the requested expansion to the closed pool ladder: 64/128/256/512. */
    static u32_t pool_slots_for_(u32_t expansion_factor, u32_t default_slots) noexcept {
        if (!expansion_factor) return default_slots;
        u32_t const slots = std::bit_ceil(divide_round_up<u32_t>(expansion_factor, 32));
        return min_of(max_of(slots, 2u), vram_widest_pool_slots_k);
    }

    /** Loads this ceiling's fatbin onto the device and finds every kernel this dtype exports bar
     *  the attends, by the mangled name @ref cuda_kernel_name spells. */
    status_t load_kernels_() noexcept {
        auto [module, status] = cuda_module_t::load(device_.ordinal, cuda_fatbin<capability_>());
        if (failed(status)) return status;
        module_ = std::move(module);
        char name[cuda_kernel_name_capacity_k];
        for (cuda_rung_t const rung : cuda_rungs) {
            cuda_kernel_name(name, rung, capability_, vector_dtype_);
            cuda_function_handle_t& function = kernels_[static_cast<u32_t>(rung.kernel)][cuda_rung_index(rung.slots)];
            if (!(function = module_.function(name))) return status_t::device_error_k;
        }
        return status_t::success_k;
    }

    /** The loaded @p kernel on its @p slots rung. */
    cuda_function_handle_t kernel_function_(cuda_kernel_t kernel, u32_t slots) const noexcept {
        return kernels_[static_cast<u32_t>(kernel)][cuda_rung_index(slots)];
    }

    /** Launches @p function over @p blocks of the standard warp complement on @p stream, raising
     *  its dynamic-shared ceiling first when @p shared_bytes outgrow the static window. */
    static status_t launch_(cuda_function_handle_t function, u32_t blocks, u32_t shared_bytes,
                            cuda_stream_handle_t stream, cuda_arguments_t const& arguments) noexcept {
        if (shared_bytes > cuda_static_scratch_limit_k &&
            (!function || cuda_api().function_set_attribute(function, cuda_function_dynamic_shared_bytes_k,
                                                            static_cast<int>(shared_bytes)) != cuda_success_k))
            return status_t::device_error_k;
        return cuda_launch(function, blocks, block_threads_k, shared_bytes, stream, arguments);
    }

    /** Launches the rows of @p build in waves, each queued behind the last and as wide as the IDs
     *  below it, never narrower than @c vram_admission_ramp_floor_k, so every ID a wave admits
     *  walks rows already published. A wave failing after an earlier one launched keeps the
     *  failure, since those rows can no longer be un-claimed. */
    status_t launch_insert_(cuda_build_t const& build, u32_t pool_slots, traversal_context_t& context) noexcept {
        cuda_arguments_t arguments {};
        cuda_function_handle_t const function = kernel_function_(cuda_kernel_t::insert_k, pool_slots);
        u32_t const blocks = static_cast<u32_t>(device_.multiprocessor_count) * insert_blocks_per_multiprocessor_k;
        for (u32_t wave_first = build.first_id; wave_first != build.end_id;) {
            u32_t const wave_end = static_cast<u32_t>(
                min_of<u64_t>(build.end_id, u64_t {wave_first} + max_of(wave_first, vram_admission_ramp_floor_k)));
            u32_t const offset = wave_first - build.first_id;
            arguments.build = build;
            arguments.build.staging = build.staging + std::size_t {offset} * row_bytes();
            arguments.build.first_id = wave_first;
            arguments.build.end_id = wave_end;
            arguments.build.epoch_base = build.epoch_base + offset;
            status_t launched = cuda_fill_async(build.assigned_rows, 0, sizeof(u32_t), context.stream()) ==
                                        cuda_success_k
                                    ? status_t::success_k
                                    : status_t::device_error_k;
            if (succeeded(launched))
                launched = launch_(function, blocks, vram_warps_per_block_k * build.scratch_bytes_per_warp,
                                   context.stream(), arguments);
            if (failed(launched)) return offset ? fail_(launched) : launched;
            wave_first = wave_end;
        }
        return status_t::success_k;
    }

    /** Walks the rows of @p launch, landing their re-selected neighbors in its shadow matrix. */
    status_t launch_refine_(refine_launch_t const& launch, traversal_context_t& context) noexcept {
        cuda_arguments_t arguments {};
        arguments.refine = {storage(),
                            launch.shadow_neighbors,
                            static_cast<u32_t>(launch.nodes.bound),
                            insert_scratch_bytes_per_warp(),
                            launch.nodes.first,
                            launch.arena,
                            launch.arena_shape,
                            launch.epoch_base,
                            context.telemetry_.get(),
                            context.assigned_rows_.get(),
                            launch.node_ids,
                            launch.nodes_count};
        return launch_(kernel_function_(cuda_kernel_t::refine_k, launch.pool_slots),
                       static_cast<u32_t>(device_.multiprocessor_count) * insert_blocks_per_multiprocessor_k,
                       vram_warps_per_block_k * arguments.refine.scratch_bytes_per_warp, context.stream(), arguments);
    }

    /** Scatters the walked rows' shadow neighbors into the reverse buckets, then merges every
     *  touched row's forward and reverse candidates into its live neighbors. */
    status_t launch_merge_(refine_launch_t const& launch, traversal_context_t& context) noexcept {
        cuda_stream_handle_t const stream = context.stream();
        u32_t const blocks = static_cast<u32_t>(device_.multiprocessor_count) * insert_blocks_per_multiprocessor_k;
        cuda_arguments_t arguments {};
        cuda_graph_t& graph = arguments.graph;
        graph.neighbors = launch.shadow_neighbors;
        graph.rows_count = launch.nodes_count;
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.first_id = static_cast<u32_t>(launch.nodes.first);
        graph.walk_count = static_cast<u32_t>(launch.nodes.size());
        graph.node_ids = launch.node_ids;
        graph.bucket_width = launch.bucket_width;
        graph.bucket_ids = launch.buckets;
        graph.bucket_counts = launch.bucket_counts;
        graph.touched = {launch.touched};
        if (status_t const launched = launch_(kernel_function_(cuda_kernel_t::scatter_links_k, 0), blocks, 0, stream,
                                              arguments);
            failed(launched))
            return launched;
        // The merge sweeps the whole ID space and skips the rows the scatter left untouched.
        arguments.refine = {storage(),
                            launch.shadow_neighbors,
                            launch.nodes_count,
                            merge_scratch_bytes_per_warp(),
                            0,
                            launch.arena,
                            launch.arena_shape,
                            launch.epoch_base,
                            context.telemetry_.get(),
                            context.assigned_rows_.get(),
                            nullptr,
                            launch.nodes_count};
        return launch_(kernel_function_(cuda_kernel_t::merge_links_k, 0), blocks,
                       vram_warps_per_block_k * arguments.refine.scratch_bytes_per_warp, stream, arguments);
    }

    /** The search view every walking verb launches over. */
    cuda_search_t search_view_(walk_launch_t const& walk, traversal_context_t& context) const noexcept {
        return {storage(),
                reinterpret_cast<byte_t const*>(walk.queries),
                walk.queries_count,
                walk.visible,
                walk.scratch_bytes_per_warp,
                walk.expansion,
                walk.arena,
                walk.arena_shape,
                walk.epoch_base,
                context.telemetry_.get()};
    }

    /** The rung a walk runs on: its register ladder's width, or zero for the arena frontier. */
    static u32_t walk_slots_(walk_launch_t const& walk) noexcept {
        return walk.frontier == frontier_kind_t::wide_arena_k ? 0 : walk.pool_slots;
    }

    /** Launches one walk per query of @p walk through @p function over @p arguments, and reports
     *  the launch's own verdict. */
    status_t launch_walk_(cuda_function_handle_t function, walk_launch_t const& walk, cuda_arguments_t const& arguments,
                          traversal_context_t& context) noexcept {
        return launch_(function, static_cast<u32_t>(device_.multiprocessor_count) * search_blocks_per_multiprocessor_k,
                       vram_warps_per_block_k * walk.scratch_bytes_per_warp, context.stream(), arguments);
    }

    /** Walks every query of @p walk, landing its nearest @p count IDs and, unless null, their
     *  distances; staged indexer weights pick the indexer-rescored walk. */
    status_t launch_search_(walk_launch_t const& walk, node_id_t* ids, distance_t* distances, u32_t count,
                            traversal_context_t& context) noexcept {
        cuda_arguments_t arguments {};
        arguments.search = search_view_(walk, context);
        arguments.search_sink = {ids, distances, count, kernel_};
        cuda_kernel_t kernel = cuda_kernel_t::search_k;
        // Only the floating dtypes take indexer heads, so only they rescore.
        if constexpr (floating_scalar<scalar_t>)
            if (walk.weights) {
                arguments.rescoring = {reinterpret_cast<byte_t const*>(walk.heads), walk.weights, walk.heads_count,
                                       row_fragments()};
                kernel = cuda_kernel_t::search_indexer_k;
            }
        return launch_walk_(kernel_function_(kernel, walk_slots_(walk)), walk, arguments, context);
    }

    /** Walks every row of @p walk, landing up to @p wanted kept pairs per row and the pairs each
     *  scored. */
    status_t launch_multiply_(walk_launch_t const& walk, node_id_t* ids, dot_t* products, u32_t* counts, u32_t wanted,
                              traversal_context_t& context) noexcept {
        cuda_arguments_t arguments {};
        arguments.search = search_view_(walk, context);
        arguments.multiply_sink = {ids, reinterpret_cast<byte_t*>(products), counts, wanted};
        return launch_walk_(kernel_function_(cuda_kernel_t::multiply_k, walk_slots_(walk)), walk, arguments, context);
    }

    /** Walks every token of @p walk under its staged heads, folding the value rows @p value_lookup
     *  resolves into the normalized @p outputs and their @p states; the lookup travels as bytes to
     *  the attend kernel its @c device_lookup_k names, found in the fatbin by name. */
    template <typename value_lookup_type_>
    status_t launch_attend_(walk_launch_t const& walk, value_lookup_type_ const& value_lookup, f32_t* outputs,
                            softmax_state_t* states, f32_t logit_scale, traversal_context_t& context) noexcept {
        static_assert(std::is_trivially_copyable_v<value_lookup_type_> &&
                          sizeof(value_lookup_type_) <= cuda_value_lookup_capacity_k &&
                          alignof(value_lookup_type_) <= 16,
                      "A device value lookup travels to the kernels as its own bytes");
        cuda_arguments_t arguments {};
        arguments.search = search_view_(walk, context);
        arguments.rescoring = {reinterpret_cast<byte_t const*>(walk.heads), nullptr, walk.heads_count, row_fragments()};
        cuda_attend_t& attend = arguments.attend;
        attend.outputs = outputs;
        attend.states = states;
        attend.missed = context.missed_.get();
        attend.logit_scale = logit_scale;
        attend.heads_count = walk.heads_count;
        attend.columns = static_cast<u32_t>(dimensions());
        std::memcpy(attend.value_lookup, &value_lookup, sizeof(value_lookup));
        char name[cuda_kernel_name_capacity_k];
        cuda_kernel_name(name, {cuda_kernel_t::attend_k, walk_slots_(walk)}, capability_, vector_dtype_,
                         value_lookup_type_::device_lookup_k);
        return launch_walk_(module_.function(name), walk, arguments, context);
    }

    /** Rebuilds the norms of rows @b [first_row,first_row+rows) from their vectors. */
    status_t launch_derive_norms_(std::size_t first_row, std::size_t rows, cuda_stream_handle_t stream) noexcept {
        cuda_arguments_t arguments {};
        arguments.graph.storage = storage();
        arguments.graph.first_id = static_cast<u32_t>(first_row);
        arguments.graph.rows_count = static_cast<u32_t>(rows);
        return launch_(kernel_function_(cuda_kernel_t::derive_norms_k, 0),
                       static_cast<u32_t>(device_.multiprocessor_count) * search_blocks_per_multiprocessor_k, 0, stream,
                       arguments);
    }

    /** Marks the audit's stride-spread landmarks over @p rows rows reached and in @p frontier. */
    status_t launch_audit_seeds_(u32_t* reached, u32_t* frontier, std::size_t rows, std::size_t seed_count,
                                 cuda_stream_handle_t stream) noexcept {
        cuda_arguments_t arguments {};
        arguments.graph.reached = {reached};
        arguments.graph.frontier = {frontier};
        arguments.graph.rows_count = static_cast<u32_t>(rows);
        arguments.graph.seed_count = static_cast<u32_t>(seed_count);
        return launch_(kernel_function_(cuda_kernel_t::mark_seeds_k, 0),
                       divide_round_up<u32_t>(static_cast<u32_t>(seed_count), block_threads_k), 0, stream, arguments);
    }

    /** Expands one level of reachability from @p frontier into @p next_frontier, counting the rows
     *  it expanded in @p expanded. */
    status_t launch_audit_sweep_(u32_t* reached, u32_t* frontier, u32_t* next_frontier, u32_t* expanded,
                                 std::size_t rows, cuda_stream_handle_t stream) noexcept {
        cuda_arguments_t arguments {};
        cuda_graph_t& graph = arguments.graph;
        graph.neighbors = reinterpret_cast<u32_t*>(neighbors_matrix_.base());
        graph.rows_count = static_cast<u32_t>(rows);
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.reached = {reached};
        graph.frontier = {frontier};
        graph.next_frontier = {next_frontier};
        graph.counter = expanded;
        return launch_(kernel_function_(cuda_kernel_t::reachability_sweep_k, 0),
                       static_cast<u32_t>(device_.multiprocessor_count) * search_blocks_per_multiprocessor_k, 0, stream,
                       arguments);
    }

    /** Counts the rows @p reached lacks in @p islands_cursor, collecting them into @p island_ids
     *  when it is given. */
    status_t launch_audit_islands_(u32_t* reached, std::size_t rows, u32_t* island_ids, u32_t* islands_cursor,
                                   cuda_stream_handle_t stream) noexcept {
        cuda_arguments_t arguments {};
        arguments.graph.reached = {reached};
        arguments.graph.rows_count = static_cast<u32_t>(rows);
        arguments.graph.island_ids = island_ids;
        arguments.graph.counter = islands_cursor;
        return launch_(kernel_function_(cuda_kernel_t::reachability_islands_k, 0),
                       divide_round_up<u32_t>(static_cast<u32_t>(rows), block_threads_k), 0, stream, arguments);
    }

    /** Counts the reached in-links of every row, then reconnects the audit's @p islands islands
     *  into reached picks or their landmarks. */
    status_t launch_reconnect_islands_(audit_t const& audit, u32_t islands, std::size_t rows,
                                       cuda_stream_handle_t stream) noexcept {
        cuda_arguments_t arguments {};
        cuda_graph_t& graph = arguments.graph;
        graph.neighbors = reinterpret_cast<u32_t*>(neighbors_matrix_.base());
        graph.rows_count = static_cast<u32_t>(rows);
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.reached = {audit.reached};
        graph.in_links = audit.in_links;
        graph.island_ids = audit.island_ids;
        graph.islands_count = islands;
        graph.seed_count = static_cast<u32_t>(min_of<std::size_t>(connectivity(), rows));
        graph.reconnected = audit.reconnected;
        if (status_t const counted = launch_(
                kernel_function_(cuda_kernel_t::reached_in_links_k, 0),
                static_cast<u32_t>(device_.multiprocessor_count) * search_blocks_per_multiprocessor_k, 0, stream,
                arguments);
            failed(counted))
            return counted;
        return launch_(kernel_function_(cuda_kernel_t::reconnect_island_k, 0),
                       divide_round_up<u32_t>(islands, block_threads_k), 0, stream, arguments);
    }

    /** Merges the attended rows and carries of @p shards walks into the first's on @p stream - the
     *  sharded attend's gather. */
    status_t launch_merge_attention_(byte_t* blocks, u32_t rows, u32_t columns, u32_t shards,
                                     cuda_stream_handle_t stream) noexcept {
        cuda_arguments_t arguments {};
        arguments.attend.outputs = reinterpret_cast<f32_t*>(blocks);
        arguments.attend.rows = rows;
        arguments.attend.columns = columns;
        arguments.attend.shards = shards;
        return launch_(kernel_function_(cuda_kernel_t::merge_attention_k, 0),
                       divide_round_up<u32_t>(rows, vram_warps_per_block_k), 0, stream, arguments);
    }

#pragma endregion Launches

    /** The vectors matrix and norms column backend, rebound to bytes. */
    using vectors_matrix_allocator_t = typename vectors_allocator_t::template rebind<byte_t>::other;

    /** The neighbors matrix backend, rebound to bytes. */
    using neighbors_matrix_allocator_t = typename neighbors_allocator_t::template rebind<byte_t>::other;

    /** The published-row counter is plain stream-ordered device memory on the legacy stream. */
    using counter_allocator_t = cuda_device_allocator<u32_t>;

    /** This ceiling's fatbin on this device, and its kernels in @ref cuda_kernel_t order, the
     *  attends left to their launches. */
    cuda_module_t module_;
    cuda_function_handle_t kernels_[static_cast<u32_t>(cuda_kernel_t::count_k)][cuda_rungs_per_kernel_k] = {};

    /** Hardware identity: the ordinal comes from the vectors allocator at construction, the
     *  remaining specs fill at the first @c reserve. */
    cuda_device_specs_t device_;

    /** The metric every walk scores by; under dot, the one metric that reads no norms, the norms
     *  column stays empty. */
    nk_kernel_kind_t kernel_;

    /** The frozen geometry word: dimensions high, connectivity low, zero when unreserved. */
    packed_geometry_id_t geometry_ = 0;

    /** The vectors matrix and norms column; each allocator is its storage, owning the base and the
     *  committed extent, so capacities are derived, never cached. */
    vectors_matrix_allocator_t vectors_matrix_, norms_column_;

    /** The neighbors matrix, on the same terms. */
    neighbors_matrix_allocator_t neighbors_matrix_;

    /** The device-resident published-row count, allocated by the first @c reserve(); every driving
     *  context polls it, none owns it. */
    cuda_unique_ptr<u32_t, counter_allocator_t> published_rows_;

    /** Held by the one batch growing and claiming at a time. */
    fu::spin_mutex_t claiming_;

    /** Each batch's dense ID range, release-published once its launch is accepted. */
    std::size_t claimed_rows_ = 0;

    /** Rows strictly below it are published and visible: turned once per batch, in claim order,
     *  after its drain. */
    std::size_t last_visible_id_ = 0;

    /** The first post-claim device failure: IDs past it may be unpublished or unfilled. */
    status_t status_ = status_t::success_k;
};

} // namespace unum::usearch::dense
