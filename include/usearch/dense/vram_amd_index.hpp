/**
 *  @file include/usearch/dense/vram_amd_index.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Vector Search index for a single uniform latency memory space on AMD GPUs.
 *
 *  The host face of the AMD family, compiled by any C++ compiler: it fills plain argument records
 *  and launches the kernels of @c vram_amd_index.hip by name through the HIP runtime it opens at
 *  first use, so it never sees a kit, a shared view or HIP beyond its runtime's declarations. Every
 *  kernel is one instantiation of a single template over five integers - the kernel, the class, the
 *  dtype, the register rung and the attend value lookup - so its mangled name is spelled here from
 *  those integers alone, and one code object per class carries every dtype.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::snprintf` - a kernel's mangled name
#include <cstring> // `std::memcpy` - an attend's value lookup travels as bytes

#include <bit>         // `std::bit_ceil`, `std::has_single_bit`
#include <limits>      // `std::numeric_limits` - the unit-RMS import's epsilon
#include <mutex>       // `std::unique_lock` while one context grows the storage and claims
#include <numbers>     // `std::numbers::log2e_v` - the base-2 softmax
#include <span>        // `std::span`
#include <type_traits> // `std::is_same_v` - which heads a walk stages
#include <utility>     // `std::exchange`

#include <forkunion/types.hpp> // `fu::spin_mutex_t`, `fu::standard_yield_t`

#include <numkong/capabilities.h> // `nk_kernel_kind_t`
#include <numkong/cast/serial.h>  // `nk_cast_plan_serial_` - the device casts' plans
#include <numkong/each/serial.h>  // `nk_each_rmsnorm_arguments_t` - the unit-RMS import's record
#include <numkong/tensor.hpp>     // `nk::matrix_span` - the batches' views
#include <numkong/types.h>
#include <numkong/vector.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_amd_device.hpp"
#include "usearch/dense/types.hpp" // `vram_telemetry_block_t` & the launch geometry constants

namespace unum::usearch::dense {

#pragma region Kernel Records

/** Bytes of every AMD kit's fragment and of one scored node, the two strides the host sizes by. */
inline constexpr std::size_t amd_fragment_bytes_k = 16;
inline constexpr std::size_t amd_scored_node_bytes_k = 8;

/** Bytes of one wavefront's static schedule state: the plain schedule's one flag. */
inline constexpr std::size_t amd_schedule_bytes_k = 1;

/** Bytes of one wavefront's staged batch over @p wave_lanes lanes: the candidate IDs, then a
 *  lane-width of scored nodes and four-byte dots. */
constexpr std::size_t amd_gathered_bytes(u32_t wave_lanes) noexcept {
    return vram_cache_rows_k * sizeof(u32_t) + wave_lanes * (amd_scored_node_bytes_k + 4);
}

/** Bytes of one walking wavefront's attend scratch over @p heads_count heads of @p columns columns
 *  and @p wave_lanes lanes: every head's batch dots, its running maximum, partition and latest
 *  rescale, then its accumulator row, in whole fragments so the next wavefront's query span stays
 *  aligned. */
constexpr std::size_t amd_attend_scratch_bytes(std::size_t heads_count, std::size_t columns,
                                               u32_t wave_lanes) noexcept {
    return round_up_to_multiple<std::size_t>(heads_count * (wave_lanes + 3 + columns) * sizeof(f32_t),
                                             amd_fragment_bytes_k);
}

/** Sets the keyed twin of a value lookup apart from the bare one in an attend kernel's name. */
inline constexpr u32_t amd_keyed_lookup_k = 1u << 16;

/** A key-level value lookup seen through one shard's device key column, so the shard's walks
 *  resolve node IDs: the kernels read a node's row as @c value_lookup(keys[node_id]). */
template <typename key_type_, typename value_lookup_type_>
struct amd_keyed_lookup {

    /** Names this lookup to the attend kernels, apart from the bare key-level one. */
    inline static constexpr u32_t device_lookup_k = value_lookup_type_::device_lookup_k | amd_keyed_lookup_k;

    /** The shard's keys on its device, by node ID. */
    key_type_ const* keys;

    /** Resolves a key to its value row. */
    value_lookup_type_ value_lookup;
};

/** The device value lookup an attend kernel resolves rows through, by the @c device_lookup_k a
 *  lookup type names itself with: whichever unit compiles a code object specializes it with
 *  @c type for every lookup that code object's attend kernels take. */
template <u32_t lookup_>
struct amd_value_lookup_of;

/** Bytes of a value lookup an attend carries to its kernels. */
inline constexpr std::size_t amd_value_lookup_capacity_k = 64;

/** One bit per row over device-resident words, marked concurrently by a whole grid. */
struct hip_bitset_t {
    u32_t* words = nullptr;
};

/** One wavefront's arena slice, mirroring the shared @c vram_arena_shape_t field by field. */
struct amd_arena_shape_t {
    u32_t visited_slots = 0;
    u32_t visited_hash_shift = 0;
    u32_t tail_entries = 0;
    u32_t cache_bytes = 0;

    /** A multiple of 16, as the kernels' twin rounds it. */
    constexpr std::size_t slice_bytes() const noexcept {
        std::size_t const bytes = visited_slots * sizeof(u64_t) + tail_entries * amd_scored_node_bytes_k + cache_bytes;
        return (bytes + 15) & ~std::size_t {15};
    }
};

/** The exact visited set sized well past @p expansion visits, then @p tail_entries of spill
 *  tail. */
inline amd_arena_shape_t amd_arena_shape(u32_t expansion, u32_t tail_entries) noexcept {
    u32_t const visited_slots = std::bit_ceil(64 * expansion);
    return {visited_slots, 32 - static_cast<u32_t>(std::bit_width(visited_slots - 1)), tail_entries, 0};
}

/** The flat storage a kernel reads, mirroring the shared flat view field by field. */
struct amd_flat_storage_t {
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
struct amd_build_t {
    amd_flat_storage_t storage;
    byte_t const* staging = nullptr;
    u32_t end_id = 0;
    u32_t scratch_bytes_per_warp = 0;
    node_id_t first_id = 0;
    byte_t* arena = nullptr;
    amd_arena_shape_t arena_shape;
    u32_t expansion = 0;
    u32_t epoch_base = 0;
    u32_t* published_rows = nullptr;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_rows = nullptr;
};

/** One walking launch, mirroring the shared search view. */
struct amd_search_t {
    amd_flat_storage_t storage;
    byte_t const* queries = nullptr;
    u32_t queries_count = 0;
    u32_t sampling_bound = 0;
    u32_t scratch_bytes_per_warp = 0;
    u32_t expansion = 0;
    byte_t* arena = nullptr;
    amd_arena_shape_t arena_shape;
    u32_t epoch_base = 0;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_queries = nullptr;
};

/** One refinement launch, mirroring the shared refine view. */
struct amd_refine_t {
    amd_flat_storage_t storage;
    u32_t* shadow_neighbors = nullptr;
    u32_t end_id = 0;
    u32_t scratch_bytes_per_warp = 0;
    node_id_t first_id = 0;
    byte_t* arena = nullptr;
    amd_arena_shape_t arena_shape;
    u32_t expansion = 0;
    u32_t epoch_base = 0;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_rows = nullptr;
    u32_t const* node_ids = nullptr;
    u32_t sampling_bound = 0;
    u32_t const* part_bases = nullptr;
    u32_t parts = 0;
    u32_t round = 0;
};

/** A search's landing, mirroring the shared search sink. */
struct amd_search_sink_t {
    node_id_t* result_ids = nullptr;
    distance_t* result_distances = nullptr;
    u32_t count = 0;
    nk_kernel_kind_t kernel = nk_kernel_dot_k;
};

/** A graph product's landing, mirroring the shared multiply sink; the products are the kit's
 *  dots. */
struct amd_multiply_sink_t {
    node_id_t* ids = nullptr;
    byte_t* products = nullptr;
    u32_t* counts = nullptr;
    u32_t wanted = 0;
};

/** An indexer's heads, mirroring the shared indexer rescoring; null heads score by the query
 *  alone. */
struct amd_rescoring_t {
    byte_t const* heads = nullptr;
    f32_t const* weights = nullptr;
    u32_t heads_count = 0;
    u32_t row_fragments = 0;
};

/** What the graph kernels read: the scatter, the merge's buckets, the norms, the rebase and the
 *  audits. */
struct amd_graph_t {
    amd_flat_storage_t storage;
    u32_t* neighbors = nullptr;
    u32_t rows_count = 0;
    u32_t connectivity = 0;
    u32_t first_id = 0;
    u32_t source_first = 0;
    u32_t source_bound = 0;
    u32_t shift = 0;
    u32_t walk_count = 0;
    u32_t const* node_ids = nullptr;
    u32_t bucket_width = 0;
    u32_t* bucket_ids = nullptr;
    u32_t* bucket_counts = nullptr;
    hip_bitset_t touched, reached, frontier, next_frontier;
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
struct amd_attend_t {
    f32_t* outputs = nullptr;
    softmax_state_t* states = nullptr;
    u32_t* missed = nullptr;
    f32_t logit_scale = 0;
    u32_t heads_count = 0;
    u32_t columns = 0;
    u32_t rows = 0;
    u32_t shards = 0;
    alignas(16) byte_t value_lookup[amd_value_lookup_capacity_k] = {};
};

/** A ranking of keys on the device, mirroring the shared ranking view. */
struct amd_ranking_t {
    byte_t* keys = nullptr;
    byte_t* scores = nullptr;
    std::size_t* counts = nullptr;
    std::size_t keys_stride = 0;
    std::size_t scores_stride = 0;
    u32_t width = 0;
    u32_t key_bytes = 0;
    u32_t score_bytes = 0;
};

/** A sharded search's rankings: a shard's @c walked IDs and distances per query resolve through
 *  its device @c keys into @c ranking, and the first shard merges @c shards rankings, gathered
 *  @c block_bytes apart from @c gathered, into it. */
struct amd_matches_t {
    node_id_t const* ids = nullptr;
    distance_t const* distances = nullptr;
    byte_t const* keys = nullptr;
    amd_ranking_t gathered;
    amd_ranking_t ranking;
    std::size_t block_bytes = 0;
    u32_t walked = 0;
    u32_t queries_count = 0;
    u32_t shards = 0;
};

/** The one argument every AMD kernel takes, each reading the records its kernel names: the casts
 *  read NumKong's own plans. */
struct amd_arguments_t {
    amd_build_t build;
    amd_search_t search;
    amd_refine_t refine;
    amd_search_sink_t search_sink;
    amd_multiply_sink_t multiply_sink;
    amd_rescoring_t rescoring;
    amd_graph_t graph;
    amd_attend_t attend;
    amd_matches_t matches;
    nk_cast_arguments_t cast;
    nk_each_rmsnorm_arguments_t rmscast;
};

#pragma endregion Kernel Records

#pragma region Kernel Names

/** The AMD kernels, the first template argument of @c amd_kernel. */
enum class amd_kernel_t : u32_t {
    insert_k,
    refine_k,
    search_k,
    search_indexer_k,
    multiply_k,
    attend_k,
    merge_back_links_k,
    scatter_back_links_k,
    derive_norms_k,
    rebase_links_k,
    mark_seeds_k,
    reachability_sweep_k,
    reachability_islands_k,
    reached_in_links_k,
    reconnect_island_k,
    merge_attention_k,
    resolve_matches_k,
    merge_matches_k,
    cast_k,
    rmscast_k,
    count_k,
};

/** One instantiated kernel: the kernel and its register rung, zero where it keeps no ladder or
 *  walks the arena frontier. */
struct amd_rung_t {
    amd_kernel_t kernel;
    u32_t slots;
};

/** Every instantiation one dtype's code exports bar the attends, which the device pass instantiates
 *  and the host looks up as it loads the code object. */
inline constexpr amd_rung_t amd_rungs[] = {
    {amd_kernel_t::insert_k, 0},
    {amd_kernel_t::insert_k, 2},
    {amd_kernel_t::insert_k, 4},
    {amd_kernel_t::insert_k, 8},
    {amd_kernel_t::insert_k, 16},
    {amd_kernel_t::refine_k, 0},
    {amd_kernel_t::refine_k, 2},
    {amd_kernel_t::refine_k, 4},
    {amd_kernel_t::refine_k, 8},
    {amd_kernel_t::refine_k, 16},
    {amd_kernel_t::search_k, 0},
    {amd_kernel_t::search_k, 2},
    {amd_kernel_t::search_k, 4},
    {amd_kernel_t::search_k, 8},
    {amd_kernel_t::search_k, 16},
    {amd_kernel_t::search_indexer_k, 0},
    {amd_kernel_t::search_indexer_k, 2},
    {amd_kernel_t::search_indexer_k, 4},
    {amd_kernel_t::search_indexer_k, 8},
    {amd_kernel_t::search_indexer_k, 16},
    {amd_kernel_t::multiply_k, 0},
    {amd_kernel_t::multiply_k, 2},
    {amd_kernel_t::multiply_k, 4},
    {amd_kernel_t::multiply_k, 8},
    {amd_kernel_t::multiply_k, 16},
    {amd_kernel_t::merge_back_links_k, 0},
    {amd_kernel_t::scatter_back_links_k, 0},
    {amd_kernel_t::derive_norms_k, 0},
    {amd_kernel_t::rebase_links_k, 0},
    {amd_kernel_t::mark_seeds_k, 0},
    {amd_kernel_t::reachability_sweep_k, 0},
    {amd_kernel_t::reachability_islands_k, 0},
    {amd_kernel_t::reached_in_links_k, 0},
    {amd_kernel_t::reconnect_island_k, 0},
    {amd_kernel_t::merge_attention_k, 0},
    {amd_kernel_t::resolve_matches_k, 0},
    {amd_kernel_t::merge_matches_k, 0},
    {amd_kernel_t::cast_k, 0},
    {amd_kernel_t::rmscast_k, 0},
};

/** The rungs every walk family, attends included, is instantiated over. */
inline constexpr u32_t amd_walk_rungs[] = {0, 2, 4, 8, 16};

/** Rungs per kernel: the arena frontier's zero, then 2, 4, 8 and 16 slots. */
inline constexpr std::size_t amd_rungs_per_kernel_k = 5;

constexpr std::size_t amd_rung_index(u32_t slots) noexcept {
    return slots == 0 ? 0 : slots == 2 ? 1 : slots == 4 ? 2 : slots == 8 ? 3 : 4;
}

/** The AMD class a capability ceiling names. */
constexpr gpu_class_t amd_class_of(nk_capability_t capability) noexcept {
    return capability & nk_cap_cdna5_k   ? gpu_class_t::amd_cdna5_k
           : capability & nk_cap_cdna4_k ? gpu_class_t::amd_cdna4_k
                                         : gpu_class_t::amd_cdna3_k;
}

/** The ceiling a class's kits are chosen under, as @c numkong_capabilities_of grants it. */
constexpr nk_capability_t amd_capabilities_of(gpu_class_t gpu_class) noexcept {
    nk_capability_t const cdna3 = nk_cap_rocm_k | nk_cap_cdna3_k, cdna4 = cdna3 | nk_cap_cdna4_k;
    return gpu_class == gpu_class_t::amd_cdna5_k   ? cdna4 | nk_cap_cdna5_k
           : gpu_class == gpu_class_t::amd_cdna4_k ? cdna4
                                                   : cdna3;
}

inline constexpr std::size_t amd_kernel_name_capacity_k = 160;

/** Spells the Itanium mangled name of @c amd_kernel<kernel,class,dtype,slots,lookup>: five unsigned
 *  template arguments, then the one @ref amd_arguments_t parameter, as the device pass emits it;
 *  @p lookup is zero for every kernel but an attend. */
inline void amd_kernel_name(char (&name)[amd_kernel_name_capacity_k], amd_rung_t rung, gpu_class_t gpu_class,
                            nk_dtype_t dtype, u32_t lookup = 0) noexcept {
    std::snprintf(name, sizeof(name),
                  "_ZN4unum7usearch5dense10amd_kernelILj%uELj%uELj%uELj%uELj%uEEEvNS1_15amd_arguments_tE",
                  static_cast<unsigned>(rung.kernel), static_cast<unsigned>(gpu_class), static_cast<unsigned>(dtype),
                  static_cast<unsigned>(rung.slots), static_cast<unsigned>(lookup));
}

/** One class's code object, an offload bundle of its listed processors carrying every compiled
 *  dtype, defined in that class's host unit. */
template <nk_capability_t capability_>
bytes_view_t amd_code_object() noexcept;

#pragma endregion Kernel Names

template <nk_dtype_t, nk_capability_t, typename, typename, typename>
struct vram_amd_index;

/** One stream's device-side scratch for driving an AMD index: the traversal arena, the batch
 *  staging, the telemetry counters and the attend miss word, riding the stream its allocator names
 *  and never opening one. Two contexts on one index share only the storage and the published
 *  counter, never scratch. */
template <hip_allocator allocator_type_ = hip_device_allocator_t>
struct hip_traversal_context {
    using allocator_t = allocator_type_;

    /** The single-query verb's landing, grown to the walk's breadth in pinned pages: the ascending
     *  matches, then, past them, the IDs and distances the device copies back. */
    struct results_t {
        growing_buffer<scored_node_t, hip_pinned_allocator_t> entries_ {hip_pinned_allocator_t {}};
        std::size_t count_ = 0;

        scored_node_t* data() noexcept { return entries_.data(); }
        scored_node_t const* data() const noexcept { return entries_.data(); }
        scored_node_t& operator[](std::size_t rank) noexcept { return entries_[rank]; }
        scored_node_t const& operator[](std::size_t rank) const noexcept { return entries_[rank]; }
        std::size_t size() const noexcept { return count_; }
    };

    results_t results;

    /** Binds this context to @p allocator's stream and device, selecting that device first, since
     *  the event is born on the current device, allocates only the fixed-size counters, and takes
     *  their host twins from the stream's pinned landings; sized buffers wait for a verb. */
    hip_traversal_context(allocator_t allocator = {}) noexcept
        : allocator_ {(hip_set_device(allocator.device()), allocator)}, //
          telemetry_ {hip_make_unique(telemetry_allocator_t {allocator_}, 1)},
          assigned_rows_ {hip_make_unique(counter_allocator_t {allocator_}, 1)},
          missed_ {hip_make_unique(counter_allocator_t {allocator_}, 1)},
          host_counters_ {take_counters_(allocator_.landings())},
          arena_ {nullptr, hip_deleter<allocator_t> {allocator_, 0}},
          staging_ {nullptr, hip_deleter<allocator_t> {allocator_, 0}} {
        if (telemetry_) hip_api().fill_async(telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream());
        if (missed_) hip_api().fill_async(missed_.get(), 0, sizeof(u32_t), stream());
    }

    hip_traversal_context(hip_traversal_context const&) = delete;
    hip_traversal_context& operator=(hip_traversal_context const&) = delete;

    /** Every owned resource frees in stream order, so a move is a member-wise steal and destruction
     *  needs no drain. */
    hip_traversal_context(hip_traversal_context&&) noexcept = default;
    ~hip_traversal_context() noexcept = default;

    hip_stream_handle_t stream() const noexcept { return allocator_.stream(); }
    int device() const noexcept { return allocator_.device(); }

    /** Whether this context can walk at all: every device-side word and its host twin claimed, and
     *  its event born. */
    explicit operator bool() const noexcept {
        return telemetry_ && assigned_rows_ && missed_ && host_counters_ && event_;
    }

    /** Waits for every transfer and launch queued on this context's stream - one stream's
     *  completion, the executor's @c drain being the fleet's - sleeping on the context's event
     *  instead of spinning a core. Selects the context's device first, since an event records only
     *  on its own device. */
    status_t drain() noexcept {
        if (hip_set_device(device()) != hip_success_k) return status_t::device_error_k;
        if (status_t const recorded = record(); failed(recorded)) return recorded;
        return event_.drain();
    }

    /** Marks everything queued on this context's stream so far as the submission @ref landed
     *  polls, behind the copies of the counters @ref spent and @ref missed read; the context's
     *  device must be current. */
    status_t record() noexcept {
        hip_api_t const& runtime = hip_api();
        if (!host_counters_ ||
            runtime.copy_async(&host_counters_->telemetry, telemetry_.get(), sizeof(vram_telemetry_block_t),
                               hip_copy_device_to_host_k, stream()) != hip_success_k ||
            runtime.copy_async(&host_counters_->missed, missed_.get(), sizeof(u32_t), hip_copy_device_to_host_k,
                               stream()) != hip_success_k)
            return status_t::device_error_k;
        return event_.record(stream());
    }

    /** Polls whether the submission @ref record marked finished - @c success_k once it has,
     *  @c deferred_k while any of it runs - without blocking, so one host thread awaits whichever
     *  of several streams lands first. */
    status_t landed() const noexcept { return event_.landed(); }

    /** Queues on this context's stream a wait for everything @p other marked last, so work here
     *  follows work there without the host waiting; the two may sit on different devices. */
    status_t follow(hip_traversal_context const& other) noexcept {
        return hip_api().stream_wait_event(stream(), other.event_, 0) == hip_success_k ? status_t::success_k
                                                                                       : status_t::device_error_k;
    }

    /** The telemetry of the last submission, zeroed when it was queued; read once it landed. */
    vram_telemetry_block_t spent() const noexcept {
        return host_counters_ ? host_counters_->telemetry : vram_telemetry_block_t {};
    }

    /** Whether the last attend met a node its value lookup had no row for, which refuses it; read
     *  once it landed. */
    bool missed() const noexcept { return host_counters_ && host_counters_->missed != 0; }

  private:

    /** Only the index's verbs reach the raw scratch - callers observe a context through @c results,
     *  @ref spent and @ref missed, never through its buffers. */
    template <nk_dtype_t, nk_capability_t, typename, typename, typename>
    friend struct vram_amd_index;

    using telemetry_allocator_t = typename allocator_t::template rebind<vram_telemetry_block_t>::other;
    using counter_allocator_t = typename allocator_t::template rebind<u32_t>::other;
    using bytes_ptr_t = hip_unique_ptr<byte_t, allocator_t>;

    /** The host's twins of the device counters, landed behind each @ref record. */
    struct host_counters_t {
        vram_telemetry_block_t telemetry;
        u32_t missed;
    };
    static_assert(sizeof(host_counters_t) <= hip_landings_t::slot_bytes_k &&
                      alignof(host_counters_t) <= hip_landings_t::slot_bytes_k,
                  "A context's counters fit one landing slot");

    /** Gives a context's counters back to the landings of the stream they came from. */
    struct counters_return_t {
        hip_landings_t* landings = nullptr;
        void operator()(host_counters_t* counters) const noexcept { landings->give_back(counters); }
    };
    using host_counters_ptr_t = std::unique_ptr<host_counters_t, counters_return_t>;

    /** Zeroed counters in a slot of @p landings, or null when there are none or it cannot pin. */
    static host_counters_ptr_t take_counters_(hip_landings_t* landings) noexcept {
        void* const slot = landings ? landings->take() : nullptr;
        return {slot ? new (slot) host_counters_t {} : nullptr, counters_return_t {landings}};
    }

    allocator_t allocator_;

    /** Monotone epoch floor over the arena's visited slots, so the arena is zeroed only when
     *  freshly allocated or when the 32-bit epoch space wraps, never per launch. */
    u64_t epoch_floor_ = 0;

    /** The device-side counters of the last submission, zeroed when it was queued. */
    hip_unique_ptr<vram_telemetry_block_t, telemetry_allocator_t> telemetry_;

    /** Rows of the current launch's batch warps have assigned themselves so far: a builder's rows,
     *  or a walk's queries. */
    hip_unique_ptr<u32_t, counter_allocator_t> assigned_rows_;

    /** Raised by the last attend's walks at a node their value lookup had no row for, zeroed when
     *  it was queued. */
    hip_unique_ptr<u32_t, counter_allocator_t> missed_;

    /** Borrowed from the stream's page-locked landings, so the copies @ref record queues into it
     *  never wait on the device, and no context pins memory of its own. */
    host_counters_ptr_t host_counters_;
    bytes_ptr_t arena_;
    std::size_t arena_bytes_ = 0;
    bytes_ptr_t staging_;
    std::size_t staging_bytes_ = 0;

    /** The point on the stream @ref drain and @ref landed await. */
    hip_event_t event_;

    /** Grow-only traversal arena for @p bytes: the exact visited sets and spill tails of every
     *  resident warp, and a builder's scratch when shared memory cannot seat it, kept across
     *  calls. @return The base, or null on exhaustion. */
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
 *  @brief Vector Search index for a single uniform latency memory space on AMD GPUs.
 *
 *  Speaks the RAM-resident dense index's verbs, with two divergences: a batch's IDs are one
 *  contiguous range, and its rows publish unordered on the device, turning visible to searches a
 *  whole batch at a time; see @ref dense_vram_amd_index_concurrency. The first successful
 *  @c reserve freezes the geometry and the virtual-address ceiling; physical slabs map on demand,
 *  and fresh neighbors are sentinel-filled before any ID covering them can be assigned. The storage
 *  grows only under the one claim the claim word admits at a time, and the claim word, the filled
 *  extent, the watermark and the kept status are the only mutable host state; the published-row
 *  count is device memory no host thread dereferences.
 */
template <                                                                     //
    nk_dtype_t vector_dtype_ = nk_i8_k,                                        //
    nk_capability_t capability_ = nk_cap_rocm_k | nk_cap_cdna3_k,              //
    typename vectors_allocator_type_ = hip_virtual_memory_allocator<byte_t>,   //
    typename neighbors_allocator_type_ = hip_virtual_memory_allocator<byte_t>, //
    typename traversals_allocator_type_ = hip_arena_allocator_t>               //
struct vram_amd_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    using vectors_allocator_t = vectors_allocator_type_;
    using neighbors_allocator_t = neighbors_allocator_type_;
    using traversals_allocator_t = traversals_allocator_type_;
    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using dot_result_t = typename scalar_t::dot_result_t;

    /** Every AMD kit accumulates the scalar's own dot type, which the norms column stores. */
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
    using bounce_t = growing_buffer<byte_t, hip_pinned_allocator_t>;

    /** Rows one checkpoint batch carries between a matrix and the caller: one slab, the row
     *  granularity the RAM residency grows by. */
    inline static constexpr std::size_t batch_rows_log2_k = 20;
    using traversal_context_t = hip_traversal_context<traversals_allocator_t>;

    /** The strongest @c synchronization_t contract this design honors; nothing routes on it.
     *  Batches claim disjoint ID ranges and the publication protocol makes overlapping searches
     *  safe. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;

    /** The class's ceiling, surfaced in the bench reports; the device pass picks each kit within
     *  it. */
    inline static constexpr nk_capability_t picked_quads_capability_k = capability_;

    /** The 32-bit ID space, less the sentinel, which also marks a closed claim word. */
    inline static constexpr std::size_t max_node_count_k = (u64_t(1) << 32) - 2u;

    /** Default visited-set slot count; the scratch slice must also fit the candidate cache. */
    inline static constexpr u32_t default_visited_capacity_k = 2048;

    /** Source rows one import window carries at most, so rows in another scalar cross through
     *  bounded staging. */
    inline static constexpr std::size_t import_window_rows_k = 4096;

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
    explicit vram_amd_index(nk_kernel_kind_t kernel, vectors_allocator_t const& vectors_allocator = {},
                            neighbors_allocator_t const& neighbors_allocator = {}) noexcept
        : device_ {vectors_allocator.device()}, kernel_ {kernel}, vectors_matrix_(vectors_allocator),
          norms_column_(vectors_allocator), neighbors_matrix_(neighbors_allocator),
          grown_((hip_set_device(vectors_allocator.device()), hip_event_t {})) {}
    vram_amd_index(vram_amd_index const&) = delete;
    vram_amd_index& operator=(vram_amd_index const&) = delete;

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
            standard_atomic_ref<packed_geometry_id_t const>(geometry_).load(std::memory_order_acquire));
    }
    std::size_t dimensions() const noexcept {
        return geometry_dimensions(
            standard_atomic_ref<packed_geometry_id_t const>(geometry_).load(std::memory_order_acquire));
    }

    /** The walk breadth @p config asks for, or twice the connectivity when it asks for none. */
    template <typename config_type_>
    small_size_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor : connectivity() * 2;
    }

    /** The pinned device's identity and limits, queried once at the first @c reserve. */
    hip_device_specs_t const& device_specs() const noexcept { return device_; }

    /** The metric every walk scores by, fixed at construction. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Whether rows keep norms: every metric but dot finalizes through them, so dot leaves the
     *  norms column empty and checkpoints carry none. */
    bool stores_norms() const noexcept { return kernel_ != nk_kernel_dot_k; }

    /** Device-side flat storage views - the audit surface for invariant checkers & tests. */
    amd_flat_storage_t storage() const noexcept {
        u32_t const fragments = row_fragments();
        return {vectors_rows_(),  norms_column_.base(),
                neighbor_rows_(), static_cast<u32_t>(connectivity()),
                fragments,        lanes_per_row(),
                steps_per_row(),  static_cast<u32_t>(((u64_t {1} << 31) + fragments - 1) / fragments),
                kernel_};
    }

    /** Bytes of one packed logical row, bit-exact for sub-byte scalars. */
    std::size_t packed_row_bytes() const noexcept {
        return divide_round_up<std::size_t>(dimensions() * scalar_t::bits_per_dimension(), 8u);
    }

    /** Fragments per vector row, derived from the frozen geometry like every row measure. */
    u32_t row_fragments() const noexcept {
        return divide_round_up<u32_t>(static_cast<u32_t>(packed_row_bytes()), amd_fragment_bytes_k);
    }

    /** The vector row's byte length, rounded up to whole fragments; the tail bytes stay zero so
     *  every dot-derived metric is unaffected. */
    std::size_t row_bytes() const noexcept { return row_fragments() * amd_fragment_bytes_k; }

    /** Lanes cooperating on one row: the smallest power of two covering its fragments, at most a
     *  warp. */
    u32_t lanes_per_row() const noexcept { return min_of(std::bit_ceil(row_fragments()), wave_lanes()); }

    /** Lanes of this class's wavefronts: 32 on CDNA5, 64 before it. */
    u32_t wave_lanes() const noexcept { return capability_ & nk_cap_cdna5_k ? 32u : 64u; }

    /** Threads per block, four wavefronts. */
    u32_t block_threads() const noexcept { return vram_warps_per_block_k * wave_lanes(); }

    /** Steps those lanes take across one row. */
    u32_t steps_per_row() const noexcept { return divide_round_up(row_fragments(), lanes_per_row()); }

    /** Bytes of one warp's dynamic-shared slice on the insertion path: the query span while
     *  traversing, then the candidate cache & re-selection spans. */
    u32_t insert_scratch_bytes_per_warp() const noexcept {
        u32_t const selection_slots = min_of(static_cast<u32_t>(connectivity()), vram_cache_rows_k);
        u32_t const selection_bytes = selection_slots * (amd_scored_node_bytes_k + 2 * sizeof(u32_t)) +
                                      vram_cache_rows_k * sizeof(f32_t);
        u32_t const cache_bytes = static_cast<u32_t>(row_bytes()) * vram_cache_rows_k;
        return max_of(static_cast<u32_t>(row_bytes()), cache_bytes + selection_bytes);
    }

    /** Bytes of one warp's dynamic-shared slice on the refinement merge path: the candidate
     *  cache - whose last row hosts the node's own vector - then the scored span and the
     *  kept-row indices. */
    u32_t merge_scratch_bytes_per_warp() const noexcept {
        u32_t const cache_bytes = static_cast<u32_t>(row_bytes()) * vram_cache_rows_k;
        return cache_bytes + vram_cache_rows_k * (amd_scored_node_bytes_k + sizeof(u32_t));
    }

    /** Whether a search, multiply or attend launch seats on this device: per warp, the query span
     *  and the @p fold_bytes an attend keeps past it, beside the static batch buffers. */
    bool walk_scratch_fits(std::size_t fold_bytes = 0) const noexcept {
        return std::size_t {vram_warps_per_block_k} * (row_bytes() + fold_bytes + amd_gathered_bytes(wave_lanes())) <=
               device_.shared_memory_per_block_bytes;
    }

    /** Whether an insert or refine launch seats its per-warp scratch in shared memory beside the
     *  static norms, batch buffers and schedule state; when it cannot, the warp's arena slice
     *  keeps it, as Apple's always does. */
    bool build_scratch_fits() const noexcept {
        std::size_t const statics = std::size_t {vram_warps_per_block_k} *
                                    (vram_cache_rows_k * sizeof(norms_t) + amd_gathered_bytes(wave_lanes()) +
                                     amd_schedule_bytes_k);
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

    /** Bytes this index occupies - the committed device storage, the published-row count among
     *  it, and the host struct; traversal scratch belongs to executors and contexts. */
    std::size_t memory_usage() const noexcept {
        return sizeof(vram_amd_index) + vectors_matrix_.committed_bytes() + norms_column_.committed_bytes() +
               neighbors_matrix_.committed_bytes();
    }

    /** Queues the copy of the stored row of @p node into @p destination with the fragment padding
     *  stripped on the caller's @p stream, whose device must be current; the caller waits once,
     *  after its last row. The bulk export is @c save. */
    status_t vector(node_id_t node, nk::vector_span<scalar_t> destination, hip_stream_handle_t stream) const noexcept {
        if (node >= size()) return status_t::config_mismatch_k;
        if (destination.size() != dimensions()) return status_t::config_mismatch_k;
        if (!destination.is_contiguous()) return status_t::unsupported_layout_k;
        std::size_t const scalar_row_bytes = packed_row_bytes();
        return hip_api().copy_async(destination.byte_data(), vectors_rows_() + node * row_bytes(), scalar_row_bytes,
                                    hip_copy_device_to_host_k, stream) == hip_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

#pragma endregion Measures

#pragma region Reservation

    /** Which frontier a warp's arena slice serves: the register ladder needs only the exact visited
     *  set, the wide arena frontier adds its spill tail. */
    enum class frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

    /** One warp's arena slice for a walk expanding @p expansion candidates: the exact visited
     *  set, sized well past the expected visit count so probing stays short, plus the spill tail
     *  when @p frontier_kind outgrows registers - a row of neighbors per expansion, the most it
     *  spills. */
    auto arena_shape_for(u32_t expansion, frontier_kind_t frontier_kind) const noexcept {
        return amd_arena_shape(expansion, frontier_kind == frontier_kind_t::wide_arena_k
                                              ? static_cast<u32_t>(connectivity()) * expansion
                                              : 0);
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
            if (hip_set_device(device_.ordinal) != hip_success_k) return {0, status_t::device_error_k};
            auto const [specs, status] = hip_device_specs_t::query(device_.ordinal);
            if (failed(status)) return {0, status};
            device_ = specs;
            // Geometry no walk can seat is refused here; a builder whose scratch shared memory
            // cannot seat keeps it in its arena slice.
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
            if constexpr (vectors_allocator_t::address_space_k == hip_address_space_t::managed_k)
                budget_bytes = vectors_allocator_t::budget_bytes();
            std::size_t const ceiling_rows = min_of(max_node_count_k,
                                                    max_of<std::size_t>(config.capacity, budget_bytes / bytes_per_row));
            // One more vectors row ahead of row zero holds the published-row count.
            if (vectors_matrix_.reserve_range((ceiling_rows + 1) * row_bytes()) != status_t::success_k ||
                (normed && norms_column_.reserve_range(ceiling_rows * sizeof(norms_t)) != status_t::success_k) ||
                neighbors_matrix_.reserve_range(ceiling_rows * connectivity * sizeof(u32_t)) != status_t::success_k)
                return {0, status_t::bad_alloc_k};
        }
        status_t const mapped = map_rows_(config.capacity);
        return {capacity(), mapped};
    }

#pragma endregion Reservation

#pragma region Insertion

    /** Optional tuning knobs; zeros mean "derive from connectivity". */
    struct insert_config_t {

        /** Breadth of the construction candidate pool; rounded up to the register pools of 2 to 16
         *  lanes' worth and walked exactly past them. Larger pools trade build speed for graph
         *  quality. */
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
        auto const arena_shape = build_arena_shape_(config.expansion_factor, pool_slots);
        std::size_t const arena_warps = device_.multiprocessor_count * insert_blocks_per_multiprocessor_k *
                                        vram_warps_per_block_k;
        std::size_t const results_bytes = // A default single query's device landing.
            non_zero_expansion_factor(search_config_t {}) * (sizeof(node_id_t) + sizeof(distance_t));
        return aligned(sizeof(vram_telemetry_block_t)) + 2 * aligned(sizeof(u32_t)) +
               aligned(arena_warps * arena_shape.slice_bytes()) +
               aligned(rows * row_bytes() + import_bytes_(rows, nk_f64_k) + results_bytes);
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
            hip_api().fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), context.stream()) !=
                hip_success_k)
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
        if (status_t const drained = context.drain(); failed(drained))
            return {{inserted.value.new_size, inserted.value.first_id}, fail_(drained)};
        if (status_t const published = publish_(inserted.value.first_id, new_rows.rows); failed(published))
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

        /** Breadth of the candidate pool, walked as given when it covers @c count and widened to
         *  twice @c count otherwise; a single-query walk takes twice the connectivity for zero. */
        small_size_t expansion_factor = 0;

        /** Matches reported per query; a walk past the widest register pool ranks every candidate
         *  its arena frontier kept. */
        small_size_t count = 10;

        /** The caller's mutation contract; every level shares one code path, since the walk is
         *  insert-safe by protocol, so the field only documents intent. */
        synchronization_t synchronization = synchronization_t::concurrent_k;

        /** How rows are reached; only @c walk_k is implemented, and a scan refuses with
         *  @c unsupported_k. */
        traversal_t traversal = traversal_t::walk_k;
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
            config.expansion_factor >= config.count ? config.expansion_factor : config.count * 2,
            non_zero_expansion_factor(config));
        bool const wide = expansion > vram_widest_pool_slots_k * wave_lanes();
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        u32_t const pool_slots = pool_slots_for_(expansion, count_slots > 2 ? count_slots : 2);
        auto const arena_shape = arena_shape_for(
            wide ? expansion : pool_slots * wave_lanes(),
            wide ? frontier_kind_t::wide_arena_k : frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = std::size_t {walk_blocks_(rows)} * vram_warps_per_block_k;
        std::size_t const staging_bytes = rows * row_bytes() + import_bytes_(rows, nk_f64_k) +
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
     *  one of @p shards, which the first shard gathers, then the window importing the heads and
     *  the outputs cast into the widest scalar. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config, std::size_t head_count,
                                       std::size_t shards = 1) const noexcept {
        std::size_t const rows_bytes = rows * (row_bytes() +
                                               shards * (dimensions() * sizeof(f32_t) + sizeof(softmax_state_t)) +
                                               dimensions() * sizeof(f64_t)) +
                                       import_bytes_(rows, nk_f64_k);
        return traversal_memory_usage(head_count ? rows / head_count : rows, config) +
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
        return settle_(submit_search_(queries, result_ids, result_distances, context, config, size()), queries.rows,
                       context, completion);
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
        return settle_(submit_search_(queries, result_ids, result_distances, context, config, size(), heads),
                       queries.rows, context, completion);
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
            submit_multiply_(left, node_ids, products, counts, context, config, size()), left.rows, context,
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
     *  @param[in] queries The query heads, in any scalar NumKong casts as in @c search; row t ·
     *      @p head_count + h is head h of token t.
     *  @param[in] head_count The query heads sharing this key-value head; it must divide the rows,
     *      and their per-wavefront softmax states and accumulator rows must seat in the LDS beside
     *      the query span.
     *  @param[in] value_lookup Device-callable: `value_lookup(node_id)` points at the node's value
     *      row on a 16-byte boundary, as wide as the keys and stored like them, in memory this
     *      device reads; a null row refuses the call with @c shape_mismatch_k. Trivially copyable,
     *      it names itself with a @c device_lookup_k the code object of @c capability_ instantiates
     *      its attend kernels over, through @ref amd_value_lookup_of.
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
        expected<f32_t*> const landed = submit_attend_(queries, head_count, value_lookup, score_scale, context, config,
                                                       1, exported_bytes_(outputs), size());
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

    /** Optional tuning knobs of @c refine(); zeros mean "derive from connectivity". */
    struct refine_config_t {

        /** Breadth of the re-selection pool, rounded up to the register pools of 2 to 16 lanes'
         *  worth and walked exactly past them; zero takes twice the connectivity, like the
         *  CPU's. */
        small_size_t expansion_factor = 0;
    };

    using refine_result_t = dense::refine_result_t;
    using health_result_t = dense::health_result_t;

  private:

    /** Sweeps an audit numbers from one and stops below: every row joins one frontier at most, so
     *  the levels are bounded by the landmarks' eccentricity, capped defensively. */
    inline static constexpr std::size_t audit_sweeps_cap_k = 256;

    /** Device scratch one refine plan carves once and its passes and audits reuse, or the audit
     *  share one @c health carves: the reached set and two frontiers, the levels counter beside
     *  every sweep's expanded counter, the unreached, reconnected and claim counters an audit lands
     *  together, then for a refine the passes' buffers, the unreached rows' IDs and in-link counts,
     *  and the parts' bases. */
    struct refine_scratch_t {
        typename traversal_context_t::bytes_ptr_t bytes;
        hip_bitset_t reached, frontier, next_frontier;

        /** Rows each sweep expanded, at the sweep's number, @b [audit_sweeps_cap_k]; slot zero,
         *  which no sweep numbers, counts the levels that expanded anything. */
        u32_t* expanded = nullptr;
        u32_t* unreached = nullptr;
        u32_t* reconnected = nullptr;

        /** The walk's and the merge's claim counters, the last pass's completion oracle. */
        u32_t* claims = nullptr;

        /** The reverse buckets' fill counters, then the bitset of rows the merge re-selects. */
        u32_t* bucket_counts = nullptr;
        hip_bitset_t touched;
        u32_t* island_ids = nullptr;
        u32_t* in_links = nullptr;

        /** The shadow neighbors the walks land and the reverse buckets, @b [rows,connectivity]
         *  each. */
        u32_t* shadow_neighbors = nullptr;
        u32_t* buckets = nullptr;

        /** The ascending first rows of the parts the walks merge, uploaded once by the plan. */
        u32_t* part_bases = nullptr;
        std::size_t words = 0;
    };

    /** The counters an audit lands in page-locked memory, in the order of the device's: the levels
     *  that expanded, the last sweep's count, which only a sweep short of convergence leaves
     *  nonzero, then the unreached, reconnected and claim counters. */
    struct audit_landing_t {
        u32_t levels;
        u32_t last_expanded;
        u32_t unreached;
        u32_t reconnected;
        u32_t claims[2];
    };

    /** Gives a landing back to the stream's slots it came from. */
    struct landing_return_t {
        hip_landings_t* landings = nullptr;
        void operator()(audit_landing_t* landing) const noexcept { landings->give_back(landing); }
    };
    using audit_landing_ptr_t = std::unique_ptr<audit_landing_t, landing_return_t>;
    static_assert(sizeof(audit_landing_t) <= hip_landings_t::slot_bytes_k);

  public:
    /**
     *  @brief One refinement in flight on this index's device - the CPU's @c refine_plan_t in the
     *      shape one host thread drives over many devices at once: @ref queue a phase, @ref wait
     *      for it, @ref advance past it, and @ref finish once no phase follows.
     *
     *  The phases re-select every node of the planned range, once per round of walks across while
     *  the shard holds several parts, then audit reachability, then heal round by round while the
     *  islands shrink: re-select them, and reconnect whatever stays unreached into a reached pick
     *  or the landmark covering it. A phase queues its launches and the landing of its counters
     *  behind the context's event and never waits; @ref plan_refine made every allocation, so no
     *  phase allocates, and only a device fault fails one. Demands an exclusive caller: the merge
     *  writes live neighbors with plain stores.
     */
    struct refine_plan_t {

        /** An unbound plan over @p scratch - what a refused plan answers with. */
        explicit refine_plan_t(refine_scratch_t&& scratch) noexcept : scratch_(std::move(scratch)) {}
        refine_plan_t(refine_plan_t&&) noexcept = default;
        refine_plan_t& operator=(refine_plan_t&&) noexcept = default;

        /** Queues the current phase on the context's stream, then the landing of its counters and
         *  telemetry behind the context's event; never waits and never allocates. */
        status_t queue() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (failed(status_)) return status_;
            hip_stream_handle_t const stream = context_->stream();
            switch (phase_) {
            case phase_t::reselect_k: {
                iota_view<node_id_t> const range = round_ ? iota_view<node_id_t> {0, nodes_.bound} : nodes_;
                walked_ = range.size();
                if (walked_) status_ = index_->refine_pass_(*context_, scratch_, config_, range, {}, parts_, round_);
                if (succeeded(status_)) status_ = land_counters_(scratch_, landing_.get(), stream);
                break;
            }
            case phase_t::audit_k:
                walked_ = 0;
                status_ = index_->queue_audit_(*context_, scratch_, landing_.get());
                break;
            case phase_t::heal_k:
                // The island IDs replace the range: offsets below their count index them. Only
                // island rows change, and no reached row points at one, so the audit before stands.
                walked_ = islands_;
                status_ = index_->refine_pass_(*context_, scratch_, config_, {0, islands_},
                                               {scratch_.island_ids, islands_});
                if (succeeded(status_)) status_ = index_->queue_reconnect_(scratch_, islands_, stream);
                if (succeeded(status_)) status_ = index_->queue_audit_(*context_, scratch_, landing_.get());
                break;
            default: return status_t::success_k;
            }
            if (succeeded(status_)) status_ = context_->record();
            return status_;
        }

        /** Sleeps until the queued phase landed, on the context's event. */
        status_t wait() noexcept {
            if (!index_ || phase_ == phase_t::done_k) return status_;
            if (succeeded(status_)) status_ = context_->event_.drain();
            return status_;
        }

        /** The barrier after a phase landed: reads its counters, picks the next phase, and answers
         *  whether one follows. */
        bool advance() noexcept {
            if (!index_ || failed(status_) || phase_ == phase_t::done_k) return false;
            audit_landing_t const& landed = *landing_;
            // The claim counts are the completion oracle: every offset claimed means every row
            // landed.
            if (walked_ && (landed.claims[0] < walked_ || landed.claims[1] < scratch_rows_)) {
                status_ = status_t::device_error_k;
                return false;
            }
            tally_.visited_nodes += walked_;
            if (phase_ == phase_t::reselect_k) {
                if (!stopped_ && ++round_ != rounds_) return true;
                phase_ = phase_t::audit_k;
                return true;
            }
            auto const [health, audit_status] = audited_(landed, scratch_rows_);
            tally_.health = health;
            tally_.islands_reconnected = landed.reconnected;
            // Rows past an unconverged audit are unjudged, not unreached, so nothing heals them.
            if (failed(audit_status)) {
                status_ = audit_status;
                phase_ = phase_t::done_k;
                return false;
            }
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
            if (phase_ == phase_t::reselect_k) stopped_ = true, phase_ = phase_t::audit_k;
        }

        /** Whether the current phase re-selects the planned range - the phase a caller's
         *  progress counts. */
        bool reselecting() const noexcept { return phase_ == phase_t::reselect_k; }

        /** Nodes re-selected so far. */
        std::size_t visited() const noexcept { return tally_.visited_nodes; }

        /** The pass: its telemetry read from the last landing, complete once @ref advance answered
         *  false, otherwise @c cancelled_k; a failure a phase kept answers instead. */
        expected<refine_result_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            refine_result_t result = tally_;
            vram_telemetry_block_t const spent = context_->spent();
            result.changed_nodes = spent.changed_nodes;
            result.visited_members = spent.hops;
            result.computed_distances = spent.scored;
            result.restarts = spent.restarts;
            if (failed(status_)) return {result, status_};
            return {result, phase_ == phase_t::done_k ? status_t::success_k : status_t::cancelled_k};
        }

      private:
        friend struct vram_amd_index;
        enum class phase_t : u8_t { reselect_k, audit_k, heal_k, done_k };

        refine_scratch_t scratch_;
        vram_amd_index* index_ = nullptr;
        traversal_context_t* context_ = nullptr;
        audit_landing_ptr_t landing_ {nullptr, landing_return_t {}};
        refine_config_t config_ {};

        /** The IDs the first round re-selects, and the rows the scratch covers. */
        iota_view<node_id_t> nodes_;
        std::size_t scratch_rows_ = 0;

        /** Parts the rounds merge, the rounds, and the round queued or, once they
         *  ended, finished. */
        u32_t parts_ = 0;
        u32_t rounds_ = 1;
        u32_t round_ = 0;
        bool stopped_ = false;

        /** Rows the queued phase walks, and the islands the last audit left. */
        std::size_t walked_ = 0;
        u32_t islands_ = 0;
        phase_t phase_ = phase_t::done_k;
        status_t status_ = status_t::success_k;
        refine_result_t tally_;
    };

    /**
     *  @brief Opens a refinement of the nodes in @p nodes, clamped to the published IDs, on
     *      @p context 's stream: carves the scratch, grows the context's arena for the widest walk,
     *      takes a page-locked landing and queues the upload of @p part_bases.
     *
     *  No phase allocates. An empty range heals alone.
     *
     *  Two or more @p part_bases below the published IDs split the rows into parts, as a shard
     *  that absorbed others' rows holds them: each node then first walks across the neighboring
     *  group of parts alone, round after round as @c part_across names them, from an empty pool
     *  and landmarks drawn there, before its own neighbors and the whole graph, and re-selects
     *  from the union. The live graph stays as it was through a round's walks, so a walk across
     *  never leaves its group, and the heal's passes walk no part first.
     *
     *  @param[in] nodes The ID range to walk; IDs at or past @c size() are skipped.
     *  @param[inout] context The driving context; its arena grows to the pass's shape.
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
        u32_t const walks = parts ? 2 : 1;
        std::size_t const arena_warps = device_.multiprocessor_count * insert_blocks_per_multiprocessor_k *
                                        vram_warps_per_block_k;
        if (!context.ensure_arena(
                arena_warps *
                build_arena_shape_(expansion * walks, pool_slots_for_(expansion, 4) * walks).slice_bytes()))
            return {std::move(plan), status_t::bad_alloc_k};
        hip_stream_handle_t const stream = context.stream();
        hip_api_t const& runtime = hip_api();
        // Zeroed once, so every pass adds up, and landed with every phase. A pageable upload stages
        // on the host at once and waits on nothing.
        if (runtime.fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream) != hip_success_k ||
            (parts && runtime.copy_async(plan.scratch_.part_bases, part_bases.data(), parts * sizeof(node_id_t),
                                         hip_copy_host_to_device_k, stream) != hip_success_k))
            return {std::move(plan), status_t::device_error_k};
        node_id_t const bound = static_cast<node_id_t>(min_of<std::size_t>(nodes.bound, rows));
        plan.index_ = this;
        plan.context_ = &context;
        plan.config_ = config;
        plan.nodes_ = {min_of(nodes.first, bound), bound};
        plan.scratch_rows_ = rows;
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
            if (status_t const landed = plan.queue(); failed(landed) || failed(plan.wait())) break;
        while (plan.advance());
        return plan.finish();
    }

    /** Measures forward reachability from as many stride-spread landmarks as one search draws -
     *  the connectivity; any shortfall against @c size() is the population no search can
     *  discover. Read-only on the graph, so concurrent insertions may leave fresh nodes
     *  uncounted. Waits once, on the context's event; @c overflow_k when the sweeps reached
     *  their cap unconverged. */
    expected<health_result_t> health(traversal_context_t& context) noexcept {
        std::size_t const rows = size();
        if (!rows) return {{}, status_t::success_k};
        auto [scratch, carve_status] = carve_scratch_(context, scratch_use_t::audit_k);
        if (failed(carve_status)) return {{}, carve_status};
        audit_landing_ptr_t const landing = take_landing_(context);
        if (!landing) return {{}, status_t::bad_alloc_k};
        if (status_t const queued = queue_audit_(context, scratch, landing.get()); failed(queued)) return {{}, queued};
        if (status_t const drained = context.drain(); failed(drained)) return {{}, drained};
        return audited_(*landing, rows);
    }

    /** The bytes an executor's run must hold before a context carved over it can refine the rows
     *  published now under @p config, @p walks walks of a row sharing one visited set: the
     *  context's counters, the walks' arena, and the scratch the refine carves once. */
    std::size_t traversal_memory_usage(refine_config_t config, u32_t walks = 1) const noexcept {
        if (!dimensions() || !device_.multiprocessor_count) return 0;
        auto aligned = [](std::size_t bytes) noexcept {
            return divide_round_up<std::size_t>(bytes, default_alignment_k) * default_alignment_k;
        };
        u32_t const expansion = non_zero_expansion_factor(config);
        auto const arena_shape = build_arena_shape_(expansion * walks, pool_slots_for_(expansion, 4) * walks);
        std::size_t const arena_warps = device_.multiprocessor_count * insert_blocks_per_multiprocessor_k *
                                        vram_warps_per_block_k;
        return aligned(sizeof(vram_telemetry_block_t)) + 2 * aligned(sizeof(u32_t)) +
               aligned(arena_warps * arena_shape.slice_bytes()) + aligned(scratch_bytes_(scratch_use_t::refine_k));
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

        /** What an ID under the watermark moves by - the row this shard starts at. */
        node_id_t id_shift = 0;

        /** The stream every device-to-host copy queues on. */
        hip_stream_handle_t stream = nullptr;

        /** Page-locked host bytes in two halves of @c half_rows rows each: packed vector rows, then
         *  their neighbor IDs, then their norms. */
        bounce_t bounce {hip_pinned_allocator_t {}};

        /** Rows under the watermark at planning. */
        node_id_t visible = 0;

        /** Rows one batch carries - a bounce's worth, or every row when fewer. */
        std::size_t batch_rows = 0;

        /** Rows one half of the bounce carries: half a batch, so two halves pin what one batch
         *  would. */
        std::size_t half_rows = 0;

        /** Rows whose vectors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_vectors = 0;

        /** Rows whose neighbors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_neighbors = 0;

        save_plan_t() noexcept = default;

        /** The index snapshotted into @p columns on @p stream under the watermark at @p visible,
         *  its IDs shifted by @p id_shift; only @ref plan_save calls it, with the index's device
         *  current, where the events are born, and sizes the bounce afterwards. */
        save_plan_t(vram_amd_index const& index, columns_t columns, node_id_t id_shift,
                    hip_executor_t::stream_t const& stream, node_id_t visible) noexcept
            : columns(columns), id_shift(id_shift), stream(stream.handle), bounce(hip_pinned_allocator_t {stream}),
              visible(visible),
              batch_rows(min_of(static_cast<std::size_t>(visible), power_of_two<std::size_t>(batch_rows_log2_k))),
              half_rows(divide_round_up<std::size_t>(batch_rows, 2)), index_(&index) {}
        save_plan_t(save_plan_t&& other) noexcept
            : columns(other.columns), id_shift(other.id_shift), stream(other.stream), bounce(std::move(other.bounce)),
              visible(other.visible), batch_rows(other.batch_rows), half_rows(other.half_rows),
              covered_vectors(other.covered_vectors), covered_neighbors(other.covered_neighbors),
              crossed_ {std::move(other.crossed_[0]), std::move(other.crossed_[1])},
              pending_(std::exchange(other.pending_, {})), pending_half_(other.pending_half_),
              index_(std::exchange(other.index_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        // The tallies are this plan's own: a copy would tally into a twin `finish` never reads.
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;

        /** A save owns nothing but its bounce, so a dying plan leaves the index as it found it,
         *  once the copies into its pending half landed. */
        ~save_plan_t() noexcept {
            if (!pending_.index_rows.empty()) [[maybe_unused]]
                status_t const drained = crossed_[pending_half_].drain();
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
         *  The vectors land with the fragment padding stripped, then the neighbors with every ID
         *  under the watermark shifted by @c id_shift and every other folded to the sentinel, then
         *  the norms where the columns take them. The step's last half is still copying when it
         *  returns, and lands with the next step or in @c finish.
         *
         *  @param[in] step The step to copy, as @ref steps hands them out.
         *  @param[inout] scratch Unused here, since the bounce is already host memory; it rides in
         *      the signature so a caller drives a device plan exactly as it drives a host one.
         */
        status_t save(step_t const& step, nk::matrix_span<node_id_t>) noexcept {
            if (!index_) return status_t::uninitialized_k;
            iota_view<node_id_t> const rows = step.index_rows;
            if (rows.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(node_id_t);
            std::size_t const first = step.source_rows.first;
            if (flat_rows(columns.vectors, first, rows.size(), scalar_row_bytes).empty() ||
                flat_rows(columns.neighbors, first, rows.size(), neighbors_row_bytes).empty() ||
                flat_rows(columns.norms, first, rows.size(), sizeof(norms_t)).empty() != columns.norms.empty())
                return status_t::config_mismatch_k;
            hip_api_t const& runtime = hip_api();
            for (std::size_t chunk = rows.first; chunk < rows.bound; chunk += half_rows) {
                std::size_t const bound = min_of<std::size_t>(chunk + half_rows, rows.bound);
                // Every half but the pending one has landed, so the other one is free.
                std::size_t const half = pending_.index_rows.empty() ? 0 : pending_half_ ^ 1;
                byte_t* const bounce_vectors =
                    bounce.data() + half * half_rows * (scalar_row_bytes + neighbors_row_bytes + sizeof(norms_t));
                byte_t* const bounce_neighbors = bounce_vectors + half_rows * scalar_row_bytes;
                byte_t* const bounce_norms = bounce_neighbors + half_rows * neighbors_row_bytes;
                std::size_t const count = bound - chunk;
                if (runtime.copy_2d_async(bounce_vectors, scalar_row_bytes,
                                          index_->vectors_rows_() + chunk * index_->row_bytes(), index_->row_bytes(),
                                          scalar_row_bytes, count, hip_copy_device_to_host_k,
                                          stream) != hip_success_k ||
                    (!columns.norms.empty() &&
                     runtime.copy_async(bounce_norms, index_->norms_column_.base() + chunk * sizeof(norms_t),
                                        count * sizeof(norms_t), hip_copy_device_to_host_k, stream) != hip_success_k) ||
                    runtime.copy_async(bounce_neighbors, index_->neighbors_matrix_.base() + chunk * neighbors_row_bytes,
                                       count * neighbors_row_bytes, hip_copy_device_to_host_k,
                                       stream) != hip_success_k ||
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
            if (status_t const landed = land_pending_(); failed(landed)) return landed;
            std::size_t const nodes = static_cast<std::size_t>(visible);
            if (standard_atomic_ref<std::size_t> {covered_vectors}.load(std::memory_order_acquire) != nodes ||
                standard_atomic_ref<std::size_t> {covered_neighbors}.load(std::memory_order_acquire) != nodes)
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
            if (status_t const drained = crossed_[pending_half_].drain(); failed(drained)) return drained;
            pending_ = {};
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const connectivity = index_->connectivity();
            std::size_t const neighbors_row_bytes = connectivity * sizeof(node_id_t);
            byte_t* const bounce_vectors =
                bounce.data() + pending_half_ * half_rows * (scalar_row_bytes + neighbors_row_bytes + sizeof(norms_t));
            byte_t* const bounce_neighbors = bounce_vectors + half_rows * scalar_row_bytes;
            byte_t* const bounce_norms = bounce_neighbors + half_rows * neighbors_row_bytes;
            bytes_span_t const vectors_rows = flat_rows(columns.vectors, rows.first, rows.size(), scalar_row_bytes);
            bytes_span_t const neighbors_rows = flat_rows(columns.neighbors, rows.first, rows.size(),
                                                          neighbors_row_bytes);
            bytes_span_t const norms_rows = flat_rows(columns.norms, rows.first, rows.size(), sizeof(norms_t));
            [[maybe_unused]] status_t const copied_vectors = copy_bytes(vectors_rows,
                                                                        bytes_of(bounce_vectors, vectors_rows.size()));
            if (!norms_rows.empty()) [[maybe_unused]]
                status_t const copied_norms = copy_bytes(norms_rows, bytes_of(bounce_norms, norms_rows.size()));
            standard_atomic_ref<std::size_t> {covered_vectors}.fetch_add(rows.size(), std::memory_order_acq_rel);

            std::span<node_id_t> const landed {reinterpret_cast<node_id_t*>(bounce_neighbors),
                                               rows.size() * connectivity};
            for (node_id_t& neighbor : landed)
                neighbor = neighbor < visible ? static_cast<node_id_t>(neighbor + id_shift) : node_id_unknown_k;
            [[maybe_unused]] status_t const copied_neighbors = copy_bytes(neighbors_rows, std::as_bytes(landed));
            standard_atomic_ref<std::size_t> {covered_neighbors}.fetch_add(rows.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Marks the last copies into each half of the bounce, born with the plan. */
        hip_event_t crossed_[2];

        /** The rows whose copies sit in a half not yet landed on the host; empty when none do. */
        step_t pending_ {};

        /** The half @c pending_ sits in. */
        std::size_t pending_half_ = 0;

        /** The index being snapshotted; null in the empty plan a refused @ref plan_save returns,
         *  and once @c finish succeeded. */
        vram_amd_index const* index_ = nullptr;
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
     *  @param[in] id_shift What an ID under the watermark moves by - the row this shard starts at.
     */
    expected<save_plan_t> plan_save(columns_t columns, synchronization_t synchronization,
                                    hip_executor_t::stream_t const& stream, node_id_t id_shift = 0) const noexcept {
        if (synchronization == synchronization_t::concurrent_k) return {{}, status_t::config_mismatch_k};
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        std::size_t const rows = size();
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return {{}, status_t::config_mismatch_k};
        if (hip_set_device(device_.ordinal) != hip_success_k) return {{}, status_t::device_error_k};
        save_plan_t plan {*this, columns, id_shift, stream, static_cast<node_id_t>(rows)};
        std::size_t const bounce_bytes = 2 * plan.half_rows *
                                         (packed_row_bytes() + connectivity() * sizeof(node_id_t) + sizeof(norms_t));
        if (bounce_bytes && failed(plan.bounce.resize_uninitialized(bounce_bytes))) return {{}, status_t::bad_alloc_k};
        return {std::move(plan), status_t::success_k};
    }

    /** Serializes every row under the watermark into @p columns on the calling thread - the
     *  one-shot over @ref plan_save for a caller with no streams to fan the steps out over. */
    status_t save(columns_t columns, synchronization_t synchronization, nk::matrix_span<node_id_t> scratch,
                  hip_executor_t::stream_t const& stream) const noexcept {
        auto [plan, status] = plan_save(columns, synchronization, stream);
        if (failed(status)) return status;
        for (typename save_plan_t::step_t const step : plan.steps())
            if (status_t const saved = plan.save(step, scratch); failed(saved)) return saved;
        return plan.finish();
    }

    /**
     *  @brief One load in flight: the rows it appends past every claimed one, the storage grown to
     *      hold them, and the steps they span, each filled from host columns through a
     *      double-buffered pinned bounce or from another index's storage on the stream;
     *      @ref finish derives the norms no step carried and publishes.
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

        /** Where the rows come from, vetted against them at planning. */
        columns_view_t columns {};

        /** The ID space the source's rows speak. */
        iota_view<node_id_t> source_range {};

        /** The stream every copy into the index and every kernel queue on, whose device every step
         *  and @c finish select first. */
        hip_stream_handle_t stream = nullptr;

        /** Page-locked host bytes in two halves of @c half_rows rows each: packed vector rows, then
         *  their neighbor IDs, then their norms, pinned at planning by @ref pin_bounce or never. */
        bounce_t bounce {hip_pinned_allocator_t {}};

        /** The first appended ID - the row the claim landed at. */
        node_id_t base = 0;

        /** Rows the plan appends. */
        std::size_t rows = 0;

        /** Rows one batch carries - a bounce's worth, or every row when fewer. */
        std::size_t batch_rows = 0;

        /** Rows one half of the bounce carries: half a batch, so two halves pin what one batch
         *  would. */
        std::size_t half_rows = 0;

        /** Rows whose vectors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_vectors = 0;

        /** Rows whose neighbors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_neighbors = 0;

        /** Rows whose norms landed from the columns, tallied through @c std::atomic_ref; @c finish
         *  derives every row's when it falls short. */
        std::size_t covered_norms = 0;

        load_plan_t() noexcept = default;

        /** The index appended to at @p base on @p stream, under the claim over its rows or a claim
         *  word its caller closed; only @ref plan_load calls it, with the index's device current,
         *  where the events are born. */
        load_plan_t(vram_amd_index& index, columns_view_t columns, iota_view<node_id_t> source_range,
                    hip_executor_t::stream_t const& stream, node_id_t base, std::size_t rows) noexcept
            : columns(columns), source_range(source_range), stream(stream.handle),
              bounce(hip_pinned_allocator_t {stream}), base(base), rows(rows),
              batch_rows(min_of(rows, power_of_two<std::size_t>(batch_rows_log2_k))),
              half_rows(divide_round_up<std::size_t>(batch_rows, 2)), index_(&index) {}
        load_plan_t(load_plan_t&& other) noexcept
            : columns(other.columns), source_range(other.source_range), stream(other.stream),
              bounce(std::move(other.bounce)), base(other.base), rows(std::exchange(other.rows, 0)),
              batch_rows(other.batch_rows), half_rows(other.half_rows), covered_vectors(other.covered_vectors),
              covered_neighbors(other.covered_neighbors), covered_norms(other.covered_norms),
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
            if (hip_set_device(index_->device_specs().ordinal) == hip_success_k &&
                succeeded(index_->grown_.record(stream)) && chunks_) [[maybe_unused]]
                status_t const drained = index_->grown_.drain();
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
         *  The vectors land in the fragment-padded matrix rows, whose fresh tails arrive zeroed,
         *  and their norms where the columns carry them, then the neighbors with every ID inside
         *  @c source_range folded onto the appended range and every other onto the sentinel.
         *
         *  @param[in] step The step to fill, as @ref steps hands them out.
         */
        status_t load(step_t const& step) noexcept { return load(step, columns, source_range); }

        /** Pins the bounce at planning, for a plan whose steps cross host memory or another
         *  device's storage; a plan whose rows move between matrices on one device never pins. */
        status_t pin_bounce() noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const row_bytes = index_->packed_row_bytes() + index_->connectivity() * sizeof(node_id_t) +
                                          sizeof(norms_t);
            return bounce.empty() ? bounce.resize_uninitialized(2 * half_rows * row_bytes) : status_t::success_k;
        }

        /**
         *  @brief The same copy out of @p columns under @p source_range rather than the plan's own.
         *
         *  This is what a sharded plan drives, one part's slice at a time, its step naming the rows
         *  in that part's row and ID spaces.
         *
         *  @param[in] step The step to fill; its source rows address @p columns.
         *  @param[in] columns The part's own columns.
         *  @param[in] source_range The ID space the part's rows speak.
         */
        status_t load(step_t const& step, columns_view_t columns, iota_view<node_id_t> source_range) noexcept {
            if (!index_ || bounce.empty()) return status_t::uninitialized_k;
            if (hip_set_device(index_->device_specs().ordinal) != hip_success_k) return status_t::device_error_k;
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
            // The device indexes its buckets by neighbor ID unchecked, so every rebased ID must
            // land in this plan.
            if (static_cast<std::size_t>(shift) + source_range.size() > static_cast<std::size_t>(base) + rows)
                return status_t::config_mismatch_k;
            // Dot reserves no norms column, so a part's column has nowhere to land.
            bool const normed = index_->stores_norms() && !norms_rows.empty();
            hip_api_t const& runtime = hip_api();
            for (std::size_t done = 0; done < range.size(); done += half_rows) {
                std::size_t const count = min_of(half_rows, range.size() - done), into = range.first + done;
                std::size_t const half = chunks_++ % 2;
                byte_t* const bounce_vectors =
                    bounce.data() + half * half_rows * (scalar_row_bytes + neighbors_row_bytes + sizeof(norms_t));
                byte_t* const bounce_neighbors = bounce_vectors + half_rows * scalar_row_bytes;
                byte_t* const bounce_norms = bounce_neighbors + half_rows * neighbors_row_bytes;
                // The host refills a half only once the copies out of it landed, while the other
                // half's copies run.
                if (chunks_ > 2 && failed(crossed_[half].drain())) return index_->fail_(status_t::device_error_k);
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
                if (runtime.copy_2d_async(index_->vectors_rows_() + into * index_->row_bytes(), index_->row_bytes(),
                                          bounce_vectors, scalar_row_bytes, scalar_row_bytes, count,
                                          hip_copy_host_to_device_k, stream) != hip_success_k ||
                    (normed &&
                     runtime.copy_async(index_->norms_column_.base() + into * sizeof(norms_t), bounce_norms,
                                        count * sizeof(norms_t), hip_copy_host_to_device_k, stream) != hip_success_k) ||
                    runtime.copy_async(index_->neighbors_matrix_.base() + into * neighbors_row_bytes, bounce_neighbors,
                                       count * neighbors_row_bytes, hip_copy_host_to_device_k,
                                       stream) != hip_success_k ||
                    failed(crossed_[half].record(stream)))
                    return index_->fail_(status_t::device_error_k);
            }
            if (normed)
                standard_atomic_ref<std::size_t> {covered_norms}.fetch_add(range.size(), std::memory_order_acq_rel);
            standard_atomic_ref<std::size_t> {covered_vectors}.fetch_add(range.size(), std::memory_order_acq_rel);
            standard_atomic_ref<std::size_t> {covered_neighbors}.fetch_add(range.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /**
         *  @brief The same copy out of @p source 's own storage rather than host columns, as an
         *      absorb drives it, one donor shard at a time, its step naming the rows in that
         *      shard's row and ID spaces.
         *
         *  Both matrices keep fragment-padded rows of one geometry, so on one device every column
         *  moves as one copy on the plan's stream. Another device's storage is mapped for that
         *  device alone, so its rows cross through the bounce half a batch at a time, read out on
         *  @p source_stream and written in on the plan's, each stream waiting on the other's event
         *  rather than the host, so one half reads out while the other writes in. The neighbors
         *  then rebase on the plan's stream. Nothing waits, so @p source must stay quiescent until
         *  the caller drains both streams.
         *
         *  @param[in] step The step to fill; its source rows address @p source.
         *  @param[in] source The index the rows come from.
         *  @param[in] source_range The ID space the source's rows speak.
         *  @param[inout] source_stream An executor stream on @p source 's device, which reads the
         *      rows out when the devices differ and marks each half filled on its event.
         */
        status_t load(step_t const& step, vram_amd_index const& source, iota_view<node_id_t> source_range,
                      hip_executor_t::stream_t& source_stream) noexcept {
            if (!index_) return status_t::uninitialized_k;
            int const device = index_->device_specs().ordinal, source_device = source.device_specs().ordinal;
            if (hip_set_device(device) != hip_success_k) return status_t::device_error_k;
            iota_view<node_id_t> const range = step.index_rows;
            if (range.empty() || range.size() != step.source_rows.size() || range.first < base ||
                range.bound > base + rows || step.source_rows.bound > source.size() ||
                source.dimensions() != index_->dimensions() || source.connectivity() != index_->connectivity())
                return status_t::config_mismatch_k;
            node_id_t const shift = range.first - step.source_rows.first + source_range.first;
            if (shift + source_range.size() > base + rows) return status_t::config_mismatch_k;
            std::size_t const row_bytes = index_->row_bytes(), scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(node_id_t);
            std::size_t const count = range.size(), into = range.first, from = step.source_rows.first;
            bool const normed = index_->stores_norms() && source.stores_norms();
            hip_api_t const& runtime = hip_api();
            if (device == source_device) {
                if (runtime.copy_async(index_->vectors_rows_() + into * row_bytes,
                                       source.vectors_rows_() + from * row_bytes, count * row_bytes,
                                       hip_copy_device_to_device_k, stream) != hip_success_k ||
                    runtime.copy_async(index_->neighbors_matrix_.base() + into * neighbors_row_bytes,
                                       source.neighbors_matrix_.base() + from * neighbors_row_bytes,
                                       count * neighbors_row_bytes, hip_copy_device_to_device_k,
                                       stream) != hip_success_k ||
                    (normed &&
                     runtime.copy_async(index_->norms_column_.base() + into * sizeof(norms_t),
                                        source.norms_column_.base() + from * sizeof(norms_t), count * sizeof(norms_t),
                                        hip_copy_device_to_device_k, stream) != hip_success_k))
                    return index_->fail_(status_t::device_error_k);
            }
            else {
                if (bounce.empty()) return status_t::uninitialized_k;
                hip_stream_handle_t const reading = source_stream.handle;
                for (std::size_t done = 0; done < count; done += half_rows) {
                    std::size_t const batch = min_of(half_rows, count - done);
                    std::size_t const read = from + done, written = into + done;
                    std::size_t const half = chunks_++ % 2;
                    byte_t* const bounce_vectors =
                        bounce.data() + half * half_rows * (scalar_row_bytes + neighbors_row_bytes + sizeof(norms_t));
                    byte_t* const bounce_neighbors = bounce_vectors + half_rows * scalar_row_bytes;
                    byte_t* const bounce_norms = bounce_neighbors + half_rows * neighbors_row_bytes;
                    // The source refills a half only once the plan's stream wrote it in; an event
                    // never recorded holds nothing back.
                    if (hip_set_device(source_device) != hip_success_k ||
                        runtime.stream_wait_event(reading, crossed_[half], 0) != hip_success_k ||
                        runtime.copy_2d_async(bounce_vectors, scalar_row_bytes,
                                              source.vectors_rows_() + read * row_bytes, row_bytes, scalar_row_bytes,
                                              batch, hip_copy_device_to_host_k, reading) != hip_success_k ||
                        runtime.copy_async(
                            bounce_neighbors, source.neighbors_matrix_.base() + read * neighbors_row_bytes,
                            batch * neighbors_row_bytes, hip_copy_device_to_host_k, reading) != hip_success_k ||
                        (normed && runtime.copy_async(
                                       bounce_norms, source.norms_column_.base() + read * sizeof(norms_t),
                                       batch * sizeof(norms_t), hip_copy_device_to_host_k, reading) != hip_success_k) ||
                        failed(source_stream.event.record(reading)))
                        return index_->fail_(status_t::device_error_k);
                    if (hip_set_device(device) != hip_success_k ||
                        runtime.stream_wait_event(stream, source_stream.event, 0) != hip_success_k ||
                        runtime.copy_2d_async(index_->vectors_rows_() + written * row_bytes, row_bytes, bounce_vectors,
                                              scalar_row_bytes, scalar_row_bytes, batch, hip_copy_host_to_device_k,
                                              stream) != hip_success_k ||
                        runtime.copy_async(index_->neighbors_matrix_.base() + written * neighbors_row_bytes,
                                           bounce_neighbors, batch * neighbors_row_bytes, hip_copy_host_to_device_k,
                                           stream) != hip_success_k ||
                        (normed && runtime.copy_async(index_->norms_column_.base() + written * sizeof(norms_t),
                                                      bounce_norms, batch * sizeof(norms_t), hip_copy_host_to_device_k,
                                                      stream) != hip_success_k) ||
                        failed(crossed_[half].record(stream)))
                        return index_->fail_(status_t::device_error_k);
                }
            }
            if (failed(index_->launch_rebase_links_(range, source_range, shift, stream)))
                return index_->fail_(status_t::device_error_k);
            if (normed) standard_atomic_ref<std::size_t> {covered_norms}.fetch_add(count, std::memory_order_acq_rel);
            standard_atomic_ref<std::size_t> {covered_vectors}.fetch_add(count, std::memory_order_acq_rel);
            standard_atomic_ref<std::size_t> {covered_neighbors}.fetch_add(count, std::memory_order_acq_rel);
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
            if (hip_set_device(index_->device_specs().ordinal) != hip_success_k) return status_t::device_error_k;
            if (standard_atomic_ref<std::size_t> {covered_vectors}.load(std::memory_order_acquire) != rows ||
                standard_atomic_ref<std::size_t> {covered_neighbors}.load(std::memory_order_acquire) != rows)
                return status_t::config_mismatch_k;
            if (rows) {
                u32_t const blocks = index_->device_.multiprocessor_count * search_blocks_per_multiprocessor_k;
                // One launch derives every row, so a load mixing carried and missing norms
                // re-derives the carried.
                if (index_->stores_norms() &&
                    standard_atomic_ref<std::size_t> {covered_norms}.load(std::memory_order_acquire) != rows &&
                    failed(index_->launch_derive_norms_(static_cast<u32_t>(base), static_cast<u32_t>(rows), blocks,
                                                        stream)))
                    return index_->fail_(status_t::device_error_k);
                // A fill carries the count by value, so no host word must outlive a copy; the one
                // wait lands every step, the norms and the count before the watermark turns.
                if (hip_api().fill_words_async(index_->published_rows_(), static_cast<int>(base + rows), 1, stream) !=
                        hip_success_k ||
                    failed(crossed_[0].record(stream)) || failed(crossed_[0].drain()))
                    return index_->fail_(status_t::device_error_k);
            }
            vram_amd_index* const index = std::exchange(index_, nullptr);
            return rows ? index->publish_(base, rows) : status_t::success_k;
        }

      private:

        /** Marks the last copies out of each half of the bounce, born with the plan. */
        hip_event_t crossed_[2];

        /** Halves of rows the steps moved through the bounce so far, alternating between them. */
        std::size_t chunks_ = 0;

        /** The index appended to; null in the empty plan a refused @ref plan_load returns, and once
         *  @c finish published. */
        vram_amd_index* index_ = nullptr;
    };

    /**
     *  @brief Opens a load of @p rows from @p columns at the watermark on @p stream: claims them,
     *      or lands them under the claim word its caller closed, grows the storage to hold them -
     *      the fresh neighbor rows arrive sentinel-filled - and pins the double-buffered bounce.
     *
     *  Refuses an unreserved index, a kept failure, columns shorter than the rows, a batch or load
     *  in flight, and rows past the ID space.
     *
     *  @param[in] columns Where the rows come from, every column but an empty norms one sized for
     *      @p rows.
     *  @param[in] rows Rows to append.
     *  @param[in] source_range The ID space the source's rows speak.
     *  @param[in] stream The stream every copy and the norms kernel queue on.
     */
    expected<load_plan_t> plan_load(columns_view_t columns, std::size_t rows, iota_view<node_id_t> source_range,
                                    hip_executor_t::stream_t const& stream) noexcept {
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return {{}, status_t::config_mismatch_k};
        auto [plan, status] = plan_load(rows, stream);
        if (failed(status)) return {{}, status};
        if (status_t const pinned = plan.pin_bounce(); failed(pinned)) return {{}, pinned};
        plan.columns = columns;
        plan.source_range = source_range;
        return {std::move(plan), status};
    }

    /** Opens a load of @p rows the caller sources per step - what a sharded plan drives, its parts
     *  arriving one slice at a time rather than as one pair of columns, and what an absorb drives,
     *  its parts read out of other indexes. */
    expected<load_plan_t> plan_load(std::size_t rows, hip_executor_t::stream_t const& stream) noexcept {
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (hip_set_device(device_.ordinal) != hip_success_k) return {{}, status_t::device_error_k};
        if (!rows)
            return {load_plan_t {*this, {}, {}, stream, static_cast<node_id_t>(claimed_()), 0}, status_t::success_k};
        // A closed word is the caller's, who refines or absorbs around the load; an open one is
        // claimed like a batch's, so no batch lands beside it.
        bool const claiming = !claims_.closed();
        node_id_t base = static_cast<node_id_t>(size());
        if (claiming) {
            auto const [first_id, claim_status] = claim_(rows);
            if (failed(claim_status)) return {{}, claim_status};
            base = first_id;
        }
        else if (rows > max_node_count_k - base) return {{}, status_t::overflow_k};
        load_plan_t plan {*this, {}, {}, stream, base, rows};
        if (status_t const grown = grow_rows_(base + rows, stream.handle); failed(grown)) return {{}, grown};
        return {std::move(plan), status_t::success_k};
    }

    /** Appends the whole source on the calling thread - the one-shot over @ref plan_load for a
     *  caller with no streams to fan the steps out over. */
    status_t load(columns_view_t columns, std::size_t rows, iota_view<node_id_t> source_range,
                  hip_executor_t::stream_t const& stream) noexcept {
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
    friend struct vram_amd_indexes;
    template <typename, typename, typename, typename, typename, template <typename> class>
    friend struct sharded_indexes;

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
        hip_stream_handle_t const stream = context.stream();
        std::size_t const import_bytes = import_bytes_(rows, new_rows.dtype);
        byte_t* const staging = context.ensure_staging(rows * row_bytes() + import_bytes);
        if (!staging) return {{}, status_t::bad_alloc_k};
        if ((import_bytes || packed_row_bytes() != row_bytes()) &&
            hip_api().fill_async(staging, 0, rows * row_bytes() + import_bytes, stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        if (status_t const imported = import_rows_(staging, new_rows, staging + rows * row_bytes(), stream);
            failed(imported))
            return {{}, imported};

        // Every inserting warp owns an arena slice for its exact visited set.
        u32_t const pool_slots = pool_slots_for_(config.expansion_factor, 4);
        auto const arena_shape = build_arena_shape_(config.expansion_factor, pool_slots);
        std::size_t const arena_warps = device_.multiprocessor_count * insert_blocks_per_multiprocessor_k *
                                        vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {{}, status_t::bad_alloc_k};

        // The claim admits no other grower, and a rejected grid publishes nothing, so the caller
        // un-claims it exactly.
        std::size_t const total_rows = first_id + rows;
        status_t claimed = grow_rows_(total_rows, stream);
        if (claimed == status_t::success_k) {
            // Only a fresh allocation or a wrapped 32-bit epoch space warrants zeroing, never a
            // warm arena.
            if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + rows > 0xFFFFFFFEull) {
                if (hip_api().fill_async(arena, 0, context.arena_bytes_, stream) != hip_success_k) {
                    claimed = status_t::device_error_k;
                    context.arena_bytes_ = 0; // Spent: a retry must re-zero, never walk epoch garbage.
                }
                context.epoch_floor_ = 0;
            }
            u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
            context.epoch_floor_ += rows; // Monotone; an un-claim below only forfeits epoch space.
            amd_build_t const build {storage(),
                                     staging,
                                     static_cast<u32_t>(total_rows),
                                     arena_shape.cache_bytes ? 0u : insert_scratch_bytes_per_warp(),
                                     first_id,
                                     arena,
                                     arena_shape,
                                     config.expansion_factor,
                                     epoch_base,
                                     published_rows_(),
                                     context.telemetry_.get(),
                                     context.assigned_rows_.get()};
            // The launch is the last fallible step; reading its verdict here keeps the claim
            // reversible.
            if (claimed == status_t::success_k && failed(launch_insert_(build, pool_slots, stream)))
                claimed = status_t::device_error_k;
        }
        if (claimed != status_t::success_k) return {{}, claimed};

        insert_result_t result;
        result.first_id = first_id;
        for (std::size_t row = 0; row != assigned_ids.size(); ++row)
            assigned_ids[row] = static_cast<node_id_t>(first_id + row);
        result.new_size = total_rows;
        return {result, status_t::success_k};
    }

#pragma endregion Insertion

#pragma region Search

    /** What every walking verb stages ahead of its sink: the device view, the frontier the walks
     *  keep, the staging past the uploads, and the rescoring the walks score under. */
    template <typename rescoring_type_>
    struct walk_launch {

        /** The device view every walk kernel reads. */
        amd_search_t view;

        /** The register ladder's width, when the ladder serves. */
        u32_t pool_slots;

        /** Which frontier the walks keep. */
        frontier_kind_t frontier;

        /** The staging bytes past the queries and any heads, where the sink writes its outputs. */
        byte_t* outputs;

        /** The rescoring of every walk, over the heads staged beside the queries. */
        rescoring_type_ rescoring;
    };

    /** The grouped-query heads an attend stages beside the rows its walks navigate by: every head
     *  of every token, @c head_count consecutive rows per token. */
    struct attend_heads_t {
        rows_view_t rows;
        std::size_t head_count;
    };

    /** The device rescoring the walks under @p heads_type_ score with. */
    template <typename heads_type_>
    using walk_rescoring = amd_rescoring_t;

    /** Validates and uploads @p queries, then any indexer or attend @p heads and the indexer's
     *  weights, sizes the arena for @p expansion, and reserves @p outputs_bytes of staging past
     *  them: everything a walking verb stages before its sink, the walks bound below the @p visible
     *  rows. Until the context drains, the queries, the heads and the context stay live. */
    template <typename heads_type_>
    expected<walk_launch<walk_rescoring<heads_type_>>> stage_walk_(rows_view_t queries,
                                                                   [[maybe_unused]] heads_type_ const& heads,
                                                                   traversal_context_t& context, u32_t expansion,
                                                                   u32_t minimum_slots, std::size_t outputs_bytes,
                                                                   std::size_t visible) noexcept {
        constexpr bool indexer_k = std::is_same_v<heads_type_, indexer_heads<scalar_t>>;
        constexpr bool attend_k = std::is_same_v<heads_type_, attend_heads_t>;
        std::size_t const queries_count = queries.rows;
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        // Validated with the rest: past here a refusal would already have burned epoch space.
        if (status_t const refusal = rows_refusal_(queries); failed(refusal)) return {{}, refusal};
        std::size_t heads_rows = 0;
        if constexpr (indexer_k) {
            heads_rows = heads.heads.extent(0);
            if (heads.weights.extent(0) != queries_count || heads_rows != queries_count * heads.weights.extent(1))
                return {{}, status_t::config_mismatch_k};
            if (status_t const refusal = rows_refusal_(heads.heads); failed(refusal)) return {{}, refusal};
        }
        if constexpr (attend_k) {
            heads_rows = heads.rows.rows;
            if (status_t const refusal = rows_refusal_(heads.rows); failed(refusal)) return {{}, refusal};
        }
        hip_stream_handle_t const stream = context.stream();

        // Past the register ladder's widest pool the arena-backed walk serves.
        frontier_kind_t const frontier = expansion > vram_widest_pool_slots_k * wave_lanes()
                                             ? frontier_kind_t::wide_arena_k
                                             : frontier_kind_t::register_ladder_k;
        u32_t const pool_slots = pool_slots_for_(expansion, minimum_slots);
        auto const arena_shape = arena_shape_for(
            frontier == frontier_kind_t::wide_arena_k ? expansion : pool_slots * wave_lanes(), frontier);
        std::size_t const arena_warps = std::size_t {walk_blocks_(queries_count)} * vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {{}, status_t::bad_alloc_k};
        // The staging holds the query rows and any head rows first, since they carry the widest
        // alignment, then the window importing them, the heads' weights, then the outputs; stream
        // order serializes its reuse across calls.
        std::size_t const rows_bytes = (queries_count + heads_rows) * row_bytes();
        std::size_t const import_bytes = import_bytes_(attend_k ? heads_rows : queries_count, queries.dtype);
        std::size_t const weights_bytes = indexer_k ? heads_rows * sizeof(f32_t) : 0;
        byte_t* const staging = context.ensure_staging(rows_bytes + import_bytes + weights_bytes + outputs_bytes);
        if (!staging) return {{}, status_t::bad_alloc_k};
        byte_t* const window = staging + rows_bytes;
        // Only a fresh allocation or a wrapped 32-bit epoch space warrants zeroing, never a warm
        // arena.
        if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + queries_count > 0xFFFFFFFEull) {
            if (hip_api().fill_async(arena, 0, context.arena_bytes_, stream) != hip_success_k) {
                context.arena_bytes_ = 0; // Spent: a retry must re-zero, never walk epoch garbage.
                return {{}, status_t::device_error_k};
            }
            context.epoch_floor_ = 0;
        }
        u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += queries_count;
        // Pitched uploads: device rows pad to whole fragments, and the pad bytes must be zero for
        // the zero-tail distance identity to hold.
        if ((import_bytes || packed_row_bytes() != row_bytes()) &&
            hip_api().fill_async(staging, 0, rows_bytes + import_bytes, stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        if (status_t const imported = import_rows_(staging, queries, window, stream); failed(imported))
            return {{}, imported};
        if (hip_api().fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream) !=
                hip_success_k ||
            hip_api().fill_async(context.assigned_rows_.get(), 0, sizeof(u32_t), stream) != hip_success_k)
            return {{}, status_t::device_error_k};

        walk_rescoring<heads_type_> rescoring {};
        if constexpr (indexer_k) {
            byte_t* const staged_heads = staging + queries_count * row_bytes();
            byte_t* const staged_weights = staging + rows_bytes + import_bytes;
            std::size_t const weights_row_bytes = heads.weights.extent(1) * sizeof(f32_t);
            if (failed(import_rows_(staged_heads, heads.heads, window, stream)) ||
                hip_api().copy_2d_async(staged_weights, weights_row_bytes, heads.weights.byte_data(),
                                        heads.weights.stride_bytes(0), weights_row_bytes, queries_count,
                                        hip_copy_default_k, stream) != hip_success_k)
                return {{}, status_t::device_error_k};
            rescoring = {staged_heads, reinterpret_cast<f32_t const*>(staged_weights),
                         static_cast<u32_t>(heads.weights.extent(1)), row_fragments()};
        }
        if constexpr (attend_k) {
            byte_t* const staged_heads = staging + queries_count * row_bytes();
            if (status_t const imported = import_rows_(staged_heads, heads.rows, window, stream); failed(imported))
                return {{}, imported};
            rescoring = {staged_heads, nullptr, static_cast<u32_t>(heads.head_count), row_fragments()};
        }

        amd_search_t const view {storage(),
                                 staging,
                                 static_cast<u32_t>(queries_count),
                                 static_cast<u32_t>(visible),
                                 static_cast<u32_t>(row_bytes()),
                                 expansion,
                                 arena,
                                 arena_shape,
                                 epoch_base,
                                 context.telemetry_.get(),
                                 context.assigned_rows_.get()};
        return {{view, pool_slots, frontier, staging + rows_bytes + import_bytes + weights_bytes, rescoring},
                status_t::success_k};
    }

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
            amd_fragment_bytes_k);
    }

    /** Imports @p rows into @p staging at the vectors matrix pitch in the stored scalar on
     *  @p stream, over @p staging and @p window already zeroed. Rows in the stored scalar take one
     *  pitched copy. Others cross a window at a time: a pitched copy pads them to the stored pitch,
     *  and one flat cast lands the window in place, its zeroed pad values casting to zero pad
     *  bytes, or a unit-RMS narrowing writes the columns alone. Repeated rows import once and
     *  double on the device. */
    status_t import_rows_(byte_t* staging, rows_view_t rows, byte_t* window, hip_stream_handle_t stream) noexcept {
        hip_api_t const& api = hip_api();
        std::size_t const distinct = rows.stride ? rows.rows : 1;
        std::size_t const source_row_bytes = packed_bytes(rows.dtype, rows.columns);
        std::size_t const source_pitch = rows.stride ? rows.stride : source_row_bytes;
        if (rows.dtype == vector_dtype_) {
            if (api.copy_2d_async(staging, row_bytes(), rows.data, source_pitch, source_row_bytes, distinct,
                                  hip_copy_host_to_device_k, stream) != hip_success_k)
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
                if (api.copy_2d_async(window, window_row_bytes, rows.row(first), source_pitch, source_row_bytes, count,
                                      hip_copy_host_to_device_k, stream) != hip_success_k)
                    return status_t::device_error_k;
                status_t landed = status_t::success_k;
                if (!narrows_at_unit_rms_(rows.dtype))
                    landed = launch_cast_(window, rows.dtype, landing, vector_dtype_, count * pitch_values, stream);
                else if (rows.dtype == nk_f32_k)
                    landed = launch_rmscast_(reinterpret_cast<f32_t const*>(window), window_row_bytes, landing, count,
                                             stream);
                else {
                    landed = launch_cast_(window, nk_f64_k, widened, nk_f32_k, count * pitch_values, stream);
                    if (succeeded(landed))
                        landed = launch_rmscast_(widened, pitch_values * sizeof(f32_t), landing, count, stream);
                }
                if (failed(landed)) return landed;
            }
        }
        for (std::size_t copied = distinct; copied < rows.rows; copied *= 2)
            if (api.copy_async(staging + copied * row_bytes(), staging,
                               min_of(copied, rows.rows - copied) * row_bytes(), hip_copy_device_to_device_k,
                               stream) != hip_success_k)
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
                            hip_stream_handle_t stream) noexcept {
        std::size_t const values = outputs.rows * dimensions();
        std::size_t const exported_row_bytes = packed_bytes(outputs.dtype, dimensions());
        byte_t* exported = reinterpret_cast<byte_t*>(landed);
        if (outputs.dtype != nk_f32_k) {
            exported += blocks * (values * sizeof(f32_t) + outputs.rows * sizeof(softmax_state_t));
            if (status_t const cast = launch_cast_(landed, nk_f32_k, exported, outputs.dtype, values, stream);
                failed(cast))
                return cast;
        }
        hip_api_t const& api = hip_api();
        if (api.copy_2d_async(outputs.data, outputs.stride, exported, exported_row_bytes, exported_row_bytes,
                              outputs.rows, hip_copy_default_k, stream) != hip_success_k ||
            (!carry.empty() && api.copy_async(carry.data(), landed + values, carry.size_bytes(), hip_copy_default_k,
                                              stream) != hip_success_k))
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
    expected<search_result_t> submit_search_(rows_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                             nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                             search_config_t config, std::size_t visible,
                                             heads_type_ const& heads = {}) noexcept {
        std::size_t const queries_count = queries.rows;
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
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
        u32_t const expansion = config.expansion_factor >= config.count ? config.expansion_factor : config.count * 2;
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
        hip_stream_handle_t const stream = context.stream();
        if (status_t const launched = launch_walk_(
                staged.value, amd_search_sink_t {device_ids, device_distances, config.count, kernel_}, stream);
            failed(launched))
            return {{}, launched};
        // The landing may be wider than the count, so a multi-index can hand every shard its own
        // columns of one results matrix.
        if (hip_api().copy_2d_async(result_ids.data(), result_ids.stride_bytes(0), device_ids,
                                    config.count * sizeof(node_id_t), config.count * sizeof(node_id_t), queries_count,
                                    hip_copy_default_k, stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        if (wants_distances &&
            hip_api().copy_2d_async(result_distances.data(), result_distances.stride_bytes(0), device_distances,
                                    config.count * sizeof(distance_t), config.count * sizeof(distance_t), queries_count,
                                    hip_copy_default_k, stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        return {{config.count}, status_t::success_k};
    }

    /** The submission half of @c multiply: stages the rows, launches walks over the @p visible
     *  rows, and queues the writeback of the kept pairs and the scored counts. */
    expected<search_result_t> submit_multiply_(rows_view_t left, nk::matrix_span<node_id_t> node_ids,
                                               nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                               traversal_context_t& context, search_config_t config,
                                               std::size_t visible) noexcept {
        std::size_t const rows_count = left.rows;
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
        auto const staged = stage_walk_(left, dummy_heads_t {}, context, non_zero_expansion_factor(config), 2,
                                        ids_bytes + products_bytes + rows_count * sizeof(u32_t), visible);
        if (!staged) return {{}, staged.status};
        node_id_t* const device_ids = reinterpret_cast<node_id_t*>(staged.value.outputs);
        byte_t* const device_products = staged.value.outputs + ids_bytes;
        u32_t* const device_counts = reinterpret_cast<u32_t*>(staged.value.outputs + ids_bytes + products_bytes);
        hip_stream_handle_t const stream = context.stream();
        amd_multiply_sink_t const sink {device_ids, device_products, device_counts, static_cast<u32_t>(wanted)};
        if (status_t const launched = launch_walk_(staged.value, sink, stream); failed(launched)) return {{}, launched};
        if (wanted &&
            (hip_api().copy_2d_async(node_ids.data(), node_ids.stride_bytes(0), device_ids, wanted * sizeof(node_id_t),
                                     wanted * sizeof(node_id_t), rows_count, hip_copy_default_k,
                                     stream) != hip_success_k ||
             hip_api().copy_2d_async(products.data(), products.stride_bytes(0), device_products, wanted * sizeof(dot_t),
                                     wanted * sizeof(dot_t), rows_count, hip_copy_default_k, stream) != hip_success_k))
            return {{}, status_t::device_error_k};
        if (hip_api().copy_async(counts.data(), device_counts, rows_count * sizeof(u32_t), hip_copy_default_k,
                                 stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        return {{}, status_t::success_k};
    }

    /** The submission half of @c attend: stages each token's first head to navigate by, then every
     *  head, and launches one walk per token over the @p visible rows, which lands the normalized
     *  rows, then the carries, in the first of @p blocks such blocks of staging, returned
     *  device-resident, with @p exported_bytes more past them for the rows cast on their way out;
     *  the sharded index gathers its other shards' into the rest. */
    template <typename value_lookup_type_>
    expected<f32_t*> submit_attend_(rows_view_t queries, std::size_t head_count, value_lookup_type_ const& value_lookup,
                                    f32_t score_scale, traversal_context_t& context, search_config_t config,
                                    std::size_t blocks, std::size_t exported_bytes, std::size_t visible) noexcept {
        std::size_t const rows_count = queries.rows;
        if (!head_count || rows_count % head_count) return {nullptr, status_t::shape_mismatch_k};
        if (!rows_count) return {nullptr, status_t::success_k};
        std::size_t const scratch_bytes = amd_attend_scratch_bytes(head_count, dimensions(), wave_lanes());
        if (!walk_scratch_fits(scratch_bytes)) return {nullptr, status_t::config_mismatch_k};
        std::size_t const outputs_bytes = rows_count * dimensions() * sizeof(f32_t);
        rows_view_t const first_heads {queries.data, rows_count / head_count, queries.columns,
                                       queries.stride * head_count, queries.dtype};
        auto staged = stage_walk_(
            first_heads, attend_heads_t {queries, head_count}, context, non_zero_expansion_factor(config), 2,
            blocks * (outputs_bytes + rows_count * sizeof(softmax_state_t)) + exported_bytes, visible);
        if (!staged) return {nullptr, staged.status};
        staged.value.view.scratch_bytes_per_warp += static_cast<u32_t>(scratch_bytes);
        if (hip_api().fill_async(context.missed_.get(), 0, sizeof(u32_t), context.stream()) != hip_success_k)
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

    /** A zeroed landing from @p context 's stream slots, or null when none can pin. */
    static audit_landing_ptr_t take_landing_(traversal_context_t& context) noexcept {
        hip_landings_t* const landings = context.allocator_.landings();
        void* const slot = landings ? landings->take() : nullptr;
        return {slot ? new (slot) audit_landing_t {} : nullptr, landing_return_t {landings}};
    }

    /** Queues one pass of the three-pass refinement engine behind @c refine_plan_t into its carved
     *  @p scratch: the walk covers @p nodes, or under a device-resident @p node_ids the listed IDs
     *  with @p nodes spanning the array's offsets, each node first walking across the neighboring
     *  group in round @p round when @p parts parts are given; the back-link passes re-select the
     *  walked rows and the rows they pick, and leave every other row as it was. Nothing waits: the
     *  landing queued after it checks its claims. */
    status_t refine_pass_(traversal_context_t& context, refine_scratch_t const& scratch, refine_config_t config,
                          iota_view<node_id_t> nodes, std::span<u32_t const> node_ids, u32_t parts = 0,
                          u32_t round = 0) noexcept {
        if (status_t const kept = status(); failed(kept)) return kept;
        std::size_t const nodes_count = size();
        std::size_t const walk_count = nodes.size();
        hip_stream_handle_t const stream = context.stream();
        u32_t const bucket_width = static_cast<u32_t>(connectivity());
        nk::matrix_span<u32_t> const buckets {scratch.buckets, nodes_count, bucket_width};
        std::span<u32_t> const bucket_counts {scratch.bucket_counts, nodes_count};
        // The claims, the reverse buckets' counters and the bitset of rows the merge re-selects lie
        // back to back.
        if (hip_api().fill_async(scratch.claims, 0, (2 + nodes_count + scratch.words) * sizeof(u32_t), stream) !=
            hip_success_k)
            return status_t::device_error_k;

        // Twice the connectivity by default, like every CPU walk, rounded up to the pool sizes the
        // kernel carries, or walked exactly past them.
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const pool_slots = pool_slots_for_(expansion, 4);
        // Walking across makes two walks into one visited set, so the slice is shaped for both.
        u32_t const walks = parts > 1 ? 2 : 1;
        auto const arena_shape = build_arena_shape_(expansion * walks, pool_slots * walks);
        std::size_t const arena_warps = device_.multiprocessor_count * insert_blocks_per_multiprocessor_k *
                                        vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return status_t::bad_alloc_k;
        // Only a fresh allocation or a wrapped 32-bit epoch space warrants zeroing, never a warm
        // arena.
        if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + walk_count > 0xFFFFFFFEull) {
            if (hip_api().fill_async(arena, 0, context.arena_bytes_, stream) != hip_success_k) {
                context.arena_bytes_ = 0; // Spent: a retry must re-zero, never walk epoch garbage.
                return status_t::device_error_k;
            }
            context.epoch_floor_ = 0;
        }
        u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += walk_count;

        amd_refine_t view {storage(), //
                           scratch.shadow_neighbors,
                           nodes.bound,
                           arena_shape.cache_bytes ? 0u : insert_scratch_bytes_per_warp(),
                           nodes.first,
                           arena,
                           arena_shape,
                           expansion,
                           epoch_base,
                           context.telemetry_.get(),
                           scratch.claims,
                           node_ids.data(),
                           static_cast<u32_t>(nodes_count),
                           parts > 1 ? scratch.part_bases : nullptr,
                           parts > 1 ? parts : 0,
                           round};
        // A range short of the whole graph starts from the live neighbors copied into the shadow
        // verbatim, so a row it never rewrites but one of its picks lands in merges its own
        // neighbors with the new reverse links; islands scatter nothing, so only their own rows
        // merge.
        if (node_ids.empty() && walk_count != nodes_count)
            if (hip_api().copy_async(scratch.shadow_neighbors, neighbors_matrix_.base(),
                                     nodes_count * connectivity() * sizeof(u32_t), hip_copy_device_to_device_k,
                                     stream) != hip_success_k)
                return status_t::device_error_k;
        if (status_t const launched = launch_refine_(view, pool_slots, stream); failed(launched)) return launched;

        // Scatter the walked rows' shadow neighbors in reverse, then merge each touched node's
        // forward & reverse candidates into its final live row. A heal round over island IDs
        // re-selects the islands' own rows alone: re-judging the rows they pick would drop the
        // links that reconnected earlier islands.
        amd_arguments_t scatter {};
        scatter.graph.neighbors = scratch.shadow_neighbors;
        scatter.graph.rows_count = static_cast<u32_t>(nodes_count);
        scatter.graph.first_id = nodes.first;
        scatter.graph.walk_count = static_cast<u32_t>(walk_count);
        scatter.graph.node_ids = node_ids.data();
        scatter.graph.connectivity = static_cast<u32_t>(connectivity());
        scatter.graph.bucket_width = bucket_width;
        scatter.graph.bucket_ids = buckets.data();
        scatter.graph.bucket_counts = bucket_counts.data();
        scatter.graph.touched = scratch.touched;
        if (status_t const launched = launch_(amd_kernel_t::scatter_back_links_k, 0,
                                              device_.multiprocessor_count * insert_blocks_per_multiprocessor_k, 0,
                                              stream, scatter);
            failed(launched))
            return launched;
        amd_refine_t merge_view = view;
        merge_view.scratch_bytes_per_warp = arena_shape.cache_bytes ? 0u : merge_scratch_bytes_per_warp();
        merge_view.assigned_rows = scratch.claims + 1;
        // The merge sweeps the whole ID space and skips the rows the scatter left untouched.
        merge_view.node_ids = nullptr;
        merge_view.first_id = 0;
        merge_view.end_id = static_cast<u32_t>(nodes_count);
        return launch_merge_back_links_(merge_view, buckets, bucket_counts, scratch.touched, stream);
    }

    /** Which share of the scratch a carve covers: the audit's alone, or the whole of a refine's. */
    enum class scratch_use_t : bool { audit_k, refine_k };

    /** Bytes of the scratch over the rows published now: the audit share, and for @c refine_k the
     *  passes' buffers, the islands' IDs and in-links, and @p parts bases too. */
    std::size_t scratch_bytes_(scratch_use_t use, std::size_t parts = 0) const noexcept {
        std::size_t const rows = size(), words = divide_round_up<std::size_t>(rows, 32);
        std::size_t const passes_words = use == scratch_use_t::refine_k
                                             ? rows * (3 + 2 * connectivity()) + words + parts
                                             : 0;
        return (words * 3 + audit_sweeps_cap_k + 4 + passes_words) * sizeof(u32_t);
    }

    /** Carves the scratch over the rows published now, for @c refine_k the whole of a refine's
     *  with room for @p parts bases, otherwise the audit share alone, the reconnected counter
     *  zeroed. */
    expected<refine_scratch_t> carve_scratch_(traversal_context_t& context, scratch_use_t use,
                                              std::size_t parts = 0) noexcept {
        refine_scratch_t scratch {typename traversal_context_t::bytes_ptr_t {
            nullptr, hip_deleter<typename traversal_context_t::allocator_t> {context.allocator_, 0}}};
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
        scratch.expanded = scratch.next_frontier.words + scratch.words;
        scratch.unreached = scratch.expanded + audit_sweeps_cap_k;
        scratch.reconnected = scratch.unreached + 1;
        scratch.claims = scratch.reconnected + 1;
        if (use == scratch_use_t::refine_k) {
            scratch.bucket_counts = scratch.claims + 2;
            scratch.touched.words = scratch.bucket_counts + rows;
            scratch.island_ids = scratch.touched.words + scratch.words;
            scratch.in_links = scratch.island_ids + rows;
            scratch.shadow_neighbors = scratch.in_links + rows;
            scratch.buckets = scratch.shadow_neighbors + rows * connectivity();
            scratch.part_bases = scratch.buckets + rows * connectivity();
        }
        if (hip_api().fill_async(scratch.reconnected, 0, sizeof(u32_t), context.stream()) != hip_success_k)
            return {std::move(scratch), status_t::device_error_k};
        return {std::move(scratch), status_t::success_k};
    }

    /** Queues the copy of @p scratch 's counters into @p landing: the levels apart, then the last
     *  sweep's count beside the unreached, reconnected and claim counters, which lie back to
     *  back. */
    static status_t land_counters_(refine_scratch_t const& scratch, audit_landing_t* landing,
                                   hip_stream_handle_t stream) noexcept {
        hip_api_t const& runtime = hip_api();
        return runtime.copy_async(&landing->levels, scratch.expanded, sizeof(u32_t), hip_copy_device_to_host_k,
                                  stream) == hip_success_k &&
                       runtime.copy_async(&landing->last_expanded, scratch.expanded + audit_sweeps_cap_k - 1,
                                          sizeof(audit_landing_t) - sizeof(u32_t), hip_copy_device_to_host_k,
                                          stream) == hip_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Queues a reachability sweep to convergence from as many stride-spread landmarks as one
     *  search draws - the connectivity - then a count of the rows it never reached, collecting
     *  their IDs when @p scratch has room for them, then the landing of every counter. Nothing
     *  waits; the caller reads @p landing once its event fired. */
    status_t queue_audit_(traversal_context_t& context, refine_scratch_t const& scratch,
                          audit_landing_t* landing) noexcept {
        std::size_t const rows = size();
        hip_stream_handle_t const stream = context.stream();
        std::size_t const seed_count = min_of<std::size_t>(connectivity(), rows);
        // The bitsets, the levels and sweeps' expanded counters and the unreached counter lie back
        // to back.
        if (hip_api().fill_async(scratch.reached.words, 0, (scratch.words * 3 + audit_sweeps_cap_k + 1) * sizeof(u32_t),
                                 stream) != hip_success_k)
            return status_t::device_error_k;
        amd_arguments_t sweep {};
        sweep.graph.neighbors = neighbor_rows_();
        sweep.graph.rows_count = static_cast<u32_t>(rows);
        sweep.graph.connectivity = static_cast<u32_t>(connectivity());
        sweep.graph.seed_count = static_cast<u32_t>(seed_count);
        sweep.graph.reached = scratch.reached;
        sweep.graph.frontier = scratch.frontier;
        sweep.graph.next_frontier = scratch.next_frontier;
        sweep.graph.counter = scratch.expanded;
        sweep.graph.levels = scratch.expanded;
        sweep.graph.island_ids = scratch.island_ids;
        if (status_t const launched = launch_(amd_kernel_t::mark_seeds_k, 0,
                                              divide_round_up<u32_t>(static_cast<u32_t>(seed_count), block_threads()),
                                              0, stream, sweep);
            failed(launched))
            return launched;

        // Every sweep up to the cap queues blind: one queued past convergence finds the one before
        // it expanded nothing and leaves at entry.
        u32_t const grid = device_.multiprocessor_count * search_blocks_per_multiprocessor_k;
        for (std::size_t sweep_number = 1; sweep_number != audit_sweeps_cap_k; ++sweep_number) {
            sweep.graph.counter = scratch.expanded + sweep_number;
            sweep.graph.previous_counter = sweep_number == 1 ? nullptr : scratch.expanded + sweep_number - 1;
            if (hip_api().fill_async(sweep.graph.next_frontier.words, 0, scratch.words * sizeof(u32_t), stream) !=
                hip_success_k)
                return status_t::device_error_k;
            if (status_t const launched = launch_(amd_kernel_t::reachability_sweep_k, 0, grid, 0, stream, sweep);
                failed(launched))
                return launched;
            std::swap(sweep.graph.frontier, sweep.graph.next_frontier);
        }
        sweep.graph.counter = scratch.unreached;
        if (status_t const launched = launch_(amd_kernel_t::reachability_islands_k, 0,
                                              divide_round_up<u32_t>(static_cast<u32_t>(rows), block_threads()), 0,
                                              stream, sweep);
            failed(launched))
            return launched;
        return land_counters_(scratch, landing, stream);
    }

    /** The audit @p landing reports over @p rows rows: @c overflow_k when its last sweep still
     *  expanded, since the rows past the cap are then unjudged rather than unreached. */
    static expected<health_result_t> audited_(audit_landing_t const& landing, std::size_t rows) noexcept {
        if (landing.last_expanded) return {{rows, rows - landing.unreached, audit_sweeps_cap_k}, status_t::overflow_k};
        // A sweep that expands nothing ends the levels, and it counts as one, as on the CPU.
        return {{rows, rows - landing.unreached, std::size_t {landing.levels} + 1}, status_t::success_k};
    }

    /** Queues the reconnection of the @p islands islands the last audit in @p scratch collected:
     *  the reached in-links first, so no reconnection takes the last link a reached node is found
     *  through, then each island into a reached pick or its landmark. */
    status_t queue_reconnect_(refine_scratch_t const& scratch, u32_t islands, hip_stream_handle_t stream) noexcept {
        std::size_t const rows = size();
        if (hip_api().fill_async(scratch.in_links, 0, rows * sizeof(u32_t), stream) != hip_success_k)
            return status_t::device_error_k;
        amd_arguments_t heal {};
        heal.graph.neighbors = neighbor_rows_();
        heal.graph.rows_count = static_cast<u32_t>(rows);
        heal.graph.connectivity = static_cast<u32_t>(connectivity());
        heal.graph.reached = scratch.reached;
        heal.graph.in_links = scratch.in_links;
        heal.graph.island_ids = scratch.island_ids;
        heal.graph.islands_count = islands;
        heal.graph.seed_count = static_cast<u32_t>(min_of<std::size_t>(connectivity(), rows));
        heal.graph.reconnected = scratch.reconnected;
        if (status_t const counted = launch_(amd_kernel_t::reached_in_links_k, 0,
                                             device_.multiprocessor_count * search_blocks_per_multiprocessor_k, 0,
                                             stream, heal);
            failed(counted))
            return counted;
        return launch_(amd_kernel_t::reconnect_island_k, 0, divide_round_up<u32_t>(islands, block_threads()), 0, stream,
                       heal);
    }

#pragma endregion Refinement

#pragma region Launches

    /** Insertion blocks per multiprocessor: the resident count saturates the path at its
     *  registers. */
    inline static constexpr u32_t insert_blocks_per_multiprocessor_k = vram_resident_blocks_per_multiprocessor_k;

    /** Search blocks per multiprocessor: oversubscribed, so enough queries stay in flight to reach
     *  the memory-level-parallelism knee. */
    inline static constexpr u32_t search_blocks_per_multiprocessor_k = 8;

    /** Blocks a walk over @p queries launches: a warp per query up to the oversubscribed grid, so
     *  the arena carves a slice only for a warp that can take a query. */
    u32_t walk_blocks_(std::size_t queries) const noexcept {
        return static_cast<u32_t>(
            min_of<std::size_t>(divide_round_up<std::size_t>(queries, vram_warps_per_block_k),
                                device_.multiprocessor_count * search_blocks_per_multiprocessor_k));
    }

    /** Keeps @p failure as the status unless one is kept already; returns it for the caller's own
     *  report. */
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
     *  with no batch or load in flight, so its holder alone grows, launches and publishes them. */
    expected<node_id_t> claim_(std::size_t rows) noexcept { return claims_.claim(size(), rows, max_node_count_k); }

    /** Gives back the claim over @p rows from @p first, which launched nothing; a word that moved
     *  on, closed by a caller around a load, stays as it is. */
    void unclaim_(std::size_t first, std::size_t rows) noexcept { claims_.unclaim(first, rows); }

    /** Closes the claim word at the watermark, so every claim refuses until @ref reopen_; refused
     *  while a batch or load is in flight. */
    status_t close_() noexcept { return claims_.close(size()); }

    /** Reopens a closed claim word at the watermark, past every row a load published meanwhile; an
     *  open one stays as it is. */
    void reopen_() noexcept { claims_.reopen(size()); }

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

    /** Row zero of the vectors matrix, one row past its base. */
    byte_t* vectors_rows_() const noexcept { return vectors_matrix_.base() + row_bytes(); }

    /** The device-resident published-row count every inserting wave advances: the vectors matrix's
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
    status_t grow_rows_(std::size_t rows, hip_stream_handle_t stream) noexcept {
        if (status_t const mapped = map_rows_(rows); failed(mapped)) return mapped;
        hip_api_t const& runtime = hip_api();
        if (runtime.stream_wait_event(stream, grown_, 0) != hip_success_k) return status_t::device_error_k;
        if (rows <= filled_rows_) return status_t::success_k;
        std::size_t const mapped_rows = capacity(), row_words = connectivity();
        if (!filled_rows_ && runtime.fill_words_async(published_rows_(), 0, 1, stream) != hip_success_k)
            return status_t::device_error_k;
        if (runtime.fill_words_async(neighbors_matrix_.base() + filled_rows_ * row_words * sizeof(u32_t),
                                     static_cast<int>(node_id_unknown_k), (mapped_rows - filled_rows_) * row_words,
                                     stream) != hip_success_k ||
            failed(grown_.record(stream)))
            return status_t::device_error_k;
        filled_rows_ = mapped_rows;
        return status_t::success_k;
    }

    /** Rounds the requested expansion to the closed pool ladder of 2, 4, 8 or 16 lanes' worth, or
     *  zero past it, where the arena frontier's rung walks the expansion itself. */
    u32_t pool_slots_for_(u32_t expansion_factor, u32_t default_slots) const noexcept {
        if (!expansion_factor) return default_slots;
        u32_t const slots = std::bit_ceil(divide_round_up<u32_t>(expansion_factor, wave_lanes()));
        return slots > vram_widest_pool_slots_k ? 0 : max_of(slots, 2u);
    }

    /** One building wavefront's arena slice on the rung of @p pool_slots, the arena frontier's
     *  over @p expansion at zero, then the builders' scratch when the LDS cannot seat it. */
    amd_arena_shape_t build_arena_shape_(u32_t expansion, u32_t pool_slots) const noexcept {
        amd_arena_shape_t shape = pool_slots
                                      ? arena_shape_for(pool_slots * wave_lanes(), frontier_kind_t::register_ladder_k)
                                      : arena_shape_for(expansion, frontier_kind_t::wide_arena_k);
        if (!build_scratch_fits())
            shape.cache_bytes = max_of(insert_scratch_bytes_per_warp(), merge_scratch_bytes_per_warp());
        return shape;
    }

    /** Loads this class's code object onto the device and finds every kernel this dtype exports bar
     *  the attends, by the mangled name @ref amd_kernel_name spells. */
    status_t load_kernels_() noexcept {
        auto [module, status] = hip_module_t::load(device_.ordinal, amd_code_object<capability_>());
        if (failed(status)) return status;
        module_ = std::move(module);
        char name[amd_kernel_name_capacity_k];
        for (amd_rung_t const rung : amd_rungs) {
            amd_kernel_name(name, rung, amd_class_of(capability_), vector_dtype_);
            hip_function_handle_t& function = kernels_[static_cast<u32_t>(rung.kernel)][amd_rung_index(rung.slots)];
            if (!(function = module_.function(name))) return status_t::device_error_k;
        }
        return status_t::success_k;
    }

    /** Launches @p kernel on its @p slots rung over @p blocks of four wavefronts on @p stream. */
    status_t launch_(amd_kernel_t kernel, u32_t slots, u32_t blocks, u32_t shared_bytes, hip_stream_handle_t stream,
                     amd_arguments_t const& arguments) noexcept {
        return hip_launch(kernels_[static_cast<u32_t>(kernel)][amd_rung_index(slots)], blocks, block_threads(),
                          shared_bytes, stream, arguments);
    }

    /** Launches the rows of @p build in waves, each queued behind the last and as wide as the IDs
     *  below it, never narrower than @c vram_admission_ramp_floor_k, so every ID a wave admits
     *  walks rows already published. A wave failing after an earlier one launched keeps the
     *  failure, since those rows can no longer be un-claimed. */
    status_t launch_insert_(amd_build_t const& build, u32_t pool_slots, hip_stream_handle_t stream) noexcept {
        amd_arguments_t arguments {};
        u32_t const blocks = device_.multiprocessor_count * insert_blocks_per_multiprocessor_k;
        for (u32_t wave_first = build.first_id; wave_first != build.end_id;) {
            u32_t const wave_end = static_cast<u32_t>(
                min_of<u64_t>(build.end_id, u64_t {wave_first} + max_of(wave_first, vram_admission_ramp_floor_k)));
            u32_t const offset = wave_first - build.first_id;
            arguments.build = build;
            arguments.build.staging = build.staging + std::size_t {offset} * row_bytes();
            arguments.build.first_id = wave_first;
            arguments.build.end_id = wave_end;
            arguments.build.epoch_base = build.epoch_base + offset;
            status_t launched = hip_api().fill_async(build.assigned_rows, 0, sizeof(u32_t), stream) == hip_success_k
                                    ? status_t::success_k
                                    : status_t::device_error_k;
            if (succeeded(launched))
                launched = launch_(amd_kernel_t::insert_k, pool_slots, blocks,
                                   vram_warps_per_block_k * build.scratch_bytes_per_warp, stream, arguments);
            if (failed(launched)) return offset ? fail_(launched) : launched;
            wave_first = wave_end;
        }
        return status_t::success_k;
    }
    status_t launch_refine_(amd_refine_t const& view, u32_t pool_slots, hip_stream_handle_t stream) noexcept {
        amd_arguments_t arguments {};
        arguments.refine = view;
        return launch_(amd_kernel_t::refine_k, pool_slots,
                       device_.multiprocessor_count * insert_blocks_per_multiprocessor_k,
                       vram_warps_per_block_k * view.scratch_bytes_per_warp, stream, arguments);
    }
    status_t launch_merge_back_links_(amd_refine_t const& view, nk::matrix_span<u32_t> buckets,
                                      std::span<u32_t> bucket_counts, hip_bitset_t touched,
                                      hip_stream_handle_t stream) noexcept {
        amd_arguments_t arguments {};
        arguments.refine = view;
        arguments.graph.bucket_ids = buckets.data();
        arguments.graph.bucket_counts = bucket_counts.data();
        arguments.graph.bucket_width = static_cast<u32_t>(buckets.extent(1));
        arguments.graph.touched = touched;
        return launch_(amd_kernel_t::merge_back_links_k, 0,
                       device_.multiprocessor_count * insert_blocks_per_multiprocessor_k,
                       vram_warps_per_block_k * view.scratch_bytes_per_warp, stream, arguments);
    }
    status_t launch_derive_norms_(u32_t first_row, u32_t rows, u32_t blocks, hip_stream_handle_t stream) noexcept {
        amd_arguments_t arguments {};
        arguments.graph.storage = storage();
        arguments.graph.first_id = first_row;
        arguments.graph.rows_count = rows;
        return launch_(amd_kernel_t::derive_norms_k, 0, blocks, 0, stream, arguments);
    }

    /** Rebases rows @p rows of the neighbors matrix from @p source_range onto the IDs from @p shift
     *  and compacts them, as a load through the bounce does on the host. */
    status_t launch_rebase_links_(iota_view<node_id_t> rows, iota_view<node_id_t> source_range, node_id_t shift,
                                  hip_stream_handle_t stream) noexcept {
        amd_arguments_t arguments {};
        amd_graph_t& graph = arguments.graph;
        graph.neighbors = neighbor_rows_();
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.first_id = rows.first;
        graph.rows_count = static_cast<u32_t>(rows.size());
        graph.source_first = source_range.first;
        graph.source_bound = source_range.bound;
        graph.shift = shift;
        return launch_(amd_kernel_t::rebase_links_k, 0,
                       device_.multiprocessor_count * search_blocks_per_multiprocessor_k, 0, stream, arguments);
    }

    /** Merges the attended rows and carries of @p shards walks into the first's on @p stream - the
     *  sharded attend's gather. */
    status_t launch_merge_attention_(byte_t* blocks, u32_t rows, u32_t columns, u32_t shards,
                                     hip_stream_handle_t stream) noexcept {
        amd_arguments_t arguments {};
        arguments.attend.outputs = reinterpret_cast<f32_t*>(blocks);
        arguments.attend.rows = rows;
        arguments.attend.columns = columns;
        arguments.attend.shards = shards;
        return launch_(amd_kernel_t::merge_attention_k, 0, divide_round_up<u32_t>(rows, vram_warps_per_block_k), 0,
                       stream, arguments);
    }

    /** Resolves the @p walked matches per query of @p queries_count at @p ids and @p distances
     *  through the device @p keys column into @p ranking on @p stream - a sharded search's own
     *  ranking, gathered or landed as is. */
    status_t launch_resolve_matches_(node_id_t const* ids, distance_t const* distances, u32_t walked,
                                     byte_t const* keys, amd_ranking_t const& ranking, u32_t queries_count,
                                     hip_stream_handle_t stream) noexcept {
        amd_arguments_t arguments {};
        arguments.matches.ids = ids;
        arguments.matches.distances = distances;
        arguments.matches.keys = keys;
        arguments.matches.ranking = ranking;
        arguments.matches.walked = walked;
        arguments.matches.queries_count = queries_count;
        return launch_(amd_kernel_t::resolve_matches_k, 0,
                       divide_round_up<u32_t>(queries_count, vram_warps_per_block_k), 0, stream, arguments);
    }

    /** Merges @p shards rankings of @p queries_count queries, gathered @p block_bytes apart from
     *  @p gathered, into @p ranking on @p stream - the sharded search's device gather. */
    status_t launch_merge_matches_(amd_ranking_t const& gathered, std::size_t block_bytes, u32_t shards,
                                   amd_ranking_t const& ranking, u32_t queries_count,
                                   hip_stream_handle_t stream) noexcept {
        amd_arguments_t arguments {};
        arguments.matches.gathered = gathered;
        arguments.matches.ranking = ranking;
        arguments.matches.block_bytes = block_bytes;
        arguments.matches.queries_count = queries_count;
        arguments.matches.shards = shards;
        return launch_(amd_kernel_t::merge_matches_k, 0, divide_round_up<u32_t>(queries_count, vram_warps_per_block_k),
                       0, stream, arguments);
    }

    /** Casts @p count values at @p from in @p from_dtype into @p to in @p to_dtype on @p stream,
     *  through NumKong's plan and its per-unit device conversions. */
    status_t launch_cast_(void const* from, nk_dtype_t from_dtype, void* to, nk_dtype_t to_dtype, std::size_t count,
                          hip_stream_handle_t stream) noexcept {
        amd_arguments_t arguments {};
        arguments.cast.from = static_cast<unsigned char const*>(from);
        arguments.cast.to = static_cast<unsigned char*>(to);
        if (!nk_cast_plan_serial_(from_dtype, to_dtype, count,
                                  reinterpret_cast<std::uintptr_t>(from) | reinterpret_cast<std::uintptr_t>(to),
                                  &arguments.cast.plan))
            return status_t::unsupported_k;
        u32_t const blocks = static_cast<u32_t>(
            min_of<std::size_t>(divide_round_up<std::size_t>(arguments.cast.plan.units, block_threads()),
                                device_.multiprocessor_count * search_blocks_per_multiprocessor_k));
        return launch_(amd_kernel_t::cast_k, 0, blocks, 0, stream, arguments);
    }

    /** Narrows @p rows rows of the index's columns at @p from, @p from_stride bytes apart, into
     *  @p to at the vectors matrix pitch at unit RMS on @p stream, one block per row. */
    status_t launch_rmscast_(f32_t const* from, std::size_t from_stride, byte_t* to, std::size_t rows,
                             hip_stream_handle_t stream) noexcept {
        amd_arguments_t arguments {};
        nk_each_rmsnorm_arguments_t& rmscast = arguments.rmscast;
        rmscast.x = reinterpret_cast<unsigned char const*>(from);
        rmscast.y = reinterpret_cast<unsigned char*>(to);
        rmscast.groups = 1;
        rmscast.columns = dimensions();
        rmscast.vectors = rows;
        rmscast.x_stride = from_stride;
        rmscast.y_stride = row_bytes();
        rmscast.epsilon = std::numeric_limits<f32_t>::min();
        u32_t const blocks = static_cast<u32_t>(
            min_of<std::size_t>(rows, device_.multiprocessor_count * search_blocks_per_multiprocessor_k));
        return launch_(amd_kernel_t::rmscast_k, 0, blocks, 0, stream, arguments);
    }

    /** Launches one walk per query of @p launch, each written out through @p sink: the search,
     *  indexed search and multiply families, each on its register rung or the arena frontier. */
    template <typename sink_, typename rescoring_>
    status_t launch_walk_(walk_launch<rescoring_> const& launch, sink_ sink, hip_stream_handle_t stream) noexcept {
        u32_t const blocks = walk_blocks_(launch.view.queries_count);
        amd_arguments_t arguments {};
        arguments.search = launch.view;
        arguments.rescoring = launch.rescoring;
        amd_kernel_t kernel = amd_kernel_t::search_k;
        if constexpr (std::is_same_v<sink_, amd_multiply_sink_t>) {
            arguments.multiply_sink = sink;
            kernel = amd_kernel_t::multiply_k;
        }
        else {
            arguments.search_sink = sink;
            if (launch.rescoring.heads) kernel = amd_kernel_t::search_indexer_k;
        }
        u32_t const slots = launch.frontier == frontier_kind_t::wide_arena_k ? 0 : launch.pool_slots;
        return launch_(kernel, slots, blocks, vram_warps_per_block_k * launch.view.scratch_bytes_per_warp, stream,
                       arguments);
    }

    /** Walks every token of @p launch under its staged heads, folding the value rows
     *  @p value_lookup resolves into the normalized @p outputs and their @p states; the lookup
     *  travels as bytes to the attend kernel its @c device_lookup_k names, found in the code object
     *  by name. */
    template <typename value_lookup_type_>
    status_t launch_attend_(walk_launch<amd_rescoring_t> const& launch, value_lookup_type_ const& value_lookup,
                            f32_t* outputs, softmax_state_t* states, f32_t logit_scale,
                            traversal_context_t& context) noexcept {
        static_assert(std::is_trivially_copyable_v<value_lookup_type_> &&
                          sizeof(value_lookup_type_) <= amd_value_lookup_capacity_k &&
                          alignof(value_lookup_type_) <= 16,
                      "A device value lookup travels to the kernels as its own bytes");
        amd_arguments_t arguments {};
        arguments.search = launch.view;
        arguments.rescoring = launch.rescoring;
        amd_attend_t& attend = arguments.attend;
        attend.outputs = outputs;
        attend.states = states;
        attend.missed = context.missed_.get();
        attend.logit_scale = logit_scale;
        attend.heads_count = launch.rescoring.heads_count;
        attend.columns = static_cast<u32_t>(dimensions());
        std::memcpy(attend.value_lookup, &value_lookup, sizeof(value_lookup));
        u32_t const slots = launch.frontier == frontier_kind_t::wide_arena_k ? 0 : launch.pool_slots;
        char name[amd_kernel_name_capacity_k];
        amd_kernel_name(name, {amd_kernel_t::attend_k, slots}, amd_class_of(capability_), vector_dtype_,
                        value_lookup_type_::device_lookup_k);
        return hip_launch(module_.function(name), walk_blocks_(launch.view.queries_count), block_threads(),
                          vram_warps_per_block_k * launch.view.scratch_bytes_per_warp, context.stream(), arguments);
    }

#pragma endregion Launches

    /** The vectors matrix and norms column backend, rebound to bytes. */
    using vectors_matrix_allocator_t = typename vectors_allocator_t::template rebind<byte_t>::other;

    /** The neighbors matrix backend, rebound to bytes. */
    using neighbors_matrix_allocator_t = typename neighbors_allocator_t::template rebind<byte_t>::other;

    /** This class's code object on this device, and its kernels in @ref amd_kernel_t order, the
     *  attends left to their launches. */
    hip_module_t module_;
    hip_function_handle_t kernels_[static_cast<u32_t>(amd_kernel_t::count_k)][amd_rungs_per_kernel_k] = {};

    /** Hardware identity: the ordinal comes from the vectors allocator at construction, the
     *  remaining specs fill at the first @c reserve. */
    hip_device_specs_t device_;

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
    vram_claim_word_t claims_;

    /** Mapped rows whose neighbors a grow sentinel-filled, read and advanced by the claim's holder
     *  alone; a reservation maps past it and the next claim fills the rest. */
    std::size_t filled_rows_ = 0;

    /** Marks the last sentinel fill, or the copies of a load that died over filled rows, which
     *  every grow's stream follows; recorded and followed by the claim's holder alone. */
    hip_event_t grown_;

    /** Rows strictly below it are published and visible: turned once per batch, in claim order,
     *  after its drain. */
    std::size_t last_visible_id_ = 0;

    /** The first post-claim device failure: IDs past it may be unpublished or unfilled. */
    status_t status_ = status_t::success_k;
};

} // namespace unum::usearch::dense
