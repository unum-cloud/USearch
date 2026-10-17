/**
 *  @file include/usearch/basic/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over variable-length values in a single uniform latency memory space on
 *      NVIDIA GPUs.
 *
 *  The host base of the NVIDIA basic faces, compiled by any C++ compiler: it fills plain argument
 *  records and launches each modality's kernels by name, as the dense host face does, through the
 *  dense traversal contexts. A modality's face derives from it, as dense faces derive from theirs:
 *  it names its kits' rungs, and reads its kits for the rung a launch's longest row needs, the
 *  lanes that rung scores a row with and the span one row's preparation takes, in its vendor
 *  metric, which only its NVIDIA unit compiles, so dispatch never names a kit.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <bit>              // `std::bit_ceil`, `std::bit_width`, `std::has_single_bit`
#include <initializer_list> // the two walks of every wave
#include <limits>           // `std::numeric_limits` - the node-ID type's sentinel
#include <memory>           // `std::unique_ptr` - an audit's landing
#include <span>             // `std::span` - the caller's row offsets, a snapshot's columns
#include <utility>          // `std::exchange` - a moved-from plan forgets its index

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the records carry the kernel kind
#include <numkong/tensor.hpp>     // `nk::matrix_span` - the matches' landing

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"        // `growing_buffer` - a bounce and its host offsets
#include "usearch/utilities/vram_device.hpp"        // `vram_kernel_t` - the kernels a rung names
#include "usearch/utilities/vram_nvidia_device.hpp" // `cuda_traversal_context` & the rungs
#include "usearch/utilities/vram_walks.hpp"         // `gpu_arena_shape_t`, `gpu_pool_slots`, `vram_bitset_t`
#include "usearch/basic/ram_index.hpp"              // `columns`, `columns_extents_t`, `refine_result_t`

namespace unum::usearch::basic {
using namespace unum::usearch::utilities;

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
    u32_t epoch_base = 0;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_rows = nullptr;
    node_id_type_* link_heads = nullptr;
    node_id_type_* link_next = nullptr;
    node_id_type_* touched_ids = nullptr;
    u32_t* touched_count = nullptr;
};

/** One walking launch and its landing, mirroring the shared search view. */
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
    node_id_type_* result_ids = nullptr;
    distance_t* result_distances = nullptr;
    u32_t count = 0;
};

/** The one argument every NVIDIA basic kernel takes, each reading the record its kernel names. */
template <typename node_id_type_>
struct vram_arguments {
    vram_build<node_id_type_> build;
    vram_search<node_id_type_> search;
};

/** One launch over the neighbors matrix alone, reading no value: the rebase of rows a donor gave
 *  up, the reachability audit and the reconnection healing it, and a split's partition and the
 *  scatter moving its rows, whose values travel as opaque bytes. */
template <typename node_id_type_>
struct vram_graph {
    node_id_type_* neighbors = nullptr;
    u32_t connectivity = 0;
    u32_t rows_count = 0;
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

    /** A split's part of each row, then its slot once the parts are ordered, the round's proposals,
     *  and per part its size, its round's tally and its root. */
    u32_t* labels = nullptr;
    u32_t* proposals = nullptr;
    u32_t* part_sizes = nullptr;
    u32_t* proposed = nullptr;
    u32_t const* roots = nullptr;
    u32_t parts_count = 0;
    u32_t part_capacity = 0;
    vram_bitset_t seams {};

    /** The receivers' neighbors matrices, tapes and offsets columns, by part, and the source's tape
     *  and offsets. */
    u32_t* const* part_neighbors = nullptr;
    byte_t* const* part_values = nullptr;
    u64_t const* const* part_offsets = nullptr;
    byte_t const* values = nullptr;
    u64_t const* value_offsets = nullptr;
};

/** Blocks of @p kernel on its @p slots rung resident per multiprocessor - the register budget its
 *  launch bounds set and its grid's share of every multiprocessor: two for the builders and the
 *  back-link riding their grid and arena, the most whose share of the register file holds the walk
 *  and its selection without spilling; a walk's by its rung; eight for the passes striding the
 *  neighbors matrix and a split's rows, oversubscribed so enough rows stay in flight; and two for
 *  the rest, whose grids follow their work. */
constexpr u32_t cuda_resident_blocks(vram_kernel_t kernel, u32_t slots) noexcept {
    switch (kernel) {
    case vram_kernel_t::search_k: return slots <= 8 ? 4 : 3;
    case vram_kernel_t::rebase_links_k:
    case vram_kernel_t::reachability_sweep_k:
    case vram_kernel_t::reached_in_links_k:
    case vram_kernel_t::partition_propose_k:
    case vram_kernel_t::partition_tally_k:
    case vram_kernel_t::partition_accept_k:
    case vram_kernel_t::partition_seams_k:
    case vram_kernel_t::split_links_k:
    case vram_kernel_t::split_tape_k: return 8;
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
    {vram_kernel_t::insert_k, 0},  {vram_kernel_t::insert_k, 2},    {vram_kernel_t::insert_k, 4},
    {vram_kernel_t::insert_k, 8},  {vram_kernel_t::insert_k, 16},   {vram_kernel_t::refine_k, 0},
    {vram_kernel_t::refine_k, 2},  {vram_kernel_t::refine_k, 4},    {vram_kernel_t::refine_k, 8},
    {vram_kernel_t::refine_k, 16}, {vram_kernel_t::back_link_k, 0}, {vram_kernel_t::search_k, 0},
    {vram_kernel_t::search_k, 2},  {vram_kernel_t::search_k, 4},    {vram_kernel_t::search_k, 8},
    {vram_kernel_t::search_k, 16},
};

/** Every kernel of the unit all NVIDIA basic faces share, over the neighbors matrix alone, so one
 *  instantiation per node-ID type serves every modality. */
inline constexpr vram_kernel_t cuda_graph_kernels[] = {
    vram_kernel_t::rebase_links_k,         vram_kernel_t::mark_seeds_k,        vram_kernel_t::reachability_sweep_k,
    vram_kernel_t::reachability_islands_k, vram_kernel_t::reached_in_links_k,  vram_kernel_t::reconnect_island_k,
    vram_kernel_t::partition_seeds_k,      vram_kernel_t::partition_propose_k, vram_kernel_t::partition_tally_k,
    vram_kernel_t::partition_accept_k,     vram_kernel_t::partition_fill_k,    vram_kernel_t::partition_islands_k,
    vram_kernel_t::partition_seams_k,      vram_kernel_t::split_links_k,       vram_kernel_t::split_tape_k,
};

/** The fatbin of @ref cuda_graph_kernels over 32-bit node IDs, defined beside them in
 *  @c src/basic/vram_variants_nvidia.cu, which every face loads at its first @c reserve. */
struct cuda_graph_fatbin_t {
    static bytes_view_t bytes() noexcept;
};

#pragma endregion Kernel Records

/**
 *  @brief Similarity Search over variable-length values in a single uniform latency memory space on
 *      NVIDIA GPUs.
 *
 *  Speaks the dense VRAM index's verbs over values of any length: a batch's IDs are one contiguous
 *  range, and its rows turn visible to searches a whole batch at a time. Values sit back to back in
 *  one tape beside a column of row starts, as the caller's own tape lays them out, and the
 *  neighbors matrix holds @c connectivity IDs per row; all three reserve their address range once
 *  and map on demand. The kit rung a launch scores on is the narrowest whose rows cover the longest
 *  one the launch meets, and the spans its walks prepare rows into are as wide as the widest of
 *  them.
 *
 *  A batch inserts in admission waves, each no wider than a quarter of the rows already below it.
 *  Every wave publishes forward rows alone, threading each pick onto its target's chain, and a
 *  back-link launch then merges every chain into the row it names, so a node never waits on a lock
 *  across a distance. Its rows then walk again over the linked graph, where they reach one
 *  another, re-select their own rows, and back-link those picks too. The merges rewrite live rows,
 *  so one batch holds the claim word from its claim until its drain, and another answers
 *  @c contended_k meanwhile, while searches on other streams keep running throughout.
 *
 *  A refinement re-selects rows the same way over the whole graph, and an absorb lands other
 *  indexes' rows on the device before refining. Both, and the reachability audit, launch kernels
 *  that read the neighbors matrix alone, which one unit shares across every modality. Saves and
 *  loads cross the snapshot columns of the basic RAM index through a pinned bounce, values through
 *  the face's codec. Plans close the claim word, or claim their rows, for their whole lives.
 *
 *  @tparam derived_ The modality's face, whose @c rungs_k kit rungs its kernels cover, narrowest
 *      first, and whose members read its kits: @c rung_serving_ rows of a length, or @c rungs_k
 *      past the widest, @c lanes_per_row_ of a rung, the @c span_bytes_ one row prepares into on
 *      every rung alike, and @c load_kernels_ and @c kernel_function_ over its fatbin; its codec,
 *      @c saved_bytes_ and @c save_value_ turning a stored row into the bytes the RAM codec saves,
 *      and @c stored_bytes_ and @c load_value_ back.
 *  @tparam node_id_type_ The node IDs of the neighbors matrix, the walks and the matches, @c u32_t
 *      or @c u64_t; the all-ones ID is the sentinel. The shared kernels cover @c u32_t alone, so
 *      the refinement, the absorb and the audit answer @c unsupported_k over any other.
 */
template <typename derived_, typename node_id_type_>
struct vram_index {
    using traversal_context_t = cuda_traversal_context<cuda_arena_allocator_t>;
    using node_id_t = node_id_type_;
    using distance_t = distance_t;
    using small_size_t = u32_t;

    static_assert(sizeof(std::size_t) == sizeof(u64_t), "The caller's offsets cross as the device's words");

    /** The node-ID space, less the sentinel. */
    inline static constexpr std::size_t max_node_count_k = std::size_t {std::numeric_limits<node_id_t>::max()} - 1u;

    /** The strongest @c synchronization_t contract this design honors; nothing routes on it. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;

    /** Pins the index to the storage allocator's device, its kits scoring under @p kernel where
     *  they cover several metrics; the first @c reserve freezes the rest. */
    explicit vram_index(nk_kernel_kind_t kernel, cuda_virtual_memory_allocator<byte_t> const& allocator = {}) noexcept
        : device_ {allocator.device()}, kernel_ {kernel}, values_tape_(allocator), offsets_column_(allocator),
          neighbors_matrix_(allocator), grown_((cuda_set_device(allocator.device()), cuda_event_t {})) {}
    vram_index(vram_index const&) = delete;
    vram_index& operator=(vram_index const&) = delete;

#pragma region Measures

    /** Rows visible to searches: the watermark every walk samples below and filters neighbors
     *  by. */
    std::size_t size() const noexcept {
        return standard_atomic_ref<std::size_t const>(last_visible_id_).load(std::memory_order_acquire);
    }

    /** The first failure that left claimed rows unpublished, kept for the index's lifetime. */
    status_t status() const noexcept {
        return standard_atomic_ref<status_t const>(status_).load(std::memory_order_acquire);
    }

    /** Neighbors per row, zero until the first @c reserve freezes it. */
    std::size_t connectivity() const noexcept {
        return standard_atomic_ref<std::size_t const>(connectivity_).load(std::memory_order_acquire);
    }

    /** Rows mapped across the offsets column and the neighbors matrix, the smaller of their
     *  counts. */
    std::size_t capacity() const noexcept {
        std::size_t const row_bytes = connectivity() * sizeof(node_id_t);
        std::size_t const offsets_rows = offsets_column_.committed_bytes() / sizeof(u64_t);
        if (!row_bytes || !offsets_rows) return 0;
        return min_of(offsets_rows - 1, neighbors_matrix_.committed_bytes() / row_bytes);
    }

    /** Value bytes mapped in the tape, less the margin past the last. */
    std::size_t bytes_capacity() const noexcept {
        std::size_t const committed = values_tape_.committed_bytes();
        return committed > tape_margin_k ? committed - tape_margin_k : 0;
    }

    /** Bytes this index occupies on the device and the host; traversal scratch is the
     *  executor's. */
    std::size_t memory_usage() const noexcept {
        return sizeof(derived_) + values_tape_.committed_bytes() + offsets_column_.committed_bytes() +
               neighbors_matrix_.committed_bytes();
    }

    /** The pinned device's identity and limits, queried once at the first @c reserve. */
    cuda_device_specs_t const& device_specs() const noexcept { return device_; }

    /** The metric the kits score under where they cover several, unknown otherwise. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Device-side storage views, the audit surface for tests. */
    vram_ragged_storage<node_id_t> storage() const noexcept {
        return {values_tape_.base(),
                reinterpret_cast<u64_t*>(offsets_column_.base()),
                reinterpret_cast<node_id_t*>(neighbors_matrix_.base()),
                static_cast<u32_t>(connectivity()),
                kernel_,
                parameters_};
    }

#pragma endregion Measures

#pragma region Reservation

    /** Requested geometry & capacity for a @c reserve() call; the connectivity is frozen by the
     *  first success. */
    struct reserve_capacity_t {
        std::size_t capacity = 0;
        connectivity_t connectivity = connectivity_32_k;

        /** Value bytes across every row to host. */
        std::size_t bytes = 0;
    };

    /** Ensures the index can host at least @p config.capacity rows of @p config.bytes value bytes
     *  in all, and answers the row capacity now in effect; the first success binds the device,
     *  loads the kernels and reserves the address ranges, later ones only map. */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (!std::has_single_bit(static_cast<u32_t>(config.connectivity))) return {0, status_t::config_mismatch_k};
        if (config.capacity > max_node_count_k) return {capacity(), status_t::overflow_k};
        std::size_t const frozen = connectivity();
        if (frozen && frozen != config.connectivity) return {capacity(), status_t::config_mismatch_k};
        if (!frozen) {
            if (cuda_set_device(device_.ordinal) != cuda_success_k) return {0, status_t::device_error_k};
            auto const [specs, status] = cuda_device_specs_t::query(device_.ordinal);
            if (failed(status)) return {0, status};
            device_ = specs;
            if (status_t const loaded = derived().load_kernels_(device_.ordinal); failed(loaded)) return {0, loaded};
            if (status_t const loaded = load_graph_kernels_(); failed(loaded)) return {0, loaded};
            // Address space is free, so the ceilings are generous and ignore the first capacity.
            std::size_t const budget_bytes = 4 * device_.vram_bytes;
            std::size_t const row_bytes = config.connectivity * sizeof(node_id_t) + sizeof(u64_t);
            std::size_t const ceiling_rows = min_of(max_node_count_k,
                                                    max_of<std::size_t>(config.capacity, budget_bytes / row_bytes));
            // The offsets column holds one start past the last row.
            if (offsets_column_.reserve_range((ceiling_rows + 1) * sizeof(u64_t)) != status_t::success_k ||
                neighbors_matrix_.reserve_range(ceiling_rows * config.connectivity * sizeof(node_id_t)) !=
                    status_t::success_k ||
                values_tape_.reserve_range(max_of(config.bytes, budget_bytes) + tape_margin_k) != status_t::success_k)
                return {0, status_t::bad_alloc_k};
            standard_atomic_ref<std::size_t>(connectivity_).store(config.connectivity, std::memory_order_release);
        }
        if (status_t const mapped = map_rows_(config.capacity); failed(mapped)) return {capacity(), mapped};
        status_t const mapped_bytes = values_tape_.commit(config.bytes + tape_margin_k);
        return {capacity(), mapped_bytes};
    }

#pragma endregion Reservation

#pragma region Insertion

    /** Optional tuning knobs; zeros mean "derive from connectivity". */
    struct insert_config_t {

        /** Breadth of the construction candidate pool, rounded up to the register pools of 64 to
         *  512 ranks and walked exactly past them. */
        small_size_t expansion_factor = 0;
    };

    /** Telemetry of one @c insert() batch, whose IDs are @b [first_id,new_size). */
    struct [[nodiscard]] insert_result_t {
        std::size_t new_size = 0;
        node_id_t first_id = std::numeric_limits<node_id_t>::max();
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        std::size_t tail_drops = 0;
        std::size_t visited_saturations = 0;
    };

    /** The bytes an executor's run must hold before a context carved over it can insert the rows
     *  @p values and @p offsets lay out, one aligned term per carve the verb makes; zero before
     *  @c reserve freezes the geometry. */
    std::size_t traversal_memory_usage(byte_t const* values, std::span<std::size_t const> offsets,
                                       insert_config_t config = {}) const noexcept {
        if (!connectivity() || !device_.multiprocessor_count || offsets.empty()) return 0;
        std::size_t const rows = offsets.size() - 1;
        std::size_t longest = longest_row_;
        u32_t const span_bytes = static_cast<u32_t>(max_of(span_bytes_, span_bytes_of_(values, offsets, longest)));
        std::size_t const rung = derived_::rung_serving_(longest);
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   cuda_resident_blocks(vram_kernel_t::insert_k, 0) * vram_warps_per_block_k;
        return aligned_(sizeof(vram_telemetry_block_t)) + 2 * aligned_(sizeof(u32_t)) +
               aligned_(slices * build_arena_shape_(config, span_bytes, rung).template slice_bytes<node_id_t>()) +
               aligned_(build_staging_bytes_(rows, claimed_() + rows));
    }

    /** Inserts the rows @p values and @p offsets lay out, row @c i spanning bytes @c offsets[i] up
     *  to @c offsets[i+1] past @p values, as the contiguous IDs @b [first_id,first_id+rows), in
     *  admission waves riding the context's stream, then drains it and turns the watermark over
     *  the range. Another batch in flight answers @c contended_k. */
    expected<insert_result_t> insert(byte_t const* values, std::span<std::size_t const> offsets,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        std::size_t const rows = offsets.empty() ? 0 : offsets.size() - 1;
        if (!rows) return {{claimed_(), static_cast<node_id_t>(claimed_())}, status_t::success_k};
        if (!connectivity()) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        cuda_stream_handle_t const stream = context.stream();

        // Walks prepare stored rows as much as the batch's, so the rung and spans cover both.
        std::size_t longest = longest_row_;
        u32_t const span_bytes = static_cast<u32_t>(max_of(span_bytes_, span_bytes_of_(values, offsets, longest)));
        std::size_t const rung = derived_::rung_serving_(longest);
        if (rung == derived_::rungs_k) return {{}, status_t::unsupported_k};
        std::size_t const batch_bytes = offsets[rows] - offsets[0];

        // Held until the drain: a wave's merges rewrite rows another batch's walks would read.
        auto const [first_id, claim_status] = claim_(rows);
        if (failed(claim_status)) return {{}, claim_status};
        std::size_t const total_rows = first_id + rows;
        auto const refuse = [&](status_t failure) noexcept -> expected<insert_result_t> {
            unclaim_(first_id, rows);
            return {{}, failure};
        };

        // Every carve and growth precedes the first launch, so a refusal publishes nothing.
        gpu_arena_shape_t const arena_shape = build_arena_shape_(config, span_bytes, rung);
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   cuda_resident_blocks(vram_kernel_t::insert_k, 0) * vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(slices * arena_shape.template slice_bytes<node_id_t>());
        byte_t* const staging = context.ensure_staging(build_staging_bytes_(rows, total_rows));
        if (!arena || !staging) return refuse(status_t::bad_alloc_k);
        if (status_t const grown = grow_rows_(total_rows, stream); failed(grown)) return refuse(grown);
        if (status_t const mapped = values_tape_.commit(tape_bytes_ + batch_bytes + tape_margin_k); failed(mapped))
            return refuse(mapped);
        if (cuda_copy_async(values_tape_.base() + tape_bytes_, values + offsets[0], batch_bytes, stream) !=
                cuda_success_k ||
            cuda_copy_async(staging, offsets.data(), (rows + 1) * sizeof(u64_t), stream) != cuda_success_k ||
            cuda_fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream) != cuda_success_k)
            return refuse(status_t::device_error_k);
        // Every row walks twice, to insert and to re-select, each walk under an epoch of its own.
        auto const [epoch_base, epochs_status] = context.take_epochs(2 * rows, arena_bytes_before);
        if (failed(epochs_status)) return refuse(epochs_status);

        vram_build<node_id_t> build;
        build.storage = storage();
        build.staged_offsets = reinterpret_cast<u64_t const*>(staging);
        build.tape_first = tape_bytes_;
        build.scratch_bytes_per_warp = arena_shape.cache_bytes ? 0u : 2 * span_bytes;
        build.span_bytes = span_bytes;
        build.arena = arena;
        build.arena_shape = arena_shape;
        build.expansion = config.expansion_factor;
        build.epoch_base = epoch_base;
        build.telemetry = context.telemetry_.get();
        build.assigned_rows = context.assigned_rows_.get();
        // Every chain starts empty, and each back-link launch empties the ones it merged.
        build.link_heads = reinterpret_cast<node_id_t*>(staging + offsets_bytes_(rows));
        build.link_next = build.link_heads + total_rows;
        build.touched_ids = build.link_next + rows * connectivity();
        build.touched_count = reinterpret_cast<u32_t*>(build.touched_ids + rows * connectivity());
        if (cuda_fill_async(build.link_heads, 0xFF, total_rows * sizeof(node_id_t), stream) != cuda_success_k)
            return refuse(status_t::device_error_k);
        status_t const launched = launch_insert_(build, offsets, first_id, gpu_pool_slots(config.expansion_factor, 4),
                                                 rung, stream);
        if (failed(launched)) return refuse(launched);

        tape_bytes_ += batch_bytes;
        longest_row_ = longest;
        span_bytes_ = span_bytes;
        if (status_t const drained = context.drain(); failed(drained)) return {{}, fail_(drained)};
        standard_atomic_ref<std::size_t>(last_visible_id_).store(total_rows, std::memory_order_release);

        vram_telemetry_block_t const spent = context.spent();
        insert_result_t result;
        result.new_size = total_rows;
        result.first_id = first_id;
        result.visited_members = spent.hops;
        result.computed_distances = spent.scored;
        result.restarts = spent.restarts;
        result.tail_drops = spent.tail_drops;
        result.visited_saturations = spent.visited_saturations;
        return {result, status_t::success_k};
    }

#pragma endregion Insertion

#pragma region Search

    /** Optional tuning knobs of @c search(); zeros mean "derive from connectivity". */
    struct search_config_t {

        /** Breadth of the candidate pool, walked as given when it covers @c count and widened to
         *  twice @c count otherwise. */
        small_size_t expansion_factor = 0;

        /** Matches reported per query. */
        small_size_t count = 10;
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

    /** The bytes an executor's run must hold before a context carved over it can search the queries
     *  @p queries and @p offsets lay out under @p config, one aligned term per carve. */
    std::size_t traversal_memory_usage(byte_t const* queries, std::span<std::size_t const> offsets,
                                       search_config_t config) const noexcept {
        if (!connectivity() || !device_.multiprocessor_count || offsets.empty()) return 0;
        std::size_t const queries_count = offsets.size() - 1;
        std::size_t longest = 0;
        u32_t const span_bytes = static_cast<u32_t>(span_bytes_of_(queries, offsets, longest));
        std::size_t const rung = derived_::rung_serving_(longest);
        gpu_arena_shape_t const arena_shape = walk_arena_shape_(config, span_bytes, rung);
        u32_t const slots = arena_shape.tail_entries ? 0u : walk_slots_(config);
        std::size_t const slices = std::size_t {walk_blocks_(queries_count, rung, slots)} * vram_warps_per_block_k;
        return aligned_(sizeof(vram_telemetry_block_t)) + 2 * aligned_(sizeof(u32_t)) +
               aligned_(slices * arena_shape.template slice_bytes<node_id_t>()) +
               aligned_(search_staging_bytes_(queries_count, offsets[queries_count] - offsets[0], config.count));
    }

    /**
     *  @brief Finds the nearest published neighbors of every query, ascending by distance with
     *      sentinel IDs marking absent matches. Safe to overlap with insertions on other streams.
     *
     *  @param[in] queries The bytes query @c i spans from @c offsets[i] up to @c offsets[i+1].
     *  @param[out] result_ids IDs of the matches, @b [queries,count], in host or device memory.
     *  @param[out] result_distances Distances of the same matches, in host or device memory.
     *  @param[inout] context The stream every transfer and the launch ride; drained before
     *      returning.
     */
    expected<search_result_t> search(byte_t const* queries, std::span<std::size_t const> offsets,
                                     nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        std::size_t const queries_count = offsets.empty() ? 0 : offsets.size() - 1;
        if (!queries_count) return {{}, status_t::success_k};
        if (!connectivity()) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (result_ids.extent(0) != queries_count || result_ids.extent(1) < config.count ||
            result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)
            return {{}, status_t::config_mismatch_k};
        std::size_t const visible = size();
        std::size_t longest = 0;
        u32_t const span_bytes = static_cast<u32_t>(span_bytes_of_(queries, offsets, longest));
        std::size_t const rung = derived_::rung_serving_(longest);
        if (rung == derived_::rungs_k) return {{}, status_t::unsupported_k};
        cuda_stream_handle_t const stream = context.stream();

        // The staging holds the queries' bytes from a word boundary, their offsets, the landing.
        std::size_t const batch_bytes = offsets[queries_count] - offsets[0];
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
        if (cuda_copy_async(staging, queries + offsets[0], batch_bytes, stream) != cuda_success_k ||
            cuda_copy_async(staged_offsets, offsets.data(), (queries_count + 1) * sizeof(u64_t), stream) !=
                cuda_success_k ||
            cuda_fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream) != cuda_success_k ||
            cuda_fill_async(context.assigned_rows_.get(), 0, sizeof(u32_t), stream) != cuda_success_k)
            return {{}, status_t::device_error_k};

        vram_arguments<node_id_t> arguments {};
        vram_search<node_id_t>& search = arguments.search;
        search.storage = storage();
        search.queries = staging;
        search.query_offsets = staged_offsets;
        search.queries_count = static_cast<u32_t>(queries_count);
        search.sampling_bound = static_cast<node_id_t>(visible);
        search.scratch_bytes_per_warp = arena_shape.cache_bytes ? 0u : span_bytes;
        search.expansion = walk_expansion_(config);
        search.arena = arena;
        search.arena_shape = arena_shape;
        search.epoch_base = epoch_base;
        search.telemetry = context.telemetry_.get();
        search.assigned_queries = context.assigned_rows_.get();
        search.result_ids = device_ids;
        search.result_distances = device_distances;
        search.count = config.count;
        if (status_t const launched = cuda_launch(derived().kernel_function_(rung, vram_kernel_t::search_k, slots),
                                                  blocks, block_threads_k,
                                                  walkers_per_block_(rung) * search.scratch_bytes_per_warp, stream,
                                                  cuda_overlap_t::serialized_k, arguments);
            failed(launched))
            return {{}, launched};
        std::size_t const row_bytes = config.count * sizeof(node_id_t);
        if (cuda_copy_2d_async(result_ids.data(), result_ids.stride_bytes(0), device_ids, row_bytes, row_bytes,
                               queries_count, stream) != cuda_success_k ||
            cuda_copy_2d_async(result_distances.data(), result_distances.stride_bytes(0), device_distances,
                               config.count * sizeof(distance_t), config.count * sizeof(distance_t), queries_count,
                               stream) != cuda_success_k)
            return {{}, status_t::device_error_k};
        if (status_t const drained = context.drain(); failed(drained)) return {{}, drained};
        vram_telemetry_block_t const spent = context.spent();
        search_result_t result;
        result.count = config.count;
        result.visited_members = spent.hops;
        result.computed_distances = spent.scored;
        result.restarts = spent.restarts;
        result.tail_drops = spent.tail_drops;
        result.visited_saturations = spent.visited_saturations;
        return {result, status_t::success_k};
    }

#pragma endregion Search

#pragma region Refinement

    /** Optional tuning knobs of @c refine(); zeros mean "derive from connectivity". */
    struct refine_config_t {

        /** Breadth of the re-selection pool, as an insertion's. */
        small_size_t expansion_factor = 0;
    };

    using refine_result_t = basic::refine_result_t;
    using health_result_t = utilities::health_result_t;

  private:

    /** Sweeps an audit numbers from one and stops below: every row joins one frontier at most, so
     *  the levels are bounded by the landmarks' eccentricity, capped defensively. */
    inline static constexpr std::size_t audit_sweeps_cap_k = 256;

    /** Device scratch an audit carves over the rows it judges: the reached set and two frontiers,
     *  then the rows each sweep expanded at the sweep's number, @b [audit_sweeps_cap_k], whose slot
     *  zero, which no sweep numbers, counts the levels that expanded anything, then the unreached
     *  and reconnected counters an audit lands together, and for a refine the unreached rows' IDs
     *  and their in-link counts, @b [rows] each. */
    struct audit_scratch_t {
        typename traversal_context_t::bytes_ptr_t bytes;
        vram_bitset_t reached, frontier, next_frontier;
        u32_t* expanded = nullptr;
        u32_t* unreached = nullptr;
        u32_t* reconnected = nullptr;
        u32_t* island_ids = nullptr;
        u32_t* in_links = nullptr;
        std::size_t words = 0;
    };

    /** Which share of the scratch a carve covers: the audit's alone, or a refine's with its
     *  heal. */
    enum class audit_use_t : u8_t { health_k, refine_k };

    /** The counters an audit lands in page-locked memory, in the order of the device's: the levels
     *  that expanded, the last sweep's count, which only a sweep short of convergence leaves
     *  nonzero, then the unreached and reconnected counters. */
    struct audit_landing_t {
        u32_t levels;
        u32_t last_expanded;
        u32_t unreached;
        u32_t reconnected;
    };

    /** Gives a landing back to the stream's slots it came from. */
    struct landing_return_t {
        cuda_landings_t* landings = nullptr;
        void operator()(audit_landing_t* landing) const noexcept { landings->give_back(landing); }
    };
    using audit_landing_ptr_t = std::unique_ptr<audit_landing_t, landing_return_t>;
    static_assert(sizeof(audit_landing_t) <= cuda_landings_t::slot_bytes_k, "An audit's counters fit one slot");

    /** One part of a split as the host deals it: its rows and the seams leading them, its value
     *  bytes and longest row, where its receiver's offsets column starts in the landing, and the
     *  next seam and interior positions while the slots are dealt. */
    struct split_part_t {
        std::size_t rows = 0, seams = 0, bytes = 0, longest = 0, offsets_first = 0;
        std::size_t next_seam = 0, next_interior = 0;
    };

    /** A split's state beside the refinement it turns into, empty for a refine or an absorb. */
    struct split_t {

        /** The index the split empties, while it still holds its rows. */
        vram_index* source = nullptr;

        /** The caller's receiver slots, one part each, the ones whose claim words the plan closed,
         *  counted from the first, and the one the refinement is bound to. */
        std::span<derived_*> receivers;
        std::size_t closed_receivers = 0;
        std::size_t bound_receiver = 0;

        /** The source's rows at planning, and the most rows one part takes: those over the
         *  receivers, rounded up. */
        std::size_t rows = 0, part_capacity = 0;

        /** The device scratch: each row's part, which the scatter reads back as its slot, the
         *  round's proposals, @b [rows] each, the two frontiers and the seams, one bit per row
         *  each, then per part its size and its round's tally, the rows each round placed,
         *  @b [audit_sweeps_cap_k], the parts' roots, and the receivers' bases. */
        typename traversal_context_t::bytes_ptr_t bytes;
        u32_t* labels = nullptr;
        u32_t* proposals = nullptr;
        vram_bitset_t frontier {}, next_frontier {}, seams {};
        u32_t* part_sizes = nullptr;
        u32_t* proposed = nullptr;
        u32_t* placed = nullptr;
        u32_t* roots = nullptr;
        byte_t** part_bases = nullptr;

        /** Page-locked host bytes: the labels, dealt into slots in place, the seams, then from
         *  @c landed_offsets the receivers' offsets columns back to back, then the bases and the
         *  roots planning uploads. */
        growing_buffer<byte_t, cuda_pinned_allocator_t> landing {cuda_pinned_allocator_t {}};
        std::size_t landed_offsets = 0;

        /** The source's row starts into its tape, landed at planning, @b [rows+1], and the parts
         *  as the host deals them. */
        growing_buffer<u64_t> source_offsets {global_allocator};
        growing_buffer<split_part_t> parts {global_allocator};

        /** The audits and islands of the receivers healed so far, summed, and whether a stop cut
         *  the re-selection, which the later receivers then skip. */
        health_result_t health {};
        std::size_t islands = 0;
        bool stopped = false;

        explicit split_t(typename traversal_context_t::bytes_ptr_t&& empty) noexcept : bytes(std::move(empty)) {}
        split_t(split_t&& other) noexcept
            : source(std::exchange(other.source, nullptr)), receivers(other.receivers),
              closed_receivers(std::exchange(other.closed_receivers, 0)), bound_receiver(other.bound_receiver),
              rows(other.rows), part_capacity(other.part_capacity), bytes(std::move(other.bytes)), labels(other.labels),
              proposals(other.proposals), frontier(other.frontier), next_frontier(other.next_frontier),
              seams(other.seams), part_sizes(other.part_sizes), proposed(other.proposed), placed(other.placed),
              roots(other.roots), part_bases(other.part_bases), landing(std::move(other.landing)),
              landed_offsets(other.landed_offsets), source_offsets(std::move(other.source_offsets)),
              parts(std::move(other.parts)), health(other.health), islands(other.islands), stopped(other.stopped) {}
        split_t& operator=(split_t&&) = delete;
    };

  public:

    /**
     *  @brief One refinement in flight on this index's device: @ref queue a phase, @ref wait for
     *      it, @ref advance past it, and @ref finish once no phase follows.
     *
     *  The phases re-select every node of the planned range in one walk launch and the back-link
     *  launch behind it, then audit reachability, then heal round by round while the islands
     *  shrink, reconnecting each unreached row into a reached pick or the landmark covering it.
     *  Planning made every allocation and closed the claim word, which the plan's death reopens, so
     *  batches answer @c contended_k meanwhile while searches on other streams run throughout. A
     *  phase only queues, and only a device fault fails one; a stopped sequence of phases leaves
     *  the graph correct and merely less improved.
     *
     *  A split's plan first partitions and scatters, as @ref plan_split describes, then binds the
     *  receivers one after another, each refined over its seams and healed, its tallies summing
     *  into the plan's; it closed every receiver's claim word and the source's, and its death
     *  reopens every one still closed.
     */
    struct refine_plan_t {

        /** An unbound plan over @p scratch - what a refused plan answers with. */
        explicit refine_plan_t(audit_scratch_t&& scratch) noexcept
            : scratch_(std::move(scratch)),
              split_(typename traversal_context_t::bytes_ptr_t {nullptr, scratch_.bytes.get_deleter()}) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : scratch_(std::move(other.scratch_)), split_(std::move(other.split_)),
              index_(std::exchange(other.index_, nullptr)), context_(std::exchange(other.context_, nullptr)),
              landing_(std::move(other.landing_)), config_(other.config_), nodes_(other.nodes_), rows_(other.rows_),
              walked_(other.walked_), islands_(other.islands_), phase_(other.phase_), status_(other.status_),
              tally_(other.tally_) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** Reopens the claim words the plan closed, whether it finished or not. */
        ~refine_plan_t() noexcept {
            if (index_ && split_.receivers.empty()) index_->reopen_();
            for (std::size_t receiver = 0; receiver != split_.closed_receivers; ++receiver)
                split_.receivers[receiver]->reopen_();
            if (split_.source) split_.source->reopen_();
        }

        /** Queues the current phase on the context's stream, then the landing of its counters and
         *  telemetry behind the context's event; never waits and never allocates. */
        status_t queue() noexcept {
            if (!context_) return status_t::uninitialized_k;
            if (failed(status_)) return status_;
            cuda_stream_handle_t const stream = context_->stream();
            switch (phase_) {
            // Planning queued the partition, so its phase only marks the landing.
            case phase_t::partition_k: break;
            case phase_t::scatter_k: status_ = split_.source->queue_scatter_(split_, stream); break;
            case phase_t::reselect_k:
                walked_ = nodes_.size();
                status_ = index_->reselect_(*context_, config_, nodes_, rows_);
                break;
            case phase_t::audit_k: status_ = index_->queue_audit_(scratch_, landing_.get(), rows_, stream); break;
            case phase_t::heal_k:
                // Only island rows gain links, and no reached row points at one, so the audit
                // before stands.
                status_ = index_->queue_reconnect_(scratch_, islands_, rows_, stream);
                if (succeeded(status_)) status_ = index_->queue_audit_(scratch_, landing_.get(), rows_, stream);
                break;
            default: return status_t::success_k;
            }
            if (succeeded(status_)) status_ = context_->record();
            return status_;
        }

        /** Sleeps until the queued phase landed, on the context's event. */
        status_t wait() noexcept {
            if (!context_ || phase_ == phase_t::done_k) return status_;
            if (succeeded(status_)) status_ = context_->event_.drain();
            return status_;
        }

        /** The barrier after a phase landed: reads its counters, picks the next phase, and answers
         *  whether one follows. */
        bool advance() noexcept {
            if (!context_ || failed(status_) || phase_ == phase_t::done_k) return false;
            if (phase_ == phase_t::partition_k) {
                status_ = deal_slots_();
                phase_ = phase_t::scatter_k;
                return succeeded(status_);
            }
            if (phase_ == phase_t::scatter_k) {
                publish_parts_();
                bind_receiver_();
                return phase_ != phase_t::done_k;
            }
            if (phase_ == phase_t::reselect_k) {
                tally_.visited_nodes += walked_;
                phase_ = phase_t::audit_k;
                return true;
            }
            audit_landing_t const& landed = *landing_;
            auto const [health, audit_status] = audited_(landed, rows_);
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
            if (!healing && !split_.receivers.empty()) return settle_receiver_();
            return healing;
        }

        /** Skips the re-selection when it has not been queued yet, so the next phase audits and
         *  heals, and a split's later receivers skip theirs too; a no-op in any other phase, a
         *  split's partition and scatter included. */
        void stop() noexcept {
            if (phase_ == phase_t::reselect_k) phase_ = phase_t::audit_k, split_.stopped = true;
        }

        /** Whether the current phase re-selects the planned range - the phase a caller's
         *  progress counts. */
        bool reselecting() const noexcept { return phase_ == phase_t::reselect_k; }

        /** Whether the current phase moves a split's rows - the phases no stop skips. */
        bool landing() const noexcept { return phase_ == phase_t::partition_k || phase_ == phase_t::scatter_k; }

        /** Where every row of a split's source goes, by its source ID: its part times
         *  @ref part_capacity, plus its row in that part's receiver; empty until the partition
         *  landed, and for a refine or an absorb. The caller moves the keys by it. */
        std::span<u32_t const> slots() const noexcept {
            if (!split_.part_capacity || phase_ == phase_t::partition_k || failed(status_)) return {};
            return {reinterpret_cast<u32_t const*>(split_.landing.data()), split_.rows};
        }

        /** The most rows one part of a split takes, the stride of @ref slots; zero otherwise. */
        std::size_t part_capacity() const noexcept { return split_.part_capacity; }

        /** Nodes re-selected so far. */
        std::size_t visited() const noexcept { return tally_.visited_nodes; }

        /** The pass: its telemetry read from the last landing, complete once @ref advance answered
         *  false, otherwise @c cancelled_k; a failure a phase kept answers instead. */
        expected<refine_result_t> finish() const noexcept {
            if (!context_) return {{}, status_t::uninitialized_k};
            refine_result_t result = tally_;
            vram_telemetry_block_t const spent = context_->spent();
            result.changed_nodes = min_of<std::size_t>(spent.changed_nodes, result.visited_nodes);
            result.telemetry = {spent.scored, spent.hops, spent.restarts, spent.visited_saturations};
            if (failed(status_)) return {result, status_};
            return {result, phase_ == phase_t::done_k ? status_t::success_k : status_t::cancelled_k};
        }

      private:
        friend struct vram_index;
        enum class phase_t : u8_t { partition_k, scatter_k, reselect_k, audit_k, heal_k, done_k };

        /** Deals every source row its slot in place in the landing, each part's seams first and
         *  both runs in ID order, lays out the receivers' offsets columns, and maps each receiver's
         *  tape for its part, the one growth past planning. */
        status_t deal_slots_() noexcept {
            std::size_t const rows = split_.rows, parts = split_.receivers.size();
            u32_t* const slots = reinterpret_cast<u32_t*>(split_.landing.data());
            u32_t const* const seams = slots + rows;
            u64_t* const offsets = reinterpret_cast<u64_t*>(split_.landing.data() + split_.landed_offsets);
            auto const seam = [&](std::size_t row) noexcept { return (seams[row / 32] >> (row % 32)) & 1u; };
            for (std::size_t part = 0; part != parts; ++part) split_.parts[part] = {};
            for (std::size_t row = 0; row != rows; ++row) {
                split_part_t& part = split_.parts[slots[row]];
                ++part.rows, part.seams += seam(row);
            }
            for (std::size_t part = 0, first = 0; part != parts; ++part) {
                split_.parts[part].next_interior = split_.parts[part].seams;
                split_.parts[part].offsets_first = first;
                first += split_.parts[part].rows + 1;
            }
            for (std::size_t row = 0; row != rows; ++row) {
                std::size_t const index = slots[row];
                split_part_t& part = split_.parts[index];
                std::size_t const position = seam(row) ? part.next_seam++ : part.next_interior++;
                std::size_t const bytes = split_.source_offsets[row + 1] - split_.source_offsets[row];
                offsets[part.offsets_first + position + 1] = bytes;
                part.bytes += bytes, part.longest = max_of(part.longest, bytes);
                slots[row] = static_cast<u32_t>(index * split_.part_capacity + position);
            }
            for (std::size_t index = 0; index != parts; ++index) {
                split_part_t const& part = split_.parts[index];
                u64_t* const column = offsets + part.offsets_first;
                column[0] = 0;
                for (std::size_t position = 0; position != part.rows; ++position)
                    column[position + 1] += column[position];
                vram_index& receiver = *split_.receivers[index];
                if (status_t const mapped = receiver.values_tape_.commit(part.bytes + tape_margin_k); failed(mapped))
                    return mapped;
            }
            return status_t::success_k;
        }

        /** Publishes every receiver's rows at once, its tape's extent and longest row its part's
         *  and its spans the source's, then empties the source and reopens its claim word at
         *  zero. */
        void publish_parts_() noexcept {
            vram_index& source = *std::exchange(split_.source, nullptr);
            for (std::size_t index = 0; index != split_.receivers.size(); ++index) {
                split_part_t const& part = split_.parts[index];
                vram_index& receiver = *split_.receivers[index];
                receiver.tape_bytes_ = part.bytes, receiver.longest_row_ = part.longest;
                receiver.span_bytes_ = source.span_bytes_;
                standard_atomic_ref<std::size_t>(receiver.last_visible_id_).store(part.rows, std::memory_order_release);
            }
            source.tape_bytes_ = 0, source.longest_row_ = 0, source.span_bytes_ = 0, source.filled_rows_ = 0;
            standard_atomic_ref<std::size_t>(source.last_visible_id_).store(0, std::memory_order_release);
            source.reopen_();
        }

        /** Turns the pass to the next receiver holding rows: its seams as the range, or straight
         *  the audit when it has none or a stop cut the pass; past the last, reports the sums. */
        void bind_receiver_() noexcept {
            while (split_.bound_receiver != split_.receivers.size() && !split_.parts[split_.bound_receiver].rows)
                ++split_.bound_receiver;
            if (split_.bound_receiver == split_.receivers.size()) {
                tally_.health = split_.health, tally_.islands_found = split_.islands;
                phase_ = phase_t::done_k;
                return;
            }
            split_part_t const& part = split_.parts[split_.bound_receiver];
            index_ = split_.receivers[split_.bound_receiver];
            rows_ = part.rows;
            nodes_ = {0, static_cast<node_id_t>(part.seams)};
            islands_ = 0;
            phase_ = split_.stopped || nodes_.empty() ? phase_t::audit_k : phase_t::reselect_k;
        }

        /** Folds a healed receiver's audit and islands into the split's, then binds the next one,
         *  answering whether a phase follows. */
        bool settle_receiver_() noexcept {
            split_.health += tally_.health;
            split_.islands += tally_.islands_found;
            ++split_.bound_receiver;
            bind_receiver_();
            return phase_ != phase_t::done_k;
        }

        audit_scratch_t scratch_;
        split_t split_;
        vram_index* index_ = nullptr;
        traversal_context_t* context_ = nullptr;
        audit_landing_ptr_t landing_ {nullptr, landing_return_t {}};
        refine_config_t config_ {};

        /** The IDs the re-selection walks, and the rows the scratch covers. */
        iota_view<node_id_t> nodes_;
        std::size_t rows_ = 0;

        /** Rows the queued phase walks, and the islands the last audit left. */
        std::size_t walked_ = 0;
        u32_t islands_ = 0;
        phase_t phase_ = phase_t::done_k;
        status_t status_ = status_t::success_k;
        refine_result_t tally_;
    };

    /**
     *  @brief Opens a refinement of the nodes in @p nodes, clamped to the published IDs, on
     *      @p context 's stream: closes the claim word, carves the scratch, grows the context's
     *      arena and staging for the walk, and takes a page-locked landing.
     *
     *  No phase allocates. An empty range heals alone. Each re-selection walks from the row's own
     *  neighbors and from landmarks over every published row, so a range refines against the whole
     *  graph.
     *
     *  @retval unsupported_k The shared kernels lack this node-ID type.
     *  @retval contended_k A batch, a load or another plan in flight.
     */
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                        refine_config_t config = {}) noexcept {
        refine_plan_t plan = unbound_plan_(context);
        if (status_t const refused = graph_refusal_(context); failed(refused)) return {std::move(plan), refused};
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        plan.index_ = this;
        std::size_t const rows = size();
        node_id_t const bound = static_cast<node_id_t>(min_of<std::size_t>(nodes.bound, rows));
        status_t const opened = open_refine_(plan, {min_of(nodes.first, bound), bound}, rows, longest_row_, span_bytes_,
                                             context, config);
        return {std::move(plan), opened};
    }

    /** One @ref refine_plan_t driven to its end on the calling thread, waiting once per phase. */
    expected<refine_result_t> refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                     refine_config_t config = {}) noexcept {
        auto [plan, status] = plan_refine(nodes, context, config);
        if (failed(status)) return {{}, status};
        do
            if (status_t const landed = plan.queue(); failed(landed) || failed(plan.wait())) break;
        while (plan.advance());
        return plan.finish();
    }

    /**
     *  @brief Opens an absorb of @p donors: lands every row of each past the watermark on this
     *      device, one donor after another, empties them, and answers the refinement re-selecting
     *      every node and healing.
     *
     *  Every allocation and every landing happens here, behind one wait: each donor's values copy
     *  past this tape, its offsets shift by this tape's bytes on the host, and its neighbor rows
     *  copy and rebase onto the IDs past the watermark on the device. The watermark turns over the
     *  landed rows before the plan returns, and the claim word stays closed until the plan dies, so
     *  searches see the landed parts at once and walk across them as the re-selection links them.
     *  Donors must share this index's device, connectivity and metric, and nothing may search them
     *  meanwhile.
     *
     *  @param[inout] donors The indexes giving their rows up, each emptied once landed.
     *  @retval config_mismatch_k A null donor, this index itself, or one of another device,
     *      connectivity or metric.
     *  @retval contended_k A batch, a load or a plan in flight on this index or on a donor.
     *  @retval overflow_k Rows past @c max_node_count_k.
     */
    expected<refine_plan_t> plan_absorb(std::span<derived_*> donors, traversal_context_t& context,
                                        refine_config_t config = {}) noexcept {
        refine_plan_t plan = unbound_plan_(context);
        if (status_t const refused = graph_refusal_(context); failed(refused)) return {std::move(plan), refused};
        std::size_t landed_rows = 0, landed_bytes = 0, longest = longest_row_, span_bytes = span_bytes_;
        for (derived_* const donor : donors) {
            if (!donor || donor == &derived()) return {std::move(plan), status_t::config_mismatch_k};
            vram_index const& source = *donor;
            if (!source.size()) continue;
            if (source.device_.ordinal != device_.ordinal || source.connectivity() != connectivity() ||
                source.kernel_ != kernel_)
                return {std::move(plan), status_t::config_mismatch_k};
            if (status_t const kept = source.status(); failed(kept)) return {std::move(plan), kept};
            landed_rows += source.size(), landed_bytes += source.tape_bytes_;
            longest = max_of(longest, source.longest_row_), span_bytes = max_of(span_bytes, source.span_bytes_);
        }
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        plan.index_ = this;
        std::size_t const base = size();
        if (landed_rows > max_node_count_k - base) return {std::move(plan), status_t::overflow_k};
        // Donors close one by one, and a refusal reopens the ones closed before it.
        std::size_t closed_donors = 0;
        auto const reopen_donors = [&]() noexcept {
            for (std::size_t donor = 0; donor != closed_donors; ++donor) donors[donor]->reopen_();
        };
        for (; closed_donors != donors.size(); ++closed_donors)
            if (status_t const closed = donors[closed_donors]->close_(); failed(closed)) {
                reopen_donors();
                return {std::move(plan), closed};
            }
        std::size_t const rows = base + landed_rows;
        cuda_stream_handle_t const stream = context.stream();
        growing_buffer<u64_t> shifted_offsets {global_allocator};
        status_t landed = open_refine_(plan, {0, static_cast<node_id_t>(rows)}, rows, longest, span_bytes, context,
                                       config);
        if (succeeded(landed)) landed = grow_rows_(rows, stream);
        if (succeeded(landed)) landed = values_tape_.commit(tape_bytes_ + landed_bytes + tape_margin_k);
        if (succeeded(landed) && failed(shifted_offsets.resize_uninitialized(rows - base + 1)))
            landed = status_t::bad_alloc_k;

        // The landed rows publish only once every donor's copies drained, so a refusal publishes
        // none.
        std::size_t const row_bytes = connectivity() * sizeof(node_id_t);
        std::size_t into = base, tape_bytes = tape_bytes_;
        for (derived_* const donor : donors) {
            vram_index& source = *donor;
            std::size_t const count = source.size();
            if (failed(landed) || !count) continue;
            if (cuda_copy(shifted_offsets.data(), source.offsets_column_.base(), (count + 1) * sizeof(u64_t)) !=
                cuda_success_k) {
                landed = status_t::device_error_k;
                continue;
            }
            for (std::size_t row = 0; row <= count; ++row) shifted_offsets[row] += tape_bytes;
            vram_graph<node_id_t> graph {};
            graph.neighbors = reinterpret_cast<node_id_t*>(neighbors_matrix_.base());
            graph.connectivity = static_cast<u32_t>(connectivity());
            graph.first_id = static_cast<node_id_t>(into);
            graph.rows_count = static_cast<u32_t>(count);
            graph.source_bound = static_cast<node_id_t>(count);
            graph.shift = static_cast<node_id_t>(into);
            if (cuda_copy(offsets_column_.base() + into * sizeof(u64_t), shifted_offsets.data(),
                          (count + 1) * sizeof(u64_t)) != cuda_success_k ||
                (source.tape_bytes_ && cuda_copy_async(values_tape_.base() + tape_bytes, source.values_tape_.base(),
                                                       source.tape_bytes_, stream) != cuda_success_k) ||
                cuda_copy_async(neighbors_matrix_.base() + into * row_bytes, source.neighbors_matrix_.base(),
                                count * row_bytes, stream) != cuda_success_k)
                landed = status_t::device_error_k;
            else
                landed = launch_graph_(vram_kernel_t::rebase_links_k, graph_blocks_(vram_kernel_t::rebase_links_k),
                                       graph, stream);
            into += count, tape_bytes += source.tape_bytes_;
        }
        if (succeeded(landed)) landed = context.drain();
        if (failed(landed)) {
            // The rows past the watermark hold whatever landed, so the next grow fills them again.
            filled_rows_ = min_of(filled_rows_, base);
            reopen_donors();
            return {std::move(plan), landed};
        }
        for (derived_* const donor : donors) {
            vram_index& source = *donor;
            source.tape_bytes_ = 0, source.longest_row_ = 0, source.span_bytes_ = 0, source.filled_rows_ = 0;
            standard_atomic_ref<std::size_t>(source.last_visible_id_).store(0, std::memory_order_release);
        }
        reopen_donors();
        tape_bytes_ = tape_bytes, longest_row_ = longest, span_bytes_ = span_bytes;
        standard_atomic_ref<std::size_t>(last_visible_id_).store(rows, std::memory_order_release);
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Opens a split of this index into @p receivers - the inverse of @ref plan_absorb: the
     *      plan's first phases move every row into one graph-local part per receiver, then refine
     *      every receiver in turn over the rows that lost a link, and heal.
     *
     *  Planning queues the partition behind every allocation: one root per part through
     *  @ref seeded_bijection under @p seed, then rounds of growth over the links, each part capped
     *  at the rows over the receivers rounded up, and each row a round reaches kept by its
     *  lowest-ID proposer, so equal seeds over equal graphs split alike; rows no round reached join
     *  the smallest parts in ID order, and the rows linking across mark the seams. The barrier
     *  after it deals every row its slot, each part's seams first, and maps each receiver's tape
     *  for its part, the one growth past planning, whose refusal leaves this index whole. The
     *  scatter moves every row's value and neighbors on the device, the links inside its part
     *  rebased and the links across dropped, and the barrier after it publishes every receiver at
     *  once and empties this index, whose claim word reopens at zero. Each receiver then re-selects
     *  its seams, audits and heals as @ref plan_refine does, and the tallies sum over the
     *  receivers.
     *
     *  The keys are the caller's, moved by the plan's @c slots once its partition landed.
     *  Receivers must share this index's device and metric, and either its connectivity or none
     *  yet, which planning then reserves; every claim word stays closed for the plan's life, and
     *  nothing may search this index meanwhile.
     *
     *  @param[inout] receivers The empty indexes taking the parts, at least two, all distinct.
     *  @param[in] seed Picks the parts' roots, so equal seeds over equal graphs split alike.
     *  @retval config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *      index itself, or one of another device, connectivity or metric.
     *  @retval contended_k A batch, a load or a plan in flight on this index or on a receiver.
     *  @retval overflow_k Part slots past the 32-bit IDs.
     */
    expected<refine_plan_t> plan_split(std::span<derived_*> receivers, traversal_context_t& context,
                                       refine_config_t config = {}, u64_t seed = 0) noexcept {
        using phase_t = typename refine_plan_t::phase_t;
        refine_plan_t plan = unbound_plan_(context);
        if (status_t const refused = graph_refusal_(context); failed(refused)) return {std::move(plan), refused};
        std::size_t const parts = receivers.size();
        if (parts < 2) return {std::move(plan), status_t::config_mismatch_k};
        for (std::size_t receiver = 0; receiver != parts; ++receiver) {
            derived_* const into = receivers[receiver];
            if (!into || into == &derived()) return {std::move(plan), status_t::config_mismatch_k};
            for (std::size_t earlier = 0; earlier != receiver; ++earlier)
                if (receivers[earlier] == into) return {std::move(plan), status_t::config_mismatch_k};
            vram_index const& part = *into;
            if (part.device_.ordinal != device_.ordinal || part.kernel_ != kernel_ ||
                (part.connectivity() && part.connectivity() != connectivity()))
                return {std::move(plan), status_t::config_mismatch_k};
        }
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        split_t& split = plan.split_;
        split.source = this;
        split.receivers = receivers;
        split.rows = size();
        split.part_capacity = divide_round_up<std::size_t>(split.rows, parts);
        if (parts * split.part_capacity >= std::numeric_limits<u32_t>::max())
            return {std::move(plan), status_t::overflow_k};
        for (derived_* const into : receivers) {
            vram_index& part = *into;
            if (status_t const closed = part.close_(); failed(closed)) return {std::move(plan), closed};
            ++split.closed_receivers;
            if (part.size()) return {std::move(plan), status_t::config_mismatch_k};
            if (status_t const kept = part.status(); failed(kept)) return {std::move(plan), kept};
        }
        // Every receiver holds a full part's rows, sentinel-filled, before the partition queues.
        cuda_stream_handle_t const stream = context.stream();
        for (derived_* const into : receivers) {
            vram_index& part = *into;
            if (status_t const reserved =
                    part.reserve({split.part_capacity, static_cast<connectivity_t>(connectivity()), 0}).status;
                failed(reserved))
                return {std::move(plan), reserved};
            if (status_t const grown = part.grow_rows_(split.part_capacity, stream); failed(grown))
                return {std::move(plan), grown};
        }
        node_id_t const capacity = static_cast<node_id_t>(split.part_capacity);
        status_t opened = open_refine_(plan, {0, capacity}, split.part_capacity, longest_row_, span_bytes_, context,
                                       config);
        if (succeeded(opened) && split.rows) opened = carve_split_(split, seed, context);
        if (succeeded(opened) && split.rows) opened = queue_partition_(split, stream);
        if (failed(opened)) return {std::move(plan), opened};
        plan.phase_ = split.rows ? phase_t::partition_k : phase_t::done_k;
        return {std::move(plan), status_t::success_k};
    }

    /** Measures forward reachability from as many stride-spread landmarks as one search draws -
     *  the connectivity; any shortfall against @c size() is the population no search can
     *  discover. Read-only on the graph, so concurrent insertions may leave fresh nodes
     *  uncounted. Waits once, on the context's event; @c overflow_k when the sweeps reached
     *  their cap unconverged. */
    expected<health_result_t> health(traversal_context_t& context) noexcept {
        std::size_t const rows = size();
        if (!rows) return {{}, status_t::success_k};
        if (status_t const refused = graph_refusal_(context); failed(refused)) return {{}, refused};
        audit_scratch_t scratch = unbound_scratch_(context);
        if (status_t const carved = carve_audit_(scratch, context, audit_use_t::health_k, rows); failed(carved))
            return {{}, carved};
        audit_landing_ptr_t const landing = take_landing_(context);
        if (!landing) return {{}, status_t::bad_alloc_k};
        if (status_t const queued = queue_audit_(scratch, landing.get(), rows, context.stream()); failed(queued))
            return {{}, queued};
        if (status_t const drained = context.drain(); failed(drained)) return {{}, drained};
        return audited_(*landing, rows);
    }

    /** The bytes an executor's run must hold before a context carved over it can refine @p rows
     *  rows under @p config - @c size() for a refine, past the donors' rows for an absorb: the
     *  context's counters, the re-selection's arena and staging, and the audit's scratch. */
    std::size_t traversal_memory_usage(refine_config_t config, std::size_t rows) const noexcept {
        if (!connectivity() || !device_.multiprocessor_count) return 0;
        std::size_t const rung = derived_::rung_serving_(longest_row_);
        gpu_arena_shape_t const arena_shape = build_arena_shape_({config.expansion_factor},
                                                                 static_cast<u32_t>(span_bytes_), rung);
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   cuda_resident_blocks(vram_kernel_t::refine_k, 0) * vram_warps_per_block_k;
        return aligned_(sizeof(vram_telemetry_block_t)) + 2 * aligned_(sizeof(u32_t)) +
               aligned_(slices * arena_shape.template slice_bytes<node_id_t>()) +
               aligned_(build_staging_bytes_(rows, rows)) + aligned_(audit_bytes_(audit_use_t::refine_k, rows));
    }

    /** The bytes an executor's run must hold before a context carved over it can split @p rows rows
     *  into @p receivers receivers under @p config: a refine's over the widest part, and the
     *  partition's scratch. */
    std::size_t traversal_memory_usage(refine_config_t config, std::size_t rows, std::size_t receivers) const noexcept {
        if (!connectivity() || !device_.multiprocessor_count || receivers < 2) return 0;
        return traversal_memory_usage(config, divide_round_up(rows, receivers)) +
               aligned_(split_bytes_(rows, receivers).device);
    }

#pragma endregion Refinement

#pragma region Checkpoints

    /**
     *  @brief One save in flight: the watermark it took, the stream its copies ride, and the rows
     *      it hands the caller's columns through a double-buffered pinned bounce, since the device
     *      storage cannot be read in place.
     *
     *  Neighbors travel as ragged rows, every sentinel and self-loop dropped, and values through
     *  the face's codec, which may unpack them. Planning sized every row, so a call writes the
     *  slices the prefix sums name, for any rows in any order. A call crosses half a bounce at a
     *  time, landing each half on the host while the next one copies off the device, and the last
     *  one lands with the next call or in @c finish. The claim word stays closed from planning to
     *  @c finish or the plan's death, so batches answer @c contended_k meanwhile, while searches
     *  run throughout. The keys are the caller's, and no call writes them.
     */
    struct [[nodiscard]] save_plan_t {

        /** The rows under the watermark, the neighbor IDs they keep, and the bytes their values
         *  save into. */
        columns_extents_t extents {};

        save_plan_t() noexcept = default;
        save_plan_t(save_plan_t&& other) noexcept
            : extents(other.extents), neighbor_offsets_(other.neighbor_offsets_), neighbor_ids_(other.neighbor_ids_),
              value_offsets_(other.value_offsets_), values_(other.values_),
              stored_offsets_(std::move(other.stored_offsets_)), bounce_(std::move(other.bounce_)),
              crossed_ {std::move(other.crossed_[0]), std::move(other.crossed_[1])}, stream_(other.stream_),
              half_rows_(other.half_rows_), half_bytes_(other.half_bytes_), neighbors_bytes_(other.neighbors_bytes_),
              pending_(std::exchange(other.pending_, {})), pending_half_(other.pending_half_), covered_(other.covered_),
              index_(std::exchange(other.index_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;

        /** A save owns nothing but its bounce, so a dying plan awaits the copies into its pending
         *  half and reopens the claim word. */
        ~save_plan_t() noexcept {
            if (!pending_.empty()) [[maybe_unused]]
                status_t const drained = crossed_[pending_half_].drain();
            if (index_) index_->reopen_();
        }

        /** Copies the neighbor IDs and the values of @p rows off the device into the slices the
         *  prefix sums name, half a bounce at a time; the last half is still copying when it
         *  returns. */
        status_t save(iota_view<node_id_t> rows) noexcept {
            if (!index_ || neighbor_offsets_.empty()) return status_t::uninitialized_k;
            if (rows.bound < rows.first || rows.bound > extents.nodes) return status_t::config_mismatch_k;
            for (node_id_t first = rows.first; first != rows.bound;) {
                node_id_t const bound = crossing_end_(to_span(stored_offsets_), first, rows.bound, half_rows_,
                                                      half_bytes_);
                // Every half but the pending one has landed, so the other one is free.
                std::size_t const half = pending_.empty() ? 0 : pending_half_ ^ 1;
                if (status_t const fetched = fetch_({first, bound}, half); failed(fetched)) return fetched;
                if (status_t const landed = land_pending_(); failed(landed)) return landed;
                pending_ = {first, bound}, pending_half_ = half, first = bound;
            }
            return status_t::success_k;
        }

        /** Pronounces the snapshot complete once the last half landed and every row did: a success
         *  spends the plan and reopens the claim word; a refusal leaves it whole, so the missing
         *  rows can still land. */
        status_t finish() noexcept {
            if (!index_ || neighbor_offsets_.empty()) return status_t::uninitialized_k;
            if (status_t const landed = land_pending_(); failed(landed)) return landed;
            if (covered_ != extents.nodes) return status_t::config_mismatch_k;
            std::exchange(index_, nullptr)->reopen_();
            return status_t::success_k;
        }

      private:
        friend struct vram_index;

        /** Queues the copies of @p rows ' neighbor rows and stored values into half @p half of the
         *  bounce, marked on its event. */
        status_t fetch_(iota_view<node_id_t> rows, std::size_t half) noexcept {
            std::size_t const row_bytes = index_->connectivity() * sizeof(node_id_t);
            byte_t* const neighbors = bounce_.data() + half * (neighbors_bytes_ + half_bytes_);
            u64_t const first_byte = stored_offsets_[rows.first];
            u64_t const bytes = stored_offsets_[rows.bound] - first_byte;
            if (cuda_copy_async(neighbors, index_->neighbors_matrix_.base() + rows.first * row_bytes,
                                rows.size() * row_bytes, stream_) != cuda_success_k ||
                (bytes && cuda_copy_async(neighbors + neighbors_bytes_, index_->values_tape_.base() + first_byte, bytes,
                                          stream_) != cuda_success_k))
                return status_t::device_error_k;
            return crossed_[half].record(stream_);
        }

        /** Awaits half @p half, then hands @p visit each of @p rows with its neighbor row and its
         *  stored value there, until one refuses. */
        template <typename visit_type_>
        status_t visit_(iota_view<node_id_t> rows, std::size_t half, visit_type_&& visit) noexcept {
            if (status_t const drained = crossed_[half].drain(); failed(drained)) return drained;
            std::size_t const connectivity = index_->connectivity();
            byte_t const* const landed = bounce_.data() + half * (neighbors_bytes_ + half_bytes_);
            node_id_t const* const neighbors = reinterpret_cast<node_id_t const*>(landed);
            for (node_id_t row = rows.first; row != rows.bound; ++row) {
                std::span<node_id_t const> const slots {neighbors + (row - rows.first) * connectivity, connectivity};
                bytes_view_t const stored = bytes_of(
                    landed + neighbors_bytes_ + stored_offsets_[row] - stored_offsets_[rows.first],
                    stored_offsets_[row + 1] - stored_offsets_[row]);
                if (status_t const visited = visit(row, slots, stored); failed(visited)) return visited;
            }
            return status_t::success_k;
        }

        /** Sizes every row under the watermark, half a bounce at a time, writing both prefix sums
         *  when they are given, @b [nodes+1] each. */
        status_t size_(std::span<u64_t> neighbor_sums, std::span<u64_t> value_sums) noexcept {
            node_id_t const nodes = static_cast<node_id_t>(extents.nodes);
            extents.neighbors = 0, extents.value_bytes = 0;
            if (!neighbor_sums.empty()) neighbor_sums[0] = 0, value_sums[0] = 0;
            auto const sized = [&](node_id_t row, std::span<node_id_t const> slots, bytes_view_t stored) noexcept {
                auto const [saved_bytes, saved_status] = index_->derived().saved_bytes_(stored);
                if (failed(saved_status)) return saved_status;
                for (node_id_t const neighbor : slots)
                    if (kept_link_(neighbor, row, nodes)) ++extents.neighbors;
                extents.value_bytes += saved_bytes;
                if (!neighbor_sums.empty())
                    neighbor_sums[row + 1] = extents.neighbors, value_sums[row + 1] = extents.value_bytes;
                return status_t::success_k;
            };
            for (node_id_t first = 0; first != nodes;) {
                node_id_t const bound = crossing_end_(to_span(stored_offsets_), first, nodes, half_rows_, half_bytes_);
                if (status_t const fetched = fetch_({first, bound}, 0); failed(fetched)) return fetched;
                if (status_t const visited = visit_({first, bound}, 0, sized); failed(visited)) return visited;
                first = bound;
            }
            return status_t::success_k;
        }

        /** Awaits the pending half's copies, then lands its rows in the columns: every neighbor
         *  under the watermark but the row's own ID, and the value through the face's codec. */
        status_t land_pending_() noexcept {
            if (pending_.empty()) return status_t::success_k;
            iota_view<node_id_t> const rows = std::exchange(pending_, {});
            node_id_t const nodes = static_cast<node_id_t>(extents.nodes);
            auto const landed = [&](node_id_t row, std::span<node_id_t const> slots, bytes_view_t stored) noexcept {
                u64_t cursor = neighbor_offsets_[row];
                u64_t const bound = neighbor_offsets_[row + 1];
                for (node_id_t const neighbor : slots) {
                    if (!kept_link_(neighbor, row, nodes)) continue;
                    if (cursor == bound) return status_t::config_mismatch_k;
                    neighbor_ids_[cursor++] = neighbor;
                }
                if (cursor != bound) return status_t::config_mismatch_k;
                return index_->derived().save_value_(
                    stored, values_.subspan(value_offsets_[row], value_offsets_[row + 1] - value_offsets_[row]));
            };
            if (status_t const visited = visit_(rows, pending_half_, landed); failed(visited)) return visited;
            covered_ += rows.size();
            return status_t::success_k;
        }

        /** The columns the calls write but the keys, empty in a plan that only sizes. */
        std::span<u64_t> neighbor_offsets_;
        std::span<node_id_t> neighbor_ids_;
        std::span<u64_t> value_offsets_;
        bytes_span_t values_;

        /** The device's row starts into the tape, landed once at planning, @b [nodes+1]. */
        growing_buffer<u64_t> stored_offsets_ {global_allocator};

        /** Page-locked host bytes in two halves: @c half_rows_ neighbor rows, then @c half_bytes_
         *  of stored values. */
        growing_buffer<byte_t, cuda_pinned_allocator_t> bounce_ {cuda_pinned_allocator_t {}};

        /** Marks the last copies into each half of the bounce, born with the plan on its device. */
        cuda_event_t crossed_[2];
        cuda_stream_handle_t stream_ = nullptr;
        std::size_t half_rows_ = 0, half_bytes_ = 0, neighbors_bytes_ = 0;

        /** The rows whose copies sit in a half not yet landed on the host, and that half. */
        iota_view<node_id_t> pending_ {};
        std::size_t pending_half_ = 0;

        /** Rows whose neighbors and values landed. */
        std::size_t covered_ = 0;

        /** The index being snapshotted, whose claim word the plan holds closed; null in a refused
         *  plan, and once @c finish succeeded. */
        vram_index* index_ = nullptr;
    };

    /** Row & element counts a save needs right now, through one sizing pass over every row under
     *  the watermark with the claim word closed; @c contended_k while a batch, a load or a plan is
     *  in flight. */
    expected<columns_extents_t> columns_extents(traversal_context_t& context) noexcept {
        auto [plan, status] = open_save_(context);
        if (succeeded(status)) status = plan.size_({}, {});
        return {plan.extents, status};
    }

    /**
     *  @brief Opens a save into @p columns of every row under the watermark on @p context 's
     *      stream: closes the claim word, lands the device's row starts, pins the bounce, sizes
     *      every row and writes both prefix sums.
     *
     *  The key column's length names the row count, and the caller lands the keys itself. A column
     *  shorter than its count answers @c overflow_k, and a longer key or sums column
     *  @c config_mismatch_k, so no call ever writes past them.
     *
     *  @param[out] columns Where the rows land, sized from @ref columns_extents.
     *  @param[inout] context The stream every copy queues on.
     */
    template <typename key_type_>
    expected<save_plan_t> plan_save(columns<key_type_, node_id_t, byte_t, u64_t> columns,
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
        return {std::move(plan), status_t::success_k};
    }

    /** Serializes every row under the watermark into @p columns on the calling thread - the
     *  one-shot over @ref plan_save. */
    template <typename key_type_>
    status_t save(columns<key_type_, node_id_t, byte_t, u64_t> columns, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_save(columns, context);
        if (failed(status)) return status;
        if (status_t const saved = plan.save({0, static_cast<node_id_t>(plan.extents.nodes)}); failed(saved))
            return saved;
        return plan.finish();
    }

    /**
     *  @brief One load in flight: the rows it appends past every claimed one, the storage grown to
     *      hold them, and the bounce they cross.
     *
     *  A call stages rows on the host half a bounce at a time - each neighbor row shifted onto the
     *  appended IDs and padded to the connectivity with sentinels, a row with none pointing at
     *  itself, each value through the face's codec, which may pack it - and copies the half to the
     *  device while the host stages the other, refilling a half only once its event says the copies
     *  out of it landed. @c finish is the one full wait, and turns the watermark over the rows.
     *  Holds the claim for its whole life, so nothing else claims rows while it runs; a plan that
     *  dies before @c finish gives its claim back and leaves the storage grown, its rows unfilled.
     */
    struct [[nodiscard]] load_plan_t {

        /** The first appended ID, where row zero of the columns lands. */
        node_id_t base = 0;

        /** Rows the plan appends. */
        std::size_t rows = 0;

        load_plan_t() noexcept = default;
        load_plan_t(load_plan_t&& other) noexcept
            : base(other.base), rows(std::exchange(other.rows, 0)), neighbor_offsets_(other.neighbor_offsets_),
              neighbor_ids_(other.neighbor_ids_), value_offsets_(other.value_offsets_), values_(other.values_),
              stored_offsets_(std::move(other.stored_offsets_)), bounce_(std::move(other.bounce_)),
              crossed_ {std::move(other.crossed_[0]), std::move(other.crossed_[1])}, stream_(other.stream_),
              half_rows_(other.half_rows_), half_bytes_(other.half_bytes_), neighbors_bytes_(other.neighbors_bytes_),
              offsets_bytes_(other.offsets_bytes_), tape_first_(other.tape_first_), longest_(other.longest_),
              span_bytes_(other.span_bytes_), chunks_(other.chunks_), covered_(other.covered_),
              index_(std::exchange(other.index_, nullptr)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** An unfinished plan awaits every copy it queued, so the next grow fills its rows again,
         *  and gives its claim back. */
        ~load_plan_t() noexcept {
            if (!index_) return;
            if (chunks_) {
                [[maybe_unused]] status_t const first_drained = crossed_[0].drain();
                [[maybe_unused]] status_t const second_drained = crossed_[1].drain();
            }
            index_->filled_rows_ = min_of(index_->filled_rows_, static_cast<std::size_t>(base));
            index_->unclaim_(base, rows);
        }

        /** Stages and copies the columns' rows @p range, counted from the columns' first row, half
         *  a bounce at a time; the last half is still copying when it returns. */
        status_t load(iota_view<node_id_t> range) noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (range.bound < range.first || range.bound > rows) return status_t::config_mismatch_k;
            std::size_t const connectivity = index_->connectivity();
            std::size_t const row_bytes = connectivity * sizeof(node_id_t);
            for (node_id_t first = range.first; first != range.bound;) {
                node_id_t const bound = crossing_end_(to_span(stored_offsets_), first, range.bound, half_rows_,
                                                      half_bytes_);
                std::size_t const half = chunks_++ % 2;
                // The host refills a half only once the copies out of it landed, while the other
                // half's copies run.
                if (chunks_ > 2 && failed(crossed_[half].drain())) return status_t::device_error_k;
                byte_t* const staged = bounce_.data() + half * (neighbors_bytes_ + offsets_bytes_ + half_bytes_);
                node_id_t* const neighbors = reinterpret_cast<node_id_t*>(staged);
                u64_t* const offsets = reinterpret_cast<u64_t*>(staged + neighbors_bytes_);
                byte_t* const values = staged + neighbors_bytes_ + offsets_bytes_;
                for (node_id_t row = first; row != bound; ++row) {
                    node_id_t* const slots = neighbors + (row - first) * connectivity;
                    std::size_t kept = 0;
                    for (u64_t link = neighbor_offsets_[row]; link != neighbor_offsets_[row + 1]; ++link)
                        slots[kept++] = static_cast<node_id_t>(neighbor_ids_[link] + base);
                    // A row with no neighbor points at itself, which reads as published and empty.
                    if (!kept) slots[kept++] = static_cast<node_id_t>(base + row);
                    for (; kept != connectivity; ++kept) slots[kept] = std::numeric_limits<node_id_t>::max();
                    offsets[row - first] = tape_first_ + stored_offsets_[row];
                    bytes_span_t const stored {values + stored_offsets_[row] - stored_offsets_[first],
                                               stored_offsets_[row + 1] - stored_offsets_[row]};
                    bytes_view_t const saved = values_.subspan(value_offsets_[row],
                                                               value_offsets_[row + 1] - value_offsets_[row]);
                    if (status_t const loaded = index_->derived().load_value_(saved, stored); failed(loaded))
                        return loaded;
                    span_bytes_ = max_of(span_bytes_, index_->derived().span_bytes_(stored.data(), stored.size()));
                }
                offsets[bound - first] = tape_first_ + stored_offsets_[bound];
                std::size_t const into = base + first;
                u64_t const bytes = stored_offsets_[bound] - stored_offsets_[first];
                if (cuda_copy_async(index_->neighbors_matrix_.base() + into * row_bytes, neighbors,
                                    (bound - first) * row_bytes, stream_) != cuda_success_k ||
                    cuda_copy_async(index_->offsets_column_.base() + into * sizeof(u64_t), offsets,
                                    (bound - first + 1) * sizeof(u64_t), stream_) != cuda_success_k ||
                    (bytes && cuda_copy_async(index_->values_tape_.base() + tape_first_ + stored_offsets_[first],
                                              values, bytes, stream_) != cuda_success_k) ||
                    failed(crossed_[half].record(stream_)))
                    return status_t::device_error_k;
                covered_ += bound - first, first = bound;
            }
            return status_t::success_k;
        }

        /** Publishes the appended rows once every one landed: the tape's extent, the longest row
         *  and the widest span grow over them, and the watermark turns, which lifts the claim. A
         *  success spends the plan; a refusal leaves it whole, so the missing rows can still
         *  land. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (covered_ != rows) return status_t::config_mismatch_k;
            if (failed(crossed_[0].drain()) || failed(crossed_[1].drain())) return status_t::device_error_k;
            vram_index& index = *std::exchange(index_, nullptr);
            index.tape_bytes_ = tape_first_ + stored_offsets_[rows];
            index.longest_row_ = max_of(index.longest_row_, longest_);
            index.span_bytes_ = max_of(index.span_bytes_, span_bytes_);
            standard_atomic_ref<std::size_t>(index.last_visible_id_).store(base + rows, std::memory_order_release);
            return status_t::success_k;
        }

      private:
        friend struct vram_index;

        /** The vetted columns but the keys, which the caller lands. */
        std::span<u64_t const> neighbor_offsets_;
        std::span<node_id_t const> neighbor_ids_;
        std::span<u64_t const> value_offsets_;
        bytes_view_t values_;

        /** Each row's start among the stored values the plan appends, from zero, @b [rows+1]. */
        growing_buffer<u64_t> stored_offsets_ {global_allocator};

        /** Page-locked host bytes in two halves: @c half_rows_ padded neighbor rows, their tape
         *  starts and the one past them, then @c half_bytes_ of stored values. */
        growing_buffer<byte_t, cuda_pinned_allocator_t> bounce_ {cuda_pinned_allocator_t {}};

        /** Marks the last copies out of each half of the bounce, born with the plan on its
         *  device. */
        cuda_event_t crossed_[2];
        cuda_stream_handle_t stream_ = nullptr;
        std::size_t half_rows_ = 0, half_bytes_ = 0, neighbors_bytes_ = 0, offsets_bytes_ = 0;

        /** Where the first appended value lands in the tape: its extent at planning. */
        std::size_t tape_first_ = 0;

        /** The longest appended row and the widest span one prepares into. */
        std::size_t longest_ = 0;
        std::size_t span_bytes_ = 0;

        /** Halves staged so far, and rows copied. */
        std::size_t chunks_ = 0;
        std::size_t covered_ = 0;

        /** The index appended to, whose claim the plan holds; null in a refused plan, and once
         *  @c finish published. */
        vram_index* index_ = nullptr;
    };

    /**
     *  @brief Opens a load of @p columns past every claimed row on @p context 's stream: vets the
     *      columns in full, sizes every stored value, claims the rows, grows the storage and pins
     *      the bounce.
     *
     *  The key column's length names the row count, and the caller lands the keys itself, at
     *  @c base on. Every neighbor ID must name a row of the columns, and no row may keep more
     *  neighbors than the connectivity, both @c malformed_k otherwise.
     *
     *  @param[in] columns The five columns a save filled, by this face or another residency's.
     *  @param[inout] context The stream every copy queues on.
     *  @retval uninitialized_k No @c reserve froze the connectivity yet.
     *  @retval unsupported_k A row past every rung of the face's kits.
     *  @retval contended_k A batch, a load or a plan in flight.
     */
    template <typename key_type_>
    expected<load_plan_t> plan_load(columns<key_type_ const, node_id_t const, byte_t const, u64_t const> columns,
                                    traversal_context_t& context) noexcept {
        if (cuda_set_device(device_.ordinal) != cuda_success_k) return {{}, status_t::device_error_k};
        load_plan_t plan;
        std::size_t const connectivity = this->connectivity(), rows = columns.keys.size();
        if (!connectivity) return {std::move(plan), status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {std::move(plan), status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {std::move(plan), kept};
        if (columns.neighbor_offsets.size() != rows + 1 || columns.value_offsets.size() != rows + 1)
            return {std::move(plan), status_t::malformed_k};
        for (status_t const vetted : {vetted_sums_(columns.neighbor_offsets, columns.neighbor_ids.size()),
                                      vetted_sums_(columns.value_offsets, columns.values.size())})
            if (failed(vetted)) return {std::move(plan), vetted};
        for (std::size_t row = 0; row != rows; ++row)
            if (columns.neighbor_offsets[row + 1] - columns.neighbor_offsets[row] > connectivity)
                return {std::move(plan), status_t::malformed_k};
        for (node_id_t const neighbor : columns.neighbor_ids.first(columns.neighbor_offsets[rows]))
            if (static_cast<std::size_t>(neighbor) >= rows) return {std::move(plan), status_t::malformed_k};

        // Every stored value's size up front, so the tape and the steps' halves are known before
        // the claim.
        if (failed(plan.stored_offsets_.resize_uninitialized(rows + 1)))
            return {std::move(plan), status_t::bad_alloc_k};
        plan.stored_offsets_[0] = 0;
        for (std::size_t row = 0; row != rows; ++row) {
            bytes_view_t const saved = columns.values.subspan(
                columns.value_offsets[row], columns.value_offsets[row + 1] - columns.value_offsets[row]);
            auto const [stored_bytes, stored_status] = derived().stored_bytes_(saved);
            if (failed(stored_status)) return {std::move(plan), stored_status};
            plan.stored_offsets_[row + 1] = plan.stored_offsets_[row] + stored_bytes;
            plan.longest_ = max_of<std::size_t>(plan.longest_, stored_bytes);
        }
        if (rows && derived_::rung_serving_(max_of(plan.longest_, longest_row_)) == derived_::rungs_k)
            return {std::move(plan), status_t::unsupported_k};

        auto const [base, claim_status] = claim_(rows);
        if (failed(claim_status)) return {std::move(plan), claim_status};
        plan.index_ = this, plan.base = base, plan.rows = rows;
        cuda_stream_handle_t const stream = context.stream();
        std::size_t const stored_total = plan.stored_offsets_[rows];
        if (status_t const grown = grow_rows_(base + rows, stream); failed(grown)) return {std::move(plan), grown};
        if (status_t const mapped = values_tape_.commit(tape_bytes_ + stored_total + tape_margin_k); failed(mapped))
            return {std::move(plan), mapped};
        plan.stream_ = stream;
        plan.tape_first_ = tape_bytes_;
        plan.half_rows_ = min_of(rows, power_of_two<std::size_t>(crossing_rows_log2_k));
        plan.half_bytes_ = crossing_bytes_(stored_total, plan.longest_, rows, plan.half_rows_);
        plan.neighbors_bytes_ = aligned_(plan.half_rows_ * connectivity * sizeof(node_id_t));
        plan.offsets_bytes_ = aligned_((plan.half_rows_ + 1) * sizeof(u64_t));
        if (rows && failed(plan.bounce_.resize_uninitialized(
                        2 * (plan.neighbors_bytes_ + plan.offsets_bytes_ + plan.half_bytes_))))
            return {std::move(plan), status_t::bad_alloc_k};
        plan.neighbor_offsets_ = columns.neighbor_offsets, plan.neighbor_ids_ = columns.neighbor_ids;
        plan.value_offsets_ = columns.value_offsets, plan.values_ = columns.values;
        return {std::move(plan), status_t::success_k};
    }

    /** Loads @p columns past every claimed row on the calling thread - the one-shot over
     *  @ref plan_load. */
    template <typename key_type_>
    status_t load(columns<key_type_ const, node_id_t const, byte_t const, u64_t const> columns,
                  traversal_context_t& context) noexcept {
        auto [plan, status] = plan_load(columns, context);
        if (failed(status)) return status;
        if (status_t const loaded = plan.load({0, static_cast<node_id_t>(plan.rows)}); failed(loaded)) return loaded;
        return plan.finish();
    }

#pragma endregion Checkpoints

  protected:

    /** The kits' runtime parameters in device memory, laid out and owned by the face, which
     *  sets them before its first launch; null for kits taking none. */
    byte_t const* parameters_ = nullptr;

  private:

    derived_& derived() noexcept { return static_cast<derived_&>(*this); }
    derived_ const& derived() const noexcept { return static_cast<derived_ const&>(*this); }

    /** Bytes of zeroed margin past the tape's last value, so a word load ending there never
     *  faults. */
    inline static constexpr std::size_t tape_margin_k = sizeof(u64_t);

    /** Threads per block, one warp's worth per resident warp. */
    inline static constexpr u32_t block_threads_k = vram_warps_per_block_k * 32;

    /** Shared bytes a kernel keeps beside its spans, at most: four warps' selection spans. */
    inline static constexpr std::size_t static_shared_bytes_k =
        vram_warps_per_block_k * vram_cache_rows_k * (4 * sizeof(node_id_t) + gpu_scored_node_bytes<node_id_t>);

    /** A wave admits at most a quarter of the rows below it: the widest share at which re-selected
     *  waves still reach the recall of rows inserted one at a time, the rows of a wave linking one
     *  another only through its walk after the merge. */
    inline static constexpr u32_t admission_share_log2_k = 2;

    static std::size_t aligned_(std::size_t bytes) noexcept {
        return divide_round_up<std::size_t>(bytes, default_alignment_k) * default_alignment_k;
    }

    /** Bytes of a batch's staged offsets, whole words of the chains' alignment. */
    static std::size_t offsets_bytes_(std::size_t rows) noexcept { return (rows + 1) * sizeof(u64_t); }

    /** A batch's insertion staging: its offsets, then a chain head over every row the batch leaves
     *  behind, a next edge per pick, the touched list as long, and its count. */
    std::size_t build_staging_bytes_(std::size_t rows, std::size_t total_rows) const noexcept {
        return offsets_bytes_(rows) + (total_rows + 2 * rows * connectivity()) * sizeof(node_id_t) + sizeof(u32_t);
    }

    /** A search's staging: the queries' bytes and margin in whole words, their offsets, then the
     *  matched IDs and distances. */
    static std::size_t search_staging_bytes_(std::size_t queries, std::size_t bytes, std::size_t count) noexcept {
        return round_up_to_multiple<std::size_t>(bytes + tape_margin_k, sizeof(u64_t)) + offsets_bytes_(queries) +
               queries * count * (sizeof(node_id_t) + sizeof(distance_t));
    }

    /** Walkers one block hosts on the kit rung @p rung: a warp each when a lane scores a row alone,
     *  one walking the whole block otherwise. */
    static u32_t walkers_per_block_(std::size_t rung) noexcept {
        return rung < derived_::rungs_k && derived_::lanes_per_row_(rung) > 1 ? 1u : vram_warps_per_block_k;
    }

    /** Blocks a walk over @p queries launches on the kit rung @p rung and the register rung
     *  @p slots: a walker per query up to the grid the device seats. */
    u32_t walk_blocks_(std::size_t queries, std::size_t rung, u32_t slots) const noexcept {
        return static_cast<u32_t>(min_of<std::size_t>(
            divide_round_up<std::size_t>(queries, walkers_per_block_(rung)),
            std::size_t {device_.multiprocessor_count} * cuda_resident_blocks(vram_kernel_t::search_k, slots)));
    }

    /** Which frontier a walk keeps: a register rung, or the arena's past the ladder. */
    enum class frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

    /** One warp's arena slice for a walk expanding @p expansion candidates: the exact visited set,
     *  plus the spill tail of a wide frontier, then any scratch shared memory cannot seat. */
    gpu_arena_shape_t arena_shape_for_(u32_t expansion, frontier_kind_t frontier_kind,
                                       u32_t cache_bytes) const noexcept {
        return gpu_arena_shape(
            expansion,
            frontier_kind == frontier_kind_t::wide_arena_k ? static_cast<u32_t>(connectivity()) * expansion : 0,
            cache_bytes);
    }

    /** One building walker's slice: its walk's frontier rung, and its two spans when shared memory
     *  cannot seat them beside the kernel's own. */
    gpu_arena_shape_t build_arena_shape_(insert_config_t config, u32_t span_bytes, std::size_t rung) const noexcept {
        u32_t const pool_slots = gpu_pool_slots(config.expansion_factor, 4);
        bool const seated = walkers_per_block_(rung) * 2 * span_bytes + static_shared_bytes_k <=
                            device_.shared_memory_per_block_bytes;
        u32_t const cache_bytes = seated ? 0 : 2 * span_bytes;
        return pool_slots ? arena_shape_for_(pool_slots * 32, frontier_kind_t::register_ladder_k, cache_bytes)
                          : arena_shape_for_(config.expansion_factor, frontier_kind_t::wide_arena_k, cache_bytes);
    }

    /** A walk's breadth: as given when it covers the count, twice the count otherwise. */
    u32_t walk_expansion_(search_config_t config) const noexcept {
        u32_t const expansion = config.expansion_factor ? config.expansion_factor
                                                        : static_cast<u32_t>(connectivity()) * 2;
        return expansion >= config.count ? expansion : config.count * 2;
    }

    /** The register rung a walk keeps, covering both its breadth and its count. */
    u32_t walk_slots_(search_config_t config) const noexcept {
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        return gpu_pool_slots(walk_expansion_(config), count_slots > 2 ? count_slots : 2);
    }

    /** One walker's slice: the register rung's visited set, or the arena frontier's with its tail
     *  past the ladder, then the query span when shared memory cannot seat it. */
    gpu_arena_shape_t walk_arena_shape_(search_config_t config, u32_t span_bytes, std::size_t rung) const noexcept {
        u32_t const expansion = walk_expansion_(config);
        bool const seated = walkers_per_block_(rung) * span_bytes + static_shared_bytes_k <=
                            device_.shared_memory_per_block_bytes;
        u32_t const cache_bytes = seated ? 0 : span_bytes;
        return expansion > vram_widest_pool_slots_k * 32
                   ? arena_shape_for_(expansion, frontier_kind_t::wide_arena_k, cache_bytes)
                   : arena_shape_for_(walk_slots_(config) * 32, frontier_kind_t::register_ladder_k, cache_bytes);
    }

    /** The widest span any row of @p values and @p offsets prepares into, raising @p longest to the
     *  longest row among them. */
    std::size_t span_bytes_of_(byte_t const* values, std::span<std::size_t const> offsets,
                               std::size_t& longest) const noexcept {
        std::size_t widest = 0;
        for (std::size_t row = 0; row + 1 < offsets.size(); ++row) {
            std::size_t const bytes = offsets[row + 1] - offsets[row];
            longest = max_of(longest, bytes);
            widest = max_of(widest, derived().span_bytes_(values + offsets[row], bytes));
        }
        return widest;
    }

    /** Keeps @p failure as the status unless one is kept already; returns it for the caller's own
     *  report. */
    status_t fail_(status_t failure) noexcept {
        status_t expected = status_t::success_k;
        standard_atomic_ref<status_t>(status_).compare_exchange_strong(expected, failure, std::memory_order_acq_rel,
                                                                       std::memory_order_acquire);
        return failure;
    }

    /** Rows claimed by batches, visible or not: where the next claim starts. */
    std::size_t claimed_() const noexcept { return claims_.claimed(size()); }

    /** Claims @p rows from the watermark: one compare-exchange that succeeds only on an open word
     *  with no batch or load in flight, so its holder alone grows, launches and publishes them. */
    expected<node_id_t> claim_(std::size_t rows) noexcept {
        auto const [first_id, claim_status] = claims_.claim(size(), rows, max_node_count_k);
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

    /** Maps the offsets column and the neighbors matrix up to @p rows, host work alone, so it rides
     *  no stream, and leaves the fresh neighbor rows for @ref grow_rows_ to fill. */
    status_t map_rows_(std::size_t rows) noexcept {
        if (rows <= capacity()) return status_t::success_k;
        if (status_t const mapped = offsets_column_.commit((rows + 1) * sizeof(u64_t)); failed(mapped)) return mapped;
        return neighbors_matrix_.commit(rows * connectivity() * sizeof(node_id_t));
    }

    /** Maps up to @p rows and sentinel-fills on @p stream every mapped neighbor row no fill reached
     *  yet, @p stream first following the last fill, whichever stream queued it; callers hold the
     *  claim. */
    status_t grow_rows_(std::size_t rows, cuda_stream_handle_t stream) noexcept {
        if (status_t const mapped = map_rows_(rows); failed(mapped)) return mapped;
        cuda_api_t const& api = cuda_api();
        if (api.stream_wait_event(stream, grown_, 0) != cuda_success_k) return status_t::device_error_k;
        if (rows <= filled_rows_) return status_t::success_k;
        // The sentinel is all ones at any width, so 32-bit words fill IDs of either.
        std::size_t const mapped_rows = capacity(), row_words = connectivity() * sizeof(node_id_t) / sizeof(u32_t);
        if (api.fill_words_async(cuda_address(neighbors_matrix_.base() + filled_rows_ * row_words * sizeof(u32_t)),
                                 0xFFFFFFFFu, (mapped_rows - filled_rows_) * row_words, stream) != cuda_success_k ||
            failed(grown_.record(stream)))
            return status_t::device_error_k;
        filled_rows_ = mapped_rows;
        return status_t::success_k;
    }

    /**
     *  @brief Launches the batch @p build stages, whose @p offsets the caller spelled, from
     *      @p first_id on in waves, each the share @c admission_share_log2_k of the IDs below it.
     *
     *  Each wave queues behind the last and admits at least @c vram_admission_ramp_floor_k rows.
     *  Every wave walks twice, inserting and then re-selecting, and each walk is followed by its
     *  back-link launch over every row its picks touched. A failure after any kernel launched keeps
     *  the failure, since those rows can no longer be un-claimed.
     */
    status_t launch_insert_(vram_build<node_id_t> const& build, std::span<std::size_t const> offsets,
                            node_id_t first_id, u32_t pool_slots, std::size_t rung,
                            cuda_stream_handle_t stream) noexcept {
        std::size_t const rows = offsets.size() - 1;
        node_id_t const end_id = static_cast<node_id_t>(first_id + rows);
        // Every launch carves its warps' slices from the one arena, sized for the insert's grid,
        // which the re-selection's and the back-link's resident counts match.
        static_assert(
            cuda_resident_blocks(vram_kernel_t::refine_k, 0) == cuda_resident_blocks(vram_kernel_t::insert_k, 0) &&
            cuda_resident_blocks(vram_kernel_t::back_link_k, 0) == cuda_resident_blocks(vram_kernel_t::insert_k, 0));
        u32_t const blocks = device_.multiprocessor_count * cuda_resident_blocks(vram_kernel_t::insert_k, pool_slots);
        u32_t const shared_bytes = walkers_per_block_(rung) * build.scratch_bytes_per_warp;
        vram_arguments<node_id_t> arguments {};
        status_t launched = status_t::success_k;
        bool published = false;
        auto const queue = [&](vram_kernel_t kernel, u32_t slots, node_id_t first, u32_t bytes) noexcept {
            arguments.build.first_id = first;
            if (succeeded(launched))
                launched = cuda_fill_async(build.assigned_rows, 0, sizeof(u32_t), stream) == cuda_success_k
                               ? cuda_launch(derived().kernel_function_(rung, kernel, slots), blocks, block_threads_k,
                                             bytes, stream, cuda_overlap_t::serialized_k, arguments)
                               : status_t::device_error_k;
            published = published || succeeded(launched);
        };
        auto const zero_touched = [&]() noexcept {
            if (succeeded(launched) && cuda_fill_async(build.touched_count, 0, sizeof(u32_t), stream) != cuda_success_k)
                launched = status_t::device_error_k;
        };
        for (node_id_t wave_first = first_id; wave_first != end_id;) {
            node_id_t const wave_end = static_cast<node_id_t>(min_of<u64_t>(
                end_id,
                u64_t {wave_first} + max_of<u64_t>(wave_first >> admission_share_log2_k, vram_admission_ramp_floor_k)));
            // The wave reads its own rows' starts, from where its first byte landed.
            std::size_t const offset = wave_first - first_id;
            arguments.build = build;
            arguments.build.end_id = wave_end;
            arguments.build.sampling_bound = wave_end;
            arguments.build.staged_offsets = build.staged_offsets + offset;
            arguments.build.tape_first = build.tape_first + (offsets[offset] - offsets[0]);
            for (vram_kernel_t const walk : {vram_kernel_t::insert_k, vram_kernel_t::refine_k}) {
                // The re-selections take the batch's second run of epochs.
                arguments.build.epoch_base = build.epoch_base + static_cast<u32_t>(offset) +
                                             (walk == vram_kernel_t::refine_k ? static_cast<u32_t>(rows) : 0u);
                zero_touched();
                queue(walk, pool_slots, wave_first, shared_bytes);
                queue(vram_kernel_t::back_link_k, 0, wave_first, shared_bytes);
            }
            if (failed(launched)) return published ? fail_(launched) : launched;
            wave_first = wave_end;
        }
        return status_t::success_k;
    }

    /** Rows one half of a save's or a load's bounce carries at most, so its two halves pin a few
     *  hundred megabytes at the widest rows. */
    inline static constexpr u32_t crossing_rows_log2_k = 19;

    /** Bytes of stored values one half of a bounce holds: a half's share of @p total over @p rows,
     *  and never less than the @p longest row, so every crossing carries at least one. */
    static std::size_t crossing_bytes_(std::size_t total, std::size_t longest, std::size_t rows,
                                       std::size_t half_rows) noexcept {
        if (!rows) return 0;
        return aligned_(max_of(longest, divide_round_up(total * half_rows, rows)));
    }

    /** The end of the crossing from @p first toward @p bound: the rows one half holds, @p half_rows
     *  at most and @p half_bytes of the values @p offsets lay out at most, and one at least. */
    static node_id_t crossing_end_(std::span<u64_t const> offsets, node_id_t first, node_id_t bound,
                                   std::size_t half_rows, std::size_t half_bytes) noexcept {
        node_id_t end = static_cast<node_id_t>(first + 1);
        while (end != bound && static_cast<std::size_t>(end - first) < half_rows &&
               offsets[end + 1] - offsets[first] <= half_bytes)
            ++end;
        return end;
    }

    /** Whether slot value @p neighbor of row @p row is a link a snapshot keeps: below @p rows, and
     *  not the row's own ID, which marks a published row with none. */
    static bool kept_link_(node_id_t neighbor, node_id_t row, node_id_t rows) noexcept {
        return neighbor < rows && neighbor != row;
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

    /** Opens a save of every row under the watermark with the claim word closed, the index's device
     *  selected first, where the plan's events are born: lands the device's row starts on the host
     *  and pins the bounce, sizing nothing yet. */
    expected<save_plan_t> open_save_(traversal_context_t& context) noexcept {
        if (cuda_set_device(device_.ordinal) != cuda_success_k) return {{}, status_t::device_error_k};
        save_plan_t plan;
        if (context.device() != device_.ordinal || !context) return {std::move(plan), status_t::config_mismatch_k};
        if (status_t const closed = close_(); failed(closed)) return {std::move(plan), closed};
        plan.index_ = this;
        std::size_t const rows = size();
        plan.extents.nodes = rows;
        plan.stream_ = context.stream();
        if (failed(plan.stored_offsets_.resize_uninitialized(rows + 1)))
            return {std::move(plan), status_t::bad_alloc_k};
        plan.stored_offsets_[0] = 0;
        // Every batch drained before publishing, so the column holds every start under the
        // watermark.
        if (rows && cuda_copy(plan.stored_offsets_.data(), offsets_column_.base(), (rows + 1) * sizeof(u64_t)) !=
                        cuda_success_k)
            return {std::move(plan), status_t::device_error_k};
        plan.half_rows_ = min_of(rows, power_of_two<std::size_t>(crossing_rows_log2_k));
        plan.half_bytes_ = crossing_bytes_(tape_bytes_, longest_row_, rows, plan.half_rows_);
        plan.neighbors_bytes_ = aligned_(plan.half_rows_ * connectivity() * sizeof(node_id_t));
        if (rows && failed(plan.bounce_.resize_uninitialized(2 * (plan.neighbors_bytes_ + plan.half_bytes_))))
            return {std::move(plan), status_t::bad_alloc_k};
        return {std::move(plan), status_t::success_k};
    }

    /** Loads the shared fatbin onto the pinned device and finds every kernel of
     *  @ref cuda_graph_kernels over this node-ID type, leaving null the ones it lacks, whose verbs
     *  answer @c unsupported_k. */
    status_t load_graph_kernels_() noexcept {
        auto [module, status] = cuda_module_t::load(device_.ordinal, cuda_graph_fatbin_t::bytes());
        if (failed(status)) return status;
        graph_module_ = std::move(module);
        vram_kernel_symbol_t symbol;
        for (vram_kernel_t const kernel : cuda_graph_kernels) {
            vram_kernel_symbol(symbol, "basic", "cuda_kernel", {"vram_graph", 1}, kernel, itanium_type<node_id_t> {});
            graph_kernels_[static_cast<u32_t>(kernel)] = graph_module_.function(symbol.text);
        }
        return status_t::success_k;
    }

    /** Why a refinement, an absorb or an audit on @p context cannot run: no reservation yet, the
     *  shared kernels lacking this node-ID type, a context of another device, or a failure kept. */
    status_t graph_refusal_(traversal_context_t const& context) const noexcept {
        if (!connectivity()) return status_t::uninitialized_k;
        for (vram_kernel_t const kernel : cuda_graph_kernels)
            if (!graph_kernels_[static_cast<u32_t>(kernel)]) return status_t::unsupported_k;
        if (context.device() != device_.ordinal || !context) return status_t::config_mismatch_k;
        return status();
    }

    /** Blocks one grid-striding pass of @p kernel over the neighbors matrix launches. */
    u32_t graph_blocks_(vram_kernel_t kernel) const noexcept {
        return device_.multiprocessor_count * cuda_resident_blocks(kernel, 0);
    }

    /** Launches the shared @p kernel over @p blocks blocks on @p stream. */
    status_t launch_graph_(vram_kernel_t kernel, u32_t blocks, vram_graph<node_id_t> const& graph,
                           cuda_stream_handle_t stream) noexcept {
        return cuda_launch(graph_kernels_[static_cast<u32_t>(kernel)], blocks, block_threads_k, 0, stream,
                           cuda_overlap_t::serialized_k, graph);
    }

    /** An unbound refinement over @p context 's allocator, which a refused plan answers with. */
    static refine_plan_t unbound_plan_(traversal_context_t& context) noexcept {
        return refine_plan_t {unbound_scratch_(context)};
    }

    /** Bytes of an audit's scratch over @p rows rows, its heal's buffers too for a refine. */
    static std::size_t audit_bytes_(audit_use_t use, std::size_t rows) noexcept {
        std::size_t const words = divide_round_up<std::size_t>(rows, 32);
        std::size_t const heal_words = use == audit_use_t::refine_k ? 2 * rows : 0;
        return (words * 3 + audit_sweeps_cap_k + 2 + heal_words) * sizeof(u32_t);
    }

    /** An empty audit scratch over @p context 's allocator. */
    static audit_scratch_t unbound_scratch_(traversal_context_t& context) noexcept {
        using bytes_ptr_t = typename traversal_context_t::bytes_ptr_t;
        return audit_scratch_t {bytes_ptr_t {nullptr, typename bytes_ptr_t::deleter_type {context.allocator_, 0}}};
    }

    /** Carves @p scratch over @p rows rows from @p context 's allocator, which it was made over,
     *  the reconnected counter zeroed. */
    static status_t carve_audit_(audit_scratch_t& scratch, traversal_context_t& context, audit_use_t use,
                                 std::size_t rows) noexcept {
        std::size_t const bytes = audit_bytes_(use, rows);
        scratch.bytes.reset(context.allocator_.allocate(bytes));
        scratch.bytes.get_deleter().count = bytes;
        if (!scratch.bytes) return status_t::bad_alloc_k;
        scratch.words = divide_round_up<std::size_t>(rows, 32);
        scratch.reached.words = reinterpret_cast<u32_t*>(scratch.bytes.get());
        scratch.frontier.words = scratch.reached.words + scratch.words;
        scratch.next_frontier.words = scratch.frontier.words + scratch.words;
        scratch.expanded = scratch.next_frontier.words + scratch.words;
        scratch.unreached = scratch.expanded + audit_sweeps_cap_k;
        scratch.reconnected = scratch.unreached + 1;
        if (use == audit_use_t::refine_k) {
            scratch.island_ids = scratch.reconnected + 1;
            scratch.in_links = scratch.island_ids + rows;
        }
        return cuda_fill_async(scratch.reconnected, 0, sizeof(u32_t), context.stream()) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** A zeroed landing in a slot of @p context 's stream, or null when it cannot pin one. */
    static audit_landing_ptr_t take_landing_(traversal_context_t& context) noexcept {
        cuda_landings_t* const landings = context.allocator_.landings();
        void* const slot = landings ? landings->take() : nullptr;
        return {slot ? new (slot) audit_landing_t {} : nullptr, landing_return_t {landings}};
    }

    /** The audit @p landing reports over @p rows rows: @c overflow_k when its last sweep still
     *  expanded, since the rows past the cap are then unjudged rather than unreached. */
    static expected<health_result_t> audited_(audit_landing_t const& landing, std::size_t rows) noexcept {
        if (landing.last_expanded) return {{rows, rows - landing.unreached, audit_sweeps_cap_k}, status_t::overflow_k};
        // A sweep that expands nothing ends the levels, and it counts as one, as on the CPU.
        return {{rows, rows - landing.unreached, std::size_t {landing.levels} + 1}, status_t::success_k};
    }

    /** Binds @p plan, whose claim word the caller closed, to a refinement of @p nodes over @p rows
     *  rows of @p longest bytes at most, preparing into @p span_bytes: carves the audit's scratch,
     *  takes a landing, and grows the context's arena and staging for the walk, so no phase
     *  allocates. */
    status_t open_refine_(refine_plan_t& plan, iota_view<node_id_t> nodes, std::size_t rows, std::size_t longest,
                          std::size_t span_bytes, traversal_context_t& context, refine_config_t config) noexcept {
        using phase_t = typename refine_plan_t::phase_t;
        if (status_t const carved = carve_audit_(plan.scratch_, context, audit_use_t::refine_k, rows); failed(carved))
            return carved;
        plan.landing_ = take_landing_(context);
        if (!plan.landing_) return status_t::bad_alloc_k;
        gpu_arena_shape_t const arena_shape = build_arena_shape_(
            {config.expansion_factor}, static_cast<u32_t>(span_bytes), derived_::rung_serving_(longest));
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   cuda_resident_blocks(vram_kernel_t::refine_k, 0) * vram_warps_per_block_k;
        if (!context.ensure_arena(slices * arena_shape.template slice_bytes<node_id_t>()) ||
            !context.ensure_staging(build_staging_bytes_(nodes.size(), rows)))
            return status_t::bad_alloc_k;
        // Zeroed once, so every phase adds up, and landed with every phase.
        if (cuda_fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), context.stream()) !=
            cuda_success_k)
            return status_t::device_error_k;
        plan.context_ = &context;
        plan.config_ = config;
        plan.nodes_ = nodes;
        plan.rows_ = rows;
        plan.phase_ = !rows ? phase_t::done_k : nodes.empty() ? phase_t::audit_k : phase_t::reselect_k;
        return status_t::success_k;
    }

    /** Queues one re-selection of every node in @p nodes over the @p rows published rows, walks and
     *  back-links as an insertion wave's second half, on the arena and staging the plan grew. */
    status_t reselect_(traversal_context_t& context, refine_config_t config, iota_view<node_id_t> nodes,
                       std::size_t rows) noexcept {
        cuda_stream_handle_t const stream = context.stream();
        std::size_t const walked = nodes.size(), rung = derived_::rung_serving_(longest_row_);
        u32_t const span_bytes = static_cast<u32_t>(span_bytes_);
        gpu_arena_shape_t const arena_shape = build_arena_shape_({config.expansion_factor}, span_bytes, rung);
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   cuda_resident_blocks(vram_kernel_t::refine_k, 0) * vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(slices * arena_shape.template slice_bytes<node_id_t>());
        byte_t* const staging = context.ensure_staging(build_staging_bytes_(walked, rows));
        if (!arena || !staging) return status_t::bad_alloc_k;
        auto const [epoch_base, epochs_status] = context.take_epochs(walked, arena_bytes_before);
        if (failed(epochs_status)) return epochs_status;

        vram_arguments<node_id_t> arguments {};
        vram_build<node_id_t>& build = arguments.build;
        build.storage = storage();
        build.first_id = nodes.first;
        build.end_id = nodes.bound;
        build.sampling_bound = static_cast<node_id_t>(rows);
        build.scratch_bytes_per_warp = arena_shape.cache_bytes ? 0u : 2 * span_bytes;
        build.span_bytes = span_bytes;
        build.arena = arena;
        build.arena_shape = arena_shape;
        build.expansion = config.expansion_factor;
        build.epoch_base = epoch_base;
        build.telemetry = context.telemetry_.get();
        build.assigned_rows = context.assigned_rows_.get();
        build.link_heads = reinterpret_cast<node_id_t*>(staging + offsets_bytes_(walked));
        build.link_next = build.link_heads + rows;
        build.touched_ids = build.link_next + walked * connectivity();
        build.touched_count = reinterpret_cast<u32_t*>(build.touched_ids + walked * connectivity());
        if (cuda_fill_async(build.link_heads, 0xFF, rows * sizeof(node_id_t), stream) != cuda_success_k ||
            cuda_fill_async(build.touched_count, 0, sizeof(u32_t), stream) != cuda_success_k)
            return status_t::device_error_k;
        u32_t const pool_slots = gpu_pool_slots(config.expansion_factor, 4);
        u32_t const blocks = device_.multiprocessor_count * cuda_resident_blocks(vram_kernel_t::refine_k, pool_slots);
        u32_t const shared_bytes = walkers_per_block_(rung) * build.scratch_bytes_per_warp;
        for (cuda_rung_t const launch :
             {cuda_rung_t {vram_kernel_t::refine_k, pool_slots}, cuda_rung_t {vram_kernel_t::back_link_k, 0}}) {
            if (cuda_fill_async(build.assigned_rows, 0, sizeof(u32_t), stream) != cuda_success_k)
                return status_t::device_error_k;
            if (status_t const launched = cuda_launch(derived().kernel_function_(rung, launch.kernel, launch.slots),
                                                      blocks, block_threads_k, shared_bytes, stream,
                                                      cuda_overlap_t::serialized_k, arguments);
                failed(launched))
                return launched;
        }
        return status_t::success_k;
    }

    /** Queues a reachability sweep to convergence over @p rows rows from as many stride-spread
     *  landmarks as one search draws - the connectivity - then a count of the rows it never
     *  reached, collecting their IDs when @p scratch has room for them, then the landing of every
     *  counter. Nothing waits; the caller reads @p landing once its event fired. */
    status_t queue_audit_(audit_scratch_t const& scratch, audit_landing_t* landing, std::size_t rows,
                          cuda_stream_handle_t stream) noexcept {
        // The bitsets, the levels and sweeps' expanded counters and the unreached counter lie back
        // to back.
        if (cuda_fill_async(scratch.reached.words, 0, (scratch.words * 3 + audit_sweeps_cap_k + 1) * sizeof(u32_t),
                            stream) != cuda_success_k)
            return status_t::device_error_k;
        vram_graph<node_id_t> graph {};
        graph.neighbors = reinterpret_cast<node_id_t*>(neighbors_matrix_.base());
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.rows_count = static_cast<u32_t>(rows);
        graph.seed_count = static_cast<u32_t>(min_of<std::size_t>(connectivity(), rows));
        graph.reached = scratch.reached;
        graph.frontier = scratch.frontier;
        if (status_t const launched = launch_graph_(
                vram_kernel_t::mark_seeds_k, divide_round_up<u32_t>(graph.seed_count, block_threads_k), graph, stream);
            failed(launched))
            return launched;

        // Every sweep up to the cap queues blind: one queued past convergence finds the one before
        // it expanded nothing and leaves at entry.
        graph.next_frontier = scratch.next_frontier;
        graph.levels = scratch.expanded;
        for (std::size_t sweep_number = 1; sweep_number != audit_sweeps_cap_k; ++sweep_number) {
            graph.counter = scratch.expanded + sweep_number;
            graph.previous_counter = sweep_number == 1 ? nullptr : scratch.expanded + sweep_number - 1;
            if (cuda_fill_async(graph.next_frontier.words, 0, scratch.words * sizeof(u32_t), stream) != cuda_success_k)
                return status_t::device_error_k;
            if (status_t const launched = launch_graph_(vram_kernel_t::reachability_sweep_k,
                                                        graph_blocks_(vram_kernel_t::reachability_sweep_k), graph,
                                                        stream);
                failed(launched))
                return launched;
            std::swap(graph.frontier, graph.next_frontier);
        }
        graph.island_ids = scratch.island_ids;
        graph.counter = scratch.unreached;
        if (status_t const launched = launch_graph_(vram_kernel_t::reachability_islands_k,
                                                    divide_round_up<u32_t>(graph.rows_count, block_threads_k), graph,
                                                    stream);
            failed(launched))
            return launched;
        return cuda_copy_async(&landing->levels, scratch.expanded, sizeof(u32_t), stream) == cuda_success_k &&
                       cuda_copy_async(&landing->last_expanded, scratch.expanded + audit_sweeps_cap_k - 1,
                                       sizeof(audit_landing_t) - sizeof(u32_t), stream) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Queues the reconnection of the @p islands islands the last audit in @p scratch collected
     *  over @p rows rows: the reached in-links first, so no reconnection takes the last link a
     *  reached node is found through, then each island into a reached pick or its landmark. */
    status_t queue_reconnect_(audit_scratch_t const& scratch, u32_t islands, std::size_t rows,
                              cuda_stream_handle_t stream) noexcept {
        if (cuda_fill_async(scratch.in_links, 0, rows * sizeof(u32_t), stream) != cuda_success_k)
            return status_t::device_error_k;
        vram_graph<node_id_t> graph {};
        graph.neighbors = reinterpret_cast<node_id_t*>(neighbors_matrix_.base());
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.rows_count = static_cast<u32_t>(rows);
        graph.seed_count = static_cast<u32_t>(min_of<std::size_t>(connectivity(), rows));
        graph.reached = scratch.reached;
        graph.in_links = scratch.in_links;
        graph.island_ids = scratch.island_ids;
        graph.islands_count = islands;
        graph.reconnected = scratch.reconnected;
        if (status_t const counted = launch_graph_(vram_kernel_t::reached_in_links_k,
                                                   graph_blocks_(vram_kernel_t::reached_in_links_k), graph, stream);
            failed(counted))
            return counted;
        return launch_graph_(vram_kernel_t::reconnect_island_k, divide_round_up<u32_t>(islands, block_threads_k), graph,
                             stream);
    }

    /** Bytes a split of @p rows rows into @p parts parts carves on the device and pins on the
     *  host, and where the pinned bytes place the offsets columns, the bases and the roots, as
     *  @c split_t lays them out. */
    struct split_bytes_t {
        std::size_t device = 0, bases_first = 0;
        std::size_t landing = 0, landed_offsets = 0, landed_bases = 0, landed_roots = 0;
    };
    static split_bytes_t split_bytes_(std::size_t rows, std::size_t parts) noexcept {
        std::size_t const words = divide_round_up<std::size_t>(rows, 32);
        split_bytes_t bytes;
        bytes.bases_first = aligned_((2 * rows + 3 * words + 3 * parts + audit_sweeps_cap_k) * sizeof(u32_t));
        bytes.device = bytes.bases_first + 3 * parts * sizeof(byte_t*);
        bytes.landed_offsets = aligned_((rows + words) * sizeof(u32_t));
        bytes.landed_bases = bytes.landed_offsets + aligned_((rows + parts) * sizeof(u64_t));
        bytes.landed_roots = bytes.landed_bases + 3 * parts * sizeof(byte_t*);
        bytes.landing = bytes.landed_roots + parts * sizeof(u32_t);
        return bytes;
    }

    /** Carves @p split 's device scratch from @p context 's allocator, pins its landing, lands this
     *  index's row starts, and queues the upload of the receivers' bases and the roots @p seed
     *  picks. */
    status_t carve_split_(split_t& split, u64_t seed, traversal_context_t& context) noexcept {
        std::size_t const rows = split.rows, parts = split.receivers.size();
        std::size_t const words = divide_round_up<std::size_t>(rows, 32);
        split_bytes_t const bytes = split_bytes_(rows, parts);
        split.bytes.reset(context.allocator_.allocate(bytes.device));
        split.bytes.get_deleter().count = bytes.device;
        if (!split.bytes) return status_t::bad_alloc_k;
        split.labels = reinterpret_cast<u32_t*>(split.bytes.get());
        split.proposals = split.labels + rows;
        split.frontier.words = split.proposals + rows;
        split.next_frontier.words = split.frontier.words + words;
        split.seams.words = split.next_frontier.words + words;
        split.part_sizes = split.seams.words + words;
        split.proposed = split.part_sizes + parts;
        split.placed = split.proposed + parts;
        split.roots = split.placed + audit_sweeps_cap_k;
        split.part_bases = reinterpret_cast<byte_t**>(split.bytes.get() + bytes.bases_first);
        split.landed_offsets = bytes.landed_offsets;
        if (failed(split.landing.resize_uninitialized(bytes.landing)) || failed(split.parts.resize(parts)) ||
            failed(split.source_offsets.resize_uninitialized(rows + 1)))
            return status_t::bad_alloc_k;
        byte_t** const bases = reinterpret_cast<byte_t**>(split.landing.data() + bytes.landed_bases);
        u32_t* const roots = reinterpret_cast<u32_t*>(split.landing.data() + bytes.landed_roots);
        for (std::size_t part = 0; part != parts; ++part) {
            vram_index& receiver = *split.receivers[part];
            bases[part] = receiver.neighbors_matrix_.base();
            bases[parts + part] = receiver.values_tape_.base();
            bases[2 * parts + part] = receiver.offsets_column_.base();
            roots[part] = static_cast<u32_t>(seeded_bijection(part, rows, seed));
        }
        cuda_stream_handle_t const stream = context.stream();
        return cuda_copy(split.source_offsets.data(), offsets_column_.base(), (rows + 1) * sizeof(u64_t)) ==
                           cuda_success_k &&
                       cuda_copy_async(split.part_bases, bases, 3 * parts * sizeof(byte_t*), stream) ==
                           cuda_success_k &&
                       cuda_copy_async(split.roots, roots, parts * sizeof(u32_t), stream) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** The record every launch of @p split reads: this index's neighbors and rows, and the split's
     *  scratch. */
    vram_graph<node_id_t> split_graph_(split_t const& split) const noexcept {
        std::size_t const parts = split.receivers.size();
        vram_graph<node_id_t> graph {};
        graph.neighbors = reinterpret_cast<node_id_t*>(neighbors_matrix_.base());
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.rows_count = static_cast<u32_t>(split.rows);
        graph.seed_count = static_cast<u32_t>(min_of(parts, split.rows));
        graph.frontier = split.frontier;
        graph.next_frontier = split.next_frontier;
        graph.labels = split.labels;
        graph.proposals = split.proposals;
        graph.part_sizes = split.part_sizes;
        graph.proposed = split.proposed;
        graph.roots = split.roots;
        graph.parts_count = static_cast<u32_t>(parts);
        graph.part_capacity = static_cast<u32_t>(split.part_capacity);
        graph.seams = split.seams;
        graph.part_neighbors = reinterpret_cast<u32_t* const*>(split.part_bases);
        graph.part_values = split.part_bases + parts;
        graph.part_offsets = reinterpret_cast<u64_t const* const*>(split.part_bases + 2 * parts);
        return graph;
    }

    /**
     *  @brief Queues @p split 's partition: the roots, the growth rounds up to the sweeps' cap,
     *      the rows no round reached, the seams, and the landing of the labels and the seams.
     *
     *  Every round queues blind, as an audit's sweeps do: one queued past convergence finds that
     *  the round before placed nothing and leaves at entry.
     */
    status_t queue_partition_(split_t& split, cuda_stream_handle_t stream) noexcept {
        std::size_t const rows = split.rows, parts = split.receivers.size();
        std::size_t const words = divide_round_up<std::size_t>(rows, 32);
        // The labels and the proposals start unknown, then the bitsets, the parts' counters and
        // the rounds' counts empty, each run back to back.
        if (cuda_fill_async(split.labels, 0xFF, 2 * rows * sizeof(u32_t), stream) != cuda_success_k ||
            cuda_fill_async(split.frontier.words, 0, (3 * words + 2 * parts + audit_sweeps_cap_k) * sizeof(u32_t),
                            stream) != cuda_success_k)
            return status_t::device_error_k;
        vram_graph<node_id_t> graph = split_graph_(split);
        if (status_t const launched = launch_graph_(vram_kernel_t::partition_seeds_k,
                                                    divide_round_up<u32_t>(graph.seed_count, block_threads_k), graph,
                                                    stream);
            failed(launched))
            return launched;
        for (std::size_t round = 1; round != audit_sweeps_cap_k; ++round) {
            graph.counter = split.placed + round;
            graph.previous_counter = round == 1 ? nullptr : split.placed + round - 1;
            if (cuda_fill_async(graph.next_frontier.words, 0, words * sizeof(u32_t), stream) != cuda_success_k)
                return status_t::device_error_k;
            for (vram_kernel_t const pass : {vram_kernel_t::partition_propose_k, vram_kernel_t::partition_tally_k,
                                             vram_kernel_t::partition_accept_k})
                if (status_t const launched = launch_graph_(pass, graph_blocks_(pass), graph, stream); failed(launched))
                    return launched;
            if (status_t const launched = launch_graph_(vram_kernel_t::partition_fill_k, static_cast<u32_t>(parts),
                                                        graph, stream);
                failed(launched))
                return launched;
            std::swap(graph.frontier, graph.next_frontier);
        }
        if (status_t const launched = launch_graph_(vram_kernel_t::partition_islands_k, 1, graph, stream);
            failed(launched))
            return launched;
        if (status_t const launched = launch_graph_(vram_kernel_t::partition_seams_k,
                                                    graph_blocks_(vram_kernel_t::partition_seams_k), graph, stream);
            failed(launched))
            return launched;
        byte_t* const landing = split.landing.data();
        return cuda_copy_async(landing, split.labels, rows * sizeof(u32_t), stream) == cuda_success_k &&
                       cuda_copy_async(landing + rows * sizeof(u32_t), split.seams.words, words * sizeof(u32_t),
                                       stream) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Queues @p split 's scatter: the dealt slots and the receivers' offsets columns up, then
     *  every row's neighbors and value into its receiver. */
    status_t queue_scatter_(split_t const& split, cuda_stream_handle_t stream) noexcept {
        byte_t const* const landing = split.landing.data();
        u64_t const* const offsets = reinterpret_cast<u64_t const*>(landing + split.landed_offsets);
        if (cuda_copy_async(split.labels, landing, split.rows * sizeof(u32_t), stream) != cuda_success_k)
            return status_t::device_error_k;
        for (std::size_t index = 0; index != split.receivers.size(); ++index) {
            split_part_t const& part = split.parts[index];
            vram_index const& receiver = *split.receivers[index];
            if (part.rows && cuda_copy_async(receiver.offsets_column_.base(), offsets + part.offsets_first,
                                             (part.rows + 1) * sizeof(u64_t), stream) != cuda_success_k)
                return status_t::device_error_k;
        }
        vram_graph<node_id_t> graph = split_graph_(split);
        graph.values = values_tape_.base();
        graph.value_offsets = reinterpret_cast<u64_t const*>(offsets_column_.base());
        for (vram_kernel_t const pass : {vram_kernel_t::split_links_k, vram_kernel_t::split_tape_k})
            if (status_t const launched = launch_graph_(pass, graph_blocks_(pass), graph, stream); failed(launched))
                return launched;
        return status_t::success_k;
    }

    /** Hardware identity: the ordinal comes from the storage allocator at construction, the
     *  remaining specs fill at the first @c reserve. */
    cuda_device_specs_t device_;

    /** The metric the kits score under, fixed at construction. */
    nk_kernel_kind_t kernel_;

    /** Neighbors per row, frozen by the first @c reserve; zero until then. */
    std::size_t connectivity_ = 0;

    /** The values tape, the offsets column and the neighbors matrix, each allocator its own
     *  storage, so capacities are derived, never cached. */
    cuda_virtual_memory_allocator<byte_t> values_tape_, offsets_column_, neighbors_matrix_;

    /** Held by the one batch claiming, growing and linking at a time, until its drain. */
    claim_word_t claims_;

    /** The value bytes every published batch holds in the tape. */
    std::size_t tape_bytes_ = 0;

    /** The longest stored row and the widest span a stored row prepares into, which every later
     *  batch's kit rung and spans cover. */
    std::size_t longest_row_ = 0;
    std::size_t span_bytes_ = 0;

    /** Mapped rows whose neighbors a grow sentinel-filled, read and advanced by the claim's
     *  holder. */
    std::size_t filled_rows_ = 0;

    /** Marks the last sentinel fill, which every grow's stream follows; recorded and followed by
     *  the claim's holder alone. */
    cuda_event_t grown_;

    /** Rows strictly below it are published and visible, turned once per batch after its drain. */
    std::size_t last_visible_id_ = 0;

    /** The first post-claim device failure. */
    status_t status_ = status_t::success_k;

    /** The shared fatbin on this device, and its kernels in @c vram_kernel_t order, null for the
     *  ones it lacks. */
    cuda_module_t graph_module_;
    cuda_function_handle_t graph_kernels_[static_cast<u32_t>(vram_kernel_t::count_k)] = {};
};

} // namespace unum::usearch::basic
