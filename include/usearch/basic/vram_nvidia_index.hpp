/**
 *  @file include/usearch/basic/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over variable-length values in a single uniform latency memory space on
 *      NVIDIA GPUs.
 *
 *  The host face of the NVIDIA basic family, compiled by any C++ compiler: it fills plain argument
 *  records and launches the kernels of @c vram_nvidia_index.cuh by name, as the dense host face
 *  does, through the dense traversal contexts. Every kernel is one instantiation of a single
 *  template over three integers, the kernel, the kit and the register rung, so its mangled name is
 *  spelled here from those integers alone. A modality hands the index its kits, whose host facts
 *  name the kit a launch's longest row needs, the lanes it scores a row with, and the span one
 *  row's preparation takes.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::snprintf` - a kernel's mangled name

#include <bit>              // `std::bit_ceil`, `std::bit_width`, `std::has_single_bit`
#include <initializer_list> // the two walks of every wave
#include <iterator>         // `std::size` - the kits' facts
#include <span>             // `std::span` - the caller's row offsets
#include <tuple>            // `std::tuple_element_t` - the narrowest kit, which sizes every kit's spans
#include <utility>          // `std::move` - the loaded module

#include <numkong/tensor.hpp> // `nk::matrix_span` - the matches' landing

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_nvidia_device.hpp" // `cuda_traversal_context` & the rungs
#include "usearch/utilities/vram_walks.hpp"         // `gpu_arena_shape_t`, `gpu_pool_slots`

namespace unum::usearch::basic {
using namespace unum::usearch::utilities;

/** The device kit an NVIDIA basic kernel scores with, by the integer its name carries: the unit
 *  compiling a modality's kernels specializes it with @c type for every kit it ships. */
template <u32_t kit_>
struct cuda_kit_of;

#pragma region Kernel Records

/** The ragged storage a kernel reads, mirroring the shared ragged view field by field. */
struct cuda_ragged_storage_t {
    byte_t const* values = nullptr;
    u64_t* offsets = nullptr;
    u32_t* neighbors = nullptr;
    u32_t connectivity = 0;
};

/** One insertion wave and its back-link passes, mirroring the shared build view. */
struct cuda_build_t {
    cuda_ragged_storage_t storage;
    u64_t const* staged_offsets = nullptr;
    u64_t tape_first = 0;
    u32_t end_id = 0;
    u32_t scratch_bytes_per_warp = 0;
    u32_t span_bytes = 0;
    node_id_t first_id = 0;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    u32_t expansion = 0;
    u32_t epoch_base = 0;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_rows = nullptr;
    u32_t* link_heads = nullptr;
    u32_t* link_next = nullptr;
    u32_t* touched_ids = nullptr;
    u32_t* touched_count = nullptr;
};

/** One walking launch and its landing, mirroring the shared search view. */
struct cuda_search_t {
    cuda_ragged_storage_t storage;
    byte_t const* queries = nullptr;
    u64_t const* query_offsets = nullptr;
    u32_t queries_count = 0;
    u32_t sampling_bound = 0;
    u32_t scratch_bytes_per_warp = 0;
    u32_t expansion = 0;
    byte_t* arena = nullptr;
    gpu_arena_shape_t arena_shape {};
    u32_t epoch_base = 0;
    vram_telemetry_block_t* telemetry = nullptr;
    u32_t* assigned_queries = nullptr;
    node_id_t* result_ids = nullptr;
    distance_t* result_distances = nullptr;
    u32_t count = 0;
};

/** The one argument every NVIDIA basic kernel takes, each reading the record its kernel names. */
struct cuda_arguments_t {
    cuda_build_t build;
    cuda_search_t search;
};

#pragma endregion Kernel Records

#pragma region Kernel Names

/** The NVIDIA basic kernels, the first template argument of @c cuda_kernel. */
enum class cuda_kernel_t : u32_t {
    insert_k,
    reselect_k,
    back_link_k,
    search_k,
    count_k,
};

/** One instantiated kernel: the kernel and its register rung, zero where it keeps no ladder or
 *  walks the arena frontier. */
struct cuda_rung_t {
    cuda_kernel_t kernel;
    u32_t slots;
};

/** Every instantiation one kit's code exports. */
inline constexpr cuda_rung_t cuda_rungs[] = {
    {cuda_kernel_t::insert_k, 0},    {cuda_kernel_t::insert_k, 2},    {cuda_kernel_t::insert_k, 4},
    {cuda_kernel_t::insert_k, 8},    {cuda_kernel_t::insert_k, 16},   {cuda_kernel_t::reselect_k, 0},
    {cuda_kernel_t::reselect_k, 2},  {cuda_kernel_t::reselect_k, 4},  {cuda_kernel_t::reselect_k, 8},
    {cuda_kernel_t::reselect_k, 16}, {cuda_kernel_t::back_link_k, 0}, {cuda_kernel_t::search_k, 0},
    {cuda_kernel_t::search_k, 2},    {cuda_kernel_t::search_k, 4},    {cuda_kernel_t::search_k, 8},
    {cuda_kernel_t::search_k, 16},
};

/** Spells the Itanium mangled name of @c cuda_kernel<kernel,kit,slots>: three unsigned template
 *  arguments, then the one @ref cuda_arguments_t parameter, as NVCC emits it. */
inline void cuda_kernel_name(char (&name)[cuda_kernel_name_capacity_k], cuda_rung_t rung, u32_t kit) noexcept {
    std::snprintf(name, sizeof(name), "_ZN4unum7usearch5basic11cuda_kernelILj%uELj%uELj%uEEEvNS1_16cuda_arguments_tE",
                  static_cast<unsigned>(rung.kernel), static_cast<unsigned>(kit), static_cast<unsigned>(rung.slots));
}

/** The fatbin carrying the kernels of every kit the index @p index_type_ scores with, defined by
 *  the unit that embeds it. */
template <typename index_type_>
bytes_view_t cuda_fatbin() noexcept;

#pragma endregion Kernel Names

/**
 *  @brief Similarity Search over variable-length values in a single uniform latency memory space on
 *      NVIDIA GPUs.
 *
 *  Speaks the dense VRAM index's verbs over values of any length: a batch's IDs are one contiguous
 *  range, and its rows turn visible to searches a whole batch at a time. Values sit back to back in
 *  one tape beside a column of row starts, as the caller's own tape lays them out, and the
 *  neighbors matrix holds @c connectivity IDs per row; all three reserve their address range once
 *  and map on demand. The kit a launch scores with is the narrowest whose rows cover the longest
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
 *  @tparam kits_ The kits, narrowest first: each names itself in the kernels by @c kit_k, says
 *      which rows it @c serves and with how many @c lanes_per_row_k, and sizes the @c span_bytes
 *      one row's preparation takes, which every kit of a modality sizes alike.
 */
template <typename... kits_>
struct vram_nvidia_index {
    using traversal_context_t = cuda_traversal_context<cuda_arena_allocator_t>;
    using node_id_t = node_id_t;
    using distance_t = distance_t;
    using small_size_t = u32_t;

    static_assert(sizeof(std::size_t) == sizeof(u64_t), "The caller's offsets cross as the device's words");

    /** The scored node's 32-bit ID space, less the sentinel. */
    inline static constexpr std::size_t max_node_count_k = (std::size_t {1} << 32) - 2u;

    /** The strongest @c synchronization_t contract this design honors; nothing routes on it. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;

    /** The integers naming the kits in the kernels, narrowest first. */
    inline static constexpr u32_t kits_k[] = {kits_::kit_k...};

    /** Lanes each kit scores a row with; past one, a whole block walks each query or node. */
    inline static constexpr u32_t lanes_per_row_k[] = {kits_::lanes_per_row_k...};

    /** Pins the index to the storage allocator's device; the first @c reserve freezes the rest. */
    explicit vram_nvidia_index(cuda_virtual_memory_allocator<byte_t> const& allocator = {}) noexcept
        : device_ {allocator.device()}, values_tape_(allocator), offsets_column_(allocator),
          neighbors_matrix_(allocator), grown_((cuda_set_device(allocator.device()), cuda_event_t {})) {}
    vram_nvidia_index(vram_nvidia_index const&) = delete;
    vram_nvidia_index& operator=(vram_nvidia_index const&) = delete;

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
        std::size_t const row_bytes = connectivity() * sizeof(u32_t);
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
        return sizeof(vram_nvidia_index) + values_tape_.committed_bytes() + offsets_column_.committed_bytes() +
               neighbors_matrix_.committed_bytes();
    }

    /** The pinned device's identity and limits, queried once at the first @c reserve. */
    cuda_device_specs_t const& device_specs() const noexcept { return device_; }

    /** Device-side storage views, the audit surface for tests. */
    cuda_ragged_storage_t storage() const noexcept {
        return {values_tape_.base(), reinterpret_cast<u64_t*>(offsets_column_.base()),
                reinterpret_cast<u32_t*>(neighbors_matrix_.base()), static_cast<u32_t>(connectivity())};
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
            if (status_t const loaded = load_kernels_(); failed(loaded)) return {0, loaded};
            // Address space is free, so the ceilings are generous and ignore the first capacity.
            std::size_t const budget_bytes = 4 * device_.vram_bytes;
            std::size_t const row_bytes = config.connectivity * sizeof(u32_t) + sizeof(u64_t);
            std::size_t const ceiling_rows = min_of(max_node_count_k,
                                                    max_of<std::size_t>(config.capacity, budget_bytes / row_bytes));
            // The offsets column holds one start past the last row.
            if (offsets_column_.reserve_range((ceiling_rows + 1) * sizeof(u64_t)) != status_t::success_k ||
                neighbors_matrix_.reserve_range(ceiling_rows * config.connectivity * sizeof(u32_t)) !=
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
        node_id_t first_id = node_id_unknown_k;
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
        std::size_t kit_position = 0;
        ((kit_position += !kits_::serves(longest)), ...);
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   cuda_insert_blocks_per_multiprocessor_k * vram_warps_per_block_k;
        return aligned_(sizeof(vram_telemetry_block_t)) + 2 * aligned_(sizeof(u32_t)) +
               aligned_(slices * build_arena_shape_(config, span_bytes, kit_position).slice_bytes()) +
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

        // Walks prepare stored rows as much as the batch's, so the kit and spans cover both.
        std::size_t longest = longest_row_;
        u32_t const span_bytes = static_cast<u32_t>(max_of(span_bytes_, span_bytes_of_(values, offsets, longest)));
        std::size_t kit_position = 0;
        ((kit_position += !kits_::serves(longest)), ...);
        if (kit_position == std::size(kits_k)) return {{}, status_t::unsupported_k};
        std::size_t const batch_bytes = offsets[rows] - offsets[0];

        // Held until the drain: a wave's merges rewrite rows another batch's walks would read.
        auto const [first_id, claim_status] = claims_.claim(size(), rows, max_node_count_k);
        if (failed(claim_status)) return {{}, claim_status};
        std::size_t const total_rows = first_id + rows;
        auto const refuse = [&](status_t failure) noexcept -> expected<insert_result_t> {
            claims_.unclaim(first_id, rows);
            return {{}, failure};
        };

        // Every carve and growth precedes the first launch, so a refusal publishes nothing.
        gpu_arena_shape_t const arena_shape = build_arena_shape_(config, span_bytes, kit_position);
        std::size_t const slices = std::size_t {device_.multiprocessor_count} *
                                   cuda_insert_blocks_per_multiprocessor_k * vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(slices * arena_shape.slice_bytes());
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

        cuda_build_t build;
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
        build.link_heads = reinterpret_cast<u32_t*>(staging + offsets_bytes_(rows));
        build.link_next = build.link_heads + total_rows;
        build.touched_ids = build.link_next + rows * connectivity();
        build.touched_count = build.touched_ids + rows * connectivity();
        if (cuda_fill_async(build.link_heads, 0xFF, total_rows * sizeof(u32_t), stream) != cuda_success_k)
            return refuse(status_t::device_error_k);
        status_t const launched = launch_insert_(build, offsets, static_cast<node_id_t>(first_id),
                                                 gpu_pool_slots(config.expansion_factor, 4), kit_position, stream);
        if (failed(launched)) return refuse(launched);

        tape_bytes_ += batch_bytes;
        longest_row_ = longest;
        span_bytes_ = span_bytes;
        if (status_t const drained = context.drain(); failed(drained)) return {{}, fail_(drained)};
        standard_atomic_ref<std::size_t>(last_visible_id_).store(total_rows, std::memory_order_release);

        vram_telemetry_block_t const spent = context.spent();
        insert_result_t result;
        result.new_size = total_rows;
        result.first_id = static_cast<node_id_t>(first_id);
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
        std::size_t kit_position = 0;
        ((kit_position += !kits_::serves(longest)), ...);
        gpu_arena_shape_t const arena_shape = walk_arena_shape_(config, span_bytes, kit_position);
        std::size_t const slices = std::size_t {walk_blocks_(queries_count, kit_position)} * vram_warps_per_block_k;
        return aligned_(sizeof(vram_telemetry_block_t)) + 2 * aligned_(sizeof(u32_t)) +
               aligned_(slices * arena_shape.slice_bytes()) +
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
        std::size_t kit_position = 0;
        ((kit_position += !kits_::serves(longest)), ...);
        if (kit_position == std::size(kits_k)) return {{}, status_t::unsupported_k};
        cuda_stream_handle_t const stream = context.stream();

        // The staging holds the queries' bytes from a word boundary, their offsets, the landing.
        std::size_t const batch_bytes = offsets[queries_count] - offsets[0];
        std::size_t const results_count = queries_count * config.count;
        gpu_arena_shape_t const arena_shape = walk_arena_shape_(config, span_bytes, kit_position);
        u32_t const blocks = walk_blocks_(queries_count, kit_position);
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(std::size_t {blocks} * vram_warps_per_block_k *
                                                   arena_shape.slice_bytes());
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

        cuda_arguments_t arguments {};
        cuda_search_t& search = arguments.search;
        search.storage = storage();
        search.queries = staging;
        search.query_offsets = staged_offsets;
        search.queries_count = static_cast<u32_t>(queries_count);
        search.sampling_bound = static_cast<u32_t>(visible);
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
        u32_t const slots = arena_shape.tail_entries ? 0u : walk_slots_(config);
        if (status_t const launched = cuda_launch(kernel_function_(kit_position, cuda_kernel_t::search_k, slots),
                                                  blocks, block_threads_k,
                                                  walkers_per_block_(kit_position) * search.scratch_bytes_per_warp,
                                                  stream, cuda_overlap_t::serialized_k, arguments);
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

  private:

    /** The narrowest kit, whose preparation sizes every kit's spans alike. */
    using narrowest_kit_t = std::tuple_element_t<0, std::tuple<kits_...>>;

    /** Bytes of zeroed margin past the tape's last value, so a word load ending there never
     *  faults. */
    inline static constexpr std::size_t tape_margin_k = sizeof(u64_t);

    /** Threads per block, one warp's worth per resident warp. */
    inline static constexpr u32_t block_threads_k = vram_warps_per_block_k * 32;

    /** Search blocks per multiprocessor: oversubscribed, so enough queries stay in flight. */
    inline static constexpr u32_t search_blocks_per_multiprocessor_k = 8;

    /** Shared bytes a kernel keeps beside its spans, at most: four warps' selection spans. */
    inline static constexpr std::size_t static_shared_bytes_k = vram_warps_per_block_k * vram_cache_rows_k *
                                                                (4 * sizeof(u32_t) + gpu_scored_node_bytes_k);

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
        return offsets_bytes_(rows) + (total_rows + 2 * rows * connectivity() + 1) * sizeof(u32_t);
    }

    /** A search's staging: the queries' bytes and margin in whole words, their offsets, then the
     *  matched IDs and distances. */
    static std::size_t search_staging_bytes_(std::size_t queries, std::size_t bytes, std::size_t count) noexcept {
        return round_up_to_multiple<std::size_t>(bytes + tape_margin_k, sizeof(u64_t)) + offsets_bytes_(queries) +
               queries * count * (sizeof(node_id_t) + sizeof(distance_t));
    }

    /** Walkers one block hosts under the kit at @p kit_position: a warp each when a lane scores a
     *  row alone, one walking the whole block otherwise. */
    static u32_t walkers_per_block_(std::size_t kit_position) noexcept {
        return kit_position < std::size(kits_k) && lanes_per_row_k[kit_position] > 1 ? 1u : vram_warps_per_block_k;
    }

    /** Blocks a walk over @p queries launches under the kit at @p kit_position: a walker per query
     *  up to the oversubscribed grid. */
    u32_t walk_blocks_(std::size_t queries, std::size_t kit_position) const noexcept {
        return static_cast<u32_t>(
            min_of<std::size_t>(divide_round_up<std::size_t>(queries, walkers_per_block_(kit_position)),
                                device_.multiprocessor_count * search_blocks_per_multiprocessor_k));
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
    gpu_arena_shape_t build_arena_shape_(insert_config_t config, u32_t span_bytes,
                                         std::size_t kit_position) const noexcept {
        u32_t const pool_slots = gpu_pool_slots(config.expansion_factor, 4);
        bool const seated = walkers_per_block_(kit_position) * 2 * span_bytes + static_shared_bytes_k <=
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
    gpu_arena_shape_t walk_arena_shape_(search_config_t config, u32_t span_bytes,
                                        std::size_t kit_position) const noexcept {
        u32_t const expansion = walk_expansion_(config);
        bool const seated = walkers_per_block_(kit_position) * span_bytes + static_shared_bytes_k <=
                            device_.shared_memory_per_block_bytes;
        u32_t const cache_bytes = seated ? 0 : span_bytes;
        return expansion > vram_widest_pool_slots_k * 32
                   ? arena_shape_for_(expansion, frontier_kind_t::wide_arena_k, cache_bytes)
                   : arena_shape_for_(walk_slots_(config) * 32, frontier_kind_t::register_ladder_k, cache_bytes);
    }

    /** The widest span any row of @p values and @p offsets prepares into, raising @p longest to the
     *  longest row among them. */
    static std::size_t span_bytes_of_(byte_t const* values, std::span<std::size_t const> offsets,
                                      std::size_t& longest) noexcept {
        std::size_t widest = 0;
        for (std::size_t row = 0; row + 1 < offsets.size(); ++row) {
            std::size_t const bytes = offsets[row + 1] - offsets[row];
            longest = max_of(longest, bytes);
            widest = max_of(widest, narrowest_kit_t::span_bytes(values + offsets[row], bytes));
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

    /** Maps the offsets column and the neighbors matrix up to @p rows, host work alone, so it rides
     *  no stream, and leaves the fresh neighbor rows for @ref grow_rows_ to fill. */
    status_t map_rows_(std::size_t rows) noexcept {
        if (rows <= capacity()) return status_t::success_k;
        if (status_t const mapped = offsets_column_.commit((rows + 1) * sizeof(u64_t)); failed(mapped)) return mapped;
        return neighbors_matrix_.commit(rows * connectivity() * sizeof(u32_t));
    }

    /** Maps up to @p rows and sentinel-fills on @p stream every mapped neighbor row no fill reached
     *  yet, @p stream first following the last fill, whichever stream queued it; callers hold the
     *  claim. */
    status_t grow_rows_(std::size_t rows, cuda_stream_handle_t stream) noexcept {
        if (status_t const mapped = map_rows_(rows); failed(mapped)) return mapped;
        cuda_api_t const& api = cuda_api();
        if (api.stream_wait_event(stream, grown_, 0) != cuda_success_k) return status_t::device_error_k;
        if (rows <= filled_rows_) return status_t::success_k;
        std::size_t const mapped_rows = capacity(), row_words = connectivity();
        if (api.fill_words_async(cuda_address(neighbors_matrix_.base() + filled_rows_ * row_words * sizeof(u32_t)),
                                 node_id_unknown_k, (mapped_rows - filled_rows_) * row_words,
                                 stream) != cuda_success_k ||
            failed(grown_.record(stream)))
            return status_t::device_error_k;
        filled_rows_ = mapped_rows;
        return status_t::success_k;
    }

    /** Loads the fatbin onto the device and finds every kernel of every kit, by the mangled name
     *  @ref cuda_kernel_name spells. */
    status_t load_kernels_() noexcept {
        auto [module, status] = cuda_module_t::load(device_.ordinal, cuda_fatbin<vram_nvidia_index>());
        if (failed(status)) return status;
        module_ = std::move(module);
        char name[cuda_kernel_name_capacity_k];
        for (std::size_t kit_position = 0; kit_position != std::size(kits_k); ++kit_position)
            for (cuda_rung_t const rung : cuda_rungs) {
                cuda_kernel_name(name, rung, kits_k[kit_position]);
                cuda_function_handle_t& function =
                    kernels_[kit_position][static_cast<u32_t>(rung.kernel)][gpu_rung_index(rung.slots)];
                if (!(function = module_.function(name))) return status_t::device_error_k;
            }
        return status_t::success_k;
    }

    /** The loaded @p kernel of the kit at @p kit_position on its @p slots rung. */
    cuda_function_handle_t kernel_function_(std::size_t kit_position, cuda_kernel_t kernel,
                                            u32_t slots) const noexcept {
        return kernels_[kit_position][static_cast<u32_t>(kernel)][gpu_rung_index(slots)];
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
    status_t launch_insert_(cuda_build_t const& build, std::span<std::size_t const> offsets, node_id_t first_id,
                            u32_t pool_slots, std::size_t kit_position, cuda_stream_handle_t stream) noexcept {
        std::size_t const rows = offsets.size() - 1;
        u32_t const end_id = static_cast<u32_t>(first_id + rows);
        u32_t const blocks = device_.multiprocessor_count * cuda_insert_blocks_per_multiprocessor_k;
        u32_t const shared_bytes = walkers_per_block_(kit_position) * build.scratch_bytes_per_warp;
        cuda_arguments_t arguments {};
        status_t launched = status_t::success_k;
        bool published = false;
        auto const queue = [&](cuda_kernel_t kernel, u32_t slots, node_id_t first, u32_t bytes) noexcept {
            arguments.build.first_id = first;
            if (succeeded(launched))
                launched = cuda_fill_async(build.assigned_rows, 0, sizeof(u32_t), stream) == cuda_success_k
                               ? cuda_launch(kernel_function_(kit_position, kernel, slots), blocks, block_threads_k,
                                             bytes, stream, cuda_overlap_t::serialized_k, arguments)
                               : status_t::device_error_k;
            published = published || succeeded(launched);
        };
        auto const zero_touched = [&]() noexcept {
            if (succeeded(launched) && cuda_fill_async(build.touched_count, 0, sizeof(u32_t), stream) != cuda_success_k)
                launched = status_t::device_error_k;
        };
        for (u32_t wave_first = first_id; wave_first != end_id;) {
            u32_t const wave_end = static_cast<u32_t>(min_of<u64_t>(
                end_id,
                u64_t {wave_first} + max_of(wave_first >> admission_share_log2_k, vram_admission_ramp_floor_k)));
            // The wave reads its own rows' starts, from where its first byte landed.
            std::size_t const offset = wave_first - first_id;
            arguments.build = build;
            arguments.build.end_id = wave_end;
            arguments.build.staged_offsets = build.staged_offsets + offset;
            arguments.build.tape_first = build.tape_first + (offsets[offset] - offsets[0]);
            for (cuda_kernel_t const walk : {cuda_kernel_t::insert_k, cuda_kernel_t::reselect_k}) {
                // The re-selections take the batch's second run of epochs.
                arguments.build.epoch_base = build.epoch_base + static_cast<u32_t>(offset) +
                                             (walk == cuda_kernel_t::reselect_k ? static_cast<u32_t>(rows) : 0u);
                zero_touched();
                queue(walk, pool_slots, wave_first, shared_bytes);
                queue(cuda_kernel_t::back_link_k, 0, wave_first, shared_bytes);
            }
            if (failed(launched)) return published ? fail_(launched) : launched;
            wave_first = wave_end;
        }
        return status_t::success_k;
    }

    /** The fatbin on this device, and its kernels by kit, then in @ref cuda_kernel_t order. */
    cuda_module_t module_;
    cuda_function_handle_t kernels_[sizeof...(kits_)][static_cast<u32_t>(cuda_kernel_t::count_k)]
                                   [gpu_rungs_per_kernel_k] = {};

    /** Hardware identity: the ordinal comes from the storage allocator at construction, the
     *  remaining specs fill at the first @c reserve. */
    cuda_device_specs_t device_;

    /** Neighbors per row, frozen by the first @c reserve; zero until then. */
    std::size_t connectivity_ = 0;

    /** The values tape, the offsets column and the neighbors matrix, each allocator its own
     *  storage, so capacities are derived, never cached. */
    cuda_virtual_memory_allocator<byte_t> values_tape_, offsets_column_, neighbors_matrix_;

    /** Held by the one batch claiming, growing and linking at a time, until its drain. */
    vram_claim_word_t claims_;

    /** The value bytes every published batch holds in the tape. */
    std::size_t tape_bytes_ = 0;

    /** The longest stored row and the widest span a stored row prepares into, which every later
     *  batch's kit and spans cover. */
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
};

} // namespace unum::usearch::basic
