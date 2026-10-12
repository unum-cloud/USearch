/**
 *  @file include/usearch/dense/nvme_index.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Vector Search index for pre-constructed read-only graphs on block storage.
 *
 *  Unlike the mutation-oriented in-memory design, this header is built around one idea: a single
 *  graph walk is a chain of dependent reads, each bounded by device latency, so throughput comes
 *  from keeping many independent walks in flight. The walk logic is therefore a pure state machine
 *  that never performs I/O itself: it only emits read requests and consumes completions, and the
 *  engines of @c utilities/nvme_device.hpp submit those requests through whatever asynchronous
 *  interface the platform offers.
 *
 *  @section dense_nvme_format Storage Format
 *
 *  A part is a SafeGraphs monolith, exactly as @c ram_indexes and @c vram_nvidia_indexes save it: a
 *  key column, a value column, a fixed-width neighbors column, and the producer's norms. The index
 *  reads the header once, at @c view, and keeps only the columns' offsets, so its memory never
 *  grows with the node count. Expanding a node costs one covering read of its neighbors row, and
 *  scoring one costs a covering read of its value row, each rounded out to the medium's block;
 *  norms are recomputed from the landed row, so the norms column is never read.
 *
 *  @section dense_nvme_tiering Hot Working Set & Cold Storage
 *
 *  Block storage is not the bottom of the hierarchy here; it is the hot working area. A shard lives
 *  long-term wherever it is cheapest to keep, such as a shared filesystem or an object bucket
 *  behind a FUSE mount, and is staged onto the local NVMe or High-Bandwidth-Flash the handle names
 *  when it is loaded, because a walk at device latency cannot run against a ranged @c GET that
 *  answers a thousand times slower. Nothing in this header decides when a shard moves or which
 *  shards stay warm; that policy belongs above it. What this header guarantees is the seam that
 *  policy needs: a shard is opened by path and addressed by a medium ID, so one that was staged
 *  from a bucket onto a drive, or evicted and staged again elsewhere, is re-attached without the
 *  state machine learning it moved.
 *
 *  @section dense_nvme_concurrency Concurrency
 *
 *  The graph is immutable, so the index object holds no atomics, no locks, and no compare-and-swap
 *  loops. Its mutable state, the walks, the rows and the reads in flight, is carved per call from
 *  the arena of the executor thread that drives it, and every engine belongs to one thread, so no
 *  two threads ever share either. The only spinning in the design is the engine-side completion
 *  wait, delegated to the same ForkUnion monitor-waiter vocabulary the in-memory index uses for
 *  lock sentinels.
 */
#pragma once

#include <cassert> // `assert`
#include <cstring> // `std::memcpy`

#include <array>       // `std::array`
#include <atomic>      // `std::atomic_ref`
#include <chrono>      // `std::chrono::steady_clock`, for per-row latencies
#include <new>         // placement `new`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::is_same_v`

#include <numkong/tensor.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `seekable_file_t`
#include "usearch/utilities/nvme_device.hpp"       // `nvme_executor`, `file_medium_t`, `io_request_t`
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`, `sorted_buffer`, `arena_allocator`
#include "usearch/utilities/ram_sorts.hpp"         // `network_sort`
#include "usearch/utilities/ram_tables.hpp"        // `growing_hash_set`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_header`
#include "usearch/dense/types.hpp"
#include "usearch/dense/ram_metric.hpp"
#include "usearch/dense/ram_index.hpp" // `filtering_fold`, `multiply_fold`

namespace unum::usearch::dense {

#pragma region Parts

/** Where one SafeGraphs part keeps its columns, read once from its header: what a walk needs to
 *  address any row, and nothing per node. */
struct nvme_part_t {

    /** Rows the part holds; neighbor IDs at or past it are sentinels. */
    u32_t nodes = 0;

    /** Logical dimensions of one value row. */
    u32_t dimensions = 0;

    /** Neighbor IDs one neighbors row holds. */
    u32_t connectivity = 0;

    /** Bytes of one key row. */
    u32_t key_row_bytes = 0;

    /** Bytes of one value row. */
    u32_t value_row_bytes = 0;

    /** Where row zero of the key column starts, in bytes from the part's own start. */
    u64_t keys_begin = 0;

    /** Where row zero of the value column starts. */
    u64_t values_begin = 0;

    /** Where row zero of the neighbors column starts. */
    u64_t neighbors_begin = 0;
};

/** Where one entry lives among the shards a call reads: what a walk ranks by until keys resolve. */
struct nvme_locator_t {

    /** The shard holding the entry, positional among the call's shards. */
    u32_t shard = 0;

    /** The entry's ID inside that shard's own space. */
    node_id_t node = 0;
};

/** Maps a walked node ID of one shard to its locator, as an ID resolver and as an ID-to-key
 *  table. */
struct nvme_shard_locators_t {

    /** The shard every resolved ID belongs to. */
    u32_t shard = 0;

    nvme_locator_t operator()(node_id_t node) const noexcept { return {shard, node}; }
    nvme_locator_t operator[](node_id_t node) const noexcept { return {shard, node}; }
};

/** Reads of one purpose a call issued, and the bytes they transferred, block rounding included. */
struct nvme_reads_t {

    /** Reads issued. */
    std::size_t count = 0;

    /** Bytes those reads transferred. */
    std::size_t bytes = 0;

    nvme_reads_t& operator+=(nvme_reads_t const& other) noexcept {
        count += other.count, bytes += other.bytes;
        return *this;
    }
};

/** The bytes the widest block-aligned read covering any @p row_bytes -byte row takes on
 *  @p block_bytes blocks. */
inline u64_t nvme_covering_bytes(u64_t row_bytes, u64_t block_bytes) noexcept {
    return round_up_to_multiple<u64_t>(row_bytes + block_bytes - 1, block_bytes);
}

/** The read of whole @p block_bytes blocks covering the @p bytes at @p begin on medium @p medium,
 *  landing in @p buffer, its completion carrying @p context back. */
inline io_request_t nvme_covering_read(u64_t begin, u64_t bytes, u64_t block_bytes, io_medium_id_t medium,
                                       io_buffer_id_t buffer, u64_t context) noexcept {
    u64_t const first = begin - begin % block_bytes;
    u64_t const bound = round_up_to_multiple<u64_t>(begin + bytes, block_bytes);
    assert(bound - first <= io_max_length_k && "A covering read must fit the length a completion counts");
    return {first, static_cast<u32_t>(bound - first), medium, buffer, context};
}

/** Reads the prefix and the header of the part @p source holds, any reader with
 *  `read_at(offset, bytes)`, into @p header_bytes; @c malformed_k when it holds no monolith. */
template <typename source_type_, typename allocator_type_>
status_t nvme_read_header(source_type_ const& source, growing_buffer<byte_t, allocator_type_>& header_bytes) noexcept {
    std::array<byte_t, sizeof(safegraphs_prefix_t)> prefix_bytes {};
    if (status_t const read = source.read_at(0, prefix_bytes); failed(read)) return status_t::malformed_k;
    safegraphs_prefix_t const prefix = std::bit_cast<safegraphs_prefix_t>(prefix_bytes);
    if (prefix.magic != safegraphs_prefix_t::magic_k) return status_t::malformed_k;
    std::size_t const header_length = static_cast<std::size_t>(little_endian(prefix.header_length));
    if (failed(header_bytes.resize_uninitialized(sizeof(prefix) + header_length))) return status_t::bad_alloc_k;
    if (status_t const read = source.read_at(0, to_writable_span(header_bytes)); failed(read))
        return status_t::malformed_k;
    return status_t::success_k;
}

/** The metadata of the checkpoint at @p path without viewing it: a manifest or a monolith in a
 *  file, or in SPDK builds a monolith a transport ID names, whose header comes through the
 *  namespace as a view reads it. */
inline expected<safegraphs_metadata_t> nvme_read_metadata(std::string_view path) noexcept {
#if USEARCH_WITH_LINUX_SPDK
    if (names_transport_id(path)) {
        auto [name_space, open_status] = spdk_medium_t::open(path);
        if (failed(open_status)) return {safegraphs_metadata_t {}, open_status};
        growing_buffer<byte_t> header_bytes {global_allocator};
        if (status_t const read = nvme_read_header(name_space, header_bytes); failed(read))
            return {safegraphs_metadata_t {}, read};
        return safegraphs_read_metadata(to_span(header_bytes));
    }
#endif
    if (names_transport_id(path)) return {safegraphs_metadata_t {}, status_t::unsupported_k};
    return safegraphs_read_metadata(path);
}

/** Which keys a walk reads besides the value and neighbors rows. */
enum class nvme_key_reads_t : u8_t {

    /** Only the outputs a row writes resolve into keys, one read each, once its walks finished. */
    outputs_k,

    /** Every scored row's key lands beside its value, for a fold or a rescoring that reads keys. */
    scored_k,
};

#pragma endregion Parts

#pragma region Index

/**
 *  @brief Vector Search index over one pre-constructed read-only SafeGraphs part on block storage.
 *
 *  The class itself is a thin frozen directory: the part's medium and its columns' offsets. Every
 *  call borrows an executor, attaches the medium to the executor's engines for its own duration,
 *  and carves its walks from the executor's arenas, so any executor serves any view, and one view
 *  serves any number of executors in turn.
 *
 *  @section dense_nvme_index_walk Walk Shape
 *
 *  A walk is the in-memory index's best-first walk, cut at every read: a hop issues one read of
 *  the expanded node's neighbors row, then one read of every unvisited neighbor's value row, and
 *  the landed rows are scored four at a time. Across the walks of a thread, everything overlaps.
 *  Rows fill an engine's queue first; whatever depth a short batch leaves idle lets each walk keep
 *  several hops in flight, expanding the next closest frontier nodes inside its pool's radius
 *  before earlier hops land, so a full batch walks hop by hop and a lone query takes the queue.
 */
template <                                          //
    nk_dtype_t vector_dtype_ = nk_bf16_k,           //
    nk_capability_t capability_ = nk_cap_icelake_k, //
    typename engine_type_ = default_io_engine_t     //
    >
struct nvme_index {

    // The sharded index walks every part of a row on one thread, through the walk machinery below.
    template <typename, nk_dtype_t, nk_capability_t, typename>
    friend struct nvme_indexes;

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;
    inline static constexpr nk_capability_t capability_k = capability_;

    using engine_t = engine_type_;
    using executor_t = nvme_executor<engine_t>;
    using medium_t = typename engine_t::medium_t;
    using arena_t = typename executor_t::arena_t;

    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using dot_result_t = typename scalar_t::dot_result_t;
    using distance_t = f32_t;

    /** Default integer type to use for loops and indexing at sub-index scale. */
    using small_size_t = u32_t;

    /** Node identifier within this part, the class-local spelling of @c dense::node_id_t. */
    using node_id_t = dense::node_id_t;

    using vector_view_t = nk::vector_view<scalar_t>;
    using vectors_view_t = nk::matrix_view<scalar_t>;
    using vector_norms_t = vector_norms<dot_result_t>;

    /** The capability bit of the distance kit the chain actually picked, lower than @c capability_
     *  when no SIMD kit is adapted for this dtype yet. */
    inline static constexpr nk_capability_t picked_quads_capability_k = quads_chain<scalar_t, capability_>::picked_k;

    /** The distance kit the chain picked. */
    using quads_kit_t = typename quads_chain<scalar_t, capability_>::kit_t;

    /** The strongest @c synchronization_t contract this design honors: the payload is immutable, so
     *  arbitrary readers share one instance with no fences at all. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::immutable_k;

    /** Per-search tuning knobs; zeros mean "derive from connectivity". */
    struct search_config_t {

        /** Breadth of each walk's candidate pool; defaults to twice the connectivity. */
        small_size_t expansion_factor = 0;

        /** How rows are reached; @c scan_k streams every part's value column, ignoring the
         *  breadth. */
        traversal_t traversal = traversal_t::walk_k;
    };

    /** Telemetry of one batched call; matches land in the caller's spans. */
    struct [[nodiscard]] search_result_t {

        /** Matches written across the batch, or what the verb counts instead. */
        std::size_t count = 0;

        /** Frontier candidates whose neighbors rows were read & expanded. */
        std::size_t visited_members = 0;

        /** Total distance computations across the batch, padding lanes excluded. */
        std::size_t computed_distances = 0;

        /** Re-seedings after a frontier drained before its pool filled. */
        std::size_t restarts = 0;

        /** Walks a scratch refusal ended early, their rankings short. */
        std::size_t saturated_walks = 0;

        /** Neighbors rows read to expand frontiers. */
        nvme_reads_t neighbor_reads;

        /** Value rows read to score. */
        nvme_reads_t value_reads;

        /** Key rows read, beside scored rows under a keyed fold or to resolve written outputs. */
        nvme_reads_t key_reads;

        /** Runs of whole rows a bulk scan of a column read. */
        nvme_reads_t scan_reads;

        /** Submit-and-wait cycles of the engines, counted once for every row in flight through
         *  each. */
        std::size_t round_trips = 0;
    };

    /** An empty index ranking by @p kernel once it views a part; a kernel the dtype's kits cannot
     *  finalize is refused at @ref view. */
    explicit nvme_index(nk_kernel_kind_t kernel = nk_kernel_angular_k) noexcept : kernel_(kernel) {}
    nvme_index(nvme_index&&) noexcept = default;
    nvme_index& operator=(nvme_index&&) noexcept = default;
    nvme_index(nvme_index const&) = delete;
    nvme_index& operator=(nvme_index const&) = delete;

    /** The distance kind every search ranks by, fixed at construction. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Whether the kernel reads rows' norms: every metric but a dot, and a dot whose kit
     *  compensates. */
    bool stores_norms() const noexcept { return stores_norms_(kernel_); }

    /** The norms of @p row the kernel reads, or none when it reads none. */
    vector_norms_t norms(vector_view_t row) const noexcept { return norms_(kernel_, row); }

    /** Opens the SafeGraphs monolith at @p path, a file, a device or, in SPDK builds, a transport
     *  ID with optional @c ns:<ID> and @c base:<bytes> keys: its header, vetted against this
     *  instantiation and its kernel, and its medium for the engines. Refuses with
     *  @c unknown_kind_k a kernel the dtype's kits cannot finalize. A failed view leaves the index
     *  empty. */
    status_t view(std::string_view path) noexcept {
        medium_ = medium_t {};
        part_ = {};
        if (!kernel_supported<quads_kit_t>(kernel_)) return status_t::unknown_kind_k;
        return open_part_(path, dtype_t::unknown_k, kernel_, std::allocator<byte_t> {}, medium_, part_);
    }

    /** Rows the part holds. */
    std::size_t size() const noexcept { return part_.nodes; }

    /** Logical dimensions per value row. */
    std::size_t dimensions() const noexcept { return part_.dimensions; }

    /** Neighbor IDs per row. */
    std::size_t connectivity() const noexcept { return part_.connectivity; }

    /** Bytes the index occupies: the directory alone, whatever the node count. */
    std::size_t memory_usage() const noexcept { return sizeof(nvme_index); }

    /** The walk breadth @p config asks for, or twice the connectivity when it asks for none. */
    template <typename config_type_>
    std::size_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor : connectivity() * 2;
    }

    /**
     *  @brief Finds the nearest rows of every query in @p queries, walks of many queries
     *      overlapping on every thread of @p executor.
     *
     *  @param[in] queries The batch, one query per row, in the part's stored scalar type.
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] config The walk breadth; @c scan_k is refused with @c unsupported_k, as exact
     *      scans run over @c nvme_indexes, which a monolith views as one shard.
     *  @param[out] node_ids Rankings by node ID, one query per row, as many matches as its width.
     *  @param[out] distances Distances of the same rankings, ascending per row.
     *  @param[out] counts Matches written per query.
     *
     *  A failed read stops the call: the engines drain, every buffer returns, and the call answers
     *  @c io_k.
     */
    expected<search_result_t> search(vectors_view_t queries, executor_t& executor, search_config_t config,
                                     nk::matrix_span<node_id_t> node_ids, nk::matrix_span<distance_t> distances,
                                     std::span<std::size_t> counts) noexcept {
        return search_(queries, dummy_heads_t {}, executor, config, node_ids, distances, counts);
    }

    /**
     *  @brief The batched @ref search under a lightning indexer, sharing its whole contract: every
     *      distance is 1 − its node's indexer score against the heads of its query row.
     *
     *  @param[in] heads The indexer heads of every query row.
     *  @return @c config_mismatch_k unless the part ranks by dot.
     */
    expected<search_result_t> search(vectors_view_t queries, indexer_heads<scalar_t> const& heads, executor_t& executor,
                                     search_config_t config, nk::matrix_span<node_id_t> node_ids,
                                     nk::matrix_span<distance_t> distances, std::span<std::size_t> counts) noexcept
        requires floating_scalar<scalar_t>
    {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return search_(queries, heads, executor, config, node_ids, distances, counts);
    }

  private:
    using scratch_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;
    using candidates_allocator_t = arena_allocator<scored_node_t, forkunion_allocator_t>;
    using results_array_t = sorted_buffer<scored_node_t, closer_scored_node_t, candidates_allocator_t>;

    /** 8-ary: eight 8-byte scored nodes fill one cache line per level scan. */
    using frontier_queue_t = max_heap_dary<8, scored_node_t, further_scored_node_t, candidates_allocator_t>;
    using visited_set_t = growing_hash_set<node_id_t, std::hash<node_id_t>, scratch_allocator_t,
                                           typename visits_hash_set_chain<capability_>::kit_t>;

    /** Whether @p kernel reads rows' norms, as @ref stores_norms answers for this index's own. */
    static bool stores_norms_(nk_kernel_kind_t kernel) noexcept {
        return kernel != nk_kernel_dot_k || quads_compensated<quads_kit_t>;
    }

    /** The norms of @p row that @p kernel reads, or none, so a plain dot never sums a landed
     *  row. */
    static vector_norms_t norms_(nk_kernel_kind_t kernel, vector_view_t row) noexcept {
        return stores_norms_(kernel) ? dense::norms<capability_, scalar_t>(row) : vector_norms_t {};
    }

#pragma region Opening

    /** Whether a part's metadata token @p name is absent, or parses to @p parsed equal to the
     *  @p wanted one. */
    static bool agrees_(safegraphs_name_t const& name, int wanted, int parsed) noexcept {
        return name.empty() || wanted == parsed;
    }

    /** Opens the part at @p path, a file or device or, in SPDK builds, a transport ID: its medium
     *  for the engines, and its header vetted against this instantiation, against @p kernel unless
     *  it is @c nk_kernel_unknown_k, which then takes the metric the header names, and, unless
     *  @p key_dtype is @c unknown_k, against the key column's dtype. A file's header comes through
     *  one buffered read, a namespace's through blocking sector reads. */
    template <typename allocator_type_>
    static status_t open_part_(std::string_view path, dtype_t key_dtype, nk_kernel_kind_t& kernel,
                               allocator_type_ allocator, medium_t& medium, nvme_part_t& part) noexcept {
#if USEARCH_WITH_LINUX_SPDK
        if constexpr (std::is_constructible_v<medium_t, spdk_medium_t&&>)
            if (names_transport_id(path)) {
                auto [name_space, open_status] = spdk_medium_t::open(path);
                if (failed(open_status)) return open_status;
                if (status_t const read = read_part_(name_space, key_dtype, kernel, allocator, part); failed(read))
                    return read;
                medium = medium_t {std::move(name_space)};
                return status_t::success_k;
            }
#endif
        if (names_transport_id(path)) return status_t::unsupported_k;
        if constexpr (!std::is_constructible_v<medium_t, file_medium_t&&>) return status_t::config_mismatch_k;
        else {
            auto [file, open_status] = seekable_file_t::open(path, file_mode_t::read_k, allocator);
            if (failed(open_status)) return open_status;
            if (status_t const read = read_part_(file, key_dtype, kernel, allocator, part); failed(read)) return read;
            auto [file_medium, medium_status] = file_medium_t::open(path);
            if (failed(medium_status)) return part = {}, medium_status;
            medium = medium_t {std::move(file_medium)};
            return status_t::success_k;
        }
    }

    /** Reads a part's header through @p source, any reader with `read_at(offset, bytes)`, into
     *  @p part, vetted as @ref open_part_ promises. */
    template <typename source_type_, typename allocator_type_>
    static status_t read_part_(source_type_ const& source, dtype_t key_dtype, nk_kernel_kind_t& kernel,
                               allocator_type_ allocator, nvme_part_t& part) noexcept {
        growing_buffer<byte_t, allocator_type_> header_bytes {allocator};
        if (status_t const read = nvme_read_header(source, header_bytes); failed(read)) return read;

        safegraphs_name_t metric, quantization;
        auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "metric") metric = safegraphs_name_t {field.value};
            if (field.key == "quantization") quantization = safegraphs_name_t {field.value};
        };
        auto const [header, header_status] = safegraphs_read_header(to_span(header_bytes), allocator, per_custom_field);
        if (failed(header_status)) return header_status;
        if (header.key.offsets || header.value.offsets || header.neighbors.offsets) return status_t::config_mismatch_k;
        if (header.value.dtype != safegraphs_value_dtype<scalar_t>() || header.neighbors.dtype != dtype_t::u32_k ||
            header.key.shape != column_shape_t {scalar_shape_t {}})
            return status_t::config_mismatch_k;
        if (key_dtype != dtype_t::unknown_k && header.key.dtype != key_dtype) return status_t::config_mismatch_k;
        nk_kernel_kind_t const named = metric.empty() ? kernel
                                                      : nk_kernel_named(metric.characters, metric.view().size());
        if ((kernel != nk_kernel_unknown_k && named != kernel) ||
            !agrees_(quantization, vector_dtype_, nk_dtype_named(quantization.characters, quantization.view().size())))
            return status_t::config_mismatch_k;
        std::size_t const nodes = header.nodes();
        std::size_t const dimensions = regular_width(header.value.shape).value_or(0);
        std::size_t const connectivity = regular_width(header.neighbors.shape).value_or(0);
        if (header.value.nodes() != nodes || header.neighbors.nodes() != nodes || !dimensions || !connectivity ||
            !header.key.row_bytes() || nodes >= node_id_unknown_k)
            return status_t::malformed_k;

        kernel = named;
        part.nodes = static_cast<u32_t>(nodes);
        part.dimensions = static_cast<u32_t>(dimensions);
        part.connectivity = static_cast<u32_t>(connectivity);
        part.key_row_bytes = static_cast<u32_t>(header.key.row_bytes());
        part.value_row_bytes = static_cast<u32_t>(header.value.row_bytes());
        part.keys_begin = header.data_offset + header.key.data.first;
        part.values_begin = header.data_offset + header.value.data.first;
        part.neighbors_begin = header.data_offset + header.neighbors.data.first;
        return status_t::success_k;
    }

#pragma endregion Opening

#pragma region Walks

    /** Where a walk stands between its reads. */
    enum class walk_stage_t : u8_t {

        /** Finished, its pool merged into its row, or never armed. */
        finished_k,

        /** Its landmarks are in flight in hop slot zero, and every row that lands enters the
         *  frontier; spare hop slots may already expand the landmarks that landed. */
        seeding_k,

        /** Its landmarks all landed, and its expansions, or the scores they issued, are in
         *  flight. */
        expanding_k,

        /** A sated fold or a refused re-table stopped it; its outstanding reads land unscored, then
         *  it finishes. */
        draining_k,
    };

    /** One walk of one row over one shard: its pools, its visited set, and the flights it waits
     *  on. */
    struct walk_t {

        /** Nodes scored so far, the best ascending. */
        results_array_t results;

        /** Nodes worth expanding next. */
        frontier_queue_t frontier;

        /** Nodes already read for scoring; exact by design. */
        visited_set_t visited;

        /** Landed score flights awaiting the next quad. */
        fu::limited_array<u32_t, 4> staged;

        /** Draws this walk's landmarks. */
        splitmix64_t random_generator;

        /** The shard walked, positional among the call's shards. */
        u32_t shard = 0;

        /** The row slot this walk belongs to. */
        u32_t row_slot = 0;

        /** Where the walk stands. */
        walk_stage_t stage = walk_stage_t::finished_k;

        explicit walk_t(scratch_allocator_t allocator) noexcept
            : results(candidates_allocator_t(allocator)), frontier(candidates_allocator_t(allocator)),
              visited(allocator) {}
    };

    /** Why a walk's read was issued, which routes its landing. */
    enum class purpose_t : u8_t {

        /** A value row, scored against the query. */
        score_k,

        /** A neighbors row, extending the frontier. */
        expand_k,

        /** A key row, for a keyed fold or a written result. */
        key_k,
    };

    /** One scored, expanded or resolved row in flight. Its first read lands in the buffer of the
     *  flight's own index, and a keyed score's key in the buffer one flight count past it. The
     *  shard it reads follows from its issuer, which keeps a flight at 16 bytes. */
    struct flight_t {

        /** The walk that issued a score or an expansion, or the row slot behind a resolution. */
        u32_t issuer = 0;

        /** The row's node ID inside its shard. */
        node_id_t node = 0;

        /** Where in its issuer the landing counts: a walk's hop slot, or a row slot's output. */
        u32_t position = 0;

        /** Reads still to land: two for a keyed score, one otherwise. */
        u8_t reads_left = 0;

        /** What its landing does: @c score_k scores, @c expand_k expands, @c key_k resolves. */
        purpose_t purpose = purpose_t::score_k;
    };

    /** Where a row slot stands. */
    enum class row_stage_t : u8_t {

        /** Free for the next row. */
        idle_k,

        /** Some of its walks are still going. */
        walking_k,

        /** Every walk finished, and its written locators are resolving into keys. */
        resolving_k,
    };

    /** One row in flight on a thread: its walks still going, and its resolution's progress. */
    struct row_slot_t {

        /** Walks of the row not finished yet. */
        u32_t walks_left = 0;

        /** Locators the row asks to resolve. */
        u32_t outputs = 0;

        /** The next locator to issue a key read for. */
        u32_t next_output = 0;

        /** Key reads of the row in flight. */
        u32_t resolving = 0;

        /** Where the slot stands. */
        row_stage_t stage = row_stage_t::idle_k;

        /** The batch row the slot holds. */
        std::size_t row = 0;

        /** When the row's walks armed, which its latency counts from; unset when the call collects
         *  none. */
        std::chrono::steady_clock::time_point armed;
    };

    /** The sizes and shared words every thread of one call reads. */
    struct walk_shape_t {

        /** The distance kind every walk ranks by. */
        nk_kernel_kind_t kernel = nk_kernel_unknown_k;

        /** Rows of the batch. */
        std::size_t rows_count = 0;

        /** Each walk's pool breadth. */
        std::size_t expansion_factor = 0;

        /** Flights one walk ever holds at once: its landmarks or one expansion and every neighbor
         *  it scores, and one expansion and its neighbors more per further hop in flight. */
        std::size_t walk_flights = 0;

        /** Rows a thread keeps in flight. */
        std::size_t rows_per_thread = 0;

        /** Hops each walk keeps in flight: the queue depth its thread's rows leave idle, one for a
         *  full batch. */
        std::size_t hops_per_walk = 1;

        /** Buffers per flight: two when keys land beside values. */
        std::size_t buffers_per_flight = 1;

        /** Bytes every thread's arena carves for its walks, rows and flights. */
        std::size_t scratch_bytes = 0;

        /** Staging buffers every engine holds: @c buffers_per_flight per flight. */
        std::size_t buffers = 0;

        /** Bytes of one staging buffer: the covering read of the widest row any part holds. */
        std::size_t buffer_bytes = 0;

        /** The next row any thread takes. */
        std::size_t* cursor = nullptr;

        /** The first failure any thread met, which stops every thread arming rows. */
        status_t* failure = nullptr;

        /** Nanoseconds per row from its first armed walk to its final write; empty collects
         *  none. */
        std::span<u64_t> latencies;
    };

    /** Bytes of one staging buffer over every part in @p parts: the covering read of the widest row
     *  any part holds, which walks and scans alike read into. */
    static std::size_t staging_bytes_(std::span<medium_t const> media, std::span<nvme_part_t const> parts) noexcept {
        u64_t bytes = 0;
        for (std::size_t part = 0; part != parts.size(); ++part) {
            u64_t const block = block_bytes_of(media[part]);
            u64_t const neighbors_bytes = parts[part].connectivity * sizeof(node_id_t);
            bytes = max_of(bytes, max_of(max_of(nvme_covering_bytes(parts[part].value_row_bytes, block),
                                                nvme_covering_bytes(parts[part].key_row_bytes, block)),
                                         nvme_covering_bytes(neighbors_bytes, block)));
        }
        return static_cast<std::size_t>(bytes);
    }

    /** The shape of a walk of @p rows_count rows of @p row_type_ over every part in @p parts, each
     *  walk @p expansion_factor wide and ranking by @p kernel, on @p threads engines @p queue_depth
     *  reads deep. */
    template <typename row_type_>
    static walk_shape_t walk_shape_(std::span<medium_t const> media, std::span<nvme_part_t const> parts,
                                    nk_kernel_kind_t kernel, std::size_t queue_depth, std::size_t threads,
                                    std::size_t rows_count, std::size_t expansion_factor) noexcept {
        using slice_t = arena_allocator<byte_t>;
        std::size_t const connectivity = parts[0].connectivity;
        walk_shape_t shape;
        shape.kernel = kernel;
        shape.rows_count = rows_count;
        shape.expansion_factor = expansion_factor;
        shape.buffers_per_flight = row_type_::key_reads_k == nvme_key_reads_t::scored_k ? 2 : 1;
        std::size_t const first_hop = max_of<std::size_t>(round_up_to_multiple<std::size_t>(connectivity, 4),
                                                          connectivity + 1);
        std::size_t const hop_depth = parts.size() * first_hop * shape.buffers_per_flight;
        // Rows fill the queue first, since a row costs no speculation; depth a short batch leaves
        // idle widens walks.
        shape.rows_per_thread = min_of(max_of<std::size_t>(1, queue_depth / hop_depth),
                                       max_of<std::size_t>(1, divide_round_up(rows_count, threads)));
        shape.hops_per_walk = max_of<std::size_t>(1, queue_depth / (shape.rows_per_thread * hop_depth));
        shape.walk_flights = first_hop + (shape.hops_per_walk - 1) * (connectivity + 1);
        std::size_t const rows = shape.rows_per_thread, walks = rows * parts.size();
        std::size_t const flights = walks * shape.walk_flights;
        shape.buffers = flights * shape.buffers_per_flight;
        // What `row_walker::reserve` carves, slice by slice, after padding to the first boundary.
        std::size_t const per_walk = slice_t::reserved_bytes(results_array_t::memory_usage(expansion_factor)) +
                                     slice_t::reserved_bytes(frontier_queue_t::memory_usage(expansion_factor * 4)) +
                                     slice_t::reserved_bytes(
                                         visited_set_t::memory_usage(expansion_factor * connectivity));
        shape.scratch_bytes = walks * per_walk + slice_t::reserved_bytes(walks * sizeof(walk_t)) +
                              slice_t::reserved_bytes(rows * sizeof(row_type_)) +
                              slice_t::reserved_bytes(rows * sizeof(row_slot_t)) +
                              slice_t::reserved_bytes(flights * sizeof(flight_t)) +
                              slice_t::reserved_bytes(flights * sizeof(u32_t)) +
                              slice_t::reserved_bytes(walks * shape.hops_per_walk * sizeof(u32_t)) +
                              slice_t::reserved_bytes(shape.buffers * sizeof(io_request_t)) +
                              slice_t::reserved_bytes(shape.buffers * sizeof(io_completion_t)) + default_alignment_k;
        shape.buffer_bytes = staging_bytes_(media, parts);
        return shape;
    }

    /** One call's hold on an executor: the claim, then the media, which detach before the claim
     *  lifts. */
    struct call_t {

        /** The executor's one-caller hold. */
        typename executor_t::claim_t claim;

        /** The media every engine reads until the call ends. */
        typename executor_t::attachment_t attachment;
    };

    /** Claims @p executor, sizes the arena of every thread at @p scratch_bytes and every engine at
     *  @p buffers staging buffers of @p buffer_bytes, and attaches @p media. Nothing rewinds the
     *  arenas again before the call ends, so whatever a call carves lasts all of it. */
    static expected<call_t> begin_call_(executor_t& executor, std::span<medium_t const> media,
                                        std::size_t scratch_bytes, std::size_t buffers,
                                        std::size_t buffer_bytes) noexcept {
        call_t call;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {std::move(call), claim_status};
        call.claim = std::move(claim);
        if (status_t const reserved = executor.reserve(scratch_bytes, buffers, buffer_bytes); failed(reserved))
            return {std::move(call), reserved};
        auto [attachment, attach_status] = executor.attach(media);
        call.attachment = std::move(attachment);
        return {std::move(call), attach_status};
    }

    /** One thread's share of a call: its walks, its rows, its flights and the reads not yet
     *  enqueued, driven until the thread's rows run out. */
    template <typename row_type_>
    class row_walker {
      public:
        using row_t = row_type_;
        using key_t = typename row_t::key_t;

        row_walker(engine_t& engine, arena_t& arena, std::span<medium_t const> media,
                   std::span<nvme_part_t const> parts, walk_shape_t const& shape) noexcept
            : engine_(engine), arena_(arena), media_(media), parts_(parts), shape_(shape) {}

        row_walker(row_walker const&) = delete;
        row_walker& operator=(row_walker const&) = delete;

        ~row_walker() noexcept {
            for (walk_t& walk : walks_) walk.~walk_t();
            for (row_t& row : rows_) row.~row_t();
        }

        /** Carves every walk, row and flight of the thread, each row built from @p inputs. */
        status_t reserve(typename row_t::inputs_t const& inputs) noexcept {
            std::size_t const parts = parts_.size(), rows = shape_.rows_per_thread;
            std::size_t const walks = rows * parts, flights = walks * shape_.walk_flights;
            walk_t* const walk_slots = carve_<walk_t>(walks);
            row_t* const row_slots = carve_<row_t>(rows);
            row_slot_t* const slot_states = carve_<row_slot_t>(rows);
            flight_t* const flight_states = carve_<flight_t>(flights);
            u32_t* const free_flights = carve_<u32_t>(flights);
            u32_t* const hop_flights = carve_<u32_t>(walks * shape_.hops_per_walk);
            io_request_t* const backlog = carve_<io_request_t>(shape_.buffers);
            io_completion_t* const completions = carve_<io_completion_t>(shape_.buffers);
            if (!walk_slots || !row_slots || !slot_states || !flight_states || !free_flights || !hop_flights ||
                !backlog || !completions)
                return status_t::bad_alloc_k;

            scratch_allocator_t const allocator {arena_};
            std::size_t const expansion_factor = shape_.expansion_factor, connectivity = parts_[0].connectivity;
            for (std::size_t walk = 0; walk != walks; ++walk) {
                walk_t& constructed = *new (walk_slots + walk) walk_t {allocator};
                walks_ = {walk_slots, walk + 1};
                if (failed(constructed.results.reserve(expansion_factor)) ||
                    failed(constructed.frontier.reserve(expansion_factor * 4)) ||
                    failed(constructed.visited.reserve(expansion_factor * connectivity)))
                    return status_t::bad_alloc_k;
            }
            for (std::size_t row = 0; row != rows; ++row) {
                row_t& constructed = *new (row_slots + row) row_t {inputs, allocator};
                rows_ = {row_slots, row + 1};
                if (status_t const reserved = constructed.reserve(parts, connectivity); failed(reserved))
                    return reserved;
            }
            for (std::size_t row = 0; row != rows; ++row) new (slot_states + row) row_slot_t {};
            for (std::size_t flight = 0; flight != flights; ++flight)
                free_flights[flight] = static_cast<u32_t>(flights - 1 - flight);
            slots_ = {slot_states, rows};
            flights_ = {flight_states, flights};
            free_flights_ = {free_flights, flights};
            hop_flights_ = {hop_flights, walks * shape_.hops_per_walk};
            free_count_ = flights;
            backlog_ = {backlog, shape_.buffers};
            completions_ = {completions, shape_.buffers};
            return status_t::success_k;
        }

        /** Arms rows until the batch runs out, keeping every slot's walks in flight, and drains the
         *  engine before it returns, whatever happened; a failure on any thread stops this one too.
         *  Answers the first row that refused, since a failure is published to every thread
         *  instead. */
        status_t run() noexcept {
            while (true) {
                if (succeeded(failure_)) {
                    if (status_t const shared = shared_failure_(); failed(shared)) failure_ = shared;
                    else arm_rows_();
                }
                if (failed(failure_) && !outstanding_) break;
                if (succeeded(failure_)) {
                    if (!rows_in_flight_ && exhausted_()) break;
                    enqueue_backlog_();
                }
                if (!outstanding_) {
                    assert((failed(failure_) || !rows_in_flight_) && "A row in flight always waits on a read");
                    continue;
                }
                if (status_t const submitted = engine_.submit(io_wait_t::blocking_k); failed(submitted)) {
                    // A refused submission returns with nothing in flight and nothing left to poll.
                    fail(submitted);
                    outstanding_ = 0;
                    break;
                }
                telemetry_.round_trips += rows_in_flight_;
                for (std::size_t polled = engine_.poll_completions(completions_); polled;
                     polled = engine_.poll_completions(completions_)) {
                    // Completions of walk reads move to the front, as their flights may serve
                    // other rows by the end of the batch.
                    std::size_t advancing = 0;
                    for (io_completion_t const& completion : completions_.first(polled))
                        if (land_(completion)) completions_[advancing++] = completion;
                    // Walks advance once the whole batch landed, so their next hops weigh every row
                    // that did.
                    if (succeeded(failure_))
                        for (io_completion_t const& completion : completions_.first(advancing))
                            advance_(static_cast<u32_t>(completion.context));
                }
            }
            return refusal_;
        }

        /** What the thread counted. */
        search_result_t const& telemetry() const noexcept { return telemetry_; }

        /** Records the thread's first failure and publishes it, so every thread stops arming
         *  rows. */
        void fail(status_t failure) noexcept {
            if (succeeded(failure_)) failure_ = failure;
            status_t unset = status_t::success_k;
            std::atomic_ref<status_t>(*shape_.failure)
                .compare_exchange_strong(unset, failure, std::memory_order_relaxed, std::memory_order_relaxed);
        }

      private:
        template <typename value_type_>
        value_type_* carve_(std::size_t count) noexcept {
            return reinterpret_cast<value_type_*>(
                arena_.allocate(max_of<std::size_t>(count, 1) * sizeof(value_type_), default_alignment_k));
        }

        status_t shared_failure_() const noexcept {
            return std::atomic_ref<status_t>(*shape_.failure).load(std::memory_order_relaxed);
        }

        bool exhausted_() const noexcept {
            return std::atomic_ref<std::size_t>(*shape_.cursor).load(std::memory_order_relaxed) >= shape_.rows_count;
        }

        u32_t take_flight_() noexcept { return free_flights_[--free_count_]; }
        void release_flight_(u32_t flight) noexcept { free_flights_[free_count_++] = flight; }

        /** The flights each hop slot of @p walk_index waits on; a zero slot is free. */
        std::span<u32_t> hop_flights_of_(u32_t walk_index) noexcept {
            return hop_flights_.subspan(walk_index * shape_.hops_per_walk, shape_.hops_per_walk);
        }

        /** Whether a hop slot of @p walk_index still waits on a flight. */
        bool in_flight_(u32_t walk_index) noexcept {
            for (u32_t const flights : hop_flights_of_(walk_index))
                if (flights) return true;
            return false;
        }

        /** Stops @p walk after a scratch refusal, counted as saturated: its outstanding reads land
         *  unscored, then it finishes. */
        void saturate_(walk_t& walk) noexcept {
            ++telemetry_.saturated_walks;
            walk.stage = walk_stage_t::draining_k;
        }

        /** The buffer a keyed score's key lands in: one flight count past its value's. */
        io_buffer_id_t key_buffer_(u32_t flight_index) const noexcept {
            return static_cast<io_buffer_id_t>(flight_index + flights_.size());
        }

        /** The shard @p flight reads: its walk's, or the shard of the output a resolution fills. */
        u32_t shard_of_(flight_t const& flight) const noexcept {
            return flight.purpose == purpose_t::key_k ? rows_[flight.issuer].locator(flight.position).shard
                                                      : walks_[flight.issuer].shard;
        }

        /** Where the row of @p flight starts in the column @p purpose reads, from the medium's byte
         *  zero. */
        u64_t row_begin_(flight_t const& flight, purpose_t purpose) const noexcept {
            u32_t const shard = shard_of_(flight);
            nvme_part_t const& part = parts_[shard];
            u64_t const base = byte_base_of(media_[shard]);
            switch (purpose) {
            case purpose_t::score_k: return base + part.values_begin + u64_t(flight.node) * part.value_row_bytes;
            case purpose_t::expand_k:
                return base + part.neighbors_begin + u64_t(flight.node) * part.connectivity * sizeof(node_id_t);
            case purpose_t::key_k: return base + part.keys_begin + u64_t(flight.node) * part.key_row_bytes;
            }
            return base;
        }

        u64_t row_bytes_(flight_t const& flight, purpose_t purpose) const noexcept {
            nvme_part_t const& part = parts_[shard_of_(flight)];
            switch (purpose) {
            case purpose_t::score_k: return part.value_row_bytes;
            case purpose_t::expand_k: return part.connectivity * sizeof(node_id_t);
            case purpose_t::key_k: return part.key_row_bytes;
            }
            return 0;
        }

        /** The landed row of @p flight_index in the column @p purpose read, inside @p buffer. */
        byte_t const* landed_row_(u32_t flight_index, purpose_t purpose, io_buffer_id_t buffer) noexcept {
            flight_t const& flight = flights_[flight_index];
            return engine_.buffer(buffer) + row_begin_(flight, purpose) % block_bytes_of(media_[shard_of_(flight)]);
        }

        /** Stages the covering read of @p flight_index 's row in the column @p purpose reads, into
         *  @p buffer, its context naming the flight's issuer. */
        void issue_(u32_t flight_index, purpose_t purpose, io_buffer_id_t buffer) noexcept {
            flight_t const& flight = flights_[flight_index];
            u32_t const shard = shard_of_(flight);
            io_request_t const request = nvme_covering_read(row_begin_(flight, purpose), row_bytes_(flight, purpose),
                                                            block_bytes_of(media_[shard]), shard, buffer,
                                                            flight.issuer);
            backlog_[backlog_size_++] = request;
            nvme_reads_t& reads = purpose == purpose_t::expand_k ? telemetry_.neighbor_reads
                                  : purpose == purpose_t::key_k  ? telemetry_.key_reads
                                                                 : telemetry_.value_reads;
            reads += nvme_reads_t {1, request.length};
        }

        /** Hands the staged reads to the engine until it holds no more; the rest wait for
         *  completions. */
        void enqueue_backlog_() noexcept {
            while (backlog_size_) {
                status_t const enqueued = engine_.enqueue_read(backlog_[backlog_size_ - 1]);
                if (enqueued == status_t::overflow_k && outstanding_) return;
                // A full engine holding none of this thread's reads would never drain, so the
                // thread fails instead.
                if (failed(enqueued)) return fail(enqueued == status_t::overflow_k ? status_t::io_k : enqueued);
                --backlog_size_;
                ++outstanding_;
            }
        }

        /** Takes the next rows of the batch into every idle slot and seeds all of their walks. */
        void arm_rows_() noexcept {
            std::size_t const parts = parts_.size();
            for (u32_t slot_index = 0; slot_index != slots_.size(); ++slot_index) {
                row_slot_t& slot = slots_[slot_index];
                if (slot.stage != row_stage_t::idle_k) continue;
                std::size_t const row =
                    std::atomic_ref<std::size_t>(*shape_.cursor).fetch_add(1, std::memory_order_relaxed);
                if (row >= shape_.rows_count) return;
                slot = row_slot_t {static_cast<u32_t>(parts), 0, 0, 0, row_stage_t::walking_k, row, {}};
                if (!shape_.latencies.empty()) slot.armed = std::chrono::steady_clock::now();
                ++rows_in_flight_;
                row_t& armed = rows_[slot_index];
                armed.arm(row);
                armed.query_norms = norms_(shape_.kernel, armed.query);
                for (u32_t shard = 0; shard != parts; ++shard) {
                    u32_t const walk_index = slot_index * static_cast<u32_t>(parts) + shard;
                    walk_t& walk = walks_[walk_index];
                    walk.results.clear();
                    walk.frontier.clear();
                    walk.visited.clear();
                    walk.staged.clear();
                    walk.random_generator = splitmix64_t {mix_bits(u64_t(row) * parts + shard + 1)};
                    walk.shard = shard;
                    walk.row_slot = slot_index;
                    walk.stage = walk_stage_t::seeding_k;
                    std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
                    std::memset(hop_flights.data(), 0, hop_flights.size_bytes());
                    std::size_t const landmarks = round_up_to_multiple<std::size_t>(parts_[shard].connectivity, 4);
                    if (!seed_(walk_index, landmarks)) finish_walk_(walk_index);
                }
            }
        }

        /** Issues up to @p count fresh landmarks of walk @p walk_index as one hop, answering how
         *  many. A walk seeds only with nothing in flight, so the hop takes slot zero. */
        std::size_t seed_(u32_t walk_index, std::size_t count) noexcept {
            walk_t& walk = walks_[walk_index];
            u32_t const nodes = parts_[walk.shard].nodes;
            if (failed(walk.visited.reserve(walk.visited.size() + count))) return saturate_(walk), 0;
            std::size_t issued = 0;
            for (; issued != count && walk.visited.size() < nodes; ++issued) {
                node_id_t landmark = uniform_below<node_id_t> {nodes}(walk.random_generator);
                while (walk.visited.contains(landmark)) landmark = landmark + 1 == nodes ? 0 : landmark + 1;
                [[maybe_unused]] membership_t const marked = walk.visited.insert(landmark);
                issue_score_(walk_index, landmark, 0);
            }
            std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
            hop_flights[0] = static_cast<u32_t>(issued);
            return issued;
        }

        void issue_score_(u32_t walk_index, node_id_t node, u32_t hop) noexcept {
            u32_t const flight = take_flight_();
            flights_[flight] = flight_t {walk_index, node, hop, static_cast<u8_t>(shape_.buffers_per_flight),
                                         purpose_t::score_k};
            issue_(flight, purpose_t::score_k, flight);
            if constexpr (row_t::key_reads_k == nvme_key_reads_t::scored_k)
                issue_(flight, purpose_t::key_k, key_buffer_(flight));
        }

        /** Routes one completion to its flight, answering whether its issuer is a walk the batch's
         *  end advances; a read short of its row fails the thread. */
        bool land_(io_completion_t const& completion) noexcept {
            --outstanding_;
            if (failed(failure_)) return false;
            u32_t const flight_index = static_cast<u32_t>(completion.buffer % flights_.size());
            flight_t& flight = flights_[flight_index];
            purpose_t const purpose = completion.buffer == flight_index ? flight.purpose : purpose_t::key_k;
            u64_t const row_begin = row_begin_(flight, purpose);
            u64_t const needed = row_begin % block_bytes_of(media_[shard_of_(flight)]) + row_bytes_(flight, purpose);
            if (completion.result < 0 || static_cast<u64_t>(completion.result) < needed)
                return fail(status_t::io_k), false;
            if (--flight.reads_left) return false;
            switch (flight.purpose) {
            case purpose_t::score_k: {
                walk_t& walk = walks_[flight.issuer];
                std::span<u32_t> const hop_flights = hop_flights_of_(flight.issuer);
                [[maybe_unused]] fu::status_t const queued = walk.staged.push_back(flight_index);
                // A quad scores once full or once a hop's last row lands, whichever hops its lanes
                // came from.
                if (--hop_flights[flight.position] && !walk.staged.full()) return true;
                flush_(walk);
                // Landmarks take hop slot zero, so seeding ends once that slot drains.
                if (walk.stage == walk_stage_t::seeding_k && !hop_flights[0]) walk.stage = walk_stage_t::expanding_k;
                return true;
            }
            case purpose_t::expand_k: expand_(flight.issuer, flight_index); return true;
            case purpose_t::key_k: {
                u32_t const slot_index = flight.issuer;
                rows_[slot_index].resolve(flight.position, landed_row_(flight_index, purpose_t::key_k, flight_index));
                release_flight_(flight_index);
                --slots_[slot_index].resolving;
                resolve_row_(slot_index);
                return false;
            }
            }
            return false;
        }

        /** Scores the staged flights of @p walk as one quad, rescores and folds them through its
         *  row, and merges them into its pool; short tails replicate lane zero, and only real lanes
         *  reach the pool or the telemetry. */
        void flush_(walk_t& walk) noexcept {
            std::size_t const fill = walk.staged.size();
            if (!fill) return;
            if (walk.stage == walk_stage_t::draining_k) {
                for (u32_t const flight : walk.staged) release_flight_(flight);
                return walk.staged.clear();
            }
            constexpr bool keyed = row_t::key_reads_k == nvme_key_reads_t::scored_k;
            row_t& row = rows_[walk.row_slot];
            nvme_part_t const& part = parts_[walk.shard];
            std::array<node_id_t, 4> ids;
            std::array<vector_view_t, 4> vectors;
            std::array<vector_norms_t, 4> norms;
            std::array<key_t, 4> keys {};
            for (std::size_t lane = 0; lane != 4; ++lane) {
                u32_t const source = walk.staged[min_of<std::size_t>(lane, fill - 1)];
                ids[lane] = flights_[source].node;
                vectors[lane] = vector_view_t {
                    reinterpret_cast<scalar_t const*>(landed_row_(source, purpose_t::score_k, source)),
                    part.dimensions};
                norms[lane] = norms_(shape_.kernel, vectors[lane]);
                if constexpr (keyed)
                    std::memcpy(&keys[lane], landed_row_(source, purpose_t::key_k, key_buffer_(source)), sizeof(key_t));
            }
            std::span<key_t const> const lane_keys = std::span<key_t const>(keys).first(keyed ? fill : 0);
            scores_result<dot_result_t, 4> scores = dense::scores<capability_, scalar_t>(
                shape_.kernel, row.query, row.query_norms, vectors, norms);
            row.rescore(ids, vectors, norms, scores, fill, lane_keys);
            telemetry_.computed_distances += fill;
            std::array<scored_node_t, 4> scored_batch;
            for (std::size_t lane = 0; lane != fill; ++lane)
                scored_batch[lane] = scored_node_t {ids[lane], scores.distances[lane]};
            std::span<scored_node_t> const scored = std::span(scored_batch).first(fill);
            std::span<dot_result_t const> const dots = std::span<dot_result_t const>(scores.dots).first(fill);
            auto&& fold = row.fold(walk);
            walk_demand_t demand = walk_demand_t::continue_k;
            if constexpr (keyed) demand = fold.absorb(scored, dots, lane_keys);
            else demand = fold.absorb(scored, dots);
            for (u32_t const flight : walk.staged) release_flight_(flight);
            walk.staged.clear();
            if (demand == walk_demand_t::sated_k) {
                walk.stage = walk_stage_t::draining_k;
                return;
            }
            network_sort(scored, closer_scored_node_t {});
            std::size_t const accepted = walk.results.insert(scored, shape_.expansion_factor);
            // Landmarks all enter the frontier, as the in-memory walk seeds it; neighbors only when
            // the pool took them.
            std::size_t const queued = walk.stage == walk_stage_t::seeding_k ? fill : accepted;
            for (std::size_t lane = 0; lane != queued; ++lane)
                if (failed(walk.frontier.insert(scored_node_t(scored[lane])))) return saturate_(walk);
        }

        /** Expands the closest frontier nodes into the free hop slots of @p walk_index while each
         *  lies inside a full pool's radius; with nothing in flight, re-seeds a frontier drained
         *  before the bound fills, or finishes. */
        void advance_(u32_t walk_index) noexcept {
            walk_t& walk = walks_[walk_index];
            if (walk.stage == walk_stage_t::finished_k) return;
            if (walk.stage == walk_stage_t::draining_k) {
                if (!in_flight_(walk_index)) finish_walk_(walk_index);
                return;
            }
            row_t const& row = rows_[walk.row_slot];
            std::size_t const expansion_factor = shape_.expansion_factor;
            results_array_t const& bound = row.bound(walk);
            results_array_t const& pool = bound.size() == expansion_factor ? bound : walk.results;
            bool const pool_full = pool.size() == expansion_factor;
            std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
            for (u32_t slot = 0; slot != hop_flights.size() && !walk.frontier.empty(); ++slot) {
                if (hop_flights[slot]) continue;
                scored_node_t const candidate = walk.frontier.top();
                if (pool_full && candidate.distance > pool.top().distance) break;
                if (pool_full && walk.frontier.capacity() - walk.frontier.size() < parts_[walk.shard].connectivity)
                    walk.frontier.erase_below(pool.top());
                walk.frontier.pop();
                ++telemetry_.visited_members;
                u32_t const flight = take_flight_();
                flights_[flight] = flight_t {walk_index, candidate.node_id, slot, 1, purpose_t::expand_k};
                issue_(flight, purpose_t::expand_k, flight);
                hop_flights[slot] = 1;
            }
            if (in_flight_(walk_index)) return;
            // A walk only reaches nodes connected to its seeds, so a frontier drained before the
            // bound fills re-seeds.
            if (walk.frontier.empty() && bound.size() < expansion_factor) {
                walk.stage = walk_stage_t::seeding_k;
                if (seed_(walk_index, 4)) return void(++telemetry_.restarts);
            }
            finish_walk_(walk_index);
        }

        /** Reads the landed neighbors row of @p flight_index and issues a score for every unvisited
         *  neighbor, in the expansion's own hop. */
        void expand_(u32_t walk_index, u32_t flight_index) noexcept {
            walk_t& walk = walks_[walk_index];
            nvme_part_t const& part = parts_[walk.shard];
            u32_t const hop = flights_[flight_index].position;
            std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
            --hop_flights[hop];
            if (walk.stage != walk_stage_t::draining_k) {
                if (failed(walk.visited.reserve(walk.visited.size() + part.connectivity))) saturate_(walk);
                else {
                    auto const* const neighbors = reinterpret_cast<node_id_t const*>(
                        landed_row_(flight_index, purpose_t::expand_k, flight_index));
                    for (std::size_t slot = 0; slot != part.connectivity; ++slot) {
                        node_id_t const neighbor = neighbors[slot];
                        if (neighbor >= part.nodes || walk.visited.insert(neighbor) == membership_t::present_k)
                            continue;
                        issue_score_(walk_index, neighbor, hop);
                        ++hop_flights[hop];
                    }
                }
            }
            release_flight_(flight_index);
        }

        /** Merges @p walk_index into its row, and once every walk of the row finished, starts the
         *  row's resolution. */
        void finish_walk_(u32_t walk_index) noexcept {
            walk_t& walk = walks_[walk_index];
            walk.stage = walk_stage_t::finished_k;
            u32_t const slot_index = walk.row_slot;
            row_slot_t& slot = slots_[slot_index];
            assert(slot.stage == row_stage_t::walking_k && "Only a walking row's walks finish");
            rows_[slot_index].finish_walk(walk);
            if (--slot.walks_left) return;
            slot.stage = row_stage_t::resolving_k;
            slot.outputs = static_cast<u32_t>(rows_[slot_index].outputs());
            resolve_row_(slot_index);
        }

        /** Issues key reads for the row's written locators within the row's flight budget, and
         *  finishes the row once every one landed. */
        void resolve_row_(u32_t slot_index) noexcept {
            row_slot_t& slot = slots_[slot_index];
            assert(slot.stage == row_stage_t::resolving_k && "Only a resolving row reads keys");
            row_t& row = rows_[slot_index];
            std::size_t const budget = parts_.size() * shape_.walk_flights;
            while (slot.next_output != slot.outputs && slot.resolving != budget) {
                nvme_locator_t const locator = row.locator(slot.next_output);
                u32_t const flight = take_flight_();
                flights_[flight] = flight_t {slot_index, locator.node, slot.next_output, 1, purpose_t::key_k};
                issue_(flight, purpose_t::key_k, flight);
                ++slot.next_output;
                ++slot.resolving;
            }
            if (slot.resolving) return;
            if (status_t const finished = row.finish(); failed(finished) && succeeded(refusal_)) refusal_ = finished;
            if (!shape_.latencies.empty())
                shape_.latencies[slot.row] = static_cast<u64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - slot.armed)
                        .count());
            slot.stage = row_stage_t::idle_k;
            --rows_in_flight_;
        }

        engine_t& engine_;
        arena_t& arena_;
        std::span<medium_t const> media_;
        std::span<nvme_part_t const> parts_;
        walk_shape_t const& shape_;

        std::span<walk_t> walks_;
        std::span<row_t> rows_;
        std::span<row_slot_t> slots_;
        std::span<flight_t> flights_;
        std::span<u32_t> free_flights_;
        std::span<u32_t> hop_flights_;
        std::span<io_request_t> backlog_;
        std::span<io_completion_t> completions_;
        std::size_t free_count_ = 0;
        std::size_t backlog_size_ = 0;
        std::size_t outstanding_ = 0;
        std::size_t rows_in_flight_ = 0;
        search_result_t telemetry_;

        /** The read failure that stops this thread. */
        status_t failure_ = status_t::success_k;

        /** The first row that refused, which leaves the other rows standing. */
        status_t refusal_ = status_t::success_k;
    };

    /**
     *  @brief Walks every row of a batch over every part in @p parts, the rows' walks interleaved
     *      on every thread of @p executor, each row built from @p inputs as @p row_type_ says.
     *
     *  Runs inside a call @ref begin_call_ began at @p shape 's sizes, and sums the threads'
     *  telemetry; @p counts are the rows' own, summed into the result's @c count. @p latencies
     *  takes one duration per row, or is empty.
     */
    template <typename row_type_>
    static expected<search_result_t> run_walks_(std::span<medium_t const> media, std::span<nvme_part_t const> parts,
                                                executor_t& executor, walk_shape_t shape,
                                                typename row_type_::inputs_t const& inputs,
                                                std::span<std::size_t const> counts,
                                                std::span<u64_t> latencies) noexcept {
        search_result_t merged;
        if (!latencies.empty() && latencies.size() != shape.rows_count) return {merged, status_t::config_mismatch_k};
        alignas(default_alignment_k) std::size_t cursor = 0;
        status_t failure = status_t::success_k;
        shape.cursor = &cursor;
        shape.failure = &failure;
        shape.latencies = latencies;
        status_t refusal = status_t::success_k;
        auto add = [](std::size_t& into, std::size_t value) noexcept {
            std::atomic_ref<std::size_t>(into).fetch_add(value, std::memory_order_relaxed);
        };
        auto add_reads = [&](nvme_reads_t& into, nvme_reads_t const& reads) noexcept {
            add(into.count, reads.count), add(into.bytes, reads.bytes);
        };
        auto walk_thread = [&](std::size_t thread, std::size_t) noexcept {
            row_walker<row_type_> walker {executor.engine(thread), executor.arena(thread), media, parts, shape};
            status_t const reserved = walker.reserve(inputs);
            if (failed(reserved)) walker.fail(reserved);
            status_t const refused = walker.run();
            search_result_t const& telemetry = walker.telemetry();
            add(merged.visited_members, telemetry.visited_members);
            add(merged.computed_distances, telemetry.computed_distances);
            add(merged.restarts, telemetry.restarts);
            add(merged.saturated_walks, telemetry.saturated_walks);
            add_reads(merged.neighbor_reads, telemetry.neighbor_reads);
            add_reads(merged.value_reads, telemetry.value_reads);
            add_reads(merged.key_reads, telemetry.key_reads);
            add(merged.round_trips, telemetry.round_trips);
            status_t unset = status_t::success_k;
            if (failed(refused))
                std::atomic_ref<status_t>(refusal).compare_exchange_strong(unset, refused, std::memory_order_relaxed,
                                                                           std::memory_order_relaxed);
        };
        if (status_t const fanned = executor.for_threads(walk_thread); failed(fanned)) return {merged, fanned};
        for (std::size_t const count : counts) merged.count += count;
        return {merged, failed(failure) ? failure : refusal};
    }

    /** The batch @ref run_walks_ walks, inside a call of its own: @p rows_count rows over every
     *  part in @p parts, each walk @p expansion_factor wide and ranking by @p kernel. */
    template <typename row_type_>
    static expected<search_result_t> walk_rows_(std::span<medium_t const> media, std::span<nvme_part_t const> parts,
                                                nk_kernel_kind_t kernel, executor_t& executor, std::size_t rows_count,
                                                std::size_t expansion_factor,
                                                typename row_type_::inputs_t const& inputs,
                                                std::span<std::size_t const> counts,
                                                std::span<u64_t> latencies) noexcept {
        if (parts.empty()) return {{}, status_t::uninitialized_k};
        if (!rows_count) return {{}, status_t::success_k};
        walk_shape_t const shape = walk_shape_<row_type_>(media, parts, kernel, executor.queue_depth(), executor.size(),
                                                          rows_count, expansion_factor);
        auto [call, call_status] = begin_call_(executor, media, shape.scratch_bytes, shape.buffers, shape.buffer_bytes);
        if (failed(call_status)) return {{}, call_status};
        return run_walks_<row_type_>(media, parts, executor, shape, inputs, counts, latencies);
    }

#pragma endregion Walks

#pragma region Rows

    /** The rows of a batched search over this one part, ranked by node ID: no key resolves. */
    template <typename heads_type_>
    struct node_search_row {
        using key_t = node_id_t;
        using fold_t = dummy_fold_t;
        inline static constexpr nvme_key_reads_t key_reads_k = nvme_key_reads_t::outputs_k;

        /** The batch every row of the call reads and writes. */
        struct inputs_t {
            vectors_view_t queries;
            heads_type_ const* heads = nullptr;
            nk::matrix_span<node_id_t> node_ids;
            nk::matrix_span<distance_t> distances;
            std::span<std::size_t> counts;
        };

        inputs_t const& inputs;
        std::size_t row = 0;
        vector_view_t query;
        vector_norms_t query_norms {};

        node_search_row(inputs_t const& inputs, scratch_allocator_t) noexcept : inputs(inputs) {}
        status_t reserve(std::size_t, std::size_t) noexcept { return status_t::success_k; }

        void arm(std::size_t new_row) noexcept {
            row = new_row;
            query = inputs.queries.row(row).as_vector();
            inputs.counts[row] = 0;
        }
        void rescore(std::array<node_id_t, 4> const& ids, std::array<vector_view_t, 4> const& vectors,
                     std::array<vector_norms_t, 4> const& norms, scores_result<dot_result_t, 4>& scores,
                     std::size_t count, std::span<key_t const>) const noexcept {
            if constexpr (!std::is_same_v<heads_type_, dummy_heads_t>)
                indexer_rescoring<capability_, scalar_t> {inputs.heads->row(row)}(ids, vectors, norms, scores, count);
        }
        fold_t fold(walk_t const&) const noexcept { return {}; }
        results_array_t const& bound(walk_t const& walk) const noexcept { return walk.results; }
        void finish_walk(walk_t const& walk) noexcept {
            std::size_t const kept = min_of(walk.results.size(), inputs.node_ids.extent(1));
            for (std::size_t rank = 0; rank != kept; ++rank) {
                inputs.node_ids(row, rank) = walk.results[rank].node_id;
                inputs.distances(row, rank) = walk.results[rank].distance;
            }
            inputs.counts[row] = kept;
        }
        std::size_t outputs() const noexcept { return 0; }
        nvme_locator_t locator(std::size_t) const noexcept { return {}; }
        void resolve(std::size_t, byte_t const*) noexcept {}
        status_t finish() noexcept { return status_t::success_k; }
    };

    /** The batched @ref search under @p heads, or as the kit scores under @c dummy_heads_t. */
    template <typename heads_type_>
    expected<search_result_t> search_(vectors_view_t queries, heads_type_ const& heads, executor_t& executor,
                                      search_config_t config, nk::matrix_span<node_id_t> node_ids,
                                      nk::matrix_span<distance_t> distances, std::span<std::size_t> counts) noexcept {
        std::size_t const queries_count = queries.extent(0);
        if (!part_.dimensions) return {{}, status_t::uninitialized_k};
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, node_ids, distances); failed(refusal))
            return {{}, refusal};
        if (queries_count && queries.extent(1) != dimensions()) return {{}, status_t::config_mismatch_k};
        using row_t = node_search_row<heads_type_>;
        typename row_t::inputs_t const inputs {queries, &heads, node_ids, distances, counts};
        return walk_rows_<row_t>(std::span<medium_t const>(&medium_, 1), std::span<nvme_part_t const>(&part_, 1),
                                 kernel_, executor, queries_count, non_zero_expansion_factor(config), inputs, counts,
                                 {});
    }

#pragma endregion Rows

    /** The part's file or device, open for the engines. */
    medium_t medium_;

    /** Where the part keeps its columns. */
    nvme_part_t part_;

    /** The distance kind every search ranks by. */
    nk_kernel_kind_t kernel_ = nk_kernel_angular_k;
};

#pragma endregion Index

} // namespace unum::usearch::dense
