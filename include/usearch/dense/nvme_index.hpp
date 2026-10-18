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

#include <array>       // `std::array`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::is_same_v`

#include <numkong/tensor.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/nvme_device.hpp"       // `nvme_executor`, `file_medium_t`, `io_request_t`
#include "usearch/utilities/nvme_walks.hpp"        // `row_walker`, `nvme_part_t`, `nvme_begin_call`
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`, `sorted_buffer`, `arena_allocator`
#include "usearch/utilities/ram_sorts.hpp"         // `network_sort`
#include "usearch/utilities/ram_tables.hpp"        // `growing_hash_set`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_header`
#include "usearch/dense/types.hpp"
#include "usearch/dense/ram_metric.hpp"
#include "usearch/dense/ram_index.hpp" // `filtering_fold`, `multiply_fold`

namespace unum::usearch::dense {

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
    template <typename, typename>
    friend class utilities::row_walker;

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

    /** The capability bit of the distance kit the ladder actually picked, lower than @c capability_
     *  when no SIMD kit is adapted for this dtype yet. */
    inline static constexpr nk_capability_t picked_quads_capability_k = quads_ladder<scalar_t, capability_>::picked_k;

    /** The distance kit the ladder picked. */
    using quads_kit_t = typename quads_ladder<scalar_t, capability_>::kit_t;

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
        return config.expansion_factor ? config.expansion_factor : default_expansion_factor(connectivity());
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
                                           typename visits_hash_set_ladder<capability_>::kit_t>;

    /** Where a part keeps its columns: every row of a column as wide as the next. */
    using part_t = nvme_part_t;

    /** Where the row of @p flight starts in the column @p purpose reads of @p part, from the part's
     *  own start. */
    static u64_t row_offset_(nvme_part_t const& part, nvme_purpose_t purpose,
                             nvme_flight<u32_t> const& flight) noexcept {
        switch (purpose) {
        case nvme_purpose_t::score_k: return part.values_begin + u64_t(flight.node) * part.value_row_bytes;
        case nvme_purpose_t::expand_k:
            return part.neighbors_begin + u64_t(flight.node) * part.connectivity * sizeof(node_id_t);
        case nvme_purpose_t::key_k: return part.keys_begin + u64_t(flight.node) * part.key_row_bytes;
        }
        return 0;
    }

    /** Bytes of any row of the column @p purpose reads in @p part. */
    static u64_t row_length_(nvme_part_t const& part, nvme_purpose_t purpose, nvme_flight<u32_t> const&) noexcept {
        switch (purpose) {
        case nvme_purpose_t::score_k: return part.value_row_bytes;
        case nvme_purpose_t::expand_k: return part.connectivity * sizeof(node_id_t);
        case nvme_purpose_t::key_k: return part.key_row_bytes;
        }
        return 0;
    }

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

    /** Opens the part at @p path, a file or device or, in SPDK builds, a transport ID: its medium
     *  for the engines, and its header vetted against this instantiation, against @p kernel unless
     *  it is @c nk_kernel_unknown_k, which then takes the metric the header names, and, unless
     *  @p key_dtype is @c unknown_k, for a key column no wider than it. The header comes through
     *  the medium itself, in blocking block-aligned reads. */
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
            auto [file, open_status] = file_medium_t::open(path);
            if (failed(open_status)) return open_status;
            if (status_t const read = read_part_(file, key_dtype, kernel, allocator, part); failed(read)) return read;
            medium = medium_t {std::move(file)};
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

        safegraphs_name_t metric;
        auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "metric") metric = safegraphs_name_t {field.value};
        };
        auto const [header, header_status] = safegraphs_read_header(to_span(header_bytes), allocator, per_custom_field);
        if (failed(header_status)) return header_status;
        if (header.key.offsets || header.value.offsets || header.neighbors.offsets) return status_t::config_mismatch_k;
        if (header.value.dtype != dtype_of(scalar_t::dtype()) || header.neighbors.dtype != dtype_t::u32_k ||
            header.key.shape != column_shape_t {scalar_shape_t {}})
            return status_t::config_mismatch_k;
        if (key_dtype != dtype_t::unknown_k && !safegraphs_key_widens(header.key.dtype, key_dtype))
            return status_t::config_mismatch_k;
        nk_kernel_kind_t const named = metric.empty() ? kernel
                                                      : nk_kernel_named(metric.characters, metric.view().size());
        if (kernel != nk_kernel_unknown_k && named != kernel) return status_t::config_mismatch_k;
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
        part.key_dtype = header.key.dtype;
        part.value_row_bytes = static_cast<u32_t>(header.value.row_bytes());
        part.keys_begin = header.data_offset + header.key.data.first;
        part.values_begin = header.data_offset + header.value.data.first;
        part.neighbors_begin = header.data_offset + header.neighbors.data.first;
        return status_t::success_k;
    }

#pragma endregion Opening

#pragma region Walks

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
        nvme_walk_stage_t stage = nvme_walk_stage_t::finished_k;

        explicit walk_t(scratch_allocator_t allocator) noexcept
            : results(candidates_allocator_t(allocator)), frontier(candidates_allocator_t(allocator)),
              visited(allocator) {}
    };

    /** The sizes and shared words every thread of one call reads. */
    struct walk_shape_t {

        /** The distance kind every walk ranks by. */
        nk_kernel_kind_t kernel = nk_kernel_unknown_k;

        /** Rows of the batch. */
        std::size_t rows_count = 0;

        /** Each walk's pool breadth. */
        std::size_t expansion_factor = 0;

        /** The queue sizes every walk starts from, at that breadth over the widest part's rows. */
        nvme_queue_sizes_t queue_sizes {};

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

        /** The first failure any thread met, which stops every thread priming rows. */
        status_t* failure = nullptr;

        /** Nanoseconds per row from its first primed walk to its final write; empty collects
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
     *  reads deep, every row in flight staging @p staged_bytes of imported queries. */
    template <typename row_type_>
    static walk_shape_t walk_shape_(std::span<medium_t const> media, std::span<nvme_part_t const> parts,
                                    nk_kernel_kind_t kernel, std::size_t queue_depth, std::size_t threads,
                                    std::size_t rows_count, std::size_t expansion_factor,
                                    std::size_t staged_bytes) noexcept {
        using slice_t = arena_allocator<byte_t>;
        std::size_t const connectivity = nvme_widest_row(parts);
        walk_shape_t shape;
        shape.kernel = kernel;
        shape.rows_count = rows_count;
        shape.expansion_factor = expansion_factor;
        shape.queue_sizes = nvme_queue_sizes(expansion_factor, connectivity);
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
        std::size_t const per_walk = slice_t::reserved_bytes(results_array_t::memory_usage(shape.queue_sizes.results)) +
                                     slice_t::reserved_bytes(
                                         frontier_queue_t::memory_usage(shape.queue_sizes.frontier)) +
                                     slice_t::reserved_bytes(visited_set_t::memory_usage(shape.queue_sizes.visited));
        shape.scratch_bytes = walks * per_walk + slice_t::reserved_bytes(walks * sizeof(walk_t)) +
                              slice_t::reserved_bytes(rows * sizeof(row_type_)) +
                              rows * slice_t::reserved_bytes(staged_bytes) +
                              slice_t::reserved_bytes(rows * sizeof(nvme_row_slot_t)) +
                              slice_t::reserved_bytes(flights * sizeof(nvme_flight<u32_t>)) +
                              slice_t::reserved_bytes(flights * sizeof(u32_t)) +
                              slice_t::reserved_bytes(walks * shape.hops_per_walk * sizeof(u32_t)) +
                              slice_t::reserved_bytes(shape.buffers * sizeof(io_request_t)) +
                              slice_t::reserved_bytes(shape.buffers * sizeof(io_completion_t)) + default_alignment_k;
        shape.buffer_bytes = staging_bytes_(media, parts);
        return shape;
    }

    /** Prepares the query of a primed @p row: the norms the call's kernel reads of it. */
    template <typename row_type_>
    static void prime_scoring_(walk_shape_t const& shape, row_type_& row) noexcept {
        row.query_norms = norms_(shape.kernel, row.query);
    }

    /** Scores the staged flights of @p walk as one quad, rescores and folds them through its row,
     *  and merges them into its pool through @p walker; short tails replicate lane zero, and only
     *  real lanes reach the pool or the telemetry. */
    template <typename walker_type_>
    [[gnu::always_inline]] static void score_staged_(walker_type_& walker, walk_t& walk) noexcept {
        using row_t = typename walker_type_::row_t;
        using key_t = typename walker_type_::key_t;
        std::size_t const fill = walk.staged.size();
        constexpr bool keyed = row_t::key_reads_k == nvme_key_reads_t::scored_k;
        row_t& row = walker.rows_[walk.row_slot];
        nvme_part_t const& part = walker.parts_[walk.shard];
        std::array<node_id_t, 4> ids;
        std::array<vector_view_t, 4> vectors;
        std::array<vector_norms_t, 4> norms;
        std::array<key_t, 4> keys {};
        for (std::size_t lane = 0; lane != 4; ++lane) {
            u32_t const source = walk.staged[min_of<std::size_t>(lane, fill - 1)];
            ids[lane] = walker.flights_[source].node;
            vectors[lane] = vector_view_t {
                reinterpret_cast<scalar_t const*>(walker.landed_row_(source, nvme_purpose_t::score_k, source)),
                part.dimensions};
            norms[lane] = norms_(walker.shape_.kernel, vectors[lane]);
            if constexpr (keyed)
                keys[lane] = safegraphs_widened_key<key_t>(
                    walker.landed_row_(source, nvme_purpose_t::key_k, walker.key_buffer_(source)), part.key_dtype);
        }
        std::span<key_t const> const lane_keys = std::span<key_t const>(keys).first(keyed ? fill : 0);
        scores_result<dot_result_t, 4> scores = dense::scores<capability_, scalar_t>(walker.shape_.kernel, row.query,
                                                                                     row.query_norms, vectors, norms);
        row.rescore(ids, vectors, norms, scores, fill, lane_keys);
        walker.telemetry_.computed_distances += fill;
        std::array<scored_node_t, 4> scored_batch;
        for (std::size_t lane = 0; lane != fill; ++lane)
            scored_batch[lane] = scored_node_t {ids[lane], scores.distances[lane]};
        std::span<scored_node_t> const scored = std::span(scored_batch).first(fill);
        std::span<dot_result_t const> const dots = std::span<dot_result_t const>(scores.dots).first(fill);
        auto&& fold = row.fold(walk);
        walk_demand_t demand = walk_demand_t::continue_k;
        if constexpr (keyed) demand = fold.absorb(scored, dots, lane_keys);
        else demand = fold.absorb(scored, dots);
        for (u32_t const flight : walk.staged) walker.release_flight_(flight);
        walk.staged.clear();
        if (demand == walk_demand_t::sated_k) {
            walk.stage = nvme_walk_stage_t::draining_k;
            return;
        }
        network_sort(scored, closer_scored_node_t {});
        walker.rank_(walk, scored);
    }

    /** The batch @ref row_walker::run_walks walks, inside a call of its own: @p rows_count rows
     *  over every part in @p parts, each walk @p expansion_factor wide and ranking by @p kernel,
     *  and each row staging @p staged_bytes of imported queries. */
    template <typename row_type_>
    static expected<search_result_t> walk_rows_(std::span<medium_t const> media, std::span<nvme_part_t const> parts,
                                                nk_kernel_kind_t kernel, executor_t& executor, std::size_t rows_count,
                                                std::size_t expansion_factor,
                                                typename row_type_::inputs_t const& inputs, std::size_t staged_bytes,
                                                std::span<std::size_t const> counts,
                                                std::span<u64_t> latencies) noexcept {
        if (parts.empty()) return {{}, status_t::uninitialized_k};
        if (!rows_count) return {{}, status_t::success_k};
        walk_shape_t const shape = walk_shape_<row_type_>(media, parts, kernel, executor.queue_depth(), executor.size(),
                                                          rows_count, expansion_factor, staged_bytes);
        auto [call, call_status] = nvme_begin_call(executor, media, shape.scratch_bytes, shape.buffers,
                                                   shape.buffer_bytes);
        if (failed(call_status)) return {{}, call_status};
        return row_walker<nvme_index, row_type_>::run_walks(media, parts, executor, shape, inputs, counts, latencies);
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

        void prime(std::size_t new_row) noexcept {
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
        nvme_locator<u32_t> locator(std::size_t) const noexcept { return {}; }
        void resolve(std::size_t, key_t const&) noexcept {}
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
                                 kernel_, executor, queries_count, non_zero_expansion_factor(config), inputs, 0, counts,
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
