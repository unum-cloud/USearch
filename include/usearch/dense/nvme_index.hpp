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
 *  norms are recomputed from the arrived row, so the norms column is never read.
 *
 *  @section dense_nvme_tiering Hot Working Set & Cold Storage
 *
 *  Block storage is not the bottom of the hierarchy here; it is the hot working area. A shard lives
 *  long-term wherever it is cheapest to keep, such as a shared filesystem or an object bucket
 *  behind a FUSE mount, and is staged onto the local NVMe or High-Bandwidth-Flash the handle names
 *  when it is loaded, because a walk at device latency cannot run against a ranged @c GET that
 *  answers a thousand times slower. Nothing in this header decides when a shard moves or which
 *  shards stay warm; that policy belongs above it. What this header guarantees is the interface
 *  that policy needs: a shard is opened by path and addressed by a medium ID, so one that was
 *  staged from a bucket onto a drive, or evicted and staged again elsewhere, is re-attached without
 *  the state machine learning it moved.
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

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref`
#include <forkunion/types.hpp>   // `fu::standard_yield_t`
#include <numkong/tensor.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/nvme_device.hpp"       // `nvme_executor`, `file_medium_t`, `io_request_t`
#include "usearch/utilities/nvme_walks.hpp"        // `query_walker`, `nvme_begin_call`
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`, `sorted_buffer`, `arena_allocator`
#include "usearch/utilities/ram_sorts.hpp"         // `network_sort`
#include "usearch/utilities/ram_tables.hpp"        // `growing_hash_set`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_persisted_part`
#include "usearch/dense/types.hpp"
#include "usearch/dense/ram_metric.hpp"
#include "usearch/dense/ram_index.hpp" // `filtering_fold`, `multiply_fold`

namespace unum::usearch::dense {

/** Where one dense part keeps its fixed-width columns, derived once from its persisted columns:
 *  what a walk needs to address any row, and nothing per node. */
struct nvme_part_t {

    /** Rows the part holds; neighbor IDs at or past it are sentinels. */
    u32_t nodes = 0;

    /** Logical dimensions of one value row. */
    u32_t dimensions = 0;

    /** Neighbor IDs one neighbors row holds. */
    u32_t connectivity = 0;

    /** Bytes of one key row. */
    u32_t key_row_bytes = 0;

    /** The key column's dtype, which @ref safegraphs_widened_key reads each row as. */
    dtype_t key_dtype = dtype_t::unknown_k;

    /** Bytes of one value row. */
    u32_t value_row_bytes = 0;

    /** Where row zero of the key column starts, in bytes from the part's own start. */
    u64_t keys_begin = 0;

    /** Where row zero of the value column starts. */
    u64_t values_begin = 0;

    /** Where row zero of the neighbors column starts. */
    u64_t neighbors_begin = 0;
};

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
 *  the arrived rows are scored four at a time. Across the walks of a thread, everything overlaps.
 *  Rows fill an engine's queue first; whatever depth a short batch leaves idle lets each walk keep
 *  several hops in flight, expanding the next closest frontier nodes inside its candidates' radius
 *  before earlier hops arrive, so a full batch walks hop by hop and a lone query takes the queue.
 */
template <                                                                    //
    typename quads_kit_type_ = bf16_serial_quads_kit_t,                       //
    typename visits_kit_type_ = serial_visits_hash_set_kit_t,                 //
    typename engine_type_ = default_io_engine_t,                              //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename wait_type_ = fu::standard_yield_t                                //
    >
struct nvme_index {

    // The sharded index walks every part of a row on one thread, through the walk machinery below.
    template <typename, typename, typename, typename, typename, typename, typename, template <typename> class, typename>
    friend struct nvme_indexes;
    template <typename, typename, typename>
    friend class utilities::query_walker;
    template <typename>
    friend struct nvme_compact_plan;

    using scalar_t = typename quads_kit_type_::scalar_t;
    inline static constexpr nk_dtype_t vector_dtype_k = scalar_t::dtype();

    using engine_t = engine_type_;
    using executor_t = nvme_executor<engine_t>;
    using medium_t = typename engine_t::medium_t;
    using arena_t = typename executor_t::arena_t;

    /** The CPU class's reference over the words one call's threads share, and its waiter, as
     *  @c src/ picked them. */
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;
    using wait_t = wait_type_;

    using dot_result_t = typename scalar_t::dot_result_t;
    using distance_t = f32_t;

    /** Default integer type to use for loops and indexing at sub-index scale. */
    using candidates_count_t = u32_t;

    /** Node identifier within this part, the class-local spelling of @c dense::node_id_t. */
    using node_id_t = dense::node_id_t;

    /** What a walk hands an aggregating predicate beside each key it gathers. */
    using walk_payload_t = nvme_walk_payload<node_id_t, distance_t>;

    using vector_view_t = nk::vector_view<scalar_t>;
    using vectors_view_t = nk::matrix_view<scalar_t>;
    using vector_norms_t = vector_norms<dot_result_t>;

    /** The distance kit @c src/ picked. */
    using quads_kit_t = quads_kit_type_;

    /** The strongest @c synchronization_t contract this design honors: the payload is immutable, so
     *  arbitrary readers share one instance with no fences at all. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::immutable_k;

    /** Per-search tuning knobs; zeros mean "derive from connectivity". */
    struct search_config_t {

        /** How many candidates each walk keeps; defaults to twice the connectivity. */
        candidates_count_t expansion_factor = 0;

        /** How entries are reached; @c scan_k streams every part's value column, ignoring the
         *  breadth. */
        traversal_t traversal = traversal_t::walk_k;
    };

    /** Telemetry of one batched call; matches go to the caller's spans. */
    struct [[nodiscard]] search_result_t {

        /** Matches written across the batch, or what the verb counts instead. */
        std::size_t count = 0;

        /** Frontier candidates whose neighbors rows were read & expanded. */
        std::size_t visited_members = 0;

        /** Total distance computations across the batch, padding lanes excluded. */
        std::size_t computed_distances = 0;

        /** Re-seedings after a frontier drained before its candidates filled. */
        std::size_t restarts = 0;

        /** Walks a scratch refusal ended early, their rankings short. */
        std::size_t saturated_walks = 0;

        /** Hint keys no shard held a row under. */
        std::size_t missing_hints = 0;

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

    /**
     *  Opens the SafeGraphs monolith at @p path, a file, a device or, in SPDK builds, a transport
     *  ID with optional @c ns and @c base keys: its header, vetted against this instantiation and
     *  its kernel, and its medium for the engines. A failed view leaves the index empty.
     *
     *  @par Statuses:
     *  - @c unknown_kind_k A kernel the dtype's kits cannot finalize.
     *  - @c unsupported_k A transport ID in a build without SPDK.
     *  - @c invalid_argument_k @p path names no location, its base or namespace is unreadable, or
     *    the base is off the medium's block; in SPDK builds also an inactive namespace or an
     *    unparsable transport ID.
     *  - @c malformed_k No monolith starts there, its header does not parse or cannot be read, or
     *    its columns have no width or too many rows.
     *  - @c config_mismatch_k The engines read no such medium, the header names another metric,
     *    or its columns are not fixed rows of this scalar with @c u32 neighbors and one key per
     *    row.
     */
    status_t view(std::string_view path) noexcept {
        medium_ = medium_t {};
        part_ = {};
        if (!kernel_supported<quads_kit_t>(kernel_)) return status_t::unknown_kind_k;
        return open_part_(path, dtype_t::unknown_k, kernel_, forkunion_allocator_t {}, medium_, part_);
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
     *  @param[out] node_ids Rankings by node ID, one query per row, as many matches as its width.
     *  @param[out] distances Distances of the same rankings, ascending per row.
     *  @param[out] counts Matches written per query.
     *  @param[in] config The walk breadth; @c scan_k is refused with @c unsupported_k, as exact
     *      scans run over @c nvme_indexes, which a monolith views as one shard.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *
     *  A failed read stops the call: the engines drain, every buffer returns, and the call answers
     *  @c io_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c unsupported_k @p config asks for a scan.
     *  - @c config_mismatch_k @p counts, @p node_ids or @p distances disagree with the batch or
     *    each other, the queries are not as wide as the rows, or a medium the engines cannot read.
     *  - @c unsupported_layout_k @p node_ids or @p distances are not packed along their rows.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    expected<search_result_t> search(vectors_view_t queries, nk::matrix_span<node_id_t> node_ids,
                                     nk::matrix_span<distance_t> distances, std::span<std::size_t> counts,
                                     search_config_t config, executor_t& executor) noexcept {
        return search_(queries, dummy_heads_t {}, executor, config, node_ids, distances, counts);
    }

    /**
     *  @brief The batched @ref search under a lightning indexer, sharing its whole contract: every
     *      distance is 1 − its node's indexer score against the heads of its query row.
     *
     *  @param[in] queries The batch, as the batched @ref search takes it.
     *  @param[in] heads The indexer heads of every query row.
     *  @param[out] node_ids IDs of the matches, as the batched @ref search writes them.
     *  @param[out] distances The same matches' distances, parallel to @p node_ids.
     *  @param[out] counts Matches written per query.
     *  @param[in] config The walk configuration, as the batched @ref search takes it.
     *  @param[in,out] executor The threads, as the batched @ref search takes them.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The part does not rank by dot.
     */
    expected<search_result_t> search(vectors_view_t queries, indexer_heads<scalar_t> const& heads,
                                     nk::matrix_span<node_id_t> node_ids, nk::matrix_span<distance_t> distances,
                                     std::span<std::size_t> counts, search_config_t config,
                                     executor_t& executor) noexcept
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
    using visited_set_t = growing_hash_set<node_id_t, std::hash<node_id_t>, scratch_allocator_t, visits_kit_type_>;

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
        case nvme_purpose_t::key_k:
        case nvme_purpose_t::confirm_k: return part.keys_begin + u64_t(flight.node) * part.key_row_bytes;
        }
        return 0;
    }

    /** Bytes of any row of the column @p purpose reads in @p part. */
    static u64_t row_length_(nvme_part_t const& part, nvme_purpose_t purpose, nvme_flight<u32_t> const&) noexcept {
        switch (purpose) {
        case nvme_purpose_t::score_k: return part.value_row_bytes;
        case nvme_purpose_t::expand_k: return part.connectivity * sizeof(node_id_t);
        case nvme_purpose_t::key_k:
        case nvme_purpose_t::confirm_k: return part.key_row_bytes;
        }
        return 0;
    }

    /** Whether @p kernel reads rows' norms, as @ref stores_norms answers for this index's own. */
    static bool stores_norms_(nk_kernel_kind_t kernel) noexcept {
        return kernel != nk_kernel_dot_k || quads_compensated<quads_kit_t>;
    }

    /** The norms of @p row that @p kernel reads, or none, so a plain dot never sums a fetched
     *  row. */
    static vector_norms_t norms_(nk_kernel_kind_t kernel, vector_view_t row) noexcept {
        return stores_norms_(kernel) ? dense::norms<quads_kit_t>(row) : vector_norms_t {};
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

    /** Reads a part's header through @p source, any reader with <tt>read_at(offset, bytes)</tt>,
     *  into @p part, vetted as @c open_part_ promises. */
    template <typename source_type_, typename allocator_type_>
    static status_t read_part_(source_type_ const& source, dtype_t key_dtype, nk_kernel_kind_t& kernel,
                               allocator_type_ allocator, nvme_part_t& part) noexcept {
        growing_buffer<byte_t, allocator_type_> header_bytes {allocator};
        if (status_t const read = nvme_read_header(source, header_bytes); failed(read)) return read;
        auto const [persisted, persisted_status] = safegraphs_persisted_part(to_span(header_bytes), allocator);
        if (failed(persisted_status)) return persisted_status;
        safegraphs_name_t const& metric = persisted.metric;
        nk_kernel_kind_t const named = metric.empty() ? kernel
                                                      : nk_kernel_named(metric.characters, metric.view().size());
        if (kernel != nk_kernel_unknown_k && named != kernel) return status_t::config_mismatch_k;
        if (status_t const derived = part_of_columns_(persisted.columns, key_dtype, part); failed(derived))
            return derived;
        kernel = named;
        return status_t::success_k;
    }

    /** Derives @p part from the persisted @p columns of a dense part: values of this index's scalar
     *  in fixed rows, @c u32 neighbor IDs of a fixed degree, and one key per row, which must widen
     *  into @p key_dtype unless it is @c unknown_k. */
    static status_t part_of_columns_(persisted_columns_t const& columns, dtype_t key_dtype,
                                     nvme_part_t& part) noexcept {
        persisted_column_t const& keys = columns.keys;
        persisted_column_t const& values = columns.values;
        persisted_column_t const& neighbors = columns.neighbors;
        if (keys.offsets || values.offsets || neighbors.offsets || keys.row_width != 1 || !keys.row_bytes ||
            values.dtype != dtype_of(scalar_t::dtype()) || neighbors.dtype != dtype_t::u32_k)
            return status_t::config_mismatch_k;
        if (key_dtype != dtype_t::unknown_k && !safegraphs_key_widens(keys.dtype, key_dtype))
            return status_t::config_mismatch_k;
        if (!values.row_width || !neighbors.row_width || columns.nodes >= node_id_unknown_k)
            return status_t::malformed_k;
        part.nodes = static_cast<u32_t>(columns.nodes);
        part.dimensions = static_cast<u32_t>(values.row_width);
        part.connectivity = static_cast<u32_t>(neighbors.row_width);
        part.key_row_bytes = static_cast<u32_t>(keys.row_bytes);
        part.key_dtype = keys.dtype;
        part.value_row_bytes = static_cast<u32_t>(values.row_bytes);
        part.keys_begin = keys.data.first;
        part.values_begin = values.data.first;
        part.neighbors_begin = neighbors.data.first;
        return status_t::success_k;
    }

#pragma endregion Opening

#pragma region Walks

    /** One walk of one row over one shard: its candidates, its admitted set, its visited set, and
     *  the flights it waits on. */
    struct walk_t {

        /** Nodes scored so far, the best ascending. */
        results_array_t results;

        /** Nodes its row admitted so far, the best ascending; empty unless the row admits. */
        results_array_t admitted;

        /** Nodes worth expanding next. */
        frontier_queue_t frontier;

        /** Nodes already read for scoring; exact by design. */
        visited_set_t visited;

        /** Fetched score flights awaiting the next quad. */
        fu::limited_array<u32_t, quad_lanes_k> staged;

        /** Draws this walk's landmarks. */
        splitmix64_t random_generator;

        /** One landmark slot per seed it may take, carved with the walk: as many as it draws
         *  landmarks. */
        std::span<nvme_seed<node_id_t>> seed_slots;

        /** The seeds still to take ahead of its draws, front first: its hints' candidates while
         *  they confirm, then the confirmed ones. */
        std::span<nvme_seed<node_id_t>> seeds;

        /** The shard walked, positional among the call's shards. */
        u32_t shard = 0;

        /** The row slot this walk belongs to. */
        u32_t query_slot = 0;

        /** Where the walk stands. */
        nvme_walk_stage_t stage = nvme_walk_stage_t::finished_k;

        explicit walk_t(scratch_allocator_t allocator) noexcept
            : results(candidates_allocator_t(allocator)), admitted(candidates_allocator_t(allocator)),
              frontier(candidates_allocator_t(allocator)), visited(allocator) {}
    };

    /** The sizes and shared words every thread of one call reads. */
    struct walk_shape_t {

        /** The distance kind every walk ranks by. */
        nk_kernel_kind_t kernel = nk_kernel_unknown_k;

        /** Rows of the batch. */
        std::size_t queries_count = 0;

        /** Each walk's candidate breadth. */
        std::size_t expansion_factor = 0;

        /** The queue sizes every walk starts from, at that breadth over the widest part's rows. */
        nvme_queue_sizes_t queue_sizes {};

        /** Flights one walk ever holds at once: its landmarks or one expansion and every neighbor
         *  it scores, and one expansion and its neighbors more per further hop in flight. */
        std::size_t walk_flights = 0;

        /** Rows a thread keeps in flight. */
        std::size_t queries_per_thread = 0;

        /** Hops each walk keeps in flight: the queue depth its thread's rows leave idle, one for a
         *  full batch. */
        std::size_t hops_per_walk = 1;

        /** Hop slots each walk's span holds: its own @c hops_per_walk, and past the batch's end the
         *  shares of every other row slot of its thread, which then sit idle. */
        std::size_t hop_slots = 1;

        /** Buffers per flight: two when keys arrive beside values. */
        std::size_t buffers_per_flight = 1;

        /** Bytes every thread's arena carves for its walks, rows and flights. */
        std::size_t scratch_bytes = 0;

        /** Staging buffers every engine holds: @c buffers_per_flight per flight. */
        std::size_t buffers = 0;

        /** Bytes of one staging buffer: the covering read of the widest row any part holds. */
        std::size_t buffer_bytes = 0;

        /** The next row any thread takes. */
        std::size_t* cursor = nullptr;

        /** The first failure any thread hit, which stops every thread priming rows. */
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

    /** The shape of a walk of @p queries_count queries of @p query_type_ over every part in
     *  @p parts, each walk @p expansion_factor wide and ranking by @p kernel, on @p threads
     *  engines @p queue_depth reads deep, every row in flight staging @p staged_bytes of imported
     *  queries. */
    template <typename query_type_>
    static walk_shape_t walk_shape_(std::span<medium_t const> media, std::span<nvme_part_t const> parts,
                                    nk_kernel_kind_t kernel, std::size_t queue_depth, std::size_t threads,
                                    std::size_t queries_count, std::size_t expansion_factor,
                                    std::size_t staged_bytes) noexcept {
        using slice_t = arena_allocator<byte_t>;
        std::size_t const connectivity = nvme_widest_row(parts);
        walk_shape_t shape;
        shape.kernel = kernel;
        shape.queries_count = queries_count;
        shape.expansion_factor = expansion_factor;
        shape.queue_sizes = nvme_queue_sizes(expansion_factor, connectivity);
        shape.buffers_per_flight = query_type_::key_reads_k == nvme_key_reads_t::scored_k ? 2 : 1;
        std::size_t const first_hop = nvme_first_hop(connectivity);
        std::size_t const hop_depth = parts.size() * first_hop * shape.buffers_per_flight;
        // Rows fill the queue first, since a row costs no speculation; depth a short batch leaves
        // idle widens walks.
        shape.queries_per_thread = min_of(max_of<std::size_t>(1, queue_depth / hop_depth),
                                          max_of<std::size_t>(1, divide_round_up(queries_count, threads)));
        shape.hops_per_walk = max_of<std::size_t>(1, queue_depth / (shape.queries_per_thread * hop_depth));
        shape.walk_flights = first_hop + (shape.hops_per_walk - 1) * (connectivity + 1);
        // Past the batch's end a walk inherits the flights of the row slots that went idle.
        shape.hop_slots = shape.hops_per_walk +
                          (shape.queries_per_thread - 1) * shape.walk_flights / (connectivity + 1);
        std::size_t const queries = shape.queries_per_thread, walks = queries * parts.size();
        std::size_t const flights = walks * shape.walk_flights;
        shape.buffers = flights * shape.buffers_per_flight;
        // What `query_walker::reserve` carves, slice by slice, after padding to the first boundary.
        std::size_t const results_bytes = slice_t::reserved_bytes(
            results_array_t::memory_usage(shape.queue_sizes.results));
        std::size_t const per_walk = results_bytes + (query_type_::admitting_k ? results_bytes : 0) +
                                     slice_t::reserved_bytes(
                                         frontier_queue_t::memory_usage(shape.queue_sizes.frontier)) +
                                     slice_t::reserved_bytes(visited_set_t::memory_usage(shape.queue_sizes.visited));
        shape.scratch_bytes = walks * per_walk + slice_t::reserved_bytes(walks * sizeof(walk_t)) +
                              walks *
                                  slice_t::reserved_bytes(nvme_landmarks(connectivity) * sizeof(nvme_seed<node_id_t>)) +
                              slice_t::reserved_bytes(queries * sizeof(query_type_)) +
                              queries * slice_t::reserved_bytes(staged_bytes) +
                              slice_t::reserved_bytes(queries * sizeof(nvme_query_slot_t)) +
                              slice_t::reserved_bytes(flights * sizeof(nvme_flight<u32_t>)) +
                              slice_t::reserved_bytes(flights * sizeof(u32_t)) +
                              slice_t::reserved_bytes(walks * shape.hop_slots * sizeof(u32_t)) +
                              slice_t::reserved_bytes(shape.buffers * sizeof(io_request_t)) +
                              slice_t::reserved_bytes(shape.buffers * sizeof(io_completion_t)) + default_alignment_k;
        shape.buffer_bytes = staging_bytes_(media, parts);
        return shape;
    }

    /** Prepares the query of a primed @p row: the norms the call's kernel reads of it. */
    template <typename query_type_>
    static void prime_scoring_(walk_shape_t const& shape, query_type_& row) noexcept {
        row.query_norms = norms_(shape.kernel, row.query);
    }

    /** Scores the staged flights of @p walk as one quad, rescores them through its row, lets an
     *  admitting row test them, folds them, and merges them into its candidates through @p walker;
     *  short tails replicate lane zero, and only real lanes reach the candidates or telemetry. */
    template <typename walker_type_>
    [[gnu::always_inline]] static void score_staged_(walker_type_& walker, walk_t& walk) noexcept {
        using query_t = typename walker_type_::query_t;
        using key_t = typename walker_type_::key_t;
        std::size_t const fill = walk.staged.size();
        constexpr bool keyed = query_t::key_reads_k == nvme_key_reads_t::scored_k;
        query_t& row = walker.queries_[walk.query_slot];
        nvme_part_t const& part = walker.parts_[walk.shard];
        quad<node_id_t> ids;
        quad<vector_view_t> vectors;
        quad<vector_norms_t> norms;
        quad<key_t> keys {};
        for (std::size_t lane = 0; lane != 4; ++lane) {
            u32_t const source = walk.staged[min_of<std::size_t>(lane, fill - 1)];
            ids[lane] = walker.flights_[source].node;
            vectors[lane] = vector_view_t {
                reinterpret_cast<scalar_t const*>(walker.fetched_row_(source, nvme_purpose_t::score_k, source)),
                part.dimensions};
            norms[lane] = norms_(walker.shape_.kernel, vectors[lane]);
            if constexpr (keyed) keys[lane] = walker.fetched_key_(source);
        }
        std::span<key_t const> const lane_keys = std::span<key_t const>(keys).first(keyed ? fill : 0);
        scores_result<dot_result_t> scores = dense::scores<quads_kit_t>(walker.shape_.kernel, row.query,
                                                                        row.query_norms, vectors, norms);
        row.rescore_quad(ids, vectors, norms, scores, fill, lane_keys);
        walker.telemetry_.computed_distances += fill;
        quad<scored_node_t> scored_batch;
        for (std::size_t lane = 0; lane != fill; ++lane)
            scored_batch[lane] = scored_node_t {ids[lane], scores.distances[lane]};
        std::span<scored_node_t> const scored = std::span(scored_batch).first(fill);
        std::span<dot_result_t const> const dots = std::span<dot_result_t const>(scores.dots).first(fill);
        if constexpr (query_t::admitting_k) walker.admit_scored_(walk, scored);
        walk_demand_t const demand = row.fold_of_walk(walk).absorb(scored, dots);
        for (u32_t const flight : walk.staged) walker.release_flight_(flight);
        walk.staged.clear();
        if (demand == walk_demand_t::sated_k) {
            walk.stage = nvme_walk_stage_t::draining_k;
            return;
        }
        network_sort(scored, closer_scored_node_t {});
        walker.rank_(walk, scored);
    }

    /** A dense row relates nothing, so every contender goes on to its predicate. */
    template <typename walker_type_, std::size_t extent_>
    static void ask_relation_(walker_type_&, walk_t const&, std::span<u32_t const, extent_>,
                              std::span<admission_t, extent_> related) noexcept {
        for (admission_t& admission : related) admission = admission_t::admit_k;
    }

    /** The batch @ref query_walker::run_walks walks, inside a call of its own: @p queries_count
     *  queries over every part in @p parts, each walk @p expansion_factor wide and ranking by
     *  @p kernel, and each row staging @p staged_bytes of imported queries. */
    template <typename query_type_>
    static expected<search_result_t> walk_queries_(std::span<medium_t const> media, std::span<nvme_part_t const> parts,
                                                   nk_kernel_kind_t kernel, executor_t& executor,
                                                   std::size_t queries_count, std::size_t expansion_factor,
                                                   typename query_type_::inputs_t const& inputs,
                                                   std::size_t staged_bytes, std::span<std::size_t const> counts,
                                                   std::span<u64_t> latencies) noexcept {
        if (parts.empty()) return {{}, status_t::uninitialized_k};
        if (!queries_count) return {{}, status_t::success_k};
        std::size_t const queue_depth = nvme_call_depth(executor, media, parts);
        walk_shape_t const shape = walk_shape_<query_type_>(media, parts, kernel, queue_depth, executor.size(),
                                                            queries_count, expansion_factor, staged_bytes);
        auto [call, call_status] = nvme_begin_call(executor, media, queue_depth, shape.scratch_bytes, shape.buffers,
                                                   shape.buffer_bytes);
        if (failed(call_status)) return {{}, call_status};
        if constexpr (query_type_::admitting_k)
            if (status_t const opened = open_predicate(*inputs.predicate, executor); failed(opened))
                return {{}, opened};
        return query_walker<nvme_index, query_type_>::run_walks(media, parts, executor, shape, inputs, counts,
                                                                latencies);
    }

#pragma endregion Walks

#pragma region Rows

    /** The rows of a batched search over this one part, ranked by node ID: no key resolves. */
    template <typename heads_type_>
    struct node_search_query {
        using key_t = node_id_t;
        using fold_t = dummy_fold_t;
        inline static constexpr nvme_key_reads_t key_reads_k = nvme_key_reads_t::outputs_k;
        inline static constexpr bool admitting_k = false;

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

        node_search_query(inputs_t const& inputs, scratch_allocator_t) noexcept : inputs(inputs) {}
        status_t reserve(std::size_t, std::size_t) noexcept { return status_t::success_k; }

        void prime_query(std::size_t new_query_index) noexcept {
            row = new_query_index;
            query = inputs.queries.row(row).as_vector();
            inputs.counts[row] = 0;
        }
        void rescore_quad(quad<node_id_t> const& ids, quad<vector_view_t> const& vectors,
                          quad<vector_norms_t> const& norms, scores_result<dot_result_t>& scores, std::size_t count,
                          std::span<key_t const>) const noexcept {
            if constexpr (!std::is_same_v<heads_type_, dummy_heads_t>)
                indexer_rescoring<quads_kit_t> {inputs.heads->row(row)}(ids, vectors, norms, scores, count);
        }
        fold_t fold_of_walk(walk_t const&) const noexcept { return {}; }
        results_array_t const& stopping_candidates(walk_t const& walk) const noexcept { return walk.results; }
        void finish_walk(walk_t const& walk) noexcept {
            std::size_t const kept = min_of(walk.results.size(), inputs.node_ids.extent(1));
            for (std::size_t rank = 0; rank != kept; ++rank) {
                inputs.node_ids(row, rank) = walk.results[rank].node_id;
                inputs.distances(row, rank) = walk.results[rank].distance;
            }
            inputs.counts[row] = kept;
        }
        std::size_t output_count() const noexcept { return 0; }
        shard_node_t output_locator(std::size_t) const noexcept { return {}; }
        void resolve_output_key(std::size_t, key_t const&) noexcept {}
        status_t finish_query() noexcept { return status_t::success_k; }
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
        using query_t = node_search_query<heads_type_>;
        typename query_t::inputs_t const inputs {queries, &heads, node_ids, distances, counts};
        return walk_queries_<query_t>(std::span<medium_t const>(&medium_, 1), std::span<nvme_part_t const>(&part_, 1),
                                      kernel_, executor, queries_count, non_zero_expansion_factor(config), inputs, 0,
                                      counts, {});
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
