/**
 *  @file include/usearch/dense/nvme_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The sharded index over SafeGraphs parts on block storage: every read verb of
 *      @c ram_indexes, walking every shard of a row on one thread, and the freestanding
 *      merge, split, refine and build that rewrite its parts.
 *
 *  A view opens the manifest or the monolith a RAM or VRAM index saved, keeps every part's medium
 *  and column offsets, and nothing per node, so its memory never grows with the rows. Every verb
 *  borrows an executor and attaches the parts' media to its engines for the call alone, so any
 *  executor reaching the same devices serves any view. Walks rank rows by shard and node, and only
 *  the rows a verb writes resolve into keys, one key read each, once the row's walks finished;
 *  a filtered search and attention read every scored row's key beside its value as well. Key
 *  lookups are bulk scans of the key columns, matched against an exact table of the batch; the
 *  cursor scans the key and value columns, and the health audit the neighbors columns. A shard is
 *  the position a verb scatters over, and a part the SafeGraphs file behind it, one per shard.
 *
 *  @section dense_nvme_rewrites Rewrites
 *
 *  Merge, split, refine and build are free functions over the views, each planning its rewrite as
 *  phases of steps that borrow an NVMe executor one at a time and land SafeGraphs parts beside a
 *  manifest. A rewrite never edits its sources. It lays its target out as spec-standard parts,
 *  keys, values, neighbors and the producer's norms, whose headers land last and the manifest
 *  after them, so a stopped rewrite reads as absent. Its RAM is the workspace the caller lends,
 *  the executor's arenas and its staging buffers: every block, bucket and batch is sized from the
 *  workspace, and nothing grows with the rows. Reads go through the executor's engines, the
 *  target's included, and writes are blocking positional runs.
 *
 *  @section dense_nvme_rewrites_phases Rewrite Phases
 *
 *  - @b Route, while a split by keys plans, scans the key column of every source part to count the
 *    rows each key range takes.
 *  - @b Copy moves keys and values in target order, derives the norms, and rewires the neighbors a
 *    target keeps.
 *  - @b Build builds contiguous blocks of rows in a RAM index inside the workspace and writes their
 *    rows.
 *  - @b Re-select re-selects target rows by walks of the target itself, or in RAM when a whole part
 *    fits.
 *  - @b Back-link merges the back-links the re-selections spilled into bucket files on disk.
 *  - @b Heal sweeps reachability from the landmarks every audit seeds from, and links islands back
 *    in.
 */
#pragma once

#include <cstdio>  // `std::remove`, `std::snprintf`
#include <cstring> // `std::memcpy`, `std::memcmp`, `std::memset`

#include <array>  // `std::array`
#include <atomic> // `std::atomic_ref`
#include <bit>    // `std::popcount`, `std::countr_zero`
#include <chrono> // `std::chrono::steady_clock`, for a scan's latencies
#include <limits> // `std::numeric_limits`
#include <memory> // `std::uninitialized_value_construct_n`, `std::unique_ptr`, `std::destroy_at` - the RAM index a block builds in
#include <mutex>       // `std::unique_lock`
#include <new>         // placement `new`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::is_same_v`, `std::remove_reference_t`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `seekable_file_t`, `mapped_file_t`, `file_span_t`
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`, `arena_allocator`, `ceil2`
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator_t`
#include "usearch/utilities/ram_sorts.hpp"         // `partial_sort`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_manifest`, `safegraphs_write_manifest`
#include "usearch/dense/types.hpp"
#include "usearch/dense/ram_index.hpp" // `ram_index`, the RAM index a block build fills
#include "usearch/dense/nvme_index.hpp"

namespace unum::usearch::dense {

template <typename indexes_type_>
struct nvme_indexes_cursor;

/**
 *  @brief The sharded index over SafeGraphs parts on block storage, read-only: the read verbs of
 *      @c ram_indexes over the parts a RAM or VRAM index saved.
 *
 *  A row's walks, one per part, share one row state on one executor thread, so the verbs reuse the
 *  in-memory index's folds and rescorings unchanged and need no lock. No mutating verb exists:
 *  growth is a new part from a RAM or VRAM save, opened by a fresh @ref view. The metric is the
 *  one the parts' headers name, so one instantiation views parts of any kernel its kits finalize.
 */
template <                                                     //
    typename key_type_ = uuid_t,                               //
    typename quads_kit_type_ = bf16_serial_quads_kit_t,        //
    typename tiles_kit_type_ = no_tiles_kit_t,                 //
    typename elementwise_kit_type_ = serial_elementwise_kit_t, //
    typename rmscast_kit_type_ = no_rmscast_kit_t,             //
    typename visits_kit_type_ = serial_visits_hash_set_kit_t,  //
    typename engine_type_ = default_io_engine_t                //
    >
struct nvme_indexes {
    friend struct nvme_indexes_cursor<nvme_indexes>;
    template <typename>
    friend struct nvme_rewrite_plan;

    /** A reader's walk over every part's rows in placement order. */
    using cursor_t = nvme_indexes_cursor<nvme_indexes>;

    /** The caller-visible identity, read from every part's key column. */
    using key_t = key_type_;

    /** The one-part index, whose walks every verb drives over all parts at once. */
    using index_t = nvme_index<quads_kit_type_, visits_kit_type_, engine_type_>;

    /** The threads, their engines and their scratch every verb runs on; it names no view, so one
     *  executor serves every view reaching the same devices. */
    using executor_t = typename index_t::executor_t;
    using engine_t = typename index_t::engine_t;
    using medium_t = typename index_t::medium_t;
    using distance_t = typename index_t::distance_t;
    using scalar_t = typename index_t::scalar_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    using vectors_span_t = nk::matrix_span<scalar_t>;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using vector_norms_t = typename index_t::vector_norms_t;
    using search_config_t = typename index_t::search_config_t;
    using search_result_t = typename index_t::search_result_t;
    using health_result_t = dense::health_result_t;

    /** What a walk hands an aggregating predicate beside each key it gathers. */
    using walk_payload_t = typename index_t::walk_payload_t;

    /** What a scan hands an aggregating predicate beside each key: the row within its run. */
    using scan_payload_t = dense::scan_payload_t;

    /** The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = index_t::vector_dtype_k;

    /** The kits every part's walks, scans, imports and rewrites run on, as @c src/ picked them. */
    using quads_kit_t = quads_kit_type_;
    using tiles_kit_t = tiles_kit_type_;
    using elementwise_kit_t = elementwise_kit_type_;
    using rmscast_kit_t = rmscast_kit_type_;
    using visits_kit_t = visits_kit_type_;

    /** Whether a matrix engine multiplies the scalar, so many queries scan through it at once. */
    inline static constexpr bool has_tiles_k = !std::is_same_v<tiles_kit_t, no_tiles_kit_t>;

    /** Every index names the host allocator its catalog of parts sits on. */
    nvme_indexes() = delete;
    explicit nvme_indexes(forkunion_allocator_t allocator) noexcept
        : media_(allocator), parts_(allocator), allocator_(allocator) {}
    nvme_indexes(nvme_indexes&&) noexcept = default;
    nvme_indexes& operator=(nvme_indexes&&) noexcept = default;
    nvme_indexes(nvme_indexes const&) = delete;
    nvme_indexes& operator=(nvme_indexes const&) = delete;

    /**
     *  @brief Opens a manifest or a monolith, whichever its content names, and every part's header
     *      and medium, refusing parts that disagree with this instantiation or with each other. A
     *      failed view leaves the index empty.
     *
     *  The first part naming a metric fixes the kernel, a later part naming another is refused with
     *  @c config_mismatch_k, and a kernel the dtype's kits cannot finalize with @c unknown_kind_k.
     *
     *  @param[in] path A monolith, the manifest whose parts sit beside it or, in SPDK builds, a
     *      transport ID naming a monolith on a namespace, with the optional keys @c ns:<ID> and
     *      @c base:<bytes>.
     *  @param[in] paths Where to serve each part from instead, one per part: a directory holding
     *      the part's file, or a file, device or transport ID holding the part itself; empty serves
     *      every part where the manifest points.
     */
    status_t view(std::string_view path, std::span<std::string_view const> paths = {}) noexcept {
        media_.clear();
        parts_.clear();
        kernel_ = nk_kernel_unknown_k;
        status_t const viewed = view_(path, paths);
        if (failed(viewed)) media_.clear(), parts_.clear(), kernel_ = nk_kernel_unknown_k;
        return viewed;
    }

    /** The distance kind the parts were built for, as @ref view read it; unknown before one. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Whether the kernel reads rows' norms, which a rewrite's targets then carry as a column. */
    bool stores_norms() const noexcept { return index_t::stores_norms_(kernel_); }

    /** The checkpoint's metadata without viewing it, @ref nvme_read_metadata. */
    static expected<safegraphs_metadata_t> metadata(std::string_view path) noexcept { return nvme_read_metadata(path); }

    /** Shards in the view, one per part of the checkpoint, so a monolith views as one shard. */
    std::size_t shards_count() const noexcept { return parts_.size(); }

    /** Entries across every shard. */
    std::size_t size() const noexcept {
        std::size_t total = 0;
        for (nvme_part_t const& part : parts_) total += part.nodes;
        return total;
    }

    /** Entries the view hosts, which are exactly those its parts hold, since a view never grows. */
    std::size_t capacity() const noexcept { return size(); }

    /** The frozen geometry, which @ref view checked every part agrees on. */
    std::size_t dimensions() const noexcept { return parts_.empty() ? 0 : parts_[0].dimensions; }

    /** Connections per node, frozen with the geometry. */
    std::size_t connectivity() const noexcept { return parts_.empty() ? 0 : parts_[0].connectivity; }

    /** Bytes the index occupies: the catalog of parts alone, whatever the rows. */
    std::size_t memory_usage() const noexcept {
        return sizeof(nvme_indexes) + media_.memory_usage() + parts_.memory_usage();
    }

    /** The walk breadth @p config asks for, or twice the connectivity when it asks for none. */
    std::size_t non_zero_expansion_factor(search_config_t const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor : default_expansion_factor(connectivity());
    }

#pragma region Search

    /**
     *  @brief Scatters a whole matrix of queries over every shard, the walks of many queries in
     *      flight on every thread of @p executor, and gathers each query's rankings into one by
     *      key.
     *
     *  @param[in] queries The batch, one query per row, in any scalar NumKong casts into the stored
     *      one; every row slot imports its own.
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[out] keys Merged rankings, one query per row, as many matches as its width.
     *  @param[out] distances The same rankings' distances as @p score_type_, parallel to @p keys.
     *  @param[out] counts Per-query merged hit counts, one per query row.
     *  @param[out] latencies Steady-clock nanoseconds per query row, from its first primed walk to
     *      its final write; empty collects none.
     *
     *  A failed read stops the call: the engines drain, every buffer returns, and the call answers
     *  @c io_k.
     */
    template <typename score_type_>
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                     std::span<std::size_t> counts, std::span<u64_t> latencies = {}) noexcept {
        return search_if(queries, executor, config, keys, distances, counts, dummy_predicate_t {}, latencies);
    }

    /**
     *  @brief The batched @ref search under a lightning indexer, sharing its whole contract: every
     *      distance is 1 − its key's indexer score against the heads of its query row.
     *
     *  @param[in] heads The indexer heads of every query row.
     *  @return @c config_mismatch_k unless the parts rank by dot.
     */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads, executor_t& executor,
                                     search_config_t config, nk::matrix_span<key_t> keys,
                                     nk::matrix_span<distance_t> distances, std::span<std::size_t> counts,
                                     std::span<u64_t> latencies = {}) noexcept
        requires floating_scalar<scalar_t>
    {
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        dummy_predicate_t predicate;
        return search_rows_(queries, heads, executor, config, keys, distances, counts, predicate, latencies);
    }

    /**
     *  @brief The filtered sibling of the batched @ref search, sharing its whole contract; a
     *      @c traversal_t::scan_k scan ranks every row exactly instead, as @ref scan_rows_ reads
     *      them.
     *
     *  @param[in] predicate The per-key admission filter; every scored row's key is read beside its
     *      value, so the predicate sees keys, never node IDs. A scan asks it only about rows closer
     *      than the worst match some query already holds.
     */
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> search_if(rows_view_t queries, executor_t& executor, search_config_t config,
                                        nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                        std::span<std::size_t> counts, predicate_type_&& predicate,
                                        std::span<u64_t> latencies = {}) noexcept {
        static_assert((!aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, walk_payload_t> &&
                       !aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, scan_payload_t>) ||
                          per_thread_predicates<std::remove_cvref_t<predicate_type_>>,
                      "An engine driving its own threads takes an aggregating predicate per thread");
        if (config.traversal == traversal_t::scan_k)
            return scan_rows_(queries, executor, keys, distances, counts, predicate, latencies);
        return search_rows_(queries, dummy_heads_t {}, executor, config, keys, distances, counts, predicate, latencies);
    }

    /** The filtered sibling for one aggregating predicate per thread, each gathering a walk's
     *  @c walk_payload_t, so it serves walks alone and a scan answers @c config_mismatch_k. */
    template <typename score_type_, typename predicate_type_>
        requires aggregating_predicate<typename std::remove_cvref_t<predicate_type_>::thread_predicate_t, key_t,
                                       walk_payload_t>
    expected<search_result_t> search_if(rows_view_t queries, executor_t& executor, search_config_t config,
                                        nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                        std::span<std::size_t> counts, predicate_type_&& predicate,
                                        std::span<u64_t> latencies = {}) noexcept {
        if (config.traversal == traversal_t::scan_k) return {{}, status_t::config_mismatch_k};
        return search_rows_(queries, dummy_heads_t {}, executor, config, keys, distances, counts, predicate, latencies);
    }

    /** The filtered sibling for one aggregating predicate per thread, each gathering a scan's
     *  @c scan_payload_t, so it serves scans alone and a walk answers @c config_mismatch_k. */
    template <typename score_type_, typename predicate_type_>
        requires aggregating_predicate<typename std::remove_cvref_t<predicate_type_>::thread_predicate_t, key_t,
                                       scan_payload_t>
    expected<search_result_t> search_if(rows_view_t queries, executor_t& executor, search_config_t config,
                                        nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                        std::span<std::size_t> counts, predicate_type_&& predicate,
                                        std::span<u64_t> latencies = {}) noexcept {
        if (config.traversal != traversal_t::scan_k) return {{}, status_t::config_mismatch_k};
        return scan_rows_(queries, executor, keys, distances, counts, predicate, latencies);
    }

    /**
     *  @brief Graph products of a whole matrix of left-hand rows against every shard: each row
     *      keeps the pairs its walks scored, keys resolved and dots cast into the products' scalar.
     *
     *  @param[in] left The left-hand rows, one walk per row and shard, in any scalar NumKong casts
     *      into the stored one; every row slot imports its own.
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[out] keys The key of each kept pair, one left-hand row per row, as many pairs as its
     *      width.
     *  @param[out] products The dot of each kept pair as @p product_type_, parallel to @p keys.
     *  @param[out] counts Pairs scored per row across every shard, above the row width when
     *      truncated.
     *  @param[out] latencies Steady-clock nanoseconds per left-hand row, as @ref search collects
     *      them.
     *  @return @c config_mismatch_k unless the parts rank by dot.
     */
    template <typename product_type_>
    expected<search_result_t> multiply(rows_view_t left, executor_t& executor, search_config_t config,
                                       nk::matrix_span<key_t> keys, nk::matrix_span<product_type_> products,
                                       std::span<std::size_t> counts, std::span<u64_t> latencies = {}) noexcept {
        return multiply_if(left, executor, config, keys, products, counts, dummy_predicate_t {}, latencies);
    }

    /**
     *  @brief The filtered sibling of @ref multiply, sharing its whole contract: only pairs whose
     *      key @p predicate admits are written and counted, while the walks route through the rest.
     *
     *  @param[in] predicate The per-key admission filter, asked about a scored quad of keys at a
     *      time; every scored row's key is read beside its value, and written pairs keep it.
     *  @return @c unsupported_k for a @c traversal_t::scan_k scan, since no exact product scan
     *      exists.
     */
    template <typename product_type_, typename predicate_type_>
    expected<search_result_t> multiply_if(rows_view_t left, executor_t& executor, search_config_t config,
                                          nk::matrix_span<key_t> keys, nk::matrix_span<product_type_> products,
                                          std::span<std::size_t> counts, predicate_type_&& predicate,
                                          std::span<u64_t> latencies = {}) noexcept {
        std::size_t const rows_count = left.rows;
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        if (counts.size() != rows_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(rows_count, keys, products); failed(refusal))
            return {{}, refusal};
        if (rows_count && left.columns != dimensions()) return {{}, status_t::config_mismatch_k};
        using row_t = multiply_row<product_type_, std::remove_reference_t<predicate_type_>>;
        typename row_t::inputs_t const inputs {left, &predicate, keys, products, counts};
        return walk_rows_<row_t>(rows_count, config, inputs, staged_rows_bytes(vector_dtype_k, 1, left.columns), counts,
                                 latencies, executor);
    }

    /** NumKong's aggregation over the stored scalar's dots, what a MaxSim total sums in. */
    using total_t = typename scalar_t::reduce_moments_sum_t;

    /**
     *  @brief Late-interaction scores of query groups against every shard: each query row takes its
     *      best dot within a key, and those maxima sum to that key's score, exact for every key the
     *      call nominates.
     *
     *  @param[in] queries The token rows of every query, laid end to end.
     *  @param[in] query_offsets Where every query's rows start in @p queries, then where the last
     *      ends: one more than @p counts.
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] config The walk configuration; each row nominates as many keys as its expansion
     *      factor, and a @c traversal_t::scan_k scan nominates every key.
     *  @param[in] wanted Keys each query keeps, the best ones; zero keeps every key it scored.
     *  @param[in] land Takes each kept key of a query in rank order, largest total first, on the
     *      thread that ranked it, as `land(thread, query, key, total) noexcept → status_t`; the
     *      first failure stops that query and is what the call answers.
     *  @param[out] latencies Steady-clock nanoseconds per query row of a walk, as @ref search
     *      collects them; normalization follows the last row.
     *  @return @c config_mismatch_k unless the parts rank by dot.
     */
    template <typename land_type_>
    expected<search_result_t> maxsim(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                     executor_t& executor, search_config_t config, std::size_t wanted,
                                     land_type_&& land, std::span<u64_t> latencies = {}) noexcept {
        return maxsim_if(queries, query_offsets, executor, config, wanted, land, dummy_predicate_t {}, latencies);
    }

    /**
     *  @brief The filtered sibling of @ref maxsim, sharing its whole contract: a key @p predicate
     *      rejects is never nominated, so it adds to no score.
     *
     *  @param[in] predicate The per-key admission filter: a walk asks it as the filtered
     *      @ref search_if asks it, a scan once per distinct key.
     */
    template <typename land_type_, typename predicate_type_>
    expected<search_result_t> maxsim_if(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                        executor_t& executor, search_config_t config, std::size_t wanted,
                                        land_type_&& land, predicate_type_&& predicate,
                                        std::span<u64_t> latencies = {}) noexcept {
        static_assert(!aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, walk_payload_t> ||
                          per_thread_predicates<std::remove_cvref_t<predicate_type_>>,
                      "An engine driving its own threads takes an aggregating predicate per thread");
        using slice_t = arena_allocator<byte_t>;
        using row_t = search_row<dummy_heads_t, std::remove_reference_t<predicate_type_>, distance_t>;
        if (query_offsets.empty()) return {{}, status_t::config_mismatch_k};
        std::size_t const queries_count = query_offsets.size() - 1, threads = executor.size();
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        for (std::size_t query = 0; query != queries_count; ++query)
            if (query_offsets[query + 1] <= query_offsets[query]) return {{}, status_t::shape_mismatch_k};
        std::size_t const rows_count = query_offsets.back() - query_offsets.front();
        if (query_offsets.back() > queries.rows || (rows_count && queries.columns != dimensions()) ||
            (!latencies.empty() && latencies.size() != rows_count))
            return {{}, status_t::config_mismatch_k};
        if (!queries_count) return {{}, status_t::success_k};

        // One call serves every pass: a walk's engines and arenas, then two scans.
        bool const scanning = config.traversal == traversal_t::scan_k;
        walk_shape_t const shape = scanning ? walk_shape_t {}
                                            : index_t::template walk_shape_<row_t>(
                                                  to_span(media_), to_span(parts_), kernel_, executor.queue_depth(),
                                                  threads, rows_count, non_zero_expansion_factor(config),
                                                  staged_rows_bytes(vector_dtype_k, 1, queries.columns));
        std::size_t const staged_bytes = staged_rows_bytes(vector_dtype_k, rows_count, queries.columns);
        std::size_t const shared_bytes = staged_bytes + slice_t::reserved_bytes(rows_count * sizeof(vector_norms_t)) +
                                         slice_t::reserved_bytes(threads * wanted * sizeof(maxsim_ranked<total_t>));
        auto [call, call_status] = scanning
                                       ? begin_scans_<2>(executor, 2, shared_bytes)
                                       : begin_scans_(executor, 2, shared_bytes + shape.scratch_bytes, shape.buffers);
        if (failed(call_status)) return {{}, call_status};
        if (status_t const opened = open_predicate(predicate, executor); failed(opened)) return {{}, opened};
        scratch_allocator_t const scratch {executor.arena(0)};
        growing_buffer<byte_t, scratch_allocator_t> staged {scratch};
        growing_buffer<vector_norms_t, scratch_allocator_t> query_norms {scratch};
        if (failed(staged.resize_uninitialized(staged_bytes)) || failed(query_norms.resize_uninitialized(rows_count)))
            return {{}, status_t::bad_alloc_k};
        // A product walks by dot, which never narrows at unit RMS.
        vectors_view_t const query_rows = import_staged_rows<elementwise_kit_t, rmscast_kit_t, scalar_t>(
            queries, query_offsets.front(), rows_count, nk_kernel_dot_k, staged.data());
        for (std::size_t row = 0; row != rows_count; ++row)
            query_norms[row] = index_t::norms_(kernel_, query_rows.row(row).as_vector());
        if (scanning)
            return scan_maxsim_(query_offsets, query_rows, to_span(query_norms), executor, wanted, land, predicate);
        return walk_maxsim_(queries, query_offsets, query_rows, to_span(query_norms), shape, executor, wanted, land,
                            predicate, latencies);
    }

    /**
     *  @brief Softmax attention of query heads over every key their walks score across the shards:
     *      a token's walks fold into one running state per head, since the shards are disjoint and
     *      online softmax is order-free, and its rows normalize once after the last walk finishes.
     *
     *  @param[in] queries Row t · @p head_count + h is head h of token t; one walk per token and
     *      shard. Any scalar NumKong casts into the stored one; every row slot imports its token's
     *      heads.
     *  @param[in] head_count Query heads sharing this key-value head; it must divide the rows.
     *  @param[in] value_lookup Resolves a scored batch's keys to their value rows, as wide as the
     *      keys and stored like them: `void(std::span<key_t const>, std::span<vector_view_t>)`, one
     *      row per key. Keys are read beside the scored rows; one without a value row refuses with
     *      @c shape_mismatch_k.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys. Packed @c f32
     *      rows accumulate in place, any others in the row slot's staging until the token exports
     *      them.
     *  @param[out] carry Every query row's softmax state, which merges this call with other
     *      branches, or empty to keep none.
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[out] latencies Steady-clock nanoseconds per token, its heads walked together, as
     *      @ref search collects them.
     *  @return The query rows attended in @c count, or @c config_mismatch_k unless the parts rank
     *      by dot.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     executor_t& executor, search_config_t config = {},
                                     std::span<u64_t> latencies = {}) noexcept {
        return attend_if(queries, head_count, value_lookup, score_scale, outputs, carry, executor, config,
                         dummy_predicate_t {}, latencies);
    }

    /**
     *  @brief The filtered sibling of @ref attend, sharing its whole contract: a key @p predicate
     *      rejects still routes the walks, but folds into no head and resolves no value row.
     *
     *  @param[in] predicate The per-key admission filter, asked about a scored quad of the keys
     *      read beside the values at a time.
     *  @return @c unsupported_k for a @c traversal_t::scan_k scan, since no exact attention scan
     *      exists.
     */
    template <typename value_lookup_type_, typename predicate_type_>
    expected<search_result_t> attend_if(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                        f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                        executor_t& executor, search_config_t config, predicate_type_&& predicate,
                                        std::span<u64_t> latencies = {}) noexcept {
        std::size_t const rows_count = queries.rows, columns = queries.columns;
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
        if (outputs.rows != rows_count || outputs.columns != columns || (!carry.empty() && carry.size() != rows_count))
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};
        if (columns != dimensions()) return {{}, status_t::config_mismatch_k};
        using row_t = attend_row<value_lookup_type_, std::remove_reference_t<predicate_type_>>;
        typename row_t::inputs_t const inputs {queries,     head_count, &value_lookup, &predicate,
                                               score_scale, outputs,    carry};
        expected<search_result_t> attended = walk_rows_<row_t>(rows_count / head_count, config, inputs,
                                                               row_t::staged_bytes(inputs), {}, latencies, executor);
        if (attended) attended.value.count = rows_count;
        return attended;
    }

#pragma endregion Search

#pragma region Key Scans

    /**
     *  @brief The entries stored under every key of @p keys: one scan over every part's key column,
     *      matched against an exact table of the batch.
     *
     *  @param[in] keys The keys to count, repeats and absent ones included.
     *  @param[out] entries The entries under each key, parallel to @p keys; zero means absent.
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     */
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        if (parts_.empty()) return status_t::uninitialized_k;
        std::memset(entries.data(), 0, entries.size_bytes());
        if (keys.empty()) return status_t::success_k;
        auto [call, call_status] = begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        return scan_keys_(keys, executor, [&](std::size_t, std::size_t position, nvme_locator<u32_t>) noexcept {
            std::atomic_ref<std::size_t>(entries[position]).fetch_add(1, std::memory_order_relaxed);
        });
    }

    /** The batched @ref count reduced to one in @p entries where a key of @p keys holds any entry,
     *  zero where it holds none. */
    status_t contains(std::span<key_t const> keys, std::span<std::size_t> entries,
                      executor_t& executor) const noexcept {
        if (status_t const counted = count(keys, entries, executor); failed(counted)) return counted;
        for (std::size_t& entry : entries) entry = entry ? 1 : 0;
        return status_t::success_k;
    }

    /**
     *  @brief Up to @p wanted rows of every key of @p keys, and every key's true total: one scan of
     *      the key columns, then one value-row read per kept row.
     *
     *  @param[in] keys The keys to export, repeats and absent ones included.
     *  @param[out] vectors @b [keys·wanted,dimensions] in any scalar NumKong casts into, where row
     *      q · @p wanted + i is the i-th row of key q in shard and row order, a key holding more
     *      keeping its earliest; refused with @c config_mismatch_k when the rows are another count
     *      or narrower than the index, while wider ones keep their tails.
     *  @param[in] wanted Rows kept per key.
     *  @param[out] entries The true total under each key, parallel to @p keys; above @p wanted when
     *      its rows truncated, zero when absent.
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     */
    status_t get(std::span<key_t const> keys, rows_span_t vectors, std::size_t wanted, std::span<std::size_t> entries,
                 executor_t& executor) const noexcept {
        std::size_t const rows_count = keys.size() * wanted;
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        if (vectors.rows != rows_count || (rows_count && vectors.columns < dimensions()))
            return status_t::config_mismatch_k;
        if (parts_.empty()) return status_t::uninitialized_k;
        std::memset(entries.data(), 0, entries.size_bytes());
        if (keys.empty()) return status_t::success_k;
        auto [call, call_status] = begin_scans_(executor, 2);
        if (failed(call_status)) return call_status;

        // Threads meet a key's rows in any order, so each key keeps its earliest under a lock
        // striped by position, one stripe per thread.
        growing_buffer<nvme_locator<u32_t>, scratch_allocator_t> kept {scratch_allocator_t {executor.arena(0)}};
        if (failed(kept.resize_uninitialized(keys.size() * wanted))) return status_t::bad_alloc_k;
        std::size_t const stripes_count = executor.size();
        auto* const stripes = reinterpret_cast<fu::spin_mutex_t*>(
            executor.arena(0).allocate(stripes_count * sizeof(fu::spin_mutex_t), alignof(fu::spin_mutex_t)));
        if (!stripes) return status_t::bad_alloc_k;
        // nvcc mislowers a bare over-aligned placement `new`, so a standard template builds them.
        std::uninitialized_value_construct_n(stripes, stripes_count);
        auto earlier = [](nvme_locator<u32_t> first, nvme_locator<u32_t> second) noexcept {
            return first.shard != second.shard ? first.shard < second.shard : first.node < second.node;
        };
        status_t const scanned = scan_keys_(
            keys, executor, [&](std::size_t, std::size_t position, nvme_locator<u32_t> locator) noexcept {
                std::unique_lock<fu::spin_mutex_t> const held {stripes[position % stripes_count]};
                nvme_locator<u32_t>* const rows = kept.data() + position * wanted;
                std::size_t slot = min_of(entries[position]++, wanted);
                if (slot == wanted && (!wanted || !earlier(locator, rows[wanted - 1]))) return;
                if (slot == wanted) --slot;
                for (; slot && earlier(locator, rows[slot - 1]); --slot) rows[slot] = rows[slot - 1];
                rows[slot] = locator;
            });
        if (failed(scanned)) return scanned;
        auto describe = [&](std::size_t read, u64_t& progress) noexcept -> std::array<extent_t, 1> {
            std::size_t const position = read / wanted, rank = read % wanted;
            if (progress++ || rank >= min_of(entries[position], wanted)) return {};
            nvme_locator<u32_t> const locator = kept[read];
            nvme_part_t const& part = parts_[locator.shard];
            return {extent_t {locator.shard, part.values_begin + u64_t(locator.node) * part.value_row_bytes,
                              part.value_row_bytes}};
        };
        rows_span_t const target {vectors.data, rows_count, dimensions(), vectors.stride, vectors.dtype};
        auto consume = [&](std::size_t, std::size_t read, auto const&, auto const& landed) noexcept {
            export_row<elementwise_kit_t>(landed[0], vector_dtype_k, target, read);
        };
        return scan_reads_(executor, rows_count, describe, consume).status;
    }

#pragma endregion Key Scans

#pragma region Health

    /**
     *  @brief Reachability of every part from as many stride-spread landmarks as one search draws,
     *      the RAM audit's seeds, each part swept level by level over its neighbors column, one
     *      part after another.
     *
     *  A part of N rows holds two bitsets of N bits, current and next, from the first thread's
     *  arena: 2·N bits in flight, 1 GiB for a part of 2³² rows, the widest part's pair reused by
     *  every part. A level reads its rows' neighbors in ID order, rows whose blocks touch merged
     *  into one read of at most a scan run, so a dense level reads near-sequentially and every
     *  reachable row is read once, by the level expanding it: at most N reads per part, and at most
     *  the part's neighbors column in bytes per level.
     *
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     */
    expected<health_result_t> health(executor_t& executor) const noexcept {
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        std::size_t widest = 0;
        for (nvme_part_t const& part : parts_) widest = max_of<std::size_t>(widest, part.nodes);
        auto [call, call_status] = begin_scans_(executor, 1);
        if (failed(call_status)) return {{}, call_status};
        std::size_t const words = max_of<std::size_t>(divide_round_up<std::size_t>(widest, 64), 1);
        arena_t& arena = executor.arena(0);
        auto* const current = reinterpret_cast<u64_t*>(arena.allocate(words * sizeof(u64_t), default_alignment_k));
        auto* const next = reinterpret_cast<u64_t*>(arena.allocate(words * sizeof(u64_t), default_alignment_k));
        if (!current || !next) return {{}, status_t::bad_alloc_k};
        health_result_t result;
        for (u32_t part = 0; part != parts_.size(); ++part) {
            auto const [audited, status] = audit_part_(part, {current, words}, {next, words}, executor);
            if (failed(status)) return {{}, status};
            result.audited_nodes += audited.audited_nodes;
            result.reachable_nodes += audited.reachable_nodes;
            result.sweeps = max_of(result.sweeps, audited.sweeps);
        }
        return {result, status_t::success_k};
    }

#pragma endregion Health

  private:
    using arena_t = typename index_t::arena_t;
    using call_t = nvme_call<executor_t>;
    using walk_shape_t = typename index_t::walk_shape_t;
    using scratch_allocator_t = typename index_t::scratch_allocator_t;
    using candidates_allocator_t = typename index_t::candidates_allocator_t;
    using results_array_t = typename index_t::results_array_t;
    using walk_t = typename index_t::walk_t;
    using node_id_t = dense::node_id_t;

#pragma region Rows

    /** The rows of a batched search across the shards: each walk's candidates merge into the row's
     *  ranking by locator, and the written locators resolve into keys once every walk finished. A
     *  filtered row reads every scored row's key beside its value for the predicate, and merges
     *  each walk's admitted set instead. Each slot imports its query into its own staging, and
     *  distances land as @p score_type_. */
    template <typename heads_type_, typename predicate_type_, typename score_type_>
    struct search_row {
        using key_t = key_type_;
        using fold_t = dummy_fold_t;

        /** A filtered row admits as it scores, so every scored row's key lands beside its value. */
        inline static constexpr bool admitting_k =
            !std::is_same_v<std::remove_cv_t<predicate_type_>, dummy_predicate_t>;
        inline static constexpr nvme_key_reads_t key_reads_k = admitting_k ? nvme_key_reads_t::scored_k
                                                                           : nvme_key_reads_t::outputs_k;

        /** The batch every row of the call reads and writes. */
        struct inputs_t {
            rows_view_t queries;
            nk_kernel_kind_t kernel = nk_kernel_unknown_k;
            heads_type_ const* heads = nullptr;
            predicate_type_* predicate = nullptr;
            std::size_t expansion_factor = 0;
            nk::matrix_span<key_t> keys;
            nk::matrix_span<score_type_> distances;
            std::span<std::size_t> counts;
        };

        inputs_t const& inputs;
        growing_buffer<nvme_locator<u32_t>, scratch_allocator_t> locators;
        scratch_allocator_t allocator;
        byte_t* staged = nullptr;
        std::size_t row = 0;
        std::size_t found = 0;
        vector_view_t query;
        vector_norms_t query_norms {};

        search_row(inputs_t const& inputs, scratch_allocator_t allocator) noexcept
            : inputs(inputs), locators(allocator), allocator(allocator) {}

        status_t reserve(std::size_t, std::size_t) noexcept {
            staged = allocator.allocate(staged_rows_bytes(vector_dtype_k, 1, inputs.queries.columns));
            if (!staged || failed(locators.resize_uninitialized(inputs.keys.extent(1)))) return status_t::bad_alloc_k;
            return status_t::success_k;
        }

        void prime_row(std::size_t new_row) noexcept {
            row = new_row;
            found = 0;
            query = import_staged_rows<elementwise_kit_t, rmscast_kit_t, scalar_t>(inputs.queries, row, 1,
                                                                                   inputs.kernel, staged)
                        .row(0)
                        .as_vector();
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

        /** A filtered walk stops on what its row admitted, so the admitted set fills before it
         *  stops. */
        results_array_t const& stopping_candidates(walk_t const& walk) const noexcept
            requires admitting_k
        {
            return walk.admitted;
        }
        void finish_walk(walk_t const& walk) noexcept {
            found = merge_search_results(to_span(stopping_candidates(walk)), nvme_shard_locators<u32_t> {walk.shard},
                                         to_writable_span(locators), to_span(inputs.distances.row(row).as_vector()),
                                         found);
        }
        std::size_t output_count() const noexcept { return found; }
        nvme_locator<u32_t> output_locator(std::size_t output) const noexcept { return locators[output]; }
        void resolve_output_key(std::size_t output, key_t const& key) noexcept { inputs.keys(row, output) = key; }
        status_t finish_row() noexcept {
            inputs.counts[row] = found;
            return status_t::success_k;
        }
    };

    /** The keys landed beside the values of one scored quad, which a filtered product writes for
     *  its nodes. */
    struct scored_keys_t {
        quad<node_id_t> nodes;
        quad<key_t> keys;

        key_t operator()(node_id_t node) const noexcept {
            std::size_t lane = 0;
            while (nodes[lane] != node) ++lane;
            return keys[lane];
        }
    };

    /** The rows of graph products: one fold per row spans its walks, each scored pair appended by
     *  locator until the row fills, its dot as @p product_type_, and the written locators resolve
     *  into keys. A filtered row reads every scored row's key beside its value instead, and its
     *  fold appends only the pairs @p predicate_type_ admits, by key. Each slot imports its
     *  left-hand row into its own staging. */
    template <typename product_type_, typename predicate_type_>
    struct multiply_row {
        using key_t = key_type_;
        inline static constexpr bool filtered_k = !std::is_same_v<std::remove_cv_t<predicate_type_>, dummy_predicate_t>;
        using fold_t = std::conditional_t<
            filtered_k, multiply_fold<key_t, product_type_, scored_keys_t, predicate_type_&>,
            multiply_fold<nvme_locator<u32_t>, product_type_, nvme_shard_locators<u32_t>, predicate_type_&>>;
        inline static constexpr nvme_key_reads_t key_reads_k = filtered_k ? nvme_key_reads_t::scored_k
                                                                          : nvme_key_reads_t::outputs_k;
        inline static constexpr bool admitting_k = false;

        /** The batch every row of the call reads and writes. */
        struct inputs_t {
            rows_view_t left;
            predicate_type_* predicate = nullptr;
            nk::matrix_span<key_t> keys;
            nk::matrix_span<product_type_> products;
            std::span<std::size_t> counts;
        };

        inputs_t const& inputs;
        growing_buffer<nvme_locator<u32_t>, scratch_allocator_t> locators;
        byte_t* staged = nullptr;
        fold_t pairs;
        std::size_t row = 0;
        vector_view_t query;
        vector_norms_t query_norms {};

        multiply_row(inputs_t const& inputs, scratch_allocator_t allocator) noexcept
            : inputs(inputs), locators(allocator),
              staged(allocator.allocate(staged_rows_bytes(vector_dtype_k, 1, inputs.left.columns))),
              pairs {{}, {}, {}, *inputs.predicate} {}

        status_t reserve(std::size_t, std::size_t) noexcept {
            return !staged || (!filtered_k && failed(locators.resize_uninitialized(inputs.keys.extent(1))))
                       ? status_t::bad_alloc_k
                       : status_t::success_k;
        }
        void prime_row(std::size_t new_row) noexcept {
            row = new_row;
            // A product walks by dot, which never narrows at unit RMS.
            query = import_staged_rows<elementwise_kit_t, rmscast_kit_t, scalar_t>(inputs.left, row, 1, nk_kernel_dot_k,
                                                                                   staged)
                        .row(0)
                        .as_vector();
            inputs.counts[row] = 0;
            pairs.ids = output_ids_();
            pairs.products = to_span(inputs.products.row(row).as_vector());
            pairs.written = 0;
            pairs.scored = 0;
        }
        void rescore_quad(quad<node_id_t> const&, quad<vector_view_t> const&, quad<vector_norms_t> const&,
                          scores_result<dot_result_t>&, std::size_t, std::span<key_t const>) const noexcept {}

        /** Keeps the keys landed beside the quad, which the fold resolves its nodes through. */
        void rescore_quad(quad<node_id_t> const& ids, quad<vector_view_t> const&, quad<vector_norms_t> const&,
                          scores_result<dot_result_t>&, std::size_t, std::span<key_t const> keys) noexcept
            requires filtered_k
        {
            pairs.resolve.nodes = ids;
            for (std::size_t lane = 0; lane != keys.size(); ++lane) pairs.resolve.keys[lane] = keys[lane];
        }
        fold_t& fold_of_walk(walk_t const& walk) noexcept {
            pairs.resolve.shard = walk.shard;
            return pairs;
        }
        fold_t& fold_of_walk(walk_t const&) noexcept
            requires filtered_k
        {
            return pairs;
        }
        results_array_t const& stopping_candidates(walk_t const& walk) const noexcept { return walk.results; }
        void finish_walk(walk_t const&) noexcept {}

        /** A filtered row wrote its keys as it folded, so nothing is left to resolve. */
        std::size_t output_count() const noexcept { return filtered_k ? 0 : pairs.written; }
        nvme_locator<u32_t> output_locator(std::size_t output) const noexcept { return locators[output]; }
        void resolve_output_key(std::size_t output, key_t const& key) noexcept { inputs.keys(row, output) = key; }
        status_t finish_row() noexcept {
            inputs.counts[row] = pairs.scored;
            return status_t::success_k;
        }

        /** Where the fold writes: the locators that resolve once the walks finished. */
        std::span<nvme_locator<u32_t>> output_ids_() noexcept { return to_writable_span(locators); }

        /** Where a filtered fold writes: the row's own keys. */
        std::span<key_t> output_ids_() noexcept
            requires filtered_k
        {
            return to_span(inputs.keys.row(row).as_vector());
        }
    };

    /** The rows of attention, one per token: the token's heads rescore every scored batch of every
     *  walk into their running states, value rows looked up by the keys read beside the scored
     *  rows, only those @p predicate_type_ admits folding. Each slot imports its token's heads into
     *  its own staging, past them the @c f32 rows and softmax states that outputs of another layout
     *  or an empty carry cannot hold. */
    template <typename value_lookup_type_, typename predicate_type_>
    struct attend_row {
        using key_t = key_type_;
        using fold_t = dummy_fold_t;
        using rescoring_t = attention_rescoring<quads_kit_t, elementwise_kit_t, value_lookup_type_, predicate_type_&>;
        using slice_t = arena_allocator<byte_t>;
        inline static constexpr nvme_key_reads_t key_reads_k = nvme_key_reads_t::scored_k;
        inline static constexpr bool admitting_k = false;

        /** The batch every row of the call reads and writes. */
        struct inputs_t {
            rows_view_t queries;
            std::size_t head_count = 0;
            std::remove_reference_t<value_lookup_type_>* value_lookup = nullptr;
            predicate_type_* predicate = nullptr;
            f32_t score_scale = 1;
            rows_span_t outputs;
            std::span<softmax_state_t> carry;
        };

        inputs_t const& inputs;
        byte_t* staged = nullptr;
        std::size_t token = 0;
        status_t status = status_t::success_k;
        vectors_view_t heads;
        vector_view_t query;
        vector_norms_t query_norms {};

        attend_row(inputs_t const& inputs, scratch_allocator_t allocator) noexcept
            : inputs(inputs), staged(allocator.allocate(staged_bytes(inputs))) {}

        /** Bytes one slot stages: the token's imported heads, then their @c f32 rows and states. */
        static std::size_t staged_bytes(inputs_t const& inputs) noexcept {
            std::size_t const head_count = inputs.head_count, columns = inputs.queries.columns;
            return staged_rows_bytes(vector_dtype_k, head_count, columns) +
                   slice_t::reserved_bytes(head_count * columns * sizeof(f32_t)) + head_count * sizeof(softmax_state_t);
        }

        /** Whether the outputs are packed @c f32 rows, which the heads accumulate into in place. */
        bool in_place() const noexcept {
            return inputs.outputs.template holds<f32_t>() &&
                   inputs.outputs.stride == inputs.queries.columns * sizeof(f32_t);
        }

        /** The token's heads over its rows of the outputs and the carry, or the staged stand-ins
         *  for them, refusing into the row's status. */
        rescoring_t rescoring() noexcept {
            std::size_t const first = token * inputs.head_count, columns = inputs.queries.columns;
            byte_t* const accumulators = staged + staged_rows_bytes(vector_dtype_k, inputs.head_count, columns);
            byte_t* const states = accumulators + slice_t::reserved_bytes(inputs.head_count * columns * sizeof(f32_t));
            return {heads,
                    *inputs.value_lookup,
                    inputs.score_scale,
                    {reinterpret_cast<f32_t*>(in_place() ? inputs.outputs.row(first) : accumulators), inputs.head_count,
                     columns},
                    inputs.carry.empty() ? std::span {reinterpret_cast<softmax_state_t*>(states), inputs.head_count}
                                         : inputs.carry.subspan(first, inputs.head_count),
                    status,
                    *inputs.predicate};
        }

        status_t reserve(std::size_t, std::size_t) noexcept {
            return staged ? status_t::success_k : status_t::bad_alloc_k;
        }
        void prime_row(std::size_t new_token) noexcept {
            token = new_token;
            status = status_t::success_k;
            // Attention walks by dot, which never narrows at unit RMS.
            heads = import_staged_rows<elementwise_kit_t, rmscast_kit_t, scalar_t>(
                inputs.queries, token * inputs.head_count, inputs.head_count, nk_kernel_dot_k, staged);
            query = heads.row(0).as_vector();
            rescoring().clear();
        }
        void rescore_quad(quad<node_id_t> const& ids, quad<vector_view_t> const& vectors,
                          quad<vector_norms_t> const& norms, scores_result<dot_result_t>& scores, std::size_t count,
                          std::span<key_t const> keys) noexcept {
            rescoring_t const attention = rescoring();
            attention(ids, vectors, norms, scores, count, keys);
        }
        fold_t fold_of_walk(walk_t const&) const noexcept { return {}; }
        results_array_t const& stopping_candidates(walk_t const& walk) const noexcept { return walk.results; }
        void finish_walk(walk_t const&) noexcept {}
        std::size_t output_count() const noexcept { return 0; }
        nvme_locator<u32_t> output_locator(std::size_t) const noexcept { return {}; }
        void resolve_output_key(std::size_t, key_t const&) noexcept {}
        status_t finish_row() noexcept {
            if (failed(status)) return status;
            rescoring_t const attention = rescoring();
            attention.finish();
            if (!in_place())
                for (std::size_t head = 0; head != inputs.head_count; ++head)
                    export_row<elementwise_kit_t>(attention.outputs.row(head).data(), nk_f32_k, inputs.outputs,
                                                  token * inputs.head_count + head);
            return status_t::success_k;
        }
    };

    /** Walks @p rows_count rows of @p row_type_ over every shard, each walk as wide as @p config
     *  asks and each row staging @p staged_bytes of imported queries. */
    template <typename row_type_>
    expected<search_result_t> walk_rows_(std::size_t rows_count, search_config_t config,
                                         typename row_type_::inputs_t const& inputs, std::size_t staged_bytes,
                                         std::span<std::size_t const> counts, std::span<u64_t> latencies,
                                         executor_t& executor) noexcept {
        return index_t::template walk_rows_<row_type_>(to_span(media_), to_span(parts_), kernel_, executor, rows_count,
                                                       non_zero_expansion_factor(config), inputs, staged_bytes, counts,
                                                       latencies);
    }

    /** The batched @ref search_if under @p heads, which only a @c dummy_predicate_t accompanies. */
    template <typename score_type_, typename heads_type_, typename predicate_type_>
    expected<search_result_t> search_rows_(rows_view_t queries, heads_type_ const& heads, executor_t& executor,
                                           search_config_t config, nk::matrix_span<key_t> keys,
                                           nk::matrix_span<score_type_> distances, std::span<std::size_t> counts,
                                           predicate_type_& predicate, std::span<u64_t> latencies) noexcept {
        std::size_t const queries_count = queries.rows;
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        if (queries_count && queries.columns != dimensions()) return {{}, status_t::config_mismatch_k};
        using row_t = search_row<heads_type_, predicate_type_, score_type_>;
        typename row_t::inputs_t const inputs {
            queries, kernel_, &heads, &predicate, non_zero_expansion_factor(config), keys, distances, counts};
        return walk_rows_<row_t>(queries_count, config, inputs, staged_rows_bytes(vector_dtype_k, 1, queries.columns),
                                 counts, latencies, executor);
    }

#pragma endregion Rows

#pragma region Scans

    /** One read a scan issues: the shard, and the bytes in its part from the part's own start; none
     *  when empty. */
    struct extent_t {

        /** The shard read. */
        u32_t shard = 0;

        /** Where the bytes start. */
        u64_t begin = 0;

        /** How many bytes; zero ends the task. */
        u64_t bytes = 0;
    };

    /** One staging slot of a scan: the task whose reads it holds, what each read asked for, where
     *  each one's bytes start in its covering read, and the reads still to land. */
    template <std::size_t reads_per_task_>
    struct scan_slot {

        /** The task the reads belong to. */
        std::size_t task = 0;

        /** What each read asked for. */
        std::array<extent_t, reads_per_task_> extents {};

        /** The offset of each extent's first byte in its covering read. */
        std::array<u64_t, reads_per_task_> offsets {};

        /** Reads issued and not landed yet. */
        std::size_t reads_left = 0;
    };

    /** One run of whole rows of one column a scan reads: where it starts, its rows, and where its
     *  first row sits among the rows the scan covers. */
    struct run_t {

        /** The shard and the run's first row inside it. */
        nvme_locator<u32_t> start;

        /** Rows in the run; zero past the last run. */
        std::size_t rows = 0;

        /** The run's first row, counted from the first row the scan covers. */
        std::size_t position = 0;
    };

    /** Begins a call of @p scans scans over every shard, each landing @p reads_per_task_ reads per
     *  task: every thread's arena sized for their slots and @p scratch_bytes more, and every engine
     *  for a queue depth of slots, each as many staging buffers as reads, and at least @p buffers
     *  for a walk sharing the call, every buffer as wide as a walk's. */
    template <std::size_t reads_per_task_ = 1>
    expected<call_t> begin_scans_(executor_t& executor, std::size_t scans, std::size_t scratch_bytes = 0,
                                  std::size_t buffers = 0) const noexcept {
        using slice_t = arena_allocator<byte_t>;
        std::size_t const depth = max_of<std::size_t>(executor.queue_depth(), 1);
        // What `scan_reads_` carves per scan, and padding to the first boundary.
        std::size_t const scan_bytes = slice_t::reserved_bytes(depth * sizeof(scan_slot<reads_per_task_>)) +
                                       slice_t::reserved_bytes(depth * sizeof(u32_t)) +
                                       slice_t::reserved_bytes(depth * reads_per_task_ * sizeof(io_completion_t));
        return nvme_begin_call(executor, to_span(media_), scans * scan_bytes + scratch_bytes + default_alignment_k,
                               max_of(depth * reads_per_task_, buffers), staging_bytes_());
    }

    /**
     *  @brief Scans the reads of @p tasks_count tasks over every thread of @p executor, as many
     *      slots in flight per thread as its engine's depth, and answers the reads issued.
     *
     *  A thread claims a task whole and asks @p describe for its reads in turn, as
     *  `describe(task, progress)` with @p progress zero at the claim and kept between the calls;
     *  each call answers @p reads_per_task_ extents, every one non-empty, or an empty first one,
     *  which ends the task. The reads of one call share a slot, read r of slot s landing in buffer
     *  s + r · depth, and @p consume takes them once all landed, as `consume(thread, task, extents,
     *  bytes)` on the thread that read them, both arrays in read order. Runs inside a call
     *  @ref begin_scans_ began for as many reads per task; a failed read stops every thread, which
     *  drains its engine.
     */
    template <std::size_t reads_per_task_ = 1, typename describe_type_, typename consume_type_>
    expected<nvme_reads_t> scan_reads_(executor_t& executor, std::size_t tasks_count, describe_type_&& describe,
                                       consume_type_&& consume) const noexcept {
        using slot_t = scan_slot<reads_per_task_>;
        constexpr std::size_t unclaimed_k = ~std::size_t {0};
        std::size_t const depth = max_of<std::size_t>(executor.queue_depth(), 1);
        alignas(default_alignment_k) std::size_t cursor = 0;
        status_t failure = status_t::success_k;
        nvme_reads_t issued;
        auto scan_thread = [&](std::size_t thread, std::size_t) noexcept {
            engine_t& engine = executor.engine(thread);
            arena_t& arena = executor.arena(thread);
            auto* const slots = reinterpret_cast<slot_t*>(arena.allocate(depth * sizeof(slot_t), default_alignment_k));
            auto* const free_slots = reinterpret_cast<u32_t*>(
                arena.allocate(depth * sizeof(u32_t), default_alignment_k));
            auto* const completions = reinterpret_cast<io_completion_t*>(
                arena.allocate(depth * reads_per_task_ * sizeof(io_completion_t), default_alignment_k));
            status_t local = slots && free_slots && completions ? status_t::success_k : status_t::bad_alloc_k;
            std::size_t free_count = succeeded(local) ? depth : 0, outstanding = 0;
            for (std::size_t slot = 0; slot != free_count; ++slot) free_slots[slot] = static_cast<u32_t>(slot);
            std::size_t task = unclaimed_k;
            u64_t progress = 0;
            // The slot whose reads are being enqueued and its next read, which a full engine may
            // refuse until a completion frees room in it; `depth` while none is.
            std::size_t issuing = depth, next_read = 0;
            nvme_reads_t reads;
            while (true) {
                while (succeeded(local) &&
                       succeeded(std::atomic_ref<status_t>(failure).load(std::memory_order_relaxed))) {
                    if (issuing == depth) {
                        if (!free_count) break;
                        if (task == unclaimed_k)
                            task = min_of(std::atomic_ref<std::size_t>(cursor).fetch_add(1, std::memory_order_relaxed),
                                          tasks_count),
                            progress = 0;
                        if (task == tasks_count) break;
                        std::array<extent_t, reads_per_task_> const extents = describe(task, progress);
                        if (!extents[0].bytes) {
                            task = unclaimed_k;
                            continue;
                        }
                        issuing = free_slots[--free_count];
                        next_read = 0;
                        slots[issuing] = slot_t {task, extents, {}, reads_per_task_};
                    }
                    slot_t& slot = slots[issuing];
                    extent_t const& extent = slot.extents[next_read];
                    medium_t const& medium = media_[extent.shard];
                    u64_t const begin = byte_base_of(medium) + extent.begin;
                    // The buffer alone names what landed, so the context stays unused.
                    io_request_t const request = nvme_covering_read(
                        begin, extent.bytes, block_bytes_of(medium), extent.shard,
                        static_cast<io_buffer_id_t>(issuing + next_read * depth), 0);
                    slot.offsets[next_read] = begin - request.offset;
                    status_t const enqueued = engine.enqueue_read(request);
                    if (enqueued == status_t::overflow_k && outstanding) break;
                    // A full engine holding none of this thread's reads would never drain, so the
                    // thread fails instead.
                    if (failed(enqueued)) {
                        local = enqueued == status_t::overflow_k ? status_t::io_k : enqueued;
                        break;
                    }
                    reads += nvme_reads_t {1, request.length};
                    ++outstanding;
                    if (++next_read == reads_per_task_) issuing = depth;
                }
                if (!outstanding) break;
                if (status_t const submitted = engine.submit(io_wait_t::blocking_k); failed(submitted)) {
                    // A refused submission returns with nothing in flight and nothing left to poll.
                    if (succeeded(local)) local = submitted;
                    break;
                }
                std::span<io_completion_t> const polling {completions, depth * reads_per_task_};
                for (std::size_t polled = engine.poll_completions(polling); polled;
                     polled = engine.poll_completions(polling))
                    for (io_completion_t const& completion : polling.first(polled)) {
                        --outstanding;
                        std::size_t const landed = completion.buffer % depth, read = completion.buffer / depth;
                        slot_t& slot = slots[landed];
                        u64_t const needed = slot.offsets[read] + slot.extents[read].bytes;
                        if (succeeded(local) &&
                            (completion.result < 0 || static_cast<u64_t>(completion.result) < needed))
                            local = status_t::io_k;
                        if (--slot.reads_left) continue;
                        if (succeeded(local)) {
                            std::array<byte_t const*, reads_per_task_> bytes;
                            for (std::size_t each = 0; each != reads_per_task_; ++each)
                                bytes[each] = engine.buffer(static_cast<io_buffer_id_t>(landed + each * depth)) +
                                              slot.offsets[each];
                            consume(thread, slot.task, slot.extents, bytes);
                        }
                        free_slots[free_count++] = static_cast<u32_t>(landed);
                    }
            }
            std::atomic_ref<std::size_t>(issued.count).fetch_add(reads.count, std::memory_order_relaxed);
            std::atomic_ref<std::size_t>(issued.bytes).fetch_add(reads.bytes, std::memory_order_relaxed);
            if (succeeded(local)) return;
            status_t unset = status_t::success_k;
            std::atomic_ref<status_t>(failure).compare_exchange_strong(unset, local, std::memory_order_relaxed,
                                                                       std::memory_order_relaxed);
        };
        if (status_t const fanned = executor.for_threads(scan_thread); failed(fanned)) return {issued, fanned};
        return {issued, failure};
    }

    /** One column of every part: where its rows start, and how wide they are. */
    struct column_t {
        u64_t nvme_part_t::* begin = nullptr;
        u32_t nvme_part_t::* row_bytes = nullptr;
    };

    inline static constexpr column_t key_column_k {&nvme_part_t::keys_begin, &nvme_part_t::key_row_bytes};
    inline static constexpr column_t value_column_k {&nvme_part_t::values_begin, &nvme_part_t::value_row_bytes};

    /** Bytes of one staging buffer: the covering read of the widest row any part holds. */
    std::size_t staging_bytes_() const noexcept { return index_t::staging_bytes_(to_span(media_), to_span(parts_)); }

    /** Rows of a column @p row_bytes wide that one scan read of part @p part takes: as many as a
     *  staging buffer covers wherever the run starts, and at least one, since it covers any row. */
    std::size_t scan_run_rows_(std::size_t part, std::size_t row_bytes) const noexcept {
        u64_t const block = block_bytes_of(media_[part]);
        return static_cast<std::size_t>((staging_bytes_() / block * block - (block - 1)) / row_bytes);
    }

    /** Runs of @p column over the @p count placement-order rows from row @p first on, cut at every
     *  part's end and every scan run, so a row never straddles two reads. */
    std::size_t runs_count_(std::size_t first, std::size_t count, column_t column) const noexcept {
        std::size_t runs = 0, base = 0;
        for (std::size_t part = 0; part != parts_.size(); base += parts_[part].nodes, ++part) {
            std::size_t const low = max_of(first, base), high = min_of(first + count, base + parts_[part].nodes);
            if (low < high) runs += divide_round_up(high - low, scan_run_rows_(part, parts_[part].*column.row_bytes));
        }
        return runs;
    }

    /** Run @p run of the runs @ref runs_count_ counts, or an empty one past the last. */
    run_t locate_run_(std::size_t first, std::size_t count, std::size_t run, column_t column) const noexcept {
        std::size_t base = 0;
        for (u32_t part = 0; part != parts_.size(); base += parts_[part].nodes, ++part) {
            std::size_t const low = max_of(first, base), high = min_of(first + count, base + parts_[part].nodes);
            if (low >= high) continue;
            std::size_t const run_rows = scan_run_rows_(part, parts_[part].*column.row_bytes);
            std::size_t const runs = divide_round_up(high - low, run_rows);
            if (run >= runs) {
                run -= runs;
                continue;
            }
            std::size_t const offset = run * run_rows;
            return {{part, static_cast<node_id_t>(low - base + offset)},
                    min_of(run_rows, high - low - offset),
                    low - first + offset};
        }
        return {};
    }

    /** The read of @p run in @p column. */
    extent_t run_extent_(run_t const& run, column_t column) const noexcept {
        nvme_part_t const& part = parts_[run.start.shard];
        u64_t const row_bytes = part.*column.row_bytes;
        return {run.start.shard, part.*column.begin + run.start.node * row_bytes, run.rows * row_bytes};
    }

    /** Copies the @p keys.size() placement-order rows from row @p first on into @p keys and, unless
     *  it holds no rows, exports them into @p vectors, as wide as the index: each column scanned in
     *  runs of whole rows, one covering read per run. */
    expected<search_result_t> read_rows_(std::size_t first, std::span<key_t> keys, rows_span_t vectors,
                                         executor_t& executor) const noexcept {
        std::size_t const count = keys.size();
        std::size_t const key_runs = runs_count_(first, count, key_column_k);
        std::size_t const value_runs = vectors.rows ? runs_count_(first, count, value_column_k) : 0;
        auto [call, call_status] = begin_scans_(executor, 1);
        if (failed(call_status)) return {{}, call_status};
        // Tasks below `key_runs` read keys, the rest read values.
        auto run_of = [&](std::size_t task) noexcept {
            return task < key_runs ? locate_run_(first, count, task, key_column_k)
                                   : locate_run_(first, count, task - key_runs, value_column_k);
        };
        auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<extent_t, 1> {
            if (progress++) return {};
            return {run_extent_(run_of(task), task < key_runs ? key_column_k : value_column_k)};
        };
        auto consume = [&](std::size_t, std::size_t task, auto const& extents, auto const& landed) noexcept {
            run_t const run = run_of(task);
            byte_t const* const bytes = landed[0];
            std::size_t const row_bytes = static_cast<std::size_t>(extents[0].bytes / run.rows);
            dtype_t const key_dtype = parts_[extents[0].shard].key_dtype;
            for (std::size_t row = 0; row != run.rows; ++row)
                if (task < key_runs)
                    keys[run.position + row] = safegraphs_widened_key<key_t>(bytes + row * row_bytes, key_dtype);
                else
                    export_row<elementwise_kit_t>(bytes + row * row_bytes, vector_dtype_k, vectors, run.position + row);
        };
        auto const [reads, status] = scan_reads_(executor, key_runs + value_runs, describe, consume);
        search_result_t result;
        result.count = count;
        result.scan_reads = reads;
        return {result, status};
    }

    /** Whether two keys are the same, by their bytes, whatever their type. */
    static bool same_key_(key_t const& first, key_t const& second) noexcept {
        return std::memcmp(&first, &second, sizeof(key_t)) == 0;
    }

    /**
     *  @brief Scans every part's key column in runs, matching every row against an exact table of
     *      @p keys, and calls @p on_hit with the scanning thread, the batch position and the row's
     *      locator for every match, repeats included.
     *
     *  The table holds each distinct key's first batch position, the positions sharing it chained
     *  behind; it is exact, so no row is ever matched by a key it does not hold. Runs inside a call
     *  @ref begin_scans_ began, the table in thread zero's arena.
     */
    template <typename on_hit_type_>
    status_t scan_keys_(std::span<key_t const> keys, executor_t& executor, on_hit_type_&& on_hit) const noexcept {
        constexpr std::size_t empty_k = ~std::size_t {0};
        std::size_t const slots_count = ceil2(max_of<std::size_t>(keys.size() * 2, 16)), mask = slots_count - 1;
        scratch_allocator_t const scratch {executor.arena(0)};
        growing_buffer<std::size_t, scratch_allocator_t> heads {scratch}, chains {scratch};
        if (failed(heads.resize_uninitialized(slots_count)) || failed(chains.resize_uninitialized(keys.size())))
            return status_t::bad_alloc_k;
        for (std::size_t& head : heads) head = empty_k;
        auto slot_of = [&](key_t const& key) noexcept {
            std::size_t slot = mix_bits(std::hash<key_t> {}(key)) & mask;
            while (heads[slot] != empty_k && !same_key_(keys[heads[slot]], key)) slot = (slot + 1) & mask;
            return slot;
        };
        for (std::size_t position = 0; position != keys.size(); ++position) {
            std::size_t const slot = slot_of(keys[position]);
            if (heads[slot] == empty_k) heads[slot] = position, chains[position] = empty_k;
            else chains[position] = chains[heads[slot]], chains[heads[slot]] = position;
        }

        std::size_t const rows_count = size();
        auto describe = [&](std::size_t run, u64_t& progress) noexcept -> std::array<extent_t, 1> {
            if (progress++) return {};
            return {run_extent_(locate_run_(0, rows_count, run, key_column_k), key_column_k)};
        };
        auto consume = [&](std::size_t thread, std::size_t, auto const& extents, auto const& landed) noexcept {
            extent_t const& extent = extents[0];
            byte_t const* const bytes = landed[0];
            nvme_part_t const& part = parts_[extent.shard];
            std::size_t const first = static_cast<std::size_t>((extent.begin - part.keys_begin) / part.key_row_bytes);
            for (std::size_t row = 0; row != extent.bytes / part.key_row_bytes; ++row) {
                key_t const key = safegraphs_widened_key<key_t>(bytes + row * part.key_row_bytes, part.key_dtype);
                std::size_t const slot = slot_of(key);
                for (std::size_t position = heads[slot]; position != empty_k; position = chains[position])
                    on_hit(thread, position, nvme_locator<u32_t> {extent.shard, static_cast<node_id_t>(first + row)});
            }
        };
        return scan_reads_(executor, runs_count_(0, rows_count, key_column_k), describe, consume).status;
    }

    /** One match a thread's scan keeps for a query: its distance and its row's key. */
    struct scan_match_t {
        distance_t distance = 0;
        key_t key {};
    };

    /** One thread's share of a scan, from its own arena: as many matches per query as a ranking is
     *  wide, unordered, each query's kept count and worst distance, one step of engine dots, and
     *  one quad of rows' distances to every query. */
    struct scan_share_t {
        scan_match_t* matches = nullptr;
        u32_t* kept = nullptr;
        distance_t* thresholds = nullptr;
        byte_t* dots = nullptr;
        distance_t* quad_distances = nullptr;
    };

    /**
     *  @brief Exact rankings of every query over every row: each part's value column is read once,
     *      in runs, every run paired with the key run of its rows and scored against the batch.
     *
     *  A batch as wide as a tile streams through the matrix engine, packed once per block of the
     *  kit's pack width; a narrower one, or a scalar no engine multiplies, scores four rows at a
     *  time per query. Both finalize through the walk's quads kit. Every thread keeps its own
     *  matches of every query, and they merge once the column ends. @p predicate sees a row's key
     *  only once the row beats some query's worst match. One pass answers the whole batch, so
     *  every latency is that pass. The batch imports once into the first thread's arena, beside
     *  its norms and packs.
     */
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> scan_rows_(rows_view_t query_rows, executor_t& executor, nk::matrix_span<key_t> keys,
                                         nk::matrix_span<score_type_> distances, std::span<std::size_t> counts,
                                         predicate_type_& predicate, std::span<u64_t> latencies) noexcept {
        using slice_t = arena_allocator<byte_t>;
        constexpr bool filtered_k = !std::is_same_v<std::remove_cv_t<predicate_type_>, dummy_predicate_t>;
        std::size_t const queries_count = query_rows.rows, wanted = keys.extent(1), threads = executor.size();
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        if (!latencies.empty() && latencies.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (queries_count && query_rows.columns != dimensions()) return {{}, status_t::config_mismatch_k};
        std::memset(counts.data(), 0, counts.size_bytes());
        if (!queries_count || !wanted) return {{}, status_t::success_k};
        auto const started = std::chrono::steady_clock::now();

        // A batch narrower than a tile scores per query, as a lone query scans on RAM.
        std::size_t pack_columns = 1, stream_rows = 0, pack_stride = 0;
        bool tiled = false;
        if constexpr (has_tiles_k) {
            nk_size_t packed_bytes = 0;
            tiled = queries_count >= tiles_kit_t::tile_columns_k &&
                    tiles_kit_t::pack_size(tiles_kit_t::pack_columns_k, dimensions(), &packed_bytes) == nk_success_k;
            pack_columns = tiles_kit_t::pack_columns_k, stream_rows = tiles_kit_t::stream_rows_k;
            pack_stride = round_up_to_multiple<std::size_t>(packed_bytes, default_alignment_k);
        }
        std::size_t const packs = tiled ? divide_round_up(queries_count, pack_columns) : 0;
        std::size_t const padded_queries = round_up_to_multiple<std::size_t>(queries_count, 4);
        // Every arena takes a share, and thread zero's the packs, the queries' norms, the shares
        // and the merge besides, so every arena is sized for both.
        std::size_t const share_bytes = slice_t::reserved_bytes(queries_count * wanted * sizeof(scan_match_t)) +
                                        slice_t::reserved_bytes(queries_count * sizeof(u32_t)) +
                                        slice_t::reserved_bytes(queries_count * sizeof(distance_t)) +
                                        slice_t::reserved_bytes(stream_rows * pack_columns * sizeof(dot_result_t)) +
                                        slice_t::reserved_bytes(queries_count * quad_lanes_k * sizeof(distance_t));
        std::size_t const staged_bytes = staged_rows_bytes(vector_dtype_k, queries_count, query_rows.columns);
        std::size_t const home_bytes = slice_t::reserved_bytes(staged_bytes) +
                                       slice_t::reserved_bytes(packs * pack_stride) +
                                       slice_t::reserved_bytes(padded_queries * sizeof(vector_norms_t)) +
                                       slice_t::reserved_bytes(padded_queries * sizeof(dot_result_t)) +
                                       slice_t::reserved_bytes(threads * sizeof(scan_share_t)) +
                                       slice_t::reserved_bytes(threads * wanted * sizeof(scan_match_t));
        auto [call, call_status] = begin_scans_<2>(executor, 1, share_bytes + home_bytes);
        if (failed(call_status)) return {{}, call_status};
        if (status_t const opened = open_predicate(predicate, executor); failed(opened)) return {{}, opened};
        auto carve = [](arena_t& arena, std::size_t bytes) noexcept {
            return arena.allocate(max_of<std::size_t>(bytes, 1), default_alignment_k);
        };
        arena_t& home = executor.arena(0);
        auto* const staged = carve(home, staged_bytes);
        auto* const packed = carve(home, packs * pack_stride);
        auto* const norms = reinterpret_cast<vector_norms_t*>(carve(home, padded_queries * sizeof(vector_norms_t)));
        auto* const sumsqs = reinterpret_cast<dot_result_t*>(carve(home, padded_queries * sizeof(dot_result_t)));
        auto* const shares = reinterpret_cast<scan_share_t*>(carve(home, threads * sizeof(scan_share_t)));
        auto* const merged = reinterpret_cast<scan_match_t*>(carve(home, threads * wanted * sizeof(scan_match_t)));
        if (!staged || !packed || !norms || !sumsqs || !shares || !merged) return {{}, status_t::bad_alloc_k};
        vectors_view_t const queries = import_staged_rows<elementwise_kit_t, rmscast_kit_t, scalar_t>(
            query_rows, 0, queries_count, kernel_, staged);
        for (std::size_t thread = 0; thread != threads; ++thread) {
            arena_t& arena = executor.arena(thread);
            scan_share_t& share = *new (shares + thread) scan_share_t {
                reinterpret_cast<scan_match_t*>(carve(arena, queries_count * wanted * sizeof(scan_match_t))),
                reinterpret_cast<u32_t*>(carve(arena, queries_count * sizeof(u32_t))),
                reinterpret_cast<distance_t*>(carve(arena, queries_count * sizeof(distance_t))),
                carve(arena, stream_rows * pack_columns * sizeof(dot_result_t)),
                reinterpret_cast<distance_t*>(carve(arena, queries_count * quad_lanes_k * sizeof(distance_t)))};
            if (!share.matches || !share.kept || !share.thresholds || !share.dots || !share.quad_distances)
                return {{}, status_t::bad_alloc_k};
            std::memset(share.kept, 0, queries_count * sizeof(u32_t));
            for (std::size_t query = 0; query != queries_count; ++query)
                share.thresholds[query] = std::numeric_limits<distance_t>::max();
            if (status_t const reserved = reserve_predicate(thread_predicate(predicate, thread),
                                                            scan_round_capacity<tiles_kit_t>());
                failed(reserved))
                return {{}, reserved};
        }
        for (std::size_t query = 0; query != padded_queries; ++query) {
            norms[query] = query < queries_count ? index_t::norms_(kernel_, queries.row(query).as_vector())
                                                 : vector_norms_t {};
            sumsqs[query] = norms[query].sumsq;
        }
        if constexpr (has_tiles_k) {
            for (std::size_t pack = 0; pack != packs; ++pack) {
                std::size_t const first = pack * pack_columns, columns = min_of(pack_columns, queries_count - first);
                vectors_view_t const block = queries(nk::range(first, first + columns), nk::slice);
                [[maybe_unused]] nk_status_t const packed_status = tiles_kit_t::pack(
                    block.raw_values_data(), columns, dimensions(), static_cast<std::size_t>(block.stride_bytes(0)),
                    packed + pack * pack_stride, 0, columns, nullptr);
                assert(packed_status == nk_success_k && "A matrix engine packs any block");
            }
        }

        // Keeps a match in a thread's share of a query, and answers that query's new worst.
        auto land_into = [wanted](scan_share_t const& share, std::size_t query, scan_match_t match) noexcept {
            scan_match_t* const kept = share.matches + query * wanted;
            u32_t& count = share.kept[query];
            if (count < wanted) {
                kept[count++] = match;
                if (count < wanted) return std::numeric_limits<distance_t>::max();
            }
            else {
                std::size_t worst = 0;
                for (std::size_t slot = 1; slot != wanted; ++slot)
                    if (kept[slot].distance > kept[worst].distance) worst = slot;
                kept[worst] = match;
            }
            distance_t threshold = kept[0].distance;
            for (std::size_t slot = 1; slot != wanted; ++slot) threshold = max_of(threshold, kept[slot].distance);
            return threshold;
        };

        // Runs follow the wider of the key and value rows, so both runs of a task fit a buffer.
        std::size_t const rows_count = size();
        column_t const runs_column = parts_[0].value_row_bytes >= parts_[0].key_row_bytes ? value_column_k
                                                                                          : key_column_k;
        auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<extent_t, 2> {
            if (progress++) return {};
            run_t const run = locate_run_(0, rows_count, task, runs_column);
            return {run_extent_(run, key_column_k), run_extent_(run, value_column_k)};
        };
        auto consume = [&](std::size_t thread, std::size_t, std::array<extent_t, 2> const& extents,
                           std::array<byte_t const*, 2> const& landed) noexcept {
            scan_share_t const& share = shares[thread];
            nvme_part_t const& part = parts_[extents[1].shard];
            std::size_t const run_rows = static_cast<std::size_t>(extents[1].bytes / part.value_row_bytes);
            auto key_of = [&](std::size_t row) noexcept {
                return safegraphs_widened_key<key_t>(landed[0] + row * part.key_row_bytes, part.key_dtype);
            };
            auto row_of = [&](std::size_t row) noexcept {
                return vector_view_t {reinterpret_cast<scalar_t const*>(landed[1] + row * part.value_row_bytes),
                                      part.dimensions};
            };
            auto admits = [&]() noexcept {
                if constexpr (filtered_k) return node_predicate {thread_predicate(predicate, thread), key_of};
                else return dummy_predicate_t {};
            }();
            if constexpr (has_tiles_k)
                if (tiled) {
                    using raw_dot_t = typename dot_result_t::raw_t;
                    // Rows may be padded past the logical width the queries were packed at.
                    vectors_view_t const dense {reinterpret_cast<scalar_t const*>(landed[1]), run_rows,
                                                std::size_t {part.dimensions}};
                    nk::shape_storage_<2> const logical {
                        {run_rows, std::size_t {part.dimensions}},
                        {static_cast<decltype(dense.stride_bytes(0))>(part.value_row_bytes), dense.stride_bytes(1)},
                        2};
                    vectors_view_t const rows {dense.byte_data(), logical};
                    nk::matrix_span<raw_dot_t> const dots {reinterpret_cast<raw_dot_t*>(share.dots), stream_rows,
                                                           pack_columns};
                    auto row_sumsq = [&](std::size_t row) noexcept {
                        return index_t::norms_(kernel_, row_of(row)).sumsq;
                    };
                    for (std::size_t pack = 0; pack != packs; ++pack) {
                        std::size_t const first = pack * pack_columns;
                        std::size_t const columns = min_of(pack_columns, queries_count - first);
                        auto land = [&](std::size_t column, scored_node_t scored) noexcept {
                            return land_into(share, first + column, {scored.distance, key_of(scored.node_id)});
                        };
                        scan_tiles<tiles_kit_t, quads_kit_t>(kernel_, rows, 0, packed + pack * pack_stride,
                                                             std::span<dot_result_t const>(sumsqs + first, columns),
                                                             std::span<distance_t>(share.thresholds + first, columns),
                                                             dots, row_sumsq, admits, land);
                    }
                    return;
                }
            for (std::size_t row = 0; row < run_rows; row += 4) {
                std::size_t const lanes = min_of<std::size_t>(4, run_rows - row);
                quad<vector_view_t> lane_rows;
                quad<vector_norms_t> lane_norms;
                for (std::size_t lane = 0; lane != 4; ++lane) {
                    lane_rows[lane] = row_of(row + min_of(lane, lanes - 1));
                    lane_norms[lane] = index_t::norms_(kernel_, lane_rows[lane]);
                }
                // Every query scores the quad first, so a lane beating some worst is asked once.
                quad<bool> contends {};
                for (std::size_t query = 0; query != queries_count; ++query) {
                    quad<distance_t> const lane_distances = dense::distances<quads_kit_t>(
                        kernel_, queries.row(query).as_vector(), norms[query], lane_rows, lane_norms);
                    std::memcpy(share.quad_distances + query * quad_lanes_k, lane_distances.data(),
                                sizeof(lane_distances));
                    for (std::size_t lane = 0; lane != lanes; ++lane)
                        contends[lane] |= lane_distances[lane] < share.thresholds[query];
                }
                auto keep_row = [&](scan_payload_t contender) noexcept {
                    for (std::size_t query = 0; query != queries_count; ++query) {
                        distance_t const distance = share.quad_distances[query * quad_lanes_k + contender - row];
                        if (distance < share.thresholds[query])
                            share.thresholds[query] = land_into(share, query, {distance, key_of(contender)});
                    }
                };
                quad<node_id_t> nodes;
                quad<scan_payload_t> payloads;
                std::size_t filled = 0;
                for (std::size_t lane = 0; lane != lanes; ++lane) {
                    if (!contends[lane]) continue;
                    nodes[filled] = static_cast<node_id_t>(row + lane);
                    payloads[filled++] = static_cast<scan_payload_t>(row + lane);
                }
                ask_contenders(admits, nodes, payloads, filled, keep_row);
                settle_predicate(admits, keep_row);
            }
        };
        auto const [reads, status] = scan_reads_<2>(executor, runs_count_(0, rows_count, runs_column), describe,
                                                    consume);
        search_result_t result;
        result.scan_reads = reads;
        result.computed_distances = rows_count * queries_count;
        if (failed(status)) return {result, status};

        // Every thread's matches of a query merge, the nearest landing ascending.
        auto nearer = [](scan_match_t const& first, scan_match_t const& second) noexcept {
            return first.distance < second.distance;
        };
        for (std::size_t query = 0; query != queries_count; ++query) {
            std::size_t gathered = 0;
            for (std::size_t thread = 0; thread != threads; ++thread) {
                scan_share_t const& share = shares[thread];
                std::memcpy(merged + gathered, share.matches + query * wanted,
                            share.kept[query] * sizeof(scan_match_t));
                gathered += share.kept[query];
            }
            std::size_t const kept = min_of(gathered, wanted);
            partial_sort(std::span<scan_match_t>(merged, gathered), kept, nearer);
            for (std::size_t rank = 0; rank != kept; ++rank)
                keys(query, rank) = merged[rank].key, distances(query, rank) = merged[rank].distance;
            counts[query] = kept;
            result.count += kept;
        }
        u64_t const elapsed = static_cast<u64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
        for (u64_t& latency : latencies) latency = elapsed;
        return {result, status_t::success_k};
    }

#pragma endregion Scans

#pragma region MaxSim

    /** A row the MaxSim key scan listed: where it sits, and the candidate whose key it holds. */
    struct maxsim_match_t {

        /** The row's shard and node. */
        nvme_locator<u32_t> locator;

        /** The candidate the row folds into. */
        u64_t position = 0;
    };

    static_assert(sizeof(maxsim_match_t) == 16, "A listed row is a 16-byte hot entry");

    /**
     *  @brief Late interaction nominated by walks, inside the call @ref maxsim_if began: every
     *      query row walks every shard, and each query sorts its rows' keys and keeps each once.
     *
     *  One key scan counts every candidate's rows and lists them, each thread into its own arena,
     *  and each candidate's rows are read on the one thread claiming it, so its maxima take plain
     *  maxima. A query's maxima start where the previous query's end, one per candidate and row.
     */
    template <typename land_type_, typename predicate_type_>
    expected<search_result_t> walk_maxsim_(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                           vectors_view_t query_rows, std::span<vector_norms_t const> query_norms,
                                           walk_shape_t const& shape, executor_t& executor, std::size_t wanted,
                                           land_type_& land, predicate_type_& predicate,
                                           std::span<u64_t> latencies) noexcept {
        using row_t = search_row<dummy_heads_t, predicate_type_, distance_t>;
        using matches_t = growing_buffer<maxsim_match_t, scratch_allocator_t>;
        std::size_t const queries_count = query_offsets.size() - 1, rows_count = query_rows.extent(0);
        std::size_t const width = shape.expansion_factor;
        scratch_allocator_t const scratch {executor.arena(0)};
        growing_buffer<key_t, scratch_allocator_t> candidates {scratch};
        growing_buffer<distance_t, scratch_allocator_t> distances {scratch};
        growing_buffer<std::size_t, scratch_allocator_t> walked {scratch}, segment_starts {scratch},
            maxima_starts {scratch};
        if (failed(candidates.resize_uninitialized(rows_count * width)) ||
            failed(distances.resize_uninitialized(rows_count * width)) ||
            failed(walked.resize_uninitialized(rows_count)) ||
            failed(segment_starts.resize_uninitialized(queries_count + 1)) ||
            failed(maxima_starts.resize_uninitialized(queries_count + 1)))
            return {{}, status_t::bad_alloc_k};
        nk::matrix_span<key_t> const walk_keys {candidates.data(), rows_count, width};
        nk::matrix_span<distance_t> const walk_distances {distances.data(), rows_count, width};
        typename row_t::inputs_t const inputs {queries.slice(query_offsets.front(), rows_count),
                                               kernel_,
                                               nullptr,
                                               &predicate,
                                               width,
                                               walk_keys,
                                               walk_distances,
                                               to_writable_span(walked)};
        auto [result, walk_status] = row_walker<index_t, row_t>::run_walks(to_span(media_), to_span(parts_), executor,
                                                                           shape, inputs, to_span(walked), latencies);
        result.count = 0;
        if (failed(walk_status)) return {result, walk_status};

        // Each query gathers its rows' keys behind the previous query's, sorts them by bytes, and
        // keeps each once.
        auto by_bytes = [](key_t const& first, key_t const& second) noexcept {
            return std::memcmp(&first, &second, sizeof(key_t)) < 0;
        };
        std::size_t nominated = 0, maxima_count = 0;
        for (std::size_t query = 0; query != queries_count; ++query) {
            std::size_t const begin = nominated;
            for (std::size_t row = query_offsets[query] - query_offsets.front();
                 row != query_offsets[query + 1] - query_offsets.front(); ++row)
                for (std::size_t hit = 0; hit != walked[row]; ++hit) candidates[nominated++] = walk_keys(row, hit);
            std::span<key_t> const gathered = to_writable_span(candidates).subspan(begin, nominated - begin);
            partial_sort(gathered, gathered.size(), by_bytes);
            nominated = begin;
            for (key_t const& key : gathered)
                if (nominated == begin || !same_key_(candidates[nominated - 1], key)) candidates[nominated++] = key;
            segment_starts[query] = begin;
            maxima_starts[query] = maxima_count;
            maxima_count += (nominated - begin) * (query_offsets[query + 1] - query_offsets[query]);
        }
        segment_starts[queries_count] = nominated;
        maxima_starts[queries_count] = maxima_count;
        if (!nominated) return {result, status_t::success_k};

        // One key scan counts every candidate's rows and lists them, each thread in its own arena.
        std::span<key_t const> const nominations = to_span(candidates).first(nominated);
        growing_buffer<std::size_t, scratch_allocator_t> row_starts {scratch};
        growing_buffer<matches_t, scratch_allocator_t> matches {scratch};
        if (failed(row_starts.resize(nominated + 1)) || failed(matches.reserve_exactly(executor.size())))
            return {result, status_t::bad_alloc_k};
        for (std::size_t thread = 0; thread != executor.size(); ++thread)
            matches.push_back_reserved(matches_t {scratch_allocator_t {executor.arena(thread)}});
        status_t listing = status_t::success_k;
        status_t const scanned = scan_keys_(
            nominations, executor, [&](std::size_t thread, std::size_t position, nvme_locator<u32_t> locator) noexcept {
                std::atomic_ref<std::size_t>(row_starts[position]).fetch_add(1, std::memory_order_relaxed);
                if (failed(matches[thread].push_back({locator, position})))
                    std::atomic_ref<status_t>(listing).store(status_t::bad_alloc_k, std::memory_order_relaxed);
            });
        if (failed(scanned)) return {result, scanned};
        if (failed(listing)) return {result, listing};

        // Each count becomes its candidate's end, and every listed row steps it back once.
        for (std::size_t position = 1; position <= nominated; ++position)
            row_starts[position] += row_starts[position - 1];
        growing_buffer<nvme_locator<u32_t>, scratch_allocator_t> locators {scratch};
        if (failed(locators.resize_uninitialized(row_starts[nominated]))) return {result, status_t::bad_alloc_k};
        for (matches_t const& listed : matches)
            for (maxsim_match_t const& match : listed) locators[--row_starts[match.position]] = match.locator;

        growing_buffer<dot_result_t, scratch_allocator_t> folded {scratch};
        if (failed(folded.resize_uninitialized(maxima_count))) return {result, status_t::bad_alloc_k};
        for (dot_result_t& maximum : folded) maximum = dot_result_t::finite_min();
        std::span<dot_result_t> const maxima = to_writable_span(folded);
        std::span<std::size_t const> const starts = to_span(segment_starts);
        auto describe = [&](std::size_t position, u64_t& progress) noexcept -> std::array<extent_t, 1> {
            std::size_t const row = row_starts[position] + static_cast<std::size_t>(progress++);
            if (row == row_starts[position + 1]) return {};
            nvme_locator<u32_t> const locator = locators[row];
            nvme_part_t const& part = parts_[locator.shard];
            return {extent_t {locator.shard, part.values_begin + u64_t(locator.node) * part.value_row_bytes,
                              part.value_row_bytes}};
        };
        auto consume = [&](std::size_t, std::size_t position, std::array<extent_t, 1> const& extents,
                           std::array<byte_t const*, 1> const& landed) noexcept {
            std::size_t const query = static_cast<std::size_t>(lower_bound(starts, position + 1) - starts.data()) - 1;
            std::size_t const first = query_offsets[query] - query_offsets.front();
            std::size_t const rows = query_offsets[query + 1] - query_offsets[query];
            vector_view_t const row {reinterpret_cast<scalar_t const*>(landed[0]), parts_[extents[0].shard].dimensions};
            fold_row_(row, query_rows(nk::range(first, first + rows), nk::slice), query_norms.subspan(first, rows),
                      maxima.subspan(maxima_starts[query] + (position - starts[query]) * rows, rows),
                      [](dot_result_t& maximum, dot_result_t dot) noexcept { maximum = max_of(maximum, dot); });
        };
        auto const [reads, read_status] = scan_reads_(executor, nominated, describe, consume);
        result.value_reads += reads;
        if (failed(read_status)) return {result, read_status};

        for (std::size_t query = 0; query != queries_count; ++query)
            result.computed_distances += (row_starts[starts[query + 1]] - row_starts[starts[query]]) *
                                         (query_offsets[query + 1] - query_offsets[query]);
        // Without a bound, every thread's ranking holds the most candidates any query has.
        std::size_t capacity = wanted;
        for (std::size_t query = 0; !wanted && query != queries_count; ++query)
            capacity = max_of(capacity, starts[query + 1] - starts[query]);
        growing_buffer<maxsim_ranked<total_t>, scratch_allocator_t> ranked {scratch};
        if (failed(ranked.resize_uninitialized(executor.size() * capacity))) return {result, status_t::bad_alloc_k};
        nk::matrix_span<maxsim_ranked<total_t>> const rankings {ranked.data(), executor.size(), capacity};
        status_t landing = status_t::success_k;
        status_t const ranking = executor.for_tasks(queries_count, [&](std::size_t query, std::size_t thread,
                                                                       std::size_t) noexcept {
            std::size_t const begin = starts[query], end = starts[query + 1];
            std::size_t const rows = query_offsets[query + 1] - query_offsets[query];
            nk::matrix_span<dot_result_t> const query_maxima {maxima.data() + maxima_starts[query], end - begin, rows};
            if (status_t const landed = rank_candidates_(nominations.subspan(begin, end - begin), query_maxima,
                                                         to_span(rankings.row(thread).as_vector()), thread, query,
                                                         land);
                failed(landed))
                std::atomic_ref<status_t>(landing).store(landed, std::memory_order_relaxed);
        });
        return {result, failed(ranking) ? ranking : landing};
    }

    /**
     *  @brief Late interaction nominated by a scan, inside the call @ref maxsim_if began: every key
     *      is a candidate, each distinct one asked of the predicate once.
     *
     *  A scan reads every key and value run once, folding each row through compare-exchanges, since
     *  a key's rows land on any thread; its maxima hold one value per distinct key and query row,
     *  refused with @c bad_alloc_k past memory.
     */
    template <typename land_type_, typename predicate_type_>
    expected<search_result_t> scan_maxsim_(std::span<std::size_t const> query_offsets, vectors_view_t query_rows,
                                           std::span<vector_norms_t const> query_norms, executor_t& executor,
                                           std::size_t wanted, land_type_& land, predicate_type_& predicate) noexcept {
        std::size_t const queries_count = query_offsets.size() - 1, rows_count = size();
        std::size_t const queries_rows = query_rows.extent(0);
        scratch_allocator_t const scratch {executor.arena(0)};
        growing_buffer<key_t, scratch_allocator_t> candidates {scratch};
        growing_buffer<u32_t, scratch_allocator_t> table {scratch};
        if (failed(candidates.resize_uninitialized(rows_count)) ||
            failed(table.resize_uninitialized(ceil2(max_of<std::size_t>(rows_count * 2, 16)))))
            return {{}, status_t::bad_alloc_k};
        search_result_t result;

        // Every key, in placement order.
        auto describe_keys = [&](std::size_t run, u64_t& progress) noexcept -> std::array<extent_t, 1> {
            if (progress++) return {};
            return {run_extent_(locate_run_(0, rows_count, run, key_column_k), key_column_k)};
        };
        auto collect_keys = [&](std::size_t, std::size_t run, std::array<extent_t, 1> const& extents,
                                std::array<byte_t const*, 1> const& landed) noexcept {
            run_t const located = locate_run_(0, rows_count, run, key_column_k);
            nvme_part_t const& part = parts_[extents[0].shard];
            for (std::size_t row = 0; row != located.rows; ++row)
                candidates[located.position + row] = safegraphs_widened_key<key_t>(landed[0] + row * part.key_row_bytes,
                                                                                   part.key_dtype);
        };
        auto const [key_reads, collect_status] = scan_reads_(executor, runs_count_(0, rows_count, key_column_k),
                                                             describe_keys, collect_keys);
        result.scan_reads += key_reads;
        if (failed(collect_status)) return {result, collect_status};

        // An exact table of positions over the first `slots` of `table`, one past each, zero empty.
        std::size_t slots = 0, admitted = 0;
        auto reset_table = [&](std::size_t entries) noexcept {
            slots = ceil2(max_of<std::size_t>(entries * 2, 16));
            std::memset(table.data(), 0, slots * sizeof(u32_t));
        };
        auto find_or_insert = [&](key_t const& key, std::size_t position) noexcept -> std::size_t {
            std::size_t const mask = slots - 1;
            for (std::size_t slot = mix_bits(std::hash<key_t> {}(key)) & mask;; slot = (slot + 1) & mask) {
                if (!table[slot]) {
                    table[slot] = static_cast<u32_t>(position + 1);
                    return position;
                }
                if (same_key_(candidates[table[slot] - 1], key)) return table[slot] - 1;
            }
        };
        auto find = [&](key_t const& key) noexcept -> std::size_t {
            std::size_t const mask = slots - 1;
            for (std::size_t slot = mix_bits(std::hash<key_t> {}(key)) & mask; table[slot]; slot = (slot + 1) & mask)
                if (same_key_(candidates[table[slot] - 1], key)) return table[slot] - 1;
            return admitted;
        };
        reset_table(rows_count);
        std::size_t kept = 0;
        for (std::size_t row = 0; row != rows_count; ++row)
            if (find_or_insert(candidates[row], kept) == kept) candidates[kept++] = candidates[row];

        // The predicate hears each distinct key once, a quad at a time, and the admitted keep their
        // order.
        auto&& admits = thread_predicate(predicate, 0);
        if (status_t const reserved = reserve_predicate(admits, kept); failed(reserved)) return {result, reserved};
        auto keep = [&](scan_payload_t position) noexcept { candidates[admitted++] = candidates[position]; };
        for (std::size_t first = 0; first < kept; first += quad_lanes_k)
            with_static_extent<quad_lanes_k>(min_of(quad_lanes_k, kept - first), [&]<std::size_t extent_>() noexcept {
                std::array<scan_payload_t, extent_> positions;
                for (std::size_t lane = 0; lane != extent_; ++lane)
                    positions[lane] = static_cast<scan_payload_t>(first + lane);
                ask_quad(admits, std::span<key_t const, extent_>(candidates.data() + first, extent_),
                         std::span<scan_payload_t const, extent_>(positions), keep);
            });
        settle_predicate(admits, keep);
        if (!admitted) return {result, status_t::success_k};
        reset_table(admitted);
        for (std::size_t position = 0; position != admitted; ++position) find_or_insert(candidates[position], position);

        growing_buffer<dot_result_t, scratch_allocator_t> folded {scratch};
        if (failed(folded.resize_uninitialized(admitted * queries_rows))) return {result, status_t::bad_alloc_k};
        for (dot_result_t& maximum : folded) maximum = dot_result_t::finite_min();
        nk::matrix_span<dot_result_t> const maxima {folded.data(), admitted, queries_rows};
        // Runs follow the wider of the key and value rows, so both runs of a task fit a buffer.
        column_t const runs_column = parts_[0].value_row_bytes >= parts_[0].key_row_bytes ? value_column_k
                                                                                          : key_column_k;
        auto describe_runs = [&](std::size_t run, u64_t& progress) noexcept -> std::array<extent_t, 2> {
            if (progress++) return {};
            run_t const located = locate_run_(0, rows_count, run, runs_column);
            return {run_extent_(located, key_column_k), run_extent_(located, value_column_k)};
        };
        auto fold_runs = [&](std::size_t, std::size_t, std::array<extent_t, 2> const& extents,
                             std::array<byte_t const*, 2> const& landed) noexcept {
            nvme_part_t const& part = parts_[extents[1].shard];
            std::size_t const run_rows = static_cast<std::size_t>(extents[1].bytes / part.value_row_bytes);
            for (std::size_t row = 0; row != run_rows; ++row) {
                std::size_t const position = find(
                    safegraphs_widened_key<key_t>(landed[0] + row * part.key_row_bytes, part.key_dtype));
                if (position == admitted) continue;
                vector_view_t const values {reinterpret_cast<scalar_t const*>(landed[1] + row * part.value_row_bytes),
                                            part.dimensions};
                fold_row_(
                    values, query_rows, query_norms, to_span(maxima.row(position).as_vector()),
                    [](dot_result_t& maximum, dot_result_t dot) noexcept {
                        std::atomic_ref<dot_result_t> const shared {maximum};
                        dot_result_t seen = shared.load(std::memory_order_relaxed);
                        while (seen < dot && !shared.compare_exchange_weak(seen, dot, std::memory_order_relaxed)) {}
                    });
            }
        };
        auto const [run_reads, fold_status] = scan_reads_<2>(executor, runs_count_(0, rows_count, runs_column),
                                                             describe_runs, fold_runs);
        result.scan_reads += run_reads;
        result.computed_distances += rows_count * queries_rows;
        if (failed(fold_status)) return {result, fold_status};

        // Without a bound, every thread's ranking holds every admitted key.
        std::size_t const capacity = wanted ? wanted : admitted;
        growing_buffer<maxsim_ranked<total_t>, scratch_allocator_t> ranked {scratch};
        if (failed(ranked.resize_uninitialized(executor.size() * capacity))) return {result, status_t::bad_alloc_k};
        nk::matrix_span<maxsim_ranked<total_t>> const rankings {ranked.data(), executor.size(), capacity};
        status_t landing = status_t::success_k;
        status_t const ranking = executor.for_tasks(
            queries_count, [&](std::size_t query, std::size_t thread, std::size_t) noexcept {
                std::size_t const first = query_offsets[query] - query_offsets.front();
                std::size_t const rows = query_offsets[query + 1] - query_offsets[query];
                if (status_t const landed = rank_candidates_(
                        to_span(candidates).first(admitted), maxima(nk::all, nk::range(first, first + rows), nk::slice),
                        to_span(rankings.row(thread).as_vector()), thread, query, land);
                    failed(landed))
                    std::atomic_ref<status_t>(landing).store(landed, std::memory_order_relaxed);
            });
        return {result, failed(ranking) ? ranking : landing};
    }

    /** Ranks @p candidates by the sums of their rows of @p maxima, largest first and ties by
     *  position, keeping the best in @p ranked, and hands as many as it holds to @p land as query
     *  @p query of thread @p thread; answers the first refusal @p land gives. */
    template <typename land_type_>
    static status_t rank_candidates_(std::span<key_t const> candidates, nk::matrix_view<dot_result_t> maxima,
                                     std::span<maxsim_ranked<total_t>> ranked, std::size_t thread, std::size_t query,
                                     land_type_& land) noexcept {
        auto better = [](maxsim_ranked<total_t> const& first, maxsim_ranked<total_t> const& second) noexcept {
            return first.total != second.total ? first.total > second.total : first.candidate < second.candidate;
        };
        std::size_t kept = 0, worst = 0;
        for (std::size_t candidate = 0; candidate != candidates.size(); ++candidate) {
            total_t total {};
            for (dot_result_t const maximum : to_span(maxima.row(candidate).as_vector())) total += total_t(maximum);
            maxsim_ranked<total_t> const entry {static_cast<u32_t>(candidate), total};
            if (kept == ranked.size() && !better(entry, ranked[worst])) continue;
            ranked[kept == ranked.size() ? worst : kept++] = entry;
            if (kept != ranked.size()) continue;
            worst = 0;
            for (std::size_t slot = 1; slot != kept; ++slot)
                if (better(ranked[worst], ranked[slot])) worst = slot;
        }
        partial_sort(ranked.first(kept), kept, better);
        for (std::size_t rank = 0; rank != kept; ++rank)
            if (status_t const landed = land(thread, query, candidates[ranked[rank].candidate], ranked[rank].total);
                failed(landed))
                return landed;
        return status_t::success_k;
    }

    /** Scores stored @p row against @p query_rows, as many as @p maxima holds, a quad of query rows
     *  a call, and hands @p update each maximum beside its dot. */
    template <typename update_type_>
    void fold_row_(vector_view_t row, vectors_view_t query_rows, std::span<vector_norms_t const> query_norms,
                   std::span<dot_result_t> maxima, update_type_&& update) const noexcept {
        vector_norms_t const row_norms = index_t::norms_(kernel_, row);
        for (std::size_t offset = 0; offset < maxima.size(); offset += quad_lanes_k) {
            std::size_t const lanes = min_of(quad_lanes_k, maxima.size() - offset);
            quad<vector_view_t> lane_rows;
            quad<vector_norms_t> lane_norms;
            for (std::size_t lane = 0; lane != quad_lanes_k; ++lane) {
                std::size_t const source = offset + min_of(lane, lanes - 1);
                lane_rows[lane] = query_rows.row(source).as_vector();
                lane_norms[lane] = query_norms[source];
            }
            scores_result<dot_result_t> const scored = dense::scores<quads_kit_t>(kernel_, row, row_norms, lane_rows,
                                                                                  lane_norms);
            for (std::size_t lane = 0; lane != lanes; ++lane) update(maxima[offset + lane], scored.dots[lane]);
        }
    }

#pragma endregion MaxSim

#pragma region Reachability

    /** One part's sweep for @ref health over @p current and @p next, each a bit per row at least. A
     *  row's two bits are its state: 00 unreached, 10 in this level, 01 in the next one, 11
     *  expanded. */
    expected<health_result_t> audit_part_(u32_t part_index, std::span<u64_t> current, std::span<u64_t> next,
                                          executor_t& executor) const noexcept {
        nvme_part_t const& part = parts_[part_index];
        std::size_t const nodes = part.nodes, words = divide_round_up<std::size_t>(nodes, 64);
        std::size_t const row_bytes = std::size_t {part.connectivity} * sizeof(node_id_t);
        std::size_t const run_rows = scan_run_rows_(part_index, row_bytes), tasks = divide_round_up(nodes, run_rows);
        u64_t const block = block_bytes_of(media_[part_index]);
        u64_t const column = byte_base_of(media_[part_index]) + part.neighbors_begin;
        std::memset(current.data(), 0, words * sizeof(u64_t));
        std::memset(next.data(), 0, words * sizeof(u64_t));
        std::size_t const seeds = min_of<std::size_t>(part.connectivity, nodes);
        for (std::size_t seed = 0; seed != seeds; ++seed) {
            std::size_t const node = seed * nodes / seeds;
            current[node / 64] |= u64_t {1} << (node % 64);
        }

        // Within a level only `next` changes, through relaxed atomics, as threads expand their rows
        // and reach others.
        auto later = [&](std::size_t word) noexcept {
            return std::atomic_ref<u64_t>(next[word]).load(std::memory_order_relaxed);
        };
        auto first_in_level = [&](std::size_t from, std::size_t bound) noexcept {
            while (from < bound) {
                std::size_t const word = from / 64;
                u64_t const lanes = (current[word] & ~later(word)) >> (from % 64);
                if (lanes) return min_of<std::size_t>(from + std::countr_zero(lanes), bound);
                from = (word + 1) * 64;
            }
            return bound;
        };
        auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<extent_t, 1> {
            std::size_t const task_first = task * run_rows, bound = min_of(task_first + run_rows, nodes);
            std::size_t const first = first_in_level(task_first + progress, bound);
            if (first == bound) return {};
            // Rows join while their blocks touch the read's, so it covers no block that a read per
            // row would not.
            std::size_t last = first;
            for (std::size_t row = first_in_level(first + 1, bound); row != bound;
                 row = first_in_level(row + 1, bound)) {
                if ((column + row * row_bytes) / block > (column + (last + 1) * row_bytes - 1) / block + 1) break;
                last = row;
            }
            progress = last + 1 - task_first;
            return {extent_t {part_index, part.neighbors_begin + first * row_bytes, (last + 1 - first) * row_bytes}};
        };
        auto consume = [&](std::size_t, std::size_t, auto const& extents, auto const& landed) noexcept {
            extent_t const& extent = extents[0];
            byte_t const* const bytes = landed[0];
            std::size_t const first = static_cast<std::size_t>((extent.begin - part.neighbors_begin) / row_bytes);
            for (std::size_t row = 0; row != extent.bytes / row_bytes; ++row) {
                std::size_t const node = first + row;
                u64_t const lane = u64_t {1} << (node % 64);
                if (!(current[node / 64] & ~later(node / 64) & lane)) continue;
                std::atomic_ref<u64_t>(next[node / 64]).fetch_or(lane, std::memory_order_relaxed);
                auto const* const neighbors = reinterpret_cast<node_id_t const*>(bytes + row * row_bytes);
                for (std::size_t slot = 0; slot != part.connectivity; ++slot) {
                    node_id_t const neighbor = neighbors[slot];
                    u64_t const reached = u64_t {1} << (neighbor % 64);
                    if (neighbor >= nodes || (current[neighbor / 64] | later(neighbor / 64)) & reached) continue;
                    std::atomic_ref<u64_t>(next[neighbor / 64]).fetch_or(reached, std::memory_order_relaxed);
                }
            }
        };

        health_result_t audited;
        audited.audited_nodes = nodes;
        std::size_t level_rows = seeds;
        while (level_rows) {
            if (status_t const swept = scan_reads_(executor, tasks, describe, consume).status; failed(swept))
                return {{}, swept};
            ++audited.sweeps;
            level_rows = 0;
            for (std::size_t word = 0; word != words; ++word) {
                u64_t const was = current[word], reached = next[word];
                level_rows += static_cast<std::size_t>(std::popcount(reached & ~was));
                current[word] = was | reached;
                next[word] = was & reached;
            }
        }
        for (std::size_t word = 0; word != words; ++word)
            audited.reachable_nodes += static_cast<std::size_t>(std::popcount(current[word]));
        return {audited, status_t::success_k};
    }

#pragma endregion Reachability

#pragma region Opening

    /** The body of @ref view, into an empty catalog. */
    status_t view_(std::string_view checkpoint, std::span<std::string_view const> paths) noexcept {
        if (checkpoint.empty()) return status_t::invalid_argument_k;
        std::array<byte_t, safegraphs_prefix_t::magic_k.size()> magic {};
        if (names_transport_id(checkpoint))
            std::memcpy(magic.data(), safegraphs_prefix_t::magic_k.data(), magic.size());
        else {
            auto [file, open_status] = seekable_file_t::open(checkpoint, file_mode_t::read_k, allocator_);
            if (failed(open_status)) return open_status;
            [[maybe_unused]] status_t const sniffed = file.read_at(0, magic);
        }
        // The magic or a transport ID makes a lone monolith, anything else a manifest naming its
        // parts beside it.
        growing_buffer<char, forkunion_allocator_t> names {allocator_};
        growing_buffer<iota_view<std::size_t>, forkunion_allocator_t> entries {allocator_};
        if (std::memcmp(magic.data(), safegraphs_prefix_t::magic_k.data(), magic.size()) == 0) {
            if (failed(entries.push_back(iota_view<std::size_t> {0, 0}))) return status_t::bad_alloc_k;
        }
        else {
            status_t kept = status_t::success_k;
            auto per_part = [&](safegraphs_part_t entry) noexcept {
                std::size_t const first = names.size();
                for (char const character : entry.path)
                    if (failed(names.push_back(character))) kept = status_t::bad_alloc_k;
                if (failed(entries.push_back(iota_view<std::size_t> {first, names.size()})))
                    kept = status_t::bad_alloc_k;
            };
            auto const [summary, summary_status] = safegraphs_read_manifest(checkpoint, allocator_, per_part);
            if (failed(summary_status)) return summary_status;
            if (failed(kept)) return kept;
            if (entries.empty()) return status_t::malformed_k;
        }
        if (!paths.empty() && paths.size() != entries.size()) return status_t::config_mismatch_k;
        if (failed(media_.reserve_exactly(entries.size())) || failed(parts_.reserve_exactly(entries.size())))
            return status_t::bad_alloc_k;

        std::string_view const checkpoint_name = safegraphs_basename(checkpoint);
        std::string_view const checkpoint_directory = checkpoint.substr(0, checkpoint.size() - checkpoint_name.size());
        growing_buffer<char, forkunion_allocator_t> resolved {allocator_};
        for (std::size_t part = 0; part != entries.size(); ++part) {
            std::string_view const relative = entries[part].empty()
                                                  ? checkpoint_name
                                                  : std::string_view {names.data() + entries[part].first,
                                                                      entries[part].size()};
            status_t joined = status_t::success_k;
            if (paths.empty())
                joined = entries[part].empty() ? nvme_join_path_(resolved, {}, checkpoint)
                                               : nvme_join_path_(resolved, checkpoint_directory, relative);
            else {
                // The directory test runs on the terminated copy, which the OS reads.
                joined = nvme_join_path_(resolved, {}, paths[part]);
                if (succeeded(joined) && nvme_is_directory_(resolved.data()))
                    joined = nvme_join_path_(resolved, paths[part], safegraphs_basename(relative));
            }
            if (failed(joined)) return joined;

            medium_t medium;
            nvme_part_t opened;
            std::string_view const resolved_path {resolved.data(), resolved.size() - 1};
            if (status_t const vetted = index_t::open_part_(resolved_path, safegraphs_key_dtype<key_t>(), kernel_,
                                                            allocator_, medium, opened);
                failed(vetted))
                return vetted;
            if (part && (opened.dimensions != parts_[0].dimensions || opened.connectivity != parts_[0].connectivity ||
                         opened.value_row_bytes != parts_[0].value_row_bytes))
                return status_t::config_mismatch_k;
            media_.push_back_reserved(std::move(medium));
            parts_.push_back_reserved(opened);
        }
        if (!kernel_supported<typename index_t::quads_kit_t>(kernel_)) return status_t::unknown_kind_k;
        return status_t::success_k;
    }

#pragma endregion Opening

    /** Every part's file or device, open for the engines, in part order. */
    growing_buffer<medium_t, forkunion_allocator_t> media_;

    /** Where every part keeps its columns, parallel to @c media_. */
    growing_buffer<nvme_part_t, forkunion_allocator_t> parts_;

    /** The host allocator behind the catalog and the paths a view resolves. */
    forkunion_allocator_t allocator_;

    /** The distance kind the parts name, every walk ranks by, and every rewrite carries on. */
    nk_kernel_kind_t kernel_ = nk_kernel_unknown_k;
};

/**
 *  @brief External reader's cursor over the rows of one @c nvme_indexes: keys and vectors in
 *      placement order, part by part, each read scanning whole runs of the key and value columns
 *      through an executor.
 *
 *  Pure position: every read names the index it reads, which must be the one the cursor was opened
 *  on, so the cursor keeps no reference that could outlive it. The parts are frozen, so one row
 *  counter is the whole position.
 */
template <typename indexes_type_>
struct nvme_indexes_cursor {

    using indexes_t = indexes_type_;
    using key_t = typename indexes_t::key_t;
    using scalar_t = typename indexes_t::scalar_t;
    using executor_t = typename indexes_t::executor_t;
    using search_result_t = typename indexes_t::search_result_t;

    /** Starts at the first row of @p indexes, the one every later @ref read must name. */
    explicit nvme_indexes_cursor(indexes_t const&) noexcept {}

    /** Copies up to @p keys.size() rows of @p indexes into @p keys, and exports their vectors into
     *  @p vectors in its scalar unless it holds no rows, through @p executor, advancing past them;
     *  the result's @c count is the rows written and its @c scan_reads what they cost. An export
     *  wanting vectors must offer at least as many rows as @p keys, each at least as wide as the
     *  index, or nothing is read and it answers @c config_mismatch_k. */
    expected<search_result_t> read(indexes_t const& indexes, std::span<key_t> keys, rows_span_t vectors,
                                   executor_t& executor) noexcept {
        std::size_t const dimensions = indexes.dimensions();
        if (vectors.rows && (vectors.rows < keys.size() || vectors.columns < dimensions))
            return {{}, status_t::config_mismatch_k};
        std::size_t const left = indexes.size() - min_of(row_, indexes.size());
        if (!left || keys.empty()) return {{}, status_t::success_k};
        rows_span_t const target {vectors.data, vectors.rows, dimensions, vectors.stride, vectors.dtype};
        expected<search_result_t> copied = indexes.read_rows_(row_, keys.first(min_of(keys.size(), left)), target,
                                                              executor);
        if (copied) row_ += copied.value.count;
        return copied;
    }

  private:

    /** The next row to read, in placement order across the parts. */
    std::size_t row_ = 0;
};

#pragma region Rewrites

/** What the current phase's steps do, declared in the order every rewrite runs them, which
 *  @ref nvme_rewrite_plan::advance steps through one at a time, skipping phases with no
 *  steps; @c finish_k once it answered false. */
enum class nvme_rewrite_phase_t : u8_t {

    /** Moves keys and values into the target in target order, with their norms and the neighbors it
     *  keeps. */
    copy_k,

    /** Builds one block of target rows in a RAM index and writes their neighbors. */
    build_k,

    /** Re-selects target rows: the appended part, repaired rows, or the refined range. */
    reselect_k,

    /** Applies one bucket of spilled back-links. */
    back_link_k,

    /** Counts every row's in-links, then links the rows nothing links to back in. */
    heal_k,

    /** Nothing left but @ref nvme_rewrite_plan::finish. */
    finish_k,
};

/** What every rewrite shares; zeros derive from the geometry. The target is always a manifest
 *  beside its parts. */
struct nvme_rewrite_config_t {

    /** Breadth of every re-selection walk and block build; zero means twice the connectivity. */
    u32_t expansion_factor = 0;

    /** The target parts' buffer alignment: a power of two, at least the floor. */
    std::size_t alignment = safegraphs_alignment_default_k;

    /** Free-form `__metadata__` strings, borrowed until the plan finishes. */
    std::span<safegraphs_custom_field_t const> provenance = {};
};

/** Outcome of a merge, split, refine or build. */
struct [[nodiscard]] nvme_rewrite_result_t {

    /** Rows the target holds. */
    std::size_t rows = 0;

    /** Parts the target holds. */
    std::size_t parts = 0;

    /** Edges dropped because their ends landed in different target parts; zero for merge, refine
     *  and build. */
    std::size_t cut_edges = 0;

    /** The re-selections, the block builds and the heal, counted as a RAM refine counts them, but
     *  for the heal of the target: its @c islands_found are the rows nothing links to, and no sweep
     *  runs on it, so @c reachable_nodes and @c sweeps stay zero; @c health on the committed view
     *  measures them. */
    refine_result_t graph;

    /** Neighbors rows the walks and the heal read. */
    nvme_reads_t neighbor_reads;

    /** Value rows the walks, the selections and the back-links read. */
    nvme_reads_t value_reads;

    /** Runs of whole rows the copies, the batches and the heal scanned, and the spills read
     *  back. */
    nvme_reads_t scan_reads;

    /** Bytes the target parts and the spills received. */
    std::size_t written_bytes = 0;
};

/** Cuts every source part into consecutive ranges of at most @c rows rows, each a target part. */
struct split_by_rows_t {
    std::size_t rows = 0;
};

/** Cuts at ascending @c fences: a row lands in the first part whose fence is above its key, the
 *  last past all of them; empty ranges make no part. At most 255 fences, since every row's range
 *  is routed as one byte. */
template <typename key_type_>
struct split_by_keys {
    std::span<key_type_ const> fences;
};

/** Hands out cache-line-aligned slices of one lent workspace, shared by pointer across rebound
 *  copies; releasing does nothing, so a RAM index built over it lives exactly inside the
 *  workspace. */
template <typename value_type_ = byte_t>
struct nvme_workspace_allocator {
    using value_type = value_type_;

    /** The workspace not handed out yet; null refuses every request. */
    bytes_span_t* remaining = nullptr;

    constexpr nvme_workspace_allocator() noexcept = default;
    explicit constexpr nvme_workspace_allocator(bytes_span_t& remaining) noexcept : remaining(&remaining) {}

    template <typename other_type_>
    constexpr nvme_workspace_allocator(nvme_workspace_allocator<other_type_> const& other) noexcept
        : remaining(other.remaining) {}

    value_type_* allocate(std::size_t count) noexcept {
        if (!remaining) return nullptr;
        std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(remaining->data());
        std::size_t const padding = static_cast<std::size_t>(
            round_up_to_multiple<std::uintptr_t>(address, default_alignment_k) - address);
        std::size_t const bytes = count * sizeof(value_type_);
        if (padding > remaining->size() || bytes > remaining->size() - padding) return nullptr;
        auto* const slice = reinterpret_cast<value_type_*>(remaining->data() + padding);
        *remaining = remaining->subspan(padding + bytes);
        return slice;
    }
    void deallocate(value_type_*, std::size_t) noexcept {}
    bool operator==(nvme_workspace_allocator const& other) const noexcept { return remaining == other.remaining; }
};

template <typename indexes_type_>
struct nvme_rewrite_plan;

/** Merges @p sources into one target part: the largest part stays resident and keeps its IDs, every
 *  other part lands after it with its edges rebased, its rows walk the target and link in, their
 *  back-links are spilled and merged, then the target heals. The sources are borrowed until the
 *  plan dies, and the target names none of their files. */
template <typename indexes_type_>
expected<nvme_rewrite_plan<indexes_type_>> plan_merge(std::span<indexes_type_ const* const> sources,
                                                      std::string_view target, bytes_span_t workspace,
                                                      nvme_rewrite_config_t const& config = {}) noexcept;

/** Cuts @p source as @p policy says, drops and counts the edges the cut crosses, and repairs every
 *  target part: in RAM when it fits the workspace, otherwise by walks, back-links and a heal. The
 *  source and the policy's fences are borrowed until the plan dies. A split by keys routes while
 *  it plans, scanning every source part's key column over @p executor. */
template <typename indexes_type_, typename split_type_>
expected<nvme_rewrite_plan<indexes_type_>> plan_split(indexes_type_ const& source, split_type_ const& policy,
                                                      std::string_view target,
                                                      typename indexes_type_::executor_t& executor,
                                                      bytes_span_t workspace,
                                                      nvme_rewrite_config_t const& config = {}) noexcept;

/** Copies @p source and re-selects @p rows of it, in placement order across its parts, by walks of
 *  the copy, or in RAM for a whole part the workspace holds, then merges the back-links and heals;
 *  empty rows heal alone. The source is borrowed until the plan dies. */
template <typename indexes_type_>
expected<nvme_rewrite_plan<indexes_type_>> plan_refine(indexes_type_ const& source, iota_view<std::size_t> rows,
                                                       std::string_view target, bytes_span_t workspace,
                                                       nvme_rewrite_config_t const& config = {}) noexcept;

/** Builds one graph over every row of the parts at @p sources, their neighbors ignored, in
 *  contiguous blocks the workspace holds as a RAM index, each healed there before it is written,
 *  then links the later blocks across as a merge does. */
template <typename indexes_type_>
expected<nvme_rewrite_plan<indexes_type_>> plan_build(std::span<std::string_view const> sources,
                                                      std::string_view target, bytes_span_t workspace,
                                                      nvme_rewrite_config_t const& config = {}) noexcept;

/**
 *  @brief One rewrite in flight: phases of steps, each step run across every thread of an executor.
 *
 *  Steps of a phase may run in any order, but one at a time, since each claims the executor while
 *  it runs, so searches on the same executor run between steps; a failed step may run again.
 *  @ref advance is the single-caller barrier between phases. A dying plan removes its spills and
 *  every target part no manifest committed, so a stopped rewrite reads as absent.
 */
template <typename indexes_type_>
struct [[nodiscard]] nvme_rewrite_plan {
    using indexes_t = indexes_type_;
    using executor_t = typename indexes_t::executor_t;
    using key_t = typename indexes_t::key_t;
    using result_t = nvme_rewrite_result_t;

    /** One unit of the current phase. */
    struct step_t {

        /** The step's position in its phase. */
        std::size_t position = 0;

        /** The source part a route or copy step reads, or the target part every other step
         *  writes. */
        std::size_t part = 0;

        /** The node IDs it covers: source ones for a route or copy step, target ones otherwise. */
        iota_view<node_id_t> nodes;
    };

    nvme_rewrite_plan() noexcept = default;
    nvme_rewrite_plan(nvme_rewrite_plan&&) noexcept = default;
    nvme_rewrite_plan& operator=(nvme_rewrite_plan&&) = delete;
    nvme_rewrite_plan(nvme_rewrite_plan const&) = delete;
    nvme_rewrite_plan& operator=(nvme_rewrite_plan const&) = delete;

    /** Removes the spills, and the target parts unless @ref finish committed them. */
    ~nvme_rewrite_plan() noexcept {
        drop_spills_();
        for (target_t& target : targets_) {
            [[maybe_unused]] status_t const closed = target.file.close();
            target.view = indexes_t {allocator_};
            std::remove(target.path.data());
        }
    }

    /** Step @p position of the current phase; an empty one past the last. */
    step_t operator[](std::size_t position) const noexcept {
        for (std::size_t unit = 0, first = 0; unit != units_(); ++unit) {
            cut_t const cut = cut_(unit);
            std::size_t const count = cut.steps();
            if (position - first < count) {
                iota_view<std::size_t> const rows = cut.rows_of(position - first);
                return {position, unit, {static_cast<node_id_t>(rows.first), static_cast<node_id_t>(rows.bound)}};
            }
            first += count;
        }
        return {};
    }

    /** The current phase's steps, positionally; none once the plan is done. */
    indexed_range<nvme_rewrite_plan> steps() const noexcept {
        std::size_t count = 0;
        for (std::size_t unit = 0; unit != units_(); ++unit) count += cut_(unit).steps();
        return {this, count};
    }

    /** The phase whose steps run now. */
    nvme_rewrite_phase_t phase() const noexcept { return phase_; }

    /** Runs @p step of the current phase across every thread of @p executor, which it claims while
     *  it runs. */
    status_t run(step_t const& step, executor_t& executor) noexcept {
        if (step.position >= steps().size()) return status_t::config_mismatch_k;
        switch (phase_) {
        case nvme_rewrite_phase_t::copy_k: return copy_(step, executor);
        case nvme_rewrite_phase_t::build_k: return build_(step, executor);
        case nvme_rewrite_phase_t::reselect_k: return reselect_(step, executor);
        case nvme_rewrite_phase_t::back_link_k: return back_link_(step, executor);
        case nvme_rewrite_phase_t::heal_k: return heal_(step, executor);
        default: return status_t::uninitialized_k;
        }
    }

    /** The barrier between phases, one caller after every step ran: enters the next phase with
     *  steps and answers whether one follows. */
    bool advance() noexcept {
        while (phase_ != nvme_rewrite_phase_t::finish_k) {
            phase_ = static_cast<nvme_rewrite_phase_t>(static_cast<u8_t>(phase_) + 1);
            if (!steps().empty()) return true;
        }
        return false;
    }

    /** Writes every part header, then the manifest: the commit. @c cancelled_k before the last
     *  phase ended. */
    expected<result_t> finish() noexcept {
        if (targets_.empty()) return {result_, status_t::uninitialized_k};
        if (phase_ != nvme_rewrite_phase_t::finish_k) return {result_, status_t::cancelled_k};
        std::span<safegraphs_custom_field_t const> const fields = to_span(fields_);
        std::size_t rows = 0;
        for (target_t& target : targets_) {
            std::size_t const bytes = target.header.monolith_bytes();
            if (status_t const framed = safegraphs_write_header(file_span_t {&target.file, 0, bytes}, target.header,
                                                                fields);
                failed(framed))
                return {result_, framed};
            if (status_t const flushed = target.file.sync(); failed(flushed)) return {result_, flushed};
            rows += target.view.size();
        }
        auto part_at = [&](std::size_t part) noexcept {
            target_t const& target = targets_[part];
            return safegraphs_part_t {safegraphs_basename(path_of_(target.path)), target.header.monolith_bytes(),
                                      target.view.size()};
        };
        if (status_t const committed = safegraphs_write_manifest(path_of_(manifest_), targets_.size(), part_at);
            failed(committed))
            return {result_, committed};
        result_.rows = rows;
        result_.parts = targets_.size();
        drop_spills_();
        // Committed parts outlive the plan, so its destructor finds none to remove.
        targets_.clear();
        return {result_, status_t::success_k};
    }

  private:
    template <typename other_type_>
    friend expected<nvme_rewrite_plan<other_type_>> plan_merge(std::span<other_type_ const* const>, std::string_view,
                                                               bytes_span_t, nvme_rewrite_config_t const&) noexcept;
    template <typename other_type_, typename split_type_>
    friend expected<nvme_rewrite_plan<other_type_>> plan_split(other_type_ const&, split_type_ const&, std::string_view,
                                                               typename other_type_::executor_t&, bytes_span_t,
                                                               nvme_rewrite_config_t const&) noexcept;
    template <typename other_type_>
    friend expected<nvme_rewrite_plan<other_type_>> plan_refine(other_type_ const&, iota_view<std::size_t>,
                                                                std::string_view, bytes_span_t,
                                                                nvme_rewrite_config_t const&) noexcept;
    template <typename other_type_>
    friend expected<nvme_rewrite_plan<other_type_>> plan_build(std::span<std::string_view const>, std::string_view,
                                                               bytes_span_t, nvme_rewrite_config_t const&) noexcept;

    using node_id_t = dense::node_id_t;
    using scalar_t = typename indexes_t::scalar_t;
    using dot_result_t = typename indexes_t::dot_result_t;
    using vector_view_t = typename indexes_t::vector_view_t;
    using vectors_view_t = typename indexes_t::vectors_view_t;
    using vector_norms_t = typename indexes_t::vector_norms_t;
    using search_config_t = typename indexes_t::search_config_t;
    using medium_t = typename indexes_t::medium_t;
    using extent_t = typename indexes_t::extent_t;
    using walk_t = typename indexes_t::walk_t;
    using results_array_t = typename indexes_t::results_array_t;
    using scratch_allocator_t = typename indexes_t::scratch_allocator_t;
    using paths_t = growing_buffer<char, forkunion_allocator_t>;

    /** The RAM index a block build or a whole-part refine places inside the workspace. */
    using ram_index_t = ram_index<typename indexes_t::quads_kit_t, typename indexes_t::tiles_kit_t,
                                  typename indexes_t::elementwise_kit_t, typename indexes_t::rmscast_kit_t,
                                  typename indexes_t::visits_kit_t, fu::standard_yield_t, fu::standard_atomic_ref,
                                  dummy_cache_hints_t, nvme_workspace_allocator<byte_t>,
                                  nvme_workspace_allocator<byte_t>, arena_allocator<byte_t, forkunion_allocator_t>>;
    using traversal_context_t = typename ram_index_t::traversal_context_t;
    using traversals_t = growing_buffer<traversal_context_t, scratch_allocator_t>;

    /** Ends the life of a RAM index built inside the workspace, which keeps its bytes. */
    struct destroy_in_place_t {
        void operator()(ram_index_t* index) const noexcept { std::destroy_at(index); }
    };
    using ram_index_ptr_t = std::unique_ptr<ram_index_t, destroy_in_place_t>;
    using ram_cursor_t =
        ram_cursor<ram_index_t, synchronization_t::immutable_k, arena_allocator<byte_t, forkunion_allocator_t>>;

    inline static constexpr node_id_t sentinel_k = node_id_unknown_k;
    static_assert(sizeof(vector_norms_t) == 2 * sizeof(dot_result_t), "A norms row is a sum and a square");

    /** Key ranges a split by keys routes at most: every row's range is tagged with one byte. */
    inline static constexpr std::size_t max_key_ranges_k = std::size_t {std::numeric_limits<u8_t>::max()} + 1;

    /** Rows of the key routing table one directory entry spans: a cache line of tags, so a rank
     *  counts within one line. */
    inline static constexpr std::size_t route_block_k = default_alignment_k / sizeof(u8_t);

    /** Slices a re-select step carves off the workspace, every carve padding its slice by less than
     *  one alignment, so each budget sets one alignment aside per slice. */
    inline static constexpr std::size_t reselect_slices_k = 11;

    /** Slices a back-link step carves before its batches. */
    inline static constexpr std::size_t back_link_slices_k = 2;

    /** Slices each batch of a back-link step carves. */
    inline static constexpr std::size_t back_link_batch_slices_k = 9;

    /** Slices a heal carves: the in-link counts, a target row and the landings. */
    inline static constexpr std::size_t heal_slices_k = 3;

    /** Slices a block build carves beside its RAM index. */
    inline static constexpr std::size_t build_slices_k = 4;

    /** Slices a whole-part refine carves beside its RAM index. */
    inline static constexpr std::size_t refine_slices_k = 1;

    /** One part of a source, flattened across the sources: where its rows land and which of its
     *  edges survive. */
    struct source_t {

        /** The view that reads it. */
        indexes_t const* view = nullptr;

        /** The part inside that view. */
        u32_t part = 0;

        /** Rows the part holds. */
        std::size_t rows = 0;

        /** The target part its first range lands in; a split by keys routes per row instead. */
        std::size_t target = 0;

        /** The target row its row zero lands at. */
        std::size_t base = 0;

        /** Rows per target part, each next range landing in the next target part. */
        std::size_t stride = 0;

        /** The source rows whose edges may survive; empty writes no edges at all. */
        iota_view<std::size_t> kept;
    };

    /** One target part: its file, its header, and a one-part view the walks and scans read it
     *  through. */
    struct target_t {
        paths_t path;
        seekable_file_t file;
        safegraphs_header_t header;
        indexes_t view;
    };

    /** One bucket of spilled back-links: the target rows it covers are fixed by its index. */
    struct bucket_t {
        seekable_file_t file;
        std::size_t bytes = 0;
    };

    /** One back-link a selection spilled: @c source picked @c target at @c distance. */
    struct proposal_t {
        node_id_t target = sentinel_k;
        node_id_t source = sentinel_k;
        f32_t distance = 0;
    };
    static_assert(sizeof(proposal_t) == 12, "Spills are read back as packed records");

    /** Every row's candidates of one batch: the best rows by exact distance, with their bytes
     *  when the batch keeps them, so a selection reads nothing again. */
    struct candidates_t {
        scored_node_t* entries = nullptr;
        byte_t* vectors = nullptr;
        u32_t* counts = nullptr;
        std::size_t slots = 0;
        std::size_t row_bytes = 0;

        scored_node_t* entries_of(std::size_t row) const noexcept { return entries + row * slots; }
        byte_t* vector_of(std::size_t row, std::size_t slot) const noexcept {
            return vectors + (row * slots + slot) * row_bytes;
        }

        /** Keeps @p candidate among the candidates of @p row, at most @p limit long, unless it is
         *  there already or every kept one is closer; answers the slot it took, or @p limit. */
        std::size_t offer(std::size_t row, scored_node_t candidate, std::size_t limit) const noexcept {
            scored_node_t* const kept = entries_of(row);
            std::size_t const count = counts[row];
            std::size_t worst = 0;
            for (std::size_t slot = 0; slot != count; ++slot) {
                if (kept[slot].node_id == candidate.node_id) return limit;
                if (kept[slot].distance > kept[worst].distance) worst = slot;
            }
            std::size_t slot = count;
            if (count >= limit) {
                if (!count || !(candidate.distance < kept[worst].distance)) return limit;
                slot = worst;
            }
            else ++counts[row];
            kept[slot] = candidate;
            return slot;
        }
    };

    /** The rows of a re-select batch: every scored row of every walk is offered to the row's
     *  candidates with its bytes, the row itself excepted. */
    struct reselect_row_t {
        using key_t = typename indexes_t::key_t;
        inline static constexpr nvme_key_reads_t key_reads_k = nvme_key_reads_t::outputs_k;
        inline static constexpr bool admitting_k = false;

        /** Offers every scored lane of a quad to the row's candidates with the bytes its rescoring
         *  saw. */
        struct fold_t {
            reselect_row_t* row = nullptr;

            walk_demand_t absorb(std::span<scored_node_t const> scored, std::span<dot_result_t const>) const noexcept {
                inputs_t const& inputs = row->inputs;
                candidates_t const& candidates = inputs.candidates;
                for (std::size_t lane = 0; lane != scored.size(); ++lane) {
                    if (scored[lane].node_id == inputs.first + row->row) continue;
                    std::size_t const slot = candidates.offer(row->row, scored[lane], inputs.limit);
                    if (slot != inputs.limit)
                        std::memcpy(candidates.vector_of(row->row, slot), row->landed[lane].byte_data(),
                                    candidates.row_bytes);
                }
                return walk_demand_t::continue_k;
            }
        };

        /** The batch every row of the call reads and fills. */
        struct inputs_t {
            vectors_view_t queries;
            candidates_t candidates;
            node_id_t first = 0;
            std::size_t limit = 0;
        };

        inputs_t const& inputs;
        std::size_t row = 0;
        vector_view_t query;
        vector_norms_t query_norms {};

        /** The quad being scored, between its rescoring and its fold. */
        quad<vector_view_t> landed {};

        reselect_row_t(inputs_t const& inputs, scratch_allocator_t) noexcept : inputs(inputs) {}

        status_t reserve(std::size_t, std::size_t) noexcept { return status_t::success_k; }

        /** Primes @p new_row; its candidates keep what earlier walks of the batch kept. */
        void prime_row(std::size_t new_row) noexcept {
            row = new_row;
            query = inputs.queries.row(row).as_vector();
        }
        void rescore_quad(quad<node_id_t> const&, quad<vector_view_t> const& vectors, quad<vector_norms_t> const&,
                          scores_result<dot_result_t>&, std::size_t, std::span<key_t const>) noexcept {
            landed = vectors;
        }
        fold_t fold_of_walk(walk_t const&) noexcept { return {this}; }
        results_array_t const& stopping_candidates(walk_t const& walk) const noexcept { return walk.results; }
        void finish_walk(walk_t const&) noexcept {}
        std::size_t output_count() const noexcept { return 0; }
        nvme_locator<u32_t> output_locator(std::size_t) const noexcept { return {}; }
        void resolve_output_key(std::size_t, key_t const&) noexcept {}
        status_t finish_row() noexcept { return status_t::success_k; }
    };

    /** One column a scan lands: where its row zero sits in the part, a row's width, and where row
     *  zero lands. */
    struct staged_column_t {
        u64_t begin = 0;
        std::size_t row_bytes = 0;
        byte_t* staging = nullptr;
    };

    /** The per-row scratch of one diverse selection. */
    struct selection_t {
        u32_t* order = nullptr;
        u32_t* kept = nullptr;
        vector_norms_t* kept_norms = nullptr;
    };

#pragma region Opening

    /** The text of @p path without its terminator. */
    static std::string_view path_of_(paths_t const& path) noexcept { return {path.data(), path.size() - 1}; }

    /** Copies @p text into @p path, terminated for the OS. */
    static status_t keep_path_(paths_t& path, std::string_view text) noexcept {
        if (failed(path.resize_uninitialized(text.size() + 1))) return status_t::bad_alloc_k;
        std::memcpy(path.data(), text.data(), text.size());
        path[text.size()] = '\0';
        return status_t::success_k;
    }

    /** Opens the plan's common state: the workspace, the configuration and the header fields
     *  naming the kernel of @p view, and the target's paths, and removes a stale manifest so no
     *  half-written part is ever named. */
    status_t open_(std::string_view target, bytes_span_t workspace, nvme_rewrite_config_t const& config,
                   indexes_t const& view) noexcept {
        save_config_t const save {checkpoint_layout_t::distributed_k, config.alignment, synchronization_t::immutable_k,
                                  config.provenance};
        if (status_t const valid = save.validate(); failed(valid)) return valid;
        if (target.empty()) return status_t::invalid_argument_k;
        if (!view.shards_count()) return status_t::uninitialized_k;
        workspace_ = workspace;
        config_ = config;
        if (failed(fields_.reserve_exactly(1 + config.provenance.size()))) return status_t::bad_alloc_k;
        fields_.push_back_reserved({"metric", nk_kernel_name(view.kernel())});
        for (safegraphs_custom_field_t const& field : config.provenance) fields_.push_back_reserved(field);
        if (failed(keep_path_(manifest_, target)) || failed(keep_path_(stem_, safegraphs_stem(target))))
            return status_t::bad_alloc_k;
        std::remove(manifest_.data());
        return status_t::success_k;
    }

    /** Appends every non-empty part of @p view to the sources, refusing a geometry or a kernel
     *  other than those of the first part of @p first, the view the plan opened with. */
    status_t add_sources_(indexes_t const& view, indexes_t const& first) noexcept {
        nvme_part_t const& shape = first.parts_[0];
        if (view.kernel() != first.kernel()) return status_t::config_mismatch_k;
        for (u32_t part = 0; part != view.parts_.size(); ++part) {
            nvme_part_t const& opened = view.parts_[part];
            if (opened.dimensions != shape.dimensions || opened.connectivity != shape.connectivity ||
                opened.value_row_bytes != shape.value_row_bytes || opened.key_dtype != safegraphs_key_dtype<key_t>())
                return status_t::config_mismatch_k;
            if (!opened.nodes) continue;
            source_t source;
            source.view = &view;
            source.part = part;
            source.rows = opened.nodes;
            source.stride = opened.nodes;
            if (failed(sources_.push_back(source))) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** Creates one target part per entry of @p rows, with its header framed, its file sized and its
     *  one-part view open, and the buckets its back-links spill into. */
    status_t open_targets_(std::span<std::size_t const> rows) noexcept {
        if (failed(targets_.reserve_exactly(rows.size()))) return status_t::bad_alloc_k;
        for (std::size_t part = 0; part != rows.size(); ++part) {
            if (!rows[part] || rows[part] >= node_id_unknown_k) return status_t::overflow_k;
            safegraphs_regular_shape_t const shape {
                rows[part],
                safegraphs_key_dtype<key_t>(),
                dtype_of(scalar_t::dtype()),
                dimensions_(),
                dtype_t::u32_k,
                connectivity_(),
                norms_bytes_() ? dtype_of(dot_result_t::dtype()) : dtype_t::unknown_k};
            auto const [header, frame_status] = safegraphs_regular_header(shape, config_.alignment, to_span(fields_));
            if (failed(frame_status)) return frame_status;
            targets_.push_back_reserved(
                target_t {paths_t {allocator_}, seekable_file_t {}, header, indexes_t {allocator_}});
            target_t& target = targets_.back();
            target.view.kernel_ = kernel_();
            paths_t name {allocator_};
            auto const [path, path_status] = safegraphs_part_path(name, path_of_(stem_), part, rows.size());
            if (failed(path_status)) return path_status;
            if (status_t const kept = keep_path_(target.path, path); failed(kept)) return kept;
            auto [file, open_status] = seekable_file_t::open(path, file_mode_t::write_k, allocator_);
            if (failed(open_status)) return open_status;
            target.file = std::move(file);
            if (status_t const sized = target.file.truncate(header.monolith_bytes()); failed(sized)) return sized;

            auto [file_medium, medium_status] = file_medium_t::open(target.path.data());
            if (failed(medium_status)) return medium_status;
            nvme_part_t viewed;
            viewed.nodes = static_cast<u32_t>(rows[part]);
            viewed.dimensions = static_cast<u32_t>(dimensions_());
            viewed.connectivity = static_cast<u32_t>(connectivity_());
            viewed.key_row_bytes = static_cast<u32_t>(header.key.row_bytes());
            viewed.key_dtype = header.key.dtype;
            viewed.value_row_bytes = static_cast<u32_t>(header.value.row_bytes());
            viewed.keys_begin = header.data_offset + header.key.data.first;
            viewed.values_begin = header.data_offset + header.value.data.first;
            viewed.neighbors_begin = header.data_offset + header.neighbors.data.first;
            if (failed(target.view.media_.push_back(medium_t {std::move(file_medium)})) ||
                failed(target.view.parts_.push_back(viewed)))
                return status_t::bad_alloc_k;
        }
        if (failed(buckets_.resize(bucket_base_(targets_.size())))) return status_t::bad_alloc_k;
        std::size_t widest = 0;
        for (std::size_t const part_rows : rows) widest = max_of(widest, part_rows);
        if (heal_bytes_(widest) > workspace_.size()) return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /** Refuses a workspace too small for a single row of any phase the plan runs. */
    status_t size_steps_() const noexcept {
        // A back-link step's proposals take at most half the workspace, its batches the rest.
        std::size_t const back_link_reserved = workspace_.size() / 2 +
                                               (back_link_slices_k + back_link_batch_slices_k) * default_alignment_k;
        bool const back_links_fit = workspace_.size() >= back_link_reserved + back_link_row_bytes_();
        if (!copy_rows_() || !reselect_rows_() || !bucket_rows_() || !back_links_fit) return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /** The units the current phase cuts into steps: source parts while copying, target parts
     *  after. */
    std::size_t units_() const noexcept {
        switch (phase_) {
        case nvme_rewrite_phase_t::copy_k: return sources_.size();
        case nvme_rewrite_phase_t::finish_k: return 0;
        default: return targets_.size();
        }
    }

    /** Rows of one unit of the current phase: @b [first,first+rows) cut at every @c stride rows
     *  past @c first, and into steps of at most @c chunk rows. */
    struct cut_t {
        std::size_t first = 0;
        std::size_t rows = 0;
        std::size_t stride = 1;
        std::size_t chunk = 1;

        std::size_t steps() const noexcept {
            return rows / stride * divide_round_up(stride, chunk) + divide_round_up(rows % stride, chunk);
        }

        /** The rows of step @p step of the unit. */
        iota_view<std::size_t> rows_of(std::size_t step) const noexcept {
            std::size_t const per_stride = divide_round_up(stride, chunk), range = step / per_stride;
            std::size_t const offset = range * stride + step % per_stride * chunk;
            return {first + offset, first + min_of(offset + chunk, min_of((range + 1) * stride, rows))};
        }
    };

    /** How the current phase cuts unit @p unit: a copy step stays inside one target range of its
     *  source part, a build step is a block, a re-select step a batch or a whole part refined in
     *  RAM, a back-link step a bucket, and a heal step a whole part. */
    cut_t cut_(std::size_t unit) const noexcept {
        switch (phase_) {
        case nvme_rewrite_phase_t::copy_k: {
            source_t const& source = sources_[unit];
            return {0, source.rows, source.stride, copy_rows_()};
        }
        case nvme_rewrite_phase_t::build_k: {
            std::size_t const nodes = targets_[unit].view.size(), block_rows = build_rows_();
            return {0, block_rows ? nodes : 0, nodes, max_of<std::size_t>(block_rows, 1)};
        }
        case nvme_rewrite_phase_t::reselect_k: {
            iota_view<std::size_t> const rows = selected_of_(unit);
            std::size_t const nodes = targets_[unit].view.size();
            // A whole part the workspace holds as one RAM index is refined in RAM, in one step.
            bool const in_ram = rows.size() == nodes && ram_fits_(nodes, neighbors_row_bytes_(), refine_slices_k);
            return {rows.first, rows.size(), max_of<std::size_t>(rows.size(), 1), in_ram ? nodes : reselect_rows_()};
        }
        case nvme_rewrite_phase_t::back_link_k: {
            std::size_t const nodes = targets_[unit].view.size();
            return {0, nodes, nodes, bucket_rows_()};
        }
        case nvme_rewrite_phase_t::heal_k: {
            std::size_t const nodes = targets_[unit].view.size();
            return {0, nodes, nodes, nodes};
        }
        default: return {};
        }
    }

    /** The rows of target part @p part the re-select phase re-selects. */
    iota_view<std::size_t> selected_of_(std::size_t part) const noexcept {
        std::size_t placed = 0;
        for (std::size_t earlier = 0; earlier != part; ++earlier) placed += targets_[earlier].view.size();
        std::size_t const low = max_of(selected_.first, placed);
        std::size_t const high = min_of(selected_.bound, placed + targets_[part].view.size());
        return low < high ? iota_view<std::size_t> {low - placed, high - placed} : iota_view<std::size_t> {};
    }

    /** The first spill bucket of target part @p part, past every bucket of the parts before it. */
    std::size_t bucket_base_(std::size_t part) const noexcept {
        std::size_t base = 0;
        for (std::size_t earlier = 0; earlier != part; ++earlier)
            base += divide_round_up(targets_[earlier].view.size(), bucket_rows_());
        return base;
    }

#pragma endregion Opening

#pragma region Sizes

    /** The first source part, whose geometry every source part and every target part shares. */
    nvme_part_t const& shape_() const noexcept { return sources_[0].view->parts_[sources_[0].part]; }
    std::size_t dimensions_() const noexcept { return shape_().dimensions; }
    std::size_t connectivity_() const noexcept { return shape_().connectivity; }
    std::size_t value_row_bytes_() const noexcept { return shape_().value_row_bytes; }

    /** The distance kind of every source, which the targets name and their selections rank by. */
    nk_kernel_kind_t kernel_() const noexcept { return sources_[0].view->kernel(); }

    /** Bytes of one norms row, zero when the kernel reads no norms and the targets carry no
     *  column. */
    std::size_t norms_bytes_() const noexcept { return sources_[0].view->stores_norms() ? sizeof(vector_norms_t) : 0; }

    std::size_t expansion_factor_() const noexcept {
        return config_.expansion_factor ? config_.expansion_factor : default_expansion_factor(connectivity_());
    }

    /** Rows one copy step covers: what the workspace holds past the routing table of a split by
     *  keys. */
    std::size_t copy_rows_() const noexcept {
        std::size_t routing = 0;
        if (!fences_.empty())
            for (source_t const& source : sources_)
                routing = max_of(routing, source.rows + divide_round_up(source.rows, route_block_k) *
                                                            (fences_.size() + 1) * sizeof(u32_t));
        std::size_t const reserved = routing + copy_slices_() * default_alignment_k;
        return workspace_.size() > reserved ? (workspace_.size() - reserved) / copy_row_bytes_() : 0;
    }

    /** Rows one re-select step covers. */
    std::size_t reselect_rows_() const noexcept {
        std::size_t const reserved = reselect_slices_k * default_alignment_k;
        return workspace_.size() > reserved ? (workspace_.size() - reserved) / reselect_row_bytes_() : 0;
    }

    /** Rows one spill bucket covers: as many as half the workspace holds proposals for. */
    std::size_t bucket_rows_() const noexcept {
        return workspace_.size() / 2 / (sizeof(u32_t) + connectivity_() * sizeof(scored_node_t));
    }

    /** Rows one build step covers, the largest block the workspace fits; none unless the plan
     *  builds over the views it opened. */
    std::size_t build_rows_() const noexcept {
        if (opened_.empty()) return 0;
        std::size_t rows = 0;
        for (source_t const& source : sources_) rows += source.rows;
        return fitting_block_rows_(rows);
    }

    std::size_t neighbors_row_bytes_() const noexcept { return connectivity_() * sizeof(node_id_t); }
    std::size_t candidate_slots_() const noexcept { return expansion_factor_() + connectivity_(); }
    std::size_t selection_bytes_() const noexcept { return connectivity_() * (sizeof(u32_t) + sizeof(vector_norms_t)); }

    /** Bytes a copy step stages per row: every column, and a second copy of them when key ranges
     *  scatter the rows. */
    std::size_t copy_row_bytes_() const noexcept {
        std::size_t const row = sizeof(key_t) + value_row_bytes_() + norms_bytes_() + neighbors_row_bytes_();
        return fences_.empty() ? row : row * 2;
    }

    /** Bytes a re-select step carves per row: the query, the neighbors in and out and as they were,
     *  the candidates with their rows, the selection scratch, the back-links and the reads of the
     *  current neighbors. */
    std::size_t reselect_row_bytes_() const noexcept {
        std::size_t const slots = candidate_slots_();
        return value_row_bytes_() + 2 * neighbors_row_bytes_() + sizeof(u32_t) +
               slots * (sizeof(scored_node_t) + value_row_bytes_() + sizeof(u32_t)) + selection_bytes_() +
               connectivity_() * (sizeof(proposal_t) + sizeof(u32_t));
    }

    /** Bytes a back-link batch carves per target row: the row in and out, its own value row,
     *  candidates numbering twice the connectivity with their rows, the selection scratch and the
     *  reads. */
    std::size_t back_link_row_bytes_() const noexcept {
        std::size_t const slots = connectivity_() * 2;
        return neighbors_row_bytes_() + value_row_bytes_() + sizeof(u32_t) +
               slots * (sizeof(scored_node_t) + value_row_bytes_() + sizeof(u32_t)) + selection_bytes_() +
               (slots + 1) * sizeof(u32_t);
    }

    /** Slices a copy step carves: keys, values, norms and neighbors, plus the routing table and the
     *  gathered columns of a split by keys. */
    std::size_t copy_slices_() const noexcept { return fences_.empty() ? 4 : 10; }

    /** Bytes a heal of a part of @p rows rows carves at least: two bits per row, one neighbors
     *  row, one landing, and the padding of its slices; the rest of the workspace holds more
     *  landings. */
    std::size_t heal_bytes_(std::size_t rows) const noexcept {
        return divide_round_up<std::size_t>(rows, 32) * sizeof(u64_t) + neighbors_row_bytes_() + sizeof(u64_t) +
               heal_slices_k * default_alignment_k;
    }

    /** Bytes a RAM index of @p rows rows takes from the workspace: the index itself and every chunk
     *  the rows reach, a slice per column, each slice padded to the alignment. */
    std::size_t ram_bytes_(std::size_t rows) const noexcept {
        using schedule_t = typename ram_index_t::chunk_schedule_t;
        std::size_t const chunks = schedule_t::chunks_for(rows);
        typename ram_index_t::geometry_t const geometry {dimensions_(), static_cast<connectivity_t>(connectivity_())};
        std::size_t const row_bytes = ram_index_t::padded_row_values(geometry) * sizeof(scalar_t) + norms_bytes_() +
                                      neighbors_row_bytes_();
        // A chunk without norms allocates only its values and neighbors.
        std::size_t const chunk_slices = norms_bytes_() ? 3 : 2;
        // The block's heal audits it on the same allocator.
        return sizeof(ram_index_t) + (chunks * chunk_slices + 1) * default_alignment_k +
               schedule_t::rows_before(chunks) * row_bytes + ram_index_t::audit_memory_usage(rows);
    }

    /** Whether a RAM index of @p rows rows fits the workspace beside @p staging bytes carved as
     *  @p slices slices. */
    bool ram_fits_(std::size_t rows, std::size_t staging, std::size_t slices) const noexcept {
        return ram_bytes_(rows) + staging + slices * default_alignment_k <= workspace_.size();
    }

    /** Rows of the largest block a build fits: the RAM index, and per row its value row, its two ID
     *  maps and its neighbors row on the way out. */
    std::size_t fitting_block_rows_(std::size_t rows) const noexcept {
        std::size_t const staging = value_row_bytes_() + 2 * sizeof(node_id_t) + neighbors_row_bytes_();
        std::size_t low = 0, high = rows;
        while (low < high) {
            std::size_t const middle = high - (high - low) / 2;
            if (ram_fits_(middle, middle * staging, build_slices_k)) low = middle;
            else high = middle - 1;
        }
        return low;
    }

#pragma endregion Sizes

#pragma region Scans and Writes

    /** Takes @p count elements off the front of @p space on a cache-line boundary; null when it is
     *  short. */
    template <typename value_type_>
    static value_type_* carve_(bytes_span_t& space, std::size_t count) noexcept {
        nvme_workspace_allocator<value_type_> slicer {space};
        return slicer.allocate(count);
    }

    /** Elements of @p element_bytes each that one carve off the front of @p space holds, past the
     *  padding to its boundary. */
    static std::size_t carvable_(bytes_span_t space, std::size_t element_bytes) noexcept {
        std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(space.data());
        std::size_t const padding = round_up_to_multiple<std::uintptr_t>(address, default_alignment_k) - address;
        return (space.size() - min_of(padding, space.size())) / element_bytes;
    }

    /** Scans rows [first, first + count) of every column of @p columns in part @p part of @p view
     *  into their staging, as runs of whole rows, in a call of its own. */
    status_t scan_rows_(indexes_t const& view, u32_t part, std::size_t first, std::size_t count,
                        std::span<staged_column_t const> columns, executor_t& executor) noexcept {
        if (!count || columns.empty()) return status_t::success_k;
        auto [call, call_status] = view.begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        return land_rows_(view, part, first, count, columns, executor);
    }

    /** The @ref scan_rows_ of a step whose call is already open, a scan's worth of slots
     *  begun. */
    status_t land_rows_(indexes_t const& view, u32_t part, std::size_t first, std::size_t count,
                        std::span<staged_column_t const> columns, executor_t& executor) noexcept {
        auto run_rows_of = [&](staged_column_t const& column) noexcept {
            return min_of(view.scan_run_rows_(part, column.row_bytes), count);
        };
        std::size_t tasks = 0;
        for (staged_column_t const& column : columns) tasks += divide_round_up(count, run_rows_of(column));
        // Task t is run t of the first column, then run t of the next once the first runs out.
        auto locate = [&](std::size_t task, std::size_t& run) noexcept -> staged_column_t const& {
            for (staged_column_t const& column : columns) {
                std::size_t const runs = divide_round_up(count, run_rows_of(column));
                if (task < runs) return run = task, column;
                task -= runs;
            }
            return run = 0, columns.back();
        };
        auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<extent_t, 1> {
            if (progress++) return {};
            std::size_t run = 0;
            staged_column_t const& column = locate(task, run);
            std::size_t const run_rows = run_rows_of(column);
            std::size_t const row = run * run_rows, rows = min_of(run_rows, count - row);
            return {
                extent_t {part, column.begin + u64_t(first + row) * column.row_bytes, u64_t(rows) * column.row_bytes}};
        };
        auto consume = [&](std::size_t, std::size_t task, auto const& extents, auto const& landed) noexcept {
            std::size_t run = 0;
            staged_column_t const& column = locate(task, run);
            std::memcpy(column.staging + run * run_rows_of(column) * column.row_bytes, landed[0],
                        static_cast<std::size_t>(extents[0].bytes));
        };
        auto const [reads, status] = view.scan_reads_(executor, tasks, describe, consume);
        result_.scan_reads += reads;
        return status;
    }

    /** Reads one row of the column at @p begin, @p row_bytes wide, of part zero of @p view per
     *  task: row @c row_of(task) lands at @c destination_of(task), and the reads count into
     *  @p reads. */
    template <typename row_of_type_, typename destination_of_type_>
    status_t read_points_(indexes_t const& view, u64_t begin, std::size_t row_bytes, std::size_t tasks,
                          row_of_type_&& row_of, destination_of_type_&& destination_of, executor_t& executor,
                          nvme_reads_t& reads) noexcept {
        if (!tasks) return status_t::success_k;
        auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<extent_t, 1> {
            if (progress++) return {};
            return {extent_t {0, begin + u64_t(row_of(task)) * row_bytes, row_bytes}};
        };
        auto consume = [&](std::size_t, std::size_t task, auto const&, auto const& landed) noexcept {
            std::memcpy(destination_of(task), landed[0], row_bytes);
        };
        auto [call, call_status] = view.begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        auto const [point_reads, status] = view.scan_reads_(executor, tasks, describe, consume);
        reads += point_reads;
        return status;
    }

    /** Scans the key column of source part @p source, calling @p on_key with every row and its key,
     *  on any thread. */
    template <typename on_key_type_>
    status_t scan_keys_(source_t const& source, executor_t& executor, on_key_type_&& on_key) noexcept {
        indexes_t const& view = *source.view;
        nvme_part_t const& part = view.parts_[source.part];
        std::size_t const run_rows = min_of(view.scan_run_rows_(source.part, sizeof(key_t)), source.rows);
        auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<extent_t, 1> {
            if (progress++) return {};
            std::size_t const row = task * run_rows, rows = min_of(run_rows, source.rows - row);
            return {extent_t {source.part, part.keys_begin + u64_t(row) * sizeof(key_t), u64_t(rows) * sizeof(key_t)}};
        };
        auto consume = [&](std::size_t, std::size_t task, auto const& extents, auto const& landed) noexcept {
            for (std::size_t row = 0; row != extents[0].bytes / sizeof(key_t); ++row) {
                key_t key;
                std::memcpy(&key, landed[0] + row * sizeof(key_t), sizeof(key_t));
                on_key(task * run_rows + row, key);
            }
        };
        auto [call, call_status] = view.begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        auto const [reads, status] = view.scan_reads_(executor, divide_round_up(source.rows, run_rows), describe,
                                                      consume);
        result_.scan_reads += reads;
        return status;
    }

    /** Runs @p per_task over @p tasks tasks on the threads of @p executor, claimed for the
     *  fan-out. */
    template <typename per_task_type_>
    static status_t fan_out_(executor_t& executor, std::size_t tasks, per_task_type_&& per_task) noexcept {
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        return executor.for_tasks(tasks, per_task);
    }

    /** Writes @p rows rows of @p row_bytes bytes from @p bytes into the column of target part
     *  @p target at @p column, from row @p first on: one positional write. */
    status_t write_rows_(target_t& target, u64_t column, std::size_t row_bytes, std::size_t first, std::size_t rows,
                         byte_t const* bytes) noexcept {
        if (!rows || !row_bytes) return status_t::success_k;
        status_t const wrote = target.file.write_at(column + first * row_bytes, bytes_view_t {bytes, rows * row_bytes});
        if (succeeded(wrote)) result_.written_bytes += rows * row_bytes;
        return wrote;
    }

    /** Where the norms column of target part @p target starts. */
    static u64_t norms_begin_(target_t const& target) noexcept {
        return target.header.data_offset + target.header.norms.data.first;
    }

    /** The one part of target part @p target. */
    static nvme_part_t const& part_of_(target_t const& target) noexcept { return target.view.parts_[0]; }

#pragma endregion Scans and Writes

#pragma region Selection

    /** The norms the kernel reads of the row whose bytes start at @p bytes, or none. */
    vector_norms_t norms_of_(byte_t const* bytes) const noexcept {
        return norms_bytes_() ? dense::norms<typename indexes_t::quads_kit_t>(view_of_(bytes)) : vector_norms_t {};
    }
    vector_view_t view_of_(byte_t const* bytes) const noexcept {
        return vector_view_t {reinterpret_cast<scalar_t const*>(bytes), dimensions_()};
    }

    /** The exact distance between two rows. */
    f32_t distance_(byte_t const* first, vector_norms_t first_norms, byte_t const* second,
                    vector_norms_t second_norms) const noexcept {
        return dense::distance<typename indexes_t::quads_kit_t>(kernel_(), view_of_(first), first_norms,
                                                                view_of_(second), second_norms);
    }

    /**
     *  @brief Picks the neighbors of candidate row @p row into @p out, as the RAM insertion picks
     *      them: walking its candidates nearest-first, a candidate is kept only when no kept one is
     *      closer to it than the row is, and the slots left take the nearest rejected candidates.
     *
     *  Sentinels pad @p out.
     *  @return The picks; @p computed counts the distances it took.
     */
    std::size_t select_diverse_(candidates_t const& candidates, std::size_t row, selection_t scratch,
                                std::span<node_id_t> out, std::size_t& computed) const noexcept {
        scored_node_t const* const entries = candidates.entries_of(row);
        std::size_t const count = candidates.counts[row], limit = out.size();
        for (u32_t slot = 0; slot != count; ++slot) scratch.order[slot] = slot;
        partial_sort(std::span<u32_t>(scratch.order, count), count, [&](u32_t first, u32_t second) noexcept {
            return entries[first].distance != entries[second].distance
                       ? entries[first].distance < entries[second].distance
                       : entries[first].node_id < entries[second].node_id;
        });
        std::size_t picks = 0;
        for (std::size_t position = 0; position != count && picks != limit; ++position) {
            u32_t const slot = scratch.order[position];
            byte_t const* const candidate = candidates.vector_of(row, slot);
            vector_norms_t const candidate_norms = norms_of_(candidate);
            std::size_t kept = 0;
            for (; kept != picks; ++kept) {
                ++computed;
                if (distance_(candidate, candidate_norms, candidates.vector_of(row, scratch.kept[kept]),
                              scratch.kept_norms[kept]) < entries[slot].distance)
                    break;
            }
            if (kept != picks) continue;
            scratch.kept[picks] = slot;
            scratch.kept_norms[picks++] = candidate_norms;
        }
        std::size_t const diverse = picks;
        for (std::size_t position = 0; position != count && picks != limit; ++position) {
            u32_t const slot = scratch.order[position];
            std::size_t kept = 0;
            while (kept != diverse && scratch.kept[kept] != slot) ++kept;
            if (kept == diverse) scratch.kept[picks++] = slot;
        }
        for (std::size_t pick = 0; pick != limit; ++pick)
            out[pick] = pick < picks ? entries[scratch.kept[pick]].node_id : sentinel_k;
        return picks;
    }

    /** The scratch of one diverse selection of batch row @p row, out of per-row arrays carved for
     *  the batch. */
    selection_t selection_of_(u32_t* orders, u32_t* kept_slots, vector_norms_t* norms, std::size_t row,
                              std::size_t slots) const noexcept {
        return {orders + row * slots, kept_slots + row * connectivity_(), norms + row * connectivity_()};
    }

    /** Whether the sentinel-padded row @p row holds @p node. */
    static bool holds_(std::span<node_id_t const> row, node_id_t node) noexcept {
        for (node_id_t const neighbor : row)
            if (neighbor == node) return true;
        return false;
    }

    /** Appends the back-links in @p proposals to the buckets of target part @p part, one positional
     *  write per bucket; sentinel records are skipped. */
    status_t spill_(std::size_t part, std::span<proposal_t> proposals) noexcept {
        // A full ascending order, so the sentinel records gather at the end, where the loop stops.
        partial_sort(proposals, proposals.size(), [](proposal_t const& first, proposal_t const& second) noexcept {
            return first.target < second.target;
        });
        paths_t name {allocator_};
        std::size_t const base = bucket_base_(part);
        for (std::size_t first = 0; first != proposals.size();) {
            if (proposals[first].target == sentinel_k) break;
            std::size_t const bucket = proposals[first].target / bucket_rows_();
            std::size_t bound = first + 1;
            while (bound != proposals.size() && proposals[bound].target != sentinel_k &&
                   proposals[bound].target / bucket_rows_() == bucket)
                ++bound;
            bucket_t& spill = buckets_[base + bucket];
            if (!spill.file) {
                if (status_t const named = spill_path_(name, base + bucket); failed(named)) return named;
                auto [file, open_status] = seekable_file_t::open(path_of_(name), file_mode_t::write_k, allocator_);
                if (failed(open_status)) return open_status;
                spill.file = std::move(file);
            }
            bytes_view_t const records = bytes_of(proposals.data() + first, bound - first);
            if (status_t const wrote = spill.file.write_at(spill.bytes, records); failed(wrote)) return wrote;
            spill.bytes += records.size();
            result_.written_bytes += records.size();
            first = bound;
        }
        return status_t::success_k;
    }

    /** The path of spill bucket @p bucket beside the target. */
    status_t spill_path_(paths_t& name, std::size_t bucket) const noexcept {
        std::string_view const stem = path_of_(stem_);
        auto print = [&](char* sink, std::size_t capacity) noexcept {
            return std::snprintf(sink, capacity, "%.*s-spill-%05zu.bin", static_cast<int>(stem.size()), stem.data(),
                                 bucket);
        };
        int const length = print(nullptr, 0);
        if (length < 0) return status_t::io_k;
        if (failed(name.resize_uninitialized(static_cast<std::size_t>(length) + 1))) return status_t::bad_alloc_k;
        print(name.data(), name.size());
        return status_t::success_k;
    }

    /** Closes and removes every spill bucket. */
    void drop_spills_() noexcept {
        paths_t name {allocator_};
        for (std::size_t bucket = 0; bucket != buckets_.size(); ++bucket) {
            if (!buckets_[bucket].file) continue;
            [[maybe_unused]] status_t const closed = buckets_[bucket].file.close();
            if (succeeded(spill_path_(name, bucket))) std::remove(name.data());
        }
        buckets_.clear();
    }

    /** Folds the telemetry of one batch of walks into the result. */
    void absorb_walks_(typename indexes_t::search_result_t const& walked) noexcept {
        result_.graph.visited_members += walked.visited_members;
        result_.graph.computed_distances += walked.computed_distances;
        result_.graph.restarts += walked.restarts;
        result_.graph.saturated_walks += walked.saturated_walks;
        result_.neighbor_reads += walked.neighbor_reads;
        result_.value_reads += walked.value_reads;
    }

#pragma endregion Selection

#pragma region Route and Copy

    /** The key range @p key falls in. */
    std::size_t tag_of_(key_t const& key) const noexcept {
        // The first fence above the key, as an upper bound finds it.
        auto const not_above = [](key_t const& fence, key_t const& probe) noexcept { return !(probe < fence); };
        return static_cast<std::size_t>(lower_bound(fences_, key, not_above) - fences_.data());
    }

    /** Counts the rows of every source part each key range takes, scanning their key columns over
     *  @p executor while a split plans, then opens one target part per non-empty range and turns
     *  the counts into every source part's first target row per range. */
    status_t route_(executor_t& executor) noexcept {
        std::size_t const ranges = fences_.size() + 1;
        if (failed(routes_.resize(sources_.size() * ranges)) || failed(range_targets_.resize(ranges)))
            return status_t::bad_alloc_k;
        for (std::size_t source = 0; source != sources_.size(); ++source) {
            std::size_t* const counts = routes_.data() + source * ranges;
            if (status_t const scanned = scan_keys_(
                    sources_[source], executor,
                    [&](std::size_t, key_t const& key) noexcept {
                        std::atomic_ref<std::size_t>(counts[tag_of_(key)]).fetch_add(1, std::memory_order_relaxed);
                    });
                failed(scanned))
                return scanned;
        }
        growing_buffer<std::size_t, forkunion_allocator_t> rows {allocator_};
        if (failed(rows.reserve_exactly(ranges))) return status_t::bad_alloc_k;
        for (std::size_t range = 0; range != ranges; ++range) {
            std::size_t total = 0;
            for (std::size_t source = 0; source != sources_.size(); ++source) {
                std::size_t& count = routes_[source * ranges + range];
                std::size_t const routed = count;
                count = total;
                total += routed;
            }
            range_targets_[range] = total ? rows.size() : ~std::size_t {0};
            if (total) rows.push_back_reserved(total);
        }
        return open_targets_(to_span(rows));
    }

    /** The routing of source part @p index carved off the front of @p space: a key range per row,
     *  and per block of @c route_block_k rows the rows of every range before it. Scans the key
     *  column unless the workspace holds it. */
    status_t route_table_(std::size_t index, executor_t& executor, bytes_span_t& space, u8_t*& tags,
                          u32_t*& directory) noexcept {
        source_t const& source = sources_[index];
        std::size_t const ranges = fences_.size() + 1, blocks = divide_round_up(source.rows, route_block_k);
        tags = carve_<u8_t>(space, source.rows);
        directory = carve_<u32_t>(space, blocks * ranges);
        if (!tags || !directory) return status_t::bad_alloc_k;
        if (routed_ == index) return status_t::success_k;
        routed_ = ~std::size_t {0};
        status_t const scanned = scan_keys_(source, executor, [&](std::size_t row, key_t const& key) noexcept {
            tags[row] = static_cast<u8_t>(tag_of_(key));
        });
        if (failed(scanned)) return scanned;
        std::array<u32_t, max_key_ranges_k> running {};
        for (std::size_t block = 0; block != blocks; ++block) {
            std::memcpy(directory + block * ranges, running.data(), ranges * sizeof(u32_t));
            for (std::size_t row = block * route_block_k; row != min_of((block + 1) * route_block_k, source.rows);
                 ++row)
                ++running[tags[row]];
        }
        routed_ = index;
        return status_t::success_k;
    }

    /** The target row of source row @p row of source part @p index under the routing table. */
    std::size_t routed_row_(std::size_t index, std::size_t row, u8_t const* tags,
                            u32_t const* directory) const noexcept {
        std::size_t const ranges = fences_.size() + 1, tag = tags[row], block_first = row - row % route_block_k;
        std::size_t rank = directory[(row / route_block_k) * ranges + tag];
        for (std::size_t earlier = block_first; earlier != row; ++earlier) rank += tags[earlier] == tag;
        return routes_[index * ranges + tag] + rank;
    }

    /** Rewires the neighbors of source row @p row of @p source in place onto the target IDs of its
     *  range, padding with sentinels, and answers the edges dropped for landing in another part. */
    std::size_t rewire_(source_t const& source, std::size_t row, node_id_t* neighbors) const noexcept {
        std::size_t const range = row / source.stride;
        std::size_t const low = max_of(source.kept.first, range * source.stride);
        std::size_t const high = min_of(source.kept.bound, (range + 1) * source.stride);
        std::size_t kept = 0, cut = 0;
        for (std::size_t slot = 0; slot != connectivity_(); ++slot) {
            std::size_t const neighbor = neighbors[slot];
            if (neighbor == row) continue;
            if (neighbor >= low && neighbor < high)
                neighbors[kept++] = static_cast<node_id_t>(source.base + neighbor - range * source.stride);
            else cut += !source.kept.empty() && neighbor < source.rows;
        }
        for (std::size_t slot = kept; slot != connectivity_(); ++slot) neighbors[slot] = sentinel_k;
        return cut;
    }

    /** The key-routed sibling of @ref rewire_. */
    std::size_t rewire_by_keys_(std::size_t index, std::size_t row, node_id_t* neighbors, u8_t const* tags,
                                u32_t const* directory) const noexcept {
        source_t const& source = sources_[index];
        std::size_t kept = 0, cut = 0;
        for (std::size_t slot = 0; slot != connectivity_(); ++slot) {
            std::size_t const neighbor = neighbors[slot];
            if (neighbor >= source.rows || neighbor == row) continue;
            if (tags[neighbor] == tags[row])
                neighbors[kept++] = static_cast<node_id_t>(routed_row_(index, neighbor, tags, directory));
            else ++cut;
        }
        for (std::size_t slot = kept; slot != connectivity_(); ++slot) neighbors[slot] = sentinel_k;
        return cut;
    }

    /** Copies one block of a source part into the target: keys and values are scanned in, norms are
     *  derived, neighbors rewired, and every target part the rows land in takes one positional
     *  write per column. */
    status_t copy_(step_t const& step, executor_t& executor) noexcept {
        std::size_t const index = step.part;
        source_t const& source = sources_[index];
        nvme_part_t const& part = source.view->parts_[source.part];
        std::size_t const count = step.nodes.size(), first = step.nodes.first;
        std::size_t const value_bytes = value_row_bytes_(), neighbor_bytes = neighbors_row_bytes_();
        bytes_span_t space = workspace_;
        u8_t* tags = nullptr;
        u32_t* directory = nullptr;
        if (!fences_.empty())
            if (status_t const routed = route_table_(index, executor, space, tags, directory); failed(routed))
                return routed;
        auto* const keys = carve_<byte_t>(space, count * sizeof(key_t));
        auto* const values = carve_<byte_t>(space, count * value_bytes);
        auto* const norms = carve_<byte_t>(space, count * norms_bytes_());
        auto* const neighbors = carve_<node_id_t>(space, count * connectivity_());
        if (!keys || !values || !norms || !neighbors) return status_t::bad_alloc_k;

        std::array<staged_column_t, 3> columns {
            {{part.keys_begin, sizeof(key_t), keys},
             {part.values_begin, value_bytes, values},
             {part.neighbors_begin, neighbor_bytes, reinterpret_cast<byte_t*>(neighbors)}}};
        // A part whose edges never survive reads no neighbors at all.
        std::size_t const scanned = source.kept.empty() ? 2 : 3;
        if (status_t const read = scan_rows_(*source.view, source.part, first, count,
                                             std::span<staged_column_t const>(columns).first(scanned), executor);
            failed(read))
            return read;

        std::size_t cut = 0;
        status_t const prepared = fan_out_(executor, count, [&](std::size_t row, std::size_t, std::size_t) noexcept {
            if (norms_bytes_()) {
                vector_norms_t const derived = norms_of_(values + row * value_bytes);
                std::memcpy(norms + row * norms_bytes_(), &derived, norms_bytes_());
            }
            node_id_t* const row_neighbors = neighbors + row * connectivity_();
            std::size_t row_cut = 0;
            if (!fences_.empty()) row_cut = rewire_by_keys_(index, first + row, row_neighbors, tags, directory);
            else if (!source.kept.empty()) row_cut = rewire_(source, first + row, row_neighbors);
            if (row_cut) std::atomic_ref<std::size_t>(cut).fetch_add(row_cut, std::memory_order_relaxed);
        });
        if (failed(prepared)) return prepared;
        result_.cut_edges += cut;

        if (fences_.empty()) {
            std::size_t const range = first / source.stride;
            target_t& target = targets_[source.target + range];
            nvme_part_t const& written = part_of_(target);
            std::size_t const landed = source.base + first - range * source.stride;
            status_t wrote = write_rows_(target, written.keys_begin, sizeof(key_t), landed, count, keys);
            if (succeeded(wrote)) wrote = write_rows_(target, written.values_begin, value_bytes, landed, count, values);
            if (succeeded(wrote))
                wrote = write_rows_(target, norms_begin_(target), norms_bytes_(), landed, count, norms);
            // A block build writes every row's neighbors itself.
            if (succeeded(wrote) && !source.kept.empty())
                wrote = write_rows_(target, written.neighbors_begin, neighbor_bytes, landed, count,
                                    reinterpret_cast<byte_t const*>(neighbors));
            return wrote;
        }
        return scatter_by_keys_(index, step, space, tags, directory, keys, values, norms, neighbors);
    }

    /** Groups one copied block by its rows' key ranges, stably, so each range's rows are contiguous
     *  in the target, and writes every range with one positional write per column. */
    status_t scatter_by_keys_(std::size_t index, step_t const& step, bytes_span_t& space, u8_t const* tags,
                              u32_t const* directory, byte_t const* keys, byte_t const* values, byte_t const* norms,
                              node_id_t const* neighbors) noexcept {
        std::size_t const count = step.nodes.size(), first = step.nodes.first, ranges = fences_.size() + 1;
        std::size_t const value_bytes = value_row_bytes_(), neighbor_bytes = neighbors_row_bytes_();
        auto* const gathered_keys = carve_<byte_t>(space, count * sizeof(key_t));
        auto* const gathered_values = carve_<byte_t>(space, count * value_bytes);
        auto* const gathered_norms = carve_<byte_t>(space, count * norms_bytes_());
        auto* const gathered_neighbors = carve_<node_id_t>(space, count * connectivity_());
        if (!gathered_keys || !gathered_values || !gathered_norms || !gathered_neighbors) return status_t::bad_alloc_k;
        std::array<std::size_t, max_key_ranges_k + 1> offsets {};
        for (std::size_t row = 0; row != count; ++row) ++offsets[tags[first + row] + 1];
        for (std::size_t range = 0; range != ranges; ++range) offsets[range + 1] += offsets[range];
        std::array<std::size_t, max_key_ranges_k> cursors {};
        std::memcpy(cursors.data(), offsets.data(), ranges * sizeof(std::size_t));
        for (std::size_t row = 0; row != count; ++row) {
            std::size_t const to = cursors[tags[first + row]]++;
            std::memcpy(gathered_keys + to * sizeof(key_t), keys + row * sizeof(key_t), sizeof(key_t));
            std::memcpy(gathered_values + to * value_bytes, values + row * value_bytes, value_bytes);
            std::memcpy(gathered_norms + to * norms_bytes_(), norms + row * norms_bytes_(), norms_bytes_());
            std::memcpy(gathered_neighbors + to * connectivity_(), neighbors + row * connectivity_(), neighbor_bytes);
        }
        for (std::size_t range = 0; range != ranges; ++range) {
            std::size_t const rows = offsets[range + 1] - offsets[range];
            if (!rows) continue;
            // The block's rows of one range are consecutive in its target, from the first of them
            // on.
            std::size_t row = 0;
            while (tags[first + row] != range) ++row;
            target_t& target = targets_[range_targets_[range]];
            nvme_part_t const& written = part_of_(target);
            std::size_t const landed = routed_row_(index, first + row, tags, directory), from = offsets[range];
            status_t wrote = write_rows_(target, written.keys_begin, sizeof(key_t), landed, rows,
                                         gathered_keys + from * sizeof(key_t));
            if (succeeded(wrote))
                wrote = write_rows_(target, written.values_begin, value_bytes, landed, rows,
                                    gathered_values + from * value_bytes);
            if (succeeded(wrote))
                wrote = write_rows_(target, norms_begin_(target), norms_bytes_(), landed, rows,
                                    gathered_norms + from * norms_bytes_());
            if (succeeded(wrote))
                wrote = write_rows_(target, written.neighbors_begin, neighbor_bytes, landed, rows,
                                    reinterpret_cast<byte_t const*>(gathered_neighbors + from * connectivity_()));
            if (failed(wrote)) return wrote;
        }
        return status_t::success_k;
    }

#pragma endregion Route and Copy

#pragma region RAM Blocks

    /** One traversal per executor thread over its own arena, sized for @p index, listed in the
     *  first arena; runs inside the call its step began, which rewound the arenas. */
    status_t traversals_(ram_index_t& index, executor_t& executor, traversals_t& traversals) noexcept {
        typename ram_index_t::insert_config_t const config {static_cast<u32_t>(expansion_factor_())};
        if (failed(traversals.reserve_exactly(executor.size()))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != executor.size(); ++thread) {
            traversals.push_back_reserved(
                traversal_context_t {typename traversal_context_t::allocator_t {executor.arena(thread)}});
            traversals[thread].random_generator.state = mix_bits(thread + 1);
            if (status_t const sized = index.reserve_traversal(traversals[thread], config); failed(sized)) return sized;
        }
        return status_t::success_k;
    }

    /** Folds the telemetry a RAM index's traversals ran up into the result. */
    void absorb_traversals_(traversals_t const& traversals) noexcept {
        for (traversal_context_t const& traversal : traversals) {
            result_.graph.visited_members += traversal.visited_members;
            result_.graph.computed_distances += traversal.computed_distances;
            result_.graph.computed_distances_in_refines += traversal.computed_distances_in_refines;
            result_.graph.computed_distances_in_reverse_refines += traversal.computed_distances_in_reverse_refines;
            result_.graph.restarts += traversal.restarts;
            result_.graph.saturated_walks += traversal.saturated_walks;
        }
    }

    /** Writes the neighbors of RAM rows [0, rows) of @p index as target rows from @p first on,
     *  every ID through @p target_of and every row through @p node_of, staged in @p space and
     *  written in runs as wide as it holds. */
    template <typename node_of_type_, typename target_of_type_>
    status_t write_ram_rows_(ram_index_t const& index, target_t& target, std::size_t first, std::size_t rows,
                             bytes_span_t space, executor_t& executor, node_of_type_&& node_of,
                             target_of_type_&& target_of) noexcept {
        std::size_t const run_rows = min_of(rows, carvable_(space, neighbors_row_bytes_()));
        auto* const staged = carve_<node_id_t>(space, run_rows * connectivity_());
        if (!run_rows || !staged) return status_t::bad_alloc_k;
        ram_cursor_t cursor {index};
        if (status_t const reserved = cursor.reserve(arena_allocator<byte_t, forkunion_allocator_t>(executor.arena(0)));
            failed(reserved))
            return reserved;
        for (std::size_t run = 0; run < rows; run += run_rows) {
            std::size_t const run_count = min_of(run_rows, rows - run);
            for (std::size_t row = 0; row != run_count; ++row) {
                node_id_t* const out = staged + row * connectivity_();
                std::size_t written = 0;
                if (succeeded(cursor.seek(node_of(run + row))))
                    for (node_id_t const neighbor : cursor.neighbors()) out[written++] = target_of(neighbor);
                for (; written != connectivity_(); ++written) out[written] = sentinel_k;
            }
            if (status_t const wrote = write_rows_(target, part_of_(target).neighbors_begin, neighbors_row_bytes_(),
                                                   first + run, run_count, reinterpret_cast<byte_t const*>(staged));
                failed(wrote))
                return wrote;
        }
        return status_t::success_k;
    }

    /** Builds target rows @p step.nodes in a RAM index inside the workspace from their value rows,
     *  heals it there, and writes their neighbors, every edge inside the block. */
    status_t build_(step_t const& step, executor_t& executor) noexcept {
        target_t& target = targets_[step.part];
        std::size_t const count = step.nodes.size(), first = step.nodes.first;
        bytes_span_t space = workspace_;
        auto* const values = carve_<byte_t>(space, count * value_row_bytes_());
        auto* const nodes = carve_<node_id_t>(space, count);
        auto* const rows = carve_<node_id_t>(space, count);
        auto* const staging = carve_<node_id_t>(space, count * connectivity_());
        if (!values || !nodes || !rows || !staging) return status_t::bad_alloc_k;
        ram_index_t* const index_bytes = carve_<ram_index_t>(space, 1);
        if (!index_bytes) return status_t::bad_alloc_k;
        ram_index_ptr_t const index_ptr {new (index_bytes) ram_index_t(
            kernel_(), nvme_workspace_allocator<byte_t>(space), nvme_workspace_allocator<byte_t>(space))};
        ram_index_t& index = *index_ptr;
        if (status_t const reserved =
                index.reserve({count, static_cast<connectivity_t>(connectivity_()), dimensions_()}).status;
            failed(reserved))
            return reserved;

        // One call holds the executor for the block's scan, its inserts, its heal and its writes.
        typename ram_index_t::insert_config_t const config {static_cast<u32_t>(expansion_factor_())};
        auto [call, call_status] = target.view.begin_scans_(executor, 1, index.traversal_memory_usage(config));
        if (failed(call_status)) return call_status;
        std::array<staged_column_t, 1> const columns {{{part_of_(target).values_begin, value_row_bytes_(), values}}};
        if (status_t const read = land_rows_(target.view, 0, first, count, columns, executor); failed(read))
            return read;
        traversals_t traversals {scratch_allocator_t {executor.arena(0)}};
        if (status_t const ready = traversals_(index, executor, traversals); failed(ready)) return ready;
        status_t refusal = status_t::success_k;
        auto refuse = [&](status_t status) noexcept {
            status_t unset = status_t::success_k;
            if (failed(status))
                std::atomic_ref<status_t>(refusal).compare_exchange_strong(unset, status, std::memory_order_relaxed);
        };
        // One row per task, as a RAM group inserts, so the threads interleave from the first row.
        auto insert = [&](std::size_t row, std::size_t thread, std::size_t) noexcept {
            vectors_view_t const inserted_row {reinterpret_cast<scalar_t const*>(values + row * value_row_bytes_()),
                                               std::size_t {1}, dimensions_()};
            refuse(index
                       .insert(inserted_row, typename ram_index_t::node_ids_span_t {nodes + row, std::size_t {1}},
                               traversals[thread], config)
                       .status);
        };
        if (status_t const fanned = executor.for_tasks(count, insert); failed(fanned)) return fanned;
        if (failed(refusal)) return refusal;
        // In RAM a block's orphan clusters cost DRAM sweeps to find; once written, only `health`
        // finds them.
        auto [healing, heal_status] = index.plan_refine({}, {static_cast<u32_t>(expansion_factor_())});
        if (failed(heal_status)) return heal_status;
        do {
            auto const steps = healing.steps();
            auto heal = [&](std::size_t position, std::size_t thread, std::size_t) noexcept {
                refuse(healing.run(steps[position], traversals[thread]));
            };
            if (status_t const fanned = executor.for_tasks(steps.size(), heal); failed(fanned)) return fanned;
            if (failed(refusal)) return refusal;
        } while (healing.advance());
        auto const [healed, healed_status] = healing.finish();
        if (failed(healed_status)) return healed_status;
        // The heal walked the same traversals, so their counters already hold its telemetry.
        absorb_traversals_(traversals);
        result_.graph.visited_nodes += count + healed.visited_nodes;
        result_.graph.changed_nodes += healed.changed_nodes;
        result_.graph.islands_found += healed.islands_found;
        result_.graph.islands_reconnected += healed.islands_reconnected;
        for (std::size_t row = 0; row != count; ++row) rows[nodes[row]] = static_cast<node_id_t>(row);
        bytes_span_t const out {reinterpret_cast<byte_t*>(staging), count * neighbors_row_bytes_()};
        return write_ram_rows_(
            index, target, first, count, out, executor, [&](std::size_t row) noexcept { return nodes[row]; },
            [&](node_id_t node) noexcept { return static_cast<node_id_t>(first + rows[node]); });
    }

    /** Refines target part @p part whole in a RAM index inside the workspace: its value and
     *  neighbors columns load through a read-only mapping, the RAM refine re-selects and heals
     *  every row, and the rows write back. */
    status_t refine_in_ram_(std::size_t part, executor_t& executor) noexcept {
        target_t& target = targets_[part];
        std::size_t const nodes = target.view.size();
        auto [mapping, map_status] = mapped_file_t::read(path_of_(target.path));
        if (failed(map_status)) return map_status;
        bytes_view_t const bytes = mapping.view();
        safegraphs_header_t const& header = target.header;
        typename ram_index_t::columns_view_t const columns {header.data_of(bytes, header.value),
                                                            header.data_of(bytes, header.neighbors),
                                                            {}};
        bytes_span_t space = workspace_;
        ram_index_t* const index_bytes = carve_<ram_index_t>(space, 1);
        if (!index_bytes) return status_t::bad_alloc_k;
        ram_index_ptr_t const index_ptr {new (index_bytes) ram_index_t(
            kernel_(), nvme_workspace_allocator<byte_t>(space), nvme_workspace_allocator<byte_t>(space))};
        ram_index_t& index = *index_ptr;
        if (status_t const reserved =
                index.reserve({nodes, static_cast<connectivity_t>(connectivity_()), dimensions_()}).status;
            failed(reserved))
            return reserved;
        // One call holds the executor for the load, the refine and the writes; it reads no medium.
        typename ram_index_t::insert_config_t const sizing {static_cast<u32_t>(expansion_factor_())};
        auto [call, call_status] = nvme_begin_call(executor, std::span<medium_t const> {},
                                                   index.traversal_memory_usage(sizing), 0, 0);
        if (failed(call_status)) return call_status;
        traversals_t traversals {scratch_allocator_t {executor.arena(0)}};
        if (status_t const ready = traversals_(index, executor, traversals); failed(ready)) return ready;
        status_t refusal = status_t::success_k;
        auto refuse = [&](status_t status) noexcept {
            status_t unset = status_t::success_k;
            if (failed(status))
                std::atomic_ref<status_t>(refusal).compare_exchange_strong(unset, status, std::memory_order_relaxed);
        };
        {
            auto [load, load_status] = index.plan_load(nodes);
            if (failed(load_status)) return load_status;
            auto const steps = load.steps();
            iota_view<node_id_t> const source_range {0, static_cast<node_id_t>(nodes)};
            if (status_t const fanned = executor.for_tasks(
                    steps.size(),
                    [&](std::size_t position, std::size_t, std::size_t) noexcept {
                        refuse(load.load(steps[position], columns, source_range));
                    });
                failed(fanned))
                return fanned;
            if (failed(refusal)) return refusal;
            if (status_t const loaded = load.finish(); failed(loaded)) return loaded;
        }
        typename ram_index_t::refine_config_t const config {static_cast<u32_t>(expansion_factor_())};
        auto [refining, refine_status] = index.plan_refine({0, static_cast<node_id_t>(nodes)}, config);
        if (failed(refine_status)) return refine_status;
        do {
            auto const steps = refining.steps();
            if (status_t const fanned = executor.for_tasks(
                    steps.size(),
                    [&](std::size_t position, std::size_t thread, std::size_t) noexcept {
                        refuse(refining.run(steps[position], traversals[thread]));
                    });
                failed(fanned))
                return fanned;
            if (failed(refusal)) return refusal;
        } while (refining.advance());
        auto const [refined, finish_status] = refining.finish();
        if (failed(finish_status)) return finish_status;
        result_.graph.visited_nodes += refined.visited_nodes;
        result_.graph.changed_nodes += refined.changed_nodes;
        result_.graph.islands_found += refined.islands_found;
        result_.graph.islands_reconnected += refined.islands_reconnected;
        result_.graph.computed_distances += refined.computed_distances;
        result_.graph.computed_distances_in_refines += refined.computed_distances_in_refines;
        result_.graph.computed_distances_in_reverse_refines += refined.computed_distances_in_reverse_refines;
        result_.graph.visited_members += refined.visited_members;
        result_.graph.restarts += refined.restarts;
        result_.graph.saturated_walks += refined.saturated_walks;
        return write_ram_rows_(
            index, target, 0, nodes, space, executor,
            [](std::size_t row) noexcept { return static_cast<node_id_t>(row); },
            [](node_id_t node) noexcept { return node; });
    }

#pragma endregion RAM Blocks

#pragma region Reselection and Back Links

    /** Re-selects target rows @p step.nodes: each walks the target part from random landmarks, the
     *  appended part of a merge or a build walking across into the resident part alone first, its
     *  candidates keeping the best rows it scored and its current neighbors, then a diverse pick
     *  lands as its row and spills its back-links. A whole part the workspace holds is refined in
     *  RAM instead. */
    status_t reselect_(step_t const& step, executor_t& executor) noexcept {
        target_t& target = targets_[step.part];
        std::size_t const nodes = target.view.size(), count = step.nodes.size(), first = step.nodes.first;
        if (count == nodes && ram_fits_(nodes, neighbors_row_bytes_(), refine_slices_k))
            return refine_in_ram_(step.part, executor);
        std::size_t const slots = candidate_slots_(), connectivity = connectivity_();
        nvme_part_t const& part = part_of_(target);
        bytes_span_t space = workspace_;
        auto* const queries = carve_<byte_t>(space, count * value_row_bytes_());
        auto* const neighbors = carve_<node_id_t>(space, count * connectivity);
        auto* const previous_neighbors = carve_<node_id_t>(space, count * connectivity);
        auto* const counts = carve_<u32_t>(space, count);
        auto* const entries = carve_<scored_node_t>(space, count * slots);
        auto* const vectors = carve_<byte_t>(space, count * slots * value_row_bytes_());
        auto* const orders = carve_<u32_t>(space, count * slots);
        auto* const kept_slots = carve_<u32_t>(space, count * connectivity);
        auto* const kept_norms = carve_<vector_norms_t>(space, count * connectivity);
        auto* const proposals = carve_<proposal_t>(space, count * connectivity);
        auto* const points = carve_<u32_t>(space, count * connectivity);
        if (!queries || !neighbors || !previous_neighbors || !counts || !entries || !vectors || !orders ||
            !kept_slots || !kept_norms || !proposals || !points)
            return status_t::bad_alloc_k;
        std::array<staged_column_t, 2> const columns {
            {{part.values_begin, value_row_bytes_(), queries},
             {part.neighbors_begin, neighbors_row_bytes_(), reinterpret_cast<byte_t*>(neighbors)}}};
        if (status_t const read = scan_rows_(target.view, 0, first, count, columns, executor); failed(read))
            return read;

        candidates_t const candidates {entries, vectors, counts, slots, value_row_bytes_()};
        std::memset(counts, 0, count * sizeof(u32_t));
        using row_t = reselect_row_t;
        typename row_t::inputs_t const inputs {
            vectors_view_t {reinterpret_cast<scalar_t const*>(queries), count, dimensions_()}, candidates,
            static_cast<node_id_t>(first), expansion_factor_()};
        search_config_t config {};
        config.expansion_factor = static_cast<typename indexes_t::index_t::small_size_t>(expansion_factor_());
        // Appended rows first walk across into the resident part alone, since no resident row links
        // to them yet: the part read as ending at the appended base draws and expands nothing past
        // it, and both walks fill the same candidates.
        if (appended_base_ && first >= appended_base_) {
            u32_t& part_nodes = target.view.parts_[0].nodes;
            u32_t const all_nodes = std::exchange(part_nodes, static_cast<u32_t>(appended_base_));
            auto const [crossed, cross_status] = target.view.template walk_rows_<row_t>(count, config, inputs, 0, {},
                                                                                        {}, executor);
            part_nodes = all_nodes;
            if (failed(cross_status)) return cross_status;
            absorb_walks_(crossed);
        }
        auto const [walked, walk_status] = target.view.template walk_rows_<row_t>(count, config, inputs, 0, {}, {},
                                                                                  executor);
        if (failed(walk_status)) return walk_status;
        absorb_walks_(walked);

        // The current neighbors compete as the RAM refine's seeds do, their rows read into the
        // candidates' spare slots.
        std::size_t current = 0;
        for (std::size_t row = 0; row != count; ++row)
            for (std::size_t slot = 0; slot != connectivity; ++slot) {
                node_id_t const neighbor = neighbors[row * connectivity + slot];
                if (neighbor >= nodes || neighbor == first + row) continue;
                std::size_t const taken = candidates.offer(row, {neighbor, 0}, slots);
                if (taken != slots) points[current++] = static_cast<u32_t>(row * slots + taken);
            }
        if (status_t const read = read_points_(
                target.view, part.values_begin, value_row_bytes_(), current,
                [&](std::size_t task) noexcept { return entries[points[task]].node_id; },
                [&](std::size_t task) noexcept { return vectors + std::size_t(points[task]) * value_row_bytes_(); },
                executor, result_.value_reads);
            failed(read))
            return read;

        status_t const scored = fan_out_(executor, current, [&](std::size_t point, std::size_t, std::size_t) noexcept {
            std::size_t const position = points[point];
            byte_t const* const query = queries + position / slots * value_row_bytes_();
            byte_t const* const candidate = vectors + position * value_row_bytes_();
            entries[position].distance = distance_(query, norms_of_(query), candidate, norms_of_(candidate));
        });
        if (failed(scored)) return scored;

        std::size_t computed = current, changed = 0;
        status_t const selected = fan_out_(executor, count, [&](std::size_t row, std::size_t, std::size_t) noexcept {
            std::size_t local_computed = 0;
            std::span<node_id_t> const out {neighbors + row * connectivity, connectivity};
            std::span<node_id_t> const before {previous_neighbors + row * connectivity, connectivity};
            std::memcpy(before.data(), out.data(), connectivity * sizeof(node_id_t));
            std::size_t const picks = select_diverse_(
                candidates, row, selection_of_(orders, kept_slots, kept_norms, row, slots), out, local_computed);
            std::size_t matched = 0, previous = 0;
            for (std::size_t slot = 0; slot != connectivity; ++slot) {
                previous += before[slot] < nodes && before[slot] != first + row;
                matched += out[slot] != sentinel_k && holds_(before, out[slot]);
            }
            proposal_t* const spilled = proposals + row * connectivity;
            for (std::size_t pick = 0; pick != connectivity; ++pick) {
                spilled[pick] = proposal_t {};
                if (pick >= picks) continue;
                scored_node_t const* const row_entries = candidates.entries_of(row);
                std::size_t slot = 0;
                while (row_entries[slot].node_id != out[pick]) ++slot;
                spilled[pick] = proposal_t {out[pick], static_cast<node_id_t>(first + row), row_entries[slot].distance};
            }
            std::atomic_ref<std::size_t>(computed).fetch_add(local_computed, std::memory_order_relaxed);
            std::atomic_ref<std::size_t>(changed).fetch_add(matched != picks || previous != picks,
                                                            std::memory_order_relaxed);
        });
        if (failed(selected)) return selected;
        result_.graph.computed_distances += computed;
        result_.graph.computed_distances_in_refines += computed;
        result_.graph.visited_nodes += count;
        result_.graph.changed_nodes += changed;
        if (status_t const wrote = write_rows_(target, part.neighbors_begin, neighbors_row_bytes_(), first, count,
                                               reinterpret_cast<byte_t const*>(neighbors));
            failed(wrote))
            return wrote;
        return spill_(step.part, {proposals, count * connectivity});
    }

    /** Applies one bucket of back-links to target rows @p step.nodes: every row keeps its closest
     *  distinct proposals, then in batches the workspace holds, a touched row re-selects among its
     *  current neighbors and those proposals, every candidate's value row read, and the batch's
     *  rows write back in one run. A bucket nothing spilled into has nothing to apply. */
    status_t back_link_(step_t const& step, executor_t& executor) noexcept {
        target_t& target = targets_[step.part];
        nvme_part_t const& part = part_of_(target);
        std::size_t const nodes = target.view.size(), connectivity = connectivity_(), rows = step.nodes.size();
        std::size_t const lowest = step.nodes.first, slots = connectivity * 2;
        bucket_t& bucket = buckets_[bucket_base_(step.part) + lowest / bucket_rows_()];
        if (!bucket.bytes) return status_t::success_k;
        bytes_span_t space = workspace_;
        auto* const proposed_counts = carve_<u32_t>(space, rows);
        auto* const proposed = carve_<scored_node_t>(space, rows * connectivity);
        if (!proposed_counts || !proposed) return status_t::bad_alloc_k;
        std::memset(proposed_counts, 0, rows * sizeof(u32_t));
        candidates_t const proposals {proposed, nullptr, proposed_counts, connectivity, 0};

        // The spill reads back in runs as wide as all that is left, which the batches reuse once it
        // is read, and every target keeps its closest sources.
        bytes_span_t const batch_space = space;
        std::size_t const run_bytes = carvable_(space, sizeof(proposal_t)) * sizeof(proposal_t);
        auto* const run = carve_<proposal_t>(space, run_bytes / sizeof(proposal_t));
        if (!run_bytes || !run) return status_t::bad_alloc_k;
        for (std::size_t offset = 0; offset < bucket.bytes; offset += run_bytes) {
            std::size_t const length = min_of(run_bytes, bucket.bytes - offset);
            if (status_t const read = bucket.file.read_at(offset, {reinterpret_cast<byte_t*>(run), length});
                failed(read))
                return read;
            result_.scan_reads += nvme_reads_t {1, length};
            for (std::size_t record = 0; record != length / sizeof(proposal_t); ++record) {
                proposal_t const& back_link = run[record];
                if (back_link.target < lowest || back_link.target >= lowest + rows ||
                    back_link.source == back_link.target)
                    continue;
                proposals.offer(back_link.target - lowest, {back_link.source, back_link.distance}, connectivity);
            }
        }

        std::size_t const batch_reserved = back_link_batch_slices_k * default_alignment_k;
        std::size_t const batch_rows = min_of(rows, batch_space.size() > batch_reserved
                                                        ? (batch_space.size() - batch_reserved) / back_link_row_bytes_()
                                                        : 0);
        if (!batch_rows) return status_t::bad_alloc_k;
        for (std::size_t batch_first = 0; batch_first < rows; batch_first += batch_rows) {
            std::size_t const count = min_of(batch_rows, rows - batch_first), first = lowest + batch_first;
            bytes_span_t carving = batch_space;
            auto* const neighbors = carve_<node_id_t>(carving, count * connectivity);
            auto* const owns = carve_<byte_t>(carving, count * value_row_bytes_());
            auto* const counts = carve_<u32_t>(carving, count);
            auto* const entries = carve_<scored_node_t>(carving, count * slots);
            auto* const vectors = carve_<byte_t>(carving, count * slots * value_row_bytes_());
            auto* const orders = carve_<u32_t>(carving, count * slots);
            auto* const kept_slots = carve_<u32_t>(carving, count * connectivity);
            auto* const kept_norms = carve_<vector_norms_t>(carving, count * connectivity);
            auto* const points = carve_<u32_t>(carving, count * (slots + 1));
            if (!neighbors || !owns || !counts || !entries || !vectors || !orders || !kept_slots || !kept_norms ||
                !points)
                return status_t::bad_alloc_k;
            std::array<staged_column_t, 1> const columns {
                {{part.neighbors_begin, neighbors_row_bytes_(), reinterpret_cast<byte_t*>(neighbors)}}};
            if (status_t const read = scan_rows_(target.view, 0, first, count, columns, executor); failed(read))
                return read;

            // Point `p` below `count` reads the batch row itself, every later one a candidate slot
            // of some row.
            candidates_t const candidates {entries, vectors, counts, slots, value_row_bytes_()};
            std::size_t reads = 0;
            for (std::size_t row = 0; row != count; ++row) {
                counts[row] = 0;
                std::size_t const local = batch_first + row;
                if (!proposed_counts[local]) continue;
                points[reads++] = static_cast<u32_t>(row);
                node_id_t const self = static_cast<node_id_t>(first + row);
                for (std::size_t slot = 0; slot != connectivity; ++slot) {
                    node_id_t const neighbor = neighbors[row * connectivity + slot];
                    if (neighbor < nodes && neighbor != self) candidates.offer(row, {neighbor, 0}, slots);
                }
                scored_node_t const* const proposed_row = proposals.entries_of(local);
                for (std::size_t slot = 0; slot != proposed_counts[local]; ++slot)
                    candidates.offer(row, {proposed_row[slot].node_id, 0}, slots);
                for (std::size_t slot = 0; slot != counts[row]; ++slot)
                    points[reads++] = static_cast<u32_t>(count + row * slots + slot);
            }
            if (status_t const read = read_points_(
                    target.view, part.values_begin, value_row_bytes_(), reads,
                    [&](std::size_t task) noexcept {
                        std::size_t const point = points[task];
                        return point < count ? static_cast<node_id_t>(first + point) : entries[point - count].node_id;
                    },
                    [&](std::size_t task) noexcept {
                        std::size_t const point = points[task];
                        return point < count ? owns + point * value_row_bytes_()
                                             : vectors + (point - count) * value_row_bytes_();
                    },
                    executor, result_.value_reads);
                failed(read))
                return read;

            std::size_t computed = 0;
            status_t const selected = fan_out_(
                executor, count, [&](std::size_t row, std::size_t, std::size_t) noexcept {
                    if (!counts[row]) return;
                    byte_t const* const own = owns + row * value_row_bytes_();
                    vector_norms_t const own_norms = norms_of_(own);
                    for (std::size_t slot = 0; slot != counts[row]; ++slot) {
                        byte_t const* const candidate = candidates.vector_of(row, slot);
                        entries[row * slots + slot].distance = distance_(own, own_norms, candidate,
                                                                         norms_of_(candidate));
                    }
                    std::size_t local_computed = counts[row];
                    std::span<node_id_t> const out {neighbors + row * connectivity, connectivity};
                    select_diverse_(candidates, row, selection_of_(orders, kept_slots, kept_norms, row, slots), out,
                                    local_computed);
                    std::atomic_ref<std::size_t>(computed).fetch_add(local_computed, std::memory_order_relaxed);
                });
            if (failed(selected)) return selected;
            result_.graph.computed_distances += computed;
            result_.graph.computed_distances_in_reverse_refines += computed;
            if (status_t const wrote = write_rows_(target, part.neighbors_begin, neighbors_row_bytes_(), first, count,
                                                   reinterpret_cast<byte_t const*>(neighbors));
                failed(wrote))
                return wrote;
        }
        return status_t::success_k;
    }

#pragma endregion Reselection and Back Links

#pragma region Heal

    /** Heals target part @p step.part: one sequential pass of its neighbors column counts every
     *  row's in-links, then each row nothing links to is read and lands, in target order, in the
     *  row of its first neighbor something links to, or of the landmark covering it. A cluster
     *  linking only to itself stays, for @c health to report. */
    status_t heal_(step_t const& step, executor_t& executor) noexcept {
        target_t& target = targets_[step.part];
        nvme_part_t const& part = part_of_(target);
        std::size_t const nodes = target.view.size(), connectivity = connectivity_(),
                          row_bytes = neighbors_row_bytes_();
        std::size_t const run_rows = target.view.scan_run_rows_(0, row_bytes);
        std::size_t const seeds = min_of<std::size_t>(connectivity, nodes), spread = heal_spread(connectivity);
        std::size_t const words = divide_round_up<std::size_t>(nodes, 32);
        bytes_span_t space = workspace_;
        // Two bits per row: the first in-link sets the low one, the second the high one.
        auto* const in_links = carve_<u64_t>(space, words);
        auto* const target_row = carve_<node_id_t>(space, connectivity);
        std::size_t const capacity = carvable_(space, sizeof(u64_t));
        auto* const landings = carve_<u64_t>(space, capacity);
        if (!in_links || !target_row || !capacity || !landings) return status_t::bad_alloc_k;
        std::memset(in_links, 0, words * sizeof(u64_t));
        auto links_of = [&](std::size_t node) noexcept {
            return static_cast<unsigned>(in_links[node / 32] >> (node % 32 * 2)) & 3u;
        };
        auto [call, call_status] = target.view.begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;

        auto whole_run = [&](std::size_t task, u64_t& progress) noexcept -> std::array<extent_t, 1> {
            if (progress++) return {};
            std::size_t const first = task * run_rows;
            return {
                extent_t {0, part.neighbors_begin + first * row_bytes, min_of(run_rows, nodes - first) * row_bytes}};
        };
        auto count_links = [&](std::size_t, std::size_t task, auto const& extents, auto const& landed) noexcept {
            auto const* const neighbors = reinterpret_cast<node_id_t const*>(landed[0]);
            for (std::size_t slot = 0; slot != extents[0].bytes / sizeof(node_id_t); ++slot) {
                std::size_t const neighbor = neighbors[slot];
                if (neighbor >= nodes || neighbor == task * run_rows + slot / connectivity) continue;
                std::atomic_ref<u64_t> const word {in_links[neighbor / 32]};
                u64_t const once = u64_t {1} << (neighbor % 32 * 2);
                if (word.fetch_or(once, std::memory_order_relaxed) & once)
                    word.fetch_or(once << 1, std::memory_order_relaxed);
            }
        };
        auto const [scanned, scan_status] = target.view.scan_reads_(executor, divide_round_up(nodes, run_rows),
                                                                    whole_run, count_links);
        result_.scan_reads += scanned;
        if (failed(scan_status)) return scan_status;

        for (std::size_t from = 0; from != nodes;) {
            // The rows from `from` on whose islands the landings hold.
            std::size_t bound = from, islands = 0;
            for (; bound != nodes && islands != capacity; ++bound) islands += !links_of(bound);
            result_.graph.islands_found += islands;
            alignas(default_alignment_k) std::size_t taken = 0;
            auto island_rows = [&](std::size_t task, u64_t& progress) noexcept -> std::array<extent_t, 1> {
                std::size_t const task_first = from + task * run_rows;
                std::size_t row = task_first + progress, end = min_of(task_first + run_rows, bound);
                while (row != end && links_of(row)) ++row;
                if (row == end) return {};
                progress = row + 1 - task_first;
                return {extent_t {0, part.neighbors_begin + row * row_bytes, row_bytes}};
            };
            auto pick_targets = [&](std::size_t, std::size_t, auto const& extents, auto const& landed) noexcept {
                std::size_t const island = static_cast<std::size_t>((extents[0].begin - part.neighbors_begin) /
                                                                    row_bytes);
                auto const* const neighbors = reinterpret_cast<node_id_t const*>(landed[0]);
                std::size_t linked = island * seeds / nodes * nodes / seeds;
                for (std::size_t slot = 0; slot != connectivity; ++slot)
                    if (neighbors[slot] < nodes && neighbors[slot] != island && links_of(neighbors[slot])) {
                        linked = neighbors[slot];
                        break;
                    }
                if (linked == island) return;
                landings[std::atomic_ref<std::size_t>(taken).fetch_add(1, std::memory_order_relaxed)] =
                    static_cast<u64_t>(linked) << 32 | island;
            };
            auto const [read, pick_status] = target.view.scan_reads_(executor, divide_round_up(bound - from, run_rows),
                                                                     island_rows, pick_targets);
            result_.neighbor_reads += read;
            if (failed(pick_status)) return pick_status;

            // In target order, so every target row is read and written once, near-sequentially.
            partial_sort(std::span<u64_t> {landings, taken}, taken);
            for (std::size_t position = 0; position != taken;) {
                std::size_t const linked = static_cast<std::size_t>(landings[position] >> 32);
                result_.neighbor_reads += nvme_reads_t {1, row_bytes};
                if (status_t const fetched = target.file.read_at(part.neighbors_begin + linked * row_bytes,
                                                                 {reinterpret_cast<byte_t*>(target_row), row_bytes});
                    failed(fetched))
                    return fetched;
                for (; position != taken && landings[position] >> 32 == linked; ++position) {
                    std::size_t const island = static_cast<u32_t>(landings[position]);
                    std::size_t landing = 0;
                    while (landing != connectivity && target_row[landing] < nodes) ++landing;
                    // A full row gives up a tail slot spread by the island's ID whose resident
                    // keeps another in-link; an island landed before counts none, so its slot
                    // stays.
                    for (std::size_t attempt = 0; landing == connectivity && attempt != spread; ++attempt) {
                        std::size_t const slot = connectivity - 1 - (island + attempt) % spread;
                        std::size_t const resident = target_row[slot];
                        if (links_of(resident) != 3u) continue;
                        in_links[resident / 32] &= ~(u64_t {2} << (resident % 32 * 2));
                        landing = slot;
                    }
                    if (landing == connectivity) continue;
                    target_row[landing] = static_cast<node_id_t>(island);
                    ++result_.graph.islands_reconnected;
                }
                if (status_t const wrote = write_rows_(target, part.neighbors_begin, row_bytes, linked, 1,
                                                       reinterpret_cast<byte_t const*>(target_row));
                    failed(wrote))
                    return wrote;
            }
            from = bound;
        }
        return status_t::success_k;
    }

#pragma endregion Heal

    /** The host allocator behind the plan's catalogs, which never grow with the rows. */
    forkunion_allocator_t allocator_ {};

    /** The memory every step carves its blocks, batches and tables from. */
    bytes_span_t workspace_;

    /** The caller's configuration, its provenance borrowed until the plan finishes. */
    nvme_rewrite_config_t config_;

    /** The custom fields every target header carries. */
    growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields_ {allocator_};

    /** The manifest's path, and the stem its parts and spills are named after. */
    paths_t manifest_ {allocator_};
    paths_t stem_ {allocator_};

    /** The views a build opened itself; the sources point into them. */
    growing_buffer<indexes_t, forkunion_allocator_t> opened_ {allocator_};

    /** Every source part, flattened across the sources. */
    growing_buffer<source_t, forkunion_allocator_t> sources_ {allocator_};

    /** Every target part, in manifest order. */
    growing_buffer<target_t, forkunion_allocator_t> targets_ {allocator_};

    /** The target rows the re-select phase re-selects, in placement order across the target
     *  parts. */
    iota_view<std::size_t> selected_;

    /** The first row of the appended part of a merge or a build, every row below it resident and
     *  linking to no appended row yet; zero for every other rewrite. */
    std::size_t appended_base_ = 0;

    /** The key fences of a split by keys; empty for every other rewrite. */
    std::span<key_t const> fences_;

    /** Per source part and key range: the rows routed, then the first target row they land at. */
    growing_buffer<std::size_t, forkunion_allocator_t> routes_ {allocator_};

    /** The target part of every key range, all ones for an empty range. */
    growing_buffer<std::size_t, forkunion_allocator_t> range_targets_ {allocator_};

    /** The source part whose routing table the workspace holds, all ones for none. */
    std::size_t routed_ = ~std::size_t {0};

    /** Every spill bucket, those of each target part after every earlier part's. */
    growing_buffer<bucket_t, forkunion_allocator_t> buckets_ {allocator_};

    nvme_rewrite_phase_t phase_ = nvme_rewrite_phase_t::finish_k;

    result_t result_;
};

template <typename indexes_type_>
expected<nvme_rewrite_plan<indexes_type_>> plan_merge(std::span<indexes_type_ const* const> sources,
                                                      std::string_view target, bytes_span_t workspace,
                                                      nvme_rewrite_config_t const& config) noexcept {
    using plan_t = nvme_rewrite_plan<indexes_type_>;
    plan_t plan;
    if (sources.empty() || !sources[0]) return {std::move(plan), status_t::uninitialized_k};
    if (status_t const opened = plan.open_(target, workspace, config, *sources[0]); failed(opened))
        return {std::move(plan), opened};
    for (indexes_type_ const* source : sources)
        if (status_t const added = source ? plan.add_sources_(*source, *sources[0]) : status_t::invalid_argument_k;
            failed(added))
            return {std::move(plan), added};
    if (plan.sources_.empty()) return {std::move(plan), status_t::uninitialized_k};

    // The largest part stays resident and keeps its IDs; every other part lands after it with its
    // own edges rebased, so an appended row's walk navigates its own part and its old neighbors
    // compete with what the walk finds.
    std::size_t resident = 0, rows = 0;
    for (std::size_t part = 0; part != plan.sources_.size(); ++part) {
        if (plan.sources_[part].rows > plan.sources_[resident].rows) resident = part;
        rows += plan.sources_[part].rows;
    }
    if (rows >= node_id_unknown_k) return {std::move(plan), status_t::overflow_k};
    std::swap(plan.sources_[0], plan.sources_[resident]);
    for (std::size_t part = 0, base = 0; part != plan.sources_.size(); base += plan.sources_[part].rows, ++part) {
        plan.sources_[part].base = base;
        plan.sources_[part].kept = {0, plan.sources_[part].rows};
    }
    if (status_t const sized = plan.size_steps_(); failed(sized)) return {std::move(plan), sized};
    std::array<std::size_t, 1> const target_rows {rows};
    if (status_t const created = plan.open_targets_(target_rows); failed(created)) return {std::move(plan), created};
    plan.appended_base_ = plan.sources_[0].rows;
    plan.selected_ = {plan.appended_base_, rows};
    plan.phase_ = nvme_rewrite_phase_t::copy_k;
    return {std::move(plan), status_t::success_k};
}

template <typename indexes_type_, typename split_type_>
expected<nvme_rewrite_plan<indexes_type_>> plan_split(indexes_type_ const& source, split_type_ const& policy,
                                                      std::string_view target,
                                                      typename indexes_type_::executor_t& executor,
                                                      bytes_span_t workspace,
                                                      nvme_rewrite_config_t const& config) noexcept {
    using plan_t = nvme_rewrite_plan<indexes_type_>;
    using key_t = typename indexes_type_::key_t;
    constexpr bool by_rows_k = std::is_same_v<split_type_, split_by_rows_t>;
    static_assert(by_rows_k || std::is_same_v<split_type_, split_by_keys<key_t>>, "Split by rows or by keys");
    plan_t plan;
    if (!source.shards_count()) return {std::move(plan), status_t::uninitialized_k};
    if (status_t const opened = plan.open_(target, workspace, config, source); failed(opened))
        return {std::move(plan), opened};
    if (status_t const added = plan.add_sources_(source, source); failed(added)) return {std::move(plan), added};
    if (plan.sources_.empty()) return {std::move(plan), status_t::uninitialized_k};

    if constexpr (by_rows_k) {
        if (!policy.rows) return {std::move(plan), status_t::invalid_argument_k};
        growing_buffer<std::size_t, forkunion_allocator_t> rows {forkunion_allocator_t {}};
        for (auto& part : plan.sources_) {
            part.target = rows.size();
            part.stride = policy.rows;
            part.kept = {0, part.rows};
            for (std::size_t first = 0; first < part.rows; first += policy.rows)
                if (failed(rows.push_back(min_of(policy.rows, part.rows - first))))
                    return {std::move(plan), status_t::bad_alloc_k};
        }
        if (status_t const sized = plan.size_steps_(); failed(sized)) return {std::move(plan), sized};
        if (status_t const created = plan.open_targets_(to_span(rows)); failed(created))
            return {std::move(plan), created};
    }
    else {
        if (policy.fences.size() >= plan_t::max_key_ranges_k) return {std::move(plan), status_t::invalid_argument_k};
        for (std::size_t fence = 1; fence < policy.fences.size(); ++fence)
            if (policy.fences[fence] < policy.fences[fence - 1]) return {std::move(plan), status_t::invalid_argument_k};
        plan.fences_ = policy.fences;
        for (auto& part : plan.sources_) part.kept = {0, part.rows};
        if (status_t const sized = plan.size_steps_(); failed(sized)) return {std::move(plan), sized};
        if (status_t const routed = plan.route_(executor); failed(routed)) return {std::move(plan), routed};
    }
    // Every row of every target part re-selects.
    plan.selected_ = {0, source.size()};
    plan.phase_ = nvme_rewrite_phase_t::copy_k;
    return {std::move(plan), status_t::success_k};
}

template <typename indexes_type_>
expected<nvme_rewrite_plan<indexes_type_>> plan_refine(indexes_type_ const& source, iota_view<std::size_t> rows,
                                                       std::string_view target, bytes_span_t workspace,
                                                       nvme_rewrite_config_t const& config) noexcept {
    using plan_t = nvme_rewrite_plan<indexes_type_>;
    plan_t plan;
    if (!source.shards_count()) return {std::move(plan), status_t::uninitialized_k};
    if (status_t const opened = plan.open_(target, workspace, config, source); failed(opened))
        return {std::move(plan), opened};
    if (status_t const added = plan.add_sources_(source, source); failed(added)) return {std::move(plan), added};
    if (plan.sources_.empty()) return {std::move(plan), status_t::uninitialized_k};
    growing_buffer<std::size_t, forkunion_allocator_t> part_rows {forkunion_allocator_t {}};
    if (failed(part_rows.reserve_exactly(plan.sources_.size()))) return {std::move(plan), status_t::bad_alloc_k};
    for (std::size_t part = 0; part != plan.sources_.size(); ++part) {
        auto& one = plan.sources_[part];
        one.target = part;
        one.kept = {0, one.rows};
        part_rows.push_back_reserved(one.rows);
    }
    if (status_t const sized = plan.size_steps_(); failed(sized)) return {std::move(plan), sized};
    if (status_t const created = plan.open_targets_(to_span(part_rows)); failed(created))
        return {std::move(plan), created};
    // The range runs in placement order across the parts, so each part re-selects its own slice of
    // it.
    plan.selected_ = rows;
    plan.phase_ = nvme_rewrite_phase_t::copy_k;
    return {std::move(plan), status_t::success_k};
}

template <typename indexes_type_>
expected<nvme_rewrite_plan<indexes_type_>> plan_build(std::span<std::string_view const> sources,
                                                      std::string_view target, bytes_span_t workspace,
                                                      nvme_rewrite_config_t const& config) noexcept {
    using plan_t = nvme_rewrite_plan<indexes_type_>;
    plan_t plan;
    if (sources.empty()) return {std::move(plan), status_t::uninitialized_k};
    if (failed(plan.opened_.reserve_exactly(sources.size()))) return {std::move(plan), status_t::bad_alloc_k};
    for (std::string_view const path : sources) {
        plan.opened_.push_back_reserved(indexes_type_ {plan.allocator_});
        if (status_t const viewed = plan.opened_.back().view(path); failed(viewed)) return {std::move(plan), viewed};
    }
    if (status_t const opened = plan.open_(target, workspace, config, plan.opened_[0]); failed(opened))
        return {std::move(plan), opened};
    for (indexes_type_ const& view : plan.opened_)
        if (status_t const added = plan.add_sources_(view, plan.opened_[0]); failed(added))
            return {std::move(plan), added};
    std::size_t rows = 0;
    for (auto& part : plan.sources_) part.base = rows, rows += part.rows;
    if (!rows) return {std::move(plan), status_t::uninitialized_k};
    if (rows >= node_id_unknown_k) return {std::move(plan), status_t::overflow_k};
    std::size_t const block_rows = plan.build_rows_();
    if (!block_rows) return {std::move(plan), status_t::bad_alloc_k};
    if (status_t const sized = plan.size_steps_(); failed(sized)) return {std::move(plan), sized};
    std::array<std::size_t, 1> const target_rows {rows};
    if (status_t const created = plan.open_targets_(target_rows); failed(created)) return {std::move(plan), created};
    // The first block stays resident, and every later block links in by walks across as a merge's
    // appended part does.
    plan.appended_base_ = min_of(block_rows, rows);
    plan.selected_ = {plan.appended_base_, rows};
    plan.phase_ = nvme_rewrite_phase_t::copy_k;
    return {std::move(plan), status_t::success_k};
}

/** Drains @p planned over @p executor: every step of every phase in order, then the commit. */
template <typename plan_type_>
expected<nvme_rewrite_result_t> nvme_rewrite_drain_(expected<plan_type_>&& planned,
                                                    typename plan_type_::executor_t& executor) noexcept {
    if (!planned) return {{}, planned.status};
    plan_type_& plan = planned.value;
    do
        for (typename plan_type_::step_t const step : plan.steps())
            if (status_t const ran = plan.run(step, executor); failed(ran)) return {{}, ran};
    while (plan.advance());
    return plan.finish();
}

/** The one-shot @ref plan_merge, drained over @p executor. */
template <typename indexes_type_>
expected<nvme_rewrite_result_t> merge(std::span<indexes_type_ const* const> sources, std::string_view target,
                                      typename indexes_type_::executor_t& executor, bytes_span_t workspace,
                                      nvme_rewrite_config_t const& config = {}) noexcept {
    return nvme_rewrite_drain_(plan_merge(sources, target, workspace, config), executor);
}

/** The one-shot @ref plan_split, drained over @p executor. */
template <typename indexes_type_, typename split_type_>
expected<nvme_rewrite_result_t> split(indexes_type_ const& source, split_type_ const& policy, std::string_view target,
                                      typename indexes_type_::executor_t& executor, bytes_span_t workspace,
                                      nvme_rewrite_config_t const& config = {}) noexcept {
    return nvme_rewrite_drain_(plan_split(source, policy, target, executor, workspace, config), executor);
}

/** The one-shot @ref plan_refine, drained over @p executor. */
template <typename indexes_type_>
expected<nvme_rewrite_result_t> refine(indexes_type_ const& source, iota_view<std::size_t> rows,
                                       std::string_view target, typename indexes_type_::executor_t& executor,
                                       bytes_span_t workspace, nvme_rewrite_config_t const& config = {}) noexcept {
    return nvme_rewrite_drain_(plan_refine(source, rows, target, workspace, config), executor);
}

/** The one-shot @ref plan_build, drained over @p executor. */
template <typename indexes_type_>
expected<nvme_rewrite_result_t> build(std::span<std::string_view const> sources, std::string_view target,
                                      typename indexes_type_::executor_t& executor, bytes_span_t workspace,
                                      nvme_rewrite_config_t const& config = {}) noexcept {
    return nvme_rewrite_drain_(plan_build<indexes_type_>(sources, target, workspace, config), executor);
}

#pragma endregion Rewrites

} // namespace unum::usearch::dense
