/**
 *  @file include/usearch/dense/nvme_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The sharded index over SafeGraphs parts on block storage: every read verb of
 *      @c ram_indexes, walking every shard of a row on one thread; @c nvme_compaction.hpp
 *      rewrites its parts.
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
 */
#pragma once

#include <cstring> // `std::memcmp`, `std::memset`

#include <array>       // `std::array`
#include <bit>         // `std::popcount`, `std::countr_zero`
#include <chrono>      // `std::chrono::steady_clock`, for a scan's latencies
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::uninitialized_value_construct_n`
#include <mutex>       // `std::unique_lock`
#include <new>         // placement `new`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::is_same_v`, `std::remove_reference_t`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the default reference

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `seekable_file_t`, `file_span_t`
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`, `arena_allocator`, `ceil2`
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator_t`
#include "usearch/utilities/ram_sorts.hpp"         // `partial_sort`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_manifest`
#include "usearch/dense/types.hpp"
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
template <                                                                    //
    typename key_type_ = uuid_t,                                              //
    typename quads_kit_type_ = bf16_serial_quads_kit_t,                       //
    typename tiles_kit_type_ = no_tiles_kit_t,                                //
    typename elementwise_kit_type_ = serial_elementwise_kit_t,                //
    typename rmscast_kit_type_ = no_rmscast_kit_t,                            //
    typename visits_kit_type_ = serial_visits_hash_set_kit_t,                 //
    typename engine_type_ = default_io_engine_t,                              //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename wait_type_ = fu::standard_yield_t                                //
    >
struct nvme_indexes {
    friend struct nvme_indexes_cursor<nvme_indexes>;
    template <typename>
    friend struct nvme_compact_plan;

    /** A reader's walk over every part's rows in placement order. */
    using cursor_t = nvme_indexes_cursor<nvme_indexes>;

    /** The caller-visible identity, read from every part's key column. */
    using key_t = key_type_;

    /** The one-part index, whose walks every verb drives over all parts at once. */
    using index_t = nvme_index<quads_kit_type_, visits_kit_type_, engine_type_, atomic_ref_template_, wait_type_>;

    /** The CPU class's reference over the words one call's threads share, and its waiter. */
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;
    using wait_t = wait_type_;

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
     *      ends: one more than the queries.
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] config The walk configuration; each row nominates as many keys as its expansion
     *      factor, and a @c traversal_t::scan_k scan nominates every key.
     *  @param[in] wanted Keys each query keeps, the best ones; zero keeps every key it scored.
     *  @param[out] scored Keys each query scored, before @p wanted cut any, one per query.
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
                                     std::span<std::size_t> scored, land_type_&& land,
                                     std::span<u64_t> latencies = {}) noexcept {
        return maxsim_if(queries, query_offsets, executor, config, wanted, scored, land, dummy_predicate_t {},
                         latencies);
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
                                        std::span<std::size_t> scored, land_type_&& land, predicate_type_&& predicate,
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
            (!latencies.empty() && latencies.size() != rows_count) || scored.size() != queries_count)
            return {{}, status_t::config_mismatch_k};
        if (!queries_count) return {{}, status_t::success_k};
        std::memset(scored.data(), 0, scored.size_bytes());

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
            return scan_maxsim_(query_offsets, query_rows, to_span(query_norms), executor, wanted, scored, land,
                                predicate);
        return walk_maxsim_(queries, query_offsets, query_rows, to_span(query_norms), shape, executor, wanted, scored,
                            land, predicate, latencies);
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
            atomic_ref<std::size_t>(entries[position]).add(1, std::memory_order_relaxed);
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
                while (succeeded(local) && succeeded(atomic_ref<status_t>(failure).load(std::memory_order_relaxed))) {
                    if (issuing == depth) {
                        if (!free_count) break;
                        if (task == unclaimed_k)
                            task = min_of(atomic_ref<std::size_t>(cursor).fetch_add(1, std::memory_order_relaxed),
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
            atomic_ref<std::size_t>(issued.count).add(reads.count, std::memory_order_relaxed);
            atomic_ref<std::size_t>(issued.bytes).add(reads.bytes, std::memory_order_relaxed);
            if (succeeded(local)) return;
            status_t unset = status_t::success_k;
            atomic_ref<status_t>(failure).compare_exchange_strong(unset, local, std::memory_order_relaxed,
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
                                           std::span<std::size_t> scored, land_type_& land, predicate_type_& predicate,
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
                atomic_ref<std::size_t>(row_starts[position]).add(1, std::memory_order_relaxed);
                if (failed(matches[thread].push_back({locator, position})))
                    atomic_ref<status_t>(listing).store(status_t::bad_alloc_k, std::memory_order_relaxed);
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
            fold_quads<quads_kit_t>(
                row, index_t::norms_(kernel_, row), query_rows(nk::range(first, first + rows), nk::slice),
                query_norms.subspan(first, rows),
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
        status_t landing = status_t::success_k;
        status_t const ranking = executor.for_tasks(queries_count, [&](std::size_t query, std::size_t thread,
                                                                       std::size_t) noexcept {
            std::size_t const begin = starts[query], end = starts[query + 1];
            std::size_t const rows = query_offsets[query + 1] - query_offsets[query];
            scored[query] = end - begin;
            nk::matrix_span<dot_result_t> const query_maxima {maxima.data() + maxima_starts[query], end - begin, rows};
            if (status_t const landed = rank_candidates_(nominations.subspan(begin, end - begin), query_maxima,
                                                         to_writable_span(ranked).subspan(thread * capacity, capacity),
                                                         thread, query, land);
                failed(landed))
                atomic_ref<status_t>(landing).store(landed, std::memory_order_relaxed);
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
                                           std::size_t wanted, std::span<std::size_t> scored, land_type_& land,
                                           predicate_type_& predicate) noexcept {
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
                fold_quads<quads_kit_t>(values, index_t::norms_(kernel_, values), query_rows, query_norms,
                                        to_span(maxima.row(position).as_vector()),
                                        [](dot_result_t& maximum, dot_result_t dot) noexcept {
                                            atomic_ref<dot_result_t> const shared {maximum};
                                            if (shared.load(std::memory_order_relaxed) < dot)
                                                shared.max(dot, std::memory_order_relaxed);
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
        status_t landing = status_t::success_k;
        status_t const ranking = executor.for_tasks(
            queries_count, [&](std::size_t query, std::size_t thread, std::size_t) noexcept {
                std::size_t const first = query_offsets[query] - query_offsets.front();
                std::size_t const rows = query_offsets[query + 1] - query_offsets[query];
                scored[query] = admitted;
                if (status_t const landed = rank_candidates_(
                        to_span(candidates).first(admitted), maxima(nk::all, nk::range(first, first + rows), nk::slice),
                        to_writable_span(ranked).subspan(thread * capacity, capacity), thread, query, land);
                    failed(landed))
                    atomic_ref<status_t>(landing).store(landed, std::memory_order_relaxed);
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
            return atomic_ref<u64_t>(next[word]).load(std::memory_order_relaxed);
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
                atomic_ref<u64_t>(next[node / 64]).set_bits(lane, std::memory_order_relaxed);
                auto const* const neighbors = reinterpret_cast<node_id_t const*>(bytes + row * row_bytes);
                for (std::size_t slot = 0; slot != part.connectivity; ++slot) {
                    node_id_t const neighbor = neighbors[slot];
                    u64_t const reached = u64_t {1} << (neighbor % 64);
                    if (neighbor >= nodes || (current[neighbor / 64] | later(neighbor / 64)) & reached) continue;
                    atomic_ref<u64_t>(next[neighbor / 64]).set_bits(reached, std::memory_order_relaxed);
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

} // namespace unum::usearch::dense
