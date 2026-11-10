/**
 *  @file include/usearch/dense/nvme_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The sharded index over SafeGraphs parts on block storage: every read verb of
 *      @c ram_indexes, walking every shard of a row on one thread; @c nvme_compaction.hpp
 *      rewrites its parts.
 *
 *  A view opens the manifest or the monolith a RAM or VRAM index saved, keeps every part's medium
 *  and column offsets, and per node only the key table @c hold_keys builds within its limit.
 *  Every verb borrows an executor and attaches the parts' media to its engines for the call alone,
 *  so any executor reaching the same devices serves any view. Walks rank rows by shard and node,
 *  and only the rows a verb writes resolve into keys, one key read each, once the row's walks
 *  finished; a filtered search and attention read every scored row's key beside its value as well.
 *  Key lookups probe that table and confirm each candidate with one key read, and refuse without
 *  one; the cursor scans the key and value columns, and the health audit the neighbors columns.
 *  A shard is the position a verb scatters over, and a part the SafeGraphs file behind it.
 */
#pragma once

#include <cstring> // `std::memcmp`, `std::memset`

#include <array>       // `std::array`
#include <bit>         // `std::bit_ceil`
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
#include "usearch/utilities/nvme_walks.hpp"        // `nvme_sample_rows` - what SAMPLE reads beside its scores
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`, `arena_allocator`, `ceil2`
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator_t`
#include "usearch/utilities/ram_sorts.hpp"         // `partial_sort`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_manifest`
#include "usearch/utilities/sample.hpp"            // `sample_plan` - the SAMPLE verb's roots, paths & regions
#include "usearch/dense/types.hpp"
#include "usearch/dense/nvme_index.hpp"

namespace unum::usearch::dense {

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
    typename key_type_ = uuid128_t,                                           //
    typename quads_kit_type_ = bf16_serial_quads_kit_t,                       //
    typename tiles_kit_type_ = no_tiles_kit_t,                                //
    typename elementwise_kit_type_ = serial_elementwise_kit_t,                //
    typename rmscast_kit_type_ = no_rmscast_kit_t,                            //
    typename visits_kit_type_ = serial_visits_hash_set_kit_t,                 //
    typename engine_type_ = default_io_engine<dummy_waiter_t>,                //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename wait_type_ = dummy_waiter_t                                      //
    >
struct nvme_indexes {
    template <typename>
    friend struct nvme_compact_plan;

    /** A reader's walk over every part's entries, as stored or breadth-first. */
    using cursor_t = nvme_parts_cursor<key_type_, dense::node_id_t>;

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

    /** A MaxSim's knobs: its walks', and the keys each query keeps, zero keeping every key it
     *  scored. */
    struct maxsim_config_t {
        search_config_t search {};
        std::size_t wanted = 0;
    };
    using reachability_t = dense::reachability_t;

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
        : media_(allocator), parts_(allocator), allocator_(allocator), resident_keys_(allocator) {}
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
     *      transport ID naming a monolith on a namespace, with the optional keys @c ns and
     *      @c base.
     *  @param[in] paths Where to serve each part from instead, one per part: a directory holding
     *      the part's file, or a file, device or transport ID holding the part itself; empty serves
     *      every part where the manifest points.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k @p path is empty, or it or a part's path does not parse or names a
     *    base off the medium's block; in SPDK builds also an inactive namespace or an unparsable
     *    transport ID.
     *  - @c malformed_k The manifest names no part, a manifest or header does not parse, or a
     *    part's columns have no width or too many rows.
     *  - @c config_mismatch_k @p paths does not name one place per part, a part names another
     *    metric or geometry than the first, its columns are not fixed rows of this scalar with
     *    @c u32 neighbors and keys this index's keys widen from, or in SPDK builds an SPDK
     *    environment this process cannot initialize.
     *  - @c unsupported_k A transport ID in a build without SPDK.
     *  - @c unknown_kind_k A kernel the dtype's kits cannot finalize.
     */
    status_t view(std::string_view path, std::span<std::string_view const> paths = {}) noexcept {
        release();
        status_t const viewed = view_(path, paths);
        if (failed(viewed)) release();
        return viewed;
    }

    /** Empties the view, giving back the controllers its SPDK namespaces hold, which must precede
     *  its death. */
    void release() noexcept {
        for (medium_t& medium : media_) release_medium<wait_t>(medium);
        media_.clear();
        parts_.clear();
        resident_keys_.reset();
        kernel_ = nk_kernel_unknown_k;
    }

    /**
     *  @brief Builds the key table in @p allocator 's domain within @p memory_limit bytes, zero
     *      setting no ceiling, scanning every part's key column once.
     *
     *  Without a key table, the key lookups of @ref count, @ref contains, @ref get,
     *  @ref find_first, @ref sample, a MaxSim and hinted walks refuse with @c unsupported_k rather
     *  than scan every key. With one, each looked-up key probes it and confirms every candidate
     *  with one key read.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c bad_alloc_k The key table passes @p memory_limit or cannot be allocated; the view then
     *    holds none.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c config_mismatch_k A medium the engines cannot read.
     */
    status_t hold_keys(forkunion_allocator_t allocator, std::size_t memory_limit, executor_t& executor) noexcept {
        resident_keys_ = resident_keys_t {allocator};
        if (parts_.empty()) return status_t::uninitialized_k;
        if (memory_limit && resident_keys_t::memory_usage_for(to_span(parts_)) > memory_limit)
            return status_t::bad_alloc_k;
        if (status_t const reserved = resident_keys_.reserve(to_span(parts_)); failed(reserved)) return reserved;
        auto [call, call_status] = begin_scans_(executor, 1);
        status_t const scanned = failed(call_status) ? call_status
                                                     : scan_key_rows_(executor, [&](std::size_t, key_t const& key,
                                                                                    shard_node_t locator) noexcept {
                                                           resident_keys_.insert(locator.shard, key, locator.node);
                                                       });
        if (failed(scanned)) resident_keys_.reset();
        return scanned;
    }

    /** The distance kind the parts were built for, as @ref view read it; unknown before one. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Whether the kernel reads rows' norms, which a rewrite's targets then carry as a column. */
    bool stores_norms() const noexcept { return index_t::stores_norms_(kernel_); }

    /**
     *  The checkpoint's metadata without viewing it, @ref nvme_read_metadata.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k @p path, or a manifest's first part path, does not parse or names an
     *    unusable base or namespace.
     *  - @c unsupported_k A transport ID in a build without SPDK.
     *  - @c malformed_k A transport ID starting no monolith, or a manifest or header that does not
     *    parse.
     */
    static expected<safegraphs_metadata_t> metadata(std::string_view path) noexcept {
        return nvme_read_metadata<wait_t>(path);
    }

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

    /** Bytes the index occupies: the catalog of parts and the key table it holds. */
    std::size_t memory_usage() const noexcept {
        return sizeof(nvme_indexes) + media_.memory_usage() + parts_.memory_usage() + resident_keys_.memory_usage();
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
     *  @param[out] matches Merged rankings, one query per row, as many matches as its width: the
     *      keys, their distances as @p score_type_, and per-query merged hit counts.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *  @param[out] latencies Steady-clock nanoseconds per query row, from its first primed walk to
     *      its final write; empty collects none.
     *  @param[in] hints Optional keys per query whose rows every shard's walk takes as landmarks
     *      ahead of its draws, in hint order; a scan ignores them.
     *
     *  A query takes as many hint keys as a walk draws landmarks, the connectivity rounded up to
     *  whole quads. Each walk probes the key table @ref hold_keys built and confirms its
     *  candidates with one key read each, and a key no shard holds counts as a missing hint.
     *
     *  A failed read stops the call: the engines drain, every buffer returns, and the call answers
     *  @c io_k.
     *
     *  @par Statuses:
     *  - @c unsupported_k A walk handed @p hints while no key table is held.
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c config_mismatch_k @p matches, @p latencies or non-empty @p hints disagree with the
     *    batch or each other, the queries are not as wide as the rows, or a medium the engines
     *    cannot read.
     *  - @c unsupported_layout_k @p matches are not packed along their rows.
     *  - @c shape_mismatch_k A query counts more @p hints than they are wide.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    template <typename score_type_>
    expected<search_result_t> search(rows_view_t queries, ranked_matches_span<key_t, score_type_> matches,
                                     search_config_t config, executor_t& executor, std::span<u64_t> latencies = {},
                                     ranked_hints_view<key_t> hints = {}) noexcept {
        return search_if(queries, matches, config, dummy_predicate_t {}, executor, latencies, hints);
    }

    /**
     *  @brief The batched @ref search under a lightning indexer, sharing its whole contract: every
     *      distance is 1 − its key's indexer score against the heads of its query row.
     *
     *  @param[in] queries The batch, as the batched @ref search takes it.
     *  @param[in] heads The indexer heads of every query row.
     *  @param[out] matches Rankings by key, with their distances and counts, as the batched @ref
     *      search writes them.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in,out] executor The threads, as the batched @ref search takes them.
     *  @param[out] latencies Steady-clock nanoseconds per query row, as the batched @ref search
     *      collects them.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The parts do not rank by dot.
     */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads,
                                     ranked_matches_span<key_t, distance_t> matches, search_config_t config,
                                     executor_t& executor, std::span<u64_t> latencies = {}) noexcept
        requires floating_scalar<scalar_t>
    {
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        dummy_predicate_t predicate;
        return search_rows_(queries, heads, executor, config, matches.keys, matches.scores, matches.counts, predicate,
                            latencies, {});
    }

    /**
     *  @brief The filtered sibling of the batched @ref search, sharing its whole contract; a
     *      @c traversal_t::scan_k scan ranks every row exactly instead, as @c scan_rows_ reads
     *      them.
     *
     *  @param[in] queries The batch, as the batched @ref search takes it.
     *  @param[out] matches Rankings by key, with their distances and counts, as the batched @ref
     *      search writes them.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in] predicate The per-key admission filter; every scored row's key is read beside its
     *      value, so the predicate sees keys, never node IDs. A scan asks it only about rows closer
     *      than the worst match some query already holds.
     *  @param[in,out] executor The threads, as the batched @ref search takes them.
     *  @param[out] latencies Steady-clock nanoseconds per query row, as the batched @ref search
     *      collects them.
     *  @param[in] hints Optional keys per query seeding every walk, as in the batched @ref search.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c config_mismatch_k @p matches, @p latencies or non-empty @p hints disagree with the
     *    batch or each other, the queries are not as wide as the rows, or a medium the engines
     *    cannot read.
     *  - @c unsupported_layout_k @p matches are not packed along their rows.
     *  - @c shape_mismatch_k A query counts more @p hints than they are wide.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> search_if(rows_view_t queries, ranked_matches_span<key_t, score_type_> matches,
                                        search_config_t config, predicate_type_&& predicate, executor_t& executor,
                                        std::span<u64_t> latencies = {}, ranked_hints_view<key_t> hints = {}) noexcept {
        static_assert((!aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, walk_payload_t> &&
                       !aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, scan_payload_t>) ||
                          per_thread_predicates<std::remove_cvref_t<predicate_type_>>,
                      "An engine driving its own threads takes an aggregating predicate per thread");
        if (config.traversal == traversal_t::scan_k)
            return scan_rows_(queries, executor, matches.keys, matches.scores, matches.counts, predicate, latencies);
        return search_rows_(queries, dummy_heads_t {}, executor, config, matches.keys, matches.scores, matches.counts,
                            predicate, latencies, hints);
    }

    /**
     *  @brief The filtered sibling for one aggregating predicate per thread, each gathering a
     *      walk's @c walk_payload_t, so it serves walks alone.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p config asks for a scan, @p matches or
     *    @p latencies disagree with the batch or each other, the queries are not as wide as the
     *    rows, or a medium the engines cannot read.
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c unsupported_layout_k @p matches are not packed along their rows.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    template <typename score_type_, typename predicate_type_>
        requires aggregating_predicate<typename std::remove_cvref_t<predicate_type_>::thread_predicate_t, key_t,
                                       walk_payload_t>
    expected<search_result_t> search_if(rows_view_t queries, ranked_matches_span<key_t, score_type_> matches,
                                        search_config_t config, predicate_type_&& predicate, executor_t& executor,
                                        std::span<u64_t> latencies = {}, ranked_hints_view<key_t> hints = {}) noexcept {
        if (config.traversal == traversal_t::scan_k) return {{}, status_t::config_mismatch_k};
        return search_rows_(queries, dummy_heads_t {}, executor, config, matches.keys, matches.scores, matches.counts,
                            predicate, latencies, hints);
    }

    /**
     *  @brief The filtered sibling for one aggregating predicate per thread, each gathering a
     *      scan's @c scan_payload_t, so it serves scans alone.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p config asks for a walk, @p matches or
     *    @p latencies disagree with the batch or each other, the queries are not as wide as the
     *    rows, or a medium the engines cannot read.
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c unsupported_layout_k @p matches are not packed along their rows.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    template <typename score_type_, typename predicate_type_>
        requires aggregating_predicate<typename std::remove_cvref_t<predicate_type_>::thread_predicate_t, key_t,
                                       scan_payload_t>
    expected<search_result_t> search_if(rows_view_t queries, ranked_matches_span<key_t, score_type_> matches,
                                        search_config_t config, predicate_type_&& predicate, executor_t& executor,
                                        std::span<u64_t> latencies = {}, ranked_hints_view<key_t> = {}) noexcept {
        if (config.traversal != traversal_t::scan_k) return {{}, status_t::config_mismatch_k};
        return scan_rows_(queries, executor, matches.keys, matches.scores, matches.counts, predicate, latencies);
    }

    /**
     *  @brief Graph products of a whole matrix of left-hand rows against every shard: each row
     *      keeps the pairs its walks scored, keys resolved and dots cast into the products' scalar.
     *
     *  @param[in] left The left-hand rows, one walk per row and shard, in any scalar NumKong casts
     *      into the stored one; every row slot imports its own.
     *  @param[out] products The tape the kept pairs land on, one run per left-hand row from its
     *      offset, each up to the tape's stride of pairs: each pair's key and its dot as
     *      @p product_type_. Each row's slot stages its run at the row's stride, and the offsets
     *      publish once every row is done.
     *  @param[out] scored The pairs scored per row across every shard, above the stride when
     *      truncated.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *  @param[out] latencies Steady-clock nanoseconds per left-hand row, as @ref search collects
     *      them.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The parts do not rank by dot, or @p scored or the tape disagree
     *    with the batch.
     */
    template <typename product_type_>
    expected<search_result_t> multiply(rows_view_t left, products_tape<key_t, product_type_> products,
                                       std::span<std::size_t> scored, search_config_t config, executor_t& executor,
                                       std::span<u64_t> latencies = {}) noexcept {
        return multiply_if(left, products, scored, config, dummy_predicate_t {}, executor, latencies);
    }

    /**
     *  @brief The filtered sibling of @ref multiply, sharing its whole contract: only pairs whose
     *      key @p predicate admits are written and counted, while the walks route through the rest.
     *
     *  @param[in] left The left-hand rows, as @ref multiply takes them.
     *  @param[out] products The tape of kept pairs, as @ref multiply writes it.
     *  @param[out] scored The pairs scored per row, as @ref multiply counts them.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in] predicate The per-key admission filter, asked about a scored quad of keys at a
     *      time; every scored row's key is read beside its value, and written pairs keep it.
     *  @param[in,out] executor The threads, as @ref multiply takes them.
     *  @param[out] latencies Steady-clock nanoseconds per left-hand row, as @ref multiply collects
     *      them.
     *
     *  @par Statuses:
     *  - @c unsupported_k A @c traversal_t::scan_k scan, since no exact product scan exists.
     */
    template <typename product_type_, typename predicate_type_>
    expected<search_result_t> multiply_if(rows_view_t left, products_tape<key_t, product_type_> products,
                                          std::span<std::size_t> scored, search_config_t config,
                                          predicate_type_&& predicate, executor_t& executor,
                                          std::span<u64_t> latencies = {}) noexcept {
        std::size_t const rows_count = left.rows;
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        if (status_t const refusal = products.refusal(rows_count, scored); failed(refusal)) return {{}, refusal};
        if (rows_count && left.columns != dimensions()) return {{}, status_t::config_mismatch_k};
        using query_t = multiply_query<product_type_, std::remove_reference_t<predicate_type_>>;
        typename query_t::inputs_t const inputs {left, &predicate, products, scored};
        expected<search_result_t> const multiplied = walk_queries_<query_t>(
            rows_count, config, inputs, staged_rows_bytes(vector_dtype_k, 1, left.columns), scored, latencies,
            executor);
        if (multiplied) products.publish_runs(scored);
        return multiplied;
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
     *  @param[out] scored Keys each query scored, before @c wanted cut any, one per query.
     *  @param[out] matches The tape each query's kept keys land on in rank order, largest total
     *      first, from its offset, their totals as @p score_type_; the tape's stride bounds the
     *      ranking of every query.
     *  @param[in] config The walk configuration, whose @c search nominates as many keys per row as
     *      its expansion factor, a @c traversal_t::scan_k scan nominating every key, and whose @c
     *      wanted keys each query keeps, the best ones, zero keeping every key it scored.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *  @param[out] latencies Steady-clock nanoseconds per query row of a walk, as @ref search
     *      collects them; normalization follows the last row.
     *  @param[in] hints Optional keys per query whose rows every row's walk takes as landmarks
     *      ahead of its draws, as @ref search takes them, a key no shard holds counted once per
     *      query; a scan ignores them.
     *
     *  The thread ranking a query stages its run at the query's stride, and the offsets publish
     *  once every query is done.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The parts do not rank by dot.
     */
    template <typename score_type_>
    expected<search_result_t> maxsim(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                     std::span<std::size_t> scored, products_tape<key_t, score_type_> matches,
                                     maxsim_config_t config, executor_t& executor, std::span<u64_t> latencies = {},
                                     ranked_hints_view<key_t> hints = {}) noexcept {
        return maxsim_if(queries, query_offsets, scored, matches, config, dummy_predicate_t {}, executor, latencies,
                         hints);
    }

    /**
     *  @brief The filtered sibling of @ref maxsim, sharing its whole contract: a key @p predicate
     *      rejects is never nominated, so it adds to no score.
     *
     *  @param[in] queries The token rows of every query, laid end to end.
     *  @param[in] query_offsets Where every query's rows start in @p queries, then where the last
     *      one ends.
     *  @param[out] scored Keys each query scored, before @c wanted cut any, one per query.
     *  @param[out] matches The tape of kept keys, as @ref maxsim writes it.
     *  @param[in] maxsim_config The per-shard walk configuration, applied uniformly, and the keys
     *      each query keeps, as @ref maxsim takes them.
     *  @param[in] predicate The per-key admission filter: a walk asks it as the filtered
     *      @ref search_if asks it, a scan once per distinct key.
     *  @param[in,out] executor The threads, as @ref maxsim takes them.
     *  @param[out] latencies Steady-clock nanoseconds per query, as @ref maxsim collects them.
     *  @param[in] hints Optional keys per query seeding every row's walk, as in @ref maxsim.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p query_offsets is empty, or non-empty @p hints differ in length
     *    from the queries.
     *  - @c shape_mismatch_k A query counts more @p hints than they are wide.
     *  - @c unsupported_k A walk handed @p hints while no key table is held, or a second pass's key
     *    lookup that neither a table nor @ref view 's scans serve.
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c config_mismatch_k The parts do not rank by dot.
     *  - @c shape_mismatch_k A query with no rows.
     *  - @c config_mismatch_k @p query_offsets run past @p queries, the queries are not as wide as
     *    the rows, @p latencies, @p scored or the tape disagree with the batch, a query keeps more
     *    than the tape's stride, or a medium the engines cannot read.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> maxsim_if(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                        std::span<std::size_t> scored, products_tape<key_t, score_type_> matches,
                                        maxsim_config_t maxsim_config, predicate_type_&& predicate,
                                        executor_t& executor, std::span<u64_t> latencies = {},
                                        ranked_hints_view<key_t> hints = {}) noexcept {
        search_config_t const config = maxsim_config.search;
        std::size_t const wanted = maxsim_config.wanted;
        static_assert(!aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, walk_payload_t> ||
                          per_thread_predicates<std::remove_cvref_t<predicate_type_>>,
                      "An engine driving its own threads takes an aggregating predicate per thread");
        using slice_t = arena_allocator<byte_t>;
        using query_t = search_query<dummy_heads_t, std::remove_reference_t<predicate_type_>, distance_t>;
        if (query_offsets.empty()) return {{}, status_t::config_mismatch_k};
        std::size_t const queries_count = query_offsets.size() - 1, threads = executor.size();
        if (status_t const refusal = hints_refusal(hints, queries_count); failed(refusal)) return {{}, refusal};
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (!hints.counts.empty() && resident_keys_.empty() && config.traversal == traversal_t::walk_k)
            return {{}, status_t::unsupported_k};
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        for (std::size_t query = 0; query != queries_count; ++query)
            if (query_offsets[query + 1] <= query_offsets[query]) return {{}, status_t::shape_mismatch_k};
        std::size_t const rows_count = query_offsets.back() - query_offsets.front();
        if (query_offsets.back() > queries.rows || (rows_count && queries.columns != dimensions()) ||
            (!latencies.empty() && latencies.size() != rows_count))
            return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = matches.refusal(queries_count, scored); failed(refusal)) return {{}, refusal};
        if (!queries_count) return {{}, status_t::success_k};
        std::memset(scored.data(), 0, scored.size_bytes());

        // One call serves every pass: a walk's engines and arenas, then two scans, and the hints.
        bool const scanning = config.traversal == traversal_t::scan_k;
        walk_shape_t const shape = scanning ? walk_shape_t {}
                                            : index_t::template walk_shape_<query_t>(
                                                  to_span(media_), to_span(parts_), kernel_,
                                                  nvme_call_depth(executor, to_span(media_), to_span(parts_)), threads,
                                                  rows_count, non_zero_expansion_factor(config),
                                                  query_t::staged_bytes(queries.columns, 0));
        std::size_t const staged_bytes = staged_rows_bytes(vector_dtype_k, rows_count, queries.columns);
        std::size_t const shared_bytes = staged_bytes + slice_t::reserved_bytes(rows_count * sizeof(vector_norms_t)) +
                                         slice_t::reserved_bytes(threads * wanted * sizeof(maxsim_ranked<total_t>));
        auto [call, call_status] = scanning ? begin_scans_<2>(executor, 2, shared_bytes)
                                            : begin_scans_(executor, hints.counts.empty() ? 2 : 3,
                                                           shared_bytes + shape.scratch_bytes, shape.buffers);
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
        expected<search_result_t> const ranked = scanning
                                                     ? scan_maxsim_(query_offsets, query_rows, to_span(query_norms),
                                                                    executor, wanted, scored, matches, predicate)
                                                     : walk_maxsim_(queries, query_offsets, query_rows,
                                                                    to_span(query_norms), shape, executor, wanted,
                                                                    scored, matches, predicate, latencies, hints);
        if (ranked) matches.publish_runs(scored);
        return ranked;
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
     *      keys and stored like them:
     *      <tt>void(std::span&lt;key_t const&gt;, std::span&lt;vector_view_t&gt;)</tt>, one row per
     *      key. Keys are read beside the scored rows; one without a value row refuses with
     *      @c shape_mismatch_k.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys. Packed @c f32
     *      rows accumulate in place, any others in the row slot's staging until the token exports
     *      them.
     *  @param[out] carry Every query row's softmax state, which merges this call with other
     *      branches, or empty to keep none.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *  @param[out] latencies Steady-clock nanoseconds per token, its heads walked together, as
     *      @ref search collects them.
     *  @return The query rows attended in @c count.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The parts do not rank by dot.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     search_config_t config, executor_t& executor,
                                     std::span<u64_t> latencies = {}) noexcept {
        return attend_if(queries, head_count, value_lookup, score_scale, outputs, carry, config, dummy_predicate_t {},
                         executor, latencies);
    }

    /**
     *  @brief The filtered sibling of @ref attend, sharing its whole contract: a key @p predicate
     *      rejects still routes the walks, but folds into no head and resolves no value row.
     *
     *  @param[in] queries Row t · @p head_count + h is head h of token t.
     *  @param[in] head_count Query heads sharing this key-value head; it must divide the rows.
     *  @param[in] value_lookup Resolves a scored batch's keys to their value rows, as @ref attend
     *      takes it.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row.
     *  @param[out] carry Every query row's softmax state.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in] predicate The per-key admission filter, asked about a scored quad of the keys
     *      read beside the values at a time.
     *  @param[in,out] executor The threads, as @ref attend takes them.
     *  @param[out] latencies Steady-clock nanoseconds per token, as @ref attend collects them.
     *
     *  @par Statuses:
     *  - @c unsupported_k A @c traversal_t::scan_k scan, since no exact attention scan exists.
     */
    template <typename value_lookup_type_, typename predicate_type_>
    expected<search_result_t> attend_if(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                        f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                        search_config_t config, predicate_type_&& predicate, executor_t& executor,
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
        using query_t = attend_query<value_lookup_type_, std::remove_reference_t<predicate_type_>>;
        typename query_t::inputs_t const inputs {queries,     head_count, &value_lookup, &predicate,
                                                 score_scale, outputs,    carry};
        expected<search_result_t> attended = walk_queries_<query_t>(
            rows_count / head_count, config, inputs, query_t::staged_bytes(inputs), {}, latencies, executor);
        if (attended) attended.value.count = rows_count;
        return attended;
    }

#pragma endregion Search

#pragma region Key Scans

    /**
     *  @brief The entries stored under every key of @p keys, looked up as @ref view allows: probed
     *      in the key table, or matched in one scan over every part's key column.
     *
     *  @param[in] keys The keys to count, repeats and absent ones included.
     *  @param[out] entries The entries under each key, parallel to @p keys; zero means absent.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p entries and @p keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c unsupported_k No key table is held.
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        if (parts_.empty()) return status_t::uninitialized_k;
        std::memset(entries.data(), 0, entries.size_bytes());
        if (keys.empty()) return status_t::success_k;
        auto [call, call_status] = begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        return lookup_keys_(keys, executor, [&](std::size_t, std::size_t position, shard_node_t) noexcept {
            atomic_ref<std::size_t>(entries[position]).store_add(1, std::memory_order_relaxed);
        });
    }

    /**
     *  The batched @ref count reduced to one in @p entries where a key of @p keys holds any entry,
     *  zero where it holds none.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p entries and @p keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    status_t contains(std::span<key_t const> keys, std::span<std::size_t> entries,
                      executor_t& executor) const noexcept {
        if (status_t const counted = count(keys, entries, executor); failed(counted)) return counted;
        for (std::size_t& entry : entries) entry = entry ? 1 : 0;
        return status_t::success_k;
    }

    /**
     *  @brief Up to @p wanted rows of every key of @p keys, and every key's true total: one lookup
     *      as @ref count takes it, then one value-row read per kept row.
     *
     *  @param[in] keys The keys to export, repeats and absent ones included.
     *  @param[out] vectors @b [keys·wanted,dimensions] in any scalar NumKong casts into, where row
     *      q · @p wanted + i is the i-th row of key q in shard and row order, a key holding more
     *      keeping its earliest; refused with @c config_mismatch_k when the rows are another count
     *      or narrower than the index, while wider ones keep their tails.
     *  @param[out] entries The true total under each key, parallel to @p keys; above @p wanted when
     *      its rows truncated, zero when absent.
     *  @param[in] wanted Rows kept per key.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p entries and @p keys differ in length, @p vectors hold another
     *    count of rows or narrower ones, or a medium the engines cannot read.
     *  - @c unsupported_k No key table is held.
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    status_t get(std::span<key_t const> keys, rows_span_t vectors, std::span<std::size_t> entries, std::size_t wanted,
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
        growing_buffer<shard_node_t, scratch_allocator_t> kept {scratch_allocator_t {executor.arena(0)}};
        if (failed(kept.resize_uninitialized(keys.size() * wanted))) return status_t::bad_alloc_k;
        std::size_t const stripes_count = executor.size();
        auto* const stripes = reinterpret_cast<fu::spin_mutex_t*>(
            executor.arena(0).allocate(stripes_count * sizeof(fu::spin_mutex_t), alignof(fu::spin_mutex_t)));
        if (!stripes) return status_t::bad_alloc_k;
        // nvcc mislowers a bare over-aligned placement `new`, so a standard template builds them.
        std::uninitialized_value_construct_n(stripes, stripes_count);
        status_t const scanned = lookup_keys_(
            keys, executor, [&](std::size_t, std::size_t position, shard_node_t locator) noexcept {
                std::unique_lock<fu::spin_mutex_t> const held {stripes[position % stripes_count]};
                shard_node_t* const rows = kept.data() + position * wanted;
                std::size_t slot = min_of(entries[position]++, wanted);
                if (slot == wanted && (!wanted || !(locator < rows[wanted - 1]))) return;
                if (slot == wanted) --slot;
                for (; slot && locator < rows[slot - 1]; --slot) rows[slot] = rows[slot - 1];
                rows[slot] = locator;
            });
        if (failed(scanned)) return scanned;
        auto describe = [&](std::size_t read, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
            std::size_t const position = read / wanted, rank = read % wanted;
            if (progress++ || rank >= min_of(entries[position], wanted)) return {};
            shard_node_t const locator = kept[read];
            return {extent_of_(locator, &nvme_part_t::values_begin, parts_[locator.shard].value_row_bytes)};
        };
        rows_span_t const target {vectors.data, rows_count, dimensions(), vectors.stride, vectors.dtype};
        auto consume = [&](std::size_t, std::size_t read, u64_t, auto const&, auto const& fetched) noexcept {
            export_row<elementwise_kit_t>(fetched[0], vector_dtype_k, target, read);
        };
        return scan_reads_(executor, rows_count, describe, consume).status;
    }

    /**
     *  @brief Copies the keys of @p keys.size() entries from entry @p first on, numbered across the
     *      parts in part and node order, into @p keys, and their vectors into @p vectors.
     *
     *  The vectors export in the scalar of @p vectors, unless it holds no rows. Each column scans
     *  in runs of whole rows, one covering read per run. The result's @c count is the entries
     *  written, fewer once the parts end, and its @c scan_reads what they cost.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p vectors hold fewer rows than @p keys or narrower ones than the
     *    index, or a medium the engines cannot read.
     *  - @c uninitialized_k @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    expected<search_result_t> read_entries(std::size_t first, std::span<key_t> keys, rows_span_t vectors,
                                           executor_t& executor) const noexcept {
        if (vectors.rows && (vectors.rows < keys.size() || vectors.columns < dimensions()))
            return {{}, status_t::config_mismatch_k};
        std::size_t const count = min_of(keys.size(), size() - min_of(first, size()));
        if (!count) return {{}, status_t::success_k};
        std::size_t const key_runs = runs_count_(first, count, key_column_k, staging_bytes_());
        std::size_t const value_runs = vectors.rows ? runs_count_(first, count, value_column_k, staging_bytes_()) : 0;
        rows_span_t const target {vectors.data, vectors.rows, dimensions(), vectors.stride, vectors.dtype};
        auto [call, call_status] = begin_scans_(executor, 1);
        if (failed(call_status)) return {{}, call_status};
        // Tasks below `key_runs` read keys, the rest read values.
        auto run_of = [&](std::size_t task) noexcept {
            return task < key_runs ? locate_run_(first, count, task, key_column_k, staging_bytes_())
                                   : locate_run_(first, count, task - key_runs, value_column_k, staging_bytes_());
        };
        auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
            if (progress++) return {};
            return {run_extent_(run_of(task), task < key_runs ? key_column_k : value_column_k)};
        };
        auto consume = [&](std::size_t, std::size_t task, u64_t, std::array<nvme_extent_t, 1> const& extents,
                           std::array<byte_t const*, 1> const& fetched) noexcept {
            run_t const run = run_of(task);
            std::size_t const row_bytes = extents[0].length / run.rows;
            dtype_t const key_dtype = parts_[extents[0].medium].key_dtype;
            for (std::size_t row = 0; row != run.rows; ++row)
                if (task < key_runs)
                    keys[run.position + row] = safegraphs_widened_key<key_t>(fetched[0] + row * row_bytes, key_dtype);
                else
                    export_row<elementwise_kit_t>(fetched[0] + row * row_bytes, vector_dtype_k, target,
                                                  run.position + row);
        };
        auto const [reads, status] = scan_reads_(executor, key_runs + value_runs, describe, consume);
        search_result_t result;
        result.count = count;
        result.scan_reads = reads;
        return {result, status};
    }

#pragma endregion Key Scans

#pragma region Cursor Reads

    /** Entries part @p part holds, its node IDs below it. */
    node_id_t part_size(std::size_t part) const noexcept { return parts_[part].nodes; }

    /**
     *  @brief The earliest entry of every key of @p keys, in part then node order as @ref get
     *      orders them, through one lookup as @ref count takes it.
     *
     *  @param[in] keys The keys to locate, repeats and absent ones included.
     *  @param[out] locators The earliest entry under each key, parallel to @p keys; a key held
     *      nowhere gets the part @ref shards_count.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p locators and @p keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c unsupported_k No key table is held.
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    status_t find_first(std::span<key_t const> keys, std::span<shard_node_t> locators,
                        executor_t& executor) const noexcept {
        if (locators.size() != keys.size()) return status_t::config_mismatch_k;
        if (parts_.empty()) return status_t::uninitialized_k;
        if (keys.empty()) return status_t::success_k;
        auto [call, call_status] = begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        return nvme_find_first<atomic_ref>(keys, locators, to_span(parts_), executor,
                                           [&](std::span<key_t const> scanned, auto&& on_hit) noexcept {
                                               return lookup_keys_(scanned, executor, on_hit);
                                           });
    }

    /**
     *  @brief The keys of the entries @p nodes locates, in any order and with repeats: the key rows
     *      sorted by locator, rows whose blocks touch merged into one read of at most a scan run,
     *      as many in flight as the engines hold.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p keys and @p nodes differ in length, or a medium the engines cannot
     *    read.
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     *  - @c io_k A key read came back short.
     */
    status_t read_keys(std::span<shard_node_t const> nodes, std::span<key_t> keys,
                       executor_t& executor) const noexcept {
        if (keys.size() != nodes.size()) return status_t::config_mismatch_k;
        if (parts_.empty()) return status_t::uninitialized_k;
        if (nodes.empty()) return status_t::success_k;
        auto [call, call_status] = begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        return nvme_read_located<atomic_ref>(
            executor, to_span(media_), staging_bytes_(), nodes,
            [&](std::size_t position) noexcept {
                return extent_of_(nodes[position], &nvme_part_t::keys_begin,
                                  parts_[nodes[position].shard].key_row_bytes);
            },
            [&](std::size_t, std::size_t position, bytes_view_t row) noexcept {
                keys[position] = safegraphs_widened_key<key_t>(row.data(), parts_[nodes[position].shard].key_dtype);
            });
    }

    /**
     *  @brief The neighbors rows of the entries @p entries locates, each its part's connectivity
     *      wide, read as @ref read_keys reads keys.
     *
     *  @param[in] entries The entries whose neighbors rows to read, in any order, repeats allowed.
     *  @param[out] offsets The prefix sums of the rows' widths, one more than @p entries: where
     *      each entry's run starts in @p neighbors, then the end of the last.
     *  @param[out] neighbors Every entry's neighbor IDs end to end, resized to the last offset;
     *      an ID at or past its part's @ref part_size is a sentinel.
     *  @param[in] executor The pool that runs the scans.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p offsets is not one longer than @p entries, or a medium the engines
     *    cannot read.
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c bad_alloc_k @p neighbors could not grow.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     *  - @c io_k A neighbors read came back short.
     */
    template <typename neighbors_allocator_type_>
    status_t read_neighbors(std::span<shard_node_t const> entries, std::span<std::size_t> offsets,
                            growing_buffer<node_id_t, neighbors_allocator_type_>& neighbors,
                            executor_t& executor) const noexcept {
        if (offsets.size() != entries.size() + 1) return status_t::config_mismatch_k;
        if (parts_.empty()) return status_t::uninitialized_k;
        offsets[0] = 0;
        for (std::size_t position = 0; position != entries.size(); ++position)
            offsets[position + 1] = offsets[position] + parts_[entries[position].shard].connectivity;
        if (failed(neighbors.resize_uninitialized(offsets.back()))) return status_t::bad_alloc_k;
        if (!offsets.back()) return status_t::success_k;
        auto [call, call_status] = begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        return nvme_read_located<atomic_ref>(
            executor, to_span(media_), staging_bytes_(), entries,
            [&](std::size_t position) noexcept {
                nvme_part_t const& part = parts_[entries[position].shard];
                return extent_of_(entries[position], &nvme_part_t::neighbors_begin,
                                  part.connectivity * static_cast<u32_t>(sizeof(node_id_t)));
            },
            [&](std::size_t, std::size_t position, bytes_view_t row) noexcept {
                std::memcpy(neighbors.data() + offsets[position], row.data(), row.size());
            });
    }

#pragma endregion Cursor Reads

#pragma region Health

    /**
     *  @brief Reachability of every part from as many stride-spread landmarks as one search draws,
     *      the RAM audit's seeds, each part swept level by level over its neighbors column, one
     *      part after another.
     *
     *  A part of N rows holds two bitsets of N bits, one per level, from the first thread's arena:
     *  2 · N bits in flight, 1 GiB for a part of 2³² rows, the widest part's pair reused by every
     *  part. A level reads its rows' neighbors in ID order, rows whose blocks touch merged into one
     *  read of at most a scan run, as many in flight as the engines hold, so a dense level reads
     *  near-sequentially and every reachable row is read once, by the level expanding it: at most N
     *  reads per part, and at most the part's neighbors column in bytes per level.
     *
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     *  - @c config_mismatch_k A medium the engines cannot read.
     */
    expected<reachability_t> health(executor_t& executor) const noexcept {
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        std::size_t widest = 0;
        for (nvme_part_t const& part : parts_) widest = max_of<std::size_t>(widest, part.nodes);
        auto [call, call_status] = begin_scans_(executor, 1);
        if (failed(call_status)) return {{}, call_status};
        std::size_t const words = max_of<std::size_t>(
            divide_round_up<std::size_t>(widest, reachability_levels<atomic_ref>::lanes_k), 1);
        arena_t& arena = executor.arena(0);
        auto* const this_level = reinterpret_cast<u64_t*>(arena.allocate(words * sizeof(u64_t), default_alignment_k));
        auto* const next_level = reinterpret_cast<u64_t*>(arena.allocate(words * sizeof(u64_t), default_alignment_k));
        if (!this_level || !next_level) return {{}, status_t::bad_alloc_k};
        reachability_t result;
        for (u32_t part_index = 0; part_index != parts_.size(); ++part_index) {
            nvme_part_t const& part = parts_[part_index];
            std::size_t const row_bytes = std::size_t {part.connectivity} * sizeof(node_id_t);
            auto neighbors_extent = [&](std::size_t node) noexcept {
                return nvme_extent_t {part.neighbors_begin + node * row_bytes, static_cast<u32_t>(row_bytes),
                                      part_index};
            };
            auto const [audited, status] = nvme_audit_reachability<node_id_t, atomic_ref>(
                executor, to_span(media_), staging_bytes_(), {this_level, words}, {next_level, words}, part.nodes,
                part.connectivity, scan_run_rows_(part_index, row_bytes), neighbors_extent);
            if (failed(status)) return {{}, status};
            result += audited;
        }
        return {result, status_t::success_k};
    }

#pragma endregion Health

#pragma region Sample

    /**
     *  @brief Draws up to @p outputs' length entries around the focus keys, or across every part
     *      without any, and assigns them three coordinates among the shown points of @p inputs, as
     *      the RAM index samples the graph the parts were saved from.
     *
     *  The plan draws and places on executor thread zero, the pool thread its engine answers,
     *  inside one call holding the media, and every verb of its source blocks until its round
     *  lands. One key lookup, as @ref count takes it, lists the entries of the shown and focus keys
     *  before the plan, and one more counts each draw's ordinal after it. Scoring reads one round
     *  per neighbors row's width: the query's row, then a row and a key per node, scored a quad at
     *  a time as a walk scores them. Without shown points each draw's row projects through
     *  @ref sample_projection.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c shape_mismatch_k An output's or the shown coordinates' length disagrees.
     *  - @c bad_alloc_k The arenas cannot hold the scratch.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c config_mismatch_k A medium the engines cannot read.
     *  - @c unsupported_k Shown or focus keys, or draws, while no key table is held.
     *  - @c io_k A read came back short; the draws written stand, their later reads unscored.
     */
    expected<sample_result_t> sample(sample_inputs<key_t> inputs, sample_outputs<key_t> outputs, sample_config_t config,
                                     executor_t& executor) const noexcept {
        using rows_t = typename dense_sample_source_t::rows_t;
        auto draw = [&](rows_t const& rows) noexcept -> expected<sample_result_t> {
            growing_buffer<f32_t, scratch_allocator_t> projected {scratch_allocator_t {executor.arena(0)}};
            if (failed(projected.resize_uninitialized(dimensions()))) return {{}, status_t::bad_alloc_k};
            dense_sample_source_t const source {*this, executor, rows, to_writable_span(projected)};
            return nvme_draw_sample(source, inputs, outputs, config, executor, *rows.failure);
        };
        return nvme_sample<key_t, node_id_t>(
            executor, to_span(media_), to_span(parts_), connectivity(), staging_bytes_(), inputs, outputs.keys.size(),
            config,
            [&](std::span<key_t const> scanned, auto&& on_hit) noexcept {
                return lookup_keys_keeping_cut_(scanned, executor, on_hit);
            },
            draw);
    }

#pragma endregion Sample

  private:
    using arena_t = typename index_t::arena_t;
    using call_t = nvme_call<executor_t>;
    using walk_shape_t = typename index_t::walk_shape_t;
    using scratch_allocator_t = typename index_t::scratch_allocator_t;
    using candidates_allocator_t = typename index_t::candidates_allocator_t;
    using results_array_t = typename index_t::results_array_t;
    using walk_t = typename index_t::walk_t;
    using resident_keys_t = nvme_resident_keys<key_t, node_id_t, atomic_ref_template_>;

#pragma region Rows

    /** The rows of a batched search across the shards: each walk's candidates merge into the row's
     *  ranking by locator, and the written locators resolve into keys once every walk finished. A
     *  filtered row reads every scored row's key beside its value for the predicate, and merges
     *  each walk's admitted set instead. Each slot imports its query into its own staging, and
     *  distances come out as @p score_type_. */
    template <typename heads_type_, typename predicate_type_, typename score_type_>
    struct search_query {
        using key_t = key_type_;
        using fold_t = dummy_fold_t;

        /** A filtered row admits as it scores, so every scored row's key arrives beside its
         *  value. */
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

            /** The keys each query's walks take as landmarks, and the key table they probe; empty
             *  counts hint nothing. */
            ranked_hints_view<key_t> hints;
            resident_keys_t const* resident_keys = nullptr;

            /** Hint keys no shard held an entry under, summed over the call's queries. */
            std::size_t* missing_hints = nullptr;

            /** Where each MaxSim query's rows start among the call's, then where the last ends,
             *  and its seeds, confirmed once for all of its rows: a landmark row per shard,
             *  query-major, and how many each row holds. Empty unless a MaxSim resolved hints. */
            std::span<std::size_t const> query_offsets;
            std::span<nvme_seed<node_id_t> const> resolved_hints;
            std::span<u32_t const> resolved_counts;
        };

        inputs_t const& inputs;
        growing_buffer<shard_node_t, scratch_allocator_t> locators;
        scratch_allocator_t allocator;
        byte_t* staged = nullptr;
        std::size_t row = 0;
        std::size_t found = 0;
        vector_view_t query;
        vector_norms_t query_norms {};

        /** The landmark slots of a walk over the widest part, which bound the hints a query takes;
         *  zero without hints. */
        std::size_t hint_slots = 0;

        /** The query's hint keys its walks take, at most @c hint_slots. */
        std::span<key_t const> hinted_keys;

        /** One bit per key of @c hinted_keys, set once a walk confirmed an entry under it. */
        std::span<u1x64_t> confirmed_hints;

        search_query(inputs_t const& inputs, scratch_allocator_t allocator) noexcept
            : inputs(inputs), locators(allocator), allocator(allocator) {}

        /** Bytes a slot stages for a query of @p columns, and for @p hint_slots confirmed hints. */
        static std::size_t staged_bytes(std::size_t columns, std::size_t hint_slots) noexcept {
            return round_up_to_multiple<std::size_t>(staged_rows_bytes(vector_dtype_k, 1, columns), sizeof(u1x64_t)) +
                   divide_round_up<std::size_t>(hint_slots, std::numeric_limits<u1x64_t>::digits) * sizeof(u1x64_t);
        }

        status_t reserve(std::size_t, std::size_t widest_row) noexcept {
            std::size_t const columns = inputs.queries.columns;
            hint_slots = inputs.hints.counts.empty() ? 0 : nvme_landmarks(widest_row);
            staged = allocator.allocate(staged_bytes(columns, hint_slots));
            if (!staged || failed(locators.resize_uninitialized(inputs.keys.extent(1)))) return status_t::bad_alloc_k;
            std::size_t const rows_bytes = staged_bytes(columns, 0);
            confirmed_hints = {reinterpret_cast<u1x64_t*>(staged + rows_bytes),
                               (staged_bytes(columns, hint_slots) - rows_bytes) / sizeof(u1x64_t)};
            return status_t::success_k;
        }

        void prime_query(std::size_t new_query_index) noexcept {
            row = new_query_index;
            found = 0;
            query = import_staged_rows<elementwise_kit_t, rmscast_kit_t, scalar_t>(inputs.queries, row, 1,
                                                                                   inputs.kernel, staged)
                        .row(0)
                        .as_vector();
            inputs.counts[row] = 0;
            if (inputs.hints.counts.empty()) return;
            std::span<key_t const> const all_hints = hints_of(inputs.hints, row);
            hinted_keys = all_hints.first(min_of(all_hints.size(), hint_slots));
            std::memset(confirmed_hints.data(), 0, confirmed_hints.size_bytes());
        }

        /** Fills @p slots with the entries of shard @p shard the table files under the fingerprints
         *  of the first hinted keys, or the seeds its MaxSim query resolved, as many as fit,
         *  answering how many. */
        std::size_t find_hinted(u32_t shard, std::span<nvme_seed<node_id_t>> slots) const noexcept {
            if (!inputs.resolved_hints.empty()) {
                std::size_t const shards = inputs.resolved_counts.size() / (inputs.query_offsets.size() - 1);
                std::size_t const run = hinted_query_() * shards + shard;
                std::size_t const stride = inputs.resolved_hints.size() / inputs.resolved_counts.size();
                std::size_t const count = min_of<std::size_t>(inputs.resolved_counts[run], slots.size());
                for (std::size_t seed = 0; seed != count; ++seed)
                    slots[seed] = inputs.resolved_hints[run * stride + seed];
                return count;
            }
            std::size_t candidates = 0;
            for (u32_t hint = 0; hint != min_of(hinted_keys.size(), slots.size()); ++hint)
                inputs.resident_keys->find_each(shard, hinted_keys[hint], [&](node_id_t node) noexcept {
                    if (candidates != slots.size()) slots[candidates++] = {node, hint};
                });
            return candidates;
        }

        /** Whether the seeds @c find_hinted answers still need a key read each: only probed
         *  ones, never those a MaxSim resolved. */
        bool confirms_hints() const noexcept { return inputs.resolved_hints.empty(); }

        /** Whether @p stored is the key of hint @p hint, marking the hint held if so. */
        bool confirm_hint(u32_t hint, key_t const& stored) noexcept {
            if (!same_key_(hinted_keys[hint], stored)) return false;
            confirmed_hints[hint / std::numeric_limits<u1x64_t>::digits] |=
                u1x64_t {1} << (hint % std::numeric_limits<u1x64_t>::digits);
            return true;
        }

        /** The MaxSim query the row belongs to. */
        std::size_t hinted_query_() const noexcept {
            std::span<std::size_t const> const offsets = inputs.query_offsets;
            // The first offset past the row, by halving; the row's query starts one before it.
            std::size_t low = 0, high = offsets.size();
            while (low != high) {
                std::size_t const middle = low + (high - low) / 2;
                if (offsets[middle] <= row + offsets.front()) low = middle + 1;
                else high = middle;
            }
            return low - 1;
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
        shard_node_t output_locator(std::size_t output) const noexcept { return locators[output]; }
        void resolve_output_key(std::size_t output, key_t const& key) noexcept { inputs.keys(row, output) = key; }

        /** Writes the row's count, and counts its unconfirmed hints. */
        status_t finish_query() noexcept {
            inputs.counts[row] = found;
            if (hinted_keys.empty()) return status_t::success_k;
            std::size_t confirmed = 0;
            for (u1x64_t const word : confirmed_hints) confirmed += static_cast<std::size_t>(std::popcount(word));
            atomic_ref<std::size_t>(*inputs.missing_hints)
                .store_add(hinted_keys.size() - confirmed, std::memory_order_relaxed);
            return status_t::success_k;
        }
    };

    /** The keys fetched beside the values of one scored quad, which a filtered product writes for
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
     *  locator until the row's stride of the tape fills, its dot as @p product_type_, and the
     *  written locators resolve into keys. A filtered row reads every scored row's key beside its
     *  value instead, and its fold appends only the pairs @p predicate_type_ admits, by key. Each
     *  slot imports its left-hand row into its own staging. */
    template <typename product_type_, typename predicate_type_>
    struct multiply_query {
        using key_t = key_type_;
        inline static constexpr bool filtered_k = !std::is_same_v<std::remove_cv_t<predicate_type_>, dummy_predicate_t>;
        using fold_t = std::conditional_t<
            filtered_k, multiply_fold<key_t, product_type_, scored_keys_t, predicate_type_&>,
            multiply_fold<shard_node_t, product_type_, nvme_shard_locators<u32_t>, predicate_type_&>>;
        inline static constexpr nvme_key_reads_t key_reads_k = filtered_k ? nvme_key_reads_t::scored_k
                                                                          : nvme_key_reads_t::outputs_k;
        inline static constexpr bool admitting_k = false;

        /** The batch every row of the call reads and writes: each row stages its run at its stride
         *  of the tape, and counts the pairs it scored. */
        struct inputs_t {
            rows_view_t left;
            predicate_type_* predicate = nullptr;
            products_tape<key_t, product_type_> tape;
            std::span<std::size_t> scored;
        };

        inputs_t const& inputs;
        growing_buffer<shard_node_t, scratch_allocator_t> locators;
        byte_t* staged = nullptr;
        fold_t pairs;
        std::size_t row = 0;
        vector_view_t query;
        vector_norms_t query_norms {};

        multiply_query(inputs_t const& inputs, scratch_allocator_t allocator) noexcept
            : inputs(inputs), locators(allocator),
              staged(allocator.allocate(staged_rows_bytes(vector_dtype_k, 1, inputs.left.columns))),
              pairs {{}, {}, {}, *inputs.predicate} {}

        status_t reserve(std::size_t, std::size_t) noexcept {
            return !staged || (!filtered_k && failed(locators.resize_uninitialized(inputs.tape.stride)))
                       ? status_t::bad_alloc_k
                       : status_t::success_k;
        }
        void prime_query(std::size_t new_query_index) noexcept {
            row = new_query_index;
            // A product walks by dot, which never narrows at unit RMS.
            query = import_staged_rows<elementwise_kit_t, rmscast_kit_t, scalar_t>(inputs.left, row, 1, nk_kernel_dot_k,
                                                                                   staged)
                        .row(0)
                        .as_vector();
            inputs.scored[row] = 0;
            pairs.ids = output_ids_();
            pairs.products = inputs.tape.scores.subspan(inputs.tape.staging_of(row), inputs.tape.stride);
            pairs.written = 0;
            pairs.scored = 0;
        }
        void rescore_quad(quad<node_id_t> const&, quad<vector_view_t> const&, quad<vector_norms_t> const&,
                          scores_result<dot_result_t>&, std::size_t, std::span<key_t const>) const noexcept {}

        /** Keeps the keys fetched beside the quad, which the fold resolves its nodes through. */
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
        shard_node_t output_locator(std::size_t output) const noexcept { return locators[output]; }
        void resolve_output_key(std::size_t output, key_t const& key) noexcept {
            inputs.tape.keys[inputs.tape.staging_of(row) + output] = key;
        }
        status_t finish_query() noexcept {
            inputs.scored[row] = pairs.scored;
            return status_t::success_k;
        }

        /** Where the fold writes: the locators that resolve once the walks finished. */
        std::span<shard_node_t> output_ids_() noexcept { return to_writable_span(locators); }

        /** Where a filtered fold writes: the row's own stride of the tape's keys. */
        std::span<key_t> output_ids_() noexcept
            requires filtered_k
        {
            return inputs.tape.keys.subspan(inputs.tape.staging_of(row), inputs.tape.stride);
        }
    };

    /** The rows of attention, one per token: the token's heads rescore every scored batch of every
     *  walk into their running states, value rows looked up by the keys read beside the scored
     *  rows, only those @p predicate_type_ admits folding. Each slot imports its token's heads into
     *  its own staging, past them the @c f32 rows and softmax states that outputs of another layout
     *  or an empty carry cannot hold. */
    template <typename value_lookup_type_, typename predicate_type_>
    struct attend_query {
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

        attend_query(inputs_t const& inputs, scratch_allocator_t allocator) noexcept
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
        void prime_query(std::size_t new_token) noexcept {
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
        shard_node_t output_locator(std::size_t) const noexcept { return {}; }
        void resolve_output_key(std::size_t, key_t const&) noexcept {}
        status_t finish_query() noexcept {
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

    /** Walks @p rows_count rows of @p query_type_ over every shard, each walk as wide as @p config
     *  asks and each row staging @p staged_bytes of imported queries. */
    template <typename query_type_>
    expected<search_result_t> walk_queries_(std::size_t rows_count, search_config_t config,
                                            typename query_type_::inputs_t const& inputs, std::size_t staged_bytes,
                                            std::span<std::size_t const> counts, std::span<u64_t> latencies,
                                            executor_t& executor) noexcept {
        return index_t::template walk_queries_<query_type_>(to_span(media_), to_span(parts_), kernel_, executor,
                                                            rows_count, non_zero_expansion_factor(config), inputs,
                                                            staged_bytes, counts, latencies);
    }

    /** The batched @ref search_if under @p heads, which only a @c dummy_predicate_t accompanies,
     *  each shard's walk seeded by the rows it holds under the query's @p hints. */
    template <typename score_type_, typename heads_type_, typename predicate_type_>
    expected<search_result_t> search_rows_(rows_view_t queries, heads_type_ const& heads, executor_t& executor,
                                           search_config_t config, nk::matrix_span<key_t> keys,
                                           nk::matrix_span<score_type_> distances, std::span<std::size_t> counts,
                                           predicate_type_& predicate, std::span<u64_t> latencies,
                                           ranked_hints_view<key_t> hints) noexcept {
        std::size_t const queries_count = queries.rows;
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        if (queries_count && queries.columns != dimensions()) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = hints_refusal(hints, queries_count); failed(refusal)) return {{}, refusal};
        if (!hints.counts.empty() && resident_keys_.empty()) return {{}, status_t::unsupported_k};
        using query_t = search_query<heads_type_, predicate_type_, score_type_>;
        std::size_t missing_hints = 0;
        typename query_t::inputs_t inputs {queries, kernel_,   &heads, &predicate, non_zero_expansion_factor(config),
                                           keys,    distances, counts};
        inputs.hints = hints, inputs.resident_keys = &resident_keys_, inputs.missing_hints = &missing_hints;
        expected<search_result_t> found = walk_queries_<query_t>(
            queries_count, config, inputs, query_t::staged_bytes(queries.columns, hint_slots_(hints)), counts,
            latencies, executor);
        found.value.missing_hints = missing_hints;
        return found;
    }

    /** Resolves each query's first hints, as many as a walk draws landmarks, once for every row of
     *  it: one lookup lists their entries, and each query keeps a landmark row of them per shard in
     *  @p seeds, in hint order, how many each row holds in @p counts, query-major, and adds every
     *  key no shard holds to @p missing. */
    status_t resolve_hints_(ranked_hints_view<key_t> hints, executor_t& executor,
                            growing_buffer<nvme_seed<node_id_t>, scratch_allocator_t>& seeds,
                            growing_buffer<u32_t, scratch_allocator_t>& counts, std::size_t& missing) const noexcept {
        std::size_t const queries_count = hints.counts.size(), shards = parts_.size(), slots = hint_slots_(hints);
        scratch_allocator_t const scratch {executor.arena(0)};
        growing_buffer<key_t, scratch_allocator_t> keys {scratch};
        growing_buffer<std::size_t, scratch_allocator_t> starts {scratch};
        growing_buffer<shard_node_t, scratch_allocator_t> listed {scratch};
        if (failed(keys.reserve_exactly(queries_count * slots)) ||
            failed(seeds.resize(queries_count * shards * slots)) || failed(counts.resize(queries_count * shards)))
            return status_t::bad_alloc_k;
        for (std::size_t query = 0; query != queries_count; ++query) {
            std::span<key_t const> const query_hints = hints_of(hints, query);
            for (key_t const& key : query_hints.first(min_of(query_hints.size(), slots))) keys.push_back_reserved(key);
        }
        if (failed(starts.resize(keys.size() + 1))) return status_t::bad_alloc_k;
        if (status_t const listed_status = nvme_list_entries(
                to_span(keys), to_writable_span(starts), listed, executor,
                [&](std::span<key_t const> looked_up, auto&& on_hit) noexcept {
                    return lookup_keys_(looked_up, executor, on_hit);
                });
            failed(listed_status))
            return listed_status;
        std::size_t position = 0;
        for (std::size_t query = 0; query != queries_count; ++query) {
            std::size_t const taken = min_of(hints.counts[query], slots);
            for (u32_t hint = 0; hint != taken; ++hint, ++position) {
                missing += starts[position] == starts[position + 1];
                for (std::size_t entry = starts[position]; entry != starts[position + 1]; ++entry) {
                    std::size_t const run = query * shards + listed[entry].shard;
                    if (counts[run] != slots) seeds[run * slots + counts[run]++] = {listed[entry].node, hint};
                }
            }
        }
        return status_t::success_k;
    }

    /** The landmark slots a hinted query's walks fill, which bound the hints it takes: none
     *  without @p hints. */
    std::size_t hint_slots_(ranked_hints_view<key_t> hints) const noexcept {
        return hints.counts.empty() ? 0 : nvme_landmarks(connectivity());
    }

#pragma endregion Rows

#pragma region Scans

    /** One run of whole rows of one column a scan reads: where it starts, its rows, and where its
     *  first row sits among the rows the scan covers. */
    struct run_t {

        /** The shard and the run's first row inside it. */
        shard_node_t start;

        /** Rows in the run; zero past the last run. */
        std::size_t rows = 0;

        /** The run's first row, counted from the first row the scan covers. */
        std::size_t position = 0;
    };

    /** Begins a call of @p scans scans over every shard, each fetching @p reads_per_task_ reads per
     *  task, as @ref nvme_begin_scans begins one: every engine as deep as the call's depth, every
     *  buffer as wide as a walk's, and at least @p buffers for a walk sharing the call. */
    template <std::size_t reads_per_task_ = 1>
    expected<call_t> begin_scans_(executor_t& executor, std::size_t scans, std::size_t scratch_bytes = 0,
                                  std::size_t buffers = 0) const noexcept {
        return nvme_begin_scans<reads_per_task_>(executor, to_span(media_),
                                                 nvme_call_depth(executor, to_span(media_), to_span(parts_)), scans,
                                                 scratch_bytes, buffers, staging_bytes_());
    }

    /** @ref nvme_scan_reads over every part's medium, inside a call @ref begin_scans_ began. */
    template <std::size_t reads_per_task_ = 1, typename describe_type_, typename consume_type_>
    expected<nvme_reads_t> scan_reads_(executor_t& executor, std::size_t tasks_count, describe_type_&& describe,
                                       consume_type_&& consume) const noexcept {
        return nvme_scan_reads<reads_per_task_, atomic_ref>(executor, to_span(media_), tasks_count, describe, consume);
    }

    /** The read of @p node 's row of the column starting at @p column_begin, @p row_bytes wide. */
    nvme_extent_t extent_of_(shard_node_t node, u64_t nvme_part_t::* column_begin, u32_t row_bytes) const noexcept {
        nvme_part_t const& part = parts_[node.shard];
        return {part.*column_begin + u64_t {node.node} * row_bytes, row_bytes, static_cast<io_medium_id_t>(node.shard)};
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

    /** Bytes of one key-scan read: a whole request of every medium the parts sit on, so the key
     *  columns stream in order in requests no disk splits, or a staging buffer where that is more
     *  or no medium reports its requests. */
    std::size_t key_scan_bytes_() const noexcept {
        std::size_t request_bytes = 0;
        for (medium_t const& medium : media_)
            if (std::size_t const medium_bytes = std::size_t {device_queues_of(medium).request_kibibytes} * 1024)
                request_bytes = request_bytes ? min_of(request_bytes, medium_bytes) : medium_bytes;
        return max_of(request_bytes, staging_bytes_());
    }

    /** Rows of a column @p row_bytes wide that one scan read of part @p part takes: as many as a
     *  staging buffer covers wherever the run starts, and at least one, since it covers any row. */
    std::size_t scan_run_rows_(std::size_t part, std::size_t row_bytes) const noexcept {
        return scan_run_rows_(part, row_bytes, staging_bytes_());
    }

    /** Rows of a column @p row_bytes wide that one read of part @p part into a buffer of
     *  @p buffer_bytes covers wherever the run starts. */
    std::size_t scan_run_rows_(std::size_t part, std::size_t row_bytes, std::size_t buffer_bytes) const noexcept {
        u64_t const block = block_bytes_of(media_[part]);
        return static_cast<std::size_t>((buffer_bytes / block * block - (block - 1)) / row_bytes);
    }

    /** Runs of @p column over the @p count placement-order rows from row @p first on, cut at every
     *  part's end and every read of @p buffer_bytes, so a row never straddles two reads. */
    std::size_t runs_count_(std::size_t first, std::size_t count, column_t column,
                            std::size_t buffer_bytes) const noexcept {
        std::size_t runs = 0, base = 0;
        for (std::size_t part = 0; part != parts_.size(); base += parts_[part].nodes, ++part) {
            std::size_t const low = max_of(first, base), high = min_of(first + count, base + parts_[part].nodes);
            if (low < high)
                runs += divide_round_up(high - low, scan_run_rows_(part, parts_[part].*column.row_bytes, buffer_bytes));
        }
        return runs;
    }

    /** Run @p run of the runs @c runs_count_ counts at @p buffer_bytes, or an empty one past the
     *  last. */
    run_t locate_run_(std::size_t first, std::size_t count, std::size_t run, column_t column,
                      std::size_t buffer_bytes) const noexcept {
        std::size_t base = 0;
        for (u32_t part = 0; part != parts_.size(); base += parts_[part].nodes, ++part) {
            std::size_t const low = max_of(first, base), high = min_of(first + count, base + parts_[part].nodes);
            if (low >= high) continue;
            std::size_t const run_rows = scan_run_rows_(part, parts_[part].*column.row_bytes, buffer_bytes);
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
    nvme_extent_t run_extent_(run_t const& run, column_t column) const noexcept {
        nvme_part_t const& part = parts_[run.start.shard];
        u64_t const row_bytes = part.*column.row_bytes;
        return {part.*column.begin + run.start.node * row_bytes, static_cast<u32_t>(run.rows * row_bytes),
                run.start.shard};
    }

    /** Whether two keys are the same, by their bytes, whatever their type. */
    static bool same_key_(key_t const& first, key_t const& second) noexcept {
        return std::memcmp(&first, &second, sizeof(key_t)) == 0;
    }

    /**
     *  @brief Every entry under every key of @p keys, repeats included, as
     *      <tt>on_hit(thread, position, locator)</tt>, probed in the key table and confirmed by
     *      one key read apiece.
     *
     *  Runs inside a call @c begin_scans_ or a walk began, its candidates in thread zero's arena.
     *
     *  @par Statuses:
     *  - @c unsupported_k No key table is held.
     *  - @c bad_alloc_k Thread zero's arena could not hold the candidates.
     *  - @c io_k A key read came back short.
     */
    template <typename on_hit_type_>
    status_t lookup_keys_(std::span<key_t const> keys, executor_t& executor, on_hit_type_&& on_hit) const noexcept {
        if (resident_keys_.empty()) return status_t::unsupported_k;
        scratch_allocator_t const scratch {executor.arena(0)};
        growing_buffer<shard_node_t, scratch_allocator_t> candidates {scratch};
        growing_buffer<std::size_t, scratch_allocator_t> positions {scratch};
        bool refused = false;
        for (std::size_t position = 0; position != keys.size(); ++position)
            for (u32_t part = 0; part != parts_.size(); ++part)
                resident_keys_.find_each(part, keys[position], [&](node_id_t node) noexcept {
                    refused = refused || failed(candidates.push_back(shard_node_t {part, node})) ||
                              failed(positions.push_back(position));
                });
        if (refused) return status_t::bad_alloc_k;
        return nvme_read_located<atomic_ref>(
            executor, to_span(media_), staging_bytes_(), to_span(candidates),
            [&](std::size_t candidate) noexcept {
                return extent_of_(candidates[candidate], &nvme_part_t::keys_begin,
                                  parts_[candidates[candidate].shard].key_row_bytes);
            },
            [&](std::size_t thread, std::size_t candidate, bytes_view_t row) noexcept {
                shard_node_t const locator = candidates[candidate];
                key_t const stored = safegraphs_widened_key<key_t>(row.data(), parts_[locator.shard].key_dtype);
                if (same_key_(stored, keys[positions[candidate]])) on_hit(thread, positions[candidate], locator);
            });
    }

    /** @ref lookup_keys_ inside a call whose own reads go on through the staging cut it began with,
     *  which a scan's cut gives back. */
    template <typename on_hit_type_>
    status_t lookup_keys_keeping_cut_(std::span<key_t const> keys, executor_t& executor,
                                      on_hit_type_&& on_hit) const noexcept {
        io_buffers_cut_t const call_cut = executor.buffers_cut();
        status_t const looked = lookup_keys_(keys, executor, on_hit);
        io_buffers_cut_t const scan_cut = executor.buffers_cut();
        if (scan_cut.buffers == call_cut.buffers && scan_cut.buffer_bytes == call_cut.buffer_bytes) return looked;
        status_t const restored = executor.reserve_buffers(call_cut.buffers, call_cut.buffer_bytes);
        return failed(looked) ? looked : restored;
    }

    /**
     *  @brief Reads every part's key column once, in order, and hands every row to @p on_row as
     *      <tt>on_row(thread, key, locator)</tt> on the thread that read it.
     *
     *  Runs inside a call @c begin_scans_ or a walk began. The call's staging region is cut into as
     *  many whole requests of @ref key_scan_bytes_ as it holds, at most one per queue slot, once
     *  per call: a later scan in the call finds it cut, and a verb reading rows after the scan
     *  reads them through those buffers, which hold any row.
     */
    template <typename on_row_type_>
    status_t scan_key_rows_(executor_t& executor, on_row_type_&& on_row) const noexcept {
        io_buffers_cut_t const call_cut = executor.buffers_cut();
        std::size_t const region_bytes = max_of(call_cut.buffers * call_cut.buffer_bytes, staging_bytes_());
        std::size_t const request_bytes = key_scan_bytes_();
        std::size_t const scan_buffers = max_of<std::size_t>(
            min_of(region_bytes / request_bytes, executor.queue_depth()), 1);
        std::size_t const read_bytes = min_of(request_bytes, region_bytes / scan_buffers);
        if (call_cut.buffers != scan_buffers || call_cut.buffer_bytes != read_bytes)
            if (status_t const cut = executor.reserve_buffers(scan_buffers, read_bytes); failed(cut)) return cut;
        std::size_t const rows_count = size();
        auto describe = [&](std::size_t run, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
            if (progress++) return {};
            return {run_extent_(locate_run_(0, rows_count, run, key_column_k, read_bytes), key_column_k)};
        };
        auto consume = [&](std::size_t thread, std::size_t, u64_t, auto const& extents, auto const& fetched) noexcept {
            nvme_extent_t const& extent = extents[0];
            byte_t const* const bytes = fetched[0];
            nvme_part_t const& part = parts_[extent.medium];
            std::size_t const first = static_cast<std::size_t>((extent.begin - part.keys_begin) / part.key_row_bytes);
            for (std::size_t row = 0; row != extent.length / part.key_row_bytes; ++row)
                on_row(thread, safegraphs_widened_key<key_t>(bytes + row * part.key_row_bytes, part.key_dtype),
                       shard_node_t {extent.medium, static_cast<node_id_t>(first + row)});
        };
        return scan_reads_(executor, runs_count_(0, rows_count, key_column_k, read_bytes), describe, consume).status;
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
                share.thresholds[query] = (std::numeric_limits<distance_t>::max)();
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
        auto keep_match = [wanted](scan_share_t const& share, std::size_t query, scan_match_t match) noexcept {
            scan_match_t* const kept = share.matches + query * wanted;
            u32_t& count = share.kept[query];
            if (count < wanted) {
                kept[count++] = match;
                if (count < wanted) return (std::numeric_limits<distance_t>::max)();
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
        auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<nvme_extent_t, 2> {
            if (progress++) return {};
            run_t const run = locate_run_(0, rows_count, task, runs_column, staging_bytes_());
            return {run_extent_(run, key_column_k), run_extent_(run, value_column_k)};
        };
        auto consume = [&](std::size_t thread, std::size_t, u64_t, std::array<nvme_extent_t, 2> const& extents,
                           std::array<byte_t const*, 2> const& fetched) noexcept {
            scan_share_t const& share = shares[thread];
            nvme_part_t const& part = parts_[extents[1].medium];
            std::size_t const run_rows = static_cast<std::size_t>(extents[1].length / part.value_row_bytes);
            auto key_of = [&](std::size_t row) noexcept {
                return safegraphs_widened_key<key_t>(fetched[0] + row * part.key_row_bytes, part.key_dtype);
            };
            auto row_of = [&](std::size_t row) noexcept {
                return vector_view_t {reinterpret_cast<scalar_t const*>(fetched[1] + row * part.value_row_bytes),
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
                    vectors_view_t const dense {reinterpret_cast<scalar_t const*>(fetched[1]), run_rows,
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
                        auto keep_scored = [&](std::size_t column, scored_node_t scored) noexcept {
                            return keep_match(share, first + column, {scored.distance, key_of(scored.node_id)});
                        };
                        scan_tiles<tiles_kit_t, quads_kit_t>(kernel_, rows, 0, packed + pack * pack_stride,
                                                             std::span<dot_result_t const>(sumsqs + first, columns),
                                                             std::span<distance_t>(share.thresholds + first, columns),
                                                             dots, row_sumsq, admits, keep_scored);
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
                auto keep_entry = [&](scan_payload_t contender) noexcept {
                    for (std::size_t query = 0; query != queries_count; ++query) {
                        distance_t const distance = share.quad_distances[query * quad_lanes_k + contender - row];
                        if (distance < share.thresholds[query])
                            share.thresholds[query] = keep_match(share, query, {distance, key_of(contender)});
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
                ask_contenders(admits, nodes, payloads, filled, keep_entry);
                settle_predicate(admits, keep_entry);
            }
        };
        auto const [reads, status] = scan_reads_<2>(executor, runs_count_(0, rows_count, runs_column, staging_bytes_()),
                                                    describe, consume);
        search_result_t result;
        result.scan_reads = reads;
        result.computed_distances = rows_count * queries_count;
        if (failed(status)) return {result, status};

        // Every thread's matches of a query merge, the nearest first.
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
        shard_node_t locator;

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
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> walk_maxsim_(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                           vectors_view_t query_rows, std::span<vector_norms_t const> query_norms,
                                           walk_shape_t const& shape, executor_t& executor, std::size_t wanted,
                                           std::span<std::size_t> scored,
                                           products_tape<key_t, score_type_> const& matches, predicate_type_& predicate,
                                           std::span<u64_t> latencies, ranked_hints_view<key_t> hints) noexcept {
        using query_t = search_query<dummy_heads_t, predicate_type_, distance_t>;
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
        typename query_t::inputs_t inputs {queries.slice(query_offsets.front(), rows_count),
                                           kernel_,
                                           nullptr,
                                           &predicate,
                                           width,
                                           walk_keys,
                                           walk_distances,
                                           to_writable_span(walked)};
        // Every row of a query takes the seeds its hints resolve to once.
        std::size_t missing_hints = 0;
        growing_buffer<nvme_seed<node_id_t>, scratch_allocator_t> resolved_hints {scratch};
        growing_buffer<u32_t, scratch_allocator_t> resolved_counts {scratch};
        if (!hints.counts.empty()) {
            if (status_t const resolved = resolve_hints_(hints, executor, resolved_hints, resolved_counts,
                                                         missing_hints);
                failed(resolved))
                return {{}, resolved};
            inputs.query_offsets = query_offsets;
            inputs.resolved_hints = to_span(resolved_hints), inputs.resolved_counts = to_span(resolved_counts);
        }
        auto [result, walk_status] = query_walker<index_t, query_t>::run_walks(
            to_span(media_), to_span(parts_), executor, shape, inputs, to_span(walked), latencies);
        result.count = 0;
        result.missing_hints = missing_hints;
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
        growing_buffer<matches_t, scratch_allocator_t> listings {scratch};
        if (failed(row_starts.resize(nominated + 1)) || failed(listings.reserve_exactly(executor.size())))
            return {result, status_t::bad_alloc_k};
        for (std::size_t thread = 0; thread != executor.size(); ++thread)
            listings.push_back_reserved(matches_t {scratch_allocator_t {executor.arena(thread)}});
        status_t listing = status_t::success_k;
        status_t const scanned = lookup_keys_(
            nominations, executor, [&](std::size_t thread, std::size_t position, shard_node_t locator) noexcept {
                atomic_ref<std::size_t>(row_starts[position]).store_add(1, std::memory_order_relaxed);
                if (failed(listings[thread].push_back({locator, position})))
                    atomic_ref<status_t>(listing).store(status_t::bad_alloc_k, std::memory_order_relaxed);
            });
        if (failed(scanned)) return {result, scanned};
        if (failed(listing)) return {result, listing};

        // Each count becomes its candidate's end, and every listed row steps it back once.
        for (std::size_t position = 1; position <= nominated; ++position)
            row_starts[position] += row_starts[position - 1];
        growing_buffer<shard_node_t, scratch_allocator_t> locators {scratch};
        if (failed(locators.resize_uninitialized(row_starts[nominated]))) return {result, status_t::bad_alloc_k};
        for (matches_t const& listed : listings)
            for (maxsim_match_t const& match : listed) locators[--row_starts[match.position]] = match.locator;

        growing_buffer<dot_result_t, scratch_allocator_t> folded {scratch};
        if (failed(folded.resize_uninitialized(maxima_count))) return {result, status_t::bad_alloc_k};
        for (dot_result_t& maximum : folded) maximum = dot_result_t::finite_min();
        std::span<dot_result_t> const maxima = to_writable_span(folded);
        std::span<std::size_t const> const starts = to_span(segment_starts);
        auto describe = [&](std::size_t position, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
            std::size_t const row = row_starts[position] + static_cast<std::size_t>(progress++);
            if (row == row_starts[position + 1]) return {};
            shard_node_t const locator = locators[row];
            return {extent_of_(locator, &nvme_part_t::values_begin, parts_[locator.shard].value_row_bytes)};
        };
        auto consume = [&](std::size_t, std::size_t position, u64_t, std::array<nvme_extent_t, 1> const& extents,
                           std::array<byte_t const*, 1> const& fetched) noexcept {
            std::size_t const query = static_cast<std::size_t>(lower_bound(starts, position + 1) - starts.data()) - 1;
            std::size_t const first = query_offsets[query] - query_offsets.front();
            std::size_t const rows = query_offsets[query + 1] - query_offsets[query];
            vector_view_t const row {reinterpret_cast<scalar_t const*>(fetched[0]),
                                     parts_[extents[0].medium].dimensions};
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
        if (capacity > matches.stride) return {result, status_t::config_mismatch_k};
        growing_buffer<maxsim_ranked<total_t>, scratch_allocator_t> ranked {scratch};
        if (failed(ranked.resize_uninitialized(executor.size() * capacity))) return {result, status_t::bad_alloc_k};
        status_t const ranking = executor.for_tasks(queries_count, [&](std::size_t query, std::size_t thread,
                                                                       std::size_t) noexcept {
            std::size_t const begin = starts[query], end = starts[query + 1];
            std::size_t const rows = query_offsets[query + 1] - query_offsets[query];
            scored[query] = end - begin;
            nk::matrix_span<dot_result_t> const query_maxima {maxima.data() + maxima_starts[query], end - begin, rows};
            rank_candidates_(nominations.subspan(begin, end - begin), query_maxima,
                             to_writable_span(ranked).subspan(thread * capacity, capacity), query, matches);
        });
        return {result, ranking};
    }

    /**
     *  @brief Late interaction nominated by a scan, inside the call @ref maxsim_if began: every key
     *      is a candidate, each distinct one asked of the predicate once.
     *
     *  A scan reads every key and value run once, folding each row through compare-exchanges, since
     *  a key's rows arrive on any thread; its maxima hold one value per distinct key and query row,
     *  refused with @c bad_alloc_k past memory.
     */
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> scan_maxsim_(std::span<std::size_t const> query_offsets, vectors_view_t query_rows,
                                           std::span<vector_norms_t const> query_norms, executor_t& executor,
                                           std::size_t wanted, std::span<std::size_t> scored,
                                           products_tape<key_t, score_type_> const& matches,
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
        auto describe_keys = [&](std::size_t run, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
            if (progress++) return {};
            return {run_extent_(locate_run_(0, rows_count, run, key_column_k, staging_bytes_()), key_column_k)};
        };
        auto collect_keys = [&](std::size_t, std::size_t run, u64_t, std::array<nvme_extent_t, 1> const& extents,
                                std::array<byte_t const*, 1> const& fetched) noexcept {
            run_t const located = locate_run_(0, rows_count, run, key_column_k, staging_bytes_());
            nvme_part_t const& part = parts_[extents[0].medium];
            for (std::size_t row = 0; row != located.rows; ++row)
                candidates[located.position + row] = safegraphs_widened_key<key_t>(
                    fetched[0] + row * part.key_row_bytes, part.key_dtype);
        };
        auto const [key_reads, collect_status] = scan_reads_(
            executor, runs_count_(0, rows_count, key_column_k, staging_bytes_()), describe_keys, collect_keys);
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
        auto describe_runs = [&](std::size_t run, u64_t& progress) noexcept -> std::array<nvme_extent_t, 2> {
            if (progress++) return {};
            run_t const located = locate_run_(0, rows_count, run, runs_column, staging_bytes_());
            return {run_extent_(located, key_column_k), run_extent_(located, value_column_k)};
        };
        auto fold_runs = [&](std::size_t, std::size_t, u64_t, std::array<nvme_extent_t, 2> const& extents,
                             std::array<byte_t const*, 2> const& fetched) noexcept {
            nvme_part_t const& part = parts_[extents[1].medium];
            std::size_t const run_rows = static_cast<std::size_t>(extents[1].length / part.value_row_bytes);
            for (std::size_t row = 0; row != run_rows; ++row) {
                std::size_t const position = find(
                    safegraphs_widened_key<key_t>(fetched[0] + row * part.key_row_bytes, part.key_dtype));
                if (position == admitted) continue;
                vector_view_t const values {reinterpret_cast<scalar_t const*>(fetched[1] + row * part.value_row_bytes),
                                            part.dimensions};
                fold_quads<quads_kit_t>(values, index_t::norms_(kernel_, values), query_rows, query_norms,
                                        to_span(maxima.row(position).as_vector()),
                                        [](dot_result_t& maximum, dot_result_t dot) noexcept {
                                            atomic_ref<dot_result_t> const shared {maximum};
                                            if (shared.load(std::memory_order_relaxed) < dot)
                                                shared.store_max(dot, std::memory_order_relaxed);
                                        });
            }
        };
        auto const [run_reads, fold_status] = scan_reads_<2>(
            executor, runs_count_(0, rows_count, runs_column, staging_bytes_()), describe_runs, fold_runs);
        result.scan_reads += run_reads;
        result.computed_distances += rows_count * queries_rows;
        if (failed(fold_status)) return {result, fold_status};

        // Without a bound, every thread's ranking holds every admitted key.
        std::size_t const capacity = wanted ? wanted : admitted;
        if (capacity > matches.stride) return {result, status_t::config_mismatch_k};
        growing_buffer<maxsim_ranked<total_t>, scratch_allocator_t> ranked {scratch};
        if (failed(ranked.resize_uninitialized(executor.size() * capacity))) return {result, status_t::bad_alloc_k};
        status_t const ranking = executor.for_tasks(
            queries_count, [&](std::size_t query, std::size_t thread, std::size_t) noexcept {
                std::size_t const first = query_offsets[query] - query_offsets.front();
                std::size_t const rows = query_offsets[query + 1] - query_offsets[query];
                scored[query] = admitted;
                rank_candidates_(to_span(candidates).first(admitted),
                                 maxima(nk::all, nk::range(first, first + rows), nk::slice),
                                 to_writable_span(ranked).subspan(thread * capacity, capacity), query, matches);
            });
        return {result, ranking};
    }

    /** Ranks @p candidates by the sums of their rows of @p maxima, largest first and ties by
     *  position, keeping the best in @p ranked, and stages as many as it holds at the stride of
     *  query @p query on @p matches. */
    template <typename score_type_>
    static void rank_candidates_(std::span<key_t const> candidates, nk::matrix_view<dot_result_t> maxima,
                                 std::span<maxsim_ranked<total_t>> ranked, std::size_t query,
                                 products_tape<key_t, score_type_> const& matches) noexcept {
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
        std::size_t const staged = matches.staging_of(query);
        for (std::size_t rank = 0; rank != kept; ++rank)
            matches.keys[staged + rank] = candidates[ranked[rank].candidate],
                                  matches.scores[staged + rank] = static_cast<score_type_>(ranked[rank].total);
    }

#pragma endregion MaxSim

#pragma region Sampling

    /** Every part as SAMPLE reads it, each verb a blocking round on thread zero's engine; see
     *  @c sample_source. A failed read lands in @c rows.failure, and its verb answers no row. */
    struct dense_sample_source_t {
        using key_t = key_type_;
        using node_id_t = dense::node_id_t;
        using rows_t = nvme_sample_rows<nvme_part_t, key_t, node_id_t, engine_t>;

        nvme_indexes const& indexes;
        executor_t& executor;
        rows_t rows;

        /** The f32 row the projection reads, as wide as the dimensions. */
        std::span<f32_t> projected;

        std::span<node_id_t const> watermarks() const noexcept { return rows.watermarks_snapshot; }

        key_t key_of(shard_node_t node) const noexcept { return rows.key_of(node); }

        template <typename callback_type_>
        void nodes_of(key_t const& key, callback_type_&& callback) const noexcept {
            rows.nodes_of(key, callback);
        }

        /** A self-loop marks no neighbors, as the RAM source reads it. */
        template <typename callback_type_>
        void neighbors(shard_node_t node, callback_type_&& callback) const noexcept {
            nvme_part_t const& part = indexes.parts_[node.shard];
            u32_t const row_bytes = part.connectivity * static_cast<u32_t>(sizeof(node_id_t));
            for (node_id_t const next : rows.neighbors_row(extent_of_(node, &nvme_part_t::neighbors_begin, row_bytes)))
                if (next < rows.watermarks_snapshot[node.shard] && next != node.node)
                    callback(shard_node_t {node.shard, next});
        }

        /** Scores in rounds of a neighbors row's width, each reading the query's row beside the
         *  nodes' rows and keys, a quad at a time as a walk scores them. */
        void distance(shard_node_t from, std::span<shard_node_t const> nodes,
                      std::span<f32_t> distances) const noexcept {
            std::size_t const width = rows.staged_neighbors.size();
            for (std::size_t first = 0; first < nodes.size(); first += width) {
                std::size_t const count = min_of(width, nodes.size() - first);
                std::span<shard_node_t const> const round = nodes.subspan(first, count);
                rows.round_extents[0] = extent_of_(from, &nvme_part_t::values_begin,
                                                   indexes.parts_[from.shard].value_row_bytes);
                for (std::size_t lane = 0; lane != count; ++lane)
                    rows.round_extents[1 + lane] = extent_of_(round[lane], &nvme_part_t::values_begin,
                                                              indexes.parts_[round[lane].shard].value_row_bytes);
                if (failed(rows.read_round(round, distances.subspan(first, count)))) continue;
                rows.remember_keys(round);
                vector_view_t const query = view_of_(from, rows.round_bytes[0]);
                vector_norms_t const query_norms = index_t::norms_(indexes.kernel_, query);
                for (std::size_t quad_first = 0; quad_first < count; quad_first += quad_lanes_k) {
                    std::size_t const lanes = min_of<std::size_t>(quad_lanes_k, count - quad_first);
                    quad<vector_view_t> lane_rows;
                    quad<vector_norms_t> lane_norms;
                    for (std::size_t lane = 0; lane != quad_lanes_k; ++lane) {
                        std::size_t const source = quad_first + min_of(lane, lanes - 1);
                        lane_rows[lane] = view_of_(round[source], rows.round_bytes[1 + source]);
                        lane_norms[lane] = index_t::norms_(indexes.kernel_, lane_rows[lane]);
                    }
                    scores_result<dot_result_t> const scores = dense::scores<quads_kit_t>(
                        indexes.kernel_, query, query_norms, lane_rows, lane_norms);
                    for (std::size_t lane = 0; lane != lanes; ++lane)
                        distances[first + quad_first + lane] = static_cast<f32_t>(scores.distances[lane]);
                }
            }
        }

        std::size_t walk(shard_node_t from, shard_node_t toward, std::span<shard_node_t> chain) const noexcept {
            return sample_greedy_walk(*this, from, toward, chain);
        }

        std::span<f32_t const> values_of(shard_node_t node) const noexcept {
            bytes_view_t const row = rows.read_row(
                extent_of_(node, &nvme_part_t::values_begin, indexes.parts_[node.shard].value_row_bytes));
            if (row.empty()) {
                std::memset(projected.data(), 0, projected.size_bytes());
                return projected;
            }
            rows_span_t const target {reinterpret_cast<byte_t*>(projected.data()), 1, projected.size(),
                                      projected.size_bytes(), nk_f32_k};
            export_row<elementwise_kit_t>(row.data(), vector_dtype_k, target, 0);
            return projected;
        }

        /** Counts, for every draw, its key's entries before it in @ref get order, in one scan. */
        status_t ordinals_of(std::span<key_t const> keys, std::span<shard_node_t const> nodes,
                             std::span<std::size_t> ordinals) const noexcept {
            std::memset(ordinals.data(), 0, ordinals.size_bytes());
            if (keys.empty()) return *rows.failure;
            status_t const scanned = indexes.lookup_keys_keeping_cut_(
                keys, executor, [&](std::size_t, std::size_t position, shard_node_t locator) noexcept {
                    if (locator < nodes[position])
                        atomic_ref<std::size_t>(ordinals[position]).store_add(1, std::memory_order_relaxed);
                });
            return failed(scanned) ? scanned : *rows.failure;
        }

      private:
        nvme_extent_t extent_of_(shard_node_t node, u64_t nvme_part_t::* column_begin, u32_t row_bytes) const noexcept {
            return indexes.extent_of_(node, column_begin, row_bytes);
        }

        vector_view_t view_of_(shard_node_t node, byte_t const* fetched) const noexcept {
            return vector_view_t {reinterpret_cast<scalar_t const*>(fetched), indexes.parts_[node.shard].dimensions};
        }
    };

#pragma endregion Sampling

#pragma region Opening

    /** The body of @ref view, into an empty catalog. */
    status_t view_(std::string_view checkpoint, std::span<std::string_view const> paths) noexcept {
        if (checkpoint.empty()) return status_t::invalid_argument_k;
        std::array<byte_t, safegraphs_prefix_t::magic_k.size()> magic {};
        if (names_transport_id(checkpoint))
            std::memcpy(magic.data(), safegraphs_prefix_t::magic_k.data(), magic.size());
        else {
            auto [file, open_status] = file_medium_t::open(checkpoint);
            if (failed(open_status)) return open_status;
            // A manifest shorter than the magic reads short; parsing it below reports that.
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
                return release_medium<wait_t>(medium), status_t::config_mismatch_k;
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

    /** The key table @ref hold_keys built, every part's; empty when none is held. */
    resident_keys_t resident_keys_;

    /** The distance kind the parts name, every walk ranks by, and every rewrite carries on. */
    nk_kernel_kind_t kernel_ = nk_kernel_unknown_k;
};

} // namespace unum::usearch::dense
