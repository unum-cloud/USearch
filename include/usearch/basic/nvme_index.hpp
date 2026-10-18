/**
 *  @file include/usearch/basic/nvme_index.hpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief Similarity Search over pre-constructed read-only basic graphs on block storage: ragged
 *      values under caller-supplied metrics, walked by the storage walk the dense tier drives.
 *
 *  A view opens the monolith or the manifest a basic RAM index saved and keeps, per part, its
 *  columns' offsets and a directory of every ragged column's row starts: an Elias-Fano sequence of
 *  a few bits per row, so a walk locates any value or neighbors row in RAM and reads it with one
 *  covering read, as the dense tier reads its fixed rows. The directories are built at the view by
 *  streaming each offsets column in chunks, and the widest rows, which size the staging buffers,
 *  come from the same pass. A fixed-degree neighbors column needs no directory.
 *
 *  A search keeps one metric per row slot, built by the caller's factory on the slot's arena, since
 *  every walk of a row shares its query, and hands it each landed value as the codec's borrowed
 *  @c view. A caller's judge reads the same views before the flights recycle, so a relation tested
 *  on stored values needs no second read. Key lookups scan the key columns, and @c get reads the
 *  value rows of the keys it found.
 */
#pragma once

#include <cstring> // `std::memcmp`, `std::memset`

#include <array>       // `std::array` - a quad's distances, scored rows and verdicts
#include <atomic>      // `std::atomic_ref` - the first failure and the counts across threads
#include <concepts>    // `std::same_as`
#include <limits>      // `std::numeric_limits` - the node ID ceiling
#include <memory>      // `std::uninitialized_value_construct_n` - the lock stripes of `get`
#include <mutex>       // `std::unique_lock` - one stripe's hold
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::remove_cvref_t`, `std::invoke_result_t`, `std::is_same_v`, `std::is_constructible_v`
#include <utility>     // `std::move`

#include <smashtable/immutable_elias_fano.hpp> // `st::immutable_elias_fano` - the row directories

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `seekable_file_t`
#include "usearch/utilities/nvme_device.hpp"       // `nvme_executor`, `file_medium_t`, `io_request_t`
#include "usearch/utilities/nvme_walks.hpp"        // `row_walker`, `nvme_begin_call`, `nvme_read_header`
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`, `sorted_buffer`, `arena_allocator`
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator`
#include "usearch/utilities/ram_sorts.hpp"         // `network_sort`
#include "usearch/utilities/ram_tables.hpp"        // `growing_hash_set`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_header`, `safegraphs_read_manifest`
#include "usearch/basic/ram_index.hpp"             // `scored_node`, `metric_staging_t`, `unbounded_radius`

namespace unum::usearch::basic {

/**
 *  @brief Similarity Search over the SafeGraphs parts a basic RAM index saved, on block storage,
 *      read-only.
 *
 *  The class is a catalog of parts: each part's medium, its columns' offsets and the directories of
 *  its ragged columns, and nothing per node beyond those few bits a row. Every call borrows an
 *  executor, attaches the media to its engines for its own duration, and walks every part of a row
 *  on one thread, merging the parts' rankings by key.
 *
 *  @tparam codec_type_ The value codec the parts were saved with, whose @c view borrows a landed
 *      row.
 *  @tparam key_type_ The caller-visible identity, read from every part's key column.
 *  @tparam node_id_type_ Identity within one part, as wide as the neighbor IDs the parts were saved
 *      with; @c u32_t or @c u64_t.
 *  @tparam distance_type_ The distance the metrics drain and the walks rank by.
 *  @tparam engine_type_ The I/O engine every executor thread drives.
 */
template <typename codec_type_, typename key_type_ = uuid_t, typename node_id_type_ = u32_t,
          typename distance_type_ = f64_t, typename engine_type_ = default_io_engine_t,
          typename row_kit_type_ = st::serial_row_kit_t>
    requires std::same_as<node_id_type_, u32_t> || std::same_as<node_id_type_, u64_t>
class nvme_index {
    template <typename, typename>
    friend class utilities::row_walker;

  public:
    using codec_t = codec_type_;
    using key_t = key_type_;
    using distance_t = distance_type_;
    using engine_t = engine_type_;
    using executor_t = nvme_executor<engine_t>;
    using medium_t = typename engine_t::medium_t;
    using arena_t = typename executor_t::arena_t;

    /** Node identifier within one part. */
    using node_id_t = node_id_type_;
    using scored_node_t = scored_node<node_id_t, distance_t>;

    /** Per-search knobs; zero derives from the parts' connectivity. */
    struct search_config_t {

        /** Breadth of each walk's candidate pool; defaults to twice the widest neighbors row. */
        std::size_t expansion_factor = 0;
    };

    /** Telemetry of one batched call; matches land in the caller's spans. */
    struct [[nodiscard]] search_result_t {

        /** Matches written across the batch. */
        std::size_t count = 0;

        /** Frontier candidates whose neighbors rows were read & expanded. */
        std::size_t visited_members = 0;

        /** Distances the metrics drained across the batch. */
        std::size_t computed_distances = 0;

        /** Re-seedings after a frontier drained before its pool filled. */
        std::size_t restarts = 0;

        /** Walks a scratch or staging refusal ended early, their rankings short. */
        std::size_t saturated_walks = 0;

        /** Neighbors rows read to expand frontiers. */
        nvme_reads_t neighbor_reads;

        /** Value rows read to score. */
        nvme_reads_t value_reads;

        /** Key rows read to resolve written outputs. */
        nvme_reads_t key_reads;

        /** Submit-and-wait cycles of the engines, counted once for every row in flight through
         *  each. */
        std::size_t round_trips = 0;
    };

    nvme_index() = delete;
    explicit nvme_index(forkunion_allocator_t allocator) noexcept
        : media_(allocator), parts_(allocator), allocator_(allocator) {}
    nvme_index(nvme_index&&) noexcept = default;
    nvme_index& operator=(nvme_index&&) noexcept = default;
    nvme_index(nvme_index const&) = delete;
    nvme_index& operator=(nvme_index const&) = delete;

    /** The checkpoint's metadata without viewing it, @ref nvme_read_metadata. */
    static expected<safegraphs_metadata_t> metadata(std::string_view path) noexcept { return nvme_read_metadata(path); }

    /**
     *  @brief Opens a manifest or a monolith, whichever its content names, with every part's
     *      header, medium and row directories, refusing parts that disagree with this
     *      instantiation. A failed view leaves the index empty.
     *
     *  @param[in] path A monolith, or the manifest whose parts sit beside it.
     *  @param[in] paths Where to serve each part from instead, one per part: a directory holding
     *      the part's file, or the file itself; empty serves every part where the manifest points.
     *  @param[in] metric The token every part must name, or empty to take any.
     *  @param[inout] executor Builds the parts' directories, one part per thread at a time.
     */
    status_t view(std::string_view path, std::span<std::string_view const> paths, std::string_view metric,
                  executor_t& executor) noexcept {
        media_.clear();
        parts_.clear();
        status_t const viewed = view_(path, paths, metric, executor);
        if (failed(viewed)) media_.clear(), parts_.clear();
        return viewed;
    }

    /** Shards in the view, one per part of the checkpoint, so a monolith views as one shard. */
    std::size_t shards_count() const noexcept { return parts_.size(); }

    /** Entries across every shard. */
    std::size_t size() const noexcept {
        std::size_t total = 0;
        for (part_t const& part : parts_) total += part.nodes;
        return total;
    }

    /** Entries the view hosts, which are exactly those its parts hold, since a view never grows. */
    std::size_t capacity() const noexcept { return size(); }

    /** Value elements across every part, in the units the value columns count: bytes of strings,
     *  or three coordinates a point of clouds. */
    std::size_t value_elements() const noexcept {
        std::size_t total = 0;
        for (part_t const& part : parts_)
            total += part.value_starts.empty() ? part.nodes * (part.value_row_bytes / part.value_element_bytes)
                                               : part.value_starts.select(part.nodes);
        return total;
    }

    /** Neighbor IDs of the widest neighbors row across every part. */
    std::size_t connectivity() const noexcept {
        std::size_t widest = 0;
        for (part_t const& part : parts_) widest = max_of<std::size_t>(widest, part.connectivity);
        return widest;
    }

    /** Bytes the index occupies: the catalog and the row directories, a few bits a row. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = sizeof(nvme_index) + media_.memory_usage() + parts_.memory_usage();
        for (part_t const& part : parts_) bytes += part.value_starts.size_bytes() + part.neighbor_starts.size_bytes();
        return bytes;
    }

    /** The walk breadth @p config asks for, or twice the widest neighbors row when it asks for
     *  none. */
    std::size_t non_zero_expansion_factor(search_config_t const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor : default_expansion_factor(connectivity());
    }

    /**
     *  @brief Finds the nearest entries of @p count queries across every shard, the walks of many
     *      queries in flight on every thread of @p executor, and gathers each query's rankings by
     *      key.
     *
     *  @param[in] count The queries in the batch.
     *  @param[in] make_metric Builds the metric of one row slot, as
     *      `make_metric(scratch_allocator, staging) noexcept → expected<metric>`, placing it on the
     *      slot's arena and staging at least @p staging candidates a round.
     *  @param[in] query_of Builds the query of one batch row, as
     *      `query_of(row, scratch_allocator) noexcept → expected<query>`, anything it needs placed
     *      on the slot's arena, which outlives the row; every walk of the row reopens the metric's
     *      round on it.
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] config The per-shard walk breadth, applied uniformly.
     *  @param[out] keys @b [count,wanted] rankings by key, as many matches per row as their width.
     *  @param[out] distances The same rankings' distances, ascending per row, cast to
     *      @p score_type_.
     *  @param[out] counts Matches written per query.
     *  @param[in] predicate The optional admission filter over keys, a quad at a time or key by
     *      key; it narrows each row to the entries it admits among a ranking as wide as the walk's
     *      pool.
     *  @param[in] judge The optional admission test over stored values, as
     *      `judge(query, values, verdicts, scratch_allocator) noexcept`, handed the codec's views
     *      of up to four landed rows with every verdict admitting; it hears only rows that would
     *      enter the row's ranking, which then holds the nearest entries it admitted across every
     *      walk.
     *
     *  A failed read stops the call, which answers @c io_k; a refused query or metric answers its
     *  own status, its row left short.
     */
    template <typename make_metric_type_, typename query_of_type_, typename score_type_,
              typename predicate_type_ = dummy_predicate_t, typename judge_type_ = dummy_predicate_t>
    expected<search_result_t> search(std::size_t count, make_metric_type_&& make_metric, query_of_type_&& query_of,
                                     executor_t& executor, search_config_t config, std::span<key_t> keys,
                                     std::span<score_type_> distances, std::span<std::size_t> counts,
                                     predicate_type_&& predicate = {}, judge_type_&& judge = {}) noexcept {
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != count || keys.size() != distances.size() || (count && keys.size() % count))
            return {{}, status_t::config_mismatch_k};
        if (!count) return {{}, status_t::success_k};
        using row_t =
            search_row<std::remove_reference_t<make_metric_type_>, std::remove_reference_t<query_of_type_>,
                       std::remove_reference_t<predicate_type_>, std::remove_reference_t<judge_type_>, score_type_>;
        std::size_t const wanted = keys.size() / count, expansion_factor = non_zero_expansion_factor(config);
        constexpr bool filtered = !std::is_same_v<std::remove_cvref_t<predicate_type_>, dummy_predicate_t>;
        typename row_t::inputs_t const inputs {
            &make_metric, &query_of, &predicate, &judge, filtered ? max_of(expansion_factor, wanted) : wanted,
            wanted,       keys,      distances,  counts};
        return walk_rows_<row_t>(to_span(media_), to_span(parts_), executor, count, expansion_factor, inputs, counts);
    }

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
        auto [call, call_status] = begin_scans_(executor);
        if (failed(call_status)) return call_status;
        return scan_keys_(keys, executor, [&](std::size_t position, locator_t) noexcept {
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
     *  @brief Up to @p wanted values of every key of @p keys, and every key's true total: one scan
     *      of the key columns, then one value-row read per kept row.
     *
     *  @param[in] keys The keys to export, repeats and absent ones included.
     *  @param[in] wanted Values kept per key, a key holding more keeping its earliest in shard and
     *      row order.
     *  @param[out] entries The true total under each key, parallel to @p keys; above @p wanted when
     *      its values truncated, zero when absent.
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] on_value Takes every kept value as `on_value(position, rank, thread, view)`, the
     *      codec's view borrowed for the call alone, on the executor thread that read it, in no
     *      particular order.
     */
    template <typename on_value_type_>
    status_t get(std::span<key_t const> keys, std::size_t wanted, std::span<std::size_t> entries, executor_t& executor,
                 on_value_type_&& on_value) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        if (parts_.empty()) return status_t::uninitialized_k;
        std::memset(entries.data(), 0, entries.size_bytes());
        if (keys.empty()) return status_t::success_k;
        auto [call, call_status] = begin_scans_(executor);
        if (failed(call_status)) return call_status;

        // Threads meet a key's rows in any order, so each key keeps its earliest under a lock
        // striped by position, one stripe per thread.
        growing_buffer<locator_t, scratch_allocator_t> kept {scratch_allocator_t {executor.arena(0)}};
        if (failed(kept.resize_uninitialized(keys.size() * wanted))) return status_t::bad_alloc_k;
        std::size_t const stripes_count = executor.size();
        auto* const stripes = reinterpret_cast<fu::spin_mutex_t*>(
            executor.arena(0).allocate(stripes_count * sizeof(fu::spin_mutex_t), alignof(fu::spin_mutex_t)));
        if (!stripes) return status_t::bad_alloc_k;
        std::uninitialized_value_construct_n(stripes, stripes_count);
        auto earlier = [](locator_t first, locator_t second) noexcept {
            return first.shard != second.shard ? first.shard < second.shard : first.node < second.node;
        };
        status_t const scanned = scan_keys_(keys, executor, [&](std::size_t position, locator_t locator) noexcept {
            std::unique_lock<fu::spin_mutex_t> const held {stripes[position % stripes_count]};
            locator_t* const rows = kept.data() + position * wanted;
            std::size_t slot = min_of(entries[position]++, wanted);
            if (slot == wanted && (!wanted || !earlier(locator, rows[wanted - 1]))) return;
            if (slot == wanted) --slot;
            for (; slot && earlier(locator, rows[slot - 1]); --slot) rows[slot] = rows[slot - 1];
            rows[slot] = locator;
        });
        if (failed(scanned)) return scanned;

        status_t failure = status_t::success_k;
        auto read_value = [&](std::size_t read, std::size_t thread, std::size_t) noexcept {
            std::size_t const position = read / wanted, rank = read % wanted;
            if (rank >= min_of(entries[position], wanted)) return;
            if (failed(std::atomic_ref<status_t>(failure).load(std::memory_order_relaxed))) return;
            locator_t const locator = kept[read];
            part_t const& part = parts_[locator.shard];
            std::size_t const bytes = static_cast<std::size_t>(
                row_length_(part, nvme_purpose_t::score_k, locator.node));
            auto const [landed, read_status] = read_(executor, thread, locator.shard,
                                                     row_offset_(part, nvme_purpose_t::score_k, locator.node), bytes);
            if (succeeded(read_status)) {
                on_value(position, rank, thread, codec_t {}.view(bytes_view_t {landed, bytes}));
                return;
            }
            status_t unset = status_t::success_k;
            std::atomic_ref<status_t>(failure).compare_exchange_strong(unset, read_status, std::memory_order_relaxed,
                                                                       std::memory_order_relaxed);
        };
        if (status_t const fanned = executor.for_tasks(keys.size() * wanted, read_value); failed(fanned)) return fanned;
        return failure;
    }

    /**
     *  @brief The keys and values of @p keys.size() rows from row @p first on, numbered across the
     *      parts in part and row order: one key-row and one value-row read per row, spread over the
     *      executor's threads.
     *
     *  @param[in] first The first row read; every row read must lie before @ref size.
     *  @param[out] keys The key of every row read, in row order.
     *  @param[inout] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] on_value Takes every row's value as `on_value(row, thread, view)`, @c row counted
     *      from @p first, the codec's view borrowed for the call alone, on the executor thread that
     *      read it, in no particular order.
     */
    template <typename on_value_type_>
    status_t read_rows(std::size_t first, std::span<key_t> keys, executor_t& executor,
                       on_value_type_&& on_value) const noexcept {
        if (parts_.empty()) return status_t::uninitialized_k;
        if (first + keys.size() > size()) return status_t::config_mismatch_k;
        if (keys.empty()) return status_t::success_k;
        auto [call, call_status] = begin_scans_(executor);
        if (failed(call_status)) return call_status;
        status_t failure = status_t::success_k;
        auto read_row = [&](std::size_t row, std::size_t thread, std::size_t) noexcept {
            if (failed(std::atomic_ref<status_t>(failure).load(std::memory_order_relaxed))) return;
            u32_t shard = 0;
            std::size_t node = first + row;
            while (node >= parts_[shard].nodes) node -= parts_[shard].nodes, ++shard;
            part_t const& part = parts_[shard];
            node_id_t const id = static_cast<node_id_t>(node);
            auto const [key_bytes, key_status] = read_(
                executor, thread, shard, row_offset_(part, nvme_purpose_t::key_k, id), part.key_row_bytes);
            status_t read_status = key_status;
            if (succeeded(read_status)) {
                keys[row] = safegraphs_widened_key<key_t>(key_bytes, part.key_dtype);
                std::size_t const bytes = static_cast<std::size_t>(row_length_(part, nvme_purpose_t::score_k, id));
                auto const [landed, value_status] = read_(executor, thread, shard,
                                                          row_offset_(part, nvme_purpose_t::score_k, id), bytes);
                read_status = value_status;
                if (succeeded(read_status)) {
                    on_value(row, thread, codec_t {}.view(bytes_view_t {landed, bytes}));
                    return;
                }
            }
            status_t unset = status_t::success_k;
            std::atomic_ref<status_t>(failure).compare_exchange_strong(unset, read_status, std::memory_order_relaxed,
                                                                       std::memory_order_relaxed);
        };
        if (status_t const fanned = executor.for_tasks(keys.size(), read_row); failed(fanned)) return fanned;
        return failure;
    }

  private:
    using locator_t = nvme_locator<node_id_t>;
    using scratch_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;
    using candidates_allocator_t = arena_allocator<scored_node_t, forkunion_allocator_t>;
    using results_array_t = sorted_buffer<scored_node_t, typename scored_node_t::closer_t, candidates_allocator_t>;

    /** 8-ary: eight scored nodes of the narrowest distances fill one cache line per level scan. */
    using frontier_queue_t = max_heap_dary<8, scored_node_t, typename scored_node_t::further_t, candidates_allocator_t>;
    using visited_set_t = growing_hash_set<node_id_t, std::hash<node_id_t>, scratch_allocator_t>;

    /** A ragged column's row starts in its own elements, then its end, a few bits a row. */
    using directory_t = st::immutable_elias_fano<u64_t, forkunion_allocator<u64_t>, row_kit_type_>;

    /** Bytes of the key rows one scan read covers, and of the offsets one directory chunk
     *  streams. */
    inline static constexpr std::size_t scan_chunk_bytes_k = std::size_t {1} << 20;

#pragma region Parts

    /** Where one part keeps its columns, its ragged ones located through directories of their row
     *  starts. */
    struct part_t {

        /** Rows the part holds; neighbor IDs at or past it are sentinels. */
        node_id_t nodes = 0;

        /** Neighbor IDs of the widest neighbors row, at least one: a fixed-degree column's width,
         *  or the longest ragged row. */
        u32_t connectivity = 0;

        /** Bytes of one key row. */
        u32_t key_row_bytes = 0;

        /** The key column's dtype, which @ref safegraphs_widened_key reads each row as. */
        dtype_t key_dtype = dtype_t::unknown_k;

        /** Bytes of one value element, the unit the value directory counts in. */
        u32_t value_element_bytes = 0;

        /** Bytes of the widest value row, or of every one when the codec's values are fixed. */
        u64_t value_row_bytes = 0;

        /** Where row zero of the key column starts, in bytes from the part's own start. */
        u64_t keys_begin = 0;

        /** Where row zero of the value column starts. */
        u64_t values_begin = 0;

        /** Where row zero of the neighbors column starts. */
        u64_t neighbors_begin = 0;

        /** Every value row's start in elements, then the column's end; empty when the values are
         *  fixed. */
        directory_t value_starts;

        /** Every neighbors row's start in IDs, then the column's end; empty for a fixed-degree
         *  column. */
        directory_t neighbor_starts;
    };

    /** Where row @p node of the column @p purpose reads starts in @p part, from the part's own
     *  start. */
    static u64_t row_offset_(part_t const& part, nvme_purpose_t purpose, node_id_t node) noexcept {
        switch (purpose) {
        case nvme_purpose_t::score_k:
            return part.values_begin + (part.value_starts.empty()
                                            ? u64_t(node) * part.value_row_bytes
                                            : part.value_starts.select(node) * part.value_element_bytes);
        case nvme_purpose_t::expand_k:
            return part.neighbors_begin + (part.neighbor_starts.empty() ? u64_t(node) * part.connectivity
                                                                        : part.neighbor_starts.select(node)) *
                                              sizeof(node_id_t);
        case nvme_purpose_t::key_k: return part.keys_begin + u64_t(node) * part.key_row_bytes;
        }
        return 0;
    }

    /** Bytes of row @p node of the column @p purpose reads in @p part. */
    static u64_t row_length_(part_t const& part, nvme_purpose_t purpose, node_id_t node) noexcept {
        switch (purpose) {
        case nvme_purpose_t::score_k:
            return part.value_starts.empty() ? part.value_row_bytes
                                             : (part.value_starts.select(node + 1) - part.value_starts.select(node)) *
                                                   part.value_element_bytes;
        case nvme_purpose_t::expand_k:
            return (part.neighbor_starts.empty()
                        ? u64_t(part.connectivity)
                        : part.neighbor_starts.select(node + 1) - part.neighbor_starts.select(node)) *
                   sizeof(node_id_t);
        case nvme_purpose_t::key_k: return part.key_row_bytes;
        }
        return 0;
    }

    /** The walker's spelling of @ref row_offset_, for the row @p flight reads. */
    static u64_t row_offset_(part_t const& part, nvme_purpose_t purpose,
                             nvme_flight<node_id_t> const& flight) noexcept {
        return row_offset_(part, purpose, flight.node);
    }

    /** The walker's spelling of @ref row_length_, for the row @p flight reads. */
    static u64_t row_length_(part_t const& part, nvme_purpose_t purpose,
                             nvme_flight<node_id_t> const& flight) noexcept {
        return row_length_(part, purpose, flight.node);
    }

#pragma endregion Parts

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
     *  any part holds. */
    static std::size_t staging_bytes_(std::span<medium_t const> media, std::span<part_t const> parts) noexcept {
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
     *  walk @p expansion_factor wide, on @p threads engines @p queue_depth reads deep, every row in
     *  flight staging @p staged_bytes. */
    template <typename row_type_>
    static walk_shape_t walk_shape_(std::span<medium_t const> media, std::span<part_t const> parts,
                                    std::size_t queue_depth, std::size_t threads, std::size_t rows_count,
                                    std::size_t expansion_factor, std::size_t staged_bytes) noexcept {
        using slice_t = arena_allocator<byte_t>;
        // Parts differ in their widest rows, so the widest of all sizes every walk.
        std::size_t const connectivity = nvme_widest_row(parts);
        walk_shape_t shape;
        shape.rows_count = rows_count;
        shape.expansion_factor = expansion_factor;
        shape.queue_sizes = nvme_queue_sizes(expansion_factor, connectivity);
        shape.buffers_per_flight = row_type_::key_reads_k == nvme_key_reads_t::scored_k ? 2 : 1;
        std::size_t const first_hop = max_of<std::size_t>(round_up_to_multiple<std::size_t>(connectivity, 4),
                                                          connectivity + 1);
        std::size_t const hop_depth = parts.size() * first_hop * shape.buffers_per_flight;
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
                              slice_t::reserved_bytes(flights * sizeof(nvme_flight<node_id_t>)) +
                              slice_t::reserved_bytes(flights * sizeof(u32_t)) +
                              slice_t::reserved_bytes(walks * shape.hops_per_walk * sizeof(u32_t)) +
                              slice_t::reserved_bytes(shape.buffers * sizeof(io_request_t)) +
                              slice_t::reserved_bytes(shape.buffers * sizeof(io_completion_t)) + default_alignment_k;
        shape.buffer_bytes = staging_bytes_(media, parts);
        return shape;
    }

    /** Walks @p rows_count rows of @p row_type_ over every part in @p parts inside a call of its
     *  own, each walk @p expansion_factor wide. */
    template <typename row_type_>
    static expected<search_result_t> walk_rows_(std::span<medium_t const> media, std::span<part_t const> parts,
                                                executor_t& executor, std::size_t rows_count,
                                                std::size_t expansion_factor,
                                                typename row_type_::inputs_t const& inputs,
                                                std::span<std::size_t const> counts) noexcept {
        walk_shape_t const shape = walk_shape_<row_type_>(media, parts, executor.queue_depth(), executor.size(),
                                                          rows_count, expansion_factor,
                                                          row_type_::staged_bytes(inputs));
        auto [call, call_status] = nvme_begin_call(executor, media, shape.scratch_bytes, shape.buffers,
                                                   shape.buffer_bytes);
        if (failed(call_status)) return {{}, call_status};
        return row_walker<nvme_index, row_type_>::run_walks(media, parts, executor, shape, inputs, counts, {});
    }

    /** Opens the round of a primed @p row on its query, which every walk of the row then
     *  reuses. */
    template <typename row_type_>
    static void prime_scoring_(walk_shape_t const&, row_type_& row) noexcept {
        if (succeeded(row.status)) row.metric.reset(row.query);
    }

    /** Hands the staged flights of @p walk to its row's metric as the codec's views of their landed
     *  values, drains them within the pool's radius, at a refusal and once all are staged, lets a
     *  judging row test them, releases them and ranks them through @p walker; a metric refusing a
     *  lone value saturates the walk. */
    template <typename walker_type_>
    [[gnu::always_inline]] static void score_staged_(walker_type_& walker, walk_t& walk) noexcept {
        using row_t = typename walker_type_::row_t;
        row_t& row = walker.rows_[walk.row_slot];
        // A full pool only admits what beats its furthest member, so that member's distance is the
        // radius.
        distance_t const radius = walk.results.size() == walker.shape_.expansion_factor
                                      ? walk.results.top().distance
                                      : unbounded_radius<distance_t>();
        auto landed_value = [&](u32_t flight) noexcept {
            return codec_t {}.view(bytes_view_t {
                walker.landed_row_(flight, nvme_purpose_t::score_k, flight),
                static_cast<std::size_t>(walker.row_bytes_(walker.flights_[flight], nvme_purpose_t::score_k))});
        };
        std::array<distance_t, 4> distances;
        std::size_t staged = 0, drained = 0;
        auto drain = [&]() noexcept {
            row.metric.drain(std::span<distance_t>(distances).subspan(drained, staged - drained), radius);
            drained = staged;
        };
        for (u32_t const flight : walk.staged) {
            if (row.metric.push(landed_value(flight)) == metric_staging_t::refused_k) {
                drain();
                if (row.metric.push(landed_value(flight)) == metric_staging_t::refused_k) break;
            }
            ++staged;
        }
        if (staged != drained) drain();
        walker.telemetry_.computed_distances += staged;
        std::array<scored_node_t, 4> scored_batch;
        for (std::size_t lane = 0; lane != staged; ++lane)
            scored_batch[lane] = scored_node_t {walker.flights_[walk.staged[lane]].node, distances[lane]};
        if constexpr (row_t::judged_k) {
            std::array<decltype(landed_value(0)), 4> values;
            for (std::size_t lane = 0; lane != staged; ++lane) values[lane] = landed_value(walk.staged[lane]);
            row.admit(walk.shard, std::span<scored_node_t const>(scored_batch).first(staged),
                      std::span<decltype(landed_value(0)) const>(values).first(staged));
        }
        bool const refused = staged != walk.staged.size();
        for (u32_t const flight : walk.staged) walker.release_flight_(flight);
        walk.staged.clear();
        std::span<scored_node_t> const scored = std::span(scored_batch).first(staged);
        network_sort(scored, typename scored_node_t::closer_t {});
        walker.rank_(walk, scored);
        if (refused) walker.saturate_(walk);
    }

#pragma endregion Walks

#pragma region Rows

    /** The rows of a batched search across the shards: each walk's pool merges into the row's
     *  ranking by locator, or with a judge every admitted row as it is scored, and the ranked
     *  locators resolve into keys once every walk finished, then pass the predicate into the
     *  outputs. Each slot keeps one metric from the caller's factory and the query of its current
     *  row. */
    template <typename make_metric_type_, typename query_of_type_, typename predicate_type_, typename judge_type_,
              typename score_type_>
    struct search_row {
        using key_t = key_type_;
        using metric_t =
            typename std::invoke_result_t<make_metric_type_&, scratch_allocator_t const&, std::size_t>::value_t;
        using query_t =
            typename std::invoke_result_t<query_of_type_&, std::size_t, scratch_allocator_t const&>::value_t;
        inline static constexpr nvme_key_reads_t key_reads_k = nvme_key_reads_t::outputs_k;
        inline static constexpr bool judged_k = !std::is_same_v<std::remove_cv_t<judge_type_>, dummy_predicate_t>;

        /** Candidates a metric stages per round: one quad of landed values. */
        inline static constexpr std::size_t staging_k = 4;

        /** The batch every row of the call reads and writes. */
        struct inputs_t {
            make_metric_type_* make_metric = nullptr;
            query_of_type_* query_of = nullptr;
            predicate_type_* predicate = nullptr;
            judge_type_* judge = nullptr;
            std::size_t ranked = 0;
            std::size_t wanted = 0;
            std::span<key_t> keys;
            std::span<score_type_> distances;
            std::span<std::size_t> counts;
        };

        inputs_t const& inputs;
        scratch_allocator_t allocator;
        metric_t metric;
        query_t query {};
        status_t status = status_t::success_k;
        growing_buffer<locator_t, scratch_allocator_t> locators;
        growing_buffer<distance_t, scratch_allocator_t> ranked_distances;
        growing_buffer<key_t, scratch_allocator_t> resolved;
        std::size_t row = 0;
        std::size_t found = 0;

        search_row(inputs_t const& inputs, scratch_allocator_t allocator) noexcept
            : search_row(inputs, allocator, (*inputs.make_metric)(allocator, staging_k)) {}

        /** Bytes one slot stages beyond the row itself: its ranking, distances and resolved
         *  keys. */
        static std::size_t staged_bytes(inputs_t const& inputs) noexcept {
            using slice_t = arena_allocator<byte_t>;
            return slice_t::reserved_bytes(inputs.ranked * sizeof(locator_t)) +
                   slice_t::reserved_bytes(inputs.ranked * sizeof(distance_t)) +
                   slice_t::reserved_bytes(inputs.ranked * sizeof(key_t));
        }

        status_t reserve(std::size_t, std::size_t) noexcept {
            if (failed(status)) return status;
            if (failed(locators.resize_uninitialized(inputs.ranked)) ||
                failed(ranked_distances.resize_uninitialized(inputs.ranked)) ||
                failed(resolved.resize_uninitialized(inputs.ranked)))
                return status_t::bad_alloc_k;
            return status_t::success_k;
        }
        void prime(std::size_t new_row) noexcept {
            row = new_row;
            found = 0;
            inputs.counts[row] = 0;
            auto [made, made_status] = (*inputs.query_of)(row, allocator);
            query = std::move(made);
            status = made_status;
        }
        results_array_t const& bound(walk_t const& walk) const noexcept { return walk.results; }
        void finish_walk(walk_t const& walk) noexcept {
            // A judging row ranked what it admitted as each walk scored it.
            if constexpr (!judged_k)
                found = merge_search_results(to_span(walk.results), nvme_shard_locators<node_id_t> {walk.shard},
                                             to_writable_span(locators), to_writable_span(ranked_distances), found);
        }

        /** Ranks the rows of @p scored, landed on @p shard with @p values, that the judge admits,
         *  asking it only about those that would enter the ranking. */
        template <typename value_type_>
        void admit(u32_t shard, std::span<scored_node_t const> scored, std::span<value_type_ const> values) noexcept {
            std::array<scored_node_t, 4> asked;
            std::array<value_type_, 4> asked_values;
            std::size_t asking = 0;
            for (std::size_t lane = 0; lane != scored.size(); ++lane) {
                if (found == inputs.ranked && (!found || ranked_distances[found - 1] < scored[lane].distance)) continue;
                asked[asking] = scored[lane], asked_values[asking] = values[lane], ++asking;
            }
            if (!asking) return;
            std::array<admission_t, 4> verdicts;
            verdicts.fill(admission_t::admit_k);
            (*inputs.judge)(query, std::span<value_type_ const>(asked_values).first(asking),
                            std::span<admission_t>(verdicts).first(asking), allocator);
            std::size_t admitted = 0;
            for (std::size_t lane = 0; lane != asking; ++lane)
                if (verdicts[lane] == admission_t::admit_k) asked[admitted++] = asked[lane];
            std::span<scored_node_t> const ranking = std::span(asked).first(admitted);
            network_sort(ranking, typename scored_node_t::closer_t {});
            found = merge_search_results(std::span<scored_node_t const>(ranking),
                                         nvme_shard_locators<node_id_t> {shard}, to_writable_span(locators),
                                         to_writable_span(ranked_distances), found);
        }
        std::size_t outputs() const noexcept { return found; }
        locator_t locator(std::size_t output) const noexcept { return locators[output]; }
        void resolve(std::size_t output, key_t const& key) noexcept { resolved[output] = key; }
        status_t finish() noexcept {
            std::span<key_t> const keys = inputs.keys.subspan(row * inputs.wanted, inputs.wanted);
            std::span<score_type_> const distances = inputs.distances.subspan(row * inputs.wanted, inputs.wanted);
            std::size_t written = 0;
            for (std::size_t first = 0; first < found && written != inputs.wanted; first += 4) {
                std::size_t const quad = min_of<std::size_t>(4, found - first);
                std::array<admission_t, 4> verdicts;
                verdicts.fill(admission_t::admit_k);
                if constexpr (!std::is_same_v<std::remove_cv_t<predicate_type_>, dummy_predicate_t>)
                    judge_keys(*inputs.predicate, std::span<key_t const>(resolved.data() + first, quad),
                               std::span<admission_t>(verdicts.data(), quad));
                for (std::size_t lane = 0; lane != quad && written != inputs.wanted; ++lane) {
                    if (verdicts[lane] != admission_t::admit_k) continue;
                    keys[written] = resolved[first + lane];
                    distances[written] = static_cast<score_type_>(ranked_distances[first + lane]);
                    ++written;
                }
            }
            inputs.counts[row] = written;
            return status;
        }

      private:
        search_row(inputs_t const& inputs, scratch_allocator_t allocator, expected<metric_t>&& made) noexcept
            : inputs(inputs), allocator(allocator), metric(std::move(made.value)), status(made.status),
              locators(allocator), ranked_distances(allocator), resolved(allocator) {}
    };

#pragma endregion Rows

#pragma region Scans

    /** Bytes of one staging buffer of a scan: a scan chunk's covering read, and any row a walk
     *  reads. */
    std::size_t scan_buffer_bytes_() const noexcept {
        u64_t bytes = staging_bytes_(to_span(media_), to_span(parts_));
        for (medium_t const& medium : media_)
            bytes = max_of(bytes, nvme_covering_bytes(scan_chunk_bytes_k, block_bytes_of(medium)));
        return static_cast<std::size_t>(bytes);
    }

    /** Begins a call of scans over every shard: one staging buffer per engine, read one at a
     *  time. */
    expected<nvme_call<executor_t>> begin_scans_(executor_t& executor) const noexcept {
        return nvme_begin_call(executor, to_span(media_), 0, 1, scan_buffer_bytes_());
    }

    /** Reads the @p bytes at @p begin of part @p part through thread @p thread 's engine into its
     *  staging buffer, blocking, and answers where they start in it; a short read answers
     *  @c io_k. */
    expected<byte_t const*> read_(executor_t& executor, std::size_t thread, u32_t part, u64_t begin,
                                  u64_t bytes) const noexcept {
        engine_t& engine = executor.engine(thread);
        medium_t const& medium = media_[part];
        u64_t const absolute = byte_base_of(medium) + begin;
        io_request_t const request = nvme_covering_read(absolute, bytes, block_bytes_of(medium), part, 0, 0);
        if (status_t const enqueued = engine.enqueue_read(request); failed(enqueued)) return {nullptr, enqueued};
        if (status_t const submitted = engine.submit(io_wait_t::blocking_k); failed(submitted))
            return {nullptr, submitted};
        std::array<io_completion_t, 1> completions;
        u64_t const skipped = absolute - request.offset;
        if (!engine.poll_completions(completions) || completions[0].result < 0 ||
            static_cast<u64_t>(completions[0].result) < skipped + bytes)
            return {nullptr, status_t::io_k};
        return {engine.buffer(0) + skipped, status_t::success_k};
    }

    /** Key rows of part @p part one scan read covers. */
    static std::size_t run_rows_(part_t const& part) noexcept {
        return max_of<std::size_t>(scan_chunk_bytes_k / part.key_row_bytes, 1);
    }

    /** Whether two keys are the same, by their bytes, whatever their type. */
    static bool same_key_(key_t const& first, key_t const& second) noexcept {
        return std::memcmp(&first, &second, sizeof(key_t)) == 0;
    }

    /**
     *  @brief Scans every part's key column in runs, matching every row against an exact table of
     *      @p keys, and calls @p on_hit with the batch position and the row's locator for every
     *      match, repeats included.
     *
     *  The table holds each distinct key's first batch position, the positions sharing it chained
     *  behind, in thread zero's arena; runs are read one at a time per thread, inside a call
     *  @ref begin_scans_ began.
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

        std::size_t tasks = 0;
        for (part_t const& part : parts_) tasks += divide_round_up<std::size_t>(part.nodes, run_rows_(part));
        status_t failure = status_t::success_k;
        auto scan_run = [&](std::size_t task, std::size_t thread, std::size_t) noexcept {
            if (failed(std::atomic_ref<status_t>(failure).load(std::memory_order_relaxed))) return;
            u32_t shard = 0;
            while (task >= divide_round_up<std::size_t>(parts_[shard].nodes, run_rows_(parts_[shard])))
                task -= divide_round_up<std::size_t>(parts_[shard].nodes, run_rows_(parts_[shard])), ++shard;
            part_t const& part = parts_[shard];
            std::size_t const first = task * run_rows_(part),
                              rows = min_of<std::size_t>(run_rows_(part), part.nodes - first);
            auto const [bytes, read_status] = read_(
                executor, thread, shard, row_offset_(part, nvme_purpose_t::key_k, static_cast<node_id_t>(first)),
                u64_t(rows) * part.key_row_bytes);
            if (failed(read_status)) {
                status_t unset = status_t::success_k;
                std::atomic_ref<status_t>(failure).compare_exchange_strong(
                    unset, read_status, std::memory_order_relaxed, std::memory_order_relaxed);
                return;
            }
            for (std::size_t row = 0; row != rows; ++row) {
                key_t const key = safegraphs_widened_key<key_t>(bytes + row * part.key_row_bytes, part.key_dtype);
                std::size_t const slot = slot_of(key);
                for (std::size_t position = heads[slot]; position != empty_k; position = chains[position])
                    on_hit(position, locator_t {shard, static_cast<node_id_t>(first + row)});
            }
        };
        if (status_t const fanned = executor.for_tasks(tasks, scan_run); failed(fanned)) return fanned;
        return failure;
    }

#pragma endregion Scans

#pragma region Opening

    /** The USearch status of a SmashTable one, as a directory build answers it. */
    static status_t status_of_(st::status_t status) noexcept {
        switch (status) {
        case st::status_t::success_k: return status_t::success_k;
        case st::status_t::out_of_memory_heap_k: return status_t::bad_alloc_k;
        case st::status_t::input_output_k: return status_t::io_k;
        case st::status_t::invalid_argument_k: return status_t::malformed_k;
        default: return status_t::internal_k;
        }
    }

    /**
     *  @brief Builds @p directory over the @p rows + 1 prefix sums @p sums of the part @p medium
     *      holds, streamed through @p chunk, and answers the widest row they delimit in @p widest.
     *
     *  @param[in] data_offset Where the part's data segment starts, which @p sums count from.
     *  @param[in] extent Elements the column's data holds, which no sum may pass.
     */
    template <typename chunk_allocator_type_>
    status_t build_directory_(file_medium_t const& medium, prefix_sums_t const& sums, std::size_t data_offset,
                              std::size_t rows, u64_t extent, growing_buffer<u64_t, chunk_allocator_type_>& chunk,
                              directory_t& directory, u64_t& widest) const noexcept {
        std::size_t const sum_bytes = dtype_bytes(sums.dtype);
        if (sums.dtype != dtype_t::u64_k && sums.dtype != dtype_t::u32_k) return status_t::config_mismatch_k;
        if (sums.bytes.size() != (rows + 1) * sum_bytes) return status_t::malformed_k;
        // Reads `count` sums from `first` on into the chunk's front, widening narrower ones back to
        // front so no sum is overwritten before it widened.
        auto read_sums = [&](std::size_t first, std::size_t count) noexcept {
            bytes_span_t const target = bytes_of(chunk.data(), count).first(count * sum_bytes);
            if (status_t const read = medium.read_at(data_offset + sums.bytes.first + first * sum_bytes, target);
                failed(read))
                return read;
            if (sum_bytes == sizeof(u64_t))
                for (std::size_t index = 0; index != count; ++index) chunk[index] = little_endian(chunk[index]);
            else
                for (std::size_t index = count; index-- != 0;) {
                    u32_t narrow;
                    std::memcpy(&narrow, target.data() + index * sizeof(u32_t), sizeof(narrow));
                    chunk[index] = little_endian(narrow);
                }
            return status_t::success_k;
        };
        if (status_t const read = read_sums(rows, 1); failed(read)) return read;
        u64_t const universe = chunk[0];
        if (universe > extent) return status_t::malformed_k;
        u64_t previous = 0;
        widest = 0;
        auto fill = [&](std::size_t first) noexcept -> st::expected<std::span<u64_t const>> {
            std::size_t const count = min_of(chunk.size(), rows + 1 - first);
            if (failed(read_sums(first, count))) return st::status_t::input_output_k;
            for (std::size_t index = 0; index != count; ++index) {
                if (first + index && chunk[index] >= previous) widest = max_of(widest, chunk[index] - previous);
                previous = chunk[index];
            }
            return st::expected<std::span<u64_t const>>(std::span<u64_t const>(chunk.data(), count));
        };
        auto made = directory_t::make(rows + 1, universe, fill, forkunion_allocator<u64_t>(allocator_));
        if (!made) return status_of_(made.status());
        directory = std::move(*made);
        return status_t::success_k;
    }

    /** Reads the header of the part @p medium holds into @p part, vetted against this instantiation
     *  and, unless empty, @p metric, with the directories of its ragged columns streamed through
     *  @p chunk. */
    template <typename chunk_allocator_type_>
    status_t read_part_(file_medium_t const& medium, std::string_view metric,
                        growing_buffer<u64_t, chunk_allocator_type_>& chunk, part_t& part) const noexcept {
        growing_buffer<byte_t, forkunion_allocator_t> header_bytes {allocator_};
        if (status_t const read = nvme_read_header(medium, header_bytes); failed(read)) return read;
        safegraphs_name_t named;
        auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "metric") named = safegraphs_name_t {field.value};
        };
        auto const [header, header_status] = safegraphs_read_header(to_span(header_bytes), allocator_,
                                                                    per_custom_field);
        if (failed(header_status)) return header_status;
        if (!metric.empty() && !named.empty() && named.view() != metric) return status_t::config_mismatch_k;

        // Keys are regular, values shaped as the codec writes them, neighbors ragged or
        // fixed-degree.
        constexpr bool fixed_width = safegraphs_value_is_fixed<codec_t>();
        safegraphs_column_t const& value = header.value;
        safegraphs_column_t const& adjacency = header.neighbors;
        if (header.key.offsets || header.key.shape != column_shape_t {scalar_shape_t {}} ||
            !safegraphs_key_widens(header.key.dtype, safegraphs_key_dtype<key_t>()))
            return status_t::config_mismatch_k;
        if (value.dtype != codec_t::value_dtype_k || value.shape != safegraphs_value_shape<codec_t>() ||
            value.offsets.has_value() == fixed_width)
            return status_t::config_mismatch_k;
        std::size_t const degree = std::holds_alternative<vector_shape_t>(adjacency.shape) && !adjacency.offsets
                                       ? std::get<vector_shape_t>(adjacency.shape).columns
                                       : 0;
        if (adjacency.dtype != safegraphs_node_id_dtype<node_id_t>() ||
            (adjacency.offsets ? adjacency.shape != column_shape_t {scalar_shape_t {}} : !degree))
            return status_t::config_mismatch_k;
        std::size_t const nodes = header.nodes();
        if (value.nodes() != nodes || adjacency.nodes() != nodes || !header.key.row_bytes() ||
            nodes >= (std::numeric_limits<node_id_t>::max)())
            return status_t::malformed_k;

        part.nodes = static_cast<node_id_t>(nodes);
        part.key_row_bytes = static_cast<u32_t>(header.key.row_bytes());
        part.key_dtype = header.key.dtype;
        part.value_element_bytes = static_cast<u32_t>(dtype_bytes(value.dtype));
        part.keys_begin = header.data_offset + header.key.data.first;
        part.values_begin = header.data_offset + value.data.first;
        part.neighbors_begin = header.data_offset + adjacency.data.first;
        if (fixed_width) part.value_row_bytes = value.row_bytes();
        else {
            u64_t widest = 0;
            if (status_t const built = build_directory_(medium, *value.offsets, header.data_offset, nodes,
                                                        value.data.size() / part.value_element_bytes, chunk,
                                                        part.value_starts, widest);
                failed(built))
                return built;
            part.value_row_bytes = widest * part.value_element_bytes;
        }
        if (degree) part.connectivity = static_cast<u32_t>(degree);
        else {
            u64_t widest = 0;
            if (status_t const built = build_directory_(medium, *adjacency.offsets, header.data_offset, nodes,
                                                        adjacency.data.size() / sizeof(node_id_t), chunk,
                                                        part.neighbor_starts, widest);
                failed(built))
                return built;
            // A part whose rows hold no neighbors still seeds its walks from one landmark at least.
            part.connectivity = static_cast<u32_t>(max_of<u64_t>(widest, 1));
        }
        return status_t::success_k;
    }

    /** Opens the part file at @p path: its medium for the engines, then its header and directories
     *  through blocking reads of that medium. */
    template <typename chunk_allocator_type_>
    status_t open_part_(std::string_view path, std::string_view metric,
                        growing_buffer<u64_t, chunk_allocator_type_>& chunk, medium_t& medium,
                        part_t& part) const noexcept {
        if constexpr (!std::is_constructible_v<medium_t, file_medium_t&&>) return status_t::config_mismatch_k;
        else {
            auto [file, open_status] = file_medium_t::open(path);
            if (failed(open_status)) return open_status;
            if (status_t const read = read_part_(file, metric, chunk, part); failed(read)) return read;
            medium = medium_t {std::move(file)};
            return status_t::success_k;
        }
    }

    /** The body of @ref view, into an empty catalog: every part's path resolves on the calling
     *  thread, then the parts open on the executor's threads, one at a time per thread. */
    status_t view_(std::string_view checkpoint, std::span<std::string_view const> paths, std::string_view metric,
                   executor_t& executor) noexcept {
        if (checkpoint.empty()) return status_t::invalid_argument_k;
        if (names_transport_id(checkpoint)) return status_t::unsupported_k;
        std::array<byte_t, safegraphs_prefix_t::magic_k.size()> magic {};
        {
            auto [file, open_status] = seekable_file_t::open(checkpoint, file_mode_t::read_k, allocator_);
            if (failed(open_status)) return open_status;
            [[maybe_unused]] status_t const sniffed = file.read_at(0, magic);
        }
        // The magic makes a lone monolith, anything else a manifest naming its parts beside it.
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

        // Every resolved path lands in one buffer, terminated for the OS, so the threads read them
        // as they are.
        std::string_view const checkpoint_name = safegraphs_basename(checkpoint);
        std::string_view const checkpoint_directory = checkpoint.substr(0, checkpoint.size() - checkpoint_name.size());
        growing_buffer<char, forkunion_allocator_t> resolved {allocator_};
        growing_buffer<char, forkunion_allocator_t> resolved_paths {allocator_};
        growing_buffer<iota_view<std::size_t>, forkunion_allocator_t> resolved_entries {allocator_};
        if (failed(resolved_entries.reserve_exactly(entries.size()))) return status_t::bad_alloc_k;
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
            std::size_t const first = resolved_paths.size();
            for (char const character : resolved)
                if (failed(resolved_paths.push_back(character))) return status_t::bad_alloc_k;
            resolved_entries.push_back_reserved(iota_view<std::size_t> {first, resolved_paths.size() - 1});
        }

        if (failed(media_.resize(entries.size())) || failed(parts_.resize(entries.size())))
            return status_t::bad_alloc_k;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        if (status_t const reserved = executor.reserve(scan_chunk_bytes_k + default_alignment_k, 0, 0);
            failed(reserved))
            return reserved;
        alignas(default_alignment_k) std::size_t cursor = 0;
        status_t failure = status_t::success_k;
        auto open_parts = [&](std::size_t thread, std::size_t) noexcept {
            growing_buffer<u64_t, scratch_allocator_t> chunk {scratch_allocator_t {executor.arena(thread)}};
            status_t opened = chunk.resize_uninitialized(scan_chunk_bytes_k / sizeof(u64_t));
            while (succeeded(opened) && succeeded(std::atomic_ref<status_t>(failure).load(std::memory_order_relaxed))) {
                std::size_t const part = std::atomic_ref<std::size_t>(cursor).fetch_add(1, std::memory_order_relaxed);
                if (part >= parts_.size()) return;
                std::string_view const path {resolved_paths.data() + resolved_entries[part].first,
                                             resolved_entries[part].size()};
                opened = open_part_(path, metric, chunk, media_[part], parts_[part]);
            }
            status_t unset = status_t::success_k;
            if (failed(opened))
                std::atomic_ref<status_t>(failure).compare_exchange_strong(unset, opened, std::memory_order_relaxed,
                                                                           std::memory_order_relaxed);
        };
        if (status_t const fanned = executor.for_threads(open_parts); failed(fanned)) return fanned;
        return failure;
    }

#pragma endregion Opening

    /** Every part's file, open for the engines, in part order. */
    growing_buffer<medium_t, forkunion_allocator_t> media_;

    /** Where every part keeps its columns and its row directories, parallel to @c media_. */
    growing_buffer<part_t, forkunion_allocator_t> parts_;

    /** The host allocator behind the catalog, the directories and the paths a view resolves. */
    forkunion_allocator_t allocator_;
};

} // namespace unum::usearch::basic
