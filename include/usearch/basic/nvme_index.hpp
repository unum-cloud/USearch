/**
 *  @file include/usearch/basic/nvme_index.hpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief Similarity Search over one pre-constructed read-only basic graph on block storage: ragged
 *      values under caller-supplied metrics, walked by the storage walk the dense tier drives.
 *
 *  An index opens one part on one medium from the @c persisted_columns_t any producer of a format
 *  fills, and keeps its columns' offsets and a directory of every ragged column's row starts: an
 *  Elias-Fano sequence of a few bits per row, so a walk locates any value or neighbors row in RAM
 *  and reads it with one covering read, as the dense tier reads its fixed rows. The directories are
 *  built at the opening by streaming each prefix sums column in chunks, and the widest rows, which
 *  size the staging buffers, come from the same pass. A fixed-degree neighbors column needs no
 *  directory. The walks live in @c nvme_parts_catalog, a component over spans of the media and
 *  parts its owner holds, so @c nvme_index over one part and @c nvme_indexes over many run the same
 *  code.
 *
 *  A search keeps one metric per query slot, built by the caller's factory on the slot's arena,
 *  since every walk of a query shares it, and hands it each landed value as the codec's borrowed
 *  @c view. A caller's relation reads the same views before the flights recycle, so a relation
 *  tested on stored values needs no second read. Key lookups scan the key columns, and @c get reads
 *  the value rows of the keys it found.
 */
#pragma once

#include <cstring> // `std::memcmp`, `std::memset`

#include <array>       // `std::array` - a quad's distances, scored rows and verdicts
#include <concepts>    // `std::same_as`
#include <limits>      // `std::numeric_limits` - the node ID ceiling
#include <memory>      // `std::uninitialized_value_construct_n` - the lock stripes of `get`
#include <mutex>       // `std::unique_lock` - one stripe's hold
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::remove_cvref_t`, `std::invoke_result_t`, `std::is_same_v`, `std::is_constructible_v`
#include <utility>     // `std::move`

#include <forkunion/atomics.hpp>               // `fu::standard_atomic_ref` - the default reference
#include <forkunion/types.hpp>                 // `fu::standard_yield_t` - the default waiter
#include <smashtable/immutable_elias_fano.hpp> // `st::immutable_elias_fano` - the row directories

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/nvme_device.hpp"       // `nvme_executor`, `file_medium_t`, `io_request_t`
#include "usearch/utilities/nvme_walks.hpp"        // `query_walker`, `nvme_begin_call`, `nvme_read_header`
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`, `sorted_buffer`, `arena_allocator`
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator`
#include "usearch/utilities/ram_sorts.hpp"         // `network_sort`
#include "usearch/utilities/ram_tables.hpp"        // `growing_hash_set`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_node_id_dtype`, `safegraphs_widened_key`
#include "usearch/basic/ram_index.hpp"             // `scored_node`, `metric_staging_t`, `unbounded_radius`

namespace unum::usearch::basic {

/**
 *  @brief The walks, scans and key lookups of basic graphs over parts on block storage, as one
 *      component over spans of the media and the parts its owner holds: @c nvme_index owns one
 *      part, @c nvme_indexes many, and each hands its own spans to a catalog for the call.
 *
 *  A catalog walks every part of a query on one thread, merging the parts' rankings by key; it owns
 *  nothing, so it lives no longer than the call its owner builds it for.
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
          typename row_kit_type_ = st::serial_row_kit_t,
          template <typename> class atomic_ref_template_ = fu::standard_atomic_ref,
          typename wait_type_ = fu::standard_yield_t>
    requires std::same_as<node_id_type_, u32_t> || std::same_as<node_id_type_, u64_t>
class nvme_parts_catalog {
    template <typename, typename, typename>
    friend class utilities::query_walker;

  public:
    using codec_t = codec_type_;
    using key_t = key_type_;
    using distance_t = distance_type_;
    using engine_t = engine_type_;
    using executor_t = nvme_executor<engine_t>;
    using medium_t = typename engine_t::medium_t;
    using arena_t = typename executor_t::arena_t;

    /** The CPU class's reference over the words one call's threads share, and its waiter, as
     *  @c src/ picked them. */
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;
    using wait_t = wait_type_;

    /** Node identifier within one part. */
    using node_id_t = node_id_type_;
    using scored_node_t = scored_node<node_id_t, distance_t>;

    /** What a scan hands an aggregating predicate beside each key: the entry within its run. */
    using scan_payload_t = u32_t;

    /** What a walk hands an aggregating predicate beside each key it gathers. */
    using walk_payload_t = nvme_walk_payload<node_id_t, distance_t>;

    /** Per-search knobs; zero derives from the parts' connectivity. */
    struct search_config_t {

        /** How many candidates each walk keeps; defaults to twice the widest neighbors row. */
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

        /** Re-seedings after a frontier drained before its candidates filled. */
        std::size_t restarts = 0;

        /** Walks a scratch or staging refusal ended early, their rankings short. */
        std::size_t saturated_walks = 0;

        /** Neighbors rows read to expand frontiers. */
        nvme_reads_t neighbor_reads;

        /** Value rows read to score. */
        nvme_reads_t value_reads;

        /** Key rows read to resolve written outputs. */
        nvme_reads_t key_reads;

        /** Submit-and-wait cycles of the engines, counted once for every query in flight through
         *  each. */
        std::size_t round_trips = 0;
    };

    /** Parts the catalog walks. */
    std::size_t parts_count() const noexcept { return parts_.size(); }

    /** Entries across every shard. */
    std::size_t size() const noexcept {
        std::size_t total = 0;
        for (part_t const& part : parts_) total += part.nodes;
        return total;
    }

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
     *  @param[in] make_metric Builds the metric of one query slot, as
     *      <tt>make_metric(scratch_allocator, staging) noexcept → expected<metric></tt>, placing it
     *      on the slot's arena and staging at least @p staging candidates a round.
     *  @param[in] query_of Builds the query of one batch query, as <tt>query_of(query,
     *      scratch_allocator) noexcept → expected<query></tt>, anything it needs placed on the
     *      slot's arena, which outlives the query; every walk of the query reopens the metric's
     *      round on it.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] config The per-shard walk breadth, applied uniformly.
     *  @param[out] keys @b [count,wanted] rankings by key, as many matches per query as their
     *      width.
     *  @param[out] distances The same rankings' distances, ascending per query, cast to
     *      @p score_type_.
     *  @param[out] counts Matches written per query.
     *  @param[in] predicate The optional admission filter over keys, asked a quad at a time as the
     *      walks score, or a family of such per executor thread, which an aggregating one must be,
     *      gathering over each completion batch: every scored entry's key lands beside its value,
     *      rejected entries still route the walks, and only what it admits enters the query's
     *      ranking.
     *  @param[in] relation The optional admission test over stored values, as <tt>relation(query,
     *      values, admissions) noexcept</tt>, or a family of such per executor thread, handed the
     *      codec's views of up to four landed values with every admission admitting; it hears only
     *      entries that could still enter their walk's admitted set, each walk stops once that set
     *      fills, and the query then holds the nearest entries both admitted across every walk.
     *
     *  A failed read stops the call, which answers @c io_k; a refused query or metric answers its
     *  own status, its matches left short.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No part is viewed, or @p executor holds no threads or pool workers.
     *  - @c config_mismatch_k @p counts is not @p count long, @p keys and @p distances differ in
     *    length or do not split into @p count queries, or a medium the engines cannot read.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    template <typename make_metric_type_, typename query_of_type_, typename score_type_,
              typename predicate_type_ = dummy_predicate_t, typename relation_type_ = dummy_predicate_t>
    expected<search_result_t> search(std::size_t count, make_metric_type_&& make_metric, query_of_type_&& query_of,
                                     executor_t& executor, search_config_t config, std::span<key_t> keys,
                                     std::span<score_type_> distances, std::span<std::size_t> counts,
                                     predicate_type_&& predicate = {}, relation_type_&& relation = {}) noexcept {
        static_assert(!aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, walk_payload_t> ||
                          per_thread_predicates<std::remove_cvref_t<predicate_type_>>,
                      "An engine driving its own threads takes an aggregating predicate per thread");
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != count || keys.size() != distances.size() || (count && keys.size() % count))
            return {{}, status_t::config_mismatch_k};
        if (!count) return {{}, status_t::success_k};
        using query_t = search_query<std::remove_reference_t<make_metric_type_>,
                                     std::remove_reference_t<query_of_type_>, std::remove_reference_t<predicate_type_>,
                                     std::remove_reference_t<relation_type_>, score_type_>;
        std::size_t const wanted = keys.size() / count, expansion_factor = non_zero_expansion_factor(config);
        typename query_t::inputs_t const inputs {&make_metric, &query_of, &predicate, &relation, wanted,
                                                 wanted,       keys,      distances,  counts};
        return walk_queries_<query_t>(to_span(media_), to_span(parts_), executor, count, expansion_factor, inputs,
                                      counts);
    }

    /**
     *  @brief The exact sibling of @c search: every entry read once and scored against every
     *      query, each thread keeping its own nearest entries of every query until they merge and
     *      their keys resolve.
     *
     *  The threads take chunks of entries and read each chunk's value rows, and its key rows when
     *  @p predicate hears keys, as one run apiece. Takes what @ref search takes but a breadth.
     *  @p predicate hears an entry once, when it would enter some query's ranking on the reading
     *  thread, and @p relation then hears it for each query it would enter, so a query holds the
     *  nearest entries both admit. A failed read stops the call, which answers @c io_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No part is viewed, or @p executor holds no threads or pool workers.
     *  - @c config_mismatch_k @p counts is not @p count long, @p keys and @p distances differ in
     *    length or do not split into @p count queries, or a medium the engines cannot read.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    template <typename make_metric_type_, typename query_of_type_, typename score_type_,
              typename predicate_type_ = dummy_predicate_t, typename relation_type_ = dummy_predicate_t>
    expected<search_result_t> scan(std::size_t count, make_metric_type_&& make_metric, query_of_type_&& query_of,
                                   executor_t& executor, std::span<key_t> keys, std::span<score_type_> distances,
                                   std::span<std::size_t> counts, predicate_type_&& predicate = {},
                                   relation_type_&& relation = {}) noexcept {
        static_assert(!aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, scan_payload_t> ||
                          per_thread_predicates<std::remove_cvref_t<predicate_type_>>,
                      "An engine driving its own threads takes an aggregating predicate per thread");
        using query_t =
            typename std::invoke_result_t<query_of_type_&, std::size_t, scratch_allocator_t const&>::value_t;
        constexpr bool filtered_k = !std::is_same_v<std::remove_cvref_t<predicate_type_>, dummy_predicate_t>;
        constexpr nvme_key_reads_t key_reads_k = filtered_k ? nvme_key_reads_t::scored_k : nvme_key_reads_t::outputs_k;
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != count || keys.size() != distances.size() || (count && keys.size() % count))
            return {{}, status_t::config_mismatch_k};
        std::memset(counts.data(), 0, counts.size_bytes());
        std::size_t const wanted = count ? keys.size() / count : 0;
        if (!count || !wanted || !size()) return {{}, status_t::success_k};

        // Every round of a run scores all queries, and an entry that would enter some query's
        // ranking is asked of the predicate once.
        auto rank_round = [&](std::size_t thread, scan_share_t& share, scan_round_t const& round,
                              std::span<query_t const> queries) noexcept {
            auto&& admits = thread_predicate(predicate, thread);
            std::size_t const round_rows = round.end_row - round.first_row;
            auto distance_of = [&](std::size_t query, std::size_t row) noexcept {
                return share.round_distances[query * round_rows + row - round.first_row];
            };
            auto keep_entry = [&](scan_payload_t entry) noexcept {
                value_view_t const view = round(entry);
                for (std::size_t query = 0; query != count; ++query) {
                    distance_t const distance = distance_of(query, entry);
                    if (!(distance < share.thresholds[query])) continue;
                    admission_t related = admission_t::admit_k;
                    thread_predicate(relation, thread)(queries[query], std::span<value_view_t const, 1>(&view, 1),
                                                       std::span<admission_t, 1>(&related, 1));
                    if (related == admission_t::admit_k)
                        keep_ranked_(share, query, wanted, round.locator_of(entry), distance);
                }
            };
            quad<scan_payload_t> contenders;
            quad<key_t> contender_keys;
            std::size_t filled = 0;
            for (std::size_t row = round.first_row; row != round.end_row; ++row) {
                bool contends = false;
                for (std::size_t query = 0; query != count && !contends; ++query)
                    contends = distance_of(query, row) < share.thresholds[query];
                if (!contends) continue;
                contenders[filled] = static_cast<scan_payload_t>(row);
                if (filtered_k) contender_keys[filled] = round.key_of(row);
                if (++filled != quad_lanes_k) continue;
                ask_quad(admits, std::span<key_t const, quad_lanes_k>(contender_keys),
                         std::span<scan_payload_t const, quad_lanes_k>(contenders), keep_entry);
                filled = 0;
            }
            with_static_extent<quad_lanes_k>(filled, [&]<std::size_t extent_>() noexcept {
                ask_quad(admits, std::span<key_t const, extent_>(contender_keys.data(), extent_),
                         std::span<scan_payload_t const, extent_>(contenders.data(), extent_), keep_entry);
            });
            settle_predicate(admits, keep_entry);
        };

        // Every query merges the threads' rankings into its ranking, the nearest kept, ascending,
        // and each kept entry then reads its key.
        std::size_t found = 0;
        auto merge_shares = [&](std::span<scan_share_t const> shares) noexcept {
            growing_buffer<locator_t, scratch_allocator_t> ranked {scratch_allocator_t {executor.arena(0)}};
            if (failed(ranked.resize_uninitialized(count * wanted))) return status_t::bad_alloc_k;
            for (std::size_t query = 0; query != count; ++query) {
                locator_t* const query_locators = ranked.data() + query * wanted;
                score_type_* const query_distances = distances.data() + query * wanted;
                std::size_t written = 0;
                for (scan_share_t const& share : shares)
                    for (std::size_t slot = query * wanted; slot != query * wanted + share.kept_counts[query]; ++slot) {
                        score_type_ const distance = static_cast<score_type_>(share.kept_distances[slot]);
                        if (written == wanted && !(distance < query_distances[wanted - 1])) continue;
                        std::size_t position = written == wanted ? wanted - 1 : written++;
                        for (; position && distance < query_distances[position - 1]; --position)
                            query_locators[position] = query_locators[position - 1],
                            query_distances[position] = query_distances[position - 1];
                        query_locators[position] = share.kept_locators[slot], query_distances[position] = distance;
                    }
                counts[query] = written;
                found += written;
            }
            status_t resolved = status_t::success_k;
            auto resolve_key = [&](std::size_t match, std::size_t thread, std::size_t) noexcept {
                if (match % wanted >= counts[match / wanted]) return;
                if (failed(atomic_ref<status_t>(resolved).load(std::memory_order_relaxed))) return;
                locator_t const locator = ranked[match];
                part_t const& part = parts_[locator.shard];
                auto const [key_row, key_status] = read_(executor, thread, locator.shard,
                                                         row_offset_(part, nvme_purpose_t::key_k, locator.node),
                                                         part.key_row_bytes);
                if (succeeded(key_status)) {
                    keys[match] = safegraphs_widened_key<key_t>(key_row, part.key_dtype);
                    return;
                }
                status_t unset = status_t::success_k;
                atomic_ref<status_t>(resolved).compare_exchange_strong(unset, key_status, std::memory_order_relaxed,
                                                                       std::memory_order_relaxed);
            };
            if (status_t const fanned = executor.for_tasks(count * wanted, resolve_key); failed(fanned)) return fanned;
            return resolved;
        };
        status_t const scanned = scan_runs_<key_reads_k>(count, wanted, {}, predicate, make_metric, query_of, executor,
                                                         rank_round, merge_shares);
        if (failed(scanned)) return {{}, scanned};
        search_result_t result;
        result.count = found;
        result.computed_distances = size() * count;
        return {result, status_t::success_k};
    }

    /**
     *  @brief The containment sibling of @c scan: every entry read once beside its key and scored
     *      against every query, each entry strictly closer to a query than its reach handed to
     *      @p land on the reading thread instead of ranked.
     *
     *  Takes what @ref scan takes but the rankings and the filters, which @p land applies.
     *
     *  @param[in] reaches Every query's exclusive distance bound, one per query.
     *  @param[in] make_metric Builds the metric of one query slot, as @ref scan takes it.
     *  @param[in] query_of Builds the query of one batch query, as @ref scan takes it.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] land Takes a quad of one query's entries within reach as <tt>land(thread, query,
     *      keys, values, distances) noexcept</tt>, three spans of one static extent in run order,
     *      the codec's views borrowed for the call alone.
     *
     *  A failed read stops the call, which answers @c io_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No part is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     *  - @c config_mismatch_k A medium the engines cannot read.
     */
    template <typename make_metric_type_, typename query_of_type_, typename land_type_>
    expected<search_result_t> scan_within(std::span<distance_t const> reaches, make_metric_type_&& make_metric,
                                          query_of_type_&& query_of, executor_t& executor, land_type_&& land) noexcept {
        using query_t =
            typename std::invoke_result_t<query_of_type_&, std::size_t, scratch_allocator_t const&>::value_t;
        std::size_t const count = reaches.size();
        if (parts_.empty()) return {{}, status_t::uninitialized_k};
        if (!count || !size()) return {{}, status_t::success_k};

        // Every query's entries within reach of a round land a quad at a time.
        auto land_round = [&](std::size_t thread, scan_share_t& share, scan_round_t const& round,
                              std::span<query_t const>) noexcept {
            std::size_t const round_rows = round.end_row - round.first_row;
            for (std::size_t query = 0; query != count; ++query) {
                std::span<distance_t const> const distances =
                    to_span(share.round_distances).subspan(query * round_rows, round_rows);
                quad<key_t> within_keys;
                quad<value_view_t> within_values;
                quad<distance_t> within_distances;
                std::size_t filled = 0;
                auto land_quad = [&]<std::size_t extent_>() noexcept {
                    land(thread, query, std::span<key_t const, extent_>(within_keys.data(), extent_),
                         std::span<value_view_t const, extent_>(within_values.data(), extent_),
                         std::span<distance_t const, extent_>(within_distances.data(), extent_));
                };
                for (std::size_t row = 0; row != round_rows; ++row) {
                    if (!(distances[row] < reaches[query])) continue;
                    within_keys[filled] = round.key_of(round.first_row + row);
                    within_values[filled] = round(round.first_row + row);
                    within_distances[filled] = distances[row];
                    if (++filled != quad_lanes_k) continue;
                    land_quad.template operator()<quad_lanes_k>();
                    filled = 0;
                }
                with_static_extent<quad_lanes_k>(filled, land_quad);
            }
        };
        auto merge_nothing = [](std::span<scan_share_t const>) noexcept { return status_t::success_k; };
        dummy_predicate_t unfiltered;
        status_t const scanned = scan_runs_<nvme_key_reads_t::scored_k>(count, 0, reaches, unfiltered, make_metric,
                                                                        query_of, executor, land_round, merge_nothing);
        if (failed(scanned)) return {{}, scanned};
        search_result_t result;
        result.computed_distances = size() * count;
        return {result, status_t::success_k};
    }

    /**
     *  @brief The entries stored under every key of @p keys: one scan over every part's key column,
     *      matched against an exact table of the batch.
     *
     *  @param[in] keys The keys to count, repeats and absent ones included.
     *  @param[out] entries The entries under each key, parallel to @p keys; zero means absent.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p entries and @p keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c uninitialized_k No part is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        if (parts_.empty()) return status_t::uninitialized_k;
        std::memset(entries.data(), 0, entries.size_bytes());
        if (keys.empty()) return status_t::success_k;
        auto [call, call_status] = begin_scans_(executor);
        if (failed(call_status)) return call_status;
        return scan_keys_(keys, executor, [&](std::size_t position, locator_t) noexcept {
            atomic_ref<std::size_t>(entries[position]).add(1, std::memory_order_relaxed);
        });
    }

    /**
     *  The batched @ref count reduced to one in @p entries where a key of @p keys holds any entry,
     *  zero where it holds none.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p entries and @p keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c uninitialized_k No part is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    status_t contains(std::span<key_t const> keys, std::span<std::size_t> entries,
                      executor_t& executor) const noexcept {
        if (status_t const counted = count(keys, entries, executor); failed(counted)) return counted;
        for (std::size_t& entry : entries) entry = entry ? 1 : 0;
        return status_t::success_k;
    }

    /**
     *  @brief Up to @p wanted values of every key of @p keys, and every key's true total: one scan
     *      of the key columns, then one value-row read per kept entry.
     *
     *  @param[in] keys The keys to export, repeats and absent ones included.
     *  @param[in] wanted Values kept per key, a key holding more keeping its earliest in shard and
     *      node order.
     *  @param[out] entries The true total under each key, parallel to @p keys; above @p wanted when
     *      its values truncated, zero when absent.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] on_value Takes every kept value as <tt>on_value(position, rank, thread,
     *      view)</tt>, the codec's view borrowed for the call alone, on the executor thread that
     *      read it, in no particular order.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p entries and @p keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c uninitialized_k No part is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
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

        // Threads meet a key's entries in any order, so each key keeps its earliest under a lock
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
            if (failed(atomic_ref<status_t>(failure).load(std::memory_order_relaxed))) return;
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
            atomic_ref<status_t>(failure).compare_exchange_strong(unset, read_status, std::memory_order_relaxed,
                                                                  std::memory_order_relaxed);
        };
        if (status_t const fanned = executor.for_tasks(keys.size() * wanted, read_value); failed(fanned)) return fanned;
        return failure;
    }

    /**
     *  @brief The keys and values of @p keys.size() entries from entry @p first on, numbered across
     *      the parts in part and node order: one key-row and one value-row read per entry, spread
     *      over the executor's threads.
     *
     *  @param[in] first The first entry read; every entry read must lie before @ref size.
     *  @param[out] keys The key of every entry read, in entry order.
     *  @param[in,out] executor The threads, their engines and their scratch; held for the call.
     *  @param[in] on_value Takes every entry's value as <tt>on_value(entry, thread, view)</tt>,
     *      @c entry counted from @p first, the codec's view borrowed for the call alone, on the
     *      executor thread that read it, in no particular order.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No part is viewed, or @p executor holds no threads or pool workers.
     *  - @c config_mismatch_k The entries run past @ref size, or a medium the engines cannot read.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    template <typename on_value_type_>
    status_t read_entries(std::size_t first, std::span<key_t> keys, executor_t& executor,
                          on_value_type_&& on_value) const noexcept {
        if (parts_.empty()) return status_t::uninitialized_k;
        if (first + keys.size() > size()) return status_t::config_mismatch_k;
        if (keys.empty()) return status_t::success_k;
        auto [call, call_status] = begin_scans_(executor);
        if (failed(call_status)) return call_status;
        status_t failure = status_t::success_k;
        auto read_entry = [&](std::size_t entry, std::size_t thread, std::size_t) noexcept {
            if (failed(atomic_ref<status_t>(failure).load(std::memory_order_relaxed))) return;
            u32_t shard = 0;
            std::size_t node = first + entry;
            while (node >= parts_[shard].nodes) node -= parts_[shard].nodes, ++shard;
            part_t const& part = parts_[shard];
            node_id_t const id = static_cast<node_id_t>(node);
            auto const [key_bytes, key_status] = read_(
                executor, thread, shard, row_offset_(part, nvme_purpose_t::key_k, id), part.key_row_bytes);
            status_t read_status = key_status;
            if (succeeded(read_status)) {
                keys[entry] = safegraphs_widened_key<key_t>(key_bytes, part.key_dtype);
                std::size_t const bytes = static_cast<std::size_t>(row_length_(part, nvme_purpose_t::score_k, id));
                auto const [landed, value_status] = read_(executor, thread, shard,
                                                          row_offset_(part, nvme_purpose_t::score_k, id), bytes);
                read_status = value_status;
                if (succeeded(read_status)) {
                    on_value(entry, thread, codec_t {}.view(bytes_view_t {landed, bytes}));
                    return;
                }
            }
            status_t unset = status_t::success_k;
            atomic_ref<status_t>(failure).compare_exchange_strong(unset, read_status, std::memory_order_relaxed,
                                                                  std::memory_order_relaxed);
        };
        if (status_t const fanned = executor.for_tasks(keys.size(), read_entry); failed(fanned)) return fanned;
        return failure;
    }

  private:
    using locator_t = nvme_locator<node_id_t>;

    using scratch_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;

    /** One thread's part of an exact scan, on its own arena: every query's ranking, a max-heap of
     *  locators beside their distances, how many each holds and its farthest once full, and one
     *  round's @b [queries,rows] distances. */
    struct scan_share_t {
        growing_buffer<locator_t, scratch_allocator_t> kept_locators;
        growing_buffer<distance_t, scratch_allocator_t> kept_distances;
        growing_buffer<u32_t, scratch_allocator_t> kept_counts;
        growing_buffer<distance_t, scratch_allocator_t> thresholds;
        growing_buffer<distance_t, scratch_allocator_t> round_distances;
        status_t failure = status_t::success_k;

        explicit scan_share_t(scratch_allocator_t const& allocator) noexcept
            : kept_locators(allocator), kept_distances(allocator), kept_counts(allocator), thresholds(allocator),
              round_distances(allocator) {}

        /** Sizes rankings of @p wanted for @p queries, every one empty, and rounds of @p rows. */
        status_t reserve(std::size_t queries, std::size_t wanted, std::size_t rows) noexcept {
            if (failed(kept_locators.resize_uninitialized(queries * wanted)) ||
                failed(kept_distances.resize_uninitialized(queries * wanted)) || failed(kept_counts.resize(queries)) ||
                failed(thresholds.resize_uninitialized(queries)) ||
                failed(round_distances.resize_uninitialized(queries * rows)))
                return status_t::bad_alloc_k;
            for (distance_t& threshold : thresholds) threshold = unbounded_radius<distance_t>();
            return status_t::success_k;
        }
    };

    /** Keeps @p locator at @p distance, under its query's threshold, in query @p query 's ranking
     *  on @p share, replacing the farthest once @p wanted are held and moving the threshold to the
     *  new farthest. */
    static void keep_ranked_(scan_share_t& share, std::size_t query, std::size_t wanted, locator_t locator,
                             distance_t distance) noexcept {
        locator_t* const locators = share.kept_locators.data() + query * wanted;
        distance_t* const ranked = share.kept_distances.data() + query * wanted;
        u32_t& kept = share.kept_counts[query];
        std::size_t slot = 0;
        if (kept != wanted)
            for (slot = kept++; slot;) {
                std::size_t const parent = (slot - 1) / 2;
                if (!(ranked[parent] < distance)) break;
                locators[slot] = locators[parent], ranked[slot] = ranked[parent];
                slot = parent;
            }
        else
            for (std::size_t child = 1; child < wanted; child = slot * 2 + 1) {
                if (child + 1 < wanted && ranked[child + 1] > ranked[child]) ++child;
                if (!(ranked[child] > distance)) break;
                locators[slot] = locators[child], ranked[slot] = ranked[child];
                slot = child;
            }
        locators[slot] = locator, ranked[slot] = distance;
        if (kept == wanted) share.thresholds[query] = ranked[0];
    }

    using candidates_allocator_t = arena_allocator<scored_node_t, forkunion_allocator_t>;
    using results_array_t = sorted_buffer<scored_node_t, typename scored_node_t::closer_t, candidates_allocator_t>;

    /** What the codec's @c view borrows a landed value row as. */
    using value_view_t = typename codec_t::view_t;

    /** 8-ary: eight scored nodes of the narrowest distances fill one cache line per level scan. */
    using frontier_queue_t = max_heap_dary<8, scored_node_t, typename scored_node_t::further_t, candidates_allocator_t>;
    using visited_set_t = growing_hash_set<node_id_t, std::hash<node_id_t>, scratch_allocator_t>;

    /** A ragged column's row starts in its own elements, then its end, a few bits a row. */
    using directory_t = st::immutable_elias_fano<u64_t, forkunion_allocator<u64_t>, row_kit_type_>;

  public:

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

  private:

#pragma region Walks

    /** One walk of one query over one shard: its candidates, its admitted set, its visited set, and
     *  the flights it waits on. */
    struct walk_t {

        /** Nodes scored so far, the best ascending. */
        results_array_t results;

        /** Nodes its query admitted so far, the best ascending; empty unless the query admits. */
        results_array_t admitted;

        /** Nodes worth expanding next. */
        frontier_queue_t frontier;

        /** Nodes already read for scoring; exact by design. */
        visited_set_t visited;

        /** Landed score flights awaiting the next quad. */
        fu::limited_array<u32_t, quad_lanes_k> staged;

        /** Draws this walk's landmarks. */
        splitmix64_t random_generator;

        /** The shard walked, positional among the call's shards. */
        u32_t shard = 0;

        /** The query slot this walk belongs to. */
        u32_t query_slot = 0;

        /** Where the walk stands. */
        nvme_walk_stage_t stage = nvme_walk_stage_t::finished_k;

        explicit walk_t(scratch_allocator_t allocator) noexcept
            : results(candidates_allocator_t(allocator)), admitted(candidates_allocator_t(allocator)),
              frontier(candidates_allocator_t(allocator)), visited(allocator) {}
    };

    /** The sizes and shared words every thread of one call reads. */
    struct walk_shape_t {

        /** Queries of the batch. */
        std::size_t queries_count = 0;

        /** How many candidates each walk keeps. */
        std::size_t expansion_factor = 0;

        /** The queue sizes every walk starts from, at that breadth over the widest part's rows. */
        nvme_queue_sizes_t queue_sizes {};

        /** Flights one walk ever holds at once: its landmarks or one expansion and every neighbor
         *  it scores, and one expansion and its neighbors more per further hop in flight. */
        std::size_t walk_flights = 0;

        /** Queries a thread keeps in flight. */
        std::size_t queries_per_thread = 0;

        /** Hops each walk keeps in flight: the queue depth its thread's queries leave idle, one for
         *  a full batch. */
        std::size_t hops_per_walk = 1;

        /** Hop slots each walk's span holds: its own @c hops_per_walk, and past the batch's end the
         *  shares of every other query slot of its thread, which then sit idle. */
        std::size_t hop_slots = 1;

        /** Buffers per flight: two when keys land beside values. */
        std::size_t buffers_per_flight = 1;

        /** Bytes every thread's arena carves for its walks, queries and flights. */
        std::size_t scratch_bytes = 0;

        /** Staging buffers every engine holds: @c buffers_per_flight per flight. */
        std::size_t buffers = 0;

        /** Bytes of one staging buffer: the covering read of the widest row any part holds. */
        std::size_t buffer_bytes = 0;

        /** The next query any thread takes. */
        std::size_t* cursor = nullptr;

        /** The first failure any thread met, which stops every thread priming queries. */
        status_t* failure = nullptr;

        /** Nanoseconds per query from its first primed walk to its final write; empty collects
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

    /** The shape of a walk of @p queries_count queries of @p query_type_ over every part in
     *  @p parts, each walk @p expansion_factor wide, on @p threads engines @p queue_depth reads
     *  deep, every query in flight staging @p staged_bytes. */
    template <typename query_type_>
    static walk_shape_t walk_shape_(std::span<medium_t const> media, std::span<part_t const> parts,
                                    std::size_t queue_depth, std::size_t threads, std::size_t queries_count,
                                    std::size_t expansion_factor, std::size_t staged_bytes) noexcept {
        using slice_t = arena_allocator<byte_t>;
        // Parts differ in their widest rows, so the widest of all sizes every walk.
        std::size_t const connectivity = nvme_widest_row(parts);
        walk_shape_t shape;
        shape.queries_count = queries_count;
        shape.expansion_factor = expansion_factor;
        shape.queue_sizes = nvme_queue_sizes(expansion_factor, connectivity);
        shape.buffers_per_flight = query_type_::key_reads_k == nvme_key_reads_t::scored_k ? 2 : 1;
        std::size_t const first_hop = max_of<std::size_t>(round_up_to_multiple<std::size_t>(connectivity, 4),
                                                          connectivity + 1);
        std::size_t const hop_depth = parts.size() * first_hop * shape.buffers_per_flight;
        shape.queries_per_thread = min_of(max_of<std::size_t>(1, queue_depth / hop_depth),
                                          max_of<std::size_t>(1, divide_round_up(queries_count, threads)));
        shape.hops_per_walk = max_of<std::size_t>(1, queue_depth / (shape.queries_per_thread * hop_depth));
        shape.walk_flights = first_hop + (shape.hops_per_walk - 1) * (connectivity + 1);
        // Past the batch's end a walk inherits the flights of the query slots that went idle.
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
                              slice_t::reserved_bytes(queries * sizeof(query_type_)) +
                              queries * slice_t::reserved_bytes(staged_bytes) +
                              slice_t::reserved_bytes(queries * sizeof(nvme_query_slot_t)) +
                              slice_t::reserved_bytes(flights * sizeof(nvme_flight<node_id_t>)) +
                              slice_t::reserved_bytes(flights * sizeof(u32_t)) +
                              slice_t::reserved_bytes(walks * shape.hop_slots * sizeof(u32_t)) +
                              slice_t::reserved_bytes(shape.buffers * sizeof(io_request_t)) +
                              slice_t::reserved_bytes(shape.buffers * sizeof(io_completion_t)) + default_alignment_k;
        shape.buffer_bytes = staging_bytes_(media, parts);
        return shape;
    }

    /** Walks @p queries_count queries of @p query_type_ over every part in @p parts inside a call
     *  of its own, each walk @p expansion_factor wide. */
    template <typename query_type_>
    static expected<search_result_t> walk_queries_(std::span<medium_t const> media, std::span<part_t const> parts,
                                                   executor_t& executor, std::size_t queries_count,
                                                   std::size_t expansion_factor,
                                                   typename query_type_::inputs_t const& inputs,
                                                   std::span<std::size_t const> counts) noexcept {
        walk_shape_t const shape = walk_shape_<query_type_>(media, parts, executor.queue_depth(), executor.size(),
                                                            queries_count, expansion_factor,
                                                            query_type_::staged_bytes(inputs));
        auto [call, call_status] = nvme_begin_call(executor, media, shape.scratch_bytes, shape.buffers,
                                                   shape.buffer_bytes);
        if (failed(call_status)) return {{}, call_status};
        if constexpr (query_type_::admitting_k)
            if (status_t const opened = open_predicate(*inputs.predicate, executor); failed(opened))
                return {{}, opened};
        return query_walker<nvme_parts_catalog, query_type_>::run_walks(media, parts, executor, shape, inputs, counts,
                                                                        {});
    }

    /** Opens the round of a primed @p query on its prepared query, which every walk of it then
     *  reuses. */
    template <typename query_type_>
    static void prime_scoring_(walk_shape_t const&, query_type_& query) noexcept {
        if (succeeded(query.status)) query.metric.reset(query.prepared);
    }

    /** The codec's view of the value landed for the score @p flight of @p walker. */
    template <typename walker_type_>
    static value_view_t landed_value_(walker_type_& walker, u32_t flight) noexcept {
        return codec_t {}.view(bytes_view_t {
            walker.landed_row_(flight, nvme_purpose_t::score_k, flight),
            static_cast<std::size_t>(walker.row_bytes_(walker.flights_[flight], nvme_purpose_t::score_k))});
    }

    /** Hands the staged flights of @p walk to its query's metric as the codec's views of their
     *  landed values, drains them within the stopping candidates' radius, at a refusal and once all
     *  are staged, lets an admitting query test them, releases them and ranks them through
     *  @p walker; a metric refusing a lone value saturates the walk. */
    template <typename walker_type_>
    [[gnu::always_inline]] static void score_staged_(walker_type_& walker, walk_t& walk) noexcept {
        using query_t = typename walker_type_::query_t;
        query_t& query = walker.queries_[walk.query_slot];
        // Full stopping candidates only take what beats their furthest member, so that member's
        // distance is the radius, and every distance an admitting query keeps stays exact.
        results_array_t const& stopping = query.stopping_candidates(walk);
        distance_t const radius = stopping.size() == walker.shape_.expansion_factor ? stopping.top().distance
                                                                                    : unbounded_radius<distance_t>();
        quad<distance_t> distances;
        std::size_t staged = 0, drained = 0;
        auto drain = [&]() noexcept {
            query.metric.drain(std::span<distance_t>(distances).subspan(drained, staged - drained), radius);
            drained = staged;
        };
        for (u32_t const flight : walk.staged) {
            if (query.metric.push(landed_value_(walker, flight)) == metric_staging_t::refused_k) {
                drain();
                if (query.metric.push(landed_value_(walker, flight)) == metric_staging_t::refused_k) break;
            }
            ++staged;
        }
        if (staged != drained) drain();
        walker.telemetry_.computed_distances += staged;
        quad<scored_node_t> scored_batch;
        for (std::size_t lane = 0; lane != staged; ++lane)
            scored_batch[lane] = scored_node_t {walker.flights_[walk.staged[lane]].node, distances[lane]};
        if constexpr (query_t::admitting_k) walker.admit_scored_(walk, std::span(scored_batch).first(staged));
        bool const refused = staged != walk.staged.size();
        for (u32_t const flight : walk.staged) walker.release_flight_(flight);
        walk.staged.clear();
        std::span<scored_node_t> const scored = std::span(scored_batch).first(staged);
        network_sort(scored, typename scored_node_t::closer_t {});
        walker.rank_(walk, scored);
        if (refused) walker.saturate_(walk);
    }

    /** A query without a relation admits every contender. */
    template <typename walker_type_, std::size_t extent_>
    static void ask_relation_(walker_type_&, walk_t const&, std::span<u32_t const, extent_>,
                              std::span<admission_t, extent_> related) noexcept {
        for (admission_t& admission : related) admission = admission_t::admit_k;
    }

    /** Asks the walker's thread's relation about the values landed for @p flights against the
     *  prepared query of the query behind @p walk, read straight from their buffers before they
     *  recycle. */
    template <typename walker_type_, std::size_t extent_>
        requires walker_type_::query_t::related_k
    static void ask_relation_(walker_type_& walker, walk_t const& walk, std::span<u32_t const, extent_> flights,
                              std::span<admission_t, extent_> related) noexcept {
        typename walker_type_::query_t& query = walker.queries_[walk.query_slot];
        std::array<value_view_t, extent_> values;
        for (std::size_t lane = 0; lane != extent_; ++lane) values[lane] = landed_value_(walker, flights[lane]);
        for (admission_t& admission : related) admission = admission_t::admit_k;
        thread_predicate(*query.inputs.relation, walker.thread_)(
            query.prepared, std::span<value_view_t const, extent_>(values), related);
    }

#pragma endregion Walks

#pragma region Queries

    /** The queries of a batched search across the shards: each walk's candidates, or with a
     *  relation or a predicate its admitted set, merge into the query's ranking by locator, and the
     *  ranked locators resolve into keys once every walk finished. Each slot keeps one metric from
     *  the caller's factory and the prepared query of its current batch query. */
    template <typename make_metric_type_, typename query_of_type_, typename predicate_type_, typename relation_type_,
              typename score_type_>
    struct search_query {
        using key_t = key_type_;
        using metric_t =
            typename std::invoke_result_t<make_metric_type_&, scratch_allocator_t const&, std::size_t>::value_t;
        using query_t =
            typename std::invoke_result_t<query_of_type_&, std::size_t, scratch_allocator_t const&>::value_t;
        inline static constexpr bool related_k = !std::is_same_v<std::remove_cv_t<relation_type_>, dummy_predicate_t>;
        inline static constexpr bool filtered_k = !std::is_same_v<std::remove_cv_t<predicate_type_>, dummy_predicate_t>;

        /** A filtered query admits as it scores, so every scored entry's key lands beside its
         *  value. */
        inline static constexpr nvme_key_reads_t key_reads_k = filtered_k ? nvme_key_reads_t::scored_k
                                                                          : nvme_key_reads_t::outputs_k;
        inline static constexpr bool admitting_k = related_k || filtered_k;

        /** Candidates a metric stages per round: one quad of landed values. */
        inline static constexpr std::size_t staging_k = 4;

        /** The batch every query of the call reads and writes. */
        struct inputs_t {
            make_metric_type_* make_metric = nullptr;
            query_of_type_* query_of = nullptr;
            predicate_type_* predicate = nullptr;
            relation_type_* relation = nullptr;
            std::size_t ranked = 0;
            std::size_t wanted = 0;
            std::span<key_t> keys;
            std::span<score_type_> distances;
            std::span<std::size_t> counts;
        };

        inputs_t const& inputs;
        scratch_allocator_t allocator;
        metric_t metric;
        query_t prepared {};
        status_t status = status_t::success_k;
        growing_buffer<locator_t, scratch_allocator_t> locators;
        growing_buffer<distance_t, scratch_allocator_t> ranked_distances;
        growing_buffer<key_t, scratch_allocator_t> resolved;
        std::size_t query_index = 0;
        std::size_t found = 0;

        search_query(inputs_t const& inputs, scratch_allocator_t allocator) noexcept
            : search_query(inputs, allocator, (*inputs.make_metric)(allocator, staging_k)) {}

        /** Bytes one slot stages beyond the query itself: its ranking, distances and resolved
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
        void prime_query(std::size_t new_query_index) noexcept {
            query_index = new_query_index;
            found = 0;
            inputs.counts[query_index] = 0;
            auto [made, made_status] = (*inputs.query_of)(query_index, allocator);
            prepared = std::move(made);
            status = made_status;
        }
        results_array_t const& stopping_candidates(walk_t const& walk) const noexcept { return walk.results; }

        /** An admitting walk stops on what its query admitted, so the admitted set fills before it
         *  stops. */
        results_array_t const& stopping_candidates(walk_t const& walk) const noexcept
            requires admitting_k
        {
            return walk.admitted;
        }
        void finish_walk(walk_t const& walk) noexcept {
            found = merge_search_results(to_span(stopping_candidates(walk)),
                                         nvme_shard_locators<node_id_t> {walk.shard}, to_writable_span(locators),
                                         to_writable_span(ranked_distances), found);
        }
        std::size_t output_count() const noexcept { return found; }
        locator_t output_locator(std::size_t output) const noexcept { return locators[output]; }
        void resolve_output_key(std::size_t output, key_t const& key) noexcept { resolved[output] = key; }
        status_t finish_query() noexcept {
            std::span<key_t> const keys = inputs.keys.subspan(query_index * inputs.wanted, inputs.wanted);
            std::span<score_type_> const distances = inputs.distances.subspan(query_index * inputs.wanted,
                                                                              inputs.wanted);
            std::size_t const written = min_of(found, inputs.wanted);
            for (std::size_t match = 0; match != written; ++match) {
                keys[match] = resolved[match];
                distances[match] = static_cast<score_type_>(ranked_distances[match]);
            }
            inputs.counts[query_index] = written;
            return status;
        }

      private:
        search_query(inputs_t const& inputs, scratch_allocator_t allocator, expected<metric_t>&& made) noexcept
            : inputs(inputs), allocator(allocator), metric(std::move(made.value)), status(made.status),
              locators(allocator), ranked_distances(allocator), resolved(allocator) {}
    };

#pragma endregion Queries

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
     *  staging @p buffer, blocking, and answers where they start in it; a short read answers
     *  @c io_k. */
    expected<byte_t const*> read_(executor_t& executor, std::size_t thread, u32_t part, u64_t begin, u64_t bytes,
                                  io_buffer_id_t buffer = 0) const noexcept {
        engine_t& engine = executor.engine(thread);
        medium_t const& medium = media_[part];
        u64_t const absolute = byte_base_of(medium) + begin;
        io_request_t const request = nvme_covering_read(absolute, bytes, block_bytes_of(medium), part, buffer, 0);
        if (status_t const enqueued = engine.enqueue_read(request); failed(enqueued)) return {nullptr, enqueued};
        if (status_t const submitted = engine.submit(io_wait_t::blocking_k); failed(submitted))
            return {nullptr, submitted};
        std::array<io_completion_t, 1> completions;
        u64_t const skipped = absolute - request.offset;
        if (!engine.poll_completions(completions) || completions[0].result < 0 ||
            static_cast<u64_t>(completions[0].result) < skipped + bytes)
            return {nullptr, status_t::io_k};
        return {engine.buffer(buffer) + skipped, status_t::success_k};
    }

    /** One round of rows a scan scores together: rows @c first_row up to @c end_row of a run of
     *  part @c shard from node @c first, whose value rows landed at @c values from the part's byte
     *  @c values_begin, and whose key rows, when read, landed at @c key_rows. */
    struct scan_round_t {
        part_t const& part;
        u32_t shard = 0;
        node_id_t first = 0;
        u64_t values_begin = 0;
        byte_t const* values = nullptr;
        byte_t const* key_rows = nullptr;
        std::size_t first_row = 0;
        std::size_t end_row = 0;

        /** The codec's view of row @p row of the run, as @ref score_queries reads it. */
        value_view_t operator()(std::size_t row) const noexcept {
            node_id_t const node = static_cast<node_id_t>(first + row);
            u64_t const offset = row_offset_(part, nvme_purpose_t::score_k, node) - values_begin;
            return codec_t {}.view(bytes_view_t {
                values + offset, static_cast<std::size_t>(row_length_(part, nvme_purpose_t::score_k, node))});
        }

        /** The key read beside row @p row of the run. */
        key_t key_of(std::size_t row) const noexcept {
            return safegraphs_widened_key<key_t>(key_rows + row * part.key_row_bytes, part.key_dtype);
        }

        /** Where row @p row of the run lives among the call's shards. */
        locator_t locator_of(std::size_t row) const noexcept { return {shard, static_cast<node_id_t>(first + row)}; }
    };

    /**
     *  @brief Reads every entry of every part once, in runs taken by the threads of @p executor,
     *      and scores each staged round of a run against all @p count queries into its thread's
     *      share, handed to @p on_round as <tt>on_round(thread, share, round, queries)</tt>.
     *
     *  Each share ranks @p wanted entries a query under thresholds starting at @p reaches, or
     *  unbounded when it is empty, and reserves the thread's @p predicate for a run's entries; key
     *  rows land beside value rows only for @c scored_k. @p on_shares reads every share as
     *  <tt>on_shares(shares) noexcept → status_t</tt> while the call still holds the media, before
     *  they release. A failed read stops the call, which answers @c io_k.
     */
    template <nvme_key_reads_t key_reads_, typename predicate_type_, typename make_metric_type_,
              typename query_of_type_, typename on_round_type_, typename on_shares_type_>
    status_t scan_runs_(std::size_t count, std::size_t wanted, std::span<distance_t const> reaches,
                        predicate_type_& predicate, make_metric_type_& make_metric, query_of_type_& query_of,
                        executor_t& executor, on_round_type_&& on_round, on_shares_type_&& on_shares) const noexcept {
        using metric_t =
            typename std::invoke_result_t<make_metric_type_&, scratch_allocator_t const&, std::size_t>::value_t;
        using query_t =
            typename std::invoke_result_t<query_of_type_&, std::size_t, scratch_allocator_t const&>::value_t;
        std::size_t const entries = size(), threads = executor.size();
        // A chunk holds as many entries as a walk's default candidates.
        std::size_t const chunk_entries = default_expansion_factor(connectivity());
        auto [call, call_status] = nvme_begin_call(executor, to_span(media_), 0, 2, scan_buffer_bytes_());
        if (failed(call_status)) return call_status;
        if (status_t const opened = open_predicate(predicate, executor); failed(opened)) return opened;

        // The queries build once and every thread reads them; a metric and a share belong to each
        // thread, on its own arena.
        arena_t& home = executor.arena(0);
        query_t* const queries = arena_allocator<query_t, forkunion_allocator_t> {home}.allocate(count);
        metric_t* const metrics = arena_allocator<metric_t, forkunion_allocator_t> {home}.allocate(threads);
        scan_share_t* const shares = arena_allocator<scan_share_t, forkunion_allocator_t> {home}.allocate(threads);
        if (!queries || !metrics || !shares) return status_t::bad_alloc_k;
        std::span<query_t const> const batch {queries, count};
        std::size_t built_queries = 0, built_metrics = 0, built_shares = 0;
        auto release = [&]() noexcept {
            std::destroy_n(queries, built_queries);
            std::destroy_n(metrics, built_metrics);
            std::destroy_n(shares, built_shares);
        };
        status_t built = status_t::success_k;
        for (; succeeded(built) && built_queries != count; ++built_queries) {
            auto [query, query_status] = query_of(built_queries, scratch_allocator_t {home});
            new (queries + built_queries) query_t(std::move(query));
            built = query_status;
        }
        for (; succeeded(built) && built_metrics != threads; ++built_metrics) {
            auto [metric, metric_status] = make_metric(scratch_allocator_t {executor.arena(built_metrics)},
                                                       chunk_entries);
            new (metrics + built_metrics) metric_t(std::move(metric));
            built = metric_status;
        }
        for (; succeeded(built) && built_shares != threads; ++built_shares) {
            scan_share_t& share = *new (shares + built_shares)
                                      scan_share_t {scratch_allocator_t {executor.arena(built_shares)}};
            built = share.reserve(count, wanted, chunk_entries);
            if (succeeded(built) && !reaches.empty())
                std::memcpy(share.thresholds.data(), reaches.data(), reaches.size_bytes());
            if (succeeded(built)) built = reserve_predicate(thread_predicate(predicate, built_shares), chunk_entries);
            open_queries(metrics[built_shares], batch);
        }
        if (failed(built)) {
            release();
            return built;
        }

        // Reads one run of entries that fits a staging buffer, its value rows in buffer zero and
        // its key rows in buffer one, and scores it a staged round at a time.
        auto score_run = [&](std::size_t thread, u32_t shard, node_id_t first, node_id_t end) noexcept {
            part_t const& part = parts_[shard];
            scan_share_t& share = shares[thread];
            std::size_t const staged = end - first;
            scan_round_t round {part, shard, first, row_offset_(part, nvme_purpose_t::score_k, first)};
            u64_t const values_end = row_offset_(part, nvme_purpose_t::score_k, end - 1) +
                                     row_length_(part, nvme_purpose_t::score_k, end - 1);
            auto const [values, values_status] = read_(executor, thread, shard, round.values_begin,
                                                       values_end - round.values_begin, 0);
            if (failed(values_status)) return values_status;
            round.values = values;
            if constexpr (key_reads_ == nvme_key_reads_t::scored_k) {
                auto const [key_rows, keys_status] = read_(executor, thread, shard,
                                                           row_offset_(part, nvme_purpose_t::key_k, first),
                                                           staged * part.key_row_bytes, 1);
                if (failed(keys_status)) return keys_status;
                round.key_rows = key_rows;
            }
            for (; round.first_row != staged; round.first_row = round.end_row) {
                round.end_row = score_queries(metrics[thread], batch, round, round.first_row, staged,
                                              to_span(share.thresholds), to_writable_span(share.round_distances));
                if (round.end_row == round.first_row) return status_t::bad_alloc_k;
                on_round(thread, share, round, batch);
            }
            return status_t::success_k;
        };

        // A chunk splits into runs at part ends and wherever its value bytes would outgrow a scan
        // read, every run holding at least one entry, which a staging buffer always fits.
        auto score_chunk = [&](std::size_t chunk, std::size_t thread, std::size_t) noexcept {
            scan_share_t& share = shares[thread];
            if (failed(share.failure)) return;
            std::size_t global = chunk * chunk_entries, shard = 0;
            std::size_t const chunk_end = min_of(global + chunk_entries, entries);
            std::size_t node = global;
            while (node >= parts_[shard].nodes) node -= parts_[shard].nodes, ++shard;
            while (global != chunk_end) {
                part_t const& part = parts_[shard];
                std::size_t const part_end = min_of<std::size_t>(part.nodes, node + chunk_end - global);
                u64_t const begin = row_offset_(part, nvme_purpose_t::score_k, static_cast<node_id_t>(node));
                std::size_t end = node + 1;
                for (; end != part_end; ++end) {
                    node_id_t const next = static_cast<node_id_t>(end);
                    u64_t const run_end = row_offset_(part, nvme_purpose_t::score_k, next) +
                                          row_length_(part, nvme_purpose_t::score_k, next);
                    if (run_end - begin > scan_chunk_bytes_k) break;
                }
                share.failure = score_run(thread, static_cast<u32_t>(shard), static_cast<node_id_t>(node),
                                          static_cast<node_id_t>(end));
                if (failed(share.failure)) return;
                global += end - node, node = end;
                if (node == part.nodes) node = 0, ++shard;
            }
        };
        status_t scanned = executor.for_tasks(divide_round_up(entries, chunk_entries), score_chunk);
        for (std::size_t thread = 0; succeeded(scanned) && thread != threads; ++thread)
            scanned = shares[thread].failure;
        if (succeeded(scanned)) scanned = on_shares(std::span<scan_share_t const>(shares, threads));
        release();
        return scanned;
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
     *  @brief Scans every part's key column in runs, matching every key against an exact table of
     *      @p keys, and calls @p on_hit with the batch position and the entry's locator for every
     *      match, repeats included.
     *
     *  The table holds each distinct key's first batch position, the positions sharing it chained
     *  behind, in thread zero's arena; runs are read one at a time per thread, inside a call
     *  @c begin_scans_ began.
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
            if (failed(atomic_ref<status_t>(failure).load(std::memory_order_relaxed))) return;
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
                atomic_ref<status_t>(failure).compare_exchange_strong(unset, read_status, std::memory_order_relaxed,
                                                                      std::memory_order_relaxed);
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

  public:
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

    /** Builds @p directory over the @p rows + 1 prefix sums of the ragged @p column of the part
     *  @p medium holds, streamed through @p chunk, and answers the widest row they delimit in
     *  @p widest. No sum may pass the elements the column holds. */
    template <typename chunk_allocator_type_>
    static status_t build_row_directory_(file_medium_t const& medium, persisted_column_t const& column,
                                         std::size_t rows, growing_buffer<u64_t, chunk_allocator_type_>& chunk,
                                         forkunion_allocator_t allocator, directory_t& directory,
                                         u64_t& widest) noexcept {
        prefix_sums_t const& sums = *column.offsets;
        std::size_t const sum_bytes = dtype_bytes(sums.dtype);
        std::size_t const element_bytes = dtype_bytes(column.dtype);
        if ((sums.dtype != dtype_t::u64_k && sums.dtype != dtype_t::u32_k) || !element_bytes)
            return status_t::config_mismatch_k;
        if (sums.bytes.size() != (rows + 1) * sum_bytes) return status_t::malformed_k;
        u64_t const sums_begin = sums.bytes.first;
        u64_t const extent = column.data.size() / element_bytes;
        // Reads `count` sums from `first` on into the chunk's front, widening narrower ones back to
        // front so no sum is overwritten before it widened.
        auto read_sums = [&](std::size_t first, std::size_t count) noexcept {
            bytes_span_t const target = bytes_of(chunk.data(), count).first(count * sum_bytes);
            if (status_t const read = medium.read_at(sums_begin + first * sum_bytes, target); failed(read)) return read;
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
        auto made = directory_t::make(rows + 1, universe, fill, forkunion_allocator<u64_t>(allocator));
        if (!made) return status_of_(made.status());
        directory = std::move(*made);
        return status_t::success_k;
    }

    /** Opens the part @p file holds as @p columns locate its rows: the directories of its ragged
     *  columns, built through blocking reads of @p file and streamed through @p chunk, then the
     *  file as the engines' medium. */
    template <typename chunk_allocator_type_>
    static status_t open_part_(file_medium_t&& file, persisted_columns_t const& columns,
                               growing_buffer<u64_t, chunk_allocator_type_>& chunk, forkunion_allocator_t allocator,
                               medium_t& medium, part_t& part) noexcept {
        if constexpr (!std::is_constructible_v<medium_t, file_medium_t&&>) return status_t::config_mismatch_k;
        else {
            persisted_column_t const& keys = columns.keys;
            persisted_column_t const& values = columns.values;
            persisted_column_t const& neighbors = columns.neighbors;
            std::size_t const key_bytes = keys.row_bytes;
            std::size_t const value_element_bytes = dtype_bytes(values.dtype);
            if (neighbors.dtype != safegraphs_node_id_dtype<node_id_t>() || !key_bytes || keys.offsets ||
                keys.row_width != 1 || !value_element_bytes || columns.nodes >= (std::numeric_limits<node_id_t>::max)())
                return status_t::config_mismatch_k;
            part.nodes = static_cast<node_id_t>(columns.nodes);
            part.key_row_bytes = static_cast<u32_t>(key_bytes);
            part.key_dtype = keys.dtype;
            part.value_element_bytes = static_cast<u32_t>(value_element_bytes);
            part.keys_begin = keys.data.first;
            part.values_begin = values.data.first;
            part.neighbors_begin = neighbors.data.first;
            if (!values.offsets) part.value_row_bytes = values.row_bytes;
            else {
                u64_t widest = 0;
                if (status_t const built = build_row_directory_(file, values, columns.nodes, chunk, allocator,
                                                                part.value_starts, widest);
                    failed(built))
                    return built;
                part.value_row_bytes = widest * value_element_bytes;
            }
            if (!neighbors.offsets) part.connectivity = static_cast<u32_t>(neighbors.row_width);
            else {
                u64_t widest = 0;
                if (status_t const built = build_row_directory_(file, neighbors, columns.nodes, chunk, allocator,
                                                                part.neighbor_starts, widest);
                    failed(built))
                    return built;
                // A part whose nodes hold no neighbors still seeds its walks from one landmark at
                // least.
                part.connectivity = static_cast<u32_t>(max_of<u64_t>(widest, 1));
            }
            medium = medium_t {std::move(file)};
            return status_t::success_k;
        }
    }

#pragma endregion Opening

    /** A catalog walking the parts @p parts locates on the media @p media, in part order. */
    nvme_parts_catalog(std::span<medium_t const> media, std::span<part_t const> parts) noexcept
        : media_(media), parts_(parts) {}

  private:

    /** The media every verb walks, in part order. */
    std::span<medium_t const> media_;

    /** Where every part keeps its columns and its row directories, parallel to @c media_. */
    std::span<part_t const> parts_;
};

/**
 *  @brief Similarity Search over one part a basic graph was saved as, on block storage, read-only.
 *
 *  The class is a frozen directory of one part: its medium, its columns' offsets and the
 *  directories of its ragged columns, and nothing per node beyond those few bits a row. Every call
 *  borrows an executor, attaches the medium to its engines for its own duration, and carves its
 *  walks from the executor's arenas, all through a @c nvme_parts_catalog over its one part.
 *
 *  It stands alone as the RAM index does: any key type and either node-ID width the part was saved
 *  with serve, and only the compiled library narrows the choice, to the 32-bit IDs its plurals
 *  write.
 *
 *  @tparam codec_type_ The value codec the part was saved with, whose @c view borrows a landed row.
 *  @tparam key_type_ The caller-visible identity, read from the part's key column.
 *  @tparam node_id_type_ Identity within the part, as wide as the neighbor IDs it was saved with;
 *      @c u32_t or @c u64_t.
 *  @tparam distance_type_ The distance the metrics drain and the walks rank by.
 *  @tparam engine_type_ The I/O engine every executor thread drives.
 */
template <typename codec_type_, typename key_type_ = uuid_t, typename node_id_type_ = u32_t,
          typename distance_type_ = f64_t, typename engine_type_ = default_io_engine_t,
          typename row_kit_type_ = st::serial_row_kit_t,
          template <typename> class atomic_ref_template_ = fu::standard_atomic_ref,
          typename wait_type_ = fu::standard_yield_t>
    requires std::same_as<node_id_type_, u32_t> || std::same_as<node_id_type_, u64_t>
class nvme_index {
  public:

    /** The walks every verb runs over this index's one part. */
    using catalog_t = nvme_parts_catalog<codec_type_, key_type_, node_id_type_, distance_type_, engine_type_,
                                         row_kit_type_, atomic_ref_template_, wait_type_>;

    using codec_t = typename catalog_t::codec_t;
    using key_t = typename catalog_t::key_t;
    using distance_t = typename catalog_t::distance_t;
    using engine_t = typename catalog_t::engine_t;
    using executor_t = typename catalog_t::executor_t;
    using medium_t = typename catalog_t::medium_t;
    using arena_t = typename catalog_t::arena_t;
    using node_id_t = typename catalog_t::node_id_t;
    using scored_node_t = typename catalog_t::scored_node_t;
    using scan_payload_t = typename catalog_t::scan_payload_t;
    using walk_payload_t = typename catalog_t::walk_payload_t;
    using search_config_t = typename catalog_t::search_config_t;
    using search_result_t = typename catalog_t::search_result_t;

    nvme_index() = delete;
    explicit nvme_index(forkunion_allocator_t allocator) noexcept : allocator_(allocator) {}
    nvme_index(nvme_index&&) noexcept = default;
    nvme_index& operator=(nvme_index&&) noexcept = default;
    nvme_index(nvme_index const&) = delete;
    nvme_index& operator=(nvme_index const&) = delete;

    /**
     *  @brief Opens the part file at @p path, a file or a device, as @p columns locate its rows,
     *      with the directories of its ragged columns built through blocking reads. A failed
     *      opening leaves the index empty.
     *
     *  @param[in] path The part file, or the device holding it.
     *  @param[in] columns The part's columns, as any producer of its format fills them, such as
     *      @c safegraphs_persisted_part over a SafeGraphs monolith's header.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k @p path names no location, its base or namespace is unreadable, or
     *    the base is off the medium's block.
     *  - @c config_mismatch_k The engines read no files, or @p columns lay out keys, neighbors or
     *    node counts this instantiation cannot walk.
     *  - @c malformed_k A row directory's prefix sums do not cover the part's rows.
     *  - @c overflow_k A row directory read spans more than one read can move.
     */
    status_t open(std::string_view path, persisted_columns_t const& columns) noexcept {
        medium_ = medium_t {};
        part_ = part_t {};
        auto [file, open_status] = file_medium_t::open(path);
        if (failed(open_status)) return open_status;
        growing_buffer<u64_t, forkunion_allocator_t> chunk {allocator_};
        status_t opened = chunk.resize_uninitialized(catalog_t::scan_chunk_bytes_k / sizeof(u64_t));
        if (succeeded(opened))
            opened = catalog_t::open_part_(std::move(file), columns, chunk, allocator_, medium_, part_);
        if (failed(opened)) medium_ = medium_t {}, part_ = part_t {};
        return opened;
    }

    /** Parts the index walks: one once opened, none before. */
    std::size_t shards_count() const noexcept { return catalog_().parts_count(); }

    /** Entries the part holds. */
    std::size_t size() const noexcept { return catalog_().size(); }

    /** Entries the index hosts, which are exactly those its part holds, since it never grows. */
    std::size_t capacity() const noexcept { return size(); }

    /** Value elements the part holds, as @ref nvme_parts_catalog::value_elements counts them. */
    std::size_t value_elements() const noexcept { return catalog_().value_elements(); }

    /** Neighbor IDs of the widest neighbors row. */
    std::size_t connectivity() const noexcept { return catalog_().connectivity(); }

    /** Bytes the index occupies: the directory and the row directories, a few bits a row. */
    std::size_t memory_usage() const noexcept {
        return sizeof(nvme_index) + part_.value_starts.size_bytes() + part_.neighbor_starts.size_bytes();
    }

    /** The walk breadth @p config asks for, or twice the widest neighbors row when it asks for
     *  none. */
    std::size_t non_zero_expansion_factor(search_config_t const& config) const noexcept {
        return catalog_().non_zero_expansion_factor(config);
    }

    /**
     *  @ref nvme_parts_catalog::search over the part.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is opened, or the executor holds no threads or pool workers.
     *  - @c config_mismatch_k The counts are not one per query, the keys and distances differ in
     *    length or do not split into one run per query, or a medium the engines cannot read.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    template <typename... arguments_type_>
    expected<search_result_t> search(arguments_type_&&... arguments) noexcept {
        return catalog_().search(std::forward<arguments_type_>(arguments)...);
    }

    /**
     *  @ref nvme_parts_catalog::scan over the part.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is opened, or the executor holds no threads or pool workers.
     *  - @c config_mismatch_k The counts are not one per query, the keys and distances differ in
     *    length or do not split into one run per query, or a medium the engines cannot read.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    template <typename... arguments_type_>
    expected<search_result_t> scan(arguments_type_&&... arguments) noexcept {
        return catalog_().scan(std::forward<arguments_type_>(arguments)...);
    }

    /**
     *  @ref nvme_parts_catalog::scan_within over the part.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is opened, or the executor holds no threads or pool workers.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     *  - @c config_mismatch_k A medium the engines cannot read.
     */
    template <typename... arguments_type_>
    expected<search_result_t> scan_within(arguments_type_&&... arguments) noexcept {
        return catalog_().scan_within(std::forward<arguments_type_>(arguments)...);
    }

    /**
     *  @ref nvme_parts_catalog::count over the part.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p entries and @p keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c uninitialized_k Nothing is opened, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) const noexcept {
        return catalog_().count(keys, entries, executor);
    }

    /**
     *  @ref nvme_parts_catalog::contains over the part.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The entries and the keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c uninitialized_k Nothing is opened, or the executor holds no threads or pool workers.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    template <typename... arguments_type_>
    status_t contains(arguments_type_&&... arguments) const noexcept {
        return catalog_().contains(std::forward<arguments_type_>(arguments)...);
    }

    /**
     *  @ref nvme_parts_catalog::get over the part.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The entries and the keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c uninitialized_k Nothing is opened, or the executor holds no threads or pool workers.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    template <typename... arguments_type_>
    status_t get(arguments_type_&&... arguments) const noexcept {
        return catalog_().get(std::forward<arguments_type_>(arguments)...);
    }

    /**
     *  @ref nvme_parts_catalog::read_entries over the part.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is opened, or the executor holds no threads or pool workers.
     *  - @c config_mismatch_k The entries run past @ref size, or a medium the engines cannot read.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    template <typename... arguments_type_>
    status_t read_entries(arguments_type_&&... arguments) const noexcept {
        return catalog_().read_entries(std::forward<arguments_type_>(arguments)...);
    }

  private:
    using part_t = typename catalog_t::part_t;

    /** The catalog over this index's part and medium: one once opened, none before. */
    catalog_t catalog_() const noexcept {
        std::size_t const opened = part_.connectivity ? 1 : 0;
        return catalog_t {std::span<medium_t const>(&medium_, opened), std::span<part_t const>(&part_, opened)};
    }

    /** The part's file, open for the engines. */
    medium_t medium_ {};

    /** Where the part keeps its columns and its row directories; a zero connectivity until
     *  opened. */
    part_t part_ {};

    /** The host allocator behind the row directories. */
    forkunion_allocator_t allocator_;
};

} // namespace unum::usearch::basic
