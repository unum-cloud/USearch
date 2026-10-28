/**
 *  @file include/usearch/utilities/vram_shards.hpp
 *  @author Ash Vardanian
 *  @date September 2, 2026
 *  @brief The fan-out an NVIDIA or AMD tower drives one batched read through: the executor's claim,
 *      one context per shard, one snapshot of every watermark, every shard queued before any is
 *      awaited, and each shard's matches merged as it lands.
 *
 *  A tower builds one per call over its shard set and the caller's executor. Everything it calls
 *  on the device goes through @p vendor_type_, so every vendor's tower shares it as a component
 *  while keeping its own drivers.
 */
#pragma once

#include <span>    // `std::span` - the caller's rankings & the shards' landings
#include <utility> // `std::move`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the batch's contexts & landings
#include "usearch/utilities/ram_device.hpp"  // `forkunion_allocator` - the landings' placement
#include "usearch/utilities/ram_shards.hpp"  // `host_keys` - a shard's host key column
#include "usearch/utilities/ram_sorts.hpp"   // `rankings_refusal` - the batch's output layout

namespace unum::usearch::utilities {

/**
 *  @brief One batched read over every shard of a device tower: the executor's claim, a context per
 *      shard sized for the widest window, and one snapshot of the shards' visible rows, so every
 *      window reads the same rows.
 *
 *  @tparam shards_type_ The tower's @c shard_set.
 *  @tparam vendor_type_ The runtime the shards' devices answer to.
 *  @tparam landed_score_type_ The scores the shards' searches land on the host, the shard's own
 *      distance unless a modality's search lands another.
 */
template <typename shards_type_, typename vendor_type_,
          typename landed_score_type_ = typename shards_type_::index_t::distance_t>
struct vram_shard_fanout {
    using shards_t = shards_type_;
    using vendor_t = vendor_type_;
    using landed_score_t = landed_score_type_;
    using index_t = typename shards_t::index_t;
    using key_t = typename shards_t::key_t;
    using executor_t = typename vendor_t::executor_t;
    using traversal_context_t = typename index_t::traversal_context_t;
    using traversals_t = growing_buffer<traversal_context_t, forkunion_allocator_t>;
    using search_config_t = typename index_t::search_config_t;
    using search_result_t = typename index_t::search_result_t;

    /** A fan-out of @p shards over @p executor, claimed and sized by @ref begin. */
    vram_shard_fanout(shards_t const& shards, executor_t& executor) noexcept
        : shards_(shards), executor_(executor), traversals_(shards.allocator()), visible_(shards.allocator()),
          pending_(shards.allocator()), landed_ids_(shards.allocator()), landed_scores_(shards.allocator()),
          landed_counts_(shards.allocator()), frame_(shards.allocator()) {}

#pragma region Pairing

    /**
     *  Whether @p executor drives a stream on every shard's device - shards sharing a device share
     *  its stream, and queue on it in order.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The tower has no shards.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device.
     */
    static status_t pairs_with(shards_t const& shards, executor_t const& executor) noexcept {
        if (shards.empty()) return status_t::uninitialized_k;
        for (std::size_t shard = 0; shard != shards.size(); ++shard)
            if (stream_of(shards, executor, shard) == executor.size()) return status_t::config_mismatch_k;
        return status_t::success_k;
    }

    /** The position of the stream on shard @p shard 's device in @p executor, or its size when it
     *  drives none there. */
    static std::size_t stream_of(shards_t const& shards, executor_t const& executor, std::size_t shard) noexcept {
        int const device = shards[shard].index->device_specs().ordinal;
        std::size_t stream = 0;
        while (stream != executor.size() && executor.device(stream) != device) ++stream;
        return stream;
    }

    /** Makes shard @p shard 's device current, since a launch fails on a stream of any other. */
    static status_t enter(shards_t const& shards, std::size_t shard) noexcept {
        return vendor_t::select(shards[shard].index->api(), shards[shard].index->device_specs().ordinal);
    }

    /**
     *  Builds one traversal context per shard over its device's stream and run, so a fan-out never
     *  allocates and never shares scratch; @p usage, the arguments of a shard's
     *  @c traversal_memory_usage, sizes them for the widest shard, and every stream holds a context
     *  for each shard it serves.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The @p executor has no streams.
     */
    template <typename... usage_types_>
    static status_t reserve_traversals(shards_t const& shards, traversals_t& traversals, executor_t& executor,
                                       usage_types_ const&... usage) noexcept {
        // Counted once, so a shard generation opening meanwhile gets no context and no scratch
        // here.
        std::size_t const count = shards.size();
        std::size_t bytes_per_shard = 0, shards_per_stream = 0;
        for (std::size_t shard = 0; shard != count; ++shard)
            bytes_per_shard = max_of(bytes_per_shard, shards[shard].index->traversal_memory_usage(usage...));
        for (std::size_t stream = 0; stream != executor.size(); ++stream) {
            std::size_t served = 0;
            for (std::size_t shard = 0; shard != count; ++shard) served += stream_of(shards, executor, shard) == stream;
            shards_per_stream = max_of(shards_per_stream, served);
        }
        if (status_t const sized = executor.reserve(bytes_per_shard * shards_per_stream); failed(sized)) return sized;
        if (failed(traversals.reserve_exactly(count))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != count; ++shard) {
            typename executor_t::stream_t& stream = executor.stream(stream_of(shards, executor, shard));
            traversals.push_back_reserved(
                traversal_context_t {typename traversal_context_t::allocator_t {stream.arena, stream}});
            if (!traversals.back()) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /**
     *  Why a batch of @p rows cannot rank into @p keys, @p scores and @p counts on @p executor: the
     *  checks every window repeats, run once over the whole batch before any window slices it.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The tower has no shards.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, @p counts is
     *    not @p rows long, or the row counts or widths of @p keys and @p scores disagree.
     *  - @c unsupported_layout_k A row of @p keys or @p scores is not contiguous.
     */
    template <typename score_type_>
    static status_t refusal(shards_t const& shards, executor_t const& executor, std::size_t rows,
                            nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> scores,
                            std::span<std::size_t const> counts) noexcept {
        if (status_t const paired = pairs_with(shards, executor); failed(paired)) return paired;
        if (counts.size() != rows) return status_t::config_mismatch_k;
        return rankings_refusal(rows, keys, scores);
    }

#pragma endregion Pairing

#pragma region Batches

    /** The executor every window queues on, held by the fan-out since @ref begin. */
    executor_t& executor() noexcept { return executor_; }

    /** Shards the snapshot counted, which every window walks; a shard generation opened since takes
     *  no part. */
    std::size_t shards() const noexcept { return visible_.size(); }

    /**
     *  Claims the executor, snapshots the shards' visible nodes and reserves every shard's
     *  traversals for @p usage, the arguments of a shard's @c traversal_memory_usage, sizing again
     *  for a shard generation opened meanwhile.
     *
     *  @par Statuses:
     *  - @c contended_k Another call holds the executor.
     *  - @c uninitialized_k The executor has no streams.
     */
    template <typename... usage_types_>
    status_t begin(usage_types_ const&... usage) noexcept {
        // The executor serializes, never the index: a second caller with its own executor searches
        // concurrently.
        auto [claim, claim_status] = executor_.claim();
        if (failed(claim_status)) return claim_status;
        claim_ = std::move(claim);
        while (true) {
            traversals_.clear();
            if (status_t const readied = reserve_traversals(shards_, traversals_, executor_, usage...); failed(readied))
                return readied;
            std::size_t const shards = traversals_.size();
            if (failed(visible_.resize(shards)) || failed(pending_.reserve_exactly(shards)))
                return status_t::bad_alloc_k;
            status_t const taken = shards_.snapshot(to_writable_span(visible_)).status;
            if (taken != status_t::shape_mismatch_k) return taken;
        }
    }

    /**
     *  @brief Queues @p submit under the snapshot on every shard before awaiting any, and hands
     *      each shard and its context to @p merge as its stream lands.
     *
     *  Whichever shard lands first merges first, so one merge overlaps the other walks. The
     *  telemetry and the rows' @p counts sum, and the first refusal is the one kept. @p submit is
     *  called as <tt>submit(shard, traversal, visible) → status_t</tt>, and @p merge as
     *  <tt>merge(shard, traversal)</tt>.
     */
    template <typename submit_type_, typename merge_type_>
    expected<search_result_t> for_each_shard(submit_type_&& submit, merge_type_&& merge,
                                             std::span<std::size_t const> counts) noexcept {
        std::size_t const shards = this->shards();

        // Queue every shard before awaiting any, so the batch costs the slowest device rather than
        // the sum. From here to the last landing there must be no return: the caller's rows and
        // the landing are DMA targets on every queued stream - the refuser's included, since its
        // upload may already be reading the caller's rows.
        status_t first_refusal = status_t::success_k;
        std::size_t queued = 0;
        for (; queued != shards; ++queued) {
            pending_.push_back_reserved(queued);
            // A launch fails on a stream of any device but the current one; a refuser is awaited
            // too.
            status_t submitted = succeeded(enter(shards_, queued))
                                     ? submit(queued, traversals_[queued], visible_[queued])
                                     : status_t::device_error_k;
            if (status_t const recorded = traversals_[queued].record(); succeeded(submitted)) submitted = recorded;
            if (failed(submitted)) {
                first_refusal = submitted;
                break;
            }
        }

        search_result_t merged;
        while (!pending_.empty())
            for (std::size_t position = 0; position < pending_.size();) {
                std::size_t const shard = pending_[position];
                status_t const landed = traversals_[shard].landed();
                if (landed == status_t::deferred_k) {
                    ++position;
                    continue;
                }
                pending_[position] = pending_.back();
                pending_.pop_back();
                if (failed(landed)) {
                    if (succeeded(first_refusal)) first_refusal = landed;
                    continue; // An unlanded landing is unreadable - this shard merges nothing.
                }
                // A landed context's sink holds this call's spend on its shard.
                vram_telemetry_block_t const spent = traversals_[shard].spent();
                merged.visited_members += spent.hops;
                merged.computed_distances += spent.scored;
                merged.restarts += spent.restarts;
                merged.tail_compactions += spent.tail_compactions;
                merged.tail_dropped += spent.tail_dropped;
                merged.visited_saturations += spent.visited_saturations;
                if (shard != queued) merge(shard, traversals_[shard]); // The refuser landed nothing to merge.
            }
        for (std::size_t const count : counts) merged.count += count;
        return {merged, first_refusal};
    }

    /** One landed match of a shard's row, scored in the type its search lands. */
    struct landed_match_t {
        node_id_t node_id;
        landed_score_t distance;
    };

    /**
     *  @brief Runs @p search_shard on every live shard's own stream, each landing up to @p wanted
     *      node IDs and @c landed_score_t scores for each of @p queries_count queries, then
     *      merges them by score into @p keys, @p scores and @p counts, resolving node IDs through
     *      each shard's host keys.
     *
     *  @p search_shard is called as <tt>search_shard(shard, node_ids, scores, counts, traversal,
     *  visible) → status_t</tt> with that shard's own block of the landing: a row ends at its
     *  count, which starts at @p wanted for a search that reports none, or at its first sentinel
     *  ID. A shard with no visible node walks nothing and merges nothing. The landing is never
     *  zeroed and grows to the widest window, which every later one reuses.
     */
    template <typename search_shard_type_, typename score_type_>
    expected<search_result_t> search_every_shard(std::size_t queries_count, std::size_t wanted,
                                                 search_shard_type_&& search_shard, nk::matrix_span<key_t> keys,
                                                 nk::matrix_span<score_type_> scores,
                                                 std::span<std::size_t> counts) noexcept {
        std::size_t const shards = this->shards(), shard_entries = queries_count * wanted;
        if (failed(landed_ids_.resize_uninitialized(shards * shard_entries)) ||
            failed(landed_scores_.resize_uninitialized(shards * shard_entries)) ||
            failed(landed_counts_.resize_uninitialized(shards * queries_count)) ||
            failed(frame_.resize_uninitialized(wanted)))
            return {{}, status_t::bad_alloc_k};
        auto shard_counts = [&](std::size_t shard) noexcept {
            return std::span<std::size_t>(landed_counts_.data() + shard * queries_count, queries_count);
        };
        auto submit_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
            if (!visible) return status_t::success_k;
            for (std::size_t& count : shard_counts(shard)) count = wanted;
            return search_shard(
                shard, nk::matrix_span<node_id_t> {landed_ids_.data() + shard * shard_entries, queries_count, wanted},
                nk::matrix_span<landed_score_t> {landed_scores_.data() + shard * shard_entries, queries_count, wanted},
                shard_counts(shard), traversal, visible);
        };
        auto merge_shard = [&](std::size_t shard, traversal_context_t const&) noexcept {
            if (!visible_[shard]) return;
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const first = shard * shard_entries + query * wanted;
                std::size_t const landed = min_of(landed_counts_[shard * queries_count + query], wanted);
                // An absent match ends the row, and must never index a key column.
                std::size_t live = 0;
                for (; live != landed && landed_ids_[first + live] != node_id_unknown_k; ++live)
                    frame_[live] = landed_match_t {landed_ids_[first + live], landed_scores_[first + live]};
                counts[query] = merge_search_results(
                    std::span<landed_match_t const> {frame_.data(), live}, host_keys(shards_[shard].keys),
                    to_span(keys.row(query).as_vector()), to_span(scores.row(query).as_vector()), counts[query]);
            }
        };
        for (std::size_t& count : counts) count = 0;
        return for_each_shard(submit_shard, merge_shard, counts);
    }

#pragma endregion Batches

  private:

    /** The shards the batch reads; borrowed from the tower for the call. */
    shards_t const& shards_;

    /** The executor every window queues on. */
    executor_t& executor_;

    /** The executor's hold, from @ref begin until the fan-out dies. */
    typename executor_t::claim_t claim_;

    /** One context per shard, sized for the widest window. */
    traversals_t traversals_;

    /** Every shard's visible rows in the one snapshot every window reads below. */
    growing_buffer<node_id_t, forkunion_allocator_t> visible_;

    /** The shards queued and not landed yet. */
    growing_buffer<std::size_t, forkunion_allocator_t> pending_;

    /** The host landing every window's matches land in, one block per shard: the IDs, the scores
     *  and each row's count; then the frame one row of them repacks into. Grown to the widest
     *  window and reused by the rest. */
    growing_buffer<node_id_t, forkunion_allocator_t> landed_ids_;
    growing_buffer<landed_score_t, forkunion_allocator_t> landed_scores_;
    growing_buffer<std::size_t, forkunion_allocator_t> landed_counts_;
    growing_buffer<landed_match_t, forkunion_allocator_t> frame_;
};

} // namespace unum::usearch::utilities
