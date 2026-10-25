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
 */
template <typename shards_type_, typename vendor_type_>
struct vram_shard_fanout {
    using shards_t = shards_type_;
    using vendor_t = vendor_type_;
    using index_t = typename shards_t::index_t;
    using key_t = typename shards_t::key_t;
    using executor_t = typename vendor_t::executor_t;
    using traversal_context_t = typename index_t::traversal_context_t;
    using traversals_t = growing_buffer<traversal_context_t, forkunion_allocator_t>;
    using distance_t = typename index_t::distance_t;
    using search_config_t = typename index_t::search_config_t;
    using search_result_t = typename index_t::search_result_t;

    /** A fan-out of @p shards over @p executor, claimed and sized by @ref begin. */
    vram_shard_fanout(shards_t const& shards, executor_t& executor) noexcept
        : shards_(shards), executor_(executor), traversals_(shards.allocator()), visible_(shards.allocator()),
          pending_(shards.allocator()), landed_ids_(shards.allocator()), landed_distances_(shards.allocator()),
          frame_(shards.allocator()) {}

#pragma region Pairing

    /** Whether @p executor drives a stream on every shard's device - shards sharing a device share
     *  its stream, and queue on it in order. */
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
        return vendor_t::select(shards[shard].index->device_specs().ordinal);
    }

    /** Builds one traversal context per shard over its device's stream and run, so a fan-out never
     *  allocates and never shares scratch; @p usage, the arguments of a shard's
     *  @c traversal_memory_usage, sizes them for the widest shard, and every stream holds a context
     *  for each shard it serves. */
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

    /** Why a batch of @p rows cannot rank into @p keys, @p scores and @p counts on @p executor: the
     *  checks every window repeats, run once over the whole batch before any window slices it. */
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

    /** Claims the executor, snapshots the shards' visible nodes and reserves every shard's traversals
     *  for @p usage, the arguments of a shard's @c traversal_memory_usage, sizing again for a shard
     *  generation opened meanwhile. */
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
     *  called as `submit(shard, traversal, visible) → status_t`, and @p merge as
     *  `merge(shard, traversal)`.
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

    /**
     *  @brief Runs @p search_shard on every live shard's own stream, each landing up to @p wanted
     *      node IDs and scores for each of @p queries_count queries, then merges them by score into
     *      @p keys, @p scores and @p counts, resolving node IDs through each shard's host keys.
     *
     *  @p search_shard is called as `search_shard(shard, ids, scores, traversal, visible) → status_t`
     *  with that shard's columns of one wide landing, a sentinel ID ending a row early. A shard with
     *  no visible node walks nothing and merges nothing. The landing is never zeroed and grows to
     *  the widest window, which every later one reuses.
     */
    template <typename search_shard_type_, typename score_type_>
    expected<search_result_t> search_every_shard(std::size_t queries_count, std::size_t wanted,
                                                 search_shard_type_&& search_shard, nk::matrix_span<key_t> keys,
                                                 nk::matrix_span<score_type_> scores,
                                                 std::span<std::size_t> counts) noexcept {
        std::size_t const shards = this->shards();
        std::size_t const landing_entries = queries_count * shards * wanted;
        frame_.clear();
        if (failed(landed_ids_.reserve_exactly(landing_entries)) ||
            failed(landed_distances_.reserve_exactly(landing_entries)) || failed(frame_.reserve_exactly(wanted)))
            return {{}, status_t::bad_alloc_k};
        frame_.extend_uninitialized(wanted);
        nk::matrix_span<node_id_t> const ids_wide {landed_ids_.data(), queries_count, shards * wanted};
        nk::matrix_span<distance_t> const distances_wide {landed_distances_.data(), queries_count, shards * wanted};

        auto shard_columns = [&](std::size_t shard) noexcept {
            return nk::range(shard * wanted, (shard + 1) * wanted);
        };
        auto submit_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
            if (!visible) return status_t::success_k;
            return search_shard(shard, ids_wide(nk::all, shard_columns(shard), nk::slice),
                                distances_wide(nk::all, shard_columns(shard), nk::slice), traversal, visible);
        };
        auto merge_shard = [&](std::size_t shard, traversal_context_t const&) noexcept {
            if (!visible_[shard]) return;
            nk::matrix_span<node_id_t> const shard_ids = ids_wide(nk::all, shard_columns(shard), nk::slice);
            nk::matrix_span<distance_t> const shard_distances = distances_wide(nk::all, shard_columns(shard),
                                                                               nk::slice);
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const live = repack_landed_matches_(
                    shard_ids.row(query).as_vector(), shard_distances.row(query).as_vector(), to_writable_span(frame_));
                counts[query] = merge_search_results(
                    std::span<scored_node_t const> {frame_.data(), live}, host_keys(shards_[shard].keys),
                    to_span(keys.row(query).as_vector()), to_span(scores.row(query).as_vector()), counts[query]);
            }
        };
        for (std::size_t& count : counts) count = 0;
        return for_each_shard(submit_shard, merge_shard, counts);
    }

#pragma endregion Batches

  private:

    /** Zips one (query, shard) pair of parallel landing rows into the scored shape
     *  @c merge_search_results consumes, stopping at the first sentinel ID - an absent match must
     *  never index a key column. @return The live matches repacked. */
    static std::size_t repack_landed_matches_(nk::vector_view<node_id_t> landed_ids,
                                              nk::vector_view<distance_t> landed_distances,
                                              std::span<scored_node_t> frame) noexcept {
        std::size_t live = 0;
        for (; live != landed_ids.size() && landed_ids[live] != node_id_unknown_k; ++live)
            frame[live] = scored_node_t {landed_ids[live], landed_distances[live]};
        return live;
    }

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

    /** The host landing every window's matches land in, and the frame one row of them repacks into,
     *  grown to the widest window and reused by the rest. */
    growing_buffer<node_id_t, forkunion_allocator_t> landed_ids_;
    growing_buffer<distance_t, forkunion_allocator_t> landed_distances_;
    growing_buffer<scored_node_t, forkunion_allocator_t> frame_;
};

} // namespace unum::usearch::utilities
