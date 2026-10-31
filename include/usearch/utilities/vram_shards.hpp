/**
 *  @file include/usearch/utilities/vram_shards.hpp
 *  @author Ash Vardanian
 *  @date September 2, 2026
 *  @brief The fan-out an NVIDIA or AMD tower drives one batched read through: the executor's claim,
 *      one context per shard, one snapshot of every watermark, every shard queued before the first
 *      shard's stream merges them, and one sleeping wait before the host reads anything.
 *
 *  A tower builds one per call over its shard set and the caller's executor. Everything it calls
 *  on the device goes through @p vendor_type_, so every vendor's tower shares it as a component
 *  while keeping its own drivers.
 */
#pragma once

#include <span>    // `std::span` - the caller's rankings & the shards' staging
#include <utility> // `std::move`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the batch's contexts & staging
#include "usearch/utilities/ram_device.hpp"  // `forkunion_allocator` - where the staging lives
#include "usearch/utilities/ram_sorts.hpp"   // `rankings_refusal` - the batch's output layout

namespace unum::usearch::utilities {

/**
 *  @brief One batched read over every shard of a device tower: the executor's claim, a context per
 *      shard sized for the widest window, and one snapshot of the shards' visible entries, so every
 *      window reads the same entries.
 *
 *  @tparam shards_type_ The tower's @c shard_set.
 *  @tparam vendor_type_ The runtime the shards' devices answer to.
 *  @tparam shard_score_type_ The scores the shards' searches copy to the host, the shard's own
 *      distance unless a modality's search copies another.
 */
template <typename shards_type_, typename vendor_type_,
          typename shard_score_type_ = typename shards_type_::index_t::distance_t>
struct vram_shard_fanout {
    using shards_t = shards_type_;
    using vendor_t = vendor_type_;
    using shard_score_t = shard_score_type_;
    using index_t = typename shards_t::index_t;
    using key_t = typename shards_t::key_t;
    using node_id_t = typename shards_t::node_id_t;
    using executor_t = typename vendor_t::executor_t;
    using traversal_context_t = typename index_t::traversal_context_t;
    using traversals_t = growing_buffer<traversal_context_t, forkunion_allocator_t>;
    using search_config_t = typename index_t::search_config_t;
    using search_result_t = typename index_t::search_result_t;

    /** A fan-out of @p shards over @p executor, claimed and sized by @ref begin. */
    vram_shard_fanout(shards_t const& shards, executor_t& executor) noexcept
        : shards_(shards), executor_(executor), traversals_(shards.allocator()), visible_(shards.allocator()),
          shard_ids_(shards.allocator()), shard_scores_(shards.allocator()), shard_counts_(shards.allocator()),
          frame_(shards.allocator()) {}

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
     *  @brief Builds one traversal context per shard over its device's stream and run, for a plan
     *      that keeps its own rather than borrowing the executor's, so it never shares scratch.
     *
     *  @p usage, the arguments of a shard's @c traversal_memory_usage, sizes them for the widest
     *  shard, and every stream holds a context for each shard it serves.
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
        if (status_t const sized = executor.reserve(walk_bytes_(shards, usage...) *
                                                    shards_per_stream_(shards, executor));
            failed(sized))
            return sized;
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
     *  Why a batch of @p queries cannot rank into @p keys, @p scores and @p counts on @p executor:
     *  the checks every window repeats, run once over the whole batch before any window slices it.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The tower has no shards.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, @p counts is
     *    not @p queries long, or the row counts or widths of @p keys and @p scores disagree.
     *  - @c unsupported_layout_k A row of @p keys or @p scores is not contiguous.
     */
    template <typename score_type_>
    static status_t refusal(shards_t const& shards, executor_t const& executor, std::size_t queries,
                            nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> scores,
                            std::span<std::size_t const> counts) noexcept {
        if (status_t const paired = pairs_with(shards, executor); failed(paired)) return paired;
        if (counts.size() != queries) return status_t::config_mismatch_k;
        return rankings_refusal(queries, keys, scores);
    }

#pragma endregion Pairing

#pragma region Batches

    /** The executor every window queues on, held by the fan-out since @ref begin. */
    executor_t& executor() noexcept { return executor_; }

    /** Shards the snapshot counted, which every window walks; a shard generation opened since takes
     *  no part. */
    std::size_t shards() const noexcept { return visible_.size(); }

    /**
     *  @brief Claims the executor, snapshots the shards' visible nodes, reserves every shard's
     *      traversals for @p usage and sizes the host staging for windows of up to @p widest
     *      queries ranking @p wanted matches each.
     *
     *  @p usage is the arguments of a shard's @c traversal_memory_usage. The sizing repeats for a
     *  shard generation opened meanwhile. A call that never ranks through @ref search_every_shard
     *  passes zero @p wanted. The outputs an await's submissions carve beside that scratch are
     *  sized by each @ref await_every_shard.
     *
     *  @par Statuses:
     *  - @c contended_k Another call holds the executor.
     *  - @c uninitialized_k The executor has no streams.
     */
    template <typename... usage_types_>
    status_t begin(std::size_t widest, std::size_t wanted, usage_types_ const&... usage) noexcept {
        // The executor serializes, never the index: a second caller with its own executor searches
        // concurrently.
        auto [claim, claim_status] = executor_.claim();
        if (failed(claim_status)) return claim_status;
        claim_ = std::move(claim);
        while (true) {
            if (status_t const borrowed = borrow_traversals_(usage...); failed(borrowed)) return borrowed;
            std::size_t const shards = traversals_.size(), entries = shards * widest * wanted;
            if (failed(visible_.resize(shards)) || failed(shard_ids_.resize_uninitialized(entries)) ||
                failed(shard_scores_.resize_uninitialized(entries)) ||
                failed(shard_counts_.resize_uninitialized(shards * widest)) ||
                failed(frame_.resize_uninitialized(wanted)))
                return status_t::bad_alloc_k;
            status_t const taken = shards_.snapshot(to_writable_span(visible_)).status;
            if (taken != status_t::shape_mismatch_k) return taken;
        }
    }

    /**
     *  @brief Rewinds every stream's run, queues @p submit under the snapshot on every shard, then
     *      @p merge on the first shard's stream behind every other shard's, and sleeps once until
     *      the awaited work finishes, never polling.
     *
     *  @p submit is called as <tt>submit(shard, traversal, visible) → status_t</tt>, and @p merge, only
     *  once every shard queued, as <tt>merge(traversal) → status_t</tt> with the first shard's context,
     *  its device current. The telemetry sums over every shard; the caller reads its own outputs,
     *  the merged count included, once this returns.
     *
     *  Each shard's submission carves up to @p outputs_bytes from its stream's run, beside the
     *  scratch @ref begin sized, so a run grows at most once per call and a warm call grows none;
     *  the last await finished everything riding the runs, so each rewinds here, and what the
     *  last merge wrote stays readable by @p merge until a submission carves over it.
     *
     *  No compare-and-swap, mutex or atomic orders the shards; the events alone do:
     *  - each shard's context records its event behind its submission, a release of everything its
     *    stream wrote, the counters it copies to the host included;
     *  - the first shard's stream waits on every other shard's event, the refuser's included, an
     *    acquire on the device, so @p merge reads every shard's writes;
     *  - the first shard's context records its blocking-sync event behind @p merge, and the host
     *    sleeps on it, an acquire on the host, before reading any output or counter;
     *  - the staging is a DMA target until that wait returns, so nothing returns before it, and a
     *    staging carve is reused only by a later window or call, both queued after this wait.
     *
     *  When the first stream cannot wait on another's event, the host sleeps on every queued
     *  shard's event in turn instead, and @p merge never runs.
     */
    template <typename submit_type_, typename merge_type_>
    expected<search_result_t> await_every_shard(std::size_t outputs_bytes, submit_type_&& submit,
                                                merge_type_&& merge) noexcept {
        if (status_t const sized = executor_.reserve((walk_bytes_per_shard_ + outputs_bytes) *
                                                     shards_per_stream_(shards_, executor_));
            failed(sized))
            return {{}, sized};
        auto [queued, queued_status] = queue_every_shard_(submit);
        traversal_context_t& first = *traversals_[0];
        status_t followed = enter(shards_, 0);
        for (std::size_t shard = 1; shard != queued && succeeded(followed); ++shard)
            followed = first.follow(*traversals_[shard]);
        if (succeeded(followed) && succeeded(queued_status)) queued_status = merge(first);

        status_t awaited = status_t::success_k;
        if (succeeded(followed)) awaited = first.wait();
        else
            for (std::size_t shard = 0; shard != queued; ++shard)
                if (status_t const drained = traversals_[shard]->wait(); succeeded(awaited)) awaited = drained;
        if (succeeded(queued_status)) queued_status = succeeded(followed) ? awaited : followed;

        search_result_t merged;
        if (succeeded(awaited))
            for (std::size_t shard = 0; shard != queued; ++shard) add_spent_(merged, *traversals_[shard]);
        return {merged, queued_status};
    }

    /** Shard @p shard 's context, whose counters a caller reads once @ref await_every_shard
     *  returned. */
    traversal_context_t const& traversal(std::size_t shard) const noexcept { return *traversals_[shard]; }

    /** The merge of a call that merges on the host after the one wait, queuing nothing more. */
    static status_t merge_nothing(traversal_context_t&) noexcept { return status_t::success_k; }

    /** One match of a shard's query, scored in the type its search copies back. */
    struct shard_match_t {
        node_id_t node_id;
        shard_score_t distance;
    };
    static_assert(sizeof(shard_match_t) == 8 || sizeof(shard_match_t) == 16);

    /**
     *  @brief Runs @p search_shard on every live shard's own stream, each copying up to the
     *      @ref begin 's wanted node IDs and @c shard_score_t scores for each of @p queries_count
     *      queries, then merges them by score into @p keys, @p scores and @p counts, resolving
     *      node IDs through each shard's own keys.
     *
     *  @p search_shard is called as <tt>search_shard(shard, node_ids, scores, counts, traversal,
     *  visible) → status_t</tt> with that shard's own block of the staging: a query's matches end at
     *  its count, which starts at the wanted count for a search that reports none, or at its first
     *  sentinel ID. Every shard merges once the one wait returned, and a refused window merges
     *  nothing, so the rankings and the counts agree. A shard with no visible node walks nothing
     *  and merges nothing. The staging is never zeroed: @ref begin sized it for the widest window,
     *  which every window slices again.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The window is wider than @ref begin sized the staging for.
     */
    template <typename search_shard_type_, typename score_type_>
    expected<search_result_t> search_every_shard(std::size_t queries_count, search_shard_type_&& search_shard,
                                                 nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> scores,
                                                 std::span<std::size_t> counts) noexcept {
        std::size_t const shards = this->shards(), wanted = frame_.size(), shard_entries = queries_count * wanted;
        if (shards * queries_count > shard_counts_.size()) return {{}, status_t::config_mismatch_k};
        auto shard_counts = [&](std::size_t shard) noexcept {
            return std::span<std::size_t>(shard_counts_.data() + shard * queries_count, queries_count);
        };
        auto submit_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
            if (!visible) return status_t::success_k;
            for (std::size_t& count : shard_counts(shard)) count = wanted;
            return search_shard(
                shard, nk::matrix_span<node_id_t> {shard_ids_.data() + shard * shard_entries, queries_count, wanted},
                nk::matrix_span<shard_score_t> {shard_scores_.data() + shard * shard_entries, queries_count, wanted},
                shard_counts(shard), traversal, visible);
        };
        auto merge_shard = [&](std::size_t shard) noexcept {
            if (!visible_[shard]) return;
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const first = shard * shard_entries + query * wanted;
                std::size_t const found = min_of(shard_counts_[shard * queries_count + query], wanted);
                // An absent match ends the row, and must never index a key column.
                std::size_t live = 0;
                for (; live != found && shard_ids_[first + live] != node_id_limits<node_id_t>::unknown_k; ++live)
                    frame_[live] = shard_match_t {shard_ids_[first + live], shard_scores_[first + live]};
                std::span<shard_match_t const> const matches {frame_.data(), live};
                counts[query] = merge_search_results(matches, shards_[shard].index->keys(),
                                                     to_span(keys.row(query).as_vector()),
                                                     to_span(scores.row(query).as_vector()), counts[query]);
            }
        };
        for (std::size_t& count : counts) count = 0;
        expected<search_result_t> found = await_every_shard(0, submit_shard, merge_nothing);
        if (!found) return found;
        for (std::size_t shard = 0; shard != shards; ++shard) merge_shard(shard);
        for (std::size_t const count : counts) found.value.count += count;
        return found;
    }

#pragma endregion Batches

  private:

    /**
     *  @brief Queues @p submit on every shard in turn, its device made current and its context's
     *      event recorded behind it, stopping at the first refusal.
     *
     *  @return The shards queued, the refuser included, since its upload may already be reading
     *      the caller's queries, so every one of them must be awaited; and the first refusal.
     */
    template <typename submit_type_>
    expected<std::size_t> queue_every_shard_(submit_type_& submit) noexcept {
        // Every shard queues before any is awaited, so the batch costs the slowest device rather
        // than the sum.
        std::size_t const shards = this->shards();
        for (std::size_t shard = 0; shard != shards; ++shard) {
            // A launch fails on a stream of any device but the current one.
            status_t submitted = succeeded(enter(shards_, shard)) ? submit(shard, *traversals_[shard], visible_[shard])
                                                                  : status_t::device_error_k;
            if (status_t const recorded = traversals_[shard]->record(); succeeded(submitted)) submitted = recorded;
            if (failed(submitted)) return {shard + 1, submitted};
        }
        return {shards, status_t::success_k};
    }

    /** The widest shard's traversal bytes for @p usage, which every shard's context is sized for;
     *  a call that walks nothing, such as a gather, passes none and needs none. */
    template <typename... usage_types_>
    static std::size_t walk_bytes_(shards_t const& shards, usage_types_ const&... usage) noexcept {
        std::size_t bytes_per_shard = 0;
        if constexpr (sizeof...(usage_types_) != 0)
            for (std::size_t shard = 0; shard != shards.size(); ++shard)
                bytes_per_shard = max_of(bytes_per_shard, shards[shard].index->traversal_memory_usage(usage...));
        return bytes_per_shard;
    }

    /** The most shards any one stream of @p executor serves. */
    static std::size_t shards_per_stream_(shards_t const& shards, executor_t const& executor) noexcept {
        std::size_t most = 0;
        for (std::size_t stream = 0; stream != executor.size(); ++stream) {
            std::size_t served = 0;
            for (std::size_t shard = 0; shard != shards.size(); ++shard)
                served += stream_of(shards, executor, shard) == stream;
            most = max_of(most, served);
        }
        return most;
    }

    /**
     *  @brief Sizes every stream's run for @p usage and borrows the executor's persistent
     *      contexts, the k-th shard paired with a stream taking that stream's k-th context, so no
     *      two shards share scratch and a warm call opens none.
     *
     *  Safe only under the claim, which keeps any other call from reserving, and so from moving,
     *  those persistent contexts.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor has no streams.
     */
    template <typename... usage_types_>
    status_t borrow_traversals_(usage_types_ const&... usage) noexcept {
        // Counted once, so a shard generation opening meanwhile gets no context here.
        std::size_t const count = shards_.size(), served = shards_per_stream_(shards_, executor_);
        walk_bytes_per_shard_ = walk_bytes_(shards_, usage...);
        if (status_t const sized = executor_.reserve(walk_bytes_per_shard_ * served, served); failed(sized))
            return sized;
        traversals_.clear();
        if (failed(traversals_.reserve_exactly(count))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != count; ++shard) {
            std::size_t const stream = stream_of(shards_, executor_, shard);
            std::size_t paired_before = 0;
            for (std::size_t earlier = 0; earlier != shard; ++earlier)
                paired_before += stream_of(shards_, executor_, earlier) == stream;
            std::span<traversal_context_t> const stream_traversals = executor_.traversals(stream);
            traversals_.push_back_reserved(&stream_traversals[paired_before]);
        }
        return status_t::success_k;
    }

    /** Adds the counters @p traversal copied home behind its last submission to @p merged; read
     *  only once that submission finished. */
    static void add_spent_(search_result_t& merged, traversal_context_t const& traversal) noexcept {
        vram_telemetry_block_t const spent = traversal.spent();
        merged.visited_members += spent.hops;
        merged.computed_distances += spent.scored;
        merged.restarts += spent.restarts;
        merged.tail_compactions += spent.tail_compactions;
        merged.tail_dropped += spent.tail_dropped;
        merged.visited_saturations += spent.visited_saturations;
    }

    /** The shards the batch reads; borrowed from the tower for the call. */
    shards_t const& shards_;

    /** The executor every window queues on. */
    executor_t& executor_;

    /** The executor's hold, from @ref begin until the fan-out dies. */
    typename executor_t::claim_t claim_;

    /** One of the executor's persistent contexts per shard, borrowed under the claim. */
    growing_buffer<traversal_context_t*, forkunion_allocator_t> traversals_;

    /** The widest shard's traversal bytes @ref begin sized every run for, per shard it serves,
     *  which each await's outputs carve beside. */
    std::size_t walk_bytes_per_shard_ = 0;

    /** Every shard's visible entries in the one snapshot every window reads below. */
    growing_buffer<node_id_t, forkunion_allocator_t> visible_;

    /** The host staging every window's matches are copied into, one block per shard: the IDs, the
     *  scores and each query's count; then the frame one query's matches repack into. Sized by
     *  @ref begin for the widest window, which every window re-slices. */
    growing_buffer<node_id_t, forkunion_allocator_t> shard_ids_;
    growing_buffer<shard_score_t, forkunion_allocator_t> shard_scores_;
    growing_buffer<std::size_t, forkunion_allocator_t> shard_counts_;
    growing_buffer<shard_match_t, forkunion_allocator_t> frame_;
};

} // namespace unum::usearch::utilities
