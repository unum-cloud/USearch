/**
 *  @file include/usearch/dense/vram_apple_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The sharded index over Apple-resident dense shards - any number per device - with
 *      rotation-balanced insertion, merged cross-shard search, and SafeGraphs checkpoints.
 *
 *  The Apple twin of @c vram_nvidia_indexes: same-geometry shards, each owned and pinned to one
 *  Apple GPU, viewed as one index. Insertions rotate across the shards so they fill evenly and each
 *  holds a representative sample; searches submit to every shard's stream before awaiting any and
 *  gather through @ref merge_search_results as each lands, resolving matches into the caller's
 *  keys. Memory is unified, so the keys live on the host alone, an absorb crosses as host copies,
 *  and a submission lands its own results; everything else is @c sharded_indexes. Attention refuses
 *  with @c unsupported_k, as the shards do.
 *
 *  Beyond the executor's claim, each shard's claim word admits one batch or load at a time and
 *  closes around a refine, an absorb, a checkpoint load or a save, which refuse with
 *  @c contended_k while a batch is in flight; a shard's kept failure refuses every later mutation,
 *  and the commit sequence word lets a reader spanning the shards see each batch whole or not at
 *  all.
 */
#pragma once

#include <memory>  // `std::unique_ptr` - an index holds atomics and never moves
#include <span>    // `std::span` - the caller's keys & the merged ranking
#include <utility> // `std::move`, `std::forward`

#include "usearch/dense/sharded_indexes.hpp"
#include "usearch/dense/vram_apple_index.hpp"

namespace unum::usearch::dense {

/**
 *  @brief The sharded index: same-geometry Apple-resident shards viewed as one.
 *
 *  Each shard's keys live on the host alone: unified memory lets a walk resolve them in place, so
 *  no device twin is kept.
 */
template <                                        //
    typename key_type_ = uuid_t,                  //
    nk_dtype_t vector_dtype_ = nk_i8_k,           //
    nk_capability_t capability_ = nk_cap_metal_k, //
    typename key_hash_type_ = std::hash<key_type_>>
struct vram_apple_indexes
    : sharded_indexes<vram_apple_indexes<key_type_, vector_dtype_, capability_, key_hash_type_>, //
                      vram_apple_index<vector_dtype_, capability_>,                              //
                      metal_executor_t, key_type_, key_hash_type_> {

    using base_t = sharded_indexes<vram_apple_indexes, vram_apple_index<vector_dtype_, capability_>, metal_executor_t,
                                   key_type_, key_hash_type_>;
    using typename base_t::distance_t;
    using typename base_t::dot_result_t;
    using typename base_t::executor_t;
    using typename base_t::index_t;
    using typename base_t::key_locator_t;
    using typename base_t::key_t;
    using typename base_t::refine_plan_t;
    using typename base_t::scalar_t;
    using typename base_t::search_config_t;
    using typename base_t::search_result_t;
    using typename base_t::traversal_context_t;
    using typename base_t::vectors_span_t;

    /** The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    /** The Apple capabilities the kernels compiled under. */
    inline static constexpr nk_capability_t capability_k = capability_;

    /** The vendor whose devices the shards sit on, beside the vendors' twins of this index. */
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::apple_k;

    /** Rows one window of a batch verb stages on a shard at most: one slab, the granularity
     *  checkpoints move by, so a batch of any size crosses in bounded staging. */
    inline static constexpr std::size_t window_rows_k = power_of_two<std::size_t>(index_t::batch_rows_log2_k);

    vram_apple_indexes() = delete;
    explicit vram_apple_indexes(nk_kernel_kind_t kernel, forkunion_allocator_t allocator) noexcept
        : base_t(kernel, allocator) {}
    vram_apple_indexes(vram_apple_indexes&& other) noexcept : base_t(other.kernel_, other.allocator_) {
        this->swap(other);
    }
    vram_apple_indexes& operator=(vram_apple_indexes&& other) noexcept { return this->swap(other), *this; }

    using base_t::emplace;

    /** Appends one shard built here on the device @p stream names, for callers with none to hand
     *  over. */
    status_t emplace(typename executor_t::stream_t const& stream) noexcept {
        std::unique_ptr<index_t> shard = build_(stream.handle.device);
        if (!shard) return status_t::bad_alloc_k;
        return this->emplace(std::move(shard));
    }

    /** Grouped-query attention of every token over every shard, vetted as on NVIDIA and AMD, then
     *  answered by the first shard's face on its own stream; @c config_mismatch_k unless the shards
     *  score by dot. */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     executor_t& executor, search_config_t config = {}) noexcept {
        std::size_t const rows_count = queries.rows;
        if (this->kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const paired = this->pairs_with_(executor); paired != status_t::success_k) return {{}, paired};
        if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
        if (outputs.rows != rows_count || outputs.columns != this->dimensions() ||
            (!carry.empty() && carry.size() != rows_count))
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};
        if (queries.columns != this->dimensions()) return {{}, status_t::config_mismatch_k};
        traversal_context_t traversal {executor.stream(this->stream_of_(executor, 0)).handle.share()};
        return shards_[0].index->attend(queries, head_count, std::forward<value_lookup_type_>(value_lookup),
                                        score_scale, outputs, carry, traversal, config);
    }

    /**
     *  @brief Copies up to @p wanted rows stored under @p key into @p vectors and reports the
     *      @b true total under that key - a return above @p wanted means the copy truncated, so
     *      the caller resizes and asks again; zero means absent.
     *
     *  Safe against concurrent insertions, which only ever add; each row is read in place from
     *  unified memory, so the executor rides in the signature unused, as in the batched @ref count.
     */
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted, executor_t&) const noexcept {
        std::size_t total = 0;
        this->locators_of_key_.find_each(key, [&](key_locator_t locator) noexcept {
            if (total < wanted && total < vectors.extent(0)) [[maybe_unused]]
                status_t const copied = shards_[locator.shard].index->vector(locator.node,
                                                                             vectors.row(total).as_vector());
            ++total;
        });
        return total;
    }

    /** The batched @ref get: up to @p wanted rows of every key of @p keys, row q · @p wanted + i
     *  being the i-th row of key q, and every key's true total in @p entries; rows in any scalar
     *  but the stored one refuse with @c unsupported_k. */
    status_t get(std::span<key_t const> keys, rows_span_t rows, std::size_t wanted, std::span<std::size_t> entries,
                 executor_t& executor) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        if (!rows.holds<scalar_t>()) return status_t::unsupported_k;
        vectors_span_t const vectors = rows.as<scalar_t>();
        if (status_t const refusal = rows_refusal(keys.size() * wanted, vectors, this->row_values_()); failed(refusal))
            return refusal;
        for (std::size_t position = 0; position != keys.size(); ++position) {
            std::size_t const first = position * wanted;
            entries[position] = get(keys[position], vectors(nk::range(first, first + wanted), nk::slice), wanted,
                                    executor);
        }
        return status_t::success_k;
    }

  private:
    friend base_t;
    using base_t::shards_;

    /** One unreserved shard placed on @p device; null when the allocation fails. */
    std::unique_ptr<index_t> build_(int device) const noexcept {
        metal_shared_allocator_t const allocator {device};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(this->kernel_, allocator, allocator));
    }

    /** One unreserved shard on the device @p like sits on. */
    std::unique_ptr<index_t> rebuild_(index_t const& like) const noexcept {
        return build_(like.device_specs().ordinal);
    }

    /** Opens one traversal context per shard over its device's stream, so a fan-out never shares
     *  scratch; each grows its buffers at its first verb, so the sizing arguments go unused. */
    template <typename... usage_types_>
    status_t reserve_traversals_(growing_buffer<traversal_context_t, forkunion_allocator_t>& traversals,
                                 executor_t& executor, usage_types_ const&...) noexcept {
        if (failed(traversals.reserve_exactly(shards_.size()))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
            traversals.push_back_reserved(
                traversal_context_t {executor.stream(this->stream_of_(executor, shard)).handle.share()});
            if (!traversals.back()) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** A refine's walk contexts own their buffers, so they open as any verb's do. */
    status_t open_walks_(refine_plan_t& plan, executor_t& executor, vram_apple_indexes const&, unsigned) noexcept {
        return reserve_traversals_(plan.traversals_, executor);
    }

    /** Every shard must seat its build launch before any row is admitted, or a mixed fleet inserts
     *  partially. */
    status_t admits_insert_() const noexcept {
        for (auto const& one : shards_)
            if (!one.index->build_scratch_fits()) return status_t::config_mismatch_k;
        return status_t::success_k;
    }

    /** Every launch landed before answering and reported its telemetry then, so nothing is left to
     *  settle. */
    static status_t settle_insert_(growing_buffer<traversal_context_t, forkunion_allocator_t>&, status_t status,
                                   typename base_t::insert_result_t&) noexcept {
        return status;
    }

    /** Submits shard @p shard 's walks of @p queries; their matches land at merge time. */
    template <typename... heads_types_>
    status_t search_shard_(std::size_t shard, rows_view_t queries, nk::matrix_span<node_id_t>,
                           nk::matrix_span<distance_t>, traversal_context_t& traversal, search_config_t const& config,
                           std::size_t visible, heads_types_ const&... heads) noexcept {
        return shards_[shard].index->submit_search_(queries, traversal, config, visible, heads...).status;
    }

    /** Reads shard @p shard 's landed matches back into @p ids and @p distances. */
    void land_search_(std::size_t shard, traversal_context_t const& traversal, nk::matrix_span<node_id_t> ids,
                      nk::matrix_span<distance_t> distances) noexcept {
        shards_[shard].index->land_search_(traversal, ids, distances);
    }

    /** Submits shard @p shard 's products of @p left, as many pairs per row as @p ids is wide;
     *  they land at merge time. */
    status_t multiply_shard_(std::size_t shard, rows_view_t left, nk::matrix_span<node_id_t> ids,
                             nk::matrix_span<dot_result_t>, std::span<u32_t>, traversal_context_t& traversal,
                             search_config_t const& config, std::size_t visible) noexcept {
        return shards_[shard].index->submit_multiply_(left, ids.extent(1), traversal, config, visible).status;
    }

    /** Reads shard @p shard 's landed pairs back into @p ids, @p products and @p counts. */
    void land_products_(std::size_t shard, traversal_context_t const& traversal, nk::matrix_span<node_id_t> ids,
                        nk::matrix_span<dot_result_t> products, std::span<u32_t> counts) noexcept {
        shards_[shard].index->land_multiply_(traversal, ids, products, counts);
    }
};

} // namespace unum::usearch::dense
