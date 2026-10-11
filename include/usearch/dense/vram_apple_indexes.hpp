/**
 *  @file include/usearch/dense/vram_apple_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The keyed index over the one Apple GPU a Mac carries - the Apple twin of
 *      @c dense_vram_nvidia_indexes.
 *
 *  Speaks the sharded index's verbs, so the C layer drives every vendor alike, over the single
 *  shard Apple's one device hosts: keys resolve through a host key column and the locator map, an
 *  insert plan publishes its steps together at @c finish, and checkpoints ride the SafeGraphs
 *  interchange - one monolith, or one part beside a manifest. Attention refuses with
 *  @c unsupported_k, as the shard does.
 */
#pragma once

#include <cstring> // `std::memcmp` - the monolith's magic

#include <memory>      // `std::unique_ptr` - a shard holds atomics and never moves
#include <mutex>       // `std::unique_lock`, `std::adopt_lock` - the one-batch-at-a-time hold
#include <span>        // `std::span`
#include <string_view> // `std::string_view` - the checkpoint paths
#include <utility>     // `std::move`, `std::exchange`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t`, `copy_bytes` - the checkpoints' bytes
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the shard table & landings
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator` - the host columns' placement
#include "usearch/utilities/ram_tables.hpp"        // `chunked_multimap` - the locator map
#include "usearch/utilities/safegraphs_format.hpp" // SafeGraphs monoliths & manifests - the checkpoints
#include "usearch/dense/types.hpp"
#include "usearch/dense/vram_apple_index.hpp"

namespace unum {
namespace usearch {

/** The keyed index over one Apple GPU's single shard. */
template <                                        //
    typename key_type_ = uuid_t,                  //
    nk_dtype_t vector_dtype_ = nk_i8_k,           //
    nk_capability_t capability_ = nk_cap_metal_k, //
    typename key_hash_type_ = std::hash<key_type_>>
struct dense_vram_apple_indexes {

    using key_t = key_type_;

    /** Where one entry lives: the shard, and its ID there. */
    struct key_locator_t {
        std::size_t shard = 0;
        dense_node_id_t node = 0;
    };

    using locators_map_t = chunked_multimap<key_t, key_locator_t, key_hash_type_, forkunion_allocator_t>;
    using index_t = dense_vram_apple_index<vector_dtype_, capability_>;
    using keys_column_t =
        chunked_column<key_t, chunk_schedule<10, 20>, index_t::max_node_count_k, forkunion_allocator_t>;
    using executor_t = metal_executor_t;
    using dense_traversal_context_t = typename index_t::dense_traversal_context_t;
    using distance_t = typename index_t::distance_t;
    using scalar_t = typename index_t::scalar_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using vectors_span_t = nk::matrix_span<scalar_t>;
    using search_config_t = typename index_t::search_config_t;
    using insert_config_t = typename index_t::insert_config_t;
    using reserve_capacity_t = typename index_t::reserve_capacity_t;
    using insert_result_t = typename index_t::insert_result_t;
    using search_result_t = typename index_t::search_result_t;
    using refine_config_t = typename index_t::refine_config_t;
    using refine_result_t = typename index_t::refine_result_t;
    using health_result_t = typename index_t::health_result_t;

    /** The graph on the device and the keys of its rows on the host. */
    struct shard_t {
        std::unique_ptr<index_t> index;
        keys_column_t keys;
    };

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;
    inline static constexpr nk_capability_t capability_k = capability_;

    /** The vendor whose device the shard sits on, beside the vendors' twins of this index. */
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::apple_k;

#pragma region Placement and Measures

    dense_vram_apple_indexes() = delete;
    explicit dense_vram_apple_indexes(nk_kernel_kind_t kernel, forkunion_allocator_t allocator) noexcept
        : kernel_(kernel), shards_(allocator), locators_of_key_(allocator), allocator_(allocator) {}
    dense_vram_apple_indexes(dense_vram_apple_indexes const&) = delete;
    dense_vram_apple_indexes& operator=(dense_vram_apple_indexes const&) = delete;
    ~dense_vram_apple_indexes() noexcept = default;

    /** Places the one shard on @p device; a second refuses, since one Apple device hosts one
     *  shard. */
    status_t emplace(int device) noexcept {
        if (!shards_.empty()) return status_t::config_mismatch_k;
        std::unique_ptr<index_t> shard {new (std::nothrow) index_t(kernel_, metal_shared_allocator_t {device})};
        if (!shard) return status_t::bad_alloc_k;
        return shards_.push_back(shard_t {std::move(shard), keys_column_t {allocator_}});
    }

    /** Empties the shard in place, keeping its device, geometry and capacity. */
    status_t clear() noexcept {
        if (shards_.empty()) return status_t::success_k;
        reserve_capacity_t const geometry {capacity(), static_cast<connectivity_t>(connectivity()), dimensions()};
        int const device = shards_[0].index->device_specs().ordinal;
        shards_.clear();
        locators_of_key_.clear();
        if (status_t const placed = emplace(device); failed(placed)) return placed;
        return geometry.capacity ? reserve(geometry).status : status_t::success_k;
    }

    nk_kernel_kind_t kernel() const noexcept { return kernel_; }
    std::size_t shards_count() const noexcept { return shards_.size(); }
    index_t& shard(std::size_t shard) noexcept { return *shards_[shard].index; }
    index_t const& shard(std::size_t shard) const noexcept { return *shards_[shard].index; }
    std::size_t dimensions() const noexcept { return shards_.empty() ? 0 : shards_[0].index->dimensions(); }
    std::size_t connectivity() const noexcept { return shards_.empty() ? 0 : shards_[0].index->connectivity(); }
    std::size_t size() const noexcept { return shards_.empty() ? 0 : shards_[0].index->size(); }
    std::size_t capacity() const noexcept { return shards_.empty() ? 0 : shards_[0].index->capacity(); }
    status_t status() const noexcept { return shards_.empty() ? status_t::success_k : shards_[0].index->status(); }

    std::size_t memory_usage() const noexcept {
        std::size_t bytes = sizeof(dense_vram_apple_indexes) + locators_of_key_.memory_usage();
        for (shard_t const& one : shards_) bytes += one.index->memory_usage() + one.keys.memory_usage();
        return bytes;
    }

    /** Grows the shard and the key column ahead of @p config.capacity rows; the first call freezes
     *  the geometry. */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (shards_.empty()) return {0, status_t::uninitialized_k};
        auto const [shard_capacity, status] = shards_[0].index->reserve(config);
        if (failed(status)) return {capacity(), status};
        if (status_t const covered = shards_[0].keys.cover(shard_capacity); failed(covered))
            return {capacity(), covered};
        if (status_t const indexed = locators_of_key_.reserve(config.capacity); failed(indexed))
            return {capacity(), indexed};
        return {capacity(), status_t::success_k};
    }

#pragma endregion Placement and Measures

#pragma region Insertion

    /** One batch in flight: the executor and the insertion gate held for the plan's whole life, and
     *  the rows each step runs on the device under their keys, all published together by
     *  @ref finish. */
    struct [[nodiscard]] insert_plan_t {
        insert_plan_t() noexcept = default;
        insert_plan_t(insert_plan_t&& other) noexcept
            : seats_(other.seats_), config_(other.config_), rows_(other.rows_), launched_(other.launched_),
              first_id_(other.first_id_), result_(other.result_), status_(other.status_),
              traversal_(std::move(other.traversal_)), streams_(std::move(other.streams_)),
              mutating_(std::move(other.mutating_)), owner_(std::exchange(other.owner_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;

        /** An unfinished plan keeps @c cancelled_k on the shard when it ran any rows. */
        ~insert_plan_t() noexcept {
            if (owner_ && launched_) [[maybe_unused]]
                status_t const kept = owner_->shards_[0].index->fail_(status_t::cancelled_k);
        }

        /** Runs rows @b [first_row,first_row+rows) of the plan under @p keys; the rows take IDs in
         *  launch order. */
        status_t insert(std::size_t first_row, std::span<key_t const> keys, vectors_view_t rows) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (failed(status_)) return status_;
            std::size_t const count = rows.extent(0);
            if (keys.size() != count || first_row > rows_ || count > rows_ - first_row ||
                (count && rows.extent(1) != owner_->dimensions()))
                return status_t::config_mismatch_k;
            // Overlapping steps would write keys past the cover the plan grew.
            if (launched_ + count > rows_) return status_t::config_mismatch_k;
            if (!count) return status_t::success_k;
            shard_t& one = owner_->shards_[0];
            dense_node_id_t const first_node = static_cast<dense_node_id_t>(first_id_ + launched_);
            for (std::size_t row = 0; row != count; ++row) one.keys[first_node + row] = keys[row];
            expected<insert_result_t> const inserted = one.index->submit_insert_(rows, {}, *traversal_, config_);
            if (!inserted) return status_ = inserted.status;
            if (inserted.value.first_id != first_node) return status_ = status_t::contended_k;
            result_.visited_members += inserted.value.visited_members;
            result_.computed_distances += inserted.value.computed_distances;
            result_.restarts += inserted.value.restarts;
            result_.unlinked_neighbors += inserted.value.unlinked_neighbors;
            result_.visited_saturations += inserted.value.visited_saturations;
            launched_ += count;
            return status_t::success_k;
        }

        /** Publishes every row at once and enters their keys; a plan short of rows refuses and
         *  stays whole. */
        expected<insert_result_t> finish() noexcept {
            if (!owner_) return {{}, status_t::uninitialized_k};
            if (succeeded(status_) && launched_ != rows_) return {{owner_->size()}, status_t::config_mismatch_k};
            dense_vram_apple_indexes& owner = *std::exchange(owner_, nullptr);
            index_t& index = *owner.shards_[0].index;
            if (failed(status_)) {
                if (launched_) [[maybe_unused]]
                    status_t const kept = index.fail_(status_);
                result_.new_size = owner.size();
                return {result_, status_};
            }
            if (launched_)
                if (status_t const published = index.publish_(first_id_, launched_); failed(published))
                    return {result_, published};
            for (std::size_t row = 0; row != launched_; ++row) {
                dense_node_id_t const node = static_cast<dense_node_id_t>(first_id_ + row);
                owner.locators_of_key_.insert(seats_, owner.shards_[0].keys[node], key_locator_t {0, node});
            }
            result_.first_id = static_cast<dense_node_id_t>(first_id_);
            result_.new_size = owner.size();
            return {result_, status_t::success_k};
        }

      private:
        typename locators_map_t::seats_t seats_ {};
        insert_config_t config_ {};
        std::size_t rows_ = 0;
        std::size_t launched_ = 0;
        std::size_t first_id_ = 0;
        insert_result_t result_ {};
        status_t status_ = status_t::success_k;
        std::unique_ptr<dense_traversal_context_t> traversal_;
        typename executor_t::claim_t streams_;
        std::unique_lock<fu::spin_mutex_t> mutating_;
        dense_vram_apple_indexes* owner_ = nullptr;

        friend dense_vram_apple_indexes;
    };

    /** Opens a batch of @p rows: takes the executor and the insertion gate, grows the shard and its
     *  key column, and claims the locator seats. */
    expected<insert_plan_t> plan_insert(std::size_t rows, executor_t& executor, insert_config_t config = {}) noexcept {
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const paired = pairs_with_(executor); failed(paired)) return {{}, paired};
        insert_plan_t plan;
        plan.rows_ = rows;
        plan.config_ = config;
        if (!rows) {
            plan.owner_ = this;
            return {std::move(plan), status_t::success_k};
        }
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.streams_ = std::move(claim);
        if (!mutating_.lock_if_free()) return {{}, status_t::contended_k};
        plan.mutating_ = std::unique_lock<fu::spin_mutex_t> {mutating_, std::adopt_lock};
        plan.traversal_.reset(new (std::nothrow) dense_traversal_context_t {traversal_queue_(executor)});
        if (!plan.traversal_ || !*plan.traversal_) return {{}, status_t::bad_alloc_k};
        shard_t& one = shards_[0];
        plan.first_id_ = one.index->claimed_();
        reserve_capacity_t const grown {plan.first_id_ + rows, static_cast<connectivity_t>(connectivity()),
                                        dimensions()};
        if (status_t const reserved = one.index->reserve(grown).status; failed(reserved)) return {{}, reserved};
        if (status_t const covered = one.keys.cover(plan.first_id_ + rows); failed(covered)) return {{}, covered};
        auto [seats, seat_status] = locators_of_key_.claim(rows);
        if (failed(seat_status)) return {{}, seat_status};
        plan.seats_ = seats;
        plan.owner_ = this;
        return {std::move(plan), status_t::success_k};
    }

    /** Inserts a batch under @p keys: @ref plan_insert, one step over every row, and its
     *  @c finish. */
    expected<insert_result_t> insert(std::span<key_t const> keys, vectors_view_t new_vectors, executor_t& executor,
                                     insert_config_t config = {}) noexcept {
        if (keys.size() != new_vectors.extent(0)) return {{size()}, status_t::config_mismatch_k};
        auto [plan, status] = plan_insert(new_vectors.extent(0), executor, config);
        if (failed(status)) return {{size()}, status};
        [[maybe_unused]] status_t const launched = plan.insert(0, keys, new_vectors);
        return plan.finish();
    }

    /** Inserts one row under @p key. */
    expected<insert_result_t> insert(key_t const& key, vector_view_t new_vector, executor_t& executor,
                                     insert_config_t config = {}) noexcept {
        if (!new_vector.is_contiguous()) return {{}, status_t::unsupported_layout_k};
        vectors_view_t const row {new_vector.data(), 1u, new_vector.size()};
        return insert(std::span<key_t const> {&key, 1u}, row, executor, config);
    }

#pragma endregion Insertion

#pragma region Refinement

    /** Re-selects the neighbors of every stored node and heals whatever that strands; @p on_visited
     *  fires once with the nodes swept and the population. */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> refine(executor_t& executor, refine_config_t config = {},
                                     progress_type_&& on_visited = {}) noexcept {
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const paired = pairs_with_(executor); failed(paired)) return {{}, paired};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        if (!mutating_.lock_if_free()) return {{}, status_t::contended_k};
        std::lock_guard const mutating {mutating_, std::adopt_lock};
        dense_traversal_context_t traversal {traversal_queue_(executor)};
        std::size_t const published = size();
        auto const [result, status] = shards_[0].index->refine({0, dense_node_id_unknown_k}, traversal, config);
        if (failed(status)) return {result, status};
        if (on_visited(result.visited_nodes, published) == continuation_t::cancel_k)
            return {result, status_t::cancelled_k};
        return {result, status_t::success_k};
    }

    /** Measures how much of the shard a search can reach. */
    expected<health_result_t> health(executor_t& executor) noexcept {
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (status_t const paired = pairs_with_(executor); failed(paired)) return {{}, paired};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        dense_traversal_context_t traversal {traversal_queue_(executor)};
        return shards_[0].index->health(traversal);
    }

#pragma endregion Refinement

#pragma region Search

    /** One query's nearest keys, ascending by distance. */
    expected<search_result_t> search(vector_view_t query_vector, executor_t& executor, search_config_t config,
                                     std::span<key_t> keys, std::span<distance_t> distances) noexcept {
        search_result_t merged;
        if (status_t const paired = pairs_with_(executor); failed(paired)) return {merged, paired};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {merged, claim_status};
        dense_traversal_context_t traversal {traversal_queue_(executor)};
        auto const [found, status] = shards_[0].index->search_one_(query_vector, traversal, config, size());
        if (failed(status)) return {merged, status};
        merged = found;
        merged.count = merge_search_results(to_span(traversal.results), shards_[0].keys, keys, distances, 0);
        return {merged, status_t::success_k};
    }

    /** The batched search, every query's matches resolved into keys. */
    expected<search_result_t> search(vectors_view_t queries, executor_t& executor, search_config_t config,
                                     nk::matrix_span<key_t> keys, nk::matrix_span<distance_t> distances,
                                     std::span<std::size_t> counts) noexcept {
        return search_(queries, executor, config, keys, distances, counts, dummy_heads_t {});
    }

    /** The batched search ranked by a lightning indexer; a dot verb, refusing every other
     *  metric. */
    expected<search_result_t> search(vectors_view_t queries, dense_indexer_heads<scalar_t> const& heads,
                                     executor_t& executor, search_config_t config, nk::matrix_span<key_t> keys,
                                     nk::matrix_span<distance_t> distances, std::span<std::size_t> counts) noexcept
        requires dense_floating_scalar<scalar_t>
    {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return search_(queries, executor, config, keys, distances, counts, heads);
    }

    /** Graph products of every left-hand row, keys resolved and dots at their native width. */
    expected<search_result_t> multiply(vectors_view_t left, executor_t& executor, search_config_t config,
                                       nk::matrix_span<key_t> keys, nk::matrix_span<dot_result_t> products,
                                       std::span<std::size_t> counts) noexcept {
        std::size_t const rows_count = left.extent(0), wanted = keys.extent(1);
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const paired = pairs_with_(executor); failed(paired)) return {{}, paired};
        if (counts.size() != rows_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = dense_rankings_refusal(rows_count, keys, products); failed(refusal))
            return {{}, refusal};
        if (!rows_count) return {{}, status_t::success_k};
        if (left.extent(1) != dimensions()) return {{}, status_t::config_mismatch_k};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        growing_buffer<dense_node_id_t, forkunion_allocator_t> ids_landing {allocator_};
        growing_buffer<u32_t, forkunion_allocator_t> counts_landing {allocator_};
        if (failed(ids_landing.resize_uninitialized(rows_count * wanted)) ||
            failed(counts_landing.resize_uninitialized(rows_count)))
            return {{}, status_t::bad_alloc_k};
        nk::matrix_span<dense_node_id_t> const ids {ids_landing.data(), rows_count, wanted};
        dense_traversal_context_t traversal {traversal_queue_(executor)};
        expected<search_result_t> multiplied = shards_[0].index->multiply_(
            left, ids, products, to_writable_span(counts_landing), traversal, config, size());
        if (!multiplied) return multiplied;
        for (std::size_t row = 0; row != rows_count; ++row) {
            counts[row] = counts_landing[row];
            for (std::size_t pair = 0; pair != min_of<std::size_t>(counts[row], wanted); ++pair)
                keys(row, pair) = shards_[0].keys[ids(row, pair)];
        }
        return multiplied;
    }

    /** Late-interaction scores of query groups, from the token rows' ranked pools. */
    expected<search_result_t> maxsim(vectors_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, nk::matrix_span<key_t> keys, nk::matrix_span<f64_t> scores,
                                     std::span<std::size_t> counts) noexcept {
        std::size_t const rows_count = queries.extent(0);
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const paired = pairs_with_(executor); failed(paired)) return {{}, paired};
        if (!rows_per_query || rows_count % rows_per_query) return {{}, status_t::shape_mismatch_k};
        std::size_t const groups_count = rows_count / rows_per_query;
        if (counts.size() != groups_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = dense_rankings_refusal(groups_count, keys, scores); failed(refusal))
            return {{}, refusal};
        if (!groups_count) return {{}, status_t::success_k};
        std::size_t const width = shards_[0].index->non_zero_expansion_factor(config);
        config.count = static_cast<typename index_t::small_size_t>(width);
        growing_buffer<key_t, forkunion_allocator_t> row_keys {allocator_};
        growing_buffer<distance_t, forkunion_allocator_t> row_distances {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> row_counts {allocator_};
        dense_maxsim_reducer<key_t, distance_t, forkunion_allocator_t> reduce {allocator_};
        if (failed(row_keys.resize_uninitialized(rows_count * width)) ||
            failed(row_distances.resize_uninitialized(rows_count * width)) ||
            failed(row_counts.resize_uninitialized(rows_count)) || failed(reduce.reserve(rows_per_query, width)))
            return {{}, status_t::bad_alloc_k};
        nk::matrix_span<key_t> const ranked_keys {row_keys.data(), rows_count, width};
        nk::matrix_span<distance_t> const ranked_distances {row_distances.data(), rows_count, width};
        expected<search_result_t> ranked = search(queries, executor, config, ranked_keys, ranked_distances,
                                                  to_writable_span(row_counts));
        if (!ranked) return ranked;
        ranked.value.count = reduce(ranked_keys, ranked_distances, to_span(row_counts), keys, scores, counts);
        return ranked;
    }

    /** Device attention is NVIDIA's alone until its value lookup has a Metal ABI. */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(vectors_view_t, std::size_t, value_lookup_type_&&, f32_t, nk::matrix_span<f32_t>,
                                     std::span<dense_softmax_state_t>, executor_t&, search_config_t = {}) noexcept {
        return {{}, status_t::unsupported_k};
    }

#pragma endregion Search

#pragma region Checkpoints

    /** Bytes a @ref save under @p config writes: one monolith, or the one part and the manifest. */
    expected<std::size_t> storage_requirements(save_config_t const& config) const noexcept {
        if (shards_.empty() || !dimensions()) return {0, status_t::uninitialized_k};
        if (status_t const valid = config.validate(); failed(valid)) return {0, valid};
        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields {allocator_};
        if (status_t const named = custom_fields_(fields, config); failed(named)) return {0, named};
        auto const [header, status] = monolith_header_(size(), config, to_span(fields));
        if (failed(status)) return {0, status};
        if (config.layout == checkpoint_layout_t::monolith_k) return {header.monolith_bytes(), status_t::success_k};
        growing_buffer<char, forkunion_allocator_t> entry_path {allocator_};
        status_t framing = status_t::success_k;
        auto part_at = [&](std::size_t part) noexcept {
            auto const [name, path_status] = safegraphs_part_path(entry_path, "index", part, 1);
            if (failed(path_status)) framing = path_status;
            return safegraphs_part_t {name, header.monolith_bytes(), size()};
        };
        auto const [manifest, manifest_status] = safegraphs_manifest_bytes(1, part_at);
        if (failed(manifest_status)) return {0, manifest_status};
        if (failed(framing)) return {0, framing};
        return {header.monolith_bytes() + manifest, status_t::success_k};
    }

    /** Serializes the index under @p path: a monolith, or one part beside a manifest that commits
     *  it. */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(std::string_view path, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        if (status_t const ready = saveable_(config, executor); failed(ready)) return ready;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        bool const distributed = config.layout == checkpoint_layout_t::distributed_k;
        growing_buffer<char, forkunion_allocator_t> part_path {allocator_};
        std::string_view file_path = path;
        if (distributed) {
            auto const [named, name_status] = safegraphs_part_path(part_path, safegraphs_stem(path), 0, 1);
            if (failed(name_status)) return name_status;
            file_path = named;
        }
        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields {allocator_};
        if (status_t const named = custom_fields_(fields, config); failed(named)) return named;
        auto const [header, header_status] = monolith_header_(size(), config, to_span(fields));
        if (failed(header_status)) return header_status;
        auto [mapping, mapping_status] = mapped_file_t::write(file_path, header.monolith_bytes());
        if (failed(mapping_status)) return mapping_status;
        if (status_t const written = write_monolith_(mapping.span(), header, config); failed(written)) return written;
        if (on_saved(0) == continuation_t::cancel_k) return status_t::cancelled_k;
        if (status_t const framed = safegraphs_write_header(mapping.span(), header, to_span(fields)); failed(framed))
            return framed;
        if (status_t const flushed = mapping.sync(); failed(flushed)) return flushed;
        if (!distributed) return status_t::success_k;
        std::string_view const base = safegraphs_basename(safegraphs_stem(path));
        growing_buffer<char, forkunion_allocator_t> entry_path {allocator_};
        auto part_at = [&](std::size_t part) noexcept {
            auto const [name, status] = safegraphs_part_path(entry_path, base, part, 1);
            return safegraphs_part_t {name, header.monolith_bytes(), header.nodes()};
        };
        return safegraphs_write_manifest(path, 1, part_at);
    }

    /** Serializes the index into @p monolith as one graph; a manifest has nowhere to resolve its
     *  part. */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(bytes_span_t monolith, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        if (status_t const ready = saveable_(config, executor); failed(ready)) return ready;
        if (config.layout != checkpoint_layout_t::monolith_k) return status_t::config_mismatch_k;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields {allocator_};
        if (status_t const named = custom_fields_(fields, config); failed(named)) return named;
        auto const [header, header_status] = monolith_header_(size(), config, to_span(fields));
        if (failed(header_status)) return header_status;
        if (monolith.size() < header.monolith_bytes()) return status_t::overflow_k;
        if (status_t const written = write_monolith_(monolith, header, config); failed(written)) return written;
        if (on_saved(0) == continuation_t::cancel_k) return status_t::cancelled_k;
        return safegraphs_write_header(monolith, header, to_span(fields));
    }

    /** Restores the checkpoint at @p path - a monolith or a manifest - into the empty shard; a
     *  failed or cancelled load clears it. */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(std::string_view path, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        growing_buffer<mapped_file_t, forkunion_allocator_t> mappings {allocator_};
        growing_buffer<part_t, forkunion_allocator_t> parts {allocator_};
        if (status_t const ready = loadable_(executor); failed(ready)) return ready;
        if (status_t const listed = catalog_(path, mappings, parts); failed(listed)) return listed;
        return load_parts_(parts, config, executor, on_loaded);
    }

    /** Restores one monolith held in @p monolith. */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(bytes_view_t monolith, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        growing_buffer<part_t, forkunion_allocator_t> parts {allocator_};
        if (status_t const ready = loadable_(executor); failed(ready)) return ready;
        if (status_t const listed = catalog_monolith_(monolith, parts); failed(listed)) return listed;
        return load_parts_(parts, config, executor, on_loaded);
    }

    static expected<safegraphs_metadata_t> metadata(std::string_view path) noexcept {
        return safegraphs_read_metadata(path);
    }
    static expected<safegraphs_metadata_t> metadata(bytes_view_t buffer) noexcept {
        return safegraphs_read_metadata(buffer);
    }

#pragma endregion Checkpoints

#pragma region Lookups

    std::size_t count(key_t const& key) const noexcept { return locators_of_key_.count(key); }
    bool contains(key_t const& key) const noexcept { return locators_of_key_.contains(key); }

    /** Copies up to @p wanted rows stored under @p key and answers the true total under it. */
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted) const noexcept {
        std::size_t total = 0;
        locators_of_key_.find_each(key, [&](key_locator_t locator) noexcept {
            if (total < wanted && total < vectors.extent(0)) [[maybe_unused]]
                status_t const copied = shards_[locator.shard].index->vector(locator.node,
                                                                             vectors.row(total).as_vector());
            ++total;
        });
        return total;
    }

    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t&) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        for (std::size_t position = 0; position != keys.size(); ++position) entries[position] = count(keys[position]);
        return status_t::success_k;
    }

    status_t contains(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t&) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        for (std::size_t position = 0; position != keys.size(); ++position)
            entries[position] = contains(keys[position]) ? 1 : 0;
        return status_t::success_k;
    }

    status_t get(std::span<key_t const> keys, vectors_span_t vectors, std::size_t wanted,
                 std::span<std::size_t> entries, executor_t&) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        std::size_t const row_values = divide_round_up<std::size_t>(dimensions(), nk::dimensions_per_value<scalar_t>());
        if (status_t const refusal = dense_rows_refusal(keys.size() * wanted, vectors, row_values); failed(refusal))
            return refusal;
        for (std::size_t position = 0; position != keys.size(); ++position) {
            std::size_t const first = position * wanted;
            entries[position] = get(keys[position], vectors(nk::range(first, first + wanted), nk::slice), wanted);
        }
        return status_t::success_k;
    }

#pragma endregion Lookups

  private:

    /** One monolith of a checkpoint: where its bytes are, its header, and the tokens its metadata
     *  carries. */
    struct part_t {
        iota_view<std::size_t> path;
        bytes_view_t monolith;
        std::size_t nodes = 0;
        safegraphs_header_t header;
        safegraphs_name_t metric;
        safegraphs_name_t quantization;
    };

    /** Whether @p executor drives this index's one device. */
    status_t pairs_with_(executor_t const& executor) const noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        if (executor.size() != 1 || executor.device(0) != shards_[0].index->device_specs().ordinal)
            return status_t::config_mismatch_k;
        return status_t::success_k;
    }

    /** The queue a context opened over @p executor submits to. */
    static dense_metal_queue_t traversal_queue_(executor_t& executor) noexcept {
        return {executor.device(0), executor.stream(0).handle};
    }

    /** The batched search body, under any indexer @p heads. */
    template <typename heads_type_>
    expected<search_result_t> search_(vectors_view_t queries, executor_t& executor, search_config_t config,
                                      nk::matrix_span<key_t> keys, nk::matrix_span<distance_t> distances,
                                      std::span<std::size_t> counts, heads_type_ const& heads) noexcept {
        std::size_t const queries_count = queries.extent(0);
        if (status_t const paired = pairs_with_(executor); failed(paired)) return {{}, paired};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = dense_rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        if (!queries_count) return {{}, status_t::success_k};
        if (queries.extent(1) != dimensions() || config.count > 512) return {{}, status_t::config_mismatch_k};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        growing_buffer<dense_node_id_t, forkunion_allocator_t> ids_landing {allocator_};
        growing_buffer<dense_distance_t, forkunion_allocator_t> distances_landing {allocator_};
        if (failed(ids_landing.resize_uninitialized(queries_count * config.count)) ||
            failed(distances_landing.resize_uninitialized(queries_count * config.count)))
            return {{}, status_t::bad_alloc_k};
        nk::matrix_span<dense_node_id_t> const ids {ids_landing.data(), queries_count, config.count};
        nk::matrix_span<dense_distance_t> const landed {distances_landing.data(), queries_count, config.count};
        dense_traversal_context_t traversal {traversal_queue_(executor)};
        expected<search_result_t> found = shards_[0].index->search_(queries, ids, landed, traversal, config, size(),
                                                                    heads);
        if (!found) return found;
        dense_scored_node_t frame[512];
        found.value.count = 0;
        for (std::size_t query = 0; query != queries_count; ++query) {
            std::size_t live = 0;
            for (; live != config.count && ids(query, live) != dense_node_id_unknown_k; ++live)
                frame[live] = dense_scored_node_t {ids(query, live), landed(query, live)};
            counts[query] = merge_search_results(std::span<dense_scored_node_t const> {frame}.first(live),
                                                 shards_[0].keys, to_span(keys.row(query).as_vector()),
                                                 to_span(distances.row(query).as_vector()), 0);
            found.value.count += counts[query];
        }
        return found;
    }

    inline static constexpr dtype_t key_dtype_k = safegraphs_key_dtype<key_t>();
    inline static constexpr dtype_t value_column_dtype_k = safegraphs_value_dtype<scalar_t>();

    dtype_t norms_column_dtype_() const noexcept {
        return !shards_.empty() && shards_[0].index->stores_norms() ? safegraphs_value_dtype<dot_result_t>()
                                                                    : dtype_t::unknown_k;
    }

    status_t saveable_(save_config_t const& config, executor_t const& executor) const noexcept {
        static_assert(key_dtype_k != dtype_t::unknown_k, "The key type has no interchange token");
        if (shards_.empty() || !dimensions()) return status_t::uninitialized_k;
        if (value_column_dtype_k == dtype_t::unknown_k) return status_t::unknown_kind_k;
        if (config.synchronization == synchronization_t::concurrent_k) return status_t::config_mismatch_k;
        if (status_t const valid = config.validate(); failed(valid)) return valid;
        if (status_t const kept = status(); failed(kept)) return kept;
        return pairs_with_(executor);
    }

    status_t loadable_(executor_t const& executor) const noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        if (size()) return status_t::config_mismatch_k;
        if (status_t const kept = status(); failed(kept)) return kept;
        return pairs_with_(executor);
    }

    status_t custom_fields_(growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t>& fields,
                            save_config_t const& config) const noexcept {
        if (failed(fields.reserve_exactly(2 + config.provenance.size()))) return status_t::bad_alloc_k;
        fields.push_back_reserved({"metric", nk_kernel_name(kernel_)});
        fields.push_back_reserved({"quantization", nk_dtype_name(vector_dtype_)});
        for (safegraphs_custom_field_t const& field : config.provenance) fields.push_back_reserved(field);
        return status_t::success_k;
    }

    expected<safegraphs_header_t> monolith_header_(std::size_t nodes, save_config_t const& config,
                                                   std::span<safegraphs_custom_field_t const> fields) const noexcept {
        safegraphs_regular_shape_t const shape {nodes,          key_dtype_k,    value_column_dtype_k, dimensions(),
                                                dtype_t::u32_k, connectivity(), norms_column_dtype_()};
        return safegraphs_regular_header(shape, config.alignment, fields);
    }

    /** Writes the shard's rows and keys into @p target where @p header places them; the header
     *  itself comes last, so a cancelled save leaves none. */
    status_t write_monolith_(bytes_span_t target, safegraphs_header_t const& header,
                             save_config_t const& config) noexcept {
        std::size_t const nodes = header.nodes();
        index_t& one = *shards_[0].index;
        typename index_t::columns_t const columns {header.rows_of(target, header.value, 0, nodes),
                                                   header.rows_of(target, header.neighbors, 0, nodes),
                                                   header.rows_of(target, header.norms, 0, nodes)};
        if (status_t const saved = one.save(columns, config.synchronization); failed(saved)) return saved;
        bytes_span_t const key_rows = header.rows_of(target, header.key, 0, nodes);
        std::size_t written = 0;
        status_t wrote = status_t::success_k;
        shards_[0].keys.for_each_span({0, nodes}, [&](std::span<key_t const> run) noexcept {
            if (succeeded(wrote))
                wrote = copy_bytes(key_rows.subspan(written, run.size_bytes()), bytes_of(run.data(), run.size()));
            written += run.size_bytes();
        });
        return wrote;
    }

    status_t describe_part_(part_t& part) const noexcept {
        auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "metric") part.metric = safegraphs_name_t {field.value};
            if (field.key == "quantization") part.quantization = safegraphs_name_t {field.value};
        };
        auto const [header, status] = safegraphs_read_header(part.monolith, allocator_, per_custom_field);
        if (failed(status)) return status;
        if (header.monolith_bytes() > part.monolith.size()) return status_t::malformed_k;
        part.header = header;
        part.nodes = header.nodes();
        return status_t::success_k;
    }

    status_t catalog_monolith_(bytes_view_t monolith,
                               growing_buffer<part_t, forkunion_allocator_t>& parts) const noexcept {
        part_t part;
        part.monolith = monolith;
        if (status_t const described = describe_part_(part); failed(described)) return described;
        return parts.push_back(part);
    }

    /** Lists the parts at @p path: a lone monolith by its magic, or every part a manifest names,
     *  each mapped and described now. */
    status_t catalog_(std::string_view path, growing_buffer<mapped_file_t, forkunion_allocator_t>& mappings,
                      growing_buffer<part_t, forkunion_allocator_t>& parts) const noexcept {
        auto [mapping, mapping_status] = mapped_file_t::read(path);
        if (failed(mapping_status)) return mapping_status;
        bytes_view_t const checkpoint = mapping.view();
        if (checkpoint.size() >= safegraphs_prefix_t::magic_k.size() &&
            std::memcmp(checkpoint.data(), safegraphs_prefix_t::magic_k.data(), safegraphs_prefix_t::magic_k.size()) ==
                0) {
            if (status_t const listed = catalog_monolith_(checkpoint, parts); failed(listed)) return listed;
            return mappings.push_back(std::move(mapping));
        }
        growing_buffer<char, forkunion_allocator_t> paths_arena {allocator_};
        status_t kept = status_t::success_k;
        auto per_part = [&](safegraphs_part_t entry) noexcept {
            std::size_t const offset = paths_arena.size();
            for (char const character : entry.path)
                if (failed(paths_arena.push_back(character))) return kept = status_t::bad_alloc_k, void();
            part_t part;
            part.path = {offset, paths_arena.size()};
            if (failed(parts.push_back(part))) kept = status_t::bad_alloc_k;
        };
        auto const [summary, summary_status] = safegraphs_read_manifest(path, allocator_, per_part);
        if (failed(summary_status)) return summary_status;
        if (failed(kept)) return kept;
        if (parts.empty()) return status_t::malformed_k;
        if (failed(mappings.reserve_exactly(parts.size()))) return status_t::bad_alloc_k;
        growing_buffer<char, forkunion_allocator_t> part_path {allocator_};
        for (part_t& part : parts) {
            auto const [sibling, sibling_status] = safegraphs_sibling_path(
                part_path, path, to_string_view(paths_arena).substr(part.path.first, part.path.size()));
            if (failed(sibling_status)) return sibling_status;
            auto [part_mapping, part_mapping_status] = mapped_file_t::read(sibling);
            if (failed(part_mapping_status)) return part_mapping_status;
            part.monolith = part_mapping.view();
            if (status_t const described = describe_part_(part); failed(described)) return described;
            mappings.push_back_reserved(std::move(part_mapping));
        }
        return status_t::success_k;
    }

    status_t vetted_part_(part_t const& part, safegraphs_header_t const& first) const noexcept {
        safegraphs_header_t const& header = part.header;
        if (header.key.offsets || header.value.offsets || header.neighbors.offsets) return status_t::config_mismatch_k;
        if (header.key.dtype != key_dtype_k || header.value.dtype != value_column_dtype_k ||
            header.neighbors.dtype != dtype_t::u32_k)
            return status_t::config_mismatch_k;
        if (header.key.shape != column_shape_t {scalar_shape_t {}} || header.value.shape != first.value.shape ||
            header.neighbors.shape != first.neighbors.shape)
            return status_t::config_mismatch_k;
        if (!part.metric.empty() && nk_kernel_named(part.metric.characters, part.metric.view().size()) != kernel_)
            return status_t::config_mismatch_k;
        if (!part.quantization.empty() &&
            nk_dtype_named(part.quantization.characters, part.quantization.view().size()) != vector_dtype_)
            return status_t::config_mismatch_k;
        if (header.key.nodes() != part.nodes || header.value.nodes() != part.nodes ||
            header.neighbors.nodes() != part.nodes)
            return status_t::malformed_k;
        return status_t::success_k;
    }

    /** Vets @p parts, reserves for them, and lands them under the executor's hold and the insertion
     *  gate; a landing that fails or is cancelled publishes nothing and clears the shard. */
    template <typename progress_type_>
    status_t load_parts_(growing_buffer<part_t, forkunion_allocator_t> const& parts, load_config_t const& config,
                         executor_t& executor, progress_type_&& on_loaded) noexcept {
        if (parts.empty()) return status_t::malformed_k;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        if (!mutating_.lock_if_free()) return status_t::contended_k;
        status_t settled = status_t::success_k;
        {
            std::lock_guard const mutating {mutating_, std::adopt_lock};
            settled = land_parts_(parts, config);
        }
        if (succeeded(settled) && on_loaded(0) == continuation_t::cancel_k) settled = status_t::cancelled_k;
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = clear();
        return settled;
    }

    /** The body of @ref load_parts_ under its holds. */
    status_t land_parts_(growing_buffer<part_t, forkunion_allocator_t> const& parts,
                         load_config_t const& config) noexcept {
        safegraphs_header_t const& first = parts[0].header;
        std::size_t rows = 0;
        for (part_t const& part : parts) {
            if (status_t const vetted = vetted_part_(part, first); failed(vetted)) return vetted;
            rows += part.nodes;
        }
        connectivity_t const connectivity = static_cast<connectivity_t>(
            regular_width(first.neighbors.shape).value_or(0));
        if (connectivity != connectivity_16_k && connectivity != connectivity_32_k &&
            connectivity != connectivity_128_k)
            return status_t::malformed_k;
        std::size_t const dimensions = regular_width(first.value.shape).value_or(0);
        if (!dimensions) return status_t::malformed_k;
        if (capacity() && (dimensions != this->dimensions() || connectivity != this->connectivity()))
            return status_t::config_mismatch_k;
        if (!capacity() && !max_of(config.capacity, rows)) return status_t::success_k;
        if (status_t const reserved = reserve({max_of(config.capacity, rows), connectivity, dimensions}).status;
            failed(reserved))
            return reserved;

        shard_t& one = shards_[0];
        auto [plan, plan_status] = one.index->plan_load(rows);
        if (failed(plan_status)) return plan_status;
        std::size_t first_row = plan.base;
        for (part_t const& part : parts) {
            typename index_t::columns_view_t const columns {part.header.data_of(part.monolith, part.header.value),
                                                            part.header.data_of(part.monolith, part.header.neighbors),
                                                            part.header.norms_of(part.monolith, norms_column_dtype_())};
            std::size_t const bound_row = first_row + part.nodes;
            for (auto const batch : plan.steps()) {
                std::size_t const low = max_of<std::size_t>(batch.index_rows.first, first_row);
                std::size_t const high = min_of<std::size_t>(batch.index_rows.bound, bound_row);
                if (low >= high) continue;
                typename index_t::load_plan_t::step_t step;
                step.position = batch.position;
                step.index_rows = {static_cast<dense_node_id_t>(low), static_cast<dense_node_id_t>(high)};
                step.source_rows = {static_cast<dense_node_id_t>(low - first_row),
                                    static_cast<dense_node_id_t>(high - first_row)};
                if (status_t const landed = plan.load(step, columns, {0, static_cast<dense_node_id_t>(part.nodes)});
                    failed(landed))
                    return landed;
            }
            if (status_t const covered = one.keys.cover(bound_row); failed(covered)) return covered;
            auto [seats, seat_status] = locators_of_key_.claim(part.nodes);
            if (failed(seat_status)) return seat_status;
            bytes_view_t const part_keys = part.header.rows_of(part.monolith, part.header.key, 0, part.nodes);
            std::size_t read_bytes = 0;
            status_t read = status_t::success_k;
            one.keys.for_each_span({first_row, bound_row}, [&](std::span<key_t> run) noexcept {
                if (succeeded(read))
                    read = copy_bytes(bytes_of(run.data(), run.size()),
                                      part_keys.subspan(read_bytes, run.size_bytes()));
                read_bytes += run.size_bytes();
            });
            if (failed(read)) return read;
            for (std::size_t row = first_row; row != bound_row; ++row)
                locators_of_key_.insert(seats, one.keys[row], key_locator_t {0, static_cast<dense_node_id_t>(row)});
            first_row = bound_row;
        }
        return plan.finish();
    }

    nk_kernel_kind_t kernel_;
    growing_buffer<shard_t, forkunion_allocator_t> shards_;
    locators_map_t locators_of_key_;
    forkunion_allocator_t allocator_;
    fu::spin_mutex_t mutating_;
};

} // namespace usearch
} // namespace unum
