/**
 *  @file include/usearch/sequence/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over byte sequences on NVIDIA GPUs: the basic VRAM index under one
 *      sequence kit of @c vram_nvidia_metric.cuh, Levenshtein over bytes or runes, overlap, or
 *      Needleman-Wunsch over packed codes.
 *
 *  Plain C++, so the host classes build with any compiler; the members reading the kit, which pick
 *  a launch's rung and size its spans, are defined beside the kits in the metric, which every unit
 *  naming a face includes.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <iterator> // `std::size` - the kit's rungs

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the parameters on the host
#include "usearch/basic/vram_nvidia_index.hpp"

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

/** The kit hooks a sequence graph on NVIDIA GPUs reads, and the fatbin @p kernels_type_ names once
 *  loaded: the sequence kit @p kit_type_ on its rungs, over node IDs of @p node_id_type_, with the
 *  parameters it scores with. */
template <typename kit_type_, typename node_id_type_, typename kernels_type_>
struct vram_nvidia_kits {
    using parameters_t = typename kit_type_::parameters_t;

    /** Rungs of the kit's kernels. */
    inline static constexpr std::size_t rungs_k = std::size(kit_type_::rungs_k);

    /**
     *  @brief Keeps the parameters the kit scores with, uploaded before the first walk. Defined in
     *      @c vram_nvidia_metric.cuh, as are the members below reading the kit.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k The host copy found no memory.
     */
    status_t parameters(parameters_t parameters) noexcept;

  private:
    template <typename, typename, typename, typename>
    friend struct basic::vram_ragged_graph;

    /** The kit's rung serving sequences of @p longest bytes under the kept parameters, or
     *  @c rungs_k past every one. */
    std::size_t rung_serving_(std::size_t longest) const noexcept;

    /** Lanes the kit scores a sequence with on @p rung. */
    static u32_t lanes_per_entry_(std::size_t rung) noexcept;

    /** Bytes the @p bytes long sequence at @p data takes once prepared, a multiple of 16. */
    std::size_t span_bytes_(byte_t const* data, std::size_t bytes) const noexcept;

    /** Bytes any loaded sequence of up to @p longest bytes takes once prepared, from length alone,
     *  or none where the device counts a load's spans. */
    std::size_t span_bytes_(std::size_t longest) const noexcept;

    /** Bytes the @p stored sequence saves into: its own, since a sequence is its bytes, as a RAM
     *  string. */
    static expected<std::size_t> saved_bytes_(bytes_view_t stored) noexcept {
        return {stored.size(), status_t::success_k};
    }

    /**
     *  Saves the @p stored sequence into @p saved as it is.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The two differ in size.
     */
    static status_t save_value_(bytes_view_t stored, bytes_span_t saved) noexcept { return copy_bytes(saved, stored); }

    /** Bytes a sequence of @p saved bytes stores into: its own. */
    static expected<std::size_t> stored_bytes_(std::size_t saved) noexcept { return {saved, status_t::success_k}; }

    /** Stores a step's sequences and sizes their spans as the kit's load hook queues them. */
    status_t queue_load_values_(basic::vram_load_step<node_id_type_> const& step) const noexcept;

    /** The fatbin holding the kits' kernels. */
    static bytes_view_t fatbin_() noexcept { return kernels_type_::bytes(); }

    /** Spells into @p symbol the kits' kernel @p instantiated on kit rung @p rung, as the fatbin
     *  exports it. */
    static void spell_kits_kernel_(vram_kernel_symbol_t& symbol, std::size_t rung,
                                   basic::cuda_rung_t instantiated) noexcept;

    /**
     *  @brief Uploads the parameters the kits read beside the sequences through @p api to whichever
     *      device is current.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The kit takes parameters, and none were set.
     */
    status_t upload_parameters_(cuda_api_t const& api) noexcept;

    /** The kit's parameters on the device, which every walk's kit reads; null for a kit taking
     *  none. */
    byte_t const* parameters_() const noexcept { return reinterpret_cast<byte_t const*>(device_parameters_.get()); }

    /** The kit's parameters as its host members read them, a span into @c host_parameters_ where
     *  they are one. */
    parameters_t kept_parameters_ {};

    /** The kit's parameters laid out as its kernels read them, on the host and, past the first
     *  @c reserve, on the device; empty for a kit taking none. */
    growing_buffer<u32_t, std::allocator<byte_t>> host_parameters_ {std::allocator<byte_t> {}};
    cuda_unique_ptr<u32_t> device_parameters_;
};

/** Similarity Search over byte sequences on NVIDIA GPUs scored by the sequence kit @p kit_type_ -
 *  Levenshtein over bytes or UTF-8 runes, window overlap, or Needleman-Wunsch - on its rungs. Node
 *  IDs are @p node_id_type_, and @p kernels_type_ names the fatbin. It owns a
 *  @c basic::vram_ragged_graph over its kits and forwards its verbs, types and constants, which
 *  that graph documents. */
template <typename kit_type_, typename node_id_type_, typename kernels_type_>
struct vram_nvidia_index {
    /* Components */
    using kits_t = vram_nvidia_kits<kit_type_, node_id_type_, kernels_type_>;
    using parameters_t = typename kits_t::parameters_t;
    using graph_t = basic::vram_ragged_graph<node_id_type_, kits_t>;
    using traversal_context_t = typename graph_t::traversal_context_t;

    /* IDs and Keys */
    using node_id_t = typename graph_t::node_id_t;
    using key_t = typename graph_t::key_t;
    using keyed_t = typename graph_t::keyed_t;
    using distance_t = typename graph_t::distance_t;
    using candidates_count_t = typename graph_t::candidates_count_t;

    /* Configs and Results */
    using reserve_capacity_t = typename graph_t::reserve_capacity_t;
    using insert_config_t = typename graph_t::insert_config_t;
    using insert_result_t = typename graph_t::insert_result_t;
    using insert_plan_t = typename graph_t::insert_plan_t;
    using search_config_t = typename graph_t::search_config_t;
    using search_result_t = typename graph_t::search_result_t;
    using refine_config_t = typename graph_t::refine_config_t;
    using split_config_t = typename graph_t::split_config_t;
    using refine_report_t = typename graph_t::refine_report_t;
    using reachability_t = typename graph_t::reachability_t;

    /* Plans */
    using refine_plan_t = typename graph_t::refine_plan_t;
    using save_plan_t = typename graph_t::save_plan_t;
    using load_plan_t = typename graph_t::load_plan_t;

    inline static constexpr std::size_t max_node_count_k = graph_t::max_node_count_k;
    inline static constexpr synchronization_t synchronization_ceiling_k = graph_t::synchronization_ceiling_k;

    /** Pins the index to the storage allocator's device; no sequence kind has a NumKong kind. */
    explicit vram_nvidia_index(cuda_virtual_memory_allocator<byte_t> const& allocator = {}) noexcept
        : graph_(nk_kernel_unknown_k, allocator) {}

    /**
     *  @brief Keeps the parameters the kit scores with, frozen once the first @c reserve has
     *      uploaded them to the device.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index is reserved already.
     *  - @c bad_alloc_k The host copy found no memory.
     */
    status_t parameters(parameters_t parameters) noexcept {
        if (graph_.connectivity()) return status_t::config_mismatch_k;
        return graph_.kits().parameters(parameters);
    }

    std::size_t size() const noexcept { return graph_.size(); }
    status_t status() const noexcept { return graph_.status(); }
    std::size_t connectivity() const noexcept { return graph_.connectivity(); }
    std::size_t capacity() const noexcept { return graph_.capacity(); }
    std::size_t bytes_capacity() const noexcept { return graph_.bytes_capacity(); }
    std::size_t memory_usage() const noexcept { return graph_.memory_usage(); }
    cuda_device_specs_t const& device_specs() const noexcept { return graph_.device_specs(); }
    cuda_api_t const& api() const noexcept { return graph_.api(); }
    nk_kernel_kind_t kernel() const noexcept { return graph_.kernel(); }
    basic::vram_ragged_storage<node_id_t> storage() const noexcept { return graph_.storage(); }
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept { return graph_.reserve(config); }

    std::size_t traversal_memory_usage(tape_view<byte_t const> tape, insert_config_t config = {}) const noexcept {
        return graph_.traversal_memory_usage(tape, config);
    }
    expected<insert_plan_t> plan_insert(std::size_t entries) noexcept { return graph_.plan_insert(entries); }
    expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        return graph_.insert(keys, tape, context, config);
    }
    std::size_t traversal_memory_usage(tape_view<byte_t const> queries, search_config_t config) const noexcept {
        return graph_.traversal_memory_usage(queries, config);
    }
    expected<search_result_t> search(tape_view<byte_t const> queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, search_config_t config,
                                     traversal_context_t& context) noexcept {
        return graph_.search(queries, result_ids, result_distances, config, context);
    }

    std::size_t traversal_memory_usage(refine_config_t config, std::size_t entries) const noexcept {
        return graph_.traversal_memory_usage(config, entries);
    }
    std::size_t traversal_memory_usage(refine_config_t config, std::size_t entries,
                                       std::size_t receivers) const noexcept {
        return graph_.traversal_memory_usage(config, entries, receivers);
    }
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, refine_config_t config,
                                        traversal_context_t& context) noexcept {
        return graph_.plan_refine(nodes, config, context);
    }
    expected<refine_report_t> refine(iota_view<node_id_t> nodes, refine_config_t config,
                                     traversal_context_t& context) noexcept {
        return graph_.refine(nodes, config, context);
    }
    expected<refine_plan_t> plan_absorb(std::span<vram_nvidia_index*> donors, refine_config_t config,
                                        traversal_context_t& context) noexcept {
        return graph_.plan_absorb(donors, &vram_nvidia_index::graph_, config, context);
    }
    expected<refine_plan_t> plan_split(std::span<vram_nvidia_index*> receivers, split_config_t config,
                                       traversal_context_t& context) noexcept {
        return graph_.plan_split(receivers, &vram_nvidia_index::graph_, config, context);
    }
    expected<reachability_t> health(traversal_context_t& context) noexcept { return graph_.health(context); }

    expected<basic::columns_extents_t> columns_extents(traversal_context_t& context) noexcept {
        return graph_.columns_extents(context);
    }
    expected<save_plan_t> plan_save(basic::columns<key_t, node_id_t, byte_t, u64_t> columns,
                                    traversal_context_t& context) noexcept {
        return graph_.plan_save(columns, context);
    }
    status_t save(basic::columns<key_t, node_id_t, byte_t, u64_t> columns, traversal_context_t& context) noexcept {
        return graph_.save(columns, context);
    }
    expected<load_plan_t> plan_load(safegraphs_saved_columns<node_id_t> const& saved, traversal_context_t& context,
                                    std::span<cuda_part_source_t const> sources = {}) noexcept {
        return graph_.plan_load(saved, context, sources);
    }
    status_t load(safegraphs_saved_columns<node_id_t> const& saved, traversal_context_t& context) noexcept {
        return graph_.load(saved, context);
    }

    keyed_t const& keys() const noexcept { return graph_.keys(); }
    expected<key_t> key_of(node_id_t node) const noexcept { return graph_.key_of(node); }
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        graph_.find_each(key, callback);
    }
    status_t neighbors(node_id_t first, std::span<node_id_t> rows) const noexcept {
        return graph_.neighbors(first, rows);
    }
    std::size_t count(key_t const& key) const noexcept { return graph_.count(key); }
    bool contains(key_t const& key) const noexcept { return graph_.contains(key); }

    /** Copies on @p context 's stream the values of the @p listed nodes back to back into
     *  @p outputs, their offsets into @p output_offsets from the base its first holds, as the
     *  graph's gather does. */
    status_t launch_gather_tape_(std::span<node_id_t const> listed, byte_t* outputs, std::span<u64_t> output_offsets,
                                 traversal_context_t const& context) noexcept {
        return graph_.launch_gather_tape_(listed, outputs, output_offsets, context);
    }

  private:

    /** A sharded index's shard table measures, closes and reopens the graph's claims. */
    template <typename, template <typename> class, typename>
    friend struct utilities::shard_set;

    /** A sharded index walks the graph below its snapshot and publishes its batches inside one
     *  commit. */
    template <typename index_type_, typename, typename, typename, typename>
        requires std::same_as<typename index_type_::node_id_t, u32_t>
    friend struct basic::vram_nvidia_indexes;

    std::size_t claimed_() const noexcept { return graph_.claimed_(); }
    status_t close_() noexcept { return graph_.close_(); }
    void reopen_() noexcept { graph_.reopen_(); }
    void retire_() noexcept { graph_.retire_(); }

    graph_t graph_;
};

} // namespace unum::usearch::sequence
