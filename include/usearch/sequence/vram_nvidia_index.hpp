/**
 *  @file include/usearch/sequence/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over byte sequences on NVIDIA GPUs: the basic VRAM index under one
 *      sequence kit of @c vram_nvidia_metric.cuh, alignment over packed codes, banded overlap,
 *      or containment over folded UTF-8.
 *
 *  Every node's value is one run of the tape: the kit's sketch of the bytes its face set, none for
 *  a kit building canonically, the sequence, and padding to a word that closes with its own count.
 *  Inserts and refinements score the sketches of a sketching kit and the sequences of any other,
 *  searches score the sequences canonically, and a checkpoint holds the sequences alone, so a load
 *  derives every sketch on the device.
 *
 *  Plain C++, so the host classes build with any compiler; the members reading the kit, which pick
 *  a launch's rung and size its spans, are defined beside the kits in the metric, which every unit
 *  naming a face includes.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <concepts> // `std::same_as` - a sharded index over 32-bit IDs
#include <iterator> // `std::size` - the kit's rungs

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the parameters on the host
#include "usearch/sequence/ram_index.hpp"    // `vram_stored_bytes`, `vram_stored_sequence` - a tape value's layout
#include "usearch/basic/vram_nvidia_index.hpp"

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

/** The kit hooks a sequence graph on NVIDIA GPUs reads, and the fatbin @p kernels_type_ names once
 *  loaded: the sequence kit @p kit_type_ on its rungs, over node IDs of @p node_id_type_, with the
 *  parameters it scores with. */
template <typename kit_type_, typename node_id_type_, typename kernels_type_>
struct vram_nvidia_kits {
    using canonical_parameters_t = typename kit_type_::canonical_parameters_t;
    using sketch_tables_t = typename kit_type_::sketch_tables_t;

    /** Rungs of the kit's canonical kernels, and the first of them its builders compile on: one
     *  under a sketching kit, as no length sizes a sketch, and every one otherwise. */
    inline static constexpr std::size_t canonical_rungs_count_k = std::size(kit_type_::canonical_rungs_k);
    inline static constexpr std::size_t build_rungs_count_k = a_device_sketching_kit<kit_type_>
                                                                  ? 1
                                                                  : canonical_rungs_count_k;

    /** Blocks of the kit's builders resident per multiprocessor. */
    inline static constexpr u32_t build_resident_blocks_k = basic::cuda_build_resident_blocks<kit_type_>;

    /**
     *  @brief Keeps the parameters the kit scores with, uploaded before the first walk. Defined in
     *      @c vram_nvidia_metric.cuh, as are the members below reading the kit.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The host copy found no memory.
     */
    status_t canonical_parameters(canonical_parameters_t parameters) noexcept;

    /**
     *  @brief Keeps the kit's sketch tables, which the first @c reserve uploads to the device,
     *      nothing for a kit reading none, and the @p sketch_bytes of every stored sketch.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The bytes are not a positive multiple of four for a sketching kit, or
     *    the tables hold fewer columns; or they are not zero for a kit writing no sketch.
     *  - @c out_of_memory_k The host copy found no memory.
     */
    status_t sketch_tables(typename kit_type_::sketch_tables_t const& tables, std::size_t sketch_bytes) noexcept;

    /** Bytes of the sketch in front of every stored sequence and of a query's; zero for a kit
     *  writing none. */
    std::size_t sketch_bytes() const noexcept { return sketch_bytes_; }

  private:
    template <typename, typename, typename, typename>
    friend struct basic::vram_ragged_graph;

    /** The kit's rung serving @p role over values of @p longest bytes under the kept parameters, or
     *  @c canonical_rungs_count_k past every one: by the value scored, but a sketch builder on the
     *  first, whatever the stored values' lengths. */
    std::size_t rung_serving_(score_role_t role, std::size_t longest) const noexcept;

    /** How a block of the kit's kernels of @p role splits with @p walkers walkers on its rung
     *  @p rung : a builder takes a walker per warp alone. */
    static basic::cuda_block_layout_t block_layout_(score_role_t role, std::size_t rung, u32_t walkers) noexcept;

    /** Bytes the @p bytes long query at @p data takes once prepared for @p role, a multiple of 16:
     *  none for a sketch builder, which reads the stored sketches in place. */
    std::size_t span_bytes_(score_role_t role, byte_t const* data, std::size_t bytes) const noexcept;

    /** Bytes the @p stored value saves into: its sequence's, as a RAM index saves it. */
    expected<std::size_t> saved_bytes_(bytes_view_t stored) const noexcept {
        return {vram_stored_sequence(stored, sketch_bytes_).size(), status_code_t::success_k};
    }

    /**
     *  @brief Saves the sequence of the @p stored value into @p saved as it is.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The two differ in size.
     */
    status_t save_value_(bytes_view_t stored, bytes_span_t saved) const noexcept {
        return copy_bytes(saved, vram_stored_sequence(stored, sketch_bytes_));
    }

    /** Bytes a sequence of @p saved bytes stores into: behind its sketch, closed to a word. */
    expected<std::size_t> stored_bytes_(std::size_t saved) const noexcept {
        return {vram_stored_bytes(saved, sketch_bytes_), status_code_t::success_k};
    }

    /** Shared bytes each warp of a load or sketch launch carves for the kept sketch bytes, which
     *  the graph's first @c reserve holds against the device's widest block grant. */
    std::size_t warp_scratch_bytes_() const noexcept { return kit_type_::sketch_scratch_bytes(sketch_bytes_); }

    /** A load scatters each saved sequence behind its sketch, which the same kernel derives from
     *  the tables uploaded with the parameters, where the kit has either. */
    status_t queue_load_values_(basic::vram_load_step<node_id_type_> step) const noexcept {
        step.arguments.load.sketch_bytes = sketch_bytes_;
        step.arguments.load.tables = device_sketch_tables_.get();
        step.warp_scratch_bytes = static_cast<u32_t>(warp_scratch_bytes_());
        return step.launch_kits(vram_kernel_t::load_values_k);
    }

    /** Queues the sketches of an insert's staged values or a search's queries, from the tables
     *  uploaded with the parameters. */
    status_t queue_sketches_(basic::vram_sketch_step<node_id_type_> step) const noexcept
        requires a_device_sketching_kit<kit_type_>
    {
        step.arguments.sketch.sketch_bytes = sketch_bytes_;
        step.arguments.sketch.tables = device_sketch_tables_.get();
        step.warp_scratch_bytes = static_cast<u32_t>(warp_scratch_bytes_());
        return step.launch_kits(vram_kernel_t::sketch_values_k);
    }

    /** The fatbin holding the kits' kernels. */
    static bytes_view_t fatbin_() noexcept { return kernels_type_::bytes(); }

    /** Spells into @p symbol the kits' kernel @p instantiated on kit rung @p rung, as the fatbin
     *  exports it. */
    static void spell_kits_kernel_(vram_kernel_symbol_t& symbol, std::size_t rung,
                                   basic::cuda_rung_t instantiated) noexcept;

    /**
     *  @brief Uploads the parameters and the sketch tables the kits read beside the sequences
     *      through @p api to whichever device is current.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The kit reads sketch tables or parameters, and one was not set.
     */
    status_t upload_parameters_(cuda_api_t const& api) noexcept;

    /** The kit's parameters on the device, which every walk's kit reads; null for a kit taking
     *  none. */
    byte_t const* parameters_() const noexcept {
        return reinterpret_cast<byte_t const*>(device_canonical_parameters_.get());
    }

    /** The kit's parameters as its host members read them. */
    canonical_parameters_t kept_canonical_parameters_ {};

    /** The kit's parameters laid out as its kernels read them, on the host and, past the first
     *  @c reserve, on the device; empty for a kit taking none. */
    growing_buffer<u32_t, std::allocator<byte_t>> host_canonical_parameters_ {std::allocator<byte_t> {}};
    cuda_unique_ptr<u32_t> device_canonical_parameters_;

    /** The kit's sketch tables as @ref sketch_tables laid them out, and on the device past the
     *  first @c reserve, where the sketch phases of an insert, a search and a load read them; empty
     *  and null for a kit reading none. */
    growing_buffer<byte_t, std::allocator<byte_t>> host_sketch_tables_ {std::allocator<byte_t> {}};
    cuda_unique_ptr<byte_t> device_sketch_tables_;

    /** Bytes of every stored sketch, as @ref sketch_tables set them. */
    u32_t sketch_bytes_ = 0;
};

/** Similarity Search over byte sequences on NVIDIA GPUs scored by the sequence kit @p kit_type_ -
 *  alignment, banded overlap, or containment - on its rungs. Node IDs are @p node_id_type_, and @p
 *  kernels_type_ names the fatbin. It owns a @c basic::vram_ragged_graph over its kits and forwards
 *  its verbs, types and constants, which that graph documents. Every value of its tape is the kit's
 *  sketch, the sequence and the closing padding, so @ref insert takes a tape laid out in that
 *  stored form, its sketches left for the device. */
template <typename kit_type_, typename node_id_type_, typename kernels_type_>
struct vram_nvidia_index {
    /* Components */
    using kits_t = vram_nvidia_kits<kit_type_, node_id_type_, kernels_type_>;
    using canonical_parameters_t = typename kits_t::canonical_parameters_t;
    using sketch_tables_t = typename kits_t::sketch_tables_t;
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
    using reshard_config_t = typename graph_t::reshard_config_t;
    using refine_report_t = typename graph_t::refine_report_t;
    using reachability_t = typename graph_t::reachability_t;

    /* Plans */
    using refine_plan_t = typename graph_t::refine_plan_t;
    using reshard_plan_t = typename graph_t::reshard_plan_t;
    using save_plan_t = typename graph_t::save_plan_t;
    using load_plan_t = typename graph_t::load_plan_t;
    using links_view_t = typename graph_t::links_view_t;

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
     *  - @c out_of_memory_k The host copy found no memory.
     */
    status_t canonical_parameters(canonical_parameters_t parameters) noexcept {
        if (graph_.connectivity()) return status_code_t::config_mismatch_k;
        return graph_.kits().canonical_parameters(parameters);
    }

    /**
     *  @brief Keeps the tables the kit's sketches read and the @p sketch_bytes of every stored
     *      sketch, frozen once the first @c reserve has uploaded them to the device.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index is reserved already, or the kits refuse the bytes.
     *  - @c out_of_memory_k The host copy found no memory.
     */
    status_t sketch_tables(typename kit_type_::sketch_tables_t const& tables, std::size_t sketch_bytes) noexcept {
        if (graph_.connectivity()) return status_code_t::config_mismatch_k;
        return graph_.kits().sketch_tables(tables, sketch_bytes);
    }

    /** Bytes of the sketch in front of every stored sequence; zero for a kit writing none. */
    std::size_t sketch_bytes() const noexcept { return graph_.kits().sketch_bytes(); }

    std::size_t size() const noexcept { return graph_.size(); }
    status_t status() const noexcept { return graph_.status(); }
    std::size_t connectivity() const noexcept { return graph_.connectivity(); }
    std::size_t capacity() const noexcept { return graph_.capacity(); }
    std::size_t bytes_capacity() const noexcept { return graph_.bytes_capacity(); }

    std::size_t memory_usage() const noexcept { return graph_.memory_usage(); }
    cuda_device_specs_t const& device_specs() const noexcept { return graph_.device_specs(); }
    cuda_api_t const& api() const noexcept { return graph_.api(); }
    cuda_extensions_t const& extensions() const noexcept { return graph_.extensions(); }
    nk_kernel_kind_t kernel() const noexcept { return graph_.kernel(); }
    basic::vram_ragged_storage<node_id_t> storage() const noexcept { return graph_.storage(); }
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept { return graph_.reserve(config); }

    std::size_t traversal_memory_usage(tape_view<byte_t const> tape, insert_config_t config = {}) const noexcept {
        return graph_.traversal_memory_usage(tape, config);
    }
    expected<insert_plan_t> plan_insert(std::size_t entries) noexcept { return graph_.plan_insert(entries); }

    /** Inserts a tape of stored values, refusing with @c config_mismatch_k one holding a value no
     *  @ref vram_stored_bytes sizes: bare sequences, or values closed otherwise. */
    expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        if (!vram_tape_is_stored(tape, sketch_bytes()))
            return {{}, {status_code_t::config_mismatch_k, "Tape not laid out behind sketches"}};
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
                                       std::size_t targets) const noexcept {
        return graph_.traversal_memory_usage(config, entries, targets);
    }
    expected<refine_plan_t> plan_refine(refine_config_t config, traversal_context_t& context) noexcept {
        return graph_.plan_refine(config, context);
    }
    expected<refine_report_t> refine(refine_config_t config, traversal_context_t& context) noexcept {
        return graph_.refine(config, context);
    }

    expected<reshard_plan_t> plan_reshard(std::span<vram_nvidia_index* const> sources,
                                          std::span<vram_nvidia_index* const> targets, reshard_config_t config,
                                          traversal_context_t& context) noexcept {
        return graph_.plan_reshard(sources, targets, &vram_nvidia_index::graph_, config, context);
    }
    expected<refine_report_t> reshard(std::span<vram_nvidia_index* const> sources,
                                      std::span<vram_nvidia_index* const> targets, reshard_config_t config,
                                      traversal_context_t& context) noexcept {
        return graph_.reshard(sources, targets, &vram_nvidia_index::graph_, config, context);
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
    status_t adopt_links(links_view_t links, traversal_context_t& context) noexcept {
        return graph_.adopt_links(links, context);
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
    template <typename, template <typename> class shards_atomic_ref_template_, typename, typename>
    friend struct util::shard_set;

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
