/**
 *  @file include/usearch/geospatial/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over observations on NVIDIA GPUs: the basic VRAM index under the kit
 *      @c geospatial_ladder picks in @c vram_nvidia_metric.cuh, and the record of the kernel
 *      judging each walk's matches.
 *
 *  Plain C++, so the host classes build with any compiler: the members reading the kit, which pick
 *  a launch's rung and size its spans, are defined beside the kit in the metric, which only the
 *  modality's NVIDIA unit compiles, as is the judging kernel. Positions are the 32-byte
 *  @c earth_observation_t, in the caller's tape and in the index alike.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span` - a batch's offsets and an absorb's donors

#include <numkong/capabilities.h> // `nk_kernel_sqeuclideans3d_k` - the metric the kit scores under

#include "usearch/utilities/types.hpp"
#include "usearch/basic/vram_nvidia_index.hpp"
#include "usearch/geospatial/ram_index.hpp"

namespace unum::usearch::geospatial {
using namespace unum::usearch::utilities;

/** The kit hooks a geospatial graph on NVIDIA GPUs reads, and the fatbin @p kernels_type_ names
 *  once loaded: the squared chord the kit @c geospatial_ladder picks under @p capability_ scores,
 *  over node IDs of @p node_id_type_. */
template <typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
struct vram_nvidia_kits {

    /** Rungs of the kit's kernels: one, serving every observation. */
    inline static constexpr std::size_t rungs_k = 1;

  private:
    template <typename, typename, typename, typename>
    friend struct basic::vram_ragged_graph;

    /** The breadth an insert or a refine walks at least: the candidates the cone selector needs to
     *  see each cone's nearest candidate, as @c surface_keep_octants_t documents. */
    inline static constexpr u32_t build_breadth_k = 128;

    /** The rung serving positions of @p longest bytes, or @c rungs_k when none does. Defined in
     *  @c vram_nvidia_metric.cuh, as are the members below reading the kit. */
    static std::size_t rung_serving_(std::size_t longest) noexcept;

    /** Lanes the kit scores a position with on @p rung. */
    static u32_t lanes_per_entry_(std::size_t rung) noexcept;

    /** Bytes the position at @p data takes once prepared. */
    static std::size_t span_bytes_(byte_t const* data, std::size_t bytes) noexcept;

    /** Bytes a loaded observation takes once prepared: none, however long. */
    static std::size_t span_bytes_(std::size_t) noexcept { return 0; }

    /** Bytes the @p stored observation saves into: its own 32, as the RAM codec saves it. */
    static expected<std::size_t> saved_bytes_(bytes_view_t stored) noexcept {
        return {stored.size(), status_t::success_k};
    }

    /**
     *  Saves the @p stored observation into @p saved as it is.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The two differ in size.
     */
    static status_t save_value_(bytes_view_t stored, bytes_span_t saved) noexcept { return copy_bytes(saved, stored); }

    /** Bytes an observation of @p saved bytes stores into, its own; @c malformed_k unless whole. */
    static expected<std::size_t> stored_bytes_(std::size_t saved) noexcept {
        if (saved != sizeof(earth_observation_t)) return {0, status_t::malformed_k};
        return {saved, status_t::success_k};
    }

    /** Stores a step's observations as saved, one copy of their run into the tape; an observation
     *  prepares into no span. */
    status_t queue_load_values_(basic::vram_load_step<node_id_type_> const& step) const noexcept {
        return step.copy_values();
    }

    /** The fatbin holding the kits' kernels. */
    static bytes_view_t fatbin_() noexcept { return kernels_type_::bytes(); }

    /** Spells into @p symbol the kits' kernel @p instantiated on kit rung @p rung, as the fatbin
     *  exports it. */
    static void spell_kits_kernel_(vram_kernel_symbol_t& symbol, std::size_t rung,
                                   basic::cuda_rung_t instantiated) noexcept;

    /** The kits read no parameters on the device, so none upload. */
    static status_t upload_parameters_(cuda_api_t const&) noexcept { return status_t::success_k; }

    /** The kit reads no parameters on the device. */
    byte_t const* parameters_() const noexcept { return nullptr; }
};

template <typename node_id_type_>
struct cuda_admit_arguments;

/** Similarity Search over observations on NVIDIA GPUs, under the squared chord the kit
 *  @c geospatial_ladder picks under @p capability_ scores: a lane scores each candidate, a warp
 *  walks each query, and nodes keep their neighbors by tangent-plane cone. Node IDs are
 *  @p node_id_type_, and @p kernels_type_ names the fatbin. It owns a @c basic::vram_ragged_graph
 *  over its kits and forwards its verbs, types and constants, which that graph documents. */
template <typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
struct vram_nvidia_index {
    /* Components */
    using kits_t = vram_nvidia_kits<node_id_type_, capability_, kernels_type_>;
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

    /** Pins the index to the storage allocator's device. */
    explicit vram_nvidia_index(cuda_virtual_memory_allocator<byte_t> const& allocator = {}) noexcept
        : graph_(nk_kernel_sqeuclideans3d_k, allocator) {}

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
    std::size_t traversal_memory_usage(tape_view<byte_t const> queries, search_config_t config,
                                       std::size_t extra_bytes) const noexcept {
        return graph_.traversal_memory_usage(queries, config, extra_bytes);
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

    /**
     *  @brief Counts on @p context 's stream the candidates of every walk @p arguments judges that
     *      it admits, into its @c matched_counts, without a cap: a radius walk's first pass, over
     *      the candidates a @ref search of the call's breadth found.
     *
     *  @par Statuses:
     *  - @c unsupported_k The fatbin lacks the pass.
     */
    status_t launch_within_count_(cuda_admit_arguments<node_id_t> const& arguments,
                                  traversal_context_t const& context) noexcept {
        return launch_within_(vram_kernel_t::within_count_k, arguments, context);
    }

    /**
     *  @brief Writes on @p context 's stream the candidates of every walk @p arguments judges that
     *      it admits, as the first pass counted them.
     *
     *  They go back to back into @c matched_ids and @c matched_metres from each query's offset,
     *  filing the offsets past the base @c matched_offsets holds first.
     *
     *  @par Statuses:
     *  - @c unsupported_k The fatbin lacks the pass.
     */
    status_t launch_within_write_(cuda_admit_arguments<node_id_t> const& arguments,
                                  traversal_context_t const& context) noexcept {
        return launch_within_(vram_kernel_t::within_write_k, arguments, context);
    }

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

    /** Launches the radius walk's pass @p kernel over @p arguments on @p context 's stream, found
     *  in the kits' table there by symbol, a warp per query, oversubscribing every multiprocessor
     *  eight times at most. */
    status_t launch_within_(vram_kernel_t kernel, cuda_admit_arguments<node_id_t> const& arguments,
                            traversal_context_t const& context) noexcept {
        auto const [kits, kits_status] = graph_t::kits_kernels_(context);
        if (failed(kits_status)) return kits_status;
        vram_kernel_symbol_t symbol;
        vram_kernel_symbol(symbol, "geospatial", "cuda_within", {"cuda_admit_arguments", 1}, capability_,
                           itanium_type<node_id_type_> {}, kernel);
        cuda_function_handle_t const function = graph_t::kits_symbol_function_(*kits, symbol.text);
        if (!function) return status_t::unsupported_k;
        if (!arguments.queries_count) return status_t::success_k;
        u32_t const blocks = min_of<u32_t>(divide_round_up<u32_t>(arguments.queries_count, vram_warps_per_block_k),
                                           device_specs().multiprocessor_count * 8);
        return cuda_launch(api(), function, blocks, vram_warps_per_block_k * 32, 0, context.stream(),
                           cuda_overlap_t::serialized_k, arguments);
    }

    graph_t graph_;
};

/** One launch of @c cuda_admit, which keeps the first @c wanted entries of each walk's candidates
 *  that the relation admits, their sight lines clearing the terrain when one is given, ascending,
 *  beside their chords in metres and their count. */
template <typename node_id_type_>
struct cuda_admit_arguments {
    basic::vram_ragged_storage<node_id_type_> storage;

    /** The queries, @b [queries], and the candidates their walks found,
     *  @b [queries,candidates_count] IDs. */
    earth_observation_t const* queries = nullptr;
    node_id_type_ const* candidate_ids = nullptr;
    u32_t queries_count = 0;
    u32_t candidates_count = 0;
    u32_t wanted = 0;

    relation_t relation = relation_t::near_k;
    f64_t radius_metres = 0;
    f64_t slack_milliseconds = 0;

    /** The terrain sight lines clear, null heights for none: the row-major heights and their pack
     *  on the device, the centre of cell [0,0] and the steps between rows and columns in degrees,
     *  and the drop of a sight line below its chord per square metre along it, refraction folded
     *  in. */
    f32_t const* terrain_heights = nullptr;
    byte_t const* terrain_pack = nullptr;
    f64_t north_latitude_degrees = 0;
    f64_t west_longitude_degrees = 0;
    f64_t row_step_degrees = 0;
    f64_t column_step_degrees = 0;
    f64_t drop_scale = 0;

    /** The outputs, @b [queries,wanted] IDs and metres, and a count per query; a radius walk's
     *  run back to back instead, from the offsets one past each query, the base at the first. */
    node_id_type_* matched_ids = nullptr;
    f64_t* matched_metres = nullptr;
    u64_t* matched_counts = nullptr;
    u64_t* matched_offsets = nullptr;
};

} // namespace unum::usearch::geospatial
