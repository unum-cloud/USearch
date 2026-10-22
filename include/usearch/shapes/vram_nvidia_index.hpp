/**
 *  @file include/usearch/shapes/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over 3D point clouds on NVIDIA GPUs: the basic VRAM index under the
 *      Chamfer and Hausdorff kits @c shapes_ladder picks in @c vram_nvidia_metric.cuh.
 *
 *  Plain C++, so the host classes build with any compiler: the members reading the kits, which size
 *  a launch's spans, are defined beside the kits in the metric, which only the modality's NVIDIA
 *  unit compiles. Rows are NumKong packs end to end, as @c nk::shapes_pack lays them out, in the
 *  caller's tape and in the index.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span` - a batch's offsets and an absorb's donors

#include <numkong/capabilities.h> // `nk_kernel_chamfers_packed_k` & kin - the metric a kit scores under
#include <numkong/shapes.hpp>     // `nk::shapes_dtype` - the scalars a kit stores

#include "usearch/utilities/types.hpp"
#include "usearch/basic/vram_nvidia_index.hpp"

namespace unum::usearch::shapes {
using namespace unum::usearch::utilities;

/** The kit hooks a shapes graph on NVIDIA GPUs reads, and the fatbin @p kernels_type_ names once
 *  loaded: the Chamfer and Hausdorff kits @c shapes_ladder picks under @p capability_ for clouds of
 *  @p scalar_, over node IDs of @p node_id_type_. */
template <nk::shapes_dtype scalar_, typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
struct vram_nvidia_kits {

    /** Rungs of the kits' kernels: one, serving every cloud. */
    inline static constexpr std::size_t rungs_k = 1;

  private:
    template <typename, typename>
    friend struct basic::vram_ragged_graph;

    /** The rung serving rows of @p longest bytes, or @c rungs_k when none does. Defined in
     *  @c vram_nvidia_metric.cuh, as are the members below reading the kit. */
    static std::size_t rung_serving_(std::size_t longest) noexcept;

    /** Lanes the kit scores a row with on @p rung. */
    static u32_t lanes_per_row_(std::size_t rung) noexcept;

    /** Bytes the @p bytes long pack at @p data takes once prepared, a multiple of 16. */
    static std::size_t span_bytes_(byte_t const* data, std::size_t bytes) noexcept;

    /** Bytes the @p stored pack saves into: its cloud unpacked, @b [points,3] scalars, as the RAM
     *  codec saves a cloud. */
    static expected<std::size_t> saved_bytes_(bytes_view_t stored) noexcept;

    /** Unpacks the @p stored pack into @p saved. */
    static status_t save_value_(bytes_view_t stored, bytes_span_t saved) noexcept;

    /** Bytes the @p saved cloud stores into once packed; @c malformed_k unless it holds whole
     *  points. */
    static expected<std::size_t> stored_bytes_(bytes_view_t saved) noexcept;

    /** Packs the @p saved cloud into @p stored, as an insertion packs one on the host. */
    static status_t load_value_(bytes_view_t saved, bytes_span_t stored) noexcept;

    /** Loads the fatbin onto @p device and finds every kernel by its symbol. */
    status_t load_kernels_(int device) noexcept;

    cuda_function_handle_t kernel_function_(std::size_t rung, vram_kernel_t kernel, u32_t slots) const noexcept {
        return kernels_[rung][static_cast<u32_t>(kernel)][gpu_rung_index(slots)];
    }

    /** The kits read no parameters on the device. */
    byte_t const* parameters_() const noexcept { return nullptr; }

    /** The fatbin on this device, and its kernels in @c vram_kernel_t order. */
    cuda_module_t module_;
    cuda_function_handle_t kernels_[rungs_k][static_cast<u32_t>(vram_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};
};

/** Similarity Search over 3D point clouds of @p scalar_ on NVIDIA GPUs, under Chamfer or Hausdorff
 *  as the kernel kind it is built with names, scored by the kit @c shapes_ladder picks under
 *  @p capability_: a block walks each query, a warp scoring each candidate. Node IDs are
 *  @p node_id_type_, and @p kernels_type_ names the fatbin. It owns a @c basic::vram_ragged_graph
 *  over its kits and forwards its verbs, types and constants, which that graph documents. */
template <nk::shapes_dtype scalar_, typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
struct vram_nvidia_index {
    using kits_t = vram_nvidia_kits<scalar_, node_id_type_, capability_, kernels_type_>;
    using graph_t = basic::vram_ragged_graph<node_id_type_, kits_t>;
    using traversal_context_t = typename graph_t::traversal_context_t;
    using node_id_t = typename graph_t::node_id_t;
    using distance_t = typename graph_t::distance_t;
    using small_size_t = typename graph_t::small_size_t;
    using reserve_capacity_t = typename graph_t::reserve_capacity_t;
    using insert_config_t = typename graph_t::insert_config_t;
    using insert_result_t = typename graph_t::insert_result_t;
    using insert_plan_t = typename graph_t::insert_plan_t;
    using search_config_t = typename graph_t::search_config_t;
    using search_result_t = typename graph_t::search_result_t;
    using refine_config_t = typename graph_t::refine_config_t;
    using refine_result_t = typename graph_t::refine_result_t;
    using health_result_t = typename graph_t::health_result_t;
    using refine_plan_t = typename graph_t::refine_plan_t;
    using save_plan_t = typename graph_t::save_plan_t;
    using load_plan_t = typename graph_t::load_plan_t;

    inline static constexpr std::size_t max_node_count_k = graph_t::max_node_count_k;
    inline static constexpr synchronization_t synchronization_ceiling_k = graph_t::synchronization_ceiling_k;

    /** Pins the index to the storage allocator's device, scoring under @p kernel,
     *  @c nk_kernel_chamfers_packed_k or @c nk_kernel_hausdorffs_packed_k. */
    explicit vram_nvidia_index(nk_kernel_kind_t kernel,
                               cuda_virtual_memory_allocator<byte_t> const& allocator = {}) noexcept
        : graph_(kernel, allocator) {}

    std::size_t size() const noexcept { return graph_.size(); }
    status_t status() const noexcept { return graph_.status(); }
    std::size_t connectivity() const noexcept { return graph_.connectivity(); }
    std::size_t capacity() const noexcept { return graph_.capacity(); }
    std::size_t bytes_capacity() const noexcept { return graph_.bytes_capacity(); }
    std::size_t memory_usage() const noexcept { return graph_.memory_usage(); }
    cuda_device_specs_t const& device_specs() const noexcept { return graph_.device_specs(); }
    nk_kernel_kind_t kernel() const noexcept { return graph_.kernel(); }
    basic::vram_ragged_storage<node_id_t> storage() const noexcept { return graph_.storage(); }
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept { return graph_.reserve(config); }

    std::size_t traversal_memory_usage(tape_view<byte_t const> tape, insert_config_t config = {}) const noexcept {
        return graph_.traversal_memory_usage(tape, config);
    }
    expected<insert_plan_t> plan_insert(std::size_t rows) noexcept { return graph_.plan_insert(rows); }
    expected<insert_result_t> insert(tape_view<byte_t const> tape, traversal_context_t& context,
                                     insert_config_t config = {}) noexcept {
        return graph_.insert(tape, context, config);
    }
    std::size_t traversal_memory_usage(tape_view<byte_t const> queries, search_config_t config) const noexcept {
        return graph_.traversal_memory_usage(queries, config);
    }
    expected<search_result_t> search(tape_view<byte_t const> queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        return graph_.search(queries, result_ids, result_distances, context, config);
    }

    std::size_t traversal_memory_usage(refine_config_t config, std::size_t rows) const noexcept {
        return graph_.traversal_memory_usage(config, rows);
    }
    std::size_t traversal_memory_usage(refine_config_t config, std::size_t rows, std::size_t receivers) const noexcept {
        return graph_.traversal_memory_usage(config, rows, receivers);
    }
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                        refine_config_t config = {}) noexcept {
        return graph_.plan_refine(nodes, context, config);
    }
    expected<refine_result_t> refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                     refine_config_t config = {}) noexcept {
        return graph_.refine(nodes, context, config);
    }
    expected<refine_plan_t> plan_absorb(std::span<vram_nvidia_index*> donors, traversal_context_t& context,
                                        refine_config_t config = {}) noexcept {
        return graph_.plan_absorb(donors, &vram_nvidia_index::graph_, context, config);
    }
    expected<refine_plan_t> plan_split(std::span<vram_nvidia_index*> receivers, traversal_context_t& context,
                                       refine_config_t config = {}, u64_t seed = 0) noexcept {
        return graph_.plan_split(receivers, &vram_nvidia_index::graph_, context, config, seed);
    }
    expected<health_result_t> health(traversal_context_t& context) noexcept { return graph_.health(context); }

    expected<basic::columns_extents_t> columns_extents(traversal_context_t& context) noexcept {
        return graph_.columns_extents(context);
    }
    template <typename key_type_>
    expected<save_plan_t> plan_save(basic::columns<key_type_, node_id_t, byte_t, u64_t> columns,
                                    traversal_context_t& context) noexcept {
        return graph_.plan_save(columns, context);
    }
    template <typename key_type_>
    status_t save(basic::columns<key_type_, node_id_t, byte_t, u64_t> columns, traversal_context_t& context) noexcept {
        return graph_.save(columns, context);
    }
    template <typename key_type_>
    expected<load_plan_t> plan_load(basic::columns<key_type_ const, node_id_t const, byte_t const, u64_t const> columns,
                                    traversal_context_t& context) noexcept {
        return graph_.plan_load(columns, context);
    }
    template <typename key_type_>
    status_t load(basic::columns<key_type_ const, node_id_t const, byte_t const, u64_t const> columns,
                  traversal_context_t& context) noexcept {
        return graph_.load(columns, context);
    }

  private:

    /** A plural's shard table measures, closes and reopens the graph's claims. */
    template <typename, typename, typename, typename, template <typename> class, typename>
    friend struct utilities::shard_set;

    /** A plural walks the graph below its snapshot and publishes its batches inside one commit. */
    template <typename, typename, typename, typename, typename>
    friend struct basic::vram_nvidia_indexes;

    std::size_t claimed_() const noexcept { return graph_.claimed_(); }
    status_t close_() noexcept { return graph_.close_(); }
    void reopen_() noexcept { graph_.reopen_(); }

    graph_t graph_;
};

} // namespace unum::usearch::shapes
