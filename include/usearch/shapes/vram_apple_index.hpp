/**
 *  @file include/usearch/shapes/vram_apple_index.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Similarity Search over 3D point clouds on Apple GPUs: the basic Apple graph under the
 *      Chamfer and Hausdorff kit of @c vram_apple_metric.metal, over f16 packs.
 *
 *  Plain C++ for a unit building with Metal, as every Apple face: the kit is MSL the graph compiles
 *  at its first @c reserve from NumKong's Metal shapes source and the kit, so the host members
 *  reading it live here. Shapes are NumKong packs end to end, as @c nk::shapes_pack lays them out,
 *  in the caller's tape and in the index; a load packs its saved clouds on the host, straight into
 *  the unified tape. f16 packs score in f32 math; NumKong ships no f32 Metal body meeting CUDA's
 *  f32 guarantee, so a face over f32 clouds refuses its first @c reserve with @c unsupported_k.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span>        // `std::span` - a batch's offsets and a reshard's sources and targets
#include <type_traits> // `std::is_same_v` - the one scalar Apple GPUs score

#include <numkong/capabilities.h> // `nk_kernel_chamfers_packed_k` & kin - the metric the kit scores under
#include <numkong/shapes.hpp>     // `nk::a_shapes_dtype` - the scalars a kit stores
#include <numkong/shapes/metal.h> // `nk_shapes_source_metal_` - the simdgroup bodies the kit calls

#include "usearch/utilities/types.hpp"
#include "usearch/shapes/ram_metric.hpp" // `shapes_ladder` - the serial kit packing a load's clouds
#include "usearch/shapes/ram_index.hpp"  // `codec` - a pack's cloud in and out
#include "usearch/basic/vram_apple_index.hpp"

namespace unum::usearch::shapes {
using namespace unum::usearch::utilities;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The shapes kit, spliced after NumKong's shapes source. */
inline constexpr char metal_shapes_metric_source_k[] = {
#embed "vram_apple_metric.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#pragma region Face

/** The kit hooks a shapes graph on Apple GPUs reads: Chamfer and Hausdorff between packs of
 *  @p scalar_ clouds, over node IDs of @p node_id_type_. Only f16 packs score; the face refuses
 *  others before any of these runs. */
template <nk::a_shapes_dtype scalar_, typename node_id_type_>
struct vram_apple_kits {

    /** Rungs of the kit's kernels: one, serving every cloud. */
    inline static constexpr std::size_t rungs_k = 1;

    inline static constexpr char metal_kit_name_k[] = "shapes::metal_f16_shapes_kit_t";
    inline static constexpr char const* metal_sources_k[] = {nk_shapes_source_metal_, metal_shapes_metric_source_k};

  private:
    template <typename, typename, typename, typename>
    friend struct basic::vram_apple_graph;

    /** The serial kit a pack's cloud goes in and out through. */
    using pack_kit_t = typename shapes_ladder<scalar_, nk_cap_serial_k>::kit_t;

    /** The one rung serves every cloud. */
    static std::size_t rung_serving_(std::size_t) noexcept { return 0; }

    /** A simdgroup scores a shape, its lanes splitting the outer cloud's points. */
    static u32_t lanes_per_entry_(std::size_t) noexcept { return metal_simdgroup_width_k; }

    /** A query is its own pack where it lies, so it prepares into no span. */
    static std::size_t span_bytes_(byte_t const*, std::size_t) noexcept { return 0; }
    static std::size_t span_bytes_(std::size_t) noexcept { return 0; }

    /** Bytes the @p stored pack saves into: its cloud unpacked, @b [points,3] scalars, as the RAM
     *  codec saves a cloud. */
    static expected<std::size_t> saved_bytes_(bytes_view_t stored) noexcept {
        nk_size_t points = 0;
        if (status_t const read = status_of(
                pack_kit_t::packed_shape(stored.data(), &points, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
            failed(read))
            return {0, read};
        return {points * 3 * sizeof(typename scalar_::raw_t), status_code_t::success_k};
    }

    /**
     *  @brief Unpacks the @p stored pack into @p saved.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p saved bytes hold other than the pack's points.
     */
    static status_t save_value_(bytes_view_t stored, bytes_span_t saved) noexcept {
        return codec<pack_kit_t>::save_pack(stored, saved);
    }

    /** Bytes a cloud of @p saved bytes stores into once packed; @c malformed_k unless it holds
     *  whole points. */
    static expected<std::size_t> stored_bytes_(std::size_t saved) noexcept {
        constexpr std::size_t point_bytes_k = 3 * sizeof(typename scalar_::raw_t);
        if (saved % point_bytes_k) return {0, status_code_t::malformed_k};
        nk_size_t bytes = 0;
        if (status_t const sized = status_of(pack_kit_t::pack_size(saved / point_bytes_k, &bytes)); failed(sized))
            return {0, sized};
        return {bytes, status_code_t::success_k};
    }

    /** Packs a step's clouds on the host, straight into their planned starts in the unified tape:
     *  the one kit whose stored bytes differ from the saved ones; spans take nothing. */
    status_t queue_load_values_(basic::metal_load_step<node_id_type_> const& step) const noexcept {
        constexpr std::size_t scalar_bytes_k = sizeof(typename scalar_::raw_t);
        u64_t const first_saved = step.saved_offsets[0], first_stored = step.stored_offsets[0];
        for (std::size_t node = 0; node != step.arguments.count; ++node) {
            bytes_view_t const pose {step.saved.data() + (step.saved_offsets[node] - first_saved) * scalar_bytes_k,
                                     (step.saved_offsets[node + 1] - step.saved_offsets[node]) * scalar_bytes_k};
            bytes_span_t const pack {step.stored + (step.stored_offsets[node] - first_stored),
                                     step.stored_offsets[node + 1] - step.stored_offsets[node]};
            if (status_t const packed = codec<pack_kit_t>::load_pack(pose, pack); failed(packed)) return packed;
        }
        return status_code_t::success_k;
    }

    /** The kit reads no parameters, so none upload. */
    static status_t upload_parameters_(metal_context_t&) noexcept { return status_code_t::success_k; }

    /** The kit reads no parameters; the graph binds the tape in their slot. */
    metal_buffer_t const& parameters_() const noexcept { return none_; }
    metal_buffer_t none_;
};

/** Similarity Search over 3D point clouds of @p scalar_ on Apple GPUs, under Chamfer or Hausdorff
 *  as the kernel kind it is built with names: a threadgroup walks each query, a simdgroup scoring
 *  each candidate. Node IDs are @p node_id_type_. It owns a @c basic::vram_apple_graph over its
 *  kits and forwards its verbs, types and constants, which that graph documents. */
template <nk::a_shapes_dtype scalar_, typename node_id_type_>
struct vram_apple_index {
    /* Components */
    using kits_t = vram_apple_kits<scalar_, node_id_type_>;
    using graph_t = basic::vram_apple_graph<node_id_type_, kits_t>;
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

    inline static constexpr std::size_t max_node_count_k = graph_t::max_node_count_k;
    inline static constexpr synchronization_t synchronization_ceiling_k = graph_t::synchronization_ceiling_k;

    /** Pins the index to the allocator's device, scoring under @p kernel,
     *  @c nk_kernel_chamfers_packed_k or @c nk_kernel_hausdorffs_packed_k. */
    explicit vram_apple_index(nk_kernel_kind_t kernel, metal_shared_allocator_t const& allocator = {}) noexcept
        : graph_(kernel, allocator) {}

    /**
     *  @brief Reserves the graph, refusing clouds of any scalar but f16.
     *
     *  @par Statuses:
     *  - @c unsupported_k The clouds are f32, which no Metal body scores to CUDA's f32 guarantee.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if constexpr (!std::is_same_v<scalar_, nk::f16_t>) return {0, status_code_t::unsupported_k};
        else return graph_.reserve(config);
    }

    std::size_t size() const noexcept { return graph_.size(); }
    status_t status() const noexcept { return graph_.status(); }
    std::size_t connectivity() const noexcept { return graph_.connectivity(); }
    std::size_t capacity() const noexcept { return graph_.capacity(); }
    std::size_t bytes_capacity() const noexcept { return graph_.bytes_capacity(); }
    std::size_t memory_usage() const noexcept { return graph_.memory_usage(); }
    metal_device_specs_t const& device_specs() const noexcept { return graph_.device_specs(); }
    metal_context_t* context() const noexcept { return graph_.context(); }
    nk_kernel_kind_t kernel() const noexcept { return graph_.kernel(); }
    basic::metal_ragged_storage<node_id_t> storage() const noexcept { return graph_.storage(); }

    /** The scratch a call of these arguments needs ahead of time: none, as a Metal context grows
     *  its own buffers; spelled for every verb the NVIDIA faces size. */
    template <typename... usage_types_>
    std::size_t traversal_memory_usage(usage_types_ const&... usage) const noexcept {
        return graph_.traversal_memory_usage(usage...);
    }

    expected<insert_plan_t> plan_insert(std::size_t entries) noexcept { return graph_.plan_insert(entries); }
    expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        return graph_.insert(keys, tape, context, config);
    }
    expected<search_result_t> search(tape_view<byte_t const> queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, search_config_t config,
                                     traversal_context_t& context) noexcept {
        return graph_.search(queries, result_ids, result_distances, config, context);
    }

    expected<refine_plan_t> plan_refine(refine_config_t config, traversal_context_t& context) noexcept {
        return graph_.plan_refine(config, context);
    }
    expected<refine_report_t> refine(refine_config_t config, traversal_context_t& context) noexcept {
        return graph_.refine(config, context);
    }
    expected<reshard_plan_t> plan_reshard(std::span<vram_apple_index* const> sources,
                                          std::span<vram_apple_index* const> targets, reshard_config_t config,
                                          traversal_context_t& context) noexcept {
        return graph_.plan_reshard(sources, targets, &vram_apple_index::graph_, config, context);
    }
    expected<refine_report_t> reshard(std::span<vram_apple_index* const> sources,
                                      std::span<vram_apple_index* const> targets, reshard_config_t config,
                                      traversal_context_t& context) noexcept {
        return graph_.reshard(sources, targets, &vram_apple_index::graph_, config, context);
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
    expected<load_plan_t> plan_load(safegraphs_saved_columns<node_id_t> const& saved,
                                    traversal_context_t& context) noexcept {
        return graph_.plan_load(saved, context);
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

    /** Copies the values of the @p listed nodes back to back into @p outputs, their offsets into
     *  @p output_offsets from the base its first holds, on the host over unified memory. */
    status_t gather_tape_(std::span<node_id_t const> listed, byte_t* outputs,
                          std::span<u64_t> output_offsets) const noexcept {
        return graph_.gather_tape_(listed, outputs, output_offsets);
    }

  private:

    /** A sharded index's shard table measures, closes and reopens the graph's claims. */
    template <typename, template <typename> class atomic_ref_template_, typename, typename>
    friend struct util::shard_set;

    /** A sharded index walks the graph below its snapshot and publishes its batches inside one
     *  commit. */
    template <typename index_type_, typename, typename, typename, typename>
        requires std::same_as<typename index_type_::node_id_t, u32_t>
    friend struct basic::vram_apple_indexes;

    std::size_t claimed_() const noexcept { return graph_.claimed_(); }
    status_t close_() noexcept { return graph_.close_(); }
    void reopen_() noexcept { graph_.reopen_(); }
    void retire_() noexcept { graph_.retire_(); }

    graph_t graph_;
};

#pragma endregion Face

} // namespace unum::usearch::shapes
