/**
 *  @file include/usearch/sequence/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over byte sequences on NVIDIA GPUs: the basic VRAM index under the
 *      sequence kits @c vram_nvidia_metric.cuh picks, Levenshtein over bytes or runes, overlap, or
 *      Needleman-Wunsch over packed codes.
 *
 *  Plain C++, so the host classes build with any compiler: the face names its rungs here, and the
 *  members reading its kits, which pick a launch's rung and size its spans, are defined beside the
 *  kits in the metric, which only the modality's NVIDIA unit compiles.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <iterator> // `std::size` - the widest Levenshtein rung
#include <limits>   // `std::numeric_limits` - the bytes a Myers word covers
#include <span>     // `std::span` - the rungs and the window widths

#include <stringzilla/capabilities.h> // `sz_capability_t` - the kits' capabilities

#include "usearch/sequence_index.h" // `usearch_sequence_metric_t` - the kind a face scores
#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the window widths on the host
#include "usearch/basic/vram_nvidia_index.hpp"
#include "usearch/sequence/ram_metric.hpp" // `needleman_wunsch_costs_t` - the costs a face keeps

/** The C enum's spelling in a kernel's symbol, whole, as it sits in the global namespace, where
 *  @c vram_kernel_symbol finds it beside the enum. */
constexpr char const* kernel_enum_mangling(usearch_sequence_metric_t) noexcept { return "25usearch_sequence_metric_t"; }

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

/** Myers words of the longest query each Levenshtein rung's kernels score, narrowest first: a warp
 *  walks each query up to sixteen words, 1 KiB, a block past them, up to 4 KiB, where a block's
 *  shared memory runs out. */
inline constexpr u32_t vram_levenshtein_words_k[] = {2, 4, 8, 16, 32, 64};

/** Bytes of the longest row @p words Myers words score, a bit of a 64-bit word per byte; a row
 *  of @c b bytes holds at most @c b runes, so it bounds rune rows too. */
constexpr std::size_t vram_levenshtein_bytes(u32_t words) noexcept {
    return std::size_t {words} * std::numeric_limits<u64_t>::digits;
}

/** Bytes of the longest row the widest Levenshtein rung scores; longer ones are refused. */
inline constexpr std::size_t vram_levenshtein_widest_bytes_k = vram_levenshtein_bytes(
    vram_levenshtein_words_k[std::size(vram_levenshtein_words_k) - 1]);

/** Lanes each overlap rung splits a row's windows among, narrowest first: one lane alone, then
 *  every power of two from a block's warps up to the kit's warp. */
inline constexpr u32_t vram_overlap_rungs_k[] = {1, 4, 8, 16, 32};

/** Query codes each lane of a Needleman-Wunsch rung holds, narrowest first: a warp's stripe of
 *  rows is as tall as the lanes times these, and the widest rung repeats its stripe down a query
 *  as long as the widest Levenshtein row. */
inline constexpr u32_t vram_needleman_wunsch_rows_k[] = {1, 2, 3, 4, 5, 6, 7, 8};

/** The rungs the kit of @p metric scores on, narrowest first. */
constexpr std::span<u32_t const> vram_sequence_rungs(usearch_sequence_metric_t metric) noexcept {
    if (metric == usearch_sequence_metric_overlap_k) return vram_overlap_rungs_k;
    if (metric == usearch_sequence_metric_needleman_wunsch_k) return vram_needleman_wunsch_rows_k;
    return vram_levenshtein_words_k;
}

/** The kit hooks a sequence graph on NVIDIA GPUs reads, and the fatbin @p kernels_type_ names once
 *  loaded: the kit of the sequence kind @p metric_ its ladder picks under @p capability_, on the
 *  rungs of @c vram_sequence_rungs, over node IDs of @p node_id_type_, with the window widths the
 *  overlap kind hashes or the costs the Needleman-Wunsch kind scores with. */
template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
struct vram_nvidia_kits {

    /** Rungs of the kit's kernels, one per entry of @c vram_sequence_rungs. */
    inline static constexpr std::size_t rungs_k = vram_sequence_rungs(metric_).size();

    /** Keeps the window widths the overlap kind hashes, which every other kind refuses. Defined in
     *  @c vram_nvidia_metric.cuh, as are the members below reading the kit. */
    status_t window_widths(std::span<std::size_t const> widths) noexcept;

    /** Keeps the folded costs the Needleman-Wunsch kind scores with, which every other kind refuses
     *  unless unfolded. */
    status_t costs(needleman_wunsch_costs_t const& costs) noexcept;

  private:
    template <typename, typename>
    friend struct basic::vram_ragged_graph;

    /** The first rung whose kit serves rows of @p longest bytes, or @c rungs_k past every one;
     *  for overlap, the widest whose lanes each still roll a widest window of such a row. */
    std::size_t rung_serving_(std::size_t longest) const noexcept;

    /** Lanes the kit scores a row with on @p rung. */
    static u32_t lanes_per_row_(std::size_t rung) noexcept;

    /** Bytes the @p bytes long row at @p data takes once prepared, a multiple of 16. */
    std::size_t span_bytes_(byte_t const* data, std::size_t bytes) const noexcept;

    /** Bytes the @p stored row saves into: its own, since a row is its bytes, as a RAM string. */
    static expected<std::size_t> saved_bytes_(bytes_view_t stored) noexcept {
        return {stored.size(), status_t::success_k};
    }

    /** Saves the @p stored row into @p saved as it is. */
    static status_t save_value_(bytes_view_t stored, bytes_span_t saved) noexcept { return copy_bytes(saved, stored); }

    /** Bytes the @p saved row stores into: its own. */
    static expected<std::size_t> stored_bytes_(bytes_view_t saved) noexcept {
        return {saved.size(), status_t::success_k};
    }

    /** Stores the @p saved row into @p stored as it is. */
    static status_t load_value_(bytes_view_t saved, bytes_span_t stored) noexcept { return copy_bytes(stored, saved); }

    /** Loads the fatbin onto @p device, finds every kernel of every rung by its symbol, and uploads
     *  the parameters the kits read beside the rows. */
    status_t load_kernels_(int device) noexcept;

    cuda_function_handle_t kernel_function_(std::size_t rung, vram_kernel_t kernel, u32_t slots) const noexcept {
        return kernels_[rung][static_cast<u32_t>(kernel)][gpu_rung_index(slots)];
    }

    /** The folded costs the Needleman-Wunsch kind keeps among its parameters. */
    needleman_wunsch_costs_t const& costs_() const noexcept {
        return elements_of<needleman_wunsch_costs_t>(bytes_of(host_parameters_.data(), host_parameters_.size()))[0];
    }

    /** The kit's parameters on the device, which every walk's kit reads; null for the Levenshtein
     *  kinds. */
    byte_t const* parameters_() const noexcept { return reinterpret_cast<byte_t const*>(device_parameters_.get()); }

    /** The fatbin on this device, and its kernels by rung, then in @c vram_kernel_t order. */
    cuda_module_t module_;
    cuda_function_handle_t kernels_[rungs_k][static_cast<u32_t>(vram_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};

    /** The kit's parameters on the host and, past the first @c reserve, on the device, where every
     *  walk's kit reads them: the overlap kind's window widths in bytes after their count, or the
     *  Needleman-Wunsch kind's folded costs; empty for the Levenshtein kinds. */
    growing_buffer<u32_t, std::allocator<byte_t>> host_parameters_ {std::allocator<byte_t> {}};
    cuda_unique_ptr<u32_t> device_parameters_;
};

/** Similarity Search over byte sequences on NVIDIA GPUs under the sequence kind @p metric_ -
 *  Levenshtein over bytes or UTF-8 runes, window overlap, or Needleman-Wunsch - scored by the kit
 *  its ladder picks under @p capability_ on the rungs of @c vram_sequence_rungs. Node IDs are
 *  @p node_id_type_, and @p kernels_type_ names the fatbin. It owns a @c basic::vram_ragged_graph
 *  over its kits and forwards its verbs, types and constants, which that graph documents. */
template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
struct vram_nvidia_index {
    using kits_t = vram_nvidia_kits<metric_, node_id_type_, capability_, kernels_type_>;
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

    /** Pins the index to the storage allocator's device; no sequence kind has a NumKong kind. */
    explicit vram_nvidia_index(cuda_virtual_memory_allocator<byte_t> const& allocator = {}) noexcept
        : graph_(nk_kernel_unknown_k, allocator) {}

    /** Keeps the window widths the overlap kind hashes, which every other kind refuses; the first
     *  @c reserve uploads them, so they freeze before it. */
    status_t window_widths(std::span<std::size_t const> widths) noexcept {
        if (graph_.connectivity()) return status_t::config_mismatch_k;
        return graph_.kits().window_widths(widths);
    }

    /** Keeps the folded costs the Needleman-Wunsch kind scores with, which every other kind refuses
     *  unless unfolded; the first @c reserve uploads them, so they freeze before it. */
    status_t costs(needleman_wunsch_costs_t const& costs) noexcept {
        if (graph_.connectivity()) return status_t::config_mismatch_k;
        return graph_.kits().costs(costs);
    }

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
    template <typename index_type_, typename, typename, typename, typename, typename>
        requires std::same_as<typename index_type_::node_id_t, utilities::node_id_t>
    friend struct basic::vram_nvidia_indexes;

    std::size_t claimed_() const noexcept { return graph_.claimed_(); }
    status_t close_() noexcept { return graph_.close_(); }
    void reopen_() noexcept { graph_.reopen_(); }
    void retire_() noexcept { graph_.retire_(); }

    graph_t graph_;
};

} // namespace unum::usearch::sequence
