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

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the window widths on the host
#include "usearch/basic/vram_nvidia_index.hpp"
#include "usearch/sequence/ram_metric.hpp" // `needleman_wunsch_costs_t` - the costs a face keeps

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

/** The distance a sequence kit scores. Its values are the C ABI's metric's, so @c src maps one
 *  onto the other with a cast. */
enum class vram_sequence_metric_t : u32_t {
    levenshtein_bytes_k = 1,
    levenshtein_runes_k,
    needleman_wunsch_k,
    overlap_k
};

/** The Itanium spelling of the sequence metric inside a sequence kernel's own scope, @c S1_, as
 *  @c vram_kernel_symbol documents. */
constexpr char const* kernel_enum_mangling(vram_sequence_metric_t) noexcept { return "NS1_22vram_sequence_metric_tE"; }

/** Myers words of the longest query each Levenshtein rung's kernels score, narrowest first: a warp
 *  walks each query up to sixteen words, 1 KiB, a block past them, up to 4 KiB, where a block's
 *  shared memory runs out. */
inline constexpr u32_t vram_levenshtein_words_k[] = {2, 4, 8, 16, 32, 64};

/** Bytes of the longest sequence @p words Myers words score, a bit of a 64-bit word per byte; a
 *  sequence of @c b bytes holds at most @c b runes, so it bounds rune sequences too. */
constexpr std::size_t vram_levenshtein_bytes(u32_t words) noexcept {
    return std::size_t {words} * std::numeric_limits<u64_t>::digits;
}

/** Bytes of the longest sequence the widest Levenshtein rung scores; longer ones are refused. */
inline constexpr std::size_t vram_levenshtein_widest_bytes_k = vram_levenshtein_bytes(
    vram_levenshtein_words_k[std::size(vram_levenshtein_words_k) - 1]);

/** Lanes each overlap rung splits a sequence's windows among, narrowest first: one lane alone, then
 *  every power of two from a block's warps up to the kit's warp. */
inline constexpr u32_t vram_overlap_rungs_k[] = {1, 4, 8, 16, 32};

/** Query codes each lane of a Needleman-Wunsch rung holds, narrowest first: a warp's stripe of
 *  rows is as tall as the lanes times these, and the widest rung repeats its stripe down a query
 *  as long as the widest Levenshtein sequence. */
inline constexpr u32_t vram_needleman_wunsch_rows_k[] = {1, 2, 3, 4, 5, 6, 7, 8};

/** The rungs the kit of @p metric scores on, narrowest first. */
constexpr std::span<u32_t const> vram_sequence_rungs(vram_sequence_metric_t metric) noexcept {
    if (metric == vram_sequence_metric_t::overlap_k) return vram_overlap_rungs_k;
    if (metric == vram_sequence_metric_t::needleman_wunsch_k) return vram_needleman_wunsch_rows_k;
    return vram_levenshtein_words_k;
}

/** The kit hooks a sequence graph on NVIDIA GPUs reads, and the fatbin @p kernels_type_ names once
 *  loaded: the kit of the sequence kind @p metric_ its ladder picks under @p capability_, on the
 *  rungs of @c vram_sequence_rungs, over node IDs of @p node_id_type_, with the window widths the
 *  overlap kind hashes or the costs the Needleman-Wunsch kind scores with. */
template <vram_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_, typename kernels_type_>
struct vram_nvidia_kits {

    /** Rungs of the kit's kernels, one per entry of @c vram_sequence_rungs. */
    inline static constexpr std::size_t rungs_k = vram_sequence_rungs(metric_).size();

    /**
     *  Keeps the window widths the overlap kind hashes. Defined in @c vram_nvidia_metric.cuh, as
     *  are the members below reading the kit.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The overlap kind gets no @p widths, another kind gets some, or a
     *    width is zero or past 32 bits.
     */
    status_t window_widths(std::span<std::size_t const> widths) noexcept;

    /**
     *  Keeps the folded costs the Needleman-Wunsch kind scores with.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The Needleman-Wunsch kind gets unfolded @p costs, or another kind
     *    gets folded ones.
     */
    status_t costs(needleman_wunsch_costs_t const& costs) noexcept;

  private:
    template <typename, typename, typename, typename>
    friend struct basic::vram_ragged_graph;

    /** The first rung whose kit serves sequences of @p longest bytes, or @c rungs_k past every one;
     *  for overlap, the widest whose lanes each still roll a widest window of such a sequence. */
    std::size_t rung_serving_(std::size_t longest) const noexcept;

    /** Lanes the kit scores a sequence with on @p rung. */
    static u32_t lanes_per_entry_(std::size_t rung) noexcept;

    /** Bytes the @p bytes long sequence at @p data takes once prepared, a multiple of 16. */
    std::size_t span_bytes_(byte_t const* data, std::size_t bytes) const noexcept;

    /** Bytes any loaded sequence of up to @p longest bytes takes once prepared, from length alone:
     *  overlap's exact span, Needleman-Wunsch's for the most codes those bytes pack, an upper bound
     *  tight for ASCII, and none for Levenshtein, whose spans the device counts. */
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

    /** Stores a step's sequences as saved, one copy of their run into the tape, then sizes their
     *  spans on the device where they follow from content: Levenshtein counts each sequence's
     *  classes, while the others size from length at planning. */
    status_t queue_load_values_(basic::vram_load_step<node_id_type_> const& step) const noexcept {
        if (status_t const copied = step.copy_values(); failed(copied)) return copied;
        if constexpr (metric_ == vram_sequence_metric_t::overlap_k ||
                      metric_ == vram_sequence_metric_t::needleman_wunsch_k)
            return status_t::success_k;
        else return step.launch_kits(vram_kernel_t::size_spans_k);
    }

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
     *  - @c uninitialized_k The kind takes parameters, and none were set.
     */
    status_t upload_parameters_(cuda_api_t const& api) noexcept;

    /** The folded costs the Needleman-Wunsch kind keeps among its parameters. */
    needleman_wunsch_costs_t const& costs_() const noexcept {
        return elements_of<needleman_wunsch_costs_t>(bytes_of(host_parameters_.data(), host_parameters_.size()))[0];
    }

    /** The kit's parameters on the device, which every walk's kit reads; null for the Levenshtein
     *  kinds. */
    byte_t const* parameters_() const noexcept { return reinterpret_cast<byte_t const*>(device_parameters_.get()); }

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
template <vram_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_, typename kernels_type_>
struct vram_nvidia_index {
    /* Components */
    using kits_t = vram_nvidia_kits<metric_, node_id_type_, capability_, kernels_type_>;
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
     *  Keeps the window widths the overlap kind hashes; the first @c reserve uploads them, so they
     *  freeze before it.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index is reserved already, the overlap kind gets no @p widths,
     *    another kind gets some, or a width is zero or past 32 bits.
     */
    status_t window_widths(std::span<std::size_t const> widths) noexcept {
        if (graph_.connectivity()) return status_t::config_mismatch_k;
        return graph_.kits().window_widths(widths);
    }

    /**
     *  Keeps the folded costs the Needleman-Wunsch kind scores with; the first @c reserve uploads
     *  them, so they freeze before it.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index is reserved already, the Needleman-Wunsch kind gets
     *    unfolded @p costs, or another kind gets folded ones.
     */
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
