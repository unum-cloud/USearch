/**
 *  @file include/usearch/sequence/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over byte sequences on NVIDIA GPUs: the basic VRAM index under the
 *      sequence kits @c vram_nvidia_metric.cuh picks, Levenshtein over bytes or runes, or overlap.
 *
 *  Plain C++, so the host classes build with any compiler: the face names its rungs here, and the
 *  members reading its kits, which pick a launch's rung and size its spans, are defined beside the
 *  kits in the metric, which only the modality's NVIDIA unit compiles.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span` - the rungs and the window widths

#include <stringzilla/capabilities.h> // `sz_capability_t` - the kits' capabilities

#include "usearch/sequence_index.h" // `usearch_sequence_metric_t` - the kind a face scores
#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the window widths on the host
#include "usearch/basic/vram_nvidia_index.hpp"

/** The C enum's spelling in a kernel's symbol, whole, as it sits in the global namespace, where
 *  @c vram_kernel_symbol finds it beside the enum. */
constexpr char const* itanium_name(usearch_sequence_metric_t) noexcept { return "25usearch_sequence_metric_t"; }

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

/** Myers words of the longest query each Levenshtein rung's kernels score, narrowest first: a warp
 *  walks each query up to sixteen words, 1 KiB, a block past them, up to 4 KiB. */
inline constexpr u32_t vram_levenshtein_words_k[] = {2, 4, 8, 16, 32, 64};

/** The lone overlap rung: a warp scores every row whatever its length, so nothing splits by it. */
inline constexpr u32_t vram_overlap_rungs_k[] = {0};

/** The rungs the kit of @p metric scores on, narrowest first. */
constexpr std::span<u32_t const> vram_sequence_rungs(usearch_sequence_metric_t metric) noexcept {
    if (metric == usearch_sequence_metric_overlap_k) return vram_overlap_rungs_k;
    return vram_levenshtein_words_k;
}

/** Similarity Search over byte sequences on NVIDIA GPUs under the sequence kind @p metric_ -
 *  Levenshtein over bytes or UTF-8 runes, or window overlap - scored by the kit its ladder picks
 *  under @p capability_ on the rungs of @c vram_sequence_rungs. Node IDs are @p node_id_type_, and
 *  @p kernels_type_ names the fatbin. */
template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
struct vram_nvidia_index
    : basic::vram_index<vram_nvidia_index<metric_, node_id_type_, capability_, kernels_type_>, node_id_type_> {

    using base_t = basic::vram_index<vram_nvidia_index, node_id_type_>;

    /** Rungs of the kit's kernels, one per entry of @c vram_sequence_rungs. */
    inline static constexpr std::size_t rungs_k = vram_sequence_rungs(metric_).size();

    /** Pins the index to the storage allocator's device; no sequence kind has a NumKong kind. */
    explicit vram_nvidia_index(cuda_virtual_memory_allocator<byte_t> const& allocator = {}) noexcept
        : base_t(nk_kernel_unknown_k, allocator) {}

    /** Keeps the window widths the overlap kind hashes, which every other kind refuses; the first
     *  @c reserve uploads them, so they freeze before it. Defined in @c vram_nvidia_metric.cuh. */
    status_t window_widths(std::span<std::size_t const> widths) noexcept;

  private:
    friend base_t;

    /** The first rung whose kit serves rows of @p longest bytes, or @c rungs_k past every one.
     *  Defined in @c vram_nvidia_metric.cuh, as are the members below reading the kit. */
    static std::size_t rung_serving_(std::size_t longest) noexcept;

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
     *  the window widths the kits read beside the rows. */
    status_t load_kernels_(int device) noexcept;

    cuda_function_handle_t kernel_function_(std::size_t rung, vram_kernel_t kernel, u32_t slots) const noexcept {
        return kernels_[rung][static_cast<u32_t>(kernel)][gpu_rung_index(slots)];
    }

    /** The fatbin on this device, and its kernels by rung, then in @c vram_kernel_t order. */
    cuda_module_t module_;
    cuda_function_handle_t kernels_[rungs_k][static_cast<u32_t>(vram_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};

    /** The overlap kind's window widths in bytes after their count, on the host and, past the
     *  first @c reserve, on the device, where every walk's kit reads them; empty for every other
     *  kind. */
    growing_buffer<u32_t, std::allocator<byte_t>> window_widths_ {std::allocator<byte_t> {}};
    cuda_unique_ptr<u32_t> device_widths_;
};

} // namespace unum::usearch::sequence
