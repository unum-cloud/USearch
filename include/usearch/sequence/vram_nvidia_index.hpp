/**
 *  @file include/usearch/sequence/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over byte sequences on NVIDIA GPUs: the basic VRAM index under the
 *      Levenshtein kits @c levenshtein_ladder picks in @c vram_nvidia_metric.cuh.
 *
 *  Plain C++, so the host classes build with any compiler: the face names its rungs here, and the
 *  members reading its kits, which pick a launch's rung and size its spans, are defined beside the
 *  kits in the metric, which only the modality's NVIDIA unit compiles.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <iterator> // `std::size` - the rungs

#include <stringzilla/capabilities.h> // `sz_capability_t` - the kits' capabilities
#include <stringzilla/types.h>        // `sz_levenshtein_symbol_t` - bytes or runes

#include "usearch/utilities/types.hpp"
#include "usearch/basic/vram_nvidia_index.hpp"

/** The C enum's spelling in a kernel's symbol, whole, as it sits in the global namespace, where
 *  @c vram_kernel_symbol finds it beside the enum. */
constexpr char const* itanium_name(sz_levenshtein_symbol_t) noexcept { return "23sz_levenshtein_symbol_t"; }

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

/** Myers words of the longest query each rung's kernels score, narrowest first: a warp walks each
 *  query up to sixteen words, 1 KiB, a block past them, up to 4 KiB. */
inline constexpr u32_t vram_levenshtein_words_k[] = {2, 4, 8, 16, 32, 64};

/** Similarity Search over byte sequences on NVIDIA GPUs under the Levenshtein distance over
 *  @p symbol_, bytes or UTF-8 runes, scored by the kit @c levenshtein_ladder picks under
 *  @p capability_ on the rungs of @c vram_levenshtein_words_k. Node IDs are @p node_id_type_, and
 *  @p kernels_type_ names the fatbin. */
template <sz_levenshtein_symbol_t symbol_, typename node_id_type_, sz_capability_t capability_, typename kernels_type_>
struct vram_nvidia_index
    : basic::vram_index<vram_nvidia_index<symbol_, node_id_type_, capability_, kernels_type_>, node_id_type_> {

    using base_t = basic::vram_index<vram_nvidia_index, node_id_type_>;

    /** Rungs of the kits' kernels, one per entry of @c vram_levenshtein_words_k. */
    inline static constexpr std::size_t rungs_k = std::size(vram_levenshtein_words_k);

    /** Pins the index to the storage allocator's device; Levenshtein has no NumKong kernel kind. */
    explicit vram_nvidia_index(cuda_virtual_memory_allocator<byte_t> const& allocator = {}) noexcept
        : base_t(nk_kernel_unknown_k, allocator) {}

  private:
    friend base_t;

    /** The first rung whose kit serves rows of @p longest bytes, or @c rungs_k past every one.
     *  Defined in @c vram_nvidia_metric.cuh, as are the members below reading the kit. */
    static std::size_t rung_serving_(std::size_t longest) noexcept;

    /** Lanes the kit scores a row with on @p rung. */
    static u32_t lanes_per_row_(std::size_t rung) noexcept;

    /** Bytes the @p bytes long row at @p data takes once prepared, a multiple of 16. */
    static std::size_t span_bytes_(byte_t const* data, std::size_t bytes) noexcept;

    /** Loads the fatbin onto @p device and finds every kernel of every rung by its symbol. */
    status_t load_kernels_(int device) noexcept;

    cuda_function_handle_t kernel_function_(std::size_t rung, vram_kernel_t kernel, u32_t slots) const noexcept {
        return kernels_[rung][static_cast<u32_t>(kernel)][gpu_rung_index(slots)];
    }

    /** The fatbin on this device, and its kernels by rung, then in @c vram_kernel_t order. */
    cuda_module_t module_;
    cuda_function_handle_t kernels_[rungs_k][static_cast<u32_t>(vram_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};
};

} // namespace unum::usearch::sequence
