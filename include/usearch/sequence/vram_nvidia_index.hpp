/**
 *  @file include/usearch/sequence/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over byte sequences on NVIDIA GPUs: the basic VRAM index over the byte
 *      Levenshtein kits of @c vram_nvidia_metric.cuh.
 *
 *  Plain C++, so the host classes build with any compiler: the kits' device members compile only
 *  into the fatbin, while their host facts pick a launch's kit and size its spans.
 */
#pragma once

#include "usearch/basic/vram_nvidia_index.hpp"
#include "usearch/sequence/vram_nvidia_metric.cuh"

namespace unum::usearch::sequence {

/** Similarity Search over byte sequences on NVIDIA GPUs under the Levenshtein distance over
 *  @p symbol_, bytes or UTF-8 runes, for rows of up to 4 KiB: a warp walks each query up to 1 KiB,
 *  a block past them. */
template <sz_levenshtein_symbol_t symbol_ = sz_levenshtein_bytes_k>
using vram_nvidia_index =
    basic::vram_nvidia_index<u8_ampere_levenshtein_kit<symbol_, 2>, u8_ampere_levenshtein_kit<symbol_, 4>,
                             u8_ampere_levenshtein_kit<symbol_, 8>, u8_ampere_levenshtein_kit<symbol_, 16>,
                             u8_ampere_levenshtein_kit<symbol_, 32>, u8_ampere_levenshtein_kit<symbol_, 64>>;

} // namespace unum::usearch::sequence
