/**
 *  @file src/basic/vram_variants_nvidia_sequence.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The NVIDIA unit of the sequence kits: Levenshtein over bytes and UTF-8 runes needs
 *      nothing past the Ampere floor, so one unit carries a cubin for every requested NVIDIA class.
 *
 *  Compiled twice: by NVCC device-only, over every kit, into the unit's fatbin, and by Clang
 *  host-only, where it names that fatbin, linked in beside it, to the host indexes.
 */
#include "usearch/sequence/vram_nvidia_index.hpp"
#include "usearch/basic/vram_nvidia_index.cuh"

namespace us = unum::usearch;

/** The Levenshtein kits, by the symbol kind and the Myers words their integers carry. */
template <us::utilities::u32_t kit_>
    requires(kit_ >> 8 <= sz_levenshtein_runes_k)
struct us::basic::cuda_kit_of<kit_> {
    using type = us::sequence::u8_ampere_levenshtein_kit<static_cast<sz_levenshtein_symbol_t>(kit_ >> 8), kit_ & 255u>;
};

/** The unit's fatbin, linked in as raw bytes, which the byte and the rune indexes both load. */
extern "C" us::utilities::byte_t const sequence_nvidia_bundle_start[], sequence_nvidia_bundle_end[];

template <>
us::utilities::bytes_view_t us::basic::cuda_fatbin<us::sequence::vram_nvidia_index<sz_levenshtein_bytes_k>>() noexcept {
    return {sequence_nvidia_bundle_start,
            static_cast<std::size_t>(sequence_nvidia_bundle_end - sequence_nvidia_bundle_start)};
}
template <>
us::utilities::bytes_view_t us::basic::cuda_fatbin<us::sequence::vram_nvidia_index<sz_levenshtein_runes_k>>() noexcept {
    return {sequence_nvidia_bundle_start,
            static_cast<std::size_t>(sequence_nvidia_bundle_end - sequence_nvidia_bundle_start)};
}

template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_bytes_k, 2>::kit_k>();
template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_bytes_k, 4>::kit_k>();
template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_bytes_k, 8>::kit_k>();
template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_bytes_k, 16>::kit_k>();
template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_bytes_k, 32>::kit_k>();
template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_bytes_k, 64>::kit_k>();
template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_runes_k, 2>::kit_k>();
template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_runes_k, 4>::kit_k>();
template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_runes_k, 8>::kit_k>();
template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_runes_k, 16>::kit_k>();
template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_runes_k, 32>::kit_k>();
template __device__ void
us::basic::cuda_instantiate<us::sequence::u8_ampere_levenshtein_kit<sz_levenshtein_runes_k, 64>::kit_k>();
