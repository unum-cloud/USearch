/**
 *  @file src/sequence/vram_variants_nvidia.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The NVIDIA unit of the sequence kits: Levenshtein over bytes and UTF-8 runes and window
 *      overlap need nothing past the Ampere floor, so one unit carries a cubin for every requested
 *      NVIDIA class.
 *
 *  Compiled twice: by NVCC device-only, over every kind and rung with 32-bit node IDs, into the
 *  unit's fatbin, and by Clang host-only, where it defines the faces' members reading the kits and
 *  names that fatbin, linked in beside it, to the host indexes.
 */
#include "sequence_specializations.cuh"
#include "usearch/sequence/vram_nvidia_metric.cuh"

/** The unit's fatbin, linked in as raw bytes, which every kind's index loads. */
extern "C" byte_t const sequence_nvidia_bundle_start[], sequence_nvidia_bundle_end[];

bytes_view_t sequence_nvidia_fatbin_t::bytes() noexcept {
    return {sequence_nvidia_bundle_start,
            static_cast<std::size_t>(sequence_nvidia_bundle_end - sequence_nvidia_bundle_start)};
}

template struct us::sequence::vram_nvidia_index<usearch_sequence_metric_levenshtein_bytes_k, u32_t,
                                                sequence_nvidia_capabilities_k, sequence_nvidia_fatbin_t>;
template struct us::sequence::vram_nvidia_index<usearch_sequence_metric_levenshtein_utf8_k, u32_t,
                                                sequence_nvidia_capabilities_k, sequence_nvidia_fatbin_t>;
template struct us::sequence::vram_nvidia_index<usearch_sequence_metric_overlap_k, u32_t,
                                                sequence_nvidia_capabilities_k, sequence_nvidia_fatbin_t>;

template __device__ void
us::sequence::cuda_instantiate<sequence_nvidia_capabilities_k, usearch_sequence_metric_levenshtein_bytes_k, u32_t>();
template __device__ void
us::sequence::cuda_instantiate<sequence_nvidia_capabilities_k, usearch_sequence_metric_levenshtein_utf8_k, u32_t>();
template __device__ void
us::sequence::cuda_instantiate<sequence_nvidia_capabilities_k, usearch_sequence_metric_overlap_k, u32_t>();
