/**
 *  @file src/sequence/vram_variants_nvidia.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The NVIDIA unit of the sequence kits: Levenshtein over bytes and UTF-8 runes and window
 *      overlap need nothing past the Ampere floor, so one unit carries a cubin for every requested
 *      NVIDIA class.
 *
 *  Clang compiles it host-only, where it defines the faces' members reading the kits and names each
 *  kind's fatbin, linked in beside it; NVCC compiles every @c vram_kernels_nvidia_<kind>.cu
 *  device-only into those fatbins.
 */
#include "sequence_specializations.cuh"
#include "usearch/sequence/vram_nvidia_metric.cuh"

/** Each kind's fatbin, linked in as raw bytes beside the variants. */
extern "C" byte_t const sequence_nvidia_levenshtein_bytes_bundle_start[],
    sequence_nvidia_levenshtein_bytes_bundle_end[];
template <>
bytes_view_t sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_bytes_k>::bytes() noexcept {
    return {sequence_nvidia_levenshtein_bytes_bundle_start,
            static_cast<std::size_t>(sequence_nvidia_levenshtein_bytes_bundle_end -
                                     sequence_nvidia_levenshtein_bytes_bundle_start)};
}
template struct us::sequence::vram_nvidia_index<usearch_sequence_metric_levenshtein_bytes_k, u32_t,
                                                sequence_nvidia_capabilities_k,
                                                sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_bytes_k>>;
template struct us::sequence::vram_nvidia_kits<usearch_sequence_metric_levenshtein_bytes_k, u32_t,
                                               sequence_nvidia_capabilities_k,
                                               sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_bytes_k>>;

extern "C" byte_t const sequence_nvidia_levenshtein_utf8_bundle_start[], sequence_nvidia_levenshtein_utf8_bundle_end[];
template <>
bytes_view_t sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_utf8_k>::bytes() noexcept {
    return {sequence_nvidia_levenshtein_utf8_bundle_start,
            static_cast<std::size_t>(sequence_nvidia_levenshtein_utf8_bundle_end -
                                     sequence_nvidia_levenshtein_utf8_bundle_start)};
}
template struct us::sequence::vram_nvidia_index<usearch_sequence_metric_levenshtein_utf8_k, u32_t,
                                                sequence_nvidia_capabilities_k,
                                                sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_utf8_k>>;
template struct us::sequence::vram_nvidia_kits<usearch_sequence_metric_levenshtein_utf8_k, u32_t,
                                               sequence_nvidia_capabilities_k,
                                               sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_utf8_k>>;

extern "C" byte_t const sequence_nvidia_overlap_bundle_start[], sequence_nvidia_overlap_bundle_end[];
template <>
bytes_view_t sequence_nvidia_fatbin<usearch_sequence_metric_overlap_k>::bytes() noexcept {
    return {sequence_nvidia_overlap_bundle_start,
            static_cast<std::size_t>(sequence_nvidia_overlap_bundle_end - sequence_nvidia_overlap_bundle_start)};
}
template struct us::sequence::vram_nvidia_index<usearch_sequence_metric_overlap_k, u32_t,
                                                sequence_nvidia_capabilities_k,
                                                sequence_nvidia_fatbin<usearch_sequence_metric_overlap_k>>;
template struct us::sequence::vram_nvidia_kits<usearch_sequence_metric_overlap_k, u32_t, sequence_nvidia_capabilities_k,
                                               sequence_nvidia_fatbin<usearch_sequence_metric_overlap_k>>;
