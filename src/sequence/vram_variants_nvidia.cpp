/**
 *  @file src/sequence/vram_variants_nvidia.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The NVIDIA unit of the sequence kits: Levenshtein over bytes and UTF-8 runes, window
 *      overlap and Needleman-Wunsch need nothing past the Ampere floor, so one unit serves every
 *      NVIDIA class, compiled as the Ampere floor or as a monoversioned build's one class.
 *
 *  Clang compiles it host-only, where it defines the faces' members reading the kits and names each
 *  kind's fatbin, linked in beside it; NVCC compiles every @c vram_kernels_nvidia_<kind>.cu
 *  device-only into those fatbins.
 */
#include "sequence_specializations.cuh"

/** The bounds of each kind's fatbin, linked in as raw bytes, from the table CMake writes for the
 *  class this unit compiles as; the other classes' tables stay unreferenced. */
extern "C" util::byte_t const* const sequence_nvidia_ampere_bundles[];
extern "C" util::byte_t const* const sequence_nvidia_ada_bundles[];
extern "C" util::byte_t const* const sequence_nvidia_hopper_bundles[];
extern "C" util::byte_t const* const sequence_nvidia_blackwell_bundles[];
extern "C" util::byte_t const* const sequence_nvidia_blackwellrtx_bundles[];
template <gpu_class_t gpu_class_>
util::byte_t const* const* sequence_bundles() noexcept {
    if constexpr (gpu_class_ == gpu_class_t::nvidia_ada_k) return sequence_nvidia_ada_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::nvidia_hopper_k) return sequence_nvidia_hopper_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::nvidia_blackwell_k) return sequence_nvidia_blackwell_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::nvidia_blackwellrtx_k) return sequence_nvidia_blackwellrtx_bundles;
    else return sequence_nvidia_ampere_bundles;
}

template <>
util::bytes_view_t sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_bytes_k>::bytes() noexcept {
    byte_t const* const* const bounds = sequence_bundles<cuda_class_k>();
    return {bounds[0], bounds[1]};
}
template struct us::sequence::vram_nvidia_index<sequence_nvidia_kit<usearch_sequence_metric_levenshtein_bytes_k>, u32_t,
                                                sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_bytes_k>>;
template struct us::sequence::vram_nvidia_kits<sequence_nvidia_kit<usearch_sequence_metric_levenshtein_bytes_k>, u32_t,
                                               sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_bytes_k>>;

template <>
util::bytes_view_t sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_utf8_k>::bytes() noexcept {
    byte_t const* const* const bounds = sequence_bundles<cuda_class_k>();
    return {bounds[2], bounds[3]};
}
template struct us::sequence::vram_nvidia_index<sequence_nvidia_kit<usearch_sequence_metric_levenshtein_utf8_k>, u32_t,
                                                sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_utf8_k>>;
template struct us::sequence::vram_nvidia_kits<sequence_nvidia_kit<usearch_sequence_metric_levenshtein_utf8_k>, u32_t,
                                               sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_utf8_k>>;

template <>
util::bytes_view_t sequence_nvidia_fatbin<usearch_sequence_metric_overlap_k>::bytes() noexcept {
    byte_t const* const* const bounds = sequence_bundles<cuda_class_k>();
    return {bounds[4], bounds[5]};
}
template struct us::sequence::vram_nvidia_index<sequence_nvidia_kit<usearch_sequence_metric_overlap_k>, u32_t,
                                                sequence_nvidia_fatbin<usearch_sequence_metric_overlap_k>>;
template struct us::sequence::vram_nvidia_kits<sequence_nvidia_kit<usearch_sequence_metric_overlap_k>, u32_t,
                                               sequence_nvidia_fatbin<usearch_sequence_metric_overlap_k>>;

template <>
util::bytes_view_t sequence_nvidia_fatbin<usearch_sequence_metric_needleman_wunsch_k>::bytes() noexcept {
    byte_t const* const* const bounds = sequence_bundles<cuda_class_k>();
    return {bounds[6], bounds[7]};
}
template struct us::sequence::vram_nvidia_index<sequence_nvidia_kit<usearch_sequence_metric_needleman_wunsch_k>, u32_t,
                                                sequence_nvidia_fatbin<usearch_sequence_metric_needleman_wunsch_k>>;
template struct us::sequence::vram_nvidia_kits<sequence_nvidia_kit<usearch_sequence_metric_needleman_wunsch_k>, u32_t,
                                               sequence_nvidia_fatbin<usearch_sequence_metric_needleman_wunsch_k>>;

template struct basic_class_tables<sequence_traits_t, vendored_residency_t::vram_nvidia_k, cuda_class_k>;
