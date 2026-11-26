/**
 *  @file src/sequence/vram_variants_amd.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The AMD unit of the sequence kits: alignment with costs, banded overlap and containment
 *      need nothing past each class's wavefront and lane shifts, so one unit serves every AMD
 *      class, compiled once per class it is kept for.
 *
 *  HIP-Clang compiles it host-only, where it defines the faces' members reading the kits and names
 *  each family's code object, linked in beside it; HIP-Clang compiles every
 *  @c vram_kernels_amd_<family>.hip device-only into those code objects, once per class.
 */
#include "sequence_specializations.cuh"

/** The bounds of each family's code object, linked in as raw bytes, from the table CMake writes for
 *  the class this unit compiles as; the other classes' tables stay unreferenced. */
extern "C" util::byte_t const* const sequence_amd_cdna3_bundles[];
extern "C" util::byte_t const* const sequence_amd_cdna4_bundles[];
extern "C" util::byte_t const* const sequence_amd_cdna5_bundles[];
template <gpu_class_t gpu_class_>
util::byte_t const* const* sequence_bundles() noexcept {
    if constexpr (gpu_class_ == gpu_class_t::amd_cdna5_k) return sequence_amd_cdna5_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::amd_cdna4_k) return sequence_amd_cdna4_bundles;
    else return sequence_amd_cdna3_bundles;
}

/** A family's code object: its pair of bounds in the table, the families in the order of their
 *  metrics. */
template <usearch_sequence_metric_t metric_, gpu_class_t gpu_class_>
util::bytes_view_t sequence_amd_code_object<metric_, gpu_class_>::bytes() noexcept {
    util::byte_t const* const* const bounds = sequence_bundles<gpu_class_>();
    std::size_t const family = metric_ - usearch_sequence_metric_protein_alignment_k;
    return {bounds[2 * family], bounds[2 * family + 1]};
}

template struct sequence_amd_code_object<usearch_sequence_metric_protein_alignment_k, hip_class_k>;
template struct us::sequence::vram_amd_index<
    sequence_vram_kit<usearch_sequence_metric_protein_alignment_k, hip_class_k>, util::u32_t,
    sequence_amd_code_object<usearch_sequence_metric_protein_alignment_k, hip_class_k>>;
template struct us::sequence::vram_amd_kits<
    sequence_vram_kit<usearch_sequence_metric_protein_alignment_k, hip_class_k>, util::u32_t,
    sequence_amd_code_object<usearch_sequence_metric_protein_alignment_k, hip_class_k>>;

template struct sequence_amd_code_object<usearch_sequence_metric_dna_overlap_k, hip_class_k>;
template struct us::sequence::vram_amd_index<
    sequence_vram_kit<usearch_sequence_metric_dna_overlap_k, hip_class_k>, util::u32_t,
    sequence_amd_code_object<usearch_sequence_metric_dna_overlap_k, hip_class_k>>;
template struct us::sequence::vram_amd_kits<
    sequence_vram_kit<usearch_sequence_metric_dna_overlap_k, hip_class_k>, util::u32_t,
    sequence_amd_code_object<usearch_sequence_metric_dna_overlap_k, hip_class_k>>;

template struct sequence_amd_code_object<usearch_sequence_metric_utf8_containment_k, hip_class_k>;
template struct us::sequence::vram_amd_index<
    sequence_vram_kit<usearch_sequence_metric_utf8_containment_k, hip_class_k>, util::u32_t,
    sequence_amd_code_object<usearch_sequence_metric_utf8_containment_k, hip_class_k>>;
template struct us::sequence::vram_amd_kits<
    sequence_vram_kit<usearch_sequence_metric_utf8_containment_k, hip_class_k>, util::u32_t,
    sequence_amd_code_object<usearch_sequence_metric_utf8_containment_k, hip_class_k>>;

template struct basic_class_tables<sequence_traits_t, vendored_residency_t::vram_amd_k, hip_class_k>;
