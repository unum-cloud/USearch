/**
 *  @file src/sequence_kits.hpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The kit each sequence metric scores with on a GPU class, the device code each vendor's
 *      unit embeds for it, and the form each metric stores a sequence in.
 *
 *  The class units, the device kernel units and the tests name the same kits and device code
 *  through it, so none of them restates a ladder pick.
 */
#pragma once

#include <type_traits> // `std::conditional_t` - the kit a metric scores with

#include "usearch/sequence_index.h" // `usearch_sequence_metric_t`

#include "usearch/sequence/ram_metric.hpp" // `alignment_ladder`, `banded_overlap_ladder`, `containment_ladder`

#include "classes.hpp" // `gpu_class_t`, `stringzilla_capabilities_of`

/** The kit the metric @p metric_ scores with under the StringZilla tiers @p capability_, its
 *  ladder's pick. */
template <usearch_sequence_metric_t metric_, sz_capability_t capability_>
using sequence_kit =
    std::conditional_t<metric_ == usearch_sequence_metric_utf8_containment_k,
                       typename us::sequence::containment_ladder<capability_>::kit_t,
                       std::conditional_t<metric_ == usearch_sequence_metric_dna_overlap_k,
                                          typename us::sequence::banded_overlap_ladder<capability_>::kit_t,
                                          typename us::sequence::alignment_ladder<capability_>::kit_t>>;

/** The kit the metric @p metric_ scores with on a GPU of @p gpu_class_ ; the vendor's metric
 *  header supplies the rungs. */
template <usearch_sequence_metric_t metric_, gpu_class_t gpu_class_>
using sequence_vram_kit = sequence_kit<metric_, stringzilla_capabilities_of(gpu_class_)>;

/** One metric's sequence kits as a fatbin, defined in @c src/sequence/vram_variants_nvidia.cpp
 *  beside the bytes it embeds. */
template <usearch_sequence_metric_t metric_>
struct sequence_nvidia_fatbin {
    static util::bytes_view_t bytes() noexcept;
};

/** One metric's sequence kits as the code object of the AMD class @p gpu_class_, defined in
 *  @c src/sequence/vram_variants_amd.hip beside the bytes it embeds, for its class alone. */
template <usearch_sequence_metric_t metric_, gpu_class_t gpu_class_>
struct sequence_amd_code_object {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static util::bytes_view_t bytes() noexcept;
};

/** Bytes of the sketch a value of @p metric keeps in front of its sequence where its index names
 *  none, on every class and residency: the metric's measured default, none for a metric that
 *  builds by its canonical score. */
constexpr std::size_t default_sketch_bytes(usearch_sequence_metric_t metric) noexcept {
    switch (metric) {
    case usearch_sequence_metric_protein_alignment_k: return us::sequence::smooth_sketch_bytes_default_k;
    case usearch_sequence_metric_utf8_containment_k: return us::sequence::hash_sketch_bytes_default_k;
    default: return 0;
    }
}

/** Bytes a text of @p symbols bytes stores into under @p metric : packed class codes for a protein
 *  alignment, and a byte apiece otherwise. */
constexpr std::size_t stored_sequence_bytes(usearch_sequence_metric_t metric, std::size_t symbols) noexcept {
    return metric == usearch_sequence_metric_protein_alignment_k ? us::sequence::packed_codes_bytes(symbols) : symbols;
}

/** Writes @p text into @p sequence in the form @p metric stores: packed class codes of @p alphabet
 *  for a protein alignment, one class code a byte for a DNA overlap, and the folded bytes of a
 *  containment as they are. */
inline void store_sequence(usearch_sequence_metric_t metric, util::bytes_view_t text,
                           us::sequence::alignment_alphabet_t const& alphabet, util::bytes_span_t sequence) noexcept {
    if (metric == usearch_sequence_metric_protein_alignment_k) alphabet.pack(text, sequence);
    else if (metric == usearch_sequence_metric_dna_overlap_k) alphabet.encode(text, sequence);
    else if (!text.empty()) std::memcpy(sequence.data(), text.data(), text.size());
}
