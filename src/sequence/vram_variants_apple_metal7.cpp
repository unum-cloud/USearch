/**
 *  @file src/sequence/vram_variants_apple_metal7.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The Apple unit of the sequence kits: alignment with costs, banded overlap and
 *      containment need nothing past the Apple7 floor, so one unit serves every Apple class,
 *      compiling each kit's library from its embedded MSL at its first @c reserve call.
 *
 *  It defines the faces the sequence variants over Apple GPUs name, and their tables.
 */
#include "sequence_specializations.cuh"

template struct us::sequence::vram_apple_index<us::sequence::apple_packed_alignment_kit_t, util::u32_t>;
template struct us::sequence::vram_apple_kits<us::sequence::apple_packed_alignment_kit_t, util::u32_t>;
template struct us::sequence::vram_apple_index<us::sequence::apple_u8_banded_overlap_kit_t, util::u32_t>;
template struct us::sequence::vram_apple_kits<us::sequence::apple_u8_banded_overlap_kit_t, util::u32_t>;
template struct us::sequence::vram_apple_index<us::sequence::apple_utf8_containment_kit_t, util::u32_t>;
template struct us::sequence::vram_apple_kits<us::sequence::apple_utf8_containment_kit_t, util::u32_t>;

template struct basic_class_tables<sequence_traits_t, vendored_residency_t::vram_apple_k, gpu_class_t::apple_metal7_k>;
