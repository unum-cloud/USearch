/**
 *  @file src/sequence/vram_variants_apple_metal7.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The Apple unit of the sequence kits: Levenshtein over bytes and UTF-8 runes, window
 *      overlap and Needleman-Wunsch need nothing past the Apple7 floor, so one unit serves every
 *      Apple class, compiling each kit's library from its embedded MSL at the first @c reserve.
 *
 *  It defines the faces the sequence variants over Apple GPUs name, and their tables.
 */
#include "sequence_specializations.cuh"

template struct us::sequence::vram_apple_index<us::sequence::apple_levenshtein_bytes_kit_t, u32_t>;
template struct us::sequence::vram_apple_kits<us::sequence::apple_levenshtein_bytes_kit_t, u32_t>;
template struct us::sequence::vram_apple_index<us::sequence::apple_levenshtein_utf8_kit_t, u32_t>;
template struct us::sequence::vram_apple_kits<us::sequence::apple_levenshtein_utf8_kit_t, u32_t>;
template struct us::sequence::vram_apple_index<us::sequence::apple_overlap_kit_t, u32_t>;
template struct us::sequence::vram_apple_kits<us::sequence::apple_overlap_kit_t, u32_t>;
template struct us::sequence::vram_apple_index<us::sequence::apple_needleman_wunsch_kit_t, u32_t>;
template struct us::sequence::vram_apple_kits<us::sequence::apple_needleman_wunsch_kit_t, u32_t>;

template struct basic_class_tables<sequence_traits_t, vendored_residency_t::vram_apple_k, gpu_class_t::apple_metal7_k>;
