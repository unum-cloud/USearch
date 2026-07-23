/**
 *  @brief The trampoline of the sequence router library: runtime dispatch over the
 *      variable-length sequence indexes.
 *  @author Ash Vardanian
 *  @file cpp/sequence_indexes.cpp
 *  @date July 17, 2026
 *
 *  The sequence modality builds on the baseline tier: StringZilla-backed similarity
 *  measures - Levenshtein, Needleman-Wunsch, Smith-Waterman - over arbitrary-length
 *  values through @c basic_volatile_index, with the per-dtype shards joining as the
 *  scoring kits land.
 */
#include <usearch/basic_index.hpp>

namespace us = unum::usearch;
namespace sz = ashvardanian::stringzilla;
