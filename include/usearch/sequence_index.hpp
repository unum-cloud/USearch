/**
 *  @brief The sequence modality's flat entry point - variable-length sequences under
 *      alignment scores and sparse dot-products.
 *  @author Ash Vardanian
 *  @file include/usearch/sequence_index.hpp
 *  @date July 23, 2026
 *
 *  The modality covers Levenshtein distances, Needleman-Wunsch global and Smith-Waterman
 *  local alignment scoring, and sparse dot-products - StringZilla-backed measures over
 *  values whose lengths differ per entry, so none of the dense family's norm tricks
 *  apply. The implementations build on the baseline tier's caller-supplied-metric
 *  indexes; the @c sequence/ directory is born with the first per-residency file.
 */
#pragma once

#include "usearch/basic_index.hpp"

namespace unum {
namespace usearch {

/**
 *  @brief Sequence-search index for a single uniform latency memory space.
 *
 *  @tparam value_type_ is any variable-length sequence payload, like a byte or token span.
 *  @tparam metric_type_ is any callable scoring two sequences - edit distances and
 *      alignment scores alike.
 */
template <typename value_type_, typename metric_type_>
struct sequence_ram_index;

} // namespace usearch
} // namespace unum
