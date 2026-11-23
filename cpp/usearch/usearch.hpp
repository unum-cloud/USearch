/**
 *  @file cpp/usearch/usearch.hpp
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief The umbrella of the USearch C++ interface: every modality header, the Apache Arrow
 *      converters and the vocabulary they share, behind one include.
 *
 *  A caller using one modality may include its header alone; this one names the whole interface
 *  once, for a caller using several.
 */
#pragma once

#include "usearch/types.hpp" // `status_t`, `expected`, `deferred`, `awaiter`, `executor_t`, `arena_t`

#include "usearch/arrow.hpp"            // `keys_view_from_arrow`, `rows_view_from_arrow`, `tape_view_from_arrow`
#include "usearch/dense_index.hpp"      // `dense_index_t`, `dense_index_on`
#include "usearch/geospatial_index.hpp" // `geospatial_index_t`, `geospatial_index_on`
#include "usearch/shapes_index.hpp"     // `shapes_index_t`, `shapes_index_on`
#include "usearch/sequence_index.hpp"   // `sequence_index_t`, `sequence_index_on`
