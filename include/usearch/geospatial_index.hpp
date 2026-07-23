/**
 *  @brief The geospatial modality's flat entry point - low-dimensional representations in
 *      space & time under haversine, vincenty, and trajectory similarities.
 *  @author Ash Vardanian
 *  @file include/usearch/geospatial_index.hpp
 *  @date July 23, 2026
 *
 *  The modality covers point queries on the sphere and the ellipsoid - haversine and
 *  vincenty distances - and trajectory similarities over time-ordered tracks, where the
 *  entry is a short sequence of coordinates rather than one equidimensional vector. The
 *  implementations build on the baseline tier's caller-supplied-metric indexes; the
 *  @c geospatial/ directory is born with the first per-residency file.
 */
#pragma once

#include "usearch/basic_index.hpp"

namespace unum {
namespace usearch {

/**
 *  @brief Geospatial-search index for a single uniform latency memory space.
 *
 *  @tparam value_type_ is any coordinate payload - a point in space & time, or a short
 *      trajectory of them.
 *  @tparam metric_type_ is any callable scoring two payloads - great-circle, ellipsoidal,
 *      or trajectory-alignment measures.
 */
template <typename value_type_, typename metric_type_>
struct geospatial_ram_index;

} // namespace usearch
} // namespace unum
