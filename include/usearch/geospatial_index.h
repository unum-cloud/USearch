/**
 *  @brief The geospatial-modality C interface: similarity search over low-dimensional
 *      representations in space & time - points on the sphere and the ellipsoid under
 *      haversine and vincenty distances, and trajectories under alignment measures.
 *
 *  @author Ash Vardanian
 *  @file include/usearch/geospatial_index.h
 *
 *  A placeholder surface: the handle and the metric vocabulary are named here so the C ABI
 *  and the language bindings can be laid out against them, while the verbs land with the
 *  first per-residency implementation. Points ride a strided matrix like the dense
 *  modality's vectors; trajectories are time-ordered runs of them, so their batch inputs
 *  will take the sequence modality's tape shape instead.
 */
#ifndef UNUM_USEARCH_GEOSPATIAL_INDEX_H
#define UNUM_USEARCH_GEOSPATIAL_INDEX_H

#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Similarity Search Index over coordinates in space & time.
 */
USEARCH_EXPORT typedef void* usearch_geospatial_t;

/**
 *  @brief A consistent snapshot cursor over the index contents, capturing the publication
 *      watermark at creation - concurrent inserts never appear mid-iteration.
 */
USEARCH_EXPORT typedef void* usearch_geospatial_cursor_t;

/** @brief The geospatial scoring kinds supported out of the box. */
USEARCH_EXPORT typedef enum usearch_geospatial_metric_t {
    usearch_geospatial_metric_unknown_k = 0,
    /** @brief Great-circle distance on a sphere - the cheap approximation. */
    usearch_geospatial_metric_haversine_k = 1,
    /** @brief Geodesic distance on an ellipsoid - the accurate, iterative measure. */
    usearch_geospatial_metric_vincenty_k = 2,
} usearch_geospatial_metric_t;

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_GEOSPATIAL_INDEX_H
