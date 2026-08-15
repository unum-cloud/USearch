/**
 *  @brief The geospatial-modality C interface: similarity search over point sets on and
 *      above the ellipsoid, in space and in time - single positions, time-ordered tracks,
 *      and closed boundaries - under nearest-approach, whole-shape, coverage, and
 *      order-respecting scoring.
 *
 *  @author Ash Vardanian
 *  @file include/usearch/geospatial_index.h
 *
 *  Every entry is a @b point @b set: one or more positions sharing a key. A single
 *  position is a set of one, a vehicle track is a set of thousands, a parcel outline is a
 *  closed set of vertices. Sets are variable-height and fixed-width, so batch inputs take a
 *  @b tape - one array of base pointers, one of point counts, index-aligned - rather than
 *  the dense modality's strided matrix. Coordinates are numeric rather than textual, so the
 *  tape carries counts instead of the sequence modality's @c usearch_string_t views.
 *
 *  @section geospatial_index_geometry Internal Geometry
 *
 *  Positions arrive as geodetic degrees and are stored as Earth-Centered Earth-Fixed
 *  metres, optionally with a fourth axis carrying time scaled into metres. The
 *  straight-line chord between two Earth-fixed positions is a strict lower bound on every
 *  path across the surface, and on a sphere it is a strictly increasing function of the
 *  great-circle arc - so ranking by chord and ranking by arc agree exactly, and the
 *  ellipsoid perturbs that agreement only in the second-order term. Traversal therefore
 *  runs on plain Euclidean distance with no transcendental functions, and an accurate
 *  geodesic is paid only when re-ranking the returned candidates.
 *
 *  @section geospatial_index_modes One Graph, Many Modes
 *
 *  The graph is built on symmetric @b Hausdorff distance over the widest axis set the
 *  layout carries. Three of the four scoring modes are then one-Lipschitz against those
 *  edge weights - for any query set @c Q and entries @c A and @c B,
 *
 *      | nearest_approach(Q, A) - nearest_approach(Q, B) |  <=  hausdorff(A, B)
 *
 *  and the same bound holds for coverage and for Hausdorff itself. Because the ground
 *  metric only shrinks when axes are dropped, and Hausdorff is monotone in its ground
 *  metric, a graph built on the full four-axis distance also bounds every projection of
 *  it. One graph therefore serves every combination of @c usearch_geospatial_mode_t and
 *  @c usearch_geospatial_axes_t, with descent guaranteed to converge in each.
 *
 *  Route scoring is the exception. Frechet distance is unbounded relative to Hausdorff -
 *  a track and its reverse are Hausdorff-identical and Frechet-far - so it earns a second
 *  edge family, selected at construction through @c usearch_geospatial_edges_t.
 *
 *  @section geospatial_index_emplacement Emplaced Queries
 *
 *  Alone among the modalities, search consumes a prepared object rather than a raw tape.
 *  That is deliberate, not drift: the geodetic conversion and time scaling are the expensive
 *  half of a point query, and one emplaced set is reused across every mode and axis
 *  combination a caller asks of it.
 *
 *  @section geospatial_index_concurrency Concurrency & Thread Safety
 *
 *  A handle may be shared across threads. On a RAM handle, insertions and searches overlap
 *  freely: searches read only entries published behind the visibility watermark and never
 *  write to the graph, while insertions synchronize through per-node locks and publish in
 *  order. A VRAM handle is driven by one host thread through streams. An NVMe handle admits
 *  arbitrarily many concurrent searches, having nothing to mutate. Serialization is the one
 *  exception: saving demands a mutation-free window, though concurrent readers keep
 *  searching throughout. An emplaced query object is @b not internally synchronized - build
 *  one per thread, or confine it to the thread that emplaced it. Per-call outputs - stats,
 *  counts, the error pointer - belong to their call and must not be shared between
 *  in-flight calls.
 */
#ifndef UNUM_USEARCH_GEOSPATIAL_INDEX_H
#define UNUM_USEARCH_GEOSPATIAL_INDEX_H

#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Similarity Search Index over point sets in space & time.
 */
USEARCH_EXPORT typedef void* usearch_geospatial_index_t;

/**
 *  @brief A query point set converted into the index's internal geometry once and reused
 *      across searches. Emplacing is the expensive half of a query - the geodetic
 *      conversion, the time scaling, and any per-set precomputation the metric needs -
 *      so a caller asking several questions of the same object pays it once.
 *
 *  @see usearch_geospatial_emplace To build one.
 *  @see usearch_geospatial_release To dispose of one.
 */
USEARCH_EXPORT typedef void* usearch_geospatial_emplaced_t;

/**
 *  @brief A consistent snapshot cursor over the index contents, capturing the publication
 *      watermark at creation - concurrent inserts never appear mid-iteration.
 */
USEARCH_EXPORT typedef void* usearch_geospatial_cursor_t;

/**
 *  @brief Which axes each incoming position carries, and in which order. Fixed for the
 *      lifetime of the index: every point set must present the same width.
 */
USEARCH_EXPORT typedef enum usearch_geospatial_layout_t {
    usearch_geospatial_layout_unknown_k = 0,
    /** @brief Latitude, longitude - degrees on the WGS-84 ellipsoid. */
    usearch_geospatial_layout_surface_k = 2,
    /** @brief Latitude, longitude, altitude - degrees and metres above the ellipsoid. */
    usearch_geospatial_layout_elevated_k = 3,
    /** @brief Latitude, longitude, seconds - a surface track through time. */
    usearch_geospatial_layout_timed_k = 4,
    /** @brief Latitude, longitude, altitude, seconds - the full four-axis form. */
    usearch_geospatial_layout_elevated_timed_k = 5,
} usearch_geospatial_layout_t;

/**
 *  @brief What the ordering of points within one set means. Decides which scoring modes
 *      the index will accept: route scoring is defined only for ordered sets.
 */
USEARCH_EXPORT typedef enum usearch_geospatial_structure_t {
    usearch_geospatial_structure_unknown_k = 0,
    /** @brief Order carries nothing - a scatter of observations, a bag of vertices. */
    usearch_geospatial_structure_cloud_k = 1,
    /** @brief Order is the traversal - a vehicle track, a vessel voyage, a flight. */
    usearch_geospatial_structure_path_k = 2,
    /** @brief A closed boundary; the last point is understood to join the first. */
    usearch_geospatial_structure_ring_k = 3,
} usearch_geospatial_structure_t;

/**
 *  @brief Which axes participate in one search. Narrowing the axes narrows the ground
 *      metric, which only shrinks Hausdorff distance, so every setting is served by the
 *      graph built on the layout's widest form.
 */
USEARCH_EXPORT typedef enum usearch_geospatial_axes_t {
    usearch_geospatial_axes_unknown_k = 0,
    /** @brief Ground track only - altitude and time ignored. */
    usearch_geospatial_axes_horizontal_k = 1,
    /** @brief Position in three dimensions - time ignored. */
    usearch_geospatial_axes_spatial_k = 2,
    /** @brief Every axis the layout carries, time included. */
    usearch_geospatial_axes_full_k = 3,
} usearch_geospatial_axes_t;

/**
 *  @brief What "close" means for one search. Modes differ in whether a single coincidence
 *      suffices or the whole set must agree, and in whether ordering is respected.
 */
USEARCH_EXPORT typedef enum usearch_geospatial_mode_t {
    usearch_geospatial_mode_unknown_k = 0,
    /**
     *  @brief Nearest approach - the smallest separation between any query point and any
     *      entry point. Answers "did these two come close to each other". Paired with
     *      @c usearch_geospatial_axes_full_k on a timed layout it answers "did they come
     *      close at the same moment", which is the rendezvous and co-travel query.
     */
    usearch_geospatial_mode_approach_k = 1,
    /**
     *  @brief Symmetric Hausdorff - the largest of all nearest approaches, in both
     *      directions. Answers "are these the same shape everywhere", and is the metric
     *      the graph itself is built on.
     */
    usearch_geospatial_mode_shape_k = 2,
    /**
     *  @brief Directed Hausdorff from the query onto the entry - the worst-covered query
     *      point. Answers "does this entry cover the whole of my corridor", and unlike
     *      the symmetric form it is untroubled by an entry far longer than the query.
     */
    usearch_geospatial_mode_coverage_k = 3,
    /**
     *  @brief Discrete Frechet - the tightest leash for two walkers who may pause but
     *      never reverse. Answers "same route, same direction". Requires
     *      @c usearch_geospatial_structure_path_k and Frechet edges.
     */
    usearch_geospatial_mode_route_k = 4,
} usearch_geospatial_mode_t;

/**
 *  @brief Which edge families the graph carries. Hausdorff edges serve approach, shape,
 *      and coverage in every axis setting; route scoring is not Lipschitz against them
 *      and needs its own.
 */
USEARCH_EXPORT typedef enum usearch_geospatial_edges_t {
    usearch_geospatial_edges_unknown_k = 0,
    /** @brief One family over Hausdorff - the smaller, faster graph. */
    usearch_geospatial_edges_hausdorff_k = 1,
    /** @brief A second family over Frechet, unioned into the same neighbor lists. */
    usearch_geospatial_edges_hausdorff_and_frechet_k = 2,
} usearch_geospatial_edges_t;

/**
 *  @brief Construction & loading options of a geospatial index. The @c residency field
 *      tags which union member is read; zeroed optional fields derive from the geometry.
 */
USEARCH_EXPORT typedef struct usearch_geospatial_options_t {

    /** @brief Which axes incoming positions carry. */
    usearch_geospatial_layout_t layout;
    /** @brief What ordering within a point set means. */
    usearch_geospatial_structure_t structure;
    /** @brief Which edge families to construct. */
    usearch_geospatial_edges_t edges;

    /**
     *  @brief Metres of horizontal separation carrying as much information as one second
     *      of temporal separation - the exchange rate that lets one distance mix both.
     *
     *  Part of index @b identity, not a search parameter: it is fixed at construction and
     *  changing it changes every ranking, exactly as a scalar type would. Zero derives it
     *  as the median speed between consecutive points of the inserted corpus, which for
     *  road vehicles lands near 13, for merchant shipping near 6, and for airliners near
     *  250. Ignored by layouts without a time axis.
     */
    double characteristic_speed;

    /** @brief The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** @brief Selects the active member of the union below, and the meaning of "load". */
    usearch_residency_t residency;
    union {
        usearch_ram_options_t ram;
        usearch_vram_options_t vram;
        usearch_nvme_options_t nvme;
    };

} usearch_geospatial_options_t;

/**
 *  @brief Initializes a geospatial index over point sets.
 *  @param[in] options Pointer to the populated options.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 *  @return A handle to the initialized index, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_geospatial_index_t usearch_geospatial_init( //
    usearch_geospatial_options_t const* options, usearch_error_t* error);

/**
 *  @brief Frees the resources associated with the index; cannot fail, so takes no error.
 */
USEARCH_EXPORT void usearch_geospatial_free(usearch_geospatial_index_t index);

/**
 *  @brief Reserves capacity ahead of insertion.
 *  @param[in] entries The desired total entry capacity, including the current size.
 *  @param[in] points Expected total point volume across all sets; zero grows lazily.
 */
USEARCH_EXPORT void usearch_geospatial_reserve( //
    usearch_geospatial_index_t index, size_t entries, size_t points, usearch_error_t* error);

/**
 *  @brief Erases all entries, keeping the reserved capacity.
 */
USEARCH_EXPORT void usearch_geospatial_clear(usearch_geospatial_index_t index, usearch_error_t* error);

/** @brief Reports the number of stored entries. */
USEARCH_EXPORT size_t usearch_geospatial_size(usearch_geospatial_index_t index, usearch_error_t* error);
/** @brief Reports the reserved capacity in entries. */
USEARCH_EXPORT size_t usearch_geospatial_capacity(usearch_geospatial_index_t index, usearch_error_t* error);
/** @brief Reports the frozen connections-per-node bound of the graph. */
USEARCH_EXPORT size_t usearch_geospatial_connectivity(usearch_geospatial_index_t index, usearch_error_t* error);
/** @brief Reports the memory usage of the index in bytes, point arenas included. */
USEARCH_EXPORT size_t usearch_geospatial_memory_usage(usearch_geospatial_index_t index, usearch_error_t* error);
/** @brief Reports the capability codename the index dispatched to on this machine. */
USEARCH_EXPORT char const* usearch_geospatial_capabilities(usearch_geospatial_index_t index, usearch_error_t* error);

/**
 *  @brief Reports the characteristic speed in force, whether supplied or derived.
 *      Zero for layouts without a time axis.
 */
USEARCH_EXPORT double usearch_geospatial_characteristic_speed( //
    usearch_geospatial_index_t index, usearch_error_t* error);

/**
 *  @brief Optional per-call settings, shared by insertion and every search form - a pure
 *      input, never written. Passing @c NULL derives every setting.
 */
USEARCH_EXPORT typedef struct usearch_geospatial_expansion_config_t {
    /** @brief The @b optional candidate-pool breadth - the `ef` of the literature; zero
     *      derives. */
    size_t expansion_factor;
} usearch_geospatial_expansion_config_t;

/**
 *  @brief Traversal telemetry, shared by insertion and every search form - a pure output,
 *      fully overwritten by the call it is passed to. Passing @c NULL skips reporting.
 */
USEARCH_EXPORT typedef struct usearch_geospatial_traversal_stats_t {
    /** @brief Graph members visited while resolving the batch. */
    size_t visited_members;
    /** @brief Distances computed while resolving the batch, diversity re-selections included. */
    size_t computed_distances;
    /** @brief Starvation restarts - re-seedings after the frontier drained prematurely,
     *      and a graph-connectivity health signal rather than a cost counter. */
    size_t restarts;
} usearch_geospatial_traversal_stats_t;

/**
 *  @brief Adds a tape of point sets under their keys - the native batch operation; a
 *      single set is a tape of one, and a single position is a set of one.
 *
 *  @param[inout] index The handle to be populated.
 *  @param[in] keys Array of @p count keys, one per point set.
 *  @param[in] point_sets Array of @p count base pointers, one per set. Each points at
 *      `point_counts[i] * layout_width` coordinates, row-major, in the index's layout
 *      order - degrees for the angular axes, metres for altitude, seconds for time.
 *  @param[in] point_counts Array of @p count point counts, index-aligned with @p point_sets.
 *  @param[in] count Number of point sets in the tape.
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 *  @return Number of successfully inserted point sets - equal to @p count unless an error
 *      stopped the batch early.
 */
USEARCH_EXPORT size_t usearch_geospatial_add(                    //
    usearch_geospatial_index_t index, usearch_key_t const* keys, //
    double const* const* point_sets, size_t const* point_counts, //
    size_t count,                                                //
    usearch_geospatial_expansion_config_t const* config,         //
    usearch_geospatial_traversal_stats_t* stats, usearch_error_t* error);

/**
 *  @brief Converts a query point set into the index's internal geometry, returning a
 *      handle reusable across searches of differing mode and axis selection.
 *
 *  @param[in] index The handle whose layout and characteristic speed govern conversion.
 *  @param[in] points Base pointer to `point_count * layout_width` coordinates.
 *  @param[in] point_count Number of positions; one is legal and is the point query.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 *  @return A handle to the emplaced object, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_geospatial_emplaced_t usearch_geospatial_emplace( //
    usearch_geospatial_index_t index, double const* points, size_t point_count, usearch_error_t* error);

/**
 *  @brief Frees an emplaced query object. The index it was emplaced against must outlive it.
 */
USEARCH_EXPORT void usearch_geospatial_release( //
    usearch_geospatial_emplaced_t emplaced, usearch_error_t* error);

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a batch of emplaced query
 *      objects under one scoring mode - the native batch operation, parallelized
 *      internally; one query is a batch of one.
 *
 *  @param[in] index The handle to be queried.
 *  @param[in] queries Array of @p queries_count emplaced query handles.
 *  @param[in] queries_count Number of emplaced queries in the batch.
 *  @param[in] mode What "close" means for this search.
 *  @param[in] axes Which axes participate; narrowing never invalidates the graph.
 *  @param[in] wanted Upper bound on matches per query - the "k" in "kANN".
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] keys Output matrix of `queries_count × wanted` keys; rows contiguous.
 *  @param[in] keys_stride Number of bytes between starts of consecutive rows in @p keys.
 *  @param[out] distances Output matrix of `queries_count × wanted` scores in metres;
 *      rows contiguous. Route scores are leash lengths, the rest are separations.
 *  @param[in] distances_stride Number of bytes between starts of consecutive @p distances rows.
 *  @param[out] counts Output array of @p queries_count per-query match counts.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 */
USEARCH_EXPORT void usearch_geospatial_search(                          //
    usearch_geospatial_index_t index,                                   //
    usearch_geospatial_emplaced_t const* queries, size_t queries_count, //
    usearch_geospatial_mode_t mode, usearch_geospatial_axes_t axes,     //
    size_t wanted,                                                      //
    usearch_geospatial_expansion_config_t const* config,                //
    usearch_geospatial_traversal_stats_t* stats,                        //
    usearch_key_t* keys, size_t keys_stride,                            //
    usearch_distance_t* distances, size_t distances_stride,             //
    size_t* counts, usearch_error_t* error);

/**
 *  @brief Reports every entry scoring within @p radius metres of one emplaced query -
 *      the containment counterpart of the ranked search, unbounded in result count.
 *
 *  Exact only up to the graph's reachability: a candidate the traversal never visits is
 *  never reported. Callers needing certainty over a small radius are better served by a
 *  cell-keyed posting list, which answers containment without approximation.
 *
 *  @param[in] capacity Number of slots available in @p keys and @p distances.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 *  @return Number of entries written, saturating at @p capacity.
 */
USEARCH_EXPORT size_t usearch_geospatial_within(                           //
    usearch_geospatial_index_t index, usearch_geospatial_emplaced_t query, //
    usearch_geospatial_mode_t mode, usearch_geospatial_axes_t axes,        //
    double radius,                                                         //
    usearch_key_t* keys, usearch_distance_t* distances,                    //
    size_t capacity, usearch_error_t* error);

/**
 *  @brief The filtered sibling of @c usearch_geospatial_search: identical batch shape,
 *      admitting only entries the @p predicate approves. Non-matching entries still route
 *      the traversal, so recall does not collapse at low selectivity.
 *
 *  @param[in] predicate Per-key admission callback; returns non-zero to admit.
 *  @param[in] predicate_state Opaque pointer forwarded into every @p predicate call.
 */
USEARCH_EXPORT void usearch_geospatial_search_if(                       //
    usearch_geospatial_index_t index,                                   //
    usearch_geospatial_emplaced_t const* queries, size_t queries_count, //
    usearch_geospatial_mode_t mode, usearch_geospatial_axes_t axes,     //
    size_t wanted,                                                      //
    usearch_geospatial_expansion_config_t const* config,                //
    usearch_geospatial_traversal_stats_t* stats,                        //
    usearch_predicate_t predicate, void* predicate_state,               //
    usearch_key_t* keys, size_t keys_stride,                            //
    usearch_distance_t* distances, size_t distances_stride,             //
    size_t* counts, usearch_error_t* error);

/** @brief What one key holds, enough to size a buffer before retrieving any of it. */
USEARCH_EXPORT typedef struct usearch_geospatial_key_stats_t {
    /** @brief Entries sharing the key; keys need not be unique. */
    size_t entries;
    /** @brief Positions summed across those entries. */
    size_t total_points;
    /** @brief Positions of the longest single entry - the per-entry stride one @c get needs. */
    size_t largest_points;
} usearch_geospatial_key_stats_t;

/**
 *  @brief Counts the entries stored under @p key; zero means the index holds none.
 *
 *  Pairing this with @c usearch_geospatial_get is not atomic: an insertion between the two
 *  calls leaves the count stale. That is harmless, because entries are only ever added, so
 *  a stale count under-reports and @c usearch_geospatial_get returns the true total - a
 *  caller seeing more than it asked for allocates again and converges.
 *
 *  @param[out] stats The @b optional extent of the key's payload; @c NULL asks for the
 *      count alone.
 *  @return The number of entries under @p key, mirroring @c stats->entries.
 */
USEARCH_EXPORT size_t usearch_geospatial_count(          //
    usearch_geospatial_index_t index, usearch_key_t key, //
    usearch_geospatial_key_stats_t* stats, usearch_error_t* error);

/**
 *  @brief Retrieves the point sets stored under @p key into a caller-owned buffer, as the
 *      geodetic degrees, metres, and seconds they were inserted as.
 *
 *  Entries are written at @p stride spacing in insertion order, so sizing the buffer takes
 *  @c usearch_geospatial_count first: @c entries slots of @c largest_points positions each.
 *
 *  @param[out] point_sets Destination for up to @p wanted entries; @c NULL probes the count
 *      without copying.
 *  @param[in] wanted Upper bound on entries to write - the entry capacity of @p point_sets.
 *  @param[in] stride Number of bytes between starts of consecutive entries in @p point_sets.
 *  @param[out] point_counts The @b optional per-entry true position count, up to @p wanted
 *      of them; @c NULL skips it.
 *  @return The @b true number of entries under @p key, never the number written - a value
 *      above @p wanted means the copy was truncated, so allocate that many and call again.
 *      Zero when the key is absent.
 */
USEARCH_EXPORT size_t usearch_geospatial_get(                               //
    usearch_geospatial_index_t index, usearch_key_t key,                    //
    double* point_sets, size_t wanted, size_t stride, size_t* point_counts, //
    usearch_error_t* error);

/**
 *  @brief Absorbs @p donor into @p index - cheaper than re-inserting its point sets one by
 *      one, wiring cross-links via landmark stitching. The donor handle is consumed and
 *      must not be used afterwards, not even for @c usearch_geospatial_free. Readers of
 *      @p index may keep searching throughout: unpublished entries hide behind the watermark.
 */
USEARCH_EXPORT void usearch_geospatial_merge( //
    usearch_geospatial_index_t index, usearch_geospatial_index_t donor, usearch_error_t* error);

/**
 *  @brief Opens a snapshot cursor over the current contents; see
 *      @c usearch_geospatial_cursor_t.
 *  @return The cursor, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_geospatial_cursor_t usearch_geospatial_cursor(usearch_geospatial_index_t index,
                                                                     usearch_error_t* error);

/**
 *  @brief Copies the next batch of entries out of the cursor.
 *  @param[out] keys Output array of up to @p wanted keys.
 *  @param[out] point_sets Output array of up to @p wanted borrowed base pointers into the
 *      index's arena, valid until the cursor moves or closes; @c NULL skips point export.
 *      Coordinates are the internal Earth-fixed metres, not the geodetic degrees inserted.
 *  @param[out] point_counts Output array of up to @p wanted position counts.
 *  @param[in] wanted Upper bound on entries to write - the capacity of every output array.
 *  @return Number of entries written; zero when the snapshot is exhausted.
 */
USEARCH_EXPORT size_t usearch_geospatial_cursor_next(        //
    usearch_geospatial_cursor_t cursor, usearch_key_t* keys, //
    float const** point_sets, size_t* point_counts, size_t wanted, usearch_error_t* error);

/** @brief Releases the cursor. */
USEARCH_EXPORT void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t cursor);

/**
 *  @brief Reports the expected serialized size of the index in bytes.
 */
USEARCH_EXPORT size_t usearch_geospatial_serialized_length(usearch_geospatial_index_t index, usearch_error_t* error);

/**
 *  @brief Saves the index under @p path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 */
USEARCH_EXPORT void usearch_geospatial_save(usearch_geospatial_index_t index, usearch_string_t path,
                                            usearch_error_t* error);

/**
 *  @brief Loads a checkpoint into an initialized handle, with the handle's residency
 *      deciding the behavior: materialize into RAM, upload to the device, or open in place
 *      on disk - the read-only form, where mutating verbs fail. A handle whose geometry was
 *      left zeroed freezes it here, from the checkpoint's own headers, characteristic speed
 *      included.
 */
USEARCH_EXPORT void usearch_geospatial_load(usearch_geospatial_index_t index, usearch_string_t path,
                                            usearch_error_t* error);

/**
 *  @brief Reads a checkpoint's geometry & metric metadata into @p options without loading it.
 */
USEARCH_EXPORT void usearch_geospatial_metadata( //
    usearch_string_t path, usearch_geospatial_options_t* options, usearch_error_t* error);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_GEOSPATIAL_INDEX_H
