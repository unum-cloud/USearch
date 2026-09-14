/**
 *  @file include/usearch/geospatial_index.h
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The geospatial-modality C interface.
 *
 *  Similarity search over positions on and above the ellipsoid, in space and in time - single
 *  positions, time-ordered tracks, and closed boundaries - under nearest, overlap, whole-shape,
 *  coverage, and order-respecting scoring.
 *
 *  Every entry is a @b point @b set: one or more positions sharing a key. A single
 *  position is a set of one, a vehicle track is a set of thousands, a parcel outline is a
 *  closed set of vertices. Sets are variable-height and fixed-width, so batch inputs take a
 *  @b tape - one array of base pointers, one of point counts, index-aligned - rather than
 *  the dense modality's strided matrix. Coordinates are numeric rather than textual, so the
 *  tape carries counts instead of the sequence modality's @c usearch_bytes_view_t views.
 *
 *  @section geospatial_index_geometry Internal Geometry
 *
 *  Positions arrive as geodetic degrees and are stored as Earth-centred, Earth-fixed
 *  @b integer @b centimetres, projected onto the ellipsoid's surface; altitude and time
 *  are attributes of a point rather than axes of its position. The coordinates are
 *  Euclidean: every ordering inside the index is the straight-line @b chord between two
 *  such positions, three subtractions and three wide multiply-adds with no rounding. On a
 *  sphere the chord is a strictly increasing function of the great-circle arc, so the two
 *  rank identically and no transcendental is ever evaluated on the walk. The @b geodesic
 *  on the WGS-84 ellipsoid is not a monotone function of the chord - two pairs sharing one
 *  chord differ in geodesic length by up to the flattening, one part in 298 - so it is
 *  computed only for the candidates that are reported, through Karney's always-converging
 *  solution rather than Vincenty's.
 *
 *  @section geospatial_index_modes One Graph, Many Modes
 *
 *  One node is one point, and a set is its points under one key. The graph is a
 *  single-layer navigable small world over the surface chord, wired by the nearest
 *  neighbor in each of eight tangent-plane cones plus four long-range links, and every
 *  scoring mode is answered from one primitive: a radius walk around each query point,
 *  bounded by the call's @c usearch_geospatial_tolerance_t, returning the entry points
 *  inside it. Nearest needs no radius. Overlap is computed during that walk, folding each
 *  query point's best entry point per key into a running sum. The set-shaped modes rerank
 *  the keys the walk found with an exact set metric, which is complete because an entry
 *  scoring within the tolerance under any of them has a point within it of some query
 *  point. Narrowing @c usearch_geospatial_axes_t only widens a candidate set, since the
 *  surface chord never exceeds a distance that adds altitude or time, so one graph serves
 *  every mode and axis combination.
 *
 *  @section geospatial_index_emplacement Emplaced Queries
 *
 *  Alone among the modalities, search consumes a prepared object rather than a raw tape.
 *  That is deliberate, not drift: the geodetic conversion is the expensive half of a point
 *  query, and one emplaced set is reused across every mode, tolerance and axis combination
 *  a caller asks of it.
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
 *
 *  Entries are permanent. No verb removes one, so an index only ever grows until it is
 *  cleared wholesale, and a key's entries accumulate up to the @c values_per_key bound its
 *  options froze.
 *
 *  The executor, not the handle, is what serializes: a handle admits any number of
 *  concurrent callers, while an executor admits @b one driver at a time and reports
 *  @c usearch_error_contended_k to the second.
 */
#ifndef UNUM_USEARCH_GEOSPATIAL_INDEX_H
#define UNUM_USEARCH_GEOSPATIAL_INDEX_H

#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Similarity Search Index over point sets in space & time. */
USEARCH_EXPORT typedef void* usearch_geospatial_index_t;

/**
 *  @brief A query point set converted into the index's internal geometry once and reused across
 *      searches.
 *
 *  Emplacing is the expensive half of a query - the geodetic conversion, the time scaling, and any
 *  per-set precomputation the metric needs - so a caller asking several questions of the same
 *  object pays it once.
 *
 *  @see usearch_geospatial_emplace To build one.
 *  @see usearch_geospatial_release To dispose of one.
 */
USEARCH_EXPORT typedef void* usearch_geospatial_emplaced_t;

/** A consistent snapshot cursor over the index contents, capturing the publication
 *  watermark at creation - concurrent inserts never appear mid-iteration. */
USEARCH_EXPORT typedef void* usearch_geospatial_cursor_t;

/** Which axes each incoming position carries, and in which order. Fixed for the
 *  lifetime of the index: every point set must present the same width. */
USEARCH_EXPORT typedef enum usearch_geospatial_layout_t {
    usearch_geospatial_layout_unknown_k = 0,

    /** Latitude, longitude - degrees on the WGS-84 ellipsoid. */
    usearch_geospatial_layout_surface_k = 2,

    /** Latitude, longitude, altitude - degrees and metres above the ellipsoid. */
    usearch_geospatial_layout_elevated_k = 3,

    /** Latitude, longitude, seconds - a surface track through time. */
    usearch_geospatial_layout_timed_k = 4,

    /** Latitude, longitude, altitude, seconds - the full four-axis form. */
    usearch_geospatial_layout_elevated_timed_k = 5,
} usearch_geospatial_layout_t;

/** What the ordering of points within one set means. Decides which scoring modes
 *  the index will accept: route scoring is defined only for ordered sets. */
USEARCH_EXPORT typedef enum usearch_geospatial_structure_t {
    usearch_geospatial_structure_unknown_k = 0,

    /** Order carries nothing - a scatter of observations, a bag of vertices. */
    usearch_geospatial_structure_cloud_k = 1,

    /** Order is the traversal - a vehicle track, a vessel voyage, a flight. */
    usearch_geospatial_structure_path_k = 2,

    /** A closed boundary; the last point is understood to join the first. */
    usearch_geospatial_structure_ring_k = 3,
} usearch_geospatial_structure_t;

/** Which axes participate in one search. The graph walks the surface chord alone;
 *  altitude and time enter when the candidates it found are scored, and a distance
 *  that adds them never falls below the surface chord, so narrowing the axes only
 *  widens the candidate set. */
USEARCH_EXPORT typedef enum usearch_geospatial_axes_t {
    usearch_geospatial_axes_unknown_k = 0,

    /** Ground track only - altitude and time ignored. */
    usearch_geospatial_axes_horizontal_k = 1,

    /** Position in three dimensions - time ignored. */
    usearch_geospatial_axes_spatial_k = 2,

    /** Every axis the layout carries, time included. */
    usearch_geospatial_axes_full_k = 3,
} usearch_geospatial_axes_t;

/**
 *  @brief What "close" means for one search.
 *
 *  Writes @c D for the @b [|Q|,|E|] matrix of the point distance between every query point
 *  and every entry point. Nearest and the two Hausdorff forms are reductions of @c D by
 *  @c min and @c max alone; overlap is a count; route is a walk across the same table.
 *  Every one of them is computed over the keys a radius walk around each query point found,
 *  so none rides an edge family of its own.
 *
 *  Reductions by @c min and @c max commute with any strictly increasing function of the
 *  point distance, so ranking by chord and by great-circle arc are identical for them. A
 *  sum does not commute, and overlap's indicator kernel is affected only exactly at the
 *  tolerance boundary; that is the standing constraint on what may be added here.
 */
USEARCH_EXPORT typedef enum usearch_geospatial_mode_t {
    usearch_geospatial_mode_unknown_k = 0,

    /**
     *  @brief Nearest approach - the smallest separation between any query point and any
     *      entry point.
     *
     *  Answers "did these two come close to each other". Paired with
     *  @c usearch_geospatial_axes_full_k and a tolerance carrying seconds it answers "did
     *  they come close at the same moment", which is the rendezvous query. One coincidence
     *  suffices, so everything else about either track is discarded. The one mode that
     *  needs no radius: a zeroed tolerance ranks by the nearest point alone.
     *
     *  A(Q,E) = min_i min_j D[i,j]
     *
     *  @code{.py}
     *  D = chord(Q, E)                              # [|Q|,|E|], shared by every mode here
     *  approach = D.min()
     *  @endcode
     */
    usearch_geospatial_mode_approach_k = 1,

    /**
     *  @brief Overlap - how many query points have an entry point within the tolerance,
     *      at the same moment when the tolerance carries seconds.
     *
     *  Answers "how much of my route did this entry share", and with time "how long were we
     *  together": co-travel, shared stops, visited places. A count rather than a worst case,
     *  so one stray point costs one unit instead of deciding the score. Computed while the
     *  radius walk runs, folding each query point's best entry point per key into a sum, and
     *  exact whenever that walk enumerates everything inside the tolerance. Scores rank
     *  descending: the largest overlap is the best match.
     *
     *  O(Q,E) = sum_i [ min_j D[i,j] <= r ]
     *
     *  @code{.py}
     *  D = chord(Q, E)
     *  overlap = (D.min(axis=1) <= r).sum()         # a count, never a distance
     *  @endcode
     */
    usearch_geospatial_mode_overlap_k = 2,

    /**
     *  @brief Symmetric Hausdorff - the largest of all nearest approaches, in both directions.
     *
     *  Answers "are these the same shape everywhere". The whole of both tracks must agree:
     *  one query point far from every entry point, or one entry point far from every query
     *  point, sets the score.
     *
     *  H(Q,E) = max( max_i min_j D[i,j], max_j min_i D[i,j] )
     *
     *  @code{.py}
     *  D = chord(Q, E)
     *  shape = max(D.min(axis=1).max(), D.min(axis=0).max())
     *  @endcode
     */
    usearch_geospatial_mode_shape_k = 3,

    /**
     *  @brief Directed Hausdorff from the query onto the entry - the worst-covered query point.
     *
     *  Answers "does this entry cover the whole of my corridor". Reducing over query rows
     *  alone is what makes it untroubled by an entry far longer than the query, and equally
     *  what makes it directional: an entry covering the query does not imply the reverse.
     *
     *  C(Q,E) = max_i min_j D[i,j]
     *
     *  @code{.py}
     *  D = chord(Q, E)
     *  coverage = D.min(axis=1).max()               # query rows only; entry length never enters
     *  @endcode
     */
    usearch_geospatial_mode_coverage_k = 4,

    /**
     *  @brief Discrete Frechet - the tightest leash for two walkers who may pause but never
     *      reverse.
     *
     *  Answers "same route, same direction". Requires @c usearch_geospatial_structure_path_k.
     *  The minimum over three predecessors is what forbids reversing, and it is why this
     *  mode is a walk rather than a reduction: two tracks covering identical ground in
     *  opposite orders are Hausdorff-identical and Frechet-far.
     *
     *  F[i,j] = max( D[i,j], min( F[i-1,j], F[i,j-1], F[i-1,j-1] ) ),  score = F[|Q|,|E|]
     *
     *  @code{.py}
     *  D, F = chord(Q, E), full((len(Q), len(E)), inf)
     *  for i in range(len(Q)):
     *      for j in range(len(E)):
     *          back = min(F[i - 1, j] if i else inf, F[i, j - 1] if j else inf,
     *                     F[i - 1, j - 1] if i and j else inf)
     *          F[i, j] = D[i, j] if i == j == 0 else max(D[i, j], back)
     *  route = F[-1, -1]                            # a walk, not a reduction
     *  @endcode
     */
    usearch_geospatial_mode_route_k = 5,
} usearch_geospatial_mode_t;

/** Construction & loading options of a geospatial index. The @c residency field
 *  tags which union member is read; zeroed optional fields derive from the geometry. */
USEARCH_EXPORT typedef struct usearch_geospatial_options_t {

    /** Which axes incoming positions carry. */
    usearch_geospatial_layout_t layout;

    /** What ordering within a point set means; paths carry a successor per point. */
    usearch_geospatial_structure_t structure;

    /** The @b optional connections-per-node bound; zero derives to twelve - eight
     *  tangent-plane cones and four long-range links. */
    size_t connectivity;

    /** Upper bound on points sharing one key; zero derives to one - unique keys. */
    size_t values_per_key;

    /** Selects the active member of the union below, and the meaning of "load". */
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
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return A handle to the initialized index, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_geospatial_index_t usearch_geospatial_init( //
    usearch_geospatial_options_t const* options, usearch_error_t* error);

/** Frees the resources associated with the index; cannot fail, so takes no error. */
USEARCH_EXPORT void usearch_geospatial_free(usearch_geospatial_index_t index);

/**
 *  @brief Reserves capacity ahead of insertion.
 *  @param[in] entries The desired total entry capacity, including the current size.
 *  @param[in] points Expected total point volume across all sets; zero grows lazily.
 */
USEARCH_EXPORT void usearch_geospatial_reserve( //
    usearch_geospatial_index_t index, size_t entries, size_t points, usearch_error_t* error);

/** Erases all entries, keeping the reserved capacity. */
USEARCH_EXPORT void usearch_geospatial_clear(usearch_geospatial_index_t index, usearch_error_t* error);

/** Reports the number of stored entries. */
USEARCH_EXPORT size_t usearch_geospatial_size(usearch_geospatial_index_t index, usearch_error_t* error);

/** Reports the reserved capacity in entries. */
USEARCH_EXPORT size_t usearch_geospatial_capacity(usearch_geospatial_index_t index, usearch_error_t* error);

/** Reports the frozen connections-per-node bound of the graph. */
USEARCH_EXPORT size_t usearch_geospatial_connectivity(usearch_geospatial_index_t index, usearch_error_t* error);

/** Reports the memory usage of the index in bytes, point arenas included. */
USEARCH_EXPORT size_t usearch_geospatial_memory_usage(usearch_geospatial_index_t index, usearch_error_t* error);

/** Reports the capability codename the index dispatched to on this machine. */
USEARCH_EXPORT char const* usearch_geospatial_capabilities(usearch_geospatial_index_t index, usearch_error_t* error);

/** Optional per-call settings of @c usearch_geospatial_add and
 *  @c usearch_geospatial_refine - a pure input, never written. Passing @c NULL derives
 *  every setting. */
USEARCH_EXPORT typedef struct usearch_geospatial_insertion_options_t {

    /** The @b optional candidate-pool breadth the build walk explores - the @c ef of
     *  the literature; zero derives. */
    size_t expansion_factor;
} usearch_geospatial_insertion_options_t;

/**
 *  @brief Optional per-call settings of @c usearch_geospatial_search and
 *      @c usearch_geospatial_search_if - a pure input, never written. Passing @c NULL derives
 *      every setting.
 *
 *  @c usearch_geospatial_within takes none: containment sizes its own pool from the cone
 *  graph's spanner bound, so a caller-chosen breadth would only make the report incomplete.
 */
USEARCH_EXPORT typedef struct usearch_geospatial_search_options_t {

    /** The @b optional candidate-pool breadth the walk explores - the @c ef of the
     *  literature; zero derives. */
    size_t expansion_factor;
} usearch_geospatial_search_options_t;

/**
 *  @brief What "close" means numerically for one search: the radius the walk around each
 *      query point enumerates, and the time window a match must fall in.
 *
 *  Zero @c metres means no radius was chosen: nearest ranks by distance alone, and every
 *  other mode doubles the radius until @c wanted keys survive its scoring. Zero @c seconds
 *  ignores time even on a timed layout. Position uncertainty widens @c metres at the
 *  walk by three combined deviations, so the candidate set stays a superset at 98.9 %
 *  confidence, and the scoring tightens back to the pair's own.
 */
USEARCH_EXPORT typedef struct usearch_geospatial_tolerance_t {

    /** Radius in metres; zero derives as described above. */
    double metres;

    /** Half-width of the time window in seconds; zero ignores time. */
    double seconds;
} usearch_geospatial_tolerance_t;

/** Traversal telemetry, shared by insertion and every search form - a pure output,
 *  fully overwritten by the call it is passed to. Passing @c NULL skips reporting. */
USEARCH_EXPORT typedef struct usearch_geospatial_traversal_stats_t {

    /** Graph members visited while resolving the batch. */
    size_t visited_members;

    /** Distances computed while resolving the batch, diversity re-selections included. */
    size_t computed_distances;

    /** Starvation restarts - re-seedings after the frontier drained prematurely,
     *  and a graph-connectivity health signal rather than a cost counter. */
    size_t restarts;
} usearch_geospatial_traversal_stats_t;

/**
 *  @brief Adds a tape of point sets under their keys - the native batch operation; a
 *      single set is a tape of one, and a single position is a set of one.
 *
 *  @param[inout] index The handle to be populated.
 *  @param[in] executor Lanes and scratch to fan out over; its kind must match the residency.
 *  @param[in] keys The identities to admit the point sets under, one per set; its count is the
 *      length of the tape.
 *  @param[in] point_sets Array of @c keys.count base pointers, one per set.
 *  @param[in] point_counts Array of @c keys.count point counts, index-aligned with @p point_sets.
 *  @param[in] deviations The @b optional array of @c keys.count base pointers to per-point
 *      position uncertainties in metres, index-aligned with @p point_sets; @c NULL records none.
 *  @param[in] options The @b optional per-call settings; @c NULL derives every setting.
 *  @param[in] progress The @b optional handler, called with entries admitted and the tape length.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return @c keys.count on success, zero on failure.
 *
 *  Set @c i points at `point_counts[i] * layout_width` coordinates, row-major, in the index's
 *  layout order - degrees for the angular axes, metres for altitude, seconds for time - and,
 *  when @p deviations is given, at `point_counts[i]` uncertainties in metres.
 *
 *  A batch is all-or-nothing. Every fallible step runs before the first entry identity is
 *  claimed, so a rejected batch publishes nothing - a cancelled one included, which reports
 *  @c usearch_error_cancelled_k. A second thread driving @p executor reports
 *  @c usearch_error_contended_k before any work begins.
 *
 *  A key already holding @c values_per_key entries rejects the whole batch with
 *  @c usearch_error_overflow_k, since a smaller retry cannot help.
 */
USEARCH_EXPORT size_t usearch_geospatial_add(                      //
    usearch_geospatial_index_t index, usearch_executor_t executor, //
    usearch_keys_view_t keys,                                      //
    double const* const* point_sets, size_t const* point_counts,   //
    double const* const* deviations,                               //
    usearch_geospatial_insertion_options_t const* options,         //
    usearch_progress_callback_t progress, void* progress_state,    //
    usearch_geospatial_traversal_stats_t* stats,                   //
    usearch_error_t* error);

/**
 *  @brief Re-selects the neighbors of every stored entry against the finished graph, improving
 *      edges that were chosen when the graph around them was still incomplete.
 *
 *  @param[inout] index The handle to be refined.
 *  @param[in] executor Lanes and scratch to fan out over; its kind must match the residency.
 *  @param[in] options The @b optional per-call settings; @c NULL derives every setting.
 *  @param[in] progress The @b optional handler, called with entries swept; @c NULL skips it.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return Number of entries whose edges changed; zero means the graph has settled.
 *
 *  This is a build step rather than a repair. A cheap insertion pass followed by one refinement
 *  reaches a better graph, in less total time, than a single expensive insertion pass - so a wide
 *  @c expansion_factor belongs here rather than in @c usearch_geospatial_add.
 *
 *  Cancelling is safe, which is what makes a whole-graph sweep affordable. Re-selecting an entry's
 *  edges is idempotent and claims no identity, so a stopped pass leaves the graph correct and
 *  merely less improved - nothing unwinds, and no @c usearch_error_cancelled_k is reported. The
 *  sweep resumes where it stopped, so calling under a time budget converges across calls.
 *
 *  Searches may run throughout. A read-only handle refuses with @c usearch_error_read_only_k, like
 *  every other mutating verb.
 */
USEARCH_EXPORT size_t usearch_geospatial_refine(                   //
    usearch_geospatial_index_t index, usearch_executor_t executor, //
    usearch_geospatial_insertion_options_t const* options,         //
    usearch_progress_callback_t progress, void* progress_state,    //
    usearch_geospatial_traversal_stats_t* stats,                   //
    usearch_error_t* error);

/**
 *  @brief Converts a query point set into the index's internal geometry, returning a
 *      handle reusable across searches of differing mode and axis selection.
 *
 *  @param[in] index The handle whose layout governs conversion.
 *  @param[in] points Base pointer to `point_count * layout_width` coordinates.
 *  @param[in] point_count Number of positions; one is legal and is the point query.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return A handle to the emplaced object, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_geospatial_emplaced_t usearch_geospatial_emplace( //
    usearch_geospatial_index_t index, double const* points, size_t point_count, usearch_error_t* error);

/** Frees an emplaced query object. The index it was emplaced against must outlive it. */
USEARCH_EXPORT void usearch_geospatial_release( //
    usearch_geospatial_emplaced_t emplaced, usearch_error_t* error);

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a batch of emplaced query
 *      objects under one scoring mode - the native batch operation, parallelized
 *      internally; one query is a batch of one.
 *
 *  @param[in] index The handle to be queried.
 *  @param[in] executor Lanes and scratch to fan out over; its kind must match the residency.
 *  @param[in] queries Array of @p queries_count emplaced query handles.
 *  @param[in] queries_count Number of emplaced queries in the batch.
 *  @param[in] mode What "close" means for this search.
 *  @param[in] axes Which axes participate; narrowing never invalidates the graph.
 *  @param[in] tolerance The @b optional radius and time window; @c NULL derives both as zero.
 *  @param[out] matches Where the run lands, sized for @p queries_count rows of @c matches->wanted.
 *  @param[in] options The @b optional per-call settings; @c NULL derives every setting.
 *  @param[in] progress The @b optional handler, called with queries resolved and their count.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *
 *  Scores are metres, except overlap, which is a count of query points: route scores are
 *  leash lengths, the rest are separations.
 *
 *  A cancelled search reports @c usearch_error_cancelled_k and leaves every output row
 *  indeterminate. A second thread driving @p executor reports @c usearch_error_contended_k
 *  before any work begins.
 */
USEARCH_EXPORT void usearch_geospatial_search(                          //
    usearch_geospatial_index_t index, usearch_executor_t executor,      //
    usearch_geospatial_emplaced_t const* queries, size_t queries_count, //
    usearch_geospatial_mode_t mode, usearch_geospatial_axes_t axes,     //
    usearch_geospatial_tolerance_t const* tolerance,                    //
    usearch_matches_span_t const* matches,                              //
    usearch_geospatial_search_options_t const* options,                 //
    usearch_progress_callback_t progress, void* progress_state,         //
    usearch_geospatial_traversal_stats_t* stats,                        //
    usearch_error_t* error);

/**
 *  @brief Reports every entry scoring within the tolerance of one emplaced query - the
 *      containment counterpart of the ranked search, unbounded in result count.
 *
 *  Complete on the converged graph: the walk widens its pool by the cone graph's spanner
 *  bound before trusting it, so every candidate inside the tolerance is enumerated, at
 *  about thirty times the cost of the ball it reports. A tolerance of zero metres is
 *  rejected here, since containment needs a radius.
 *
 *  @param[in] index The handle to be queried.
 *  @param[in] executor Lanes and scratch to fan out over; its kind must match the residency.
 *  @param[in] query The emplaced query the containment is measured from.
 *  @param[in] mode What "close" means for this walk.
 *  @param[in] axes Which axes participate; narrowing never invalidates the graph.
 *  @param[in] tolerance The radius and time window; @c metres must be positive.
 *  @param[in] progress The @b optional handler, called with entries written and @c keys.capacity.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] keys Where the identities land; its capacity bounds the walk's report.
 *  @param[out] distances Output array of as many scores, positionally paired with @p keys.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return Number of entries written, saturating at @c keys.capacity.
 *
 *  A cancelled walk reports @c usearch_error_cancelled_k and leaves every output slot
 *  indeterminate. A second thread driving @p executor reports @c usearch_error_contended_k
 *  before any work begins.
 */
USEARCH_EXPORT size_t usearch_geospatial_within(                    //
    usearch_geospatial_index_t index, usearch_executor_t executor,  //
    usearch_geospatial_emplaced_t query,                            //
    usearch_geospatial_mode_t mode, usearch_geospatial_axes_t axes, //
    usearch_geospatial_tolerance_t const* tolerance,                //
    usearch_progress_callback_t progress, void* progress_state,     //
    usearch_keys_span_t keys, usearch_distance_t* distances,        //
    usearch_error_t* error);

/**
 *  @brief The filtered sibling of @c usearch_geospatial_search: identical batch shape,
 *      admitting only entries the @p predicate approves. Every parameter it shares carries
 *      the meaning documented there.
 *
 *  @param[in] predicate Per-key admission callback; returns non-zero to admit.
 *  @param[in] predicate_state Opaque pointer forwarded into every @p predicate call.
 *
 *  Non-matching entries still route the traversal, so recall does not collapse at low
 *  selectivity.
 */
USEARCH_EXPORT void usearch_geospatial_search_if(                       //
    usearch_geospatial_index_t index, usearch_executor_t executor,      //
    usearch_geospatial_emplaced_t const* queries, size_t queries_count, //
    usearch_geospatial_mode_t mode, usearch_geospatial_axes_t axes,     //
    usearch_geospatial_tolerance_t const* tolerance,                    //
    usearch_matches_span_t const* matches,                              //
    usearch_geospatial_search_options_t const* options,                 //
    usearch_predicate_t predicate, void* predicate_state,               //
    usearch_progress_callback_t progress, void* progress_state,         //
    usearch_geospatial_traversal_stats_t* stats,                        //
    usearch_error_t* error);

/** What one key holds, enough to size a buffer before retrieving any of it. */
USEARCH_EXPORT typedef struct usearch_geospatial_key_stats_t {

    /** Entries sharing the key; keys need not be unique. */
    size_t entries;

    /** Positions summed across those entries. */
    size_t total_points;

    /** Positions of the longest single entry - the per-entry stride one @c get needs. */
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
 *  @param[out] stats The @b optional extent of the key's payload; @c NULL asks for the count.
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
 *  @param[out] point_sets Destination for up to @p wanted entries; @c NULL probes the count.
 *  @param[in] wanted Upper bound on entries to write - the entry capacity of @p point_sets.
 *  @param[in] stride Number of bytes between starts of consecutive entries in @p point_sets.
 *  @param[out] point_counts The @b optional per-entry position counts; @c NULL skips it.
 *  @return The @b true number of entries under @p key; zero when absent.
 *
 *  The return is never the number written. A value above @p wanted means the copy was
 *  truncated, so allocate that many and call again.
 */
USEARCH_EXPORT size_t usearch_geospatial_get(                               //
    usearch_geospatial_index_t index, usearch_key_t key,                    //
    double* point_sets, size_t wanted, size_t stride, size_t* point_counts, //
    usearch_error_t* error);

/**
 *  @brief Absorbs every handle in @p donors into @p index - cheaper than re-inserting their
 *      point sets one by one, wiring cross-links via landmark stitching. Donor handles are
 *      consumed and must not be used afterwards, not even for @c usearch_geospatial_free.
 *
 *  @param[inout] index The handle absorbing the donors.
 *  @param[in] executor The threads and scratch the merge fans out over, one donor per thread.
 *  @param[in] donors Array of @p donors_count handles, every one of them consumed.
 *  @param[in] donors_count Number of entries in @p donors.
 *  @param[in] progress The @b optional handler, called with donors absorbed and their count.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *
 *  Readers of @p index may keep searching throughout, since unpublished entries hide behind
 *  the watermark. Absorbing a group costs the wall clock of its largest source rather than
 *  the sum of them, because the threads advance in parallel.
 *
 *  Cancellation stops @b between donors, which makes this the one verb where it is not
 *  all-or-nothing: donors already absorbed stay absorbed, and the rest stay live handles the
 *  caller still owns and must free.
 */
USEARCH_EXPORT void usearch_geospatial_merge(                      //
    usearch_geospatial_index_t index, usearch_executor_t executor, //
    usearch_geospatial_index_t const* donors, size_t donors_count, //
    usearch_progress_callback_t progress, void* progress_state,    //
    usearch_error_t* error);

/**
 *  @brief Opens a snapshot cursor over the current contents; see
 *      @c usearch_geospatial_cursor_t.
 *  @return The cursor, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_geospatial_cursor_t usearch_geospatial_cursor(usearch_geospatial_index_t index,
                                                                     usearch_error_t* error);

/**
 *  @brief Copies the next batch of entries out of the cursor.
 *  @param[inout] cursor The cursor to advance.
 *  @param[out] keys Where the identities land; its capacity bounds the batch.
 *  @param[out] point_sets Output array of up to @c keys.capacity borrowed base pointers; @c NULL skips.
 *  @param[out] point_counts Output array of up to @c keys.capacity position counts.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return Number of entries written; zero when the snapshot is exhausted.
 *
 *  The borrowed pointers reach into the index's arena and stay valid until the cursor moves
 *  or closes. Their coordinates are the internal Earth-fixed integer centimetres, three per
 *  point, not the geodetic degrees inserted.
 */
USEARCH_EXPORT size_t usearch_geospatial_cursor_next(             //
    usearch_geospatial_cursor_t cursor, usearch_keys_span_t keys, //
    int32_t const** point_sets, size_t* point_counts, usearch_error_t* error);

/** Releases the cursor. */
USEARCH_EXPORT void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t cursor);

/** Bytes one monolith takes under the default layout; a manifest's parts sum to about the same. */
USEARCH_EXPORT size_t usearch_geospatial_storage_requirements(usearch_geospatial_index_t index, usearch_error_t* error);

/**
 *  @brief Saves the index under @p path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  @param[in] index The handle to be written.
 *  @param[in] executor The threads and scratch the write fans out over, one shard per thread.
 *  @param[in] path Destination path, whose extension picks the layout.
 *  @param[in] progress The @b optional handler, called with parts written and their count.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *
 *  A cancelled save leaves the parts it already wrote and no manifest referencing them, so the
 *  checkpoint reads as absent rather than half-valid. Those leftovers are the caller's to remove.
 */
USEARCH_EXPORT void usearch_geospatial_save(                       //
    usearch_geospatial_index_t index, usearch_executor_t executor, //
    usearch_bytes_view_t path,                                     //
    usearch_progress_callback_t progress, void* progress_state,    //
    usearch_error_t* error);

/**
 *  @brief Loads a checkpoint into an initialized handle, with the handle's residency deciding the
 *      behavior: materialize into RAM, upload to the device, or open in place on disk - the
 *      read-only form, where mutating verbs fail.
 *
 *  A handle whose geometry was left zeroed freezes it here, from the checkpoint's own headers.
 *
 *  @param[inout] index The handle to be populated.
 *  @param[in] executor The threads and scratch the read fans out over, one shard per thread.
 *  @param[in] path Checkpoint path - a monolith, or a manifest naming many.
 *  @param[in] progress The @b optional handler, called with parts placed and their count.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *
 *  Where the parts end up is the handle's own memory domains, never the manifest's. A part
 *  carries its own length in its header, so a second load skips whatever already arrived
 *  whole and cancelling costs the part in flight rather than everything behind it.
 */
USEARCH_EXPORT void usearch_geospatial_load(                       //
    usearch_geospatial_index_t index, usearch_executor_t executor, //
    usearch_bytes_view_t path,                                     //
    usearch_progress_callback_t progress, void* progress_state,    //
    usearch_error_t* error);

/** Reads a checkpoint's geometry & metric metadata into @p options without loading it. */
USEARCH_EXPORT void usearch_geospatial_metadata( //
    usearch_bytes_view_t path, usearch_geospatial_options_t* options, usearch_error_t* error);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_GEOSPATIAL_INDEX_H
