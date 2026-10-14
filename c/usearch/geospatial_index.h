/**
 *  @file c/usearch/geospatial_index.h
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
 *  @b tape - every set's points end to end, and @c count + 1 Apache Arrow offsets counting
 *  points around them - rather than the dense modality's run of rows. Coordinates are numeric
 *  rather than textual, so the elements are points of the layout's width rather than the
 *  sequence modality's bytes.
 *
 *  Handles live in RAM or VRAM, created empty by @c usearch_geospatial_init_ram or
 *  @c usearch_geospatial_init_vram, copied from a checkpoint by @c usearch_geospatial_load_ram
 *  or @c usearch_geospatial_load_vram, or served from one in place by
 *  @c usearch_geospatial_view_ram, which is read-only and refuses every mutating verb with
 *  @c usearch_status_read_only_k.
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
 *  order. A VRAM handle is driven through the streams of its executor. A viewed handle admits
 *  arbitrarily many concurrent searches, having nothing to mutate. Serialization is the one
 *  exception: saving demands a mutation-free window, though concurrent readers keep
 *  searching throughout. An emplaced query object is @b not internally synchronized - build
 *  one per thread, or confine it to the thread that emplaced it. Per-call outputs - stats,
 *  counts, the outcome - belong to their call and must not be shared between calls that are not final.
 *
 *  An insertion is all-or-nothing per call, and on a RAM handle its entries become visible at
 *  once, so a concurrent search sees every entry of the call or none of them. Their keys land
 *  right after, so a key lookup only finds entries a search already sees.
 *
 *  Entries are permanent. No verb removes one, so an index only ever grows until it is
 *  cleared wholesale, and a key's entries accumulate up to the @c values_per_key bound its
 *  creation froze.
 *
 *  The executor, not the handle, is what orders: a handle admits any number of concurrent
 *  callers, while calls on one executor run in the order they were submitted. Device inputs are
 *  read in place and only pageable host inputs pass through the executor's pinned bounce buffer,
 *  as the staging rules beside @c usearch_executor_t describe.
 */
#ifndef UNUM_USEARCH_GEOSPATIAL_INDEX_H
#define UNUM_USEARCH_GEOSPATIAL_INDEX_H

#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Similarity Search Index over point sets in space & time. */
typedef struct usearch_geospatial_index_handle_t* usearch_geospatial_index_t;

/**
 *  @brief A query point set converted into the index's internal geometry once and reused across
 *      searches.
 *
 *  Emplacing is the expensive half of a query - the geodetic conversion, the time scaling, and any
 *  per-set precomputation the metric needs - so a caller asking several questions of the same
 *  object pays it once.
 *
 *  @see usearch_geospatial_emplace To build one.
 *  @see usearch_geospatial_emplaced_free To dispose of one.
 */
typedef struct usearch_geospatial_emplaced_handle_t* usearch_geospatial_emplaced_t;

/** A consistent snapshot cursor over the index contents, capturing the publication
 *  watermark at creation - concurrent inserts never appear mid-iteration.
 *  Each pull names the index again; an overlapping pull reports @c usearch_status_contended_k. */
typedef struct usearch_geospatial_cursor_handle_t* usearch_geospatial_cursor_t;

/** Which axes each incoming position carries, and in which order. Fixed for the
 *  lifetime of the index: every point set must present the same width. */
typedef enum usearch_geospatial_layout_t {
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
typedef enum usearch_geospatial_structure_t {
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
typedef enum usearch_geospatial_axes_t {
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
typedef enum usearch_geospatial_mode_t {
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

/** The arguments of @c usearch_geospatial_init_ram. */
typedef struct usearch_geospatial_init_ram_t {

    /** Output for the new index. */
    usearch_geospatial_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Which axes incoming positions carry. */
    usearch_geospatial_layout_t layout;

    /** What ordering within a point set means; paths carry a successor per point. */
    usearch_geospatial_structure_t structure;

    /** The @b optional connections-per-node bound; zero derives to twelve - eight
     *  tangent-plane cones and four long-range links. */
    size_t connectivity;

    /** Upper bound on points sharing one key; zero derives to one - unique keys. */
    size_t values_per_key;

    /** The memory domains to place one shard on apiece. */
    usearch_ram_options_t ram;
} usearch_geospatial_init_ram_t;

/** Creates an empty, mutable geospatial index in host memory. */
USEARCH_EXPORT void usearch_geospatial_init_ram(usearch_geospatial_init_ram_t* arguments);

/** The arguments of @c usearch_geospatial_init_vram, which match
 *  @c usearch_geospatial_init_ram_t but for the placement. */
typedef struct usearch_geospatial_init_vram_t {

    /** Output for the new index. */
    usearch_geospatial_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Which axes incoming positions carry. */
    usearch_geospatial_layout_t layout;

    /** What ordering within a point set means. */
    usearch_geospatial_structure_t structure;

    /** The @b optional connections-per-node bound; zero derives to twelve. */
    size_t connectivity;

    /** Upper bound on points sharing one key; zero derives to one - unique keys. */
    size_t values_per_key;

    /** The devices to place one shard on apiece. */
    usearch_vram_options_t vram;
} usearch_geospatial_init_vram_t;

/** Creates an empty, mutable geospatial index in device memory. */
USEARCH_EXPORT void usearch_geospatial_init_vram(usearch_geospatial_init_vram_t* arguments);

/** The arguments of @c usearch_geospatial_load_ram. */
typedef struct usearch_geospatial_load_ram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_geospatial_index_t* index;

    /** The executor the read fans out over, one shard per thread. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** The memory domains to place the parts on. */
    usearch_ram_options_t ram;

    /** Upper bound on points sharing one key, which no checkpoint records; zero derives to one. */
    size_t values_per_key;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;
} usearch_geospatial_load_ram_t;

/**
 *  @brief Copies a checkpoint into a new, mutable index in host memory, its geometry read from
 *      the checkpoint's own headers.
 *
 *  Where the parts end up is the call's own memory domains, never the manifest's. A part
 *  carries its own length in its header, so a second load skips whatever already arrived
 *  whole and cancelling costs the part being read rather than everything behind it.
 */
USEARCH_EXPORT void usearch_geospatial_load_ram(usearch_geospatial_load_ram_t* arguments);

/** The arguments of @c usearch_geospatial_load_vram. */
typedef struct usearch_geospatial_load_vram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_geospatial_index_t* index;

    /** The executor the upload runs on, one stream per device. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** The devices to place the parts on. */
    usearch_vram_options_t vram;

    /** Upper bound on points sharing one key, which no checkpoint records; zero derives to one. */
    size_t values_per_key;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;
} usearch_geospatial_load_vram_t;

/** Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
 *  @c usearch_geospatial_load_ram. */
USEARCH_EXPORT void usearch_geospatial_load_vram(usearch_geospatial_load_vram_t* arguments);

/** The arguments of @c usearch_geospatial_view_ram. */
typedef struct usearch_geospatial_view_ram_t {

    /** Output for the new, read-only index. */
    usearch_geospatial_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;
} usearch_geospatial_view_ram_t;

/** Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
 *  verbs report @c usearch_status_read_only_k. */
USEARCH_EXPORT void usearch_geospatial_view_ram(usearch_geospatial_view_ram_t* arguments);

/** Frees the resources associated with the index; @c NULL is safe, and it cannot fail, so takes
 *  no outcome.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_geospatial_free(usearch_geospatial_index_t index);

/** The arguments of @c usearch_geospatial_reserve. */
typedef struct usearch_geospatial_reserve_t {

    /** The handle to be resized. */
    usearch_geospatial_index_t index;

    /** The executor the growth runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The desired total entry capacity, including the current size. */
    size_t entries;

    /** Expected total point volume across all sets; zero grows lazily. */
    size_t points;
} usearch_geospatial_reserve_t;

/** Reserves capacity ahead of insertion. */
USEARCH_EXPORT void usearch_geospatial_reserve(usearch_geospatial_reserve_t* arguments);

/** The arguments of @c usearch_geospatial_clear. */
typedef struct usearch_geospatial_clear_t {

    /** The handle to be cleared. */
    usearch_geospatial_index_t index;

    /** The executor the erasure runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;
} usearch_geospatial_clear_t;

/** Erases all entries, keeping the reserved capacity. */
USEARCH_EXPORT void usearch_geospatial_clear(usearch_geospatial_clear_t* arguments);

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
typedef struct usearch_geospatial_tolerance_t {

    /** Radius in metres; zero derives as described above. */
    double metres;

    /** Half-width of the time window in seconds; zero ignores time. */
    double seconds;
} usearch_geospatial_tolerance_t;

/** Traversal telemetry, shared by insertion and every search form - a pure output,
 *  fully overwritten by the call it is passed to. Passing @c NULL skips reporting. */
typedef struct usearch_geospatial_traversal_stats_t {

    /** Graph members visited while resolving the batch. */
    size_t visited_members;

    /** Distances computed while resolving the batch, diversity re-selections included. */
    size_t computed_distances;

    /** Starvation restarts - re-seedings after the frontier drained prematurely,
     *  and a graph-connectivity health signal rather than a cost counter. */
    size_t restarts;
} usearch_geospatial_traversal_stats_t;

/** The arguments of @c usearch_geospatial_insert. */
typedef struct usearch_geospatial_insert_t {

    /** The handle to be populated. */
    usearch_geospatial_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Point sets to admit - the batch axis of @c keys and @c points. */
    size_t count;

    /** The identities to admit the point sets under, one per set. */
    usearch_keys_view_t keys;

    /** The point sets, as @c double coordinates end to end around @c count + 1 point offsets. */
    usearch_tape_view_t points;

    /** The @b optional per-point position uncertainties in metres, one per point of @c points
     *  and addressed by the same offsets; @c NULL records none. */
    double const* deviations;

    /** The @b optional candidate-pool breadth the build walk explores - the @c ef of
     *  the literature; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries admitted and the batch length. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_geospatial_traversal_stats_t* stats;
} usearch_geospatial_insert_t;

/**
 *  @brief Adds a tape of point sets under their keys - the native batch operation; a
 *      single set is a tape of one, and a single position is a set of one.
 *  @retval usearch_status_read_only_k The handle is a view.
 *  @retval usearch_status_overflow_k A key already holds @c values_per_key entries.
 *
 *  A point is @c layout_width coordinates, row-major, in the index's layout order - degrees for
 *  the angular axes, metres for altitude, seconds for time.
 *
 *  An insertion is all-or-nothing per call. Every fallible step runs before the first entry
 *  identity is claimed, so a rejected call publishes nothing - a cancelled one included, which
 *  reports @c usearch_status_cancelled_k - and on a RAM handle the entries of an accepted one
 *  become visible at once, their keys right after.
 *
 *  A key already holding @c values_per_key entries rejects the whole call, since a smaller retry
 *  cannot help.
 */
USEARCH_EXPORT void usearch_geospatial_insert(usearch_geospatial_insert_t* arguments);

/** The arguments of @c usearch_geospatial_refine. */
typedef struct usearch_geospatial_refine_t {

    /** The handle to be refined. */
    usearch_geospatial_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The @b optional output for the number of entries whose edges changed; zero means settled. */
    size_t* changed;

    /** The @b optional candidate-pool breadth the re-selection walk explores; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries swept; @c NULL skips it. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_geospatial_traversal_stats_t* stats;
} usearch_geospatial_refine_t;

/**
 *  @brief Re-selects the neighbors of every stored entry against the finished graph, improving
 *      edges that were chosen when the graph around them was still incomplete.
 *
 *  This is a build step rather than a repair. A cheap insertion pass followed by one refinement
 *  reaches a better graph, in less total time, than a single expensive insertion pass - so a wide
 *  @c expansion_factor belongs here rather than in @c usearch_geospatial_insert.
 *
 *  Cancelling is safe, which is what makes a whole-graph sweep affordable. Re-selecting an entry's
 *  edges is idempotent and claims no identity, so a stopped pass leaves the graph correct and
 *  merely less improved - nothing unwinds, and no @c usearch_status_cancelled_k is reported. The
 *  sweep resumes where it stopped, so calling under a time budget converges across calls.
 *
 *  Searches may run throughout. A viewed handle refuses with @c usearch_status_read_only_k, like
 *  every other mutating verb.
 */
USEARCH_EXPORT void usearch_geospatial_refine(usearch_geospatial_refine_t* arguments);

/** The arguments of @c usearch_geospatial_emplace. */
typedef struct usearch_geospatial_emplace_t {

    /** Output for the emplaced query; @c NULL on failure. */
    usearch_geospatial_emplaced_t* emplaced;

    /** The handle whose layout governs conversion, borrowed until the emplaced query is freed. */
    usearch_geospatial_index_t index;

    /** The executor the conversion runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Base pointer to `point_count * layout_width` coordinates. */
    double const* points;

    /** Number of positions; one is legal and is the point query. */
    size_t point_count;
} usearch_geospatial_emplace_t;

/** Converts a query point set into the index's internal geometry, as a handle reusable across
 *  searches of differing mode and axis selection. */
USEARCH_EXPORT void usearch_geospatial_emplace(usearch_geospatial_emplace_t* arguments);

/** Frees an emplaced query object; @c NULL is safe, and it cannot fail, so takes no outcome. The
 *  index it was emplaced against must outlive it. */
USEARCH_EXPORT void usearch_geospatial_emplaced_free(usearch_geospatial_emplaced_t emplaced);

/** The arguments of @c usearch_geospatial_search. */
typedef struct usearch_geospatial_search_t {

    /** The handle to be queried. */
    usearch_geospatial_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Emplaced queries to resolve; the outputs have @c count rows of @c wanted entries. */
    size_t count;

    /** Array of @c count emplaced query handles. */
    usearch_geospatial_emplaced_t const* queries;

    /** What "close" means for this search. */
    usearch_geospatial_mode_t mode;

    /** Which axes participate; narrowing never invalidates the graph. */
    usearch_geospatial_axes_t axes;

    /** The radius and time window; zeroed fields derive as the type describes. */
    usearch_geospatial_tolerance_t tolerance;

    /** Output for the run, inside @c arena. */
    usearch_matches_view_t* matches;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /** Columns per row, and the upper bound on matches - the "k" in "kANN". */
    size_t wanted;

    /** The @b optional candidate-pool breadth the walk explores - the @c ef of the
     *  literature; zero derives. */
    size_t expansion_factor;

    /** The @b optional per-key admission callback; @c NULL admits everything. */
    usearch_predicate_t predicate;

    /** Opaque pointer forwarded into every @c predicate call. */
    void* predicate_state;

    /** The @b optional handler, called with queries resolved and their count. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_geospatial_traversal_stats_t* stats;
} usearch_geospatial_search_t;

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a batch of emplaced query
 *      objects under one scoring mode - the native batch operation, parallelized
 *      internally; one query is a batch of one.
 *
 *  Scores are metres, except overlap, which is a count of query points: route scores are
 *  leash lengths, the rest are separations.
 *
 *  A @c predicate narrows the matches to the entries it admits. Non-matching entries still route
 *  the traversal, so recall does not collapse at low selectivity.
 *
 *  A cancelled search reports @c usearch_status_cancelled_k and leaves @c matches @c NULL.
 */
USEARCH_EXPORT void usearch_geospatial_search(usearch_geospatial_search_t* arguments);

/** The arguments of @c usearch_geospatial_within. */
typedef struct usearch_geospatial_within_t {

    /** The handle to be queried. */
    usearch_geospatial_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Emplaced queries to resolve, one walk apiece. */
    size_t count;

    /** Array of @c count emplaced query handles. */
    usearch_geospatial_emplaced_t const* queries;

    /** What "close" means for these walks. */
    usearch_geospatial_mode_t mode;

    /** Which axes participate; narrowing never invalidates the graph. */
    usearch_geospatial_axes_t axes;

    /** The radius and time window; @c metres must be positive. */
    usearch_geospatial_tolerance_t tolerance;

    /** Output for the report inside @c arena, query @c i spanning entries @c matches->offsets[i] up
     *  to @c matches->offsets[i+1]. */
    usearch_matches_tape_view_t* matches;

    /** The @b optional output for the entries written across every query. */
    size_t* written;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /** The @b optional per-key admission callback; @c NULL admits everything. */
    usearch_predicate_t predicate;

    /** Opaque pointer forwarded into every @c predicate call. */
    void* predicate_state;

    /** The @b optional handler, called with queries resolved and their count. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_geospatial_traversal_stats_t* stats;
} usearch_geospatial_within_t;

/**
 *  @brief Reports every entry scoring within the tolerance of each emplaced query - the
 *      containment counterpart of the ranked search, unbounded in result count.
 *  @retval usearch_status_invalid_argument_k A tolerance of zero metres, since containment needs a
 *      radius.
 *
 *  Complete on the converged graph: the walk widens its pool by the cone graph's spanner
 *  bound before trusting it, so every candidate inside the tolerance is enumerated, at
 *  about thirty times the cost of the ball it reports. Containment sizes its own pool from that
 *  bound, so it takes no breadth a caller could make the report incomplete with. Only the walks
 *  learn how large the report is, so the keys, scores and Apache Arrow offsets land in @c arena as
 *  one ragged output, as @c usearch_dense_multiply lands its pairs. A @c predicate narrows the
 *  report without stopping the walk from crossing the entries it rejects.
 *
 *  A cancelled call reports @c usearch_status_cancelled_k and leaves the outputs @c NULL.
 */
USEARCH_EXPORT void usearch_geospatial_within(usearch_geospatial_within_t* arguments);

/** What one key holds, enough to size a buffer before retrieving any of it. */
typedef struct usearch_geospatial_key_stats_t {

    /** Entries sharing the key; keys need not be unique. */
    size_t entries;

    /** Positions summed across those entries. */
    size_t total_points;

    /** Positions of the longest single entry. */
    size_t largest_points;
} usearch_geospatial_key_stats_t;

/** The arguments of @c usearch_geospatial_count. */
typedef struct usearch_geospatial_count_t {

    /** The handle to be queried. */
    usearch_geospatial_index_t index;

    /** The executor the lookup runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are counted. */
    usearch_key_t key;

    /** Output for what @c key holds; zero @c entries means the index holds none. */
    usearch_geospatial_key_stats_t* key_stats;
} usearch_geospatial_count_t;

/** Counts the entries stored under one key, with the extent of their points. */
USEARCH_EXPORT void usearch_geospatial_count(usearch_geospatial_count_t* arguments);

/** The arguments of @c usearch_geospatial_get. */
typedef struct usearch_geospatial_get_t {

    /** The handle to be queried. */
    usearch_geospatial_index_t index;

    /** The executor whose stream or thread copies the points; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are exported. */
    usearch_key_t key;

    /** Output for the tape inside @c arena: every set's @c double coordinates end to end, as many
     *  per point as the frozen layout carries, around @c written + 1 point offsets. */
    usearch_tape_view_t* points;

    /** The @b optional output for the entries written; zero when the key is absent. */
    size_t* written;

    /** Memory for @c points, rewound by the call. */
    usearch_arena_t arena;
} usearch_geospatial_get_t;

/**
 *  @brief Retrieves every point set stored under one key as an Apache Arrow tape, as the geodetic
 *      degrees, metres, and seconds they were inserted as, in insertion order.
 *
 *  An absent key leaves the tape's pointers @c NULL.
 */
USEARCH_EXPORT void usearch_geospatial_get(usearch_geospatial_get_t* arguments);

/** The arguments of @c usearch_geospatial_merge. */
typedef struct usearch_geospatial_merge_t {

    /** The handle absorbing the donors. */
    usearch_geospatial_index_t index;

    /** The executor the merge fans out over, one donor per thread. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Donor handles to absorb. */
    size_t count;

    /** Array of @c count handles, every one of them consumed and nulled as it goes. */
    usearch_geospatial_index_t* donors;

    /** The @b optional handler, called with donors absorbed and their count. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;
} usearch_geospatial_merge_t;

/**
 *  @brief Absorbs every handle in @c donors into @c index - cheaper than re-inserting their
 *      point sets one by one, wiring cross-links via landmark stitching. Donor handles are
 *      consumed and must not be used afterwards, not even for @c usearch_geospatial_free.
 *
 *  Readers of @c index may keep searching throughout, since unpublished entries hide behind
 *  the watermark. Absorbing a group costs the wall clock of its largest source rather than
 *  the sum of them, because the threads advance in parallel.
 *
 *  Cancellation stops @b between donors, which makes this the one verb where it is not
 *  all-or-nothing: donors already absorbed stay absorbed, and the rest stay live handles the
 *  caller still owns and must free. Every consumed slot is nulled as it goes, so the array itself
 *  says which handles survived.
 */
USEARCH_EXPORT void usearch_geospatial_merge(usearch_geospatial_merge_t* arguments);

/** The arguments of @c usearch_geospatial_cursor_init. */
typedef struct usearch_geospatial_cursor_init_t {

    /** Output for the new cursor; @c NULL on failure. */
    usearch_geospatial_cursor_t* cursor;

    /** The handle to be scanned. */
    usearch_geospatial_index_t index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;
} usearch_geospatial_cursor_init_t;

/** Opens a snapshot cursor over the current contents; see @c usearch_geospatial_cursor_t. */
USEARCH_EXPORT void usearch_geospatial_cursor_init(usearch_geospatial_cursor_init_t* arguments);

/** The arguments of @c usearch_geospatial_cursor_next. */
typedef struct usearch_geospatial_cursor_next_t {

    /** The handle the cursor was opened on. */
    usearch_geospatial_index_t index;

    /** The executor the pull runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The cursor to advance. */
    usearch_geospatial_cursor_t cursor;

    /** Output for the identities, inside @c arena. */
    usearch_keys_view_t* keys;

    /** The @b optional output for the point sets as a tape inside @c arena, @c int32_t coordinates
     *  around @c written + 1 point offsets; @c NULL skips them. */
    usearch_tape_view_t* points;

    /** Output for the entries written; zero when the snapshot is exhausted. */
    size_t* written;

    /** Memory for @c keys and @c points, rewound by the call. */
    usearch_arena_t arena;

    /** The upper bound on entries this pull writes. */
    size_t wanted;
} usearch_geospatial_cursor_next_t;

/**
 *  @brief Copies the next batch of entries out of the cursor into @c arena.
 *  @retval usearch_status_invalid_argument_k @c index is not the handle the cursor was opened on.
 *
 *  The batch stays valid until the next call given the same arena, however the cursor moves
 *  meanwhile. Its coordinates are the internal Earth-fixed integer centimetres, three per point,
 *  not the geodetic degrees inserted.
 */
USEARCH_EXPORT void usearch_geospatial_cursor_next(usearch_geospatial_cursor_next_t* arguments);

/** Releases the cursor; @c NULL is safe.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t cursor);

/** The arguments of @c usearch_geospatial_save. */
typedef struct usearch_geospatial_save_t {

    /** The handle to be written. */
    usearch_geospatial_index_t index;

    /** The executor the write fans out over, one shard per thread or stream. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Destination path, whose extension picks the layout. */
    usearch_bytes_view_t path;

    /** The @b optional handler, called with parts written and their count. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;
} usearch_geospatial_save_t;

/**
 *  @brief Saves a mutable index under @c path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  A cancelled save leaves the parts it already wrote and no manifest referencing them, so the
 *  checkpoint reads as absent rather than half-valid. Those leftovers are the caller's to remove.
 */
USEARCH_EXPORT void usearch_geospatial_save(usearch_geospatial_save_t* arguments);

/** The layout and geometry a checkpoint froze - everything a reader learns without opening it. */
typedef struct usearch_geospatial_metadata_t {

    /** Which axes its positions carry. */
    usearch_geospatial_layout_t layout;

    /** What ordering within its point sets means. */
    usearch_geospatial_structure_t structure;

    /** Connections per node the graph was built with. */
    size_t connectivity;
} usearch_geospatial_metadata_t;

/** The arguments of @c usearch_geospatial_metadata_read. */
typedef struct usearch_geospatial_metadata_read_t {

    /** The executor the read runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** Output for every field the checkpoint froze. */
    usearch_geospatial_metadata_t* metadata;
} usearch_geospatial_metadata_read_t;

/** Reads a checkpoint's layout & geometry without loading it. */
USEARCH_EXPORT void usearch_geospatial_metadata_read(usearch_geospatial_metadata_read_t* arguments);

/** A live index's state. Each counter is read once, so under concurrent insertions they may come
 *  from slightly different moments. */
typedef struct usearch_geospatial_index_description_t {

    /** The frozen layout, structure and connectivity, as a checkpoint reports them. */
    usearch_geospatial_metadata_t metadata;

    /** Stored entries. */
    size_t size;

    /** Entries hosted without growing again. */
    size_t capacity;

    /** Bytes the handle holds, point storage included; traversal scratch is the executor's. */
    size_t memory_usage;

    /** Static codename of the kernels the index dispatched to on this machine. */
    char const* capabilities;
} usearch_geospatial_index_description_t;

/** The arguments of @c usearch_geospatial_describe. */
typedef struct usearch_geospatial_describe_t {

    /** The handle to be described. */
    usearch_geospatial_index_t index;

    /** The executor the report runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output, filled whole. */
    usearch_geospatial_index_description_t* description;
} usearch_geospatial_describe_t;

/** Reports a live index's frozen metadata and current counters in one call. */
USEARCH_EXPORT void usearch_geospatial_describe(usearch_geospatial_describe_t* arguments);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_GEOSPATIAL_INDEX_H
