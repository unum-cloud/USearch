/**
 *  @file c/usearch/geospatial_index.h
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The geospatial-modality C interface.
 *
 *  Similarity search over observations on and above the Earth: where a device was, how sure it
 *  was, which way it looked, how far it saw, and when - photos, video frames, drone footage,
 *  sensor fixes - under nearness, line of sight, and view-cone relations.
 *
 *  Every entry is one @b observation, a @c usearch_geospatial_observation_t of eleven doubles in
 *  natural units, and a key may hold several, up to the @c values_per_key bound its creation
 *  froze: a camera's frames, a vehicle's fixes, or a photo and the subject it shows. A device
 *  that recorded nothing for a field leaves it NaN, and each field documents where its value
 *  comes from in the metadata real devices write.
 *
 *  Handles live in RAM or VRAM, created empty by @c usearch_geospatial_init_ram or
 *  @c usearch_geospatial_init_vram, copied from a checkpoint by @c usearch_geospatial_load_ram
 *  or @c usearch_geospatial_load_vram, or served from one in place by
 *  @c usearch_geospatial_view_ram, which is read-only and refuses every mutating verb with
 *  @c usearch_status_read_only_k.
 *
 *  @section geospatial_index_geometry Internal Geometry
 *
 *  Positions arrive as geodetic degrees and metres above the WGS-84 ellipsoid and are stored as
 *  Earth-centred, Earth-fixed @b integer @b centimetres, altitude included, beside the rest of
 *  the observation quantized so that every rounding only widens a test. Every ordering inside the
 *  index is the straight-line @b chord between two positions, three subtractions and three wide
 *  multiply-adds with no rounding, and every score is that chord in metres. On a sphere the chord
 *  is a strictly increasing function of the great-circle arc, so the two rank identically; it
 *  reads 1 mm short of the surface geodesic at 10 km and 1 m short at 100 km, and callers wanting
 *  geodesics compute them for the reported matches.
 *
 *  @section geospatial_index_relations One Graph, Several Relations
 *
 *  One node is one observation. The graph is a single-layer navigable small world over the chord,
 *  wired by the nearest neighbor in each of eight tangent-plane cones plus four long-range links,
 *  and every relation is answered from one primitive: a walk around each query position, followed
 *  by tests on what it found. @c usearch_geospatial_relation_t picks the test: nearness widened by
 *  both positions' uncertainty, an unobstructed line of sight over the handle's terrain, or one
 *  side's view cone holding the other. Time intervals and the caller's predicate narrow every
 *  relation alike.
 *
 *  @section geospatial_index_terrain Terrain
 *
 *  A handle may test lines of sight over a terrain given at its creation, as a
 *  @c usearch_geospatial_terrain_t: a north-up grid of heights, borrowed in place or filled on
 *  demand by the caller. A checkpoint never carries it, so every load and view names it again.
 *  A sight line clears the terrain by the least gap between it and the heights it crosses, with
 *  the Earth's curvature and the refraction of light folded in.
 *
 *  @section geospatial_index_concurrency Concurrency & Thread Safety
 *
 *  A handle may be shared across threads. On a RAM handle, insertions and searches overlap freely:
 *  searches read only entries published behind the visibility watermark and never write to the
 *  graph, while insertions synchronize through per-node locks and publish in order. A VRAM handle
 *  is driven through the streams of its executor. A viewed handle admits arbitrarily many
 *  concurrent searches, having nothing to mutate. Serialization is the one exception: saving
 *  demands a mutation-free window, though concurrent readers keep searching throughout. A terrain
 *  callback is called concurrently from the executor's threads. Per-call outputs - stats, counts,
 *  the outcome - belong to their call and must not be shared between calls that are not final.
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

/** Similarity Search Index over observations in space & time. */
typedef struct usearch_geospatial_index_handle_t* usearch_geospatial_index_t;

/** A consistent snapshot cursor over the index contents, capturing the publication
 *  watermark at creation - concurrent inserts never appear mid-iteration.
 *  Each pull names the index again; an overlapping pull reports @c usearch_status_contended_k. */
typedef struct usearch_geospatial_cursor_handle_t* usearch_geospatial_cursor_t;

/**
 *  @brief One observation as devices record it: every field in its natural unit, and NaN wherever
 *      nothing was recorded.
 *
 *  An unknown accuracy counts as exact, an unknown view as looking everywhere without limit, and
 *  an unknown time as always, so a missing field only ever widens the tests that read it. The
 *  index stores a quantized copy that rounds every field outward, and reads it back through
 *  @c usearch_geospatial_get within that rounding: centimetres, a 2⁻¹⁶ turn of heading, a 2⁻⁸
 *  turn of pitch and field, whole metres of accuracy and range, and milliseconds.
 */
typedef struct usearch_geospatial_observation_t {

    /** Degrees on WGS-84, north and east positive. EXIF @c GPSLatitude and @c GPSLongitude with
     *  their @c Ref signs; QuickTime @c ISO6709; GoPro GPMF @c GPS5; MISB ST 0601 tags 13 and 14;
     *  ROS @c NavSatFix. */
    double latitude_degrees;
    double longitude_degrees;

    /** Metres above the WGS-84 ellipsoid. EXIF @c GPSAltitude means the sea level before Exif 3.0
     *  and the ellipsoid after, so older files need the geoid's height added; DJI XMP
     *  @c AbsoluteAltitude; MISB tag 75; ROS @c NavSatFix. NaN means on the ground: the handle's
     *  terrain when it has one, the ellipsoid otherwise. */
    double altitude_metres;

    /** One standard deviation of the horizontal position, in metres. EXIF
     *  @c GPSHPositioningError; QuickTime @c location.accuracy.horizontal; a phone's 68 %
     *  accuracy radius divided by 1.51; the ROS @c NavSatFix covariance. */
    double accuracy_metres;

    /** Where the device looked: degrees clockwise from true north. EXIF @c GPSImgDirection,
     *  corrected to true north when its @c Ref is magnetic; XMP @c GPano:PoseHeadingDegrees; DJI
     *  @c GimbalYawDegree; MISB tag 5 composed with tag 18; ROS @c Imu. */
    double heading_degrees;

    /** Degrees above the horizon. XMP @c GPano:PosePitchDegrees; DJI @c GimbalPitchDegree; MISB
     *  tag 6 composed with tag 19; ROS @c Imu. NaN tests the heading alone. */
    double pitch_degrees;

    /** One standard deviation of the heading, in degrees, which widens the field of view by
     *  three on each side: ROS @c orientation_covariance, or a compass's own. */
    double heading_accuracy_degrees;

    /** The full field of view in degrees: 2·atan(18 mm / EXIF @c FocalLengthIn35mmFilm) for a
     *  landscape frame; MISB tag 16; the ROS @c CameraInfo matrix. */
    double field_of_view_degrees;

    /** How far the device saw, in metres along the slant. EXIF @c SubjectDistance; MISB tag 21;
     *  a laser rangefinder; the farthest surface in the depth map of a HEIC or Dynamic Depth
     *  image; a LiDAR return. */
    double range_metres;

    /** The first and last instant the observation covers, in Unix seconds, UTC. EXIF
     *  @c DateTimeOriginal with @c OffsetTimeOriginal, or @c GPSDateStamp with @c GPSTimeStamp;
     *  a clip's creation date and duration; MISB tag 2; a ROS @c header.stamp. One NaN makes the
     *  observation an instant at the other. */
    double begin_seconds;
    double end_seconds;
} usearch_geospatial_observation_t;

/** Which entries count as matches of a query; the radius and time window of
 *  @c usearch_geospatial_tolerance_t narrow every relation alike. */
typedef enum usearch_geospatial_relation_t {
    usearch_geospatial_relation_unknown_k = 0,

    /** Within the radius of each other, widened by three standard deviations of both positions
     *  combined. */
    usearch_geospatial_relation_near_k = 1,

    /** Near, with a clear line of sight over the handle's terrain; refused by a handle without
     *  one. */
    usearch_geospatial_relation_visible_k = 2,

    /** Near, and the entry's view cone holds the query: "which cameras see this place". The
     *  sight line must also clear the handle's terrain when it has one. */
    usearch_geospatial_relation_sees_k = 3,

    /** Near, and the query's view cone holds the entry: "what does this camera see". The sight
     *  line must also clear the handle's terrain when it has one. */
    usearch_geospatial_relation_seen_k = 4,
} usearch_geospatial_relation_t;

/** Fills @p rows × @p columns heights of a terrain, in metres above the WGS-84 ellipsoid, starting
 *  at row @p first_row and column @p first_column, with @p heights_stride bytes between rows; a
 *  cell the source does not have stays NaN, which never blocks a sight line. Called concurrently
 *  from the executor's threads. */
typedef void (*usearch_geospatial_terrain_callback_t)(size_t first_row, size_t first_column, size_t rows,
                                                      size_t columns, float* heights, size_t heights_stride,
                                                      void* state);

/**
 *  @brief The elevation every line of sight is tested over: a north-up geographic grid of
 *      heights, given by exactly one of a borrowed buffer, a fill callback or a dataset path.
 *
 *  A zeroed terrain means none. Heights are metres above the WGS-84 ellipsoid, like observation
 *  altitudes, so orthometric grids such as the Copernicus DEM's need the geoid's height added
 *  first. The grid is treated as locally flat at the latitude of its middle row, which keeps
 *  distances within 1 % across a one-degree tile.
 */
typedef struct usearch_geospatial_terrain_t {

    /** The grid's extent: rows run south from the north edge, columns east from the west edge. */
    size_t rows;
    size_t columns;

    /** The centre of cell [0,0], in degrees. */
    double north_latitude_degrees;
    double west_longitude_degrees;

    /** The steps between rows and between columns, in degrees, both positive. */
    double row_step_degrees;
    double column_step_degrees;

    /** The whole grid, row-major, borrowed for the handle's lifetime. */
    float const* heights;

    /** The bytes between consecutive rows of @c heights; zero means packed. */
    size_t heights_stride;

    /** Or a handler filling windows of the grid on demand, called with @c state. */
    usearch_geospatial_terrain_callback_t callback;
    void* state;

    /** Or a dataset on disk, which only an NVMe residency will read; RAM and VRAM handles refuse
     *  it with @c usearch_status_unsupported_k. */
    usearch_bytes_view_t path;
} usearch_geospatial_terrain_t;

/** The arguments of @c usearch_geospatial_init_ram. */
typedef struct usearch_geospatial_init_ram_t {

    /** Output for the new index. */
    usearch_geospatial_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The @b optional terrain lines of sight are tested over; a zeroed one means none. */
    usearch_geospatial_terrain_t terrain;

    /** The @b optional connections-per-node bound; zero derives to twelve - eight
     *  tangent-plane cones and four long-range links. */
    size_t connectivity;

    /** Upper bound on observations sharing one key; zero derives to one - unique keys. */
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

    /** The @b optional terrain lines of sight are tested over; a zeroed one means none. */
    usearch_geospatial_terrain_t terrain;

    /** The @b optional connections-per-node bound; zero derives to twelve. */
    size_t connectivity;

    /** Upper bound on observations sharing one key; zero derives to one - unique keys. */
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

    /** Observations one key may hold at most, which no checkpoint records; zero derives to one. */
    size_t values_per_key;

    /** The @b optional terrain lines of sight are tested over, which no checkpoint carries. */
    usearch_geospatial_terrain_t terrain;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
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

    /** Observations one key may hold at most, which no checkpoint records; zero derives to one. */
    size_t values_per_key;

    /** The @b optional terrain lines of sight are tested over, which no checkpoint carries. */
    usearch_geospatial_terrain_t terrain;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
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

    /** The @b optional terrain lines of sight are tested over, which no checkpoint carries. */
    usearch_geospatial_terrain_t terrain;
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
 *  @brief What "close" means numerically for one search: the radius around each query
 *      position, and the slack its time interval allows.
 *
 *  Zero @c metres means no radius: the search ranks by distance alone. Position uncertainty
 *  widens the radius by three standard deviations of the query's and the entry's accuracy
 *  combined, so a pair whose true positions lie within it matches at 98.9 % confidence. An entry
 *  matches in time when its interval and the query's, widened by @c seconds on each side,
 *  overlap; a query without a time matches every entry.
 */
typedef struct usearch_geospatial_tolerance_t {

    /** Radius in metres; zero means none. */
    double metres;

    /** Slack added to both ends of the query's time interval, in seconds. */
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

    /** Observations to admit - the batch axis of @c keys and @c observations. */
    size_t count;

    /** The identities to admit the observations under, one per entry and not necessarily unique. */
    usearch_keys_view_t keys;

    /** The observations, @c count of them. */
    usearch_geospatial_observation_t const* observations;

    /** The @b optional candidate-pool breadth the build walk explores - the @c ef of
     *  the literature; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries admitted and the batch length. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_geospatial_traversal_stats_t* stats;
} usearch_geospatial_insert_t;

/**
 *  @brief Adds observations under their keys - the native batch operation; a single observation
 *      is a batch of one.
 *  @retval usearch_status_read_only_k The handle is a view.
 *  @retval usearch_status_overflow_k A key already holds @c values_per_key entries.
 *  @retval usearch_status_invalid_argument_k An observation lacks a latitude or a longitude, or
 *      ends before it begins.
 *
 *  An observation without an altitude is placed on the handle's terrain when it has one, and on
 *  the ellipsoid otherwise.
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

    /** The @b optional handler, called with entries swept. */
    usearch_progress_t progress;

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

    /** Queries to resolve; the outputs have @c count rows of @c wanted entries. */
    size_t count;

    /** The @c count query observations. */
    usearch_geospatial_observation_t const* queries;

    /** Which entries count as matches. */
    usearch_geospatial_relation_t relation;

    /** The radius and time slack; a zero radius ranks by distance alone. */
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

    /** The @b optional admission filter, asked about a batch of candidates at a time. */
    usearch_predicate_t predicate;

    /** The @b optional handler, called with queries resolved and their count. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_geospatial_traversal_stats_t* stats;
} usearch_geospatial_search_t;

/**
 *  @brief Finds the nearest entries of each query observation that the relation admits - the
 *      native batch operation, parallelized internally; one query is a batch of one.
 *  @retval usearch_status_unsupported_k A @c visible relation on a handle without terrain.
 *
 *  Scores are chord metres between the two positions, altitude included. The relation, the time
 *  window and the predicate narrow the walk's pool, so a selective one wants a wider
 *  @c expansion_factor, and a row may hold fewer than @c wanted matches.
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

    /** Queries to resolve, one walk apiece. */
    size_t count;

    /** The @c count query observations. */
    usearch_geospatial_observation_t const* queries;

    /** Which entries count as matches. */
    usearch_geospatial_relation_t relation;

    /** The radius and time slack; @c metres must be positive. */
    usearch_geospatial_tolerance_t tolerance;

    /** Output for the report inside @c arena, query @c i spanning entries @c matches->offsets[i] up
     *  to @c matches->offsets[i+1]. */
    usearch_matches_tape_view_t* matches;

    /** The @b optional output for the entries written across every query. */
    size_t* written;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /** The @b optional admission filter, asked about a batch of candidates at a time. */
    usearch_predicate_t predicate;

    /** The @b optional handler, called with queries resolved and their count. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_geospatial_traversal_stats_t* stats;
} usearch_geospatial_within_t;

/**
 *  @brief Reports every entry within the radius of each query observation that the relation
 *      admits - the containment counterpart of the ranked search, unbounded in result count.
 *  @retval usearch_status_invalid_argument_k A tolerance of zero metres, since containment needs a
 *      radius.
 *  @retval usearch_status_unsupported_k A @c visible relation on a handle without terrain.
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

/** Counts the entries stored under one key. */
USEARCH_EXPORT void usearch_geospatial_count(usearch_geospatial_count_t* arguments);

/** The arguments of @c usearch_geospatial_get. */
typedef struct usearch_geospatial_get_t {

    /** The handle to be queried. */
    usearch_geospatial_index_t index;

    /** The executor whose stream or thread copies the observations; its kind must match the
     *  residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are exported. */
    usearch_key_t key;

    /** Output for the observations inside @c arena, @c written of them. */
    usearch_geospatial_observation_t const** observations;

    /** The @b optional output for the entries written; zero when the key is absent. */
    size_t* written;

    /** Memory for @c observations, rewound by the call. */
    usearch_arena_t arena;
} usearch_geospatial_get_t;

/**
 *  @brief Retrieves every observation stored under one key, in insertion order, within the
 *      rounding the index stores them at.
 *
 *  An absent key leaves @c observations @c NULL.
 */
USEARCH_EXPORT void usearch_geospatial_get(usearch_geospatial_get_t* arguments);

/** The arguments of @c usearch_geospatial_absorb. */
typedef struct usearch_geospatial_absorb_t {

    /** The handle absorbing the donors. */
    usearch_geospatial_index_t index;

    /** The executor the absorb fans out over, one donor per thread. */
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
    usearch_progress_t progress;
} usearch_geospatial_absorb_t;

/**
 *  @brief Absorbs every handle in @c donors into @c index - cheaper than re-inserting their
 *      observations one by one, wiring cross-links by walks across from landmarks. Donor handles are
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
USEARCH_EXPORT void usearch_geospatial_absorb(usearch_geospatial_absorb_t* arguments);

/** The arguments of @c usearch_geospatial_cursor_init. */
typedef struct usearch_geospatial_cursor_init_t {

    /** Output for the new cursor; @c NULL on failure. */
    usearch_geospatial_cursor_t* cursor;

    /** The handle to be scanned. */
    usearch_geospatial_index_t index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The @b optional key to step from: the cursor then yields the entries that the key's entries
     *  link to in the graph, once each and in link order; @c NULL scans every entry as stored. */
    usearch_key_t const* start;
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

    /** The @b optional output for the observations inside @c arena, one per key; @c NULL skips
     *  them. */
    usearch_geospatial_observation_t const** observations;

    /** Output for the entries written; zero when the snapshot is exhausted. */
    size_t* written;

    /** Memory for @c keys and @c observations, rewound by the call. */
    usearch_arena_t arena;

    /** The upper bound on entries this pull writes. */
    size_t wanted;
} usearch_geospatial_cursor_next_t;

/**
 *  @brief Copies the next batch of entries out of the cursor into @c arena.
 *  @retval usearch_status_invalid_argument_k @c index is not the handle the cursor was opened on.
 *
 *  The batch stays valid until the next call given the same arena, however the cursor moves
 *  meanwhile. Observations come back as @c usearch_geospatial_get returns them.
 */
USEARCH_EXPORT void usearch_geospatial_cursor_next(usearch_geospatial_cursor_next_t* arguments);

/** Releases the cursor; @c NULL is safe.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t cursor);

/** The arguments of @c usearch_geospatial_sample. */
typedef struct usearch_geospatial_sample_t {

    /** The handle to be sampled. */
    usearch_geospatial_index_t index;

    /** The executor the draws run on; its kind must match the handle's residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The @b optional count of @c anchors; zero draws from the whole index. */
    size_t anchors_count;

    /** The observations that bound the region around them, one anchor apiece. */
    usearch_geospatial_observation_t const* anchors;

    /** Output for the identities drawn, inside @c arena. */
    usearch_keys_view_t* keys;

    /** Output for each draw's nearest earlier draws - their keys and distances, row @c i spanning
     *  @c offsets[i] up to @c offsets[i+1] - inside @c arena. */
    usearch_matches_tape_view_t* links;

    /** Output for the draws written; fewer than @c wanted only once the region is exhausted. */
    size_t* written;

    /** Memory for @c keys and @c links, rewound by the call. */
    usearch_arena_t arena;

    /** The upper bound on draws this call writes. */
    size_t wanted;

    /** Draws already taken in this sequence, skipped so successive calls extend one sample. */
    size_t offset;

    /** Up to how many earlier draws each draw links to; one already keeps the sample connected. */
    size_t links_per_draw;

    /** The sequence: equal seeds, anchors and radius over an unchanged index draw equally. */
    uint64_t seed;

    /** The @b optional radius around the anchors in chord metres; zero bounds nothing. */
    usearch_distance_t radius;

    /** The @b optional candidate-pool breadth of the walks behind the draws; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with draws written and @c wanted. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_geospatial_traversal_stats_t* stats;
} usearch_geospatial_sample_t;

/**
 *  @brief Draws entries uniformly, across the index or around anchors, each linked to its nearest
 *      earlier draws - so the sample is one connected component, ready to be laid out.
 *
 *  Links name earlier draws of the same sequence, a later call's reaching an earlier call's. They
 *  are exact in a scan and approximate as any walk otherwise.
 */
USEARCH_EXPORT void usearch_geospatial_sample(usearch_geospatial_sample_t* arguments);

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
    usearch_progress_t progress;
} usearch_geospatial_save_t;

/**
 *  @brief Saves a mutable index under @c path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  A cancelled save leaves the parts it already wrote and no manifest referencing them, so the
 *  checkpoint reads as absent rather than half-valid. Those leftovers are the caller's to remove.
 */
USEARCH_EXPORT void usearch_geospatial_save(usearch_geospatial_save_t* arguments);

/** The geometry a checkpoint froze - everything a reader learns without opening it. */
typedef struct usearch_geospatial_metadata_t {

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

/** Reads a checkpoint's geometry without loading it. */
USEARCH_EXPORT void usearch_geospatial_metadata_read(usearch_geospatial_metadata_read_t* arguments);

/** A live index's state. Each counter is read once, so under concurrent insertions they may come
 *  from slightly different moments. */
typedef struct usearch_geospatial_index_description_t {

    /** The frozen connectivity, as a checkpoint reports it. */
    usearch_geospatial_metadata_t metadata;

    /** Stored entries. */
    size_t size;

    /** Entries hosted without growing again. */
    size_t capacity;

    /** Bytes the handle holds, observations and the terrain pack included; traversal scratch is the
     *  executor's. */
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
