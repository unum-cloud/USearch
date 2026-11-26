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
 *  @c usearch_geospatial_view_ram or @c usearch_geospatial_view_nvme, which are read-only and
 *  refuse every mutating verb with @c usearch_status_read_only_k. An NVMe handle answers every
 *  ranked search and containment walk as a RAM handle does, judging each relation on the
 *  observations its walks read, while saving answers @c usearch_status_unsupported_k.
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
 *  @c usearch_geospatial_terrain_t: a north-up grid of heights, read in place or filled on
 *  demand by the caller. A checkpoint never holds it, so every load and view names it again.
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

/** A consistent snapshot cursor over the graph. Captures the publication watermark at creation,
 *  so concurrent inserts never appear mid-iteration, and yields each entry's key with the keys it
 *  links to - every entry as stored, or breadth-first from start keys - which export, migration,
 *  and visualization feed off. Each pull names the index again; an overlapping pull reports
 *  @c usearch_status_contended_k. */
typedef struct usearch_geospatial_cursor_handle_t* usearch_geospatial_cursor_t;

/**
 *  @brief One refine in flight: the round in hand, and the root every round's health checks share.
 *
 *  A @c usearch_geospatial_refine holding it continues the round a cancel stopped, or starts the
 *  next round from the same root once the last one ended; @c usearch_geospatial_refine_plan_finish
 *  closes the round in hand and frees it. The index admits no insertion and no save while the plan
 *  lives, and each call names the index and executor the plan was opened with; both must outlive
 *  the call and the plan's finish.
 */
typedef struct usearch_geospatial_refine_plan_handle_t* usearch_geospatial_refine_plan_t;

/**
 *  @brief One reshard in flight: its own lists of sources and targets, the entries still to move,
 *      then the rounds of every target that took entries.
 *
 *  A @c usearch_geospatial_reshard holding it moves what is still pending and continues the rounds
 *  where a cancel stopped them; @c usearch_geospatial_reshard_plan_finish closes them and frees it.
 *  Every handle it names admits no insertion and no save while the plan lives, and each call names
 *  the executor the plan was opened with. Every handle a call or the plan names, and that executor,
 *  must outlive the call and the plan's finish.
 */
typedef struct usearch_geospatial_reshard_plan_handle_t* usearch_geospatial_reshard_plan_t;

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

    /** Degrees north on WGS-84, south negative. EXIF @c GPSLatitude with its @c Ref sign;
     *  QuickTime @c ISO6709; GoPro GPMF @c GPS5; MISB ST 0601 tag 13; ROS @c NavSatFix. */
    double latitude_degrees;

    /** Degrees east on WGS-84, west negative. EXIF @c GPSLongitude with its @c Ref sign;
     *  QuickTime @c ISO6709; GoPro GPMF @c GPS5; MISB ST 0601 tag 14; ROS @c NavSatFix. */
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

    /** The first instant the observation covers, in Unix seconds, UTC. EXIF @c DateTimeOriginal
     *  with @c OffsetTimeOriginal, or @c GPSDateStamp with @c GPSTimeStamp; a clip's creation date;
     *  MISB tag 2; a ROS @c header.stamp. NaN makes it an instant at @c end_seconds. */
    double begin_seconds;

    /** The last instant the observation covers, in Unix seconds, UTC: a clip's creation date plus
     *  its duration, or the sources of @c begin_seconds. NaN makes it an instant at the start. */
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

/** The canonical spelling of a relation, as @c usearch_geospatial_relation_sees_k spells "sees". */
USEARCH_EXPORT char const* usearch_geospatial_relation_name(usearch_geospatial_relation_t relation);

/** Fills @p rows × @p columns heights of a terrain, in metres above the WGS-84 ellipsoid, starting
 *  at row @p first_row and column @p first_column, with @p heights_stride bytes between rows; a
 *  cell the source does not have stays NaN, which never blocks a sight line. Called concurrently
 *  from the executor's threads. */
typedef void (*usearch_geospatial_terrain_callback_t)(size_t first_row, size_t first_column, size_t rows,
                                                      size_t columns, float* heights, size_t heights_stride,
                                                      void* state);

/**
 *  @brief The elevation every line of sight is tested over: a north-up geographic grid of
 *      heights, given by exactly one of a caller's buffer, a fill callback or a dataset path.
 *
 *  A zeroed terrain means none. Heights are metres above the WGS-84 ellipsoid, like observation
 *  altitudes, so orthometric grids such as the Copernicus DEM's need the geoid's height added
 *  first. The grid is treated as locally flat at the latitude of its middle row, which keeps
 *  distances within 1 % across a one-degree tile.
 */
typedef struct usearch_geospatial_terrain_t {

    /** The grid's row count; rows run south from the north edge. */
    size_t rows;

    /** The grid's column count; columns run east from the west edge. */
    size_t columns;

    /** The latitude of the centre of cell [0,0], in degrees. */
    double north_latitude_degrees;

    /** The longitude of the centre of cell [0,0], in degrees. */
    double west_longitude_degrees;

    /** The step between rows, in degrees, positive. */
    double row_step_degrees;

    /** The step between columns, in degrees, positive. */
    double column_step_degrees;

    /** The whole grid, row-major, read in place for the handle's lifetime. */
    float const* heights;

    /** The bytes between consecutive rows of @c heights; zero means packed. */
    size_t heights_stride;

    /** Or a handler filling windows of the grid on demand, called with @c state; VRAM handles
     *  refuse it with @c usearch_status_unsupported_k, as no device calls the host. */
    usearch_geospatial_terrain_callback_t callback;

    /** The opaque pointer passed back to every @c callback call. */
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

    /** Upper bound on observations sharing one key; zero is unbounded. */
    size_t values_per_key;

    /** The memory domains to place one shard on apiece. */
    usearch_ram_options_t ram;
} usearch_geospatial_init_ram_t;

/**
 *  @brief Creates an empty, mutable geospatial index in host memory.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A terrain both pointing at @c heights and naming a
 *    @c callback, or a @c cpu_class_ceiling no class takes.
 *  - @c usearch_status_unsupported_k A terrain @c path, which no residency reads yet, or a
 *    @c cpu_class_ceiling below every compiled class this CPU runs.
 *  - @c usearch_status_config_mismatch_k A terrain whose row or column step is not positive.
 */
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

    /** Upper bound on observations sharing one key; zero is unbounded. */
    size_t values_per_key;

    /** The devices to place one shard on apiece. */
    usearch_vram_options_t vram;
} usearch_geospatial_init_vram_t;

/**
 *  @brief Creates an empty, mutable geospatial index in device memory.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A terrain both pointing at @c heights and naming a
 *    @c callback, or a @c gpu_class_ceiling no class takes.
 *  - @c usearch_status_unsupported_k A terrain @c path or @c callback, a @c values_per_key bound,
 *    devices that load no class under @c gpu_class_ceiling, or an Apple GPU, which no geospatial
 *    kernel serves to the f64 contract.
 *  - @c usearch_status_config_mismatch_k Devices of two classes, or a terrain whose row or column
 *    step is not positive.
 *  - @c usearch_status_device_k A device past the listing, one this build holds no kernels for,
 *    or a build without GPU kernels.
 */
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

    /** Observations one key may hold at most, which no checkpoint records; zero is unbounded. */
    size_t values_per_key;

    /** The @b optional terrain lines of sight are tested over, which no checkpoint holds. */
    usearch_geospatial_terrain_t terrain;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
} usearch_geospatial_load_ram_t;

/**
 *  @brief Copies a checkpoint into a new, mutable index in host memory, its geometry read from
 *      the checkpoint's own headers.
 *
 *  Where the parts end up is the call's own memory domains, never the manifest's: part @c p is
 *  written whole on shard @c p, beside any placed shards the parts leave empty, and the shards past
 *  the placed ones open over those domains in turn.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c path, a checkpoint or part path that does not
 *    parse, a terrain both pointing at @c heights and naming a @c callback, or a
 *    @c cpu_class_ceiling no class takes.
 *  - @c usearch_status_malformed_k The checkpoint lists no part, or a part, its header, columns or
 *    sums do not parse.
 *  - @c usearch_status_unsupported_k A terrain @c path, or a @c cpu_class_ceiling below every
 *    compiled class this CPU runs.
 *  - @c usearch_status_uninitialized_k The executor holds no pool workers.
 *  - @c usearch_status_config_mismatch_k A terrain whose row or column step is not positive, or
 *    the checkpoint's columns differ from the shards' in dtype, shape or key width.
 *  - @c usearch_status_device_k The executor is not a RAM executor.
 *  - @c usearch_status_contended_k Another call holds the executor, or its pool runs another
 *    dispatch.
 *  - @c usearch_status_overflow_k A part passes a shard's bound or its key multimap.
 */
USEARCH_EXPORT void usearch_geospatial_load_ram(usearch_geospatial_load_ram_t* arguments);

/** The arguments of @c usearch_geospatial_load_ram_buffer. */
typedef struct usearch_geospatial_load_ram_buffer_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_geospatial_index_t* index;

    /** The executor the read fans out over, one shard per thread. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Source holding one serialized monolith, or the manifest of several parts leading them. */
    usearch_bytes_view_t buffer;

    /** The memory domains to place the parts on. */
    usearch_ram_options_t ram;

    /** Observations one key may hold at most, which no checkpoint records; zero is unbounded. */
    size_t values_per_key;

    /** The @b optional terrain lines of sight are tested over, which no checkpoint holds. */
    usearch_geospatial_terrain_t terrain;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
} usearch_geospatial_load_ram_buffer_t;

/**
 *  @brief Copies a checkpoint from a memory buffer into a new, mutable index in host memory.
 *
 *  This places, reports and cancels like @c usearch_geospatial_load_ram, which reaches the same
 *  bytes through a mapping, so @c buffer needs no file behind it, and a mapped file no copy.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c buffer, a terrain both pointing at @c heights
 *    and naming a @c callback, or a @c cpu_class_ceiling no class takes.
 *  - @c usearch_status_malformed_k The manifest lists no part, or a part, its header, columns or
 *    sums do not parse.
 *  - @c usearch_status_unsupported_k A terrain @c path, or a @c cpu_class_ceiling below every
 *    compiled class this CPU runs.
 *  - @c usearch_status_uninitialized_k The executor holds no pool workers.
 *  - @c usearch_status_config_mismatch_k A terrain whose row or column step is not positive, or
 *    the checkpoint's columns differ from the shards' in dtype, shape or key width.
 *  - @c usearch_status_device_k The executor is not a RAM executor.
 *  - @c usearch_status_contended_k Another call holds the executor, or its pool runs another
 *    dispatch.
 *  - @c usearch_status_overflow_k A part passes a shard's bound or its key multimap.
 */
USEARCH_EXPORT void usearch_geospatial_load_ram_buffer(usearch_geospatial_load_ram_buffer_t* arguments);

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

    /** Observations one key may hold at most, which no checkpoint records; zero is unbounded. */
    size_t values_per_key;

    /** The @b optional terrain lines of sight are tested over, which no checkpoint holds. */
    usearch_geospatial_terrain_t terrain;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
} usearch_geospatial_load_vram_t;

/**
 *  @brief Copies a checkpoint into a new, mutable index in device memory, with the placement rules
 *      of @c usearch_geospatial_load_ram.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c path, a checkpoint or part path that does not
 *    parse, a terrain both pointing at @c heights and naming a @c callback, or a
 *    @c gpu_class_ceiling no class takes.
 *  - @c usearch_status_malformed_k The checkpoint lists no part, or a part, its header, columns or
 *    sums do not parse.
 *  - @c usearch_status_unsupported_k A terrain @c path or @c callback, a @c values_per_key bound,
 *    devices that load no class under @c gpu_class_ceiling, or an Apple GPU, which no geospatial
 *    kernel serves to the f64 contract.
 *  - @c usearch_status_config_mismatch_k Devices of two classes, a terrain whose row or column
 *    step is not positive, the executor driving no stream on some shard's device, or the
 *    checkpoint's columns differing from the shards' in dtype, shape or key width.
 *  - @c usearch_status_device_k An ordinal past the listing, devices this build holds no kernels
 *    for, a build without GPU kernels, or an executor that is not a VRAM executor.
 *  - @c usearch_status_contended_k Another call holds the executor.
 *  - @c usearch_status_overflow_k A part passes a shard's bound or its key multimap.
 *  - @c usearch_status_io_k A read of the checkpoint failed mid-load.
 */
USEARCH_EXPORT void usearch_geospatial_load_vram(usearch_geospatial_load_vram_t* arguments);

/** The arguments of @c usearch_geospatial_load_vram_buffer. */
typedef struct usearch_geospatial_load_vram_buffer_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_geospatial_index_t* index;

    /** The executor the upload runs on, one stream per device. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Source holding one serialized monolith, or the manifest of several parts leading them, in
     *  host memory. */
    usearch_bytes_view_t buffer;

    /** The devices to place the parts on. */
    usearch_vram_options_t vram;

    /** Observations one key may hold at most, which no checkpoint records; zero is unbounded. */
    size_t values_per_key;

    /** The @b optional terrain lines of sight are tested over, which no checkpoint holds. */
    usearch_geospatial_terrain_t terrain;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
} usearch_geospatial_load_vram_buffer_t;

/**
 *  @brief Copies a checkpoint from a memory buffer into a new, mutable index in device memory,
 *      reporting and cancelling like @c usearch_geospatial_load_vram.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c buffer, a terrain both pointing at @c heights
 *    and naming a @c callback, or a @c gpu_class_ceiling no class takes.
 *  - @c usearch_status_malformed_k The manifest lists no part, or a part, its header, columns or
 *    sums do not parse.
 *  - @c usearch_status_unsupported_k A @c buffer in device memory, a terrain @c path or
 *    @c callback, a @c values_per_key bound, devices that load no class under
 *    @c gpu_class_ceiling, or an Apple GPU, which no geospatial kernel serves to the f64 contract.
 *  - @c usearch_status_config_mismatch_k Devices of two classes, a terrain whose row or column
 *    step is not positive, the executor driving no stream on some shard's device, or the
 *    checkpoint's columns differing from the shards' in dtype, shape or key width.
 *  - @c usearch_status_device_k An ordinal past the listing, devices this build holds no kernels
 *    for, a build without GPU kernels, or an executor that is not a VRAM executor.
 *  - @c usearch_status_contended_k Another call holds the executor.
 *  - @c usearch_status_overflow_k A part passes a shard's bound or its key multimap.
 */
USEARCH_EXPORT void usearch_geospatial_load_vram_buffer(usearch_geospatial_load_vram_buffer_t* arguments);

/** The arguments of @c usearch_geospatial_view_ram. */
typedef struct usearch_geospatial_view_ram_t {

    /** Output for the new, read-only index. */
    usearch_geospatial_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** The @b optional terrain lines of sight are tested over, which no checkpoint holds. */
    usearch_geospatial_terrain_t terrain;

    /** The class to dispatch as; a view reads its shards from the checkpoint, so only
     *  @c usearch_ram_options_t::cpu_class_ceiling is read. */
    usearch_ram_options_t ram;
} usearch_geospatial_view_ram_t;

/**
 *  @brief Serves a checkpoint in place through a memory mapping, as a read-only index whose
 *      mutating verbs report @c usearch_status_read_only_k.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c path, a checkpoint or part path that does not
 *    parse, a terrain both pointing at @c heights and naming a @c callback, or a
 *    @c cpu_class_ceiling no class takes.
 *  - @c usearch_status_malformed_k The checkpoint lists no part, or a part or its header does not
 *    parse.
 *  - @c usearch_status_config_mismatch_k A terrain whose row or column step is not positive.
 *  - @c usearch_status_unsupported_k Every other checkpoint, as RAM shards cannot serve a mapping
 *    in place yet.
 */
USEARCH_EXPORT void usearch_geospatial_view_ram(usearch_geospatial_view_ram_t* arguments);

/** The arguments of @c usearch_geospatial_view_nvme. */
typedef struct usearch_geospatial_view_nvme_t {

    /** Output for the new, read-only index, written by the time the call is final. */
    usearch_geospatial_index_t* index;

    /** The NVMe executor the call is ordered on; the headers come through blocking reads. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** The devices the parts are served from; naming none serves each where its manifest points. */
    usearch_nvme_options_t nvme;

    /** The @b optional terrain lines of sight are tested over, which no checkpoint holds. */
    usearch_geospatial_terrain_t terrain;
} usearch_geospatial_view_nvme_t;

/**
 *  @brief Serves a checkpoint in place from NVMe drives as a read-only index, every later call
 *      reading it through its executor's engines.
 *
 *  Opening reads the headers alone. Every mutating verb on the handle reports
 *  @c usearch_status_read_only_k, and the parts must stay unmodified until the handle is freed.
 *
 *  @par Statuses:
 *  - @c usearch_status_device_k @c executor is not an NVMe executor.
 *  - @c usearch_status_invalid_argument_k An empty @c path, @c NULL paths for a nonzero count, a
 *    path that does not parse or names a base off its medium's block, a terrain both pointing at
 *    @c heights and naming a @c callback, or a @c cpu_class_ceiling no class takes.
 *  - @c usearch_status_malformed_k The manifest names no part, or a manifest, header or row
 *    directory does not parse.
 *  - @c usearch_status_config_mismatch_k Parts disagreeing with this index's values, keys, node
 *    IDs or metric, or with @c nvme, or a terrain whose row or column step is not positive.
 *  - @c usearch_status_unsupported_k A terrain @c path, or a @c cpu_class_ceiling below every
 *    compiled class this CPU runs.
 *  - @c usearch_status_uninitialized_k The executor holds no threads or pool workers.
 *  - @c usearch_status_contended_k Another call holds the executor, or its pool runs another
 *    dispatch.
 */
USEARCH_EXPORT void usearch_geospatial_view_nvme(usearch_geospatial_view_nvme_t* arguments);

/** Frees the resources associated with the index; @c NULL is safe, and it cannot fail, so takes
 *  no outcome.
 *  Waits out every executor call naming the handle; never call it from their callbacks. */
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

/**
 *  Grows the shards ahead of incoming entries - a capacity hint, since insertions grow them on
 *  demand.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k The handle is an NVMe view.
 *  - @c usearch_status_overflow_k @c entries passes a shard's ceiling, its ID space or its key
 *    column.
 */
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

/**
 *  Erases all entries, keeping the reserved capacity.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k The handle is an NVMe view.
 *  - @c usearch_status_contended_k A call in flight, a batch or load holding claims, or a plan
 *    holding the handle closed.
 */
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

    /** The @b optional breadth of the candidates the build walk explores - the @c ef of
     *  the literature; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries admitted and the batch length. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_geospatial_insert_t;

/**
 *  @brief Adds observations under their keys - the native batch operation; a single observation
 *      is a batch of one.
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
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k The handle is a view.
 *  - @c usearch_status_invalid_argument_k An observation lacks a latitude or a longitude, or
 *    ends before it begins.
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

    /** The @b optional output for what the call did to the graph, as @c usearch_refine_report_t
     *  counts it; @c NULL skips reporting. */
    usearch_refine_report_t* report;

    /** The @b optional breadth and bytes of the re-selection pass. */
    usearch_refine_options_t refine;

    /** The @b optional handler, called with entries re-selected; a cancel stops the re-selection,
     *  still heals, and answers @c usearch_status_cancelled_k. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;

    /** The @b optional plan: @c NULL runs one round to its end, and a cancel closes it as
     *  @c usearch_geospatial_refine_plan_finish does; a @c NULL pointee opens one that a cancel
     *  leaves live; a live one continues its round where it stopped, or starts the next. */
    usearch_geospatial_refine_plan_t* plan;

    /** The @b optional output for where this call stopped, @c usearch_phase_done_k once its round
     *  ended. */
    usearch_phase_progress_t* phase;
} usearch_geospatial_refine_t;

/**
 *  @brief Re-selects the neighbors of every stored entry against the finished graph, improving
 *      neighbors chosen when the graph around them was still incomplete.
 *
 *  This is a build step rather than a repair. A cheap insertion pass followed by one refinement
 *  reaches a better graph, in less total time, than a single expensive insertion pass - so a wide
 *  @c refine.expansion_factor belongs here rather than in @c usearch_geospatial_insert. A refined
 *  entry keeps its cone winners and the longest of its other candidates, then has its long links
 *  drawn again by the insertion's rule, so a refine never leaves an entry without them.
 *
 *  Cancelling is safe, which is what makes a whole-graph sweep affordable. Re-selecting an entry's
 *  neighbors is idempotent and claims no identity, so a stopped pass leaves the graph correct and
 *  merely less improved: nothing unwinds, and the call answers @c usearch_status_cancelled_k. The
 *  index keeps nothing of a stopped pass without a @c plan, so the next call starts again; with
 *  one, a cancel leaves the plan live, the next call naming it continues where the cancel came, and
 *  @c usearch_geospatial_refine_plan_finish closes it.
 *
 *  Re-selection alone can leave an island, an entry no search reaches, so every call closes by
 *  re-linking them: unreachable entries re-select from scratch, and whatever the selection still
 *  drops is reconnected to a reachable one. A cancelled call skips the rest of the re-selection but
 *  still re-links, so it never leaves entries unreachable.
 *
 *  Searches may run throughout: re-selection rewrites one entry at a time and never removes one,
 *  so a concurrent walk sees either the old neighbors or the new ones. Insertions never overlap:
 *  one in flight refuses the call with @c usearch_status_contended_k, and one arriving meanwhile
 *  is refused the same way, as is a save.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A @c plan without an executor, or a live @c plan opened
 *    on another index or executor.
 *  - @c usearch_status_read_only_k The handle is an NVMe view.
 *  - @c usearch_status_unsupported_k An Apple GPU handle, which no basic tower serves.
 *  - @c usearch_status_contended_k An insertion or another refinement holds the handle.
 */
USEARCH_EXPORT void usearch_geospatial_refine(usearch_geospatial_refine_t* arguments);

/** The arguments of @c usearch_geospatial_refine_plan_finish. */
typedef struct usearch_geospatial_refine_plan_finish_t {

    /** The handle the plan was opened on. */
    usearch_geospatial_index_t index;

    /** The executor the plan was opened with. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The plan to close; freed by the call, whatever it answers. */
    usearch_geospatial_refine_plan_t plan;

    /** The @b optional output for what the whole plan did to the graph. */
    usearch_refine_report_t* report;
} usearch_geospatial_refine_plan_finish_t;

/**
 *  @brief Closes and frees @c plan whatever the call answers, reopening the index.
 *
 *  A round a cancel stopped closes as a stop at that barrier would, as @c usearch_geospatial_refine
 *  describes, its verdict reading stopped; a round that ended reports again.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A @c NULL @c plan, or one opened on another index or
 *    executor; the plan is still freed.
 */
USEARCH_EXPORT void usearch_geospatial_refine_plan_finish(usearch_geospatial_refine_plan_finish_t* arguments);

/** The arguments of @c usearch_geospatial_health. */
typedef struct usearch_geospatial_health_t {

    /** The handle to be audited. */
    usearch_geospatial_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output for the audit; zeroed on failure. */
    usearch_reachability_t* reachability;
} usearch_geospatial_health_t;

/**
 *  @brief Audits how much of the graph a search can reach: every shard swept level by level from
 *      its first entry, forward and back. Read-only, so concurrent insertions merely leave fresh
 *      entries out of the audit.
 *
 *  An NVMe view sweeps forward alone, from entries spaced through each part, so its
 *  @c depth_to_root stays zero.
 *
 *  @par Statuses:
 *  - @c usearch_status_device_k The executor's kind does not match the residency.
 *  - @c usearch_status_uninitialized_k No shard is placed, or the executor holds no pool workers
 *    or streams.
 *  - @c usearch_status_contended_k Another call holds the executor or its pool, or a destructive
 *    window overlapped the audit twice.
 *  - @c usearch_status_config_mismatch_k The executor drives no stream on some VRAM shard's
 *    device, or an NVMe medium the engines cannot read.
 *  - @c usearch_status_io_k A read from an NVMe view's drives failed.
 */
USEARCH_EXPORT void usearch_geospatial_health(usearch_geospatial_health_t* arguments);

/** The arguments of @c usearch_geospatial_search. */
typedef struct usearch_geospatial_search_t {

    /* Call */
    /** The handle to be queried. */
    usearch_geospatial_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /* Inputs */
    /** Queries to resolve; the outputs have @c count rows of @c wanted entries. */
    size_t count;

    /** The @c count query observations. */
    usearch_geospatial_observation_t const* queries;

    /** Which entries count as matches. */
    usearch_geospatial_relation_t relation;

    /** The radius and time slack; a zero radius ranks by distance alone. */
    usearch_geospatial_tolerance_t tolerance;

    /* Outputs */
    /** Output for the run, inside @c arena. */
    usearch_matches_view_t* matches;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /* Knobs */
    /** Columns per row, and the upper bound on matches - the "k" in "kANN". */
    size_t wanted;

    /** The @b optional breadth of the candidates a walk explores - the @c ef of the literature;
     *  zero derives, a scan ignores it. */
    size_t expansion_factor;

    /** How entries are reached: a walk, the default, or an exact scan of every entry, on RAM and
     *  NVMe. */
    usearch_traversal_t traversal;

    /** The @b optional predicate, asked about a batch of candidates at a time. */
    usearch_predicate_t predicate;

    /* Observers */
    /** The @b optional handler, called with queries resolved and their count. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_geospatial_search_t;

/**
 *  @brief Finds the nearest entries of each query observation that the relation admits, approximate
 *      by walking the graph or exact by scanning every entry - the native batch operation,
 *      parallelized internally; one query is a batch of one.
 *
 *  Scores are chord metres between the two positions, altitude included. The relation, the time
 *  window and the predicate narrow the walk's candidates, so a selective one wants a wider
 *  @c expansion_factor, and a query may hold fewer than @c wanted matches.
 *
 *  A @c predicate narrows the matches to the entries it admits. Non-matching entries still route
 *  the traversal, so recall does not collapse at low selectivity.
 *
 *  A cancelled search reports @c usearch_status_cancelled_k and leaves @c matches @c NULL.
 *
 *  @par Statuses:
 *  - @c usearch_status_unknown_kind_k A @c relation or @c traversal this build does not know.
 *  - @c usearch_status_unsupported_k A @c visible relation on a handle without terrain, or a
 *    scan of a VRAM handle, or a @c predicate on one.
 */
USEARCH_EXPORT void usearch_geospatial_search(usearch_geospatial_search_t* arguments);

/** The arguments of @c usearch_geospatial_within. */
typedef struct usearch_geospatial_within_t {

    /* Call */
    /** The handle to be queried. */
    usearch_geospatial_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /* Inputs */
    /** Queries to resolve, one walk apiece. */
    size_t count;

    /** The @c count query observations. */
    usearch_geospatial_observation_t const* queries;

    /** Which entries count as matches. */
    usearch_geospatial_relation_t relation;

    /** The radius and time slack; @c metres must be positive. */
    usearch_geospatial_tolerance_t tolerance;

    /* Outputs */
    /** Output for the report inside @c arena, query @c i spanning entries @c matches->offsets[i] up
     *  to @c matches->offsets[i+1]. */
    usearch_matches_tape_view_t* matches;

    /** The @b optional output for the entries written across every query. */
    size_t* written;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /* Knobs */
    /** How entries are reached: a complete walk, the default, or an exact scan of every entry, on
     *  RAM and NVMe. */
    usearch_traversal_t traversal;

    /** The @b optional predicate, asked about a batch of candidates at a time. */
    usearch_predicate_t predicate;

    /* Observers */
    /** The @b optional handler, called with queries resolved and their count. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_geospatial_within_t;

/**
 *  @brief Reports every entry within the radius of each query observation that the relation
 *      admits - the containment counterpart of the ranked search, unbounded in result count.
 *
 *  Complete on the converged graph: the walk widens its candidates by the cone graph's spanner
 *  bound before trusting it, so every candidate inside the tolerance is enumerated, at about thirty
 *  times the cost of the ball it reports. Containment sizes its own candidates from that bound, so
 *  it takes no breadth a caller could make the report incomplete with. Only the walks learn how
 *  large the report is, so the keys, scores and Apache Arrow offsets are written in @c arena as one
 *  ragged output, as @c usearch_dense_multiply writes its pairs. A @c predicate narrows the report
 *  without stopping the walk from crossing the entries it rejects.
 *
 *  A cancelled call reports @c usearch_status_cancelled_k and leaves the outputs @c NULL.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A tolerance of zero metres, since containment needs a
 *    radius.
 *  - @c usearch_status_unknown_kind_k A @c relation or @c traversal this build does not know.
 *  - @c usearch_status_unsupported_k A @c visible relation on a handle without terrain, or a
 *    scan of a VRAM handle, or a @c predicate on one.
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

/**
 *  @brief Counts the entries stored under one key, on every residency.
 *
 *  @par Statuses:
 *  - @c usearch_status_device_k An NVMe view's executor is not an NVMe executor.
 *  - @c usearch_status_uninitialized_k An NVMe view's executor holds no pool workers.
 *  - @c usearch_status_contended_k Another call holds an NVMe view's executor, or its pool runs
 *    another dispatch.
 *  - @c usearch_status_config_mismatch_k An NVMe medium the engines cannot read.
 */
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
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A @c NULL @c arena.
 *  - @c usearch_status_device_k An arena or an executor on another residency or device.
 *  - @c usearch_status_contended_k The arena serves another call, or another call holds an NVMe
 *    view's executor or its pool.
 *  - @c usearch_status_unsupported_k The handle lives on a GPU, whose index exports nothing yet.
 *  - @c usearch_status_uninitialized_k An NVMe view's executor holds no pool workers.
 *  - @c usearch_status_config_mismatch_k An NVMe medium the engines cannot read.
 */
USEARCH_EXPORT void usearch_geospatial_get(usearch_geospatial_get_t* arguments);

/** The arguments of @c usearch_geospatial_reshard. Every handle @c sources and @c targets name must
 *  outlive the call and, with a @c plan, the plan's finish. */
typedef struct usearch_geospatial_reshard_t {

    /** Source handles whose entries move, each named once. */
    size_t sources_count;

    /** Array of @c sources_count handles; a source the targets do not name ends empty and live. */
    usearch_geospatial_index_t const* sources;

    /** Target handles the entries move into, each named once. */
    size_t targets_count;

    /** Array of @c targets_count handles of the sources' residency, class, scalar, kind and
     *  geometry, each a source itself or empty. */
    usearch_geospatial_index_t const* targets;

    /** The executor every move and the closing rounds fan out over. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The @b optional output for what the call did to the graph, as @c usearch_refine_report_t
     *  counts it; @c NULL skips reporting. */
    usearch_refine_report_t* report;

    /** The @b optional spending of the closing rounds, as @c usearch_refine_options_t names it. */
    usearch_refine_options_t refine;

    /** Picks the part roots of a cut: equal seeds over an unchanged source cut it equally. */
    uint64_t seed;

    /** The @b optional handler, called at every barrier of the closing rounds' phases with that
     *  phase's units; a cancel stops them as @c usearch_geospatial_refine does, never a move. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;

    /** The @b optional plan: @c NULL moves and runs to the end, a @c NULL pointee opens one that a
     *  cancel leaves live, a live one continues where it stopped. */
    usearch_geospatial_reshard_plan_t* plan;

    /** The @b optional output for where this call stopped, @c usearch_phase_done_k once its rounds
     *  ended. */
    usearch_phase_progress_t* phase;
} usearch_geospatial_reshard_t;

/**
 *  @brief Moves the entries of @c sources into @c targets, keeping their links rather than
 *      re-inserting their observations, then runs a round on every target that took entries.
 *
 *  Gathering names several sources and one target, itself a source or empty. Each other source
 *  lands all-or-nothing, its entries appending to the target with their edges rebased, so at most
 *  one source's entries are ever held twice.
 *
 *  Cutting names one source and several targets, one graph-local part each. The part roots are
 *  drawn from @c seed without a scan, each part grows breadth-first over the links from its root,
 *  capped at an equal share of the entries, and entries no root reaches enter the smallest part.
 *  Each part enters its target with its edges into other parts dropped and the entries that lost
 *  one placed first.
 *
 *  Gathering then cutting names several sources and several targets, none of them a source: every
 *  other source gathers into the first as a gather would, with no round, and the first is then
 *  cut into the targets, every source ending empty and live.
 *
 *  No walk links a moved entry in: each target then runs a round, as @c usearch_geospatial_refine
 *  does, healing what arrived as an island and re-stitching the neighborhoods it shares with the
 *  entries already there. Every handle is checked and held before the first entry moves, so a
 *  refusal moves nothing, and every handle stays the caller's: a source the targets do not name
 *  ends empty and live, and @c usearch_geospatial_free frees it as any other. Readers of a target
 *  may keep searching throughout, since unpublished entries hide behind the watermark, and
 *  insertions into any handle named arriving meanwhile are refused with
 *  @c usearch_status_contended_k.
 *
 *  The handler reports and cancels the closing rounds alone, exactly as in
 *  @c usearch_geospatial_refine, and answers @c usearch_status_cancelled_k with every entry moved.
 *  A caller wanting to continue later passes a @c plan: a cancel then leaves it live, the next call
 *  naming it moves what is still pending and continues, and
 *  @c usearch_geospatial_reshard_plan_finish closes it. A live plan reads its own sources and
 *  targets, never the call's.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k No sources or no targets, a @c NULL list or slot, a
 *    handle named twice among the sources or among the targets, a @c plan without an executor, or a
 *    live @c plan opened with another executor.
 *  - @c usearch_status_read_only_k A handle is an NVMe view.
 *  - @c usearch_status_config_mismatch_k A handle of another residency, class, scalar, kind or
 *    geometry than the first source, a target neither a source nor empty, or a source cut into
 *    targets that include it.
 *  - @c usearch_status_unsupported_k One handle as its only source and target, since a handle of
 *    one shard has nothing to recut, or a handle on an Apple GPU, which no tower serves.
 *  - @c usearch_status_contended_k An insertion or refinement in flight on a handle named; nothing
 *    moves.
 *  - @c usearch_status_overflow_k A part passes a target shard's ceiling.
 */
USEARCH_EXPORT void usearch_geospatial_reshard(usearch_geospatial_reshard_t* arguments);

/** The arguments of @c usearch_geospatial_reshard_plan_finish. */
typedef struct usearch_geospatial_reshard_plan_finish_t {

    /** The executor the plan was opened with. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The plan to close; freed by the call, whatever it answers. */
    usearch_geospatial_reshard_plan_t plan;

    /** The @b optional output for what the whole plan did to the graph. */
    usearch_refine_report_t* report;
} usearch_geospatial_reshard_plan_finish_t;

/**
 *  @brief Closes and frees @c plan whatever the call answers, reopening every handle it names.
 *
 *  The rounds a cancel stopped close as a stop at that barrier would, their verdict reading
 *  stopped; a plan whose entries have not all moved runs no round and reports zeros, every handle
 *  holding what it held when the plan stopped.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A @c NULL @c plan, or one opened with another executor;
 *    the plan is still freed.
 */
USEARCH_EXPORT void usearch_geospatial_reshard_plan_finish(usearch_geospatial_reshard_plan_finish_t* arguments);

/** The arguments of @c usearch_geospatial_cursor_init. */
typedef struct usearch_geospatial_cursor_init_t {

    /** The handle to be scanned. */
    usearch_geospatial_index_t index;

    /** The @b optional executor whose first thread's memory domain holds the cursor's state;
     *  @c NULL keeps it on no domain. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The @b optional count of @c starts; zero walks every entry as stored. */
    size_t starts_count;

    /** The keys to walk from, in the order the walk begins with them. */
    usearch_key_t const* starts;

    /** Output for the new cursor; @c NULL on failure. */
    usearch_geospatial_cursor_t* cursor;
} usearch_geospatial_cursor_init_t;

/**
 *  @brief Opens a snapshot cursor over the current contents; see @c usearch_geospatial_cursor_t.
 *
 *  Without @c starts the cursor yields every entry as stored. With them it yields the start keys'
 *  own entries first, in the order given, then the entries those link to, breadth-first, each
 *  entry once, so a caller wanting only the starts' immediate links reads the links of the first
 *  pulled entries. A key the index lacks starts nothing, and a key several entries share starts
 *  from the first stored.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c NULL @c starts for a nonzero @c starts_count.
 *  - @c usearch_status_unsupported_k The handle lives on a GPU, which walks no links yet.
 */
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

    /** The @b optional output for each entry's links, the keys of the entries it links to in the
     *  graph, entry @c i spanning @c offsets[i] up to @c offsets[i+1], inside @c arena; @c NULL
     *  skips them. */
    usearch_keys_tape_view_t* links;

    /** Output for the entries written; zero when the snapshot is exhausted. */
    size_t* written;

    /** Memory for @c keys and @c links, rewound by the call. */
    usearch_arena_t arena;

    /** The upper bound on entries this pull writes. */
    size_t wanted;
} usearch_geospatial_cursor_next_t;

/**
 *  @brief Copies the next batch of entries out of the cursor into @c arena, each with the keys it
 *      links to, in the order @c usearch_geospatial_cursor_init describes; the observations
 *      themselves come from @c usearch_geospatial_get.
 *
 *  Links name only entries the snapshot holds, as the graph stored them when the pull ran. The keys
 *  are copied rather than read in place from the index, so the batch stays valid until the next
 *  call given the same arena, however the cursor moves meanwhile.
 *
 *  @par Statuses:
 *  - @c usearch_status_io_k A read from an NVMe view's drives failed.
 *  - @c usearch_status_invalid_argument_k A @c NULL @c cursor, or @c index not the handle the
 *    cursor was opened on.
 */
USEARCH_EXPORT void usearch_geospatial_cursor_next(usearch_geospatial_cursor_next_t* arguments);

/** Releases the cursor; @c NULL is safe.
 *  Waits out every executor call naming the handle; never call it from their callbacks. */
USEARCH_EXPORT void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t cursor);

/** The arguments of @c usearch_geospatial_sample. */
typedef struct usearch_geospatial_sample_t {

    /* Call */
    /** The handle to be sampled. */
    usearch_geospatial_index_t index;

    /** The executor the draws run on; its kind must match the handle's residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /* Inputs */

    /** The @b optional count of @c shown; zero shows nothing yet. */
    size_t shown_count;

    /** The keys already shown: never drawn again, and the paths reach toward them. */
    usearch_keys_view_t shown;

    /** Where the @c shown keys sit, @c shown_count rows of three @c usearch_scalar_f32_k values,
     *  the points later draws are placed against. */
    usearch_rows_view_t shown_coordinates;

    /** The @b optional count of @c focus; zero looks around the whole index. */
    size_t focus_count;

    /** The keys to look around, the roots the draws grow from. */
    usearch_keys_view_t focus;

    /* Outputs */

    /** Output for the keys drawn, inside @c arena. */
    usearch_keys_view_t* keys;

    /** Output for which of its key's entries each draw is, in the order @c usearch_geospatial_get
     *  writes them, inside @c arena. */
    size_t const** entry_ordinals;

    /** Output for the one shown or earlier draw each draw hangs from and its distance, one column,
     *  a draw with none counting zero, inside @c arena. */
    usearch_matches_view_t* parents;

    /** Output for where each draw is placed, @c written rows of three @c usearch_scalar_f32_k
     *  values, Earth-fixed positions in Earth radii, inside @c arena. */
    usearch_rows_view_t* coordinates;

    /** Output for the draws written; fewer than @c wanted only once the region is exhausted. */
    size_t* written;

    /** Memory for every output, rewound by the call. */
    usearch_arena_t arena;

    /* Knobs */
    /** The upper bound on draws this call writes. */
    size_t wanted;

    /** The draw: equal seeds, shown keys and focus over an unchanged index draw equally, and a
     *  larger @c wanted draws a superset. */
    uint64_t seed;

    /** The @b optional breadth of the candidates of the walks behind the draws; zero derives. */
    size_t expansion_factor;

    /* Observers */
    /** The @b optional handler, called with draws written and @c wanted. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_geospatial_sample_t;

/**
 *  @brief Draws keys around @c focus, or across the whole index, each hanging from a shown or
 *      earlier draw and placed in three dimensions beside the shown ones - so a view grows one
 *      connected sample as it zooms, never scanning the index.
 *
 *  The draws follow the graph: along paths joining the focus to each other and toward the nearest
 *  shown key, then around each focus key as far as its nearest neighbor on those paths, ranked by
 *  @c seed so a closer view draws a superset of a farther one. A key holding several entries draws
 *  each as its own, named by its entry ordinal. Each draw is placed at its Earth-fixed position, so
 *  @c shown_coordinates place nothing.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c NULL @c shown or @c focus for a nonzero count, or
 *    @c shown_coordinates not three @c usearch_scalar_f32_k columns.
 *  - @c usearch_status_unsupported_k The handle lives on a GPU, which does not sample yet.
 *  - @c usearch_status_io_k A read from an NVMe view's drives failed.
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

    /** Destination path: the monolith of a lone shard, or the manifest beside the parts of many. */
    usearch_bytes_view_t path;

    /** The @b optional handler, called with parts written and their count. */
    usearch_progress_t progress;
} usearch_geospatial_save_t;

/**
 *  @brief Saves a mutable index under @c path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  The shard count picks the layout: one shard writes one monolith at @c path, and several write
 *  the manifest at @c path beside one part per shard, each with its shard's own node IDs.
 *  A cancelled save leaves the parts it already wrote and no manifest referencing them, so the
 *  checkpoint reads as absent rather than half-valid. Those leftovers are the caller's to remove.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c path, or a part or manifest path that does
 *    not parse.
 *  - @c usearch_status_device_k The executor's kind does not match the residency.
 *  - @c usearch_status_unsupported_k The handle is an NVMe view, which already is its checkpoint.
 *  - @c usearch_status_contended_k Another call holds the executor or its pool, a dispensed batch
 *    has yet to publish, or a call, batch, load or plan holds the handle.
 *  - @c usearch_status_uninitialized_k The executor holds no pool workers.
 *  - @c usearch_status_config_mismatch_k The executor drives no stream on some VRAM shard's
 *    device.
 *  - @c usearch_status_overflow_k @c path names a device that ends before a part or the manifest.
 */
USEARCH_EXPORT void usearch_geospatial_save(usearch_geospatial_save_t* arguments);

/** The arguments of @c usearch_geospatial_save_buffer. */
typedef struct usearch_geospatial_save_buffer_t {

    /** The handle to be written. */
    usearch_geospatial_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output for the checkpoint's bytes and length, inside @c arena. */
    usearch_bytes_view_t* monolith;

    /** Memory for @c monolith, rewound by the call. */
    usearch_arena_t arena;

    /** The @b optional handler, called with parts written and their count. */
    usearch_progress_t progress;
} usearch_geospatial_save_buffer_t;

/**
 *  @brief Saves a mutable index into @c arena: one monolith for a handle of one shard, or a
 *      manifest leading one part per shard. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  A lone shard's monolith is the same bytes @c usearch_geospatial_save writes. Several shards
 *  write the manifest first, listing each part by its offset inside the buffer, then every part on
 *  the header's alignment, each with its shard's own node IDs, so a load opens one shard per part
 *  again. A cancelled call leaves @c monolith->data @c NULL.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A @c NULL @c arena.
 *  - @c usearch_status_device_k An arena or an executor on another residency or device.
 *  - @c usearch_status_contended_k The arena serves another call, another call holds the
 *    executor or its pool, a dispensed batch has yet to publish, or a call, batch, load or plan
 *    holds the handle.
 *  - @c usearch_status_unsupported_k The handle is an NVMe view, which already is its checkpoint,
 *    or lives on a GPU, which writes no checkpoint into memory yet.
 *  - @c usearch_status_uninitialized_k No shard is live, or the executor holds no pool workers.
 *  - @c usearch_status_config_mismatch_k A stored value disagrees with the shape its column
 *    records.
 *  - @c usearch_status_out_of_memory_k The arena cannot grow to the checkpoint's length.
 */
USEARCH_EXPORT void usearch_geospatial_save_buffer(usearch_geospatial_save_buffer_t* arguments);

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

/**
 *  Reads a checkpoint's geometry without loading it.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k The checkpoint or a part path does not parse.
 *  - @c usearch_status_malformed_k The checkpoint or a part is empty, its manifest lists no part,
 *    or a header does not parse or spans more bytes than its part.
 */
USEARCH_EXPORT void usearch_geospatial_metadata_read(usearch_geospatial_metadata_read_t* arguments);

/** The arguments of @c usearch_geospatial_metadata_read_buffer. */
typedef struct usearch_geospatial_metadata_read_buffer_t {

    /** The executor the read runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Source holding one serialized monolith, or the manifest of several parts leading them. */
    usearch_bytes_view_t buffer;

    /** Output for every field the checkpoint froze. */
    usearch_geospatial_metadata_t* metadata;
} usearch_geospatial_metadata_read_buffer_t;

/**
 *  @brief Reads the fields a checkpoint in a memory buffer froze without loading it, as
 *      @c usearch_geospatial_metadata_read reads them from a path.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c buffer.
 *  - @c usearch_status_malformed_k The manifest lists no part, or a header does not parse or
 *    spans more bytes than its part.
 */
USEARCH_EXPORT void usearch_geospatial_metadata_read_buffer(usearch_geospatial_metadata_read_buffer_t* arguments);

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

    /** The CPU class whose kits the index dispatched to on RAM and NVMe;
     *  @c usearch_cpu_class_unknown_k on VRAM. */
    usearch_cpu_class_t cpu_class;

    /** The GPU class whose kernels the index dispatched to on VRAM; @c usearch_gpu_class_unknown_k
     *  on RAM and NVMe. */
    usearch_gpu_class_t gpu_class;

    /** Live shards, each a graph of its own; a fresh index has one per memory domain or device it
     *  was placed on. */
    size_t shards_count;

    /** How the last load from a path read the parts onto the devices. */
    usearch_load_path_t load_path;
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
