/**
 *  @file c/usearch/dense_index.h
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The dense-modality C interface: Vector Search over equidimensional embeddings with
 *      inner-product, angular, euclidean, hamming, and jaccard metrics - all derived from a single
 *      inner-product kernel with pre-computed magnitudes.
 *
 *  Every batch input is a run of rows along the call's @c count: a base pointer and a byte stride
 *  between consecutive rows, whose width is the geometry the index froze - a zero stride repeats
 *  one row, and a single row is a count of one. Handles are created by a lifecycle verb naming
 *  both an action and a residency:
 *
 *      Action                           RAM                       VRAM                       NVMe
 *      init: start empty, mutable       usearch_dense_init_ram    usearch_dense_init_vram    none
 *      load: copy a checkpoint in       usearch_dense_load_ram    usearch_dense_load_vram    none
 *      view: serve one in place         usearch_dense_view_ram    none                       usearch_dense_view_nvme
 *      save: write a checkpoint         usearch_dense_save        usearch_dense_save         none
 *
 *  A viewed handle is read-only, and refuses every mutating verb with @c usearch_status_read_only_k.
 *
 *  @section dense_index_serialization Serialization
 *
 *  Checkpoints ride the SafeGraphs interchange format, and the shard count picks the layout. A
 *  handle holding one shard writes one monolith at the path: a self-describing file holding its
 *  vectors, neighbor lists and keys. A handle holding several writes a manifest at the path, a
 *  small JSON catalog of one monolith per shard named by the stem and @c -00001-of-0000S.safegraphs
 *  beside it, each with its shard's own node IDs, so neighbors never cross files. Only compaction
 *  merges graphs. Loading reads the content, never the extension, and opens one shard per part:
 *  part @c p is written whole on shard @c p, beside any placed shards the parts leave empty, and
 *  the shards past the placed ones open over the call's memory domains or devices in turn. A
 *  manifest records where its parts were written, never where a reader must keep them, so a
 *  checkpoint written from a mounted bucket spreads across a handle's own memory domains as it
 *  loads. An in-memory buffer of several shards leads with the same manifest, listing each part by
 *  its offset inside the buffer. A cancelled or failed save leaves a file without its header, which
 *  every reader rejects, and a cancelled or failed load creates no handle.
 *
 *  A file is read through a memory mapping, so a path and a buffer are the same bytes reached two
 *  ways. Loading copies them into the handle's own shards; viewing serves the mapping itself, so a
 *  checkpoint larger than memory costs only the pages a traversal touches.
 *
 *  @section dense_index_gpu_acceleration GPU Acceleration & Constraints
 *
 *  A VRAM handle builds and searches on the devices its placement names, its shards spread over
 *  them, several to a device when asked, and never replicated, so capacity is what those devices
 *  host together: the geometry freezes at init, and capacity grows on demand up to what they hold.
 *  Planes stay device-resident and never migrate, so an index outgrowing its devices fails rather
 *  than paging to the host - the answer to a dataset past one accelerator is to name more of them.
 *
 *  Devices are named by USearch ordinal, a position in @c usearch_gpu_devices, so one binary drives
 *  NVIDIA, AMD and Apple GPUs alike. An index spans one vendor and one generation; Apple's unified
 *  memory counts as device memory, since the GPU manages it, and its single device hosts one shard.
 *  Attention runs on NVIDIA and AMD devices, and the Apple handles refuse it with
 *  @c usearch_status_unsupported_k.
 *
 *  Inputs already on the executor's device in the right layout are read in place, pinned and
 *  managed host inputs upload directly, and only pageable host inputs pass through the executor's
 *  pinned bounce buffer. Outputs land in an arena on the executor's device, and a search's shards
 *  merge their rankings on the first shard's device, so on NVIDIA and AMD no match routes through
 *  the host, while Apple's host merge writes the unified memory both share.
 *
 *  @section dense_index_nvme_walks NVMe Walks & Mergers
 *
 *  An NVMe handle serves a checkpoint in place and walks it through whichever I/O engine can reach
 *  the devices its paths name - a filesystem path selects a kernel engine, a transport identifier
 *  a userspace one, and a FUSE mount over a bucket reads as an ordinary path.
 *  External-memory graphs are never modified in place, so the handle is a view, and every mutating
 *  verb fails on it. Bigger indexes come from merging, never mutating: shards build in RAM,
 *  serialize as monoliths, and enter a manifest - the compaction shape of LSM storage.
 *
 *  @section dense_index_concurrency Concurrency & Thread Safety
 *
 *  An insertion is all-or-nothing per call, and the entries of one call become visible at once on
 *  RAM and VRAM handles alike, so a concurrent search sees all of them or none. On a RAM handle
 *  every fallible step precedes the claim of the first entry identity, so a failed call publishes
 *  nothing and leaves the reported size untouched. A VRAM call that fails after its first launch
 *  publishes none of its entries either, and leaves the handle refusing every later mutation, while
 *  searches continue.
 *
 *  A handle may be shared across threads. On a RAM handle, insertions and searches overlap freely:
 *  searches read only entries published behind the visibility watermark and never write to the
 *  graph, while insertions synchronize through per-node locks and publish in order. A VRAM handle
 *  is driven through the streams of its executor - calls on distinct executors overlap, host
 *  threading adds nothing - and admits one insertion at a time. A viewed handle admits arbitrarily
 *  many concurrent searches, having nothing to mutate. Per-call outputs - stats, counts, the
 *  outcome - belong to their call and must not be shared between calls that are not final.
 *
 *  Refinement, absorption and clearing hold the handle closed to insertions for their whole call,
 *  and so does a save of a VRAM handle. None of them waits: one meeting an insertion in flight
 *  reports @c usearch_status_contended_k and leaves the handle as it was, an insertion arriving
 *  while one runs reports the same and writes nothing, and so does a second VRAM insertion beside
 *  the first. The caller retries, typically once its own insertions drained. A RAM save settles
 *  optimistically instead, reporting @c usearch_status_contended_k when an insertion was in flight
 *  or published under it, and concurrent readers keep searching throughout every save.
 *
 *  Every verb is safe beside every other on the same handle, with no rule to keep any of them from
 *  starting. Each RAM or VRAM handle carries one gate: insertions hold its shared side for their
 *  own call, and the verbs that empty or move entries in place - a clear, an absorb consuming a
 *  donor, a compaction, a split emptying its source, and the closing step of every refinement,
 *  absorb, compaction, split, load and VRAM save - take it alone or report
 *  @c usearch_status_contended_k, never waiting. Readers take no lock at all: a search, a product,
 *  an attention call, a lookup of vectors or a cursor pull reads one epoch word before and after
 *  its work, repeats once when a destructive step ran in between, and reports
 *  @c usearch_status_contended_k when another one did. Nothing a reader can reach is freed while
 *  the handle lives: an emptied shard keeps its memory until the handle is freed, and a count of a
 *  key stays exact throughout, as a compaction moves each key's entry with one atomic store.
 *
 *  The executor, not the handle, is what orders: a handle admits any number of concurrent
 *  callers, while calls on one executor run in the order they were submitted. Callers wanting
 *  genuine parallelism give each thread its own executor, and a @c NULL executor runs the call on
 *  the caller's own thread.
 *
 *  @section dense_index_residencies Verbs per Residency
 *
 *  Every verb this header declares is exported by the engine and by the stand-in alike, so a
 *  residency that lacks one answers @c usearch_status_unsupported_k rather than a missing symbol.
 *
 *  @c usearch_dense_insert, @c usearch_dense_count, @c usearch_dense_refine,
 *  @c usearch_dense_absorb and the serialization family answer on a RAM handle and on a VRAM one
 *  alike, an absorb taking donors of the handle's own residency. @c usearch_dense_search answers on
 *  both, a VRAM handle ranking its shards' matches on the arena's device, and so does
 *  @c usearch_dense_attend, which resolves value rows through @c value_lookup, a host function on
 *  RAM and a device one on NVIDIA and AMD; Apple's Metal answers it with
 *  @c usearch_status_unsupported_k, since its kernels cannot call the caller's function.
 *  @c usearch_dense_get, @c usearch_dense_save_buffer, @c usearch_dense_multiply and
 *  @c usearch_dense_maxsim write their outputs into the arena's device memory on NVIDIA and AMD,
 *  through the stream of the call's own executor, and into its host pages elsewhere.
 *
 *  A VRAM handle refuses a search with a @c predicate, since a host predicate cannot narrow a
 *  device walk, and an AMD or Apple one refuses a @c usearch_traversal_scan_k scan, both with
 *  @c usearch_status_unsupported_k; RAM and NVMe answer both. @c usearch_dense_cursor_init answers
 *  on RAM, while VRAM vectors are read back by key or by search. @c usearch_dense_split answers on
 *  both, @c usearch_dense_sample on RAM and NVMe, and @c usearch_dense_search_stream_init on none.
 *
 *  An NVMe handle is a view that answers every read verb as a RAM handle does, the cursor with its
 *  @c starts and @c links, @c usearch_dense_health and @c usearch_dense_sample included, its
 *  entries read through the executor's engines. Every mutating verb answers
 *  @c usearch_status_read_only_k, and saving answers @c usearch_status_unsupported_k, since a view
 *  already is its checkpoint.
 *
 *  @section dense_index_linear_algebra Approximate Products & Attention
 *
 *  Three opt-in verbs at the end of this header read the graph as linear algebra rather than as a
 *  search. Each is one graph walk per query row, like a search, differing only in what it keeps:
 *  a search keeps the nearest matches, @c usearch_dense_multiply every scored pair,
 *  @c usearch_dense_attend an online softmax over them, and @c usearch_dense_maxsim each row's
 *  best score, summed per key.
 *
 *  @c usearch_dense_multiply is an approximate A × Bᵀ in the shape of BLAS' @c ?GEMM, with the
 *  indexed vectors as B, and a sparse output: every pair the traversal scored, not only the
 *  nearest. Unlike a sparse @c ?GEMM, the output pattern is not implied by the operands' own
 *  pattern - it is whatever the walk touched - so the product is dense-in, sparse-out, and its
 *  sparsity follows the dots: the walk ranks by 1 − dot, so it skips the low and negative ones.
 *
 *  @c usearch_dense_attend folds a softmax over that same walk. A running maximum, partition,
 *  and weighted value sum update per scored batch and rescale whenever the maximum moves - the
 *  online-softmax recurrence of Flash Attention, with the graph replacing the linear sweep over
 *  key tiles. Softmax weights decay exponentially, so scoring a superset of the true top matches
 *  costs accuracy nothing, and neither verb pays for the ranking a search would. One index serves
 *  one layer's one key-value head: grouped-query and multi-head-latent attention already share
 *  their keys across query heads, so every query head is another row of the queries, and the
 *  heads of one token share its walk.
 *
 *  @c usearch_dense_maxsim is late interaction, and it is what @c values_per_key above one is for.
 *  A document is one key holding its token rows, a query is a run of token rows, and the score is
 *  the sum over query tokens of the best match each finds in a document. The graph stays over
 *  individual tokens under a plain inner product, so no set-to-set metric is ever evaluated during
 *  the walk - the grouping happens in the reduction, which is why this belongs here and not in a
 *  modality whose entries are whole matrices.
 */
#ifndef UNUM_USEARCH_DENSE_INDEX_H
#define UNUM_USEARCH_DENSE_INDEX_H

#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Vector Search Index over equidimensional dense representations.
 *
 *  @sa usearch_dense_init_ram
 *  @sa usearch_dense_insert
 *  @sa usearch_dense_search
 */
typedef struct usearch_dense_index_handle_t* usearch_dense_index_t;

/**
 *  @brief A walk over the graph yielding each entry's key with the keys it links to, every entry as
 *      stored or breadth-first from start keys, which export, migration, and visualization read.
 *
 *  Each shard's publication watermark is read as the walk enters that shard, so an entry once
 *  passed is never revisited, while a later shard may still carry entries published after the walk
 *  began. Each pull names the index again; an overlapping pull reports
 *  @c usearch_status_contended_k.
 */
typedef struct usearch_dense_cursor_handle_t* usearch_dense_cursor_t;

/**
 *  @brief A resumable search over one query: best-first traversal is naturally pausable, so the
 *      stream keeps the walk's frontier & visited set between pulls and yields matches lazily.
 *
 *  Callers that post-filter aggressively pull until satisfied instead of guessing @c wanted upfront.
 *  The stream holds state, never a thread, so any number of them may stay open on one executor.
 *  Each pull names the index again; an overlapping pull reports @c usearch_status_contended_k.
 */
typedef struct usearch_dense_search_stream_handle_t* usearch_dense_search_stream_t;

/**
 *  @brief One refinement in flight, optionally absorbing donors or splitting into receivers first.
 *
 *  It holds everything the pass needs between calls, so a cancelled
 *  @c usearch_dense_refinement_next continues where it stopped, and nothing about it stays in the
 *  index once it finishes. The index admits no insertion and no save while the handle lives;
 *  searches proceed as during @c usearch_dense_refine. Each call names the index and the executor
 *  it was opened with.
 */
typedef struct usearch_dense_refinement_handle_t* usearch_dense_refinement_t;

/** The dense metric kinds supported out of the box, with SIMD kernels dispatched by the
 *  machine's capabilities. The last two read their vectors as @c usearch_scalar_u1x8_k bit sets
 *  and refuse any other quantization; the first four refuse that one. */
typedef enum usearch_dense_metric_t {

    /** No kind named - an unset field, refused by every constructor. Arbitrary metrics belong to
     *  the C++ baseline tier, whose index takes any caller-supplied batched metric. */
    usearch_dense_metric_unknown_k = 0,

    /** Sum of elementwise products, unnormalized and scale-sensitive; the only kind the
     *  linear-algebra verbs accept, since a product and a softmax both need logits rather
     *  than distances. */
    usearch_dense_metric_inner_k = 1,

    /** Cosine distance in [0, 2], zero for identical directions and blind to magnitude,
     *  which is precomputed once per vector at insertion rather than per comparison. */
    usearch_dense_metric_angular_k = 2,

    /** L2 distance, non-negative and zero only for identical vectors - the scale-sensitive
     *  counterpart to angular, where a doubled vector is far from its original rather than
     *  identical to it. */
    usearch_dense_metric_euclidean_k = 3,

    /** Count of differing bits, so its range is the dimensionality and every bit weighs
     *  the same. */
    usearch_dense_metric_hamming_k = 4,

    /** Differing bits over union bits in [0, 1], ignoring the zeros both vectors share - the
     *  kind for sparse sets, where Hamming would call two nearly-empty vectors identical. */
    usearch_dense_metric_jaccard_k = 5,
} usearch_dense_metric_t;

/** The canonical spelling of a metric kind, as @c usearch_dense_metric_inner_k spells
 *  "inner"; the same lifts @c usearch_scalar_name and its kin provide for the shared sets. */
USEARCH_EXPORT char const* usearch_dense_metric_name(usearch_dense_metric_t metric);

/** Reads a metric kind back; @c usearch_dense_metric_unknown_k when unrecognized. */
USEARCH_EXPORT usearch_dense_metric_t usearch_dense_metric_from_name(usearch_bytes_view_t name);

/**
 *  @brief The arguments of @c usearch_dense_init_ram - the geometry and metric the empty index
 *      freezes at init, and where its shards live.
 *
 *  Stored vectors round up to whole steps of the distance kit the build picked, zero-tailed, so no
 *  step load straddles a vector's alignment; the memory is slightly larger for it, and that is not
 *  a knob. Saved payloads keep logical vectors, so they move between builds with different kits.
 */
typedef struct usearch_dense_init_ram_t {

    /** Output for the new index. */
    usearch_dense_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The metric kind used for distance calculations between vectors. */
    usearch_dense_metric_t metric;

    /** The scalar type vectors are stored & quantized into. A build compiles the first
     *  @c USEARCH_DENSE_DTYPES of bf16, u1x8, i8, e4m3, f16, u8, e5m2, i4x2, u4x2, e3m2 and e2m3,
     *  four by default and all eleven in the prebuilt packages, and refuses the rest with
     *  @c usearch_status_unsupported_k. */
    usearch_scalar_t quantization;

    /** The number of dimensions in the indexed vectors. */
    size_t dimensions;

    /** The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** Upper bound on entries sharing one key; zero is unbounded. */
    size_t values_per_key;

    /** The memory domains to place one shard on apiece. */
    usearch_ram_options_t ram;
} usearch_dense_init_ram_t;

/**
 *  @brief Creates an empty, mutable index in host memory, its geometry frozen, so it takes vectors
 *      without a reservation.
 *
 *  @par Statuses:
 *  - @c usearch_status_config_mismatch_k Zero @c dimensions, @c dimensions a packed
 *    quantization like @c usearch_scalar_u1x8_k leaves a storage value partly filled by, or a
 *    @c connectivity that is not a power of two.
 */
USEARCH_EXPORT void usearch_dense_init_ram(usearch_dense_init_ram_t* arguments);

/** The arguments of @c usearch_dense_init_vram, which match @c usearch_dense_init_ram_t but for
 *  the placement. */
typedef struct usearch_dense_init_vram_t {

    /** Output for the new index. */
    usearch_dense_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The metric kind used for distance calculations between vectors. */
    usearch_dense_metric_t metric;

    /** The scalar type vectors are stored & quantized into. */
    usearch_scalar_t quantization;

    /** The number of dimensions in the indexed vectors. */
    size_t dimensions;

    /** The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** Upper bound on entries sharing one key; zero is unbounded. */
    size_t values_per_key;

    /** The devices to place one shard on apiece, by USearch ordinal. */
    usearch_vram_options_t vram;
} usearch_dense_init_vram_t;

/**
 *  @brief Creates an empty, mutable index in device memory, its geometry frozen, so it takes
 *      vectors without a reservation. @ref dense_index_gpu_acceleration
 *
 *  @par Statuses:
 *  - @c usearch_status_config_mismatch_k Zero @c dimensions, a @c connectivity that is not a
 *    power of two, or on NVIDIA and AMD a geometry whose walks cannot fit the device's shared
 *    memory.
 */
USEARCH_EXPORT void usearch_dense_init_vram(usearch_dense_init_vram_t* arguments);

/** The arguments of @c usearch_dense_load_ram. */
typedef struct usearch_dense_load_ram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_dense_index_t* index;

    /** The executor the read fans out over, one shard per thread. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The manifest of a sharded checkpoint, or a single @c .safegraphs monolith. */
    usearch_bytes_view_t path;

    /** The memory domains to place the parts on, part @c p loading into shard @c p, those past the
     *  named domains opened over them in turn. */
    usearch_ram_options_t ram;

    /** Neighbor IDs per node, at least the checkpoint's, whose wider rows keep the stored neighbors
     *  and leave the rest empty for a refine to fill; zero keeps the checkpoint's. */
    size_t connectivity;

    /** Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded. */
    size_t values_per_key;

    /** The @b optional handler, called with shards placed. */
    usearch_progress_t progress;
} usearch_dense_load_ram_t;

/**
 *  @brief Copies a checkpoint into a new, mutable index in host memory, shard by shard.
 *  @ref dense_index_serialization
 *
 *  The geometry and metric come from the checkpoint's own headers, and only a non-zero
 *  @c connectivity widens the rows. Point @c path at the @c .safegraphs.json manifest, never at one
 *  of the parts it enumerates. The manifest is a small catalog that names every part and resolves
 *  each one relative to itself, so a checkpoint spread over a hundred files is opened by naming a
 *  single one. Naming a part directly is legal and loads exactly that part as a monolith, which is
 *  a fragment of the index rather than the index.
 *
 *  Where the parts end up is the call's own memory domains, never the manifest's: part @c p is
 *  written whole on shard @c p, and shards past the placed ones open over those domains in turn, so
 *  a checkpoint of eight parts loaded naming four domains places two parts on each.
 *
 *  A cancelled or failed load reports its status and creates no handle.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c path, or a checkpoint or part path that
 *    does not parse.
 *  - @c usearch_status_malformed_k The checkpoint lists no part, or a part, its header or its key
 *    column does not parse, or its connectivity or dimensions are unusable.
 *  - @c usearch_status_unknown_kind_k An unknown quantization, a metric its quantization does not
 *    score, or a kernel a shard cannot finalize.
 *  - @c usearch_status_unsupported_k A quantization this build leaves out, or a pinned class the
 *    CPU does not admit.
 *  - @c usearch_status_uninitialized_k No memory domain to place a shard on, or the executor holds
 *    no pool workers.
 *  - @c usearch_status_config_mismatch_k A memory domain refusing a shard, a @c connectivity
 *    narrower than the checkpoint's, or the checkpoint's geometry, dtypes or metric differ from the
 *    shards'.
 *  - @c usearch_status_device_k The executor is not a RAM executor.
 *  - @c usearch_status_contended_k Another call holds the executor, or its pool runs another
 *    dispatch.
 *  - @c usearch_status_overflow_k A part passes a shard's bound.
 */
USEARCH_EXPORT void usearch_dense_load_ram(usearch_dense_load_ram_t* arguments);

/** The arguments of @c usearch_dense_load_vram. */
typedef struct usearch_dense_load_vram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_dense_index_t* index;

    /** The executor the upload runs on, one stream per device. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The manifest of a sharded checkpoint, or a single @c .safegraphs monolith. */
    usearch_bytes_view_t path;

    /** The devices to place the parts on, part @c p loading into shard @c p, those past the named
     *  devices opened over them in turn. */
    usearch_vram_options_t vram;

    /** Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded. */
    size_t values_per_key;

    /** The @b optional handler, called with shards placed. */
    usearch_progress_t progress;
} usearch_dense_load_vram_t;

/**
 *  Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
 *  @c usearch_dense_load_ram. @ref dense_index_serialization
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c path, or a checkpoint or part path that
 *    does not parse.
 *  - @c usearch_status_malformed_k The checkpoint lists no part, or a part, its header or its key
 *    column does not parse, or its connectivity or dimensions are unusable.
 *  - @c usearch_status_unknown_kind_k An unknown quantization, a metric its quantization does not
 *    score, or a kernel a shard cannot finalize.
 *  - @c usearch_status_unsupported_k A quantization this build leaves out, or devices that do not
 *    run the pinned class.
 *  - @c usearch_status_config_mismatch_k Devices of more than one vendor or generation, the
 *    executor driving no stream on some shard's device, or the checkpoint's geometry, dtypes or
 *    metric differing from the shards'.
 *  - @c usearch_status_device_k An ordinal past the listing, devices this build carries no kernels
 *    for, or an executor that is not a VRAM executor.
 *  - @c usearch_status_contended_k Another call holds the executor.
 *  - @c usearch_status_overflow_k A part passes a shard's bound.
 */
USEARCH_EXPORT void usearch_dense_load_vram(usearch_dense_load_vram_t* arguments);

/** The arguments of @c usearch_dense_load_ram_buffer. */
typedef struct usearch_dense_load_ram_buffer_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_dense_index_t* index;

    /** The executor the copy fans out over. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Source holding one serialized monolith, or the manifest of several parts leading them. */
    usearch_bytes_view_t buffer;

    /** The memory domains to place shards on; part @c p is written whole on shard @c p. */
    usearch_ram_options_t ram;

    /** Neighbor IDs per node, at least the checkpoint's, as @c usearch_dense_load_ram takes it. */
    size_t connectivity;

    /** Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded. */
    size_t values_per_key;

    /** The @b optional handler, called with bytes placed. */
    usearch_progress_t progress;
} usearch_dense_load_ram_buffer_t;

/**
 *  @brief Copies a checkpoint from a memory buffer into a new, mutable index in host memory.
 *
 *  This reports and cancels like its path-taking sibling and creates no handle when stopped. That
 *  sibling reaches the same bytes through a mapping, so @c buffer needs no file behind it and a
 *  mapped file needs no copy into one.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c buffer.
 *  - @c usearch_status_malformed_k The manifest lists no part, or a part, its header or its key
 *    column does not parse, or its connectivity or dimensions are unusable.
 *  - @c usearch_status_unknown_kind_k An unknown quantization, a metric its quantization does not
 *    score, or a kernel a shard cannot finalize.
 *  - @c usearch_status_unsupported_k A quantization this build leaves out, or a pinned class the
 *    CPU does not admit.
 *  - @c usearch_status_uninitialized_k No memory domain to place a shard on, or the executor holds
 *    no pool workers.
 *  - @c usearch_status_config_mismatch_k A memory domain refusing a shard, a @c connectivity
 *    narrower than the checkpoint's, or the checkpoint's geometry, dtypes or metric differ from the
 *    shards'.
 *  - @c usearch_status_device_k The executor is not a RAM executor.
 *  - @c usearch_status_contended_k Another call holds the executor, or its pool runs another
 *    dispatch.
 *  - @c usearch_status_overflow_k A part passes a shard's bound.
 */
USEARCH_EXPORT void usearch_dense_load_ram_buffer(usearch_dense_load_ram_buffer_t* arguments);

/** The arguments of @c usearch_dense_load_vram_buffer. */
typedef struct usearch_dense_load_vram_buffer_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_dense_index_t* index;

    /** The executor the upload runs on, one stream per device. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Source holding one serialized monolith, or the manifest of several parts leading them, in
     *  host or device memory; a device one is copied to the host first, so the call holds twice
     *  its bytes while it runs. */
    usearch_bytes_view_t buffer;

    /** The devices to place shards on; part @c p is written whole on shard @c p. */
    usearch_vram_options_t vram;

    /** Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded. */
    size_t values_per_key;

    /** The @b optional handler, called with bytes placed. */
    usearch_progress_t progress;
} usearch_dense_load_vram_buffer_t;

/**
 *  Copies a checkpoint from a memory buffer into a new, mutable index in device memory, reporting
 *  and cancelling like @c usearch_dense_load_ram_buffer.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c buffer.
 *  - @c usearch_status_malformed_k The manifest lists no part, or a part, its header or its key
 *    column does not parse, or its connectivity or dimensions are unusable.
 *  - @c usearch_status_unknown_kind_k An unknown quantization, a metric its quantization does not
 *    score, or a kernel a shard cannot finalize.
 *  - @c usearch_status_unsupported_k A quantization this build leaves out, or devices that do not
 *    run the pinned class.
 *  - @c usearch_status_config_mismatch_k Devices of more than one vendor or generation, the
 *    executor driving no stream on some shard's device, or the checkpoint's geometry, dtypes or
 *    metric differing from the shards'.
 *  - @c usearch_status_device_k An ordinal past the listing, devices this build carries no kernels
 *    for, or an executor that is not a VRAM executor.
 *  - @c usearch_status_contended_k Another call holds the executor.
 *  - @c usearch_status_overflow_k A part passes a shard's bound.
 */
USEARCH_EXPORT void usearch_dense_load_vram_buffer(usearch_dense_load_vram_buffer_t* arguments);

/** The arguments of @c usearch_dense_view_ram. */
typedef struct usearch_dense_view_ram_t {

    /** Output for the new, read-only index. */
    usearch_dense_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The manifest of a sharded checkpoint, or a single @c .safegraphs monolith. */
    usearch_bytes_view_t path;
} usearch_dense_view_ram_t;

/**
 *  @brief Serves a checkpoint in place through a memory mapping, as a read-only index.
 *  @ref dense_index_serialization
 *
 *  Nothing is copied, so opening costs the headers alone and a traversal pays only for the pages
 *  it touches. Every mutating verb on the handle reports @c usearch_status_read_only_k, and the
 *  files must stay unmodified until the handle is freed.
 *
 *  @par Statuses:
 *  - @c usearch_status_unsupported_k Always, as RAM shards cannot serve a mapping in place yet.
 */
USEARCH_EXPORT void usearch_dense_view_ram(usearch_dense_view_ram_t* arguments);

/** The arguments of @c usearch_dense_view_nvme. */
typedef struct usearch_dense_view_nvme_t {

    /** Output for the new, read-only index, written by the time the call is final. */
    usearch_dense_index_t* index;

    /** The NVMe executor the call is ordered on; the headers come through blocking reads. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The manifest of a sharded checkpoint, or a single @c .safegraphs monolith. */
    usearch_bytes_view_t path;

    /** The devices the parts are served from; naming none serves each where its manifest points. */
    usearch_nvme_options_t nvme;

    /** The @b optional RAM the view may hold beyond its headers, borrowed for the call: a key
     *  table on the first of @c ram->memory_domains within @c ram->memory_limit; @c NULL holds
     *  nothing per entry, so key lookups and hints refuse. */
    usearch_ram_options_t const* ram;
} usearch_dense_view_nvme_t;

/**
 *  @brief Serves a checkpoint in place from NVMe drives as a read-only index, every later call
 *      reading it through its executor's engines. @ref dense_index_nvme_walks
 *
 *  Opening reads the headers, and with @c ram every part's key column once, on @c executor's
 *  engines, into a key table of about 11 bytes per entry: an 8-byte fingerprint and node per
 *  slot, a third more slots than entries. Every mutating verb on the handle reports
 *  @c usearch_status_read_only_k, and the parts must stay unmodified until the handle is freed.
 *
 *  Without a key table, @c usearch_dense_count, @c usearch_dense_get, @c usearch_dense_sample, a
 *  MaxSim and hinted walks refuse with @c usearch_status_unsupported_k rather than scan every key.
 *
 *  @par Statuses:
 *  - @c usearch_status_device_k @c executor is not an NVMe executor.
 *  - @c usearch_status_invalid_argument_k @c ram without an @c executor to scan the keys on.
 *  - @c usearch_status_out_of_memory_k The key table passes @c ram->memory_limit.
 *  - @c usearch_status_invalid_argument_k An empty @c path, @c NULL paths for a nonzero count,
 *    or a path that does not parse or names a base off its medium's block.
 *  - @c usearch_status_config_mismatch_k Parts disagreeing with each other, with the scalar or
 *    with @c nvme.
 *  - @c usearch_status_unsupported_k A transport identifier in a build without SPDK, a
 *    quantization this build leaves out, or a pinned class the CPU does not admit.
 *  - @c usearch_status_malformed_k The manifest names no part, or a manifest or header does not
 *    parse.
 *  - @c usearch_status_unknown_kind_k An unknown quantization, a metric its quantization does not
 *    score, or a kernel the stored scalar's kits cannot finalize.
 */
USEARCH_EXPORT void usearch_dense_view_nvme(usearch_dense_view_nvme_t* arguments);

/**
 *  @brief Frees the resources associated with the index; @c NULL is safe, and it cannot fail, so
 *      takes no outcome.
 *  @param[in,out] index The handle to be freed; invalid afterwards.
 *
 *  Waits until every deferred call naming the handle is final.
 */
USEARCH_EXPORT void usearch_dense_free(usearch_dense_index_t index);

/** The arguments of @c usearch_dense_reserve. */
typedef struct usearch_dense_reserve_t {

    /** The handle to be resized. */
    usearch_dense_index_t index;

    /** The executor the growth runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The desired total capacity, including the current size. */
    size_t capacity;
} usearch_dense_reserve_t;

/**
 *  Grows the shards ahead of a number of incoming vectors - a capacity hint, since insertions grow
 *  them on demand and init fixed the geometry already.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k The handle is an NVMe view.
 *  - @c usearch_status_uninitialized_k No shard is placed.
 *  - @c usearch_status_unknown_kind_k The stored scalar's kits cannot finalize the metric.
 *  - @c usearch_status_config_mismatch_k On VRAM, a shard's neighbors live on another device or no
 *    walk fits the device.
 *  - @c usearch_status_overflow_k @c capacity passes a shard's ceiling, its ID space, its chunk
 *    directory or its key column.
 */
USEARCH_EXPORT void usearch_dense_reserve(usearch_dense_reserve_t* arguments);

/** The arguments of @c usearch_dense_clear. */
typedef struct usearch_dense_clear_t {

    /** The handle to be cleared. */
    usearch_dense_index_t index;

    /** The executor the erasure runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;
} usearch_dense_clear_t;

/**
 *  @brief Erases all entries, keeping the reserved capacity and frozen geometry. An insertion in
 *      flight, or a refinement, absorb, compaction, split or load holding the handle, refuses it
 *      with @c usearch_status_contended_k; a search it reaches under repeats or reports the same.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k The handle is an NVMe view.
 *  - @c usearch_status_contended_k A call in flight, a batch or load holding claims, or a plan
 *    holding the handle closed.
 */
USEARCH_EXPORT void usearch_dense_clear(usearch_dense_clear_t* arguments);

/** The arguments of @c usearch_dense_insert. */
typedef struct usearch_dense_insert_t {

    /** The handle to be populated. */
    usearch_dense_index_t index;

    /** The executor driving the batch; its kind must match the handle's residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Entries to admit - the batch axis of @c keys and @c vectors. */
    size_t count;

    /** The identities to admit the entries under, one per entry. */
    usearch_keys_view_t keys;

    /** The vectors to admit, converted on ingest to the stored type. */
    usearch_rows_view_t vectors;

    /** The @b optional breadth of the candidates the build walk explores; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries admitted, which reports and cannot cancel. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_dense_insert_t;

/**
 *  @brief Adds a run of vectors under their keys - the native batch operation; a single vector is
 *      a count of one. Overlaps safely with concurrent searches on the same handle.
 *  @ref dense_index_concurrency
 *
 *  An insertion is all-or-nothing per call, with or without a handler, and its entries become
 *  visible at once on RAM and VRAM handles alike, so a concurrent search sees every entry of the
 *  call or none of them. Every entry is claimed before the handler's first report, and a claimed
 *  entry cannot be withdrawn, so the handler reports progress but cannot cancel: a returned
 *  @c usearch_cancel_k is ignored.
 *
 *  On a RAM handle every fallible step runs before the first entry identity is claimed, so a failed
 *  call leaves the index untouched. A VRAM call that fails after its first launch publishes none of
 *  its entries and leaves the handle refusing every later mutation with that failure, while
 *  searches continue.
 *
 *  @code{.c}
 *  usearch_outcome_t outcome = {0};
 *  usearch_dense_insert(&(usearch_dense_insert_t){
 *      .index = index, .executor = executor, .outcome = &outcome,
 *      .count = 256, .keys = {keys, sizeof(usearch_key_t)},
 *      .vectors = {rows_base, 128 * sizeof(float), 128, usearch_scalar_f32_k}});
 *  if (outcome.status != usearch_status_success_k) report(outcome.message);
 *  @endcode
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k The handle is a view.
 *  - @c usearch_status_contended_k A refinement, absorption or clear holds the handle, or on
 *    VRAM a save or another insertion does; nothing is inserted.
 */
USEARCH_EXPORT void usearch_dense_insert(usearch_dense_insert_t* arguments);

/** The arguments of @c usearch_dense_refine. */
typedef struct usearch_dense_refine_t {

    /** The handle to be refined. */
    usearch_dense_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The @b optional output for what the call did to the graph, as @c usearch_refine_report_t
     *  counts it; @c NULL skips reporting. */
    usearch_refine_report_t* report;

    /** The @b optional breadth of the candidates the re-selection walk explores; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with nodes re-selected; a cancel stops the re-selection,
     *  still heals, and answers @c usearch_status_cancelled_k. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_dense_refine_t;

/**
 *  @brief Re-selects the neighbors of every stored entry against the finished graph, improving
 *      neighbors chosen when the graph around them was still incomplete.
 *
 *  This is a build step rather than a repair. A cheap insertion pass followed by one refinement
 *  reaches a better graph, in less total time, than a single expensive insertion pass - so a wide
 *  @c expansion_factor belongs here rather than in @c usearch_dense_insert.
 *
 *  Cancelling is safe, which is what makes a whole-graph sweep affordable. Re-selecting a node's
 *  neighbors is idempotent and claims no identity, so a stopped pass leaves the graph correct and
 *  merely less improved: nothing unwinds, and the call answers @c usearch_status_cancelled_k. The
 *  index keeps nothing of a stopped pass, so the next call starts at the first entry again; a
 *  caller wanting to continue where a cancel stopped holds a @c usearch_dense_refinement_t instead.
 *  A caller loops until the report's @c changed falls below whatever change rate they consider
 *  settled.
 *
 *  Re-selection alone can leave an island, an entry no search reaches, so every call closes by
 *  re-linking them: unreachable entries re-select from scratch, and whatever the diversity judgment
 *  still drops is reconnected to a reachable one. A cancelled call skips the rest of the
 *  re-selection but still re-links, so it never leaves entries unreachable.
 *
 *  On a RAM handle searches may run throughout: re-selection publishes one node at a time and never
 *  removes a node, so a concurrent walk sees either the old neighbors or the new ones. On a VRAM
 *  handle the merge writes neighbors in place, so searches may not overlap. Insertions never
 *  overlap: one in flight refuses the call with @c usearch_status_contended_k, and one arriving
 *  meanwhile is refused the same way, as is a VRAM save. A viewed handle refuses with
 *  @c usearch_status_read_only_k, like every other mutating verb.
 *
 *  @par Statuses:
 *  - @c usearch_status_device_k The executor's kind does not match the residency.
 *  - @c usearch_status_read_only_k The handle is an NVMe view.
 *  - @c usearch_status_uninitialized_k No shard is placed, the executor holds no threads, or a VRAM
 *    handle keeps such an earlier failure.
 *  - @c usearch_status_contended_k An insertion, load, save, refinement or absorption holds the
 *    handle, or another call holds the executor or its pool.
 *  - @c usearch_status_config_mismatch_k The executor drives no stream on some VRAM shard's
 *    device, or a VRAM handle keeps such an earlier failure.
 *  - @c usearch_status_overflow_k A queue capacity wraps when rounded, or a VRAM handle keeps an
 *    earlier overflow.
 *  - @c usearch_status_unsupported_k or @c usearch_status_unsupported_layout_k A VRAM handle
 *    keeps such an earlier failure.
 */
USEARCH_EXPORT void usearch_dense_refine(usearch_dense_refine_t* arguments);

/** The arguments of @c usearch_dense_health. */
typedef struct usearch_dense_health_t {

    /** The handle to be audited. */
    usearch_dense_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output for the audit; zeroed on failure. */
    usearch_reachability_t* reachability;
} usearch_dense_health_t;

/**
 *  Audits how much of the graph a search can reach: forward reachability from as many landmarks
 *  as one search draws, swept level by level. Read-only, so concurrent insertions merely leave
 *  fresh entries unaudited.
 *
 *  @par Statuses:
 *  - @c usearch_status_device_k The executor's kind does not match the residency.
 *  - @c usearch_status_uninitialized_k No shard is placed, the executor holds no pool workers or
 *    streams, or a VRAM handle keeps such an earlier failure.
 *  - @c usearch_status_contended_k Another call holds the executor or its pool, or a destructive
 *    window overlapped the audit twice.
 *  - @c usearch_status_config_mismatch_k The executor drives no stream on some VRAM shard's
 *    device, an NVMe medium the engines cannot read, or a VRAM handle keeps such an earlier
 *    failure.
 *  - @c usearch_status_overflow_k A VRAM handle keeps an earlier overflow.
 *  - @c usearch_status_unsupported_k or @c usearch_status_unsupported_layout_k A VRAM handle
 *    keeps such an earlier failure.
 */
USEARCH_EXPORT void usearch_dense_health(usearch_dense_health_t* arguments);

/** The arguments of @c usearch_dense_search. */
typedef struct usearch_dense_search_t {

    /** The handle to be queried. */
    usearch_dense_index_t index;

    /** The executor driving the batch; its kind must match the handle's residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Queries to resolve; the outputs have @c count rows of @c wanted entries. */
    size_t count;

    /** The query vectors to resolve, one per query. */
    usearch_rows_view_t queries;

    /** Output for the run, inside @c arena. */
    usearch_matches_view_t* matches;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /** Columns per row, and the upper bound on matches - the "k" in "kANN". */
    size_t wanted;

    /** The @b optional breadth of the candidates a walk explores; zero derives, a scan ignores
     *  it. */
    size_t expansion_factor;

    /** How entries are reached: a walk, the default, or an exact scan of every entry, on all but
     *  AMD and Apple VRAM. */
    usearch_traversal_t traversal;

    /** The @b optional predicate, asked about a scored quad of candidates at a time. */
    usearch_predicate_t predicate;

    /** The @b optional keys each query's walk starts from ahead of its random landmarks, one row
     *  of @c counts[query] keys per query, as a search writes its matches, at most one per
     *  landmark; scores are not read, a scan ignores them, and keys no entry holds are skipped
     *  and counted in the stats. */
    usearch_matches_view_t const* hints;

    /** The @b optional handler, called with queries resolved. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_dense_search_t;

/**
 *  @brief Performs k-Nearest Neighbors Search for a run of queries, approximate by walking the
 *      graph or exact by scanning every entry - the native batch operation, parallelized
 *      internally.
 *
 *  One query is a count of one. Never writes to the index, so any number may overlap with
 *  insertions. @ref dense_index_concurrency
 *
 *  A scan walks no graph. On RAM the queries split into blocks, each taken by one of the executor's
 *  threads: a block of many queries packs once into the operand layout of the CPU's matrix kernels
 *  and streams every stored vector against the pack where it lies, while a lone query takes the
 *  walk's own 4-way kernels. On NVMe the queries pack once and the threads split the value column
 *  instead, each run written beside its keys, so the column is read once and in order. Either way
 *  each query's matches merge into its output row as the pass advances. On NVIDIA VRAM the batch
 *  scans every shard's vectors on the device, where the shards' matches merge too. Every residency
 *  ranks by the distances its walk reports. A @c predicate is asked only about entries closer than
 *  the worst match the query already holds, which makes the scan the fallback for predicates too
 *  selective for a walk to cross.
 *
 *  A @c predicate narrows the matches to the entries it admits. Non-matching entries still route
 *  the traversal rather than terminating it, so recall does not collapse at low selectivity - the
 *  walk crosses rejected regions to reach admitted ones behind them. The cost is that the
 *  predicate judges far more keys than @c wanted, every scored candidate, asked about a quad of
 *  them at a time, so it belongs on the caller's fast path - and it is invoked concurrently from
 *  the executor's threads, so it must tolerate that.
 *
 *  @c hints hand each query's walk landmarks the caller already trusts, such as the matches of the
 *  same queries over another index of the same keys. A query takes as many keys as a walk draws
 *  landmarks, the connectivity rounded up to a multiple of four, and ignores the rest; the entries
 *  stored under them take the landmark slots ahead of the random draws, in hint order, as many as
 *  fit. They are advisory, so a key no entry holds is skipped and counted in @c missing_hints. An
 *  NVMe view resolves them through its key table, confirming each entry with one key read.
 *
 *  A cancelled search reports @c usearch_status_cancelled_k and leaves @c matches @c NULL, since a
 *  query resolved before the stop is indistinguishable from one that never ran. A zero @c wanted,
 *  or an index holding no entries, succeeds without walking and writes only the counts, every one
 *  zero.
 *
 *  @code{.c}
 *  static void admits_tenant(usearch_key_t const* keys, usearch_admission_t* admissions, size_t count, void* state) {
 *      for (size_t i = 0; i != count; ++i)
 *          admissions[i] = keys[i].bytes[0] == *(uint8_t const*)state ? usearch_admit_k : usearch_reject_k;
 *  }
 *
 *  uint8_t tenant = 7;
 *  usearch_matches_view_t matches;
 *  usearch_outcome_t outcome = {0};
 *  usearch_dense_search(&(usearch_dense_search_t){
 *      .index = index, .executor = executor, .outcome = &outcome,
 *      .count = 32, .queries = {rows_base, 256 * sizeof(float), 256, usearch_scalar_f32_k},
 *      .matches = &matches, .arena = arena, .wanted = 10,
 *      .predicate = {admits_tenant, &tenant}});
 *  // Query 3's nearest key, valid until the next call given arena.
 *  usearch_key_t nearest = matches.keys[3 * matches.columns];
 *  @endcode
 *
 *  @par Statuses:
 *  - @c usearch_status_unknown_kind_k A @c traversal other than a walk or a scan.
 *  - @c usearch_status_unsupported_k A scan of an AMD or Apple VRAM handle, a @c predicate or
 *    @c hints on any VRAM handle, or @c hints on an NVMe view holding no key table.
 *  - @c usearch_status_invalid_argument_k @c hints naming no counts, or columns but no keys.
 *  - @c usearch_status_shape_mismatch_k A hinted query counts more keys than the hints' columns.
 */
USEARCH_EXPORT void usearch_dense_search(usearch_dense_search_t* arguments);

/** The arguments of @c usearch_dense_count. */
typedef struct usearch_dense_count_t {

    /** The handle to be queried. */
    usearch_dense_index_t index;

    /** The executor the lookup runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are counted. */
    usearch_key_t key;

    /** Output for the entries sharing @c key; zero means the index holds none. */
    size_t* entries;
} usearch_dense_count_t;

/**
 *  @brief Counts the entries stored under one key.
 *
 *  Every entry is one row of @c dimensions, so the count is the whole extent and there is nothing
 *  further to report - the sibling modalities report stats here only because their entries vary
 *  in size.
 *
 *  @par Statuses:
 *  - @c usearch_status_device_k The executor's kind does not match the residency.
 *  - @c usearch_status_uninitialized_k The executor holds no pool workers.
 *  - @c usearch_status_contended_k Another call holds the executor, or its pool runs another
 *    dispatch.
 *  - @c usearch_status_config_mismatch_k An NVMe medium the engines cannot read.
 */
USEARCH_EXPORT void usearch_dense_count(usearch_dense_count_t* arguments);

/** The arguments of @c usearch_dense_get. */
typedef struct usearch_dense_get_t {

    /** The handle to be queried. */
    usearch_dense_index_t index;

    /** The executor whose stream or thread copies the vectors; its kind must match the
     *  residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are exported. */
    usearch_key_t key;

    /** Output for every vector, packed at their natural width, inside @c arena. */
    usearch_rows_view_t* vectors;

    /** The @b optional output for the vectors written; zero when the key is absent. */
    size_t* written;

    /** Memory for @c vectors, rewound by the call. */
    usearch_arena_t arena;

    /** The scalar type to export as, converted from storage where it differs. */
    usearch_scalar_t outputs_scalar;
} usearch_dense_get_t;

/**
 *  @brief Retrieves every vector stored under one key into @c arena, one row per entry sharing the
 *      key, shard by shard and in insertion order within each - one call, however many there are.
 *
 *  The vectors span @c written entries; an absent key leaves @c vectors->data @c NULL.
 *
 *  @par Statuses:
 *  - @c usearch_status_unknown_kind_k An unknown @c outputs_scalar.
 *  - @c usearch_status_shape_mismatch_k Sub-byte vectors ending inside a byte.
 *  - @c usearch_status_invalid_argument_k A @c NULL @c arena.
 *  - @c usearch_status_device_k An arena or an executor on another residency or device.
 *  - @c usearch_status_contended_k The arena serves another call, another call holds the
 *    executor or its pool, or a destructive window overlapped the read twice.
 *  - @c usearch_status_uninitialized_k The executor holds no pool workers, or a VRAM index has no
 *    shards.
 *  - @c usearch_status_config_mismatch_k The executor drives no stream on some shard's device,
 *    or an NVMe medium the engines cannot read.
 */
USEARCH_EXPORT void usearch_dense_get(usearch_dense_get_t* arguments);

/** The arguments of @c usearch_dense_absorb. */
typedef struct usearch_dense_absorb_t {

    /** The handle absorbing the donors. */
    usearch_dense_index_t index;

    /** The executor each donor's move and the closing refinement fan out over. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Donor handles to absorb. */
    size_t count;

    /** Array of @c count handles, each consumed and nulled once its entries are published here. */
    usearch_dense_index_t* donors;

    /** The @b optional output for what the call did to the graph, as @c usearch_refine_report_t
     *  counts it; @c NULL skips reporting. */
    usearch_refine_report_t* report;

    /** The @b optional breadth of the candidates of the closing refine's walks; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with nodes the closing refine re-selected; a cancel stops it
     *  as it stops @c usearch_dense_refine, never a move. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_dense_absorb_t;

/**
 *  @brief Absorbs every handle in @c donors into @c index, one after another, keeping their graphs
 *      rather than re-inserting their vectors and linking each in by walks across. Donor handles
 *      are consumed and must not be used afterwards, not even for @c usearch_dense_free.
 *
 *  The donors land one after another, each all-or-nothing: its entries append to @c index shard by
 *  shard with their edges rebased, publish at once, and the donor is consumed, so at most one
 *  donor's entries are ever held twice. One refinement then walks every shard across the parts the
 *  donors appended, round by round, so a group of donors costs one refinement rather than one per
 *  donor. Readers of a RAM @c index may keep searching throughout, since unpublished entries hide
 *  behind the watermark, while a VRAM one rewrites neighbors in place and admits no overlapping
 *  search, as refinement does. Insertions into @c index or into a donor arriving meanwhile are
 *  refused with @c usearch_status_contended_k, while searches of a donor run throughout.
 *
 *  A failure leaves the donor it struck and every later one as live handles the caller still owns
 *  and must free, and @c index holding the donors before it; every consumed slot is nulled, so the
 *  array itself says which handles survived. The handler reports and cancels the closing
 *  refinement alone, exactly as in @c usearch_dense_refine: a cancel skips the rest of the walks
 *  across, still re-links, and answers @c usearch_status_cancelled_k with every donor consumed. A
 *  caller wanting to continue those walks later holds a @c usearch_dense_refinement_t instead.
 *
 *  @code{.c}
 *  usearch_dense_index_t shards[4] = {built_on_socket_0, built_on_socket_1, built_on_socket_2, built_on_socket_3};
 *  usearch_outcome_t outcome = {0};
 *  usearch_dense_absorb(&(usearch_dense_absorb_t){
 *      .index = index, .executor = executor, .outcome = &outcome, .count = 4, .donors = shards,
 *      .progress = {report_to_terminal, &ui, 4}});
 *  // Every entry of `shards` is consumed here; none of them may be freed afterwards.
 *  @endcode
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k @c index is an NVMe view.
 *  - @c usearch_status_invalid_argument_k A @c NULL slot in @c donors, or @c NULL @c donors
 *    for a nonzero @c count.
 *  - @c usearch_status_config_mismatch_k A donor of another residency, scalar, class, metric or
 *    geometry, a donor on a device the executor does not drive, or @c index itself.
 *  - @c usearch_status_contended_k An insertion in flight on @c index or on the next donor, or
 *    on VRAM a save of either; that donor and every later one stay live handles.
 */
USEARCH_EXPORT void usearch_dense_absorb(usearch_dense_absorb_t* arguments);

/** The arguments of @c usearch_dense_compact. */
typedef struct usearch_dense_compact_t {

    /** The handle the sources compact into, possibly one of them. */
    usearch_dense_index_t target;

    /** The executor every move and the closing refinement fan out over. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Source handles to compact. */
    size_t count;

    /** Array of @c count distinct handles; each but @c target is freed and nulled once every
     *  source moved. */
    usearch_dense_index_t* sources;

    /** The @b optional most entries one target shard holds; zero, or more than the target's own
     *  bound, means that bound. */
    size_t shard_capacity;

    /** The @b optional breadth of the candidates of the closing refine's walks; zero derives. */
    size_t expansion_factor;

    /** The @b optional output for what the call did to the graph, as @c usearch_refine_report_t
     *  counts it; @c NULL skips reporting. */
    usearch_refine_report_t* report;

    /** The @b optional handler, called with nodes the closing refine re-selected; a cancel stops it
     *  as it stops @c usearch_dense_refine, never a move. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_dense_compact_t;

/**
 *  @brief Compacts every shard of the handles in @c sources into @c target under @c shard_capacity
 *      entries per shard, largest first, each into the first shard with room, so it ends with as
 *      few shards as fit; in place when @c target is among @c sources.
 *
 *  The shard a bin keeps stays resident with its IDs, every other source shard follows it with
 *  its edges rebased and its keys beside it, and only the appended entries walk the target across
 *  the parts and link in, then each target shard re-links the islands that leaves, as
 *  @c usearch_dense_refine does. In place, a moved entry's key moves with one atomic store, so
 *  lookups and searches of @c target run throughout and emptied shards stay as idle slots for its
 *  next generation. Once the call moved every source, each one but @c target is freed and nulled,
 *  as @c usearch_dense_absorb frees its donors; a failure frees none, those it consumed left empty
 *  and refusing insertions, so every handle in @c sources stays the caller's to free.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k @c target is an NVMe view.
 *  - @c usearch_status_invalid_argument_k A @c NULL slot in @c sources, or @c NULL @c sources
 *    for a nonzero @c count.
 *  - @c usearch_status_config_mismatch_k A repeated source, or one of another residency, scalar,
 *    class, metric or geometry.
 *  - @c usearch_status_contended_k An insertion in flight on @c target or a source, or another
 *    call holding either; nothing moves.
 *  - @c usearch_status_unsupported_k A source shard past @c shard_capacity, which would have to
 *    be cut first, or a target outside RAM.
 */
USEARCH_EXPORT void usearch_dense_compact(usearch_dense_compact_t* arguments);

/** The arguments of @c usearch_dense_split. */
typedef struct usearch_dense_split_t {

    /** The handle split into the receivers, left empty and live. */
    usearch_dense_index_t index;

    /** The executor the partition, each part's move and the closing refinements fan out over. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Receiver handles to split into, at least two. */
    size_t count;

    /** Array of @c count empty handles of the source's residency, scalar, class, metric, geometry
     *  and shard count, each receiving one part. */
    usearch_dense_index_t const* receivers;

    /** The @b optional output for what the call did to the graph, as @c usearch_refine_report_t
     *  counts it; @c NULL skips reporting. */
    usearch_refine_report_t* report;

    /** The @b optional breadth of the candidates of the closing refines' walks; zero derives. */
    size_t expansion_factor;

    /** Picks the part roots: equal seeds over an unchanged index split equally. */
    uint64_t seed;

    /** The @b optional handler, called with nodes the closing refines re-selected; a cancel stops
     *  them as it stops @c usearch_dense_refine, never a move. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_dense_split_t;

/**
 *  @brief Splits @c index into the handles in @c receivers, the inverse of @c usearch_dense_absorb:
 *      graph-local parts keeping their edges rather than re-inserting their vectors, one part per
 *      receiver, and @c index left empty but live.
 *
 *  Every receiver is checked and held before the first part moves, so a refusal moves nothing. The
 *  part roots are drawn from @c seed without a scan, and each part grows breadth-first over the
 *  links from its root, capped at an equal share of the entries, while entries no root reaches
 *  enter the smallest part. A sharded handle splits shard by shard, part @c p of shard @c s moving
 *  into shard @c s of receiver @c p. Each part then enters its receiver all-or-nothing, its edges
 *  rebased, its edges into other parts dropped, and the entries that lost one placed first;
 *  @c index clears once every part moved. Each receiver closes by re-selecting the entries that
 *  lost an edge and re-linking the islands that leaves, as @c usearch_dense_refine does. Meanwhile,
 *  insertions into @c index or a receiver are refused with @c usearch_status_contended_k, while
 *  searches of either run throughout.
 *
 *  The handler reports and cancels the closing refinements alone, exactly as in
 *  @c usearch_dense_refine: a cancel skips the rest of the re-selection, still re-links, and
 *  answers @c usearch_status_cancelled_k with every part moved. A caller wanting to continue that
 *  re-selection later holds a @c usearch_dense_refinement_t opened with @c receivers instead.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k @c index is an NVMe view.
 *  - @c usearch_status_invalid_argument_k Fewer than two receivers, @c NULL @c receivers, or a
 *    @c NULL slot in them.
 *  - @c usearch_status_config_mismatch_k A receiver of another residency, scalar, class, metric,
 *    geometry or shard count, one holding entries, one on a device the executor does not drive, one
 *    named twice, or @c index itself.
 *  - @c usearch_status_contended_k An insertion in flight on @c index or on a receiver, or on
 *    VRAM a save of either; nothing moves.
 *  - @c usearch_status_overflow_k A part passes a receiver's capacity or its node IDs.
 */
USEARCH_EXPORT void usearch_dense_split(usearch_dense_split_t* arguments);

/** The arguments of @c usearch_dense_refinement_init. */
typedef struct usearch_dense_refinement_init_t {

    /** The handle to be refined. */
    usearch_dense_index_t index;

    /** The executor every call on the refinement runs on; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The @b optional donors to absorb first, each consumed and nulled once its entries land; the
     *  array must outlive the refinement. */
    usearch_dense_index_t* donors;

    /** Donor handles in @c donors. */
    size_t count;

    /** The @b optional receivers to split @c index into first, as @c usearch_dense_split fills
     *  them, each refined in turn; the array must outlive the refinement. */
    usearch_dense_index_t const* receivers;

    /** Receiver handles in @c receivers, zero or at least two. */
    size_t receivers_count;

    /** Output for the new handle; @c NULL on failure. The index admits no insertion or save until
     *  @c usearch_dense_refinement_finish frees it. */
    usearch_dense_refinement_t* refinement;

    /** The @b optional breadth of the candidates the re-selection walk explores; zero derives. */
    size_t expansion_factor;

    /** Picks the part roots when @c receivers are named, as in @c usearch_dense_split. */
    uint64_t seed;
} usearch_dense_refinement_init_t;

/**
 *  @brief Opens a refinement of @c index, absorbing @c donors first when any are named, as
 *      @c usearch_dense_absorb moves them, or splitting it into @c receivers, as
 *      @c usearch_dense_split fills them. Every allocation the refinement needs happens here or in
 *      a move, never in a step.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k @c index is an NVMe view.
 *  - @c usearch_status_invalid_argument_k A @c NULL slot in @c donors or @c receivers, @c NULL
 *    @c donors for a nonzero @c count, one receiver alone, or donors and receivers both named.
 *  - @c usearch_status_config_mismatch_k A donor or receiver of another residency, scalar,
 *    class, metric or geometry, a receiver holding entries, or @c index itself.
 *  - @c usearch_status_contended_k An insertion, save or another refinement in flight on
 *    @c index, or an insertion on a receiver.
 *  - @c usearch_status_unsupported_k @c receivers on an Apple GPU handle, whose index does
 *    not split anything yet.
 */
USEARCH_EXPORT void usearch_dense_refinement_init(usearch_dense_refinement_init_t* arguments);

/** The arguments of @c usearch_dense_refinement_next. */
typedef struct usearch_dense_refinement_next_t {

    /** The handle the refinement was opened on. */
    usearch_dense_index_t index;

    /** The executor the refinement was opened with. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The refinement to continue. */
    usearch_dense_refinement_t refinement;

    /** The @b optional handler, called with nodes re-selected across every call so far. */
    usearch_progress_t progress;

    /** The @b optional outputs for the nodes re-selected so far and, of those, the ones whose
     *  neighbors changed; their ratio falling is what says another call buys little. */
    size_t* reselected;
    size_t* changed;

    /** The @b optional traversal telemetry of every call on the refinement so far; @c NULL skips
     *  reporting. */
    usearch_traversal_stats_t* stats;
} usearch_dense_refinement_next_t;

/**
 *  @brief Continues the refinement until it ends or @c progress cancels, moving any donors still
 *      pending first. Answers @c usearch_status_cancelled_k when a later call has more to do; that
 *      call continues exactly where this one stopped. The executor is held only for the call.
 *
 *  Every call reaching the refinement writes the running totals, a cancelled or failed one too, so
 *  they only grow from one call to the next and @c changed never exceeds @c reselected; all stay
 *  zero while donors are pending.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c index is not the handle the refinement was opened
 *    on.
 *  - @c usearch_status_contended_k An insertion in flight on the next donor; that donor and
 *    every later one stay live handles, and a later call retries them.
 */
USEARCH_EXPORT void usearch_dense_refinement_next(usearch_dense_refinement_next_t* arguments);

/** The arguments of @c usearch_dense_refinement_finish. */
typedef struct usearch_dense_refinement_finish_t {

    /** The handle the refinement was opened on. */
    usearch_dense_index_t index;

    /** The executor the refinement was opened with. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The refinement to finish; freed by the call, whatever it answers. */
    usearch_dense_refinement_t refinement;

    /** The @b optional output for what the call did to the graph, as @c usearch_refine_report_t
     *  counts it; @c NULL skips reporting. */
    usearch_refine_report_t* report;

    /** The @b optional traversal telemetry of every call on the refinement; @c NULL skips
     *  reporting. */
    usearch_traversal_stats_t* stats;
} usearch_dense_refinement_finish_t;

/**
 *  @brief Skips whatever re-selection a stopped refinement left, re-links the islands it leaves as
 *      @c usearch_dense_refine does, reports, reopens the index, and frees the handle. A
 *      refinement whose donors have not all moved re-links nothing and reports zeros.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c index is not the handle the refinement was opened
 *    on; the handle is still freed.
 */
USEARCH_EXPORT void usearch_dense_refinement_finish(usearch_dense_refinement_finish_t* arguments);

/** The arguments of @c usearch_dense_cursor_init. */
typedef struct usearch_dense_cursor_init_t {

    /** The handle to be scanned. */
    usearch_dense_index_t index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The @b optional count of @c starts; zero walks every entry as stored. */
    size_t starts_count;

    /** The keys to walk from, in the order the walk begins with them. */
    usearch_key_t const* starts;

    /** Output for the new cursor; @c NULL on failure. */
    usearch_dense_cursor_t* cursor;
} usearch_dense_cursor_init_t;

/**
 *  @brief Opens a walk over the current contents; see @c usearch_dense_cursor_t.
 *
 *  Without @c starts the cursor yields every entry as stored. With them it yields the start keys'
 *  own entries first, in the order given, then the entries those link to, breadth-first, each entry
 *  once, so a caller wanting only the starts' immediate links reads the links of the first pulled
 *  entries. A key the index lacks starts nothing, and a key several entries share starts from its
 *  first stored entry.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c NULL @c starts for a nonzero @c starts_count.
 *  - @c usearch_status_unsupported_k The handle lives on a GPU, which walks no links yet.
 */
USEARCH_EXPORT void usearch_dense_cursor_init(usearch_dense_cursor_init_t* arguments);

/** The arguments of @c usearch_dense_cursor_next. */
typedef struct usearch_dense_cursor_next_t {

    /** The handle the cursor was opened on. */
    usearch_dense_index_t index;

    /** The executor the pull runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The cursor to advance. */
    usearch_dense_cursor_t cursor;

    /** Output for the identities, inside @c arena. */
    usearch_keys_view_t* keys;

    /** The @b optional output for each entry's links, the keys of the entries it links to, entry
     *  @c i spanning @c offsets[i] up to @c offsets[i+1], inside @c arena; @c NULL skips them. */
    usearch_keys_tape_view_t* links;

    /** Output for the entries written; zero when the walk is exhausted. */
    size_t* written;

    /** Memory for @c keys and @c links, rewound by the call. */
    usearch_arena_t arena;

    /** The upper bound on entries this pull writes. */
    size_t wanted;
} usearch_dense_cursor_next_t;

/**
 *  @brief Copies the next batch of entries out of the cursor into @c arena, each with the keys it
 *      links to, in the order @c usearch_dense_cursor_init describes; the vectors themselves come
 *      from @c usearch_dense_get.
 *
 *  Links name only entries published when the pull ran, as the graph stored them then. The keys are
 *  copied rather than borrowed from the index, so the batch stays valid until the next call given
 *  the same arena, however the cursor moves meanwhile.
 *
 *  @par Statuses:
 *  - @c usearch_status_io_k A read from an NVMe view's drives failed.
 *  - @c usearch_status_invalid_argument_k @c index is not the handle the cursor was opened on.
 */
USEARCH_EXPORT void usearch_dense_cursor_next(usearch_dense_cursor_next_t* arguments);

/** Releases the cursor; @c NULL is safe.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_dense_cursor_free(usearch_dense_cursor_t cursor);

/** The arguments of @c usearch_dense_sample. */
typedef struct usearch_dense_sample_t {

    /* Call */
    /** The handle to be sampled. */
    usearch_dense_index_t index;

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

    /** Output for which of its key's entries each draw is, in the order @c usearch_dense_get
     *  writes them, inside @c arena. */
    size_t const** entry_ordinals;

    /** Output for the one shown or earlier draw each draw hangs from and its distance, one column,
     *  a draw with none counting zero, inside @c arena. */
    usearch_matches_view_t* parents;

    /** Output for where each draw is placed, @c written rows of three @c usearch_scalar_f32_k
     *  values, inside @c arena. */
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
} usearch_dense_sample_t;

/**
 *  @brief Draws keys around @c focus, or across the whole index, each hanging from a shown or
 *      earlier draw and placed in three dimensions beside the shown ones - so a view grows one
 *      connected sample as it zooms, never scanning the index.
 *
 *  The draws follow the graph: along paths joining the focus to each other and toward the nearest
 *  shown key, then around each focus key as far as its nearest neighbor on those paths, ranked by
 *  @c seed so a closer view draws a superset of a farther one. A key holding several entries draws
 *  each as its own, named by its entry ordinal. Each draw is placed by its distances to its nearest
 *  shown keys, the shown keys keeping their coordinates; with nothing shown, its row is projected
 *  through a seeded Gaussian matrix.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c NULL @c shown or @c focus for a nonzero count, or
 *    @c shown_coordinates not three @c usearch_scalar_f32_k columns.
 *  - @c usearch_status_unsupported_k The handle lives on a GPU, which does not sample yet.
 *  - @c usearch_status_io_k A read from an NVMe view's drives failed.
 */
USEARCH_EXPORT void usearch_dense_sample(usearch_dense_sample_t* arguments);

/** The arguments of @c usearch_dense_search_stream_init. */
typedef struct usearch_dense_search_stream_init_t {

    /** Output for the new stream; @c NULL on failure. */
    usearch_dense_search_stream_t* stream;

    /** The handle to be queried. */
    usearch_dense_index_t index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The single query row. */
    usearch_rows_view_t query;

    /** The @b optional breadth of the candidates the walk explores, as in
     *  @c usearch_dense_search_t; zero derives. */
    size_t expansion_factor;
} usearch_dense_search_stream_init_t;

/**
 *  @brief Opens a resumable search for one query; see @c usearch_dense_search_stream_t.
 *
 *  @par Statuses:
 *  - @c usearch_status_unsupported_k No residency implements the stream yet.
 */
USEARCH_EXPORT void usearch_dense_search_stream_init(usearch_dense_search_stream_init_t* arguments);

/** The arguments of @c usearch_dense_search_stream_next. */
typedef struct usearch_dense_search_stream_next_t {

    /** The handle the stream was opened on. */
    usearch_dense_index_t index;

    /** The executor the pull runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The stream to advance. */
    usearch_dense_search_stream_t stream;

    /** Output for the pull, as one row of a search inside @c arena: up to @c wanted more matches,
     *  their number in @c matches->counts[0], zero once the walk is drained. */
    usearch_matches_view_t* matches;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /** The upper bound on matches this pull writes. */
    size_t wanted;
} usearch_dense_search_stream_next_t;

/**
 *  @brief Pulls the next-nearest matches from the stream.
 *
 *  Ordering beyond the expansion horizon is best-effort nearest rather than globally exact, so a
 *  caller pulling far past the breadth it configured sees the ranking degrade rather than the
 *  stream end.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c index is not the handle the stream was opened on.
 */
USEARCH_EXPORT void usearch_dense_search_stream_next(usearch_dense_search_stream_next_t* arguments);

/** Releases the stream and the walk state it keeps; @c NULL is safe.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_dense_search_stream_free(usearch_dense_search_stream_t stream);

/** The arguments of @c usearch_dense_save. */
typedef struct usearch_dense_save_t {

    /** The handle to be written. */
    usearch_dense_index_t index;

    /** The executor the write fans out over, one shard per thread or stream. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Destination path: the monolith of a lone shard, or the manifest beside the parts of many. */
    usearch_bytes_view_t path;

    /** The @b optional handler, called with shards written. */
    usearch_progress_t progress;
} usearch_dense_save_t;

/**
 *  @brief Saves a mutable index under @c path in the layout its shard count picks - one monolith,
 *      or a manifest beside its shards; concurrent readers keep searching throughout.
 *      @ref dense_index_serialization
 *
 *  A cancelled or failed monolith save leaves a file without its header, which every reader
 *  rejects; a cancelled manifest save leaves the parts it already wrote and no manifest, so the
 *  checkpoint reads as absent rather than half-valid. Those leftovers are the caller's to remove.
 *
 *  @par Statuses:
 *  - @c usearch_status_contended_k A RAM save meeting an insertion in flight or one publishing
 *    under it, or a VRAM save meeting an insertion, refinement, absorption or another save in
 *    flight; an insertion arriving while a VRAM save runs is refused the same way.
 */
USEARCH_EXPORT void usearch_dense_save(usearch_dense_save_t* arguments);

/** The arguments of @c usearch_dense_save_buffer. */
typedef struct usearch_dense_save_buffer_t {

    /** The handle to be written. */
    usearch_dense_index_t index;

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

    /** The @b optional handler, called with bytes written. */
    usearch_progress_t progress;
} usearch_dense_save_buffer_t;

/**
 *  @brief Saves a mutable index into @c arena: one monolith for a handle of one shard, or a
 *      manifest leading one part per shard. @ref dense_index_serialization
 *
 *  A lone shard's monolith is the same bytes the path-taking sibling writes into its own mapping.
 *  Several shards write the manifest first, listing each part by its offset inside the buffer,
 *  then every part on the header's alignment, each with its shard's own node IDs, so a load opens
 *  one shard per part again. A caller who needs the bytes in memory it already owns, such as a
 *  mapped file, saves to a path instead. A VRAM handle's checkpoint is written in device memory,
 *  written on the host first, so the call holds twice its bytes while it runs.
 *
 *  Writing to memory is not writing quickly. A terabyte checkpoint on a handful of cores runs for
 *  minutes, so this reports and cancels like its path-taking sibling, and a cancelled call leaves
 *  @c monolith->data @c NULL.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A @c NULL @c arena.
 *  - @c usearch_status_device_k An arena or an executor on another residency or device.
 *  - @c usearch_status_contended_k The arena serves another call, another call holds the
 *    executor or its pool, a dispensed batch has yet to publish, or a call, batch, load or plan
 *    holds the handle.
 *  - @c usearch_status_unsupported_k The handle is an NVMe view, which already is its checkpoint.
 *  - @c usearch_status_uninitialized_k No shard is placed, or the executor holds no pool workers.
 *  - @c usearch_status_unknown_kind_k The stored scalar has no interchange dtype.
 *  - @c usearch_status_config_mismatch_k The executor drives no stream on some VRAM shard's
 *    device.
 */
USEARCH_EXPORT void usearch_dense_save_buffer(usearch_dense_save_buffer_t* arguments);

/**
 *  @brief The geometry and metric a checkpoint froze - everything a reader learns without opening
 *      it, and nothing a reader must choose for itself.
 *
 *  Placement is absent because a checkpoint records where it was written, never where a reader
 *  must keep it. So is @c values_per_key, which bounds a live handle's mutable state rather than
 *  anything the serialized form carries. Both remain the caller's to choose when loading, and no
 *  caller has to know which fields this call filled.
 */
typedef struct usearch_dense_metadata_t {

    /** The metric kind the checkpoint scores with. */
    usearch_dense_metric_t metric;

    /** The scalar type its rows are stored & quantized into. */
    usearch_scalar_t quantization;

    /** Scalars per stored vector. */
    size_t dimensions;

    /** Connections per node the graph was built with. */
    size_t connectivity;
} usearch_dense_metadata_t;

/** The arguments of @c usearch_dense_metadata_read. */
typedef struct usearch_dense_metadata_read_t {

    /** The executor the read runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The manifest of a sharded checkpoint, or a single @c .safegraphs monolith. */
    usearch_bytes_view_t path;

    /** Output for every field the checkpoint froze. */
    usearch_dense_metadata_t* metadata;
} usearch_dense_metadata_read_t;

/**
 *  @brief Reads a checkpoint's geometry & metric without loading it.
 *
 *  A manifest answers from its summary and its first part's header, so the cost does not grow with
 *  the index.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c path, or a path that does not parse.
 *  - @c usearch_status_malformed_k The file is empty, or its manifest or header does not parse.
 */
USEARCH_EXPORT void usearch_dense_metadata_read(usearch_dense_metadata_read_t* arguments);

/** The arguments of @c usearch_dense_metadata_read_buffer. */
typedef struct usearch_dense_metadata_read_buffer_t {

    /** The executor the read runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Source holding one serialized monolith, or the manifest of several parts leading them. */
    usearch_bytes_view_t buffer;

    /** Output for every field the checkpoint froze. */
    usearch_dense_metadata_t* metadata;
} usearch_dense_metadata_read_buffer_t;

/**
 *  Reads geometry & metric from a serialized checkpoint buffer.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c buffer.
 *  - @c usearch_status_malformed_k The manifest or the header does not parse.
 */
USEARCH_EXPORT void usearch_dense_metadata_read_buffer(usearch_dense_metadata_read_buffer_t* arguments);

/** A live index's state. Each counter is read once, so under concurrent insertions they may come
 *  from slightly different moments. */
typedef struct usearch_dense_index_description_t {

    /** The frozen metric, quantization, dimensions and connectivity, as a checkpoint reports them. */
    usearch_dense_metadata_t metadata;

    /** Stored entries. */
    size_t size;

    /** Entries hosted without growing again. */
    size_t capacity;

    /** Bytes the handle holds; traversal scratch is the executor's. */
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
} usearch_dense_index_description_t;

/** The arguments of @c usearch_dense_describe. */
typedef struct usearch_dense_describe_t {

    /** The handle to be described. */
    usearch_dense_index_t index;

    /** The executor the report runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output, filled whole. */
    usearch_dense_index_description_t* description;
} usearch_dense_describe_t;

/** Reports a live index's frozen metadata and current counters in one call. */
USEARCH_EXPORT void usearch_dense_describe(usearch_dense_describe_t* arguments);

/** The arguments of @c usearch_dense_multiply. */
typedef struct usearch_dense_multiply_t {

    /* Call */
    /** The handle supplying the right-hand rows. */
    usearch_dense_index_t index;

    /** The executor driving the batch; its kind must match the handle's residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /* Inputs */
    /** Left-hand rows - the batch axis. */
    size_t count;

    /** The left-hand rows, multiplied against every indexed vector the walk reaches. */
    usearch_rows_view_t left;

    /* Outputs */
    /** Output for the pairs inside @c arena: each one's indexed key and product, query @c i
     *  spanning @c products->offsets[i] up to @c products->offsets[i+1]. */
    usearch_matches_tape_view_t* products;

    /** The @b optional output for the pairs written across every query. */
    size_t* written;

    /** The @b optional output for the pairs scored across every query, before @c wanted cut any. */
    size_t* required;

    /** Memory for @c products, rewound by the call. */
    usearch_arena_t arena;

    /* Knobs */
    /** The @b optional bound on pairs kept per query, the first ones its walk scored; zero keeps
     *  every pair. */
    size_t wanted;

    /** The @b optional breadth of the candidates, which widens the emitted pattern; zero
     *  derives. */
    size_t expansion_factor;

    /** How entries are reached: a walk, the default; no residency multiplies by an exact scan
     *  yet. */
    usearch_traversal_t traversal;

    /** The @b optional predicate, asked about a scored quad of keys at a time; a pair it
     *  rejects is neither written nor counted, while the walk still routes through it. */
    usearch_predicate_t predicate;

    /* Observers */

    /** The @b optional handler, called with queries resolved. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_dense_multiply_t;

/**
 *  @brief Approximate sparse product of a run of rows against the indexed vectors - the linear
 *      algebra sibling of @c usearch_dense_search, reporting @b every pair the traversal scored
 *      instead of only the nearest. @ref dense_index_linear_algebra
 *
 *  A walk scores many times the pairs a search reports, and each of them is an exact product the
 *  caller would otherwise recompute - this verb hands them all back. Pairs arrive in traversal
 *  order rather than sorted, and carry no duplicates: the traversal's visited set admits each
 *  indexed vector at most once per left-hand row. A wider @c expansion_factor widens the emitted
 *  pattern, that breadth being what decides how much the walk touches.
 *
 *  How many pairs a walk scores is known only once it finishes, so the call writes the keys, the
 *  scores and their Apache Arrow offsets in @c arena as one ragged output, in one call. A non-zero
 *  @c wanted keeps the first pairs each left-hand row's walks scored, shard by shard in placement
 *  order, and @c required says how many there were.
 *
 *  Cost is not O(m · n · k) but roughly O(count × expansion × connectivity × dimensions), sublinear
 *  in the indexed vectors, which is the whole point of reading a product off a graph. The result is
 *  approximate, and the pairs the walk never reaches are the low and negative dots.
 *
 *  Runs on the inner-product kinds, whose traversal scores are the products themselves. A graph
 *  frozen under a distance kind scored a different quantity, so the products it would report are
 *  not the ones asked for, and the call refuses with @c usearch_status_config_mismatch_k.
 *
 *  @see BLAS, whose @c ?GEMM is the dense counterpart this is shaped after: https://netlib.org/blas
 *
 *  @code{.c}
 *  // BLAS: B is an operand, C is dense m × n, and every pair is computed.
 *  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k,
 *              1.0f, a, lda, b, ldb, 0.0f, c, ldc);
 *
 *  // Here: B was built once and is the index, C is ragged and holds only the pairs the walk
 *  // scored, and the executor carries what a cuBLAS handle plus its stream would.
 *  usearch_arena_t arena = NULL;
 *  usearch_matches_tape_view_t c;
 *  usearch_outcome_t outcome = {0};
 *  usearch_arena_init(&(usearch_arena_init_t){.arena = &arena, .executor = executor, .outcome = &outcome});
 *  usearch_dense_multiply(&(usearch_dense_multiply_t){
 *      .index = index, .executor = executor, .outcome = &outcome,
 *      .count = m, .left = {a, a_stride, k, usearch_scalar_f32_k}, .products = &c, .arena = arena});
 *  for (size_t row = 0; row != m; ++row)
 *      for (size_t pair = c.offsets[row]; pair != c.offsets[row + 1]; ++pair)
 *          accumulate(row, c.keys[pair], c.scores[pair]);
 *
 *  // The next product reuses the arena, and frees nothing until the last one.
 *  usearch_arena_free(arena);
 *  @endcode
 *
 *  What the comparison leaves out is the whole of BLAS' tuning surface: no transpose flags, no
 *  @c alpha and @c beta, no leading-dimension triple, and nothing accumulating into @c C.
 *
 *  @par Statuses:
 *  - @c usearch_status_unknown_kind_k A @c traversal other than a walk or a scan.
 *  - @c usearch_status_unsupported_k A scan, or a @c predicate on a VRAM handle.
 */
USEARCH_EXPORT void usearch_dense_multiply(usearch_dense_multiply_t* arguments);

/**
 *  @brief The online-softmax carry for one query row - Flash Attention's running maximum and
 *      partition, which is what lets one call's output fold into another's.
 *
 *  The two are one concept: a maximum without its partition is a number nothing can rescale, so
 *  they share an element rather than travelling as two arrays a caller could supply half of.
 */
typedef struct usearch_dense_softmax_state_t {

    /** Largest logit seen so far - the recurrence's @c m. */
    float maximum_score;

    /** Sum of exponentials taken against that maximum - the recurrence's @c l. */
    float partition;
} usearch_dense_softmax_state_t;

/** Resolves @p count keys to their value rows, one row per key, @p columns values of @p scalar each;
 *  a NULL row refuses the call. A host function for a RAM handle; for a VRAM one, a @c __device__
 *  function whose address comes from @c cudaMemcpyFromSymbol on NVIDIA or @c hipMemcpyFromSymbol on
 *  AMD, its @p state and rows 16-byte aligned in memory that device reads.
 *  No driver query tells the two kinds apart, so passing the wrong one is undefined, never refused. */
typedef void (*usearch_value_lookup_t)(void* state, usearch_key_t const* keys, size_t count, usearch_scalar_t scalar,
                                       size_t columns, void const** rows);

/** The arguments of @c usearch_dense_attend. */
typedef struct usearch_dense_attend_t {

    /* Call */
    /** The handle holding one layer's keys for one key-value head. */
    usearch_dense_index_t index;

    /** The executor driving the batch; its kind must match the handle's residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /* Inputs */
    /** Query rows - the batch axis; row t · @c head_count + h is head h of token t. */
    size_t count;

    /** The query rows. */
    usearch_rows_view_t queries;

    /** The @b optional query heads sharing the index's key-value head, and with it one walk per token;
     *  it must divide @c count, and zero means one. */
    size_t head_count;

    /** Resolves the keys of every scored batch to their value rows, on the host or the device. */
    usearch_value_lookup_t value_lookup;

    /** Opaque pointer forwarded into every @c value_lookup call. */
    void* value_lookup_state;

    /** The @b optional online-softmax state, one per query row, lent by the caller; @c NULL skips it. */
    usearch_dense_softmax_state_t* carry;

    /* Outputs */
    /** Output for the attended rows inside @c arena, @c count of them, as wide as the keys. */
    usearch_rows_view_t* outputs;

    /** Memory for @c outputs, rewound by the call. */
    usearch_arena_t arena;

    /* Knobs */
    /** The @b optional scalar type the attended rows land as; zero takes the queries' own. */
    usearch_scalar_t outputs_scalar;

    /** The @b optional breadth of the candidates; zero derives. Wider candidates walk further,
     *  folding more of the tail. */
    size_t expansion_factor;

    /** The @b optional logit scale, conventionally the inverse square root of the head
     *  dimensionality; zero derives it from the index's frozen geometry. */
    float score_scale;

    /** How entries are reached: a walk, the default; no residency attends by an exact scan yet. */
    usearch_traversal_t traversal;

    /** The @b optional predicate, asked about a scored quad of keys at a time; a key it rejects
     *  folds into no head and resolves no value row, while the walk still routes through it. */
    usearch_predicate_t predicate;

    /* Observers */

    /** The @b optional handler, called with queries resolved. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_dense_attend_t;

/**
 *  @brief Softmax attention of a run of query heads over every indexed key their graph walks score,
 *      folded as Flash Attention folds key tiles. @ref dense_index_linear_algebra
 *
 *  One walk per token runs by a search's own frontier and stopping rule, ranking a key by the best
 *  of the token's heads, so every head's own region is explored. Every key the walk scores folds into
 *  each head's running maximum, partition and weighted value sum, rescaled whenever the maximum
 *  rises - Flash Attention's online softmax, with the graph replacing the sweep over key tiles. No
 *  top-k cut is taken: the long tail the walk scores is what brings the result toward exact dense
 *  attention, and @c outputs comes back divided by the partition.
 *
 *  Rows of @c queries, @c outputs and @c carry are tasks: row t · @c head_count + h is head h of
 *  token t. One index holds one layer's one key-value head, so the @c head_count query heads of
 *  grouped-query attention share its keys and one walk per token; one head per token is plain
 *  multi-head attention.
 *
 *  The index stores no values, so only keys a walk scores are ever resolved to value rows. A RAM
 *  handle calls @c value_lookup on the host once per scored batch; a GPU handle calls it on the
 *  device inside the walk. A value row is exactly as wide as the keys and stored in the index's own
 *  scalar kind, which every call states as @c columns and @c scalar. A NULL row refuses the call with
 *  @c usearch_status_shape_mismatch_k rather than skipping the key: a missing row is a wrong softmax,
 *  not a smaller one.
 *
 *  Keep the recent window out of the index, inserting a token once it leaves the window, so the dense
 *  window and the graph never share a key. @c carry holds the softmax's @c m and @c l, which is
 *  what merges this branch with another over the same queries - the recent window, attention sinks,
 *  or a second index of one context. The two travel in one element because a maximum without its
 *  partition is a number nothing can rescale. Two branches over the same queries merge as:
 *
 *  m = max(m₁, m₂),  aᵢ = lᵢ · e^(mᵢ − m),  l = a₁ + a₂,  o = (a₁ · o₁ + a₂ · o₂) / l
 *
 *  @see Flash Attention, whose recurrence this reuses: https://arxiv.org/abs/2205.14135
 *
 *  Runs on the inner-product kinds, since a softmax needs logits and only those kinds score one.
 *  A graph frozen under a distance kind refuses with @c usearch_status_config_mismatch_k.
 *
 *  Never writes to the graph, so any number of these may overlap with concurrent insertions -
 *  which is what streaming a growing context through this verb amounts to, the publication
 *  watermark acting as the causal mask. @ref dense_index_concurrency
 *
 *  @code{.c}
 *  // One layer, one key-value head shared by 8 query heads: row t · 8 + h is head h of token t.
 *  usearch_rows_view_t attended;
 *  usearch_outcome_t outcome = {0};
 *  usearch_dense_attend(&(usearch_dense_attend_t){
 *      .index = index, .executor = executor, .outcome = &outcome, .count = tokens * 8,
 *      .queries = {query_rows, dimensions * 2, dimensions, usearch_scalar_bf16_k}, .head_count = 8,
 *      .value_lookup = value_rows_of, .value_lookup_state = value_cache,
 *      .outputs = &attended, .carry = carry, .arena = arena});
 *
 *  // The recent window never entered the index, so it merges in by the same rule, counting no key twice.
 *  merge_recent_window(attended.data, attended.stride, carry, tokens * 8, dimensions);
 *  @endcode
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c value_lookup is NULL.
 *  - @c usearch_status_shape_mismatch_k @c head_count does not divide @c count, or a lookup yields
 *    a NULL row.
 *  - @c usearch_status_unknown_kind_k A @c traversal other than a walk or a scan.
 *  - @c usearch_status_unsupported_k A scan, or a @c predicate on a VRAM handle.
 */
USEARCH_EXPORT void usearch_dense_attend(usearch_dense_attend_t* arguments);

/** The arguments of @c usearch_dense_maxsim. */
typedef struct usearch_dense_maxsim_t {

    /* Call */
    /** The handle holding one token row per entry, several sharing a key. */
    usearch_dense_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /* Inputs */
    /** Queries - the batch axis; @c matches spans @c count of them. */
    size_t count;

    /** The token rows of every query laid end to end, @c count times @c rows_per_query of them, or
     *  as many as @c query_offsets spans. */
    usearch_rows_view_t queries;

    /** The @b optional offsets of every query's rows, @c count + 1 of them; @c NULL keeps
     *  @c rows_per_query. */
    size_t const* query_offsets;

    /** Rows in every query; uniform, so ColBERT's padding applies as is; unread when
     *  @c query_offsets is set. */
    size_t rows_per_query;

    /* Outputs */
    /** Output for each query's best-scoring keys, largest first, as many as it found up to
     *  @c wanted, or every key it scored when @c wanted is zero, packed into @c arena: query @c i
     *  spans @c matches->offsets[i] up to @c matches->offsets[i+1], @c scores carrying each key's
     *  summed maxima. */
    usearch_matches_tape_view_t* matches;

    /** The @b optional output for the keys written across every query, read on the host even
     *  when @c matches lives in device memory. */
    size_t* written;

    /** The @b optional output for the keys scored across every query, before @c wanted cut any. */
    size_t* required;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /* Knobs */
    /** The @b optional bound on keys per query - the "k" over documents, not token rows; zero keeps
     *  every key the query scored. */
    size_t wanted;

    /** The @b optional breadth of the candidates; zero derives. */
    size_t expansion_factor;

    /** How candidates are nominated: a walk, the default, or a scan nominating every key. */
    usearch_traversal_t traversal;

    /** The @b optional predicate over keys, asked before any key's rows are reduced. */
    usearch_predicate_t predicate;

    /** The @b optional keys every row's walk of a query starts from ahead of its random landmarks,
     *  one row of @c counts[query] keys per query, as a search writes its matches and takes them,
     *  resolved once per query; scores are not read, a scan ignores them, and keys no entry holds
     *  are skipped and counted in @c stats once per query. */
    usearch_matches_view_t const* hints;

    /* Observers */

    /** The @b optional handler, called with queries resolved. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_dense_maxsim_t;

/**
 *  @brief Late-interaction scoring of multi-row queries against multi-row entries: every query row
 *      takes its best match within a key, and those maxima sum to that key's score.
 *  @ref dense_index_linear_algebra
 *
 *  This is what @c values_per_key above one exists for. A document is one key holding its token
 *  rows, and the walk stays over individual tokens under a plain inner product - no set-to-set
 *  metric is ever evaluated, since the grouping happens in the reduction rather than in the graph.
 *  That is also why an entry's rows need no ordering and two documents need not share a row count.
 *
 *  Exact for every key it scores: the walks only nominate keys, each query row taking as many
 *  tokens as the expansion factor, and every row of a nominated key then folds into that key's
 *  per-row maxima. A walk may miss a key none of its rows ranked; a scan nominates every key.
 *  Widening the expansion factor narrows what a walk misses.
 *
 *  Runs on the inner-product kinds, since a maximum over logits is what late interaction sums.
 *  A graph frozen under a distance kind refuses with @c usearch_status_config_mismatch_k.
 *
 *  @see ColBERT, which introduced this scoring: https://arxiv.org/abs/2004.12832
 *
 *  @code{.c}
 *  // 4 queries of 32 padded token rows each, scored against documents of any length.
 *  usearch_matches_tape_view_t documents;
 *  usearch_outcome_t outcome = {0};
 *  usearch_dense_maxsim(&(usearch_dense_maxsim_t){
 *      .index = index, .executor = executor, .outcome = &outcome,
 *      .count = 4, .queries = {query_tokens, 128 * sizeof(float), 128, usearch_scalar_f32_k}, .rows_per_query = 32,
 *      .matches = &documents, .arena = arena, .wanted = 10});
 *  @endcode
 *
 *  @par Statuses:
 *  - @c usearch_status_unknown_kind_k A @c traversal other than a walk or a scan.
 *  - @c usearch_status_shape_mismatch_k A query without rows, or @c query_offsets decreasing.
 *  - @c usearch_status_unsupported_k A scan, a @c predicate, @c hints, ragged @c query_offsets or a
 *    zero @c wanted on a VRAM handle; on an NVMe view holding no key table, a walk with @c hints,
 *    or any walk when the view allows no key scans either.
 *  - @c usearch_status_invalid_argument_k @c hints naming no counts, or columns but no keys.
 *  - @c usearch_status_shape_mismatch_k A hinted query counts more keys than the hints' columns.
 */
USEARCH_EXPORT void usearch_dense_maxsim(usearch_dense_maxsim_t* arguments);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_DENSE_INDEX_H
