/**
 *  @file c/usearch/shapes_index.h
 *  @author Ash Vardanian
 *  @date August 15, 2026
 *  @brief The shapes-modality C interface: similarity search over 3D point clouds of any size -
 *      molecules, conformers, protein backbones - under distances that need no correspondence.
 *
 *  One entry is a shape: a cloud of 3D points whose count varies per entry. A small molecule is
 *  a few dozen atom positions, a protein backbone a few thousand. Neither the two sides of a
 *  comparison nor two entries of the index need agree on their point count - a 20-atom molecule
 *  scores against a 60-atom molecule, without padding, truncation, or resampling anywhere in the
 *  interface.
 *
 *  Points are unordered and carry no identity, so no kind pairs point @c i with point @c i of
 *  the other side: each finds its own matching. Every kind is invariant to rotation, reflection
 *  and translation, so a shape scores zero against any rigid motion of itself - and against its
 *  mirror image, which is why enantiomers compare as equals. Sets of embeddings scored by late
 *  interaction belong to the dense modality's @c usearch_dense_maxsim, and chains whose order
 *  carries meaning to the sequence modality.
 *
 *  Handles are created by a lifecycle verb naming both an action and a residency, each spelled
 *  @c usearch_shapes_ and the suffix below:
 *
 *      Action                           RAM          VRAM         NVMe
 *      init: start empty, mutable       init_ram     init_vram    none
 *      load: copy a checkpoint in       load_ram     load_vram    none
 *      view: serve one in place         view_ram     none         view_nvme
 *      save: write a checkpoint         save         save         none
 *
 *  A viewed handle is read-only, refusing every mutating verb with @c usearch_status_read_only_k.
 *  An NVMe handle answers every read verb as a RAM handle does, its rows read through the
 *  executor's engines, and saving answers @c usearch_status_unsupported_k, since a view already is
 *  its checkpoint.
 *
 *  @section shapes_index_tape Batch Shape
 *
 *  Variable point counts cannot ride a run of equal rows, so every batch input is a @b tape:
 *  every entry's points end to end, and @c count + 1 Apache Arrow offsets counting points around
 *  them. Entry @c i holds `(offsets[i+1] - offsets[i]) * 3` scalars, @c x, @c y and @c z
 *  interleaved and tightly packed. Keys need not be unique - three conformers of one molecule
 *  share its key, and the modality reports how many entries a key holds.
 *
 *  @section shapes_index_scoring Scoring Kinds
 *
 *  Each shape is centred on its centroid and turned onto its principal axes once, when it is
 *  stored. Chamfer and Hausdorff then compare points to points under the eight sign flips of those
 *  axes and keep the best flip: each point of one side finds its nearest point on the other, and
 *  the two kinds differ only in how those distances are aggregated. Chamfer averages them, so one
 *  stray point is diluted, and Hausdorff takes the worst, so one stray point decides.
 *
 *  Wasserstein-Procrustes is Kabsch without a known correspondence. An exact transport plan
 *  pairs the points, the rotation or reflection that best superposes the plan follows from it,
 *  and the two alternate from each of the eight flips until the cost stops falling. Every point
 *  is accounted for rather than one point absorbing many partners, which is what lets it rank
 *  conformers markedly better than Chamfer.
 *
 *  Gromov-Wasserstein compares each side's @b internal geometry - the distances between its own
 *  points - and never the coordinates, so it needs no superposition at all. It starts from the
 *  best Wasserstein-Procrustes plans and ranks shapes best of the four, at the highest cost.
 *
 *  All four kinds are symmetric bit for bit, which is what lets them weigh a graph edge.
 *
 *  @section shapes_index_determinism Determinism
 *
 *  The transport kinds iterate, so the modality pins every source of variation a graph index
 *  cannot tolerate: fixed starts from the eight flips, a Wasserstein-Procrustes descent that
 *  stops only when its cost stops falling, a fixed Gromov-Wasserstein step budget rather than a
 *  convergence-tolerance early exit, and a canonical ordering of each unordered pair before it
 *  is solved. Without that last step a solver can settle in two different local optima
 *  depending on argument order, and the graph would disagree with itself about an edge it had
 *  already inserted.
 *
 *  A checkpoint keeps every shape in the pose it was inserted in, and loading turns it onto its
 *  principal axes again, so distances after a reload agree with those before it to the precision
 *  of the stored scalar rather than bit for bit.
 *
 *  @section shapes_index_concurrency Concurrency & Thread Safety
 *
 *  A handle may be shared across threads. On a RAM handle, insertions and searches overlap
 *  freely: searches read only entries published behind the visibility watermark and never
 *  write to the graph, while insertions synchronize through per-node locks and publish in
 *  order. A VRAM handle is driven through the streams of its executor. A viewed handle admits
 *  arbitrarily many concurrent searches, having nothing to mutate. Serialization is the one
 *  exception: saving demands a mutation-free window, though concurrent readers keep searching
 *  throughout. Per-call outputs - counts and the outcome - belong to their call and must not
 *  be shared between calls that are not final.
 *
 *  An insertion is all-or-nothing per call, and on a RAM handle its entries become visible at
 *  once, so a concurrent search sees every entry of the call or none of them.
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
#ifndef UNUM_USEARCH_SHAPES_INDEX_H
#define UNUM_USEARCH_SHAPES_INDEX_H

#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Similarity Search Index over 3D point clouds of varying size.
 *
 *  @see usearch_shapes_init_ram
 *  @see usearch_shapes_insert
 *  @see usearch_shapes_search
 */
typedef struct usearch_shapes_index_handle_t* usearch_shapes_index_t;

/** A consistent snapshot cursor over the graph. Captures the publication watermark at creation,
 *  so concurrent inserts never appear mid-iteration, and yields each entry's key with the keys it
 *  links to - every entry as stored, or breadth-first from start keys - which export, migration,
 *  and visualization feed off. Each pull names the index again; an overlapping pull reports
 *  @c usearch_status_contended_k. */
typedef struct usearch_shapes_cursor_handle_t* usearch_shapes_cursor_t;

/**
 *  @brief One refinement in flight, optionally absorbing donors or splitting into receivers first.
 *
 *  It holds everything the pass needs between calls, so a cancelled
 *  @c usearch_shapes_refinement_next continues where it stopped, and nothing about it stays in the
 *  index once it finishes. The index admits no insertion and no save while the handle lives;
 *  searches proceed as during @c usearch_shapes_refine. Each call names the index and the executor
 *  it was opened with.
 */
typedef struct usearch_shapes_refinement_handle_t* usearch_shapes_refinement_t;

/**
 *  @brief How two shapes compare, with no correspondence between their points.
 *
 *  Writes @c d for the distance between two points, @c S for one of the eight sign flips
 *  @b diag(±1,±1,±1), and @c P for a transport plan: a non-negative @b [|A|,|B|] matrix whose
 *  rows sum to @c 1/|A| and columns to @c 1/|B|. Both shapes are taken as already centred and
 *  turned onto their principal axes, as the index stores them.
 *  @ref shapes_index_scoring
 */
typedef enum usearch_shapes_metric_t {
    usearch_shapes_metric_unknown_k = 0,

    /**
     *  @brief Mean distance to each point's nearest partner, both directions, halved.
     *
     *  Every point is charged the distance to the nearest point of the other shape, and those
     *  charges are averaged, so a single unpartnered point is diluted rather than decisive. The
     *  cheapest kind: eight passes over the @b [|A|,|B|] pairs.
     *
     *  CD(A,B) = min_S ½ ( mean_i min_j d(aᵢ, S·bⱼ) + mean_j min_i d(aᵢ, S·bⱼ) )
     *
     *  @code{.py}
     *  flips = [diag(s) for s in product((1, -1), repeat=3)]
     *  distances = [cdist(A, B @ S) for S in flips] # [|A|,|B|] per flip
     *  chamfer = min((D.min(axis=1).mean() + D.min(axis=0).mean()) / 2 for D in distances)
     *  @endcode
     */
    usearch_shapes_metric_chamfer_k = 1,

    /**
     *  @brief The worst of all nearest-partner distances, both directions.
     *
     *  Unforgiving where Chamfer is lenient: one point whose nearest point on the other shape is
     *  far away sets the entire score. Costs what Chamfer costs, off the same eight passes.
     *
     *  H(A,B) = min_S max( max_i min_j d(aᵢ, S·bⱼ), max_j min_i d(aᵢ, S·bⱼ) )
     *
     *  @code{.py}
     *  hausdorff = min(max(D.min(axis=1).max(), D.min(axis=0).max()) for D in distances)
     *  @endcode
     */
    usearch_shapes_metric_hausdorff_k = 2,

    /**
     *  @brief Transport over each side's internal geometry - the best ranker, and the costliest.
     *
     *  Compares how each shape's own points sit relative to one another and never their
     *  coordinates. It starts from the best proper and the best mirrored Wasserstein-Procrustes
     *  plans and from the uniform plan, then takes a fixed budget of Frank-Wolfe steps, each
     *  solving an exact transport plan over a cubic gradient - a molecule-scale kind, too slow for
     *  thousands of points.
     *
     *  GW(A,B) = min_P ½ √( Σ_{i,k,j,l} (d(aᵢ,aₖ) − d(bⱼ,bₗ))² · P[i,j] · P[k,l] )
     *
     *  @code{.py}
     *  DA, DB = cdist(A, A), cdist(B, B) # [|A|,|A|] and [|B|,|B|]; never A vs B
     *  L = (DA[:, None, :, None] - DB[None, :, None, :]) ** 2 # [|A|,|B|,|A|,|B|]
     *  gromov = min(sqrt(einsum("ijkl,ij,kl->", L, P, P)) / 2 for P in plans)
     *  @endcode
     */
    usearch_shapes_metric_gromov_wasserstein_k = 3,

    /**
     *  @brief Residual after the best rigid motion and the best matching - Wasserstein-Procrustes.
     *
     *  Kabsch without a known correspondence: an exact transport plan pairs the points, and the
     *  rotation or reflection that best superposes the plan comes out of its @b [3,3] covariance.
     *  The two alternate from each of the eight flips until the cost stops falling. On equal
     *  counts it never exceeds the Kabsch RMSD of a known atom order, and equals it where that
     *  order is an optimal matching.
     *
     *  WP(A,B) = min_{R ∈ O(3)} min_P √( Σ_{i,j} P[i,j] · ‖aᵢ − R·bⱼ‖² )
     *
     *  @code{.py}
     *  for S in flips:                                   # each start descends on its own
     *      R = S
     *      while cost falls:
     *          P = transport(cdist(A, B @ R.T) ** 2)     # exact plan between uniform weights
     *          U, _, Vt = svd(A.T @ P @ B)               # [3,3] covariance of the plan
     *          R = U @ diag([1, 1, det(U @ Vt) * det(S)]) @ Vt
     *  @endcode
     */
    usearch_shapes_metric_wasserstein_procrustes_k = 4,
} usearch_shapes_metric_t;

/** The arguments of @c usearch_shapes_init_ram - the geometry and metric an empty index freezes,
 *  and where its shards live. */
typedef struct usearch_shapes_init_ram_t {

    /** Output for the new index. */
    usearch_shapes_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The kind comparing two shapes - @c usearch_shapes_metric_wasserstein_procrustes_k for
     *  conformers and pockets, @c usearch_shapes_metric_chamfer_k where speed comes first. */
    usearch_shapes_metric_t metric;

    /** The scalar type points are stored in: @c f16 or @c f32, any other refused with
     *  @c usearch_status_unsupported_k. */
    usearch_scalar_t quantization;

    /** The frozen point count, when every entry has one - a fixed-length backbone, a landmark
     *  set. Zero is the default and admits any count; a non-zero value rejects insertions of any
     *  other. */
    size_t points;

    /** The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** Upper bound on entries sharing one key; zero derives to one - unique keys. */
    size_t values_per_key;

    /** The memory domains to place one shard on apiece. */
    usearch_ram_options_t ram;
} usearch_shapes_init_ram_t;

/** Creates an empty, mutable shape index in host memory. */
USEARCH_EXPORT void usearch_shapes_init_ram(usearch_shapes_init_ram_t* arguments);

/** The arguments of @c usearch_shapes_init_vram, which match @c usearch_shapes_init_ram_t but
 *  for the placement. */
typedef struct usearch_shapes_init_vram_t {

    /** Output for the new index. */
    usearch_shapes_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The kind comparing two shapes. */
    usearch_shapes_metric_t metric;

    /** The scalar type points are stored in: @c f16 or @c f32. */
    usearch_scalar_t quantization;

    /** The frozen point count, when every entry has one; zero admits any count. */
    size_t points;

    /** The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** Upper bound on entries sharing one key; zero derives to one - unique keys. */
    size_t values_per_key;

    /** The devices to place one shard on apiece. */
    usearch_vram_options_t vram;
} usearch_shapes_init_vram_t;

/** Creates an empty, mutable shape index in device memory. */
USEARCH_EXPORT void usearch_shapes_init_vram(usearch_shapes_init_vram_t* arguments);

/** The arguments of @c usearch_shapes_load_ram. */
typedef struct usearch_shapes_load_ram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_shapes_index_t* index;

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

    /** Upper bound on entries sharing one key, which no checkpoint records; zero derives to one. */
    size_t values_per_key;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
} usearch_shapes_load_ram_t;

/**
 *  @brief Copies a checkpoint into a new, mutable index in host memory, its geometry read from
 *      the checkpoint's own headers.
 *
 *  Where the parts end up is the call's own memory domains, never the manifest's. A part
 *  carries its own length in its header, so a second load skips whatever already arrived
 *  whole and cancelling costs the part being read rather than everything behind it.
 */
USEARCH_EXPORT void usearch_shapes_load_ram(usearch_shapes_load_ram_t* arguments);

/** The arguments of @c usearch_shapes_load_vram. */
typedef struct usearch_shapes_load_vram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_shapes_index_t* index;

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

    /** Upper bound on entries sharing one key, which no checkpoint records; zero derives to one. */
    size_t values_per_key;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
} usearch_shapes_load_vram_t;

/** Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
 *  @c usearch_shapes_load_ram. */
USEARCH_EXPORT void usearch_shapes_load_vram(usearch_shapes_load_vram_t* arguments);

/** The arguments of @c usearch_shapes_view_ram. */
typedef struct usearch_shapes_view_ram_t {

    /** Output for the new, read-only index. */
    usearch_shapes_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;
} usearch_shapes_view_ram_t;

/** Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
 *  verbs report @c usearch_status_read_only_k. */
USEARCH_EXPORT void usearch_shapes_view_ram(usearch_shapes_view_ram_t* arguments);

/** The arguments of @c usearch_shapes_view_nvme. */
typedef struct usearch_shapes_view_nvme_t {

    /** Output for the new, read-only index, written by the time the call is final. */
    usearch_shapes_index_t* index;

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
} usearch_shapes_view_nvme_t;

/**
 *  @brief Serves a checkpoint in place from NVMe drives as a read-only index, every later call
 *      reading it through its executor's engines.
 *
 *  Opening reads the headers alone. Every mutating verb on the handle reports
 *  @c usearch_status_read_only_k, and the parts must stay unmodified until the handle is freed.
 */
USEARCH_EXPORT void usearch_shapes_view_nvme(usearch_shapes_view_nvme_t* arguments);

/** Frees the resources associated with the index; @c NULL is safe, and it cannot fail, so takes
 *  no outcome.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_shapes_free(usearch_shapes_index_t index);

/** The arguments of @c usearch_shapes_reserve. */
typedef struct usearch_shapes_reserve_t {

    /** The handle to be resized. */
    usearch_shapes_index_t index;

    /** The executor the growth runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The desired total entry capacity, including the current size. */
    size_t entries;

    /** Expected total point volume across all entries; zero grows lazily. */
    size_t points;
} usearch_shapes_reserve_t;

/** Reserves capacity ahead of insertion. */
USEARCH_EXPORT void usearch_shapes_reserve(usearch_shapes_reserve_t* arguments);

/** The arguments of @c usearch_shapes_clear. */
typedef struct usearch_shapes_clear_t {

    /** The handle to be cleared. */
    usearch_shapes_index_t index;

    /** The executor the erasure runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;
} usearch_shapes_clear_t;

/** Erases all entries, keeping the reserved capacity and frozen geometry. */
USEARCH_EXPORT void usearch_shapes_clear(usearch_shapes_clear_t* arguments);

/** The arguments of @c usearch_shapes_insert. */
typedef struct usearch_shapes_insert_t {

    /** The handle to be populated. */
    usearch_shapes_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Shapes to admit - the batch axis of @c keys and @c shapes. */
    size_t count;

    /** The identities to admit the shapes under, one per entry and not necessarily unique. */
    usearch_keys_view_t keys;

    /** The shapes, points end to end around @c count + 1 point offsets. @ref shapes_index_tape */
    usearch_tape_view_t shapes;

    /** The @b optional breadth of the candidates the build walk explores; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries admitted and the batch length. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_shapes_insert_t;

/**
 *  @brief Adds a tape of shapes under their keys - the native batch operation; a single shape is
 *      a tape of one. Point counts may differ freely between entries. Overlaps safely with
 *      concurrent searches on the same handle. @ref shapes_index_concurrency
 *  @retval usearch_status_read_only_k The handle is a view.
 *  @retval usearch_status_overflow_k A key already holds @c values_per_key entries.
 *  @retval usearch_status_shape_mismatch_k An entry's point count differs from a frozen @c points.
 *
 *  An insertion is all-or-nothing per call. Every fallible step runs before the first entry
 *  identity is claimed, so a rejected call publishes nothing - a cancelled one included, which
 *  reports @c usearch_status_cancelled_k - and on a RAM handle the entries of an accepted one
 *  become visible at once.
 *
 *  A key already holding @c values_per_key entries rejects the whole call, since a smaller retry
 *  cannot help.
 */
USEARCH_EXPORT void usearch_shapes_insert(usearch_shapes_insert_t* arguments);

/** The arguments of @c usearch_shapes_refine. */
typedef struct usearch_shapes_refine_t {

    /** The handle to be refined. */
    usearch_shapes_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The @b optional output for the entries whose neighbors changed; zero means settled. */
    size_t* changed;

    /** The @b optional output for the audit closing the heal; @c reachable short of @c audited is
     *  what no reconnection could reach. */
    usearch_reachability_t* reachability;

    /** The @b optional breadth of the candidates the re-selection walk explores; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries re-selected; a cancel stops the re-selection,
     *  still heals, and answers @c usearch_status_cancelled_k. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_shapes_refine_t;

/**
 *  @brief Re-selects the neighbors of every stored entry against the finished graph, improving
 *      neighbors chosen when the graph around them was still incomplete.
 *  @retval usearch_status_read_only_k The handle is an NVMe view.
 *  @retval usearch_status_unsupported_k The handle lives on a GPU, whose index refines nothing yet.
 *  @retval usearch_status_contended_k An insertion or another refinement holds the handle.
 *
 *  This is a build step rather than a repair. A cheap insertion pass followed by one refinement
 *  reaches a better graph, in less total time, than a single expensive insertion pass - so a wide
 *  @c expansion_factor belongs here rather than in @c usearch_shapes_insert. It pays most in this
 *  modality, where one comparison scores a whole shape against another.
 *
 *  Cancelling is safe, which is what makes a whole-graph sweep affordable. Re-selecting an entry's
 *  neighbors is idempotent and claims no identity, so a stopped pass leaves the graph correct and
 *  merely less improved: nothing unwinds, and the call answers @c usearch_status_cancelled_k. The
 *  index keeps nothing of a stopped pass, so the next call starts at the first entry again; a
 *  caller wanting to continue where a cancel stopped holds a @c usearch_shapes_refinement_t
 *  instead.
 *
 *  Re-selection alone can strand an entry no search reaches, so every call closes by re-linking
 *  them: unreachable entries re-select from scratch, and whatever the selection still drops is
 *  reconnected to a reachable one. A cancelled call skips the rest of the re-selection but still
 *  re-links, so it never leaves entries unreachable.
 *
 *  Searches may run throughout: re-selection rewrites one entry at a time and never removes one,
 *  so a concurrent walk sees either the old neighbors or the new ones. Insertions never overlap:
 *  one in flight refuses the call with @c usearch_status_contended_k, and one arriving meanwhile
 *  is refused the same way, as is a save.
 */
USEARCH_EXPORT void usearch_shapes_refine(usearch_shapes_refine_t* arguments);

/** The arguments of @c usearch_shapes_search. */
typedef struct usearch_shapes_search_t {

    /** The handle to be queried. */
    usearch_shapes_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Query shapes to resolve; the outputs have @c count rows of @c wanted entries. */
    size_t count;

    /** The query shapes, as points end to end around @c count + 1 point offsets. */
    usearch_tape_view_t queries;

    /** Output for the run, inside @c arena. */
    usearch_matches_view_t* matches;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /** Columns per row, and the upper bound on matches - the "k" in "kANN". */
    size_t wanted;

    /** The @b optional breadth of the candidates a walk explores; zero derives, a scan ignores
     *  it. */
    size_t expansion_factor;

    /** How rows are reached: a walk, the default, or an exact scan of every row, on RAM and
     *  NVMe. */
    usearch_traversal_t traversal;

    /** The @b optional admission filter, asked about a batch of candidates at a time. */
    usearch_predicate_t predicate;

    /** The @b optional handler, called with queries resolved and their count. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_shapes_search_t;

/**
 *  @brief Performs k-Nearest Neighbors Search for a tape of query shapes, approximate by walking
 *      the graph or exact by scanning every row; one query is a tape of one.
 *  @retval usearch_status_unknown_kind_k A @c traversal other than a walk or a scan.
 *  @retval usearch_status_unsupported_k A scan of a VRAM handle, or a @c predicate on one.
 *
 *  The native batch operation, parallelized internally. Query point counts are unconstrained and
 *  need not match any indexed entry. Never writes to the graph, so any number may overlap with
 *  concurrent insertions.
 *  @ref shapes_index_concurrency
 *
 *  A @c predicate narrows the matches to the entries it admits. Non-matching entries still route
 *  the traversal, so recall does not collapse at low selectivity.
 *
 *  A cancelled search reports @c usearch_status_cancelled_k and leaves @c matches @c NULL.
 */
USEARCH_EXPORT void usearch_shapes_search(usearch_shapes_search_t* arguments);

/** What one key holds, enough to size a buffer before retrieving any of it. */
typedef struct usearch_shapes_key_stats_t {

    /** Entries sharing the key; keys need not be unique. */
    size_t entries;

    /** Points summed across those entries. */
    size_t total_points;

    /** Points of the largest single entry. */
    size_t largest_points;
} usearch_shapes_key_stats_t;

/** The arguments of @c usearch_shapes_count. */
typedef struct usearch_shapes_count_t {

    /** The handle to be queried. */
    usearch_shapes_index_t index;

    /** The executor the lookup runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are counted. */
    usearch_key_t key;

    /** Output for what @c key holds; zero @c entries means the index holds none. */
    usearch_shapes_key_stats_t* key_stats;
} usearch_shapes_count_t;

/** Counts the entries stored under one key, with the extent of their points. */
USEARCH_EXPORT void usearch_shapes_count(usearch_shapes_count_t* arguments);

/** The arguments of @c usearch_shapes_get. */
typedef struct usearch_shapes_get_t {

    /** The handle to be queried. */
    usearch_shapes_index_t index;

    /** The executor whose stream or thread copies the points; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are exported. */
    usearch_key_t key;

    /** Output for the tape inside @c arena: every entry's points end to end, @c x, @c y and @c z
     *  interleaved in @c outputs_scalar, around @c written + 1 point offsets. */
    usearch_tape_view_t* shapes;

    /** The @b optional output for the entries written; zero when the key is absent. */
    size_t* written;

    /** Memory for @c shapes, rewound by the call. */
    usearch_arena_t arena;

    /** The scalar type to export as, converted from storage where it differs. */
    usearch_scalar_t outputs_scalar;
} usearch_shapes_get_t;

/**
 *  @brief Retrieves every shape stored under one key as an Apache Arrow tape, in insertion order -
 *      one call, however many and however large they turn out to be.
 *
 *  Each shape comes back in the pose it was inserted in, to the precision of the stored scalar.
 *
 *  An absent key leaves the tape's pointers @c NULL.
 */
USEARCH_EXPORT void usearch_shapes_get(usearch_shapes_get_t* arguments);

/** The arguments of @c usearch_shapes_absorb. */
typedef struct usearch_shapes_absorb_t {

    /** The handle absorbing the donors. */
    usearch_shapes_index_t index;

    /** The executor each donor's landing and the closing refinement fan out over. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Donor handles to absorb. */
    size_t count;

    /** Array of @c count handles, each consumed and nulled once its entries are published here. */
    usearch_shapes_index_t* donors;

    /** The @b optional output for the entries the closing refine changed. */
    size_t* changed;

    /** The @b optional output for the audit closing the heal. */
    usearch_reachability_t* reachability;

    /** The @b optional breadth of the candidates of the closing refine's walks; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries the closing refine re-selected; a cancel stops
     *  it as it stops @c usearch_shapes_refine, never a landing. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_shapes_absorb_t;

/**
 *  @brief Absorbs every handle in @c donors into @c index, one after another, keeping their graphs
 *      rather than re-inserting their shapes, and links each in by walks across. Donor handles are
 *      consumed and must not be used afterwards, not even for @c usearch_shapes_free.
 *  @retval usearch_status_read_only_k @c index is an NVMe view.
 *  @retval usearch_status_unsupported_k @c index lives on a GPU, whose index absorbs nothing yet.
 *  @retval usearch_status_invalid_argument_k A @c NULL slot in @c donors, or @c NULL @c donors
 *      for a nonzero @c count.
 *  @retval usearch_status_config_mismatch_k A donor of another residency, class, scalar, kind or
 *      geometry, or @c index itself.
 *  @retval usearch_status_contended_k An insertion or refinement in flight on @c index or on a
 *      donor; every donor stays a live handle.
 *
 *  Every donor is checked and held before the first one lands, so a refusal consumes none. The
 *  donors then land one after another, each all-or-nothing: its entries append to @c index with
 *  their edges rebased, publish at once, and the donor is consumed, so at most one donor's entries
 *  are ever held twice. One refinement then walks every entry across the parts the donors appended,
 *  round by round, so a group of donors costs one refinement rather than one per donor. Readers of
 *  @c index may keep searching throughout, since unpublished entries hide behind the watermark,
 *  while nothing may search a donor. Insertions into @c index arriving meanwhile are refused with
 *  @c usearch_status_contended_k.
 *
 *  The handler reports and cancels the closing refinement alone, exactly as in
 *  @c usearch_shapes_refine: a cancel skips the rest of the walks across, still re-links, and
 *  answers @c usearch_status_cancelled_k with every donor consumed. A caller wanting to continue
 *  those walks later holds a @c usearch_shapes_refinement_t instead.
 */
USEARCH_EXPORT void usearch_shapes_absorb(usearch_shapes_absorb_t* arguments);

/** The arguments of @c usearch_shapes_split. */
typedef struct usearch_shapes_split_t {

    /** The handle split into the receivers, left empty and live. */
    usearch_shapes_index_t index;

    /** The executor the partition, each part's landing and the closing refinements fan out over. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Receiver handles to split into, at least two. */
    size_t count;

    /** Array of @c count empty handles of the source's residency, class, scalar, kind and geometry,
     *  each receiving one part. */
    usearch_shapes_index_t const* receivers;

    /** The @b optional output for the entries the closing refines changed, across the receivers. */
    size_t* changed;

    /** The @b optional output for the audits closing the heals, across the receivers. */
    usearch_reachability_t* reachability;

    /** The @b optional breadth of the candidates of the closing refines' walks; zero derives. */
    size_t expansion_factor;

    /** Picks the part roots: equal seeds over an unchanged index split equally. */
    uint64_t seed;

    /** The @b optional handler, called with entries the closing refines re-selected; a cancel stops
     *  them as it stops @c usearch_shapes_refine, never a landing. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_shapes_split_t;

/**
 *  @brief Splits @c index into the handles in @c receivers, the inverse of
 *      @c usearch_shapes_absorb: graph-local parts keeping their edges rather than re-inserting
 *      their shapes, one part per receiver, and @c index left empty but live.
 *  @retval usearch_status_read_only_k @c index is a view.
 *  @retval usearch_status_invalid_argument_k Fewer than two receivers, @c NULL @c receivers, or a
 *      @c NULL slot in them.
 *  @retval usearch_status_config_mismatch_k A receiver of another residency, class, scalar, kind or
 *      geometry, one holding entries, one named twice, or @c index itself.
 *  @retval usearch_status_contended_k An insertion or refinement in flight on @c index or on a
 *      receiver; nothing moves.
 *
 *  Every receiver is checked and held before the first part lands, so a refusal moves nothing. The
 *  part roots are drawn from @c seed without a scan, and each part grows breadth-first over the
 *  links from its root, capped at an equal share of the entries, while entries no root reaches join
 *  the smallest part. Each part then lands in its receiver all-or-nothing, its edges rebased, its
 *  edges into other parts dropped, and the entries that lost one placed first; @c index clears once
 *  every part landed. Each receiver closes by refining the entries that lost an edge and re-linking
 *  what that strands, as @c usearch_shapes_refine does. Insertions into @c index or a receiver
 *  arriving meanwhile are refused with @c usearch_status_contended_k, and nothing may search either
 *  until the call is final.
 *
 *  The handler reports and cancels the closing refinements alone, exactly as in
 *  @c usearch_shapes_refine: a cancel skips the rest of the re-selection, still re-links, and
 *  answers @c usearch_status_cancelled_k with every part landed. A caller wanting to continue that
 *  re-selection later holds a @c usearch_shapes_refinement_t opened with @c receivers instead.
 */
USEARCH_EXPORT void usearch_shapes_split(usearch_shapes_split_t* arguments);

/** The arguments of @c usearch_shapes_refinement_init. */
typedef struct usearch_shapes_refinement_init_t {

    /** The handle to be refined. */
    usearch_shapes_index_t index;

    /** The executor every call on the refinement runs on; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output for the new handle; @c NULL on failure. The index admits no insertion or save until
     *  @c usearch_shapes_refinement_finish frees it. */
    usearch_shapes_refinement_t* refinement;

    /** The @b optional donors to absorb first, each consumed and nulled once its entries land; the
     *  array must outlive the refinement. */
    usearch_shapes_index_t* donors;

    /** Donor handles in @c donors. */
    size_t count;

    /** The @b optional receivers to split @c index into first, as @c usearch_shapes_split lands
     *  them, each refined in turn; the array must outlive the refinement. */
    usearch_shapes_index_t const* receivers;

    /** Receiver handles in @c receivers, zero or at least two. */
    size_t receivers_count;

    /** The @b optional breadth of the candidates the re-selection walk explores; zero derives. */
    size_t expansion_factor;

    /** Picks the part roots when @c receivers are named, as in @c usearch_shapes_split. */
    uint64_t seed;
} usearch_shapes_refinement_init_t;

/**
 *  @brief Opens a refinement of @c index, absorbing @c donors first when any are named, as
 *      @c usearch_shapes_absorb lands them, or splitting it into @c receivers, as
 *      @c usearch_shapes_split lands them. Every allocation the refinement needs in the index
 *      happens here, never in a step.
 *  @retval usearch_status_read_only_k @c index is an NVMe view.
 *  @retval usearch_status_unsupported_k @c index lives on a GPU, whose index refines nothing yet.
 *  @retval usearch_status_invalid_argument_k No executor, a @c NULL slot in @c donors or
 *      @c receivers, @c NULL @c donors for a nonzero @c count, one receiver alone, or donors and
 *      receivers both named.
 *  @retval usearch_status_config_mismatch_k A donor or receiver of another residency, class,
 *      scalar, kind or geometry, a receiver holding entries, or @c index itself.
 *  @retval usearch_status_contended_k An insertion or another refinement in flight on @c index, on
 *      a donor or on a receiver.
 */
USEARCH_EXPORT void usearch_shapes_refinement_init(usearch_shapes_refinement_init_t* arguments);

/** The arguments of @c usearch_shapes_refinement_next. */
typedef struct usearch_shapes_refinement_next_t {

    /** The handle the refinement was opened on. */
    usearch_shapes_index_t index;

    /** The executor the refinement was opened with. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The refinement to continue. */
    usearch_shapes_refinement_t refinement;

    /** The @b optional handler, called with entries re-selected across every call so far. */
    usearch_progress_t progress;

    /** The @b optional outputs for the entries re-selected so far and, of those, the ones whose
     *  neighbors changed; their ratio falling is what says another call buys little. */
    size_t* visited;
    size_t* changed;

    /** The @b optional traversal telemetry of every call on the refinement so far; @c NULL skips
     *  reporting. */
    usearch_traversal_stats_t* stats;
} usearch_shapes_refinement_next_t;

/**
 *  @brief Continues the refinement until it ends or @c progress cancels, landing any donors still
 *      pending first. Answers @c usearch_status_cancelled_k when a later call has more to do; that
 *      call continues exactly where this one stopped. The executor is held only for the call.
 *
 *  Every call reaching the refinement writes the running totals, a cancelled or failed one too, so
 *  they only grow from one call to the next and @c changed never exceeds @c visited; all stay zero
 *  while donors are pending.
 *  @retval usearch_status_invalid_argument_k @c index or @c executor is not the one the refinement
 *      was opened with.
 */
USEARCH_EXPORT void usearch_shapes_refinement_next(usearch_shapes_refinement_next_t* arguments);

/** The arguments of @c usearch_shapes_refinement_finish. */
typedef struct usearch_shapes_refinement_finish_t {

    /** The handle the refinement was opened on. */
    usearch_shapes_index_t index;

    /** The executor the refinement was opened with. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The refinement to finish; freed by the call, whatever it answers. */
    usearch_shapes_refinement_t refinement;

    /** The @b optional output for the number of entries whose neighbors changed. */
    size_t* changed;

    /** The @b optional output for the audit closing the heal. */
    usearch_reachability_t* reachability;

    /** The @b optional traversal telemetry of every call on the refinement; @c NULL skips
     *  reporting. */
    usearch_traversal_stats_t* stats;
} usearch_shapes_refinement_finish_t;

/**
 *  @brief Skips whatever re-selection a stopped refinement left, re-links what it strands as
 *      @c usearch_shapes_refine does, reports, reopens the index, and frees the handle. A
 *      refinement whose donors have not all landed re-links nothing and reports zeros.
 *  @retval usearch_status_invalid_argument_k @c index or @c executor is not the one the refinement
 *      was opened with; the handle is still freed.
 */
USEARCH_EXPORT void usearch_shapes_refinement_finish(usearch_shapes_refinement_finish_t* arguments);

/** The arguments of @c usearch_shapes_cursor_init. */
typedef struct usearch_shapes_cursor_init_t {

    /** Output for the new cursor; @c NULL on failure. */
    usearch_shapes_cursor_t* cursor;

    /** The handle to be scanned. */
    usearch_shapes_index_t index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The @b optional count of @c starts; zero walks every entry as stored. */
    size_t starts_count;

    /** The keys to walk from, in the order the walk begins with them. */
    usearch_key_t const* starts;
} usearch_shapes_cursor_init_t;

/**
 *  @brief Opens a snapshot cursor over the current contents; see @c usearch_shapes_cursor_t.
 *  @retval usearch_status_invalid_argument_k @c NULL @c starts for a nonzero @c starts_count.
 *
 *  Without @c starts the cursor yields every entry as stored. With them it yields the start keys'
 *  own entries first, in the order given, then the entries those link to, breadth-first, each
 *  entry once, so a caller wanting only the starts' immediate links reads the rows of the first
 *  pulled entries. A key the index lacks starts nothing.
 */
USEARCH_EXPORT void usearch_shapes_cursor_init(usearch_shapes_cursor_init_t* arguments);

/** The arguments of @c usearch_shapes_cursor_next. */
typedef struct usearch_shapes_cursor_next_t {

    /** The handle the cursor was opened on. */
    usearch_shapes_index_t index;

    /** The executor the pull runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The cursor to advance. */
    usearch_shapes_cursor_t cursor;

    /** Output for the identities, inside @c arena. */
    usearch_keys_view_t* keys;

    /** The @b optional output for each entry's links, the keys of the entries it links to in the
     *  graph, row @c i spanning @c offsets[i] up to @c offsets[i+1], inside @c arena; @c NULL skips
     *  them. */
    usearch_keys_tape_view_t* links;

    /** Output for the entries written; zero when the snapshot is exhausted. */
    size_t* written;

    /** Memory for @c keys and @c links, rewound by the call. */
    usearch_arena_t arena;

    /** The upper bound on entries this pull writes. */
    size_t wanted;
} usearch_shapes_cursor_next_t;

/**
 *  @brief Copies the next batch of entries out of the cursor into @c arena, each with the keys it
 *      links to, in the order @c usearch_shapes_cursor_init describes; the shapes themselves come
 *      from @c usearch_shapes_get.
 *  @retval usearch_status_invalid_argument_k @c index is not the handle the cursor was opened on.
 *
 *  Links name only entries the snapshot holds, as the graph stored them when the pull ran. The
 *  keys are copied rather than borrowed from the index, so the batch stays valid until the next
 *  call given the same arena, however the cursor moves meanwhile.
 */
USEARCH_EXPORT void usearch_shapes_cursor_next(usearch_shapes_cursor_next_t* arguments);

/** Releases the cursor; @c NULL is safe.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_shapes_cursor_free(usearch_shapes_cursor_t cursor);

/** The arguments of @c usearch_shapes_sample. */
typedef struct usearch_shapes_sample_t {

    /** The handle to be sampled. */
    usearch_shapes_index_t index;

    /** The executor the draws run on; its kind must match the handle's residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The @b optional count of @c anchors; zero draws from the whole index. */
    size_t anchors_count;

    /** The shapes that bound the region around them, one anchor apiece, as points end to end
     *  around @c anchors_count + 1 point offsets. @ref shapes_index_tape */
    usearch_tape_view_t anchors;

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

    /** The @b optional radius around the anchors; zero bounds nothing. */
    usearch_distance_t radius;

    /** The @b optional breadth of the candidates of the walks behind the draws; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with draws written and @c wanted. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_shapes_sample_t;

/**
 *  @brief Draws entries uniformly, across the index or around anchors, each linked to its nearest
 *      earlier draws - so the sample is one connected component, ready to be laid out.
 *
 *  Links name earlier draws of the same sequence, a later call's reaching an earlier call's. They
 *  are exact in a scan and approximate as any walk otherwise.
 */
USEARCH_EXPORT void usearch_shapes_sample(usearch_shapes_sample_t* arguments);

/** The arguments of @c usearch_shapes_save. */
typedef struct usearch_shapes_save_t {

    /** The handle to be written. */
    usearch_shapes_index_t index;

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
} usearch_shapes_save_t;

/**
 *  @brief Saves a mutable index under @c path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  A cancelled save leaves the parts it already wrote and no manifest referencing them, so the
 *  checkpoint reads as absent rather than half-valid. Those leftovers are the caller's to remove.
 */
USEARCH_EXPORT void usearch_shapes_save(usearch_shapes_save_t* arguments);

/** The geometry and metric a checkpoint froze - everything a reader learns without opening it. */
typedef struct usearch_shapes_metadata_t {

    /** The kind the checkpoint compares shapes with. */
    usearch_shapes_metric_t metric;

    /** The scalar type its points are stored in. */
    usearch_scalar_t quantization;

    /** The frozen point count, or zero for any count. */
    size_t points;

    /** Connections per node the graph was built with. */
    size_t connectivity;
} usearch_shapes_metadata_t;

/** The arguments of @c usearch_shapes_metadata_read. */
typedef struct usearch_shapes_metadata_read_t {

    /** The executor the read runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** Output for every field the checkpoint froze. */
    usearch_shapes_metadata_t* metadata;
} usearch_shapes_metadata_read_t;

/** Reads a checkpoint's geometry & metric without loading it. */
USEARCH_EXPORT void usearch_shapes_metadata_read(usearch_shapes_metadata_read_t* arguments);

/** A live index's state. Each counter is read once, so under concurrent insertions they may come
 *  from slightly different moments. */
typedef struct usearch_shapes_index_description_t {

    /** The frozen metric, quantization, points and connectivity, as a checkpoint reports them. */
    usearch_shapes_metadata_t metadata;

    /** Stored entries. */
    size_t size;

    /** Entries hosted without growing again. */
    size_t capacity;

    /** Points summed across every stored entry. */
    size_t total_points;

    /** Bytes the handle holds, point storage included; traversal scratch is the executor's. */
    size_t memory_usage;

    /** Static codename of the kernels the index dispatched to on this machine. */
    char const* capabilities;
} usearch_shapes_index_description_t;

/** The arguments of @c usearch_shapes_describe. */
typedef struct usearch_shapes_describe_t {

    /** The handle to be described. */
    usearch_shapes_index_t index;

    /** The executor the report runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output, filled whole. */
    usearch_shapes_index_description_t* description;
} usearch_shapes_describe_t;

/** Reports a live index's frozen metadata and current counters in one call. */
USEARCH_EXPORT void usearch_shapes_describe(usearch_shapes_describe_t* arguments);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_SHAPES_INDEX_H
