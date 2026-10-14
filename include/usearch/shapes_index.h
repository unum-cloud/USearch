/**
 *  @file include/usearch/shapes_index.h
 *  @author Ash Vardanian
 *  @date August 15, 2026
 *  @brief The matrix-modality C interface: similarity search over rank-2 tensors - sets of
 *      equidimensional rows whose count varies per entry - under set-to-set and transport scoring.
 *
 *  One entry is a matrix: @c rows varies, @c columns is frozen index-wide. A molecular
 *  conformer is a few dozen atom positions of 3 columns; a point cloud is a few thousand of
 *  them. Neither the two sides of a comparison nor two entries of the index need agree on
 *  their row count - a 20-atom molecule scores against a 60-atom molecule, without padding,
 *  truncation, or resampling anywhere in the interface.
 *
 *  Rows are unordered, and a row alone means nothing - three coordinates are a position only
 *  in the company of the rest of the conformer. That is what separates this modality from
 *  late interaction, where each row is a whole embedding and the set is scored by reducing
 *  over rows the graph already indexes individually: that belongs to the dense modality's
 *  @c usearch_dense_maxsim. Where ordering carries meaning the sequence modality owns the
 *  object, and where every entry has exactly one row the dense modality does.
 *
 *  Handles live in RAM or VRAM, created empty by @c usearch_matrix_init_ram or
 *  @c usearch_matrix_init_vram, copied from a checkpoint by @c usearch_matrix_load_ram or
 *  @c usearch_matrix_load_vram, or served from one in place by @c usearch_matrix_view_ram, which
 *  is read-only and refuses every mutating verb with @c usearch_status_read_only_k.
 *
 *  @section matrix_index_tape Batch Shape
 *
 *  Variable row counts cannot ride a run of equal rows, so every batch input is a @b tape: every
 *  entry's rows end to end, and @c count + 1 Apache Arrow offsets counting rows around them.
 *  Entry @c i holds `(offsets[i+1] - offsets[i]) * columns` scalars, row-major and tightly
 *  packed. Keys need not be unique - three conformers of one molecule share its key, and the
 *  modality reports how many entries a key holds.
 *
 *  @section matrix_index_scoring Scoring Kinds
 *
 *  The first two compare rows to rows: each row of one side finds its best partner on the
 *  other, and they differ only in how those bests are aggregated. Chamfer averages them in
 *  both directions, Hausdorff takes the worst. Both are symmetric and both obey the triangle
 *  inequality, which is what lets them weight a graph edge at all.
 *
 *  Gromov-Wasserstein instead compares each side's @b internal geometry, transporting rows
 *  onto rows so every row is accounted for rather than letting one row absorb many.
 *  Comparing internal geometry is what lets it score objects whose rows live in unrelated
 *  frames, and for coordinates it is invariant to rotation and translation with no
 *  superposition step - so two molecules of different atom counts compare directly.
 *  Handedness survives, since pairwise distances alone cannot see it and the derived row
 *  features carry it.
 *
 *  Kabsch and Umeyama take the opposite route, reporting the residual after actually
 *  superposing one set onto the other. That needs to know which row pairs with which, so
 *  they ask for a fixed @c rows - three conformers of one molecule in a fixed atom order, a
 *  protein backbone of fixed length, a landmark set. Equal heights alone are not enough:
 *  two unrelated molecules of thirty atoms each have no correspondence to superpose along,
 *  which is why the fixed count is opt-in rather than inferred from what arrives.
 *
 *  @section matrix_index_determinism Determinism
 *
 *  Transport scores come from an iterative solver, so the modality pins every
 *  source of variation a graph index cannot tolerate: a fixed initialization, a fixed
 *  iteration budget rather than a convergence-tolerance early exit, and a canonical
 *  ordering of each unordered pair before it is solved. Without that last step a solver
 *  can settle in two different local optima depending on argument order, and the graph
 *  would disagree with itself about an edge it had already inserted.
 *
 *  @section matrix_index_concurrency Concurrency & Thread Safety
 *
 *  A handle may be shared across threads. On a RAM handle, insertions and searches overlap
 *  freely: searches read only entries published behind the visibility watermark and never
 *  write to the graph, while insertions synchronize through per-node locks and publish in
 *  order. A VRAM handle is driven through the streams of its executor. A viewed handle admits
 *  arbitrarily many concurrent searches, having nothing to mutate. Serialization is the one
 *  exception: saving demands a mutation-free window, though concurrent readers keep
 *  searching throughout. Per-call outputs - counts and the outcome - belong to their
 *  call and must not be shared between calls that are not final.
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
#ifndef UNUM_USEARCH_MATRIX_INDEX_H
#define UNUM_USEARCH_MATRIX_INDEX_H

#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Similarity Search Index over rank-2 objects of varying height.
 *
 *  @see usearch_matrix_init_ram
 *  @see usearch_matrix_insert
 *  @see usearch_matrix_search
 */
typedef struct usearch_matrix_index_handle_t* usearch_matrix_index_t;

/** A consistent snapshot cursor over the index contents. Captures the publication
 *  watermark at creation, so concurrent inserts never appear mid-iteration, and the
 *  walk is a sequential entry scan - export, migration, and visualization feed off it.
 *  Each pull names the index again; an overlapping pull reports @c usearch_status_contended_k. */
typedef struct usearch_matrix_cursor_handle_t* usearch_matrix_cursor_t;

/**
 *  @brief How row-level comparisons become one score for the whole matrix.
 *
 *  Writes @c d for the distance between two individual rows, and @c D for the @b [|A|,|B|]
 *  matrix of every such distance. The three families differ in what they build: the
 *  best-partner kinds form @c D once and reduce it twice, the transport kind never forms it
 *  and compares each side's own @b [|A|,|A|] and @b [|B|,|B|] geometries instead, and the
 *  superposition kinds form no matrix at all, pairing row @c i with row @c i.
 *
 *  Only the symmetric kinds can weigh the index's own edges. A kind that absorbs a scale
 *  factor onto one argument answers differently when its two sides swap, and an edge has no
 *  query to be asymmetric about, so such a kind is served from an index built under a
 *  symmetric neighbour that bounds it - picked by the library and never named by the caller.
 *  @ref matrix_index_scoring
 */
typedef enum usearch_matrix_metric_t {
    usearch_matrix_metric_unknown_k = 0,

    /**
     *  @brief Mean of each side's best partners, both directions summed.
     *
     *  Every row is charged the distance to the nearest row of the other matrix, and those
     *  charges are averaged, so a single unpartnered row is diluted rather than decisive.
     *
     *  CD(A,B) = mean_i min_j D[i,j]  +  mean_j min_i D[i,j]
     *
     *  @code{.py}
     *  D = cdist(A, B) # [|A|,|B|] - the whole cost
     *  chamfer = D.min(axis=1).mean() + D.min(axis=0).mean()
     *  @endcode
     */
    usearch_matrix_metric_chamfer_k = 1,

    /**
     *  @brief The worst of all best partners, both directions - the metric of the family.
     *
     *  Unforgiving where Chamfer is lenient: one row whose nearest row in the other matrix is
     *  far away sets the entire score, which is what makes it a true metric and therefore
     *  usable as an edge weight.
     *
     *  H(A,B) = max( max_i min_j D[i,j], max_j min_i D[i,j] )
     *
     *  @code{.py}
     *  D = cdist(A, B) # [|A|,|B|] - the same matrix Chamfer forms
     *  hausdorff = max(D.min(axis=1).max(), D.min(axis=0).max())
     *  @endcode
     */
    usearch_matrix_metric_hausdorff_k = 2,

    /**
     *  @brief Transport over each side's internal geometry - the molecular-shape kind.
     *
     *  Compares how each matrix's own rows sit relative to one another and never their
     *  absolute coordinates, so a rotated, reflected or row-permuted copy scores zero, no
     *  shared row order is needed, and ragged heights are natural. @c P is the transport plan
     *  pairing mass between the two sides, and the objective is quartic in the row counts
     *  before any solver iterates - far and away the most expensive kind here.
     *
     *  GW(A,B) = min_P √( Σ_{i,k,j,l} |D_A[i,k] − D_B[j,l]|² · P[i,j] · P[k,l] )
     *
     *  @code{.py}
     *  DA, DB = cdist(A, A), cdist(B, B) # [|A|,|A|] and [|B|,|B|]; never A vs B
     *  L = (DA[:, None, :, None] - DB[None, :, None, :]) ** 2 # [|A|,|B|,|A|,|B|]
     *  gromov = min(sqrt(einsum("ijkl,ij,kl->", L, P, P)) for P in plans)
     *  @endcode
     */
    usearch_matrix_metric_gromov_wasserstein_k = 3,

    /**
     *  @brief Residual after the best rotation - Kabsch. Needs a fixed @c rows.
     *
     *  Row @c i pairs with row @c i rather than with its nearest neighbour, which is what the
     *  frozen height buys and what makes this the one family linear in the row count. Both
     *  sides are centred first, so translation costs nothing and only orientation is solved
     *  for, out of a covariance no larger than @b [columns,columns].
     *
     *  RMSD(A,B) = min_{R ∈ SO(3)} √( 1/n · Σ_i ‖aᵢ − R·bᵢ‖² )
     *
     *  @code{.py}
     *  A, B = A - A.mean(0), B - B.mean(0)          # centring makes translation free
     *  U, S, Vt = svd(B.T @ A)                      # [3,3] covariance from an O(n) pass
     *  R = U @ diag([1, 1, sign(det(U @ Vt))]) @ Vt # reflections rejected: enantiomers differ
     *  kabsch = sqrt(((A - B @ R) ** 2).sum() / len(A))
     *  @endcode
     */
    usearch_matrix_metric_kabsch_k = 4,

    /**
     *  @brief Residual after the best rotation and uniform scaling - Umeyama. Ranks only.
     *
     *  Needs a fixed @c rows, and is served from its symmetric counterpart: @c s multiplies
     *  one argument and not the other, so swapping the two sides solves for @c 1/s and a
     *  different residual, leaving the kind symmetric only where it is identically zero. The
     *  asymmetry is visible below as the divisor, which reads one side's spread alone.
     *
     *  U(A,B) = min_{s>0, R ∈ SO(3)} √( 1/n · Σ_i ‖aᵢ − s·R·bᵢ‖² )
     *
     *  @code{.py}
     *  A, B = A - A.mean(0), B - B.mean(0)
     *  U, S, Vt = svd(B.T @ A)                      # [3,3] covariance, as Kabsch
     *  R = U @ diag([1, 1, sign(det(U @ Vt))]) @ Vt
     *  s = S.sum() / (B ** 2).sum()                 # B's spread alone - hence asymmetric
     *  umeyama = sqrt(((A - s * (B @ R)) ** 2).sum() / len(A))
     *  @endcode
     */
    usearch_matrix_metric_umeyama_k = 5,
} usearch_matrix_metric_t;

/** The arguments of @c usearch_matrix_init_ram - the geometry and metric an empty index freezes,
 *  and where its shards live. */
typedef struct usearch_matrix_init_ram_t {

    /** Output for the new index. */
    usearch_matrix_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The kind comparing two matrices - @c usearch_matrix_metric_hausdorff_k for point
     *  sets in one frame, @c usearch_matrix_metric_gromov_wasserstein_k for conformers. */
    usearch_matrix_metric_t metric;

    /** The scalar type rows are stored & quantized into. */
    usearch_scalar_t quantization;

    /** The frozen row width shared by every entry - 3 for a conformer, 6 for a pose. */
    size_t columns;

    /** The frozen row count, when every entry has one - a fixed-length backbone, a
     *  landmark set, a conformer pool in one atom order. Zero is the default and admits
     *  ragged heights; a non-zero value rejects insertions of any other height and is
     *  what lets row @c i of one entry pair with row @c i of another, which the
     *  superposition kinds need and the rest ignore. */
    size_t rows;

    /** The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** Upper bound on entries sharing one key; zero derives to one - unique keys. */
    size_t values_per_key;

    /** The memory domains to place one shard on apiece. */
    usearch_ram_options_t ram;
} usearch_matrix_init_ram_t;

/** Creates an empty, mutable matrix index in host memory. */
USEARCH_EXPORT void usearch_matrix_init_ram(usearch_matrix_init_ram_t* arguments);

/** The arguments of @c usearch_matrix_init_vram, which match @c usearch_matrix_init_ram_t but
 *  for the placement. */
typedef struct usearch_matrix_init_vram_t {

    /** Output for the new index. */
    usearch_matrix_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The kind comparing two matrices. */
    usearch_matrix_metric_t metric;

    /** The scalar type rows are stored & quantized into. */
    usearch_scalar_t quantization;

    /** The frozen row width shared by every entry. */
    size_t columns;

    /** The frozen row count, when every entry has one; zero admits ragged heights. */
    size_t rows;

    /** The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** Upper bound on entries sharing one key; zero derives to one - unique keys. */
    size_t values_per_key;

    /** The devices to place one shard on apiece. */
    usearch_vram_options_t vram;
} usearch_matrix_init_vram_t;

/** Creates an empty, mutable matrix index in device memory. */
USEARCH_EXPORT void usearch_matrix_init_vram(usearch_matrix_init_vram_t* arguments);

/** The arguments of @c usearch_matrix_load_ram. */
typedef struct usearch_matrix_load_ram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_matrix_index_t* index;

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
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;
} usearch_matrix_load_ram_t;

/**
 *  @brief Copies a checkpoint into a new, mutable index in host memory, its geometry read from
 *      the checkpoint's own headers.
 *
 *  Where the parts end up is the call's own memory domains, never the manifest's. A part
 *  carries its own length in its header, so a second load skips whatever already arrived
 *  whole and cancelling costs the part being read rather than everything behind it.
 */
USEARCH_EXPORT void usearch_matrix_load_ram(usearch_matrix_load_ram_t* arguments);

/** The arguments of @c usearch_matrix_load_vram. */
typedef struct usearch_matrix_load_vram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_matrix_index_t* index;

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
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;
} usearch_matrix_load_vram_t;

/** Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
 *  @c usearch_matrix_load_ram. */
USEARCH_EXPORT void usearch_matrix_load_vram(usearch_matrix_load_vram_t* arguments);

/** The arguments of @c usearch_matrix_view_ram. */
typedef struct usearch_matrix_view_ram_t {

    /** Output for the new, read-only index. */
    usearch_matrix_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;
} usearch_matrix_view_ram_t;

/** Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
 *  verbs report @c usearch_status_read_only_k. */
USEARCH_EXPORT void usearch_matrix_view_ram(usearch_matrix_view_ram_t* arguments);

/** Frees the resources associated with the index; @c NULL is safe, and it cannot fail, so takes
 *  no outcome.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_matrix_free(usearch_matrix_index_t index);

/** The arguments of @c usearch_matrix_reserve. */
typedef struct usearch_matrix_reserve_t {

    /** The handle to be resized. */
    usearch_matrix_index_t index;

    /** The executor the growth runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The desired total entry capacity, including the current size. */
    size_t entries;

    /** Expected total row volume across all entries; zero grows lazily. */
    size_t rows;
} usearch_matrix_reserve_t;

/** Reserves capacity ahead of insertion. */
USEARCH_EXPORT void usearch_matrix_reserve(usearch_matrix_reserve_t* arguments);

/** The arguments of @c usearch_matrix_clear. */
typedef struct usearch_matrix_clear_t {

    /** The handle to be cleared. */
    usearch_matrix_index_t index;

    /** The executor the erasure runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;
} usearch_matrix_clear_t;

/** Erases all entries, keeping the reserved capacity and frozen geometry. */
USEARCH_EXPORT void usearch_matrix_clear(usearch_matrix_clear_t* arguments);

/** Traversal telemetry, shared by insertion and search - a pure output, fully overwritten by the
 *  call it is passed to. Passing @c NULL skips reporting. */
typedef struct usearch_matrix_traversal_stats_t {

    /** Graph members visited while resolving the batch. */
    size_t visited_members;

    /** Distances computed while resolving the batch, diversity re-selections included. */
    size_t computed_distances;

    /** Starvation restarts - re-seedings after the frontier drained prematurely,
     *  and a graph-connectivity health signal rather than a cost counter. */
    size_t restarts;
} usearch_matrix_traversal_stats_t;

/** The arguments of @c usearch_matrix_insert. */
typedef struct usearch_matrix_insert_t {

    /** The handle to be populated. */
    usearch_matrix_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Matrices to admit - the batch axis of @c keys and @c matrices. */
    size_t count;

    /** The identities to admit the matrices under, one per entry and not necessarily unique. */
    usearch_keys_view_t keys;

    /** The matrices, as rows end to end around @c count + 1 row offsets. @ref matrix_index_tape */
    usearch_tape_view_t matrices;

    /** The @b optional candidate-pool breadth the build walk explores; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries admitted and the batch length. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_matrix_traversal_stats_t* stats;
} usearch_matrix_insert_t;

/**
 *  @brief Adds a tape of matrices under their keys - the native batch operation; a single
 *      matrix is a tape of one. Row counts may differ freely between entries. Overlaps
 *      safely with concurrent searches on the same handle. @ref matrix_index_concurrency
 *  @retval usearch_status_read_only_k The handle is a view.
 *  @retval usearch_status_overflow_k A key already holds @c values_per_key entries.
 *
 *  An insertion is all-or-nothing per call. Every fallible step runs before the first entry
 *  identity is claimed, so a rejected call publishes nothing - a cancelled one included, which
 *  reports @c usearch_status_cancelled_k - and on a RAM handle the entries of an accepted one
 *  become visible at once.
 *
 *  A key already holding @c values_per_key entries rejects the whole call, since a smaller retry
 *  cannot help.
 */
USEARCH_EXPORT void usearch_matrix_insert(usearch_matrix_insert_t* arguments);

/** The arguments of @c usearch_matrix_refine. */
typedef struct usearch_matrix_refine_t {

    /** The handle to be refined. */
    usearch_matrix_index_t index;

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
    usearch_matrix_traversal_stats_t* stats;
} usearch_matrix_refine_t;

/**
 *  @brief Re-selects the neighbors of every stored entry against the finished graph, improving
 *      edges that were chosen when the graph around them was still incomplete.
 *
 *  This is a build step rather than a repair. A cheap insertion pass followed by one refinement
 *  reaches a better graph, in less total time, than a single expensive insertion pass - so a wide
 *  @c expansion_factor belongs here rather than in @c usearch_matrix_insert. It pays most in this
 *  modality, where one comparison scores a whole matrix against another.
 *
 *  Cancelling is safe, which is what makes a whole-graph sweep affordable. Re-selecting an entry's
 *  edges is idempotent and claims no identity, so a stopped pass leaves the graph correct and
 *  merely less improved - nothing unwinds, and no @c usearch_status_cancelled_k is reported. The
 *  sweep resumes where it stopped, so calling under a time budget converges across calls.
 *
 *  Searches may run throughout. A viewed handle refuses with @c usearch_status_read_only_k, like
 *  every other mutating verb.
 */
USEARCH_EXPORT void usearch_matrix_refine(usearch_matrix_refine_t* arguments);

/** The arguments of @c usearch_matrix_search. */
typedef struct usearch_matrix_search_t {

    /** The handle to be queried. */
    usearch_matrix_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Query matrices to resolve; the outputs have @c count rows of @c wanted entries. */
    size_t count;

    /** The query matrices, as rows end to end around @c count + 1 row offsets. */
    usearch_tape_view_t queries;

    /** Output for the run, inside @c arena. */
    usearch_matches_view_t* matches;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /** Columns per row, and the upper bound on matches - the "k" in "kANN". */
    size_t wanted;

    /** The @b optional candidate-pool breadth the walk explores; zero derives. */
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
    usearch_matrix_traversal_stats_t* stats;
} usearch_matrix_search_t;

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a tape of query matrices - the native
 *      batch operation, parallelized internally; one query is a tape of one. Query row counts are
 *      unconstrained and need not match any indexed entry.
 *
 *  Never writes to the graph, so any number may overlap with concurrent insertions.
 *  @ref matrix_index_concurrency
 *
 *  A @c predicate narrows the matches to the entries it admits. Non-matching entries still route
 *  the traversal, so recall does not collapse at low selectivity.
 *
 *  A cancelled search reports @c usearch_status_cancelled_k and leaves @c matches @c NULL.
 */
USEARCH_EXPORT void usearch_matrix_search(usearch_matrix_search_t* arguments);

/** What one key holds, enough to size a buffer before retrieving any of it. */
typedef struct usearch_matrix_key_stats_t {

    /** Entries sharing the key; keys need not be unique. */
    size_t entries;

    /** Rows summed across those entries. */
    size_t total_rows;

    /** Rows of the tallest single entry. */
    size_t largest_rows;
} usearch_matrix_key_stats_t;

/** The arguments of @c usearch_matrix_count. */
typedef struct usearch_matrix_count_t {

    /** The handle to be queried. */
    usearch_matrix_index_t index;

    /** The executor the lookup runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are counted. */
    usearch_key_t key;

    /** Output for what @c key holds; zero @c entries means the index holds none. */
    usearch_matrix_key_stats_t* key_stats;
} usearch_matrix_count_t;

/** Counts the entries stored under one key, with the extent of their rows. */
USEARCH_EXPORT void usearch_matrix_count(usearch_matrix_count_t* arguments);

/** The arguments of @c usearch_matrix_get. */
typedef struct usearch_matrix_get_t {

    /** The handle to be queried. */
    usearch_matrix_index_t index;

    /** The executor whose stream or thread copies the rows; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are exported. */
    usearch_key_t key;

    /** Output for the tape inside @c arena: every entry's rows end to end, packed at the frozen
     *  width in @c outputs_scalar, around @c written + 1 row offsets. */
    usearch_tape_view_t* matrices;

    /** The @b optional output for the entries written; zero when the key is absent. */
    size_t* written;

    /** Memory for @c matrices, rewound by the call. */
    usearch_arena_t arena;

    /** The scalar type to export as, converted from storage where it differs. */
    usearch_scalar_t outputs_scalar;
} usearch_matrix_get_t;

/**
 *  @brief Retrieves every matrix stored under one key as an Apache Arrow tape, in insertion order -
 *      one call, however many and however tall they turn out to be.
 *
 *  An absent key leaves the tape's pointers @c NULL.
 */
USEARCH_EXPORT void usearch_matrix_get(usearch_matrix_get_t* arguments);

/** The arguments of @c usearch_matrix_merge. */
typedef struct usearch_matrix_merge_t {

    /** The handle absorbing the donors. */
    usearch_matrix_index_t index;

    /** The executor the merge fans out over, one donor per thread. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Donor handles to absorb. */
    size_t count;

    /** Array of @c count handles, every one of them consumed and nulled as it goes. */
    usearch_matrix_index_t* donors;

    /** The @b optional handler, called with donors absorbed and their count. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;
} usearch_matrix_merge_t;

/**
 *  @brief Absorbs every handle in @c donors into @c index - cheaper than re-inserting their
 *      matrices one by one, wiring cross-links by walks across from landmarks. Donor handles are
 *      consumed and must not be used afterwards, not even for @c usearch_matrix_free.
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
USEARCH_EXPORT void usearch_matrix_merge(usearch_matrix_merge_t* arguments);

/** The arguments of @c usearch_matrix_cursor_init. */
typedef struct usearch_matrix_cursor_init_t {

    /** Output for the new cursor; @c NULL on failure. */
    usearch_matrix_cursor_t* cursor;

    /** The handle to be scanned. */
    usearch_matrix_index_t index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;
} usearch_matrix_cursor_init_t;

/** Opens a snapshot cursor over the current contents; see @c usearch_matrix_cursor_t. */
USEARCH_EXPORT void usearch_matrix_cursor_init(usearch_matrix_cursor_init_t* arguments);

/** The arguments of @c usearch_matrix_cursor_next. */
typedef struct usearch_matrix_cursor_next_t {

    /** The handle the cursor was opened on. */
    usearch_matrix_index_t index;

    /** The executor the pull runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The cursor to advance. */
    usearch_matrix_cursor_t cursor;

    /** Output for the identities, inside @c arena. */
    usearch_keys_view_t* keys;

    /** The @b optional output for the entries as a tape inside @c arena, rows packed at the frozen
     *  width around @c written + 1 row offsets; @c NULL skips them. */
    usearch_tape_view_t* matrices;

    /** Output for the entries written; zero when the snapshot is exhausted. */
    size_t* written;

    /** Memory for @c keys and @c matrices, rewound by the call. */
    usearch_arena_t arena;

    /** The upper bound on entries this pull writes. */
    size_t wanted;

    /** The scalar type to export @c matrices as, converted from storage where it differs. */
    usearch_scalar_t outputs_scalar;
} usearch_matrix_cursor_next_t;

/**
 *  @brief Copies the next batch of entries out of the cursor into @c arena.
 *  @retval usearch_status_invalid_argument_k @c index is not the handle the cursor was opened on.
 *
 *  The rows are copied rather than borrowed from the index, so the batch stays valid until the
 *  next call given the same arena, however the cursor moves meanwhile.
 */
USEARCH_EXPORT void usearch_matrix_cursor_next(usearch_matrix_cursor_next_t* arguments);

/** Releases the cursor; @c NULL is safe.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_matrix_cursor_free(usearch_matrix_cursor_t cursor);

/** The arguments of @c usearch_matrix_save. */
typedef struct usearch_matrix_save_t {

    /** The handle to be written. */
    usearch_matrix_index_t index;

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
} usearch_matrix_save_t;

/**
 *  @brief Saves a mutable index under @c path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  A cancelled save leaves the parts it already wrote and no manifest referencing them, so the
 *  checkpoint reads as absent rather than half-valid. Those leftovers are the caller's to remove.
 */
USEARCH_EXPORT void usearch_matrix_save(usearch_matrix_save_t* arguments);

/** The geometry and metric a checkpoint froze - everything a reader learns without opening it. */
typedef struct usearch_matrix_metadata_t {

    /** The kind the checkpoint compares matrices with. */
    usearch_matrix_metric_t metric;

    /** The scalar type its rows are stored & quantized into. */
    usearch_scalar_t quantization;

    /** The row width shared by every entry. */
    size_t columns;

    /** The frozen row count, or zero for ragged heights. */
    size_t rows;

    /** Connections per node the graph was built with. */
    size_t connectivity;
} usearch_matrix_metadata_t;

/** The arguments of @c usearch_matrix_metadata_read. */
typedef struct usearch_matrix_metadata_read_t {

    /** The executor the read runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** Output for every field the checkpoint froze. */
    usearch_matrix_metadata_t* metadata;
} usearch_matrix_metadata_read_t;

/** Reads a checkpoint's geometry & metric without loading it. */
USEARCH_EXPORT void usearch_matrix_metadata_read(usearch_matrix_metadata_read_t* arguments);

/** A live index's state. Each counter is read once, so under concurrent insertions they may come
 *  from slightly different moments. */
typedef struct usearch_matrix_index_description_t {

    /** The frozen metric, quantization, columns, rows and connectivity, as a checkpoint reports them. */
    usearch_matrix_metadata_t metadata;

    /** Stored entries. */
    size_t size;

    /** Entries hosted without growing again. */
    size_t capacity;

    /** Rows summed across every stored entry. */
    size_t total_rows;

    /** Bytes the handle holds, row storage included; traversal scratch is the executor's. */
    size_t memory_usage;

    /** Static codename of the kernels the index dispatched to on this machine. */
    char const* capabilities;
} usearch_matrix_index_description_t;

/** The arguments of @c usearch_matrix_describe. */
typedef struct usearch_matrix_describe_t {

    /** The handle to be described. */
    usearch_matrix_index_t index;

    /** The executor the report runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output, filled whole. */
    usearch_matrix_index_description_t* description;
} usearch_matrix_describe_t;

/** Reports a live index's frozen metadata and current counters in one call. */
USEARCH_EXPORT void usearch_matrix_describe(usearch_matrix_describe_t* arguments);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_MATRIX_INDEX_H
