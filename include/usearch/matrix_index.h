/**
 *  @brief The matrix-modality C interface: similarity search over rank-2 tensors - sets of
 *      equidimensional rows whose count varies per entry - under late-interaction and
 *      transport scoring.
 *
 *  @author Ash Vardanian
 *  @file include/usearch/matrix_index.h
 *
 *  One entry is a matrix: @c rows varies, @c columns is frozen index-wide. A ColBERT-style
 *  document is a few hundred token embeddings of 128 columns; a molecular conformer is a
 *  few dozen atom positions of 3 columns. Neither the two sides of a comparison nor two
 *  entries of the index need agree on their row count - a 40-token query scores against a
 *  512-token document, and a 20-atom molecule scores against a 60-atom molecule, without
 *  padding, truncation, or resampling anywhere in the interface.
 *
 *  Rows are unordered. Where ordering carries meaning the sequence modality owns the
 *  object, and where every entry has exactly one row the dense modality does.
 *
 *  @section matrix_index_tape Batch Shape
 *
 *  Variable row counts cannot ride a strided matrix, so every batch input is a @b tape:
 *  one array of base pointers and one of row counts, index-aligned. Entry @c i holds
 *  `row_counts[i] * columns` scalars at `matrices[i]`, row-major and tightly packed. Keys
 *  need not be unique - three conformers of one molecule share its key, and the modality
 *  reports how many entries a key holds.
 *
 *  @section matrix_index_scoring Scoring Kinds
 *
 *  The first two compare rows to rows: each row of one side finds its best partner on the
 *  other, and they differ only in how those bests are aggregated. Chamfer averages them in
 *  both directions, Hausdorff takes the worst. Both build the same matrix of row pairings
 *  that ColBERT's MaxSim does, at the same cost, and both are symmetric where MaxSim is
 *  not - so late-interaction retrieval runs on Hausdorff.
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
 *  order. A VRAM handle is driven by one host thread through streams. An NVMe handle admits
 *  arbitrarily many concurrent searches, having nothing to mutate. Serialization is the one
 *  exception: saving demands a mutation-free window, though concurrent readers keep
 *  searching throughout. Per-call outputs - counts and the error pointer - belong to their
 *  call and must not be shared between in-flight calls.
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
 *  @see usearch_matrix_init
 *  @see usearch_matrix_add
 *  @see usearch_matrix_search
 */
USEARCH_EXPORT typedef void* usearch_matrix_index_t;

/**
 *  @brief A consistent snapshot cursor over the index contents. Captures the publication
 *      watermark at creation, so concurrent inserts never appear mid-iteration, and the
 *      walk is a sequential entry scan - export, migration, and visualization feed off it.
 */
USEARCH_EXPORT typedef void* usearch_matrix_cursor_t;

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
USEARCH_EXPORT typedef enum usearch_matrix_metric_t {
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
     *  The kind ColBERT-style late interaction runs on. It pairs rows exactly as MaxSim does
     *  and costs the same, differing only in taking the worst pairing rather than their sum,
     *  which is what makes it symmetric and usable as an edge weight where MaxSim is neither.
     *  Unforgiving where Chamfer is lenient: one row whose nearest row in the other matrix
     *  is far away sets the entire score.
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

/**
 *  @brief Construction & loading options of a matrix index. The @c residency field tags
 *      which union member is read; zeroed optional fields derive from the geometry.
 */
USEARCH_EXPORT typedef struct usearch_matrix_options_t {
    /** @brief The kind comparing two matrices - @c usearch_matrix_metric_hausdorff_k for
     *      ColBERT token matrices, @c usearch_matrix_metric_gromov_wasserstein_k for
     *      molecular conformers. */
    usearch_matrix_metric_t metric;
    /** @brief The scalar type rows are stored & quantized into. */
    usearch_scalar_t quantization;
    /** @brief The frozen row width shared by every entry - 128 for a token embedding, 3 for
     *      a conformer; zero defers the geometry to a later @c usearch_matrix_load. */
    size_t columns;
    /** @brief The frozen row count, when every entry has one - a fixed-length backbone, a
     *      landmark set, a conformer pool in one atom order. Zero is the default and admits
     *      ragged heights; a non-zero value rejects insertions of any other height and is
     *      what lets row @c i of one entry pair with row @c i of another, which the
     *      superposition kinds need and the rest ignore. */
    size_t rows;
    /** @brief The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;
    /** @brief Selects the active member of the union below, and the meaning of "load". */
    usearch_residency_t residency;
    union {
        usearch_ram_options_t ram;
        usearch_vram_options_t vram;
        usearch_nvme_options_t nvme;
    };
} usearch_matrix_options_t;

/**
 *  @brief Initializes a new matrix index for the residency the options select.
 *  @param[in] options Pointer to the populated options structure.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 *  @return A handle to the initialized index, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_matrix_index_t usearch_matrix_init( //
    usearch_matrix_options_t const* options, usearch_error_t* error);

/**
 *  @brief Frees the resources associated with the index; cannot fail, so takes no error.
 */
USEARCH_EXPORT void usearch_matrix_free(usearch_matrix_index_t index);

/**
 *  @brief Reserves capacity ahead of insertion.
 *  @param[in] entries The desired total entry capacity, including the current size.
 *  @param[in] rows Expected total row volume across all entries; zero grows lazily.
 */
USEARCH_EXPORT void usearch_matrix_reserve( //
    usearch_matrix_index_t index, size_t entries, size_t rows, usearch_error_t* error);

/**
 *  @brief Erases all entries, keeping the reserved capacity and frozen geometry.
 */
USEARCH_EXPORT void usearch_matrix_clear(usearch_matrix_index_t index, usearch_error_t* error);

/** @brief Reports the number of stored entries. */
USEARCH_EXPORT size_t usearch_matrix_size(usearch_matrix_index_t index, usearch_error_t* error);
/** @brief Reports the reserved capacity in entries. */
USEARCH_EXPORT size_t usearch_matrix_capacity(usearch_matrix_index_t index, usearch_error_t* error);
/** @brief Reports the frozen row width shared by every entry. */
USEARCH_EXPORT size_t usearch_matrix_columns(usearch_matrix_index_t index, usearch_error_t* error);
/** @brief Reports the total number of stored rows across all entries. */
USEARCH_EXPORT size_t usearch_matrix_rows(usearch_matrix_index_t index, usearch_error_t* error);
/** @brief Reports the frozen connections-per-node bound of the graph. */
USEARCH_EXPORT size_t usearch_matrix_connectivity(usearch_matrix_index_t index, usearch_error_t* error);
/** @brief Reports the memory usage of the index in bytes, row arenas included. */
USEARCH_EXPORT size_t usearch_matrix_memory_usage(usearch_matrix_index_t index, usearch_error_t* error);
/** @brief Reports the capability codename the index dispatched to on this machine. */
USEARCH_EXPORT char const* usearch_matrix_capabilities(usearch_matrix_index_t index, usearch_error_t* error);

/**
 *  @brief Optional per-call settings, shared by insertion and both search forms - a pure
 *      input, never written. Passing @c NULL derives every setting.
 */
USEARCH_EXPORT typedef struct usearch_matrix_expansion_config_t {
    /** @brief The @b optional candidate-pool breadth; zero derives. */
    size_t expansion_factor;
} usearch_matrix_expansion_config_t;

/**
 *  @brief Traversal telemetry, shared by insertion and both search forms - a pure output,
 *      fully overwritten by the call it is passed to. Passing @c NULL skips reporting.
 */
USEARCH_EXPORT typedef struct usearch_matrix_traversal_stats_t {
    /** @brief Graph members visited while resolving the batch. */
    size_t visited_members;
    /** @brief Distances computed while resolving the batch, diversity re-selections included. */
    size_t computed_distances;
    /** @brief Starvation restarts - re-seedings after the frontier drained prematurely,
     *      and a graph-connectivity health signal rather than a cost counter. */
    size_t restarts;
} usearch_matrix_traversal_stats_t;

/**
 *  @brief Adds a tape of matrices under their keys - the native batch operation; a single
 *      matrix is a tape of one. Row counts may differ freely between entries. Overlaps
 *      safely with concurrent searches on the same handle. @ref matrix_index_concurrency
 *
 *  @param[inout] index The handle to be populated.
 *  @param[in] keys Array of @p count keys, one per matrix; keys need not be unique.
 *  @param[in] matrices Array of @p count base pointers. Entry @c i holds
 *      `row_counts[i] * columns` scalars, row-major and tightly packed.
 *  @param[in] row_counts Array of @p count row counts, index-aligned with @p matrices.
 *  @param[in] count Number of matrices in the tape.
 *  @param[in] scalar_kind The scalar type of the incoming rows, converted on ingest when it
 *      differs from the stored quantization.
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 *  @return Number of successfully inserted matrices - equal to @p count unless an error
 *      stopped the batch early.
 */
USEARCH_EXPORT size_t usearch_matrix_add(                    //
    usearch_matrix_index_t index, usearch_key_t const* keys, //
    void const* const* matrices, size_t const* row_counts,   //
    size_t count, usearch_scalar_t scalar_kind,              //
    usearch_matrix_expansion_config_t const* config,         //
    usearch_matrix_traversal_stats_t* stats, usearch_error_t* error);

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a tape of query matrices -
 *      the native batch operation, parallelized internally; one query is a tape of one.
 *      Query row counts are unconstrained and need not match any indexed entry. Never
 *      writes to the graph, so any number may overlap with concurrent insertions.
 *  @ref matrix_index_concurrency
 *
 *  @param[in] index The handle to be queried.
 *  @param[in] queries Array of @p queries_count query base pointers.
 *  @param[in] row_counts Array of @p queries_count query row counts.
 *  @param[in] queries_count Number of query matrices in the tape.
 *  @param[in] scalar_kind The scalar type of the query rows.
 *  @param[in] wanted Upper bound on matches per query - the "k" in "kANN".
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] keys Output matrix of `queries_count × wanted` keys; rows contiguous.
 *  @param[in] keys_stride Number of bytes between starts of consecutive rows in @p keys.
 *  @param[out] distances Output matrix of `queries_count × wanted` scores; rows contiguous.
 *  @param[in] distances_stride Number of bytes between starts of consecutive @p distances rows.
 *  @param[out] counts Output array of @p queries_count per-query match counts.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 */
USEARCH_EXPORT void usearch_matrix_search(                  //
    usearch_matrix_index_t index,                           //
    void const* const* queries, size_t const* row_counts,   //
    size_t queries_count, usearch_scalar_t scalar_kind,     //
    size_t wanted,                                          //
    usearch_matrix_expansion_config_t const* config,        //
    usearch_matrix_traversal_stats_t* stats,                //
    usearch_key_t* keys, size_t keys_stride,                //
    usearch_distance_t* distances, size_t distances_stride, //
    size_t* counts, usearch_error_t* error);

/**
 *  @brief The filtered sibling of @c usearch_matrix_search: identical tape shape, admitting
 *      only entries the @p predicate approves. Non-matching entries still route the
 *      traversal, so recall does not collapse at low selectivity.
 *
 *  @param[in] predicate Per-key admission callback; returns non-zero to admit.
 *  @param[in] predicate_state Opaque pointer forwarded into every @p predicate call.
 */
USEARCH_EXPORT void usearch_matrix_search_if(               //
    usearch_matrix_index_t index,                           //
    void const* const* queries, size_t const* row_counts,   //
    size_t queries_count, usearch_scalar_t scalar_kind,     //
    size_t wanted,                                          //
    usearch_matrix_expansion_config_t const* config,        //
    usearch_matrix_traversal_stats_t* stats,                //
    usearch_predicate_t predicate, void* predicate_state,   //
    usearch_key_t* keys, size_t keys_stride,                //
    usearch_distance_t* distances, size_t distances_stride, //
    size_t* counts, usearch_error_t* error);

/** @brief What one key holds, enough to size a buffer before retrieving any of it. */
USEARCH_EXPORT typedef struct usearch_matrix_key_stats_t {
    /** @brief Entries sharing the key; keys need not be unique. */
    size_t entries;
    /** @brief Rows summed across those entries. */
    size_t total_rows;
    /** @brief Rows of the tallest single entry - the buffer one @c get needs. */
    size_t largest_rows;
} usearch_matrix_key_stats_t;

/**
 *  @brief Counts the entries stored under @p key; zero means the index holds none.
 *
 *  Pairing this with @c usearch_matrix_get is not atomic: an insertion between the two
 *  calls leaves the count stale. That is harmless, because entries are only ever added, so
 *  a stale count under-reports and @c usearch_matrix_get returns the true extent - a caller
 *  seeing more than it asked for allocates again and converges.
 *
 *  @param[out] stats The @b optional extent of the key's payload; @c NULL asks for the
 *      count alone.
 *  @return The number of entries under @p key, mirroring @c stats->entries.
 */
USEARCH_EXPORT size_t usearch_matrix_count(          //
    usearch_matrix_index_t index, usearch_key_t key, //
    usearch_matrix_key_stats_t* stats, usearch_error_t* error);

/**
 *  @brief Retrieves the matrices stored under @p key into a caller-owned buffer.
 *
 *  Entries are written at @p stride spacing in insertion order, so sizing the buffer takes
 *  @c usearch_matrix_count first: @c entries slots of @c largest_rows rows each.
 *
 *  @param[out] matrices Destination for up to @p wanted entries; @c NULL probes the count
 *      without copying.
 *  @param[in] wanted Upper bound on entries to write - the entry capacity of @p matrices.
 *  @param[in] stride Number of bytes between starts of consecutive entries in @p matrices.
 *  @param[out] row_counts The @b optional per-entry true row count, up to @p wanted of them;
 *      @c NULL skips it.
 *  @param[in] scalar_kind The scalar type to export as, converted from storage when different.
 *  @return The @b true number of entries under @p key, never the number written - a value
 *      above @p wanted means the copy was truncated, so allocate that many and call again.
 *      Zero when the key is absent.
 */
USEARCH_EXPORT size_t usearch_matrix_get(                             //
    usearch_matrix_index_t index, usearch_key_t key,                  //
    void* matrices, size_t wanted, size_t stride, size_t* row_counts, //
    usearch_scalar_t scalar_kind, usearch_error_t* error);

/**
 *  @brief Absorbs @p donor into @p index - cheaper than re-inserting its matrices one by
 *      one, wiring cross-links via landmark stitching. The donor handle is consumed and
 *      must not be used afterwards, not even for @c usearch_matrix_free. Readers of
 *      @p index may keep searching throughout: unpublished entries hide behind the watermark.
 */
USEARCH_EXPORT void usearch_matrix_merge( //
    usearch_matrix_index_t index, usearch_matrix_index_t donor, usearch_error_t* error);

/**
 *  @brief Opens a snapshot cursor over the current contents; see @c usearch_matrix_cursor_t.
 *  @return The cursor, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_matrix_cursor_t usearch_matrix_cursor(usearch_matrix_index_t index, usearch_error_t* error);

/**
 *  @brief Copies the next batch of entries out of the cursor.
 *  @param[out] keys Output array of up to @p wanted keys.
 *  @param[out] matrices Output array of up to @p wanted borrowed base pointers into the
 *      index's arena, valid until the cursor moves or closes; @c NULL skips row export.
 *  @param[out] row_counts Output array of up to @p wanted row counts.
 *  @param[in] wanted Upper bound on entries to write - the capacity of every output array.
 *  @return Number of entries written; zero when the snapshot is exhausted.
 */
USEARCH_EXPORT size_t usearch_matrix_cursor_next(        //
    usearch_matrix_cursor_t cursor, usearch_key_t* keys, //
    void const** matrices, size_t* row_counts, size_t wanted, usearch_error_t* error);

/** @brief Releases the cursor. */
USEARCH_EXPORT void usearch_matrix_cursor_free(usearch_matrix_cursor_t cursor);

/**
 *  @brief Reports the expected serialized size of the index in bytes.
 */
USEARCH_EXPORT size_t usearch_matrix_serialized_length(usearch_matrix_index_t index, usearch_error_t* error);

/**
 *  @brief Saves the index under @p path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 */
USEARCH_EXPORT void usearch_matrix_save(usearch_matrix_index_t index, usearch_string_t path, usearch_error_t* error);

/**
 *  @brief Loads a checkpoint into an initialized handle, with the handle's residency
 *      deciding the behavior: materialize into RAM, upload to the device, or open in place
 *      on disk - the read-only form, where mutating verbs fail. A handle whose geometry was
 *      left zeroed freezes it here, from the checkpoint's own headers.
 */
USEARCH_EXPORT void usearch_matrix_load(usearch_matrix_index_t index, usearch_string_t path, usearch_error_t* error);

/**
 *  @brief Reads a checkpoint's geometry & metric metadata into @p options without loading it.
 */
USEARCH_EXPORT void usearch_matrix_metadata( //
    usearch_string_t path, usearch_matrix_options_t* options, usearch_error_t* error);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_MATRIX_INDEX_H
