/**
 *  @file c/usearch/sequence_index.h
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The sequence-modality C interface: similarity search over variable-length sequences -
 *      textual documents, DNA, RNA, and protein strings - under Levenshtein, window overlap,
 *      Needleman-Wunsch, Smith-Waterman, and Gotoh-affine scoring.
 *
 *  Variable-length payloads cannot ride a run of equal rows, so every batch input is a @b tape:
 *  the bytes of every sequence end to end, and @c count + 1 Apache Arrow offsets around them - the
 *  sequence-modality twin of the dense header's rows. One constructor serves every metric family:
 *  uniform edit distances need no parameters, window overlap takes its window widths, weighted
 *  alignments take a substitution matrix and gap costs. Handles are created by a lifecycle verb
 *  naming both an action and a residency:
 *
 *      Action                           RAM          VRAM         NVMe
 *      init: start empty, mutable       init_ram     init_vram    none
 *      load: copy a checkpoint in       load_ram     load_vram    none
 *      view: serve one in place         view_ram     none         view_nvme
 *      save: write a checkpoint         save         save         none
 *
 *  Each is spelled @c usearch_sequence_ and the suffix above. A VRAM handle serves byte and UTF-8
 *  Levenshtein, on one GPU, through the insertion, search and description verbs, refusing the rest
 *  with @c usearch_status_unsupported_k, and a search there takes no @c predicate.
 *
 *  A viewed handle is read-only, refusing every mutating verb with @c usearch_status_read_only_k.
 *  An NVMe handle answers every read verb as a RAM handle does, its rows read through the
 *  executor's engines, and saving answers @c usearch_status_unsupported_k, since a view already is
 *  its checkpoint.
 *
 *  @section sequence_index_concurrency Concurrency & Thread Safety
 *
 *  A handle may be shared across threads. On a mutable handle, insertions and searches overlap
 *  freely: searches read only entries published behind the visibility watermark and never
 *  write to the graph, while insertions synchronize through per-node locks and publish in
 *  order. A viewed handle admits arbitrarily many concurrent searches, having nothing to
 *  mutate. Serialization is the one exception: saving demands a mutation-free window,
 *  though concurrent readers keep searching throughout. Per-call outputs - counts and the
 *  outcome - belong to their call and must not be shared between calls that are not final.
 *
 *  An insertion is all-or-nothing per call, and its entries become visible at once, so a
 *  concurrent search sees every entry of the call or none of them. Their keys land right after,
 *  so a key lookup only finds entries a search already sees.
 *
 *  Entries are permanent. No verb removes one, so an index only ever grows until it is
 *  cleared wholesale, and a key's entries accumulate up to the @c values_per_key bound its
 *  creation froze.
 *
 *  The executor, not the handle, is what orders: a handle admits any number of concurrent
 *  callers, while calls on one executor run in the order they were submitted.
 */
#ifndef UNUM_USEARCH_SEQUENCE_INDEX_H
#define UNUM_USEARCH_SEQUENCE_INDEX_H

#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Similarity Search Index over variable-length sequences.
 *
 *  @see usearch_sequence_init_ram
 *  @see usearch_sequence_insert
 *  @see usearch_sequence_search
 */
typedef struct usearch_sequence_index_handle_t* usearch_sequence_index_t;

/** A consistent snapshot cursor over the graph. Captures the publication watermark at creation,
 *  so concurrent inserts never appear mid-iteration, and yields each entry's key with the keys it
 *  links to - every entry as stored, or breadth-first from start keys - which export, migration,
 *  and visualization feed off. Each pull names the index again; an overlapping pull reports
 *  @c usearch_status_contended_k. */
typedef struct usearch_sequence_cursor_handle_t* usearch_sequence_cursor_t;

/**
 *  @brief One refinement in flight, optionally absorbing donors or splitting into receivers first.
 *
 *  It holds everything the pass needs between calls, so a cancelled
 *  @c usearch_sequence_refinement_next continues where it stopped, and nothing about it stays in
 *  the index once it finishes. The index admits no insertion and no save while the handle lives;
 *  searches proceed as during @c usearch_sequence_refine. Each call names the index and the
 *  executor it was opened with.
 */
typedef struct usearch_sequence_refinement_handle_t* usearch_sequence_refinement_t;

/**
 *  @brief The sequence scoring kinds supported out of the box.
 *
 *  The alignment kinds form a grid: global or local, against linear or affine gaps. All of
 *  them fill the same @b [|a|+1,|b|+1] table of cell @c D[i,j], the best score for the two
 *  prefixes ending at @c i and @c j, so every kind here is quadratic in time. Only the live width
 *  differs: the linear kinds carry one row, the affine kinds three.
 *
 *  Only the global half can weigh the index's own edges. A local score is blind to everything
 *  outside its matched window, so every superstring of a sequence scores as that sequence
 *  does and an index built on one would collapse distinct entries onto a single point. Each
 *  local kind is therefore served from an index built under its own global counterpart, which
 *  shares its costs and bounds it - picked by the library and never named by the caller.
 *
 *  The global kinds minimize a cost, writing @c c for the substitution cost and @c g, @c o,
 *  @c e for the flat, opening and extending gap costs. The local kinds maximize a similarity,
 *  writing @c w for the substitution score, so a matrix admissible for one is not admissible
 *  for the other with its signs unchanged.
 */
typedef enum usearch_sequence_metric_t {
    usearch_sequence_metric_unknown_k = 0,

    /**
     *  @brief Levenshtein edit distance over raw bytes.
     *
     *  The number of single-byte insertions, deletions and substitutions turning one sequence
     *  into the other, every edit costing one. A true metric, and the cheapest kind here.
     *
     *  D[i,j] = min( D[i-1,j] + 1, D[i,j-1] + 1, D[i-1,j-1] + (a[i] ≠ b[j]) )
     *
     *  @code{.py}
     *  row = list(range(len(b) + 1))                # one [|b|+1] row is the whole working set
     *  for i, x in enumerate(a, 1):
     *      row, prev = [i] + [0] * len(b), row
     *      for j, y in enumerate(b, 1):
     *          row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (x != y))
     *  levenshtein = row[-1]
     *  @endcode
     */
    usearch_sequence_metric_levenshtein_bytes_k = 1,

    /**
     *  @brief Levenshtein edit distance over UTF-8 code points.
     *
     *  The same recurrence, run over decoded code points rather than bytes, so one accented
     *  character costs one edit instead of the two or three its encoding occupies.
     */
    usearch_sequence_metric_levenshtein_utf8_k = 2,

    /**
     *  @brief Global alignment with linear gap costs - Needleman-Wunsch.
     *
     *  Levenshtein with the unit costs replaced by a substitution matrix and a flat gap cost,
     *  so a mismatch between similar symbols can be cheaper than one between dissimilar ones.
     *  Both sequences are aligned end to end, and every unaligned symbol is charged.
     *
     *  D[i,j] = min( D[i-1,j] + g, D[i,j-1] + g, D[i-1,j-1] + c(a[i], b[j]) )
     *
     *  @code{.py}
     *  row = [j * g for j in range(len(b) + 1)]
     *  for i, x in enumerate(a, 1):
     *      row, prev = [i * g] + [0] * len(b), row
     *      for j, y in enumerate(b, 1):
     *          row[j] = min(prev[j] + g, row[j - 1] + g, prev[j - 1] + c[x][y])
     *  needleman_wunsch = row[-1]
     *  @endcode
     */
    usearch_sequence_metric_needleman_wunsch_k = 3,

    /**
     *  @brief Local alignment with affine gap costs - Smith-Waterman-Gotoh. Ranks only.
     *
     *  The affine recurrence under a zero floor, reporting the best cell anywhere in the
     *  table rather than the corner. The floor is the blindness: a prefix or suffix that
     *  aligns badly is dropped at no cost instead of being charged.
     *
     *  S[i,j] = max( 0, X[i,j], Y[i,j], S[i-1,j-1] + w(a[i], b[j]) ),  score = max_{i,j} S[i,j]
     *
     *  @code{.py}
     *  # the needleman_wunsch_gotoh loop with max for min, w for c, and a 0 floor on S[i][j],
     *  # returning max(S) over every cell rather than the last one
     *  @endcode
     */
    usearch_sequence_metric_smith_waterman_gotoh_k = 4,

    /**
     *  @brief Global alignment with affine gap costs - Needleman-Wunsch-Gotoh.
     *
     *  Charges a gap run @c o for opening and @c e for each symbol after, so one long gap is
     *  cheaper than several short ones - the realistic model for an indel. Needs three live tracks:
     *  the aligned cell and one per gap direction. Both gap tracks reopen from the
     *  combined cell rather than from the aligned one, which is what lets a deletion become
     *  an insertion directly and keeps the kind a metric.
     *
     *  X[i,j] = min( D[i-1,j] + o, X[i-1,j] + e )
     *  Y[i,j] = min( D[i,j-1] + o, Y[i,j-1] + e )
     *  D[i,j] = min( D[i-1,j-1] + c(a[i], b[j]), X[i,j], Y[i,j] )
     *
     *  @code{.py}
     *  D = [0 if j == 0 else o + (j - 1) * e for j in range(len(b) + 1)]
     *  X = [inf] * (len(b) + 1)                     # gap in b, carried down the column
     *  for i, x in enumerate(a, 1):
     *      D, prev, Y = [o + (i - 1) * e] + [0] * len(b), D, inf
     *      for j, y in enumerate(b, 1):
     *          X[j] = min(prev[j] + o, X[j] + e)
     *          Y = min(D[j - 1] + o, Y + e)         # gap in a, carried along the row
     *          D[j] = min(prev[j - 1] + c[x][y], X[j], Y)
     *  gotoh = D[-1]
     *  @endcode
     */
    usearch_sequence_metric_needleman_wunsch_gotoh_k = 5,

    /**
     *  @brief Local alignment with linear gap costs - Smith-Waterman. Ranks only.
     *
     *  Needleman-Wunsch under a zero floor, reporting the best cell anywhere in the table.
     *  Finds the single best-matching window and charges nothing for what surrounds it.
     *
     *  S[i,j] = max( 0, S[i-1,j] − g, S[i,j-1] − g, S[i-1,j-1] + w(a[i], b[j]) )
     *
     *  @code{.py}
     *  best, row = 0, [0] * (len(b) + 1)
     *  for i, x in enumerate(a, 1):
     *      row, prev = [0] * (len(b) + 1), row
     *      for j, y in enumerate(b, 1):
     *          row[j] = max(0, prev[j] - g, row[j - 1] - g, prev[j - 1] + w[x][y])
     *          best = max(best, row[j])             # the 0 floor is the blindness
     *  smith_waterman = best
     *  @endcode
     */
    usearch_sequence_metric_smith_waterman_k = 6,

    /**
     *  @brief Window overlap: the share of the larger window set the other side misses.
     *
     *  Every window of every configured width is taken from both sequences, and a window of the
     *  stored @c b is a hit when the query @c a holds the same bytes at the same width. The
     *  distance is the misses over whichever side has more windows: zero where the windows
     *  coincide at every width, one where they share none. Exact at every radius; the window
     *  widths are in bytes, fixed at creation and recorded in the checkpoint.
     *
     *  d(a, b) = (max(|W(a)|, |W(b)|) − |{ v ∈ W(b) : v ∈ W(a) }|) / max(|W(a)|, |W(b)|)
     *
     *  @code{.py}
     *  windows = lambda s: [(w, s[i:i + w]) for w in widths for i in range(len(s) - w + 1)]
     *  query, stored = windows(a), windows(b)       # repeats kept, every width at once
     *  hits = sum(window in set(query) for window in stored)
     *  longer = max(len(query), len(stored))
     *  overlap = (longer - hits) / longer if longer else 1
     *  @endcode
     */
    usearch_sequence_metric_overlap_k = 7,
} usearch_sequence_metric_t;

/** The substitution matrix and gap costs an alignment kind scores with; the Levenshtein kinds
 *  score with unit costs and take none. */
typedef struct usearch_sequence_weights_t {

    /** Row-major 256×256 signed scores over byte alphabets. */
    int8_t const* substitution_matrix;

    /** Cost of opening a gap; the linear kinds use it as the flat gap cost. */
    int8_t gap_open;

    /** Cost of extending an open gap; ignored by the linear kinds. */
    int8_t gap_extend;
} usearch_sequence_weights_t;

/** The arguments of @c usearch_sequence_init_ram. */
typedef struct usearch_sequence_init_ram_t {

    /** Output for the new index. */
    usearch_sequence_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The scoring kind. */
    usearch_sequence_metric_t metric;

    /** The weights an alignment kind needs; @c NULL for the Levenshtein kinds, which refuse any. */
    usearch_sequence_weights_t const* weights;

    /** Window widths in bytes for the overlap kind, which needs at least one; every other kind
     *  refuses any. */
    size_t const* window_widths;

    /** Number of entries in @c window_widths. */
    size_t window_widths_count;

    /** The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** Upper bound on entries sharing one key; zero derives to one - unique keys. */
    size_t values_per_key;

    /** The memory domains to place one shard on apiece. */
    usearch_ram_options_t ram;
} usearch_sequence_init_ram_t;

/** Creates an empty, mutable sequence index in host memory, for the uniform edit distances or
 *  the weighted alignments alike. */
USEARCH_EXPORT void usearch_sequence_init_ram(usearch_sequence_init_ram_t* arguments);

/** The arguments of @c usearch_sequence_init_vram, which match @c usearch_sequence_init_ram_t but
 *  for the placement. */
typedef struct usearch_sequence_init_vram_t {

    /** Output for the new index. */
    usearch_sequence_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The scoring kind; byte and UTF-8 Levenshtein serve a GPU. */
    usearch_sequence_metric_t metric;

    /** The weights an alignment kind needs; @c NULL for the Levenshtein kinds, which refuse any. */
    usearch_sequence_weights_t const* weights;

    /** Window widths in bytes for the overlap kind; every other kind refuses any. */
    size_t const* window_widths;

    /** Number of entries in @c window_widths. */
    size_t window_widths_count;

    /** The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** Zero alone: a GPU index keeps no key lookup, so it bounds no key's entries and refuses any
     *  bound it is given. */
    size_t values_per_key;

    /** The one device to place the index on; naming none picks device zero. */
    usearch_vram_options_t vram;
} usearch_sequence_init_vram_t;

/**
 *  @brief Creates an empty, mutable sequence index in device memory.
 *  @retval usearch_status_unsupported_k A kind other than byte or UTF-8 Levenshtein, a
 *      @c values_per_key bound, or more than one device.
 *  @retval usearch_status_device_k A device past the listing, one whose vendor no kernel serves, or
 *      a build without GPU kernels.
 *
 *  Insertions and searches on it run on an executor driving its device, and a search lands its
 *  matches in device memory in an arena made for that executor, or on host pages in a host one.
 */
USEARCH_EXPORT void usearch_sequence_init_vram(usearch_sequence_init_vram_t* arguments);

/** The arguments of @c usearch_sequence_load_ram. */
typedef struct usearch_sequence_load_ram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_sequence_index_t* index;

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

    /** The weights an alignment checkpoint was built with; @c NULL for the Levenshtein kinds. */
    usearch_sequence_weights_t const* weights;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
} usearch_sequence_load_ram_t;

/**
 *  @brief Copies a checkpoint into a new, mutable index in host memory.
 *
 *  Where the parts end up is the call's own memory domains, never the manifest's. A part
 *  carries its own length in its header, so a second load skips whatever already arrived
 *  whole and cancelling costs the part being read rather than everything behind it.
 */
USEARCH_EXPORT void usearch_sequence_load_ram(usearch_sequence_load_ram_t* arguments);

/** The arguments of @c usearch_sequence_load_vram. */
typedef struct usearch_sequence_load_vram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_sequence_index_t* index;

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

    /** Zero alone: a GPU index keeps no key lookup, so it bounds no key's entries and refuses any
     *  bound it is given. */
    size_t values_per_key;

    /** The weights an alignment checkpoint was built with; @c NULL for the Levenshtein kinds. */
    usearch_sequence_weights_t const* weights;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
} usearch_sequence_load_vram_t;

/** Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
 *  @c usearch_sequence_load_ram. */
USEARCH_EXPORT void usearch_sequence_load_vram(usearch_sequence_load_vram_t* arguments);

/** The arguments of @c usearch_sequence_view_ram. */
typedef struct usearch_sequence_view_ram_t {

    /** Output for the new, read-only index. */
    usearch_sequence_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** The weights an alignment checkpoint was built with; @c NULL for the Levenshtein kinds. */
    usearch_sequence_weights_t const* weights;
} usearch_sequence_view_ram_t;

/** Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
 *  verbs report @c usearch_status_read_only_k. */
USEARCH_EXPORT void usearch_sequence_view_ram(usearch_sequence_view_ram_t* arguments);

/** The arguments of @c usearch_sequence_view_nvme. */
typedef struct usearch_sequence_view_nvme_t {

    /** Output for the new, read-only index, written by the time the call is final. */
    usearch_sequence_index_t* index;

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

    /** The weights an alignment checkpoint was built with; @c NULL for the Levenshtein kinds. */
    usearch_sequence_weights_t const* weights;
} usearch_sequence_view_nvme_t;

/**
 *  @brief Serves a checkpoint in place from NVMe drives as a read-only index, every later call
 *      reading it through its executor's engines.
 *
 *  Opening reads the headers alone. Every mutating verb on the handle reports
 *  @c usearch_status_read_only_k, and the parts must stay unmodified until the handle is freed.
 */
USEARCH_EXPORT void usearch_sequence_view_nvme(usearch_sequence_view_nvme_t* arguments);

/** Frees the resources associated with the index; @c NULL is safe, and it cannot fail, so takes
 *  no outcome.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_sequence_free(usearch_sequence_index_t index);

/** The arguments of @c usearch_sequence_reserve. */
typedef struct usearch_sequence_reserve_t {

    /** The handle to be resized. */
    usearch_sequence_index_t index;

    /** The executor the growth runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The desired total entry capacity, including the current size. */
    size_t entries;

    /** Expected total payload volume; zero lets the payload storage grow lazily. */
    size_t payload_bytes;
} usearch_sequence_reserve_t;

/** Reserves capacity for a number of incoming sequences ahead of insertion. */
USEARCH_EXPORT void usearch_sequence_reserve(usearch_sequence_reserve_t* arguments);

/** The arguments of @c usearch_sequence_clear. */
typedef struct usearch_sequence_clear_t {

    /** The handle to be cleared. */
    usearch_sequence_index_t index;

    /** The executor the erasure runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;
} usearch_sequence_clear_t;

/** Erases all entries, keeping the reserved capacity. */
USEARCH_EXPORT void usearch_sequence_clear(usearch_sequence_clear_t* arguments);

/** The arguments of @c usearch_sequence_insert. */
typedef struct usearch_sequence_insert_t {

    /** The handle to be populated. */
    usearch_sequence_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Sequences to admit - the batch axis of @c keys and @c sequences. */
    size_t count;

    /** The identities to admit the sequences under, one per sequence. */
    usearch_keys_view_t keys;

    /** The sequences, as bytes end to end around @c count + 1 offsets. */
    usearch_tape_view_t sequences;

    /** The @b optional breadth of the candidates the build walk explores - the @c ef of the
     *  literature; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries admitted and the batch length. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_insert_t;

/**
 *  @brief Adds a tape of sequences under their keys - the native batch operation; a single
 *      sequence is a tape of one.
 *  @retval usearch_status_read_only_k The handle is a view.
 *  @retval usearch_status_overflow_k A key already holds @c values_per_key entries.
 *
 *  An insertion is all-or-nothing per call. Every fallible step runs before the first entry
 *  identity is claimed, so a rejected call publishes nothing - a cancelled one included, which
 *  reports @c usearch_status_cancelled_k - and the entries of an accepted one become visible at
 *  once, their keys right after.
 *
 *  A key already holding @c values_per_key entries rejects the whole call, since a smaller retry
 *  cannot help.
 */
USEARCH_EXPORT void usearch_sequence_insert(usearch_sequence_insert_t* arguments);

/** The arguments of @c usearch_sequence_refine. */
typedef struct usearch_sequence_refine_t {

    /** The handle to be refined. */
    usearch_sequence_index_t index;

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
} usearch_sequence_refine_t;

/**
 *  @brief Re-selects the neighbors of every stored entry against the finished graph, improving
 *      neighbors chosen when the graph around them was still incomplete.
 *  @retval usearch_status_read_only_k The handle is an NVMe view.
 *  @retval usearch_status_unsupported_k The handle lives on a GPU, whose index refines nothing yet.
 *  @retval usearch_status_contended_k An insertion or another refinement holds the handle.
 *
 *  This is a build step rather than a repair. A cheap insertion pass followed by one refinement
 *  reaches a better graph, in less total time, than a single expensive insertion pass - so a wide
 *  @c expansion_factor belongs here rather than in @c usearch_sequence_insert. It pays most in this
 *  modality, where a single alignment is quadratic and a stingy insertion pass is tempting.
 *
 *  Cancelling is safe, which is what makes a whole-graph sweep affordable. Re-selecting an entry's
 *  neighbors is idempotent and claims no identity, so a stopped pass leaves the graph correct and
 *  merely less improved: nothing unwinds, and the call answers @c usearch_status_cancelled_k. The
 *  index keeps nothing of a stopped pass, so the next call starts at the first entry again; a
 *  caller wanting to continue where a cancel stopped holds a @c usearch_sequence_refinement_t
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
USEARCH_EXPORT void usearch_sequence_refine(usearch_sequence_refine_t* arguments);

/** The arguments of @c usearch_sequence_search. */
typedef struct usearch_sequence_search_t {

    /** The handle to be queried. */
    usearch_sequence_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Queries to resolve; the outputs have @c count rows of @c wanted entries. */
    size_t count;

    /** The queries, as bytes end to end around @c count + 1 offsets. */
    usearch_tape_view_t queries;

    /** Output for the run, inside @c arena. */
    usearch_matches_view_t* matches;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /** Columns per row, and the upper bound on matches - the "k" in "kANN". */
    size_t wanted;

    /** The @b optional breadth of the candidates the walk explores - the @c ef of the
     *  literature; zero derives. */
    size_t expansion_factor;

    /** The @b optional admission filter, asked about a batch of candidates at a time. */
    usearch_predicate_t predicate;

    /** The @b optional handler, called with queries resolved and their count. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_search_t;

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a tape of queries - the native batch
 *      operation, parallelized internally; one query is a tape of one. Never writes to the graph,
 *      so any number may overlap with concurrent insertions. @ref sequence_index_concurrency
 *
 *  A @c predicate narrows the matches to the entries it admits. Non-matching entries still route
 *  the traversal rather than terminating it, so recall does not collapse at low selectivity - the
 *  walk crosses rejected regions to reach admitted ones behind them. The cost is that the
 *  predicate judges far more keys than @c wanted, every scored candidate, asked about a batch of
 *  them at a time, so it belongs on the caller's fast path.
 *
 *  A cancelled search reports @c usearch_status_cancelled_k and leaves @c matches @c NULL.
 */
USEARCH_EXPORT void usearch_sequence_search(usearch_sequence_search_t* arguments);

/** What one key holds, enough to size a buffer before retrieving any of it. */
typedef struct usearch_sequence_key_stats_t {

    /** Entries sharing the key; keys need not be unique. */
    size_t entries;

    /** Bytes summed across those entries. */
    size_t total_bytes;

    /** Bytes of the longest single entry. */
    size_t largest_bytes;
} usearch_sequence_key_stats_t;

/** The arguments of @c usearch_sequence_count. */
typedef struct usearch_sequence_count_t {

    /** The handle to be queried. */
    usearch_sequence_index_t index;

    /** The executor the lookup runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are counted. */
    usearch_key_t key;

    /** Output for what @c key holds; zero @c entries means the index holds none. */
    usearch_sequence_key_stats_t* key_stats;
} usearch_sequence_count_t;

/** Counts the entries stored under one key, with their payload extent. */
USEARCH_EXPORT void usearch_sequence_count(usearch_sequence_count_t* arguments);

/** The arguments of @c usearch_sequence_get. */
typedef struct usearch_sequence_get_t {

    /** The handle to be queried. */
    usearch_sequence_index_t index;

    /** The executor whose thread copies the entries. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are exported. */
    usearch_key_t key;

    /** Output for the tape inside @c arena: entries' bytes end to end, never terminated, around
     *  @c written + 1 offsets. */
    usearch_tape_view_t* sequences;

    /** The @b optional output for the entries written; zero when the key is absent. */
    size_t* written;

    /** Memory for @c sequences, rewound by the call. */
    usearch_arena_t arena;
} usearch_sequence_get_t;

/**
 *  @brief Retrieves every sequence stored under one key as an Apache Arrow tape, in insertion order -
 *      one call, however many and however long they turn out to be.
 *
 *  An absent key leaves the tape's pointers @c NULL.
 */
USEARCH_EXPORT void usearch_sequence_get(usearch_sequence_get_t* arguments);

/** The arguments of @c usearch_sequence_absorb. */
typedef struct usearch_sequence_absorb_t {

    /** The handle absorbing the donors. */
    usearch_sequence_index_t index;

    /** The executor each donor's landing and the closing refinement fan out over. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Donor handles to absorb. */
    size_t count;

    /** Array of @c count handles, each consumed and nulled once its entries are published here. */
    usearch_sequence_index_t* donors;

    /** The @b optional output for the entries the closing refine changed. */
    size_t* changed;

    /** The @b optional output for the audit closing the heal. */
    usearch_reachability_t* reachability;

    /** The @b optional breadth of the candidates of the closing refine's walks; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries the closing refine re-selected; a cancel stops
     *  it as it stops @c usearch_sequence_refine, never a landing. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_absorb_t;

/**
 *  @brief Absorbs every handle in @c donors into @c index, one after another, keeping their graphs
 *      rather than re-inserting their sequences, and links each in by walks across. Donor handles
 *      are consumed and must not be used afterwards, not even for @c usearch_sequence_free.
 *  @retval usearch_status_read_only_k @c index is an NVMe view.
 *  @retval usearch_status_unsupported_k @c index lives on a GPU, whose index absorbs nothing yet.
 *  @retval usearch_status_invalid_argument_k A @c NULL slot in @c donors, or @c NULL @c donors
 *      for a nonzero @c count.
 *  @retval usearch_status_config_mismatch_k A donor of another residency, class, kind or geometry,
 *      or @c index itself.
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
 *  @c usearch_sequence_refine: a cancel skips the rest of the walks across, still re-links, and
 *  answers @c usearch_status_cancelled_k with every donor consumed. A caller wanting to continue
 *  those walks later holds a @c usearch_sequence_refinement_t instead.
 */
USEARCH_EXPORT void usearch_sequence_absorb(usearch_sequence_absorb_t* arguments);

/** The arguments of @c usearch_sequence_split. */
typedef struct usearch_sequence_split_t {

    /** The handle split into the receivers, left empty and live. */
    usearch_sequence_index_t index;

    /** The executor the partition, each part's landing and the closing refinements fan out over. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Receiver handles to split into, at least two. */
    size_t count;

    /** Array of @c count empty handles of the source's residency, class, kind and geometry, each
     *  receiving one part. */
    usearch_sequence_index_t const* receivers;

    /** The @b optional output for the entries the closing refines changed, across the receivers. */
    size_t* changed;

    /** The @b optional output for the audits closing the heals, across the receivers. */
    usearch_reachability_t* reachability;

    /** The @b optional breadth of the candidates of the closing refines' walks; zero derives. */
    size_t expansion_factor;

    /** Picks the part roots: equal seeds over an unchanged index split equally. */
    uint64_t seed;

    /** The @b optional handler, called with entries the closing refines re-selected; a cancel stops
     *  them as it stops @c usearch_sequence_refine, never a landing. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_split_t;

/**
 *  @brief Splits @c index into the handles in @c receivers, the inverse of
 *      @c usearch_sequence_absorb: graph-local parts keeping their edges rather than re-inserting
 *      their sequences, one part per receiver, and @c index left empty but live.
 *  @retval usearch_status_read_only_k @c index is a view.
 *  @retval usearch_status_invalid_argument_k Fewer than two receivers, @c NULL @c receivers, or a
 *      @c NULL slot in them.
 *  @retval usearch_status_config_mismatch_k A receiver of another residency, class, kind or
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
 *  what that strands, as @c usearch_sequence_refine does. Insertions into @c index or a receiver
 *  arriving meanwhile are refused with @c usearch_status_contended_k, and nothing may search either
 *  until the call is final.
 *
 *  The handler reports and cancels the closing refinements alone, exactly as in
 *  @c usearch_sequence_refine: a cancel skips the rest of the re-selection, still re-links, and
 *  answers @c usearch_status_cancelled_k with every part landed. A caller wanting to continue that
 *  re-selection later holds a @c usearch_sequence_refinement_t opened with @c receivers instead.
 */
USEARCH_EXPORT void usearch_sequence_split(usearch_sequence_split_t* arguments);

/** The arguments of @c usearch_sequence_refinement_init. */
typedef struct usearch_sequence_refinement_init_t {

    /** The handle to be refined. */
    usearch_sequence_index_t index;

    /** The executor every call on the refinement runs on; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output for the new handle; @c NULL on failure. The index admits no insertion or save until
     *  @c usearch_sequence_refinement_finish frees it. */
    usearch_sequence_refinement_t* refinement;

    /** The @b optional donors to absorb first, each consumed and nulled once its entries land; the
     *  array must outlive the refinement. */
    usearch_sequence_index_t* donors;

    /** Donor handles in @c donors. */
    size_t count;

    /** The @b optional receivers to split @c index into first, as @c usearch_sequence_split lands
     *  them, each refined in turn; the array must outlive the refinement. */
    usearch_sequence_index_t const* receivers;

    /** Receiver handles in @c receivers, zero or at least two. */
    size_t receivers_count;

    /** The @b optional breadth of the candidates the re-selection walk explores; zero derives. */
    size_t expansion_factor;

    /** Picks the part roots when @c receivers are named, as in @c usearch_sequence_split. */
    uint64_t seed;
} usearch_sequence_refinement_init_t;

/**
 *  @brief Opens a refinement of @c index, absorbing @c donors first when any are named, as
 *      @c usearch_sequence_absorb lands them, or splitting it into @c receivers, as
 *      @c usearch_sequence_split lands them. Every allocation the refinement needs in the index
 *      happens here, never in a step.
 *  @retval usearch_status_read_only_k @c index is an NVMe view.
 *  @retval usearch_status_unsupported_k @c index lives on a GPU, whose index refines nothing yet.
 *  @retval usearch_status_invalid_argument_k No executor, a @c NULL slot in @c donors or
 *      @c receivers, @c NULL @c donors for a nonzero @c count, one receiver alone, or donors and
 *      receivers both named.
 *  @retval usearch_status_config_mismatch_k A donor or receiver of another residency, class, kind
 *      or geometry, a receiver holding entries, or @c index itself.
 *  @retval usearch_status_contended_k An insertion or another refinement in flight on @c index, on
 *      a donor or on a receiver.
 */
USEARCH_EXPORT void usearch_sequence_refinement_init(usearch_sequence_refinement_init_t* arguments);

/** The arguments of @c usearch_sequence_refinement_next. */
typedef struct usearch_sequence_refinement_next_t {

    /** The handle the refinement was opened on. */
    usearch_sequence_index_t index;

    /** The executor the refinement was opened with. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The refinement to continue. */
    usearch_sequence_refinement_t refinement;

    /** The @b optional handler, called with entries re-selected across every call so far. */
    usearch_progress_t progress;

    /** The @b optional outputs for the entries re-selected so far and, of those, the ones whose
     *  neighbors changed; their ratio falling is what says another call buys little. */
    size_t* visited;
    size_t* changed;

    /** The @b optional traversal telemetry of every call on the refinement so far; @c NULL skips
     *  reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_refinement_next_t;

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
USEARCH_EXPORT void usearch_sequence_refinement_next(usearch_sequence_refinement_next_t* arguments);

/** The arguments of @c usearch_sequence_refinement_finish. */
typedef struct usearch_sequence_refinement_finish_t {

    /** The handle the refinement was opened on. */
    usearch_sequence_index_t index;

    /** The executor the refinement was opened with. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The refinement to finish; freed by the call, whatever it answers. */
    usearch_sequence_refinement_t refinement;

    /** The @b optional output for the number of entries whose neighbors changed. */
    size_t* changed;

    /** The @b optional output for the audit closing the heal. */
    usearch_reachability_t* reachability;

    /** The @b optional traversal telemetry of every call on the refinement; @c NULL skips
     *  reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_refinement_finish_t;

/**
 *  @brief Skips whatever re-selection a stopped refinement left, re-links what it strands as
 *      @c usearch_sequence_refine does, reports, reopens the index, and frees the handle. A
 *      refinement whose donors have not all landed re-links nothing and reports zeros.
 *  @retval usearch_status_invalid_argument_k @c index or @c executor is not the one the refinement
 *      was opened with; the handle is still freed.
 */
USEARCH_EXPORT void usearch_sequence_refinement_finish(usearch_sequence_refinement_finish_t* arguments);

/** The arguments of @c usearch_sequence_cursor_init. */
typedef struct usearch_sequence_cursor_init_t {

    /** Output for the new cursor; @c NULL on failure. */
    usearch_sequence_cursor_t* cursor;

    /** The handle to be scanned. */
    usearch_sequence_index_t index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The @b optional count of @c starts; zero walks every entry as stored. */
    size_t starts_count;

    /** The keys to walk from, in the order the walk begins with them. */
    usearch_key_t const* starts;
} usearch_sequence_cursor_init_t;

/**
 *  @brief Opens a snapshot cursor over the current contents; see @c usearch_sequence_cursor_t.
 *  @retval usearch_status_invalid_argument_k @c NULL @c starts for a nonzero @c starts_count.
 *
 *  Without @c starts the cursor yields every entry as stored. With them it yields the start keys'
 *  own entries first, in the order given, then the entries those link to, breadth-first, each
 *  entry once, so a caller wanting only the starts' immediate links reads the rows of the first
 *  pulled entries. A key the index lacks starts nothing.
 */
USEARCH_EXPORT void usearch_sequence_cursor_init(usearch_sequence_cursor_init_t* arguments);

/** The arguments of @c usearch_sequence_cursor_next. */
typedef struct usearch_sequence_cursor_next_t {

    /** The handle the cursor was opened on. */
    usearch_sequence_index_t index;

    /** The executor the pull runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The cursor to advance. */
    usearch_sequence_cursor_t cursor;

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
} usearch_sequence_cursor_next_t;

/**
 *  @brief Copies the next batch of entries out of the cursor into @c arena, each with the keys it
 *      links to, in the order @c usearch_sequence_cursor_init describes; the sequences themselves
 *      come from @c usearch_sequence_get.
 *  @retval usearch_status_invalid_argument_k @c index is not the handle the cursor was opened on.
 *
 *  Links name only entries the snapshot holds, as the graph stored them when the pull ran. The
 *  keys are copied rather than borrowed from the index, so the batch stays valid until the next
 *  call given the same arena, however the cursor moves meanwhile.
 */
USEARCH_EXPORT void usearch_sequence_cursor_next(usearch_sequence_cursor_next_t* arguments);

/** Releases the cursor; @c NULL is safe.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_sequence_cursor_free(usearch_sequence_cursor_t cursor);

/** The arguments of @c usearch_sequence_sample. */
typedef struct usearch_sequence_sample_t {

    /** The handle to be sampled. */
    usearch_sequence_index_t index;

    /** The executor the draws run on; its kind must match the handle's residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The @b optional count of @c anchors; zero draws from the whole index. */
    size_t anchors_count;

    /** The sequences that bound the region around them, one anchor apiece, as bytes end to end
     *  around @c anchors_count + 1 offsets. */
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
} usearch_sequence_sample_t;

/**
 *  @brief Draws entries uniformly, across the index or around anchors, each linked to its nearest
 *      earlier draws - so the sample is one connected component, ready to be laid out.
 *
 *  Links name earlier draws of the same sequence, a later call's reaching an earlier call's. They
 *  are exact in a scan and approximate as any walk otherwise.
 */
USEARCH_EXPORT void usearch_sequence_sample(usearch_sequence_sample_t* arguments);

/** The arguments of @c usearch_sequence_save. */
typedef struct usearch_sequence_save_t {

    /** The handle to be written. */
    usearch_sequence_index_t index;

    /** The executor the write fans out over, one shard per thread. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Destination path, whose extension picks the layout. */
    usearch_bytes_view_t path;

    /** The @b optional handler, called with parts written and their count. */
    usearch_progress_t progress;
} usearch_sequence_save_t;

/**
 *  @brief Saves a mutable index under @c path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  A cancelled save leaves the parts it already wrote and no manifest referencing them, so the
 *  checkpoint reads as absent rather than half-valid. Those leftovers are the caller's to remove.
 */
USEARCH_EXPORT void usearch_sequence_save(usearch_sequence_save_t* arguments);

/** The metric and geometry a checkpoint froze - everything a reader learns without opening it. */
typedef struct usearch_sequence_metadata_t {

    /** The scoring kind the checkpoint was built under. */
    usearch_sequence_metric_t metric;

    /** Connections per node the graph was built with. */
    size_t connectivity;
} usearch_sequence_metadata_t;

/** The arguments of @c usearch_sequence_metadata_read. */
typedef struct usearch_sequence_metadata_read_t {

    /** The executor the read runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** Output for every field the checkpoint froze. */
    usearch_sequence_metadata_t* metadata;
} usearch_sequence_metadata_read_t;

/** Reads a checkpoint's metric & geometry without loading it. */
USEARCH_EXPORT void usearch_sequence_metadata_read(usearch_sequence_metadata_read_t* arguments);

/** A live index's state. Each counter is read once, so under concurrent insertions they may come
 *  from slightly different moments. */
typedef struct usearch_sequence_index_description_t {

    /** The frozen metric and connectivity, as a checkpoint reports them. */
    usearch_sequence_metadata_t metadata;

    /** Stored entries. */
    size_t size;

    /** Entries hosted without growing again. */
    size_t capacity;

    /** Bytes the handle holds, payload storage included; traversal scratch is the executor's. */
    size_t memory_usage;

    /** Static codename of the kernels the index dispatched to on this machine. */
    char const* capabilities;
} usearch_sequence_index_description_t;

/** The arguments of @c usearch_sequence_describe. */
typedef struct usearch_sequence_describe_t {

    /** The handle to be described. */
    usearch_sequence_index_t index;

    /** The executor the report runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output, filled whole. */
    usearch_sequence_index_description_t* description;
} usearch_sequence_describe_t;

/** Reports a live index's frozen metadata and current counters in one call. */
USEARCH_EXPORT void usearch_sequence_describe(usearch_sequence_describe_t* arguments);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_SEQUENCE_INDEX_H
