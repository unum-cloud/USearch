/**
 *  @file python/lib.h
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief Declarations shared by every USearch CPython translation unit.
 *
 *  One extension, one file per modality: @c types.c owns the vocabulary every modality
 *  shares, each @c <modality>_index.c owns its verbs, and @c lib.c owns the module glue
 *  alone. Types are defined once here and readied once in @c lib.c, so a @c Key crosses
 *  modality boundaries without a check ever failing.
 *
 *  @section python_calling_convention Latency, Quality, and Arguments Parsing
 *
 *  No PyBind11, NanoBind, or SWIG - only the CPython API, and never
 *  @c PyArg_ParseTupleAndKeywords, whose format specifiers are parsed at runtime and which
 *  cannot use the fast calling convention. Every verb is @c METH_FASTCALL | @c METH_KEYWORDS and
 *  binds its arguments through one table of names, @c usearch_py_arguments_parse.
 *
 *  @section python_receivers Receivers and Deferred Calls
 *
 *  The type of the object a verb is called on chooses where its call runs and how it completes: an
 *  index runs it on the calling thread, an @c On blocks on its executor, and a @c DeferredOn
 *  submits it there and answers a @c Deferred. A verb fills one call record, kept on the stack
 *  while it blocks and inside its @c Deferred once submitted, so one body serves every receiver and
 *  nothing the library borrows ever moves. Its result object is the output slot the library writes
 *  into, so a deferred call answers it as a blocking one does. The binding starts no threads, takes
 *  no locks and re-checks nothing the library refuses; it counts only the GPU arenas an executor
 *  made, since each frees on the executor's stream, so a close leaves its handle to the last of
 *  them. Callbacks never raise into the library: an exception escaping one goes to
 *  @c sys.unraisablehook and the callback answers cancel or reject. Dropping a @c Deferred waits on
 *  its outcome, and every other wait belongs to the C library.
 *
 *  @section python_buffers Buffer Protocol and NumPy Compatibility
 *
 *  Batch inputs arrive through PEP 3118 first and @c __array_interface__ second, so a PyTorch or
 *  TensorFlow tensor reaches the index without a NumPy round-trip and NumPy itself is never a
 *  build-time dependency. Every output whose length scales with the call lands in the @c Arena the
 *  caller passes and leaves as zero-copy results over it. A result views its arena until the next
 *  call on that arena rewinds it, as a C view does, and @c Arena.describe says where it lives.
 *
 *  @section python_arrow Arrow PyCapsule Interface
 *
 *  Anything else exporting @c __arrow_c_array__ or @c __arrow_c_stream__ reads as a third source,
 *  so a PyArrow or Polars column reaches the index with neither library imported: one chunk stays
 *  in place, and several gather once, since one call is one batch. Results export the same
 *  protocol over their arena, an exported array holding the arena until its release.
 *
 *  @section python_naming Naming
 *
 *  A Python object's structure takes its class name, @c Matches for @c usearch.Matches, and its type
 *  object @c MatchesType. A C helper shared through this header takes the @c usearch_py_ prefix and
 *  the @c _t suffix, since C has no namespaces. A helper one file keeps to itself takes only @c _t.
 */
#ifndef UNUM_USEARCH_PYTHON_LIB_H
#define UNUM_USEARCH_PYTHON_LIB_H

/*  CPython defines preprocessor macros that reach into the standard headers, so it leads rather
 *  than joining the third-party group; every unit here includes this file before anything else. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdarg.h>    // `va_list` - the cause-preserving raise
#include <stdatomic.h> // `atomic_load_explicit`, `atomic_compare_exchange_strong`
#include <stddef.h>    // `size_t`
#include <stdint.h>    // `uint8_t`, `uint64_t`, `PTRDIFF_MAX`
#include <string.h>    // `memcpy`, `memset`

#include "usearch/arrow.h"
#include "usearch/dense_index.h"
#include "usearch/geospatial_index.h"
#include "usearch/sequence_index.h"
#include "usearch/shapes_index.h"
#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Calling Convention

/** Compile-time constants of the binding layer. */
enum {

    /** Fast-call plus keywords, the calling convention every batch verb uses. */
    usearch_py_method_flags_k = METH_FASTCALL | METH_KEYWORDS,

    /** The rank ceiling every batch verb accepts: a matrix, or a vector read as one row. */
    usearch_py_max_rank_k = 2,

    /** Objects one call holds until it is final: callables, telemetry, plans, lists of indexes. */
    usearch_py_holds_k = 5,

    /** Buffers one call borrows until it is final. */
    usearch_py_buffers_k = 3,

    /** Runs of keys one call borrows until it is final. */
    usearch_py_keys_k = 2,

    /** Heap blocks one call lends until it is final: ordinals and arrays of handles. */
    usearch_py_blocks_k = 4,

    /** Words of the C arguments one call carries, room for the widest verb and its own outputs. */
    usearch_py_arguments_words_k = 48,
};

/** @p value divided by @p divisor, rounded up - the packing arithmetic sub-byte kinds need. */
static inline size_t usearch_py_divide_round_up_(size_t value, size_t divisor) {
    return value / divisor + (value % divisor != 0);
}

/** Element count of a statically-sized table, which every naming table asks for. */
#define usearch_py_table_size_(table) (sizeof(table) / sizeof((table)[0]))

/** One argument a verb takes and the slot its value lands in; a slot the caller skipped stays NULL. */
typedef struct usearch_py_argument_t {

    /** The keyword spelling, which also names the argument in every refusal. */
    char const* name;

    /** Where the borrowed value lands. */
    PyObject** value;
} usearch_py_argument_t;

/**
 *  @brief Binds a fast-call's arguments to @p arguments, refusing unknown names, duplicates and missing ones.
 *
 *  The first @p positional_only arguments bind by position alone and must be given; the ones after
 *  them up to @p positional_max bind by position or by name; the rest bind by name alone.
 *
 *  @return 0 on success, -1 with a @c TypeError set otherwise.
 */
int usearch_py_arguments_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                               PyObject* args_names_tuple, Py_ssize_t positional_only, Py_ssize_t positional_max,
                               usearch_py_argument_t const* arguments, size_t count);

#pragma endregion Calling Convention

#pragma region Buffer Views

/** Backing storage for a synthesized @c Py_buffer, so the @c __array_interface__
 *  fallback never allocates. Pair one with every @c usearch_py_matrix_from_object call and
 *  release it with @c usearch_py_buffer_release once the call is final. */
typedef struct usearch_py_buffer_backing_t {

    /** The buffer, exported by PEP 3118 or synthesized from @c __array_interface__ . */
    Py_buffer buffer;

    /** Per-axis extents the synthesized buffer points at; unused on the PEP 3118 path. */
    Py_ssize_t shape[usearch_py_max_rank_k];

    /** Per-axis byte strides the synthesized buffer points at; likewise unused. */
    Py_ssize_t strides[usearch_py_max_rank_k];

    /** The exporter a synthesized buffer borrows from, or the Arrow column one was read from, kept
     *  alive; NULL otherwise. */
    PyObject* synthesized_owner;
} usearch_py_buffer_backing_t;

/**
 *  @brief Identities gathered for one batch, and whatever storage that gathering needed.
 *
 *  A caller passing an @c (N,16) buffer of identities hands us the array outright and @c owned
 *  stays @c NULL; a caller passing integers, as a list or a vector gathered in C, gets one
 *  allocation, freed by @c usearch_py_keys_release.
 */
typedef struct usearch_py_keys_t {

    /** The identities, aliasing @c owned or the exported buffer. */
    usearch_key_t* keys;

    /** The allocation backing @c keys, or NULL when they live in the caller's buffer. */
    usearch_key_t* owned;

    /** The exported buffer backing @c keys, zeroed when they were gathered. */
    usearch_py_buffer_backing_t backing;
} usearch_py_keys_t;

/**
 *  @brief The paths a residency arm borrows for the length of one factory call.
 *
 *  A path arrives as text and the ABI reads bytes, so the encoded objects have to outlive the
 *  call. The ABI reads the views contiguously, so the objects cannot ride inside them and
 *  trail them in the same block instead - one allocation, released as one.
 */
typedef struct usearch_py_paths_t {

    /** The run the arm reads, or NULL when the caller named no paths. */
    usearch_bytes_view_t* views;

    /** Paths the run holds; zero when the caller named none. */
    size_t count;
} usearch_py_paths_t;

#pragma endregion Buffer Views

#pragma region Shared Types

extern PyTypeObject KeyType;
extern PyTypeObject MatchType;
extern PyTypeObject MatchesType;
extern PyTypeObject ExecutorType;
extern PyTypeObject ArenaType;
extern PyTypeObject ArenaDescriptionType;
extern PyTypeObject ArenaBufferType;
extern PyTypeObject DenseVectorsType;
extern PyTypeObject DenseProductsType;
extern PyTypeObject MonolithType;
extern PyTypeObject DeferredType;
extern PyTypeObject OnType;
extern PyTypeObject DeferredOnType;

extern PyTypeObject DenseIndexType;
extern PyTypeObject DenseMetadataType;
extern PyTypeObject DenseIndexDescriptionType;
extern PyTypeObject DenseReachabilityType;
extern PyTypeObject RefineReportType;
extern PyTypeObject SampleType;
extern PyTypeObject DenseTraversalStatsType;
extern PyTypeObject DenseRefinePlanType;
extern PyTypeObject DenseReshardPlanType;
extern PyTypeObject DenseCursorType;
extern PyTypeObject DenseBatchType;
extern PyTypeObject DenseSearchStreamType;

extern PyTypeObject SequenceIndexType;
extern PyTypeObject SequenceMetadataType;
extern PyTypeObject SequenceIndexDescriptionType;
extern PyTypeObject SequenceRefinePlanType;
extern PyTypeObject SequenceReshardPlanType;
extern PyTypeObject ShapesIndexType;
extern PyTypeObject ShapesMetadataType;
extern PyTypeObject ShapesIndexDescriptionType;
extern PyTypeObject ShapesRefinePlanType;
extern PyTypeObject ShapesReshardPlanType;
extern PyTypeObject GeospatialIndexType;
extern PyTypeObject GeospatialMetadataType;
extern PyTypeObject GeospatialIndexDescriptionType;
extern PyTypeObject GeospatialRefinePlanType;
extern PyTypeObject GeospatialReshardPlanType;

/** The layouts of the struct sequences, which `lib.c` readies through their own initializer. */
extern PyStructSequence_Desc usearch_py_match_desc;
extern PyStructSequence_Desc usearch_py_arena_description_desc;
extern PyStructSequence_Desc usearch_py_dense_metadata_desc;
extern PyStructSequence_Desc usearch_py_sequence_metadata_desc;
extern PyStructSequence_Desc usearch_py_shapes_metadata_desc;
extern PyStructSequence_Desc usearch_py_geospatial_metadata_desc;

/** The base of the one exception class per failing status, subclassing @c RuntimeError. */
extern PyObject* ErrorType;

/** The workers a fanned-out verb runs on, owned apart from any index so one pool serves many. */
typedef struct Executor {
    PyObject ob_base;

    /** The library's executor as verbs read it, NULL once closed. */
    _Atomic(usearch_executor_t) handle;

    /** The library's executor, freed once nothing holds it. */
    usearch_executor_t owned;

    /** One while it is open, plus one per GPU arena it made that is still alive, since each frees
     *  on its first stream. */
    atomic_size_t holds;
} Executor;

/** The memory every output whose length scales with a call lands in, rewound by the next call
 *  handed it. */
typedef struct Arena {
    PyObject ob_base;
    usearch_arena_t handle;

    /** The @c Executor a GPU arena serves alone, held until the arena is freed; NULL for host
     *  pages. */
    PyObject* executor;
} Arena;

/** A C enum mirrored as an `enum.IntEnum`, its members spelled by the C name table. */
typedef struct usearch_py_enum_t {

    /** The class name, as `usearch` exports it, such as "Status". */
    char const* name;

    /** The lowest and highest values C spells; a gap spelling "unknown" or "unrecognized" is no member. */
    int first, last;

    /** A one-line wrapper over `usearch_status_name` and its siblings. */
    char const* (*spell)(int);

    /** The class, built once at module init and never written again. */
    PyObject* type;
} usearch_py_enum_t;

extern usearch_py_enum_t usearch_py_status, usearch_py_mood, usearch_py_dense_metric, usearch_py_scalar,
    usearch_py_residency, usearch_py_traversal, usearch_py_cpu_class, usearch_py_gpu_class, usearch_py_nvme_class;
extern usearch_py_enum_t usearch_py_sequence_metric, usearch_py_shapes_metric, usearch_py_geospatial_relation,
    usearch_py_refine_verdict, usearch_py_phase;

/**
 *  @brief Builds every mirrored enum's class from `enum.IntEnum` and adds it to @p module, one
 *      member per value C spells, `out-of-memory` becoming `OUT_OF_MEMORY`.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_register_enums(PyObject* module);

/**
 *  @brief Reads a member of @p mirrored from an `int`, refusing a `bool`, text and values no member carries.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_enum_from_object(PyObject* object, char const* field, usearch_py_enum_t const* mirrored, int* value);

/** The member of @p mirrored valued @p value; returns a new reference, or @c NULL with an exception set. */
PyObject* usearch_py_enum_to_object(usearch_py_enum_t const* mirrored, int value);

#pragma endregion Shared Types

#pragma region Calls

/** What a verb's receiver names: the index, the executor the call runs on and how it completes. */
typedef struct usearch_py_receiver_t {

    /** The receiver object, which a deferred call holds so its index and executor outlive the call. */
    PyObject* origin;

    /** The index the verb addresses, borrowed from @c origin. */
    PyObject* index;

    /** The workers the call runs on; NULL runs it on the calling thread. */
    usearch_executor_t executor;

    /** Blocking for an index or an @c On, deferred for a @c DeferredOn. */
    usearch_completion_t completion;
} usearch_py_receiver_t;

/** The verbs the receivers over one index type offer, which @c On and @c DeferredOn look up by the
 *  type of the index they hold, so one receiver type serves every modality. */
typedef struct usearch_py_receiver_verbs_t {

    /** The index type the receivers address. */
    PyTypeObject* index;

    /** The verbs an @c On offers, each blocking on its executor. */
    PyMethodDef* on;

    /** The verbs a @c DeferredOn offers, each answering a @c Deferred; cursors and streams block on
     *  every pull, so they are absent. */
    PyMethodDef* deferred_on;
} usearch_py_receiver_verbs_t;

extern usearch_py_receiver_verbs_t const usearch_py_dense_receiver_verbs, usearch_py_sequence_receiver_verbs,
    usearch_py_shapes_receiver_verbs, usearch_py_geospatial_receiver_verbs;

/**
 *  @brief Reads what @p self names: an index runs on the calling thread, an @c On or a @c DeferredOn on its executor.
 *  @return 0 on success, -1 with an exception set when the receiver's executor was closed.
 */
int usearch_py_receiver_from_object(PyObject* self, usearch_py_receiver_t* receiver);

/** A new @c On or @c DeferredOn, as @p type names, holding @p index and the @c Executor @p executor;
 *  NULL on failure. */
PyObject* usearch_py_on_new(PyTypeObject* type, PyObject* index, PyObject* executor);

/**
 *  @brief Everything one call keeps until it is final: on the stack while it blocks, inside its
 *      @c Deferred once submitted.
 *
 *  A verb fills the slots it needs and leaves the rest zeroed, so letting go of a call is the same
 *  for every verb, and the C arguments ride at the end, where the library reads them.
 */
typedef struct usearch_py_call_t {

    /** The outcome the library finishes. */
    usearch_outcome_t outcome;

    /** Builds the answer from outputs a call keeps in its arguments; NULL answers @c result. */
    PyObject* (*answer)(struct usearch_py_call_t* call);

    /** The @c Arena the outputs land in, held; NULL for a verb with none. */
    PyObject* arena;

    /** The object the library writes the outputs into and the call answers; NULL answers None. */
    PyObject* result;

    /** Objects the library reads until the call is final, each held. */
    PyObject* holds[usearch_py_holds_k];

    /** Buffers the library reads until the call is final. */
    usearch_py_buffer_backing_t buffers[usearch_py_buffers_k];

    /** Keys the library reads until the call is final. */
    usearch_py_keys_t keys[usearch_py_keys_k];

    /** Paths the library reads until the call is final. */
    usearch_py_paths_t paths;

    /** Blocks the library reads until the call is final, freed with @c PyMem_Free. */
    void* blocks[usearch_py_blocks_k];

    /** The verb's C arguments, followed by any outputs @c answer converts. */
    uint64_t arguments[usearch_py_arguments_words_k];
} usearch_py_call_t;

/** A call a @c DeferredOn submitted, the object its allocation, so nothing it lends ever moves. */
typedef struct Deferred {
    PyObject ob_base;

    /** The receiver the call was made through, held so its index and executor outlive the call. */
    PyObject* origin;

    /** Nonzero once a thread took the call to answer it, and let go of its inputs. */
    atomic_int answering;

    /** The answer, once the call finished without raising. */
    _Atomic(PyObject*) answer;

    /** The exception, once the call finished raising it. */
    _Atomic(PyObject*) raised;

    usearch_py_call_t call;
} Deferred;

/**
 *  @brief Opens a call: the zeroed @p stack for a blocking receiver, or the record of a new
 *      @c Deferred holding @p receiver for a deferred one.
 *  @return The call to fill, or NULL with an exception set.
 */
usearch_py_call_t* usearch_py_call_open(usearch_py_receiver_t const* receiver, usearch_py_call_t* stack);

/** Answers a submitted @p call: a blocking one's answer or exception, or its @c Deferred. */
PyObject* usearch_py_call_finish(usearch_py_receiver_t const* receiver, usearch_py_call_t* call);

/** Lets go of a call whose fill failed, keeping the exception set; always returns NULL. */
PyObject* usearch_py_call_discard(usearch_py_receiver_t const* receiver, usearch_py_call_t* call);

/**
 *  @brief Reads an optional callable, holding it in @p callable and leaving NULL or None alone.
 *  @return 0 on success, -1 with a @c TypeError set when @p object is neither callable nor None.
 */
int usearch_py_callable_from_object(PyObject* object, char const* field, PyObject** callable);

/** The progress to register for @p callable and @p steps, the callable its state; NULL for none. */
usearch_progress_t usearch_py_progress_registered(PyObject* callable, size_t steps);

/** The predicate to register for @p callable, the callable its state; no trampoline for NULL. */
usearch_predicate_t usearch_py_predicate_registered(PyObject* callable);

#pragma endregion Calls

#pragma region Shared Helpers

/**
 *  @brief Raises the class @p outcome 's status maps to when it carries a failure.
 *  @return -1 with an exception set when @p outcome failed, 0 otherwise.
 */
int usearch_py_raise_on_outcome(usearch_outcome_t const* outcome);

/**
 *  @brief Stores @p value under @p key in @p dict and drops the caller's reference to it, whether or not the store
 *      succeeds - the step a hand-built result dict repeats per field, in place of a @c Py_BuildValue format.
 *  @return -1 with an exception set when @p value is null or the store failed, 0 otherwise.
 */
int usearch_py_dict_steal_item(PyObject* dict, char const* key, PyObject* value);

/** Raises the @c Error subclass @p status maps to, with a formatted message, its @c status a @c Status member;
 *  always returns -1. */
int usearch_py_raise(usearch_status_t status, char const* format, ...);

/**
 *  @brief Raises @p kind with a formatted message, keeping whatever is already raised as its
 *      cause - the exporter's own diagnosis is more precise than any guess we could make.
 *  @return -1 with an exception set, always.
 */
int usearch_py_raise_from_cause(PyObject* kind, char const* format, ...);

/**
 *  @brief Reads one identity from an @c int, @c str, @c bytes, @c uuid.UUID, or a @c Key.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_key_from_object(PyObject* object, usearch_key_t* key);

/** Wraps sixteen identity bytes into a new @c Key; returns a new reference, or @c NULL. */
PyObject* usearch_py_key_to_object(usearch_key_t key);

/** Builds a new @c Match; returns a new reference, or @c NULL. */
PyObject* usearch_py_match_to_object(usearch_key_t key, usearch_distance_t distance);

/** A new struct sequence of @p type taking every one of @p fields, or NULL dropping them all when any is NULL. */
PyObject* usearch_py_struct_sequence_steal(PyTypeObject* type, PyObject** fields, Py_ssize_t count);

/**
 *  @brief The scalar kind @p name spells, accepting the canonical names alongside NumPy
 *      typestrings, dtype names and PEP 3118 format characters.
 *  @return The kind, or @c usearch_scalar_unknown_k when unrecognized.
 */
usearch_scalar_t usearch_py_scalar_from_spelling(char const* name, Py_ssize_t length);

/** Bytes one row of @p columns scalars of @p kind occupies, sub-byte packing included. */
size_t usearch_py_scalar_row_bytes(usearch_scalar_t kind, size_t columns);

/**
 *  @brief The @c memoryview format character for @p kind, or NULL when packing forbids one.
 *
 *  Sub-byte kinds carry a format in the naming table but pack several dimensions per byte, so
 *  a view cast to it would report the wrong extent; those leave as raw bytes instead.
 */
char const* usearch_py_scalar_buffer_format(usearch_scalar_t kind);

/**
 *  @brief Bytes a @p rows × @p columns matrix of @p unit -byte cells occupies.
 *
 *  Every ranked output goes through here: @c wanted is an arbitrary caller-supplied count, and
 *  the product silently wraps at a size that would then under-allocate the arena the engine
 *  writes into.
 *
 *  @return 0 on success, -1 with an exception set when the extent does not fit a @c Py_ssize_t .
 */
int usearch_py_matrix_bytes(size_t rows, size_t columns, size_t unit, Py_ssize_t* bytes);

/**
 *  @brief Reads a @c Scalar member, or NULL or @c None to mean "not stated".
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_scalar_from_object(PyObject* object, char const* field, usearch_scalar_t* kind);

/**
 *  @brief Borrows a contiguous-rowed matrix, PEP 3118 first, @c __array_interface__ second, and
 *      an Arrow column third.
 *
 *  A one-dimensional input is read as a single row, matching the C surface's rule that one
 *  vector is a one-row matrix. Only the row stride may vary; a strided column would force a
 *  gather the SIMD kernels have no form for. The view reads and never writes, so an exporter
 *  that forbids writing is accepted.
 *
 *  @param[in] object The tensor-like input.
 *  @param[in] scalar_hint Names the scalar kind; unknown infers it from the buffer, and a named one
 *      must be as wide as the buffer's own items.
 *  @param[out] view The borrowed matrix, whose @c columns counts logical dimensions rather than
 *      the storage units a sub-byte kind packs them into.
 *  @param[out] backing Storage the view points into; release it with @c usearch_py_buffer_release.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_matrix_from_object(PyObject* object, usearch_scalar_t scalar_hint, usearch_matrix_view_t* view,
                                  usearch_py_buffer_backing_t* backing);

/**
 *  @brief Borrows a writable contiguous byte run of exactly @p bytes for state the caller lends a
 *      verb, such as attention's carry.
 *
 *  The state is a byte run rather than a matrix, so this validates the total extent instead of the
 *  shape.
 *
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_output_from_object(PyObject* object, char const* field, Py_ssize_t bytes,
                                  usearch_py_buffer_backing_t* backing, void** data);

/** Releases whatever @c usearch_py_matrix_from_object acquired; a zeroed backing is a no-op. */
void usearch_py_buffer_release(usearch_py_buffer_backing_t* backing);

/**
 *  @brief Gathers @p expected identities from a buffer of 16-byte rows, an Arrow column of 16-byte
 *      keys or integers, or a sequence of keys.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_keys_from_object(PyObject* object, size_t expected, usearch_py_keys_t* batch);

/** Releases whatever @c usearch_py_keys_from_object acquired; a zeroed batch is a no-op. */
void usearch_py_keys_release(usearch_py_keys_t* batch);

/**
 *  @brief Borrows a filesystem path from a @c str, @c bytes, or @c os.PathLike.
 *  @param[out] decoded A new reference the caller must release once @p path is spent.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_path_from_object(PyObject* object, usearch_bytes_view_t* path, PyObject** decoded);

/** Readies and registers @c ErrorType and one subclass per failing status on @p module; returns 0 or -1. */
int usearch_py_register_error_types(PyObject* module);

/**
 *  @brief Reads a sequence of NUMA node IDs or device ordinals into a caller-freed array.
 *
 *  Residency variants borrow their arrays only for the duration of the factory call, so the
 *  allocation lives no longer than the constructor that made it.
 *
 *  @param[out] ordinals Receives a @c PyMem_Free -able array; @c NULL when @p object is NULL or @c None.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_ordinals_from_object(PyObject* object, char const* field, int32_t** ordinals, size_t* count);

/**
 *  @brief Reads a sequence of filesystem or transport identifiers a residency arm borrows.
 *
 *  A path arrives as text and the ABI reads bytes, so the encoded objects have to outlive the
 *  call; they trail the views in the same block, since the ABI reads the views contiguously.
 *
 *  @param[out] paths The run to hand the arm; release it with @c usearch_py_paths_release.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_paths_from_object(PyObject* object, usearch_py_paths_t* paths);

/** Releases the encoded paths and the block they share. */
void usearch_py_paths_release(usearch_py_paths_t* paths);

/** Reads a non-negative count, leaving @p value alone when @p object is NULL; returns 0 or -1 with an exception set. */
int usearch_py_size_from_object(PyObject* object, char const* field, size_t* value);

/** Reads an optional seed, leaving @p seed alone when @p object is NULL; returns 0 or -1 with an
 *  exception set. */
int usearch_py_seed_from_object(PyObject* object, uint64_t* seed);

/** Reads the class a placement pins, a member of @p classes, leaving zero for NULL or None; returns
 *  0 or -1 with an exception set. */
int usearch_py_class_from_object(PyObject* object, char const* field, usearch_py_enum_t const* classes, int* value);

/** A new member of @p classes for @p value, or None for the zero value that means unknown. */
PyObject* usearch_py_class_to_object(usearch_py_enum_t const* classes, int value);

/** Reads the @c size_t at the byte offset @p closure inside @p self, as a report's getters do. */
PyObject* usearch_py_size_getter(PyObject* self, void* closure);

/** One reachability audit, the slot @c health writes it into. */
typedef struct DenseReachability {
    PyObject ob_base;
    usearch_reachability_t reachability;
} DenseReachability;

/** Traversal telemetry a caller passes into a walking verb of any modality to have overwritten. */
typedef struct DenseTraversalStats {
    PyObject ob_base;
    usearch_traversal_stats_t stats;
} DenseTraversalStats;

/** What one refine or reshard did to the graph, the slot the call writes it into. */
typedef struct RefineReport {
    PyObject ob_base;
    usearch_refine_report_t report;
} RefineReport;

/** A new @c DenseReachability of @p reachability, or NULL with an exception set. */
PyObject* usearch_py_reachability_to_object(usearch_reachability_t const* reachability);

/** A new @c RefineReport of @p report, or NULL with an exception set. */
PyObject* usearch_py_refine_report_to_object(usearch_refine_report_t const* report);

/** Borrows a whole checkpoint blob, whatever its length; returns 0 or -1 with an exception set. */
int usearch_py_checkpoint_from_object(PyObject* object, usearch_py_buffer_backing_t* backing);

/**
 *  @brief The handle of the @c Executor @p object, refusing anything else and a closed one.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_executor_from_object(PyObject* object, usearch_executor_t* executor);

/** Reads an optional executor, NULL or @c None running on the calling thread; returns 0 or -1 with
 *  an exception set. */
int usearch_py_optional_executor_from_object(PyObject* object, usearch_executor_t* executor);

/** Asks the library where the outputs of the @c Arena @p arena live; 0, or -1 with an exception. */
int usearch_py_arena_describe(PyObject* arena, usearch_arena_description_t* description);

/** The handle of the @c Arena @p arena, or NULL for none, which the library refuses. */
usearch_arena_t usearch_py_arena_handle(PyObject* arena);

/** The handle of the @c Arena @p object, or NULL for None; 0, or -1 with a @c TypeError set. */
int usearch_py_arena_from_object(PyObject* object, usearch_arena_t* handle);

/** Holds the @c Arena @p object in @p call, or none for None; 0, or -1 with a @c TypeError set. */
int usearch_py_call_hold_arena(usearch_py_call_t* call, PyObject* object);

/**
 *  @brief Holds the @c Arena @p object in @p call, or none for None, and opens its result of
 *      @p type over it, which leads with its arena as @c ArenaBuffer does.
 *  @return The result the library writes into, or NULL with an exception set.
 */
void* usearch_py_call_arena(usearch_py_call_t* call, PyObject* object, PyTypeObject* type);

/** Opens the result of @p type, a report the library writes into, as the answer of @p call; NULL
 *  with an exception set. */
void* usearch_py_call_result(usearch_py_call_t* call, PyTypeObject* type);

/** One output a call left in an arena, exported zero-copy, and the base of every such result. */
typedef struct ArenaBuffer {
    PyObject ob_base;

    /** The arena the output lives in, kept alive so the memory outlives every other name for it. */
    PyObject* arena;

    /** The rows a call writes into a @c DenseVectors, which carry its scalar and columns; zeroed
     *  for every other output. */
    usearch_rows_view_t rows;

    /** The rows a call wrote into @c rows. */
    size_t count;

    void const* data;
    char const* format;
    Py_ssize_t item_bytes;
    int rank;
    Py_ssize_t shape[3];
    Py_ssize_t strides[3];
} ArenaBuffer;

/** A ranked run a verb writes into an arena, the slot the library writes its view into. */
typedef struct Matches {
    PyObject ob_base;
    PyObject* arena;
    usearch_matches_view_t view;

    /** Rows in each output, one per query. */
    size_t queries;
} Matches;

/** The ragged pairs @c multiply or @c maxsim writes into an arena: its view and counts. */
typedef struct DenseProducts {
    PyObject ob_base;
    PyObject* arena;
    usearch_matches_tape_view_t view;

    /** The left rows or queries, one fewer than the offsets. */
    size_t queries;

    /** Pairs written across them. */
    size_t pairs;

    /** Pairs scored before the bound cut any. */
    size_t required;
} DenseProducts;

/** Answers the @c DenseVectors a call wrote its @c rows and @c count into, laid out for export. */
PyObject* usearch_py_rows_answer(usearch_py_call_t* call);

/**
 *  @brief A result of @p type - @c ArenaBufferType or a subtype - over the C-contiguous @p data a
 *      call wrote inside @p arena, which it keeps alive and views until a later call rewinds it.
 *
 *  Every output exports DLPack; host memory also exports the writable buffer protocol, and device
 *  memory @c __cuda_array_interface__.
 *
 *  @return A new reference, or @c NULL with an exception set.
 */
PyObject* usearch_py_arena_buffer(PyTypeObject* type, PyObject* arena, void const* data, char const* format,
                                  Py_ssize_t item_bytes, Py_ssize_t const* shape, int rank);

/**
 *  @brief A @c DenseVectors over the @p count rows a call wrote packed into the @c Arena @p arena.
 *
 *  Kinds a @c memoryview cannot format - the sub-byte and mini-float ones - leave as a
 *  @b (count,row_bytes) run of @c 'B', since a cast to their format would report the wrong extent.
 *
 *  @return A new reference, or NULL with an exception set.
 */
PyObject* usearch_py_rows_new(PyObject* arena, usearch_rows_view_t const* rows, size_t count);

/** A new @c Matches over the @p queries a ranked verb wrote into @p arena; NULL with an exception
 *  set. */
PyObject* usearch_py_matches_new(PyObject* arena, usearch_matches_view_t const* matches, size_t queries);

/** A new @c DenseProducts over the @p pairs across @p queries left rows or queries @c multiply or
 *  @c maxsim wrote into @p arena, and the pairs it @p required; NULL with an exception set. */
PyObject* usearch_py_products_new(PyObject* arena, usearch_matches_tape_view_t const* products, size_t queries,
                                  size_t pairs, size_t required);

extern char const doc_version[];
extern char const doc_capabilities_compiled[];
extern char const doc_capabilities_available[];

PyObject* api_version(PyObject* self, PyObject* unused);
PyObject* api_capabilities_compiled(PyObject* self, PyObject* unused);
PyObject* api_capabilities_available(PyObject* self, PyObject* unused);

#pragma endregion Shared Helpers

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_PYTHON_LIB_H
