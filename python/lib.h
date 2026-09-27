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
 *  cannot use the fast calling convention. Every verb is @c METH_FASTCALL | @c METH_KEYWORDS
 *  and walks its keywords by hand.
 *
 *  @section python_buffers Buffer Protocol and NumPy Compatibility
 *
 *  Batch inputs arrive through PEP 3118 first and @c __array_interface__ second, so a
 *  PyTorch or TensorFlow tensor reaches the index without a NumPy round-trip and NumPy
 *  itself is never a build-time dependency. Outputs are buffers the caller already owns and
 *  a verb fills, and a @c Matches reads them back as @c memoryview objects.
 */
#ifndef UNUM_USEARCH_PYTHON_LIB_H
#define UNUM_USEARCH_PYTHON_LIB_H

/*  CPython defines preprocessor macros that reach into the standard headers, so it leads rather
 *  than joining the third-party group; every unit here includes this file before anything else. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdarg.h> // `va_list` - the cause-preserving raise
#include <stddef.h> // `size_t`
#include <stdint.h> // `uint8_t`, `PTRDIFF_MAX`
#include <string.h> // `memcpy`, `memset`

#include <usearch/dense_index.h>
#include <usearch/types.h>

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
};

/** @p value divided by @p divisor, rounded up - the packing arithmetic sub-byte kinds need. */
static inline size_t usearch_py_divide_round_up_(size_t value, size_t divisor) {
    return value / divisor + (value % divisor != 0);
}

/** Element count of a statically-sized table, which every naming table asks for. */
#define usearch_py_table_size_(table) (sizeof(table) / sizeof((table)[0]))

/**
 *  @brief Guards a mutable object's fields against a concurrent verb.
 *
 *  Free-threading, and the critical sections that fence it, arrived in 3.13; on 3.12 the
 *  interpreter lock already serializes what these protect, so they compile to the bare block.
 */
#if PY_VERSION_HEX >= 0x030D0000
#define usearch_py_critical_section_begin_(object) Py_BEGIN_CRITICAL_SECTION(object)
#define usearch_py_critical_section_end_()         Py_END_CRITICAL_SECTION()
#else
#define usearch_py_critical_section_begin_(object) do {
#define usearch_py_critical_section_end_() \
    }                                      \
    while (0)
#endif

#pragma endregion Calling Convention

#pragma region Buffer Views

/** Backing storage for a synthesized @c Py_buffer, so the @c __array_interface__
 *  fallback never allocates. Pair one with every @c usearch_py_matrix_from_object call and
 *  release it with @c usearch_py_buffer_release once the call has returned. */
typedef struct usearch_py_buffer_backing_t {

    /** The buffer, exported by PEP 3118 or synthesized from @c __array_interface__ . */
    Py_buffer buffer;

    /** Per-axis extents the synthesized buffer points at; unused on the PEP 3118 path. */
    Py_ssize_t shape[usearch_py_max_rank_k];

    /** Per-axis byte strides the synthesized buffer points at; likewise unused. */
    Py_ssize_t strides[usearch_py_max_rank_k];

    /** The exporter a synthesized buffer borrows from, kept alive; NULL otherwise. */
    PyObject* synthesized_owner;
} usearch_py_buffer_backing_t;

/** What a borrowed byte run will be used for, and therefore what the exporter must permit. */
typedef enum usearch_py_matrix_access_t {
    usearch_py_matrix_access_read_k = 0,
    usearch_py_matrix_access_write_k = 1,
} usearch_py_matrix_access_t;

/** Where a key batch's storage came from, and therefore how it is released. */
typedef enum usearch_py_key_storage_t {

    /** One allocation this arena made, freed with @c PyMem_Free . */
    usearch_py_key_storage_owned_k = 0,

    /** An @c (N,16) buffer the caller exported, released with @c PyBuffer_Release . */
    usearch_py_key_storage_borrowed_k = 1,
} usearch_py_key_storage_t;

/**
 *  @brief Identities gathered for one batch, and whatever storage that gathering needed.
 *
 *  A caller passing an @c (N,16) buffer of identities hands us the array outright and
 *  @c owned stays @c NULL; a caller passing a list of integers gets one allocation, freed by
 *  @c usearch_py_keys_release.
 */
typedef struct usearch_py_key_arena_t {

    /** The identities, aliasing whichever storage @c storage names. */
    usearch_key_t* keys;

    /** The allocation backing @c keys when @c storage is owned; NULL otherwise. */
    usearch_key_t* owned;

    /** The exported buffer backing @c keys when @c storage is borrowed. */
    usearch_py_buffer_backing_t backing;

    /** Which of the two above @c keys points into. */
    usearch_py_key_storage_t storage;
} usearch_py_key_arena_t;

/**
 *  @brief The paths a residency arm borrows for the length of one factory call.
 *
 *  A path arrives as text and the ABI reads bytes, so the encoded objects have to outlive the
 *  call. The ABI reads the views contiguously, so the objects cannot ride inside them and
 *  trail them in the same block instead - one allocation, released as one.
 */
typedef struct usearch_py_paths_arena_t {

    /** The run the arm reads, or NULL when the caller named no paths. */
    usearch_bytes_view_t* views;

    /** Paths the arena holds; zero when the caller named none. */
    size_t count;
} usearch_py_paths_arena_t;

#pragma endregion Buffer Views

#pragma region Shared Types

extern PyTypeObject KeyType;
extern PyTypeObject StatsType;
extern PyTypeObject MatchType;
extern PyTypeObject MatchesType;
extern PyTypeObject ExecutorType;

extern PyTypeObject DenseIndexType;
extern PyTypeObject DenseCursorType;
extern PyTypeObject DenseFinderType;

extern PyTypeObject SequenceIndexType;
extern PyTypeObject MatrixIndexType;
extern PyTypeObject GeospatialIndexType;

/** The one exception every failing verb raises, subclassing @c RuntimeError. */
extern PyObject* ErrorType;

/** The workers a fanned-out verb runs on, owned apart from any index so one pool serves many. */
typedef struct Executor {
    PyObject ob_base;
    usearch_executor_t handle;
} Executor;

/** Traversal telemetry a caller may pass into @c add or @c search to have overwritten. */
typedef struct Stats {
    PyObject ob_base;
    usearch_dense_traversal_stats_t stats;
} Stats;

/** One scored entry, as a finder yields them and as a @c Matches row indexes them. */
typedef struct Match {
    PyObject ob_base;
    usearch_key_t key;
    usearch_distance_t distance;
} Match;

/**
 *  @brief A ranked run read back off the three buffers a verb filled.
 *
 *  Every field borrows: the exported buffers keep their owners alive for the view's whole
 *  life, and the row width follows from their lengths rather than travelling beside them.
 */
typedef struct Matches {
    PyObject ob_base;

    /** The `queries * wanted` identities, sixteen bytes each. */
    usearch_py_buffer_backing_t keys;

    /** The `queries * wanted` distances, positionally paired with the identities. */
    usearch_py_buffer_backing_t distances;

    /** The `queries` match counts, one per query. */
    usearch_py_buffer_backing_t counts;

    /** Rows in each buffer - one per query. */
    size_t queries;

    /** Columns in the key and distance buffers - the "k" in "kANN". */
    size_t wanted;
} Matches;

#pragma endregion Shared Types

#pragma region Shared Helpers

/**
 *  @brief Raises the mapped exception when @p error carries a failure.
 *  @return -1 with an exception set when @p error failed, 0 otherwise.
 */
int usearch_py_raise_on_error(usearch_error_t const* error);

/**
 *  @brief Stores @p value under @p key in @p dict and drops the caller's reference to it, whether or not the store
 *      succeeds - the step a hand-built result dict repeats per field, in place of a @c Py_BuildValue format.
 *  @return -1 with an exception set when @p value is null or the store failed, 0 otherwise.
 */
int usearch_py_dict_steal_item(PyObject* dict, char const* key, PyObject* value);

/** Raises @c ErrorType with @p status and a formatted message; always returns -1. */
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

/** Builds a new @c Stats holding @p stats; returns a new reference, or @c NULL. */
PyObject* usearch_py_stats_to_object(usearch_dense_traversal_stats_t const* stats);

/**
 *  @brief Replaces the counters of the @c Stats object @p target with @p stats; @c None skips.
 *  @return 0 on success, -1 with an exception set when @p target is neither.
 */
int usearch_py_stats_overwrite(PyObject* target, usearch_dense_traversal_stats_t const* stats);

/** Builds a new @c Match; returns a new reference, or @c NULL. */
PyObject* usearch_py_match_to_object(usearch_key_t key, usearch_distance_t distance);

/**
 *  @brief The scalar kind @p name spells, accepting the canonical names alongside NumPy
 *      typestrings and PEP 3118 format characters.
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
 *  @brief A read-only @c (rows,columns) view over @p arena in @p kind 's own format.
 *
 *  Kinds a @c memoryview cannot format - the sub-byte and mini-float ones - leave as a
 *  @c (rows,row_bytes) run of @c 'B' instead, since a cast to their format would report the
 *  wrong extent. Returns a new reference, or @c NULL with an exception set.
 */
PyObject* usearch_py_arena_matrix(PyObject* arena, size_t rows, size_t columns, usearch_scalar_t kind);

/**
 *  @brief Bytes a @p rows × @p columns matrix of @p unit -byte cells occupies.
 *
 *  Every output extent goes through here: @c wanted is an arbitrary caller-supplied count, and
 *  the product silently wraps at a size that would then under-allocate the arena the engine
 *  writes into.
 *
 *  @return 0 on success, -1 with an exception set when the extent does not fit a @c Py_ssize_t .
 */
int usearch_py_matrix_bytes(size_t rows, size_t columns, size_t unit, Py_ssize_t* bytes);

/**
 *  @brief Reads a scalar kind from a Python @c str, or from @c None to mean "not stated".
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_scalar_from_object(PyObject* object, usearch_scalar_t* kind);

/**
 *  @brief Borrows a contiguous-rowed matrix, PEP 3118 first, @c __array_interface__ second.
 *
 *  A one-dimensional input is read as a single row, matching the C surface's rule that one
 *  vector is a one-row matrix. Only the row stride may vary; a strided column would force a
 *  gather the SIMD kernels have no form for. The view reads and never writes, which is why it
 *  names no access mode and accepts an exporter that forbids one.
 *
 *  @param[in] object The tensor-like input.
 *  @param[in] scalar_hint Overrides the inferred scalar kind; unknown infers from the buffer.
 *  @param[out] view The borrowed matrix, whose @c columns counts logical dimensions rather than
 *      the storage units a sub-byte kind packs them into.
 *  @param[out] backing Storage the view points into; release it with @c usearch_py_buffer_release.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_matrix_from_object(PyObject* object, usearch_scalar_t scalar_hint, usearch_matrix_view_t* view,
                                  usearch_py_buffer_backing_t* backing);

/**
 *  @brief Borrows a contiguous byte run of exactly @p bytes bytes for a verb to write into.
 *
 *  Outputs are byte runs rather than matrices - a key output is rank three - so this validates
 *  the total extent instead of the shape.
 *
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_buffer_from_object(PyObject* object, char const* field, usearch_py_matrix_access_t access,
                                  Py_ssize_t bytes, usearch_py_buffer_backing_t* backing, void** data);

/** Releases whatever @c usearch_py_matrix_from_object acquired. */
void usearch_py_buffer_release(usearch_py_buffer_backing_t* backing);

/**
 *  @brief Gathers @p expected identities from a buffer of 16-byte rows or a sequence of keys.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_keys_from_object(PyObject* object, size_t expected, usearch_py_key_arena_t* arena);

/** Releases whatever @c usearch_py_keys_from_object acquired. */
void usearch_py_keys_release(usearch_py_key_arena_t* arena);

/**
 *  @brief Borrows a filesystem path from a @c str, @c bytes, or @c os.PathLike.
 *  @param[out] decoded A new reference the caller must release once @p path is spent.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_path_from_object(PyObject* object, usearch_bytes_view_t* path, PyObject** decoded);

/** Whether @p object exports a buffer, distinguishing a checkpoint blob from a path. */
int usearch_py_object_is_buffer(PyObject* object);

/** Readies and registers @c ErrorType on @p module; returns 0 or -1. */
int usearch_py_register_error_type(PyObject* module);

/** One spelling of a closed-set option and the value it selects. */
typedef struct usearch_py_choice_naming_t {

    /** The keyword's spelling, as a caller writes it. */
    char const* name;

    /** The enumerator the spelling selects. */
    int value;
} usearch_py_choice_naming_t;

/**
 *  @brief Reads one of a closed set of names into the value it selects.
 *
 *  The value travels in the table rather than being re-derived from the match position, so
 *  reordering a table cannot silently change what a name means.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_choice_from_object(PyObject* object, char const* field, usearch_py_choice_naming_t const* namings,
                                  size_t count, int* value);

/**
 *  @brief Checks a fast-call's shape and reports how many keywords followed.
 *
 *  Every verb fixes its positional count exactly, so the keyword walk that follows can index
 *  from it without a second bound.
 *
 *  @return The keyword count, or -1 with an exception set when the shape is wrong.
 */
Py_ssize_t usearch_py_args_names_count_checked(char const* verb, Py_ssize_t positional_args_count,
                                               PyObject* args_names_tuple, Py_ssize_t required_positional);

/**
 *  @brief Reads a sequence of NUMA node IDs or device ordinals into a caller-freed array.
 *
 *  Residency arms borrow their arrays only for the duration of the factory call, so the
 *  allocation lives no longer than the constructor that made it.
 *
 *  @param[out] ordinals Receives a @c PyMem_Free -able array; @c NULL when @p object is @c None.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_ordinals_from_object(PyObject* object, char const* field, int32_t** ordinals, size_t* count);

/**
 *  @brief Reads a sequence of filesystem or transport identifiers a residency arm borrows.
 *
 *  A path arrives as text and the ABI reads bytes, so the encoded objects have to outlive the
 *  call; they trail the views in the same block, since the ABI reads the views contiguously.
 *
 *  @param[out] arena The run to hand the arm; release it with @c usearch_py_paths_release.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_py_paths_from_object(PyObject* object, usearch_py_paths_arena_t* arena);

/** Releases the encoded paths and the block they share. */
void usearch_py_paths_release(usearch_py_paths_arena_t* arena);

/** Reads a non-negative count; returns 0 or -1 with an exception set. */
int usearch_py_size_from_object(PyObject* object, char const* field, size_t* value);

/**
 *  @brief Unwraps the @c executor keyword every fanned-out verb takes.
 *  @return 0 on success, -1 with an exception set when @p object is not an @c Executor.
 */
int usearch_py_executor_from_object(PyObject* object, usearch_executor_t* executor);

#pragma endregion Shared Helpers

#pragma region Documentation

extern char const doc_version[];
extern char const doc_capabilities_compiled[];
extern char const doc_capabilities_available[];

PyObject* api_version(PyObject* self, PyObject* unused);
PyObject* api_capabilities_compiled(PyObject* self, PyObject* unused);
PyObject* api_capabilities_available(PyObject* self, PyObject* unused);

#pragma endregion Documentation

#ifdef __cplusplus
}
#endif

#endif // USEARCH_PYTHON_USEARCH_H
