/**
 *  @brief Declarations shared by every USearch CPython translation unit.
 *  @file python/lib.h
 *  @author Ash Vardanian
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
 *  itself is never a build-time dependency. Outputs leave as @c memoryview objects over
 *  arenas the result owns, which NumPy wraps with no copy.
 */
#ifndef UNUM_USEARCH_PYTHON_LIB_H
#define UNUM_USEARCH_PYTHON_LIB_H

#include <stddef.h> // `size_t`
#include <stdint.h> // `uint8_t`, `PTRDIFF_MAX`
#include <string.h> // `memcpy`, `memset`

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <usearch/dense_index.h>
#include <usearch/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Calling Convention

/** @brief Compile-time constants of the binding layer. */
enum {
    /** @brief Fast-call plus keywords, the calling convention every batch verb uses. */
    usearch_py_method_flags_k = METH_FASTCALL | METH_KEYWORDS,
    /** @brief The rank ceiling every batch verb accepts: a matrix, or a vector read as one row. */
    usearch_py_max_rank_k = 2,
};

/** @brief @p value divided by @p divisor, rounded up - the packing arithmetic sub-byte kinds need. */
static inline size_t usearch_py_divide_round_up_(size_t value, size_t divisor) {
    return value / divisor + (value % divisor != 0);
}

/** @brief Element count of a statically-sized table, which every naming table asks for. */
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

/**
 *  @brief A borrowed matrix with contiguous rows - what every batch verb accepts.
 *
 *  A one-dimensional input is read as a single row, matching the C surface's rule that one
 *  vector is a one-row matrix. Only the row pitch may stride; a strided column would force
 *  a gather the SIMD kernels have no form for.
 */
typedef struct usearch_matrix_view_t {
    /** @brief Pointer to the first scalar of the first row. */
    void* data;
    /** @brief Number of rows; one for a vector input. */
    size_t rows;
    /** @brief Number of logical scalars per row. */
    size_t columns;
    /** @brief Bytes between the starts of consecutive rows. */
    size_t pitch;
    /** @brief The scalar type the rows hold. */
    usearch_scalar_t scalar_kind;
    /** @brief Whether the exporter forbids writes. */
    int readonly;
} usearch_matrix_view_t;

/**
 *  @brief Backing storage for a synthesized @c Py_buffer, so the @c __array_interface__
 *      fallback never allocates. Pair one with every @c usearch_matrix_from_object call and
 *      release it with @c usearch_matrix_release once the call has returned.
 */
typedef struct usearch_buffer_backing_t {
    /** @brief The buffer, exported by PEP 3118 or synthesized from @c __array_interface__ . */
    Py_buffer buffer;
    /** @brief Per-axis extents the synthesized buffer points at; unused on the PEP 3118 path. */
    Py_ssize_t shape[usearch_py_max_rank_k];
    /** @brief Per-axis byte strides the synthesized buffer points at; likewise unused. */
    Py_ssize_t strides[usearch_py_max_rank_k];
    /** @brief The exporter a synthesized buffer borrows from, kept alive; NULL otherwise. */
    PyObject* synthesized_owner;
} usearch_buffer_backing_t;

/** @brief What a borrowed matrix will be used for, and therefore what the exporter must permit. */
typedef enum usearch_matrix_access_t {
    usearch_matrix_access_read_k = 0,
    usearch_matrix_access_write_k = 1,
} usearch_matrix_access_t;

/** @brief Where a key batch's storage came from, and therefore how it is released. */
typedef enum usearch_key_storage_t {
    /** @brief One allocation this arena made, freed with @c PyMem_Free . */
    usearch_key_storage_owned_k = 0,
    /** @brief An @c (N,16) buffer the caller exported, released with @c PyBuffer_Release . */
    usearch_key_storage_borrowed_k = 1,
} usearch_key_storage_t;

/**
 *  @brief Identities gathered for one batch, and whatever storage that gathering needed.
 *
 *  A caller passing an @c (N,16) buffer of identities hands us the array outright and
 *  @c owned stays @c NULL; a caller passing a list of integers gets one allocation, freed by
 *  @c usearch_keys_release.
 */
typedef struct usearch_key_arena_t {
    /** @brief The identities, aliasing whichever storage @c storage names. */
    usearch_key_t* keys;
    /** @brief The allocation backing @c keys when @c storage is owned; NULL otherwise. */
    usearch_key_t* owned;
    /** @brief The exported buffer backing @c keys when @c storage is borrowed. */
    usearch_buffer_backing_t backing;
    /** @brief Which of the two above @c keys points into. */
    usearch_key_storage_t storage;
} usearch_key_arena_t;

#pragma endregion Buffer Views

#pragma region Shared Types

extern PyTypeObject KeyType;
extern PyTypeObject StatsType;
extern PyTypeObject MatchType;
extern PyTypeObject MatchesType;
extern PyTypeObject DenseIndexType;
extern PyTypeObject DenseCursorType;
extern PyTypeObject DenseFinderType;
extern PyTypeObject SequenceIndexType;
extern PyTypeObject MatrixIndexType;
extern PyTypeObject GeospatialIndexType;

/** @brief The one exception every failing verb raises, subclassing @c RuntimeError. */
extern PyObject* ErrorType;

/** @brief Traversal telemetry a caller may pass into @c add or @c search to accumulate. */
typedef struct Stats {
    PyObject ob_base;
    usearch_dense_traversal_stats_t stats;
} Stats;

/** @brief One scored entry, as a finder yields them and as a @c Matches row indexes them. */
typedef struct Match {
    PyObject ob_base;
    usearch_key_t key;
    usearch_distance_t distance;
} Match;

/**
 *  @brief The result of a batch search: three arenas and the telemetry that produced them.
 *
 *  The arenas are filled before the object is ever visible and never mutated afterwards, so
 *  the @c memoryview objects handed out stay valid for the result's whole lifetime.
 */
typedef struct Matches {
    PyObject ob_base;
    /** @brief `bytes` of `queries * wanted` keys, sixteen bytes each. */
    PyObject* keys_arena;
    /** @brief `bytes` of `queries * wanted` doubles. */
    PyObject* distances_arena;
    /** @brief `bytes` of `queries` counts. */
    PyObject* counts_arena;
    /** @brief Rows in each arena - one per query. */
    size_t queries;
    /** @brief Columns in the key and distance arenas - the "k" the search was asked for. */
    size_t wanted;
    /** @brief Telemetry of the walk that filled the arenas. */
    usearch_dense_traversal_stats_t stats;
} Matches;

#pragma endregion Shared Types

#pragma region Shared Helpers

/**
 *  @brief Raises the mapped exception when @p error carries a failure.
 *  @return -1 with an exception set when @p error failed, 0 otherwise.
 */
int usearch_raise_on_error(usearch_error_t const* error);

/** @brief Raises @c ErrorType with @p status and @p message; always returns -1. */
int usearch_raise(usearch_status_t status, char const* message);

/** @brief The lowercase, dash-separated spelling of @p status, as @c Error.status reports it. */
char const* usearch_status_name(usearch_status_t status);

/**
 *  @brief Reads one identity from an @c int, @c str, @c bytes, @c uuid.UUID, or a @c Key.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_key_from_object(PyObject* object, usearch_key_t* key);

/** @brief Wraps sixteen identity bytes into a new @c Key; returns a new reference, or @c NULL. */
PyObject* usearch_key_to_object(usearch_key_t key);

/** @brief Builds a new @c Stats holding @p stats; returns a new reference, or @c NULL. */
PyObject* usearch_stats_to_object(usearch_dense_traversal_stats_t const* stats);

/**
 *  @brief Folds @p stats into the @c Stats object @p target, which may be @c None.
 *  @return 0 on success, -1 with an exception set when @p target is neither.
 */
int usearch_stats_accumulate(PyObject* target, usearch_dense_traversal_stats_t const* stats);

/** @brief Builds a new @c Match; returns a new reference, or @c NULL. */
PyObject* usearch_match_to_object(usearch_key_t key, usearch_distance_t distance);

/**
 *  @brief The scalar kind @p name spells, accepting NumKong dtype names, NumPy typestrings,
 *      and PEP 3118 format characters alike.
 *  @return The kind, or @c usearch_scalar_unknown_k when unrecognized.
 */
usearch_scalar_t usearch_scalar_from_name(char const* name, Py_ssize_t length);

/** @brief The canonical name of @p kind, as the constructor and the stubs spell it. */
char const* usearch_scalar_name(usearch_scalar_t kind);

/** @brief Bytes one row of @p columns scalars of @p kind occupies, sub-byte packing included. */
size_t usearch_scalar_row_bytes(usearch_scalar_t kind, size_t columns);

/**
 *  @brief The @c memoryview format character for @p kind, or NULL when packing forbids one.
 *
 *  Sub-byte kinds carry a format in the naming table but pack several dimensions per byte, so
 *  a view cast to it would report the wrong extent; those leave as raw bytes instead.
 */
char const* usearch_scalar_buffer_format(usearch_scalar_t kind);

/**
 *  @brief A read-only @c (rows,columns) view over @p arena in @p kind 's own format.
 *
 *  Kinds a @c memoryview cannot format - the sub-byte and mini-float ones - leave as a
 *  @c (rows,row_bytes) run of @c 'B' instead, since a cast to their format would report the
 *  wrong extent. Returns a new reference, or @c NULL with an exception set.
 */
PyObject* usearch_arena_matrix(PyObject* arena, size_t rows, size_t columns, usearch_scalar_t kind);

/**
 *  @brief Bytes a @p rows × @p columns matrix of @p unit -byte cells occupies.
 *
 *  Every output extent goes through here: @c wanted is an arbitrary caller-supplied count, and
 *  the product silently wraps at a size that would then under-allocate the arena the engine
 *  writes into.
 *
 *  @return 0 on success, -1 with an exception set when the extent does not fit a @c Py_ssize_t .
 */
int usearch_matrix_bytes(size_t rows, size_t columns, size_t unit, Py_ssize_t* bytes);

/**
 *  @brief Reads a scalar kind from a Python @c str, or from @c None to mean "not stated".
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_scalar_from_object(PyObject* object, usearch_scalar_t* kind);

/**
 *  @brief Borrows a contiguous-rowed matrix, PEP 3118 first, @c __array_interface__ second.
 *  @param[in] object The tensor-like input.
 *  @param[in] access What the caller intends to do through the view.
 *  @param[in] scalar_hint Overrides the inferred scalar kind; unknown infers from the buffer.
 *  @param[out] view The borrowed matrix.
 *  @param[out] backing Storage the view points into; release it with @c usearch_matrix_release.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_matrix_from_object(PyObject* object, usearch_matrix_access_t access, usearch_scalar_t scalar_hint,
                               usearch_matrix_view_t* view, usearch_buffer_backing_t* backing);

/**
 *  @brief Borrows a contiguous byte run of exactly @p bytes bytes for a verb to write into.
 *
 *  Outputs are byte runs rather than matrices - a key output is rank three - so this validates
 *  the total extent instead of the shape.
 *
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_buffer_borrow_exact(PyObject* object, char const* field, usearch_matrix_access_t access, Py_ssize_t bytes,
                                usearch_buffer_backing_t* backing, void** data);

/** @brief Releases whatever @c usearch_matrix_from_object acquired. */
void usearch_matrix_release(usearch_buffer_backing_t* backing);

/**
 *  @brief Gathers @p expected identities from a buffer of 16-byte rows or a sequence of keys.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_keys_from_object(PyObject* object, size_t expected, usearch_key_arena_t* arena);

/** @brief Releases whatever @c usearch_keys_from_object acquired. */
void usearch_keys_release(usearch_key_arena_t* arena);

/**
 *  @brief Borrows a filesystem path from a @c str, @c bytes, or @c os.PathLike.
 *  @param[out] decoded A new reference the caller must release once @p path is spent.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
int usearch_path_from_object(PyObject* object, usearch_string_t* path, PyObject** decoded);

/** @brief Whether @p object exports a buffer, distinguishing a checkpoint blob from a path. */
int usearch_object_is_buffer(PyObject* object);

/** @brief Readies and registers @c ErrorType on @p module; returns 0 or -1. */
int usearch_register_error_type(PyObject* module);

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
