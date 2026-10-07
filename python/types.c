/**
 *  @file python/types.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief The vocabulary every modality shares - the CPython twin of `usearch/types.h`.
 *
 *  Keys, distances, failures, and telemetry live here rather than inside any one modality,
 *  the same split `rust/types.rs` takes: a `Key` produced by a dense search is the very
 *  object a sequence index will accept, because there is one definition and one address.
 *
 *  Tensor ingestion lives here too, being the same kind of shared vocabulary: every batch verb
 *  of every modality takes its input the same way, PEP 3118 first and @c __array_interface__
 *  second, so frameworks whose exotic scalars the buffer protocol cannot spell - a @c bf16
 *  TensorFlow tensor raises "cannot include dtype 'E' in a buffer" - still reach an index
 *  without a NumPy round-trip. The fallback synthesizes a @c Py_buffer over storage the caller
 *  already owns, so neither path allocates.
 */
#include "lib.h"

#pragma region Failures

PyObject* ErrorType = NULL;

/** The class each failing status raises, indexed by status; success and pending never raise. */
static PyObject* status_error_types[usearch_status_invalid_argument_k + 1];

/** One failing status, the dotted name of the class it raises, and what that class documents. */
static struct {
    usearch_status_t status;
    char const* name;
    char const* doc;
} const status_errors[] = {
    {usearch_status_out_of_memory_k, "usearch.OutOfMemoryError",
     "An allocation or reservation could not be satisfied."},
    {usearch_status_shape_mismatch_k, "usearch.ShapeMismatchError",
     "Dimensionality, row counts, or output capacity disagree."},
    {usearch_status_unsupported_layout_k, "usearch.UnsupportedLayoutError",
     "A stride or packing the engine cannot traverse."},
    {usearch_status_unknown_kind_k, "usearch.UnknownKindError",
     "A scalar, metric, or residency this build does not know."},
    {usearch_status_read_only_k, "usearch.ReadOnlyError",
     "A mutating verb on a handle that serves its checkpoint in place."},
    {usearch_status_io_k, "usearch.InputOutputError", "A file could not be opened, read, written, or parsed."},
    {usearch_status_device_k, "usearch.DeviceError",
     "A device could not be selected, or its memory could not be claimed."},
    {usearch_status_malformed_k, "usearch.MalformedError",
     "A representation could not be decoded - a key, a buffer, a checkpoint."},
    {usearch_status_internal_k, "usearch.InternalError",
     "The library reported neither success nor a failure - a defect."},
    {usearch_status_overflow_k, "usearch.CapacityOverflowError",
     "A hard design ceiling was reached; a smaller request cannot help."},
    {usearch_status_uninitialized_k, "usearch.UninitializedError",
     "The handle is not ready for the verb - unfrozen, or consumed by a merge."},
    {usearch_status_config_mismatch_k, "usearch.ConfigMismatchError",
     "The request contradicts an already-frozen configuration."},
    {usearch_status_contended_k, "usearch.ContendedError", "An executor or arena is busy with another call."},
    {usearch_status_cancelled_k, "usearch.CancelledError",
     "A progress handler stopped the call, which published nothing."},
    {usearch_status_unsupported_k, "usearch.UnsupportedError",
     "This residency does not implement the verb; pick another residency."},
    {usearch_status_invalid_argument_k, "usearch.InvalidArgumentError", "An argument breaks the verb's contract."},
};

int usearch_py_raise(usearch_status_t status, char const* format, ...) {
    // A status this specific supersedes whatever the caller was carrying.
    PyErr_Clear();
    PyObject* type = (size_t)status < usearch_py_table_size_(status_error_types) ? status_error_types[status] : NULL;
    if (!type) type = status_error_types[usearch_status_internal_k];
    va_list arguments;
    va_start(arguments, format);
    PyObject* const message = PyUnicode_FromFormatV(format ? format : "", arguments);
    va_end(arguments);
    if (!message) return -1;
    PyErr_SetObject(type, message);
    Py_DECREF(message);
    return -1;
}

int usearch_py_raise_on_outcome(usearch_outcome_t const* outcome) {
    if (outcome->status == usearch_status_success_k) return 0;
    return usearch_py_raise(outcome->status, "%s", outcome->message ? outcome->message : "");
}

int usearch_py_dict_steal_item(PyObject* dict, char const* key, PyObject* value) {
    if (!value) return -1;
    int const stored = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return stored;
}

int usearch_py_raise_from_cause(PyObject* kind, char const* format, ...) {
    PyObject* const cause = PyErr_GetRaisedException();
    va_list arguments;
    va_start(arguments, format);
    PyObject* const message = PyUnicode_FromFormatV(format, arguments);
    va_end(arguments);
    if (!message) {
        Py_XDECREF(cause);
        return -1;
    }

    PyErr_SetObject(kind, message);
    Py_DECREF(message);
    if (cause) {
        // Both setters steal, so the cause needs a second reference before it is handed over twice.
        PyObject* const raised = PyErr_GetRaisedException();
        PyException_SetCause(raised, Py_NewRef(cause));
        PyException_SetContext(raised, cause);
        PyErr_SetRaisedException(raised);
    }
    return -1;
}

static char const doc_Error[] =                                                             //
    "Error(message)\n"                                                                      //
    "\n"                                                                                    //
    "The base of every failure the library reports. Each status shared with the C API\n"    //
    "raises a subclass of its own, such as `ShapeMismatchError`, so a caller catches one\n" //
    "fault by its type or every fault here. A failure caught before the boundary raises\n"  //
    "the class the engine would have raised for the same fault.\n"                          //
    "\n"                                                                                    //
    "Attributes:\n"                                                                         //
    "  status (str): The raised subclass's lowercase, dash-separated status, such as\n"     //
    "    'shape-mismatch'.\n"                                                               //
    "  args (tuple): The detail the status cannot carry, as on any exception.";

int usearch_py_register_error_types(PyObject* module) {
    ErrorType = PyErr_NewExceptionWithDoc("usearch.Error", (char*)doc_Error, PyExc_RuntimeError, NULL);
    if (!ErrorType || PyModule_AddObjectRef(module, "Error", ErrorType) < 0) return -1;
    for (size_t row = 0; row != usearch_py_table_size_(status_errors); ++row) {
        PyObject* const attributes = PyDict_New();
        if (!attributes) return -1;
        if (usearch_py_dict_steal_item(attributes, "status",
                                       PyUnicode_FromString(usearch_status_name(status_errors[row].status))) < 0) {
            Py_DECREF(attributes);
            return -1;
        }
        PyObject* const type = PyErr_NewExceptionWithDoc(status_errors[row].name, (char*)status_errors[row].doc,
                                                         ErrorType, attributes);
        Py_DECREF(attributes);
        if (!type) return -1;
        status_error_types[status_errors[row].status] = type;
        if (PyModule_AddObjectRef(module, status_errors[row].name + sizeof("usearch.") - 1, type) < 0) return -1;
    }
    return 0;
}

#pragma endregion Failures

#pragma region Scalar Names

/** One scalar kind, the spellings the ecosystem gives it, and how it packs in memory. */
typedef struct scalar_naming_t {
    usearch_scalar_t kind;       //! The kind this row describes; @c usearch_scalar_name spells it.
    char const* numpy_typestr;   //! The NumPy array-interface spelling, or NULL when none exists.
    char const* buffer_typestr;  //! The PEP 3118 format string, or NULL when none exists.
    unsigned char values_packed; //! Logical scalars per storage unit; one for byte-wide kinds.
    unsigned char unit_bytes;    //! Bytes one storage unit occupies.
} scalar_naming_t;

/** Ordered by shrinking dynamic range - floats widest exponent first, then signed
 *  integers descending, then unsigned - the order every dtype-indexed list in the
 *  project takes. */
static scalar_naming_t const scalar_namings[] = {
    {usearch_scalar_f64_k, "<f8", "d", 1, 8},   {usearch_scalar_f32_k, "<f4", "f", 1, 4},
    {usearch_scalar_bf16_k, "<V2", NULL, 1, 2}, {usearch_scalar_f16_k, "<f2", "e", 1, 2},
    {usearch_scalar_e5m2_k, "|V1", NULL, 1, 1}, {usearch_scalar_e4m3_k, "|V1", NULL, 1, 1},
    {usearch_scalar_e3m2_k, "|V1", NULL, 1, 1}, {usearch_scalar_e2m3_k, "|V1", NULL, 1, 1},
    {usearch_scalar_i8_k, "|i1", "b", 1, 1},    {usearch_scalar_i4x2_k, "|V1", NULL, 2, 1},
    {usearch_scalar_u8_k, "|u1", "B", 1, 1},    {usearch_scalar_u4x2_k, "|V1", NULL, 2, 1},
    {usearch_scalar_u1x8_k, "|V1", "?", 8, 1},
};

/** Aliases the ecosystem spells differently than we do, resolved after the table. */
static struct {
    char const* name;
    usearch_scalar_t kind;
} const scalar_aliases[] = {
    {"float64", usearch_scalar_f64_k},   {"double", usearch_scalar_f64_k},  {"<f8", usearch_scalar_f64_k},
    {"float32", usearch_scalar_f32_k},   {"float", usearch_scalar_f32_k},   {"<f4", usearch_scalar_f32_k},
    {"bfloat16", usearch_scalar_bf16_k}, {"float16", usearch_scalar_f16_k}, {"half", usearch_scalar_f16_k},
    {"<f2", usearch_scalar_f16_k},       {"int8", usearch_scalar_i8_k},     {"|i1", usearch_scalar_i8_k},
    {"int4", usearch_scalar_i4x2_k},     {"i4", usearch_scalar_i4x2_k},     {"uint8", usearch_scalar_u8_k},
    {"|u1", usearch_scalar_u8_k},        {"uint4", usearch_scalar_u4x2_k},  {"u4", usearch_scalar_u4x2_k},
    {"uint1", usearch_scalar_u1x8_k},    {"u1", usearch_scalar_u1x8_k},     {"b1", usearch_scalar_u1x8_k},
    {"bool", usearch_scalar_u1x8_k},
};

static scalar_naming_t const* scalar_naming(usearch_scalar_t kind) {
    for (size_t naming_index = 0; naming_index != usearch_py_table_size_(scalar_namings); ++naming_index)
        if (scalar_namings[naming_index].kind == kind) return &scalar_namings[naming_index];
    return NULL;
}

static int same_string_sized(char const* input, Py_ssize_t input_length, char const* literal) {
    size_t const literal_length = strlen(literal);
    return (size_t)input_length == literal_length && memcmp(input, literal, literal_length) == 0;
}

usearch_scalar_t usearch_py_scalar_from_spelling(char const* name, Py_ssize_t length) {
    if (!name || length <= 0) return usearch_scalar_unknown_k;
    // The canonical spellings belong to the library, so only what the ecosystem adds is
    // resolved here. A kind this table cannot lay out is no answer, whatever the library said.
    usearch_bytes_view_t const spelling = {(uint8_t const*)name, (size_t)length};
    usearch_scalar_t const canonical = usearch_scalar_from_name(spelling);
    if (scalar_naming(canonical)) return canonical;
    for (size_t naming_index = 0; naming_index != usearch_py_table_size_(scalar_namings); ++naming_index) {
        scalar_naming_t const* const naming = &scalar_namings[naming_index];
        if (naming->buffer_typestr && same_string_sized(name, length, naming->buffer_typestr)) return naming->kind;
        if (naming->numpy_typestr && same_string_sized(name, length, naming->numpy_typestr)) return naming->kind;
    }
    for (size_t alias_index = 0; alias_index != usearch_py_table_size_(scalar_aliases); ++alias_index)
        if (same_string_sized(name, length, scalar_aliases[alias_index].name)) return scalar_aliases[alias_index].kind;
    return usearch_scalar_unknown_k;
}

size_t usearch_py_scalar_row_bytes(usearch_scalar_t kind, size_t columns) {
    scalar_naming_t const* const naming = scalar_naming(kind);
    if (!naming) return 0;
    size_t const units = usearch_py_divide_round_up_(columns, naming->values_packed);
    return units * naming->unit_bytes;
}

char const* usearch_py_scalar_buffer_format(usearch_scalar_t kind) {
    scalar_naming_t const* const naming = scalar_naming(kind);
    return naming && naming->values_packed == 1 ? naming->buffer_typestr : NULL;
}

PyObject* usearch_py_shaped_view(PyObject* exporter, char const* format, Py_ssize_t const* shape, Py_ssize_t rank) {
    PyObject* const exported = PyMemoryView_FromObject(exporter);
    if (!exported) return NULL;
    // A cast crosses formats only through the byte one, so the caller's own format - `d` for a
    // float64 output, say - reaches the shape below in two steps rather than one.
    PyObject* const flat = PyObject_CallMethod(exported, "cast", "s", "B");
    Py_DECREF(exported);
    if (!flat) return NULL;
    int empty = 0;
    for (Py_ssize_t axis = 0; axis != rank; ++axis) empty |= shape[axis] == 0;
    if (empty) {
        PyObject* const typed = PyObject_CallMethod(flat, "cast", "s", format);
        Py_DECREF(flat);
        return typed;
    }
    PyObject* const shape_tuple = PyTuple_New(rank);
    if (!shape_tuple) {
        Py_DECREF(flat);
        return NULL;
    }
    for (Py_ssize_t axis = 0; axis != rank; ++axis) {
        PyObject* const extent = PyLong_FromSsize_t(shape[axis]);
        if (!extent) {
            Py_DECREF(shape_tuple);
            Py_DECREF(flat);
            return NULL;
        }
        PyTuple_SET_ITEM(shape_tuple, axis, extent);
    }
    PyObject* const reshaped = PyObject_CallMethod(flat, "cast", "sO", format, shape_tuple);
    Py_DECREF(shape_tuple);
    Py_DECREF(flat);
    return reshaped;
}

PyObject* usearch_py_arena_matrix(PyObject* arena, size_t rows, size_t columns, usearch_scalar_t kind) {
    Py_ssize_t occupied = 0;
    if (usearch_py_matrix_bytes(rows, usearch_py_scalar_row_bytes(kind, columns), 1, &occupied) != 0) return NULL;

    PyObject* const flat = PyMemoryView_FromObject(arena);
    if (!flat) return NULL;
    // `cast` demands the shape account for the whole buffer, so an arena reserved for more rows
    // than were filled is sliced down first. The slice is a view; nothing is copied.
    PyObject* const occupied_view = PySequence_GetSlice(flat, 0, occupied);
    Py_DECREF(flat);
    if (!occupied_view) return NULL;

    char const* const format = usearch_py_scalar_buffer_format(kind);
    Py_ssize_t const shape[2] = {(Py_ssize_t)rows,
                                 format ? (Py_ssize_t)columns : (Py_ssize_t)usearch_py_scalar_row_bytes(kind, columns)};
    PyObject* const shaped = usearch_py_shaped_view(occupied_view, format ? format : "B", shape, 2);
    Py_DECREF(occupied_view);
    return shaped;
}

int usearch_py_matrix_bytes(size_t rows, size_t columns, size_t unit, Py_ssize_t* bytes) {
    size_t const cells = rows * columns;
    if (columns && cells / columns != rows)
        return usearch_py_raise(usearch_status_overflow_k, "the requested output matrix does not fit memory");
    size_t const total = cells * unit;
    if (unit && total / unit != cells)
        return usearch_py_raise(usearch_status_overflow_k, "the requested output matrix does not fit memory");
    // `PTRDIFF_MAX` rather than `PY_SSIZE_T_MAX`, which resolves to a POSIX macro a strict-C
    // translation unit does not see; `Py_ssize_t` is a `ptrdiff_t` either way.
    if (total > (size_t)PTRDIFF_MAX)
        return usearch_py_raise(usearch_status_overflow_k, "the requested output matrix does not fit memory");
    *bytes = (Py_ssize_t)total;
    return 0;
}

int usearch_py_scalar_from_object(PyObject* object, usearch_scalar_t* kind) {
    if (!object || object == Py_None) {
        *kind = usearch_scalar_unknown_k;
        return 0;
    }
    if (!PyUnicode_Check(object)) {
        PyErr_Format(PyExc_TypeError, "Scalar kind must be a string, got %S", Py_TYPE(object));
        return -1;
    }
    Py_ssize_t length = 0;
    char const* const name = PyUnicode_AsUTF8AndSize(object, &length);
    if (!name) return -1;
    usearch_scalar_t const resolved = usearch_py_scalar_from_spelling(name, length);
    if (resolved == usearch_scalar_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unsupported scalar kind '%S'", object);
        return -1;
    }
    *kind = resolved;
    return 0;
}

#pragma endregion Scalar Names

#pragma region Buffer Acquisition

/**
 *  @brief Synthesizes a @c Py_buffer from @c __array_interface__ when PEP 3118 refuses.
 *  @return 1 when the fallback produced a buffer, 0 when the object has no interface.
 */
static int buffer_via_array_interface(PyObject* object, usearch_py_buffer_backing_t* backing) {
    PyObject* const interface = PyObject_GetAttrString(object, "__array_interface__");
    if (!interface) {
        PyErr_Clear();
        return 0;
    }
    if (!PyDict_Check(interface)) {
        Py_DECREF(interface);
        return 0;
    }

    PyObject* const data_tuple = PyDict_GetItemString(interface, "data");
    if (!data_tuple || !PyTuple_Check(data_tuple) || PyTuple_GET_SIZE(data_tuple) < 1) {
        Py_DECREF(interface);
        PyErr_SetString(PyExc_TypeError, "__array_interface__['data'] must be a tuple of (pointer, readonly)");
        return 0;
    }
    void* const data = PyLong_AsVoidPtr(PyTuple_GET_ITEM(data_tuple, 0));
    if (!data && PyErr_Occurred()) {
        Py_DECREF(interface);
        return 0;
    }
    int readonly = 0;
    if (PyTuple_GET_SIZE(data_tuple) > 1) {
        readonly = PyObject_IsTrue(PyTuple_GET_ITEM(data_tuple, 1));
        if (readonly < 0) {
            Py_DECREF(interface);
            return 0;
        }
    }

    PyObject* const shape_object = PyDict_GetItemString(interface, "shape");
    if (!shape_object || !PyTuple_Check(shape_object)) {
        Py_DECREF(interface);
        PyErr_SetString(PyExc_TypeError, "__array_interface__['shape'] must be a tuple");
        return 0;
    }
    Py_ssize_t const rank = PyTuple_GET_SIZE(shape_object);
    if (rank < 1 || rank > usearch_py_max_rank_k) {
        Py_DECREF(interface);
        PyErr_Format(PyExc_ValueError, "Expected a vector or a matrix, got rank %zd", rank);
        return 0;
    }
    for (Py_ssize_t axis = 0; axis != rank; ++axis) {
        backing->shape[axis] = PyLong_AsSsize_t(PyTuple_GET_ITEM(shape_object, axis));
        if (backing->shape[axis] == -1 && PyErr_Occurred()) {
            Py_DECREF(interface);
            return 0;
        }
    }

    // The exporter's own dtype name is richer than the typestring, so ask for it first.
    usearch_scalar_t kind = usearch_scalar_unknown_k;
    PyObject* const dtype_attribute = PyObject_GetAttrString(object, "dtype");
    if (dtype_attribute) {
        PyObject* const name_attribute = PyObject_GetAttrString(dtype_attribute, "name");
        if (name_attribute) {
            Py_ssize_t name_length = 0;
            char const* const name = PyUnicode_AsUTF8AndSize(name_attribute, &name_length);
            if (name) kind = usearch_py_scalar_from_spelling(name, name_length);
            Py_DECREF(name_attribute);
        }
        else { PyErr_Clear(); }
        Py_DECREF(dtype_attribute);
    }
    else { PyErr_Clear(); }

    if (kind == usearch_scalar_unknown_k) {
        PyObject* const typestr_object = PyDict_GetItemString(interface, "typestr");
        if (typestr_object) {
            Py_ssize_t typestr_length = 0;
            char const* const typestr = PyUnicode_AsUTF8AndSize(typestr_object, &typestr_length);
            if (typestr) kind = usearch_py_scalar_from_spelling(typestr, typestr_length);
        }
    }
    scalar_naming_t const* const naming = scalar_naming(kind);
    if (!naming) {
        Py_DECREF(interface);
        PyErr_SetString(PyExc_ValueError, "Cannot determine the scalar kind from __array_interface__");
        return 0;
    }
    Py_ssize_t const item_bytes = (Py_ssize_t)naming->unit_bytes;

    PyObject* const strides_object = PyDict_GetItemString(interface, "strides");
    if (strides_object && PyTuple_Check(strides_object) && PyTuple_GET_SIZE(strides_object) == rank) {
        for (Py_ssize_t axis = 0; axis != rank; ++axis) {
            backing->strides[axis] = PyLong_AsSsize_t(PyTuple_GET_ITEM(strides_object, axis));
            if (backing->strides[axis] == -1 && PyErr_Occurred()) {
                Py_DECREF(interface);
                return 0;
            }
        }
    }
    else {
        Py_ssize_t stride = item_bytes;
        for (Py_ssize_t axis = rank - 1; axis >= 0; --axis) {
            backing->strides[axis] = stride;
            stride *= backing->shape[axis];
        }
    }
    Py_DECREF(interface);

    Py_buffer* const buffer = &backing->buffer;
    memset(buffer, 0, sizeof(*buffer));
    buffer->buf = data;
    buffer->readonly = readonly;
    buffer->itemsize = item_bytes;
    buffer->format = (char*)(naming->buffer_typestr ? naming->buffer_typestr : naming->numpy_typestr);
    buffer->ndim = (int)rank;
    buffer->len = item_bytes;
    for (Py_ssize_t axis = 0; axis != rank; ++axis) buffer->len *= backing->shape[axis];
    buffer->shape = backing->shape;
    buffer->strides = backing->strides;
    // `PyBuffer_Release` becomes a no-op with no exporter, so the reference below is what keeps
    // the memory alive - a temporary exporter would otherwise die inside `Py_BEGIN_ALLOW_THREADS`.
    buffer->obj = NULL;
    backing->synthesized_owner = Py_NewRef(object);
    return 1;
}

/**
 *  @brief Acquires a strided, formatted buffer over @p object by whichever route works.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int buffer_from_object(PyObject* object, usearch_py_buffer_backing_t* backing) {
    backing->synthesized_owner = NULL;
    if (PyObject_GetBuffer(object, &backing->buffer, PyBUF_STRIDES | PyBUF_FORMAT) == 0) return 0;
    PyErr_Clear();
    if (buffer_via_array_interface(object, backing)) return 0;
    if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError, "Expected an object supporting the buffer protocol or __array_interface__");
    return -1;
}

int usearch_py_object_is_buffer(PyObject* object) {
    if (PyObject_CheckBuffer(object)) return 1;
    // A probe, not a verb: a missing attribute means "no". Anything else is a real failure and
    // is reported as such, rather than being laundered into a "no" the caller cannot see.
    PyObject* const interface = PyObject_GetAttrString(object, "__array_interface__");
    if (interface) {
        Py_DECREF(interface);
        return 1;
    }
    if (!PyErr_ExceptionMatches(PyExc_AttributeError)) return -1;
    PyErr_Clear();
    return 0;
}

void usearch_py_buffer_release(usearch_py_buffer_backing_t* backing) {
    PyBuffer_Release(&backing->buffer);
    Py_CLEAR(backing->synthesized_owner);
}

int usearch_py_buffer_from_object(PyObject* object, char const* field, usearch_py_matrix_access_t access,
                                  Py_ssize_t bytes, usearch_py_buffer_backing_t* backing, void** data) {
    backing->synthesized_owner = NULL;
    int const flags = PyBUF_C_CONTIGUOUS | (access == usearch_py_matrix_access_write_k ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, &backing->buffer, flags) != 0) {
        // The exporter's own diagnosis is more precise than any guess we could make, so it
        // rides along as the cause rather than being replaced.
        return usearch_py_raise_from_cause(PyExc_ValueError, "The '%s' buffer must be contiguous%s", field,
                                           access == usearch_py_matrix_access_write_k ? " and writable" : "");
    }
    if (backing->buffer.len != bytes) {
        usearch_py_buffer_release(backing);
        PyErr_Format(PyExc_ValueError, "The '%s' buffer must hold exactly %zd bytes, not %zd", field, bytes,
                     backing->buffer.len);
        return -1;
    }
    *data = backing->buffer.buf;
    return 0;
}

int usearch_py_matrix_from_object(PyObject* object, usearch_scalar_t scalar_hint, usearch_matrix_view_t* view,
                                  usearch_py_buffer_backing_t* backing) {
    if (buffer_from_object(object, backing) != 0) return -1;
    Py_buffer const* const buffer = &backing->buffer;

    if (buffer->ndim < 1 || buffer->ndim > usearch_py_max_rank_k) {
        PyErr_Format(PyExc_ValueError, "Expected a vector or a matrix, got rank %d", buffer->ndim);
        usearch_py_buffer_release(backing);
        return -1;
    }

    usearch_scalar_t kind = scalar_hint;
    if (kind == usearch_scalar_unknown_k && buffer->format)
        kind = usearch_py_scalar_from_spelling(buffer->format, (Py_ssize_t)strlen(buffer->format));
    scalar_naming_t const* const naming = scalar_naming(kind);
    if (!naming) {
        PyErr_Format(PyExc_ValueError, "Unsupported '%s' scalar kind; pass `dtype=` to name it",
                     buffer->format ? buffer->format : "?");
        usearch_py_buffer_release(backing);
        return -1;
    }

    // Only the row stride may vary: a strided column would force a gather the kernels have
    // no form for, and the C surface has no argument to describe one.
    int const last_axis = buffer->ndim - 1;
    if (buffer->strides && buffer->strides[last_axis] != buffer->itemsize) {
        PyErr_SetString(PyExc_ValueError, "Expected a matrix whose rows are contiguous; only the stride may vary");
        usearch_py_buffer_release(backing);
        return -1;
    }

    // A shape counts storage units and the descriptor counts logical dimensions, so a sub-byte
    // kind widens by its packing factor - an eight-byte `u1x8` row is sixty-four dimensions.
    view->data = buffer->buf;
    view->scalar = kind;
    if (buffer->ndim == 1) {
        view->rows = 1;
        view->columns = (size_t)buffer->shape[0] * naming->values_packed;
        view->stride = (size_t)buffer->len;
    }
    else {
        view->rows = (size_t)buffer->shape[0];
        view->columns = (size_t)buffer->shape[1] * naming->values_packed;
        Py_ssize_t const stride_bytes = buffer->strides ? buffer->strides[0] : buffer->shape[1] * buffer->itemsize;
        if (stride_bytes < 0) {
            PyErr_SetString(PyExc_ValueError, "Matrix rows must ascend in memory");
            usearch_py_buffer_release(backing);
            return -1;
        }
        view->stride = (size_t)stride_bytes;
    }
    return 0;
}

#pragma endregion Buffer Acquisition

#pragma region Keys

/** A 128-bit entry identity, rendered as a lowercase dashed UUID. */
typedef struct Key {
    PyObject ob_base;
    usearch_key_t key;
} Key;

static char const doc_Key[] =                                                              //
    "Key(value=0)\n"                                                                       //
    "\n"                                                                                   //
    "A 128-bit entry identity, rendered as a lowercase dashed UUID.\n"                     //
    "Keys need not be unique: several entries may share one, and `count` reports how\n"    //
    "many. Integer identities park in the low eight bytes.\n"                              //
    "\n"                                                                                   //
    "Args:\n"                                                                              //
    "  value (int | str | bytes | uuid.UUID | Key, optional): The identity. Defaults 0.\n" //
    "Example:\n"                                                                           //
    "  >>> str(usearch.Key(42))\n"                                                         //
    "  '00000000-0000-0000-0000-00000000002a'\n"                                           //
    "  >>> int(usearch.Key('00000000-0000-0000-0000-00000000002a'))\n"                     //
    "  42";

static int key_from_text(char const* text, Py_ssize_t length, usearch_key_t* key) {
    unsigned char digits[32];
    Py_ssize_t written = 0;
    for (Py_ssize_t position = 0; position != length; ++position) {
        char const character = text[position];
        if (character == '-') continue;
        if (written == 32) return -1;
        unsigned char digit;
        if (character >= '0' && character <= '9') { digit = (unsigned char)(character - '0'); }
        else if (character >= 'a' && character <= 'f') { digit = (unsigned char)(character - 'a' + 10); }
        else if (character >= 'A' && character <= 'F') { digit = (unsigned char)(character - 'A' + 10); }
        else { return -1; }
        digits[written++] = digit;
    }
    if (written != 32) return -1;
    for (size_t byte_index = 0; byte_index != 16; ++byte_index)
        key->bytes[byte_index] = (uint8_t)((digits[byte_index * 2] << 4) | digits[byte_index * 2 + 1]);
    return 0;
}

int usearch_py_key_from_object(PyObject* object, usearch_key_t* key) {
    if (PyObject_TypeCheck(object, &KeyType)) {
        *key = ((Key*)object)->key;
        return 0;
    }
    // Sixteen raw bytes - a `bytes` literal, a `memoryview`, or one row of an identity
    // matrix - are the wire form already. Checked before the integer path, since an array
    // row exports both protocols and only this reading of it is meaningful.
    if (PyObject_CheckBuffer(object)) {
        Py_buffer buffer;
        if (PyObject_GetBuffer(object, &buffer, PyBUF_SIMPLE) == 0) {
            int const is_identity = buffer.len == 16 && buffer.itemsize == 1;
            if (is_identity) memcpy(key->bytes, buffer.buf, 16);
            PyBuffer_Release(&buffer);
            if (is_identity) return 0;
        }
        else { PyErr_Clear(); }
    }
    // `PyIndex_Check` rather than `PyLong_Check`, so a NumPy scalar - which is not a Python
    // integer but knows how to become one - is read the same way a literal is.
    if (PyIndex_Check(object)) {
        PyObject* const integer = PyNumber_Index(object);
        if (!integer) return -1;
        // Big-endian and unsigned, so integer identities park in the low eight bytes and sort
        // the way the dashed rendering reads. `int.to_bytes` raises on negatives and overflow
        // for us, and this is never the batch path - those arrive as buffers.
        PyObject* const raw = PyObject_CallMethod(integer, "to_bytes", "ns", (Py_ssize_t)16, "big");
        Py_DECREF(integer);
        if (!raw) return -1;
        memcpy(key->bytes, PyBytes_AS_STRING(raw), 16);
        Py_DECREF(raw);
        return 0;
    }
    if (PyUnicode_Check(object)) {
        Py_ssize_t length = 0;
        char const* const text = PyUnicode_AsUTF8AndSize(object, &length);
        if (!text) return -1;
        if (key_from_text(text, length, key) != 0)
            return usearch_py_raise(usearch_status_malformed_k, "the key is not thirty-two hexadecimal digits");
        return 0;
    }
    // `uuid.UUID` and anything else shaped like it, without importing the module.
    PyObject* const raw_bytes = PyObject_GetAttrString(object, "bytes");
    if (raw_bytes) {
        int const outcome = usearch_py_key_from_object(raw_bytes, key);
        Py_DECREF(raw_bytes);
        return outcome;
    }
    PyErr_Clear();
    PyErr_Format(PyExc_TypeError, "Cannot read a key from %S", Py_TYPE(object));
    return -1;
}

PyObject* usearch_py_key_to_object(usearch_key_t key) {
    Key* const instance = PyObject_New(Key, &KeyType);
    if (!instance) return NULL;
    instance->key = key;
    return (PyObject*)instance;
}

static PyObject* Key_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    Py_ssize_t const positional_args_count = PyTuple_GET_SIZE(args);
    if (positional_args_count > 1 || (keywords && PyDict_GET_SIZE(keywords) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Key() takes one positional argument");
        return NULL;
    }
    PyObject* const value = positional_args_count == 1 ? PyTuple_GET_ITEM(args, 0) : NULL;

    Key* const instance = (Key*)type->tp_alloc(type, 0);
    if (!instance) return NULL;
    memset(instance->key.bytes, 0, sizeof(instance->key.bytes));
    if (value && usearch_py_key_from_object(value, &instance->key) != 0) {
        Py_DECREF(instance);
        return NULL;
    }
    return (PyObject*)instance;
}

static void Key_dealloc(Key* self) { Py_TYPE(self)->tp_free((PyObject*)self); }

static PyObject* Key_str(Key* self) {
    char rendered[36];
    size_t written = 0;
    for (size_t byte_index = 0; byte_index != 16; ++byte_index) {
        if (byte_index == 4 || byte_index == 6 || byte_index == 8 || byte_index == 10) rendered[written++] = '-';
        static char const digits[] = "0123456789abcdef";
        rendered[written++] = digits[self->key.bytes[byte_index] >> 4];
        rendered[written++] = digits[self->key.bytes[byte_index] & 0x0F];
    }
    return PyUnicode_FromStringAndSize(rendered, (Py_ssize_t)written);
}

static PyObject* Key_repr(Key* self) {
    PyObject* const rendered = Key_str(self);
    if (!rendered) return NULL;
    PyObject* const wrapped = PyUnicode_FromFormat("Key('%U')", rendered);
    Py_DECREF(rendered);
    return wrapped;
}

static PyObject* Key_int(Key* self) {
    PyObject* const raw = PyBytes_FromStringAndSize((char const*)self->key.bytes, 16);
    if (!raw) return NULL;
    PyObject* const value = PyObject_CallMethod((PyObject*)&PyLong_Type, "from_bytes", "Os", raw, "big");
    Py_DECREF(raw);
    return value;
}

/**
 *  @brief Hashes as the integer this key compares equal to.
 *
 *  `Key(256) == 256` is true, so the two must hash alike or a dict keyed by one cannot be
 *  read by the other. Any hash of our own would coincide with `hash(int)` only by accident.
 */
static Py_hash_t Key_hash(Key* self) {
    PyObject* const integer = Key_int(self);
    if (!integer) return -1;
    Py_hash_t const hashed = PyObject_Hash(integer);
    Py_DECREF(integer);
    return hashed;
}

static PyObject* Key_richcompare(PyObject* first, PyObject* second, int operation) {
    usearch_key_t left, right;
    if (usearch_py_key_from_object(first, &left) != 0 || usearch_py_key_from_object(second, &right) != 0) {
        // Only "that is not a key" means "not comparable"; a `MemoryError` raised while reading
        // one must not be laundered into a silent `False`.
        if (!PyErr_ExceptionMatches(PyExc_TypeError) && !PyErr_ExceptionMatches(ErrorType)) return NULL;
        PyErr_Clear();
        Py_RETURN_NOTIMPLEMENTED;
    }
    int const ordering = memcmp(left.bytes, right.bytes, 16);
    Py_RETURN_RICHCOMPARE(ordering, 0, operation);
}

static PyObject* Key_get_bytes(Key* self, void* closure) {
    return PyBytes_FromStringAndSize((char const*)self->key.bytes, 16);
}

static char const doc_Key_bytes[] = "The sixteen identity bytes, most significant first.";

static PyGetSetDef Key_getset[] = {
    {"bytes", (getter)Key_get_bytes, NULL, doc_Key_bytes, NULL}, //
    {NULL, NULL, NULL, NULL, NULL},
};

static PyNumberMethods Key_as_number = {
    .nb_int = (unaryfunc)Key_int,
    .nb_index = (unaryfunc)Key_int,
};

PyTypeObject KeyType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.Key",
    .tp_doc = doc_Key,
    .tp_basicsize = sizeof(Key),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Key_tp_new,
    .tp_dealloc = (destructor)Key_dealloc,
    .tp_str = (reprfunc)Key_str,
    .tp_repr = (reprfunc)Key_repr,
    .tp_hash = (hashfunc)Key_hash,
    .tp_richcompare = Key_richcompare,
    .tp_as_number = &Key_as_number,
    .tp_getset = Key_getset,
};

#pragma endregion Keys

#pragma region Key Batches

void usearch_py_keys_release(usearch_py_key_arena_t* arena) {
    if (arena->storage == usearch_py_key_storage_borrowed_k) usearch_py_buffer_release(&arena->backing);
    else if (arena->owned) PyMem_Free(arena->owned);
    arena->owned = NULL;
    arena->keys = NULL;
    arena->storage = usearch_py_key_storage_owned_k;
}

/** Which storage a key batch will use, once its argument has been looked at. */
typedef enum keys_route_t {

    /** An (N, 16) contiguous identity buffer; the arena borrows it outright. */
    keys_route_borrowed_k,

    /** Not that shape, so each element converts on its own terms. */
    keys_route_sequence_k,

    /** Something went wrong and an exception is set. */
    keys_route_failed_k,
} keys_route_t;

/**
 *  @brief Tries the zero-copy route, distinguishing "not an identity buffer" from a real failure.
 *
 *  A blanket clear here would swallow a @c KeyboardInterrupt delivered inside an exporter, so
 *  only the failure this function itself provoked is discarded.
 */
static keys_route_t keys_borrow_buffer(PyObject* object, size_t expected, usearch_py_key_arena_t* arena) {
    if (!PyObject_CheckBuffer(object)) return keys_route_sequence_k;
    if (PyObject_GetBuffer(object, &arena->backing.buffer, PyBUF_STRIDES) != 0) {
        PyErr_Clear();
        return keys_route_sequence_k;
    }
    arena->backing.synthesized_owner = NULL;
    arena->storage = usearch_py_key_storage_borrowed_k;
    Py_buffer const* const buffer = &arena->backing.buffer;
    if (buffer->ndim != 2 || buffer->shape[1] != (Py_ssize_t)sizeof(usearch_key_t) || buffer->itemsize != 1 ||
        !PyBuffer_IsContiguous((Py_buffer*)buffer, 'C')) {
        usearch_py_keys_release(arena);
        return keys_route_sequence_k;
    }
    if ((size_t)buffer->shape[0] != expected) {
        usearch_py_keys_release(arena);
        usearch_py_raise(usearch_status_shape_mismatch_k, "the key count does not match the vector row count");
        return keys_route_failed_k;
    }
    arena->keys = (usearch_key_t*)buffer->buf;
    return keys_route_borrowed_k;
}

int usearch_py_keys_from_object(PyObject* object, size_t expected, usearch_py_key_arena_t* arena) {
    memset(arena, 0, sizeof(*arena));

    // An integer array exports a buffer too, so only the identity shape takes the zero-copy
    // route; everything else walks the sequence below.
    switch (keys_borrow_buffer(object, expected, arena)) {
    case keys_route_borrowed_k: return 0;
    case keys_route_failed_k: return -1;
    case keys_route_sequence_k: break;
    }

    PyObject* const sequence = PySequence_Fast(object, "Expected a sequence of keys or a buffer of identities");
    if (!sequence) return -1;
    Py_ssize_t const count = PySequence_Fast_GET_SIZE(sequence);
    if ((size_t)count != expected) {
        Py_DECREF(sequence);
        usearch_py_raise(usearch_status_shape_mismatch_k, "the key count does not match the vector row count");
        return -1;
    }

    arena->owned = (usearch_key_t*)PyMem_Malloc(expected ? expected * sizeof(usearch_key_t) : 1);
    if (!arena->owned) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t entry_index = 0; entry_index != count; ++entry_index) {
        if (usearch_py_key_from_object(PySequence_Fast_GET_ITEM(sequence, entry_index), &arena->owned[entry_index]) !=
            0) {
            Py_DECREF(sequence);
            usearch_py_keys_release(arena);
            return -1;
        }
    }
    Py_DECREF(sequence);
    arena->keys = arena->owned;
    return 0;
}

#pragma endregion Key Batches

#pragma region Telemetry

static char const doc_Stats[] =                                                  //
    "Stats()\n"                                                                  //
    "\n"                                                                         //
    "Traversal telemetry, fully overwritten by every call it is passed to.\n"    //
    "\n"                                                                         //
    "One reading belongs to one call, so a sweep that wants totals folds each\n" //
    "reading into a separate `Stats` with `accumulate`.\n"                       //
    "\n"                                                                         //
    "Attributes:\n"                                                              //
    "  visited_members (int): Entries whose neighborhoods the walk expanded.\n"  //
    "  computed_distances (int): Distance evaluations the walk paid for.\n"      //
    "  restarts (int): Times the walk fell back to a fresh entry point.";

static PyObject* Stats_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    if (PyTuple_GET_SIZE(args) != 0 || (keywords && PyDict_Size(keywords) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Stats() takes no arguments");
        return NULL;
    }
    Stats* const instance = (Stats*)type->tp_alloc(type, 0);
    if (!instance) return NULL;
    memset(&instance->stats, 0, sizeof(instance->stats));
    return (PyObject*)instance;
}

static void Stats_dealloc(Stats* self) { Py_TYPE(self)->tp_free((PyObject*)self); }

static PyObject* Stats_repr(Stats* self) {
    return PyUnicode_FromFormat("Stats(visited_members=%zu, computed_distances=%zu, restarts=%zu)",
                                self->stats.visited_members, self->stats.computed_distances, self->stats.restarts);
}

PyObject* usearch_py_stats_to_object(usearch_dense_traversal_stats_t const* stats) {
    Stats* const instance = PyObject_New(Stats, &StatsType);
    if (!instance) return NULL;
    instance->stats = *stats;
    return (PyObject*)instance;
}

int usearch_py_stats_overwrite(PyObject* target, usearch_dense_traversal_stats_t const* stats) {
    if (!target || target == Py_None) return 0;
    if (!PyObject_TypeCheck(target, &StatsType)) {
        PyErr_Format(PyExc_TypeError, "Expected a Stats object, got %S", Py_TYPE(target));
        return -1;
    }
    ((Stats*)target)->stats = *stats;
    return 0;
}

static char const doc_Stats_accumulate[] =                                      //
    "accumulate(other) -> None\n"                                               //
    "\n"                                                                        //
    "Fold another reading's counters into this one, summing a sweep's calls.\n" //
    "\n"                                                                        //
    "Args:\n"                                                                   //
    "  other (Stats): The reading to add.\n"                                    //
    "\n"                                                                        //
    "Example:\n"                                                                //
    "  >>> total, reading = usearch.Stats(), usearch.Stats()\n"                 //
    "  >>> total.accumulate(reading)\n"                                         //
    "  >>> total.visited_members\n"                                             //
    "  0";

static PyObject* Stats_accumulate(PyObject* self, PyObject* other) {
    if (!PyObject_TypeCheck(other, &StatsType)) {
        PyErr_Format(PyExc_TypeError, "Expected a Stats object, got %S", Py_TYPE(other));
        return NULL;
    }
    // A read-modify-write, unlike the overwrite a verb performs, so two threads folding into
    // one total are fenced against each other - the module declares it does not need the GIL.
    Stats* const total = (Stats*)self;
    usearch_dense_traversal_stats_t const addend = ((Stats*)other)->stats;
    usearch_py_critical_section_begin_(self);
    total->stats.visited_members += addend.visited_members;
    total->stats.computed_distances += addend.computed_distances;
    total->stats.restarts += addend.restarts;
    usearch_py_critical_section_end_();
    Py_RETURN_NONE;
}

static char const doc_Stats_visited_members[] = "Entries whose neighborhoods the walk expanded.";

static PyObject* Stats_get_visited_members(Stats* self, void* closure) {
    return PyLong_FromSize_t(self->stats.visited_members);
}

static char const doc_Stats_computed_distances[] = "Distance evaluations the walk paid for.";

static PyObject* Stats_get_computed_distances(Stats* self, void* closure) {
    return PyLong_FromSize_t(self->stats.computed_distances);
}

static char const doc_Stats_restarts[] = "Times the walk fell back to a fresh entry point.";

static PyObject* Stats_get_restarts(Stats* self, void* closure) { return PyLong_FromSize_t(self->stats.restarts); }

static PyGetSetDef Stats_getset[] = {
    {"visited_members", (getter)Stats_get_visited_members, NULL, doc_Stats_visited_members, NULL},          //
    {"computed_distances", (getter)Stats_get_computed_distances, NULL, doc_Stats_computed_distances, NULL}, //
    {"restarts", (getter)Stats_get_restarts, NULL, doc_Stats_restarts, NULL},                               //
    {NULL, NULL, NULL, NULL, NULL},
};

static PyMethodDef Stats_methods[] = {
    {"accumulate", (PyCFunction)Stats_accumulate, METH_O, doc_Stats_accumulate}, //
    {NULL, NULL, 0, NULL},
};

PyTypeObject StatsType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.Stats",
    .tp_doc = doc_Stats,
    .tp_basicsize = sizeof(Stats),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Stats_tp_new,
    .tp_dealloc = (destructor)Stats_dealloc,
    .tp_repr = (reprfunc)Stats_repr,
    .tp_methods = Stats_methods,
    .tp_getset = Stats_getset,
};

#pragma endregion Telemetry

#pragma region Matches

static char const doc_Match[] =                                //
    "One scored entry: the identity found and its distance.\n" //
    "\n"                                                       //
    "Attributes:\n"                                            //
    "  key (Key): The identity of the matched entry.\n"        //
    "  distance (float): The distance to the query, in the index's own metric.";

PyObject* usearch_py_match_to_object(usearch_key_t key, usearch_distance_t distance) {
    Match* const instance = PyObject_New(Match, &MatchType);
    if (!instance) return NULL;
    instance->key = key;
    instance->distance = distance;
    return (PyObject*)instance;
}

static void Match_dealloc(Match* self) { Py_TYPE(self)->tp_free((PyObject*)self); }

static char const doc_Match_key[] = "The identity of the matched entry.";

static PyObject* Match_get_key(Match* self, void* closure) { return usearch_py_key_to_object(self->key); }

static char const doc_Match_distance[] = "The distance to the query, in the index's own metric.";

static PyObject* Match_get_distance(Match* self, void* closure) { return PyFloat_FromDouble(self->distance); }

static PyObject* Match_repr(Match* self) {
    PyObject* const key_object = usearch_py_key_to_object(self->key);
    if (!key_object) return NULL;
    PyObject* const distance_object = PyFloat_FromDouble(self->distance);
    if (!distance_object) {
        Py_DECREF(key_object);
        return NULL;
    }
    PyObject* const rendered = PyUnicode_FromFormat("Match(key=%R, distance=%R)", key_object, distance_object);
    Py_DECREF(distance_object);
    Py_DECREF(key_object);
    return rendered;
}

static PyGetSetDef Match_getset[] = {
    {"key", (getter)Match_get_key, NULL, doc_Match_key, NULL},                //
    {"distance", (getter)Match_get_distance, NULL, doc_Match_distance, NULL}, //
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject MatchType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.Match",
    .tp_doc = doc_Match,
    .tp_basicsize = sizeof(Match),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, //! Yielded by searches and finders, never constructed by name.
    .tp_dealloc = (destructor)Match_dealloc,
    .tp_repr = (reprfunc)Match_repr,
    .tp_getset = Match_getset,
};

static char const doc_Matches[] =                                                             //
    "A ranked run, as `search` and `maxsim` return it.\n"                                     //
    "\n"                                                                                      //
    "Reads through to three buffers, so `np.asarray` costs nothing and NumPy is never\n"      //
    "required. A verb returns one owning fresh buffers; constructing one over buffers the\n"  //
    "caller already owns, and passing it back as `out=`, is how a sweep stops allocating.\n"  //
    "No NumPy dtype expresses a 128-bit integer, so keys arrive as big-endian sixteen-byte\n" //
    "rows. The matches-per-query width follows from the three lengths, so nothing can\n"      //
    "contradict them.\n"                                                                      //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  keys (buffer): A (queries, wanted, 16) buffer of identity bytes.\n"                    //
    "  distances (buffer): A (queries, wanted) buffer of float64 distances.\n"                //
    "  counts (buffer): A (queries,) buffer of unsigned word-sized counts.\n"                 //
    "\n"                                                                                      //
    "Attributes:\n"                                                                           //
    "  keys (memoryview): (queries, wanted, 16) identities, most significant byte first.\n"   //
    "  distances (memoryview): (queries, wanted) distances.\n"                                //
    "  counts (memoryview): (queries,) match counts, each bounding its own row.\n"            //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> keys, distances, counts = bytearray(2 * 16), bytearray(2 * 8), bytearray(8)\n"     //
    "  >>> len(usearch.Matches(keys, distances, counts))\n"                                   //
    "  1";

static void Matches_dealloc(Matches* self) {
    usearch_py_buffer_release(&self->counts);
    usearch_py_buffer_release(&self->distances);
    usearch_py_buffer_release(&self->keys);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** Borrows one contiguous output for a @c Matches to read through. */
static int matches_borrow(PyObject* object, char const* field, usearch_py_buffer_backing_t* backing) {
    backing->synthesized_owner = NULL;
    if (PyObject_GetBuffer(object, &backing->buffer, PyBUF_C_CONTIGUOUS) != 0)
        return usearch_py_raise_from_cause(PyExc_ValueError, "The '%s' buffer must be contiguous", field);
    return 0;
}

static PyObject* Matches_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    if (PyTuple_GET_SIZE(args) != 3 || (keywords && PyDict_GET_SIZE(keywords) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Matches() takes three positional arguments: keys, distances, counts");
        return NULL;
    }
    Matches* const instance = (Matches*)type->tp_alloc(type, 0);
    if (!instance) return NULL;
    if (matches_borrow(PyTuple_GET_ITEM(args, 0), "keys", &instance->keys) != 0 ||
        matches_borrow(PyTuple_GET_ITEM(args, 1), "distances", &instance->distances) != 0 ||
        matches_borrow(PyTuple_GET_ITEM(args, 2), "counts", &instance->counts) != 0) {
        Py_DECREF(instance);
        return NULL;
    }

    Py_ssize_t const keys_bytes = instance->keys.buffer.len;
    Py_ssize_t const distances_bytes = instance->distances.buffer.len;
    Py_ssize_t const counts_bytes = instance->counts.buffer.len;
    if (keys_bytes % (Py_ssize_t)sizeof(usearch_key_t) != 0 ||
        distances_bytes % (Py_ssize_t)sizeof(usearch_distance_t) != 0 ||
        counts_bytes % (Py_ssize_t)sizeof(size_t) != 0) {
        usearch_py_raise(usearch_status_shape_mismatch_k, "the buffers do not hold whole keys, distances and counts");
        Py_DECREF(instance);
        return NULL;
    }

    size_t const keyed = (size_t)keys_bytes / sizeof(usearch_key_t);
    size_t const scored = (size_t)distances_bytes / sizeof(usearch_distance_t);
    size_t const queries = (size_t)counts_bytes / sizeof(size_t);
    if (keyed != scored) {
        usearch_py_raise(usearch_status_shape_mismatch_k, "the key and distance buffers are not the same shape");
        Py_DECREF(instance);
        return NULL;
    }
    if (queries == 0 || keyed == 0 || keyed % queries != 0) {
        usearch_py_raise(usearch_status_shape_mismatch_k, "the buffers do not divide into one row per query");
        Py_DECREF(instance);
        return NULL;
    }

    instance->queries = queries;
    instance->wanted = keyed / queries;
    return (PyObject*)instance;
}

static char const doc_Matches_keys[] = "(queries, wanted, 16) identities, most significant byte first.";

static PyObject* Matches_get_keys(Matches* self, void* closure) {
    Py_ssize_t const shape[3] = {(Py_ssize_t)self->queries, (Py_ssize_t)self->wanted, 16};
    return usearch_py_shaped_view(self->keys.buffer.obj, "B", shape, 3);
}

static char const doc_Matches_distances[] = "(queries, wanted) distances, in the index's own metric.";

static PyObject* Matches_get_distances(Matches* self, void* closure) {
    Py_ssize_t const shape[2] = {(Py_ssize_t)self->queries, (Py_ssize_t)self->wanted};
    return usearch_py_shaped_view(self->distances.buffer.obj, "d", shape, 2);
}

static char const doc_Matches_counts[] = "(queries,) match counts, each bounding its own row.";

static PyObject* Matches_get_counts(Matches* self, void* closure) {
    Py_ssize_t const shape[1] = {(Py_ssize_t)self->queries};
    return usearch_py_shaped_view(self->counts.buffer.obj, "N", shape, 1);
}

PyObject* usearch_py_matches_prepare(PyObject* out, size_t queries, size_t wanted, usearch_matches_span_t* span) {
    Py_ssize_t extents[3] = {0, 0, 0};
    if (usearch_py_matrix_bytes(queries, wanted, sizeof(usearch_key_t), &extents[0]) != 0 ||
        usearch_py_matrix_bytes(queries, wanted, sizeof(usearch_distance_t), &extents[1]) != 0 ||
        usearch_py_matrix_bytes(queries, 1, sizeof(size_t), &extents[2]) != 0)
        return NULL;

    Matches* matches = NULL;
    if (out && out != Py_None) {
        if (!PyObject_TypeCheck(out, &MatchesType)) {
            PyErr_Format(PyExc_TypeError, "The 'out' option must be a Matches, got %S", Py_TYPE(out));
            return NULL;
        }
        matches = (Matches*)out;
        if (matches->queries != queries || matches->wanted != wanted) {
            usearch_py_raise(usearch_status_shape_mismatch_k, "the 'out' matches hold %zu rows of %zu, not %zu of %zu",
                             matches->queries, matches->wanted, queries, wanted);
            return NULL;
        }
        if (matches->keys.buffer.readonly || matches->distances.buffer.readonly || matches->counts.buffer.readonly) {
            PyErr_SetString(PyExc_ValueError, "The 'out' matches must read through to writable buffers");
            return NULL;
        }
        Py_INCREF(matches);
    }
    else {
        // Zeroed slots, so a cell past its row's count never surfaces stale heap bytes.
        matches = (Matches*)MatchesType.tp_alloc(&MatchesType, 0);
        if (!matches) return NULL;
        usearch_py_buffer_backing_t* const backings[3] = {&matches->keys, &matches->distances, &matches->counts};
        char const* const fields[3] = {"keys", "distances", "counts"};
        for (size_t output = 0; output != 3; ++output) {
            PyObject* const storage = PyByteArray_FromStringAndSize(NULL, extents[output]);
            if (!storage) {
                Py_DECREF(matches);
                return NULL;
            }
            memset(PyByteArray_AS_STRING(storage), 0, (size_t)extents[output]);
            int const borrowed = matches_borrow(storage, fields[output], backings[output]);
            Py_DECREF(storage);
            if (borrowed != 0) {
                Py_DECREF(matches);
                return NULL;
            }
        }
        matches->queries = queries;
        matches->wanted = wanted;
    }

    span->keys = (usearch_key_t*)matches->keys.buffer.buf;
    span->keys_stride = wanted * sizeof(usearch_key_t);
    span->scores = (usearch_distance_t*)matches->distances.buffer.buf;
    span->scores_stride = wanted * sizeof(usearch_distance_t);
    span->counts = (size_t*)matches->counts.buffer.buf;
    span->capacity = queries;
    span->wanted = wanted;
    return (PyObject*)matches;
}

static Py_ssize_t Matches_length(Matches* self) { return (Py_ssize_t)self->queries; }

static PyObject* Matches_item(Matches* self, Py_ssize_t query_index) {
    // `PySequence_GetItem` normalizes a negative index before reaching here.
    if (query_index < 0 || (size_t)query_index >= self->queries) {
        PyErr_SetString(PyExc_IndexError, "Query index out of range");
        return NULL;
    }
    size_t const* const counts = (size_t const*)self->counts.buffer.buf;
    usearch_key_t const* const keys = (usearch_key_t const*)self->keys.buffer.buf;
    usearch_distance_t const* const distances = (usearch_distance_t const*)self->distances.buffer.buf;

    // A count is the engine's word, and the sibling `multiply` verb documents counts that
    // exceed the column width outright - so it bounds a read here rather than defining one.
    size_t const found = counts[query_index] < self->wanted ? counts[query_index] : self->wanted;
    size_t const offset = (size_t)query_index * self->wanted;
    PyObject* const row = PyTuple_New((Py_ssize_t)found);
    if (!row) return NULL;
    for (size_t match_index = 0; match_index != found; ++match_index) {
        PyObject* const match = usearch_py_match_to_object(keys[offset + match_index], distances[offset + match_index]);
        if (!match) {
            Py_DECREF(row);
            return NULL;
        }
        PyTuple_SET_ITEM(row, (Py_ssize_t)match_index, match);
    }
    return row;
}

static PyObject* Matches_repr(Matches* self) {
    return PyUnicode_FromFormat("Matches(queries=%zu, wanted=%zu)", self->queries, self->wanted);
}

static PyGetSetDef Matches_getset[] = {
    {"keys", (getter)Matches_get_keys, NULL, doc_Matches_keys, NULL},                //
    {"distances", (getter)Matches_get_distances, NULL, doc_Matches_distances, NULL}, //
    {"counts", (getter)Matches_get_counts, NULL, doc_Matches_counts, NULL},          //
    {NULL, NULL, NULL, NULL, NULL},
};

static PySequenceMethods Matches_as_sequence = {
    .sq_length = (lenfunc)Matches_length,
    .sq_item = (ssizeargfunc)Matches_item,
};

PyTypeObject MatchesType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.Matches",
    .tp_doc = doc_Matches,
    .tp_basicsize = sizeof(Matches),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Matches_tp_new,
    .tp_dealloc = (destructor)Matches_dealloc,
    .tp_repr = (reprfunc)Matches_repr,
    .tp_as_sequence = &Matches_as_sequence,
    .tp_getset = Matches_getset,
};

#pragma endregion Matches

#pragma region Closed Sets

int usearch_py_choice_from_object(PyObject* object, char const* field, usearch_py_choice_naming_t const* namings,
                                  size_t count, int* value) {
    if (!PyUnicode_Check(object)) {
        PyErr_Format(PyExc_TypeError, "The '%s' option must be a string, got %S", field, Py_TYPE(object));
        return -1;
    }
    for (size_t naming_index = 0; naming_index != count; ++naming_index)
        if (PyUnicode_CompareWithASCIIString(object, namings[naming_index].name) == 0) {
            *value = namings[naming_index].value;
            return 0;
        }
    PyErr_Format(PyExc_ValueError, "Unsupported '%s' option %R", field, object);
    return -1;
}

Py_ssize_t usearch_py_args_names_count_checked(char const* verb, Py_ssize_t positional_args_count,
                                               PyObject* args_names_tuple, Py_ssize_t required_positional) {
    if (positional_args_count != required_positional) {
        PyErr_Format(PyExc_TypeError, "%s() takes %zd positional arguments, received %zd", verb, required_positional,
                     positional_args_count);
        return -1;
    }
    return args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
}

int usearch_py_ordinals_from_object(PyObject* object, char const* field, int32_t** ordinals, size_t* count) {
    *ordinals = NULL;
    *count = 0;
    if (object == Py_None) return 0;

    PyObject* const sequence = PySequence_Fast(object, "expected a sequence of integers");
    if (!sequence) {
        usearch_py_raise_from_cause(PyExc_TypeError, "The '%s' option must be a sequence of integers", field);
        return -1;
    }
    Py_ssize_t const length = PySequence_Fast_GET_SIZE(sequence);
    if (length == 0) {
        Py_DECREF(sequence);
        return 0;
    }

    int32_t* const claimed = (int32_t*)PyMem_Malloc((size_t)length * sizeof(int32_t));
    if (!claimed) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t position = 0; position != length; ++position) {
        long const ordinal = PyLong_AsLong(PySequence_Fast_GET_ITEM(sequence, position));
        if (ordinal == -1 && PyErr_Occurred()) {
            PyMem_Free(claimed);
            Py_DECREF(sequence);
            usearch_py_raise_from_cause(PyExc_TypeError, "The '%s' option must hold integers", field);
            return -1;
        }
        if (ordinal < INT32_MIN || ordinal > INT32_MAX) {
            PyMem_Free(claimed);
            Py_DECREF(sequence);
            PyErr_Format(PyExc_ValueError, "The '%s' option holds %ld, outside the 32-bit range", field, ordinal);
            return -1;
        }
        claimed[position] = (int32_t)ordinal;
    }
    Py_DECREF(sequence);
    *ordinals = claimed;
    *count = (size_t)length;
    return 0;
}

/** The encoded objects trailing the views in an arena's single block. */
static PyObject** paths_encoded(usearch_py_paths_arena_t const* arena) {
    return (PyObject**)(arena->views + arena->count);
}

int usearch_py_paths_from_object(PyObject* object, usearch_py_paths_arena_t* arena) {
    arena->views = NULL;
    arena->count = 0;
    if (object == Py_None) return 0;

    PyObject* const sequence = PySequence_Fast(object, "expected a sequence of paths");
    if (!sequence) {
        usearch_py_raise_from_cause(PyExc_TypeError, "The 'paths' option must be a sequence of paths");
        return -1;
    }
    Py_ssize_t const length = PySequence_Fast_GET_SIZE(sequence);
    if (length == 0) {
        Py_DECREF(sequence);
        return 0;
    }

    // Both runs are pointer-aligned, so one block carves into views followed by their objects.
    size_t const block_bytes = (size_t)length * (sizeof(usearch_bytes_view_t) + sizeof(PyObject*));
    arena->views = (usearch_bytes_view_t*)PyMem_Malloc(block_bytes);
    if (!arena->views) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return -1;
    }

    PyObject** const encoded = (PyObject**)(arena->views + length);
    for (Py_ssize_t position = 0; position != length; ++position) {
        if (usearch_py_path_from_object(PySequence_Fast_GET_ITEM(sequence, position), &arena->views[position],
                                        &encoded[position]) != 0) {
            while (position-- != 0) Py_DECREF(encoded[position]);
            PyMem_Free(arena->views);
            arena->views = NULL;
            Py_DECREF(sequence);
            return -1;
        }
    }
    Py_DECREF(sequence);
    arena->count = (size_t)length;
    return 0;
}

void usearch_py_paths_release(usearch_py_paths_arena_t* arena) {
    if (!arena->views) return;
    PyObject** const encoded = paths_encoded(arena);
    for (size_t position = 0; position != arena->count; ++position) Py_XDECREF(encoded[position]);
    PyMem_Free(arena->views);
    arena->views = NULL;
    arena->count = 0;
}

int usearch_py_size_from_object(PyObject* object, char const* field, size_t* value) {
    Py_ssize_t const signed_value = PyNumber_AsSsize_t(object, PyExc_OverflowError);
    if (signed_value == -1 && PyErr_Occurred()) return -1;
    if (signed_value < 0) {
        PyErr_Format(PyExc_ValueError, "The '%s' option must not be negative", field);
        return -1;
    }
    *value = (size_t)signed_value;
    return 0;
}

#pragma endregion Closed Sets

#pragma region Executors

static char const doc_Executor[] =                                                          //
    "Executor\n"                                                                            //
    "\n"                                                                                    //
    "The workers a fanned-out verb runs on, shared across indexes.\n"                       //
    "\n"                                                                                    //
    "An executor owns host threads or device streams; an index owns rows. Keeping them\n"   //
    "apart is what lets one pool serve several indexes, and what makes the cost of\n"       //
    "spawning visible at the line that spawns rather than hidden in a constructor.\n"       //
    "\n"                                                                                    //
    "Construct one through the residency it drives - `for_ram`, `on_pool`, `for_vram` or\n" //
    "`for_nvme` - since each spawns something different and takes different arguments.\n"   //
    "Its residency must match that of every index it is handed to. A verb handed none\n"    //
    "runs on the caller's thread instead.\n"                                                //
    "\n"                                                                                    //
    "Example:\n"                                                                            //
    "  >>> from array import array\n"                                                       //
    "  >>> pool = usearch.Executor.for_ram(threads=8)\n"                                    //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=4)\n"                              //
    "  >>> index.reserve(1)\n"                                                              //
    "  >>> index.add([1], array('d', [0.1, 0.2, 0.3, 0.4]), executor=pool)\n"               //
    "  >>> pool.drain()";

/**
 *  @brief Wraps a fresh handle, or frees it when the wrapper cannot be allocated.
 *  @return A new reference, or NULL with an exception set.
 */
static PyObject* executor_adopt(PyTypeObject* type, usearch_executor_t handle) {
    Executor* const executor = (Executor*)type->tp_alloc(type, 0);
    if (!executor) {
        usearch_executor_free(handle);
        return NULL;
    }
    executor->handle = handle;
    return (PyObject*)executor;
}

static PyObject* Executor_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_TypeError,
                    "Executor is built through Executor.for_ram(), .on_pool(), .for_vram() or .for_nvme()");
    return NULL;
}

static void Executor_dealloc(Executor* executor) {
    // Freeing drains, which may wait on deferred calls.
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_free(executor->handle);
    Py_END_ALLOW_THREADS;
    Py_TYPE(executor)->tp_free((PyObject*)executor);
}

int usearch_py_executor_from_object(PyObject* object, usearch_executor_t* executor) {
    if (object == Py_None) {
        *executor = NULL;
        return 0;
    }
    if (!PyObject_TypeCheck(object, &ExecutorType)) {
        PyErr_Format(PyExc_TypeError, "The 'executor' option must be an Executor or None, got %S", Py_TYPE(object));
        return -1;
    }
    *executor = ((Executor*)object)->handle;
    return 0;
}

static char const doc_Executor_for_ram[] =                                                    //
    "for_ram(threads=0, *, memory_domains=None, in_flight_limit=0) -> Executor\n"             //
    "\n"                                                                                      //
    "Spawns host worker threads for a RAM-resident index.\n"                                  //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  threads (int, optional): Workers to spawn; 0 derives from the hardware.\n"             //
    "  memory_domains (Sequence[int], optional): NUMA node ids the workers stay local\n"      //
    "    to; None derives every node the process may allocate on.\n"                          //
    "  in_flight_limit (int, optional): Deferred calls tracked at once; 0 takes a default.\n" //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> pool = usearch.Executor.for_ram(threads=8)\n"                                      //
    "  >>> pool.drain()";

static PyObject* Executor_for_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                  PyObject* args_names_tuple) {
    size_t threads = 0, in_flight_limit = 0;
    PyObject* memory_domains_object = Py_None;

    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "for_ram() takes at most one positional argument");
        return NULL;
    }
    if (positional_args_count == 1 && usearch_py_size_from_object(args[0], "threads", &threads) != 0) return NULL;

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "threads") == 0) {
            if (usearch_py_size_from_object(value, "threads", &threads) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "memory_domains") == 0) { memory_domains_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "in_flight_limit") == 0) {
            if (usearch_py_size_from_object(value, "in_flight_limit", &in_flight_limit) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    if (usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        return NULL;

    usearch_executor_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_executor_for_ram_t arguments = {.executor = &handle,
                                            .outcome = &outcome,
                                            .ram = {memory_domains, memory_domains_count},
                                            .threads = threads,
                                            .in_flight_limit = in_flight_limit};
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_for_ram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);

    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle);
}

static char const doc_Executor_on_pool[] =                                                    //
    "on_pool(pool, *, memory_domains=None, in_flight_limit=0) -> Executor\n"                  //
    "\n"                                                                                      //
    "Drives RAM-resident indexes on a ForkUnion pool another library already spawned, so\n"   //
    "one process runs one set of threads while each library keeps its own scratch. The\n"     //
    "pool keeps its own threads and its lender's mood.\n"                                     //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  pool (int): The address of the `fu_pool_t` to borrow, as `ctypes` and `cffi` hand\n"   //
    "    pointers over; the pool must outlive the executor.\n"                                //
    "  memory_domains (Sequence[int], optional): NUMA node ids of the indexes it will\n"      //
    "    drive; None derives every node the process may allocate on.\n"                       //
    "  in_flight_limit (int, optional): Deferred calls tracked at once; 0 takes a default.\n" //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> pool = usearch.Executor.on_pool(fu_pool_address)   # doctest: +SKIP";

static PyObject* Executor_on_pool(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                  PyObject* args_names_tuple) {
    size_t in_flight_limit = 0;
    PyObject* memory_domains_object = Py_None;

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("on_pool", positional_args_count,
                                                                            args_names_tuple, 1);
    if (args_names_count < 0) return NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "memory_domains") == 0) { memory_domains_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "in_flight_limit") == 0) {
            if (usearch_py_size_from_object(value, "in_flight_limit", &in_flight_limit) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    PyObject* const address = PyNumber_Index(args[0]);
    if (!address) return NULL;
    void* const pool = PyLong_AsVoidPtr(address);
    Py_DECREF(address);
    if (!pool && PyErr_Occurred()) return NULL;

    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    if (usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        return NULL;

    usearch_executor_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_executor_on_pool_t arguments = {.executor = &handle,
                                            .outcome = &outcome,
                                            .ram = {memory_domains, memory_domains_count},
                                            .pool = pool,
                                            .in_flight_limit = in_flight_limit};
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_on_pool(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);

    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle);
}

static char const doc_Executor_for_vram[] =                                              //
    "for_vram(*, devices=None, in_flight_limit=0) -> Executor\n"                         //
    "\n"                                                                                 //
    "Opens one stream per device for a VRAM-resident index.\n"                           //
    "\n"                                                                                 //
    "Takes no thread count: a single host thread drives every stream, so the width of\n" //
    "the pool is the device list rather than a worker count.\n"                          //
    "\n"                                                                                 //
    "Args:\n"                                                                            //
    "  devices (Sequence[int], optional): CUDA ordinals to drive; None derives one\n"    //
    "    free device.\n"                                                                 //
    "  in_flight_limit (int, optional): Deferred calls tracked at once; 0 takes a\n"     //
    "    default.\n"                                                                     //
    "\n"                                                                                 //
    "Example:\n"                                                                         //
    "  >>> pool = usearch.Executor.for_vram(devices=[0, 1])   # doctest: +SKIP";

static PyObject* Executor_for_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject* devices_object = Py_None;
    size_t in_flight_limit = 0;

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("for_vram", positional_args_count,
                                                                            args_names_tuple, 0);
    if (args_names_count < 0) return NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "devices") == 0) { devices_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "in_flight_limit") == 0) {
            if (usearch_py_size_from_object(value, "in_flight_limit", &in_flight_limit) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    int32_t* devices = NULL;
    size_t devices_count = 0;
    if (usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0) return NULL;

    usearch_executor_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_executor_for_vram_t arguments = {.executor = &handle,
                                             .outcome = &outcome,
                                             .vram = {devices, devices_count},
                                             .in_flight_limit = in_flight_limit};
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_for_vram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(devices);

    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle);
}

static char const doc_Executor_for_nvme[] =                                             //
    "for_nvme(threads=0, *, paths=None, in_flight_limit=0) -> Executor\n"               //
    "\n"                                                                                //
    "Opens one submission ring per worker for an NVMe-resident index.\n"                //
    "\n"                                                                                //
    "Args:\n"                                                                           //
    "  threads (int, optional): Rings to open; 0 derives from the hardware.\n"          //
    "  paths (Sequence[str | bytes | PathLike], optional): Devices or directories to\n" //
    "    stripe shards across; None leaves every shard where its manifest points. A\n"  //
    "    path may name a kernel device, a transport identifier, or a FUSE mount.\n"     //
    "  in_flight_limit (int, optional): Deferred calls tracked at once; 0 takes a\n"    //
    "    default.\n"                                                                    //
    "\n"                                                                                //
    "Example:\n"                                                                        //
    "  >>> pool = usearch.Executor.for_nvme(threads=8, paths=['/mnt/nvme0'])  # doctest: +SKIP";

static PyObject* Executor_for_nvme(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    size_t threads = 0, in_flight_limit = 0;
    PyObject* paths_object = Py_None;

    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "for_nvme() takes at most one positional argument");
        return NULL;
    }
    if (positional_args_count == 1 && usearch_py_size_from_object(args[0], "threads", &threads) != 0) return NULL;

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "threads") == 0) {
            if (usearch_py_size_from_object(value, "threads", &threads) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "paths") == 0) { paths_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "in_flight_limit") == 0) {
            if (usearch_py_size_from_object(value, "in_flight_limit", &in_flight_limit) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    usearch_py_paths_arena_t paths;
    if (usearch_py_paths_from_object(paths_object, &paths) != 0) return NULL;
    usearch_executor_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_executor_for_nvme_t arguments = {.executor = &handle,
                                             .outcome = &outcome,
                                             .nvme = {paths.views, paths.count},
                                             .threads = threads,
                                             .in_flight_limit = in_flight_limit};
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_for_nvme(&arguments);
    Py_END_ALLOW_THREADS;
    usearch_py_paths_release(&paths);

    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle);
}

/** What an executor does with its workers once idle; the C interface names no table to read these from. */
static usearch_py_choice_naming_t const mood_namings[] = {
    {"chill", usearch_mood_chill_k},
    {"grind", usearch_mood_grind_k},
};

static char const doc_Executor_drain[] =                                                    //
    "drain(mood='chill') -> None\n"                                                         //
    "\n"                                                                                    //
    "Waits for every call submitted to this executor before the drain began, then sets\n"   //
    "what it does with its workers whenever it next goes idle; the last drain wins.\n"      //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  mood (str, optional): 'chill' parks the workers, so the next call pays a wake-up;\n" //
    "    'grind' keeps them spinning, trading power for the next call's latency. A\n"       //
    "    borrowed pool keeps its lender's mood.\n"                                          //
    "Raises:\n"                                                                             //
    "  Error: What befell the executor itself, such as a lost device; every call\n"         //
    "    reports its own failure through its own verb.";

static PyObject* Executor_drain(Executor* executor, PyObject* const* args, Py_ssize_t positional_args_count,
                                PyObject* args_names_tuple) {
    PyObject* mood_object = NULL;
    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "drain() takes at most one positional argument");
        return NULL;
    }
    if (positional_args_count == 1) mood_object = args[0];
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        if (PyUnicode_CompareWithASCIIString(name, "mood") == 0) { mood_object = args[args_progress]; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }
    int mood = usearch_mood_chill_k;
    if (mood_object && usearch_py_choice_from_object(mood_object, "mood", mood_namings,
                                                     usearch_py_table_size_(mood_namings), &mood) != 0)
        return NULL;

    usearch_outcome_t outcome = {0};
    usearch_executor_drain_t arguments = {.executor = executor->handle,
                                          .outcome = &outcome,
                                          .mood = (usearch_mood_t)mood};
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_drain(&arguments);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject* Executor_repr(Executor* executor) {
    return PyUnicode_FromFormat("usearch.Executor(at=%p)", (void*)executor->handle);
}

static PyMethodDef Executor_methods[] = {
    {"for_ram", (PyCFunction)Executor_for_ram, usearch_py_method_flags_k | METH_CLASS, doc_Executor_for_ram},    //
    {"on_pool", (PyCFunction)Executor_on_pool, usearch_py_method_flags_k | METH_CLASS, doc_Executor_on_pool},    //
    {"for_vram", (PyCFunction)Executor_for_vram, usearch_py_method_flags_k | METH_CLASS, doc_Executor_for_vram}, //
    {"for_nvme", (PyCFunction)Executor_for_nvme, usearch_py_method_flags_k | METH_CLASS, doc_Executor_for_nvme}, //
    {"drain", (PyCFunction)Executor_drain, usearch_py_method_flags_k, doc_Executor_drain},                       //
    {NULL, NULL, 0, NULL},
};

PyTypeObject ExecutorType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.Executor",
    .tp_doc = doc_Executor,
    .tp_basicsize = sizeof(Executor),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Executor_tp_new,
    .tp_dealloc = (destructor)Executor_dealloc,
    .tp_methods = Executor_methods,
    .tp_repr = (reprfunc)Executor_repr,
};

#pragma endregion Executors

#pragma region Arenas

static char const doc_Arena[] =                                                               //
    "Arena()\n"                                                                               //
    "\n"                                                                                      //
    "The memory a verb lands the outputs only it can size in - `get`, `multiply` and\n"       //
    "`save_buffer` - handed back to the next call so a steady loop allocates once.\n"         //
    "\n"                                                                                      //
    "The outputs leave as zero-copy memoryviews that keep the arena alive. Passing it to\n"   //
    "another call rewinds it, so that call raises `ContendedError` while any view over the\n" //
    "previous outputs is still held; release them, or pass another arena. A verb handed\n"    //
    "none uses a fresh one that its outputs alone keep alive.\n"                              //
    "\n"                                                                                      //
    "The memory lives where the executor of the call that first filled it computes, and a\n"  //
    "call on an executor of another residency raises `DeviceError`.\n"                        //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> from array import array\n"                                                         //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=2)\n"                                //
    "  >>> index.reserve(2)\n"                                                                //
    "  >>> index.add([1], array('d', [0.1, 0.2]))\n"                                          //
    "  >>> index.add([2], array('d', [0.3, 0.4]))\n"                                          //
    "  >>> arena = usearch.Arena()\n"                                                         //
    "  >>> first = index.get(1, arena=arena)\n"                                               //
    "  >>> first.release()\n"                                                                 //
    "  >>> index.get(2, arena=arena).shape\n"                                                 //
    "  (1, 2)";

static PyObject* Arena_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    if (PyTuple_GET_SIZE(args) != 0 || (keywords && PyDict_GET_SIZE(keywords) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Arena() takes no arguments");
        return NULL;
    }
    return type->tp_alloc(type, 0); //! Zeroed: no handle until a call creates one, and nothing exported.
}

static void Arena_dealloc(Arena* self) {
    Py_BEGIN_ALLOW_THREADS;
    usearch_arena_free(self->handle);
    Py_END_ALLOW_THREADS;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyTypeObject ArenaType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.Arena",
    .tp_doc = doc_Arena,
    .tp_basicsize = sizeof(Arena),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Arena_tp_new,
    .tp_dealloc = (destructor)Arena_dealloc,
};

/** Adds @p delta to the exports of @p arena, fenced against views released on other threads. */
static Py_ssize_t arena_exports_add(Arena* arena, Py_ssize_t delta) {
    Py_ssize_t exports = 0;
    usearch_py_critical_section_begin_((PyObject*)arena);
    exports = arena->exports += delta;
    usearch_py_critical_section_end_();
    return exports;
}

int usearch_py_arena_claim(PyObject* object, PyObject** arena) {
    if (!object || object == Py_None) {
        object = ArenaType.tp_alloc(&ArenaType, 0);
        if (!object) return -1;
    }
    else if (!PyObject_TypeCheck(object, &ArenaType)) {
        PyErr_Format(PyExc_TypeError, "The 'arena' option must be an Arena or None, got %S", Py_TYPE(object));
        return -1;
    }
    else { Py_INCREF(object); }

    Arena* const claimed = (Arena*)object;
    Py_ssize_t held = 0;
    usearch_py_critical_section_begin_(object);
    held = claimed->exports;
    if (!held) claimed->exports = 1;
    usearch_py_critical_section_end_();
    if (held) {
        Py_DECREF(object);
        return usearch_py_raise(usearch_status_contended_k,
                                "the arena still serves a call or views over its outputs; release them first");
    }
    *arena = object;
    return 0;
}

void usearch_py_arena_unclaim(PyObject* arena) {
    arena_exports_add((Arena*)arena, -1);
    Py_DECREF(arena);
}

/** One output a call left in an arena, the exporter behind the view a verb returns. */
typedef struct ArenaBuffer {
    PyObject ob_base;

    /** The arena the output lives in, kept alive and counted against reuse while exported. */
    PyObject* arena;

    void* data;
    char const* format;
    Py_ssize_t item_bytes;
    int rank;
    Py_ssize_t shape[usearch_py_max_rank_k];
    Py_ssize_t strides[usearch_py_max_rank_k];
} ArenaBuffer;

static int ArenaBuffer_getbuffer(ArenaBuffer* self, Py_buffer* view, int flags) {
    Py_ssize_t bytes = self->item_bytes;
    for (int axis = 0; axis != self->rank; ++axis) bytes *= self->shape[axis];
    int const formatted = (flags & PyBUF_FORMAT) == PyBUF_FORMAT;
    view->obj = Py_NewRef((PyObject*)self);
    view->buf = self->data;
    view->len = bytes;
    view->readonly = 0;
    view->itemsize = formatted ? self->item_bytes : 1;
    view->format = formatted ? (char*)self->format : NULL;
    view->ndim = self->rank;
    view->shape = (flags & PyBUF_ND) == PyBUF_ND ? self->shape : NULL;
    view->strides = (flags & PyBUF_STRIDES) == PyBUF_STRIDES ? self->strides : NULL;
    view->suboffsets = NULL;
    view->internal = NULL;
    arena_exports_add((Arena*)self->arena, 1);
    return 0;
}

static void ArenaBuffer_releasebuffer(ArenaBuffer* self, Py_buffer* view) {
    arena_exports_add((Arena*)self->arena, -1);
}

static void ArenaBuffer_dealloc(ArenaBuffer* self) {
    Py_XDECREF(self->arena);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyBufferProcs ArenaBuffer_as_buffer = {
    .bf_getbuffer = (getbufferproc)ArenaBuffer_getbuffer,
    .bf_releasebuffer = (releasebufferproc)ArenaBuffer_releasebuffer,
};

PyTypeObject ArenaBufferType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.ArenaBuffer",
    .tp_doc = "One output a call left in an Arena, exported zero-copy.",
    .tp_basicsize = sizeof(ArenaBuffer),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, //! Behind the views a verb returns, never constructed by name.
    .tp_dealloc = (destructor)ArenaBuffer_dealloc,
    .tp_as_buffer = &ArenaBuffer_as_buffer,
};

PyObject* usearch_py_arena_view(PyObject* arena, void* data, char const* format, Py_ssize_t item_bytes,
                                Py_ssize_t const* shape, int rank) {
    // A call with nothing to report leaves its output null, which no exporter may hand out.
    static char empty_output;
    ArenaBuffer* const exporter = PyObject_New(ArenaBuffer, &ArenaBufferType);
    if (!exporter) return NULL;
    exporter->arena = Py_NewRef(arena);
    exporter->data = data ? data : &empty_output;
    exporter->format = format;
    exporter->item_bytes = item_bytes;
    exporter->rank = rank;
    Py_ssize_t stride = item_bytes;
    for (int axis = rank - 1; axis >= 0; --axis) {
        exporter->shape[axis] = shape[axis];
        exporter->strides[axis] = stride;
        stride *= shape[axis];
    }
    PyObject* const view = PyMemoryView_FromObject((PyObject*)exporter);
    Py_DECREF(exporter);
    return view;
}

#pragma endregion Arenas

#pragma region Paths

int usearch_py_path_from_object(PyObject* object, usearch_bytes_view_t* path, PyObject** decoded) {
    PyObject* bytes_object = NULL;
    if (!PyUnicode_FSConverter(object, &bytes_object)) return -1;
    path->data = (uint8_t const*)PyBytes_AS_STRING(bytes_object);
    path->length = (size_t)PyBytes_GET_SIZE(bytes_object);
    *decoded = bytes_object;
    return 0;
}

#pragma endregion Paths

#pragma region Library Probes

char const doc_version[] = //
    "version() -> str\n"   //
    "\n"                   //
    "The library's semantic version.";

char const doc_capabilities_compiled[] = //
    "capabilities_compiled() -> str\n"   //
    "\n"                                 //
    "The kernel families this build carries, whatever the machine offers.";

char const doc_capabilities_available[] = //
    "capabilities_available() -> str\n"   //
    "\n"                                  //
    "The kernel families this machine can actually run, and dispatch picks from.";

PyObject* api_version(PyObject* self, PyObject* unused) { return PyUnicode_FromString(usearch_version_name()); }

PyObject* api_capabilities_compiled(PyObject* self, PyObject* unused) {
    return PyUnicode_FromString(usearch_capabilities_compiled());
}

PyObject* api_capabilities_available(PyObject* self, PyObject* unused) {
    return PyUnicode_FromString(usearch_capabilities_available());
}

#pragma endregion Library Probes
