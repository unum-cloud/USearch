/**
 *  @brief The vocabulary every modality shares - the CPython twin of `usearch/types.h`.
 *  @file python/types.c
 *  @author Ash Vardanian
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

char const* usearch_status_name(usearch_status_t status) {
    switch (status) {
    case usearch_success_k: return "success";
    case usearch_error_out_of_memory_k: return "out-of-memory";
    case usearch_error_shape_mismatch_k: return "shape-mismatch";
    case usearch_error_unsupported_layout_k: return "unsupported-layout";
    case usearch_error_unknown_kind_k: return "unknown-kind";
    case usearch_error_read_only_k: return "read-only";
    case usearch_error_io_k: return "io";
    case usearch_error_device_k: return "device";
    case usearch_error_malformed_input_k: return "malformed-input";
    case usearch_error_internal_k: return "internal";
    case usearch_error_overflow_k: return "overflow";
    case usearch_error_uninitialized_k: return "uninitialized";
    case usearch_error_config_mismatch_k: return "config-mismatch";
    case usearch_error_contended_k: return "contended";
    default: return "unrecognized";
    }
}

int usearch_raise(usearch_status_t status, char const* message) {
    // A status this specific supersedes whatever the caller was carrying, and building the
    // instance below must not start with an exception already pending.
    PyErr_Clear();
    PyObject* const instance = PyObject_CallFunction(ErrorType, "s", message ? message : "");
    if (!instance) return -1;
    PyObject* const status_name = PyUnicode_FromString(usearch_status_name(status));
    if (!status_name) {
        Py_DECREF(instance);
        return -1;
    }
    // A status-less `Error` would misreport the fault, so a failure here is reported as itself
    // rather than papered over with a half-built instance.
    int const attached = PyObject_SetAttrString(instance, "status", status_name);
    Py_DECREF(status_name);
    if (attached != 0) {
        Py_DECREF(instance);
        return -1;
    }
    PyErr_SetObject(ErrorType, instance);
    Py_DECREF(instance);
    return -1;
}

int usearch_raise_on_error(usearch_error_t const* error) {
    if (error->status == usearch_success_k) return 0;
    return usearch_raise(error->status, error->message);
}

static char const doc_Error[] =                                                         //
    "Error(message)\n"                                                                  //
    "\n"                                                                                //
    "Every failure the engine reports, carrying the status shared with the C API.\n"    //
    "A failure caught before the boundary reports the same identity the engine would\n" //
    "have reported for the same fault.\n"                                               //
    "\n"                                                                                //
    "Attributes:\n"                                                                     //
    "  status (str): Lowercase, dash-separated status, such as 'shape-mismatch'.\n"     //
    "  args (tuple): The detail the status cannot carry, as on any exception.";

int usearch_register_error_type(PyObject* module) {
    ErrorType = PyErr_NewExceptionWithDoc("usearch.Error", (char*)doc_Error, PyExc_RuntimeError, NULL);
    if (!ErrorType) return -1;
    Py_INCREF(ErrorType);
    if (PyModule_AddObject(module, "Error", ErrorType) < 0) {
        Py_DECREF(ErrorType);
        return -1;
    }
    return 0;
}

#pragma endregion Failures

#pragma region Scalar Names

/** @brief One scalar kind, and every spelling a caller might hand us for it. */
typedef struct usearch_scalar_naming_t {
    usearch_scalar_t kind;
    char const* name;            //! The canonical name, as the stubs and constructor spell it.
    char const* numpy_typestr;   //! The NumPy array-interface spelling, or NULL when none exists.
    char const* buffer_typestr;  //! The PEP 3118 format string, or NULL when none exists.
    unsigned char values_packed; //! Logical scalars per storage unit; one for byte-wide kinds.
    unsigned char unit_bytes;    //! Bytes one storage unit occupies.
} usearch_scalar_naming_t;

/**
 *  @brief Ordered by shrinking dynamic range - floats widest exponent first, then signed
 *      integers descending, then unsigned - the order every dtype-indexed list in the
 *      project takes.
 */
static usearch_scalar_naming_t const scalar_namings[] = {
    {usearch_scalar_f64_k, "f64", "<f8", "d", 1, 8},    {usearch_scalar_f32_k, "f32", "<f4", "f", 1, 4},
    {usearch_scalar_bf16_k, "bf16", "<V2", NULL, 1, 2}, {usearch_scalar_f16_k, "f16", "<f2", "e", 1, 2},
    {usearch_scalar_e5m2_k, "e5m2", "|V1", NULL, 1, 1}, {usearch_scalar_e4m3_k, "e4m3", "|V1", NULL, 1, 1},
    {usearch_scalar_e3m2_k, "e3m2", "|V1", NULL, 1, 1}, {usearch_scalar_e2m3_k, "e2m3", "|V1", NULL, 1, 1},
    {usearch_scalar_i8_k, "i8", "|i1", "b", 1, 1},      {usearch_scalar_i4x2_k, "i4x2", "|V1", NULL, 2, 1},
    {usearch_scalar_u8_k, "u8", "|u1", "B", 1, 1},      {usearch_scalar_u4x2_k, "u4x2", "|V1", NULL, 2, 1},
    {usearch_scalar_u1x8_k, "u1x8", "|V1", "?", 8, 1},
};

static size_t const scalar_namings_count = usearch_py_table_size_(scalar_namings);

/** @brief Aliases the ecosystem spells differently than we do, resolved before the table. */
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

static size_t const scalar_aliases_count = usearch_py_table_size_(scalar_aliases);

static usearch_scalar_naming_t const* scalar_naming(usearch_scalar_t kind) {
    for (size_t naming_index = 0; naming_index != scalar_namings_count; ++naming_index)
        if (scalar_namings[naming_index].kind == kind) return &scalar_namings[naming_index];
    return NULL;
}

static int same_string_sized(char const* input, Py_ssize_t input_length, char const* literal) {
    size_t const literal_length = strlen(literal);
    return (size_t)input_length == literal_length && memcmp(input, literal, literal_length) == 0;
}

usearch_scalar_t usearch_scalar_from_name(char const* name, Py_ssize_t length) {
    if (!name || length <= 0) return usearch_scalar_unknown_k;
    for (size_t naming_index = 0; naming_index != scalar_namings_count; ++naming_index) {
        usearch_scalar_naming_t const* const naming = &scalar_namings[naming_index];
        if (same_string_sized(name, length, naming->name)) return naming->kind;
        if (naming->buffer_typestr && same_string_sized(name, length, naming->buffer_typestr)) return naming->kind;
        if (naming->numpy_typestr && same_string_sized(name, length, naming->numpy_typestr)) return naming->kind;
    }
    for (size_t alias_index = 0; alias_index != scalar_aliases_count; ++alias_index)
        if (same_string_sized(name, length, scalar_aliases[alias_index].name)) return scalar_aliases[alias_index].kind;
    return usearch_scalar_unknown_k;
}

char const* usearch_scalar_name(usearch_scalar_t kind) {
    usearch_scalar_naming_t const* const naming = scalar_naming(kind);
    return naming ? naming->name : "unknown";
}

size_t usearch_scalar_row_bytes(usearch_scalar_t kind, size_t columns) {
    usearch_scalar_naming_t const* const naming = scalar_naming(kind);
    if (!naming) return 0;
    size_t const units = usearch_py_divide_round_up_(columns, naming->values_packed);
    return units * naming->unit_bytes;
}

char const* usearch_scalar_buffer_format(usearch_scalar_t kind) {
    usearch_scalar_naming_t const* const naming = scalar_naming(kind);
    return naming && naming->values_packed == 1 ? naming->buffer_typestr : NULL;
}

PyObject* usearch_arena_matrix(PyObject* arena, size_t rows, size_t columns, usearch_scalar_t kind) {
    Py_ssize_t occupied = 0;
    if (usearch_matrix_bytes(rows, usearch_scalar_row_bytes(kind, columns), 1, &occupied) != 0) return NULL;

    PyObject* const flat = PyMemoryView_FromObject(arena);
    if (!flat) return NULL;
    // `cast` demands the shape account for the whole buffer, so an arena reserved for more rows
    // than were filled is sliced down first. The slice is a view; nothing is copied.
    PyObject* const occupied_view = PySequence_GetSlice(flat, 0, occupied);
    Py_DECREF(flat);
    if (!occupied_view) return NULL;

    char const* const format = usearch_scalar_buffer_format(kind);
    Py_ssize_t const width = format ? (Py_ssize_t)columns : (Py_ssize_t)usearch_scalar_row_bytes(kind, columns);
    PyObject* const shaped = PyObject_CallMethod(occupied_view, "cast", "s(nn)", format ? format : "B",
                                                 (Py_ssize_t)rows, width);
    Py_DECREF(occupied_view);
    return shaped;
}

int usearch_matrix_bytes(size_t rows, size_t columns, size_t unit, Py_ssize_t* bytes) {
    size_t const cells = rows * columns;
    if (columns && cells / columns != rows)
        return usearch_raise(usearch_error_overflow_k, "the requested output matrix does not fit memory");
    size_t const total = cells * unit;
    if (unit && total / unit != cells)
        return usearch_raise(usearch_error_overflow_k, "the requested output matrix does not fit memory");
    // `PTRDIFF_MAX` rather than `PY_SSIZE_T_MAX`, which resolves to a POSIX macro a strict-C
    // translation unit does not see; `Py_ssize_t` is a `ptrdiff_t` either way.
    if (total > (size_t)PTRDIFF_MAX)
        return usearch_raise(usearch_error_overflow_k, "the requested output matrix does not fit memory");
    *bytes = (Py_ssize_t)total;
    return 0;
}

int usearch_scalar_from_object(PyObject* object, usearch_scalar_t* kind) {
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
    usearch_scalar_t const resolved = usearch_scalar_from_name(name, length);
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
static int buffer_via_array_interface(PyObject* object, usearch_buffer_backing_t* backing) {
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
            if (name) kind = usearch_scalar_from_name(name, name_length);
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
            if (typestr) kind = usearch_scalar_from_name(typestr, typestr_length);
        }
    }
    usearch_scalar_naming_t const* const naming = scalar_naming(kind);
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
static int buffer_acquire(PyObject* object, usearch_matrix_access_t access, usearch_buffer_backing_t* backing) {
    backing->synthesized_owner = NULL;
    int const flags = PyBUF_STRIDES | PyBUF_FORMAT | (access == usearch_matrix_access_write_k ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, &backing->buffer, flags) == 0) return 0;
    PyErr_Clear();
    if (buffer_via_array_interface(object, backing)) {
        // The fallback bypasses the flag checks `PyObject_GetBuffer` would have made.
        if (access == usearch_matrix_access_write_k && backing->buffer.readonly) {
            usearch_matrix_release(backing);
            PyErr_SetString(PyExc_BufferError, "Expected a writable buffer");
            return -1;
        }
        return 0;
    }
    if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError, "Expected an object supporting the buffer protocol or __array_interface__");
    return -1;
}

int usearch_object_is_buffer(PyObject* object) {
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

void usearch_matrix_release(usearch_buffer_backing_t* backing) {
    PyBuffer_Release(&backing->buffer);
    Py_CLEAR(backing->synthesized_owner);
}

int usearch_buffer_borrow_exact(PyObject* object, char const* field, usearch_matrix_access_t access, Py_ssize_t bytes,
                                usearch_buffer_backing_t* backing, void** data) {
    backing->synthesized_owner = NULL;
    int const flags = PyBUF_C_CONTIGUOUS | (access == usearch_matrix_access_write_k ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, &backing->buffer, flags) != 0) {
        // The exporter's own diagnosis is more precise than any guess we could make, so it
        // rides along as the cause rather than being replaced.
        _PyErr_FormatFromCause(PyExc_ValueError, "The '%s' buffer must be contiguous%s", field,
                               access == usearch_matrix_access_write_k ? " and writable" : "");
        return -1;
    }
    if (backing->buffer.len != bytes) {
        usearch_matrix_release(backing);
        PyErr_Format(PyExc_ValueError, "The '%s' buffer must hold exactly %zd bytes, not %zd", field, bytes,
                     backing->buffer.len);
        return -1;
    }
    *data = backing->buffer.buf;
    return 0;
}

int usearch_matrix_from_object(PyObject* object, usearch_matrix_access_t access, usearch_scalar_t scalar_hint,
                               usearch_matrix_view_t* view, usearch_buffer_backing_t* backing) {
    if (buffer_acquire(object, access, backing) != 0) return -1;
    Py_buffer const* const buffer = &backing->buffer;

    if (buffer->ndim < 1 || buffer->ndim > usearch_py_max_rank_k) {
        PyErr_Format(PyExc_ValueError, "Expected a vector or a matrix, got rank %d", buffer->ndim);
        usearch_matrix_release(backing);
        return -1;
    }

    usearch_scalar_t kind = scalar_hint;
    if (kind == usearch_scalar_unknown_k && buffer->format)
        kind = usearch_scalar_from_name(buffer->format, (Py_ssize_t)strlen(buffer->format));
    if (kind == usearch_scalar_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unsupported '%s' scalar kind; pass `dtype=` to name it",
                     buffer->format ? buffer->format : "?");
        usearch_matrix_release(backing);
        return -1;
    }

    // Only the row pitch may stride: a strided column would force a gather the kernels have
    // no form for, and the C surface has no argument to describe one.
    int const last_axis = buffer->ndim - 1;
    if (buffer->strides && buffer->strides[last_axis] != buffer->itemsize) {
        PyErr_SetString(PyExc_ValueError, "Expected a matrix whose rows are contiguous; only the pitch may stride");
        usearch_matrix_release(backing);
        return -1;
    }

    view->data = buffer->buf;
    view->scalar_kind = kind;
    view->readonly = buffer->readonly;
    if (buffer->ndim == 1) {
        view->rows = 1;
        view->columns = (size_t)buffer->shape[0];
        view->pitch = (size_t)buffer->len;
    }
    else {
        view->rows = (size_t)buffer->shape[0];
        view->columns = (size_t)buffer->shape[1];
        Py_ssize_t const pitch = buffer->strides ? buffer->strides[0] : buffer->shape[1] * buffer->itemsize;
        if (pitch < 0) {
            PyErr_SetString(PyExc_ValueError, "Matrix rows must ascend in memory");
            usearch_matrix_release(backing);
            return -1;
        }
        view->pitch = (size_t)pitch;
    }
    return 0;
}

#pragma endregion Buffer Acquisition

#pragma region Keys

/** @brief A 128-bit entry identity, rendered as a lowercase dashed UUID. */
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

int usearch_key_from_object(PyObject* object, usearch_key_t* key) {
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
            return usearch_raise(usearch_error_malformed_input_k, "the key is not thirty-two hexadecimal digits");
        return 0;
    }
    // `uuid.UUID` and anything else shaped like it, without importing the module.
    PyObject* const raw_bytes = PyObject_GetAttrString(object, "bytes");
    if (raw_bytes) {
        int const outcome = usearch_key_from_object(raw_bytes, key);
        Py_DECREF(raw_bytes);
        return outcome;
    }
    PyErr_Clear();
    PyErr_Format(PyExc_TypeError, "Cannot read a key from %S", Py_TYPE(object));
    return -1;
}

PyObject* usearch_key_to_object(usearch_key_t key) {
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
    if (value && usearch_key_from_object(value, &instance->key) != 0) {
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
    if (usearch_key_from_object(first, &left) != 0 || usearch_key_from_object(second, &right) != 0) {
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

void usearch_keys_release(usearch_key_arena_t* arena) {
    if (arena->storage == usearch_key_storage_borrowed_k) usearch_matrix_release(&arena->backing);
    else if (arena->owned) PyMem_Free(arena->owned);
    arena->owned = NULL;
    arena->keys = NULL;
    arena->storage = usearch_key_storage_owned_k;
}

/** @brief Which storage a key batch will use, once its argument has been looked at. */
typedef enum keys_route_t {
    /** @brief An (N, 16) contiguous identity buffer; the arena borrows it outright. */
    keys_route_borrowed_k,
    /** @brief Not that shape, so each element converts on its own terms. */
    keys_route_sequence_k,
    /** @brief Something went wrong and an exception is set. */
    keys_route_failed_k,
} keys_route_t;

/**
 *  @brief Tries the zero-copy route, distinguishing "not an identity buffer" from a real failure.
 *
 *  A blanket clear here would swallow a @c KeyboardInterrupt delivered inside an exporter, so
 *  only the failure this function itself provoked is discarded.
 */
static keys_route_t keys_borrow_buffer(PyObject* object, size_t expected, usearch_key_arena_t* arena) {
    if (!PyObject_CheckBuffer(object)) return keys_route_sequence_k;
    if (PyObject_GetBuffer(object, &arena->backing.buffer, PyBUF_STRIDES) != 0) {
        PyErr_Clear();
        return keys_route_sequence_k;
    }
    arena->backing.synthesized_owner = NULL;
    arena->storage = usearch_key_storage_borrowed_k;
    Py_buffer const* const buffer = &arena->backing.buffer;
    if (buffer->ndim != 2 || buffer->shape[1] != (Py_ssize_t)sizeof(usearch_key_t) || buffer->itemsize != 1 ||
        !PyBuffer_IsContiguous((Py_buffer*)buffer, 'C')) {
        usearch_keys_release(arena);
        return keys_route_sequence_k;
    }
    if ((size_t)buffer->shape[0] != expected) {
        usearch_keys_release(arena);
        usearch_raise(usearch_error_shape_mismatch_k, "the key count does not match the vector row count");
        return keys_route_failed_k;
    }
    arena->keys = (usearch_key_t*)buffer->buf;
    return keys_route_borrowed_k;
}

int usearch_keys_from_object(PyObject* object, size_t expected, usearch_key_arena_t* arena) {
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
        usearch_raise(usearch_error_shape_mismatch_k, "the key count does not match the vector row count");
        return -1;
    }

    arena->owned = (usearch_key_t*)PyMem_Malloc(expected ? expected * sizeof(usearch_key_t) : 1);
    if (!arena->owned) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t entry_index = 0; entry_index != count; ++entry_index) {
        if (usearch_key_from_object(PySequence_Fast_GET_ITEM(sequence, entry_index), &arena->owned[entry_index]) != 0) {
            Py_DECREF(sequence);
            usearch_keys_release(arena);
            return -1;
        }
    }
    Py_DECREF(sequence);
    arena->keys = arena->owned;
    return 0;
}

#pragma endregion Key Batches

#pragma region Telemetry

static char const doc_Stats[] =                                                 //
    "Stats()\n"                                                                 //
    "\n"                                                                        //
    "Traversal telemetry, accumulated across the calls it is passed to.\n"      //
    "\n"                                                                        //
    "Attributes:\n"                                                             //
    "  visited_members (int): Entries whose neighborhoods the walk expanded.\n" //
    "  computed_distances (int): Distance evaluations the walk paid for.\n"     //
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

PyObject* usearch_stats_to_object(usearch_dense_traversal_stats_t const* stats) {
    Stats* const instance = PyObject_New(Stats, &StatsType);
    if (!instance) return NULL;
    instance->stats = *stats;
    return (PyObject*)instance;
}

int usearch_stats_accumulate(PyObject* target, usearch_dense_traversal_stats_t const* stats) {
    if (!target || target == Py_None) return 0;
    if (!PyObject_TypeCheck(target, &StatsType)) {
        PyErr_Format(PyExc_TypeError, "Expected a Stats object, got %S", Py_TYPE(target));
        return -1;
    }
    // The docstring invites sharing one `Stats` across concurrent calls, so the increments are
    // guarded rather than merely quick - the module declares it does not need the GIL.
    Stats* const accumulator = (Stats*)target;
    usearch_py_critical_section_begin_(target);
    accumulator->stats.visited_members += stats->visited_members;
    accumulator->stats.computed_distances += stats->computed_distances;
    accumulator->stats.restarts += stats->restarts;
    usearch_py_critical_section_end_();
    return 0;
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

PyTypeObject StatsType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.Stats",
    .tp_doc = doc_Stats,
    .tp_basicsize = sizeof(Stats),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Stats_tp_new,
    .tp_dealloc = (destructor)Stats_dealloc,
    .tp_repr = (reprfunc)Stats_repr,
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

PyObject* usearch_match_to_object(usearch_key_t key, usearch_distance_t distance) {
    Match* const instance = PyObject_New(Match, &MatchType);
    if (!instance) return NULL;
    instance->key = key;
    instance->distance = distance;
    return (PyObject*)instance;
}

static void Match_dealloc(Match* self) { Py_TYPE(self)->tp_free((PyObject*)self); }

static char const doc_Match_key[] = "The identity of the matched entry.";

static PyObject* Match_get_key(Match* self, void* closure) { return usearch_key_to_object(self->key); }

static char const doc_Match_distance[] = "The distance to the query, in the index's own metric.";

static PyObject* Match_get_distance(Match* self, void* closure) { return PyFloat_FromDouble(self->distance); }

static PyObject* Match_repr(Match* self) {
    PyObject* const key_object = usearch_key_to_object(self->key);
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

static char const doc_Matches[] =                                                           //
    "The result of a batch search: three matrices and the telemetry that produced them.\n"  //
    "\n"                                                                                    //
    "The fields are zero-copy views over arenas this object owns, so `np.asarray` costs\n"  //
    "nothing and NumPy is never required. No NumPy dtype expresses a 128-bit integer, so\n" //
    "keys arrive as big-endian sixteen-byte rows.\n"                                        //
    "\n"                                                                                    //
    "Attributes:\n"                                                                         //
    "  keys (memoryview): (queries, wanted, 16) identities, most significant byte first.\n" //
    "  distances (memoryview): (queries, wanted) distances.\n"                              //
    "  counts (memoryview): (queries,) match counts; entries past a count are zeroed.\n"    //
    "  visited_members (int): Entries whose neighborhoods the walk expanded.\n"             //
    "  computed_distances (int): Distance evaluations the walk paid for.\n"                 //
    "  restarts (int): Times the walk fell back to a fresh entry point.";

static void Matches_dealloc(Matches* self) {
    Py_XDECREF(self->keys_arena);
    Py_XDECREF(self->distances_arena);
    Py_XDECREF(self->counts_arena);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** @brief Wraps @p arena in a read-only `memoryview` reshaped to @p shape. */
static PyObject* arena_view(PyObject* arena, char const* format, Py_ssize_t const* shape, Py_ssize_t rank) {
    PyObject* const flat = PyMemoryView_FromObject(arena);
    if (!flat) return NULL;
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

static char const doc_Matches_keys[] = "(queries, wanted, 16) identities, most significant byte first.";

static PyObject* Matches_get_keys(Matches* self, void* closure) {
    Py_ssize_t const shape[3] = {(Py_ssize_t)self->queries, (Py_ssize_t)self->wanted, 16};
    return arena_view(self->keys_arena, "B", shape, 3);
}

static char const doc_Matches_distances[] = "(queries, wanted) distances, in the index's own metric.";

static PyObject* Matches_get_distances(Matches* self, void* closure) {
    Py_ssize_t const shape[2] = {(Py_ssize_t)self->queries, (Py_ssize_t)self->wanted};
    return arena_view(self->distances_arena, "d", shape, 2);
}

static char const doc_Matches_counts[] = "(queries,) match counts; entries past a count are zeroed.";

static PyObject* Matches_get_counts(Matches* self, void* closure) {
    Py_ssize_t const shape[1] = {(Py_ssize_t)self->queries};
    return arena_view(self->counts_arena, "N", shape, 1);
}

static char const doc_Matches_visited_members[] = "Entries whose neighborhoods the walk expanded.";

static PyObject* Matches_get_visited_members(Matches* self, void* closure) {
    return PyLong_FromSize_t(self->stats.visited_members);
}

static char const doc_Matches_computed_distances[] = "Distance evaluations the walk paid for.";

static PyObject* Matches_get_computed_distances(Matches* self, void* closure) {
    return PyLong_FromSize_t(self->stats.computed_distances);
}

static char const doc_Matches_restarts[] = "Times the walk fell back to a fresh entry point.";

static PyObject* Matches_get_restarts(Matches* self, void* closure) { return PyLong_FromSize_t(self->stats.restarts); }

static Py_ssize_t Matches_length(Matches* self) { return (Py_ssize_t)self->queries; }

static PyObject* Matches_item(Matches* self, Py_ssize_t query_index) {
    // `PySequence_GetItem` normalizes a negative index before reaching here.
    if (query_index < 0 || (size_t)query_index >= self->queries) {
        PyErr_SetString(PyExc_IndexError, "Query index out of range");
        return NULL;
    }
    size_t const* const counts = (size_t const*)PyBytes_AS_STRING(self->counts_arena);
    usearch_key_t const* const keys = (usearch_key_t const*)PyBytes_AS_STRING(self->keys_arena);
    usearch_distance_t const* const distances = (usearch_distance_t const*)PyBytes_AS_STRING(self->distances_arena);

    // A count is the engine's word, and the sibling `multiply` verb documents counts that
    // exceed the column width outright - so it bounds a read here rather than defining one.
    size_t const found = counts[query_index] < self->wanted ? counts[query_index] : self->wanted;
    size_t const offset = (size_t)query_index * self->wanted;
    PyObject* const row = PyTuple_New((Py_ssize_t)found);
    if (!row) return NULL;
    for (size_t match_index = 0; match_index != found; ++match_index) {
        PyObject* const match = usearch_match_to_object(keys[offset + match_index], distances[offset + match_index]);
        if (!match) {
            Py_DECREF(row);
            return NULL;
        }
        PyTuple_SET_ITEM(row, (Py_ssize_t)match_index, match);
    }
    return row;
}

static PyObject* Matches_repr(Matches* self) {
    return PyUnicode_FromFormat("Matches(queries=%zu, wanted=%zu, computed_distances=%zu)", self->queries, self->wanted,
                                self->stats.computed_distances);
}

static PyGetSetDef Matches_getset[] = {
    {"keys", (getter)Matches_get_keys, NULL, doc_Matches_keys, NULL},                                           //
    {"distances", (getter)Matches_get_distances, NULL, doc_Matches_distances, NULL},                            //
    {"counts", (getter)Matches_get_counts, NULL, doc_Matches_counts, NULL},                                     //
    {"visited_members", (getter)Matches_get_visited_members, NULL, doc_Matches_visited_members, NULL},          //
    {"computed_distances", (getter)Matches_get_computed_distances, NULL, doc_Matches_computed_distances, NULL}, //
    {"restarts", (getter)Matches_get_restarts, NULL, doc_Matches_restarts, NULL},                               //
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
    .tp_new = NULL, //! Returned by `search`, never constructed by name.
    .tp_dealloc = (destructor)Matches_dealloc,
    .tp_repr = (reprfunc)Matches_repr,
    .tp_as_sequence = &Matches_as_sequence,
    .tp_getset = Matches_getset,
};

#pragma endregion Matches

#pragma region Paths

int usearch_path_from_object(PyObject* object, usearch_string_t* path, PyObject** decoded) {
    PyObject* bytes_object = NULL;
    if (!PyUnicode_FSConverter(object, &bytes_object)) return -1;
    path->data = PyBytes_AS_STRING(bytes_object);
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

PyObject* api_version(PyObject* self, PyObject* unused) { return PyUnicode_FromString(usearch_version()); }

PyObject* api_capabilities_compiled(PyObject* self, PyObject* unused) {
    return PyUnicode_FromString(usearch_capabilities_compiled());
}

PyObject* api_capabilities_available(PyObject* self, PyObject* unused) {
    return PyUnicode_FromString(usearch_capabilities_available());
}

#pragma endregion Library Probes
