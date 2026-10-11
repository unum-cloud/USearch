/**
 *  @file python/types.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief The vocabulary every modality shares - the CPython twin of `usearch/types.h`.
 *
 *  Keys, distances, failures, enums, calls and telemetry live here rather than inside any one
 *  modality, the same split `rust/types.rs` takes: a `Key` produced by a dense search is the very
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

/** The class each failing status raises, indexed by status; success and deferred never raise. */
static PyObject* status_error_types[usearch_status_invalid_argument_k + 1];

/** One failing status, the dotted name of the class it raises, the built-in it also subclasses, and its doc. */
static struct {
    usearch_status_t status;
    char const* name;

    /** NULL when no built-in names the fault. */
    PyObject** builtin;
    char const* doc;
} const status_errors[] = {
    {usearch_status_out_of_memory_k, "usearch.OutOfMemoryError", &PyExc_MemoryError,
     "An allocation or reservation could not be satisfied."},
    {usearch_status_shape_mismatch_k, "usearch.ShapeMismatchError", &PyExc_ValueError,
     "Dimensionality or row counts disagree."},
    {usearch_status_unsupported_layout_k, "usearch.UnsupportedLayoutError", &PyExc_ValueError,
     "A stride or packing the engine cannot traverse."},
    {usearch_status_unknown_kind_k, "usearch.UnknownKindError", &PyExc_ValueError,
     "A scalar, metric, or residency this build does not know."},
    {usearch_status_read_only_k, "usearch.ReadOnlyError", NULL,
     "A mutating verb on a handle that serves its checkpoint in place."},
    {usearch_status_io_k, "usearch.InputOutputError", &PyExc_OSError,
     "A file could not be opened, read, written, or parsed."},
    {usearch_status_device_k, "usearch.DeviceError", NULL,
     "A device could not be selected, or its memory could not be claimed."},
    {usearch_status_malformed_k, "usearch.MalformedError", &PyExc_ValueError,
     "A representation could not be decoded - a key, a buffer, a checkpoint."},
    {usearch_status_internal_k, "usearch.InternalError", NULL,
     "The library reported neither success nor a failure - a defect."},
    {usearch_status_overflow_k, "usearch.CapacityOverflowError", &PyExc_OverflowError,
     "A hard design ceiling was reached; a smaller request cannot help."},
    {usearch_status_uninitialized_k, "usearch.UninitializedError", NULL,
     "The handle is not ready for the verb - unfrozen, consumed by a merge, or closed."},
    {usearch_status_config_mismatch_k, "usearch.ConfigMismatchError", &PyExc_ValueError,
     "The request contradicts an already-frozen configuration."},
    {usearch_status_contended_k, "usearch.ContendedError", NULL,
     "An executor's deferred table, an arena, or a walk is already serving another call."},
    {usearch_status_cancelled_k, "usearch.CancelledError", NULL,
     "A progress handler stopped the call, which published nothing."},
    {usearch_status_unsupported_k, "usearch.UnsupportedError", &PyExc_NotImplementedError,
     "This residency does not implement the verb; pick another residency."},
    {usearch_status_invalid_argument_k, "usearch.InvalidArgumentError", &PyExc_ValueError,
     "An argument breaks the verb's contract."},
};

int usearch_py_raise(usearch_status_t status, char const* format, ...) {
    // A status this specific supersedes whatever the caller was carrying.
    PyErr_Clear();
    PyObject* type = (size_t)status < usearch_py_table_size_(status_error_types) ? status_error_types[status] : NULL;
    if (!type) {
        status = usearch_status_internal_k;
        type = status_error_types[status];
    }
    va_list arguments;
    va_start(arguments, format);
    PyObject* const message = PyUnicode_FromFormatV(format ? format : "", arguments);
    va_end(arguments);
    if (!message) return -1;

    PyObject* const instance = PyObject_CallOneArg(type, message);
    Py_DECREF(message);
    if (!instance) return -1;
    PyObject* const member = usearch_py_enum_to_object(&usearch_py_status, (int)status);
    int const attached = member ? PyObject_SetAttrString(instance, "status", member) : -1;
    Py_XDECREF(member);
    if (attached != 0) {
        Py_DECREF(instance);
        return -1;
    }
    PyErr_SetRaisedException(instance);
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

static char const doc_Error[] =                                                                //
    "Error(message)\n"                                                                         //
    "\n"                                                                                       //
    "The base of every failure the library reports. Each status shared with the C API\n"       //
    "raises a subclass of its own, such as `ShapeMismatchError`, so a caller catches one\n"    //
    "fault by its type or every fault here. A subclass also derives the built-in that names\n" //
    "the same fault, so `except ValueError` still catches a `ShapeMismatchError`. A failure\n" //
    "caught before the boundary raises the class the engine would have raised for it.\n"       //
    "\n"                                                                                       //
    "Attributes:\n"                                                                            //
    "  status (Status): The status the raised subclass mirrors, such as\n"                     //
    "    `Status.SHAPE_MISMATCH`.\n"                                                           //
    "  args (tuple): The detail the status cannot carry, as on any exception.";

int usearch_py_register_error_types(PyObject* module) {
    ErrorType = PyErr_NewExceptionWithDoc("usearch.Error", (char*)doc_Error, PyExc_RuntimeError, NULL);
    if (!ErrorType || PyModule_AddObjectRef(module, "Error", ErrorType) < 0) return -1;
    for (size_t row = 0; row != usearch_py_table_size_(status_errors); ++row) {
        PyObject* const bases = status_errors[row].builtin ? PyTuple_Pack(2, ErrorType, *status_errors[row].builtin)
                                                           : Py_NewRef(ErrorType);
        if (!bases) return -1;
        PyObject* const type = PyErr_NewExceptionWithDoc(status_errors[row].name, (char*)status_errors[row].doc, bases,
                                                         NULL);
        Py_DECREF(bases);
        if (!type) return -1;
        status_error_types[status_errors[row].status] = type;
        if (PyModule_AddObjectRef(module, status_errors[row].name + sizeof("usearch.") - 1, type) < 0) return -1;
    }
    return 0;
}

#pragma endregion Failures

#pragma region Enums

static char const* spell_status(int value) { return usearch_status_name((usearch_status_t)value); }
static char const* spell_mood(int value) { return usearch_mood_name((usearch_mood_t)value); }
static char const* spell_dense_metric(int value) { return usearch_dense_metric_name((usearch_dense_metric_t)value); }
static char const* spell_scalar(int value) { return usearch_scalar_name((usearch_scalar_t)value); }
static char const* spell_residency(int value) { return usearch_residency_name((usearch_residency_t)value); }
static char const* spell_traversal(int value) { return usearch_traversal_name((usearch_traversal_t)value); }

usearch_py_enum_t usearch_py_status = {"Status", usearch_status_success_k, usearch_status_invalid_argument_k,
                                       spell_status, NULL};
usearch_py_enum_t usearch_py_mood = {"Mood", usearch_mood_parked_k, usearch_mood_spinning_k, spell_mood, NULL};
usearch_py_enum_t usearch_py_dense_metric = {"DenseMetric", usearch_dense_metric_inner_k,
                                             usearch_dense_metric_jaccard_k, spell_dense_metric, NULL};
usearch_py_enum_t usearch_py_scalar = {"Scalar", usearch_scalar_f64_k, usearch_scalar_u1x8_k, spell_scalar, NULL};
usearch_py_enum_t usearch_py_residency = {"Residency", usearch_residency_ram_k, usearch_residency_nvme_k,
                                          spell_residency, NULL};
usearch_py_enum_t usearch_py_traversal = {"Traversal", usearch_traversal_walk_k, usearch_traversal_scan_k,
                                          spell_traversal, NULL};

/** The C spelling of @p value, or NULL when C names no member by it. */
static char const* enum_spelling(usearch_py_enum_t const* mirrored, long value) {
    if (value < mirrored->first || value > mirrored->last) return NULL;
    char const* const spelled = mirrored->spell((int)value);
    if (!spelled || strcmp(spelled, "unknown") == 0 || strcmp(spelled, "unrecognized") == 0) return NULL;
    return spelled;
}

/** Builds the class of @p mirrored by calling @p int_enum with @p keywords, members in value order. */
static int enum_register(PyObject* module, PyObject* int_enum, PyObject* keywords, usearch_py_enum_t* mirrored) {
    PyObject* members = PyDict_New();
    for (long value = mirrored->first; members && value <= mirrored->last; ++value) {
        char const* const spelled = enum_spelling(mirrored, value);
        if (!spelled) continue;
        char member_name[32];
        size_t length = 0;
        for (; spelled[length] && length + 1 != sizeof(member_name); ++length) {
            char const letter = spelled[length];
            member_name[length] = letter == '-'                    ? '_'
                                  : letter >= 'a' && letter <= 'z' ? (char)(letter - 'a' + 'A')
                                                                   : letter;
        }
        member_name[length] = '\0';
        if (usearch_py_dict_steal_item(members, member_name, PyLong_FromLong(value)) != 0) Py_CLEAR(members);
    }
    PyObject* const name = members ? PyUnicode_FromString(mirrored->name) : NULL;
    PyObject* const arguments = name ? PyTuple_Pack(2, name, members) : NULL;
    mirrored->type = arguments ? PyObject_Call(int_enum, arguments, keywords) : NULL;
    Py_XDECREF(arguments);
    Py_XDECREF(name);
    Py_XDECREF(members);
    return mirrored->type ? PyModule_AddObjectRef(module, mirrored->name, mirrored->type) : -1;
}

int usearch_py_register_enums(PyObject* module) {
    usearch_py_enum_t* const mirrored[] = {&usearch_py_status, &usearch_py_mood,      &usearch_py_dense_metric,
                                           &usearch_py_scalar, &usearch_py_residency, &usearch_py_traversal};
    PyObject* const enum_module = PyImport_ImportModule("enum");
    PyObject* const int_enum = enum_module ? PyObject_GetAttrString(enum_module, "IntEnum") : NULL;
    PyObject* keywords = int_enum ? PyDict_New() : NULL;
    if (keywords && usearch_py_dict_steal_item(keywords, "module", PyUnicode_FromString("usearch")) != 0)
        Py_CLEAR(keywords);
    int registered = keywords ? 0 : -1;
    for (size_t row = 0; registered == 0 && row != usearch_py_table_size_(mirrored); ++row)
        registered = enum_register(module, int_enum, keywords, mirrored[row]);
    Py_XDECREF(keywords);
    Py_XDECREF(int_enum);
    Py_XDECREF(enum_module);
    return registered;
}

int usearch_py_enum_from_object(PyObject* object, char const* field, usearch_py_enum_t const* mirrored, int* value) {
    if (!PyLong_Check(object) || PyBool_Check(object)) {
        PyErr_Format(PyExc_TypeError, "the '%s' option takes a usearch.%s, got %S", field, mirrored->name,
                     Py_TYPE(object));
        return -1;
    }
    int overflow = 0;
    long const read = PyLong_AsLongAndOverflow(object, &overflow);
    if (read == -1 && PyErr_Occurred()) return -1;
    if (!overflow && enum_spelling(mirrored, read)) {
        *value = (int)read;
        return 0;
    }
    return usearch_py_raise(usearch_status_unknown_kind_k, "the '%s' option takes a usearch.%s, and %R is none", field,
                            mirrored->name, object);
}

PyObject* usearch_py_enum_to_object(usearch_py_enum_t const* mirrored, int value) {
    PyObject* const integer = PyLong_FromLong(value);
    PyObject* const member = integer ? PyObject_CallOneArg(mirrored->type, integer) : NULL;
    Py_XDECREF(integer);
    return member;
}

#pragma endregion Enums

#pragma region Scalar Names

/**
 *  @brief Calls the method @p name of @p self with up to two positional arguments, NULL ones left out.
 *  @return A new reference, or NULL with an exception set.
 */
static PyObject* method_call(PyObject* self, char const* name, PyObject* first, PyObject* second) {
    PyObject* const method = PyUnicode_InternFromString(name);
    if (!method) return NULL;
    PyObject* const arguments[3] = {self, first, second};
    size_t const count = 1 + (first != NULL) + (second != NULL);
    PyObject* const result = PyObject_VectorcallMethod(method, arguments, count, NULL);
    Py_DECREF(method);
    return result;
}

/** One scalar kind, the spellings the ecosystem gives it, and how it packs in memory. */
typedef struct scalar_naming_t {

    /** The kind this row describes; @c usearch_scalar_name spells it. */
    usearch_scalar_t kind;

    /** The NumPy array-interface spelling, or NULL when none exists. */
    char const* numpy_typestr;

    /** The PEP 3118 format string, or NULL when none exists. */
    char const* buffer_typestr;

    /** Logical scalars per storage unit; one for byte-wide kinds. */
    unsigned char values_packed;

    /** Bytes one storage unit occupies. */
    unsigned char unit_bytes;
} scalar_naming_t;

/** Ordered by shrinking dynamic range - floats widest exponent first, then signed
 *  integers descending, then unsigned - the order every dtype-indexed list in the
 *  project takes. */
static scalar_naming_t const scalar_namings[] = {
    {usearch_scalar_f64_k, "<f8", "d", 1, 8},   {usearch_scalar_f32_k, "<f4", "f", 1, 4},
    {usearch_scalar_bf16_k, "<V2", NULL, 1, 2}, {usearch_scalar_f16_k, "<f2", "e", 1, 2},
    {usearch_scalar_e5m2_k, "|V1", NULL, 1, 1}, {usearch_scalar_e4m3_k, "|V1", NULL, 1, 1},
    {usearch_scalar_e3m2_k, "|V1", NULL, 1, 1}, {usearch_scalar_e2m3_k, "|V1", NULL, 1, 1},
    {usearch_scalar_i32_k, "<i4", "i", 1, 4},   {usearch_scalar_i8_k, "|i1", "b", 1, 1},
    {usearch_scalar_i4x2_k, "|V1", NULL, 2, 1}, {usearch_scalar_u8_k, "|u1", "B", 1, 1},
    {usearch_scalar_u4x2_k, "|V1", NULL, 2, 1}, {usearch_scalar_u1x8_k, "|V1", "?", 8, 1},
};

/** The dtype names an `__array_interface__` exporter reports, resolved after the table. */
static struct {
    char const* name;
    usearch_scalar_t kind;
} const scalar_aliases[] = {
    {"float64", usearch_scalar_f64_k}, {"float32", usearch_scalar_f32_k}, {"bfloat16", usearch_scalar_bf16_k},
    {"float16", usearch_scalar_f16_k}, {"int32", usearch_scalar_i32_k},   {"int8", usearch_scalar_i8_k},
    {"uint8", usearch_scalar_u8_k},    {"bool", usearch_scalar_u1x8_k},
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

int usearch_py_scalar_from_object(PyObject* object, char const* field, usearch_scalar_t* kind) {
    *kind = usearch_scalar_unknown_k;
    if (!object || object == Py_None) return 0;
    int value = usearch_scalar_unknown_k;
    if (usearch_py_enum_from_object(object, field, &usearch_py_scalar, &value) != 0) return -1;
    *kind = (usearch_scalar_t)value;
    return 0;
}

#pragma endregion Scalar Names

#pragma region Buffer Acquisition

/**
 *  @brief Clears an exporter's refusal to export, the one failure a fallback may replace.
 *  @return 1 when the raised exception was a refusal and is gone, 0 when it is a real failure and stays.
 *
 *  NumPy refuses a dtype the protocol cannot spell with a @c ValueError, which is what the
 *  @c __array_interface__ fallback exists for; a @c KeyboardInterrupt or @c MemoryError must survive.
 */
static int buffer_refusal_cleared(void) {
    if (!PyErr_ExceptionMatches(PyExc_TypeError) && !PyErr_ExceptionMatches(PyExc_BufferError) &&
        !PyErr_ExceptionMatches(PyExc_ValueError))
        return 0;
    PyErr_Clear();
    return 1;
}

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
        usearch_py_raise(usearch_status_shape_mismatch_k, "expected a vector or a matrix, got rank %zd", rank);
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
        PyErr_Clear();
        Py_DECREF(dtype_attribute);
    }
    else { PyErr_Clear(); }

    if (kind == usearch_scalar_unknown_k) {
        PyObject* const typestr_object = PyDict_GetItemString(interface, "typestr");
        if (typestr_object) {
            Py_ssize_t typestr_length = 0;
            char const* const typestr = PyUnicode_AsUTF8AndSize(typestr_object, &typestr_length);
            if (typestr) kind = usearch_py_scalar_from_spelling(typestr, typestr_length);
            PyErr_Clear();
        }
    }
    scalar_naming_t const* const naming = scalar_naming(kind);
    if (!naming) {
        Py_DECREF(interface);
        usearch_py_raise(usearch_status_unknown_kind_k, "__array_interface__ names no scalar kind this build knows");
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
    // the memory alive - a temporary exporter would otherwise die while the call is deferred.
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
    if (!buffer_refusal_cleared()) return -1;
    if (buffer_via_array_interface(object, backing)) return 0;
    if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError, "expected an object supporting the buffer protocol or __array_interface__");
    return -1;
}

void usearch_py_buffer_release(usearch_py_buffer_backing_t* backing) {
    PyBuffer_Release(&backing->buffer);
    Py_CLEAR(backing->synthesized_owner);
}

int usearch_py_output_from_object(PyObject* object, char const* field, Py_ssize_t bytes,
                                  usearch_py_buffer_backing_t* backing, void** data) {
    backing->synthesized_owner = NULL;
    if (PyObject_GetBuffer(object, &backing->buffer, PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) != 0)
        // The exporter's own diagnosis is more precise than any guess we could make, so it rides
        // along as the cause rather than being replaced.
        return usearch_py_raise_from_cause(PyExc_ValueError, "the '%s' buffer must be contiguous and writable", field);
    if (backing->buffer.len != bytes) {
        Py_ssize_t const held = backing->buffer.len;
        usearch_py_buffer_release(backing);
        return usearch_py_raise(usearch_status_shape_mismatch_k, "the '%s' buffer must hold exactly %zd bytes, not %zd",
                                field, bytes, held);
    }
    *data = backing->buffer.buf;
    return 0;
}

int usearch_py_matrix_from_object(PyObject* object, usearch_scalar_t scalar_hint, usearch_matrix_view_t* view,
                                  usearch_py_buffer_backing_t* backing) {
    if (buffer_from_object(object, backing) != 0) return -1;
    Py_buffer const* const buffer = &backing->buffer;

    if (buffer->ndim < 1 || buffer->ndim > usearch_py_max_rank_k) {
        int const rank = buffer->ndim;
        usearch_py_buffer_release(backing);
        return usearch_py_raise(usearch_status_shape_mismatch_k, "expected a vector or a matrix, got rank %d", rank);
    }

    usearch_scalar_t kind = scalar_hint;
    if (kind == usearch_scalar_unknown_k && buffer->format)
        kind = usearch_py_scalar_from_spelling(buffer->format, (Py_ssize_t)strlen(buffer->format));
    scalar_naming_t const* const naming = scalar_naming(kind);
    if (!naming) {
        PyObject* const format = PyUnicode_FromString(buffer->format ? buffer->format : "?");
        usearch_py_buffer_release(backing);
        if (!format) return -1;
        usearch_py_raise(usearch_status_unknown_kind_k, "the buffer's %R items name no scalar kind; pass `dtype=`",
                         format);
        Py_DECREF(format);
        return -1;
    }
    // A named kind reinterprets the bytes, so it must be exactly as wide as the items the buffer
    // holds; anything else would read a float32 row as twice as many halves.
    if (scalar_hint != usearch_scalar_unknown_k && buffer->itemsize != (Py_ssize_t)naming->unit_bytes) {
        Py_ssize_t const itemsize = buffer->itemsize;
        usearch_py_buffer_release(backing);
        return usearch_py_raise(usearch_status_unknown_kind_k,
                                "the buffer's items are %zd bytes wide, but the named kind stores %d", itemsize,
                                (int)naming->unit_bytes);
    }

    // Only the row stride may vary: a strided column would force a gather the kernels have
    // no form for, and the C surface has no argument to describe one.
    int const last_axis = buffer->ndim - 1;
    if (buffer->strides && buffer->strides[last_axis] != buffer->itemsize) {
        usearch_py_buffer_release(backing);
        return usearch_py_raise(usearch_status_unsupported_layout_k,
                                "expected a matrix whose rows are contiguous; only the row stride may vary");
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
            usearch_py_buffer_release(backing);
            return usearch_py_raise(usearch_status_unsupported_layout_k, "matrix rows must ascend in memory");
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

static char const doc_Key[] =                                                                 //
    "Key(value=0)\n"                                                                          //
    "\n"                                                                                      //
    "A 128-bit entry identity, rendered as a lowercase dashed UUID.\n"                        //
    "Keys need not be unique: several entries may share one, and `count` reports how\n"       //
    "many. Integer identities sit in the low eight bytes. A key equals and hashes as\n"       //
    "the integer or `uuid.UUID` it reads as, and pickles as its sixteen bytes.\n"             //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  value (int | str | bytes | uuid.UUID | Key, optional): The identity. Defaults to 0.\n" //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> str(usearch.Key(42))\n"                                                            //
    "  '00000000-0000-0000-0000-00000000002a'\n"                                              //
    "  >>> int(usearch.Key('00000000-0000-0000-0000-00000000002a'))\n"                        //
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
        else if (!buffer_refusal_cleared()) { return -1; }
    }
    // `PyIndex_Check` rather than `PyLong_Check`, so a NumPy scalar - which is not a Python
    // integer but knows how to become one - is read the same way a literal is.
    if (PyIndex_Check(object)) {
        PyObject* const integer = PyNumber_Index(object);
        PyObject* const length = integer ? PyLong_FromLong(16) : NULL;
        PyObject* const order = length ? PyUnicode_FromString("big") : NULL;
        // Big-endian and unsigned, so integer identities sit in the low eight bytes and sort
        // the way the dashed rendering reads. `int.to_bytes` raises on negatives and overflow
        // for us, and this is never the batch path - those arrive as buffers.
        PyObject* const raw = order ? method_call(integer, "to_bytes", length, order) : NULL;
        Py_XDECREF(order);
        Py_XDECREF(length);
        Py_XDECREF(integer);
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
    if (!PyErr_ExceptionMatches(PyExc_AttributeError)) return -1;
    PyErr_Clear();
    PyErr_Format(PyExc_TypeError, "cannot read a key from %S", Py_TYPE(object));
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
    PyObject* const order = raw ? PyUnicode_FromString("big") : NULL;
    PyObject* const value = order ? method_call((PyObject*)&PyLong_Type, "from_bytes", raw, order) : NULL;
    Py_XDECREF(order);
    Py_XDECREF(raw);
    return value;
}

/**
 *  @brief Hashes as the integer this key compares equal to.
 *
 *  `Key(256) == 256` is true, so the two must hash alike or a dict keyed by one cannot be
 *  read by the other; a `uuid.UUID` hashes as its integer too, so it agrees for free.
 */
static Py_hash_t Key_hash(Key* self) {
    PyObject* const integer = Key_int(self);
    if (!integer) return -1;
    Py_hash_t const hashed = PyObject_Hash(integer);
    Py_DECREF(integer);
    return hashed;
}

/** Whether @p object hashes as the identity it reads as - a @c Key, an @c int or a @c uuid.UUID - rather than merely
 *  parsing into one, as text and bytes do while hashing as themselves. */
static int key_comparable(PyObject* object) {
    return PyObject_TypeCheck(object, &KeyType) || PyLong_Check(object) || PyObject_HasAttrString(object, "int");
}

static PyObject* Key_richcompare(PyObject* first, PyObject* second, int operation) {
    if (!key_comparable(first) || !key_comparable(second)) Py_RETURN_NOTIMPLEMENTED;
    usearch_key_t left, right;
    if (usearch_py_key_from_object(first, &left) != 0 || usearch_py_key_from_object(second, &right) != 0) {
        // An integer no key can hold is merely unequal; a `MemoryError` raised while reading one must not be
        // laundered into a silent `False`.
        if (!PyErr_ExceptionMatches(PyExc_TypeError) && !PyErr_ExceptionMatches(PyExc_OverflowError) &&
            !PyErr_ExceptionMatches(ErrorType))
            return NULL;
        PyErr_Clear();
        Py_RETURN_NOTIMPLEMENTED;
    }
    int const ordering = memcmp(left.bytes, right.bytes, 16);
    Py_RETURN_RICHCOMPARE(ordering, 0, operation);
}

/** Pickles as the type and its sixteen bytes, which `Key()` reads back. */
static PyObject* Key_reduce(PyObject* self, PyObject* unused) {
    PyObject* const raw = PyBytes_FromStringAndSize((char const*)((Key*)self)->key.bytes, 16);
    PyObject* const arguments = raw ? PyTuple_Pack(1, raw) : NULL;
    PyObject* const reduced = arguments ? PyTuple_Pack(2, (PyObject*)Py_TYPE(self), arguments) : NULL;
    Py_XDECREF(arguments);
    Py_XDECREF(raw);
    return reduced;
}

static PyObject* Key_get_bytes(Key* self, void* closure) {
    return PyBytes_FromStringAndSize((char const*)self->key.bytes, 16);
}

static char const doc_Key_bytes[] = "The sixteen identity bytes, most significant first.";

static PyGetSetDef Key_getset[] = {
    {"bytes", (getter)Key_get_bytes, NULL, doc_Key_bytes, NULL}, //
    {NULL, NULL, NULL, NULL, NULL},
};

static PyMethodDef Key_methods[] = {
    {"__reduce__", (PyCFunction)Key_reduce, METH_NOARGS, NULL}, //
    {NULL, NULL, 0, NULL},
};

static PyNumberMethods Key_as_number = {.nb_int = (unaryfunc)Key_int};

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
    .tp_methods = Key_methods,
    .tp_getset = Key_getset,
};

#pragma endregion Keys

#pragma region Key Batches

void usearch_py_keys_release(usearch_py_keys_t* batch) {
    PyMem_Free(batch->owned);
    usearch_py_buffer_release(&batch->backing);
    memset(batch, 0, sizeof(*batch));
}

/** Whether @p buffer is a vector of native integers, which @c keys_from_integers gathers without a
 *  Python call. */
static int keys_are_integers(Py_buffer const* buffer) {
    char const* format = buffer->format ? buffer->format : "B";
    if (*format == '@' || *format == '=') ++format;
    int const width = buffer->itemsize == 1 || buffer->itemsize == 2 || buffer->itemsize == 4 || buffer->itemsize == 8;
    return buffer->ndim == 1 && width && format[0] && !format[1] && strchr("bBhHiIlLqQnN", format[0]);
}

/** Gathers the integers of @p buffer into @p keys as @c usearch_py_key_from_object reads each; 0,
 *  or -1 with an @c OverflowError on a negative one, as `int.to_bytes` raises. */
static int keys_from_integers(Py_buffer const* buffer, usearch_key_t* keys) {
    char const* const format = buffer->format ? buffer->format : "B";
    int const is_signed = strchr("bhilqn", format[strlen(format) - 1]) != NULL;
    Py_ssize_t const stride = buffer->strides ? buffer->strides[0] : buffer->itemsize;
    unsigned const sign_shift = (unsigned)buffer->itemsize * 8 - 1;
    for (Py_ssize_t position = 0; position != buffer->shape[0]; ++position) {
        char const* const item = (char const*)buffer->buf + position * stride;
        uint8_t narrow8;
        uint16_t narrow16;
        uint32_t narrow32;
        uint64_t value;
        switch (buffer->itemsize) {
        case 1:
            memcpy(&narrow8, item, 1);
            value = narrow8;
            break;
        case 2:
            memcpy(&narrow16, item, 2);
            value = narrow16;
            break;
        case 4:
            memcpy(&narrow32, item, 4);
            value = narrow32;
            break;
        default: memcpy(&value, item, 8); break;
        }
        if (is_signed && (value >> sign_shift) & 1) {
            PyErr_SetString(PyExc_OverflowError, "can't convert negative int to unsigned");
            return -1;
        }
        memset(keys[position].bytes, 0, 8);
        for (size_t byte_index = 0; byte_index != 8; ++byte_index)
            keys[position].bytes[15 - byte_index] = (uint8_t)(value >> (byte_index * 8));
    }
    return 0;
}

int usearch_py_keys_from_object(PyObject* object, size_t expected, usearch_py_keys_t* batch) {
    memset(batch, 0, sizeof(*batch));

    // An identity matrix is borrowed outright and an integer vector gathered in C; everything else
    // walks the sequence below.
    if (PyObject_CheckBuffer(object)) {
        if (PyObject_GetBuffer(object, &batch->backing.buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
            if (!buffer_refusal_cleared()) return -1;
        }
        else {
            Py_buffer* const buffer = &batch->backing.buffer;
            int const is_identity = buffer->ndim == 2 && buffer->shape[1] == (Py_ssize_t)sizeof(usearch_key_t) &&
                                    buffer->itemsize == 1 && PyBuffer_IsContiguous(buffer, 'C');
            int const is_integer = !is_identity && keys_are_integers(buffer);
            if ((is_identity || is_integer) && (size_t)buffer->shape[0] != expected) {
                usearch_py_keys_release(batch);
                return usearch_py_raise(usearch_status_shape_mismatch_k,
                                        "the key count does not match the vector row count");
            }
            if (is_identity) {
                batch->keys = (usearch_key_t*)buffer->buf;
                return 0;
            }
            if (is_integer) {
                batch->owned = (usearch_key_t*)PyMem_Malloc(expected ? expected * sizeof(usearch_key_t) : 1);
                if (!batch->owned) PyErr_NoMemory();
                if (!batch->owned || keys_from_integers(buffer, batch->owned) != 0) {
                    usearch_py_keys_release(batch);
                    return -1;
                }
                usearch_py_buffer_release(&batch->backing);
                batch->keys = batch->owned;
                return 0;
            }
            usearch_py_keys_release(batch);
        }
    }

    PyObject* const sequence = PySequence_Fast(object, "expected a sequence of keys or a buffer of identities");
    if (!sequence) return -1;
    if ((size_t)PySequence_Fast_GET_SIZE(sequence) != expected) {
        Py_DECREF(sequence);
        return usearch_py_raise(usearch_status_shape_mismatch_k, "the key count does not match the vector row count");
    }

    batch->owned = (usearch_key_t*)PyMem_Malloc(expected ? expected * sizeof(usearch_key_t) : 1);
    if (!batch->owned) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return -1;
    }
    for (size_t entry_index = 0; entry_index != expected; ++entry_index) {
        if (usearch_py_key_from_object(PySequence_Fast_GET_ITEM(sequence, entry_index), &batch->owned[entry_index]) !=
            0) {
            Py_DECREF(sequence);
            usearch_py_keys_release(batch);
            return -1;
        }
    }
    Py_DECREF(sequence);
    batch->keys = batch->owned;
    return 0;
}

#pragma endregion Key Batches

#pragma region Matches

static char const doc_Match[] =                                             //
    "Match(key, distance)\n"                                                //
    "\n"                                                                    //
    "One scored entry: the identity found and its distance, unpacking as\n" //
    "`key, distance = match`.\n"                                            //
    "\n"                                                                    //
    "Attributes:\n"                                                         //
    "  key (Key): The identity of the matched entry.\n"                     //
    "  distance (float): The distance to the query, in the index's own metric.";

static PyStructSequence_Field Match_fields[] = {
    {"key", "The identity of the matched entry."},
    {"distance", "The distance to the query, in the index's own metric."},
    {NULL, NULL},
};

PyStructSequence_Desc usearch_py_match_desc = {"usearch.Match", doc_Match, Match_fields, 2};

PyTypeObject MatchType;

PyObject* usearch_py_match_to_object(usearch_key_t key, usearch_distance_t distance) {
    PyObject* const match = PyStructSequence_New(&MatchType);
    PyObject* const key_object = match ? usearch_py_key_to_object(key) : NULL;
    PyObject* const distance_object = key_object ? PyFloat_FromDouble(distance) : NULL;
    if (!distance_object) {
        Py_XDECREF(key_object);
        Py_XDECREF(match);
        return NULL;
    }
    PyStructSequence_SET_ITEM(match, 0, key_object);
    PyStructSequence_SET_ITEM(match, 1, distance_object);
    return match;
}

PyObject* usearch_py_struct_sequence_steal(PyTypeObject* type, PyObject** fields, Py_ssize_t count) {
    PyObject* result = NULL;
    Py_ssize_t filled = 0;
    while (filled != count && fields[filled]) ++filled;
    if (filled == count) result = PyStructSequence_New(type);
    if (!result) {
        for (Py_ssize_t field = 0; field != count; ++field) Py_XDECREF(fields[field]);
        return NULL;
    }
    for (Py_ssize_t field = 0; field != count; ++field) PyStructSequence_SET_ITEM(result, field, fields[field]);
    return result;
}

#pragma endregion Matches

#pragma region Arguments

int usearch_py_arguments_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                               PyObject* args_names_tuple, Py_ssize_t positional_only, Py_ssize_t positional_max,
                               usearch_py_argument_t const* arguments, size_t count) {
    if (positional_args_count > positional_max) {
        PyErr_Format(PyExc_TypeError, "%s() takes at most %zd positional arguments, received %zd", verb, positional_max,
                     positional_args_count);
        return -1;
    }
    for (size_t position = 0; position != count; ++position) *arguments[position].value = NULL;
    for (Py_ssize_t position = 0; position != positional_args_count; ++position)
        *arguments[position].value = args[position];

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t name_index = 0; name_index != args_names_count; ++name_index) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, name_index);
        size_t position = (size_t)positional_only;
        while (position != count && PyUnicode_CompareWithASCIIString(name, arguments[position].name) != 0) ++position;
        if (position == count) {
            PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument %R", verb, name);
            return -1;
        }
        if (*arguments[position].value) {
            PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument '%s'", verb, arguments[position].name);
            return -1;
        }
        *arguments[position].value = args[positional_args_count + name_index];
    }
    for (Py_ssize_t position = 0; position != positional_only; ++position)
        if (!*arguments[position].value) {
            PyErr_Format(PyExc_TypeError, "%s() missing required argument '%s'", verb, arguments[position].name);
            return -1;
        }
    return 0;
}

int usearch_py_ordinals_from_object(PyObject* object, char const* field, int32_t** ordinals, size_t* count) {
    *ordinals = NULL;
    *count = 0;
    if (!object || object == Py_None) return 0;

    PyObject* const sequence = PySequence_Fast(object, "expected a sequence of integers");
    if (!sequence) {
        usearch_py_raise_from_cause(PyExc_TypeError, "the '%s' option must be a sequence of integers", field);
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
            usearch_py_raise_from_cause(PyExc_TypeError, "the '%s' option must hold integers", field);
            return -1;
        }
        if (ordinal < INT32_MIN || ordinal > INT32_MAX) {
            PyMem_Free(claimed);
            Py_DECREF(sequence);
            return usearch_py_raise(usearch_status_invalid_argument_k,
                                    "the '%s' option holds %ld, outside the 32-bit range", field, ordinal);
        }
        claimed[position] = (int32_t)ordinal;
    }
    Py_DECREF(sequence);
    *ordinals = claimed;
    *count = (size_t)length;
    return 0;
}

/** The encoded objects trailing the views in the single block of @p paths. */
static PyObject** paths_encoded(usearch_py_paths_t const* paths) { return (PyObject**)(paths->views + paths->count); }

int usearch_py_paths_from_object(PyObject* object, usearch_py_paths_t* paths) {
    paths->views = NULL;
    paths->count = 0;
    if (!object || object == Py_None) return 0;

    PyObject* const sequence = PySequence_Fast(object, "expected a sequence of paths");
    if (!sequence) {
        usearch_py_raise_from_cause(PyExc_TypeError, "the 'paths' option must be a sequence of paths");
        return -1;
    }
    Py_ssize_t const length = PySequence_Fast_GET_SIZE(sequence);
    if (length == 0) {
        Py_DECREF(sequence);
        return 0;
    }

    // Both runs are pointer-aligned, so one block carves into views followed by their objects.
    size_t const block_bytes = (size_t)length * (sizeof(usearch_bytes_view_t) + sizeof(PyObject*));
    paths->views = (usearch_bytes_view_t*)PyMem_Malloc(block_bytes);
    if (!paths->views) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return -1;
    }

    PyObject** const encoded = (PyObject**)(paths->views + length);
    for (Py_ssize_t position = 0; position != length; ++position) {
        if (usearch_py_path_from_object(PySequence_Fast_GET_ITEM(sequence, position), &paths->views[position],
                                        &encoded[position]) != 0) {
            while (position-- != 0) Py_DECREF(encoded[position]);
            PyMem_Free(paths->views);
            paths->views = NULL;
            Py_DECREF(sequence);
            return -1;
        }
    }
    Py_DECREF(sequence);
    paths->count = (size_t)length;
    return 0;
}

void usearch_py_paths_release(usearch_py_paths_t* paths) {
    if (!paths->views) return;
    PyObject** const encoded = paths_encoded(paths);
    for (size_t position = 0; position != paths->count; ++position) Py_XDECREF(encoded[position]);
    PyMem_Free(paths->views);
    paths->views = NULL;
    paths->count = 0;
}

int usearch_py_size_from_object(PyObject* object, char const* field, size_t* value) {
    if (!object) return 0;
    Py_ssize_t const signed_value = PyNumber_AsSsize_t(object, PyExc_OverflowError);
    if (signed_value == -1 && PyErr_Occurred()) return -1;
    if (signed_value < 0)
        return usearch_py_raise(usearch_status_invalid_argument_k, "the '%s' option must not be negative", field);
    *value = (size_t)signed_value;
    return 0;
}

#pragma endregion Arguments

#pragma region Calls

/** The @c Deferred whose trailing storage holds @p call. */
static Deferred* deferred_of(void* call) { return (Deferred*)(void*)((char*)call - offsetof(Deferred, call)); }

void* usearch_py_call_open(usearch_py_receiver_t const* receiver, usearch_py_verb_t const* verb, void* stack) {
    void* call = stack;
    if (receiver->completion == usearch_completion_deferred_k) {
        Deferred* const deferred = PyObject_GC_NewVar(Deferred, &DeferredType, (Py_ssize_t)verb->call_bytes);
        if (!deferred) return NULL;
        deferred->verb = verb;
        deferred->origin = Py_NewRef(receiver->origin);
        deferred->answer = NULL;
        deferred->raised = NULL;
        call = deferred->call;
        memset(call, 0, verb->call_bytes);
        PyObject_GC_Track(deferred);
        return call;
    }
    memset(call, 0, verb->call_bytes);
    return call;
}

/** Answers a final @p call: a callback's first exception, else the outcome's failure, else the verb's answer. */
static PyObject* call_answer(usearch_py_verb_t const* verb, void* call) {
    usearch_py_call_t* const header = (usearch_py_call_t*)call;
    if (header->raised) {
        PyErr_SetRaisedException(header->raised);
        header->raised = NULL;
        return NULL;
    }
    return usearch_py_raise_on_outcome(&header->outcome) == 0 ? verb->answer(call) : NULL;
}

/** Lets go of what @p call gathered, keeping whatever exception is set. */
static void call_release(usearch_py_verb_t const* verb, void* call) {
    PyObject* const raised = PyErr_GetRaisedException();
    verb->release(call);
    PyErr_SetRaisedException(raised);
}

PyObject* usearch_py_call_run(usearch_py_receiver_t const* receiver, usearch_py_verb_t const* verb, void* call) {
    Py_BEGIN_ALLOW_THREADS;
    verb->submit(call);
    Py_END_ALLOW_THREADS;
    if (receiver->completion == usearch_completion_deferred_k) return (PyObject*)deferred_of(call);
    PyObject* const answer = call_answer(verb, call);
    call_release(verb, call);
    return answer;
}

PyObject* usearch_py_call_discard(usearch_py_receiver_t const* receiver, usearch_py_verb_t const* verb, void* call) {
    call_release(verb, call);
    if (receiver->completion == usearch_completion_blocking_k) return NULL;
    Deferred* const deferred = deferred_of(call);
    deferred->verb = NULL;
    PyObject* const raised = PyErr_GetRaisedException();
    Py_DECREF(deferred);
    PyErr_SetRaisedException(raised);
    return NULL;
}

void usearch_py_call_release(void* call) {
    usearch_py_call_t* const header = (usearch_py_call_t*)call;
    Py_CLEAR(header->raised);
    if (header->arena) usearch_py_arena_unclaim(header->arena);
    header->arena = NULL;
}

PyObject* usearch_py_call_answer_none(void* call) { Py_RETURN_NONE; }

int usearch_py_callback_from_object(PyObject* object, char const* field, usearch_py_call_t* call,
                                    usearch_py_callback_t* callback) {
    callback->call = call;
    callback->callable = NULL;
    if (!object || object == Py_None) return 0;
    if (!PyCallable_Check(object)) {
        PyErr_Format(PyExc_TypeError, "the '%s' handler must be callable, got %S", field, Py_TYPE(object));
        return -1;
    }
    callback->callable = Py_NewRef(object);
    return 0;
}

void usearch_py_callback_latch(usearch_py_call_t* call) {
    PyObject* const raised = PyErr_GetRaisedException();
    PyObject* expected = NULL;
    // Predicates run concurrently on the library's threads, and without the interpreter lock two
    // of them may fail at once; only the first exception is kept, and the store is the fence.
    if (raised && !atomic_compare_exchange_strong((_Atomic(PyObject*)*)&call->raised, &expected, raised))
        Py_DECREF(raised);
}

int usearch_py_callback_raised(usearch_py_call_t* call) {
    return atomic_load_explicit((_Atomic(PyObject*)*)&call->raised, memory_order_acquire) != NULL;
}

/**
 *  @brief Re-enters Python for one progress report, from whichever library thread advances.
 *
 *  Reports are serialized, but nothing guarantees they land on the calling thread, so the GIL is
 *  taken here rather than assumed. A handler answering @c None carries on, and one that raises
 *  also stops the walk, so the call unwinds at the next boundary instead of running on.
 */
static usearch_continuation_t progress_trampoline(size_t completed, size_t total, void* state) {
    usearch_py_callback_t const* const callback = (usearch_py_callback_t const*)state;
    if (usearch_py_callback_raised(callback->call)) return usearch_cancel_k;
    PyGILState_STATE const gil_state = PyGILState_Ensure();
    usearch_continuation_t continuation = usearch_cancel_k;
    PyObject* const completed_object = PyLong_FromSize_t(completed);
    PyObject* const total_object = completed_object ? PyLong_FromSize_t(total) : NULL;
    PyObject* const arguments[2] = {completed_object, total_object};
    PyObject* const verdict = total_object ? PyObject_Vectorcall(callback->callable, arguments, 2, NULL) : NULL;
    if (verdict) {
        int const truth = verdict == Py_None ? 1 : PyObject_IsTrue(verdict);
        continuation = truth > 0 ? usearch_continue_k : usearch_cancel_k;
        Py_DECREF(verdict);
    }
    Py_XDECREF(total_object);
    Py_XDECREF(completed_object);
    if (PyErr_Occurred()) usearch_py_callback_latch(callback->call);
    PyGILState_Release(gil_state);
    return continuation;
}

usearch_progress_callback_t usearch_py_progress_registered(usearch_py_callback_t const* callback) {
    return callback->callable ? progress_trampoline : NULL;
}

/**
 *  @brief Re-enters Python for one admission decision, from whichever library thread traverses.
 *
 *  A worker's thread state dies with the callback and the caller's would surface the exception
 *  at an unrelated return, so a raising predicate latches its failure into the call for the answer
 *  to raise, and every later decision rejects without re-entering Python.
 */
static usearch_admission_t predicate_trampoline(usearch_key_t key, void* state) {
    usearch_py_callback_t const* const callback = (usearch_py_callback_t const*)state;
    if (usearch_py_callback_raised(callback->call)) return usearch_reject_k;
    PyGILState_STATE const gil_state = PyGILState_Ensure();
    usearch_admission_t admitted = usearch_reject_k;
    PyObject* const key_object = usearch_py_key_to_object(key);
    PyObject* const verdict = key_object ? PyObject_CallOneArg(callback->callable, key_object) : NULL;
    Py_XDECREF(key_object);
    if (verdict) {
        admitted = PyObject_IsTrue(verdict) > 0 ? usearch_admit_k : usearch_reject_k;
        Py_DECREF(verdict);
    }
    if (PyErr_Occurred()) usearch_py_callback_latch(callback->call);
    PyGILState_Release(gil_state);
    return admitted;
}

usearch_predicate_t usearch_py_predicate_registered(usearch_py_callback_t const* callback) {
    return callback->callable ? predicate_trampoline : NULL;
}

#pragma endregion Calls

#pragma region Receivers and Deferred Calls

/** An index bound to the executor its verbs run on; its type alone chooses how they complete. */
typedef struct On {
    PyObject ob_base;

    /** The index the verbs address, held. */
    PyObject* index;

    /** The @c Executor the calls run on, held. */
    PyObject* executor;
} On;

static char const doc_On[] =                                                                   //
    "On[Index]\n"                                                                              //
    "\n"                                                                                       //
    "An index bound to the executor its verbs block on, as `index.on(executor)` answers it.\n" //
    "\n"                                                                                       //
    "Offers the verbs of the index with the same arguments and answers; each call runs on\n"   //
    "the executor after every call submitted to it before. It holds the index and the\n"       //
    "executor, so it may outlive the names they were bound to.\n"                              //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> pool = usearch.Executor.for_ram(threads=2)\n"                                       //
    "  >>> usearch.DenseIndex.init_ram(4).on(pool).describe().size\n"                          //
    "  0";

static char const doc_DeferredOn[] =                                                          //
    "DeferredOn[Index]\n"                                                                     //
    "\n"                                                                                      //
    "An index bound to the executor its verbs submit to, as `index.deferred(executor)`\n"     //
    "answers it.\n"                                                                           //
    "\n"                                                                                      //
    "Offers the verbs of the index with the same arguments, each answering a `Deferred` at\n" //
    "once. Arguments the binding refuses raise at the call, while the engine's refusals,\n"   //
    "`ContendedError` from a full deferred table included, raise at `await` or `wait()`.\n"   //
    "`cursor` and `search_stream` answer iterators whose every step blocks, so it lacks them.";

int usearch_py_receiver_from_object(PyObject* self, usearch_py_receiver_t* receiver) {
    int const deferred = Py_IS_TYPE(self, &DeferredOnType);
    receiver->origin = self;
    receiver->index = self;
    receiver->executor = NULL;
    receiver->completion = deferred ? usearch_completion_deferred_k : usearch_completion_blocking_k;
    if (!deferred && !Py_IS_TYPE(self, &OnType)) return 0;
    receiver->index = ((On*)self)->index;
    return usearch_py_executor_from_object(((On*)self)->executor, &receiver->executor);
}

PyObject* usearch_py_on_new(PyTypeObject* type, PyObject* index, PyObject* executor) {
    usearch_executor_t handle = NULL;
    if (usearch_py_executor_from_object(executor, &handle) != 0) return NULL;
    On* const on = PyObject_New(On, type);
    if (!on) return NULL;
    on->index = Py_NewRef(index);
    on->executor = Py_NewRef(executor);
    return (PyObject*)on;
}

static PyObject* On_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_TypeError, "receivers are built through index.on(executor) or index.deferred(executor)");
    return NULL;
}

static void On_dealloc(On* self) {
    Py_XDECREF(self->executor);
    Py_XDECREF(self->index);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* On_repr(On* self) {
    return PyUnicode_FromFormat("%s(%R, %R)", Py_TYPE(self)->tp_name + sizeof("usearch.") - 1, self->index,
                                self->executor);
}

PyTypeObject OnType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.On",
    .tp_doc = doc_On,
    .tp_basicsize = sizeof(On),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = On_tp_new,
    .tp_dealloc = (destructor)On_dealloc,
    .tp_repr = (reprfunc)On_repr,
    .tp_methods = usearch_py_on_methods,
};

PyTypeObject DeferredOnType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DeferredOn",
    .tp_doc = doc_DeferredOn,
    .tp_basicsize = sizeof(On),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = On_tp_new,
    .tp_dealloc = (destructor)On_dealloc,
    .tp_repr = (reprfunc)On_repr,
    .tp_methods = usearch_py_deferred_on_methods,
};

static char const doc_Deferred[] =                                                             //
    "Deferred[Answer]\n"                                                                       //
    "\n"                                                                                       //
    "A call a `DeferredOn` submitted, which answers once it is final.\n"                       //
    "\n"                                                                                       //
    "`await` suspends the coroutine until the call is final, and `wait()` blocks the thread\n" //
    "instead. Either answers what the blocking verb would have answered, or raises what it\n"  //
    "would have raised, as many times as it is asked; `status` polls without waiting. It\n"    //
    "holds the call and the buffers it reads until the call is final, so dropping it\n"        //
    "waits, and borrows its arena as the blocking verb would.\n"                               //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> import asyncio\n"                                                                   //
    "  >>> from array import array\n"                                                          //
    "  >>> pool = usearch.Executor.for_ram(threads=2)\n"                                       //
    "  >>> index = usearch.DenseIndex.init_ram(4)\n"                                           //
    "  >>> index.reserve(1)\n"                                                                 //
    "  >>> async def insert():\n"                                                              //
    "  ...     call = index.deferred(pool).insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))\n"    //
    "  ...     return await call  # or: call.wait()\n"                                         //
    "  >>> asyncio.run(insert())\n"                                                            //
    "  >>> len(index)\n"                                                                       //
    "  1";

/** The outcome of the call @p deferred holds. */
static usearch_outcome_t* deferred_outcome(Deferred* deferred) {
    return &((usearch_py_call_t*)(void*)deferred->call)->outcome;
}

/** The status of the call @p deferred holds, as the library finishes it elsewhere. */
static usearch_status_t deferred_status(Deferred* deferred) {
    return usearch_outcome_status(deferred_outcome(deferred));
}

/** Blocks with the thread state detached until the call @p deferred holds is final. */
static void deferred_block(Deferred* deferred) {
    if (deferred_status(deferred) != usearch_status_deferred_k) return;
    usearch_outcome_t* const outcome = deferred_outcome(deferred);
    Py_BEGIN_ALLOW_THREADS;
    usearch_outcome_wait(outcome);
    Py_END_ALLOW_THREADS;
}

/**
 *  @brief Finishes a final @p self exactly once: the thread taking its verb answers, publishes,
 *      then lets go of its inputs.
 *
 *  Answering may run Python, which lets another thread in even under the GIL, so the verb is taken
 *  before answering and the answer published after it, never one check spanning both.
 */
static void deferred_finish(Deferred* self) {
    usearch_py_verb_t const* finishing = NULL;
    usearch_py_critical_section_begin_((PyObject*)self);
    finishing = self->verb;
    self->verb = NULL;
    usearch_py_critical_section_end_();
    if (!finishing) return;
    PyObject* const answer = call_answer(finishing, self->call);
    PyObject* const raised = answer ? NULL : PyErr_GetRaisedException();
    usearch_py_critical_section_begin_((PyObject*)self);
    self->answer = answer;
    self->raised = raised;
    usearch_py_critical_section_end_();
    call_release(finishing, self->call);
}

/** Answers what a final @p self finished with, finishing it first when nobody has. */
static PyObject* deferred_answer(Deferred* self) {
    deferred_finish(self);
    for (;;) {
        PyObject* answer = NULL;
        int published = 0;
        usearch_py_critical_section_begin_((PyObject*)self);
        published = self->answer || self->raised;
        if (self->raised) PyErr_SetRaisedException(Py_NewRef(self->raised));
        else if (self->answer) answer = Py_NewRef(self->answer);
        usearch_py_critical_section_end_();
        if (published) return answer;
        // Another thread took the verb and is still answering, so this one yields until it
        // publishes.
        Py_BEGIN_ALLOW_THREADS;
        Py_END_ALLOW_THREADS;
    }
}

static char const doc_Deferred_status[] =                                                        //
    "The status of the call: `Status.DEFERRED` until it is final, then the final one, such as\n" //
    "`Status.SUCCESS`. Calls `usearch_outcome_status`.";

static PyObject* Deferred_get_status(Deferred* self, void* closure) {
    usearch_status_t const status = deferred_status(self);
    // A poller that saw the call final never needs to wait, so the inputs are let go of here too.
    if (status != usearch_status_deferred_k) deferred_finish(self);
    return usearch_py_enum_to_object(&usearch_py_status, (int)status);
}

static char const doc_Deferred_wait[] =                                                      //
    "wait() -> object\n"                                                                     //
    "\n"                                                                                     //
    "Blocks the calling thread, letting other threads run, until the call is final,\n"       //
    "then answers what the blocking verb would have answered; `await` is the native form.\n" //
    "Calls `usearch_outcome_wait`.\n"                                                        //
    "\n"                                                                                     //
    "Raises:\n"                                                                              //
    "  Error: What the blocking verb would have raised, `ContendedError` included when\n"    //
    "    the executor's deferred table was full.";

static PyObject* Deferred_wait(Deferred* self, PyObject* unused) {
    deferred_block(self);
    return deferred_answer(self);
}

/** Answers the iterator of an `asyncio.Future` that `wait()` resolves on the running loop's default executor. */
static PyObject* Deferred_await(PyObject* self) {
    PyObject* const asyncio = PyImport_ImportModule("asyncio");
    PyObject* const loop = asyncio ? method_call(asyncio, "get_running_loop", NULL, NULL) : NULL;
    Py_XDECREF(asyncio);
    PyObject* const wait = loop ? PyObject_GetAttrString(self, "wait") : NULL;
    PyObject* const future = wait ? method_call(loop, "run_in_executor", Py_None, wait) : NULL;
    Py_XDECREF(wait);
    Py_XDECREF(loop);
    PyObject* const awaiting = future ? method_call(future, "__await__", NULL, NULL) : NULL;
    Py_XDECREF(future);
    return awaiting;
}

static void Deferred_dealloc(Deferred* self) {
    PyObject_GC_UnTrack(self);
    // The library writes into the call this object holds, and reads the inputs it borrows, until the call is final.
    if (self->verb) {
        deferred_block(self);
        call_release(self->verb, self->call);
    }
    Py_XDECREF(self->answer);
    Py_XDECREF(self->raised);
    Py_XDECREF(self->origin);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** Visits what @p self holds; an unfinished call's own objects stay unvisited, so every cycle
 *  through them waits for a later collection rather than being cleared while the library still
 *  reads them. */
static int Deferred_traverse(Deferred* self, visitproc visit, void* arg) {
    Py_VISIT(self->origin);
    Py_VISIT(self->answer);
    Py_VISIT(self->raised);
    if (!self->verb || deferred_status(self) == usearch_status_deferred_k) return 0;
    usearch_py_call_t const* const header = (usearch_py_call_t const*)(void const*)self->call;
    Py_VISIT(header->raised);
    Py_VISIT(header->arena);
    return self->verb->traverse ? self->verb->traverse(self->call, visit, arg) : 0;
}

/** Breaks a garbage cycle through a final @p self; an unfinished one is left whole, as its
 *  traversal hid the cycle. */
static int Deferred_clear(Deferred* self) {
    if (deferred_status(self) == usearch_status_deferred_k) return 0;
    usearch_py_verb_t const* const releasing = self->verb;
    self->verb = NULL;
    if (releasing) call_release(releasing, self->call);
    Py_CLEAR(self->answer);
    Py_CLEAR(self->raised);
    Py_CLEAR(self->origin);
    return 0;
}

static PyObject* Deferred_repr(Deferred* self) {
    return PyUnicode_FromFormat("Deferred(status=%s)", usearch_status_name(deferred_status(self)));
}

static PyMethodDef Deferred_methods[] = {
    {"wait", (PyCFunction)Deferred_wait, METH_NOARGS, doc_Deferred_wait},
    {"__class_getitem__", Py_GenericAlias, METH_O | METH_CLASS, "Names what `wait()` answers, as `Deferred[Matches]`."},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef Deferred_getset[] = {
    {"status", (getter)Deferred_get_status, NULL, doc_Deferred_status, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyAsyncMethods Deferred_as_async = {.am_await = Deferred_await};

PyTypeObject DeferredType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.Deferred",
    .tp_doc = doc_Deferred,
    .tp_basicsize = sizeof(Deferred),
    .tp_itemsize = 1,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_new = NULL, // Answered by the verbs of a deferred receiver, never constructed by name.
    .tp_dealloc = (destructor)Deferred_dealloc,
    .tp_traverse = (traverseproc)Deferred_traverse,
    .tp_clear = (inquiry)Deferred_clear,
    .tp_repr = (reprfunc)Deferred_repr,
    .tp_as_async = &Deferred_as_async,
    .tp_methods = Deferred_methods,
    .tp_getset = Deferred_getset,
};

#pragma endregion Receivers and Deferred Calls

#pragma region Executors

static char const doc_Executor[] =                                                          //
    "Executor\n"                                                                            //
    "\n"                                                                                    //
    "The workers a verb runs on, shared across indexes.\n"                                  //
    "\n"                                                                                    //
    "An executor owns host threads or device streams; an index owns rows. Keeping them\n"   //
    "apart is what lets one pool serve several indexes, and what makes the cost of\n"       //
    "spawning visible at the line that spawns rather than hidden in a constructor.\n"       //
    "\n"                                                                                    //
    "Construct one through the residency it drives - `for_ram`, `on_pool`, `for_vram` or\n" //
    "`for_nvme` - since each spawns something different and takes different arguments.\n"   //
    "Its residency must match that of every index it is handed to. A receiver names it:\n"  //
    "`index.on(executor)` blocks on it, `index.deferred(executor)` submits to it. Calls\n"  //
    "start in submission order, a blocking one after every earlier submission finishes,\n"  //
    "and a deferred one finding its `deferred_limit` table full raises `ContendedError`\n"  //
    "once waited for. Used in a `with` block, it waits for its calls and closes on exit.\n" //
    "\n"                                                                                    //
    "Example:\n"                                                                            //
    "  >>> from array import array\n"                                                       //
    "  >>> pool = usearch.Executor.for_ram(threads=8)\n"                                    //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=4)\n"                             //
    "  >>> index.reserve(1)\n"                                                              //
    "  >>> index.on(pool).insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))\n"                  //
    "  >>> pool.drain(usearch.Mood.PARKED)";

/**
 *  @brief Wraps a fresh handle, or frees it when the wrapper cannot be allocated.
 *  @return A new reference, or NULL with an exception set.
 */
static PyObject* executor_adopt(PyTypeObject* type, usearch_executor_t handle, usearch_residency_t residency) {
    Executor* const executor = (Executor*)type->tp_alloc(type, 0);
    if (!executor) {
        usearch_executor_free(handle);
        return NULL;
    }
    executor->handle = handle;
    executor->residency = residency;
    return (PyObject*)executor;
}

static PyObject* Executor_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_TypeError,
                    "Executor is built through Executor.for_ram(), .on_pool(), .for_vram() or .for_nvme()");
    return NULL;
}

/** Frees the handle of @p executor once, waiting in C for every deferred call naming it, fenced
 *  against a concurrent close. */
static void executor_close(Executor* executor) {
    usearch_executor_t handle = NULL;
    usearch_py_critical_section_begin_((PyObject*)executor);
    handle = executor->handle;
    executor->handle = NULL;
    usearch_py_critical_section_end_();
    if (!handle) return;
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_free(handle);
    Py_END_ALLOW_THREADS;
}

static void Executor_dealloc(Executor* executor) {
    executor_close(executor);
    Py_TYPE(executor)->tp_free((PyObject*)executor);
}

static PyObject* Executor_enter(PyObject* self, PyObject* unused) { return Py_NewRef(self); }

static PyObject* Executor_exit(PyObject* self, PyObject* const* args, Py_ssize_t count) {
    executor_close((Executor*)self);
    Py_RETURN_NONE;
}

int usearch_py_executor_from_object(PyObject* object, usearch_executor_t* executor) {
    if (!PyObject_TypeCheck(object, &ExecutorType)) {
        PyErr_Format(PyExc_TypeError, "expected an Executor, got %S", Py_TYPE(object));
        return -1;
    }
    usearch_py_critical_section_begin_(object);
    *executor = ((Executor*)object)->handle;
    usearch_py_critical_section_end_();
    // Without this, a closed executor would quietly run the call on the calling thread.
    if (!*executor) return usearch_py_raise(usearch_status_uninitialized_k, "the executor was closed");
    return 0;
}

static char const doc_Executor_for_ram[] =                                                   //
    "for_ram(threads=0, *, memory_domains=None, deferred_limit=0) -> Executor\n"             //
    "\n"                                                                                     //
    "Spawns host worker threads for a RAM-resident index.\n"                                 //
    "Calls `usearch_executor_for_ram`.\n"                                                    //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  threads (int, optional): Workers to spawn; 0 derives from the hardware.\n"            //
    "  memory_domains (Sequence[int], optional): NUMA node IDs the workers stay local\n"     //
    "    to; None derives every node the process may allocate on.\n"                         //
    "  deferred_limit (int, optional): Deferred calls tracked at once; 0 takes a default.\n" //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> with usearch.Executor.for_ram(threads=8) as pool:\n"                              //
    "  ...     pool.drain()";

static PyObject* Executor_for_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                  PyObject* args_names_tuple) {
    PyObject *threads_object, *memory_domains_object, *deferred_limit_object;
    usearch_py_argument_t const arguments[] = {
        {"threads", &threads_object},
        {"memory_domains", &memory_domains_object},
        {"deferred_limit", &deferred_limit_object},
    };
    size_t threads = 0, deferred_limit = 0;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    if (usearch_py_arguments_parse("for_ram", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_size_from_object(threads_object, "threads", &threads) != 0 ||
        usearch_py_size_from_object(deferred_limit_object, "deferred_limit", &deferred_limit) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        return NULL;

    usearch_executor_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_executor_for_ram_t executor_arguments = {.executor = &handle,
                                                     .outcome = &outcome,
                                                     .ram = {memory_domains, memory_domains_count},
                                                     .threads = threads,
                                                     .deferred_limit = deferred_limit};
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_for_ram(&executor_arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);

    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle, usearch_residency_ram_k);
}

static char const doc_Executor_on_pool[] =                                                   //
    "on_pool(pool, /, *, memory_domains=None, deferred_limit=0) -> Executor\n"               //
    "\n"                                                                                     //
    "Drives RAM-resident indexes on a ForkUnion pool another library already spawned, so\n"  //
    "one process runs one set of threads while each library keeps its own scratch. The\n"    //
    "pool keeps its own threads and its lender's mood.\n"                                    //
    "Calls `usearch_executor_on_pool`.\n"                                                    //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  pool (int): The address of the `fu_pool_t` to borrow, as `ctypes` and `cffi` hand\n"  //
    "    pointers over; the pool must outlive the executor.\n"                               //
    "  memory_domains (Sequence[int], optional): NUMA node IDs of the indexes it will\n"     //
    "    drive; None derives every node the process may allocate on.\n"                      //
    "  deferred_limit (int, optional): Deferred calls tracked at once; 0 takes a default.\n" //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> pool = usearch.Executor.on_pool(fu_pool_address)   # doctest: +SKIP";

static PyObject* Executor_on_pool(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                  PyObject* args_names_tuple) {
    PyObject *pool_object, *memory_domains_object, *deferred_limit_object;
    usearch_py_argument_t const arguments[] = {
        {"pool", &pool_object},
        {"memory_domains", &memory_domains_object},
        {"deferred_limit", &deferred_limit_object},
    };
    size_t deferred_limit = 0;
    if (usearch_py_arguments_parse("on_pool", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_size_from_object(deferred_limit_object, "deferred_limit", &deferred_limit) != 0)
        return NULL;

    PyObject* const address = PyNumber_Index(pool_object);
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
    usearch_executor_on_pool_t executor_arguments = {.executor = &handle,
                                                     .outcome = &outcome,
                                                     .ram = {memory_domains, memory_domains_count},
                                                     .pool = pool,
                                                     .deferred_limit = deferred_limit};
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_on_pool(&executor_arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);

    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle, usearch_residency_ram_k);
}

static char const doc_Executor_for_vram[] =                                              //
    "for_vram(*, devices=None, deferred_limit=0) -> Executor\n"                          //
    "\n"                                                                                 //
    "Opens one stream per device for a VRAM-resident index.\n"                           //
    "\n"                                                                                 //
    "Takes no thread count: a single host thread drives every stream, so the width of\n" //
    "the pool is the device list rather than a worker count.\n"                          //
    "Calls `usearch_executor_for_vram`.\n"                                               //
    "\n"                                                                                 //
    "Args:\n"                                                                            //
    "  devices (Sequence[int], optional): CUDA ordinals to drive; None derives one\n"    //
    "    free device.\n"                                                                 //
    "  deferred_limit (int, optional): Deferred calls tracked at once; 0 takes a\n"      //
    "    default.\n"                                                                     //
    "\n"                                                                                 //
    "Example:\n"                                                                         //
    "  >>> pool = usearch.Executor.for_vram(devices=[0, 1])   # doctest: +SKIP";

static PyObject* Executor_for_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *devices_object, *deferred_limit_object;
    usearch_py_argument_t const arguments[] = {
        {"devices", &devices_object},
        {"deferred_limit", &deferred_limit_object},
    };
    size_t deferred_limit = 0;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    if (usearch_py_arguments_parse("for_vram", args, positional_args_count, args_names_tuple, 0, 0, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_size_from_object(deferred_limit_object, "deferred_limit", &deferred_limit) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        return NULL;

    usearch_executor_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_executor_for_vram_t executor_arguments = {.executor = &handle,
                                                      .outcome = &outcome,
                                                      .vram = {devices, devices_count},
                                                      .deferred_limit = deferred_limit};
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_for_vram(&executor_arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(devices);

    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle, usearch_residency_vram_k);
}

static char const doc_Executor_for_nvme[] =                                             //
    "for_nvme(threads=0, *, paths=None, deferred_limit=0) -> Executor\n"                //
    "\n"                                                                                //
    "Opens one submission ring per worker for an NVMe-resident index.\n"                //
    "Calls `usearch_executor_for_nvme`.\n"                                              //
    "\n"                                                                                //
    "Args:\n"                                                                           //
    "  threads (int, optional): Rings to open; 0 derives from the hardware.\n"          //
    "  paths (Sequence[str | bytes | PathLike], optional): Devices or directories to\n" //
    "    stripe shards across; None leaves every shard where its manifest points. A\n"  //
    "    path may name a kernel device, a transport identifier, or a FUSE mount.\n"     //
    "  deferred_limit (int, optional): Deferred calls tracked at once; 0 takes a\n"     //
    "    default.\n"                                                                    //
    "\n"                                                                                //
    "Example:\n"                                                                        //
    "  >>> pool = usearch.Executor.for_nvme(threads=8, paths=['/mnt/nvme0'])  # doctest: +SKIP";

static PyObject* Executor_for_nvme(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *threads_object, *paths_object, *deferred_limit_object;
    usearch_py_argument_t const arguments[] = {
        {"threads", &threads_object},
        {"paths", &paths_object},
        {"deferred_limit", &deferred_limit_object},
    };
    size_t threads = 0, deferred_limit = 0;
    usearch_py_paths_t paths;
    if (usearch_py_arguments_parse("for_nvme", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_size_from_object(threads_object, "threads", &threads) != 0 ||
        usearch_py_size_from_object(deferred_limit_object, "deferred_limit", &deferred_limit) != 0 ||
        usearch_py_paths_from_object(paths_object, &paths) != 0)
        return NULL;

    usearch_executor_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_executor_for_nvme_t executor_arguments = {.executor = &handle,
                                                      .outcome = &outcome,
                                                      .nvme = {paths.views, paths.count},
                                                      .threads = threads,
                                                      .deferred_limit = deferred_limit};
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_for_nvme(&executor_arguments);
    Py_END_ALLOW_THREADS;
    usearch_py_paths_release(&paths);

    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return executor_adopt(type, handle, usearch_residency_nvme_k);
}

static char const doc_Executor_drain[] =                                                  //
    "drain(mood=Mood.PARKED) -> None\n"                                                   //
    "\n"                                                                                  //
    "Waits for every call submitted to this executor before the drain began, then sets\n" //
    "what it does with its workers whenever it next goes idle; the last drain wins.\n"    //
    "Calls `usearch_executor_drain`.\n"                                                   //
    "\n"                                                                                  //
    "Args:\n"                                                                             //
    "  mood (Mood, optional): `Mood.PARKED` lets idle workers sleep, so the next call\n"  //
    "    pays a wake-up; `Mood.SPINNING` keeps them hot, trading power for the next\n"    //
    "    call's latency. A borrowed pool keeps its lender's mood.\n"                      //
    "\n"                                                                                  //
    "Raises:\n"                                                                           //
    "  Error: What befell the executor itself, such as a lost device; every call\n"       //
    "    reports its own failure through its own verb.";

static PyObject* Executor_drain(Executor* executor, PyObject* const* args, Py_ssize_t positional_args_count,
                                PyObject* args_names_tuple) {
    PyObject* mood_object;
    usearch_py_argument_t const arguments[] = {{"mood", &mood_object}};
    int mood = usearch_mood_parked_k;
    usearch_executor_t handle = NULL;
    if (usearch_py_arguments_parse("drain", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        (mood_object && usearch_py_enum_from_object(mood_object, "mood", &usearch_py_mood, &mood) != 0) ||
        usearch_py_executor_from_object((PyObject*)executor, &handle) != 0)
        return NULL;

    usearch_outcome_t outcome = {0};
    usearch_executor_drain_t drain_arguments = {.executor = handle, .outcome = &outcome, .mood = (usearch_mood_t)mood};
    Py_BEGIN_ALLOW_THREADS;
    usearch_executor_drain(&drain_arguments);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject* Executor_get_residency(Executor* executor, void* closure) {
    return usearch_py_enum_to_object(&usearch_py_residency, executor->residency);
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
    {"__enter__", (PyCFunction)Executor_enter, METH_NOARGS, NULL},                                               //
    {"__exit__", (PyCFunction)Executor_exit, METH_FASTCALL, NULL},                                               //
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef Executor_getset[] = {
    {"residency", (getter)Executor_get_residency, NULL, "The `Residency` its calls compute in. Python-only.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
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
    .tp_getset = Executor_getset,
    .tp_repr = (reprfunc)Executor_repr,
};

#pragma endregion Executors

#pragma region Arenas

static char const doc_Arena[] =                                                                //
    "Arena(executor=None, /)\n"                                                                //
    "\n"                                                                                       //
    "The memory every output whose length scales with a call lands in - matches, rows,\n"      //
    "products, monoliths - which every such verb requires and each call rewinds, so a\n"       //
    "steady loop allocates once.\n"                                                            //
    "\n"                                                                                       //
    "The executor makes it where its calls compute, host pages or the memory of its GPU,\n"    //
    "and none makes host pages for calls on the calling thread; a call of another residency\n" //
    "raises `DeviceError`. Results are zero-copy views over it. Like a `bytearray`, it\n"      //
    "refuses to rewind with `ContendedError` while a buffer over its outputs is exported,\n"   //
    "and a result read after a later call rewound it raises `UninitializedError`. A\n"         //
    "deferred call holds it as a blocking one does, until it is waited on or dropped.\n"       //
    "Results over device memory export `__cuda_array_interface__`, and host reads of them\n"   //
    "raise `DeviceError`.\n"                                                                   //
    "Calls `usearch_arena_init`.\n"                                                            //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  executor (Executor, optional): The executor whose calls it serves; None serves the\n"   //
    "    calling thread.\n"                                                                    //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> from array import array\n"                                                          //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=2)\n"                                //
    "  >>> index.reserve(2)\n"                                                                 //
    "  >>> index.insert([1], array('d', [0.1, 0.2]))\n"                                        //
    "  >>> index.insert([2], array('d', [0.3, 0.4]))\n"                                        //
    "  >>> arena = usearch.Arena()\n"                                                          //
    "  >>> first = memoryview(index.get(1, arena))\n"                                          //
    "  >>> first.release()\n"                                                                  //
    "  >>> memoryview(index.get(2, arena)).shape\n"                                            //
    "  (1, 2)";

static PyObject* Arena_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    Py_ssize_t const positional_args_count = PyTuple_GET_SIZE(args);
    if (positional_args_count > 1 || (keywords && PyDict_GET_SIZE(keywords) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Arena() takes one optional positional argument, the executor");
        return NULL;
    }
    PyObject* const executor_object = positional_args_count ? PyTuple_GET_ITEM(args, 0) : Py_None;
    usearch_executor_t executor = NULL;
    usearch_residency_t residency = usearch_residency_ram_k;
    if (executor_object != Py_None) {
        if (usearch_py_executor_from_object(executor_object, &executor) != 0) return NULL;
        residency = ((Executor*)executor_object)->residency;
    }

    usearch_arena_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_arena_init_t arguments = {.arena = &handle, .executor = executor, .outcome = &outcome};
    usearch_arena_init(&arguments);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    Arena* const arena = (Arena*)type->tp_alloc(type, 0);
    if (!arena) {
        usearch_arena_free(handle);
        return NULL;
    }
    arena->handle = handle;
    arena->residency = residency;
    return (PyObject*)arena;
}

static void Arena_dealloc(Arena* self) {
    Py_BEGIN_ALLOW_THREADS;
    usearch_arena_free(self->handle);
    Py_END_ALLOW_THREADS;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Arena_get_residency(Arena* self, void* closure) {
    return usearch_py_enum_to_object(&usearch_py_residency, self->residency);
}

static PyGetSetDef Arena_getset[] = {
    {"residency", (getter)Arena_get_residency, NULL, "The `Residency` its outputs live in. Python-only.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject ArenaType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.Arena",
    .tp_doc = doc_Arena,
    .tp_basicsize = sizeof(Arena),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Arena_tp_new,
    .tp_dealloc = (destructor)Arena_dealloc,
    .tp_getset = Arena_getset,
};

int usearch_py_arena_claim(PyObject* object, PyObject** arena) {
    if (!PyObject_TypeCheck(object, &ArenaType)) {
        PyErr_Format(PyExc_TypeError, "the 'arena' argument must be an Arena, got %S", Py_TYPE(object));
        return -1;
    }
    Arena* const claimed = (Arena*)object;
    Py_ssize_t held = 0;
    usearch_py_critical_section_begin_(object);
    held = claimed->exports;
    if (!held) {
        claimed->exports = 1;
        ++claimed->generation;
    }
    usearch_py_critical_section_end_();
    if (held)
        return usearch_py_raise(usearch_status_contended_k,
                                "the arena serves a call or exports a buffer over its outputs; release it first");
    *arena = Py_NewRef(object);
    return 0;
}

void usearch_py_arena_unclaim(PyObject* arena) {
    usearch_py_critical_section_begin_(arena);
    --((Arena*)arena)->exports;
    usearch_py_critical_section_end_();
    Py_DECREF(arena);
}

/** The rank of a ranked run's keys, queries by columns by sixteen bytes, and so of every output in
 *  an arena. */
enum { arena_buffer_max_rank_k = 3 };

/** One output a call left in an arena, exported zero-copy, and the base of every result that is one. */
typedef struct ArenaBuffer {
    PyObject ob_base;

    /** The arena the output lives in, kept alive so the memory outlives every other name for it. */
    PyObject* arena;

    /** The arena's generation when the call wrote the output; any later one means a call rewound
     *  it. */
    size_t generation;

    void const* data;
    char const* format;
    Py_ssize_t item_bytes;
    int rank;
    Py_ssize_t shape[arena_buffer_max_rank_k];
    Py_ssize_t strides[arena_buffer_max_rank_k];
} ArenaBuffer;

/** Whether @p buffer lives in device memory, which the host may not dereference. */
static int arena_buffer_on_device(ArenaBuffer const* buffer) {
    return ((Arena*)buffer->arena)->residency == usearch_residency_vram_k;
}

/** Raises @c UninitializedError unless the call that wrote @p buffer is still the last to claim
 *  its arena; returns 0 or -1. */
static int arena_buffer_fresh(ArenaBuffer const* buffer, size_t generation) {
    if (generation == buffer->generation) return 0;
    return usearch_py_raise(usearch_status_uninitialized_k, "a later call rewound the arena this output lived in");
}

/**
 *  @brief Holds the arena of @p buffer against a rewind while its host memory is read, as one more
 *      export.
 *  @return 0 on success, -1 with a @c DeviceError set when the output is on a device, or an
 *      @c UninitializedError when a later call rewound it.
 */
static int arena_buffer_pin(ArenaBuffer const* buffer) {
    if (arena_buffer_on_device(buffer))
        return usearch_py_raise(usearch_status_device_k,
                                "the output lives in device memory; read `__cuda_array_interface__`");
    Arena* const arena = (Arena*)buffer->arena;
    size_t generation = 0;
    usearch_py_critical_section_begin_((PyObject*)arena);
    generation = arena->generation;
    if (generation == buffer->generation) ++arena->exports;
    usearch_py_critical_section_end_();
    return arena_buffer_fresh(buffer, generation);
}

/** Ends the hold @c arena_buffer_pin took. */
static void arena_buffer_unpin(ArenaBuffer const* buffer) {
    usearch_py_critical_section_begin_(buffer->arena);
    --((Arena*)buffer->arena)->exports;
    usearch_py_critical_section_end_();
}

static int ArenaBuffer_getbuffer(ArenaBuffer* self, Py_buffer* view, int flags) {
    view->obj = NULL;
    if (arena_buffer_pin(self) != 0) return -1;
    Py_ssize_t bytes = self->item_bytes;
    for (int axis = 0; axis != self->rank; ++axis) bytes *= self->shape[axis];
    view->obj = Py_NewRef((PyObject*)self);
    view->buf = (void*)self->data;
    view->len = bytes;
    view->readonly = 0;
    // The item size describes the cells the shape counts, whether or not the consumer asked for their format.
    view->itemsize = self->item_bytes;
    view->format = (flags & PyBUF_FORMAT) == PyBUF_FORMAT ? (char*)self->format : NULL;
    view->ndim = self->rank;
    view->shape = (flags & PyBUF_ND) == PyBUF_ND ? self->shape : NULL;
    view->strides = (flags & PyBUF_STRIDES) == PyBUF_STRIDES ? self->strides : NULL;
    view->suboffsets = NULL;
    view->internal = NULL;
    return 0;
}

static void ArenaBuffer_releasebuffer(ArenaBuffer* self, Py_buffer* view) { arena_buffer_unpin(self); }

/** A tuple of the first @p rank of @p extents; a new reference, or NULL. */
static PyObject* extents_to_tuple(Py_ssize_t const* extents, int rank) {
    PyObject* const tuple = PyTuple_New(rank);
    for (int axis = 0; tuple && axis != rank; ++axis) {
        PyObject* const extent = PyLong_FromSsize_t(extents[axis]);
        if (!extent) {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, axis, extent);
    }
    return tuple;
}

static PyObject* ArenaBuffer_get_cuda_array_interface(ArenaBuffer* self, void* closure) {
    if (!arena_buffer_on_device(self)) {
        PyErr_SetString(PyExc_AttributeError, "the output lives in host memory; read it through the buffer protocol");
        return NULL;
    }
    size_t generation = 0;
    usearch_py_critical_section_begin_(self->arena);
    generation = ((Arena*)self->arena)->generation;
    usearch_py_critical_section_end_();
    if (arena_buffer_fresh(self, generation) != 0) return NULL;
    char const kind = *self->format == 'b' ? 'i' : strchr("efd", *self->format) ? 'f' : 'u';
    char const typestr[3] = {self->item_bytes == 1 ? '|' : '<', kind, (char)('0' + self->item_bytes)};
    PyObject* const address = PyLong_FromVoidPtr((void*)self->data);
    PyObject* const data = address ? PyTuple_Pack(2, address, Py_False) : NULL;
    Py_XDECREF(address);
    PyObject* const interface = data ? PyDict_New() : NULL;
    if (!interface) {
        Py_XDECREF(data);
        return NULL;
    }
    if (usearch_py_dict_steal_item(interface, "data", data) != 0 ||
        usearch_py_dict_steal_item(interface, "shape", extents_to_tuple(self->shape, self->rank)) != 0 ||
        usearch_py_dict_steal_item(interface, "strides", extents_to_tuple(self->strides, self->rank)) != 0 ||
        usearch_py_dict_steal_item(interface, "typestr", PyUnicode_FromStringAndSize(typestr, 3)) != 0 ||
        usearch_py_dict_steal_item(interface, "version", PyLong_FromLong(3)) != 0) {
        Py_DECREF(interface);
        return NULL;
    }
    return interface;
}

static void ArenaBuffer_dealloc(ArenaBuffer* self) {
    Py_DECREF(self->arena);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static Py_ssize_t ArenaBuffer_length(ArenaBuffer* self) { return self->shape[0]; }

static PySequenceMethods ArenaBuffer_as_sequence = {.sq_length = (lenfunc)ArenaBuffer_length};

static PyBufferProcs ArenaBuffer_as_buffer = {
    .bf_getbuffer = (getbufferproc)ArenaBuffer_getbuffer,
    .bf_releasebuffer = (releasebufferproc)ArenaBuffer_releasebuffer,
};

static PyGetSetDef ArenaBuffer_getset[] = {
    {"__cuda_array_interface__", (getter)ArenaBuffer_get_cuda_array_interface, NULL,
     "Version 3 of the CUDA Array Interface, present only over device memory. Python-only.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject ArenaBufferType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.ArenaBuffer",
    .tp_doc = "One output a call left in an Arena, exported zero-copy.",
    .tp_basicsize = sizeof(ArenaBuffer),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, // Behind the results a verb returns, never constructed by name.
    .tp_dealloc = (destructor)ArenaBuffer_dealloc,
    .tp_as_sequence = &ArenaBuffer_as_sequence,
    .tp_as_buffer = &ArenaBuffer_as_buffer,
    .tp_getset = ArenaBuffer_getset,
};

static char const doc_DenseVectors[] =                                                           //
    "DenseVectors\n"                                                                             //
    "\n"                                                                                         //
    "The rows `get`, `attend` or a cursor wrote into an `Arena`, viewing it until it rewinds.\n" //
    "\n"                                                                                         //
    "A (count, dimensions) matrix for the kinds a buffer can format, and a (count,\n"            //
    "row_bytes) run of 'B' for the rest. Host rows export the writable buffer protocol;\n"       //
    "device rows export `__cuda_array_interface__` instead.";

PyTypeObject DenseVectorsType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseVectors",
    .tp_doc = doc_DenseVectors,
    .tp_basicsize = sizeof(ArenaBuffer),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_base = &ArenaBufferType,
};

static char const doc_Monolith[] =                                                         //
    "Monolith\n"                                                                           //
    "\n"                                                                                   //
    "The bytes `save_buffer` wrote into an `Arena`, viewing it until it rewinds: one\n"    //
    "(length,) run of 'B'. Host bytes export the writable buffer protocol; device bytes\n" //
    "export `__cuda_array_interface__` instead.";

PyTypeObject MonolithType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.Monolith",
    .tp_doc = doc_Monolith,
    .tp_basicsize = sizeof(ArenaBuffer),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_base = &ArenaBufferType,
};

PyObject* usearch_py_arena_buffer(PyTypeObject* type, PyObject* arena, void const* data, char const* format,
                                  Py_ssize_t item_bytes, Py_ssize_t const* shape, int rank) {
    // A call with nothing to report leaves its output null, which no exporter may hand out.
    static char const empty_output = 0;
    ArenaBuffer* const exporter = PyObject_New(ArenaBuffer, type);
    if (!exporter) return NULL;
    exporter->arena = Py_NewRef(arena);
    // Built while the writing call still holds the arena, so its generation is that call's.
    exporter->generation = ((Arena*)arena)->generation;
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
    return (PyObject*)exporter;
}

/** Rows @p begin up to @p end of @p whole, a new @c ArenaBuffer over the same output; NULL with an
 *  exception set. */
static PyObject* arena_buffer_slice(ArenaBuffer const* whole, size_t begin, size_t end) {
    Py_ssize_t shape[arena_buffer_max_rank_k];
    memcpy(shape, whole->shape, sizeof(shape));
    shape[0] = (Py_ssize_t)(end - begin);
    ArenaBuffer* const slice = (ArenaBuffer*)usearch_py_arena_buffer(
        &ArenaBufferType, whole->arena, (char const*)whole->data + begin * whole->strides[0], whole->format,
        whole->item_bytes, shape, whole->rank);
    if (slice) slice->generation = whole->generation;
    return (PyObject*)slice;
}

/** A ranked run a verb wrote into an arena: three outputs of one row per query. */
typedef struct Matches {
    PyObject ob_base;

    /** The (queries, columns, 16) @c ArenaBuffer of identities. */
    PyObject* keys;

    /** The (queries, columns) @c ArenaBuffer of float64 distances, positionally paired with
     *  @c keys. */
    PyObject* distances;

    /** The (queries,) @c ArenaBuffer of match counts, one per query. */
    PyObject* counts;

    /** Rows in each output - one per query. */
    size_t queries;

    /** Columns in the key and distance outputs, at most the call's "k" in "kANN". */
    size_t columns;
} Matches;

static char const doc_Matches[] =                                                               //
    "Matches\n"                                                                                 //
    "\n"                                                                                        //
    "A ranked run `search` or `maxsim` wrote into an `Arena`, viewing it until it rewinds.\n"   //
    "\n"                                                                                        //
    "Reads through to three outputs, so `np.asarray` costs nothing and NumPy is never\n"        //
    "required; a buffer exported over them holds the arena against the next call. No NumPy\n"   //
    "dtype expresses a 128-bit integer, so keys arrive as big-endian sixteen-byte rows. Each\n" //
    "count bounds its own row, and the cells past it hold nothing a call wrote.\n"              //
    "\n"                                                                                        //
    "Attributes:\n"                                                                             //
    "  keys (buffer): (queries, columns, 16) identities, most significant byte first.\n"        //
    "  distances (buffer): (queries, columns) distances, columns at most the call's wanted.\n"  //
    "  counts (buffer): (queries,) match counts, each bounding its own row.\n"                  //
    "\n"                                                                                        //
    "Example:\n"                                                                                //
    "  >>> from array import array\n"                                                           //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=2)\n"                                 //
    "  >>> index.reserve(1)\n"                                                                  //
    "  >>> index.insert([5], array('d', [0.1, 0.2]))\n"                                         //
    "  >>> matches = index.search(array('d', [0.1, 0.2]), 3, usearch.Arena())\n"                //
    "  >>> len(matches), int(matches[0][0].key)\n"                                              //
    "  (1, 5)";

PyObject* usearch_py_matches_new(PyObject* arena, usearch_matches_view_t const* view, size_t queries) {
    Py_ssize_t const keys_shape[3] = {(Py_ssize_t)queries, (Py_ssize_t)view->columns,
                                      (Py_ssize_t)sizeof(usearch_key_t)};
    Py_ssize_t const counts_shape[1] = {(Py_ssize_t)queries};
    Matches* matches = PyObject_New(Matches, &MatchesType);
    if (!matches) return NULL;
    matches->queries = queries;
    matches->columns = view->columns;
    matches->distances = matches->counts = NULL;
    matches->keys = usearch_py_arena_buffer(&ArenaBufferType, arena, view->keys, "B", 1, keys_shape, 3);
    if (matches->keys)
        matches->distances = usearch_py_arena_buffer(&ArenaBufferType, arena, view->scores, "d",
                                                     sizeof(usearch_distance_t), keys_shape, 2);
    if (matches->distances)
        matches->counts = usearch_py_arena_buffer(&ArenaBufferType, arena, view->counts, "N", sizeof(size_t),
                                                  counts_shape, 1);
    if (!matches->counts) Py_CLEAR(matches);
    return (PyObject*)matches;
}

static void Matches_dealloc(Matches* self) {
    Py_XDECREF(self->counts);
    Py_XDECREF(self->distances);
    Py_XDECREF(self->keys);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static Py_ssize_t Matches_length(Matches* self) { return (Py_ssize_t)self->queries; }

static PyObject* Matches_item(Matches* self, Py_ssize_t query_index) {
    // `PySequence_GetItem` normalizes a negative index before reaching here.
    if (query_index < 0 || (size_t)query_index >= self->queries) {
        PyErr_SetString(PyExc_IndexError, "query index out of range");
        return NULL;
    }
    ArenaBuffer const* const keys_buffer = (ArenaBuffer const*)self->keys;
    if (arena_buffer_pin(keys_buffer) != 0) return NULL;
    size_t const* const counts = (size_t const*)((ArenaBuffer const*)self->counts)->data;
    usearch_key_t const* const keys = (usearch_key_t const*)keys_buffer->data;
    usearch_distance_t const* const distances = (usearch_distance_t const*)((ArenaBuffer const*)self->distances)->data;

    // A count is the engine's word, and the sibling `multiply` verb documents counts that
    // exceed the column width outright - so it bounds a read here rather than defining one.
    size_t const found = counts[query_index] < self->columns ? counts[query_index] : self->columns;
    size_t const offset = (size_t)query_index * self->columns;
    PyObject* row = PyTuple_New((Py_ssize_t)found);
    for (size_t match_index = 0; row && match_index != found; ++match_index) {
        PyObject* const match = usearch_py_match_to_object(keys[offset + match_index], distances[offset + match_index]);
        if (!match) Py_CLEAR(row);
        else PyTuple_SET_ITEM(row, (Py_ssize_t)match_index, match);
    }
    arena_buffer_unpin(keys_buffer);
    return row;
}

static PyObject* Matches_repr(Matches* self) {
    return PyUnicode_FromFormat("Matches(queries=%zu, columns=%zu)", self->queries, self->columns);
}

static PyMemberDef Matches_members[] = {
    {"keys", Py_T_OBJECT_EX, offsetof(Matches, keys), Py_READONLY,
     "(queries, columns, 16) identities, most significant byte first."},
    {"distances", Py_T_OBJECT_EX, offsetof(Matches, distances), Py_READONLY,
     "(queries, columns) distances, in the index's own metric."},
    {"counts", Py_T_OBJECT_EX, offsetof(Matches, counts), Py_READONLY,
     "(queries,) match counts, each bounding its own row."},
    {NULL, 0, 0, 0, NULL},
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
    .tp_new = NULL, // Returned by `search` and `maxsim`, never constructed by name.
    .tp_dealloc = (destructor)Matches_dealloc,
    .tp_repr = (reprfunc)Matches_repr,
    .tp_as_sequence = &Matches_as_sequence,
    .tp_members = Matches_members,
};

/** The ragged pairs `multiply` wrote into an arena: Apache Arrow offsets around keys and scores. */
typedef struct DenseProducts {
    PyObject ob_base;

    /** The (rows + 1,) @c ArenaBuffer of Apache Arrow offsets. */
    PyObject* offsets;

    /** The (pairs, 16) @c ArenaBuffer naming each pair's indexed row. */
    PyObject* keys;

    /** The (pairs,) @c ArenaBuffer of float64 products, positionally paired with @c keys. */
    PyObject* scores;

    /** The pairs scored before the bound cut any, an @c int. */
    PyObject* required;

    /** The left rows, one more than the offsets. */
    size_t rows;
} DenseProducts;

static char const doc_DenseProducts[] =                                                          //
    "DenseProducts\n"                                                                            //
    "\n"                                                                                         //
    "The ragged sparse product `multiply` wrote into an `Arena`, viewing it until it rewinds:\n" //
    "per left row, every pair its walk scored, in traversal order, ragged by Apache Arrow\n"     //
    "offsets.\n"                                                                                 //
    "\n"                                                                                         //
    "`offsets` is a (rows + 1,) run of unsigned words, row `i` spanning pairs `offsets[i]`\n"    //
    "up to `offsets[i + 1]`. `keys`, a (pairs, 16) run naming each pair's indexed row, and\n"    //
    "`scores`, the (pairs,) float64 products, sit beside them. `required` counts the pairs\n"    //
    "scored before `wanted` cut any. `len()` counts the left rows, and `row(i)` answers the\n"   //
    "(keys, scores) of one.";

PyObject* usearch_py_products_new(PyObject* arena, usearch_matches_tape_view_t const* view, size_t rows, size_t pairs,
                                  size_t required) {
    Py_ssize_t const offsets_shape[1] = {(Py_ssize_t)rows + 1};
    Py_ssize_t const keys_shape[2] = {(Py_ssize_t)pairs, (Py_ssize_t)sizeof(usearch_key_t)};
    DenseProducts* products = PyObject_New(DenseProducts, &DenseProductsType);
    if (!products) return NULL;
    products->rows = rows;
    products->keys = products->scores = products->required = NULL;
    products->offsets = usearch_py_arena_buffer(&ArenaBufferType, arena, view->offsets, "N", sizeof(size_t),
                                                offsets_shape, 1);
    if (products->offsets)
        products->keys = usearch_py_arena_buffer(&ArenaBufferType, arena, view->keys, "B", 1, keys_shape, 2);
    if (products->keys)
        products->scores = usearch_py_arena_buffer(&ArenaBufferType, arena, view->scores, "d",
                                                   sizeof(usearch_distance_t), keys_shape, 1);
    if (products->scores) products->required = PyLong_FromSize_t(required);
    if (!products->required) Py_CLEAR(products);
    return (PyObject*)products;
}

static void DenseProducts_dealloc(DenseProducts* self) {
    Py_XDECREF(self->required);
    Py_XDECREF(self->scores);
    Py_XDECREF(self->keys);
    Py_XDECREF(self->offsets);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static Py_ssize_t DenseProducts_length(DenseProducts* self) { return (Py_ssize_t)self->rows; }

static PyObject* DenseProducts_row(DenseProducts* self, PyObject* row_object) {
    Py_ssize_t const row = PyNumber_AsSsize_t(row_object, PyExc_IndexError);
    if (row == -1 && PyErr_Occurred()) return NULL;
    if (row < 0 || (size_t)row >= self->rows) {
        PyErr_SetString(PyExc_IndexError, "row index out of range");
        return NULL;
    }
    ArenaBuffer const* const offsets_buffer = (ArenaBuffer const*)self->offsets;
    if (arena_buffer_pin(offsets_buffer) != 0) return NULL;
    size_t const* const offsets = (size_t const*)offsets_buffer->data;
    size_t const begin = offsets[row], end = offsets[row + 1];
    arena_buffer_unpin(offsets_buffer);
    PyObject* const keys = arena_buffer_slice((ArenaBuffer*)self->keys, begin, end);
    PyObject* const scores = keys ? arena_buffer_slice((ArenaBuffer*)self->scores, begin, end) : NULL;
    PyObject* const pair = scores ? PyTuple_Pack(2, keys, scores) : NULL;
    Py_XDECREF(scores);
    Py_XDECREF(keys);
    return pair;
}

static PyObject* DenseProducts_repr(DenseProducts* self) {
    return PyUnicode_FromFormat("DenseProducts(rows=%zu, required=%S)", self->rows, self->required);
}

static PyMethodDef DenseProducts_methods[] = {
    {"row", (PyCFunction)DenseProducts_row, METH_O,
     "The (keys, scores) of left row `i`, viewing the arena. Python-only."},
    {NULL, NULL, 0, NULL},
};

static PyMemberDef DenseProducts_members[] = {
    {"offsets", Py_T_OBJECT_EX, offsetof(DenseProducts, offsets), Py_READONLY, "(rows + 1,) Apache Arrow offsets."},
    {"keys", Py_T_OBJECT_EX, offsetof(DenseProducts, keys), Py_READONLY, "(pairs, 16) identities of the indexed rows."},
    {"scores", Py_T_OBJECT_EX, offsetof(DenseProducts, scores), Py_READONLY, "(pairs,) float64 products."},
    {"required", Py_T_OBJECT_EX, offsetof(DenseProducts, required), Py_READONLY,
     "Pairs scored before the bound cut any."},
    {NULL, 0, 0, 0, NULL},
};

static PySequenceMethods DenseProducts_as_sequence = {.sq_length = (lenfunc)DenseProducts_length};

PyTypeObject DenseProductsType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseProducts",
    .tp_doc = doc_DenseProducts,
    .tp_basicsize = sizeof(DenseProducts),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, // Returned by `multiply`, never constructed by name.
    .tp_dealloc = (destructor)DenseProducts_dealloc,
    .tp_repr = (reprfunc)DenseProducts_repr,
    .tp_as_sequence = &DenseProducts_as_sequence,
    .tp_methods = DenseProducts_methods,
    .tp_members = DenseProducts_members,
};

#pragma endregion Arenas

#pragma region Paths and Library Probes

int usearch_py_path_from_object(PyObject* object, usearch_bytes_view_t* path, PyObject** decoded) {
    PyObject* bytes_object = NULL;
    if (!PyUnicode_FSConverter(object, &bytes_object)) return -1;
    path->data = (uint8_t const*)PyBytes_AS_STRING(bytes_object);
    path->length = (size_t)PyBytes_GET_SIZE(bytes_object);
    *decoded = bytes_object;
    return 0;
}

char const doc_version[] =              //
    "version() -> str\n"                //
    "\n"                                //
    "The library's semantic version.\n" //
    "Calls `usearch_version_name`.";

char const doc_capabilities_compiled[] =                                     //
    "capabilities_compiled() -> str\n"                                       //
    "\n"                                                                     //
    "The kernel families this build carries, whatever the machine offers.\n" //
    "Calls `usearch_capabilities_compiled`.";

char const doc_capabilities_available[] =                                           //
    "capabilities_available() -> str\n"                                             //
    "\n"                                                                            //
    "The kernel families this machine can actually run, and dispatch picks from.\n" //
    "Calls `usearch_capabilities_available`.";

PyObject* api_version(PyObject* self, PyObject* unused) { return PyUnicode_FromString(usearch_version_name()); }

PyObject* api_capabilities_compiled(PyObject* self, PyObject* unused) {
    return PyUnicode_FromString(usearch_capabilities_compiled());
}

PyObject* api_capabilities_available(PyObject* self, PyObject* unused) {
    return PyUnicode_FromString(usearch_capabilities_available());
}

#pragma endregion Paths and Library Probes
