/**
 *  @brief Vector search over equidimensional dense embeddings - the CPython twin of
 *      `include/usearch/dense_index.h`.
 *  @file python/dense_index.c
 *  @author Ash Vardanian
 *
 *  Every verb is a batch: a one-dimensional input is one row, so a memory-mapped dataset
 *  reaches the index with no reshape and no copy. The candidate-pool breadth rides the call
 *  rather than the handle, which lets a recall sweep run against one immutable index with no
 *  mutation between points.
 *
 *  Handles are shareable across threads and the library parallelizes internally, so every
 *  blocking verb drops the GIL around the call. The filtered search is the exception it
 *  cannot be: its predicate re-enters Python from the library's own worker threads, so the
 *  trampoline takes the GIL back for the duration of each callback.
 */
#include "lib.h"

#pragma region Handles

/** @brief An owned index handle, plus the geometry the verbs need without a round-trip. */
typedef struct DenseIndex {
    PyObject ob_base;
    usearch_dense_index_t handle;
} DenseIndex;

/** @brief What one cursor step yields. */
typedef enum dense_cursor_yields_t {
    /** @brief One @c (batch,16) matrix of identities. */
    dense_cursor_yields_keys_k = 0,
    /** @brief A @c (keys,vectors) pair of matrices. */
    dense_cursor_yields_keys_and_vectors_k = 1,
} dense_cursor_yields_t;

/** @brief A snapshot walk over the entries an index held when the cursor opened. */
typedef struct DenseCursor {
    PyObject ob_base;
    /** @brief The engine cursor, pinned to the watermark it opened at. */
    usearch_dense_cursor_t handle;
    /** @brief Keeps the index alive for the cursor's whole life. */
    PyObject* index;
    /** @brief Rows one step asks for. */
    size_t batch;
    /** @brief The index's frozen row width, cached at open. */
    size_t columns;
    /** @brief The scalar type vectors are exported as. */
    usearch_scalar_t scalar_kind;
    /** @brief Whether a step carries vectors beside the identities. */
    dense_cursor_yields_t yields;
} DenseCursor;

/** @brief A resumable best-first stream of matches for one query. */
typedef struct DenseFinder {
    PyObject ob_base;
    usearch_dense_finder_t handle;
    PyObject* index;
    usearch_key_t keys[64];
    usearch_distance_t distances[64];
    size_t buffered;
    size_t consumed;
    int drained;
} DenseFinder;

#pragma endregion Handles

#pragma region Option Names

/** @brief One spelling of a closed-set option and the value it selects. */
typedef struct choice_naming_t {
    char const* name;
    int value;
} choice_naming_t;

/**
 *  @brief The metric kinds, one name each.
 *
 *  No aliases: `l2` and `l2sq` would name one enumerator while making opposite claims about
 *  whether the distance is squared, and every other synonym only invites the same drift.
 */
static choice_naming_t const metric_namings[] = {
    {"inner", usearch_dense_metric_inner_k},         {"angular", usearch_dense_metric_angular_k},
    {"euclidean", usearch_dense_metric_euclidean_k}, {"hamming", usearch_dense_metric_hamming_k},
    {"jaccard", usearch_dense_metric_jaccard_k},
};

static choice_naming_t const residency_namings[] = {
    {"ram", usearch_residency_ram_k},
    {"vram", usearch_residency_vram_k},
    {"nvme", usearch_residency_nvme_k},
};

static choice_naming_t const pages_namings[] = {{"base", 0}, {"huge", 1}};
static choice_naming_t const addressing_namings[] = {{"device-only", 0}, {"unified", 1}};
static choice_naming_t const io_namings[] = {{"direct", 1}, {"page-cached", 0}};

static char const* metric_name(usearch_dense_metric_t metric) {
    for (size_t naming_index = 0; naming_index != usearch_py_table_size_(metric_namings); ++naming_index)
        if (metric_namings[naming_index].value == (int)metric) return metric_namings[naming_index].name;
    return "unknown";
}

/**
 *  @brief Reads one of a closed set of names into the value it selects.
 *
 *  The value travels in the table rather than being re-derived from the match position, so
 *  reordering a table cannot silently change what a name means.
 */
static int choice_from_object(PyObject* object, char const* field, choice_naming_t const* namings, size_t count,
                              int* value) {
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

static int size_from_object(PyObject* object, char const* field, size_t* value) {
    Py_ssize_t const signed_value = PyNumber_AsSsize_t(object, PyExc_OverflowError);
    if (signed_value == -1 && PyErr_Occurred()) return -1;
    if (signed_value < 0) {
        PyErr_Format(PyExc_ValueError, "The '%s' option must not be negative", field);
        return -1;
    }
    *value = (size_t)signed_value;
    return 0;
}

#pragma endregion Option Names

#pragma region Construction

static char const
    doc_DenseIndex[] = "DenseIndex(dimensions=0, metric='angular', quantization='bf16', *, " "connectivity=0,\n" "    " "pad_dimensions=" "False, " "residency=" "'ram', " "thr" "ead" "s=" "0, " "pages='base', " "device=None,\n" "    " "addres" "sing='" "device" "-only'" ", " "sector" "_bytes" "=0, " "queue_" "depth=" "0, " "io='" "direct" "')\n" "\n" "Vector search over equidimensional dense embeddings.\n" "\n" "Every verb is a batch, and one vector is a one-row matrix. The stored scalar type is\n" "the `quantization`; query rows may arrive in any other type and convert on the way in.\n" "\n" "Args:\n" "  dimensions (int, optional): Scalars per vector; 0 defers the geometry to `load`.\n" "  metric (str, optional): 'inner', 'angular', 'euclidean', 'hamming', or 'jaccard'.\n" "  quantization (str, optional): The stored scalar type. Defaults to 'bf16'.\n" "  connectivity (int, optional): Connections per node; 0 derives it from the geometry.\n" "  pad_dimensions (bool, optional): Round stored rows up to the SIMD step quantum.\n" "  residency (str, optional): 'ram', 'vram', or 'nvme'. Decides what `load` means.\n" "  threads (int, optional): RAM worker threads; 0 takes 7/8 of the logical cores.\n" "  pages (str, optional): 'base' or 'huge'; 'huge' fails loudly without a pool.\n" "  device (int, optional): VRAM device ordinal; None picks the first free device.\n" "  addressing (str, optional): 'device-only' keeps everything resident, 'unified' spills.\n" "  sector_bytes (int, optional): NVMe I/O granularity; 0 probes the device.\n" "  queue_depth (int, optional): NVMe submission ring depth; 0 derives it.\n" "  io (str, optional): 'direct' bypasses the page cache, 'page-cached' does not.\n" "\n" "Example:\n" "  >>> index = usearch.DenseIndex(dimensions=256, metric='angular', quantization='bf16')\n" "  >>> index.capabilities is not None\n" "  True";

/** @brief The constructor keywords, in the order @c dense_option_names spells them. */
typedef enum dense_option_t {
    dense_option_dimensions_k = 0,
    dense_option_metric_k,
    dense_option_quantization_k,
    dense_option_connectivity_k,
    dense_option_pad_dimensions_k,
    dense_option_residency_k,
    dense_option_threads_k,
    dense_option_pages_k,
    dense_option_device_k,
    dense_option_addressing_k,
    dense_option_sector_bytes_k,
    dense_option_queue_depth_k,
    dense_option_io_k,
    dense_option_unknown_k,
} dense_option_t;

static char const* const dense_option_names[] = {
    "dimensions", "metric", "quantization", "connectivity", "pad_dimensions", "residency", "threads",
    "pages",      "device", "addressing",   "sector_bytes", "queue_depth",    "io",
};

static dense_option_t dense_option_from_name(PyObject* name) {
    for (size_t option_index = 0; option_index != dense_option_unknown_k; ++option_index)
        if (PyUnicode_CompareWithASCIIString(name, dense_option_names[option_index]) == 0)
            return (dense_option_t)option_index;
    return dense_option_unknown_k;
}

/**
 *  @brief Fills @p options from the constructor's keywords, leaving unmentioned fields at
 *      whatever @p options already holds - which is how `metadata` round-trips through here.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int options_from_keywords(usearch_dense_options_t* options, PyObject* keywords) {
    if (!keywords) return 0;

    PyObject *name = NULL, *value = NULL;
    Py_ssize_t position = 0;
    while (PyDict_Next(keywords, &position, &name, &value)) {
        int choice = 0;
        size_t width = 0;
        switch (dense_option_from_name(name)) {
        case dense_option_dimensions_k:
            if (size_from_object(value, "dimensions", &options->dimensions) != 0) return -1;
            break;
        case dense_option_metric_k:
            if (choice_from_object(value, "metric", metric_namings, usearch_py_table_size_(metric_namings), &choice) !=
                0)
                return -1;
            options->metric = (usearch_dense_metric_t)choice;
            break;
        case dense_option_quantization_k:
            if (usearch_scalar_from_object(value, &options->quantization) != 0) return -1;
            break;
        case dense_option_connectivity_k:
            if (size_from_object(value, "connectivity", &options->connectivity) != 0) return -1;
            break;
        case dense_option_pad_dimensions_k: {
            int const truth = PyObject_IsTrue(value);
            if (truth < 0) return -1;
            options->pad_dimensions = truth != 0;
            break;
        }
        case dense_option_residency_k:
            if (choice_from_object(value, "residency", residency_namings, usearch_py_table_size_(residency_namings),
                                   &choice) != 0)
                return -1;
            options->residency = (usearch_residency_t)choice;
            break;
        case dense_option_threads_k:
            if (size_from_object(value, "threads", &options->ram.threads) != 0) return -1;
            break;
        case dense_option_pages_k:
            if (choice_from_object(value, "pages", pages_namings, usearch_py_table_size_(pages_namings), &choice) != 0)
                return -1;
            options->ram.huge_pages = choice != 0;
            break;
        case dense_option_device_k:
            if (value == Py_None) { options->vram.device = -1; }
            else {
                Py_ssize_t const ordinal = PyNumber_AsSsize_t(value, PyExc_OverflowError);
                if (ordinal == -1 && PyErr_Occurred()) return -1;
                options->vram.device = (int32_t)ordinal;
            }
            break;
        case dense_option_addressing_k:
            if (choice_from_object(value, "addressing", addressing_namings, usearch_py_table_size_(addressing_namings),
                                   &choice) != 0)
                return -1;
            options->vram.unified_addressing = choice != 0;
            break;
        case dense_option_sector_bytes_k:
            if (size_from_object(value, "sector_bytes", &width) != 0) return -1;
            options->nvme.sector_bytes = (uint32_t)width;
            break;
        case dense_option_queue_depth_k:
            if (size_from_object(value, "queue_depth", &width) != 0) return -1;
            options->nvme.queue_depth = (uint32_t)width;
            break;
        case dense_option_io_k:
            if (choice_from_object(value, "io", io_namings, usearch_py_table_size_(io_namings), &choice) != 0)
                return -1;
            options->nvme.direct_io = choice != 0;
            break;
        case dense_option_unknown_k:
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return -1;
        }
    }
    return 0;
}

/** @brief The options a bare `DenseIndex()` would build, before any keyword touches them. */
static void options_defaults(usearch_dense_options_t* options) {
    memset(options, 0, sizeof(*options));
    options->metric = usearch_dense_metric_angular_k;
    options->quantization = usearch_scalar_bf16_k;
    options->residency = usearch_residency_ram_k;
    options->nvme.direct_io = true;
}

static PyObject* DenseIndex_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    usearch_dense_options_t options;
    options_defaults(&options);

    // The three settings every caller names get to be positional; the rest are keywords,
    // because a call site reading `DenseIndex(256, 'angular', 'bf16', 0, False, 'ram')`
    // tells the reader nothing about the last three.
    Py_ssize_t const positional_args_count = PyTuple_GET_SIZE(args);
    if (positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "Only first 3 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }
    if (positional_args_count >= 1 &&
        size_from_object(PyTuple_GET_ITEM(args, 0), "dimensions", &options.dimensions) != 0)
        return NULL;
    if (positional_args_count >= 2) {
        int choice = 0;
        if (choice_from_object(PyTuple_GET_ITEM(args, 1), "metric", metric_namings,
                               usearch_py_table_size_(metric_namings), &choice) != 0)
            return NULL;
        options.metric = (usearch_dense_metric_t)choice;
    }
    if (positional_args_count >= 3 && usearch_scalar_from_object(PyTuple_GET_ITEM(args, 2), &options.quantization) != 0)
        return NULL;
    if (options_from_keywords(&options, keywords) != 0) return NULL;

    usearch_error_t error = {0};
    usearch_dense_index_t const handle = usearch_dense_init(&options, &error);
    if (usearch_raise_on_error(&error) != 0) return NULL;
    if (!handle) {
        usearch_raise(usearch_error_internal_k, "the library returned no handle and no error");
        return NULL;
    }

    DenseIndex* const instance = (DenseIndex*)type->tp_alloc(type, 0);
    if (!instance) {
        usearch_dense_free(handle);
        return NULL;
    }
    instance->handle = handle;
    return (PyObject*)instance;
}

static void DenseIndex_dealloc(DenseIndex* self) {
    if (self->handle) usearch_dense_free(self->handle);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** @brief Rejects a handle already consumed by `merge`. */
static int index_alive(DenseIndex const* self) {
    if (self->handle) return 0;
    usearch_raise(usearch_error_uninitialized_k, "the index handle was consumed by a merge");
    return -1;
}

#pragma endregion Construction

#pragma region Introspection

/** @brief The whole family of `(handle, error) → size_t` probes, through one call site. */
static PyObject* measure(DenseIndex* self, size_t (*probe)(usearch_dense_index_t, usearch_error_t*)) {
    if (index_alive(self) != 0) return NULL;
    usearch_error_t error = {0};
    size_t const measured = probe(self->handle, &error);
    if (usearch_raise_on_error(&error) != 0) return NULL;
    return PyLong_FromSize_t(measured);
}

static char const doc_DenseIndex_size[] = "Entries published so far, which `len` reports too.";

static PyObject* DenseIndex_get_size(DenseIndex* self, void* closure) { return measure(self, usearch_dense_size); }

static char const doc_DenseIndex_capacity[] = "Entries the arenas can hold before another reservation.";

static PyObject* DenseIndex_get_capacity(DenseIndex* self, void* closure) {
    return measure(self, usearch_dense_capacity);
}

static char const doc_DenseIndex_dimensions[] = "Scalars per vector; zero until the geometry is frozen.";

static PyObject* DenseIndex_get_dimensions(DenseIndex* self, void* closure) {
    return measure(self, usearch_dense_dimensions);
}

static char const doc_DenseIndex_connectivity[] = "Connections per node in the graph.";

static PyObject* DenseIndex_get_connectivity(DenseIndex* self, void* closure) {
    return measure(self, usearch_dense_connectivity);
}

static char const doc_DenseIndex_memory_usage[] = "Bytes the index occupies right now.";

static PyObject* DenseIndex_get_memory_usage(DenseIndex* self, void* closure) {
    return measure(self, usearch_dense_memory_usage);
}

static char const doc_DenseIndex_storage_requirements[] = "Bytes one monolith checkpoint of this index takes.";

static PyObject* DenseIndex_get_storage_requirements(DenseIndex* self, void* closure) {
    return measure(self, usearch_dense_storage_requirements);
}

static char const doc_DenseIndex_capabilities[] = "The kernel family dispatch picked for this geometry.";

static PyObject* DenseIndex_get_capabilities(DenseIndex* self, void* closure) {
    if (index_alive(self) != 0) return NULL;
    usearch_error_t error = {0};
    char const* const codename = usearch_dense_capabilities(self->handle, &error);
    if (usearch_raise_on_error(&error) != 0) return NULL;
    return PyUnicode_FromString(codename ? codename : "");
}

/**
 *  @brief The frozen row width, and the row's byte extent in @p scalar_kind .
 *
 *  A handle whose geometry is not frozen yet reports zero dimensions, which would size an
 *  arena at zero bytes and hand the engine the immortal empty-bytes singleton to write into -
 *  so the unfrozen case is a failure here rather than a zero downstream.
 *
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int index_row_extent(DenseIndex* self, usearch_scalar_t scalar_kind, size_t* columns, size_t* row_bytes) {
    usearch_error_t error = {0};
    size_t const dimensions = usearch_dense_dimensions(self->handle, &error);
    if (usearch_raise_on_error(&error) != 0) return -1;
    size_t const bytes = usearch_scalar_row_bytes(scalar_kind, dimensions);
    if (bytes == 0)
        return usearch_raise(usearch_error_uninitialized_k,
                             "the index geometry is not frozen yet; reserve or load first");
    *columns = dimensions;
    *row_bytes = bytes;
    return 0;
}

static Py_ssize_t DenseIndex_length(DenseIndex* self) {
    if (index_alive(self) != 0) return -1;
    usearch_error_t error = {0};
    size_t const size = usearch_dense_size(self->handle, &error);
    if (usearch_raise_on_error(&error) != 0) return -1;
    return (Py_ssize_t)size;
}

static PyObject* DenseIndex_repr(DenseIndex* self) {
    if (!self->handle) return PyUnicode_FromString("DenseIndex(<consumed>)");
    // One slot per probe, so a failing first call is not erased by a succeeding second. And no
    // `PyErr_Clear` on the way out: nothing here sets a Python exception, and `tp_repr` runs
    // inside `PyErr_Format("%R", ...)` chains where clearing would destroy the caller's.
    usearch_error_t size_error = {0}, dimensions_error = {0};
    size_t const size = usearch_dense_size(self->handle, &size_error);
    size_t const dimensions = usearch_dense_dimensions(self->handle, &dimensions_error);
    if (size_error.status != usearch_success_k || dimensions_error.status != usearch_success_k)
        return PyUnicode_FromString("DenseIndex(<unavailable>)");
    return PyUnicode_FromFormat("DenseIndex(dimensions=%zu, size=%zu)", dimensions, size);
}

#pragma endregion Introspection

#pragma region Ingest

static char const doc_DenseIndex_reserve[] =                                        //
    "reserve(capacity) -> None\n"                                                   //
    "\n"                                                                            //
    "Grow the arenas to hold at least `capacity` entries, freezing the geometry.\n" //
    "\n"                                                                            //
    "Args:\n"                                                                       //
    "  capacity (int): Entries the index must accept without another reservation.";

static PyObject* DenseIndex_reserve(PyObject* self, PyObject* capacity_object) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;
    size_t capacity = 0;
    if (size_from_object(capacity_object, "capacity", &capacity) != 0) return NULL;

    usearch_error_t error = {0};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_reserve(index->handle, capacity, &error);
    Py_END_ALLOW_THREADS;
    if (usearch_raise_on_error(&error) != 0) return NULL;
    Py_RETURN_NONE;
}

static char const doc_DenseIndex_clear[] = //
    "clear() -> None\n"                    //
    "\n"                                   //
    "Drop every entry, keeping the capacity and the frozen geometry.";

static PyObject* DenseIndex_clear(PyObject* self, PyObject* unused) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;
    usearch_error_t error = {0};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_clear(index->handle, &error);
    Py_END_ALLOW_THREADS;
    if (usearch_raise_on_error(&error) != 0) return NULL;
    Py_RETURN_NONE;
}

/**
 *  @brief Checks a fast-call's shape and reports how many keywords followed.
 *
 *  Every verb here fixes its positional count exactly, so the keyword walk that follows can
 *  index from it without a second bound.
 *
 *  @return The keyword count, or -1 with an exception set when the shape is wrong.
 */
static Py_ssize_t args_names_count_checked(char const* verb, Py_ssize_t positional_args_count,
                                           PyObject* args_names_tuple, Py_ssize_t required_positional) {
    if (positional_args_count != required_positional) {
        PyErr_Format(PyExc_TypeError, "%s() takes %zd positional arguments, received %zd", verb, required_positional,
                     positional_args_count);
        return -1;
    }
    return args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
}

static char const doc_DenseIndex_add[] =                                                  //
    "add(keys, vectors, *, expansion_factor=0, dtype=None, stats=None) -> int\n"          //
    "\n"                                                                                  //
    "Insert a batch of vectors under a batch of keys; one vector is a one-row matrix.\n"  //
    "A batch is all-or-nothing, so a rejected batch leaves the index untouched.\n"        //
    "Keys need not be unique: inserting one twice stores two entries, and `count`\n"      //
    "reports both.\n"                                                                     //
    "\n"                                                                                  //
    "Args:\n"                                                                             //
    "  keys (Sequence[Key | int] | buffer): One key per row, or an (N, 16) buffer.\n"     //
    "  vectors (buffer): An (N, dimensions) matrix; only the row pitch may stride.\n"     //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"         //
    "  dtype (str, optional): Names the input scalar type when the buffer cannot.\n"      //
    "  stats (Stats, optional): Accumulates traversal telemetry; None skips reporting.\n" //
    "Returns:\n"                                                                          //
    "  int: The number of rows inserted, equal to the number of keys.\n"                  //
    "\n"                                                                                  //
    "Example:\n"                                                                          //
    "  >>> from array import array\n"                                                     //
    "  >>> index = usearch.DenseIndex(dimensions=4)\n"                                    //
    "  >>> index.reserve(1)\n"                                                            //
    "  >>> index.add([1], array('d', [0.1, 0.2, 0.3, 0.4]))\n"                            //
    "  1";

static PyObject* DenseIndex_add(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    PyObject *stats_object = NULL, *dtype_object = NULL;
    size_t expansion_factor = 0;

    Py_ssize_t const args_names_count = args_names_count_checked("add", positional_args_count, args_names_tuple, 2);
    if (args_names_count < 0) return NULL;
    PyObject* const keys_object = args[0];
    PyObject* const vectors_object = args[1];

    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "expansion_factor") == 0) {
            if (size_from_object(value, "expansion_factor", &expansion_factor) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { dtype_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "stats") == 0) { stats_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    usearch_scalar_t scalar_hint = usearch_scalar_unknown_k;
    if (usearch_scalar_from_object(dtype_object, &scalar_hint) != 0) return NULL;

    usearch_matrix_view_t vectors;
    usearch_buffer_backing_t backing;
    if (usearch_matrix_from_object(vectors_object, usearch_matrix_access_read_k, scalar_hint, &vectors, &backing) != 0)
        return NULL;

    usearch_key_arena_t arena;
    if (usearch_keys_from_object(keys_object, vectors.rows, &arena) != 0) {
        usearch_matrix_release(&backing);
        return NULL;
    }

    usearch_dense_expansion_config_t const config = {.expansion_factor = expansion_factor};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_error_t error = {0};
    size_t inserted = 0;

    Py_BEGIN_ALLOW_THREADS;
    inserted = usearch_dense_add(index->handle, arena.keys, vectors.data, vectors.rows, vectors.pitch,
                                 vectors.scalar_kind, &config, &stats, &error);
    Py_END_ALLOW_THREADS;

    usearch_keys_release(&arena);
    usearch_matrix_release(&backing);
    if (usearch_raise_on_error(&error) != 0) return NULL;
    if (usearch_stats_accumulate(stats_object, &stats) != 0) return NULL;
    return PyLong_FromSize_t(inserted);
}

#pragma endregion Ingest

#pragma region Search

/** @brief What the two search doors agree on before either calls the library. */
typedef struct search_request_t {
    /** @brief The borrowed query matrix. */
    usearch_matrix_view_t queries;
    /** @brief Storage the query view points into. */
    usearch_buffer_backing_t queries_backing;
    /** @brief Matches wanted per query - the "k" in "kANN". */
    size_t wanted;
    /** @brief Candidate-pool breadth for this call; zero derives. */
    size_t expansion_factor;
    /** @brief The admission callable, or NULL for an unfiltered walk. */
    PyObject* predicate;
    /** @brief The @c Stats to accumulate into, or NULL to skip reporting. */
    PyObject* stats_object;
} search_request_t;

/** @brief What one filtered search hands its trampoline. */
typedef struct predicate_state_t {
    /** @brief The caller's admission callable. */
    PyObject* callable;
    /** @brief The first exception any callback raised; NULL while none has. */
    PyObject* raised;
} predicate_state_t;

/**
 *  @brief Re-enters Python for one admission decision, from whichever library thread traverses.
 *
 *  A worker's thread state dies with the callback and the caller's would surface the exception
 *  at an unrelated return, so a raising predicate leaves its failure in the state for
 *  @c search_dispatch to re-raise rather than on whichever thread caught it.
 */
static int predicate_trampoline(usearch_key_t key, void* state_pointer) {
    predicate_state_t* const state = (predicate_state_t*)state_pointer;
    PyGILState_STATE const gil_state = PyGILState_Ensure();
    int admitted = 0;
    PyObject* const key_object = usearch_key_to_object(key);
    if (key_object) {
        PyObject* const verdict = PyObject_CallOneArg(state->callable, key_object);
        Py_DECREF(key_object);
        if (verdict) {
            int const truth = PyObject_IsTrue(verdict);
            Py_DECREF(verdict);
            admitted = truth > 0;
        }
    }
    PyObject* const raised = PyErr_GetRaisedException(); //! NULL when nothing was raised.
    if (raised && !state->raised) state->raised = raised;
    else Py_XDECREF(raised);
    PyGILState_Release(gil_state);
    return admitted;
}

/** @brief Parses the arguments both `search` and `search_into` share, in the same order. */
static int search_request_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                                PyObject* args_names_tuple, Py_ssize_t required_positional, search_request_t* request) {
    memset(request, 0, sizeof(*request));

    Py_ssize_t const args_names_count = args_names_count_checked(verb, positional_args_count, args_names_tuple,
                                                                 required_positional);
    if (args_names_count < 0) return -1;

    PyObject* dtype_object = NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "expansion_factor") == 0) {
            if (size_from_object(value, "expansion_factor", &request->expansion_factor) != 0) return -1;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { dtype_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "predicate") == 0) {
            if (value != Py_None) {
                if (!PyCallable_Check(value)) {
                    PyErr_Format(PyExc_TypeError, "The 'predicate' must be callable, got %S", Py_TYPE(value));
                    return -1;
                }
                request->predicate = value;
            }
        }
        else if (PyUnicode_CompareWithASCIIString(name, "stats") == 0) { request->stats_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return -1;
        }
    }

    usearch_scalar_t scalar_hint = usearch_scalar_unknown_k;
    if (usearch_scalar_from_object(dtype_object, &scalar_hint) != 0) return -1;
    if (usearch_matrix_from_object(args[0], usearch_matrix_access_read_k, scalar_hint, &request->queries,
                                   &request->queries_backing) != 0)
        return -1;
    if (size_from_object(args[1], "wanted", &request->wanted) != 0) {
        usearch_matrix_release(&request->queries_backing);
        return -1;
    }
    if (request->wanted == 0) {
        usearch_matrix_release(&request->queries_backing);
        usearch_raise(usearch_error_shape_mismatch_k, "a search must want at least one match");
        return -1;
    }
    return 0;
}

/** @brief The one call into the library both doors make, with the outputs already sited. */
static int search_dispatch(DenseIndex* index, search_request_t const* request, usearch_key_t* keys,
                           usearch_distance_t* distances, size_t* counts, usearch_dense_traversal_stats_t* stats) {
    usearch_dense_expansion_config_t const config = {.expansion_factor = request->expansion_factor};
    usearch_error_t error = {0};
    predicate_state_t predicate_state = {request->predicate, NULL};
    size_t const keys_pitch = request->wanted * sizeof(usearch_key_t);
    size_t const distances_pitch = request->wanted * sizeof(usearch_distance_t);

    Py_BEGIN_ALLOW_THREADS;
    if (request->predicate)
        usearch_dense_search_if(index->handle, request->queries.data, request->queries.rows, request->queries.pitch,
                                request->queries.scalar_kind, request->wanted, &config, stats, predicate_trampoline,
                                &predicate_state, keys, keys_pitch, distances, distances_pitch, counts, &error);
    else
        usearch_dense_search(index->handle, request->queries.data, request->queries.rows, request->queries.pitch,
                             request->queries.scalar_kind, request->wanted, &config, stats, keys, keys_pitch, distances,
                             distances_pitch, counts, &error);
    Py_END_ALLOW_THREADS;

    // A predicate that raised outranks whatever the walk went on to report, since every
    // decision after the first failure was made on a guess.
    if (predicate_state.raised) {
        PyErr_SetRaisedException(predicate_state.raised);
        return -1;
    }
    return usearch_raise_on_error(&error);
}

static char const doc_DenseIndex_search[] =                                                   //
    "search(queries, wanted, *, expansion_factor=0, dtype=None, predicate=None) -> Matches\n" //
    "\n"                                                                                      //
    "Find the `wanted` nearest entries for every query row; one query is a one-row matrix.\n" //
    "Never writes to the graph, so any number of searches may overlap with insertions.\n"     //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  queries (buffer): A (Q, dimensions) matrix of query rows.\n"                           //
    "  wanted (int): Upper bound on matches per query - the 'k' in 'kANN'.\n"                 //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"             //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.\n"          //
    "  predicate (callable, optional): Admits a Key when it returns truthy. Non-matching\n"   //
    "    entries still route the traversal, so recall survives low selectivity. The\n"        //
    "    callback re-enters Python, so it runs under the GIL and serializes the walk.\n"      //
    "Returns:\n"                                                                              //
    "  Matches: Zero-copy views over the keys, distances, and per-query counts.\n"            //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> from array import array\n"                                                         //
    "  >>> index = usearch.DenseIndex(dimensions=4)\n"                                        //
    "  >>> index.reserve(1)\n"                                                                //
    "  >>> query = array('d', [0.1, 0.2, 0.3, 0.4])\n"                                        //
    "  >>> _ = index.add([7], query)\n"                                                       //
    "  >>> int(index.search(query, 1)[0][0].key)\n"                                           //
    "  7";

static PyObject* DenseIndex_search(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    search_request_t request;
    if (search_request_parse("search", args, positional_args_count, args_names_tuple, 2, &request) != 0) return NULL;

    Py_ssize_t keys_bytes = 0, distances_bytes = 0, counts_bytes = 0;
    if (usearch_matrix_bytes(request.queries.rows, request.wanted, sizeof(usearch_key_t), &keys_bytes) != 0 ||
        usearch_matrix_bytes(request.queries.rows, request.wanted, sizeof(usearch_distance_t), &distances_bytes) != 0 ||
        usearch_matrix_bytes(request.queries.rows, 1, sizeof(size_t), &counts_bytes) != 0) {
        usearch_matrix_release(&request.queries_backing);
        return NULL;
    }

    Matches* const matches = PyObject_New(Matches, &MatchesType);
    if (!matches) {
        usearch_matrix_release(&request.queries_backing);
        return NULL;
    }
    // `PyObject_New` hands back uninitialized storage, and the deallocator walks all three
    // arena slots, so they must be null before the first thing that can fail.
    matches->keys_arena = matches->distances_arena = matches->counts_arena = NULL;
    matches->queries = request.queries.rows;
    matches->wanted = request.wanted;
    memset(&matches->stats, 0, sizeof(matches->stats));
    // The arenas are filled before the object is ever visible, so handing out `memoryview`
    // objects over immutable `bytes` stays honest.
    matches->keys_arena = PyBytes_FromStringAndSize(NULL, keys_bytes);
    matches->distances_arena = PyBytes_FromStringAndSize(NULL, distances_bytes);
    matches->counts_arena = PyBytes_FromStringAndSize(NULL, counts_bytes);
    if (!matches->keys_arena || !matches->distances_arena || !matches->counts_arena) {
        usearch_matrix_release(&request.queries_backing);
        Py_DECREF(matches);
        return NULL;
    }
    // `PyBytes_FromStringAndSize(NULL, n)` hands back uninitialized heap, and the engine fills
    // only `counts[query]` cells per row. Without this, everything past a count is whatever a
    // freed object left there - and reading the whole matrix is the obvious thing to do.
    memset(PyBytes_AS_STRING(matches->keys_arena), 0, (size_t)keys_bytes);
    memset(PyBytes_AS_STRING(matches->distances_arena), 0, (size_t)distances_bytes);
    memset(PyBytes_AS_STRING(matches->counts_arena), 0, (size_t)counts_bytes);

    int const failed = search_dispatch(index, &request, (usearch_key_t*)PyBytes_AS_STRING(matches->keys_arena),
                                       (usearch_distance_t*)PyBytes_AS_STRING(matches->distances_arena),
                                       (size_t*)PyBytes_AS_STRING(matches->counts_arena), &matches->stats);
    usearch_matrix_release(&request.queries_backing);
    if (failed != 0) {
        Py_DECREF(matches);
        return NULL;
    }
    if (usearch_stats_accumulate(request.stats_object, &matches->stats) != 0) {
        Py_DECREF(matches);
        return NULL;
    }
    return (PyObject*)matches;
}

static char const doc_DenseIndex_search_into[] =                                                  //
    "search_into(queries, wanted, keys, distances, counts, *, expansion_factor=0, ...) -> None\n" //
    "\n"                                                                                          //
    "The non-allocating door onto `search`: writes into buffers the caller already owns.\n"       //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  queries (buffer): A (Q, dimensions) matrix of query rows.\n"                               //
    "  wanted (int): Upper bound on matches per query.\n"                                         //
    "  keys (buffer): A writable (Q, wanted, 16) buffer of identity bytes.\n"                     //
    "  distances (buffer): A writable (Q, wanted) buffer of float64 distances.\n"                 //
    "  counts (buffer): A writable (Q,) buffer of unsigned word-sized counts.\n"                  //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"                 //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.\n"              //
    "  predicate (callable, optional): Admits a Key when it returns truthy.\n"                    //
    "  stats (Stats, optional): Accumulates traversal telemetry.\n"                               //
    "\n"                                                                                          //
    "Example:\n"                                                                                  //
    "  >>> from array import array\n"                                                             //
    "  >>> index = usearch.DenseIndex(dimensions=4)\n"                                            //
    "  >>> index.reserve(1)\n"                                                                    //
    "  >>> query = array('d', [0.1, 0.2, 0.3, 0.4])\n"                                            //
    "  >>> _ = index.add([7], query)\n"                                                           //
    "  >>> keys, distances, counts = bytearray(16), bytearray(8), bytearray(8)\n"                 //
    "  >>> index.search_into(query, 1, keys, distances, counts)\n"                                //
    "  >>> int.from_bytes(counts, 'little')\n"                                                    //
    "  1";

/** @brief One output a batch verb writes into, and the extent it must have. */
typedef struct output_naming_t {
    char const* field;
    Py_ssize_t bytes;
} output_naming_t;

static PyObject* DenseIndex_search_into(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                        PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    search_request_t request;
    if (search_request_parse("search_into", args, positional_args_count, args_names_tuple, 5, &request) != 0)
        return NULL;

    output_naming_t output_namings[3] = {{"keys", 0}, {"distances", 0}, {"counts", 0}};
    if (usearch_matrix_bytes(request.queries.rows, request.wanted, sizeof(usearch_key_t), &output_namings[0].bytes) !=
            0 ||
        usearch_matrix_bytes(request.queries.rows, request.wanted, sizeof(usearch_distance_t),
                             &output_namings[1].bytes) != 0 ||
        usearch_matrix_bytes(request.queries.rows, 1, sizeof(size_t), &output_namings[2].bytes) != 0) {
        usearch_matrix_release(&request.queries_backing);
        return NULL;
    }

    // One acquisition loop and one release loop, so the unwind order lives in a single place
    // rather than being hand-maintained once per output.
    usearch_buffer_backing_t backings[3];
    void* pointers[3] = {NULL, NULL, NULL};
    usearch_dense_traversal_stats_t stats = {0};
    PyObject* result = NULL;
    size_t borrowed = 0;

    for (; borrowed != 3; ++borrowed)
        if (usearch_buffer_borrow_exact(args[2 + borrowed], output_namings[borrowed].field,
                                        usearch_matrix_access_write_k, output_namings[borrowed].bytes,
                                        &backings[borrowed], &pointers[borrowed]) != 0)
            goto release;

    if (search_dispatch(index, &request, (usearch_key_t*)pointers[0], (usearch_distance_t*)pointers[1],
                        (size_t*)pointers[2], &stats) != 0)
        goto release;
    if (usearch_stats_accumulate(request.stats_object, &stats) != 0) goto release;
    result = Py_NewRef(Py_None);

release:
    while (borrowed--) usearch_matrix_release(&backings[borrowed]);
    usearch_matrix_release(&request.queries_backing);
    return result;
}

#pragma endregion Search

#pragma region Retrieval

static char const doc_DenseIndex_count[] =                             //
    "count(key) -> int\n"                                              //
    "\n"                                                               //
    "How many entries share `key`; zero means the index holds none.\n" //
    "\n"                                                               //
    "Example:\n"                                                       //
    "  >>> from array import array\n"                                  //
    "  >>> index = usearch.DenseIndex(dimensions=2)\n"                 //
    "  >>> index.reserve(2)\n"                                         //
    "  >>> row = array('d', [0.1, 0.2])\n"                             //
    "  >>> _ = index.add([7], row)\n"                                  //
    "  >>> _ = index.add([7], row)\n"                                  //
    "  >>> index.count(7)\n"                                           //
    "  2";

static PyObject* DenseIndex_count(PyObject* self, PyObject* key_object) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;
    usearch_key_t key;
    if (usearch_key_from_object(key_object, &key) != 0) return NULL;
    usearch_error_t error = {0};
    size_t const found = usearch_dense_count(index->handle, key, &error);
    if (usearch_raise_on_error(&error) != 0) return NULL;
    return PyLong_FromSize_t(found);
}

static int DenseIndex_contains(PyObject* self, PyObject* key_object) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return -1;
    usearch_key_t key;
    if (usearch_key_from_object(key_object, &key) != 0) return -1;
    usearch_error_t error = {0};
    size_t const found = usearch_dense_count(index->handle, key, &error);
    if (usearch_raise_on_error(&error) != 0) return -1;
    return found != 0;
}

static char const doc_DenseIndex_get[] =                                                     //
    "get(key, *, dtype='f32') -> memoryview | None\n"                                        //
    "\n"                                                                                     //
    "Every vector stored under `key`, in insertion order, or None when the key is absent.\n" //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  key (Key | int | str): The identity to retrieve.\n"                                   //
    "  dtype (str, optional): The scalar type to export as; converted from storage.\n"       //
    "Returns:\n"                                                                             //
    "  memoryview: A (count, dimensions) view for the kinds `memoryview` can format -\n"     //
    "    f64, f32, f16, i8, u8. Sub-byte and mini-float kinds, `bf16` among them, leave\n"   //
    "    as a (count, row_bytes) run of 'B' instead.\n"                                      //
    "  None: When the index holds no entry under `key`.";

static PyObject* DenseIndex_get(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    Py_ssize_t const args_names_count = args_names_count_checked("get", positional_args_count, args_names_tuple, 1);
    if (args_names_count < 0) return NULL;
    PyObject* dtype_object = NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { dtype_object = args[args_progress]; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    usearch_scalar_t scalar_kind = usearch_scalar_f32_k;
    if (dtype_object && usearch_scalar_from_object(dtype_object, &scalar_kind) != 0) return NULL;

    usearch_key_t key;
    if (usearch_key_from_object(args[0], &key) != 0) return NULL;

    usearch_error_t error = {0};
    size_t const found = usearch_dense_count(index->handle, key, &error);
    if (usearch_raise_on_error(&error) != 0) return NULL;
    if (found == 0) Py_RETURN_NONE;

    size_t columns = 0, row_bytes = 0;
    if (index_row_extent(index, scalar_kind, &columns, &row_bytes) != 0) return NULL;

    Py_ssize_t arena_bytes = 0;
    if (usearch_matrix_bytes(found, row_bytes, 1, &arena_bytes) != 0) return NULL;
    PyObject* const arena = PyBytes_FromStringAndSize(NULL, arena_bytes);
    if (!arena) return NULL;
    memset(PyBytes_AS_STRING(arena), 0, (size_t)arena_bytes);

    size_t written = 0;
    Py_BEGIN_ALLOW_THREADS;
    written = usearch_dense_get(index->handle, key, PyBytes_AS_STRING(arena), found, row_bytes, scalar_kind, &error);
    Py_END_ALLOW_THREADS;
    if (usearch_raise_on_error(&error) != 0) {
        Py_DECREF(arena);
        return NULL;
    }
    if (written > found) {
        // The count and the copy are not atomic, but harmlessly so: entries are only ever
        // added, so a stale count under-reports and the caller simply asks again.
        Py_DECREF(arena);
        usearch_raise(usearch_error_contended_k, "entries were added under this key mid-copy; retry the read");
        return NULL;
    }

    // Shaped by what was written rather than what was counted, so a short copy never publishes
    // rows the engine did not fill.
    PyObject* const shaped = usearch_arena_matrix(arena, written, columns, scalar_kind);
    Py_DECREF(arena);
    return shaped;
}

static PyObject* DenseIndex_subscript(PyObject* self, PyObject* key_object) {
    PyObject* const args[1] = {key_object};
    PyObject* const vectors = DenseIndex_get(self, args, 1, NULL);
    if (!vectors || vectors != Py_None) return vectors;
    // `get` answers "nothing here" with None because it is a `get`; subscription must raise,
    // or `in`, `[]`, and `.get()` mean three different things to the same reader.
    Py_DECREF(vectors);
    PyErr_SetObject(PyExc_KeyError, key_object);
    return NULL;
}

#pragma endregion Retrieval

#pragma region Linear Algebra

static char const doc_DenseIndex_multiply_into[] =                                            //
    "multiply_into(rows, wanted, columns, products, counts, *, expansion_factor=0,\n"         //
    "    dtype=None, stats=None) -> int\n"                                                    //
    "\n"                                                                                      //
    "Approximate sparse product of a matrix against the indexed rows - the linear algebra\n"  //
    "sibling of `search`, reporting every pair the traversal scored rather than only the\n"   //
    "nearest. Pairs arrive in traversal order rather than sorted, and carry no duplicates.\n" //
    "\n"                                                                                      //
    "Into-only: the width of the emitted pattern is the caller's to choose, so there is no\n" //
    "allocating door to choose it for them.\n"                                                //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  rows (buffer): An (R, dimensions) matrix of left-hand rows.\n"                         //
    "  wanted (int): Upper bound on pairs emitted per row - the width of the outputs.\n"      //
    "  columns (buffer): A writable (R, wanted, 16) buffer of indexed-row identities.\n"      //
    "  products (buffer): A writable (R, wanted) buffer of float64 products.\n"               //
    "  counts (buffer): A writable (R,) buffer of unsigned word-sized counts. Each reports\n" //
    "    the TRUE pair count, which may exceed `wanted` - that row was truncated to its\n"    //
    "    nearest pairs. Unlike `search_into`, a count is therefore not a safe slice bound.\n" //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it. A wider\n"     //
    "    factor widens the emitted pattern, being what the walk's breadth decides.\n"         //
    "  dtype (str, optional): Names the left-hand scalar type when the buffer cannot.\n"      //
    "  stats (Stats, optional): Accumulates traversal telemetry; None skips reporting.\n"     //
    "Returns:\n"                                                                              //
    "  int: How many rows truncated; zero means every pair the walk scored was written.\n"    //
    "    This is the only `_into` verb with a return, because it is the only one with\n"      //
    "    something to report - the retry loop reads as `while index.multiply_into(...)`.";

static PyObject* DenseIndex_multiply_into(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                          PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    // Structurally `search_into` minus the predicate: `multiply` has no `_if` sibling, so a
    // `predicate=` keyword must be refused rather than quietly ignored.
    search_request_t request;
    if (search_request_parse("multiply_into", args, positional_args_count, args_names_tuple, 5, &request) != 0)
        return NULL;
    if (request.predicate) {
        usearch_matrix_release(&request.queries_backing);
        PyErr_SetString(PyExc_TypeError, "Got unexpected keyword argument: predicate");
        return NULL;
    }

    output_naming_t output_namings[3] = {{"columns", 0}, {"products", 0}, {"counts", 0}};
    if (usearch_matrix_bytes(request.queries.rows, request.wanted, sizeof(usearch_key_t), &output_namings[0].bytes) !=
            0 ||
        usearch_matrix_bytes(request.queries.rows, request.wanted, sizeof(usearch_distance_t),
                             &output_namings[1].bytes) != 0 ||
        usearch_matrix_bytes(request.queries.rows, 1, sizeof(size_t), &output_namings[2].bytes) != 0) {
        usearch_matrix_release(&request.queries_backing);
        return NULL;
    }

    usearch_buffer_backing_t backings[3];
    void* pointers[3] = {NULL, NULL, NULL};
    usearch_dense_expansion_config_t const config = {.expansion_factor = request.expansion_factor};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_error_t error = {0};
    PyObject* result = NULL;
    size_t borrowed = 0;

    for (; borrowed != 3; ++borrowed)
        if (usearch_buffer_borrow_exact(args[2 + borrowed], output_namings[borrowed].field,
                                        usearch_matrix_access_write_k, output_namings[borrowed].bytes,
                                        &backings[borrowed], &pointers[borrowed]) != 0)
            goto release;

    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_multiply(index->handle, request.queries.data, request.queries.rows, request.queries.pitch,
                           request.queries.scalar_kind, request.wanted, &config, &stats, (usearch_key_t*)pointers[0],
                           request.wanted * sizeof(usearch_key_t), (usearch_distance_t*)pointers[1],
                           request.wanted * sizeof(usearch_distance_t), (size_t*)pointers[2], &error);
    Py_END_ALLOW_THREADS;

    if (usearch_raise_on_error(&error) != 0) goto release;
    if (usearch_stats_accumulate(request.stats_object, &stats) != 0) goto release;

    // One pass over a buffer already in cache, so the caller need not make a second.
    size_t truncated = 0;
    size_t const* const counts = (size_t const*)pointers[2];
    for (size_t row = 0; row != request.queries.rows; ++row) truncated += counts[row] > request.wanted;
    result = PyLong_FromSize_t(truncated);

release:
    while (borrowed--) usearch_matrix_release(&backings[borrowed]);
    usearch_matrix_release(&request.queries_backing);
    return result;
}

static char const doc_DenseIndex_attend_into[] =                                                 //
    "attend_into(queries, values, outputs, *, expansion_factor=0, score_scale=0.0,\n"            //
    "    mass_tolerance=0.0, dtype=None, values_dtype=None, outputs_dtype=None,\n"               //
    "    maximum_scores=None, partitions=None, stats=None) -> None\n"                            //
    "\n"                                                                                         //
    "Softmax-weighted attention over the indexed keys, folded into the graph walk in one\n"      //
    "pass - no match list is materialized, and each value row is gathered exactly once, at\n"    //
    "the moment its key was scored. Never writes to the graph, so a growing context may\n"       //
    "stream through this while insertions continue.\n"                                           //
    "\n"                                                                                         //
    "The index holds the keys; `values` holds the value rows, addressed by key - an entry's\n"   //
    "key is its row offset into `values`. Token positions already are such keys, so a\n"         //
    "context inserted in position order needs no mapping, and the publication watermark\n"       //
    "becomes the causal mask.\n"                                                                 //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  queries (buffer): A (Q, dimensions) matrix; every query head sharing these keys is\n"     //
    "    one more row here.\n"                                                                   //
    "  values (buffer): A (V, value_dimensions) matrix, addressed by key. Its width is the\n"    //
    "    value dimensionality, independent of the key width the index froze.\n"                  //
    "  outputs (buffer): A writable (Q, value_dimensions) buffer, already divided by its\n"      //
    "    partitions. Must hold exactly the bytes its shape and `outputs_dtype` imply.\n"         //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"                //
    "  score_scale (float, optional): The logit scale, conventionally the inverse square\n"      //
    "    root of the head dimensionality; 0.0 derives it from the frozen geometry.\n"            //
    "  mass_tolerance (float, optional): The residual-mass threshold the walk stops at, the\n"   //
    "    mass-denominated sibling of a search's radius. 0.0 derives it, and does NOT mean\n"     //
    "    zero residual mass.\n"                                                                  //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.\n"             //
    "  values_dtype (str, optional): The same, for the value rows.\n"                            //
    "  outputs_dtype (str, optional): The scalar type to write `outputs` as.\n"                  //
    "  maximum_scores (buffer, optional): A writable (Q,) buffer of float32 running maxima.\n"   //
    "  partitions (buffer, optional): A writable (Q,) buffer of float32 softmax denominators.\n" //
    "    Pass both or neither: together the pair recombines this branch with another over\n"     //
    "    the same queries - a dense recent window, or a second shard of one context - by\n"      //
    "    partial-softmax rescaling, which is the only use either one has alone.\n"               //
    "  stats (Stats, optional): Accumulates traversal telemetry; None skips reporting.";

/** @brief What one attention call needs, gathered before anything is borrowed. */
typedef struct attend_request_t {
    /** @brief Candidate-pool breadth, logit scale, and the residual-mass stopping threshold. */
    usearch_dense_attention_config_t config;
    /** @brief The scalar kind of the query rows, or unknown to infer it. */
    usearch_scalar_t queries_kind;
    /** @brief The scalar kind of the value rows, or unknown to infer it. */
    usearch_scalar_t values_kind;
    /** @brief The scalar kind to write outputs as, or unknown to infer it. */
    usearch_scalar_t outputs_kind;
    /** @brief The optional running-maxima output, or NULL. */
    PyObject* maximum_scores;
    /** @brief The optional partition output, or NULL. */
    PyObject* partitions;
    /** @brief The `Stats` to accumulate into, or NULL. */
    PyObject* stats_object;
} attend_request_t;

static int attend_request_parse(PyObject* const* args, Py_ssize_t positional_args_count, PyObject* args_names_tuple,
                                attend_request_t* request) {
    memset(request, 0, sizeof(*request));
    Py_ssize_t const args_names_count = args_names_count_checked("attend_into", positional_args_count, args_names_tuple,
                                                                 3);
    if (args_names_count < 0) return -1;

    PyObject *queries_dtype = NULL, *values_dtype = NULL, *outputs_dtype = NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "expansion_factor") == 0) {
            if (size_from_object(value, "expansion_factor", &request->config.expansion_factor) != 0) return -1;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "score_scale") == 0) {
            double const scale = PyFloat_AsDouble(value);
            if (scale == -1.0 && PyErr_Occurred()) return -1;
            request->config.score_scale = (float)scale;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "mass_tolerance") == 0) {
            double const tolerance = PyFloat_AsDouble(value);
            if (tolerance == -1.0 && PyErr_Occurred()) return -1;
            request->config.mass_tolerance = (float)tolerance;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { queries_dtype = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "values_dtype") == 0) { values_dtype = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "outputs_dtype") == 0) { outputs_dtype = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "maximum_scores") == 0) {
            request->maximum_scores = value == Py_None ? NULL : value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "partitions") == 0) {
            request->partitions = value == Py_None ? NULL : value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "stats") == 0) { request->stats_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return -1;
        }
    }

    if (usearch_scalar_from_object(queries_dtype, &request->queries_kind) != 0) return -1;
    if (usearch_scalar_from_object(values_dtype, &request->values_kind) != 0) return -1;
    if (usearch_scalar_from_object(outputs_dtype, &request->outputs_kind) != 0) return -1;

    // The C surface permits one without the other, but a maximum without its partition is a
    // number nothing can rescale - so the narrowing here deletes a silent-wrong-result path.
    if ((request->maximum_scores == NULL) != (request->partitions == NULL))
        return usearch_raise(usearch_error_shape_mismatch_k,
                             "pass `maximum_scores` and `partitions` together or neither");
    return 0;
}

static PyObject* DenseIndex_attend_into(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                        PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    attend_request_t request;
    if (attend_request_parse(args, positional_args_count, args_names_tuple, &request) != 0) return NULL;

    usearch_matrix_view_t queries, values;
    usearch_buffer_backing_t queries_backing, values_backing, outputs_backing;
    usearch_buffer_backing_t maxima_backing, partitions_backing;
    void *outputs = NULL, *maxima = NULL, *partitions = NULL;
    usearch_dense_traversal_stats_t stats = {0};
    usearch_error_t error = {0};
    PyObject* result = NULL;
    int borrowed_values = 0, borrowed_outputs = 0, borrowed_maxima = 0, borrowed_partitions = 0;

    if (usearch_matrix_from_object(args[0], usearch_matrix_access_read_k, request.queries_kind, &queries,
                                   &queries_backing) != 0)
        return NULL;
    if (usearch_matrix_from_object(args[1], usearch_matrix_access_read_k, request.values_kind, &values,
                                   &values_backing) != 0)
        goto release;
    borrowed_values = 1;

    // The C surface takes `value_dimensions` explicitly because C has no shapes; a rank-two
    // buffer carries its own width, which is a stronger contract than the ABI can express.
    if (values_backing.buffer.ndim != 2) {
        usearch_raise(usearch_error_shape_mismatch_k, "attention values must be a (rows, value_dimensions) matrix");
        goto release;
    }

    usearch_scalar_t const outputs_kind = request.outputs_kind == usearch_scalar_unknown_k ? values.scalar_kind
                                                                                           : request.outputs_kind;
    Py_ssize_t outputs_bytes = 0;
    if (usearch_matrix_bytes(queries.rows, usearch_scalar_row_bytes(outputs_kind, values.columns), 1, &outputs_bytes) !=
        0)
        goto release;
    if (usearch_buffer_borrow_exact(args[2], "outputs", usearch_matrix_access_write_k, outputs_bytes, &outputs_backing,
                                    &outputs) != 0)
        goto release;
    borrowed_outputs = 1;

    if (request.maximum_scores) {
        Py_ssize_t running_bytes = 0;
        if (usearch_matrix_bytes(queries.rows, sizeof(float), 1, &running_bytes) != 0) goto release;
        if (usearch_buffer_borrow_exact(request.maximum_scores, "maximum_scores", usearch_matrix_access_write_k,
                                        running_bytes, &maxima_backing, &maxima) != 0)
            goto release;
        borrowed_maxima = 1;
        if (usearch_buffer_borrow_exact(request.partitions, "partitions", usearch_matrix_access_write_k, running_bytes,
                                        &partitions_backing, &partitions) != 0)
            goto release;
        borrowed_partitions = 1;
    }

    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_attend(index->handle, queries.data, queries.rows, queries.pitch, queries.scalar_kind, values.data,
                         values.columns, values.pitch, values.scalar_kind, &request.config, &stats, outputs,
                         usearch_scalar_row_bytes(outputs_kind, values.columns), outputs_kind, (float*)maxima,
                         (float*)partitions, &error);
    Py_END_ALLOW_THREADS;

    if (usearch_raise_on_error(&error) != 0) goto release;
    if (usearch_stats_accumulate(request.stats_object, &stats) != 0) goto release;
    result = Py_NewRef(Py_None);

release:
    if (borrowed_partitions) usearch_matrix_release(&partitions_backing);
    if (borrowed_maxima) usearch_matrix_release(&maxima_backing);
    if (borrowed_outputs) usearch_matrix_release(&outputs_backing);
    if (borrowed_values) usearch_matrix_release(&values_backing);
    usearch_matrix_release(&queries_backing);
    return result;
}

#pragma endregion Linear Algebra

#pragma region Composition

static char const doc_DenseIndex_merge[] =                                                  //
    "merge(donor) -> None\n"                                                                //
    "\n"                                                                                    //
    "Absorb `donor` into this index, wiring cross-links via landmark stitching - cheaper\n" //
    "than re-inserting its vectors one by one. The donor handle is consumed and every\n"    //
    "verb on it afterwards raises. Readers may keep searching throughout.\n"                //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  donor (DenseIndex): The index to absorb.";

static PyObject* DenseIndex_merge(PyObject* self, PyObject* donor_object) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;
    if (!PyObject_TypeCheck(donor_object, &DenseIndexType)) {
        PyErr_Format(PyExc_TypeError, "Expected a DenseIndex donor, got %S", Py_TYPE(donor_object));
        return NULL;
    }
    DenseIndex* const donor = (DenseIndex*)donor_object;
    if (donor == index) {
        usearch_raise(usearch_error_malformed_input_k, "an index cannot merge into itself");
        return NULL;
    }
    if (index_alive(donor) != 0) return NULL;

    usearch_dense_index_t const donated = donor->handle;
    // The library consumes the donor, so surrender our claim before the call rather than
    // after: a failure that still consumed it must not leave us holding a second owner.
    donor->handle = NULL;

    usearch_error_t error = {0};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_merge(index->handle, donated, &error);
    Py_END_ALLOW_THREADS;
    if (usearch_raise_on_error(&error) != 0) return NULL;
    Py_RETURN_NONE;
}

#pragma endregion Composition

#pragma region Iteration

static void DenseCursor_dealloc(DenseCursor* self) {
    if (self->handle) usearch_dense_cursor_free(self->handle);
    Py_XDECREF(self->index);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* DenseCursor_iter(PyObject* self) {
    Py_INCREF(self);
    return self;
}

static PyObject* DenseCursor_next(DenseCursor* self) {
    if (!self->handle) return NULL;
    int const with_vectors = self->yields == dense_cursor_yields_keys_and_vectors_k;
    size_t const row_bytes = with_vectors ? usearch_scalar_row_bytes(self->scalar_kind, self->columns) : 0;

    Py_ssize_t keys_bytes = 0, vectors_bytes = 0;
    if (usearch_matrix_bytes(self->batch, 1, sizeof(usearch_key_t), &keys_bytes) != 0) return NULL;
    if (with_vectors && usearch_matrix_bytes(self->batch, row_bytes, 1, &vectors_bytes) != 0) return NULL;

    PyObject* keys_arena = PyBytes_FromStringAndSize(NULL, keys_bytes);
    if (!keys_arena) return NULL;
    PyObject* vectors_arena = NULL;
    if (with_vectors) {
        vectors_arena = PyBytes_FromStringAndSize(NULL, vectors_bytes);
        if (!vectors_arena) {
            Py_DECREF(keys_arena);
            return NULL;
        }
    }

    usearch_error_t error = {0};
    size_t written = 0;
    void* const vectors = vectors_arena ? PyBytes_AS_STRING(vectors_arena) : NULL;
    Py_BEGIN_ALLOW_THREADS;
    written = usearch_dense_cursor_next(self->handle, (usearch_key_t*)PyBytes_AS_STRING(keys_arena), vectors,
                                        self->batch, row_bytes, self->scalar_kind, &error);
    Py_END_ALLOW_THREADS;

    PyObject* result = NULL;
    PyObject* keys_shaped = NULL;
    if (usearch_raise_on_error(&error) != 0) goto release;
    if (written == 0) goto release; //! A drained snapshot ends the iteration, with no exception.

    // The views are cast to what the snapshot yielded rather than to what was reserved, so the
    // arenas keep their slack and no reader ever sees it - no resize needed.
    keys_shaped = usearch_arena_matrix(keys_arena, written, sizeof(usearch_key_t), usearch_scalar_u8_k);
    if (!keys_shaped) goto release;
    if (!vectors_arena) {
        result = keys_shaped;
        keys_shaped = NULL;
        goto release;
    }

    PyObject* const vectors_shaped = usearch_arena_matrix(vectors_arena, written, self->columns, self->scalar_kind);
    if (!vectors_shaped) goto release;
    result = PyTuple_Pack(2, keys_shaped, vectors_shaped);
    Py_DECREF(vectors_shaped);

release:
    Py_XDECREF(keys_shaped);
    Py_DECREF(keys_arena);
    Py_XDECREF(vectors_arena);
    return result;
}

static char const doc_DenseCursor[] =                                                 //
    "A snapshot walk over the entries an index held when the cursor opened.\n"        //
    "\n"                                                                              //
    "Yields (keys, vectors) pairs of memoryviews when opened with vectors, and key\n" //
    "matrices alone otherwise. Entries added after the cursor opened stay hidden.";

PyTypeObject DenseCursorType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseCursor",
    .tp_doc = doc_DenseCursor,
    .tp_basicsize = sizeof(DenseCursor),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, //! Returned by `batches`, never constructed by name.
    .tp_dealloc = (destructor)DenseCursor_dealloc,
    .tp_iter = DenseCursor_iter,
    .tp_iternext = (iternextfunc)DenseCursor_next,
};

static char const doc_DenseIndex_batches[] =                                               //
    "batches(size=1024, *, dtype=None) -> DenseCursor\n"                                   //
    "\n"                                                                                   //
    "Walk a snapshot of the index in batches, yielding (keys, vectors) memoryview pairs\n" //
    "when `dtype` names a scalar type, and key matrices alone otherwise.\n"                //
    "\n"                                                                                   //
    "Args:\n"                                                                              //
    "  size (int, optional): Rows per batch. Defaults to 1024.\n"                          //
    "  dtype (str, optional): Export vectors as this scalar type; None walks keys only.";

static PyObject* DenseIndex_batches(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                    PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    size_t batch = 1024;
    PyObject* dtype_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    if (positional_args_count > 1) {
        PyErr_Format(PyExc_TypeError, "Only first argument can be positional, received %zd", positional_args_count);
        return NULL;
    }
    if (positional_args_count == 1 && size_from_object(args[0], "size", &batch) != 0) return NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        if (PyUnicode_CompareWithASCIIString(name, "size") == 0) {
            if (size_from_object(args[args_progress], "size", &batch) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { dtype_object = args[args_progress]; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }
    if (batch == 0) {
        PyErr_SetString(PyExc_ValueError, "The 'size' option must be positive");
        return NULL;
    }

    usearch_scalar_t scalar_kind = usearch_scalar_unknown_k;
    if (usearch_scalar_from_object(dtype_object, &scalar_kind) != 0) return NULL;

    // Only a walk that exports vectors needs the geometry, and only then must it be frozen.
    size_t columns = 0, row_bytes = 0;
    if (scalar_kind != usearch_scalar_unknown_k && index_row_extent(index, scalar_kind, &columns, &row_bytes) != 0)
        return NULL;

    usearch_error_t error = {0};
    usearch_dense_cursor_t const handle = usearch_dense_cursor(index->handle, &error);
    if (usearch_raise_on_error(&error) != 0) return NULL;
    if (!handle) {
        usearch_raise(usearch_error_internal_k, "the library returned no cursor and no error");
        return NULL;
    }

    DenseCursor* const cursor = PyObject_New(DenseCursor, &DenseCursorType);
    if (!cursor) {
        usearch_dense_cursor_free(handle);
        return NULL;
    }
    cursor->handle = handle;
    Py_INCREF(self);
    cursor->index = self;
    cursor->batch = batch;
    cursor->columns = columns;
    cursor->scalar_kind = scalar_kind == usearch_scalar_unknown_k ? usearch_scalar_f32_k : scalar_kind;
    cursor->yields = scalar_kind == usearch_scalar_unknown_k ? dense_cursor_yields_keys_k
                                                             : dense_cursor_yields_keys_and_vectors_k;
    return (PyObject*)cursor;
}

static PyObject* DenseIndex_iter(PyObject* self) {
    PyObject* const size = PyLong_FromSize_t(1024);
    if (!size) return NULL;
    PyObject* const positional[1] = {size};
    PyObject* const cursor = DenseIndex_batches(self, positional, 1, NULL);
    Py_DECREF(size);
    return cursor;
}

static void DenseFinder_dealloc(DenseFinder* self) {
    if (self->handle) usearch_dense_find_free(self->handle);
    Py_XDECREF(self->index);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* DenseFinder_iter(PyObject* self) {
    Py_INCREF(self);
    return self;
}

static PyObject* DenseFinder_next(DenseFinder* self) {
    if (self->consumed == self->buffered) {
        if (self->drained) return NULL;
        usearch_error_t error = {0};
        size_t written = 0;
        Py_BEGIN_ALLOW_THREADS;
        written = usearch_dense_find_next(self->handle, self->keys, self->distances, 64, &error);
        Py_END_ALLOW_THREADS;
        if (usearch_raise_on_error(&error) != 0) return NULL;
        if (written == 0) {
            self->drained = 1;
            return NULL;
        }
        self->buffered = written;
        self->consumed = 0;
    }
    size_t const position = self->consumed++;
    return usearch_match_to_object(self->keys[position], self->distances[position]);
}

static char const doc_DenseFinder[] =                                                   //
    "A resumable best-first stream of matches for one query, yielding Match objects.\n" //
    "\n"                                                                                //
    "Ordering beyond the expansion horizon is best-effort nearest, not globally exact.";

PyTypeObject DenseFinderType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseFinder",
    .tp_doc = doc_DenseFinder,
    .tp_basicsize = sizeof(DenseFinder),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, //! Returned by `find`, never constructed by name.
    .tp_dealloc = (destructor)DenseFinder_dealloc,
    .tp_iter = DenseFinder_iter,
    .tp_iternext = (iternextfunc)DenseFinder_next,
};

static char const doc_DenseIndex_find[] =                                              //
    "find(query, *, dtype=None) -> DenseFinder\n"                                      //
    "\n"                                                                               //
    "Open a resumable nearest-neighbor stream for one query, yielding Match objects\n" //
    "for as long as the caller keeps pulling.\n"                                       //
    "\n"                                                                               //
    "Args:\n"                                                                          //
    "  query (buffer): One vector of `dimensions` scalars.\n"                          //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.";

static PyObject* DenseIndex_find(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                 PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    Py_ssize_t const args_names_count = args_names_count_checked("find", positional_args_count, args_names_tuple, 1);
    if (args_names_count < 0) return NULL;
    PyObject* dtype_object = NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { dtype_object = args[args_progress]; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    usearch_scalar_t scalar_hint = usearch_scalar_unknown_k;
    if (usearch_scalar_from_object(dtype_object, &scalar_hint) != 0) return NULL;

    usearch_matrix_view_t query;
    usearch_buffer_backing_t backing;
    if (usearch_matrix_from_object(args[0], usearch_matrix_access_read_k, scalar_hint, &query, &backing) != 0)
        return NULL;
    if (query.rows != 1) {
        usearch_matrix_release(&backing);
        usearch_raise(usearch_error_shape_mismatch_k, "a finder takes exactly one query row");
        return NULL;
    }

    usearch_error_t error = {0};
    usearch_dense_finder_t const handle = usearch_dense_find(index->handle, query.data, query.scalar_kind, &error);
    usearch_matrix_release(&backing);
    if (usearch_raise_on_error(&error) != 0) return NULL;
    if (!handle) {
        usearch_raise(usearch_error_internal_k, "the library returned no finder and no error");
        return NULL;
    }

    DenseFinder* const finder = PyObject_New(DenseFinder, &DenseFinderType);
    if (!finder) {
        usearch_dense_find_free(handle);
        return NULL;
    }
    finder->handle = handle;
    Py_INCREF(self);
    finder->index = self;
    finder->buffered = 0;
    finder->consumed = 0;
    finder->drained = 0;
    return (PyObject*)finder;
}

#pragma endregion Iteration

#pragma region Serialization

static char const doc_DenseIndex_save[] =                                                     //
    "save(target) -> None\n"                                                                  //
    "\n"                                                                                      //
    "Write the index to a path, or into a writable buffer of `storage_requirements` bytes.\n" //
    "Demands a mutation-free window; concurrent readers keep searching throughout.\n"         //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  target (str | os.PathLike | buffer): Where the checkpoint goes.";

/** @brief Where a checkpoint lives, as the argument's own type spells it. */
typedef enum checkpoint_source_t {
    /** @brief A @c str , @c bytes , or @c os.PathLike naming a file. */
    checkpoint_source_path_k,
    /** @brief An exported buffer holding one serialized monolith. */
    checkpoint_source_buffer_k,
} checkpoint_source_t;

/**
 *  @brief Reads the argument's intent from its type rather than sniffing a protocol.
 *
 *  A @c bytes exports a buffer and is also what @c os.fsencode returns, so sniffing would read
 *  every encoded path as a thirty-byte checkpoint. The path reading wins; a caller who truly
 *  means a blob wraps it in a @c memoryview .
 */
static checkpoint_source_t checkpoint_source(PyObject* object) {
    if (PyUnicode_Check(object) || PyBytes_Check(object)) return checkpoint_source_path_k;
    if (PyObject_HasAttrString(object, "__fspath__")) return checkpoint_source_path_k;
    return checkpoint_source_buffer_k;
}

/** @brief Borrows a whole checkpoint blob, whatever its length. */
static int checkpoint_borrow(PyObject* object, usearch_matrix_access_t access, usearch_buffer_backing_t* backing) {
    backing->synthesized_owner = NULL;
    int const flags = PyBUF_C_CONTIGUOUS | (access == usearch_matrix_access_write_k ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, &backing->buffer, flags) != 0) {
        _PyErr_FormatFromCause(PyExc_ValueError, "The checkpoint buffer must be contiguous%s",
                               access == usearch_matrix_access_write_k ? " and writable" : "");
        return -1;
    }
    return 0;
}

static PyObject* DenseIndex_save(PyObject* self, PyObject* target) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;
    usearch_error_t error = {0};

    if (checkpoint_source(target) == checkpoint_source_buffer_k) {
        usearch_buffer_backing_t backing;
        if (checkpoint_borrow(target, usearch_matrix_access_write_k, &backing) != 0) return NULL;
        void* const data = backing.buffer.buf;
        size_t const length = (size_t)backing.buffer.len;
        Py_BEGIN_ALLOW_THREADS;
        usearch_dense_save_buffer(index->handle, data, length, &error);
        Py_END_ALLOW_THREADS;
        usearch_matrix_release(&backing);
        if (usearch_raise_on_error(&error) != 0) return NULL;
        Py_RETURN_NONE;
    }

    usearch_string_t path;
    PyObject* decoded = NULL;
    if (usearch_path_from_object(target, &path, &decoded) != 0) return NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_save(index->handle, path, &error);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    if (usearch_raise_on_error(&error) != 0) return NULL;
    Py_RETURN_NONE;
}

static char const doc_DenseIndex_load[] =                                                  //
    "load(source) -> None\n"                                                               //
    "\n"                                                                                   //
    "Read a checkpoint into this handle, with the residency deciding what that means:\n"   //
    "materialize into RAM, upload to a device, or open in place on disk. A handle whose\n" //
    "geometry was left at zero freezes it here, from the checkpoint's own headers.\n"      //
    "\n"                                                                                   //
    "Args:\n"                                                                              //
    "  source (str | os.PathLike | buffer): The checkpoint to read.";

static PyObject* DenseIndex_load(PyObject* self, PyObject* source) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;
    usearch_error_t error = {0};

    if (checkpoint_source(source) == checkpoint_source_buffer_k) {
        usearch_buffer_backing_t backing;
        if (checkpoint_borrow(source, usearch_matrix_access_read_k, &backing) != 0) return NULL;
        void const* const data = backing.buffer.buf;
        size_t const length = (size_t)backing.buffer.len;
        Py_BEGIN_ALLOW_THREADS;
        usearch_dense_load_buffer(index->handle, data, length, &error);
        Py_END_ALLOW_THREADS;
        usearch_matrix_release(&backing);
        if (usearch_raise_on_error(&error) != 0) return NULL;
        Py_RETURN_NONE;
    }

    usearch_string_t path;
    PyObject* decoded = NULL;
    if (usearch_path_from_object(source, &path, &decoded) != 0) return NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load(index->handle, path, &error);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    if (usearch_raise_on_error(&error) != 0) return NULL;
    Py_RETURN_NONE;
}

static char const doc_DenseIndex_metadata[] =                                              //
    "metadata(source) -> dict\n"                                                           //
    "\n"                                                                                   //
    "Read a checkpoint's geometry and metric without loading it.\n"                        //
    "\n"                                                                                   //
    "The keys are the constructor keywords the checkpoint fixes, so\n"                     //
    "`DenseIndex(**DenseIndex.metadata(path))` rebuilds a handle that can read it.\n"      //
    "Residency is never among them - where the payload lives stays the caller's choice.\n" //
    "\n"                                                                                   //
    "Args:\n"                                                                              //
    "  source (str | os.PathLike | buffer): The checkpoint to inspect. A `bytes` is\n"     //
    "    read as a path; wrap a blob in a `memoryview` to have it read as one.\n"          //
    "Returns:\n"                                                                           //
    "  dict: 'dimensions', 'metric', 'quantization', 'connectivity', 'pad_dimensions'.";

static PyObject* DenseIndex_metadata(PyObject* type, PyObject* source) {
    usearch_dense_options_t options;
    options_defaults(&options);
    usearch_error_t error = {0};

    if (checkpoint_source(source) == checkpoint_source_buffer_k) {
        usearch_buffer_backing_t backing;
        if (checkpoint_borrow(source, usearch_matrix_access_read_k, &backing) != 0) return NULL;
        void const* const data = backing.buffer.buf;
        size_t const length = (size_t)backing.buffer.len;
        Py_BEGIN_ALLOW_THREADS;
        usearch_dense_metadata_buffer(data, length, &options, &error);
        Py_END_ALLOW_THREADS;
        usearch_matrix_release(&backing);
    }
    else {
        usearch_string_t path;
        PyObject* decoded = NULL;
        if (usearch_path_from_object(source, &path, &decoded) != 0) return NULL;
        Py_BEGIN_ALLOW_THREADS;
        usearch_dense_metadata(path, &options, &error);
        Py_END_ALLOW_THREADS;
        Py_DECREF(decoded);
    }
    if (usearch_raise_on_error(&error) != 0) return NULL;

    return Py_BuildValue("{s:n,s:s,s:s,s:n,s:O}",                                   //
                         "dimensions", (Py_ssize_t)options.dimensions,              //
                         "metric", metric_name(options.metric),                     //
                         "quantization", usearch_scalar_name(options.quantization), //
                         "connectivity", (Py_ssize_t)options.connectivity,          //
                         "pad_dimensions", options.pad_dimensions ? Py_True : Py_False);
}

#pragma endregion Serialization

#pragma region Registration

static PyMethodDef DenseIndex_methods[] = {
    {"reserve", (PyCFunction)DenseIndex_reserve, METH_O, doc_DenseIndex_reserve},                                //
    {"clear", (PyCFunction)DenseIndex_clear, METH_NOARGS, doc_DenseIndex_clear},                                 //
    {"add", (PyCFunction)DenseIndex_add, usearch_py_method_flags_k, doc_DenseIndex_add},                         //
    {"search", (PyCFunction)DenseIndex_search, usearch_py_method_flags_k, doc_DenseIndex_search},                //
    {"search_into", (PyCFunction)DenseIndex_search_into, usearch_py_method_flags_k, doc_DenseIndex_search_into}, //
    {"count", (PyCFunction)DenseIndex_count, METH_O, doc_DenseIndex_count},                                      //
    {"get", (PyCFunction)DenseIndex_get, usearch_py_method_flags_k, doc_DenseIndex_get},                         //
    {"multiply_into", (PyCFunction)DenseIndex_multiply_into, usearch_py_method_flags_k, doc_DenseIndex_multiply_into},
    {"attend_into", (PyCFunction)DenseIndex_attend_into, usearch_py_method_flags_k, doc_DenseIndex_attend_into},
    {"merge", (PyCFunction)DenseIndex_merge, METH_O, doc_DenseIndex_merge},                          //
    {"find", (PyCFunction)DenseIndex_find, usearch_py_method_flags_k, doc_DenseIndex_find},          //
    {"batches", (PyCFunction)DenseIndex_batches, usearch_py_method_flags_k, doc_DenseIndex_batches}, //
    {"save", (PyCFunction)DenseIndex_save, METH_O, doc_DenseIndex_save},                             //
    {"load", (PyCFunction)DenseIndex_load, METH_O, doc_DenseIndex_load},                             //
    {"metadata", (PyCFunction)DenseIndex_metadata, METH_O | METH_STATIC, doc_DenseIndex_metadata},   //
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef DenseIndex_getset[] = {
    {"size", (getter)DenseIndex_get_size, NULL, doc_DenseIndex_size, NULL},                         //
    {"capacity", (getter)DenseIndex_get_capacity, NULL, doc_DenseIndex_capacity, NULL},             //
    {"dimensions", (getter)DenseIndex_get_dimensions, NULL, doc_DenseIndex_dimensions, NULL},       //
    {"connectivity", (getter)DenseIndex_get_connectivity, NULL, doc_DenseIndex_connectivity, NULL}, //
    {"memory_usage", (getter)DenseIndex_get_memory_usage, NULL, doc_DenseIndex_memory_usage, NULL}, //
    {"storage_requirements", (getter)DenseIndex_get_storage_requirements, NULL, doc_DenseIndex_storage_requirements,
     NULL},                                                                                         //
    {"capabilities", (getter)DenseIndex_get_capabilities, NULL, doc_DenseIndex_capabilities, NULL}, //
    {NULL, NULL, NULL, NULL, NULL},
};

static PySequenceMethods DenseIndex_as_sequence = {
    .sq_length = (lenfunc)DenseIndex_length,
    .sq_contains = (objobjproc)DenseIndex_contains,
};

static PyMappingMethods DenseIndex_as_mapping = {
    .mp_length = (lenfunc)DenseIndex_length,
    .mp_subscript = (binaryfunc)DenseIndex_subscript,
};

PyTypeObject DenseIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseIndex",
    .tp_doc = doc_DenseIndex,
    .tp_basicsize = sizeof(DenseIndex),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = DenseIndex_tp_new,
    .tp_dealloc = (destructor)DenseIndex_dealloc,
    .tp_repr = (reprfunc)DenseIndex_repr,
    .tp_iter = DenseIndex_iter,
    .tp_methods = DenseIndex_methods,
    .tp_getset = DenseIndex_getset,
    .tp_as_sequence = &DenseIndex_as_sequence,
    .tp_as_mapping = &DenseIndex_as_mapping,
};

#pragma endregion Registration
