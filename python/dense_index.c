/**
 *  @file python/dense_index.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief Vector search over equidimensional dense embeddings - the CPython twin of
 *      `include/usearch/dense_index.h`.
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

/** An owned index handle, plus the geometry the verbs need without a round-trip. */
typedef struct DenseIndex {
    PyObject ob_base;
    usearch_dense_index_t handle;
} DenseIndex;

/** What one cursor step yields. */
typedef enum dense_cursor_yields_t {

    /** One @c (batch,16) matrix of identities. */
    dense_cursor_yields_keys_k = 0,

    /** A @c (keys,vectors) pair of matrices. */
    dense_cursor_yields_keys_and_vectors_k = 1,
} dense_cursor_yields_t;

/** A snapshot walk over the entries an index held when the cursor opened. */
typedef struct DenseCursor {
    PyObject ob_base;

    /** The engine cursor, pinned to the watermark it opened at. */
    usearch_dense_cursor_t handle;

    /** Keeps the index alive for the cursor's whole life. */
    PyObject* index;

    /** Rows one step asks for. */
    size_t batch;

    /** The index's frozen row width, cached at open. */
    size_t columns;

    /** The scalar type vectors are exported as. */
    usearch_scalar_t scalar_kind;

    /** Whether a step carries vectors beside the identities. */
    dense_cursor_yields_t yields;
} DenseCursor;

/** A resumable best-first stream of matches for one query. */
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

/**
 *  @brief The metric kinds, one name each.
 *
 *  No aliases: `l2` and `l2sq` would name one enumerator while making opposite claims about
 *  whether the distance is squared, and every other synonym only invites the same drift.
 */
static usearch_py_choice_naming_t const metric_namings[] = {
    {"inner", usearch_dense_metric_inner_k},         {"angular", usearch_dense_metric_angular_k},
    {"euclidean", usearch_dense_metric_euclidean_k}, {"hamming", usearch_dense_metric_hamming_k},
    {"jaccard", usearch_dense_metric_jaccard_k},
};

#pragma endregion Option Names

#pragma region Construction

static char const doc_DenseIndex[] =                                                           //
    "DenseIndex\n"                                                                             //
    "\n"                                                                                       //
    "Vector search over equidimensional dense embeddings.\n"                                   //
    "\n"                                                                                       //
    "Every verb is a batch, and one vector is a one-row matrix. The stored scalar type is\n"   //
    "the `quantization`; query rows may arrive in any other type and convert on the way in.\n" //
    "\n"                                                                                       //
    "Construct one through the residency its rows live in - `for_ram`, `for_vram` or\n"        //
    "`for_nvme` - since each places shards differently and takes different arguments. The\n"   //
    "residency also decides what `load` means: materialize into host memory, upload to the\n"  //
    "named devices, or open in place on disk.\n"                                               //
    "\n"                                                                                       //
    "The verbs that fan out take an `Executor`, which owns the workers and is shared across\n" //
    "indexes; its residency must match this handle's. See `usearch.Executor`.\n"               //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=256, metric='angular')\n"             //
    "  >>> index.capabilities is not None\n"                                                   //
    "  True";

/** Whether a keyword belonged to the shared geometry, to the factory's own arm, or to neither. */
typedef enum keyword_verdict_t {
    keyword_consumed_k,
    keyword_foreign_k,
    keyword_failed_k,
} keyword_verdict_t;

/** Reads one geometry keyword into @p options, leaving the residency arm to the caller. */
static keyword_verdict_t geometry_from_keyword(usearch_dense_options_t* options, PyObject* name, PyObject* value) {
    if (PyUnicode_CompareWithASCIIString(name, "dimensions") == 0)
        return usearch_py_size_from_object(value, "dimensions", &options->dimensions) == 0 ? keyword_consumed_k
                                                                                           : keyword_failed_k;
    if (PyUnicode_CompareWithASCIIString(name, "metric") == 0) {
        int choice = 0;
        if (usearch_py_choice_from_object(value, "metric", metric_namings, usearch_py_table_size_(metric_namings),
                                          &choice) != 0)
            return keyword_failed_k;
        options->metric = (usearch_dense_metric_t)choice;
        return keyword_consumed_k;
    }
    if (PyUnicode_CompareWithASCIIString(name, "quantization") == 0)
        return usearch_py_scalar_from_object(value, &options->quantization) == 0 ? keyword_consumed_k
                                                                                 : keyword_failed_k;
    if (PyUnicode_CompareWithASCIIString(name, "connectivity") == 0)
        return usearch_py_size_from_object(value, "connectivity", &options->connectivity) == 0 ? keyword_consumed_k
                                                                                               : keyword_failed_k;
    if (PyUnicode_CompareWithASCIIString(name, "values_per_key") == 0)
        return usearch_py_size_from_object(value, "values_per_key", &options->values_per_key) == 0 ? keyword_consumed_k
                                                                                                   : keyword_failed_k;
    return keyword_foreign_k;
}

/**
 *  @brief Reads the three settings every caller names, positionally.
 *
 *  The rest are keyword-only, because a call site reading `for_ram(256, 'angular', 'bf16', 0, 0)`
 *  tells the reader nothing about the last two.
 *
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int geometry_from_positionals(usearch_dense_options_t* options, char const* verb, PyObject* const* args,
                                     Py_ssize_t positional_args_count) {
    if (positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "%s() takes at most 3 positional arguments, received %zd", verb,
                     positional_args_count);
        return -1;
    }
    if (positional_args_count >= 1 && usearch_py_size_from_object(args[0], "dimensions", &options->dimensions) != 0)
        return -1;
    if (positional_args_count >= 2) {
        int choice = 0;
        if (usearch_py_choice_from_object(args[1], "metric", metric_namings, usearch_py_table_size_(metric_namings),
                                          &choice) != 0)
            return -1;
        options->metric = (usearch_dense_metric_t)choice;
    }
    if (positional_args_count >= 3 && usearch_py_scalar_from_object(args[2], &options->quantization) != 0) return -1;
    return 0;
}

/** The geometry a factory starts from, before any argument touches it. */
static void options_defaults(usearch_dense_options_t* options, usearch_residency_t residency) {
    memset(options, 0, sizeof(*options));
    options->metric = usearch_dense_metric_angular_k;
    options->quantization = usearch_scalar_bf16_k;
    options->residency = residency;
}

/**
 *  @brief Builds the handle and wraps it, or frees it when the wrapper cannot be allocated.
 *  @return A new reference, or NULL with an exception set.
 */
static PyObject* index_adopt(PyTypeObject* type, usearch_dense_options_t const* options) {
    usearch_error_t error = {0};
    usearch_dense_index_t const handle = usearch_dense_init(options, &error);
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
    if (!handle) {
        usearch_py_raise(usearch_error_internal_k, "the library returned no handle and no error");
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

static PyObject* DenseIndex_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_TypeError, "DenseIndex is built through DenseIndex.for_ram(), .for_vram() or .for_nvme()");
    return NULL;
}

static char const doc_DenseIndex_for_ram[] =                                                  //
    "for_ram(dimensions=0, metric='angular', quantization='bf16', *, connectivity=0,\n"       //
    "    values_per_key=0, memory_domains=None) -> DenseIndex\n"                              //
    "\n"                                                                                      //
    "Places the graph in host memory, one shard per named NUMA node.\n"                       //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  dimensions (int, optional): Scalars per vector; 0 defers the geometry to `load`.\n"    //
    "  metric (str, optional): 'inner', 'angular', 'euclidean', 'hamming', or 'jaccard'.\n"   //
    "  quantization (str, optional): The stored scalar type. Defaults to 'bf16'.\n"           //
    "  connectivity (int, optional): Connections per node; 0 derives it from the geometry.\n" //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"          //
    "  memory_domains (Sequence[int], optional): NUMA node ids to shard across, so a\n"       //
    "    dual-socket box naming two gets two graphs; None derives every node.\n"              //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> index = usearch.DenseIndex.for_ram(256, 'angular')\n"                              //
    "  >>> index.dimensions\n"                                                                //
    "  256";

static PyObject* DenseIndex_for_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                    PyObject* args_names_tuple) {
    usearch_dense_options_t options;
    options_defaults(&options, usearch_residency_ram_k);
    if (geometry_from_positionals(&options, "for_ram", args, positional_args_count) != 0) return NULL;

    PyObject* memory_domains_object = Py_None;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        switch (geometry_from_keyword(&options, name, value)) {
        case keyword_consumed_k: continue;
        case keyword_failed_k: return NULL;
        case keyword_foreign_k: break;
        }
        if (PyUnicode_CompareWithASCIIString(name, "memory_domains") == 0) { memory_domains_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    int32_t* memory_domains = NULL;
    if (usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &options.ram.memory_domains_count) != 0)
        return NULL;
    options.ram.memory_domains = memory_domains;

    PyObject* const instance = index_adopt(type, &options);
    PyMem_Free(memory_domains);
    return instance;
}

static char const doc_DenseIndex_for_vram[] =                                                 //
    "for_vram(dimensions=0, metric='angular', quantization='bf16', *, connectivity=0,\n"      //
    "    values_per_key=0, devices=None) -> DenseIndex\n"                                     //
    "\n"                                                                                      //
    "Places the graph on CUDA devices, one shard per named ordinal.\n"                        //
    "\n"                                                                                      //
    "Shards are sharded and never replicated, so eight devices host eight slices of one\n"    //
    "graph rather than eight copies. An index outgrowing them fails rather than paging to\n"  //
    "the host, and the answer is to name more devices.\n"                                     //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  dimensions (int, optional): Scalars per vector; 0 defers the geometry to `load`.\n"    //
    "  metric (str, optional): 'inner', 'angular', 'euclidean', 'hamming', or 'jaccard'.\n"   //
    "  quantization (str, optional): The stored scalar type. Defaults to 'bf16'.\n"           //
    "  connectivity (int, optional): Connections per node; 0 derives it from the geometry.\n" //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"          //
    "  devices (Sequence[int], optional): CUDA ordinals to shard across; None derives one\n"  //
    "    free device.\n"                                                                      //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> index = usearch.DenseIndex.for_vram(256, devices=[0, 1])   # doctest: +SKIP";

static PyObject* DenseIndex_for_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    usearch_dense_options_t options;
    options_defaults(&options, usearch_residency_vram_k);
    if (geometry_from_positionals(&options, "for_vram", args, positional_args_count) != 0) return NULL;

    PyObject* devices_object = Py_None;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        switch (geometry_from_keyword(&options, name, value)) {
        case keyword_consumed_k: continue;
        case keyword_failed_k: return NULL;
        case keyword_foreign_k: break;
        }
        if (PyUnicode_CompareWithASCIIString(name, "devices") == 0) { devices_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    int32_t* devices = NULL;
    if (usearch_py_ordinals_from_object(devices_object, "devices", &devices, &options.vram.devices_count) != 0)
        return NULL;
    options.vram.devices = devices;

    PyObject* const instance = index_adopt(type, &options);
    PyMem_Free(devices);
    return instance;
}

static char const doc_DenseIndex_for_nvme[] =                                                 //
    "for_nvme(dimensions=0, metric='angular', quantization='bf16', *, connectivity=0,\n"      //
    "    values_per_key=0, paths=None) -> DenseIndex\n"                                       //
    "\n"                                                                                      //
    "Places the graph on disk, one shard per named path.\n"                                   //
    "\n"                                                                                      //
    "A path is whatever identifies a device to an engine that can reach it: a filesystem\n"   //
    "path names a kernel device, a transport identifier selects a userspace engine, and a\n"  //
    "FUSE mount is how object storage stages in without an HTTP stack.\n"                     //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  dimensions (int, optional): Scalars per vector; 0 defers the geometry to `load`.\n"    //
    "  metric (str, optional): 'inner', 'angular', 'euclidean', 'hamming', or 'jaccard'.\n"   //
    "  quantization (str, optional): The stored scalar type. Defaults to 'bf16'.\n"           //
    "  connectivity (int, optional): Connections per node; 0 derives it from the geometry.\n" //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"          //
    "  paths (Sequence[str | bytes | PathLike], optional): Devices or directories to\n"       //
    "    stripe across; None leaves every shard where its manifest points.\n"                 //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> index = usearch.DenseIndex.for_nvme(256, paths=['/mnt/nvme0'])   # doctest: +SKIP";

static PyObject* DenseIndex_for_nvme(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    usearch_dense_options_t options;
    options_defaults(&options, usearch_residency_nvme_k);
    if (geometry_from_positionals(&options, "for_nvme", args, positional_args_count) != 0) return NULL;

    PyObject* paths_object = Py_None;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        switch (geometry_from_keyword(&options, name, value)) {
        case keyword_consumed_k: continue;
        case keyword_failed_k: return NULL;
        case keyword_foreign_k: break;
        }
        if (PyUnicode_CompareWithASCIIString(name, "paths") == 0) { paths_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    usearch_py_paths_arena_t paths;
    if (usearch_py_paths_from_object(paths_object, &paths) != 0) return NULL;
    options.nvme.paths = paths.views;
    options.nvme.paths_count = paths.count;

    PyObject* const instance = index_adopt(type, &options);
    usearch_py_paths_release(&paths);
    return instance;
}

static void DenseIndex_dealloc(DenseIndex* self) {
    if (self->handle) usearch_dense_free(self->handle);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** Rejects a handle already consumed by `merge`. */
static int index_alive(DenseIndex const* self) {
    if (self->handle) return 0;
    usearch_py_raise(usearch_error_uninitialized_k, "the index handle was consumed by a merge");
    return -1;
}

/**
 *  @brief Refuses a fanned-out verb that was handed no executor.
 *
 *  The library never derives one: a null handle reads as a residency mismatch three frames down,
 *  so the absence is named here, where the keyword is.
 *
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int executor_named(usearch_executor_t executor, char const* verb) {
    if (executor) return 0;
    PyErr_Format(PyExc_TypeError,                                                                           //
                 "%s() needs an 'executor'; build one with Executor.for_ram(), .for_vram() or .for_nvme()", //
                 verb);
    return -1;
}

#pragma endregion Construction

#pragma region Progress

/** What one watched verb hands its progress trampoline. */
typedef struct progress_state_t {

    /** The caller's progress callable, or NULL when the verb was handed none. */
    PyObject* callable;

    /** The first exception any report raised; NULL while none has. */
    PyObject* raised;
} progress_state_t;

/**
 *  @brief Re-enters Python for one progress report, from whichever library thread advances.
 *
 *  Reports are serialized, but nothing promises they land on the caller's thread, so the GIL is
 *  taken here rather than assumed. A report that raises also stops the walk, so the call unwinds
 *  at the next boundary instead of running to completion with a doomed result.
 */
static usearch_continuation_t progress_trampoline(size_t completed, size_t total, void* state_pointer) {
    progress_state_t* const state = (progress_state_t*)state_pointer;
    PyGILState_STATE const gil_state = PyGILState_Ensure();
    usearch_continuation_t continuation = usearch_cancel_k;
    PyObject* const completed_object = PyLong_FromSize_t(completed);
    PyObject* const total_object = PyLong_FromSize_t(total);
    if (completed_object && total_object) {
        PyObject* const verdict = PyObject_CallFunctionObjArgs(state->callable, completed_object, total_object, NULL);
        if (verdict) {
            int const truth = PyObject_IsTrue(verdict);
            Py_DECREF(verdict);
            continuation = truth > 0 ? usearch_continue_k : usearch_cancel_k;
        }
    }
    Py_XDECREF(total_object);
    Py_XDECREF(completed_object);
    PyObject* const raised = PyErr_GetRaisedException(); //! NULL when nothing was raised.
    if (raised && !state->raised) state->raised = raised;
    else Py_XDECREF(raised);
    PyGILState_Release(gil_state);
    return continuation;
}

/**
 *  @brief Reads the `progress` keyword, leaving @p callable alone when the value names None.
 *  @return 0 on success, -1 with an exception set when the value is not callable.
 */
static int progress_from_object(PyObject* value, PyObject** callable) {
    if (value == Py_None) return 0;
    if (!PyCallable_Check(value)) {
        PyErr_Format(PyExc_TypeError, "The 'progress' handler must be callable, got %S", Py_TYPE(value));
        return -1;
    }
    *callable = value;
    return 0;
}

/** The trampoline to register for @p state, or NULL when the verb was handed no handler. */
static usearch_progress_callback_t progress_registered(progress_state_t const* state) {
    return state->callable ? progress_trampoline : NULL;
}

/**
 *  @brief Re-raises on the caller's thread whatever a report left latched.
 *  @return -1 with the handler's exception set when one raised, 0 otherwise.
 */
static int progress_raised(progress_state_t* state) {
    if (!state->raised) return 0;
    PyErr_SetRaisedException(state->raised);
    state->raised = NULL;
    return -1;
}

#pragma endregion Progress

#pragma region Introspection

/** The whole family of `(handle, error) → size_t` probes, through one call site. */
static PyObject* measure(DenseIndex* self, size_t (*probe)(usearch_dense_index_t, usearch_error_t*)) {
    if (index_alive(self) != 0) return NULL;
    usearch_error_t error = {0};
    size_t const measured = probe(self->handle, &error);
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
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
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
    return PyUnicode_FromString(codename ? codename : "");
}

/**
 *  @brief The frozen row width in logical dimensions, and the row's byte extent in @p scalar_kind .
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
    if (usearch_py_raise_on_error(&error) != 0) return -1;
    size_t const bytes = usearch_py_scalar_row_bytes(scalar_kind, dimensions);
    if (bytes == 0)
        return usearch_py_raise(usearch_error_uninitialized_k,                                  //
                                "the index geometry is not frozen yet; reserve or load first"); //
    *columns = dimensions;
    *row_bytes = bytes;
    return 0;
}

/**
 *  @brief Refuses a matrix whose rows disagree with the frozen geometry.
 *
 *  The library takes a base pointer and a stride and is never told how wide the caller believes
 *  a row is, so a narrower row here is an out-of-bounds read there.
 *
 *  @param[in] role What the matrix is to the verb, named as the message should read it.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int index_rows_match(DenseIndex* self, usearch_matrix_view_t const* matrix, char const* role) {
    size_t columns = 0, row_bytes = 0;
    if (index_row_extent(self, matrix->scalar_kind, &columns, &row_bytes) != 0) return -1;
    if (matrix->columns == columns) return 0;
    // A packed row carries whole storage units, so a geometry stopping mid-unit is still the row
    // the engine reads: the byte extent settles what the dimension count cannot express.
    if (matrix->columns > columns && usearch_py_scalar_row_bytes(matrix->scalar_kind, matrix->columns) == row_bytes)
        return 0;
    return usearch_py_raise(usearch_error_shape_mismatch_k, "the %s are %zu scalars wide, but the index is %zu", role,
                            matrix->columns, columns);
}

static Py_ssize_t DenseIndex_length(DenseIndex* self) {
    if (index_alive(self) != 0) return -1;
    usearch_error_t error = {0};
    size_t const size = usearch_dense_size(self->handle, &error);
    if (usearch_py_raise_on_error(&error) != 0) return -1;
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
    if (usearch_py_size_from_object(capacity_object, "capacity", &capacity) != 0) return NULL;

    usearch_error_t error = {0};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_reserve(index->handle, capacity, &error);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
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
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
    Py_RETURN_NONE;
}

static char const doc_DenseIndex_add[] =                                                    //
    "add(keys, vectors, *, executor, expansion_factor=0, dtype=None, progress=None,\n"      //
    "    stats=None) -> int\n"                                                              //
    "\n"                                                                                    //
    "Insert a batch of vectors under a batch of keys; one vector is a one-row matrix.\n"    //
    "Keys need not be unique: inserting one twice stores two entries, and `count`\n"        //
    "reports both.\n"                                                                       //
    "\n"                                                                                    //
    "A batch without a handler is all-or-nothing, so a rejected batch leaves the index\n"   //
    "untouched. With one it advances in increments instead: a cancellation stops at an\n"   //
    "increment boundary, keeps every row already published, and the return counts what\n"   //
    "stands.\n"                                                                             //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  keys (Sequence[Key | int] | buffer): One key per row, or an (N, 16) buffer.\n"       //
    "  vectors (buffer): An (N, dimensions) matrix; only the row stride may vary.\n"        //
    "  executor (Executor): The workers to fan out over; its residency must match.\n"       //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"           //
    "  dtype (str, optional): Names the input scalar type when the buffer cannot.\n"        //
    "  progress (callable, optional): Called with (completed, total) rows; returning\n"     //
    "    falsy stops the batch. Reports are serialized, so the handler needs no locking.\n" //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None\n"   //
    "    skips reporting.\n"                                                                //
    "Returns:\n"                                                                            //
    "  int: The number of rows inserted - every key, unless a handler stopped early.\n"     //
    "\n"                                                                                    //
    "Example:\n"                                                                            //
    "  >>> from array import array\n"                                                       //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=4)\n"                              //
    "  >>> index.reserve(1)\n"                                                              //
    "  >>> pool = usearch.Executor.for_ram()\n"                                             //
    "  >>> index.add([1], array('d', [0.1, 0.2, 0.3, 0.4]), executor=pool)\n"               //
    "  1";

static PyObject* DenseIndex_add(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    PyObject *stats_object = NULL, *dtype_object = NULL;
    progress_state_t progress_state = {NULL, NULL};
    usearch_executor_t executor = NULL;
    size_t expansion_factor = 0;

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("add", positional_args_count,
                                                                            args_names_tuple, 2);
    if (args_names_count < 0) return NULL;
    PyObject* const keys_object = args[0];
    PyObject* const vectors_object = args[1];

    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "expansion_factor") == 0) {
            if (usearch_py_size_from_object(value, "expansion_factor", &expansion_factor) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { dtype_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "progress") == 0) {
            if (progress_from_object(value, &progress_state.callable) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "stats") == 0) { stats_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(value, &executor) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    if (executor_named(executor, "add") != 0) return NULL;

    usearch_scalar_t scalar_hint = usearch_scalar_unknown_k;
    if (usearch_py_scalar_from_object(dtype_object, &scalar_hint) != 0) return NULL;

    usearch_matrix_view_t vectors;
    usearch_py_buffer_backing_t backing;
    if (usearch_py_matrix_from_object(vectors_object, scalar_hint, &vectors, &backing) != 0) return NULL;
    if (index_rows_match(index, &vectors, "vectors") != 0) {
        usearch_py_buffer_release(&backing);
        return NULL;
    }

    usearch_py_key_arena_t arena;
    if (usearch_py_keys_from_object(keys_object, vectors.rows, &arena) != 0) {
        usearch_py_buffer_release(&backing);
        return NULL;
    }

    usearch_keys_view_t const keys = {arena.keys, vectors.rows};
    usearch_dense_insertion_options_t const options = {.expansion_factor = expansion_factor};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_error_t error = {0};
    size_t inserted = 0;

    Py_BEGIN_ALLOW_THREADS;
    inserted = usearch_dense_add(index->handle, executor, keys, &vectors, &options,
                                 progress_registered(&progress_state), &progress_state, &stats, &error);
    Py_END_ALLOW_THREADS;

    usearch_py_keys_release(&arena);
    usearch_py_buffer_release(&backing);
    // A handler that raised outranks whatever the batch went on to report, and a cancellation it
    // asked for is the rows that stand rather than a failure.
    if (progress_raised(&progress_state) != 0) return NULL;
    if (error.status != usearch_error_cancelled_k && usearch_py_raise_on_error(&error) != 0) return NULL;
    if (usearch_py_stats_overwrite(stats_object, &stats) != 0) return NULL;
    return PyLong_FromSize_t(inserted);
}

static char const doc_DenseIndex_refine[] =                                                  //
    "refine(*, executor, expansion_factor=0, progress=None, stats=None) -> int\n"            //
    "\n"                                                                                     //
    "Re-select the neighbors of every stored entry against the finished graph, improving\n"  //
    "edges that were chosen when the graph around them was still incomplete.\n"              //
    "\n"                                                                                     //
    "A build step rather than a repair: a cheap insertion pass followed by one refinement\n" //
    "reaches a better graph, in less total time, than one expensive insertion pass - so a\n" //
    "wide `expansion_factor` belongs here rather than in `add`. Searches may run\n"          //
    "throughout, and the sweep resumes where it stopped, so a caller loops until the\n"      //
    "return falls below whatever change rate they consider settled.\n"                       //
    "\n"                                                                                     //
    "Stopping is not an error here and reports no cancelled status: re-selecting edges\n"    //
    "claims no identity, so a stopped sweep leaves the graph correct and merely less\n"      //
    "improved. A handler is therefore how a sweep is put under a time budget.\n"             //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  executor (Executor): The workers to fan out over; its residency must match.\n"        //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"            //
    "  progress (callable, optional): Called with (completed, total) nodes; returning\n"     //
    "    falsy ends the sweep, and the next call resumes where it stopped.\n"                //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None\n"    //
    "    skips reporting.\n"                                                                 //
    "Returns:\n"                                                                             //
    "  int: Nodes whose edges changed; zero means the graph has settled.\n"                  //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> from array import array\n"                                                        //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=4)\n"                               //
    "  >>> index.reserve(1)\n"                                                               //
    "  >>> pool = usearch.Executor.for_ram()\n"                                              //
    "  >>> _ = index.add([1], array('d', [0.1, 0.2, 0.3, 0.4]), executor=pool)\n"            //
    "  >>> index.refine(executor=pool)\n"                                                    //
    "  0";

static PyObject* DenseIndex_refine(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    PyObject* stats_object = NULL;
    progress_state_t progress_state = {NULL, NULL};
    usearch_executor_t executor = NULL;
    size_t expansion_factor = 0;

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("refine", positional_args_count,
                                                                            args_names_tuple, 0);
    if (args_names_count < 0) return NULL;

    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "expansion_factor") == 0) {
            if (usearch_py_size_from_object(value, "expansion_factor", &expansion_factor) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "progress") == 0) {
            if (progress_from_object(value, &progress_state.callable) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "stats") == 0) { stats_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(value, &executor) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    if (executor_named(executor, "refine") != 0) return NULL;

    usearch_dense_insertion_options_t const options = {.expansion_factor = expansion_factor};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_error_t error = {0};
    size_t changed = 0;

    Py_BEGIN_ALLOW_THREADS;
    changed = usearch_dense_refine(index->handle, executor, &options, progress_registered(&progress_state),
                                   &progress_state, &stats, &error);
    Py_END_ALLOW_THREADS;

    if (progress_raised(&progress_state) != 0) return NULL;
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
    if (usearch_py_stats_overwrite(stats_object, &stats) != 0) return NULL;
    return PyLong_FromSize_t(changed);
}

static char const doc_DenseIndex_health[] =                                                    //
    "health(*, executor) -> dict\n"                                                            //
    "\n"                                                                                       //
    "Audit how much of the graph a search can reach, sweeping forward reachability from as\n"  //
    "many landmarks as one search draws. Read-only.\n"                                         //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  executor (Executor): The workers to fan out over; its residency must match.\n"          //
    "Returns:\n"                                                                               //
    "  dict: 'audited' entries published when the audit started, the 'reachable' ones among\n" //
    "    them, and the 'sweeps' it took to converge.\n"                                        //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> from array import array\n"                                                          //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=4)\n"                                 //
    "  >>> index.reserve(1)\n"                                                                 //
    "  >>> pool = usearch.Executor.for_ram()\n"                                                //
    "  >>> _ = index.add([1], array('d', [0.1, 0.2, 0.3, 0.4]), executor=pool)\n"              //
    "  >>> health = index.health(executor=pool)\n"                                             //
    "  >>> health['reachable'] == health['audited'] == 1\n"                                    //
    "  True";

static PyObject* DenseIndex_health(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    usearch_executor_t executor = NULL;
    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("health", positional_args_count,
                                                                            args_names_tuple, 0);
    if (args_names_count < 0) return NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(value, &executor) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }
    if (executor_named(executor, "health") != 0) return NULL;

    usearch_error_t error = {0};
    usearch_dense_health_t health = {0};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_health(index->handle, executor, &health, &error);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_error(&error) != 0) return NULL;

    PyObject* const result = PyDict_New();
    if (!result) return NULL;
    if (usearch_py_dict_steal_item(result, "audited", PyLong_FromSize_t(health.audited)) != 0 ||
        usearch_py_dict_steal_item(result, "reachable", PyLong_FromSize_t(health.reachable)) != 0 ||
        usearch_py_dict_steal_item(result, "sweeps", PyLong_FromSize_t(health.sweeps)) != 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

#pragma endregion Ingest

#pragma region Search

/** What every ranked verb agrees on before it calls the library. */
typedef struct search_request_t {

    /** The borrowed query matrix. */
    usearch_matrix_view_t queries;

    /** Storage the query view points into. */
    usearch_py_buffer_backing_t queries_backing;

    /** Matches wanted per query - the "k" in "kANN". */
    size_t wanted;

    /** Candidate-pool breadth for this call; zero derives. */
    size_t expansion_factor;

    /** The admission callable, or NULL for an unfiltered walk. */
    PyObject* predicate;

    /** The progress callable, or NULL for an unwatched walk. */
    PyObject* progress;

    /** The @c Stats to overwrite, or NULL to skip reporting. */
    PyObject* stats_object;

    /** The workers this call fans out over. */
    usearch_executor_t executor;
} search_request_t;

/** What one filtered search hands its trampoline. */
typedef struct predicate_state_t {

    /** The caller's admission callable. */
    PyObject* callable;

    /** The first exception any callback raised; NULL while none has. */
    PyObject* raised;
} predicate_state_t;

/**
 *  @brief Re-enters Python for one admission decision, from whichever library thread traverses.
 *
 *  A worker's thread state dies with the callback and the caller's would surface the exception
 *  at an unrelated return, so a raising predicate leaves its failure in the state for
 *  @c DenseIndex_search to re-raise rather than on whichever thread caught it.
 */
static usearch_admission_t predicate_trampoline(usearch_key_t key, void* state_pointer) {
    predicate_state_t* const state = (predicate_state_t*)state_pointer;
    PyGILState_STATE const gil_state = PyGILState_Ensure();
    usearch_admission_t admitted = usearch_reject_k;
    PyObject* const key_object = usearch_py_key_to_object(key);
    if (key_object) {
        PyObject* const verdict = PyObject_CallOneArg(state->callable, key_object);
        Py_DECREF(key_object);
        if (verdict) {
            int const truth = PyObject_IsTrue(verdict);
            Py_DECREF(verdict);
            admitted = truth > 0 ? usearch_admit_k : usearch_reject_k;
        }
    }
    PyObject* const raised = PyErr_GetRaisedException(); //! NULL when nothing was raised.
    if (raised && !state->raised) state->raised = raised;
    else Py_XDECREF(raised);
    PyGILState_Release(gil_state);
    return admitted;
}

/**
 *  @brief Parses the arguments every read-only walk shares: the queries at position zero, the
 *      output width at @p wanted_position, and the keywords that follow.
 *
 *  The width does not always sit second - `maxsim` names the query grouping before it - so the
 *  position travels rather than being assumed.
 *
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int search_request_parse(DenseIndex* index, char const* verb, PyObject* const* args,
                                Py_ssize_t positional_args_count, PyObject* args_names_tuple,
                                Py_ssize_t required_positional, Py_ssize_t wanted_position, search_request_t* request) {
    memset(request, 0, sizeof(*request));

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked(verb, positional_args_count,
                                                                            args_names_tuple, required_positional);
    if (args_names_count < 0) return -1;

    PyObject* dtype_object = NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "expansion_factor") == 0) {
            if (usearch_py_size_from_object(value, "expansion_factor", &request->expansion_factor) != 0) return -1;
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
        else if (PyUnicode_CompareWithASCIIString(name, "progress") == 0) {
            if (progress_from_object(value, &request->progress) != 0) return -1;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "stats") == 0) { request->stats_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(value, &request->executor) != 0) return -1;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return -1;
        }
    }

    if (executor_named(request->executor, verb) != 0) return -1;

    usearch_scalar_t scalar_hint = usearch_scalar_unknown_k;
    if (usearch_py_scalar_from_object(dtype_object, &scalar_hint) != 0) return -1;
    if (usearch_py_matrix_from_object(args[0], scalar_hint, &request->queries, &request->queries_backing) != 0)
        return -1;
    if (index_rows_match(index, &request->queries, "queries") != 0) {
        usearch_py_buffer_release(&request->queries_backing);
        return -1;
    }
    if (usearch_py_size_from_object(args[wanted_position], "wanted", &request->wanted) != 0) {
        usearch_py_buffer_release(&request->queries_backing);
        return -1;
    }
    if (request->wanted == 0) {
        usearch_py_buffer_release(&request->queries_backing);
        usearch_py_raise(usearch_error_shape_mismatch_k, "a search must want at least one match");
        return -1;
    }
    return 0;
}

/**
 *  @brief Borrows the three outputs a ranked verb fills, sized for @p rows rows of @p wanted.
 *
 *  One acquisition loop and one release, so the unwind order lives in a single place rather
 *  than being hand-maintained once per verb. Nothing stays borrowed when it fails.
 *
 *  @param[in] outputs The three output objects, in the order @p fields names them.
 *  @param[in] fields The names each output answers to in a diagnostic.
 *  @param[out] backings Storage the pointers came from; release with @c matches_outputs_release.
 *  @param[out] pointers The first byte of each output.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int matches_outputs_borrow(PyObject* const* outputs, char const* const* fields, size_t rows, size_t wanted,
                                  usearch_py_buffer_backing_t* backings, void** pointers) {
    Py_ssize_t extents[3] = {0, 0, 0};
    if (usearch_py_matrix_bytes(rows, wanted, sizeof(usearch_key_t), &extents[0]) != 0 ||
        usearch_py_matrix_bytes(rows, wanted, sizeof(usearch_distance_t), &extents[1]) != 0 ||
        usearch_py_matrix_bytes(rows, 1, sizeof(size_t), &extents[2]) != 0)
        return -1;

    for (size_t borrowed = 0; borrowed != 3; ++borrowed) {
        if (usearch_py_buffer_from_object(outputs[borrowed], fields[borrowed], usearch_py_matrix_access_write_k,
                                          extents[borrowed], &backings[borrowed], &pointers[borrowed]) == 0)
            continue;
        while (borrowed--) usearch_py_buffer_release(&backings[borrowed]);
        return -1;
    }
    return 0;
}

/** Releases the three outputs @c matches_outputs_borrow acquired. */
static void matches_outputs_release(usearch_py_buffer_backing_t* backings) {
    for (size_t borrowed = 3; borrowed--;) usearch_py_buffer_release(&backings[borrowed]);
}

static char const doc_DenseIndex_search[] =                                                    //
    "search(queries, wanted, keys, distances, counts, *, executor,\n"                          //
    "    expansion_factor=0, dtype=None, predicate=None, progress=None, stats=None) -> None\n" //
    "\n"                                                                                       //
    "Find the `wanted` nearest entries for every query row; one query is a one-row matrix.\n"  //
    "Never writes to the graph, so any number of searches may overlap with insertions.\n"      //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  queries (buffer): A (Q, dimensions) matrix of query rows.\n"                            //
    "  wanted (int): Upper bound on matches per query - the 'k' in 'kANN'.\n"                  //
    "  keys (buffer): A writable (Q, wanted, 16) buffer of identity bytes.\n"                  //
    "  distances (buffer): A writable (Q, wanted) buffer of float64 distances.\n"              //
    "  counts (buffer): A writable (Q,) buffer of unsigned word-sized counts. Each bounds\n"   //
    "    its own row; cells past it keep whatever the buffer already held.\n"                  //
    "  executor (Executor): The workers to fan out over; its residency must match.\n"          //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"              //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.\n"           //
    "  predicate (callable, optional): Admits a Key when it returns truthy. Non-matching\n"    //
    "    entries still route the traversal, so recall survives low selectivity. The\n"         //
    "    callback re-enters Python, so it runs under the GIL and serializes the walk.\n"       //
    "  progress (callable, optional): Called with (completed, total) queries; returning\n"     //
    "    falsy stops the batch and raises, every output row being indeterminate then.\n"       //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None\n"      //
    "    skips reporting.\n"                                                                   //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> from array import array\n"                                                          //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=4)\n"                                 //
    "  >>> index.reserve(1)\n"                                                                 //
    "  >>> query = array('d', [0.1, 0.2, 0.3, 0.4])\n"                                         //
    "  >>> pool = usearch.Executor.for_ram()\n"                                                //
    "  >>> _ = index.add([7], query, executor=pool)\n"                                         //
    "  >>> keys, distances, counts = bytearray(16), bytearray(8), bytearray(8)\n"              //
    "  >>> index.search(query, 1, keys, distances, counts, executor=pool)\n"                   //
    "  >>> int(usearch.Matches(keys, distances, counts)[0][0].key)\n"                          //
    "  7";

static PyObject* DenseIndex_search(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    search_request_t request;
    if (search_request_parse(index, "search", args, positional_args_count, args_names_tuple, 5, 1, &request) != 0)
        return NULL;

    char const* const fields[3] = {"keys", "distances", "counts"};
    usearch_py_buffer_backing_t backings[3];
    void* pointers[3] = {NULL, NULL, NULL};
    if (matches_outputs_borrow(args + 2, fields, request.queries.rows, request.wanted, backings, pointers) != 0) {
        usearch_py_buffer_release(&request.queries_backing);
        return NULL;
    }

    usearch_key_t* const keys = (usearch_key_t*)pointers[0];
    usearch_distance_t* const distances = (usearch_distance_t*)pointers[1];
    size_t* const counts = (size_t*)pointers[2];
    usearch_dense_search_options_t const options = {.expansion_factor = request.expansion_factor};
    size_t const keys_stride = request.wanted * sizeof(usearch_key_t);
    size_t const scores_stride = request.wanted * sizeof(usearch_distance_t);
    usearch_matches_span_t const matches = {keys, keys_stride, distances, scores_stride, counts, request.wanted};
    predicate_state_t predicate_state = {request.predicate, NULL};
    progress_state_t progress_state = {request.progress, NULL};
    usearch_progress_callback_t const progress = progress_registered(&progress_state);
    usearch_dense_traversal_stats_t stats = {0};
    usearch_error_t error = {0};
    PyObject* result = NULL;

    Py_BEGIN_ALLOW_THREADS;
    if (request.predicate)
        usearch_dense_search_if(index->handle, request.executor, &request.queries, &matches, &options,
                                predicate_trampoline, &predicate_state, progress, &progress_state, &stats, &error);
    else
        usearch_dense_search(index->handle, request.executor, &request.queries, &matches, &options, progress,
                             &progress_state, &stats, &error);
    Py_END_ALLOW_THREADS;

    // A callback that raised outranks whatever the walk went on to report, since every
    // decision after the first failure was made on a guess.
    if (predicate_state.raised) {
        Py_XDECREF(progress_state.raised);
        PyErr_SetRaisedException(predicate_state.raised);
    }
    else if (progress_raised(&progress_state) == 0 && usearch_py_raise_on_error(&error) == 0 &&
             usearch_py_stats_overwrite(request.stats_object, &stats) == 0)
        result = Py_NewRef(Py_None);

    matches_outputs_release(backings);
    usearch_py_buffer_release(&request.queries_backing);
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
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=2)\n"         //
    "  >>> index.reserve(2)\n"                                         //
    "  >>> row = array('d', [0.1, 0.2])\n"                             //
    "  >>> pool = usearch.Executor.for_ram()\n"                        //
    "  >>> _ = index.add([7], row, executor=pool)\n"                   //
    "  >>> _ = index.add([7], row, executor=pool)\n"                   //
    "  >>> index.count(7)\n"                                           //
    "  2";

static PyObject* DenseIndex_count(PyObject* self, PyObject* key_object) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;
    usearch_key_t key;
    if (usearch_py_key_from_object(key_object, &key) != 0) return NULL;
    usearch_error_t error = {0};
    size_t const found = usearch_dense_count(index->handle, key, &error);
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
    return PyLong_FromSize_t(found);
}

static int DenseIndex_contains(PyObject* self, PyObject* key_object) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return -1;
    usearch_key_t key;
    if (usearch_py_key_from_object(key_object, &key) != 0) return -1;
    usearch_error_t error = {0};
    size_t const found = usearch_dense_count(index->handle, key, &error);
    if (usearch_py_raise_on_error(&error) != 0) return -1;
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

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("get", positional_args_count,
                                                                            args_names_tuple, 1);
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
    if (dtype_object && usearch_py_scalar_from_object(dtype_object, &scalar_kind) != 0) return NULL;

    usearch_key_t key;
    if (usearch_py_key_from_object(args[0], &key) != 0) return NULL;

    usearch_error_t error = {0};
    size_t const found = usearch_dense_count(index->handle, key, &error);
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
    if (found == 0) Py_RETURN_NONE;

    size_t columns = 0, row_bytes = 0;
    if (index_row_extent(index, scalar_kind, &columns, &row_bytes) != 0) return NULL;

    Py_ssize_t arena_bytes = 0;
    if (usearch_py_matrix_bytes(found, row_bytes, 1, &arena_bytes) != 0) return NULL;
    PyObject* const arena = PyBytes_FromStringAndSize(NULL, arena_bytes);
    if (!arena) return NULL;
    memset(PyBytes_AS_STRING(arena), 0, (size_t)arena_bytes);

    usearch_matrix_span_t const vectors = {PyBytes_AS_STRING(arena), found, columns, row_bytes, scalar_kind};
    size_t written = 0;
    Py_BEGIN_ALLOW_THREADS;
    written = usearch_dense_get(index->handle, key, &vectors, &error);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_error(&error) != 0) {
        Py_DECREF(arena);
        return NULL;
    }
    if (written > found) {
        // The count and the copy are not atomic, but harmlessly so: entries are only ever
        // added, so a stale count under-reports and the caller simply asks again.
        Py_DECREF(arena);
        usearch_py_raise(usearch_error_contended_k, "entries were added under this key mid-copy; retry the read");
        return NULL;
    }

    // Shaped by what was written rather than what was counted, so a short copy never publishes
    // rows the engine did not fill.
    PyObject* const shaped = usearch_py_arena_matrix(arena, written, columns, scalar_kind);
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

static char const doc_DenseIndex_multiply[] =                                                 //
    "multiply(rows, wanted, columns, products, counts, *, executor, expansion_factor=0,\n"    //
    "    dtype=None, stats=None) -> int\n"                                                    //
    "\n"                                                                                      //
    "Approximate sparse product of a matrix against the indexed rows - the linear algebra\n"  //
    "sibling of `search`, reporting every pair the traversal scored rather than only the\n"   //
    "nearest. Pairs arrive in traversal order rather than sorted, and carry no duplicates.\n" //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  rows (buffer): An (R, dimensions) matrix of left-hand rows.\n"                         //
    "  wanted (int): Upper bound on pairs emitted per row - the width of the outputs.\n"      //
    "  columns (buffer): A writable (R, wanted, 16) buffer of indexed-row identities.\n"      //
    "  products (buffer): A writable (R, wanted) buffer of float64 products.\n"               //
    "  counts (buffer): A writable (R,) buffer of unsigned word-sized counts. Each reports\n" //
    "    the true pair count, which may exceed `wanted` - that row was truncated to its\n"    //
    "    nearest pairs. Unlike `search`, a count is therefore not a safe slice bound.\n"      //
    "  executor (Executor): The workers to fan out over; its residency must match.\n"         //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it. A wider\n"     //
    "    factor widens the emitted pattern, being what the walk's breadth decides.\n"         //
    "  dtype (str, optional): Names the left-hand scalar type when the buffer cannot.\n"      //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None\n"     //
    "    skips reporting.\n"                                                                  //
    "Returns:\n"                                                                              //
    "  int: How many rows truncated; zero means every pair the walk scored was written.\n"    //
    "    This is the only ranked verb with a return, because it is the only one with\n"       //
    "    something to report - the retry loop reads as `while index.multiply(...)`.";

static PyObject* DenseIndex_multiply(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    // Structurally `search` minus its two callbacks, neither of which `multiply` takes: a
    // `predicate=` or `progress=` keyword is refused rather than quietly ignored.
    search_request_t request;
    if (search_request_parse(index, "multiply", args, positional_args_count, args_names_tuple, 5, 1, &request) != 0)
        return NULL;
    if (request.predicate || request.progress) {
        usearch_py_buffer_release(&request.queries_backing);
        PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %s",
                     request.predicate ? "predicate" : "progress");
        return NULL;
    }

    char const* const fields[3] = {"columns", "products", "counts"};
    usearch_py_buffer_backing_t backings[3];
    void* pointers[3] = {NULL, NULL, NULL};
    if (matches_outputs_borrow(args + 2, fields, request.queries.rows, request.wanted, backings, pointers) != 0) {
        usearch_py_buffer_release(&request.queries_backing);
        return NULL;
    }

    usearch_key_t* const columns = (usearch_key_t*)pointers[0];
    usearch_distance_t* const scores = (usearch_distance_t*)pointers[1];
    size_t* const counts = (size_t*)pointers[2];
    usearch_dense_search_options_t const options = {.expansion_factor = request.expansion_factor};
    size_t const keys_stride = request.wanted * sizeof(usearch_key_t);
    size_t const scores_stride = request.wanted * sizeof(usearch_distance_t);
    usearch_matches_span_t const products = {columns, keys_stride, scores, scores_stride, counts, request.wanted};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_error_t error = {0};
    PyObject* result = NULL;

    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_multiply(index->handle, request.executor, &request.queries, &products, &options, NULL, NULL, &stats,
                           &error);
    Py_END_ALLOW_THREADS;

    if (usearch_py_raise_on_error(&error) == 0 && usearch_py_stats_overwrite(request.stats_object, &stats) == 0) {
        // One pass over a buffer already in cache, so the caller need not make a second.
        size_t truncated = 0;
        for (size_t row = 0; row != request.queries.rows; ++row) truncated += counts[row] > request.wanted;
        result = PyLong_FromSize_t(truncated);
    }

    matches_outputs_release(backings);
    usearch_py_buffer_release(&request.queries_backing);
    return result;
}

static char const doc_DenseIndex_attend[] =                                                     //
    "attend(queries, values, outputs, *, executor, expansion_factor=0, score_scale=0.0,\n"      //
    "    mass_tolerance=0.0, dtype=None, values_dtype=None, outputs_dtype=None,\n"              //
    "    carry=None, stats=None) -> None\n"                                                     //
    "\n"                                                                                        //
    "Softmax-weighted attention over the indexed keys, folded into the graph walk in one\n"     //
    "pass - no match list is materialized, and each value row is gathered exactly once, at\n"   //
    "the moment its key was scored. Never writes to the graph, so a growing context may\n"      //
    "stream through this while insertions continue.\n"                                          //
    "\n"                                                                                        //
    "The index holds the keys; `values` holds the value rows, addressed by key - an entry's\n"  //
    "key is its row offset into `values`. Token positions already are such keys, so a\n"        //
    "context inserted in position order needs no mapping, and the publication watermark\n"      //
    "becomes the causal mask.\n"                                                                //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  queries (buffer): A (Q, dimensions) matrix; every query head sharing these keys is\n"    //
    "    one more row here.\n"                                                                  //
    "  values (buffer): A (V, value_dimensions) matrix, addressed by key. Its width is the\n"   //
    "    value dimensionality, independent of the key width the index froze.\n"                 //
    "  executor (Executor): The workers to fan out over; its residency must match.\n"           //
    "  outputs (buffer): A writable (Q, value_dimensions) buffer, already divided by its\n"     //
    "    partitions. Must hold exactly the bytes its shape and `outputs_dtype` imply.\n"        //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"               //
    "  score_scale (float, optional): The logit scale, conventionally the inverse square\n"     //
    "    root of the head dimensionality; 0.0 derives it from the frozen geometry.\n"           //
    "  mass_tolerance (float, optional): The residual-mass threshold the walk stops at, the\n"  //
    "    mass-denominated sibling of a search's radius. 0.0 derives it, and does not mean\n"    //
    "    zero residual mass.\n"                                                                 //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.\n"            //
    "  values_dtype (str, optional): The same, for the value rows.\n"                           //
    "  outputs_dtype (str, optional): The scalar type to write `outputs` as.\n"                 //
    "  carry (buffer, optional): A writable (Q, 2) buffer of float32, holding each query's\n"   //
    "    running maximum and the softmax denominator taken against it. The pair recombines\n"   //
    "    this branch with another over the same queries - a dense recent window, or a second\n" //
    "    shard of one context - by partial-softmax rescaling; a maximum without its\n"          //
    "    partition is a number nothing can rescale, which is why they travel as one row.\n"     //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None skips\n" //
    "    reporting.";

static PyObject* DenseIndex_attend(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    usearch_dense_attention_options_t options = {0};
    usearch_scalar_t queries_kind = usearch_scalar_unknown_k, values_kind = usearch_scalar_unknown_k,
                     requested_outputs_kind = usearch_scalar_unknown_k;
    PyObject *queries_dtype = NULL, *values_dtype = NULL, *outputs_dtype = NULL;
    PyObject *carry_object = NULL, *stats_object = NULL;
    usearch_executor_t executor = NULL;

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("attend", positional_args_count,
                                                                            args_names_tuple, 3);
    if (args_names_count < 0) return NULL;

    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "expansion_factor") == 0) {
            if (usearch_py_size_from_object(value, "expansion_factor", &options.expansion_factor) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "score_scale") == 0) {
            double const scale = PyFloat_AsDouble(value);
            if (scale == -1.0 && PyErr_Occurred()) return NULL;
            options.score_scale = (float)scale;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "mass_tolerance") == 0) {
            double const tolerance = PyFloat_AsDouble(value);
            if (tolerance == -1.0 && PyErr_Occurred()) return NULL;
            options.mass_tolerance = (float)tolerance;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { queries_dtype = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "values_dtype") == 0) { values_dtype = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "outputs_dtype") == 0) { outputs_dtype = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "carry") == 0) {
            carry_object = value == Py_None ? NULL : value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "stats") == 0) { stats_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(value, &executor) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    if (executor_named(executor, "attend") != 0) return NULL;
    if (usearch_py_scalar_from_object(queries_dtype, &queries_kind) != 0) return NULL;
    if (usearch_py_scalar_from_object(values_dtype, &values_kind) != 0) return NULL;
    if (usearch_py_scalar_from_object(outputs_dtype, &requested_outputs_kind) != 0) return NULL;

    usearch_matrix_view_t queries, values;
    usearch_py_buffer_backing_t queries_backing, values_backing, outputs_backing, carry_backing;
    void *outputs_data = NULL, *carry = NULL;
    usearch_dense_traversal_stats_t stats = {0};
    usearch_error_t error = {0};
    PyObject* result = NULL;
    int borrowed_values = 0, borrowed_outputs = 0, borrowed_carry = 0;

    if (usearch_py_matrix_from_object(args[0], queries_kind, &queries, &queries_backing) != 0) return NULL;
    // Only the queries answer to the index; the value rows carry their own width.
    if (index_rows_match(index, &queries, "queries") != 0) goto release;
    if (usearch_py_matrix_from_object(args[1], values_kind, &values, &values_backing) != 0) goto release;
    borrowed_values = 1;

    // A rank-two buffer carries its own width, which is a stronger contract than the ABI can
    // express: the value dimensionality is read off the matrix rather than named beside it.
    if (values_backing.buffer.ndim != 2) {
        usearch_py_raise(usearch_error_shape_mismatch_k, "attention values must be a (rows, value_dimensions) matrix");
        goto release;
    }

    usearch_scalar_t const outputs_kind = requested_outputs_kind == usearch_scalar_unknown_k ? values.scalar_kind
                                                                                             : requested_outputs_kind;
    size_t const outputs_row_bytes = usearch_py_scalar_row_bytes(outputs_kind, values.columns);
    Py_ssize_t outputs_bytes = 0;
    if (usearch_py_matrix_bytes(queries.rows, outputs_row_bytes, 1, &outputs_bytes) != 0) goto release;
    if (usearch_py_buffer_from_object(args[2], "outputs", usearch_py_matrix_access_write_k, outputs_bytes,
                                      &outputs_backing, &outputs_data) != 0)
        goto release;
    borrowed_outputs = 1;

    if (carry_object) {
        Py_ssize_t carry_bytes = 0;
        if (usearch_py_matrix_bytes(queries.rows, sizeof(usearch_dense_softmax_state_t), 1, &carry_bytes) != 0)
            goto release;
        if (usearch_py_buffer_from_object(carry_object, "carry", usearch_py_matrix_access_write_k, carry_bytes,
                                          &carry_backing, &carry) != 0)
            goto release;
        borrowed_carry = 1;
    }

    usearch_matrix_span_t const outputs = {outputs_data, queries.rows, values.columns, outputs_row_bytes, outputs_kind};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_attend(index->handle, executor, &queries, &values, &outputs, (usearch_dense_softmax_state_t*)carry,
                         &options, NULL, NULL, &stats, &error);
    Py_END_ALLOW_THREADS;

    if (usearch_py_raise_on_error(&error) != 0) goto release;
    if (usearch_py_stats_overwrite(stats_object, &stats) != 0) goto release;
    result = Py_NewRef(Py_None);

release:
    if (borrowed_carry) usearch_py_buffer_release(&carry_backing);
    if (borrowed_outputs) usearch_py_buffer_release(&outputs_backing);
    if (borrowed_values) usearch_py_buffer_release(&values_backing);
    usearch_py_buffer_release(&queries_backing);
    return result;
}

static char const doc_DenseIndex_maxsim[] =                                                    //
    "maxsim(queries, rows_per_query, wanted, keys, distances, counts, *, executor,\n"          //
    "    expansion_factor=0, dtype=None, stats=None) -> None\n"                                //
    "\n"                                                                                       //
    "Late-interaction scoring of multi-row queries against multi-row entries: every query\n"   //
    "row takes its best match within a key, and those maxima sum to that key's score.\n"       //
    "\n"                                                                                       //
    "A document is one key holding its token rows, which is what `values_per_key` above one\n" //
    "exists for, and the walk stays over individual tokens under a plain inner product - so\n" //
    "an entry's rows need no ordering and two documents need not share a row count. The\n"     //
    "outputs are sized over documents rather than over rows.\n"                                //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  queries (buffer): The token rows of every query laid end to end, an (Q * R,\n"          //
    "    dimensions) matrix.\n"                                                                //
    "  rows_per_query (int): Rows every query occupies; uniform, so ColBERT's padding\n"       //
    "    applies as is. The query row count must be a whole multiple of it.\n"                 //
    "  wanted (int): Upper bound on matches per query - the 'k' over documents, not rows.\n"   //
    "  keys (buffer): A writable (Q, wanted, 16) buffer of document identities.\n"             //
    "  distances (buffer): A writable (Q, wanted) buffer of float64 summed maxima.\n"          //
    "  counts (buffer): A writable (Q,) buffer of unsigned word-sized counts.\n"               //
    "  executor (Executor): The workers to fan out over; its residency must match.\n"          //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it. Widening it\n"  //
    "    narrows the gap to the exact score, which a walk can only under-report.\n"            //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.\n"           //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None\n"      //
    "    skips reporting.";

static PyObject* DenseIndex_maxsim(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    // Structurally `search` minus its two callbacks, neither of which `maxsim` takes: a
    // `predicate=` or `progress=` keyword is refused rather than quietly ignored.
    search_request_t request;
    if (search_request_parse(index, "maxsim", args, positional_args_count, args_names_tuple, 6, 2, &request) != 0)
        return NULL;
    if (request.predicate || request.progress) {
        usearch_py_buffer_release(&request.queries_backing);
        PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %s",
                     request.predicate ? "predicate" : "progress");
        return NULL;
    }

    size_t rows_per_query = 0;
    if (usearch_py_size_from_object(args[1], "rows_per_query", &rows_per_query) != 0) {
        usearch_py_buffer_release(&request.queries_backing);
        return NULL;
    }
    if (rows_per_query == 0 || request.queries.rows % rows_per_query != 0) {
        usearch_py_buffer_release(&request.queries_backing);
        usearch_py_raise(usearch_error_shape_mismatch_k, "the %zu query rows are not a whole multiple of %zu",
                         request.queries.rows, rows_per_query);
        return NULL;
    }

    char const* const fields[3] = {"keys", "distances", "counts"};
    usearch_py_buffer_backing_t backings[3];
    void* pointers[3] = {NULL, NULL, NULL};
    size_t const documents = request.queries.rows / rows_per_query;
    if (matches_outputs_borrow(args + 3, fields, documents, request.wanted, backings, pointers) != 0) {
        usearch_py_buffer_release(&request.queries_backing);
        return NULL;
    }

    usearch_key_t* const keys = (usearch_key_t*)pointers[0];
    usearch_distance_t* const distances = (usearch_distance_t*)pointers[1];
    size_t* const counts = (size_t*)pointers[2];
    usearch_dense_search_options_t const options = {.expansion_factor = request.expansion_factor};
    size_t const keys_stride = request.wanted * sizeof(usearch_key_t);
    size_t const scores_stride = request.wanted * sizeof(usearch_distance_t);
    usearch_matches_span_t const scored = {keys, keys_stride, distances, scores_stride, counts, request.wanted};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_error_t error = {0};
    PyObject* result = NULL;

    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_maxsim(index->handle, request.executor, &request.queries, rows_per_query, &scored, &options, NULL,
                         NULL, &stats, &error);
    Py_END_ALLOW_THREADS;

    if (usearch_py_raise_on_error(&error) == 0 && usearch_py_stats_overwrite(request.stats_object, &stats) == 0)
        result = Py_NewRef(Py_None);

    matches_outputs_release(backings);
    usearch_py_buffer_release(&request.queries_backing);
    return result;
}

#pragma endregion Linear Algebra

#pragma region Composition

static char const doc_DenseIndex_merge[] =                                                     //
    "merge(donor, *, executor, progress=None) -> None\n"                                       //
    "\n"                                                                                       //
    "Absorb `donor` into this index, wiring cross-links via landmark stitching - cheaper\n"    //
    "than re-inserting its vectors one by one. The donor handle is consumed and every\n"       //
    "verb on it afterwards raises. Readers may keep searching throughout.\n"                   //
    "\n"                                                                                       //
    "Cancellation stops between donors, so a donor already absorbed stays absorbed and one\n"  //
    "not yet reached stays a live handle its caller still owns - the count a handler last\n"   //
    "saw is the only way to learn where that boundary fell. This verb takes one donor, so a\n" //
    "cancellation before it is absorbed leaves that handle usable and the merge retryable.\n"  //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  donor (DenseIndex): The index to absorb.\n"                                             //
    "  executor (Executor): The workers to fan out over; its residency must match.\n"          //
    "  progress (callable, optional): Called with (completed, total) donors; returning\n"      //
    "    falsy stops the merge and raises.";

static PyObject* DenseIndex_merge(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                  PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("merge", positional_args_count,
                                                                            args_names_tuple, 1);
    if (args_names_count < 0) return NULL;
    PyObject* const donor_object = args[0];
    progress_state_t progress_state = {NULL, NULL};
    usearch_executor_t executor = NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(args[args_progress], &executor) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "progress") == 0) {
            if (progress_from_object(args[args_progress], &progress_state.callable) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    if (executor_named(executor, "merge") != 0) return NULL;

    if (!PyObject_TypeCheck(donor_object, &DenseIndexType)) {
        PyErr_Format(PyExc_TypeError, "Expected a DenseIndex donor, got %S", Py_TYPE(donor_object));
        return NULL;
    }
    DenseIndex* const donor = (DenseIndex*)donor_object;
    if (donor == index) {
        usearch_py_raise(usearch_error_malformed_input_k, "an index cannot merge into itself");
        return NULL;
    }
    if (index_alive(donor) != 0) return NULL;

    // The library consumes the donor and takes the array mutably, so our claim is surrendered
    // before the call rather than after: a failure that still consumed it must not leave us
    // holding a second owner.
    usearch_dense_index_t donated = donor->handle;
    donor->handle = NULL;

    usearch_error_t error = {0};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_merge(index->handle, executor, &donated, 1, progress_registered(&progress_state), &progress_state,
                        &error);
    Py_END_ALLOW_THREADS;

    // The library nulls the slot of every donor it consumed, so a stopped merge hands the
    // unabsorbed handle back and the wrapper reclaims it rather than leaking it.
    if (error.status == usearch_error_cancelled_k && donated) donor->handle = donated;
    if (progress_raised(&progress_state) != 0) return NULL;
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
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
    size_t const row_bytes = with_vectors ? usearch_py_scalar_row_bytes(self->scalar_kind, self->columns) : 0;

    Py_ssize_t keys_bytes = 0, vectors_bytes = 0;
    if (usearch_py_matrix_bytes(self->batch, 1, sizeof(usearch_key_t), &keys_bytes) != 0) return NULL;
    if (with_vectors && usearch_py_matrix_bytes(self->batch, row_bytes, 1, &vectors_bytes) != 0) return NULL;

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
    usearch_keys_span_t const keys = {(usearch_key_t*)PyBytes_AS_STRING(keys_arena), self->batch};
    usearch_matrix_span_t const vectors = {vectors_arena ? PyBytes_AS_STRING(vectors_arena) : NULL, self->batch,
                                           self->columns, row_bytes, self->scalar_kind};
    Py_BEGIN_ALLOW_THREADS;
    written = usearch_dense_cursor_next(self->handle, keys, vectors_arena ? &vectors : NULL, &error);
    Py_END_ALLOW_THREADS;

    PyObject* result = NULL;
    PyObject* keys_shaped = NULL;
    if (usearch_py_raise_on_error(&error) != 0) goto release;
    if (written == 0) goto release; //! A drained snapshot ends the iteration, with no exception.

    // The views are cast to what the snapshot yielded rather than to what was reserved, so the
    // arenas keep their slack and no reader ever sees it - no resize needed.
    keys_shaped = usearch_py_arena_matrix(keys_arena, written, sizeof(usearch_key_t), usearch_scalar_u8_k);
    if (!keys_shaped) goto release;
    if (!vectors_arena) {
        result = keys_shaped;
        keys_shaped = NULL;
        goto release;
    }

    PyObject* const vectors_shaped = usearch_py_arena_matrix(vectors_arena, written, self->columns, self->scalar_kind);
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
    .tp_new = NULL, //! Returned by `cursor`, never constructed by name.
    .tp_dealloc = (destructor)DenseCursor_dealloc,
    .tp_iter = DenseCursor_iter,
    .tp_iternext = (iternextfunc)DenseCursor_next,
};

static char const doc_DenseIndex_cursor[] =                                                  //
    "cursor(size=1024, *, dtype=None) -> DenseCursor\n"                                      //
    "\n"                                                                                     //
    "Walk a snapshot of the index in batches, yielding (keys, vectors) memoryview pairs\n"   //
    "when `dtype` names a scalar type, and key matrices alone otherwise. The watermark is\n" //
    "captured here, so an entry published afterwards never appears mid-iteration.\n"         //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  size (int, optional): Rows per batch. Defaults to 1024.\n"                            //
    "  dtype (str, optional): Export vectors as this scalar type; None walks keys only.";

static PyObject* DenseIndex_cursor(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
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
    if (positional_args_count == 1 && usearch_py_size_from_object(args[0], "size", &batch) != 0) return NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        if (PyUnicode_CompareWithASCIIString(name, "size") == 0) {
            if (usearch_py_size_from_object(args[args_progress], "size", &batch) != 0) return NULL;
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
    if (usearch_py_scalar_from_object(dtype_object, &scalar_kind) != 0) return NULL;

    // Only a walk that exports vectors needs the geometry, and only then must it be frozen.
    size_t columns = 0, row_bytes = 0;
    if (scalar_kind != usearch_scalar_unknown_k && index_row_extent(index, scalar_kind, &columns, &row_bytes) != 0)
        return NULL;

    usearch_error_t error = {0};
    usearch_dense_cursor_t const handle = usearch_dense_cursor(index->handle, &error);
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
    if (!handle) {
        usearch_py_raise(usearch_error_internal_k, "the library returned no cursor and no error");
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
    PyObject* const cursor = DenseIndex_cursor(self, positional, 1, NULL);
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
        usearch_keys_span_t const keys = {self->keys, usearch_py_table_size_(self->keys)};
        Py_BEGIN_ALLOW_THREADS;
        written = usearch_dense_find_next(self->handle, keys, self->distances, &error);
        Py_END_ALLOW_THREADS;
        if (usearch_py_raise_on_error(&error) != 0) return NULL;
        if (written == 0) {
            self->drained = 1;
            return NULL;
        }
        self->buffered = written;
        self->consumed = 0;
    }
    size_t const position = self->consumed++;
    return usearch_py_match_to_object(self->keys[position], self->distances[position]);
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
    "find(query, *, executor, dtype=None) -> DenseFinder\n"                            //
    "\n"                                                                               //
    "Open a resumable nearest-neighbor stream for one query, yielding Match objects\n" //
    "for as long as the caller keeps pulling.\n"                                       //
    "\n"                                                                               //
    "Args:\n"                                                                          //
    "  query (buffer): One vector of `dimensions` scalars.\n"                          //
    "  executor (Executor): The workers to fan out over; its residency must match.\n"  //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.";

static PyObject* DenseIndex_find(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                 PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("find", positional_args_count,
                                                                            args_names_tuple, 1);
    if (args_names_count < 0) return NULL;
    PyObject* dtype_object = NULL;
    usearch_executor_t executor = NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { dtype_object = args[args_progress]; }
        else if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(args[args_progress], &executor) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    if (executor_named(executor, "find") != 0) return NULL;

    usearch_scalar_t scalar_hint = usearch_scalar_unknown_k;
    if (usearch_py_scalar_from_object(dtype_object, &scalar_hint) != 0) return NULL;

    usearch_matrix_view_t query;
    usearch_py_buffer_backing_t backing;
    if (usearch_py_matrix_from_object(args[0], scalar_hint, &query, &backing) != 0) return NULL;
    if (query.rows != 1) {
        usearch_py_buffer_release(&backing);
        usearch_py_raise(usearch_error_shape_mismatch_k, "a finder takes exactly one query row");
        return NULL;
    }
    if (index_rows_match(index, &query, "query") != 0) {
        usearch_py_buffer_release(&backing);
        return NULL;
    }

    usearch_error_t error = {0};
    usearch_dense_finder_t const handle = usearch_dense_find(index->handle, executor, &query, &error);
    usearch_py_buffer_release(&backing);
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
    if (!handle) {
        usearch_py_raise(usearch_error_internal_k, "the library returned no finder and no error");
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

/** Where a checkpoint lives, as the argument's own type spells it. */
typedef enum checkpoint_source_t {

    /** A @c str , @c bytes , or @c os.PathLike naming a file. */
    checkpoint_source_path_k,

    /** An exported buffer holding one serialized monolith. */
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

/** Borrows a whole checkpoint blob, whatever its length. */
static int checkpoint_borrow(PyObject* object, usearch_py_matrix_access_t access,
                             usearch_py_buffer_backing_t* backing) {
    backing->synthesized_owner = NULL;
    int const flags = PyBUF_C_CONTIGUOUS | (access == usearch_py_matrix_access_write_k ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, &backing->buffer, flags) != 0) {
        return usearch_py_raise_from_cause(PyExc_ValueError, "The checkpoint buffer must be contiguous%s",
                                           access == usearch_py_matrix_access_write_k ? " and writable" : "");
    }
    return 0;
}

/** Walks the two keywords `save` and `load` accept beyond their positional target. */
static int checkpoint_keywords_parse(PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple, Py_ssize_t args_names_count,
                                     usearch_executor_t* executor, progress_state_t* progress_state) {
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(args[args_progress], executor) != 0) return -1;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "progress") == 0) {
            if (progress_from_object(args[args_progress], &progress_state->callable) != 0) return -1;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return -1;
        }
    }
    return 0;
}

static char const doc_DenseIndex_save[] =                                                     //
    "save(target, *, executor, progress=None) -> None\n"                                      //
    "\n"                                                                                      //
    "Write the index to a path, or into a writable buffer of `storage_requirements` bytes.\n" //
    "Demands a mutation-free window; concurrent readers keep searching throughout.\n"         //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  target (str | os.PathLike | buffer): Where the checkpoint goes.\n"                     //
    "  executor (Executor): The workers to fan out over; its residency must match.\n"         //
    "  progress (callable, optional): Called with (completed, total) shards; returning\n"     //
    "    falsy stops the write and raises, leaving a checkpoint no reader accepts and\n"      //
    "    whatever parts it already wrote for the caller to remove.";

static PyObject* DenseIndex_save(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                 PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("save", positional_args_count,
                                                                            args_names_tuple, 1);
    if (args_names_count < 0) return NULL;
    PyObject* const target = args[0];
    progress_state_t progress_state = {NULL, NULL};
    usearch_executor_t executor = NULL;
    if (checkpoint_keywords_parse(args, positional_args_count, args_names_tuple, args_names_count, &executor,
                                  &progress_state) != 0)
        return NULL;

    if (executor_named(executor, "save") != 0) return NULL;

    usearch_progress_callback_t const progress = progress_registered(&progress_state);
    usearch_error_t error = {0};
    if (checkpoint_source(target) == checkpoint_source_buffer_k) {
        usearch_py_buffer_backing_t backing;
        if (checkpoint_borrow(target, usearch_py_matrix_access_write_k, &backing) != 0) return NULL;
        usearch_bytes_span_t const buffer = {(uint8_t*)backing.buffer.buf, (size_t)backing.buffer.len};
        Py_BEGIN_ALLOW_THREADS;
        usearch_dense_save_buffer(index->handle, executor, buffer, progress, &progress_state, &error);
        Py_END_ALLOW_THREADS;
        usearch_py_buffer_release(&backing);
        if (progress_raised(&progress_state) != 0) return NULL;
        if (usearch_py_raise_on_error(&error) != 0) return NULL;
        Py_RETURN_NONE;
    }

    usearch_bytes_view_t path;
    PyObject* decoded = NULL;
    if (usearch_py_path_from_object(target, &path, &decoded) != 0) return NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_save(index->handle, executor, path, progress, &progress_state, &error);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    if (progress_raised(&progress_state) != 0) return NULL;
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
    Py_RETURN_NONE;
}

static char const doc_DenseIndex_load[] =                                                  //
    "load(source, *, executor, progress=None) -> None\n"                                   //
    "\n"                                                                                   //
    "Read a checkpoint into this handle, with the residency deciding what that means:\n"   //
    "materialize into RAM, upload to a device, or open in place on disk. A handle whose\n" //
    "geometry was left at zero freezes it here, from the checkpoint's own headers.\n"      //
    "\n"                                                                                   //
    "Args:\n"                                                                              //
    "  source (str | os.PathLike | buffer): The checkpoint to read.\n"                     //
    "  executor (Executor): The workers to fan out over; its residency must match.\n"      //
    "  progress (callable, optional): Called with (completed, total) shards; returning\n"  //
    "    falsy stops the read and raises, leaving the handle empty and loadable again.";

static PyObject* DenseIndex_load(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                 PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("load", positional_args_count,
                                                                            args_names_tuple, 1);
    if (args_names_count < 0) return NULL;
    PyObject* const source = args[0];
    progress_state_t progress_state = {NULL, NULL};
    usearch_executor_t executor = NULL;
    if (checkpoint_keywords_parse(args, positional_args_count, args_names_tuple, args_names_count, &executor,
                                  &progress_state) != 0)
        return NULL;

    if (executor_named(executor, "load") != 0) return NULL;

    usearch_progress_callback_t const progress = progress_registered(&progress_state);
    usearch_error_t error = {0};
    if (checkpoint_source(source) == checkpoint_source_buffer_k) {
        usearch_py_buffer_backing_t backing;
        if (checkpoint_borrow(source, usearch_py_matrix_access_read_k, &backing) != 0) return NULL;
        usearch_bytes_view_t const buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len};
        Py_BEGIN_ALLOW_THREADS;
        usearch_dense_load_buffer(index->handle, executor, buffer, progress, &progress_state, &error);
        Py_END_ALLOW_THREADS;
        usearch_py_buffer_release(&backing);
        if (progress_raised(&progress_state) != 0) return NULL;
        if (usearch_py_raise_on_error(&error) != 0) return NULL;
        Py_RETURN_NONE;
    }

    usearch_bytes_view_t path;
    PyObject* decoded = NULL;
    if (usearch_py_path_from_object(source, &path, &decoded) != 0) return NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load(index->handle, executor, path, progress, &progress_state, &error);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    if (progress_raised(&progress_state) != 0) return NULL;
    if (usearch_py_raise_on_error(&error) != 0) return NULL;
    Py_RETURN_NONE;
}

static char const doc_DenseIndex_metadata[] =                                               //
    "metadata(source) -> dict\n"                                                            //
    "\n"                                                                                    //
    "Read a checkpoint's geometry and metric without loading it.\n"                         //
    "\n"                                                                                    //
    "The keys are the geometry keywords the checkpoint fixes, so\n"                         //
    "`DenseIndex.for_ram(**DenseIndex.metadata(path))` rebuilds a handle that can read\n"   //
    "it. Residency is never among them - the factory names where the payload lands - and\n" //
    "neither is `values_per_key`, which bounds a live handle rather than the file.\n"       //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  source (str | os.PathLike | buffer): The checkpoint to inspect. A `bytes` is\n"      //
    "    read as a path; wrap a blob in a `memoryview` to have it read as one.\n"           //
    "Returns:\n"                                                                            //
    "  dict: 'dimensions', 'metric', 'quantization', 'connectivity'.";

static PyObject* DenseIndex_metadata(PyObject* type, PyObject* source) {
    usearch_dense_metadata_t metadata = {0};
    usearch_error_t error = {0};

    if (checkpoint_source(source) == checkpoint_source_buffer_k) {
        usearch_py_buffer_backing_t backing;
        if (checkpoint_borrow(source, usearch_py_matrix_access_read_k, &backing) != 0) return NULL;
        usearch_bytes_view_t const buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len};
        Py_BEGIN_ALLOW_THREADS;
        usearch_dense_metadata_buffer(buffer, &metadata, &error);
        Py_END_ALLOW_THREADS;
        usearch_py_buffer_release(&backing);
    }
    else {
        usearch_bytes_view_t path;
        PyObject* decoded = NULL;
        if (usearch_py_path_from_object(source, &path, &decoded) != 0) return NULL;
        Py_BEGIN_ALLOW_THREADS;
        usearch_dense_metadata(path, &metadata, &error);
        Py_END_ALLOW_THREADS;
        Py_DECREF(decoded);
    }
    if (usearch_py_raise_on_error(&error) != 0) return NULL;

    PyObject* const result = PyDict_New();
    if (!result) return NULL;
    if (usearch_py_dict_steal_item(result, "dimensions", PyLong_FromSize_t(metadata.dimensions)) != 0 ||
        usearch_py_dict_steal_item(result, "metric",
                                   PyUnicode_FromString(usearch_dense_metric_name(metadata.metric))) != 0 ||
        usearch_py_dict_steal_item(result, "quantization",
                                   PyUnicode_FromString(usearch_scalar_name(metadata.quantization))) != 0 ||
        usearch_py_dict_steal_item(result, "connectivity", PyLong_FromSize_t(metadata.connectivity)) != 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

#pragma endregion Serialization

#pragma region Registration

static PyMethodDef DenseIndex_methods[] = {
    {"for_ram", (PyCFunction)DenseIndex_for_ram, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_for_ram},    //
    {"for_vram", (PyCFunction)DenseIndex_for_vram, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_for_vram}, //
    {"for_nvme", (PyCFunction)DenseIndex_for_nvme, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_for_nvme}, //
    {"reserve", (PyCFunction)DenseIndex_reserve, METH_O, doc_DenseIndex_reserve},                                    //
    {"clear", (PyCFunction)DenseIndex_clear, METH_NOARGS, doc_DenseIndex_clear},                                     //
    {"add", (PyCFunction)DenseIndex_add, usearch_py_method_flags_k, doc_DenseIndex_add},                             //
    {"refine", (PyCFunction)DenseIndex_refine, usearch_py_method_flags_k, doc_DenseIndex_refine},                    //
    {"health", (PyCFunction)DenseIndex_health, usearch_py_method_flags_k, doc_DenseIndex_health},                    //
    {"search", (PyCFunction)DenseIndex_search, usearch_py_method_flags_k, doc_DenseIndex_search},                    //
    {"count", (PyCFunction)DenseIndex_count, METH_O, doc_DenseIndex_count},                                          //
    {"get", (PyCFunction)DenseIndex_get, usearch_py_method_flags_k, doc_DenseIndex_get},                             //
    {"multiply", (PyCFunction)DenseIndex_multiply, usearch_py_method_flags_k, doc_DenseIndex_multiply},              //
    {"attend", (PyCFunction)DenseIndex_attend, usearch_py_method_flags_k, doc_DenseIndex_attend},                    //
    {"maxsim", (PyCFunction)DenseIndex_maxsim, usearch_py_method_flags_k, doc_DenseIndex_maxsim},                    //
    {"merge", (PyCFunction)DenseIndex_merge, usearch_py_method_flags_k, doc_DenseIndex_merge},                       //
    {"find", (PyCFunction)DenseIndex_find, usearch_py_method_flags_k, doc_DenseIndex_find},                          //
    {"cursor", (PyCFunction)DenseIndex_cursor, usearch_py_method_flags_k, doc_DenseIndex_cursor},                    //
    {"save", (PyCFunction)DenseIndex_save, usearch_py_method_flags_k, doc_DenseIndex_save},                          //
    {"load", (PyCFunction)DenseIndex_load, usearch_py_method_flags_k, doc_DenseIndex_load},                          //
    {"metadata", (PyCFunction)DenseIndex_metadata, METH_O | METH_STATIC, doc_DenseIndex_metadata},                   //
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
