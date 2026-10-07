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

/** An owned index handle, nulled once a merge consumes it. */
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

    /** The index every step passes back to the library, kept alive for the cursor's whole life. */
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
typedef struct DenseSearchStream {
    PyObject ob_base;
    usearch_dense_search_stream_t handle;

    /** The index every pull passes back to the library, kept alive for the stream's whole life. */
    PyObject* index;

    /** The @c Executor every pull runs on, kept alive alongside; NULL pulls on the caller's thread. */
    PyObject* executor;

    usearch_key_t keys[64];
    usearch_distance_t distances[64];
    size_t buffered;
    size_t consumed;
    int drained;
} DenseSearchStream;

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
    "Start an empty one in the residency its rows live in - `for_ram` or `for_vram` - copy\n"  //
    "a checkpoint in with `load_ram` or `load_vram`, or serve one in place with `view_ram`\n"  //
    "or `view_nvme`, which are read-only.\n"                                                   //
    "\n"                                                                                       //
    "The verbs that fan out take an optional `Executor`, which owns the workers and is\n"      //
    "shared across indexes; its residency must match this handle's, and leaving it out runs\n" //
    "the call on the caller's thread. See `usearch.Executor`.\n"                               //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=256, metric='angular')\n"             //
    "  >>> index.capabilities is not None\n"                                                   //
    "  True";

/** The geometry both creation verbs freeze, read before the factory's residency arm. */
typedef struct dense_geometry_t {
    usearch_dense_metric_t metric;
    usearch_scalar_t quantization;
    size_t dimensions;
    size_t connectivity;
    size_t values_per_key;
} dense_geometry_t;

/** Whether a keyword belonged to the shared geometry, to the factory's own arm, or to neither. */
typedef enum keyword_verdict_t {
    keyword_consumed_k,
    keyword_foreign_k,
    keyword_failed_k,
} keyword_verdict_t;

/** Reads one geometry keyword into @p geometry, leaving the residency arm to the caller. */
static keyword_verdict_t geometry_from_keyword(dense_geometry_t* geometry, PyObject* name, PyObject* value) {
    if (PyUnicode_CompareWithASCIIString(name, "dimensions") == 0)
        return usearch_py_size_from_object(value, "dimensions", &geometry->dimensions) == 0 ? keyword_consumed_k
                                                                                            : keyword_failed_k;
    if (PyUnicode_CompareWithASCIIString(name, "metric") == 0) {
        int choice = 0;
        if (usearch_py_choice_from_object(value, "metric", metric_namings, usearch_py_table_size_(metric_namings),
                                          &choice) != 0)
            return keyword_failed_k;
        geometry->metric = (usearch_dense_metric_t)choice;
        return keyword_consumed_k;
    }
    if (PyUnicode_CompareWithASCIIString(name, "quantization") == 0)
        return usearch_py_scalar_from_object(value, &geometry->quantization) == 0 ? keyword_consumed_k
                                                                                  : keyword_failed_k;
    if (PyUnicode_CompareWithASCIIString(name, "connectivity") == 0)
        return usearch_py_size_from_object(value, "connectivity", &geometry->connectivity) == 0 ? keyword_consumed_k
                                                                                                : keyword_failed_k;
    if (PyUnicode_CompareWithASCIIString(name, "values_per_key") == 0)
        return usearch_py_size_from_object(value, "values_per_key", &geometry->values_per_key) == 0 ? keyword_consumed_k
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
static int geometry_from_positionals(dense_geometry_t* geometry, char const* verb, PyObject* const* args,
                                     Py_ssize_t positional_args_count) {
    if (positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "%s() takes at most 3 positional arguments, received %zd", verb,
                     positional_args_count);
        return -1;
    }
    if (positional_args_count >= 1 && usearch_py_size_from_object(args[0], "dimensions", &geometry->dimensions) != 0)
        return -1;
    if (positional_args_count >= 2) {
        int choice = 0;
        if (usearch_py_choice_from_object(args[1], "metric", metric_namings, usearch_py_table_size_(metric_namings),
                                          &choice) != 0)
            return -1;
        geometry->metric = (usearch_dense_metric_t)choice;
    }
    if (positional_args_count >= 3 && usearch_py_scalar_from_object(args[2], &geometry->quantization) != 0) return -1;
    return 0;
}

/** The geometry a factory starts from, before any argument touches it. */
static void geometry_defaults(dense_geometry_t* geometry) {
    memset(geometry, 0, sizeof(*geometry));
    geometry->metric = usearch_dense_metric_angular_k;
    geometry->quantization = usearch_scalar_bf16_k;
}

/**
 *  @brief Wraps the handle a lifecycle verb produced, or frees it when it failed or the wrapper
 *      cannot be allocated.
 *  @return A new reference, or NULL with an exception set.
 */
static PyObject* index_adopt(PyTypeObject* type, usearch_dense_index_t handle, usearch_outcome_t const* outcome) {
    if (usearch_py_raise_on_outcome(outcome) != 0) {
        usearch_dense_free(handle);
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
    PyErr_SetString(
        PyExc_TypeError,
        "DenseIndex is built through for_ram(), for_vram(), load_ram(), load_vram(), view_ram() or view_nvme()");
    return NULL;
}

static char const doc_DenseIndex_for_ram[] =                                                  //
    "for_ram(dimensions, metric='angular', quantization='bf16', *, connectivity=0,\n"         //
    "    values_per_key=0, memory_domains=None) -> DenseIndex\n"                              //
    "\n"                                                                                      //
    "Starts an empty, mutable graph in host memory, one shard per named NUMA node.\n"         //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  dimensions (int): Scalars per vector.\n"                                               //
    "  metric (str, optional): 'inner', 'angular', 'euclidean', 'hamming', or 'jaccard'.\n"   //
    "  quantization (str, optional): The stored scalar type. Defaults to 'bf16'.\n"           //
    "  connectivity (int, optional): Connections per node; 0 derives it from the geometry.\n" //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"          //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to shard across, so a\n"       //
    "    dual-socket box naming two gets two graphs; None derives every node.\n"              //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> index = usearch.DenseIndex.for_ram(256, 'angular')\n"                              //
    "  >>> index.dimensions\n"                                                                //
    "  256";

static PyObject* DenseIndex_for_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                    PyObject* args_names_tuple) {
    dense_geometry_t geometry;
    geometry_defaults(&geometry);
    if (geometry_from_positionals(&geometry, "for_ram", args, positional_args_count) != 0) return NULL;

    PyObject* memory_domains_object = Py_None;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        switch (geometry_from_keyword(&geometry, name, value)) {
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
    size_t memory_domains_count = 0;
    if (usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        return NULL;

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_init_ram_t arguments = {
        .index = &handle,
        .outcome = &outcome,
        .metric = geometry.metric,
        .quantization = geometry.quantization,
        .dimensions = geometry.dimensions,
        .connectivity = geometry.connectivity,
        .values_per_key = geometry.values_per_key,
        .ram = {memory_domains, memory_domains_count},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_init_ram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);
    return index_adopt(type, handle, &outcome);
}

static char const doc_DenseIndex_for_vram[] =                                                 //
    "for_vram(dimensions, metric='angular', quantization='bf16', *, connectivity=0,\n"        //
    "    values_per_key=0, devices=None) -> DenseIndex\n"                                     //
    "\n"                                                                                      //
    "Starts an empty, mutable graph on CUDA devices, one shard per named ordinal.\n"          //
    "\n"                                                                                      //
    "Shards are sharded and never replicated, so eight devices host eight slices of one\n"    //
    "graph rather than eight copies. An index outgrowing them fails rather than paging to\n"  //
    "the host, and the answer is to name more devices.\n"                                     //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  dimensions (int): Scalars per vector.\n"                                               //
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
    dense_geometry_t geometry;
    geometry_defaults(&geometry);
    if (geometry_from_positionals(&geometry, "for_vram", args, positional_args_count) != 0) return NULL;

    PyObject* devices_object = Py_None;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        switch (geometry_from_keyword(&geometry, name, value)) {
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
    size_t devices_count = 0;
    if (usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0) return NULL;

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_init_vram_t arguments = {
        .index = &handle,
        .outcome = &outcome,
        .metric = geometry.metric,
        .quantization = geometry.quantization,
        .dimensions = geometry.dimensions,
        .connectivity = geometry.connectivity,
        .values_per_key = geometry.values_per_key,
        .vram = {devices, devices_count},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_init_vram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(devices);
    return index_adopt(type, handle, &outcome);
}

static void DenseIndex_dealloc(DenseIndex* self) {
    // Releasing a large graph walks every shard, which no other thread should wait on.
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_free(self->handle);
    Py_END_ALLOW_THREADS;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** Rejects a handle already consumed by `merge`. */
static int index_alive(DenseIndex const* self) {
    if (self->handle) return 0;
    usearch_py_raise(usearch_status_uninitialized_k, "the index handle was consumed by a merge");
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

#pragma region Keywords

/**
 *  @brief Walks the keywords the lighter verbs share past their fixed positionals; a @c NULL
 *      output, or a @c NULL @p extra_name , names a keyword the verb does not take.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int keywords_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                          PyObject* args_names_tuple, Py_ssize_t required_positional, usearch_executor_t* executor,
                          progress_state_t* progress_state, size_t* values_per_key, char const* extra_name,
                          PyObject** extra) {
    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked(verb, positional_args_count,
                                                                            args_names_tuple, required_positional);
    if (args_names_count < 0) return -1;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (executor && PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(value, executor) != 0) return -1;
        }
        else if (progress_state && PyUnicode_CompareWithASCIIString(name, "progress") == 0) {
            if (progress_from_object(value, &progress_state->callable) != 0) return -1;
        }
        else if (values_per_key && PyUnicode_CompareWithASCIIString(name, "values_per_key") == 0) {
            if (usearch_py_size_from_object(value, "values_per_key", values_per_key) != 0) return -1;
        }
        else if (extra_name && PyUnicode_CompareWithASCIIString(name, extra_name) == 0) { *extra = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return -1;
        }
    }
    return 0;
}

#pragma endregion Keywords

#pragma region Introspection

/**
 *  @brief Reads the handle's frozen geometry and current counters in one call.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int index_describe(DenseIndex* self, usearch_executor_t executor,
                          usearch_dense_index_description_t* description) {
    if (index_alive(self) != 0) return -1;
    usearch_outcome_t outcome = {0};
    usearch_dense_describe_t arguments = {.index = self->handle,
                                          .executor = executor,
                                          .outcome = &outcome,
                                          .description = description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_describe(&arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_raise_on_outcome(&outcome);
}

/** The geometry keywords @p metadata fixes, as a new dict; NULL with an exception set on failure. */
static PyObject* metadata_to_dict(usearch_dense_metadata_t const* metadata) {
    PyObject* const result = PyDict_New();
    if (!result) return NULL;
    if (usearch_py_dict_steal_item(result, "dimensions", PyLong_FromSize_t(metadata->dimensions)) != 0 ||
        usearch_py_dict_steal_item(result, "metric",
                                   PyUnicode_FromString(usearch_dense_metric_name(metadata->metric))) != 0 ||
        usearch_py_dict_steal_item(result, "quantization",
                                   PyUnicode_FromString(usearch_scalar_name(metadata->quantization))) != 0 ||
        usearch_py_dict_steal_item(result, "connectivity", PyLong_FromSize_t(metadata->connectivity)) != 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

static char const doc_DenseIndex_describe[] =                                                 //
    "describe(*, executor=None) -> dict\n"                                                    //
    "\n"                                                                                      //
    "The frozen geometry and the current counters, read in one call. Each counter is read\n"  //
    "once, so under concurrent insertions they may come from slightly different moments.\n"   //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  executor (Executor, optional): The workers the report runs on; None reads on the\n"    //
    "    caller's thread.\n"                                                                  //
    "Returns:\n"                                                                              //
    "  dict: 'dimensions', 'metric', 'quantization' and 'connectivity', as `metadata_read`\n" //
    "    reports them, beside 'size', 'capacity', 'memory_usage' and 'capabilities'.\n"       //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> usearch.DenseIndex.for_ram(256, 'angular').describe()['size']\n"                   //
    "  0";

static PyObject* DenseIndex_describe(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    usearch_executor_t executor = NULL;
    if (keywords_parse("describe", args, positional_args_count, args_names_tuple, 0, &executor, NULL, NULL, NULL,
                       NULL) != 0)
        return NULL;
    usearch_dense_index_description_t description;
    if (index_describe((DenseIndex*)self, executor, &description) != 0) return NULL;
    PyObject* const result = metadata_to_dict(&description.metadata);
    if (!result) return NULL;
    char const* const capabilities = description.capabilities ? description.capabilities : "";
    if (usearch_py_dict_steal_item(result, "size", PyLong_FromSize_t(description.size)) != 0 ||
        usearch_py_dict_steal_item(result, "capacity", PyLong_FromSize_t(description.capacity)) != 0 ||
        usearch_py_dict_steal_item(result, "memory_usage", PyLong_FromSize_t(description.memory_usage)) != 0 ||
        usearch_py_dict_steal_item(result, "capabilities", PyUnicode_FromString(capabilities)) != 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

/** Reads the @c size_t counter at the byte offset @p closure names inside a description. */
static PyObject* DenseIndex_get_counter(DenseIndex* self, void* closure) {
    usearch_dense_index_description_t description;
    if (index_describe(self, NULL, &description) != 0) return NULL;
    return PyLong_FromSize_t(*(size_t const*)((char const*)&description + (size_t)closure));
}

static char const doc_DenseIndex_size[] = "Entries published so far, which `len` reports too.";
static char const doc_DenseIndex_capacity[] = "Entries the index hosts before growing again.";
static char const doc_DenseIndex_dimensions[] = "Scalars per vector.";
static char const doc_DenseIndex_connectivity[] = "Connections per node in the graph.";
static char const doc_DenseIndex_memory_usage[] = "Bytes the handle holds; traversal scratch is the executor's.";
static char const doc_DenseIndex_capabilities[] = "The kernel family dispatch picked for this geometry.";

static PyObject* DenseIndex_get_capabilities(DenseIndex* self, void* closure) {
    usearch_dense_index_description_t description;
    if (index_describe(self, NULL, &description) != 0) return NULL;
    return PyUnicode_FromString(description.capabilities ? description.capabilities : "");
}

/** The rows a borrowed matrix holds, as a verb's batch input reads them. */
static usearch_rows_view_t rows_of(usearch_matrix_view_t const* matrix) {
    usearch_rows_view_t const rows = {matrix->data, matrix->stride, matrix->columns, matrix->scalar};
    return rows;
}

static Py_ssize_t DenseIndex_length(DenseIndex* self) {
    usearch_dense_index_description_t description;
    if (index_describe(self, NULL, &description) != 0) return -1;
    return (Py_ssize_t)description.size;
}

static PyObject* DenseIndex_repr(DenseIndex* self) {
    if (!self->handle) return PyUnicode_FromString("DenseIndex(<consumed>)");
    // No exception is raised here: `tp_repr` runs inside `PyErr_Format("%R", ...)` chains, where
    // raising would destroy the caller's.
    usearch_dense_index_description_t description;
    usearch_outcome_t outcome = {0};
    usearch_dense_describe_t arguments = {.index = self->handle, .outcome = &outcome, .description = &description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_describe(&arguments);
    Py_END_ALLOW_THREADS;
    if (outcome.status != usearch_status_success_k) return PyUnicode_FromString("DenseIndex(<unavailable>)");
    return PyUnicode_FromFormat("DenseIndex(dimensions=%zu, size=%zu)", description.metadata.dimensions,
                                description.size);
}

#pragma endregion Introspection

#pragma region Ingest

static char const doc_DenseIndex_reserve[] =                                                //
    "reserve(capacity, *, executor=None) -> None\n"                                         //
    "\n"                                                                                    //
    "Grow the index to host at least `capacity` entries.\n"                                 //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  capacity (int): Entries the index must accept without growing again.\n"              //
    "  executor (Executor, optional): The workers the growth runs on; its residency must\n" //
    "    match. None grows on the caller's thread.";

static PyObject* DenseIndex_reserve(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                    PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;
    usearch_executor_t executor = NULL;
    if (keywords_parse("reserve", args, positional_args_count, args_names_tuple, 1, &executor, NULL, NULL, NULL,
                       NULL) != 0)
        return NULL;
    size_t capacity = 0;
    if (usearch_py_size_from_object(args[0], "capacity", &capacity) != 0) return NULL;

    usearch_outcome_t outcome = {0};
    usearch_dense_reserve_t arguments = {.index = index->handle,
                                         .executor = executor,
                                         .outcome = &outcome,
                                         .capacity = capacity};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_reserve(&arguments);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    Py_RETURN_NONE;
}

static char const doc_DenseIndex_clear[] =                                                   //
    "clear(*, executor=None) -> None\n"                                                      //
    "\n"                                                                                     //
    "Drop every entry, keeping the capacity and the frozen geometry.\n"                      //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  executor (Executor, optional): The workers the erasure runs on; its residency must\n" //
    "    match. None erases on the caller's thread.";

static PyObject* DenseIndex_clear(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                  PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;
    usearch_executor_t executor = NULL;
    if (keywords_parse("clear", args, positional_args_count, args_names_tuple, 0, &executor, NULL, NULL, NULL, NULL) !=
        0)
        return NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_clear_t arguments = {.index = index->handle, .executor = executor, .outcome = &outcome};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_clear(&arguments);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    Py_RETURN_NONE;
}

static char const doc_DenseIndex_add[] =                                                    //
    "add(keys, vectors, *, executor=None, expansion_factor=0, dtype=None, progress=None,\n" //
    "    stats=None) -> None\n"                                                             //
    "\n"                                                                                    //
    "Insert a batch of vectors under a batch of keys; one vector is a one-row matrix.\n"    //
    "Keys need not be unique: inserting one twice stores two entries, and `count`\n"        //
    "reports both.\n"                                                                       //
    "\n"                                                                                    //
    "A batch is all-or-nothing, a stopped one included, so a rejected or cancelled batch\n" //
    "leaves the index untouched.\n"                                                         //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  keys (Sequence[Key | int] | buffer): One key per row, or an (N, 16) buffer.\n"       //
    "  vectors (buffer): An (N, dimensions) matrix; only the row stride may vary.\n"        //
    "  executor (Executor, optional): The workers to fan out over; its residency must\n"    //
    "    match. None runs the batch on the caller's thread.\n"                              //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"           //
    "  dtype (str, optional): Names the input scalar type when the buffer cannot.\n"        //
    "  progress (callable, optional): Called with (completed, total) rows; returning\n"     //
    "    falsy stops the batch and raises `CancelledError`. Reports are serialized, so\n"   //
    "    the handler needs no locking.\n"                                                   //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None\n"   //
    "    skips reporting.\n"                                                                //
    "\n"                                                                                    //
    "Example:\n"                                                                            //
    "  >>> from array import array\n"                                                       //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=4)\n"                              //
    "  >>> index.reserve(1)\n"                                                              //
    "  >>> index.add([1], array('d', [0.1, 0.2, 0.3, 0.4]))\n"                              //
    "  >>> len(index)\n"                                                                    //
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

    usearch_scalar_t scalar_hint = usearch_scalar_unknown_k;
    if (usearch_py_scalar_from_object(dtype_object, &scalar_hint) != 0) return NULL;

    usearch_matrix_view_t vectors;
    usearch_py_buffer_backing_t backing;
    if (usearch_py_matrix_from_object(vectors_object, scalar_hint, &vectors, &backing) != 0) return NULL;

    usearch_py_key_arena_t arena;
    if (usearch_py_keys_from_object(keys_object, vectors.rows, &arena) != 0) {
        usearch_py_buffer_release(&backing);
        return NULL;
    }

    usearch_dense_traversal_stats_t stats = {0};
    usearch_outcome_t outcome = {0};
    usearch_dense_add_t arguments = {
        .index = index->handle,
        .executor = executor,
        .outcome = &outcome,
        .count = vectors.rows,
        .keys = {arena.keys, sizeof(usearch_key_t)},
        .vectors = rows_of(&vectors),
        .expansion_factor = expansion_factor,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
        .stats = &stats,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_add(&arguments);
    Py_END_ALLOW_THREADS;

    usearch_py_keys_release(&arena);
    usearch_py_buffer_release(&backing);
    // A handler that raised outranks the cancellation its exception turned into.
    if (progress_raised(&progress_state) != 0) return NULL;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    if (usearch_py_stats_overwrite(stats_object, &stats) != 0) return NULL;
    Py_RETURN_NONE;
}

static char const doc_DenseIndex_refine[] =                                                  //
    "refine(*, executor=None, expansion_factor=0, progress=None, stats=None) -> int\n"       //
    "\n"                                                                                     //
    "Re-select the neighbors of every stored entry against the finished graph, improving\n"  //
    "edges that were chosen when the graph around them was still incomplete.\n"              //
    "\n"                                                                                     //
    "A build step rather than a repair: a cheap insertion pass followed by one refinement\n" //
    "reaches a better graph, in less total time, than one expensive insertion pass - so a\n" //
    "wide `expansion_factor` belongs here rather than in `add`. Searches may run\n"          //
    "throughout, and a caller loops until the return falls below whatever change rate\n"     //
    "they consider settled.\n"                                                               //
    "\n"                                                                                     //
    "Stopping is not an error here and raises nothing: re-selecting edges claims no\n"       //
    "identity, so a stopped sweep leaves the graph correct and merely less improved. A\n"    //
    "handler is therefore how a sweep is put under a time budget.\n"                         //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  executor (Executor, optional): The workers to fan out over; its residency must\n"     //
    "    match. None runs the sweep on the caller's thread.\n"                               //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"            //
    "  progress (callable, optional): Called with (completed, total) nodes; returning\n"     //
    "    falsy ends the sweep.\n"                                                            //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None\n"    //
    "    skips reporting.\n"                                                                 //
    "Returns:\n"                                                                             //
    "  int: Nodes whose edges changed; zero means the graph has settled.\n"                  //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> from array import array\n"                                                        //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=4)\n"                               //
    "  >>> index.reserve(1)\n"                                                               //
    "  >>> index.add([1], array('d', [0.1, 0.2, 0.3, 0.4]))\n"                               //
    "  >>> index.refine()\n"                                                                 //
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

    usearch_dense_traversal_stats_t stats = {0};
    usearch_outcome_t outcome = {0};
    size_t changed = 0;
    usearch_dense_refine_t arguments = {
        .index = index->handle,
        .executor = executor,
        .outcome = &outcome,
        .changed = &changed,
        .expansion_factor = expansion_factor,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
        .stats = &stats,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_refine(&arguments);
    Py_END_ALLOW_THREADS;

    if (progress_raised(&progress_state) != 0) return NULL;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    if (usearch_py_stats_overwrite(stats_object, &stats) != 0) return NULL;
    return PyLong_FromSize_t(changed);
}

static char const doc_DenseIndex_health[] =                                                    //
    "health(*, executor=None) -> dict\n"                                                       //
    "\n"                                                                                       //
    "Audit how much of the graph a search can reach, sweeping forward reachability from as\n"  //
    "many landmarks as one search draws. Read-only.\n"                                         //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  executor (Executor, optional): The workers to fan out over; its residency must\n"       //
    "    match. None runs the audit on the caller's thread.\n"                                 //
    "Returns:\n"                                                                               //
    "  dict: 'audited' entries published when the audit started, the 'reachable' ones among\n" //
    "    them, and the 'sweeps' it took to converge.\n"                                        //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> from array import array\n"                                                          //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=4)\n"                                 //
    "  >>> index.reserve(1)\n"                                                                 //
    "  >>> index.add([1], array('d', [0.1, 0.2, 0.3, 0.4]))\n"                                 //
    "  >>> health = index.health()\n"                                                          //
    "  >>> health['reachable'] == health['audited'] == 1\n"                                    //
    "  True";

static PyObject* DenseIndex_health(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    usearch_executor_t executor = NULL;
    if (keywords_parse("health", args, positional_args_count, args_names_tuple, 0, &executor, NULL, NULL, NULL, NULL) !=
        0)
        return NULL;

    usearch_dense_reachability_t reachability = {0};
    usearch_outcome_t outcome = {0};
    usearch_dense_health_t arguments = {.index = index->handle,
                                        .executor = executor,
                                        .outcome = &outcome,
                                        .reachability = &reachability};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_health(&arguments);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;

    PyObject* const result = PyDict_New();
    if (!result) return NULL;
    if (usearch_py_dict_steal_item(result, "audited", PyLong_FromSize_t(reachability.audited)) != 0 ||
        usearch_py_dict_steal_item(result, "reachable", PyLong_FromSize_t(reachability.reachable)) != 0 ||
        usearch_py_dict_steal_item(result, "sweeps", PyLong_FromSize_t(reachability.sweeps)) != 0) {
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

    /** The outputs to reuse, or NULL to return fresh ones. */
    PyObject* out;

    /** The @c Arena to land discovered-size outputs in, or NULL for a fresh one. */
    PyObject* arena;

    /** The workers this call fans out over; NULL runs it on the caller's thread. */
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
    if (state->raised) {
        PyGILState_Release(gil_state);
        return admitted;
    }
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
static int search_request_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                                PyObject* args_names_tuple, Py_ssize_t required_positional, Py_ssize_t wanted_position,
                                search_request_t* request) {
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
        else if (PyUnicode_CompareWithASCIIString(name, "out") == 0) { request->out = value == Py_None ? NULL : value; }
        else if (PyUnicode_CompareWithASCIIString(name, "arena") == 0) {
            request->arena = value == Py_None ? NULL : value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(value, &request->executor) != 0) return -1;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return -1;
        }
    }

    usearch_scalar_t scalar_hint = usearch_scalar_unknown_k;
    if (usearch_py_scalar_from_object(dtype_object, &scalar_hint) != 0) return -1;
    if (usearch_py_matrix_from_object(args[0], scalar_hint, &request->queries, &request->queries_backing) != 0)
        return -1;
    if (usearch_py_size_from_object(args[wanted_position], "wanted", &request->wanted) != 0) {
        usearch_py_buffer_release(&request->queries_backing);
        return -1;
    }
    return 0;
}

/**
 *  @brief Refuses a keyword the parser shares across walks but @p verb does not take, rather than
 *      quietly ignoring it, and releases the request when it does.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int search_request_refuse(search_request_t* request, PyObject* keyword, char const* name) {
    if (!keyword) return 0;
    usearch_py_buffer_release(&request->queries_backing);
    PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %s", name);
    return -1;
}

static char const doc_DenseIndex_search[] =                                                   //
    "search(queries, wanted, *, executor=None, expansion_factor=0, dtype=None,\n"             //
    "    predicate=None, progress=None, stats=None, out=None) -> Matches\n"                   //
    "\n"                                                                                      //
    "Find the `wanted` nearest entries for every query row; one query is a one-row matrix.\n" //
    "Never writes to the graph, so any number of searches may overlap with insertions.\n"     //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  queries (buffer): A (Q, dimensions) matrix of query rows.\n"                           //
    "  wanted (int): Upper bound on matches per query - the 'k' in 'kANN'.\n"                 //
    "  executor (Executor, optional): The workers to fan out over; its residency must\n"      //
    "    match. None runs the batch on the caller's thread.\n"                                //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"             //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.\n"          //
    "  predicate (callable, optional): Admits a Key when it returns truthy. Non-matching\n"   //
    "    entries still route the traversal, so recall survives low selectivity. The\n"        //
    "    callback re-enters Python, so it runs under the GIL and serializes the walk.\n"      //
    "  progress (callable, optional): Called with (completed, total) queries; returning\n"    //
    "    falsy stops the batch and raises `CancelledError`.\n"                                //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None\n"     //
    "    skips reporting.\n"                                                                  //
    "  out (Matches, optional): A writable run of (Q, wanted) to land in and return, so a\n"  //
    "    sweep allocates nothing. Each count bounds its own row; cells past it keep\n"        //
    "    whatever they already held.\n"                                                       //
    "Returns:\n"                                                                              //
    "  Matches: `out`, or a fresh run owning its buffers.\n"                                  //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> from array import array\n"                                                         //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=4)\n"                                //
    "  >>> index.reserve(1)\n"                                                                //
    "  >>> query = array('d', [0.1, 0.2, 0.3, 0.4])\n"                                        //
    "  >>> index.add([7], query)\n"                                                           //
    "  >>> int(index.search(query, 1)[0][0].key)\n"                                           //
    "  7";

static PyObject* DenseIndex_search(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    search_request_t request;
    if (search_request_parse("search", args, positional_args_count, args_names_tuple, 2, 1, &request) != 0 ||
        search_request_refuse(&request, request.arena, "arena") != 0)
        return NULL;

    usearch_matches_span_t matches;
    PyObject* const result = usearch_py_matches_prepare(request.out, request.queries.rows, request.wanted, &matches);
    if (!result) {
        usearch_py_buffer_release(&request.queries_backing);
        return NULL;
    }

    predicate_state_t predicate_state = {request.predicate, NULL};
    progress_state_t progress_state = {request.progress, NULL};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_outcome_t outcome = {0};
    usearch_dense_search_t arguments = {
        .index = index->handle,
        .executor = request.executor,
        .outcome = &outcome,
        .count = request.queries.rows,
        .queries = rows_of(&request.queries),
        .matches = matches,
        .expansion_factor = request.expansion_factor,
        .predicate = request.predicate ? predicate_trampoline : NULL,
        .predicate_state = &predicate_state,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
        .stats = &stats,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_search(&arguments);
    Py_END_ALLOW_THREADS;
    usearch_py_buffer_release(&request.queries_backing);

    // A callback that raised outranks whatever the walk went on to report, since every
    // decision after the first failure was made on a guess.
    if (predicate_state.raised) {
        Py_XDECREF(progress_state.raised);
        PyErr_SetRaisedException(predicate_state.raised);
    }
    else if (progress_raised(&progress_state) == 0 && usearch_py_raise_on_outcome(&outcome) == 0 &&
             usearch_py_stats_overwrite(request.stats_object, &stats) == 0)
        return result;
    Py_DECREF(result);
    return NULL;
}

#pragma endregion Search

#pragma region Retrieval

/** How many entries share the key @p key_object reads as, or -1 with an exception set. */
static Py_ssize_t index_count(DenseIndex* self, PyObject* key_object, usearch_executor_t executor) {
    if (index_alive(self) != 0) return -1;
    usearch_key_t key;
    if (usearch_py_key_from_object(key_object, &key) != 0) return -1;
    size_t entries = 0;
    usearch_outcome_t outcome = {0};
    usearch_dense_count_t arguments = {.index = self->handle,
                                       .executor = executor,
                                       .outcome = &outcome,
                                       .key = key,
                                       .entries = &entries};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_count(&arguments);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return -1;
    return (Py_ssize_t)entries;
}

static char const doc_DenseIndex_count[] =                                                  //
    "count(key, *, executor=None) -> int\n"                                                 //
    "\n"                                                                                    //
    "How many entries share `key`; zero means the index holds none.\n"                      //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  key (Key | int | str): The identity to count.\n"                                     //
    "  executor (Executor, optional): The workers the lookup runs on; its residency must\n" //
    "    match. None looks up on the caller's thread.\n"                                    //
    "\n"                                                                                    //
    "Example:\n"                                                                            //
    "  >>> from array import array\n"                                                       //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=2)\n"                              //
    "  >>> index.reserve(2)\n"                                                              //
    "  >>> row = array('d', [0.1, 0.2])\n"                                                  //
    "  >>> index.add([7], row)\n"                                                           //
    "  >>> index.add([7], row)\n"                                                           //
    "  >>> index.count(7)\n"                                                                //
    "  2";

static PyObject* DenseIndex_count(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                  PyObject* args_names_tuple) {
    usearch_executor_t executor = NULL;
    if (keywords_parse("count", args, positional_args_count, args_names_tuple, 1, &executor, NULL, NULL, NULL, NULL) !=
        0)
        return NULL;
    Py_ssize_t const entries = index_count((DenseIndex*)self, args[0], executor);
    return entries < 0 ? NULL : PyLong_FromSsize_t(entries);
}

static int DenseIndex_contains(PyObject* self, PyObject* key_object) {
    Py_ssize_t const entries = index_count((DenseIndex*)self, key_object, NULL);
    return entries < 0 ? -1 : entries != 0;
}

static char const doc_DenseIndex_get[] =                                                        //
    "get(key, *, executor=None, dtype='f32', arena=None) -> memoryview | None\n"                //
    "\n"                                                                                        //
    "Every vector stored under `key`, in insertion order, or None when the key is absent.\n"    //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  key (Key | int | str): The identity to retrieve.\n"                                      //
    "  executor (Executor, optional): The workers whose thread or stream copies the rows;\n"    //
    "    its residency must match. None copies on the caller's thread.\n"                       //
    "  dtype (str, optional): The scalar type to export as; converted from storage.\n"          //
    "  arena (Arena, optional): The arena to land the rows in, which must export nothing\n"     //
    "    else; None lands them in a fresh one.\n"                                               //
    "Returns:\n"                                                                                //
    "  memoryview: A writable (count, dimensions) view into the arena, keeping it alive, for\n" //
    "    the kinds `memoryview` can format - f64, f32, f16, i8, u8. Sub-byte and mini-float\n"  //
    "    kinds, `bf16` among them, leave as a (count, row_bytes) run of 'B' instead.\n"         //
    "  None: When the index holds no entry under `key`.";

static PyObject* DenseIndex_get(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("get", positional_args_count,
                                                                            args_names_tuple, 1);
    if (args_names_count < 0) return NULL;
    PyObject *dtype_object = NULL, *arena_object = NULL;
    usearch_executor_t executor = NULL;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { dtype_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "arena") == 0) { arena_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(value, &executor) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    usearch_scalar_t scalar_kind = usearch_scalar_unknown_k;
    if (usearch_py_scalar_from_object(dtype_object, &scalar_kind) != 0) return NULL;
    if (scalar_kind == usearch_scalar_unknown_k) scalar_kind = usearch_scalar_f32_k;
    usearch_key_t key;
    if (usearch_py_key_from_object(args[0], &key) != 0) return NULL;
    PyObject* arena = NULL;
    if (usearch_py_arena_claim(arena_object, &arena) != 0) return NULL;

    void* vectors = NULL;
    size_t written = 0, stride = 0;
    usearch_outcome_t outcome = {0};
    usearch_dense_get_t arguments = {
        .index = index->handle,
        .executor = executor,
        .outcome = &outcome,
        .key = key,
        .scalar = scalar_kind,
        .vectors = &vectors,
        .written = &written,
        .stride = &stride,
        .arena = &((Arena*)arena)->handle,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_get(&arguments);
    Py_END_ALLOW_THREADS;

    PyObject* result = NULL;
    if (usearch_py_raise_on_outcome(&outcome) == 0) {
        // The rows are packed, so a format `memoryview` can read splits each into whole scalars.
        char const* const format = usearch_py_scalar_buffer_format(scalar_kind);
        Py_ssize_t const item_bytes = format ? (Py_ssize_t)usearch_py_scalar_row_bytes(scalar_kind, 1) : 1;
        Py_ssize_t const shape[2] = {(Py_ssize_t)written, (Py_ssize_t)stride / item_bytes};
        result = written ? usearch_py_arena_view(arena, vectors, format ? format : "B", item_bytes, shape, 2)
                         : Py_NewRef(Py_None);
    }
    usearch_py_arena_unclaim(arena);
    return result;
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

static char const doc_DenseIndex_multiply[] =                                                   //
    "multiply(rows, wanted, *, executor=None, expansion_factor=0, dtype=None, progress=None,\n" //
    "    stats=None, arena=None) -> dict\n"                                                     //
    "\n"                                                                                        //
    "Approximate sparse product of a matrix against the indexed rows - the linear algebra\n"    //
    "sibling of `search`, reporting every pair the traversal scored rather than only the\n"     //
    "nearest. Pairs arrive in traversal order rather than sorted, and carry no duplicates.\n"   //
    "How many a walk scores is known only once it ends, so the pairs come back ragged.\n"       //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  rows (buffer): An (R, dimensions) matrix of left-hand rows.\n"                           //
    "  wanted (int): Upper bound on pairs kept per row, the first ones its walk scored;\n"      //
    "    0 keeps every pair.\n"                                                                 //
    "  executor (Executor, optional): The workers to fan out over; its residency must\n"        //
    "    match. None runs the batch on the caller's thread.\n"                                  //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it. A wider\n"       //
    "    factor widens the emitted pattern, being what the walk's breadth decides.\n"           //
    "  dtype (str, optional): Names the left-hand scalar type when the buffer cannot.\n"        //
    "  progress (callable, optional): Called with (completed, total) rows; returning falsy\n"   //
    "    stops the batch and raises `CancelledError`.\n"                                        //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None\n"       //
    "    skips reporting.\n"                                                                    //
    "  arena (Arena, optional): The arena to land the pairs in, which must export nothing\n"    //
    "    else; None lands them in a fresh one.\n"                                               //
    "Returns:\n"                                                                                //
    "  dict: 'offsets', an (R + 1,) memoryview of Apache Arrow offsets, row `i` spanning\n"     //
    "    pairs `offsets[i]` up to `offsets[i + 1]`; 'keys', a (pairs, 16) memoryview naming\n"  //
    "    each pair's indexed row; 'scores', a (pairs,) memoryview of float64 products; and\n"   //
    "    'required', the pairs scored before `wanted` cut any. 'keys' and 'scores' view the\n"  //
    "    arena and keep it alive.";

static PyObject* DenseIndex_multiply(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    search_request_t request;
    if (search_request_parse("multiply", args, positional_args_count, args_names_tuple, 2, 1, &request) != 0)
        return NULL;
    if (search_request_refuse(&request, request.predicate, "predicate") != 0 ||
        search_request_refuse(&request, request.out, "out") != 0)
        return NULL;

    size_t const rows = request.queries.rows;
    Py_ssize_t offsets_bytes = 0;
    PyObject *offsets_object = NULL, *arena = NULL;
    if (usearch_py_matrix_bytes(rows + 1, 1, sizeof(size_t), &offsets_bytes) == 0)
        offsets_object = PyByteArray_FromStringAndSize(NULL, offsets_bytes);
    if (!offsets_object || usearch_py_arena_claim(request.arena, &arena) != 0) {
        Py_XDECREF(offsets_object);
        usearch_py_buffer_release(&request.queries_backing);
        return NULL;
    }

    usearch_key_t* keys = NULL;
    usearch_distance_t* scores = NULL;
    size_t written = 0, required = 0;
    progress_state_t progress_state = {request.progress, NULL};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_outcome_t outcome = {0};
    usearch_dense_multiply_t arguments = {
        .index = index->handle,
        .executor = request.executor,
        .outcome = &outcome,
        .count = rows,
        .left = rows_of(&request.queries),
        .offsets = (size_t*)PyByteArray_AS_STRING(offsets_object),
        .keys = &keys,
        .scores = &scores,
        .arena = &((Arena*)arena)->handle,
        .written = &written,
        .required = &required,
        .wanted = request.wanted,
        .expansion_factor = request.expansion_factor,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
        .stats = &stats,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_multiply(&arguments);
    Py_END_ALLOW_THREADS;
    usearch_py_buffer_release(&request.queries_backing);

    PyObject* result = NULL;
    if (progress_raised(&progress_state) != 0 || usearch_py_raise_on_outcome(&outcome) != 0 ||
        usearch_py_stats_overwrite(request.stats_object, &stats) != 0)
        goto release;

    Py_ssize_t const offsets_shape[1] = {(Py_ssize_t)rows + 1};
    Py_ssize_t const keys_shape[2] = {(Py_ssize_t)written, (Py_ssize_t)sizeof(usearch_key_t)};
    Py_ssize_t const scores_shape[1] = {(Py_ssize_t)written};
    result = PyDict_New();
    if (!result) goto release;
    if (usearch_py_dict_steal_item(result, "offsets", usearch_py_shaped_view(offsets_object, "N", offsets_shape, 1)) !=
            0 ||
        usearch_py_dict_steal_item(result, "keys", usearch_py_arena_view(arena, keys, "B", 1, keys_shape, 2)) != 0 ||
        usearch_py_dict_steal_item(
            result, "scores", usearch_py_arena_view(arena, scores, "d", sizeof(usearch_distance_t), scores_shape, 1)) !=
            0 ||
        usearch_py_dict_steal_item(result, "required", PyLong_FromSize_t(required)) != 0)
        Py_CLEAR(result);

release:
    Py_DECREF(offsets_object);
    usearch_py_arena_unclaim(arena);
    return result;
}

static char const doc_DenseIndex_attend[] =                                                       //
    "attend(queries, values, *, executor=None, expansion_factor=0, score_scale=0.0,\n"            //
    "    mass_tolerance=0.0, dtype=None, values_dtype=None, outputs_dtype=None, carry=None,\n"    //
    "    progress=None, stats=None, out=None) -> memoryview | buffer\n"                           //
    "\n"                                                                                          //
    "Softmax-weighted attention over the indexed keys, folded into the graph walk in one\n"       //
    "pass - no match list is materialized, and each value row is gathered exactly once, at\n"     //
    "the moment its key was scored. Never writes to the graph, so a growing context may\n"        //
    "stream through this while insertions continue.\n"                                            //
    "\n"                                                                                          //
    "The index holds the keys; `values` holds the value rows, addressed by key - an entry's\n"    //
    "key is its row offset into `values`. Token positions already are such keys, so a\n"          //
    "context inserted in position order needs no mapping, and the publication watermark\n"        //
    "becomes the causal mask.\n"                                                                  //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  queries (buffer): A (Q, dimensions) matrix; every query head sharing these keys is\n"      //
    "    one more row here.\n"                                                                    //
    "  values (buffer): A (V, value_dimensions) matrix, addressed by key. Its width is the\n"     //
    "    value dimensionality, independent of the key width the index froze.\n"                   //
    "  executor (Executor, optional): The workers to fan out over; its residency must\n"          //
    "    match. None runs the batch on the caller's thread.\n"                                    //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"                 //
    "  score_scale (float, optional): The logit scale, conventionally the inverse square\n"       //
    "    root of the head dimensionality; 0.0 derives it from the frozen geometry.\n"             //
    "  mass_tolerance (float, optional): The residual-mass threshold the walk stops at, the\n"    //
    "    mass-denominated sibling of a search's radius. 0.0 derives it, and does not mean\n"      //
    "    zero residual mass.\n"                                                                   //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.\n"              //
    "  values_dtype (str, optional): The same, for the value rows.\n"                             //
    "  outputs_dtype (str, optional): The scalar type to write the outputs as; defaults to\n"     //
    "    the values' own.\n"                                                                      //
    "  carry (buffer, optional): A writable (Q, 2) buffer of float32, holding each query's\n"     //
    "    running maximum and the softmax denominator taken against it. The pair recombines\n"     //
    "    this branch with another over the same queries - a dense recent window, or a second\n"   //
    "    shard of one context - by partial-softmax rescaling; a maximum without its\n"            //
    "    partition is a number nothing can rescale, which is why they travel as one row.\n"       //
    "  progress (callable, optional): Called with (completed, total) queries; returning\n"        //
    "    falsy stops the batch and raises `CancelledError`.\n"                                    //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None skips\n"   //
    "    reporting.\n"                                                                            //
    "  out (buffer, optional): A writable buffer of exactly the bytes a (Q, value_dimensions)\n"  //
    "    matrix of `outputs_dtype` takes, landed in and returned so a sweep allocates nothing.\n" //
    "Returns:\n"                                                                                  //
    "  memoryview: The (Q, value_dimensions) attended rows, already divided by their\n"           //
    "    partitions, or `out` itself when one was given. Kinds `memoryview` cannot format\n"      //
    "    leave as a (Q, row_bytes) run of 'B'.";

static PyObject* DenseIndex_attend(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    usearch_dense_attend_t arguments = {.index = index->handle};
    usearch_scalar_t queries_kind = usearch_scalar_unknown_k, values_kind = usearch_scalar_unknown_k,
                     requested_outputs_kind = usearch_scalar_unknown_k;
    PyObject *queries_dtype = NULL, *values_dtype = NULL, *outputs_dtype = NULL;
    PyObject *carry_object = NULL, *stats_object = NULL, *out_object = NULL;
    progress_state_t progress_state = {NULL, NULL};

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("attend", positional_args_count,
                                                                            args_names_tuple, 2);
    if (args_names_count < 0) return NULL;

    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "expansion_factor") == 0) {
            if (usearch_py_size_from_object(value, "expansion_factor", &arguments.expansion_factor) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "score_scale") == 0) {
            double const scale = PyFloat_AsDouble(value);
            if (scale == -1.0 && PyErr_Occurred()) return NULL;
            arguments.score_scale = (float)scale;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "mass_tolerance") == 0) {
            double const tolerance = PyFloat_AsDouble(value);
            if (tolerance == -1.0 && PyErr_Occurred()) return NULL;
            arguments.mass_tolerance = (float)tolerance;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { queries_dtype = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "values_dtype") == 0) { values_dtype = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "outputs_dtype") == 0) { outputs_dtype = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "carry") == 0) {
            carry_object = value == Py_None ? NULL : value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "progress") == 0) {
            if (progress_from_object(value, &progress_state.callable) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "stats") == 0) { stats_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "out") == 0) { out_object = value == Py_None ? NULL : value; }
        else if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(value, &arguments.executor) != 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    if (usearch_py_scalar_from_object(queries_dtype, &queries_kind) != 0) return NULL;
    if (usearch_py_scalar_from_object(values_dtype, &values_kind) != 0) return NULL;
    if (usearch_py_scalar_from_object(outputs_dtype, &requested_outputs_kind) != 0) return NULL;

    usearch_matrix_view_t queries, values;
    usearch_py_buffer_backing_t queries_backing, values_backing, outputs_backing, carry_backing;
    void *outputs_data = NULL, *carry = NULL;
    usearch_dense_traversal_stats_t stats = {0};
    usearch_outcome_t outcome = {0};
    PyObject *outputs_object = NULL, *result = NULL;
    int borrowed_values = 0, borrowed_outputs = 0, borrowed_carry = 0;

    if (usearch_py_matrix_from_object(args[0], queries_kind, &queries, &queries_backing) != 0) return NULL;
    if (usearch_py_matrix_from_object(args[1], values_kind, &values, &values_backing) != 0) goto release;
    borrowed_values = 1;

    // A rank-two buffer carries its own width, which is a stronger contract than the ABI can
    // express: the value dimensionality is read off the matrix rather than named beside it.
    if (values_backing.buffer.ndim != 2) {
        usearch_py_raise(usearch_status_shape_mismatch_k, "attention values must be a (rows, value_dimensions) matrix");
        goto release;
    }

    usearch_scalar_t const outputs_kind = requested_outputs_kind == usearch_scalar_unknown_k ? values.scalar
                                                                                             : requested_outputs_kind;
    size_t const outputs_row_bytes = usearch_py_scalar_row_bytes(outputs_kind, values.columns);
    Py_ssize_t outputs_bytes = 0;
    if (usearch_py_matrix_bytes(queries.rows, outputs_row_bytes, 1, &outputs_bytes) != 0) goto release;
    if (out_object) {
        if (usearch_py_buffer_from_object(out_object, "out", usearch_py_matrix_access_write_k, outputs_bytes,
                                          &outputs_backing, &outputs_data) != 0)
            goto release;
        borrowed_outputs = 1;
        outputs_object = Py_NewRef(out_object);
    }
    else {
        outputs_object = PyByteArray_FromStringAndSize(NULL, outputs_bytes);
        if (!outputs_object) goto release;
        outputs_data = PyByteArray_AS_STRING(outputs_object);
    }

    if (carry_object) {
        Py_ssize_t carry_bytes = 0;
        if (usearch_py_matrix_bytes(queries.rows, sizeof(usearch_dense_softmax_state_t), 1, &carry_bytes) != 0)
            goto release;
        if (usearch_py_buffer_from_object(carry_object, "carry", usearch_py_matrix_access_write_k, carry_bytes,
                                          &carry_backing, &carry) != 0)
            goto release;
        borrowed_carry = 1;
    }

    arguments.outcome = &outcome;
    arguments.count = queries.rows;
    arguments.queries = rows_of(&queries);
    arguments.values = values;
    arguments.outputs = (usearch_rows_span_t) {outputs_data, outputs_row_bytes, values.columns, queries.rows,
                                               outputs_kind};
    arguments.carry = (usearch_dense_softmax_state_t*)carry;
    arguments.progress = progress_registered(&progress_state);
    arguments.progress_state = &progress_state;
    arguments.stats = &stats;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_attend(&arguments);
    Py_END_ALLOW_THREADS;

    if (progress_raised(&progress_state) != 0) goto release;
    if (usearch_py_raise_on_outcome(&outcome) != 0) goto release;
    if (usearch_py_stats_overwrite(stats_object, &stats) != 0) goto release;
    result = out_object ? Py_NewRef(out_object)
                        : usearch_py_arena_matrix(outputs_object, queries.rows, values.columns, outputs_kind);

release:
    Py_XDECREF(outputs_object);
    if (borrowed_carry) usearch_py_buffer_release(&carry_backing);
    if (borrowed_outputs) usearch_py_buffer_release(&outputs_backing);
    if (borrowed_values) usearch_py_buffer_release(&values_backing);
    usearch_py_buffer_release(&queries_backing);
    return result;
}

static char const doc_DenseIndex_maxsim[] =                                                    //
    "maxsim(queries, rows_per_query, wanted, *, executor=None, expansion_factor=0,\n"          //
    "    dtype=None, progress=None, stats=None, out=None) -> Matches\n"                        //
    "\n"                                                                                       //
    "Late-interaction scoring of multi-row queries against multi-row entries: every query\n"   //
    "row takes its best match within a key, and those maxima sum to that key's score.\n"       //
    "\n"                                                                                       //
    "A document is one key holding its token rows, which is what `values_per_key` above one\n" //
    "exists for, and the walk stays over individual tokens under a plain inner product - so\n" //
    "an entry's rows need no ordering and two documents need not share a row count. The\n"     //
    "result is sized over documents rather than over rows.\n"                                  //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  queries (buffer): The token rows of every query laid end to end, an (Q * R,\n"          //
    "    dimensions) matrix.\n"                                                                //
    "  rows_per_query (int): Rows every query occupies; uniform, so ColBERT's padding\n"       //
    "    applies as is. The query row count must be a whole multiple of it.\n"                 //
    "  wanted (int): Upper bound on matches per query - the 'k' over documents, not rows.\n"   //
    "  executor (Executor, optional): The workers to fan out over; its residency must\n"       //
    "    match. None runs the batch on the caller's thread.\n"                                 //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it. Widening it\n"  //
    "    narrows the gap to the exact score, which a walk can only under-report.\n"            //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.\n"           //
    "  progress (callable, optional): Called with (completed, total) queries; returning\n"     //
    "    falsy stops the batch and raises `CancelledError`.\n"                                 //
    "  stats (Stats, optional): Overwritten with this call's traversal telemetry; None\n"      //
    "    skips reporting.\n"                                                                   //
    "  out (Matches, optional): A writable run of (Q, wanted) to land in and return.\n"        //
    "Returns:\n"                                                                               //
    "  Matches: Document identities and their summed maxima, largest first.";

static PyObject* DenseIndex_maxsim(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    search_request_t request;
    if (search_request_parse("maxsim", args, positional_args_count, args_names_tuple, 3, 2, &request) != 0) return NULL;
    if (search_request_refuse(&request, request.predicate, "predicate") != 0 ||
        search_request_refuse(&request, request.arena, "arena") != 0)
        return NULL;

    size_t rows_per_query = 0;
    if (usearch_py_size_from_object(args[1], "rows_per_query", &rows_per_query) != 0) {
        usearch_py_buffer_release(&request.queries_backing);
        return NULL;
    }
    if (rows_per_query == 0 || request.queries.rows % rows_per_query != 0) {
        usearch_py_buffer_release(&request.queries_backing);
        usearch_py_raise(usearch_status_shape_mismatch_k, "the %zu query rows are not a whole multiple of %zu",
                         request.queries.rows, rows_per_query);
        return NULL;
    }

    size_t const queries_count = request.queries.rows / rows_per_query;
    usearch_matches_span_t matches;
    PyObject* const result = usearch_py_matches_prepare(request.out, queries_count, request.wanted, &matches);
    if (!result) {
        usearch_py_buffer_release(&request.queries_backing);
        return NULL;
    }

    progress_state_t progress_state = {request.progress, NULL};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_outcome_t outcome = {0};
    usearch_dense_maxsim_t arguments = {
        .index = index->handle,
        .executor = request.executor,
        .outcome = &outcome,
        .count = queries_count,
        .queries = rows_of(&request.queries),
        .rows_per_query = rows_per_query,
        .matches = matches,
        .expansion_factor = request.expansion_factor,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
        .stats = &stats,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_maxsim(&arguments);
    Py_END_ALLOW_THREADS;
    usearch_py_buffer_release(&request.queries_backing);

    if (progress_raised(&progress_state) == 0 && usearch_py_raise_on_outcome(&outcome) == 0 &&
        usearch_py_stats_overwrite(request.stats_object, &stats) == 0)
        return result;
    Py_DECREF(result);
    return NULL;
}

#pragma endregion Linear Algebra

#pragma region Composition

static char const doc_DenseIndex_merge[] =                                                    //
    "merge(donor, *, executor=None, progress=None) -> None\n"                                 //
    "\n"                                                                                      //
    "Absorb `donor` into this index, wiring cross-links via landmark stitching - cheaper\n"   //
    "than re-inserting its vectors one by one. The donor handle is consumed and every\n"      //
    "verb on it afterwards raises. Readers may keep searching throughout.\n"                  //
    "\n"                                                                                      //
    "Cancellation stops between donors, so a donor already absorbed stays absorbed and one\n" //
    "not yet reached stays a live handle its caller still owns. This verb takes one donor,\n" //
    "so a merge that fails before absorbing it leaves that handle usable and retryable;\n"    //
    "one with a cursor or search stream still open raises `ContendedError` that way.\n"       //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  donor (DenseIndex): The index to absorb.\n"                                            //
    "  executor (Executor, optional): The workers to fan out over; its residency must\n"      //
    "    match. None runs the merge on the caller's thread.\n"                                //
    "  progress (callable, optional): Called with (completed, total) donors; returning\n"     //
    "    falsy stops the merge and raises `CancelledError`.";

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

    if (!PyObject_TypeCheck(donor_object, &DenseIndexType)) {
        PyErr_Format(PyExc_TypeError, "Expected a DenseIndex donor, got %S", Py_TYPE(donor_object));
        return NULL;
    }
    DenseIndex* const donor = (DenseIndex*)donor_object;
    if (donor == index) {
        usearch_py_raise(usearch_status_invalid_argument_k, "an index cannot merge into itself");
        return NULL;
    }
    if (index_alive(donor) != 0) return NULL;

    // The library consumes the donor and takes the array mutably, so our claim is surrendered
    // before the call rather than after: a failure that still consumed it must not leave us
    // holding a second owner.
    usearch_dense_index_t donated = donor->handle;
    donor->handle = NULL;

    usearch_outcome_t outcome = {0};
    usearch_dense_merge_t arguments = {
        .index = index->handle,
        .executor = executor,
        .outcome = &outcome,
        .count = 1,
        .donors = &donated,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_merge(&arguments);
    Py_END_ALLOW_THREADS;

    // The library nulls the slot of every donor it consumed, so whatever it left there is a
    // handle the wrapper still owns, whichever way the merge ended.
    donor->handle = donated;
    if (progress_raised(&progress_state) != 0) return NULL;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
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

    size_t written = 0;
    usearch_outcome_t outcome = {0};
    usearch_dense_cursor_next_t arguments = {
        .index = ((DenseIndex*)self->index)->handle,
        .outcome = &outcome,
        .cursor = self->handle,
        .keys = {(usearch_key_t*)PyBytes_AS_STRING(keys_arena), sizeof(usearch_key_t), self->batch},
        .vectors = {vectors_arena ? PyBytes_AS_STRING(vectors_arena) : NULL, row_bytes, self->columns, self->batch,
                    self->scalar_kind},
        .written = &written,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_cursor_next(&arguments);
    Py_END_ALLOW_THREADS;

    PyObject* result = NULL;
    PyObject* keys_shaped = NULL;
    if (usearch_py_raise_on_outcome(&outcome) != 0) goto release;
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

static char const doc_DenseCursor[] =                                                    //
    "A snapshot walk over the entries an index held when the cursor opened.\n"           //
    "\n"                                                                                 //
    "Yields (keys, vectors) pairs of memoryviews when opened with vectors, and key\n"    //
    "matrices alone otherwise. Entries added after the cursor opened stay hidden. One\n" //
    "step runs at a time; an overlapping one raises `ContendedError`.";

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

    usearch_scalar_t scalar_kind = usearch_scalar_unknown_k;
    if (usearch_py_scalar_from_object(dtype_object, &scalar_kind) != 0) return NULL;

    // Only a walk that exports vectors sizes rows, and only the frozen width can size them.
    usearch_dense_index_description_t description = {0};
    if (scalar_kind != usearch_scalar_unknown_k && index_describe(index, NULL, &description) != 0) return NULL;

    usearch_dense_cursor_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_cursor_init_t arguments = {.cursor = &handle, .index = index->handle, .outcome = &outcome};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_cursor_init(&arguments);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;

    DenseCursor* const cursor = PyObject_New(DenseCursor, &DenseCursorType);
    if (!cursor) {
        usearch_dense_cursor_free(handle);
        return NULL;
    }
    cursor->handle = handle;
    Py_INCREF(self);
    cursor->index = self;
    cursor->batch = batch;
    cursor->columns = description.metadata.dimensions;
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

static void DenseSearchStream_dealloc(DenseSearchStream* self) {
    if (self->handle) usearch_dense_search_stream_free(self->handle);
    Py_XDECREF(self->executor);
    Py_XDECREF(self->index);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* DenseSearchStream_iter(PyObject* self) {
    Py_INCREF(self);
    return self;
}

static PyObject* DenseSearchStream_next(DenseSearchStream* self) {
    if (self->consumed == self->buffered) {
        if (self->drained) return NULL;
        size_t written = 0;
        usearch_outcome_t outcome = {0};
        usearch_dense_search_stream_next_t arguments = {
            .index = ((DenseIndex*)self->index)->handle,
            .executor = self->executor ? ((Executor*)self->executor)->handle : NULL,
            .outcome = &outcome,
            .stream = self->handle,
            .matches = {self->keys, sizeof(self->keys), self->distances, sizeof(self->distances), &written, 1,
                        usearch_py_table_size_(self->keys)},
        };
        Py_BEGIN_ALLOW_THREADS;
        usearch_dense_search_stream_next(&arguments);
        Py_END_ALLOW_THREADS;
        if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
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

static char const doc_DenseSearchStream[] =                                               //
    "A resumable best-first stream of matches for one query, yielding Match objects.\n"   //
    "\n"                                                                                  //
    "Ordering beyond the expansion horizon is best-effort nearest, not globally exact.\n" //
    "One pull runs at a time; an overlapping one raises `ContendedError`.";

PyTypeObject DenseSearchStreamType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseSearchStream",
    .tp_doc = doc_DenseSearchStream,
    .tp_basicsize = sizeof(DenseSearchStream),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, //! Returned by `search_stream`, never constructed by name.
    .tp_dealloc = (destructor)DenseSearchStream_dealloc,
    .tp_iter = DenseSearchStream_iter,
    .tp_iternext = (iternextfunc)DenseSearchStream_next,
};

static char const doc_DenseIndex_search_stream[] =                                         //
    "search_stream(query, *, executor=None, expansion_factor=0, dtype=None)\n"             //
    "    -> DenseSearchStream\n"                                                           //
    "\n"                                                                                   //
    "Open a resumable nearest-neighbor stream for one query, yielding Match objects for\n" //
    "as long as the caller keeps pulling.\n"                                               //
    "\n"                                                                                   //
    "Args:\n"                                                                              //
    "  query (buffer): One vector of `dimensions` scalars.\n"                              //
    "  executor (Executor, optional): The workers every pull runs on, kept alive by the\n" //
    "    stream; its residency must match. None pulls on the caller's thread.\n"           //
    "  expansion_factor (int, optional): Candidate-pool breadth; 0 derives it.\n"          //
    "  dtype (str, optional): Names the query scalar type when the buffer cannot.";

static PyObject* DenseIndex_search_stream(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                          PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    Py_ssize_t const args_names_count = usearch_py_args_names_count_checked("search_stream", positional_args_count,
                                                                            args_names_tuple, 1);
    if (args_names_count < 0) return NULL;
    PyObject *dtype_object = NULL, *executor_object = Py_None;
    usearch_executor_t executor = NULL;
    size_t expansion_factor = 0;
    for (Py_ssize_t names_progress = 0, args_progress = positional_args_count; names_progress != args_names_count;
         ++args_progress, ++names_progress) {
        PyObject* const name = PyTuple_GET_ITEM(args_names_tuple, names_progress);
        PyObject* const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) { dtype_object = value; }
        else if (PyUnicode_CompareWithASCIIString(name, "expansion_factor") == 0) {
            if (usearch_py_size_from_object(value, "expansion_factor", &expansion_factor) != 0) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "executor") == 0) {
            if (usearch_py_executor_from_object(value, &executor) != 0) return NULL;
            executor_object = value;
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    usearch_scalar_t scalar_hint = usearch_scalar_unknown_k;
    if (usearch_py_scalar_from_object(dtype_object, &scalar_hint) != 0) return NULL;

    usearch_matrix_view_t query;
    usearch_py_buffer_backing_t backing;
    if (usearch_py_matrix_from_object(args[0], scalar_hint, &query, &backing) != 0) return NULL;
    if (query.rows != 1) {
        usearch_py_buffer_release(&backing);
        usearch_py_raise(usearch_status_shape_mismatch_k, "a search stream takes exactly one query row");
        return NULL;
    }

    usearch_dense_search_stream_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_search_stream_init_t arguments = {
        .stream = &handle,
        .index = index->handle,
        .outcome = &outcome,
        .query = rows_of(&query),
        .expansion_factor = expansion_factor,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_search_stream_init(&arguments);
    Py_END_ALLOW_THREADS;
    usearch_py_buffer_release(&backing);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;

    DenseSearchStream* const stream = PyObject_New(DenseSearchStream, &DenseSearchStreamType);
    if (!stream) {
        usearch_dense_search_stream_free(handle);
        return NULL;
    }
    stream->handle = handle;
    stream->index = Py_NewRef(self);
    stream->executor = executor ? Py_NewRef(executor_object) : NULL;
    stream->buffered = 0;
    stream->consumed = 0;
    stream->drained = 0;
    return (PyObject*)stream;
}

#pragma endregion Iteration

#pragma region Serialization

/** Borrows a whole checkpoint blob, whatever its length. */
static int checkpoint_borrow(PyObject* object, usearch_py_buffer_backing_t* backing) {
    backing->synthesized_owner = NULL;
    if (PyObject_GetBuffer(object, &backing->buffer, PyBUF_C_CONTIGUOUS) != 0)
        return usearch_py_raise_from_cause(PyExc_ValueError, "The checkpoint buffer must be contiguous");
    return 0;
}

/** Adopts the handle a checkpoint verb produced, a handler's own exception outranking the outcome. */
static PyObject* index_loaded(PyTypeObject* type, usearch_dense_index_t handle, usearch_outcome_t const* outcome,
                              progress_state_t* progress_state) {
    if (progress_raised(progress_state) != 0) {
        usearch_dense_free(handle);
        return NULL;
    }
    return index_adopt(type, handle, outcome);
}

static char const doc_DenseIndex_save[] =                                                  //
    "save(path, *, executor=None, progress=None) -> None\n"                                //
    "\n"                                                                                   //
    "Write the index under `path`, in the layout its extension picks: one `.safegraphs`\n" //
    "monolith, or a `.safegraphs.json` manifest beside one part per shard. Demands a\n"    //
    "mutation-free window; concurrent readers keep searching throughout.\n"                //
    "\n"                                                                                   //
    "Args:\n"                                                                              //
    "  path (str | bytes | os.PathLike): Where the checkpoint goes.\n"                     //
    "  executor (Executor, optional): The workers to fan out over; its residency must\n"   //
    "    match. None writes on the caller's thread.\n"                                     //
    "  progress (callable, optional): Called with (completed, total) shards; returning\n"  //
    "    falsy stops the write and raises `CancelledError`, leaving a checkpoint no\n"     //
    "    reader accepts and whatever parts it already wrote for the caller to remove.";

static PyObject* DenseIndex_save(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                 PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    usearch_executor_t executor = NULL;
    progress_state_t progress_state = {NULL, NULL};
    if (keywords_parse("save", args, positional_args_count, args_names_tuple, 1, &executor, &progress_state, NULL, NULL,
                       NULL) != 0)
        return NULL;

    usearch_bytes_view_t path;
    PyObject* decoded = NULL;
    if (usearch_py_path_from_object(args[0], &path, &decoded) != 0) return NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_save_t arguments = {
        .index = index->handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_save(&arguments);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    if (progress_raised(&progress_state) != 0) return NULL;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    Py_RETURN_NONE;
}

static char const doc_DenseIndex_save_buffer[] =                                               //
    "save_buffer(*, executor=None, progress=None, arena=None) -> memoryview\n"                 //
    "\n"                                                                                       //
    "Serialize the index as one monolith of every shard, the very bytes `save` writes to\n"    //
    "a `.safegraphs` path.\n"                                                                  //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  executor (Executor, optional): The workers to fan out over; its residency must\n"       //
    "    match. None writes on the caller's thread.\n"                                         //
    "  progress (callable, optional): Called with (completed, total) bytes; returning\n"       //
    "    falsy stops the write and raises `CancelledError`.\n"                                 //
    "  arena (Arena, optional): The arena to land the monolith in, which must export\n"        //
    "    nothing else; None lands it in a fresh one.\n"                                        //
    "Returns:\n"                                                                               //
    "  memoryview: The monolith's bytes, viewed in place in the arena and keeping it alive.\n" //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> index = usearch.DenseIndex.for_ram(dimensions=4)\n"                                 //
    "  >>> checkpoint = index.save_buffer()\n"                                                 //
    "  >>> usearch.DenseIndex.metadata_read_buffer(checkpoint)['dimensions']\n"                //
    "  4";

static PyObject* DenseIndex_save_buffer(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                        PyObject* args_names_tuple) {
    DenseIndex* const index = (DenseIndex*)self;
    if (index_alive(index) != 0) return NULL;

    usearch_executor_t executor = NULL;
    progress_state_t progress_state = {NULL, NULL};
    PyObject *arena_object = NULL, *arena = NULL;
    if (keywords_parse("save_buffer", args, positional_args_count, args_names_tuple, 0, &executor, &progress_state,
                       NULL, "arena", &arena_object) != 0 ||
        usearch_py_arena_claim(arena_object, &arena) != 0)
        return NULL;

    uint8_t* bytes = NULL;
    size_t length = 0;
    usearch_outcome_t outcome = {0};
    usearch_dense_save_buffer_t arguments = {
        .index = index->handle,
        .executor = executor,
        .outcome = &outcome,
        .bytes = &bytes,
        .length = &length,
        .arena = &((Arena*)arena)->handle,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_save_buffer(&arguments);
    Py_END_ALLOW_THREADS;

    PyObject* result = NULL;
    Py_ssize_t const shape[1] = {(Py_ssize_t)length};
    if (progress_raised(&progress_state) == 0 && usearch_py_raise_on_outcome(&outcome) == 0)
        result = usearch_py_arena_view(arena, bytes, "B", 1, shape, 1);
    usearch_py_arena_unclaim(arena);
    return result;
}

static char const doc_DenseIndex_load_ram[] =                                                //
    "load_ram(path, *, executor=None, memory_domains=None, values_per_key=0,\n"              //
    "    progress=None) -> DenseIndex\n"                                                     //
    "\n"                                                                                     //
    "Copy a checkpoint into a new, mutable index in host memory, its geometry read from\n"   //
    "the checkpoint's own headers. Name the `.safegraphs.json` manifest, never one of the\n" //
    "parts it enumerates; part `p` lands whole on shard `p` modulo the shard count.\n"       //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"              //
    "  executor (Executor, optional): The workers the read fans out over. None reads on\n"   //
    "    the caller's thread.\n"                                                             //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place the parts on;\n"     //
    "    None derives every node.\n"                                                         //
    "  values_per_key (int, optional): Entries sharing one key, which no checkpoint\n"       //
    "    records; 0 derives to one.\n"                                                       //
    "  progress (callable, optional): Called with (completed, total) shards; returning\n"    //
    "    falsy stops the read and raises `CancelledError`, creating no index.";

static PyObject* DenseIndex_load_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    usearch_executor_t executor = NULL;
    progress_state_t progress_state = {NULL, NULL};
    size_t values_per_key = 0;
    PyObject* memory_domains_object = Py_None;
    if (keywords_parse("load_ram", args, positional_args_count, args_names_tuple, 1, &executor, &progress_state,
                       &values_per_key, "memory_domains", &memory_domains_object) != 0)
        return NULL;

    usearch_bytes_view_t path;
    PyObject* decoded = NULL;
    if (usearch_py_path_from_object(args[0], &path, &decoded) != 0) return NULL;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    if (usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0) {
        Py_DECREF(decoded);
        return NULL;
    }

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_load_ram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .ram = {memory_domains, memory_domains_count},
        .values_per_key = values_per_key,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_ram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);
    Py_DECREF(decoded);
    return index_loaded(type, handle, &outcome, &progress_state);
}

static char const doc_DenseIndex_load_ram_buffer[] =                                        //
    "load_ram_buffer(buffer, *, executor=None, memory_domains=None, values_per_key=0,\n"    //
    "    progress=None) -> DenseIndex\n"                                                    //
    "\n"                                                                                    //
    "Copy a monolith out of a buffer into a new, mutable index in host memory, as\n"        //
    "`load_ram` does from a path; the monolith lands whole on the first memory domain.\n"   //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  buffer (buffer): One serialized monolith, such as `save_buffer` returns.\n"          //
    "  executor (Executor, optional): The workers the copy fans out over. None copies on\n" //
    "    the caller's thread.\n"                                                            //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place shards on.\n"       //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"        //
    "  progress (callable, optional): Called with (completed, total) bytes; returning\n"    //
    "    falsy stops the read and raises `CancelledError`, creating no index.";

static PyObject* DenseIndex_load_ram_buffer(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                            PyObject* args_names_tuple) {
    usearch_executor_t executor = NULL;
    progress_state_t progress_state = {NULL, NULL};
    size_t values_per_key = 0;
    PyObject* memory_domains_object = Py_None;
    if (keywords_parse("load_ram_buffer", args, positional_args_count, args_names_tuple, 1, &executor, &progress_state,
                       &values_per_key, "memory_domains", &memory_domains_object) != 0)
        return NULL;

    usearch_py_buffer_backing_t backing;
    if (checkpoint_borrow(args[0], &backing) != 0) return NULL;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    if (usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0) {
        usearch_py_buffer_release(&backing);
        return NULL;
    }

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_load_ram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .ram = {memory_domains, memory_domains_count},
        .values_per_key = values_per_key,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_ram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);
    usearch_py_buffer_release(&backing);
    return index_loaded(type, handle, &outcome, &progress_state);
}

static char const doc_DenseIndex_load_vram[] =                                                //
    "load_vram(path, *, executor=None, devices=None, values_per_key=0, progress=None)\n"      //
    "    -> DenseIndex\n"                                                                     //
    "\n"                                                                                      //
    "Copy a checkpoint into a new, mutable index on CUDA devices, with the placement rules\n" //
    "of `load_ram`: part `p` lands whole on device `p` modulo the device count.\n"            //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"               //
    "  executor (Executor, optional): The streams the upload runs on. None uploads on the\n"  //
    "    handle's devices from the caller's thread.\n"                                        //
    "  devices (Sequence[int], optional): CUDA ordinals to place the parts on.\n"             //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"          //
    "  progress (callable, optional): Called with (completed, total) shards; returning\n"     //
    "    falsy stops the read and raises `CancelledError`, creating no index.";

static PyObject* DenseIndex_load_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    usearch_executor_t executor = NULL;
    progress_state_t progress_state = {NULL, NULL};
    size_t values_per_key = 0;
    PyObject* devices_object = Py_None;
    if (keywords_parse("load_vram", args, positional_args_count, args_names_tuple, 1, &executor, &progress_state,
                       &values_per_key, "devices", &devices_object) != 0)
        return NULL;

    usearch_bytes_view_t path;
    PyObject* decoded = NULL;
    if (usearch_py_path_from_object(args[0], &path, &decoded) != 0) return NULL;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    if (usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0) {
        Py_DECREF(decoded);
        return NULL;
    }

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_load_vram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .vram = {devices, devices_count},
        .values_per_key = values_per_key,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_vram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(devices);
    Py_DECREF(decoded);
    return index_loaded(type, handle, &outcome, &progress_state);
}

static char const doc_DenseIndex_load_vram_buffer[] =                                        //
    "load_vram_buffer(buffer, *, executor=None, devices=None, values_per_key=0,\n"           //
    "    progress=None) -> DenseIndex\n"                                                     //
    "\n"                                                                                     //
    "Copy a monolith out of a host or device buffer into a new, mutable index on CUDA\n"     //
    "devices; the monolith lands whole on the first device.\n"                               //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  buffer (buffer): One serialized monolith.\n"                                          //
    "  executor (Executor, optional): The streams the upload runs on. None uploads on the\n" //
    "    caller's thread.\n"                                                                 //
    "  devices (Sequence[int], optional): CUDA ordinals to place shards on.\n"               //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"         //
    "  progress (callable, optional): Called with (completed, total) bytes; returning\n"     //
    "    falsy stops the read and raises `CancelledError`, creating no index.";

static PyObject* DenseIndex_load_vram_buffer(PyTypeObject* type, PyObject* const* args,
                                             Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    usearch_executor_t executor = NULL;
    progress_state_t progress_state = {NULL, NULL};
    size_t values_per_key = 0;
    PyObject* devices_object = Py_None;
    if (keywords_parse("load_vram_buffer", args, positional_args_count, args_names_tuple, 1, &executor, &progress_state,
                       &values_per_key, "devices", &devices_object) != 0)
        return NULL;

    usearch_py_buffer_backing_t backing;
    if (checkpoint_borrow(args[0], &backing) != 0) return NULL;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    if (usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0) {
        usearch_py_buffer_release(&backing);
        return NULL;
    }

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_load_vram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .vram = {devices, devices_count},
        .values_per_key = values_per_key,
        .progress = progress_registered(&progress_state),
        .progress_state = &progress_state,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_vram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(devices);
    usearch_py_buffer_release(&backing);
    return index_loaded(type, handle, &outcome, &progress_state);
}

static char const doc_DenseIndex_view_ram[] =                                               //
    "view_ram(path) -> DenseIndex\n"                                                        //
    "\n"                                                                                    //
    "Serve a checkpoint in place through a memory mapping, as a read-only index: opening\n" //
    "costs the headers alone, and a traversal pays only for the pages it touches. Every\n"  //
    "mutating verb raises `ReadOnlyError`, and the files must stay unmodified until the\n"  //
    "index is gone.\n"                                                                      //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.";

static PyObject* DenseIndex_view_ram(PyTypeObject* type, PyObject* path_object) {
    usearch_bytes_view_t path;
    PyObject* decoded = NULL;
    if (usearch_py_path_from_object(path_object, &path, &decoded) != 0) return NULL;
    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_view_ram_t arguments = {.index = &handle, .outcome = &outcome, .path = path};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_view_ram(&arguments);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    return index_adopt(type, handle, &outcome);
}

static char const doc_DenseIndex_view_nvme[] =                                                //
    "view_nvme(path, *, executor=None, paths=None) -> DenseIndex\n"                           //
    "\n"                                                                                      //
    "Serve a checkpoint in place from NVMe drives, read through the executor's submission\n"  //
    "rings, as a read-only index. External-memory graphs are never modified in place, so\n"   //
    "bigger ones come from merging shards built in RAM.\n"                                    //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"               //
    "  executor (Executor, optional): The NVMe workers whose rings read the headers.\n"       //
    "  paths (Sequence[str | bytes | PathLike], optional): Devices or directories to serve\n" //
    "    the parts from; None serves each where its manifest points. A path may name a\n"     //
    "    kernel device, a transport identifier, or a FUSE mount.";

static PyObject* DenseIndex_view_nvme(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    usearch_executor_t executor = NULL;
    PyObject* paths_object = Py_None;
    if (keywords_parse("view_nvme", args, positional_args_count, args_names_tuple, 1, &executor, NULL, NULL, "paths",
                       &paths_object) != 0)
        return NULL;

    usearch_bytes_view_t path;
    PyObject* decoded = NULL;
    if (usearch_py_path_from_object(args[0], &path, &decoded) != 0) return NULL;
    usearch_py_paths_arena_t paths;
    if (usearch_py_paths_from_object(paths_object, &paths) != 0) {
        Py_DECREF(decoded);
        return NULL;
    }

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_view_nvme_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .nvme = {paths.views, paths.count},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_view_nvme(&arguments);
    Py_END_ALLOW_THREADS;
    usearch_py_paths_release(&paths);
    Py_DECREF(decoded);
    return index_adopt(type, handle, &outcome);
}

static char const doc_DenseIndex_metadata_read[] =                                           //
    "metadata_read(path, *, executor=None) -> dict\n"                                        //
    "\n"                                                                                     //
    "Read a checkpoint's geometry and metric without loading it.\n"                          //
    "\n"                                                                                     //
    "The keys are the geometry keywords the checkpoint fixes, so\n"                          //
    "`DenseIndex.for_ram(**DenseIndex.metadata_read(path))` starts an empty handle shaped\n" //
    "like it. Residency is never among them - the factory names where the payload lands -\n" //
    "and neither is `values_per_key`, which bounds a live handle rather than the file.\n"    //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  path (str | bytes | os.PathLike): The checkpoint to inspect.\n"                       //
    "  executor (Executor, optional): The workers the read runs on; None reads on the\n"     //
    "    caller's thread.\n"                                                                 //
    "Returns:\n"                                                                             //
    "  dict: 'dimensions', 'metric', 'quantization', 'connectivity'.";

static PyObject* DenseIndex_metadata_read(PyObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                          PyObject* args_names_tuple) {
    usearch_executor_t executor = NULL;
    if (keywords_parse("metadata_read", args, positional_args_count, args_names_tuple, 1, &executor, NULL, NULL, NULL,
                       NULL) != 0)
        return NULL;
    usearch_bytes_view_t path;
    PyObject* decoded = NULL;
    if (usearch_py_path_from_object(args[0], &path, &decoded) != 0) return NULL;
    usearch_dense_metadata_t metadata = {0};
    usearch_outcome_t outcome = {0};
    usearch_dense_metadata_read_t arguments = {.executor = executor,
                                               .outcome = &outcome,
                                               .path = path,
                                               .metadata = &metadata};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_metadata_read(&arguments);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return metadata_to_dict(&metadata);
}

static char const doc_DenseIndex_metadata_read_buffer[] =                          //
    "metadata_read_buffer(buffer, *, executor=None) -> dict\n"                     //
    "\n"                                                                           //
    "Read a serialized monolith's geometry and metric, as `metadata_read`\n"       //
    "does from a path.\n"                                                          //
    "\n"                                                                           //
    "Args:\n"                                                                      //
    "  buffer (buffer): One serialized monolith, such as `save_buffer` returns.\n" //
    "  executor (Executor, optional): The workers the read runs on; None reads\n"  //
    "    on the caller's thread.\n"                                                //
    "Returns:\n"                                                                   //
    "  dict: 'dimensions', 'metric', 'quantization', 'connectivity'.";

static PyObject* DenseIndex_metadata_read_buffer(PyObject* type, PyObject* const* args,
                                                 Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    usearch_executor_t executor = NULL;
    if (keywords_parse("metadata_read_buffer", args, positional_args_count, args_names_tuple, 1, &executor, NULL, NULL,
                       NULL, NULL) != 0)
        return NULL;
    usearch_py_buffer_backing_t backing;
    if (checkpoint_borrow(args[0], &backing) != 0) return NULL;
    usearch_dense_metadata_t metadata = {0};
    usearch_outcome_t outcome = {0};
    usearch_dense_metadata_read_buffer_t arguments = {
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .metadata = &metadata,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_metadata_read_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    usearch_py_buffer_release(&backing);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return metadata_to_dict(&metadata);
}

#pragma endregion Serialization

#pragma region Registration

static PyMethodDef DenseIndex_methods[] = {
    {"for_ram", (PyCFunction)DenseIndex_for_ram, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_for_ram},    //
    {"for_vram", (PyCFunction)DenseIndex_for_vram, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_for_vram}, //
    {"load_ram", (PyCFunction)DenseIndex_load_ram, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_load_ram}, //
    {"load_ram_buffer", (PyCFunction)DenseIndex_load_ram_buffer, usearch_py_method_flags_k | METH_CLASS,
     doc_DenseIndex_load_ram_buffer}, //
    {"load_vram", (PyCFunction)DenseIndex_load_vram, usearch_py_method_flags_k | METH_CLASS,
     doc_DenseIndex_load_vram}, //
    {"load_vram_buffer", (PyCFunction)DenseIndex_load_vram_buffer, usearch_py_method_flags_k | METH_CLASS,
     doc_DenseIndex_load_vram_buffer},                                                            //
    {"view_ram", (PyCFunction)DenseIndex_view_ram, METH_O | METH_CLASS, doc_DenseIndex_view_ram}, //
    {"view_nvme", (PyCFunction)DenseIndex_view_nvme, usearch_py_method_flags_k | METH_CLASS,
     doc_DenseIndex_view_nvme},                                                                         //
    {"describe", (PyCFunction)DenseIndex_describe, usearch_py_method_flags_k, doc_DenseIndex_describe}, //
    {"reserve", (PyCFunction)DenseIndex_reserve, usearch_py_method_flags_k, doc_DenseIndex_reserve},    //
    {"clear", (PyCFunction)DenseIndex_clear, usearch_py_method_flags_k, doc_DenseIndex_clear},          //
    {"add", (PyCFunction)DenseIndex_add, usearch_py_method_flags_k, doc_DenseIndex_add},                //
    {"refine", (PyCFunction)DenseIndex_refine, usearch_py_method_flags_k, doc_DenseIndex_refine},       //
    {"health", (PyCFunction)DenseIndex_health, usearch_py_method_flags_k, doc_DenseIndex_health},       //
    {"search", (PyCFunction)DenseIndex_search, usearch_py_method_flags_k, doc_DenseIndex_search},       //
    {"count", (PyCFunction)DenseIndex_count, usearch_py_method_flags_k, doc_DenseIndex_count},          //
    {"get", (PyCFunction)DenseIndex_get, usearch_py_method_flags_k, doc_DenseIndex_get},                //
    {"multiply", (PyCFunction)DenseIndex_multiply, usearch_py_method_flags_k, doc_DenseIndex_multiply}, //
    {"attend", (PyCFunction)DenseIndex_attend, usearch_py_method_flags_k, doc_DenseIndex_attend},       //
    {"maxsim", (PyCFunction)DenseIndex_maxsim, usearch_py_method_flags_k, doc_DenseIndex_maxsim},       //
    {"merge", (PyCFunction)DenseIndex_merge, usearch_py_method_flags_k, doc_DenseIndex_merge},          //
    {"search_stream", (PyCFunction)DenseIndex_search_stream, usearch_py_method_flags_k,
     doc_DenseIndex_search_stream},                                                                              //
    {"cursor", (PyCFunction)DenseIndex_cursor, usearch_py_method_flags_k, doc_DenseIndex_cursor},                //
    {"save", (PyCFunction)DenseIndex_save, usearch_py_method_flags_k, doc_DenseIndex_save},                      //
    {"save_buffer", (PyCFunction)DenseIndex_save_buffer, usearch_py_method_flags_k, doc_DenseIndex_save_buffer}, //
    {"metadata_read", (PyCFunction)DenseIndex_metadata_read, usearch_py_method_flags_k | METH_STATIC,
     doc_DenseIndex_metadata_read},
    {"metadata_read_buffer", (PyCFunction)DenseIndex_metadata_read_buffer, usearch_py_method_flags_k | METH_STATIC,
     doc_DenseIndex_metadata_read_buffer},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef DenseIndex_getset[] = {
    {"size", (getter)DenseIndex_get_counter, NULL, doc_DenseIndex_size,
     (void*)offsetof(usearch_dense_index_description_t, size)},
    {"capacity", (getter)DenseIndex_get_counter, NULL, doc_DenseIndex_capacity,
     (void*)offsetof(usearch_dense_index_description_t, capacity)},
    {"dimensions", (getter)DenseIndex_get_counter, NULL, doc_DenseIndex_dimensions,
     (void*)offsetof(usearch_dense_index_description_t, metadata.dimensions)},
    {"connectivity", (getter)DenseIndex_get_counter, NULL, doc_DenseIndex_connectivity,
     (void*)offsetof(usearch_dense_index_description_t, metadata.connectivity)},
    {"memory_usage", (getter)DenseIndex_get_counter, NULL, doc_DenseIndex_memory_usage,
     (void*)offsetof(usearch_dense_index_description_t, memory_usage)},
    {"capabilities", (getter)DenseIndex_get_capabilities, NULL, doc_DenseIndex_capabilities, NULL},
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
