/**
 *  @file python/dense_index.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief CPython vector search over equidimensional dense embeddings.
 *
 *  Every verb is a batch: a one-dimensional input is one row, so a memory-mapped dataset
 *  reaches the index with no reshape and no copy. The breadth of the candidates rides the call
 *  rather than the handle, which lets a recall sweep run against one immutable index with no
 *  mutation between points.
 *
 *  Every verb is written once and reached through three receivers: the index runs it on the
 *  calling thread, an @c On from @c index.on(executor) blocks on that executor, and a @c DeferredOn
 *  from @c index.deferred(executor) submits it and answers a @c Deferred. A verb fills one call
 *  structure that the receiver keeps on the stack or inside the @c Deferred, so the same body serves
 *  every completion. Every call detaches the thread state; the predicate, progress and value-lookup
 *  callbacks re-enter Python from the library's own threads, so their trampolines take the GIL back
 *  for each report.
 */
#include "lib.h"

#pragma region Handles

/** An owned index handle, nulled once another index absorbs it. */
typedef struct DenseIndex {
    PyObject ob_base;
    usearch_dense_index_t handle;
} DenseIndex;

/** A sequential row scan over the entries an index holds, in batches. */
typedef struct DenseCursor {
    PyObject ob_base;

    /** The engine cursor, reading each shard's watermark as it enters that shard; NULL once closed. */
    usearch_dense_cursor_t handle;

    /** The index or @c On that opened the cursor, held, which every step runs through. */
    PyObject* receiver;
} DenseCursor;

/** The rows one cursor step wrote into an arena: keys always, their links when the step asked. */
typedef struct DenseBatch {
    PyObject ob_base;

    /** The (count, 16) @c ArenaBuffer of identities. */
    PyObject* keys;

    /** The (offsets, keys) pair of each row's linked identities, or None for a keys-only step. */
    PyObject* links;
} DenseBatch;

/** A resumable best-first stream of matches for one query. */
typedef struct DenseSearchStream {
    PyObject ob_base;

    /** The engine stream; NULL once drained or closed. */
    usearch_dense_search_stream_t handle;

    /** The index or @c On that opened the stream, held, which every pull runs through. */
    PyObject* receiver;
} DenseSearchStream;

/** Traversal telemetry a caller passes into a walking verb to have overwritten. */
typedef struct DenseTraversalStats {
    PyObject ob_base;
    usearch_traversal_stats_t stats;
} DenseTraversalStats;

/** The handle of @p index, read under its critical section since another index may absorb it from
 *  another thread. */
static usearch_dense_index_t index_handle(DenseIndex* index) {
    usearch_dense_index_t handle = NULL;
    usearch_py_critical_section_begin_((PyObject*)index);
    handle = index->handle;
    usearch_py_critical_section_end_();
    if (!handle) usearch_py_raise(usearch_status_uninitialized_k, "the index handle was absorbed into another index");
    return handle;
}

/**
 *  @brief The live handle @p self addresses and how its verbs run there, refusing a receiver over another modality.
 *  @return The handle, or NULL with an exception set.
 */
static usearch_dense_index_t receiver_from_object(PyObject* self, usearch_py_receiver_t* receiver) {
    if (usearch_py_receiver_from_object(self, receiver) != 0) return NULL;
    if (!Py_IS_TYPE(receiver->index, &DenseIndexType)) {
        PyErr_Format(PyExc_TypeError, "expected a receiver over a DenseIndex, got one over %S",
                     Py_TYPE(receiver->index));
        return NULL;
    }
    return index_handle((DenseIndex*)receiver->index);
}

/** The rows a borrowed matrix holds, as a verb's batch input reads them. */
static usearch_rows_view_t rows_of(usearch_matrix_view_t const* matrix) {
    usearch_rows_view_t const rows = {matrix->data, matrix->stride, matrix->columns, matrix->scalar};
    return rows;
}

/**
 *  @brief A @c DenseVectors over the @p count rows a call wrote packed into the @c Arena @p arena
 *      it claimed.
 *
 *  Kinds a @c memoryview cannot format - the sub-byte and mini-float ones - leave as a
 *  @c (count,row_bytes) run of @c 'B', since a cast to their format would report the wrong extent.
 *
 *  @return A new reference, or NULL with an exception set.
 */
static PyObject* rows_answer(PyObject* arena, usearch_rows_view_t const* rows, size_t count) {
    char const* const format = usearch_py_scalar_buffer_format(rows->scalar);
    size_t const row_bytes = usearch_py_scalar_row_bytes(rows->scalar, rows->columns);
    Py_ssize_t const shape[2] = {(Py_ssize_t)count, format ? (Py_ssize_t)rows->columns : (Py_ssize_t)row_bytes};
    Py_ssize_t const item_bytes = format ? (Py_ssize_t)usearch_py_scalar_row_bytes(rows->scalar, 1) : 1;
    return usearch_py_arena_buffer(&DenseVectorsType, arena, rows->data, format ? format : "B", item_bytes, shape, 2);
}

/** Reads an optional `executor`: NULL or @c None runs on the calling thread. Returns 0 or -1 with an exception set. */
static int executor_optional(PyObject* object, usearch_executor_t* executor) {
    *executor = NULL;
    return !object || object == Py_None ? 0 : usearch_py_executor_from_object(object, executor);
}

/** What a traversal reports beside its answer: progress to its handler, telemetry to a caller's stats. */
typedef struct walk_reports_t {

    /** The progress handler, or an empty callback when the caller passed none. */
    usearch_py_callback_t progress;

    /** The telemetry the library writes. */
    usearch_traversal_stats_t stats;

    /** The @c DenseTraversalStats the telemetry is copied into once the call succeeds, held, or NULL. */
    PyObject* stats_object;
} walk_reports_t;

/** Reads the `progress` and `stats` arguments into @p reports; returns 0 or -1 with an exception set. */
static int walk_reports_fill(walk_reports_t* reports, usearch_py_call_t* call, PyObject* progress_object,
                             PyObject* stats_object) {
    if (usearch_py_callback_from_object(progress_object, "progress", call, &reports->progress) != 0) return -1;
    reports->stats_object = NULL;
    if (!stats_object || stats_object == Py_None) return 0;
    if (!Py_IS_TYPE(stats_object, &DenseTraversalStatsType)) {
        PyErr_Format(PyExc_TypeError, "the 'stats' option must be a DenseTraversalStats or None, got %S",
                     Py_TYPE(stats_object));
        return -1;
    }
    reports->stats_object = Py_NewRef(stats_object);
    return 0;
}

/** Copies the telemetry into the caller's stats, fenced against a concurrent `+=`; a caller who passed none skips. */
static void walk_reports_answer(walk_reports_t const* reports) {
    if (!reports->stats_object) return;
    usearch_py_critical_section_begin_(reports->stats_object);
    ((DenseTraversalStats*)reports->stats_object)->stats = reports->stats;
    usearch_py_critical_section_end_();
}

static void walk_reports_release(walk_reports_t* reports) {
    Py_CLEAR(reports->progress.callable);
    Py_CLEAR(reports->stats_object);
}

static int walk_reports_traverse(walk_reports_t const* reports, visitproc visit, void* arg) {
    Py_VISIT(reports->progress.callable);
    Py_VISIT(reports->stats_object);
    return 0;
}

#pragma endregion Handles

#pragma region Construction

static char const doc_DenseIndex[] =                                                            //
    "DenseIndex\n"                                                                              //
    "\n"                                                                                        //
    "Vector search over equidimensional dense embeddings.\n"                                    //
    "\n"                                                                                        //
    "Every verb is a batch, and one vector is a one-row matrix. The stored scalar type is\n"    //
    "the `quantization`; query rows may arrive in any other type and convert on the way in.\n"  //
    "\n"                                                                                        //
    "Start an empty one in the residency its rows live in - `init_ram` or `init_vram` - copy\n" //
    "a checkpoint in with `load_ram` or `load_vram`, or serve one in place with `view_ram`\n"   //
    "or `view_nvme`, which are read-only.\n"                                                    //
    "\n"                                                                                        //
    "The receiver chooses where a verb runs, and the verb stays the same: the index runs it\n"  //
    "on the calling thread, `on(executor)` blocks on that executor, and `deferred(executor)`\n" //
    "submits it there and answers a `Deferred`. See `usearch.Executor`.\n"                      //
    "\n"                                                                                        //
    "Example:\n"                                                                                //
    "  >>> index = usearch.DenseIndex.init_ram(256, usearch.DenseMetric.ANGULAR)\n"             //
    "  >>> index.capabilities is not None\n"                                                    //
    "  True";

/** The geometry both creation verbs freeze, read before the factory's residency arm. */
typedef struct dense_geometry_t {
    usearch_dense_metric_t metric;
    usearch_scalar_t quantization;
    size_t dimensions;
    size_t connectivity;
    size_t values_per_key;
} dense_geometry_t;

/**
 *  @brief Reads the geometry both creation verbs take: three settings positionally or by name, then
 *      `connectivity` and `values_per_key` by name, and the residency arm @p arm_name into @p arm.
 *
 *  The rest are keyword-only, because a call site reading `init_ram(256, DenseMetric.ANGULAR, Scalar.BF16, 0, 0)`
 *  tells the reader nothing about the last two.
 *
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int geometry_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                          PyObject* args_names_tuple, char const* arm_name, PyObject** arm,
                          dense_geometry_t* geometry) {
    PyObject *dimensions_object, *metric_object, *quantization_object, *connectivity_object, *values_per_key_object;
    usearch_py_argument_t const arguments[] = {
        {"dimensions", &dimensions_object},         {"metric", &metric_object},
        {"quantization", &quantization_object},     {"connectivity", &connectivity_object},
        {"values_per_key", &values_per_key_object}, {arm_name, arm},
    };
    memset(geometry, 0, sizeof(*geometry));
    int metric = usearch_dense_metric_angular_k;
    if (usearch_py_arguments_parse(verb, args, positional_args_count, args_names_tuple, 0, 3, arguments,
                                   usearch_py_table_size_(arguments)) != 0)
        return -1;
    if (!dimensions_object) {
        PyErr_Format(PyExc_TypeError, "%s() missing required argument 'dimensions'", verb);
        return -1;
    }
    if (usearch_py_size_from_object(dimensions_object, "dimensions", &geometry->dimensions) != 0 ||
        (metric_object &&
         usearch_py_enum_from_object(metric_object, "metric", &usearch_py_dense_metric, &metric) != 0) ||
        usearch_py_scalar_from_object(quantization_object, "quantization", &geometry->quantization) != 0 ||
        usearch_py_size_from_object(connectivity_object, "connectivity", &geometry->connectivity) != 0 ||
        usearch_py_size_from_object(values_per_key_object, "values_per_key", &geometry->values_per_key) != 0)
        return -1;
    geometry->metric = (usearch_dense_metric_t)metric;
    if (geometry->quantization == usearch_scalar_unknown_k) geometry->quantization = usearch_scalar_bf16_k;
    return 0;
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

/** Adopts the handle a factory produced, a progress handler's own exception outranking the outcome. */
static PyObject* index_loaded(PyTypeObject* type, usearch_dense_index_t handle, usearch_py_call_t* call) {
    if (call->raised) {
        usearch_dense_free(handle);
        PyErr_SetRaisedException(call->raised);
        call->raised = NULL;
        return NULL;
    }
    return index_adopt(type, handle, &call->outcome);
}

static PyObject* DenseIndex_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(
        PyExc_TypeError,
        "DenseIndex is built through init_ram(), init_vram(), load_ram*(), load_vram*(), view_ram() or view_nvme()");
    return NULL;
}

static char const doc_DenseIndex_init_ram[] =                                                         //
    "init_ram(dimensions, metric=DenseMetric.ANGULAR, quantization=Scalar.BF16, *, connectivity=0,\n" //
    "    values_per_key=0, memory_domains=None) -> DenseIndex\n"                                      //
    "\n"                                                                                              //
    "Starts an empty, mutable graph in host memory, one shard per named NUMA node.\n"                 //
    "Calls `usearch_dense_init_ram`.\n"                                                               //
    "\n"                                                                                              //
    "Args:\n"                                                                                         //
    "  dimensions (int): Scalars per vector.\n"                                                       //
    "  metric (DenseMetric, optional): How two rows are scored. Defaults to `DenseMetric.ANGULAR`.\n" //
    "  quantization (Scalar, optional): The stored scalar type. Defaults to `Scalar.BF16`.\n"         //
    "  connectivity (int, optional): Connections per node; 0 derives it from the geometry.\n"         //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"                  //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to shard across, so a\n"               //
    "    dual-socket box naming two gets two graphs; None derives every node.\n"                      //
    "\n"                                                                                              //
    "Example:\n"                                                                                      //
    "  >>> index = usearch.DenseIndex.init_ram(256, usearch.DenseMetric.ANGULAR)\n"                   //
    "  >>> index.dimensions\n"                                                                        //
    "  256";

static PyObject* DenseIndex_init_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    dense_geometry_t geometry;
    PyObject* memory_domains_object;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    if (geometry_parse("init_ram", args, positional_args_count, args_names_tuple, "memory_domains",
                       &memory_domains_object, &geometry) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
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

static char const doc_DenseIndex_init_vram[] =                                                         //
    "init_vram(dimensions, metric=DenseMetric.ANGULAR, quantization=Scalar.BF16, *, connectivity=0,\n" //
    "    values_per_key=0, devices=None) -> DenseIndex\n"                                              //
    "\n"                                                                                               //
    "Starts an empty, mutable graph on CUDA devices, one shard per named ordinal.\n"                   //
    "Calls `usearch_dense_init_vram`.\n"                                                               //
    "\n"                                                                                               //
    "Rows are sharded and never replicated, so eight devices host eight slices of one\n"               //
    "graph rather than eight copies. An index outgrowing them fails rather than paging to\n"           //
    "the host, and the answer is to name more devices.\n"                                              //
    "\n"                                                                                               //
    "Args:\n"                                                                                          //
    "  dimensions (int): Scalars per vector.\n"                                                        //
    "  metric (DenseMetric, optional): How two rows are scored. Defaults to `DenseMetric.ANGULAR`.\n"  //
    "  quantization (Scalar, optional): The stored scalar type. Defaults to `Scalar.BF16`.\n"          //
    "  connectivity (int, optional): Connections per node; 0 derives it from the geometry.\n"          //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"                   //
    "  devices (Sequence[int], optional): CUDA ordinals to shard across; None derives one\n"           //
    "    free device.\n"                                                                               //
    "\n"                                                                                               //
    "Example:\n"                                                                                       //
    "  >>> index = usearch.DenseIndex.init_vram(256, devices=[0, 1])   # doctest: +SKIP";

static PyObject* DenseIndex_init_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    dense_geometry_t geometry;
    PyObject* devices_object;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    if (geometry_parse("init_vram", args, positional_args_count, args_names_tuple, "devices", &devices_object,
                       &geometry) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        return NULL;

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

static char const doc_DenseIndex_on[] =                                                        //
    "on(executor) -> On[DenseIndex]\n"                                                         //
    "\n"                                                                                       //
    "A receiver whose verbs block on `executor`, after every call submitted to it before.\n"   //
    "It offers every verb the index does, with the same arguments and answers. Python-only.\n" //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  executor (Executor): The workers the calls run on; its residency must match.\n"         //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> pool = usearch.Executor.for_ram(threads=2)\n"                                       //
    "  >>> usearch.DenseIndex.init_ram(4).on(pool).describe().size\n"                          //
    "  0";

static char const doc_DenseIndex_deferred[] =                                                   //
    "deferred(executor) -> DeferredOn[DenseIndex]\n"                                            //
    "\n"                                                                                        //
    "A receiver whose verbs submit their call to `executor` and answer a `Deferred` at once,\n" //
    "with the same arguments the index takes. It holds the index and the executor until\n"      //
    "every `Deferred` it answered is gone. Python-only.\n"                                      //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  executor (Executor): The workers the calls run on; its residency must match.";

static PyObject* DenseIndex_on(PyObject* self, PyObject* executor) {
    return usearch_py_on_new(&OnType, self, executor);
}

static PyObject* DenseIndex_deferred(PyObject* self, PyObject* executor) {
    return usearch_py_on_new(&DeferredOnType, self, executor);
}

#pragma endregion Construction

#pragma region Reports

static char const doc_DenseTraversalStats[] =                                   //
    "DenseTraversalStats()\n"                                                   //
    "\n"                                                                        //
    "Traversal telemetry, fully overwritten by every call it is passed to.\n"   //
    "\n"                                                                        //
    "One reading belongs to one call, so a sweep that wants totals adds each\n" //
    "reading into a separate one with `+=`, which is Python-only.\n"            //
    "\n"                                                                        //
    "Attributes:\n"                                                             //
    "  visited_members (int): Entries whose neighborhoods the walk expanded.\n" //
    "  computed_distances (int): Distance evaluations the walk paid for.\n"     //
    "  restarts (int): Times the walk fell back to a fresh entry point.\n"      //
    "\n"                                                                        //
    "Example:\n"                                                                //
    "  >>> total = usearch.DenseTraversalStats()\n"                             //
    "  >>> total += usearch.DenseTraversalStats()\n"                            //
    "  >>> total.visited_members\n"                                             //
    "  0";

static PyObject* DenseTraversalStats_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    if (PyTuple_GET_SIZE(args) != 0 || (keywords && PyDict_Size(keywords) != 0)) {
        PyErr_SetString(PyExc_TypeError, "DenseTraversalStats() takes no arguments");
        return NULL;
    }
    return type->tp_alloc(type, 0); // Zeroed, so a fresh reading has counted nothing.
}

static void DenseTraversalStats_dealloc(DenseTraversalStats* self) { Py_TYPE(self)->tp_free((PyObject*)self); }

static PyObject* DenseTraversalStats_repr(DenseTraversalStats* self) {
    return PyUnicode_FromFormat("DenseTraversalStats(visited_members=%zu, computed_distances=%zu, restarts=%zu)",
                                self->stats.visited_members, self->stats.computed_distances, self->stats.restarts);
}

/** Adds another reading's counters into this one, fenced like the overwrite a verb performs. */
static PyObject* DenseTraversalStats_inplace_add(PyObject* self, PyObject* other) {
    if (!Py_IS_TYPE(other, &DenseTraversalStatsType)) Py_RETURN_NOTIMPLEMENTED;
    usearch_traversal_stats_t addend;
    usearch_py_critical_section_begin_(other);
    addend = ((DenseTraversalStats*)other)->stats;
    usearch_py_critical_section_end_();
    DenseTraversalStats* const total = (DenseTraversalStats*)self;
    usearch_py_critical_section_begin_(self);
    total->stats.visited_members += addend.visited_members;
    total->stats.computed_distances += addend.computed_distances;
    total->stats.restarts += addend.restarts;
    usearch_py_critical_section_end_();
    return Py_NewRef(self);
}

/** Reads the @c size_t counter at the byte offset @p closure names inside the telemetry. */
static PyObject* DenseTraversalStats_get_counter(DenseTraversalStats* self, void* closure) {
    return PyLong_FromSize_t(*(size_t const*)((char const*)&self->stats + (size_t)closure));
}

static PyGetSetDef DenseTraversalStats_getset[] = {
    {"visited_members", (getter)DenseTraversalStats_get_counter, NULL, "Entries whose neighborhoods the walk expanded.",
     (void*)offsetof(usearch_traversal_stats_t, visited_members)},
    {"computed_distances", (getter)DenseTraversalStats_get_counter, NULL, "Distance evaluations the walk paid for.",
     (void*)offsetof(usearch_traversal_stats_t, computed_distances)},
    {"restarts", (getter)DenseTraversalStats_get_counter, NULL, "Times the walk fell back to a fresh entry point.",
     (void*)offsetof(usearch_traversal_stats_t, restarts)},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyNumberMethods DenseTraversalStats_as_number = {.nb_inplace_add = DenseTraversalStats_inplace_add};

PyTypeObject DenseTraversalStatsType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseTraversalStats",
    .tp_doc = doc_DenseTraversalStats,
    .tp_basicsize = sizeof(DenseTraversalStats),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = DenseTraversalStats_tp_new,
    .tp_dealloc = (destructor)DenseTraversalStats_dealloc,
    .tp_repr = (reprfunc)DenseTraversalStats_repr,
    .tp_as_number = &DenseTraversalStats_as_number,
    .tp_getset = DenseTraversalStats_getset,
};

static PyStructSequence_Field DenseMetadata_fields[] = {
    {"dimensions", "Scalars per stored vector."},
    {"metric", "The `DenseMetric` the checkpoint scores with."},
    {"quantization", "The `Scalar` its rows are stored and quantized into."},
    {"connectivity", "Connections per node the graph was built with."},
    {NULL, NULL},
};

PyStructSequence_Desc usearch_py_dense_metadata_desc = {
    "usearch.DenseMetadata", "The geometry and metric a checkpoint froze, as `usearch_dense_metadata_t` holds them.",
    DenseMetadata_fields, 4};

PyTypeObject DenseMetadataType;

static PyStructSequence_Field DenseIndexDescription_fields[] = {
    {"metadata", "The frozen `DenseMetadata`, as a checkpoint reports it."},
    {"size", "Stored entries."},
    {"capacity", "Entries hosted without growing again."},
    {"memory_usage", "Bytes the handle holds; traversal scratch is the executor's."},
    {"capabilities", "The codename of the kernels the index dispatched to on this machine."},
    {NULL, NULL},
};

PyStructSequence_Desc usearch_py_dense_index_description_desc = {
    "usearch.DenseIndexDescription",
    "A live index's frozen metadata and current counters, as `usearch_dense_index_description_t` holds them.",
    DenseIndexDescription_fields, 5};

PyTypeObject DenseIndexDescriptionType;

static PyStructSequence_Field DenseReachability_fields[] = {
    {"audited", "Entries published when the audit started, the population it judged."},
    {"reachable", "Of those, the entries a search reaches; the rest wait for a `refine` to re-link them."},
    {"sweeps", "Levels the sweep took to converge."},
    {"back_sweeps", "Levels the sweep back toward the entry point took, where an audit sweeps both ways."},
    {NULL, NULL},
};

PyStructSequence_Desc usearch_py_dense_reachability_desc = {
    "usearch.DenseReachability", "One reachability audit, as `usearch_reachability_t` holds it.",
    DenseReachability_fields, 4};

PyTypeObject DenseReachabilityType;

/** A new @c DenseMetadata of @p metadata, or NULL with an exception set. */
static PyObject* metadata_to_object(usearch_dense_metadata_t const* metadata) {
    PyObject* fields[4] = {NULL, NULL, NULL, NULL};
    (void)((fields[0] = PyLong_FromSize_t(metadata->dimensions)) &&
           (fields[1] = usearch_py_enum_to_object(&usearch_py_dense_metric, (int)metadata->metric)) &&
           (fields[2] = usearch_py_enum_to_object(&usearch_py_scalar, (int)metadata->quantization)) &&
           (fields[3] = PyLong_FromSize_t(metadata->connectivity)));
    return usearch_py_struct_sequence_steal(&DenseMetadataType, fields, 4);
}

#pragma endregion Reports

#pragma region Introspection

/**
 *  @brief Reads the frozen geometry and current counters of @p handle on the calling thread.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int index_describe(usearch_dense_index_t handle, usearch_dense_index_description_t* description) {
    usearch_outcome_t outcome = {0};
    usearch_dense_describe_t arguments = {.index = handle, .outcome = &outcome, .description = description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_describe(&arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_raise_on_outcome(&outcome);
}

static char const doc_DenseIndex_describe[] =                                                //
    "describe() -> DenseIndexDescription\n"                                                  //
    "\n"                                                                                     //
    "The frozen metadata and the current counters, read in one call. Each counter is read\n" //
    "once, so under concurrent insertions they may come from slightly different moments.\n"  //
    "Calls `usearch_dense_describe`.\n"                                                      //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> usearch.DenseIndex.init_ram(256).describe().metadata.dimensions\n"                //
    "  256";

/** A @c usearch_dense_describe call and the description it lands. */
typedef struct describe_call_t {
    usearch_py_call_t call;
    usearch_dense_describe_t arguments;
    usearch_dense_index_description_t description;
} describe_call_t;

static void describe_submit(void* call) { usearch_dense_describe(&((describe_call_t*)call)->arguments); }

static PyObject* describe_answer(void* erased) {
    usearch_dense_index_description_t const* const description = &((describe_call_t*)erased)->description;
    PyObject* fields[5] = {NULL, NULL, NULL, NULL, NULL};
    (void)((fields[0] = metadata_to_object(&description->metadata)) &&
           (fields[1] = PyLong_FromSize_t(description->size)) &&
           (fields[2] = PyLong_FromSize_t(description->capacity)) &&
           (fields[3] = PyLong_FromSize_t(description->memory_usage)) &&
           (fields[4] = PyUnicode_FromString(description->capabilities ? description->capabilities : "")));
    return usearch_py_struct_sequence_steal(&DenseIndexDescriptionType, fields, 5);
}

static usearch_py_verb_t const describe_verb = {sizeof(describe_call_t), describe_submit, describe_answer,
                                                usearch_py_call_release};

static PyObject* DenseIndex_describe(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t const handle = receiver_from_object(self, &receiver);
    if (!handle) return NULL;
    describe_call_t stack;
    describe_call_t* const call = usearch_py_call_open(&receiver, &describe_verb, &stack);
    if (!call) return NULL;
    call->arguments = (usearch_dense_describe_t) {.index = handle,
                                                  .executor = receiver.executor,
                                                  .outcome = &call->call.outcome,
                                                  .completion = receiver.completion,
                                                  .description = &call->description};
    return usearch_py_call_run(&receiver, &describe_verb, call);
}

static char const doc_DenseIndex_health[] =                                                    //
    "health() -> DenseReachability\n"                                                          //
    "\n"                                                                                       //
    "Audits how much of the graph a search can reach, sweeping forward reachability from as\n" //
    "many landmarks as one search draws. Read-only.\n"                                         //
    "Calls `usearch_dense_health`.\n"                                                          //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> from array import array\n"                                                          //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=4)\n"                                //
    "  >>> index.reserve(1)\n"                                                                 //
    "  >>> index.insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))\n"                              //
    "  >>> health = index.health()\n"                                                          //
    "  >>> health.reachable == health.audited == 1\n"                                          //
    "  True";

/** A @c usearch_dense_health call and the reachability it lands. */
typedef struct health_call_t {
    usearch_py_call_t call;
    usearch_dense_health_t arguments;
    usearch_reachability_t reachability;
} health_call_t;

static void health_submit(void* call) { usearch_dense_health(&((health_call_t*)call)->arguments); }

static PyObject* health_answer(void* erased) {
    usearch_reachability_t const* const reachability = &((health_call_t*)erased)->reachability;
    PyObject* fields[4] = {NULL, NULL, NULL, NULL};
    (void)((fields[0] = PyLong_FromSize_t(reachability->audited)) &&
           (fields[1] = PyLong_FromSize_t(reachability->reachable)) &&
           (fields[2] = PyLong_FromSize_t(reachability->sweeps)) &&
           (fields[3] = PyLong_FromSize_t(reachability->back_sweeps)));
    return usearch_py_struct_sequence_steal(&DenseReachabilityType, fields, 4);
}

static usearch_py_verb_t const health_verb = {sizeof(health_call_t), health_submit, health_answer,
                                              usearch_py_call_release};

static PyObject* DenseIndex_health(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t const handle = receiver_from_object(self, &receiver);
    if (!handle) return NULL;
    health_call_t stack;
    health_call_t* const call = usearch_py_call_open(&receiver, &health_verb, &stack);
    if (!call) return NULL;
    call->arguments = (usearch_dense_health_t) {.index = handle,
                                                .executor = receiver.executor,
                                                .outcome = &call->call.outcome,
                                                .completion = receiver.completion,
                                                .reachability = &call->reachability};
    return usearch_py_call_run(&receiver, &health_verb, call);
}

/** Reads the @c size_t counter at the byte offset @p closure names inside a description. */
static PyObject* DenseIndex_get_counter(DenseIndex* self, void* closure) {
    usearch_dense_index_description_t description;
    usearch_dense_index_t const handle = index_handle(self);
    if (!handle || index_describe(handle, &description) != 0) return NULL;
    return PyLong_FromSize_t(*(size_t const*)((char const*)&description + (size_t)closure));
}

static char const doc_DenseIndex_capacity[] =
    "Entries the index hosts without growing again. Calls `usearch_dense_describe`.";
static char const doc_DenseIndex_dimensions[] = "Scalars per vector. Calls `usearch_dense_describe`.";
static char const doc_DenseIndex_connectivity[] = "Connections per node in the graph. Calls `usearch_dense_describe`.";
static char const doc_DenseIndex_memory_usage[] =
    "Bytes the handle holds; traversal scratch is the executor's. Calls `usearch_dense_describe`.";
static char const doc_DenseIndex_capabilities[] =
    "The kernel family dispatch picked for this geometry. Calls `usearch_dense_describe`.";

static PyObject* DenseIndex_get_capabilities(DenseIndex* self, void* closure) {
    usearch_dense_index_description_t description;
    usearch_dense_index_t const handle = index_handle(self);
    if (!handle || index_describe(handle, &description) != 0) return NULL;
    return PyUnicode_FromString(description.capabilities ? description.capabilities : "");
}

static Py_ssize_t DenseIndex_length(DenseIndex* self) {
    usearch_dense_index_description_t description;
    usearch_dense_index_t const handle = index_handle(self);
    if (!handle || index_describe(handle, &description) != 0) return -1;
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

static char const doc_DenseIndex_reserve[] =                 //
    "reserve(capacity) -> None\n"                            //
    "\n"                                                     //
    "Grows the index to host at least `capacity` entries.\n" //
    "Calls `usearch_dense_reserve`.\n"                       //
    "\n"                                                     //
    "Args:\n"                                                //
    "  capacity (int): Entries the index must accept without growing again.";

/** A @c usearch_dense_reserve call. */
typedef struct reserve_call_t {
    usearch_py_call_t call;
    usearch_dense_reserve_t arguments;
} reserve_call_t;

static void reserve_submit(void* call) { usearch_dense_reserve(&((reserve_call_t*)call)->arguments); }

static usearch_py_verb_t const reserve_verb = {sizeof(reserve_call_t), reserve_submit, usearch_py_call_answer_none,
                                               usearch_py_call_release};

static PyObject* DenseIndex_reserve(PyObject* self, PyObject* capacity_object) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t const handle = receiver_from_object(self, &receiver);
    size_t capacity = 0;
    if (!handle || usearch_py_size_from_object(capacity_object, "capacity", &capacity) != 0) return NULL;
    reserve_call_t stack;
    reserve_call_t* const call = usearch_py_call_open(&receiver, &reserve_verb, &stack);
    if (!call) return NULL;
    call->arguments = (usearch_dense_reserve_t) {.index = handle,
                                                 .executor = receiver.executor,
                                                 .outcome = &call->call.outcome,
                                                 .completion = receiver.completion,
                                                 .capacity = capacity};
    return usearch_py_call_run(&receiver, &reserve_verb, call);
}

static char const doc_DenseIndex_clear[] =                               //
    "clear() -> None\n"                                                  //
    "\n"                                                                 //
    "Drops every entry, keeping the capacity and the frozen geometry.\n" //
    "Calls `usearch_dense_clear`.";

/** A @c usearch_dense_clear call. */
typedef struct clear_call_t {
    usearch_py_call_t call;
    usearch_dense_clear_t arguments;
} clear_call_t;

static void clear_submit(void* call) { usearch_dense_clear(&((clear_call_t*)call)->arguments); }

static usearch_py_verb_t const clear_verb = {sizeof(clear_call_t), clear_submit, usearch_py_call_answer_none,
                                             usearch_py_call_release};

static PyObject* DenseIndex_clear(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t const handle = receiver_from_object(self, &receiver);
    if (!handle) return NULL;
    clear_call_t stack;
    clear_call_t* const call = usearch_py_call_open(&receiver, &clear_verb, &stack);
    if (!call) return NULL;
    call->arguments = (usearch_dense_clear_t) {.index = handle,
                                               .executor = receiver.executor,
                                               .outcome = &call->call.outcome,
                                               .completion = receiver.completion};
    return usearch_py_call_run(&receiver, &clear_verb, call);
}

static char const doc_DenseIndex_insert[] =                                                  //
    "insert(keys, vectors, /, *, expansion_factor=0, dtype=None, progress=None,\n"           //
    "    progress_steps=0, stats=None) -> None\n"                                            //
    "\n"                                                                                     //
    "Inserts a batch of vectors under a batch of keys; one vector is a one-row matrix.\n"    //
    "Keys need not be unique: inserting one twice stores two entries, and `count`\n"         //
    "reports both. A batch is all-or-nothing and becomes visible at once, so a rejected\n"   //
    "batch leaves the index untouched and a concurrent search sees all of it or none.\n"     //
    "Calls `usearch_dense_insert`.\n"                                                        //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  keys (Sequence[Key | int] | buffer): One key per row, or an (N, 16) buffer.\n"        //
    "  vectors (buffer): An (N, dimensions) matrix; only the row stride may vary.\n"         //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"         //
    "  dtype (Scalar, optional): Names the input scalar type when the buffer cannot; it\n"   //
    "    must be as wide as the buffer's items.\n"                                           //
    "  progress (callable, optional): Called with (completed, total) rows. It reports and\n" //
    "    cannot stop the batch, every row being claimed before the first report; what it\n"  //
    "    raises surfaces once the batch has landed. Reports are serialized.\n"               //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"  //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"     //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> from array import array\n"                                                        //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=4)\n"                              //
    "  >>> index.reserve(1)\n"                                                               //
    "  >>> index.insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))\n"                            //
    "  >>> len(index)\n"                                                                     //
    "  1";

/** A @c usearch_dense_insert call and the inputs it borrows until it is final. */
typedef struct insert_call_t {
    usearch_py_call_t call;
    usearch_dense_insert_t arguments;
    usearch_py_buffer_backing_t vectors;
    usearch_py_keys_t keys;
    walk_reports_t reports;
} insert_call_t;

static void insert_submit(void* call) { usearch_dense_insert(&((insert_call_t*)call)->arguments); }

static PyObject* insert_answer(void* erased) {
    walk_reports_answer(&((insert_call_t*)erased)->reports);
    Py_RETURN_NONE;
}

static void insert_release(void* erased) {
    insert_call_t* const call = (insert_call_t*)erased;
    usearch_py_keys_release(&call->keys);
    usearch_py_buffer_release(&call->vectors);
    walk_reports_release(&call->reports);
    usearch_py_call_release(&call->call);
}

static int insert_traverse(void* call, visitproc visit, void* arg) {
    return walk_reports_traverse(&((insert_call_t*)call)->reports, visit, arg);
}

static usearch_py_verb_t const insert_verb = {sizeof(insert_call_t), insert_submit, insert_answer, insert_release,
                                              insert_traverse};

static PyObject* DenseIndex_insert(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *keys_object, *vectors_object, *expansion_object, *dtype_object, *progress_object, *progress_steps_object,
        *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"keys", &keys_object},   {"vectors", &vectors_object},   {"expansion_factor", &expansion_object},
        {"dtype", &dtype_object}, {"progress", &progress_object}, {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t expansion_factor = 0, progress_steps = 0;
    usearch_scalar_t dtype = usearch_scalar_unknown_k;
    if (usearch_py_arguments_parse("insert", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)) ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0)
        return NULL;

    insert_call_t stack;
    insert_call_t* const call = usearch_py_call_open(&receiver, &insert_verb, &stack);
    if (!call) return NULL;
    usearch_matrix_view_t vectors;
    if (usearch_py_matrix_from_object(vectors_object, dtype, &vectors, &call->vectors) != 0 ||
        usearch_py_keys_from_object(keys_object, vectors.rows, &call->keys) != 0 ||
        walk_reports_fill(&call->reports, &call->call, progress_object, stats_object) != 0)
        return usearch_py_call_discard(&receiver, &insert_verb, call);

    call->arguments = (usearch_dense_insert_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .count = vectors.rows,
        .keys = {call->keys.keys, sizeof(usearch_key_t)},
        .vectors = rows_of(&vectors),
        .expansion_factor = expansion_factor,
        .progress = usearch_py_progress_registered(&call->reports.progress, progress_steps),
        .stats = &call->reports.stats,
    };
    return usearch_py_call_run(&receiver, &insert_verb, call);
}

static char const doc_DenseIndex_refine[] =                                                  //
    "refine(*, expansion_factor=0, progress=None, progress_steps=0, stats=None) -> int\n"    //
    "\n"                                                                                     //
    "Re-selects the neighbors of every stored entry against the finished graph, improving\n" //
    "edges that were chosen when the graph around them was still incomplete.\n"              //
    "\n"                                                                                     //
    "A build step rather than a repair: a cheap insertion pass followed by one refinement\n" //
    "reaches a better graph, in less total time, than one expensive insertion pass - so a\n" //
    "wide `expansion_factor` belongs here rather than in `insert`. Searches may run\n"       //
    "throughout on a RAM handle but not on a VRAM one, and a caller loops until the\n"       //
    "return falls below whatever change rate they consider settled.\n"                       //
    "\n"                                                                                     //
    "Stopping is not an error here and raises nothing: re-selecting edges claims no\n"       //
    "identity, so a stopped sweep leaves the graph correct and merely less improved. A\n"    //
    "handler is therefore how a sweep is put under a time budget.\n"                         //
    "Calls `usearch_dense_refine`.\n"                                                        //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"         //
    "  progress (callable, optional): Called with (completed, total) nodes; returning\n"     //
    "    False ends the sweep, None carries on.\n"                                           //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"  //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"     //
    "\n"                                                                                     //
    "Returns:\n"                                                                             //
    "  int: Nodes whose edges changed; zero means the graph has settled.\n"                  //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> from array import array\n"                                                        //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=4)\n"                              //
    "  >>> index.reserve(1)\n"                                                               //
    "  >>> index.insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))\n"                            //
    "  >>> index.refine()\n"                                                                 //
    "  0";

/** A @c usearch_dense_refine call and the count it lands. */
typedef struct refine_call_t {
    usearch_py_call_t call;
    usearch_dense_refine_t arguments;
    walk_reports_t reports;
    size_t changed;
} refine_call_t;

static void refine_submit(void* call) { usearch_dense_refine(&((refine_call_t*)call)->arguments); }

static PyObject* refine_answer(void* erased) {
    refine_call_t* const call = (refine_call_t*)erased;
    walk_reports_answer(&call->reports);
    return PyLong_FromSize_t(call->changed);
}

static void refine_release(void* erased) {
    refine_call_t* const call = (refine_call_t*)erased;
    walk_reports_release(&call->reports);
    usearch_py_call_release(&call->call);
}

static int refine_traverse(void* call, visitproc visit, void* arg) {
    return walk_reports_traverse(&((refine_call_t*)call)->reports, visit, arg);
}

static usearch_py_verb_t const refine_verb = {sizeof(refine_call_t), refine_submit, refine_answer, refine_release,
                                              refine_traverse};

static PyObject* DenseIndex_refine(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *expansion_object, *progress_object, *progress_steps_object, *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"expansion_factor", &expansion_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t expansion_factor = 0, progress_steps = 0;
    if (usearch_py_arguments_parse("refine", args, positional_args_count, args_names_tuple, 0, 0, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)) ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0)
        return NULL;

    refine_call_t stack;
    refine_call_t* const call = usearch_py_call_open(&receiver, &refine_verb, &stack);
    if (!call) return NULL;
    if (walk_reports_fill(&call->reports, &call->call, progress_object, stats_object) != 0)
        return usearch_py_call_discard(&receiver, &refine_verb, call);

    call->arguments = (usearch_dense_refine_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .changed = &call->changed,
        .expansion_factor = expansion_factor,
        .progress = usearch_py_progress_registered(&call->reports.progress, progress_steps),
        .stats = &call->reports.stats,
    };
    return usearch_py_call_run(&receiver, &refine_verb, call);
}

#pragma endregion Ingest

#pragma region Search

static char const doc_DenseIndex_search[] =                                                    //
    "search(queries, wanted, arena, /, *, expansion_factor=0, traversal=Traversal.WALK,\n"     //
    "    dtype=None, predicate=None, progress=None, progress_steps=0, stats=None)\n"           //
    "    -> Matches\n"                                                                         //
    "\n"                                                                                       //
    "Finds the `wanted` nearest entries for every query row; one query is a one-row matrix.\n" //
    "Never writes to the graph, so any number of searches may overlap with insertions.\n"      //
    "Calls `usearch_dense_search`.\n"                                                          //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  queries (buffer): A (Q, dimensions) matrix of query rows.\n"                            //
    "  wanted (int): Upper bound on matches per query - the 'k' in 'kANN'.\n"                  //
    "  arena (Arena): Where the matches land, rewound by the call.\n"                          //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"           //
    "  traversal (Traversal, optional): `Traversal.WALK`, the default, walks the graph;\n"     //
    "    `Traversal.SCAN` scores every row exactly, on RAM and NVMe handles.\n"                //
    "  dtype (Scalar, optional): Names the query scalar type when the buffer cannot.\n"        //
    "  predicate (callable, optional): Admits a Key when it returns truthy. Non-matching\n"    //
    "    entries still route the traversal, so recall survives low selectivity. The\n"         //
    "    callback re-enters Python, so it runs under the GIL and serializes the walk.\n"       //
    "  progress (callable, optional): Called with (completed, total) queries; returning\n"     //
    "    False stops the batch and raises `CancelledError`, None carries on.\n"                //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"    //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"       //
    "\n"                                                                                       //
    "Returns:\n"                                                                               //
    "  Matches: (Q, columns) runs in the arena, columns at most `wanted`, each count\n"        //
    "    bounding its own row.\n"                                                              //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> from array import array\n"                                                          //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=4)\n"                                //
    "  >>> index.reserve(1)\n"                                                                 //
    "  >>> query = array('d', [0.1, 0.2, 0.3, 0.4])\n"                                         //
    "  >>> index.insert([7], query)\n"                                                         //
    "  >>> key, distance = index.search(query, 1, usearch.Arena())[0][0]\n"                    //
    "  >>> int(key)\n"                                                                         //
    "  7";

/** A @c usearch_dense_search call, the inputs it borrows until it is final, and the matches it
 *  lands. */
typedef struct search_call_t {
    usearch_py_call_t call;
    usearch_dense_search_t arguments;
    usearch_py_buffer_backing_t queries;
    usearch_py_callback_t predicate;
    walk_reports_t reports;
    usearch_matches_view_t matches;
} search_call_t;

static void search_submit(void* call) { usearch_dense_search(&((search_call_t*)call)->arguments); }

static PyObject* search_answer(void* erased) {
    search_call_t* const call = (search_call_t*)erased;
    walk_reports_answer(&call->reports);
    return usearch_py_matches_new(call->call.arena, &call->matches, call->arguments.count);
}

static void search_release(void* erased) {
    search_call_t* const call = (search_call_t*)erased;
    usearch_py_buffer_release(&call->queries);
    Py_CLEAR(call->predicate.callable);
    walk_reports_release(&call->reports);
    usearch_py_call_release(&call->call);
}

static int search_traverse(void* erased, visitproc visit, void* arg) {
    search_call_t* const call = (search_call_t*)erased;
    Py_VISIT(call->predicate.callable);
    return walk_reports_traverse(&call->reports, visit, arg);
}

static usearch_py_verb_t const search_verb = {sizeof(search_call_t), search_submit, search_answer, search_release,
                                              search_traverse};

static PyObject* DenseIndex_search(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *queries_object, *wanted_object, *arena_object, *expansion_object, *traversal_object, *dtype_object,
        *predicate_object, *progress_object, *progress_steps_object, *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"queries", &queries_object},
        {"wanted", &wanted_object},
        {"arena", &arena_object},
        {"expansion_factor", &expansion_object},
        {"traversal", &traversal_object},
        {"dtype", &dtype_object},
        {"predicate", &predicate_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t wanted = 0, expansion_factor = 0, progress_steps = 0;
    int traversal = usearch_traversal_walk_k;
    usearch_scalar_t dtype = usearch_scalar_unknown_k;
    if (usearch_py_arguments_parse("search", args, positional_args_count, args_names_tuple, 3, 3, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)) ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        (traversal_object &&
         usearch_py_enum_from_object(traversal_object, "traversal", &usearch_py_traversal, &traversal) != 0) ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0)
        return NULL;

    search_call_t stack;
    search_call_t* const call = usearch_py_call_open(&receiver, &search_verb, &stack);
    if (!call) return NULL;
    usearch_matrix_view_t queries;
    Py_ssize_t extent = 0;
    if (usearch_py_matrix_from_object(queries_object, dtype, &queries, &call->queries) != 0 ||
        usearch_py_matrix_bytes(queries.rows, wanted, sizeof(usearch_key_t), &extent) != 0 ||
        usearch_py_callback_from_object(predicate_object, "predicate", &call->call, &call->predicate) != 0 ||
        walk_reports_fill(&call->reports, &call->call, progress_object, stats_object) != 0 ||
        usearch_py_arena_claim(arena_object, &call->call.arena) != 0)
        return usearch_py_call_discard(&receiver, &search_verb, call);

    call->arguments = (usearch_dense_search_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .count = queries.rows,
        .queries = rows_of(&queries),
        .matches = &call->matches,
        .arena = ((Arena*)call->call.arena)->handle,
        .wanted = wanted,
        .expansion_factor = expansion_factor,
        .traversal = (usearch_traversal_t)traversal,
        .predicate = usearch_py_predicate_registered(&call->predicate),
        .progress = usearch_py_progress_registered(&call->reports.progress, progress_steps),
        .stats = &call->reports.stats,
    };
    return usearch_py_call_run(&receiver, &search_verb, call);
}

#pragma endregion Search

#pragma region Retrieval

static char const doc_DenseIndex_count[] =                             //
    "count(key) -> int\n"                                              //
    "\n"                                                               //
    "How many entries share `key`; zero means the index holds none.\n" //
    "Calls `usearch_dense_count`.\n"                                   //
    "\n"                                                               //
    "Args:\n"                                                          //
    "  key (Key | int | str): The identity to count.\n"                //
    "\n"                                                               //
    "Example:\n"                                                       //
    "  >>> from array import array\n"                                  //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=2)\n"        //
    "  >>> index.reserve(2)\n"                                         //
    "  >>> row = array('d', [0.1, 0.2])\n"                             //
    "  >>> index.insert([7], row)\n"                                   //
    "  >>> index.insert([7], row)\n"                                   //
    "  >>> index.count(7)\n"                                           //
    "  2";

/** A @c usearch_dense_count call and the count it lands. */
typedef struct count_call_t {
    usearch_py_call_t call;
    usearch_dense_count_t arguments;
    size_t entries;
} count_call_t;

static void count_submit(void* call) { usearch_dense_count(&((count_call_t*)call)->arguments); }

static PyObject* count_answer(void* call) { return PyLong_FromSize_t(((count_call_t*)call)->entries); }

static usearch_py_verb_t const count_verb = {sizeof(count_call_t), count_submit, count_answer, usearch_py_call_release};

static PyObject* DenseIndex_count(PyObject* self, PyObject* key_object) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t const handle = receiver_from_object(self, &receiver);
    usearch_key_t key;
    if (!handle || usearch_py_key_from_object(key_object, &key) != 0) return NULL;
    count_call_t stack;
    count_call_t* const call = usearch_py_call_open(&receiver, &count_verb, &stack);
    if (!call) return NULL;
    call->arguments = (usearch_dense_count_t) {.index = handle,
                                               .executor = receiver.executor,
                                               .outcome = &call->call.outcome,
                                               .completion = receiver.completion,
                                               .key = key,
                                               .entries = &call->entries};
    return usearch_py_call_run(&receiver, &count_verb, call);
}

static int DenseIndex_contains(PyObject* self, PyObject* key_object) {
    PyObject* const entries = DenseIndex_count(self, key_object);
    if (!entries) return -1;
    int const present = PyObject_IsTrue(entries);
    Py_DECREF(entries);
    return present;
}

static char const doc_DenseIndex_get[] =                                                          //
    "get(key, arena, /, *, outputs_dtype=Scalar.F32) -> DenseVectors\n"                           //
    "\n"                                                                                          //
    "Every vector stored under `key`, in insertion order, and none when the key is absent.\n"     //
    "Calls `usearch_dense_get`.\n"                                                                //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  key (Key | int | str): The identity to retrieve.\n"                                        //
    "  arena (Arena): Where the rows land, rewound by the call.\n"                                //
    "  outputs_dtype (Scalar, optional): The scalar type to export as; converted from storage.\n" //
    "\n"                                                                                          //
    "Returns:\n"                                                                                  //
    "  DenseVectors: A writable (count, dimensions) matrix in the arena, for the kinds\n"         //
    "    `memoryview` can format - f64, f32, f16, i8, u8. Sub-byte and mini-float kinds,\n"       //
    "    `bf16` among them, leave as a (count, row_bytes) run of 'B' instead. Empty, so falsy,\n" //
    "    when the index holds no entry under `key`.";

/** A @c usearch_dense_get call and the rows it lands in the claimed arena. */
typedef struct get_call_t {
    usearch_py_call_t call;
    usearch_dense_get_t arguments;
    usearch_rows_view_t vectors;
    size_t written;
} get_call_t;

static void get_submit(void* call) { usearch_dense_get(&((get_call_t*)call)->arguments); }

static PyObject* get_answer(void* erased) {
    get_call_t* const call = (get_call_t*)erased;
    call->vectors.scalar = call->arguments.outputs_scalar; // An absent key leaves the view zeroed.
    return rows_answer(call->call.arena, &call->vectors, call->written);
}

static usearch_py_verb_t const get_verb = {sizeof(get_call_t), get_submit, get_answer, usearch_py_call_release};

static PyObject* DenseIndex_get(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                PyObject* args_names_tuple) {
    PyObject *key_object, *arena_object, *outputs_dtype_object;
    usearch_py_argument_t const arguments[] = {{"key", &key_object},
                                               {"arena", &arena_object},
                                               {"outputs_dtype", &outputs_dtype_object}};
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    usearch_scalar_t outputs_kind = usearch_scalar_unknown_k;
    usearch_key_t key;
    if (usearch_py_arguments_parse("get", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)) ||
        usearch_py_scalar_from_object(outputs_dtype_object, "outputs_dtype", &outputs_kind) != 0 ||
        usearch_py_key_from_object(key_object, &key) != 0)
        return NULL;

    get_call_t stack;
    get_call_t* const call = usearch_py_call_open(&receiver, &get_verb, &stack);
    if (!call) return NULL;
    if (usearch_py_arena_claim(arena_object, &call->call.arena) != 0)
        return usearch_py_call_discard(&receiver, &get_verb, call);
    call->arguments = (usearch_dense_get_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .key = key,
        .vectors = &call->vectors,
        .written = &call->written,
        .arena = ((Arena*)call->call.arena)->handle,
        .outputs_scalar = outputs_kind == usearch_scalar_unknown_k ? usearch_scalar_f32_k : outputs_kind,
    };
    return usearch_py_call_run(&receiver, &get_verb, call);
}

#pragma endregion Retrieval

#pragma region Linear Algebra

static char const doc_DenseIndex_multiply[] =                                                 //
    "multiply(rows, wanted, arena, /, *, expansion_factor=0, dtype=None, progress=None,\n"    //
    "    progress_steps=0, stats=None) -> DenseProducts\n"                                    //
    "\n"                                                                                      //
    "Approximate sparse product of a matrix against the indexed rows - the linear algebra\n"  //
    "sibling of `search`, reporting every pair the traversal scored rather than only the\n"   //
    "nearest. Pairs arrive in traversal order rather than sorted, and carry no duplicates.\n" //
    "How many a walk scores is known only once it ends, so the pairs come back ragged.\n"     //
    "Calls `usearch_dense_multiply`.\n"                                                       //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  rows (buffer): An (R, dimensions) matrix of left-hand rows.\n"                         //
    "  wanted (int): Upper bound on pairs kept per row, the first ones its walk scored;\n"    //
    "    0 keeps every pair.\n"                                                               //
    "  arena (Arena): Where the pairs and their offsets land, rewound by the call.\n"         //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it. A wider\n"  //
    "    factor widens the emitted pattern, being what the walk's breadth decides.\n"         //
    "  dtype (Scalar, optional): Names the left-hand scalar type when the buffer cannot.\n"   //
    "  progress (callable, optional): Called with (completed, total) rows; returning False\n" //
    "    stops the batch and raises `CancelledError`, None carries on.\n"                     //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"   //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"      //
    "\n"                                                                                      //
    "Returns:\n"                                                                              //
    "  DenseProducts: `offsets`, (R + 1,) Apache Arrow offsets, row `i` spanning pairs\n"     //
    "    `offsets[i]` up to `offsets[i + 1]`; `keys`, a (pairs, 16) run naming each pair's\n" //
    "    indexed row; `scores`, the (pairs,) float64 products; and `required`, the pairs\n"   //
    "    scored before `wanted` cut any. All of them live in the arena.";

/** A @c usearch_dense_multiply call and the ragged pairs it lands in the claimed arena. */
typedef struct multiply_call_t {
    usearch_py_call_t call;
    usearch_dense_multiply_t arguments;
    usearch_py_buffer_backing_t rows;
    walk_reports_t reports;
    usearch_matches_tape_view_t products;
    size_t written;
    size_t required;
} multiply_call_t;

static void multiply_submit(void* call) { usearch_dense_multiply(&((multiply_call_t*)call)->arguments); }

static PyObject* multiply_answer(void* erased) {
    multiply_call_t* const call = (multiply_call_t*)erased;
    walk_reports_answer(&call->reports);
    return usearch_py_products_new(call->call.arena, &call->products, call->arguments.count, call->written,
                                   call->required);
}

static void multiply_release(void* erased) {
    multiply_call_t* const call = (multiply_call_t*)erased;
    usearch_py_buffer_release(&call->rows);
    walk_reports_release(&call->reports);
    usearch_py_call_release(&call->call);
}

static int multiply_traverse(void* call, visitproc visit, void* arg) {
    return walk_reports_traverse(&((multiply_call_t*)call)->reports, visit, arg);
}

static usearch_py_verb_t const multiply_verb = {sizeof(multiply_call_t), multiply_submit, multiply_answer,
                                                multiply_release, multiply_traverse};

static PyObject* DenseIndex_multiply(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    PyObject *rows_object, *wanted_object, *arena_object, *expansion_object, *dtype_object, *progress_object,
        *progress_steps_object, *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"rows", &rows_object},
        {"wanted", &wanted_object},
        {"arena", &arena_object},
        {"expansion_factor", &expansion_object},
        {"dtype", &dtype_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t wanted = 0, expansion_factor = 0, progress_steps = 0;
    usearch_scalar_t dtype = usearch_scalar_unknown_k;
    if (usearch_py_arguments_parse("multiply", args, positional_args_count, args_names_tuple, 3, 3, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)) ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0)
        return NULL;

    multiply_call_t stack;
    multiply_call_t* const call = usearch_py_call_open(&receiver, &multiply_verb, &stack);
    if (!call) return NULL;
    usearch_matrix_view_t rows;
    if (usearch_py_matrix_from_object(rows_object, dtype, &rows, &call->rows) != 0 ||
        walk_reports_fill(&call->reports, &call->call, progress_object, stats_object) != 0 ||
        usearch_py_arena_claim(arena_object, &call->call.arena) != 0)
        return usearch_py_call_discard(&receiver, &multiply_verb, call);

    call->arguments = (usearch_dense_multiply_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .count = rows.rows,
        .left = rows_of(&rows),
        .products = &call->products,
        .written = &call->written,
        .required = &call->required,
        .arena = ((Arena*)call->call.arena)->handle,
        .wanted = wanted,
        .expansion_factor = expansion_factor,
        .progress = usearch_py_progress_registered(&call->reports.progress, progress_steps),
        .stats = &call->reports.stats,
    };
    return usearch_py_call_run(&receiver, &multiply_verb, call);
}

static char const doc_DenseIndex_attend[] =                                                      //
    "attend(queries, values, row_of, arena, /, *, head_count=1, expansion_factor=0,\n"           //
    "    score_scale=0.0, dtype=None, values_dtype=None, outputs_dtype=None, carry=None,\n"      //
    "    progress=None, progress_steps=0, stats=None) -> DenseVectors\n"                         //
    "\n"                                                                                         //
    "Softmax attention of every query row over every indexed key its walk scores, folded\n"      //
    "online as Flash Attention folds key tiles, with no top-k cut. Never writes to the\n"        //
    "graph, so a growing context may stream through this while insertions continue.\n"           //
    "\n"                                                                                         //
    "The index holds one layer's keys for one key-value head and no values: `row_of` names\n"    //
    "the row of `values` each scored key reads, so only keys a walk scores are ever resolved.\n" //
    "Calls `usearch_dense_attend`.\n"                                                            //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  queries (buffer): A (Q, dimensions) matrix; row `t * head_count + h` is head `h`\n"       //
    "    of token `t`.\n"                                                                        //
    "  values (buffer): An (N, dimensions) matrix of value rows, as wide as the keys and\n"      //
    "    stored in the index's quantization, borrowed until the call is final.\n"                //
    "  row_of (callable): Called with each scored Key, answers its row of `values`. None, or\n"  //
    "    a row outside them, refuses the call with `ShapeMismatchError`, as do values of\n"      //
    "    another width or scalar type. It re-enters Python from the library's threads, so it\n"  //
    "    runs under the GIL and serializes the walk. A VRAM walk resolves values on the\n"       //
    "    device, where no Python callable runs, so a VRAM arena raises `DeviceError`.\n"         //
    "  arena (Arena): Where the attended rows land, rewound by the call.\n"                      //
    "  head_count (int, optional): The query heads sharing the index's key-value head, and\n"    //
    "    with it one walk per token; it must divide Q.\n"                                        //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it. Wider\n"       //
    "    candidates walk further, folding more of the tail.\n"                                   //
    "  score_scale (float, optional): The logit scale, conventionally the inverse square\n"      //
    "    root of the head dimensionality; 0.0 derives it from the frozen geometry.\n"            //
    "  dtype (Scalar, optional): Names the query scalar type when the buffer cannot.\n"          //
    "  values_dtype (Scalar, optional): Names the values' scalar type when the buffer cannot.\n" //
    "  outputs_dtype (Scalar, optional): The scalar type to write the outputs as; defaults\n"    //
    "    to the queries' own.\n"                                                                 //
    "  carry (buffer, optional): A writable (Q, 2) buffer of float32, holding each query's\n"    //
    "    largest logit and the softmax denominator taken against it. The pair recombines\n"      //
    "    this branch with another over the same queries - attention sinks, a dense recent\n"     //
    "    window, or a second index of one context - by partial-softmax rescaling; a maximum\n"   //
    "    without its partition is a number nothing can rescale, which is why they travel as\n"   //
    "    one row.\n"                                                                             //
    "  progress (callable, optional): Called with (completed, total) tokens; returning\n"        //
    "    False stops the batch and raises `CancelledError`, None carries on.\n"                  //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"      //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"         //
    "\n"                                                                                         //
    "Returns:\n"                                                                                 //
    "  DenseVectors: The (Q, dimensions) attended rows in the arena, already divided by the\n"   //
    "    partition; kinds `memoryview` cannot format leave as a (Q, row_bytes) run of 'B'.";

/** What one attention call hands its value-lookup trampoline: the values it borrows and the
 *  key-to-row callable. */
typedef struct value_rows_t {

    /** The caller's `row_of`, and the call its failure latches into. */
    usearch_py_callback_t row_of;

    /** The caller's value rows, exported until the call is final. */
    usearch_matrix_view_t values;
} value_rows_t;

/**
 *  @brief Re-enters Python for the rows of one scored batch, from whichever library thread walks.
 *
 *  Values of another scalar type or width than the library reads resolve no row. A key whose
 *  `row_of` answers None, a row outside the values, or an exception leaves its row and the rest
 *  NULL, which refuses the call; an exception is latched into the call for its answer to raise, as
 *  the predicate's is.
 */
static void value_rows_trampoline(void* state, usearch_key_t const* keys, size_t count, usearch_scalar_t scalar,
                                  size_t columns, void const** rows) {
    value_rows_t const* const lookup = (value_rows_t const*)state;
    for (size_t position = 0; position != count; ++position) rows[position] = NULL;
    if (lookup->values.scalar != scalar || lookup->values.columns != columns ||
        usearch_py_callback_raised(lookup->row_of.call))
        return;
    PyGILState_STATE const gil_state = PyGILState_Ensure();
    for (size_t position = 0; position != count; ++position) {
        PyObject* const key = usearch_py_key_to_object(keys[position]);
        PyObject* const answered = key ? PyObject_CallOneArg(lookup->row_of.callable, key) : NULL;
        Py_XDECREF(key);
        Py_ssize_t const row = answered && answered != Py_None ? PyNumber_AsSsize_t(answered, PyExc_IndexError) : -1;
        Py_XDECREF(answered);
        if (row < 0 || (size_t)row >= lookup->values.rows) break;
        rows[position] = (char const*)lookup->values.data + (size_t)row * lookup->values.stride;
    }
    if (PyErr_Occurred()) usearch_py_callback_latch(lookup->row_of.call);
    PyGILState_Release(gil_state);
}

/** A @c usearch_dense_attend call, the inputs it borrows until it is final, and the rows it lands
 *  in the arena. */
typedef struct attend_call_t {
    usearch_py_call_t call;
    usearch_dense_attend_t arguments;
    usearch_py_buffer_backing_t queries;
    usearch_py_buffer_backing_t values;
    usearch_py_buffer_backing_t carry;
    value_rows_t lookup;
    walk_reports_t reports;
    usearch_rows_view_t outputs;
} attend_call_t;

static void attend_submit(void* call) { usearch_dense_attend(&((attend_call_t*)call)->arguments); }

static PyObject* attend_answer(void* erased) {
    attend_call_t* const call = (attend_call_t*)erased;
    walk_reports_answer(&call->reports);
    return rows_answer(call->call.arena, &call->outputs, call->arguments.count);
}

static void attend_release(void* erased) {
    attend_call_t* const call = (attend_call_t*)erased;
    usearch_py_buffer_release(&call->queries);
    usearch_py_buffer_release(&call->values);
    usearch_py_buffer_release(&call->carry);
    Py_CLEAR(call->lookup.row_of.callable);
    walk_reports_release(&call->reports);
    usearch_py_call_release(&call->call);
}

static int attend_traverse(void* erased, visitproc visit, void* arg) {
    attend_call_t* const call = (attend_call_t*)erased;
    Py_VISIT(call->lookup.row_of.callable);
    return walk_reports_traverse(&call->reports, visit, arg);
}

static usearch_py_verb_t const attend_verb = {sizeof(attend_call_t), attend_submit, attend_answer, attend_release,
                                              attend_traverse};

static PyObject* DenseIndex_attend(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *queries_object, *values_object, *row_of_object, *arena_object, *head_count_object, *expansion_object,
        *scale_object, *dtype_object, *values_dtype_object, *outputs_dtype_object, *carry_object, *progress_object,
        *progress_steps_object, *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"queries", &queries_object},
        {"values", &values_object},
        {"row_of", &row_of_object},
        {"arena", &arena_object},
        {"head_count", &head_count_object},
        {"expansion_factor", &expansion_object},
        {"score_scale", &scale_object},
        {"dtype", &dtype_object},
        {"values_dtype", &values_dtype_object},
        {"outputs_dtype", &outputs_dtype_object},
        {"carry", &carry_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t head_count = 0, expansion_factor = 0, progress_steps = 0;
    double score_scale = 0;
    usearch_scalar_t dtype = usearch_scalar_unknown_k, values_kind = usearch_scalar_unknown_k,
                     outputs_kind = usearch_scalar_unknown_k;
    if (usearch_py_arguments_parse("attend", args, positional_args_count, args_names_tuple, 4, 4, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)) ||
        usearch_py_size_from_object(head_count_object, "head_count", &head_count) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        (scale_object && (score_scale = PyFloat_AsDouble(scale_object)) == -1.0 && PyErr_Occurred()) ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0 ||
        usearch_py_scalar_from_object(values_dtype_object, "values_dtype", &values_kind) != 0 ||
        usearch_py_scalar_from_object(outputs_dtype_object, "outputs_dtype", &outputs_kind) != 0)
        return NULL;
    if (!PyCallable_Check(row_of_object)) {
        PyErr_Format(PyExc_TypeError, "the 'row_of' argument must be callable, got %S", Py_TYPE(row_of_object));
        return NULL;
    }
    attend_call_t stack;
    attend_call_t* const call = usearch_py_call_open(&receiver, &attend_verb, &stack);
    if (!call) return NULL;
    usearch_matrix_view_t queries;
    Py_ssize_t carry_bytes = 0;
    void* carry = NULL;
    if (usearch_py_matrix_from_object(queries_object, dtype, &queries, &call->queries) != 0 ||
        usearch_py_matrix_from_object(values_object, values_kind, &call->lookup.values, &call->values) != 0 ||
        usearch_py_callback_from_object(row_of_object, "row_of", &call->call, &call->lookup.row_of) != 0 ||
        walk_reports_fill(&call->reports, &call->call, progress_object, stats_object) != 0)
        return usearch_py_call_discard(&receiver, &attend_verb, call);
    if (carry_object && carry_object != Py_None &&
        (usearch_py_matrix_bytes(queries.rows, sizeof(usearch_dense_softmax_state_t), 1, &carry_bytes) != 0 ||
         usearch_py_output_from_object(carry_object, "carry", carry_bytes, &call->carry, &carry) != 0))
        return usearch_py_call_discard(&receiver, &attend_verb, call);
    if (usearch_py_arena_claim(arena_object, &call->call.arena) != 0)
        return usearch_py_call_discard(&receiver, &attend_verb, call);
    // A VRAM walk calls its value lookup on the device, which no Python callable can serve.
    if (((Arena*)call->call.arena)->residency == usearch_residency_vram_k) {
        usearch_py_raise(usearch_status_device_k, "a VRAM walk needs a device-callable value lookup");
        return usearch_py_call_discard(&receiver, &attend_verb, call);
    }

    call->arguments = (usearch_dense_attend_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .count = queries.rows,
        .queries = rows_of(&queries),
        .head_count = head_count,
        .value_lookup = value_rows_trampoline,
        .value_lookup_state = &call->lookup,
        .carry = (usearch_dense_softmax_state_t*)carry,
        .outputs = &call->outputs,
        .arena = ((Arena*)call->call.arena)->handle,
        .outputs_scalar = outputs_kind,
        .expansion_factor = expansion_factor,
        .score_scale = (float)score_scale,
        .progress = usearch_py_progress_registered(&call->reports.progress, progress_steps),
        .stats = &call->reports.stats,
    };
    return usearch_py_call_run(&receiver, &attend_verb, call);
}

static char const doc_DenseIndex_maxsim[] =                                                      //
    "maxsim(queries, rows_per_query, wanted, arena, /, *, expansion_factor=0,\n"                 //
    "    traversal=Traversal.WALK, dtype=None, predicate=None, progress=None,\n"                 //
    "    progress_steps=0, stats=None) -> Matches\n"                                             //
    "\n"                                                                                         //
    "Late-interaction scoring of multi-row queries against multi-row entries: every query\n"     //
    "row takes its best match within a key, and those maxima sum to that key's score.\n"         //
    "\n"                                                                                         //
    "A document is one key holding its token rows, which is what `values_per_key` above one\n"   //
    "exists for, and the walk stays over individual tokens under a plain inner product - so\n"   //
    "an entry's rows need no ordering and two documents need not share a row count. The\n"       //
    "result is sized over documents rather than over rows.\n"                                    //
    "Calls `usearch_dense_maxsim`.\n"                                                            //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  queries (buffer): The token rows of every query laid end to end, an (Q * R,\n"            //
    "    dimensions) matrix.\n"                                                                  //
    "  rows_per_query (int): Rows every query occupies; uniform, so ColBERT's padding\n"         //
    "    applies as is. The query row count must be a whole multiple of it.\n"                   //
    "  wanted (int): Upper bound on matches per query - the 'k' over documents, not rows.\n"     //
    "  arena (Arena): Where the scored documents land, rewound by the call.\n"                   //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it. Widening it\n" //
    "    nominates more documents, each still scored exactly.\n"                                 //
    "  traversal (Traversal, optional): `Traversal.WALK`, the default, nominates documents by\n" //
    "    walking the graph; `Traversal.SCAN` nominates every document, on RAM and NVMe.\n"       //
    "  dtype (Scalar, optional): Names the query scalar type when the buffer cannot.\n"          //
    "  predicate (callable, optional): Admits a Key when it returns truthy, asked before any\n"  //
    "    of its rows are scored, so a rejected document never scores.\n"                         //
    "  progress (callable, optional): Called with (completed, total) queries; returning\n"       //
    "    False stops the batch and raises `CancelledError`, None carries on.\n"                  //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"      //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"         //
    "\n"                                                                                         //
    "Returns:\n"                                                                                 //
    "  Matches: Document identities and their summed maxima, largest first, in the arena.";

/** A @c usearch_dense_maxsim call, the queries it borrows, and the matches it lands in the
 *  arena. */
typedef struct maxsim_call_t {
    usearch_py_call_t call;
    usearch_dense_maxsim_t arguments;
    usearch_py_buffer_backing_t queries;
    usearch_py_callback_t predicate;
    walk_reports_t reports;
    usearch_matches_view_t matches;
} maxsim_call_t;

static void maxsim_submit(void* call) { usearch_dense_maxsim(&((maxsim_call_t*)call)->arguments); }

static PyObject* maxsim_answer(void* erased) {
    maxsim_call_t* const call = (maxsim_call_t*)erased;
    walk_reports_answer(&call->reports);
    return usearch_py_matches_new(call->call.arena, &call->matches, call->arguments.count);
}

static void maxsim_release(void* erased) {
    maxsim_call_t* const call = (maxsim_call_t*)erased;
    usearch_py_buffer_release(&call->queries);
    Py_CLEAR(call->predicate.callable);
    walk_reports_release(&call->reports);
    usearch_py_call_release(&call->call);
}

static int maxsim_traverse(void* erased, visitproc visit, void* arg) {
    maxsim_call_t* const call = (maxsim_call_t*)erased;
    Py_VISIT(call->predicate.callable);
    return walk_reports_traverse(&call->reports, visit, arg);
}

static usearch_py_verb_t const maxsim_verb = {sizeof(maxsim_call_t), maxsim_submit, maxsim_answer, maxsim_release,
                                              maxsim_traverse};

static PyObject* DenseIndex_maxsim(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *queries_object, *rows_per_query_object, *wanted_object, *arena_object, *expansion_object,
        *traversal_object, *dtype_object, *predicate_object, *progress_object, *progress_steps_object, *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"queries", &queries_object},
        {"rows_per_query", &rows_per_query_object},
        {"wanted", &wanted_object},
        {"arena", &arena_object},
        {"expansion_factor", &expansion_object},
        {"traversal", &traversal_object},
        {"dtype", &dtype_object},
        {"predicate", &predicate_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t rows_per_query = 0, wanted = 0, expansion_factor = 0, progress_steps = 0;
    int traversal = usearch_traversal_walk_k;
    usearch_scalar_t dtype = usearch_scalar_unknown_k;
    if (usearch_py_arguments_parse("maxsim", args, positional_args_count, args_names_tuple, 4, 4, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)) ||
        usearch_py_size_from_object(rows_per_query_object, "rows_per_query", &rows_per_query) != 0 ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        (traversal_object &&
         usearch_py_enum_from_object(traversal_object, "traversal", &usearch_py_traversal, &traversal) != 0) ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0)
        return NULL;

    maxsim_call_t stack;
    maxsim_call_t* const call = usearch_py_call_open(&receiver, &maxsim_verb, &stack);
    if (!call) return NULL;
    usearch_matrix_view_t queries;
    if (usearch_py_matrix_from_object(queries_object, dtype, &queries, &call->queries) != 0)
        return usearch_py_call_discard(&receiver, &maxsim_verb, call);
    if (rows_per_query == 0 || queries.rows % rows_per_query != 0) {
        usearch_py_raise(usearch_status_shape_mismatch_k, "the %zu query rows are not a whole multiple of %zu",
                         queries.rows, rows_per_query);
        return usearch_py_call_discard(&receiver, &maxsim_verb, call);
    }
    size_t const queries_count = queries.rows / rows_per_query;
    Py_ssize_t extent = 0;
    if (usearch_py_matrix_bytes(queries_count, wanted, sizeof(usearch_key_t), &extent) != 0 ||
        usearch_py_callback_from_object(predicate_object, "predicate", &call->call, &call->predicate) != 0 ||
        walk_reports_fill(&call->reports, &call->call, progress_object, stats_object) != 0 ||
        usearch_py_arena_claim(arena_object, &call->call.arena) != 0)
        return usearch_py_call_discard(&receiver, &maxsim_verb, call);

    call->arguments = (usearch_dense_maxsim_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .count = queries_count,
        .queries = rows_of(&queries),
        .rows_per_query = rows_per_query,
        .matches = &call->matches,
        .arena = ((Arena*)call->call.arena)->handle,
        .wanted = wanted,
        .expansion_factor = expansion_factor,
        .progress = usearch_py_progress_registered(&call->reports.progress, progress_steps),
        .stats = &call->reports.stats,
        .traversal = (usearch_traversal_t)traversal,
        .predicate = usearch_py_predicate_registered(&call->predicate),
    };
    return usearch_py_call_run(&receiver, &maxsim_verb, call);
}

#pragma endregion Linear Algebra

#pragma region Composition

static char const doc_DenseIndex_absorb[] =                                                   //
    "absorb(*donors, progress=None, progress_steps=0) -> None\n"                              //
    "\n"                                                                                      //
    "Absorbs every donor into this index, one after another, keeping their graphs rather\n"   //
    "than re-inserting their vectors, and linking each in by walks across the appended\n"     //
    "part. Each donor handle is consumed and every verb on it afterwards raises. Readers\n"   //
    "of a RAM index may keep searching throughout, while a VRAM one admits no overlapping\n"  //
    "search.\n"                                                                               //
    "\n"                                                                                      //
    "Cancellation stops between donors, so a donor already absorbed stays absorbed and one\n" //
    "not yet reached stays a live handle its caller still owns. A donor whose absorption\n"   //
    "fails stays usable too, though its rows may already have landed in this index.\n"        //
    "Calls `usearch_dense_absorb`.\n"                                                         //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  donors (DenseIndex): The indexes to absorb.\n"                                         //
    "  progress (callable, optional): Called with (completed, total) donors; returning\n"     //
    "    False stops the absorb and raises `CancelledError`, None carries on.\n"              //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

/** A @c usearch_dense_absorb call, holding the donors whose handles it took until the call is
 *  final. */
typedef struct absorb_call_t {
    usearch_py_call_t call;
    usearch_dense_absorb_t arguments;
    usearch_py_callback_t progress;

    /** The tuple of @c DenseIndex donors, held, or NULL before the fill gathered it. */
    PyObject* donors;

    /** The slots the library reads the donors from, and nulls as it consumes them; NULL where
     *  nothing was taken. */
    usearch_dense_index_t* donated;
} absorb_call_t;

static void absorb_submit(void* call) { usearch_dense_absorb(&((absorb_call_t*)call)->arguments); }

static void absorb_release(void* erased) {
    absorb_call_t* const call = (absorb_call_t*)erased;
    // The library nulls the slot of every donor it consumed, so whatever it left there is a
    // handle the wrapper still owns, whichever way the absorb ended.
    for (Py_ssize_t position = 0; call->donated && position != PyTuple_GET_SIZE(call->donors); ++position) {
        if (!call->donated[position]) continue;
        PyObject* const donor = PyTuple_GET_ITEM(call->donors, position);
        usearch_py_critical_section_begin_(donor);
        ((DenseIndex*)donor)->handle = call->donated[position];
        usearch_py_critical_section_end_();
    }
    PyMem_Free(call->donated);
    call->donated = NULL;
    Py_CLEAR(call->donors);
    Py_CLEAR(call->progress.callable);
    usearch_py_call_release(&call->call);
}

static int absorb_traverse(void* erased, visitproc visit, void* arg) {
    absorb_call_t* const call = (absorb_call_t*)erased;
    Py_VISIT(call->donors);
    Py_VISIT(call->progress.callable);
    return 0;
}

static usearch_py_verb_t const absorb_verb = {sizeof(absorb_call_t), absorb_submit, usearch_py_call_answer_none,
                                              absorb_release, absorb_traverse};

static PyObject* DenseIndex_absorb(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {{"progress", &progress_object},
                                               {"progress_steps", &progress_steps_object}};
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t progress_steps = 0;
    // Every positional argument is a donor, so only the keywords go through the table.
    if (usearch_py_arguments_parse("absorb", args + positional_args_count, 0, args_names_tuple, 0, 0, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)) ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0)
        return NULL;
    for (Py_ssize_t position = 0; position != positional_args_count; ++position) {
        if (!PyObject_TypeCheck(args[position], &DenseIndexType)) {
            PyErr_Format(PyExc_TypeError, "expected DenseIndex donors, got %S", Py_TYPE(args[position]));
            return NULL;
        }
        if (args[position] == receiver.index) {
            usearch_py_raise(usearch_status_config_mismatch_k, "an index cannot absorb itself");
            return NULL;
        }
    }

    absorb_call_t stack;
    absorb_call_t* const call = usearch_py_call_open(&receiver, &absorb_verb, &stack);
    if (!call) return NULL;
    if (usearch_py_callback_from_object(progress_object, "progress", &call->call, &call->progress) != 0 ||
        !(call->donors = PyTuple_New(positional_args_count)))
        return usearch_py_call_discard(&receiver, &absorb_verb, call);
    for (Py_ssize_t position = 0; position != positional_args_count; ++position)
        PyTuple_SET_ITEM(call->donors, position, Py_NewRef(args[position]));
    call->donated = (usearch_dense_index_t*)PyMem_Calloc(positional_args_count ? positional_args_count : 1,
                                                         sizeof(usearch_dense_index_t));
    if (!call->donated) {
        PyErr_NoMemory();
        return usearch_py_call_discard(&receiver, &absorb_verb, call);
    }

    // The library consumes the donors and takes the array mutably, so each wrapper's claim is
    // surrendered before the call rather than after: a failure that still consumed one must not
    // leave two owners, and a concurrent verb on a donor sees it consumed meanwhile.
    for (Py_ssize_t position = 0; position != positional_args_count; ++position) {
        DenseIndex* const donor = (DenseIndex*)args[position];
        usearch_py_critical_section_begin_(args[position]);
        call->donated[position] = donor->handle;
        donor->handle = NULL;
        usearch_py_critical_section_end_();
        if (!call->donated[position]) {
            usearch_py_raise(usearch_status_uninitialized_k, "a donor handle was already absorbed");
            return usearch_py_call_discard(&receiver, &absorb_verb, call);
        }
    }

    call->arguments = (usearch_dense_absorb_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .count = (size_t)positional_args_count,
        .donors = call->donated,
        .progress = usearch_py_progress_registered(&call->progress, progress_steps),
    };
    return usearch_py_call_run(&receiver, &absorb_verb, call);
}

#pragma endregion Composition

#pragma region Iteration

static char const doc_DenseCursor[] =                                                       //
    "A sequential row scan over the entries an index holds, pulled in batches by `next`.\n" //
    "\n"                                                                                    //
    "Each shard's publication watermark is read as the scan enters that shard, so a row\n"  //
    "once passed is never revisited, while a later shard may still carry rows inserted\n"   //
    "after the cursor opened. One step runs at a time; an overlapping step or `close()`\n"  //
    "raises `ContendedError`. `close()`, or the end of a `with` block, releases the scan\n" //
    "before the cursor is collected.";

/** Stands in the handle slot of a cursor or a stream while one step holds its handle, refusing a
 *  second step or a close until the step gives it back. */
static char handle_lent;

/**
 *  @brief Releases the engine cursor of @p self once, fenced against a concurrent close and refused
 *      during a step.
 *  @return 0 on success, -1 with a @c ContendedError set while a step holds the handle.
 */
static int cursor_close(DenseCursor* self) {
    usearch_dense_cursor_t const lent = (usearch_dense_cursor_t)(void*)&handle_lent;
    usearch_dense_cursor_t handle = NULL;
    usearch_py_critical_section_begin_((PyObject*)self);
    handle = self->handle;
    if (handle != lent) self->handle = NULL;
    usearch_py_critical_section_end_();
    if (handle == lent) return usearch_py_raise(usearch_status_contended_k, "a step of this cursor is still running");
    if (!handle) return 0;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_cursor_free(handle);
    Py_END_ALLOW_THREADS;
    return 0;
}

static void DenseCursor_dealloc(DenseCursor* self) {
    cursor_close(self); // No step can hold the handle, since every step holds the cursor.
    Py_XDECREF(self->receiver);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static char const doc_DenseBatch[] =                                                         //
    "DenseBatch\n"                                                                           //
    "\n"                                                                                     //
    "The rows one cursor step wrote into an `Arena`, viewing it until it rewinds. Its\n"     //
    "length counts them, so the empty batch a drained cursor answers is falsy.\n"            //
    "\n"                                                                                     //
    "Attributes:\n"                                                                          //
    "  keys (buffer): (count, 16) identities, most significant byte first.\n"                //
    "  links (tuple | None): (offsets, keys) of the identities each row links to, row `i`\n" //
    "    spanning `offsets[i]` up to `offsets[i + 1]`, or None when the step asked for none.";

/** The (offsets, keys) pair of the @p written rows' links in @p arena; a new reference, or NULL
 *  with an exception set. */
static PyObject* links_answer(PyObject* arena, usearch_keys_tape_view_t const* links, size_t written) {
    Py_ssize_t const offsets_shape[1] = {(Py_ssize_t)written + 1};
    Py_ssize_t const linked_shape[2] = {(Py_ssize_t)links->offsets[written], (Py_ssize_t)sizeof(usearch_key_t)};
    PyObject* const offsets = usearch_py_arena_buffer(&ArenaBufferType, arena, links->offsets, "N", sizeof(size_t),
                                                      offsets_shape, 1);
    PyObject* const linked =
        offsets ? usearch_py_arena_buffer(&ArenaBufferType, arena, links->keys, "B", 1, linked_shape, 2) : NULL;
    PyObject* const pair = linked ? PyTuple_Pack(2, offsets, linked) : NULL;
    Py_XDECREF(linked);
    Py_XDECREF(offsets);
    return pair;
}

/** A new @c DenseBatch over the @p written rows a step wrote into @p arena; NULL @p links means
 *  keys alone. */
static PyObject* batch_answer(PyObject* arena, usearch_keys_view_t const* keys, usearch_keys_tape_view_t const* links,
                              size_t written) {
    Py_ssize_t const keys_shape[2] = {(Py_ssize_t)written, (Py_ssize_t)sizeof(usearch_key_t)};
    DenseBatch* batch = PyObject_New(DenseBatch, &DenseBatchType);
    if (!batch) return NULL;
    batch->links = NULL;
    batch->keys = usearch_py_arena_buffer(&ArenaBufferType, arena, keys->data, "B", 1, keys_shape, 2);
    if (batch->keys) batch->links = links ? links_answer(arena, links, written) : Py_NewRef(Py_None);
    if (!batch->links) Py_CLEAR(batch);
    return (PyObject*)batch;
}

static void DenseBatch_dealloc(DenseBatch* self) {
    Py_XDECREF(self->links);
    Py_XDECREF(self->keys);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static Py_ssize_t DenseBatch_length(DenseBatch* self) { return PyObject_Length(self->keys); }

static PyMemberDef DenseBatch_members[] = {
    {"keys", Py_T_OBJECT_EX, offsetof(DenseBatch, keys), Py_READONLY,
     "(count, 16) identities, most significant byte first."},
    {"links", Py_T_OBJECT_EX, offsetof(DenseBatch, links), Py_READONLY,
     "(offsets, keys) of each row's linked identities, or None when the step asked for none."},
    {NULL, 0, 0, 0, NULL},
};

static PySequenceMethods DenseBatch_as_sequence = {.sq_length = (lenfunc)DenseBatch_length};

PyTypeObject DenseBatchType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseBatch",
    .tp_doc = doc_DenseBatch,
    .tp_basicsize = sizeof(DenseBatch),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, // Returned by `DenseCursor.next`, never constructed by name.
    .tp_dealloc = (destructor)DenseBatch_dealloc,
    .tp_as_sequence = &DenseBatch_as_sequence,
    .tp_members = DenseBatch_members,
};

/** Runs one step of @p self over the engine cursor @p handle it took out of the slot. */
static PyObject* cursor_step(DenseCursor* self, usearch_dense_cursor_t handle, size_t wanted, PyObject* arena_object,
                             int with_links) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t const index = receiver_from_object(self->receiver, &receiver);
    PyObject* arena = NULL;
    if (!index || usearch_py_arena_claim(arena_object, &arena) != 0) return NULL;

    usearch_keys_view_t keys = {0};
    usearch_keys_tape_view_t links = {0};
    size_t written = 0;
    usearch_outcome_t outcome = {0};
    usearch_dense_cursor_next_t arguments = {
        .index = index,
        .executor = receiver.executor,
        .outcome = &outcome,
        .cursor = handle,
        .keys = &keys,
        .links = with_links ? &links : NULL,
        .written = &written,
        .arena = ((Arena*)arena)->handle,
        .wanted = wanted,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_cursor_next(&arguments);
    Py_END_ALLOW_THREADS;

    PyObject* batch = NULL;
    if (usearch_py_raise_on_outcome(&outcome) == 0)
        batch = batch_answer(arena, &keys, with_links && links.offsets ? &links : NULL, written);
    usearch_py_arena_unclaim(arena);
    return batch;
}

static char const doc_DenseCursor_next[] =                                                    //
    "next(wanted, arena, /, *, links=False) -> DenseBatch\n"                                  //
    "\n"                                                                                      //
    "The next batch of at most `wanted` rows, in `arena`, and an empty one, so falsy, once\n" //
    "the scan is drained. Blocks on the executor the cursor was opened through, or on the\n"  //
    "calling thread.\n"                                                                       //
    "Calls `usearch_dense_cursor_next`.\n"                                                    //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  wanted (int): The most rows the batch holds.\n"                                        //
    "  arena (Arena): Where the batch lands, rewound by the step.\n"                          //
    "  links (bool): Also export the identities each row links to; the rows themselves\n"     //
    "    come from `get`.\n"                                                                  //
    "\n"                                                                                      //
    "Raises:\n"                                                                               //
    "  UninitializedError: When the cursor was closed.";

static PyObject* DenseCursor_next(DenseCursor* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                  PyObject* args_names_tuple) {
    PyObject *wanted_object, *arena_object, *links_object;
    usearch_py_argument_t const arguments[] = {
        {"wanted", &wanted_object},
        {"arena", &arena_object},
        {"links", &links_object},
    };
    size_t wanted = 0;
    int with_links = 0;
    if (usearch_py_arguments_parse("next", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0 ||
        (links_object && (with_links = PyObject_IsTrue(links_object)) < 0))
        return NULL;

    usearch_dense_cursor_t const lent = (usearch_dense_cursor_t)(void*)&handle_lent;
    usearch_dense_cursor_t handle = NULL;
    usearch_py_critical_section_begin_((PyObject*)self);
    handle = self->handle;
    if (handle) self->handle = lent;
    usearch_py_critical_section_end_();
    if (!handle) {
        usearch_py_raise(usearch_status_uninitialized_k, "the cursor was closed");
        return NULL;
    }
    if (handle == lent) {
        usearch_py_raise(usearch_status_contended_k, "another step of this cursor is running");
        return NULL;
    }
    PyObject* const batch = cursor_step(self, handle, wanted, arena_object, with_links);
    usearch_py_critical_section_begin_((PyObject*)self);
    self->handle = handle;
    usearch_py_critical_section_end_();
    return batch;
}

static PyObject* DenseCursor_close(PyObject* self, PyObject* unused) {
    if (cursor_close((DenseCursor*)self) != 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject* DenseCursor_enter(PyObject* self, PyObject* unused) { return Py_NewRef(self); }

static PyObject* DenseCursor_exit(PyObject* self, PyObject* const* args, Py_ssize_t count) {
    return DenseCursor_close(self, NULL);
}

static PyMethodDef DenseCursor_methods[] = {
    {"next", (PyCFunction)DenseCursor_next, usearch_py_method_flags_k, doc_DenseCursor_next},
    {"close", (PyCFunction)DenseCursor_close, METH_NOARGS,
     "Releases the scan before the cursor is collected. Calls `usearch_dense_cursor_free`."},
    {"__enter__", (PyCFunction)DenseCursor_enter, METH_NOARGS, NULL},
    {"__exit__", (PyCFunction)DenseCursor_exit, METH_FASTCALL, NULL},
    {NULL, NULL, 0, NULL},
};

PyTypeObject DenseCursorType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseCursor",
    .tp_doc = doc_DenseCursor,
    .tp_basicsize = sizeof(DenseCursor),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, // Returned by `cursor`, never constructed by name.
    .tp_dealloc = (destructor)DenseCursor_dealloc,
    .tp_methods = DenseCursor_methods,
};

static char const doc_DenseIndex_cursor[] =                                                   //
    "cursor(starts=None) -> DenseCursor\n"                                                    //
    "\n"                                                                                      //
    "Opens a walk over the index, which `DenseCursor.next` pulls in batches, each into the\n" //
    "arena that pull names. Without `starts` it yields every row as stored; with them, the\n" //
    "start keys' own rows first in order, then the rows those link to breadth-first, each\n"  //
    "once. Each shard's watermark is read as the scan enters that shard, so a row once\n"     //
    "passed is never revisited, while a later shard may still carry rows inserted after\n"    //
    "this call. Every pull blocks, on the receiver's executor or the calling thread.\n"       //
    "Calls `usearch_dense_cursor_init`.\n"                                                    //
    "\n"                                                                                      //
    "Example:\n"                                                                              //
    "  >>> from array import array\n"                                                         //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=2)\n"                               //
    "  >>> index.reserve(3)\n"                                                                //
    "  >>> for key in (1, 2, 3):\n"                                                           //
    "  ...     index.insert([key], array('d', [0.1, 0.2]))\n"                                 //
    "  >>> arena, cursor = usearch.Arena(), index.cursor()\n"                                 //
    "  >>> while batch := cursor.next(2, arena):\n"                                           //
    "  ...     print(len(batch))\n"                                                           //
    "  2\n"                                                                                   //
    "  1";

static PyObject* DenseIndex_cursor(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject* starts_object;
    usearch_py_argument_t const arguments[] = {{"starts", &starts_object}};
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    if (usearch_py_arguments_parse("cursor", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)))
        return NULL;

    usearch_py_keys_t starts = {0};
    size_t starts_count = 0;
    if (starts_object && starts_object != Py_None) {
        Py_ssize_t const length = PyObject_Length(starts_object);
        if (length < 0 || usearch_py_keys_from_object(starts_object, (size_t)length, &starts) != 0) return NULL;
        starts_count = (size_t)length;
    }

    usearch_dense_cursor_t cursor_handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_cursor_init_t arguments_init = {.cursor = &cursor_handle,
                                                  .index = handle,
                                                  .outcome = &outcome,
                                                  .starts_count = starts_count,
                                                  .starts = starts.keys};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_cursor_init(&arguments_init);
    Py_END_ALLOW_THREADS;
    usearch_py_keys_release(&starts);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;

    DenseCursor* const cursor = PyObject_New(DenseCursor, &DenseCursorType);
    if (!cursor) {
        usearch_dense_cursor_free(cursor_handle);
        return NULL;
    }
    cursor->handle = cursor_handle;
    cursor->receiver = Py_NewRef(self);
    return (PyObject*)cursor;
}

static char const doc_DenseSearchStream[] =                                                //
    "A resumable best-first stream of matches for one query, pulled by `next`.\n"          //
    "\n"                                                                                   //
    "Ordering beyond the expansion horizon is best-effort nearest, not globally exact.\n"  //
    "One pull runs at a time; an overlapping pull or `close()` raises `ContendedError`.\n" //
    "`close()`, or the end of a `with` block, releases the walk before the stream is\n"    //
    "collected.";

/**
 *  @brief Releases the engine stream of @p self once, fenced against a concurrent close and refused
 *      during a pull.
 *  @return 0 on success, -1 with a @c ContendedError set while a pull holds the handle.
 */
static int stream_close(DenseSearchStream* self) {
    usearch_dense_search_stream_t const lent = (usearch_dense_search_stream_t)(void*)&handle_lent;
    usearch_dense_search_stream_t handle = NULL;
    usearch_py_critical_section_begin_((PyObject*)self);
    handle = self->handle;
    if (handle != lent) self->handle = NULL;
    usearch_py_critical_section_end_();
    if (handle == lent) return usearch_py_raise(usearch_status_contended_k, "a pull of this stream is still running");
    if (!handle) return 0;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_search_stream_free(handle);
    Py_END_ALLOW_THREADS;
    return 0;
}

static void DenseSearchStream_dealloc(DenseSearchStream* self) {
    stream_close(self); // No pull can hold the handle, since every pull holds the stream.
    Py_XDECREF(self->receiver);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** Pulls the next matches of @p self from the engine stream @p handle it took out of the slot. */
static PyObject* stream_pull(DenseSearchStream* self, usearch_dense_search_stream_t handle, size_t wanted,
                             PyObject* arena_object) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t const index = receiver_from_object(self->receiver, &receiver);
    PyObject* arena = NULL;
    if (!index || usearch_py_arena_claim(arena_object, &arena) != 0) return NULL;
    usearch_matches_view_t matches = {0};
    usearch_outcome_t outcome = {0};
    usearch_dense_search_stream_next_t arguments = {
        .index = index,
        .executor = receiver.executor,
        .outcome = &outcome,
        .stream = handle,
        .matches = &matches,
        .arena = ((Arena*)arena)->handle,
        .wanted = wanted,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_search_stream_next(&arguments);
    Py_END_ALLOW_THREADS;
    PyObject* const pulled = usearch_py_raise_on_outcome(&outcome) == 0 ? usearch_py_matches_new(arena, &matches, 1)
                                                                        : NULL;
    usearch_py_arena_unclaim(arena);
    return pulled;
}

static char const doc_DenseSearchStream_next[] =                                              //
    "next(wanted, arena, /) -> Matches\n"                                                     //
    "\n"                                                                                      //
    "The next at most `wanted` matches, as one row of `Matches` in `arena`, whose count is\n" //
    "zero once the walk is drained. Blocks on the executor the stream was opened through,\n"  //
    "or on the calling thread.\n"                                                             //
    "Calls `usearch_dense_search_stream_next`.\n"                                             //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  wanted (int): The most matches the pull writes.\n"                                     //
    "  arena (Arena): Where the matches land, rewound by the pull.\n"                         //
    "\n"                                                                                      //
    "Raises:\n"                                                                               //
    "  UninitializedError: When the stream was closed.";

static PyObject* DenseSearchStream_next(DenseSearchStream* self, PyObject* const* args,
                                        Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    PyObject *wanted_object, *arena_object;
    usearch_py_argument_t const arguments[] = {{"wanted", &wanted_object}, {"arena", &arena_object}};
    size_t wanted = 0;
    if (usearch_py_arguments_parse("next", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0)
        return NULL;

    usearch_dense_search_stream_t const lent = (usearch_dense_search_stream_t)(void*)&handle_lent;
    usearch_dense_search_stream_t handle = NULL;
    usearch_py_critical_section_begin_((PyObject*)self);
    handle = self->handle;
    if (handle) self->handle = lent;
    usearch_py_critical_section_end_();
    if (!handle) {
        usearch_py_raise(usearch_status_uninitialized_k, "the stream was closed");
        return NULL;
    }
    if (handle == lent) {
        usearch_py_raise(usearch_status_contended_k, "another pull of this stream is running");
        return NULL;
    }
    PyObject* const pulled = stream_pull(self, handle, wanted, arena_object);
    usearch_py_critical_section_begin_((PyObject*)self);
    self->handle = handle;
    usearch_py_critical_section_end_();
    return pulled;
}

static PyObject* DenseSearchStream_close(PyObject* self, PyObject* unused) {
    if (stream_close((DenseSearchStream*)self) != 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject* DenseSearchStream_enter(PyObject* self, PyObject* unused) { return Py_NewRef(self); }

static PyObject* DenseSearchStream_exit(PyObject* self, PyObject* const* args, Py_ssize_t count) {
    return DenseSearchStream_close(self, NULL);
}

static PyMethodDef DenseSearchStream_methods[] = {
    {"next", (PyCFunction)DenseSearchStream_next, usearch_py_method_flags_k, doc_DenseSearchStream_next},
    {"close", (PyCFunction)DenseSearchStream_close, METH_NOARGS,
     "Releases the walk before the stream is collected. Calls `usearch_dense_search_stream_free`."},
    {"__enter__", (PyCFunction)DenseSearchStream_enter, METH_NOARGS, NULL},
    {"__exit__", (PyCFunction)DenseSearchStream_exit, METH_FASTCALL, NULL},
    {NULL, NULL, 0, NULL},
};

PyTypeObject DenseSearchStreamType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseSearchStream",
    .tp_doc = doc_DenseSearchStream,
    .tp_basicsize = sizeof(DenseSearchStream),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, // Returned by `search_stream`, never constructed by name.
    .tp_dealloc = (destructor)DenseSearchStream_dealloc,
    .tp_methods = DenseSearchStream_methods,
};

static char const doc_DenseIndex_search_stream[] =                                           //
    "search_stream(query, /, *, expansion_factor=0, dtype=None) -> DenseSearchStream\n"      //
    "\n"                                                                                     //
    "Opens a resumable nearest-neighbor stream for one query, which `next` pulls for as\n"   //
    "long as the caller wants more, each pull into the arena it names. Every pull blocks,\n" //
    "on the receiver's executor, which the stream keeps alive, or on the calling thread.\n"  //
    "Calls `usearch_dense_search_stream_init`.\n"                                            //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  query (buffer): One vector of `dimensions` scalars.\n"                                //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"         //
    "  dtype (Scalar, optional): Names the query scalar type when the buffer cannot.\n"      //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> from array import array\n"                                                        //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=2)\n"                              //
    "  >>> index.reserve(1)\n"                                                               //
    "  >>> index.insert([5], array('d', [0.1, 0.2]))\n"                                      //
    "  >>> stream, arena = index.search_stream(array('d', [0.1, 0.2])), usearch.Arena()\n"   //
    "  >>> [int(match.key) for match in stream.next(4, arena)[0]]\n"                         //
    "  [5]\n"                                                                                //
    "  >>> len(stream.next(4, arena)[0])\n"                                                  //
    "  0";

static PyObject* DenseIndex_search_stream(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                          PyObject* args_names_tuple) {
    PyObject *query_object, *expansion_object, *dtype_object;
    usearch_py_argument_t const arguments[] = {
        {"query", &query_object},
        {"expansion_factor", &expansion_object},
        {"dtype", &dtype_object},
    };
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t expansion_factor = 0;
    usearch_scalar_t dtype = usearch_scalar_unknown_k;
    if (usearch_py_arguments_parse("search_stream", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)) ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0)
        return NULL;

    usearch_matrix_view_t query;
    usearch_py_buffer_backing_t backing;
    if (usearch_py_matrix_from_object(query_object, dtype, &query, &backing) != 0) return NULL;
    if (query.rows != 1) {
        usearch_py_buffer_release(&backing);
        usearch_py_raise(usearch_status_shape_mismatch_k, "a search stream takes exactly one query row");
        return NULL;
    }

    usearch_dense_search_stream_t stream_handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_search_stream_init_t arguments_init = {
        .stream = &stream_handle,
        .index = handle,
        .outcome = &outcome,
        .query = rows_of(&query),
        .expansion_factor = expansion_factor,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_search_stream_init(&arguments_init);
    Py_END_ALLOW_THREADS;
    usearch_py_buffer_release(&backing);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;

    DenseSearchStream* const stream = PyObject_New(DenseSearchStream, &DenseSearchStreamType);
    if (!stream) {
        usearch_dense_search_stream_free(stream_handle);
        return NULL;
    }
    stream->handle = stream_handle;
    stream->receiver = Py_NewRef(self);
    return (PyObject*)stream;
}

#pragma endregion Iteration

#pragma region Serialization

/** Borrows a whole checkpoint blob, whatever its length. */
static int checkpoint_borrow(PyObject* object, usearch_py_buffer_backing_t* backing) {
    backing->synthesized_owner = NULL;
    if (PyObject_GetBuffer(object, &backing->buffer, PyBUF_C_CONTIGUOUS) != 0)
        return usearch_py_raise_from_cause(PyExc_ValueError, "the checkpoint buffer must be contiguous");
    return 0;
}

static char const doc_DenseIndex_save[] =                                                   //
    "save(path, /, *, progress=None, progress_steps=0) -> None\n"                           //
    "\n"                                                                                    //
    "Writes the index under `path`, in the layout its extension picks: one `.safegraphs`\n" //
    "monolith, or a `.safegraphs.json` manifest beside one part per shard. Demands a\n"     //
    "mutation-free window; concurrent readers keep searching throughout.\n"                 //
    "Calls `usearch_dense_save`.\n"                                                         //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  path (str | bytes | os.PathLike): Where the checkpoint goes.\n"                      //
    "  progress (callable, optional): Called with (completed, total) shards; returning\n"   //
    "    False stops the write and raises `CancelledError`, leaving a checkpoint no\n"      //
    "    reader accepts and whatever parts it already wrote for the caller to remove.\n"    //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

/** A @c usearch_dense_save call and the encoded path it borrows. */
typedef struct save_call_t {
    usearch_py_call_t call;
    usearch_dense_save_t arguments;
    usearch_py_callback_t progress;

    /** The encoded path the arguments point into, held. */
    PyObject* decoded;
} save_call_t;

static void save_submit(void* call) { usearch_dense_save(&((save_call_t*)call)->arguments); }

static void save_release(void* erased) {
    save_call_t* const call = (save_call_t*)erased;
    Py_CLEAR(call->decoded);
    Py_CLEAR(call->progress.callable);
    usearch_py_call_release(&call->call);
}

static int save_traverse(void* call, visitproc visit, void* arg) {
    Py_VISIT(((save_call_t*)call)->progress.callable);
    return 0;
}

static usearch_py_verb_t const save_verb = {sizeof(save_call_t), save_submit, usearch_py_call_answer_none, save_release,
                                            save_traverse};

static PyObject* DenseIndex_save(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                 PyObject* args_names_tuple) {
    PyObject *path_object, *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {{"path", &path_object},
                                               {"progress", &progress_object},
                                               {"progress_steps", &progress_steps_object}};
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t progress_steps = 0;
    if (usearch_py_arguments_parse("save", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)) ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0)
        return NULL;

    save_call_t stack;
    save_call_t* const call = usearch_py_call_open(&receiver, &save_verb, &stack);
    if (!call) return NULL;
    usearch_bytes_view_t path;
    if (usearch_py_path_from_object(path_object, &path, &call->decoded) != 0 ||
        usearch_py_callback_from_object(progress_object, "progress", &call->call, &call->progress) != 0)
        return usearch_py_call_discard(&receiver, &save_verb, call);
    call->arguments = (usearch_dense_save_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .path = path,
        .progress = usearch_py_progress_registered(&call->progress, progress_steps),
    };
    return usearch_py_call_run(&receiver, &save_verb, call);
}

static char const doc_DenseIndex_save_buffer[] =                                             //
    "save_buffer(arena, /, *, progress=None, progress_steps=0) -> Monolith\n"                //
    "\n"                                                                                     //
    "Serializes the index as one monolith of every shard, the very bytes `save` writes to\n" //
    "a `.safegraphs` path.\n"                                                                //
    "Calls `usearch_dense_save_buffer`.\n"                                                   //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  arena (Arena): Where the monolith lands, rewound by the call.\n"                      //
    "  progress (callable, optional): Called with (completed, total) bytes; returning\n"     //
    "    False stops the write and raises `CancelledError`, None carries on.\n"              //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"  //
    "\n"                                                                                     //
    "Returns:\n"                                                                             //
    "  Monolith: The monolith's bytes, in place in the arena.\n"                             //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=4)\n"                              //
    "  >>> index.reserve(1)\n"                                                               //
    "  >>> checkpoint = index.save_buffer(usearch.Arena())\n"                                //
    "  >>> usearch.DenseIndex.metadata_read_buffer(checkpoint).dimensions\n"                 //
    "  4";

/** A @c usearch_dense_save_buffer call and the monolith it lands in the claimed arena. */
typedef struct save_buffer_call_t {
    usearch_py_call_t call;
    usearch_dense_save_buffer_t arguments;
    usearch_py_callback_t progress;
    usearch_bytes_view_t monolith;
} save_buffer_call_t;

static void save_buffer_submit(void* call) { usearch_dense_save_buffer(&((save_buffer_call_t*)call)->arguments); }

static PyObject* save_buffer_answer(void* erased) {
    save_buffer_call_t* const call = (save_buffer_call_t*)erased;
    Py_ssize_t const shape[1] = {(Py_ssize_t)call->monolith.length};
    return usearch_py_arena_buffer(&MonolithType, call->call.arena, call->monolith.data, "B", 1, shape, 1);
}

static void save_buffer_release(void* erased) {
    save_buffer_call_t* const call = (save_buffer_call_t*)erased;
    Py_CLEAR(call->progress.callable);
    usearch_py_call_release(&call->call);
}

static int save_buffer_traverse(void* call, visitproc visit, void* arg) {
    Py_VISIT(((save_buffer_call_t*)call)->progress.callable);
    return 0;
}

static usearch_py_verb_t const save_buffer_verb = {sizeof(save_buffer_call_t), save_buffer_submit, save_buffer_answer,
                                                   save_buffer_release, save_buffer_traverse};

static PyObject* DenseIndex_save_buffer(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                        PyObject* args_names_tuple) {
    PyObject *arena_object, *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {{"arena", &arena_object},
                                               {"progress", &progress_object},
                                               {"progress_steps", &progress_steps_object}};
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t progress_steps = 0;
    if (usearch_py_arguments_parse("save_buffer", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        !(handle = receiver_from_object(self, &receiver)) ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0)
        return NULL;

    save_buffer_call_t stack;
    save_buffer_call_t* const call = usearch_py_call_open(&receiver, &save_buffer_verb, &stack);
    if (!call) return NULL;
    if (usearch_py_callback_from_object(progress_object, "progress", &call->call, &call->progress) != 0 ||
        usearch_py_arena_claim(arena_object, &call->call.arena) != 0)
        return usearch_py_call_discard(&receiver, &save_buffer_verb, call);
    call->arguments = (usearch_dense_save_buffer_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->call.outcome,
        .completion = receiver.completion,
        .monolith = &call->monolith,
        .arena = ((Arena*)call->call.arena)->handle,
        .progress = usearch_py_progress_registered(&call->progress, progress_steps),
    };
    return usearch_py_call_run(&receiver, &save_buffer_verb, call);
}

/**
 *  @brief Reads what every load takes beside its source: the residency arm @p arm_name, the `executor`,
 *      `values_per_key`, `progress` and `progress_steps`, the source itself landing in @p source.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int load_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                      PyObject* args_names_tuple, char const* source_name, PyObject** source, char const* arm_name,
                      PyObject** arm, usearch_executor_t* executor, size_t* values_per_key, usearch_py_call_t* call,
                      usearch_py_callback_t* progress, size_t* progress_steps) {
    PyObject *executor_object, *values_per_key_object, *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {
        {source_name, source},          {arm_name, arm},
        {"executor", &executor_object}, {"values_per_key", &values_per_key_object},
        {"progress", &progress_object}, {"progress_steps", &progress_steps_object},
    };
    if (usearch_py_arguments_parse(verb, args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        executor_optional(executor_object, executor) != 0 ||
        usearch_py_size_from_object(values_per_key_object, "values_per_key", values_per_key) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", progress_steps) != 0)
        return -1;
    return usearch_py_callback_from_object(progress_object, "progress", call, progress);
}

static char const doc_DenseIndex_load_ram[] =                                                     //
    "load_ram(path, /, *, memory_domains=None, executor=None, values_per_key=0, progress=None,\n" //
    "    progress_steps=0) -> DenseIndex\n"                                                       //
    "\n"                                                                                          //
    "Copies a checkpoint into a new, mutable index in host memory, its geometry read from the\n"  //
    "checkpoint's own headers. Name the `.safegraphs.json` manifest, never one of the parts\n"    //
    "it enumerates; part `p` lands whole on shard `p` modulo the shard count.\n"                  //
    "Calls `usearch_dense_load_ram`.\n"                                                           //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                   //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place the parts on;\n"          //
    "    None derives every node.\n"                                                              //
    "  executor (Executor, optional): The workers the read fans out over, one shard per\n"        //
    "    thread; None reads on the calling thread.\n"                                             //
    "  values_per_key (int, optional): Entries sharing one key, which no checkpoint\n"            //
    "    records; 0 derives to one.\n"                                                            //
    "  progress (callable, optional): Called with (completed, total) shards; returning\n"         //
    "    False stops the read and raises `CancelledError`, creating no index.\n"                  //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject* DenseIndex_load_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    PyObject *path_object, *memory_domains_object, *decoded = NULL;
    usearch_py_call_t call = {0};
    usearch_py_callback_t progress = {0};
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    usearch_bytes_view_t path;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    PyObject* result = NULL;
    if (load_parse("load_ram", args, positional_args_count, args_names_tuple, "path", &path_object, "memory_domains",
                   &memory_domains_object, &executor, &values_per_key, &call, &progress, &progress_steps) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        goto release;

    usearch_dense_index_t handle = NULL;
    usearch_dense_load_ram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &call.outcome,
        .path = path,
        .ram = {memory_domains, memory_domains_count},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(&progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_ram(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_loaded(type, handle, &call);

release:
    PyMem_Free(memory_domains);
    Py_XDECREF(decoded);
    Py_XDECREF(progress.callable);
    return result;
}

static char const doc_DenseIndex_load_ram_buffer[] =                                        //
    "load_ram_buffer(buffer, /, *, memory_domains=None, executor=None, values_per_key=0,\n" //
    "    progress=None, progress_steps=0) -> DenseIndex\n"                                  //
    "\n"                                                                                    //
    "Copies a monolith out of a buffer into a new, mutable index in host memory, as\n"      //
    "`load_ram` does from a path; the monolith lands whole on the first memory domain.\n"   //
    "Calls `usearch_dense_load_ram_buffer`.\n"                                              //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  buffer (buffer): One serialized monolith, such as `save_buffer` returns.\n"          //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place shards on.\n"       //
    "  executor (Executor, optional): The workers the copy fans out over; None copies on\n" //
    "    the calling thread.\n"                                                             //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"        //
    "  progress (callable, optional): Called with (completed, total) bytes; returning\n"    //
    "    False stops the read and raises `CancelledError`, creating no index.\n"            //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject* DenseIndex_load_ram_buffer(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                            PyObject* args_names_tuple) {
    PyObject *buffer_object, *memory_domains_object;
    usearch_py_call_t call = {0};
    usearch_py_callback_t progress = {0};
    usearch_py_buffer_backing_t backing = {0};
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    PyObject* result = NULL;
    if (load_parse("load_ram_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                   "memory_domains", &memory_domains_object, &executor, &values_per_key, &call, &progress,
                   &progress_steps) != 0 ||
        checkpoint_borrow(buffer_object, &backing) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        goto release;

    usearch_dense_index_t handle = NULL;
    usearch_dense_load_ram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &call.outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .ram = {memory_domains, memory_domains_count},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(&progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_ram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_loaded(type, handle, &call);

release:
    PyMem_Free(memory_domains);
    usearch_py_buffer_release(&backing);
    Py_XDECREF(progress.callable);
    return result;
}

static char const doc_DenseIndex_load_vram[] =                                                  //
    "load_vram(path, /, *, devices=None, executor=None, values_per_key=0, progress=None,\n"     //
    "    progress_steps=0) -> DenseIndex\n"                                                     //
    "\n"                                                                                        //
    "Copies a checkpoint into a new, mutable index on CUDA devices, with the placement rules\n" //
    "of `load_ram`: part `p` lands whole on device `p` modulo the device count.\n"              //
    "Calls `usearch_dense_load_vram`.\n"                                                        //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                 //
    "  devices (Sequence[int], optional): CUDA ordinals to place the parts on.\n"               //
    "  executor (Executor, optional): The streams the upload runs on, one per device; None\n"   //
    "    uploads from the calling thread.\n"                                                    //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"            //
    "  progress (callable, optional): Called with (completed, total) shards; returning\n"       //
    "    False stops the read and raises `CancelledError`, creating no index.\n"                //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject* DenseIndex_load_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    PyObject *path_object, *devices_object, *decoded = NULL;
    usearch_py_call_t call = {0};
    usearch_py_callback_t progress = {0};
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    usearch_bytes_view_t path;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    PyObject* result = NULL;
    if (load_parse("load_vram", args, positional_args_count, args_names_tuple, "path", &path_object, "devices",
                   &devices_object, &executor, &values_per_key, &call, &progress, &progress_steps) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        goto release;

    usearch_dense_index_t handle = NULL;
    usearch_dense_load_vram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &call.outcome,
        .path = path,
        .vram = {devices, devices_count},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(&progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_vram(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_loaded(type, handle, &call);

release:
    PyMem_Free(devices);
    Py_XDECREF(decoded);
    Py_XDECREF(progress.callable);
    return result;
}

static char const doc_DenseIndex_load_vram_buffer[] =                                         //
    "load_vram_buffer(buffer, /, *, devices=None, executor=None, values_per_key=0,\n"         //
    "    progress=None, progress_steps=0) -> DenseIndex\n"                                    //
    "\n"                                                                                      //
    "Copies a monolith out of a host or device buffer into a new, mutable index on CUDA\n"    //
    "devices; the monolith lands whole on the first device.\n"                                //
    "Calls `usearch_dense_load_vram_buffer`.\n"                                               //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  buffer (buffer): One serialized monolith.\n"                                           //
    "  devices (Sequence[int], optional): CUDA ordinals to place shards on.\n"                //
    "  executor (Executor, optional): The streams the upload runs on, one per device; None\n" //
    "    uploads from the calling thread.\n"                                                  //
    "  values_per_key (int, optional): Entries sharing one key; 0 derives to one.\n"          //
    "  progress (callable, optional): Called with (completed, total) bytes; returning\n"      //
    "    False stops the read and raises `CancelledError`, creating no index.\n"              //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject* DenseIndex_load_vram_buffer(PyTypeObject* type, PyObject* const* args,
                                             Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    PyObject *buffer_object, *devices_object;
    usearch_py_call_t call = {0};
    usearch_py_callback_t progress = {0};
    usearch_py_buffer_backing_t backing = {0};
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    PyObject* result = NULL;
    if (load_parse("load_vram_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                   "devices", &devices_object, &executor, &values_per_key, &call, &progress, &progress_steps) != 0 ||
        checkpoint_borrow(buffer_object, &backing) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        goto release;

    usearch_dense_index_t handle = NULL;
    usearch_dense_load_vram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &call.outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .vram = {devices, devices_count},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(&progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_vram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_loaded(type, handle, &call);

release:
    PyMem_Free(devices);
    usearch_py_buffer_release(&backing);
    Py_XDECREF(progress.callable);
    return result;
}

static char const doc_DenseIndex_view_ram[] =                                                //
    "view_ram(path, /) -> DenseIndex\n"                                                      //
    "\n"                                                                                     //
    "Serves a checkpoint in place through a memory mapping, as a read-only index: opening\n" //
    "costs the headers alone, and a traversal pays only for the pages it touches. Every\n"   //
    "mutating verb raises `ReadOnlyError`, and the files must stay unmodified until the\n"   //
    "index is gone.\n"                                                                       //
    "Calls `usearch_dense_view_ram`.\n"                                                      //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
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

static char const doc_DenseIndex_view_nvme[] =                                               //
    "view_nvme(path, /, *, paths=None, executor=None) -> DenseIndex\n"                       //
    "\n"                                                                                     //
    "Serves a checkpoint in place from NVMe drives as a read-only index. External-memory\n"  //
    "graphs are never modified in place, so bigger ones come from merging shards built in\n" //
    "RAM.\n"                                                                                 //
    "Calls `usearch_dense_view_nvme`.\n"                                                     //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"              //
    "  paths (Sequence[str | bytes | PathLike], optional): Devices or directories to\n"      //
    "    serve the parts from; None serves each where its manifest points. A path may\n"     //
    "    name a kernel device, a transport identifier, or a FUSE mount.\n"                   //
    "  executor (Executor, optional): The NVMe executor whose rings read the headers;\n"     //
    "    None reads them on the calling thread.";

static PyObject* DenseIndex_view_nvme(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    PyObject *path_object, *paths_object, *executor_object, *decoded = NULL;
    usearch_py_argument_t const arguments[] = {
        {"path", &path_object},
        {"paths", &paths_object},
        {"executor", &executor_object},
    };
    usearch_bytes_view_t path;
    usearch_py_paths_t paths = {0};
    usearch_executor_t executor = NULL;
    PyObject* result = NULL;
    if (usearch_py_arguments_parse("view_nvme", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        executor_optional(executor_object, &executor) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0 ||
        usearch_py_paths_from_object(paths_object, &paths) != 0)
        goto release;

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_view_nvme_t view_arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .nvme = {paths.views, paths.count},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_view_nvme(&view_arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    usearch_py_paths_release(&paths);
    Py_XDECREF(decoded);
    return result;
}

/** Reads the source and the optional `executor` every metadata read takes; returns 0 or -1 with an exception set. */
static int metadata_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                          PyObject* args_names_tuple, char const* source_name, PyObject** source,
                          usearch_executor_t* executor) {
    PyObject* executor_object;
    usearch_py_argument_t const arguments[] = {{source_name, source}, {"executor", &executor_object}};
    if (usearch_py_arguments_parse(verb, args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0)
        return -1;
    return executor_optional(executor_object, executor);
}

static char const doc_DenseIndex_metadata_read[] =                                             //
    "metadata_read(path, /, *, executor=None) -> DenseMetadata\n"                              //
    "\n"                                                                                       //
    "Reads the geometry and metric a checkpoint froze, without loading it. The residency is\n" //
    "the constructor's to name, and `values_per_key` bounds a live handle rather than the\n"   //
    "file, so neither is among the fields.\n"                                                  //
    "Calls `usearch_dense_metadata_read`.\n"                                                   //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  path (str | bytes | os.PathLike): The checkpoint to inspect.\n"                         //
    "  executor (Executor, optional): The workers the read runs on; None reads on the\n"       //
    "    calling thread.";

static PyObject* DenseIndex_metadata_read(PyObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                          PyObject* args_names_tuple) {
    PyObject *path_object, *decoded = NULL;
    usearch_executor_t executor = NULL;
    usearch_bytes_view_t path;
    if (metadata_parse("metadata_read", args, positional_args_count, args_names_tuple, "path", &path_object,
                       &executor) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0)
        return NULL;
    usearch_dense_metadata_t metadata = {0};
    usearch_outcome_t outcome = {0};
    usearch_dense_metadata_read_t arguments = {
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .metadata = &metadata,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_metadata_read(&arguments);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return metadata_to_object(&metadata);
}

static char const doc_DenseIndex_metadata_read_buffer[] =                                //
    "metadata_read_buffer(buffer, /, *, executor=None) -> DenseMetadata\n"               //
    "\n"                                                                                 //
    "Reads a serialized monolith's geometry and metric, as `metadata_read`\n"            //
    "does from a path.\n"                                                                //
    "Calls `usearch_dense_metadata_read_buffer`.\n"                                      //
    "\n"                                                                                 //
    "Args:\n"                                                                            //
    "  buffer (buffer): One serialized monolith, such as `save_buffer` returns.\n"       //
    "  executor (Executor, optional): The workers the read runs on; None reads on the\n" //
    "    calling thread.";

static PyObject* DenseIndex_metadata_read_buffer(PyObject* type, PyObject* const* args,
                                                 Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    PyObject* buffer_object;
    usearch_executor_t executor = NULL;
    usearch_py_buffer_backing_t backing;
    if (metadata_parse("metadata_read_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                       &executor) != 0 ||
        checkpoint_borrow(buffer_object, &backing) != 0)
        return NULL;
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
    return metadata_to_object(&metadata);
}

#pragma endregion Serialization

#pragma region Registration

static PyMethodDef DenseIndex_methods[] = {
    {"init_ram", (PyCFunction)DenseIndex_init_ram, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_init_ram},
    {"init_vram", (PyCFunction)DenseIndex_init_vram, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_init_vram},
    {"load_ram", (PyCFunction)DenseIndex_load_ram, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_load_ram},
    {"load_ram_buffer", (PyCFunction)DenseIndex_load_ram_buffer, usearch_py_method_flags_k | METH_CLASS,
     doc_DenseIndex_load_ram_buffer},
    {"load_vram", (PyCFunction)DenseIndex_load_vram, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_load_vram},
    {"load_vram_buffer", (PyCFunction)DenseIndex_load_vram_buffer, usearch_py_method_flags_k | METH_CLASS,
     doc_DenseIndex_load_vram_buffer},
    {"view_ram", (PyCFunction)DenseIndex_view_ram, METH_O | METH_CLASS, doc_DenseIndex_view_ram},
    {"view_nvme", (PyCFunction)DenseIndex_view_nvme, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_view_nvme},
    {"metadata_read", (PyCFunction)DenseIndex_metadata_read, usearch_py_method_flags_k | METH_STATIC,
     doc_DenseIndex_metadata_read},
    {"metadata_read_buffer", (PyCFunction)DenseIndex_metadata_read_buffer, usearch_py_method_flags_k | METH_STATIC,
     doc_DenseIndex_metadata_read_buffer},
    {"on", (PyCFunction)DenseIndex_on, METH_O, doc_DenseIndex_on},
    {"deferred", (PyCFunction)DenseIndex_deferred, METH_O, doc_DenseIndex_deferred},
    {"cursor", (PyCFunction)DenseIndex_cursor, usearch_py_method_flags_k, doc_DenseIndex_cursor},
    {"search_stream", (PyCFunction)DenseIndex_search_stream, usearch_py_method_flags_k, doc_DenseIndex_search_stream},
    {"describe", (PyCFunction)DenseIndex_describe, METH_NOARGS, doc_DenseIndex_describe},
    {"reserve", (PyCFunction)DenseIndex_reserve, METH_O, doc_DenseIndex_reserve},
    {"clear", (PyCFunction)DenseIndex_clear, METH_NOARGS, doc_DenseIndex_clear},
    {"insert", (PyCFunction)DenseIndex_insert, usearch_py_method_flags_k, doc_DenseIndex_insert},
    {"refine", (PyCFunction)DenseIndex_refine, usearch_py_method_flags_k, doc_DenseIndex_refine},
    {"health", (PyCFunction)DenseIndex_health, METH_NOARGS, doc_DenseIndex_health},
    {"search", (PyCFunction)DenseIndex_search, usearch_py_method_flags_k, doc_DenseIndex_search},
    {"count", (PyCFunction)DenseIndex_count, METH_O, doc_DenseIndex_count},
    {"get", (PyCFunction)DenseIndex_get, usearch_py_method_flags_k, doc_DenseIndex_get},
    {"multiply", (PyCFunction)DenseIndex_multiply, usearch_py_method_flags_k, doc_DenseIndex_multiply},
    {"attend", (PyCFunction)DenseIndex_attend, usearch_py_method_flags_k, doc_DenseIndex_attend},
    {"maxsim", (PyCFunction)DenseIndex_maxsim, usearch_py_method_flags_k, doc_DenseIndex_maxsim},
    {"absorb", (PyCFunction)DenseIndex_absorb, usearch_py_method_flags_k, doc_DenseIndex_absorb},
    {"save", (PyCFunction)DenseIndex_save, usearch_py_method_flags_k, doc_DenseIndex_save},
    {"save_buffer", (PyCFunction)DenseIndex_save_buffer, usearch_py_method_flags_k, doc_DenseIndex_save_buffer},
    {NULL, NULL, 0, NULL},
};

static char const doc_receiver_class_getitem[] =
    "Names the index the receiver addresses, as `On[DenseIndex]` or `DeferredOn[DenseIndex]`.";

PyMethodDef usearch_py_on_methods[] = {
    {"cursor", (PyCFunction)DenseIndex_cursor, usearch_py_method_flags_k, doc_DenseIndex_cursor},
    {"search_stream", (PyCFunction)DenseIndex_search_stream, usearch_py_method_flags_k, doc_DenseIndex_search_stream},
    {"describe", (PyCFunction)DenseIndex_describe, METH_NOARGS, doc_DenseIndex_describe},
    {"reserve", (PyCFunction)DenseIndex_reserve, METH_O, doc_DenseIndex_reserve},
    {"clear", (PyCFunction)DenseIndex_clear, METH_NOARGS, doc_DenseIndex_clear},
    {"insert", (PyCFunction)DenseIndex_insert, usearch_py_method_flags_k, doc_DenseIndex_insert},
    {"refine", (PyCFunction)DenseIndex_refine, usearch_py_method_flags_k, doc_DenseIndex_refine},
    {"health", (PyCFunction)DenseIndex_health, METH_NOARGS, doc_DenseIndex_health},
    {"search", (PyCFunction)DenseIndex_search, usearch_py_method_flags_k, doc_DenseIndex_search},
    {"count", (PyCFunction)DenseIndex_count, METH_O, doc_DenseIndex_count},
    {"get", (PyCFunction)DenseIndex_get, usearch_py_method_flags_k, doc_DenseIndex_get},
    {"multiply", (PyCFunction)DenseIndex_multiply, usearch_py_method_flags_k, doc_DenseIndex_multiply},
    {"attend", (PyCFunction)DenseIndex_attend, usearch_py_method_flags_k, doc_DenseIndex_attend},
    {"maxsim", (PyCFunction)DenseIndex_maxsim, usearch_py_method_flags_k, doc_DenseIndex_maxsim},
    {"absorb", (PyCFunction)DenseIndex_absorb, usearch_py_method_flags_k, doc_DenseIndex_absorb},
    {"save", (PyCFunction)DenseIndex_save, usearch_py_method_flags_k, doc_DenseIndex_save},
    {"save_buffer", (PyCFunction)DenseIndex_save_buffer, usearch_py_method_flags_k, doc_DenseIndex_save_buffer},
    {"__class_getitem__", Py_GenericAlias, METH_O | METH_CLASS, doc_receiver_class_getitem},
    {NULL, NULL, 0, NULL},
};

PyMethodDef usearch_py_deferred_on_methods[] = {
    {"describe", (PyCFunction)DenseIndex_describe, METH_NOARGS, doc_DenseIndex_describe},
    {"reserve", (PyCFunction)DenseIndex_reserve, METH_O, doc_DenseIndex_reserve},
    {"clear", (PyCFunction)DenseIndex_clear, METH_NOARGS, doc_DenseIndex_clear},
    {"insert", (PyCFunction)DenseIndex_insert, usearch_py_method_flags_k, doc_DenseIndex_insert},
    {"refine", (PyCFunction)DenseIndex_refine, usearch_py_method_flags_k, doc_DenseIndex_refine},
    {"health", (PyCFunction)DenseIndex_health, METH_NOARGS, doc_DenseIndex_health},
    {"search", (PyCFunction)DenseIndex_search, usearch_py_method_flags_k, doc_DenseIndex_search},
    {"count", (PyCFunction)DenseIndex_count, METH_O, doc_DenseIndex_count},
    {"get", (PyCFunction)DenseIndex_get, usearch_py_method_flags_k, doc_DenseIndex_get},
    {"multiply", (PyCFunction)DenseIndex_multiply, usearch_py_method_flags_k, doc_DenseIndex_multiply},
    {"attend", (PyCFunction)DenseIndex_attend, usearch_py_method_flags_k, doc_DenseIndex_attend},
    {"maxsim", (PyCFunction)DenseIndex_maxsim, usearch_py_method_flags_k, doc_DenseIndex_maxsim},
    {"absorb", (PyCFunction)DenseIndex_absorb, usearch_py_method_flags_k, doc_DenseIndex_absorb},
    {"save", (PyCFunction)DenseIndex_save, usearch_py_method_flags_k, doc_DenseIndex_save},
    {"save_buffer", (PyCFunction)DenseIndex_save_buffer, usearch_py_method_flags_k, doc_DenseIndex_save_buffer},
    {"__class_getitem__", Py_GenericAlias, METH_O | METH_CLASS, doc_receiver_class_getitem},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef DenseIndex_getset[] = {
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

PyTypeObject DenseIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseIndex",
    .tp_doc = doc_DenseIndex,
    .tp_basicsize = sizeof(DenseIndex),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = DenseIndex_tp_new,
    .tp_dealloc = (destructor)DenseIndex_dealloc,
    .tp_repr = (reprfunc)DenseIndex_repr,
    .tp_methods = DenseIndex_methods,
    .tp_getset = DenseIndex_getset,
    .tp_as_sequence = &DenseIndex_as_sequence,
};

#pragma endregion Registration
