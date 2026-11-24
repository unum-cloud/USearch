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
 *  record that the receiver keeps on the stack or inside the @c Deferred, and opens its result as
 *  the slot the library writes into, so the same body serves every completion. Every call detaches
 *  the thread state; the predicate, progress and value-lookup callbacks re-enter Python from the
 *  library's own threads, so their trampolines take the GIL back for each report.
 */
#include "lib.h"

#pragma region Handles

/** An owned index handle. */
typedef struct DenseIndex {
    PyObject ob_base;
    usearch_dense_index_t handle;
} DenseIndex;

/** A sequential scan over the entries an index holds, in batches. */
typedef struct DenseCursor {
    PyObject ob_base;

    /** The engine cursor, reading each shard's watermark as it enters that shard; NULL once closed. */
    _Atomic(usearch_dense_cursor_t) handle;

    /** The index or @c On that opened the cursor, held, which every step runs through. */
    PyObject* receiver;
} DenseCursor;

/** The entries one cursor step wrote into an arena: keys always, their links when asked. */
typedef struct DenseBatch {
    PyObject ob_base;

    /** The (count, 16) @c ArenaBuffer of identities. */
    PyObject* keys;

    /** The (offsets, keys) pair of each entry's linked identities, or None for a keys-only step. */
    PyObject* links;
} DenseBatch;

/** A resumable best-first stream of matches for one query. */
typedef struct DenseSearchStream {
    PyObject ob_base;

    /** The engine stream; NULL once closed. */
    _Atomic(usearch_dense_search_stream_t) handle;

    /** The index or @c On that opened the stream, held, which every pull runs through. */
    PyObject* receiver;
} DenseSearchStream;

/** A live index's frozen metadata and current counters, the slot @c describe writes them into. */
typedef struct DenseIndexDescription {
    PyObject ob_base;
    usearch_dense_index_description_t description;
} DenseIndexDescription;

/** The draws one @c sample call wrote into an arena, the slot of its outputs. */
typedef struct Sample {
    PyObject ob_base;
    PyObject* arena;
    usearch_keys_view_t keys;
    size_t const* entry_ordinals;
    usearch_matches_view_t parents;
    usearch_rows_view_t coordinates;
    size_t written;
} Sample;

/** A refine or a reshard in flight, continued by every call naming it until its finish verb. */
typedef struct DensePlan {
    PyObject ob_base;

    /** The engine plan, NULL until a call opens it and once its finish verb closed it. */
    union {
        usearch_dense_refine_plan_t refine;
        usearch_dense_reshard_plan_t reshard;
    } handle;

    /** Where the last call naming the plan stopped. */
    usearch_phase_progress_t progress;
} DensePlan;

/** The holds every dense call keeps in the same slots. */
enum {
    progress_slot_k = 0,
    stats_slot_k = 1,
    callback_slot_k = 2,
    sources_slot_k = 3,
    targets_slot_k = 4,
};

/**
 *  @brief Reads what @p self names and its live handle, refusing a receiver over another modality.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int receiver_from_object(PyObject* self, usearch_py_receiver_t* receiver, usearch_dense_index_t* handle) {
    if (usearch_py_receiver_from_object(self, receiver) != 0) return -1;
    if (!Py_IS_TYPE(receiver->index, &DenseIndexType)) {
        PyErr_Format(PyExc_TypeError, "expected a receiver over a DenseIndex, got one over %S",
                     Py_TYPE(receiver->index));
        return -1;
    }
    *handle = ((DenseIndex*)receiver->index)->handle;
    return 0;
}

/** The rows a viewed matrix holds, as a verb's batch input reads them. */
static usearch_rows_view_t rows_of(usearch_matrix_view_t const* matrix) {
    usearch_rows_view_t const rows = {matrix->data, matrix->stride, matrix->columns, matrix->scalar};
    return rows;
}

/** Holds the progress handler and the telemetry a walking verb overwrites in @p call; returns 0 or
 *  -1 with an exception set. */
static int walk_parse(usearch_py_call_t* call, PyObject* progress_object, PyObject* stats_object) {
    if (usearch_py_callable_from_object(progress_object, "progress", &call->holds[progress_slot_k]) != 0) return -1;
    if (!stats_object || stats_object == Py_None) return 0;
    if (!Py_IS_TYPE(stats_object, &DenseTraversalStatsType)) {
        PyErr_Format(PyExc_TypeError, "the 'stats' option must be a DenseTraversalStats or None, got %S",
                     Py_TYPE(stats_object));
        return -1;
    }
    call->holds[stats_slot_k] = Py_NewRef(stats_object);
    return 0;
}

/** The telemetry the library overwrites for @p call, NULL when the caller passed none. */
static usearch_traversal_stats_t* walk_stats(usearch_py_call_t const* call) {
    PyObject* const stats = call->holds[stats_slot_k];
    return stats ? &((DenseTraversalStats*)stats)->stats : NULL;
}

/** Holds the @p plan of @p type a call continues or closes in @p call; NULL or None names none.
 *  Returns 0 or -1 with an exception set. */
static int plan_parse(usearch_py_call_t* call, PyObject* plan, PyTypeObject* type) {
    if (!plan || plan == Py_None) return 0;
    if (!Py_IS_TYPE(plan, type)) {
        PyErr_Format(PyExc_TypeError, "the 'plan' option must be a %s or None, got %S", type->tp_name, Py_TYPE(plan));
        return -1;
    }
    call->holds[callback_slot_k] = Py_NewRef(plan);
    return 0;
}

#pragma endregion Handles

#pragma region Construction

static char const doc_DenseIndex[] =                                                               //
    "DenseIndex\n"                                                                                 //
    "\n"                                                                                           //
    "Vector search over equidimensional dense embeddings.\n"                                       //
    "\n"                                                                                           //
    "Every verb is a batch, and one vector is a one-row matrix. The stored scalar type is\n"       //
    "the `quantization`; query rows may arrive in any other type and convert on the way in.\n"     //
    "\n"                                                                                           //
    "Start an empty one in the residency its entries live in - `init_ram` or `init_vram` - copy\n" //
    "a checkpoint in with `load_ram` or `load_vram`, or serve one in place with `view_ram`\n"      //
    "or `view_nvme`, which are read-only.\n"                                                       //
    "\n"                                                                                           //
    "The receiver chooses where a verb runs, and the verb stays the same: the index runs it\n"     //
    "on the calling thread, `on(executor)` blocks on that executor, and `deferred(executor)`\n"    //
    "submits it there and answers a `Deferred`. See `usearch.Executor`.\n"                         //
    "\n"                                                                                           //
    "Example:\n"                                                                                   //
    "  >>> index = usearch.DenseIndex.init_ram(256, usearch.DenseMetric.ANGULAR)\n"                //
    "  >>> index.cpu_class is not None\n"                                                          //
    "  True";

/** The geometry both creation verbs freeze, read before the factory's residency arm. */
typedef struct dense_geometry_t {
    usearch_dense_metric_t metric;
    usearch_scalar_t quantization;
    size_t dimensions;
    size_t connectivity;
    size_t values_per_key;
} dense_geometry_t;

/** The keyword a verb names its class ceiling by: @c cpu_class_ceiling for host placements,
 *  @c gpu_class_ceiling for devices. */
static char const* ceiling_name(usearch_py_enum_t const* ceiling_classes) {
    return ceiling_classes == &usearch_py_gpu_class ? "gpu_class_ceiling" : "cpu_class_ceiling";
}

/** Reads the class ceiling a placement names, a member of @p ceiling_classes, leaving zero for
 *  none; returns 0, or -1 with an exception set. */
static int ceiling_parse(PyObject* object, usearch_py_enum_t const* ceiling_classes, int* ceiling) {
    *ceiling = 0;
    if (!object || object == Py_None) return 0;
    return usearch_py_enum_from_object(object, ceiling_name(ceiling_classes), ceiling_classes, ceiling);
}

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
                          usearch_py_enum_t const* ceiling_classes, int* ceiling, dense_geometry_t* geometry) {
    PyObject *dimensions_object, *metric_object, *quantization_object, *connectivity_object, *values_per_key_object;
    PyObject* ceiling_object;
    usearch_py_argument_t const arguments[] = {
        {"dimensions", &dimensions_object},
        {"metric", &metric_object},
        {"quantization", &quantization_object},
        {"connectivity", &connectivity_object},
        {"values_per_key", &values_per_key_object},
        {arm_name, arm},
        {ceiling_name(ceiling_classes), &ceiling_object},
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
        usearch_py_size_from_object(values_per_key_object, "values_per_key", &geometry->values_per_key) != 0 ||
        ceiling_parse(ceiling_object, ceiling_classes, ceiling) != 0)
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

static PyObject* DenseIndex_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_TypeError,
                    "DenseIndex is built through init_ram(), init_vram(), load_ram*(), load_vram*(), load_nvme(), " //
                    "view_ram() or view_nvme()");
    return NULL;
}

static char const doc_DenseIndex_init_ram[] =                                                            //
    "``init_ram(dimensions, metric=DenseMetric.ANGULAR, quantization=Scalar.BF16, *, connectivity=0,\n"  //
    "values_per_key=0, memory_domains=None, cpu_class_ceiling=None) -> DenseIndex``\n"                   //
    "\n"                                                                                                 //
    "Starts an empty, mutable graph in host memory, one shard per named NUMA node.\n"                    //
    "Calls `usearch_dense_init_ram`.\n"                                                                  //
    "\n"                                                                                                 //
    "Args:\n"                                                                                            //
    "  dimensions (int): Scalars per vector.\n"                                                          //
    "  metric (DenseMetric, optional): How two vectors are scored. Defaults to `DenseMetric.ANGULAR`.\n" //
    "  quantization (Scalar, optional): The stored scalar type. Defaults to `Scalar.BF16`.\n"            //
    "  connectivity (int, optional): Connections per node; 0 derives it from the geometry.\n"            //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                       //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to shard across, so a\n"                  //
    "    dual-socket box naming two gets two graphs; None derives every node.\n"                         //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n"        //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.\n"              //
    "\n"                                                                                                 //
    "Example:\n"                                                                                         //
    "  >>> index = usearch.DenseIndex.init_ram(256, usearch.DenseMetric.ANGULAR)\n"                      //
    "  >>> index.dimensions\n"                                                                           //
    "  256";

static PyObject* DenseIndex_init_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    dense_geometry_t geometry;
    PyObject* memory_domains_object;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    int cpu_class_ceiling = 0;
    if (geometry_parse("init_ram", args, positional_args_count, args_names_tuple, "memory_domains",
                       &memory_domains_object, &usearch_py_cpu_class, &cpu_class_ceiling, &geometry) != 0 ||
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
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_init_ram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);
    return index_adopt(type, handle, &outcome);
}

static char const doc_DenseIndex_init_vram[] =                                                           //
    "``init_vram(dimensions, metric=DenseMetric.ANGULAR, quantization=Scalar.BF16, *, connectivity=0,\n" //
    "values_per_key=0, devices=None, gpu_class_ceiling=None) -> DenseIndex``\n"                          //
    "\n"                                                                                                 //
    "Starts an empty, mutable graph on CUDA devices, one shard per named ordinal.\n"                     //
    "Calls `usearch_dense_init_vram`.\n"                                                                 //
    "\n"                                                                                                 //
    "Entries are sharded and never replicated, so eight devices host eight slices of one\n"              //
    "graph rather than eight copies. An index outgrowing them fails rather than paging to\n"             //
    "the host, and the answer is to name more devices.\n"                                                //
    "\n"                                                                                                 //
    "Args:\n"                                                                                            //
    "  dimensions (int): Scalars per vector.\n"                                                          //
    "  metric (DenseMetric, optional): How two vectors are scored. Defaults to `DenseMetric.ANGULAR`.\n" //
    "  quantization (Scalar, optional): The stored scalar type. Defaults to `Scalar.BF16`.\n"            //
    "  connectivity (int, optional): Connections per node; 0 derives it from the geometry.\n"            //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                       //
    "  devices (Sequence[int], optional): CUDA ordinals to shard across; None derives one\n"             //
    "    free device.\n"                                                                                 //
    "  gpu_class_ceiling (GPUClass, optional): The newest class dispatch may pick, capped below\n"       //
    "    the best class every device loads, else unsupported; None means no ceiling.\n"                  //
    "\n"                                                                                                 //
    "Example:\n"                                                                                         //
    "  >>> index = usearch.DenseIndex.init_vram(256, devices=[0, 1])   # doctest: +SKIP";

static PyObject* DenseIndex_init_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    dense_geometry_t geometry;
    PyObject* devices_object;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    int gpu_class_ceiling = 0;
    if (geometry_parse("init_vram", args, positional_args_count, args_names_tuple, "devices", &devices_object,
                       &usearch_py_gpu_class, &gpu_class_ceiling, &geometry) != 0 ||
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
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
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
    "A receiver whose verbs block on `executor`, refused while another call holds it.\n"       //
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
    "  missing_hints (int): Hint keys no shard held an entry under.\n"          //
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
    char const* const format =
        "DenseTraversalStats(visited_members=%zu, computed_distances=%zu, restarts=%zu, missing_hints=%zu)";
    return PyUnicode_FromFormat(format, self->stats.visited_members, self->stats.computed_distances,
                                self->stats.restarts, self->stats.missing_hints);
}

/** Adds another reading's counters into this one. */
static PyObject* DenseTraversalStats_inplace_add(PyObject* self, PyObject* other) {
    if (!Py_IS_TYPE(other, &DenseTraversalStatsType)) Py_RETURN_NOTIMPLEMENTED;
    usearch_traversal_stats_t const addend = ((DenseTraversalStats*)other)->stats;
    usearch_traversal_stats_t* const total = &((DenseTraversalStats*)self)->stats;
    total->visited_members += addend.visited_members;
    total->computed_distances += addend.computed_distances;
    total->restarts += addend.restarts;
    total->missing_hints += addend.missing_hints;
    return Py_NewRef(self);
}

static PyGetSetDef DenseTraversalStats_getset[] = {
    {"visited_members", usearch_py_size_getter, NULL, "Entries whose neighborhoods the walk expanded.",
     (void*)offsetof(DenseTraversalStats, stats.visited_members)},
    {"computed_distances", usearch_py_size_getter, NULL, "Distance evaluations the walk paid for.",
     (void*)offsetof(DenseTraversalStats, stats.computed_distances)},
    {"restarts", usearch_py_size_getter, NULL, "Times the walk fell back to a fresh entry point.",
     (void*)offsetof(DenseTraversalStats, stats.restarts)},
    {"missing_hints", usearch_py_size_getter, NULL, "Hint keys no shard held an entry under.",
     (void*)offsetof(DenseTraversalStats, stats.missing_hints)},
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
    {"quantization", "The `Scalar` its vectors are stored and quantized into."},
    {"connectivity", "Connections per node the graph was built with."},
    {NULL, NULL},
};

PyStructSequence_Desc usearch_py_dense_metadata_desc = {
    "usearch.DenseMetadata", "The geometry and metric a checkpoint froze, as `usearch_dense_metadata_t` holds them.",
    DenseMetadata_fields, 4};

PyTypeObject DenseMetadataType;

/** A new @c DenseMetadata of @p metadata, or NULL with an exception set. */
static PyObject* metadata_to_object(usearch_dense_metadata_t const* metadata) {
    PyObject* fields[4] = {NULL, NULL, NULL, NULL};
    (void)((fields[0] = PyLong_FromSize_t(metadata->dimensions)) &&
           (fields[1] = usearch_py_enum_to_object(&usearch_py_dense_metric, (int)metadata->metric)) &&
           (fields[2] = usearch_py_enum_to_object(&usearch_py_scalar, (int)metadata->quantization)) &&
           (fields[3] = PyLong_FromSize_t(metadata->connectivity)));
    return usearch_py_struct_sequence_steal(&DenseMetadataType, fields, 4);
}

static char const doc_DenseIndexDescription[] =                                             //
    "DenseIndexDescription\n"                                                               //
    "\n"                                                                                    //
    "A live index's frozen metadata and current counters, as `describe` read them in one\n" //
    "call.";

static PyObject* DenseIndexDescription_get_metadata(DenseIndexDescription* self, void* closure) {
    return metadata_to_object(&self->description.metadata);
}

static PyObject* DenseIndexDescription_get_cpu_class(DenseIndexDescription* self, void* closure) {
    return usearch_py_class_to_object(&usearch_py_cpu_class, (int)self->description.cpu_class);
}

static PyObject* DenseIndexDescription_get_gpu_class(DenseIndexDescription* self, void* closure) {
    return usearch_py_class_to_object(&usearch_py_gpu_class, (int)self->description.gpu_class);
}

static PyObject* DenseIndexDescription_get_load_path(DenseIndexDescription* self, void* closure) {
    return PyUnicode_FromString(usearch_load_path_name(self->description.load_path));
}

static PyGetSetDef DenseIndexDescription_getset[] = {
    {"metadata", (getter)DenseIndexDescription_get_metadata, NULL,
     "The frozen `DenseMetadata`, as a checkpoint reports it.", NULL},
    {"size", usearch_py_size_getter, NULL, "Stored entries.", (void*)offsetof(DenseIndexDescription, description.size)},
    {"capacity", usearch_py_size_getter, NULL, "Entries hosted without growing again.",
     (void*)offsetof(DenseIndexDescription, description.capacity)},
    {"memory_usage", usearch_py_size_getter, NULL, "Bytes the handle holds; traversal scratch is the executor's.",
     (void*)offsetof(DenseIndexDescription, description.memory_usage)},
    {"cpu_class", (getter)DenseIndexDescription_get_cpu_class, NULL,
     "The `CPUClass` dispatch picked on this machine, or None.", NULL},
    {"gpu_class", (getter)DenseIndexDescription_get_gpu_class, NULL,
     "The `GPUClass` dispatch picked on the index's devices, or None for a host index.", NULL},
    {"shards_count", usearch_py_size_getter, NULL, "Live shards.",
     (void*)offsetof(DenseIndexDescription, description.shards_count)},
    {"load_path", (getter)DenseIndexDescription_get_load_path, NULL,
     "How the last load from a path read the parts onto the devices: none, host-copy or cufile.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject* DenseIndexDescription_repr(DenseIndexDescription* self) {
    return PyUnicode_FromFormat("DenseIndexDescription(dimensions=%zu, size=%zu, capacity=%zu)",
                                self->description.metadata.dimensions, self->description.size,
                                self->description.capacity);
}

PyTypeObject DenseIndexDescriptionType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseIndexDescription",
    .tp_doc = doc_DenseIndexDescription,
    .tp_basicsize = sizeof(DenseIndexDescription),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_repr = (reprfunc)DenseIndexDescription_repr,
    .tp_getset = DenseIndexDescription_getset,
};

static char const doc_Sample[] =                                                             //
    "Sample\n"                                                                               //
    "\n"                                                                                     //
    "The draws one `sample` call wrote into an `Arena`, viewing it until it rewinds.\n"      //
    "\n"                                                                                     //
    "Attributes:\n"                                                                          //
    "  keys (buffer): Identities drawn, `(written, 16)`, most significant byte first.\n"     //
    "  entry_ordinals (buffer): Which of its key's entries each draw is, `(written,)`, in\n" //
    "    the order `get` writes them.\n"                                                     //
    "  parents (Matches): One column naming the shown or earlier draw each draw hangs\n"     //
    "    from, and its distance.\n"                                                          //
    "  coordinates (buffer): Float32 places of the draws, `(written, 3)`, beside the shown keys.";

static void Sample_dealloc(Sample* self) {
    Py_XDECREF(self->arena);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Sample_get_keys(Sample* self, void* closure) {
    Py_ssize_t const shape[2] = {(Py_ssize_t)self->written, (Py_ssize_t)sizeof(usearch_key_t)};
    return usearch_py_arena_buffer(&ArenaBufferType, self->arena, self->keys.data, "B", 1, shape, 2);
}

static PyObject* Sample_get_entry_ordinals(Sample* self, void* closure) {
    Py_ssize_t const shape[1] = {(Py_ssize_t)self->written};
    return usearch_py_arena_buffer(&ArenaBufferType, self->arena, self->entry_ordinals, "N", sizeof(size_t), shape, 1);
}

static PyObject* Sample_get_parents(Sample* self, void* closure) {
    return usearch_py_matches_new(self->arena, &self->parents, self->written);
}

static PyObject* Sample_get_coordinates(Sample* self, void* closure) {
    Py_ssize_t const shape[2] = {(Py_ssize_t)self->written, 3};
    return usearch_py_arena_buffer(&ArenaBufferType, self->arena, self->coordinates.data, "f", sizeof(float), shape, 2);
}

static PyGetSetDef Sample_getset[] = {
    {"keys", (getter)Sample_get_keys, NULL, "(written, 16) identities drawn, most significant byte first.", NULL},
    {"entry_ordinals", (getter)Sample_get_entry_ordinals, NULL,
     "(written,) which of its key's entries each draw is, in the order `get` writes them.", NULL},
    {"parents", (getter)Sample_get_parents, NULL,
     "`Matches` of one column: the shown or earlier draw each draw hangs from, and its distance.", NULL},
    {"coordinates", (getter)Sample_get_coordinates, NULL,
     "(written, 3) float32 places of the draws, beside the shown keys.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject SampleType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.Sample",
    .tp_doc = doc_Sample,
    .tp_basicsize = sizeof(Sample),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NULL, // Returned by `sample`, never constructed by name.
    .tp_dealloc = (destructor)Sample_dealloc,
    .tp_getset = Sample_getset,
};

#pragma endregion Reports

#pragma region Introspection

/**
 *  @brief Reads the frozen geometry and current counters of @p handle on the calling thread.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int index_describe(usearch_dense_index_t handle, usearch_dense_index_description_t* description) {
    memset(description, 0, sizeof(*description));
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

static PyObject* DenseIndex_describe(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    DenseIndexDescription* const description = usearch_py_call_result(call, &DenseIndexDescriptionType);
    if (!description) return usearch_py_call_discard(&receiver, call);
    usearch_dense_describe_t* const arguments = (usearch_dense_describe_t*)call->arguments;
    *arguments = (usearch_dense_describe_t) {.index = handle,
                                             .executor = receiver.executor,
                                             .outcome = &call->outcome,
                                             .completion = receiver.completion,
                                             .description = &description->description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_describe(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
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

static PyObject* DenseIndex_health(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    DenseReachability* const audit = usearch_py_call_result(call, &DenseReachabilityType);
    if (!audit) return usearch_py_call_discard(&receiver, call);
    usearch_dense_health_t* const arguments = (usearch_dense_health_t*)call->arguments;
    *arguments = (usearch_dense_health_t) {.index = handle,
                                           .executor = receiver.executor,
                                           .outcome = &call->outcome,
                                           .completion = receiver.completion,
                                           .reachability = &audit->reachability};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_health(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

/** Reads the @c size_t counter at the byte offset @p closure names inside a description. */
static PyObject* DenseIndex_get_counter(DenseIndex* self, void* closure) {
    usearch_dense_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return PyLong_FromSize_t(*(size_t const*)((char const*)&description + (size_t)closure));
}

static char const doc_DenseIndex_capacity[] =
    "Entries the index hosts without growing again. Calls `usearch_dense_describe`.";
static char const doc_DenseIndex_dimensions[] = "Scalars per vector. Calls `usearch_dense_describe`.";
static char const doc_DenseIndex_connectivity[] = "Connections per node in the graph. Calls `usearch_dense_describe`.";
static char const doc_DenseIndex_memory_usage[] =
    "Bytes the handle holds; traversal scratch is the executor's. Calls `usearch_dense_describe`.";
static char const doc_DenseIndex_cpu_class[] =
    "The `CPUClass` dispatch picked on this machine, or None. Calls `usearch_dense_describe`.";
static char const doc_DenseIndex_gpu_class[] =
    "The `GPUClass` dispatch picked on the index's devices, or None for a host index. Calls `usearch_dense_describe`.";
static char const doc_DenseIndex_shards_count[] = "Live shards. Calls `usearch_dense_describe`.";

static PyObject* DenseIndex_get_cpu_class(DenseIndex* self, void* closure) {
    usearch_dense_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return usearch_py_class_to_object(&usearch_py_cpu_class, (int)description.cpu_class);
}

static PyObject* DenseIndex_get_gpu_class(DenseIndex* self, void* closure) {
    usearch_dense_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return usearch_py_class_to_object(&usearch_py_gpu_class, (int)description.gpu_class);
}

static Py_ssize_t DenseIndex_length(DenseIndex* self) {
    usearch_dense_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return -1;
    return (Py_ssize_t)description.size;
}

static PyObject* DenseIndex_repr(DenseIndex* self) {
    // No exception is raised here: `tp_repr` runs inside `PyErr_Format("%R", ...)` chains, where
    // raising would destroy the caller's.
    usearch_dense_index_description_t description = {0};
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

static char const doc_DenseIndex_reserve[] =                         //
    "reserve(capacity) -> None\n"                                    //
    "\n"                                                             //
    "Grows the index to host at least `capacity` entries, a hint,\n" //
    "since insertions grow the index on demand.\n"                   //
    "Calls `usearch_dense_reserve`.\n"                               //
    "\n"                                                             //
    "Args:\n"                                                        //
    "  capacity (int): Entries the index must accept without growing again.";

static PyObject* DenseIndex_reserve(PyObject* self, PyObject* capacity_object) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t capacity = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(capacity_object, "capacity", &capacity) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    usearch_dense_reserve_t* const arguments = (usearch_dense_reserve_t*)call->arguments;
    *arguments = (usearch_dense_reserve_t) {.index = handle,
                                            .executor = receiver.executor,
                                            .outcome = &call->outcome,
                                            .completion = receiver.completion,
                                            .capacity = capacity};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_reserve(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_DenseIndex_clear[] =                               //
    "clear() -> None\n"                                                  //
    "\n"                                                                 //
    "Drops every entry, keeping the capacity and the frozen geometry.\n" //
    "Calls `usearch_dense_clear`.";

static PyObject* DenseIndex_clear(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    usearch_dense_clear_t* const arguments = (usearch_dense_clear_t*)call->arguments;
    *arguments = (usearch_dense_clear_t) {.index = handle,
                                          .executor = receiver.executor,
                                          .outcome = &call->outcome,
                                          .completion = receiver.completion};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_clear(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_DenseIndex_insert[] =                                                     //
    "``insert(keys, vectors, /, *, expansion_factor=0, dtype=None, progress=None,\n"            //
    "progress_steps=0, stats=None) -> None``\n"                                                 //
    "\n"                                                                                        //
    "Inserts a batch of vectors under a batch of keys; one vector is a one-row matrix.\n"       //
    "Keys need not be unique: inserting one twice stores two entries, and `count`\n"            //
    "reports both. A batch is all-or-nothing and becomes visible at once, so a rejected\n"      //
    "batch leaves the index untouched and a concurrent search sees all of it or none.\n"        //
    "Calls `usearch_dense_insert`.\n"                                                           //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  keys (Sequence[Key | int] | buffer): One key per row, an N by 16 buffer, or an Arrow\n"  //
    "    column of 16-byte keys or integers.\n"                                                 //
    "  vectors (buffer): An N by dimensions matrix, only the row stride varying, or an Arrow\n" //
    "    fixed-size list column; one chunk stays in place and several gather once.\n"           //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"            //
    "  dtype (Scalar, optional): Names the input scalar type when the buffer cannot; it\n"      //
    "    must be as wide as the buffer's items.\n"                                              //
    "  progress (callable, optional): Called with (completed, total) entries. It reports and\n" //
    "    cannot stop the batch, every entry being claimed before the first report; what it\n"   //
    "    raises goes to `sys.unraisablehook`. Reports are serialized.\n"                        //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"     //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"        //
    "\n"                                                                                        //
    "Example:\n"                                                                                //
    "  >>> from array import array\n"                                                           //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=4)\n"                                 //
    "  >>> index.reserve(1)\n"                                                                  //
    "  >>> index.insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))\n"                               //
    "  >>> len(index)\n"                                                                        //
    "  1";

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
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("insert", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    usearch_matrix_view_t vectors;
    if (usearch_py_matrix_from_object(vectors_object, dtype, &vectors, &call->buffers[0]) != 0 ||
        usearch_py_keys_from_object(keys_object, vectors.rows, &call->keys[0]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0)
        return usearch_py_call_discard(&receiver, call);

    usearch_dense_insert_t* const insert = (usearch_dense_insert_t*)call->arguments;
    *insert = (usearch_dense_insert_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .count = vectors.rows,
        .keys = {call->keys[0].keys, sizeof(usearch_key_t)},
        .vectors = rows_of(&vectors),
        .expansion_factor = expansion_factor,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_insert(insert);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_DenseIndex_refine[] =                                                    //
    "``refine(*, expansion_factor=0, memory_limit=0, progress=None, progress_steps=0,\n"       //
    "stats=None, plan=None) -> RefineReport``\n"                                               //
    "\n"                                                                                       //
    "Re-selects the neighbors of every stored entry against the finished graph, improving\n"   //
    "edges that were chosen when the graph around them was still incomplete.\n"                //
    "\n"                                                                                       //
    "A build step rather than a repair: a cheap insertion pass followed by one refinement\n"   //
    "reaches a better graph, in less total time, than one expensive insertion pass - so a\n"   //
    "wide `expansion_factor` belongs here rather than in `insert`. Searches may run\n"         //
    "throughout on a RAM handle but not on a VRAM one, and a caller loops until the\n"         //
    "report's `verdict` is no longer `RefineVerdict.IMPROVED`.\n"                              //
    "\n"                                                                                       //
    "Stopping is safe: re-selecting edges claims no identity, so a stopped sweep still\n"      //
    "re-links its islands, leaves the graph correct and merely less improved, and raises\n"    //
    "`CancelledError`. A handler is therefore how a sweep is put under a time budget. With\n"  //
    "a `plan`, a stop leaves it live and the next call naming it continues the round;\n"       //
    "`refine_plan_finish(plan)` closes it.\n"                                                  //
    "Calls `usearch_dense_refine`.\n"                                                          //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"           //
    "  memory_limit (int, optional): Bytes a loaded NVMe handle holds for the part it\n"       //
    "    rewrites; 0 derives what rounds that part in RAM.\n"                                  //
    "  progress (callable, optional): Called with (completed, total) units of each phase;\n"   //
    "    returning False stops the round, None continues.\n"                                   //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"    //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"       //
    "  plan (DenseRefinePlan, optional): The round to open or continue; None runs one round\n" //
    "    to its end.\n"                                                                        //
    "\n"                                                                                       //
    "Returns:\n"                                                                               //
    "  RefineReport: Entries re-selected and changed, the islands found and reconnected,\n"    //
    "    the neighborhoods, links added and dropped, the `RefineVerdict`, and the health\n"    //
    "    checks opening the round and closing the heal.\n"                                     //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> from array import array\n"                                                          //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=4)\n"                                //
    "  >>> index.reserve(1)\n"                                                                 //
    "  >>> index.insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))\n"                              //
    "  >>> index.refine().verdict\n"                                                           //
    "  <RefineVerdict.CONVERGED: 2>";

static PyObject* DenseIndex_refine(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *expansion_object, *memory_limit_object, *progress_object, *progress_steps_object, *stats_object,
        *plan_object;
    usearch_py_argument_t const arguments[] = {
        {"expansion_factor", &expansion_object},
        {"memory_limit", &memory_limit_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
        {"plan", &plan_object},
    };
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t expansion_factor = 0, memory_limit = 0, progress_steps = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("refine", args, positional_args_count, args_names_tuple, 0, 0, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(memory_limit_object, "memory_limit", &memory_limit) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    RefineReport* const report = usearch_py_call_result(call, &RefineReportType);
    if (!report || walk_parse(call, progress_object, stats_object) != 0 ||
        plan_parse(call, plan_object, &DenseRefinePlanType) != 0)
        return usearch_py_call_discard(&receiver, call);

    DensePlan* const plan = (DensePlan*)call->holds[callback_slot_k];
    usearch_dense_refine_t* const refine = (usearch_dense_refine_t*)call->arguments;
    *refine = (usearch_dense_refine_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .report = &report->report,
        .refine = {.expansion_factor = expansion_factor, .memory_limit = memory_limit},
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
        .plan = plan ? &plan->handle.refine : NULL,
        .phase = plan ? &plan->progress : NULL,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_refine(refine);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Ingest

#pragma region Search

static char const doc_DenseIndex_search[] =                                                    //
    "``search(queries, wanted, arena, /, *, expansion_factor=0, traversal=Traversal.WALK,\n"   //
    "dtype=None, predicate=None, progress=None, progress_steps=0, stats=None) -> Matches``\n"  //
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
    "    `Traversal.SCAN` scores every entry exactly, on RAM and NVMe handles.\n"              //
    "  dtype (Scalar, optional): Names the query scalar type when the buffer cannot.\n"        //
    "  predicate (callable, optional): Admits a Key when it returns truthy. Non-matching\n"    //
    "    entries still route the traversal, so recall survives low selectivity. The\n"         //
    "    callback re-enters Python, so it runs under the GIL and serializes the walk; what\n"  //
    "    it raises goes to `sys.unraisablehook`, and the key is rejected.\n"                   //
    "  progress (callable, optional): Called with (completed, total) queries; returning\n"     //
    "    False stops the batch and raises `CancelledError`, None continues.\n"                 //
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
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("search", args, positional_args_count, args_names_tuple, 3, 3, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        (traversal_object &&
         usearch_py_enum_from_object(traversal_object, "traversal", &usearch_py_traversal, &traversal) != 0) ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    usearch_matrix_view_t queries;
    Matches* matches = NULL;
    if (usearch_py_matrix_from_object(queries_object, dtype, &queries, &call->buffers[0]) != 0 ||
        usearch_py_callable_from_object(predicate_object, "predicate", &call->holds[callback_slot_k]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0 ||
        !(matches = usearch_py_call_arena(call, arena_object, &MatchesType)))
        return usearch_py_call_discard(&receiver, call);

    matches->queries = queries.rows;
    usearch_dense_search_t* const search = (usearch_dense_search_t*)call->arguments;
    *search = (usearch_dense_search_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .count = queries.rows,
        .queries = rows_of(&queries),
        .matches = &matches->view,
        .arena = usearch_py_arena_handle(call->arena),
        .wanted = wanted,
        .expansion_factor = expansion_factor,
        .traversal = (usearch_traversal_t)traversal,
        .predicate = usearch_py_predicate_registered(call->holds[callback_slot_k]),
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_search(search);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
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

/** A @c usearch_dense_count call and the count it lands, inside the record's arguments. */
typedef struct count_call_t {
    usearch_dense_count_t arguments;
    size_t entries;
} count_call_t;

static PyObject* count_answer(usearch_py_call_t* call) {
    return PyLong_FromSize_t(((count_call_t const*)call->arguments)->entries);
}

static PyObject* DenseIndex_count(PyObject* self, PyObject* key_object) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    usearch_key_t key;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || usearch_py_key_from_object(key_object, &key) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    count_call_t* const count = (count_call_t*)call->arguments;
    call->answer = count_answer;
    count->arguments = (usearch_dense_count_t) {.index = handle,
                                                .executor = receiver.executor,
                                                .outcome = &call->outcome,
                                                .completion = receiver.completion,
                                                .key = key,
                                                .entries = &count->entries};
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_count(&count->arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static int DenseIndex_contains(PyObject* self, PyObject* key_object) {
    PyObject* const entries = DenseIndex_count(self, key_object);
    if (!entries) return -1;
    int const present = PyObject_IsTrue(entries);
    Py_DECREF(entries);
    return present;
}

static char const doc_DenseIndex_get[] =                                                          //
    "``get(key, arena, /, *, outputs_dtype=Scalar.F32) -> DenseVectors``\n"                       //
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

/** Answers the rows a get wrote, typed as asked even when an absent key left the view zeroed. */
static PyObject* get_answer(usearch_py_call_t* call) {
    ArenaBuffer* const vectors = (ArenaBuffer*)call->result;
    if (!vectors->count) vectors->rows.scalar = ((usearch_dense_get_t const*)call->arguments)->outputs_scalar;
    return usearch_py_rows_answer(call);
}

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
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("get", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_scalar_from_object(outputs_dtype_object, "outputs_dtype", &outputs_kind) != 0 ||
        usearch_py_key_from_object(key_object, &key) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    ArenaBuffer* const vectors = usearch_py_call_arena(call, arena_object, &DenseVectorsType);
    if (!vectors) return usearch_py_call_discard(&receiver, call);
    call->answer = get_answer;
    usearch_dense_get_t* const get = (usearch_dense_get_t*)call->arguments;
    *get = (usearch_dense_get_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .key = key,
        .vectors = &vectors->rows,
        .written = &vectors->count,
        .arena = usearch_py_arena_handle(call->arena),
        .outputs_scalar = outputs_kind == usearch_scalar_unknown_k ? usearch_scalar_f32_k : outputs_kind,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_get(get);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Retrieval

#pragma region Linear Algebra

static char const doc_DenseIndex_multiply[] =                                                   //
    "``multiply(rows, wanted, arena, /, *, expansion_factor=0, dtype=None, progress=None,\n"    //
    "progress_steps=0, stats=None) -> DenseProducts``\n"                                        //
    "\n"                                                                                        //
    "Approximate sparse product of a matrix against the indexed vectors - the linear algebra\n" //
    "sibling of `search`, reporting every pair the traversal scored rather than only the\n"     //
    "nearest. Pairs arrive in traversal order rather than sorted, and hold no duplicates.\n"    //
    "How many a walk scores is known only once it ends, so the pairs come back ragged.\n"       //
    "Calls `usearch_dense_multiply`.\n"                                                         //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  rows (buffer): An (R, dimensions) matrix of left-hand rows.\n"                           //
    "  wanted (int): Upper bound on pairs kept per row, the first ones its walk scored;\n"      //
    "    0 keeps every pair.\n"                                                                 //
    "  arena (Arena): Where the pairs and their offsets land, rewound by the call.\n"           //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it. A wider\n"    //
    "    factor widens the emitted pattern, being what the walk's breadth decides.\n"           //
    "  dtype (Scalar, optional): Names the left-hand scalar type when the buffer cannot.\n"     //
    "  progress (callable, optional): Called with (completed, total) rows; returning False\n"   //
    "    stops the batch and raises `CancelledError`, None continues.\n"                        //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"     //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"        //
    "\n"                                                                                        //
    "Returns:\n"                                                                                //
    "  DenseProducts: `offsets`, (R + 1,) Apache Arrow offsets, left row `i` spanning pairs\n"  //
    "    `offsets[i]` up to `offsets[i + 1]`; `keys`, a (pairs, 16) run naming each pair's\n"   //
    "    indexed entry; `scores`, the (pairs,) float64 products; and `required`, the pairs\n"   //
    "    scored before `wanted` cut any. All of them live in the arena.";

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
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("multiply", args, positional_args_count, args_names_tuple, 3, 3, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    usearch_matrix_view_t rows;
    DenseProducts* products = NULL;
    if (usearch_py_matrix_from_object(rows_object, dtype, &rows, &call->buffers[0]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0 ||
        !(products = usearch_py_call_arena(call, arena_object, &DenseProductsType)))
        return usearch_py_call_discard(&receiver, call);

    products->queries = rows.rows;
    usearch_dense_multiply_t* const multiply = (usearch_dense_multiply_t*)call->arguments;
    *multiply = (usearch_dense_multiply_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .count = rows.rows,
        .left = rows_of(&rows),
        .products = &products->view,
        .written = &products->pairs,
        .required = &products->required,
        .arena = usearch_py_arena_handle(call->arena),
        .wanted = wanted,
        .expansion_factor = expansion_factor,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_multiply(multiply);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_DenseIndex_attend[] =                                                      //
    "``attend(queries, values, row_of, arena, /, *, head_count=1, expansion_factor=0,\n"         //
    "score_scale=0.0, dtype=None, values_dtype=None, outputs_dtype=None, carry=None,\n"          //
    "progress=None, progress_steps=0, stats=None) -> DenseVectors``\n"                           //
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
    "    stored in the index's quantization, read until the call is final.\n"                    //
    "  row_of (callable): Called with each scored Key, answers its row of `values`. None, a\n"   //
    "    row outside them or an exception, which goes to `sys.unraisablehook`, refuses the\n"    //
    "    call with `ShapeMismatchError`, as do values of another width or scalar type. It\n"     //
    "    re-enters Python from the library's threads, so it\n"                                   //
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
    "    False stops the batch and raises `CancelledError`, None continues.\n"                   //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"      //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"         //
    "\n"                                                                                         //
    "Returns:\n"                                                                                 //
    "  DenseVectors: The (Q, dimensions) attended rows in the arena, already divided by the\n"   //
    "    partition; kinds `memoryview` cannot format leave as a (Q, row_bytes) run of 'B'.";

/** A @c usearch_dense_attend call and what its value lookup reads, in the record's arguments. */
typedef struct attend_call_t {
    usearch_dense_attend_t arguments;

    /** The caller's value lookup, held among the call's holds. */
    PyObject* row_of;

    /** The caller's value rows, exported until the call is final. */
    usearch_matrix_view_t values;
} attend_call_t;

/**
 *  @brief Re-enters Python for the rows of one scored batch, from whichever library thread walks.
 *
 *  Values of another scalar type or width than the library reads resolve no row. A key whose
 *  lookup answers None, a row outside the values, or an exception, which goes to the unraisable
 *  hook, leaves its row and the rest NULL, which refuses the call.
 */
static void value_rows_trampoline(void* state, usearch_key_t const* keys, size_t count, usearch_scalar_t scalar,
                                  size_t columns, void const** rows) {
    attend_call_t const* const lookup = (attend_call_t const*)state;
    for (size_t position = 0; position != count; ++position) rows[position] = NULL;
    if (lookup->values.scalar != scalar || lookup->values.columns != columns) return;
    PyGILState_STATE const gil_state = PyGILState_Ensure();
    for (size_t position = 0; position != count; ++position) {
        PyObject* const key = usearch_py_key_to_object(keys[position]);
        PyObject* const answered = key ? PyObject_CallOneArg(lookup->row_of, key) : NULL;
        Py_XDECREF(key);
        Py_ssize_t const row = answered && answered != Py_None ? PyNumber_AsSsize_t(answered, PyExc_IndexError) : -1;
        Py_XDECREF(answered);
        if (row < 0 || (size_t)row >= lookup->values.rows) break;
        rows[position] = (char const*)lookup->values.data + (size_t)row * lookup->values.stride;
    }
    if (PyErr_Occurred()) PyErr_WriteUnraisable(lookup->row_of);
    PyGILState_Release(gil_state);
}

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
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("attend", args, positional_args_count, args_names_tuple, 4, 4, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(head_count_object, "head_count", &head_count) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        (scale_object && (score_scale = PyFloat_AsDouble(scale_object)) == -1.0 && PyErr_Occurred()) ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0 ||
        usearch_py_scalar_from_object(values_dtype_object, "values_dtype", &values_kind) != 0 ||
        usearch_py_scalar_from_object(outputs_dtype_object, "outputs_dtype", &outputs_kind) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    attend_call_t* const attend = (attend_call_t*)call->arguments;
    usearch_matrix_view_t queries;
    Py_ssize_t carry_bytes = 0;
    void* carry = NULL;
    ArenaBuffer* outputs = NULL;
    usearch_arena_description_t placed = {0};
    if (usearch_py_matrix_from_object(queries_object, dtype, &queries, &call->buffers[0]) != 0 ||
        usearch_py_matrix_from_object(values_object, values_kind, &attend->values, &call->buffers[1]) != 0 ||
        usearch_py_callable_from_object(row_of_object, "row_of", &call->holds[callback_slot_k]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0 ||
        (carry_object && carry_object != Py_None &&
         (usearch_py_matrix_bytes(queries.rows, sizeof(usearch_dense_softmax_state_t), 1, &carry_bytes) != 0 ||
          usearch_py_output_from_object(carry_object, "carry", carry_bytes, &call->buffers[2], &carry) != 0)) ||
        !(outputs = usearch_py_call_arena(call, arena_object, &DenseVectorsType)))
        return usearch_py_call_discard(&receiver, call);
    // A VRAM walk calls its value lookup on the device, which no Python callable can serve.
    if (call->arena && usearch_py_arena_describe(call->arena, &placed) != 0)
        return usearch_py_call_discard(&receiver, call);
    if (placed.residency == usearch_residency_vram_k) {
        usearch_py_raise(usearch_status_device_k, "a VRAM walk needs a device-callable value lookup");
        return usearch_py_call_discard(&receiver, call);
    }

    call->answer = usearch_py_rows_answer;
    attend->row_of = call->holds[callback_slot_k];
    attend->arguments = (usearch_dense_attend_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .count = queries.rows,
        .queries = rows_of(&queries),
        .head_count = head_count,
        .value_lookup = attend->row_of ? value_rows_trampoline : NULL,
        .value_lookup_state = attend,
        .carry = (usearch_dense_softmax_state_t*)carry,
        .outputs = &outputs->rows,
        .arena = usearch_py_arena_handle(call->arena),
        .outputs_scalar = outputs_kind,
        .expansion_factor = expansion_factor,
        .score_scale = (float)score_scale,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
    };
    outputs->count = queries.rows;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_attend(&attend->arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_DenseIndex_maxsim[] =                                                      //
    "``maxsim(queries, rows_per_query, wanted, arena, /, *, expansion_factor=0,\n"               //
    "traversal=Traversal.WALK, dtype=None, predicate=None, progress=None,\n"                     //
    "progress_steps=0, stats=None) -> DenseProducts``\n"                                         //
    "\n"                                                                                         //
    "Late-interaction scoring of multi-row queries against multi-vector documents: every\n"      //
    "query row takes its best match within a key, and those maxima sum to that key's score.\n"   //
    "\n"                                                                                         //
    "A document is one key holding its token rows, which is what `values_per_key` above one\n"   //
    "exists for, and the walk stays over individual tokens under a plain inner product - so\n"   //
    "a document's vectors need no ordering and two documents need not share a vector count.\n"   //
    "The result is sized over documents rather than over rows.\n"                                //
    "Calls `usearch_dense_maxsim`.\n"                                                            //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  queries (buffer): The token rows of every query laid end to end, an (Q * R,\n"            //
    "    dimensions) matrix.\n"                                                                  //
    "  rows_per_query (int): Rows every query occupies; uniform, so ColBERT's padding\n"         //
    "    applies as is. The query row count must be a whole multiple of it.\n"                   //
    "  wanted (int): Upper bound on matches per query - the 'k' over documents, not rows;\n"     //
    "    0 keeps every document the query scored.\n"                                             //
    "  arena (Arena): Where the scored documents land, rewound by the call.\n"                   //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it. Widening it\n" //
    "    nominates more documents, each still scored exactly.\n"                                 //
    "  traversal (Traversal, optional): `Traversal.WALK`, the default, nominates documents by\n" //
    "    walking the graph; `Traversal.SCAN` nominates every document, on RAM and NVMe.\n"       //
    "  dtype (Scalar, optional): Names the query scalar type when the buffer cannot.\n"          //
    "  predicate (callable, optional): Admits a Key when it returns truthy, asked before any\n"  //
    "    of its vectors are scored, so a rejected document never scores.\n"                      //
    "  progress (callable, optional): Called with (completed, total) queries; returning\n"       //
    "    False stops the batch and raises `CancelledError`, None continues.\n"                   //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"      //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"         //
    "\n"                                                                                         //
    "Returns:\n"                                                                                 //
    "  DenseProducts: `offsets`, (Q + 1,) Apache Arrow offsets, query `i` spanning documents\n"  //
    "    `offsets[i]` up to `offsets[i + 1]`, largest first; `keys`, a (documents, 16) run of\n" //
    "    identities; `scores`, their (documents,) summed maxima; and `required`, the\n"          //
    "    documents scored before `wanted` cut any. All of them live in the arena.";

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
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("maxsim", args, positional_args_count, args_names_tuple, 4, 4, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(rows_per_query_object, "rows_per_query", &rows_per_query) != 0 ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        (traversal_object &&
         usearch_py_enum_from_object(traversal_object, "traversal", &usearch_py_traversal, &traversal) != 0) ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    usearch_matrix_view_t queries;
    if (usearch_py_matrix_from_object(queries_object, dtype, &queries, &call->buffers[0]) != 0)
        return usearch_py_call_discard(&receiver, call);
    // The library takes a query count, which only the rows of this matrix can tell.
    if (rows_per_query == 0 || queries.rows % rows_per_query != 0) {
        usearch_py_raise(usearch_status_shape_mismatch_k, "the %zu query rows are not a whole multiple of %zu",
                         queries.rows, rows_per_query);
        return usearch_py_call_discard(&receiver, call);
    }
    DenseProducts* matches = NULL;
    if (usearch_py_callable_from_object(predicate_object, "predicate", &call->holds[callback_slot_k]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0 ||
        !(matches = usearch_py_call_arena(call, arena_object, &DenseProductsType)))
        return usearch_py_call_discard(&receiver, call);

    matches->queries = queries.rows / rows_per_query;
    usearch_dense_maxsim_t* const maxsim = (usearch_dense_maxsim_t*)call->arguments;
    *maxsim = (usearch_dense_maxsim_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .count = matches->queries,
        .queries = rows_of(&queries),
        .rows_per_query = rows_per_query,
        .matches = &matches->view,
        .written = &matches->pairs,
        .required = &matches->required,
        .arena = usearch_py_arena_handle(call->arena),
        .wanted = wanted,
        .expansion_factor = expansion_factor,
        .traversal = (usearch_traversal_t)traversal,
        .predicate = usearch_py_predicate_registered(call->holds[callback_slot_k]),
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_maxsim(maxsim);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Linear Algebra

#pragma region Composition

static char const doc_DenseIndex_reshard[] =                                                     //
    "``reshard(sources, targets, /, *, memory_domains=None, devices=None, paths=None,\n"         //
    "shard_capacity=0, expansion_factor=0, memory_limit=0, seed=0, progress=None,\n"             //
    "progress_steps=0, stats=None, plan=None) -> RefineReport``\n"                               //
    "\n"                                                                                         //
    "Moves the entries of `sources` into `targets`, keeping their links rather than\n"           //
    "re-inserting their vectors, then runs a round on every target that took entries. Several\n" //
    "sources into one target, itself among them, gather into it; one source into several\n"      //
    "targets cuts it into one graph-local part per target; one index as its only source and\n"   //
    "target recuts its own shards in place under the placement's `shard_capacity`. Every\n"      //
    "index stays the caller's, and a source the targets do not name ends empty and live. The\n"  //
    "receiver names where the call runs, and need not be among them. Readers of a RAM target\n"  //
    "keep searching throughout, while a VRAM target admits no overlapping search.\n"             //
    "\n"                                                                                         //
    "The handler stops the closing rounds alone, never a move: a stop raises `CancelledError`\n" //
    "with every entry moved, or, with a `plan`, leaves it live for the next call naming it to\n" //
    "continue, until `reshard_plan_finish(plan)` closes it. Calls `usearch_dense_reshard`.\n"    //
    "\n"                                                                                         //
    "A live plan holds no index, so each index it names must outlive its finish.\n"              //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  sources (Sequence[DenseIndex]): The indexes whose entries move, each named once.\n"       //
    "  targets (Sequence[DenseIndex]): The indexes they move into, each a source or empty.\n"    //
    "  memory_domains (Sequence[int], optional): NUMA node IDs of a RAM placement recut in\n"    //
    "    place; None keeps the placement.\n"                                                     //
    "  devices (Sequence[int], optional): Device ordinals of a VRAM placement; None keeps it.\n" //
    "  paths (Sequence[str | bytes | PathLike], optional): Drives of an NVMe placement;\n"       //
    "    None keeps it.\n"                                                                       //
    "  shard_capacity (int, optional): Entries per shard of the RAM or VRAM placement; 0\n"      //
    "    keeps the handle's own bound.\n"                                                        //
    "  expansion_factor (int, optional): Breadth of any walk beside the rounds; 0 derives it.\n" //
    "  memory_limit (int, optional): Bytes a loaded NVMe handle holds for the part it\n"         //
    "    rewrites; 0 derives what rounds that part in RAM.\n"                                    //
    "  seed (int, optional): Picks the part roots of a cut; equal seeds cut equally.\n"          //
    "  progress (callable, optional): Called with (completed, total) units of each phase of\n"   //
    "    the closing rounds; returning False stops them, None continues.\n"                      //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"      //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"         //
    "  plan (DenseReshardPlan, optional): The reshard to open or continue; None runs it to\n"    //
    "    its end.\n"                                                                             //
    "\n"                                                                                         //
    "Returns:\n"                                                                                 //
    "  RefineReport: What the closing rounds did to the graph, across the targets.\n"            //
    "\n"                                                                                         //
    "Example:\n"                                                                                 //
    "  >>> from array import array\n"                                                            //
    "  >>> base, donor = usearch.DenseIndex.init_ram(2), usearch.DenseIndex.init_ram(2)\n"       //
    "  >>> base.reserve(1)\n"                                                                    //
    "  >>> donor.reserve(1)\n"                                                                   //
    "  >>> base.insert([1], array('d', [0.1, 0.2]))\n"                                           //
    "  >>> donor.insert([2], array('d', [0.3, 0.4]))\n"                                          //
    "  >>> pool = usearch.Executor.for_ram(threads=2)\n"                                         //
    "  >>> _ = base.on(pool).reshard([base, donor], [base])\n"                                   //
    "  >>> len(base), len(donor)\n"                                                              //
    "  (2, 0)";

/** A @c usearch_dense_reshard call and the placements it passes, inside the record's arguments. */
typedef struct reshard_call_t {
    usearch_dense_reshard_t arguments;
    usearch_ram_options_t ram;
    usearch_vram_options_t vram;
    usearch_nvme_options_t nvme;
} reshard_call_t;

/**
 *  @brief Holds the sequence of indexes @p object names in @p held and passes their handles as one
 *      block in @p handles.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int indexes_from_object(PyObject* object, char const* field, PyObject** held, void** handles, size_t* count) {
    PyObject* const sequence = PySequence_Fast(object, "expected a sequence of indexes");
    if (!sequence) return usearch_py_raise_from_cause(PyExc_TypeError, "the '%s' argument must be a sequence", field);
    *held = sequence;
    Py_ssize_t const length = PySequence_Fast_GET_SIZE(sequence);
    usearch_dense_index_t* const block = PyMem_Malloc((length ? (size_t)length : 1) * sizeof(usearch_dense_index_t));
    if (!block) return PyErr_NoMemory(), -1;
    *handles = block;
    for (Py_ssize_t position = 0; position != length; ++position) {
        PyObject* const index = PySequence_Fast_GET_ITEM(sequence, position);
        if (!Py_IS_TYPE(index, &DenseIndexType)) {
            PyErr_Format(PyExc_TypeError, "the '%s' argument must hold DenseIndex objects, got %S", field,
                         Py_TYPE(index));
            return -1;
        }
        block[position] = ((DenseIndex*)index)->handle;
    }
    *count = (size_t)length;
    return 0;
}

static PyObject* DenseIndex_reshard(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                    PyObject* args_names_tuple) {
    PyObject *sources_object, *targets_object, *memory_domains_object, *devices_object, *paths_object,
        *shard_capacity_object, *expansion_object, *memory_limit_object, *seed_object, *progress_object,
        *progress_steps_object, *stats_object, *plan_object;
    usearch_py_argument_t const arguments[] = {
        {"sources", &sources_object},
        {"targets", &targets_object},
        {"memory_domains", &memory_domains_object},
        {"devices", &devices_object},
        {"paths", &paths_object},
        {"shard_capacity", &shard_capacity_object},
        {"expansion_factor", &expansion_object},
        {"memory_limit", &memory_limit_object},
        {"seed", &seed_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
        {"plan", &plan_object},
    };
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t shard_capacity = 0, expansion_factor = 0, memory_limit = 0, progress_steps = 0;
    uint64_t seed = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("reshard", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(shard_capacity_object, "shard_capacity", &shard_capacity) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(memory_limit_object, "memory_limit", &memory_limit) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        usearch_py_seed_from_object(seed_object, &seed) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    reshard_call_t* const reshard = (reshard_call_t*)call->arguments;
    size_t sources_count = 0, targets_count = 0, memory_domains_count = 0, devices_count = 0;
    RefineReport* const report = usearch_py_call_result(call, &RefineReportType);
    if (!report ||
        indexes_from_object(sources_object, "sources", &call->holds[sources_slot_k], &call->blocks[0],
                            &sources_count) != 0 ||
        indexes_from_object(targets_object, "targets", &call->holds[targets_slot_k], &call->blocks[1],
                            &targets_count) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", (int32_t**)&call->blocks[2],
                                        &memory_domains_count) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", (int32_t**)&call->blocks[3], &devices_count) != 0 ||
        usearch_py_paths_from_object(paths_object, &call->paths) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0 ||
        plan_parse(call, plan_object, &DenseReshardPlanType) != 0)
        return usearch_py_call_discard(&receiver, call);

    DensePlan* const plan = (DensePlan*)call->holds[callback_slot_k];
    reshard->ram = (usearch_ram_options_t) {.memory_domains = call->blocks[2],
                                            .memory_domains_count = memory_domains_count,
                                            .shard_capacity = shard_capacity};
    reshard->vram = (usearch_vram_options_t) {.devices = call->blocks[3],
                                              .devices_count = devices_count,
                                              .shard_capacity = shard_capacity};
    reshard->nvme = (usearch_nvme_options_t) {.paths = call->paths.views, .paths_count = call->paths.count};
    reshard->arguments = (usearch_dense_reshard_t) {
        .sources_count = sources_count,
        .sources = call->blocks[0],
        .targets_count = targets_count,
        .targets = call->blocks[1],
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .ram = memory_domains_object && memory_domains_object != Py_None ? &reshard->ram : NULL,
        .vram = devices_object && devices_object != Py_None ? &reshard->vram : NULL,
        .nvme = paths_object && paths_object != Py_None ? &reshard->nvme : NULL,
        .report = &report->report,
        .refine = {.expansion_factor = expansion_factor, .memory_limit = memory_limit},
        .seed = seed,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
        .plan = plan ? &plan->handle.reshard : NULL,
        .phase = plan ? &plan->progress : NULL,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_reshard(&reshard->arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Composition

#pragma region Sampling

static char const doc_DenseIndex_sample[] =                                                          //
    "``sample(wanted, arena, /, *, shown=None, shown_coordinates=None, focus=None, seed=0,\n"        //
    "expansion_factor=0, progress=None, progress_steps=0, stats=None) -> Sample``\n"                 //
    "\n"                                                                                             //
    "Draws keys around `focus`, or across the whole index, each hanging from a shown or\n"           //
    "earlier draw and placed in three dimensions beside the shown ones - so a view grows one\n"      //
    "connected sample as it zooms, never scanning the index. Equal seeds, shown keys and\n"          //
    "focus over an unchanged index draw equally, and a larger `wanted` draws a superset.\n"          //
    "Only a RAM index samples yet.\n"                                                                //
    "Calls `usearch_dense_sample`.\n"                                                                //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  wanted (int): The upper bound on draws.\n"                                                    //
    "  arena (Arena): Where the draws are written, rewound by the call.\n"                           //
    "  shown (Sequence[Key | int] | buffer, optional): The keys already shown, never drawn again.\n" //
    "  shown_coordinates (tensor, optional): (len(shown), 3) float32 places of the shown\n"          //
    "    keys, the points later draws are placed against.\n"                                         //
    "  focus (Sequence[Key | int] | buffer, optional): The keys to look around; None looks\n"        //
    "    across the index.\n"                                                                        //
    "  seed (int, optional): The draw; equal seeds draw equally.\n"                                  //
    "  expansion_factor (int, optional): Breadth of the walks behind the draws; 0 derives.\n"        //
    "  progress (callable, optional): Called with (written, wanted) draws.\n"                        //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"          //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"             //
    "\n"                                                                                             //
    "Returns:\n"                                                                                     //
    "  Sample: The keys, entry ordinals, parents and coordinates of the draws, fewer than\n"         //
    "    `wanted` only once the region is exhausted.";

/** Gathers the optional keys @p object names into @p keys; None or NULL leaves them empty. Returns
 *  0 or -1 with an exception set. */
static int optional_keys_from_object(PyObject* object, size_t* count, usearch_py_keys_t* keys) {
    *count = 0;
    if (!object || object == Py_None) return 0;
    Py_ssize_t const length = PyObject_Length(object);
    if (length < 0 || usearch_py_keys_from_object(object, (size_t)length, keys) != 0) return -1;
    *count = (size_t)length;
    return 0;
}

static PyObject* DenseIndex_sample(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *wanted_object, *arena_object, *shown_object, *coordinates_object, *focus_object, *seed_object,
        *expansion_object, *progress_object, *progress_steps_object, *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"wanted", &wanted_object},
        {"arena", &arena_object},
        {"shown", &shown_object},
        {"shown_coordinates", &coordinates_object},
        {"focus", &focus_object},
        {"seed", &seed_object},
        {"expansion_factor", &expansion_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t wanted = 0, expansion_factor = 0, progress_steps = 0;
    uint64_t seed = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("sample", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        usearch_py_seed_from_object(seed_object, &seed) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    usearch_matrix_view_t coordinates = {0};
    size_t shown_count = 0, focus_count = 0;
    Sample* sample = NULL;
    if ((coordinates_object && coordinates_object != Py_None &&
         usearch_py_matrix_from_object(coordinates_object, usearch_scalar_f32_k, &coordinates, &call->buffers[0]) !=
             0) ||
        optional_keys_from_object(shown_object, &shown_count, &call->keys[0]) != 0 ||
        optional_keys_from_object(focus_object, &focus_count, &call->keys[1]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0 ||
        !(sample = usearch_py_call_arena(call, arena_object, &SampleType)))
        return usearch_py_call_discard(&receiver, call);
    // The library reads as many coordinates as shown keys, which only these two arguments can tell.
    if (shown_count != coordinates.rows) {
        PyErr_Format(PyExc_ValueError, "%zu shown keys need as many coordinates, got %zu", shown_count,
                     coordinates.rows);
        return usearch_py_call_discard(&receiver, call);
    }

    usearch_dense_sample_t* const arguments_sample = (usearch_dense_sample_t*)call->arguments;
    *arguments_sample = (usearch_dense_sample_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .shown_count = shown_count,
        .shown = {call->keys[0].keys, sizeof(usearch_key_t)},
        .shown_coordinates = rows_of(&coordinates),
        .focus_count = focus_count,
        .focus = {call->keys[1].keys, sizeof(usearch_key_t)},
        .keys = &sample->keys,
        .entry_ordinals = &sample->entry_ordinals,
        .parents = &sample->parents,
        .coordinates = &sample->coordinates,
        .written = &sample->written,
        .arena = usearch_py_arena_handle(call->arena),
        .wanted = wanted,
        .seed = seed,
        .expansion_factor = expansion_factor,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_sample(arguments_sample);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Sampling

#pragma region Plans

static char const doc_DenseRefinePlan[] =                                                          //
    "DenseRefinePlan()\n"                                                                          //
    "\n"                                                                                           //
    "A refine in flight: the round in hand, and the root every round's health checks share.\n"     //
    "\n"                                                                                           //
    "Passed as `plan=` to `refine`, the first call opens it and a stop leaves it live, so the\n"   //
    "next call naming it continues the round where it stopped, or starts the next, and\n"          //
    "`refine_plan_finish` closes it. Until then the index admits no insertion and no save, and\n"  //
    "every call names the index and executor it was opened through. A plan dropped unfinished\n"   //
    "leaks and keeps the index closed to insertion.\n"                                             //
    "\n"                                                                                           //
    "Attributes:\n"                                                                                //
    "  phase (Phase): Where the last call naming it stopped, `Phase.DONE` once its round ended.\n" //
    "  ordinal (int): That phase's position among the ones the plan runs.\n"                       //
    "  completed (int): Units of that phase done, entries or islands.\n"                           //
    "  total (int): Units of that phase in all.\n"                                                 //
    "\n"                                                                                           //
    "Example:\n"                                                                                   //
    "  >>> from array import array\n"                                                              //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=2)\n"                                    //
    "  >>> index.reserve(1)\n"                                                                     //
    "  >>> index.insert([1], array('d', [0.1, 0.2]))\n"                                            //
    "  >>> pool, plan = usearch.Executor.for_ram(threads=2), usearch.DenseRefinePlan()\n"          //
    "  >>> _ = index.on(pool).refine(plan=plan)\n"                                                 //
    "  >>> plan.phase\n"                                                                           //
    "  <Phase.DONE: 11>\n"                                                                         //
    "  >>> index.on(pool).refine_plan_finish(plan).verdict\n"                                      //
    "  <RefineVerdict.CONVERGED: 2>";

static char const doc_DenseReshardPlan[] =                                                          //
    "DenseReshardPlan()\n"                                                                          //
    "\n"                                                                                            //
    "A reshard in flight: its own sources and targets, the entries still to move, then the\n"       //
    "rounds of every target that took entries.\n"                                                   //
    "\n"                                                                                            //
    "Passed as `plan=` to `reshard`, the first call opens it and a stop leaves it live, so the\n"   //
    "next call naming it moves what is still pending and continues the rounds, reading its own\n"   //
    "sources, targets and placement rather than the call's, and `reshard_plan_finish` closes\n"     //
    "it. Until then every index it names admits no insertion and no save, and every call names\n"   //
    "the executor it was opened with. A plan dropped unfinished leaks and keeps its indexes\n"      //
    "closed to insertion.\n"                                                                        //
    "\n"                                                                                            //
    "Attributes:\n"                                                                                 //
    "  phase (Phase): Where the last call naming it stopped, `Phase.DONE` once its rounds ended.\n" //
    "  ordinal (int): That phase's position among the ones the plan runs.\n"                        //
    "  completed (int): Units of that phase done, entries or islands.\n"                            //
    "  total (int): Units of that phase in all.";

static PyObject* DensePlan_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    if (PyTuple_GET_SIZE(args) != 0 || (keywords && PyDict_Size(keywords) != 0)) {
        PyErr_Format(PyExc_TypeError, "%s() takes no arguments", type->tp_name + sizeof("usearch.") - 1);
        return NULL;
    }
    return type->tp_alloc(type, 0);
}

static PyObject* DensePlan_get_phase(DensePlan* self, void* closure) {
    return usearch_py_enum_to_object(&usearch_py_phase, (int)self->progress.phase);
}

static PyGetSetDef DensePlan_getset[] = {
    {"phase", (getter)DensePlan_get_phase, NULL, "Where the last call naming the plan stopped, as a `Phase`.", NULL},
    {"ordinal", usearch_py_size_getter, NULL, "That phase's position among the ones the plan runs.",
     (void*)offsetof(DensePlan, progress.ordinal)},
    {"completed", usearch_py_size_getter, NULL, "Units of that phase done, entries or islands.",
     (void*)offsetof(DensePlan, progress.completed)},
    {"total", usearch_py_size_getter, NULL, "Units of that phase in all.", (void*)offsetof(DensePlan, progress.total)},
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject DenseRefinePlanType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseRefinePlan",
    .tp_doc = doc_DenseRefinePlan,
    .tp_basicsize = sizeof(DensePlan),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = DensePlan_tp_new,
    .tp_getset = DensePlan_getset,
};

PyTypeObject DenseReshardPlanType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.DenseReshardPlan",
    .tp_doc = doc_DenseReshardPlan,
    .tp_basicsize = sizeof(DensePlan),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = DensePlan_tp_new,
    .tp_getset = DensePlan_getset,
};

static char const doc_DenseIndex_refine_plan_finish[] =                                         //
    "refine_plan_finish(plan) -> RefineReport\n"                                                //
    "\n"                                                                                        //
    "Closes a `DenseRefinePlan` on this index and the receiver's executor, the ones it was\n"   //
    "opened with, and frees it whatever the call answers; a round a stop left open closes as\n" //
    "a stop at that barrier would. Calls `usearch_dense_refine_plan_finish`.\n"                 //
    "\n"                                                                                        //
    "Returns:\n"                                                                                //
    "  RefineReport: What the whole plan did to the graph.";

static char const doc_DenseIndex_reshard_plan_finish[] =                                         //
    "reshard_plan_finish(plan) -> RefineReport\n"                                                //
    "\n"                                                                                         //
    "Closes a `DenseReshardPlan` on the receiver's executor, the one it was opened with, and\n"  //
    "frees it whatever the call answers, reopening every index it names; a plan whose entries\n" //
    "have not all moved runs no round and reports zeros. Calls\n"                                //
    "`usearch_dense_reshard_plan_finish`.\n"                                                     //
    "\n"                                                                                         //
    "Returns:\n"                                                                                 //
    "  RefineReport: What the whole plan did to the graph.";

/** Opens a finish of the plan @p plan_object of @p type through @p self: the report it writes into,
 *  the plan held, and its handle taken, since the library frees it whatever the call answers. */
static usearch_py_call_t* plan_finish_open(PyObject* self, PyObject* plan_object, PyTypeObject* type,
                                           usearch_py_receiver_t* receiver, usearch_dense_index_t* handle,
                                           usearch_py_call_t* stack, RefineReport** report) {
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, receiver, handle) != 0 || !(call = usearch_py_call_open(receiver, stack)))
        return NULL;
    if (!plan_object || plan_object == Py_None) {
        PyErr_Format(PyExc_TypeError, "expected a %s, got None", type->tp_name);
        usearch_py_call_discard(receiver, call);
        return NULL;
    }
    if (plan_parse(call, plan_object, type) != 0 || !(*report = usearch_py_call_result(call, &RefineReportType))) {
        usearch_py_call_discard(receiver, call);
        return NULL;
    }
    return call;
}

static PyObject* DenseIndex_refine_plan_finish(PyObject* self, PyObject* plan_object) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    usearch_py_call_t stack;
    RefineReport* report = NULL;
    usearch_py_call_t* const call = plan_finish_open(self, plan_object, &DenseRefinePlanType, &receiver, &handle,
                                                     &stack, &report);
    if (!call) return NULL;
    DensePlan* const plan = (DensePlan*)plan_object;
    usearch_dense_refine_plan_finish_t* const finish = (usearch_dense_refine_plan_finish_t*)call->arguments;
    *finish = (usearch_dense_refine_plan_finish_t) {.index = handle,
                                                    .executor = receiver.executor,
                                                    .outcome = &call->outcome,
                                                    .completion = receiver.completion,
                                                    .plan = plan->handle.refine,
                                                    .report = &report->report};
    plan->handle.refine = NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_refine_plan_finish(finish);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static PyObject* DenseIndex_reshard_plan_finish(PyObject* self, PyObject* plan_object) {
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    usearch_py_call_t stack;
    RefineReport* report = NULL;
    usearch_py_call_t* const call = plan_finish_open(self, plan_object, &DenseReshardPlanType, &receiver, &handle,
                                                     &stack, &report);
    if (!call) return NULL;
    DensePlan* const plan = (DensePlan*)plan_object;
    usearch_dense_reshard_plan_finish_t* const finish = (usearch_dense_reshard_plan_finish_t*)call->arguments;
    *finish = (usearch_dense_reshard_plan_finish_t) {.executor = receiver.executor,
                                                     .outcome = &call->outcome,
                                                     .completion = receiver.completion,
                                                     .plan = plan->handle.reshard,
                                                     .report = &report->report};
    plan->handle.reshard = NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_reshard_plan_finish(finish);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Plans

#pragma region Iteration

static char const doc_DenseCursor[] =                                                         //
    "A sequential scan over the entries an index holds, pulled in batches by `next`.\n"       //
    "\n"                                                                                      //
    "Each shard's publication watermark is read as the scan enters that shard, so an entry\n" //
    "once passed is never revisited, while a later shard may still hold entries inserted\n"   //
    "after the cursor opened. One step runs at a time; an overlapping step raises\n"          //
    "`ContendedError`. `close()`, or the end of a `with` block, releases the scan before\n"   //
    "the cursor is collected.";

/** Releases the engine cursor of @p self once, whichever thread closes it first. */
static void cursor_close(DenseCursor* self) {
    usearch_dense_cursor_t const handle = atomic_exchange(&self->handle, NULL);
    if (!handle) return;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_cursor_free(handle);
    Py_END_ALLOW_THREADS;
}

static void DenseCursor_dealloc(DenseCursor* self) {
    cursor_close(self);
    Py_XDECREF(self->receiver);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static char const doc_DenseBatch[] =                                                         //
    "DenseBatch\n"                                                                           //
    "\n"                                                                                     //
    "The entries one cursor step wrote into an `Arena`, viewing it until it rewinds. Its\n"  //
    "length counts them, so the empty batch a drained cursor answers is falsy.\n"            //
    "\n"                                                                                     //
    "Attributes:\n"                                                                          //
    "  keys (buffer): (count, 16) identities, most significant byte first.\n"                //
    "  links (tuple | None): (offsets, keys) of the identities each entry links to, entry\n" //
    "    `i` spanning `offsets[i]` up to `offsets[i + 1]`, or None when the step asked for none.";

/** The (offsets, keys) pair of the @p written entries' links in @p arena; a new reference, or NULL
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

/** A new @c DenseBatch over the @p written entries a step wrote into @p arena; NULL @p links means
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
     "(offsets, keys) of each entry's linked identities, or None when the step asked for none."},
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

static char const doc_DenseCursor_next[] =                                                       //
    "``next(wanted, arena, /, *, links=False) -> DenseBatch``\n"                                 //
    "\n"                                                                                         //
    "The next batch of at most `wanted` entries, in `arena`, and an empty one, so falsy, once\n" //
    "the scan is drained. Blocks on the executor the cursor was opened through, or on the\n"     //
    "calling thread.\n"                                                                          //
    "Calls `usearch_dense_cursor_next`.\n"                                                       //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  wanted (int): The most entries the batch holds.\n"                                        //
    "  arena (Arena): Where the batch lands, rewound by the step.\n"                             //
    "  links (bool): Also export the identities each entry links to; the vectors themselves\n"   //
    "    come from `get`.\n"                                                                     //
    "\n"                                                                                         //
    "Raises:\n"                                                                                  //
    "  InvalidArgumentError: When the cursor was closed.";

static PyObject* DenseCursor_next(DenseCursor* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                  PyObject* args_names_tuple) {
    PyObject *wanted_object, *arena_object, *links_object;
    usearch_py_argument_t const arguments[] = {
        {"wanted", &wanted_object},
        {"arena", &arena_object},
        {"links", &links_object},
    };
    usearch_py_receiver_t receiver;
    usearch_dense_index_t index = NULL;
    usearch_arena_t arena = NULL;
    size_t wanted = 0;
    int with_links = 0;
    if (usearch_py_arguments_parse("next", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0 ||
        (links_object && (with_links = PyObject_IsTrue(links_object)) < 0) ||
        receiver_from_object(self->receiver, &receiver, &index) != 0 ||
        usearch_py_arena_from_object(arena_object, &arena) != 0)
        return NULL;

    usearch_keys_view_t keys = {0};
    usearch_keys_tape_view_t links = {0};
    size_t written = 0;
    usearch_outcome_t outcome = {0};
    usearch_dense_cursor_next_t step = {
        .index = index,
        .executor = receiver.executor,
        .outcome = &outcome,
        .cursor = atomic_load(&self->handle),
        .keys = &keys,
        .links = with_links ? &links : NULL,
        .written = &written,
        .arena = arena,
        .wanted = wanted,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_cursor_next(&step);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return batch_answer(arena_object, &keys, with_links && links.offsets ? &links : NULL, written);
}

static PyObject* DenseCursor_close(PyObject* self, PyObject* unused) {
    cursor_close((DenseCursor*)self);
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

static char const doc_DenseIndex_cursor[] =                                                     //
    "cursor(starts=None) -> DenseCursor\n"                                                      //
    "\n"                                                                                        //
    "Opens a walk over the index, which `DenseCursor.next` pulls in batches, each into the\n"   //
    "arena that pull names. Without `starts` it yields every entry as stored; with them, the\n" //
    "start keys' own entries first in order, then the entries those link to breadth-first,\n"   //
    "each once. Each shard's watermark is read as the scan enters that shard, so an entry\n"    //
    "once passed is never revisited, while a later shard may still hold entries inserted\n"     //
    "after this call. Every pull blocks, on the receiver's executor or the calling thread.\n"   //
    "Calls `usearch_dense_cursor_init`.\n"                                                      //
    "\n"                                                                                        //
    "Example:\n"                                                                                //
    "  >>> from array import array\n"                                                           //
    "  >>> index = usearch.DenseIndex.init_ram(dimensions=2)\n"                                 //
    "  >>> index.reserve(3)\n"                                                                  //
    "  >>> for key in (1, 2, 3):\n"                                                             //
    "  ...     index.insert([key], array('d', [0.1, 0.2]))\n"                                   //
    "  >>> arena, cursor = usearch.Arena(), index.cursor()\n"                                   //
    "  >>> while batch := cursor.next(2, arena):\n"                                             //
    "  ...     print(len(batch))\n"                                                             //
    "  2\n"                                                                                     //
    "  1";

static PyObject* DenseIndex_cursor(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject* starts_object;
    usearch_py_argument_t const arguments[] = {{"starts", &starts_object}};
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    if (usearch_py_arguments_parse("cursor", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0)
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
    usearch_dense_cursor_init_t arguments_init = {.index = handle,
                                                  .outcome = &outcome,
                                                  .starts_count = starts_count,
                                                  .starts = starts.keys,
                                                  .cursor = &cursor_handle};
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
    atomic_init(&cursor->handle, cursor_handle);
    cursor->receiver = Py_NewRef(self);
    return (PyObject*)cursor;
}

static char const doc_DenseSearchStream[] =                                                 //
    "A resumable best-first stream of matches for one query, pulled by `next`.\n"           //
    "\n"                                                                                    //
    "Ordering beyond the expansion horizon is best-effort nearest, not globally exact.\n"   //
    "One pull runs at a time; an overlapping pull raises `ContendedError`. `close()`, or\n" //
    "the end of a `with` block, releases the walk before the stream is collected.";

/** Releases the engine stream of @p self once, whichever thread closes it first. */
static void stream_close(DenseSearchStream* self) {
    usearch_dense_search_stream_t const handle = atomic_exchange(&self->handle, NULL);
    if (!handle) return;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_search_stream_free(handle);
    Py_END_ALLOW_THREADS;
}

static void DenseSearchStream_dealloc(DenseSearchStream* self) {
    stream_close(self);
    Py_XDECREF(self->receiver);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static char const doc_DenseSearchStream_next[] =                                                //
    "next(wanted, arena, /) -> Matches\n"                                                       //
    "\n"                                                                                        //
    "The next at most `wanted` matches, as one query of `Matches` in `arena`, whose count is\n" //
    "zero once the walk is drained. Blocks on the executor the stream was opened through,\n"    //
    "or on the calling thread.\n"                                                               //
    "Calls `usearch_dense_search_stream_next`.\n"                                               //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  wanted (int): The most matches the pull writes.\n"                                       //
    "  arena (Arena): Where the matches land, rewound by the pull.\n"                           //
    "\n"                                                                                        //
    "Raises:\n"                                                                                 //
    "  InvalidArgumentError: When the stream was closed.";

static PyObject* DenseSearchStream_next(DenseSearchStream* self, PyObject* const* args,
                                        Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    PyObject *wanted_object, *arena_object;
    usearch_py_argument_t const arguments[] = {{"wanted", &wanted_object}, {"arena", &arena_object}};
    usearch_py_receiver_t receiver;
    usearch_dense_index_t index = NULL;
    usearch_arena_t arena = NULL;
    size_t wanted = 0;
    if (usearch_py_arguments_parse("next", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0 ||
        receiver_from_object(self->receiver, &receiver, &index) != 0 ||
        usearch_py_arena_from_object(arena_object, &arena) != 0)
        return NULL;

    usearch_matches_view_t matches = {0};
    usearch_outcome_t outcome = {0};
    usearch_dense_search_stream_next_t pull = {
        .index = index,
        .executor = receiver.executor,
        .outcome = &outcome,
        .stream = atomic_load(&self->handle),
        .matches = &matches,
        .arena = arena,
        .wanted = wanted,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_search_stream_next(&pull);
    Py_END_ALLOW_THREADS;
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return usearch_py_matches_new(arena_object, &matches, 1);
}

static PyObject* DenseSearchStream_close(PyObject* self, PyObject* unused) {
    stream_close((DenseSearchStream*)self);
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
    "``search_stream(query, /, *, expansion_factor=0, dtype=None) -> DenseSearchStream``\n"  //
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
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0)
        return NULL;

    usearch_matrix_view_t query;
    usearch_py_buffer_backing_t backing;
    if (usearch_py_matrix_from_object(query_object, dtype, &query, &backing) != 0) return NULL;

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
    atomic_init(&stream->handle, stream_handle);
    stream->receiver = Py_NewRef(self);
    return (PyObject*)stream;
}

#pragma endregion Iteration

#pragma region Serialization

/** Views a whole checkpoint blob, whatever its length. */
static int checkpoint_view(PyObject* object, usearch_py_buffer_backing_t* backing) {
    backing->synthesized_owner = NULL;
    if (PyObject_GetBuffer(object, &backing->buffer, PyBUF_C_CONTIGUOUS) != 0)
        return usearch_py_raise_from_cause(PyExc_ValueError, "the checkpoint buffer must be contiguous");
    return 0;
}

static char const doc_DenseIndex_save[] =                                                   //
    "``save(path, /, *, progress=None, progress_steps=0) -> None``\n"                       //
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

static PyObject* DenseIndex_save(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                 PyObject* args_names_tuple) {
    PyObject *path_object, *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {{"path", &path_object},
                                               {"progress", &progress_object},
                                               {"progress_steps", &progress_steps_object}};
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t progress_steps = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("save", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    usearch_bytes_view_t path;
    if (usearch_py_path_from_object(path_object, &path, &call->holds[callback_slot_k]) != 0 ||
        usearch_py_callable_from_object(progress_object, "progress", &call->holds[progress_slot_k]) != 0)
        return usearch_py_call_discard(&receiver, call);
    usearch_dense_save_t* const save = (usearch_dense_save_t*)call->arguments;
    *save = (usearch_dense_save_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .path = path,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_save(save);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_DenseIndex_save_buffer[] =                                             //
    "``save_buffer(arena, /, *, progress=None, progress_steps=0) -> Monolith``\n"            //
    "\n"                                                                                     //
    "Serializes the index as one monolith of every shard, the very bytes `save` writes to\n" //
    "a `.safegraphs` path.\n"                                                                //
    "Calls `usearch_dense_save_buffer`.\n"                                                   //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  arena (Arena): Where the monolith lands, rewound by the call.\n"                      //
    "  progress (callable, optional): Called with (completed, total) bytes; returning\n"     //
    "    False stops the write and raises `CancelledError`, None continues.\n"               //
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

/** A @c usearch_dense_save_buffer call and the monolith it lands, inside the record's arguments. */
typedef struct save_buffer_call_t {
    usearch_dense_save_buffer_t arguments;
    usearch_bytes_view_t monolith;
} save_buffer_call_t;

static PyObject* save_buffer_answer(usearch_py_call_t* call) {
    usearch_bytes_view_t const monolith = ((save_buffer_call_t const*)call->arguments)->monolith;
    Py_ssize_t const shape[1] = {(Py_ssize_t)monolith.length};
    return usearch_py_arena_buffer(&MonolithType, call->arena, monolith.data, "B", 1, shape, 1);
}

static PyObject* DenseIndex_save_buffer(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                        PyObject* args_names_tuple) {
    PyObject *arena_object, *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {{"arena", &arena_object},
                                               {"progress", &progress_object},
                                               {"progress_steps", &progress_steps_object}};
    usearch_py_receiver_t receiver;
    usearch_dense_index_t handle = NULL;
    size_t progress_steps = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("save_buffer", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    if (usearch_py_callable_from_object(progress_object, "progress", &call->holds[progress_slot_k]) != 0 ||
        usearch_py_call_hold_arena(call, arena_object) != 0)
        return usearch_py_call_discard(&receiver, call);
    save_buffer_call_t* const save = (save_buffer_call_t*)call->arguments;
    call->answer = save_buffer_answer;
    save->arguments = (usearch_dense_save_buffer_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .monolith = &save->monolith,
        .arena = usearch_py_arena_handle(call->arena),
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_save_buffer(&save->arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

/**
 *  @brief Reads what every load takes beside its source: the residency arm @p arm_name, the
 *      @p executor, @p values_per_key, @p progress and @p progress_steps, the source itself written
 *      to @p source.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int load_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                      PyObject* args_names_tuple, char const* source_name, PyObject** source, char const* arm_name,
                      PyObject** arm, usearch_py_enum_t const* ceiling_classes, int* ceiling,
                      usearch_executor_t* executor, size_t* values_per_key, PyObject** progress,
                      size_t* progress_steps) {
    PyObject *executor_object, *values_per_key_object, *progress_object, *progress_steps_object, *ceiling_object;
    usearch_py_argument_t const arguments[] = {
        {source_name, source},
        {arm_name, arm},
        {"executor", &executor_object},
        {"values_per_key", &values_per_key_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {ceiling_name(ceiling_classes), &ceiling_object},
    };
    if (usearch_py_arguments_parse(verb, args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        ceiling_parse(ceiling_object, ceiling_classes, ceiling) != 0 ||
        usearch_py_optional_executor_from_object(executor_object, executor) != 0 ||
        usearch_py_size_from_object(values_per_key_object, "values_per_key", values_per_key) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", progress_steps) != 0)
        return -1;
    return usearch_py_callable_from_object(progress_object, "progress", progress);
}

static char const doc_DenseIndex_load_ram[] =                                                       //
    "``load_ram(path, /, *, memory_domains=None, executor=None, values_per_key=0, progress=None,\n" //
    "progress_steps=0, cpu_class_ceiling=None) -> DenseIndex``\n"                                   //
    "\n"                                                                                            //
    "Copies a checkpoint into a new, mutable index in host memory, its geometry read from the\n"    //
    "checkpoint's own headers. Name the `.safegraphs.json` manifest, never one of the parts\n"      //
    "it enumerates; part `p` lands whole on shard `p` modulo the shard count.\n"                    //
    "Calls `usearch_dense_load_ram`.\n"                                                             //
    "\n"                                                                                            //
    "Args:\n"                                                                                       //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                     //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place the parts on;\n"            //
    "    None derives every node.\n"                                                                //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n"   //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.\n"         //
    "  executor (Executor, optional): The workers the read fans out over, one shard per\n"          //
    "    thread; None reads on the calling thread.\n"                                               //
    "  values_per_key (int, optional): Entries sharing one key, which no checkpoint\n"              //
    "    records; 0 is unbounded.\n"                                                                //
    "  progress (callable, optional): Called with (completed, total) shards; returning\n"           //
    "    False stops the read and raises `CancelledError`, creating no index.\n"                    //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject* DenseIndex_load_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    PyObject *path_object, *memory_domains_object, *decoded = NULL;
    usearch_outcome_t outcome = {0};
    PyObject* progress = NULL;
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    usearch_bytes_view_t path;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    PyObject* result = NULL;
    int cpu_class_ceiling = 0;
    if (load_parse("load_ram", args, positional_args_count, args_names_tuple, "path", &path_object, "memory_domains",
                   &memory_domains_object, &usearch_py_cpu_class, &cpu_class_ceiling, &executor, &values_per_key,
                   &progress, &progress_steps) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        goto release;

    usearch_dense_index_t handle = NULL;
    usearch_dense_load_ram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_ram(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(memory_domains);
    Py_XDECREF(decoded);
    Py_XDECREF(progress);
    return result;
}

static char const doc_DenseIndex_load_ram_buffer[] =                                              //
    "``load_ram_buffer(buffer, /, *, memory_domains=None, executor=None, values_per_key=0,\n"     //
    "progress=None, progress_steps=0, cpu_class_ceiling=None) -> DenseIndex``\n"                  //
    "\n"                                                                                          //
    "Copies a monolith out of a buffer into a new, mutable index in host memory, as\n"            //
    "`load_ram` does from a path; the monolith lands whole on the first memory domain.\n"         //
    "Calls `usearch_dense_load_ram_buffer`.\n"                                                    //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  buffer (buffer): One serialized monolith, such as `save_buffer` returns.\n"                //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place shards on.\n"             //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n" //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.\n"       //
    "  executor (Executor, optional): The workers the copy fans out over; None copies on\n"       //
    "    the calling thread.\n"                                                                   //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                //
    "  progress (callable, optional): Called with (completed, total) bytes; returning\n"          //
    "    False stops the read and raises `CancelledError`, creating no index.\n"                  //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject* DenseIndex_load_ram_buffer(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                            PyObject* args_names_tuple) {
    PyObject *buffer_object, *memory_domains_object;
    usearch_outcome_t outcome = {0};
    PyObject* progress = NULL;
    usearch_py_buffer_backing_t backing = {0};
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    PyObject* result = NULL;
    int cpu_class_ceiling = 0;
    if (load_parse("load_ram_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                   "memory_domains", &memory_domains_object, &usearch_py_cpu_class, &cpu_class_ceiling, &executor,
                   &values_per_key, &progress, &progress_steps) != 0 ||
        checkpoint_view(buffer_object, &backing) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        goto release;

    usearch_dense_index_t handle = NULL;
    usearch_dense_load_ram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_ram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(memory_domains);
    usearch_py_buffer_release(&backing);
    Py_XDECREF(progress);
    return result;
}

static char const doc_DenseIndex_load_vram[] =                                                     //
    "``load_vram(path, /, *, devices=None, executor=None, values_per_key=0, progress=None,\n"      //
    "progress_steps=0, gpu_class_ceiling=None) -> DenseIndex``\n"                                  //
    "\n"                                                                                           //
    "Copies a checkpoint into a new, mutable index on CUDA devices, with the placement rules\n"    //
    "of `load_ram`: part `p` lands whole on device `p` modulo the device count.\n"                 //
    "Calls `usearch_dense_load_vram`.\n"                                                           //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                    //
    "  devices (Sequence[int], optional): CUDA ordinals to place the parts on.\n"                  //
    "  gpu_class_ceiling (GPUClass, optional): The newest class dispatch may pick, capped below\n" //
    "    the best class every device loads, else unsupported; None means no ceiling.\n"            //
    "  executor (Executor, optional): The streams the upload runs on, one per device; None\n"      //
    "    uploads from the calling thread.\n"                                                       //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                 //
    "  progress (callable, optional): Called with (completed, total) shards; returning\n"          //
    "    False stops the read and raises `CancelledError`, creating no index.\n"                   //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject* DenseIndex_load_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    PyObject *path_object, *devices_object, *decoded = NULL;
    usearch_outcome_t outcome = {0};
    PyObject* progress = NULL;
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    usearch_bytes_view_t path;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    PyObject* result = NULL;
    int gpu_class_ceiling = 0;
    if (load_parse("load_vram", args, positional_args_count, args_names_tuple, "path", &path_object, "devices",
                   &devices_object, &usearch_py_gpu_class, &gpu_class_ceiling, &executor, &values_per_key, &progress,
                   &progress_steps) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        goto release;

    usearch_dense_index_t handle = NULL;
    usearch_dense_load_vram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_vram(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(devices);
    Py_XDECREF(decoded);
    Py_XDECREF(progress);
    return result;
}

static char const doc_DenseIndex_load_vram_buffer[] =                                              //
    "``load_vram_buffer(buffer, /, *, devices=None, executor=None, values_per_key=0,\n"            //
    "progress=None, progress_steps=0, gpu_class_ceiling=None) -> DenseIndex``\n"                   //
    "\n"                                                                                           //
    "Copies a monolith out of a host or device buffer into a new, mutable index on CUDA\n"         //
    "devices; the monolith lands whole on the first device.\n"                                     //
    "Calls `usearch_dense_load_vram_buffer`.\n"                                                    //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  buffer (buffer): One serialized monolith.\n"                                                //
    "  devices (Sequence[int], optional): CUDA ordinals to place shards on.\n"                     //
    "  gpu_class_ceiling (GPUClass, optional): The newest class dispatch may pick, capped below\n" //
    "    the best class every device loads, else unsupported; None means no ceiling.\n"            //
    "  executor (Executor, optional): The streams the upload runs on, one per device; None\n"      //
    "    uploads from the calling thread.\n"                                                       //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                 //
    "  progress (callable, optional): Called with (completed, total) bytes; returning\n"           //
    "    False stops the read and raises `CancelledError`, creating no index.\n"                   //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject* DenseIndex_load_vram_buffer(PyTypeObject* type, PyObject* const* args,
                                             Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    PyObject *buffer_object, *devices_object;
    usearch_outcome_t outcome = {0};
    PyObject* progress = NULL;
    usearch_py_buffer_backing_t backing = {0};
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    PyObject* result = NULL;
    int gpu_class_ceiling = 0;
    if (load_parse("load_vram_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                   "devices", &devices_object, &usearch_py_gpu_class, &gpu_class_ceiling, &executor, &values_per_key,
                   &progress, &progress_steps) != 0 ||
        checkpoint_view(buffer_object, &backing) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        goto release;

    usearch_dense_index_t handle = NULL;
    usearch_dense_load_vram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_vram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(devices);
    usearch_py_buffer_release(&backing);
    Py_XDECREF(progress);
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

static char const doc_DenseIndex_view_nvme[] =                                                    //
    "``view_nvme(path, /, *, paths=None, executor=None, cpu_class_ceiling=None,\n"                //
    "key_table_domains=None) -> DenseIndex``\n"                                                   //
    "\n"                                                                                          //
    "Serves a checkpoint in place from NVMe drives as a read-only index. External-memory\n"       //
    "graphs are never modified in place, so bigger ones come from merging shards built in\n"      //
    "RAM.\n"                                                                                      //
    "Calls `usearch_dense_view_nvme`.\n"                                                          //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                   //
    "  paths (Sequence[str | bytes | PathLike], optional): Devices or directories to\n"           //
    "    serve the parts from; None serves each where its manifest points. A path may\n"          //
    "    name a kernel device, a transport identifier, or a FUSE mount.\n"                        //
    "  executor (Executor, optional): The NVMe executor whose rings read the headers;\n"          //
    "    None reads them on the calling thread.\n"                                                //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n" //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.\n"       //
    "  key_table_domains (Sequence[int], optional): NUMA node IDs of a key table held in\n"       //
    "    RAM, which `count`, `get`, `sample`, `maxsim` and hinted searches read; an empty\n"      //
    "    sequence derives them, and None holds none.";

static PyObject* DenseIndex_view_nvme(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    PyObject *path_object, *paths_object, *executor_object, *cpu_class_ceiling_object, *key_table_object,
        *decoded = NULL;
    usearch_py_argument_t const arguments[] = {
        {"path", &path_object},
        {"paths", &paths_object},
        {"executor", &executor_object},
        {"cpu_class_ceiling", &cpu_class_ceiling_object},
        {"key_table_domains", &key_table_object},
    };
    usearch_bytes_view_t path;
    usearch_py_paths_t paths = {0};
    usearch_executor_t executor = NULL;
    int cpu_class_ceiling = 0;
    int32_t* key_table_domains = NULL;
    size_t key_table_domains_count = 0;
    PyObject* result = NULL;
    if (usearch_py_arguments_parse("view_nvme", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_optional_executor_from_object(executor_object, &executor) != 0 ||
        ceiling_parse(cpu_class_ceiling_object, &usearch_py_cpu_class, &cpu_class_ceiling) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0 ||
        usearch_py_paths_from_object(paths_object, &paths) != 0 ||
        usearch_py_ordinals_from_object(key_table_object, "key_table_domains", &key_table_domains,
                                        &key_table_domains_count) != 0)
        goto release;

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_view_nvme_t view_arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .nvme = {paths.views, paths.count, (usearch_cpu_class_t)cpu_class_ceiling},
    };
    usearch_ram_options_t const key_table = {key_table_domains, key_table_domains_count, 0,
                                             (usearch_cpu_class_t)cpu_class_ceiling};
    if (key_table_object && key_table_object != Py_None) view_arguments.ram = &key_table;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_view_nvme(&view_arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(key_table_domains);
    usearch_py_paths_release(&paths);
    Py_XDECREF(decoded);
    return result;
}

static char const doc_DenseIndex_load_nvme[] =                                                    //
    "``load_nvme(path, /, *, paths=None, executor=None, cpu_class_ceiling=None,\n"                //
    "key_table_domains=None) -> DenseIndex``\n"                                                   //
    "\n"                                                                                          //
    "Opens a checkpoint's manifest from NVMe drives as a writable index: it answers every\n"      //
    "read a view answers, and `refine` rewrites its parts one at a time, replacing the\n"         //
    "manifest durably as each commits. A lone monolith has no manifest to replace and\n"          //
    "raises `UnsupportedError`.\n"                                                                //
    "Calls `usearch_dense_load_nvme`.\n"                                                          //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  path (str | bytes | os.PathLike): The manifest.\n"                                         //
    "  paths (Sequence[str | bytes | PathLike], optional): Must name none, since the\n"           //
    "    parts are served where the manifest points.\n"                                           //
    "  executor (Executor, optional): The NVMe executor whose rings read the headers;\n"          //
    "    None reads them on the calling thread.\n"                                                //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n" //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.\n"       //
    "  key_table_domains (Sequence[int], optional): NUMA node IDs of a key table held in\n"       //
    "    RAM, as `view_nvme` takes them.";

static PyObject* DenseIndex_load_nvme(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    PyObject *path_object, *paths_object, *executor_object, *cpu_class_ceiling_object, *key_table_object,
        *decoded = NULL;
    usearch_py_argument_t const arguments[] = {
        {"path", &path_object},
        {"paths", &paths_object},
        {"executor", &executor_object},
        {"cpu_class_ceiling", &cpu_class_ceiling_object},
        {"key_table_domains", &key_table_object},
    };
    usearch_bytes_view_t path;
    usearch_py_paths_t paths = {0};
    usearch_executor_t executor = NULL;
    int cpu_class_ceiling = 0;
    int32_t* key_table_domains = NULL;
    size_t key_table_domains_count = 0;
    PyObject* result = NULL;
    if (usearch_py_arguments_parse("load_nvme", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_optional_executor_from_object(executor_object, &executor) != 0 ||
        ceiling_parse(cpu_class_ceiling_object, &usearch_py_cpu_class, &cpu_class_ceiling) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0 ||
        usearch_py_paths_from_object(paths_object, &paths) != 0 ||
        usearch_py_ordinals_from_object(key_table_object, "key_table_domains", &key_table_domains,
                                        &key_table_domains_count) != 0)
        goto release;

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_dense_load_nvme_t load_arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .nvme = {paths.views, paths.count, (usearch_cpu_class_t)cpu_class_ceiling},
    };
    usearch_ram_options_t const key_table = {key_table_domains, key_table_domains_count, 0,
                                             (usearch_cpu_class_t)cpu_class_ceiling};
    if (key_table_object && key_table_object != Py_None) load_arguments.ram = &key_table;
    Py_BEGIN_ALLOW_THREADS;
    usearch_dense_load_nvme(&load_arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(key_table_domains);
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
    return usearch_py_optional_executor_from_object(executor_object, executor);
}

static char const doc_DenseIndex_metadata_read[] =                                             //
    "``metadata_read(path, /, *, executor=None) -> DenseMetadata``\n"                          //
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
    "``metadata_read_buffer(buffer, /, *, executor=None) -> DenseMetadata``\n"           //
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
        checkpoint_view(buffer_object, &backing) != 0)
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
    {"load_nvme", (PyCFunction)DenseIndex_load_nvme, usearch_py_method_flags_k | METH_CLASS, doc_DenseIndex_load_nvme},
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
    {"reshard", (PyCFunction)DenseIndex_reshard, usearch_py_method_flags_k, doc_DenseIndex_reshard},
    {"refine_plan_finish", (PyCFunction)DenseIndex_refine_plan_finish, METH_O, doc_DenseIndex_refine_plan_finish},
    {"reshard_plan_finish", (PyCFunction)DenseIndex_reshard_plan_finish, METH_O, doc_DenseIndex_reshard_plan_finish},
    {"sample", (PyCFunction)DenseIndex_sample, usearch_py_method_flags_k, doc_DenseIndex_sample},
    {"save", (PyCFunction)DenseIndex_save, usearch_py_method_flags_k, doc_DenseIndex_save},
    {"save_buffer", (PyCFunction)DenseIndex_save_buffer, usearch_py_method_flags_k, doc_DenseIndex_save_buffer},
    {NULL, NULL, 0, NULL},
};

static PyMethodDef DenseIndex_on_methods[] = {
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
    {"reshard", (PyCFunction)DenseIndex_reshard, usearch_py_method_flags_k, doc_DenseIndex_reshard},
    {"refine_plan_finish", (PyCFunction)DenseIndex_refine_plan_finish, METH_O, doc_DenseIndex_refine_plan_finish},
    {"reshard_plan_finish", (PyCFunction)DenseIndex_reshard_plan_finish, METH_O, doc_DenseIndex_reshard_plan_finish},
    {"sample", (PyCFunction)DenseIndex_sample, usearch_py_method_flags_k, doc_DenseIndex_sample},
    {"save", (PyCFunction)DenseIndex_save, usearch_py_method_flags_k, doc_DenseIndex_save},
    {"save_buffer", (PyCFunction)DenseIndex_save_buffer, usearch_py_method_flags_k, doc_DenseIndex_save_buffer},
    {NULL, NULL, 0, NULL},
};

static PyMethodDef DenseIndex_deferred_on_methods[] = {
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
    {"reshard", (PyCFunction)DenseIndex_reshard, usearch_py_method_flags_k, doc_DenseIndex_reshard},
    {"refine_plan_finish", (PyCFunction)DenseIndex_refine_plan_finish, METH_O, doc_DenseIndex_refine_plan_finish},
    {"reshard_plan_finish", (PyCFunction)DenseIndex_reshard_plan_finish, METH_O, doc_DenseIndex_reshard_plan_finish},
    {"sample", (PyCFunction)DenseIndex_sample, usearch_py_method_flags_k, doc_DenseIndex_sample},
    {"save", (PyCFunction)DenseIndex_save, usearch_py_method_flags_k, doc_DenseIndex_save},
    {"save_buffer", (PyCFunction)DenseIndex_save_buffer, usearch_py_method_flags_k, doc_DenseIndex_save_buffer},
    {NULL, NULL, 0, NULL},
};

usearch_py_receiver_verbs_t const usearch_py_dense_receiver_verbs = {&DenseIndexType, DenseIndex_on_methods,
                                                                     DenseIndex_deferred_on_methods};

static PyGetSetDef DenseIndex_getset[] = {
    {"capacity", (getter)DenseIndex_get_counter, NULL, doc_DenseIndex_capacity,
     (void*)offsetof(usearch_dense_index_description_t, capacity)},
    {"dimensions", (getter)DenseIndex_get_counter, NULL, doc_DenseIndex_dimensions,
     (void*)offsetof(usearch_dense_index_description_t, metadata.dimensions)},
    {"connectivity", (getter)DenseIndex_get_counter, NULL, doc_DenseIndex_connectivity,
     (void*)offsetof(usearch_dense_index_description_t, metadata.connectivity)},
    {"memory_usage", (getter)DenseIndex_get_counter, NULL, doc_DenseIndex_memory_usage,
     (void*)offsetof(usearch_dense_index_description_t, memory_usage)},
    {"cpu_class", (getter)DenseIndex_get_cpu_class, NULL, doc_DenseIndex_cpu_class, NULL},
    {"gpu_class", (getter)DenseIndex_get_gpu_class, NULL, doc_DenseIndex_gpu_class, NULL},
    {"shards_count", (getter)DenseIndex_get_counter, NULL, doc_DenseIndex_shards_count,
     (void*)offsetof(usearch_dense_index_description_t, shards_count)},
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
