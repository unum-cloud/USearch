/**
 *  @file python/geospatial_index.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief CPython similarity search over observations in space and time.
 *
 *  Every entry is one observation of eleven doubles in natural units, so a batch is a float64
 *  matrix of eleven columns read in place, NaN wherever a device recorded nothing. The handle
 *  holds no metric or quantization, and the surface adds a containment walk, @c within, beside
 *  the ranked search.
 *
 *  Every verb is written once and reached through three receivers, as in @c dense_index.c: the
 *  index runs it on the calling thread, an @c On blocks on its executor, and a @c DeferredOn
 *  submits it there and answers a @c Deferred. A verb fills one call record and opens its result
 *  as the slot the library writes into, so the same body serves every completion.
 */
#include "lib.h"

#pragma region Handles

/** An owned index handle. */
typedef struct GeospatialIndex {
    PyObject ob_base;
    usearch_geospatial_index_t handle;
} GeospatialIndex;

/** A live index's frozen metadata and current counters, the slot @c describe writes them into. */
typedef struct GeospatialIndexDescription {
    PyObject ob_base;
    usearch_geospatial_index_description_t description;
} GeospatialIndexDescription;

/** A refine or a reshard in flight, continued by every call naming it until its finish verb. */
typedef struct GeospatialPlan {
    PyObject ob_base;

    /** The engine plan, NULL until a call opens it and once its finish verb closed it. */
    union {
        usearch_geospatial_refine_plan_t refine;
        usearch_geospatial_reshard_plan_t reshard;
    } handle;

    /** Where the last call naming the plan stopped. */
    usearch_phase_progress_t progress;
} GeospatialPlan;

/** The holds every geospatial call keeps in the same slots. */
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
static int receiver_from_object(PyObject* self, usearch_py_receiver_t* receiver, usearch_geospatial_index_t* handle) {
    if (usearch_py_receiver_from_object(self, receiver) != 0) return -1;
    if (!Py_IS_TYPE(receiver->index, &GeospatialIndexType)) {
        PyErr_Format(PyExc_TypeError, "expected a receiver over a GeospatialIndex, got one over %S",
                     Py_TYPE(receiver->index));
        return -1;
    }
    *handle = ((GeospatialIndex*)receiver->index)->handle;
    return 0;
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

/** Fields per observation, as @c usearch_geospatial_observation_t lays them out. */
static size_t const observation_fields = sizeof(usearch_geospatial_observation_t) / sizeof(double);

/**
 *  @brief Views a float64 matrix of observations in place; one observation is a one-row matrix.
 *
 *  The library reads an array of observation structures, so the rows must be packed as the
 *  structure lays them out.
 *
 *  @return The observations, or NULL with an exception set.
 */
static usearch_geospatial_observation_t const* observations_from_object(PyObject* object, size_t* count,
                                                                        usearch_py_buffer_backing_t* backing) {
    usearch_matrix_view_t matrix;
    if (usearch_py_matrix_from_object(object, usearch_scalar_f64_k, &matrix, backing) != 0) return NULL;
    if (matrix.columns != observation_fields ||
        (matrix.rows > 1 && matrix.stride != sizeof(usearch_geospatial_observation_t))) {
        usearch_py_buffer_release(backing);
        usearch_py_raise(usearch_status_shape_mismatch_k,
                         "observations are packed rows of %zu float64 fields, got %zu columns %zu bytes apart",
                         observation_fields, matrix.columns, matrix.stride);
        return NULL;
    }
    *count = matrix.rows;
    return (usearch_geospatial_observation_t const*)matrix.data;
}

/** The radius and time slack every relation narrows by, read from @p metres_object and
 *  @p seconds_object. */
static int tolerance_parse(PyObject* metres_object, PyObject* seconds_object,
                           usearch_geospatial_tolerance_t* tolerance) {
    tolerance->metres = metres_object ? PyFloat_AsDouble(metres_object) : 0.0;
    if (tolerance->metres == -1.0 && PyErr_Occurred()) return -1;
    tolerance->seconds = seconds_object ? PyFloat_AsDouble(seconds_object) : 0.0;
    if (tolerance->seconds == -1.0 && PyErr_Occurred()) return -1;
    return 0;
}

#pragma endregion Handles

#pragma region Construction

static char const doc_GeospatialIndex[] =                                                          //
    "GeospatialIndex\n"                                                                            //
    "\n"                                                                                           //
    "Similarity search over observations on and above the Earth: where a device was, how sure\n"   //
    "it was, which way it looked, how far it saw, and when, under nearness, line of sight, and\n"  //
    "view-cone relations. Every score is the straight-line chord between two positions, in\n"      //
    "metres.\n"                                                                                    //
    "\n"                                                                                           //
    "Every verb is a batch of observations, an `(N, 11)` float64 matrix whose columns are\n"       //
    "latitude and longitude in degrees, altitude and accuracy in metres, heading, pitch,\n"        //
    "heading accuracy and field of view in degrees, range in metres, and the begin and end of\n"   //
    "the interval in seconds, NaN wherever nothing was recorded.\n"                                //
    "\n"                                                                                           //
    "Start an empty one in the residency its entries live in, `init_ram` or `init_vram`, copy a\n" //
    "checkpoint in with `load_ram` or `load_vram`, or serve one in place with `view_ram` or\n"     //
    "`view_nvme`, which are read-only.\n"                                                          //
    "\n"                                                                                           //
    "The receiver chooses where a verb runs, and the verb stays the same: the index runs it\n"     //
    "on the calling thread, `on(executor)` blocks on that executor, and `deferred(executor)`\n"    //
    "submits it there and answers a `Deferred`. See `usearch.Executor`.";

/** The geometry both creation verbs freeze, read before the factory's residency arm. */
typedef struct geospatial_geometry_t {
    size_t connectivity;
    size_t values_per_key;
} geospatial_geometry_t;

/** The keyword a verb names its class ceiling by: @c cpu_class_ceiling for host placements,
 *  @c gpu_class_ceiling for devices. */
static char const* ceiling_name(usearch_py_enum_t const* ceiling_classes) {
    return ceiling_classes == &usearch_py_gpu_class ? "gpu_class_ceiling" : "cpu_class_ceiling";
}

/**
 *  @brief Reads the geometry both creation verbs take, every setting by name, and the residency arm
 *      @p arm_name into @p arm.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int geometry_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                          PyObject* args_names_tuple, char const* arm_name, PyObject** arm,
                          usearch_py_enum_t const* ceiling_classes, int* ceiling, geospatial_geometry_t* geometry) {
    PyObject *connectivity_object, *values_per_key_object, *ceiling_object;
    usearch_py_argument_t const arguments[] = {
        {"connectivity", &connectivity_object},
        {"values_per_key", &values_per_key_object},
        {arm_name, arm},
        {ceiling_name(ceiling_classes), &ceiling_object},
    };
    memset(geometry, 0, sizeof(*geometry));
    if (usearch_py_arguments_parse(verb, args, positional_args_count, args_names_tuple, 0, 0, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_size_from_object(connectivity_object, "connectivity", &geometry->connectivity) != 0 ||
        usearch_py_size_from_object(values_per_key_object, "values_per_key", &geometry->values_per_key) != 0 ||
        usearch_py_class_from_object(ceiling_object, ceiling_name(ceiling_classes), ceiling_classes, ceiling) != 0)
        return -1;
    return 0;
}

/**
 *  @brief Wraps the handle a lifecycle verb produced, or frees it when it failed or the wrapper
 *      cannot be allocated.
 *  @return A new reference, or NULL with an exception set.
 */
static PyObject* index_adopt(PyTypeObject* type, usearch_geospatial_index_t handle, usearch_outcome_t const* outcome) {
    if (usearch_py_raise_on_outcome(outcome) != 0) {
        usearch_geospatial_free(handle);
        return NULL;
    }
    GeospatialIndex* const instance = (GeospatialIndex*)type->tp_alloc(type, 0);
    if (!instance) {
        usearch_geospatial_free(handle);
        return NULL;
    }
    instance->handle = handle;
    return (PyObject*)instance;
}

static PyObject* GeospatialIndex_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_TypeError, "GeospatialIndex is built through init_*(), load_*(), view_ram() or view_nvme()");
    return NULL;
}

static char const doc_GeospatialIndex_init_ram[] =                                                   //
    "``init_ram(*, connectivity=0, values_per_key=0, memory_domains=None, cpu_class_ceiling=None)\n" //
    "-> GeospatialIndex``\n"                                                                         //
    "\n"                                                                                             //
    "Starts an empty, mutable graph in host memory, one shard per named NUMA node.\n"                //
    "Calls `usearch_geospatial_init_ram`.\n"                                                         //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  connectivity (int, optional): Connections per node; 0 derives twelve, eight\n"                //
    "    tangent-plane cones and four long-range links.\n"                                           //
    "  values_per_key (int, optional): Observations sharing one key; 0 is unbounded.\n"              //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to shard across; None derives\n"      //
    "    every node.\n"                                                                              //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n"    //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.";

static PyObject* GeospatialIndex_init_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                          PyObject* args_names_tuple) {
    geospatial_geometry_t geometry;
    PyObject* memory_domains_object;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    int cpu_class_ceiling = 0;
    if (geometry_parse("init_ram", args, positional_args_count, args_names_tuple, "memory_domains",
                       &memory_domains_object, &usearch_py_cpu_class, &cpu_class_ceiling, &geometry) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        return NULL;

    usearch_geospatial_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_geospatial_init_ram_t arguments = {
        .index = &handle,
        .outcome = &outcome,
        .connectivity = geometry.connectivity,
        .values_per_key = geometry.values_per_key,
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_init_ram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);
    return index_adopt(type, handle, &outcome);
}

static char const doc_GeospatialIndex_init_vram[] =                                                //
    "``init_vram(*, connectivity=0, values_per_key=0, devices=None, gpu_class_ceiling=None)\n"     //
    "-> GeospatialIndex``\n"                                                                       //
    "\n"                                                                                           //
    "Starts an empty, mutable graph on GPU devices, one shard per named ordinal.\n"                //
    "Calls `usearch_geospatial_init_vram`.\n"                                                      //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  connectivity (int, optional): Connections per node; 0 derives twelve.\n"                    //
    "  values_per_key (int, optional): Observations sharing one key; 0 is unbounded.\n"            //
    "  devices (Sequence[int], optional): GPU ordinals to shard across.\n"                         //
    "  gpu_class_ceiling (GPUClass, optional): The newest class dispatch may pick, capped below\n" //
    "    the best class every device loads, else unsupported; None means no ceiling.";

static PyObject* GeospatialIndex_init_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                           PyObject* args_names_tuple) {
    geospatial_geometry_t geometry;
    PyObject* devices_object;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    int gpu_class_ceiling = 0;
    if (geometry_parse("init_vram", args, positional_args_count, args_names_tuple, "devices", &devices_object,
                       &usearch_py_gpu_class, &gpu_class_ceiling, &geometry) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        return NULL;

    usearch_geospatial_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_geospatial_init_vram_t arguments = {
        .index = &handle,
        .outcome = &outcome,
        .connectivity = geometry.connectivity,
        .values_per_key = geometry.values_per_key,
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_init_vram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(devices);
    return index_adopt(type, handle, &outcome);
}

static void GeospatialIndex_dealloc(GeospatialIndex* self) {
    // Releasing a large graph walks every shard, which no other thread should wait on.
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_free(self->handle);
    Py_END_ALLOW_THREADS;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static char const doc_GeospatialIndex_on[] =                                                   //
    "on(executor) -> On[GeospatialIndex]\n"                                                    //
    "\n"                                                                                       //
    "A receiver whose verbs block on `executor`, refused while another call holds it.\n"       //
    "It offers every verb the index does, with the same arguments and answers. Python-only.\n" //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  executor (Executor): The workers the calls run on; its residency must match.";

static char const doc_GeospatialIndex_deferred[] =                                              //
    "deferred(executor) -> DeferredOn[GeospatialIndex]\n"                                       //
    "\n"                                                                                        //
    "A receiver whose verbs submit their call to `executor` and answer a `Deferred` at once,\n" //
    "with the same arguments the index takes. It holds the index and the executor until\n"      //
    "every `Deferred` it answered is gone. Python-only.\n"                                      //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  executor (Executor): The workers the calls run on; its residency must match.";

static PyObject* GeospatialIndex_on(PyObject* self, PyObject* executor) {
    return usearch_py_on_new(&OnType, self, executor);
}

static PyObject* GeospatialIndex_deferred(PyObject* self, PyObject* executor) {
    return usearch_py_on_new(&DeferredOnType, self, executor);
}

#pragma endregion Construction

#pragma region Reports

static PyStructSequence_Field GeospatialMetadata_fields[] = {
    {"connectivity", "Connections per node the graph was built with."},
    {NULL, NULL},
};

PyStructSequence_Desc usearch_py_geospatial_metadata_desc = {
    "usearch.GeospatialMetadata", "The geometry a checkpoint froze, as `usearch_geospatial_metadata_t` holds it.",
    GeospatialMetadata_fields, 1};

PyTypeObject GeospatialMetadataType;

/** A new @c GeospatialMetadata of @p metadata, or NULL with an exception set. */
static PyObject* metadata_to_object(usearch_geospatial_metadata_t const* metadata) {
    PyObject* fields[1] = {PyLong_FromSize_t(metadata->connectivity)};
    return usearch_py_struct_sequence_steal(&GeospatialMetadataType, fields, 1);
}

static char const doc_GeospatialIndexDescription[] =                                        //
    "GeospatialIndexDescription\n"                                                          //
    "\n"                                                                                    //
    "A live index's frozen metadata and current counters, as `describe` read them in one\n" //
    "call.";

static PyObject* GeospatialIndexDescription_get_metadata(GeospatialIndexDescription* self, void* closure) {
    return metadata_to_object(&self->description.metadata);
}

static PyObject* GeospatialIndexDescription_get_cpu_class(GeospatialIndexDescription* self, void* closure) {
    return usearch_py_class_to_object(&usearch_py_cpu_class, (int)self->description.cpu_class);
}

static PyObject* GeospatialIndexDescription_get_gpu_class(GeospatialIndexDescription* self, void* closure) {
    return usearch_py_class_to_object(&usearch_py_gpu_class, (int)self->description.gpu_class);
}

static PyObject* GeospatialIndexDescription_get_load_path(GeospatialIndexDescription* self, void* closure) {
    return PyUnicode_FromString(usearch_load_path_name(self->description.load_path));
}

static PyGetSetDef GeospatialIndexDescription_getset[] = {
    {"metadata", (getter)GeospatialIndexDescription_get_metadata, NULL,
     "The frozen `GeospatialMetadata`, as a checkpoint reports it.", NULL},
    {"size", usearch_py_size_getter, NULL, "Stored observations.",
     (void*)offsetof(GeospatialIndexDescription, description.size)},
    {"capacity", usearch_py_size_getter, NULL, "Observations hosted without growing again.",
     (void*)offsetof(GeospatialIndexDescription, description.capacity)},
    {"memory_usage", usearch_py_size_getter, NULL, "Bytes the handle holds; traversal scratch is the executor's.",
     (void*)offsetof(GeospatialIndexDescription, description.memory_usage)},
    {"cpu_class", (getter)GeospatialIndexDescription_get_cpu_class, NULL,
     "The `CPUClass` dispatch picked on this machine, or None.", NULL},
    {"gpu_class", (getter)GeospatialIndexDescription_get_gpu_class, NULL,
     "The `GPUClass` dispatch picked on the index's devices, or None for a host index.", NULL},
    {"shards_count", usearch_py_size_getter, NULL, "Live shards.",
     (void*)offsetof(GeospatialIndexDescription, description.shards_count)},
    {"load_path", (getter)GeospatialIndexDescription_get_load_path, NULL,
     "How the last load from a path read the parts onto the devices: none, host-copy or cufile.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject* GeospatialIndexDescription_repr(GeospatialIndexDescription* self) {
    return PyUnicode_FromFormat("GeospatialIndexDescription(size=%zu, capacity=%zu)", self->description.size,
                                self->description.capacity);
}

PyTypeObject GeospatialIndexDescriptionType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.GeospatialIndexDescription",
    .tp_doc = doc_GeospatialIndexDescription,
    .tp_basicsize = sizeof(GeospatialIndexDescription),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_repr = (reprfunc)GeospatialIndexDescription_repr,
    .tp_getset = GeospatialIndexDescription_getset,
};

#pragma endregion Reports

#pragma region Introspection

/**
 *  @brief Reads the frozen metric and current counters of @p handle on the calling thread.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int index_describe(usearch_geospatial_index_t handle, usearch_geospatial_index_description_t* description) {
    memset(description, 0, sizeof(*description));
    usearch_outcome_t outcome = {0};
    usearch_geospatial_describe_t arguments = {.index = handle, .outcome = &outcome, .description = description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_describe(&arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_raise_on_outcome(&outcome);
}

static char const doc_GeospatialIndex_describe[] =                                           //
    "describe() -> GeospatialIndexDescription\n"                                             //
    "\n"                                                                                     //
    "The frozen metadata and the current counters, read in one call. Each counter is read\n" //
    "once, so under concurrent insertions they may come from slightly different moments.\n"  //
    "Calls `usearch_geospatial_describe`.";

static PyObject* GeospatialIndex_describe(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    GeospatialIndexDescription* const description = usearch_py_call_result(call, &GeospatialIndexDescriptionType);
    if (!description) return usearch_py_call_discard(&receiver, call);
    usearch_geospatial_describe_t* const arguments = (usearch_geospatial_describe_t*)call->arguments;
    *arguments = (usearch_geospatial_describe_t) {.index = handle,
                                                  .executor = receiver.executor,
                                                  .outcome = &call->outcome,
                                                  .completion = receiver.completion,
                                                  .description = &description->description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_describe(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_GeospatialIndex_health[] =                                          //
    "health() -> DenseReachability\n"                                                     //
    "\n"                                                                                  //
    "Audits how much of the graph a search can reach, every shard swept from its first\n" //
    "entry, forward and back. Read-only.\n"                                               //
    "Calls `usearch_geospatial_health`.";

static PyObject* GeospatialIndex_health(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    DenseReachability* const audit = usearch_py_call_result(call, &DenseReachabilityType);
    if (!audit) return usearch_py_call_discard(&receiver, call);
    usearch_geospatial_health_t* const arguments = (usearch_geospatial_health_t*)call->arguments;
    *arguments = (usearch_geospatial_health_t) {.index = handle,
                                                .executor = receiver.executor,
                                                .outcome = &call->outcome,
                                                .completion = receiver.completion,
                                                .reachability = &audit->reachability};
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_health(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

/** Reads the @c size_t counter at the byte offset @p closure names inside a description. */
static PyObject* GeospatialIndex_get_counter(GeospatialIndex* self, void* closure) {
    usearch_geospatial_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return PyLong_FromSize_t(*(size_t const*)((char const*)&description + (size_t)closure));
}

static PyObject* GeospatialIndex_get_cpu_class(GeospatialIndex* self, void* closure) {
    usearch_geospatial_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return usearch_py_class_to_object(&usearch_py_cpu_class, (int)description.cpu_class);
}

static PyObject* GeospatialIndex_get_gpu_class(GeospatialIndex* self, void* closure) {
    usearch_geospatial_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return usearch_py_class_to_object(&usearch_py_gpu_class, (int)description.gpu_class);
}

static Py_ssize_t GeospatialIndex_length(GeospatialIndex* self) {
    usearch_geospatial_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return -1;
    return (Py_ssize_t)description.size;
}

static PyObject* GeospatialIndex_repr(GeospatialIndex* self) {
    // No exception is raised here: `tp_repr` runs inside `PyErr_Format("%R", ...)` chains, where
    // raising would destroy the caller's.
    usearch_geospatial_index_description_t description = {0};
    usearch_outcome_t outcome = {0};
    usearch_geospatial_describe_t arguments = {.index = self->handle, .outcome = &outcome, .description = &description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_describe(&arguments);
    Py_END_ALLOW_THREADS;
    if (outcome.status != usearch_status_success_k) return PyUnicode_FromString("GeospatialIndex(<unavailable>)");
    return PyUnicode_FromFormat("GeospatialIndex(size=%zu)", description.size);
}

#pragma endregion Introspection

#pragma region Ingest

static char const doc_GeospatialIndex_reserve[] =                   //
    "reserve(capacity) -> None\n"                                   //
    "\n"                                                            //
    "Grows the index to host at least `capacity` observations, a\n" //
    "hint, since insertions grow the index on demand.\n"            //
    "Calls `usearch_geospatial_reserve`.\n"                         //
    "\n"                                                            //
    "Args:\n"                                                       //
    "  capacity (int): Observations the index must accept without growing again.";

static PyObject* GeospatialIndex_reserve(PyObject* self, PyObject* capacity_object) {
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    size_t capacity = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(capacity_object, "capacity", &capacity) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    usearch_geospatial_reserve_t* const reserve = (usearch_geospatial_reserve_t*)call->arguments;
    *reserve = (usearch_geospatial_reserve_t) {.index = handle,
                                               .executor = receiver.executor,
                                               .outcome = &call->outcome,
                                               .completion = receiver.completion,
                                               .entries = capacity};
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_reserve(reserve);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_GeospatialIndex_clear[] =                          //
    "clear() -> None\n"                                                  //
    "\n"                                                                 //
    "Drops every entry, keeping the capacity and the frozen geometry.\n" //
    "Calls `usearch_geospatial_clear`.";

static PyObject* GeospatialIndex_clear(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    usearch_geospatial_clear_t* const arguments = (usearch_geospatial_clear_t*)call->arguments;
    *arguments = (usearch_geospatial_clear_t) {.index = handle,
                                               .executor = receiver.executor,
                                               .outcome = &call->outcome,
                                               .completion = receiver.completion};
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_clear(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_GeospatialIndex_insert[] =                                                    //
    "``insert(keys, observations, /, *, expansion_factor=0, progress=None, progress_steps=0,\n"     //
    "stats=None) -> None``\n"                                                                       //
    "\n"                                                                                            //
    "Inserts a batch of observations under a batch of keys; one observation is a one-row matrix.\n" //
    "Keys need not be unique, so a camera's frames may share its key, and a batch is\n"             //
    "all-or-nothing. An observation without an altitude is placed on the ellipsoid.\n"              //
    "Calls `usearch_geospatial_insert`.\n"                                                          //
    "\n"                                                                                            //
    "Args:\n"                                                                                       //
    "  keys (Sequence[Key | int] | buffer): One key per observation, or an N by 16 buffer.\n"       //
    "  observations (buffer): An `(N, 11)` float64 matrix of packed rows, read in place.\n"         //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"                //
    "  progress (callable, optional): Called with (completed, total) entries; what it raises\n"     //
    "    goes to `sys.unraisablehook`.\n"                                                           //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"         //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"            //
    "\n"                                                                                            //
    "Example:\n"                                                                                    //
    "  >>> from array import array\n"                                                               //
    "  >>> nan = float('nan')\n"                                                                    //
    "  >>> paris = array('d', [48.8566, 2.3522] + [nan] * 9)\n"                                     //
    "  >>> index = usearch.GeospatialIndex.init_ram()\n"                                            //
    "  >>> index.insert([7], paris)\n"                                                              //
    "  >>> len(index)\n"                                                                            //
    "  1";

static PyObject* GeospatialIndex_insert(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                        PyObject* args_names_tuple) {
    PyObject *keys_object, *observations_object, *expansion_object, *progress_object, *progress_steps_object,
        *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"keys", &keys_object},
        {"observations", &observations_object},
        {"expansion_factor", &expansion_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    size_t expansion_factor = 0, progress_steps = 0, count = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("insert", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    usearch_geospatial_observation_t const* const observations = observations_from_object(observations_object, &count,
                                                                                          &call->buffers[0]);
    if (!observations || usearch_py_keys_from_object(keys_object, count, &call->keys[0]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0)
        return usearch_py_call_discard(&receiver, call);

    usearch_geospatial_insert_t* const insert = (usearch_geospatial_insert_t*)call->arguments;
    *insert = (usearch_geospatial_insert_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .count = count,
        .keys = {call->keys[0].keys, sizeof(usearch_key_t)},
        .observations = observations,
        .expansion_factor = expansion_factor,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_insert(insert);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_GeospatialIndex_refine[] =                                                  //
    "``refine(*, expansion_factor=0, memory_limit=0, progress=None, progress_steps=0,\n"          //
    "stats=None, plan=None) -> RefineReport``\n"                                                  //
    "\n"                                                                                          //
    "Re-selects the neighbors of every stored entry against the finished graph, improving\n"      //
    "edges chosen while the graph around them was still incomplete. A cheap insertion pass\n"     //
    "followed by one refinement reaches a better graph than one expensive insertion pass, and\n"  //
    "a caller loops until the report's `verdict` is no longer `RefineVerdict.IMPROVED`.\n"        //
    "\n"                                                                                          //
    "Stopping is safe: a stopped sweep still re-links its islands and raises `CancelledError`.\n" //
    "With a `plan`, a stop leaves it live and the next call naming it continues the round;\n"     //
    "`refine_plan_finish(plan)` closes it.\n"                                                     //
    "Calls `usearch_geospatial_refine`.\n"                                                        //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"              //
    "  memory_limit (int, optional): Bytes a loaded NVMe handle holds for the part it\n"          //
    "    rewrites; 0 derives what rounds that part in RAM.\n"                                     //
    "  progress (callable, optional): Called with (completed, total) units of each phase;\n"      //
    "    returning False stops the round, None continues.\n"                                      //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"       //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"          //
    "  plan (GeospatialRefinePlan, optional): The round to open or continue; None runs one\n"     //
    "    round to its end.\n"                                                                     //
    "\n"                                                                                          //
    "Returns:\n"                                                                                  //
    "  RefineReport: What the round did to the graph, and its `RefineVerdict`.";

static PyObject* GeospatialIndex_refine(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
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
    usearch_geospatial_index_t handle = NULL;
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

    RefineReport* const report_call = usearch_py_call_result(call, &RefineReportType);
    if (!report_call || walk_parse(call, progress_object, stats_object) != 0 ||
        plan_parse(call, plan_object, &GeospatialRefinePlanType) != 0)
        return usearch_py_call_discard(&receiver, call);

    GeospatialPlan* const plan = (GeospatialPlan*)call->holds[callback_slot_k];
    usearch_geospatial_refine_t* const refine = (usearch_geospatial_refine_t*)call->arguments;
    *refine = (usearch_geospatial_refine_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .report = &report_call->report,
        .refine = {.expansion_factor = expansion_factor, .memory_limit = memory_limit},
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
        .plan = plan ? &plan->handle.refine : NULL,
        .phase = plan ? &plan->progress : NULL,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_refine(refine);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Ingest

#pragma region Search

static char const doc_GeospatialIndex_search[] =                                                  //
    "``search(queries, wanted, /, arena=None, *, relation=GeospatialRelation.NEAR, metres=0.0,\n" //
    "seconds=0.0, expansion_factor=0, traversal=Traversal.WALK, predicate=None, progress=None,\n" //
    "progress_steps=0, stats=None) -> Matches``\n"                                                //
    "\n"                                                                                          //
    "Finds the nearest entries of each query observation that the relation admits; one query\n"   //
    "is a one-row matrix. Scores are chord metres between the two positions, altitude\n"          //
    "included, and a selective relation may leave a query fewer than `wanted` matches.\n"         //
    "Calls `usearch_geospatial_search`.\n"                                                        //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  queries (buffer): An `(N, 11)` float64 matrix of query observations.\n"                    //
    "  wanted (int): Upper bound on matches per query, the 'k' in 'kANN'.\n"                      //
    "  arena (Arena, optional): Where the matches land, rewound by the call; None makes one\n"    //
    "    the result owns.\n"                                                                      //
    "  relation (GeospatialRelation, optional): Which entries count as matches.\n"                //
    "  metres (float, optional): The radius around each query; 0 ranks by distance alone.\n"      //
    "  seconds (float, optional): Slack on both ends of the query's time interval.\n"             //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"              //
    "  traversal (Traversal, optional): `Traversal.WALK`, the default, walks the graph;\n"        //
    "    `Traversal.SCAN` scores every entry exactly.\n"                                          //
    "  predicate (callable, optional): Admits a Key when it returns truthy; it re-enters\n"       //
    "    Python, so it runs under the GIL; what it raises goes to `sys.unraisablehook`, and\n"    //
    "    the key is rejected.\n"                                                                  //
    "    A `Batch` answers many keys per call instead.\n"                                         //
    "  progress (callable, optional): Called with (completed, total) queries; returning False\n"  //
    "    stops the batch and raises `CancelledError`, None continues.\n"                          //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"       //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"          //
    "\n"                                                                                          //
    "Returns:\n"                                                                                  //
    "  Matches: `(Q, columns)` runs in the arena, each count bounding its own row.";

static PyObject* GeospatialIndex_search(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                        PyObject* args_names_tuple) {
    PyObject *queries_object, *wanted_object, *arena_object, *relation_object, *metres_object;
    PyObject *seconds_object, *expansion_object, *traversal_object, *predicate_object, *progress_object;
    PyObject *progress_steps_object, *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"queries", &queries_object},
        {"wanted", &wanted_object},
        {"arena", &arena_object},
        {"relation", &relation_object},
        {"metres", &metres_object},
        {"seconds", &seconds_object},
        {"expansion_factor", &expansion_object},
        {"traversal", &traversal_object},
        {"predicate", &predicate_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    size_t wanted = 0, expansion_factor = 0, progress_steps = 0, count = 0;
    int traversal = usearch_traversal_walk_k, relation = usearch_geospatial_relation_near_k;
    usearch_geospatial_tolerance_t tolerance;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("search", args, positional_args_count, args_names_tuple, 2, 3, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(wanted_object, "wanted", &wanted) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        (traversal_object &&
         usearch_py_enum_from_object(traversal_object, "traversal", &usearch_py_traversal, &traversal) != 0) ||
        (relation_object &&
         usearch_py_enum_from_object(relation_object, "relation", &usearch_py_geospatial_relation, &relation) != 0) ||
        tolerance_parse(metres_object, seconds_object, &tolerance) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    Matches* matches = NULL;
    usearch_geospatial_observation_t const* const queries = observations_from_object(queries_object, &count,
                                                                                     &call->buffers[0]);
    if (!queries || usearch_py_predicate_from_object(predicate_object, &call->holds[callback_slot_k]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0 ||
        !(matches = usearch_py_call_arena(&receiver, call, arena_object, &MatchesType)))
        return usearch_py_call_discard(&receiver, call);

    matches->queries = count;
    usearch_geospatial_search_t* const search = (usearch_geospatial_search_t*)call->arguments;
    *search = (usearch_geospatial_search_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .count = count,
        .queries = queries,
        .relation = (usearch_geospatial_relation_t)relation,
        .tolerance = tolerance,
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
    usearch_geospatial_search(search);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_GeospatialIndex_within[] =                                                //
    "``within(queries, /, arena=None, *, metres, relation=GeospatialRelation.NEAR,\n"           //
    "seconds=0.0, traversal=Traversal.WALK, predicate=None, progress=None, progress_steps=0,\n" //
    "stats=None) -> DenseProducts``\n"                                                          //
    "\n"                                                                                        //
    "Reports every entry within `metres` of each query observation that the relation admits,\n" //
    "the containment counterpart of `search`, unbounded in result count and complete on the\n"  //
    "converged graph.\n"                                                                        //
    "Calls `usearch_geospatial_within`.\n"                                                      //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  queries (buffer): An `(N, 11)` float64 matrix of query observations.\n"                  //
    "  arena (Arena, optional): Where the report lands, rewound by the call; None makes one\n"  //
    "    the result owns.\n"                                                                    //
    "  metres (float): The radius around each query, which containment needs.\n"                //
    "  relation (GeospatialRelation, optional): Which entries count as matches.\n"              //
    "  seconds (float, optional): Slack on both ends of the query's time interval.\n"           //
    "  traversal (Traversal, optional): `Traversal.WALK`, the default, walks the graph;\n"      //
    "    `Traversal.SCAN` scores every entry exactly.\n"                                        //
    "  predicate (callable, optional): Admits a Key when it returns truthy.\n"                  //
    "    A `Batch` answers many keys per call instead.\n"                                       //
    "  progress (callable, optional): Called with (completed, total) queries.\n"                //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"     //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"        //
    "\n"                                                                                        //
    "Returns:\n"                                                                                //
    "  DenseProducts: Per query, every entry inside the tolerance with its chord metres,\n"     //
    "    ragged by Apache Arrow offsets.";

/** Answers the report a containment walk wrote, every pair it scored being one it kept. */
static PyObject* within_answer(usearch_py_call_t* call) {
    DenseProducts* const products = (DenseProducts*)call->result;
    products->required = products->pairs;
    return Py_NewRef(call->result);
}

static PyObject* GeospatialIndex_within(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                        PyObject* args_names_tuple) {
    PyObject *queries_object, *arena_object, *metres_object, *relation_object, *seconds_object, *traversal_object,
        *predicate_object, *progress_object, *progress_steps_object, *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"queries", &queries_object},     {"arena", &arena_object},       {"metres", &metres_object},
        {"relation", &relation_object},   {"seconds", &seconds_object},   {"traversal", &traversal_object},
        {"predicate", &predicate_object}, {"progress", &progress_object}, {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    size_t progress_steps = 0, count = 0;
    int traversal = usearch_traversal_walk_k, relation = usearch_geospatial_relation_near_k;
    usearch_geospatial_tolerance_t tolerance;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("within", args, positional_args_count, args_names_tuple, 1, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        (traversal_object &&
         usearch_py_enum_from_object(traversal_object, "traversal", &usearch_py_traversal, &traversal) != 0) ||
        (relation_object &&
         usearch_py_enum_from_object(relation_object, "relation", &usearch_py_geospatial_relation, &relation) != 0) ||
        tolerance_parse(metres_object, seconds_object, &tolerance) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    DenseProducts* products = NULL;
    usearch_geospatial_observation_t const* const queries = observations_from_object(queries_object, &count,
                                                                                     &call->buffers[0]);
    if (!queries || usearch_py_predicate_from_object(predicate_object, &call->holds[callback_slot_k]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0 ||
        !(products = usearch_py_call_arena(&receiver, call, arena_object, &DenseProductsType)))
        return usearch_py_call_discard(&receiver, call);

    products->queries = count;
    call->answer = within_answer;
    usearch_geospatial_within_t* const within = (usearch_geospatial_within_t*)call->arguments;
    *within = (usearch_geospatial_within_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .count = count,
        .queries = queries,
        .relation = (usearch_geospatial_relation_t)relation,
        .tolerance = tolerance,
        .matches = &products->view,
        .written = &products->pairs,
        .arena = usearch_py_arena_handle(call->arena),
        .traversal = (usearch_traversal_t)traversal,
        .predicate = usearch_py_predicate_registered(call->holds[callback_slot_k]),
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_within(within);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Search

#pragma region Retrieval

static char const doc_GeospatialIndex_count[] =                        //
    "count(key) -> int\n"                                              //
    "\n"                                                               //
    "How many entries share `key`; zero means the index holds none.\n" //
    "Calls `usearch_geospatial_count`.\n"                              //
    "\n"                                                               //
    "Args:\n"                                                          //
    "  key (Key | int | str): The identity to count.";

/** A @c usearch_geospatial_count call and the statistics it lands, inside the arguments. */
typedef struct count_call_t {
    usearch_geospatial_count_t arguments;
    usearch_geospatial_key_stats_t key_stats;
} count_call_t;

static PyObject* count_answer(usearch_py_call_t* call) {
    return PyLong_FromSize_t(((count_call_t const*)call->arguments)->key_stats.entries);
}

static PyObject* GeospatialIndex_count(PyObject* self, PyObject* key_object) {
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    usearch_key_t key;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || usearch_py_key_from_object(key_object, &key) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    count_call_t* const count = (count_call_t*)call->arguments;
    call->answer = count_answer;
    count->arguments = (usearch_geospatial_count_t) {.index = handle,
                                                     .executor = receiver.executor,
                                                     .outcome = &call->outcome,
                                                     .completion = receiver.completion,
                                                     .key = key,
                                                     .key_stats = &count->key_stats};
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_count(&count->arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static int GeospatialIndex_contains(PyObject* self, PyObject* key_object) {
    PyObject* const entries = GeospatialIndex_count(self, key_object);
    if (!entries) return -1;
    int const present = PyObject_IsTrue(entries);
    Py_DECREF(entries);
    return present;
}

static char const doc_GeospatialIndex_get[] =                                                         //
    "``get(key, /, arena=None) -> buffer``\n"                                                         //
    "\n"                                                                                              //
    "Every observation stored under `key`, in insertion order, within the rounding the index\n"       //
    "stores them at, as a `(count, 11)` float64 matrix in the arena; empty when the key is absent.\n" //
    "Calls `usearch_geospatial_get`.\n"                                                               //
    "\n"                                                                                              //
    "Args:\n"                                                                                         //
    "  key (Key | int | str): The identity to retrieve.\n"                                            //
    "  arena (Arena, optional): Where the observations land, rewound by the call; None makes one\n"   //
    "    the result owns.";

/** A @c usearch_geospatial_get call and the observations it lands in the arena, inside the record's
 *  arguments. */
typedef struct get_call_t {
    usearch_geospatial_get_t arguments;
    usearch_geospatial_observation_t const* observations;
    size_t written;
} get_call_t;

static PyObject* get_answer(usearch_py_call_t* call) {
    get_call_t const* const get = (get_call_t const*)call->arguments;
    Py_ssize_t const shape[2] = {(Py_ssize_t)get->written, (Py_ssize_t)observation_fields};
    return usearch_py_arena_buffer(&ArenaBufferType, call->arena, call->generation, get->observations, "d",
                                   sizeof(double), shape, 2);
}

static PyObject* GeospatialIndex_get(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    PyObject *key_object, *arena_object;
    usearch_py_argument_t const arguments[] = {{"key", &key_object}, {"arena", &arena_object}};
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    usearch_key_t key;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("get", args, positional_args_count, args_names_tuple, 1, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 || usearch_py_key_from_object(key_object, &key) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    if (usearch_py_call_hold_arena(&receiver, call, arena_object) != 0) return usearch_py_call_discard(&receiver, call);
    get_call_t* const get = (get_call_t*)call->arguments;
    call->answer = get_answer;
    get->arguments = (usearch_geospatial_get_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .key = key,
        .observations = &get->observations,
        .written = &get->written,
        .arena = usearch_py_arena_handle(call->arena),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_get(&get->arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Retrieval

#pragma region Composition

static char const doc_GeospatialIndex_reshard[] =                                                  //
    "``reshard(sources, targets, /, *, expansion_factor=0, memory_limit=0, seed=0,\n"              //
    "progress=None, progress_steps=0, stats=None, plan=None) -> RefineReport``\n"                  //
    "\n"                                                                                           //
    "Moves the entries of `sources` into `targets`, keeping their links rather than\n"             //
    "re-inserting their observations, then runs a round on every target that took entries.\n"      //
    "Several sources into one target, itself among them, gather into it; one source into\n"        //
    "several targets cuts it into one graph-local part per target. Every index stays the\n"        //
    "caller's, and a source the targets do not name ends empty and live. The receiver names\n"     //
    "where the call runs, and need not be among them.\n"                                           //
    "\n"                                                                                           //
    "The handler stops the closing rounds alone, never a move: a stop raises `CancelledError`\n"   //
    "with every entry moved, or, with a `plan`, leaves it live for the next call naming it to\n"   //
    "continue, until `reshard_plan_finish(plan)` closes it. Calls `usearch_geospatial_reshard`.\n" //
    "\n"                                                                                           //
    "A live plan holds no index, so each index it names must outlive its finish.\n"                //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  sources (Sequence[GeospatialIndex]): The indexes whose entries move, each named once.\n"    //
    "  targets (Sequence[GeospatialIndex]): The indexes they move into, each a source or empty.\n" //
    "  expansion_factor (int, optional): Breadth of any walk beside the rounds; 0 derives it.\n"   //
    "  memory_limit (int, optional): Bytes a loaded NVMe handle holds for the part it\n"           //
    "    rewrites; 0 derives what rounds that part in RAM.\n"                                      //
    "  seed (int, optional): Picks the part roots of a cut; equal seeds cut equally.\n"            //
    "  progress (callable, optional): Called with (completed, total) units of each phase of\n"     //
    "    the closing rounds; returning False stops them, None continues.\n"                        //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"        //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"           //
    "  plan (GeospatialReshardPlan, optional): The reshard to open or continue; None runs it to\n" //
    "    its end.\n"                                                                               //
    "\n"                                                                                           //
    "Returns:\n"                                                                                   //
    "  RefineReport: What the closing rounds did to the graph, across the targets.";

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
    usearch_geospatial_index_t* const block = PyMem_Malloc((length ? (size_t)length : 1) *
                                                           sizeof(usearch_geospatial_index_t));
    if (!block) return PyErr_NoMemory(), -1;
    *handles = block;
    for (Py_ssize_t position = 0; position != length; ++position) {
        PyObject* const index = PySequence_Fast_GET_ITEM(sequence, position);
        if (!Py_IS_TYPE(index, &GeospatialIndexType)) {
            PyErr_Format(PyExc_TypeError, "the '%s' argument must hold GeospatialIndex objects, got %S", field,
                         Py_TYPE(index));
            return -1;
        }
        block[position] = ((GeospatialIndex*)index)->handle;
    }
    *count = (size_t)length;
    return 0;
}

static PyObject* GeospatialIndex_reshard(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                         PyObject* args_names_tuple) {
    PyObject *sources_object, *targets_object, *expansion_object, *memory_limit_object, *seed_object, *progress_object,
        *progress_steps_object, *stats_object, *plan_object;
    usearch_py_argument_t const arguments[] = {
        {"sources", &sources_object},
        {"targets", &targets_object},
        {"expansion_factor", &expansion_object},
        {"memory_limit", &memory_limit_object},
        {"seed", &seed_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
        {"plan", &plan_object},
    };
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    size_t expansion_factor = 0, memory_limit = 0, progress_steps = 0;
    uint64_t seed = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("reshard", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(memory_limit_object, "memory_limit", &memory_limit) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        usearch_py_seed_from_object(seed_object, &seed) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    size_t sources_count = 0, targets_count = 0;
    RefineReport* const report_call = usearch_py_call_result(call, &RefineReportType);
    if (!report_call ||
        indexes_from_object(sources_object, "sources", &call->holds[sources_slot_k], &call->blocks[0],
                            &sources_count) != 0 ||
        indexes_from_object(targets_object, "targets", &call->holds[targets_slot_k], &call->blocks[1],
                            &targets_count) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0 ||
        plan_parse(call, plan_object, &GeospatialReshardPlanType) != 0)
        return usearch_py_call_discard(&receiver, call);

    GeospatialPlan* const plan = (GeospatialPlan*)call->holds[callback_slot_k];
    usearch_geospatial_reshard_t* const reshard = (usearch_geospatial_reshard_t*)call->arguments;
    *reshard = (usearch_geospatial_reshard_t) {
        .sources_count = sources_count,
        .sources = call->blocks[0],
        .targets_count = targets_count,
        .targets = call->blocks[1],
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .report = &report_call->report,
        .refine = {.expansion_factor = expansion_factor, .memory_limit = memory_limit},
        .seed = seed,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
        .plan = plan ? &plan->handle.reshard : NULL,
        .phase = plan ? &plan->progress : NULL,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_reshard(reshard);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Composition

#pragma region Plans

static char const doc_GeospatialRefinePlan[] =                                                     //
    "GeospatialRefinePlan()\n"                                                                     //
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
    "  total (int): Units of that phase in all.";

static char const doc_GeospatialReshardPlan[] =                                                     //
    "GeospatialReshardPlan()\n"                                                                     //
    "\n"                                                                                            //
    "A reshard in flight: its own sources and targets, the entries still to move, then the\n"       //
    "rounds of every target that took entries.\n"                                                   //
    "\n"                                                                                            //
    "Passed as `plan=` to `reshard`, the first call opens it and a stop leaves it live, so the\n"   //
    "next call naming it moves what is still pending and continues the rounds, reading its own\n"   //
    "sources and targets rather than the call's, and `reshard_plan_finish` closes it. Until\n"      //
    "then every index it names admits no insertion and no save, and every call names the\n"         //
    "executor it was opened with. A plan dropped unfinished leaks and keeps its indexes closed\n"   //
    "to insertion.\n"                                                                               //
    "\n"                                                                                            //
    "Attributes:\n"                                                                                 //
    "  phase (Phase): Where the last call naming it stopped, `Phase.DONE` once its rounds ended.\n" //
    "  ordinal (int): That phase's position among the ones the plan runs.\n"                        //
    "  completed (int): Units of that phase done, entries or islands.\n"                            //
    "  total (int): Units of that phase in all.";

static PyObject* GeospatialPlan_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    if (PyTuple_GET_SIZE(args) != 0 || (keywords && PyDict_Size(keywords) != 0)) {
        PyErr_Format(PyExc_TypeError, "%s() takes no arguments", type->tp_name + sizeof("usearch.") - 1);
        return NULL;
    }
    return type->tp_alloc(type, 0);
}

static PyObject* GeospatialPlan_get_phase(GeospatialPlan* self, void* closure) {
    return usearch_py_enum_to_object(&usearch_py_phase, (int)self->progress.phase);
}

static PyGetSetDef GeospatialPlan_getset[] = {
    {"phase", (getter)GeospatialPlan_get_phase, NULL, "Where the last call naming the plan stopped, as a `Phase`.",
     NULL},
    {"ordinal", usearch_py_size_getter, NULL, "That phase's position among the ones the plan runs.",
     (void*)offsetof(GeospatialPlan, progress.ordinal)},
    {"completed", usearch_py_size_getter, NULL, "Units of that phase done, entries or islands.",
     (void*)offsetof(GeospatialPlan, progress.completed)},
    {"total", usearch_py_size_getter, NULL, "Units of that phase in all.",
     (void*)offsetof(GeospatialPlan, progress.total)},
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject GeospatialRefinePlanType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.GeospatialRefinePlan",
    .tp_doc = doc_GeospatialRefinePlan,
    .tp_basicsize = sizeof(GeospatialPlan),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = GeospatialPlan_tp_new,
    .tp_getset = GeospatialPlan_getset,
};

PyTypeObject GeospatialReshardPlanType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.GeospatialReshardPlan",
    .tp_doc = doc_GeospatialReshardPlan,
    .tp_basicsize = sizeof(GeospatialPlan),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = GeospatialPlan_tp_new,
    .tp_getset = GeospatialPlan_getset,
};

static char const doc_GeospatialIndex_refine_plan_finish[] =                                       //
    "refine_plan_finish(plan) -> RefineReport\n"                                                   //
    "\n"                                                                                           //
    "Closes a `GeospatialRefinePlan` on this index and the receiver's executor, the ones it was\n" //
    "opened with, and frees it whatever the call answers; a round a stop left open closes as a\n"  //
    "stop at that barrier would. Calls `usearch_geospatial_refine_plan_finish`.\n"                 //
    "\n"                                                                                           //
    "Returns:\n"                                                                                   //
    "  RefineReport: What the whole plan did to the graph.";

static char const doc_GeospatialIndex_reshard_plan_finish[] =                                        //
    "reshard_plan_finish(plan) -> RefineReport\n"                                                    //
    "\n"                                                                                             //
    "Closes a `GeospatialReshardPlan` on the receiver's executor, the one it was opened with, and\n" //
    "frees it whatever the call answers, reopening every index it names; a plan whose entries\n"     //
    "have not all moved runs no round and reports zeros. Calls\n"                                    //
    "`usearch_geospatial_reshard_plan_finish`.\n"                                                    //
    "\n"                                                                                             //
    "Returns:\n"                                                                                     //
    "  RefineReport: What the whole plan did to the graph.";

/** Opens a finish of the plan @p plan_object of @p type through @p self: the report it writes into,
 *  the plan held, and its handle taken, since the library frees it whatever the call answers. */
static usearch_py_call_t* plan_finish_open(PyObject* self, PyObject* plan_object, PyTypeObject* type,
                                           usearch_py_receiver_t* receiver, usearch_geospatial_index_t* handle,
                                           usearch_py_call_t* stack, RefineReport** report_call) {
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, receiver, handle) != 0 || !(call = usearch_py_call_open(receiver, stack)))
        return NULL;
    if (!plan_object || plan_object == Py_None) {
        PyErr_Format(PyExc_TypeError, "expected a %s, got None", type->tp_name);
        usearch_py_call_discard(receiver, call);
        return NULL;
    }
    if (plan_parse(call, plan_object, type) != 0 || !(*report_call = usearch_py_call_result(call, &RefineReportType))) {
        usearch_py_call_discard(receiver, call);
        return NULL;
    }
    return call;
}

static PyObject* GeospatialIndex_refine_plan_finish(PyObject* self, PyObject* plan_object) {
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    usearch_py_call_t stack;
    RefineReport* report_call = NULL;
    usearch_py_call_t* const call = plan_finish_open(self, plan_object, &GeospatialRefinePlanType, &receiver, &handle,
                                                     &stack, &report_call);
    if (!call) return NULL;
    GeospatialPlan* const plan = (GeospatialPlan*)plan_object;
    usearch_geospatial_refine_plan_finish_t* const finish = (usearch_geospatial_refine_plan_finish_t*)call->arguments;
    *finish = (usearch_geospatial_refine_plan_finish_t) {.index = handle,
                                                         .executor = receiver.executor,
                                                         .outcome = &call->outcome,
                                                         .completion = receiver.completion,
                                                         .plan = plan->handle.refine,
                                                         .report = &report_call->report};
    plan->handle.refine = NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_refine_plan_finish(finish);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static PyObject* GeospatialIndex_reshard_plan_finish(PyObject* self, PyObject* plan_object) {
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    usearch_py_call_t stack;
    RefineReport* report_call = NULL;
    usearch_py_call_t* const call = plan_finish_open(self, plan_object, &GeospatialReshardPlanType, &receiver, &handle,
                                                     &stack, &report_call);
    if (!call) return NULL;
    GeospatialPlan* const plan = (GeospatialPlan*)plan_object;
    usearch_geospatial_reshard_plan_finish_t* const finish = (usearch_geospatial_reshard_plan_finish_t*)call->arguments;
    *finish = (usearch_geospatial_reshard_plan_finish_t) {.executor = receiver.executor,
                                                          .outcome = &call->outcome,
                                                          .completion = receiver.completion,
                                                          .plan = plan->handle.reshard,
                                                          .report = &report_call->report};
    plan->handle.reshard = NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_reshard_plan_finish(finish);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Plans

#pragma region Serialization

static char const doc_GeospatialIndex_save[] =                                              //
    "``save(path, /, *, progress=None, progress_steps=0) -> None``\n"                       //
    "\n"                                                                                    //
    "Writes the index under `path`, in the layout its extension picks: one `.safegraphs`\n" //
    "monolith, or a `.safegraphs.json` manifest beside one part per shard. Demands a\n"     //
    "mutation-free window; concurrent readers keep searching throughout.\n"                 //
    "Calls `usearch_geospatial_save`.\n"                                                    //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  path (str | bytes | os.PathLike): Where the checkpoint goes.\n"                      //
    "  progress (callable, optional): Called with (completed, total) parts; returning\n"    //
    "    False stops the write and raises `CancelledError`.\n"                              //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject* GeospatialIndex_save(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    PyObject *path_object, *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {{"path", &path_object},
                                               {"progress", &progress_object},
                                               {"progress_steps", &progress_steps_object}};
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
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
    usearch_geospatial_save_t* const save = (usearch_geospatial_save_t*)call->arguments;
    *save = (usearch_geospatial_save_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .path = path,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_save(save);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_GeospatialIndex_save_buffer[] =                                         //
    "``save_buffer(arena=None, *, progress=None, progress_steps=0) -> Monolith``\n"           //
    "\n"                                                                                      //
    "Serializes the index into the arena: one monolith for a handle of one shard, the very\n" //
    "bytes `save` writes, or a manifest leading one part per shard.\n"                        //
    "Calls `usearch_geospatial_save_buffer`.\n"                                               //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  arena (Arena, optional): Where the checkpoint lands, rewound by the call; None\n"      //
    "    makes one the result owns.\n"                                                        //
    "  progress (callable, optional): Called with (completed, total) parts; returning\n"      //
    "    False stops the write and raises `CancelledError`, None continues.\n"                //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"   //
    "\n"                                                                                      //
    "Returns:\n"                                                                              //
    "  Monolith: The checkpoint's bytes, in place in the arena.";

/** A @c usearch_geospatial_save_buffer call and the monolith it lands, inside the arguments. */
typedef struct save_buffer_call_t {
    usearch_geospatial_save_buffer_t arguments;
    usearch_bytes_view_t monolith;
} save_buffer_call_t;

static PyObject* save_buffer_answer(usearch_py_call_t* call) {
    usearch_bytes_view_t const monolith = ((save_buffer_call_t const*)call->arguments)->monolith;
    Py_ssize_t const shape[1] = {(Py_ssize_t)monolith.length};
    return usearch_py_arena_buffer(&MonolithType, call->arena, call->generation, monolith.data, "B", 1, shape, 1);
}

static PyObject* GeospatialIndex_save_buffer(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                             PyObject* args_names_tuple) {
    PyObject *arena_object, *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {{"arena", &arena_object},
                                               {"progress", &progress_object},
                                               {"progress_steps", &progress_steps_object}};
    usearch_py_receiver_t receiver;
    usearch_geospatial_index_t handle = NULL;
    size_t progress_steps = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("save_buffer", args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    if (usearch_py_callable_from_object(progress_object, "progress", &call->holds[progress_slot_k]) != 0 ||
        usearch_py_call_hold_arena(&receiver, call, arena_object) != 0)
        return usearch_py_call_discard(&receiver, call);
    save_buffer_call_t* const save = (save_buffer_call_t*)call->arguments;
    call->answer = save_buffer_answer;
    save->arguments = (usearch_geospatial_save_buffer_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .monolith = &save->monolith,
        .arena = usearch_py_arena_handle(call->arena),
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_save_buffer(&save->arguments);
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
        usearch_py_class_from_object(ceiling_object, ceiling_name(ceiling_classes), ceiling_classes, ceiling) != 0 ||
        usearch_py_optional_executor_from_object(executor_object, executor) != 0 ||
        usearch_py_size_from_object(values_per_key_object, "values_per_key", values_per_key) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", progress_steps) != 0)
        return -1;
    return usearch_py_callable_from_object(progress_object, "progress", progress);
}

static char const doc_GeospatialIndex_load_ram[] =                                                  //
    "``load_ram(path, /, *, memory_domains=None, executor=None, values_per_key=0, progress=None,\n" //
    "progress_steps=0, cpu_class_ceiling=None) -> GeospatialIndex``\n"                              //
    "\n"                                                                                            //
    "Copies a checkpoint into a new, mutable index in host memory, its geometry read from the\n"    //
    "checkpoint's own headers. Name the `.safegraphs.json` manifest, never one of its parts.\n"     //
    "Calls `usearch_geospatial_load_ram`.\n"                                                        //
    "\n"                                                                                            //
    "Args:\n"                                                                                       //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                     //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place the parts on.\n"            //
    "  executor (Executor, optional): The workers the read fans out over; None reads on the\n"      //
    "    calling thread.\n"                                                                         //
    "  values_per_key (int, optional): Entries sharing one key, which no checkpoint records.\n"     //
    "  progress (callable, optional): Called with (completed, total) parts; returning False\n"      //
    "    stops the read and raises `CancelledError`, creating no index.\n"                          //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"         //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n"   //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.";

static PyObject* GeospatialIndex_load_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
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

    usearch_geospatial_index_t handle = NULL;
    usearch_geospatial_load_ram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_load_ram(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(memory_domains);
    Py_XDECREF(decoded);
    Py_XDECREF(progress);
    return result;
}

static char const doc_GeospatialIndex_load_ram_buffer[] =                                         //
    "``load_ram_buffer(buffer, /, *, memory_domains=None, executor=None, values_per_key=0,\n"     //
    "progress=None, progress_steps=0, cpu_class_ceiling=None) -> GeospatialIndex``\n"             //
    "\n"                                                                                          //
    "Copies a checkpoint out of a buffer into a new, mutable index in host memory, as\n"          //
    "`load_ram` does from a path.\n"                                                              //
    "Calls `usearch_geospatial_load_ram_buffer`.\n"                                               //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  buffer (buffer): One serialized checkpoint, such as `save_buffer` returns.\n"              //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place shards on.\n"             //
    "  executor (Executor, optional): The workers the copy fans out over.\n"                      //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                //
    "  progress (callable, optional): Called with (completed, total) parts.\n"                    //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"       //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n" //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.";

static PyObject* GeospatialIndex_load_ram_buffer(PyTypeObject* type, PyObject* const* args,
                                                 Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
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
        usearch_py_checkpoint_from_object(buffer_object, &backing) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        goto release;

    usearch_geospatial_index_t handle = NULL;
    usearch_geospatial_load_ram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_load_ram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(memory_domains);
    usearch_py_buffer_release(&backing);
    Py_XDECREF(progress);
    return result;
}

static char const doc_GeospatialIndex_load_vram[] =                                                //
    "``load_vram(path, /, *, devices=None, executor=None, values_per_key=0, progress=None,\n"      //
    "progress_steps=0, gpu_class_ceiling=None) -> GeospatialIndex``\n"                             //
    "\n"                                                                                           //
    "Copies a checkpoint into a new, mutable index on GPU devices, with the placement rules\n"     //
    "of `load_ram`.\n"                                                                             //
    "Calls `usearch_geospatial_load_vram`.\n"                                                      //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                    //
    "  devices (Sequence[int], optional): GPU ordinals to place the parts on.\n"                   //
    "  executor (Executor, optional): The streams the upload runs on, one per device.\n"           //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                 //
    "  progress (callable, optional): Called with (completed, total) parts.\n"                     //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"        //
    "  gpu_class_ceiling (GPUClass, optional): The newest class dispatch may pick, capped below\n" //
    "    the best class every device loads, else unsupported; None means no ceiling.";

static PyObject* GeospatialIndex_load_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
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

    usearch_geospatial_index_t handle = NULL;
    usearch_geospatial_load_vram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_load_vram(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(devices);
    Py_XDECREF(decoded);
    Py_XDECREF(progress);
    return result;
}

static char const doc_GeospatialIndex_load_vram_buffer[] =                                         //
    "``load_vram_buffer(buffer, /, *, devices=None, executor=None, values_per_key=0,\n"            //
    "progress=None, progress_steps=0, gpu_class_ceiling=None) -> GeospatialIndex``\n"              //
    "\n"                                                                                           //
    "Copies a checkpoint out of a host buffer into a new, mutable index on GPU devices.\n"         //
    "Calls `usearch_geospatial_load_vram_buffer`.\n"                                               //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  buffer (buffer): One serialized checkpoint, such as `save_buffer` returns.\n"               //
    "  devices (Sequence[int], optional): GPU ordinals to place shards on.\n"                      //
    "  executor (Executor, optional): The streams the upload runs on, one per device.\n"           //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                 //
    "  progress (callable, optional): Called with (completed, total) parts.\n"                     //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"        //
    "  gpu_class_ceiling (GPUClass, optional): The newest class dispatch may pick, capped below\n" //
    "    the best class every device loads, else unsupported; None means no ceiling.";

static PyObject* GeospatialIndex_load_vram_buffer(PyTypeObject* type, PyObject* const* args,
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
        usearch_py_checkpoint_from_object(buffer_object, &backing) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        goto release;

    usearch_geospatial_index_t handle = NULL;
    usearch_geospatial_load_vram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_load_vram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(devices);
    usearch_py_buffer_release(&backing);
    Py_XDECREF(progress);
    return result;
}

static char const doc_GeospatialIndex_view_ram[] =                                                //
    "``view_ram(path, /, *, cpu_class_ceiling=None) -> GeospatialIndex``\n"                       //
    "\n"                                                                                          //
    "Serves a checkpoint in place through a memory mapping, as a read-only index whose every\n"   //
    "mutating verb raises `ReadOnlyError`.\n"                                                     //
    "Calls `usearch_geospatial_view_ram`.\n"                                                      //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                   //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n" //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.";

static PyObject* GeospatialIndex_view_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                          PyObject* args_names_tuple) {
    PyObject *path_object, *cpu_class_ceiling_object, *decoded = NULL;
    usearch_py_argument_t const arguments[] = {
        {"path", &path_object},
        {"cpu_class_ceiling", &cpu_class_ceiling_object},
    };
    usearch_bytes_view_t path;
    int cpu_class_ceiling = 0;
    if (usearch_py_arguments_parse("view_ram", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_class_from_object(cpu_class_ceiling_object, "cpu_class_ceiling", &usearch_py_cpu_class,
                                     &cpu_class_ceiling) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0)
        return NULL;

    usearch_geospatial_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_geospatial_view_ram_t view_arguments = {
        .index = &handle,
        .outcome = &outcome,
        .path = path,
        .ram = {NULL, 0, 0, (usearch_cpu_class_t)cpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_view_ram(&view_arguments);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    return index_adopt(type, handle, &outcome);
}

static char const doc_GeospatialIndex_view_nvme[] =                                                     //
    "``view_nvme(path, /, *, paths=None, executor=None, cpu_class_ceiling=None) -> GeospatialIndex``\n" //
    "\n"                                                                                                //
    "Serves a checkpoint in place from NVMe drives as a read-only index, every later call\n"            //
    "reading it through its executor's engines.\n"                                                      //
    "Calls `usearch_geospatial_view_nvme`.\n"                                                           //
    "\n"                                                                                                //
    "Args:\n"                                                                                           //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                         //
    "  paths (Sequence[str | bytes | PathLike], optional): Devices or directories to serve\n"           //
    "    the parts from; None serves each where its manifest points.\n"                                 //
    "  executor (Executor, optional): The NVMe executor whose rings read the headers.\n"                //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n"       //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.";

static PyObject* GeospatialIndex_view_nvme(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                           PyObject* args_names_tuple) {
    PyObject *path_object, *paths_object, *executor_object, *cpu_class_ceiling_object, *decoded = NULL, *result = NULL;
    usearch_py_argument_t const arguments[] = {
        {"path", &path_object},
        {"paths", &paths_object},
        {"executor", &executor_object},
        {"cpu_class_ceiling", &cpu_class_ceiling_object},
    };
    usearch_bytes_view_t path;
    usearch_py_paths_t paths = {0};
    usearch_executor_t executor = NULL;
    int cpu_class_ceiling = 0;
    if (usearch_py_arguments_parse("view_nvme", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_optional_executor_from_object(executor_object, &executor) != 0 ||
        usearch_py_class_from_object(cpu_class_ceiling_object, "cpu_class_ceiling", &usearch_py_cpu_class,
                                     &cpu_class_ceiling) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0 ||
        usearch_py_paths_from_object(paths_object, &paths) != 0)
        goto release;

    usearch_geospatial_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_geospatial_view_nvme_t view_arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .nvme = {paths.views, paths.count, (usearch_cpu_class_t)cpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_view_nvme(&view_arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    usearch_py_paths_release(&paths);
    Py_XDECREF(decoded);
    return result;
}

/** Reads the source and the optional @p executor every metadata read takes; returns 0 or -1 with an
 *  exception set. */
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

static char const doc_GeospatialIndex_metadata_read[] =                                  //
    "``metadata_read(path, /, *, executor=None) -> GeospatialMetadata``\n"               //
    "\n"                                                                                 //
    "Reads the metric and geometry a checkpoint froze, without loading it.\n"            //
    "Calls `usearch_geospatial_metadata_read`.\n"                                        //
    "\n"                                                                                 //
    "Args:\n"                                                                            //
    "  path (str | bytes | os.PathLike): The checkpoint to inspect.\n"                   //
    "  executor (Executor, optional): The workers the read runs on; None reads on the\n" //
    "    calling thread.";

static PyObject* GeospatialIndex_metadata_read(PyObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                               PyObject* args_names_tuple) {
    PyObject *path_object, *decoded = NULL;
    usearch_executor_t executor = NULL;
    usearch_bytes_view_t path;
    if (metadata_parse("metadata_read", args, positional_args_count, args_names_tuple, "path", &path_object,
                       &executor) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0)
        return NULL;
    usearch_geospatial_metadata_t metadata = {0};
    usearch_outcome_t outcome = {0};
    usearch_geospatial_metadata_read_t arguments = {
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .metadata = &metadata,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_metadata_read(&arguments);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return metadata_to_object(&metadata);
}

static char const doc_GeospatialIndex_metadata_read_buffer[] =                           //
    "``metadata_read_buffer(buffer, /, *, executor=None) -> GeospatialMetadata``\n"      //
    "\n"                                                                                 //
    "Reads a serialized checkpoint's metric and geometry, as `metadata_read`\n"          //
    "does from a path.\n"                                                                //
    "Calls `usearch_geospatial_metadata_read_buffer`.\n"                                 //
    "\n"                                                                                 //
    "Args:\n"                                                                            //
    "  buffer (buffer): One serialized checkpoint, such as `save_buffer` returns.\n"     //
    "  executor (Executor, optional): The workers the read runs on; None reads on the\n" //
    "    calling thread.";

static PyObject* GeospatialIndex_metadata_read_buffer(PyObject* type, PyObject* const* args,
                                                      Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    PyObject* buffer_object;
    usearch_executor_t executor = NULL;
    usearch_py_buffer_backing_t backing;
    if (metadata_parse("metadata_read_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                       &executor) != 0 ||
        usearch_py_checkpoint_from_object(buffer_object, &backing) != 0)
        return NULL;
    usearch_geospatial_metadata_t metadata = {0};
    usearch_outcome_t outcome = {0};
    usearch_geospatial_metadata_read_buffer_t arguments = {
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .metadata = &metadata,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_geospatial_metadata_read_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    usearch_py_buffer_release(&backing);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return metadata_to_object(&metadata);
}

#pragma endregion Serialization

#pragma region Registration

static PyMethodDef GeospatialIndex_methods[] = {
    {"init_ram", (PyCFunction)GeospatialIndex_init_ram, usearch_py_method_flags_k | METH_CLASS,
     doc_GeospatialIndex_init_ram},
    {"init_vram", (PyCFunction)GeospatialIndex_init_vram, usearch_py_method_flags_k | METH_CLASS,
     doc_GeospatialIndex_init_vram},
    {"load_ram", (PyCFunction)GeospatialIndex_load_ram, usearch_py_method_flags_k | METH_CLASS,
     doc_GeospatialIndex_load_ram},
    {"load_ram_buffer", (PyCFunction)GeospatialIndex_load_ram_buffer, usearch_py_method_flags_k | METH_CLASS,
     doc_GeospatialIndex_load_ram_buffer},
    {"load_vram", (PyCFunction)GeospatialIndex_load_vram, usearch_py_method_flags_k | METH_CLASS,
     doc_GeospatialIndex_load_vram},
    {"load_vram_buffer", (PyCFunction)GeospatialIndex_load_vram_buffer, usearch_py_method_flags_k | METH_CLASS,
     doc_GeospatialIndex_load_vram_buffer},
    {"view_ram", (PyCFunction)GeospatialIndex_view_ram, usearch_py_method_flags_k | METH_CLASS,
     doc_GeospatialIndex_view_ram},
    {"view_nvme", (PyCFunction)GeospatialIndex_view_nvme, usearch_py_method_flags_k | METH_CLASS,
     doc_GeospatialIndex_view_nvme},
    {"metadata_read", (PyCFunction)GeospatialIndex_metadata_read, usearch_py_method_flags_k | METH_STATIC,
     doc_GeospatialIndex_metadata_read},
    {"metadata_read_buffer", (PyCFunction)GeospatialIndex_metadata_read_buffer, usearch_py_method_flags_k | METH_STATIC,
     doc_GeospatialIndex_metadata_read_buffer},
    {"on", (PyCFunction)GeospatialIndex_on, METH_O, doc_GeospatialIndex_on},
    {"deferred", (PyCFunction)GeospatialIndex_deferred, METH_O, doc_GeospatialIndex_deferred},
    {"describe", (PyCFunction)GeospatialIndex_describe, METH_NOARGS, doc_GeospatialIndex_describe},
    {"reserve", (PyCFunction)GeospatialIndex_reserve, METH_O, doc_GeospatialIndex_reserve},
    {"clear", (PyCFunction)GeospatialIndex_clear, METH_NOARGS, doc_GeospatialIndex_clear},
    {"insert", (PyCFunction)GeospatialIndex_insert, usearch_py_method_flags_k, doc_GeospatialIndex_insert},
    {"refine", (PyCFunction)GeospatialIndex_refine, usearch_py_method_flags_k, doc_GeospatialIndex_refine},
    {"health", (PyCFunction)GeospatialIndex_health, METH_NOARGS, doc_GeospatialIndex_health},
    {"search", (PyCFunction)GeospatialIndex_search, usearch_py_method_flags_k, doc_GeospatialIndex_search},
    {"within", (PyCFunction)GeospatialIndex_within, usearch_py_method_flags_k, doc_GeospatialIndex_within},
    {"count", (PyCFunction)GeospatialIndex_count, METH_O, doc_GeospatialIndex_count},
    {"get", (PyCFunction)GeospatialIndex_get, usearch_py_method_flags_k, doc_GeospatialIndex_get},
    {"reshard", (PyCFunction)GeospatialIndex_reshard, usearch_py_method_flags_k, doc_GeospatialIndex_reshard},
    {"refine_plan_finish", (PyCFunction)GeospatialIndex_refine_plan_finish, METH_O,
     doc_GeospatialIndex_refine_plan_finish},
    {"reshard_plan_finish", (PyCFunction)GeospatialIndex_reshard_plan_finish, METH_O,
     doc_GeospatialIndex_reshard_plan_finish},
    {"save", (PyCFunction)GeospatialIndex_save, usearch_py_method_flags_k, doc_GeospatialIndex_save},
    {"save_buffer", (PyCFunction)GeospatialIndex_save_buffer, usearch_py_method_flags_k,
     doc_GeospatialIndex_save_buffer},
    {NULL, NULL, 0, NULL},
};

/** Every verb a receiver runs on its executor, the same for an @c On and a @c DeferredOn, since no
 *  geospatial verb answers an iterator. */
static PyMethodDef GeospatialIndex_receiver_methods[] = {
    {"describe", (PyCFunction)GeospatialIndex_describe, METH_NOARGS, doc_GeospatialIndex_describe},
    {"reserve", (PyCFunction)GeospatialIndex_reserve, METH_O, doc_GeospatialIndex_reserve},
    {"clear", (PyCFunction)GeospatialIndex_clear, METH_NOARGS, doc_GeospatialIndex_clear},
    {"insert", (PyCFunction)GeospatialIndex_insert, usearch_py_method_flags_k, doc_GeospatialIndex_insert},
    {"refine", (PyCFunction)GeospatialIndex_refine, usearch_py_method_flags_k, doc_GeospatialIndex_refine},
    {"health", (PyCFunction)GeospatialIndex_health, METH_NOARGS, doc_GeospatialIndex_health},
    {"search", (PyCFunction)GeospatialIndex_search, usearch_py_method_flags_k, doc_GeospatialIndex_search},
    {"within", (PyCFunction)GeospatialIndex_within, usearch_py_method_flags_k, doc_GeospatialIndex_within},
    {"count", (PyCFunction)GeospatialIndex_count, METH_O, doc_GeospatialIndex_count},
    {"get", (PyCFunction)GeospatialIndex_get, usearch_py_method_flags_k, doc_GeospatialIndex_get},
    {"reshard", (PyCFunction)GeospatialIndex_reshard, usearch_py_method_flags_k, doc_GeospatialIndex_reshard},
    {"refine_plan_finish", (PyCFunction)GeospatialIndex_refine_plan_finish, METH_O,
     doc_GeospatialIndex_refine_plan_finish},
    {"reshard_plan_finish", (PyCFunction)GeospatialIndex_reshard_plan_finish, METH_O,
     doc_GeospatialIndex_reshard_plan_finish},
    {"save", (PyCFunction)GeospatialIndex_save, usearch_py_method_flags_k, doc_GeospatialIndex_save},
    {"save_buffer", (PyCFunction)GeospatialIndex_save_buffer, usearch_py_method_flags_k,
     doc_GeospatialIndex_save_buffer},
    {NULL, NULL, 0, NULL},
};

usearch_py_receiver_verbs_t const usearch_py_geospatial_receiver_verbs = {
    &GeospatialIndexType, GeospatialIndex_receiver_methods, GeospatialIndex_receiver_methods};

static PyGetSetDef GeospatialIndex_getset[] = {
    {"capacity", (getter)GeospatialIndex_get_counter, NULL,
     "Entries the index hosts without growing again. Calls `usearch_geospatial_describe`.",
     (void*)offsetof(usearch_geospatial_index_description_t, capacity)},
    {"connectivity", (getter)GeospatialIndex_get_counter, NULL,
     "Connections per node in the graph. Calls `usearch_geospatial_describe`.",
     (void*)offsetof(usearch_geospatial_index_description_t, metadata.connectivity)},
    {"memory_usage", (getter)GeospatialIndex_get_counter, NULL,
     "Bytes the handle holds; traversal scratch is the executor's. Calls `usearch_geospatial_describe`.",
     (void*)offsetof(usearch_geospatial_index_description_t, memory_usage)},
    {"cpu_class", (getter)GeospatialIndex_get_cpu_class, NULL,
     "The `CPUClass` dispatch picked on this machine, or None. Calls `usearch_geospatial_describe`.", NULL},
    {"gpu_class", (getter)GeospatialIndex_get_gpu_class, NULL,
     "The `GPUClass` dispatch picked on the index's devices, or None. Calls `usearch_geospatial_describe`.", NULL},
    {"shards_count", (getter)GeospatialIndex_get_counter, NULL, "Live shards. Calls `usearch_geospatial_describe`.",
     (void*)offsetof(usearch_geospatial_index_description_t, shards_count)},
    {NULL, NULL, NULL, NULL, NULL},
};

static PySequenceMethods GeospatialIndex_as_sequence = {
    .sq_length = (lenfunc)GeospatialIndex_length,
    .sq_contains = (objobjproc)GeospatialIndex_contains,
};

PyTypeObject GeospatialIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.GeospatialIndex",
    .tp_doc = doc_GeospatialIndex,
    .tp_basicsize = sizeof(GeospatialIndex),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = GeospatialIndex_tp_new,
    .tp_dealloc = (destructor)GeospatialIndex_dealloc,
    .tp_repr = (reprfunc)GeospatialIndex_repr,
    .tp_methods = GeospatialIndex_methods,
    .tp_getset = GeospatialIndex_getset,
    .tp_as_sequence = &GeospatialIndex_as_sequence,
};

#pragma endregion Registration
