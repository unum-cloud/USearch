/**
 *  @file python/shapes_index.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief CPython similarity search over 3D point clouds of any size.
 *
 *  Entries are point clouds rather than vectors, so the geometry is an optionally frozen point
 *  count, and the metrics score whole shapes against each other: Chamfer, Hausdorff,
 *  Wasserstein-Procrustes, and Gromov-Wasserstein over molecules, conformers, and protein
 *  backbones. Point counts vary per entry, so a batch arrives as a list of (points, 3) matrices
 *  gathered once into one tape.
 *
 *  Every verb is written once and reached through three receivers, as in @c dense_index.c: the
 *  index runs it on the calling thread, an @c On blocks on its executor, and a @c DeferredOn
 *  submits it there and answers a @c Deferred. A verb fills one call record and opens its result
 *  as the slot the library writes into, so the same body serves every completion.
 */
#include "lib.h"

#pragma region Handles

/** An owned index handle. */
typedef struct ShapesIndex {
    PyObject ob_base;
    usearch_shapes_index_t handle;
} ShapesIndex;

/** A live index's frozen metadata and current counters, the slot @c describe writes them into. */
typedef struct ShapesIndexDescription {
    PyObject ob_base;
    usearch_shapes_index_description_t description;
} ShapesIndexDescription;

/** A refine or a reshard in flight, continued by every call naming it until its finish verb. */
typedef struct ShapesPlan {
    PyObject ob_base;

    /** The engine plan, NULL until a call opens it and once its finish verb closed it. */
    union {
        usearch_shapes_refine_plan_t refine;
        usearch_shapes_reshard_plan_t reshard;
    } handle;

    /** Where the last call naming the plan stopped. */
    usearch_phase_progress_t progress;
} ShapesPlan;

/** The holds every shapes call keeps in the same slots. */
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
static int receiver_from_object(PyObject* self, usearch_py_receiver_t* receiver, usearch_shapes_index_t* handle) {
    if (usearch_py_receiver_from_object(self, receiver) != 0) return -1;
    if (!Py_IS_TYPE(receiver->index, &ShapesIndexType)) {
        PyErr_Format(PyExc_TypeError, "expected a receiver over a ShapesIndex, got one over %S",
                     Py_TYPE(receiver->index));
        return -1;
    }
    *handle = ((ShapesIndex*)receiver->index)->handle;
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

/** Views the points of one shape, a matrix of x, y and z columns, into @p points. */
static int shape_view(PyObject* item, usearch_scalar_t scalar_hint, usearch_matrix_view_t* points,
                      usearch_py_buffer_backing_t* backing) {
    if (usearch_py_matrix_from_object(item, scalar_hint, points, backing) != 0) return -1;
    if (points->columns == 3) return 0;
    size_t const columns = points->columns;
    usearch_py_buffer_release(backing);
    return usearch_py_raise(usearch_status_shape_mismatch_k, "a shape is a matrix of x, y and z columns, got %zu",
                            columns);
}

/**
 *  @brief Gathers a matrix of x, y and z columns, or a list or tuple of them, into one tape; a lone
 *      matrix is a tape of one.
 *
 *  The items are separate objects, so their points are copied once into the run the library
 *  reads: the @c count + 1 offsets, then every shape's points end to end, in one block passed to
 *  @p block. The point counts are read in a first pass and checked again while copying, since an
 *  exporter may be resized by another thread in between.
 *
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int shapes_from_object(PyObject* object, usearch_scalar_t scalar_hint, void** block, size_t* count,
                              usearch_tape_view_t* tape) {
    int const batch = PyList_Check(object) || PyTuple_Check(object);
    PyObject* const items = batch ? PySequence_Fast(object, "expected a sequence of shapes") : PyTuple_Pack(1, object);
    if (!items) return -1;
    Py_ssize_t const items_count = PySequence_Fast_GET_SIZE(items);
    PyObject* const* const item_objects = PySequence_Fast_ITEMS(items);
    usearch_matrix_view_t points;
    usearch_py_buffer_backing_t backing;
    usearch_scalar_t scalar = scalar_hint;
    size_t total_points = 0;
    for (Py_ssize_t position = 0; position != items_count; ++position) {
        if (shape_view(item_objects[position], scalar_hint, &points, &backing) != 0) goto failed;
        usearch_py_buffer_release(&backing);
        // One tape holds one scalar type, so a batch mixing them has no single reading.
        if (position && points.scalar != scalar) {
            usearch_py_raise(usearch_status_shape_mismatch_k, "shape %zd holds another scalar type than the first",
                             position);
            goto failed;
        }
        scalar = points.scalar;
        total_points += points.rows;
    }

    size_t const row_bytes = usearch_py_scalar_row_bytes(scalar, 3);
    uint64_t* const offsets = PyMem_Malloc(((size_t)items_count + 1) * sizeof(uint64_t) + total_points * row_bytes);
    if (!offsets) {
        PyErr_NoMemory();
        goto failed;
    }
    *block = offsets;
    uint8_t* const bytes = (uint8_t*)(offsets + items_count + 1);
    offsets[0] = 0;
    for (Py_ssize_t position = 0; position != items_count; ++position) {
        if (shape_view(item_objects[position], scalar, &points, &backing) != 0) goto failed;
        size_t const begin = (size_t)offsets[position];
        if (points.rows > total_points - begin) {
            usearch_py_buffer_release(&backing);
            usearch_py_raise(usearch_status_shape_mismatch_k, "shape %zd grew while the batch was gathered", position);
            goto failed;
        }
        for (size_t row = 0; row != points.rows; ++row)
            memcpy(bytes + (begin + row) * row_bytes, (uint8_t const*)points.data + row * points.stride, row_bytes);
        offsets[position + 1] = begin + points.rows;
        usearch_py_buffer_release(&backing);
    }
    Py_DECREF(items);
    *count = (size_t)items_count;
    *tape = (usearch_tape_view_t) {.data = bytes, .layout = usearch_tape_u64_k, .scalar = scalar};
    tape->offsets.u64 = offsets;
    return 0;

failed:
    Py_DECREF(items);
    return -1;
}

#pragma endregion Handles

#pragma region Construction

static char const doc_ShapesIndex[] =                                                              //
    "ShapesIndex\n"                                                                                //
    "\n"                                                                                           //
    "Similarity search over 3D point clouds of any size, such as molecules, conformers, and\n"     //
    "protein backbones, under Chamfer, Hausdorff, Wasserstein-Procrustes, and\n"                   //
    "Gromov-Wasserstein, every kind invariant to rotation, reflection and translation.\n"          //
    "\n"                                                                                           //
    "Every verb is a batch, and one shape is a batch of one. A shape is a `(points, 3)` matrix\n"  //
    "of x, y and z, and a batch is a list of them, each with its own point count.\n"               //
    "\n"                                                                                           //
    "Start an empty one in the residency its entries live in, `init_ram` or `init_vram`, copy a\n" //
    "checkpoint in with `load_ram` or `load_vram`, or serve one in place with `view_ram` or\n"     //
    "`view_nvme`, which are read-only.\n"                                                          //
    "\n"                                                                                           //
    "The receiver chooses where a verb runs, and the verb stays the same: the index runs it\n"     //
    "on the calling thread, `on(executor)` blocks on that executor, and `deferred(executor)`\n"    //
    "submits it there and answers a `Deferred`. See `usearch.Executor`.";

/** The geometry both creation verbs freeze, read before the factory's residency arm. */
typedef struct shapes_geometry_t {
    usearch_shapes_metric_t metric;
    usearch_scalar_t quantization;
    size_t points;
    size_t connectivity;
    size_t values_per_key;
} shapes_geometry_t;

/** The keyword a verb names its class ceiling by: @c cpu_class_ceiling for host placements,
 *  @c gpu_class_ceiling for devices. */
static char const* ceiling_name(usearch_py_enum_t const* ceiling_classes) {
    return ceiling_classes == &usearch_py_gpu_class ? "gpu_class_ceiling" : "cpu_class_ceiling";
}

/**
 *  @brief Reads the geometry both creation verbs take: the metric and quantization positionally or
 *      by name, the rest by name, and the residency arm @p arm_name into @p arm.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int geometry_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                          PyObject* args_names_tuple, char const* arm_name, PyObject** arm,
                          usearch_py_enum_t const* ceiling_classes, int* ceiling, shapes_geometry_t* geometry) {
    PyObject *metric_object, *quantization_object, *points_object, *connectivity_object, *values_per_key_object;
    PyObject* ceiling_object;
    usearch_py_argument_t const arguments[] = {
        {"metric", &metric_object},
        {"quantization", &quantization_object},
        {"points", &points_object},
        {"connectivity", &connectivity_object},
        {"values_per_key", &values_per_key_object},
        {arm_name, arm},
        {ceiling_name(ceiling_classes), &ceiling_object},
    };
    memset(geometry, 0, sizeof(*geometry));
    int metric = usearch_shapes_metric_chamfer_k;
    if (usearch_py_arguments_parse(verb, args, positional_args_count, args_names_tuple, 0, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        (metric_object &&
         usearch_py_enum_from_object(metric_object, "metric", &usearch_py_shapes_metric, &metric) != 0) ||
        usearch_py_scalar_from_object(quantization_object, "quantization", &geometry->quantization) != 0 ||
        usearch_py_size_from_object(points_object, "points", &geometry->points) != 0 ||
        usearch_py_size_from_object(connectivity_object, "connectivity", &geometry->connectivity) != 0 ||
        usearch_py_size_from_object(values_per_key_object, "values_per_key", &geometry->values_per_key) != 0 ||
        usearch_py_class_from_object(ceiling_object, ceiling_name(ceiling_classes), ceiling_classes, ceiling) != 0)
        return -1;
    geometry->metric = (usearch_shapes_metric_t)metric;
    if (geometry->quantization == usearch_scalar_unknown_k) geometry->quantization = usearch_scalar_f32_k;
    return 0;
}

/**
 *  @brief Wraps the handle a lifecycle verb produced, or frees it when it failed or the wrapper
 *      cannot be allocated.
 *  @return A new reference, or NULL with an exception set.
 */
static PyObject* index_adopt(PyTypeObject* type, usearch_shapes_index_t handle, usearch_outcome_t const* outcome) {
    if (usearch_py_raise_on_outcome(outcome) != 0) {
        usearch_shapes_free(handle);
        return NULL;
    }
    ShapesIndex* const instance = (ShapesIndex*)type->tp_alloc(type, 0);
    if (!instance) {
        usearch_shapes_free(handle);
        return NULL;
    }
    instance->handle = handle;
    return (PyObject*)instance;
}

static PyObject* ShapesIndex_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_TypeError, "ShapesIndex is built through init_*(), load_*(), view_ram() or view_nvme()");
    return NULL;
}

static char const doc_ShapesIndex_init_ram[] =                                                         //
    "``init_ram(metric=ShapesMetric.CHAMFER, quantization=Scalar.F32, *, points=0, connectivity=0,\n"  //
    "values_per_key=0, memory_domains=None, cpu_class_ceiling=None) -> ShapesIndex``\n"                //
    "\n"                                                                                               //
    "Starts an empty, mutable graph in host memory, one shard per named NUMA node.\n"                  //
    "Calls `usearch_shapes_init_ram`.\n"                                                               //
    "\n"                                                                                               //
    "Args:\n"                                                                                          //
    "  metric (ShapesMetric, optional): How two shapes are scored.\n"                                  //
    "  quantization (Scalar, optional): The scalar type points are stored in.\n"                       //
    "  points (int, optional): The frozen point count every entry shares; 0 admits any count.\n"       //
    "  connectivity (int, optional): Connections per node; 0 derives it.\n"                            //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                     //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to shard across; None derives\n"        //
    "    every node.\n"                                                                                //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n"      //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.\n"            //
    "\n"                                                                                               //
    "Example:\n"                                                                                       //
    "  >>> from array import array\n"                                                                  //
    "  >>> index = usearch.ShapesIndex.init_ram(usearch.ShapesMetric.CHAMFER)\n"                       //
    "  >>> points = memoryview(array('f', [0, 0, 0, 1, 0, 0, 0, 1, 0])).cast('B').cast('f', [3, 3])\n" //
    "  >>> index.insert([7], points)\n"                                                                //
    "  >>> len(index)\n"                                                                               //
    "  1";

static PyObject* ShapesIndex_init_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    shapes_geometry_t geometry;
    PyObject* memory_domains_object;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    int cpu_class_ceiling = 0;
    if (geometry_parse("init_ram", args, positional_args_count, args_names_tuple, "memory_domains",
                       &memory_domains_object, &usearch_py_cpu_class, &cpu_class_ceiling, &geometry) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        return NULL;

    usearch_shapes_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_shapes_init_ram_t arguments = {
        .index = &handle,
        .outcome = &outcome,
        .metric = geometry.metric,
        .quantization = geometry.quantization,
        .points = geometry.points,
        .connectivity = geometry.connectivity,
        .values_per_key = geometry.values_per_key,
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_init_ram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);
    return index_adopt(type, handle, &outcome);
}

static char const doc_ShapesIndex_init_vram[] =                                                        //
    "``init_vram(metric=ShapesMetric.CHAMFER, quantization=Scalar.F32, *, points=0, connectivity=0,\n" //
    "values_per_key=0, devices=None, gpu_class_ceiling=None) -> ShapesIndex``\n"                       //
    "\n"                                                                                               //
    "Starts an empty, mutable graph on GPU devices, one shard per named ordinal.\n"                    //
    "Calls `usearch_shapes_init_vram`.\n"                                                              //
    "\n"                                                                                               //
    "Args:\n"                                                                                          //
    "  metric (ShapesMetric, optional): How two shapes are scored.\n"                                  //
    "  quantization (Scalar, optional): The scalar type points are stored in.\n"                       //
    "  points (int, optional): The frozen point count every entry shares; 0 admits any count.\n"       //
    "  connectivity (int, optional): Connections per node; 0 derives it.\n"                            //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                     //
    "  devices (Sequence[int], optional): GPU ordinals to shard across.\n"                             //
    "  gpu_class_ceiling (GPUClass, optional): The newest class dispatch may pick, capped below\n"     //
    "    the best class every device loads, else unsupported; None means no ceiling.";

static PyObject* ShapesIndex_init_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                       PyObject* args_names_tuple) {
    shapes_geometry_t geometry;
    PyObject* devices_object;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    int gpu_class_ceiling = 0;
    if (geometry_parse("init_vram", args, positional_args_count, args_names_tuple, "devices", &devices_object,
                       &usearch_py_gpu_class, &gpu_class_ceiling, &geometry) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        return NULL;

    usearch_shapes_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_shapes_init_vram_t arguments = {
        .index = &handle,
        .outcome = &outcome,
        .metric = geometry.metric,
        .quantization = geometry.quantization,
        .points = geometry.points,
        .connectivity = geometry.connectivity,
        .values_per_key = geometry.values_per_key,
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_init_vram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(devices);
    return index_adopt(type, handle, &outcome);
}

static void ShapesIndex_dealloc(ShapesIndex* self) {
    // Releasing a large graph walks every shard, which no other thread should wait on.
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_free(self->handle);
    Py_END_ALLOW_THREADS;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static char const doc_ShapesIndex_on[] =                                                       //
    "on(executor) -> On[ShapesIndex]\n"                                                        //
    "\n"                                                                                       //
    "A receiver whose verbs block on `executor`, refused while another call holds it.\n"       //
    "It offers every verb the index does, with the same arguments and answers. Python-only.\n" //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  executor (Executor): The workers the calls run on; its residency must match.";

static char const doc_ShapesIndex_deferred[] =                                                  //
    "deferred(executor) -> DeferredOn[ShapesIndex]\n"                                           //
    "\n"                                                                                        //
    "A receiver whose verbs submit their call to `executor` and answer a `Deferred` at once,\n" //
    "with the same arguments the index takes. It holds the index and the executor until\n"      //
    "every `Deferred` it answered is gone. Python-only.\n"                                      //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  executor (Executor): The workers the calls run on; its residency must match.";

static PyObject* ShapesIndex_on(PyObject* self, PyObject* executor) {
    return usearch_py_on_new(&OnType, self, executor);
}

static PyObject* ShapesIndex_deferred(PyObject* self, PyObject* executor) {
    return usearch_py_on_new(&DeferredOnType, self, executor);
}

#pragma endregion Construction

#pragma region Reports

static PyStructSequence_Field ShapesMetadata_fields[] = {
    {"metric", "The `ShapesMetric` the checkpoint scores with."},
    {"quantization", "The `Scalar` its points are stored in."},
    {"points", "The frozen point count every entry shares, or 0 for any count."},
    {"connectivity", "Connections per node the graph was built with."},
    {NULL, NULL},
};

PyStructSequence_Desc usearch_py_shapes_metadata_desc = {
    "usearch.ShapesMetadata", "The metric and geometry a checkpoint froze, as `usearch_shapes_metadata_t` holds them.",
    ShapesMetadata_fields, 4};

PyTypeObject ShapesMetadataType;

/** A new @c ShapesMetadata of @p metadata, or NULL with an exception set. */
static PyObject* metadata_to_object(usearch_shapes_metadata_t const* metadata) {
    PyObject* fields[4] = {NULL, NULL, NULL, NULL};
    (void)((fields[0] = usearch_py_enum_to_object(&usearch_py_shapes_metric, (int)metadata->metric)) &&
           (fields[1] = usearch_py_enum_to_object(&usearch_py_scalar, (int)metadata->quantization)) &&
           (fields[2] = PyLong_FromSize_t(metadata->points)) &&
           (fields[3] = PyLong_FromSize_t(metadata->connectivity)));
    return usearch_py_struct_sequence_steal(&ShapesMetadataType, fields, 4);
}

static char const doc_ShapesIndexDescription[] =                                            //
    "ShapesIndexDescription\n"                                                              //
    "\n"                                                                                    //
    "A live index's frozen metadata and current counters, as `describe` read them in one\n" //
    "call.";

static PyObject* ShapesIndexDescription_get_metadata(ShapesIndexDescription* self, void* closure) {
    return metadata_to_object(&self->description.metadata);
}

static PyObject* ShapesIndexDescription_get_cpu_class(ShapesIndexDescription* self, void* closure) {
    return usearch_py_class_to_object(&usearch_py_cpu_class, (int)self->description.cpu_class);
}

static PyObject* ShapesIndexDescription_get_gpu_class(ShapesIndexDescription* self, void* closure) {
    return usearch_py_class_to_object(&usearch_py_gpu_class, (int)self->description.gpu_class);
}

static PyObject* ShapesIndexDescription_get_load_path(ShapesIndexDescription* self, void* closure) {
    return PyUnicode_FromString(usearch_load_path_name(self->description.load_path));
}

static PyGetSetDef ShapesIndexDescription_getset[] = {
    {"metadata", (getter)ShapesIndexDescription_get_metadata, NULL,
     "The frozen `ShapesMetadata`, as a checkpoint reports it.", NULL},
    {"size", usearch_py_size_getter, NULL, "Stored entries.",
     (void*)offsetof(ShapesIndexDescription, description.size)},
    {"capacity", usearch_py_size_getter, NULL, "Entries hosted without growing again.",
     (void*)offsetof(ShapesIndexDescription, description.capacity)},
    {"total_points", usearch_py_size_getter, NULL, "Points summed across every stored entry.",
     (void*)offsetof(ShapesIndexDescription, description.total_points)},
    {"memory_usage", usearch_py_size_getter, NULL, "Bytes the handle holds; traversal scratch is the executor's.",
     (void*)offsetof(ShapesIndexDescription, description.memory_usage)},
    {"cpu_class", (getter)ShapesIndexDescription_get_cpu_class, NULL,
     "The `CPUClass` dispatch picked on this machine, or None.", NULL},
    {"gpu_class", (getter)ShapesIndexDescription_get_gpu_class, NULL,
     "The `GPUClass` dispatch picked on the index's devices, or None for a host index.", NULL},
    {"shards_count", usearch_py_size_getter, NULL, "Live shards.",
     (void*)offsetof(ShapesIndexDescription, description.shards_count)},
    {"load_path", (getter)ShapesIndexDescription_get_load_path, NULL,
     "How the last load from a path read the parts onto the devices: none, host-copy or cufile.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject* ShapesIndexDescription_repr(ShapesIndexDescription* self) {
    return PyUnicode_FromFormat("ShapesIndexDescription(metric=%s, size=%zu, capacity=%zu)",
                                usearch_shapes_metric_name(self->description.metadata.metric), self->description.size,
                                self->description.capacity);
}

PyTypeObject ShapesIndexDescriptionType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.ShapesIndexDescription",
    .tp_doc = doc_ShapesIndexDescription,
    .tp_basicsize = sizeof(ShapesIndexDescription),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_repr = (reprfunc)ShapesIndexDescription_repr,
    .tp_getset = ShapesIndexDescription_getset,
};

#pragma endregion Reports

#pragma region Introspection

/**
 *  @brief Reads the frozen metric and current counters of @p handle on the calling thread.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int index_describe(usearch_shapes_index_t handle, usearch_shapes_index_description_t* description) {
    memset(description, 0, sizeof(*description));
    usearch_outcome_t outcome = {0};
    usearch_shapes_describe_t arguments = {.index = handle, .outcome = &outcome, .description = description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_describe(&arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_raise_on_outcome(&outcome);
}

static char const doc_ShapesIndex_describe[] =                                               //
    "describe() -> ShapesIndexDescription\n"                                                 //
    "\n"                                                                                     //
    "The frozen metadata and the current counters, read in one call. Each counter is read\n" //
    "once, so under concurrent insertions they may come from slightly different moments.\n"  //
    "Calls `usearch_shapes_describe`.";

static PyObject* ShapesIndex_describe(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_shapes_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    ShapesIndexDescription* const description = usearch_py_call_result(call, &ShapesIndexDescriptionType);
    if (!description) return usearch_py_call_discard(&receiver, call);
    usearch_shapes_describe_t* const arguments = (usearch_shapes_describe_t*)call->arguments;
    *arguments = (usearch_shapes_describe_t) {.index = handle,
                                              .executor = receiver.executor,
                                              .outcome = &call->outcome,
                                              .completion = receiver.completion,
                                              .description = &description->description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_describe(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_ShapesIndex_health[] =                                              //
    "health() -> DenseReachability\n"                                                     //
    "\n"                                                                                  //
    "Audits how much of the graph a search can reach, every shard swept from its first\n" //
    "entry, forward and back. Read-only.\n"                                               //
    "Calls `usearch_shapes_health`.";

static PyObject* ShapesIndex_health(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_shapes_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    DenseReachability* const audit = usearch_py_call_result(call, &DenseReachabilityType);
    if (!audit) return usearch_py_call_discard(&receiver, call);
    usearch_shapes_health_t* const arguments = (usearch_shapes_health_t*)call->arguments;
    *arguments = (usearch_shapes_health_t) {.index = handle,
                                            .executor = receiver.executor,
                                            .outcome = &call->outcome,
                                            .completion = receiver.completion,
                                            .reachability = &audit->reachability};
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_health(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

/** Reads the @c size_t counter at the byte offset @p closure names inside a description. */
static PyObject* ShapesIndex_get_counter(ShapesIndex* self, void* closure) {
    usearch_shapes_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return PyLong_FromSize_t(*(size_t const*)((char const*)&description + (size_t)closure));
}

static PyObject* ShapesIndex_get_cpu_class(ShapesIndex* self, void* closure) {
    usearch_shapes_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return usearch_py_class_to_object(&usearch_py_cpu_class, (int)description.cpu_class);
}

static PyObject* ShapesIndex_get_gpu_class(ShapesIndex* self, void* closure) {
    usearch_shapes_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return usearch_py_class_to_object(&usearch_py_gpu_class, (int)description.gpu_class);
}

static Py_ssize_t ShapesIndex_length(ShapesIndex* self) {
    usearch_shapes_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return -1;
    return (Py_ssize_t)description.size;
}

static PyObject* ShapesIndex_repr(ShapesIndex* self) {
    // No exception is raised here: `tp_repr` runs inside `PyErr_Format("%R", ...)` chains, where
    // raising would destroy the caller's.
    usearch_shapes_index_description_t description = {0};
    usearch_outcome_t outcome = {0};
    usearch_shapes_describe_t arguments = {.index = self->handle, .outcome = &outcome, .description = &description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_describe(&arguments);
    Py_END_ALLOW_THREADS;
    if (outcome.status != usearch_status_success_k) return PyUnicode_FromString("ShapesIndex(<unavailable>)");
    return PyUnicode_FromFormat("ShapesIndex(metric=%s, size=%zu)",
                                usearch_shapes_metric_name(description.metadata.metric), description.size);
}

#pragma endregion Introspection

#pragma region Ingest

static char const doc_ShapesIndex_reserve[] =                               //
    "``reserve(capacity, /, *, points=0) -> None``\n"                       //
    "\n"                                                                    //
    "Grows the index to host at least `capacity` entries, a hint, since\n"  //
    "insertions grow the index on demand.\n"                                //
    "Calls `usearch_shapes_reserve`.\n"                                     //
    "\n"                                                                    //
    "Args:\n"                                                               //
    "  capacity (int): Entries the index must accept without growing.\n"    //
    "  points (int, optional): Expected total points across every entry;\n" //
    "    0 grows lazily.";

static PyObject* ShapesIndex_reserve(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                     PyObject* args_names_tuple) {
    PyObject *capacity_object, *points_object;
    usearch_py_argument_t const arguments[] = {{"capacity", &capacity_object}, {"points", &points_object}};
    usearch_py_receiver_t receiver;
    usearch_shapes_index_t handle = NULL;
    size_t capacity = 0, points = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("reserve", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(capacity_object, "capacity", &capacity) != 0 ||
        usearch_py_size_from_object(points_object, "points", &points) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    usearch_shapes_reserve_t* const reserve = (usearch_shapes_reserve_t*)call->arguments;
    *reserve = (usearch_shapes_reserve_t) {.index = handle,
                                           .executor = receiver.executor,
                                           .outcome = &call->outcome,
                                           .completion = receiver.completion,
                                           .entries = capacity,
                                           .points = points};
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_reserve(reserve);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_ShapesIndex_clear[] =                            //
    "clear() -> None\n"                                                //
    "\n"                                                               //
    "Drops every entry, keeping the capacity and the frozen metric.\n" //
    "Calls `usearch_shapes_clear`.";

static PyObject* ShapesIndex_clear(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_shapes_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    usearch_shapes_clear_t* const arguments = (usearch_shapes_clear_t*)call->arguments;
    *arguments = (usearch_shapes_clear_t) {.index = handle,
                                           .executor = receiver.executor,
                                           .outcome = &call->outcome,
                                           .completion = receiver.completion};
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_clear(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_ShapesIndex_insert[] =                                                          //
    "``insert(keys, shapes, /, *, expansion_factor=0, dtype=None, progress=None,\n"                   //
    "progress_steps=0, stats=None) -> None``\n"                                                       //
    "\n"                                                                                              //
    "Inserts a batch of shapes under a batch of keys; one shape is a batch of one. Keys need\n"       //
    "not be unique, so three conformers of one molecule may share its key, and a batch is\n"          //
    "all-or-nothing and becomes visible at once.\n"                                                   //
    "Calls `usearch_shapes_insert`.\n"                                                                //
    "\n"                                                                                              //
    "Args:\n"                                                                                         //
    "  keys (Sequence[Key | int] | buffer): One key per shape, or an N by 16 buffer.\n"               //
    "  shapes (Sequence[buffer] | buffer): `(points, 3)` matrices of x, y and z, each with its own\n" //
    "    point count, gathered once into one tape.\n"                                                 //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"                  //
    "  dtype (Scalar, optional): Names the point scalar type when the buffers cannot.\n"              //
    "  progress (callable, optional): Called with (completed, total) entries; what it raises\n"       //
    "    goes to `sys.unraisablehook`.\n"                                                             //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"           //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.";

static PyObject* ShapesIndex_insert(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                    PyObject* args_names_tuple) {
    PyObject *keys_object, *shapes_object, *expansion_object, *dtype_object, *progress_object, *progress_steps_object,
        *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"keys", &keys_object},   {"shapes", &shapes_object},     {"expansion_factor", &expansion_object},
        {"dtype", &dtype_object}, {"progress", &progress_object}, {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_shapes_index_t handle = NULL;
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

    size_t count = 0;
    usearch_tape_view_t shapes;
    if (shapes_from_object(shapes_object, dtype, &call->blocks[0], &count, &shapes) != 0 ||
        usearch_py_keys_from_object(keys_object, count, &call->keys[0]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0)
        return usearch_py_call_discard(&receiver, call);

    usearch_shapes_insert_t* const insert = (usearch_shapes_insert_t*)call->arguments;
    *insert = (usearch_shapes_insert_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .count = count,
        .keys = {call->keys[0].keys, sizeof(usearch_key_t)},
        .shapes = shapes,
        .expansion_factor = expansion_factor,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_insert(insert);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_ShapesIndex_refine[] =                                                      //
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
    "Calls `usearch_shapes_refine`.\n"                                                            //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"              //
    "  memory_limit (int, optional): Bytes a loaded NVMe handle holds for the part it\n"          //
    "    rewrites; 0 derives what rounds that part in RAM.\n"                                     //
    "  progress (callable, optional): Called with (completed, total) units of each phase;\n"      //
    "    returning False stops the round, None continues.\n"                                      //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"       //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"          //
    "  plan (ShapesRefinePlan, optional): The round to open or continue; None runs one\n"         //
    "    round to its end.\n"                                                                     //
    "\n"                                                                                          //
    "Returns:\n"                                                                                  //
    "  RefineReport: What the round did to the graph, and its `RefineVerdict`.";

static PyObject* ShapesIndex_refine(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
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
    usearch_shapes_index_t handle = NULL;
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
        plan_parse(call, plan_object, &ShapesRefinePlanType) != 0)
        return usearch_py_call_discard(&receiver, call);

    ShapesPlan* const plan = (ShapesPlan*)call->holds[callback_slot_k];
    usearch_shapes_refine_t* const refine = (usearch_shapes_refine_t*)call->arguments;
    *refine = (usearch_shapes_refine_t) {
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
    usearch_shapes_refine(refine);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Ingest

#pragma region Search

static char const doc_ShapesIndex_search[] =                                                     //
    "``search(queries, wanted, /, arena=None, *, expansion_factor=0,\n"                          //
    "traversal=Traversal.WALK, dtype=None, predicate=None, progress=None,\n"                     //
    "progress_steps=0, stats=None) -> Matches``\n"                                               //
    "\n"                                                                                         //
    "Finds the `wanted` nearest entries for every query shape; one query is a batch of one.\n"   //
    "Query point counts need not match any entry's. Never writes to the graph, so any number\n"  //
    "of searches may overlap with insertions.\n"                                                 //
    "Calls `usearch_shapes_search`.\n"                                                           //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  queries (Sequence[buffer] | buffer): The query shapes, `(points, 3)` matrices.\n"         //
    "  wanted (int): Upper bound on matches per query, the 'k' in 'kANN'.\n"                     //
    "  arena (Arena, optional): Where the matches land, rewound by the call; None makes one\n"   //
    "    the result owns.\n"                                                                     //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"             //
    "  traversal (Traversal, optional): `Traversal.WALK`, the default, walks the graph;\n"       //
    "    `Traversal.SCAN` scores every entry exactly.\n"                                         //
    "  dtype (Scalar, optional): Names the point scalar type when the buffers cannot.\n"         //
    "  predicate (callable, optional): Admits a Key when it returns truthy; it re-enters\n"      //
    "    Python, so it runs under the GIL; what it raises goes to `sys.unraisablehook`, and\n"   //
    "    the key is rejected.\n"                                                                 //
    "    A `Batch` answers many keys per call instead.\n"                                        //
    "  progress (callable, optional): Called with (completed, total) queries; returning False\n" //
    "    stops the batch and raises `CancelledError`, None continues.\n"                         //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"      //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"         //
    "\n"                                                                                         //
    "Returns:\n"                                                                                 //
    "  Matches: `(Q, columns)` runs in the arena, each count bounding its own row.";

static PyObject* ShapesIndex_search(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
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
    usearch_shapes_index_t handle = NULL;
    size_t wanted = 0, expansion_factor = 0, progress_steps = 0;
    int traversal = usearch_traversal_walk_k;
    usearch_scalar_t dtype = usearch_scalar_unknown_k;
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
        usearch_py_scalar_from_object(dtype_object, "dtype", &dtype) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    size_t count = 0;
    usearch_tape_view_t queries;
    Matches* matches = NULL;
    if (shapes_from_object(queries_object, dtype, &call->blocks[0], &count, &queries) != 0 ||
        usearch_py_predicate_from_object(predicate_object, &call->holds[callback_slot_k]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0 ||
        !(matches = usearch_py_call_arena(&receiver, call, arena_object, &MatchesType)))
        return usearch_py_call_discard(&receiver, call);

    matches->queries = count;
    usearch_shapes_search_t* const search = (usearch_shapes_search_t*)call->arguments;
    *search = (usearch_shapes_search_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .count = count,
        .queries = queries,
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
    usearch_shapes_search(search);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Search

#pragma region Retrieval

static char const doc_ShapesIndex_count[] =                            //
    "count(key) -> int\n"                                              //
    "\n"                                                               //
    "How many entries share `key`; zero means the index holds none.\n" //
    "Calls `usearch_shapes_count`.\n"                                  //
    "\n"                                                               //
    "Args:\n"                                                          //
    "  key (Key | int | str): The identity to count.";

/** A @c usearch_shapes_count call and the statistics it lands, inside the record's arguments. */
typedef struct count_call_t {
    usearch_shapes_count_t arguments;
    usearch_shapes_key_stats_t key_stats;
} count_call_t;

static PyObject* count_answer(usearch_py_call_t* call) {
    return PyLong_FromSize_t(((count_call_t const*)call->arguments)->key_stats.entries);
}

static PyObject* ShapesIndex_count(PyObject* self, PyObject* key_object) {
    usearch_py_receiver_t receiver;
    usearch_shapes_index_t handle = NULL;
    usearch_key_t key;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || usearch_py_key_from_object(key_object, &key) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    count_call_t* const count = (count_call_t*)call->arguments;
    call->answer = count_answer;
    count->arguments = (usearch_shapes_count_t) {.index = handle,
                                                 .executor = receiver.executor,
                                                 .outcome = &call->outcome,
                                                 .completion = receiver.completion,
                                                 .key = key,
                                                 .key_stats = &count->key_stats};
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_count(&count->arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static int ShapesIndex_contains(PyObject* self, PyObject* key_object) {
    PyObject* const entries = ShapesIndex_count(self, key_object);
    if (!entries) return -1;
    int const present = PyObject_IsTrue(entries);
    Py_DECREF(entries);
    return present;
}

static char const doc_ShapesIndex_get[] =                                                            //
    "``get(key, /, arena=None, *, outputs_dtype=Scalar.F32) -> tuple[buffer, buffer]``\n"            //
    "\n"                                                                                             //
    "Every shape stored under `key`, in insertion order and in the pose it was inserted in, as an\n" //
    "Apache Arrow tape in the arena: the `(count + 1,)` point offsets and a `(points, 3)` matrix\n"  //
    "of every entry's points end to end, entry `i` spanning rows `offsets[i]` up to\n"               //
    "`offsets[i + 1]`. Both are empty when the key is absent.\n"                                     //
    "Calls `usearch_shapes_get`.\n"                                                                  //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  key (Key | int | str): The identity to retrieve.\n"                                           //
    "  arena (Arena, optional): Where the tape lands, rewound by the call; None makes one the\n"     //
    "    result owns.\n"                                                                             //
    "  outputs_dtype (Scalar, optional): The scalar type to export as; converted from storage.";

/** A @c usearch_shapes_get call and the tape it lands, inside the record's arguments. */
typedef struct get_call_t {
    usearch_shapes_get_t arguments;
    usearch_tape_view_t shapes;
    size_t written;
} get_call_t;

static PyObject* get_answer(usearch_py_call_t* call) {
    // The offsets size the views, read here on the host before any view guards them.
    if (usearch_py_refuse_host_read(call->arena) != 0) return NULL;
    get_call_t const* const get = (get_call_t const*)call->arguments;
    usearch_tape_view_t const* const tape = &get->shapes;
    int const narrow = tape->layout == usearch_tape_u32_k;
    void const* const offsets = narrow ? (void const*)tape->offsets.u32 : (void const*)tape->offsets.u64;
    // An absent key leaves the tape's pointers NULL, so there is no offset run to view.
    size_t const first = !offsets ? 0 : narrow ? tape->offsets.u32[0] : tape->offsets.u64[0];
    size_t const last = !offsets ? 0 : narrow ? tape->offsets.u32[get->written] : tape->offsets.u64[get->written];
    usearch_scalar_t const scalar = get->arguments.outputs_scalar;
    char const* const format = usearch_py_scalar_buffer_format(scalar);
    size_t const row_bytes = usearch_py_scalar_row_bytes(scalar, 3);
    Py_ssize_t const offsets_shape[1] = {offsets ? (Py_ssize_t)get->written + 1 : 0};
    Py_ssize_t const points_shape[2] = {(Py_ssize_t)(last - first), format ? 3 : (Py_ssize_t)row_bytes};
    PyObject* const offsets_object = usearch_py_arena_buffer(&ArenaBufferType, call->arena, call->generation, offsets,
                                                             narrow ? "I" : "Q", narrow ? 4 : 8, offsets_shape, 1);
    PyObject* const points_object = offsets_object ? usearch_py_arena_buffer(
                                                         &ArenaBufferType, call->arena, call->generation, tape->data,
                                                         format ? format : "B",
                                                         format ? (Py_ssize_t)(row_bytes / 3) : 1, points_shape, 2)
                                                   : NULL;
    PyObject* const pair = points_object ? PyTuple_Pack(2, offsets_object, points_object) : NULL;
    Py_XDECREF(points_object);
    Py_XDECREF(offsets_object);
    return pair;
}

static PyObject* ShapesIndex_get(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                 PyObject* args_names_tuple) {
    PyObject *key_object, *arena_object, *outputs_dtype_object;
    usearch_py_argument_t const arguments[] = {{"key", &key_object},
                                               {"arena", &arena_object},
                                               {"outputs_dtype", &outputs_dtype_object}};
    usearch_py_receiver_t receiver;
    usearch_shapes_index_t handle = NULL;
    usearch_scalar_t outputs_kind = usearch_scalar_unknown_k;
    usearch_key_t key;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("get", args, positional_args_count, args_names_tuple, 1, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_scalar_from_object(outputs_dtype_object, "outputs_dtype", &outputs_kind) != 0 ||
        usearch_py_key_from_object(key_object, &key) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    if (usearch_py_call_hold_arena(&receiver, call, arena_object) != 0) return usearch_py_call_discard(&receiver, call);
    get_call_t* const get = (get_call_t*)call->arguments;
    call->answer = get_answer;
    get->arguments = (usearch_shapes_get_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .key = key,
        .shapes = &get->shapes,
        .written = &get->written,
        .arena = usearch_py_arena_handle(call->arena),
        .outputs_scalar = outputs_kind == usearch_scalar_unknown_k ? usearch_scalar_f32_k : outputs_kind,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_get(&get->arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Retrieval

#pragma region Composition

static char const doc_ShapesIndex_reshard[] =                                                    //
    "``reshard(sources, targets, /, *, expansion_factor=0, memory_limit=0, seed=0,\n"            //
    "progress=None, progress_steps=0, stats=None, plan=None) -> RefineReport``\n"                //
    "\n"                                                                                         //
    "Moves the entries of `sources` into `targets`, keeping their links rather than\n"           //
    "re-inserting their shapes, then runs a round on every target that took entries. Several\n"  //
    "sources into one target, itself among them, gather into it; one source into several\n"      //
    "targets cuts it into one graph-local part per target. Every index stays the caller's,\n"    //
    "and a source the targets do not name ends empty and live. The receiver names where the\n"   //
    "call runs, and need not be among them.\n"                                                   //
    "\n"                                                                                         //
    "The handler stops the closing rounds alone, never a move: a stop raises `CancelledError`\n" //
    "with every entry moved, or, with a `plan`, leaves it live for the next call naming it to\n" //
    "continue, until `reshard_plan_finish(plan)` closes it. Calls `usearch_shapes_reshard`.\n"   //
    "\n"                                                                                         //
    "A live plan holds no index, so each index it names must outlive its finish.\n"              //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  sources (Sequence[ShapesIndex]): The indexes whose entries move, each named once.\n"      //
    "  targets (Sequence[ShapesIndex]): The indexes they move into, each a source or empty.\n"   //
    "  expansion_factor (int, optional): Breadth of any walk beside the rounds; 0 derives it.\n" //
    "  memory_limit (int, optional): Bytes a loaded NVMe handle holds for the part it\n"         //
    "    rewrites; 0 derives what rounds that part in RAM.\n"                                    //
    "  seed (int, optional): Picks the part roots of a cut; equal seeds cut equally.\n"          //
    "  progress (callable, optional): Called with (completed, total) units of each phase of\n"   //
    "    the closing rounds; returning False stops them, None continues.\n"                      //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"      //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"         //
    "  plan (ShapesReshardPlan, optional): The reshard to open or continue; None runs it to\n"   //
    "    its end.\n"                                                                             //
    "\n"                                                                                         //
    "Returns:\n"                                                                                 //
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
    usearch_shapes_index_t* const block = PyMem_Malloc((length ? (size_t)length : 1) * sizeof(usearch_shapes_index_t));
    if (!block) return PyErr_NoMemory(), -1;
    *handles = block;
    for (Py_ssize_t position = 0; position != length; ++position) {
        PyObject* const index = PySequence_Fast_GET_ITEM(sequence, position);
        if (!Py_IS_TYPE(index, &ShapesIndexType)) {
            PyErr_Format(PyExc_TypeError, "the '%s' argument must hold ShapesIndex objects, got %S", field,
                         Py_TYPE(index));
            return -1;
        }
        block[position] = ((ShapesIndex*)index)->handle;
    }
    *count = (size_t)length;
    return 0;
}

static PyObject* ShapesIndex_reshard(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
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
    usearch_shapes_index_t handle = NULL;
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
        plan_parse(call, plan_object, &ShapesReshardPlanType) != 0)
        return usearch_py_call_discard(&receiver, call);

    ShapesPlan* const plan = (ShapesPlan*)call->holds[callback_slot_k];
    usearch_shapes_reshard_t* const reshard = (usearch_shapes_reshard_t*)call->arguments;
    *reshard = (usearch_shapes_reshard_t) {
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
    usearch_shapes_reshard(reshard);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Composition

#pragma region Plans

static char const doc_ShapesRefinePlan[] =                                                         //
    "ShapesRefinePlan()\n"                                                                         //
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

static char const doc_ShapesReshardPlan[] =                                                         //
    "ShapesReshardPlan()\n"                                                                         //
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

static PyObject* ShapesPlan_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    if (PyTuple_GET_SIZE(args) != 0 || (keywords && PyDict_Size(keywords) != 0)) {
        PyErr_Format(PyExc_TypeError, "%s() takes no arguments", type->tp_name + sizeof("usearch.") - 1);
        return NULL;
    }
    return type->tp_alloc(type, 0);
}

static PyObject* ShapesPlan_get_phase(ShapesPlan* self, void* closure) {
    return usearch_py_enum_to_object(&usearch_py_phase, (int)self->progress.phase);
}

static PyGetSetDef ShapesPlan_getset[] = {
    {"phase", (getter)ShapesPlan_get_phase, NULL, "Where the last call naming the plan stopped, as a `Phase`.", NULL},
    {"ordinal", usearch_py_size_getter, NULL, "That phase's position among the ones the plan runs.",
     (void*)offsetof(ShapesPlan, progress.ordinal)},
    {"completed", usearch_py_size_getter, NULL, "Units of that phase done, entries or islands.",
     (void*)offsetof(ShapesPlan, progress.completed)},
    {"total", usearch_py_size_getter, NULL, "Units of that phase in all.", (void*)offsetof(ShapesPlan, progress.total)},
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject ShapesRefinePlanType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.ShapesRefinePlan",
    .tp_doc = doc_ShapesRefinePlan,
    .tp_basicsize = sizeof(ShapesPlan),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = ShapesPlan_tp_new,
    .tp_getset = ShapesPlan_getset,
};

PyTypeObject ShapesReshardPlanType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.ShapesReshardPlan",
    .tp_doc = doc_ShapesReshardPlan,
    .tp_basicsize = sizeof(ShapesPlan),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = ShapesPlan_tp_new,
    .tp_getset = ShapesPlan_getset,
};

static char const doc_ShapesIndex_refine_plan_finish[] =                                        //
    "refine_plan_finish(plan) -> RefineReport\n"                                                //
    "\n"                                                                                        //
    "Closes a `ShapesRefinePlan` on this index and the receiver's executor, the ones it was\n"  //
    "opened with, and frees it whatever the call answers; a round a stop left open closes as\n" //
    "a stop at that barrier would. Calls `usearch_shapes_refine_plan_finish`.\n"                //
    "\n"                                                                                        //
    "Returns:\n"                                                                                //
    "  RefineReport: What the whole plan did to the graph.";

static char const doc_ShapesIndex_reshard_plan_finish[] =                                        //
    "reshard_plan_finish(plan) -> RefineReport\n"                                                //
    "\n"                                                                                         //
    "Closes a `ShapesReshardPlan` on the receiver's executor, the one it was opened with, and\n" //
    "frees it whatever the call answers, reopening every index it names; a plan whose entries\n" //
    "have not all moved runs no round and reports zeros. Calls\n"                                //
    "`usearch_shapes_reshard_plan_finish`.\n"                                                    //
    "\n"                                                                                         //
    "Returns:\n"                                                                                 //
    "  RefineReport: What the whole plan did to the graph.";

/** Opens a finish of the plan @p plan_object of @p type through @p self: the report it writes into,
 *  the plan held, and its handle taken, since the library frees it whatever the call answers. */
static usearch_py_call_t* plan_finish_open(PyObject* self, PyObject* plan_object, PyTypeObject* type,
                                           usearch_py_receiver_t* receiver, usearch_shapes_index_t* handle,
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

static PyObject* ShapesIndex_refine_plan_finish(PyObject* self, PyObject* plan_object) {
    usearch_py_receiver_t receiver;
    usearch_shapes_index_t handle = NULL;
    usearch_py_call_t stack;
    RefineReport* report_call = NULL;
    usearch_py_call_t* const call = plan_finish_open(self, plan_object, &ShapesRefinePlanType, &receiver, &handle,
                                                     &stack, &report_call);
    if (!call) return NULL;
    ShapesPlan* const plan = (ShapesPlan*)plan_object;
    usearch_shapes_refine_plan_finish_t* const finish = (usearch_shapes_refine_plan_finish_t*)call->arguments;
    *finish = (usearch_shapes_refine_plan_finish_t) {.index = handle,
                                                     .executor = receiver.executor,
                                                     .outcome = &call->outcome,
                                                     .completion = receiver.completion,
                                                     .plan = plan->handle.refine,
                                                     .report = &report_call->report};
    plan->handle.refine = NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_refine_plan_finish(finish);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static PyObject* ShapesIndex_reshard_plan_finish(PyObject* self, PyObject* plan_object) {
    usearch_py_receiver_t receiver;
    usearch_shapes_index_t handle = NULL;
    usearch_py_call_t stack;
    RefineReport* report_call = NULL;
    usearch_py_call_t* const call = plan_finish_open(self, plan_object, &ShapesReshardPlanType, &receiver, &handle,
                                                     &stack, &report_call);
    if (!call) return NULL;
    ShapesPlan* const plan = (ShapesPlan*)plan_object;
    usearch_shapes_reshard_plan_finish_t* const finish = (usearch_shapes_reshard_plan_finish_t*)call->arguments;
    *finish = (usearch_shapes_reshard_plan_finish_t) {.executor = receiver.executor,
                                                      .outcome = &call->outcome,
                                                      .completion = receiver.completion,
                                                      .plan = plan->handle.reshard,
                                                      .report = &report_call->report};
    plan->handle.reshard = NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_reshard_plan_finish(finish);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Plans

#pragma region Serialization

static char const doc_ShapesIndex_save[] =                                                  //
    "``save(path, /, *, progress=None, progress_steps=0) -> None``\n"                       //
    "\n"                                                                                    //
    "Writes the index under `path`, in the layout its extension picks: one `.safegraphs`\n" //
    "monolith, or a `.safegraphs.json` manifest beside one part per shard. Demands a\n"     //
    "mutation-free window; concurrent readers keep searching throughout.\n"                 //
    "Calls `usearch_shapes_save`.\n"                                                        //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  path (str | bytes | os.PathLike): Where the checkpoint goes.\n"                      //
    "  progress (callable, optional): Called with (completed, total) parts; returning\n"    //
    "    False stops the write and raises `CancelledError`.\n"                              //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject* ShapesIndex_save(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                  PyObject* args_names_tuple) {
    PyObject *path_object, *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {{"path", &path_object},
                                               {"progress", &progress_object},
                                               {"progress_steps", &progress_steps_object}};
    usearch_py_receiver_t receiver;
    usearch_shapes_index_t handle = NULL;
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
    usearch_shapes_save_t* const save = (usearch_shapes_save_t*)call->arguments;
    *save = (usearch_shapes_save_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .path = path,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_save(save);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_ShapesIndex_save_buffer[] =                                             //
    "``save_buffer(arena=None, *, progress=None, progress_steps=0) -> Monolith``\n"           //
    "\n"                                                                                      //
    "Serializes the index into the arena: one monolith for a handle of one shard, the very\n" //
    "bytes `save` writes, or a manifest leading one part per shard.\n"                        //
    "Calls `usearch_shapes_save_buffer`.\n"                                                   //
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

/** A @c usearch_shapes_save_buffer call and the monolith it lands, inside the arguments. */
typedef struct save_buffer_call_t {
    usearch_shapes_save_buffer_t arguments;
    usearch_bytes_view_t monolith;
} save_buffer_call_t;

static PyObject* save_buffer_answer(usearch_py_call_t* call) {
    usearch_bytes_view_t const monolith = ((save_buffer_call_t const*)call->arguments)->monolith;
    Py_ssize_t const shape[1] = {(Py_ssize_t)monolith.length};
    return usearch_py_arena_buffer(&MonolithType, call->arena, call->generation, monolith.data, "B", 1, shape, 1);
}

static PyObject* ShapesIndex_save_buffer(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                         PyObject* args_names_tuple) {
    PyObject *arena_object, *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {{"arena", &arena_object},
                                               {"progress", &progress_object},
                                               {"progress_steps", &progress_steps_object}};
    usearch_py_receiver_t receiver;
    usearch_shapes_index_t handle = NULL;
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
    save->arguments = (usearch_shapes_save_buffer_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .monolith = &save->monolith,
        .arena = usearch_py_arena_handle(call->arena),
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_save_buffer(&save->arguments);
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

static char const doc_ShapesIndex_load_ram[] =                                                      //
    "``load_ram(path, /, *, memory_domains=None, executor=None, values_per_key=0, progress=None,\n" //
    "progress_steps=0, cpu_class_ceiling=None) -> ShapesIndex``\n"                                  //
    "\n"                                                                                            //
    "Copies a checkpoint into a new, mutable index in host memory, its geometry read from the\n"    //
    "checkpoint's own headers, every shape turned onto its principal axes again. Name the\n"        //
    "`.safegraphs.json` manifest, never one of its parts.\n"                                        //
    "Calls `usearch_shapes_load_ram`.\n"                                                            //
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

static PyObject* ShapesIndex_load_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
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

    usearch_shapes_index_t handle = NULL;
    usearch_shapes_load_ram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_load_ram(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(memory_domains);
    Py_XDECREF(decoded);
    Py_XDECREF(progress);
    return result;
}

static char const doc_ShapesIndex_load_ram_buffer[] =                                             //
    "``load_ram_buffer(buffer, /, *, memory_domains=None, executor=None, values_per_key=0,\n"     //
    "progress=None, progress_steps=0, cpu_class_ceiling=None) -> ShapesIndex``\n"                 //
    "\n"                                                                                          //
    "Copies a checkpoint out of a buffer into a new, mutable index in host memory, as\n"          //
    "`load_ram` does from a path.\n"                                                              //
    "Calls `usearch_shapes_load_ram_buffer`.\n"                                                   //
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

static PyObject* ShapesIndex_load_ram_buffer(PyTypeObject* type, PyObject* const* args,
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

    usearch_shapes_index_t handle = NULL;
    usearch_shapes_load_ram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_load_ram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(memory_domains);
    usearch_py_buffer_release(&backing);
    Py_XDECREF(progress);
    return result;
}

static char const doc_ShapesIndex_load_vram[] =                                                    //
    "``load_vram(path, /, *, devices=None, executor=None, values_per_key=0, progress=None,\n"      //
    "progress_steps=0, gpu_class_ceiling=None) -> ShapesIndex``\n"                                 //
    "\n"                                                                                           //
    "Copies a checkpoint into a new, mutable index on GPU devices, with the placement rules\n"     //
    "of `load_ram`.\n"                                                                             //
    "Calls `usearch_shapes_load_vram`.\n"                                                          //
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

static PyObject* ShapesIndex_load_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
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

    usearch_shapes_index_t handle = NULL;
    usearch_shapes_load_vram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_load_vram(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(devices);
    Py_XDECREF(decoded);
    Py_XDECREF(progress);
    return result;
}

static char const doc_ShapesIndex_load_vram_buffer[] =                                             //
    "``load_vram_buffer(buffer, /, *, devices=None, executor=None, values_per_key=0,\n"            //
    "progress=None, progress_steps=0, gpu_class_ceiling=None) -> ShapesIndex``\n"                  //
    "\n"                                                                                           //
    "Copies a checkpoint out of a host buffer into a new, mutable index on GPU devices.\n"         //
    "Calls `usearch_shapes_load_vram_buffer`.\n"                                                   //
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

static PyObject* ShapesIndex_load_vram_buffer(PyTypeObject* type, PyObject* const* args,
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

    usearch_shapes_index_t handle = NULL;
    usearch_shapes_load_vram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
        .values_per_key = values_per_key,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_load_vram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(devices);
    usearch_py_buffer_release(&backing);
    Py_XDECREF(progress);
    return result;
}

static char const doc_ShapesIndex_view_ram[] =                                                    //
    "``view_ram(path, /, *, cpu_class_ceiling=None) -> ShapesIndex``\n"                           //
    "\n"                                                                                          //
    "Serves a checkpoint in place through a memory mapping, as a read-only index whose every\n"   //
    "mutating verb raises `ReadOnlyError`.\n"                                                     //
    "Calls `usearch_shapes_view_ram`.\n"                                                          //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                   //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n" //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.";

static PyObject* ShapesIndex_view_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
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

    usearch_shapes_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_shapes_view_ram_t view_arguments = {
        .index = &handle,
        .outcome = &outcome,
        .path = path,
        .ram = {NULL, 0, 0, (usearch_cpu_class_t)cpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_view_ram(&view_arguments);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    return index_adopt(type, handle, &outcome);
}

static char const doc_ShapesIndex_view_nvme[] =                                                     //
    "``view_nvme(path, /, *, paths=None, executor=None, cpu_class_ceiling=None) -> ShapesIndex``\n" //
    "\n"                                                                                            //
    "Serves a checkpoint in place from NVMe drives as a read-only index, every later call\n"        //
    "reading it through its executor's engines.\n"                                                  //
    "Calls `usearch_shapes_view_nvme`.\n"                                                           //
    "\n"                                                                                            //
    "Args:\n"                                                                                       //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                     //
    "  paths (Sequence[str | bytes | PathLike], optional): Devices or directories to serve\n"       //
    "    the parts from; None serves each where its manifest points.\n"                             //
    "  executor (Executor, optional): The NVMe executor whose rings read the headers.\n"            //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n"   //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.";

static PyObject* ShapesIndex_view_nvme(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
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

    usearch_shapes_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_shapes_view_nvme_t view_arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .nvme = {paths.views, paths.count, (usearch_cpu_class_t)cpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_view_nvme(&view_arguments);
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

static char const doc_ShapesIndex_metadata_read[] =                                      //
    "``metadata_read(path, /, *, executor=None) -> ShapesMetadata``\n"                   //
    "\n"                                                                                 //
    "Reads the metric and geometry a checkpoint froze, without loading it.\n"            //
    "Calls `usearch_shapes_metadata_read`.\n"                                            //
    "\n"                                                                                 //
    "Args:\n"                                                                            //
    "  path (str | bytes | os.PathLike): The checkpoint to inspect.\n"                   //
    "  executor (Executor, optional): The workers the read runs on; None reads on the\n" //
    "    calling thread.";

static PyObject* ShapesIndex_metadata_read(PyObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                           PyObject* args_names_tuple) {
    PyObject *path_object, *decoded = NULL;
    usearch_executor_t executor = NULL;
    usearch_bytes_view_t path;
    if (metadata_parse("metadata_read", args, positional_args_count, args_names_tuple, "path", &path_object,
                       &executor) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0)
        return NULL;
    usearch_shapes_metadata_t metadata = {0};
    usearch_outcome_t outcome = {0};
    usearch_shapes_metadata_read_t arguments = {
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .metadata = &metadata,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_metadata_read(&arguments);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return metadata_to_object(&metadata);
}

static char const doc_ShapesIndex_metadata_read_buffer[] =                               //
    "``metadata_read_buffer(buffer, /, *, executor=None) -> ShapesMetadata``\n"          //
    "\n"                                                                                 //
    "Reads a serialized checkpoint's metric and geometry, as `metadata_read`\n"          //
    "does from a path.\n"                                                                //
    "Calls `usearch_shapes_metadata_read_buffer`.\n"                                     //
    "\n"                                                                                 //
    "Args:\n"                                                                            //
    "  buffer (buffer): One serialized checkpoint, such as `save_buffer` returns.\n"     //
    "  executor (Executor, optional): The workers the read runs on; None reads on the\n" //
    "    calling thread.";

static PyObject* ShapesIndex_metadata_read_buffer(PyObject* type, PyObject* const* args,
                                                  Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    PyObject* buffer_object;
    usearch_executor_t executor = NULL;
    usearch_py_buffer_backing_t backing;
    if (metadata_parse("metadata_read_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                       &executor) != 0 ||
        usearch_py_checkpoint_from_object(buffer_object, &backing) != 0)
        return NULL;
    usearch_shapes_metadata_t metadata = {0};
    usearch_outcome_t outcome = {0};
    usearch_shapes_metadata_read_buffer_t arguments = {
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .metadata = &metadata,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_shapes_metadata_read_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    usearch_py_buffer_release(&backing);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return metadata_to_object(&metadata);
}

#pragma endregion Serialization

#pragma region Registration

static PyMethodDef ShapesIndex_methods[] = {
    {"init_ram", (PyCFunction)ShapesIndex_init_ram, usearch_py_method_flags_k | METH_CLASS, doc_ShapesIndex_init_ram},
    {"init_vram", (PyCFunction)ShapesIndex_init_vram, usearch_py_method_flags_k | METH_CLASS,
     doc_ShapesIndex_init_vram},
    {"load_ram", (PyCFunction)ShapesIndex_load_ram, usearch_py_method_flags_k | METH_CLASS, doc_ShapesIndex_load_ram},
    {"load_ram_buffer", (PyCFunction)ShapesIndex_load_ram_buffer, usearch_py_method_flags_k | METH_CLASS,
     doc_ShapesIndex_load_ram_buffer},
    {"load_vram", (PyCFunction)ShapesIndex_load_vram, usearch_py_method_flags_k | METH_CLASS,
     doc_ShapesIndex_load_vram},
    {"load_vram_buffer", (PyCFunction)ShapesIndex_load_vram_buffer, usearch_py_method_flags_k | METH_CLASS,
     doc_ShapesIndex_load_vram_buffer},
    {"view_ram", (PyCFunction)ShapesIndex_view_ram, usearch_py_method_flags_k | METH_CLASS, doc_ShapesIndex_view_ram},
    {"view_nvme", (PyCFunction)ShapesIndex_view_nvme, usearch_py_method_flags_k | METH_CLASS,
     doc_ShapesIndex_view_nvme},
    {"metadata_read", (PyCFunction)ShapesIndex_metadata_read, usearch_py_method_flags_k | METH_STATIC,
     doc_ShapesIndex_metadata_read},
    {"metadata_read_buffer", (PyCFunction)ShapesIndex_metadata_read_buffer, usearch_py_method_flags_k | METH_STATIC,
     doc_ShapesIndex_metadata_read_buffer},
    {"on", (PyCFunction)ShapesIndex_on, METH_O, doc_ShapesIndex_on},
    {"deferred", (PyCFunction)ShapesIndex_deferred, METH_O, doc_ShapesIndex_deferred},
    {"describe", (PyCFunction)ShapesIndex_describe, METH_NOARGS, doc_ShapesIndex_describe},
    {"reserve", (PyCFunction)ShapesIndex_reserve, usearch_py_method_flags_k, doc_ShapesIndex_reserve},
    {"clear", (PyCFunction)ShapesIndex_clear, METH_NOARGS, doc_ShapesIndex_clear},
    {"insert", (PyCFunction)ShapesIndex_insert, usearch_py_method_flags_k, doc_ShapesIndex_insert},
    {"refine", (PyCFunction)ShapesIndex_refine, usearch_py_method_flags_k, doc_ShapesIndex_refine},
    {"health", (PyCFunction)ShapesIndex_health, METH_NOARGS, doc_ShapesIndex_health},
    {"search", (PyCFunction)ShapesIndex_search, usearch_py_method_flags_k, doc_ShapesIndex_search},
    {"count", (PyCFunction)ShapesIndex_count, METH_O, doc_ShapesIndex_count},
    {"get", (PyCFunction)ShapesIndex_get, usearch_py_method_flags_k, doc_ShapesIndex_get},
    {"reshard", (PyCFunction)ShapesIndex_reshard, usearch_py_method_flags_k, doc_ShapesIndex_reshard},
    {"refine_plan_finish", (PyCFunction)ShapesIndex_refine_plan_finish, METH_O, doc_ShapesIndex_refine_plan_finish},
    {"reshard_plan_finish", (PyCFunction)ShapesIndex_reshard_plan_finish, METH_O, doc_ShapesIndex_reshard_plan_finish},
    {"save", (PyCFunction)ShapesIndex_save, usearch_py_method_flags_k, doc_ShapesIndex_save},
    {"save_buffer", (PyCFunction)ShapesIndex_save_buffer, usearch_py_method_flags_k, doc_ShapesIndex_save_buffer},
    {NULL, NULL, 0, NULL},
};

/** Every verb a receiver runs on its executor, the same for an @c On and a @c DeferredOn, since no
 *  shapes verb answers an iterator. */
static PyMethodDef ShapesIndex_receiver_methods[] = {
    {"describe", (PyCFunction)ShapesIndex_describe, METH_NOARGS, doc_ShapesIndex_describe},
    {"reserve", (PyCFunction)ShapesIndex_reserve, usearch_py_method_flags_k, doc_ShapesIndex_reserve},
    {"clear", (PyCFunction)ShapesIndex_clear, METH_NOARGS, doc_ShapesIndex_clear},
    {"insert", (PyCFunction)ShapesIndex_insert, usearch_py_method_flags_k, doc_ShapesIndex_insert},
    {"refine", (PyCFunction)ShapesIndex_refine, usearch_py_method_flags_k, doc_ShapesIndex_refine},
    {"health", (PyCFunction)ShapesIndex_health, METH_NOARGS, doc_ShapesIndex_health},
    {"search", (PyCFunction)ShapesIndex_search, usearch_py_method_flags_k, doc_ShapesIndex_search},
    {"count", (PyCFunction)ShapesIndex_count, METH_O, doc_ShapesIndex_count},
    {"get", (PyCFunction)ShapesIndex_get, usearch_py_method_flags_k, doc_ShapesIndex_get},
    {"reshard", (PyCFunction)ShapesIndex_reshard, usearch_py_method_flags_k, doc_ShapesIndex_reshard},
    {"refine_plan_finish", (PyCFunction)ShapesIndex_refine_plan_finish, METH_O, doc_ShapesIndex_refine_plan_finish},
    {"reshard_plan_finish", (PyCFunction)ShapesIndex_reshard_plan_finish, METH_O, doc_ShapesIndex_reshard_plan_finish},
    {"save", (PyCFunction)ShapesIndex_save, usearch_py_method_flags_k, doc_ShapesIndex_save},
    {"save_buffer", (PyCFunction)ShapesIndex_save_buffer, usearch_py_method_flags_k, doc_ShapesIndex_save_buffer},
    {NULL, NULL, 0, NULL},
};

usearch_py_receiver_verbs_t const usearch_py_shapes_receiver_verbs = {&ShapesIndexType, ShapesIndex_receiver_methods,
                                                                      ShapesIndex_receiver_methods};

static PyGetSetDef ShapesIndex_getset[] = {
    {"capacity", (getter)ShapesIndex_get_counter, NULL,
     "Entries the index hosts without growing again. Calls `usearch_shapes_describe`.",
     (void*)offsetof(usearch_shapes_index_description_t, capacity)},
    {"connectivity", (getter)ShapesIndex_get_counter, NULL,
     "Connections per node in the graph. Calls `usearch_shapes_describe`.",
     (void*)offsetof(usearch_shapes_index_description_t, metadata.connectivity)},
    {"memory_usage", (getter)ShapesIndex_get_counter, NULL,
     "Bytes the handle holds; traversal scratch is the executor's. Calls `usearch_shapes_describe`.",
     (void*)offsetof(usearch_shapes_index_description_t, memory_usage)},
    {"cpu_class", (getter)ShapesIndex_get_cpu_class, NULL,
     "The `CPUClass` dispatch picked on this machine, or None. Calls `usearch_shapes_describe`.", NULL},
    {"gpu_class", (getter)ShapesIndex_get_gpu_class, NULL,
     "The `GPUClass` dispatch picked on the index's devices, or None. Calls `usearch_shapes_describe`.", NULL},
    {"shards_count", (getter)ShapesIndex_get_counter, NULL, "Live shards. Calls `usearch_shapes_describe`.",
     (void*)offsetof(usearch_shapes_index_description_t, shards_count)},
    {NULL, NULL, NULL, NULL, NULL},
};

static PySequenceMethods ShapesIndex_as_sequence = {
    .sq_length = (lenfunc)ShapesIndex_length,
    .sq_contains = (objobjproc)ShapesIndex_contains,
};

PyTypeObject ShapesIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.ShapesIndex",
    .tp_doc = doc_ShapesIndex,
    .tp_basicsize = sizeof(ShapesIndex),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = ShapesIndex_tp_new,
    .tp_dealloc = (destructor)ShapesIndex_dealloc,
    .tp_repr = (reprfunc)ShapesIndex_repr,
    .tp_methods = ShapesIndex_methods,
    .tp_getset = ShapesIndex_getset,
    .tp_as_sequence = &ShapesIndex_as_sequence,
};

#pragma endregion Registration
