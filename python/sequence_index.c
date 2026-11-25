/**
 *  @file python/sequence_index.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief CPython similarity search over variable-length sequences.
 *
 *  Entries vary in size, so batches arrive as tapes rather than strided matrices: a list of text
 *  or bytes-like objects is gathered once into one run of bytes and the offsets around it, and the
 *  handle holds no dimensions or quantization.
 *
 *  Every verb is written once and reached through three receivers, as in @c dense_index.c: the
 *  index runs it on the calling thread, an @c On blocks on its executor, and a @c DeferredOn
 *  submits it there and answers a @c Deferred. A verb fills one call record and opens its result
 *  as the slot the library writes into, so the same body serves every completion.
 */
#include "lib.h"

#pragma region Handles

/** An owned index handle. */
typedef struct SequenceIndex {
    PyObject ob_base;
    usearch_sequence_index_t handle;
} SequenceIndex;

/** A live index's frozen metadata and current counters, the slot @c describe writes them into. */
typedef struct SequenceIndexDescription {
    PyObject ob_base;
    usearch_sequence_index_description_t description;
} SequenceIndexDescription;

/** A refine or a reshard in flight, continued by every call naming it until its finish verb. */
typedef struct SequencePlan {
    PyObject ob_base;

    /** The engine plan, NULL until a call opens it and once its finish verb closed it. */
    union {
        usearch_sequence_refine_plan_t refine;
        usearch_sequence_reshard_plan_t reshard;
    } handle;

    /** Where the last call naming the plan stopped. */
    usearch_phase_progress_t progress;
} SequencePlan;

/** The holds every sequence call keeps in the same slots. */
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
static int receiver_from_object(PyObject* self, usearch_py_receiver_t* receiver, usearch_sequence_index_t* handle) {
    if (usearch_py_receiver_from_object(self, receiver) != 0) return -1;
    if (!Py_IS_TYPE(receiver->index, &SequenceIndexType)) {
        PyErr_Format(PyExc_TypeError, "expected a receiver over a SequenceIndex, got one over %S",
                     Py_TYPE(receiver->index));
        return -1;
    }
    *handle = ((SequenceIndex*)receiver->index)->handle;
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

/** Views the bytes of one sequence: text as UTF-8, anything else through the buffer protocol. */
static int sequence_view(PyObject* item, Py_buffer* buffer) {
    if (PyUnicode_Check(item)) {
        Py_ssize_t length = 0;
        char const* const text = PyUnicode_AsUTF8AndSize(item, &length);
        return text ? PyBuffer_FillInfo(buffer, item, (void*)text, length, 1, PyBUF_SIMPLE) : -1;
    }
    if (PyObject_GetBuffer(item, buffer, PyBUF_C_CONTIGUOUS) == 0) return 0;
    return usearch_py_raise_from_cause(PyExc_TypeError, "a sequence is a str or a contiguous bytes-like object, got %S",
                                       Py_TYPE(item));
}

/**
 *  @brief Gathers text, a bytes-like object, or a sequence of them into one tape; a lone sequence
 *      is a tape of one.
 *
 *  The items are separate objects, so their bytes are copied once into the run the library reads:
 *  the @c count + 1 offsets, then every sequence's bytes end to end, in one block passed to
 *  @p block. The lengths are read in a first pass and checked again while copying, since another
 *  thread may resize a byte array in between.
 *
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int sequences_from_object(PyObject* object, void** block, size_t* count, usearch_tape_view_t* tape) {
    int const lone = PyUnicode_Check(object) || PyObject_CheckBuffer(object);
    PyObject* const items = lone ? PyTuple_Pack(1, object)
                                 : PySequence_Fast(object, "expected a sequence of sequences");
    if (!items) return -1;
    Py_ssize_t const items_count = PySequence_Fast_GET_SIZE(items);
    PyObject* const* const item_objects = PySequence_Fast_ITEMS(items);
    size_t total = 0;
    Py_buffer buffer;
    for (Py_ssize_t position = 0; position != items_count; ++position) {
        if (sequence_view(item_objects[position], &buffer) != 0) goto failed;
        total += (size_t)buffer.len;
        PyBuffer_Release(&buffer);
    }

    uint64_t* const offsets = PyMem_Malloc(((size_t)items_count + 1) * sizeof(uint64_t) + total);
    if (!offsets) {
        PyErr_NoMemory();
        goto failed;
    }
    *block = offsets;
    uint8_t* const bytes = (uint8_t*)(offsets + items_count + 1);
    offsets[0] = 0;
    for (Py_ssize_t position = 0; position != items_count; ++position) {
        if (sequence_view(item_objects[position], &buffer) != 0) goto failed;
        size_t const begin = (size_t)offsets[position];
        if ((size_t)buffer.len > total - begin) {
            PyBuffer_Release(&buffer);
            usearch_py_raise(usearch_status_shape_mismatch_k, "sequence %zd grew while the batch was gathered",
                             position);
            goto failed;
        }
        if (buffer.len) memcpy(bytes + begin, buffer.buf, (size_t)buffer.len);
        offsets[position + 1] = begin + (size_t)buffer.len;
        PyBuffer_Release(&buffer);
    }
    Py_DECREF(items);
    *count = (size_t)items_count;
    *tape = (usearch_tape_view_t) {.data = bytes, .layout = usearch_tape_u64_k, .scalar = usearch_scalar_u8_k};
    tape->offsets.u64 = offsets;
    return 0;

failed:
    Py_DECREF(items);
    return -1;
}

/** The weights an alignment kind scores with, and the matrix buffer they view. */
typedef struct weights_t {
    usearch_sequence_weights_t values;
    usearch_py_buffer_backing_t matrix;
} weights_t;

/**
 *  @brief Reads the @p matrix_object and @p gap_object an alignment kind takes, the matrix viewed
 *      as 256 × 256 signed bytes.
 *  @param[out] passed The weights to hand the library, or NULL when the caller named neither.
 *  @return 0 on success, -1 with an exception set otherwise; the matrix is released either way by
 *      @c usearch_py_buffer_release on @c weights->matrix.
 */
static int weights_parse(PyObject* matrix_object, PyObject* gap_object, weights_t* weights,
                         usearch_sequence_weights_t const** passed) {
    memset(weights, 0, sizeof(*weights));
    *passed = NULL;
    int const matrix_named = matrix_object && matrix_object != Py_None;
    int const gap_named = gap_object && gap_object != Py_None;
    if (!matrix_named && !gap_named) return 0;
    if (gap_named) {
        long const gap = PyLong_AsLong(gap_object);
        if (gap == -1 && PyErr_Occurred()) return -1;
        if (gap < INT8_MIN || gap > INT8_MAX) {
            PyErr_Format(PyExc_OverflowError, "the 'gap' option must fit a signed byte, got %ld", gap);
            return -1;
        }
        weights->values.gap = (int8_t)gap;
    }
    if (matrix_named) {
        if (PyObject_GetBuffer(matrix_object, &weights->matrix.buffer, PyBUF_C_CONTIGUOUS) != 0)
            return usearch_py_raise_from_cause(PyExc_TypeError,
                                               "the 'substitution_matrix' must be a contiguous buffer");
        if (weights->matrix.buffer.len != 256 * 256) {
            Py_ssize_t const length = weights->matrix.buffer.len;
            usearch_py_buffer_release(&weights->matrix);
            return usearch_py_raise(usearch_status_shape_mismatch_k,
                                    "the 'substitution_matrix' holds 256 × 256 signed bytes, got %zd bytes", length);
        }
        weights->values.substitution_matrix = (int8_t const*)weights->matrix.buffer.buf;
    }
    *passed = &weights->values;
    return 0;
}

/** Reads a sequence of non-negative widths into a caller-freed array; NULL or None reads none. */
static int widths_from_object(PyObject* object, size_t** widths, size_t* count) {
    *widths = NULL;
    *count = 0;
    if (!object || object == Py_None) return 0;
    PyObject* const sequence = PySequence_Fast(object, "the 'window_widths' option must be a sequence of integers");
    if (!sequence) return -1;
    Py_ssize_t const length = PySequence_Fast_GET_SIZE(sequence);
    *widths = (size_t*)PyMem_Malloc((length ? (size_t)length : 1) * sizeof(size_t));
    if (!*widths) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t position = 0; position != length; ++position) {
        if (usearch_py_size_from_object(PySequence_Fast_GET_ITEM(sequence, position), "window_widths",
                                        &(*widths)[position]) != 0) {
            Py_DECREF(sequence);
            PyMem_Free(*widths);
            *widths = NULL;
            return -1;
        }
    }
    Py_DECREF(sequence);
    *count = (size_t)length;
    return 0;
}

#pragma endregion Handles

#pragma region Construction

static char const doc_SequenceIndex[] =                                                              //
    "SequenceIndex\n"                                                                                //
    "\n"                                                                                             //
    "Similarity search over variable-length sequences: textual documents, DNA, RNA, and proteins,\n" //
    "under Levenshtein edits, window overlap, and Needleman-Wunsch alignment.\n"                     //
    "\n"                                                                                             //
    "Every verb is a batch, and one sequence is a batch of one. A sequence is a `str`, read as\n"    //
    "UTF-8, or any contiguous bytes-like object, and a batch is a list of them.\n"                   //
    "\n"                                                                                             //
    "Start an empty one in the residency its entries live in, `init_ram` or `init_vram`, copy a\n"   //
    "checkpoint in with `load_ram` or `load_vram`, or serve one in place with `view_ram` or\n"       //
    "`view_nvme`, which are read-only.\n"                                                            //
    "\n"                                                                                             //
    "The receiver chooses where a verb runs, and the verb stays the same: the index runs it\n"       //
    "on the calling thread, `on(executor)` blocks on that executor, and `deferred(executor)`\n"      //
    "submits it there and answers a `Deferred`. See `usearch.Executor`.\n"                           //
    "\n"                                                                                             //
    "Example:\n"                                                                                     //
    "  >>> index = usearch.SequenceIndex.init_ram(usearch.SequenceMetric.LEVENSHTEIN)\n"             //
    "  >>> index.insert([7], ['kitten'])\n"                                                          //
    "  >>> key, distance = index.search(['sitten'], 1, usearch.Arena())[0][0]\n"                     //
    "  >>> int(key), distance\n"                                                                     //
    "  (7, 1.0)";

/** The configuration both creation verbs freeze, read before the factory's residency arm. */
typedef struct sequence_options_t {
    usearch_sequence_metric_t metric;
    weights_t weights;

    /** The weights to hand the library, or NULL when the caller named none. */
    usearch_sequence_weights_t const* passed_weights;

    /** The overlap kind's widths, a @c PyMem_Free -able array, or NULL. */
    size_t* window_widths;
    size_t window_widths_count;
    size_t connectivity;
    size_t values_per_key;
} sequence_options_t;

static void options_release(sequence_options_t* options) {
    PyMem_Free(options->window_widths);
    options->window_widths = NULL;
    usearch_py_buffer_release(&options->weights.matrix);
}

/** The keyword a verb names its class ceiling by: @c cpu_class_ceiling for host placements,
 *  @c gpu_class_ceiling for devices. */
static char const* ceiling_name(usearch_py_enum_t const* ceiling_classes) {
    return ceiling_classes == &usearch_py_gpu_class ? "gpu_class_ceiling" : "cpu_class_ceiling";
}

/**
 *  @brief Reads the configuration both creation verbs take: the metric positionally or by name, the
 *      rest by name, and the residency arm @p arm_name into @p arm.
 *  @return 0 on success, -1 with an exception set and nothing left to release otherwise.
 */
static int options_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                         PyObject* args_names_tuple, char const* arm_name, PyObject** arm,
                         usearch_py_enum_t const* ceiling_classes, int* ceiling, sequence_options_t* options) {
    PyObject *metric_object, *matrix_object, *gap_object, *widths_object, *connectivity_object, *values_per_key_object;
    PyObject* ceiling_object;
    usearch_py_argument_t const arguments[] = {
        {"metric", &metric_object},
        {"substitution_matrix", &matrix_object},
        {"gap", &gap_object},
        {"window_widths", &widths_object},
        {"connectivity", &connectivity_object},
        {"values_per_key", &values_per_key_object},
        {arm_name, arm},
        {ceiling_name(ceiling_classes), &ceiling_object},
    };
    memset(options, 0, sizeof(*options));
    int metric = usearch_sequence_metric_levenshtein_bytes_k;
    if (usearch_py_arguments_parse(verb, args, positional_args_count, args_names_tuple, 0, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        (metric_object &&
         usearch_py_enum_from_object(metric_object, "metric", &usearch_py_sequence_metric, &metric) != 0) ||
        usearch_py_size_from_object(connectivity_object, "connectivity", &options->connectivity) != 0 ||
        usearch_py_size_from_object(values_per_key_object, "values_per_key", &options->values_per_key) != 0 ||
        usearch_py_class_from_object(ceiling_object, ceiling_name(ceiling_classes), ceiling_classes, ceiling) != 0 ||
        widths_from_object(widths_object, &options->window_widths, &options->window_widths_count) != 0)
        return -1;
    if (weights_parse(matrix_object, gap_object, &options->weights, &options->passed_weights) != 0) {
        options_release(options);
        return -1;
    }
    options->metric = (usearch_sequence_metric_t)metric;
    return 0;
}

/**
 *  @brief Wraps the handle a lifecycle verb produced, or frees it when it failed or the wrapper
 *      cannot be allocated.
 *  @return A new reference, or NULL with an exception set.
 */
static PyObject* index_adopt(PyTypeObject* type, usearch_sequence_index_t handle, usearch_outcome_t const* outcome) {
    if (usearch_py_raise_on_outcome(outcome) != 0) {
        usearch_sequence_free(handle);
        return NULL;
    }
    SequenceIndex* const instance = (SequenceIndex*)type->tp_alloc(type, 0);
    if (!instance) {
        usearch_sequence_free(handle);
        return NULL;
    }
    instance->handle = handle;
    return (PyObject*)instance;
}

static PyObject* SequenceIndex_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_TypeError, "SequenceIndex is built through init_*(), load_*(), view_ram() or view_nvme()");
    return NULL;
}

static char const doc_SequenceIndex_init_ram[] =                                                           //
    "``init_ram(metric=SequenceMetric.LEVENSHTEIN, *, substitution_matrix=None, gap=None,\n"               //
    "window_widths=None, connectivity=0, values_per_key=0, memory_domains=None, cpu_class_ceiling=None)\n" //
    "-> SequenceIndex``\n"                                                                                 //
    "\n"                                                                                                   //
    "Starts an empty, mutable graph in host memory, one shard per named NUMA node.\n"                      //
    "Calls `usearch_sequence_init_ram`.\n"                                                                 //
    "\n"                                                                                                   //
    "Args:\n"                                                                                              //
    "  metric (SequenceMetric, optional): How two sequences are scored.\n"                                 //
    "  substitution_matrix (buffer, optional): 256 × 256 signed byte scores the alignment kind\n"          //
    "    reads; the Levenshtein kinds take none.\n"                                                        //
    "  gap (int, optional): The cost of one unaligned symbol, beside the matrix.\n"                        //
    "  window_widths (Sequence[int], optional): Window widths in bytes the overlap kind needs.\n"          //
    "  connectivity (int, optional): Connections per node; 0 derives it.\n"                                //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                         //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to shard across; None derives\n"            //
    "    every node.\n"                                                                                    //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n"          //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.\n"                //
    "\n"                                                                                                   //
    "Example:\n"                                                                                           //
    "  >>> index = usearch.SequenceIndex.init_ram(usearch.SequenceMetric.OVERLAP, window_widths=[3])\n"    //
    "  >>> index.describe().metadata.metric is usearch.SequenceMetric.OVERLAP\n"                           //
    "  True";

static PyObject* SequenceIndex_init_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                        PyObject* args_names_tuple) {
    sequence_options_t options;
    PyObject* memory_domains_object;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    int cpu_class_ceiling = 0;
    if (options_parse("init_ram", args, positional_args_count, args_names_tuple, "memory_domains",
                      &memory_domains_object, &usearch_py_cpu_class, &cpu_class_ceiling, &options) != 0)
        return NULL;
    if (usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0) {
        options_release(&options);
        return NULL;
    }

    usearch_sequence_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_sequence_init_ram_t arguments = {
        .index = &handle,
        .outcome = &outcome,
        .metric = options.metric,
        .weights = options.passed_weights,
        .window_widths = options.window_widths,
        .window_widths_count = options.window_widths_count,
        .connectivity = options.connectivity,
        .values_per_key = options.values_per_key,
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_init_ram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(memory_domains);
    options_release(&options);
    return index_adopt(type, handle, &outcome);
}

static char const doc_SequenceIndex_init_vram[] =                                                   //
    "``init_vram(metric=SequenceMetric.LEVENSHTEIN, *, substitution_matrix=None, gap=None,\n"       //
    "window_widths=None, connectivity=0, values_per_key=0, devices=None, gpu_class_ceiling=None)\n" //
    "-> SequenceIndex``\n"                                                                          //
    "\n"                                                                                            //
    "Starts an empty, mutable graph on GPU devices, one shard per named ordinal. Entries are\n"     //
    "sharded and never replicated, and an index outgrowing them fails rather than paging to\n"      //
    "the host.\n"                                                                                   //
    "Calls `usearch_sequence_init_vram`.\n"                                                         //
    "\n"                                                                                            //
    "Args:\n"                                                                                       //
    "  metric (SequenceMetric, optional): How two sequences are scored.\n"                          //
    "  substitution_matrix (buffer, optional): 256 × 256 signed byte scores the alignment kind\n"   //
    "    reads.\n"                                                                                  //
    "  gap (int, optional): The cost of one unaligned symbol, beside the matrix.\n"                 //
    "  window_widths (Sequence[int], optional): Window widths in bytes the overlap kind needs.\n"   //
    "  connectivity (int, optional): Connections per node; 0 derives it.\n"                         //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                  //
    "  devices (Sequence[int], optional): GPU ordinals to shard across; None picks device zero.\n"  //
    "  gpu_class_ceiling (GPUClass, optional): The newest class dispatch may pick, capped below\n"  //
    "    the best class every device loads, else unsupported; None means no ceiling.";

static PyObject* SequenceIndex_init_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                         PyObject* args_names_tuple) {
    sequence_options_t options;
    PyObject* devices_object;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    int gpu_class_ceiling = 0;
    if (options_parse("init_vram", args, positional_args_count, args_names_tuple, "devices", &devices_object,
                      &usearch_py_gpu_class, &gpu_class_ceiling, &options) != 0)
        return NULL;
    if (usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0) {
        options_release(&options);
        return NULL;
    }

    usearch_sequence_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_sequence_init_vram_t arguments = {
        .index = &handle,
        .outcome = &outcome,
        .metric = options.metric,
        .weights = options.passed_weights,
        .window_widths = options.window_widths,
        .window_widths_count = options.window_widths_count,
        .connectivity = options.connectivity,
        .values_per_key = options.values_per_key,
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_init_vram(&arguments);
    Py_END_ALLOW_THREADS;
    PyMem_Free(devices);
    options_release(&options);
    return index_adopt(type, handle, &outcome);
}

static void SequenceIndex_dealloc(SequenceIndex* self) {
    // Releasing a large graph walks every shard, which no other thread should wait on.
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_free(self->handle);
    Py_END_ALLOW_THREADS;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static char const doc_SequenceIndex_on[] =                                                     //
    "on(executor) -> On[SequenceIndex]\n"                                                      //
    "\n"                                                                                       //
    "A receiver whose verbs block on `executor`, refused while another call holds it.\n"       //
    "It offers every verb the index does, with the same arguments and answers. Python-only.\n" //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  executor (Executor): The workers the calls run on; its residency must match.";

static char const doc_SequenceIndex_deferred[] =                                                //
    "deferred(executor) -> DeferredOn[SequenceIndex]\n"                                         //
    "\n"                                                                                        //
    "A receiver whose verbs submit their call to `executor` and answer a `Deferred` at once,\n" //
    "with the same arguments the index takes. It holds the index and the executor until\n"      //
    "every `Deferred` it answered is gone. Python-only.\n"                                      //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  executor (Executor): The workers the calls run on; its residency must match.";

static PyObject* SequenceIndex_on(PyObject* self, PyObject* executor) {
    return usearch_py_on_new(&OnType, self, executor);
}

static PyObject* SequenceIndex_deferred(PyObject* self, PyObject* executor) {
    return usearch_py_on_new(&DeferredOnType, self, executor);
}

#pragma endregion Construction

#pragma region Reports

static PyStructSequence_Field SequenceMetadata_fields[] = {
    {"metric", "The `SequenceMetric` the checkpoint scores with."},
    {"connectivity", "Connections per node the graph was built with."},
    {NULL, NULL},
};

PyStructSequence_Desc usearch_py_sequence_metadata_desc = {
    "usearch.SequenceMetadata",
    "The metric and geometry a checkpoint froze, as `usearch_sequence_metadata_t` holds them.", SequenceMetadata_fields,
    2};

PyTypeObject SequenceMetadataType;

/** A new @c SequenceMetadata of @p metadata, or NULL with an exception set. */
static PyObject* metadata_to_object(usearch_sequence_metadata_t const* metadata) {
    PyObject* fields[2] = {NULL, NULL};
    (void)((fields[0] = usearch_py_enum_to_object(&usearch_py_sequence_metric, (int)metadata->metric)) &&
           (fields[1] = PyLong_FromSize_t(metadata->connectivity)));
    return usearch_py_struct_sequence_steal(&SequenceMetadataType, fields, 2);
}

static char const doc_SequenceIndexDescription[] =                                          //
    "SequenceIndexDescription\n"                                                            //
    "\n"                                                                                    //
    "A live index's frozen metadata and current counters, as `describe` read them in one\n" //
    "call.";

static PyObject* SequenceIndexDescription_get_metadata(SequenceIndexDescription* self, void* closure) {
    return metadata_to_object(&self->description.metadata);
}

static PyObject* SequenceIndexDescription_get_cpu_class(SequenceIndexDescription* self, void* closure) {
    return usearch_py_class_to_object(&usearch_py_cpu_class, (int)self->description.cpu_class);
}

static PyObject* SequenceIndexDescription_get_gpu_class(SequenceIndexDescription* self, void* closure) {
    return usearch_py_class_to_object(&usearch_py_gpu_class, (int)self->description.gpu_class);
}

static PyObject* SequenceIndexDescription_get_load_path(SequenceIndexDescription* self, void* closure) {
    return PyUnicode_FromString(usearch_load_path_name(self->description.load_path));
}

static PyGetSetDef SequenceIndexDescription_getset[] = {
    {"metadata", (getter)SequenceIndexDescription_get_metadata, NULL,
     "The frozen `SequenceMetadata`, as a checkpoint reports it.", NULL},
    {"size", usearch_py_size_getter, NULL, "Stored entries.",
     (void*)offsetof(SequenceIndexDescription, description.size)},
    {"capacity", usearch_py_size_getter, NULL, "Entries hosted without growing again.",
     (void*)offsetof(SequenceIndexDescription, description.capacity)},
    {"memory_usage", usearch_py_size_getter, NULL,
     "Bytes the handle holds, payload storage included; traversal scratch is the executor's.",
     (void*)offsetof(SequenceIndexDescription, description.memory_usage)},
    {"cpu_class", (getter)SequenceIndexDescription_get_cpu_class, NULL,
     "The `CPUClass` dispatch picked on this machine, or None.", NULL},
    {"gpu_class", (getter)SequenceIndexDescription_get_gpu_class, NULL,
     "The `GPUClass` dispatch picked on the index's devices, or None for a host index.", NULL},
    {"shards_count", usearch_py_size_getter, NULL, "Live shards.",
     (void*)offsetof(SequenceIndexDescription, description.shards_count)},
    {"load_path", (getter)SequenceIndexDescription_get_load_path, NULL,
     "How the last load from a path read the parts onto the devices: none, host-copy or cufile.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject* SequenceIndexDescription_repr(SequenceIndexDescription* self) {
    return PyUnicode_FromFormat("SequenceIndexDescription(metric=%s, size=%zu, capacity=%zu)",
                                usearch_sequence_metric_name(self->description.metadata.metric), self->description.size,
                                self->description.capacity);
}

PyTypeObject SequenceIndexDescriptionType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.SequenceIndexDescription",
    .tp_doc = doc_SequenceIndexDescription,
    .tp_basicsize = sizeof(SequenceIndexDescription),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_repr = (reprfunc)SequenceIndexDescription_repr,
    .tp_getset = SequenceIndexDescription_getset,
};

#pragma endregion Reports

#pragma region Introspection

/**
 *  @brief Reads the frozen metric and current counters of @p handle on the calling thread.
 *  @return 0 on success, -1 with an exception set otherwise.
 */
static int index_describe(usearch_sequence_index_t handle, usearch_sequence_index_description_t* description) {
    memset(description, 0, sizeof(*description));
    usearch_outcome_t outcome = {0};
    usearch_sequence_describe_t arguments = {.index = handle, .outcome = &outcome, .description = description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_describe(&arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_raise_on_outcome(&outcome);
}

static char const doc_SequenceIndex_describe[] =                                             //
    "describe() -> SequenceIndexDescription\n"                                               //
    "\n"                                                                                     //
    "The frozen metadata and the current counters, read in one call. Each counter is read\n" //
    "once, so under concurrent insertions they may come from slightly different moments.\n"  //
    "Calls `usearch_sequence_describe`.";

static PyObject* SequenceIndex_describe(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    SequenceIndexDescription* const description = usearch_py_call_result(call, &SequenceIndexDescriptionType);
    if (!description) return usearch_py_call_discard(&receiver, call);
    usearch_sequence_describe_t* const arguments = (usearch_sequence_describe_t*)call->arguments;
    *arguments = (usearch_sequence_describe_t) {.index = handle,
                                                .executor = receiver.executor,
                                                .outcome = &call->outcome,
                                                .completion = receiver.completion,
                                                .description = &description->description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_describe(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_SequenceIndex_health[] =                                            //
    "health() -> DenseReachability\n"                                                     //
    "\n"                                                                                  //
    "Audits how much of the graph a search can reach, every shard swept from its first\n" //
    "entry, forward and back. Read-only.\n"                                               //
    "Calls `usearch_sequence_health`.";

static PyObject* SequenceIndex_health(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    DenseReachability* const audit = usearch_py_call_result(call, &DenseReachabilityType);
    if (!audit) return usearch_py_call_discard(&receiver, call);
    usearch_sequence_health_t* const arguments = (usearch_sequence_health_t*)call->arguments;
    *arguments = (usearch_sequence_health_t) {.index = handle,
                                              .executor = receiver.executor,
                                              .outcome = &call->outcome,
                                              .completion = receiver.completion,
                                              .reachability = &audit->reachability};
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_health(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

/** Reads the @c size_t counter at the byte offset @p closure names inside a description. */
static PyObject* SequenceIndex_get_counter(SequenceIndex* self, void* closure) {
    usearch_sequence_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return PyLong_FromSize_t(*(size_t const*)((char const*)&description + (size_t)closure));
}

static PyObject* SequenceIndex_get_cpu_class(SequenceIndex* self, void* closure) {
    usearch_sequence_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return usearch_py_class_to_object(&usearch_py_cpu_class, (int)description.cpu_class);
}

static PyObject* SequenceIndex_get_gpu_class(SequenceIndex* self, void* closure) {
    usearch_sequence_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return NULL;
    return usearch_py_class_to_object(&usearch_py_gpu_class, (int)description.gpu_class);
}

static Py_ssize_t SequenceIndex_length(SequenceIndex* self) {
    usearch_sequence_index_description_t description;
    if (index_describe(self->handle, &description) != 0) return -1;
    return (Py_ssize_t)description.size;
}

static PyObject* SequenceIndex_repr(SequenceIndex* self) {
    // No exception is raised here: `tp_repr` runs inside `PyErr_Format("%R", ...)` chains, where
    // raising would destroy the caller's.
    usearch_sequence_index_description_t description = {0};
    usearch_outcome_t outcome = {0};
    usearch_sequence_describe_t arguments = {.index = self->handle, .outcome = &outcome, .description = &description};
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_describe(&arguments);
    Py_END_ALLOW_THREADS;
    if (outcome.status != usearch_status_success_k) return PyUnicode_FromString("SequenceIndex(<unavailable>)");
    return PyUnicode_FromFormat("SequenceIndex(metric=%s, size=%zu)",
                                usearch_sequence_metric_name(description.metadata.metric), description.size);
}

#pragma endregion Introspection

#pragma region Ingest

static char const doc_SequenceIndex_reserve[] =                            //
    "``reserve(capacity, /, *, payload_bytes=0) -> None``\n"               //
    "\n"                                                                   //
    "Grows the index to host at least `capacity` entries, a hint, since\n" //
    "insertions grow the index on demand.\n"                               //
    "Calls `usearch_sequence_reserve`.\n"                                  //
    "\n"                                                                   //
    "Args:\n"                                                              //
    "  capacity (int): Entries the index must accept without growing.\n"   //
    "  payload_bytes (int, optional): Expected total bytes across every\n" //
    "    entry; 0 lets the payload storage grow lazily.";

static PyObject* SequenceIndex_reserve(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                       PyObject* args_names_tuple) {
    PyObject *capacity_object, *payload_object;
    usearch_py_argument_t const arguments[] = {{"capacity", &capacity_object}, {"payload_bytes", &payload_object}};
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
    size_t capacity = 0, payload_bytes = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("reserve", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(capacity_object, "capacity", &capacity) != 0 ||
        usearch_py_size_from_object(payload_object, "payload_bytes", &payload_bytes) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    usearch_sequence_reserve_t* const reserve = (usearch_sequence_reserve_t*)call->arguments;
    *reserve = (usearch_sequence_reserve_t) {.index = handle,
                                             .executor = receiver.executor,
                                             .outcome = &call->outcome,
                                             .completion = receiver.completion,
                                             .entries = capacity,
                                             .payload_bytes = payload_bytes};
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_reserve(reserve);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_SequenceIndex_clear[] =                          //
    "clear() -> None\n"                                                //
    "\n"                                                               //
    "Drops every entry, keeping the capacity and the frozen metric.\n" //
    "Calls `usearch_sequence_clear`.";

static PyObject* SequenceIndex_clear(PyObject* self, PyObject* unused) {
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    usearch_sequence_clear_t* const arguments = (usearch_sequence_clear_t*)call->arguments;
    *arguments = (usearch_sequence_clear_t) {.index = handle,
                                             .executor = receiver.executor,
                                             .outcome = &call->outcome,
                                             .completion = receiver.completion};
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_clear(arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_SequenceIndex_insert[] =                                                   //
    "``insert(keys, sequences, /, *, expansion_factor=0, progress=None, progress_steps=0,\n"     //
    "stats=None) -> None``\n"                                                                    //
    "\n"                                                                                         //
    "Inserts a batch of sequences under a batch of keys; one sequence is a batch of one.\n"      //
    "Keys need not be unique, and a batch is all-or-nothing and becomes visible at once.\n"      //
    "Calls `usearch_sequence_insert`.\n"                                                         //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  keys (Sequence[Key | int] | buffer): One key per sequence, or an N by 16 buffer.\n"       //
    "  sequences (Sequence[str | buffer] | str | buffer): The sequences, each a `str` read as\n" //
    "    UTF-8 or a contiguous bytes-like object, gathered once into one tape.\n"                //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"             //
    "  progress (callable, optional): Called with (completed, total) entries; what it raises\n"  //
    "    goes to `sys.unraisablehook`.\n"                                                        //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"      //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.";

static PyObject* SequenceIndex_insert(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    PyObject *keys_object, *sequences_object, *expansion_object, *progress_object, *progress_steps_object,
        *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"keys", &keys_object},
        {"sequences", &sequences_object},
        {"expansion_factor", &expansion_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
    size_t expansion_factor = 0, progress_steps = 0;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("insert", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 ||
        usearch_py_size_from_object(expansion_object, "expansion_factor", &expansion_factor) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", &progress_steps) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    size_t count = 0;
    usearch_tape_view_t sequences;
    if (sequences_from_object(sequences_object, &call->blocks[0], &count, &sequences) != 0 ||
        usearch_py_keys_from_object(keys_object, count, &call->keys[0]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0)
        return usearch_py_call_discard(&receiver, call);

    usearch_sequence_insert_t* const insert = (usearch_sequence_insert_t*)call->arguments;
    *insert = (usearch_sequence_insert_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .count = count,
        .keys = {call->keys[0].keys, sizeof(usearch_key_t)},
        .sequences = sequences,
        .expansion_factor = expansion_factor,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
        .stats = walk_stats(call),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_insert(insert);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_SequenceIndex_refine[] =                                                    //
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
    "Calls `usearch_sequence_refine`.\n"                                                          //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"              //
    "  memory_limit (int, optional): Bytes a loaded NVMe handle holds for the part it\n"          //
    "    rewrites; 0 derives what rounds that part in RAM.\n"                                     //
    "  progress (callable, optional): Called with (completed, total) units of each phase;\n"      //
    "    returning False stops the round, None continues.\n"                                      //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"       //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"          //
    "  plan (SequenceRefinePlan, optional): The round to open or continue; None runs one\n"       //
    "    round to its end.\n"                                                                     //
    "\n"                                                                                          //
    "Returns:\n"                                                                                  //
    "  RefineReport: What the round did to the graph, and its `RefineVerdict`.";

static PyObject* SequenceIndex_refine(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
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
    usearch_sequence_index_t handle = NULL;
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
        plan_parse(call, plan_object, &SequenceRefinePlanType) != 0)
        return usearch_py_call_discard(&receiver, call);

    SequencePlan* const plan = (SequencePlan*)call->holds[callback_slot_k];
    usearch_sequence_refine_t* const refine = (usearch_sequence_refine_t*)call->arguments;
    *refine = (usearch_sequence_refine_t) {
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
    usearch_sequence_refine(refine);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Ingest

#pragma region Search

static char const doc_SequenceIndex_search[] =                                                   //
    "``search(queries, wanted, arena, /, *, expansion_factor=0, traversal=Traversal.WALK,\n"     //
    "predicate=None, progress=None, progress_steps=0, stats=None) -> Matches``\n"                //
    "\n"                                                                                         //
    "Finds the `wanted` nearest entries for every query; one query is a batch of one. Never\n"   //
    "writes to the graph, so any number of searches may overlap with insertions.\n"              //
    "Calls `usearch_sequence_search`.\n"                                                         //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  queries (Sequence[str | buffer] | str | buffer): The query sequences.\n"                  //
    "  wanted (int): Upper bound on matches per query, the 'k' in 'kANN'.\n"                     //
    "  arena (Arena): Where the matches land, rewound by the call.\n"                            //
    "  expansion_factor (int, optional): Breadth of the candidates; 0 derives it.\n"             //
    "  traversal (Traversal, optional): `Traversal.WALK`, the default, walks the graph;\n"       //
    "    `Traversal.SCAN` scores every entry exactly.\n"                                         //
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

static PyObject* SequenceIndex_search(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                      PyObject* args_names_tuple) {
    PyObject *queries_object, *wanted_object, *arena_object, *expansion_object, *traversal_object, *predicate_object,
        *progress_object, *progress_steps_object, *stats_object;
    usearch_py_argument_t const arguments[] = {
        {"queries", &queries_object},     {"wanted", &wanted_object},
        {"arena", &arena_object},         {"expansion_factor", &expansion_object},
        {"traversal", &traversal_object}, {"predicate", &predicate_object},
        {"progress", &progress_object},   {"progress_steps", &progress_steps_object},
        {"stats", &stats_object},
    };
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
    size_t wanted = 0, expansion_factor = 0, progress_steps = 0;
    int traversal = usearch_traversal_walk_k;
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
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    size_t count = 0;
    usearch_tape_view_t queries;
    Matches* matches = NULL;
    if (sequences_from_object(queries_object, &call->blocks[0], &count, &queries) != 0 ||
        usearch_py_predicate_from_object(predicate_object, &call->holds[callback_slot_k]) != 0 ||
        walk_parse(call, progress_object, stats_object) != 0 ||
        !(matches = usearch_py_call_arena(call, arena_object, &MatchesType)))
        return usearch_py_call_discard(&receiver, call);

    matches->queries = count;
    usearch_sequence_search_t* const search = (usearch_sequence_search_t*)call->arguments;
    *search = (usearch_sequence_search_t) {
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
    usearch_sequence_search(search);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Search

#pragma region Retrieval

static char const doc_SequenceIndex_count[] =                          //
    "count(key) -> int\n"                                              //
    "\n"                                                               //
    "How many entries share `key`; zero means the index holds none.\n" //
    "Calls `usearch_sequence_count`.\n"                                //
    "\n"                                                               //
    "Args:\n"                                                          //
    "  key (Key | int | str): The identity to count.";

/** A @c usearch_sequence_count call and the statistics it lands, inside the record's arguments. */
typedef struct count_call_t {
    usearch_sequence_count_t arguments;
    usearch_sequence_key_stats_t key_stats;
} count_call_t;

static PyObject* count_answer(usearch_py_call_t* call) {
    return PyLong_FromSize_t(((count_call_t const*)call->arguments)->key_stats.entries);
}

static PyObject* SequenceIndex_count(PyObject* self, PyObject* key_object) {
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
    usearch_key_t key;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (receiver_from_object(self, &receiver, &handle) != 0 || usearch_py_key_from_object(key_object, &key) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;
    count_call_t* const count = (count_call_t*)call->arguments;
    call->answer = count_answer;
    count->arguments = (usearch_sequence_count_t) {.index = handle,
                                                   .executor = receiver.executor,
                                                   .outcome = &call->outcome,
                                                   .completion = receiver.completion,
                                                   .key = key,
                                                   .key_stats = &count->key_stats};
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_count(&count->arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static int SequenceIndex_contains(PyObject* self, PyObject* key_object) {
    PyObject* const entries = SequenceIndex_count(self, key_object);
    if (!entries) return -1;
    int const present = PyObject_IsTrue(entries);
    Py_DECREF(entries);
    return present;
}

static char const doc_SequenceIndex_get[] =                                                   //
    "``get(key, arena, /) -> tuple[buffer, buffer]``\n"                                       //
    "\n"                                                                                      //
    "Every sequence stored under `key`, in insertion order, as an Apache Arrow tape in the\n" //
    "arena: the (count + 1,) offsets and the bytes of every entry end to end, entry `i`\n"    //
    "spanning `offsets[i]` up to `offsets[i + 1]`. Both are empty when the key is absent.\n"  //
    "Calls `usearch_sequence_get`.\n"                                                         //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  key (Key | int | str): The identity to retrieve.\n"                                    //
    "  arena (Arena): Where the tape lands, rewound by the call.";

/** A @c usearch_sequence_get call and the tape it lands, inside the record's arguments. */
typedef struct get_call_t {
    usearch_sequence_get_t arguments;
    usearch_tape_view_t sequences;
    size_t written;
} get_call_t;

static PyObject* get_answer(usearch_py_call_t* call) {
    get_call_t const* const get = (get_call_t const*)call->arguments;
    usearch_tape_view_t const* const tape = &get->sequences;
    int const narrow = tape->layout == usearch_tape_u32_k;
    void const* const offsets = narrow ? (void const*)tape->offsets.u32 : (void const*)tape->offsets.u64;
    // An absent key leaves the tape's pointers NULL, so there is no offset run to view.
    size_t const first = !offsets ? 0 : narrow ? tape->offsets.u32[0] : tape->offsets.u64[0];
    size_t const last = !offsets ? 0 : narrow ? tape->offsets.u32[get->written] : tape->offsets.u64[get->written];
    Py_ssize_t const offsets_shape[1] = {offsets ? (Py_ssize_t)get->written + 1 : 0};
    Py_ssize_t const bytes_shape[1] = {(Py_ssize_t)(last - first)};
    PyObject* const offsets_object = usearch_py_arena_buffer(&ArenaBufferType, call->arena, offsets, narrow ? "I" : "Q",
                                                             narrow ? 4 : 8, offsets_shape, 1);
    PyObject* const bytes_object = offsets_object ? usearch_py_arena_buffer(&ArenaBufferType, call->arena, tape->data,
                                                                            "B", 1, bytes_shape, 1)
                                                  : NULL;
    PyObject* const pair = bytes_object ? PyTuple_Pack(2, offsets_object, bytes_object) : NULL;
    Py_XDECREF(bytes_object);
    Py_XDECREF(offsets_object);
    return pair;
}

static PyObject* SequenceIndex_get(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                   PyObject* args_names_tuple) {
    PyObject *key_object, *arena_object;
    usearch_py_argument_t const arguments[] = {{"key", &key_object}, {"arena", &arena_object}};
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
    usearch_key_t key;
    usearch_py_call_t stack;
    usearch_py_call_t* call = NULL;
    if (usearch_py_arguments_parse("get", args, positional_args_count, args_names_tuple, 2, 2, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        receiver_from_object(self, &receiver, &handle) != 0 || usearch_py_key_from_object(key_object, &key) != 0 ||
        !(call = usearch_py_call_open(&receiver, &stack)))
        return NULL;

    if (usearch_py_call_hold_arena(call, arena_object) != 0) return usearch_py_call_discard(&receiver, call);
    get_call_t* const get = (get_call_t*)call->arguments;
    call->answer = get_answer;
    get->arguments = (usearch_sequence_get_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .key = key,
        .sequences = &get->sequences,
        .written = &get->written,
        .arena = usearch_py_arena_handle(call->arena),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_get(&get->arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Retrieval

#pragma region Composition

static char const doc_SequenceIndex_reshard[] =                                                    //
    "``reshard(sources, targets, /, *, expansion_factor=0, memory_limit=0, seed=0,\n"              //
    "progress=None, progress_steps=0, stats=None, plan=None) -> RefineReport``\n"                  //
    "\n"                                                                                           //
    "Moves the entries of `sources` into `targets`, keeping their links rather than\n"             //
    "re-inserting their sequences, then runs a round on every target that took entries. Several\n" //
    "sources into one target, itself among them, gather into it; one source into several\n"        //
    "targets cuts it into one graph-local part per target. Every index stays the caller's, and\n"  //
    "a source the targets do not name ends empty and live. The receiver names where the call\n"    //
    "runs, and need not be among them.\n"                                                          //
    "\n"                                                                                           //
    "The handler stops the closing rounds alone, never a move: a stop raises `CancelledError`\n"   //
    "with every entry moved, or, with a `plan`, leaves it live for the next call naming it to\n"   //
    "continue, until `reshard_plan_finish(plan)` closes it. Calls `usearch_sequence_reshard`.\n"   //
    "\n"                                                                                           //
    "A live plan holds no index, so each index it names must outlive its finish.\n"                //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  sources (Sequence[SequenceIndex]): The indexes whose entries move, each named once.\n"      //
    "  targets (Sequence[SequenceIndex]): The indexes they move into, each a source or empty.\n"   //
    "  expansion_factor (int, optional): Breadth of any walk beside the rounds; 0 derives it.\n"   //
    "  memory_limit (int, optional): Bytes a loaded NVMe handle holds for the part it\n"           //
    "    rewrites; 0 derives what rounds that part in RAM.\n"                                      //
    "  seed (int, optional): Picks the part roots of a cut; equal seeds cut equally.\n"            //
    "  progress (callable, optional): Called with (completed, total) units of each phase of\n"     //
    "    the closing rounds; returning False stops them, None continues.\n"                        //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"        //
    "  stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.\n"           //
    "  plan (SequenceReshardPlan, optional): The reshard to open or continue; None runs it to\n"   //
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
    usearch_sequence_index_t* const block = PyMem_Malloc((length ? (size_t)length : 1) *
                                                         sizeof(usearch_sequence_index_t));
    if (!block) return PyErr_NoMemory(), -1;
    *handles = block;
    for (Py_ssize_t position = 0; position != length; ++position) {
        PyObject* const index = PySequence_Fast_GET_ITEM(sequence, position);
        if (!Py_IS_TYPE(index, &SequenceIndexType)) {
            PyErr_Format(PyExc_TypeError, "the '%s' argument must hold SequenceIndex objects, got %S", field,
                         Py_TYPE(index));
            return -1;
        }
        block[position] = ((SequenceIndex*)index)->handle;
    }
    *count = (size_t)length;
    return 0;
}

static PyObject* SequenceIndex_reshard(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
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
    usearch_sequence_index_t handle = NULL;
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
        plan_parse(call, plan_object, &SequenceReshardPlanType) != 0)
        return usearch_py_call_discard(&receiver, call);

    SequencePlan* const plan = (SequencePlan*)call->holds[callback_slot_k];
    usearch_sequence_reshard_t* const reshard = (usearch_sequence_reshard_t*)call->arguments;
    *reshard = (usearch_sequence_reshard_t) {
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
    usearch_sequence_reshard(reshard);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Composition

#pragma region Plans

static char const doc_SequenceRefinePlan[] =                                                       //
    "SequenceRefinePlan()\n"                                                                       //
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

static char const doc_SequenceReshardPlan[] =                                                       //
    "SequenceReshardPlan()\n"                                                                       //
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

static PyObject* SequencePlan_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    if (PyTuple_GET_SIZE(args) != 0 || (keywords && PyDict_Size(keywords) != 0)) {
        PyErr_Format(PyExc_TypeError, "%s() takes no arguments", type->tp_name + sizeof("usearch.") - 1);
        return NULL;
    }
    return type->tp_alloc(type, 0);
}

static PyObject* SequencePlan_get_phase(SequencePlan* self, void* closure) {
    return usearch_py_enum_to_object(&usearch_py_phase, (int)self->progress.phase);
}

static PyGetSetDef SequencePlan_getset[] = {
    {"phase", (getter)SequencePlan_get_phase, NULL, "Where the last call naming the plan stopped, as a `Phase`.", NULL},
    {"ordinal", usearch_py_size_getter, NULL, "That phase's position among the ones the plan runs.",
     (void*)offsetof(SequencePlan, progress.ordinal)},
    {"completed", usearch_py_size_getter, NULL, "Units of that phase done, entries or islands.",
     (void*)offsetof(SequencePlan, progress.completed)},
    {"total", usearch_py_size_getter, NULL, "Units of that phase in all.",
     (void*)offsetof(SequencePlan, progress.total)},
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject SequenceRefinePlanType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.SequenceRefinePlan",
    .tp_doc = doc_SequenceRefinePlan,
    .tp_basicsize = sizeof(SequencePlan),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = SequencePlan_tp_new,
    .tp_getset = SequencePlan_getset,
};

PyTypeObject SequenceReshardPlanType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.SequenceReshardPlan",
    .tp_doc = doc_SequenceReshardPlan,
    .tp_basicsize = sizeof(SequencePlan),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = SequencePlan_tp_new,
    .tp_getset = SequencePlan_getset,
};

static char const doc_SequenceIndex_refine_plan_finish[] =                                       //
    "refine_plan_finish(plan) -> RefineReport\n"                                                 //
    "\n"                                                                                         //
    "Closes a `SequenceRefinePlan` on this index and the receiver's executor, the ones it was\n" //
    "opened with, and frees it whatever the call answers; a round a stop left open closes as\n"  //
    "a stop at that barrier would. Calls `usearch_sequence_refine_plan_finish`.\n"               //
    "\n"                                                                                         //
    "Returns:\n"                                                                                 //
    "  RefineReport: What the whole plan did to the graph.";

static char const doc_SequenceIndex_reshard_plan_finish[] =                                        //
    "reshard_plan_finish(plan) -> RefineReport\n"                                                  //
    "\n"                                                                                           //
    "Closes a `SequenceReshardPlan` on the receiver's executor, the one it was opened with, and\n" //
    "frees it whatever the call answers, reopening every index it names; a plan whose entries\n"   //
    "have not all moved runs no round and reports zeros. Calls\n"                                  //
    "`usearch_sequence_reshard_plan_finish`.\n"                                                    //
    "\n"                                                                                           //
    "Returns:\n"                                                                                   //
    "  RefineReport: What the whole plan did to the graph.";

/** Opens a finish of the plan @p plan_object of @p type through @p self: the report it writes into,
 *  the plan held, and its handle taken, since the library frees it whatever the call answers. */
static usearch_py_call_t* plan_finish_open(PyObject* self, PyObject* plan_object, PyTypeObject* type,
                                           usearch_py_receiver_t* receiver, usearch_sequence_index_t* handle,
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

static PyObject* SequenceIndex_refine_plan_finish(PyObject* self, PyObject* plan_object) {
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
    usearch_py_call_t stack;
    RefineReport* report_call = NULL;
    usearch_py_call_t* const call = plan_finish_open(self, plan_object, &SequenceRefinePlanType, &receiver, &handle,
                                                     &stack, &report_call);
    if (!call) return NULL;
    SequencePlan* const plan = (SequencePlan*)plan_object;
    usearch_sequence_refine_plan_finish_t* const finish = (usearch_sequence_refine_plan_finish_t*)call->arguments;
    *finish = (usearch_sequence_refine_plan_finish_t) {.index = handle,
                                                       .executor = receiver.executor,
                                                       .outcome = &call->outcome,
                                                       .completion = receiver.completion,
                                                       .plan = plan->handle.refine,
                                                       .report = &report_call->report};
    plan->handle.refine = NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_refine_plan_finish(finish);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static PyObject* SequenceIndex_reshard_plan_finish(PyObject* self, PyObject* plan_object) {
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
    usearch_py_call_t stack;
    RefineReport* report_call = NULL;
    usearch_py_call_t* const call = plan_finish_open(self, plan_object, &SequenceReshardPlanType, &receiver, &handle,
                                                     &stack, &report_call);
    if (!call) return NULL;
    SequencePlan* const plan = (SequencePlan*)plan_object;
    usearch_sequence_reshard_plan_finish_t* const finish = (usearch_sequence_reshard_plan_finish_t*)call->arguments;
    *finish = (usearch_sequence_reshard_plan_finish_t) {.executor = receiver.executor,
                                                        .outcome = &call->outcome,
                                                        .completion = receiver.completion,
                                                        .plan = plan->handle.reshard,
                                                        .report = &report_call->report};
    plan->handle.reshard = NULL;
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_reshard_plan_finish(finish);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

#pragma endregion Plans

#pragma region Serialization

static char const doc_SequenceIndex_save[] =                                                //
    "``save(path, /, *, progress=None, progress_steps=0) -> None``\n"                       //
    "\n"                                                                                    //
    "Writes the index under `path`, in the layout its extension picks: one `.safegraphs`\n" //
    "monolith, or a `.safegraphs.json` manifest beside one part per shard. Demands a\n"     //
    "mutation-free window; concurrent readers keep searching throughout.\n"                 //
    "Calls `usearch_sequence_save`.\n"                                                      //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  path (str | bytes | os.PathLike): Where the checkpoint goes.\n"                      //
    "  progress (callable, optional): Called with (completed, total) parts; returning\n"    //
    "    False stops the write and raises `CancelledError`.\n"                              //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.";

static PyObject* SequenceIndex_save(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                    PyObject* args_names_tuple) {
    PyObject *path_object, *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {{"path", &path_object},
                                               {"progress", &progress_object},
                                               {"progress_steps", &progress_steps_object}};
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
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
    usearch_sequence_save_t* const save = (usearch_sequence_save_t*)call->arguments;
    *save = (usearch_sequence_save_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .path = path,
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_save(save);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

static char const doc_SequenceIndex_save_buffer[] =                                           //
    "``save_buffer(arena, /, *, progress=None, progress_steps=0) -> Monolith``\n"             //
    "\n"                                                                                      //
    "Serializes the index into the arena: one monolith for a handle of one shard, the very\n" //
    "bytes `save` writes, or a manifest leading one part per shard.\n"                        //
    "Calls `usearch_sequence_save_buffer`.\n"                                                 //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  arena (Arena): Where the checkpoint lands, rewound by the call.\n"                     //
    "  progress (callable, optional): Called with (completed, total) parts; returning\n"      //
    "    False stops the write and raises `CancelledError`, None continues.\n"                //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"   //
    "\n"                                                                                      //
    "Returns:\n"                                                                              //
    "  Monolith: The checkpoint's bytes, in place in the arena.";

/** A @c usearch_sequence_save_buffer call and the monolith it lands, inside the arguments. */
typedef struct save_buffer_call_t {
    usearch_sequence_save_buffer_t arguments;
    usearch_bytes_view_t monolith;
} save_buffer_call_t;

static PyObject* save_buffer_answer(usearch_py_call_t* call) {
    usearch_bytes_view_t const monolith = ((save_buffer_call_t const*)call->arguments)->monolith;
    Py_ssize_t const shape[1] = {(Py_ssize_t)monolith.length};
    return usearch_py_arena_buffer(&MonolithType, call->arena, monolith.data, "B", 1, shape, 1);
}

static PyObject* SequenceIndex_save_buffer(PyObject* self, PyObject* const* args, Py_ssize_t positional_args_count,
                                           PyObject* args_names_tuple) {
    PyObject *arena_object, *progress_object, *progress_steps_object;
    usearch_py_argument_t const arguments[] = {{"arena", &arena_object},
                                               {"progress", &progress_object},
                                               {"progress_steps", &progress_steps_object}};
    usearch_py_receiver_t receiver;
    usearch_sequence_index_t handle = NULL;
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
    save->arguments = (usearch_sequence_save_buffer_t) {
        .index = handle,
        .executor = receiver.executor,
        .outcome = &call->outcome,
        .completion = receiver.completion,
        .monolith = &save->monolith,
        .arena = usearch_py_arena_handle(call->arena),
        .progress = usearch_py_progress_registered(call->holds[progress_slot_k], progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_save_buffer(&save->arguments);
    Py_END_ALLOW_THREADS;
    return usearch_py_call_finish(&receiver, call);
}

/**
 *  @brief Reads what every load takes beside its source: the residency arm @p arm_name, the
 *      @p executor, @p values_per_key, the @p weights, @p progress and @p progress_steps, the
 *      source itself written to @p source.
 *  @return 0 on success, -1 with an exception set otherwise; the weights' matrix is the caller's to
 *      release either way.
 */
static int load_parse(char const* verb, PyObject* const* args, Py_ssize_t positional_args_count,
                      PyObject* args_names_tuple, char const* source_name, PyObject** source, char const* arm_name,
                      PyObject** arm, usearch_py_enum_t const* ceiling_classes, int* ceiling,
                      usearch_executor_t* executor, size_t* values_per_key, weights_t* weights,
                      usearch_sequence_weights_t const** passed_weights, PyObject** progress, size_t* progress_steps) {
    PyObject *executor_object, *values_per_key_object, *matrix_object, *gap_object, *progress_object,
        *progress_steps_object, *ceiling_object;
    usearch_py_argument_t const arguments[] = {
        {source_name, source},
        {arm_name, arm},
        {"executor", &executor_object},
        {"values_per_key", &values_per_key_object},
        {"substitution_matrix", &matrix_object},
        {"gap", &gap_object},
        {"progress", &progress_object},
        {"progress_steps", &progress_steps_object},
        {ceiling_name(ceiling_classes), &ceiling_object},
    };
    memset(weights, 0, sizeof(*weights));
    if (usearch_py_arguments_parse(verb, args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_class_from_object(ceiling_object, ceiling_name(ceiling_classes), ceiling_classes, ceiling) != 0 ||
        usearch_py_optional_executor_from_object(executor_object, executor) != 0 ||
        usearch_py_size_from_object(values_per_key_object, "values_per_key", values_per_key) != 0 ||
        usearch_py_size_from_object(progress_steps_object, "progress_steps", progress_steps) != 0 ||
        weights_parse(matrix_object, gap_object, weights, passed_weights) != 0)
        return -1;
    return usearch_py_callable_from_object(progress_object, "progress", progress);
}

static char const doc_SequenceIndex_load_ram[] =                                                     //
    "``load_ram(path, /, *, memory_domains=None, executor=None, values_per_key=0,\n"                 //
    "substitution_matrix=None, gap=None, progress=None, progress_steps=0, cpu_class_ceiling=None)\n" //
    "-> SequenceIndex``\n"                                                                           //
    "\n"                                                                                             //
    "Copies a checkpoint into a new, mutable index in host memory, its metric read from the\n"       //
    "checkpoint's own headers. Name the `.safegraphs.json` manifest, never one of its parts.\n"      //
    "Calls `usearch_sequence_load_ram`.\n"                                                           //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                      //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place the parts on.\n"             //
    "  executor (Executor, optional): The workers the read fans out over; None reads on the\n"       //
    "    calling thread.\n"                                                                          //
    "  values_per_key (int, optional): Entries sharing one key, which no checkpoint records.\n"      //
    "  substitution_matrix (buffer, optional): The scores an alignment checkpoint was built\n"       //
    "    with; the Levenshtein kinds take none.\n"                                                   //
    "  gap (int, optional): The gap cost beside the matrix.\n"                                       //
    "  progress (callable, optional): Called with (completed, total) parts; returning False\n"       //
    "    stops the read and raises `CancelledError`, creating no index.\n"                           //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"          //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n"    //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.";

static PyObject* SequenceIndex_load_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                        PyObject* args_names_tuple) {
    PyObject *path_object, *memory_domains_object, *decoded = NULL;
    usearch_outcome_t outcome = {0};
    PyObject* progress = NULL;
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    weights_t weights = {0};
    usearch_sequence_weights_t const* passed_weights = NULL;
    usearch_bytes_view_t path;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    PyObject* result = NULL;
    int cpu_class_ceiling = 0;
    if (load_parse("load_ram", args, positional_args_count, args_names_tuple, "path", &path_object, "memory_domains",
                   &memory_domains_object, &usearch_py_cpu_class, &cpu_class_ceiling, &executor, &values_per_key,
                   &weights, &passed_weights, &progress, &progress_steps) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        goto release;

    usearch_sequence_index_t handle = NULL;
    usearch_sequence_load_ram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
        .values_per_key = values_per_key,
        .weights = passed_weights,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_load_ram(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(memory_domains);
    Py_XDECREF(decoded);
    usearch_py_buffer_release(&weights.matrix);
    Py_XDECREF(progress);
    return result;
}

static char const doc_SequenceIndex_load_ram_buffer[] =                                              //
    "``load_ram_buffer(buffer, /, *, memory_domains=None, executor=None, values_per_key=0,\n"        //
    "substitution_matrix=None, gap=None, progress=None, progress_steps=0, cpu_class_ceiling=None)\n" //
    "-> SequenceIndex``\n"                                                                           //
    "\n"                                                                                             //
    "Copies a checkpoint out of a buffer into a new, mutable index in host memory, as\n"             //
    "`load_ram` does from a path.\n"                                                                 //
    "Calls `usearch_sequence_load_ram_buffer`.\n"                                                    //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  buffer (buffer): One serialized checkpoint, such as `save_buffer` returns.\n"                 //
    "  memory_domains (Sequence[int], optional): NUMA node IDs to place shards on.\n"                //
    "  executor (Executor, optional): The workers the copy fans out over.\n"                         //
    "  values_per_key (int, optional): Entries sharing one key; 0 is unbounded.\n"                   //
    "  substitution_matrix (buffer, optional): The scores an alignment checkpoint was built\n"       //
    "    with.\n"                                                                                    //
    "  gap (int, optional): The gap cost beside the matrix.\n"                                       //
    "  progress (callable, optional): Called with (completed, total) parts.\n"                       //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"          //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n"    //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.";

static PyObject* SequenceIndex_load_ram_buffer(PyTypeObject* type, PyObject* const* args,
                                               Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    PyObject *buffer_object, *memory_domains_object;
    usearch_outcome_t outcome = {0};
    PyObject* progress = NULL;
    usearch_py_buffer_backing_t backing = {0};
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    weights_t weights = {0};
    usearch_sequence_weights_t const* passed_weights = NULL;
    int32_t* memory_domains = NULL;
    size_t memory_domains_count = 0;
    PyObject* result = NULL;
    int cpu_class_ceiling = 0;
    if (load_parse("load_ram_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                   "memory_domains", &memory_domains_object, &usearch_py_cpu_class, &cpu_class_ceiling, &executor,
                   &values_per_key, &weights, &passed_weights, &progress, &progress_steps) != 0 ||
        usearch_py_checkpoint_from_object(buffer_object, &backing) != 0 ||
        usearch_py_ordinals_from_object(memory_domains_object, "memory_domains", &memory_domains,
                                        &memory_domains_count) != 0)
        goto release;

    usearch_sequence_index_t handle = NULL;
    usearch_sequence_load_ram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .ram = {memory_domains, memory_domains_count, 0, (usearch_cpu_class_t)cpu_class_ceiling},
        .values_per_key = values_per_key,
        .weights = passed_weights,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_load_ram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(memory_domains);
    usearch_py_buffer_release(&backing);
    usearch_py_buffer_release(&weights.matrix);
    Py_XDECREF(progress);
    return result;
}

static char const doc_SequenceIndex_load_vram[] =                                                    //
    "``load_vram(path, /, *, devices=None, executor=None, values_per_key=0,\n"                       //
    "substitution_matrix=None, gap=None, progress=None, progress_steps=0, gpu_class_ceiling=None)\n" //
    "-> SequenceIndex``\n"                                                                           //
    "\n"                                                                                             //
    "Copies a checkpoint into a new, mutable index on GPU devices, with the placement rules\n"       //
    "of `load_ram`.\n"                                                                               //
    "Calls `usearch_sequence_load_vram`.\n"                                                          //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                      //
    "  devices (Sequence[int], optional): GPU ordinals to place the parts on.\n"                     //
    "  executor (Executor, optional): The streams the upload runs on, one per device.\n"             //
    "  values_per_key (int, optional): Zero alone, as a GPU index keeps no key lookup.\n"            //
    "  substitution_matrix (buffer, optional): The scores an alignment checkpoint was built\n"       //
    "    with.\n"                                                                                    //
    "  gap (int, optional): The gap cost beside the matrix.\n"                                       //
    "  progress (callable, optional): Called with (completed, total) parts.\n"                       //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"          //
    "  gpu_class_ceiling (GPUClass, optional): The newest class dispatch may pick, capped below\n"   //
    "    the best class every device loads, else unsupported; None means no ceiling.";

static PyObject* SequenceIndex_load_vram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                         PyObject* args_names_tuple) {
    PyObject *path_object, *devices_object, *decoded = NULL;
    usearch_outcome_t outcome = {0};
    PyObject* progress = NULL;
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    weights_t weights = {0};
    usearch_sequence_weights_t const* passed_weights = NULL;
    usearch_bytes_view_t path;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    PyObject* result = NULL;
    int gpu_class_ceiling = 0;
    if (load_parse("load_vram", args, positional_args_count, args_names_tuple, "path", &path_object, "devices",
                   &devices_object, &usearch_py_gpu_class, &gpu_class_ceiling, &executor, &values_per_key, &weights,
                   &passed_weights, &progress, &progress_steps) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        goto release;

    usearch_sequence_index_t handle = NULL;
    usearch_sequence_load_vram_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
        .values_per_key = values_per_key,
        .weights = passed_weights,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_load_vram(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(devices);
    Py_XDECREF(decoded);
    usearch_py_buffer_release(&weights.matrix);
    Py_XDECREF(progress);
    return result;
}

static char const doc_SequenceIndex_load_vram_buffer[] =                                             //
    "``load_vram_buffer(buffer, /, *, devices=None, executor=None, values_per_key=0,\n"              //
    "substitution_matrix=None, gap=None, progress=None, progress_steps=0, gpu_class_ceiling=None)\n" //
    "-> SequenceIndex``\n"                                                                           //
    "\n"                                                                                             //
    "Copies a checkpoint out of a host buffer into a new, mutable index on GPU devices.\n"           //
    "Calls `usearch_sequence_load_vram_buffer`.\n"                                                   //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  buffer (buffer): One serialized checkpoint, such as `save_buffer` returns.\n"                 //
    "  devices (Sequence[int], optional): GPU ordinals to place shards on.\n"                        //
    "  executor (Executor, optional): The streams the upload runs on, one per device.\n"             //
    "  values_per_key (int, optional): Zero alone, as a GPU index keeps no key lookup.\n"            //
    "  substitution_matrix (buffer, optional): The scores an alignment checkpoint was built\n"       //
    "    with.\n"                                                                                    //
    "  gap (int, optional): The gap cost beside the matrix.\n"                                       //
    "  progress (callable, optional): Called with (completed, total) parts.\n"                       //
    "  progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.\n"          //
    "  gpu_class_ceiling (GPUClass, optional): The newest class dispatch may pick, capped below\n"   //
    "    the best class every device loads, else unsupported; None means no ceiling.";

static PyObject* SequenceIndex_load_vram_buffer(PyTypeObject* type, PyObject* const* args,
                                                Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    PyObject *buffer_object, *devices_object;
    usearch_outcome_t outcome = {0};
    PyObject* progress = NULL;
    usearch_py_buffer_backing_t backing = {0};
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0, progress_steps = 0;
    weights_t weights = {0};
    usearch_sequence_weights_t const* passed_weights = NULL;
    int32_t* devices = NULL;
    size_t devices_count = 0;
    PyObject* result = NULL;
    int gpu_class_ceiling = 0;
    if (load_parse("load_vram_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                   "devices", &devices_object, &usearch_py_gpu_class, &gpu_class_ceiling, &executor, &values_per_key,
                   &weights, &passed_weights, &progress, &progress_steps) != 0 ||
        usearch_py_checkpoint_from_object(buffer_object, &backing) != 0 ||
        usearch_py_ordinals_from_object(devices_object, "devices", &devices, &devices_count) != 0)
        goto release;

    usearch_sequence_index_t handle = NULL;
    usearch_sequence_load_vram_buffer_t arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .vram = {devices, devices_count, 0, (usearch_gpu_class_t)gpu_class_ceiling},
        .values_per_key = values_per_key,
        .weights = passed_weights,
        .progress = usearch_py_progress_registered(progress, progress_steps),
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_load_vram_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    PyMem_Free(devices);
    usearch_py_buffer_release(&backing);
    usearch_py_buffer_release(&weights.matrix);
    Py_XDECREF(progress);
    return result;
}

static char const doc_SequenceIndex_view_ram[] =                                                  //
    "``view_ram(path, /, *, substitution_matrix=None, gap=None, cpu_class_ceiling=None)\n"        //
    "-> SequenceIndex``\n"                                                                        //
    "\n"                                                                                          //
    "Serves a checkpoint in place through a memory mapping, as a read-only index whose every\n"   //
    "mutating verb raises `ReadOnlyError`.\n"                                                     //
    "Calls `usearch_sequence_view_ram`.\n"                                                        //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                   //
    "  substitution_matrix (buffer, optional): The scores an alignment checkpoint was built\n"    //
    "    with.\n"                                                                                 //
    "  gap (int, optional): The gap cost beside the matrix.\n"                                    //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n" //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.";

static PyObject* SequenceIndex_view_ram(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                        PyObject* args_names_tuple) {
    PyObject *path_object, *matrix_object, *gap_object, *cpu_class_ceiling_object, *decoded = NULL, *result = NULL;
    usearch_py_argument_t const arguments[] = {
        {"path", &path_object},
        {"substitution_matrix", &matrix_object},
        {"gap", &gap_object},
        {"cpu_class_ceiling", &cpu_class_ceiling_object},
    };
    weights_t weights = {0};
    usearch_sequence_weights_t const* passed_weights = NULL;
    usearch_bytes_view_t path;
    int cpu_class_ceiling = 0;
    if (usearch_py_arguments_parse("view_ram", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_class_from_object(cpu_class_ceiling_object, "cpu_class_ceiling", &usearch_py_cpu_class,
                                     &cpu_class_ceiling) != 0 ||
        weights_parse(matrix_object, gap_object, &weights, &passed_weights) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0)
        goto release;

    usearch_sequence_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_sequence_view_ram_t view_arguments = {
        .index = &handle,
        .outcome = &outcome,
        .path = path,
        .weights = passed_weights,
        .ram = {NULL, 0, 0, (usearch_cpu_class_t)cpu_class_ceiling},
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_view_ram(&view_arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    Py_XDECREF(decoded);
    usearch_py_buffer_release(&weights.matrix);
    return result;
}

static char const doc_SequenceIndex_view_nvme[] =                                                 //
    "``view_nvme(path, /, *, paths=None, executor=None, substitution_matrix=None, gap=None,\n"    //
    "cpu_class_ceiling=None) -> SequenceIndex``\n"                                                //
    "\n"                                                                                          //
    "Serves a checkpoint in place from NVMe drives as a read-only index, every later call\n"      //
    "reading it through its executor's engines.\n"                                                //
    "Calls `usearch_sequence_view_nvme`.\n"                                                       //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  path (str | bytes | os.PathLike): The manifest, or a single monolith.\n"                   //
    "  paths (Sequence[str | bytes | PathLike], optional): Devices or directories to serve\n"     //
    "    the parts from; None serves each where its manifest points.\n"                           //
    "  executor (Executor, optional): The NVMe executor whose rings read the headers.\n"          //
    "  substitution_matrix (buffer, optional): The scores an alignment checkpoint was built\n"    //
    "    with.\n"                                                                                 //
    "  gap (int, optional): The gap cost beside the matrix.\n"                                    //
    "  cpu_class_ceiling (CPUClass, optional): The newest class dispatch may pick, intersected\n" //
    "    with what the machine runs, so a class it lacks narrows; None means no ceiling.";

static PyObject* SequenceIndex_view_nvme(PyTypeObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                         PyObject* args_names_tuple) {
    PyObject *path_object, *paths_object, *executor_object, *matrix_object, *gap_object, *cpu_class_ceiling_object;
    PyObject *decoded = NULL, *result = NULL;
    usearch_py_argument_t const arguments[] = {
        {"path", &path_object},         {"paths", &paths_object},
        {"executor", &executor_object}, {"substitution_matrix", &matrix_object},
        {"gap", &gap_object},           {"cpu_class_ceiling", &cpu_class_ceiling_object},
    };
    weights_t weights = {0};
    usearch_sequence_weights_t const* passed_weights = NULL;
    usearch_bytes_view_t path;
    usearch_py_paths_t paths = {0};
    usearch_executor_t executor = NULL;
    int cpu_class_ceiling = 0;
    if (usearch_py_arguments_parse("view_nvme", args, positional_args_count, args_names_tuple, 1, 1, arguments,
                                   usearch_py_table_size_(arguments)) != 0 ||
        usearch_py_optional_executor_from_object(executor_object, &executor) != 0 ||
        usearch_py_class_from_object(cpu_class_ceiling_object, "cpu_class_ceiling", &usearch_py_cpu_class,
                                     &cpu_class_ceiling) != 0 ||
        weights_parse(matrix_object, gap_object, &weights, &passed_weights) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0 ||
        usearch_py_paths_from_object(paths_object, &paths) != 0)
        goto release;

    usearch_sequence_index_t handle = NULL;
    usearch_outcome_t outcome = {0};
    usearch_sequence_view_nvme_t view_arguments = {
        .index = &handle,
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .nvme = {paths.views, paths.count, (usearch_cpu_class_t)cpu_class_ceiling},
        .weights = passed_weights,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_view_nvme(&view_arguments);
    Py_END_ALLOW_THREADS;
    result = index_adopt(type, handle, &outcome);

release:
    usearch_py_paths_release(&paths);
    Py_XDECREF(decoded);
    usearch_py_buffer_release(&weights.matrix);
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

static char const doc_SequenceIndex_metadata_read[] =                                    //
    "``metadata_read(path, /, *, executor=None) -> SequenceMetadata``\n"                 //
    "\n"                                                                                 //
    "Reads the metric and geometry a checkpoint froze, without loading it.\n"            //
    "Calls `usearch_sequence_metadata_read`.\n"                                          //
    "\n"                                                                                 //
    "Args:\n"                                                                            //
    "  path (str | bytes | os.PathLike): The checkpoint to inspect.\n"                   //
    "  executor (Executor, optional): The workers the read runs on; None reads on the\n" //
    "    calling thread.";

static PyObject* SequenceIndex_metadata_read(PyObject* type, PyObject* const* args, Py_ssize_t positional_args_count,
                                             PyObject* args_names_tuple) {
    PyObject *path_object, *decoded = NULL;
    usearch_executor_t executor = NULL;
    usearch_bytes_view_t path;
    if (metadata_parse("metadata_read", args, positional_args_count, args_names_tuple, "path", &path_object,
                       &executor) != 0 ||
        usearch_py_path_from_object(path_object, &path, &decoded) != 0)
        return NULL;
    usearch_sequence_metadata_t metadata = {0};
    usearch_outcome_t outcome = {0};
    usearch_sequence_metadata_read_t arguments = {
        .executor = executor,
        .outcome = &outcome,
        .path = path,
        .metadata = &metadata,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_metadata_read(&arguments);
    Py_END_ALLOW_THREADS;
    Py_DECREF(decoded);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return metadata_to_object(&metadata);
}

static char const doc_SequenceIndex_metadata_read_buffer[] =                             //
    "``metadata_read_buffer(buffer, /, *, executor=None) -> SequenceMetadata``\n"        //
    "\n"                                                                                 //
    "Reads a serialized checkpoint's metric and geometry, as `metadata_read`\n"          //
    "does from a path.\n"                                                                //
    "Calls `usearch_sequence_metadata_read_buffer`.\n"                                   //
    "\n"                                                                                 //
    "Args:\n"                                                                            //
    "  buffer (buffer): One serialized checkpoint, such as `save_buffer` returns.\n"     //
    "  executor (Executor, optional): The workers the read runs on; None reads on the\n" //
    "    calling thread.";

static PyObject* SequenceIndex_metadata_read_buffer(PyObject* type, PyObject* const* args,
                                                    Py_ssize_t positional_args_count, PyObject* args_names_tuple) {
    PyObject* buffer_object;
    usearch_executor_t executor = NULL;
    usearch_py_buffer_backing_t backing;
    if (metadata_parse("metadata_read_buffer", args, positional_args_count, args_names_tuple, "buffer", &buffer_object,
                       &executor) != 0 ||
        usearch_py_checkpoint_from_object(buffer_object, &backing) != 0)
        return NULL;
    usearch_sequence_metadata_t metadata = {0};
    usearch_outcome_t outcome = {0};
    usearch_sequence_metadata_read_buffer_t arguments = {
        .executor = executor,
        .outcome = &outcome,
        .buffer = {(uint8_t const*)backing.buffer.buf, (size_t)backing.buffer.len},
        .metadata = &metadata,
    };
    Py_BEGIN_ALLOW_THREADS;
    usearch_sequence_metadata_read_buffer(&arguments);
    Py_END_ALLOW_THREADS;
    usearch_py_buffer_release(&backing);
    if (usearch_py_raise_on_outcome(&outcome) != 0) return NULL;
    return metadata_to_object(&metadata);
}

#pragma endregion Serialization

#pragma region Registration

static PyMethodDef SequenceIndex_methods[] = {
    {"init_ram", (PyCFunction)SequenceIndex_init_ram, usearch_py_method_flags_k | METH_CLASS,
     doc_SequenceIndex_init_ram},
    {"init_vram", (PyCFunction)SequenceIndex_init_vram, usearch_py_method_flags_k | METH_CLASS,
     doc_SequenceIndex_init_vram},
    {"load_ram", (PyCFunction)SequenceIndex_load_ram, usearch_py_method_flags_k | METH_CLASS,
     doc_SequenceIndex_load_ram},
    {"load_ram_buffer", (PyCFunction)SequenceIndex_load_ram_buffer, usearch_py_method_flags_k | METH_CLASS,
     doc_SequenceIndex_load_ram_buffer},
    {"load_vram", (PyCFunction)SequenceIndex_load_vram, usearch_py_method_flags_k | METH_CLASS,
     doc_SequenceIndex_load_vram},
    {"load_vram_buffer", (PyCFunction)SequenceIndex_load_vram_buffer, usearch_py_method_flags_k | METH_CLASS,
     doc_SequenceIndex_load_vram_buffer},
    {"view_ram", (PyCFunction)SequenceIndex_view_ram, usearch_py_method_flags_k | METH_CLASS,
     doc_SequenceIndex_view_ram},
    {"view_nvme", (PyCFunction)SequenceIndex_view_nvme, usearch_py_method_flags_k | METH_CLASS,
     doc_SequenceIndex_view_nvme},
    {"metadata_read", (PyCFunction)SequenceIndex_metadata_read, usearch_py_method_flags_k | METH_STATIC,
     doc_SequenceIndex_metadata_read},
    {"metadata_read_buffer", (PyCFunction)SequenceIndex_metadata_read_buffer, usearch_py_method_flags_k | METH_STATIC,
     doc_SequenceIndex_metadata_read_buffer},
    {"on", (PyCFunction)SequenceIndex_on, METH_O, doc_SequenceIndex_on},
    {"deferred", (PyCFunction)SequenceIndex_deferred, METH_O, doc_SequenceIndex_deferred},
    {"describe", (PyCFunction)SequenceIndex_describe, METH_NOARGS, doc_SequenceIndex_describe},
    {"reserve", (PyCFunction)SequenceIndex_reserve, usearch_py_method_flags_k, doc_SequenceIndex_reserve},
    {"clear", (PyCFunction)SequenceIndex_clear, METH_NOARGS, doc_SequenceIndex_clear},
    {"insert", (PyCFunction)SequenceIndex_insert, usearch_py_method_flags_k, doc_SequenceIndex_insert},
    {"refine", (PyCFunction)SequenceIndex_refine, usearch_py_method_flags_k, doc_SequenceIndex_refine},
    {"health", (PyCFunction)SequenceIndex_health, METH_NOARGS, doc_SequenceIndex_health},
    {"search", (PyCFunction)SequenceIndex_search, usearch_py_method_flags_k, doc_SequenceIndex_search},
    {"count", (PyCFunction)SequenceIndex_count, METH_O, doc_SequenceIndex_count},
    {"get", (PyCFunction)SequenceIndex_get, usearch_py_method_flags_k, doc_SequenceIndex_get},
    {"reshard", (PyCFunction)SequenceIndex_reshard, usearch_py_method_flags_k, doc_SequenceIndex_reshard},
    {"refine_plan_finish", (PyCFunction)SequenceIndex_refine_plan_finish, METH_O, doc_SequenceIndex_refine_plan_finish},
    {"reshard_plan_finish", (PyCFunction)SequenceIndex_reshard_plan_finish, METH_O,
     doc_SequenceIndex_reshard_plan_finish},
    {"save", (PyCFunction)SequenceIndex_save, usearch_py_method_flags_k, doc_SequenceIndex_save},
    {"save_buffer", (PyCFunction)SequenceIndex_save_buffer, usearch_py_method_flags_k, doc_SequenceIndex_save_buffer},
    {NULL, NULL, 0, NULL},
};

/** Every verb a receiver runs on its executor, the same for an @c On and a @c DeferredOn, since no
 *  sequence verb answers an iterator. */
static PyMethodDef SequenceIndex_receiver_methods[] = {
    {"describe", (PyCFunction)SequenceIndex_describe, METH_NOARGS, doc_SequenceIndex_describe},
    {"reserve", (PyCFunction)SequenceIndex_reserve, usearch_py_method_flags_k, doc_SequenceIndex_reserve},
    {"clear", (PyCFunction)SequenceIndex_clear, METH_NOARGS, doc_SequenceIndex_clear},
    {"insert", (PyCFunction)SequenceIndex_insert, usearch_py_method_flags_k, doc_SequenceIndex_insert},
    {"refine", (PyCFunction)SequenceIndex_refine, usearch_py_method_flags_k, doc_SequenceIndex_refine},
    {"health", (PyCFunction)SequenceIndex_health, METH_NOARGS, doc_SequenceIndex_health},
    {"search", (PyCFunction)SequenceIndex_search, usearch_py_method_flags_k, doc_SequenceIndex_search},
    {"count", (PyCFunction)SequenceIndex_count, METH_O, doc_SequenceIndex_count},
    {"get", (PyCFunction)SequenceIndex_get, usearch_py_method_flags_k, doc_SequenceIndex_get},
    {"reshard", (PyCFunction)SequenceIndex_reshard, usearch_py_method_flags_k, doc_SequenceIndex_reshard},
    {"refine_plan_finish", (PyCFunction)SequenceIndex_refine_plan_finish, METH_O, doc_SequenceIndex_refine_plan_finish},
    {"reshard_plan_finish", (PyCFunction)SequenceIndex_reshard_plan_finish, METH_O,
     doc_SequenceIndex_reshard_plan_finish},
    {"save", (PyCFunction)SequenceIndex_save, usearch_py_method_flags_k, doc_SequenceIndex_save},
    {"save_buffer", (PyCFunction)SequenceIndex_save_buffer, usearch_py_method_flags_k, doc_SequenceIndex_save_buffer},
    {NULL, NULL, 0, NULL},
};

usearch_py_receiver_verbs_t const usearch_py_sequence_receiver_verbs = {
    &SequenceIndexType, SequenceIndex_receiver_methods, SequenceIndex_receiver_methods};

static PyGetSetDef SequenceIndex_getset[] = {
    {"capacity", (getter)SequenceIndex_get_counter, NULL,
     "Entries the index hosts without growing again. Calls `usearch_sequence_describe`.",
     (void*)offsetof(usearch_sequence_index_description_t, capacity)},
    {"connectivity", (getter)SequenceIndex_get_counter, NULL,
     "Connections per node in the graph. Calls `usearch_sequence_describe`.",
     (void*)offsetof(usearch_sequence_index_description_t, metadata.connectivity)},
    {"memory_usage", (getter)SequenceIndex_get_counter, NULL,
     "Bytes the handle holds, payload storage included. Calls `usearch_sequence_describe`.",
     (void*)offsetof(usearch_sequence_index_description_t, memory_usage)},
    {"cpu_class", (getter)SequenceIndex_get_cpu_class, NULL,
     "The `CPUClass` dispatch picked on this machine, or None. Calls `usearch_sequence_describe`.", NULL},
    {"gpu_class", (getter)SequenceIndex_get_gpu_class, NULL,
     "The `GPUClass` dispatch picked on the index's devices, or None. Calls `usearch_sequence_describe`.", NULL},
    {"shards_count", (getter)SequenceIndex_get_counter, NULL, "Live shards. Calls `usearch_sequence_describe`.",
     (void*)offsetof(usearch_sequence_index_description_t, shards_count)},
    {NULL, NULL, NULL, NULL, NULL},
};

static PySequenceMethods SequenceIndex_as_sequence = {
    .sq_length = (lenfunc)SequenceIndex_length,
    .sq_contains = (objobjproc)SequenceIndex_contains,
};

PyTypeObject SequenceIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.SequenceIndex",
    .tp_doc = doc_SequenceIndex,
    .tp_basicsize = sizeof(SequenceIndex),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = SequenceIndex_tp_new,
    .tp_dealloc = (destructor)SequenceIndex_dealloc,
    .tp_repr = (reprfunc)SequenceIndex_repr,
    .tp_methods = SequenceIndex_methods,
    .tp_getset = SequenceIndex_getset,
    .tp_as_sequence = &SequenceIndex_as_sequence,
};

#pragma endregion Registration
