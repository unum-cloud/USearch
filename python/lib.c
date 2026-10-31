/**
 *  @file python/lib.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief The module-init glue - `PyModuleDef`, type registration, and `PyInit_lib`.
 *
 *  Every modality contributes rows to one table here and its verbs to one file beside this
 *  one, so a modality joining the library touches two lines of glue and nothing else.
 */
#include "lib.h"

static char const doc_module[] =                                                                              //
    "Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices.\n" //
    "\n"                                                                                                      //
    "One index type per modality, each sitting directly on the stable C ABI. Every verb\n"                    //
    "is a batch over buffer-protocol inputs, so a memory-mapped dataset reaches the index\n"                  //
    "with no reshape and no copy, and NumPy is never required.";

/** One type the module readies, the name it exports it under, and the layout of a struct sequence. */
static struct {

    /** NULL for a type exported only through the views a verb returns. */
    char const* name;
    PyTypeObject* type;

    /** NULL for every type but a struct sequence. */
    PyStructSequence_Desc* layout;
} const module_types[] = {
    {"Key", &KeyType, NULL},
    {"Match", &MatchType, &usearch_py_match_desc},
    {"Matches", &MatchesType, NULL},
    {"Executor", &ExecutorType, NULL},
    {"Arena", &ArenaType, NULL},
    {NULL, &ArenaBufferType, NULL},
    {"DenseVectors", &DenseVectorsType, NULL},
    {"DenseProducts", &DenseProductsType, NULL},
    {"Monolith", &MonolithType, NULL},
    {"Deferred", &DeferredType, NULL},
    {"On", &OnType, NULL},
    {"DeferredOn", &DeferredOnType, NULL},

    {"DenseIndex", &DenseIndexType, NULL},
    {"DenseMetadata", &DenseMetadataType, &usearch_py_dense_metadata_desc},
    {"DenseIndexDescription", &DenseIndexDescriptionType, &usearch_py_dense_index_description_desc},
    {"DenseReachability", &DenseReachabilityType, &usearch_py_dense_reachability_desc},
    {"RefineReport", &RefineReportType, &usearch_py_refine_report_desc},
    {"Sample", &SampleType, &usearch_py_sample_desc},
    {"DenseTraversalStats", &DenseTraversalStatsType, NULL},
    {"DenseCursor", &DenseCursorType, NULL},
    {"DenseBatch", &DenseBatchType, NULL},
    {"DenseSearchStream", &DenseSearchStreamType, NULL},

    {"SequenceIndex", &SequenceIndexType, NULL},
    {"ShapesIndex", &ShapesIndexType, NULL},
    {"GeospatialIndex", &GeospatialIndexType, NULL},
};

static PyMethodDef module_methods[] = {
    {"version", (PyCFunction)api_version, METH_NOARGS, doc_version},                                              //
    {"capabilities_compiled", (PyCFunction)api_capabilities_compiled, METH_NOARGS, doc_capabilities_compiled},    //
    {"capabilities_available", (PyCFunction)api_capabilities_available, METH_NOARGS, doc_capabilities_available}, //
    {"layouts", (PyCFunction)api_layouts, METH_NOARGS, doc_layouts},                                              //
    {NULL, NULL, 0, NULL},
};

// The dotted name the extension is built as, so `lib.__name__` agrees with where the import
// machinery found it. The types keep the undotted `usearch.` prefix in their `tp_name`, which
// is where the package re-exports them and therefore where `pickle` looks.
static PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT, .m_name = "usearch.lib", .m_doc = doc_module, .m_size = -1, .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit_lib(void) {
    size_t const types_count = usearch_py_table_size_(module_types);
    for (size_t type_index = 0; type_index != types_count; ++type_index) {
        PyTypeObject* const type = module_types[type_index].type;
        PyStructSequence_Desc* const layout = module_types[type_index].layout;
        if ((layout ? PyStructSequence_InitType2(type, layout) : PyType_Ready(type)) < 0) return NULL;
    }

    PyObject* const module = PyModule_Create(&module_definition);
    if (!module) return NULL;

#ifdef Py_GIL_DISABLED
    // Index handles are shareable across threads by contract, and every blocking verb drops
    // the GIL around its call, so nothing here needs the interpreter lock to stay correct.
    PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
#endif

    if (usearch_py_register_error_types(module) < 0 || usearch_py_register_enums(module) < 0 ||
        PyModule_AddStringConstant(module, "__version__", usearch_version_name()) < 0)
        goto failed;

    for (size_t type_index = 0; type_index != types_count; ++type_index) {
        char const* const name = module_types[type_index].name;
        if (name && PyModule_AddObjectRef(module, name, (PyObject*)module_types[type_index].type) < 0) goto failed;
    }
    return module;

failed:
    Py_DECREF(module);
    return NULL;
}
