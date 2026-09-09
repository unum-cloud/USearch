/**
 *  @brief The module-init glue - `PyModuleDef`, type registration, and `PyInit_lib`.
 *  @file python/lib.c
 *  @author Ash Vardanian
 *
 *  Every modality contributes rows to one table here and its verbs to one file beside this
 *  one, so a modality joining the library touches two lines of glue and nothing else.
 */
#include "lib.h"

static char const doc_module[] =                                                             //
    "Smaller & Faster Single-File Vector Search Engine from Unum.\n"                         //
    "\n"                                                                                     //
    "One index type per modality, each sitting directly on the stable C ABI. Every verb\n"   //
    "is a batch over buffer-protocol inputs, so a memory-mapped dataset reaches the index\n" //
    "with no reshape and no copy, and NumPy is never required.";

static struct {
    char const* name;
    PyTypeObject* type;
} const module_types[] = {
    {"Key", &KeyType},
    {"Stats", &StatsType},
    {"Match", &MatchType},
    {"Matches", &MatchesType},
    {"Executor", &ExecutorType},

    {"DenseIndex", &DenseIndexType},
    {NULL, &DenseCursorType}, //! Returned by `cursor`, never constructed by name.
    {NULL, &DenseFinderType}, //! Returned by `find`, likewise.

    {"SequenceIndex", &SequenceIndexType},
    {"MatrixIndex", &MatrixIndexType},
    {"GeospatialIndex", &GeospatialIndexType},
};

static PyMethodDef module_methods[] = {
    {"version", (PyCFunction)api_version, METH_NOARGS, doc_version},                                              //
    {"capabilities_compiled", (PyCFunction)api_capabilities_compiled, METH_NOARGS, doc_capabilities_compiled},    //
    {"capabilities_available", (PyCFunction)api_capabilities_available, METH_NOARGS, doc_capabilities_available}, //
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
    for (size_t type_index = 0; type_index != types_count; ++type_index)
        if (PyType_Ready(module_types[type_index].type) < 0) return NULL;

    PyObject* const module = PyModule_Create(&module_definition);
    if (!module) return NULL;

#ifdef Py_GIL_DISABLED
    // Index handles are shareable across threads by contract, and every blocking verb drops
    // the GIL around its call, so nothing here needs the interpreter lock to stay correct.
    PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
#endif

    if (usearch_py_register_error_type(module) < 0) goto failed;
    if (PyModule_AddStringConstant(module, "__version__", usearch_version()) < 0) goto failed;

    for (size_t type_index = 0; type_index != types_count; ++type_index) {
        char const* const name = module_types[type_index].name;
        if (!name) continue;
        PyTypeObject* const type = module_types[type_index].type;
        Py_INCREF(type);
        // Only a successful `PyModule_AddObject` steals the reference, so undo just this one
        // and let the dying module release the types it already owns.
        if (PyModule_AddObject(module, name, (PyObject*)type) < 0) {
            Py_DECREF(type);
            goto failed;
        }
    }

    return module;

failed:
    Py_DECREF(module);
    return NULL;
}
