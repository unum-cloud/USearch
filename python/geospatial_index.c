/**
 *  @file python/geospatial_index.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief CPython similarity search over space and time.
 *
 *  Entries are points, trajectories, and boundaries rather than embeddings, so the handle
 *  carries a layout, a structure, and a characteristic speed instead of a quantization, and
 *  the surface adds a range query beside the nearest-neighbor one.
 */
#include "lib.h"

static char const doc_GeospatialIndex[] =                                                   //
    "GeospatialIndex(...)\n"                                                                //
    "\n"                                                                                    //
    "Similarity search over space and time - Karney geodesics, trajectory alignment, and\n" //
    "boundary similarities.\n"                                                              //
    "\n"                                                                                    //
    "Not yet instantiated in this build; constructing one raises NotImplementedError.";

static PyObject* GeospatialIndex_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_NotImplementedError, "the geospatial modality is not yet instantiated in this build");
    return NULL;
}

PyTypeObject GeospatialIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.GeospatialIndex",
    .tp_doc = doc_GeospatialIndex,
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = GeospatialIndex_tp_new,
};
