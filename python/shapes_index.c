/**
 *  @file python/shapes_index.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief CPython similarity search over 3D point clouds of any size.
 *
 *  Entries are point clouds rather than vectors, so the geometry is an optionally frozen point
 *  count, and the metrics score whole shapes against each other - Chamfer, Hausdorff,
 *  Wasserstein-Procrustes, and Gromov-Wasserstein over molecules, conformers, and protein
 *  backbones.
 */
#include "lib.h"

static char const doc_ShapesIndex[] =                                                      //
    "ShapesIndex(...)\n"                                                                   //
    "\n"                                                                                   //
    "Similarity search over 3D point clouds of any size, such as molecules, conformers,\n" //
    "and protein backbones, under Chamfer, Hausdorff, Wasserstein-Procrustes, and\n"       //
    "Gromov-Wasserstein.\n"                                                                //
    "\n"                                                                                   //
    "Not yet instantiated in this build; constructing one raises NotImplementedError.";

static PyObject* ShapesIndex_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_NotImplementedError, "the shapes modality is not yet instantiated in this build");
    return NULL;
}

PyTypeObject ShapesIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.ShapesIndex",
    .tp_doc = doc_ShapesIndex,
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = ShapesIndex_tp_new,
};
