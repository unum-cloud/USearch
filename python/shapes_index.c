/**
 *  @file python/shapes_index.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief CPython similarity search over rank-2 tensors.
 *
 *  Entries are token matrices rather than vectors, so the geometry is a column count with an
 *  optionally frozen row count, and the metrics score whole matrices against each other -
 *  Chamfer and Hausdorff for late interaction, Gromov-Wasserstein, Kabsch, and Umeyama for
 *  molecular conformers.
 */
#include "lib.h"

static char const doc_MatrixIndex[] =                                                     //
    "MatrixIndex(...)\n"                                                                  //
    "\n"                                                                                  //
    "Similarity search over rank-2 tensors - late interaction over token matrices, and\n" //
    "Gromov-Wasserstein over molecular conformers.\n"                                     //
    "\n"                                                                                  //
    "Not yet instantiated in this build; constructing one raises NotImplementedError.";

static PyObject* MatrixIndex_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_NotImplementedError, "the matrix modality is not yet instantiated in this build");
    return NULL;
}

PyTypeObject MatrixIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.MatrixIndex",
    .tp_doc = doc_MatrixIndex,
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = MatrixIndex_tp_new,
};
