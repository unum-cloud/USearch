/**
 *  @file python/sequence_index.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief CPython similarity search over variable-length sequences.
 *
 *  The C surface exists; the verbs land here as the library instantiates them. Entries vary
 *  in size, so batches arrive as `usearch_bytes_view_t` tapes rather than strided matrices, the
 *  handle carries no dimensions or quantization, and `count` gains a stats out-parameter the
 *  dense modality has no use for.
 */
#include "lib.h"

static char const doc_SequenceIndex[] =                                                   //
    "SequenceIndex(...)\n"                                                                //
    "\n"                                                                                  //
    "Similarity search over variable-length sequences - Levenshtein, Needleman-Wunsch,\n" //
    "and overlap scoring over byte and UTF-8 tapes.\n"                                    //
    "\n"                                                                                  //
    "Not yet instantiated in this build; constructing one raises NotImplementedError.";

static PyObject* SequenceIndex_tp_new(PyTypeObject* type, PyObject* args, PyObject* keywords) {
    PyErr_SetString(PyExc_NotImplementedError, "the sequence modality is not yet instantiated in this build");
    return NULL;
}

PyTypeObject SequenceIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "usearch.SequenceIndex",
    .tp_doc = doc_SequenceIndex,
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = SequenceIndex_tp_new,
};
