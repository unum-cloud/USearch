/**
 *  @file include/usearch/usearch.h
 *  @author Ash Vardanian
 *  @date October 1, 2026
 *  @brief The umbrella of the USearch C interface: every modality header and the vocabulary they
 *      share, behind one include.
 *
 *  A caller using one modality may include its header alone; this one exists so that a binding
 *  generator, or a caller using several, names the whole interface once.
 */
#ifndef UNUM_USEARCH_H
#define UNUM_USEARCH_H

#include "usearch/types.h"

#include "usearch/dense_index.h"
#include "usearch/geospatial_index.h"
#include "usearch/shapes_index.h"
#include "usearch/sequence_index.h"

#endif // UNUM_USEARCH_H
