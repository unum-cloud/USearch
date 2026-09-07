/**
 *  @brief The one translation unit of the @c USearchC target, which exists so SwiftPM has a
 *      compiled half to hang the public headers off: a target declaring @c publicHeadersPath
 *      is what puts @c include/ on the header search path of everything importing the module,
 *      and the modality headers reach each other through that path.
 *
 *  @file swift/shim.c
 *
 *  Nothing here implements the ABI - the engine is built by CMake into @c libusearch_c and
 *  linked, since @c src/ pulls the NumKong, ForkUnion, and StringZilla stack and carries a CUDA
 *  sibling SwiftPM cannot drive. Including the headers does buy one check: the C surface has to
 *  keep parsing as C, not merely as C++.
 */
#include "usearch/types.h"
#include "usearch/dense_index.h"
