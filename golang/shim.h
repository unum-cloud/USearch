/**
 *  @file golang/shim.h
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief Declarations the Go binding needs and the C header cannot give it: the Go-side
 *      trampolines a filtered search, a progress handler, and an attention's value lookup
 *      install as C callbacks.
 *
 *  The definitions live in @c shim.c because a Go file carrying an @c //export directive may hold
 *  only declarations in its preamble.
 */
#ifndef UNUM_USEARCH_GO_SHIM_H
#define UNUM_USEARCH_GO_SHIM_H

#include "usearch/dense_index.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief The callbacks Go defines and C calls - the symbols here that cross in that direction,
 *      which is why they alone wear Go's spelling.
 *
 *  cgo emits a C symbol named exactly as the Go function, and Go spells an unexported function
 *  in camel case, so the names are not this file's to choose. cgo also drops @c const from the
 *  pointers it declares, so the predicate and the value lookup reach C through adapters in
 *  @c shim.c. Everything else here is C that Go calls, and reads as C.
 */
void usearchAdmits(usearch_key_t* keys, usearch_admission_t* admissions, size_t count, void* state);
usearch_continuation_t usearchContinues(size_t completed, size_t total, void* state);
void usearchLooksUpValues(void* state, usearch_key_t* keys, size_t count, usearch_scalar_t scalar, size_t columns,
                          void** rows);

/** Hands back the trampolines above as function pointers. cgo renders a C function as a Go
 *  wrapper rather than an address, so the pointers have to be taken on this side. */
usearch_predicate_callback_t usearch_go_admits(void);
usearch_progress_callback_t usearch_go_continues(void);
usearch_value_lookup_t usearch_go_looks_up_values(void);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_GO_SHIM_H
