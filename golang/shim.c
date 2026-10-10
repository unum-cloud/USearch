/**
 *  @file golang/shim.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief The addresses of the Go-side trampolines, taken where C can take them.
 */
#include "shim.h"

usearch_predicate_t usearch_go_admits(void) { return &usearchAdmits; }

usearch_progress_callback_t usearch_go_continues(void) { return &usearchContinues; }

static void usearch_go_value_lookup(void* state, usearch_key_t const* keys, size_t count, usearch_scalar_t scalar,
                                    size_t columns, void const** rows) {
    usearchLooksUpValues(state, (usearch_key_t*)keys, count, scalar, columns, (void**)rows);
}

usearch_value_lookup_t usearch_go_looks_up_values(void) { return &usearch_go_value_lookup; }
