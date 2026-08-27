/**
 *  @brief The vocabulary verbs that belong to no modality - version and capabilities.
 *
 *  @file test/stub/types.cpp
 *
 *  The engine renders these from NumKong's dispatch mask in @c src/types.cpp. A stand-in
 *  dispatches to nothing, so it reports @c serial and says so wherever a binding asks.
 */
#include "usearch/types.h"

extern "C" {

char const* usearch_version(void) { return "3.0.0"; }
char const* usearch_capabilities_compiled(void) { return "serial"; }
char const* usearch_capabilities_available(void) { return "serial"; }

} // extern "C"
