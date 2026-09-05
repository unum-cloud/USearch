/**
 *  @brief The vocabulary verbs that belong to no modality - version, capabilities, executors.
 *
 *  @file src_stub/types.cpp
 *
 *  The engine renders the capability strings from NumKong's dispatch mask in @c src/types.cpp. A
 *  stand-in dispatches to nothing, so it reports @c serial and says so wherever a binding asks.
 *
 *  Its executors are inert by the same logic. A handle is spawned, validated and freed exactly as
 *  the engine's is, so a binding suite exercises the whole lifetime, but no thread and no stream
 *  ever exists behind it: every verb runs on the caller's thread. Only the RAM constructor
 *  succeeds, because those are the residencies this library implements at all.
 */
#include "usearch/types.h"

#include "lib.hpp"

namespace us = unum::usearch;
using namespace unum::usearch;
using namespace unum::usearch::stub;

extern "C" {

char const* usearch_version(void) { return "3.0.0"; }
char const* usearch_capabilities_compiled(void) { return "serial"; }
char const* usearch_capabilities_available(void) { return "serial"; }

usearch_executor_t usearch_executor_for_ram(usearch_ram_options_t const*, std::size_t threads, usearch_error_t* error) {
    propagate_success(error);
    executor_t* const executor = new (std::nothrow) executor_t {threads ? threads : 1};
    if (!executor) {
        propagate_error(usearch_error_out_of_memory_k, error, "The executor could not be allocated");
        return nullptr;
    }
    return executor;
}

usearch_executor_t usearch_executor_for_nvme(usearch_nvme_options_t const*, std::size_t, usearch_error_t* error) {
    propagate_error(usearch_error_device_k, error, "This build carries no NVMe backends");
    return nullptr;
}

usearch_executor_t usearch_executor_for_vram(usearch_vram_options_t const*, usearch_error_t* error) {
    propagate_error(usearch_error_device_k, error, "This build carries no CUDA backends");
    return nullptr;
}

void usearch_executor_drain(usearch_executor_t executor, usearch_error_t* error) {
    propagate_success(error);
    // Nothing is ever queued, every verb having run on the thread that called it.
    reject_if(!executor, usearch_error_malformed_input_k, "The executor must not be null", error);
}

void usearch_executor_free(usearch_executor_t executor) { delete static_cast<executor_t*>(executor); }

} // extern "C"
