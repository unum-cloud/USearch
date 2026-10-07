/**
 *  @file src_stub/types.cpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief The vocabulary verbs that belong to no modality - outcomes, arenas, executors, version,
 *      capabilities.
 *
 *  The engine renders the capability strings from NumKong's dispatch mask in @c src/types.cpp. A
 *  stand-in dispatches to nothing, so it reports @c serial and says so wherever a binding asks.
 *
 *  Its executors are inert by the same logic. A handle is spawned, validated and freed exactly as
 *  the engine's is, so a binding suite exercises the whole lifetime, but no thread and no stream
 *  ever exists behind it: every verb runs to completion on the caller's thread before it returns,
 *  as the headers allow an executor without threads of its own to. Only the RAM constructors
 *  succeed, because those are the residencies this library implements at all.
 *
 *  Nothing here queues, so an in-flight table never fills: two threads driving one executor both
 *  run, where the engine may refuse the second with @c usearch_status_contended_k.
 */
#include "usearch/types.h"

#include "lib.hpp"

using namespace unum::usearch;
using namespace unum::usearch::stub;

extern "C" {

void usearch_version(int* major, int* minor, int* patch) {
    if (major) *major = USEARCH_VERSION_MAJOR;
    if (minor) *minor = USEARCH_VERSION_MINOR;
    if (patch) *patch = USEARCH_VERSION_PATCH;
}

char const* usearch_version_name(void) { return "3.0.0"; }
char const* usearch_capabilities_compiled(void) { return "serial"; }
char const* usearch_capabilities_available(void) { return "serial"; }

usearch_status_t usearch_outcome_wait(usearch_outcome_t const* outcome) {
    if (!outcome) return usearch_status_invalid_argument_k;
    std::atomic_ref status(const_cast<usearch_status_t&>(outcome->status));
    usearch_status_t current = status.load(std::memory_order_acquire);
    for (; current == usearch_status_pending_k; current = status.load(std::memory_order_acquire))
        status.wait(usearch_status_pending_k, std::memory_order_acquire);
    return current;
}

void usearch_arena_free(usearch_arena_t arena) { delete arena; }

void usearch_executor_for_ram(usearch_executor_for_ram_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    usearch_executor_handle_t* const executor = new (std::nothrow)
        usearch_executor_handle_t {arguments->threads ? arguments->threads : 1};
    if (reject_if(!executor, usearch_status_out_of_memory_k, "The executor could not be allocated", outcome)) return;
    *arguments->executor = executor;
}

void usearch_executor_on_pool(usearch_executor_on_pool_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    if (reject_if(!arguments->pool, usearch_status_invalid_argument_k, "The pool must not be null", outcome)) return;
    // The pool is borrowed and never driven, every verb running on the thread that called it.
    usearch_executor_handle_t* const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (reject_if(!executor, usearch_status_out_of_memory_k, "The executor could not be allocated", outcome)) return;
    *arguments->executor = executor;
}

void usearch_executor_for_nvme(usearch_executor_for_nvme_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "This build carries no NVMe residency");
}

void usearch_executor_for_vram(usearch_executor_for_vram_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no CUDA backends");
}

void usearch_executor_drain(usearch_executor_drain_t* arguments) {
    // Nothing is ever queued, every verb having run on the thread that called it.
    pending_outcome_t outcome(arguments->outcome);
}

void usearch_executor_free(usearch_executor_t executor) { delete executor; }

#pragma region Vocabulary

char const* usearch_status_name(usearch_status_t status) {
    switch (status) {
    case usearch_status_success_k: return "success";
    case usearch_status_out_of_memory_k: return "out-of-memory";
    case usearch_status_shape_mismatch_k: return "shape-mismatch";
    case usearch_status_unsupported_layout_k: return "unsupported-layout";
    case usearch_status_unknown_kind_k: return "unknown-kind";
    case usearch_status_read_only_k: return "read-only";
    case usearch_status_io_k: return "io";
    case usearch_status_device_k: return "device";
    case usearch_status_malformed_k: return "malformed";
    case usearch_status_internal_k: return "internal";
    case usearch_status_overflow_k: return "overflow";
    case usearch_status_uninitialized_k: return "uninitialized";
    case usearch_status_config_mismatch_k: return "config-mismatch";
    case usearch_status_contended_k: return "contended";
    case usearch_status_cancelled_k: return "cancelled";
    case usearch_status_unsupported_k: return "unsupported";
    case usearch_status_pending_k: return "pending";
    case usearch_status_invalid_argument_k: return "invalid-argument";
    // An unrecognized value spells like the status an unrecognized name reads back as.
    default: return "invalid-argument";
    }
}

usearch_status_t usearch_status_from_name(usearch_bytes_view_t name) {
    for (int code = usearch_status_success_k; code <= usearch_status_invalid_argument_k; ++code) {
        usearch_status_t const status = static_cast<usearch_status_t>(code);
        if (named(name, usearch_status_name(status))) return status;
    }
    return usearch_status_invalid_argument_k;
}

char const* usearch_scalar_name(usearch_scalar_t scalar_kind) {
    switch (scalar_kind) {
    case usearch_scalar_f64_k: return "f64";
    case usearch_scalar_f32_k: return "f32";
    case usearch_scalar_bf16_k: return "bf16";
    case usearch_scalar_f16_k: return "f16";
    case usearch_scalar_e5m2_k: return "e5m2";
    case usearch_scalar_e4m3_k: return "e4m3";
    case usearch_scalar_e3m2_k: return "e3m2";
    case usearch_scalar_e2m3_k: return "e2m3";
    case usearch_scalar_i8_k: return "i8";
    case usearch_scalar_i4x2_k: return "i4x2";
    case usearch_scalar_u8_k: return "u8";
    case usearch_scalar_u4x2_k: return "u4x2";
    case usearch_scalar_u1x8_k: return "u1x8";
    default: return "unknown";
    }
}

usearch_scalar_t usearch_scalar_from_name(usearch_bytes_view_t name) {
    // The enumerators jump over the gaps their families leave, and every gap spells "unknown", so
    // a counted walk would read that spelling back as a code naming no scalar.
    usearch_scalar_t const kinds[] = {
        usearch_scalar_f64_k,  usearch_scalar_f32_k,  usearch_scalar_bf16_k, usearch_scalar_f16_k,
        usearch_scalar_e5m2_k, usearch_scalar_e4m3_k, usearch_scalar_e3m2_k, usearch_scalar_e2m3_k,
        usearch_scalar_i8_k,   usearch_scalar_i4x2_k, usearch_scalar_u8_k,   usearch_scalar_u4x2_k,
        usearch_scalar_u1x8_k,
    };
    for (usearch_scalar_t const scalar_kind : kinds)
        if (named(name, usearch_scalar_name(scalar_kind))) return scalar_kind;
    return usearch_scalar_unknown_k;
}

char const* usearch_residency_name(usearch_residency_t residency) {
    switch (residency) {
    case usearch_residency_ram_k: return "ram";
    case usearch_residency_vram_k: return "vram";
    case usearch_residency_nvme_k: return "nvme";
    default: return "unknown";
    }
}

usearch_residency_t usearch_residency_from_name(usearch_bytes_view_t name) {
    if (named(name, "vram")) return usearch_residency_vram_k;
    if (named(name, "nvme")) return usearch_residency_nvme_k;
    return usearch_residency_ram_k;
}

char const* usearch_cpu_class_name(usearch_cpu_class_t) { return "serial"; }
char const* usearch_gpu_class_name(usearch_gpu_class_t) { return "none"; }

#pragma endregion Vocabulary

} // extern "C"
