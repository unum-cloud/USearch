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

#pragma region Vocabulary

char const* usearch_status_name(usearch_status_t status) {
    switch (status) {
    case usearch_success_k: return "success";
    case usearch_error_out_of_memory_k: return "out-of-memory";
    case usearch_error_shape_mismatch_k: return "shape-mismatch";
    case usearch_error_unsupported_layout_k: return "unsupported-layout";
    case usearch_error_unknown_kind_k: return "unknown-kind";
    case usearch_error_read_only_k: return "read-only";
    case usearch_error_io_k: return "io";
    case usearch_error_device_k: return "device";
    case usearch_error_malformed_input_k: return "malformed-input";
    case usearch_error_internal_k: return "internal";
    case usearch_error_overflow_k: return "overflow";
    case usearch_error_uninitialized_k: return "uninitialized";
    case usearch_error_config_mismatch_k: return "config-mismatch";
    case usearch_error_contended_k: return "contended";
    case usearch_error_cancelled_k: return "cancelled";
    default: return "unrecognized";
    }
}

usearch_status_t usearch_status_from_name(usearch_bytes_view_t name) {
    for (int code = usearch_success_k; code <= usearch_error_cancelled_k; ++code) {
        usearch_status_t const status = static_cast<usearch_status_t>(code);
        if (named(name, usearch_status_name(status))) return status;
    }
    return usearch_error_internal_k;
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
    for (int code = usearch_scalar_f64_k; code <= usearch_scalar_u1x8_k; ++code) {
        usearch_scalar_t const scalar_kind = static_cast<usearch_scalar_t>(code);
        if (named(name, usearch_scalar_name(scalar_kind))) return scalar_kind;
    }
    return usearch_scalar_unknown_k;
}

char const* usearch_residency_name(usearch_residency_t residency) {
    switch (residency) {
    case usearch_residency_vram_k: return "vram";
    case usearch_residency_nvme_k: return "nvme";
    default: return "ram";
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
