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
 *  Its executors start no threads: every call runs on its caller at submission, whatever its
 *  completion, and one that meets a running call on its executor finishes contended. Only the RAM
 *  constructors succeed.
 */
#include <algorithm> // `std::find`, `std::any_of`
#include <mutex>     // `std::mutex`, `std::lock_guard`, `std::unique_lock`
#include <variant>   // `std::visit`
#include <vector>    // `std::vector`

#include "usearch/types.h"

#include "lib.hpp"

using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace unum {
namespace usearch {
namespace stub {

#pragma region Calls

/** Every call running on some caller's thread, so a wait or a free finds the calls using its handle
 *  without asking each executor. */
struct calls_t {
    std::mutex guard;

    /** Counted up whenever a running call becomes final, under @c guard; waiters read it bare. */
    u32_t finished = 0;

    /** Calls running on some caller's thread right now, each owned by the frame that runs it. */
    std::vector<call_t const*> running;
};

static calls_t calls;

/** Runs @p call, already listed as running, then delists it and wakes every waiter. */
static void run_listed(call_t& call) {
    std::visit([&](auto& arguments) { call.verb(&arguments); }, call.arguments);
    std::lock_guard<std::mutex> lock(calls.guard);
    calls.running.erase(std::find(calls.running.begin(), calls.running.end(), &call));
    serial_atomic_ref<u32_t>(calls.finished).add(1, std::memory_order_release);
}

void run_holding(call_t call, usearch_outcome_t* outcome) {
    {
        std::lock_guard<std::mutex> lock(calls.guard);
        if (std::any_of(calls.running.begin(), calls.running.end(),
                        [&](call_t const* running) { return running->held_executor == call.held_executor; }))
            return propagate_error(usearch_status_contended_k, outcome, "Another call holds the executor"), void();
        calls.running.push_back(&call);
    }
    run_listed(call);
}

void wait_for_calls_using(void const* handle) {
    if (!handle) return;
    auto const uses_handle = [=](call_t const* call) {
        return std::find(call->handles.begin(), call->handles.end(), handle) != call->handles.end();
    };
    std::unique_lock<std::mutex> lock(calls.guard);
    while (std::any_of(calls.running.begin(), calls.running.end(), uses_handle)) {
        u32_t const finished_before = calls.finished;
        lock.unlock(); // the running call delists itself under this lock
        serial_atomic_ref<u32_t const> const finished {calls.finished};
        while (finished.load(std::memory_order_acquire) == finished_before)
            serial_waiter_t {}(&calls.finished, finished_before, 0u, fu::wait_capped_t {});
        lock.lock();
    }
}

#pragma endregion Calls

} // namespace stub
} // namespace usearch
} // namespace unum

extern "C" {

void usearch_version(int* major, int* minor, int* patch) {
    if (major) *major = USEARCH_VERSION_MAJOR;
    if (minor) *minor = USEARCH_VERSION_MINOR;
    if (patch) *patch = USEARCH_VERSION_PATCH;
}

char const* usearch_version_name(void) { return "3.0.0"; }
char const* usearch_capabilities_compiled(void) { return "serial"; }
char const* usearch_capabilities_available(void) { return "serial"; }

usearch_status_t usearch_outcome_wait(usearch_outcome_t* outcome) {
    if (!outcome) return usearch_status_invalid_argument_k;
    wait_for_calls_using(outcome);
    serial_atomic_ref<usearch_status_t const> const status {outcome->status};
    usearch_status_t current = status.load(std::memory_order_acquire);
    for (; current == usearch_status_deferred_k; current = status.load(std::memory_order_acquire))
        serial_waiter_t {}(&outcome->status, current, 0u, fu::wait_capped_t {});
    return current;
}

usearch_status_t usearch_outcome_status(usearch_outcome_t* outcome) {
    if (!outcome) return usearch_status_invalid_argument_k;
    return serial_atomic_ref<usearch_status_t const>(outcome->status).load(std::memory_order_acquire);
}

void usearch_arena_init(usearch_arena_init_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->arena = nullptr;
    // Every stand-in executor computes on the host, so every arena is host pages whoever made it.
    usearch_arena_handle_t* const arena = new (std::nothrow) usearch_arena_handle_t {};
    if (reject_if(!arena, usearch_status_out_of_memory_k, "The arena could not be allocated", outcome)) return;
    *arguments->arena = arena;
}

void usearch_arena_free(usearch_arena_t arena) {
    wait_for_calls_using(arena);
    delete arena;
}

void usearch_arena_describe(usearch_arena_describe_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    if (reject_if(!arguments->arena || !arguments->description, usearch_status_invalid_argument_k,
                  "The arena and description must not be NULL", outcome))
        return;
    *arguments->description = {usearch_residency_ram_k, 0};
}

void usearch_executor_for_ram(usearch_executor_for_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    usearch_executor_handle_t* const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (reject_if(!executor, usearch_status_out_of_memory_k, "The executor could not be allocated", outcome)) return;
    if (arguments->threads) executor->threads = arguments->threads;
    *arguments->executor = executor;
}

void usearch_executor_on_pool(usearch_executor_on_pool_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    if (reject_if(!arguments->pool, usearch_status_invalid_argument_k, "The pool must not be null", outcome)) return;
    // The pool is borrowed and never driven, every verb running on the thread that called it.
    usearch_executor_handle_t* const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (reject_if(!executor, usearch_status_out_of_memory_k, "The executor could not be allocated", outcome)) return;
    *arguments->executor = executor;
}

void usearch_executor_for_nvme(usearch_executor_for_nvme_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    if (reject_if(arguments->nvme.paths_count && !arguments->nvme.paths, usearch_status_invalid_argument_k,
                  "The paths must name storage", outcome))
        return;
    if (reject_if(static_cast<unsigned>(arguments->nvme_class) > usearch_nvme_class_windows_ioring_k,
                  usearch_status_invalid_argument_k, "The NVMe class is unknown", outcome))
        return;
    if (reject_if(arguments->nvme_class == usearch_nvme_class_linux_spdk_k, usearch_status_unsupported_k,
                  "This build carries no SPDK", outcome))
        return;
    // A checkpoint from storage is mapped and read on the host, so this executor is the RAM one.
    usearch_executor_handle_t* const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (reject_if(!executor, usearch_status_out_of_memory_k, "The executor could not be allocated", outcome)) return;
    if (arguments->threads) executor->threads = arguments->threads;
    *arguments->executor = executor;
}

void usearch_executor_for_vram(usearch_executor_for_vram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no GPU backends");
}

void usearch_gpu_devices(usearch_gpu_devices_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    if (arguments->devices_count) *arguments->devices_count = 0;
}

void usearch_executor_drain(usearch_executor_drain_t* arguments) {
    // Every call is final when it returns, so nothing waits behind a drain.
    deferred_outcome_t outcome(arguments->outcome);
}

void usearch_executor_free(usearch_executor_t executor) {
    wait_for_calls_using(executor);
    delete executor;
}

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
    case usearch_status_deferred_k: return "deferred";
    case usearch_status_invalid_argument_k: return "invalid-argument";
    default: return "unrecognized";
    }
}

usearch_status_t usearch_status_from_name(usearch_bytes_view_t name, usearch_status_t* status) {
    for (int code = usearch_status_success_k; code <= usearch_status_invalid_argument_k; ++code)
        if (named(name, usearch_status_name(static_cast<usearch_status_t>(code))))
            return *status = static_cast<usearch_status_t>(code), usearch_status_success_k;
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
    case usearch_scalar_i32_k: return "i32";
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
        usearch_scalar_i8_k,   usearch_scalar_i4x2_k, usearch_scalar_i32_k,  usearch_scalar_u8_k,
        usearch_scalar_u4x2_k, usearch_scalar_u1x8_k,
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

usearch_status_t usearch_residency_from_name(usearch_bytes_view_t name, usearch_residency_t* residency) {
    for (int code = usearch_residency_ram_k; code <= usearch_residency_nvme_k; ++code)
        if (named(name, usearch_residency_name(static_cast<usearch_residency_t>(code))))
            return *residency = static_cast<usearch_residency_t>(code), usearch_status_success_k;
    return usearch_status_invalid_argument_k;
}

char const* usearch_traversal_name(usearch_traversal_t traversal) {
    switch (traversal) {
    case usearch_traversal_walk_k: return "walk";
    case usearch_traversal_scan_k: return "scan";
    default: return "unknown";
    }
}

usearch_status_t usearch_traversal_from_name(usearch_bytes_view_t name, usearch_traversal_t* traversal) {
    if (named(name, "walk")) return *traversal = usearch_traversal_walk_k, usearch_status_success_k;
    if (named(name, "scan")) return *traversal = usearch_traversal_scan_k, usearch_status_success_k;
    return usearch_status_invalid_argument_k;
}

char const* usearch_load_path_name(usearch_load_path_t load_path) {
    switch (load_path) {
    case usearch_load_path_none_k: return "none";
    case usearch_load_path_host_copy_k: return "host-copy";
    case usearch_load_path_cufile_k: return "cufile";
    default: return "unknown";
    }
}

char const* usearch_mood_name(usearch_mood_t mood) {
    switch (mood) {
    case usearch_mood_parked_k: return "parked";
    case usearch_mood_spinning_k: return "spinning";
    default: return "unknown";
    }
}

usearch_status_t usearch_mood_from_name(usearch_bytes_view_t name, usearch_mood_t* mood) {
    if (named(name, "parked")) return *mood = usearch_mood_parked_k, usearch_status_success_k;
    if (named(name, "spinning")) return *mood = usearch_mood_spinning_k, usearch_status_success_k;
    return usearch_status_invalid_argument_k;
}

char const* usearch_cpu_class_name(usearch_cpu_class_t cpu_class) {
    switch (cpu_class) {
    case usearch_cpu_class_serial_k: return "serial";
    case usearch_cpu_class_intel_haswell_k: return "intel-haswell";
    case usearch_cpu_class_intel_sierra_k: return "intel-sierra";
    case usearch_cpu_class_intel_skylake_k: return "intel-skylake";
    case usearch_cpu_class_intel_icelake_k: return "intel-icelake";
    case usearch_cpu_class_amd_genoa_k: return "amd-genoa";
    case usearch_cpu_class_intel_sapphire_k: return "intel-sapphire";
    case usearch_cpu_class_intel_granite_k: return "intel-granite";
    case usearch_cpu_class_intel_diamond_k: return "intel-diamond";
    case usearch_cpu_class_aws_g2_k: return "aws-g2";
    case usearch_cpu_class_apple_m2_k: return "apple-m2";
    case usearch_cpu_class_apple_m4_k: return "apple-m4";
    case usearch_cpu_class_aws_g5_k: return "aws-g5";
    case usearch_cpu_class_nvidia_vera_k: return "nvidia-vera";
    case usearch_cpu_class_alibaba_c930_k: return "alibaba-c930";
    case usearch_cpu_class_wasm32_v128_k: return "wasm32-v128";
    case usearch_cpu_class_wasm64_extended_k: return "wasm64-extended";
    case usearch_cpu_class_monoversioned_k: return "monoversioned";
    default: return "unknown";
    }
}

usearch_cpu_class_t usearch_cpu_class_from_name(usearch_bytes_view_t name) {
    for (int code = usearch_cpu_class_serial_k; code <= usearch_cpu_class_monoversioned_k; ++code) {
        usearch_cpu_class_t const cpu_class = static_cast<usearch_cpu_class_t>(code);
        if (named(name, usearch_cpu_class_name(cpu_class))) return cpu_class;
    }
    return usearch_cpu_class_unknown_k;
}

char const* usearch_gpu_class_name(usearch_gpu_class_t gpu_class) {
    switch (gpu_class) {
    case usearch_gpu_class_nvidia_ampere_k: return "nvidia-ampere";
    case usearch_gpu_class_nvidia_ada_k: return "nvidia-ada";
    case usearch_gpu_class_nvidia_hopper_k: return "nvidia-hopper";
    case usearch_gpu_class_nvidia_blackwell_k: return "nvidia-blackwell";
    case usearch_gpu_class_nvidia_blackwellrtx_k: return "nvidia-blackwellrtx";
    case usearch_gpu_class_nvidia_rubin_k: return "nvidia-rubin";
    case usearch_gpu_class_amd_cdna3_k: return "amd-cdna3";
    case usearch_gpu_class_amd_cdna4_k: return "amd-cdna4";
    case usearch_gpu_class_amd_cdna5_k: return "amd-cdna5";
    case usearch_gpu_class_apple_metal7_k: return "apple-metal7";
    case usearch_gpu_class_apple_metal9_k: return "apple-metal9";
    case usearch_gpu_class_apple_metal10_k: return "apple-metal10";
    default: return "unknown";
    }
}

usearch_gpu_class_t usearch_gpu_class_from_name(usearch_bytes_view_t name) {
    for (int code = usearch_gpu_class_nvidia_ampere_k; code <= usearch_gpu_class_apple_metal10_k; ++code) {
        usearch_gpu_class_t const gpu_class = static_cast<usearch_gpu_class_t>(code);
        if (named(name, usearch_gpu_class_name(gpu_class))) return gpu_class;
    }
    return usearch_gpu_class_unknown_k;
}

char const* usearch_nvme_class_name(usearch_nvme_class_t nvme_class) {
    switch (nvme_class) {
    case usearch_nvme_class_posix_k: return "posix";
    case usearch_nvme_class_linux_aio_k: return "linux-aio";
    case usearch_nvme_class_linux_uring_k: return "linux-uring";
    case usearch_nvme_class_linux_uring_passthrough_k: return "linux-uring-passthrough";
    case usearch_nvme_class_linux_spdk_k: return "linux-spdk";
    case usearch_nvme_class_windows_ioring_k: return "windows-ioring";
    default: return "unknown";
    }
}

usearch_nvme_class_t usearch_nvme_class_from_name(usearch_bytes_view_t name) {
    for (int code = usearch_nvme_class_posix_k; code <= usearch_nvme_class_windows_ioring_k; ++code) {
        usearch_nvme_class_t const nvme_class = static_cast<usearch_nvme_class_t>(code);
        if (named(name, usearch_nvme_class_name(nvme_class))) return nvme_class;
    }
    return usearch_nvme_class_unknown_k;
}

#pragma endregion Vocabulary

#pragma region Arrow

usearch_status_t usearch_keys_view_from_arrow(ArrowSchema const*, ArrowArray const*, usearch_keys_view_t*, size_t*) {
    return usearch_status_unsupported_k;
}
usearch_status_t usearch_rows_view_from_arrow(ArrowSchema const*, ArrowArray const*, usearch_rows_view_t*, size_t*) {
    return usearch_status_unsupported_k;
}
usearch_status_t usearch_tape_view_from_arrow(ArrowSchema const*, ArrowArray const*, usearch_tape_view_t*, size_t*) {
    return usearch_status_unsupported_k;
}
usearch_status_t usearch_keys_view_to_arrow(usearch_keys_view_t, size_t, usearch_arena_t, ArrowSchema*,
                                            ArrowDeviceArray*) {
    return usearch_status_unsupported_k;
}
usearch_status_t usearch_rows_view_to_arrow(usearch_rows_view_t, size_t, usearch_arena_t, ArrowSchema*,
                                            ArrowDeviceArray*) {
    return usearch_status_unsupported_k;
}
usearch_status_t usearch_tape_view_to_arrow(usearch_tape_view_t, size_t, usearch_arena_t, ArrowSchema*,
                                            ArrowDeviceArray*) {
    return usearch_status_unsupported_k;
}
usearch_status_t usearch_keys_tape_view_to_arrow(usearch_keys_tape_view_t, size_t, usearch_arena_t, ArrowSchema*,
                                                 ArrowDeviceArray*) {
    return usearch_status_unsupported_k;
}
usearch_status_t usearch_matches_tape_view_to_arrow(usearch_matches_tape_view_t, size_t, usearch_arena_t, ArrowSchema*,
                                                    ArrowDeviceArray*) {
    return usearch_status_unsupported_k;
}

#pragma endregion Arrow

} // extern "C"
