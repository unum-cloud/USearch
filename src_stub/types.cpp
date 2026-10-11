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
 *  Its executors start no threads: a deferred call is held in one stub-wide list until its wait, its
 *  executor's next drain or blocking call, or the free of a handle it uses runs it on that caller's
 *  thread. Only the RAM constructors succeed.
 */
#include <algorithm>          // `std::find`, `std::find_if`, `std::any_of`, `std::count_if`
#include <condition_variable> // `std::condition_variable`
#include <iterator>           // `std::next`
#include <list>               // `std::list`
#include <mutex>              // `std::mutex`, `std::lock_guard`, `std::unique_lock`
#include <variant>            // `std::visit`
#include <vector>             // `std::vector`

#include "usearch/types.h"

#include "lib.hpp"

using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace unum {
namespace usearch {
namespace stub {

#pragma region Calls

/** Every call given to any executor that is not yet final, in one list, so a wait or a free finds the
 *  calls using its handle without asking each executor. */
struct calls_t {
    std::mutex guard;

    /** Signalled whenever a running call becomes final. */
    std::condition_variable finished;

    /** Deferred calls no drain has started yet. */
    std::list<call_t> held;

    /** Calls running on some caller's thread right now. */
    std::list<call_t> running;
};

static calls_t calls;

bool submit_deferred(call_t call, usearch_outcome_t* outcome) {
    executor_t const* const executor = call.deferred_on;
    auto const in_slot_of_executor = [=](call_t const& listed) { return listed.deferred_on == executor; };
    std::lock_guard<std::mutex> lock(calls.guard);
    std::size_t const slots_taken = std::count_if(calls.held.begin(), calls.held.end(), in_slot_of_executor) +
                                    std::count_if(calls.running.begin(), calls.running.end(), in_slot_of_executor);
    if (slots_taken == executor->deferred_limit)
        return propagate_error(usearch_status_contended_k, outcome, "The deferred table is full"), false;
    propagate_deferred(outcome);
    calls.held.push_back(std::move(call));
    return true;
}

/** Runs a call already listed as running, then delists it and wakes every waiter. */
static void run_listed(std::list<call_t>::iterator call) {
    std::visit([&](auto& arguments) { call->verb(&arguments); }, call->arguments);
    std::lock_guard<std::mutex> lock(calls.guard);
    calls.running.erase(call);
    calls.finished.notify_all();
}

void run_blocking(call_t call) {
    std::list<call_t>::iterator running;
    {
        std::lock_guard<std::mutex> lock(calls.guard);
        running = calls.running.insert(calls.running.end(), std::move(call));
    }
    run_listed(running);
}

void drain(executor_t const* executor) {
    std::vector<std::list<call_t>::iterator> started;
    {
        std::lock_guard<std::mutex> lock(calls.guard);
        for (auto call = calls.held.begin(); call != calls.held.end();) {
            auto const next = std::next(call);
            if (call->deferred_on == executor)
                calls.running.splice(calls.running.end(), calls.held, call), started.push_back(call);
            call = next;
        }
    }
    for (auto const call : started) run_listed(call);
}

void wait_for_calls_using(void const* handle) {
    if (!handle) return;
    auto const uses_handle = [=](call_t const& call) {
        return std::find(call.handles.begin(), call.handles.end(), handle) != call.handles.end();
    };
    std::unique_lock<std::mutex> lock(calls.guard);
    for (;;) {
        auto const held = std::find_if(calls.held.begin(), calls.held.end(), uses_handle);
        if (held != calls.held.end()) {
            // Draining runs verbs that take this lock themselves, so it runs outside it.
            executor_t const* const executor = held->deferred_on;
            lock.unlock();
            drain(executor);
            lock.lock();
        }
        else if (std::any_of(calls.running.begin(), calls.running.end(), uses_handle)) calls.finished.wait(lock);
        else return;
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
    std::atomic_ref status(outcome->status);
    usearch_status_t current = status.load(std::memory_order_acquire);
    for (; current == usearch_status_deferred_k; current = status.load(std::memory_order_acquire))
        status.wait(usearch_status_deferred_k, std::memory_order_acquire);
    return current;
}

usearch_status_t usearch_outcome_status(usearch_outcome_t* outcome) {
    if (!outcome) return usearch_status_invalid_argument_k;
    return std::atomic_ref(outcome->status).load(std::memory_order_acquire);
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

void usearch_executor_for_ram(usearch_executor_for_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    usearch_executor_handle_t* const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (reject_if(!executor, usearch_status_out_of_memory_k, "The executor could not be allocated", outcome)) return;
    if (arguments->threads) executor->threads = arguments->threads;
    if (arguments->deferred_limit) executor->deferred_limit = arguments->deferred_limit;
    *arguments->executor = executor;
}

void usearch_executor_on_pool(usearch_executor_on_pool_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    if (reject_if(!arguments->pool, usearch_status_invalid_argument_k, "The pool must not be null", outcome)) return;
    // The pool is borrowed and never driven, every verb running on the thread that called it.
    usearch_executor_handle_t* const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (reject_if(!executor, usearch_status_out_of_memory_k, "The executor could not be allocated", outcome)) return;
    if (arguments->deferred_limit) executor->deferred_limit = arguments->deferred_limit;
    *arguments->executor = executor;
}

void usearch_executor_for_nvme(usearch_executor_for_nvme_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "This build carries no NVMe residency");
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
    deferred_outcome_t outcome(arguments->outcome);
    if (arguments->executor) drain(arguments->executor);
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

usearch_residency_t usearch_residency_from_name(usearch_bytes_view_t name) {
    if (named(name, "vram")) return usearch_residency_vram_k;
    if (named(name, "nvme")) return usearch_residency_nvme_k;
    return usearch_residency_ram_k;
}

char const* usearch_traversal_name(usearch_traversal_t traversal) {
    switch (traversal) {
    case usearch_traversal_walk_k: return "walk";
    case usearch_traversal_scan_k: return "scan";
    default: return "unknown";
    }
}

usearch_traversal_t usearch_traversal_from_name(usearch_bytes_view_t name) {
    return named(name, "scan") ? usearch_traversal_scan_k : usearch_traversal_walk_k;
}

char const* usearch_mood_name(usearch_mood_t mood) {
    switch (mood) {
    case usearch_mood_parked_k: return "parked";
    case usearch_mood_spinning_k: return "spinning";
    default: return "unknown";
    }
}

usearch_mood_t usearch_mood_from_name(usearch_bytes_view_t name) {
    return named(name, "spinning") ? usearch_mood_spinning_k : usearch_mood_parked_k;
}

char const* usearch_cpu_class_name(usearch_cpu_class_t) { return "serial"; }
char const* usearch_gpu_class_name(usearch_gpu_class_t) { return "none"; }

#pragma endregion Vocabulary

} // extern "C"
