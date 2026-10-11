/**
 *  @file include/usearch/utilities/vram_apple_device.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Host-side vocabulary for driving one Apple GPU - specs, queues, shared buffers,
 *      pipelines.
 *
 *  Metal's API is Objective-C, reached here through @c objc_msgSend cast to each call's exact
 *  prototype, so a plain C++ unit drives it and links against Metal, Foundation and the Objective-C
 *  runtime alone. Kernels travel as source and compile on the device at first use, so no Metal
 *  toolchain is needed to build. Memory is unified and every buffer is shared, so the host reads
 *  and writes a plane in place and a launch binds the same buffer by handle.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::strcmp`, `std::memcpy`

#include <mutex>   // `std::mutex`, `std::unique_lock` - the pipeline registry and the executor's hold
#include <span>    // `std::span` - the device list an executor opens over
#include <utility> // `std::exchange`, `std::move`

#include <objc/message.h> // `objc_msgSend`
#include <objc/runtime.h> // `sel_registerName`, `objc_getClass`

#include <forkunion/types.hpp> // `fu::spin_mutex_t` - the executor's hold

#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - an executor's queues and the pipeline registry
#include "usearch/utilities/types.hpp"

extern "C" {
void* MTLCopyAllDevices(void);
void* objc_autoreleasePoolPush(void);
void objc_autoreleasePoolPop(void* pool);
}

namespace unum {
namespace usearch {

/** Sends @p selector to @p target with @p arguments, cast to the exact prototype the call has. */
template <typename result_type_ = void*, typename... argument_types_>
inline result_type_ metal_send(void* target, char const* selector, argument_types_... arguments) noexcept {
    using function_t = result_type_ (*)(void*, SEL, argument_types_...);
    return reinterpret_cast<function_t>(objc_msgSend)(target, sel_registerName(selector), arguments...);
}

/** An @c NSString over @p text, autoreleased. */
inline void* metal_string(char const* text) noexcept {
    return metal_send(reinterpret_cast<void*>(objc_getClass("NSString")), "stringWithUTF8String:", text);
}

/** A launch extent, laid out as @c MTLSize so it passes to Metal by value. */
struct metal_size_t {
    std::size_t width = 1, height = 1, depth = 1;
};

/** @c MTLGPUFamilyApple7, @c MTLGPUFamilyApple9 and @c MTLGPUFamilyApple10 - the families the
 *  classes map onto. */
enum metal_family_t : std::size_t {
    metal_family_apple7_k = 1007,
    metal_family_apple9_k = 1009,
    metal_family_apple10_k = 1010,
};

/** @c MTLLanguageVersion3_2, the first with device-scope atomic fences. */
inline constexpr std::size_t metal_language_version_k = (3u << 16) | 2u;

/** The class whose unit runs on @p device: the newest Apple family it supports, or @c unknown_k for
 *  a GPU older than the Apple7 floor or not Apple's own. */
inline gpu_class_t gpu_class_of_metal(void* device) noexcept {
    auto const supports = [&](metal_family_t family) noexcept {
        return metal_send<bool>(device, "supportsFamily:", static_cast<std::size_t>(family));
    };
    if (supports(metal_family_apple10_k)) return gpu_class_t::apple_metal10_k;
    if (supports(metal_family_apple9_k)) return gpu_class_t::apple_metal9_k;
    if (supports(metal_family_apple7_k)) return gpu_class_t::apple_metal7_k;
    return gpu_class_t::unknown_k;
}

/** One compiled entry point and the source library it shares with the other entry points. */
struct metal_pipeline_t {
    char const* source = nullptr;
    char const* name = nullptr;
    void* library = nullptr;
    void* pipeline = nullptr;
};

/** One listed Apple GPU's process-wide state: the device, its default queue - the twin of CUDA's
 *  null stream - its class, and every pipeline built for it, guarded by one lock. */
struct metal_context_t {
    void* device = nullptr;
    void* queue = nullptr;
    gpu_class_t gpu_class = gpu_class_t::unknown_k;
    std::mutex lock;
    growing_buffer<metal_pipeline_t> pipelines {global_allocator};

    /** Kernel @p name's pipeline from the library built out of @p source, built on first use; null
     *  when it fails. */
    void* pipeline(char const* source, char const* name) noexcept {
        std::unique_lock guard {lock};
        void* library = nullptr;
        for (metal_pipeline_t const& entry : pipelines) {
            if (entry.source != source) continue;
            library = entry.library;
            if (std::strcmp(entry.name, name) == 0) return entry.pipeline;
        }
        if (failed(pipelines.reserve(pipelines.size() + 1))) return nullptr;
        void* const pool = objc_autoreleasePoolPush();
        if (!library) {
            void* const options = metal_send(
                metal_send(reinterpret_cast<void*>(objc_getClass("MTLCompileOptions")), "alloc"), "init");
            metal_send<void>(options, "setLanguageVersion:", metal_language_version_k);
            void* error = nullptr;
            library = metal_send(device, "newLibraryWithSource:options:error:", metal_string(source), options, &error);
            metal_send<void>(options, "release");
        }
        void* pipeline = nullptr;
        if (library) {
            void* const function = metal_send(library, "newFunctionWithName:", metal_string(name));
            if (function) {
                void* error = nullptr;
                pipeline = metal_send(device, "newComputePipelineStateWithFunction:error:", function, &error);
                metal_send<void>(function, "release");
            }
        }
        objc_autoreleasePoolPop(pool);
        if (pipeline) pipelines.push_back_reserved({source, name, library, pipeline});
        return pipeline;
    }
};

/** Every Apple GPU of the Apple7 family or newer, in Metal's own order, opened once per process and
 *  kept for it, as Metal's own device objects are; a listed device's Apple ordinal is its position
 *  here. */
struct metal_contexts_t {
    metal_context_t contexts[gpu_devices_capacity_k];
    gpu_device_t devices[gpu_devices_capacity_k];
    std::size_t count = 0;

    metal_contexts_t() noexcept {
        void* const listed = MTLCopyAllDevices();
        if (!listed) return;
        std::size_t const listed_count = metal_send<std::size_t>(listed, "count");
        for (std::size_t position = 0; position != listed_count && count != gpu_devices_capacity_k; ++position) {
            void* const device = metal_send(listed, "objectAtIndex:", position);
            gpu_class_t const gpu_class = gpu_class_of_metal(device);
            void* const queue = gpu_class != gpu_class_t::unknown_k ? metal_send(device, "newCommandQueue") : nullptr;
            if (!queue) continue;
            metal_context_t& context = contexts[count];
            context.device = metal_send(device, "retain");
            context.queue = queue;
            context.gpu_class = gpu_class;
            devices[count] = {gpu_vendor_t::apple_k, static_cast<u32_t>(count), gpu_class};
            ++count;
        }
        metal_send<void>(listed, "release");
    }
};

/** The process-wide listing, opened on first use. */
inline metal_contexts_t& metal_contexts() noexcept {
    static metal_contexts_t contexts;
    return contexts;
}

/** Every listed Apple GPU, as @c gpu_devices appends them after NVIDIA's and AMD's. */
inline std::span<gpu_device_t const> apple_gpu_devices() noexcept {
    metal_contexts_t const& all = metal_contexts();
    return {all.devices, all.count};
}

/** The context of the Apple GPU listed at @p device, or null past the listing. */
inline metal_context_t* metal_context(int device) noexcept {
    metal_contexts_t& all = metal_contexts();
    return device >= 0 && static_cast<std::size_t>(device) < all.count ? &all.contexts[device] : nullptr;
}

/** One Apple GPU's hardware specification - the facts the launches consult. */
struct metal_device_specs_t {

    /** The device's position in the Apple listing, which every context and queue routes through. */
    int ordinal = 0;

    /** The class whose kernels run on it. */
    gpu_class_t gpu_class = gpu_class_t::unknown_k;

    /** The longest buffer one allocation may span. */
    std::size_t max_buffer_bytes = 0;

    /** The memory Metal recommends one process keep resident - the planes' budget. */
    std::size_t working_set_bytes = 0;

    /** Queries Metal for @p ordinal's specification; meaningful only on @c success_k. */
    static expected<metal_device_specs_t> query(int ordinal) noexcept {
        metal_context_t const* const context = metal_context(ordinal);
        if (!context) return {{ordinal}, status_t::device_error_k};
        return {{ordinal, context->gpu_class, metal_send<std::size_t>(context->device, "maxBufferLength"),
                 static_cast<std::size_t>(metal_send<u64_t>(context->device, "recommendedMaxWorkingSetSize"))},
                status_t::success_k};
    }
};

/** One shared-storage @c MTLBuffer: the handle a launch binds and the host address the same bytes
 *  sit at. Move-only, and released when it dies. */
struct metal_buffer_t {
    void* handle = nullptr;
    byte_t* host = nullptr;
    std::size_t bytes = 0;

    metal_buffer_t() noexcept = default;
    metal_buffer_t(metal_buffer_t&& other) noexcept
        : handle(std::exchange(other.handle, nullptr)), host(std::exchange(other.host, nullptr)),
          bytes(std::exchange(other.bytes, 0)) {}
    metal_buffer_t& operator=(metal_buffer_t&& other) noexcept {
        metal_buffer_t dying {std::move(*this)};
        handle = std::exchange(other.handle, nullptr), host = std::exchange(other.host, nullptr);
        bytes = std::exchange(other.bytes, 0);
        return *this;
    }
    metal_buffer_t(metal_buffer_t const&) = delete;
    metal_buffer_t& operator=(metal_buffer_t const&) = delete;
    ~metal_buffer_t() noexcept {
        if (handle) metal_send<void>(handle, "release");
    }

    explicit operator bool() const noexcept { return handle != nullptr; }
};

/** Allocates @p bytes of zeroed shared storage on @p context's device; an empty buffer on
 *  exhaustion. Metal commits the pages lazily, so the bytes cost nothing until touched. */
inline metal_buffer_t metal_allocate(metal_context_t& context, std::size_t bytes) noexcept {
    metal_buffer_t buffer;
    if (!bytes) return buffer;
    constexpr std::size_t shared_storage_k = 0; // `MTLResourceStorageModeShared`
    buffer.handle = metal_send(context.device, "newBufferWithLength:options:", bytes, shared_storage_k);
    if (!buffer.handle) return buffer;
    buffer.host = static_cast<byte_t*>(metal_send(buffer.handle, "contents"));
    buffer.bytes = bytes;
    return buffer;
}

/** The plane backend of the Apple index: shared buffers on one device, each allocation its own
 *  @c MTLBuffer, so a plane grows by allocating a wider one and copying. */
struct metal_shared_allocator_t {
    metal_shared_allocator_t() noexcept = default;
    explicit metal_shared_allocator_t(int device) noexcept : device_(device) {}

    metal_buffer_t allocate(std::size_t bytes) noexcept {
        metal_context_t* const context = metal_context(device_);
        return context ? metal_allocate(*context, bytes) : metal_buffer_t {};
    }
    void deallocate(metal_buffer_t& buffer) noexcept { buffer = {}; }
    int device() const noexcept { return device_; }

  private:
    int device_ = 0;
};

/** One command buffer being encoded on one queue: binds buffers and bytes, dispatches, and commits.
 *  Every call to a device runs through one of these, so encoders live on the caller's stack and any
 *  number of threads encode at once. */
struct metal_launch_t {
    void* commands = nullptr;
    void* encoder = nullptr;

    /** Opens one command buffer on @p queue with a compute encoder for @p pipeline. */
    metal_launch_t(void* queue, void* pipeline) noexcept {
        void* const pool = objc_autoreleasePoolPush();
        commands = metal_send(queue, "commandBuffer");
        if (commands) metal_send<void>(commands, "retain");
        if (commands) encoder = metal_send(commands, "computeCommandEncoder");
        if (encoder) {
            metal_send<void>(encoder, "retain");
            metal_send<void>(encoder, "setComputePipelineState:", pipeline);
        }
        objc_autoreleasePoolPop(pool);
    }
    metal_launch_t(metal_launch_t const&) = delete;
    metal_launch_t& operator=(metal_launch_t const&) = delete;
    ~metal_launch_t() noexcept {
        if (encoder) metal_send<void>(encoder, "endEncoding"), metal_send<void>(encoder, "release");
        if (commands) metal_send<void>(commands, "release");
    }

    explicit operator bool() const noexcept { return encoder != nullptr; }

    /** Binds @p buffer from @p offset at buffer slot @p index. */
    void bind(metal_buffer_t const& buffer, std::size_t index, std::size_t offset = 0) noexcept {
        metal_send<void>(encoder, "setBuffer:offset:atIndex:", buffer.handle, offset, index);
    }

    /** Copies @p bytes of @p arguments into buffer slot @p index. */
    void bind_bytes(void const* arguments, std::size_t bytes, std::size_t index) noexcept {
        metal_send<void>(encoder, "setBytes:length:atIndex:", arguments, bytes, index);
    }

    /** Dispatches @p groups threadgroups of @p threads each, commits, and waits for completion;
     *  answers whether the device finished it. */
    status_t run(std::size_t groups, std::size_t threads) noexcept {
        if (!encoder) return status_t::device_error_k;
        metal_send<void, metal_size_t, metal_size_t>(
            encoder, "dispatchThreadgroups:threadsPerThreadgroup:", metal_size_t {groups}, metal_size_t {threads});
        metal_send<void>(encoder, "endEncoding");
        metal_send<void>(encoder, "release");
        encoder = nullptr;
        metal_send<void>(commands, "commit");
        metal_send<void>(commands, "waitUntilCompleted");
        constexpr std::size_t completed_k = 4; // `MTLCommandBufferStatusCompleted`
        return metal_send<std::size_t>(commands, "status") == completed_k ? status_t::success_k
                                                                          : status_t::device_error_k;
    }
};

/** Memory that one call at a time rewinds and refills for one Apple device: the Apple arm of
 *  @ref gpu_arena_t. Memory is unified, so host pages are the device's and an upload is a copy. */
struct metal_device_arena_t {
    explicit metal_device_arena_t(int device) noexcept : device_(device) {}

    /** Carves @p count values, or null when the arena cannot grow. */
    template <typename value_type_>
    value_type_* allocate(std::size_t count) noexcept {
        return reinterpret_cast<value_type_*>(memory_.allocate(count * sizeof(value_type_), default_alignment_k));
    }

    /** Whether @p address lies in memory handed out since the last @ref clear. */
    bool contains(void const* address) const noexcept { return memory_.contains(address); }

    /** Rewinds the arena, so the previous call's outputs die. */
    void clear() noexcept { memory_.clear(); }

    /** Copies @p bytes from @p source to @p destination in the arena. */
    status_t upload(void* destination, void const* source, std::size_t bytes) noexcept {
        std::memcpy(destination, source, bytes);
        return status_t::success_k;
    }

    int device() const noexcept { return device_; }

  private:
    growing_arena<> memory_ {global_allocator};
    int device_ = 0;
};

/** The queues a bulk verb submits to - one per device. Every call blocks until its command buffers
 *  complete, so one host thread drives every device, and the hold serializes callers as the CUDA
 *  executor's does. */
struct metal_executor_t {

    /** Where a call names this executor, the C layer lands outputs in this arena. */
    using device_arena_t = metal_device_arena_t;
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::apple_k;

    /** One call's hold on the executor, released when it dies. */
    using claim_t = std::unique_lock<fu::spin_mutex_t>;

    /** One device's queue. */
    struct stream_t {

        /** The device this queue submits to. */
        int device = 0;

        /** The @c id<MTLCommandQueue>, owned, or null for the device's default queue, which its
         *  context keeps. */
        void* handle = nullptr;

        stream_t(int device, void* handle) noexcept : device(device), handle(handle) {}
        stream_t(stream_t&& other) noexcept : device(other.device), handle(std::exchange(other.handle, nullptr)) {}
        stream_t(stream_t const&) = delete;
        stream_t& operator=(stream_t const&) = delete;
        ~stream_t() noexcept {
            if (handle) metal_send<void>(handle, "release");
        }
    };

    metal_executor_t() = default;
    metal_executor_t(metal_executor_t&& other) noexcept { swap(other); }
    metal_executor_t& operator=(metal_executor_t&& other) noexcept { return swap(other), *this; }
    metal_executor_t(metal_executor_t const&) = delete;
    metal_executor_t& operator=(metal_executor_t const&) = delete;

    void swap(metal_executor_t& other) noexcept { std::swap(streams_, other.streams_); }

    /** Claims the executor for one call; a second concurrent caller gets @c contended_k. */
    expected<claim_t> claim() noexcept {
        if (!held_.lock_if_free()) return {{}, status_t::contended_k};
        return {claim_t {held_, std::adopt_lock}, status_t::success_k};
    }

    /** Opens one queue on every named device. */
    static expected<metal_executor_t> spawn(std::span<int const> devices) noexcept {
        if (devices.empty()) return {{}, status_t::config_mismatch_k};
        metal_executor_t executor;
        if (failed(executor.streams_.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int const device : devices) {
            metal_context_t* const context = metal_context(device);
            if (!context) return {{}, status_t::device_error_k};
            void* const queue = metal_send(context->device, "newCommandQueue");
            if (!queue) return {{}, status_t::device_error_k};
            executor.streams_.push_back_reserved(stream_t {device, queue});
        }
        return {std::move(executor), status_t::success_k};
    }

    /** The twin of @ref spawn for a call that names no executor: every device's default queue, as
     *  the CUDA executor takes the null stream. */
    static expected<metal_executor_t> on_default_streams(std::span<int const> devices) noexcept {
        if (devices.empty()) return {{}, status_t::config_mismatch_k};
        metal_executor_t executor;
        if (failed(executor.streams_.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int const device : devices) {
            if (!metal_context(device)) return {{}, status_t::device_error_k};
            executor.streams_.push_back_reserved(stream_t {device, nullptr});
        }
        return {std::move(executor), status_t::success_k};
    }

    /** Every call already waited for its command buffers, so there is nothing left to drain. */
    status_t drain() noexcept { return status_t::success_k; }

    stream_t& stream(std::size_t index) noexcept { return streams_[index]; }
    int device(std::size_t index) const noexcept { return streams_[index].device; }
    std::size_t size() const noexcept { return streams_.size(); }

  private:
    growing_buffer<stream_t> streams_ {global_allocator};
    fu::spin_mutex_t held_;
};

} // namespace usearch
} // namespace unum
