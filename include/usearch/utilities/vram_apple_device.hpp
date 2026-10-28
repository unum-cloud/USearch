/**
 *  @file include/usearch/utilities/vram_apple_device.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Host-side vocabulary for driving one Apple GPU - specs, streams, shared buffers,
 *      pipelines, launch counters and the traversal context.
 *
 *  Metal's API is Objective-C, reached here through @c objc_msgSend cast to each call's exact
 *  prototype, so a plain C++ unit drives it, with no SDK header: the few runtime prototypes are
 *  spelled here. Nothing links Metal, Foundation or the Objective-C runtime:
 *  @ref unum::usearch::utilities::metal_api_t::load opens them and resolves the few C entry points
 *  USearch calls into one table, which stays empty where any is absent, so Apple then lists no
 *  devices and the binary still loads. Kernels travel as source and compile on the device at first
 *  use, so no Metal toolchain is needed to build. Memory is unified and every buffer is shared, so
 *  the host reads and writes a matrix in place and a launch binds the same buffer by handle.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::strcmp`, `std::strncpy`, `std::memcpy`

#include <mutex>   // `std::mutex` - the pipeline registry, `std::unique_lock`, `std::adopt_lock` - the executor's hold
#include <span>    // `std::span` - the device list an executor opens over
#include <utility> // `std::exchange`, `std::move`

#include <forkunion/types.hpp> // `fu::spin_mutex_t` - the executor's hold

#include "usearch/utilities/blocking_files.hpp" // `shared_library_t` - Metal opened at first use
#include "usearch/utilities/ram_buffers.hpp"    // `growing_buffer` - an executor's streams and the pipeline registry
#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_apple_shared.h" // `metal_counter_t` - the words a context reads its telemetry from

namespace unum::usearch::utilities {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

#if USEARCH_ARCH_METAL_

/** The vocabulary every Apple context and its kernels share, which each MSL source starts with,
 *  since MSL cannot include. */
inline constexpr char metal_context_source_k[] = {
#embed "vram_apple_shared.h" suffix(, 0)
};
#else
// A unit without Metal compiles no pipeline, and NVCC's front end knows no `#embed`.
inline constexpr char metal_context_source_k[] = {0};
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/** The Objective-C runtime's selector, @c SEL, opaque to everyone but the runtime. */
using metal_selector_t = struct metal_selector_opaque_t*;

/**
 *  @brief The Metal, Objective-C runtime, I/O Kit and Core Foundation entry points USearch calls,
 *      resolved on first use from the system's own frameworks and @c libobjc.
 *
 *  Every entry is null when any library is absent or lacks any of them, which leaves Apple with no
 *  devices. Foundation is opened for @c NSString alone, which no entry names, and I/O Kit for the
 *  GPU core count, which Metal does not report. Each entry spells the C prototype of its symbol,
 *  with Core Foundation's references as plain pointers and I/O Kit's objects as their 32-bit Mach
 *  ports; @c objc_msgSend takes none of its own, so every send casts it to the call's exact one.
 */
struct metal_api_t {
    void* (*copy_all_devices)() = nullptr;
    void (*send)() = nullptr;
    metal_selector_t (*register_selector)(char const*) = nullptr;
    void* (*get_class)(char const*) = nullptr;
    void* (*registry_entry_id_matching)(u64_t) = nullptr;
    u32_t (*get_matching_service)(u32_t, void*) = nullptr;
    void* (*search_registry_property)(u32_t, char const*, void const*, void const*, u32_t) = nullptr;
    int (*release_registry_object)(u32_t) = nullptr;
    void* (*create_string)(void const*, char const*, u32_t) = nullptr;
    unsigned char (*get_number_value)(void const*, long, void*) = nullptr;
    void (*release_reference)(void const*) = nullptr;
    void* (*autorelease_pool_push)() = nullptr;
    void (*autorelease_pool_pop)(void*) = nullptr;

    /** Whether every library loaded with every entry resolved. */
    explicit operator bool() const noexcept { return autorelease_pool_pop != nullptr; }

    /** Opens the libraries and resolves every entry, or answers the empty table. The libraries stay
     *  open for the life of the process, as the table points into them. */
    static metal_api_t load() noexcept {
        shared_library_t const metal = shared_library_t::open("/System/Library/Frameworks/Metal.framework/Metal");
        shared_library_t const foundation = shared_library_t::open(
            "/System/Library/Frameworks/Foundation.framework/Foundation");
        shared_library_t const runtime = shared_library_t::open("/usr/lib/libobjc.A.dylib");
        shared_library_t const registry = shared_library_t::open("/System/Library/Frameworks/IOKit.framework/IOKit");
        shared_library_t const core = shared_library_t::open(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation");
        metal_api_t api;
        bool const resolved = //
            foundation && metal.resolve("MTLCopyAllDevices", api.copy_all_devices) &&
            runtime.resolve("objc_msgSend", api.send) && runtime.resolve("sel_registerName", api.register_selector) &&
            runtime.resolve("objc_getClass", api.get_class) &&
            registry.resolve("IORegistryEntryIDMatching", api.registry_entry_id_matching) &&
            registry.resolve("IOServiceGetMatchingService", api.get_matching_service) &&
            registry.resolve("IORegistryEntrySearchCFProperty", api.search_registry_property) &&
            registry.resolve("IOObjectRelease", api.release_registry_object) &&
            core.resolve("CFStringCreateWithCString", api.create_string) &&
            core.resolve("CFNumberGetValue", api.get_number_value) &&
            core.resolve("CFRelease", api.release_reference) &&
            runtime.resolve("objc_autoreleasePoolPush", api.autorelease_pool_push) &&
            runtime.resolve("objc_autoreleasePoolPop", api.autorelease_pool_pop);
        return resolved ? api : metal_api_t {};
    }
};

/** Sends @p selector to @p target with @p arguments through @p api, cast to the exact prototype the
 *  call has. */
template <typename result_type_ = void*, typename... argument_types_>
inline result_type_ metal_send(metal_api_t const& api, void* target, char const* selector,
                               argument_types_... arguments) noexcept {
    using function_t = result_type_ (*)(void*, metal_selector_t, argument_types_...);
    return reinterpret_cast<function_t>(api.send)(target, api.register_selector(selector), arguments...);
}

/** An @c NSString over @p text, autoreleased. */
inline void* metal_string(metal_api_t const& api, char const* text) noexcept {
    return metal_send(api, reinterpret_cast<void*>(api.get_class("NSString")), "stringWithUTF8String:", text);
}

/** A launch extent, laid out as @c MTLSize so it passes to Metal by value. */
struct metal_size_t {
    std::size_t width = 1, height = 1, depth = 1;
};

/** @c MTLGPUFamilyApple7 through @c MTLGPUFamilyApple10 - the families a device record names. */
enum metal_family_t : std::size_t {
    metal_family_apple7_k = 1007,
    metal_family_apple8_k = 1008,
    metal_family_apple9_k = 1009,
    metal_family_apple10_k = 1010,
};

/** @c MTLLanguageVersion3_2, the first with device-scope atomic fences. */
inline constexpr std::size_t metal_language_version_k = (3u << 16) | 2u;

/** The newest Apple GPU family @p device supports, as its number, such as 9 for
 *  @c MTLGPUFamilyApple9; zero for a GPU older than Apple7 or not Apple's own. */
inline u32_t apple_family_of(metal_api_t const& api, void* device) noexcept {
    for (metal_family_t const family :
         {metal_family_apple10_k, metal_family_apple9_k, metal_family_apple8_k, metal_family_apple7_k})
        if (metal_send<bool>(api, device, "supportsFamily:", static_cast<std::size_t>(family)))
            return static_cast<u32_t>(family - metal_family_apple7_k + 7);
    return 0;
}

/** The GPU cores of @p device, which Metal does not report: the @c gpu-core-count its accelerator,
 *  the I/O Registry entry Metal names by @c registryID, or that entry's parents and children list;
 *  zero when none does. */
inline u32_t apple_core_count_of(metal_api_t const& api, void* device) noexcept {
    constexpr u32_t main_port_k = 0;          // `kIOMainPortDefault`
    constexpr u32_t utf8_k = 0x08000100;      // `kCFStringEncodingUTF8`
    constexpr u32_t search_options_k = 1 | 2; // `kIORegistryIterateRecursively | ...Parents`
    constexpr long signed_64_bits_k = 4;      // `kCFNumberSInt64Type`
    // The lookup consumes the matching dictionary, so only the service is released here.
    u32_t const service = api.get_matching_service(
        main_port_k, api.registry_entry_id_matching(metal_send<u64_t>(api, device, "registryID")));
    if (!service) return 0;
    void* const key = api.create_string(nullptr, "gpu-core-count", utf8_k);
    void* const listed = key ? api.search_registry_property(service, "IOService", key, nullptr, search_options_k)
                             : nullptr;
    i64_t cores = 0;
    if (listed && !api.get_number_value(listed, signed_64_bits_k, &cores)) cores = 0;
    if (listed) api.release_reference(listed);
    if (key) api.release_reference(key);
    api.release_registry_object(service);
    return cores > 0 ? static_cast<u32_t>(cores) : 0;
}

/** Bytes of the longest kernel name a pipeline is built for, its terminator included. */
inline constexpr std::size_t metal_pipeline_name_capacity_k = 64;

/** One built entry point as a launch reads it: the pipeline state, and the threads one threadgroup
 *  of it holds at most, its @c maxTotalThreadsPerThreadgroup, which its register use bounds. */
struct metal_pipeline_state_t {
    void* pipeline = nullptr;
    u32_t max_threads = 0;
};

/** One compiled entry point and the source library it shares with the other entry points; the name
 *  is copied, as callers spell it into their own buffers. */
struct metal_pipeline_t {
    char const* source = nullptr;
    char name[metal_pipeline_name_capacity_k] = {};
    void* library = nullptr;
    metal_pipeline_state_t state;
};

/** One listed Apple GPU's process-wide state: the table its calls go through, the device, its
 *  default queue - the twin of CUDA's null stream - its family and cores, and every pipeline built
 *  for it, guarded by one lock. */
struct metal_context_t {
    metal_api_t const* api = nullptr;
    void* device = nullptr;
    void* queue = nullptr;

    /** The newest Apple GPU family the device supports, as @ref apple_family_of numbers it. */
    u32_t family = 0;

    /** The GPU cores the I/O Registry lists for the device, zero when it lists none. */
    u32_t multiprocessor_count = 0;

    /** Lanes per simdgroup every pipeline built here reports, its @c threadExecutionWidth; zero
     *  until the first builds. */
    u32_t wave_lanes = 0;

    std::mutex lock;
    growing_buffer<metal_pipeline_t> pipelines {global_allocator};

    /**
     *  Kernel @p name's pipeline from the library built out of @p source, built on first use; a
     *  caller meeting another index's build sleeps until it ends, as a module load waits in the
     *  CUDA and HIP drivers.
     *
     *  @par Statuses:
     *  - @c unsupported_k The pipeline's simdgroups are not the @c metal_simdgroup_width_k lanes
     *    every MSL source is written against.
     */
    expected<metal_pipeline_state_t> pipeline(char const* source, char const* name) noexcept {
        std::lock_guard guard {lock};
        void* library = nullptr;
        for (metal_pipeline_t const& entry : pipelines) {
            if (entry.source != source) continue;
            library = entry.library;
            if (std::strcmp(entry.name, name) == 0) return {entry.state, status_t::success_k};
        }
        if (failed(pipelines.reserve(pipelines.size() + 1))) return {{}, status_t::bad_alloc_k};
        metal_api_t const& api = *this->api;
        void* const pool = api.autorelease_pool_push();
        if (!library) {
            void* const options = metal_send(
                api, metal_send(api, reinterpret_cast<void*>(api.get_class("MTLCompileOptions")), "alloc"), "init");
            metal_send<void>(api, options, "setLanguageVersion:", metal_language_version_k);
            void* error = nullptr;
            library = metal_send(api, device, "newLibraryWithSource:options:error:", metal_string(api, source), options,
                                 &error);
            metal_send<void>(api, options, "release");
        }
        void* pipeline = nullptr;
        if (library) {
            void* const function = metal_send(api, library, "newFunctionWithName:", metal_string(api, name));
            if (function) {
                void* error = nullptr;
                pipeline = metal_send(api, device, "newComputePipelineStateWithFunction:error:", function, &error);
                metal_send<void>(api, function, "release");
            }
        }
        api.autorelease_pool_pop(pool);
        if (!pipeline) return {{}, status_t::device_error_k};
        u32_t const width = static_cast<u32_t>(metal_send<std::size_t>(api, pipeline, "threadExecutionWidth"));
        if (width != metal_simdgroup_width_k) {
            metal_send<void>(api, pipeline, "release");
            return {{}, status_t::unsupported_k};
        }
        wave_lanes = width;
        u32_t const max_threads = static_cast<u32_t>(
            metal_send<std::size_t>(api, pipeline, "maxTotalThreadsPerThreadgroup"));
        metal_pipeline_t entry {source, {}, library, {pipeline, max_threads}};
        std::strncpy(entry.name, name, metal_pipeline_name_capacity_k - 1);
        pipelines.push_back_reserved(entry);
        return {entry.state, status_t::success_k};
    }
};

/** Every Apple GPU Metal lists, in Metal's own order, opened once per process by its caller and
 *  kept for it, as Metal's own device objects are; a device's Apple ordinal is its position
 *  here. */
struct metal_contexts_t {
    metal_context_t contexts[gpu_devices_capacity_k];
    gpu_device_t devices[gpu_devices_capacity_k];
    std::size_t count = 0;

    /** Opens every device @p api lists, none when the table is empty. */
    explicit metal_contexts_t(metal_api_t const& api) noexcept {
        void* const listed = api ? api.copy_all_devices() : nullptr;
        if (!listed) return;
        void* const pool = api.autorelease_pool_push();
        std::size_t const listed_count = metal_send<std::size_t>(api, listed, "count");
        for (std::size_t position = 0; position != listed_count && count != gpu_devices_capacity_k; ++position) {
            void* const device = metal_send(api, listed, "objectAtIndex:", position);
            void* const queue = metal_send(api, device, "newCommandQueue");
            if (!queue) continue;
            metal_context_t& context = contexts[count];
            context.api = &api;
            context.device = metal_send(api, device, "retain");
            context.queue = queue;
            context.family = apple_family_of(api, device);
            context.multiprocessor_count = apple_core_count_of(api, device);
            gpu_device_t& listing = devices[count];
            listing = {gpu_vendor_t::apple_k,
                       static_cast<u32_t>(count),
                       context.family,
                       0,
                       0,
                       context.multiprocessor_count,
                       metal_send<u64_t>(api, device, "recommendedMaxWorkingSetSize")};
            char const* const name = metal_send<char const*>(api, metal_send(api, device, "name"), "UTF8String");
            if (name) std::strncpy(listing.name, name, sizeof(listing.name) - 1);
            ++count;
        }
        api.autorelease_pool_pop(pool);
        metal_send<void>(api, listed, "release");
    }
    metal_contexts_t(metal_contexts_t const&) = delete;
    metal_contexts_t& operator=(metal_contexts_t const&) = delete;

    /** The context of the Apple GPU listed at @p device, or null past the listing. */
    metal_context_t* context(int device) noexcept {
        return device >= 0 && static_cast<std::size_t>(device) < count ? &contexts[device] : nullptr;
    }
};

/** Copies every Apple GPU @p contexts opened into @p into, answering how many it wrote. */
inline std::size_t list_apple_gpu_devices(metal_contexts_t const& contexts, std::span<gpu_device_t> into) noexcept {
    std::size_t const count = min_of(contexts.count, into.size());
    for (std::size_t position = 0; position != count; ++position) into[position] = contexts.devices[position];
    return count;
}

/** One Apple GPU's hardware specification - the facts the launches consult. */
struct metal_device_specs_t {

    /** The device's position in the Apple listing, which every context and queue routes through. */
    int ordinal = 0;

    /** GPU-core count, the accelerator's @c gpu-core-count in the I/O Registry - the launch-grid
     *  currency. */
    u32_t multiprocessor_count = 0;

    /** The newest Apple GPU family it supports, as @ref apple_family_of numbers it. */
    u32_t family = 0;

    /** Lanes per simdgroup, the @c threadExecutionWidth every pipeline built for it reports. */
    u32_t wave_lanes = 0;

    /** Threads one threadgroup may hold, @c maxThreadsPerThreadgroup, before a pipeline's register
     *  use lowers it. */
    u32_t max_threads_per_block = 0;

    /** Widest threadgroup memory one threadgroup can be granted, @c maxThreadgroupMemoryLength. */
    std::size_t shared_memory_per_block_bytes = 0;

    /** The longest buffer one allocation may span, @c maxBufferLength. */
    std::size_t max_buffer_bytes = 0;

    /** The memory Metal recommends one process keep resident, @c recommendedMaxWorkingSetSize - the
     *  storage budget. */
    std::size_t working_set_bytes = 0;

    /**
     *  Queries Metal and the I/O Registry for the specification of @p context, the device at
     *  @p ordinal, once a pipeline built for it reported its simdgroup width; meaningful only on
     *  @c success_k.
     *
     *  @par Statuses:
     *  - @c device_error_k No pipeline reported the device's simdgroup width yet.
     *  - @c unsupported_k The device's registry lists no core count.
     */
    static expected<metal_device_specs_t> query(metal_context_t const& context, int ordinal) noexcept {
        if (!context.wave_lanes) return {{ordinal}, status_t::device_error_k};
        if (!context.multiprocessor_count) return {{ordinal}, status_t::unsupported_k};
        metal_api_t const& api = *context.api;
        void* const device = context.device;
        metal_device_specs_t specs {ordinal, context.multiprocessor_count, context.family, context.wave_lanes};
        specs.max_threads_per_block = static_cast<u32_t>(
            metal_send<metal_size_t>(api, device, "maxThreadsPerThreadgroup").width);
        specs.shared_memory_per_block_bytes = metal_send<std::size_t>(api, device, "maxThreadgroupMemoryLength");
        specs.max_buffer_bytes = metal_send<std::size_t>(api, device, "maxBufferLength");
        specs.working_set_bytes = static_cast<std::size_t>(
            metal_send<u64_t>(api, device, "recommendedMaxWorkingSetSize"));
        return {specs, status_t::success_k};
    }
};

/** One shared-storage @c MTLBuffer: the table it is released through, the handle a launch binds
 *  and the host address the same bytes sit at. Move-only, and released when it dies. */
struct metal_buffer_t {
    metal_api_t const* api = nullptr;
    void* handle = nullptr;
    byte_t* host = nullptr;
    std::size_t bytes = 0;

    metal_buffer_t() noexcept = default;
    metal_buffer_t(metal_buffer_t&& other) noexcept
        : api(other.api), handle(std::exchange(other.handle, nullptr)), host(std::exchange(other.host, nullptr)),
          bytes(std::exchange(other.bytes, 0)) {}
    metal_buffer_t& operator=(metal_buffer_t&& other) noexcept {
        metal_buffer_t dying {std::move(*this)};
        api = other.api, handle = std::exchange(other.handle, nullptr), host = std::exchange(other.host, nullptr);
        bytes = std::exchange(other.bytes, 0);
        return *this;
    }
    metal_buffer_t(metal_buffer_t const&) = delete;
    metal_buffer_t& operator=(metal_buffer_t const&) = delete;
    ~metal_buffer_t() noexcept {
        if (handle) metal_send<void>(*api, handle, "release");
    }

    explicit operator bool() const noexcept { return handle != nullptr; }
};

/** Allocates @p bytes of zeroed shared storage on @p context's device; an empty buffer on
 *  exhaustion. Metal commits the pages lazily, so the bytes cost nothing until touched. */
inline metal_buffer_t metal_allocate(metal_context_t& context, std::size_t bytes) noexcept {
    metal_buffer_t buffer;
    if (!bytes) return buffer;
    constexpr std::size_t shared_storage_k = 0; // `MTLResourceStorageModeShared`
    metal_api_t const& api = *context.api;
    buffer.api = &api;
    buffer.handle = metal_send(api, context.device, "newBufferWithLength:options:", bytes, shared_storage_k);
    if (!buffer.handle) return buffer;
    buffer.host = static_cast<byte_t*>(metal_send(api, buffer.handle, "contents"));
    buffer.bytes = bytes;
    return buffer;
}

/** Owned @c id<MTLCommandQueue> on one Apple GPU, the twin of @c cuda_stream_t; null rides the
 *  device's default queue, which its @ref metal_context_t keeps, so the stream names its device
 *  and context. Move-only, and released when it dies. */
struct metal_stream_t {

    /** The Apple ordinal this stream submits to. */
    int device = 0;

    /** A stream on no device, which only fills the default of a context built later. */
    metal_stream_t() noexcept = default;

    /** Takes @p handle, owned and opened on @p device, whose context is @p context, or rides that
     *  device's default queue when null. */
    metal_stream_t(metal_context_t& context, int device, void* handle = nullptr) noexcept
        : device(device), context_(&context), handle_(handle) {}
    metal_stream_t(metal_stream_t&& other) noexcept
        : device(other.device), context_(other.context_), handle_(std::exchange(other.handle_, nullptr)) {}
    metal_stream_t(metal_stream_t const&) = delete;
    metal_stream_t& operator=(metal_stream_t const&) = delete;
    ~metal_stream_t() noexcept {
        if (handle_) metal_send<void>(*context_->api, handle_, "release");
    }

    /** Another owner of the same queue, retained, so a context holds an executor's stream without
     *  borrowing it. */
    metal_stream_t share() const noexcept {
        if (!context_) return {};
        return metal_stream_t {*context_, device, handle_ ? metal_send(*context_->api, handle_, "retain") : nullptr};
    }

    /** The owned queue, or null for the device's default one. */
    void* handle() const noexcept { return handle_; }

    /** The context of the stream's device, or null for a stream on no device. */
    metal_context_t* context() const noexcept { return context_; }

  private:
    metal_context_t* context_ = nullptr;
    void* handle_ = nullptr;
};

/** The storage backend of the Apple index: shared buffers on one device, each allocation its own
 *  @c MTLBuffer, so a matrix grows by allocating a wider one and copying. */
struct metal_shared_allocator_t {
    metal_shared_allocator_t() noexcept = default;

    /** Allocates on the device and context of @p stream. */
    explicit metal_shared_allocator_t(metal_stream_t const& stream) noexcept
        : context_(stream.context()), device_(stream.device) {}

    metal_buffer_t allocate(std::size_t bytes) noexcept {
        return context_ ? metal_allocate(*context_, bytes) : metal_buffer_t {};
    }
    void deallocate(metal_buffer_t& buffer) noexcept { buffer = {}; }
    int device() const noexcept { return device_; }

    /** The context of the allocator's device, or null for a default-built one. */
    metal_context_t* context() const noexcept { return context_; }

  private:
    metal_context_t* context_ = nullptr;
    int device_ = 0;
};

/** One command buffer being encoded on one queue: binds buffers and bytes, dispatches, and commits
 *  without waiting. Every call to a device runs through one of these, so encoders live on the
 *  caller's stack and any number of threads encode at once; a committed one stays retained as
 *  Apple's event, the twin of @c cuda_event_t, since a queue runs its command buffers in commit
 *  order, so its landing means every earlier one's. */
struct metal_launch_t {
    metal_api_t const* api = nullptr;
    void* commands = nullptr;
    void* encoder = nullptr;

    /** No command buffer - one with nothing left to run. */
    metal_launch_t() noexcept = default;

    /** Opens one command buffer on @p queue with a compute encoder for @p pipeline, through
     *  @p api. */
    metal_launch_t(metal_api_t const& api, void* queue, void* pipeline) noexcept : api(&api) {
        void* const pool = api.autorelease_pool_push();
        commands = metal_send(api, queue, "commandBuffer");
        if (commands) metal_send<void>(api, commands, "retain");
        if (commands) encoder = metal_send(api, commands, "computeCommandEncoder");
        if (encoder) {
            metal_send<void>(api, encoder, "retain");
            metal_send<void>(api, encoder, "setComputePipelineState:", pipeline);
        }
        // A command buffer without its encoder is never committed, so none is kept to await.
        else if (commands) metal_send<void>(api, std::exchange(commands, nullptr), "release");
        api.autorelease_pool_pop(pool);
    }
    metal_launch_t(metal_launch_t&& other) noexcept
        : api(other.api), commands(std::exchange(other.commands, nullptr)),
          encoder(std::exchange(other.encoder, nullptr)) {}
    metal_launch_t& operator=(metal_launch_t&& other) noexcept {
        metal_launch_t dying {std::move(*this)};
        api = other.api;
        commands = std::exchange(other.commands, nullptr), encoder = std::exchange(other.encoder, nullptr);
        return *this;
    }
    metal_launch_t(metal_launch_t const&) = delete;
    metal_launch_t& operator=(metal_launch_t const&) = delete;
    ~metal_launch_t() noexcept {
        if (encoder) metal_send<void>(*api, encoder, "endEncoding"), metal_send<void>(*api, encoder, "release");
        if (commands) metal_send<void>(*api, commands, "release");
    }

    /** Whether the encoder is open, so the launch can still bind and dispatch. */
    explicit operator bool() const noexcept { return encoder != nullptr; }

    /** Binds @p buffer from @p offset at buffer slot @p index. */
    void bind(metal_buffer_t const& buffer, std::size_t index, std::size_t offset = 0) noexcept {
        metal_send<void>(*api, encoder, "setBuffer:offset:atIndex:", buffer.handle, offset, index);
    }

    /** Copies @p bytes of @p arguments into buffer slot @p index. */
    void bind_bytes(void const* arguments, std::size_t bytes, std::size_t index) noexcept {
        metal_send<void>(*api, encoder, "setBytes:length:atIndex:", arguments, bytes, index);
    }

    /**
     *  Dispatches @p groups threadgroups of @p threads each and commits, without waiting.
     *
     *  @par Statuses:
     *  - @c device_error_k The encoder is not open.
     */
    status_t submit(std::size_t groups, std::size_t threads) noexcept {
        if (!encoder) return status_t::device_error_k;
        metal_send<void, metal_size_t, metal_size_t>(*api, encoder, "dispatchThreadgroups:threadsPerThreadgroup:",
                                                     metal_size_t {groups}, metal_size_t {threads});
        metal_send<void>(*api, encoder, "endEncoding");
        metal_send<void>(*api, encoder, "release");
        encoder = nullptr;
        metal_send<void>(*api, commands, "commit");
        return status_t::success_k;
    }

    /** Polls the committed command buffer without blocking; nothing committed has nothing to
     *  run. */
    status_t landed() const noexcept {
        if (!commands || encoder) return status_t::success_k;
        constexpr std::size_t completed_k = 4, error_k = 5; // `MTLCommandBufferStatusCompleted`, `...Error`
        std::size_t const state = metal_send<std::size_t>(*api, commands, "status");
        return state == completed_k ? status_t::success_k
               : state == error_k   ? status_t::device_error_k
                                    : status_t::deferred_k;
    }

    /** Waits for the committed command buffer and answers whether the device finished it. */
    status_t wait() noexcept {
        if (commands && !encoder) metal_send<void>(*api, commands, "waitUntilCompleted");
        return landed();
    }
};

/** Memory that one call at a time rewinds and refills for one Apple device: the Apple alternative
 *  of @ref gpu_arena_t. Memory is unified, so host pages are the device's and an upload is a copy;
 *  it still serves only the executor that made it, as every vendor's arena does. */
struct metal_device_arena_t {

    /** Takes the device of the executor's @p stream; a template only because that @c stream_t is
     *  declared after it. */
    template <typename stream_type_>
    explicit metal_device_arena_t(stream_type_ const& stream) noexcept : device_(stream.handle.device) {}

    /** Carves @p count values, or null when the arena cannot grow. */
    template <typename value_type_>
    value_type_* allocate(std::size_t count) noexcept {
        return reinterpret_cast<value_type_*>(memory_.allocate(count * sizeof(value_type_), default_alignment_k));
    }

    /** Whether @p address lies in memory handed out since the last @ref clear. */
    bool contains(void const* address) const noexcept { return memory_.contains(address); }

    /** Rewinds the arena, so the previous call's outputs die. */
    void clear() noexcept { memory_.clear(); }

    /** Copies @p bytes from @p source to @p destination in the arena; unified memory needs no
     *  stream, so the copy is done when it returns. */
    status_t upload(void* destination, void const* source, std::size_t bytes) noexcept {
        std::memcpy(destination, source, bytes);
        return status_t::success_k;
    }

    /** Every upload already landed, so there is nothing to wait for. */
    status_t wait() noexcept { return status_t::success_k; }

    int device() const noexcept { return device_; }

  private:
    growing_arena<> memory_ {global_allocator};
    int device_ = 0;
};

/** The streams a bulk verb submits to and the scratch they walk with - one of each per device,
 *  which every shard there shares in order. Work is committed, never waited on, so one host thread
 *  drives every device, and the hold serializes callers as the CUDA executor's does. */
struct metal_executor_t {

    /** Where a call names this executor, the C layer lands outputs in this arena. */
    using device_arena_t = metal_device_arena_t;
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::apple_k;

    /** One device's run of scratch: one shared buffer, sized by @ref reserve. */
    using arena_t = metal_buffer_t;

    /** One call's hold on the executor, released when it dies. */
    using claim_t = std::unique_lock<fu::spin_mutex_t>;

    /** One shard's stream and run. */
    struct stream_t {

        /** The ordering identity every launch on this device rides, naming the device. */
        metal_stream_t handle;

        /** The scratch a walk carves its state from, sized by @ref reserve. */
        arena_t arena;

        /** Takes @p handle, whose run waits for @ref reserve. */
        explicit stream_t(metal_stream_t&& handle) noexcept : handle(std::move(handle)) {}
    };

    metal_executor_t() = default;
    metal_executor_t(metal_executor_t&& other) noexcept { swap(other); }
    metal_executor_t& operator=(metal_executor_t&& other) noexcept { return swap(other), *this; }
    metal_executor_t(metal_executor_t const&) = delete;
    metal_executor_t& operator=(metal_executor_t const&) = delete;

    void swap(metal_executor_t& other) noexcept {
        std::swap(streams_, other.streams_);
        std::swap(scratch_limit_, other.scratch_limit_);
    }

    /**
     *  Claims the executor for one call.
     *
     *  @par Statuses:
     *  - @c contended_k Another call holds the executor.
     */
    expected<claim_t> claim() noexcept {
        if (!held_.lock_if_free()) return {{}, status_t::contended_k};
        return {claim_t {held_, std::adopt_lock}, status_t::success_k};
    }

    /**
     *  Takes over streams the caller already opened, owned and released here from now on.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p streams are empty.
     */
    static expected<metal_executor_t> acquire(growing_buffer<stream_t>&& streams) noexcept {
        if (streams.empty()) return {{}, status_t::config_mismatch_k};
        metal_executor_t executor;
        executor.streams_ = std::move(streams);
        return {std::move(executor), status_t::success_k};
    }

    /**
     *  Opens one stream per entry of @p devices among @p contexts, whose scratch never grows past
     *  @p scratch_limit bytes, zero for no ceiling.
     *
     *  @par Statuses:
     *  - @c device_error_k A device is not listed.
     *  - @c config_mismatch_k The @p devices are empty.
     */
    static expected<metal_executor_t> spawn(metal_contexts_t& contexts, std::span<int const> devices,
                                            std::size_t scratch_limit = 0) noexcept {
        growing_buffer<stream_t> streams {global_allocator};
        if (failed(streams.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int const device : devices) {
            metal_context_t* const context = contexts.context(device);
            if (!context) return {{}, status_t::device_error_k};
            void* const queue = metal_send(*context->api, context->device, "newCommandQueue");
            if (!queue) return {{}, status_t::device_error_k};
            streams.push_back_reserved(stream_t {metal_stream_t {*context, device, queue}});
        }
        auto [executor, status] = acquire(std::move(streams));
        executor.scratch_limit_ = scratch_limit;
        return {std::move(executor), status};
    }

    /**
     *  Claims @p bytes_per_stream of scratch on every stream, ahead of the fan-out that maps state
     *  onto it; the old run lives on in whichever command buffer still binds it.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor has no streams.
     *  - @c bad_alloc_k The @p bytes_per_stream pass the scratch ceiling.
     *  - @c device_error_k A stream's device is not listed.
     */
    status_t reserve(std::size_t bytes_per_stream) noexcept {
        if (streams_.empty()) return status_t::uninitialized_k;
        if (scratch_limit_ && bytes_per_stream > scratch_limit_) return status_t::bad_alloc_k;
        for (stream_t& opened : streams_) {
            metal_context_t* const context = opened.handle.context();
            if (!context) return status_t::device_error_k;
            opened.arena = metal_allocate(*context, bytes_per_stream);
            if (bytes_per_stream && !opened.arena) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /**
     *  Waits for every command buffer committed to every stream - the fleet's completion: one empty
     *  command buffer per stream, which completes only after every one committed before it.
     *
     *  @par Statuses:
     *  - @c device_error_k A stream's device is not listed.
     */
    status_t wait() noexcept {
        constexpr std::size_t completed_k = 4; // `MTLCommandBufferStatusCompleted`
        for (stream_t const& opened : streams_) {
            metal_stream_t const& stream = opened.handle;
            metal_context_t* const context = stream.context();
            if (!context) return status_t::device_error_k;
            metal_api_t const& api = *context->api;
            void* const pool = api.autorelease_pool_push();
            void* const commands = metal_send(api, stream.handle() ? stream.handle() : context->queue, "commandBuffer");
            std::size_t state = 0;
            if (commands) {
                metal_send<void>(api, commands, "commit");
                metal_send<void>(api, commands, "waitUntilCompleted");
                state = metal_send<std::size_t>(api, commands, "status");
            }
            api.autorelease_pool_pop(pool);
            if (state != completed_k) return status_t::device_error_k;
        }
        return status_t::success_k;
    }

    /** The stream at @p index, in the order the devices were named. */
    stream_t& stream(std::size_t index) noexcept { return streams_[index]; }

    /** The Apple ordinal stream @p index submits to - what a caller pairs its shards against. */
    int device(std::size_t index) const noexcept { return streams_[index].handle.device; }

    /** The first stream on @p device, or null when the executor drives none there. */
    stream_t* stream_on(int device) noexcept {
        for (stream_t& opened : streams_)
            if (opened.handle.device == device) return &opened;
        return nullptr;
    }

    /** Streams this executor drives, one per device. */
    std::size_t size() const noexcept { return streams_.size(); }

  private:
    growing_buffer<stream_t> streams_ {global_allocator};
    fu::spin_mutex_t held_;

    /** The ceiling on every stream's scratch bytes; zero lets scratch grow as geometry asks. */
    std::size_t scratch_limit_ = 0;
};

/** One context's reachability audit on the host, grown to the widest index it audited and kept, so
 *  a heal's rounds and later calls allocate nothing. */
struct metal_audit_t {

    /** The reached set and the two frontiers, a byte per row each. */
    growing_buffer<u8_t> reached {global_allocator};
    growing_buffer<u8_t> frontier {global_allocator};
    growing_buffer<u8_t> next_frontier {global_allocator};

    /** The rows the last audit never reached. */
    growing_buffer<u32_t> islands {global_allocator};

    /** The reached rows pointing at each row, counted before a reconnection takes any. */
    growing_buffer<u32_t> in_links {global_allocator};

    /** Levels the last audit swept to convergence. */
    std::size_t sweeps = 0;
};

/** The tally at @p counter in @p words: its low word joined with the high word it carries into. */
inline u64_t metal_tally(u32_t const* words, u32_t counter) noexcept {
    return words[counter] | u64_t {words[counter + 1]} << 32;
}

/** The counter words @p counters holds, @c metal_counter_words_k of them. */
inline u32_t* metal_words(metal_buffer_t const& counters) noexcept { return reinterpret_cast<u32_t*>(counters.host); }

/** One caller's scratch for driving an Apple index: the walks' arena, the staging of one batch, the
 *  telemetry counters and the single-query landing. Two contexts on one index share only the
 *  storage. */
template <typename buffers_allocator_type_ = metal_shared_allocator_t>
struct metal_traversal_context {

    /** The stream the context submits to, named an allocator because the residency-level harnesses
     *  build every context from its @c allocator_t. */
    using allocator_t = metal_stream_t;

    /** The backend of the arena, the staging and the counters, on the stream's device. */
    using buffers_allocator_t = buffers_allocator_type_;

    /** The single-query verb's landing, grown to the walk's breadth: the ascending matches, then,
     *  past them, the IDs and distances read back from the staging. */
    struct results_t {
        growing_buffer<scored_node_t> entries_ {global_allocator};
        std::size_t count_ = 0;

        scored_node_t* data() noexcept { return entries_.data(); }
        scored_node_t const* data() const noexcept { return entries_.data(); }
        scored_node_t& operator[](std::size_t rank) noexcept { return entries_[rank]; }
        scored_node_t const& operator[](std::size_t rank) const noexcept { return entries_[rank]; }
        std::size_t size() const noexcept { return count_; }
    };

    results_t results;

    /** Binds the context to @p stream's device; buffers wait for the first verb. */
    metal_traversal_context(allocator_t stream = {}) noexcept
        : stream_(std::move(stream)), context_(stream_.context()), buffers_allocator_(stream_) {
        if (context_) counters_ = buffers_allocator_.allocate(metal_counter_words_k * sizeof(u32_t));
    }
    metal_traversal_context(metal_traversal_context&&) noexcept = default;
    metal_traversal_context(metal_traversal_context const&) = delete;
    metal_traversal_context& operator=(metal_traversal_context const&) = delete;

    int device() const noexcept { return stream_.device; }
    void* stream() const noexcept { return stream_.handle() ? stream_.handle() : context_->queue; }

    /** The table every launch of this context goes through. */
    metal_api_t const& api() const noexcept { return *context_->api; }
    explicit operator bool() const noexcept { return context_ && counters_; }

    /** Waits for the last command buffer this context committed - one stream's completion, the
     *  executor's @c drain being the fleet's. */
    status_t wait() noexcept { return committed_.wait(); }

    /** Polls whether the last command buffer this context committed finished without blocking, so
     *  one host thread awaits whichever of several streams lands first. */
    status_t landed() const noexcept { return committed_.landed(); }

    /** The telemetry of the last submission, zeroed when it was staged; read once it landed. */
    vram_telemetry_block_t spent() const noexcept {
        u32_t const* const words = metal_words(counters_);
        vram_telemetry_block_t block {};
        block.hops = metal_tally(words, metal_hops_k);
        block.scored = metal_tally(words, metal_scored_k);
        block.restarts = metal_tally(words, metal_restarts_k);
        block.lock_attempts = metal_tally(words, metal_lock_attempts_k);
        block.lock_skips = metal_tally(words, metal_lock_skips_k);
        block.reselections = metal_tally(words, metal_reselections_k);
        block.locked_drops = metal_tally(words, metal_locked_drops_k);
        block.visited_saturations = metal_tally(words, metal_saturations_k);
        block.changed_nodes = words[metal_changed_k];
        block.tail_compactions = metal_tally(words, metal_tail_compactions_k);
        block.tail_dropped = metal_tally(words, metal_tail_dropped_k);
        return block;
    }

    /** Whether the last walk met a value-lookup miss, a node its value lookup had no row for, which
     *  refuses it; read once it landed. */
    bool missed() const noexcept {
        u32_t const* const words = metal_words(counters_);
        return words[metal_missed_k] != 0;
    }

    /** Where the last walk left its outputs in the staging, read back once the context lands. */
    struct landing_t {
        std::size_t rows = 0, count = 0;
        std::size_t first_offset = 0, second_offset = 0, third_offset = 0;
    };

    allocator_t stream_;
    metal_context_t* context_ = nullptr;
    buffers_allocator_t buffers_allocator_;
    metal_buffer_t counters_;
    metal_buffer_t arena_;
    metal_buffer_t staging_;
    landing_t landing_;

    /** The host-side audit a refine heals by and a health check reports, grown like the staging. */
    metal_audit_t audit_;

    /** The last command buffer submitted through this context; its stream runs them in commit
     *  order, so this one landing means every earlier one did. */
    metal_launch_t committed_;

    /** Monotone epoch floor over the arena's visited slots, so the arena zeroes only when fresh or
     *  wrapped. */
    u64_t epoch_floor_ = 0;

    /** Dispatches @p launch over @p groups threadgroups of @p threads and keeps it as the command
     *  buffer the context awaits. */
    status_t submit_(metal_launch_t& launch, std::size_t groups, std::size_t threads) noexcept {
        if (status_t const submitted = launch.submit(groups, threads); failed(submitted)) return submitted;
        committed_ = std::move(launch);
        return status_t::success_k;
    }

    /** Grows @p buffer to at least @p bytes; answers whether it was reallocated, zeroed by
     *  Metal. */
    status_t ensure_(metal_buffer_t& buffer, std::size_t bytes, bool& fresh) noexcept {
        fresh = false;
        if (bytes <= buffer.bytes) return status_t::success_k;
        buffer = buffers_allocator_.allocate(bytes);
        fresh = true;
        return buffer ? status_t::success_k : status_t::bad_alloc_k;
    }
};

} // namespace unum::usearch::utilities
