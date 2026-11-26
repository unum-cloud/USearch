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

#include <span>    // `std::span` - the device list an executor opens over
#include <utility> // `std::exchange`, `std::move`, `std::swap`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the executor's held word, the pipeline count
#include <forkunion/locks.hpp>   // `fu::fair_mutex_t` - the pipeline tables

#include "usearch/utilities/blocking_files.hpp" // `shared_library_t` - Metal opened at first use
#include "usearch/utilities/ram_buffers.hpp"    // `growing_buffer` - an executor's streams and the pipeline registry
#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_apple_shared.hpp" // `metal_counter_t` - the words a context reads its telemetry from

namespace unum::usearch::utilities {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

#if USEARCH_ARCH_METAL_

/** The vocabulary and the walk queue every Apple context and its kernels share, which each MSL
 *  source starts with, since MSL cannot include. */
inline constexpr char metal_context_source_k[] = {
#embed "vram_apple_shared.hpp" suffix(, '\n', )
#embed "vram_queue.metal" suffix(, 0)
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

#pragma region Shared VRAM Driver

/** The Metal Shading Language versions an engine compiles its sources under, each spelled as its
 *  @c MTLLanguageVersion value. */
enum class metal_language_t : std::size_t {
    msl_3_1_k = (3u << 16) | 1u,
    msl_3_2_k = (3u << 16) | 2u,
    msl_4_0_k = (4u << 16) | 0u,
};

/**
 *  @brief The Metal, Objective-C runtime, I/O Kit and Core Foundation entry points every engine
 *      calls, resolved on first use from the system's own frameworks and @c libobjc.
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
    metal_selector_t (*register_selector)(char const*) = nullptr; // vendor-boundary
    void* (*get_class)(char const*) = nullptr;                    // vendor-boundary
    void* (*registry_entry_id_matching)(u64_t) = nullptr;
    u32_t (*get_matching_service)(u32_t, void*) = nullptr;
    void* (*search_registry_property)(u32_t, char const*, void const*, void const*, u32_t) = nullptr; // vendor-boundary
    int (*release_registry_object)(u32_t) = nullptr;
    void* (*create_string)(void const*, char const*, u32_t) = nullptr; // vendor-boundary
    unsigned char (*get_number_value)(void const*, long, void*) = nullptr;
    void (*release_reference)(void const*) = nullptr;
    void* (*autorelease_push)() = nullptr;
    void (*autorelease_pop)(void*) = nullptr;

    /** Whether every library loaded with every entry resolved. */
    explicit operator bool() const noexcept { return autorelease_pop != nullptr; }

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
            runtime.resolve("objc_autoreleasePoolPush", api.autorelease_push) &&
            runtime.resolve("objc_autoreleasePoolPop", api.autorelease_pop);
        return resolved ? api : metal_api_t {};
    }
};

/** Sends @p selector to @p target with @p arguments through @p api, cast to the exact prototype the
 *  call has. */
template <typename result_type_ = void*, typename... argument_types_>
inline result_type_ metal_send(metal_api_t const& api, void* target, char const* selector, // vendor-boundary
                               argument_types_... arguments) noexcept {
    using function_t = result_type_ (*)(void*, metal_selector_t, argument_types_...);
    return reinterpret_cast<function_t>(api.send)(target, api.register_selector(selector), arguments...);
}

/** An @c NSString over @p text, autoreleased. */
inline void* metal_string(metal_api_t const& api, char const* text) noexcept { // vendor-boundary
    return metal_send(api, reinterpret_cast<void*>(api.get_class("NSString")), "stringWithUTF8String:", text);
}

/** One library of MSL source compiled for one Apple device and released with this owner; kernels
 *  are found by name, each built into a pipeline. */
struct metal_module_t {
    using function_t = void*;

    metal_module_t() noexcept = default;
    metal_module_t(metal_module_t&& other) noexcept
        : api_(other.api_), device_(other.device_), library_(std::exchange(other.library_, nullptr)) {}
    metal_module_t& operator=(metal_module_t&& other) noexcept {
        std::swap(api_, other.api_), std::swap(device_, other.device_), std::swap(library_, other.library_);
        return *this;
    }
    metal_module_t(metal_module_t const&) = delete;
    metal_module_t& operator=(metal_module_t const&) = delete;
    ~metal_module_t() noexcept {
        if (library_) metal_send<void>(*api_, library_, "release");
    }

    /**
     *  @brief Compiles @p source, NUL-terminated MSL text, under @p language on @p device, which
     *      is the @c id<MTLDevice> the library runs on.
     *
     *  @par Statuses:
     *  - @c device_k The @p source is empty or unterminated, or the compiler rejected it.
     */
    static expected<metal_module_t> load(metal_api_t const& api, void* device, bytes_view_t source,
                                         metal_language_t language) noexcept {
        if (source.empty() || source.back() != byte_t {0} || !api || !device)
            return {{}, {status_code_t::device_k, "The device refused the MSL source"}};
        metal_module_t module;
        module.api_ = &api;
        module.device_ = device;
        void* const autoreleased = api.autorelease_push();
        void* const options = metal_send(api, metal_send(api, api.get_class("MTLCompileOptions"), "alloc"), "init");
        metal_send<void>(api, options, "setLanguageVersion:", static_cast<std::size_t>(language));
        void* error = nullptr;
        module.library_ = metal_send(api, device, "newLibraryWithSource:options:error:",
                                     metal_string(api, reinterpret_cast<char const*>(source.data())), options, &error);
        metal_send<void>(api, options, "release");
        api.autorelease_pop(autoreleased);
        if (!module.library_) return {{}, {status_code_t::device_k, "The device refused the MSL source"}};
        return {std::move(module), status_code_t::success_k};
    }

    /** The table the library was compiled through, which launches of its kernels go through. */
    metal_api_t const& api() const noexcept { return *api_; }

    /** The pipeline of the kernel exported as @p name, owned by the caller, or null when the
     *  library has none or its simdgroups are not the @c metal_simdgroup_width_k lanes every MSL
     *  source is written against. */
    function_t function(char const* name) const noexcept { // vendor-boundary
        if (!library_) return nullptr;
        metal_api_t const& api = *api_;
        void* const autoreleased = api.autorelease_push();
        void* const function = metal_send(api, library_, "newFunctionWithName:", metal_string(api, name));
        void* pipeline = nullptr;
        if (function) {
            void* error = nullptr;
            pipeline = metal_send(api, device_, "newComputePipelineStateWithFunction:error:", function, &error);
            metal_send<void>(api, function, "release");
        }
        api.autorelease_pop(autoreleased);
        if (pipeline && metal_send<std::size_t>(api, pipeline, "threadExecutionWidth") != metal_simdgroup_width_k)
            metal_send<void>(api, std::exchange(pipeline, nullptr), "release");
        return pipeline;
    }

  private:
    metal_api_t const* api_ = nullptr;
    void* device_ = nullptr;
    void* library_ = nullptr;
};

#pragma endregion Shared VRAM Driver

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

/** One compiled entry point, on the device and from the source it was built for, and the library
 *  it shares with that source's other entry points; the name is copied, as callers spell it into
 *  their own buffers. */
struct metal_pipeline_t {
    void* device = nullptr;
    char const* source = nullptr;
    char name[metal_pipeline_name_capacity_k] = {};
    std::size_t library = 0;
    metal_pipeline_state_t state;
};

/** Every pipeline the process built on every Apple GPU, and the libraries they come from, compiled
 *  at the first ask and kept for the life of the runtime holding them, so every executor and index
 *  on one device shares them. */
template <>
struct kernel_tables<metal_module_t> {

    /** Pipelines and libraries the process keeps at most, every kernel a build embeds having room
     *  on a few devices. */
    inline static constexpr std::size_t pipelines_capacity_k = 4096, libraries_capacity_k = 256;

    /**
     *  @brief Kernel @p name's pipeline on @p device from the library compiled out of @p source,
     *      built on first use.
     *
     *  A built pipeline is found without a lock, as launches ask. A build takes the lock, so a
     *  caller meeting another's build sleeps until it ends, as module loads wait in other drivers.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The pipeline or its library could not be listed.
     *  - @c device_k The source would not compile, or it lacks the kernel, or the pipeline's
     *    simdgroups are not the @c metal_simdgroup_width_k lanes of every MSL source.
     */
    expected<metal_pipeline_state_t> pipeline(metal_api_t const& api, void* device, char const* source,
                                              char const* name) noexcept {
        if (metal_pipeline_t const* const built = find_(device, source, name))
            return {built->state, status_code_t::success_k};
        guard_.lock();
        auto [state, status] = build_(api, device, source, name);
        guard_.unlock();
        return {state, status};
    }

  private:
    /** The built pipeline of @p name on @p device from @p source, or null; the count is read with
     *  acquire, so every pipeline it covers is whole. */
    metal_pipeline_t const* find_(void* device, char const* source, char const* name) noexcept {
        std::size_t const count =
            fu::standard_atomic_ref<std::size_t>(pipelines_count_).load(std::memory_order_acquire);
        for (std::size_t position = 0; position != count; ++position) {
            metal_pipeline_t const& entry = pipelines_[position];
            if (entry.device == device && entry.source == source && std::strcmp(entry.name, name) == 0) return &entry;
        }
        return nullptr;
    }

    /** Builds the pipeline of @p name on @p device from @p source unless built meanwhile, with
     *  @c guard_ held; a new one is published by a release of the count. */
    expected<metal_pipeline_state_t> build_(metal_api_t const& api, void* device, char const* source,
                                            char const* name) noexcept {
        if (metal_pipeline_t const* const built = find_(device, source, name))
            return {built->state, status_code_t::success_k};
        std::size_t const count = pipelines_count_;
        if (count == pipelines_capacity_k) return {{}, status_code_t::out_of_memory_k};
        std::size_t library = libraries_count_;
        for (std::size_t position = 0; position != count; ++position)
            if (pipelines_[position].device == device && pipelines_[position].source == source)
                library = pipelines_[position].library;
        if (library == libraries_count_) {
            if (library == libraries_capacity_k) return {{}, status_code_t::out_of_memory_k};
            bytes_view_t const text {reinterpret_cast<byte_t const*>(source), std::strlen(source) + 1};
            // MSL 3.2 is the first with the device-scope atomic fences every USearch source uses.
            auto [compiled, load_status] = metal_module_t::load(api, device, text, metal_language_t::msl_3_2_k);
            if (failed(load_status)) return {{}, load_status};
            libraries_[library] = std::move(compiled);
            ++libraries_count_;
        }
        void* const pipeline = libraries_[library].function(name);
        if (!pipeline) return {{}, {status_code_t::device_k, "The library builds no such pipeline"}};
        u32_t const max_threads = static_cast<u32_t>(
            metal_send<std::size_t>(api, pipeline, "maxTotalThreadsPerThreadgroup"));
        metal_pipeline_t& entry = pipelines_[count];
        entry = {device, source, {}, library, {pipeline, max_threads}};
        std::strncpy(entry.name, name, metal_pipeline_name_capacity_k - 1);
        fu::standard_atomic_ref<std::size_t>(pipelines_count_).store(count + 1, std::memory_order_release);
        return {entry.state, status_code_t::success_k};
    }

    /** Every built pipeline, in build order. */
    metal_pipeline_t pipelines_[pipelines_capacity_k];

    /** How many pipelines are whole, grown under @c guard_ and read through the atomic reference
     *  without it. */
    std::size_t pipelines_count_ = 0;

    /** Every compiled library, which only builds touch. */
    metal_module_t libraries_[libraries_capacity_k];
    std::size_t libraries_count_ = 0;

    /** Held across a build, so each pipeline builds once. */
    fu::fair_mutex_t guard_;
};

/** One listed Apple GPU's process-wide state: the table its calls go through, the device, its
 *  default queue - the twin of CUDA's null stream - and its family and cores, as it was listed. */
struct metal_context_t {
    metal_api_t const* api = nullptr;
    void* device = nullptr;
    void* queue = nullptr;

    /** The newest Apple GPU family the device supports, as @ref apple_family_of numbers it. */
    u32_t family = 0;

    /** The GPU cores the I/O Registry lists for the device, zero when it lists none. */
    u32_t multiprocessor_count = 0;
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
        void* const autoreleased = api.autorelease_push();
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
        api.autorelease_pop(autoreleased);
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

    /** Lanes per simdgroup, the @c threadExecutionWidth every pipeline built for it reports, as
     *  pipelines of any other width are refused. */
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

    /** Bytes one threadgroup keeps to itself on one GPU core, the twin of
     *  @c cuda_device_specs_t::cache_bytes_per_core; Metal names no wider per-core figure. */
    std::size_t cache_bytes_per_core() const noexcept { return shared_memory_per_block_bytes; }

    /**
     *  @brief Queries Metal and the I/O Registry for the specification of @p context, the device at
     *      @p ordinal, whose simdgroups are the @c metal_simdgroup_width_k lanes every pipeline
     *      built for it has; meaningful only on @c success_k.
     *
     *  @par Statuses:
     *  - @c unsupported_k The device's registry lists no core count.
     */
    static expected<metal_device_specs_t> query(metal_context_t const& context, int ordinal) noexcept {
        if (!context.multiprocessor_count) return {{ordinal}, status_code_t::unsupported_k};
        metal_api_t const& api = *context.api;
        void* const device = context.device;
        metal_device_specs_t specs {ordinal, context.multiprocessor_count, context.family, metal_simdgroup_width_k};
        specs.max_threads_per_block = static_cast<u32_t>(
            metal_send<metal_size_t>(api, device, "maxThreadsPerThreadgroup").width);
        specs.shared_memory_per_block_bytes = metal_send<std::size_t>(api, device, "maxThreadgroupMemoryLength");
        specs.max_buffer_bytes = metal_send<std::size_t>(api, device, "maxBufferLength");
        specs.working_set_bytes = static_cast<std::size_t>(
            metal_send<u64_t>(api, device, "recommendedMaxWorkingSetSize"));
        return {specs, status_code_t::success_k};
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

/**
 *  @brief Allocates @p bytes of shared storage on @p context's device through a buffer registry,
 *      answering their host address.
 *
 *  The engine's half of @c metal_executor_t::stream_t's allocation, which USearch keeps no
 *  registry for yet, as its indexes own their @c metal_buffer_t outright.
 *
 *  @par Statuses:
 *  - @c unsupported_k USearch keeps no buffer registry.
 */
inline expected<std::uintptr_t> metal_allocate_block(metal_context_t&, std::size_t) noexcept {
    return {0, {status_code_t::unsupported_k, "USearch keeps no Metal buffer registry"}};
}

/** Releases what @ref metal_allocate_block answered, which USearch never does. */
inline void metal_free_block(metal_context_t&, std::uintptr_t) noexcept {}

/** The kernels a stream launches: the process's pipeline tables, which the GPU runtime owns. */
using metal_kernels_t = kernel_tables<metal_module_t>*;

/**
 *  @brief USearch's memory on one Apple stream: the one shared buffer a walk carves its state from,
 *      sized by @ref reserve and replaced whole when a call asks for more.
 *
 *  The pipelines it launches sit in the process's tables, which every device's context points at.
 */
struct metal_scratch_t {

    /** One device's run of scratch: one shared buffer. */
    metal_buffer_t arena;

    /** The ceiling on the run's bytes, zero for none. */
    std::size_t limit = 0;

    /** Keeps @p scratch_limit as the ceiling on the run, which waits for the first @ref reserve,
     *  and points the stream's kernels at the @c metal_kernels of @p runtime, which each device's
     *  context points at too. */
    template <typename stream_type_, typename runtime_type_>
    status_t open(stream_type_& stream, runtime_type_& runtime, std::size_t& scratch_limit) noexcept {
        stream.kernels = &runtime.metal_kernels;
        limit = scratch_limit;
        return status_code_t::success_k;
    }

    /**
     *  @brief Replaces the run with @p bytes of shared storage on @p stream 's device; the old run
     *      lives on in whichever command buffer still binds it.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The device refused the buffer.
     */
    template <typename stream_type_>
    status_t reserve(stream_type_ const& stream, std::size_t bytes) noexcept {
        arena = metal_allocate(*stream.handle.context(), bytes);
        if (bytes && !arena) return status_code_t::out_of_memory_k;
        return status_code_t::success_k;
    }

    /** Releases the run once @p stream drained. */
    template <typename stream_type_>
    void release(stream_type_ const&) noexcept {
        arena = {};
    }
};

/** One command buffer being encoded on one queue: binds buffers and bytes, dispatches, and commits
 *  without waiting. Every call to a device runs through one of these, so encoders live on the
 *  caller's stack and any number of threads encode at once; a committed one stays retained as
 *  Apple's event, the twin of @c cuda_event_t, since a queue runs its command buffers in commit
 *  order, so its finishing means every earlier one's. */
struct metal_launch_t {
    metal_api_t const* api = nullptr;
    void* commands = nullptr;
    void* encoder = nullptr;

    /** No command buffer - one with nothing left to run. */
    metal_launch_t() noexcept = default;

    /** Opens one command buffer on @p queue with a compute encoder for @p pipeline, through
     *  @p api. */
    metal_launch_t(metal_api_t const& api, void* queue, void* pipeline) noexcept : api(&api) {
        void* const autoreleased = api.autorelease_push();
        commands = metal_send(api, queue, "commandBuffer");
        if (commands) metal_send<void>(api, commands, "retain");
        if (commands) encoder = metal_send(api, commands, "computeCommandEncoder");
        if (encoder) {
            metal_send<void>(api, encoder, "retain");
            metal_send<void>(api, encoder, "setComputePipelineState:", pipeline);
        }
        // A command buffer without its encoder is never committed, so none is kept to await.
        else if (commands) metal_send<void>(api, std::exchange(commands, nullptr), "release");
        api.autorelease_pop(autoreleased);
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
     *  @brief Dispatches @p groups threadgroups of @p threads each and commits, without waiting.
     *
     *  @par Statuses:
     *  - @c device_k The encoder is not open.
     */
    status_t submit(std::size_t groups, std::size_t threads) noexcept {
        if (!encoder) return status_code_t::device_k;
        metal_send<void, metal_size_t, metal_size_t>(*api, encoder, "dispatchThreadgroups:threadsPerThreadgroup:",
                                                     metal_size_t {groups}, metal_size_t {threads});
        metal_send<void>(*api, encoder, "endEncoding");
        metal_send<void>(*api, encoder, "release");
        encoder = nullptr;
        metal_send<void>(*api, commands, "commit");
        return status_code_t::success_k;
    }

    /** Polls the committed command buffer without blocking; nothing committed has nothing to
     *  run. */
    status_t finished() const noexcept {
        if (!commands || encoder) return status_code_t::success_k;
        constexpr std::size_t completed_k = 4, error_k = 5; // `MTLCommandBufferStatusCompleted`, `...Error`
        std::size_t const state = metal_send<std::size_t>(*api, commands, "status");
        return state == completed_k ? status_code_t::success_k
               : state == error_k   ? status_code_t::device_k
                                    : status_code_t::deferred_k;
    }

    /** Waits for the committed command buffer and answers whether the device finished it. */
    status_t wait() noexcept {
        if (commands && !encoder) metal_send<void>(*api, commands, "waitUntilCompleted");
        return finished();
    }
};

#pragma region Shared VRAM Executor

/** Owned @c id<MTLCommandQueue> on one Apple GPU, the twin of @c cuda_stream_t; null rides the
 *  device's default queue, which its @ref metal_context_t keeps, so the stream names its device
 *  and context. Move-only, and released when it dies. */
struct metal_stream_t {

    /** A stream on no device, which only fills the default of a record built later. */
    metal_stream_t() noexcept = default;

    /** Takes @p handle, owned and opened on the device of @p context, or rides that device's
     *  default queue when null. */
    explicit metal_stream_t(metal_context_t& context, void* handle = nullptr) noexcept
        : context_(&context), handle_(handle) {}
    metal_stream_t(metal_stream_t&& other) noexcept
        : context_(other.context_), handle_(std::exchange(other.handle_, nullptr)) {}
    metal_stream_t& operator=(metal_stream_t&& other) noexcept {
        std::swap(context_, other.context_), std::swap(handle_, other.handle_);
        return *this;
    }
    metal_stream_t(metal_stream_t const&) = delete;
    metal_stream_t& operator=(metal_stream_t const&) = delete;
    ~metal_stream_t() noexcept {
        if (handle_) metal_send<void>(*context_->api, handle_, "release");
    }

    /**
     *  @brief Opens a command queue on the device of @p context, owned from now on.
     *
     *  @par Statuses:
     *  - @c device_k The device refused the queue.
     */
    static expected<metal_stream_t> open(metal_context_t& context) noexcept {
        void* const queue = metal_send(*context.api, context.device, "newCommandQueue");
        if (!queue) return {{}, {status_code_t::device_k, "Device refused a command queue"}};
        return {metal_stream_t {context, queue}, status_code_t::success_k};
    }

    /**
     *  @brief Checks that @p handle, a caller's @c id<MTLCommandQueue>, may be adopted on the
     *      device of @p context, without retaining it.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k The @p handle is null.
     *  - @c config_mismatch_k The queue sits on another device.
     */
    static status_t adoptable(metal_context_t const& context, void* handle) noexcept {
        if (!handle) return {status_code_t::invalid_argument_k, "A default stream cannot be adopted"};
        if (metal_send(*context.api, handle, "device") != context.device)
            return {status_code_t::config_mismatch_k, "A stream sits on another device"};
        return {};
    }

    /** The owned queue, or null for the device's default one. */
    void* owned() const noexcept { return handle_; }

    /** The queue work is committed to: the owned one, or the device's default one. */
    void* queue() const noexcept { return handle_ ? handle_ : context_->queue; }

    /** The context of the stream's device, or null for a stream on no device. */
    metal_context_t* context() const noexcept { return context_; }

  private:
    metal_context_t* context_ = nullptr;
    void* handle_ = nullptr;
};

/**
 *  @brief The queues a call commits to, one per entry of the devices it was spawned over, in the
 *      caller's order, each with its kernels and the engine's memory.
 *
 *  Work is committed, never waited on, so one host thread drives every device. Each stream record
 *  is built in place on the heap and never moves; it dies waiting for its queue, then releasing
 *  its memory, then releasing its queue. Apple's memory is unified, so a copy or a fill is a host
 *  one, done when it returns.
 */
struct metal_executor_t {

    /** The engine's memory on one stream. */
    using scratch_t = metal_scratch_t;

    /** One call's hold on the executor: the word it set, which it clears when it dies. Move-only,
     *  so exactly one owner releases the hold. */
    class claim_t {
      public:
        claim_t() noexcept = default;
        explicit claim_t(u32_t& held_word) noexcept : held_word_(&held_word) {}
        claim_t(claim_t&& other) noexcept : held_word_(std::exchange(other.held_word_, nullptr)) {}
        claim_t& operator=(claim_t&& other) noexcept {
            std::swap(held_word_, other.held_word_);
            return *this;
        }
        claim_t(claim_t const&) = delete;
        claim_t& operator=(claim_t const&) = delete;
        ~claim_t() noexcept {
            if (held_word_) fu::standard_atomic_ref<u32_t>(*held_word_).store(0, std::memory_order_release);
        }

      private:
        u32_t* held_word_ = nullptr;
    };

    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::apple_k;

    /** One device's queue, the kernels it launches, and the engine's memory, declared last so it
     *  dies first. */
    struct stream_t {

        /** The Apple ordinal of the device this stream submits to, as the GPU runtime lists it. */
        std::size_t ordinal = 0;

        /** Opened by the executor, adopted from the caller, or the device's default queue. */
        metal_stream_t handle;

        /** The kernels this stream launches, as the engine resolves them when the record opens. */
        metal_kernels_t kernels = {};

        /** The engine's memory, safe to reuse only in this stream's queue order, opened once this
         *  record sits in the executor. */
        scratch_t memory;

        /** Takes @p opened as the queue of device @p device. */
        stream_t(std::size_t device, metal_stream_t&& opened) noexcept : ordinal(device), handle(std::move(opened)) {}
        stream_t(stream_t const&) = delete;
        stream_t& operator=(stream_t const&) = delete;

        /** Waits for an owned queue, then releases the memory; the queue is released after. A
         *  record owning none, a placeholder or the device's default queue, waits for nothing. */
        ~stream_t() noexcept {
            [[maybe_unused]] status_t const ready = handle.owned() ? wait() : bind();
            memory.release(*this);
        }

        /**
         *  @brief Checks the stream names a device, which Metal needs no thread binding for.
         *
         *  @par Statuses:
         *  - @c device_k The stream names no device.
         */
        status_t bind() const noexcept {
            if (!handle.context()) return {status_code_t::device_k, "The stream names no device"};
            return {};
        }

        /**
         *  @brief Waits for every command buffer committed so far, committing an empty one, which
         *      a queue completes only after every one committed before it.
         *
         *  @par Statuses:
         *  - @c device_k The stream names no device, or the device failed the work.
         */
        status_t wait() noexcept {
            // The state of a command buffer the device ran, `MTLCommandBufferStatusCompleted`.
            constexpr std::size_t completed_k = 4;
            metal_context_t const* const context = handle.context();
            if (!context) return {status_code_t::device_k, "The stream names no device"};
            metal_api_t const& api = *context->api;
            void* const autoreleased = api.autorelease_push();
            void* const commands = metal_send(api, handle.queue(), "commandBuffer");
            std::size_t state = 0;
            if (commands) {
                metal_send<void>(api, commands, "commit");
                metal_send<void>(api, commands, "waitUntilCompleted");
                state = metal_send<std::size_t>(api, commands, "status");
            }
            api.autorelease_pop(autoreleased);
            if (state != completed_k) return {status_code_t::device_k, "The device failed the committed work"};
            return {};
        }

        /**
         *  @brief Orders this stream after everything @p other committed so far. Metal has no
         *      event entry here, so it waits on @p other from the host, and the follow is done
         *      once it returns.
         *
         *  @par Statuses:
         *  - @c device_k @p other names no device, or the device failed its work.
         */
        status_t follow(stream_t& other) noexcept { return other.wait(); }

        /**
         *  @brief Allocates @p bytes of shared storage on this stream's device through its buffer
         *      registry, answering their host address, which kernels reach too.
         *
         *  @par Statuses:
         *  - @c out_of_memory_k The device refused the allocation.
         *  - @c unsupported_k The engine keeps no buffer registry.
         */
        expected<std::uintptr_t> allocate_async(std::size_t bytes) noexcept {
            return metal_allocate_block(*handle.context(), bytes);
        }

        /** Releases @p address, which @ref allocate_async answered; work still committed on it
         *  keeps it alive until it finishes. */
        void free_async(std::uintptr_t address) noexcept { metal_free_block(*handle.context(), address); }

        /** Copies @p bytes from @p source to @p target in unified memory; done when it returns. */
        status_t copy_async(std::uintptr_t target, std::uintptr_t source, std::size_t bytes) noexcept {
            std::memcpy(reinterpret_cast<void*>(target), reinterpret_cast<void const*>(source), bytes);
            return {};
        }

        /** Copies @p source into memory at @p target, as @ref copy_async does. */
        status_t upload_async(std::uintptr_t target, bytes_view_t source) noexcept {
            return copy_async(target, reinterpret_cast<std::uintptr_t>(source.data()), source.size());
        }

        /** Copies memory at @p source into @p target, as @ref copy_async does. */
        status_t download_async(bytes_span_t target, std::uintptr_t source) noexcept {
            return copy_async(reinterpret_cast<std::uintptr_t>(target.data()), source, target.size());
        }

        /** Sets @p bytes at @p target to @p value in unified memory; done when it returns. */
        status_t fill_async(std::uintptr_t target, byte_t value, std::size_t bytes) noexcept {
            std::memset(reinterpret_cast<void*>(target), static_cast<int>(value), bytes);
            return {};
        }
    };

    metal_executor_t() noexcept = default;
    metal_executor_t(metal_executor_t&& other) noexcept { swap(other); }
    metal_executor_t& operator=(metal_executor_t&& other) noexcept { return swap(other), *this; }
    metal_executor_t(metal_executor_t const&) = delete;
    metal_executor_t& operator=(metal_executor_t const&) = delete;

    /** Exchanges the stream records and the ceiling, none of which moves in memory; the hold
     *  stays, since a move with a call in flight is misuse. */
    void swap(metal_executor_t& other) noexcept {
        std::swap(streams_, other.streams_);
        std::swap(streams_count_, other.streams_count_);
    }

    /**
     *  @brief Claims the executor for one call, answering the hold, which releases it when it dies.
     *
     *  @par Statuses:
     *  - @c contended_k Another call holds the executor.
     */
    expected<claim_t> claim() noexcept {
        u32_t idle = 0;
        if (!fu::standard_atomic_ref<u32_t>(held_).compare_exchange_strong(idle, 1, std::memory_order_acquire,
                                                                           std::memory_order_relaxed))
            return {{}, {status_code_t::contended_k, "Another call holds the executor"}};
        return {claim_t {held_}, status_code_t::success_k};
    }

    /**
     *  @brief Spawns one command queue per entry of @p devices among @p contexts, or retains the
     *      @c id<MTLCommandQueue> of @p streams beside it.
     *
     *  Each stream's memory and kernels open with @p memory_arguments. The executor releases its
     *  own reference to a retained queue when it dies, while the caller keeps and releases theirs.
     *  Adoption follows every fallible step, so a refused spawn retains none of @p streams.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k No device is named, more than a listing holds, the @p streams count
     *    differs from theirs, or a queue is null.
     *  - @c config_mismatch_k A queue sits on another device.
     *  - @c device_k A device is not listed, or refused its queue.
     *  - @c out_of_memory_k A stream record or its memory could not be allocated.
     */
    template <typename... memory_arguments_>
    static expected<metal_executor_t> spawn(metal_contexts_t& contexts, std::span<int const> devices,
                                            std::span<void* const> streams,
                                            memory_arguments_&... memory_arguments) noexcept {
        if (!streams.empty() && streams.size() != devices.size())
            return {{}, {status_code_t::invalid_argument_k, "One stream per device"}};
        for (std::size_t index = 0; index != streams.size(); ++index) {
            metal_context_t* const context = contexts.context(devices[index]);
            if (!context) return {{}, {status_code_t::device_k, "Device is not listed"}};
            if (status_t const checked = metal_stream_t::adoptable(*context, streams[index]); checked.failed())
                return {{}, checked};
        }
        // Adopted queues wait on placeholders that own nothing until every fallible step is behind.
        auto const placeholder = [](metal_context_t& context) noexcept -> expected<metal_stream_t> {
            return {metal_stream_t {context}, status_code_t::success_k};
        };
        auto [executor, build_status] = build(contexts, devices, streams.empty() ? &metal_stream_t::open : +placeholder,
                                              memory_arguments...);
        if (build_status.failed()) return {{}, build_status};
        // Every fallible step is behind, so a refusal above retained none of the caller's queues.
        for (std::size_t index = 0; index != streams.size(); ++index) {
            metal_context_t& context = *contexts.context(devices[index]);
            executor.streams_[index]->handle = metal_stream_t {context,
                                                               metal_send(*context.api, streams[index], "retain")};
        }
        return {std::move(executor), status_code_t::success_k};
    }

    /** Waits for every command buffer committed to every stream, the fleet's completion. */
    status_t wait() noexcept {
        for (std::size_t index = 0; index != streams_count_; ++index)
            if (status_t const drained = streams_[index]->wait(); drained.failed()) return drained;
        return {};
    }

    /** The stream at @p index, in the order the devices were named. */
    stream_t& stream(std::size_t index) noexcept { return *streams_[index]; }
    stream_t const& stream(std::size_t index) const noexcept { return *streams_[index]; }

    /** Streams this executor drives, one per device. */
    std::size_t size() const noexcept { return streams_count_; }

    /** Builds one stream record per entry of @p devices over the queue @p open answers there,
     *  and opens its memory and kernels in place. */
    template <typename... memory_arguments_>
    static expected<metal_executor_t> build(metal_contexts_t& contexts, std::span<int const> devices,
                                            expected<metal_stream_t> (*open)(metal_context_t&),
                                            memory_arguments_&... memory_arguments) noexcept {
        if (devices.empty() || devices.size() > gpu_devices_capacity_k)
            return {{}, {status_code_t::invalid_argument_k, "An executor names one to a listing's devices"}};
        metal_executor_t executor;
        for (int const device : devices) {
            metal_context_t* const context = contexts.context(device);
            if (!context) return {{}, {status_code_t::device_k, "Device is not listed"}};
            auto [opened, open_status] = open(*context);
            if (open_status.failed()) return {{}, open_status};
            std::unique_ptr<stream_t>& record = executor.streams_[executor.streams_count_];
            record.reset(new (std::nothrow) stream_t {static_cast<std::size_t>(device), std::move(opened)});
            if (!record) return {{}, {status_code_t::out_of_memory_k, "No room for a stream record"}};
            ++executor.streams_count_;
            if (status_t const opened_memory = record->memory.open(*record, memory_arguments...);
                opened_memory.failed())
                return {{}, opened_memory};
        }
        return {std::move(executor), status_code_t::success_k};
    }

  private:
    /** One record per named entry, in the caller's order, each on the heap so it never moves. */
    std::unique_ptr<stream_t> streams_[gpu_devices_capacity_k];
    std::size_t streams_count_ = 0;

    /** One while a caller drives the streams, set and cleared through the class's atomic
     *  reference; never swapped. */
    u32_t held_ = 0;
};

#pragma endregion Shared VRAM Executor

/** The default queue of the device of @p context, which the context made when the process listed
 *  the device and keeps, for a call without an executor; it owns nothing. */
inline expected<metal_stream_t> metal_default_stream(metal_context_t& context) noexcept {
    return {metal_stream_t {context}, status_code_t::success_k};
}

/**
 *  @brief Spawns an executor submitting a call without one to the default queue of each of
 *      @p devices among @p contexts, opening no queue, so the call is final once its work ends.
 *
 *  Each stream's memory opens with @p memory_arguments.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k No device is named, or more than a listing holds.
 *  - @c device_k A device is not listed.
 *  - @c out_of_memory_k A stream record or its memory could not be allocated.
 */
template <typename... memory_arguments_>
expected<metal_executor_t> on_default_streams(metal_contexts_t& contexts, std::span<int const> devices,
                                              memory_arguments_&... memory_arguments) noexcept {
    return metal_executor_t::build(contexts, devices, &metal_default_stream, memory_arguments...);
}

/**
 *  @brief Claims @p bytes of scratch on @p stream ahead of the fan-out that maps state onto it.
 *
 *  @par Statuses:
 *  - @c out_of_memory_k The @p bytes pass the stream's ceiling, or its memory could not grow.
 *  - @c device_k The stream names no device.
 */
inline status_t reserve_scratch(metal_executor_t::stream_t& stream, std::size_t bytes) noexcept {
    if (stream.memory.limit && bytes > stream.memory.limit) return status_code_t::out_of_memory_k;
    if (status_t const bound = stream.bind(); failed(bound)) return bound;
    return stream.memory.reserve(stream, bytes);
}

/** The storage backend of the Apple index and the scratch of its walks: shared buffers on one
 *  device, each allocation its own @c MTLBuffer, so a matrix grows by allocating a wider one and
 *  copying. It names the queue work is committed to, pointing at an executor's stream owner, which
 *  never moves, or the device's default queue, and the pipeline tables launches read. */
struct metal_shared_allocator_t {
    metal_shared_allocator_t() noexcept = default;

    /** Allocates on the Apple GPU @p device, whose context is @p context, committing to its
     *  default queue and launching through @p kernels. */
    metal_shared_allocator_t(metal_context_t& context, int device, kernel_tables<metal_module_t>& kernels) noexcept
        : context_(&context), device_(device), kernels_(&kernels) {}

    /** Allocates on the device of an executor's @p stream and commits to its queue; a template only
     *  because that @c stream_t is declared after it. */
    template <typename stream_type_>
    explicit metal_shared_allocator_t(stream_type_ const& stream) noexcept
        : context_(stream.handle.context()), stream_(&stream.handle), device_(static_cast<int>(stream.ordinal)),
          kernels_(stream.kernels) {}

    metal_buffer_t allocate(std::size_t bytes) noexcept {
        return context_ ? metal_allocate(*context_, bytes) : metal_buffer_t {};
    }
    void deallocate(metal_buffer_t& buffer) noexcept { buffer = {}; }
    int device() const noexcept { return device_; }

    /** The context of the allocator's device, or null for a default-built one. */
    metal_context_t* context() const noexcept { return context_; }

    /** The queue work is committed to: the stream's own, or the device's default one. */
    void* queue() const noexcept { return stream_ ? stream_->queue() : context_->queue; }

    /** The pipelines the process built, which launches read; null for a default-built one. */
    kernel_tables<metal_module_t>* kernels() const noexcept { return kernels_; }

  private:
    metal_context_t* context_ = nullptr;
    metal_stream_t const* stream_ = nullptr;
    int device_ = 0;
    kernel_tables<metal_module_t>* kernels_ = nullptr;
};

/** Memory that one call at a time rewinds and refills for one Apple device: the Apple alternative
 *  of @ref gpu_arena_t. It grows Shared @c MTLBuffer chunks on the device, each at least twice the
 *  last, and hands out their contents, which the host and the device both read; it still serves
 *  only the executor that made it, as every vendor's arena does. */
struct metal_device_arena_t {

    /** Grows on the device of the executor's @p stream; a template only because that @c stream_t
     *  is declared after it. */
    template <typename stream_type_>
    explicit metal_device_arena_t(stream_type_ const& stream) noexcept : allocator_(stream) {}

    /** Carves @p count values from the newest chunk, opening a wider one when they do not fit, or
     *  null when the device refuses it. */
    template <typename value_type_>
    value_type_* allocate(std::size_t count) noexcept {
        if (wraps_on_multiply(count, sizeof(value_type_))) return nullptr;
        std::size_t const bytes = count * sizeof(value_type_);
        if (!chunks_.empty()) {
            std::size_t const start = round_up_to_multiple(used_, default_alignment_k);
            if (start <= chunks_.back().bytes && bytes <= chunks_.back().bytes - start)
                return used_ = start + bytes, reinterpret_cast<value_type_*>(chunks_.back().host + start);
        }
        std::size_t const width = chunks_.empty() ? bytes : max_of(bytes, 2 * chunks_.back().bytes);
        metal_buffer_t opened = allocator_.allocate(width);
        if (!opened || failed(chunks_.push_back(std::move(opened)))) return nullptr;
        used_ = bytes;
        return reinterpret_cast<value_type_*>(chunks_.back().host);
    }

    /** Whether @p address lies in memory handed out since the last @ref clear: the older chunks
     *  whole, and the newest up to its last carve. */
    bool contains(void const* address) const noexcept {
        byte_t const* const byte = static_cast<byte_t const*>(address);
        for (std::size_t chunk = 0; chunk != chunks_.size(); ++chunk) {
            std::size_t const handed = chunk + 1 == chunks_.size() ? used_ : chunks_[chunk].bytes;
            if (byte >= chunks_[chunk].host && byte < chunks_[chunk].host + handed) return true;
        }
        return false;
    }

    /** Rewinds the arena, so the previous call's outputs die, keeping only its widest chunk. */
    void clear() noexcept {
        if (chunks_.size() > 1) chunks_[0] = chunks_.pop_back();
        while (chunks_.size() > 1) chunks_.pop_back();
        used_ = 0;
    }

    /** Copies @p bytes from @p source to @p destination in the arena; Shared storage needs no
     *  stream, so the copy is done when it returns. */
    status_t upload(void* destination, void const* source, std::size_t bytes) noexcept {
        std::memcpy(destination, source, bytes);
        return status_code_t::success_k;
    }

    /** Every upload already arrived, so there is nothing to wait for. */
    status_t wait() noexcept { return status_code_t::success_k; }

    int device() const noexcept { return allocator_.device(); }

  private:
    [[no_unique_address]] metal_shared_allocator_t allocator_;

    /** The Shared buffers handed out since the last rewind, oldest first, each twice the last. */
    growing_buffer<metal_buffer_t> chunks_ {global_allocator};

    /** Bytes of the newest chunk handed out, padding included. */
    std::size_t used_ = 0;
};

/** Bytes of threadgroup memory one GPU core keeps to itself on the smallest of @p executor 's
 *  devices, what a round's hubs and neighborhoods are sized by, as a CPU executor's cache; asked of
 *  each device at each call, zero where a stream names none. */
inline std::size_t cache_bytes_per_core(metal_executor_t const& executor) noexcept {
    std::size_t smallest = 0;
    for (std::size_t index = 0; index != executor.size(); ++index) {
        metal_context_t const* const context = executor.stream(index).handle.context();
        if (!context) return 0;
        std::size_t const bytes = metal_send<std::size_t>(*context->api, context->device, "maxThreadgroupMemoryLength");
        smallest = smallest ? min_of(smallest, bytes) : bytes;
    }
    return smallest;
}

/** The tally at @p counter in @p words: its low word joined with the high word it carries into. */
inline u64_t metal_tally(u32_t const* words, u32_t counter) noexcept {
    return words[counter] | u64_t {words[counter + 1]} << 32;
}

/** The counter words @p counters holds, @c metal_counter_words_k of them. */
inline u32_t* metal_words(metal_buffer_t const& counters) noexcept { return reinterpret_cast<u32_t*>(counters.host); }

/** One caller's scratch for driving an Apple index: the walks' arena, the staging of one batch, the
 *  telemetry counters and the single-query staging. Two contexts on one index share only the
 *  storage. */
template <typename buffers_allocator_type_ = metal_shared_allocator_t>
struct metal_traversal_context {

    /** The device, queue and pipeline tables the context submits through, an allocator because the
     *  residency-level harnesses build every context from its @c allocator_t. */
    using allocator_t = metal_shared_allocator_t;

    /** The backend of the arena, the staging and the counters, on the stream's device. */
    using buffers_allocator_t = buffers_allocator_type_;

    /** The single-query verb's staging, grown to the walk's breadth: the ascending matches, then,
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

    /** Binds the context to @p allocator's device and queue; buffers wait for the first verb. */
    metal_traversal_context(allocator_t allocator = {}) noexcept
        : stream_(allocator), context_(allocator.context()), buffers_allocator_(allocator) {
        if (context_) counters_ = buffers_allocator_.allocate(metal_counter_words_k * sizeof(u32_t));
    }
    metal_traversal_context(metal_traversal_context&&) noexcept = default;
    metal_traversal_context(metal_traversal_context const&) = delete;
    metal_traversal_context& operator=(metal_traversal_context const&) = delete;

    int device() const noexcept { return stream_.device(); }
    void* stream() const noexcept { return stream_.queue(); }

    /** The device, queue and pipeline tables this context submits through. */
    allocator_t const& allocator() const noexcept { return stream_; }

    /** The table every launch of this context goes through. */
    metal_api_t const& api() const noexcept { return *context_->api; }
    explicit operator bool() const noexcept { return context_ && counters_; }

    /** Waits for the last command buffer this context committed - one stream's completion, the
     *  executor's @c drain being the fleet's. */
    status_t wait() noexcept { return committed_.wait(); }

    /** Polls whether the last command buffer this context committed finished without blocking, so
     *  one host thread awaits whichever of several streams finishes first. */
    status_t finished() const noexcept { return committed_.finished(); }

    /** The telemetry of the last submission, zeroed when it was staged; read once it finished. */
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
        block.sketched = metal_tally(words, metal_sketched_k);
        return block;
    }

    /** Whether the last walk hit a value-lookup miss, a node its value lookup had no row for, which
     *  refuses it; read once it finished. */
    bool missed() const noexcept {
        u32_t const* const words = metal_words(counters_);
        return words[metal_missed_k] != 0;
    }

    /** The sum a count-sum launch on this context wrote, such as the matches a batch found; read
     *  once it finished. */
    std::size_t matched() const noexcept {
        u32_t const* const words = metal_words(counters_);
        return std::size_t {words[metal_matched_k]} | std::size_t {words[metal_matched_k + 1]} << 32;
    }

    /** Where the last walk left its outputs in the staging, read back once the context finished. */
    struct walk_output_offsets_t {
        std::size_t queries = 0, count = 0;
        std::size_t first_offset = 0, second_offset = 0, third_offset = 0;
    };

    allocator_t stream_;
    metal_context_t* context_ = nullptr;
    buffers_allocator_t buffers_allocator_;
    metal_buffer_t counters_;
    metal_buffer_t arena_;
    metal_buffer_t staging_;
    walk_output_offsets_t walk_output_offsets_;

    /** The columns a health check and a round keep on the device, grown like the staging. */
    metal_buffer_t round_;

    /** The last command buffer submitted through this context; its stream runs them in commit
     *  order, so this one finishing means every earlier one did. */
    metal_launch_t committed_;

    /** Monotone epoch floor over the arena's visited slots, so the arena zeroes only when fresh or
     *  wrapped. */
    u64_t epoch_floor_ = 0;

    /** Dispatches @p launch over @p groups threadgroups of @p threads and keeps it as the command
     *  buffer the context awaits. */
    status_t submit_(metal_launch_t& launch, std::size_t groups, std::size_t threads) noexcept {
        if (status_t const submitted = launch.submit(groups, threads); failed(submitted)) return submitted;
        committed_ = std::move(launch);
        return status_code_t::success_k;
    }

    /** Grows @p buffer to at least @p bytes; answers whether it was reallocated, zeroed by
     *  Metal. */
    status_t ensure_(metal_buffer_t& buffer, std::size_t bytes, bool& fresh) noexcept {
        fresh = false;
        if (bytes <= buffer.bytes) return status_code_t::success_k;
        buffer = buffers_allocator_.allocate(bytes);
        fresh = true;
        return buffer ? status_code_t::success_k : status_code_t::out_of_memory_k;
    }
};

} // namespace unum::usearch::utilities
