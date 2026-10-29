/**
 *  @file include/usearch/utilities/vram_nvidia_device.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Host-side vocabulary for driving one NVIDIA GPU - the driver, specs, streams, memory.
 *
 *  The library never links @c libcuda or the CUDA runtime:
 *  @ref unum::usearch::utilities::cuda_api_t opens the driver and resolves the entry points
 *  USearch calls into one table, which stays empty where the driver is absent, so NVIDIA then
 *  lists no devices and the binary still loads. Its caller loads it once and hands it to every
 *  executor, allocator and stream it builds. Kernels reach a device as fatbins compiled ahead of
 *  time and embedded in the library, loaded by @ref unum::usearch::utilities::cuda_module_t and
 *  launched by name, since NVCC's fatbin registration would link the runtime outright.
 *
 *  Plain C++ needing no CUDA SDK, hence the @c .hpp extension: the few driver types, constants and
 *  prototypes USearch calls are spelled here, copied from the driver's stable ABI, so any host
 *  compiler builds it. The warp policy the shared walk code is written against lives in
 *  @c vram_nvidia_device.cuh. The flat @c cuda_allocator carries @c std::allocator's call surface
 *  with the stream bound at construction; the @c cuda_reservable_allocator is the flat storage
 *  backend, a reserve-once / commit-on-demand range whose base survives growth. Nothing models
 *  both.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` for device-pointer casts

#include <atomic>      // `std::memory_order` - the committed extent, `std::atomic` - the primary contexts
#include <bitset>      // `std::bitset` - the devices a range maps its slabs for
#include <memory>      // `std::unique_ptr` for owned device allocations
#include <mutex>       // `std::unique_lock`, `std::adopt_lock` - the executor's one-caller hold
#include <span>        // `std::span` - the device list an executor opens over, a fatbin's bytes
#include <type_traits> // `std::is_trivially_copyable_v` - a launch's arguments travel as bytes
#include <utility>     // `std::move`, `std::exchange`, `std::swap`

#include <forkunion/atomics.hpp>  // `fu::standard_atomic_ref` - the mapped and committed byte counts
#include <forkunion/topology.hpp> // `fu::volume_ram` - the managed storage's host-memory budget
#include <forkunion/types.hpp>    // `fu::round_up_to_multiple`, `fu::spin_mutex_t` - the hold

#include "usearch/utilities/blocking_files.hpp" // `shared_library_t` - the driver opened at load
#include "usearch/utilities/ram_buffers.hpp"    // `growing_buffer` - an executor's table of contexts
#include "usearch/utilities/types.hpp"          // `gpu_device_t`, `gpu_devices_capacity_k`

namespace unum::usearch::utilities {

#pragma region Driver ABI

/** The driver's status word, @c CUresult, naming the few values USearch tells apart; any other is
 *  a failure. */
enum cuda_result_t : int {
    cuda_success_k = 0,
    cuda_error_not_initialized_k = 3,
    cuda_error_invalid_device_k = 101,
    cuda_error_not_ready_k = 600,
};

/** A device address, @c CUdeviceptr, which unified addressing shares with the host. */
using cuda_device_address_t = unsigned long long;

/** The driver's opaque handles: @c CUcontext, @c CUstream, @c CUevent, @c CUmodule and
 *  @c CUfunction. */
using cuda_context_handle_t = struct cuda_context_opaque_t*;
using cuda_stream_handle_t = struct cuda_stream_opaque_t*;
using cuda_event_handle_t = struct cuda_event_opaque_t*;
using cuda_module_handle_t = struct cuda_module_opaque_t*;
using cuda_function_handle_t = struct cuda_function_opaque_t*;

/** The @c CUdevice_attribute values USearch queries. */
enum cuda_device_attribute_t : int {
    cuda_attribute_warp_size_k = 10,
    cuda_attribute_multiprocessor_count_k = 16,
    cuda_attribute_compute_capability_major_k = 75,
    cuda_attribute_compute_capability_minor_k = 76,
    cuda_attribute_shared_memory_per_multiprocessor_k = 81,
    cuda_attribute_shared_memory_per_block_optin_k = 97,
    cuda_attribute_reserved_shared_memory_per_block_k = 111,
};

/** The @c CUfunction_attribute USearch reads and raises, the dynamic shared memory a launch may ask
 *  for. */
inline constexpr int cuda_function_dynamic_shared_bytes_k = 8;

/** The @c CUmem_advise values USearch gives managed memory. */
enum cuda_memory_advice_t : int {
    cuda_advise_read_mostly_k = 1,
    cuda_advise_preferred_location_k = 3,
    cuda_advise_accessed_by_k = 5,
};

/** Where memory lives, @c CUmemLocation: a device ordinal, or the host. */
struct cuda_memory_location_t {
    inline static constexpr int device_k = 1, host_k = 2;
    int type = 0;
    int id = 0;
};

/** How @c cuMemCreate backs one physical slab, @c CUmemAllocationProp. */
struct cuda_allocation_properties_t {
    inline static constexpr int pinned_k = 1;
    int type = 0;
    int requested_handle_types = 0;
    cuda_memory_location_t location;
    void* win32_handle_metadata = nullptr;
    struct allocation_flags_t {
        unsigned char compression_type = 0;
        unsigned char gpu_direct_rdma_capable = 0;
        unsigned short usage = 0;
        unsigned char reserved[4] = {};
    };
    allocation_flags_t allocation_flags;
};

/** Who may touch a mapped range and how, @c CUmemAccessDesc. */
struct cuda_access_descriptor_t {
    inline static constexpr int read_write_k = 3;
    cuda_memory_location_t location;
    int flags = 0;
};

/** One pitched copy, @c CUDA_MEMCPY2D; USearch names both sides by unified address. */
struct cuda_copy_2d_t {
    inline static constexpr int unified_k = 4;
    std::size_t source_x_bytes = 0, source_y = 0;
    int source_memory_type = 0;
    void const* source_host = nullptr;
    cuda_device_address_t source_device = 0;
    void* source_array = nullptr;
    std::size_t source_pitch = 0;
    std::size_t destination_x_bytes = 0, destination_y = 0;
    int destination_memory_type = 0;
    void* destination_host = nullptr;
    cuda_device_address_t destination_device = 0;
    void* destination_array = nullptr;
    std::size_t destination_pitch = 0;
    std::size_t width_bytes = 0, height = 0;
};

/** One launch attribute, @c CUlaunchAttribute: its ID, then the driver's 64-byte value union. */
struct cuda_launch_attribute_t {
    int id = 0;
    alignas(8) int value[16] = {};
};

/** @c CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: a kernel that waits on its predecessor
 *  at entry may start while that predecessor drains. */
inline constexpr int cuda_launch_programmatic_serialization_k = 6;

/** One launch's geometry, @c CUlaunchConfig, and the attributes it carries. */
struct cuda_launch_config_t {
    unsigned grid_x = 1, grid_y = 1, grid_z = 1;
    unsigned block_x = 1, block_y = 1, block_z = 1;
    unsigned shared_bytes = 0;
    cuda_stream_handle_t stream = nullptr;
    cuda_launch_attribute_t* attributes = nullptr;
    unsigned attributes_count = 0;
};

/** The driver's @c CUtensorMap: the opaque descriptor of one tensor in global memory that the
 *  Tensor Memory Accelerator copies through, read by a kernel from its parameters. */
struct alignas(64) cuda_tensor_map_t {
    u64_t opaque[16] = {};
};

/** The @c CUtensorMapDataType USearch counts rows in, @c CU_TENSOR_MAP_DATA_TYPE_UINT64. */
inline constexpr int cuda_tensor_map_u64_k = 4;

/** The flags USearch passes: @c CU_STREAM_NON_BLOCKING, @c CU_EVENT_BLOCKING_SYNC,
 *  @c CU_EVENT_DISABLE_TIMING, @c CU_MEMHOSTALLOC_PORTABLE, which @c CU_MEMHOSTREGISTER_PORTABLE
 *  equals, @c CU_MEMHOSTREGISTER_READ_ONLY, @c CU_MEM_ATTACH_GLOBAL and
 *  @c CU_MEM_ALLOC_GRANULARITY_MINIMUM. */
inline constexpr unsigned cuda_stream_non_blocking_k = 1;
inline constexpr unsigned cuda_event_blocking_sync_k = 1;
inline constexpr unsigned cuda_event_disable_timing_k = 2;
inline constexpr unsigned cuda_host_portable_k = 1;
inline constexpr unsigned cuda_host_read_only_k = 8;
inline constexpr unsigned cuda_attach_global_k = 1;
inline constexpr int cuda_granularity_minimum_k = 0;
inline constexpr int cuda_pointer_memory_type_k = 2;
inline constexpr int cuda_pointer_device_ordinal_k = 9;
inline constexpr unsigned cuda_memory_type_device_k = 2;

#pragma endregion Driver ABI

/**
 *  @brief The driver entry points USearch calls, resolved from @c libcuda once per process.
 *
 *  Every entry is null when the driver is absent or lacks any of them, which leaves NVIDIA with no
 *  devices; each spells the prototype of the exported symbol it is resolved by, the @c _v2 one
 *  where @c <cuda.h> maps the name onto it.
 */
struct cuda_api_t {
    cuda_result_t (*init)(unsigned) = nullptr;
    cuda_result_t (*get_device_count)(int*) = nullptr;
    cuda_result_t (*get_device)(int*, int) = nullptr;
    cuda_result_t (*get_device_attribute)(int*, cuda_device_attribute_t, int) = nullptr;
    cuda_result_t (*get_device_name)(char*, int, int) = nullptr;
    cuda_result_t (*get_device_memory)(std::size_t*, int) = nullptr;
    cuda_result_t (*can_access_peer)(int*, int, int) = nullptr;
    cuda_result_t (*retain_primary_context)(cuda_context_handle_t*, int) = nullptr;
    cuda_result_t (*set_current_context)(cuda_context_handle_t) = nullptr;
    cuda_result_t (*allocate_async)(cuda_device_address_t*, std::size_t, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*free_async)(cuda_device_address_t, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*allocate_managed)(cuda_device_address_t*, std::size_t, unsigned) = nullptr;
    cuda_result_t (*memory_advise)(cuda_device_address_t, std::size_t, cuda_memory_advice_t,
                                   cuda_memory_location_t) = nullptr;
    cuda_result_t (*free)(cuda_device_address_t) = nullptr;
    cuda_result_t (*allocate_host)(void**, std::size_t, unsigned) = nullptr;
    cuda_result_t (*free_host)(void*) = nullptr;
    cuda_result_t (*register_host)(void const*, std::size_t, unsigned) = nullptr;
    cuda_result_t (*unregister_host)(void const*) = nullptr;
    cuda_result_t (*copy)(cuda_device_address_t, cuda_device_address_t, std::size_t) = nullptr;
    cuda_result_t (*copy_async)(cuda_device_address_t, cuda_device_address_t, std::size_t,
                                cuda_stream_handle_t) = nullptr;
    cuda_result_t (*copy_2d_async)(cuda_copy_2d_t const*, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*fill)(cuda_device_address_t, unsigned char, std::size_t) = nullptr;
    cuda_result_t (*fill_async)(cuda_device_address_t, unsigned char, std::size_t, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*fill_words_async)(cuda_device_address_t, unsigned, std::size_t, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*stream_create)(cuda_stream_handle_t*, unsigned) = nullptr;
    cuda_result_t (*stream_destroy)(cuda_stream_handle_t) = nullptr;
    cuda_result_t (*stream_synchronize)(cuda_stream_handle_t) = nullptr;
    cuda_result_t (*stream_query)(cuda_stream_handle_t) = nullptr;
    cuda_result_t (*pointer_attribute)(void*, int, cuda_device_address_t) = nullptr;
    cuda_result_t (*stream_wait_event)(cuda_stream_handle_t, cuda_event_handle_t, unsigned) = nullptr;
    cuda_result_t (*event_create)(cuda_event_handle_t*, unsigned) = nullptr;
    cuda_result_t (*event_record)(cuda_event_handle_t, cuda_stream_handle_t) = nullptr;
    cuda_result_t (*event_query)(cuda_event_handle_t) = nullptr;
    cuda_result_t (*event_synchronize)(cuda_event_handle_t) = nullptr;
    cuda_result_t (*event_destroy)(cuda_event_handle_t) = nullptr;
    cuda_result_t (*module_load_data)(cuda_module_handle_t*, void const*) = nullptr;
    cuda_result_t (*module_unload)(cuda_module_handle_t) = nullptr;
    cuda_result_t (*module_get_function)(cuda_function_handle_t*, cuda_module_handle_t, char const*) = nullptr;
    cuda_result_t (*function_get_attribute)(int*, int, cuda_function_handle_t) = nullptr;
    cuda_result_t (*function_set_attribute)(cuda_function_handle_t, int, int) = nullptr;
    cuda_result_t (*launch_kernel)(cuda_launch_config_t const*, cuda_function_handle_t, void**, void**) = nullptr;
    cuda_result_t (*memory_granularity)(std::size_t*, cuda_allocation_properties_t const*, int) = nullptr;
    cuda_result_t (*memory_address_reserve)(cuda_device_address_t*, std::size_t, std::size_t, cuda_device_address_t,
                                            unsigned long long) = nullptr;
    cuda_result_t (*memory_address_free)(cuda_device_address_t, std::size_t) = nullptr;
    cuda_result_t (*memory_create)(unsigned long long*, std::size_t, cuda_allocation_properties_t const*,
                                   unsigned long long) = nullptr;
    cuda_result_t (*memory_release)(unsigned long long) = nullptr;
    cuda_result_t (*memory_map)(cuda_device_address_t, std::size_t, std::size_t, unsigned long long,
                                unsigned long long) = nullptr;
    cuda_result_t (*memory_unmap)(cuda_device_address_t, std::size_t) = nullptr;
    cuda_result_t (*memory_set_access)(cuda_device_address_t, std::size_t, cuda_access_descriptor_t const*,
                                       std::size_t) = nullptr;

    /** The one entry only the Blackwell row gathers use, resolved apart from the rest, so a driver
     *  lacking it keeps every device and loses only the gathers. */
    cuda_result_t (*encode_tensor_map_tiled)(cuda_tensor_map_t*, int, u32_t, void*, u64_t const*, u64_t const*,
                                             u32_t const*, u32_t const*, int, int, int, int) = nullptr;

    /** Whether the driver loaded and initialized with every entry resolved. */
    explicit operator bool() const noexcept { return memory_set_access != nullptr; }

    /** Opens the driver, resolves every entry and initializes it, or answers the empty table. The
     *  library stays open for the life of the process, as the table points into it. */
    static cuda_api_t load() noexcept {
        // Each system names the driver its own way, and opening another's name merely fails.
        shared_library_t library;
        for (char const* name : {"libcuda.so.1", "nvcuda.dll"})
            if ((library = shared_library_t::open(name))) break;
        if (!library) return {};
        cuda_api_t api;
        bool const resolved = //
            library.resolve("cuInit", api.init) && library.resolve("cuDeviceGetCount", api.get_device_count) &&
            library.resolve("cuDeviceGet", api.get_device) &&
            library.resolve("cuDeviceGetAttribute", api.get_device_attribute) &&
            library.resolve("cuDeviceGetName", api.get_device_name) &&
            library.resolve("cuDeviceTotalMem_v2", api.get_device_memory) &&
            library.resolve("cuDeviceCanAccessPeer", api.can_access_peer) &&
            library.resolve("cuDevicePrimaryCtxRetain", api.retain_primary_context) &&
            library.resolve("cuCtxSetCurrent", api.set_current_context) &&
            library.resolve("cuMemAllocAsync", api.allocate_async) &&
            library.resolve("cuMemFreeAsync", api.free_async) &&
            library.resolve("cuMemAllocManaged", api.allocate_managed) &&
            library.resolve("cuMemAdvise_v2", api.memory_advise) && library.resolve("cuMemFree_v2", api.free) &&
            library.resolve("cuMemHostAlloc", api.allocate_host) && library.resolve("cuMemFreeHost", api.free_host) &&
            library.resolve("cuMemHostRegister_v2", api.register_host) &&
            library.resolve("cuMemHostUnregister", api.unregister_host) && library.resolve("cuMemcpy", api.copy) &&
            library.resolve("cuMemcpyAsync", api.copy_async) &&
            library.resolve("cuMemcpy2DAsync_v2", api.copy_2d_async) && library.resolve("cuMemsetD8_v2", api.fill) &&
            library.resolve("cuMemsetD8Async", api.fill_async) &&
            library.resolve("cuMemsetD32Async", api.fill_words_async) &&
            library.resolve("cuStreamCreate", api.stream_create) &&
            library.resolve("cuStreamDestroy_v2", api.stream_destroy) &&
            library.resolve("cuStreamSynchronize", api.stream_synchronize) &&
            library.resolve("cuStreamQuery", api.stream_query) &&
            library.resolve("cuPointerGetAttribute", api.pointer_attribute) &&
            library.resolve("cuStreamWaitEvent", api.stream_wait_event) &&
            library.resolve("cuEventCreate", api.event_create) && library.resolve("cuEventRecord", api.event_record) &&
            library.resolve("cuEventQuery", api.event_query) &&
            library.resolve("cuEventSynchronize", api.event_synchronize) &&
            library.resolve("cuEventDestroy_v2", api.event_destroy) &&
            library.resolve("cuModuleLoadData", api.module_load_data) &&
            library.resolve("cuModuleUnload", api.module_unload) &&
            library.resolve("cuModuleGetFunction", api.module_get_function) &&
            library.resolve("cuFuncGetAttribute", api.function_get_attribute) &&
            library.resolve("cuFuncSetAttribute", api.function_set_attribute) &&
            library.resolve("cuLaunchKernelEx", api.launch_kernel) &&
            library.resolve("cuMemGetAllocationGranularity", api.memory_granularity) &&
            library.resolve("cuMemAddressReserve", api.memory_address_reserve) &&
            library.resolve("cuMemAddressFree", api.memory_address_free) &&
            library.resolve("cuMemCreate", api.memory_create) && library.resolve("cuMemRelease", api.memory_release) &&
            library.resolve("cuMemMap", api.memory_map) && library.resolve("cuMemUnmap", api.memory_unmap) &&
            library.resolve("cuMemSetAccess", api.memory_set_access);
        library.resolve("cuTensorMapEncodeTiled", api.encode_tensor_map_tiled);
        // A driver without a usable GPU fails its initialization, which lists no devices either.
        if (resolved && api.init(0) == cuda_success_k) return api;
        return {};
    }
};

/** Makes the primary context of device @p ordinal current on the calling thread, retained once per
 *  process: the driver twin of selecting a device, which every stream, allocation and launch after
 *  it lands on. Asked on every entry, so a caller switching devices between USearch calls is
 *  rebound, and finds the last device USearch bound left current. */
inline cuda_result_t cuda_set_device(cuda_api_t const& api, int ordinal) noexcept {
    static std::atomic<cuda_context_handle_t> primary_contexts[gpu_devices_capacity_k] = {};
    if (!api) return cuda_error_not_initialized_k;
    if (ordinal < 0 || static_cast<std::size_t>(ordinal) >= gpu_devices_capacity_k) return cuda_error_invalid_device_k;
    cuda_context_handle_t context = primary_contexts[ordinal].load(std::memory_order_acquire);
    if (!context) {
        // Two racing threads may both retain it, which only raises a count nothing ever releases.
        int device = 0;
        if (cuda_result_t const found = api.get_device(&device, ordinal); found != cuda_success_k) return found;
        if (cuda_result_t const retained = api.retain_primary_context(&context, device); retained != cuda_success_k)
            return retained;
        primary_contexts[ordinal].store(context, std::memory_order_release);
    }
    return api.set_current_context(context);
}

/** The driver's integer spelling of a pointer, which unified addressing shares with the host. */
inline cuda_device_address_t cuda_address(void const* pointer) noexcept {
    return static_cast<cuda_device_address_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

/** Copies @p bytes from @p source to @p destination and waits, any direction unified addressing
 *  tells apart. */
inline cuda_result_t cuda_copy(cuda_api_t const& api, void* destination, void const* source,
                               std::size_t bytes) noexcept {
    return api ? api.copy(cuda_address(destination), cuda_address(source), bytes) : cuda_error_not_initialized_k;
}

/** The ordinal of the device whose memory holds @p pointer, or -1 for host memory, pinned or not,
 *  which unified addressing tells apart through the driver. */
inline int cuda_device_of(cuda_api_t const& api, void const* pointer) noexcept {
    unsigned memory_type = 0;
    int ordinal = -1;
    if (!api || !pointer ||
        api.pointer_attribute(&memory_type, cuda_pointer_memory_type_k, cuda_address(pointer)) != cuda_success_k ||
        memory_type != cuda_memory_type_device_k ||
        api.pointer_attribute(&ordinal, cuda_pointer_device_ordinal_k, cuda_address(pointer)) != cuda_success_k)
        return -1;
    return ordinal;
}

/** Queues the copy of @p bytes from @p source to @p destination on @p stream. */
inline cuda_result_t cuda_copy_async(cuda_api_t const& api, void* destination, void const* source, std::size_t bytes,
                                     cuda_stream_handle_t stream) noexcept {
    return api ? api.copy_async(cuda_address(destination), cuda_address(source), bytes, stream)
               : cuda_error_not_initialized_k;
}

/** Queues the copy of @p height rows of @p width bytes on @p stream, each row starting a pitch
 *  past the last on either side. */
inline cuda_result_t cuda_copy_2d_async(cuda_api_t const& api, void* destination, std::size_t destination_pitch,
                                        void const* source, std::size_t source_pitch, std::size_t width,
                                        std::size_t height, cuda_stream_handle_t stream) noexcept {
    if (!api) return cuda_error_not_initialized_k;
    cuda_copy_2d_t copy;
    copy.source_memory_type = cuda_copy_2d_t::unified_k;
    copy.source_device = cuda_address(source);
    copy.source_pitch = source_pitch;
    copy.destination_memory_type = cuda_copy_2d_t::unified_k;
    copy.destination_device = cuda_address(destination);
    copy.destination_pitch = destination_pitch;
    copy.width_bytes = width;
    copy.height = height;
    return api.copy_2d_async(&copy, stream);
}

/** Queues setting @p bytes at @p destination to @p value on @p stream. */
inline cuda_result_t cuda_fill_async(cuda_api_t const& api, void* destination, u8_t value, std::size_t bytes,
                                     cuda_stream_handle_t stream) noexcept {
    return api ? api.fill_async(cuda_address(destination), value, bytes, stream) : cuda_error_not_initialized_k;
}

/**
 *  @brief GPU generations, one compile target each, named like NumKong's GPU tiers and numbered on
 *      the @c __CUDA_ARCH__ scale, so tiers order against each other.
 *
 *  NVCC compiles three kinds of target, and none of them runs on a GPU older than it names. A plain
 *  one, like @c 80 or @c 89, runs on its own compute capability and every later one of the same
 *  major - an 8.0 cubin runs on 8.6 and 8.9 - and the PTX it embeds is JIT-compiled for any newer
 *  GPU, which makes it the only forward-compatible kind. An @c a target, like @c 90a, adds what
 *  only that exact part keeps, like @c wgmma, so its cubin runs on 9.0 alone and its PTX nowhere
 *  else. An @c f target, from CUDA 12.9 and compute capability 10 on, adds what a whole family
 *  shares: @c 100f runs on 10.0 and 10.3, @c 120f on 12.0 and 12.1, and neither on any other
 *  family. A GPU newer than the newest tier therefore runs the Ampere unit's PTX until a tier of
 *  its own lands.
 */
enum cuda_arch_t : int {

    /** @c 80, the floor - A100, A10 & RTX 30: @c dp4a, packed-half & bf16x2 fma, @c cp.async. */
    cuda_arch_ampere_k = 800,

    /** @c 89 - L4, L40S & RTX 40: the fp8 conversions and FP8 @c mma.sync. */
    cuda_arch_ada_k = 890,

    /** @c 90a - H100 & H200: TMA, thread-block clusters, distributed shared memory, and
     *  @c wgmma. */
    cuda_arch_hopper_k = 900,

    /** @c 100f - B200, B300 & GB300: mixed-precision @c fma.f32.f16 & @c fma.f32.bf16, 256-bit
     *  loads, TMA @c gather4, the fp6 & fp4 conversions, and @c tcgen05. */
    cuda_arch_blackwell_k = 1000,

    /** @c 120f - RTX 50, RTX PRO 6000 & DGX Spark: Blackwell's multiply-adds & conversions, without
     *  @c tcgen05. */
    cuda_arch_blackwellrtx_k = 1200,
};

/** Compute capability @p major. @p minor on the @c cuda_arch_t scale, as the driver reports it. */
constexpr cuda_arch_t cuda_arch_of(u32_t major, u32_t minor) noexcept {
    return static_cast<cuda_arch_t>(major * 100 + minor * 10);
}

/** The compute capabilities one class's cubins run on, both ends included: each class lists the
 *  family its @c f or @c a target covers, never a whole major version. */
struct nvidia_class_span_t {
    gpu_class_t gpu_class;
    cuda_arch_t first;
    cuda_arch_t last;
};

/** Every NVIDIA class and the GPUs its unit runs natively; a new family is a new row. */
inline constexpr nvidia_class_span_t nvidia_class_spans_k[] = {
    {gpu_class_t::nvidia_ampere_k, cuda_arch_ampere_k, static_cast<cuda_arch_t>(870)},
    {gpu_class_t::nvidia_ada_k, cuda_arch_ada_k, cuda_arch_ada_k},
    {gpu_class_t::nvidia_hopper_k, cuda_arch_hopper_k, cuda_arch_hopper_k},
    {gpu_class_t::nvidia_blackwell_k, cuda_arch_blackwell_k, static_cast<cuda_arch_t>(1030)},
    {gpu_class_t::nvidia_blackwellrtx_k, cuda_arch_blackwellrtx_k, static_cast<cuda_arch_t>(1210)},
};

/** The class whose unit carries cubins for compute capability @p arch: the row listing it, else the
 *  Ampere unit, whose PTX JIT-compiles for any newer GPU, and none older than Ampere. */
constexpr gpu_class_t nvidia_class_of(cuda_arch_t arch) noexcept {
    if (arch < cuda_arch_ampere_k) return gpu_class_t::unknown_k;
    for (nvidia_class_span_t const& span : nvidia_class_spans_k)
        if (span.first <= arch && arch <= span.last) return span.gpu_class;
    return gpu_class_t::nvidia_ampere_k;
}

/** One GPU's hardware specification - the facts runtime dispatch consults; hardware only,
 *  never derived policy. */
struct cuda_device_specs_t {

    /** The device ordinal every driver call routes through. */
    int ordinal = 0;

    /** Streaming-multiprocessor count - the launch-grid currency. */
    u32_t multiprocessor_count = 0;

    /** Compute-capability major version - the hardware generation, e.g. 9 = Hopper. */
    int compute_major = 0;

    /** Compute-capability minor version - the in-generation revision. */
    int compute_minor = 0;

    /** Lanes of one warp, as the driver reports them. */
    u32_t warp_lanes = 0;

    /** Total device-resident memory in bytes. */
    std::size_t vram_bytes = 0;

    /** Widest dynamic shared memory one block can be granted at launch, on explicit request. */
    std::size_t shared_memory_per_block_bytes = 0;

    /** Shared memory one multiprocessor splits between its resident blocks. */
    std::size_t shared_memory_per_multiprocessor_bytes = 0;

    /** Shared memory the driver keeps from every resident block beside what it asks for. */
    std::size_t reserved_shared_memory_per_block_bytes = 0;

    /** This device's generation on the @c cuda_arch_t scale - the kit chains' runtime ceiling. */
    cuda_arch_t arch() const noexcept { return cuda_arch_of(compute_major, compute_minor); }

    /**
     *  Queries the driver for @p ordinal's specification; meaningful only on @c success_k.
     *
     *  @par Statuses:
     *  - @c device_error_k The @p ordinal is past the device listing.
     */
    static expected<cuda_device_specs_t> query(cuda_api_t const& api, int ordinal) noexcept {
        int device = 0;
        cuda_device_specs_t specs {ordinal};
        int multiprocessors = 0, warp_lanes = 0, shared_memory_per_block = 0, shared_memory_per_multiprocessor = 0,
            reserved_shared_memory_per_block = 0;
        if (!api || api.get_device(&device, ordinal) != cuda_success_k ||
            api.get_device_attribute(&multiprocessors, cuda_attribute_multiprocessor_count_k, device) !=
                cuda_success_k ||
            api.get_device_attribute(&warp_lanes, cuda_attribute_warp_size_k, device) != cuda_success_k ||
            api.get_device_attribute(&specs.compute_major, cuda_attribute_compute_capability_major_k, device) !=
                cuda_success_k ||
            api.get_device_attribute(&specs.compute_minor, cuda_attribute_compute_capability_minor_k, device) !=
                cuda_success_k ||
            api.get_device_attribute(&shared_memory_per_block, cuda_attribute_shared_memory_per_block_optin_k,
                                     device) != cuda_success_k ||
            api.get_device_attribute(&shared_memory_per_multiprocessor,
                                     cuda_attribute_shared_memory_per_multiprocessor_k, device) != cuda_success_k ||
            api.get_device_attribute(&reserved_shared_memory_per_block,
                                     cuda_attribute_reserved_shared_memory_per_block_k, device) != cuda_success_k ||
            api.get_device_memory(&specs.vram_bytes, device) != cuda_success_k)
            return {{ordinal}, status_t::device_error_k};
        specs.multiprocessor_count = static_cast<u32_t>(multiprocessors);
        specs.warp_lanes = static_cast<u32_t>(warp_lanes);
        specs.shared_memory_per_block_bytes = static_cast<std::size_t>(shared_memory_per_block);
        specs.shared_memory_per_multiprocessor_bytes = static_cast<std::size_t>(shared_memory_per_multiprocessor);
        specs.reserved_shared_memory_per_block_bytes = static_cast<std::size_t>(reserved_shared_memory_per_block);
        return {specs, status_t::success_k};
    }
};

/** The address space an allocation lives in, in LLVM's sense: who can dereference the pointer and
 *  what transfer path moves its bytes; all three are @c global to device code. */
enum class cuda_address_space_t : u32_t {

    /** Plain device memory; explicit host-device copies required. */
    device_k,

    /** Unified memory; migratable, host- and device-visible. */
    managed_k,

    /** Page-locked host memory; the DMA source & sink for staging. */
    host_pinned_k,
};

/** The device analog of @c std::allocator, with the ordering stream and the target device
 *  bound at construction; @c address_space_k fixes host visibility. */
template <typename allocator_>
concept cuda_allocator = requires(allocator_ allocator, typename allocator_::value_type* pointer, std::size_t count) {
    typename allocator_::value_type;
    { allocator_::address_space_k } -> std::convertible_to<cuda_address_space_t>;
    { allocator.allocate(count) } -> std::same_as<typename allocator_::value_type*>;
    { allocator.deallocate(pointer, count) } -> std::same_as<void>;
    { allocator.stream() } -> std::convertible_to<cuda_stream_handle_t>;
    { allocator.device() } -> std::convertible_to<int>;
    { allocator.api() } -> std::same_as<cuda_api_t const&>;
};

/** The flat storage backend contract: a reserve-once / commit-on-demand address range whose base
 *  never moves. Not a refinement of @c cuda_allocator, since flat storage never frees mid-life. */
template <typename allocator_>
concept cuda_reservable_allocator = requires(allocator_ allocator, std::size_t bytes) {
    typename allocator_::value_type;
    { allocator_::address_space_k } -> std::convertible_to<cuda_address_space_t>;
    { allocator.reserve_range(bytes) } -> std::same_as<status_t>;
    { allocator.commit(bytes) } -> std::same_as<status_t>;
    { allocator.base() } -> std::same_as<typename allocator_::value_type*>;
    { allocator.reserved_bytes() } -> std::same_as<std::size_t>;
    { allocator.committed_bytes() } -> std::same_as<std::size_t>;
    { allocator.release() } -> std::same_as<void>;
    { allocator.device() } -> std::convertible_to<int>;
    { allocator.api() } -> std::same_as<cuda_api_t const&>;
};

/** Virtual-memory flat storage backend: a fixed driver-API address reservation backed by slabs
 *  mapped on demand, host-driven ahead of any kernel; fresh slabs arrive zeroed. */
template <typename value_type_ = byte_t>
struct cuda_virtual_memory_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = cuda_virtual_memory_allocator<other_>;
    };

    /** Largest physical slab, rounded to the driver's granularity at reservation; smaller ones
     *  double up to it, so a sparse shard maps one granularity, not one full slab per column. */
    inline static constexpr std::size_t preferred_slab_bytes_k = 128u << 20;

    cuda_api_t const* api_ = nullptr;
    cuda_device_address_t base_ = 0;
    std::size_t virtual_address_bytes_ = 0;

    /** Bytes mapped so far - the one grower advances it with a release, and any thread may read it
     *  through @c committed_bytes with an acquire; @c mutable for the reference. */
    std::size_t mapped_bytes_ = 0;
    std::size_t granularity_ = 0;
    std::size_t slab_bytes_ = 0;
    int device_ = 0;

    /** CUDA ordinals besides the owner that every slab maps for, each able to reach the owner. */
    std::bitset<gpu_devices_capacity_k> peers_;

    cuda_virtual_memory_allocator() = default;

    /** Names the GPU the future reservation pins to, through @p api - placement only, no driver
     *  call. */
    explicit cuda_virtual_memory_allocator(cuda_api_t const& api, int device) noexcept : api_(&api), device_(device) {}

    /** Copies the placement only, never a live reservation. */
    cuda_virtual_memory_allocator(cuda_virtual_memory_allocator const& other) noexcept
        : api_(other.api_), device_(other.device_), peers_(other.peers_) {}

    /** Takes over a live reservation, so a table of them relocates without remapping. */
    cuda_virtual_memory_allocator(cuda_virtual_memory_allocator&& other) noexcept
        : api_(other.api_), base_(std::exchange(other.base_, 0)),
          virtual_address_bytes_(std::exchange(other.virtual_address_bytes_, 0)),
          mapped_bytes_(std::exchange(other.mapped_bytes_, 0)), granularity_(other.granularity_),
          slab_bytes_(other.slab_bytes_), device_(other.device_), peers_(other.peers_) {}
    template <typename other_>
    cuda_virtual_memory_allocator(cuda_virtual_memory_allocator<other_> const& other) noexcept
        : api_(other.api_), device_(other.device_), peers_(other.peers_) {}
    cuda_virtual_memory_allocator& operator=(cuda_virtual_memory_allocator const&) = delete;
    ~cuda_virtual_memory_allocator() noexcept { release(); }

    /** Reserves @p virtual_address_bytes of virtual address space on the device named at
     *  construction - no physical memory yet, so the ceiling costs nothing and should be
     *  generous. */
    status_t reserve_range(std::size_t virtual_address_bytes) noexcept {
        if (!api_ || !*api_) return status_t::device_error_k;
        cuda_api_t const& api = *api_;
        cuda_allocation_properties_t properties = allocation_properties_(device_);
        if (api.memory_granularity(&granularity_, &properties, cuda_granularity_minimum_k) != cuda_success_k)
            return status_t::device_error_k;
        slab_bytes_ = max_of(granularity_, preferred_slab_bytes_k / granularity_ * granularity_);
        virtual_address_bytes_ = fu::round_up_to_multiple(virtual_address_bytes, slab_bytes_);
        if (api.memory_address_reserve(&base_, virtual_address_bytes_, 0, 0, 0) != cuda_success_k)
            return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /** Maps every slab for those of @p devices that can reach the owner too, so their kernels and
     *  copies read and write it directly; placement only, named before the first @ref commit. */
    void share_with(std::bitset<gpu_devices_capacity_k> devices) noexcept { peers_ = reachable_(devices); }

    /** Whether kernels and copies on CUDA ordinal @p device reach the mapped slabs. */
    bool shared_with(int device) const noexcept {
        return device == device_ || peers_[static_cast<std::size_t>(device)];
    }

    /**
     *  Maps every slab, those committed already included, for @p devices too, as a plural absorbing
     *  this storage names its own. No slab may be committed meanwhile.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k One of @p devices cannot reach the owner; nothing is granted.
     */
    status_t grant(std::bitset<gpu_devices_capacity_k> devices) noexcept {
        if (!api_ || !*api_) return status_t::device_error_k;
        cuda_api_t const& api = *api_;
        devices[static_cast<std::size_t>(device_)] = false;
        if (reachable_(devices) != devices) return status_t::config_mismatch_k;
        std::bitset<gpu_devices_capacity_k> const granted = peers_ | devices;
        if (granted == peers_) return status_t::success_k;
        cuda_access_descriptor_t access[gpu_devices_capacity_k];
        std::size_t const readers = access_of_(granted, access);
        for (std::size_t offset = 0, slab = 0; offset != mapped_bytes_; offset += slab) {
            slab = min_of(slab_bytes_, max_of(granularity_, offset));
            if (api.memory_set_access(base_ + offset, slab, access, readers) != cuda_success_k)
                return status_t::device_error_k;
        }
        peers_ = granted;
        return status_t::success_k;
    }

    /**
     *  Maps physical slabs until at least @p bytes are usable, for the owner and every peer; never
     *  unmaps early. Each slab is as large as everything mapped before it, from the granularity up
     *  to the preferred slab, so a shard's footprint tracks its rows. Each handle is released right
     *  after mapping, since the mapping alone keeps the memory alive.
     *
     *  @par Statuses:
     *  - @c overflow_k The @p bytes pass the reserved range.
     */
    status_t commit(std::size_t bytes) noexcept {
        if (bytes > virtual_address_bytes_) return status_t::overflow_k;
        // A reservation exists only once `reserve_range` found the driver, so the table is complete
        // here.
        cuda_api_t const& api = *api_;
        cuda_allocation_properties_t properties = allocation_properties_(device_);
        cuda_access_descriptor_t access[gpu_devices_capacity_k];
        std::size_t const readers = access_of_(peers_, access);
        while (mapped_bytes_ < bytes) {
            unsigned long long handle = 0;
            std::size_t const slab = min_of(slab_bytes_, max_of(granularity_, mapped_bytes_));
            if (api.memory_create(&handle, slab, &properties, 0) != cuda_success_k) return status_t::bad_alloc_k;
            bool const mapped = api.memory_map(base_ + mapped_bytes_, slab, 0, handle, 0) == cuda_success_k &&
                                api.memory_set_access(base_ + mapped_bytes_, slab, access, readers) == cuda_success_k;
            api.memory_release(handle);
            if (!mapped) return status_t::device_error_k;
            fu::standard_atomic_ref<std::size_t>(mapped_bytes_).store(mapped_bytes_ + slab, std::memory_order_release);
        }
        return status_t::success_k;
    }

    value_type* base() const noexcept { return reinterpret_cast<value_type*>(static_cast<std::uintptr_t>(base_)); }
    std::size_t reserved_bytes() const noexcept { return virtual_address_bytes_; }
    std::size_t committed_bytes() const noexcept {
        return fu::standard_atomic_ref<std::size_t const>(mapped_bytes_).load(std::memory_order_acquire);
    }
    int device() const noexcept { return device_; }
    cuda_api_t const& api() const noexcept { return *api_; }

    void release() noexcept {
        if (mapped_bytes_) api_->memory_unmap(base_, mapped_bytes_);
        if (base_) api_->memory_address_free(base_, virtual_address_bytes_);
        base_ = 0, virtual_address_bytes_ = 0, mapped_bytes_ = 0;
    }

  private:

    /** Those of @p devices besides the owner whose kernels and copies can reach its slabs. */
    std::bitset<gpu_devices_capacity_k> reachable_(std::bitset<gpu_devices_capacity_k> devices) const noexcept {
        std::bitset<gpu_devices_capacity_k> reachable;
        int owner = 0;
        if (!api_ || !*api_ || api_->get_device(&owner, device_) != cuda_success_k) return reachable;
        cuda_api_t const& api = *api_;
        for (std::size_t ordinal = 0; ordinal != devices.size(); ++ordinal) {
            int peer = 0, reaches = 0;
            if (!devices[ordinal] || static_cast<int>(ordinal) == device_) continue;
            reachable[ordinal] = api.get_device(&peer, static_cast<int>(ordinal)) == cuda_success_k &&
                                 api.can_access_peer(&reaches, peer, owner) == cuda_success_k && reaches;
        }
        return reachable;
    }

    static cuda_allocation_properties_t allocation_properties_(int device) noexcept {
        cuda_allocation_properties_t properties;
        properties.type = cuda_allocation_properties_t::pinned_k;
        properties.location.type = cuda_memory_location_t::device_k;
        properties.location.id = device;
        return properties;
    }

    /** Writes the owner's read-write access, then each of @p peers', into @p access, and answers
     *  how many it wrote. */
    std::size_t access_of_(std::bitset<gpu_devices_capacity_k> peers,
                           cuda_access_descriptor_t (&access)[gpu_devices_capacity_k]) const noexcept {
        access[0] = {allocation_properties_(device_).location, cuda_access_descriptor_t::read_write_k};
        std::size_t readers = 1;
        for (std::size_t ordinal = 0; ordinal != peers.size(); ++ordinal)
            if (peers[ordinal])
                access[readers++] = {{cuda_memory_location_t::device_k, static_cast<int>(ordinal)},
                                     cuda_access_descriptor_t::read_write_k};
        return readers;
    }
};

/** Unified-memory flat storage backend: one @c cuMemAllocManaged reservation for the whole ceiling,
 *  preferring the device and oversubscribable past VRAM. */
template <typename value_type_ = byte_t>
struct cuda_managed_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::managed_k;
    template <typename other_>
    struct rebind {
        using other = cuda_managed_allocator<other_>;
    };

    cuda_api_t const* api_ = nullptr;
    value_type* base_ = nullptr;
    std::size_t virtual_address_bytes_ = 0;
    std::size_t committed_bytes_ = 0; // ? Released by the one grower, acquired by any reader
    int device_ = 0;

    cuda_managed_allocator() = default;

    /** Names the GPU the future reservation prefers, through @p api - placement only, no driver
     *  call. */
    explicit cuda_managed_allocator(cuda_api_t const& api, int device) noexcept : api_(&api), device_(device) {}

    /** Copies the placement configuration only, never a live reservation. */
    cuda_managed_allocator(cuda_managed_allocator const& other) noexcept : api_(other.api_), device_(other.device_) {}
    template <typename other_>
    cuda_managed_allocator(cuda_managed_allocator<other_> const& other) noexcept
        : api_(other.api_), device_(other.device_) {}
    cuda_managed_allocator& operator=(cuda_managed_allocator const&) = delete;
    ~cuda_managed_allocator() noexcept { release(); }

    /** Reserves @p virtual_address_bytes of managed memory, advised to prefer the device named at
     *  construction and to stay reachable from the host. */
    status_t reserve_range(std::size_t virtual_address_bytes) noexcept {
        virtual_address_bytes_ = virtual_address_bytes;
        cuda_device_address_t base = 0;
        if (!api_ || !*api_) return status_t::bad_alloc_k;
        cuda_api_t const& api = *api_;
        if (api.allocate_managed(&base, virtual_address_bytes, cuda_attach_global_k) != cuda_success_k)
            return status_t::bad_alloc_k;
        base_ = reinterpret_cast<value_type*>(static_cast<std::uintptr_t>(base));
        api.memory_advise(base, virtual_address_bytes, cuda_advise_preferred_location_k,
                          {cuda_memory_location_t::device_k, device_});
        api.memory_advise(base, virtual_address_bytes, cuda_advise_accessed_by_k, {cuda_memory_location_t::host_k, 0});
        return status_t::success_k;
    }

    /**
     *  Zeroes managed pages until at least @p bytes are usable, which also commits their residency.
     *
     *  @par Statuses:
     *  - @c overflow_k The @p bytes pass the reserved range.
     */
    status_t commit(std::size_t bytes) noexcept {
        if (bytes > virtual_address_bytes_) return status_t::overflow_k;
        if (bytes <= committed_bytes_) return status_t::success_k;
        // Managed pages arrive unzeroed, unlike slabs; zeroing the fresh range also commits its
        // residency.
        if (api_->fill(cuda_address(reinterpret_cast<byte_t*>(base_) + committed_bytes_), 0,
                       bytes - committed_bytes_) != cuda_success_k)
            return status_t::device_error_k;
        fu::standard_atomic_ref<std::size_t>(committed_bytes_).store(bytes, std::memory_order_release);
        return status_t::success_k;
    }

    value_type* base() const noexcept { return base_; }
    std::size_t reserved_bytes() const noexcept { return virtual_address_bytes_; }
    std::size_t committed_bytes() const noexcept {
        return fu::standard_atomic_ref<std::size_t const>(committed_bytes_).load(std::memory_order_acquire);
    }
    int device() const noexcept { return device_; }
    cuda_api_t const& api() const noexcept { return *api_; }

    /** The reservation ceiling: half the host RAM, since managed reservations are real
     *  allocations; only this backend consults the topology. */
    static std::size_t budget_bytes() noexcept { return fu::volume_ram() / 2; }

    void release() noexcept {
        if (base_) api_->free(cuda_address(base_));
        base_ = nullptr, virtual_address_bytes_ = 0, committed_bytes_ = 0;
    }
};

struct cuda_landings_t;

/** Stream-ordered device allocator for transient scratch, riding the executor stream it was built
 *  from, or the null stream when built from the driver table alone; a default-built one rides
 *  none, so it only fills the deleter of an empty owner. */
template <typename value_type_ = byte_t>
struct cuda_device_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = cuda_device_allocator<other_>;
    };

    cuda_device_allocator() noexcept = default;

    /** Rides the null stream of the current device through @p api. */
    explicit cuda_device_allocator(cuda_api_t const& api) noexcept : api_(&api) {}

    /** Rides the executor's @p stream on the device it names; a template only because that
     *  @c stream_t holds a run of this allocator, so it is declared after it. */
    template <typename stream_type_>
    explicit cuda_device_allocator(stream_type_ const& stream) noexcept
        : api_(stream.api), stream_(stream.handle), device_(stream.device), landings_(stream.landings.get()) {}
    template <typename other_>
    cuda_device_allocator(cuda_device_allocator<other_> const& other) noexcept
        : api_(other.api_), stream_(other.stream()), device_(other.device()), landings_(other.landings()) {}

    value_type* allocate(std::size_t count) noexcept {
        cuda_device_address_t pointer = 0;
        return api_ && *api_ && api_->allocate_async(&pointer, count * sizeof(value_type), stream_) == cuda_success_k
                   ? reinterpret_cast<value_type*>(static_cast<std::uintptr_t>(pointer))
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept {
        if (pointer) api_->free_async(cuda_address(pointer), stream_);
    }
    cuda_stream_handle_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }
    cuda_api_t const& api() const noexcept { return *api_; }

    /** The pinned slots of the stream it rides, which a traversal context lands its counters in. */
    cuda_landings_t* landings() const noexcept { return landings_; }

  private:
    template <typename>
    friend struct cuda_device_allocator;

    cuda_api_t const* api_ = nullptr;
    cuda_stream_handle_t stream_ = nullptr;
    int device_ = 0;
    cuda_landings_t* landings_ = nullptr;
};

/** The byte specialization device storage is placed through. */
using cuda_device_allocator_t = cuda_device_allocator<byte_t>;

/**
 *  @brief The device twin of @c arena_allocator: carves device bytes from one stream's run.
 *
 *  A spent run answers @c nullptr rather than spilling to a pool, so a mis-sized executor is a
 *  reported failure, never a silent allocation; releasing a carve does nothing, and the run is
 *  rewound by whoever owns it, between calls. A context wanting pool-backed scratch names
 *  @ref cuda_device_allocator instead - the allocator type picks the behavior.
 */
template <typename value_type_ = byte_t>
struct cuda_arena_allocator {
    using value_type = value_type_;
    using arena_t = growing_arena<cuda_device_allocator_t>;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = cuda_arena_allocator<other_>;
    };

    /** Carves from @p arena on the executor's @p stream, which names the device. */
    template <typename stream_type_>
    cuda_arena_allocator(arena_t& arena, stream_type_ const& stream) noexcept
        : arena_(arena), api_(stream.api), stream_(stream.handle), device_(stream.device),
          landings_(stream.landings.get()) {}

    /** The rebinding conversion @c std::allocator_traits performs - the identity rides
     *  along, and the run is shared rather than copied. */
    template <typename other_>
    constexpr cuda_arena_allocator(cuda_arena_allocator<other_> const& other) noexcept
        : arena_(other.arena()), api_(&other.api()), stream_(other.stream()), device_(other.device()),
          landings_(other.landings()) {}

    value_type* allocate(std::size_t count) noexcept {
        // Host-side pointer arithmetic over device bytes: the carve never touches them.
        return arena_allocator<value_type_, cuda_device_allocator_t> {arena_}.allocate(count);
    }

    /** A carve is never returned; the run's owner rewinds it between calls. */
    void deallocate(value_type*, std::size_t) noexcept {}

    cuda_stream_handle_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }
    cuda_api_t const& api() const noexcept { return *api_; }

    /** The pinned slots of the stream it rides, which a traversal context lands its counters in. */
    cuda_landings_t* landings() const noexcept { return landings_; }

    /** The run every rebound copy shares. */
    arena_t& arena() const noexcept { return arena_; }
    bool operator==(cuda_arena_allocator const& other) const noexcept { return &arena_ == &other.arena_; }

  private:
    arena_t& arena_;
    cuda_api_t const* api_ = nullptr;
    cuda_stream_handle_t stream_ = nullptr;
    int device_ = 0;
    cuda_landings_t* landings_ = nullptr;
};

/** The byte specialization a walk carves its scratch through. */
using cuda_arena_allocator_t = cuda_arena_allocator<byte_t>;

/** Page-locked host allocator - the DMA source and sink; pinning is synchronous, the
 *  stream is identity only. */
template <typename value_type_ = byte_t>
struct cuda_pinned_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::host_pinned_k;
    template <typename other_>
    struct rebind {
        using other = cuda_pinned_allocator<other_>;
    };

    cuda_pinned_allocator() noexcept = default;

    /** Pins through @p api for host bounces that ride no stream. */
    explicit cuda_pinned_allocator(cuda_api_t const& api) noexcept : api_(&api) {}

    /** Takes the identity of the executor's @p stream and the device it names. */
    template <typename stream_type_>
    explicit cuda_pinned_allocator(stream_type_ const& stream) noexcept
        : api_(stream.api), stream_(stream.handle), device_(stream.device) {}
    template <typename other_>
    cuda_pinned_allocator(cuda_pinned_allocator<other_> const& other) noexcept
        : api_(other.api_), stream_(other.stream()), device_(other.device()) {}

    value_type* allocate(std::size_t count) noexcept {
        void* pointer = nullptr;
        return api_ && *api_ &&
                       api_->allocate_host(&pointer, count * sizeof(value_type), cuda_host_portable_k) == cuda_success_k
                   ? static_cast<value_type*>(pointer)
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept {
        if (pointer) api_->free_host(pointer);
    }

    /** Page-locks @p bytes the caller owns, like a read-only file mapping, so copies from them
     *  run as DMA; @c unpin releases them before they unmap. */
    static status_t pin(cuda_api_t const& api, bytes_view_t bytes) noexcept {
        if (bytes.empty()) return status_t::success_k;
        unsigned const flags = cuda_host_portable_k | cuda_host_read_only_k;
        bool const pinned = api && api.register_host(bytes.data(), bytes.size(), flags) == cuda_success_k;
        return pinned ? status_t::success_k : status_t::device_error_k;
    }
    static void unpin(cuda_api_t const& api, bytes_view_t bytes) noexcept {
        if (!bytes.empty()) api.unregister_host(bytes.data());
    }

    cuda_stream_handle_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }
    cuda_api_t const& api() const noexcept { return *api_; }

  private:
    template <typename>
    friend struct cuda_pinned_allocator;

    cuda_api_t const* api_ = nullptr;
    cuda_stream_handle_t stream_ = nullptr;
    int device_ = 0;
};

/** The byte specialization a host bounce is pinned through. */
using cuda_pinned_allocator_t = cuda_pinned_allocator<byte_t>;

/** Page-locked slots the traversal contexts on one executor stream land their counters in. Each
 *  context takes one at birth and gives it back at death, so two alive at once never share one, and
 *  pinning, which waits on the device, runs only past the most contexts the stream held alive at
 *  once. A slot given back may still await copies its last owner queued, but every owner rides this
 *  one stream, so those land before the next owner's, which it reads only once they landed. */
struct cuda_landings_t {

    /** Bytes one slot holds: a context's telemetry snapshot and its missed word, with room to
     *  spare. */
    inline static constexpr std::size_t slot_bytes_k = 128;

    /** Slots the first pin holds, more than the shards one device usually serves; past them the
     *  run pins twice as many. */
    inline static constexpr std::size_t slots_per_pin_k = 64;

    /** Pins its slots through @p api. */
    explicit cuda_landings_t(cuda_api_t const& api) noexcept : slots_ {cuda_pinned_allocator_t {api}} {}

    /** Hands out a slot, a returned one first; null when pinning more fails. */
    void* take() noexcept {
        std::lock_guard<fu::spin_mutex_t> held {guard_};
        if (!returned_.empty()) return returned_.pop_back();
        if (failed(slots_.reserve_exactly(slot_bytes_k * slots_per_pin_k))) return nullptr;
        return slots_.allocate(slot_bytes_k, slot_bytes_k);
    }

    /** Takes @p slot back for the next context; one the list cannot hold stays pinned until the
     *  stream dies. */
    void give_back(void* slot) noexcept {
        std::lock_guard<fu::spin_mutex_t> held {guard_};
        [[maybe_unused]] status_t const listed = returned_.push_back(slot);
    }

  private:
    growing_arena<cuda_pinned_allocator_t> slots_;
    growing_buffer<void*> returned_ {global_allocator};

    /** Held across a take or a give-back, so contexts born and dying on several threads share
     *  the slots. */
    fu::spin_mutex_t guard_;
};

/** Owned non-blocking stream, born in the current context; destruction drains it first. */
struct cuda_stream_t {
    explicit cuda_stream_t(cuda_api_t const& api) noexcept : api_(&api) {
        if (api) api.stream_create(&handle_, cuda_stream_non_blocking_k);
    }
    cuda_stream_t(cuda_stream_t&& other) noexcept : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)) {}
    cuda_stream_t(cuda_stream_t const&) = delete;
    cuda_stream_t& operator=(cuda_stream_t const&) = delete;
    ~cuda_stream_t() noexcept {
        if (handle_) api_->stream_synchronize(handle_), api_->stream_destroy(handle_);
    }

    operator cuda_stream_handle_t() const noexcept { return handle_; }

  private:
    cuda_api_t const* api_ = nullptr;
    cuda_stream_handle_t handle_ = nullptr;
};

/** Owned event, born in the current context, marking a point on a stream that the host polls or
 *  awaits; it blocks on sync, so a drain sleeps instead of spinning a core. Destruction never
 *  waits. */
struct cuda_event_t {

    /** An empty event, which a refused plan holds and which records nothing. */
    cuda_event_t() noexcept = default;

    explicit cuda_event_t(cuda_api_t const& api) noexcept : api_(&api) {
        if (api) api.event_create(&handle_, cuda_event_blocking_sync_k | cuda_event_disable_timing_k);
    }
    cuda_event_t(cuda_event_t&& other) noexcept : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)) {}
    cuda_event_t(cuda_event_t const&) = delete;
    cuda_event_t& operator=(cuda_event_t const&) = delete;
    ~cuda_event_t() noexcept {
        if (handle_) api_->event_destroy(handle_);
    }

    operator cuda_event_handle_t() const noexcept { return handle_; }

    /**
     *  Marks everything queued on @p stream so far, which @ref landed and @ref wait then await.
     *
     *  @par Statuses:
     *  - @c device_error_k The event was never created.
     */
    status_t record(cuda_stream_handle_t stream) noexcept {
        return handle_ && api_->event_record(handle_, stream) == cuda_success_k ? status_t::success_k
                                                                                : status_t::device_error_k;
    }

    /**
     *  Polls the recorded work without blocking; nothing recorded has nothing to run.
     *
     *  @par Statuses:
     *  - @c device_error_k The event was never created.
     */
    status_t landed() const noexcept {
        if (!handle_) return status_t::device_error_k;
        cuda_result_t const queried = api_->event_query(handle_);
        return queried == cuda_success_k           ? status_t::success_k
               : queried == cuda_error_not_ready_k ? status_t::deferred_k
                                                   : status_t::device_error_k;
    }

    /**
     *  Sleeps until the recorded work finishes and answers whether the device finished it.
     *
     *  @par Statuses:
     *  - @c device_error_k The event was never created.
     */
    status_t wait() noexcept {
        return handle_ && api_->event_synchronize(handle_) == cuda_success_k ? status_t::success_k
                                                                             : status_t::device_error_k;
    }

  private:
    cuda_api_t const* api_ = nullptr;
    cuda_event_handle_t handle_ = nullptr;
};

/** Device memory that one call at a time rewinds and refills, on one device: the NVIDIA alternative
 *  of @ref gpu_arena_t, where the C layer lands a VRAM call's outputs. It grows, uploads and frees
 *  in the order of the first stream of the executor that made it, so it never waits on the device
 *  until @ref wait, and must die before that executor. */
struct cuda_device_arena_t {

    /** Rides the executor's @p stream on the device it names, and drains through its event; a
     *  template only because that @c stream_t is declared after it. */
    template <typename stream_type_>
    explicit cuda_device_arena_t(stream_type_& stream) noexcept
        : memory_(cuda_device_allocator_t {stream}), api_(stream.api), stream_(stream.handle), event_(&stream.event),
          device_(stream.device) {}

    /** Carves @p count values on the arena's device, or null when it cannot grow. */
    template <typename value_type_>
    value_type_* allocate(std::size_t count) noexcept {
        if (cuda_set_device(*api_, device_) != cuda_success_k) return nullptr;
        return reinterpret_cast<value_type_*>(memory_.allocate(count * sizeof(value_type_), default_alignment_k));
    }

    /** Whether @p address lies in memory handed out since the last @ref clear. */
    bool contains(void const* address) const noexcept { return memory_.contains(address); }

    /** Rewinds the arena, so the previous call's outputs die. */
    void clear() noexcept { memory_.clear(); }

    /** Queues a copy of @p bytes of host memory at @p source to @p destination in the arena behind
     *  the call's work; @p source must stay alive until @ref wait. */
    status_t upload(void* destination, void const* source, std::size_t bytes) noexcept {
        return cuda_set_device(*api_, device_) == cuda_success_k &&
                       cuda_copy_async(*api_, destination, source, bytes, stream_) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Waits once for every upload queued so far - the end of the call holding the arena -
     *  sleeping on the stream's event. */
    status_t wait() noexcept {
        if (cuda_set_device(*api_, device_) != cuda_success_k) return status_t::device_error_k;
        if (status_t const recorded = event_->record(stream_); failed(recorded)) return recorded;
        return event_->wait();
    }

    int device() const noexcept { return device_; }

  private:
    growing_arena<cuda_device_allocator_t> memory_;
    cuda_api_t const* api_ = nullptr;
    cuda_stream_handle_t stream_ = nullptr;
    cuda_event_t* event_ = nullptr;
    int device_ = 0;
};

/** The streams a bulk verb advances on and the scratch they walk with - one of each per device,
 *  which every shard there shares in order. Work is queued, never waited on, so one host thread
 *  drives every device. */
struct cuda_executor_t {

    /** Where a call names this executor, the C layer lands outputs in this arena. */
    using device_arena_t = cuda_device_arena_t;
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::nvidia_k;

    /** One device's run of scratch, carved and rewound between calls; sized once by
     *  @ref reserve, since relocation would be a host loop over device bytes. */
    using arena_t = growing_arena<cuda_device_allocator_t>;

    /** One call's hold on the executor, released when it dies. */
    using claim_t = std::unique_lock<fu::spin_mutex_t>;

    /** One shard's stream and run; the landings are declared before the stream, so they unpin only
     *  once it drained, and the arena last, so it frees onto a live stream. */
    struct stream_t {

        /** The driver table every call on this stream goes through. */
        cuda_api_t const* api = nullptr;

        /** The device this stream was opened on - placement and launch target both. */
        int device = 0;

        /** The pinned slots every traversal context on this stream lands its counters in, held
         *  apart so its allocators keep their address as the stream moves. */
        std::unique_ptr<cuda_landings_t> landings;

        /** The ordering identity every transfer and launch on this device rides. */
        cuda_stream_t handle;

        /** The point @ref wait and the C layer's arena record and sleep on, so a wait frees its
         *  core. */
        cuda_event_t event;

        /** The scratch a walk carves its state from, sized once by @ref reserve. */
        arena_t arena;

        /** Takes @p handle, opened on @p device through @p api, and binds the arena to it; the
         *  event is born there too, so @p device must be current. */
        stream_t(cuda_api_t const& api, int device, cuda_stream_t&& handle) noexcept
            : api(&api), device(device), landings(new (std::nothrow) cuda_landings_t {api}), handle(std::move(handle)),
              event(api), arena(cuda_device_allocator_t {*this}) {}
    };

    cuda_executor_t() = default;
    cuda_executor_t(cuda_executor_t&& other) noexcept { swap(other); }
    cuda_executor_t& operator=(cuda_executor_t&& other) noexcept { return swap(other), *this; }
    cuda_executor_t(cuda_executor_t const&) = delete;
    cuda_executor_t& operator=(cuda_executor_t const&) = delete;

    /** Exchanges the stream tables; @c held_ stays, since a move with a call in flight is
     *  misuse. */
    void swap(cuda_executor_t& other) noexcept {
        std::swap(api_, other.api_);
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
     *  Takes over streams the caller already opened, owned and destroyed here from now on.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p streams are empty.
     */
    static expected<cuda_executor_t> acquire(growing_buffer<stream_t>&& streams) noexcept {
        if (streams.empty()) return {{}, status_t::config_mismatch_k};
        cuda_executor_t executor;
        executor.api_ = streams[0].api;
        executor.streams_ = std::move(streams);
        return {std::move(executor), status_t::success_k};
    }

    /**
     *  Opens one stream per entry of @p devices through @p api, binding each first, since a stream
     *  is born on the current device; their scratch never grows past @p scratch_limit bytes, zero
     *  for none.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p devices are empty.
     */
    static expected<cuda_executor_t> spawn(cuda_api_t const& api, std::span<int const> devices,
                                           std::size_t scratch_limit = 0) noexcept {
        growing_buffer<stream_t> streams {global_allocator};
        if (failed(streams.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int device : devices) {
            if (cuda_set_device(api, device) != cuda_success_k) return {{}, status_t::device_error_k};
            stream_t opened {api, device, cuda_stream_t {api}};
            if (!opened.landings) return {{}, status_t::bad_alloc_k};
            if (!opened.handle || !opened.event) return {{}, status_t::device_error_k};
            streams.push_back_reserved(std::move(opened));
        }
        auto [executor, status] = acquire(std::move(streams));
        executor.scratch_limit_ = scratch_limit;
        return {std::move(executor), status};
    }

    /**
     *  Claims @p bytes_per_stream of scratch on every stream, ahead of the fan-out that maps state
     *  onto it; the old run frees in stream order behind whatever still rides it.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor has no streams.
     *  - @c bad_alloc_k The @p bytes_per_stream pass the scratch ceiling.
     */
    status_t reserve(std::size_t bytes_per_stream) noexcept {
        if (streams_.empty()) return status_t::uninitialized_k;
        if (scratch_limit_ && bytes_per_stream > scratch_limit_) return status_t::bad_alloc_k;
        for (stream_t& opened : streams_) {
            opened.arena.clear();
            if (failed(opened.arena.reserve_exactly(bytes_per_stream))) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** Waits for every transfer and launch queued on every stream - the fleet's completion -
     *  marking each stream with its device bound, then sleeping on each mark in turn. */
    status_t wait() noexcept {
        for (stream_t& opened : streams_)
            if (cuda_set_device(*api_, opened.device) != cuda_success_k || failed(opened.event.record(opened.handle)))
                return status_t::device_error_k;
        for (stream_t& opened : streams_)
            if (status_t const drained = opened.event.wait(); failed(drained)) return drained;
        return status_t::success_k;
    }

    /** The stream at @p index, in the order the devices were named. */
    stream_t& stream(std::size_t index) noexcept { return streams_[index]; }

    /** The device stream @p index drives - what a caller pairs its shards against. */
    int device(std::size_t index) const noexcept { return streams_[index].device; }

    /** The first stream on @p device, or null when the executor drives none there. */
    stream_t* stream_on(int device) noexcept {
        for (stream_t& opened : streams_)
            if (opened.device == device) return &opened;
        return nullptr;
    }

    /** Streams this executor drives, one per device. */
    std::size_t size() const noexcept { return streams_.size(); }

    /** The driver table every stream here goes through. */
    cuda_api_t const& api() const noexcept { return *api_; }

  private:
    cuda_api_t const* api_ = nullptr;

    /** One stream per named entry, in the caller's order. */
    growing_buffer<stream_t> streams_ {global_allocator};

    /** Held by the one caller driving the streams; never swapped. */
    fu::spin_mutex_t held_;

    /** The ceiling on every stream's scratch bytes; zero lets scratch grow as geometry asks. */
    std::size_t scratch_limit_ = 0;
};

/** The @c std::unique_ptr deleter carrying the allocator and the element count. */
template <typename allocator_>
struct cuda_deleter {
    allocator_ allocator;
    std::size_t count = 0;

    void operator()(typename allocator_::value_type* pointer) noexcept { allocator.deallocate(pointer, count); }
};

/** Owned device or pinned allocation; host code takes @c .get() for views and transfers. */
template <typename value_type_, typename allocator_ = cuda_device_allocator<value_type_>>
using cuda_unique_ptr = std::unique_ptr<value_type_, cuda_deleter<allocator_>>;

/** The @c std::make_unique analog, null on exhaustion; the bytes arrive uninitialized. */
template <typename allocator_>
cuda_unique_ptr<typename allocator_::value_type, allocator_> cuda_make_unique(allocator_ allocator,
                                                                              std::size_t count) noexcept {
    typename allocator_::value_type* pointer = allocator.allocate(count);
    return {pointer, cuda_deleter<allocator_> {allocator, count}};
}

/** One fatbin loaded into one GPU's primary context and unloaded with this owner; kernels are found
 *  by name. */
struct cuda_module_t {
    cuda_module_t() noexcept = default;
    cuda_module_t(cuda_module_t&& other) noexcept
        : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)), device_(other.device_) {}
    cuda_module_t& operator=(cuda_module_t&& other) noexcept {
        std::swap(api_, other.api_), std::swap(handle_, other.handle_), std::swap(device_, other.device_);
        return *this;
    }
    cuda_module_t(cuda_module_t const&) = delete;
    cuda_module_t& operator=(cuda_module_t const&) = delete;
    ~cuda_module_t() noexcept {
        if (handle_ && cuda_set_device(*api_, device_) == cuda_success_k) api_->module_unload(handle_);
    }

    /**
     *  Loads @p fatbin onto device @p device through @p api, binding it first, since a module
     *  loads into the current context; the driver picks the cubin matching the device, or
     *  JIT-compiles its PTX.
     *
     *  @par Statuses:
     *  - @c device_error_k The @p fatbin is empty.
     */
    static expected<cuda_module_t> load(cuda_api_t const& api, int device, bytes_view_t fatbin) noexcept {
        cuda_module_t module;
        module.api_ = &api;
        module.device_ = device;
        if (fatbin.empty() || cuda_set_device(api, device) != cuda_success_k ||
            api.module_load_data(&module.handle_, fatbin.data()) != cuda_success_k)
            return {{}, status_t::device_error_k};
        return {std::move(module), status_t::success_k};
    }

    /** The driver table the fatbin was loaded through, which launches of its kernels go through. */
    cuda_api_t const& api() const noexcept { return *api_; }

    /** The kernel exported as @p name, or null when the fatbin has none. */
    cuda_function_handle_t function(char const* name) const noexcept {
        cuda_function_handle_t function = nullptr;
        return handle_ && api_->module_get_function(&function, handle_, name) == cuda_success_k ? function : nullptr;
    }

  private:
    cuda_api_t const* api_ = nullptr;
    cuda_module_handle_t handle_ = nullptr;
    int device_ = 0;
};

/** A row-major matrix of equal rows as the two-dimensional tensor map the Tensor Memory Accelerator
 *  gathers rows through, four at a time, one window of each per copy. The zeroed record spans no
 *  rows, and kernels reading it gather none. */
struct cuda_rows_map_t {

    /** The driver's descriptor of the matrix. */
    cuda_tensor_map_t tensor_map;

    /** Rows the map spans, which its signed coordinates cap below 2^31; zero without a map. */
    u32_t rows_count = 0;

    /** Bytes of one row a box spans: the whole row, or a leading window of it. */
    u32_t window_bytes = 0;
};

/** Encodes @p rows_count rows of @p row_bytes bytes from @p first_row in eight-byte words, a box
 *  spanning @p window_bytes of one row, unswizzled, reading zeros past a row's end. Answers the
 *  zeroed record where the driver lacks the encoder or refuses the shape; the device it encodes for
 *  must be current. */
inline cuda_rows_map_t cuda_encode_rows_map(cuda_api_t const& api, void* first_row, u32_t row_bytes, u32_t rows_count,
                                            u32_t window_bytes) noexcept {
    if (!api.encode_tensor_map_tiled || !rows_count || row_bytes % 16 || window_bytes % 16) return {};
    cuda_rows_map_t map;
    u64_t const dimensions[2] = {row_bytes / 8, rows_count}, strides[1] = {row_bytes};
    u32_t const box[2] = {window_bytes / 8, 1}, steps[2] = {1, 1};
    // No interleave, no swizzle, no L2 promotion, and zeros out of bounds.
    if (api.encode_tensor_map_tiled(&map.tensor_map, cuda_tensor_map_u64_k, 2, first_row, dimensions, strides, box,
                                    steps, 0, 0, 0, 0) != cuda_success_k)
        return {};
    map.rows_count = rows_count, map.window_bytes = window_bytes;
    return map;
}

/** Whether a launch waits for its stream predecessor to finish, or may start while it drains. */
enum class cuda_overlap_t : bool { serialized_k, overlapping_k };

/**
 *  Launches @p function over @p blocks of @p threads on @p stream - the only launch path - passing
 *  each argument by address, as the driver takes them, so each must already be the kernel's own
 *  type, raising the kernel's dynamic-shared ceiling first when @p shared_bytes outgrows the window
 *  the kernel grants. With @p overlapping the kernel may start while its predecessor on @p stream
 *  drains, which only a kernel waiting on that predecessor at entry may ask for.
 *
 *  @par Statuses:
 *  - @c device_error_k The @p function is null, or the launch refuses a bad configuration or too
 *    much shared memory.
 */
template <typename... argument_types_>
status_t cuda_launch(cuda_api_t const& api, cuda_function_handle_t function, u32_t blocks, u32_t threads,
                     u32_t shared_bytes, cuda_stream_handle_t stream, cuda_overlap_t overlapping,
                     argument_types_... arguments) noexcept {
    static_assert((std::is_trivially_copyable_v<argument_types_> && ...), "Kernels take their arguments by bytes");
    if (!function) return status_t::device_error_k;
    int granted_bytes = 0;
    if (shared_bytes &&
        api.function_get_attribute(&granted_bytes, cuda_function_dynamic_shared_bytes_k, function) != cuda_success_k)
        return status_t::device_error_k;
    if (shared_bytes > static_cast<u32_t>(granted_bytes) &&
        api.function_set_attribute(function, cuda_function_dynamic_shared_bytes_k, static_cast<int>(shared_bytes)) !=
            cuda_success_k)
        return status_t::device_error_k;
    void* addresses[] = {static_cast<void*>(&arguments)...};
    cuda_launch_attribute_t overlap {cuda_launch_programmatic_serialization_k, {1}};
    cuda_launch_config_t config;
    config.grid_x = blocks, config.block_x = threads, config.shared_bytes = shared_bytes, config.stream = stream;
    config.attributes = &overlap, config.attributes_count = overlapping == cuda_overlap_t::overlapping_k ? 1u : 0u;
    return api.launch_kernel(&config, function, addresses, nullptr) == cuda_success_k ? status_t::success_k
                                                                                      : status_t::device_error_k;
}

static_assert(cuda_reservable_allocator<cuda_virtual_memory_allocator<byte_t>>);

/** Claim counters a traversal context keeps: three for each of the insert waves one fill zeroes
 *  ahead, so overlapping launches chain wave after wave with no fill between them. Waves double
 *  from 32 rows, so no 32-bit ID range spans more than 28. */
inline constexpr u32_t cuda_claim_counters_k = 3 * (std::numeric_limits<u32_t>::digits -
                                                    std::countr_zero(vram_admission_ramp_floor_k) + 1);

/** One stream's device-side scratch for driving a CUDA index: the traversal arena, the batch
 *  staging, the telemetry counters and the attend miss word, riding the stream its allocator names
 *  and never opening one. Two contexts on one index share only the storage and the published
 *  counter, never scratch. Callers observe it through @c results, @ref spent and @ref missed, its
 *  scratch being the indexes' verbs' alone. */
template <cuda_allocator allocator_type_ = cuda_device_allocator_t>
struct cuda_traversal_context {
    using allocator_t = allocator_type_;

    /** The single-query verb's landing, grown to the walk's breadth in pinned pages: the ascending
     *  matches, then, past them, the IDs and distances the device copies back. */
    struct results_t {
        growing_buffer<scored_node_t, cuda_pinned_allocator_t> entries_;
        std::size_t count_ = 0;

        scored_node_t* data() noexcept { return entries_.data(); }
        scored_node_t const* data() const noexcept { return entries_.data(); }
        scored_node_t& operator[](std::size_t rank) noexcept { return entries_[rank]; }
        scored_node_t const& operator[](std::size_t rank) const noexcept { return entries_[rank]; }
        std::size_t size() const noexcept { return count_; }
    };

    results_t results;

    /** Binds this context to @p allocator's stream and device, selecting that device first, since
     *  the event is born in the current context, allocates only the fixed-size counters, and
     *  borrows their host twins from the stream's pinned landings; sized buffers wait for the first
     *  verb. */
    cuda_traversal_context(allocator_t allocator) noexcept
        : results {growing_buffer<scored_node_t, cuda_pinned_allocator_t> {cuda_pinned_allocator_t {allocator.api()}}},
          allocator_ {(cuda_set_device(allocator.api(), allocator.device()), allocator)}, //
          telemetry_ {cuda_make_unique(telemetry_allocator_t {allocator_}, 1)},
          assigned_nodes_ {cuda_make_unique(counter_allocator_t {allocator_}, cuda_claim_counters_k)},
          missed_ {cuda_make_unique(counter_allocator_t {allocator_}, 1)},
          host_counters_ {take_counters_(allocator_.landings())},
          arena_ {nullptr, cuda_deleter<allocator_t> {allocator_, 0}},
          staging_ {nullptr, cuda_deleter<allocator_t> {allocator_, 0}}, event_ {allocator_.api()} {
        if (telemetry_) cuda_fill_async(api(), telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream());
        if (missed_) cuda_fill_async(api(), missed_.get(), 0, sizeof(u32_t), stream());
    }

    cuda_traversal_context(cuda_traversal_context const&) = delete;
    cuda_traversal_context& operator=(cuda_traversal_context const&) = delete;

    /** Every owned resource frees in stream order, so a move is a member-wise steal and destruction
     *  needs no drain. */
    cuda_traversal_context(cuda_traversal_context&&) noexcept = default;
    ~cuda_traversal_context() noexcept = default;

    cuda_stream_handle_t stream() const noexcept { return allocator_.stream(); }
    int device() const noexcept { return allocator_.device(); }
    cuda_api_t const& api() const noexcept { return allocator_.api(); }

    /** The allocator this context carves its scratch from, for a caller's own launches beside
     *  it. */
    allocator_t allocator() const noexcept { return allocator_; }

    /** Whether this context can walk at all: every device-side word and its host twin claimed, and
     *  its event born. */
    explicit operator bool() const noexcept {
        return telemetry_ && assigned_nodes_ && missed_ && host_counters_ && event_;
    }

    /** Waits for every transfer and launch queued on this context's stream - one stream's
     *  completion, the executor's @c drain being the fleet's - sleeping on the context's event
     *  instead of spinning a core. Selects the context's device first, since an event records only
     *  in its own device's context. */
    status_t wait() noexcept {
        if (cuda_set_device(api(), device()) != cuda_success_k) return status_t::device_error_k;
        if (status_t const recorded = record(); failed(recorded)) return recorded;
        return event_.wait();
    }

    /**
     *  Marks everything queued on this context's stream so far as the submission @ref landed
     *  polls, behind the copies of the counters @ref spent and @ref missed read; the context's
     *  device must be current.
     *
     *  @par Statuses:
     *  - @c device_error_k The context has no counters.
     */
    status_t record() noexcept {
        if (!host_counters_ ||
            cuda_copy_async(api(), &host_counters_->telemetry, telemetry_.get(), sizeof(vram_telemetry_block_t),
                            stream()) != cuda_success_k ||
            cuda_copy_async(api(), &host_counters_->missed, missed_.get(), sizeof(u32_t), stream()) != cuda_success_k)
            return status_t::device_error_k;
        return event_.record(stream());
    }

    /**
     *  Polls whether the submission @ref record marked finished without blocking, so one host
     *  thread awaits whichever of several streams lands first.
     *
     *  @par Statuses:
     *  - @c device_error_k The event was never created.
     */
    status_t landed() const noexcept { return event_.landed(); }

    /** Queues on this context's stream a wait for everything @p other marked last, so work here
     *  follows work there without the host waiting; the two may sit on different devices. */
    status_t follow(cuda_traversal_context const& other) noexcept {
        return api().stream_wait_event(stream(), other.event_, 0) == cuda_success_k ? status_t::success_k
                                                                                    : status_t::device_error_k;
    }

    /** The telemetry of the last submission, zeroed when it was queued; read once it landed. */
    vram_telemetry_block_t spent() const noexcept {
        return host_counters_ ? host_counters_->telemetry : vram_telemetry_block_t {};
    }

    /** Whether the last attend met a node its value lookup had no row for, which refuses it; read
     *  once it landed. */
    bool missed() const noexcept { return host_counters_ && host_counters_->missed != 0; }

    using telemetry_allocator_t = typename allocator_t::template rebind<vram_telemetry_block_t>::other;
    using counter_allocator_t = typename allocator_t::template rebind<u32_t>::other;
    using bytes_ptr_t = cuda_unique_ptr<byte_t, allocator_t>;

    /** The host's twins of the device counters, landed behind each @ref record. */
    struct host_counters_t {
        vram_telemetry_block_t telemetry;
        u32_t missed;
    };
    static_assert(sizeof(host_counters_t) <= cuda_landings_t::slot_bytes_k &&
                      alignof(host_counters_t) <= cuda_landings_t::slot_bytes_k,
                  "A context's counters fit one landing slot");

    /** Gives a context's counters back to the landings of the stream they came from. */
    struct counters_return_t {
        cuda_landings_t* landings = nullptr;
        void operator()(host_counters_t* counters) const noexcept { landings->give_back(counters); }
    };
    using host_counters_ptr_t = std::unique_ptr<host_counters_t, counters_return_t>;

    /** Zeroed counters in a slot of @p landings, or null when there are none or it cannot pin. */
    static host_counters_ptr_t take_counters_(cuda_landings_t* landings) noexcept {
        void* const slot = landings ? landings->take() : nullptr;
        return {slot ? new (slot) host_counters_t {} : nullptr, counters_return_t {landings}};
    }

    allocator_t allocator_;

    /** Monotone epoch floor over the arena's visited slots, so the arena is zeroed only when
     *  freshly allocated or when the 32-bit epoch space wraps, never per launch. */
    u64_t epoch_floor_ = 0;

    /** The device-side counters of the last submission, zeroed when it was queued. */
    cuda_unique_ptr<vram_telemetry_block_t, telemetry_allocator_t> telemetry_;

    /** Entries the current launch's batch warps have assigned themselves so far: a builder's nodes,
     *  or a walk's queries; @ref cuda_claim_counters_k words, the first serving one launch. */
    cuda_unique_ptr<u32_t, counter_allocator_t> assigned_nodes_;

    /** Raised by the last attend's walks at a node their value lookup had no row for, zeroed when
     *  it was queued. */
    cuda_unique_ptr<u32_t, counter_allocator_t> missed_;

    /** Borrowed from the stream's page-locked landings, so the copies @ref record queues into it
     *  never wait on the device, and no context pins memory of its own. */
    host_counters_ptr_t host_counters_;
    bytes_ptr_t arena_;
    std::size_t arena_bytes_ = 0;
    bytes_ptr_t staging_;
    std::size_t staging_bytes_ = 0;

    /** The point on the stream @ref wait and @ref landed await. */
    cuda_event_t event_;

    /** Grow-only traversal arena for @p bytes: the exact visited sets and spill tails of every
     *  resident warp, and a builder's scratch when shared memory cannot fit it, kept across
     *  calls. @return The base, or null on exhaustion. */
    byte_t* ensure_arena(std::size_t bytes) noexcept { return ensure_(arena_, arena_bytes_, bytes); }

    /**
     *  Takes @p count epochs off the floor for one launch's walks, zeroing the arena first when it
     *  grew past the @p before bytes it held or the 32-bit epoch space would wrap, never per
     *  launch; a refused zeroing spends the arena, so a retry zeroes it again.
     *
     *  @return The first epoch.
     */
    expected<u32_t> take_epochs(std::size_t count, std::size_t before) noexcept {
        if (arena_bytes_ != before || epoch_floor_ + count > 0xFFFFFFFEull) {
            if (cuda_fill_async(api(), arena_.get(), 0, arena_bytes_, stream()) != cuda_success_k) {
                arena_bytes_ = 0;
                return {0, status_t::device_error_k};
            }
            epoch_floor_ = 0;
        }
        u32_t const first = static_cast<u32_t>(epoch_floor_);
        epoch_floor_ += count;
        return {first, status_t::success_k};
    }

    /** Grow-only staging for one batch's device transport: an insert's padded vectors, or a
     *  search's queries and result landing. */
    byte_t* ensure_staging(std::size_t bytes) noexcept { return ensure_(staging_, staging_bytes_, bytes); }

    /** Grows @p buffer to @p bytes when @p capacity falls short; @return the base, or null. */
    byte_t* ensure_(bytes_ptr_t& buffer, std::size_t& capacity, std::size_t bytes) noexcept {
        if (bytes <= capacity) return buffer.get();
        // The deleter already carries this context's allocator, so only the run and its count
        // change.
        buffer.reset(allocator_.allocate(bytes));
        buffer.get_deleter().count = bytes;
        capacity = buffer ? bytes : 0;
        return buffer.get();
    }
};

static_assert(cuda_reservable_allocator<cuda_managed_allocator<byte_t>>);
static_assert(cuda_allocator<cuda_device_allocator_t>);
static_assert(cuda_allocator<cuda_arena_allocator_t>);
static_assert(cuda_allocator<cuda_pinned_allocator_t>);

/** The NVIDIA types and driver calls a vendor-neutral VRAM index declares itself with and queues
 *  through; every copy tells its direction apart by unified addressing. */
struct cuda_vendor_t {
    using api_t = cuda_api_t;
    using stream_handle_t = cuda_stream_handle_t;
    using event_t = cuda_event_t;
    using landings_t = cuda_landings_t;
    using pinned_allocator_t = cuda_pinned_allocator_t;
    using device_specs_t = cuda_device_specs_t;
    using address_space_t = cuda_address_space_t;
    using executor_t = cuda_executor_t;

    template <typename allocator_type_>
    using traversal_context = cuda_traversal_context<allocator_type_>;

    /**
     *  Makes the primary context of device @p ordinal current on the calling thread.
     *
     *  @par Statuses:
     *  - @c device_error_k The @p ordinal is past the device listing.
     */
    static status_t select(api_t const& api, int ordinal) noexcept { return verdict_(cuda_set_device(api, ordinal)); }

    /** Queues a copy of @p bytes from @p from to @p to on @p stream, in either direction. */
    static status_t copy(api_t const& api, void* to, void const* from, std::size_t bytes,
                         stream_handle_t stream) noexcept {
        return verdict_(cuda_copy_async(api, to, from, bytes, stream));
    }

    /** Queues a copy of @p height rows of @p width bytes between pitched buffers on @p stream. */
    static status_t copy_2d(api_t const& api, void* to, std::size_t to_pitch, void const* from, std::size_t from_pitch,
                            std::size_t width, std::size_t height, stream_handle_t stream) noexcept {
        return verdict_(cuda_copy_2d_async(api, to, to_pitch, from, from_pitch, width, height, stream));
    }

    /** Queues setting @p bytes at @p to to @p value on @p stream. */
    static status_t fill(api_t const& api, void* to, u8_t value, std::size_t bytes, stream_handle_t stream) noexcept {
        return verdict_(cuda_fill_async(api, to, value, bytes, stream));
    }

    /** Queues setting @p count 32-bit words at @p to to @p value on @p stream. */
    static status_t fill_words(api_t const& api, void* to, u32_t value, std::size_t count,
                               stream_handle_t stream) noexcept {
        return verdict_(api.fill_words_async(cuda_address(to), value, count, stream));
    }

    /** Queues on @p stream a wait for everything @p event marked last. */
    static status_t follow(api_t const& api, stream_handle_t stream, event_t const& event) noexcept {
        return verdict_(api.stream_wait_event(stream, event, 0));
    }

  private:
    static status_t verdict_(cuda_result_t result) noexcept {
        return result == cuda_success_k ? status_t::success_k : status_t::device_error_k;
    }
};

/** Writes every NVIDIA device @p api reaches into @p into, in CUDA ordinal order, as the driver
 *  reports it, and answers how many it wrote; none without a driver. */
inline std::size_t list_nvidia_gpu_devices(cuda_api_t const& api, std::span<gpu_device_t> into) noexcept {
    std::size_t count = 0;
    int devices_count = 0;
    if (!api || api.get_device_count(&devices_count) != cuda_success_k) return count;
    for (int ordinal = 0; ordinal != devices_count && count != into.size(); ++ordinal) {
        auto const [specs, status] = cuda_device_specs_t::query(api, ordinal);
        if (failed(status)) continue;
        gpu_device_t& device = into[count++];
        device = {gpu_vendor_t::nvidia_k,
                  static_cast<u32_t>(ordinal),
                  static_cast<u32_t>(specs.compute_major),
                  static_cast<u32_t>(specs.compute_minor),
                  specs.warp_lanes,
                  specs.multiprocessor_count,
                  specs.vram_bytes};
        int handle = 0;
        if (api.get_device(&handle, ordinal) == cuda_success_k)
            api.get_device_name(device.name, static_cast<int>(sizeof(device.name)), handle);
    }
    return count;
}

} // namespace unum::usearch::utilities
