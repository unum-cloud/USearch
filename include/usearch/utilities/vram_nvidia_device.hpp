/**
 *  @file include/usearch/utilities/vram_nvidia_device.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Host-side vocabulary for driving one NVIDIA GPU - the driver, specs, streams, memory.
 *
 *  The library never links @c libcuda or the CUDA runtime: @ref cuda_api opens the driver at first
 *  use and resolves the entry points USearch calls into one table, which stays empty where the
 *  driver is absent, so NVIDIA then lists no devices and the binary still loads. Kernels reach a
 *  device as fatbins compiled ahead of time and embedded in the library, loaded by
 *  @ref cuda_module_t and launched by name, since NVCC's fatbin registration would link the
 *  runtime outright.
 *
 *  Plain C++ over the driver's declarations in @c <cuda.h>, hence the @c .hpp extension; the warp
 *  policy the shared walk code is written against lives beside the kits in
 *  @c vram_nvidia_metric.cuh. The flat @c cuda_allocator carries @c std::allocator's call surface
 *  with the stream bound at construction; the @c cuda_reservable_allocator is the flat storage
 *  backend, a reserve-once / commit-on-demand range whose base survives growth. Nothing models
 *  both.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` for device-pointer casts

#if defined(_WIN32)
#include <windows.h> // `LoadLibraryA`, `GetProcAddress` - the driver opened at first use
#else
#include <dlfcn.h> // `dlopen`, `dlsym` - the driver opened at first use
#endif

#include <atomic>      // `std::memory_order` - the committed extent, `std::atomic` - the primary contexts
#include <memory>      // `std::unique_ptr` for owned device allocations
#include <mutex>       // `std::unique_lock`, `std::adopt_lock` - the executor's one-caller hold
#include <span>        // `std::span` - the device list an executor opens over, a fatbin's bytes
#include <type_traits> // `std::is_trivially_copyable_v` - a launch's arguments travel as bytes
#include <utility>     // `std::move`, `std::exchange`, `std::swap`

#include <cuda.h> // Driver API declarations only: `CUdeviceptr`, `CUmemAllocationProp`, `CUlaunchConfig`

#include <forkunion/topology.hpp> // `fu::volume_ram` - the managed storage's host-memory budget
#include <forkunion/types.hpp>    // `fu::round_up_to_multiple`, `fu::spin_mutex_t` - the hold

#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - an executor's table of contexts
#include "usearch/utilities/types.hpp"       // `gpu_device_t`, `gpu_devices_capacity_k`

namespace unum::usearch::utilities {

/**
 *  @brief The driver entry points USearch calls, resolved from @c libcuda on first use.
 *
 *  Every entry is null when the driver is absent or lacks any of them, which leaves NVIDIA with no
 *  devices; each is typed by the declaration @c <cuda.h> maps its name to and resolved by that same
 *  exported symbol, so a @c _v2 macro and its lookup always agree.
 */
struct cuda_api_t {
    decltype(&::cuInit) init = nullptr;
    decltype(&::cuDeviceGetCount) get_device_count = nullptr;
    decltype(&::cuDeviceGet) get_device = nullptr;
    decltype(&::cuDeviceGetAttribute) get_device_attribute = nullptr;
    decltype(&::cuDeviceGetName) get_device_name = nullptr;
    decltype(&::cuDeviceTotalMem) get_device_memory = nullptr;
    decltype(&::cuDevicePrimaryCtxRetain) retain_primary_context = nullptr;
    decltype(&::cuCtxSetCurrent) set_current_context = nullptr;
    decltype(&::cuMemAllocAsync) allocate_async = nullptr;
    decltype(&::cuMemFreeAsync) free_async = nullptr;
    decltype(&::cuMemAllocManaged) allocate_managed = nullptr;
    decltype(&::cuMemAdvise) memory_advise = nullptr;
    decltype(&::cuMemFree) free_managed = nullptr;
    decltype(&::cuMemHostAlloc) allocate_host = nullptr;
    decltype(&::cuMemFreeHost) free_host = nullptr;
    decltype(&::cuMemcpy) copy = nullptr;
    decltype(&::cuMemcpyAsync) copy_async = nullptr;
    decltype(&::cuMemcpy2DAsync) copy_2d_async = nullptr;
    decltype(&::cuMemsetD8) fill = nullptr;
    decltype(&::cuMemsetD8Async) fill_async = nullptr;
    decltype(&::cuMemsetD32Async) fill_words_async = nullptr;
    decltype(&::cuStreamCreate) stream_create = nullptr;
    decltype(&::cuStreamDestroy) stream_destroy = nullptr;
    decltype(&::cuStreamSynchronize) stream_synchronize = nullptr;
    decltype(&::cuModuleLoadData) module_load_data = nullptr;
    decltype(&::cuModuleUnload) module_unload = nullptr;
    decltype(&::cuModuleGetFunction) module_get_function = nullptr;
    decltype(&::cuFuncSetAttribute) function_set_attribute = nullptr;
    decltype(&::cuLaunchKernelEx) launch_kernel = nullptr;
    decltype(&::cuMemGetAllocationGranularity) memory_granularity = nullptr;
    decltype(&::cuMemAddressReserve) memory_address_reserve = nullptr;
    decltype(&::cuMemAddressFree) memory_address_free = nullptr;
    decltype(&::cuMemCreate) memory_create = nullptr;
    decltype(&::cuMemRelease) memory_release = nullptr;
    decltype(&::cuMemMap) memory_map = nullptr;
    decltype(&::cuMemUnmap) memory_unmap = nullptr;
    decltype(&::cuMemSetAccess) memory_set_access = nullptr;

    /** Whether the driver loaded and initialized with every entry resolved. */
    explicit operator bool() const noexcept { return memory_set_access != nullptr; }

    /** Opens the driver, resolves every entry and initializes it, or answers the empty table. The
     *  library stays open for the life of the process, as the table points into it. */
    static cuda_api_t load() noexcept {
#if defined(_WIN32)
        void* library = reinterpret_cast<void*>(LoadLibraryA("nvcuda.dll"));
#else
        void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
        if (!library) return {};
        cuda_api_t api;
        bool const resolved = //
            resolve_(library, "cuInit", api.init) && resolve_(library, "cuDeviceGetCount", api.get_device_count) &&
            resolve_(library, "cuDeviceGet", api.get_device) &&
            resolve_(library, "cuDeviceGetAttribute", api.get_device_attribute) &&
            resolve_(library, "cuDeviceGetName", api.get_device_name) &&
            resolve_(library, "cuDeviceTotalMem_v2", api.get_device_memory) &&
            resolve_(library, "cuDevicePrimaryCtxRetain", api.retain_primary_context) &&
            resolve_(library, "cuCtxSetCurrent", api.set_current_context) &&
            resolve_(library, "cuMemAllocAsync", api.allocate_async) &&
            resolve_(library, "cuMemFreeAsync", api.free_async) &&
            resolve_(library, "cuMemAllocManaged", api.allocate_managed) &&
            resolve_(library, "cuMemAdvise_v2", api.memory_advise) &&
            resolve_(library, "cuMemFree_v2", api.free_managed) &&
            resolve_(library, "cuMemHostAlloc", api.allocate_host) &&
            resolve_(library, "cuMemFreeHost", api.free_host) && resolve_(library, "cuMemcpy", api.copy) &&
            resolve_(library, "cuMemcpyAsync", api.copy_async) &&
            resolve_(library, "cuMemcpy2DAsync_v2", api.copy_2d_async) &&
            resolve_(library, "cuMemsetD8_v2", api.fill) && resolve_(library, "cuMemsetD8Async", api.fill_async) &&
            resolve_(library, "cuMemsetD32Async", api.fill_words_async) &&
            resolve_(library, "cuStreamCreate", api.stream_create) &&
            resolve_(library, "cuStreamDestroy_v2", api.stream_destroy) &&
            resolve_(library, "cuStreamSynchronize", api.stream_synchronize) &&
            resolve_(library, "cuModuleLoadData", api.module_load_data) &&
            resolve_(library, "cuModuleUnload", api.module_unload) &&
            resolve_(library, "cuModuleGetFunction", api.module_get_function) &&
            resolve_(library, "cuFuncSetAttribute", api.function_set_attribute) &&
            resolve_(library, "cuLaunchKernelEx", api.launch_kernel) &&
            resolve_(library, "cuMemGetAllocationGranularity", api.memory_granularity) &&
            resolve_(library, "cuMemAddressReserve", api.memory_address_reserve) &&
            resolve_(library, "cuMemAddressFree", api.memory_address_free) &&
            resolve_(library, "cuMemCreate", api.memory_create) &&
            resolve_(library, "cuMemRelease", api.memory_release) && resolve_(library, "cuMemMap", api.memory_map) &&
            resolve_(library, "cuMemUnmap", api.memory_unmap) &&
            resolve_(library, "cuMemSetAccess", api.memory_set_access);
        // A driver without a usable GPU fails its initialization, which lists no devices either.
        if (resolved && api.init(0) == CUDA_SUCCESS) return api;
        return {};
    }

  private:
    template <typename function_type_>
    static bool resolve_(void* library, char const* name, function_type_& entry) noexcept {
#if defined(_WIN32)
        entry = reinterpret_cast<function_type_>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
        entry = reinterpret_cast<function_type_>(dlsym(library, name));
#endif
        return entry != nullptr;
    }
};

/** The one table every NVIDIA call goes through, filled on first use by whichever thread comes
 *  first. */
inline cuda_api_t const& cuda_api() noexcept {
    static cuda_api_t const api = cuda_api_t::load();
    return api;
}

/** Makes the primary context of device @p ordinal current on the calling thread, retained once per
 *  process: the driver twin of selecting a device, which every stream, allocation and launch after
 *  it lands on. */
inline CUresult cuda_set_device(int ordinal) noexcept {
    static std::atomic<CUcontext> primary_contexts[gpu_devices_capacity_k] = {};
    cuda_api_t const& api = cuda_api();
    if (!api) return CUDA_ERROR_NOT_INITIALIZED;
    if (ordinal < 0 || static_cast<std::size_t>(ordinal) >= gpu_devices_capacity_k) return CUDA_ERROR_INVALID_DEVICE;
    CUcontext context = primary_contexts[ordinal].load(std::memory_order_acquire);
    if (!context) {
        // Two racing threads may both retain it, which only raises a count nothing ever releases.
        CUdevice device = 0;
        if (CUresult const found = api.get_device(&device, ordinal); found != CUDA_SUCCESS) return found;
        if (CUresult const retained = api.retain_primary_context(&context, device); retained != CUDA_SUCCESS)
            return retained;
        primary_contexts[ordinal].store(context, std::memory_order_release);
    }
    return api.set_current_context(context);
}

/** The driver's integer spelling of a pointer, which unified addressing shares with the host. */
inline CUdeviceptr cuda_address(void const* pointer) noexcept {
    return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(pointer));
}

/** Copies @p bytes from @p source to @p destination and waits, any direction unified addressing
 *  tells apart. */
inline CUresult cuda_copy(void* destination, void const* source, std::size_t bytes) noexcept {
    cuda_api_t const& api = cuda_api();
    return api ? api.copy(cuda_address(destination), cuda_address(source), bytes) : CUDA_ERROR_NOT_INITIALIZED;
}

/** Queues the copy of @p bytes from @p source to @p destination on @p stream. */
inline CUresult cuda_copy_async(void* destination, void const* source, std::size_t bytes, CUstream stream) noexcept {
    cuda_api_t const& api = cuda_api();
    return api ? api.copy_async(cuda_address(destination), cuda_address(source), bytes, stream)
               : CUDA_ERROR_NOT_INITIALIZED;
}

/** Queues the copy of @p height rows of @p width bytes on @p stream, each row starting a pitch
 *  past the last on either side. */
inline CUresult cuda_copy_2d_async(void* destination, std::size_t destination_pitch, void const* source,
                                   std::size_t source_pitch, std::size_t width, std::size_t height,
                                   CUstream stream) noexcept {
    cuda_api_t const& api = cuda_api();
    if (!api) return CUDA_ERROR_NOT_INITIALIZED;
    CUDA_MEMCPY2D copy {};
    copy.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
    copy.srcDevice = cuda_address(source);
    copy.srcPitch = source_pitch;
    copy.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
    copy.dstDevice = cuda_address(destination);
    copy.dstPitch = destination_pitch;
    copy.WidthInBytes = width;
    copy.Height = height;
    return api.copy_2d_async(&copy, stream);
}

/** Queues setting @p bytes at @p destination to @p value on @p stream. */
inline CUresult cuda_fill_async(void* destination, u8_t value, std::size_t bytes, CUstream stream) noexcept {
    cuda_api_t const& api = cuda_api();
    return api ? api.fill_async(cuda_address(destination), value, bytes, stream) : CUDA_ERROR_NOT_INITIALIZED;
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

/** One GPU's hardware specification - the facts runtime dispatch consults; hardware only,
 *  never derived policy. */
struct cuda_device_specs_t {

    /** The device ordinal every driver call routes through. */
    int ordinal = 0;

    /** Streaming-multiprocessor count - the launch-grid currency. */
    int multiprocessor_count = 0;

    /** Compute-capability major version - the hardware generation, e.g. 9 = Hopper. */
    int compute_major = 0;

    /** Compute-capability minor version - the in-generation revision. */
    int compute_minor = 0;

    /** Total device-resident memory in bytes. */
    std::size_t vram_bytes = 0;

    /** Widest dynamic shared memory one block can be granted at launch, on explicit request. */
    std::size_t shared_memory_per_block_bytes = 0;

    /** This device's generation on the @c cuda_arch_t scale - the kit chains' runtime ceiling. */
    cuda_arch_t arch() const noexcept { return static_cast<cuda_arch_t>(compute_major * 100 + compute_minor * 10); }

    /** Queries the driver for @p ordinal's specification; meaningful only on @c success_k. */
    static expected<cuda_device_specs_t> query(int ordinal) noexcept {
        cuda_api_t const& api = cuda_api();
        CUdevice device = 0;
        cuda_device_specs_t specs {ordinal};
        int shared_memory_per_block = 0;
        if (!api || api.get_device(&device, ordinal) != CUDA_SUCCESS ||
            api.get_device_attribute(&specs.multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) !=
                CUDA_SUCCESS ||
            api.get_device_attribute(&specs.compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) !=
                CUDA_SUCCESS ||
            api.get_device_attribute(&specs.compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) !=
                CUDA_SUCCESS ||
            api.get_device_attribute(&shared_memory_per_block, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                                     device) != CUDA_SUCCESS ||
            api.get_device_memory(&specs.vram_bytes, device) != CUDA_SUCCESS)
            return {{ordinal}, status_t::device_error_k};
        specs.shared_memory_per_block_bytes = static_cast<std::size_t>(shared_memory_per_block);
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
    { allocator.stream() } -> std::convertible_to<CUstream>;
    { allocator.device() } -> std::convertible_to<int>;
};

/** The flat storage backend contract: a reserve-once / commit-on-demand address range whose base
 *  never moves. Not a refinement of @c cuda_allocator, since flat storage never frees mid-life. */
template <typename allocator_>
concept cuda_reservable_allocator = requires(allocator_ allocator, std::size_t bytes, int device) {
    typename allocator_::value_type;
    { allocator_::address_space_k } -> std::convertible_to<cuda_address_space_t>;
    { allocator.reserve_range(bytes, device) } -> std::same_as<status_t>;
    { allocator.commit(bytes) } -> std::same_as<status_t>;
    { allocator.base() } -> std::same_as<typename allocator_::value_type*>;
    { allocator.reserved_bytes() } -> std::same_as<std::size_t>;
    { allocator.committed_bytes() } -> std::same_as<std::size_t>;
    { allocator.release() } -> std::same_as<void>;
    { allocator.device() } -> std::convertible_to<int>;
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

    /** Preferred physical slab, rounded to the driver's granularity at reservation. */
    inline static constexpr std::size_t preferred_slab_bytes_k = 128u << 20;

    CUdeviceptr base_ = 0;
    std::size_t virtual_address_bytes_ = 0;

    /** Bytes mapped so far - the one grower advances it with a release, and any thread may read it
     *  through @c committed_bytes with an acquire; @c mutable for the reference. */
    std::size_t mapped_bytes_ = 0;
    std::size_t slab_bytes_ = 0;
    int device_ = 0;

    cuda_virtual_memory_allocator() = default;

    /** Names the GPU the future reservation pins to - placement only, no driver call. */
    explicit cuda_virtual_memory_allocator(int device) noexcept : device_(device) {}

    /** Copies the placement only, never a live reservation. */
    cuda_virtual_memory_allocator(cuda_virtual_memory_allocator const& other) noexcept : device_(other.device_) {}

    /** Takes over a live reservation, so a table of them relocates without remapping. */
    cuda_virtual_memory_allocator(cuda_virtual_memory_allocator&& other) noexcept
        : base_(std::exchange(other.base_, 0)), virtual_address_bytes_(std::exchange(other.virtual_address_bytes_, 0)),
          mapped_bytes_(std::exchange(other.mapped_bytes_, 0)), slab_bytes_(other.slab_bytes_), device_(other.device_) {
    }
    template <typename other_>
    cuda_virtual_memory_allocator(cuda_virtual_memory_allocator<other_> const& other) noexcept
        : device_(other.device_) {}
    cuda_virtual_memory_allocator& operator=(cuda_virtual_memory_allocator const&) = delete;
    ~cuda_virtual_memory_allocator() noexcept { release(); }

    /** Reserves @p virtual_address_bytes of virtual address space on @p device - no physical
     *  memory yet, so the ceiling costs nothing and should be generous. */
    status_t reserve_range(std::size_t virtual_address_bytes, int device) noexcept {
        cuda_api_t const& api = cuda_api();
        if (!api) return status_t::device_error_k;
        CUmemAllocationProp properties = allocation_properties_(device);
        std::size_t granularity = 0;
        if (api.memory_granularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
            return status_t::device_error_k;
        slab_bytes_ = max_of(granularity, preferred_slab_bytes_k / granularity * granularity);
        virtual_address_bytes_ = fu::round_up_to_multiple(virtual_address_bytes, slab_bytes_);
        device_ = device;
        if (api.memory_address_reserve(&base_, virtual_address_bytes_, 0, 0, 0) != CUDA_SUCCESS)
            return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /** Maps physical slabs until at least @p bytes are usable; never unmaps early. Each handle is
     *  released right after mapping, since the mapping alone keeps the memory alive. */
    status_t commit(std::size_t bytes) noexcept {
        if (bytes > virtual_address_bytes_) return status_t::overflow_k;
        // A reservation exists only once `reserve_range` found the driver, so the table is complete
        // here.
        cuda_api_t const& api = cuda_api();
        CUmemAllocationProp properties = allocation_properties_(device_);
        CUmemAccessDesc access = {};
        access.location = properties.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        while (mapped_bytes_ < bytes) {
            CUmemGenericAllocationHandle handle;
            if (api.memory_create(&handle, slab_bytes_, &properties, 0) != CUDA_SUCCESS) return status_t::bad_alloc_k;
            bool const mapped = api.memory_map(base_ + mapped_bytes_, slab_bytes_, 0, handle, 0) == CUDA_SUCCESS &&
                                api.memory_set_access(base_ + mapped_bytes_, slab_bytes_, &access, 1) == CUDA_SUCCESS;
            api.memory_release(handle);
            if (!mapped) return status_t::device_error_k;
            standard_atomic_ref<std::size_t>(mapped_bytes_)
                .store(mapped_bytes_ + slab_bytes_, std::memory_order_release);
        }
        return status_t::success_k;
    }

    value_type* base() const noexcept { return reinterpret_cast<value_type*>(static_cast<std::uintptr_t>(base_)); }
    std::size_t reserved_bytes() const noexcept { return virtual_address_bytes_; }
    std::size_t committed_bytes() const noexcept {
        return standard_atomic_ref<std::size_t const>(mapped_bytes_).load(std::memory_order_acquire);
    }
    int device() const noexcept { return device_; }

    void release() noexcept {
        if (mapped_bytes_) cuda_api().memory_unmap(base_, mapped_bytes_);
        if (base_) cuda_api().memory_address_free(base_, virtual_address_bytes_);
        base_ = 0, virtual_address_bytes_ = 0, mapped_bytes_ = 0;
    }

  private:
    static CUmemAllocationProp allocation_properties_(int device) noexcept {
        CUmemAllocationProp properties = {};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = device;
        return properties;
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

    value_type* base_ = nullptr;
    std::size_t virtual_address_bytes_ = 0;
    std::size_t committed_bytes_ = 0; // ? Released by the one grower, acquired by any reader
    int device_ = 0;

    cuda_managed_allocator() = default;

    /** Names the GPU the future reservation prefers - placement only, no driver call. */
    explicit cuda_managed_allocator(int device) noexcept : device_(device) {}

    /** Copies the placement configuration only, never a live reservation. */
    cuda_managed_allocator(cuda_managed_allocator const& other) noexcept : device_(other.device_) {}
    template <typename other_>
    cuda_managed_allocator(cuda_managed_allocator<other_> const& other) noexcept : device_(other.device_) {}
    cuda_managed_allocator& operator=(cuda_managed_allocator const&) = delete;
    ~cuda_managed_allocator() noexcept { release(); }

    status_t reserve_range(std::size_t virtual_address_bytes, int device) noexcept {
        cuda_api_t const& api = cuda_api();
        device_ = device;
        virtual_address_bytes_ = virtual_address_bytes;
        CUdeviceptr base = 0;
        if (!api || api.allocate_managed(&base, virtual_address_bytes, CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS)
            return status_t::bad_alloc_k;
        base_ = reinterpret_cast<value_type*>(static_cast<std::uintptr_t>(base));
        CUmemLocation device_location = {};
        device_location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        device_location.id = device;
        CUmemLocation host_location = {};
        host_location.type = CU_MEM_LOCATION_TYPE_HOST;
        api.memory_advise(base, virtual_address_bytes, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, device_location);
        api.memory_advise(base, virtual_address_bytes, CU_MEM_ADVISE_SET_ACCESSED_BY, host_location);
        return status_t::success_k;
    }

    status_t commit(std::size_t bytes) noexcept {
        if (bytes > virtual_address_bytes_) return status_t::overflow_k;
        if (bytes <= committed_bytes_) return status_t::success_k;
        // Managed pages arrive unzeroed, unlike slabs; zeroing the fresh range also commits its
        // residency.
        if (cuda_api().fill(cuda_address(reinterpret_cast<byte_t*>(base_) + committed_bytes_), 0,
                            bytes - committed_bytes_) != CUDA_SUCCESS)
            return status_t::device_error_k;
        standard_atomic_ref<std::size_t>(committed_bytes_).store(bytes, std::memory_order_release);
        return status_t::success_k;
    }

    value_type* base() const noexcept { return base_; }
    std::size_t reserved_bytes() const noexcept { return virtual_address_bytes_; }
    std::size_t committed_bytes() const noexcept {
        return standard_atomic_ref<std::size_t const>(committed_bytes_).load(std::memory_order_acquire);
    }
    int device() const noexcept { return device_; }

    /** The reservation ceiling: half the host RAM, since managed reservations are real
     *  allocations; only this backend consults the topology. */
    static std::size_t budget_bytes() noexcept { return fu::volume_ram() / 2; }

    void release() noexcept {
        if (base_) cuda_api().free_managed(cuda_address(base_));
        base_ = nullptr, virtual_address_bytes_ = 0, committed_bytes_ = 0;
    }
};

/** Stream-ordered device allocator for transient scratch; a null stream rides the legacy
 *  default. */
template <typename value_type_ = byte_t>
struct cuda_device_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = cuda_device_allocator<other_>;
    };

    cuda_device_allocator() noexcept = default;
    cuda_device_allocator(CUstream stream, int device) noexcept : stream_(stream), device_(device) {}
    template <typename other_>
    cuda_device_allocator(cuda_device_allocator<other_> const& other) noexcept
        : stream_(other.stream()), device_(other.device()) {}

    value_type* allocate(std::size_t count) noexcept {
        cuda_api_t const& api = cuda_api();
        CUdeviceptr pointer = 0;
        return api && api.allocate_async(&pointer, count * sizeof(value_type), stream_) == CUDA_SUCCESS
                   ? reinterpret_cast<value_type*>(static_cast<std::uintptr_t>(pointer))
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept {
        cuda_api().free_async(cuda_address(pointer), stream_);
    }
    CUstream stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }

  private:
    CUstream stream_ = nullptr;
    int device_ = 0;
};

/** The byte specialization device storage is placed through. */
using cuda_device_allocator_t = cuda_device_allocator<byte_t>;

/**
 *  @brief The device twin of @ref arena_allocator: carves device bytes from one stream's run.
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

    cuda_arena_allocator(arena_t& arena, CUstream stream, int device) noexcept
        : arena_(arena), stream_(stream), device_(device) {}

    /** The rebinding conversion @c std::allocator_traits performs - the identity rides
     *  along, and the run is shared rather than copied. */
    template <typename other_>
    constexpr cuda_arena_allocator(cuda_arena_allocator<other_> const& other) noexcept
        : arena_(other.arena()), stream_(other.stream()), device_(other.device()) {}

    value_type* allocate(std::size_t count) noexcept {
        // Host-side pointer arithmetic over device bytes: the carve never touches them.
        return arena_allocator<value_type_, cuda_device_allocator_t> {arena_}.allocate(count);
    }

    /** A carve is never returned; the run's owner rewinds it between calls. */
    void deallocate(value_type*, std::size_t) noexcept {}

    CUstream stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }

    /** The run every rebound copy shares. */
    arena_t& arena() const noexcept { return arena_; }
    bool operator==(cuda_arena_allocator const& other) const noexcept { return &arena_ == &other.arena_; }

  private:
    arena_t& arena_;
    CUstream stream_ = nullptr;
    int device_ = 0;
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
    cuda_pinned_allocator(CUstream stream, int device) noexcept : stream_(stream), device_(device) {}
    template <typename other_>
    cuda_pinned_allocator(cuda_pinned_allocator<other_> const& other) noexcept
        : stream_(other.stream()), device_(other.device()) {}

    value_type* allocate(std::size_t count) noexcept {
        cuda_api_t const& api = cuda_api();
        void* pointer = nullptr;
        return api && api.allocate_host(&pointer, count * sizeof(value_type), CU_MEMHOSTALLOC_PORTABLE) == CUDA_SUCCESS
                   ? static_cast<value_type*>(pointer)
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept { cuda_api().free_host(pointer); }
    CUstream stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }

  private:
    CUstream stream_ = nullptr;
    int device_ = 0;
};

/** The byte specialization a host bounce is pinned through. */
using cuda_pinned_allocator_t = cuda_pinned_allocator<byte_t>;

/** Owned non-blocking stream, born in the current context; destruction drains it first. */
struct cuda_stream_t {
    cuda_stream_t() noexcept {
        if (cuda_api_t const& api = cuda_api(); api) api.stream_create(&handle_, CU_STREAM_NON_BLOCKING);
    }

    /** The current context's legacy default stream, which nothing here owns or destroys. */
    explicit cuda_stream_t(std::nullptr_t) noexcept {}
    cuda_stream_t(cuda_stream_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    cuda_stream_t(cuda_stream_t const&) = delete;
    cuda_stream_t& operator=(cuda_stream_t const&) = delete;
    ~cuda_stream_t() noexcept {
        if (handle_) cuda_api().stream_synchronize(handle_), cuda_api().stream_destroy(handle_);
    }

    operator CUstream() const noexcept { return handle_; }

  private:
    CUstream handle_ = nullptr;
};

/** Device memory that one call at a time rewinds and refills, on one device: the NVIDIA arm of
 *  @ref gpu_arena_t, where the C layer lands a VRAM call's outputs. */
struct cuda_device_arena_t {
    explicit cuda_device_arena_t(int device) noexcept
        : memory_(cuda_device_allocator_t {nullptr, device}), device_(device) {}

    /** Carves @p count values on the arena's device, or null when it cannot grow. */
    template <typename value_type_>
    value_type_* allocate(std::size_t count) noexcept {
        if (cuda_set_device(device_) != CUDA_SUCCESS) return nullptr;
        return reinterpret_cast<value_type_*>(memory_.allocate(count * sizeof(value_type_), default_alignment_k));
    }

    /** Whether @p address lies in memory handed out since the last @ref clear. */
    bool contains(void const* address) const noexcept { return memory_.contains(address); }

    /** Rewinds the arena, so the previous call's outputs die. */
    void clear() noexcept { memory_.clear(); }

    /** Synchronously copies @p bytes of host memory at @p source to @p destination in the arena. */
    status_t upload(void* destination, void const* source, std::size_t bytes) noexcept {
        if (cuda_set_device(device_) != CUDA_SUCCESS) return status_t::device_error_k;
        return cuda_copy(destination, source, bytes) == CUDA_SUCCESS ? status_t::success_k : status_t::device_error_k;
    }

    int device() const noexcept { return device_; }

  private:
    growing_arena<cuda_device_allocator_t> memory_;
    int device_ = 0;
};

/** The streams a bulk verb advances on and the scratch they walk with - one of each per
 *  device. Work is queued, never waited on, so one host thread drives every device. */
struct cuda_executor_t {

    /** Where a call names this executor, the C layer lands outputs in this arena. */
    using device_arena_t = cuda_device_arena_t;
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::nvidia_k;

    /** One device's run of scratch, carved and rewound between calls; sized once by
     *  @ref reserve, since relocation would be a host loop over device bytes. */
    using arena_t = growing_arena<cuda_device_allocator_t>;

    /** One call's hold on the executor, released when it dies. */
    using claim_t = std::unique_lock<fu::spin_mutex_t>;

    /** One device's stream and run; the arena is declared last so it frees onto a live
     *  stream. */
    struct stream_t {

        /** The device this stream was opened on - placement and launch target both. */
        int device = 0;

        /** The ordering identity every transfer and launch on this device rides. */
        cuda_stream_t handle;

        /** The scratch a walk carves its state from, sized once by @ref reserve. */
        arena_t arena;

        /** Takes @p handle, opened on @p device, and binds the arena to it. */
        stream_t(int device, cuda_stream_t&& handle) noexcept
            : device(device), handle(std::move(handle)), arena(cuda_device_allocator_t {this->handle, device}) {}
    };

    cuda_executor_t() = default;
    cuda_executor_t(cuda_executor_t&& other) noexcept { swap(other); }
    cuda_executor_t& operator=(cuda_executor_t&& other) noexcept { return swap(other), *this; }
    cuda_executor_t(cuda_executor_t const&) = delete;
    cuda_executor_t& operator=(cuda_executor_t const&) = delete;

    /** Exchanges the stream tables; @c held_ stays, since a move with a call in flight is
     *  misuse. */
    void swap(cuda_executor_t& other) noexcept { std::swap(streams_, other.streams_); }

    /** Claims the executor for one call; a second concurrent caller gets @c contended_k. */
    expected<claim_t> claim() noexcept {
        if (!held_.lock_if_free()) return {{}, status_t::contended_k};
        return {claim_t {held_, std::adopt_lock}, status_t::success_k};
    }

    /** Takes over streams the caller already opened, owned and destroyed here from now on. */
    static expected<cuda_executor_t> acquire(growing_buffer<stream_t>&& streams) noexcept {
        if (streams.empty()) return {{}, status_t::config_mismatch_k};
        cuda_executor_t executor;
        executor.streams_ = std::move(streams);
        return {std::move(executor), status_t::success_k};
    }

    /** Opens one stream per named device, binding each first, since a stream is born on
     *  the current device. */
    static expected<cuda_executor_t> spawn(std::span<int const> devices) noexcept {
        growing_buffer<stream_t> streams {global_allocator};
        if (failed(streams.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int device : devices) {
            if (cuda_set_device(device) != CUDA_SUCCESS) return {{}, status_t::device_error_k};
            stream_t opened {device, cuda_stream_t {}};
            if (!opened.handle) return {{}, status_t::device_error_k};
            streams.push_back_reserved(std::move(opened));
        }
        return acquire(std::move(streams));
    }

    /** The twin of @ref spawn over each named device's legacy default stream, for a call that names
     *  no executor; every use binds its device first, since a null stream is the current one's. */
    static expected<cuda_executor_t> on_default_streams(std::span<int const> devices) noexcept {
        growing_buffer<stream_t> streams {global_allocator};
        if (failed(streams.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int device : devices) {
            if (cuda_set_device(device) != CUDA_SUCCESS) return {{}, status_t::device_error_k};
            streams.push_back_reserved(stream_t {device, cuda_stream_t {nullptr}});
        }
        return acquire(std::move(streams));
    }

    /** Claims @p bytes_per_device of scratch on every stream, ahead of the fan-out that maps state
     *  onto it; the old run frees in stream order behind whatever still rides it. */
    status_t reserve(std::size_t bytes_per_device) noexcept {
        if (streams_.empty()) return status_t::uninitialized_k;
        for (stream_t& opened : streams_) {
            opened.arena.clear();
            if (failed(opened.arena.reserve_exactly(bytes_per_device))) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** Waits for every transfer and launch queued on every stream - the fleet's completion; binds
     *  each device first, since a null stream is the current device's. */
    status_t drain() noexcept {
        for (stream_t& opened : streams_)
            if (cuda_set_device(opened.device) != CUDA_SUCCESS ||
                cuda_api().stream_synchronize(opened.handle) != CUDA_SUCCESS)
                return status_t::device_error_k;
        return status_t::success_k;
    }

    /** The stream driving @p index, in the order the devices were named. */
    stream_t& stream(std::size_t index) noexcept { return streams_[index]; }

    /** The device @p index drives - what a caller pairs its shards against. */
    int device(std::size_t index) const noexcept { return streams_[index].device; }

    /** Devices this executor drives. */
    std::size_t size() const noexcept { return streams_.size(); }

  private:

    /** One stream per named device, in the caller's order. */
    growing_buffer<stream_t> streams_ {global_allocator};

    /** Held by the one caller driving the streams; never swapped. */
    fu::spin_mutex_t held_;
};

/** The @c std::unique_ptr deleter carrying the allocator and the element count. */
template <cuda_allocator allocator_>
struct cuda_deleter {
    allocator_ allocator;
    std::size_t count = 0;

    void operator()(typename allocator_::value_type* pointer) noexcept { allocator.deallocate(pointer, count); }
};

/** Owned device or pinned allocation; host code takes @c .get() for views and transfers. */
template <typename value_type_, cuda_allocator allocator_ = cuda_device_allocator<value_type_>>
using cuda_unique_ptr = std::unique_ptr<value_type_, cuda_deleter<allocator_>>;

/** The @c std::make_unique analog, null on exhaustion; the bytes arrive uninitialized. */
template <cuda_allocator allocator_>
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
        : handle_(std::exchange(other.handle_, nullptr)), device_(other.device_) {}
    cuda_module_t& operator=(cuda_module_t&& other) noexcept {
        std::swap(handle_, other.handle_), std::swap(device_, other.device_);
        return *this;
    }
    cuda_module_t(cuda_module_t const&) = delete;
    cuda_module_t& operator=(cuda_module_t const&) = delete;
    ~cuda_module_t() noexcept {
        if (handle_ && cuda_set_device(device_) == CUDA_SUCCESS) cuda_api().module_unload(handle_);
    }

    /** Loads @p fatbin onto device @p device, binding it first, since a module loads into the
     *  current context; the driver picks the cubin matching the device, or JIT-compiles its PTX. */
    static expected<cuda_module_t> load(int device, bytes_view_t fatbin) noexcept {
        cuda_module_t module;
        module.device_ = device;
        if (cuda_set_device(device) != CUDA_SUCCESS ||
            cuda_api().module_load_data(&module.handle_, fatbin.data()) != CUDA_SUCCESS)
            return {{}, status_t::device_error_k};
        return {std::move(module), status_t::success_k};
    }

    /** The kernel exported as @p name, or null when the fatbin has none. */
    CUfunction function(char const* name) const noexcept {
        CUfunction function = nullptr;
        return cuda_api().module_get_function(&function, handle_, name) == CUDA_SUCCESS ? function : nullptr;
    }

  private:
    CUmodule handle_ = nullptr;
    int device_ = 0;
};

/** Launches @p function over @p blocks of @p threads on @p stream - the only launch path - passing
 *  each argument by address, as the driver takes them, so each must already be the kernel's own
 *  type; answers the launch's own error, like a bad configuration or too much shared memory. */
template <typename... argument_types_>
status_t cuda_launch(CUfunction function, u32_t blocks, u32_t threads, u32_t shared_bytes, CUstream stream,
                     argument_types_... arguments) noexcept {
    static_assert((std::is_trivially_copyable_v<argument_types_> && ...), "Kernels take their arguments by bytes");
    if (!function) return status_t::device_error_k;
    void* addresses[] = {static_cast<void*>(&arguments)...};
    CUlaunchConfig config {};
    config.gridDimX = blocks, config.gridDimY = 1, config.gridDimZ = 1;
    config.blockDimX = threads, config.blockDimY = 1, config.blockDimZ = 1;
    config.sharedMemBytes = shared_bytes;
    config.hStream = stream;
    return cuda_api().launch_kernel(&config, function, addresses, nullptr) == CUDA_SUCCESS ? status_t::success_k
                                                                                           : status_t::device_error_k;
}

static_assert(cuda_reservable_allocator<cuda_virtual_memory_allocator<byte_t>>);
static_assert(cuda_reservable_allocator<cuda_managed_allocator<byte_t>>);
static_assert(cuda_allocator<cuda_device_allocator_t>);
static_assert(cuda_allocator<cuda_arena_allocator_t>);
static_assert(cuda_allocator<cuda_pinned_allocator_t>);

/** The class whose unit carries cubins for a device of compute capability @p arch; any generation
 *  without a class of its own runs the Ampere unit, natively on 8.x and through its PTX past
 *  that. */
constexpr gpu_class_t gpu_class_of(cuda_arch_t arch) noexcept {
    switch (arch / 100) {
    case 8: return arch >= cuda_arch_ada_k ? gpu_class_t::nvidia_ada_k : gpu_class_t::nvidia_ampere_k;
    case 9: return gpu_class_t::nvidia_hopper_k;
    case 10: return gpu_class_t::nvidia_blackwell_k;
    case 12: return gpu_class_t::nvidia_blackwellrtx_k;
    default: return gpu_class_t::nvidia_ampere_k;
    }
}

/** Every NVIDIA device the driver reaches, in CUDA ordinal order, listed once per process; none
 *  without a driver. An older device than the Ampere floor is left out, since no unit runs on
 *  it. */
inline std::span<gpu_device_t const> nvidia_gpu_devices() noexcept {
    struct listing_t {
        gpu_device_t devices[gpu_devices_capacity_k];
        std::size_t count = 0;
    };
    static listing_t const listing = [] {
        listing_t found;
        cuda_api_t const& api = cuda_api();
        int devices_count = 0;
        if (!api || api.get_device_count(&devices_count) != CUDA_SUCCESS) return found;
        for (int ordinal = 0; ordinal != devices_count && found.count != gpu_devices_capacity_k; ++ordinal) {
            auto const [specs, status] = cuda_device_specs_t::query(ordinal);
            if (failed(status) || specs.arch() < cuda_arch_ampere_k) continue;
            gpu_device_t& device = found.devices[found.count++];
            device = {gpu_vendor_t::nvidia_k, static_cast<u32_t>(ordinal), gpu_class_of(specs.arch()),
                      static_cast<u32_t>(specs.multiprocessor_count), specs.vram_bytes};
            CUdevice handle = 0;
            if (api.get_device(&handle, ordinal) == CUDA_SUCCESS)
                api.get_device_name(device.name, static_cast<int>(sizeof(device.name)), handle);
        }
        return found;
    }();
    return {listing.devices, listing.count};
}

} // namespace unum::usearch::utilities
