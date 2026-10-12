/**
 *  @file include/usearch/utilities/vram_amd_device.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Host-side vocabulary for driving one AMD GPU - the HIP runtime, specs, streams, memory.
 *
 *  The library never links @c libamdhip64: @ref hip_api opens it at first use and resolves the
 *  entry points USearch calls into one table, which stays empty where the library is absent, so AMD
 *  then lists no devices and the binary still loads. Kernels reach a device as code objects
 *  compiled ahead of time and embedded in the library, loaded by @ref hip_module_t and launched by
 *  name, since HIP-Clang's fat-binary registration would link the runtime outright.
 *
 *  Plain C++ needing no ROCm SDK: the few runtime types, constants and prototypes USearch calls are
 *  spelled here, copied from HIP's stable ABI, so any host compiler builds it. The warp policy the
 *  shared walk code is written against lives beside the kits in @c vram_amd_metric.hip.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <atomic>      // `std::memory_order` - the committed extent
#include <memory>      // `std::unique_ptr` for owned device allocations
#include <mutex>       // `std::unique_lock`, `std::adopt_lock` - the executor's one-caller hold
#include <span>        // `std::span` - the device list an executor opens over, a code object's bytes
#include <type_traits> // `std::is_trivially_copyable_v` - a launch's arguments travel as bytes
#include <utility>     // `std::move`, `std::exchange`

#include <forkunion/topology.hpp> // `fu::volume_ram` - the managed storage's host-memory budget
#include <forkunion/types.hpp>    // `fu::round_up_to_multiple`, `fu::spin_mutex_t` - the hold

#include "usearch/utilities/blocking_files.hpp" // `shared_library_t` - the HIP runtime opened at first use
#include "usearch/utilities/ram_buffers.hpp"    // `growing_buffer` - an executor's table of streams
#include "usearch/utilities/types.hpp"          // `gpu_device_t`, `gpu_devices_capacity_k`

namespace unum::usearch::utilities {

#pragma region Runtime ABI

/** The runtime's status word, @c hipError_t, naming the few values USearch tells apart; any other
 *  is a failure. */
enum hip_result_t : int {
    hip_success_k = 0,
    hip_error_not_initialized_k = 3,
    hip_error_not_ready_k = 600,
};

/** The runtime's opaque handles: @c hipStream_t, @c hipModule_t, @c hipFunction_t and
 *  @c hipMemGenericAllocationHandle_t. */
using hip_stream_handle_t = struct hip_stream_opaque_t*;
using hip_module_handle_t = struct hip_module_opaque_t*;
using hip_function_handle_t = struct hip_function_opaque_t*;
using hip_allocation_handle_t = struct hip_allocation_opaque_t*;

/** The @c hipMemcpyKind of a copy; unified addressing tells the sides apart under the default. */
enum hip_copy_kind_t : int {
    hip_copy_host_to_device_k = 1,
    hip_copy_device_to_host_k = 2,
    hip_copy_device_to_device_k = 3,
    hip_copy_default_k = 4,
};

/** The @c hipDeviceAttribute_t values USearch queries; on AMD the compute capability is the
 *  @c gfx processor's major and minor version. */
enum hip_device_attribute_t : int {
    hip_attribute_compute_capability_major_k = 23,
    hip_attribute_compute_capability_minor_k = 61,
    hip_attribute_multiprocessor_count_k = 63,
    hip_attribute_shared_memory_per_block_optin_k = 75,
};

/** The @c hipMemoryAdvise values USearch gives managed memory. */
enum hip_memory_advice_t : int {
    hip_advise_preferred_location_k = 3,
    hip_advise_accessed_by_k = 5,
};

/** Where memory lives, @c hipMemLocation: a device ordinal. */
struct hip_memory_location_t {
    inline static constexpr int device_k = 1;
    int type = 0;
    int id = 0;
};

/** How @c hipMemCreate backs one physical slab, @c hipMemAllocationProp. */
struct hip_allocation_properties_t {
    inline static constexpr int pinned_k = 1;
    int type = 0;
    int requested_handle_type = 0;
    hip_memory_location_t location;
    void* win32_handle_metadata = nullptr;
    struct {
        unsigned char compression_type = 0;
        unsigned char gpu_direct_rdma_capable = 0;
        unsigned short usage = 0;
    } allocation_flags;
};

/** Who may touch a mapped range and how, @c hipMemAccessDesc. */
struct hip_access_descriptor_t {
    inline static constexpr int read_write_k = 3;
    hip_memory_location_t location;
    int flags = 0;
};

/** The flags USearch passes: @c hipStreamNonBlocking, @c hipHostMallocPortable,
 *  @c hipMemAttachGlobal, @c hipMemAllocationGranularityMinimum and @c hipCpuDeviceId. */
inline constexpr unsigned hip_stream_non_blocking_k = 1;
inline constexpr unsigned hip_host_portable_k = 1;
inline constexpr unsigned hip_attach_global_k = 1;
inline constexpr int hip_granularity_minimum_k = 0;
inline constexpr int hip_host_device_k = -1;

static_assert(sizeof(hip_allocation_properties_t) == 32 && sizeof(hip_access_descriptor_t) == 12,
              "HIP's 64-bit layouts");

#pragma endregion Runtime ABI

/**
 *  @brief The HIP entry points USearch calls, resolved from @c libamdhip64 on first use.
 *
 *  Every entry is null when the library is absent or lacks any of them, which leaves AMD with no
 *  devices; each spells the C prototype of the exported symbol it is resolved by.
 */
struct hip_api_t {
    hip_result_t (*get_device_count)(int*) = nullptr;
    hip_result_t (*get_device)(int*, int) = nullptr;
    hip_result_t (*get_device_attribute)(int*, hip_device_attribute_t, int) = nullptr;
    hip_result_t (*get_device_name)(char*, int, int) = nullptr;
    hip_result_t (*get_device_memory)(std::size_t*, int) = nullptr;
    hip_result_t (*set_device)(int) = nullptr;
    hip_result_t (*malloc_async)(void**, std::size_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*free_async)(void*, hip_stream_handle_t) = nullptr;
    hip_result_t (*malloc_managed)(void**, std::size_t, unsigned) = nullptr;
    hip_result_t (*memory_advise)(void const*, std::size_t, hip_memory_advice_t, int) = nullptr;
    hip_result_t (*free_managed)(void*) = nullptr;
    hip_result_t (*host_malloc)(void**, std::size_t, unsigned) = nullptr;
    hip_result_t (*host_free)(void*) = nullptr;
    hip_result_t (*copy)(void*, void const*, std::size_t, hip_copy_kind_t) = nullptr;
    hip_result_t (*copy_async)(void*, void const*, std::size_t, hip_copy_kind_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*copy_2d_async)(void*, std::size_t, void const*, std::size_t, std::size_t, std::size_t,
                                  hip_copy_kind_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*fill)(void*, int, std::size_t) = nullptr;
    hip_result_t (*fill_async)(void*, int, std::size_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*fill_words_async)(void*, int, std::size_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*stream_create_with_flags)(hip_stream_handle_t*, unsigned) = nullptr;
    hip_result_t (*stream_destroy)(hip_stream_handle_t) = nullptr;
    hip_result_t (*stream_synchronize)(hip_stream_handle_t) = nullptr;
    hip_result_t (*stream_query)(hip_stream_handle_t) = nullptr;
    hip_result_t (*module_load_data)(hip_module_handle_t*, void const*) = nullptr;
    hip_result_t (*module_unload)(hip_module_handle_t) = nullptr;
    hip_result_t (*module_get_function)(hip_function_handle_t*, hip_module_handle_t, char const*) = nullptr;
    hip_result_t (*module_launch_kernel)(hip_function_handle_t, unsigned, unsigned, unsigned, unsigned, unsigned,
                                         unsigned, unsigned, hip_stream_handle_t, void**, void**) = nullptr;
    hip_result_t (*memory_granularity)(std::size_t*, hip_allocation_properties_t const*, int) = nullptr;
    hip_result_t (*memory_address_reserve)(void**, std::size_t, std::size_t, void*, unsigned long long) = nullptr;
    hip_result_t (*memory_address_free)(void*, std::size_t) = nullptr;
    hip_result_t (*memory_create)(hip_allocation_handle_t*, std::size_t, hip_allocation_properties_t const*,
                                  unsigned long long) = nullptr;
    hip_result_t (*memory_release)(hip_allocation_handle_t) = nullptr;
    hip_result_t (*memory_map)(void*, std::size_t, std::size_t, hip_allocation_handle_t, unsigned long long) = nullptr;
    hip_result_t (*memory_unmap)(void*, std::size_t) = nullptr;
    hip_result_t (*memory_set_access)(void*, std::size_t, hip_access_descriptor_t const*, std::size_t) = nullptr;

    /** Whether the runtime loaded with every entry resolved. */
    explicit operator bool() const noexcept { return memory_set_access != nullptr; }

    /** Opens the runtime and resolves every entry, or answers the empty table. The library stays
     *  open for the life of the process, as the table points into it. */
    static hip_api_t load() noexcept {
        // Each system names the runtime its own way, and opening another's name merely fails.
        shared_library_t library;
        for (char const* name :
             {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "amdhip64_7.dll", "amdhip64_6.dll"})
            if ((library = shared_library_t::open(name))) break;
        if (!library) return {};
        hip_api_t runtime;
        bool const resolved = //
            library.resolve("hipGetDeviceCount", runtime.get_device_count) &&
            library.resolve("hipDeviceGet", runtime.get_device) &&
            library.resolve("hipDeviceGetAttribute", runtime.get_device_attribute) &&
            library.resolve("hipDeviceGetName", runtime.get_device_name) &&
            library.resolve("hipDeviceTotalMem", runtime.get_device_memory) &&
            library.resolve("hipSetDevice", runtime.set_device) &&
            library.resolve("hipMallocAsync", runtime.malloc_async) &&
            library.resolve("hipFreeAsync", runtime.free_async) &&
            library.resolve("hipMallocManaged", runtime.malloc_managed) &&
            library.resolve("hipMemAdvise", runtime.memory_advise) &&
            library.resolve("hipFree", runtime.free_managed) && library.resolve("hipHostMalloc", runtime.host_malloc) &&
            library.resolve("hipHostFree", runtime.host_free) && library.resolve("hipMemcpy", runtime.copy) &&
            library.resolve("hipMemcpyAsync", runtime.copy_async) &&
            library.resolve("hipMemcpy2DAsync", runtime.copy_2d_async) && library.resolve("hipMemset", runtime.fill) &&
            library.resolve("hipMemsetAsync", runtime.fill_async) &&
            library.resolve("hipMemsetD32Async", runtime.fill_words_async) &&
            library.resolve("hipStreamCreateWithFlags", runtime.stream_create_with_flags) &&
            library.resolve("hipStreamDestroy", runtime.stream_destroy) &&
            library.resolve("hipStreamSynchronize", runtime.stream_synchronize) &&
            library.resolve("hipStreamQuery", runtime.stream_query) &&
            library.resolve("hipModuleLoadData", runtime.module_load_data) &&
            library.resolve("hipModuleUnload", runtime.module_unload) &&
            library.resolve("hipModuleGetFunction", runtime.module_get_function) &&
            library.resolve("hipModuleLaunchKernel", runtime.module_launch_kernel) &&
            library.resolve("hipMemGetAllocationGranularity", runtime.memory_granularity) &&
            library.resolve("hipMemAddressReserve", runtime.memory_address_reserve) &&
            library.resolve("hipMemAddressFree", runtime.memory_address_free) &&
            library.resolve("hipMemCreate", runtime.memory_create) &&
            library.resolve("hipMemRelease", runtime.memory_release) &&
            library.resolve("hipMemMap", runtime.memory_map) && library.resolve("hipMemUnmap", runtime.memory_unmap) &&
            library.resolve("hipMemSetAccess", runtime.memory_set_access);
        return resolved ? runtime : hip_api_t {};
    }
};

/** The one table every AMD call goes through, filled on first use by whichever thread is first. */
inline hip_api_t const& hip_api() noexcept {
    static hip_api_t const api = hip_api_t::load();
    return api;
}

/** Makes HIP ordinal @p ordinal current on the calling thread, which every stream, allocation and
 *  launch after it lands on - the twin of @c cuda_set_device. The runtime is asked only when this
 *  thread last selected another device through here, so a caller switching devices on its own
 *  must not interleave with USearch calls on the same thread. */
inline hip_result_t hip_set_device(int ordinal) noexcept {
    thread_local int current = -1;
    hip_api_t const& runtime = hip_api();
    if (!runtime) return hip_error_not_initialized_k;
    if (current == ordinal) return hip_success_k;
    hip_result_t const selected = runtime.set_device(ordinal);
    if (selected == hip_success_k) current = ordinal;
    return selected;
}

/** The class whose kernels run on an AMD GPU of @c gfx processor version @p major. @p minor: 9.4
 *  is MI300's @c gfx942, 9.5 is MI350's @c gfx950, and 12.5 is MI400's @c gfx1250 and @c gfx1251;
 *  any other has no code object to load. */
constexpr gpu_class_t hip_gpu_class_of(int major, int minor) noexcept {
    if (major == 9 && minor == 4) return gpu_class_t::amd_cdna3_k;
    if (major == 9 && minor == 5) return gpu_class_t::amd_cdna4_k;
    if (major == 12 && minor == 5) return gpu_class_t::amd_cdna5_k;
    return gpu_class_t::unknown_k;
}

/** Lanes of a wavefront on @p gpu_class: 64 on CDNA3 and CDNA4, 32 on CDNA5. */
constexpr u32_t hip_wave_lanes(gpu_class_t gpu_class) noexcept {
    return gpu_class == gpu_class_t::amd_cdna5_k ? 32u : 64u;
}

/** One AMD GPU's hardware specification - the facts runtime dispatch consults; hardware only, never
 *  derived policy. */
struct hip_device_specs_t {

    /** The device ordinal every HIP call routes through. */
    int ordinal = 0;

    /** Compute-unit count - the launch-grid currency. */
    int multiprocessor_count = 0;

    /** Total device-resident memory in bytes. */
    std::size_t vram_bytes = 0;

    /** Widest dynamic shared memory one block can be granted at launch. */
    std::size_t shared_memory_per_block_bytes = 0;

    /** The class whose kernels run here, unknown for an architecture USearch ships none for. */
    gpu_class_t gpu_class = gpu_class_t::unknown_k;

    /** Queries the runtime for @p ordinal's specification; meaningful only on @c success_k. */
    static expected<hip_device_specs_t> query(int ordinal) noexcept {
        hip_api_t const& runtime = hip_api();
        hip_device_specs_t specs {ordinal};
        int device = 0, major = 0, minor = 0, shared_memory_per_block = 0;
        if (!runtime || runtime.get_device(&device, ordinal) != hip_success_k ||
            runtime.get_device_attribute(&specs.multiprocessor_count, hip_attribute_multiprocessor_count_k, ordinal) !=
                hip_success_k ||
            runtime.get_device_attribute(&major, hip_attribute_compute_capability_major_k, ordinal) != hip_success_k ||
            runtime.get_device_attribute(&minor, hip_attribute_compute_capability_minor_k, ordinal) != hip_success_k ||
            runtime.get_device_attribute(&shared_memory_per_block, hip_attribute_shared_memory_per_block_optin_k,
                                         ordinal) != hip_success_k ||
            runtime.get_device_memory(&specs.vram_bytes, device) != hip_success_k)
            return {{ordinal}, status_t::device_error_k};
        specs.shared_memory_per_block_bytes = static_cast<std::size_t>(shared_memory_per_block);
        specs.gpu_class = hip_gpu_class_of(major, minor);
        return {specs, status_t::success_k};
    }
};

/** Every AMD GPU the HIP runtime reaches, in HIP ordinal order, listed once per process for the
 *  neutral listing to append after NVIDIA's; none without @c libamdhip64. A device of an
 *  architecture no unit runs on is left out. */
inline std::span<gpu_device_t const> amd_gpu_devices() noexcept {
    struct listing_t {
        gpu_device_t devices[gpu_devices_capacity_k];
        std::size_t count = 0;
    };
    static listing_t const listing = [] {
        listing_t found;
        hip_api_t const& runtime = hip_api();
        int devices_count = 0;
        if (!runtime || runtime.get_device_count(&devices_count) != hip_success_k) return found;
        for (int ordinal = 0; ordinal != devices_count && found.count != gpu_devices_capacity_k; ++ordinal) {
            auto const [specs, status] = hip_device_specs_t::query(ordinal);
            if (failed(status) || specs.gpu_class == gpu_class_t::unknown_k) continue;
            gpu_device_t& device = found.devices[found.count++];
            device = {gpu_vendor_t::amd_k, static_cast<u32_t>(ordinal), specs.gpu_class,
                      static_cast<u32_t>(specs.multiprocessor_count), specs.vram_bytes};
            int handle = 0;
            if (runtime.get_device(&handle, ordinal) == hip_success_k)
                runtime.get_device_name(device.name, static_cast<int>(sizeof(device.name)), handle);
        }
        return found;
    }();
    return {listing.devices, listing.count};
}

/** The address space an allocation lives in, in LLVM's sense: who can dereference the pointer and
 *  what transfer path moves its bytes; all three are @c global to device code. */
enum class hip_address_space_t : u32_t {

    /** Plain device memory; explicit host-device copies required. */
    device_k,

    /** Unified memory; migratable, host- and device-visible. */
    managed_k,

    /** Page-locked host memory; the DMA source & sink for staging. */
    host_pinned_k,
};

/** The device analog of @c std::allocator, with the ordering stream and the target device bound at
 *  construction; @c address_space_k fixes host visibility. */
template <typename allocator_>
concept hip_allocator = requires(allocator_ allocator, typename allocator_::value_type* pointer, std::size_t count) {
    typename allocator_::value_type;
    { allocator_::address_space_k } -> std::convertible_to<hip_address_space_t>;
    { allocator.allocate(count) } -> std::same_as<typename allocator_::value_type*>;
    { allocator.deallocate(pointer, count) } -> std::same_as<void>;
    { allocator.stream() } -> std::convertible_to<hip_stream_handle_t>;
    { allocator.device() } -> std::convertible_to<int>;
};

/** The flat storage backend contract: a reserve-once / commit-on-demand address range whose base
 *  never moves. Not a refinement of @c hip_allocator, since flat storage never frees mid-life. */
template <typename allocator_>
concept hip_reservable_allocator = requires(allocator_ allocator, std::size_t bytes, int device) {
    typename allocator_::value_type;
    { allocator_::address_space_k } -> std::convertible_to<hip_address_space_t>;
    { allocator.reserve_range(bytes, device) } -> std::same_as<status_t>;
    { allocator.commit(bytes) } -> std::same_as<status_t>;
    { allocator.base() } -> std::same_as<typename allocator_::value_type*>;
    { allocator.reserved_bytes() } -> std::same_as<std::size_t>;
    { allocator.committed_bytes() } -> std::same_as<std::size_t>;
    { allocator.release() } -> std::same_as<void>;
    { allocator.device() } -> std::convertible_to<int>;
};

/** Virtual-memory flat storage backend: a fixed address reservation backed by slabs mapped on
 *  demand, host-driven ahead of any kernel; each fresh slab is zeroed as it maps. */
template <typename value_type_ = byte_t>
struct hip_virtual_memory_allocator {
    using value_type = value_type_;
    inline static constexpr hip_address_space_t address_space_k = hip_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = hip_virtual_memory_allocator<other_>;
    };

    /** Preferred physical slab, rounded to the runtime's granularity at reservation. */
    inline static constexpr std::size_t preferred_slab_bytes_k = 128u << 20;

    byte_t* base_ = nullptr;
    std::size_t virtual_address_bytes_ = 0;

    /** Bytes mapped so far - the one grower advances it with a release, and any thread may read it
     *  through @c committed_bytes with an acquire; @c mutable for the reference. */
    std::size_t mapped_bytes_ = 0;
    std::size_t slab_bytes_ = 0;
    int device_ = 0;

    hip_virtual_memory_allocator() = default;

    /** Names the GPU the future reservation pins to - placement only, no runtime call. */
    explicit hip_virtual_memory_allocator(int device) noexcept : device_(device) {}

    /** Copies the placement only, never a live reservation. */
    hip_virtual_memory_allocator(hip_virtual_memory_allocator const& other) noexcept : device_(other.device_) {}

    /** Takes over a live reservation, so a table of them relocates without remapping. */
    hip_virtual_memory_allocator(hip_virtual_memory_allocator&& other) noexcept
        : base_(std::exchange(other.base_, nullptr)),
          virtual_address_bytes_(std::exchange(other.virtual_address_bytes_, 0)),
          mapped_bytes_(std::exchange(other.mapped_bytes_, 0)), slab_bytes_(other.slab_bytes_), device_(other.device_) {
    }
    template <typename other_>
    hip_virtual_memory_allocator(hip_virtual_memory_allocator<other_> const& other) noexcept : device_(other.device_) {}
    hip_virtual_memory_allocator& operator=(hip_virtual_memory_allocator const&) = delete;
    ~hip_virtual_memory_allocator() noexcept { release(); }

    /** Reserves @p virtual_address_bytes of virtual address space on @p device - no physical memory
     *  yet, so the ceiling costs nothing and should be generous. */
    status_t reserve_range(std::size_t virtual_address_bytes, int device) noexcept {
        hip_api_t const& runtime = hip_api();
        hip_allocation_properties_t const properties = allocation_properties_(device);
        std::size_t granularity = 0;
        if (!runtime ||
            runtime.memory_granularity(&granularity, &properties, hip_granularity_minimum_k) != hip_success_k)
            return status_t::device_error_k;
        slab_bytes_ = max_of(granularity, preferred_slab_bytes_k / granularity * granularity);
        virtual_address_bytes_ = fu::round_up_to_multiple(virtual_address_bytes, slab_bytes_);
        device_ = device;
        void* base = nullptr;
        if (runtime.memory_address_reserve(&base, virtual_address_bytes_, 0, nullptr, 0) != hip_success_k)
            return status_t::bad_alloc_k;
        base_ = static_cast<byte_t*>(base);
        return status_t::success_k;
    }

    /** Maps and zeroes physical slabs until at least @p bytes are usable; never unmaps early. Each
     *  handle is released right after mapping, since the mapping alone keeps the memory alive. */
    status_t commit(std::size_t bytes) noexcept {
        if (bytes > virtual_address_bytes_) return status_t::overflow_k;
        hip_api_t const& runtime = hip_api();
        hip_allocation_properties_t const properties = allocation_properties_(device_);
        hip_access_descriptor_t access;
        access.location = properties.location;
        access.flags = hip_access_descriptor_t::read_write_k;
        while (mapped_bytes_ < bytes) {
            hip_allocation_handle_t handle = nullptr;
            if (runtime.memory_create(&handle, slab_bytes_, &properties, 0) != hip_success_k)
                return status_t::bad_alloc_k;
            byte_t* const slab = base_ + mapped_bytes_;
            bool const mapped = runtime.memory_map(slab, slab_bytes_, 0, handle, 0) == hip_success_k &&
                                runtime.memory_set_access(slab, slab_bytes_, &access, 1) == hip_success_k;
            runtime.memory_release(handle);
            if (!mapped) return status_t::device_error_k;
            if (runtime.fill(slab, 0, slab_bytes_) != hip_success_k) return status_t::device_error_k;
            standard_atomic_ref<std::size_t>(mapped_bytes_)
                .store(mapped_bytes_ + slab_bytes_, std::memory_order_release);
        }
        return status_t::success_k;
    }

    value_type* base() const noexcept { return reinterpret_cast<value_type*>(base_); }
    std::size_t reserved_bytes() const noexcept { return virtual_address_bytes_; }
    std::size_t committed_bytes() const noexcept {
        return standard_atomic_ref<std::size_t const>(mapped_bytes_).load(std::memory_order_acquire);
    }
    int device() const noexcept { return device_; }

    void release() noexcept {
        if (mapped_bytes_) hip_api().memory_unmap(base_, mapped_bytes_);
        if (base_) hip_api().memory_address_free(base_, virtual_address_bytes_);
        base_ = nullptr, virtual_address_bytes_ = 0, mapped_bytes_ = 0;
    }

  private:
    static hip_allocation_properties_t allocation_properties_(int device) noexcept {
        hip_allocation_properties_t properties;
        properties.type = hip_allocation_properties_t::pinned_k;
        properties.location.type = hip_memory_location_t::device_k;
        properties.location.id = device;
        return properties;
    }
};

/** Unified-memory flat storage backend: one @c hipMallocManaged reservation for the whole ceiling,
 *  preferring the device and oversubscribable past VRAM. */
template <typename value_type_ = byte_t>
struct hip_managed_allocator {
    using value_type = value_type_;
    inline static constexpr hip_address_space_t address_space_k = hip_address_space_t::managed_k;
    template <typename other_>
    struct rebind {
        using other = hip_managed_allocator<other_>;
    };

    value_type* base_ = nullptr;
    std::size_t virtual_address_bytes_ = 0;
    std::size_t committed_bytes_ = 0; // ? Released by the one grower, acquired by any reader
    int device_ = 0;

    hip_managed_allocator() = default;

    /** Names the GPU the future reservation prefers - placement only, no runtime call. */
    explicit hip_managed_allocator(int device) noexcept : device_(device) {}

    /** Copies the placement configuration only, never a live reservation. */
    hip_managed_allocator(hip_managed_allocator const& other) noexcept : device_(other.device_) {}
    template <typename other_>
    hip_managed_allocator(hip_managed_allocator<other_> const& other) noexcept : device_(other.device_) {}
    hip_managed_allocator& operator=(hip_managed_allocator const&) = delete;
    ~hip_managed_allocator() noexcept { release(); }

    status_t reserve_range(std::size_t virtual_address_bytes, int device) noexcept {
        hip_api_t const& runtime = hip_api();
        device_ = device;
        virtual_address_bytes_ = virtual_address_bytes;
        void* base = nullptr;
        if (!runtime || runtime.malloc_managed(&base, virtual_address_bytes, hip_attach_global_k) != hip_success_k)
            return status_t::bad_alloc_k;
        base_ = static_cast<value_type*>(base);
        runtime.memory_advise(base, virtual_address_bytes, hip_advise_preferred_location_k, device);
        runtime.memory_advise(base, virtual_address_bytes, hip_advise_accessed_by_k, hip_host_device_k);
        return status_t::success_k;
    }

    status_t commit(std::size_t bytes) noexcept {
        if (bytes > virtual_address_bytes_) return status_t::overflow_k;
        if (bytes <= committed_bytes_) return status_t::success_k;
        // Managed pages arrive unzeroed; zeroing the fresh range also commits its residency.
        if (hip_api().fill(reinterpret_cast<byte_t*>(base_) + committed_bytes_, 0, bytes - committed_bytes_) !=
            hip_success_k)
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

    /** The reservation ceiling: half the host RAM, since managed reservations are real allocations;
     *  only this backend consults the topology. */
    static std::size_t budget_bytes() noexcept { return fu::volume_ram() / 2; }

    void release() noexcept {
        if (base_) hip_api().free_managed(base_);
        base_ = nullptr, virtual_address_bytes_ = 0, committed_bytes_ = 0;
    }
};

/** Stream-ordered device allocator for transient scratch; a null stream rides the default one. */
template <typename value_type_ = byte_t>
struct hip_device_allocator {
    using value_type = value_type_;
    inline static constexpr hip_address_space_t address_space_k = hip_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = hip_device_allocator<other_>;
    };

    hip_device_allocator() noexcept = default;
    hip_device_allocator(hip_stream_handle_t stream, int device) noexcept : stream_(stream), device_(device) {}
    template <typename other_>
    hip_device_allocator(hip_device_allocator<other_> const& other) noexcept
        : stream_(other.stream()), device_(other.device()) {}

    value_type* allocate(std::size_t count) noexcept {
        void* pointer = nullptr;
        return hip_api().malloc_async(&pointer, count * sizeof(value_type), stream_) == hip_success_k
                   ? static_cast<value_type*>(pointer)
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept { hip_api().free_async(pointer, stream_); }
    hip_stream_handle_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }

  private:
    hip_stream_handle_t stream_ = nullptr;
    int device_ = 0;
};

/** The byte specialization device storage is placed through. */
using hip_device_allocator_t = hip_device_allocator<byte_t>;

/**
 *  @brief The device twin of @ref arena_allocator: carves device bytes from one stream's run.
 *
 *  A spent run answers @c nullptr rather than spilling to a pool, so a mis-sized executor is a
 *  reported failure, never a silent allocation; releasing a carve does nothing, and the run is
 *  rewound by whoever owns it, between calls. A context wanting pool-backed scratch names
 *  @ref hip_device_allocator instead - the allocator type picks the behavior.
 */
template <typename value_type_ = byte_t>
struct hip_arena_allocator {
    using value_type = value_type_;
    using arena_t = growing_arena<hip_device_allocator_t>;
    inline static constexpr hip_address_space_t address_space_k = hip_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = hip_arena_allocator<other_>;
    };

    hip_arena_allocator(arena_t& arena, hip_stream_handle_t stream, int device) noexcept
        : arena_(arena), stream_(stream), device_(device) {}

    /** The rebinding conversion @c std::allocator_traits performs - the identity rides along, and
     *  the run is shared rather than copied. */
    template <typename other_>
    constexpr hip_arena_allocator(hip_arena_allocator<other_> const& other) noexcept
        : arena_(other.arena()), stream_(other.stream()), device_(other.device()) {}

    value_type* allocate(std::size_t count) noexcept {
        // Host-side pointer arithmetic over device bytes: the carve never touches them.
        return arena_allocator<value_type_, hip_device_allocator_t> {arena_}.allocate(count);
    }

    /** A carve is never returned; the run's owner rewinds it between calls. */
    void deallocate(value_type*, std::size_t) noexcept {}

    hip_stream_handle_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }

    /** The run every rebound copy shares. */
    arena_t& arena() const noexcept { return arena_; }
    bool operator==(hip_arena_allocator const& other) const noexcept { return &arena_ == &other.arena_; }

  private:
    arena_t& arena_;
    hip_stream_handle_t stream_ = nullptr;
    int device_ = 0;
};

/** The byte specialization a walk carves its scratch through. */
using hip_arena_allocator_t = hip_arena_allocator<byte_t>;

/** Page-locked host allocator - the DMA source and sink; pinning is synchronous, the stream is
 *  identity only. */
template <typename value_type_ = byte_t>
struct hip_pinned_allocator {
    using value_type = value_type_;
    inline static constexpr hip_address_space_t address_space_k = hip_address_space_t::host_pinned_k;
    template <typename other_>
    struct rebind {
        using other = hip_pinned_allocator<other_>;
    };

    hip_pinned_allocator() noexcept = default;
    hip_pinned_allocator(hip_stream_handle_t stream, int device) noexcept : stream_(stream), device_(device) {}
    template <typename other_>
    hip_pinned_allocator(hip_pinned_allocator<other_> const& other) noexcept
        : stream_(other.stream()), device_(other.device()) {}

    value_type* allocate(std::size_t count) noexcept {
        void* pointer = nullptr;
        return hip_api().host_malloc(&pointer, count * sizeof(value_type), hip_host_portable_k) == hip_success_k
                   ? static_cast<value_type*>(pointer)
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept { hip_api().host_free(pointer); }
    hip_stream_handle_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }

  private:
    hip_stream_handle_t stream_ = nullptr;
    int device_ = 0;
};

/** The byte specialization a host bounce is pinned through. */
using hip_pinned_allocator_t = hip_pinned_allocator<byte_t>;

/** Owned non-blocking stream, born on the current device; destruction drains it first. */
struct hip_stream_t {
    hip_stream_t() noexcept {
        if (hip_api_t const& runtime = hip_api(); runtime)
            runtime.stream_create_with_flags(&handle_, hip_stream_non_blocking_k);
    }

    /** The current device's legacy default stream, which nothing here owns or destroys. */
    explicit hip_stream_t(std::nullptr_t) noexcept {}
    hip_stream_t(hip_stream_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    hip_stream_t(hip_stream_t const&) = delete;
    hip_stream_t& operator=(hip_stream_t const&) = delete;
    ~hip_stream_t() noexcept {
        if (handle_) hip_api().stream_synchronize(handle_), hip_api().stream_destroy(handle_);
    }

    operator hip_stream_handle_t() const noexcept { return handle_; }

  private:
    hip_stream_handle_t handle_ = nullptr;
};

/** Device memory that one call at a time rewinds and refills, on one AMD device: the AMD arm of
 *  @c gpu_arena_t, where the C layer lands a VRAM call's outputs. */
struct hip_device_arena_t {
    explicit hip_device_arena_t(int vendor_ordinal) noexcept
        : memory_(hip_device_allocator_t {nullptr, vendor_ordinal}), device_(vendor_ordinal) {}

    /** Carves @p count values on the arena's device, or null when it cannot grow. */
    template <typename value_type_>
    value_type_* allocate(std::size_t count) noexcept {
        if (hip_set_device(device_) != hip_success_k) return nullptr;
        return reinterpret_cast<value_type_*>(memory_.allocate(count * sizeof(value_type_), default_alignment_k));
    }

    /** Whether @p address lies in memory handed out since the last @ref clear. */
    bool contains(void const* address) const noexcept { return memory_.contains(address); }

    /** Rewinds the arena, so the previous call's outputs die. */
    void clear() noexcept { memory_.clear(); }

    /** Copies @p bytes of host memory at @p source to @p destination in the arena,
     *  synchronously. */
    status_t upload(void* destination, void const* source, std::size_t bytes) noexcept {
        if (hip_set_device(device_) != hip_success_k) return status_t::device_error_k;
        return hip_api().copy(destination, source, bytes, hip_copy_host_to_device_k) == hip_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    int device() const noexcept { return device_; }

  private:
    growing_arena<hip_device_allocator_t> memory_;
    int device_ = 0;
};

/** The streams a bulk verb advances on and the scratch they walk with - one of each per device,
 *  which every shard there shares in order. Work is queued, never waited on, so one host thread
 *  drives every device. */
struct hip_executor_t {
    using device_arena_t = hip_device_arena_t;
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::amd_k;

    /** One device's run of scratch, carved and rewound between calls; sized once by @ref reserve,
     *  since relocation would be a host loop over device bytes. */
    using arena_t = growing_arena<hip_device_allocator_t>;

    /** One call's hold on the executor, released when it dies. */
    using claim_t = std::unique_lock<fu::spin_mutex_t>;

    /** One shard's stream and run; the arena is declared last so it frees onto a live stream. */
    struct stream_t {

        /** The HIP ordinal this stream was opened on - placement and launch target both. */
        int device = 0;

        /** The ordering identity every transfer and launch on this device rides. */
        hip_stream_t handle;

        /** The scratch a walk carves its state from, sized once by @ref reserve. */
        arena_t arena;

        /** Takes @p handle, opened on @p device, and binds the arena to it. */
        stream_t(int device, hip_stream_t&& handle) noexcept
            : device(device), handle(std::move(handle)), arena(hip_device_allocator_t {this->handle, device}) {}
    };

    hip_executor_t() = default;
    hip_executor_t(hip_executor_t&& other) noexcept { swap(other); }
    hip_executor_t& operator=(hip_executor_t&& other) noexcept { return swap(other), *this; }
    hip_executor_t(hip_executor_t const&) = delete;
    hip_executor_t& operator=(hip_executor_t const&) = delete;

    /** Exchanges the stream tables; @c held_ stays, as a move with a call in flight is misuse. */
    void swap(hip_executor_t& other) noexcept { std::swap(streams_, other.streams_); }

    /** Claims the executor for one call; a second concurrent caller gets @c contended_k. */
    expected<claim_t> claim() noexcept {
        if (!held_.lock_if_free()) return {{}, status_t::contended_k};
        return {claim_t {held_, std::adopt_lock}, status_t::success_k};
    }

    /** Takes over streams the caller already opened, owned and destroyed here from now on. */
    static expected<hip_executor_t> acquire(growing_buffer<stream_t>&& streams) noexcept {
        if (streams.empty()) return {{}, status_t::config_mismatch_k};
        hip_executor_t executor;
        executor.streams_ = std::move(streams);
        return {std::move(executor), status_t::success_k};
    }

    /** Opens one stream per entry of @p devices, binding each first, since a stream is born on the
     *  current device. */
    static expected<hip_executor_t> spawn(std::span<int const> devices) noexcept {
        growing_buffer<stream_t> streams {global_allocator};
        if (failed(streams.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int device : devices) {
            if (hip_set_device(device) != hip_success_k) return {{}, status_t::device_error_k};
            stream_t opened {device, hip_stream_t {}};
            if (!opened.handle) return {{}, status_t::device_error_k};
            streams.push_back_reserved(std::move(opened));
        }
        return acquire(std::move(streams));
    }

    /** The twin of @ref spawn over each named device's legacy default stream, for a call that names
     *  no executor; every use binds its device first, since a null stream is the current one's. */
    static expected<hip_executor_t> on_default_streams(std::span<int const> devices) noexcept {
        growing_buffer<stream_t> streams {global_allocator};
        if (failed(streams.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int device : devices) {
            if (hip_set_device(device) != hip_success_k) return {{}, status_t::device_error_k};
            streams.push_back_reserved(stream_t {device, hip_stream_t {nullptr}});
        }
        return acquire(std::move(streams));
    }

    /** Claims @p bytes_per_stream of scratch on every stream, ahead of the fan-out that maps state
     *  onto it; the old run frees in stream order behind whatever still rides it. */
    status_t reserve(std::size_t bytes_per_stream) noexcept {
        if (streams_.empty()) return status_t::uninitialized_k;
        for (stream_t& opened : streams_) {
            opened.arena.clear();
            if (failed(opened.arena.reserve_exactly(bytes_per_stream))) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** Waits for every transfer and launch queued on every stream - the fleet's completion; binds
     *  each device first, since a null stream is the current device's. */
    status_t drain() noexcept {
        for (stream_t& opened : streams_)
            if (hip_set_device(opened.device) != hip_success_k ||
                hip_api().stream_synchronize(opened.handle) != hip_success_k)
                return status_t::device_error_k;
        return status_t::success_k;
    }

    /** The stream at @p index, in the order the devices were named. */
    stream_t& stream(std::size_t index) noexcept { return streams_[index]; }

    /** The HIP ordinal stream @p index drives - what a caller pairs its shards against. */
    int device(std::size_t index) const noexcept { return streams_[index].device; }

    /** Streams this executor drives, one per device. */
    std::size_t size() const noexcept { return streams_.size(); }

  private:

    /** One stream per named entry, in the caller's order. */
    growing_buffer<stream_t> streams_ {global_allocator};

    /** Held by the one caller driving the streams; never swapped. */
    fu::spin_mutex_t held_;
};

/** The @c std::unique_ptr deleter carrying the allocator and the element count. */
template <hip_allocator allocator_>
struct hip_deleter {
    allocator_ allocator;
    std::size_t count = 0;

    void operator()(typename allocator_::value_type* pointer) noexcept { allocator.deallocate(pointer, count); }
};

/** Owned device or pinned allocation; host code takes @c .get() for views and transfers. */
template <typename value_type_, hip_allocator allocator_ = hip_device_allocator<value_type_>>
using hip_unique_ptr = std::unique_ptr<value_type_, hip_deleter<allocator_>>;

/** The @c std::make_unique analog, null on exhaustion; the bytes arrive uninitialized. */
template <hip_allocator allocator_>
hip_unique_ptr<typename allocator_::value_type, allocator_> hip_make_unique(allocator_ allocator,
                                                                            std::size_t count) noexcept {
    typename allocator_::value_type* pointer = allocator.allocate(count);
    return {pointer, hip_deleter<allocator_> {allocator, count}};
}

/** One code object loaded onto one GPU and unloaded with this owner; kernels are found by name. */
struct hip_module_t {
    hip_module_t() noexcept = default;
    hip_module_t(hip_module_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    hip_module_t& operator=(hip_module_t&& other) noexcept { return std::swap(handle_, other.handle_), *this; }
    hip_module_t(hip_module_t const&) = delete;
    hip_module_t& operator=(hip_module_t const&) = delete;
    ~hip_module_t() noexcept {
        if (handle_) hip_api().module_unload(handle_);
    }

    /** Loads @p code_object onto HIP ordinal @p device, binding it first, since a module loads into
     *  the current device; an offload bundle loads the entry matching the device's architecture. */
    static expected<hip_module_t> load(int device, bytes_view_t code_object) noexcept {
        hip_module_t module;
        if (code_object.empty() || hip_set_device(device) != hip_success_k ||
            hip_api().module_load_data(&module.handle_, code_object.data()) != hip_success_k)
            return {{}, status_t::device_error_k};
        return {std::move(module), status_t::success_k};
    }

    /** The kernel exported as @p name, or null when the code object has none. */
    hip_function_handle_t function(char const* name) const noexcept {
        hip_function_handle_t function = nullptr;
        return hip_api().module_get_function(&function, handle_, name) == hip_success_k ? function : nullptr;
    }

  private:
    hip_module_handle_t handle_ = nullptr;
};

/** Launches @p function over @p blocks of @p threads on @p stream - the only launch path - passing
 *  each argument by address, as the module API takes them, so each must already be the kernel's own
 *  type; answers the launch's own error, like a bad configuration or too much shared memory. */
template <typename... argument_types_>
status_t hip_launch(hip_function_handle_t function, u32_t blocks, u32_t threads, u32_t shared_bytes,
                    hip_stream_handle_t stream, argument_types_... arguments) noexcept {
    static_assert((std::is_trivially_copyable_v<argument_types_> && ...), "Kernels take their arguments by bytes");
    void* addresses[] = {static_cast<void*>(&arguments)...};
    return hip_api().module_launch_kernel(function, blocks, 1, 1, threads, 1, 1, shared_bytes, stream, addresses,
                                          nullptr) == hip_success_k
               ? status_t::success_k
               : status_t::device_error_k;
}

static_assert(hip_reservable_allocator<hip_virtual_memory_allocator<byte_t>>);
static_assert(hip_reservable_allocator<hip_managed_allocator<byte_t>>);
static_assert(hip_allocator<hip_device_allocator_t>);
static_assert(hip_allocator<hip_arena_allocator_t>);
static_assert(hip_allocator<hip_pinned_allocator_t>);

} // namespace unum::usearch::utilities
