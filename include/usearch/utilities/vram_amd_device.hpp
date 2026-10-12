/**
 *  @file include/usearch/utilities/vram_amd_device.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Host-side vocabulary for driving one AMD GPU - the HIP runtime, specs, streams, memory.
 *
 *  The library never links @c libamdhip64: @ref hip_runtime opens it at first use and resolves the
 *  entry points USearch calls into one table, which stays empty where the library is absent, so AMD
 *  then lists no devices and the binary still loads. Kernels reach a device as code objects
 *  compiled ahead of time and embedded in the library, loaded by @ref hip_module_t and launched by
 *  name, since HIP-Clang's fat-binary registration would link the runtime outright.
 *
 *  The host half is plain C++ that never includes the HIP headers, so the HIP types it passes are
 *  opaque handles and field mirrors, whose handles and constants HIP-Clang checks against its own.
 *  The device half, the @ref hip_warp_t policy, compiles only there.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy` - a field read out of the properties' bytes

#include <dlfcn.h> // `dlopen`, `dlsym` - the HIP runtime opened at first use

#include <atomic>      // `std::memory_order` - the committed extent and the device atomics' orders
#include <bit>         // `std::countr_zero`, `std::popcount` - lane masks
#include <memory>      // `std::unique_ptr` for owned device allocations
#include <mutex>       // `std::unique_lock`, `std::adopt_lock` - the executor's one-caller hold
#include <span>        // `std::span` - the device list an executor opens over, a code object's bytes
#include <string_view> // `std::string_view` - an architecture name
#include <utility>     // `std::move`, `std::exchange`

#include <forkunion/topology.hpp> // `fu::volume_ram` - the managed planes' host-memory budget
#include <forkunion/types.hpp>    // `fu::round_up_to_multiple`, `fu::spin_mutex_t` - the hold

#include <numkong/types.h> // `NUMKONG_ARCH_ROCM_` - set where HIP-Clang compiles

#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - an executor's table of streams
#include "usearch/utilities/types.hpp"       // `gpu_device_t`, `gpu_devices_capacity_k`

#if NUMKONG_ARCH_ROCM_
#include <hip/hip_runtime.h> // `__shfl_xor`, `__ballot`, `__syncwarp` & the mirrored types
#endif                       // NUMKONG_ARCH_ROCM_

/*  HIP's handle types point to these, so the aliases below are HIP's own types where both meet. */
struct ihipStream_t;
struct ihipModule_t;
struct ihipModuleSymbol_t;
struct ihipMemGenericAllocationHandle;

namespace unum {
namespace usearch {

/** A HIP stream, the ordering identity transfers and launches ride; null is the legacy default. */
using hip_stream_handle_t = ihipStream_t*;

/** A code object loaded onto one device. */
using hip_module_handle_t = ihipModule_t*;

/** One kernel of a loaded code object. */
using hip_function_handle_t = ihipModuleSymbol_t*;

/** Physical memory created for a virtual-memory mapping. */
using hip_allocation_handle_t = ihipMemGenericAllocationHandle*;

/** A HIP call's outcome, zero on success. */
using hip_error_t = int;
inline constexpr hip_error_t hip_success_k = 0;

/** The copy directions HIP names, valued as @c hipMemcpyKind. */
enum class hip_memcpy_kind_t : int {
    host_to_device_k = 1,
    device_to_host_k = 2,
    device_to_device_k = 3,

    /** Inferred from the pointers, which unified addressing tells apart. */
    default_k = 4,
};

/** Where an allocation lives, as @c hipMemLocation spells it. */
struct hip_memory_location_t {
    int type = 0;
    int id = 0;
};

/** How a virtual-memory slab is created, as @c hipMemAllocationProp lays it out. */
struct hip_allocation_properties_t {
    int type = 0;
    int requested_handle_types = 0;
    hip_memory_location_t location;
    void* win32_handle_metadata = nullptr;
    u8_t compression_type = 0;
    u8_t gpu_direct_rdma_capable = 0;
    u16_t usage = 0;
};

/** Who may touch a mapped range, as @c hipMemAccessDesc lays it out. */
struct hip_access_descriptor_t {
    hip_memory_location_t location;
    int flags = 0;
};

/** The flag values the allocators pass, valued as HIP's own enumerators and macros. */
inline constexpr int hip_memory_allocation_pinned_k = 1;
inline constexpr int hip_memory_location_device_k = 1;
inline constexpr int hip_memory_access_read_write_k = 3;
inline constexpr int hip_memory_granularity_minimum_k = 0;
inline constexpr unsigned hip_memory_attach_global_k = 1;
inline constexpr int hip_memory_advise_preferred_location_k = 3;
inline constexpr int hip_memory_advise_accessed_by_k = 5;
inline constexpr int hip_cpu_device_k = -1;
inline constexpr unsigned hip_host_malloc_portable_k = 1;
inline constexpr unsigned hip_stream_non_blocking_k = 1;

/** HIP's @c hipDeviceProp_tR0600, which its versioned entry point fills whatever the runtime's own
 *  version: opaque bytes, read at the offsets of the few fields USearch consults. */
struct alignas(8) hip_device_properties_t {
    inline static constexpr std::size_t total_global_memory_offset_k = 288;
    inline static constexpr std::size_t multiprocessor_count_offset_k = 388;
    inline static constexpr std::size_t shared_memory_per_block_optin_offset_k = 696;
    inline static constexpr std::size_t arch_name_offset_k = 1160;
    inline static constexpr std::size_t arch_name_capacity_k = 256;

    byte_t bytes[1472];

    template <typename field_type_>
    field_type_ read(std::size_t offset) const noexcept {
        field_type_ field;
        std::memcpy(&field, bytes + offset, sizeof(field_type_));
        return field;
    }

    /** The architecture the runtime reports, like @c "gfx942:sramecc+:xnack-". */
    std::string_view arch_name() const noexcept {
        std::string_view const padded {reinterpret_cast<char const*>(bytes + arch_name_offset_k), arch_name_capacity_k};
        return padded.substr(0, padded.find('\0'));
    }
};

/**
 *  @brief The HIP entry points USearch calls, resolved from @c libamdhip64 on first use.
 *
 *  Every entry is null when the library is absent or lacks any of them, which leaves AMD with no
 *  devices; the parameter types mirror HIP's own, with the handles above and @c int for its enums.
 */
struct hip_runtime_t {
    hip_error_t (*get_device_count)(int*) = nullptr;
    hip_error_t (*get_device_properties)(hip_device_properties_t*, int) = nullptr;
    hip_error_t (*set_device)(int) = nullptr;
    hip_error_t (*malloc_async)(void**, std::size_t, hip_stream_handle_t) = nullptr;
    hip_error_t (*free_async)(void*, hip_stream_handle_t) = nullptr;
    hip_error_t (*malloc_managed)(void**, std::size_t, unsigned) = nullptr;
    hip_error_t (*memory_advise)(void const*, std::size_t, int, int) = nullptr;
    hip_error_t (*free_managed)(void*) = nullptr;
    hip_error_t (*host_malloc)(void**, std::size_t, unsigned) = nullptr;
    hip_error_t (*host_free)(void*) = nullptr;
    hip_error_t (*copy)(void*, void const*, std::size_t, hip_memcpy_kind_t) = nullptr;
    hip_error_t (*copy_async)(void*, void const*, std::size_t, hip_memcpy_kind_t, hip_stream_handle_t) = nullptr;
    hip_error_t (*copy_2d_async)(void*, std::size_t, void const*, std::size_t, std::size_t, std::size_t,
                                 hip_memcpy_kind_t, hip_stream_handle_t) = nullptr;
    hip_error_t (*fill)(void*, int, std::size_t) = nullptr;
    hip_error_t (*fill_async)(void*, int, std::size_t, hip_stream_handle_t) = nullptr;
    hip_error_t (*fill_words_async)(void*, int, std::size_t, hip_stream_handle_t) = nullptr;
    hip_error_t (*stream_create_with_flags)(hip_stream_handle_t*, unsigned) = nullptr;
    hip_error_t (*stream_destroy)(hip_stream_handle_t) = nullptr;
    hip_error_t (*stream_synchronize)(hip_stream_handle_t) = nullptr;
    hip_error_t (*module_load_data)(hip_module_handle_t*, void const*) = nullptr;
    hip_error_t (*module_unload)(hip_module_handle_t) = nullptr;
    hip_error_t (*module_get_function)(hip_function_handle_t*, hip_module_handle_t, char const*) = nullptr;
    hip_error_t (*module_launch_kernel)(hip_function_handle_t, unsigned, unsigned, unsigned, unsigned, unsigned,
                                        unsigned, unsigned, hip_stream_handle_t, void**, void**) = nullptr;
    hip_error_t (*memory_granularity)(std::size_t*, hip_allocation_properties_t const*, int) = nullptr;
    hip_error_t (*memory_address_reserve)(void**, std::size_t, std::size_t, void*, unsigned long long) = nullptr;
    hip_error_t (*memory_address_free)(void*, std::size_t) = nullptr;
    hip_error_t (*memory_create)(hip_allocation_handle_t*, std::size_t, hip_allocation_properties_t const*,
                                 unsigned long long) = nullptr;
    hip_error_t (*memory_release)(hip_allocation_handle_t) = nullptr;
    hip_error_t (*memory_map)(void*, std::size_t, std::size_t, hip_allocation_handle_t, unsigned long long) = nullptr;
    hip_error_t (*memory_unmap)(void*, std::size_t) = nullptr;
    hip_error_t (*memory_set_access)(void*, std::size_t, hip_access_descriptor_t const*, std::size_t) = nullptr;

    /** Whether the runtime loaded with every entry resolved. */
    explicit operator bool() const noexcept { return memory_set_access != nullptr; }

    /** Opens the runtime and resolves every entry, or answers the empty table. The library stays
     *  open for the life of the process, as the table points into it. */
    static hip_runtime_t load() noexcept {
        char const* const names[] = {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"};
        void* library = nullptr;
        for (char const* name : names)
            if ((library = dlopen(name, RTLD_NOW | RTLD_LOCAL))) break;
        if (!library) return {};
        hip_runtime_t runtime;
        bool const resolved = //
            resolve_(library, "hipGetDeviceCount", runtime.get_device_count) &&
            resolve_(library, "hipGetDevicePropertiesR0600", runtime.get_device_properties) &&
            resolve_(library, "hipSetDevice", runtime.set_device) &&
            resolve_(library, "hipMallocAsync", runtime.malloc_async) &&
            resolve_(library, "hipFreeAsync", runtime.free_async) &&
            resolve_(library, "hipMallocManaged", runtime.malloc_managed) &&
            resolve_(library, "hipMemAdvise", runtime.memory_advise) &&
            resolve_(library, "hipFree", runtime.free_managed) &&
            resolve_(library, "hipHostMalloc", runtime.host_malloc) &&
            resolve_(library, "hipHostFree", runtime.host_free) && resolve_(library, "hipMemcpy", runtime.copy) &&
            resolve_(library, "hipMemcpyAsync", runtime.copy_async) &&
            resolve_(library, "hipMemcpy2DAsync", runtime.copy_2d_async) &&
            resolve_(library, "hipMemset", runtime.fill) && resolve_(library, "hipMemsetAsync", runtime.fill_async) &&
            resolve_(library, "hipMemsetD32Async", runtime.fill_words_async) &&
            resolve_(library, "hipStreamCreateWithFlags", runtime.stream_create_with_flags) &&
            resolve_(library, "hipStreamDestroy", runtime.stream_destroy) &&
            resolve_(library, "hipStreamSynchronize", runtime.stream_synchronize) &&
            resolve_(library, "hipModuleLoadData", runtime.module_load_data) &&
            resolve_(library, "hipModuleUnload", runtime.module_unload) &&
            resolve_(library, "hipModuleGetFunction", runtime.module_get_function) &&
            resolve_(library, "hipModuleLaunchKernel", runtime.module_launch_kernel) &&
            resolve_(library, "hipMemGetAllocationGranularity", runtime.memory_granularity) &&
            resolve_(library, "hipMemAddressReserve", runtime.memory_address_reserve) &&
            resolve_(library, "hipMemAddressFree", runtime.memory_address_free) &&
            resolve_(library, "hipMemCreate", runtime.memory_create) &&
            resolve_(library, "hipMemRelease", runtime.memory_release) &&
            resolve_(library, "hipMemMap", runtime.memory_map) &&
            resolve_(library, "hipMemUnmap", runtime.memory_unmap) &&
            resolve_(library, "hipMemSetAccess", runtime.memory_set_access);
        if (resolved) return runtime;
        dlclose(library);
        return {};
    }

  private:
    template <typename function_type_>
    static bool resolve_(void* library, char const* name, function_type_& entry) noexcept {
        entry = reinterpret_cast<function_type_>(dlsym(library, name));
        return entry != nullptr;
    }
};

/** The one table every AMD call goes through, filled on first use by whichever thread is first. */
inline hip_runtime_t const& hip_runtime() noexcept {
    static hip_runtime_t const runtime = hip_runtime_t::load();
    return runtime;
}

/** The class whose kernels run on an AMD GPU of architecture @p arch_name, as the runtime spells
 *  it: exact processors only, since an older or unlisted one has no code object to load. */
constexpr gpu_class_t hip_gpu_class_of(std::string_view arch_name) noexcept {
    std::string_view const processor = arch_name.substr(0, arch_name.find(':'));
    if (processor == "gfx942") return gpu_class_t::amd_cdna3_k;
    if (processor == "gfx950") return gpu_class_t::amd_cdna4_k;
    if (processor == "gfx1250" || processor == "gfx1251") return gpu_class_t::amd_cdna5_k;
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
        hip_runtime_t const& runtime = hip_runtime();
        hip_device_properties_t properties;
        if (!runtime || runtime.get_device_properties(&properties, ordinal) != hip_success_k)
            return {{ordinal}, status_t::device_error_k};
        return {{ordinal, properties.read<int>(hip_device_properties_t::multiprocessor_count_offset_k),
                 properties.read<std::size_t>(hip_device_properties_t::total_global_memory_offset_k),
                 properties.read<std::size_t>(hip_device_properties_t::shared_memory_per_block_optin_offset_k),
                 hip_gpu_class_of(properties.arch_name())},
                status_t::success_k};
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
        hip_runtime_t const& runtime = hip_runtime();
        int devices_count = 0;
        if (!runtime || runtime.get_device_count(&devices_count) != hip_success_k) return found;
        for (int ordinal = 0; ordinal != devices_count && found.count != gpu_devices_capacity_k; ++ordinal) {
            auto const [specs, status] = hip_device_specs_t::query(ordinal);
            if (failed(status) || specs.gpu_class == gpu_class_t::unknown_k) continue;
            found.devices[found.count++] = {gpu_vendor_t::amd_k, static_cast<u32_t>(ordinal), specs.gpu_class};
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

/** The plane backend contract: a reserve-once / commit-on-demand address range whose base never
 *  moves. Not a refinement of @c hip_allocator, since planes never free mid-life. */
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

/** Virtual-memory plane backend: a fixed address reservation backed by slabs mapped on demand,
 *  host-driven ahead of any kernel; each fresh slab is zeroed as it maps. */
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
        hip_runtime_t const& runtime = hip_runtime();
        hip_allocation_properties_t const properties = allocation_properties_(device);
        std::size_t granularity = 0;
        if (!runtime ||
            runtime.memory_granularity(&granularity, &properties, hip_memory_granularity_minimum_k) != hip_success_k)
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
        hip_runtime_t const& runtime = hip_runtime();
        hip_allocation_properties_t const properties = allocation_properties_(device_);
        hip_access_descriptor_t access;
        access.location = properties.location;
        access.flags = hip_memory_access_read_write_k;
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
        if (mapped_bytes_) hip_runtime().memory_unmap(base_, mapped_bytes_);
        if (base_) hip_runtime().memory_address_free(base_, virtual_address_bytes_);
        base_ = nullptr, virtual_address_bytes_ = 0, mapped_bytes_ = 0;
    }

  private:
    static hip_allocation_properties_t allocation_properties_(int device) noexcept {
        hip_allocation_properties_t properties;
        properties.type = hip_memory_allocation_pinned_k;
        properties.location.type = hip_memory_location_device_k;
        properties.location.id = device;
        return properties;
    }
};

/** Unified-memory plane backend: one @c hipMallocManaged reservation for the whole ceiling,
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
        hip_runtime_t const& runtime = hip_runtime();
        device_ = device;
        virtual_address_bytes_ = virtual_address_bytes;
        void* base = nullptr;
        if (!runtime ||
            runtime.malloc_managed(&base, virtual_address_bytes, hip_memory_attach_global_k) != hip_success_k)
            return status_t::bad_alloc_k;
        base_ = static_cast<value_type*>(base);
        runtime.memory_advise(base, virtual_address_bytes, hip_memory_advise_preferred_location_k, device);
        runtime.memory_advise(base, virtual_address_bytes, hip_memory_advise_accessed_by_k, hip_cpu_device_k);
        return status_t::success_k;
    }

    status_t commit(std::size_t bytes) noexcept {
        if (bytes > virtual_address_bytes_) return status_t::overflow_k;
        if (bytes <= committed_bytes_) return status_t::success_k;
        // Managed pages arrive unzeroed; zeroing the fresh range also commits its residency.
        if (hip_runtime().fill(reinterpret_cast<byte_t*>(base_) + committed_bytes_, 0, bytes - committed_bytes_) !=
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
        if (base_) hip_runtime().free_managed(base_);
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
        return hip_runtime().malloc_async(&pointer, count * sizeof(value_type), stream_) == hip_success_k
                   ? static_cast<value_type*>(pointer)
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept { hip_runtime().free_async(pointer, stream_); }
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
        return hip_runtime().host_malloc(&pointer, count * sizeof(value_type), hip_host_malloc_portable_k) ==
                       hip_success_k
                   ? static_cast<value_type*>(pointer)
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept { hip_runtime().host_free(pointer); }
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
        if (hip_runtime_t const& runtime = hip_runtime(); runtime)
            runtime.stream_create_with_flags(&handle_, hip_stream_non_blocking_k);
    }

    /** The current device's legacy default stream, which nothing here owns or destroys. */
    explicit hip_stream_t(std::nullptr_t) noexcept {}
    hip_stream_t(hip_stream_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    hip_stream_t(hip_stream_t const&) = delete;
    hip_stream_t& operator=(hip_stream_t const&) = delete;
    ~hip_stream_t() noexcept {
        if (handle_) hip_runtime().stream_synchronize(handle_), hip_runtime().stream_destroy(handle_);
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
        hip_runtime_t const& runtime = hip_runtime();
        if (!runtime || runtime.set_device(device_) != hip_success_k) return nullptr;
        return reinterpret_cast<value_type_*>(memory_.allocate(count * sizeof(value_type_), default_alignment_k));
    }

    /** Whether @p address lies in memory handed out since the last @ref clear. */
    bool contains(void const* address) const noexcept { return memory_.contains(address); }

    /** Rewinds the arena, so the previous call's outputs die. */
    void clear() noexcept { memory_.clear(); }

    /** Copies @p bytes of host memory at @p source to @p destination in the arena,
     *  synchronously. */
    status_t upload(void* destination, void const* source, std::size_t bytes) noexcept {
        hip_runtime_t const& runtime = hip_runtime();
        if (!runtime || runtime.set_device(device_) != hip_success_k) return status_t::device_error_k;
        return runtime.copy(destination, source, bytes, hip_memcpy_kind_t::host_to_device_k) == hip_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    int device() const noexcept { return device_; }

  private:
    growing_arena<hip_device_allocator_t> memory_;
    int device_ = 0;
};

/** The streams a bulk verb advances on and the scratch they walk with - one of each per device.
 *  Work is queued, never waited on, so one host thread drives every device. */
struct hip_executor_t {
    using device_arena_t = hip_device_arena_t;
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::amd_k;

    /** One device's run of scratch, carved and rewound between calls; sized once by @ref reserve,
     *  since relocation would be a host loop over device bytes. */
    using arena_t = growing_arena<hip_device_allocator_t>;

    /** One call's hold on the executor, released when it dies. */
    using claim_t = std::unique_lock<fu::spin_mutex_t>;

    /** One device's stream and run; the arena is declared last so it frees onto a live stream. */
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

    /** Opens one stream per named HIP ordinal, binding each first, since a stream is born on the
     *  current device. */
    static expected<hip_executor_t> spawn(std::span<int const> devices) noexcept {
        hip_runtime_t const& runtime = hip_runtime();
        growing_buffer<stream_t> streams {global_allocator};
        if (!runtime) return {{}, status_t::device_error_k};
        if (failed(streams.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int device : devices) {
            if (runtime.set_device(device) != hip_success_k) return {{}, status_t::device_error_k};
            stream_t opened {device, hip_stream_t {}};
            if (!opened.handle) return {{}, status_t::device_error_k};
            streams.push_back_reserved(std::move(opened));
        }
        return acquire(std::move(streams));
    }

    /** The twin of @ref spawn over each named device's legacy default stream, for a call that names
     *  no executor; every use binds its device first, since a null stream is the current one's. */
    static expected<hip_executor_t> on_default_streams(std::span<int const> devices) noexcept {
        hip_runtime_t const& runtime = hip_runtime();
        growing_buffer<stream_t> streams {global_allocator};
        if (!runtime) return {{}, status_t::device_error_k};
        if (failed(streams.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int device : devices) {
            if (runtime.set_device(device) != hip_success_k) return {{}, status_t::device_error_k};
            streams.push_back_reserved(stream_t {device, hip_stream_t {nullptr}});
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
        hip_runtime_t const& runtime = hip_runtime();
        for (stream_t& opened : streams_)
            if (runtime.set_device(opened.device) != hip_success_k ||
                runtime.stream_synchronize(opened.handle) != hip_success_k)
                return status_t::device_error_k;
        return status_t::success_k;
    }

    /** The stream driving @p index, in the order the devices were named. */
    stream_t& stream(std::size_t index) noexcept { return streams_[index]; }

    /** The HIP ordinal @p index drives - what a caller pairs its shards against. */
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
        if (handle_) hip_runtime().module_unload(handle_);
    }

    /** Loads @p code_object onto HIP ordinal @p device, binding it first, since a module loads into
     *  the current device; an offload bundle loads the entry matching the device's architecture. */
    static expected<hip_module_t> load(int device, bytes_view_t code_object) noexcept {
        hip_runtime_t const& runtime = hip_runtime();
        hip_module_t module;
        if (!runtime || runtime.set_device(device) != hip_success_k ||
            runtime.module_load_data(&module.handle_, code_object.data()) != hip_success_k)
            return {{}, status_t::device_error_k};
        return {std::move(module), status_t::success_k};
    }

    /** The kernel exported as @p name, or null when the code object has none. */
    hip_function_handle_t function(char const* name) const noexcept {
        hip_function_handle_t function = nullptr;
        return hip_runtime().module_get_function(&function, handle_, name) == hip_success_k ? function : nullptr;
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
    return hip_runtime().module_launch_kernel(function, blocks, 1, 1, threads, 1, 1, shared_bytes, stream, addresses,
                                              nullptr) == hip_success_k
               ? status_t::success_k
               : status_t::device_error_k;
}

static_assert(hip_reservable_allocator<hip_virtual_memory_allocator<byte_t>>);
static_assert(hip_reservable_allocator<hip_managed_allocator<byte_t>>);
static_assert(hip_allocator<hip_device_allocator_t>);
static_assert(hip_allocator<hip_arena_allocator_t>);
static_assert(hip_allocator<hip_pinned_allocator_t>);

#if NUMKONG_ARCH_ROCM_

static_assert(std::is_same_v<hip_stream_handle_t, hipStream_t> && std::is_same_v<hip_module_handle_t, hipModule_t>);
static_assert(std::is_same_v<hip_function_handle_t, hipFunction_t>);
static_assert(std::is_same_v<hip_allocation_handle_t, hipMemGenericAllocationHandle_t>);
static_assert(hip_success_k == hipSuccess && static_cast<int>(hip_memcpy_kind_t::default_k) == hipMemcpyDefault);
static_assert(static_cast<int>(hip_memcpy_kind_t::host_to_device_k) == hipMemcpyHostToDevice);
static_assert(static_cast<int>(hip_memcpy_kind_t::device_to_host_k) == hipMemcpyDeviceToHost);
static_assert(static_cast<int>(hip_memcpy_kind_t::device_to_device_k) == hipMemcpyDeviceToDevice);
static_assert(hip_memory_allocation_pinned_k == hipMemAllocationTypePinned);
static_assert(hip_memory_location_device_k == hipMemLocationTypeDevice);
static_assert(hip_memory_access_read_write_k == hipMemAccessFlagsProtReadWrite);
static_assert(hip_memory_granularity_minimum_k == hipMemAllocationGranularityMinimum);
static_assert(hip_memory_attach_global_k == hipMemAttachGlobal && hip_host_malloc_portable_k == hipHostMallocPortable);
static_assert(hip_memory_advise_preferred_location_k == hipMemAdviseSetPreferredLocation);
static_assert(hip_memory_advise_accessed_by_k == hipMemAdviseSetAccessedBy && hip_cpu_device_k == hipCpuDeviceId);
static_assert(hip_stream_non_blocking_k == hipStreamNonBlocking);

/*  A wavefront spans 32 lanes on the GFX12 parts, MI400, and 64 on the GFX9 ones, MI300 and MI350.
 *  HIP-Clang learns the wavefront size only after code generation, so the width comes from the
 *  processor family each code object is compiled for. */
#if defined(__GFX12__)
inline constexpr u32_t hip_device_wave_lanes_k = 32;
#else
inline constexpr u32_t hip_device_wave_lanes_k = 64;
#endif // defined(__GFX12__)

/**
 *  @brief The AMD warp policy the shared walk code is written against, the twin of
 *      @c cuda_warp_t: wavefronts as wide as the processor's, agent-scope atomics, and plain
 *      staging loads.
 *
 *  A wavefront moves in lockstep, so the votes and shuffles take no participation mask, and
 *  @ref sync is a wave-scoped fence pair around the scheduling barrier. Masks are as wide as the
 *  wavefront, @c u64_t on CDNA3 and CDNA4.
 */
struct hip_warp_t {
    inline static constexpr u32_t width_k = hip_device_wave_lanes_k;
    using mask_t = std::conditional_t<width_k == 64, u64_t, u32_t>;
    inline static constexpr mask_t all_k = ~mask_t(0);

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle(value_type_ value, int source_lane) noexcept {
        return __shfl(value, source_lane, static_cast<int>(width_k));
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_xor(value_type_ value, int lane_mask) noexcept {
        return __shfl_xor(value, lane_mask, static_cast<int>(width_k));
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_up(value_type_ value, u32_t delta) noexcept {
        return __shfl_up(value, delta, static_cast<int>(width_k));
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_down(value_type_ value, u32_t delta,
                                                               int width = width_k) noexcept {
        return __shfl_down(value, delta, width);
    }
    static __device__ __forceinline__ mask_t ballot(int predicate) noexcept {
        return static_cast<mask_t>(__ballot(predicate));
    }
    static __device__ __forceinline__ bool any(int predicate) noexcept { return __any(predicate) != 0; }
    static __device__ __forceinline__ bool all(int predicate) noexcept { return __all(predicate) != 0; }
    static __device__ __forceinline__ void sync() noexcept { __syncwarp(); }

    /** The lanes holding the same @p value as this one, this lane included: one ballot per distinct
     *  value, since the ISA has no matching vote. */
    static __device__ __forceinline__ mask_t match_any(u32_t value) noexcept {
        mask_t unmatched = all_k, matched = 0;
        while (unmatched) {
            mask_t const same = ballot(shuffle(value, first(unmatched)) == value);
            if ((same >> __lane_id()) & 1u) matched = same;
            unmatched &= ~same;
        }
        return matched;
    }

    static __device__ __forceinline__ u32_t count(mask_t mask) noexcept {
        return static_cast<u32_t>(std::popcount(mask));
    }

    /** The lowest lane of a non-empty @p mask. */
    static __device__ __forceinline__ int first(mask_t mask) noexcept { return std::countr_zero(mask); }

    /** The lanes strictly below @p lane. */
    static __device__ __forceinline__ mask_t below(int lane) noexcept { return (mask_t(1) << lane) - 1; }

    /** Backs off for about @p nanoseconds: each @c s_sleep step idles the wavefront 64 clocks, some
     *  30 nanoseconds at MI300's clocks. */
    static __device__ __forceinline__ void sleep(u32_t nanoseconds) noexcept {
        for (u32_t slept = 0; slept < nanoseconds; slept += 32) asm volatile("s_sleep 1");
    }

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_relaxed(value_type_& word) noexcept {
        return __hip_atomic_load(&word, order_(std::memory_order_relaxed), __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_acquire(value_type_& word) noexcept {
        return __hip_atomic_load(&word, order_(std::memory_order_acquire), __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_relaxed(value_type_& word, value_type_ value) noexcept {
        __hip_atomic_store(&word, value, order_(std::memory_order_relaxed), __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_release(value_type_& word, value_type_ value) noexcept {
        __hip_atomic_store(&word, value, order_(std::memory_order_release), __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ exchange_acquire(value_type_& word, value_type_ value) noexcept {
        return __hip_atomic_exchange(&word, value, order_(std::memory_order_acquire), __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ fetch_add_relaxed(value_type_& word, value_type_ value) noexcept {
        return __hip_atomic_fetch_add(&word, value, order_(std::memory_order_relaxed), __HIP_MEMORY_SCOPE_AGENT);
    }

    /** Orders every store before it ahead of any store after it, for readers across the device. */
    static __device__ __forceinline__ void fence_release() noexcept { __threadfence(); }

    /** Copies @p source into @p destination through registers, since the ISA has no register-free
     *  staging copy; @ref copy_wait has nothing left to land. */
    template <typename value_type_>
    static __device__ __forceinline__ void copy_async(value_type_* destination, value_type_ const* source) noexcept {
        *destination = *source;
    }
    static __device__ __forceinline__ void copy_wait() noexcept {}

  private:

    /** The atomic builtins' order operand, which both standard libraries' orders equal by value. */
    static __device__ __forceinline__ int order_(std::memory_order order) noexcept { return static_cast<int>(order); }
};

#endif // NUMKONG_ARCH_ROCM_

} // namespace usearch
} // namespace unum
