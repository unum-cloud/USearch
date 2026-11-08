/**
 *  @file include/usearch/utilities/vram_amd_device.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Host-side vocabulary for driving one AMD GPU - the HIP runtime, specs, streams, memory.
 *
 *  The library never links @c libamdhip64: @ref unum::usearch::utilities::hip_api_t::load opens
 *  it and resolves the entry points USearch calls into one table, which the caller loads once and
 *  hands every stream, allocator and executor it builds; the table stays empty where the library
 *  is absent, so AMD then lists no devices and the binary still loads. Kernels reach a device as
 *  code objects compiled ahead of time and embedded in the library, loaded by
 *  @ref unum::usearch::utilities::hip_module_t and launched by name, since HIP-Clang's fat-binary
 *  registration would link the runtime outright.
 *
 *  Plain C++ needing no ROCm SDK: the few runtime types, constants and prototypes USearch calls are
 *  spelled here, copied from HIP's stable ABI, so any host compiler builds it. The warp policy the
 *  shared walk code is written against lives in its device twin, @c vram_amd_device.hip.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <bitset>      // `std::bitset` - the devices a range maps its slabs for
#include <memory>      // `std::unique_ptr` for owned device allocations
#include <mutex>       // `std::unique_lock`, `std::adopt_lock` - the executor's one-caller hold
#include <span>        // `std::span` - the device list an executor opens over, a code object's bytes
#include <type_traits> // `std::is_trivially_copyable_v` - a launch's arguments travel as bytes
#include <utility>     // `std::move`, `std::exchange`

#include <numkong/capabilities.h> // `nk_capability_t` - the ceilings the classes grant

#include <forkunion/atomics.hpp>  // `fu::standard_atomic_ref` - the mapped and committed byte counts
#include <forkunion/topology.hpp> // `fu::volume_ram` - the managed storage's host-memory budget
#include <forkunion/types.hpp>    // `fu::round_up_to_multiple`, `fu::spin_mutex_t` - the hold

#include "usearch/utilities/blocking_files.hpp" // `shared_library_t` - the HIP runtime opened by `hip_api_t::load`
#include "usearch/utilities/ram_buffers.hpp"    // `growing_buffer` - an executor's table of streams
#include "usearch/utilities/ram_device.hpp" // `forkunion_allocator_t`, `fu_memory_domain_id_of_pci` - pinned placement
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

/** The runtime's opaque handles: @c hipStream_t, @c hipEvent_t, @c hipModule_t, @c hipFunction_t
 *  and @c hipMemGenericAllocationHandle_t. */
using hip_stream_handle_t = struct hip_stream_opaque_t*;
using hip_event_handle_t = struct hip_event_opaque_t*;
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
    hip_attribute_shared_memory_per_multiprocessor_k = 76,
    hip_attribute_warp_size_k = 87,
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
    struct allocation_flags_t {
        unsigned char compression_type = 0;
        unsigned char gpu_direct_rdma_capable = 0;
        unsigned short usage = 0;
    };
    allocation_flags_t allocation_flags;
};

/** Who may touch a mapped range and how, @c hipMemAccessDesc. */
struct hip_access_descriptor_t {
    inline static constexpr int read_write_k = 3;
    hip_memory_location_t location;
    int flags = 0;
};

/** The flags USearch passes: @c hipStreamNonBlocking, @c hipEventBlockingSync,
 *  @c hipEventDisableTiming, @c hipHostMallocPortable, which @c hipHostRegisterPortable equals,
 *  @c hipMemAttachGlobal, @c hipMemAllocationGranularityMinimum and @c hipCpuDeviceId. */
inline constexpr unsigned hip_stream_non_blocking_k = 1;
inline constexpr unsigned hip_event_blocking_sync_k = 1;
inline constexpr unsigned hip_event_disable_timing_k = 2;
inline constexpr unsigned hip_host_portable_k = 1;
inline constexpr unsigned hip_attach_global_k = 1;
inline constexpr int hip_granularity_minimum_k = 0;
inline constexpr int hip_host_device_k = -1;
inline constexpr int hip_pointer_memory_type_k = 2;
inline constexpr int hip_pointer_device_ordinal_k = 9;
inline constexpr unsigned hip_memory_type_device_k = 2;

#pragma endregion Runtime ABI

/**
 *  @brief The HIP entry points USearch calls, resolved from @c libamdhip64 by @ref load.
 *
 *  Every entry is null when the library is absent or lacks any of them, which leaves AMD with no
 *  devices; each spells the C prototype of the exported symbol it is resolved by.
 */
struct hip_api_t {
    hip_result_t (*get_device_count)(int*) = nullptr;
    hip_result_t (*get_device)(int*, int) = nullptr;
    hip_result_t (*get_device_attribute)(int*, hip_device_attribute_t, int) = nullptr;
    hip_result_t (*get_device_name)(char*, int, int) = nullptr;
    hip_result_t (*get_pci_bus_id)(char*, int, int) = nullptr;
    hip_result_t (*get_device_memory)(std::size_t*, int) = nullptr;
    hip_result_t (*can_access_peer)(int*, int, int) = nullptr;
    hip_result_t (*set_device)(int) = nullptr;
    hip_result_t (*malloc_async)(void**, std::size_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*free_async)(void*, hip_stream_handle_t) = nullptr;
    hip_result_t (*malloc_managed)(void**, std::size_t, unsigned) = nullptr;
    hip_result_t (*memory_advise)(void const*, std::size_t, hip_memory_advice_t, int) = nullptr;
    hip_result_t (*free)(void*) = nullptr;
    hip_result_t (*host_malloc)(void**, std::size_t, unsigned) = nullptr;
    hip_result_t (*host_free)(void*) = nullptr;
    hip_result_t (*host_register)(void*, std::size_t, unsigned) = nullptr;
    hip_result_t (*host_unregister)(void*) = nullptr;
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
    hip_result_t (*pointer_attribute)(void*, int, void const*) = nullptr;
    hip_result_t (*stream_wait_event)(hip_stream_handle_t, hip_event_handle_t, unsigned) = nullptr;
    hip_result_t (*event_create_with_flags)(hip_event_handle_t*, unsigned) = nullptr;
    hip_result_t (*event_record)(hip_event_handle_t, hip_stream_handle_t) = nullptr;
    hip_result_t (*event_query)(hip_event_handle_t) = nullptr;
    hip_result_t (*event_synchronize)(hip_event_handle_t) = nullptr;
    hip_result_t (*event_destroy)(hip_event_handle_t) = nullptr;
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

    /** The memory domain nearest each device, where its pinned host bytes go, or
     *  @c no_memory_domain_k where none is known. */
    fu_memory_domain_id_t memory_domains[gpu_devices_capacity_k] = {};

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
            library.resolve("hipDeviceGetPCIBusId", runtime.get_pci_bus_id) &&
            library.resolve("hipDeviceTotalMem", runtime.get_device_memory) &&
            library.resolve("hipDeviceCanAccessPeer", runtime.can_access_peer) &&
            library.resolve("hipSetDevice", runtime.set_device) &&
            library.resolve("hipMallocAsync", runtime.malloc_async) &&
            library.resolve("hipFreeAsync", runtime.free_async) &&
            library.resolve("hipMallocManaged", runtime.malloc_managed) &&
            library.resolve("hipMemAdvise", runtime.memory_advise) && library.resolve("hipFree", runtime.free) &&
            library.resolve("hipHostMalloc", runtime.host_malloc) &&
            library.resolve("hipHostFree", runtime.host_free) &&
            library.resolve("hipHostRegister", runtime.host_register) &&
            library.resolve("hipHostUnregister", runtime.host_unregister) &&
            library.resolve("hipMemcpy", runtime.copy) && library.resolve("hipMemcpyAsync", runtime.copy_async) &&
            library.resolve("hipMemcpy2DAsync", runtime.copy_2d_async) && library.resolve("hipMemset", runtime.fill) &&
            library.resolve("hipMemsetAsync", runtime.fill_async) &&
            library.resolve("hipMemsetD32Async", runtime.fill_words_async) &&
            library.resolve("hipStreamCreateWithFlags", runtime.stream_create_with_flags) &&
            library.resolve("hipStreamDestroy", runtime.stream_destroy) &&
            library.resolve("hipStreamSynchronize", runtime.stream_synchronize) &&
            library.resolve("hipStreamQuery", runtime.stream_query) &&
            library.resolve("hipPointerGetAttribute", runtime.pointer_attribute) &&
            library.resolve("hipStreamWaitEvent", runtime.stream_wait_event) &&
            library.resolve("hipEventCreateWithFlags", runtime.event_create_with_flags) &&
            library.resolve("hipEventRecord", runtime.event_record) &&
            library.resolve("hipEventQuery", runtime.event_query) &&
            library.resolve("hipEventSynchronize", runtime.event_synchronize) &&
            library.resolve("hipEventDestroy", runtime.event_destroy) &&
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
        if (!resolved) return {};
        // A bus that names no memory domain leaves the device's pinned bytes on the heap.
        for (int device = 0; static_cast<std::size_t>(device) != gpu_devices_capacity_k; ++device) {
            char bus_id[32] = {};
            [[maybe_unused]] hip_result_t const named = runtime.get_pci_bus_id(bus_id, sizeof(bus_id), device);
            [[maybe_unused]] fu_status_t const placed = fu_memory_domain_id_of_pci(bus_id,
                                                                                   &runtime.memory_domains[device]);
        }
        return runtime;
    }
};

/** Makes HIP ordinal @p ordinal current on the calling thread, which every stream, allocation and
 *  launch after it runs on - the twin of @c cuda_set_device. Asked on every entry, so a caller
 *  switching devices between USearch calls is rebound, and finds the last device USearch bound
 *  left current. */
inline hip_result_t hip_set_device(hip_api_t const& api, int ordinal) noexcept {
    if (!api) return hip_error_not_initialized_k;
    return api.set_device(ordinal);
}

/** The ordinal of the AMD device whose memory holds @p pointer, or -1 for host memory, pinned or
 *  not - the twin of @c cuda_device_of. */
inline int hip_device_of(hip_api_t const& api, void const* pointer) noexcept {
    unsigned memory_type = 0;
    int ordinal = -1;
    if (!api || !pointer || api.pointer_attribute(&memory_type, hip_pointer_memory_type_k, pointer) != hip_success_k ||
        memory_type != hip_memory_type_device_k ||
        api.pointer_attribute(&ordinal, hip_pointer_device_ordinal_k, pointer) != hip_success_k)
        return -1;
    return ordinal;
}

/** Lanes of a wavefront under the ceiling @p capability: 32 once it reaches CDNA5, 64 below. */
constexpr u32_t hip_wave_lanes(nk_capability_t capability) noexcept { return capability & nk_cap_cdna5_k ? 32u : 64u; }

/** One AMD GPU's hardware specification - the facts runtime dispatch consults; hardware only, never
 *  derived policy. */
struct hip_device_specs_t {

    /** The device ordinal every HIP call routes through. */
    int ordinal = 0;

    /** Compute-unit count - the launch-grid currency. */
    u32_t multiprocessor_count = 0;

    /** Total device-resident memory in bytes. */
    std::size_t vram_bytes = 0;

    /** Widest dynamic shared memory one block can be granted at launch. */
    std::size_t shared_memory_per_block_bytes = 0;

    /** Local data share one compute unit splits between its resident blocks. */
    std::size_t shared_memory_per_multiprocessor_bytes = 0;

    /** The @c gfx processor's major version, 9 for MI300's @c gfx942. */
    int compute_major = 0;

    /** The @c gfx processor's minor version, 4 for MI300's @c gfx942. */
    int compute_minor = 0;

    /** Lanes of one wavefront, as the runtime reports them. */
    u32_t warp_lanes = 0;

    /**
     *  Queries the runtime for @p ordinal's specification; meaningful only on @c success_k.
     *
     *  @par Statuses:
     *  - @c device_error_k The @p ordinal is past the device listing.
     */
    static expected<hip_device_specs_t> query(hip_api_t const& api, int ordinal) noexcept {
        hip_device_specs_t specs {ordinal};
        int device = 0, multiprocessors = 0, shared_memory_per_block = 0, shared_memory_per_multiprocessor = 0,
            warp_lanes = 0;
        if (!api || api.get_device(&device, ordinal) != hip_success_k ||
            api.get_device_attribute(&multiprocessors, hip_attribute_multiprocessor_count_k, ordinal) !=
                hip_success_k ||
            api.get_device_attribute(&specs.compute_major, hip_attribute_compute_capability_major_k, ordinal) !=
                hip_success_k ||
            api.get_device_attribute(&specs.compute_minor, hip_attribute_compute_capability_minor_k, ordinal) !=
                hip_success_k ||
            api.get_device_attribute(&warp_lanes, hip_attribute_warp_size_k, ordinal) != hip_success_k ||
            api.get_device_attribute(&shared_memory_per_block, hip_attribute_shared_memory_per_block_optin_k,
                                     ordinal) != hip_success_k ||
            api.get_device_attribute(&shared_memory_per_multiprocessor,
                                     hip_attribute_shared_memory_per_multiprocessor_k, ordinal) != hip_success_k ||
            api.get_device_memory(&specs.vram_bytes, device) != hip_success_k)
            return {{ordinal}, status_t::device_error_k};
        specs.multiprocessor_count = static_cast<u32_t>(multiprocessors);
        specs.shared_memory_per_block_bytes = static_cast<std::size_t>(shared_memory_per_block);
        specs.shared_memory_per_multiprocessor_bytes = static_cast<std::size_t>(shared_memory_per_multiprocessor);
        specs.warp_lanes = static_cast<u32_t>(warp_lanes);
        return {specs, status_t::success_k};
    }
};

/** Lists every AMD GPU the HIP runtime reaches into @p into, in HIP ordinal order, as the runtime
 *  reports it, and answers how many it wrote; none without @c libamdhip64. */
inline std::size_t list_amd_gpu_devices(hip_api_t const& api, std::span<gpu_device_t> into) noexcept {
    int devices_count = 0;
    if (!api || api.get_device_count(&devices_count) != hip_success_k) return 0;
    std::size_t count = 0;
    for (int ordinal = 0; ordinal != devices_count && count != into.size(); ++ordinal) {
        auto const [specs, status] = hip_device_specs_t::query(api, ordinal);
        if (failed(status)) continue;
        gpu_device_t& device = into[count++];
        device = {gpu_vendor_t::amd_k,
                  static_cast<u32_t>(ordinal),
                  static_cast<u32_t>(specs.compute_major),
                  static_cast<u32_t>(specs.compute_minor),
                  specs.warp_lanes,
                  specs.multiprocessor_count,
                  specs.vram_bytes};
        int handle = 0;
        if (api.get_device(&handle, ordinal) == hip_success_k)
            api.get_device_name(device.name, static_cast<int>(sizeof(device.name)), handle);
    }
    return count;
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
concept hip_reservable_allocator = requires(allocator_ allocator, std::size_t bytes) {
    typename allocator_::value_type;
    { allocator_::address_space_k } -> std::convertible_to<hip_address_space_t>;
    { allocator.reserve_range(bytes) } -> std::same_as<status_t>;
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

    /** Largest physical slab, rounded to the runtime's granularity at reservation; smaller ones
     *  double up to it, so a sparse shard maps one granularity, not one full slab per column. */
    inline static constexpr std::size_t preferred_slab_bytes_k = 128u << 20;

    byte_t* base_ = nullptr;
    std::size_t virtual_address_bytes_ = 0;

    /** Bytes mapped so far - the one grower advances it with a release, and any thread may read it
     *  through @c committed_bytes with an acquire; @c mutable for the reference. */
    std::size_t mapped_bytes_ = 0;
    std::size_t granularity_ = 0;
    std::size_t slab_bytes_ = 0;
    int device_ = 0;

    /** HIP ordinals besides the owner that every slab maps for, each able to reach the owner. */
    std::bitset<gpu_devices_capacity_k> peers_;

    /** The runtime every mapping goes through; null in a default-built one, which owns nothing. */
    hip_api_t const* api_ = nullptr;

    hip_virtual_memory_allocator() = default;

    /** Names the GPU the future reservation pins to - placement only, no runtime call. */
    explicit hip_virtual_memory_allocator(hip_api_t const& api, int device) noexcept : device_(device), api_(&api) {}

    /** Copies the placement only, never a live reservation. */
    hip_virtual_memory_allocator(hip_virtual_memory_allocator const& other) noexcept
        : device_(other.device_), peers_(other.peers_), api_(other.api_) {}

    /** Takes over a live reservation, so a table of them relocates without remapping. */
    hip_virtual_memory_allocator(hip_virtual_memory_allocator&& other) noexcept
        : base_(std::exchange(other.base_, nullptr)),
          virtual_address_bytes_(std::exchange(other.virtual_address_bytes_, 0)),
          mapped_bytes_(std::exchange(other.mapped_bytes_, 0)), granularity_(other.granularity_),
          slab_bytes_(other.slab_bytes_), device_(other.device_), peers_(other.peers_), api_(other.api_) {}
    template <typename other_>
    hip_virtual_memory_allocator(hip_virtual_memory_allocator<other_> const& other) noexcept
        : device_(other.device_), peers_(other.peers_), api_(other.api_) {}
    hip_virtual_memory_allocator& operator=(hip_virtual_memory_allocator const&) = delete;
    ~hip_virtual_memory_allocator() noexcept { release(); }

    hip_api_t const& api() const noexcept { return *api_; }

    /** Reserves @p virtual_address_bytes of virtual address space on the device named at
     *  construction - no physical memory yet, so the ceiling costs nothing and should be
     *  generous. */
    status_t reserve_range(std::size_t virtual_address_bytes) noexcept {
        hip_allocation_properties_t const properties = allocation_properties_(device_);
        if (!api_ || !*api_ ||
            api_->memory_granularity(&granularity_, &properties, hip_granularity_minimum_k) != hip_success_k)
            return status_t::device_error_k;
        slab_bytes_ = max_of(granularity_, preferred_slab_bytes_k / granularity_ * granularity_);
        virtual_address_bytes_ = fu::round_up_to_multiple(virtual_address_bytes, slab_bytes_);
        void* base = nullptr;
        if (api_->memory_address_reserve(&base, virtual_address_bytes_, 0, nullptr, 0) != hip_success_k)
            return status_t::bad_alloc_k;
        base_ = static_cast<byte_t*>(base);
        return status_t::success_k;
    }

    /** Maps every slab for those of @p devices that can reach the owner too, so their kernels and
     *  copies read and write it directly; placement only, named before the first @ref commit. */
    void share_with(std::bitset<gpu_devices_capacity_k> devices) noexcept { peers_ = reachable_(devices); }

    /** Whether kernels and copies on HIP ordinal @p device reach the mapped slabs. */
    bool shared_with(int device) const noexcept {
        return device == device_ || peers_[static_cast<std::size_t>(device)];
    }

    /**
     *  @brief Maps every slab, those committed already included, for @p devices too, as a sharded
     *      index absorbing this storage names its own. No slab may be committed meanwhile.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k One of @p devices cannot reach the owner; nothing is granted.
     */
    status_t grant(std::bitset<gpu_devices_capacity_k> devices) noexcept {
        if (!api_ || !*api_) return status_t::device_error_k;
        devices[static_cast<std::size_t>(device_)] = false;
        if (reachable_(devices) != devices) return status_t::config_mismatch_k;
        std::bitset<gpu_devices_capacity_k> const granted = peers_ | devices;
        if (granted == peers_) return status_t::success_k;
        hip_access_descriptor_t access[gpu_devices_capacity_k];
        std::size_t const readers = access_of_(granted, access);
        for (std::size_t offset = 0, slab = 0; offset != mapped_bytes_; offset += slab) {
            slab = min_of(slab_bytes_, max_of(granularity_, offset));
            if (api_->memory_set_access(base_ + offset, slab, access, readers) != hip_success_k)
                return status_t::device_error_k;
        }
        peers_ = granted;
        return status_t::success_k;
    }

    /**
     *  Maps and zeroes physical slabs until at least @p bytes are usable, for the owner and every
     *  peer; never unmaps early. Each slab is as large as everything mapped before it, from the
     *  granularity up to the preferred slab, so a shard's footprint tracks its rows. Each handle is
     *  released right after mapping, since the mapping alone keeps the memory alive.
     *
     *  @par Statuses:
     *  - @c overflow_k The @p bytes pass the reserved range.
     */
    status_t commit(std::size_t bytes) noexcept {
        if (bytes > virtual_address_bytes_) return status_t::overflow_k;
        // A reservation exists only once `reserve_range` found the runtime, so the table is
        // complete.
        hip_api_t const& runtime = *api_;
        hip_allocation_properties_t const properties = allocation_properties_(device_);
        hip_access_descriptor_t access[gpu_devices_capacity_k];
        std::size_t const readers = access_of_(peers_, access);
        while (mapped_bytes_ < bytes) {
            hip_allocation_handle_t handle = nullptr;
            std::size_t const slab_bytes = min_of(slab_bytes_, max_of(granularity_, mapped_bytes_));
            if (runtime.memory_create(&handle, slab_bytes, &properties, 0) != hip_success_k)
                return status_t::bad_alloc_k;
            byte_t* const slab = base_ + mapped_bytes_;
            bool const mapped = runtime.memory_map(slab, slab_bytes, 0, handle, 0) == hip_success_k &&
                                runtime.memory_set_access(slab, slab_bytes, access, readers) == hip_success_k;
            runtime.memory_release(handle);
            if (!mapped) return status_t::device_error_k;
            if (runtime.fill(slab, 0, slab_bytes) != hip_success_k) return status_t::device_error_k;
            fu::standard_atomic_ref<std::size_t>(mapped_bytes_)
                .store(mapped_bytes_ + slab_bytes, std::memory_order_release);
        }
        return status_t::success_k;
    }

    value_type* base() const noexcept { return reinterpret_cast<value_type*>(base_); }
    std::size_t reserved_bytes() const noexcept { return virtual_address_bytes_; }
    std::size_t committed_bytes() const noexcept {
        return fu::standard_atomic_ref<std::size_t const>(mapped_bytes_).load(std::memory_order_acquire);
    }
    int device() const noexcept { return device_; }

    void release() noexcept {
        if (mapped_bytes_) api_->memory_unmap(base_, mapped_bytes_);
        if (base_) api_->memory_address_free(base_, virtual_address_bytes_);
        base_ = nullptr, virtual_address_bytes_ = 0, mapped_bytes_ = 0;
    }

  private:

    /** Those of @p devices besides the owner whose kernels and copies can reach its slabs. */
    std::bitset<gpu_devices_capacity_k> reachable_(std::bitset<gpu_devices_capacity_k> devices) const noexcept {
        std::bitset<gpu_devices_capacity_k> reachable;
        if (!api_ || !*api_) return reachable;
        for (std::size_t ordinal = 0; ordinal != devices.size(); ++ordinal) {
            int reaches = 0;
            if (!devices[ordinal] || static_cast<int>(ordinal) == device_) continue;
            reachable[ordinal] = api_->can_access_peer(&reaches, static_cast<int>(ordinal), device_) == hip_success_k &&
                                 reaches;
        }
        return reachable;
    }

    static hip_allocation_properties_t allocation_properties_(int device) noexcept {
        hip_allocation_properties_t properties;
        properties.type = hip_allocation_properties_t::pinned_k;
        properties.location.type = hip_memory_location_t::device_k;
        properties.location.id = device;
        return properties;
    }

    /** Writes the owner's read-write access, then each of @p peers', into @p access, and answers
     *  how many it wrote. */
    std::size_t access_of_(std::bitset<gpu_devices_capacity_k> peers,
                           hip_access_descriptor_t (&access)[gpu_devices_capacity_k]) const noexcept {
        access[0] = {allocation_properties_(device_).location, hip_access_descriptor_t::read_write_k};
        std::size_t readers = 1;
        for (std::size_t ordinal = 0; ordinal != peers.size(); ++ordinal)
            if (peers[ordinal])
                access[readers++] = {{hip_memory_location_t::device_k, static_cast<int>(ordinal)},
                                     hip_access_descriptor_t::read_write_k};
        return readers;
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

    /** The runtime every reservation goes through; null in a default-built one, which owns
     *  nothing. */
    hip_api_t const* api_ = nullptr;

    hip_managed_allocator() = default;

    /** Names the GPU the future reservation prefers - placement only, no runtime call. */
    explicit hip_managed_allocator(hip_api_t const& api, int device) noexcept : device_(device), api_(&api) {}

    /** Copies the placement configuration only, never a live reservation. */
    hip_managed_allocator(hip_managed_allocator const& other) noexcept : device_(other.device_), api_(other.api_) {}
    template <typename other_>
    hip_managed_allocator(hip_managed_allocator<other_> const& other) noexcept
        : device_(other.device_), api_(other.api_) {}
    hip_managed_allocator& operator=(hip_managed_allocator const&) = delete;
    ~hip_managed_allocator() noexcept { release(); }

    hip_api_t const& api() const noexcept { return *api_; }

    /** Reserves @p virtual_address_bytes of managed memory, advised to prefer the device named at
     *  construction and to stay reachable from the host. */
    status_t reserve_range(std::size_t virtual_address_bytes) noexcept {
        virtual_address_bytes_ = virtual_address_bytes;
        void* base = nullptr;
        if (!api_ || !*api_ || api_->malloc_managed(&base, virtual_address_bytes, hip_attach_global_k) != hip_success_k)
            return status_t::bad_alloc_k;
        base_ = static_cast<value_type*>(base);
        api_->memory_advise(base, virtual_address_bytes, hip_advise_preferred_location_k, device_);
        api_->memory_advise(base, virtual_address_bytes, hip_advise_accessed_by_k, hip_host_device_k);
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
        // Managed pages arrive unzeroed; zeroing the fresh range also commits its residency.
        if (api_->fill(reinterpret_cast<byte_t*>(base_) + committed_bytes_, 0, bytes - committed_bytes_) !=
            hip_success_k)
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

    /** The reservation ceiling: half the host RAM, since managed reservations are real allocations;
     *  only this backend consults the topology. */
    static std::size_t budget_bytes() noexcept { return fu::volume_ram() / 2; }

    void release() noexcept {
        if (base_) api_->free(base_);
        base_ = nullptr, virtual_address_bytes_ = 0, committed_bytes_ = 0;
    }
};

struct hip_pinned_slots_t;
struct hip_kernel_tables_t;

template <typename value_type_>
struct hip_arena_allocator;

/** Stream-ordered device allocator for transient scratch, riding the executor stream it was built
 *  from; a default-built one rides none, so it only fills the deleter of an empty owner. */
template <typename value_type_ = byte_t>
struct hip_device_allocator {
    using value_type = value_type_;
    inline static constexpr hip_address_space_t address_space_k = hip_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = hip_device_allocator<other_>;
    };

    hip_device_allocator() noexcept = default;

    /** Rides the executor's @p stream on the device it names; a template only because that
     *  @c stream_t holds a run of this allocator, so it is declared after it. */
    template <typename stream_type_>
    explicit hip_device_allocator(stream_type_ const& stream) noexcept
        : api_(stream.api), stream_(stream.handle), device_(stream.device), pinned_slots_(stream.pinned_slots.get()),
          kernel_tables_(stream.kernel_tables.get()) {}

    /** Rides the stream @p carver carves for, placing in that stream's pool what must outlive the
     *  rewinds of its run. */
    template <typename other_>
    explicit hip_device_allocator(hip_arena_allocator<other_> const& carver) noexcept
        : api_(&carver.api()), stream_(carver.stream()), device_(carver.device()), pinned_slots_(carver.pinned_slots()),
          kernel_tables_(carver.kernel_tables()) {}
    template <typename other_>
    hip_device_allocator(hip_device_allocator<other_> const& other) noexcept
        : api_(other.api_), stream_(other.stream()), device_(other.device()), pinned_slots_(other.pinned_slots()),
          kernel_tables_(other.kernel_tables()) {}

    value_type* allocate(std::size_t count) noexcept {
        void* pointer = nullptr;
        return api_ && api_->malloc_async(&pointer, count * sizeof(value_type), stream_) == hip_success_k
                   ? static_cast<value_type*>(pointer)
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept {
        if (api_) api_->free_async(pointer, stream_);
    }
    hip_api_t const& api() const noexcept { return *api_; }
    hip_stream_handle_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }

    /** The pinned slots of the stream it rides, which a context copies its counters into. */
    hip_pinned_slots_t* pinned_slots() const noexcept { return pinned_slots_; }

    /** The code objects loaded on the stream it rides, which a context launches through. */
    hip_kernel_tables_t* kernel_tables() const noexcept { return kernel_tables_; }

  private:
    template <typename>
    friend struct hip_device_allocator;

    hip_api_t const* api_ = nullptr;
    hip_stream_handle_t stream_ = nullptr;
    int device_ = 0;
    hip_pinned_slots_t* pinned_slots_ = nullptr;
    hip_kernel_tables_t* kernel_tables_ = nullptr;
};

/** The byte specialization device storage is placed through. */
using hip_device_allocator_t = hip_device_allocator<byte_t>;

/**
 *  @brief The device twin of @c arena_allocator: carves device bytes from one stream's run.
 *
 *  A spent run answers @c nullptr rather than spilling to a pool, so a mis-sized executor is a
 *  reported failure, never a silent allocation; releasing a carve does nothing, and the run is
 *  rewound by whoever owns it, between calls, so a traversal context carves here only a call's
 *  scratch and keeps its own buffers in the stream's pool. A context wanting pool-backed call
 *  scratch names @ref hip_device_allocator instead - the allocator type picks the behavior.
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

    /** Carves from @p arena on the executor's @p stream, which names the device. */
    template <typename stream_type_>
    hip_arena_allocator(arena_t& arena, stream_type_ const& stream) noexcept
        : arena_(arena), api_(stream.api), stream_(stream.handle), device_(stream.device),
          pinned_slots_(stream.pinned_slots.get()), kernel_tables_(stream.kernel_tables.get()) {}

    /** The rebinding conversion @c std::allocator_traits performs - the identity rides along, and
     *  the run is shared rather than copied. */
    template <typename other_>
    constexpr hip_arena_allocator(hip_arena_allocator<other_> const& other) noexcept
        : arena_(other.arena()), api_(other.api_), stream_(other.stream()), device_(other.device()),
          pinned_slots_(other.pinned_slots()), kernel_tables_(other.kernel_tables()) {}

    value_type* allocate(std::size_t count) noexcept {
        // Host-side pointer arithmetic over device bytes: the carve never touches them.
        return arena_allocator<value_type_, hip_device_allocator_t> {arena_}.allocate(count);
    }

    /** A carve is never returned; the run's owner rewinds it between calls. */
    void deallocate(value_type*, std::size_t) noexcept {}

    hip_api_t const& api() const noexcept { return *api_; }
    hip_stream_handle_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }

    /** The pinned slots of the stream it rides, which a context copies its counters into. */
    hip_pinned_slots_t* pinned_slots() const noexcept { return pinned_slots_; }

    /** The code objects loaded on the stream it rides, which a context launches through. */
    hip_kernel_tables_t* kernel_tables() const noexcept { return kernel_tables_; }

    /** The run every rebound copy shares. */
    arena_t& arena() const noexcept { return arena_; }
    bool operator==(hip_arena_allocator const& other) const noexcept { return &arena_ == &other.arena_; }

  private:
    template <typename>
    friend struct hip_arena_allocator;

    arena_t& arena_;
    hip_api_t const* api_ = nullptr;
    hip_stream_handle_t stream_ = nullptr;
    int device_ = 0;
    hip_pinned_slots_t* pinned_slots_ = nullptr;
    hip_kernel_tables_t* kernel_tables_ = nullptr;
};

/** The byte specialization a walk carves its scratch through. */
using hip_arena_allocator_t = hip_arena_allocator<byte_t>;

/** Page-locked host allocator - the DMA source and sink; pinning is synchronous, the stream is
 *  identity only. Knowing the device, it places its pages on the memory domain nearest the device's
 *  PCIe slot, where the copy engines read them, and page-locks them there; otherwise, or where the
 *  system names no domain, the runtime allocates them. */
template <typename value_type_ = byte_t>
struct hip_pinned_allocator {
    using value_type = value_type_;
    inline static constexpr hip_address_space_t address_space_k = hip_address_space_t::host_pinned_k;
    template <typename other_>
    struct rebind {
        using other = hip_pinned_allocator<other_>;
    };

    hip_pinned_allocator() noexcept = default;

    /** Pins through @p api for a host bounce that rides no stream. */
    explicit hip_pinned_allocator(hip_api_t const& api) noexcept : api_(&api) {}

    /** Pins through @p api beside device @p device, for buffers that ride no stream. */
    hip_pinned_allocator(hip_api_t const& api, int device) noexcept
        : api_(&api), device_(device), memory_domain_(api.memory_domains[device]) {}

    /** Takes the identity of the executor's @p stream and the device it names. */
    template <typename stream_type_>
    explicit hip_pinned_allocator(stream_type_ const& stream) noexcept
        : api_(stream.api), stream_(stream.handle), device_(stream.device),
          memory_domain_(stream.api->memory_domains[stream.device]) {}
    template <typename other_>
    hip_pinned_allocator(hip_pinned_allocator<other_> const& other) noexcept
        : api_(other.api_), stream_(other.stream()), device_(other.device()), memory_domain_(other.memory_domain_) {}

    value_type* allocate(std::size_t count) noexcept {
        if (!api_) return nullptr;
        std::size_t const bytes = pinned_bytes_(count);
        if (memory_domain_ == no_memory_domain_k) {
            void* pointer = nullptr;
            return api_->host_malloc(&pointer, bytes, hip_host_portable_k) == hip_success_k
                       ? static_cast<value_type*>(pointer)
                       : nullptr;
        }
        byte_t* const pointer = forkunion_allocator_t {memory_domain_}.allocate(bytes);
        if (pointer && api_->host_register(pointer, bytes, hip_host_portable_k) != hip_success_k)
            return forkunion_allocator_t {memory_domain_}.deallocate(pointer, bytes), nullptr;
        return reinterpret_cast<value_type*>(pointer);
    }
    void deallocate(value_type* pointer, std::size_t count) noexcept {
        if (!api_ || !pointer) return;
        if (memory_domain_ == no_memory_domain_k) return void(api_->host_free(pointer));
        api_->host_unregister(pointer);
        forkunion_allocator_t {memory_domain_}.deallocate(reinterpret_cast<byte_t*>(pointer), pinned_bytes_(count));
    }
    hip_api_t const& api() const noexcept { return *api_; }
    hip_stream_handle_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }

  private:
    template <typename>
    friend struct hip_pinned_allocator;

    hip_api_t const* api_ = nullptr;
    hip_stream_handle_t stream_ = nullptr;
    int device_ = 0;

    /** Where the pages sit, @c no_memory_domain_k where the runtime allocates them. */
    fu_memory_domain_id_t memory_domain_ = no_memory_domain_k;

    /** The bytes @p count values pin: whole pages, since two registrations sharing a page refuse
     *  each other. */
    static std::size_t pinned_bytes_(std::size_t count) noexcept {
        return round_up_to_multiple<std::size_t>(count * sizeof(value_type), fu::ram_page_size());
    }
};

/** The byte specialization a host bounce is pinned through. */
using hip_pinned_allocator_t = hip_pinned_allocator<byte_t>;

/** Page-locked slots the traversal contexts on one executor stream copy their counters into - the
 *  twin of @c cuda_pinned_slots_t. Each context takes one at birth and gives it back at death, so
 *  two alive at once never share one, and pinning, which waits on the device, runs only past the
 *  most contexts the stream held alive at once. A slot given back may still await copies its last
 *  owner queued, but every owner rides this one stream, so those finish before the next owner's. */
struct hip_pinned_slots_t {

    /** Bytes one slot holds: a context's telemetry snapshot and its missed word, with room to
     *  spare. */
    inline static constexpr std::size_t slot_bytes_k = 128;

    /** Slots the first pin holds, more than the shards one device usually serves; past them the
     *  run pins twice as many. */
    inline static constexpr std::size_t slots_per_pin_k = 64;

    /** Pins its slots through @p api beside device @p device. */
    hip_pinned_slots_t(hip_api_t const& api, int device) noexcept : slots_ {hip_pinned_allocator_t {api, device}} {}

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
    growing_arena<hip_pinned_allocator_t> slots_;
    growing_buffer<void*> returned_ {global_allocator};

    /** Held across a take or a give-back, so contexts born and dying on several threads share
     *  the slots. */
    fu::spin_mutex_t guard_;
};

/** Owned non-blocking stream, born on the current device; destruction drains it first. */
struct hip_stream_t {
    explicit hip_stream_t(hip_api_t const& api) noexcept : api_(&api) {
        if (api) api.stream_create_with_flags(&handle_, hip_stream_non_blocking_k);
    }
    hip_stream_t(hip_stream_t&& other) noexcept : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)) {}
    hip_stream_t(hip_stream_t const&) = delete;
    hip_stream_t& operator=(hip_stream_t const&) = delete;
    ~hip_stream_t() noexcept {
        if (handle_) api_->stream_synchronize(handle_), api_->stream_destroy(handle_);
    }

    operator hip_stream_handle_t() const noexcept { return handle_; }

  private:
    hip_api_t const* api_ = nullptr;
    hip_stream_handle_t handle_ = nullptr;
};

/** Owned event, born on the current device, marking a point on a stream that the host polls or
 *  awaits; it blocks on sync, so a drain sleeps instead of spinning a core. Destruction never
 *  waits. */
struct hip_event_t {

    /** An empty event, which a refused plan holds and which records nothing. */
    hip_event_t() noexcept = default;

    explicit hip_event_t(hip_api_t const& api) noexcept : api_(&api) {
        if (api) api.event_create_with_flags(&handle_, hip_event_blocking_sync_k | hip_event_disable_timing_k);
    }
    hip_event_t(hip_event_t&& other) noexcept : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)) {}
    hip_event_t(hip_event_t const&) = delete;
    hip_event_t& operator=(hip_event_t const&) = delete;
    ~hip_event_t() noexcept {
        if (handle_) api_->event_destroy(handle_);
    }

    operator hip_event_handle_t() const noexcept { return handle_; }

    /**
     *  @brief Marks everything queued on @p stream so far, which @ref finished and @ref wait
     *      then await in order.
     *
     *  @par Statuses:
     *  - @c device_error_k The event was never created.
     */
    status_t record(hip_stream_handle_t stream) noexcept {
        return handle_ && api_->event_record(handle_, stream) == hip_success_k ? status_t::success_k
                                                                               : status_t::device_error_k;
    }

    /**
     *  Polls the recorded work without blocking; nothing recorded has nothing to run.
     *
     *  @par Statuses:
     *  - @c device_error_k The event was never created.
     */
    status_t finished() const noexcept {
        if (!handle_) return status_t::device_error_k;
        hip_result_t const queried = api_->event_query(handle_);
        return queried == hip_success_k           ? status_t::success_k
               : queried == hip_error_not_ready_k ? status_t::deferred_k
                                                  : status_t::device_error_k;
    }

    /**
     *  Sleeps until the recorded work finishes and answers whether the device finished it.
     *
     *  @par Statuses:
     *  - @c device_error_k The event was never created.
     */
    status_t wait() noexcept {
        return handle_ && api_->event_synchronize(handle_) == hip_success_k ? status_t::success_k
                                                                            : status_t::device_error_k;
    }

  private:
    hip_api_t const* api_ = nullptr;
    hip_event_handle_t handle_ = nullptr;
};

/** Device memory that one call at a time rewinds and refills, on one AMD device: the AMD
 *  alternative of @c gpu_arena_t, where the C layer writes a VRAM call's outputs. It grows, uploads
 *  and frees in the order of the first stream of the executor that made it, so it never waits on
 *  the device until @ref wait, and must die before that executor. */
struct hip_device_arena_t {

    /** Rides the executor's @p stream on the device it names, and drains through its event; a
     *  template only because that @c stream_t is declared after it. */
    template <typename stream_type_>
    explicit hip_device_arena_t(stream_type_& stream) noexcept
        : memory_(hip_device_allocator_t {stream}), api_(stream.api), stream_(stream.handle), event_(&stream.event),
          device_(stream.device) {}

    /** Carves @p count values on the arena's device, or null when it cannot grow. */
    template <typename value_type_>
    value_type_* allocate(std::size_t count) noexcept {
        if (hip_set_device(*api_, device_) != hip_success_k) return nullptr;
        return reinterpret_cast<value_type_*>(memory_.allocate(count * sizeof(value_type_), default_alignment_k));
    }

    /** Whether @p address lies in memory handed out since the last @ref clear. */
    bool contains(void const* address) const noexcept { return memory_.contains(address); }

    /** Rewinds the arena, so the previous call's outputs die. */
    void clear() noexcept { memory_.clear(); }

    /** Queues a copy of @p bytes of host memory at @p source to @p destination in the arena behind
     *  the call's work; @p source must stay alive until @ref wait. */
    status_t upload(void* destination, void const* source, std::size_t bytes) noexcept {
        return hip_set_device(*api_, device_) == hip_success_k &&
                       api_->copy_async(destination, source, bytes, hip_copy_host_to_device_k, stream_) == hip_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Waits once for every upload queued so far - the end of the call holding the arena -
     *  sleeping on the stream's event. */
    status_t wait() noexcept {
        if (hip_set_device(*api_, device_) != hip_success_k) return status_t::device_error_k;
        if (status_t const recorded = event_->record(stream_); failed(recorded)) return recorded;
        return event_->wait();
    }

    int device() const noexcept { return device_; }

  private:
    growing_arena<hip_device_allocator_t> memory_;
    hip_api_t const* api_ = nullptr;
    hip_stream_handle_t stream_ = nullptr;
    hip_event_t* event_ = nullptr;
    int device_ = 0;
};

/** The @c std::unique_ptr deleter carrying the allocator and the element count. */
template <typename allocator_>
struct hip_deleter {
    allocator_ allocator;
    std::size_t count = 0;

    void operator()(typename allocator_::value_type* pointer) noexcept { allocator.deallocate(pointer, count); }
};

/** Owned device or pinned allocation; host code takes @c .get() for views and transfers. */
template <typename value_type_, typename allocator_ = hip_device_allocator<value_type_>>
using hip_unique_ptr = std::unique_ptr<value_type_, hip_deleter<allocator_>>;

/** The @c std::make_unique analog, null on exhaustion; the bytes arrive uninitialized. */
template <typename allocator_>
hip_unique_ptr<typename allocator_::value_type, allocator_> hip_make_unique(allocator_ allocator,
                                                                            std::size_t count) noexcept {
    typename allocator_::value_type* pointer = allocator.allocate(count);
    return {pointer, hip_deleter<allocator_> {allocator, count}};
}

/** One code object loaded onto one GPU and unloaded with this owner; kernels are found by name. */
struct hip_module_t {
    hip_module_t() noexcept = default;
    hip_module_t(hip_module_t&& other) noexcept : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)) {}
    hip_module_t& operator=(hip_module_t&& other) noexcept {
        return std::swap(api_, other.api_), std::swap(handle_, other.handle_), *this;
    }
    hip_module_t(hip_module_t const&) = delete;
    hip_module_t& operator=(hip_module_t const&) = delete;
    ~hip_module_t() noexcept {
        if (handle_) api_->module_unload(handle_);
    }

    /**
     *  Loads @p code_object onto HIP ordinal @p device, binding it first, since a module loads into
     *  the current device; an offload bundle loads the entry matching the device's architecture.
     *
     *  @par Statuses:
     *  - @c device_error_k The @p code_object is empty.
     */
    static expected<hip_module_t> load(hip_api_t const& api, int device, bytes_view_t code_object) noexcept {
        hip_module_t module;
        module.api_ = &api;
        if (code_object.empty() || hip_set_device(api, device) != hip_success_k ||
            api.module_load_data(&module.handle_, code_object.data()) != hip_success_k)
            return {{}, status_t::device_error_k};
        return {std::move(module), status_t::success_k};
    }

    /** The kernel exported as @p name, or null when the code object has none. */
    hip_function_handle_t function(char const* name) const noexcept {
        if (!api_) return nullptr;
        hip_function_handle_t function = nullptr;
        return api_->module_get_function(&function, handle_, name) == hip_success_k ? function : nullptr;
    }

    /** The runtime the code object was loaded through, which launches its kernels. */
    hip_api_t const& api() const noexcept { return *api_; }

  private:
    hip_api_t const* api_ = nullptr;
    hip_module_handle_t handle_ = nullptr;
};

/** One code object loaded on one device, and every kernel handle the shards there launch through,
 *  laid out as the speller the load was handed indexes them; the code object's address identifies
 *  the table, and the module stays public for the kernels looked up by symbol at their launch. */
struct hip_kernel_table_t {
    bytes_view_t fatbin;
    int device = 0;
    hip_module_t module;
    growing_buffer<hip_function_handle_t> functions {global_allocator};

    /**
     *  @brief Loads @p bytes onto @p on_device through @p api and finds the kernel @p speller names
     *      at every position below its count, null where the code object lacks a kernel the speller
     *      does not require.
     *
     *  @par Statuses:
     *  - @c device_error_k The code object is empty or lacks a required kernel, or the runtime
     *    would not load it.
     */
    template <typename speller_type_>
    status_t load(hip_api_t const& api, int on_device, bytes_view_t bytes, speller_type_ const& speller) noexcept {
        auto [loaded, status] = hip_module_t::load(api, on_device, bytes);
        if (failed(status)) return status;
        if (failed(functions.resize(speller.count()))) return status_t::bad_alloc_k;
        fatbin = bytes;
        device = on_device;
        module = std::move(loaded);
        typename speller_type_::symbol_t symbol;
        for (std::size_t position = 0; position != functions.size(); ++position) {
            speller.spell(symbol, position);
            functions[position] = module.function(symbol.text);
            if (!functions[position] && speller.required(position)) return status_t::device_error_k;
        }
        return status_t::success_k;
    }
};

/** The code objects loaded on one device's stream, one table each, owned here and kept until the
 *  stream dies, so a verb or a plan may hold a table for its life. */
struct hip_kernel_tables_t {

    /**
     *  @brief The table of @p fatbin on @p device, loaded through @p speller at the first ask.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k The table could not be allocated.
     *  - @c device_error_k The code object would not load, or lacks a required kernel.
     */
    template <typename speller_type_>
    expected<hip_kernel_table_t const*> table_of(hip_api_t const& api, int device, bytes_view_t fatbin,
                                                 speller_type_ const& speller) noexcept {
        for (std::unique_ptr<hip_kernel_table_t> const& table : tables_)
            if (table->fatbin.data() == fatbin.data()) return {table.get(), status_t::success_k};
        std::unique_ptr<hip_kernel_table_t> table {new (std::nothrow) hip_kernel_table_t};
        if (!table) return {nullptr, status_t::bad_alloc_k};
        if (status_t const loaded = table->load(api, device, fatbin, speller); failed(loaded)) return {nullptr, loaded};
        if (failed(tables_.push_back(std::move(table)))) return {nullptr, status_t::bad_alloc_k};
        return {tables_.back().get(), status_t::success_k};
    }

  private:
    growing_buffer<std::unique_ptr<hip_kernel_table_t>> tables_ {global_allocator};
};

/**
 *  Launches @p function over @p blocks of @p threads on @p stream - the only launch path - passing
 *  each argument by address, as the module API takes them, so each must already be the kernel's own
 *  type.
 *
 *  @par Statuses:
 *  - @c device_error_k The launch refuses a bad configuration or too much shared memory.
 */
template <typename... argument_types_>
status_t hip_launch(hip_api_t const& api, hip_function_handle_t function, u32_t blocks, u32_t threads,
                    u32_t shared_bytes, hip_stream_handle_t stream, argument_types_... arguments) noexcept {
    static_assert((std::is_trivially_copyable_v<argument_types_> && ...), "Kernels take their arguments by bytes");
    void* addresses[] = {static_cast<void*>(&arguments)...};
    return api.module_launch_kernel(function, blocks, 1, 1, threads, 1, 1, shared_bytes, stream, addresses, nullptr) ==
                   hip_success_k
               ? status_t::success_k
               : status_t::device_error_k;
}

static_assert(hip_reservable_allocator<hip_virtual_memory_allocator<byte_t>>);
static_assert(hip_reservable_allocator<hip_managed_allocator<byte_t>>);
static_assert(hip_allocator<hip_device_allocator_t>);
static_assert(hip_allocator<hip_arena_allocator_t>);
static_assert(hip_allocator<hip_pinned_allocator_t>);

/** One stream's device-side scratch for driving an AMD index: the traversal arena, the batch
 *  staging, the output staging, the telemetry counters and the attend miss word, riding the stream
 *  its allocator names and never opening one. Those buffers come from that stream's pool and only
 *  grow, so a context kept across calls allocates nothing once warm; the allocator carves only a
 *  call's own scratch, from a run its owner rewinds. Two contexts on one index share only the
 *  storage and the published counter, never scratch. Callers observe it through @c results,
 *  @ref spent, @ref missed and @ref matched, its scratch being the index's verbs' alone. */
template <hip_allocator allocator_type_ = hip_device_allocator_t>
struct hip_traversal_context {
    using allocator_t = allocator_type_;
    using kernel_table_t = hip_kernel_table_t;

    /** The single-query verb's staging, grown to the walk's breadth in pinned pages: the ascending
     *  matches, then, past them, the IDs and distances the device copies back. */
    struct results_t {
        growing_buffer<scored_node_t, hip_pinned_allocator_t> entries_;
        std::size_t count_ = 0;

        scored_node_t* data() noexcept { return entries_.data(); }
        scored_node_t const* data() const noexcept { return entries_.data(); }
        scored_node_t& operator[](std::size_t rank) noexcept { return entries_[rank]; }
        scored_node_t const& operator[](std::size_t rank) const noexcept { return entries_[rank]; }
        std::size_t size() const noexcept { return count_; }
    };

    results_t results;

    /** Binds this context to @p allocator's stream and device, selecting that device first, since
     *  the event is born on the current device, allocates only the fixed-size counters from the
     *  stream's pool, and takes their host twins from the stream's pinned slots; sized buffers wait
     *  for a verb. */
    hip_traversal_context(allocator_t allocator = {}) noexcept
        : results {growing_buffer<scored_node_t, hip_pinned_allocator_t> {
              hip_pinned_allocator_t {allocator.api(), allocator.device()}}},
          allocator_ {(hip_set_device(allocator.api(), allocator.device()), allocator)}, //
          telemetry_ {hip_make_unique(telemetry_allocator_t {device_allocator_t {allocator_}}, 1)},
          assigned_nodes_ {hip_make_unique(counter_allocator_t {device_allocator_t {allocator_}}, 2)},
          missed_ {hip_make_unique(counter_allocator_t {device_allocator_t {allocator_}}, 1)},
          host_counters_ {take_counters_(allocator_.pinned_slots())},
          arena_ {nullptr, hip_deleter<device_allocator_t> {device_allocator_t {allocator_}, 0}},
          staging_ {nullptr, hip_deleter<device_allocator_t> {device_allocator_t {allocator_}, 0}},
          event_ {allocator_.api()} {
        if (telemetry_) api().fill_async(telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream());
        if (missed_) api().fill_async(missed_.get(), 0, sizeof(u32_t), stream());
    }

    hip_traversal_context(hip_traversal_context const&) = delete;
    hip_traversal_context& operator=(hip_traversal_context const&) = delete;

    /** Every owned resource frees in stream order, so a move is a member-wise steal and destruction
     *  needs no drain. */
    hip_traversal_context(hip_traversal_context&&) noexcept = default;
    ~hip_traversal_context() noexcept = default;

    hip_api_t const& api() const noexcept { return allocator_.api(); }
    hip_stream_handle_t stream() const noexcept { return allocator_.stream(); }
    int device() const noexcept { return allocator_.device(); }

    /** The allocator this context carves its scratch from, for a caller's own launches beside
     *  it. */
    allocator_t allocator() const noexcept { return allocator_; }

    /**
     *  @brief The table of @p fatbin on this context's stream, loaded through @p speller at the
     *      first ask there and kept until the stream dies.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The context rides no executor stream.
     *  - @c bad_alloc_k The table could not be allocated.
     *  - @c device_error_k The code object would not load, or lacks a required kernel.
     */
    template <typename speller_type_>
    expected<kernel_table_t const*> table_of(bytes_view_t fatbin, speller_type_ const& speller) const noexcept {
        hip_kernel_tables_t* const tables = allocator_.kernel_tables();
        if (!tables) return {nullptr, status_t::uninitialized_k};
        return tables->table_of(api(), device(), fatbin, speller);
    }

    /** Whether this context can walk at all: every device-side word and its host twin claimed, and
     *  its event born. */
    explicit operator bool() const noexcept {
        return telemetry_ && assigned_nodes_ && missed_ && host_counters_ && event_;
    }

    /** Waits for every transfer and launch queued on this context's stream - one stream's
     *  completion, the executor's @c drain being the fleet's - sleeping on the context's event
     *  instead of spinning a core. Selects the context's device first, since an event records only
     *  on its own device. */
    status_t wait() noexcept {
        if (hip_set_device(api(), device()) != hip_success_k) return status_t::device_error_k;
        if (status_t const recorded = record(); failed(recorded)) return recorded;
        return event_.wait();
    }

    /**
     *  @brief Marks everything queued on this context's stream so far as the submission
     *      @ref finished polls, behind the copies of the counters @ref spent and @ref missed
     *      read; the context's device must be current.
     *
     *  @par Statuses:
     *  - @c device_error_k The context has no counters.
     */
    status_t record() noexcept {
        hip_api_t const& runtime = api();
        if (!host_counters_ ||
            runtime.copy_async(&host_counters_->telemetry, telemetry_.get(), sizeof(vram_telemetry_block_t),
                               hip_copy_device_to_host_k, stream()) != hip_success_k ||
            runtime.copy_async(&host_counters_->missed, missed_.get(), sizeof(u32_t), hip_copy_device_to_host_k,
                               stream()) != hip_success_k)
            return status_t::device_error_k;
        return event_.record(stream());
    }

    /**
     *  @brief Polls whether the submission @ref record captured finished without blocking, so one
     *      host thread awaits whichever of several streams finishes first.
     *
     *  @par Statuses:
     *  - @c device_error_k The event was never created.
     */
    status_t finished() const noexcept { return event_.finished(); }

    /** Queues on this context's stream a wait for everything @p other recorded last, so work here
     *  follows work there without the host waiting; the two may sit on different devices. */
    status_t follow(hip_traversal_context const& other) noexcept {
        return api().stream_wait_event(stream(), other.event_, 0) == hip_success_k ? status_t::success_k
                                                                                   : status_t::device_error_k;
    }

    /** The telemetry of the last submission, zeroed when it was queued; read once it finished. */
    vram_telemetry_block_t spent() const noexcept {
        return host_counters_ ? host_counters_->telemetry : vram_telemetry_block_t {};
    }

    /** Whether the last attend visited a node its value lookup had no row for, which refuses it;
     *  read once it finished. */
    bool missed() const noexcept { return host_counters_ && host_counters_->missed != 0; }

    /**
     *  @brief Queues on this context's stream the copy of the device word @p total into the host
     *      twin @ref matched reads, behind everything queued here so far.
     *
     *  @par Statuses:
     *  - @c device_error_k The context has no counters, or the copy refused.
     */
    status_t copy_matched(std::size_t const* total) noexcept {
        return host_counters_ && api().copy_async(&host_counters_->matched, total, sizeof(std::size_t),
                                                  hip_copy_device_to_host_k, stream()) == hip_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** The word @ref copy_matched last copied, such as the matches a batch found; read once it
     *  finished. */
    std::size_t matched() const noexcept { return host_counters_ ? host_counters_->matched : 0; }

    /** The stream-ordered pool behind the allocator's stream, where the context's own buffers live,
     *  so they outlive every rewind of a run the allocator carves from. */
    using device_allocator_t = hip_device_allocator_t;
    using telemetry_allocator_t = hip_device_allocator<vram_telemetry_block_t>;
    using counter_allocator_t = hip_device_allocator<u32_t>;
    using bytes_ptr_t = hip_unique_ptr<byte_t, allocator_t>;
    using device_bytes_ptr_t = hip_unique_ptr<byte_t, device_allocator_t>;

    /** The host's twins of the device counters, copied behind each @ref record, and the word
     *  @ref copy_matched copies. */
    struct host_counters_t {
        vram_telemetry_block_t telemetry;
        u32_t missed;
        std::size_t matched;
    };
    static_assert(sizeof(host_counters_t) <= hip_pinned_slots_t::slot_bytes_k &&
                      alignof(host_counters_t) <= hip_pinned_slots_t::slot_bytes_k,
                  "A context's counters fit one pinned slot");

    /** Gives a context's counters back to the pinned slots of the stream they came from. */
    struct counters_return_t {
        hip_pinned_slots_t* pinned_slots = nullptr;
        void operator()(host_counters_t* counters) const noexcept { pinned_slots->give_back(counters); }
    };
    using host_counters_ptr_t = std::unique_ptr<host_counters_t, counters_return_t>;

    /** Zeroed counters in a slot of @p pinned_slots, or null when none is free or it cannot pin. */
    static host_counters_ptr_t take_counters_(hip_pinned_slots_t* pinned_slots) noexcept {
        void* const slot = pinned_slots ? pinned_slots->take() : nullptr;
        return {slot ? new (slot) host_counters_t {} : nullptr, counters_return_t {pinned_slots}};
    }

    allocator_t allocator_;

    /** Monotone epoch floor over the arena's visited slots, so the arena is zeroed only when
     *  freshly allocated or when the 32-bit epoch space wraps, never per launch. */
    u64_t epoch_floor_ = 0;

    /** The device-side counters of the last submission, zeroed when it was queued. */
    hip_unique_ptr<vram_telemetry_block_t, telemetry_allocator_t> telemetry_;

    /** Entries the current launch's batch warps have assigned themselves so far: a builder's nodes,
     *  or a walk's queries; a walk's second word counts the walks running. */
    hip_unique_ptr<u32_t, counter_allocator_t> assigned_nodes_;

    /** The value-lookup miss word: raised by the last walk at a node its value lookup had no row
     *  for, zeroed when it was queued. */
    hip_unique_ptr<u32_t, counter_allocator_t> missed_;

    /** Borrowed from the stream's page-locked slots, so the copies @ref record queues into it
     *  never wait on the device, and no context pins memory of its own. */
    host_counters_ptr_t host_counters_;
    device_bytes_ptr_t arena_;
    std::size_t arena_bytes_ = 0;
    device_bytes_ptr_t staging_;
    std::size_t staging_bytes_ = 0;

    /** The point on the stream @ref wait and @ref finished await. */
    hip_event_t event_;

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
            if (api().fill_async(arena_.get(), 0, arena_bytes_, stream()) != hip_success_k) {
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
     *  search's queries and result IDs. */
    byte_t* ensure_staging(std::size_t bytes) noexcept { return ensure_(staging_, staging_bytes_, bytes); }

    /** Grows @p buffer to @p bytes when @p capacity falls short; @return the base, or null. */
    byte_t* ensure_(device_bytes_ptr_t& buffer, std::size_t& capacity, std::size_t bytes) noexcept {
        if (bytes <= capacity) return buffer.get();
        // The deleter already carries the stream's pool, so only the allocation and its count
        // change; the old one frees behind the work queued on it.
        buffer.reset(buffer.get_deleter().allocator.allocate(bytes));
        buffer.get_deleter().count = bytes;
        capacity = buffer ? bytes : 0;
        return buffer.get();
    }
};

/** The streams a bulk verb advances on and the scratch they walk with - one of each per device,
 *  which every shard there shares in order. Work is queued, never waited on, so one host thread
 *  drives every device. */
struct hip_executor_t {
    using device_arena_t = hip_device_arena_t;
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::amd_k;

    /** One device's run of scratch, carved and rewound between calls; it only grows, by
     *  @ref reserve, since relocation would be a host loop over device bytes. */
    using arena_t = growing_arena<hip_device_allocator_t>;

    /** A walk's context on one stream, carving a call's scratch from that stream's run. */
    using traversal_context_t = hip_traversal_context<hip_arena_allocator_t>;

    /** One call's hold on the executor, released when it dies. */
    using claim_t = std::unique_lock<fu::spin_mutex_t>;

    /** One shard's stream, run and contexts; the pinned slots are declared before the stream, so
     *  they unpin only once it drained, then the arena and the contexts, so they free onto a live
     *  stream, the contexts first. */
    struct stream_t {

        /** The runtime every transfer, launch and allocation on this stream goes through. */
        hip_api_t const* api = nullptr;

        /** The HIP ordinal this stream was opened on - placement and launch target both. */
        int device = 0;

        /** The pinned slots every traversal context on this stream copies its counters into, held
         *  apart so its allocators keep their address as the stream moves. */
        std::unique_ptr<hip_pinned_slots_t> pinned_slots;

        /** The code objects loaded on this device, one table each, held apart so the contexts
         *  keep their address as the stream moves, and declared before the stream, so they unload
         *  only once it drained. */
        std::unique_ptr<hip_kernel_tables_t> kernel_tables;

        /** The ordering identity every transfer and launch on this device rides. */
        hip_stream_t handle;

        /** The point @ref wait and the C layer's arena record and sleep on, so a wait frees its
         *  core. */
        hip_event_t event;

        /** The scratch a call carves from, rewound and only grown by @ref reserve. */
        arena_t arena;

        /** The contexts @ref reserve keeps over @c arena, one per shard this stream serves, alive
         *  across calls with their buffers, events and pinned slots. */
        growing_buffer<traversal_context_t> contexts {global_allocator};

        /** Takes @p handle, opened on @p device, and binds the arena to it; the event is born
         *  there too, so @p device must be current. */
        stream_t(hip_api_t const& api, int device, hip_stream_t&& handle) noexcept
            : api(&api), device(device), pinned_slots(new (std::nothrow) hip_pinned_slots_t {api, device}),
              kernel_tables(new (std::nothrow) hip_kernel_tables_t), handle(std::move(handle)), event(api),
              arena(hip_device_allocator_t {*this}) {}

        /**
         *  @brief The table of @p fatbin on this device, loaded by @p speller at the first ask.
         *
         *  @par Statuses:
         *  - @c bad_alloc_k The table could not be allocated.
         *  - @c device_error_k The code object would not load, or lacks a required kernel.
         */
        template <typename speller_type_>
        expected<hip_kernel_table_t const*> table_of(bytes_view_t fatbin, speller_type_ const& speller) const noexcept {
            return kernel_tables->table_of(*api, device, fatbin, speller);
        }
    };

    hip_executor_t() = default;
    hip_executor_t(hip_executor_t&& other) noexcept { swap(other); }
    hip_executor_t& operator=(hip_executor_t&& other) noexcept { return swap(other), *this; }
    hip_executor_t(hip_executor_t const&) = delete;
    hip_executor_t& operator=(hip_executor_t const&) = delete;

    /** Exchanges the stream tables; @c held_ stays, as a move with a call in flight is misuse. */
    void swap(hip_executor_t& other) noexcept {
        std::swap(api_, other.api_);
        std::swap(streams_, other.streams_);
        std::swap(scratch_limit_, other.scratch_limit_);
    }

    /**
     *  @brief Claims the executor for one call.
     *
     *  @par Statuses:
     *  - @c contended_k Another call holds the executor.
     */
    expected<claim_t> claim() noexcept {
        if (!held_.lock_if_free()) return {{}, status_t::contended_k};
        return {claim_t {held_, std::adopt_lock}, status_t::success_k};
    }

    /**
     *  @brief Takes over streams the caller already opened, owned and destroyed here from now on.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p streams are empty.
     */
    static expected<hip_executor_t> acquire(growing_buffer<stream_t>&& streams) noexcept {
        if (streams.empty()) return {{}, status_t::config_mismatch_k};
        hip_executor_t executor;
        executor.api_ = streams[0].api;
        executor.streams_ = std::move(streams);
        return {std::move(executor), status_t::success_k};
    }

    /**
     *  @brief Opens one stream per entry of @p devices, binding each first, since a stream is born
     *      on the current device; their runs never grow past @p scratch_limit bytes, zero for none.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p devices are empty.
     */
    static expected<hip_executor_t> spawn(hip_api_t const& api, std::span<int const> devices,
                                          std::size_t scratch_limit = 0) noexcept {
        growing_buffer<stream_t> streams {global_allocator};
        if (failed(streams.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int device : devices) {
            if (hip_set_device(api, device) != hip_success_k) return {{}, status_t::device_error_k};
            stream_t opened {api, device, hip_stream_t {api}};
            if (!opened.pinned_slots || !opened.kernel_tables) return {{}, status_t::bad_alloc_k};
            if (!opened.handle || !opened.event) return {{}, status_t::device_error_k};
            streams.push_back_reserved(std::move(opened));
        }
        auto [executor, status] = acquire(std::move(streams));
        executor.scratch_limit_ = scratch_limit;
        return {std::move(executor), status};
    }

    /**
     *  @brief Rewinds every stream's run for the carves to come, growing it to
     *      @p bytes_per_stream where it holds less, and keeps @p contexts_per_stream contexts on
     *      each, opening only the missing ones; a warm call asks no more and allocates nothing.
     *      A wider run frees the old one in stream order behind whatever still rides it.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor has no streams.
     *  - @c bad_alloc_k The @p bytes_per_stream pass the scratch ceiling, or a run or a context
     *    could not grow.
     */
    status_t reserve(std::size_t bytes_per_stream, std::size_t contexts_per_stream = 0) noexcept {
        if (streams_.empty()) return status_t::uninitialized_k;
        if (scratch_limit_ && bytes_per_stream > scratch_limit_) return status_t::bad_alloc_k;
        for (stream_t& opened : streams_) {
            opened.arena.clear();
            // A rewound run holds one chunk, its widest, which a wider call replaces whole.
            if (opened.arena.capacity() < bytes_per_stream) {
                opened.arena.reset();
                if (failed(opened.arena.reserve_exactly(bytes_per_stream))) return status_t::bad_alloc_k;
            }
            if (opened.contexts.size() >= contexts_per_stream) continue;
            if (hip_set_device(*api_, opened.device) != hip_success_k) return status_t::device_error_k;
            if (failed(opened.contexts.reserve_exactly(contexts_per_stream))) return status_t::bad_alloc_k;
            while (opened.contexts.size() != contexts_per_stream) {
                opened.contexts.push_back_reserved(traversal_context_t {hip_arena_allocator_t {opened.arena, opened}});
                if (!opened.contexts.back()) return status_t::bad_alloc_k;
            }
        }
        return status_t::success_k;
    }

    /** The contexts @ref reserve keeps on stream @p index, the ones its shards borrow in the order
     *  they are paired with it. */
    std::span<traversal_context_t> traversals(std::size_t index) noexcept {
        return to_writable_span(streams_[index].contexts);
    }

    /** Waits for every transfer and launch queued on every stream - the fleet's completion -
     *  marking each stream with its device bound, then sleeping on each mark in turn. */
    status_t wait() noexcept {
        for (stream_t& opened : streams_)
            if (hip_set_device(*api_, opened.device) != hip_success_k || failed(opened.event.record(opened.handle)))
                return status_t::device_error_k;
        for (stream_t& opened : streams_)
            if (status_t const drained = opened.event.wait(); failed(drained)) return drained;
        return status_t::success_k;
    }

    /** The stream at @p index, in the order the devices were named. */
    stream_t& stream(std::size_t index) noexcept { return streams_[index]; }

    /** The HIP ordinal stream @p index drives - what a caller pairs its shards against. */
    int device(std::size_t index) const noexcept { return streams_[index].device; }

    /** The first stream on @p device, or null when the executor drives none there. */
    stream_t* stream_on(int device) noexcept {
        for (stream_t& opened : streams_)
            if (opened.device == device) return &opened;
        return nullptr;
    }

    /** Streams this executor drives, one per device. */
    std::size_t size() const noexcept { return streams_.size(); }

    /** The runtime every stream of this executor goes through. */
    hip_api_t const& api() const noexcept { return *api_; }

  private:

    /** The runtime the streams were opened through. */
    hip_api_t const* api_ = nullptr;

    /** One stream per named entry, in the caller's order. */
    growing_buffer<stream_t> streams_ {global_allocator};

    /** Held by the one caller driving the streams; never swapped. */
    fu::spin_mutex_t held_;

    /** The ceiling on each stream's run, which every call's scratch and outputs carve from; zero
     *  lets it grow as geometry asks. */
    std::size_t scratch_limit_ = 0;
};

/** The AMD types and HIP calls a vendor-neutral VRAM index declares itself with and queues
 *  through; every copy tells its direction apart by unified addressing. */
struct hip_vendor_t {
    using api_t = hip_api_t;
    using stream_handle_t = hip_stream_handle_t;
    using event_t = hip_event_t;
    using pinned_slots_t = hip_pinned_slots_t;
    using pinned_allocator_t = hip_pinned_allocator_t;
    using device_specs_t = hip_device_specs_t;
    using address_space_t = hip_address_space_t;
    using executor_t = hip_executor_t;
    using kernel_table_t = hip_kernel_table_t;

    template <typename allocator_type_>
    using traversal_context = hip_traversal_context<allocator_type_>;

    /**
     *  Makes HIP ordinal @p ordinal current on the calling thread.
     *
     *  @par Statuses:
     *  - @c device_error_k The @p ordinal is past the device listing.
     */
    static status_t select(api_t const& api, int ordinal) noexcept { return verdict_(hip_set_device(api, ordinal)); }

    /** Queues a copy of @p bytes from @p from to @p to on @p stream, in either direction. */
    static status_t copy(api_t const& api, void* to, void const* from, std::size_t bytes,
                         stream_handle_t stream) noexcept {
        return verdict_(api.copy_async(to, from, bytes, hip_copy_default_k, stream));
    }

    /** Queues a copy of @p height rows of @p width bytes between pitched buffers on @p stream. */
    static status_t copy_2d(api_t const& api, void* to, std::size_t to_pitch, void const* from, std::size_t from_pitch,
                            std::size_t width, std::size_t height, stream_handle_t stream) noexcept {
        return verdict_(api.copy_2d_async(to, to_pitch, from, from_pitch, width, height, hip_copy_default_k, stream));
    }

    /** Queues setting @p bytes at @p to to @p value on @p stream. */
    static status_t fill(api_t const& api, void* to, u8_t value, std::size_t bytes, stream_handle_t stream) noexcept {
        return verdict_(api.fill_async(to, value, bytes, stream));
    }

    /** Queues setting @p count 32-bit words at @p to to @p value on @p stream. */
    static status_t fill_words(api_t const& api, void* to, u32_t value, std::size_t count,
                               stream_handle_t stream) noexcept {
        return verdict_(api.fill_words_async(to, static_cast<int>(value), count, stream));
    }

    /** Queues on @p stream a wait for everything @p event recorded last. */
    static status_t follow(api_t const& api, stream_handle_t stream, event_t const& event) noexcept {
        return verdict_(api.stream_wait_event(stream, event, 0));
    }

  private:
    static status_t verdict_(hip_result_t result) noexcept {
        return result == hip_success_k ? status_t::success_k : status_t::device_error_k;
    }
};

} // namespace unum::usearch::utilities
