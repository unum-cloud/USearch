/**
 *  @brief Host-side vocabulary for driving one CUDA device - specs, streams, allocators.
 *  @author Ash Vardanian
 *  @file include/usearch/utilities/vram_device.hpp
 *  @date July 22, 2026
 *
 *  Everything here is host-side driver & runtime API code - no kernels, no device-side
 *  functions, hence the @c .hpp extension - so the whole layer compiles under a plain host
 *  compiler with the CUDA toolkit headers on the include path, and unit-tests without
 *  @c nvcc. Three groups of vocabulary: the device specs runtime dispatch consults, the
 *  owning stream & allocation handles, and the allocator concepts with their backends.
 *
 *  Two independent concepts split the roles the CPU covers with one @c std::allocator
 *  shape. The flat @c cuda_allocator carries @c std::allocator's exact call surface -
 *  @c allocate(n) / @c deallocate(p,n) - with the CUDA stream bound at construction,
 *  the way a NUMA-aware host allocator binds its domain: ordering is the allocator's
 *  identity, never a per-call parameter. The @c cuda_reservable_allocator is the plane
 *  backend: a reserve-once / commit-on-demand address range whose base pointer survives
 *  growth, so a node id stays one multiply away from its row. Nothing forces one type to
 *  model both - planes never free mid-life, scratch never needs a stable base.
 */
#pragma once

#include <cstddef> // `std::byte` for raw spans
#include <cstdint> // `std::uintptr_t` for device-pointer casts
#include <memory>  // `std::unique_ptr` for owned device allocations

#include <cuda.h>         // Driver API: `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`
#include <cuda_runtime.h> // Runtime API: `cudaMallocAsync`, `cudaMallocManaged`, `cudaHostAlloc`

#include <forkunion/topology.hpp> // `fu::volume_ram` - the managed planes' host-memory budget
#include <forkunion/types.hpp>    // `fu::round_up_to_multiple` & friends

#include "usearch/utilities/types.hpp"

namespace unum {
namespace usearch {

/** @brief GPU generations on the @c __CUDA_ARCH__ scale - a plain enum, so tiers order
 *      against each other and against that macro directly. */
enum cuda_arch_t : int {
    /** @brief The sm_80 floor: @c dp4a, packed-half & bf16x2 fma, @c cp.async. */
    cuda_arch_ampere_k = 800,
    /** @brief sm_89 adds the fp8 conversions - the e4m3/e5m2 kits' requirement. */
    cuda_arch_ada_k = 890,
    /** @brief sm_90 adds TMA, thread-block clusters, and distributed shared memory. */
    cuda_arch_hopper_k = 900,
    /** @brief sm_100 adds Tensor Memory, fp6/fp4 conversions, and scalar @c fma.f32.f16. */
    cuda_arch_blackwell_k = 1000,
};

/**
 *  @brief One GPU's hardware specification - the facts runtime dispatch consults, in the
 *      spirit of StringZilla's & NumKong's capability queries. Hardware only, never derived
 *      policy like reservation ceilings; queried once, copied freely.
 *
 *  The compute capability pair is the dispatch key for generation-specific code paths -
 *  Hopper's @b 9.x unlocks @c cp.async bulk tensor moves and distributed shared memory,
 *  Blackwell's @b 10.x re-tiers them - while @p sm_count sizes launch grids and
 *  @p vram_bytes bounds reservation policy decided elsewhere.
 */
struct cuda_device_specs_t {
    /** @brief The device ordinal every CUDA runtime & driver call routes through. */
    int ordinal = 0;
    /** @brief Streaming-multiprocessor count - the launch-grid currency. */
    int sm_count = 0;
    /** @brief Compute-capability major version - the hardware generation, e.g. 9 = Hopper. */
    int compute_major = 0;
    /** @brief Compute-capability minor version - the in-generation revision. */
    int compute_minor = 0;
    /** @brief Total device-resident memory in bytes. */
    std::size_t vram_bytes = 0;

    /** @brief This device's generation on the @c cuda_arch_t scale - the runtime side of
     *      the kit chains' architecture ceilings, like the CPU's detected capability mask. */
    cuda_arch_t arch() const noexcept { return static_cast<cuda_arch_t>(compute_major * 100 + compute_minor * 10); }

    /** @brief Queries the runtime for @p ordinal's specification - fallible like every
     *      probe of external state, so the value is only meaningful on @c success_k. */
    static expected<cuda_device_specs_t> try_query(int ordinal) noexcept {
        cudaDeviceProp properties = {};
        if (cudaGetDeviceProperties(&properties, ordinal) != cudaSuccess) return {{ordinal}, status_t::device_error_k};
        return {
            {ordinal, properties.multiProcessorCount, properties.major, properties.minor, properties.totalGlobalMem},
            status_t::success_k};
    }
};

/** @brief The address space an allocation lives in, in LLVM's sense of the word: the
 *      classification that decides who can dereference a pointer and what transfer path
 *      moves its bytes - not a PTX state space, all three are @c global to device code. */
enum class cuda_address_space_t : u32_t {
    /** @brief Plain device memory; explicit host-device copies required. */
    device_k,
    /** @brief Unified memory; migratable, host- and device-visible. */
    managed_k,
    /** @brief Page-locked host memory; the DMA source & sink for staging. */
    host_pinned_k,
};

/**
 *  @brief The device analog of @c std::allocator: the same @c allocate(n) / @c deallocate(p, n)
 *      call surface, with the ordering @c cudaStream_t and the target device bound at
 *      construction - so rebinds and traits stay line-for-line with host allocators, and
 *      every allocation is implicitly stream-ordered without leaking ordering context into
 *      call sites. @c device() names the GPU the memory is pinned to, and @c address_space_k
 *      fixes host visibility.
 */
template <typename allocator_>
concept cuda_allocator = std::constructible_from<allocator_, cudaStream_t, int> &&
                         requires(allocator_ allocator, typename allocator_::value_type* pointer, std::size_t count) {
                             typename allocator_::value_type;
                             { allocator_::address_space_k } -> std::convertible_to<cuda_address_space_t>;
                             { allocator.allocate(count) } -> std::same_as<typename allocator_::value_type*>;
                             { allocator.deallocate(pointer, count) } -> std::same_as<void>;
                             { allocator.device() } -> std::convertible_to<int>;
                         };

/**
 *  @brief The plane backend contract: a reserve-once / commit-on-demand address range whose
 *      base never moves - growth maps fresh physical pages without relocating a single
 *      node. The virtual-memory backend reserves address space and maps slabs; the managed
 *      backend reserves the whole ceiling in one allocation and lets the pager commit
 *      pages. Deliberately not a refinement of @c cuda_allocator: planes never free
 *      mid-life, so the flat surface would be dead weight on every model.
 */
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

/**
 *  @brief Virtual-memory-management plane backend: a fixed driver-API address reservation
 *      backed by physical slabs mapped on demand, so the base pointer never moves and node
 *      addressing stays one multiply. Mapping is host-driven and always completes ahead of
 *      any kernel that could touch the fresh range - growth hazards cannot reach device
 *      code. Fresh slabs arrive zeroed. The in-VRAM performance default for the planes.
 */
template <typename value_type_ = std::byte>
struct cuda_vmm_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = cuda_vmm_allocator<other_>;
    };

    /** @brief Preferred physical slab, rounded to the driver's granularity at reservation -
     *      large enough to amortize mapping calls, small enough for incremental growth. */
    inline static constexpr std::size_t preferred_slab_bytes_k = 128u << 20;

    CUdeviceptr base_ = 0;
    std::size_t va_bytes_ = 0;
    std::size_t mapped_bytes_ = 0;
    std::size_t slab_bytes_ = 0;
    int device_ = 0;

    cuda_vmm_allocator() = default;
    /** @brief Names the GPU the future reservation pins to - placement only, no driver call. */
    explicit cuda_vmm_allocator(int device) noexcept : device_(device) {}
    /** @brief Copies the placement configuration only - a live reservation is never
     *      copyable; indexes copy allocators before any reservation exists, the STL way. */
    cuda_vmm_allocator(cuda_vmm_allocator const& other) noexcept : device_(other.device_) {}
    template <typename other_>
    cuda_vmm_allocator(cuda_vmm_allocator<other_> const& other) noexcept : device_(other.device_) {}
    cuda_vmm_allocator& operator=(cuda_vmm_allocator const&) = delete;
    ~cuda_vmm_allocator() noexcept { release(); }

    /** @brief Reserves @p va_bytes of virtual address space on @p device - no physical
     *      memory yet, so the ceiling costs nothing and should be generous. */
    status_t reserve_range(std::size_t va_bytes, int device) noexcept {
        CUmemAllocationProp properties = allocation_properties_(device);
        std::size_t granularity = 0;
        if (cuMemGetAllocationGranularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
            return status_t::device_error_k;
        slab_bytes_ = max_of(granularity, preferred_slab_bytes_k / granularity * granularity);
        va_bytes_ = fu::round_up_to_multiple(va_bytes, slab_bytes_);
        device_ = device;
        if (cuMemAddressReserve(&base_, va_bytes_, 0, 0, 0) != CUDA_SUCCESS) return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /**
     *  @brief Maps physical slabs until at least @p bytes are usable; never unmaps early.
     *
     *  Each allocation handle is released right after a successful mapping: the driver
     *  reference-counts them, so the mapping alone keeps the physical memory alive and no
     *  handle registry is needed - the destructor's unmap is the final release.
     */
    status_t commit(std::size_t bytes) noexcept {
        if (bytes > va_bytes_) return status_t::overflow_k;
        CUmemAllocationProp properties = allocation_properties_(device_);
        CUmemAccessDesc access = {};
        access.location = properties.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        while (mapped_bytes_ < bytes) {
            CUmemGenericAllocationHandle handle;
            if (cuMemCreate(&handle, slab_bytes_, &properties, 0) != CUDA_SUCCESS) return status_t::bad_alloc_k;
            bool const mapped = cuMemMap(base_ + mapped_bytes_, slab_bytes_, 0, handle, 0) == CUDA_SUCCESS &&
                                cuMemSetAccess(base_ + mapped_bytes_, slab_bytes_, &access, 1) == CUDA_SUCCESS;
            cuMemRelease(handle);
            if (!mapped) return status_t::device_error_k;
            mapped_bytes_ += slab_bytes_;
        }
        return status_t::success_k;
    }

    value_type* base() const noexcept { return reinterpret_cast<value_type*>(static_cast<std::uintptr_t>(base_)); }
    std::size_t reserved_bytes() const noexcept { return va_bytes_; }
    std::size_t committed_bytes() const noexcept { return mapped_bytes_; }
    int device() const noexcept { return device_; }

    void release() noexcept {
        if (mapped_bytes_) cuMemUnmap(base_, mapped_bytes_);
        if (base_) cuMemAddressFree(base_, va_bytes_);
        base_ = 0, va_bytes_ = 0, mapped_bytes_ = 0;
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

/**
 *  @brief Unified-memory plane backend: one @c cudaMallocManaged reservation for the whole
 *      ceiling, pages committing on touch, @c cudaMemAdvise pinning their preferred home to
 *      the device while the host stays an allowed accessor. Address-stable by construction
 *      and oversubscribable past VRAM - the drop-in for datasets beyond device memory and
 *      the natural Grace-coherence path. @c commit is a residency hint, never correctness.
 */
template <typename value_type_ = std::byte>
struct cuda_managed_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::managed_k;
    template <typename other_>
    struct rebind {
        using other = cuda_managed_allocator<other_>;
    };

    value_type* base_ = nullptr;
    std::size_t va_bytes_ = 0;
    std::size_t committed_bytes_ = 0;
    int device_ = 0;

    cuda_managed_allocator() = default;
    /** @brief Names the GPU the future reservation prefers - placement only, no driver call. */
    explicit cuda_managed_allocator(int device) noexcept : device_(device) {}
    /** @brief Copies the placement configuration only, never a live reservation. */
    cuda_managed_allocator(cuda_managed_allocator const& other) noexcept : device_(other.device_) {}
    template <typename other_>
    cuda_managed_allocator(cuda_managed_allocator<other_> const& other) noexcept : device_(other.device_) {}
    cuda_managed_allocator& operator=(cuda_managed_allocator const&) = delete;
    ~cuda_managed_allocator() noexcept { release(); }

    status_t reserve_range(std::size_t va_bytes, int device) noexcept {
        device_ = device;
        va_bytes_ = va_bytes;
        if (cudaMallocManaged(reinterpret_cast<void**>(&base_), va_bytes, cudaMemAttachGlobal) != cudaSuccess)
            return status_t::bad_alloc_k;
        cudaMemLocation device_location = {};
        device_location.type = cudaMemLocationTypeDevice;
        device_location.id = device;
        cudaMemLocation host_location = {};
        host_location.type = cudaMemLocationTypeHost;
        cudaMemAdvise(base_, va_bytes, cudaMemAdviseSetPreferredLocation, device_location);
        cudaMemAdvise(base_, va_bytes, cudaMemAdviseSetAccessedBy, host_location);
        return status_t::success_k;
    }

    status_t commit(std::size_t bytes) noexcept {
        if (bytes > va_bytes_) return status_t::overflow_k;
        // Managed pages are not guaranteed zeroed, and the planes' sentinel fills assume
        // fresh rows arrive zeroed like virtual-memory slabs do - so growth zeroes exactly
        // the fresh range on the device, which also commits its residency there.
        if (bytes > committed_bytes_) {
            if (cudaMemset(reinterpret_cast<std::byte*>(base_) + committed_bytes_, 0, bytes - committed_bytes_) !=
                cudaSuccess)
                return status_t::device_error_k;
            committed_bytes_ = bytes;
        }
        return status_t::success_k;
    }

    value_type* base() const noexcept { return base_; }
    std::size_t reserved_bytes() const noexcept { return va_bytes_; }
    std::size_t committed_bytes() const noexcept { return committed_bytes_; }
    int device() const noexcept { return device_; }

    /** @brief The reservation ceiling this backend budgets against: half the physical
     *      host RAM - managed reservations are real allocations, and claiming it all
     *      starves the host of staging buffers and page cache. Device planes budget
     *      against VRAM instead, so only this backend consults the topology. */
    static std::size_t budget_bytes() noexcept { return fu::volume_ram() / 2; }

    void release() noexcept {
        if (base_) cudaFree(base_);
        base_ = nullptr, va_bytes_ = 0, committed_bytes_ = 0;
    }
};

/** @brief Stream-ordered device allocator for transient scratch - traversal arenas,
 *      staging rings, transport buffers. The stream given at construction orders every
 *      @c allocate / @c deallocate; a default-constructed instance rides the legacy
 *      default stream. */
template <typename value_type_ = std::byte>
struct cuda_device_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = cuda_device_allocator<other_>;
    };

    cudaStream_t stream_ = nullptr;
    int device_ = 0;

    cuda_device_allocator() noexcept = default;
    cuda_device_allocator(cudaStream_t stream, int device = 0) noexcept : stream_(stream), device_(device) {}
    template <typename other_>
    cuda_device_allocator(cuda_device_allocator<other_> const& other) noexcept
        : stream_(other.stream_), device_(other.device_) {}

    value_type* allocate(std::size_t count) noexcept {
        void* pointer = nullptr;
        return cudaMallocAsync(&pointer, count * sizeof(value_type), stream_) == cudaSuccess
                   ? static_cast<value_type*>(pointer)
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept { cudaFreeAsync(pointer, stream_); }
    cudaStream_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }
};

/** @brief Page-locked host allocator - the DMA source for uploads and sink for results.
 *      Portable pinning, so one buffer serves any device. Pinning itself is synchronous;
 *      the constructor's stream is carried only as placement identity, matching the flat
 *      allocator shape. */
template <typename value_type_ = std::byte>
struct cuda_pinned_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::host_pinned_k;
    template <typename other_>
    struct rebind {
        using other = cuda_pinned_allocator<other_>;
    };

    cudaStream_t stream_ = nullptr;
    int device_ = 0;

    cuda_pinned_allocator() noexcept = default;
    cuda_pinned_allocator(cudaStream_t stream, int device = 0) noexcept : stream_(stream), device_(device) {}
    template <typename other_>
    cuda_pinned_allocator(cuda_pinned_allocator<other_> const& other) noexcept
        : stream_(other.stream_), device_(other.device_) {}

    value_type* allocate(std::size_t count) noexcept {
        void* pointer = nullptr;
        return cudaHostAlloc(&pointer, count * sizeof(value_type), cudaHostAllocPortable) == cudaSuccess
                   ? static_cast<value_type*>(pointer)
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept { cudaFreeHost(pointer); }
    cudaStream_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }
};

/** @brief Owned non-blocking CUDA stream - the ordering identity a driving context binds
 *      its allocator and launches to. Movable so contexts relocate; destruction drains the
 *      stream before destroying it, so in-flight work never outlives its owner. */
struct cuda_stream_t {
    cudaStream_t handle_ = nullptr;

    cuda_stream_t() noexcept { cudaStreamCreateWithFlags(&handle_, cudaStreamNonBlocking); }
    cuda_stream_t(cuda_stream_t&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    cuda_stream_t(cuda_stream_t const&) = delete;
    cuda_stream_t& operator=(cuda_stream_t const&) = delete;
    ~cuda_stream_t() noexcept {
        if (handle_) cudaStreamSynchronize(handle_), cudaStreamDestroy(handle_);
    }

    operator cudaStream_t() const noexcept { return handle_; }
};

/** @brief The @c std::unique_ptr deleter closing the loop on @c cuda_allocator: it carries
 *      the allocator - and with it the stream and device identity - plus the element count
 *      the flat surface demands back at deallocation. */
template <cuda_allocator allocator_>
struct cuda_deleter {
    allocator_ allocator;
    std::size_t count = 0;

    void operator()(typename allocator_::value_type* pointer) noexcept { allocator.deallocate(pointer, count); }
};

/** @brief Owned device (or pinned) allocation - @c std::unique_ptr over a @c cuda_allocator.
 *      Host code never dereferences it: views take @c .get(), transfers take @c cudaMemcpy. */
template <typename value_type_, cuda_allocator allocator_ = cuda_device_allocator<value_type_>>
using cuda_unique_ptr = std::unique_ptr<value_type_, cuda_deleter<allocator_>>;

/** @brief Allocates @p count elements through @p allocator into an owning pointer - the
 *      @c std::make_unique analog, with null signalling exhaustion instead of a throw. The
 *      bytes arrive uninitialized; callers that need zeros issue their own stream-ordered
 *      memset, which the allocator's stream then orders. */
template <cuda_allocator allocator_>
cuda_unique_ptr<typename allocator_::value_type, allocator_> cuda_make_unique(allocator_ allocator,
                                                                              std::size_t count) noexcept {
    typename allocator_::value_type* pointer = allocator.allocate(count);
    return {pointer, cuda_deleter<allocator_> {allocator, count}};
}

static_assert(cuda_reservable_allocator<cuda_vmm_allocator<std::byte>>);
static_assert(cuda_reservable_allocator<cuda_managed_allocator<std::byte>>);
static_assert(cuda_allocator<cuda_device_allocator<std::byte>>);
static_assert(cuda_allocator<cuda_pinned_allocator<std::byte>>);

} // namespace usearch
} // namespace unum
