/**
 *  @brief Host-side vocabulary for driving one CUDA device - specs, streams, allocators.
 *  @author Ash Vardanian
 *  @file include/usearch/utilities/vram_device.hpp
 *  @date July 22, 2026
 *
 *  Host-side driver & runtime API code only - no kernels, hence the @c .hpp extension - so the
 *  layer compiles under a plain host compiler with the toolkit headers on the include path.
 *  The flat @c cuda_allocator carries @c std::allocator's call surface with the stream bound
 *  at construction; the @c cuda_reservable_allocator is the plane backend, a reserve-once /
 *  commit-on-demand range whose base survives growth. Nothing models both.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` for device-pointer casts

#include <memory>  // `std::unique_ptr` for owned device allocations
#include <mutex>   // `std::unique_lock`, `std::adopt_lock` - the executor's one-caller hold
#include <span>    // `std::span` - the device list an executor opens over
#include <utility> // `std::move` - relocating an opened context into its table

#include <cuda.h>         // Driver API: `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`
#include <cuda_runtime.h> // Runtime API: `cudaMallocAsync`, `cudaMallocManaged`, `cudaHostAlloc`

#include <forkunion/capabilities.hpp> // `fu::preferred_yield_t` - the hold's spin policy
#include <forkunion/topology.hpp>     // `fu::volume_ram` - the managed planes' host-memory budget
#include <forkunion/types.hpp>        // `fu::round_up_to_multiple`, `fu::spin_mutex`

#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - an executor's table of contexts
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

/** @brief One GPU's hardware specification - the facts runtime dispatch consults; hardware only,
 *      never derived policy. */
struct cuda_device_specs_t {
    /** @brief The device ordinal every CUDA runtime & driver call routes through. */
    int ordinal = 0;
    /** @brief Streaming-multiprocessor count - the launch-grid currency. */
    int multiprocessor_count = 0;
    /** @brief Compute-capability major version - the hardware generation, e.g. 9 = Hopper. */
    int compute_major = 0;
    /** @brief Compute-capability minor version - the in-generation revision. */
    int compute_minor = 0;
    /** @brief Total device-resident memory in bytes. */
    std::size_t vram_bytes = 0;
    /** @brief Widest dynamic shared memory one block can be granted at launch, on explicit request. */
    std::size_t shared_memory_per_block_bytes = 0;

    /** @brief This device's generation on the @c cuda_arch_t scale - the kit chains' runtime ceiling. */
    cuda_arch_t arch() const noexcept { return static_cast<cuda_arch_t>(compute_major * 100 + compute_minor * 10); }

    /** @brief Queries the runtime for @p ordinal's specification; meaningful only on @c success_k. */
    static expected<cuda_device_specs_t> query(int ordinal) noexcept {
        cudaDeviceProp properties = {};
        if (cudaGetDeviceProperties(&properties, ordinal) != cudaSuccess) return {{ordinal}, status_t::device_error_k};
        return {{ordinal, properties.multiProcessorCount, properties.major, properties.minor, properties.totalGlobalMem,
                 properties.sharedMemPerBlockOptin},
                status_t::success_k};
    }
};

/** @brief The address space an allocation lives in, in LLVM's sense: who can dereference the
 *      pointer and what transfer path moves its bytes; all three are @c global to device code. */
enum class cuda_address_space_t : u32_t {
    /** @brief Plain device memory; explicit host-device copies required. */
    device_k,
    /** @brief Unified memory; migratable, host- and device-visible. */
    managed_k,
    /** @brief Page-locked host memory; the DMA source & sink for staging. */
    host_pinned_k,
};

/** @brief The device analog of @c std::allocator, with the ordering stream and the target device
 *      bound at construction; @c address_space_k fixes host visibility. */
template <typename allocator_>
concept cuda_allocator = requires(allocator_ allocator, typename allocator_::value_type* pointer, std::size_t count) {
    typename allocator_::value_type;
    { allocator_::address_space_k } -> std::convertible_to<cuda_address_space_t>;
    { allocator.allocate(count) } -> std::same_as<typename allocator_::value_type*>;
    { allocator.deallocate(pointer, count) } -> std::same_as<void>;
    { allocator.stream() } -> std::convertible_to<cudaStream_t>;
    { allocator.device() } -> std::convertible_to<int>;
};

/** @brief The plane backend contract: a reserve-once / commit-on-demand address range whose base
 *      never moves. Not a refinement of @c cuda_allocator, since planes never free mid-life. */
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

/** @brief Virtual-memory plane backend: a fixed driver-API address reservation backed by slabs
 *      mapped on demand, host-driven ahead of any kernel; fresh slabs arrive zeroed. */
template <typename value_type_ = byte_t>
struct cuda_virtual_memory_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = cuda_virtual_memory_allocator<other_>;
    };

    /** @brief Preferred physical slab, rounded to the driver's granularity at reservation. */
    inline static constexpr std::size_t preferred_slab_bytes_k = 128u << 20;

    CUdeviceptr base_ = 0;
    std::size_t virtual_address_bytes_ = 0;
    std::size_t mapped_bytes_ = 0;
    std::size_t slab_bytes_ = 0;
    int device_ = 0;

    cuda_virtual_memory_allocator() = default;
    /** @brief Names the GPU the future reservation pins to - placement only, no driver call. */
    explicit cuda_virtual_memory_allocator(int device) noexcept : device_(device) {}
    /** @brief Copies the placement only, never a live reservation. */
    cuda_virtual_memory_allocator(cuda_virtual_memory_allocator const& other) noexcept : device_(other.device_) {}
    template <typename other_>
    cuda_virtual_memory_allocator(cuda_virtual_memory_allocator<other_> const& other) noexcept
        : device_(other.device_) {}
    cuda_virtual_memory_allocator& operator=(cuda_virtual_memory_allocator const&) = delete;
    ~cuda_virtual_memory_allocator() noexcept { release(); }

    /** @brief Reserves @p virtual_address_bytes of virtual address space on @p device - no physical
     *      memory yet, so the ceiling costs nothing and should be generous. */
    status_t reserve_range(std::size_t virtual_address_bytes, int device) noexcept {
        CUmemAllocationProp properties = allocation_properties_(device);
        std::size_t granularity = 0;
        if (cuMemGetAllocationGranularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
            return status_t::device_error_k;
        slab_bytes_ = max_of(granularity, preferred_slab_bytes_k / granularity * granularity);
        virtual_address_bytes_ = fu::round_up_to_multiple(virtual_address_bytes, slab_bytes_);
        device_ = device;
        if (cuMemAddressReserve(&base_, virtual_address_bytes_, 0, 0, 0) != CUDA_SUCCESS) return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /** @brief Maps physical slabs until at least @p bytes are usable; never unmaps early. Each
     *      handle is released right after mapping, since the mapping alone keeps the memory alive. */
    status_t commit(std::size_t bytes) noexcept {
        if (bytes > virtual_address_bytes_) return status_t::overflow_k;
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
    std::size_t reserved_bytes() const noexcept { return virtual_address_bytes_; }
    std::size_t committed_bytes() const noexcept { return mapped_bytes_; }
    int device() const noexcept { return device_; }

    void release() noexcept {
        if (mapped_bytes_) cuMemUnmap(base_, mapped_bytes_);
        if (base_) cuMemAddressFree(base_, virtual_address_bytes_);
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

/** @brief Unified-memory plane backend: one @c cudaMallocManaged reservation for the whole ceiling,
 *      preferring the device and oversubscribable past VRAM. */
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

    status_t reserve_range(std::size_t virtual_address_bytes, int device) noexcept {
        device_ = device;
        virtual_address_bytes_ = virtual_address_bytes;
        if (cudaMallocManaged(reinterpret_cast<void**>(&base_), virtual_address_bytes, cudaMemAttachGlobal) !=
            cudaSuccess)
            return status_t::bad_alloc_k;
        cudaMemLocation device_location = {};
        device_location.type = cudaMemLocationTypeDevice;
        device_location.id = device;
        cudaMemLocation host_location = {};
        host_location.type = cudaMemLocationTypeHost;
        cudaMemAdvise(base_, virtual_address_bytes, cudaMemAdviseSetPreferredLocation, device_location);
        cudaMemAdvise(base_, virtual_address_bytes, cudaMemAdviseSetAccessedBy, host_location);
        return status_t::success_k;
    }

    status_t commit(std::size_t bytes) noexcept {
        if (bytes > virtual_address_bytes_) return status_t::overflow_k;
        if (bytes <= committed_bytes_) return status_t::success_k;
        // Managed pages arrive unzeroed, unlike slabs; zeroing the fresh range also commits its residency.
        if (cudaMemset(reinterpret_cast<byte_t*>(base_) + committed_bytes_, 0, bytes - committed_bytes_) != cudaSuccess)
            return status_t::device_error_k;
        committed_bytes_ = bytes;
        return status_t::success_k;
    }

    value_type* base() const noexcept { return base_; }
    std::size_t reserved_bytes() const noexcept { return virtual_address_bytes_; }
    std::size_t committed_bytes() const noexcept { return committed_bytes_; }
    int device() const noexcept { return device_; }

    /** @brief The reservation ceiling: half the host RAM, since managed reservations are real
     *      allocations; only this backend consults the topology. */
    static std::size_t budget_bytes() noexcept { return fu::volume_ram() / 2; }

    void release() noexcept {
        if (base_) cudaFree(base_);
        base_ = nullptr, virtual_address_bytes_ = 0, committed_bytes_ = 0;
    }
};

/** @brief Stream-ordered device allocator for transient scratch; a null stream rides the legacy
 *      default. */
template <typename value_type_ = byte_t>
struct cuda_device_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = cuda_device_allocator<other_>;
    };

    cuda_device_allocator() noexcept = default;
    cuda_device_allocator(cudaStream_t stream, int device) noexcept : stream_(stream), device_(device) {}
    template <typename other_>
    cuda_device_allocator(cuda_device_allocator<other_> const& other) noexcept
        : stream_(other.stream()), device_(other.device()) {}

    value_type* allocate(std::size_t count) noexcept {
        void* pointer = nullptr;
        return cudaMallocAsync(&pointer, count * sizeof(value_type), stream_) == cudaSuccess
                   ? static_cast<value_type*>(pointer)
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept { cudaFreeAsync(pointer, stream_); }
    cudaStream_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }

  private:
    cudaStream_t stream_ = nullptr;
    int device_ = 0;
};

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
    using arena_t = growing_buffer<byte_t, cuda_device_allocator<byte_t>>;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = cuda_arena_allocator<other_>;
    };

    cuda_arena_allocator(arena_t& arena, cudaStream_t stream, int device) noexcept
        : arena_(arena), stream_(stream), device_(device) {}
    /** @brief The rebinding conversion @c std::allocator_traits performs - the identity rides
     *      along, and the run is shared rather than copied. */
    template <typename other_>
    constexpr cuda_arena_allocator(cuda_arena_allocator<other_> const& other) noexcept
        : arena_(other.arena()), stream_(other.stream()), device_(other.device()) {}

    value_type* allocate(std::size_t count) noexcept {
        // Host-side pointer arithmetic over device bytes: the carve never touches them.
        return arena_allocator<value_type_, cuda_device_allocator<byte_t>> {arena_}.allocate(count);
    }
    /** @brief A carve is never returned; the run's owner rewinds it between calls. */
    void deallocate(value_type*, std::size_t) noexcept {}

    cudaStream_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }
    /** @brief The run every rebound copy shares. */
    arena_t& arena() const noexcept { return arena_; }
    bool operator==(cuda_arena_allocator const& other) const noexcept { return &arena_ == &other.arena_; }

  private:
    arena_t& arena_;
    cudaStream_t stream_ = nullptr;
    int device_ = 0;
};

/** @brief Page-locked host allocator - the DMA source and sink; pinning is synchronous, the
 *      stream is identity only. */
template <typename value_type_ = byte_t>
struct cuda_pinned_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::host_pinned_k;
    template <typename other_>
    struct rebind {
        using other = cuda_pinned_allocator<other_>;
    };

    cuda_pinned_allocator() noexcept = default;
    cuda_pinned_allocator(cudaStream_t stream, int device) noexcept : stream_(stream), device_(device) {}
    template <typename other_>
    cuda_pinned_allocator(cuda_pinned_allocator<other_> const& other) noexcept
        : stream_(other.stream()), device_(other.device()) {}

    value_type* allocate(std::size_t count) noexcept {
        void* pointer = nullptr;
        return cudaHostAlloc(&pointer, count * sizeof(value_type), cudaHostAllocPortable) == cudaSuccess
                   ? static_cast<value_type*>(pointer)
                   : nullptr;
    }
    void deallocate(value_type* pointer, std::size_t) noexcept { cudaFreeHost(pointer); }
    cudaStream_t stream() const noexcept { return stream_; }
    int device() const noexcept { return device_; }

  private:
    cudaStream_t stream_ = nullptr;
    int device_ = 0;
};

/** @brief Owned non-blocking stream; destruction drains it first. */
struct cuda_stream_t {
    cuda_stream_t() noexcept { cudaStreamCreateWithFlags(&handle_, cudaStreamNonBlocking); }
    cuda_stream_t(cuda_stream_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    cuda_stream_t(cuda_stream_t const&) = delete;
    cuda_stream_t& operator=(cuda_stream_t const&) = delete;
    ~cuda_stream_t() noexcept {
        if (handle_) cudaStreamSynchronize(handle_), cudaStreamDestroy(handle_);
    }

    operator cudaStream_t() const noexcept { return handle_; }

  private:
    cudaStream_t handle_ = nullptr;
};

/** @brief The streams a bulk verb advances on and the scratch they walk with - one of each per
 *      device. Work is queued, never waited on, so one host thread drives every device. */
struct cuda_executor_t {
    /** @brief One device's run of scratch; sized once by @ref reserve, since relocation would be
     *      a host loop over device bytes. */
    using arena_t = growing_buffer<byte_t, cuda_device_allocator<byte_t>>;
    /** @brief One call's hold on the executor, released when it dies. */
    using claim_t = std::unique_lock<fu::spin_mutex<fu::preferred_yield_t>>;

    /** @brief One device's stream and run; the arena is declared last so it frees onto a live
     *      stream. */
    struct stream_t {
        /** @brief The device this stream was opened on - placement and launch target both. */
        int device = 0;
        /** @brief The ordering identity every transfer and launch on this device rides. */
        cuda_stream_t handle;
        /** @brief The scratch a walk carves its state from, sized once by @ref reserve. */
        arena_t arena;

        /** @brief Opens a stream on the current device and binds the arena to it. */
        explicit stream_t(int device) noexcept
            : device(device), arena(cuda_device_allocator<byte_t> {handle, device}) {}
    };

    cuda_executor_t() = default;
    cuda_executor_t(cuda_executor_t&& other) noexcept { swap(other); }
    cuda_executor_t& operator=(cuda_executor_t&& other) noexcept { return swap(other), *this; }
    cuda_executor_t(cuda_executor_t const&) = delete;
    cuda_executor_t& operator=(cuda_executor_t const&) = delete;

    /** @brief Exchanges the stream tables; @c held_ stays, since a move with a call in flight is
     *      misuse. */
    void swap(cuda_executor_t& other) noexcept { std::swap(streams_, other.streams_); }

    /** @brief Claims the executor for one call; a second concurrent caller gets @c contended_k. */
    expected<claim_t> claim() noexcept {
        if (!held_.lock_if_free()) return {{}, status_t::contended_k};
        return {claim_t {held_, std::adopt_lock}, status_t::success_k};
    }

    /** @brief Takes over streams the caller already opened, owned and destroyed here from now on. */
    static expected<cuda_executor_t> acquire(growing_buffer<stream_t>&& streams) noexcept {
        if (streams.empty()) return {{}, status_t::config_mismatch_k};
        cuda_executor_t executor;
        executor.streams_ = std::move(streams);
        return {std::move(executor), status_t::success_k};
    }

    /** @brief Opens one stream per named device, binding each first, since a stream is born on
     *      the current device. */
    static expected<cuda_executor_t> spawn(std::span<int const> devices) noexcept {
        growing_buffer<stream_t> streams {global_allocator};
        if (failed(streams.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
        for (int device : devices) {
            if (cudaSetDevice(device) != cudaSuccess) return {{}, status_t::device_error_k};
            stream_t opened {device};
            if (!opened.handle) return {{}, status_t::device_error_k};
            streams.push_back_reserved(std::move(opened));
        }
        return acquire(std::move(streams));
    }

    /** @brief Claims @p bytes_per_device of scratch on every stream, ahead of the fan-out that
     *      maps state onto it; the old run frees in stream order behind whatever still rides it. */
    status_t reserve(std::size_t bytes_per_device) noexcept {
        if (streams_.empty()) return status_t::uninitialized_k;
        for (stream_t& opened : streams_) {
            opened.arena.clear();
            if (failed(opened.arena.reserve_exactly(bytes_per_device))) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** @brief Waits for every transfer and launch queued on every stream - the fleet's completion. */
    status_t drain() noexcept {
        for (stream_t& opened : streams_)
            if (cudaStreamSynchronize(opened.handle) != cudaSuccess) return status_t::device_error_k;
        return status_t::success_k;
    }

    /** @brief The stream driving @p index, in the order the devices were named. */
    stream_t& stream(std::size_t index) noexcept { return streams_[index]; }
    /** @brief The device @p index drives - what a caller pairs its shards against. */
    int device(std::size_t index) const noexcept { return streams_[index].device; }
    /** @brief Devices this executor drives. */
    std::size_t size() const noexcept { return streams_.size(); }

  private:
    /** @brief One stream per named device, in the caller's order. */
    growing_buffer<stream_t> streams_ {global_allocator};
    /** @brief Held by the one caller driving the streams; never swapped. */
    fu::spin_mutex<fu::preferred_yield_t> held_;
};

/** @brief The @c std::unique_ptr deleter carrying the allocator and the element count. */
template <cuda_allocator allocator_>
struct cuda_deleter {
    allocator_ allocator;
    std::size_t count = 0;

    void operator()(typename allocator_::value_type* pointer) noexcept { allocator.deallocate(pointer, count); }
};

/** @brief Owned device or pinned allocation; host code takes @c .get() for views and transfers. */
template <typename value_type_, cuda_allocator allocator_ = cuda_device_allocator<value_type_>>
using cuda_unique_ptr = std::unique_ptr<value_type_, cuda_deleter<allocator_>>;

/** @brief The @c std::make_unique analog, null on exhaustion; the bytes arrive uninitialized. */
template <cuda_allocator allocator_>
cuda_unique_ptr<typename allocator_::value_type, allocator_> cuda_make_unique(allocator_ allocator,
                                                                              std::size_t count) noexcept {
    typename allocator_::value_type* pointer = allocator.allocate(count);
    return {pointer, cuda_deleter<allocator_> {allocator, count}};
}

static_assert(cuda_reservable_allocator<cuda_virtual_memory_allocator<byte_t>>);
static_assert(cuda_reservable_allocator<cuda_managed_allocator<byte_t>>);
static_assert(cuda_allocator<cuda_device_allocator<byte_t>>);
static_assert(cuda_allocator<cuda_arena_allocator<byte_t>>);
static_assert(cuda_allocator<cuda_pinned_allocator<byte_t>>);

} // namespace usearch
} // namespace unum
