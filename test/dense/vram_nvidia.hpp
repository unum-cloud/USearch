/**
 *  @file test/dense/vram_nvidia.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The NVIDIA index the residency-level VRAM scenarios run over: its ledgered plane backend,
 *      the device-bound scratch allocator and the cell the shared composers build it through.
 *
 *  Plain C++ over the NVIDIA faces, so the shared shards compile with the host compiler; the
 *  cells' kernels instantiate in @c vram_nvidia.cu, which also covers device attention, since its
 *  value lookups run on the device.
 */
#pragma once

#include "usearch/dense/vram_nvidia_indexes.hpp"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** The stream-ordered device allocator bound to the device under test, so the contexts the
 *  scenarios build by default walk the device the index sits on. */
template <typename value_type_ = byte_t>
struct selected_cuda_allocator : cuda_device_allocator<value_type_> {
    template <typename other_>
    struct rebind {
        using other = selected_cuda_allocator<other_>;
    };

    selected_cuda_allocator() noexcept : cuda_device_allocator<value_type_>(nullptr, selected_device) {}
    template <typename other_>
    selected_cuda_allocator(selected_cuda_allocator<other_> const& other) noexcept
        : cuda_device_allocator<value_type_>(other.stream(), other.device()) {}
};

/** Ledgered device allocator: the virtual-memory plane backend with every committed slab tallied
 *  into the harness's @c allocation_ledger_t, and a converting constructor from the harness's
 *  host-side @c counting_allocator - so @c make_ledgered_index constructs the CUDA index unchanged
 *  and the ledger provably balances device bytes. */
template <typename value_type_ = std::byte>
struct counting_cuda_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = counting_cuda_allocator<other_>;
    };

    cuda_virtual_memory_allocator<value_type_> virtual_memory_;
    allocation_ledger_t* ledger_ = nullptr;
    std::size_t tallied_bytes_ = 0;

    counting_cuda_allocator() noexcept = default;
    counting_cuda_allocator(counting_allocator_t const& host) noexcept
        : virtual_memory_(selected_device), ledger_(host.ledger_) {}
    counting_cuda_allocator(counting_cuda_allocator const& other) noexcept
        : virtual_memory_(other.virtual_memory_), ledger_(other.ledger_) {}
    template <typename other_>
    counting_cuda_allocator(counting_cuda_allocator<other_> const& other) noexcept
        : virtual_memory_(other.virtual_memory_), ledger_(other.ledger_) {}
    ~counting_cuda_allocator() noexcept { release(); }

    status_t reserve_range(std::size_t bytes, int device) noexcept {
        return virtual_memory_.reserve_range(bytes, device);
    }
    status_t commit(std::size_t bytes) noexcept {
        status_t const status = virtual_memory_.commit(bytes);
        if (ledger_ && virtual_memory_.committed_bytes() > tallied_bytes_) {
            ledger_->live_bytes.fetch_add(
                static_cast<std::ptrdiff_t>(virtual_memory_.committed_bytes() - tallied_bytes_));
            ledger_->total_allocations.fetch_add(1);
            tallied_bytes_ = virtual_memory_.committed_bytes();
        }
        return status;
    }
    value_type* base() const noexcept { return virtual_memory_.base(); }
    std::size_t reserved_bytes() const noexcept { return virtual_memory_.reserved_bytes(); }
    std::size_t committed_bytes() const noexcept { return virtual_memory_.committed_bytes(); }
    int device() const noexcept { return virtual_memory_.device(); }
    void release() noexcept {
        if (ledger_ && tallied_bytes_) ledger_->live_bytes.fetch_sub(static_cast<std::ptrdiff_t>(tallied_bytes_));
        tallied_bytes_ = 0;
        virtual_memory_.release();
    }
};

/** The byte specialization every CUDA fixture places its index through. */
using counting_cuda_allocator_t = counting_cuda_allocator<byte_t>;

static_assert(cuda_reservable_allocator<counting_cuda_allocator_t>);

/** A device lookup over a value table whose rows sit a 16-byte pitch apart: an ID below @c count
 *  names its row, a node ID on one GPU and a key on shards, and any other has none. */
template <typename fragment_type_>
struct dense_cuda_value_lookup {

    /** Names the lookup to the attend kernels the suite's fatbin carries. */
    inline static constexpr u32_t device_lookup_k = 1;

    byte_t const* rows;
    std::size_t pitch;
    std::size_t count;

#if defined(__CUDACC__)
    __device__ fragment_type_ const* operator()(u32_t id) const noexcept {
        return id < count ? reinterpret_cast<fragment_type_ const*>(rows + id * pitch) : nullptr;
    }
#endif
};

/** The NVIDIA index of one (dtype, kernel) cell over the ledgered planes, its sharded collection,
 *  and the factory the shared composers build the index with; device attention runs in
 *  @c vram_nvidia.cu instead of the shared composers. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
struct vram_cell<gpu_vendor_t::nvidia_k, dtype_, kernel_> {
    using allocator_t = counting_cuda_allocator_t;
    using index_t =
        vram_nvidia_index<dtype_, dense_nvidia_capabilities_k, allocator_t, allocator_t, selected_cuda_allocator<>>;
    using indexes_t = vram_nvidia_indexes<u32_t, dtype_, dense_nvidia_capabilities_k>;
    inline static constexpr char const* vendor_prefix_k = "cuda";
    inline static constexpr bool attends_k = true;

    static index_t make_index(allocation_ledger_t& ledger) noexcept {
        return index_t {kernel_, allocator_t {counting_allocator_t(ledger)},
                        allocator_t {counting_allocator_t(ledger)}};
    }
};

} // namespace unum::usearch::test::dense
