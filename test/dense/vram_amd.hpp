/**
 *  @file test/dense/vram_amd.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD index the residency-level VRAM scenarios run over: its ledgered flat-matrix
 *      backend, the device-bound scratch allocator and the index factory.
 *
 *  The shared composers include it for the AMD suite, so the scenarios stay vendor-neutral while
 *  the harness's live-bytes-must-balance ledger audits the AMD flat matrices.
 */
#pragma once

#include "usearch/dense/vram_amd_indexes.hpp"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** The stream-ordered device allocator bound to the device under test, so the contexts the
 *  scenarios build by default walk the device the index sits on. */
template <typename value_type_ = byte_t>
struct selected_hip_allocator : hip_device_allocator<value_type_> {
    template <typename other_>
    struct rebind {
        using other = selected_hip_allocator<other_>;
    };

    selected_hip_allocator() noexcept : hip_device_allocator<value_type_>(nullptr, selected_device) {}
    template <typename other_>
    selected_hip_allocator(selected_hip_allocator<other_> const& other) noexcept
        : hip_device_allocator<value_type_>(other.stream(), other.device()) {}
};

/** Ledgered AMD flat-matrix backend: the virtual-memory one with every committed slab tallied into
 *  the harness's @c allocation_ledger_t, and a converting constructor from its host-side
 *  @c counting_allocator, so the ledger provably balances device bytes. */
template <typename value_type_ = std::byte>
struct counting_hip_allocator {
    using value_type = value_type_;
    inline static constexpr hip_address_space_t address_space_k = hip_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = counting_hip_allocator<other_>;
    };

    hip_virtual_memory_allocator<value_type_> virtual_memory_;
    allocation_ledger_t* ledger_ = nullptr;
    std::size_t tallied_bytes_ = 0;

    counting_hip_allocator() noexcept = default;
    counting_hip_allocator(counting_allocator_t const& host) noexcept
        : virtual_memory_(selected_device), ledger_(host.ledger_) {}
    counting_hip_allocator(counting_hip_allocator const& other) noexcept
        : virtual_memory_(other.virtual_memory_), ledger_(other.ledger_) {}
    template <typename other_>
    counting_hip_allocator(counting_hip_allocator<other_> const& other) noexcept
        : virtual_memory_(other.virtual_memory_), ledger_(other.ledger_) {}
    ~counting_hip_allocator() noexcept { release(); }

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

/** The byte specialization every AMD fixture places its index through. */
using counting_hip_allocator_t = counting_hip_allocator<byte_t>;

static_assert(hip_reservable_allocator<counting_hip_allocator_t>);

/** The AMD class the suite links, whose code object its device must load: CDNA3, the floor. */
inline constexpr nk_capability_t dense_amd_capabilities_k = amd_capabilities_of(gpu_class_t::amd_cdna3_k);

/** The AMD index of one (dtype, kernel) cell over the ledgered flat matrices, its sharded
 *  collection, and the factory the shared composers build the index with. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
struct vram_cell<gpu_vendor_t::amd_k, dtype_, kernel_> {
    using index_t = vram_amd_index<dtype_, dense_amd_capabilities_k, counting_hip_allocator_t, counting_hip_allocator_t,
                                   selected_hip_allocator<>>;
    using indexes_t = vram_amd_indexes<u32_t, dtype_, dense_amd_capabilities_k>;
    inline static constexpr char const* vendor_prefix_k = "rocm";

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) noexcept {
        return std::unique_ptr<index_t>(new (std::nothrow)
                                            index_t(kernel_, counting_hip_allocator_t {counting_allocator_t(ledger)},
                                                    counting_hip_allocator_t {counting_allocator_t(ledger)}));
    }
};

} // namespace unum::usearch::test::dense
