/**
 *  @file test/dense/vram_apple.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The Apple index the residency-level VRAM scenarios run over: its ledgered plane backend
 *      and the cell the shared composers build it through.
 */
#pragma once

#include "usearch/dense/vram_apple_indexes.hpp"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** Ledgered Metal plane backend: shared buffers on the device under test, each one tallied into
 *  the harness's @c allocation_ledger_t while it lives, so the ledger provably balances the Apple
 *  planes. */
struct counting_metal_allocator_t {
    counting_metal_allocator_t() noexcept = default;
    counting_metal_allocator_t(counting_allocator_t const& host) noexcept : ledger_(host.ledger_) {}

    metal_buffer_t allocate(std::size_t bytes) noexcept {
        metal_buffer_t buffer = shared_.allocate(bytes);
        if (ledger_ && buffer) {
            ledger_->live_bytes.fetch_add(static_cast<std::ptrdiff_t>(buffer.bytes));
            ledger_->total_allocations.fetch_add(1);
        }
        return buffer;
    }
    void deallocate(metal_buffer_t& buffer) noexcept {
        if (ledger_ && buffer) ledger_->live_bytes.fetch_sub(static_cast<std::ptrdiff_t>(buffer.bytes));
        shared_.deallocate(buffer);
    }
    int device() const noexcept { return shared_.device(); }

  private:
    metal_shared_allocator_t shared_ {selected_device};
    allocation_ledger_t* ledger_ = nullptr;
};

/** The Apple index of one (dtype, kernel) cell over the ledgered planes, its one-shard collection,
 *  and the factory the shared composers build the index with. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
struct vram_cell<gpu_vendor_t::apple_k, dtype_, kernel_> {
    using index_t = dense_vram_apple_index<dtype_, nk_cap_metal_k, counting_metal_allocator_t>;
    using indexes_t = dense_vram_apple_indexes<u32_t, dtype_, nk_cap_metal_k>;
    inline static constexpr char const* vendor_prefix_k = "metal";

    static index_t make_index(allocation_ledger_t& ledger) noexcept {
        return index_t {kernel_, counting_metal_allocator_t {counting_allocator_t(ledger)}};
    }
};

} // namespace unum::usearch::test::dense
