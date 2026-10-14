/**
 *  @file test/dense/vram_apple.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The Apple index the residency-level VRAM scenarios run over: its ledgered flat-matrix
 *      backend and the cell the shared composers build it through.
 */
#pragma once

#include "usearch/dense/vram_apple_indexes.hpp"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** Ledgered Metal flat-matrix backend: shared buffers on the device under test, each one tallied
 *  into the harness's @c allocation_ledger_t while it lives, so the ledger provably balances the
 *  Apple flat matrices. */
struct counting_metal_allocator_t {
    counting_metal_allocator_t() noexcept = default;
    counting_metal_allocator_t(allocation_ledger_t& ledger, int device) noexcept : shared_(device), ledger_(&ledger) {}

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
    metal_shared_allocator_t shared_;
    allocation_ledger_t* ledger_ = nullptr;
};

/** The Apple index of one (dtype, kernel) cell over the ledgered flat matrices and its one-shard
 *  collection, built over the stream of the executor the Metal entry opened. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
struct vram_cell<gpu_vendor_t::apple_k, dtype_, kernel_> {
    using index_t = vram_apple_index<dtype_, nk_cap_metal_k, counting_metal_allocator_t, counting_metal_allocator_t>;
    using indexes_t = vram_apple_indexes<u32_t, dtype_, nk_cap_metal_k>;
    inline static constexpr char const* vendor_prefix_k = "metal";

    /** The stream the cell's contexts ride, on the device its indexes sit on. */
    metal_executor_t::stream_t const& stream;

    /** Builds the cell's index over @p ledger, on the stream's device. */
    std::unique_ptr<index_t> operator()(allocation_ledger_t& ledger) const noexcept {
        counting_metal_allocator_t const allocator {ledger, stream.handle.device};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(kernel_, allocator, allocator));
    }

    /** A fresh traversal context submitting to the stream. */
    typename index_t::traversal_context_t context() const noexcept { return {stream.handle.share()}; }
};

} // namespace unum::usearch::test::dense
