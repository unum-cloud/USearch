/**
 *  @file test/dense/vram_index.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The CUDA-side suite composer for @c dense_vram_index smoke & concurrency tests, shared by
 *      the per-dtype shards.
 *
 *  Adapts the harness's host-side allocation ledger to device memory, so the same
 *  live-bytes-must-balance discipline audits the GPU planes.
 */
#pragma once

#include "usearch/dense/vram_index.cuh"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** Ledgered device allocator: the virtual-memory plane backend with every committed
 *  slab tallied into the harness's @c allocation_ledger_t, and a converting constructor
 *  from the harness's host-side @c counting_allocator - so @c make_ledgered_index
 *  constructs the CUDA index unchanged and the ledger provably balances device bytes. */
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
    counting_cuda_allocator(counting_allocator_t const& host) noexcept : ledger_(host.ledger_) {}
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

/** Composes one (dtype, kernel) cell's suite over the ledgered device allocator. */
template <nk_kernel_kind_t kernel_, nk_dtype_t dtype_ = nk_i8_k>
std::size_t test_cuda_configuration(char const* name) {
    using allocator_t = counting_cuda_allocator_t;
    using index_t =
        dense_vram_index<dtype_, kernel_, cuda_arch_blackwellrtx_k, allocator_t, allocator_t, cuda_device_allocator_t>;
    auto make_index = [](allocation_ledger_t& ledger) {
        return index_t {allocator_t {counting_allocator_t(ledger)}, allocator_t {counting_allocator_t(ledger)}};
    };
    // The suite is composed here, not reflected: every call below is a capability claim
    // this fully-concurrent, serializable category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_guards<index_t>(make_index);
    failures_count += test_dense_three_vectors<index_t, kernel_>(make_index);
    failures_count += test_dense_growth_and_recall<index_t, kernel_>(make_index);
    failures_count += test_dense_save_load_roundtrip<index_t, kernel_>(make_index);
    failures_count += test_dense_load_append<index_t, kernel_>(make_index);
    failures_count += test_dense_concurrent_build<index_t, kernel_>(make_index);
    failures_count += test_dense_concurrent_search<index_t, kernel_>(make_index);
    return report(name, failures_count);
}

#pragma region Suite Entry Points

/** Per-dtype CUDA suite entries the shard translation units define and the suite's
 *  @c main composes explicitly; each returns its shard's failure count. */
std::size_t check_vram_index_bf16();
std::size_t check_vram_index_f16();
std::size_t check_vram_index_e5m2();
std::size_t check_vram_index_e4m3();
std::size_t check_vram_index_i8();
std::size_t check_vram_index_i4();
std::size_t check_vram_index_u8();
std::size_t check_vram_index_u4();
std::size_t check_vram_index_u1();

#pragma endregion Suite Entry Points

} // namespace unum::usearch::test::dense
