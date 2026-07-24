/**
 *  @brief The CUDA-side suite composer for @c dense_vram_index smoke & concurrency tests,
 *      shared by the per-dtype shards - adapts the harness's host-side allocation ledger
 *      to device memory, so the same live-bytes-must-balance discipline audits the GPU
 *      planes.
 *  @author Ash Vardanian
 *  @file test/dense_index_cuda.cuh
 *  @date July 22, 2026
 */
#pragma once

#include "usearch/dense/vram_index.cuh"

#include "dense_index.cuh"

namespace unum {
namespace usearch {

/**
 *  @brief Ledgered device allocator: the virtual-memory plane backend with every committed
 *      slab tallied into the harness's @c allocation_ledger_t, and a converting constructor
 *      from the harness's host-side @c counting_allocator - so @c make_ledgered_index
 *      constructs the CUDA index unchanged and the ledger provably balances device bytes.
 */
template <typename value_type_ = std::byte>
struct counting_cuda_allocator {
    using value_type = value_type_;
    inline static constexpr cuda_address_space_t address_space_k = cuda_address_space_t::device_k;
    template <typename other_>
    struct rebind {
        using other = counting_cuda_allocator<other_>;
    };

    cuda_vmm_allocator<value_type_> vmm_;
    allocation_ledger_t* ledger_ = nullptr;
    std::size_t tallied_bytes_ = 0;

    counting_cuda_allocator() noexcept = default;
    counting_cuda_allocator(counting_allocator<std::byte> const& host) noexcept : ledger_(host.ledger_) {}
    counting_cuda_allocator(counting_cuda_allocator const& other) noexcept : vmm_(other.vmm_), ledger_(other.ledger_) {}
    template <typename other_>
    counting_cuda_allocator(counting_cuda_allocator<other_> const& other) noexcept
        : vmm_(other.vmm_), ledger_(other.ledger_) {}
    ~counting_cuda_allocator() noexcept { release(); }

    status_t reserve_range(std::size_t bytes, int device) noexcept { return vmm_.reserve_range(bytes, device); }
    status_t commit(std::size_t bytes) noexcept {
        status_t const status = vmm_.commit(bytes);
        if (ledger_ && vmm_.committed_bytes() > tallied_bytes_) {
            ledger_->live_bytes.fetch_add(static_cast<std::ptrdiff_t>(vmm_.committed_bytes() - tallied_bytes_));
            ledger_->total_allocations.fetch_add(1);
            tallied_bytes_ = vmm_.committed_bytes();
        }
        return status;
    }
    value_type* base() const noexcept { return vmm_.base(); }
    std::size_t reserved_bytes() const noexcept { return vmm_.reserved_bytes(); }
    std::size_t committed_bytes() const noexcept { return vmm_.committed_bytes(); }
    int device() const noexcept { return vmm_.device(); }
    void release() noexcept {
        if (ledger_ && tallied_bytes_) ledger_->live_bytes.fetch_sub(static_cast<std::ptrdiff_t>(tallied_bytes_));
        tallied_bytes_ = 0;
        vmm_.release();
    }
};

static_assert(cuda_reservable_allocator<counting_cuda_allocator<std::byte>>);

/** @brief Composes one (dtype, kernel) cell's suite over the ledgered device allocator. */
template <nk_kernel_kind_t kernel_, nk_dtype_t dtype_ = nk_i8_k>
std::size_t test_cuda_configuration(char const* name) {
    using allocator_t = counting_cuda_allocator<std::byte>;
    using index_t = dense_vram_index<dtype_, kernel_, cuda_arch_blackwell_k, allocator_t, allocator_t>;
    auto make_index = [](allocation_ledger_t& ledger) {
        return index_t {allocator_t {counting_allocator<std::byte>(ledger)},
                        allocator_t {counting_allocator<std::byte>(ledger)}};
    };
    // The suite is composed here, not reflected: every call below is a capability claim
    // this fully-concurrent, serializable category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_guards<index_t>(make_index);
    failures_count += test_three_vectors<index_t, kernel_>(make_index);
    failures_count += test_growth_and_recall<index_t, kernel_>(make_index);
    failures_count += test_save_load_roundtrip<index_t, kernel_>(make_index);
    failures_count += test_concurrent_build<index_t, kernel_>(make_index);
    failures_count += test_concurrent_search<index_t, kernel_>(make_index);
    return report_scenarios(name, failures_count);
}

} // namespace usearch
} // namespace unum

#pragma region Suite Entry Points

/** @brief Per-dtype CUDA suite entries the shard translation units define and the suite's
 *      @c main composes explicitly; each returns its shard's failure count. */
std::size_t test_dense_index_i8_cuda();
std::size_t test_dense_index_u8_cuda();
std::size_t test_dense_index_f16_cuda();
std::size_t test_dense_index_bf16_cuda();
std::size_t test_dense_index_i4_cuda();
std::size_t test_dense_index_u1_cuda();

#pragma endregion Suite Entry Points
