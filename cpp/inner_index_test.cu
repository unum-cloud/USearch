/**
 *  @brief Smoke & concurrency tests for the GPU-resident inner index - the CUDA-side
 *      consumer of the shared scenario library.
 *
 *  @author Ash Vardanian
 *  @file cpp/inner_index_test.cu
 *  @date July 22, 2026
 *
 *  The scenarios live in @c inner_index_test.cuh, templated over the index type; this
 *  translation unit names the CUDA index cells, composes their suite explicitly, and adapts
 *  the harness's host-side allocation ledger to device memory, so the same
 *  live-bytes-must-balance discipline audits the GPU planes.
 */
#include <cuda_runtime.h>

#include <fmt/format.h>

#include "usearch/inner_cuda_index.cuh"

#include "inner_index_test.cuh"

using namespace unum::usearch;

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

template <nk_kernel_kind_t kernel_, nk_dtype_t dtype_ = nk_i8_k>
static std::size_t test_configuration(char const* name) {
    using allocator_t = counting_cuda_allocator<std::byte>;
    using index_t = inner_cuda_index<dtype_, kernel_, allocator_t, allocator_t>;
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

int main() {
    int devices_count = 0;
    cudaError_t const probed = cudaGetDeviceCount(&devices_count);
    if (probed != cudaSuccess || !devices_count) {
        fmt::print("skipped  {:<20} : no CUDA device reachable\n", "cuda");
        return 0;
    }
    fmt::print("CUDA devices: {}\n", devices_count);
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_kernel_sqeuclidean_k>("cuda-i8+sqeuclidean");
    failures_count += test_configuration<nk_kernel_euclidean_k>("cuda-i8+euclidean");
    failures_count += test_configuration<nk_kernel_angular_k>("cuda-i8+angular");
    failures_count += test_configuration<nk_kernel_euclidean_k, nk_u8_k>("cuda-u8+euclidean");
    failures_count += test_configuration<nk_kernel_angular_k, nk_f16_k>("cuda-f16+angular");
    failures_count += test_configuration<nk_kernel_euclidean_k, nk_bf16_k>("cuda-bf16+euclidean");
    failures_count += test_configuration<nk_kernel_angular_k, nk_i4_k>("cuda-i4+angular");
    failures_count += test_configuration<nk_kernel_hamming_k, nk_u1_k>("cuda-u1+hamming");
    failures_count += test_configuration<nk_kernel_jaccard_k, nk_u1_k>("cuda-u1+jaccard");
    return failures_count ? 1 : 0;
}
