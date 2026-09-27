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

#pragma region Folds

/** Device graph products against the native dots: every scored pair is counted, a narrow row keeps its width
 *  while counting past it, a short row ends in sentinels, every kept pair names a distinct stored row with its
 *  exact dot, and mismatched spans refuse. */
template <typename index_type_, typename make_index_type_>
    requires dense_device_resident_k<index_type_>
std::size_t dense_expect_multiply(make_index_type_ const& make_index, allocation_ledger_t& ledger) {
    using scalar_t = typename index_type_::scalar_t;
    using dot_result_t = typename index_type_::dot_result_t;
    using search_config_t = typename index_type_::search_config_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, left_count = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;
    index_type_ index = make_index(ledger);
    typename index_type_::dense_traversal_context_t context {{}};
    failures += expect(index.reserve({rows_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto vectors = make_vectors<scalar_t>(rows_count, dimensions, dense_seed_multiply_k);
    auto const left = vectors.view()(nk::range {std::size_t {0}, left_count}, nk::slice);

    std::array<dense_node_id_t, left_count * rows_count> node_ids;
    std::array<dot_result_t, left_count * rows_count> products;
    std::array<u32_t, left_count> counts;
    auto multiply = [&](std::size_t wanted, std::size_t products_width, search_config_t config) noexcept {
        return index.multiply(left, nk::matrix_span<dense_node_id_t> {node_ids.data(), left_count, wanted},
                              nk::matrix_span<dot_result_t> {products.data(), left_count, products_width}, counts,
                              context, config);
    };

    // An empty index scores nothing, and every row it leaves short ends in sentinels.
    node_ids.fill(0);
    auto const empty = multiply(4, 4, {});
    failures += expect(empty.status == status_t::success_k && empty.value.count == 0);
    failures += expect(std::all_of(node_ids.begin(), node_ids.begin() + left_count * 4,
                                   [](dense_node_id_t id) { return id == dense_node_id_unknown_k; }));

    auto assigned_ids = nk::vector<dense_node_id_t>::try_zeros(rows_count);
    failures += expect(index.insert(vectors.view(), assigned_ids.span(), context).status == status_t::success_k);

    search_config_t const exhaustive {static_cast<typename index_type_::small_size_t>(rows_count)};
    for (std::size_t const wanted : {rows_count, std::size_t {4}}) {
        auto const [multiplied, status] = multiply(wanted, wanted, exhaustive);
        failures += expect(status == status_t::success_k);
        failures += expect_eq(multiplied.count, multiplied.computed_distances);
        std::size_t scored_count = 0;
        for (std::size_t row = 0; row != left_count; ++row) {
            scored_count += counts[row];
            failures += expect(counts[row] > 0 && counts[row] <= rows_count);
            if (wanted < rows_count) failures += expect(counts[row] > wanted);
            failures += dense_expect_product_row<rows_count>(
                left.row(row).as_vector(), vectors.view(), node_ids.data() + row * wanted,
                products.data() + row * wanted, min_of<std::size_t>(counts[row], wanted),
                [](dense_node_id_t id) noexcept { return std::size_t {id}; });
        }
        failures += expect_eq(multiplied.count, scored_count);
    }

    failures += expect(multiply(4, 3, {}).status == status_t::config_mismatch_k);
    return failures;
}

/** @ref dense_expect_multiply, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
    requires(kernel_ == nk_kernel_dot_k && dense_device_resident_k<index_type_>)
std::size_t test_dense_multiply(make_index_type_ const& make_index) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_multiply<index_type_>(make_index, ledger);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

#pragma endregion Folds

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
    failures_count += test_dense_multiply<index_t, kernel_>(make_index);
    return report(name, failures_count);
}

#pragma region Suite Entry Points

/** Per-dtype CUDA suite entries the shard translation units define and the suite's
 *  @c main composes explicitly; each returns its shard's failure count. */
std::size_t check_vram_index_bf16();
std::size_t check_vram_index_f16();
std::size_t check_vram_index_e5m2();
std::size_t check_vram_index_e4m3();
std::size_t check_vram_index_e3m2();
std::size_t check_vram_index_e2m3();
std::size_t check_vram_index_i8();
std::size_t check_vram_index_i4();
std::size_t check_vram_index_u8();
std::size_t check_vram_index_u4();
std::size_t check_vram_index_u1();

#pragma endregion Suite Entry Points

} // namespace unum::usearch::test::dense
