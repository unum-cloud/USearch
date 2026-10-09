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
    requires dense_device_resident<index_type_>
std::size_t dense_expect_multiply(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using dot_result_t = typename index_type_::dot_result_t;
    using search_config_t = typename index_type_::search_config_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, left_count = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;
    index_type_ index = make_index(ledger);
    typename index_type_::dense_traversal_context_t context {{}};
    failures += expect(index.reserve({rows_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "multiply"));
    auto const left = vectors(nk::range(std::size_t {0}, left_count), nk::slice);

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

    auto assigned_ids = nk::vector<dense_node_id_t>::zeros(rows_count).value;
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

/** The host overloads - the dot kernel's and every other kernel's empty one - pick alongside the device one below. */
using test::test_dense_multiply;

/** @ref dense_expect_multiply, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
    requires(kernel_ == nk_kernel_dot_k && dense_device_resident<index_type_>)
std::size_t test_dense_multiply(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_multiply<index_type_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** A device lookup over a value table whose rows sit a 16-byte pitch apart: an ID below @c count names its row, a node
 *  ID on one GPU and a key on shards, and any other has none. */
template <typename fragment_type_>
struct dense_cuda_value_lookup {
    byte_t const* rows;
    std::size_t pitch;
    std::size_t count;

    __device__ fragment_type_ const* operator()(u32_t id) const noexcept {
        return id < count ? reinterpret_cast<fragment_type_ const*>(rows + id * pitch) : nullptr;
    }
};

/** A value table copied onto the device, its rows a 16-byte pitch apart; null rows when the copy failed. */
template <typename fragment_type_>
struct dense_cuda_value_table {
    cuda_unique_ptr<byte_t> rows;
    std::size_t pitch;

    /** The device lookup over the first @p count rows. */
    dense_cuda_value_lookup<fragment_type_> lookup(std::size_t count) const noexcept {
        return {rows.get(), pitch, count};
    }
};

/** Copies @p values onto the device as the table every attend lookup in these suites reads. */
template <typename fragment_type_, typename scalar_type_>
dense_cuda_value_table<fragment_type_> dense_cuda_upload_values(nk::matrix_view<scalar_type_> values) {
    std::size_t const rows_count = values.extent(0);
    std::size_t const row_bytes = divide_round_up<std::size_t>(values.extent(1) * nk_dtype_bits(scalar_type_::dtype()),
                                                               8);
    std::size_t const pitch = round_up_to_multiple<std::size_t>(row_bytes, 16);
    dense_cuda_value_table<fragment_type_> table {cuda_make_unique(cuda_device_allocator_t {}, rows_count * pitch),
                                                  pitch};
    if (table.rows && cudaMemcpy2D(table.rows.get(), pitch, values.byte_data(), values.stride_bytes(0), row_bytes,
                                   rows_count, cudaMemcpyHostToDevice) != cudaSuccess)
        table.rows.reset();
    return table;
}

/** Device attention against exact full attention, value rows looked up by node ID in a device table: an exhaustive
 *  walk scores every node, so each head's output and carry match the @c f64 oracle over every key, with one head per
 *  token and with four sharing its walk, while a narrow walk stays within the oracle's bounds; a node without a value
 *  row, mismatched outputs, and a head count that does not divide the rows or whose softmax states and accumulators
 *  no warp seats refuse. */
template <typename index_type_, typename make_index_type_>
    requires dense_device_resident<index_type_>
std::size_t dense_expect_attend(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using lookup_t = dense_cuda_value_lookup<typename index_type_::fragment_t>;
    constexpr std::size_t nodes_count = dense_thread_nodes_k, tokens_count = 4, crowded_heads = 1024;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;
    index_type_ index = make_index(ledger);
    typename index_type_::dense_traversal_context_t context {{}};
    failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto const vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "attend"));
    auto const values = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "attend", 1));
    std::vector<dense_node_id_t> assigned_ids(nodes_count);
    if (failed(index.insert(vectors.view(), {assigned_ids.data(), nodes_count}, context).status))
        return failures + expect(false);
    failures += expect_contiguous_ids(std::span {assigned_ids}, 0);
    auto const table = dense_cuda_upload_values<typename index_type_::fragment_t>(values.view());
    if (!table.rows) return failures + expect(false);

    typename index_type_::search_config_t const exhaustive {nodes_count};
    auto const queries = make_vectors<scalar_t>(tokens_count * 4, dimensions, stream_key(seed, "attend", 2));
    std::vector<f32_t> outputs(crowded_heads * dimensions);
    std::vector<dense_softmax_state_t> carry(crowded_heads);
    auto attend = [&](nk::matrix_view<scalar_t> heads, std::size_t head_count, lookup_t lookup, f32_t score_scale,
                      typename index_type_::search_config_t config) noexcept {
        auto const [attended, status] = index.attend(heads, head_count, lookup, score_scale,
                                                     std::span {outputs}.first(heads.extent(0) * dimensions),
                                                     std::span {carry}.first(heads.extent(0)), context, config);
        failures += expect(failed(status) || attended.count == heads.extent(0));
        return status;
    };
    for (std::size_t const head_count : {std::size_t {1}, std::size_t {4}}) {
        auto const heads = queries(nk::range(std::size_t {0}, tokens_count * head_count), nk::slice);
        f32_t const score_scale = dense_attend_scale(
            dense_oracle_dots(dense_widened_rows(heads, index_type_::vector_dtype_k),
                              dense_widened_rows(vectors.view(), index_type_::vector_dtype_k), dimensions));
        std::span<f32_t const> const attended {outputs.data(), heads.extent(0) * dimensions};
        std::span<dense_softmax_state_t const> const carried {carry.data(), heads.extent(0)};
        failures += expect(attend(heads, head_count, table.lookup(nodes_count), score_scale, exhaustive) ==
                           status_t::success_k);
        failures += dense_expect_full_attention(
            vectors.view(), heads, head_count,
            [&](std::size_t row) noexcept { return values.view().row(row).as_vector(); }, score_scale, attended,
            carried);
        failures += expect(attend(heads, head_count, table.lookup(nodes_count), score_scale, {}) ==
                           status_t::success_k);
        failures += dense_expect_attention_bounds(vectors.view(), values.view(), heads, score_scale, attended, carried);
    }

    // The walk scores every node, so the one past a shortened table refuses.
    failures += expect(attend(queries.view(), 4, table.lookup(nodes_count - 1), 1, exhaustive) ==
                       status_t::shape_mismatch_k);
    failures += expect(attend(queries.view(), 3, table.lookup(nodes_count), 1, exhaustive) ==
                       status_t::shape_mismatch_k);
    failures += expect(index
                           .attend(queries.view(), 4, table.lookup(nodes_count), 1, std::span {outputs}.first(1),
                                   std::span {carry}.first(queries.extent(0)), context, exhaustive)
                           .status == status_t::config_mismatch_k);
    // A thousand heads' softmax states and accumulator rows seat in no warp's shared memory.
    auto const crowded = make_vectors<scalar_t>(crowded_heads, dimensions, stream_key(seed, "attend", 3));
    failures += expect(attend(crowded.view(), crowded_heads, table.lookup(nodes_count), 1, exhaustive) ==
                       status_t::config_mismatch_k);
    return failures;
}

/** @ref dense_expect_attend for the dot kernel, then the ledger it left must balance; no other kernel offers it. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
    requires dense_device_resident<index_type_>
std::size_t test_dense_attend(make_index_type_ const& make_index, u64_t seed) {
    if constexpr (kernel_ != nk_kernel_dot_k) return 0;
    else {
        allocation_ledger_t ledger;
        std::size_t const failures = dense_expect_attend<index_type_>(make_index, ledger, seed);
        return failures + expect_eq(ledger.live_bytes.load(), 0);
    }
}

/** Device search under indexer heads against the @c f64 oracle, every node ID its stored row: a pool as wide as the
 *  index walks every node, so each query lands exactly the top by the score Σₕ wₕ · ReLU(qₕ · k), at 1 − score. */
template <typename index_type_, typename make_index_type_>
    requires dense_device_resident<index_type_>
std::size_t dense_expect_indexer_search(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    constexpr std::size_t nodes_count = dense_thread_nodes_k, queries_count = 4, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;
    index_type_ index = make_index(ledger);
    typename index_type_::dense_traversal_context_t context {{}};
    failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto const vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexer"));
    std::vector<dense_node_id_t> assigned_ids(nodes_count);
    if (failed(index.insert(vectors.view(), {assigned_ids.data(), nodes_count}, context).status))
        return failures + expect(false);
    failures += expect_contiguous_ids(std::span {assigned_ids}, 0);

    auto const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
    dense_indexer_inputs<scalar_t> const inputs = dense_make_indexer_inputs(vectors.view(), queries_count);
    typename index_type_::search_config_t const exhaustive {nodes_count, wanted};
    std::array<dense_node_id_t, queries_count * wanted> node_ids;
    std::array<dense_distance_t, queries_count * wanted> distances;
    failures += expect(
        index
            .search(queries, inputs.heads(), nk::matrix_span<dense_node_id_t> {node_ids.data(), queries_count, wanted},
                    nk::matrix_span<dense_distance_t> {distances.data(), queries_count, wanted}, context, exhaustive)
            .status == status_t::success_k);
    for (std::size_t query = 0; query != queries_count; ++query)
        failures += dense_expect_indexer_matches(
            inputs, query, std::span<dense_node_id_t const> {node_ids.data() + query * wanted, wanted},
            std::span<dense_distance_t const> {distances.data() + query * wanted, wanted});
    return failures;
}

/** @ref dense_expect_indexer_search for the dot kernel over floating-point keys, then the ledger must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
    requires dense_device_resident<index_type_>
std::size_t test_dense_indexer_search(make_index_type_ const& make_index, u64_t seed) {
    if constexpr (kernel_ != nk_kernel_dot_k || nk::is_integral_dtype<typename index_type_::scalar_t>()) return 0;
    else {
        allocation_ledger_t ledger;
        std::size_t const failures = dense_expect_indexer_search<index_type_>(make_index, ledger, seed);
        return failures + expect_eq(ledger.live_bytes.load(), 0);
    }
}

#pragma endregion Folds

/** Composes one (dtype, kernel) cell's suite over the ledgered device allocator. */
template <nk_kernel_kind_t kernel_, nk_dtype_t dtype_ = nk_i8_k>
std::size_t test_cuda_configuration(char const* name, u64_t seed) {
    using allocator_t = counting_cuda_allocator_t;
    using index_t =
        dense_vram_index<dtype_, kernel_, cuda_arch_blackwellrtx_k, allocator_t, allocator_t, cuda_device_allocator_t>;
    auto make_index = [](allocation_ledger_t& ledger) {
        return index_t {allocator_t {counting_allocator_t(ledger)}, allocator_t {counting_allocator_t(ledger)}};
    };
    // The suite is composed here, not reflected: every call below is a capability claim
    // this fully-concurrent, serializable category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_guards<index_t>(make_index, seed);
    failures_count += test_dense_three_vectors<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_growth_and_recall<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_save_load_roundtrip<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_load_append<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_concurrent_build<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_concurrent_search<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_refine_settles<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_health_audits<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_heal_islands<index_t>(make_index, seed);
    failures_count += test_dense_multiply<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_attend<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_indexer_search<index_t, kernel_>(make_index, seed);
    return report(name, failures_count);
}

#pragma region Suite Entry Points

/** Per-dtype CUDA suite entries the shard translation units define and the suite's
 *  @c main composes explicitly; each returns its shard's failure count. */
std::size_t check_vram_index_bf16(settings_t const& settings);
std::size_t check_vram_index_f16(settings_t const& settings);
std::size_t check_vram_index_e5m2(settings_t const& settings);
std::size_t check_vram_index_e4m3(settings_t const& settings);
std::size_t check_vram_index_e3m2(settings_t const& settings);
std::size_t check_vram_index_e2m3(settings_t const& settings);
std::size_t check_vram_index_i8(settings_t const& settings);
std::size_t check_vram_index_i4(settings_t const& settings);
std::size_t check_vram_index_u8(settings_t const& settings);
std::size_t check_vram_index_u4(settings_t const& settings);
std::size_t check_vram_index_u1(settings_t const& settings);

#pragma endregion Suite Entry Points

} // namespace unum::usearch::test::dense
