/**
 *  @file test/dense/vram_nvidia.cu
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The NVIDIA suite's entry for the one VRAM runner: selects the device, runs every dtype
 *      cell and the sharded suite the settings select, then device attention, whose value lookups
 *      run on the device.
 *
 *  Compiled twice, like the library's class units: by NVCC device-only into the suite's own fatbin,
 *  carrying every cell's kernels and the attends over this file's value lookups, and by Clang
 *  host-only into the cells' index types, which the shared shards see only through the faces.
 */
#include <fmt/format.h>

#include "usearch/dense/vram_nvidia_index.cuh"

#include "vram_index.cuh"
#include "vram_indexes.cuh"

namespace unum::usearch::dense {

using test::dense_nvidia_capabilities_k;
using test::dense::counting_cuda_allocator_t;
using test::dense::selected_cuda_allocator;

/** The lookups the suite attends through: by node ID on one GPU, and by key on shards. */
using cuda_test_lookup_t = test::dense::dense_cuda_value_lookup<uint4>;
using cuda_test_keyed_lookup_t = cuda_keyed_lookup<u32_t, cuda_test_lookup_t>;

template <>
struct cuda_value_lookup_of<cuda_test_lookup_t::device_lookup_k> {
    using type = cuda_test_lookup_t;
};
template <>
struct cuda_value_lookup_of<cuda_test_keyed_lookup_t::device_lookup_k> {
    using type = cuda_test_keyed_lookup_t;
};

/** The suite's fatbin, linked in as raw bytes beside its host half. */
extern "C" byte_t const test_vram_nvidia_bundle_start[], test_vram_nvidia_bundle_end[];

template <>
bytes_view_t cuda_fatbin<dense_nvidia_capabilities_k>() noexcept {
    return {test_vram_nvidia_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_bundle_end - test_vram_nvidia_bundle_start)};
}

// Every cell's kernels and attends, and the keyed attends of the sharded suite's collection.
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_bf16_k, cuda_test_lookup_t::device_lookup_k,
                                          cuda_test_keyed_lookup_t::device_lookup_k>();
#if USEARCH_DENSE_DTYPES >= 2
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_u1_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 3
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_i8_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 4
template __device__ void
cuda_instantiate<dense_nvidia_capabilities_k, nk_e4m3_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 5
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_f16_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 6
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_u8_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 7
template __device__ void
cuda_instantiate<dense_nvidia_capabilities_k, nk_e5m2_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 8
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_i4_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 9
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_u4_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 10
template __device__ void
cuda_instantiate<dense_nvidia_capabilities_k, nk_e3m2_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 11
template __device__ void
cuda_instantiate<dense_nvidia_capabilities_k, nk_e2m3_k, cuda_test_lookup_t::device_lookup_k>();
#endif

// The ledgered index of every NVIDIA cell and the collections the sharded suite drives.
template struct vram_nvidia_index<nk_bf16_k, dense_nvidia_capabilities_k, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, selected_cuda_allocator<>>;
template struct vram_nvidia_indexes<u32_t, nk_bf16_k, dense_nvidia_capabilities_k>;
#if USEARCH_DENSE_DTYPES >= 2
template struct vram_nvidia_index<nk_u1_k, dense_nvidia_capabilities_k, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, selected_cuda_allocator<>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct vram_nvidia_index<nk_i8_k, dense_nvidia_capabilities_k, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, selected_cuda_allocator<>>;
template struct vram_nvidia_indexes<u32_t, nk_i8_k, dense_nvidia_capabilities_k>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct vram_nvidia_index<nk_e4m3_k, dense_nvidia_capabilities_k, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, selected_cuda_allocator<>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct vram_nvidia_index<nk_f16_k, dense_nvidia_capabilities_k, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, selected_cuda_allocator<>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct vram_nvidia_index<nk_u8_k, dense_nvidia_capabilities_k, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, selected_cuda_allocator<>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct vram_nvidia_index<nk_e5m2_k, dense_nvidia_capabilities_k, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, selected_cuda_allocator<>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct vram_nvidia_index<nk_i4_k, dense_nvidia_capabilities_k, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, selected_cuda_allocator<>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct vram_nvidia_index<nk_u4_k, dense_nvidia_capabilities_k, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, selected_cuda_allocator<>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct vram_nvidia_index<nk_e3m2_k, dense_nvidia_capabilities_k, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, selected_cuda_allocator<>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct vram_nvidia_index<nk_e2m3_k, dense_nvidia_capabilities_k, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, selected_cuda_allocator<>>;
#endif

} // namespace unum::usearch::dense

namespace unum::usearch::test::dense {

/** A value table copied onto the device, its rows a 16-byte pitch apart; null rows when the copy
 *  failed. */
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
    if (table.rows && (cuda_copy_2d_async(table.rows.get(), pitch, values.byte_data(), values.stride_bytes(0),
                                          row_bytes, rows_count, nullptr) != cuda_success_k ||
                       cuda_api().stream_synchronize(nullptr) != cuda_success_k))
        table.rows.reset();
    return table;
}

/** Device attention against exact full attention, value rows looked up by node ID in a device
 *  table: an exhaustive walk scores every node, so each head's output and carry match the @c f64
 *  oracle over every key, with one head per token and with four sharing its walk, while a narrow
 *  walk stays within the oracle's bounds; a node without a value row, mismatched outputs, and a
 *  head count that does not divide the rows or whose softmax states and accumulators no warp seats
 *  refuse. */
template <typename index_type_, typename make_index_type_>
std::size_t dense_expect_vram_attend(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using lookup_t = dense_cuda_value_lookup<typename index_type_::fragment_t>;
    constexpr std::size_t nodes_count = dense_thread_nodes_k, tokens_count = 4, crowded_heads = 1024;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto const vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "attend"));
    auto const values = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "attend", 1));
    std::vector<node_id_t> assigned_ids(nodes_count);
    if (failed(index.insert(vectors.view(), {assigned_ids.data(), nodes_count}, context).status))
        return failures + expect(false);
    failures += expect_contiguous_ids(std::span {assigned_ids}, 0);
    auto const table = dense_cuda_upload_values<typename index_type_::fragment_t>(values.view());
    if (!table.rows) return failures + expect(false);

    typename index_type_::search_config_t const exhaustive {nodes_count};
    auto const queries = make_vectors<scalar_t>(tokens_count * 4, dimensions, stream_key(seed, "attend", 2));
    auto outputs = nk::matrix<f32_t>::zeros({crowded_heads, dimensions}).value;
    std::vector<softmax_state_t> carry(crowded_heads);
    auto attend = [&](nk::matrix_view<scalar_t> heads, std::size_t head_count, lookup_t lookup, f32_t score_scale,
                      typename index_type_::search_config_t config) noexcept {
        auto const [attended, status] = index.attend(heads, head_count, lookup, score_scale,
                                                     outputs(nk::range(std::size_t {0}, heads.extent(0)), nk::slice),
                                                     std::span {carry}.first(heads.extent(0)), context, config);
        failures += expect(failed(status) || attended.count == heads.extent(0));
        return status;
    };
    for (std::size_t const head_count : {std::size_t {1}, std::size_t {4}}) {
        auto const heads = queries(nk::range(std::size_t {0}, tokens_count * head_count), nk::slice);
        f32_t const score_scale = dense_attend_scale(
            dense_oracle_dots(dense_widened_rows(heads, index_type_::vector_dtype_k),
                              dense_widened_rows(vectors.view(), index_type_::vector_dtype_k), dimensions));
        nk::matrix_view<f32_t> const attended = outputs(nk::range(std::size_t {0}, heads.extent(0)), nk::slice);
        std::span<softmax_state_t const> const carried {carry.data(), heads.extent(0)};
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
                           .attend(queries.view(), 4, table.lookup(nodes_count), 1,
                                   outputs(nk::range(std::size_t {0}, std::size_t {1}), nk::slice),
                                   std::span {carry}.first(queries.extent(0)), context, exhaustive)
                           .status == status_t::config_mismatch_k);
    // A thousand heads' softmax states and accumulator rows seat in no warp's shared memory.
    auto const crowded = make_vectors<scalar_t>(crowded_heads, dimensions, stream_key(seed, "attend", 3));
    failures += expect(attend(crowded.view(), crowded_heads, table.lookup(nodes_count), 1, exhaustive) ==
                       status_t::config_mismatch_k);
    return failures;
}

/** The sharded attention, keys equal to their rows and value rows looked up by key in a device
 *  table: every shard walks every node, so each head's merged output and carry match exact full
 *  attention, with one head per token and with four sharing its walk, while a narrow walk stays
 *  within the oracle's bounds; a key without a value row, mismatched outputs, and a head count that
 *  does not divide the rows refuse. */
template <typename indexes_type_>
std::size_t dense_expect_vram_indexes_attend(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using lookup_t = dense_cuda_value_lookup<typename index_t::fragment_t>;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t nodes_count = dense_thread_nodes_k, tokens_count = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {nk_kernel_dot_k, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(driver.place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions}).status));
    auto const vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexes_attend"));
    auto const values = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexes_attend", 1));
    std::vector<key_t> keys(nodes_count);
    for (std::size_t row = 0; row != nodes_count; ++row) keys[row] = static_cast<key_t>(row);
    failures += expect(
        succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));
    auto const table = dense_cuda_upload_values<typename index_t::fragment_t>(values.view());
    if (!table.rows) return failures + expect(false);

    typename indexes_type_::search_config_t const exhaustive {nodes_count};
    auto const queries = make_vectors<scalar_t>(tokens_count * 4, dimensions, stream_key(seed, "indexes_attend", 2));
    auto outputs = nk::matrix<f32_t>::zeros({queries.extent(0), dimensions}).value;
    std::vector<softmax_state_t> carry(queries.extent(0));
    auto attend = [&](nk::matrix_view<scalar_t> heads, std::size_t head_count, lookup_t lookup, f32_t score_scale,
                      typename indexes_type_::search_config_t config) noexcept {
        auto const [attended, status] = indexes.attend(
            heads, head_count, lookup, score_scale, outputs(nk::range(std::size_t {0}, heads.extent(0)), nk::slice),
            std::span {carry}.first(heads.extent(0)), driver.executor, config);
        failures += expect(failed(status) || attended.count == heads.extent(0));
        return status;
    };
    for (std::size_t const head_count : {std::size_t {1}, std::size_t {4}}) {
        auto const heads = queries(nk::range(std::size_t {0}, tokens_count * head_count), nk::slice);
        f32_t const score_scale = dense_attend_scale(
            dense_oracle_dots(dense_widened_rows(heads, index_t::vector_dtype_k),
                              dense_widened_rows(vectors.view(), index_t::vector_dtype_k), dimensions));
        nk::matrix_view<f32_t> const attended = outputs(nk::range(std::size_t {0}, heads.extent(0)), nk::slice);
        std::span<softmax_state_t const> const carried {carry.data(), heads.extent(0)};
        failures += expect(succeeded(attend(heads, head_count, table.lookup(nodes_count), score_scale, exhaustive)));
        failures += dense_expect_full_attention(
            vectors.view(), heads, head_count,
            [&](std::size_t row) noexcept { return values.view().row(row).as_vector(); }, score_scale, attended,
            carried);
        failures += expect(succeeded(attend(heads, head_count, table.lookup(nodes_count), score_scale, {})));
        failures += dense_expect_attention_bounds(vectors.view(), values.view(), heads, score_scale, attended, carried);
    }

    // Every shard walks every node, so the key past a shortened table refuses.
    failures += expect_eq(attend(queries.view(), 4, table.lookup(nodes_count - 1), 1, exhaustive),
                          status_t::shape_mismatch_k);
    failures += expect_eq(attend(queries.view(), 3, table.lookup(nodes_count), 1, exhaustive),
                          status_t::shape_mismatch_k);
    failures += expect_eq(indexes
                              .attend(queries.view(), 4, table.lookup(nodes_count), 1,
                                      outputs(nk::range(std::size_t {0}, std::size_t {1}), nk::slice),
                                      std::span {carry}, driver.executor, exhaustive)
                              .status,
                          status_t::config_mismatch_k);
    return failures;
}

/** One dtype's device attention over its dot cell, then a kernel that is not a dot refusing it
 *  before reading any input; every ledger must balance. */
template <nk_dtype_t dtype_, nk_kernel_kind_t refusing_kernel_>
std::size_t test_vram_attend_cell(char const* name, u64_t seed) {
    using dot_cell_t = vram_cell<gpu_vendor_t::nvidia_k, dtype_, nk_kernel_dot_k>;
    using refusing_cell_t = vram_cell<gpu_vendor_t::nvidia_k, dtype_, refusing_kernel_>;
    using index_t = typename dot_cell_t::index_t;
    std::size_t failures = 0;
    {
        allocation_ledger_t ledger;
        failures += dense_expect_vram_attend<index_t>(
            [](allocation_ledger_t& into) { return dot_cell_t::make_index(into); }, ledger, seed);
        failures += expect_eq(ledger.live_bytes.load(), 0);
    }
    {
        allocation_ledger_t ledger;
        {
            std::unique_ptr<typename refusing_cell_t::index_t> const index = refusing_cell_t::make_index(ledger);
            typename refusing_cell_t::index_t::traversal_context_t context {{}};
            failures += expect_eq(index
                                      ->attend(typename refusing_cell_t::index_t::vectors_view_t {}, 1,
                                               dense_cuda_value_lookup<typename index_t::fragment_t> {}, 1.f,
                                               nk::matrix_span<f32_t> {}, std::span<softmax_state_t> {}, context)
                                      .status,
                                  status_t::config_mismatch_k);
        }
        failures += expect_eq(ledger.live_bytes.load(), 0);
    }
    return report(fmt::format("cuda-{}+attend", name).c_str(), failures);
}

/** Device attention over every dtype cell the settings select, then over the sharded collection. */
static std::size_t check_vram_nvidia_attend(settings_t const& settings) {
    std::size_t failures_count = 0;
    if (settings.selects("vram_index_bf16"))
        failures_count += test_vram_attend_cell<nk_bf16_k, nk_kernel_angular_k>("bf16", settings.seed);
#if USEARCH_DENSE_DTYPES >= 2
    if (settings.selects("vram_index_u1"))
        failures_count += test_vram_attend_cell<nk_u1_k, nk_kernel_hamming_k>("u1", settings.seed);
#endif
#if USEARCH_DENSE_DTYPES >= 3
    if (settings.selects("vram_index_i8"))
        failures_count += test_vram_attend_cell<nk_i8_k, nk_kernel_angular_k>("i8", settings.seed);
#endif
#if USEARCH_DENSE_DTYPES >= 4
    if (settings.selects("vram_index_e4m3"))
        failures_count += test_vram_attend_cell<nk_e4m3_k, nk_kernel_angular_k>("e4m3", settings.seed);
#endif
#if USEARCH_DENSE_DTYPES >= 5
    if (settings.selects("vram_index_f16"))
        failures_count += test_vram_attend_cell<nk_f16_k, nk_kernel_angular_k>("f16", settings.seed);
#endif
#if USEARCH_DENSE_DTYPES >= 6
    if (settings.selects("vram_index_u8"))
        failures_count += test_vram_attend_cell<nk_u8_k, nk_kernel_angular_k>("u8", settings.seed);
#endif
#if USEARCH_DENSE_DTYPES >= 7
    if (settings.selects("vram_index_e5m2"))
        failures_count += test_vram_attend_cell<nk_e5m2_k, nk_kernel_angular_k>("e5m2", settings.seed);
#endif
#if USEARCH_DENSE_DTYPES >= 8
    if (settings.selects("vram_index_i4"))
        failures_count += test_vram_attend_cell<nk_i4_k, nk_kernel_angular_k>("i4", settings.seed);
#endif
#if USEARCH_DENSE_DTYPES >= 9
    if (settings.selects("vram_index_u4"))
        failures_count += test_vram_attend_cell<nk_u4_k, nk_kernel_angular_k>("u4", settings.seed);
#endif
#if USEARCH_DENSE_DTYPES >= 10
    if (settings.selects("vram_index_e3m2"))
        failures_count += test_vram_attend_cell<nk_e3m2_k, nk_kernel_angular_k>("e3m2", settings.seed);
#endif
#if USEARCH_DENSE_DTYPES >= 11
    if (settings.selects("vram_index_e2m3"))
        failures_count += test_vram_attend_cell<nk_e2m3_k, nk_kernel_angular_k>("e2m3", settings.seed);
#endif
    if (settings.selects("vram_indexes"))
        failures_count += report(
            "cuda-sharded bf16+attend",
            dense_expect_vram_indexes_attend<vram_nvidia_indexes<u32_t, nk_bf16_k, dense_nvidia_capabilities_k>>(
                settings.seed));
    return failures_count;
}

std::size_t check_vram_nvidia(settings_t const& settings, gpu_device_t const& device) {
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    if (cuda_set_device(ordinal) != cuda_success_k)
        return fmt::println("- CUDA: device #{} does not answer", ordinal), 1;
    fmt::println("- CUDA: {} {} #{}, {} SMs, {} MiB", device.name,
                 usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(device.gpu_class)), ordinal,
                 device.compute_units, device.memory_bytes >> 20);
    selected_device = ordinal;
    std::size_t failures_count = check_vram_index_cells<gpu_vendor_t::nvidia_k>(settings);
    if (settings.selects("vram_indexes")) failures_count += check_vram_indexes<gpu_vendor_t::nvidia_k>(settings);
    return failures_count + check_vram_nvidia_attend(settings);
}

static bool const vram_nvidia_registered = (vram_suites({gpu_vendor_t::nvidia_k, &check_vram_nvidia}), true);

} // namespace unum::usearch::test::dense
