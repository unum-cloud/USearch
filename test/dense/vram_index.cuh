/**
 *  @file test/dense/vram_index.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The suite composer for the VRAM-resident index's smoke & concurrency tests, shared by the
 *      per-dtype shards and composed over each vendor's cell.
 *
 *  The shards are vendor-neutral C++, compiled once into each vendor's object library with that
 *  vendor's @c USEARCH_ARCH_<VENDOR>_ set. A vendor's test header specializes @ref vram_cell with
 *  its ledgered index, its sharded collection and their factory, so the scenarios stay templated
 *  on the index alone while the harness's live-bytes-must-balance ledger audits the GPU planes.
 */
#pragma once

#include <string> // `std::string` - a cell's vendor-prefixed name

#include "dense.cuh"

// The one place the test objects name a vendor: the cell this object library composes over.
#if USEARCH_ARCH_CUDA_
#include "vram_nvidia.hpp"
#elif USEARCH_ARCH_ROCM_
#include "vram_amd.hpp"
#elif USEARCH_ARCH_METAL_
#include "vram_apple.hpp"
#endif // USEARCH_ARCH_CUDA_

namespace unum::usearch::test::dense {

/** The vendor whose object library compiles this unit; internal, so each copy keeps its own. */
constexpr gpu_vendor_t vram_object_vendor_k = USEARCH_ARCH_CUDA_    ? gpu_vendor_t::nvidia_k
                                              : USEARCH_ARCH_ROCM_  ? gpu_vendor_t::amd_k
                                              : USEARCH_ARCH_METAL_ ? gpu_vendor_t::apple_k
                                                                    : gpu_vendor_t::unknown_k;

#pragma region Algebra

/** Device graph products against the native dots: every scored pair is counted, a narrow row keeps its width
 *  while counting past it, a short row ends in sentinels, every kept pair names a distinct stored row with its
 *  exact dot, and mismatched spans refuse. */
template <typename index_type_, typename make_index_type_>
std::size_t dense_expect_vram_multiply(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using dot_result_t = typename index_type_::dot_result_t;
    using search_config_t = typename index_type_::search_config_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, left_count = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;
    index_type_ index = make_index(ledger);
    typename index_type_::traversal_context_t context {{}};
    failures += expect(index.reserve({rows_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "multiply"));
    auto const left = vectors(nk::range(std::size_t {0}, left_count), nk::slice);

    auto node_ids = nk::matrix<node_id_t>::zeros({left_count, rows_count}).value;
    auto products = nk::matrix<dot_result_t>::zeros({left_count, rows_count}).value;
    std::array<u32_t, left_count> counts;
    // A narrow call writes the leading columns, so its output rows stride past their width.
    auto multiply = [&](std::size_t wanted, std::size_t products_width, search_config_t config) noexcept {
        return index.multiply(left, node_ids(nk::all, nk::range(std::size_t {0}, wanted), nk::slice),
                              products(nk::all, nk::range(std::size_t {0}, products_width), nk::slice), counts, context,
                              config);
    };

    // An empty index scores nothing, and every row it leaves short ends in sentinels.
    auto const empty = multiply(4, 4, {});
    failures += expect(empty.status == status_t::success_k && empty.value.count == 0);
    nk::matrix_view<node_id_t> const short_rows = node_ids(nk::all, nk::range(std::size_t {0}, std::size_t {4}),
                                                           nk::slice);
    failures += expect(std::all_of(short_rows.begin(), short_rows.end(),
                                   [](auto const& entry) { return entry.second == node_id_unknown_k; }));

    auto assigned_ids = nk::vector<node_id_t>::zeros(rows_count).value;
    failures += expect(index.insert(vectors.view(), assigned_ids.span(), context).status == status_t::success_k);
    failures += expect_eq(index.size(), rows_count);

    search_config_t const exhaustive {static_cast<typename index_type_::small_size_t>(rows_count)};
    for (std::size_t const wanted : {rows_count, std::size_t {4}}) {
        auto const [multiplied, status] = multiply(wanted, wanted, exhaustive);
        failures += expect(status == status_t::success_k);
        failures += expect_eq(multiplied.count, multiplied.computed_distances);
        std::size_t scored_count = 0;
        nk::matrix_view<node_id_t> const kept_ids = node_ids.view();
        nk::matrix_view<dot_result_t> const kept_products = products.view();
        for (std::size_t row = 0; row != left_count; ++row) {
            scored_count += counts[row];
            failures += expect(counts[row] > 0 && counts[row] <= rows_count);
            if (wanted < rows_count) failures += expect(counts[row] > wanted);
            nk::range_t const kept = nk::range(std::size_t {0}, min_of<std::size_t>(counts[row], wanted));
            failures += dense_expect_product_row<rows_count>(left.row(row).as_vector(), vectors.view(),
                                                             kept_ids(row, kept, nk::slice).as_vector(),
                                                             kept_products(row, kept, nk::slice).as_vector(),
                                                             [](node_id_t id) noexcept { return std::size_t {id}; });
        }
        failures += expect_eq(multiplied.count, scored_count);
    }

    failures += expect(multiply(4, 3, {}).status == status_t::config_mismatch_k);
    return failures;
}

/** @ref dense_expect_vram_multiply for the dot kernel, then the ledger it left must balance; no
 *  other kernel offers it. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_vram_multiply(make_index_type_ const& make_index, u64_t seed) {
    if constexpr (kernel_ != nk_kernel_dot_k) return 0;
    else {
        allocation_ledger_t ledger;
        std::size_t const failures = dense_expect_vram_multiply<index_type_>(make_index, ledger, seed);
        return failures + expect_eq(ledger.live_bytes.load(), 0);
    }
}

/** An index without device attention refuses it with @c unsupported_k, whatever it is given. */
template <typename index_type_, typename make_index_type_>
std::size_t test_dense_vram_attend_refused(make_index_type_ const& make_index) {
    allocation_ledger_t ledger;
    std::size_t failures = 0;
    {
        index_type_ index = make_index(ledger);
        typename index_type_::traversal_context_t context {{}};
        failures += expect_eq(index
                                  .attend(typename index_type_::vectors_view_t {}, 1, nullptr, 1.f,
                                          nk::matrix_span<f32_t> {}, std::span<softmax_state_t> {}, context)
                                  .status,
                              status_t::unsupported_k);
    }
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** A metric outside the dtype's family reaches no kit, so the first reservation refuses it before
 *  mapping anything. */
template <typename index_type_>
    requires dense_device_resident<index_type_>
std::size_t test_dense_foreign_metric() {
    index_type_ index {index_type_::vector_dtype_k == nk_u1_k ? nk_kernel_angular_k : nk_kernel_hamming_k};
    return expect_eq(index.reserve({64, connectivity_16_k, 64}).status, status_t::unknown_kind_k);
}

/** Device search under indexer heads against the @c f64 oracle, every node ID its stored row: a pool as wide as the
 *  index walks every node, so each query lands exactly the top by the score Σₕ wₕ · ReLU(qₕ · k), at 1 − score. */
template <typename index_type_, typename make_index_type_>
std::size_t dense_expect_vram_indexer_search(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                             u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    constexpr std::size_t nodes_count = dense_thread_nodes_k, queries_count = 4, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;
    index_type_ index = make_index(ledger);
    typename index_type_::traversal_context_t context {{}};
    failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto const vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexer"));
    std::vector<node_id_t> assigned_ids(nodes_count);
    if (failed(index.insert(vectors.view(), {assigned_ids.data(), nodes_count}, context).status))
        return failures + expect(false);
    failures += expect_contiguous_ids(std::span {assigned_ids}, 0);

    auto const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
    dense_indexer_inputs<scalar_t> const inputs = dense_make_indexer_inputs(vectors.view(), queries_count);
    typename index_type_::search_config_t const exhaustive {nodes_count, wanted};
    auto node_ids = nk::matrix<node_id_t>::zeros({queries_count, wanted}).value;
    auto distances = nk::matrix<distance_t>::zeros({queries_count, wanted}).value;
    failures += expect(
        index.search(queries, inputs.heads(), node_ids.span(), distances.span(), context, exhaustive).status ==
        status_t::success_k);
    for (std::size_t query = 0; query != queries_count; ++query)
        failures += dense_expect_indexer_matches(inputs, query, node_ids.view().row(query).as_vector(),
                                                 distances.view().row(query).as_vector());
    return failures;
}

/** @ref dense_expect_vram_indexer_search for the dot kernel over floating-point keys, then the
 *  ledger must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_vram_indexer_search(make_index_type_ const& make_index, u64_t seed) {
    if constexpr (kernel_ != nk_kernel_dot_k || nk::is_integral_dtype<typename index_type_::scalar_t>()) return 0;
    else {
        allocation_ledger_t ledger;
        std::size_t const failures = dense_expect_vram_indexer_search<index_type_>(make_index, ledger, seed);
        return failures + expect_eq(ledger.live_bytes.load(), 0);
    }
}

#pragma endregion Algebra

/** Composes one (dtype, kernel) cell's suite over @p vendor_ 's ledgered index, reported under
 *  @p name prefixed by the vendor's toolkit. */
template <gpu_vendor_t vendor_, nk_kernel_kind_t kernel_, nk_dtype_t dtype_ = nk_i8_k>
std::size_t test_vram_configuration(char const* name, u64_t seed) {
    using cell_t = vram_cell<vendor_, dtype_, kernel_>;
    using index_t = typename cell_t::index_t;
    static_assert(std::is_move_constructible_v<typename index_t::save_plan_t> &&
                      !std::is_copy_constructible_v<typename index_t::save_plan_t> &&
                      !std::is_move_assignable_v<typename index_t::save_plan_t>,
                  "The device plans are move-only and never assigned over");
    static_assert(std::is_move_constructible_v<typename index_t::load_plan_t> &&
                      !std::is_copy_constructible_v<typename index_t::load_plan_t> &&
                      !std::is_move_assignable_v<typename index_t::load_plan_t>,
                  "The device plans are move-only and never assigned over");
    auto make_index = [](allocation_ledger_t& ledger) { return cell_t::make_index(ledger); };
    std::string const vendor_name = fmt::format("{}-{}", cell_t::vendor_prefix_k, name);
    // The suite is composed here, not reflected: every call below is a capability claim
    // this fully-concurrent, serializable category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_guards<index_t>(make_index, seed);
    failures_count += test_dense_foreign_metric<index_t>();
    failures_count += test_dense_three_vectors<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_growth_and_recall<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_save_load_roundtrip<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_load_append<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_concurrent_build<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_concurrent_search<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_refine_settles<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_health_audits<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_heal_islands<index_t>(make_index, seed);
    failures_count += test_dense_vram_multiply<index_t, kernel_>(make_index, seed);
    if constexpr (!requires { cell_t::attends_k; })
        failures_count += test_dense_vram_attend_refused<index_t>(make_index);
    failures_count += test_dense_vram_indexer_search<index_t, kernel_>(make_index, seed);
    return report(vendor_name.c_str(), failures_count);
}

/** Per-dtype VRAM suite entries the shard translation units define for the vendor of their object
 *  library and its entry composes explicitly; each returns its shard's failure count. */
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_bf16(settings_t const& settings);
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_f16(settings_t const& settings);
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_e5m2(settings_t const& settings);
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_e4m3(settings_t const& settings);
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_e3m2(settings_t const& settings);
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_e2m3(settings_t const& settings);
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_i8(settings_t const& settings);
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_i4(settings_t const& settings);
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_u8(settings_t const& settings);
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_u4(settings_t const& settings);
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_u1(settings_t const& settings);

/** Runs every per-dtype cell the settings select for @p vendor_, in the dtype ladder's order. */
template <gpu_vendor_t vendor_>
std::size_t check_vram_index_cells(settings_t const& settings) {
    std::size_t failures_count = 0;
    if (settings.selects("vram_index_bf16")) failures_count += check_vram_index_bf16<vendor_>(settings);
#if USEARCH_DENSE_DTYPES >= 2
    if (settings.selects("vram_index_u1")) failures_count += check_vram_index_u1<vendor_>(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 3
    if (settings.selects("vram_index_i8")) failures_count += check_vram_index_i8<vendor_>(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 4
    if (settings.selects("vram_index_e4m3")) failures_count += check_vram_index_e4m3<vendor_>(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 5
    if (settings.selects("vram_index_f16")) failures_count += check_vram_index_f16<vendor_>(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 6
    if (settings.selects("vram_index_u8")) failures_count += check_vram_index_u8<vendor_>(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 7
    if (settings.selects("vram_index_e5m2")) failures_count += check_vram_index_e5m2<vendor_>(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 8
    if (settings.selects("vram_index_i4")) failures_count += check_vram_index_i4<vendor_>(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 9
    if (settings.selects("vram_index_u4")) failures_count += check_vram_index_u4<vendor_>(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 10
    if (settings.selects("vram_index_e3m2")) failures_count += check_vram_index_e3m2<vendor_>(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 11
    if (settings.selects("vram_index_e2m3")) failures_count += check_vram_index_e2m3<vendor_>(settings);
#endif
    return failures_count;
}

} // namespace unum::usearch::test::dense
