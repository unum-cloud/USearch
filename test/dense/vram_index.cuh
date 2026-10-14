/**
 *  @file test/dense/vram_index.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The suite composer for the VRAM-resident index's smoke & concurrency tests, shared by the
 *      per-dtype shards and composed over each vendor's cell.
 *
 *  The shards are vendor-neutral C++, compiled once into the one object library, each instantiating
 *  every vendor and composing over those @c USEARCH_ARCH_<VENDOR>_ carries. A vendor's test header,
 *  free of its SDK like the library's own, specializes @ref vram_cell with its ledgered index, its
 *  sharded collection and their factory over the stream its entry opened, so the scenarios stay
 *  templated on the index alone while the harness's live-bytes-must-balance ledger audits the GPU
 *  flat matrices.
 */
#pragma once

#include <string> // `std::string` - a cell's vendor-prefixed name

#include "dense.cuh"

#include "vram_amd.hpp"
#include "vram_apple.hpp"
#include "vram_nvidia.hpp"

namespace unum::usearch::test::dense {

/** Whether this build carries @p vendor_ 's GPUs; the shards instantiate every vendor, and the
 *  ones it leaves out compose nothing. */
template <gpu_vendor_t vendor_>
inline constexpr bool vram_compiled = (vendor_ == gpu_vendor_t::nvidia_k && USEARCH_ARCH_CUDA_) ||
                                      (vendor_ == gpu_vendor_t::amd_k && USEARCH_ARCH_ROCM_) ||
                                      (vendor_ == gpu_vendor_t::apple_k && USEARCH_ARCH_METAL_);

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
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
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
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        failures += expect_eq(index
                                  .attend(typename index_type_::vectors_view_t {}, 1, nullptr, 1.f,
                                          nk::matrix_span<f32_t> {}, std::span<softmax_state_t> {}, context)
                                  .status,
                              status_t::unsupported_k);
    }
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** Device attention against exact full attention, value rows looked up by node ID in a device table
 *  the vendor's @c value_table_type_ uploads: an exhaustive walk scores every node, so each head's
 *  output and carry match the @c f64 oracle over every key, with one head per token and with four
 *  sharing its walk, while a narrow walk stays within the oracle's bounds; a node without a value
 *  row, mismatched outputs, and a head count that does not divide the rows or whose softmax states
 *  and accumulators no warp seats refuse. */
template <typename index_type_, typename value_table_type_, typename make_index_type_>
std::size_t dense_expect_vram_attend(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using lookup_t = typename value_table_type_::lookup_t;
    constexpr std::size_t nodes_count = dense_thread_nodes_k, tokens_count = 4, crowded_heads = 1024;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto const vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "attend"));
    auto const values = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "attend", 1));
    std::vector<node_id_t> assigned_ids(nodes_count);
    if (failed(index.insert(vectors.view(), {assigned_ids.data(), nodes_count}, context).status))
        return failures + expect(false);
    failures += expect_contiguous_ids(std::span {assigned_ids}, 0);
    auto const table = value_table_type_::upload(values.view(), make_index.stream);
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

/** A metric outside the dtype's family reaches no kit, so the first reservation refuses it before
 *  mapping anything. */
template <typename index_type_>
    requires dense_device_resident<index_type_>
std::size_t test_dense_foreign_metric() {
    std::unique_ptr<index_type_> const index {new (std::nothrow) index_type_(
        index_type_::vector_dtype_k == nk_u1_k ? nk_kernel_angular_k : nk_kernel_hamming_k)};
    return expect_eq(index->reserve({64, connectivity_16_k, 64}).status, status_t::unknown_kind_k);
}

/** Device search under indexer heads against the @c f64 oracle, every node ID its stored row: a pool as wide as the
 *  index walks every node, so each query lands exactly the top by the score Σₕ wₕ · ReLU(qₕ · k), at 1 − score. */
template <typename index_type_, typename make_index_type_>
std::size_t dense_expect_vram_indexer_search(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                             u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    constexpr std::size_t nodes_count = dense_thread_nodes_k, queries_count = 4, wanted = 200;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto const vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexer"));
    std::vector<node_id_t> assigned_ids(nodes_count);
    if (failed(index.insert(vectors.view(), {assigned_ids.data(), nodes_count}, context).status))
        return failures + expect(false);
    failures += expect_contiguous_ids(std::span {assigned_ids}, 0);

    auto const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
    dense_indexer_inputs<scalar_t> const inputs = dense_make_indexer_inputs(vectors.view(), queries_count);
    // Past the widest register pool and past one ranking round, so the arena frontier's
    // select runs.
    typename index_type_::search_config_t const exhaustive {4 * nodes_count, wanted};
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

/** Builds and refines rows of 2 KB whatever the dtype, too wide for shared memory to seat a
 *  builder's candidate cache, so every insert, refine and merge keeps its scratch in the warp's
 *  arena slice; the refine must reach every row and the walks must find them, then the ledger must
 *  balance. The rows are sign sketches, since independent values this wide leave every pair of
 *  rows near-equidistant. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_vram_wide_build(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    allocation_ledger_t ledger;
    std::size_t failures = 0;
    {
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        std::size_t const dimensions = 8 * 2048 / scalar_t::bits_per_dimension();
        auto const vectors = make_sketched_vectors<scalar_t>(dense_growth_nodes_k, dimensions,
                                                             stream_key(seed, "wide"));
        failures += expect(index.reserve({dense_growth_nodes_k, connectivity_16_k, dimensions}).status ==
                           status_t::success_k);
        // Apple keeps every builder's scratch in the arena, so only shared-memory devices can miss.
        if constexpr (requires { index.device_specs().shared_memory_per_block_bytes; })
            failures += expect(!index.build_scratch_fits());
        failures += expect(index.insert(vectors.view(), {}, context).status == status_t::success_k);
        failures += expect_eq(index.size(), dense_growth_nodes_k);
        auto const [refined, refined_status] = index.refine({0, static_cast<node_id_t>(index.size())}, context);
        failures += expect(refined_status == status_t::success_k && refined.changed_nodes > 0);
        // A dot graph's hubs can stall the heal short of every row.
        if constexpr (kernel_ != nk_kernel_dot_k) {
            failures += expect_eq(refined.reachable_nodes, dense_growth_nodes_k);
            failures += dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
        }
    }
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** Builds and refines the growth set at an expansion past the widest register pool of 32 lanes, so
 *  both walk the arena frontier's rung: no tail entry drops, the walks still find the rows, and the
 *  ledger must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_vram_broad_build(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    allocation_ledger_t ledger;
    std::size_t failures = 0;
    {
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        auto const vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dimensions, stream_key(seed, "broad"));
        failures += expect(index.reserve({dense_growth_nodes_k, connectivity_16_k, dimensions}).status ==
                           status_t::success_k);
        typename index_type_::insert_config_t broad_insert;
        broad_insert.expansion_factor = 1024;
        auto const [inserted, inserted_status] = index.insert(vectors.view(), {}, context, broad_insert);
        failures += expect(inserted_status == status_t::success_k && inserted.tail_drops == 0);
        if constexpr (kernel_ != nk_kernel_dot_k)
            failures += dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
        typename index_type_::refine_config_t broad_refine;
        broad_refine.expansion_factor = 1024;
        failures += expect(index.refine({0, static_cast<node_id_t>(index.size())}, context, broad_refine).status ==
                           status_t::success_k);
    }
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** A vendor this build leaves out composes nothing, so its shards instantiate empty. */
template <nk_kernel_kind_t kernel_, nk_dtype_t dtype_ = nk_i8_k, typename executor_type_>
std::size_t test_vram_configuration(char const*, u64_t, executor_type_&) {
    return 0;
}

/** Composes one (dtype, kernel) cell's suite over the ledgered index of @p executor 's vendor, on
 *  its first stream, reported under @p name prefixed by the vendor's toolkit. */
template <nk_kernel_kind_t kernel_, nk_dtype_t dtype_ = nk_i8_k, typename executor_type_>
    requires vram_compiled<executor_type_::vendor_k>
std::size_t test_vram_configuration(char const* name, u64_t seed, executor_type_& executor) {
    using cell_t = vram_cell<executor_type_::vendor_k, dtype_, kernel_>;
    using index_t = typename cell_t::index_t;
    static_assert(std::is_move_constructible_v<typename index_t::save_plan_t> &&
                      !std::is_copy_constructible_v<typename index_t::save_plan_t> &&
                      !std::is_move_assignable_v<typename index_t::save_plan_t>,
                  "The device plans are move-only and never assigned over");
    static_assert(std::is_move_constructible_v<typename index_t::load_plan_t> &&
                      !std::is_copy_constructible_v<typename index_t::load_plan_t> &&
                      !std::is_move_assignable_v<typename index_t::load_plan_t>,
                  "The device plans are move-only and never assigned over");
    cell_t const cell {executor.stream(0)};
    std::string const vendor_name = fmt::format("{}-{}", cell_t::vendor_prefix_k, name);
    // The suite is composed here, not reflected: every call below is a capability claim
    // this fully-concurrent, serializable category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_guards<index_t>(cell, seed);
    failures_count += test_dense_foreign_metric<index_t>();
    failures_count += test_dense_three_vectors<index_t, kernel_>(cell, seed);
    failures_count += test_dense_growth_and_recall<index_t, kernel_>(cell, seed);
    failures_count += test_dense_save_load_roundtrip<index_t, kernel_>(cell, seed);
    failures_count += test_dense_load_append<index_t, kernel_>(cell, seed);
    failures_count += test_dense_concurrent_build<index_t, kernel_>(cell, seed);
    failures_count += test_dense_concurrent_search<index_t, kernel_>(cell, seed);
    failures_count += test_dense_refine_settles<index_t, kernel_>(cell, seed);
    failures_count += test_dense_health_audits<index_t, kernel_>(cell, seed);
    failures_count += test_dense_heal_islands<index_t>(cell, seed);
    failures_count += test_dense_vram_multiply<index_t, kernel_>(cell, seed);
    if constexpr (!requires { cell_t::attends_k; }) failures_count += test_dense_vram_attend_refused<index_t>(cell);
    failures_count += test_dense_vram_indexer_search<index_t, kernel_>(cell, seed);
    failures_count += test_dense_vram_wide_build<index_t, kernel_>(cell, seed);
    failures_count += test_dense_vram_broad_build<index_t, kernel_>(cell, seed);
    return report(vendor_name.c_str(), failures_count);
}

/** One dtype's device attention over its dot cell on @p executor 's stream, value rows in a
 *  @c value_table_type_ table, then a kernel that is not a dot refusing it before reading any
 *  input; every ledger must balance. */
template <nk_dtype_t dtype_, nk_kernel_kind_t refusing_kernel_, typename value_table_type_, typename executor_type_>
std::size_t test_vram_attend_cell(char const* name, u64_t seed, executor_type_& executor) {
    using dot_cell_t = vram_cell<executor_type_::vendor_k, dtype_, nk_kernel_dot_k>;
    using refusing_cell_t = vram_cell<executor_type_::vendor_k, dtype_, refusing_kernel_>;
    using index_t = typename dot_cell_t::index_t;
    std::size_t failures = 0;
    {
        allocation_ledger_t ledger;
        failures += dense_expect_vram_attend<index_t, value_table_type_>(dot_cell_t {executor.stream(0)}, ledger, seed);
        failures += expect_eq(ledger.live_bytes.load(), 0);
    }
    {
        allocation_ledger_t ledger;
        {
            refusing_cell_t const refusing {executor.stream(0)};
            std::unique_ptr<typename refusing_cell_t::index_t> const index = refusing(ledger);
            typename refusing_cell_t::index_t::traversal_context_t context = refusing.context();
            failures += expect_eq(index
                                      ->attend(typename refusing_cell_t::index_t::vectors_view_t {}, 1,
                                               typename value_table_type_::lookup_t {}, 1.f, nk::matrix_span<f32_t> {},
                                               std::span<softmax_state_t> {}, context)
                                      .status,
                                  status_t::config_mismatch_k);
        }
        failures += expect_eq(ledger.live_bytes.load(), 0);
    }
    return report(fmt::format("{}-{}+attend", dot_cell_t::vendor_prefix_k, name).c_str(), failures);
}

/** Per-dtype VRAM suite entries the shard translation units define for every vendor's executor and
 *  each carried vendor's entry composes explicitly over the executor it opened; each returns its
 *  shard's failure count. */
template <typename executor_type_>
std::size_t check_vram_index_bf16(settings_t const& settings, executor_type_& executor);
template <typename executor_type_>
std::size_t check_vram_index_f16(settings_t const& settings, executor_type_& executor);
template <typename executor_type_>
std::size_t check_vram_index_e5m2(settings_t const& settings, executor_type_& executor);
template <typename executor_type_>
std::size_t check_vram_index_e4m3(settings_t const& settings, executor_type_& executor);
template <typename executor_type_>
std::size_t check_vram_index_e3m2(settings_t const& settings, executor_type_& executor);
template <typename executor_type_>
std::size_t check_vram_index_e2m3(settings_t const& settings, executor_type_& executor);
template <typename executor_type_>
std::size_t check_vram_index_i8(settings_t const& settings, executor_type_& executor);
template <typename executor_type_>
std::size_t check_vram_index_i4(settings_t const& settings, executor_type_& executor);
template <typename executor_type_>
std::size_t check_vram_index_u8(settings_t const& settings, executor_type_& executor);
template <typename executor_type_>
std::size_t check_vram_index_u4(settings_t const& settings, executor_type_& executor);
template <typename executor_type_>
std::size_t check_vram_index_u1(settings_t const& settings, executor_type_& executor);

/** Runs every per-dtype cell the settings select on @p executor 's device, in the dtype ladder's
 *  order. */
template <typename executor_type_>
std::size_t check_vram_index_cells(settings_t const& settings, executor_type_& executor) {
    std::size_t failures_count = 0;
    if (settings.selects("vram_index_bf16")) failures_count += check_vram_index_bf16(settings, executor);
#if USEARCH_DENSE_DTYPES >= 2
    if (settings.selects("vram_index_u1")) failures_count += check_vram_index_u1(settings, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 3
    if (settings.selects("vram_index_i8")) failures_count += check_vram_index_i8(settings, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 4
    if (settings.selects("vram_index_e4m3")) failures_count += check_vram_index_e4m3(settings, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 5
    if (settings.selects("vram_index_f16")) failures_count += check_vram_index_f16(settings, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 6
    if (settings.selects("vram_index_u8")) failures_count += check_vram_index_u8(settings, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 7
    if (settings.selects("vram_index_e5m2")) failures_count += check_vram_index_e5m2(settings, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 8
    if (settings.selects("vram_index_i4")) failures_count += check_vram_index_i4(settings, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 9
    if (settings.selects("vram_index_u4")) failures_count += check_vram_index_u4(settings, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 10
    if (settings.selects("vram_index_e3m2")) failures_count += check_vram_index_e3m2(settings, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 11
    if (settings.selects("vram_index_e2m3")) failures_count += check_vram_index_e2m3(settings, executor);
#endif
    return failures_count;
}

/** Device attention over every dtype cell the settings select on @p executor 's device, value rows
 *  in @c value_table_type_ tables, in the dtype ladder's order. */
template <typename value_table_type_, typename executor_type_>
std::size_t check_vram_attend_cells(settings_t const& settings, executor_type_& executor) {
    std::size_t failures_count = 0;
    if (settings.selects("vram_index_bf16"))
        failures_count += test_vram_attend_cell<nk_bf16_k, nk_kernel_angular_k, value_table_type_>(
            "bf16", settings.seed, executor);
#if USEARCH_DENSE_DTYPES >= 2
    if (settings.selects("vram_index_u1"))
        failures_count += test_vram_attend_cell<nk_u1_k, nk_kernel_hamming_k, value_table_type_>("u1", settings.seed,
                                                                                                 executor);
#endif
#if USEARCH_DENSE_DTYPES >= 3
    if (settings.selects("vram_index_i8"))
        failures_count += test_vram_attend_cell<nk_i8_k, nk_kernel_angular_k, value_table_type_>("i8", settings.seed,
                                                                                                 executor);
#endif
#if USEARCH_DENSE_DTYPES >= 4
    if (settings.selects("vram_index_e4m3"))
        failures_count += test_vram_attend_cell<nk_e4m3_k, nk_kernel_angular_k, value_table_type_>(
            "e4m3", settings.seed, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 5
    if (settings.selects("vram_index_f16"))
        failures_count += test_vram_attend_cell<nk_f16_k, nk_kernel_angular_k, value_table_type_>("f16", settings.seed,
                                                                                                  executor);
#endif
#if USEARCH_DENSE_DTYPES >= 6
    if (settings.selects("vram_index_u8"))
        failures_count += test_vram_attend_cell<nk_u8_k, nk_kernel_angular_k, value_table_type_>("u8", settings.seed,
                                                                                                 executor);
#endif
#if USEARCH_DENSE_DTYPES >= 7
    if (settings.selects("vram_index_e5m2"))
        failures_count += test_vram_attend_cell<nk_e5m2_k, nk_kernel_angular_k, value_table_type_>(
            "e5m2", settings.seed, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 8
    if (settings.selects("vram_index_i4"))
        failures_count += test_vram_attend_cell<nk_i4_k, nk_kernel_angular_k, value_table_type_>("i4", settings.seed,
                                                                                                 executor);
#endif
#if USEARCH_DENSE_DTYPES >= 9
    if (settings.selects("vram_index_u4"))
        failures_count += test_vram_attend_cell<nk_u4_k, nk_kernel_angular_k, value_table_type_>("u4", settings.seed,
                                                                                                 executor);
#endif
#if USEARCH_DENSE_DTYPES >= 10
    if (settings.selects("vram_index_e3m2"))
        failures_count += test_vram_attend_cell<nk_e3m2_k, nk_kernel_angular_k, value_table_type_>(
            "e3m2", settings.seed, executor);
#endif
#if USEARCH_DENSE_DTYPES >= 11
    if (settings.selects("vram_index_e2m3"))
        failures_count += test_vram_attend_cell<nk_e2m3_k, nk_kernel_angular_k, value_table_type_>(
            "e2m3", settings.seed, executor);
#endif
    return failures_count;
}

} // namespace unum::usearch::test::dense
