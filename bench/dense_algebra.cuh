/**
 *  @file bench/dense_algebra.cuh
 *  @author Ash Vardanian
 *  @date September 27, 2026
 *  @brief The backend-neutral surface of the algebra benchmarks - graph products and graph attention - shared by
 *      every dense-index residency.
 *
 *  The search harnesses build the index and report it through @c dense_search.cuh; this header adds what only
 *  @c multiply and @c attend read: their flags, the value tables attend looks up, the exact attention oracle with its
 *  reference files and pair dumps, and their reports. Like its sibling it names no index, so the same surface compiles
 *  under host compilers and nvcc.
 */
#pragma once

#include <cmath>  // `std::exp` & `std::sin` for the attention oracle
#include <cstdio> // `std::fopen`, `std::fwrite` - the pair dumps

#include <limits> // `std::numeric_limits` for the oracle's running maximum

#include "dense_search.cuh"

namespace unum::usearch::bench {

#pragma region Options and Styling

/** The flags only the algebra binaries take, beside the @ref settings_t every binary shares. */
struct algebra_settings_t {

    /** Optional `.bf16bin` values, one row per base vector as wide as the keys, that attend looks up; absent draws
     *  a synthetic table. */
    std::string_view values_path;

    /** Query heads per token: consecutive query rows that share one attend walk. */
    count_t head_count {.value = 1};

    /** Optional `.f64bin` exact references, one row of maximum, partition and attended row per judged query;
     *  absent computes them on the host. */
    std::string_view reference_path;

    /** Optional file receiving every pair each judged query's multiply scored: per query a `u32` count, then
     *  that many `u32` key rows and `f32` dots. */
    std::string_view pairs_path;

    /** The factor turning a dot into a logit; zero takes the reciprocal square root of the dimensions. */
    double attend_scale = 0;
};

/** The flags only the algebra binaries take; both verbs are defined over the dot kernel, so it becomes the default. */
inline std::array<flag_t, 5> algebra_flags(settings_t& settings, algebra_settings_t& algebra) noexcept {
    settings.kernel = nk_kernel_dot_k;
    return {{
        {"--values", "path", "`.bf16bin`, one value row per base vector as wide as the keys; synthetic by default",
         &algebra.values_path},
        {"--head-count", "count", "query heads per token, consecutive rows sharing one attend walk; 1 by default",
         &algebra.head_count},
        {"--reference", "path", "`.f64bin` exact attention of the judged queries, skipping the host's full pass",
         &algebra.reference_path},
        {"--pairs", "path", "file receiving every pair each judged query's multiply scored", &algebra.pairs_path},
        {"--attend-scale", "factor", "dot-to-logit factor; one over the root of the dimensions by default",
         &algebra.attend_scale},
    }};
}

/** Checks the algebra flags against the shared ones: the metric must stay the dot both verbs are
 *  defined over. */
inline bool resolve_algebra_settings(settings_t const& settings) {
    if (settings.kernel == nk_kernel_dot_k) return true;
    fmt::println(stderr, "--metric=\"{}\" does not parse, expected dot, the only kernel of multiply and attend",
                 settings.metric_name);
    return false;
}

/** Prints the algebra settings as "- Name: value", after the shared ones. */
inline void print(algebra_settings_t const& algebra) {
    fmt::println("- Values: {}", algebra.values_path.empty() ? "synthetic" : algebra.values_path);
    fmt::println("- Head count: {}", algebra.head_count.value);
    fmt::println("- Reference: {}", algebra.reference_path.empty() ? "the host oracle" : algebra.reference_path);
    fmt::println("- Pairs: {}", algebra.pairs_path.empty() ? "none" : algebra.pairs_path);
    if (algebra.attend_scale) fmt::println("- Attend scale: {}", algebra.attend_scale);
    else fmt::println("- Attend scale: one over the root of the dimensions");
}

#pragma endregion

#pragma region Inputs

/** Maps one RetriEval matrix file into @p view, printing why on failure. */
template <typename scalar_type_>
bool map_bench_matrix(std::string_view path, mapped_file_t& mapping, nk::matrix_view<scalar_type_>& view) {
    auto opened = mapped_file_t::read(path);
    if (failed(opened.status)) return fmt::println("Failed to map {}", path), false;
    auto const extents = utilities::parse_matrix_header<scalar_type_, u32_t>(opened.value.view());
    if (failed(extents.status)) return fmt::println("Failed to parse the header of {}", path), false;
    mapping = std::move(opened.value);
    view = utilities::view_matrix_payload<scalar_type_, u32_t>(mapping.view(), extents.value.rows,
                                                               extents.value.columns);
    return true;
}

/** What both phases read, opened before any clock: the queries staged in the stored scalar, the
 *  value rows, and the logit scales. */
template <typename scalar_type_>
struct algebra_inputs {

    /** Every query row in the stored scalar, and its view over whole tokens of `--head-count` rows,
     *  @b [queries,dimensions]. */
    std::vector<scalar_type_> staged;
    nk::matrix_view<scalar_type_> rows;

    /** One bf16 value row per viewed dataset row, as wide as the keys, which the oracle reads: the
     *  mapped `--values` file, or a deterministic synthetic table. */
    mapped_file_t values_file;
    std::vector<nk::bf16_t> synthetic_values;
    nk::matrix_view<nk::bf16_t> value_rows;

    /** The same rows in node order, stored like the keys, which attend looks up. Every value
     *  carries @c value_offset, so unsigned kinds keep their negatives, under @c value_scale, so
     *  integer and byte-sized kinds spend their whole range; attended rows undo both. */
    std::vector<byte_t> node_values;
    f32_t value_offset = 0, value_scale = 1;

    /** The logit scale over the dataset's own scalars, and over the rescaled keys and queries the
     *  index scores. */
    f32_t oracle_scale = 0, score_scale = 0;

    /** Queries the oracle judges. */
    std::size_t judged = 0;

    /** Bytes of one node-ordered value row. */
    std::size_t value_row_bytes() const noexcept {
        return value_rows.extent(1) * nk_dtype_bits(scalar_type_::dtype()) / 8;
    }

    nk::matrix_view<scalar_type_> node_values_view() const noexcept {
        return {reinterpret_cast<scalar_type_ const*>(node_values.data()), node_values.size() / value_row_bytes(),
                value_rows.extent(1)};
    }
};

/** Opens everything @ref algebra_inputs holds for the searched queries of @p streamed, with value rows
 *  gathered into node order through @p id_to_row, each node's dataset row; the table spans every
 *  viewed row and every row an ID names. */
template <typename scalar_type_, typename streamed_dataset_type_>
bool open_algebra_inputs(streamed_dataset_type_ const& streamed, algebra_settings_t const& algebra,
                         std::span<u32_t const> id_to_row, algebra_inputs<scalar_type_>& inputs) {
    settings_t const& settings = streamed.bench.settings;
    std::size_t const dimensions = searched_rows(streamed).extent(1);
    std::size_t const first_row = settings.first_vector.value;
    std::size_t rows_count = streamed.vectors.extent(0);
    for (u32_t const row : id_to_row) rows_count = std::max<std::size_t>(rows_count, row - first_row + 1);
    if (!settings.heads_path.empty() || !settings.weights_path.empty())
        return fmt::println("--heads ranks searches alone; attend weighs plain logits"), false;

    std::vector<f32_t> widened(dimensions);
    if (algebra.values_path.empty()) {
        inputs.synthetic_values.resize(rows_count * dimensions);
        for (std::size_t row = 0; row != rows_count; ++row) {
            for (std::size_t column = 0; column != dimensions; ++column)
                widened[column] = static_cast<f32_t>(std::sin(0.37 * double(row) + 0.11 * double(column)));
            [[maybe_unused]] nk_status_t const cast = nk_cast_serial(widened.data(), nk_f32_k,
                                                                     inputs.synthetic_values.data() + row * dimensions,
                                                                     nk_bf16_k, dimensions, nullptr);
        }
        inputs.value_rows = {inputs.synthetic_values.data(), rows_count, dimensions};
    }
    else {
        nk::matrix_view<nk::bf16_t> mapped;
        if (!map_bench_matrix(algebra.values_path, inputs.values_file, mapped)) return false;
        if (mapped.extent(0) < first_row + rows_count || mapped.extent(1) != dimensions)
            return fmt::println("--values {} must be a .bf16bin of at least {} rows of {} columns", algebra.values_path,
                                first_row + rows_count, dimensions),
                   false;
        inputs.value_rows = mapped(nk::range(first_row, first_row + rows_count), nk::slice);
    }

    if constexpr (nk::is_integral_dtype<scalar_type_>() || sizeof(scalar_type_) == 1) {
        f32_t largest = 0;
        for (std::size_t row = 0; row != rows_count; ++row) {
            widen_row(inputs.value_rows.row(row).as_vector(), std::span {widened});
            for (f32_t const value : widened) largest = std::max(largest, std::abs(value));
        }
        // A cast to an unsigned kind would clamp every negative value to zero.
        if constexpr (nk::is_integral_dtype<scalar_type_>() && !nk::is_signed_dtype<scalar_type_>())
            inputs.value_offset = largest;
        if (largest > 0) inputs.value_scale = component_max<scalar_type_>() / (largest + inputs.value_offset);
    }
    std::size_t const row_bytes = inputs.value_row_bytes();
    inputs.node_values.resize(id_to_row.size() * row_bytes);
    for (std::size_t node = 0; node != id_to_row.size(); ++node) {
        widen_row(inputs.value_rows.row(id_to_row[node] - first_row).as_vector(), std::span {widened});
        for (f32_t& value : widened) value = (value + inputs.value_offset) * inputs.value_scale;
        [[maybe_unused]] nk_status_t const cast = nk_cast_serial(widened.data(), nk_f32_k,
                                                                 inputs.node_values.data() + node * row_bytes,
                                                                 scalar_type_::dtype(), dimensions, nullptr);
    }

    inputs.staged = stage_rows<scalar_type_>(searched_rows(streamed), streamed.scale);
    std::size_t const head_count = algebra.head_count.value;
    std::size_t const queries_count = searched_rows(streamed).extent(0) / head_count * head_count;
    inputs.rows = {inputs.staged.data(), queries_count, dimensions};
    inputs.judged = std::min(settings.oracle_queries.value, queries_count);
    inputs.oracle_scale = algebra.attend_scale ? static_cast<f32_t>(algebra.attend_scale)
                                               : static_cast<f32_t>(1 / std::sqrt(double(dimensions)));
    // The index scores keys and queries both rescaled on ingest, so its logits undo the factor twice.
    inputs.score_scale = inputs.oracle_scale / (streamed.scale * streamed.scale);
    return true;
}

#pragma endregion

#pragma region Telemetry and Judgment

/** Exact softmax attention of one query over every viewed row, in @c f64 over the dataset's own scalars -
 *  the reference graph attention over stored keys is judged against. */
struct exact_attention_t {

    /** The largest logit. */
    f64_t maximum = std::numeric_limits<f64_t>::lowest();

    /** The partition against @c maximum. */
    f64_t partition = 0;

    /** The attended row, one value row wide. */
    std::vector<f64_t> output;
};

/** Loads `--reference` rows for the first @p judged queries, or returns empty when the file disagrees. */
inline std::vector<exact_attention_t> load_exact_attention(std::string_view path, std::size_t judged,
                                                           std::size_t columns) {
    mapped_file_t mapping;
    nk::matrix_view<nk::f64_t> rows;
    if (!map_bench_matrix(path, mapping, rows) || rows.extent(0) < judged || rows.extent(1) != columns + 2) return {};
    std::vector<exact_attention_t> exact(judged);
    for (std::size_t query = 0; query != judged; ++query) {
        f64_t const* const row = reinterpret_cast<f64_t const*>(rows.row(query).data());
        exact[query] = {row[0], row[1], std::vector<f64_t>(row + 2, row + 2 + columns)};
    }
    return exact;
}

/** Exact attention of the first @p judged queries over every row of @p vectors, in one online pass, so each value
 *  row is widened once for all queries. */
template <typename dataset_scalar_type_>
std::vector<exact_attention_t> exact_attention(nk::matrix_view<dataset_scalar_type_> vectors,
                                               nk::matrix_view<dataset_scalar_type_> queries, std::size_t judged,
                                               f32_t score_scale, nk::matrix_view<nk::bf16_t> value_rows) {
    std::size_t const columns = value_rows.extent(1);
    std::vector<exact_scorer_t> scorers;
    std::vector<exact_attention_t> exact(judged);
    for (std::size_t query = 0; query != judged; ++query) {
        scorers.push_back(exact_scorer_of(queries.row(query).as_vector(), nk_kernel_dot_k));
        exact[query].output.assign(columns, 0);
    }
    std::vector<f32_t> row_values(columns);
    for (std::size_t row = 0; row != vectors.extent(0); ++row) {
        widen_row(value_rows.row(row).as_vector(), std::span {row_values});
        for (std::size_t index = 0; index != judged; ++index) {
            f64_t const logit = scorers[index](vectors.row(row).as_vector()) * score_scale;
            exact_attention_t& query = exact[index];
            if (logit > query.maximum) {
                f64_t const rescale = query.partition > 0 ? std::exp(query.maximum - logit) : 0;
                for (f64_t& value : query.output) value *= rescale;
                query.partition *= rescale;
                query.maximum = logit;
            }
            f64_t const weight = std::exp(logit - query.maximum);
            query.partition += weight;
            for (std::size_t column = 0; column != columns; ++column)
                query.output[column] += weight * row_values[column];
        }
    }
    for (exact_attention_t& query : exact)
        for (f64_t& value : query.output) value /= query.partition;
    return exact;
}

/** One pair a judged multiply scored: the key's dataset row and its dot. */
struct scored_pair_t {

    /** The key's dataset row. */
    u32_t row;

    /** The dot the walk scored. */
    f32_t dot;
};

/** Writes `--pairs`: per judged query a `u32` count, then that many pairs. */
inline bool save_scored_pairs(std::string_view path, std::span<std::vector<scored_pair_t> const> pairs) {
    std::FILE* file = std::fopen(std::string(path).c_str(), "wb");
    if (!file) return false;
    bool written = true;
    for (std::vector<scored_pair_t> const& query : pairs) {
        u32_t const count = static_cast<u32_t>(query.size());
        written &= std::fwrite(&count, sizeof(count), 1, file) == 1;
        written &= std::fwrite(query.data(), sizeof(scored_pair_t), query.size(), file) == query.size();
    }
    return std::fclose(file) == 0 && written;
}

/** Throughput and pair counts of the multiply phase. */
inline void report_multiply_metrics(std::size_t rows_count, std::size_t nodes_count, double seconds,
                                    std::size_t scored_pairs) {
    fmt::println("\nMultiply");
    fmt::println("  rows: {} over {} nodes", rows_count, nodes_count);
    fmt::println("  wall: {:.2f} s", seconds);
    fmt::println("  throughput: {} rows/s", humanized(double(rows_count) / seconds));
    fmt::println("  pairs/row: {} · {:.3f}% of the nodes", humanized(double(scored_pairs) / double(rows_count)),
                 100.0 * double(scored_pairs) / double(rows_count) / double(nodes_count));
}

/** Throughput and keys scored of the attend phase over @p rows_count query rows, a query being the @p head_count
 *  consecutive rows that share one walk. */
inline void report_attend_metrics(std::size_t rows_count, std::size_t head_count, std::size_t nodes_count,
                                  double seconds, std::size_t scored_keys) {
    std::size_t const queries_count = rows_count / head_count;
    fmt::println("\nAttend");
    fmt::println("  queries: {} × {} heads over {} nodes", queries_count, head_count, nodes_count);
    fmt::println("  wall: {:.2f} s", seconds);
    fmt::println("  throughput: {} queries/s", humanized(double(queries_count) / seconds));
    fmt::println("  keys/query: {} · {:.3f}% of the nodes", humanized(double(scored_keys) / double(queries_count)),
                 100.0 * double(scored_keys) / double(queries_count) / double(nodes_count));
}

/** Judges attend on the first `--oracle-queries` rows, each head on its own: the relative L2 error of its attended
 *  row in @p outputs against full attention over every key, read from `--reference` or computed in one pass. */
template <typename scalar_type_, typename streamed_dataset_type_>
bool judge_attention(streamed_dataset_type_ const& streamed, algebra_settings_t const& algebra,
                     algebra_inputs<scalar_type_> const& inputs, nk::matrix_span<f32_t> outputs) {
    std::size_t const judged = inputs.judged, columns = inputs.value_rows.extent(1);
    // Softmax weights sum to one, so an attended row carries the values' shift and scale whole.
    for (std::size_t row = 0; row != judged; ++row)
        for (std::size_t column = 0; column != columns; ++column)
            outputs(row, column) = outputs(row, column) / inputs.value_scale - inputs.value_offset;
    auto const started = steady_clock_t::now();
    std::vector<exact_attention_t> const exact = algebra.reference_path.empty()
                                                     ? exact_attention(streamed.vectors, searched_rows(streamed),
                                                                       judged, inputs.oracle_scale, inputs.value_rows)
                                                     : load_exact_attention(algebra.reference_path, judged, columns);
    if (exact.size() != judged)
        return fmt::println("--reference {} must hold {} rows of {} columns", algebra.reference_path, judged,
                            columns + 2),
               false;
    double error_mean = 0, error_worst = 0;
    for (std::size_t row = 0; row != judged; ++row) {
        double difference = 0, magnitude = 0;
        for (std::size_t column = 0; column != columns; ++column) {
            double const delta = double(outputs(row, column)) - exact[row].output[column];
            difference += delta * delta;
            magnitude += exact[row].output[column] * exact[row].output[column];
        }
        double const error = magnitude > 0 ? std::sqrt(difference / magnitude) : std::sqrt(difference);
        error_mean += error, error_worst = std::max(error_worst, error);
    }
    double const oracle_seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    fmt::println("  oracle: {} heads in {:.2f} s", judged, oracle_seconds);
    fmt::println("  relative L2 error: {:.2e} mean · {:.2e} worst", error_mean / double(judged), error_worst);
    return true;
}

#pragma endregion

} // namespace unum::usearch::bench
