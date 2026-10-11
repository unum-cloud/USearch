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

#include <cmath>   // `std::exp` & `std::sin` for the attention oracle
#include <cstdio>  // `std::fopen`, `std::fwrite` - the pair dumps
#include <cstdlib> // `std::strtod` - the attend scale
#include <cstring> // `std::memcpy` - the bf16 value rows

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

    /** The `--attend-scale` factor as spelled, which @ref resolve_algebra_settings parses. */
    std::string_view attend_scale_text;

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
         &algebra.attend_scale_text},
    }};
}

/** Checks the algebra flags against the shared ones, printing the first that does not parse: the metric must
 *  stay the dot both verbs are defined over, and the attend scale must be a non-negative real. */
inline bool resolve_algebra_settings(settings_t const& settings, algebra_settings_t& algebra) {
    if (settings.kernel != nk_kernel_dot_k)
        return fmt::println(stderr,
                            "--metric=\"{}\" does not parse, expected dot, the only kernel of multiply and attend",
                            settings.metric_name),
               false;
    if (algebra.attend_scale_text.empty()) return true;
    // Text values view the tail of their `argv` entry, so `strtod` stops at its terminator.
    char* end = nullptr;
    algebra.attend_scale = std::strtod(algebra.attend_scale_text.data(), &end);
    if (end == algebra.attend_scale_text.data() + algebra.attend_scale_text.size() && algebra.attend_scale >= 0)
        return true;
    fmt::println(stderr, "--attend-scale=\"{}\" does not parse, expected a non-negative real",
                 algebra.attend_scale_text);
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

/** Value rows addressed by viewed dataset row: a mapped `--values` file, or a deterministic synthetic table. */
struct attention_value_rows_t {

    /** The `--values` file, when one was given. */
    mapped_file_t mapping;

    /** The synthetic rows, when no file was given. */
    std::vector<nk::bf16_t> synthetic;

    /** One bf16 value row per viewed dataset row. */
    nk::matrix_view<nk::bf16_t> rows;
};

/** Opens `--values` over the viewed rows from @p first_row, @p columns wide like the keys, or draws the synthetic
 *  table in its absence. */
inline bool open_attention_value_rows(algebra_settings_t const& algebra, std::size_t first_row, std::size_t rows_count,
                                      std::size_t columns, attention_value_rows_t& values) {
    if (algebra.values_path.empty()) {
        values.synthetic.resize(rows_count * columns);
        std::vector<f32_t> row_values(columns);
        for (std::size_t row = 0; row != rows_count; ++row) {
            for (std::size_t column = 0; column != columns; ++column)
                row_values[column] = static_cast<f32_t>(std::sin(0.37 * double(row) + 0.11 * double(column)));
            [[maybe_unused]] nk_status_t const cast = nk_cast_serial(
                row_values.data(), nk_f32_k, values.synthetic.data() + row * columns, nk_bf16_k, columns, nullptr);
        }
        values.rows = {values.synthetic.data(), rows_count, columns};
        return true;
    }
    nk::matrix_view<nk::bf16_t> mapped;
    if (!map_bench_matrix(algebra.values_path, values.mapping, mapped)) return false;
    if (mapped.extent(0) < first_row + rows_count || mapped.extent(1) != columns)
        return fmt::println("--values {} must be a .bf16bin of at least {} rows of {} columns", algebra.values_path,
                            first_row + rows_count, columns),
               false;
    values.rows = mapped(nk::range(first_row, first_row + rows_count), nk::slice);
    return true;
}

/** The node-addressed copy attend looks up, as wide as the keys and stored like them, so every backend finds each
 *  value row under its own node. */
template <typename scalar_type_>
struct attention_values {

    /** Columns per value row. */
    std::size_t columns = 0;

    /** The factor every stored value carries; attended rows divide by it. */
    f32_t scale = 1;

    /** The rows, one per node ID, back to back. */
    std::vector<byte_t> stored;

    /** Bytes of one stored row. */
    std::size_t row_bytes() const noexcept { return columns * nk_dtype_bits(scalar_type_::dtype()) / 8; }

    /** The value view over @p data, wherever the rows were copied. */
    nk::matrix_view<scalar_type_> view(byte_t const* data) const noexcept {
        return {reinterpret_cast<scalar_type_ const*>(data), stored.size() / row_bytes(), columns};
    }
};

/** Gathers @p value_rows into node order through @p id_to_row, stored like the keys; integer and byte-sized
 *  kinds map the largest magnitude onto their own, as the keys do, so small values keep their precision. */
template <typename scalar_type_>
attention_values<scalar_type_> make_attention_values(std::span<u32_t const> id_to_row,
                                                     nk::matrix_view<nk::bf16_t> value_rows) {
    std::size_t const columns = value_rows.extent(1);
    attention_values<scalar_type_> values {columns};
    values.stored.resize(id_to_row.size() * values.row_bytes());
    if constexpr (std::is_same_v<scalar_type_, nk::bf16_t>) {
        for (std::size_t node = 0; node != id_to_row.size(); ++node)
            std::memcpy(values.stored.data() + node * values.row_bytes(), value_rows.row(id_to_row[node]).data(),
                        values.row_bytes());
        return values;
    }
    std::vector<f32_t> widened(columns);
    if constexpr (nk::is_integral_dtype<scalar_type_>() || sizeof(scalar_type_) == 1) {
        f32_t largest = 0;
        for (std::size_t row = 0; row != value_rows.extent(0); ++row) {
            widen_row(value_rows.row(row).as_vector(), std::span {widened});
            for (f32_t const value : widened) largest = std::max(largest, std::abs(value));
        }
        values.scale = largest > 0 ? component_max<scalar_type_>() / largest : 1;
    }
    for (std::size_t node = 0; node != id_to_row.size(); ++node) {
        widen_row(value_rows.row(id_to_row[node]).as_vector(), std::span {widened});
        for (f32_t& value : widened) value *= values.scale;
        [[maybe_unused]] nk_status_t const cast = nk_cast_serial(widened.data(), nk_f32_k,
                                                                 values.stored.data() + node * values.row_bytes(),
                                                                 scalar_type_::dtype(), columns, nullptr);
    }
    return values;
}

/** The logit scale @c --attend-scale asks for, or the reciprocal square root of the dimensions. */
inline f32_t attention_score_scale(algebra_settings_t const& algebra, std::size_t dimensions) noexcept {
    return algebra.attend_scale ? static_cast<f32_t>(algebra.attend_scale)
                                : static_cast<f32_t>(1 / std::sqrt(double(dimensions)));
}

/** What both phases read, opened before any clock: the queries staged in the stored scalar, the value rows and
 *  their node-addressed copy, and the logit scales. */
template <typename scalar_type_>
struct algebra_inputs {

    /** Every query row in the stored scalar, and its view over whole tokens of `--head-count` rows,
     *  @b [queries,dimensions]. */
    std::vector<scalar_type_> staged;
    nk::matrix_view<scalar_type_> rows;

    attention_value_rows_t value_rows;
    attention_values<scalar_type_> values;

    /** The logit scale over the dataset's own scalars, and over the rescaled keys and queries the index scores. */
    f32_t oracle_scale = 0, score_scale = 0;

    /** Queries the oracle judges. */
    std::size_t judged = 0;
};

/** Opens everything @ref algebra_inputs holds for the searched queries of @p streamed, with value rows
 *  gathered into node order through @p id_to_row. */
template <typename scalar_type_, typename streamed_dataset_type_>
bool open_algebra_inputs(streamed_dataset_type_ const& streamed, algebra_settings_t const& algebra,
                         std::span<u32_t const> id_to_row, algebra_inputs<scalar_type_>& inputs) {
    settings_t const& settings = streamed.bench.settings;
    std::size_t const rows_count = streamed.vectors.extent(0), dimensions = streamed.vectors.extent(1);
    if (!settings.heads_path.empty() || !settings.weights_path.empty())
        return fmt::println("--heads ranks searches alone; attend weighs plain logits"), false;
    if (!open_attention_value_rows(algebra, settings.first_vector.value, rows_count, dimensions, inputs.value_rows))
        return false;
    inputs.staged = stage_rows<scalar_type_>(searched_rows(streamed), streamed.scale);
    std::size_t const head_count = algebra.head_count.value;
    std::size_t const queries_count = searched_rows(streamed).extent(0) / head_count * head_count;
    inputs.rows = {inputs.staged.data(), queries_count, dimensions};
    inputs.judged = std::min(settings.oracle_queries.value, queries_count);
    inputs.oracle_scale = attention_score_scale(algebra, dimensions);
    // The index scores keys and queries both rescaled on ingest, so its logits undo the factor twice.
    inputs.score_scale = inputs.oracle_scale / (streamed.scale * streamed.scale);
    inputs.values = make_attention_values<scalar_type_>(id_to_row, inputs.value_rows.rows);
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
        scorers.push_back(exact_scorer_of(queries.row(query).as_vector()));
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

    /** The key's viewed dataset row. */
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
    std::size_t const judged = inputs.judged, columns = inputs.values.columns;
    if (inputs.values.scale != 1)
        for (std::size_t row = 0; row != judged; ++row)
            for (std::size_t column = 0; column != columns; ++column) outputs(row, column) /= inputs.values.scale;
    auto const started = steady_clock_t::now();
    std::vector<exact_attention_t> const exact = algebra.reference_path.empty()
                                                     ? exact_attention(streamed.vectors, searched_rows(streamed),
                                                                       judged, inputs.oracle_scale,
                                                                       inputs.value_rows.rows)
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
