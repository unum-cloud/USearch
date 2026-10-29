/**
 *  @file bench/dense_index_algebra.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The dense algebra benchmark over the C interface: a RetriEval dataset built into one
 *      inner-product index of any residency and class, then graph products and graph attention of
 *      its queries, attention judged against exact softmax attention over every row.
 *
 *  Both verbs are defined over the inner product, so the index scores by it. Every row is keyed by
 *  its global dataset row, as in the search benchmark, so `--base-vectors` takes a glob of parts
 *  and `--first-vector` a slice inside one part. A multiply reports every pair its walks scored,
 *  and `--pairs` keeps those of the judged queries: per query a @c u32 count, then that many @c u64
 *  global rows beside their @c f64 dots. Attention looks values up on the host, so it runs on RAM
 *  and NVMe: one row per base row as wide as the keys, from `--values` or a synthetic table, cast
 *  into the stored scalar, shifted and scaled into an integer kind's range and undone on the
 *  attended rows.
 *
 *  @verbatim
 *  Flag                Default                      Meaning
 *  --base-vectors      required                     Base vectors, a RetriEval file or a glob of its parts
 *  --query-vectors     the first 1000 base vectors  Query vectors of the same dimensions
 *  --stored-dtype      the file's, else bf16        Stored scalar, cast on ingest
 *  --connectivity      derived                      Graph degree
 *  --expansion-add     derived                      Insertion breadth
 *  --expansion-search  derived                      Breadth of the multiply and attend walks
 *  --first-vector      0                            First dataset row viewed, keyed by its global row
 *  --max-base-vectors  the whole file               Cap on ingested rows
 *  --oracle-queries    16                           Queries the exact attention judges
 *  --batch             65536                        Rows per insert, multiply or attend call
 *  --values            synthetic                    A .bf16bin of one value row per base row
 *  --head-count        1                            Query heads per token sharing one attend walk
 *  --reference         the host oracle              An .f64bin of exact attention per judged query
 *  --pairs             none                         File receiving every pair a judged query's multiply scored
 *  --attend-scale      one over the root of the dimensions  Dot-to-logit factor
 *  @endverbatim
 */
#include <cmath>   // `std::exp`, `std::sin`, `std::sqrt` - the attention oracle and the synthetic values
#include <cstdio>  // `std::fopen`, `std::fwrite` - the pairs file
#include <cstring> // `std::memcpy` - the attended rows

#include <algorithm>  // `std::min`, `std::max`
#include <chrono>     // `std::chrono::steady_clock` - the phase clocks
#include <filesystem> // `std::filesystem::temp_directory_path` - where an NVMe run saves
#include <limits>     // `std::numeric_limits` - the oracle's running maximum
#include <string>     // `std::string` - the pairs path
#include <vector>     // `std::vector` - the value table, the oracle

#include <fmt/format.h>

#include <numkong/numkong.h> // `nk_cast_serial` - the value table, the widened rows

#include "abi.hpp"
#include "bench.hpp" // `humanized`

namespace unum::usearch::bench {
using namespace unum::usearch::tools;

/** Every flag this binary reads beyond the shared settings, its default as the initializer. */
struct algebra_settings_t {
    std::string_view base_vectors, query_vectors, stored_dtype_name, values, reference, pairs;
    count_t connectivity {.minimum = 0};
    count_t expansion_add {.minimum = 0};
    count_t expansion_search {.minimum = 0};
    count_t first_vector {.minimum = 0};
    count_t max_base_vectors {.minimum = 0};
    count_t oracle_queries {.value = 16};
    count_t batch {.value = 65536};
    count_t head_count {.value = 1};
    double attend_scale = 0;
};

/** The flags @ref algebra_settings_t holds. */
inline std::array<flag_t, 15> algebra_flags(algebra_settings_t& algebra) noexcept {
    return {{
        {"--base-vectors", "path", "base vectors, a RetriEval file or a glob of its parts", &algebra.base_vectors},
        {"--query-vectors", "path", "query vectors; the first 1000 base vectors by default", &algebra.query_vectors},
        {"--stored-dtype", "name", "stored scalar, cast on ingest; the file's, else bf16, by default",
         &algebra.stored_dtype_name},
        {"--connectivity", "degree", "graph degree; derived by default", &algebra.connectivity},
        {"--expansion-add", "breadth", "insertion breadth; derived by default", &algebra.expansion_add},
        {"--expansion-search", "breadth", "breadth of the multiply and attend walks; derived by default",
         &algebra.expansion_search},
        {"--first-vector", "row", "first dataset row viewed, keyed by its global row; 0 by default",
         &algebra.first_vector},
        {"--max-base-vectors", "count", "cap on ingested rows; the whole file by default", &algebra.max_base_vectors},
        {"--oracle-queries", "count", "queries the exact attention judges; 16 by default", &algebra.oracle_queries},
        {"--batch", "rows", "rows per insert, multiply or attend call; 65536 by default", &algebra.batch},
        {"--values", "path", "a .bf16bin of one value row per base row; synthetic by default", &algebra.values},
        {"--head-count", "count", "query heads per token sharing one attend walk; 1 by default", &algebra.head_count},
        {"--reference", "path", "an .f64bin of exact attention per judged query; the host oracle by default",
         &algebra.reference},
        {"--pairs", "path", "file receiving every pair a judged query's multiply scored", &algebra.pairs},
        {"--attend-scale", "factor", "dot-to-logit factor; one over the root of the dimensions by default",
         &algebra.attend_scale},
    }};
}

/** The NumKong dtype of @p scalar, for the casts the table and the oracle make. */
inline nk_dtype_t dtype_of(usearch_scalar_t scalar) noexcept {
    switch (scalar) {
    case usearch_scalar_f64_k: return nk_f64_k;
    case usearch_scalar_f32_k: return nk_f32_k;
    case usearch_scalar_bf16_k: return nk_bf16_k;
    case usearch_scalar_f16_k: return nk_f16_k;
    case usearch_scalar_e5m2_k: return nk_e5m2_k;
    case usearch_scalar_e4m3_k: return nk_e4m3_k;
    case usearch_scalar_e3m2_k: return nk_e3m2_k;
    case usearch_scalar_e2m3_k: return nk_e2m3_k;
    case usearch_scalar_i8_k: return nk_i8_k;
    case usearch_scalar_i4x2_k: return nk_i4_k;
    case usearch_scalar_u8_k: return nk_u8_k;
    case usearch_scalar_u4x2_k: return nk_u4_k;
    case usearch_scalar_u1x8_k: return nk_u1_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Row @p row of @p rows widened to @c f32 into @p widened. */
inline void widen_row(usearch_rows_view_t rows, std::size_t row, std::span<f32_t> widened) noexcept {
    [[maybe_unused]] nk_status_t const cast = nk_cast_serial(static_cast<byte_t const*>(rows.data) + row * rows.stride,
                                                             dtype_of(rows.scalar), widened.data(), nk_f32_k,
                                                             widened.size(), nullptr);
}

/** One value row per viewed base row, in the stored scalar, which the attend lookup hands out by
 *  key: every value shifted by @c offset and scaled by @c scale into the stored kind's range. */
struct value_table_t {
    std::vector<byte_t> rows;
    std::size_t row_bytes = 0, first_row = 0, count = 0;
    f32_t offset = 0, scale = 1;

    /** The seed the looked-up keys were minted under. */
    u64_t seed = 0;
};

/** The attend lookup over a @ref value_table_t: each key's global row, less the table's first. */
inline void lookup_values(void* state, usearch_key_t const* keys, std::size_t count, usearch_scalar_t, std::size_t,
                          void const** rows) {
    value_table_t const& table = *static_cast<value_table_t const*>(state);
    for (std::size_t position = 0; position != count; ++position) {
        u64_t const row = offset_of_key(table.seed, keys[position]).value_or(key_offsets_k);
        rows[position] = row >= table.first_row && row - table.first_row < table.count
                             ? table.rows.data() + (row - table.first_row) * table.row_bytes
                             : nullptr;
    }
}

/** The value rows @p count base rows from @p first_row look up: the `--values` file's, or a
 *  synthetic table, widened to @c f32 row-major. */
inline bool widened_values(std::string_view path, std::size_t first_row, std::size_t count, std::size_t columns,
                           std::vector<f32_t>& values) {
    values.resize(count * columns);
    if (path.empty()) {
        for (std::size_t row = 0; row != count; ++row)
            for (std::size_t column = 0; column != columns; ++column)
                values[row * columns + column] = static_cast<f32_t>(
                    std::sin(0.37 * double(row) + 0.11 * double(column)));
        return true;
    }
    auto [file, file_status] = mapped_rows_t::open(path);
    usearch_rows_view_t const rows = file.view(first_row, count);
    if (failed(file_status) || file.columns() != columns || !rows.data) return false;
    for (std::size_t row = 0; row != count; ++row) widen_row(rows, row, {values.data() + row * columns, columns});
    return true;
}

/** Casts @p values into the stored scalar @p stored, shifting an unsigned kind past every negative
 *  value and scaling an integer or byte-sized kind over its whole range. */
inline value_table_t value_table_of(std::span<f32_t const> values, std::size_t columns, usearch_scalar_t stored,
                                    std::size_t first_row) {
    value_table_t table;
    nk_dtype_t const dtype = dtype_of(stored);
    table.first_row = first_row, table.count = values.size() / columns;
    table.row_bytes = divide_round_up<std::size_t>(columns * nk_dtype_bits(dtype), 8);
    bool const integral = stored == usearch_scalar_i8_k || stored == usearch_scalar_u8_k ||
                          stored == usearch_scalar_i4x2_k || stored == usearch_scalar_u4x2_k;
    bool const unsigned_kind = stored == usearch_scalar_u8_k || stored == usearch_scalar_u4x2_k;
    if (integral) {
        f32_t largest = 0;
        for (f32_t const value : values) largest = std::max(largest, std::abs(value));
        if (unsigned_kind) table.offset = largest;
        f32_t const top = stored == usearch_scalar_i8_k     ? 127.f
                          : stored == usearch_scalar_u8_k   ? 255.f
                          : stored == usearch_scalar_i4x2_k ? 7.f
                                                            : 15.f;
        if (largest > 0) table.scale = top / (largest + table.offset);
    }
    table.rows.resize(table.count * table.row_bytes);
    std::vector<f32_t> shifted(columns);
    for (std::size_t row = 0; row != table.count; ++row) {
        for (std::size_t column = 0; column != columns; ++column)
            shifted[column] = (values[row * columns + column] + table.offset) * table.scale;
        [[maybe_unused]] nk_status_t const cast = nk_cast_serial(
            shifted.data(), nk_f32_k, table.rows.data() + row * table.row_bytes, dtype, columns, nullptr);
    }
    return table;
}

/** Exact softmax attention of one query: the largest logit, the partition against it, and the
 *  attended row. */
struct exact_attention_t {
    f64_t maximum = std::numeric_limits<f64_t>::lowest();
    f64_t partition = 0;
    std::vector<f64_t> output;
};

/** Exact attention of the first @p judged rows of @p queries over every row of @p keys, in @c f64
 *  over the dataset's own scalars, under @p scale and the widened @p values. */
inline std::vector<exact_attention_t> exact_attention(usearch_rows_view_t keys, std::size_t keys_count,
                                                      usearch_rows_view_t queries, std::size_t judged, f64_t scale,
                                                      std::span<f32_t const> values) {
    std::size_t const columns = keys.columns;
    std::vector<f32_t> widened_queries(judged * columns), key(columns);
    for (std::size_t query = 0; query != judged; ++query)
        widen_row(queries, query, {widened_queries.data() + query * columns, columns});
    std::vector<exact_attention_t> exact(judged);
    for (exact_attention_t& query : exact) query.output.assign(columns, 0);
    for (std::size_t row = 0; row != keys_count; ++row) {
        widen_row(keys, row, key);
        for (std::size_t index = 0; index != judged; ++index) {
            f64_t dot = 0;
            for (std::size_t column = 0; column != columns; ++column)
                dot += f64_t(widened_queries[index * columns + column]) * f64_t(key[column]);
            f64_t const logit = dot * scale;
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
                query.output[column] += weight * values[row * columns + column];
        }
    }
    for (exact_attention_t& query : exact)
        for (f64_t& value : query.output) value /= query.partition;
    return exact;
}

/** One pair a judged multiply scored: the key's global row and its dot, packed without padding. */
struct scored_pair_t {
    u64_t row;
    f64_t dot;
};

/** Multiplies every query against @p index, timing the phase and collecting the pairs of the first
 *  @p judged queries into @p pairs. */
inline bool run_multiply(handle<dense_verbs_t>& index, usearch_rows_view_t queries, std::size_t queries_count,
                         algebra_settings_t const& algebra, std::size_t judged,
                         std::vector<std::vector<scored_pair_t>>& pairs) {
    std::size_t const batch = std::max<std::size_t>(1, std::min(algebra.batch.value, queries_count));
    std::size_t scored = 0;
    std::chrono::steady_clock::time_point const started = std::chrono::steady_clock::now();
    for (std::size_t first = 0; first < queries_count; first += batch) {
        std::size_t const count = std::min(batch, queries_count - first);
        usearch_outcome_t outcome {};
        usearch_matches_tape_view_t products {};
        std::size_t written = 0;
        usearch_dense_multiply_t multiply {};
        multiply.index = index.get(), multiply.executor = index.executor(), multiply.outcome = &outcome;
        multiply.count = count, multiply.left = dense_verbs_t::slice(queries, first), multiply.products = &products;
        multiply.written = &written, multiply.arena = index.session->arena;
        multiply.expansion_factor = algebra.expansion_search.value;
        usearch_dense_multiply(&multiply);
        if (report_failure(outcome, "Multiplying")) return false;
        scored += written;
        if (first >= judged) continue;
        // The judged rows' pairs come back off the arena, wherever it lives.
        std::size_t const kept = std::min(count, judged - first);
        std::vector<std::size_t> offsets(kept + 1);
        if (!download(offsets.data(), products.offsets, offsets.size() * sizeof(std::size_t))) return false;
        std::vector<usearch_key_t> keys(offsets[kept]);
        std::vector<usearch_distance_t> dots(offsets[kept]);
        if (!download(keys.data(), products.keys, keys.size() * sizeof(usearch_key_t)) ||
            !download(dots.data(), products.scores, dots.size() * sizeof(usearch_distance_t)))
            return false;
        for (std::size_t row = 0; row != kept; ++row)
            for (std::size_t pair = offsets[row]; pair != offsets[row + 1]; ++pair) {
                std::optional<u64_t> const paired = offset_of_key(index.session->seed, keys[pair]);
                if (!paired) return fmt::println(stderr, "A multiplied pair names a key no row was minted"), false;
                pairs[first + row].push_back({*paired, dots[pair]});
            }
    }
    double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::size_t const nodes = describe(index).size;
    fmt::println("- Multiply: {} rows over {} nodes in {:.2f} s · {}/s · {} pairs per row, {:.3f}% of the nodes",
                 humanized(queries_count), humanized(nodes), seconds, humanized(double(queries_count) / seconds),
                 humanized(double(scored) / double(queries_count)),
                 100.0 * double(scored) / double(queries_count) / double(nodes));
    return true;
}

/** Writes `--pairs`: per judged query a @c u32 count, then that many pairs. */
inline bool save_scored_pairs(std::string_view path, std::span<std::vector<scored_pair_t> const> pairs) {
    std::FILE* file = std::fopen(std::string(path).c_str(), "wb");
    if (!file) return false;
    bool written = true;
    for (std::vector<scored_pair_t> const& query : pairs) {
        u32_t const count = static_cast<u32_t>(query.size());
        written = written && std::fwrite(&count, sizeof(count), 1, file) == 1;
        written = written && std::fwrite(query.data(), sizeof(scored_pair_t), query.size(), file) == query.size();
    }
    return std::fclose(file) == 0 && written;
}

/** Attends every whole token of queries over @p index, judging the first @p judged heads against
 *  exact attention over @p keys_count rows of @p keys. */
inline bool run_attend(handle<dense_verbs_t>& index, usearch_rows_view_t keys, std::size_t keys_count,
                       usearch_rows_view_t queries, std::size_t queries_count, algebra_settings_t const& algebra,
                       std::size_t judged) {
    std::size_t const columns = keys.columns, head_count = algebra.head_count.value;
    std::size_t const rows_count = queries_count / head_count * head_count;
    f64_t const scale = algebra.attend_scale ? algebra.attend_scale : 1 / std::sqrt(double(columns));
    std::vector<f32_t> values;
    if (!widened_values(algebra.values, algebra.first_vector.value, keys_count, columns, values))
        return fmt::println(stderr, "--values=\"{}\" holds no row per base row of {} columns", algebra.values, columns),
               false;
    value_table_t table = value_table_of(values, columns, index.geometry.quantization, algebra.first_vector.value);
    table.seed = index.session->seed;

    std::vector<f32_t> attended(judged * columns);
    std::size_t const batch = std::max<std::size_t>(head_count, algebra.batch.value / head_count * head_count);
    usearch_traversal_stats_t totals {};
    std::chrono::steady_clock::time_point const started = std::chrono::steady_clock::now();
    for (std::size_t first = 0; first < rows_count; first += batch) {
        std::size_t const count = std::min(batch, rows_count - first);
        usearch_outcome_t outcome {};
        usearch_rows_view_t outputs {};
        usearch_traversal_stats_t stats {};
        usearch_dense_attend_t attend {};
        attend.index = index.get(), attend.executor = index.executor(), attend.outcome = &outcome;
        attend.count = count, attend.queries = dense_verbs_t::slice(queries, first), attend.head_count = head_count;
        attend.value_lookup = lookup_values, attend.value_lookup_state = &table, attend.outputs = &outputs;
        attend.arena = index.session->arena, attend.outputs_scalar = usearch_scalar_f32_k;
        attend.expansion_factor = algebra.expansion_search.value, attend.score_scale = static_cast<float>(scale);
        attend.stats = &stats;
        usearch_dense_attend(&attend);
        if (report_failure(outcome, "Attending")) return false;
        totals.computed_distances += stats.computed_distances;
        for (std::size_t row = first; row < std::min(first + count, judged); ++row)
            std::memcpy(attended.data() + row * columns,
                        static_cast<byte_t const*>(outputs.data) + (row - first) * outputs.stride,
                        columns * sizeof(f32_t));
    }
    double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::size_t const tokens = rows_count / head_count;
    fmt::println("- Attend: {} tokens × {} heads in {:.2f} s · {}/s · {} keys per token", humanized(tokens), head_count,
                 seconds, humanized(double(tokens) / seconds),
                 humanized(double(totals.computed_distances) / double(tokens)));

    // Softmax weights sum to one, so an attended row carries the values' shift and scale whole.
    for (f32_t& value : attended) value = value / table.scale - table.offset;
    std::vector<exact_attention_t> exact;
    if (algebra.reference.empty()) exact = exact_attention(keys, keys_count, queries, judged, scale, values);
    else {
        auto [reference, reference_status] = mapped_rows_t::open(algebra.reference);
        usearch_rows_view_t const rows = reference.view(0, judged);
        if (failed(reference_status) || reference.columns() != columns + 2 || !rows.data)
            return fmt::println(stderr, "--reference=\"{}\" holds no {} rows of {} columns", algebra.reference, judged,
                                columns + 2),
                   false;
        for (std::size_t query = 0; query != judged; ++query) {
            f64_t const* const row = reinterpret_cast<f64_t const*>(static_cast<byte_t const*>(rows.data) +
                                                                    query * rows.stride);
            exact.push_back({row[0], row[1], std::vector<f64_t>(row + 2, row + 2 + columns)});
        }
    }
    double error_mean = 0, error_worst = 0;
    for (std::size_t row = 0; row != judged; ++row) {
        double difference = 0, magnitude = 0;
        for (std::size_t column = 0; column != columns; ++column) {
            double const delta = double(attended[row * columns + column]) - exact[row].output[column];
            difference += delta * delta, magnitude += exact[row].output[column] * exact[row].output[column];
        }
        double const error = magnitude > 0 ? std::sqrt(difference / magnitude) : std::sqrt(difference);
        error_mean += error, error_worst = std::max(error_worst, error);
    }
    fmt::println("  relative L2 error over {} heads: {:.2e} mean · {:.2e} worst", judged,
                 judged ? error_mean / double(judged) : 0.0, error_worst);
    return true;
}

/** Runs the dense algebra benchmark over the flags in @p argv. */
int bench_dense_index_algebra(int argc, char const* const* argv) {
    algebra_settings_t algebra;
    std::array<flag_t, 15> const flags = algebra_flags(algebra);
    settings_t const settings = read_settings(argc, argv,
                                              "usearch_bench_dense_index_algebra --base-vectors <path> [flags]", flags);
    print(settings);

    auto [base, base_status] = mapped_rows_t::open(algebra.base_vectors);
    if (failed(base_status) || base.scalar() == usearch_scalar_unknown_k)
        return fmt::println(stderr, "--base-vectors=\"{}\" maps no RetriEval vectors", algebra.base_vectors), 1;
    std::size_t const first_row = algebra.first_vector.value;
    if (first_row >= base.rows())
        return fmt::println(stderr, "--first-vector {} is past the {} rows", first_row, base.rows()), 1;
    std::size_t const rows = algebra.max_base_vectors.value
                                 ? std::min(algebra.max_base_vectors.value, base.rows() - first_row)
                                 : base.rows() - first_row;
    usearch_rows_view_t const vectors = base.view(first_row, rows);
    if (!vectors.data)
        return fmt::println(stderr, "--first-vector {} and --max-base-vectors {} span two parts", first_row, rows), 1;
    usearch_scalar_t stored = base.scalar() == usearch_scalar_f32_k || base.scalar() == usearch_scalar_f64_k
                                  ? usearch_scalar_bf16_k
                                  : base.scalar();
    if (!algebra.stored_dtype_name.empty()) stored = usearch_scalar_from_name(bytes_view(algebra.stored_dtype_name));
    if (stored == usearch_scalar_unknown_k)
        return fmt::println(stderr, "--stored-dtype=\"{}\" names nothing", algebra.stored_dtype_name), 1;

    session_t session = open_session(settings, settings.cpu_class);
    if (!session) return 1;
    usearch_dense_metadata_t const geometry {usearch_dense_metric_inner_k, stored, base.columns(),
                                             algebra.connectivity.value};
    handle<dense_verbs_t> index = open_index<dense_verbs_t>(session, geometry, settings.shards.value);
    if (!index) return 1;
    usearch_traversal_stats_t built {};
    std::chrono::steady_clock::time_point const build_start = std::chrono::steady_clock::now();
    if (report_failure(
            insert_entries(index, vectors, rows, first_row, algebra.batch.value, algebra.expansion_add.value, built),
            "Inserting"))
        return 1;
    fmt::println("- Build: {} rows in {:.2f} s", humanized(rows),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count());
    if (settings.residency == usearch_residency_nvme_k || !settings.checkpoint.empty()) {
        std::string const checkpoint =
            settings.checkpoint.empty()
                ? (std::filesystem::temp_directory_path() / "usearch-bench-algebra.safegraphs").string()
                : std::string(settings.checkpoint);
        index = save_and_reopen(std::move(index), checkpoint, settings.residency);
        if (!index) return 1;
    }
    print_fill(index);

    mapped_rows_t query_file;
    usearch_rows_view_t queries = vectors;
    std::size_t queries_count = std::min<std::size_t>(rows, 1000);
    if (!algebra.query_vectors.empty()) {
        auto [opened, query_status] = mapped_rows_t::open(algebra.query_vectors);
        if (failed(query_status) || opened.columns() != base.columns())
            return fmt::println(stderr, "--query-vectors=\"{}\" maps no queries of {} columns", algebra.query_vectors,
                                base.columns()),
                   1;
        query_file = std::move(opened);
        queries = query_file.view(0, query_file.rows()), queries_count = query_file.rows();
    }
    std::size_t const judged = std::min(algebra.oracle_queries.value, queries_count);

    std::vector<std::vector<scored_pair_t>> pairs(judged);
    if (!run_multiply(index, queries, queries_count, algebra, judged, pairs)) return 1;
    if (!algebra.pairs.empty() && !save_scored_pairs(algebra.pairs, pairs))
        return fmt::println(stderr, "--pairs=\"{}\" could not be written", algebra.pairs), 1;
    if (settings.residency == usearch_residency_vram_k)
        return fmt::println("- Attend: skipped, a VRAM index looks values up on the device"), 0;
    return run_attend(index, vectors, rows, queries, queries_count, algebra, judged) ? 0 : 1;
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_dense_index_algebra(argc, argv); }
