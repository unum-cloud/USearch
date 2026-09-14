/**
 *  @file bench/utilities/persisted_dataset.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief Datasets in the RetriEval binary layout: memory-mapped reads and dense writes.
 *
 *  Each file starts with two little-endian `u32` counters - rows, then columns - followed
 *  by the row-major payload. The scalar type is not stored in the file; it is inferred
 *  from the extension: `.fbin` for f32, `.i8bin` / `.u8bin` / `.bf16bin` accordingly, and
 *  `.ibin` for `u32` neighbor IDs in ground-truth files.
 *
 *  The layout is the big-ANN-benchmarks `*bin` convention of the NeurIPS'21 billion-scale
 *  challenge and the DiskANN ecosystem, as codified by RetriEval - counter width, endianness,
 *  and the extension-to-dtype table pinned down. Benchmark checkpoints, dataset files, and
 *  ground-truth IDs all share this one shape; @ref matrix_save writes it. Reads ride
 *  @ref mapped_file_t, so a dataset costs the pages a run actually touches.
 *
 *  @see https://github.com/ashvardanian/RetriEval
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`
#include <cstring> // `std::memcpy`

#include <string_view> // `std::string_view`
#include <utility>     // `std::move`

#include <numkong/tensor.hpp>
#include <numkong/vector.hpp>
#include <stringzilla/stringzilla.hpp>

#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t`, `copy_bytes` - the files' bytes
#include "usearch/utilities/safegraphs_format.hpp"
#include "usearch/utilities/types.hpp"

namespace unum::usearch {

/** Infers the payload dtype from the RetriEval extension family: `.fbin` → f32,
 *  `.bf16bin` / `.f16bin` / `.i8bin` / `.u8bin` accordingly, `.ibin` → u32 neighbor IDs. */
inline nk_dtype_t dtype_of_extension(sz::string_view path) noexcept {
    if (path.ends_with(".fbin")) return nk_f32_k;
    if (path.ends_with(".i8bin")) return nk_i8_k;
    if (path.ends_with(".u8bin")) return nk_u8_k;
    if (path.ends_with(".bf16bin")) return nk_bf16_k;
    if (path.ends_with(".f16bin")) return nk_f16_k;
    if (path.ends_with(".ibin")) return nk_u32_k;
    return nk_dtype_unknown_k;
}

/** Row and column counts of one matrix file - logical dimensions, sub-byte packing
 *  included. */
struct matrix_extents_t {

    /** Rows of the payload matrix. */
    std::size_t rows = 0;

    /** Logical values per row - what the producing model emitted, not storage words. */
    std::size_t columns = 0;
};

/** Writes @p matrix as two little-endian @p counter_type_ counters - rows, then
 *  columns - and the dense row-major payload; the file extension names the element type. */
template <typename value_type_, typename counter_type_ = u32_t>
status_t matrix_save(std::string_view path, nk::matrix_view<value_type_> matrix) noexcept {
    counter_type_ const counters[2] = {static_cast<counter_type_>(matrix.extent(0)),
                                       static_cast<counter_type_>(matrix.extent(1))};
    if (counters[0] != matrix.extent(0) || counters[1] != matrix.extent(1)) return status_t::overflow_k;
    std::size_t const payload_bytes = matrix.extent(0) * static_cast<std::size_t>(matrix.stride_bytes(0));
    auto [mapping, status] = mapped_file_t::write(path, sizeof(counters) + payload_bytes);
    if (failed(status)) return status;
    bytes_span_t const image = mapping.span();
    if (failed(copy_bytes(image.subspan(0, sizeof(counters)), bytes_of(counters, 2)))) return status_t::io_k;
    if (failed(copy_bytes(image.subspan(sizeof(counters), payload_bytes),
                          bytes_of(matrix.data(), payload_bytes / sizeof(value_type_)))))
        return status_t::io_k;
    return mapping.sync();
}

/** Splits a mapped matrix file into its extents, validating that the payload holds
 *  enough @p scalar_type_ values - sub-byte packing included. The header is two
 *  little-endian @p counter_type_ counters, so both `u32`-headed and `u64`-headed
 *  exports parse with the right instantiation. */
template <typename scalar_type_, typename counter_type_>
expected<matrix_extents_t> parse_matrix_header(bytes_view_t mapping) noexcept {
    matrix_extents_t extents;
    constexpr std::size_t header_bytes_k = 2 * sizeof(counter_type_);
    if (mapping.size() < header_bytes_k) return {extents, status_t::config_mismatch_k};
    counter_type_ rows_header, columns_header;
    std::memcpy(&rows_header, mapping.data(), sizeof(rows_header));
    std::memcpy(&columns_header, mapping.data() + sizeof(rows_header), sizeof(columns_header));

    std::size_t const values_per_row = divide_round_up<std::size_t>(static_cast<std::size_t>(columns_header),
                                                                    nk::dimensions_per_value<scalar_type_>());
    std::size_t const expected_bytes = header_bytes_k +
                                       static_cast<std::size_t>(rows_header) * values_per_row * sizeof(scalar_type_);
    if (mapping.size() < expected_bytes) return {extents, status_t::config_mismatch_k};
    extents.rows = static_cast<std::size_t>(rows_header);
    extents.columns = static_cast<std::size_t>(columns_header);
    return {extents, status_t::success_k};
}

/** Views the payload behind a parsed matrix header as a NumKong matrix. */
template <typename scalar_type_, typename counter_type_>
nk::matrix_view<scalar_type_> view_matrix_payload(bytes_view_t mapping, std::size_t rows,
                                                  std::size_t columns) noexcept {
    if (mapping.empty()) return {};
    return {reinterpret_cast<scalar_type_ const*>(mapping.data() + 2 * sizeof(counter_type_)), rows, columns};
}

/**
 *  @brief The RetriEval dataset triple - base vectors, queries, and ground-truth neighbor
 *      IDs - owning three whole-file mappings & spilling NumKong views, never copying a byte.
 *
 *  Deliberately not a set of @c nk::tensor members with a mapping allocator: tensors own
 *  and value-initialize their storage, while these bytes ARE the files, behind read-only
 *  mappings - the first zero-fill would fault. Queries & neighbors are optional: passing a
 *  null path downgrades the run - self-queries and skipped recall respectively - it never
 *  fails it.
 */
template <typename scalar_type_, typename neighbor_id_type_ = u32_t, typename header_counter_type_ = u32_t>
class persisted_dataset {
    mapped_file_t vectors_mapping_, queries_mapping_, neighbors_mapping_;
    std::size_t vectors_rows_ = 0, queries_rows_ = 0, neighbors_rows_ = 0;
    std::size_t dimensions_ = 0, neighbors_depth_ = 0;

  public:
    using scalar_t = scalar_type_;
    using neighbor_id_t = neighbor_id_type_;
    using header_counter_t = header_counter_type_;

    /** Opens the triple, tolerating empty @p queries_path / @p neighbors_path, and
     *  failing on dimensionality mismatches between vectors & queries, or row-count
     *  mismatches between queries & ground truth. */
    static expected<persisted_dataset> open(std::string_view vectors_path, std::string_view queries_path,
                                            std::string_view neighbors_path) noexcept {
        persisted_dataset dataset;
        auto vectors = mapped_file_t::read(vectors_path);
        if (vectors.status != status_t::success_k) return {std::move(dataset), vectors.status};
        auto const vectors_extents = parse_matrix_header<scalar_type_, header_counter_type_>(vectors.value.view());
        if (vectors_extents.status != status_t::success_k) return {std::move(dataset), vectors_extents.status};
        dataset.vectors_rows_ = vectors_extents.value.rows;
        dataset.dimensions_ = vectors_extents.value.columns;
        dataset.vectors_mapping_ = std::move(vectors.value);

        if (!queries_path.empty()) {
            auto queries = mapped_file_t::read(queries_path);
            if (queries.status != status_t::success_k) return {std::move(dataset), queries.status};
            auto const queries_extents = parse_matrix_header<scalar_type_, header_counter_type_>(queries.value.view());
            if (queries_extents.status != status_t::success_k) return {std::move(dataset), queries_extents.status};
            if (queries_extents.value.columns != dataset.dimensions_)
                return {std::move(dataset), status_t::config_mismatch_k};
            dataset.queries_rows_ = queries_extents.value.rows;
            dataset.queries_mapping_ = std::move(queries.value);
        }
        if (!neighbors_path.empty()) {
            auto neighbors = mapped_file_t::read(neighbors_path);
            if (neighbors.status != status_t::success_k) return {std::move(dataset), neighbors.status};
            auto const neighbors_extents = parse_matrix_header<neighbor_id_type_, header_counter_type_>(
                neighbors.value.view());
            if (neighbors_extents.status != status_t::success_k) return {std::move(dataset), neighbors_extents.status};
            if (dataset.queries_mapping_ && neighbors_extents.value.rows != dataset.queries_rows_)
                return {std::move(dataset), status_t::config_mismatch_k};
            dataset.neighbors_rows_ = neighbors_extents.value.rows;
            dataset.neighbors_depth_ = neighbors_extents.value.columns;
            dataset.neighbors_mapping_ = std::move(neighbors.value);
        }
        return {std::move(dataset), status_t::success_k};
    }

    nk::matrix_view<scalar_type_> vectors() const noexcept {
        return view_matrix_payload<scalar_type_, header_counter_type_>(vectors_mapping_.view(), vectors_rows_,
                                                                       dimensions_);
    }
    nk::matrix_view<scalar_type_> queries() const noexcept {
        return view_matrix_payload<scalar_type_, header_counter_type_>(queries_mapping_.view(), queries_rows_,
                                                                       dimensions_);
    }
    nk::matrix_view<neighbor_id_type_> neighbors() const noexcept {
        return view_matrix_payload<neighbor_id_type_, header_counter_type_>(neighbors_mapping_.view(), neighbors_rows_,
                                                                            neighbors_depth_);
    }

    bool has_queries() const noexcept { return static_cast<bool>(queries_mapping_); }
    bool has_ground_truth() const noexcept { return static_cast<bool>(neighbors_mapping_); }
    std::size_t dimensions() const noexcept { return dimensions_; }
    std::size_t neighbors_depth() const noexcept { return neighbors_depth_; }
    explicit operator bool() const noexcept { return static_cast<bool>(vectors_mapping_); }
};

} // namespace unum::usearch
