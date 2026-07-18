/**
 *  @brief Read-only memory-mapped datasets in the RetriEval binary layout.
 *
 *  @author Ash Vardanian
 *  @file cpp/persisted_dataset.hpp
 *  @date July 17, 2026
 *
 *  Each file starts with two little-endian `u32` counters - rows, then columns - followed
 *  by the row-major payload. The scalar type is not stored in the file; it is inferred
 *  from the extension: `.fbin` for f32, `.i8bin` / `.u8bin` / `.bf16bin` accordingly, and
 *  `.ibin` for `u32` neighbor ids in ground-truth files.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`
#include <cstring> // `std::memcpy`

#include <utility> // `std::exchange`

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h> // `CreateFileA`, `CreateFileMappingA`, `MapViewOfFile`
#else
#include <fcntl.h>    // `open`
#include <sys/mman.h> // `mmap`, `munmap`
#include <sys/stat.h> // `fstat`
#include <unistd.h>   // `close`
#endif

#include <numkong/tensor.hpp>
#include <numkong/vector.hpp>
#include <stringzilla/stringzilla.hpp>

#include "usearch/types.hpp"

namespace unum::usearch {

namespace nk = ashvardanian::numkong;
namespace sz = ashvardanian::stringzilla;

/** @brief Infers the payload dtype from the RetriEval extension family: `.fbin` → f32,
 *      `.i8bin` / `.u8bin` / `.bf16bin` accordingly, `.ibin` → u32 neighbor ids. */
inline nk_dtype_t dtype_of_extension(sz::string_view path) noexcept {
    if (path.ends_with(".fbin")) return nk_f32_k;
    if (path.ends_with(".i8bin")) return nk_i8_k;
    if (path.ends_with(".u8bin")) return nk_u8_k;
    if (path.ends_with(".bf16bin")) return nk_bf16_k;
    if (path.ends_with(".ibin")) return nk_u32_k;
    return nk_dtype_unknown_k;
}

/**
 *  @brief Read-only whole-file memory mapping, owning the descriptor & the mapping, with
 *      separate POSIX (`mmap`) and Windows (`MapViewOfFile`) implementations picked by macro.
 */
class mmap_view_t {
    char const* data_ = nullptr;
    std::size_t size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int file_descriptor_ = -1;
#endif

  public:
    mmap_view_t() noexcept = default;
    mmap_view_t(mmap_view_t const&) = delete;
    mmap_view_t& operator=(mmap_view_t const&) = delete;

#if defined(_WIN32)
    mmap_view_t(mmap_view_t&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
          file_handle_(std::exchange(other.file_handle_, nullptr)),
          mapping_handle_(std::exchange(other.mapping_handle_, nullptr)) {}
#else
    mmap_view_t(mmap_view_t&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
          file_descriptor_(std::exchange(other.file_descriptor_, -1)) {}
#endif

    mmap_view_t& operator=(mmap_view_t&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
#if defined(_WIN32)
            file_handle_ = std::exchange(other.file_handle_, nullptr);
            mapping_handle_ = std::exchange(other.mapping_handle_, nullptr);
#else
            file_descriptor_ = std::exchange(other.file_descriptor_, -1);
#endif
        }
        return *this;
    }

    ~mmap_view_t() noexcept { reset(); }

    char const* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    explicit operator bool() const noexcept { return data_ != nullptr; }

#if defined(_WIN32)

    void reset() noexcept {
        if (data_) UnmapViewOfFile(data_);
        if (mapping_handle_) CloseHandle(mapping_handle_);
        if (file_handle_) CloseHandle(file_handle_);
        data_ = nullptr, size_ = 0, mapping_handle_ = nullptr, file_handle_ = nullptr;
    }

    /** @brief Maps the whole of @p path read-only, or reports @c bad_alloc_k. */
    static expected<mmap_view_t> try_map(char const* path) noexcept {
        mmap_view_t view;
        view.file_handle_ = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                        FILE_ATTRIBUTE_NORMAL, nullptr);
        if (view.file_handle_ == INVALID_HANDLE_VALUE) {
            view.file_handle_ = nullptr;
            return {std::move(view), status_t::bad_alloc_k};
        }
        LARGE_INTEGER file_bytes;
        if (!GetFileSizeEx(view.file_handle_, &file_bytes) || file_bytes.QuadPart == 0) {
            view.reset();
            return {std::move(view), status_t::bad_alloc_k};
        }
        view.mapping_handle_ = CreateFileMappingA(view.file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!view.mapping_handle_) {
            view.reset();
            return {std::move(view), status_t::bad_alloc_k};
        }
        view.data_ = static_cast<char const*>(MapViewOfFile(view.mapping_handle_, FILE_MAP_READ, 0, 0, 0));
        if (!view.data_) {
            view.reset();
            return {std::move(view), status_t::bad_alloc_k};
        }
        view.size_ = static_cast<std::size_t>(file_bytes.QuadPart);
        return {std::move(view), status_t::success_k};
    }

#else

    void reset() noexcept {
        if (data_) munmap(const_cast<char*>(data_), size_);
        if (file_descriptor_ >= 0) close(file_descriptor_);
        data_ = nullptr, size_ = 0, file_descriptor_ = -1;
    }

    /** @brief Maps the whole of @p path read-only, or reports @c bad_alloc_k. */
    static expected<mmap_view_t> try_map(char const* path) noexcept {
        mmap_view_t view;
        view.file_descriptor_ = open(path, O_RDONLY);
        if (view.file_descriptor_ < 0) return {std::move(view), status_t::bad_alloc_k};

        struct stat file_stats;
        if (fstat(view.file_descriptor_, &file_stats) != 0 || file_stats.st_size == 0) {
            view.reset();
            return {std::move(view), status_t::bad_alloc_k};
        }
        std::size_t const file_bytes = static_cast<std::size_t>(file_stats.st_size);
        void* mapping = mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, view.file_descriptor_, 0);
        if (mapping == MAP_FAILED) {
            view.reset();
            return {std::move(view), status_t::bad_alloc_k};
        }
        view.data_ = static_cast<char const*>(mapping);
        view.size_ = file_bytes;
        return {std::move(view), status_t::success_k};
    }

#endif
};

/** @brief Splits a mapped matrix file into (@p rows, @p columns), validating that the payload
 *      holds enough @p scalar_type_ values - sub-byte packing included. The header is two
 *      little-endian @p counter_type_ counters, so both `u32`-headed RetriEval files and
 *      `u64`-headed billion-scale exports parse with the right instantiation. */
template <typename scalar_type_, typename counter_type_>
status_t parse_matrix_header(mmap_view_t const& mapping, std::size_t& rows, std::size_t& columns) noexcept {
    constexpr std::size_t header_bytes_k = 2 * sizeof(counter_type_);
    if (mapping.size() < header_bytes_k) return status_t::config_mismatch_k;
    counter_type_ rows_header, columns_header;
    std::memcpy(&rows_header, mapping.data(), sizeof(rows_header));
    std::memcpy(&columns_header, mapping.data() + sizeof(rows_header), sizeof(columns_header));

    std::size_t const values_per_row = divide_round_up<std::size_t>(static_cast<std::size_t>(columns_header),
                                                                    nk::dimensions_per_value<scalar_type_>());
    std::size_t const expected_bytes = header_bytes_k +
                                       static_cast<std::size_t>(rows_header) * values_per_row * sizeof(scalar_type_);
    if (mapping.size() < expected_bytes) return status_t::config_mismatch_k;
    rows = static_cast<std::size_t>(rows_header), columns = static_cast<std::size_t>(columns_header);
    return status_t::success_k;
}

/** @brief Views the payload behind a parsed matrix header as a NumKong matrix. */
template <typename scalar_type_, typename counter_type_>
nk::matrix_view<scalar_type_> view_matrix_payload(mmap_view_t const& mapping, std::size_t rows,
                                                  std::size_t columns) noexcept {
    if (!mapping) return {};
    return {reinterpret_cast<scalar_type_ const*>(mapping.data() + 2 * sizeof(counter_type_)), rows, columns};
}

/**
 *  @brief The RetriEval dataset triple - base vectors, queries, and ground-truth neighbor
 *      ids - owning three whole-file mappings & spilling NumKong views, never copying a byte.
 *
 *  Deliberately not a set of @c nk::tensor members with a mapping allocator: tensors own
 *  and value-initialize their storage, while these bytes ARE the files, behind read-only
 *  mappings - the first zero-fill would fault. Queries & neighbors are optional: passing a
 *  null path downgrades the run - self-queries and skipped recall respectively - it never
 *  fails it.
 */
template <typename scalar_type_, typename neighbor_id_type_ = std::uint32_t,
          typename header_counter_type_ = std::uint32_t>
class persisted_dataset {
    mmap_view_t vectors_mapping_, queries_mapping_, neighbors_mapping_;
    std::size_t vectors_rows_ = 0, queries_rows_ = 0, neighbors_rows_ = 0;
    std::size_t dimensions_ = 0, neighbors_depth_ = 0;

  public:
    using scalar_t = scalar_type_;
    using neighbor_id_t = neighbor_id_type_;
    using header_counter_t = header_counter_type_;

    /** @brief Opens the triple, tolerating null @p queries_path / @p neighbors_path, and
     *      failing on dimensionality mismatches between vectors & queries, or row-count
     *      mismatches between queries & ground truth. */
    static expected<persisted_dataset> try_open(char const* vectors_path, char const* queries_path,
                                                char const* neighbors_path) noexcept {
        persisted_dataset dataset;
        auto vectors = mmap_view_t::try_map(vectors_path);
        if (vectors.status != status_t::success_k) return {std::move(dataset), vectors.status};
        status_t parsed = parse_matrix_header<scalar_type_, header_counter_type_>(vectors.value, dataset.vectors_rows_,
                                                                                  dataset.dimensions_);
        if (parsed != status_t::success_k) return {std::move(dataset), parsed};
        dataset.vectors_mapping_ = std::move(vectors.value);

        if (queries_path) {
            auto queries = mmap_view_t::try_map(queries_path);
            if (queries.status != status_t::success_k) return {std::move(dataset), queries.status};
            std::size_t queries_columns = 0;
            parsed = parse_matrix_header<scalar_type_, header_counter_type_>(queries.value, dataset.queries_rows_,
                                                                             queries_columns);
            if (parsed != status_t::success_k) return {std::move(dataset), parsed};
            if (queries_columns != dataset.dimensions_) return {std::move(dataset), status_t::config_mismatch_k};
            dataset.queries_mapping_ = std::move(queries.value);
        }
        if (neighbors_path) {
            auto neighbors = mmap_view_t::try_map(neighbors_path);
            if (neighbors.status != status_t::success_k) return {std::move(dataset), neighbors.status};
            parsed = parse_matrix_header<neighbor_id_type_, header_counter_type_>(
                neighbors.value, dataset.neighbors_rows_, dataset.neighbors_depth_);
            if (parsed != status_t::success_k) return {std::move(dataset), parsed};
            if (dataset.queries_mapping_ && dataset.neighbors_rows_ != dataset.queries_rows_)
                return {std::move(dataset), status_t::config_mismatch_k};
            dataset.neighbors_mapping_ = std::move(neighbors.value);
        }
        return {std::move(dataset), status_t::success_k};
    }

    nk::matrix_view<scalar_type_> vectors() const noexcept {
        return view_matrix_payload<scalar_type_, header_counter_type_>(vectors_mapping_, vectors_rows_, dimensions_);
    }
    nk::matrix_view<scalar_type_> queries() const noexcept {
        return view_matrix_payload<scalar_type_, header_counter_type_>(queries_mapping_, queries_rows_, dimensions_);
    }
    nk::matrix_view<neighbor_id_type_> neighbors() const noexcept {
        return view_matrix_payload<neighbor_id_type_, header_counter_type_>(neighbors_mapping_, neighbors_rows_,
                                                                            neighbors_depth_);
    }

    bool has_queries() const noexcept { return static_cast<bool>(queries_mapping_); }
    bool has_ground_truth() const noexcept { return static_cast<bool>(neighbors_mapping_); }
    std::size_t dimensions() const noexcept { return dimensions_; }
    std::size_t neighbors_depth() const noexcept { return neighbors_depth_; }
    explicit operator bool() const noexcept { return static_cast<bool>(vectors_mapping_); }
};

} // namespace unum::usearch
