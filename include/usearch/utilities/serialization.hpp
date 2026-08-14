/**
 *  @brief Reading the SafeGraphs interchange header - the self-describing JSON schema in
 *      front of every `.safegraphs` file - through plain LibC I/O and yyjson.
 *
 *  @author Ash Vardanian
 *  @file include/usearch/utilities/serialization.hpp
 *  @date July 19, 2026
 *
 *  Every backend serializes its payloads differently - volatile in-RAM snapshots, GPU-resident
 *  uploads, persisted on-NVMe layouts - but they all sniff the same header, so only the reader
 *  lives here. @ref safegraphs_read_header consumes the magic, the header size, and the JSON
 *  bytes - nothing is memory-mapped and no payload byte is ever touched. Columns come back as
 *  byte ranges relative to the data segment, and @c safegraphs_header_t::data_offset says
 *  where that segment begins, so the caller picks its own I/O strategy per section:
 *  @c std::fread, @c mmap, @c io_uring, or a GPU DMA engine.
 *
 *  The JSON itself goes through yyjson - exceptions-free C, no schema compilation - and every
 *  allocation flows through the caller's allocator, trampolined into @c yyjson_alc.
 *
 *  @see https://github.com/safegraphs/SafeGraphs
 *
 *  Beside the SafeGraphs reader live the RetriEval-layout matrix helpers - @ref matrix_save,
 *  @ref matrix_extents & @ref matrix_load: two little-endian counters, then the dense
 *  row-major payload, with the element type carried by the file extension. The layout is the
 *  big-ANN-benchmarks `*bin` convention of the NeurIPS'21 billion-scale challenge and the
 *  DiskANN ecosystem, as codified by RetriEval - counter width, endianness, and the
 *  extension-to-dtype table pinned down. Benchmark checkpoints, dataset files, and
 *  ground-truth ids all share this one shape.
 *
 *  @see https://github.com/ashvardanian/RetriEval
 */
#pragma once

#include <algorithm>   // `std::copy_n`
#include <cstdio>      // `std::FILE`, `std::fread`, `std::fopen`
#include <memory>      // `std::allocator_traits`
#include <string_view> // `std::string_view`

#include <yyjson.h> // The JSON walk - exceptions-free C

#include <numkong/tensor.hpp> // `nk::matrix_view`, `nk::matrix_span`

#include "usearch/utilities/types.hpp"

namespace unum {
namespace usearch {

/** @brief One column of a SafeGraphs header: the element type, the fixed row width, and its
 *      buffers as byte ranges @b relative to the data segment - never resolved pointers, so
 *      no memory-mapped representation is assumed anywhere. */
struct safegraphs_column_t {
    /** @brief Element type of the values buffer. */
    dtype_t dtype = dtype_t::unknown_k;
    /** @brief Prefix-sum element type; @c unknown_k for regular columns. */
    dtype_t offsets_dtype = dtype_t::unknown_k;
    /** @brief Values per row of a regular column; 1 when omitted, meaningless when irregular. */
    std::size_t shape = 1;
    /** @brief Byte position of the values buffer within the data segment. */
    std::size_t values_start = 0;
    /** @brief Byte size of the values buffer. */
    std::size_t values_length = 0;
    /** @brief Byte position of the prefix-sum buffer; zero-length means a regular column. */
    std::size_t offsets_start = 0;
    /** @brief Byte size of the prefix-sum buffer - @b nodes+1 prefix sums when present. */
    std::size_t offsets_length = 0;

    /** @brief True for an Arrow @c FixedSizeList layout - one dense values buffer, row @b i
     *      at @c values[i*shape..(i+1)*shape] - as opposed to the offsets-driven lists. */
    bool is_regular() const noexcept { return offsets_length == 0; }
};

/** @brief A parsed SafeGraphs monolith header: the dataset metadata, the three canonical
 *      columns, and where the data segment begins in the file. */
struct safegraphs_header_t {
    /** @brief The expected first eight file bytes. */
    inline static constexpr char magic_k[8] = {'S', 'A', 'F', 'E', 'G', 'R', 'P', 'H'};

    /** @brief Node count from the metadata; zero when the producer omitted it. */
    std::size_t nodes = 0;
    /** @brief Buffer alignment in bytes the producer declared; 4096 by default. */
    std::size_t alignment = 4096;
    /** @brief Absolute file offset of the data segment every column range is relative to. */
    std::size_t data_offset = 0;
    /** @brief Node identity column. */
    safegraphs_column_t key;
    /** @brief Payload column - the vectors, strings, or feature sets. */
    safegraphs_column_t value;
    /** @brief Adjacency column of node indices; out-of-range ids are positional sentinels. */
    safegraphs_column_t neighbors;
};

/** @brief Trampolines a standard byte allocator into @c yyjson_alc's C ABI. Standard
 *      allocators demand the size back on deallocation while yyjson's free callback supplies
 *      only the pointer, so every block carries its size in a max-aligned prefix. */
template <typename allocator_type_>
struct safegraphs_yyjson_allocator_ {
    using byte_allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<std::byte>;
    inline static constexpr std::size_t prefix_bytes_k = alignof(std::max_align_t);

    byte_allocator_t allocator;

    static void* allocate_(void* context, std::size_t size) noexcept {
        auto& self = *static_cast<safegraphs_yyjson_allocator_*>(context);
        std::byte* block = self.allocator.allocate(prefix_bytes_k + size);
        if (!block) return nullptr;
        *reinterpret_cast<std::size_t*>(block) = size;
        return block + prefix_bytes_k;
    }
    static void deallocate_(void* context, void* pointer) noexcept {
        if (!pointer) return;
        auto& self = *static_cast<safegraphs_yyjson_allocator_*>(context);
        std::byte* block = static_cast<std::byte*>(pointer) - prefix_bytes_k;
        self.allocator.deallocate(block, prefix_bytes_k + *reinterpret_cast<std::size_t*>(block));
    }
    static void* reallocate_(void* context, void* pointer, std::size_t old_size, std::size_t size) noexcept {
        void* grown = allocate_(context, size);
        if (!grown || !pointer) return grown;
        std::copy_n(static_cast<std::byte const*>(pointer), min_of(old_size, size), static_cast<std::byte*>(grown));
        deallocate_(context, pointer);
        return grown;
    }

    /** @brief The C-ABI view yyjson consumes; valid while this trampoline stays alive. */
    yyjson_alc c_abi() noexcept { return {&allocate_, &reallocate_, &deallocate_, this}; }
};

/** @brief Walks one column object, filling the recognized scalar fields & ignoring extras. */
inline void safegraphs_parse_column_(yyjson_val* object, safegraphs_column_t& column) noexcept {
    if (!yyjson_is_obj(object)) return;
    if (char const* name = yyjson_get_str(yyjson_obj_get(object, "dtype"))) //
        column.dtype = dtype_named(name);
    if (char const* name = yyjson_get_str(yyjson_obj_get(object, "offsets_dtype")))
        column.offsets_dtype = dtype_named(name);
    if (yyjson_val* shape = yyjson_obj_get(object, "shape")) column.shape = yyjson_get_uint(shape);
    column.values_start = yyjson_get_uint(yyjson_obj_get(object, "values_start"));
    column.values_length = yyjson_get_uint(yyjson_obj_get(object, "values_length"));
    column.offsets_start = yyjson_get_uint(yyjson_obj_get(object, "offsets_start"));
    column.offsets_length = yyjson_get_uint(yyjson_obj_get(object, "offsets_length"));
}

/**
 *  @brief Reads & parses the SafeGraphs header of @p file - the magic, the header size, and
 *      the JSON schema - touching no payload bytes; the stream must be positioned at the
 *      file's start.
 *
 *  A free-standing reader with no I/O opinions: the returned columns are byte ranges
 *  relative to the data segment at @c data_offset, and how those bytes are then fetched -
 *  @c std::fread, memory mapping, @c io_uring, or a DMA engine - stays the caller's choice.
 *  Every allocation - the JSON bytes and yyjson's own document - flows through @p allocator.
 *
 *  @param[in] file An open, readable stream positioned at the file's start.
 *  @param[in] allocator Backs the JSON buffer & the yyjson document; any byte allocator.
 *  @return The parsed header, or @c malformed_k / @c bad_alloc_k explaining the failure.
 */
template <typename allocator_type_ = std::allocator<std::byte>>
expected<safegraphs_header_t> safegraphs_read_header(std::FILE* file, allocator_type_ allocator = {}) noexcept {
    safegraphs_header_t header;
    char prefix[16];
    if (std::fread(prefix, 1, sizeof(prefix), file) != sizeof(prefix)) return {header, status_t::malformed_k};
    for (std::size_t index = 0; index != sizeof(header.magic_k); ++index)
        if (prefix[index] != safegraphs_header_t::magic_k[index]) return {header, status_t::malformed_k};

    std::size_t header_size = 0;
    // The header size arrives little-endian; assembling bytes keeps the read host-agnostic.
    for (std::size_t byte = 0; byte != 8; ++byte)
        header_size |= std::size_t(static_cast<unsigned char>(prefix[8 + byte])) << (byte * 8);

    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    yyjson_alc alc = trampoline.c_abi();
    char* json = static_cast<char*>(safegraphs_yyjson_allocator_<allocator_type_>::allocate_(&trampoline, //
                                                                                             header_size));
    if (!json) return {header, status_t::bad_alloc_k};
    bool const read_whole = std::fread(json, 1, header_size, file) == header_size;
    yyjson_doc* document = read_whole ? yyjson_read_opts(json, header_size, 0, &alc, nullptr) : nullptr;
    if (!document) {
        safegraphs_yyjson_allocator_<allocator_type_>::deallocate_(&trampoline, json);
        return {header, status_t::malformed_k};
    }

    yyjson_val* root = yyjson_doc_get_root(document);
    if (yyjson_val* metadata = yyjson_obj_get(root, "__metadata__"); yyjson_is_obj(metadata)) {
        if (yyjson_val* nodes = yyjson_obj_get(metadata, "nodes")) header.nodes = yyjson_get_uint(nodes);
        if (yyjson_val* alignment = yyjson_obj_get(metadata, "alignment"))
            header.alignment = yyjson_get_uint(alignment);
    }
    safegraphs_parse_column_(yyjson_obj_get(root, "key"), header.key);
    safegraphs_parse_column_(yyjson_obj_get(root, "value"), header.value);
    safegraphs_parse_column_(yyjson_obj_get(root, "neighbors"), header.neighbors);
    bool const object_root = yyjson_is_obj(root);
    yyjson_doc_free(document);
    safegraphs_yyjson_allocator_<allocator_type_>::deallocate_(&trampoline, json);
    if (!object_root) return {header, status_t::malformed_k};

    // The data segment starts on the declared alignment boundary at or past the JSON's end.
    if (!header.alignment || (header.alignment & (header.alignment - 1)) != 0) return {header, status_t::malformed_k};
    std::size_t const header_end = sizeof(prefix) + header_size;
    header.data_offset = divide_round_up<std::size_t>(header_end, header.alignment) * header.alignment;
    return {header, status_t::success_k};
}

/** @brief The path-taking overload: opens @p path read-only, parses its header through the
 *      stream overload, and closes it - @c device_error_k when the file cannot open. */
template <typename allocator_type_ = std::allocator<std::byte>>
expected<safegraphs_header_t> safegraphs_read_header(char const* path, allocator_type_ allocator = {}) noexcept {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return {safegraphs_header_t {}, status_t::device_error_k};
    expected<safegraphs_header_t> const result = safegraphs_read_header(file, allocator);
    std::fclose(file);
    return result;
}

/** @brief Row & column counts of one RetriEval-layout matrix file - logical dimensions,
 *      exactly what the tensors' own extents speak, sub-byte packing included. */
struct matrix_extents_t {
    /** @brief Rows of the payload matrix. */
    std::size_t rows = 0;
    /** @brief Logical values per row - what the producing model emitted, not storage words. */
    std::size_t columns = 0;
};

/**
 *  @brief Writes @p matrix as one RetriEval-layout file at @p path: two little-endian
 *      @p counter_type_ counters - rows, then columns - and the dense row-major payload.
 *      The element type is never stored - the file extension carries it, RetriEval-style.
 *      Extents are logical dimensions; the tensor's own strides account for sub-byte packing.
 *
 *  @param[in] path Destination file, truncated & rewritten whole.
 *  @param[in] matrix The payload; must be contiguous.
 *  @return @c success_k, @c overflow_k when an extent exceeds the counter range, or
 *      @c device_error_k on any I/O failure.
 */
template <typename value_type_, typename counter_type_ = u32_t>
status_t matrix_save(char const* path, nk::matrix_view<value_type_> matrix) noexcept {
    counter_type_ const counters[2] = {static_cast<counter_type_>(matrix.extent(0)),
                                       static_cast<counter_type_>(matrix.extent(1))};
    if (counters[0] != matrix.extent(0) || counters[1] != matrix.extent(1)) return status_t::overflow_k;
    std::size_t const payload_bytes = matrix.extent(0) * static_cast<std::size_t>(matrix.stride_bytes(0));
    std::FILE* file = std::fopen(path, "wb");
    if (!file) return status_t::device_error_k;
    bool written = std::fwrite(counters, sizeof(counter_type_), 2, file) == 2;
    written = written && std::fwrite(matrix.data(), 1, payload_bytes, file) == payload_bytes;
    bool const closed = std::fclose(file) == 0;
    return written && closed ? status_t::success_k : status_t::device_error_k;
}

/** @brief Reads just the two-counter header of @p path - the caller sizes its destination
 *      from the returned extents before fetching the payload its own way. */
template <typename counter_type_ = u32_t>
expected<matrix_extents_t> matrix_extents(char const* path) noexcept {
    matrix_extents_t extents;
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return {extents, status_t::device_error_k};
    counter_type_ counters[2];
    bool const read_whole = std::fread(counters, sizeof(counter_type_), 2, file) == 2;
    std::fclose(file);
    if (!read_whole) return {extents, status_t::malformed_k};
    extents.rows = static_cast<std::size_t>(counters[0]);
    extents.columns = static_cast<std::size_t>(counters[1]);
    return {extents, status_t::success_k};
}

/**
 *  @brief Fills the caller-shaped @p destination from the payload of @p path, after checking
 *      the file's counters against the expected extents.
 *
 *  @param[in] path A RetriEval-layout matrix file.
 *  @param[out] destination Receives the payload; must be contiguous.
 *  @return @c success_k, @c device_error_k when the file cannot open, @c malformed_k on a
 *      short read, or @c config_mismatch_k when the counters disagree with @p destination.
 */
template <typename value_type_, typename counter_type_ = u32_t>
status_t matrix_load(char const* path, nk::matrix_span<value_type_> destination) noexcept {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return status_t::device_error_k;
    counter_type_ counters[2];
    if (std::fread(counters, sizeof(counter_type_), 2, file) != 2) return std::fclose(file), status_t::malformed_k;
    if (counters[0] != destination.extent(0) || counters[1] != destination.extent(1))
        return std::fclose(file), status_t::config_mismatch_k;
    std::size_t const payload_bytes = destination.extent(0) * static_cast<std::size_t>(destination.stride_bytes(0));
    bool const read_whole = std::fread(destination.data(), 1, payload_bytes, file) == payload_bytes;
    std::fclose(file);
    return read_whole ? status_t::success_k : status_t::malformed_k;
}

} // namespace usearch
} // namespace unum
