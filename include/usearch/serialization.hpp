/**
 *  @brief Reading the SafeGraphs interchange header - the self-describing JSON schema in
 *      front of every `.safegraphs` file - through plain LibC I/O and yyjson.
 *
 *  @author Ash Vardanian
 *  @file serialization.hpp
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
 */
#pragma once

#include <algorithm>   // `std::copy_n`
#include <cstdio>      // `std::FILE`, `std::fread`, `std::fopen`
#include <memory>      // `std::allocator_traits`
#include <string_view> // `std::string_view`

#include <yyjson.h> // The JSON walk - exceptions-free C

#include "types.hpp"

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

/** @brief Maps the specification's dtype spellings onto the USearch @ref dtype_t vocabulary. */
inline dtype_t safegraphs_dtype_of_(std::string_view name) noexcept {
    if (name == "u8") return dtype_t::u8_k;
    if (name == "i8") return dtype_t::i8_k;
    if (name == "u16") return dtype_t::u16_k;
    if (name == "i16") return dtype_t::i16_k;
    if (name == "u32") return dtype_t::u32_k;
    if (name == "i32") return dtype_t::i32_k;
    if (name == "u64") return dtype_t::u64_k;
    if (name == "i64") return dtype_t::i64_k;
    if (name == "f16") return dtype_t::f16_k;
    if (name == "bf16") return dtype_t::bf16_k;
    if (name == "f32") return dtype_t::f32_k;
    if (name == "f64") return dtype_t::f64_k;
    if (name == "e4m3") return dtype_t::e4m3_k;
    if (name == "e5m2") return dtype_t::e5m2_k;
    if (name == "bin16") return dtype_t::bin16_k;
    if (name == "utf8") return dtype_t::utf8_k;
    return dtype_t::unknown_k;
}

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
        column.dtype = safegraphs_dtype_of_(name);
    if (char const* name = yyjson_get_str(yyjson_obj_get(object, "offsets_dtype")))
        column.offsets_dtype = safegraphs_dtype_of_(name);
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

} // namespace usearch
} // namespace unum
