/**
 *  @brief The SafeGraphs interchange - the self-describing JSON header in front of every
 *      `.safegraphs` file and the manifest beside it - read & written through the owned
 *      @ref file_t and yyjson.
 *
 *  @author Ash Vardanian
 *  @file include/usearch/utilities/serialization.hpp
 *  @date July 19, 2026
 *
 *  Every backend serializes its payloads differently - volatile in-RAM snapshots, GPU-resident
 *  uploads, persisted on-NVMe layouts - but they all speak the same header, so its reader and
 *  writer live here, payload-free. @ref safegraphs_read_header consumes the magic, the header
 *  size, and the JSON bytes - nothing is memory-mapped. Columns come back as byte ranges
 *  relative to the data segment, and @c safegraphs_header_t::data_offset says where that
 *  segment begins, so the caller picks its own I/O strategy per section: @c std::fread,
 *  @c mmap, @c io_uring, or a GPU DMA engine.
 *
 *  The JSON itself goes through yyjson - exceptions-free C, no schema compilation - and every
 *  allocation flows through the caller's allocator, trampolined into @c yyjson_alc.
 *
 *  @see https://github.com/safegraphs/SafeGraphs
 */
#pragma once

#include <cstddef> // `std::byte`
#include <cstdio>  // `std::FILE`, `std::fread`, `std::fopen`, `std::snprintf`
#include <cstring> // `std::memcpy`, `std::memset`

#include <algorithm>   // `std::copy_n`
#include <array>       // `std::array` - the prefix magic
#include <bit>         // `std::endian`, `std::has_single_bit`
#include <concepts>    // `std::unsigned_integral`
#include <memory>      // `std::allocator_traits`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <utility>     // `std::exchange`

#include <yyjson.h> // The JSON reader & writer - exceptions-free C

#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - shard path formatting
#include "usearch/utilities/types.hpp"

namespace unum {
namespace usearch {

/** @brief How @ref file_t::open opens a file: reading an existing one, or truncating a fresh one. */
enum class file_mode_t { read_k, write_k };

/** @brief An owned LibC stream; every verb answers @c status_t, and no raw @c std::FILE*
 *      crosses this boundary. */
struct file_t {

    file_t() = default;
    file_t(file_t&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}
    file_t& operator=(file_t&& other) noexcept { return std::swap(stream_, other.stream_), *this; }
    file_t(file_t const&) = delete;
    file_t& operator=(file_t const&) = delete;
    ~file_t() noexcept {
        if (stream_) std::fclose(stream_);
    }

    /** @brief Truthy exactly while a file is open. */
    explicit operator bool() const noexcept { return stream_ != nullptr; }

    /** @brief Opens @p path; the NUL terminator LibC demands is one transient allocation
     *      through @p allocator. */
    template <typename allocator_type_ = std::allocator<std::byte>>
    static expected<file_t> open(std::string_view path, file_mode_t mode, allocator_type_ allocator = {}) noexcept {
        using chars_allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<char>;
        chars_allocator_t chars {allocator};
        char* terminated = chars.allocate(path.size() + 1);
        if (!terminated) return {file_t {}, status_t::bad_alloc_k};
        std::memcpy(terminated, path.data(), path.size());
        terminated[path.size()] = '\0';
        file_t file;
        file.stream_ = std::fopen(terminated, mode == file_mode_t::read_k ? "rb" : "wb");
        chars.deallocate(terminated, path.size() + 1);
        status_t const opened = file ? status_t::success_k : status_t::io_k;
        return {std::move(file), opened};
    }

    /** @brief An anonymous read-write file the OS unlinks on close. */
    static expected<file_t> temporary() noexcept {
        file_t file;
        file.stream_ = std::tmpfile();
        status_t const opened = file ? status_t::success_k : status_t::io_k;
        return {std::move(file), opened};
    }

    /** @brief Positions the cursor at @p offset through the 64-bit seek; @c std::fseek takes
     *      a @c long. */
    status_t seek(std::size_t offset) noexcept {
#if defined(_WIN32)
        int const outcome = _fseeki64(stream_, static_cast<long long>(offset), SEEK_SET);
#else
        int const outcome = fseeko(stream_, static_cast<off_t>(offset), SEEK_SET);
#endif
        return outcome == 0 ? status_t::success_k : status_t::io_k;
    }

    /** @brief Reads exactly @p length bytes into @p destination; a short read answers @c io_k. */
    status_t read(void* destination, std::size_t length) noexcept {
        return std::fread(destination, 1, length, stream_) == length ? status_t::success_k : status_t::io_k;
    }

    /** @brief Writes exactly @p length bytes from @p source; a short write answers @c io_k. */
    status_t write(void const* source, std::size_t length) noexcept {
        return std::fwrite(source, 1, length, stream_) == length ? status_t::success_k : status_t::io_k;
    }

    /** @brief The span sibling of the byte-count read. */
    status_t read(bytes_span_t destination) noexcept { return read(destination.data(), destination.size()); }

    /** @brief The span sibling of the byte-count write. */
    status_t write(bytes_view_t source) noexcept { return write(source.data(), source.size()); }

    /** @brief Writes @p length copies of @p value - what pads a header out to its alignment. */
    status_t fill(char value, std::size_t length) noexcept {
        char batch[default_alignment_k];
        std::memset(batch, value, sizeof(batch));
        for (; length > sizeof(batch); length -= sizeof(batch))
            if (failed(write(batch, sizeof(batch)))) return status_t::io_k;
        return write(batch, length);
    }

    /** @brief Flushes and closes, surfacing the buffered failure the destructor would swallow. */
    status_t close() noexcept {
        if (!stream_) return status_t::success_k;
        return std::fclose(std::exchange(stream_, nullptr)) == 0 ? status_t::success_k : status_t::io_k;
    }

    /** @brief Byte length of the file, measured by seeking to the end and rewinding. */
    expected<std::size_t> length() noexcept {
#if defined(_WIN32)
        int const sought = _fseeki64(stream_, 0, SEEK_END);
        long long const told = _ftelli64(stream_);
#else
        int const sought = fseeko(stream_, 0, SEEK_END);
        off_t const told = ftello(stream_);
#endif
        if (sought != 0 || told < 0 || failed(seek(0))) return {0, status_t::io_k};
        return {static_cast<std::size_t>(told), status_t::success_k};
    }

  private:
    /** @brief The wrapped LibC stream; null marks the empty state a move-out leaves behind. */
    std::FILE* stream_ = nullptr;
};

/** @brief One header column: element type, row width, and buffers as byte ranges relative to
 *      the data segment. */
struct safegraphs_column_t {
    /** @brief Element type of the values buffer. */
    dtype_t dtype = dtype_t::unknown_k;
    /** @brief Prefix-sum element type; @c unknown_k for regular columns. */
    dtype_t offsets_dtype = dtype_t::unknown_k;
    /** @brief Stored values per row of a regular column; 1 when omitted. */
    std::size_t shape = 1;
    /** @brief Logical values per row; equals @c shape unless the stored stride pads or packs. */
    std::size_t shape_effective = 1;
    /** @brief Byte position of the values buffer within the data segment. */
    std::size_t values_start = 0;
    /** @brief Byte size of the values buffer. */
    std::size_t values_length = 0;
    /** @brief Byte position of the prefix-sum buffer; zero-length means a regular column. */
    std::size_t offsets_start = 0;
    /** @brief Byte size of the prefix-sum buffer - @b nodes+1 prefix sums when present. */
    std::size_t offsets_length = 0;

    /** @brief True for a @c FixedSizeList layout, false for an offsets-driven list. */
    bool is_regular() const noexcept { return offsets_length == 0; }
};

/** @brief A parsed monolith header: the metadata, the three columns, and where the data
 *      segment begins. */
struct safegraphs_header_t {
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
    /** @brief Adjacency column of node indices; out-of-range IDs are positional sentinels. */
    safegraphs_column_t neighbors;
};

/** @brief The fixed 16 bytes every monolith opens with: the magic, then the little-endian
 *      byte length of the JSON header. */
struct safegraphs_prefix_t {
    inline static constexpr std::array<char, 8> magic_k = {'S', 'A', 'F', 'E', 'G', 'R', 'P', 'H'};
    std::array<char, 8> magic = magic_k;
    u64_t header_length = 0;
};
static_assert(sizeof(safegraphs_prefix_t) == 16 && std::endian::native == std::endian::little,
              "The prefix is read and written as raw little-endian bytes");

/** @brief One custom `__metadata__` entry: a producer's own key and its string value. */
struct safegraphs_custom_field_t {
    std::string_view key;
    std::string_view value;
};

/** @brief Trampolines a byte allocator into @c yyjson_alc; each block carries its size in a
 *      max-aligned prefix, since yyjson frees by pointer alone. */
template <typename allocator_type_>
struct safegraphs_yyjson_allocator_ {
    using byte_allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<std::byte>;
    inline static constexpr std::size_t prefix_bytes_k = alignof(std::max_align_t);

    byte_allocator_t allocator;

    static void* allocate_(void* context, std::size_t size) noexcept {
        auto& self = *static_cast<safegraphs_yyjson_allocator_*>(context);
        std::byte* block = self.allocator.allocate(prefix_bytes_k + size);
        if (!block) return nullptr;
        std::memcpy(block, &size, sizeof(size));
        return block + prefix_bytes_k;
    }
    static void deallocate_(void* context, void* pointer) noexcept {
        if (!pointer) return;
        auto& self = *static_cast<safegraphs_yyjson_allocator_*>(context);
        std::byte* block = static_cast<std::byte*>(pointer) - prefix_bytes_k;
        std::size_t size = 0;
        std::memcpy(&size, block, sizeof(size));
        self.allocator.deallocate(block, prefix_bytes_k + size);
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

/** @brief The text of a string value; empty for anything else. */
inline std::string_view safegraphs_string_(yyjson_val* value) noexcept {
    return yyjson_is_str(value) ? std::string_view {yyjson_get_str(value), yyjson_get_len(value)} : std::string_view {};
}

/** @brief Reads @p length bytes of @p file and parses them into an object-rooted document the
 *      caller frees; the text itself is released here. */
template <typename allocator_type_>
expected<yyjson_doc*> safegraphs_read_document_(file_t& file, std::size_t length,
                                                safegraphs_yyjson_allocator_<allocator_type_>& trampoline) noexcept {
    yyjson_alc allocator_abi = trampoline.c_abi();
    char* text = static_cast<char*>(allocator_abi.malloc(allocator_abi.ctx, length));
    if (!text) return {nullptr, status_t::bad_alloc_k};
    yyjson_doc* document = succeeded(file.read(text, length)) //
                               ? yyjson_read_opts(text, length, 0, &allocator_abi, nullptr)
                               : nullptr;
    allocator_abi.free(allocator_abi.ctx, text);
    if (document && yyjson_is_obj(yyjson_doc_get_root(document))) return {document, status_t::success_k};
    yyjson_doc_free(document);
    return {nullptr, status_t::malformed_k};
}

/** @brief Walks one column object, filling the recognized fields and ignoring extras. */
inline void safegraphs_parse_column_(yyjson_val* object, safegraphs_column_t& column) noexcept {
    if (!yyjson_is_obj(object)) return;
    column.dtype = dtype_named(safegraphs_string_(yyjson_obj_get(object, "dtype")));
    column.offsets_dtype = dtype_named(safegraphs_string_(yyjson_obj_get(object, "offsets_dtype")));
    if (yyjson_val* shape = yyjson_obj_get(object, "shape")) column.shape = yyjson_get_uint(shape);
    yyjson_val* effective = yyjson_obj_get(object, "shape_effective");
    column.shape_effective = effective ? yyjson_get_uint(effective) : column.shape;
    column.values_start = yyjson_get_uint(yyjson_obj_get(object, "values_start"));
    column.values_length = yyjson_get_uint(yyjson_obj_get(object, "values_length"));
    column.offsets_start = yyjson_get_uint(yyjson_obj_get(object, "offsets_start"));
    column.offsets_length = yyjson_get_uint(yyjson_obj_get(object, "offsets_length"));
}

/**
 *  @brief Reads the prefix and parses the header JSON, touching no payload; the stream must
 *      sit at the file's start.
 *  @param[in] file An open, readable file positioned at its start.
 *  @param[in] allocator Backs the JSON text and the yyjson document.
 *  @param[in] per_custom_field Fires once per string-valued `__metadata__` entry; its views
 *      die with this call.
 */
template <typename allocator_type_ = std::allocator<std::byte>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_header_t> safegraphs_read_header(file_t& file, allocator_type_ allocator = {},
                                                     callback_type_&& per_custom_field = {}) noexcept {
    safegraphs_header_t header;
    safegraphs_prefix_t prefix;
    if (failed(file.read(&prefix, sizeof(prefix))) || prefix.magic != safegraphs_prefix_t::magic_k)
        return {header, status_t::malformed_k};
    std::size_t const header_length = static_cast<std::size_t>(prefix.header_length);
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    auto [document, parsed] = safegraphs_read_document_(file, header_length, trampoline);
    if (failed(parsed)) return {header, parsed};

    yyjson_val* root = yyjson_doc_get_root(document);
    if (yyjson_val* metadata = yyjson_obj_get(root, "__metadata__"); yyjson_is_obj(metadata)) {
        header.nodes = yyjson_get_uint(yyjson_obj_get(metadata, "nodes"));
        if (yyjson_val* alignment = yyjson_obj_get(metadata, "alignment"))
            header.alignment = yyjson_get_uint(alignment);
        yyjson_obj_iter entries = yyjson_obj_iter_with(metadata);
        while (yyjson_val* key = yyjson_obj_iter_next(&entries))
            if (yyjson_val* value = yyjson_obj_iter_get_val(key); yyjson_is_str(value))
                per_custom_field(safegraphs_custom_field_t {safegraphs_string_(key), safegraphs_string_(value)});
    }
    safegraphs_parse_column_(yyjson_obj_get(root, "key"), header.key);
    safegraphs_parse_column_(yyjson_obj_get(root, "value"), header.value);
    safegraphs_parse_column_(yyjson_obj_get(root, "neighbors"), header.neighbors);
    yyjson_doc_free(document);

    // The data segment starts on the declared alignment boundary at or past the JSON's end.
    if (!std::has_single_bit(header.alignment)) return {header, status_t::malformed_k};
    header.data_offset = divide_round_up<std::size_t>(sizeof(prefix) + header_length, header.alignment) * //
                         header.alignment;
    return {header, status_t::success_k};
}

/** @brief Opens @p path read-only and parses its header. */
template <typename allocator_type_ = std::allocator<std::byte>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_header_t> safegraphs_read_header(std::string_view path, allocator_type_ allocator = {},
                                                     callback_type_&& per_custom_field = {}) noexcept {
    auto [file, opened] = file_t::open(path, file_mode_t::read_k, allocator);
    if (failed(opened)) return {safegraphs_header_t {}, opened};
    return safegraphs_read_header(file, allocator, std::forward<callback_type_>(per_custom_field));
}

/** @brief Builds the header's JSON document; null when an allocation failed. */
inline yyjson_mut_doc* safegraphs_header_document_(safegraphs_header_t const& header,
                                                   std::span<safegraphs_custom_field_t const> custom_fields,
                                                   yyjson_alc const& allocator_abi) noexcept {
    yyjson_mut_doc* document = yyjson_mut_doc_new(&allocator_abi);
    if (!document) return nullptr;
    yyjson_mut_val* root = yyjson_mut_obj(document);
    yyjson_mut_doc_set_root(document, root);
    yyjson_mut_val* metadata = yyjson_mut_obj_add_obj(document, root, "__metadata__");
    bool complete = yyjson_mut_obj_add_uint(document, metadata, "nodes", header.nodes) &&
                    yyjson_mut_obj_add_uint(document, metadata, "alignment", header.alignment);
    for (safegraphs_custom_field_t const& field : custom_fields)
        complete = complete && yyjson_mut_obj_add(metadata, //
                                                  yyjson_mut_strncpy(document, field.key.data(), field.key.size()),
                                                  yyjson_mut_strncpy(document, field.value.data(), field.value.size()));
    auto add_column = [&](char const* label, safegraphs_column_t const& column) noexcept {
        yyjson_mut_val* object = yyjson_mut_obj_add_obj(document, root, label);
        return yyjson_mut_obj_add_str(document, object, "dtype", dtype_name(column.dtype)) &&
               yyjson_mut_obj_add_uint(document, object, "shape", column.shape) &&
               (column.shape_effective == column.shape ||
                yyjson_mut_obj_add_uint(document, object, "shape_effective", column.shape_effective)) &&
               yyjson_mut_obj_add_uint(document, object, "values_start", column.values_start) &&
               yyjson_mut_obj_add_uint(document, object, "values_length", column.values_length);
    };
    complete = complete && add_column("key", header.key) && add_column("value", header.value) &&
               add_column("neighbors", header.neighbors);
    if (complete) return document;
    yyjson_mut_doc_free(document);
    return nullptr;
}

/** @brief Serializes @p document, hands the text to @p consume, then frees both; a null document
 *      reports @c bad_alloc_k. */
template <typename consumer_type_>
status_t safegraphs_print_(yyjson_mut_doc* document, yyjson_alc const& allocator_abi,
                           consumer_type_&& consume) noexcept {
    std::size_t length = 0;
    char* text = document ? yyjson_mut_write_opts(document, 0, &allocator_abi, &length, nullptr) : nullptr;
    yyjson_mut_doc_free(document);
    if (!text) return status_t::bad_alloc_k;
    status_t const consumed = consume(std::string_view {text, length});
    allocator_abi.free(allocator_abi.ctx, text);
    return consumed;
}

/** @brief Where a header lands its data segment: the prefix and JSON rounded up to the declared
 *      alignment; zero when the header cannot be formatted. */
template <typename allocator_type_ = std::allocator<std::byte>>
std::size_t safegraphs_data_offset(safegraphs_header_t const& header,
                                   std::span<safegraphs_custom_field_t const> custom_fields = {},
                                   allocator_type_ allocator = {}) noexcept {
    if (!std::has_single_bit(header.alignment)) return 0;
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    yyjson_alc const allocator_abi = trampoline.c_abi();
    std::size_t json_bytes = 0;
    status_t const printed = safegraphs_print_(
        safegraphs_header_document_(header, custom_fields, allocator_abi), allocator_abi,
        [&](std::string_view json) noexcept { return json_bytes = json.size(), status_t::success_k; });
    if (failed(printed)) return 0;
    return divide_round_up<std::size_t>(sizeof(safegraphs_prefix_t) + json_bytes, header.alignment) * //
           header.alignment;
}

/** @brief Writes the prefix, the JSON, and space padding up to @c header.data_offset - exactly
 *      the bytes @ref safegraphs_read_header consumes; the file must sit at its start. */
template <typename allocator_type_ = std::allocator<std::byte>>
status_t safegraphs_write_header(file_t& file, safegraphs_header_t const& header,
                                 std::span<safegraphs_custom_field_t const> custom_fields = {},
                                 allocator_type_ allocator = {}) noexcept {
    if (header.data_offset < sizeof(safegraphs_prefix_t)) return status_t::config_mismatch_k;
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    yyjson_alc const allocator_abi = trampoline.c_abi();
    safegraphs_prefix_t prefix;
    prefix.header_length = header.data_offset - sizeof(prefix);
    return safegraphs_print_(safegraphs_header_document_(header, custom_fields, allocator_abi), allocator_abi,
                             [&](std::string_view json) noexcept {
                                 if (json.size() > prefix.header_length) return status_t::config_mismatch_k;
                                 if (status_t const wrote = file.write(&prefix, sizeof(prefix)); failed(wrote))
                                     return wrote;
                                 if (status_t const wrote = file.write(json.data(), json.size()); failed(wrote))
                                     return wrote;
                                 return file.fill(' ', static_cast<std::size_t>(prefix.header_length) - json.size());
                             });
}

/** @brief Writes the three regular columns at their declared starts past @c header.data_offset;
 *      each buffer must match its column's length, so a header never contradicts its columns. */
inline status_t safegraphs_write_columns(file_t& file, safegraphs_header_t const& header, bytes_view_t keys,
                                         bytes_view_t values, bytes_view_t neighbors) noexcept {
    auto write_column = [&](safegraphs_column_t const& column, bytes_view_t bytes) noexcept {
        if (column.values_length != bytes.size()) return status_t::config_mismatch_k;
        if (bytes.empty()) return status_t::success_k;
        if (status_t const sought = file.seek(header.data_offset + column.values_start); failed(sought)) return sought;
        return file.write(bytes);
    };
    if (status_t const wrote = write_column(header.key, keys); failed(wrote)) return wrote;
    if (status_t const wrote = write_column(header.value, values); failed(wrote)) return wrote;
    return write_column(header.neighbors, neighbors);
}

/** @brief Reads the three regular columns into caller buffers; each buffer must match its
 *      column's length, so a crafted header never outruns what it lands in. */
inline status_t safegraphs_read_columns(file_t& file, safegraphs_header_t const& header, bytes_span_t keys,
                                        bytes_span_t values, bytes_span_t neighbors) noexcept {
    auto read_column = [&](safegraphs_column_t const& column, bytes_span_t bytes) noexcept {
        if (column.values_length != bytes.size()) return status_t::malformed_k;
        if (bytes.empty()) return status_t::success_k;
        if (status_t const sought = file.seek(header.data_offset + column.values_start); failed(sought)) return sought;
        return file.read(bytes);
    };
    if (status_t const read = read_column(header.key, keys); failed(read)) return read;
    if (status_t const read = read_column(header.value, values); failed(read)) return read;
    return read_column(header.neighbors, neighbors);
}

/** @brief Folds every out-of-range neighbor ID onto @p sentinel: the interchange admits any,
 *      a traversal knows one. */
template <std::unsigned_integral id_type_>
void safegraphs_canonicalize_neighbors(std::span<id_type_> neighbors, std::size_t nodes, id_type_ sentinel) noexcept {
    for (id_type_& neighbor : neighbors)
        if (neighbor >= nodes) neighbor = sentinel;
}

/** @brief One manifest entry: a shard's path relative to the manifest, its byte size, and its
 *      node count. */
struct safegraphs_shard_t {
    /** @brief The monolith's path, relative to the manifest. */
    std::string_view path;
    /** @brief The monolith's whole size in bytes. */
    std::size_t bytes = 0;
    /** @brief Nodes the monolith holds. */
    std::size_t nodes = 0;
};

/** @brief Holds the longest NumKong kernel spelling, 22 characters, with its NUL. */
inline constexpr std::size_t safegraphs_name_capacity_k = 32;

/** @brief A NUL-terminated metadata name; longer input truncates, so a foreign name fails a
 *      comparison instead of overrunning. */
struct safegraphs_name_t {
    char characters[safegraphs_name_capacity_k] = {};

    safegraphs_name_t() = default;
    explicit safegraphs_name_t(std::string_view text) noexcept {
        if (!text.empty()) std::memcpy(characters, text.data(), min_of(text.size(), safegraphs_name_capacity_k - 1));
    }
    std::string_view view() const noexcept { return characters; }
    bool empty() const noexcept { return !characters[0]; }
};

/** @brief What a manifest or a lone monolith says about itself without opening a column. */
struct safegraphs_metadata_t {
    /** @brief The metric the graph was built for; empty when the producer omitted it. */
    safegraphs_name_t metric;
    /** @brief The stored scalar by its own name - what a packed @c u8 value column cannot say;
     *      empty when the producer omitted it. */
    safegraphs_name_t quantization;
    /** @brief Element type of every file's value column. */
    dtype_t value_dtype = dtype_t::unknown_k;
    /** @brief Element type of every file's key column. */
    dtype_t key_dtype = dtype_t::unknown_k;
    /** @brief Stored values per row of the value column - its @c shape, padded or packed. */
    std::size_t value_length = 0;
    /** @brief Logical values per row, from the first file's header. */
    std::size_t dimensions = 0;
    /** @brief Neighbor IDs per node's list, from the first file's header. */
    std::size_t connectivity = 0;
    /** @brief Total nodes across every file. */
    std::size_t nodes = 0;
    /** @brief How many monoliths the checkpoint spans. */
    std::size_t files = 0;
};

/** @brief @p path with its @c .json and @c .safegraphs suffixes removed. */
inline std::string_view safegraphs_stem(std::string_view path) noexcept {
    if (path.ends_with(".json")) path.remove_suffix(std::string_view {".json"}.size());
    if (path.ends_with(".safegraphs")) path.remove_suffix(std::string_view {".safegraphs"}.size());
    return path;
}

/** @brief The last path component - what a manifest entry records. */
inline std::string_view safegraphs_basename(std::string_view path) noexcept {
    return path.substr(path.find_last_of("/\\") + 1);
}

/** @brief Formats @c <stem>.<shard>.safegraphs into @p buffer, measured first. */
template <typename allocator_type_>
expected<std::string_view> safegraphs_shard_path(growing_buffer<char, allocator_type_>& buffer, std::string_view stem,
                                                 std::size_t shard) noexcept {
    auto print = [&](char* sink, std::size_t capacity) noexcept {
        return std::snprintf(sink, capacity, "%.*s.%zu.safegraphs", static_cast<int>(stem.size()), stem.data(), shard);
    };
    int const length = print(nullptr, 0);
    if (length < 0) return {{}, status_t::io_k};
    if (failed(buffer.resize_uninitialized(static_cast<std::size_t>(length) + 1))) return {{}, status_t::bad_alloc_k};
    print(buffer.data(), buffer.size());
    return {std::string_view {buffer.data(), static_cast<std::size_t>(length)}, status_t::success_k};
}

/** @brief Resolves an entry's @p relative path beside @p manifest_path into @p buffer. */
template <typename allocator_type_>
expected<std::string_view> safegraphs_sibling_path(growing_buffer<char, allocator_type_>& buffer,
                                                   std::string_view manifest_path, std::string_view relative) noexcept {
    std::string_view const directory = manifest_path.substr(
        0, manifest_path.size() - safegraphs_basename(manifest_path).size());
    if (failed(buffer.resize_uninitialized(directory.size() + relative.size()))) return {{}, status_t::bad_alloc_k};
    std::copy_n(directory.data(), directory.size(), buffer.data());
    std::copy_n(relative.data(), relative.size(), buffer.data() + directory.size());
    return {to_string_view(buffer), status_t::success_k};
}

/** @brief Reads a manifest: the metadata returns, its geometry taken from the first file's header,
 *      and @p per_shard fires once per entry in file order with a path that dies with this call. */
template <typename allocator_type_ = std::allocator<std::byte>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_metadata_t> safegraphs_read_manifest(std::string_view path, allocator_type_ allocator = {},
                                                         callback_type_&& per_shard = {}) noexcept {
    safegraphs_metadata_t metadata;
    auto [file, opened] = file_t::open(path, file_mode_t::read_k, allocator);
    if (failed(opened)) return {metadata, opened};
    auto const [bytes, measured] = file.length();
    if (failed(measured) || !bytes) return {metadata, status_t::io_k};
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    auto [document, parsed] = safegraphs_read_document_(file, bytes, trampoline);
    if (failed(parsed)) return {metadata, parsed};

    yyjson_val* root = yyjson_doc_get_root(document);
    if (yyjson_val* summary = yyjson_obj_get(root, "__metadata__"); yyjson_is_obj(summary)) {
        metadata.metric = safegraphs_name_t {safegraphs_string_(yyjson_obj_get(summary, "metric"))};
        metadata.quantization = safegraphs_name_t {safegraphs_string_(yyjson_obj_get(summary, "quantization"))};
        metadata.value_dtype = dtype_named(safegraphs_string_(yyjson_obj_get(summary, "value_dtype")));
        metadata.key_dtype = dtype_named(safegraphs_string_(yyjson_obj_get(summary, "key_dtype")));
        metadata.value_length = yyjson_get_uint(yyjson_obj_get(summary, "value_length"));
        metadata.nodes = yyjson_get_uint(yyjson_obj_get(summary, "nodes"));
    }
    yyjson_val* files = yyjson_obj_get(root, "files");
    metadata.files = yyjson_arr_size(files);
    std::string_view first_relative;
    for (std::size_t file_index = 0; file_index != metadata.files; ++file_index) {
        yyjson_val* entry = yyjson_arr_get(files, file_index);
        std::string_view const relative = safegraphs_string_(yyjson_obj_get(entry, "path"));
        if (relative.empty()) continue;
        if (first_relative.empty()) first_relative = relative;
        per_shard(safegraphs_shard_t {relative, yyjson_get_uint(yyjson_obj_get(entry, "bytes")),
                                      yyjson_get_uint(yyjson_obj_get(entry, "nodes"))});
    }
    status_t listed = status_t::success_k;
    // Column schema lives in the files, so the first one's header answers for the geometry.
    if (!first_relative.empty()) {
        growing_buffer<char> sibling;
        auto const [first_path, resolved] = safegraphs_sibling_path(sibling, path, first_relative);
        auto const [header, parsed_header] = failed(resolved) ? expected<safegraphs_header_t> {{}, resolved}
                                                              : safegraphs_read_header(first_path, allocator);
        listed = parsed_header;
        metadata.dimensions = header.value.shape_effective;
        metadata.connectivity = header.neighbors.shape;
    }
    yyjson_doc_free(document);
    return {metadata, listed};
}

/** @brief Writes a manifest: the metadata, then @c metadata.files entries drawn from
 *      @p shard_at in order - the commit record a sharded save writes last. */
template <typename callback_type_, typename allocator_type_ = std::allocator<std::byte>>
status_t safegraphs_write_manifest(std::string_view path, safegraphs_metadata_t const& metadata,
                                   callback_type_&& shard_at, allocator_type_ allocator = {}) noexcept {
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    yyjson_alc const allocator_abi = trampoline.c_abi();
    yyjson_mut_doc* document = yyjson_mut_doc_new(&allocator_abi);
    if (!document) return status_t::bad_alloc_k;
    yyjson_mut_val* root = yyjson_mut_obj(document);
    yyjson_mut_doc_set_root(document, root);
    yyjson_mut_val* summary = yyjson_mut_obj_add_obj(document, root, "__metadata__");
    bool complete = yyjson_mut_obj_add_str(document, summary, "metric", metadata.metric.characters) &&
                    yyjson_mut_obj_add_str(document, summary, "quantization", metadata.quantization.characters) &&
                    yyjson_mut_obj_add_str(document, summary, "value_dtype", dtype_name(metadata.value_dtype)) &&
                    yyjson_mut_obj_add_str(document, summary, "key_dtype", dtype_name(metadata.key_dtype)) &&
                    yyjson_mut_obj_add_uint(document, summary, "value_length", metadata.value_length) &&
                    yyjson_mut_obj_add_uint(document, summary, "nodes", metadata.nodes);
    yyjson_mut_val* files = yyjson_mut_obj_add_arr(document, root, "files");
    for (std::size_t file_index = 0; file_index != metadata.files && complete; ++file_index) {
        safegraphs_shard_t const entry = shard_at(file_index);
        yyjson_mut_val* object = yyjson_mut_arr_add_obj(document, files);
        complete = yyjson_mut_obj_add_strncpy(document, object, "path", entry.path.data(), entry.path.size()) &&
                   yyjson_mut_obj_add_uint(document, object, "bytes", entry.bytes) &&
                   yyjson_mut_obj_add_uint(document, object, "nodes", entry.nodes);
    }
    if (!complete) return yyjson_mut_doc_free(document), status_t::bad_alloc_k;
    return safegraphs_print_(document, allocator_abi, [&](std::string_view json) noexcept {
        auto [file, opened] = file_t::open(path, file_mode_t::write_k, allocator);
        if (failed(opened)) return opened;
        if (status_t const wrote = file.write(json.data(), json.size()); failed(wrote)) return wrote;
        return file.close();
    });
}

/** @brief The metadata of a manifest or, for a lone monolith, of its header. */
template <typename allocator_type_ = std::allocator<std::byte>>
expected<safegraphs_metadata_t> safegraphs_read_metadata(std::string_view path,
                                                         allocator_type_ allocator = {}) noexcept {
    if (path.ends_with(".json")) return safegraphs_read_manifest(path, allocator);
    safegraphs_metadata_t metadata;
    auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") metadata.metric = safegraphs_name_t {field.value};
        if (field.key == "quantization") metadata.quantization = safegraphs_name_t {field.value};
    };
    auto const [header, parsed] = safegraphs_read_header(path, allocator, per_custom_field);
    if (failed(parsed)) return {metadata, parsed};
    metadata.value_dtype = header.value.dtype;
    metadata.key_dtype = header.key.dtype;
    metadata.value_length = header.value.shape;
    metadata.dimensions = header.value.shape_effective;
    metadata.connectivity = header.neighbors.shape;
    metadata.nodes = header.nodes;
    metadata.files = 1;
    return {metadata, status_t::success_k};
}

} // namespace usearch
} // namespace unum
