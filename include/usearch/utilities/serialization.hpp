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
 *  size, and the JSON bytes - nothing is memory-mapped. Columns come back as
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

#include <cstddef> // `std::byte` - the staged headers' element type
#include <cstdio>  // `std::FILE`, `std::fread`, `std::fopen`, `std::snprintf`
#include <cstring> // `std::memcpy`, `std::memset` - the transient path termination & padding

#include <algorithm>   // `std::copy_n`
#include <memory>      // `std::allocator_traits`
#include <span>        // `std::span` - the writers' custom-field bags
#include <string_view> // `std::string_view`
#include <utility>     // `std::exchange` - the move plumbing of `file_t`

#include <yyjson.h> // The JSON walk - exceptions-free C

#include <numkong/tensor.hpp> // `nk::matrix_view`, `nk::matrix_span`

#include "usearch/utilities/types.hpp"

namespace unum {
namespace usearch {

/** @brief How @ref file_t::open opens a file: reading an existing one, or truncating a fresh one. */
enum class file_mode_t { read_k, write_k };

/**
 *  @brief An owned file handle: LibC's raw stream and the OS-specific positioning &
 *      measurement calls live behind it, and every verb answers the house @c status_t -
 *      no raw @c std::FILE* or C error convention crosses this boundary.
 */
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

    /** @brief Opens @p path through the caller's allocator: LibC's file APIs demand
     *      NUL-terminated strings, so the length-aware view gains its terminator in one
     *      transient allocation - no @c std::string, and no global bound like @c PATH_MAX
     *      pretends to be. */
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
        return {std::move(file), file ? status_t::success_k : status_t::io_k};
    }

    /** @brief Opens an anonymous read-write temporary - unlinked by the OS, gone on close -
     *      for staging that must survive a rewind but never deserves a name. */
    static expected<file_t> temporary() noexcept {
        file_t file;
        file.stream_ = std::tmpfile();
        return {std::move(file), file ? status_t::success_k : status_t::io_k};
    }

    /** @brief Positions the cursor at the absolute @p offset through the platform's 64-bit
     *      seek - @c std::fseek takes a @c long, which wraps past 2 GB wherever longs are
     *      32 bits. */
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

    /** @brief Writes @p text verbatim - the JSON-fragment sibling of the byte-count form. */
    status_t write(std::string_view text) noexcept { return write(text.data(), text.size()); }

    /** @brief The span sibling of the byte-count read. */
    status_t read(bytes_span_t destination) noexcept { return read(destination.data(), destination.size()); }

    /** @brief The span sibling of the byte-count write. */
    status_t write(bytes_view_t source) noexcept { return write(source.data(), source.size()); }

    /** @brief Writes @p length copies of @p value - what pads a header out to its alignment. */
    status_t fill(char value, std::size_t length) noexcept {
        char batch[64];
        std::memset(batch, value, sizeof(batch));
        for (; length > sizeof(batch); length -= sizeof(batch))
            if (failed(write(batch, sizeof(batch)))) return status_t::io_k;
        return write(batch, length);
    }

    /** @brief Flushes & closes explicitly, answering what the destructor must swallow - the
     *      write path ends here, where a buffered failure would otherwise vanish. */
    status_t close() noexcept {
        if (!stream_) return status_t::success_k;
        return std::fclose(std::exchange(stream_, nullptr)) == 0 ? status_t::success_k : status_t::io_k;
    }

    /** @brief Byte length of the open file - measured by a 64-bit seek to the end, rewinding to
     *      the start, which is where every consumer of a whole file reads from anyway. */
    expected<std::size_t> length() noexcept {
#if defined(_WIN32)
        if (_fseeki64(stream_, 0, SEEK_END) != 0) return {0, status_t::io_k};
        long long const told = _ftelli64(stream_);
#else
        if (fseeko(stream_, 0, SEEK_END) != 0) return {0, status_t::io_k};
        off_t const told = ftello(stream_);
#endif
        if (told < 0 || failed(seek(0))) return {0, status_t::io_k};
        return {static_cast<std::size_t>(told), status_t::success_k};
    }

  private:
    /** @brief The wrapped LibC stream; null marks the empty state a move-out leaves behind. */
    std::FILE* stream_ = nullptr;
};

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

/** @brief The fixed prefix every monolith opens with: the magic, then the little-endian 64-bit
 *      JSON length. */
inline constexpr std::size_t safegraphs_prefix_length_k = sizeof(safegraphs_header_t::magic_k) + sizeof(u64_t);

/** @brief One custom field of the open `__metadata__` bag - a producer's own key & its
 *      string value. The format defines only @c nodes & @c alignment; everything else rides
 *      here, written from the caller's spans and read back through its callback. */
struct safegraphs_custom_field_t {
    std::string_view key;
    std::string_view value;
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
    if (yyjson_val* dtype = yyjson_obj_get(object, "dtype"); yyjson_is_str(dtype))
        column.dtype = dtype_named({yyjson_get_str(dtype), yyjson_get_len(dtype)});
    if (yyjson_val* dtype = yyjson_obj_get(object, "offsets_dtype"); yyjson_is_str(dtype))
        column.offsets_dtype = dtype_named({yyjson_get_str(dtype), yyjson_get_len(dtype)});
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
 *  @param[in] file An open, readable file positioned at its start.
 *  @param[in] allocator Backs the JSON buffer & the yyjson document; any byte allocator.
 *  @param[in] per_custom_field Fires once per string-valued `__metadata__` entry - a
 *      producer's custom field; the views borrow the parsed document and die with this call.
 *  @return The parsed header, or @c malformed_k / @c bad_alloc_k explaining the failure.
 */
template <typename allocator_type_ = std::allocator<std::byte>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_header_t> safegraphs_read_header(file_t& file, allocator_type_ allocator = {},
                                                     callback_type_&& per_custom_field = {}) noexcept {
    safegraphs_header_t header;
    char prefix[safegraphs_prefix_length_k];
    if (failed(file.read(prefix, sizeof(prefix)))) return {header, status_t::malformed_k};
    for (std::size_t index = 0; index != sizeof(header.magic_k); ++index)
        if (prefix[index] != safegraphs_header_t::magic_k[index]) return {header, status_t::malformed_k};

    std::size_t header_size = 0;
    // The header size arrives little-endian; assembling bytes keeps the read host-agnostic.
    for (std::size_t byte = 0; byte != sizeof(u64_t); ++byte)
        header_size |= std::size_t(static_cast<unsigned char>(prefix[sizeof(header.magic_k) + byte])) << (byte * 8);

    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    yyjson_alc allocator_abi = trampoline.c_abi();
    char* json = static_cast<char*>(safegraphs_yyjson_allocator_<allocator_type_>::allocate_(&trampoline, //
                                                                                             header_size));
    if (!json) return {header, status_t::bad_alloc_k};
    bool const read_whole = succeeded(file.read(json, header_size));
    yyjson_doc* document = read_whole ? yyjson_read_opts(json, header_size, 0, &allocator_abi, nullptr) : nullptr;
    if (!document) {
        safegraphs_yyjson_allocator_<allocator_type_>::deallocate_(&trampoline, json);
        return {header, status_t::malformed_k};
    }

    yyjson_val* root = yyjson_doc_get_root(document);
    if (yyjson_val* metadata = yyjson_obj_get(root, "__metadata__"); yyjson_is_obj(metadata)) {
        if (yyjson_val* nodes = yyjson_obj_get(metadata, "nodes")) header.nodes = yyjson_get_uint(nodes);
        if (yyjson_val* alignment = yyjson_obj_get(metadata, "alignment"))
            header.alignment = yyjson_get_uint(alignment);
        yyjson_obj_iter entries = yyjson_obj_iter_with(metadata);
        while (yyjson_val* key = yyjson_obj_iter_next(&entries)) {
            yyjson_val* value = yyjson_obj_iter_get_val(key);
            if (yyjson_is_str(value))
                per_custom_field(safegraphs_custom_field_t {{yyjson_get_str(key), yyjson_get_len(key)},
                                                            {yyjson_get_str(value), yyjson_get_len(value)}});
        }
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
 *      file overload, and closes it - @c io_k when the file cannot open. */
template <typename allocator_type_ = std::allocator<std::byte>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_header_t> safegraphs_read_header(std::string_view path, allocator_type_ allocator = {},
                                                     callback_type_&& per_custom_field = {}) noexcept {
    auto [file, opened] = file_t::open(path, file_mode_t::read_k, allocator);
    if (failed(opened)) return {safegraphs_header_t {}, opened};
    return safegraphs_read_header(file, allocator, std::forward<callback_type_>(per_custom_field));
}

/** @brief Prints the header's JSON into @p sink, answering the character count; an empty
 *      @p sink measures instead, pricing every field through the very snprintf calls that
 *      would emit it, so the measure and the emission can never disagree. Negative when a real
 *      @p sink cannot hold the emission. The field vocabulary is exactly what the reader
 *      consumes; regular columns omit the offsets fields the reader defaults to zero. */
inline std::ptrdiff_t safegraphs_format_header_(bytes_span_t sink, safegraphs_header_t const& header,
                                                std::span<safegraphs_custom_field_t const> custom_fields) noexcept {
    char* const text = sink.empty() ? nullptr : reinterpret_cast<char*>(sink.data());
    std::size_t const capacity = sink.size();
    std::ptrdiff_t at = 0;
    auto print = [&]<typename... args_types_>(char const* format, args_types_... arguments) noexcept {
        if (at < 0) return;
        int const printed = std::snprintf(text ? text + at : nullptr,
                                          text ? capacity - static_cast<std::size_t>(at) : 0, format, arguments...);
        bool const overflown = printed < 0 || (text && static_cast<std::size_t>(at + printed) >= capacity);
        at = overflown ? -1 : at + printed;
    };
    auto print_column = [&](std::string_view label, safegraphs_column_t const& column,
                            std::string_view suffix) noexcept {
        std::string_view const token {dtype_name(column.dtype)};
        print("\"%.*s\":{\"dtype\":\"%.*s\",\"shape\":%zu,\"values_start\":%zu,\"values_length\":%zu}%.*s",
              static_cast<int>(label.size()), label.data(), static_cast<int>(token.size()), token.data(), column.shape,
              column.values_start, column.values_length, static_cast<int>(suffix.size()), suffix.data());
    };

    print("{\"__metadata__\":{\"nodes\":%zu,\"alignment\":%zu", header.nodes, header.alignment);
    for (safegraphs_custom_field_t const& field : custom_fields)
        print(",\"%.*s\":\"%.*s\"", static_cast<int>(field.key.size()), field.key.data(),
              static_cast<int>(field.value.size()), field.value.data());
    print("%s", "},");
    print_column("key", header.key, ",");
    print_column("value", header.value, ",");
    print_column("neighbors", header.neighbors, "}");
    return at;
}

/** @brief The @c data_offset a written header lands the payload at - the prefix and JSON padded
 *      out to the declared alignment; zero when the header cannot be formatted. @p custom_fields
 *      must be the very bag the write will carry, since its fields are part of the JSON. */
inline std::size_t safegraphs_header_length(safegraphs_header_t const& header,
                                            std::span<safegraphs_custom_field_t const> custom_fields = {}) noexcept {
    std::ptrdiff_t const length = safegraphs_format_header_({}, header, custom_fields);
    if (length < 0 || !header.alignment || (header.alignment & (header.alignment - 1)) != 0) return 0;
    return divide_round_up<std::size_t>(safegraphs_prefix_length_k + static_cast<std::size_t>(length),
                                        header.alignment) *
           header.alignment;
}

/**
 *  @brief Writes the magic, the little-endian header size, the JSON, and the alignment padding -
 *      the exact prefix @ref safegraphs_read_header consumes.
 *
 *  The file must be positioned at its start; the caller then writes each column's bytes
 *  at @c data_offset plus its declared range, in whatever order and through whatever I/O it
 *  prefers - this writer touches no payload byte, mirroring the reader. The JSON stages
 *  through one transient @p allocator allocation, sized by a measuring pass - no fixed
 *  ceiling exists to outgrow.
 */
template <typename allocator_type_ = std::allocator<std::byte>>
status_t safegraphs_write_header(file_t& file, safegraphs_header_t const& header,
                                 std::span<safegraphs_custom_field_t const> custom_fields = {},
                                 allocator_type_ allocator = {}) noexcept {
    if (!header.alignment || (header.alignment & (header.alignment - 1)) != 0) return status_t::malformed_k;
    // Fields are quoted verbatim, so one that would need JSON escaping is refused rather than
    // emitted as a header the reader would reject.
    for (safegraphs_custom_field_t const& field : custom_fields)
        if (field.key.find_first_of("\"\\") != std::string_view::npos ||
            field.value.find_first_of("\"\\") != std::string_view::npos)
            return status_t::malformed_k;
    std::ptrdiff_t const measured = safegraphs_format_header_({}, header, custom_fields);
    if (measured < 0) return status_t::malformed_k;
    std::size_t const data_offset = divide_round_up<std::size_t>(
                                        safegraphs_prefix_length_k + static_cast<std::size_t>(measured),
                                        header.alignment) *
                                    header.alignment;
    std::size_t const header_size = data_offset - safegraphs_prefix_length_k;

    char prefix[safegraphs_prefix_length_k];
    std::copy_n(safegraphs_header_t::magic_k, sizeof(safegraphs_header_t::magic_k), prefix);
    // The header size leaves little-endian; assembling bytes keeps the write host-agnostic.
    for (std::size_t byte = 0; byte != sizeof(u64_t); ++byte)
        prefix[sizeof(safegraphs_header_t::magic_k) + byte] = static_cast<char>((header_size >> (byte * 8)) & 0xFF);

    using bytes_allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<std::byte>;
    bytes_allocator_t bytes {allocator};
    std::size_t const staged_length = static_cast<std::size_t>(measured) + 1;
    std::byte* staged = bytes.allocate(staged_length);
    if (!staged) return status_t::bad_alloc_k;
    std::ptrdiff_t const length = safegraphs_format_header_({staged, staged_length}, header, custom_fields);
    status_t written = length == measured ? status_t::success_k : status_t::malformed_k;
    if (succeeded(written) && failed(file.write(prefix, sizeof(prefix)))) written = status_t::io_k;
    if (succeeded(written) && failed(file.write(staged, static_cast<std::size_t>(measured)))) written = status_t::io_k;
    bytes.deallocate(staged, staged_length);
    if (failed(written)) return written;
    return file.fill(' ', header_size - static_cast<std::size_t>(measured));
}

/** @brief One shard's entry in a manifest: where its monolith sits relative to the manifest,
 *      and how many nodes it carries. */
struct safegraphs_shard_t {
    std::string_view path;
    std::size_t nodes = 0;
};

/** @brief A manifest's own summary - what a metadata query answers without opening a shard,
 *      which is why the cost of asking never grows with the index. */
struct safegraphs_manifest_t {
    /** @brief Element type of every shard's value column. */
    dtype_t dtype = dtype_t::unknown_k;
    /** @brief The `__metadata__` kernel token, verbatim & NUL-terminated; empty when the
     *      producer omitted it. A foreign over-long token truncates, failing the consumer's
     *      comparison rather than being trusted. */
    char kernel[16] = {};
    /** @brief Scalar dimensions per stored vector. */
    std::size_t dimensions = 0;
    /** @brief Neighbor ids per node's list. */
    std::size_t connectivity = 0;
    /** @brief Total nodes across every shard. */
    std::size_t nodes = 0;
    /** @brief How many shard monoliths the manifest enumerates. */
    std::size_t shards = 0;
};

/**
 *  @brief Reads a @c .safegraphs.json manifest: the summary returns, and @p per_shard fires once
 *      per enumerated shard with its relative path and node count, in file order.
 *
 *  The paths handed to @p per_shard borrow the parsed document and die with this call - resolve
 *  or copy them inside the callback.
 */
template <typename allocator_type_ = std::allocator<std::byte>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_manifest_t> safegraphs_read_manifest(std::string_view path, allocator_type_ allocator = {},
                                                         callback_type_&& per_shard = {}) noexcept {
    safegraphs_manifest_t manifest;
    auto [file, opened] = file_t::open(path, file_mode_t::read_k, allocator);
    if (failed(opened)) return {manifest, opened};
    auto const [bytes, measured] = file.length();
    if (failed(measured) || !bytes) return {manifest, status_t::io_k};

    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    yyjson_alc allocator_abi = trampoline.c_abi();
    char* json = static_cast<char*>(safegraphs_yyjson_allocator_<allocator_type_>::allocate_(&trampoline, bytes));
    if (!json) return {manifest, status_t::bad_alloc_k};
    bool const read_whole = succeeded(file.read(json, bytes));
    yyjson_doc* document = read_whole ? yyjson_read_opts(json, bytes, 0, &allocator_abi, nullptr) : nullptr;
    if (!document) {
        safegraphs_yyjson_allocator_<allocator_type_>::deallocate_(&trampoline, json);
        return {manifest, status_t::malformed_k};
    }

    yyjson_val* root = yyjson_doc_get_root(document);
    if (yyjson_val* metadata = yyjson_obj_get(root, "__metadata__"); yyjson_is_obj(metadata)) {
        if (yyjson_val* dtype = yyjson_obj_get(metadata, "dtype"); yyjson_is_str(dtype))
            manifest.dtype = dtype_named({yyjson_get_str(dtype), yyjson_get_len(dtype)});
        if (yyjson_val* kernel = yyjson_obj_get(metadata, "kernel"); yyjson_is_str(kernel)) {
            std::string_view const token {yyjson_get_str(kernel), yyjson_get_len(kernel)};
            std::size_t const copied = token.size() < sizeof(manifest.kernel) ? token.size() //
                                                                              : sizeof(manifest.kernel) - 1;
            std::memcpy(manifest.kernel, token.data(), copied);
        }
        manifest.dimensions = yyjson_get_uint(yyjson_obj_get(metadata, "dimensions"));
        manifest.connectivity = yyjson_get_uint(yyjson_obj_get(metadata, "connectivity"));
        manifest.nodes = yyjson_get_uint(yyjson_obj_get(metadata, "nodes"));
    }
    yyjson_val* shards = yyjson_obj_get(root, "shards");
    std::size_t const shards_count = yyjson_arr_size(shards);
    for (std::size_t shard = 0; shard != shards_count; ++shard) {
        yyjson_val* entry = yyjson_arr_get(shards, shard);
        char const* relative = yyjson_get_str(yyjson_obj_get(entry, "path"));
        std::size_t const nodes = yyjson_get_uint(yyjson_obj_get(entry, "nodes"));
        if (relative) per_shard(safegraphs_shard_t {relative, nodes});
    }
    manifest.shards = shards_count;
    bool const object_root = yyjson_is_obj(root);
    yyjson_doc_free(document);
    safegraphs_yyjson_allocator_<allocator_type_>::deallocate_(&trampoline, json);
    return {manifest, object_root ? status_t::success_k : status_t::malformed_k};
}

/**
 *  @brief Writes a @c .safegraphs.json manifest: the summary, then @p shards_count entries drawn
 *      from @p shard_at in order - the commit record a sharded save writes last, so a checkpoint
 *      whose save died reads as absent rather than half-valid.
 */
template <typename callback_type_, typename allocator_type_ = std::allocator<std::byte>>
status_t safegraphs_write_manifest(std::string_view path, safegraphs_manifest_t const& manifest,
                                   std::size_t shards_count, callback_type_&& shard_at,
                                   allocator_type_ allocator = {}) noexcept {
    auto [file, opened] = file_t::open(path, file_mode_t::write_k, allocator);
    if (failed(opened)) return opened;
    char staged[192];
    std::string_view const dtype_token {dtype_name(manifest.dtype)};
    std::string_view const kernel_token {manifest.kernel};
    int const headed = std::snprintf(
        staged, sizeof(staged),
        "{\"__metadata__\":{\"dtype\":\"%.*s\",\"kernel\":\"%.*s\",\"dimensions\":%zu," "\"connectivity\":%zu,"
                                                                                        "\"nodes\":%zu},\"shards\":[",
        static_cast<int>(dtype_token.size()), dtype_token.data(), static_cast<int>(kernel_token.size()),
        kernel_token.data(), manifest.dimensions, manifest.connectivity, manifest.nodes);
    if (headed < 0 || static_cast<std::size_t>(headed) >= sizeof(staged)) return status_t::io_k;
    if (failed(file.write(staged, static_cast<std::size_t>(headed)))) return status_t::io_k;
    for (std::size_t shard = 0; shard != shards_count; ++shard) {
        safegraphs_shard_t const entry = shard_at(shard);
        // Paths are quoted verbatim, so one that would need JSON escaping is refused rather
        // than emitted as a manifest the reader would reject. Being caller-length, the path
        // crosses raw between two staged fragments.
        if (entry.path.find_first_of("\"\\") != std::string_view::npos) return status_t::malformed_k;
        constexpr std::string_view first_prefix_k = "{\"path\":\"";
        constexpr std::string_view later_prefix_k = ",{\"path\":\"";
        if (failed(file.write(shard ? later_prefix_k : first_prefix_k))) return status_t::io_k;
        if (failed(file.write(entry.path))) return status_t::io_k;
        int const closed = std::snprintf(staged, sizeof(staged), "\",\"nodes\":%zu}", entry.nodes);
        if (closed < 0 || static_cast<std::size_t>(closed) >= sizeof(staged)) return status_t::io_k;
        if (failed(file.write(staged, static_cast<std::size_t>(closed)))) return status_t::io_k;
    }
    constexpr std::string_view closer_k = "]}";
    if (failed(file.write(closer_k))) return status_t::io_k;
    return file.close();
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
 *      @c io_k on any I/O failure.
 */
template <typename value_type_, typename counter_type_ = u32_t>
status_t matrix_save(std::string_view path, nk::matrix_view<value_type_> matrix) noexcept {
    counter_type_ const counters[2] = {static_cast<counter_type_>(matrix.extent(0)),
                                       static_cast<counter_type_>(matrix.extent(1))};
    if (counters[0] != matrix.extent(0) || counters[1] != matrix.extent(1)) return status_t::overflow_k;
    std::size_t const payload_bytes = matrix.extent(0) * static_cast<std::size_t>(matrix.stride_bytes(0));
    auto [file, opened] = file_t::open(path, file_mode_t::write_k);
    if (failed(opened)) return opened;
    if (failed(file.write(counters, sizeof(counters)))) return status_t::io_k;
    if (failed(file.write(matrix.data(), payload_bytes))) return status_t::io_k;
    return file.close();
}

/** @brief Reads just the two-counter header of @p path - the caller sizes its destination
 *      from the returned extents before fetching the payload its own way. */
template <typename counter_type_ = u32_t>
expected<matrix_extents_t> matrix_extents(std::string_view path) noexcept {
    matrix_extents_t extents;
    auto [file, opened] = file_t::open(path, file_mode_t::read_k);
    if (failed(opened)) return {extents, opened};
    counter_type_ counters[2];
    if (failed(file.read(counters, sizeof(counters)))) return {extents, status_t::malformed_k};
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
 *  @return @c success_k, @c io_k when the file cannot open, @c malformed_k on a
 *      short read, or @c config_mismatch_k when the counters disagree with @p destination.
 */
template <typename value_type_, typename counter_type_ = u32_t>
status_t matrix_load(std::string_view path, nk::matrix_span<value_type_> destination) noexcept {
    auto [file, opened] = file_t::open(path, file_mode_t::read_k);
    if (failed(opened)) return opened;
    counter_type_ counters[2];
    if (failed(file.read(counters, sizeof(counters)))) return status_t::malformed_k;
    if (counters[0] != destination.extent(0) || counters[1] != destination.extent(1))
        return status_t::config_mismatch_k;
    std::size_t const payload_bytes = destination.extent(0) * static_cast<std::size_t>(destination.stride_bytes(0));
    bool const read_whole = succeeded(file.read(destination.data(), payload_bytes));
    return read_whole ? status_t::success_k : status_t::malformed_k;
}

} // namespace usearch
} // namespace unum
