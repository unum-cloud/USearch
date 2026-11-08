/**
 *  @file tools/arrow_ipc.hpp
 *  @author Ash Vardanian
 *  @date October 29, 2026
 *  @brief A codec of Apache Arrow IPC streams and files with no Arrow library and no @c flatc, over
 *      hand-numbered FlatBuffers tables for byte strings, fixed-width values, numbers, lists and
 *      structs, with C Data adapters.
 *
 *  The codec neither allocates nor does I/O, since every buffer is the caller's. Readers name how
 *  many bytes the next step needs, so a pipe reads exactly that much, and writers answer
 *  @c ipc_status_t::too_small_k with the bytes they need, so an empty output measures. Bodies are
 *  never copied: a writer answers them as gathers for @c writev, and a reader answers spans into
 *  the body it was lent. Every scalar crosses the wire little-endian, through @c ipc_load_little
 *  and @c ipc_store_little, whatever the host.
 *
 *  The slots follow <tt>Schema.fbs</tt>, <tt>Message.fbs</tt> and <tt>File.fbs</tt> of Apache Arrow's <tt>format/</tt>
 *  directory. The names carry no type of the project they ship in, so another project copies the
 *  file with only its namespace changed.
 */
#pragma once

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::int32_t`, `std::int64_t`, `std::uint32_t`, `std::uint64_t`, `int64_t`
#include <cstring> // `std::memcmp`, `std::memcpy`, `std::memset`

#include <span>        // `std::span`
#include <string_view> // `std::string_view`

/* The C linkage matches other copies of the block, so whichever lands first defines one type. */
#pragma region C Data Interface

extern "C" {

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE           2
#define ARROW_FLAG_MAP_KEYS_SORTED    4

struct ArrowSchema {
    // Array type description
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;

    // Release callback
    void (*release)(struct ArrowSchema*);
    // Opaque producer-specific data
    void* private_data;
};

struct ArrowArray {
    // Array data description
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    struct ArrowArray** children;
    struct ArrowArray* dictionary;

    // Release callback
    void (*release)(struct ArrowArray*);
    // Opaque producer-specific data
    void* private_data;
};

#endif // ARROW_C_DATA_INTERFACE

} // extern "C"

#pragma endregion C Data Interface

namespace unum::usearch::tools {

using bytes_span_t = std::span<std::byte>;
using bytes_view_t = std::span<std::byte const>;

#pragma region Vocabulary

/** Where buffers start unless the caller asks otherwise, as Arrow recommends, so SIMD kernels read
 *  them in place. */
inline constexpr std::uint32_t ipc_alignment_default_k = 64;

/** Where buffers start for readers of the file through @c O_DIRECT and cuFile, a page, also the
 *  largest alignment. */
inline constexpr std::uint32_t ipc_alignment_direct_k = 4096;

/** The most fields a schema carries, nested ones counted, which sizes @c ipc_arrow_storage_t; each
 *  level of nesting costs a field, which also bounds how deep the codec recurses. */
inline constexpr std::size_t ipc_fields_max_k = 16;

/** The most buffers a batch carries: validity, offsets and data of every field. */
inline constexpr std::size_t ipc_buffers_max_k = 3 * ipc_fields_max_k;

/** The bytes a file starts with, of which the first 6 also end it. */
inline constexpr std::string_view ipc_file_magic_k {"ARROW1\0\0", 8};

/** The field metadata key that names an extension type. */
inline constexpr std::string_view ipc_extension_name_key_k = "ARROW:extension:name";

/** The name of Arrow's canonical UUID extension over <tt>FixedSizeBinary(16)</tt>. */
inline constexpr std::string_view ipc_extension_uuid_k = "arrow.uuid";

/** What a codec call answers. */
enum class ipc_status_t : std::uint8_t {

    /** The call did what it was asked. */
    success_k,

    /** The input ends before what it frames, so a reader needs more bytes. */
    truncated_k,

    /** The input contradicts itself or the format. */
    malformed_k,

    /** Valid Arrow the codec does not handle: big-endian, dictionaries, compression, views, unions,
     *  maps or nulls. */
    unsupported_k,

    /** An output the caller lent is too short; the call names the bytes or entries it needs where
     *  it can. */
    too_small_k,
};

/** The value types the codec reads and writes, numbered apart from Arrow's union so more widths and
 *  nested types append without renumbering. */
enum class ipc_type_t : std::uint8_t {

    /** Byte strings with 32-bit offsets, @c z in the C Data interface. */
    binary_k,

    /** Byte strings with 64-bit offsets, @c Z. */
    large_binary_k,

    /** UTF-8 strings with 32-bit offsets, @c u, which share the layout of @c binary_k. */
    utf8_k,

    /** UTF-8 strings with 64-bit offsets, @c U, which share the layout of @c large_binary_k. */
    large_utf8_k,

    /** Values of @c ipc_field_t::width bytes each, <tt>w:N</tt>. */
    fixed_size_binary_k,

    /** Integers of @c ipc_field_t::width bits, 8 to 64, signed or not, @c c @c C @c s @c S @c i
     *  @c I @c l @c L. */
    int_k,

    /** Floats of @c ipc_field_t::width bits, 16, 32 or 64, @c e @c f @c g. */
    floating_point_k,

    /** Lists of @c ipc_field_t::width values of their one child each, <tt>+w:N</tt>. */
    fixed_size_list_k,

    /** Lists of their one child with 32-bit offsets, <tt>+l</tt>. */
    list_k,

    /** Lists of their one child with 64-bit offsets, <tt>+L</tt>. */
    large_list_k,

    /** Rows of their children side by side, <tt>+s</tt>. */
    struct_k,
};

/** Whether an integer field holds signed values. */
enum class ipc_signedness_t : std::uint8_t {
    unsigned_k,
    signed_k,
};

/** Whether a field may hold nulls. Batches with nulls are refused either way; the bit only
 *  round-trips. */
enum class ipc_nullability_t : std::uint8_t {
    non_nullable_k,
    nullable_k,
};

/** What a message carries. */
enum class ipc_message_kind_t : std::uint8_t {
    schema_k,
    record_batch_k,
};

/** One pair of custom metadata, viewing bytes the caller keeps alive. */
struct ipc_key_value_t {
    std::string_view key;
    std::string_view value;
};

/** One column of a schema, or one child of a nested column. */
struct ipc_field_t {
    std::string_view name;
    ipc_type_t type = ipc_type_t::binary_k;

    /** Bytes per value of @c ipc_type_t::fixed_size_binary_k, values per list of
     *  @c ipc_type_t::fixed_size_list_k, and bits per value of @c ipc_type_t::int_k and
     *  @c ipc_type_t::floating_point_k; zero for every other type. */
    std::int32_t width = 0;

    /** Whether an @c ipc_type_t::int_k field is signed. */
    ipc_signedness_t signedness = ipc_signedness_t::unsigned_k;
    ipc_nullability_t nullability = ipc_nullability_t::non_nullable_k;

    /** Field metadata, where <tt>ARROW:extension:name</tt> names an extension type. */
    std::span<ipc_key_value_t const> metadata;

    /** The child of a list, or the children of a struct, empty for every other type. */
    std::span<ipc_field_t const> children;
};

/** A schema: its columns and its own metadata. */
struct ipc_schema_t {
    std::span<ipc_field_t const> fields;
    std::span<ipc_key_value_t const> metadata;
};

/** One field's row count and null count in a batch, as Arrow's @c FieldNode. */
struct ipc_node_t {
    std::int64_t length = 0;
    std::int64_t null_count = 0;
};

/**
 *  @brief One record batch: a node per field and its buffers, viewing bytes the caller keeps alive,
 *      both in the depth-first order of the fields, so each nested field follows its parent.
 *
 *  Every field starts with a validity buffer, empty since nulls are refused. Byte strings then take
 *  offsets and data, fixed-width values, integers and floats take data, lists take offsets, and
 *  fixed-size lists and structs take nothing more, their values living in their children.
 */
struct ipc_batch_t {
    std::int64_t length = 0;
    std::span<ipc_node_t const> nodes;
    std::span<bytes_view_t const> buffers;
};

/** One @c writev entry of a body: bytes to write, then that many zeros to align what follows. */
struct ipc_gather_t {
    bytes_view_t bytes;
    std::uint32_t padding = 0;
};

/** Where one record batch lies in a file, as Arrow's @c Block. */
struct ipc_block_t {

    /** File position of the message's prefix. */
    std::int64_t offset = 0;

    /** Bytes of the prefix, the metadata and its padding. */
    std::int32_t metadata_bytes = 0;

    /** Bytes of the body after them. */
    std::int64_t body_bytes = 0;
};

/** How a writer lays messages out. */
struct ipc_options_t {

    /** Where every body and buffer starts in the file, a power of two between 8 and
     *  @c ipc_alignment_direct_k. */
    std::uint32_t alignment = ipc_alignment_default_k;
};

#pragma endregion Vocabulary

#pragma region FlatBuffers

/** The smallest multiple of @p alignment at or above @p value, where messages and buffers start. */
constexpr std::uint64_t ipc_round_up(std::uint64_t value, std::uint64_t alignment) noexcept {
    std::uint64_t const remainder = value % alignment;
    return remainder ? value + (alignment - remainder) : value;
}

/** Reads a little-endian @p scalar_type_ from @p bytes, whatever the host's order and the address's
 *  alignment. */
template <typename scalar_type_>
constexpr scalar_type_ ipc_load_little(std::byte const* bytes) noexcept {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index != sizeof(scalar_type_); ++index)
        value |= static_cast<std::uint64_t>(bytes[index]) << (8 * index);
    return static_cast<scalar_type_>(value);
}

/** Writes @p value to @p bytes little-endian, whatever the host's order and the address's
 *  alignment. */
template <typename scalar_type_>
constexpr void ipc_store_little(std::byte* bytes, scalar_type_ value) noexcept {
    std::uint64_t const bits = static_cast<std::uint64_t>(value);
    for (std::size_t index = 0; index != sizeof(scalar_type_); ++index)
        bytes[index] = static_cast<std::byte>((bits >> (8 * index)) & 0xFF);
}

/** Slots of the @c Message table in <tt>Message.fbs</tt>. */
enum ipc_message_slot_t : std::uint16_t {
    ipc_message_version_k = 0,
    ipc_message_header_type_k = 1,
    ipc_message_header_k = 2,
    ipc_message_body_length_k = 3,
    ipc_message_custom_metadata_k = 4,
};

/** Members of the @c MessageHeader union in <tt>Message.fbs</tt>. */
enum ipc_message_header_t : std::uint8_t {
    ipc_message_header_none_k = 0,
    ipc_message_header_schema_k = 1,
    ipc_message_header_dictionary_batch_k = 2,
    ipc_message_header_record_batch_k = 3,
    ipc_message_header_tensor_k = 4,
    ipc_message_header_sparse_tensor_k = 5,
};

/** Values of @c MetadataVersion in <tt>Schema.fbs</tt> the codec reads; it writes the newest. */
enum ipc_metadata_version_t : std::int16_t {
    ipc_metadata_version_4_k = 3,
    ipc_metadata_version_5_k = 4,
};

/** Slots of the @c Schema table in <tt>Schema.fbs</tt>. */
enum ipc_schema_slot_t : std::uint16_t {
    ipc_schema_endianness_k = 0,
    ipc_schema_fields_k = 1,
    ipc_schema_custom_metadata_k = 2,
    ipc_schema_features_k = 3,
};

/** Slots of the @c Field table in <tt>Schema.fbs</tt>, where the @c type union takes two. */
enum ipc_field_slot_t : std::uint16_t {
    ipc_field_name_k = 0,
    ipc_field_nullable_k = 1,
    ipc_field_type_type_k = 2,
    ipc_field_type_k = 3,
    ipc_field_dictionary_k = 4,
    ipc_field_children_k = 5,
    ipc_field_custom_metadata_k = 6,
};

/** Members of the @c Type union in <tt>Schema.fbs</tt> the codec reads; every other member is refused. */
enum ipc_field_type_t : std::uint8_t {
    ipc_field_type_none_k = 0,
    ipc_field_type_int_k = 2,
    ipc_field_type_floating_point_k = 3,
    ipc_field_type_binary_k = 4,
    ipc_field_type_utf8_k = 5,
    ipc_field_type_list_k = 12,
    ipc_field_type_struct_k = 13,
    ipc_field_type_fixed_size_binary_k = 15,
    ipc_field_type_fixed_size_list_k = 16,
    ipc_field_type_large_binary_k = 19,
    ipc_field_type_large_utf8_k = 20,
    ipc_field_type_large_list_k = 21,
};

/** Slots of the @c Int table in <tt>Schema.fbs</tt>. */
enum ipc_int_slot_t : std::uint16_t {
    ipc_int_bit_width_k = 0,
    ipc_int_is_signed_k = 1,
};

/** Slots of the @c FloatingPoint table in <tt>Schema.fbs</tt>, whose @c Precision counts 0, 1 and 2 for
 *  16, 32 and 64 bits. */
enum ipc_floating_point_slot_t : std::uint16_t {
    ipc_floating_point_precision_k = 0,
};

/** Slots of the @c FixedSizeList table in <tt>Schema.fbs</tt>. */
enum ipc_fixed_size_list_slot_t : std::uint16_t {
    ipc_fixed_size_list_list_size_k = 0,
};

/** Slots of the @c FixedSizeBinary table in <tt>Schema.fbs</tt>. */
enum ipc_fixed_size_binary_slot_t : std::uint16_t {
    ipc_fixed_size_binary_byte_width_k = 0,
};

/** Slots of the @c KeyValue table in <tt>Schema.fbs</tt>. */
enum ipc_key_value_slot_t : std::uint16_t {
    ipc_key_value_key_k = 0,
    ipc_key_value_value_k = 1,
};

/** Slots of the @c RecordBatch table in <tt>Message.fbs</tt>. */
enum ipc_record_batch_slot_t : std::uint16_t {
    ipc_record_batch_length_k = 0,
    ipc_record_batch_nodes_k = 1,
    ipc_record_batch_buffers_k = 2,
    ipc_record_batch_compression_k = 3,
    ipc_record_batch_variadic_buffer_counts_k = 4,
};

/** Slots of the @c Footer table in <tt>File.fbs</tt>. */
enum ipc_footer_slot_t : std::uint16_t {
    ipc_footer_version_k = 0,
    ipc_footer_schema_k = 1,
    ipc_footer_dictionaries_k = 2,
    ipc_footer_record_batches_k = 3,
    ipc_footer_custom_metadata_k = 4,
};

/** Bytes of the @c FieldNode and @c Buffer structs, two longs each. */
inline constexpr std::size_t ipc_flatbuffer_pair_bytes_k = 16;

/** Bytes of the @c Block struct: a long, an int padded to 8, and a long. */
inline constexpr std::size_t ipc_flatbuffer_block_bytes_k = 24;

/** Where a FlatBuffers vector holds its elements, none when it is absent. */
struct ipc_flatbuffer_vector_t {
    std::size_t first = 0;
    std::size_t count = 0;
};

/**
 *  @brief A table inside a FlatBuffer, whose vtable and extent were checked against the buffer when
 *      it opened, so every slot reads in bounds.
 *
 *  Every read takes the caller's status and does nothing once it failed, so a reader checks once
 *  after a run of reads and answers the first failure. A reference past the end of the buffer
 *  records @c ipc_status_t::truncated_k, and one that contradicts the format
 *  @c ipc_status_t::malformed_k. A table that never opened reads as absent, every slot defaulted.
 */
class ipc_flatbuffer_table_t {
    bytes_view_t flatbuffer_;
    std::size_t table_ = 0;
    std::size_t vtable_ = 0;
    std::size_t vtable_bytes_ = 0;
    std::size_t table_bytes_ = 0;

    /** Answers in @p target where the @c uoffset at @p position points, with @p target_bytes there
     *  in bounds. */
    ipc_status_t follow(std::size_t position, std::size_t target_bytes, std::size_t& target) const noexcept {
        if (position > flatbuffer_.size() || flatbuffer_.size() - position < 4) return ipc_status_t::truncated_k;
        std::uint32_t const offset = ipc_load_little<std::uint32_t>(flatbuffer_.data() + position);
        if (offset == 0) return ipc_status_t::malformed_k;
        if (offset > flatbuffer_.size() - position || flatbuffer_.size() - position - offset < target_bytes)
            return ipc_status_t::truncated_k;
        target = position + offset;
        return ipc_status_t::success_k;
    }

    /** Opens the table the @c uoffset at @p position points to, leaving the table absent on a
     *  failure. */
    ipc_status_t open(bytes_view_t flatbuffer, std::size_t position) noexcept {
        flatbuffer_ = flatbuffer;
        std::size_t table = 0;
        if (ipc_status_t const status = follow(position, 4, table); status != ipc_status_t::success_k) return status;
        std::int64_t const vtable = static_cast<std::int64_t>(table) -
                                    ipc_load_little<std::int32_t>(flatbuffer.data() + table);
        if (vtable < 0) return ipc_status_t::malformed_k;
        if (static_cast<std::uint64_t>(vtable) + 4 > flatbuffer.size()) return ipc_status_t::truncated_k;
        std::size_t const start = static_cast<std::size_t>(vtable);
        std::size_t const vtable_bytes = ipc_load_little<std::uint16_t>(flatbuffer.data() + start);
        std::size_t const table_bytes = ipc_load_little<std::uint16_t>(flatbuffer.data() + start + 2);
        if (vtable_bytes < 4 || vtable_bytes % 2 || table_bytes < 4) return ipc_status_t::malformed_k;
        if (vtable_bytes > flatbuffer.size() - start || table_bytes > flatbuffer.size() - table)
            return ipc_status_t::truncated_k;
        table_ = table, vtable_ = start, vtable_bytes_ = vtable_bytes, table_bytes_ = table_bytes;
        return ipc_status_t::success_k;
    }

  public:
    ipc_flatbuffer_table_t() noexcept = default;

    /** Opens the table that the @c uoffset at @p position of @p flatbuffer points to, unless
     *  @p status already failed. */
    ipc_flatbuffer_table_t(bytes_view_t flatbuffer, std::size_t position, ipc_status_t& status) noexcept {
        if (status == ipc_status_t::success_k) status = open(flatbuffer, position);
    }

    /** The buffer the table lives in, which nested tables and vectors resolve against. */
    [[nodiscard]] bytes_view_t flatbuffer() const noexcept { return flatbuffer_; }

    /** Whether the table opened. */
    [[nodiscard]] bool present() const noexcept { return table_ != 0; }

    /** Where the @p bytes of @p slot start, or zero when the slot is absent. */
    std::size_t field(std::uint16_t slot, std::size_t bytes, ipc_status_t& status) const noexcept {
        std::size_t const entry = 4 + 2 * std::size_t(slot);
        if (status != ipc_status_t::success_k || entry + 2 > vtable_bytes_) return 0;
        std::size_t const offset = ipc_load_little<std::uint16_t>(flatbuffer_.data() + vtable_ + entry);
        if (offset == 0) return 0;
        if (offset < 4 || offset + bytes > table_bytes_) {
            status = ipc_status_t::malformed_k;
            return 0;
        }
        return table_ + offset;
    }

    /** The scalar of @p slot, or @p fallback when the slot is absent. */
    template <typename scalar_type_>
    scalar_type_ scalar(std::uint16_t slot, scalar_type_ fallback, ipc_status_t& status) const noexcept {
        std::size_t const position = field(slot, sizeof(scalar_type_), status);
        return position ? ipc_load_little<scalar_type_>(flatbuffer_.data() + position) : fallback;
    }

    /** The table of @p slot, absent when the slot is. */
    ipc_flatbuffer_table_t table(std::uint16_t slot, ipc_status_t& status) const noexcept {
        std::size_t const position = field(slot, 4, status);
        return position ? ipc_flatbuffer_table_t(flatbuffer_, position, status) : ipc_flatbuffer_table_t {};
    }

    /** Where the vector of @p slot holds its elements of @p element_bytes each. */
    ipc_flatbuffer_vector_t vector(std::uint16_t slot, std::size_t element_bytes, ipc_status_t& status) const noexcept {
        std::size_t const position = field(slot, 4, status);
        std::size_t start = 0;
        if (!position) return {};
        status = follow(position, 4, start);
        if (status != ipc_status_t::success_k) return {};
        std::uint32_t const count = ipc_load_little<std::uint32_t>(flatbuffer_.data() + start);
        start += 4;
        if (count > (flatbuffer_.size() - start) / element_bytes) {
            status = ipc_status_t::truncated_k;
            return {};
        }
        return {start, count};
    }

    /** The string of @p slot, empty when absent. */
    std::string_view string(std::uint16_t slot, ipc_status_t& status) const noexcept {
        ipc_flatbuffer_vector_t const bytes = vector(slot, 1, status);
        return {reinterpret_cast<char const*>(flatbuffer_.data()) + bytes.first, bytes.count};
    }
};

/** One field of a table being built: its slot in the vtable and its byte width, 1, 2, 4 or 8. */
struct ipc_flatbuffer_slot_t {
    std::uint16_t slot;
    std::uint16_t bytes;
};

/**
 *  @brief Lays a FlatBuffer out front to back, each vtable before its table and each child after
 *      its parent, so every @c uoffset points forward as the format requires.
 *
 *  Writes past the end of the output are dropped while the size still grows, so one pass both
 *  writes and measures. Everything claimed is zeroed first, padding included.
 */
class ipc_flatbuffer_builder_t {
    bytes_span_t output_;
    std::size_t size_ = 0;

  public:
    explicit ipc_flatbuffer_builder_t(bytes_span_t output) noexcept : output_(output) {}

    /** Bytes laid out so far, whether or not they fit the output. */
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /** Claims and zeroes @p bytes at the next position that leaves @p remainder modulo
     *  @p alignment. */
    std::size_t claim(std::size_t bytes, std::size_t alignment, std::size_t remainder = 0) noexcept {
        std::size_t const start = size_ + ((remainder - size_) & (alignment - 1));
        std::size_t const end = start + bytes;
        if (size_ < output_.size())
            std::memset(output_.data() + size_, 0, (end < output_.size() ? end : output_.size()) - size_);
        size_ = end;
        return start;
    }

    /** Stores @p value little-endian at @p position, if the output reaches that far. */
    template <typename scalar_type_>
    void store(std::size_t position, scalar_type_ value) noexcept {
        if (position + sizeof(scalar_type_) <= output_.size()) ipc_store_little(output_.data() + position, value);
    }

    /** Points the @c uoffset at @p slot to @p target, which lies after it. */
    void link(std::size_t slot, std::size_t target) noexcept { store(slot, static_cast<std::uint32_t>(target - slot)); }

    /** Lays out a vtable and its table holding @p fields, widest first, and answers in @p positions
     *  where each lands, and as the result where the table starts. */
    std::size_t table(std::span<ipc_flatbuffer_slot_t const> fields, std::span<std::size_t> positions) noexcept {
        std::size_t slots = 0, table_bytes = 4;
        for (ipc_flatbuffer_slot_t const& field : fields) {
            slots = field.slot + 1u > slots ? field.slot + 1u : slots;
            table_bytes += field.bytes;
        }
        bool const has_longs = !fields.empty() && fields.front().bytes == 8;
        std::size_t const vtable_bytes = 4 + 2 * slots;
        std::size_t const vtable = claim(vtable_bytes, 2);
        // A table at 4 modulo 8 puts its longs, past the 4-byte `soffset`, on 8-byte boundaries
        std::size_t const table = claim(table_bytes, has_longs ? 8 : 4, has_longs ? 4 : 0);
        store(vtable, static_cast<std::uint16_t>(vtable_bytes));
        store(vtable + 2, static_cast<std::uint16_t>(table_bytes));
        store(table, static_cast<std::int32_t>(table - vtable));
        std::size_t offset = 4;
        for (std::size_t index = 0; index != fields.size(); ++index) {
            store(vtable + 4 + 2 * std::size_t(fields[index].slot), static_cast<std::uint16_t>(offset));
            positions[index] = table + offset;
            offset += fields[index].bytes;
        }
        return table;
    }

    /** Lays out @p text with its length and terminator, and answers where it starts. */
    std::size_t string(std::string_view text) noexcept {
        std::size_t const start = claim(4 + text.size() + 1, 4);
        store(start, static_cast<std::uint32_t>(text.size()));
        if (start + 4 + text.size() <= output_.size() && !text.empty())
            std::memcpy(output_.data() + start + 4, text.data(), text.size());
        return start;
    }

    /** Lays out a vector of @p count zeroed elements of @p element_bytes, those holding longs
     *  8-aligned, and answers where its count starts, 4 bytes before the first element. */
    std::size_t vector(std::size_t count, std::size_t element_bytes) noexcept {
        bool const has_longs = element_bytes % 8 == 0;
        std::size_t const start = claim(4 + count * element_bytes, has_longs ? 8 : 4, has_longs ? 4 : 0);
        store(start, static_cast<std::uint32_t>(count));
        return start;
    }
};

#pragma endregion FlatBuffers

#pragma region Layouts

/** Whether @p type holds byte strings behind offsets. */
constexpr bool ipc_type_is_variable(ipc_type_t type) noexcept {
    return type == ipc_type_t::binary_k || type == ipc_type_t::large_binary_k || type == ipc_type_t::utf8_k ||
           type == ipc_type_t::large_utf8_k;
}

/** Whether @p type holds lists behind offsets. */
constexpr bool ipc_type_is_list(ipc_type_t type) noexcept {
    return type == ipc_type_t::list_k || type == ipc_type_t::large_list_k;
}

/** Whether @p type keeps its values in one data buffer of fixed-width values. */
constexpr bool ipc_type_is_fixed(ipc_type_t type) noexcept {
    return type == ipc_type_t::fixed_size_binary_k || type == ipc_type_t::int_k || type == ipc_type_t::floating_point_k;
}

/** Bytes of one offset of a byte string or list type. */
constexpr std::size_t ipc_offset_bytes(ipc_type_t type) noexcept {
    return type == ipc_type_t::large_binary_k || type == ipc_type_t::large_utf8_k || type == ipc_type_t::large_list_k
               ? 8
               : 4;
}

/** Bytes of one value of a field of fixed-width values. */
constexpr std::size_t ipc_value_bytes(ipc_field_t const& field) noexcept {
    std::size_t const width = static_cast<std::size_t>(field.width);
    return field.type == ipc_type_t::fixed_size_binary_k ? width : width / 8;
}

/** Buffers a field of @p type takes in a batch, its validity bitmap included, its children's
 *  not. */
constexpr std::size_t ipc_buffers_of(ipc_type_t type) noexcept {
    if (ipc_type_is_variable(type)) return 3;
    return ipc_type_is_fixed(type) || ipc_type_is_list(type) ? 2 : 1;
}

/** Fields of @p fields with every nested one, as many as a batch holds nodes. */
constexpr std::size_t ipc_fields_of(std::span<ipc_field_t const> fields) noexcept {
    std::size_t count = fields.size();
    for (ipc_field_t const& field : fields) count += ipc_fields_of(field.children);
    return count;
}

/** Buffers @p fields take in a batch, with those of every nested field. */
constexpr std::size_t ipc_buffers_of(std::span<ipc_field_t const> fields) noexcept {
    std::size_t count = 0;
    for (ipc_field_t const& field : fields) count += ipc_buffers_of(field.type) + ipc_buffers_of(field.children);
    return count;
}

/** Checks that a field the caller built, with its children, is one the codec writes. */
constexpr ipc_status_t ipc_check_field(ipc_field_t const& field) noexcept {
    std::int32_t const width = field.width;
    std::size_t const children = field.children.size();
    bool fits = false;
    switch (field.type) {
    case ipc_type_t::binary_k:
    case ipc_type_t::large_binary_k:
    case ipc_type_t::utf8_k:
    case ipc_type_t::large_utf8_k: fits = children == 0; break;
    case ipc_type_t::fixed_size_binary_k: fits = children == 0 && width > 0; break;
    case ipc_type_t::int_k: fits = children == 0 && (width == 8 || width == 16 || width == 32 || width == 64); break;
    case ipc_type_t::floating_point_k: fits = children == 0 && (width == 16 || width == 32 || width == 64); break;
    case ipc_type_t::fixed_size_list_k: fits = children == 1 && width > 0; break;
    case ipc_type_t::list_k:
    case ipc_type_t::large_list_k: fits = children == 1; break;
    case ipc_type_t::struct_k: fits = true; break;
    default: return ipc_status_t::unsupported_k;
    }
    if (!fits) return ipc_status_t::malformed_k;
    for (ipc_field_t const& child : field.children)
        if (ipc_status_t const status = ipc_check_field(child); status != ipc_status_t::success_k) return status;
    return ipc_status_t::success_k;
}

/** Checks that a schema the caller built is one the codec writes. */
constexpr ipc_status_t ipc_check_schema(ipc_schema_t const& schema) noexcept {
    if (ipc_fields_of(schema.fields) > ipc_fields_max_k) return ipc_status_t::unsupported_k;
    for (ipc_field_t const& field : schema.fields)
        if (ipc_status_t const status = ipc_check_field(field); status != ipc_status_t::success_k) return status;
    return ipc_status_t::success_k;
}

/** Checks a writer's alignment and the absolute @p position its message starts at, which must keep
 *  8-byte order. */
constexpr ipc_status_t ipc_check_layout(ipc_options_t options, std::uint64_t position) noexcept {
    std::uint32_t const alignment = options.alignment;
    if (alignment < 8 || alignment > ipc_alignment_direct_k || (alignment & (alignment - 1)))
        return ipc_status_t::unsupported_k;
    return position % 8 ? ipc_status_t::unsupported_k : ipc_status_t::success_k;
}

#pragma endregion Layouts

#pragma region Reading

/** Reads the custom metadata vector of @p slot into the next entries of @p pool. */
inline ipc_status_t ipc_read_key_values(ipc_flatbuffer_table_t const& table, std::uint16_t slot,
                                        std::span<ipc_key_value_t> pool, std::size_t& used,
                                        std::span<ipc_key_value_t const>& pairs) noexcept {
    ipc_status_t status = ipc_status_t::success_k;
    ipc_flatbuffer_vector_t const vector = table.vector(slot, 4, status);
    if (status != ipc_status_t::success_k) return status;
    if (vector.count > pool.size() - used) return ipc_status_t::too_small_k;
    for (std::size_t index = 0; index != vector.count; ++index) {
        ipc_flatbuffer_table_t const pair(table.flatbuffer(), vector.first + 4 * index, status);
        pool[used + index] = {pair.string(ipc_key_value_key_k, status), pair.string(ipc_key_value_value_k, status)};
    }
    pairs = pool.subspan(used, vector.count);
    used += vector.count;
    return status;
}

/** Maps the @c type union of a @c Field table onto @p field, refusing what the codec does not
 *  handle. */
inline ipc_status_t ipc_read_type(ipc_flatbuffer_table_t const& table, ipc_field_t& field) noexcept {
    ipc_status_t status = ipc_status_t::success_k;
    std::uint8_t const type_type = table.scalar<std::uint8_t>(ipc_field_type_type_k, 0, status);
    ipc_flatbuffer_table_t const type = table.table(ipc_field_type_k, status);
    field.width = 0;
    field.signedness = ipc_signedness_t::unsigned_k;
    switch (type_type) {
    case ipc_field_type_binary_k: field.type = ipc_type_t::binary_k; break;
    case ipc_field_type_large_binary_k: field.type = ipc_type_t::large_binary_k; break;
    case ipc_field_type_utf8_k: field.type = ipc_type_t::utf8_k; break;
    case ipc_field_type_large_utf8_k: field.type = ipc_type_t::large_utf8_k; break;
    case ipc_field_type_list_k: field.type = ipc_type_t::list_k; break;
    case ipc_field_type_large_list_k: field.type = ipc_type_t::large_list_k; break;
    case ipc_field_type_struct_k: field.type = ipc_type_t::struct_k; break;
    case ipc_field_type_fixed_size_binary_k:
        field.type = ipc_type_t::fixed_size_binary_k;
        field.width = type.scalar<std::int32_t>(ipc_fixed_size_binary_byte_width_k, 0, status);
        break;
    case ipc_field_type_fixed_size_list_k:
        field.type = ipc_type_t::fixed_size_list_k;
        field.width = type.scalar<std::int32_t>(ipc_fixed_size_list_list_size_k, 0, status);
        break;
    case ipc_field_type_int_k: {
        field.type = ipc_type_t::int_k;
        field.width = type.scalar<std::int32_t>(ipc_int_bit_width_k, 0, status);
        bool const is_signed = type.scalar<std::uint8_t>(ipc_int_is_signed_k, 0, status) != 0;
        field.signedness = is_signed ? ipc_signedness_t::signed_k : ipc_signedness_t::unsigned_k;
        break;
    }
    case ipc_field_type_floating_point_k: {
        field.type = ipc_type_t::floating_point_k;
        std::int16_t const precision = type.scalar<std::int16_t>(ipc_floating_point_precision_k, 0, status);
        if (status == ipc_status_t::success_k && (precision < 0 || precision > 2)) return ipc_status_t::malformed_k;
        field.width = 16 << precision;
        break;
    }
    case ipc_field_type_none_k: return status == ipc_status_t::success_k ? ipc_status_t::malformed_k : status;
    default: return status == ipc_status_t::success_k ? ipc_status_t::unsupported_k : status;
    }
    return status;
}

/** Where a schema being read puts its fields and its pairs of metadata, and how many of each it
 *  took. */
struct ipc_schema_pools_t {
    std::span<ipc_field_t> fields;
    std::span<ipc_key_value_t> pairs;
    std::size_t fields_used = 0;
    std::size_t pairs_used = 0;
};

inline ipc_status_t ipc_read_field(ipc_flatbuffer_table_t const& table, ipc_schema_pools_t& pools,
                                   ipc_field_t& field) noexcept;

/** Reads the vector of @c Field tables in @p slot into the next fields of @p pools, all siblings
 *  side by side, then each one's children after them, and answers the siblings in @p fields. */
inline ipc_status_t ipc_read_fields(ipc_flatbuffer_table_t const& parent, std::uint16_t slot, ipc_schema_pools_t& pools,
                                    std::span<ipc_field_t const>& fields) noexcept {
    ipc_status_t status = ipc_status_t::success_k;
    ipc_flatbuffer_vector_t const vector = parent.vector(slot, 4, status);
    if (status != ipc_status_t::success_k) return status;
    // Capping the fields also caps the depth, since every level of nesting takes one
    if (vector.count > ipc_fields_max_k - pools.fields_used) return ipc_status_t::unsupported_k;
    if (vector.count > pools.fields.size() - pools.fields_used) return ipc_status_t::too_small_k;
    std::span<ipc_field_t> const siblings = pools.fields.subspan(pools.fields_used, vector.count);
    pools.fields_used += vector.count;
    for (std::size_t index = 0; index != vector.count; ++index) {
        ipc_flatbuffer_table_t const field(parent.flatbuffer(), vector.first + 4 * index, status);
        if (status == ipc_status_t::success_k) status = ipc_read_field(field, pools, siblings[index]);
        if (status != ipc_status_t::success_k) return status;
    }
    fields = siblings;
    return ipc_status_t::success_k;
}

/** Reads one @c Field table into @p field, its metadata and children into the next entries of
 *  @p pools. */
inline ipc_status_t ipc_read_field(ipc_flatbuffer_table_t const& table, ipc_schema_pools_t& pools,
                                   ipc_field_t& field) noexcept {
    ipc_status_t status = ipc_status_t::success_k;
    field.name = table.string(ipc_field_name_k, status);
    bool const nullable = table.scalar<std::uint8_t>(ipc_field_nullable_k, 0, status) != 0;
    field.nullability = nullable ? ipc_nullability_t::nullable_k : ipc_nullability_t::non_nullable_k;
    std::size_t const dictionary = table.field(ipc_field_dictionary_k, 4, status);
    if (status != ipc_status_t::success_k) return status;
    if (dictionary) return ipc_status_t::unsupported_k;
    status = ipc_read_type(table, field);
    if (status == ipc_status_t::success_k)
        status = ipc_read_key_values(table, ipc_field_custom_metadata_k, pools.pairs, pools.pairs_used, field.metadata);
    if (status == ipc_status_t::success_k) status = ipc_read_fields(table, ipc_field_children_k, pools, field.children);
    return status;
}

/** Reads a @c Schema table, from a message or a footer, into the entries the caller lent. */
inline ipc_status_t ipc_read_schema_table(ipc_flatbuffer_table_t const& table, std::span<ipc_field_t> fields,
                                          std::span<ipc_key_value_t> metadata, ipc_schema_t& schema) noexcept {
    ipc_status_t status = ipc_status_t::success_k;
    std::int16_t const endianness = table.scalar<std::int16_t>(ipc_schema_endianness_k, 0, status);
    if (status != ipc_status_t::success_k) return status;
    if (endianness != 0) return ipc_status_t::unsupported_k;
    ipc_schema_pools_t pools {fields, metadata};
    status = ipc_read_fields(table, ipc_schema_fields_k, pools, schema.fields);
    if (status == ipc_status_t::success_k)
        status = ipc_read_key_values(table, ipc_schema_custom_metadata_k, pools.pairs, pools.pairs_used,
                                     schema.metadata);
    return status == ipc_status_t::success_k ? ipc_check_schema(schema) : status;
}

/** Whether the codec reads a @c MetadataVersion: V4, from format 0.15, and V5. */
constexpr bool ipc_reads_version(std::int16_t version) noexcept {
    return version == ipc_metadata_version_4_k || version == ipc_metadata_version_5_k;
}

/** One message's metadata: its kind, its body's length and its header table, viewing the metadata
 *  it was read from. */
struct ipc_message_t {
    ipc_message_kind_t kind = ipc_message_kind_t::schema_k;
    std::int64_t body_bytes = 0;
    ipc_flatbuffer_table_t header;
};

/**
 *  @brief Reads the prefix of a message from the first bytes of @p bytes: the <tt>0xFFFFFFFF</tt>
 *      continuation marker and an @c int32 length, or the bare length writers before 0.15 emit.
 *
 *  @p prefix_bytes always names the bytes the prefix needs, 4 and then 8 once a marker shows, so a
 *  pipe reader that gets @c ipc_status_t::truncated_k reads up to that many and calls again.
 *  @p metadata_bytes is zero at the end of the stream.
 */
inline ipc_status_t ipc_read_prefix(bytes_view_t bytes, std::uint32_t& metadata_bytes,
                                    std::size_t& prefix_bytes) noexcept {
    metadata_bytes = 0, prefix_bytes = 4;
    if (bytes.size() < 4) return ipc_status_t::truncated_k;
    std::uint32_t const first = ipc_load_little<std::uint32_t>(bytes.data());
    if (first == 0xFFFFFFFFu) {
        prefix_bytes = 8;
        if (bytes.size() < 8) return ipc_status_t::truncated_k;
        std::int32_t const length = ipc_load_little<std::int32_t>(bytes.data() + 4);
        if (length < 0) return ipc_status_t::malformed_k;
        metadata_bytes = static_cast<std::uint32_t>(length);
        return ipc_status_t::success_k;
    }
    if (static_cast<std::int32_t>(first) < 0) return ipc_status_t::malformed_k;
    metadata_bytes = first;
    return ipc_status_t::success_k;
}

/** Reads the @c Message in the @p metadata the prefix framed, which outlives @p message. The end of
 *  the stream is the zero length @c ipc_read_prefix answers, so empty metadata is truncated. */
inline ipc_status_t ipc_read_message(bytes_view_t metadata, ipc_message_t& message) noexcept {
    ipc_status_t status = ipc_status_t::success_k;
    ipc_flatbuffer_table_t const root(metadata, 0, status);
    std::int16_t const version = root.scalar<std::int16_t>(ipc_message_version_k, 0, status);
    std::uint8_t const header_type = root.scalar<std::uint8_t>(ipc_message_header_type_k, 0, status);
    message.body_bytes = root.scalar<std::int64_t>(ipc_message_body_length_k, 0, status);
    message.header = root.table(ipc_message_header_k, status);
    if (status != ipc_status_t::success_k) return status;
    if (!ipc_reads_version(version)) return ipc_status_t::unsupported_k;
    if (message.body_bytes < 0) return ipc_status_t::malformed_k;
    switch (header_type) {
    case ipc_message_header_schema_k: message.kind = ipc_message_kind_t::schema_k; break;
    case ipc_message_header_record_batch_k: message.kind = ipc_message_kind_t::record_batch_k; break;
    case ipc_message_header_dictionary_batch_k:
    case ipc_message_header_tensor_k:
    case ipc_message_header_sparse_tensor_k: return ipc_status_t::unsupported_k;
    default: return ipc_status_t::malformed_k;
    }
    return message.header.present() ? ipc_status_t::success_k : ipc_status_t::malformed_k;
}

/** Reads the schema a message carries: its fields into @p fields, the columns first and every
 *  nested field after, each field's children side by side, and its pairs into @p metadata. */
inline ipc_status_t ipc_read_schema(ipc_message_t const& message, std::span<ipc_field_t> fields,
                                    std::span<ipc_key_value_t> metadata, ipc_schema_t& schema) noexcept {
    if (message.kind != ipc_message_kind_t::schema_k) return ipc_status_t::malformed_k;
    return ipc_read_schema_table(message.header, fields, metadata, schema);
}

/**
 *  @brief Reads the record batch a message carries, its nodes into @p nodes and views of its
 *      buffers into @p buffers, each checked to lie inside @p body.
 *
 *  Compression is refused before the body is needed. The values themselves are left to
 *  @c ipc_validate_batch.
 */
inline ipc_status_t ipc_read_batch(ipc_message_t const& message, bytes_view_t body, ipc_schema_t const& schema,
                                   std::span<ipc_node_t> nodes, std::span<bytes_view_t> buffers,
                                   ipc_batch_t& batch) noexcept {
    if (message.kind != ipc_message_kind_t::record_batch_k) return ipc_status_t::malformed_k;
    ipc_flatbuffer_table_t const& header = message.header;
    ipc_status_t status = ipc_status_t::success_k;
    std::size_t const compression = header.field(ipc_record_batch_compression_k, 4, status);
    std::size_t const variadic = header.vector(ipc_record_batch_variadic_buffer_counts_k, 8, status).count;
    std::int64_t const length = header.scalar<std::int64_t>(ipc_record_batch_length_k, 0, status);
    ipc_flatbuffer_vector_t const node_vector = header.vector(ipc_record_batch_nodes_k, ipc_flatbuffer_pair_bytes_k,
                                                              status);
    ipc_flatbuffer_vector_t const buffer_vector = header.vector(ipc_record_batch_buffers_k, ipc_flatbuffer_pair_bytes_k,
                                                                status);
    if (status != ipc_status_t::success_k) return status;
    if (compression || variadic) return ipc_status_t::unsupported_k;
    if (static_cast<std::uint64_t>(message.body_bytes) > body.size()) return ipc_status_t::truncated_k;
    if (length < 0 || node_vector.count != ipc_fields_of(schema.fields) ||
        buffer_vector.count != ipc_buffers_of(schema.fields))
        return ipc_status_t::malformed_k;
    if (node_vector.count > nodes.size() || buffer_vector.count > buffers.size()) return ipc_status_t::too_small_k;
    body = body.first(static_cast<std::size_t>(message.body_bytes));

    std::byte const* const metadata = header.flatbuffer().data();
    for (std::size_t index = 0; index != node_vector.count; ++index) {
        std::byte const* const node = metadata + node_vector.first + ipc_flatbuffer_pair_bytes_k * index;
        nodes[index] = {ipc_load_little<std::int64_t>(node), ipc_load_little<std::int64_t>(node + 8)};
    }
    for (std::size_t index = 0; index != buffer_vector.count; ++index) {
        std::byte const* const buffer = metadata + buffer_vector.first + ipc_flatbuffer_pair_bytes_k * index;
        std::int64_t const offset = ipc_load_little<std::int64_t>(buffer);
        std::int64_t const bytes = ipc_load_little<std::int64_t>(buffer + 8);
        if (offset < 0 || bytes < 0 || static_cast<std::uint64_t>(offset) > body.size() ||
            static_cast<std::uint64_t>(bytes) > body.size() - static_cast<std::uint64_t>(offset))
            return ipc_status_t::malformed_k;
        buffers[index] = body.subspan(static_cast<std::size_t>(offset), static_cast<std::size_t>(bytes));
    }
    batch = {length, nodes.first(node_vector.count), buffers.first(buffer_vector.count)};
    return ipc_status_t::success_k;
}

/** Checks that the @p offsets of @p length strings or lists climb from a non-negative start and end
 *  at most at @p limit, the bytes of their data or the rows of their child. */
template <typename offset_type_>
constexpr ipc_status_t ipc_validate_offsets(bytes_view_t offsets, std::uint64_t limit, std::uint64_t length) noexcept {
    if (length == 0 && offsets.empty()) return ipc_status_t::success_k;
    if (offsets.size() / sizeof(offset_type_) <= length) return ipc_status_t::malformed_k;
    offset_type_ previous = ipc_load_little<offset_type_>(offsets.data());
    if (previous < 0) return ipc_status_t::malformed_k;
    for (std::uint64_t index = 1; index <= length; ++index) {
        offset_type_ const offset = ipc_load_little<offset_type_>(offsets.data() + index * sizeof(offset_type_));
        if (offset < previous) return ipc_status_t::malformed_k;
        previous = offset;
    }
    return static_cast<std::uint64_t>(previous) <= limit ? ipc_status_t::success_k : ipc_status_t::malformed_k;
}

/** How far a walk over the nodes and buffers of a batch, in the depth-first order of its fields,
 *  has come. */
struct ipc_batch_cursor_t {
    std::size_t node = 0;
    std::size_t buffer = 0;
};

/** Checks the values of @p field, whose node must hold at least @p needed rows, and of its
 *  children, moving @p cursor past all of them. */
constexpr ipc_status_t ipc_validate_field(ipc_field_t const& field, ipc_batch_t const& batch, std::uint64_t needed,
                                          ipc_batch_cursor_t& cursor) noexcept {
    ipc_node_t const& node = batch.nodes[cursor.node++];
    if (node.length < 0 || node.null_count < 0 || static_cast<std::uint64_t>(node.length) < needed)
        return ipc_status_t::malformed_k;
    if (node.null_count != 0) return ipc_status_t::unsupported_k;
    std::uint64_t const length = static_cast<std::uint64_t>(node.length);
    std::size_t const first = cursor.buffer + 1; // Past the validity bitmap, which no null reads
    cursor.buffer += ipc_buffers_of(field.type);
    bool const wide = ipc_offset_bytes(field.type) == 8;
    ipc_status_t status = ipc_status_t::success_k;
    std::uint64_t child_needed = length;
    if (ipc_type_is_variable(field.type)) {
        std::uint64_t const limit = batch.buffers[first + 1].size();
        status = wide ? ipc_validate_offsets<std::int64_t>(batch.buffers[first], limit, length)
                      : ipc_validate_offsets<std::int32_t>(batch.buffers[first], limit, length);
    }
    else if (ipc_type_is_fixed(field.type)) {
        if (batch.buffers[first].size() / ipc_value_bytes(field) < length) status = ipc_status_t::malformed_k;
    }
    else if (ipc_type_is_list(field.type)) {
        std::int64_t const child_length = batch.nodes[cursor.node].length;
        std::uint64_t const limit = child_length < 0 ? 0 : static_cast<std::uint64_t>(child_length);
        status = wide ? ipc_validate_offsets<std::int64_t>(batch.buffers[first], limit, length)
                      : ipc_validate_offsets<std::int32_t>(batch.buffers[first], limit, length);
        child_needed = 0;
    }
    else if (field.type == ipc_type_t::fixed_size_list_k) {
        std::uint64_t const width = static_cast<std::uint64_t>(field.width);
        if (length > ~std::uint64_t(0) / width) status = ipc_status_t::malformed_k;
        child_needed = length * width;
    }
    for (std::size_t index = 0; status == ipc_status_t::success_k && index != field.children.size(); ++index)
        status = ipc_validate_field(field.children[index], batch, child_needed, cursor);
    return status;
}

/** Checks the values of a batch against its schema: a node per field without nulls, every column as
 *  long as the batch and every child long enough for its parent, offsets that climb and end inside
 *  their data or child, and fixed-width data long enough. UTF-8 is not checked, as for bytes. */
constexpr ipc_status_t ipc_validate_batch(ipc_schema_t const& schema, ipc_batch_t const& batch) noexcept {
    if (ipc_status_t const status = ipc_check_schema(schema); status != ipc_status_t::success_k) return status;
    if (batch.length < 0 || batch.nodes.size() != ipc_fields_of(schema.fields) ||
        batch.buffers.size() != ipc_buffers_of(schema.fields))
        return ipc_status_t::malformed_k;
    ipc_batch_cursor_t cursor;
    for (ipc_field_t const& field : schema.fields) {
        if (batch.nodes[cursor.node].length != batch.length) return ipc_status_t::malformed_k;
        ipc_status_t const status = ipc_validate_field(field, batch, static_cast<std::uint64_t>(batch.length), cursor);
        if (status != ipc_status_t::success_k) return status;
    }
    return ipc_status_t::success_k;
}

/**
 *  @brief Reads the footer at the end of a whole @p file: its schema into the entries lent, and the
 *      Blocks of its record batches into @p blocks, so a mapped file imports them in parallel.
 *
 *  @p blocks_count names how many Blocks the footer holds, also when @p blocks is too short for
 *  them. A file that starts with the magic but does not end with it reads as truncated.
 */
inline ipc_status_t ipc_read_footer(bytes_view_t file, std::span<ipc_field_t> fields,
                                    std::span<ipc_key_value_t> metadata, ipc_schema_t& schema,
                                    std::span<ipc_block_t> blocks, std::size_t& blocks_count) noexcept {
    blocks_count = 0;
    std::size_t const tail_bytes = 4 + 6;
    if (file.size() < ipc_file_magic_k.size()) return ipc_status_t::truncated_k;
    if (std::memcmp(file.data(), ipc_file_magic_k.data(), ipc_file_magic_k.size()) != 0)
        return ipc_status_t::malformed_k;
    if (file.size() < ipc_file_magic_k.size() + tail_bytes ||
        std::memcmp(file.data() + file.size() - 6, ipc_file_magic_k.data(), 6) != 0)
        return ipc_status_t::truncated_k;
    std::int32_t const footer_bytes = ipc_load_little<std::int32_t>(file.data() + file.size() - tail_bytes);
    if (footer_bytes < 0 || static_cast<std::size_t>(footer_bytes) > file.size() - ipc_file_magic_k.size() - tail_bytes)
        return ipc_status_t::malformed_k;
    std::size_t const footer_start = file.size() - tail_bytes - static_cast<std::size_t>(footer_bytes);

    ipc_status_t status = ipc_status_t::success_k;
    ipc_flatbuffer_table_t const root(file.subspan(footer_start, static_cast<std::size_t>(footer_bytes)), 0, status);
    std::int16_t const version = root.scalar<std::int16_t>(ipc_footer_version_k, 0, status);
    ipc_flatbuffer_table_t const schema_table = root.table(ipc_footer_schema_k, status);
    std::size_t const dictionaries = root.vector(ipc_footer_dictionaries_k, ipc_flatbuffer_block_bytes_k, status).count;
    ipc_flatbuffer_vector_t const vector = root.vector(ipc_footer_record_batches_k, ipc_flatbuffer_block_bytes_k,
                                                       status);
    if (status != ipc_status_t::success_k) return status;
    if (!ipc_reads_version(version) || dictionaries) return ipc_status_t::unsupported_k;
    if (!schema_table.present()) return ipc_status_t::malformed_k;
    status = ipc_read_schema_table(schema_table, fields, metadata, schema);
    if (status != ipc_status_t::success_k) return status;
    blocks_count = vector.count;
    if (vector.count > blocks.size()) return ipc_status_t::too_small_k;
    std::byte const* const footer = file.data() + footer_start;
    for (std::size_t index = 0; index != vector.count; ++index) {
        std::byte const* const entry = footer + vector.first + ipc_flatbuffer_block_bytes_k * index;
        ipc_block_t& block = blocks[index];
        block.offset = ipc_load_little<std::int64_t>(entry);
        block.metadata_bytes = ipc_load_little<std::int32_t>(entry + 8);
        block.body_bytes = ipc_load_little<std::int64_t>(entry + 16);
        std::uint64_t const limit = footer_start;
        if (block.offset < static_cast<std::int64_t>(ipc_file_magic_k.size()) || block.metadata_bytes < 0 ||
            block.body_bytes < 0 || static_cast<std::uint64_t>(block.offset) > limit ||
            static_cast<std::uint64_t>(block.metadata_bytes) > limit - static_cast<std::uint64_t>(block.offset) ||
            static_cast<std::uint64_t>(block.body_bytes) >
                limit - static_cast<std::uint64_t>(block.offset) - static_cast<std::uint64_t>(block.metadata_bytes))
            return ipc_status_t::malformed_k;
    }
    return ipc_status_t::success_k;
}

#pragma endregion Reading

#pragma region Writing

/** Lays out custom metadata as a vector of @c KeyValue tables and answers where it starts. */
inline std::size_t ipc_build_key_values(ipc_flatbuffer_builder_t& builder,
                                        std::span<ipc_key_value_t const> pairs) noexcept {
    ipc_flatbuffer_slot_t const slots[] {{ipc_key_value_key_k, 4}, {ipc_key_value_value_k, 4}};
    std::size_t const vector = builder.vector(pairs.size(), 4);
    for (std::size_t index = 0; index != pairs.size(); ++index) {
        std::size_t positions[2];
        builder.link(vector + 4 + 4 * index, builder.table(slots, positions));
        builder.link(positions[0], builder.string(pairs[index].key));
        builder.link(positions[1], builder.string(pairs[index].value));
    }
    return vector;
}

/** The member of the @c Type union a field of @p type writes. */
constexpr ipc_field_type_t ipc_field_type_of(ipc_type_t type) noexcept {
    switch (type) {
    case ipc_type_t::binary_k: return ipc_field_type_binary_k;
    case ipc_type_t::large_binary_k: return ipc_field_type_large_binary_k;
    case ipc_type_t::utf8_k: return ipc_field_type_utf8_k;
    case ipc_type_t::large_utf8_k: return ipc_field_type_large_utf8_k;
    case ipc_type_t::fixed_size_binary_k: return ipc_field_type_fixed_size_binary_k;
    case ipc_type_t::int_k: return ipc_field_type_int_k;
    case ipc_type_t::floating_point_k: return ipc_field_type_floating_point_k;
    case ipc_type_t::fixed_size_list_k: return ipc_field_type_fixed_size_list_k;
    case ipc_type_t::list_k: return ipc_field_type_list_k;
    case ipc_type_t::large_list_k: return ipc_field_type_large_list_k;
    case ipc_type_t::struct_k: return ipc_field_type_struct_k;
    }
    return ipc_field_type_none_k;
}

/** Lays out the table of a field's @c type union and answers where it starts. */
inline std::size_t ipc_build_type(ipc_flatbuffer_builder_t& builder, ipc_field_t const& field) noexcept {
    std::size_t positions[2];
    switch (field.type) {
    case ipc_type_t::fixed_size_binary_k: {
        ipc_flatbuffer_slot_t const slots[] {{ipc_fixed_size_binary_byte_width_k, 4}};
        std::size_t const table = builder.table(slots, positions);
        builder.store(positions[0], field.width);
        return table;
    }
    case ipc_type_t::fixed_size_list_k: {
        ipc_flatbuffer_slot_t const slots[] {{ipc_fixed_size_list_list_size_k, 4}};
        std::size_t const table = builder.table(slots, positions);
        builder.store(positions[0], field.width);
        return table;
    }
    case ipc_type_t::int_k: {
        ipc_flatbuffer_slot_t const slots[] {{ipc_int_bit_width_k, 4}, {ipc_int_is_signed_k, 1}};
        std::size_t const table = builder.table(slots, positions);
        builder.store(positions[0], field.width);
        builder.store(positions[1], std::uint8_t(field.signedness == ipc_signedness_t::signed_k));
        return table;
    }
    case ipc_type_t::floating_point_k: {
        ipc_flatbuffer_slot_t const slots[] {{ipc_floating_point_precision_k, 2}};
        std::size_t const table = builder.table(slots, positions);
        builder.store(positions[0], static_cast<std::int16_t>(field.width == 16 ? 0 : field.width == 32 ? 1 : 2));
        return table;
    }
    default: return builder.table({}, positions);
    }
}

inline std::size_t ipc_build_field(ipc_flatbuffer_builder_t& builder, ipc_field_t const& field) noexcept;

/** Lays out a vector of @c Field tables, each followed by its own children, and answers where it
 *  starts. */
inline std::size_t ipc_build_fields(ipc_flatbuffer_builder_t& builder, std::span<ipc_field_t const> fields) noexcept {
    std::size_t const vector = builder.vector(fields.size(), 4);
    for (std::size_t index = 0; index != fields.size(); ++index)
        builder.link(vector + 4 + 4 * index, ipc_build_field(builder, fields[index]));
    return vector;
}

/** Lays out a @c Field table, whose @c children vector is written even when empty as Arrow's own
 *  readers insist, and answers where it starts. */
inline std::size_t ipc_build_field(ipc_flatbuffer_builder_t& builder, ipc_field_t const& field) noexcept {
    ipc_flatbuffer_slot_t const slots[] {{ipc_field_name_k, 4},     {ipc_field_type_k, 4},
                                         {ipc_field_children_k, 4}, {ipc_field_custom_metadata_k, 4},
                                         {ipc_field_nullable_k, 1}, {ipc_field_type_type_k, 1}};
    std::size_t positions[6];
    std::size_t const table = builder.table(slots, positions);
    builder.store(positions[4], std::uint8_t(field.nullability == ipc_nullability_t::nullable_k));
    builder.store(positions[5], static_cast<std::uint8_t>(ipc_field_type_of(field.type)));
    builder.link(positions[0], builder.string(field.name));
    builder.link(positions[1], ipc_build_type(builder, field));
    builder.link(positions[2], ipc_build_fields(builder, field.children));
    builder.link(positions[3], ipc_build_key_values(builder, field.metadata));
    return table;
}

/** Lays out a little-endian @c Schema table and answers where it starts. */
inline std::size_t ipc_build_schema(ipc_flatbuffer_builder_t& builder, ipc_schema_t const& schema) noexcept {
    ipc_flatbuffer_slot_t const slots[] {{ipc_schema_fields_k, 4},
                                         {ipc_schema_custom_metadata_k, 4},
                                         {ipc_schema_endianness_k, 2}};
    std::size_t positions[3];
    std::size_t const table = builder.table(slots, positions);
    builder.link(positions[0], ipc_build_fields(builder, schema.fields));
    builder.link(positions[1], ipc_build_key_values(builder, schema.metadata));
    return table;
}

/** Lays out the root @c uoffset and a V5 @c Message table, and answers the @c header slot for the
 *  caller to link. */
inline std::size_t ipc_build_message(ipc_flatbuffer_builder_t& builder, ipc_message_header_t header_type,
                                     std::uint64_t body_bytes) noexcept {
    ipc_flatbuffer_slot_t const slots[] {{ipc_message_body_length_k, 8},
                                         {ipc_message_header_k, 4},
                                         {ipc_message_version_k, 2},
                                         {ipc_message_header_type_k, 1}};
    std::size_t positions[4];
    std::size_t const root = builder.claim(4, 4);
    builder.link(root, builder.table(slots, positions));
    builder.store(positions[0], static_cast<std::int64_t>(body_bytes));
    builder.store(positions[2], static_cast<std::int16_t>(ipc_metadata_version_5_k));
    builder.store(positions[3], static_cast<std::uint8_t>(header_type));
    return positions[1];
}

/** Frames metadata the builder laid after the prefix in @p output: pads it so the message ends on
 *  the alignment at its absolute file @p position, then writes the marker and the length. */
inline ipc_status_t ipc_frame_message(bytes_span_t output, std::uint64_t position, ipc_options_t options,
                                      ipc_flatbuffer_builder_t& builder, std::size_t& written) noexcept {
    std::uint64_t const end = ipc_round_up(position + 8 + builder.size(), options.alignment);
    written = static_cast<std::size_t>(end - position);
    builder.claim(written - 8 - builder.size(), 1);
    if (written > output.size()) return ipc_status_t::too_small_k;
    ipc_store_little(output.data(), std::uint32_t(0xFFFFFFFFu));
    ipc_store_little(output.data() + 4, static_cast<std::int32_t>(written - 8));
    return ipc_status_t::success_k;
}

/** The output past the 8 bytes of a message's prefix, where its metadata goes. */
inline bytes_span_t ipc_past_prefix(bytes_span_t output) noexcept {
    return output.size() > 8 ? output.subspan(8) : bytes_span_t {};
}

/**
 *  @brief Writes a schema message for absolute file @p position into @p output, answering its bytes
 *      in @p written.
 *
 *  A short @p output answers @c ipc_status_t::too_small_k with the bytes needed in @p written, so
 *  an empty one measures.
 */
inline ipc_status_t ipc_write_schema(ipc_schema_t const& schema, ipc_options_t options, std::uint64_t position,
                                     bytes_span_t output, std::size_t& written) noexcept {
    written = 0;
    if (ipc_status_t const status = ipc_check_schema(schema); status != ipc_status_t::success_k) return status;
    if (ipc_status_t const status = ipc_check_layout(options, position); status != ipc_status_t::success_k)
        return status;
    ipc_flatbuffer_builder_t builder(ipc_past_prefix(output));
    std::size_t const header = ipc_build_message(builder, ipc_message_header_schema_k, 0);
    builder.link(header, ipc_build_schema(builder, schema));
    return ipc_frame_message(output, position, options, builder, written);
}

/**
 *  @brief Writes the metadata of a record batch message for absolute file @p position into
 *      @p metadata, and its body as one gather per buffer into @p body, copying none of it.
 *
 *  The metadata ends on the alignment, so the body starts aligned at its absolute file position,
 *  and every buffer is padded to the alignment after it. A short @p metadata or @p body answers
 *  @c ipc_status_t::too_small_k with @p metadata_written still naming the metadata bytes needed,
 *  while the body needs one gather per buffer.
 */
inline ipc_status_t ipc_write_batch(ipc_schema_t const& schema, ipc_batch_t const& batch, ipc_options_t options,
                                    std::uint64_t position, bytes_span_t metadata, std::size_t& metadata_written,
                                    std::span<ipc_gather_t> body, std::uint64_t& body_bytes) noexcept {
    metadata_written = 0, body_bytes = 0;
    if (ipc_status_t const status = ipc_check_schema(schema); status != ipc_status_t::success_k) return status;
    if (ipc_status_t const status = ipc_check_layout(options, position); status != ipc_status_t::success_k)
        return status;
    if (batch.length < 0 || batch.nodes.size() != ipc_fields_of(schema.fields) ||
        batch.buffers.size() != ipc_buffers_of(schema.fields))
        return ipc_status_t::malformed_k;

    std::uint64_t body_end = 0;
    for (bytes_view_t const buffer : batch.buffers)
        body_end = ipc_round_up(body_end + buffer.size(), options.alignment);

    ipc_flatbuffer_builder_t builder(ipc_past_prefix(metadata));
    std::size_t const header = ipc_build_message(builder, ipc_message_header_record_batch_k, body_end);
    ipc_flatbuffer_slot_t const slots[] {{ipc_record_batch_length_k, 8},
                                         {ipc_record_batch_nodes_k, 4},
                                         {ipc_record_batch_buffers_k, 4}};
    std::size_t positions[3];
    builder.link(header, builder.table(slots, positions));
    builder.store(positions[0], batch.length);

    std::size_t const nodes = builder.vector(batch.nodes.size(), ipc_flatbuffer_pair_bytes_k);
    builder.link(positions[1], nodes);
    for (std::size_t index = 0; index != batch.nodes.size(); ++index) {
        builder.store(nodes + 4 + ipc_flatbuffer_pair_bytes_k * index, batch.nodes[index].length);
        builder.store(nodes + 12 + ipc_flatbuffer_pair_bytes_k * index, batch.nodes[index].null_count);
    }

    std::size_t const buffers = builder.vector(batch.buffers.size(), ipc_flatbuffer_pair_bytes_k);
    builder.link(positions[2], buffers);
    bool const gathers_fit = body.size() >= batch.buffers.size();
    std::uint64_t offset = 0;
    for (std::size_t index = 0; index != batch.buffers.size(); ++index) {
        bytes_view_t const buffer = batch.buffers[index];
        std::uint64_t const next = ipc_round_up(offset + buffer.size(), options.alignment);
        builder.store(buffers + 4 + ipc_flatbuffer_pair_bytes_k * index, static_cast<std::int64_t>(offset));
        builder.store(buffers + 12 + ipc_flatbuffer_pair_bytes_k * index, static_cast<std::int64_t>(buffer.size()));
        if (gathers_fit) body[index] = {buffer, static_cast<std::uint32_t>(next - offset - buffer.size())};
        offset = next;
    }
    body_bytes = body_end;
    ipc_status_t const status = ipc_frame_message(metadata, position, options, builder, metadata_written);
    return gathers_fit ? status : ipc_status_t::too_small_k;
}

/** Writes the 8 bytes that end a stream, the marker and a zero length. */
inline ipc_status_t ipc_write_end_of_stream(bytes_span_t output, std::size_t& written) noexcept {
    written = 8;
    if (output.size() < written) return ipc_status_t::too_small_k;
    ipc_store_little(output.data(), std::uint32_t(0xFFFFFFFFu));
    ipc_store_little(output.data() + 4, std::uint32_t(0));
    return ipc_status_t::success_k;
}

/**
 *  @brief Writes what ends a file after its end-of-stream marker: the @c Footer with the schema and
 *      the @p blocks of every batch, its @c int32 length, and the magic.
 *
 *  A file is @c ipc_file_magic_k, the stream, then this, so writing it never seeks and a pipe takes
 *  it. A short @p output answers @c ipc_status_t::too_small_k with the bytes needed in @p written.
 */
inline ipc_status_t ipc_write_footer(ipc_schema_t const& schema, std::span<ipc_block_t const> blocks,
                                     bytes_span_t output, std::size_t& written) noexcept {
    written = 0;
    if (ipc_status_t const status = ipc_check_schema(schema); status != ipc_status_t::success_k) return status;
    ipc_flatbuffer_builder_t builder(output);
    ipc_flatbuffer_slot_t const slots[] {{ipc_footer_schema_k, 4},
                                         {ipc_footer_dictionaries_k, 4},
                                         {ipc_footer_record_batches_k, 4},
                                         {ipc_footer_version_k, 2}};
    std::size_t positions[4];
    std::size_t const root = builder.claim(4, 4);
    builder.link(root, builder.table(slots, positions));
    builder.store(positions[3], static_cast<std::int16_t>(ipc_metadata_version_5_k));
    builder.link(positions[0], ipc_build_schema(builder, schema));
    builder.link(positions[1], builder.vector(0, ipc_flatbuffer_block_bytes_k));
    std::size_t const vector = builder.vector(blocks.size(), ipc_flatbuffer_block_bytes_k);
    builder.link(positions[2], vector);
    for (std::size_t index = 0; index != blocks.size(); ++index) {
        std::size_t const entry = vector + 4 + ipc_flatbuffer_block_bytes_k * index;
        builder.store(entry, blocks[index].offset);
        builder.store(entry + 8, blocks[index].metadata_bytes);
        builder.store(entry + 16, blocks[index].body_bytes);
    }
    builder.claim(0, 8);
    std::size_t const footer_bytes = builder.size();
    written = footer_bytes + 4 + 6;
    if (written > output.size()) return ipc_status_t::too_small_k;
    ipc_store_little(output.data() + footer_bytes, static_cast<std::int32_t>(footer_bytes));
    std::memcpy(output.data() + footer_bytes + 4, ipc_file_magic_k.data(), 6);
    return ipc_status_t::success_k;
}

#pragma endregion Writing

#pragma region C Data Adapters

/**
 *  @brief The memory an exported @c ArrowSchema or @c ArrowArray tree lives in, which the caller
 *      lends and keeps alive until the tree is released, so exporting allocates nothing.
 *
 *  Every field takes one entry of each array, nested ones included, as @c ipc_fields_max_k counts
 *  them: the columns first, then each field's children side by side, depth-first. Schemas also need
 *  @c text for their names, formats and metadata, which the caller lends; batches need none.
 *  Release marks the tree released and frees nothing.
 */
struct ipc_arrow_storage_t {
    ArrowSchema schema_children[ipc_fields_max_k];
    ArrowSchema* schema_pointers[ipc_fields_max_k];
    ArrowArray array_children[ipc_fields_max_k];
    ArrowArray* array_pointers[ipc_fields_max_k];
    void const* root_buffers[1];
    void const* child_buffers[ipc_fields_max_k][3];

    /** Characters of the names, formats and metadata of an exported schema. */
    std::span<char> text;
};

/** The offsets of an empty array of either width, since the C Data interface wants one offset even
 *  then. */
inline constexpr std::int64_t ipc_zero_offsets_k[1] {};

/** Releases an exported schema tree in caller-lent storage: its children, then itself. */
inline void ipc_release_schema(ArrowSchema* schema) noexcept {
    for (std::int64_t index = 0; index != schema->n_children; ++index)
        if (schema->children[index]->release) schema->children[index]->release(schema->children[index]);
    schema->release = nullptr;
}

/** Releases an exported array tree in caller-lent storage: its children, then itself. */
inline void ipc_release_array(ArrowArray* array) noexcept {
    for (std::int64_t index = 0; index != array->n_children; ++index)
        if (array->children[index]->release) array->children[index]->release(array->children[index]);
    array->release = nullptr;
}

/** The C Data format of @p field when it takes no parameter, or the prefix before its width, <tt>w:</tt>
 *  or <tt>+w:</tt>. */
constexpr std::string_view ipc_arrow_format_of(ipc_field_t const& field) noexcept {
    bool const is_signed = field.signedness == ipc_signedness_t::signed_k;
    switch (field.type) {
    case ipc_type_t::binary_k: return "z";
    case ipc_type_t::large_binary_k: return "Z";
    case ipc_type_t::utf8_k: return "u";
    case ipc_type_t::large_utf8_k: return "U";
    case ipc_type_t::fixed_size_binary_k: return "w:";
    case ipc_type_t::fixed_size_list_k: return "+w:";
    case ipc_type_t::list_k: return "+l";
    case ipc_type_t::large_list_k: return "+L";
    case ipc_type_t::struct_k: return "+s";
    case ipc_type_t::floating_point_k: return field.width == 16 ? "e" : field.width == 32 ? "f" : "g";
    case ipc_type_t::int_k:
        switch (field.width) {
        case 8: return is_signed ? "c" : "C";
        case 16: return is_signed ? "s" : "S";
        case 32: return is_signed ? "i" : "I";
        default: return is_signed ? "l" : "L";
        }
    }
    return "";
}

/** Whether @p field spells its width after its C Data format. */
constexpr bool ipc_arrow_format_has_width(ipc_field_t const& field) noexcept {
    return field.type == ipc_type_t::fixed_size_binary_k || field.type == ipc_type_t::fixed_size_list_k;
}

/** Decimal digits of a positive @p value. */
constexpr std::size_t ipc_decimal_digits(std::uint32_t value) noexcept {
    std::size_t digits = 1;
    while (value >= 10) value /= 10, ++digits;
    return digits;
}

/** Characters metadata takes in the C Data encoding: an @c int32 count, then a length and the bytes
 *  of each key and value, or none when there is no pair. */
constexpr std::size_t ipc_arrow_metadata_bytes(std::span<ipc_key_value_t const> pairs) noexcept {
    if (pairs.empty()) return 0;
    std::size_t bytes = 4;
    for (ipc_key_value_t const& pair : pairs) bytes += 8 + pair.key.size() + pair.value.size();
    return bytes;
}

/** Characters the formats, names and metadata of @p fields and their children take. */
constexpr std::size_t ipc_arrow_text_bytes(std::span<ipc_field_t const> fields) noexcept {
    std::size_t bytes = 0;
    for (ipc_field_t const& field : fields) {
        bytes += ipc_arrow_format_of(field).size() + 1 + field.name.size() + 1 +
                 ipc_arrow_metadata_bytes(field.metadata);
        if (ipc_arrow_format_has_width(field)) bytes += ipc_decimal_digits(static_cast<std::uint32_t>(field.width));
        bytes += ipc_arrow_text_bytes(field.children);
    }
    return bytes;
}

/** Appends @p bytes at @p cursor, advancing it. */
inline void ipc_append_text(char*& cursor, void const* bytes, std::size_t count) noexcept {
    if (count) std::memcpy(cursor, bytes, count);
    cursor += count;
}

/** Appends @p text and a terminator, answering where it starts. */
inline char const* ipc_append_string(char*& cursor, std::string_view text) noexcept {
    char const* const start = cursor;
    ipc_append_text(cursor, text.data(), text.size());
    *cursor++ = 0;
    return start;
}

/** Appends @p pairs in the C Data encoding, whose integers are native-endian, and answers where
 *  they start, or null when there is no pair. */
inline char const* ipc_append_metadata(char*& cursor, std::span<ipc_key_value_t const> pairs) noexcept {
    if (pairs.empty()) return nullptr;
    char const* const start = cursor;
    std::int32_t const count = static_cast<std::int32_t>(pairs.size());
    ipc_append_text(cursor, &count, 4);
    for (ipc_key_value_t const& pair : pairs) {
        std::int32_t const key_bytes = static_cast<std::int32_t>(pair.key.size());
        std::int32_t const value_bytes = static_cast<std::int32_t>(pair.value.size());
        ipc_append_text(cursor, &key_bytes, 4);
        ipc_append_text(cursor, pair.key.data(), pair.key.size());
        ipc_append_text(cursor, &value_bytes, 4);
        ipc_append_text(cursor, pair.value.data(), pair.value.size());
    }
    return start;
}

/** Exports @p fields into @p storage from entry @p first on, their children into the entries
 *  @p next hands out. */
inline void ipc_export_schemas(std::span<ipc_field_t const> fields, std::size_t first, std::size_t& next,
                               ipc_arrow_storage_t& storage, char*& cursor) noexcept {
    for (std::size_t index = 0; index != fields.size(); ++index) {
        ipc_field_t const& field = fields[index];
        ArrowSchema& out = storage.schema_children[first + index];
        out = {};
        storage.schema_pointers[first + index] = &out;
        std::string_view const format = ipc_arrow_format_of(field);
        out.format = cursor;
        ipc_append_text(cursor, format.data(), format.size());
        if (ipc_arrow_format_has_width(field)) {
            std::uint32_t value = static_cast<std::uint32_t>(field.width);
            std::size_t const digits = ipc_decimal_digits(value);
            for (std::size_t digit = digits; digit != 0; --digit, value /= 10)
                cursor[digit - 1] = char('0' + value % 10);
            cursor += digits;
        }
        *cursor++ = 0;
        out.name = ipc_append_string(cursor, field.name);
        out.metadata = ipc_append_metadata(cursor, field.metadata);
        out.flags = field.nullability == ipc_nullability_t::nullable_k ? ARROW_FLAG_NULLABLE : 0;
        out.release = &ipc_release_schema;
        if (field.children.empty()) continue;
        std::size_t const children = next;
        next += field.children.size();
        out.n_children = static_cast<std::int64_t>(field.children.size());
        out.children = storage.schema_pointers + children;
        ipc_export_schemas(field.children, children, next, storage, cursor);
    }
}

/** Exports @p schema as a C Data struct schema in @p storage, its strings in <tt>storage.text</tt>. */
inline ipc_status_t ipc_schema_to_arrow(ipc_schema_t const& schema, ipc_arrow_storage_t& storage,
                                        ArrowSchema& arrow) noexcept {
    if (ipc_status_t const status = ipc_check_schema(schema); status != ipc_status_t::success_k) return status;
    if (ipc_arrow_metadata_bytes(schema.metadata) + ipc_arrow_text_bytes(schema.fields) > storage.text.size())
        return ipc_status_t::too_small_k;
    char* cursor = storage.text.data();
    std::size_t next = schema.fields.size();
    ipc_export_schemas(schema.fields, 0, next, storage, cursor);
    arrow = {};
    arrow.format = "+s";
    arrow.name = "";
    arrow.metadata = ipc_append_metadata(cursor, schema.metadata);
    arrow.n_children = static_cast<std::int64_t>(schema.fields.size());
    arrow.children = storage.schema_pointers;
    arrow.release = &ipc_release_schema;
    return ipc_status_t::success_k;
}

/** Exports the arrays of @p fields from the nodes and buffers at @p cursor into @p storage from
 *  entry @p first on, their children into the entries @p next hands out. */
inline ipc_status_t ipc_export_arrays(std::span<ipc_field_t const> fields, ipc_batch_t const& batch,
                                      ipc_batch_cursor_t& cursor, std::size_t first, std::size_t& next,
                                      ipc_arrow_storage_t& storage) noexcept {
    for (std::size_t index = 0; index != fields.size(); ++index) {
        ipc_field_t const& field = fields[index];
        ArrowArray& out = storage.array_children[first + index];
        void const** const pointers = storage.child_buffers[first + index];
        ipc_node_t const& node = batch.nodes[cursor.node++];
        if (node.null_count != 0) return ipc_status_t::unsupported_k;
        std::size_t const buffer = cursor.buffer + 1;
        cursor.buffer += ipc_buffers_of(field.type);
        out = {};
        storage.array_pointers[first + index] = &out;
        out.length = node.length;
        out.n_buffers = static_cast<std::int64_t>(ipc_buffers_of(field.type));
        pointers[0] = nullptr;
        if (ipc_type_is_variable(field.type) || ipc_type_is_list(field.type)) {
            bytes_view_t const offsets = batch.buffers[buffer];
            pointers[1] = offsets.empty() ? static_cast<void const*>(ipc_zero_offsets_k) : offsets.data();
        }
        else if (ipc_type_is_fixed(field.type)) pointers[1] = batch.buffers[buffer].data();
        if (ipc_type_is_variable(field.type)) pointers[2] = batch.buffers[buffer + 1].data();
        out.buffers = pointers;
        out.release = &ipc_release_array;
        if (field.children.empty()) continue;
        std::size_t const children = next;
        next += field.children.size();
        out.n_children = static_cast<std::int64_t>(field.children.size());
        out.children = storage.array_pointers + children;
        if (ipc_status_t const status = ipc_export_arrays(field.children, batch, cursor, children, next, storage);
            status != ipc_status_t::success_k)
            return status;
    }
    return ipc_status_t::success_k;
}

/** Exports @p batch as a C Data struct array in @p storage, its buffers viewing the body the batch
 *  views. */
inline ipc_status_t ipc_batch_to_arrow(ipc_schema_t const& schema, ipc_batch_t const& batch,
                                       ipc_arrow_storage_t& storage, ArrowArray& arrow) noexcept {
    if (ipc_status_t const status = ipc_check_schema(schema); status != ipc_status_t::success_k) return status;
    if (batch.nodes.size() != ipc_fields_of(schema.fields) || batch.buffers.size() != ipc_buffers_of(schema.fields))
        return ipc_status_t::malformed_k;
    ipc_batch_cursor_t cursor;
    std::size_t next = schema.fields.size();
    if (ipc_status_t const status = ipc_export_arrays(schema.fields, batch, cursor, 0, next, storage);
        status != ipc_status_t::success_k)
        return status;
    storage.root_buffers[0] = nullptr;
    arrow = {};
    arrow.length = batch.length;
    arrow.n_buffers = 1;
    arrow.n_children = static_cast<std::int64_t>(schema.fields.size());
    arrow.buffers = storage.root_buffers;
    arrow.children = storage.array_pointers;
    arrow.release = &ipc_release_array;
    return ipc_status_t::success_k;
}

/** Reads C Data metadata, whose integers are native-endian, into the next entries of @p pool. */
inline ipc_status_t ipc_read_arrow_metadata(char const* metadata, std::span<ipc_key_value_t> pool, std::size_t& used,
                                            std::span<ipc_key_value_t const>& pairs) noexcept {
    pairs = {};
    if (!metadata) return ipc_status_t::success_k;
    std::int32_t count = 0;
    std::memcpy(&count, metadata, 4);
    metadata += 4;
    if (count < 0) return ipc_status_t::malformed_k;
    if (static_cast<std::size_t>(count) > pool.size() - used) return ipc_status_t::too_small_k;
    for (std::int32_t index = 0; index != count; ++index) {
        std::int32_t key_bytes = 0, value_bytes = 0;
        std::memcpy(&key_bytes, metadata, 4);
        if (key_bytes < 0) return ipc_status_t::malformed_k;
        std::string_view const key(metadata + 4, static_cast<std::size_t>(key_bytes));
        metadata += 4 + key_bytes;
        std::memcpy(&value_bytes, metadata, 4);
        if (value_bytes < 0) return ipc_status_t::malformed_k;
        std::string_view const value(metadata + 4, static_cast<std::size_t>(value_bytes));
        metadata += 4 + value_bytes;
        pool[used + static_cast<std::size_t>(index)] = {key, value};
    }
    pairs = pool.subspan(used, static_cast<std::size_t>(count));
    used += static_cast<std::size_t>(count);
    return ipc_status_t::success_k;
}

/** Reads the positive decimal width after a C Data format's prefix. */
constexpr ipc_status_t ipc_read_arrow_width(std::string_view digits, std::int32_t& width) noexcept {
    std::uint64_t value = 0;
    if (digits.empty() || digits.size() > 10) return ipc_status_t::malformed_k;
    for (char const digit : digits) {
        if (digit < '0' || digit > '9') return ipc_status_t::malformed_k;
        value = value * 10 + std::uint64_t(digit - '0');
    }
    if (value == 0 || value > 0x7FFFFFFFu) return ipc_status_t::malformed_k;
    width = static_cast<std::int32_t>(value);
    return ipc_status_t::success_k;
}

/** Maps a C Data @p format onto @p field, refusing what the codec does not handle. */
constexpr ipc_status_t ipc_read_arrow_format(std::string_view format, ipc_field_t& field) noexcept {
    field.width = 0;
    field.signedness = ipc_signedness_t::unsigned_k;
    if (format.starts_with("w:")) {
        field.type = ipc_type_t::fixed_size_binary_k;
        return ipc_read_arrow_width(format.substr(2), field.width);
    }
    if (format.starts_with("+w:")) {
        field.type = ipc_type_t::fixed_size_list_k;
        return ipc_read_arrow_width(format.substr(3), field.width);
    }
    if (format == "+l" || format == "+L" || format == "+s") {
        field.type = format[1] == 'l'   ? ipc_type_t::list_k
                     : format[1] == 'L' ? ipc_type_t::large_list_k
                                        : ipc_type_t::struct_k;
        return ipc_status_t::success_k;
    }
    if (format.size() != 1) return ipc_status_t::unsupported_k;
    constexpr std::string_view integers = "cCsSiIlL", floats = "efg";
    if (std::size_t const integer = integers.find(format.front()); integer != std::string_view::npos) {
        field.type = ipc_type_t::int_k;
        field.width = 8 << (integer / 2);
        field.signedness = integer % 2 ? ipc_signedness_t::unsigned_k : ipc_signedness_t::signed_k;
        return ipc_status_t::success_k;
    }
    if (std::size_t const real = floats.find(format.front()); real != std::string_view::npos) {
        field.type = ipc_type_t::floating_point_k;
        field.width = 16 << real;
        return ipc_status_t::success_k;
    }
    switch (format.front()) {
    case 'z': field.type = ipc_type_t::binary_k; return ipc_status_t::success_k;
    case 'Z': field.type = ipc_type_t::large_binary_k; return ipc_status_t::success_k;
    case 'u': field.type = ipc_type_t::utf8_k; return ipc_status_t::success_k;
    case 'U': field.type = ipc_type_t::large_utf8_k; return ipc_status_t::success_k;
    default: return ipc_status_t::unsupported_k;
    }
}

/** Imports @p count C Data @p children into the next fields of @p pools, siblings side by side and
 *  each one's children after them, as an IPC reader does, and answers the siblings in @p fields. */
inline ipc_status_t ipc_import_schemas(ArrowSchema* const* children, std::int64_t count, ipc_schema_pools_t& pools,
                                       std::span<ipc_field_t const>& fields) noexcept {
    if (count < 0 || (count && !children)) return ipc_status_t::malformed_k;
    std::size_t const siblings_count = static_cast<std::size_t>(count);
    if (siblings_count > ipc_fields_max_k - pools.fields_used) return ipc_status_t::unsupported_k;
    if (siblings_count > pools.fields.size() - pools.fields_used) return ipc_status_t::too_small_k;
    std::span<ipc_field_t> const siblings = pools.fields.subspan(pools.fields_used, siblings_count);
    pools.fields_used += siblings_count;
    for (std::size_t index = 0; index != siblings_count; ++index) {
        ArrowSchema const* const child = children[index];
        ipc_field_t& field = siblings[index];
        if (!child || !child->release || !child->format) return ipc_status_t::malformed_k;
        if (child->dictionary) return ipc_status_t::unsupported_k;
        ipc_status_t status = ipc_read_arrow_format(child->format, field);
        field.name = child->name ? std::string_view(child->name) : std::string_view {};
        field.nullability = child->flags & ARROW_FLAG_NULLABLE ? ipc_nullability_t::nullable_k
                                                               : ipc_nullability_t::non_nullable_k;
        if (status == ipc_status_t::success_k)
            status = ipc_read_arrow_metadata(child->metadata, pools.pairs, pools.pairs_used, field.metadata);
        if (status == ipc_status_t::success_k)
            status = ipc_import_schemas(child->children, child->n_children, pools, field.children);
        if (status != ipc_status_t::success_k) return status;
    }
    fields = siblings;
    return ipc_status_t::success_k;
}

/** Imports a C Data struct schema, viewing its strings, which the producer keeps alive until it is
 *  released. */
inline ipc_status_t ipc_schema_from_arrow(ArrowSchema const& arrow, std::span<ipc_field_t> fields,
                                          std::span<ipc_key_value_t> metadata, ipc_schema_t& schema) noexcept {
    if (!arrow.release || !arrow.format) return ipc_status_t::malformed_k;
    if (std::string_view(arrow.format) != "+s" || arrow.dictionary) return ipc_status_t::unsupported_k;
    ipc_schema_pools_t pools {fields, metadata};
    ipc_status_t status = ipc_import_schemas(arrow.children, arrow.n_children, pools, schema.fields);
    if (status == ipc_status_t::success_k)
        status = ipc_read_arrow_metadata(arrow.metadata, pools.pairs, pools.pairs_used, schema.metadata);
    return status == ipc_status_t::success_k ? ipc_check_schema(schema) : status;
}

/** Whether a C Data array holds no null: a zero count, or an unknown one with no validity
 *  bitmap. */
constexpr bool ipc_arrow_has_no_nulls(ArrowArray const& array) noexcept {
    return array.null_count == 0 || (array.null_count == -1 && (array.n_buffers == 0 || !array.buffers[0]));
}

/** Reads the native-endian offset @p index of an offsets buffer of @p offset_bytes each. */
inline std::int64_t ipc_arrow_offset(std::byte const* offsets, std::size_t offset_bytes, std::uint64_t index) noexcept {
    if (offset_bytes == 8) {
        std::int64_t wide = 0;
        std::memcpy(&wide, offsets + index * 8, 8);
        return wide;
    }
    std::int32_t narrow = 0;
    std::memcpy(&narrow, offsets + index * 4, 4);
    return narrow;
}

/** Imports the C Data @p arrays of @p fields, rows @p skip to @p skip + @p length of each before
 *  its own offset, into the nodes and buffers at @p cursor, depth-first, and their children after
 *  each of them. */
inline ipc_status_t ipc_import_arrays(std::span<ipc_field_t const> fields, ArrowArray* const* arrays,
                                      std::uint64_t skip, std::uint64_t length, std::span<ipc_node_t> nodes,
                                      std::span<bytes_view_t> buffers, ipc_batch_cursor_t& cursor) noexcept {
    for (std::size_t index = 0; index != fields.size(); ++index) {
        ipc_field_t const& field = fields[index];
        ArrowArray const* const array = arrays ? arrays[index] : nullptr;
        if (!array || !array->release || array->offset < 0 || array->length < 0 ||
            static_cast<std::uint64_t>(array->length) < skip + length)
            return ipc_status_t::malformed_k;
        if (!ipc_arrow_has_no_nulls(*array) || array->dictionary) return ipc_status_t::unsupported_k;
        if (array->n_children != static_cast<std::int64_t>(field.children.size()) ||
            array->n_buffers != static_cast<std::int64_t>(ipc_buffers_of(field.type)))
            return ipc_status_t::malformed_k;
        std::uint64_t const start = skip + static_cast<std::uint64_t>(array->offset);
        nodes[cursor.node++] = {static_cast<std::int64_t>(length), 0};
        buffers[cursor.buffer++] = {};
        std::uint64_t child_skip = start, child_length = length;
        if (ipc_type_is_variable(field.type) || ipc_type_is_list(field.type)) {
            std::size_t const offset_bytes = ipc_offset_bytes(field.type);
            std::byte const* const offsets = static_cast<std::byte const*>(array->buffers[1]);
            if (!offsets && length != 0) return ipc_status_t::malformed_k;
            std::int64_t const end = offsets ? ipc_arrow_offset(offsets, offset_bytes, start + length) : 0;
            if (end < 0) return ipc_status_t::malformed_k;
            buffers[cursor.buffer++] = {offsets ? offsets + start * offset_bytes : offsets,
                                        offsets ? static_cast<std::size_t>((length + 1) * offset_bytes) : 0};
            // A list's offsets index its child from the child's own start, so the child goes whole
            // up to the end
            child_skip = 0, child_length = static_cast<std::uint64_t>(end);
            if (ipc_type_is_variable(field.type)) {
                std::byte const* const data = static_cast<std::byte const*>(array->buffers[2]);
                if (!data && end != 0) return ipc_status_t::malformed_k;
                buffers[cursor.buffer++] = {data, static_cast<std::size_t>(end)};
            }
        }
        else if (ipc_type_is_fixed(field.type)) {
            std::size_t const value_bytes = ipc_value_bytes(field);
            std::byte const* const data = static_cast<std::byte const*>(array->buffers[1]);
            if (!data && length != 0) return ipc_status_t::malformed_k;
            buffers[cursor.buffer++] = {data ? data + start * value_bytes : data,
                                        static_cast<std::size_t>(length * value_bytes)};
        }
        else if (field.type == ipc_type_t::fixed_size_list_k) {
            std::uint64_t const width = static_cast<std::uint64_t>(field.width);
            child_skip = start * width, child_length = length * width;
        }
        if (ipc_status_t const status = ipc_import_arrays(field.children, array->children, child_skip, child_length,
                                                          nodes, buffers, cursor);
            status != ipc_status_t::success_k)
            return status;
    }
    return ipc_status_t::success_k;
}

/**
 *  @brief Imports a C Data struct array of @p schema as a batch whose buffers view the producer's,
 *      which it keeps alive until the array is released, so writing it copies nothing.
 *
 *  Offsets keep the producer's start, which IPC allows, so byte strings span from their data's
 *  first byte to the last offset, and a list's child from its first row to the list's last offset.
 *  Native-endian buffers are taken as little-endian, as on every host the codec targets.
 */
inline ipc_status_t ipc_batch_from_arrow(ipc_schema_t const& schema, ArrowArray const& arrow,
                                         std::span<ipc_node_t> nodes, std::span<bytes_view_t> buffers,
                                         ipc_batch_t& batch) noexcept {
    if (ipc_status_t const status = ipc_check_schema(schema); status != ipc_status_t::success_k) return status;
    if (!arrow.release || arrow.length < 0 || arrow.offset < 0) return ipc_status_t::malformed_k;
    if (arrow.n_children != static_cast<std::int64_t>(schema.fields.size())) return ipc_status_t::malformed_k;
    if (!ipc_arrow_has_no_nulls(arrow) || arrow.dictionary) return ipc_status_t::unsupported_k;
    std::size_t const nodes_count = ipc_fields_of(schema.fields), buffers_count = ipc_buffers_of(schema.fields);
    if (nodes.size() < nodes_count || buffers.size() < buffers_count) return ipc_status_t::too_small_k;
    ipc_batch_cursor_t cursor;
    if (ipc_status_t const status = ipc_import_arrays(schema.fields, arrow.children,
                                                      static_cast<std::uint64_t>(arrow.offset),
                                                      static_cast<std::uint64_t>(arrow.length), nodes, buffers, cursor);
        status != ipc_status_t::success_k)
        return status;
    batch = {arrow.length, nodes.first(nodes_count), buffers.first(buffers_count)};
    return ipc_status_t::success_k;
}

#pragma endregion C Data Adapters

} // namespace unum::usearch::tools
