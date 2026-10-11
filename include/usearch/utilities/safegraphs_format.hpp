/**
 *  @file include/usearch/utilities/safegraphs_format.hpp
 *  @author Ash Vardanian
 *  @date July 19, 2026
 *  @brief The SafeGraphs interchange - the self-describing JSON header in front of every
 *      `.safegraphs` file and the manifest beside it - read & written positionally through a
 *      mapping, a buffer, or a file slice - all of them bytes - and yyjson.
 *
 *  Every backend serializes its payloads differently - volatile in-RAM snapshots, GPU-resident
 *  uploads, persisted on-NVMe layouts - but they all speak the same header, so its reader and
 *  writer live here, payload-free. Every column names its buffers as @b [begin,end] byte
 *  positions within the data segment, so any number of threads fill disjoint rows of one
 *  column at once and nothing here holds a cursor.
 *
 *  The JSON itself goes through yyjson - exceptions-free C, no schema compilation - and every
 *  allocation flows through the caller's allocator, trampolined into @c yyjson_alc.
 *
 *  @see https://github.com/safegraphs/SafeGraphs
 */
#pragma once

#include <cstddef> // `std::byte`
#include <cstdint> // `std::uintptr_t`
#include <cstdio>  // `std::snprintf`
#include <cstring> // `std::memcpy`, `std::memset`

#include <limits>      // `std::numeric_limits` - the addressable ceiling a 32-bit host refuses past
#include <array>       // `std::array` - the prefix magic
#include <atomic>      // `std::atomic` - the first failure across regions
#include <bit>         // `std::bit_cast`, `std::endian`, `std::has_single_bit`
#include <concepts>    // `std::unsigned_integral`
#include <memory>      // `std::allocator_traits`
#include <optional>    // `std::optional` - a column's prefix sums, present or absent
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <utility>     // `std::forward`
#include <variant>     // `std::variant` - the interchange's three row shapes

#include <yyjson.h> // The JSON reader & writer - exceptions-free C

#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t`, `file_span_t`, `copy_bytes`
#include "usearch/utilities/ram_buffers.hpp"    // `growing_buffer` - path formatting & header images
#include "usearch/utilities/types.hpp"

namespace unum {
namespace usearch {

#pragma region Configs

/** The two file forms the interchange defines. */
enum class checkpoint_layout_t : u32_t {

    /** One file or buffer holding one graph: every shard's rows in placement order. */
    monolith_k,

    /** A manifest beside one monolith per shard: the interchange's partition record. */
    distributed_k,
};

/** How a checkpoint's parts land on the placed shards of a multi-index. */
enum class checkpoint_placement_t : u32_t {

    /** Part @c p lands on shard @c p mod S whole: exact when the counts agree, a union
     *  of parts past @c S, empty shards below it; never cuts a part. */
    partitioned_k,

    /** The checkpoint's rows cut into @c S contiguous stripes, one per shard; exact for
     *  a monolith that @c S rotation-filled shards wrote, well-formed for any other. */
    split_k,
};

/** One custom `__metadata__` entry: a producer's own key and its string value. */
struct safegraphs_custom_field_t {
    std::string_view key;
    std::string_view value;
};

/** The smallest buffer alignment the interchange allows. */
inline constexpr std::size_t safegraphs_alignment_floor_k = 64;

/** The buffer alignment a header declares when the producer names none. */
inline constexpr std::size_t safegraphs_alignment_default_k = 4096;

/** What a save writes and how. */
struct save_config_t {

    /** One monolith, or a manifest beside one monolith per shard. */
    checkpoint_layout_t layout = checkpoint_layout_t::monolith_k;

    /** The header's buffer alignment: a power of two, at least the floor. */
    std::size_t alignment = safegraphs_alignment_default_k;

    /** The contract the caller certifies; @c concurrent_k probes for settlement and may
     *  answer @c contended_k, @c immutable_k copies plain. */
    synchronization_t synchronization = synchronization_t::concurrent_k;

    /** Free-form `__metadata__` strings - @c created_by, @c embedding_model, @c built_at;
     *  the metric, quantization and alignment are the writer's own. */
    std::span<safegraphs_custom_field_t const> provenance = {};

    /** Whether this config can be written: a legal alignment, and no provenance key the
     *  writer already owns. */
    constexpr status_t validate() const noexcept {
        if (alignment < safegraphs_alignment_floor_k || !std::has_single_bit(alignment))
            return status_t::config_mismatch_k;
        for (safegraphs_custom_field_t const& field : provenance)
            if (field.key == "metric" || field.key == "quantization" || field.key == "nodes" ||
                field.key == "alignment")
                return status_t::config_mismatch_k;
        return status_t::success_k;
    }
};

/** Where a load places parts and how much room it leaves. */
struct load_config_t {

    /** Parts onto shards as written, or the rows cut into stripes. */
    checkpoint_placement_t placement = checkpoint_placement_t::partitioned_k;

    /** Rows to reserve across the shards; zero fits the checkpoint exactly and leaves
     *  the index full. */
    std::size_t capacity = 0;
};

#pragma endregion Configs

#pragma region Headers

/** One element per row; the header names no @c shape at all. */
struct scalar_shape_t {
    constexpr bool operator==(scalar_shape_t const&) const noexcept = default;
};

/** @c columns elements per row - a dense vector or a coordinate pair. */
struct vector_shape_t {

    /** Elements one row holds. */
    std::size_t columns = 1;
    constexpr bool operator==(vector_shape_t const&) const noexcept = default;
};

/** @c rows by @c columns per node; zero rows is the axis that resizes per node. */
struct matrix_shape_t {

    /** Rows per node; zero leaves the count to the prefix sums. */
    std::size_t rows = 0;

    /** Elements one row holds. */
    std::size_t columns = 1;
    constexpr bool operator==(matrix_shape_t const&) const noexcept = default;
};

/** One row's shape, in the interchange's own three cases: an absent @c shape is a scalar,
 *  a bare number a vector of that width, and a @b [rows,columns] pair a matrix. */
using column_shape_t = std::variant<scalar_shape_t, vector_shape_t, matrix_shape_t>;

/** Elements one fixed row holds; nothing when an axis resizes per node, which is the
 *  one case a fixed width cannot describe and the prefix sums must. */
constexpr std::optional<std::size_t> regular_width(column_shape_t shape) noexcept {
    if (auto const* matrix = std::get_if<matrix_shape_t>(&shape))
        return matrix->rows ? std::optional<std::size_t> {matrix->rows * matrix->columns} : std::nullopt;
    if (auto const* vector = std::get_if<vector_shape_t>(&shape)) return vector->columns;
    return 1;
}

/** The prefix sums an irregular column addresses its rows through: one element count per
 *  node, ending at the column's own element total. */
struct prefix_sums_t {

    /** Element type of the sums - @c u32_k or @c u64_k. */
    dtype_t dtype = dtype_t::u64_k;

    /** The sums themselves, within the data segment. */
    iota_view<std::size_t> bytes;
};

/** One header column: an element type, the shape of one row, its rows, and the prefix
 *  sums that address them when their width varies. */
struct safegraphs_column_t {

    /** Element type of the data buffer. */
    dtype_t dtype = dtype_t::unknown_k;

    /** What one row holds. */
    column_shape_t shape;

    /** The rows themselves, within the data segment. */
    iota_view<std::size_t> data;

    /** The prefix sums; absent for a fixed-width column, which needs none. */
    std::optional<prefix_sums_t> offsets;

    /** Bytes one row occupies - both axes, the packed families rounded up to whole
     *  bytes; zero when the row has no fixed width. */
    std::size_t row_bytes() const noexcept {
        std::optional<std::size_t> const width = regular_width(shape);
        std::size_t const bits = width ? *width * dtype_bits(dtype) : 0;
        return bits ? divide_round_up<std::size_t>(bits, 8) : 0;
    }

    /** Nodes this column's own buffers describe: its data over one row's width when
     *  fixed, its prefix sums less one otherwise; zero when neither says. */
    std::size_t nodes() const noexcept {
        if (!offsets) {
            std::size_t const width = row_bytes();
            return width ? data.size() / width : 0;
        }
        std::size_t const element_bytes = dtype_bytes(offsets->dtype);
        std::size_t const sums = element_bytes ? offsets->bytes.size() / element_bytes : 0;
        return sums ? sums - 1 : 0;
    }
};

/** The norms column's row: a node's sum, then its sum of squares, both in the dot product's dtype. */
inline constexpr column_shape_t safegraphs_norms_shape_k = vector_shape_t {2};

/** A parsed monolith header: the metadata, the three graph columns and the optional norms, and
 *  where the data segment begins. */
struct safegraphs_header_t {

    /** Buffer alignment in bytes the producer declared; 4096 by default. */
    std::size_t alignment = safegraphs_alignment_default_k;

    /** Absolute file offset of the data segment every column range is relative to. */
    std::size_t data_offset = 0;

    /** Node identity column. */
    safegraphs_column_t key;

    /** Payload column - the vectors, strings, or feature sets. */
    safegraphs_column_t value;

    /** Adjacency column of node indices; out-of-range IDs are positional sentinels. */
    safegraphs_column_t neighbors;

    /** The producer column of pre-computed norms, its dtype @c unknown_k when the part carries none; graph
     *  readers ignore it. */
    safegraphs_column_t norms;

    /** The @p length bytes of @p monolith at @p offset, empty when they do not fit -
     *  @c subspan asserts its precondition, so every range is checked before it is taken. */
    template <typename bytes_type_>
    static bytes_type_ slice_of(bytes_type_ monolith, std::size_t offset, std::size_t length) noexcept {
        if (offset > monolith.size() || length > monolith.size() - offset) return {};
        return monolith.subspan(offset, length);
    }

    /** The node count - the key column's own rows, the one source of truth every other
     *  column must agree with. */
    std::size_t nodes() const noexcept { return key.nodes(); }

    /** Bytes the monolith occupies: the header segment and the furthest column buffer. */
    std::size_t monolith_bytes() const noexcept {
        auto const end_of = [](safegraphs_column_t const& column) noexcept {
            return max_of(column.data.bound, column.offsets ? column.offsets->bytes.bound : 0);
        };
        return data_offset + max_of(max_of(end_of(key), end_of(value)), max_of(end_of(neighbors), end_of(norms)));
    }

    /** The bytes of @p rows rows of a regular @p column, starting at row @p first_row -
     *  the one slice a reader and a writer both address. Empty when the range does not fit
     *  the column, which every caller then refuses. */
    template <typename bytes_type_>
    bytes_type_ rows_of(bytes_type_ monolith, safegraphs_column_t const& column, std::size_t first_row,
                        std::size_t rows) const noexcept {
        std::size_t const width = column.row_bytes();
        if (column.offsets || !width) return {};
        std::size_t const start = first_row * width, length = rows * width;
        if (start > column.data.size() || length > column.data.size() - start) return {};
        return slice_of(monolith, data_offset + column.data.first + start, length);
    }

    /** The bytes of one whole column's rows. */
    template <typename bytes_type_>
    bytes_type_ data_of(bytes_type_ monolith, safegraphs_column_t const& column) const noexcept {
        return slice_of(monolith, data_offset + column.data.first, column.data.size());
    }

    /** The bytes of one column's prefix sums; empty for a regular column. */
    template <typename bytes_type_>
    bytes_type_ offsets_of(bytes_type_ monolith, safegraphs_column_t const& column) const noexcept {
        if (!column.offsets) return {};
        return slice_of(monolith, data_offset + column.offsets->bytes.first, column.offsets->bytes.size());
    }

    /** The norms column's bytes when they are the ones a reader keeping @p dtype norms takes: a sum
     *  and a sum of squares for every node; empty otherwise, and always for an @c unknown_k
     *  @p dtype, which keeps none. */
    template <typename bytes_type_>
    bytes_type_ norms_of(bytes_type_ monolith, dtype_t dtype) const noexcept {
        if (dtype == dtype_t::unknown_k || norms.dtype != dtype || norms.offsets ||
            norms.shape != safegraphs_norms_shape_k || norms.nodes() != nodes())
            return {};
        return data_of(monolith, norms);
    }
};

/** Reads @p value into @p extent, answering false where this host cannot address it. A
 *  checkpoint written where pointers are wider may name an extent past @c size_t, and
 *  truncating it would pass every later check carrying a plausible wrong number. */
inline bool safegraphs_addressable_(u64_t value, std::size_t& extent) noexcept {
    if (value > std::numeric_limits<std::size_t>::max()) return false;
    extent = static_cast<std::size_t>(value);
    return true;
}

/** The fixed 16 bytes every monolith opens with: the magic, then the little-endian
 *  byte length of the JSON header. */
struct safegraphs_prefix_t {
    inline static constexpr std::array<char, 8> magic_k = {'S', 'A', 'F', 'E', 'G', 'R', 'P', 'H'};
    std::array<char, 8> magic = magic_k;
    u64_t header_length = 0;
};
static_assert(sizeof(safegraphs_prefix_t) == 16, "The prefix is read and written as sixteen raw bytes");

/** Trampolines a byte allocator into @c yyjson_alc; each block carries its size in a
 *  max-aligned prefix, since yyjson frees by pointer alone. */
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
        std::memcpy(grown, pointer, min_of(old_size, size));
        deallocate_(context, pointer);
        return grown;
    }

    /** The C-ABI view yyjson consumes; valid while this trampoline stays alive. */
    yyjson_alc c_abi() noexcept { return {&allocate_, &reallocate_, &deallocate_, this}; }
};

/** The text of a string value; empty for anything else. */
inline std::string_view safegraphs_string_(yyjson_val* value) noexcept {
    return yyjson_is_str(value) ? std::string_view {yyjson_get_str(value), yyjson_get_len(value)} : std::string_view {};
}

/** Parses @p json into an object-rooted document the caller frees; the text is copied
 *  because yyjson mutates its input while reading it. */
template <typename allocator_type_>
expected<yyjson_doc*> safegraphs_read_document_(bytes_view_t json,
                                                safegraphs_yyjson_allocator_<allocator_type_>& trampoline) noexcept {
    if (json.empty()) return {nullptr, status_t::malformed_k};
    yyjson_alc allocator_abi = trampoline.c_abi();
    char* text = static_cast<char*>(allocator_abi.malloc(allocator_abi.ctx, json.size()));
    if (!text) return {nullptr, status_t::bad_alloc_k};
    std::memcpy(text, json.data(), json.size());
    yyjson_doc* document = yyjson_read_opts(text, json.size(), 0, &allocator_abi, nullptr);
    allocator_abi.free(allocator_abi.ctx, text);
    if (document && yyjson_is_obj(yyjson_doc_get_root(document))) return {document, status_t::success_k};
    yyjson_doc_free(document);
    return {nullptr, status_t::malformed_k};
}

/** Reads one @b [begin,end] pair into @p begin and @p end; a missing or malformed pair
 *  leaves both zero, which every extent check then refuses. */
inline iota_view<std::size_t> safegraphs_parse_span_(yyjson_val* pair) noexcept {
    if (!yyjson_is_arr(pair) || yyjson_arr_size(pair) != 2) return {};
    std::size_t first = 0, last = 0;
    if (!safegraphs_addressable_(yyjson_get_uint(yyjson_arr_get(pair, 0)), first)) return {};
    if (!safegraphs_addressable_(yyjson_get_uint(yyjson_arr_get(pair, 1)), last)) return {};
    return {first, last};
}

/** Walks one column object, reading the recognized fields and ignoring extras; the shape
 *  and the prefix sums are the one pair this file checks against each other. */
inline expected<safegraphs_column_t> safegraphs_parse_column_(yyjson_val* object) noexcept {
    safegraphs_column_t column;
    if (!yyjson_is_obj(object)) return {column, status_t::malformed_k};
    column.dtype = dtype_named(safegraphs_string_(yyjson_obj_get(object, "dtype")));
    column.data = safegraphs_parse_span_(yyjson_obj_get(object, "data"));

    // An absent shape is a scalar; a bare number or a one-element array is a vector of that
    // width; a two-element array is a matrix, whose leading zero marks the ragged axis.
    yyjson_val* const shape = yyjson_obj_get(object, "shape");
    std::size_t const axes = yyjson_is_arr(shape) ? yyjson_arr_size(shape) : 0;
    if (axes > 2) return {column, status_t::malformed_k};
    std::size_t rows = 0, columns = 0;
    if (axes == 2) {
        if (!safegraphs_addressable_(yyjson_get_uint(yyjson_arr_get(shape, 0)), rows))
            return {column, status_t::malformed_k};
        if (!safegraphs_addressable_(yyjson_get_uint(yyjson_arr_get(shape, 1)), columns))
            return {column, status_t::malformed_k};
        column.shape = matrix_shape_t {rows, columns};
    }
    else if (axes == 1) {
        if (!safegraphs_addressable_(yyjson_get_uint(yyjson_arr_get(shape, 0)), columns))
            return {column, status_t::malformed_k};
        column.shape = vector_shape_t {columns};
    }
    else if (shape) {
        if (!safegraphs_addressable_(yyjson_get_uint(shape), columns)) return {column, status_t::malformed_k};
        column.shape = vector_shape_t {columns};
    }

    iota_view<std::size_t> const sums = safegraphs_parse_span_(yyjson_obj_get(object, "offsets"));
    if (sums.empty()) return {column, regular_width(column.shape) ? status_t::success_k : status_t::malformed_k};
    column.offsets = prefix_sums_t {dtype_named(safegraphs_string_(yyjson_obj_get(object, "offsets_dtype"))), sums};
    return {column, status_t::success_k};
}

/**
 *  @brief Reads the prefix at the front of @p monolith and parses the header JSON behind it,
 *      touching no payload.
 *  @param[in] monolith The checkpoint's bytes, from its magic onward.
 *  @param[in] allocator Backs the JSON text and the yyjson document.
 *  @param[in] per_custom_field Fires once per string-valued `__metadata__` entry; its views
 *      die with this call.
 */
template <typename allocator_type_ = std::allocator<byte_t>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_header_t> safegraphs_read_header(bytes_view_t monolith, allocator_type_ allocator = {},
                                                     callback_type_&& per_custom_field = {}) noexcept {
    safegraphs_header_t header;
    if (monolith.size() < sizeof(safegraphs_prefix_t)) return {header, status_t::malformed_k};
    std::array<std::byte, sizeof(safegraphs_prefix_t)> prefix_bytes {};
    std::memcpy(prefix_bytes.data(), monolith.data(), prefix_bytes.size());
    safegraphs_prefix_t const prefix = std::bit_cast<safegraphs_prefix_t>(prefix_bytes);
    if (prefix.magic != safegraphs_prefix_t::magic_k) return {header, status_t::malformed_k};
    std::size_t const header_length = static_cast<std::size_t>(little_endian(prefix.header_length));
    if (header_length > monolith.size() - sizeof(prefix)) return {header, status_t::malformed_k};
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    auto [document, parse_status] = safegraphs_read_document_(monolith.subspan(sizeof(prefix), header_length),
                                                              trampoline);
    if (failed(parse_status)) return {header, parse_status};

    yyjson_val* root = yyjson_doc_get_root(document);
    if (yyjson_val* metadata = yyjson_obj_get(root, "__metadata__"); yyjson_is_obj(metadata)) {
        // A producer may declare any power of two; anything under the floor reads as the floor.
        if (yyjson_val* alignment = yyjson_obj_get(metadata, "alignment"))
            header.alignment = max_of<std::size_t>(yyjson_get_uint(alignment), safegraphs_alignment_floor_k);
        yyjson_obj_iter entries = yyjson_obj_iter_with(metadata);
        while (yyjson_val* key = yyjson_obj_iter_next(&entries))
            if (yyjson_val* value = yyjson_obj_iter_get_val(key); yyjson_is_str(value))
                per_custom_field(safegraphs_custom_field_t {safegraphs_string_(key), safegraphs_string_(value)});
    }
    // Each column parses whole, and `malformed_k` is the only verdict any of them can reach.
    auto const [key, key_status] = safegraphs_parse_column_(yyjson_obj_get(root, "key"));
    auto const [value, value_status] = safegraphs_parse_column_(yyjson_obj_get(root, "value"));
    auto const [neighbors, neighbors_status] = safegraphs_parse_column_(yyjson_obj_get(root, "neighbors"));
    yyjson_val* const norms_object = yyjson_obj_get(root, "norms");
    auto const [norms, norms_status] = norms_object ? safegraphs_parse_column_(norms_object)
                                                    : expected<safegraphs_column_t> {{}, status_t::success_k};
    yyjson_doc_free(document);
    if (failed(key_status) || failed(value_status) || failed(neighbors_status) || failed(norms_status))
        return {header, status_t::malformed_k};
    header.key = key, header.value = value, header.neighbors = neighbors, header.norms = norms;

    // The data segment starts on the declared alignment boundary at or past the JSON's end.
    if (!std::has_single_bit(header.alignment)) return {header, status_t::malformed_k};
    header.data_offset = round_up_to_multiple<std::size_t>(sizeof(prefix) + header_length, header.alignment);
    return {header, status_t::success_k};
}

/** Maps @p path read-only and parses its header. */
template <typename allocator_type_ = std::allocator<byte_t>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_header_t> safegraphs_read_header(std::string_view path, allocator_type_ allocator = {},
                                                     callback_type_&& per_custom_field = {}) noexcept {
    auto [mapping, map_status] = mapped_file_t::read(path);
    if (failed(map_status)) return {safegraphs_header_t {}, map_status};
    return safegraphs_read_header(mapping.view(), allocator, std::forward<callback_type_>(per_custom_field));
}

/** Builds the header's JSON document; null when an allocation failed. A regular column
 *  carries its shape, an irregular one its prefix-sum buffer, and both their data pair. */
inline yyjson_mut_doc* safegraphs_header_document_(safegraphs_header_t const& header,
                                                   std::span<safegraphs_custom_field_t const> custom_fields,
                                                   yyjson_alc const& allocator_abi) noexcept {
    yyjson_mut_doc* document = yyjson_mut_doc_new(&allocator_abi);
    if (!document) return nullptr;
    yyjson_mut_val* root = yyjson_mut_obj(document);
    yyjson_mut_doc_set_root(document, root);
    yyjson_mut_val* metadata = yyjson_mut_obj_add_obj(document, root, "__metadata__");
    bool complete = yyjson_mut_obj_add_uint(document, metadata, "alignment", header.alignment);
    for (safegraphs_custom_field_t const& field : custom_fields)
        complete = complete && yyjson_mut_obj_add(metadata, //
                                                  yyjson_mut_strncpy(document, field.key.data(), field.key.size()),
                                                  yyjson_mut_strncpy(document, field.value.data(), field.value.size()));
    auto add_span = [&](yyjson_mut_val* object, char const* label, iota_view<std::size_t> span) noexcept {
        yyjson_mut_val* pair = yyjson_mut_obj_add_arr(document, object, label);
        return pair && yyjson_mut_arr_add_uint(document, pair, span.first) &&
               yyjson_mut_arr_add_uint(document, pair, span.bound);
    };
    // A scalar names no shape, a vector a bare width, a matrix both of its axes.
    auto add_shape = [&](yyjson_mut_val* object, column_shape_t shape) noexcept {
        if (auto const* matrix = std::get_if<matrix_shape_t>(&shape)) {
            yyjson_mut_val* axes = yyjson_mut_obj_add_arr(document, object, "shape");
            return axes && yyjson_mut_arr_add_uint(document, axes, matrix->rows) &&
                   yyjson_mut_arr_add_uint(document, axes, matrix->columns);
        }
        if (auto const* vector = std::get_if<vector_shape_t>(&shape))
            return bool(yyjson_mut_obj_add_uint(document, object, "shape", vector->columns));
        return true;
    };
    auto add_column = [&](char const* label, safegraphs_column_t const& column) noexcept {
        yyjson_mut_val* object = yyjson_mut_obj_add_obj(document, root, label);
        bool added = yyjson_mut_obj_add_str(document, object, "dtype", dtype_name(column.dtype)) &&
                     add_shape(object, column.shape);
        if (column.offsets)
            added = added &&
                    yyjson_mut_obj_add_str(document, object, "offsets_dtype", dtype_name(column.offsets->dtype)) &&
                    add_span(object, "offsets", column.offsets->bytes);
        return added && add_span(object, "data", column.data);
    };
    complete = complete && add_column("key", header.key) && add_column("value", header.value) &&
               add_column("neighbors", header.neighbors);
    if (header.norms.dtype != dtype_t::unknown_k) complete = complete && add_column("norms", header.norms);
    if (complete) return document;
    yyjson_mut_doc_free(document);
    return nullptr;
}

/** Serializes @p document, hands the text to @p consume, then frees both; a null document
 *  reports @c bad_alloc_k. */
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

/** Where a header lands its data segment: the prefix and JSON rounded up to the declared
 *  alignment. */
template <typename allocator_type_ = std::allocator<byte_t>>
expected<std::size_t> safegraphs_data_offset(safegraphs_header_t const& header,
                                             std::span<safegraphs_custom_field_t const> custom_fields = {},
                                             allocator_type_ allocator = {}) noexcept {
    if (!std::has_single_bit(header.alignment)) return {0, status_t::config_mismatch_k};
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    yyjson_alc const allocator_abi = trampoline.c_abi();
    std::size_t json_bytes = 0;
    status_t const printed = safegraphs_print_(
        safegraphs_header_document_(header, custom_fields, allocator_abi), allocator_abi,
        [&](std::string_view json) noexcept { return json_bytes = json.size(), status_t::success_k; });
    if (failed(printed)) return {0, printed};
    return {round_up_to_multiple<std::size_t>(sizeof(safegraphs_prefix_t) + json_bytes, header.alignment),
            status_t::success_k};
}

/** Writes the prefix, the JSON, and space padding up to @c header.data_offset over the
 *  front of @p monolith - exactly the bytes @ref safegraphs_read_header consumes. Written
 *  last, it is what turns a run of bytes into a checkpoint. */
template <typename bytes_type_, typename allocator_type_ = std::allocator<byte_t>>
status_t safegraphs_write_header(bytes_type_ monolith, safegraphs_header_t const& header,
                                 std::span<safegraphs_custom_field_t const> custom_fields = {},
                                 allocator_type_ allocator = {}) noexcept {
    if (header.data_offset < sizeof(safegraphs_prefix_t)) return status_t::config_mismatch_k;
    if (monolith.size() < header.data_offset) return status_t::overflow_k;
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    yyjson_alc const allocator_abi = trampoline.c_abi();
    growing_buffer<byte_t, allocator_type_> image {allocator};
    if (failed(image.resize_uninitialized(header.data_offset))) return status_t::bad_alloc_k;
    std::size_t const header_length = header.data_offset - sizeof(safegraphs_prefix_t);
    safegraphs_prefix_t prefix;
    prefix.header_length = little_endian(static_cast<u64_t>(header_length));
    status_t const staged = safegraphs_print_(
        safegraphs_header_document_(header, custom_fields, allocator_abi), allocator_abi,
        [&](std::string_view json) noexcept {
            if (json.size() > header_length) return status_t::config_mismatch_k;
            auto const prefix_bytes = std::bit_cast<std::array<byte_t, sizeof(safegraphs_prefix_t)>>(prefix);
            bytes_span_t const staging = to_writable_span(image);
            [[maybe_unused]] status_t const framed = copy_bytes(staging.subspan(0, prefix_bytes.size()),
                                                                bytes_of(prefix_bytes.data(), prefix_bytes.size()));
            [[maybe_unused]] status_t const printed = copy_bytes(staging.subspan(sizeof(prefix), json.size()),
                                                                 std::as_bytes(std::span {json}));
            std::memset(image.data() + sizeof(prefix) + json.size(), ' ', header_length - json.size());
            return status_t::success_k;
        });
    if (failed(staged)) return staged;
    return copy_bytes(monolith.subspan(0, header.data_offset), bytes_of(image.data(), image.size()));
}

#pragma endregion Headers

#pragma region Columns

/** The interchange token of one trivially-copyable key by its width; @c unknown_k for
 *  widths the interchange has no name for. */
template <typename key_type_>
constexpr dtype_t safegraphs_key_dtype() noexcept {
    if constexpr (sizeof(key_type_) == 4) return dtype_t::u32_k;
    else if constexpr (sizeof(key_type_) == 8) return dtype_t::u64_k;
    else if constexpr (sizeof(key_type_) == 16) return dtype_t::uuid_k;
    else return dtype_t::unknown_k;
}

/** The interchange token of one node ID: the interchange admits @c u32 and @c u64. */
template <typename node_id_type_>
constexpr dtype_t safegraphs_node_id_dtype() noexcept {
    if constexpr (sizeof(node_id_type_) == 4) return dtype_t::u32_k;
    else if constexpr (sizeof(node_id_type_) == 8) return dtype_t::u64_k;
    else return dtype_t::unknown_k;
}

/** The value column's dtype for a stored scalar - its own token, the packed families
 *  included, whose @c x2 and @c x8 suffixes let a shape count logical elements. */
template <typename scalar_type_>
constexpr dtype_t safegraphs_value_dtype() noexcept {
    return dtype_named(scalar_type_::dtype_name());
}

/** What the regular columns need to be laid out: the node count, each column's element
 *  type, and the two row widths. */
struct safegraphs_regular_shape_t {

    /** Nodes every column describes. */
    std::size_t nodes = 0;

    /** Element type of the key column. */
    dtype_t key_dtype = dtype_t::unknown_k;

    /** Element type of the value column. */
    dtype_t value_dtype = dtype_t::unknown_k;

    /** Elements per value row - the logical dimensions, the packed families included. */
    std::size_t value_columns = 0;

    /** Element type of the neighbors column. */
    dtype_t neighbors_dtype = dtype_t::unknown_k;

    /** Neighbor IDs per node's list. */
    std::size_t connectivity = 0;

    /** Element type of the norms column; @c unknown_k lays out none. */
    dtype_t norms_dtype = dtype_t::unknown_k;
};

/** A header of three regular columns at aligned starts - keys, fixed-width values, and
 *  fixed-degree neighbors - then the norms where @p shape names their dtype, with the data
 *  offset they follow. */
template <typename allocator_type_ = std::allocator<byte_t>>
expected<safegraphs_header_t> safegraphs_regular_header(safegraphs_regular_shape_t const& shape, std::size_t alignment,
                                                        std::span<safegraphs_custom_field_t const> custom_fields = {},
                                                        allocator_type_ allocator = {}) noexcept {
    safegraphs_header_t header;
    header.alignment = alignment;
    header.key.dtype = shape.key_dtype;
    header.key.data = {0, shape.nodes * dtype_bytes(shape.key_dtype)};
    header.value.dtype = shape.value_dtype;
    header.value.shape = vector_shape_t {shape.value_columns};
    header.value.data.first = round_up_to_multiple(header.key.data.bound, alignment);
    header.value.data.bound = header.value.data.first + shape.nodes * header.value.row_bytes();
    header.neighbors.dtype = shape.neighbors_dtype;
    header.neighbors.shape = vector_shape_t {shape.connectivity};
    header.neighbors.data.first = round_up_to_multiple(header.value.data.bound, alignment);
    header.neighbors.data.bound = header.neighbors.data.first + shape.nodes * header.neighbors.row_bytes();
    if (shape.norms_dtype != dtype_t::unknown_k) {
        header.norms.dtype = shape.norms_dtype;
        header.norms.shape = safegraphs_norms_shape_k;
        header.norms.data.first = round_up_to_multiple(header.neighbors.data.bound, alignment);
        header.norms.data.bound = header.norms.data.first + shape.nodes * header.norms.row_bytes();
    }
    auto const [data_offset, status] = safegraphs_data_offset(header, custom_fields, allocator);
    header.data_offset = data_offset;
    return {header, status};
}

#pragma endregion Columns

#pragma region Parts

/** One monolith of a checkpoint: a manifest's file entry, or the lone monolith itself. */
struct safegraphs_part_t {

    /** The monolith's path, relative to the manifest; empty for a lone monolith. */
    std::string_view path;

    /** The monolith's whole size in bytes. */
    std::size_t bytes = 0;

    /** Nodes the monolith holds. */
    std::size_t nodes = 0;
};

/** Holds the longest NumKong kernel spelling, 22 characters, with its NUL. */
inline constexpr std::size_t safegraphs_name_capacity_k = 32;

/** A NUL-terminated metadata name; longer input truncates, so a foreign name fails a
 *  comparison instead of overrunning. */
struct safegraphs_name_t {
    char characters[safegraphs_name_capacity_k] = {};

    safegraphs_name_t() = default;
    explicit safegraphs_name_t(std::string_view text) noexcept {
        if (!text.empty()) std::memcpy(characters, text.data(), min_of(text.size(), safegraphs_name_capacity_k - 1));
    }
    std::string_view view() const noexcept { return characters; }
    bool empty() const noexcept { return !characters[0]; }
};

/** What a manifest or a lone monolith says about itself without opening a column. */
struct safegraphs_metadata_t {

    /** The metric the graph was built for; empty when the producer omitted it. */
    safegraphs_name_t metric;

    /** The stored scalar by its own name - what a packed @c u8 value column cannot say;
     *  empty when the producer omitted it. */
    safegraphs_name_t quantization;

    /** Element type of every part's value column. */
    dtype_t value_dtype = dtype_t::unknown_k;

    /** Element type of every part's key column. */
    dtype_t key_dtype = dtype_t::unknown_k;

    /** The value column's shape, from the first part's header. */
    column_shape_t value_shape;

    /** Neighbor IDs per node's list, from the first part's header. */
    std::size_t connectivity = 0;

    /** Total nodes across every part. */
    std::size_t nodes = 0;

    /** Monoliths the checkpoint spans; one for a lone monolith. */
    std::size_t parts = 0;
};

/** @p path with its @c .json and @c .safegraphs suffixes removed. */
inline std::string_view safegraphs_stem(std::string_view path) noexcept {
    if (path.ends_with(".json")) path.remove_suffix(std::string_view {".json"}.size());
    if (path.ends_with(".safegraphs")) path.remove_suffix(std::string_view {".safegraphs"}.size());
    return path;
}

/** The last path component - what a manifest entry records. */
inline std::string_view safegraphs_basename(std::string_view path) noexcept {
    return path.substr(path.find_last_of("/\\") + 1);
}

/** Formats @c <stem>-00001-of-00002.safegraphs into @p buffer, measured first; parts
 *  count from one, as the interchange illustrates. */
template <typename allocator_type_>
expected<std::string_view> safegraphs_part_path(growing_buffer<char, allocator_type_>& buffer, std::string_view stem,
                                                std::size_t part, std::size_t parts) noexcept {
    auto print = [&](char* sink, std::size_t capacity) noexcept {
        return std::snprintf(sink, capacity, "%.*s-%05zu-of-%05zu.safegraphs", static_cast<int>(stem.size()),
                             stem.data(), part + 1, parts);
    };
    int const length = print(nullptr, 0);
    if (length < 0) return {{}, status_t::io_k};
    if (failed(buffer.resize_uninitialized(static_cast<std::size_t>(length) + 1))) return {{}, status_t::bad_alloc_k};
    print(buffer.data(), buffer.size());
    return {std::string_view {buffer.data(), static_cast<std::size_t>(length)}, status_t::success_k};
}

/** Resolves an entry's @p relative path beside @p manifest_path into @p buffer. */
template <typename allocator_type_>
expected<std::string_view> safegraphs_sibling_path(growing_buffer<char, allocator_type_>& buffer,
                                                   std::string_view manifest_path, std::string_view relative) noexcept {
    std::string_view const directory = manifest_path.substr(
        0, manifest_path.size() - safegraphs_basename(manifest_path).size());
    if (failed(buffer.resize_uninitialized(directory.size() + relative.size()))) return {{}, status_t::bad_alloc_k};
    std::memcpy(buffer.data(), directory.data(), directory.size());
    std::memcpy(buffer.data() + directory.size(), relative.data(), relative.size());
    return {to_string_view(buffer), status_t::success_k};
}

/** Rows @p stripe of @p stripes takes from @p rows: the leading stripes take one more,
 *  which is exactly how rotation fills shards. */
constexpr iota_view<std::size_t> checkpoint_stripe(std::size_t rows, std::size_t stripes, std::size_t stripe) noexcept {
    std::size_t const quotient = rows / stripes;
    std::size_t const remainder = rows % stripes;
    std::size_t const first = stripe * quotient + min_of(stripe, remainder);
    return {first, first + quotient + (stripe < remainder ? 1 : 0)};
}

/** One slice of one part landing on one shard. */
struct checkpoint_assignment_t {

    /** The shard receiving the rows. */
    std::size_t shard = 0;

    /** The part the rows come from. */
    std::size_t part = 0;

    /** The part-local rows taken, in the part's own ID space. */
    iota_view<std::size_t> rows;
};

/** Maps a checkpoint's parts onto @p shards under @p placement, firing @p per_assignment
 *  grouped by shard and in part order within one; empty slices never fire. */
template <typename callback_type_>
void assign_parts(checkpoint_placement_t placement, std::span<std::size_t const> part_rows, std::size_t shards,
                  callback_type_&& per_assignment) noexcept {
    switch (placement) {
    case checkpoint_placement_t::partitioned_k:
        for (std::size_t shard = 0; shard != shards; ++shard)
            for (std::size_t part = shard; part < part_rows.size(); part += shards)
                if (part_rows[part]) per_assignment(checkpoint_assignment_t {shard, part, {0, part_rows[part]}});
        return;
    case checkpoint_placement_t::split_k: break;
    }
    std::size_t total = 0;
    for (std::size_t const rows : part_rows) total += rows;
    for (std::size_t shard = 0; shard != shards; ++shard) {
        iota_view<std::size_t> const stripe = checkpoint_stripe(total, shards, shard);
        if (stripe.empty()) continue;
        std::size_t part_first = 0;
        for (std::size_t part = 0; part != part_rows.size(); ++part) {
            std::size_t const part_bound = part_first + part_rows[part];
            std::size_t const first = max_of(stripe.first, part_first);
            std::size_t const bound = min_of(stripe.bound, part_bound);
            if (first < bound)
                per_assignment(checkpoint_assignment_t {shard, part, {first - part_first, bound - part_first}});
            part_first = part_bound;
        }
    }
}

/** Builds a manifest document; null when an allocation failed. It catalogs parts and
 *  nothing else: the schema, the metric and the provenance live in each part's own header,
 *  so a part lifted out of the set still describes itself completely. */
template <typename callback_type_>
yyjson_mut_doc* safegraphs_manifest_document_(std::size_t parts, callback_type_&& part_at,
                                              std::span<safegraphs_custom_field_t const> custom_fields,
                                              yyjson_alc const& allocator_abi) noexcept {
    yyjson_mut_doc* document = yyjson_mut_doc_new(&allocator_abi);
    if (!document) return nullptr;
    yyjson_mut_val* root = yyjson_mut_obj(document);
    yyjson_mut_doc_set_root(document, root);
    bool complete = true;
    if (!custom_fields.empty()) {
        yyjson_mut_val* summary = yyjson_mut_obj_add_obj(document, root, "__metadata__");
        for (safegraphs_custom_field_t const& field : custom_fields)
            complete = complete &&
                       yyjson_mut_obj_add(summary, //
                                          yyjson_mut_strncpy(document, field.key.data(), field.key.size()),
                                          yyjson_mut_strncpy(document, field.value.data(), field.value.size()));
    }
    yyjson_mut_val* files = yyjson_mut_obj_add_arr(document, root, "files");
    for (std::size_t part = 0; part != parts && complete; ++part) {
        safegraphs_part_t const entry = part_at(part);
        yyjson_mut_val* object = yyjson_mut_arr_add_obj(document, files);
        complete = yyjson_mut_obj_add_strncpy(document, object, "path", entry.path.data(), entry.path.size()) &&
                   yyjson_mut_obj_add_uint(document, object, "bytes", entry.bytes) &&
                   yyjson_mut_obj_add_uint(document, object, "nodes", entry.nodes);
    }
    if (complete) return document;
    yyjson_mut_doc_free(document);
    return nullptr;
}

/** Bytes a manifest over @p parts entries from @p part_at occupies. */
template <typename callback_type_, typename allocator_type_ = std::allocator<byte_t>>
expected<std::size_t> safegraphs_manifest_bytes(std::size_t parts, callback_type_&& part_at,
                                                std::span<safegraphs_custom_field_t const> custom_fields = {},
                                                allocator_type_ allocator = {}) noexcept {
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    yyjson_alc const allocator_abi = trampoline.c_abi();
    std::size_t json_bytes = 0;
    status_t const printed = safegraphs_print_(
        safegraphs_manifest_document_(parts, part_at, custom_fields, allocator_abi), allocator_abi,
        [&](std::string_view json) noexcept { return json_bytes = json.size(), status_t::success_k; });
    return {json_bytes, printed};
}

/** The metadata one monolith's header implies - the answer every schema question has,
 *  whether it was asked of a lone monolith or of a checkpoint's first part. */
inline safegraphs_metadata_t safegraphs_metadata_of_(safegraphs_header_t const& header, safegraphs_name_t metric,
                                                     safegraphs_name_t quantization) noexcept {
    safegraphs_metadata_t metadata;
    metadata.metric = metric;
    metadata.quantization = quantization;
    metadata.value_dtype = header.value.dtype;
    metadata.key_dtype = header.key.dtype;
    metadata.value_shape = header.value.shape;
    metadata.connectivity = regular_width(header.neighbors.shape).value_or(0);
    metadata.nodes = header.nodes();
    metadata.parts = 1;
    return metadata;
}

/** Reads a manifest: @p per_part fires once per entry in file order with a path that dies
 *  with this call, and the metadata comes from the first part's header, the one place a
 *  schema is written. Node counts sum over the entries that declare one. */
template <typename allocator_type_ = std::allocator<byte_t>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_metadata_t> safegraphs_read_manifest(std::string_view path, allocator_type_ allocator = {},
                                                         callback_type_&& per_part = {}) noexcept {
    safegraphs_metadata_t metadata;
    auto [mapping, map_status] = mapped_file_t::read(path);
    if (failed(map_status)) return {metadata, map_status};
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    auto [document, parse_status] = safegraphs_read_document_(mapping.view(), trampoline);
    if (failed(parse_status)) return {metadata, parse_status};

    yyjson_val* root = yyjson_doc_get_root(document);
    yyjson_val* files = yyjson_obj_get(root, "files");
    metadata.parts = yyjson_arr_size(files);
    std::string_view first_relative;
    for (std::size_t part = 0; part != metadata.parts; ++part) {
        yyjson_val* entry = yyjson_arr_get(files, part);
        std::string_view const relative = safegraphs_string_(yyjson_obj_get(entry, "path"));
        if (relative.empty()) continue;
        if (first_relative.empty()) first_relative = relative;
        std::size_t nodes = 0, bytes = 0;
        if (!safegraphs_addressable_(yyjson_get_uint(yyjson_obj_get(entry, "nodes")), nodes) ||
            !safegraphs_addressable_(yyjson_get_uint(yyjson_obj_get(entry, "bytes")), bytes)) {
            yyjson_doc_free(document);
            return {metadata, status_t::malformed_k};
        }
        metadata.nodes += nodes;
        per_part(safegraphs_part_t {relative, bytes, nodes});
    }

    // The schema lives in the parts, so the first one's header answers every question about it.
    growing_buffer<char, allocator_type_> sibling {allocator};
    auto const [first_path, resolve_status] = safegraphs_sibling_path(sibling, path, first_relative);
    yyjson_doc_free(document); // Only now: `first_relative` points into it
    if (first_relative.empty()) return {metadata, status_t::malformed_k};
    if (failed(resolve_status)) return {metadata, resolve_status};
    safegraphs_name_t metric;
    safegraphs_name_t quantization;
    auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") metric = safegraphs_name_t {field.value};
        if (field.key == "quantization") quantization = safegraphs_name_t {field.value};
    };
    auto const [header, header_status] = safegraphs_read_header(first_path, allocator, per_custom_field);
    if (failed(header_status)) return {metadata, header_status};
    std::size_t const parts = metadata.parts;
    std::size_t const nodes = metadata.nodes;
    metadata = safegraphs_metadata_of_(header, metric, quantization);
    metadata.parts = parts;
    metadata.nodes = nodes ? nodes : metadata.nodes;
    return {metadata, status_t::success_k};
}

/** Writes a manifest over @p parts entries drawn from @p part_at in order - the commit
 *  record a distributed save writes last. */
template <typename callback_type_, typename allocator_type_ = std::allocator<byte_t>>
status_t safegraphs_write_manifest(std::string_view path, std::size_t parts, callback_type_&& part_at,
                                   std::span<safegraphs_custom_field_t const> custom_fields = {},
                                   allocator_type_ allocator = {}) noexcept {
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    yyjson_alc const allocator_abi = trampoline.c_abi();
    return safegraphs_print_(
        safegraphs_manifest_document_(parts, part_at, custom_fields, allocator_abi), allocator_abi,
        [&](std::string_view json) noexcept {
            auto [mapping, map_status] = mapped_file_t::write(path, json.size());
            if (failed(map_status)) return map_status;
            if (status_t const wrote = copy_bytes(mapping.span(), bytes_of(json.data(), json.size())); failed(wrote))
                return wrote;
            return mapping.sync();
        });
}

/** The metadata of a monolith held in @p monolith. */
template <typename allocator_type_ = std::allocator<byte_t>>
expected<safegraphs_metadata_t> safegraphs_read_metadata(bytes_view_t monolith,
                                                         allocator_type_ allocator = {}) noexcept {
    safegraphs_name_t metric;
    safegraphs_name_t quantization;
    auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") metric = safegraphs_name_t {field.value};
        if (field.key == "quantization") quantization = safegraphs_name_t {field.value};
    };
    auto const [header, parse_status] = safegraphs_read_header(monolith, allocator, per_custom_field);
    if (failed(parse_status)) return {safegraphs_metadata_t {}, parse_status};
    return {safegraphs_metadata_of_(header, metric, quantization), status_t::success_k};
}

/** The metadata of a manifest or of a lone monolith at @p path - told apart by content,
 *  the monolith's magic, never by the extension. */
template <typename allocator_type_ = std::allocator<byte_t>>
expected<safegraphs_metadata_t> safegraphs_read_metadata(std::string_view path,
                                                         allocator_type_ allocator = {}) noexcept {
    auto [mapping, map_status] = mapped_file_t::read(path);
    if (failed(map_status)) return {safegraphs_metadata_t {}, map_status};
    bytes_view_t const monolith = mapping.view();
    if (monolith.size() >= safegraphs_prefix_t::magic_k.size() &&
        std::memcmp(monolith.data(), safegraphs_prefix_t::magic_k.data(), safegraphs_prefix_t::magic_k.size()) == 0)
        return safegraphs_read_metadata(monolith, allocator);
    return safegraphs_read_manifest(path, allocator);
}

#pragma endregion Parts

#pragma region Columned Indexes

/** A value codec that also names the dtype of the elements it emits, so a header can
 *  describe its column; elements only, since the prefix sums delimit every value. */
template <typename codec_type_>
concept interchange_value_codec = requires {
    { codec_type_::value_dtype_k } -> std::convertible_to<dtype_t const&>;
};

/** The shape a codec declares, or one element per row when it names none. */
template <typename codec_type_>
constexpr column_shape_t safegraphs_value_shape() noexcept {
    if constexpr (requires { codec_type_::value_shape_k; }) return codec_type_::value_shape_k;
    else return scalar_shape_t {};
}

/** Whether every value of a codec occupies the same bytes, so its column carries no prefix
 *  sums; a codec that declares no shape writes a tape, which always does. */
template <typename codec_type_>
constexpr bool safegraphs_value_is_fixed() noexcept {
    if constexpr (requires { codec_type_::value_shape_k; })
        return regular_width(codec_type_::value_shape_k).has_value();
    else return false;
}

/** The header and extents a columned index's monolith takes. */
struct safegraphs_index_layout_t {

    /** A regular key column, a value column shaped by the codec, and ragged neighbors. */
    safegraphs_header_t header;

    /** Bytes of the data segment, from the key column to the last neighbor ID. */
    std::size_t segment_bytes = 0;

    /** Bytes of the whole monolith, header segment included. */
    std::size_t monolith_bytes = 0;
};

/** Lays out a columned index's monolith: keys regular; values regular when the codec's
 *  shape is fixed and ragged with element prefix sums otherwise; neighbors ragged with
 *  prefix sums; each buffer on an @c alignment boundary. A fixed shape is held to the
 *  index's byte count, so a codec cannot claim a width its values do not have. */
template <typename key_type_, typename node_id_type_, typename codec_type_,
          typename allocator_type_ = std::allocator<byte_t>>
    requires interchange_value_codec<codec_type_>
expected<safegraphs_index_layout_t> safegraphs_index_layout(std::size_t nodes, std::size_t neighbors,
                                                            std::size_t value_bytes, save_config_t const& config,
                                                            std::span<safegraphs_custom_field_t const> custom_fields,
                                                            allocator_type_ allocator = {}) noexcept {
    safegraphs_index_layout_t layout;
    if (status_t const valid = config.validate(); failed(valid)) return {layout, valid};
    constexpr dtype_t key_dtype = safegraphs_key_dtype<key_type_>();
    constexpr dtype_t node_id_dtype = safegraphs_node_id_dtype<node_id_type_>();
    constexpr dtype_t value_dtype = codec_type_::value_dtype_k;
    constexpr column_shape_t shape = safegraphs_value_shape<codec_type_>();
    constexpr std::size_t element_bytes = dtype_bytes(value_dtype);
    constexpr bool fixed_width = safegraphs_value_is_fixed<codec_type_>();
    if (key_dtype == dtype_t::unknown_k || node_id_dtype == dtype_t::unknown_k || !element_bytes)
        return {layout, status_t::config_mismatch_k};

    std::size_t const alignment = config.alignment;
    std::size_t const sums_bytes = (nodes + 1) * sizeof(u64_t);
    safegraphs_header_t& header = layout.header;
    header.alignment = alignment;
    header.key.dtype = key_dtype;
    header.key.data = {0, nodes * sizeof(key_type_)};

    // A fixed shape is a regular column; anything else carries element prefix sums before its rows.
    safegraphs_column_t& value = header.value;
    value.dtype = value_dtype;
    value.shape = shape;
    if constexpr (fixed_width) {
        if (value_bytes != nodes * *regular_width(shape) * element_bytes) return {layout, status_t::config_mismatch_k};
        value.data.first = round_up_to_multiple(header.key.data.bound, alignment);
    }
    else {
        std::size_t const sums_begin = round_up_to_multiple(header.key.data.bound, alignment);
        value.offsets = prefix_sums_t {dtype_t::u64_k, {sums_begin, sums_begin + sums_bytes}};
        value.data.first = round_up_to_multiple(value.offsets->bytes.bound, alignment);
    }
    value.data.bound = value.data.first + value_bytes;

    // Neighbors are one element per node and as many of them as each node wired, so a scalar
    // shape over prefix sums - the interchange's own spelling for a tape.
    safegraphs_column_t& adjacency = header.neighbors;
    std::size_t const adjacency_sums_begin = round_up_to_multiple(value.data.bound, alignment);
    adjacency.dtype = node_id_dtype;
    adjacency.shape = scalar_shape_t {};
    adjacency.offsets = prefix_sums_t {dtype_t::u64_k, {adjacency_sums_begin, adjacency_sums_begin + sums_bytes}};
    adjacency.data.first = round_up_to_multiple(adjacency.offsets->bytes.bound, alignment);
    adjacency.data.bound = adjacency.data.first + neighbors * sizeof(node_id_type_);
    layout.segment_bytes = adjacency.data.bound;
    auto const [data_offset, status] = safegraphs_data_offset(header, custom_fields, allocator);
    if (failed(status)) return {layout, status};
    header.data_offset = data_offset;
    layout.monolith_bytes = header.data_offset + layout.segment_bytes;
    return {layout, status_t::success_k};
}

/** The custom fields a columned index's monolith carries: the metric, the producer, and
 *  the caller's provenance. */
template <typename allocator_type_>
status_t safegraphs_index_fields_(growing_buffer<safegraphs_custom_field_t, allocator_type_>& fields,
                                  std::string_view metric, save_config_t const& config) noexcept {
    if (failed(fields.reserve_exactly(2 + config.provenance.size()))) return status_t::bad_alloc_k;
    fields.push_back_reserved({"metric", metric});
    fields.push_back_reserved({"created_by", "usearch"});
    for (safegraphs_custom_field_t const& field : config.provenance) fields.push_back_reserved(field);
    return status_t::success_k;
}

/** A typed span over @p count elements of the data segment starting at byte @p start. */
template <typename element_type_, typename byte_type_>
std::span<element_type_> safegraphs_typed_span_(std::span<byte_type_> segment, std::size_t start,
                                                std::size_t count) noexcept {
    return {reinterpret_cast<element_type_*>(segment.data() + start), count};
}

/** Bytes a save of @p index under @p config writes; @c contended_k while a plan is open. */
template <typename index_type_, typename codec_type_, typename allocator_type_ = std::allocator<byte_t>>
    requires interchange_value_codec<codec_type_>
expected<std::size_t> safegraphs_storage_requirements(index_type_ const& index, codec_type_ const& codec,
                                                      save_config_t const& config, std::string_view metric,
                                                      allocator_type_ allocator = {}) noexcept {
    auto const [extents, measure_status] = index.columns_extents(codec);
    if (failed(measure_status)) return {0, measure_status};
    growing_buffer<safegraphs_custom_field_t, allocator_type_> fields {allocator};
    if (status_t const named = safegraphs_index_fields_(fields, metric, config); failed(named)) return {0, named};
    auto const [layout, layout_status] =
        safegraphs_index_layout<typename index_type_::key_t, typename index_type_::node_id_t, codec_type_>(
            extents.nodes, extents.neighbors, extents.value_bytes, config, to_span(fields), allocator);
    return {layout.monolith_bytes, layout_status};
}

/**
 *  @brief Frames a columned index as one monolith: a regular key column, a value column shaped
 *      by the codec, and a ragged neighbors column, the metric in the metadata.
 *
 *  A target lending its bytes in place hands the plan its data segment directly, so regions
 *  write zero-copy; any other target stages the segment and writes it once. The index counts
 *  value bytes, the interchange counts elements, so the value prefix sums stage beside the
 *  segment and land re-expressed once every region finished. The header lands last, so a
 *  failed or cancelled save leaves no magic behind.
 *
 *  @param[inout] monolith The bytes the checkpoint lands in - a buffer, or a file's mapping.
 *  @param[in] index Anything with @c columns_extents, @c plan_save and their column types.
 *  @param[in] codec Encodes each stored value into its slice and names the column's dtype and shape.
 *  @param[inout] executor Runs one region per thread through @c for_tasks.
 *  @param[in] config The alignment and provenance; the layout field is ignored, this is a monolith.
 *  @param[in] metric The token the graph was built for, recorded verbatim.
 *  @param[in] progress Fires per finished region; answering @c cancel_k stops the rest.
 */
template <typename index_type_, typename codec_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t, typename allocator_type_ = std::allocator<byte_t>>
    requires interchange_value_codec<codec_type_>
status_t safegraphs_write_index(bytes_span_t monolith, index_type_ const& index, codec_type_ const& codec,
                                executor_type_& executor, save_config_t const& config, std::string_view metric,
                                progress_type_&& progress = {}, allocator_type_ allocator = {}) noexcept {
    using key_t = typename index_type_::key_t;
    using node_id_t = typename index_type_::node_id_t;
    using columns_t = typename index_type_::columns_t;
    constexpr column_shape_t shape = safegraphs_value_shape<codec_type_>();
    constexpr std::size_t element_bytes = dtype_bytes(codec_type_::value_dtype_k);

    auto const [extents, measure_status] = index.columns_extents(codec);
    if (failed(measure_status)) return measure_status;
    growing_buffer<safegraphs_custom_field_t, allocator_type_> fields {allocator};
    if (status_t const named = safegraphs_index_fields_(fields, metric, config); failed(named)) return named;
    auto const [layout, layout_status] = safegraphs_index_layout<key_t, node_id_t, codec_type_>(
        extents.nodes, extents.neighbors, extents.value_bytes, config, to_span(fields), allocator);
    if (failed(layout_status)) return layout_status;
    safegraphs_header_t const& header = layout.header;
    std::size_t const nodes = extents.nodes;

    // The caller handed us the bytes, and a freshly grown file or buffer reads as zeros, so the
    // gaps between columns need no sweeping.
    if (monolith.size() < layout.monolith_bytes) return status_t::overflow_k;
    bytes_span_t const segment = monolith.subspan(header.data_offset, layout.segment_bytes);
    if (reinterpret_cast<std::uintptr_t>(segment.data()) % alignof(u64_t)) return status_t::config_mismatch_k;

    // Neighbor sums count IDs on both sides, so the plan writes the monolith's own buffer. Value
    // sums count bytes here and elements on the wire, so a fixed column has none to write and a
    // ragged one of wider elements stages them and lands them scaled after the plan.
    std::span<u64_t> const neighbor_sums = safegraphs_typed_span_<u64_t>(segment, header.neighbors.offsets->bytes.first,
                                                                         nodes + 1);
    growing_buffer<u64_t, allocator_type_> staged_value_sums {allocator};
    if (failed(staged_value_sums.resize(nodes + 1))) return status_t::bad_alloc_k;
    std::span<u64_t> const value_sums = element_bytes == 1 && header.value.offsets
                                            ? safegraphs_typed_span_<u64_t>(segment, header.value.offsets->bytes.first,
                                                                            nodes + 1)
                                            : std::span<u64_t> {staged_value_sums};
    columns_t const columns {
        safegraphs_typed_span_<key_t>(segment, header.key.data.first, nodes),
        neighbor_sums,
        safegraphs_typed_span_<node_id_t>(segment, header.neighbors.data.first, extents.neighbors),
        value_sums,
        segment.subspan(header.value.data.first, extents.value_bytes),
    };
    auto planned = index.plan_save(columns, codec);
    if (failed(planned.status)) return planned.status;

    // One region per thread, so nothing allocates and the fan-out needs no scratch.
    std::size_t const regions = max_of<std::size_t>(executor.size(), 1);
    std::size_t const per_region = divide_round_up(nodes, regions);
    status_t outcome = status_t::success_k;
    auto save_region = [&](std::size_t region, std::size_t, std::size_t) noexcept {
        if (failed(standard_atomic_ref<status_t>(outcome).load(std::memory_order_relaxed))) return;
        std::size_t const first = min_of(region * per_region, nodes);
        std::size_t const bound = min_of(first + per_region, nodes);
        status_t saved = planned.value.save(
            iota_view<node_id_t> {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)}, codec);
        if (succeeded(saved) && progress(region) == continuation_t::cancel_k) saved = status_t::cancelled_k;
        status_t expected_status = status_t::success_k;
        if (failed(saved))
            standard_atomic_ref<status_t>(outcome).compare_exchange_strong(
                expected_status, saved, std::memory_order_relaxed, std::memory_order_relaxed);
    };
    if (status_t const fanned = executor.for_tasks(regions, save_region); failed(fanned)) return fanned;
    if (status_t const settled = standard_atomic_ref<status_t>(outcome).load(std::memory_order_relaxed);
        failed(settled))
        return settled;
    if (status_t const finished = planned.value.finish(); failed(finished)) return finished;

    // A fixed shape holds every value to its width; a wider ragged element scales the staged sums.
    if constexpr (safegraphs_value_is_fixed<codec_type_>()) {
        u64_t const row_bytes = *regular_width(shape) * element_bytes;
        for (std::size_t node = 0; node != nodes; ++node)
            if (value_sums[node + 1] - value_sums[node] != row_bytes) return status_t::config_mismatch_k;
    }
    else if (element_bytes != 1) {
        std::span<u64_t> const value_elements = safegraphs_typed_span_<u64_t>(
            segment, header.value.offsets->bytes.first, nodes + 1);
        for (std::size_t node = 0; node <= nodes; ++node) {
            if (value_sums[node] % element_bytes) return status_t::config_mismatch_k;
            value_elements[node] = value_sums[node] / element_bytes;
        }
    }

    return safegraphs_write_header(monolith, header, to_span(fields), allocator);
}

/** What a columned index's monolith says about itself without touching the payload. */
struct safegraphs_index_metadata_t {

    /** Nodes the monolith holds. */
    std::size_t nodes = 0;

    /** Neighbor IDs across every node. */
    std::size_t neighbors = 0;

    /** Value bytes across every node - what a tape must hold to load them. */
    std::size_t value_bytes = 0;

    /** The header's buffer alignment. */
    std::size_t alignment = 0;

    /** Element type of the key column. */
    dtype_t key_dtype = dtype_t::unknown_k;

    /** Element type of the value column. */
    dtype_t value_dtype = dtype_t::unknown_k;

    /** Element type of the neighbors column. */
    dtype_t neighbors_dtype = dtype_t::unknown_k;

    /** The value column's shape, in the interchange's own three cases. */
    column_shape_t value_shape;

    /** The metric token the producer recorded; empty when it omitted one. */
    safegraphs_name_t metric;
};

/** Reads a columned index's header from @p source into its metadata. */
template <typename allocator_type_ = std::allocator<byte_t>>
expected<safegraphs_index_metadata_t> safegraphs_read_index_metadata(bytes_view_t monolith,
                                                                     allocator_type_ allocator = {}) noexcept {
    safegraphs_index_metadata_t metadata;
    auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") metadata.metric = safegraphs_name_t {field.value};
    };
    auto const [header, parse_status] = safegraphs_read_header(monolith, allocator, per_custom_field);
    if (failed(parse_status)) return {metadata, parse_status};
    safegraphs_column_t const& value = header.value;
    std::size_t const id_bytes = dtype_bytes(header.neighbors.dtype);
    metadata.nodes = header.nodes();
    metadata.neighbors = id_bytes ? header.neighbors.data.size() / id_bytes : 0;
    metadata.value_bytes = value.data.size();
    metadata.alignment = header.alignment;
    metadata.key_dtype = header.key.dtype;
    metadata.value_dtype = value.dtype;
    metadata.neighbors_dtype = header.neighbors.dtype;
    metadata.value_shape = value.shape;
    return {metadata, status_t::success_k};
}

/**
 *  @brief Restores a columned index from one monolith: the header is vetted against the codec's
 *      dtype and shape before any payload byte, the index is reserved when it has no capacity,
 *      and regions decode in parallel.
 *
 *  A source lending its bytes in place hands the plan its data segment directly; any other
 *  source stages the segment in one read. The interchange counts elements and the index bytes,
 *  so the value prefix sums are re-expressed on the way in - synthesized for a fixed shape,
 *  scaled from the header's sums otherwise.
 *
 *  @param[in] monolith The checkpoint's bytes - a buffer, or a file's mapping.
 *  @param[inout] index Anything with @c capacity, @c reserve, @c plan_load and their column types.
 *  @param[in] codec Decodes each value slice into its record and names the expected dtype and shape.
 *  @param[inout] executor Runs one region per thread through @c for_tasks.
 *  @param[in] config The reservation to take when the index has none; zero fits the monolith.
 *  @param[in] progress Fires per finished region; answering @c cancel_k stops the rest.
 */
template <typename index_type_, typename codec_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t, typename allocator_type_ = std::allocator<byte_t>>
    requires interchange_value_codec<codec_type_>
status_t safegraphs_read_index(bytes_view_t monolith, index_type_& index, codec_type_ const& codec,
                               executor_type_& executor, load_config_t const& config, progress_type_&& progress = {},
                               allocator_type_ allocator = {}) noexcept {
    using key_t = typename index_type_::key_t;
    using node_id_t = typename index_type_::node_id_t;
    using columns_view_t = typename index_type_::columns_view_t;
    constexpr column_shape_t shape = safegraphs_value_shape<codec_type_>();
    constexpr std::size_t element_bytes = dtype_bytes(codec_type_::value_dtype_k);

    auto const [header, parse_status] = safegraphs_read_header(monolith, allocator);
    if (failed(parse_status)) return parse_status;

    // Every column's dtype, layout and extent is vetted before a payload byte lands.
    constexpr bool fixed_width = safegraphs_value_is_fixed<codec_type_>();
    safegraphs_column_t const& value = header.value;
    safegraphs_column_t const& adjacency = header.neighbors;
    if (header.key.offsets || header.key.dtype != safegraphs_key_dtype<key_t>() ||
        header.key.shape != column_shape_t {scalar_shape_t {}})
        return status_t::config_mismatch_k;
    if (header.key.data.size() % sizeof(key_t)) return status_t::malformed_k;
    std::size_t const nodes = header.key.data.size() / sizeof(key_t);
    std::size_t const sums_bytes = (nodes + 1) * sizeof(u64_t);
    if (value.dtype != codec_type_::value_dtype_k || value.offsets.has_value() == fixed_width || value.shape != shape)
        return status_t::config_mismatch_k;
    if (!adjacency.offsets || adjacency.dtype != safegraphs_node_id_dtype<node_id_t>() ||
        adjacency.shape != column_shape_t {scalar_shape_t {}})
        return status_t::config_mismatch_k;
    if constexpr (fixed_width) {
        if (value.data.size() != nodes * *regular_width(shape) * element_bytes) return status_t::malformed_k;
    }
    else {
        if (value.offsets->dtype != dtype_t::u64_k) return status_t::config_mismatch_k;
        if (value.offsets->bytes.size() != sums_bytes || value.data.size() % element_bytes)
            return status_t::malformed_k;
    }
    if (adjacency.offsets->dtype != dtype_t::u64_k) return status_t::config_mismatch_k;
    if (adjacency.offsets->bytes.size() != sums_bytes || adjacency.data.size() % sizeof(node_id_t))
        return status_t::malformed_k;
    std::size_t const neighbors = adjacency.data.size() / sizeof(node_id_t);
    if (header.key.data.first % alignof(u64_t) || value.data.first % alignof(u64_t) ||
        adjacency.offsets->bytes.first % alignof(u64_t) || adjacency.data.first % alignof(u64_t) ||
        (value.offsets && value.offsets->bytes.first % alignof(u64_t)))
        return status_t::config_mismatch_k;
    if (header.monolith_bytes() > monolith.size()) return status_t::malformed_k;
    std::size_t const segment_bytes = header.monolith_bytes() - header.data_offset;

    if (!index.capacity())
        if (auto const reserved = index.reserve(config.capacity ? config.capacity : max_of<std::size_t>(nodes, 1));
            failed(reserved.status))
            return reserved.status;

    bytes_view_t const segment = monolith.subspan(header.data_offset, segment_bytes);
    if (reinterpret_cast<std::uintptr_t>(segment.data()) % alignof(u64_t)) return status_t::config_mismatch_k;

    // Neighbor sums are the file's own buffer; @c plan_load proves they start at zero, never
    // fall and end at the ID count. Value sums count bytes here, so a fixed shape synthesizes
    // them and a ragged one of wider elements scales them up.
    std::span<u64_t const> const neighbor_sums = safegraphs_typed_span_<u64_t const>(
        segment, adjacency.offsets->bytes.first, nodes + 1);
    growing_buffer<u64_t, allocator_type_> staged_value_sums {allocator};
    std::span<u64_t const> value_sums;
    if constexpr (fixed_width) {
        if (failed(staged_value_sums.resize(nodes + 1))) return status_t::bad_alloc_k;
        for (std::size_t node = 0; node <= nodes; ++node)
            staged_value_sums[node] = node * *regular_width(shape) * element_bytes;
        value_sums = std::span<u64_t const> {staged_value_sums};
    }
    else if (element_bytes == 1)
        value_sums = safegraphs_typed_span_<u64_t const>(segment, value.offsets->bytes.first, nodes + 1);
    else {
        std::span<u64_t const> const value_elements = safegraphs_typed_span_<u64_t const>(
            segment, value.offsets->bytes.first, nodes + 1);
        std::size_t const elements = value.data.size() / element_bytes;
        if (failed(staged_value_sums.resize(nodes + 1))) return status_t::bad_alloc_k;
        for (std::size_t node = 0; node <= nodes; ++node) {
            if (value_elements[node] > elements) return status_t::malformed_k;
            staged_value_sums[node] = value_elements[node] * element_bytes;
        }
        value_sums = std::span<u64_t const> {staged_value_sums};
    }

    columns_view_t const columns {
        safegraphs_typed_span_<key_t const>(segment, header.key.data.first, nodes),
        neighbor_sums,
        safegraphs_typed_span_<node_id_t const>(segment, adjacency.data.first, neighbors),
        value_sums,
        segment.subspan(value.data.first, value.data.size()),
    };
    auto planned = index.plan_load(columns);
    if (failed(planned.status)) return planned.status;

    std::size_t const regions = max_of<std::size_t>(executor.size(), 1);
    std::size_t const per_region = divide_round_up(nodes, regions);
    status_t outcome = status_t::success_k;
    auto load_region = [&](std::size_t region, std::size_t, std::size_t) noexcept {
        if (failed(standard_atomic_ref<status_t>(outcome).load(std::memory_order_relaxed))) return;
        std::size_t const first = min_of(region * per_region, nodes);
        std::size_t const bound = min_of(first + per_region, nodes);
        status_t loaded = planned.value.load(
            iota_view<node_id_t> {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)}, codec);
        if (succeeded(loaded) && progress(region) == continuation_t::cancel_k) loaded = status_t::cancelled_k;
        status_t expected_status = status_t::success_k;
        if (failed(loaded))
            standard_atomic_ref<status_t>(outcome).compare_exchange_strong(
                expected_status, loaded, std::memory_order_relaxed, std::memory_order_relaxed);
    };
    if (status_t const fanned = executor.for_tasks(regions, load_region); failed(fanned)) return fanned;
    if (status_t const settled = standard_atomic_ref<status_t>(outcome).load(std::memory_order_relaxed);
        failed(settled))
        return settled;
    return planned.value.finish();
}

#pragma endregion Columned Indexes

} // namespace usearch
} // namespace unum
