/**
 *  @file include/usearch/utilities/safegraphs_format.hpp
 *  @author Ash Vardanian
 *  @date July 19, 2026
 *  @brief The SafeGraphs interchange - the self-describing JSON header in front of every
 *      <tt>.safegraphs</tt> file and the manifest beside it - read & written positionally through a
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
#include <bit>         // `std::bit_cast`, `std::endian`, `std::has_single_bit`
#include <concepts>    // `std::unsigned_integral`
#include <memory>      // `std::allocator_traits`
#include <optional>    // `std::optional` - a column's prefix sums, present or absent
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <utility>     // `std::forward`, `std::move`
#include <variant>     // `std::variant` - the interchange's three row shapes

#include <yyjson.h> // The JSON reader & writer - exceptions-free C

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the shared outcome

#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t`, `file_span_t`, `copy_bytes`
#include "usearch/utilities/ram_buffers.hpp"    // `growing_buffer` - path formatting & header images
#include "usearch/utilities/types.hpp"

namespace unum::usearch::utilities {

#pragma region Configs

/** One custom @c __metadata__ entry: a producer's own key and its string value. */
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

    /** The header's buffer alignment: a power of two, at least the floor. */
    std::size_t alignment = safegraphs_alignment_default_k;

    /** The contract the caller certifies; @c concurrent_k probes for settlement and may
     *  answer @c contended_k, @c immutable_k copies plain. */
    synchronization_t synchronization = synchronization_t::concurrent_k;

    /** Free-form @c __metadata__ strings - @c created_by, @c embedding_model, @c built_at;
     *  the metric and alignment are the writer's own. */
    std::span<safegraphs_custom_field_t const> provenance = {};

    /**
     *  Whether this config can be written: a legal alignment, and no provenance key the
     *  writer already owns.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The alignment is below the floor or not a power of two.
     *  - @c config_mismatch_k A provenance field names @c metric or @c alignment.
     */
    constexpr status_t validate() const noexcept {
        if (alignment < safegraphs_alignment_floor_k || !std::has_single_bit(alignment))
            return status_t::config_mismatch_k;
        for (safegraphs_custom_field_t const& field : provenance)
            if (field.key == "metric" || field.key == "alignment") return status_t::config_mismatch_k;
        return status_t::success_k;
    }
};

/** How much room a load leaves; each part lands on a shard of its own. */
struct load_config_t {

    /** Entries to reserve across the shards; zero fits the checkpoint exactly and leaves
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

/** A column a producer adds beside the graph's three, named in the header. */
struct safegraphs_named_column_t {
    safegraphs_name_t name;
    safegraphs_column_t column;
};

/** A parsed monolith header: the metadata, the three graph columns and any producer columns, and
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

    /** Producer columns by name, in header order; a reader takes the ones it knows. */
    std::span<safegraphs_named_column_t const> extra_columns;

    /** The producer column named @p name, or none. */
    safegraphs_column_t const* extra_column(std::string_view name) const noexcept {
        for (safegraphs_named_column_t const& extra : extra_columns)
            if (extra.name.view() == name) return &extra.column;
        return nullptr;
    }

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
        std::size_t furthest = max_of(max_of(end_of(key), end_of(value)), end_of(neighbors));
        for (safegraphs_named_column_t const& extra : extra_columns) furthest = max_of(furthest, end_of(extra.column));
        return data_offset + furthest;
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
 *  @param[in] per_custom_field Fires once per string-valued @c __metadata__ entry; its views
 *      die with this call.
 *  @param[out] extra_columns Takes the producer columns, appended, which the header's
 *      @c extra_columns then views until the buffer grows; without it the header lists none.
 *
 *  @par Statuses:
 *  - @c malformed_k The bytes are shorter than the prefix, carry another magic, or declare a
 *    header past their end.
 *  - @c malformed_k The JSON is not an object, declares an alignment that is not a power of two,
 *    or holds a column that does not parse.
 */
template <typename allocator_type_ = std::allocator<byte_t>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_header_t> safegraphs_read_header(
    bytes_view_t monolith, allocator_type_ allocator = {}, callback_type_&& per_custom_field = {},
    growing_buffer<safegraphs_named_column_t, allocator_type_>* extra_columns = nullptr) noexcept {
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
        if (yyjson_val* alignment = yyjson_obj_get(metadata, "alignment")) {
            std::size_t declared = 0;
            if (!safegraphs_addressable_(yyjson_get_uint(alignment), declared) || !std::has_single_bit(declared)) {
                yyjson_doc_free(document);
                return {header, status_t::malformed_k};
            }
            header.alignment = max_of<std::size_t>(declared, safegraphs_alignment_floor_k);
        }
        yyjson_obj_iter entries = yyjson_obj_iter_with(metadata);
        while (yyjson_val* key = yyjson_obj_iter_next(&entries))
            if (yyjson_val* value = yyjson_obj_iter_get_val(key); yyjson_is_str(value))
                per_custom_field(safegraphs_custom_field_t {safegraphs_string_(key), safegraphs_string_(value)});
    }
    // Each column parses whole, and `malformed_k` is the only verdict any of them can reach.
    auto const [key, key_status] = safegraphs_parse_column_(yyjson_obj_get(root, "key"));
    auto const [value, value_status] = safegraphs_parse_column_(yyjson_obj_get(root, "value"));
    auto const [neighbors, neighbors_status] = safegraphs_parse_column_(yyjson_obj_get(root, "neighbors"));
    // Every other column object is a producer's, kept in header order when the caller takes them.
    auto take_extra_columns = [&]() noexcept {
        if (!extra_columns) return status_t::success_k;
        yyjson_obj_iter entries = yyjson_obj_iter_with(root);
        while (yyjson_val* name = yyjson_obj_iter_next(&entries)) {
            std::string_view const label = safegraphs_string_(name);
            yyjson_val* const object = yyjson_obj_iter_get_val(name);
            if (!yyjson_is_obj(object) || label == "__metadata__" || label == "key" || label == "value" ||
                label == "neighbors")
                continue;
            auto const [extra, extra_status] = safegraphs_parse_column_(object);
            if (failed(extra_status) || label.size() >= safegraphs_name_capacity_k) return status_t::malformed_k;
            if (failed(extra_columns->push_back(safegraphs_named_column_t {safegraphs_name_t {label}, extra})))
                return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    };
    std::size_t const extras_first = extra_columns ? extra_columns->size() : 0;
    status_t const extras_status = take_extra_columns();
    yyjson_doc_free(document);
    if (failed(key_status) || failed(value_status) || failed(neighbors_status)) return {header, status_t::malformed_k};
    if (failed(extras_status)) return {header, extras_status};
    header.key = key, header.value = value, header.neighbors = neighbors;
    if (extra_columns) header.extra_columns = to_span(*extra_columns).subspan(extras_first);

    // The data segment starts on the declared alignment boundary at or past the JSON's end.
    header.data_offset = round_up_to_multiple<std::size_t>(sizeof(prefix) + header_length, header.alignment);
    return {header, status_t::success_k};
}

/**
 *  Maps @p path read-only and parses its header.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k The path does not parse.
 *  - @c malformed_k The file is empty, or its header does not parse.
 */
template <typename allocator_type_ = std::allocator<byte_t>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_header_t> safegraphs_read_header(
    std::string_view path, allocator_type_ allocator = {}, callback_type_&& per_custom_field = {},
    growing_buffer<safegraphs_named_column_t, allocator_type_>* extra_columns = nullptr) noexcept {
    auto [mapping, map_status] = mapped_file_t::read(path);
    if (failed(map_status)) return {safegraphs_header_t {}, map_status};
    return safegraphs_read_header(mapping.view(), allocator, std::forward<callback_type_>(per_custom_field),
                                  extra_columns);
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
    auto add_column = [&](std::string_view label, safegraphs_column_t const& column) noexcept {
        yyjson_mut_val* const name = yyjson_mut_strncpy(document, label.data(), label.size());
        yyjson_mut_val* const object = yyjson_mut_obj(document);
        bool added = name && object && yyjson_mut_obj_add(root, name, object) &&
                     yyjson_mut_obj_add_str(document, object, "dtype", dtype_name(column.dtype)) &&
                     add_shape(object, column.shape);
        if (column.offsets)
            added = added &&
                    yyjson_mut_obj_add_str(document, object, "offsets_dtype", dtype_name(column.offsets->dtype)) &&
                    add_span(object, "offsets", column.offsets->bytes);
        return added && add_span(object, "data", column.data);
    };
    complete = complete && add_column("key", header.key) && add_column("value", header.value) &&
               add_column("neighbors", header.neighbors);
    for (safegraphs_named_column_t const& extra : header.extra_columns)
        complete = complete && add_column(extra.name.view(), extra.column);
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

/**
 *  Where a header lands its data segment: the prefix and JSON rounded up to the declared
 *  alignment.
 *
 *  @par Statuses:
 *  - @c config_mismatch_k The header's alignment is not a power of two.
 */
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

/** Writes the prefix, the JSON, and space padding up to @c header.data_offset over the front of
 *  @p monolith - exactly the bytes @ref unum::usearch::utilities::safegraphs_read_header consumes.
 *  Written last, it is what turns a run of bytes into a checkpoint.
 *
 *  @par Statuses:
 *  - @c config_mismatch_k @c header.data_offset falls inside the prefix.
 *  - @c overflow_k @p monolith is shorter than @c header.data_offset.
 *  - @c config_mismatch_k The JSON does not fit before @c header.data_offset.
 *  - @c uninitialized_k A file-backed @p monolith names no file.
 */
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

/** Whether a key column of @p stored dtype loads into keys of @p wanted dtype: the same token, or a
 *  narrower one zero-extended - @c u32 into @c u64 or @c uuid, @c u64 into @c uuid. */
constexpr bool safegraphs_key_widens(dtype_t stored, dtype_t wanted) noexcept {
    if (stored == wanted) return true;
    if (stored == dtype_t::u32_k) return wanted == dtype_t::u64_k || wanted == dtype_t::uuid_k;
    return stored == dtype_t::u64_k && wanted == dtype_t::uuid_k;
}

/** The key a column of @p stored dtype holds at @p source, widened as @ref safegraphs_key_widens
 *  admits: copied as is at the key's own dtype, else read as a little-endian integer and
 *  zero-extended - numerically into an integer key, through @ref uuid_of into a @c uuid. */
template <typename key_type_>
key_type_ safegraphs_widened_key(byte_t const* source, dtype_t stored) noexcept {
    key_type_ key;
    if (stored == safegraphs_key_dtype<key_type_>()) return std::memcpy(&key, source, sizeof(key)), key;
    u64_t number;
    if (stored == dtype_t::u32_k) {
        u32_t narrow;
        std::memcpy(&narrow, source, sizeof(narrow));
        number = narrow;
    }
    else std::memcpy(&number, source, sizeof(number));
    if constexpr (std::integral<key_type_>) return static_cast<key_type_>(number);
    else return uuid_of(number);
}

/** The interchange token of one node ID: the interchange admits @c u32 and @c u64. */
template <typename node_id_type_>
constexpr dtype_t safegraphs_node_id_dtype() noexcept {
    if constexpr (sizeof(node_id_type_) == 4) return dtype_t::u32_k;
    else if constexpr (sizeof(node_id_type_) == 8) return dtype_t::u64_k;
    else return dtype_t::unknown_k;
}

#pragma endregion Columns

#pragma region Parts

/** One monolith of a checkpoint: a manifest's file entry, or the lone monolith itself. */
struct safegraphs_part_t {

    /** The monolith's path, relative to the manifest; empty for a lone monolith and for a part of
     *  a buffer manifest. */
    std::string_view path;

    /** The monolith's whole size in bytes. */
    std::size_t bytes = 0;

    /** Nodes the monolith holds. */
    std::size_t nodes = 0;

    /** Where a buffer manifest's part starts inside the buffer it leads. */
    std::size_t offset = 0;
};

/** What a manifest or a lone monolith says about itself without opening a column. */
struct safegraphs_metadata_t {

    /** The metric the graph was built for; empty when the producer omitted it. */
    safegraphs_name_t metric;

    /** Element type of every part's value column - the stored scalar's own token. */
    dtype_t value_dtype = dtype_t::unknown_k;

    /** Element type of every part's key column. */
    dtype_t key_dtype = dtype_t::unknown_k;

    /** Element type of every part's neighbor IDs, @c u32 or @c u64. */
    dtype_t neighbors_dtype = dtype_t::unknown_k;

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

/** Whether @p bytes open with the monolith magic; anything else a reader takes for a manifest. */
inline bool safegraphs_is_monolith(bytes_view_t bytes) noexcept {
    return bytes.size() >= safegraphs_prefix_t::magic_k.size() &&
           std::memcmp(bytes.data(), safegraphs_prefix_t::magic_k.data(), safegraphs_prefix_t::magic_k.size()) == 0;
}

/** Formats the stem and @c -00001-of-00002.safegraphs into @p buffer, measured first; parts
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

/** Builds a manifest document; null when an allocation failed. It catalogs parts and nothing else:
 *  the schema, the metric and the provenance live in each part's own header, so a part lifted out
 *  of the set still describes itself completely. An entry without a path is a part of a buffer
 *  manifest, named by its offset inside that buffer. */
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
        complete = (entry.path.empty()
                        ? yyjson_mut_obj_add_uint(document, object, "offset", entry.offset)
                        : yyjson_mut_obj_add_strncpy(document, object, "path", entry.path.data(), entry.path.size())) &&
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
inline safegraphs_metadata_t safegraphs_metadata_of_(safegraphs_header_t const& header,
                                                     safegraphs_name_t metric) noexcept {
    safegraphs_metadata_t metadata;
    metadata.metric = metric;
    metadata.value_dtype = header.value.dtype;
    metadata.key_dtype = header.key.dtype;
    metadata.neighbors_dtype = header.neighbors.dtype;
    metadata.value_shape = header.value.shape;
    metadata.connectivity = regular_width(header.neighbors.shape).value_or(0);
    metadata.nodes = header.nodes();
    metadata.parts = 1;
    return metadata;
}

/** Reads a manifest: @p per_part fires once per entry in file order with a path that dies
 *  with this call, and the metadata comes from the first part's header, the one place a
 *  schema is written. Node counts sum over the entries that declare one.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k The manifest or first part path does not parse.
 *  - @c malformed_k The manifest is empty or not a JSON object, an entry's counts are unreadable,
 *    or no entry names a path.
 *  - @c malformed_k The first part is empty, or its header does not parse.
 */
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
    auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") metric = safegraphs_name_t {field.value};
    };
    auto const [header, header_status] = safegraphs_read_header(first_path, allocator, per_custom_field);
    if (failed(header_status)) return {metadata, header_status};
    std::size_t const parts = metadata.parts;
    std::size_t const nodes = metadata.nodes;
    metadata = safegraphs_metadata_of_(header, metric);
    metadata.parts = parts;
    metadata.nodes = nodes ? nodes : metadata.nodes;
    return {metadata, status_t::success_k};
}

/**
 *  Writes a manifest over @p parts entries drawn from @p part_at in order - the commit
 *  record a distributed save writes last.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k The path does not parse.
 *  - @c overflow_k A device ends before its base plus the manifest.
 */
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

/** Lays @p parts monoliths, @p part_at answering each one's bytes and nodes, behind the manifest
 *  that leads one buffer: fills each part's offset into @p offsets, every one on an @p alignment
 *  boundary past the manifest and the zero byte that ends it, and answers the buffer's whole
 *  size.
 *
 *  @par Statuses:
 *  - @c config_mismatch_k @p parts is zero, @p offsets holds fewer slots, or @p alignment is not
 *    a power of two.
 */
template <typename callback_type_, typename allocator_type_ = std::allocator<byte_t>>
expected<std::size_t> safegraphs_buffer_layout(std::size_t parts, callback_type_&& part_at, std::size_t alignment,
                                               std::span<std::size_t> offsets,
                                               allocator_type_ allocator = {}) noexcept {
    if (!parts || offsets.size() < parts || !std::has_single_bit(alignment)) return {0, status_t::config_mismatch_k};
    // Wider offsets print longer, so the manifest is measured again until the parts clear it.
    for (std::size_t lead = 0;;) {
        std::size_t end = 0;
        for (std::size_t part = 0, next = round_up_to_multiple(lead + 1, alignment); part != parts; ++part) {
            offsets[part] = next;
            end = next + part_at(part).bytes;
            next = round_up_to_multiple(end, alignment);
        }
        auto entry_at = [&](std::size_t part) noexcept {
            safegraphs_part_t entry = part_at(part);
            entry.path = {};
            entry.offset = offsets[part];
            return entry;
        };
        auto const [manifest, status] = safegraphs_manifest_bytes(parts, entry_at, {}, allocator);
        if (failed(status)) return {0, status};
        if (manifest <= lead) return {end, status_t::success_k};
        lead = manifest;
    }
}

/** Writes the manifest leading @p buffer over @p parts entries drawn from @p part_at, each naming
 *  its offset, and zeroes the bytes up to the first part, where a reader finds the manifest's end;
 *  @c overflow_k when it does not fit before that part.
 *
 *  @par Statuses:
 *  - @c config_mismatch_k @p parts is zero.
 *  - @c overflow_k The first part starts past @p buffer.
 *  - @c overflow_k The manifest and its zero byte do not fit before the first part.
 */
template <typename callback_type_, typename allocator_type_ = std::allocator<byte_t>>
status_t safegraphs_write_manifest(bytes_span_t buffer, std::size_t parts, callback_type_&& part_at,
                                   allocator_type_ allocator = {}) noexcept {
    if (!parts) return status_t::config_mismatch_k;
    std::size_t const first = part_at(0).offset;
    if (first > buffer.size()) return status_t::overflow_k;
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    yyjson_alc const allocator_abi = trampoline.c_abi();
    return safegraphs_print_(safegraphs_manifest_document_(parts, part_at, {}, allocator_abi), allocator_abi,
                             [&](std::string_view json) noexcept {
                                 if (json.size() >= first) return status_t::overflow_k;
                                 std::memcpy(buffer.data(), json.data(), json.size());
                                 std::memset(buffer.data() + json.size(), 0, first - json.size());
                                 return status_t::success_k;
                             });
}

/** Reads the manifest leading @p buffer, which ends at its first zero byte: @p per_part fires once
 *  per entry in file order with its offset, every part inside the buffer, and the metadata comes
 *  from the first part's header.
 *
 *  @par Statuses:
 *  - @c malformed_k No zero byte ends the manifest.
 *  - @c malformed_k The manifest is not a JSON object, an entry's offset or counts are unreadable
 *    or fall outside @p buffer, or the first part is empty.
 *  - @c malformed_k The first part's header does not parse.
 */
template <typename allocator_type_ = std::allocator<byte_t>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_metadata_t> safegraphs_read_manifest(bytes_view_t buffer, allocator_type_ allocator = {},
                                                         callback_type_&& per_part = {}) noexcept {
    safegraphs_metadata_t metadata;
    void const* const ending = std::memchr(buffer.data(), 0, buffer.size());
    if (!ending) return {metadata, status_t::malformed_k};
    std::size_t const json_bytes = static_cast<std::size_t>(static_cast<byte_t const*>(ending) - buffer.data());
    safegraphs_yyjson_allocator_<allocator_type_> trampoline {allocator};
    auto [document, parse_status] = safegraphs_read_document_(buffer.first(json_bytes), trampoline);
    if (failed(parse_status)) return {metadata, parse_status};

    yyjson_val* files = yyjson_obj_get(yyjson_doc_get_root(document), "files");
    metadata.parts = yyjson_arr_size(files);
    bytes_view_t first;
    for (std::size_t part = 0; part != metadata.parts; ++part) {
        yyjson_val* entry = yyjson_arr_get(files, part);
        safegraphs_part_t found;
        if (!yyjson_is_uint(yyjson_obj_get(entry, "offset")) ||
            !safegraphs_addressable_(yyjson_get_uint(yyjson_obj_get(entry, "offset")), found.offset) ||
            !safegraphs_addressable_(yyjson_get_uint(yyjson_obj_get(entry, "bytes")), found.bytes) ||
            !safegraphs_addressable_(yyjson_get_uint(yyjson_obj_get(entry, "nodes")), found.nodes) ||
            found.offset < json_bytes || found.offset > buffer.size() || found.bytes > buffer.size() - found.offset) {
            yyjson_doc_free(document);
            return {metadata, status_t::malformed_k};
        }
        if (!part) first = buffer.subspan(found.offset, found.bytes);
        metadata.nodes += found.nodes;
        per_part(found);
    }
    yyjson_doc_free(document);
    if (first.empty()) return {metadata, status_t::malformed_k};

    // The schema lives in the parts, so the first one's header answers every question about it.
    safegraphs_name_t metric;
    auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") metric = safegraphs_name_t {field.value};
    };
    auto const [header, header_status] = safegraphs_read_header(first, allocator, per_custom_field);
    if (failed(header_status)) return {metadata, header_status};
    std::size_t const parts = metadata.parts, nodes = metadata.nodes;
    metadata = safegraphs_metadata_of_(header, metric);
    metadata.parts = parts;
    metadata.nodes = nodes;
    return {metadata, status_t::success_k};
}

/**
 *  The metadata of the checkpoint held in @p monolith: a lone monolith, or the manifest leading a
 *  buffer of parts.
 *
 *  @par Statuses:
 *  - @c malformed_k The manifest or the header does not parse.
 */
template <typename allocator_type_ = std::allocator<byte_t>>
expected<safegraphs_metadata_t> safegraphs_read_metadata(bytes_view_t monolith,
                                                         allocator_type_ allocator = {}) noexcept {
    safegraphs_name_t metric;
    auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") metric = safegraphs_name_t {field.value};
    };
    if (!safegraphs_is_monolith(monolith)) return safegraphs_read_manifest(monolith, allocator);
    auto const [header, parse_status] = safegraphs_read_header(monolith, allocator, per_custom_field);
    if (failed(parse_status)) return {safegraphs_metadata_t {}, parse_status};
    return {safegraphs_metadata_of_(header, metric), status_t::success_k};
}

/**
 *  The metadata of a manifest or of a lone monolith at @p path - told apart by content,
 *  the monolith's magic, never by the extension.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k A path does not parse.
 *  - @c malformed_k The file is empty, or its manifest or header does not parse.
 */
template <typename allocator_type_ = std::allocator<byte_t>>
expected<safegraphs_metadata_t> safegraphs_read_metadata(std::string_view path,
                                                         allocator_type_ allocator = {}) noexcept {
    auto [mapping, map_status] = mapped_file_t::read(path);
    if (failed(map_status)) return {safegraphs_metadata_t {}, map_status};
    bytes_view_t const monolith = mapping.view();
    if (safegraphs_is_monolith(monolith)) return safegraphs_read_metadata(monolith, allocator);
    return safegraphs_read_manifest(path, allocator);
}

/** One monolith of a checkpoint a load lists: where it is, what its header says, and the metric
 *  token its metadata carries. */
struct safegraphs_catalog_part_t {

    /** The relative path's characters inside the catalog's paths; empty names the checkpoint path
     *  itself. */
    iota_view<std::size_t> path;

    /** The part's whole bytes: a mapping's view, or the caller's own buffer. */
    bytes_view_t monolith;

    /** Rows the monolith holds, per its own header and columns. */
    std::size_t nodes = 0;

    /** The monolith's header. */
    safegraphs_header_t header;

    /** The metric token the monolith recorded; empty when it omitted one. */
    safegraphs_name_t metric;
};

/** Copies @p path into @p buffer, so a plan outlives the caller's view of it. */
template <typename allocator_type_>
status_t safegraphs_keep_path(growing_buffer<char, allocator_type_>& buffer, std::string_view path) noexcept {
    if (failed(buffer.reserve_exactly(path.size()))) return status_t::bad_alloc_k;
    for (char const character : path) buffer.push_back_reserved(character);
    return status_t::success_k;
}

/**
 *  Reads @p part 's header and metric token out of the bytes it carries, parsing on
 *  @p allocator, its producer columns appended to @p extra_columns.
 *
 *  @par Statuses:
 *  - @c malformed_k The header does not parse.
 *  - @c malformed_k The header spans more bytes than the part carries.
 */
template <typename allocator_type_>
status_t safegraphs_describe_part(safegraphs_catalog_part_t& part,
                                  growing_buffer<safegraphs_named_column_t, allocator_type_>& extra_columns,
                                  allocator_type_ allocator) noexcept {
    auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") part.metric = safegraphs_name_t {field.value};
    };
    auto const [header, status] = safegraphs_read_header(part.monolith, allocator, per_custom_field, &extra_columns);
    if (failed(status)) return status;
    if (header.monolith_bytes() > part.monolith.size()) return status_t::malformed_k;
    part.header = header;
    part.nodes = header.nodes();
    return status_t::success_k;
}

/**
 *  Lists @p monolith as a checkpoint's only part.
 *
 *  @par Statuses:
 *  - @c malformed_k The header does not parse, or spans more bytes than @p monolith.
 *  - @c overflow_k The grown capacity of @p parts wraps.
 */
template <typename allocator_type_>
status_t safegraphs_catalog_monolith(bytes_view_t monolith,
                                     growing_buffer<safegraphs_catalog_part_t, allocator_type_>& parts,
                                     growing_buffer<safegraphs_named_column_t, allocator_type_>& extra_columns,
                                     allocator_type_ allocator) noexcept {
    safegraphs_catalog_part_t part;
    part.monolith = monolith;
    if (status_t const described = safegraphs_describe_part(part, extra_columns, allocator); failed(described))
        return described;
    return parts.push_back(part);
}

/** Points every part's header at its own run of @p extra_columns, once the buffer stopped growing:
 *  the parts' runs follow one another in part order. */
template <typename allocator_type_>
void safegraphs_bind_extra_columns(
    growing_buffer<safegraphs_catalog_part_t, allocator_type_>& parts,
    growing_buffer<safegraphs_named_column_t, allocator_type_> const& extra_columns) noexcept {
    std::size_t first = 0;
    for (safegraphs_catalog_part_t& part : parts) {
        std::size_t const count = part.header.extra_columns.size();
        part.header.extra_columns = to_span(extra_columns).subspan(first, count);
        first += count;
    }
}

/**
 *  @brief Lists the parts of the checkpoint at @p path: the magic makes a lone monolith, anything
 *      else a manifest whose every part is mapped and its header read now, so no entry is taken on
 *      trust.
 *
 *  The mappings land in @p mappings, which must outlive every read of the parts, and the producer
 *  columns in @p extra_columns, which every part's header views.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k The checkpoint or a part path does not parse.
 *  - @c malformed_k The checkpoint or a part is empty, its manifest lists no part, or a header
 *    does not parse or spans more bytes than its part.
 *  - @c overflow_k The grown capacity of an output buffer wraps.
 */
template <typename allocator_type_>
status_t safegraphs_catalog(std::string_view path, growing_buffer<mapped_file_t, allocator_type_>& mappings,
                            growing_buffer<safegraphs_catalog_part_t, allocator_type_>& parts,
                            growing_buffer<safegraphs_named_column_t, allocator_type_>& extra_columns,
                            allocator_type_ allocator) noexcept {
    auto [mapping, mapping_status] = mapped_file_t::read(path);
    if (failed(mapping_status)) return mapping_status;
    bytes_view_t const checkpoint = mapping.view();
    if (safegraphs_is_monolith(checkpoint)) {
        if (status_t const listed = safegraphs_catalog_monolith(checkpoint, parts, extra_columns, allocator);
            failed(listed))
            return listed;
        return mappings.push_back(std::move(mapping));
    }

    growing_buffer<char, allocator_type_> paths_arena {allocator};
    status_t kept = status_t::success_k;
    auto per_part = [&](safegraphs_part_t entry) noexcept {
        std::size_t const offset = paths_arena.size();
        for (char const character : entry.path)
            if (failed(paths_arena.push_back(character))) return kept = status_t::bad_alloc_k, void();
        safegraphs_catalog_part_t part;
        part.path = {offset, paths_arena.size()};
        if (failed(parts.push_back(part))) kept = status_t::bad_alloc_k;
    };
    auto const [summary, summary_status] = safegraphs_read_manifest(path, allocator, per_part);
    if (failed(summary_status)) return summary_status;
    if (failed(kept)) return kept;
    if (parts.empty()) return status_t::malformed_k;
    if (failed(mappings.reserve_exactly(parts.size()))) return status_t::bad_alloc_k;
    growing_buffer<char, allocator_type_> part_path {allocator};
    for (safegraphs_catalog_part_t& part : parts) {
        auto const [sibling, sibling_status] = safegraphs_sibling_path(
            part_path, path, to_string_view(paths_arena).substr(part.path.first, part.path.size()));
        if (failed(sibling_status)) return sibling_status;
        auto [part_mapping, part_mapping_status] = mapped_file_t::read(sibling);
        if (failed(part_mapping_status)) return part_mapping_status;
        part.monolith = part_mapping.view();
        if (status_t const described = safegraphs_describe_part(part, extra_columns, allocator); failed(described))
            return described;
        mappings.push_back_reserved(std::move(part_mapping));
    }
    safegraphs_bind_extra_columns(parts, extra_columns);
    return status_t::success_k;
}

/**
 *  Lists the parts of the checkpoint held in @p buffer: the magic makes a lone monolith, anything
 *  else the manifest leading the buffer, whose every part's header is read now, its producer
 *  columns into @p extra_columns.
 *
 *  @par Statuses:
 *  - @c malformed_k The manifest does not parse or lists no part, or a header does not parse or
 *    spans more bytes than its part.
 *  - @c overflow_k The grown capacity of @p parts wraps.
 */
template <typename allocator_type_>
status_t safegraphs_catalog(bytes_view_t buffer, growing_buffer<safegraphs_catalog_part_t, allocator_type_>& parts,
                            growing_buffer<safegraphs_named_column_t, allocator_type_>& extra_columns,
                            allocator_type_ allocator) noexcept {
    if (safegraphs_is_monolith(buffer)) return safegraphs_catalog_monolith(buffer, parts, extra_columns, allocator);
    status_t kept = status_t::success_k;
    auto per_part = [&](safegraphs_part_t entry) noexcept {
        safegraphs_catalog_part_t part;
        part.monolith = buffer.subspan(entry.offset, entry.bytes);
        if (failed(parts.push_back(part))) kept = status_t::bad_alloc_k;
    };
    auto const [summary, summary_status] = safegraphs_read_manifest(buffer, allocator, per_part);
    if (failed(summary_status)) return summary_status;
    if (failed(kept)) return kept;
    if (parts.empty()) return status_t::malformed_k;
    for (safegraphs_catalog_part_t& part : parts)
        if (status_t const described = safegraphs_describe_part(part, extra_columns, allocator); failed(described))
            return described;
    safegraphs_bind_extra_columns(parts, extra_columns);
    return status_t::success_k;
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

/**
 *  Lays out a columned index's monolith: keys regular; values regular when the codec's
 *  shape is fixed and ragged with element prefix sums otherwise; neighbors ragged with
 *  prefix sums; each buffer on an @c alignment boundary. A fixed shape is held to the
 *  index's byte count, so a codec cannot claim a width its values do not have.
 *
 *  @par Statuses:
 *  - @c config_mismatch_k @p config does not validate.
 *  - @c config_mismatch_k The key, node ID or value element has no interchange dtype.
 *  - @c config_mismatch_k A fixed shape disagrees with @p value_bytes.
 */
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

/**
 *  Bytes a save of @p index under @p config writes; @c contended_k while a plan is open.
 *
 *  @par Statuses:
 *  - @c contended_k A mutator, a batch or an open plan holds @p index.
 *  - @c config_mismatch_k On NVIDIA, the shard has no stream.
 *  - @c config_mismatch_k @p config does not validate, the key, node ID or value element has no
 *    interchange dtype, or a fixed shape disagrees with the value bytes.
 */
template <typename index_type_, typename codec_type_, typename allocator_type_ = std::allocator<byte_t>>
    requires interchange_value_codec<codec_type_>
expected<std::size_t> safegraphs_storage_requirements(index_type_& index, codec_type_ const& codec,
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
 *  @param[in,out] monolith The bytes the checkpoint lands in - a buffer, or a file's mapping.
 *  @param[in] index Anything with @c columns_extents, @c plan_save and their column types.
 *  @param[in] codec Encodes each stored value into its slice and names the column's dtype and shape.
 *  @param[in,out] executor Runs one region per thread through @c for_tasks.
 *  @param[in] config The alignment and provenance; the layout field is ignored, this is a monolith.
 *  @param[in] metric The token the graph was built for, recorded verbatim.
 *  @param[in] progress Fires per finished region; answering @c cancel_k stops the rest.
 *  @param[in] allocator The allocator for the call's temporary storage.
 *
 *  A refusal of @p codec passes through.
 *
 *  @par Statuses:
 *  - @c contended_k A mutator, a batch or an open plan holds @p index, or the pool of
 *    @p executor runs another dispatch.
 *  - @c config_mismatch_k On NVIDIA, the shard has no stream.
 *  - @c config_mismatch_k @p config does not validate, an element has no interchange dtype, a
 *    fixed shape or ragged sum disagrees with the value bytes, or the segment is misaligned.
 *  - @c overflow_k @p monolith or a column is shorter than the layout needs.
 *  - @c uninitialized_k The index's plan is spent, or the pool of @p executor has no workers.
 */
template <typename index_type_, typename codec_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t, typename allocator_type_ = std::allocator<byte_t>>
    requires interchange_value_codec<codec_type_>
status_t safegraphs_write_index(bytes_span_t monolith, index_type_& index, codec_type_ const& codec,
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
    using outcome_ref_t = fu::standard_atomic_ref<status_t>;
    status_t outcome = status_t::success_k;
    auto save_region = [&](std::size_t region, std::size_t, std::size_t) noexcept {
        if (failed(outcome_ref_t(outcome).load(std::memory_order_relaxed))) return;
        std::size_t const first = min_of(region * per_region, nodes);
        std::size_t const bound = min_of(first + per_region, nodes);
        status_t saved = planned.value.save(
            iota_view<node_id_t> {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)}, codec);
        if (succeeded(saved) && progress(region) == continuation_t::cancel_k) saved = status_t::cancelled_k;
        status_t expected_status = status_t::success_k;
        if (failed(saved))
            outcome_ref_t(outcome).compare_exchange_strong(expected_status, saved, std::memory_order_relaxed,
                                                           std::memory_order_relaxed);
    };
    if (status_t const fanned = executor.for_tasks(regions, save_region); failed(fanned)) return fanned;
    if (status_t const settled = outcome_ref_t(outcome).load(std::memory_order_relaxed); failed(settled))
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

/**
 *  Writes the monolith at @p path: sized from the index, created fresh, mapped, and flushed once
 *  every region landed. A refusal of @p codec passes through.
 *
 *  @par Statuses:
 *  - @c contended_k A mutator, a batch or an open plan holds @p index, or the pool of
 *    @p executor runs another dispatch.
 *  - @c config_mismatch_k On NVIDIA, the shard has no stream.
 *  - @c config_mismatch_k @p config does not validate, an element has no interchange dtype, a
 *    fixed shape or ragged sum disagrees with the value bytes, or the segment is misaligned.
 *  - @c invalid_argument_k The path does not parse.
 *  - @c overflow_k A device ends before the monolith does, or a column is shorter than the
 *    layout needs.
 *  - @c uninitialized_k The index's plan is spent, or the pool of @p executor has no workers.
 */
template <typename index_type_, typename codec_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t, typename allocator_type_ = std::allocator<byte_t>>
    requires interchange_value_codec<codec_type_>
status_t safegraphs_write_index(std::string_view path, index_type_& index, codec_type_ const& codec,
                                executor_type_& executor, save_config_t const& config, std::string_view metric,
                                progress_type_&& progress = {}, allocator_type_ allocator = {}) noexcept {
    auto const [bytes, measure_status] = safegraphs_storage_requirements(index, codec, config, metric, allocator);
    if (failed(measure_status)) return measure_status;
    auto [mapping, map_status] = mapped_file_t::write(path, bytes);
    if (failed(map_status)) return map_status;
    if (status_t const saved = safegraphs_write_index(mapping.span(), index, codec, executor, config, metric,
                                                      std::forward<progress_type_>(progress), allocator);
        failed(saved))
        return saved;
    return mapping.sync();
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

/**
 *  @brief Reads a columned index's header from @p monolith into its metadata.
 *  @param[in] monolith The checkpoint's bytes - a buffer, or a file's mapping.
 *  @param[in] allocator The allocator for the metadata's own storage.
 *  @param[in] per_custom_field Fires once per string-valued @c __metadata__ entry, the metric
 *      included, for the parameters a modality records beside it; its views die with this call.
 *
 *  @par Statuses:
 *  - @c malformed_k The header does not parse.
 */
template <typename allocator_type_ = std::allocator<byte_t>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_index_metadata_t> safegraphs_read_index_metadata(bytes_view_t monolith,
                                                                     allocator_type_ allocator = {},
                                                                     callback_type_&& per_custom_field = {}) noexcept {
    safegraphs_index_metadata_t metadata;
    auto per_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") metadata.metric = safegraphs_name_t {field.value};
        per_custom_field(field);
    };
    auto const [header, parse_status] = safegraphs_read_header(monolith, allocator, per_field);
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
 *  The metadata of the checkpoint at @p path, mapped read-only: a lone monolith's, or a manifest's
 *  with the counts summed over its parts and the custom fields read off the first one, which every
 *  part repeats.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k The checkpoint or a part path does not parse.
 *  - @c malformed_k The checkpoint or a part is empty, its manifest lists no part, or a header
 *    does not parse or spans more bytes than its part.
 *  - @c overflow_k The grown capacity of a catalog buffer wraps.
 */
template <typename allocator_type_ = std::allocator<byte_t>, typename callback_type_ = dummy_callback_t>
expected<safegraphs_index_metadata_t> safegraphs_read_index_metadata(std::string_view path,
                                                                     allocator_type_ allocator = {},
                                                                     callback_type_&& per_custom_field = {}) noexcept {
    growing_buffer<mapped_file_t, allocator_type_> mappings {allocator};
    growing_buffer<safegraphs_catalog_part_t, allocator_type_> parts {allocator};
    growing_buffer<safegraphs_named_column_t, allocator_type_> extra_columns {allocator};
    if (status_t const listed = safegraphs_catalog(path, mappings, parts, extra_columns, allocator); failed(listed))
        return {safegraphs_index_metadata_t {}, listed};
    auto [metadata, status] = safegraphs_read_index_metadata(parts[0].monolith, allocator,
                                                             std::forward<callback_type_>(per_custom_field));
    for (std::size_t part = 1; part != parts.size() && succeeded(status); ++part) {
        auto const [more, more_status] = safegraphs_read_index_metadata(parts[part].monolith, allocator);
        metadata.nodes += more.nodes, metadata.neighbors += more.neighbors, metadata.value_bytes += more.value_bytes;
        status = more_status;
    }
    return {metadata, status};
}

/**
 *  @brief Restores a columned index from one monolith: the header is vetted against the codec's
 *      dtype and shape before any payload byte, the index is reserved when it has no capacity,
 *      and regions decode in parallel.
 *
 *  A source lending its bytes in place hands the plan its data segment directly; any other
 *  source stages the segment in one read. The interchange counts elements and the index bytes,
 *  so the value prefix sums are re-expressed on the way in - synthesized for a fixed shape,
 *  scaled from the header's sums otherwise. Neighbors load from ragged rows or from fixed-degree
 *  ones, whose sentinels, any ID at or past the node count, are skipped.
 *
 *  @param[in] monolith The checkpoint's bytes - a buffer, or a file's mapping.
 *  @param[in,out] index Anything with @c capacity, @c reserve, @c plan_load and their column types.
 *  @param[in] codec Decodes each value slice into its record and names the expected dtype and shape.
 *  @param[in,out] executor Runs one region per thread through @c for_tasks.
 *  @param[in] config The reservation to take when the index has none; zero fits the monolith.
 *  @param[in] progress Fires per finished region; answering @c cancel_k stops the rest.
 *  @param[in] allocator The allocator for the call's temporary storage.
 *
 *  A refusal of @p codec passes through.
 *
 *  @par Statuses:
 *  - @c malformed_k The header does not parse, or a column's bytes, sums or extents disagree
 *    with its rows or the monolith.
 *  - @c config_mismatch_k A column's dtype, shape or key width differs from what @p index and
 *    @p codec take, the segment is misaligned, @p index already holds entries, or a step's nodes
 *    fall outside the plan.
 *  - @c overflow_k The node count passes the node IDs or the key multimap of @p index.
 *  - @c uninitialized_k On NVIDIA, no reserve froze the connectivity, or the pool of
 *    @p executor has no workers.
 *  - @c unsupported_k On NVIDIA, an entry past every rung of the face's kits.
 *  - @c contended_k A mutator, a batch, a load or a plan holds @p index, or the pool of
 *    @p executor runs another dispatch.
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

    // Every column's dtype, layout and extent is vetted before a payload byte lands. Neighbors
    // without prefix sums are fixed-degree rows.
    constexpr bool fixed_width = safegraphs_value_is_fixed<codec_type_>();
    safegraphs_column_t const& value = header.value;
    safegraphs_column_t const& adjacency = header.neighbors;
    constexpr dtype_t wanted_key_dtype = safegraphs_key_dtype<key_t>();
    if (header.key.offsets || !safegraphs_key_widens(header.key.dtype, wanted_key_dtype) ||
        header.key.shape != column_shape_t {scalar_shape_t {}})
        return status_t::config_mismatch_k;
    std::size_t const key_bytes = dtype_bytes(header.key.dtype);
    if (header.key.data.size() % key_bytes) return status_t::malformed_k;
    std::size_t const nodes = header.key.data.size() / key_bytes;
    // A narrower key column that miscounts the other columns' rows is another dtype relabelled.
    if (header.key.dtype != wanted_key_dtype && (header.value.nodes() != nodes || header.neighbors.nodes() != nodes))
        return status_t::config_mismatch_k;
    std::size_t const sums_bytes = (nodes + 1) * sizeof(u64_t);
    if (value.dtype != codec_type_::value_dtype_k || value.offsets.has_value() == fixed_width || value.shape != shape)
        return status_t::config_mismatch_k;
    std::size_t const degree = std::holds_alternative<vector_shape_t>(adjacency.shape) && !adjacency.offsets
                                   ? std::get<vector_shape_t>(adjacency.shape).columns
                                   : 0;
    if (adjacency.dtype != safegraphs_node_id_dtype<node_id_t>() ||
        (adjacency.offsets ? adjacency.shape != column_shape_t {scalar_shape_t {}} : !degree))
        return status_t::config_mismatch_k;
    if constexpr (fixed_width) {
        if (value.data.size() != nodes * *regular_width(shape) * element_bytes) return status_t::malformed_k;
    }
    else {
        if (value.offsets->dtype != dtype_t::u64_k) return status_t::config_mismatch_k;
        if (value.offsets->bytes.size() != sums_bytes || value.data.size() % element_bytes)
            return status_t::malformed_k;
    }
    if (degree) {
        if (adjacency.data.size() != nodes * degree * sizeof(node_id_t)) return status_t::malformed_k;
    }
    else {
        if (adjacency.offsets->dtype != dtype_t::u64_k) return status_t::config_mismatch_k;
        if (adjacency.offsets->bytes.size() != sums_bytes || adjacency.data.size() % sizeof(node_id_t))
            return status_t::malformed_k;
    }
    if (header.key.data.first % alignof(u64_t) || value.data.first % alignof(u64_t) ||
        adjacency.data.first % alignof(u64_t) ||
        (adjacency.offsets && adjacency.offsets->bytes.first % alignof(u64_t)) ||
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

    // Ragged neighbor sums are the file's own buffer; `plan_load` proves they start at zero, never
    // fall and end at the ID count. Fixed rows stage their IDs below the node count packed, every
    // sentinel skipped. Value sums count bytes here, so a fixed shape synthesizes them and a ragged
    // one of wider elements scales them up.
    growing_buffer<u64_t, allocator_type_> staged_neighbor_sums {allocator};
    growing_buffer<node_id_t, allocator_type_> staged_neighbor_ids {allocator};
    std::span<u64_t const> neighbor_sums;
    std::span<node_id_t const> neighbor_ids;
    if (degree) {
        std::span<node_id_t const> const rows = safegraphs_typed_span_<node_id_t const>(segment, adjacency.data.first,
                                                                                        nodes * degree);
        std::size_t kept = 0;
        for (node_id_t const neighbor : rows) kept += static_cast<std::size_t>(neighbor) < nodes;
        if (failed(staged_neighbor_sums.resize(nodes + 1)) || failed(staged_neighbor_ids.resize_uninitialized(kept)))
            return status_t::bad_alloc_k;
        kept = 0;
        for (std::size_t node = 0; node != nodes; ++node) {
            for (node_id_t const neighbor : rows.subspan(node * degree, degree))
                if (static_cast<std::size_t>(neighbor) < nodes) staged_neighbor_ids[kept++] = neighbor;
            staged_neighbor_sums[node + 1] = kept;
        }
        neighbor_sums = std::span<u64_t const> {staged_neighbor_sums};
        neighbor_ids = std::span<node_id_t const> {staged_neighbor_ids};
    }
    else {
        neighbor_sums = safegraphs_typed_span_<u64_t const>(segment, adjacency.offsets->bytes.first, nodes + 1);
        neighbor_ids = safegraphs_typed_span_<node_id_t const>(segment, adjacency.data.first,
                                                               adjacency.data.size() / sizeof(node_id_t));
    }
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

    // Keys of the index's own dtype are read in place; narrower ones widen into a staged copy.
    growing_buffer<key_t, allocator_type_> staged_keys {allocator};
    std::span<key_t const> keys;
    if (header.key.dtype == wanted_key_dtype)
        keys = safegraphs_typed_span_<key_t const>(segment, header.key.data.first, nodes);
    else {
        if (failed(staged_keys.resize(nodes))) return status_t::bad_alloc_k;
        byte_t const* const stored = segment.data() + header.key.data.first;
        for (std::size_t node = 0; node != nodes; ++node)
            staged_keys[node] = safegraphs_widened_key<key_t>(stored + node * key_bytes, header.key.dtype);
        keys = std::span<key_t const> {staged_keys};
    }
    columns_view_t const columns {
        keys, neighbor_sums, neighbor_ids, value_sums, segment.subspan(value.data.first, value.data.size()),
    };
    auto planned = index.plan_load(columns);
    if (failed(planned.status)) return planned.status;

    std::size_t const regions = max_of<std::size_t>(executor.size(), 1);
    std::size_t const per_region = divide_round_up(nodes, regions);
    using outcome_ref_t = fu::standard_atomic_ref<status_t>;
    status_t outcome = status_t::success_k;
    auto load_region = [&](std::size_t region, std::size_t, std::size_t) noexcept {
        if (failed(outcome_ref_t(outcome).load(std::memory_order_relaxed))) return;
        std::size_t const first = min_of(region * per_region, nodes);
        std::size_t const bound = min_of(first + per_region, nodes);
        status_t loaded = planned.value.load(
            iota_view<node_id_t> {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)}, codec);
        if (succeeded(loaded) && progress(region) == continuation_t::cancel_k) loaded = status_t::cancelled_k;
        status_t expected_status = status_t::success_k;
        if (failed(loaded))
            outcome_ref_t(outcome).compare_exchange_strong(expected_status, loaded, std::memory_order_relaxed,
                                                           std::memory_order_relaxed);
    };
    if (status_t const fanned = executor.for_tasks(regions, load_region); failed(fanned)) return fanned;
    if (status_t const settled = outcome_ref_t(outcome).load(std::memory_order_relaxed); failed(settled))
        return settled;
    return planned.value.finish();
}

/**
 *  Restores a columned index from the monolith at @p path, mapped read-only; every value is
 *  decoded into its record, so nothing borrows the mapping once this returns. A refusal of
 *  @p codec passes through.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k The path does not parse.
 *  - @c malformed_k The file is empty, the header does not parse, or a column's bytes, sums or
 *    extents disagree with its rows or the monolith.
 *  - @c config_mismatch_k A column's dtype, shape or key width differs from what @p index and
 *    @p codec take, the segment is misaligned, @p index already holds entries, or a step's nodes
 *    fall outside the plan.
 *  - @c overflow_k The node count passes the node IDs or the key multimap of @p index.
 *  - @c uninitialized_k On NVIDIA, no reserve froze the connectivity, or the pool of
 *    @p executor has no workers.
 *  - @c unsupported_k On NVIDIA, an entry past every rung of the face's kits.
 *  - @c contended_k A mutator, a batch, a load or a plan holds @p index, or the pool of
 *    @p executor runs another dispatch.
 */
template <typename index_type_, typename codec_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t, typename allocator_type_ = std::allocator<byte_t>>
    requires interchange_value_codec<codec_type_>
status_t safegraphs_read_index(std::string_view path, index_type_& index, codec_type_ const& codec,
                               executor_type_& executor, load_config_t const& config, progress_type_&& progress = {},
                               allocator_type_ allocator = {}) noexcept {
    auto [mapping, map_status] = mapped_file_t::read(path);
    if (failed(map_status)) return map_status;
    return safegraphs_read_index(mapping.view(), index, codec, executor, config, std::forward<progress_type_>(progress),
                                 allocator);
}

/** The columns of a monolith as an external-memory reader addresses them, beside what its header
 *  says of them that only the reading family can check. */
struct safegraphs_persisted_part_t {

    /** The key, value and neighbors columns, every offset from the monolith's own start. */
    persisted_columns_t columns;

    /** The header's metric token, whatever it names; empty when the header names none. */
    safegraphs_name_t metric;

    /** What one value row holds, for the family to check against its own. */
    column_shape_t value_shape;
};

/** The columns @p header lays out, their byte ranges moved from the data segment's start to the
 *  monolith's; a fixed-width column counts its logical elements per row, packed families
 *  included. */
inline persisted_columns_t safegraphs_header_columns(safegraphs_header_t const& header) noexcept {
    auto const persisted_column_of = [&header](safegraphs_column_t const& column) noexcept {
        persisted_column_t persisted_column;
        persisted_column.dtype = column.dtype;
        persisted_column.data = {header.data_offset + column.data.first, header.data_offset + column.data.bound};
        if (column.offsets)
            persisted_column.offsets = prefix_sums_t {
                column.offsets->dtype,
                {header.data_offset + column.offsets->bytes.first, header.data_offset + column.offsets->bytes.bound}};
        else {
            persisted_column.row_width = regular_width(column.shape).value_or(0);
            persisted_column.row_bytes = column.row_bytes();
        }
        return persisted_column;
    };
    persisted_columns_t persisted;
    persisted.nodes = header.nodes();
    persisted.keys = persisted_column_of(header.key);
    persisted.values = persisted_column_of(header.value);
    persisted.neighbors = persisted_column_of(header.neighbors);
    return persisted;
}

/**
 *  @brief The part whose monolith header @p header_bytes holds, as
 *      @ref unum::usearch::utilities::safegraphs_header_columns lays it out, with its metric token
 *      and value shape.
 *
 *  Refuses with @c config_mismatch_k a graph no reader walks: keys other than one fixed element
 *  per row, and neighbors other than ragged single IDs or a fixed degree. A header whose columns
 *  disagree on their rows, or whose prefix sums cover other than them, answers @c malformed_k. The
 *  value dtype and shape, the key and node-ID widths, and the metric are the caller's to check.
 *
 *  @par Statuses:
 *  - @c malformed_k The header does not parse.
 *  - @c config_mismatch_k The keys are not one fixed element per row, or the neighbors are
 *    neither ragged single IDs nor a fixed degree.
 *  - @c malformed_k The columns disagree on their rows, the keys have no width, or a prefix sum
 *    buffer covers other than the rows.
 */
template <typename allocator_type_ = std::allocator<byte_t>>
expected<safegraphs_persisted_part_t> safegraphs_persisted_part(bytes_view_t header_bytes,
                                                                allocator_type_ allocator = {}) noexcept {
    safegraphs_persisted_part_t part;
    auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") part.metric = safegraphs_name_t {field.value};
    };
    auto const [header, header_status] = safegraphs_read_header(header_bytes, allocator, per_custom_field);
    if (failed(header_status)) return {part, header_status};
    safegraphs_column_t const& adjacency = header.neighbors;
    bool const degree_fixed = std::holds_alternative<vector_shape_t>(adjacency.shape) && !adjacency.offsets &&
                              std::get<vector_shape_t>(adjacency.shape).columns;
    if (header.key.offsets || header.key.shape != column_shape_t {scalar_shape_t {}} ||
        (adjacency.offsets ? adjacency.shape != column_shape_t {scalar_shape_t {}} : !degree_fixed))
        return {part, status_t::config_mismatch_k};
    std::size_t const nodes = header.nodes();
    if (header.value.nodes() != nodes || adjacency.nodes() != nodes || !header.key.row_bytes())
        return {part, status_t::malformed_k};
    auto const sums_cover = [nodes](std::optional<prefix_sums_t> const& sums) noexcept {
        return !sums || sums->bytes.size() == (nodes + 1) * dtype_bytes(sums->dtype);
    };
    if (!sums_cover(header.value.offsets) || !sums_cover(adjacency.offsets)) return {part, status_t::malformed_k};
    part.columns = safegraphs_header_columns(header);
    part.value_shape = header.value.shape;
    return {part, status_t::success_k};
}

#pragma endregion Columned Indexes

} // namespace unum::usearch::utilities
