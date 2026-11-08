/**
 *  @file src/arrow.cpp
 *  @author Ash Vardanian
 *  @date November 6, 2026
 *  @brief Implements <tt>c/usearch/arrow.h</tt>: Arrow columns read into views and views exported as
 *      Arrow columns in place.
 */
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int32_t`, `std::int64_t`, `std::uint32_t`, `std::uint64_t`
#include <cstring> // `std::memcpy` - the counts inside a schema's metadata

#include <array>            // `std::array`
#include <bit>              // `std::endian`
#include <charconv>         // `std::from_chars`, `std::to_chars` - the widths a format spells
#include <initializer_list> // `std::initializer_list`
#include <limits>           // `std::numeric_limits`
#include <new>              // `std::nothrow`
#include <span>             // `std::span`
#include <string_view>      // `std::string_view`
#include <type_traits>      // `std::is_same_v`
#include <variant>          // `std::monostate`, `std::visit`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the count of live nested structures

#include "usearch/arrow.h" // `usearch_keys_view_from_arrow`, `usearch_tape_view_to_arrow`

#include "lib.hpp" // `arena_contains`, `copy_off_device`

#pragma region Scalars

/** Where an exported column of no rows points its buffers, as consumers may read the one offset of
 *  an empty tape, and what a converted tape of no bytes points its data at. */
inline constexpr std::uint64_t no_rows_k[1] {};

/** One key and its value in the metadata of a schema. */
struct metadata_pair_t {
    std::string_view key;
    std::string_view value;
};

/** Bytes the metadata of an exported column takes at most. */
constexpr std::size_t metadata_capacity_k = 96;

/** Encodes @p pairs as Arrow metadata: the count of pairs, then every key and value behind its
 *  length, each count a 32-bit integer in the native byte order. */
consteval std::array<char, metadata_capacity_k> encode_metadata(std::initializer_list<metadata_pair_t> pairs) noexcept {
    std::array<char, metadata_capacity_k> bytes {};
    std::size_t length = 0;
    auto const append_count = [&](std::size_t count) {
        for (std::size_t byte = 0; byte != sizeof(std::int32_t); ++byte) {
            std::size_t const shift = 8 * (std::endian::native == std::endian::little ? byte : 3 - byte);
            bytes[length++] = static_cast<char>((count >> shift) & 0xFF);
        }
    };
    auto const append_text = [&](std::string_view text) {
        append_count(text.size());
        for (char const character : text) bytes[length++] = character;
    };
    append_count(pairs.size());
    for (metadata_pair_t const& pair : pairs) append_text(pair.key), append_text(pair.value);
    return bytes;
}

/** The metadata tagging a column with the extension @p name. */
consteval std::array<char, metadata_capacity_k> extension_metadata(std::string_view name) noexcept {
    return encode_metadata({{"ARROW:extension:name", name}, {"ARROW:extension:metadata", ""}});
}

/** The name of the extension Arrow @p metadata tags a column with, or empty for none. */
static std::string_view extension_of(char const* metadata) noexcept {
    if (!metadata) return {};
    char const* cursor = metadata;
    auto const read_count = [&]() noexcept {
        std::int32_t count = 0;
        std::memcpy(&count, cursor, sizeof(count));
        cursor += sizeof(count);
        return static_cast<std::size_t>(count);
    };
    auto const read_text = [&]() noexcept {
        std::size_t const length = read_count();
        std::string_view const text {cursor, length};
        cursor += length;
        return text;
    };
    for (std::size_t pairs = read_count(); pairs; --pairs) {
        std::string_view const key = read_text();
        std::string_view const value = read_text();
        if (key == "ARROW:extension:name") return value;
    }
    return {};
}

/** How a scalar crosses into Arrow: the format of one element, the metadata tagging it where Arrow
 *  has no name for it, its bytes, and the columns one element packs. */
struct scalar_layout_t {
    usearch_scalar_t scalar = usearch_scalar_unknown_k;
    std::string_view format;
    std::array<char, metadata_capacity_k> metadata {};
    std::size_t element_bytes = 0;
    std::size_t columns_per_element = 1;

    /** The metadata a column of these elements carries, null where Arrow names them. */
    char const* tag() const noexcept { return extension_of(metadata.data()).empty() ? nullptr : metadata.data(); }
};

/** Every scalar Arrow carries, the untagged first so a format alone finds them; @c u1x8 is a column
 *  of its own, never an element. */
inline constexpr std::array<scalar_layout_t, 13> scalar_layouts_k {{
    {usearch_scalar_f64_k, "g", {}, 8, 1},
    {usearch_scalar_f32_k, "f", {}, 4, 1},
    {usearch_scalar_f16_k, "e", {}, 2, 1},
    {usearch_scalar_i8_k, "c", {}, 1, 1},
    {usearch_scalar_u8_k, "C", {}, 1, 1},
    {usearch_scalar_i32_k, "i", {}, 4, 1},
    {usearch_scalar_bf16_k, "S", extension_metadata("numkong.bf16"), 2, 1},
    {usearch_scalar_e5m2_k, "C", extension_metadata("numkong.e5m2"), 1, 1},
    {usearch_scalar_e4m3_k, "C", extension_metadata("numkong.e4m3"), 1, 1},
    {usearch_scalar_e3m2_k, "C", extension_metadata("numkong.e3m2"), 1, 1},
    {usearch_scalar_e2m3_k, "C", extension_metadata("numkong.e2m3"), 1, 1},
    {usearch_scalar_i4x2_k, "C", extension_metadata("numkong.i4x2"), 1, 2},
    {usearch_scalar_u4x2_k, "C", extension_metadata("numkong.u4x2"), 1, 2},
}};

/** The layout of @p scalar as an element, or null for one that crosses as none. */
static scalar_layout_t const* layout_of(usearch_scalar_t scalar) noexcept {
    for (scalar_layout_t const& layout : scalar_layouts_k)
        if (layout.scalar == scalar) return &layout;
    return nullptr;
}

/** The layout of elements of @p format tagged with @p extension, or null for none. Another
 *  project's tag leaves the storage readable, while an unknown one of ours does not. */
static scalar_layout_t const* layout_of(std::string_view format, std::string_view extension) noexcept {
    if (!extension.starts_with("numkong.")) extension = {};
    for (scalar_layout_t const& layout : scalar_layouts_k)
        if (layout.format == format && extension_of(layout.metadata.data()) == extension) return &layout;
    return nullptr;
}

#pragma endregion Scalars

#pragma region Converters

/** The format of a column, or empty for one missing or released. */
static std::string_view format_of(ArrowSchema const* schema, ArrowArray const* array) noexcept {
    if (!schema || !array || !schema->release || !array->release || !schema->format) return {};
    return schema->format;
}

/** The positive width @p format spells past @p prefix, or zero for none. */
static std::size_t width_of(std::string_view format, std::string_view prefix) noexcept {
    std::size_t width = 0;
    char const* const end = format.data() + format.size();
    std::from_chars_result const parsed = std::from_chars(format.data() + prefix.size(), end, width);
    return parsed.ec == std::errc {} && parsed.ptr == end ? width : 0;
}

/** Refuses a column of the right format that is dictionary-encoded, holds a null, or carries fewer
 *  than @p buffers buffers, reading only its lengths and pointers. */
static usearch_status_t check_column(ArrowSchema const& schema, ArrowArray const& array,
                                     std::int64_t buffers) noexcept {
    if (schema.dictionary || array.dictionary) return usearch_status_unsupported_layout_k;
    if (array.length < 0 || array.offset < 0 || array.n_buffers < buffers || !array.buffers)
        return usearch_status_malformed_k;
    // A count of -1 is unknown, so only a missing validity buffer rules nulls out
    if (array.null_count > 0 || (array.null_count < 0 && array.buffers[0])) return usearch_status_invalid_argument_k;
    if (static_cast<std::uint64_t>(array.length) > std::numeric_limits<std::size_t>::max())
        return usearch_status_overflow_k;
    return usearch_status_success_k;
}

/** The one child of a column, or nulls for one that does not have exactly one. */
struct child_column_t {
    ArrowSchema const* schema = nullptr;
    ArrowArray const* array = nullptr;
};

static child_column_t child_of(ArrowSchema const& schema, ArrowArray const& array) noexcept {
    if (schema.n_children != 1 || array.n_children != 1 || !schema.children || !array.children) return {};
    return {schema.children[0], array.children[0]};
}

/** Where the values of a fixed-size list of @p width elements start, in elements. */
static std::size_t first_value(ArrowArray const& list, ArrowArray const& values, std::size_t width) noexcept {
    return static_cast<std::size_t>(list.offset) * width + static_cast<std::size_t>(values.offset);
}

/** Whether @p values hold every row of a fixed-size list of @p width elements. */
static bool holds_rows(ArrowArray const& list, ArrowArray const& values, std::size_t width) noexcept {
    return static_cast<std::size_t>(values.length) >= static_cast<std::size_t>(list.offset + list.length) * width;
}

#pragma endregion Converters

#pragma region Exporters

/** The most nested structures one column exports: list, struct, then its key and distance. */
constexpr std::size_t nodes_capacity_k = 4;

/** The longest format an export spells, @c "+w:" and a width. */
constexpr std::size_t format_capacity_k = 4 + std::numeric_limits<std::size_t>::digits10 + 1;

/** One nested level of a column before it crosses: what its schema spells and what its array
 *  points at. */
struct column_node_t {
    std::string_view format;
    char const* name = "item";
    char const* metadata = nullptr;
    std::size_t length = 0;
    std::array<void const*, 3> buffers {};
    std::int64_t buffers_count = 1;

    /** Where the node's children sit among the column's nodes, and how many it has. */
    std::size_t first_child = 0;
    std::size_t children_count = 0;
};

/** The nested schemas of one exported column, shared by every level, which a consumer may move
 *  out and release in any order. */
struct exported_schema_t {
    std::size_t live = 0;
    std::array<ArrowSchema, nodes_capacity_k> schemas {};
    std::array<ArrowSchema*, nodes_capacity_k> children {};
    std::array<std::array<char, format_capacity_k>, nodes_capacity_k> formats {};
};

/** The nested arrays of one exported column, as @c exported_schema_t. */
struct exported_array_t {
    std::size_t live = 0;
    std::array<ArrowArray, nodes_capacity_k> arrays {};
    std::array<ArrowArray*, nodes_capacity_k> children {};
    std::array<std::array<void const*, 3>, nodes_capacity_k> buffers {};
};

/** Releases one nested schema or array and the children still in it, the last one standing freeing
 *  what they share. */
template <typename node_type_, typename shared_type_>
static void release_exported(node_type_* node) noexcept {
    auto* const shared = static_cast<shared_type_*>(node->private_data);
    for (std::int64_t child = 0; child != node->n_children; ++child)
        if (node->children[child]->release) node->children[child]->release(node->children[child]);
    node->release = nullptr;
    if (fu::standard_atomic_ref<std::size_t>(shared->live).fetch_sub(1, std::memory_order_acq_rel) == 1) delete shared;
}

/** The Arrow device an arena's outputs sit on, and the vendor's ordinal of it. */
struct arrow_device_t {
    ArrowDeviceType type = ARROW_DEVICE_CPU;
    std::int64_t ordinal = -1;
};

static arrow_device_t arrow_device_of(usearch_arena_handle_t const& arena) noexcept {
    return std::visit(
        []<typename arena_type_>(arena_type_ const& device) noexcept -> arrow_device_t {
            if constexpr (std::is_same_v<arena_type_, std::monostate>) return {};
            else if constexpr (std::is_same_v<arena_type_, cuda_device_arena_t>)
                return {ARROW_DEVICE_CUDA, device.device()};
            else if constexpr (std::is_same_v<arena_type_, hip_device_arena_t>)
                return {ARROW_DEVICE_ROCM, device.device()};
            else return {ARROW_DEVICE_METAL, device.device()};
        },
        arena.device_memory.arena);
}

/** Exports @p nodes of a column out of @p arena, the first being the column itself, into @p schema
 *  and @p array. Every call filling the arena drained its uploads, so its outputs are final and no
 *  event guards them. */
static usearch_status_t export_column(usearch_arena_handle_t const& arena, std::span<column_node_t const> nodes,
                                      ArrowSchema* schema, ArrowDeviceArray* array) noexcept {
    exported_schema_t* const schemas = new (std::nothrow) exported_schema_t {};
    exported_array_t* const arrays = schemas ? new (std::nothrow) exported_array_t {} : nullptr;
    if (!arrays) {
        delete schemas;
        return usearch_status_out_of_memory_k;
    }
    schemas->live = nodes.size();
    arrays->live = nodes.size();
    *array = {};
    for (std::size_t position = 0; position != nodes.size(); ++position) {
        column_node_t const& node = nodes[position];
        ArrowSchema& exported_schema = position ? schemas->schemas[position] : *schema;
        ArrowArray& exported_array = position ? arrays->arrays[position] : array->array;
        schemas->children[position] = &schemas->schemas[position];
        arrays->children[position] = &arrays->arrays[position];
        arrays->buffers[position] = node.buffers;
        node.format.copy(schemas->formats[position].data(), format_capacity_k - 1);
        exported_schema = {.format = schemas->formats[position].data(),
                           .name = position ? node.name : "",
                           .metadata = node.metadata,
                           .flags = 0,
                           .n_children = static_cast<std::int64_t>(node.children_count),
                           .children = node.children_count ? &schemas->children[node.first_child] : nullptr,
                           .dictionary = nullptr,
                           .release = &release_exported<ArrowSchema, exported_schema_t>,
                           .private_data = schemas};
        exported_array = {.length = static_cast<std::int64_t>(node.length),
                          .null_count = 0,
                          .offset = 0,
                          .n_buffers = node.buffers_count,
                          .n_children = static_cast<std::int64_t>(node.children_count),
                          .buffers = arrays->buffers[position].data(),
                          .children = node.children_count ? &arrays->children[node.first_child] : nullptr,
                          .dictionary = nullptr,
                          .release = &release_exported<ArrowArray, exported_array_t>,
                          .private_data = arrays};
    }
    arrow_device_t const device = arrow_device_of(arena);
    array->device_id = device.ordinal;
    array->device_type = device.type;
    return usearch_status_success_k;
}

/** Whether the @p bytes at @p data lie in memory @p arena handed out; an empty run always does. */
static bool arena_holds(usearch_arena_handle_t const& arena, void const* data, std::size_t bytes) noexcept {
    if (!bytes) return true;
    return data && arena_contains(arena, data) && arena_contains(arena, static_cast<byte_t const*>(data) + (bytes - 1));
}

/** The offset at @p offset of @p arena, which crosses from a device through its driver; Apple's
 *  unified memory reads in place. */
template <typename offset_type_>
static expected<std::size_t> read_offset(usearch_arena_handle_t const& arena, offset_type_ const* offset) noexcept {
    ArrowDeviceType const device = arrow_device_of(arena).type;
    if (device != ARROW_DEVICE_CUDA && device != ARROW_DEVICE_ROCM)
        return {static_cast<std::size_t>(*offset), status_t::success_k};
    offset_type_ copied = 0;
    status_t const status = copy_off_device(std::as_writable_bytes(std::span {&copied, 1}),
                                            std::as_bytes(std::span {offset, 1}));
    return {static_cast<std::size_t>(copied), status};
}

/** The last of the @p count + 1 packed offsets at @p offsets inside @p arena, the child length of a
 *  list column. */
template <typename offset_type_>
static expected<std::size_t> last_offset(usearch_arena_handle_t const& arena, offset_type_ const* offsets,
                                         std::size_t count) noexcept {
    if (!count) return {0, status_t::success_k};
    if (!arena_holds(arena, offsets, (count + 1) * sizeof(offset_type_))) return {0, status_t::invalid_argument_k};
    return read_offset(arena, offsets + count);
}

/** A list of @p count rows over the @c std::size_t @p offsets into the node after it. */
static column_node_t list_node(std::size_t count, void const* offsets) noexcept {
    column_node_t list;
    list.format = sizeof(std::size_t) == sizeof(std::uint64_t) ? "+L" : "+l";
    list.length = count;
    list.buffers = {nullptr, count ? offsets : no_rows_k, nullptr};
    list.buffers_count = 2;
    list.first_child = 1;
    list.children_count = 1;
    return list;
}

/** A column of @p length packed keys at @p keys named @p name. */
static column_node_t keys_node(std::size_t length, void const* keys, char const* name) noexcept {
    column_node_t column;
    column.format = "w:16";
    column.name = name;
    static constexpr std::array<char, metadata_capacity_k> uuid_metadata = extension_metadata("arrow.uuid");
    column.metadata = uuid_metadata.data();
    column.length = length;
    column.buffers = {nullptr, length ? keys : no_rows_k, nullptr};
    column.buffers_count = 2;
    return column;
}

/** The @c struct{key, distance} over @p length matches, then its two columns, as the column's nodes
 *  from its second on. */
static void matches_nodes(std::size_t length, usearch_key_t const* keys, usearch_distance_t const* scores,
                          std::span<column_node_t> nodes) noexcept {
    nodes[0].format = "+s";
    nodes[0].length = length;
    nodes[0].first_child = 2;
    nodes[0].children_count = 2;
    nodes[1] = keys_node(length, keys, "key");
    nodes[2].format = "g";
    nodes[2].name = "distance";
    nodes[2].length = length;
    nodes[2].buffers = {nullptr, length ? scores : static_cast<void const*>(no_rows_k), nullptr};
    nodes[2].buffers_count = 2;
}

#pragma endregion Exporters

USEARCH_EXPORT usearch_status_t usearch_keys_view_from_arrow(ArrowSchema const* schema, ArrowArray const* array,
                                                             usearch_keys_view_t* keys, size_t* count) {
    std::string_view const format = format_of(schema, array);
    if (!keys || !count || format.empty()) return usearch_status_invalid_argument_k;
    if (format != "w:16") return usearch_status_unsupported_layout_k;
    if (usearch_status_t const checked = check_column(*schema, *array, 2); checked != usearch_status_success_k)
        return checked;
    auto const* const data = static_cast<usearch_key_t const*>(array->buffers[1]);
    if (!data && array->length) return usearch_status_malformed_k;
    *keys = {data ? data + array->offset : nullptr, sizeof(usearch_key_t)};
    *count = static_cast<size_t>(array->length);
    return usearch_status_success_k;
}

USEARCH_EXPORT usearch_status_t usearch_rows_view_from_arrow(ArrowSchema const* schema, ArrowArray const* array,
                                                             usearch_rows_view_t* rows, size_t* count) {
    std::string_view const format = format_of(schema, array);
    if (!rows || !count || format.empty()) return usearch_status_invalid_argument_k;
    if (format.starts_with("w:") && extension_of(schema->metadata) == "numkong.u1x8") {
        std::size_t const row_bytes = width_of(format, "w:");
        if (!row_bytes) return usearch_status_malformed_k;
        if (usearch_status_t const checked = check_column(*schema, *array, 2); checked != usearch_status_success_k)
            return checked;
        auto const* const data = static_cast<byte_t const*>(array->buffers[1]);
        if (!data && array->length) return usearch_status_malformed_k;
        *rows = {data ? data + array->offset * row_bytes : nullptr, row_bytes, row_bytes * 8, usearch_scalar_u1x8_k};
        *count = static_cast<size_t>(array->length);
        return usearch_status_success_k;
    }
    if (!format.starts_with("+w:")) return usearch_status_unsupported_layout_k;
    std::size_t const width = width_of(format, "+w:");
    if (!width) return usearch_status_malformed_k;
    if (usearch_status_t const checked = check_column(*schema, *array, 1); checked != usearch_status_success_k)
        return checked;
    child_column_t const values = child_of(*schema, *array);
    if (!values.schema || !values.array || !values.schema->format) return usearch_status_malformed_k;
    scalar_layout_t const* const layout = layout_of(values.schema->format, extension_of(values.schema->metadata));
    if (!layout) return usearch_status_unsupported_layout_k;
    if (usearch_status_t const checked = check_column(*values.schema, *values.array, 2);
        checked != usearch_status_success_k)
        return checked;
    if (!holds_rows(*array, *values.array, width)) return usearch_status_malformed_k;
    auto const* const data = static_cast<byte_t const*>(values.array->buffers[1]);
    if (!data && array->length) return usearch_status_malformed_k;
    std::size_t const first = first_value(*array, *values.array, width);
    *rows = {data ? data + first * layout->element_bytes : nullptr, width * layout->element_bytes,
             width * layout->columns_per_element, layout->scalar};
    *count = static_cast<size_t>(array->length);
    return usearch_status_success_k;
}

USEARCH_EXPORT usearch_status_t usearch_tape_view_from_arrow(ArrowSchema const* schema, ArrowArray const* array,
                                                             usearch_tape_view_t* tape, size_t* count) {
    std::string_view const format = format_of(schema, array);
    if (!tape || !count || format.empty()) return usearch_status_invalid_argument_k;
    bool const bytes = format == "z" || format == "u" || format == "Z" || format == "U";
    if (!bytes && format != "+l" && format != "+L") return usearch_status_unsupported_layout_k;
    if (usearch_status_t const checked = check_column(*schema, *array, bytes ? 3 : 2);
        checked != usearch_status_success_k)
        return checked;
    void const* const offsets = array->buffers[1];
    if (!offsets) return usearch_status_malformed_k;

    usearch_tape_view_t viewed {};
    if (format == "Z" || format == "U" || format == "+L") {
        viewed.layout = usearch_tape_u64_k;
        viewed.offsets.u64 = static_cast<std::uint64_t const*>(offsets) + array->offset;
    }
    else viewed.offsets.u32 = static_cast<std::uint32_t const*>(offsets) + array->offset;

    if (bytes) {
        viewed.data = array->buffers[2] ? array->buffers[2] : no_rows_k;
        viewed.scalar = usearch_scalar_u8_k;
    }
    else {
        child_column_t const points = child_of(*schema, *array);
        if (!points.schema || !points.array || !points.schema->format) return usearch_status_malformed_k;
        if (std::string_view {points.schema->format} != "+w:3") return usearch_status_unsupported_layout_k;
        if (usearch_status_t const checked = check_column(*points.schema, *points.array, 1);
            checked != usearch_status_success_k)
            return checked;
        child_column_t const values = child_of(*points.schema, *points.array);
        if (!values.schema || !values.array || !values.schema->format) return usearch_status_malformed_k;
        scalar_layout_t const* const layout = layout_of(values.schema->format, extension_of(values.schema->metadata));
        if (!layout || layout->columns_per_element != 1) return usearch_status_unsupported_layout_k;
        if (usearch_status_t const checked = check_column(*values.schema, *values.array, 2);
            checked != usearch_status_success_k)
            return checked;
        if (!holds_rows(*points.array, *values.array, 3)) return usearch_status_malformed_k;
        auto const* const data = static_cast<byte_t const*>(values.array->buffers[1]);
        if (!data && array->length) return usearch_status_malformed_k;
        viewed.data = data ? data + first_value(*points.array, *values.array, 3) * layout->element_bytes
                           : reinterpret_cast<byte_t const*>(no_rows_k);
        viewed.scalar = layout->scalar;
    }
    *tape = viewed;
    *count = static_cast<size_t>(array->length);
    return usearch_status_success_k;
}

USEARCH_EXPORT usearch_status_t usearch_keys_view_to_arrow(usearch_keys_view_t keys, size_t count,
                                                           usearch_arena_t arena, ArrowSchema* schema,
                                                           ArrowDeviceArray* array) {
    if (!arena || !schema || !array || (count && !keys.data)) return usearch_status_invalid_argument_k;
    if (count > 1 && keys.stride != sizeof(usearch_key_t)) return usearch_status_unsupported_layout_k;
    if (!arena_holds(*arena, keys.data, count * sizeof(usearch_key_t))) return usearch_status_invalid_argument_k;
    std::array const nodes {keys_node(count, keys.data, nullptr)};
    return export_column(*arena, nodes, schema, array);
}

USEARCH_EXPORT usearch_status_t usearch_rows_view_to_arrow(usearch_rows_view_t rows, size_t count,
                                                           usearch_arena_t arena, ArrowSchema* schema,
                                                           ArrowDeviceArray* array) {
    if (!arena || !schema || !array || (count && !rows.data)) return usearch_status_invalid_argument_k;
    bool const bits = rows.scalar == usearch_scalar_u1x8_k;
    scalar_layout_t const* const layout = layout_of(rows.scalar);
    if (!rows.columns || (!bits && !layout)) return usearch_status_unsupported_layout_k;
    std::size_t const columns_per_element = bits ? 8 : layout->columns_per_element;
    if (rows.columns % columns_per_element) return usearch_status_unsupported_layout_k;
    std::size_t const width = rows.columns / columns_per_element;
    std::size_t const row_bytes = bits ? width : width * layout->element_bytes;
    if (count > 1 && rows.stride != row_bytes) return usearch_status_unsupported_layout_k;
    if (!arena_holds(*arena, rows.data, count * row_bytes)) return usearch_status_invalid_argument_k;

    void const* const data = count ? rows.data : no_rows_k;
    std::array<char, format_capacity_k> format {};
    std::string_view const prefix = bits ? "w:" : "+w:";
    prefix.copy(format.data(), prefix.size());
    std::to_chars_result const spelled = std::to_chars(format.data() + prefix.size(), format.data() + format.size(),
                                                       width);
    std::string_view const spelling {format.data(), static_cast<std::size_t>(spelled.ptr - format.data())};

    std::array<column_node_t, 2> nodes {};
    nodes[0].format = spelling;
    nodes[0].length = count;
    if (bits) {
        static constexpr std::array<char, metadata_capacity_k> u1x8_metadata = extension_metadata("numkong.u1x8");
        nodes[0].metadata = u1x8_metadata.data();
        nodes[0].buffers = {nullptr, data, nullptr};
        nodes[0].buffers_count = 2;
        return export_column(*arena, std::span {nodes}.first(1), schema, array);
    }
    nodes[0].first_child = 1;
    nodes[0].children_count = 1;
    nodes[1].format = layout->format;
    nodes[1].metadata = layout->tag();
    nodes[1].length = count * width;
    nodes[1].buffers = {nullptr, data, nullptr};
    nodes[1].buffers_count = 2;
    return export_column(*arena, nodes, schema, array);
}

USEARCH_EXPORT usearch_status_t usearch_tape_view_to_arrow(usearch_tape_view_t tape, size_t count,
                                                           usearch_arena_t arena, ArrowSchema* schema,
                                                           ArrowDeviceArray* array) {
    if (!arena || !schema || !array) return usearch_status_invalid_argument_k;
    bool const wide = tape.layout == usearch_tape_u64_k;
    if (!wide && tape.layout != usearch_tape_u32_k) return usearch_status_unsupported_layout_k;
    void const* const offsets = wide ? static_cast<void const*>(tape.offsets.u64) : tape.offsets.u32;
    if (count && !offsets) return usearch_status_invalid_argument_k;
    scalar_layout_t const* const layout = layout_of(tape.scalar);
    if (!layout || layout->columns_per_element != 1) return usearch_status_unsupported_layout_k;

    bool const bytes = tape.scalar == usearch_scalar_u8_k;
    std::size_t const element_bytes = bytes ? 1 : 3 * layout->element_bytes;
    auto const [elements, status] = wide ? last_offset(*arena, tape.offsets.u64, count)
                                         : last_offset(*arena, tape.offsets.u32, count);
    if (failed(status)) return static_cast<usearch_status_t>(status);
    if (!arena_holds(*arena, tape.data, elements * element_bytes)) return usearch_status_invalid_argument_k;

    void const* const data = tape.data ? tape.data : no_rows_k;
    if (bytes) {
        column_node_t column;
        column.format = wide ? "Z" : "z";
        column.length = count;
        column.buffers = {nullptr, count ? offsets : no_rows_k, data};
        column.buffers_count = 3;
        std::array const nodes {column};
        return export_column(*arena, nodes, schema, array);
    }
    std::array<column_node_t, 3> nodes {};
    nodes[0] = list_node(count, offsets);
    nodes[0].format = wide ? "+L" : "+l";
    nodes[1].format = "+w:3";
    nodes[1].length = elements;
    nodes[1].first_child = 2;
    nodes[1].children_count = 1;
    nodes[2].format = layout->format;
    nodes[2].metadata = layout->tag();
    nodes[2].length = elements * 3;
    nodes[2].buffers = {nullptr, data, nullptr};
    nodes[2].buffers_count = 2;
    return export_column(*arena, nodes, schema, array);
}

USEARCH_EXPORT usearch_status_t usearch_keys_tape_view_to_arrow(usearch_keys_tape_view_t keys, size_t count,
                                                                usearch_arena_t arena, ArrowSchema* schema,
                                                                ArrowDeviceArray* array) {
    if (!arena || !schema || !array || (count && !keys.offsets)) return usearch_status_invalid_argument_k;
    auto const [elements, status] = last_offset(*arena, keys.offsets, count);
    if (failed(status)) return static_cast<usearch_status_t>(status);
    if (!arena_holds(*arena, keys.keys, elements * sizeof(usearch_key_t))) return usearch_status_invalid_argument_k;
    std::array const nodes {list_node(count, keys.offsets), keys_node(elements, keys.keys, "item")};
    return export_column(*arena, nodes, schema, array);
}

USEARCH_EXPORT usearch_status_t usearch_matches_tape_view_to_arrow(usearch_matches_tape_view_t matches, size_t count,
                                                                   usearch_arena_t arena, ArrowSchema* schema,
                                                                   ArrowDeviceArray* array) {
    if (!arena || !schema || !array || (count && !matches.offsets)) return usearch_status_invalid_argument_k;
    auto const [elements, status] = last_offset(*arena, matches.offsets, count);
    if (failed(status)) return static_cast<usearch_status_t>(status);
    if (!arena_holds(*arena, matches.keys, elements * sizeof(usearch_key_t)) ||
        !arena_holds(*arena, matches.scores, elements * sizeof(usearch_distance_t)))
        return usearch_status_invalid_argument_k;
    std::array<column_node_t, 4> nodes {};
    nodes[0] = list_node(count, matches.offsets);
    matches_nodes(elements, matches.keys, matches.scores, std::span {nodes}.subspan(1));
    return export_column(*arena, nodes, schema, array);
}
