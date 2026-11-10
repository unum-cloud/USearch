/**
 *  @file tools/arrow_checkpoints.hpp
 *  @author Ash Vardanian
 *  @date November 1, 2026
 *  @brief SafeGraphs checkpoints as Apache Arrow IPC files: an export writes one record batch per
 *      part straight from its column bytes, and an import lays every batch out as a monolith again.
 *
 *  Every column of a part becomes one field, named as the part's header names it, so a dense part
 *  exports its @c key, @c value, @c neighbors and @c norms columns, its bytes as the buffers:
 *
 *  @verbatim
 *  SafeGraphs column                 Arrow field
 *  regular, one Arrow-typed element  that type: uint8 to uint64, int8 to int64, float16 to float64
 *  regular, a uuid                   fixed_size_binary(16) as the arrow.uuid extension
 *  regular, several elements         fixed_size_list(width) of the element
 *  regular, a packed dtype           fixed_size_binary(row bytes), like u1x8 or i4x2
 *  ragged, utf8                      large_utf8, the prefix sums as its offsets
 *  ragged, any other element         large_list of the element, the prefix sums as its offsets
 *  @endverbatim
 *
 *  An element Arrow has no type for, like bf16 or e4m3, travels as fixed-size binary of its bytes.
 *  Each field's metadata names the column's own dtype and shape under @c safegraphs:dtype and
 *  @c safegraphs:shape, so an import restores what Arrow cannot spell, and reads a field lacking
 *  them from its Arrow type alone. The schema's metadata carries the first part's string metadata,
 *  the metric among them, since every part carries the same schema.
 *
 *  An import validates every batch against the schema, then lays each one out as a monolith whose
 *  columns hold the batch's buffers, behind a manifest when there are several, in the one buffer
 *  every @c load_*_buffer verb takes. Parquet, and the Arrow C stream interface through which a
 *  reader the CLI loads would hand over its batches, are the next readers to add.
 *
 *  Matches travel apart from checkpoints, one row per query: its @c query key, then @c matches, a
 *  large list of structs of @c key and @c distance, nearest first. Keys are the UUID extension and
 *  distances are @c float64. Vectors a get retrieves travel alike, one row per entry: its @c key,
 *  then its @c vector, typed as the library exports rows.
 *
 *  @ref arrow_input_t reads an Arrow file or stream of any producer from a path or a pipe, in
 *  exactly the bytes each message names, and hands every batch's columns over as C Data, so they
 *  reach the library through the same converters any other caller uses.
 */
#pragma once

#include <cstdio>  // `std::remove`
#include <cstring> // `std::memcpy`

#include <array>       // `std::array`
#include <bit>         // `std::bit_cast` - distances off the wire
#include <charconv>    // `std::from_chars`
#include <filesystem>  // `std::filesystem::is_regular_file` - what an input maps
#include <limits>      // `std::numeric_limits`
#include <optional>    // `std::optional`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <utility>     // `std::pair`
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include "usearch/arrow.h"                         // `usearch_rows_view_from_arrow`, `usearch_keys_view_from_arrow`
#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t`, `read_native_file`, `write_native_file`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_catalog`, `safegraphs_write_header`

#include "abi.hpp"       // `copied_matches_t`, `match_offsets`
#include "arrow_ipc.hpp" // `ipc_write_batch`, `ipc_read_footer`, `ipc_validate_batch`

namespace unum::usearch::tools {
using namespace unum::usearch::utilities;

#pragma region Columns

/** The field metadata key naming the SafeGraphs dtype of a column, as its header spells it. */
inline constexpr std::string_view arrow_dtype_key_k = "safegraphs:dtype";

/** The field metadata key naming a column's shape: @c [] for a scalar, @c [128] for a vector, and
 *  @c [0,3] for a matrix of ragged rows. */
inline constexpr std::string_view arrow_shape_key_k = "safegraphs:shape";

/** The name every list and fixed-size list gives its child, as Arrow's own writers do. */
inline constexpr std::string_view arrow_child_name_k = "item";

/** The metadata of the canonical UUID extension, which a reader matches by name. */
inline constexpr ipc_key_value_t arrow_uuid_metadata_k[2] {{ipc_extension_name_key_k, ipc_extension_uuid_k},
                                                           {"ARROW:extension:metadata", ""}};

/** What an export or an import moved, for the line the CLI prints. */
struct arrow_summary_t {

    /** Record batches, one per part. */
    std::size_t parts = 0;

    /** Rows across every batch, one per node. */
    std::size_t nodes = 0;

    /** Bytes of the Arrow file written, or of the checkpoint laid out. */
    std::size_t bytes = 0;

    /** The metric the checkpoint records, empty when it records none. */
    std::string metric;
};

/** @p shape as the @c safegraphs:shape metadata spells it. */
inline std::string arrow_shape_text(column_shape_t shape) {
    if (auto const* matrix = std::get_if<matrix_shape_t>(&shape))
        return fmt::format("[{},{}]", matrix->rows, matrix->columns);
    if (auto const* vector = std::get_if<vector_shape_t>(&shape)) return fmt::format("[{}]", vector->columns);
    return "[]";
}

/** The shape @p text spells as @ref arrow_shape_text does, or nothing. */
inline std::optional<column_shape_t> arrow_shape_of(std::string_view text) {
    if (text.size() < 2 || text.front() != '[' || text.back() != ']') return std::nullopt;
    text = text.substr(1, text.size() - 2);
    if (text.empty()) return scalar_shape_t {};
    std::array<std::size_t, 2> axes {};
    std::size_t count = 0;
    for (char const* cursor = text.data(); count != axes.size(); ++cursor) {
        std::from_chars_result const parsed = std::from_chars(cursor, text.data() + text.size(), axes[count++]);
        if (parsed.ec != std::errc {}) return std::nullopt;
        cursor = parsed.ptr;
        if (cursor == text.data() + text.size()) break;
        if (*cursor != ',') return std::nullopt;
    }
    if (count == 1) return vector_shape_t {axes[0]};
    return matrix_shape_t {axes[0], axes[1]};
}

/** The value of @p key among @p pairs, empty when absent. */
inline std::string_view arrow_metadata_value(std::span<ipc_key_value_t const> pairs, std::string_view key) noexcept {
    for (ipc_key_value_t const& pair : pairs)
        if (pair.key == key) return pair.value;
    return {};
}

/** Types @p element as one element of @p dtype: an integer or a float where Arrow has one, the
 *  UUID extension, or fixed-size binary of its bytes; false for a packed dtype, whose elements
 *  share a byte. */
inline bool arrow_element_of(dtype_t dtype, ipc_field_t& element) noexcept {
    std::int32_t const bits = static_cast<std::int32_t>(dtype_bits(dtype));
    switch (dtype) {
    case dtype_t::f64_k:
    case dtype_t::f32_k:
    case dtype_t::f16_k: element.type = ipc_type_t::floating_point_k, element.width = bits; return true;
    case dtype_t::i64_k:
    case dtype_t::i32_k:
    case dtype_t::i16_k:
    case dtype_t::i8_k:
        element.type = ipc_type_t::int_k, element.width = bits, element.signedness = ipc_signedness_t::signed_k;
        return true;
    case dtype_t::u64_k:
    case dtype_t::u32_k:
    case dtype_t::u16_k:
    case dtype_t::u8_k: element.type = ipc_type_t::int_k, element.width = bits; return true;
    case dtype_t::uuid_k: element.metadata = arrow_uuid_metadata_k; break;
    default: break;
    }
    if (!dtype_bytes(dtype)) return false;
    element.type = ipc_type_t::fixed_size_binary_k, element.width = static_cast<std::int32_t>(dtype_bytes(dtype));
    return true;
}

/** The dtype a foreign producer's @p element holds: an integer or a float by its width, a UUID by
 *  its extension, UTF-8 text, or bytes; unknown otherwise. */
inline dtype_t arrow_dtype_of(ipc_field_t const& element) noexcept {
    bool const signed_integer = element.signedness == ipc_signedness_t::signed_k;
    switch (element.type) {
    case ipc_type_t::floating_point_k:
        return element.width == 64 ? dtype_t::f64_k : element.width == 32 ? dtype_t::f32_k : dtype_t::f16_k;
    case ipc_type_t::int_k:
        switch (element.width) {
        case 64: return signed_integer ? dtype_t::i64_k : dtype_t::u64_k;
        case 32: return signed_integer ? dtype_t::i32_k : dtype_t::u32_k;
        case 16: return signed_integer ? dtype_t::i16_k : dtype_t::u16_k;
        default: return signed_integer ? dtype_t::i8_k : dtype_t::u8_k;
        }
    case ipc_type_t::fixed_size_binary_k:
        return arrow_metadata_value(element.metadata, ipc_extension_name_key_k) == ipc_extension_uuid_k
                   ? dtype_t::uuid_k
                   : dtype_t::unknown_k;
    case ipc_type_t::utf8_k:
    case ipc_type_t::large_utf8_k: return dtype_t::utf8_k;
    case ipc_type_t::binary_k:
    case ipc_type_t::large_binary_k: return dtype_t::u8_k;
    default: return dtype_t::unknown_k;
    }
}

/** One column of a part as an Arrow field, holding the child and the metadata the field views, so
 *  it stays in place while the field is used. */
struct arrow_field_t {

    /** The text @c safegraphs:shape names. */
    std::string shape;

    /** The field's own pairs: the UUID extension's where the field is a UUID, then the dtype and
     *  the shape. */
    std::array<ipc_key_value_t, 4> metadata {};

    /** The element a list or a fixed-size list holds. */
    ipc_field_t child;

    /** The field the schema lists. */
    ipc_field_t field;
};

/**
 *  @brief Lays out @p column, named @p name, as a field in @p laid.
 *
 *  @param[in] name The column's name, which the field views.
 *  @param[in] column The column of the first part, whose dtype and shape every part shares.
 *  @param[out] laid The field, its child and its metadata.
 *  @return False for a column Arrow has no layout for: a ragged packed dtype, or narrow sums.
 */
inline bool arrow_field_of(std::string_view name, safegraphs_column_t const& column, arrow_field_t& laid) {
    ipc_field_t element {.name = arrow_child_name_k};
    bool const typed = arrow_element_of(column.dtype, element);
    std::size_t pairs = 0;
    for (ipc_key_value_t const& pair : element.metadata) laid.metadata[pairs++] = pair;
    laid.shape = arrow_shape_text(column.shape);
    laid.metadata[pairs++] = {arrow_dtype_key_k, dtype_name(column.dtype)};
    laid.metadata[pairs++] = {arrow_shape_key_k, laid.shape};
    std::span<ipc_key_value_t const> const metadata {laid.metadata.data(), pairs};
    laid.field = {.name = name, .metadata = metadata.last(2)};

    if (column.offsets) {
        dtype_t const sums = column.offsets->dtype;
        if (!typed || (sums != dtype_t::u64_k && sums != dtype_t::u32_k)) return false;
        bool const wide = sums == dtype_t::u64_k;
        if (column.dtype == dtype_t::utf8_k) laid.field.type = wide ? ipc_type_t::large_utf8_k : ipc_type_t::utf8_k;
        else {
            laid.child = element;
            laid.field.type = wide ? ipc_type_t::large_list_k : ipc_type_t::list_k;
            laid.field.children = {&laid.child, 1};
        }
        return true;
    }
    std::size_t const width = regular_width(column.shape).value_or(0);
    if (!width) return false;
    if (!typed) {
        laid.field.type = ipc_type_t::fixed_size_binary_k;
        laid.field.width = static_cast<std::int32_t>(column.row_bytes());
    }
    else if (width == 1) {
        laid.field.type = element.type, laid.field.width = element.width, laid.field.signedness = element.signedness;
        laid.field.metadata = metadata;
    }
    else {
        laid.child = element;
        laid.field.type = ipc_type_t::fixed_size_list_k;
        laid.field.width = static_cast<std::int32_t>(width);
        laid.field.children = {&laid.child, 1};
    }
    return true;
}

/** Whether two parts lay a column out alike, so one schema describes both. */
inline bool arrow_same_layout(safegraphs_column_t const& first, safegraphs_column_t const& second) noexcept {
    if (first.dtype != second.dtype || first.shape != second.shape || bool(first.offsets) != bool(second.offsets))
        return false;
    return !first.offsets || first.offsets->dtype == second.offsets->dtype;
}

#pragma endregion Columns

#pragma region Export

/** The nodes and buffers of one record batch, viewing the bytes of one part. */
struct arrow_batch_t {
    std::array<ipc_node_t, ipc_fields_max_k> nodes {};
    std::array<bytes_view_t, ipc_buffers_max_k> buffers {};
    std::size_t nodes_count = 0;
    std::size_t buffers_count = 0;

    /** Appends @p column of @p monolith, laid out as @p laid, holding @p rows rows. */
    void append(arrow_field_t const& laid, safegraphs_header_t const& header, bytes_view_t monolith,
                safegraphs_column_t const& column, std::size_t rows) noexcept {
        bytes_view_t const values = header.data_of(monolith, column);
        std::int64_t const length = static_cast<std::int64_t>(rows);
        nodes[nodes_count++] = {length, 0};
        // No validity bitmap, since a column holds no nulls.
        buffers[buffers_count++] = {};
        switch (laid.field.type) {
        case ipc_type_t::utf8_k:
        case ipc_type_t::large_utf8_k:
            buffers[buffers_count++] = header.offsets_of(monolith, column);
            buffers[buffers_count++] = values;
            return;
        case ipc_type_t::list_k:
        case ipc_type_t::large_list_k:
            buffers[buffers_count++] = header.offsets_of(monolith, column);
            nodes[nodes_count++] = {static_cast<std::int64_t>(values.size() / dtype_bytes(column.dtype)), 0};
            buffers[buffers_count++] = {};
            buffers[buffers_count++] = values;
            return;
        case ipc_type_t::fixed_size_list_k:
            nodes[nodes_count++] = {length * laid.field.width, 0};
            buffers[buffers_count++] = {};
            buffers[buffers_count++] = values;
            return;
        default: buffers[buffers_count++] = values; return;
        }
    }

    ipc_batch_t view(std::size_t rows) const noexcept {
        return {static_cast<std::int64_t>(rows), {nodes.data(), nodes_count}, {buffers.data(), buffers_count}};
    }
};

/** How a writer frames its messages: a file a reader seeks through by its footer, or a stream a
 *  pipe reads in order. */
enum class arrow_framing_t { file_k, stream_k };

/** Writes @p schema and @p batches to @p file from @p position on, counting the bytes into it;
 *  false once a write failed. */
inline bool arrow_write_messages(native_file_t file, arrow_framing_t framing, ipc_schema_t const& schema,
                                 std::span<ipc_batch_t const> batches, std::size_t& position) {
    static constexpr std::array<byte_t, ipc_alignment_default_k> zeros_k {};
    ipc_options_t const options {};
    native_file_event_t const event;
    bool written = true;
    auto write = [&](bytes_view_t bytes) {
        written = written && write_native_file(file, event, bytes, position) == static_cast<i64_t>(bytes.size());
        position += bytes.size();
    };
    // Each writer measures on an empty output first, answering the bytes it needs.
    std::vector<byte_t> metadata;
    auto measured = [&](std::size_t needed) { return metadata.resize(needed), bytes_span_t {metadata}; };
    ipc_status_t status = ipc_status_t::success_k;
    std::size_t needed = 0, metadata_written = 0;

    bool const footed = framing == arrow_framing_t::file_k;
    if (footed) write(std::as_bytes(std::span {ipc_file_magic_k}));
    status = ipc_write_schema(schema, options, position, {}, needed);
    if (status == ipc_status_t::too_small_k)
        status = ipc_write_schema(schema, options, position, measured(needed), metadata_written);
    if (status == ipc_status_t::success_k) write(std::span {metadata}.first(metadata_written));

    std::vector<ipc_block_t> blocks;
    std::array<ipc_gather_t, ipc_buffers_max_k> gathers;
    for (ipc_batch_t const& batch : batches) {
        if (status != ipc_status_t::success_k) break;
        std::uint64_t body_bytes = 0;
        status = ipc_write_batch(schema, batch, options, position, {}, needed, gathers, body_bytes);
        if (status == ipc_status_t::too_small_k)
            status = ipc_write_batch(schema, batch, options, position, measured(needed), metadata_written, gathers,
                                     body_bytes);
        if (status != ipc_status_t::success_k) break;
        blocks.push_back({static_cast<std::int64_t>(position), static_cast<std::int32_t>(metadata_written),
                          static_cast<std::int64_t>(body_bytes)});
        write(std::span {metadata}.first(metadata_written));
        for (std::size_t buffer = 0; buffer != batch.buffers.size(); ++buffer)
            write(gathers[buffer].bytes), write(std::span {zeros_k}.first(gathers[buffer].padding));
    }

    std::array<byte_t, 8> end_of_stream {};
    if (status == ipc_status_t::success_k) status = ipc_write_end_of_stream(end_of_stream, metadata_written);
    if (status == ipc_status_t::success_k) write(end_of_stream);
    if (!footed) return status == ipc_status_t::success_k && written;
    if (status == ipc_status_t::success_k) status = ipc_write_footer(schema, blocks, {}, needed);
    if (status == ipc_status_t::too_small_k)
        status = ipc_write_footer(schema, blocks, measured(needed), metadata_written);
    if (status == ipc_status_t::success_k) write(std::span {metadata}.first(metadata_written));
    return status == ipc_status_t::success_k && written;
}

/**
 *  @brief Exports the checkpoint at @p checkpoint, a monolith or a manifest at any address the
 *      kernel reaches, as the Arrow IPC file @p arrow_path, one record batch per part.
 *
 *  Every buffer is written from the part's mapping as it lies, so the export does no work per row.
 *
 *  @param[in] checkpoint The monolith or the manifest.
 *  @param[in] arrow_path The file to create, replacing any.
 *  @param[out] summary What the file holds.
 *  @return Why the export stopped, leaving no file, or nothing once the file is complete.
 */
inline std::string arrow_export(std::string_view checkpoint, std::string const& arrow_path, arrow_summary_t& summary) {
    using allocator_t = std::allocator<byte_t>;
    growing_buffer<mapped_file_t, allocator_t> mappings {allocator_t {}};
    growing_buffer<safegraphs_catalog_part_t, allocator_t> parts {allocator_t {}};
    growing_buffer<safegraphs_named_column_t, allocator_t> extra_columns {allocator_t {}};
    if (status_t const listed = safegraphs_catalog(checkpoint, mappings, parts, extra_columns, allocator_t {});
        failed(listed))
        return fmt::format("{} holds no SafeGraphs checkpoint: {}", checkpoint, status_to_string(listed));

    // Every part carries the whole schema, so the first one's header and string metadata name it.
    safegraphs_header_t const& first = parts[0].header;
    std::vector<std::pair<std::string, std::string>> custom_fields;
    auto keep_field = [&](safegraphs_custom_field_t field) { custom_fields.emplace_back(field.key, field.value); };
    if (status_t const read = safegraphs_read_header(parts[0].monolith, allocator_t {}, keep_field).status;
        failed(read))
        return fmt::format("{} has a header that does not parse: {}", checkpoint, status_to_string(read));
    std::vector<ipc_key_value_t> schema_metadata;
    for (auto const& [key, value] : custom_fields) schema_metadata.push_back({key, value});

    std::vector<std::string_view> names {"key", "value", "neighbors"};
    std::vector<safegraphs_column_t const*> columns {&first.key, &first.value, &first.neighbors};
    for (safegraphs_named_column_t const& extra : first.extra_columns)
        names.push_back(extra.name.view()), columns.push_back(&extra.column);
    std::vector<arrow_field_t> laid(columns.size());
    std::vector<ipc_field_t> fields;
    for (std::size_t column = 0; column != columns.size(); ++column) {
        if (!arrow_field_of(names[column], *columns[column], laid[column]))
            return fmt::format("{} holds a {} column of {} Arrow has no layout for", checkpoint, names[column],
                               dtype_name(columns[column]->dtype));
        fields.push_back(laid[column].field);
    }
    ipc_schema_t const schema {fields, schema_metadata};
    if (ipc_check_schema(schema) != ipc_status_t::success_k)
        return fmt::format("{} holds {} columns, more than an Arrow schema here takes", checkpoint, columns.size());

    // A part whose columns or rows disagree with the schema would make a batch no reader accepts.
    for (safegraphs_catalog_part_t const& part : parts) {
        safegraphs_header_t const& header = part.header;
        bool alike = header.extra_columns.size() == first.extra_columns.size() &&
                     arrow_same_layout(header.key, first.key) && arrow_same_layout(header.value, first.value) &&
                     arrow_same_layout(header.neighbors, first.neighbors);
        for (std::size_t extra = 0; alike && extra != first.extra_columns.size(); ++extra)
            alike = header.extra_columns[extra].name.view() == first.extra_columns[extra].name.view() &&
                    arrow_same_layout(header.extra_columns[extra].column, first.extra_columns[extra].column);
        alike = alike && header.value.nodes() == part.nodes && header.neighbors.nodes() == part.nodes;
        for (safegraphs_named_column_t const& extra : header.extra_columns)
            alike = alike && extra.column.nodes() == part.nodes;
        if (!alike) return fmt::format("{} holds parts whose columns differ", checkpoint);
    }

    // Every batch views its part's mapping, so laying them out here copies nothing.
    std::vector<arrow_batch_t> batches(parts.size());
    std::vector<ipc_batch_t> views;
    for (std::size_t part = 0; part != parts.size(); ++part) {
        safegraphs_catalog_part_t const& source = parts[part];
        safegraphs_header_t const& header = source.header;
        arrow_batch_t& batch = batches[part];
        batch.append(laid[0], header, source.monolith, header.key, source.nodes);
        batch.append(laid[1], header, source.monolith, header.value, source.nodes);
        batch.append(laid[2], header, source.monolith, header.neighbors, source.nodes);
        for (std::size_t extra = 0; extra != header.extra_columns.size(); ++extra)
            batch.append(laid[3 + extra], header, source.monolith, header.extra_columns[extra].column, source.nodes);
        views.push_back(batch.view(source.nodes));
        summary.nodes += source.nodes;
    }

    native_file_t const file = open_native_file(arrow_path.c_str(), file_access_t::truncate_read_write_k,
                                                file_buffering_t::buffered_k);
    if (file == invalid_native_file_k) return fmt::format("cannot create {}", arrow_path);
    std::size_t position = 0;
    bool const framed = arrow_write_messages(file, arrow_framing_t::file_k, schema, views, position);
    if (failed(close_native_file(file)) || !framed) {
        std::remove(arrow_path.c_str());
        return fmt::format("writing {} failed", arrow_path);
    }
    summary.parts = parts.size(), summary.bytes = position, summary.metric = parts[0].metric.view();
    return {};
}

/**
 *  @brief Writes @p schema and its one @p batch to @p path.
 *
 *  @param[in] path Standard output for @c -, or the file to create, replacing any. Both take the
 *      stream framing for @c - or an @c .arrows path, and a file's otherwise.
 *  @return Why the write stopped, leaving no file, or nothing once it is complete.
 */
inline std::string arrow_write_batch(std::string_view path, ipc_schema_t const& schema, ipc_batch_t const& batch) {
    bool const piped = path == "-";
    std::string const arrow_path {path};
    native_file_t const file = piped ? standard_output_native_file()
                                     : open_native_file(arrow_path.c_str(), file_access_t::truncate_read_write_k,
                                                        file_buffering_t::buffered_k);
    if (file == invalid_native_file_k) return fmt::format("cannot create {}", path);
    arrow_framing_t const framing = piped || path.ends_with(".arrows") ? arrow_framing_t::stream_k
                                                                       : arrow_framing_t::file_k;
    std::size_t position = 0;
    bool const framed = arrow_write_messages(file, framing, schema, {&batch, 1}, position);
    if ((!piped && failed(close_native_file(file))) || !framed) {
        if (!piped) std::remove(arrow_path.c_str());
        return fmt::format("writing {} failed", path);
    }
    return {};
}

/** The field every column of keys takes, named @p name. */
inline ipc_field_t arrow_keys_field(std::string_view name) noexcept {
    return {.name = name,
            .type = ipc_type_t::fixed_size_binary_k,
            .width = sizeof(usearch_key_t),
            .metadata = arrow_uuid_metadata_k};
}

/**
 *  @brief Writes the matches of a search to @p path, one row per query.
 *
 *  @param[in] path As @ref arrow_write_batch takes it.
 *  @param[in] queries The key of every query.
 *  @param[in] matches What each query found, nearest first.
 *  @return Why the write stopped, leaving no file, or nothing once it is complete.
 */
inline std::string arrow_matches_write(std::string_view path, std::span<usearch_key_t const> queries,
                                       copied_matches_t const& matches) {
    std::array<ipc_field_t, 2> const match_fields {{
        arrow_keys_field("key"),
        {.name = "distance", .type = ipc_type_t::floating_point_k, .width = 64},
    }};
    ipc_field_t const match {.name = arrow_child_name_k, .type = ipc_type_t::struct_k, .children = match_fields};
    std::array<ipc_field_t, 2> const fields {{
        arrow_keys_field("query"),
        {.name = "matches", .type = ipc_type_t::large_list_k, .children = {&match, 1}},
    }};
    std::vector<std::int64_t> offsets(queries.size() + 1);
    for (std::size_t query = 0; query != queries.size(); ++query)
        offsets[query + 1] = offsets[query] + static_cast<std::int64_t>(matches.counts[query]);

    std::int64_t const rows = static_cast<std::int64_t>(queries.size()), entries = offsets.back();
    std::array<ipc_node_t, 5> const nodes {{{rows, 0}, {rows, 0}, {entries, 0}, {entries, 0}, {entries, 0}}};
    // Depth-first, each validity buffer empty, since no column holds nulls.
    std::array<bytes_view_t, 9> const buffers {
        bytes_view_t {},
        std::as_bytes(queries),
        bytes_view_t {},
        std::as_bytes(std::span {offsets}),
        bytes_view_t {},
        bytes_view_t {},
        std::as_bytes(std::span {matches.keys}),
        bytes_view_t {},
        std::as_bytes(std::span {matches.scores}),
    };
    return arrow_write_batch(path, {fields, {}}, {rows, nodes, buffers});
}

/**
 *  @brief Writes the vectors a get retrieved to @p path, one row per entry.
 *
 *  @param[in] path As @ref arrow_write_batch takes it.
 *  @param[in] keys The key of every row.
 *  @param[in] vector The schema of one rows column, as the library exports it, which names the
 *      column's type and tags and which stays alive through the call.
 *  @param[in] rows The packed rows.
 *  @return Why the write stopped, leaving no file, or nothing once it is complete.
 */
inline std::string arrow_vectors_write(std::string_view path, std::span<usearch_key_t const> keys, ArrowSchema& vector,
                                       bytes_view_t rows) {
    std::array<ipc_field_t, ipc_fields_max_k> fields_pool {};
    std::array<ipc_key_value_t, 4 * ipc_fields_max_k> pairs_pool {};
    ipc_schema_pools_t pools {fields_pool, pairs_pool};
    ArrowSchema* const exported = &vector;
    std::span<ipc_field_t const> imported;
    if (ipc_import_schemas(&exported, 1, pools, imported) != ipc_status_t::success_k)
        return fmt::format("the rows export as a type {} takes no field for", path);
    ipc_field_t named = imported[0];
    named.name = "vector";
    std::array<ipc_field_t, 2> const fields {arrow_keys_field("key"), named};

    bool const nested = !named.children.empty();
    std::int64_t const length = static_cast<std::int64_t>(keys.size());
    std::int64_t const elements = nested ? length * named.width : length;
    std::array<ipc_node_t, 3> const nodes {{{length, 0}, {length, 0}, {elements, 0}}};
    std::array<bytes_view_t, 5> const buffers {
        bytes_view_t {}, std::as_bytes(keys), bytes_view_t {}, nested ? bytes_view_t {} : rows, rows,
    };
    return arrow_write_batch(
        path, {fields, {}},
        {length, std::span {nodes}.first(nested ? 3 : 2), std::span {buffers}.first(nested ? 5 : 4)});
}

#pragma endregion Export

#pragma region Input

/** What names no column of a schema. */
inline constexpr std::size_t arrow_no_column_k = (std::numeric_limits<std::size_t>::max)();

/** Reads standard input to its end into @p bytes, as a checkpoint piped in arrives; false once a
 *  read failed. */
inline bool read_standard_input(std::vector<byte_t>& bytes) {
    native_file_event_t const event;
    std::size_t length = 0;
    bytes.resize(std::size_t {1} << 20);
    while (true) {
        if (length == bytes.size()) bytes.resize(bytes.size() * 2);
        i64_t const landed = read_native_file(standard_input_native_file(), event, bytes_span_t {bytes}.subspan(length),
                                              length);
        if (landed < 0) return false;
        if (!landed) break;
        length += static_cast<std::size_t>(landed);
    }
    bytes.resize(length);
    return true;
}

/**
 *  @brief An Arrow IPC file or stream, sniffed by its first bytes and walked batch by batch.
 *
 *  A file is its magic, then a stream, then a footer, so one walk over the messages reads both, and
 *  an input missing its end refuses instead of reading as a shorter one. A regular file is mapped,
 *  and standard input or anything else is read in exactly the byte counts each message names, so
 *  pipes work, the first bytes the sniff took handed out to the first reads. Every batch views the
 *  mapping or the body last read, until the next call.
 */
class arrow_input_t {
    std::string name_;
    mapped_file_t mapping_;
    bytes_view_t remaining_;
    native_file_t descriptor_ = invalid_native_file_k;
    native_file_t owned_descriptor_ = invalid_native_file_k;
    native_file_event_t event_;
    u64_t position_ = 0;
    std::array<byte_t, 8> sniffed_ {};
    std::size_t sniffed_size_ = 0;
    std::size_t sniffed_taken_ = 0;

    std::vector<byte_t> schema_metadata_, metadata_, body_;
    std::array<ipc_field_t, ipc_fields_max_k> fields_ {};
    std::array<ipc_key_value_t, 4 * ipc_fields_max_k> pairs_ {};
    std::array<ipc_node_t, ipc_fields_max_k> nodes_ {};
    std::array<bytes_view_t, ipc_buffers_max_k> buffers_ {};
    ipc_schema_t schema_;
    ipc_batch_t batch_;

    std::vector<char> text_;
    ipc_arrow_storage_t storage_ {};
    ArrowSchema columns_schema_ {};
    ArrowArray columns_array_ {};

    /** Fills @p bytes off the input, answering how many landed, fewer only at its end. */
    std::size_t read_(bytes_span_t bytes) noexcept {
        if (descriptor_ == invalid_native_file_k) {
            std::size_t const landed = min_of(bytes.size(), remaining_.size());
            if (landed) std::memcpy(bytes.data(), remaining_.data(), landed);
            remaining_ = remaining_.subspan(landed);
            return landed;
        }
        std::size_t filled = 0;
        for (; sniffed_taken_ != sniffed_size_ && filled != bytes.size(); ++filled)
            bytes[filled] = sniffed_[sniffed_taken_++];
        while (filled != bytes.size()) {
            i64_t const landed = read_native_file(descriptor_, event_, bytes.subspan(filled), position_);
            if (landed <= 0) break;
            filled += static_cast<std::size_t>(landed);
            position_ += static_cast<u64_t>(landed);
        }
        return filled;
    }

    /** The next @p count bytes, viewing the mapping or read into @p storage, fewer only at the
     *  end. */
    bytes_view_t take_(std::size_t count, std::vector<byte_t>& storage) {
        if (descriptor_ == invalid_native_file_k) {
            bytes_view_t const taken = remaining_.first(min_of(count, remaining_.size()));
            remaining_ = remaining_.subspan(taken.size());
            return taken;
        }
        storage.resize(count);
        return bytes_view_t {storage}.first(read_(storage));
    }

    /** Why the input stopped at @p status. */
    std::string refused_(ipc_status_t status) const {
        switch (status) {
        case ipc_status_t::truncated_k: return fmt::format("{} ends inside a message or before its end", name_);
        case ipc_status_t::unsupported_k:
            return fmt::format("{} is compressed, dictionary-encoded or big-endian, or holds a null or a column " //
                               "type this reader takes none of",                                                  //
                               name_);
        case ipc_status_t::too_small_k: return fmt::format("{} holds more than {} fields", name_, ipc_fields_max_k);
        default: return fmt::format("{} is malformed Arrow IPC", name_);
        }
    }

    /** Reads the next message into @p message, its metadata kept in @p metadata and its body in
     *  @p body; @p ended turns true at the end of the stream. */
    std::string read_message_(std::vector<byte_t>& metadata, ipc_message_t& message, bytes_view_t& body, bool& ended) {
        std::array<byte_t, 8> prefix {};
        bytes_span_t const whole {prefix};
        std::uint32_t metadata_bytes = 0;
        std::size_t prefix_bytes = 0;
        if (read_(whole.first(4)) != 4) return refused_(ipc_status_t::truncated_k);
        ipc_status_t status = ipc_read_prefix(whole.first(4), metadata_bytes, prefix_bytes);
        if (status == ipc_status_t::truncated_k) {
            if (read_(whole.subspan(4)) != 4) return refused_(status);
            status = ipc_read_prefix(whole, metadata_bytes, prefix_bytes);
        }
        if (status != ipc_status_t::success_k) return refused_(status);
        ended = !metadata_bytes;
        if (ended) return {};
        bytes_view_t const framed = take_(metadata_bytes, metadata);
        if (framed.size() != metadata_bytes) return refused_(ipc_status_t::truncated_k);
        if (status = ipc_read_message(framed, message); status != ipc_status_t::success_k) return refused_(status);
        std::size_t const body_bytes = static_cast<std::size_t>(message.body_bytes);
        body = take_(body_bytes, body_);
        return body.size() == body_bytes ? std::string {} : refused_(ipc_status_t::truncated_k);
    }

    /** Sniffs the first bytes, then reads the schema that leads every Arrow input. */
    std::string start_() {
        bytes_view_t head = remaining_.first(min_of(sniffed_.size(), remaining_.size()));
        if (descriptor_ != invalid_native_file_k) {
            sniffed_size_ = read_(sniffed_);
            head = bytes_view_t {sniffed_}.first(sniffed_size_);
        }
        auto const opens_with = [&](std::string_view magic) {
            return head.size() >= magic.size() && std::memcmp(head.data(), magic.data(), magic.size()) == 0;
        };
        if (opens_with("PAR1"))
            return fmt::format("{} is Parquet, which this tool has no reader for; rewrite it as Arrow IPC", name_);
        if (opens_with(ipc_file_magic_k.substr(0, 6))) {
            std::array<byte_t, ipc_file_magic_k.size()> magic {};
            if (read_(magic) != magic.size()) return refused_(ipc_status_t::truncated_k);
        }
        else if (!opens_with("\xFF\xFF\xFF\xFF"))
            return fmt::format("{} is neither an Arrow file nor an Arrow stream", name_);

        ipc_message_t message;
        bytes_view_t body;
        bool ended = false;
        if (std::string failure = read_message_(schema_metadata_, message, body, ended); !failure.empty())
            return failure;
        if (ended || message.kind != ipc_message_kind_t::schema_k)
            return fmt::format("{} does not start with a schema", name_);
        if (ipc_status_t const status = ipc_read_schema(message, fields_, pairs_, schema_);
            status != ipc_status_t::success_k)
            return refused_(status);
        text_.resize(ipc_arrow_metadata_bytes(schema_.metadata) + ipc_arrow_text_bytes(schema_.fields) + 1);
        storage_.text = text_;
        ipc_status_t const exported = ipc_schema_to_arrow(schema_, storage_, columns_schema_);
        return exported == ipc_status_t::success_k ? std::string {} : refused_(exported);
    }

  public:
    arrow_input_t() = default;
    arrow_input_t(arrow_input_t const&) = delete;
    arrow_input_t& operator=(arrow_input_t const&) = delete;
    ~arrow_input_t() noexcept {
        if (owned_descriptor_ != invalid_native_file_k) [[maybe_unused]]
            status_t const closed = close_native_file(owned_descriptor_);
    }

    /** Opens @p path, standard input for @c -, and reads its schema; answers why it cannot. */
    std::string open(std::string_view path) {
        name_ = path == "-" ? std::string {"standard input"} : std::string {path};
        if (path == "-") {
            descriptor_ = standard_input_native_file();
            return start_();
        }
        std::error_code unknown;
        if (std::filesystem::is_regular_file(name_, unknown)) {
            auto [mapping, status] = mapped_file_t::read(path);
            if (failed(status)) return fmt::format("cannot read {}: {}", name_, status_to_string(status));
            mapping_ = std::move(mapping);
            remaining_ = mapping_.view();
            return start_();
        }
        owned_descriptor_ = open_native_file(name_.c_str(), file_access_t::read_only_k, file_buffering_t::buffered_k);
        if (owned_descriptor_ == invalid_native_file_k) return fmt::format("cannot read {}", name_);
        descriptor_ = owned_descriptor_;
        return start_();
    }

    /** Reads the next record batch, @p more turning false at the end of the input; answers why it
     *  stopped. */
    std::string next(bool& more) {
        more = false;
        ipc_message_t message;
        bytes_view_t body;
        bool ended = false;
        if (std::string failure = read_message_(metadata_, message, body, ended); !failure.empty()) return failure;
        if (ended) return {};
        if (message.kind != ipc_message_kind_t::record_batch_k)
            return fmt::format("{} repeats its schema mid-stream", name_);
        ipc_status_t status = ipc_read_batch(message, body, schema_, nodes_, buffers_, batch_);
        if (status == ipc_status_t::success_k) status = ipc_validate_batch(schema_, batch_);
        if (status == ipc_status_t::success_k) status = ipc_batch_to_arrow(schema_, batch_, storage_, columns_array_);
        more = status == ipc_status_t::success_k;
        return more ? std::string {} : refused_(status);
    }

    std::string const& name() const noexcept { return name_; }
    ipc_schema_t const& schema() const noexcept { return schema_; }

    /** The batch @ref next read. */
    ipc_batch_t const& batch() const noexcept { return batch_; }

    /** Rows of the batch @ref next read. */
    std::size_t rows() const noexcept { return static_cast<std::size_t>(batch_.length); }

    /** The schema of column @p column. */
    ArrowSchema const* column_schema(std::size_t column) const noexcept { return columns_schema_.children[column]; }

    /** Column @p column of the batch @ref next read, viewing its buffers until the next call. */
    ArrowArray const* column_array(std::size_t column) const noexcept { return columns_array_.children[column]; }

    /** Where the column @p name sits, or @ref arrow_no_column_k; the first of several alike. */
    std::size_t column_named(std::string_view name) const noexcept {
        for (std::size_t column = 0; column != schema_.fields.size(); ++column)
            if (schema_.fields[column].name == name) return column;
        return arrow_no_column_k;
    }
};

/** Why the column @p name of @p input cannot be read, for a converter's refusal @p status. */
inline std::string arrow_column_refused(arrow_input_t const& input, std::string_view name, usearch_status_t status) {
    switch (status) {
    case usearch_status_invalid_argument_k: return fmt::format("{} holds a null in {}", input.name(), name);
    case usearch_status_unsupported_layout_k:
        return fmt::format("{} holds {} as a type no converter reads", input.name(), name);
    default:
        return fmt::format("{} holds {} that a converter refuses with {}", input.name(), name,
                           usearch_status_name(status));
    }
}

/** Finds the column @p name of @p input into @p column, or answers why it cannot. */
inline std::string arrow_pick_column(arrow_input_t const& input, std::string_view name, std::size_t& column) {
    column = input.column_named(name);
    return column == arrow_no_column_k ? fmt::format("{} holds no column named {}", input.name(), name)
                                       : std::string {};
}

/** Finds the column of rows @p name names, or else the only one of @p input, into @p column, or
 *  answers why it cannot. */
inline std::string arrow_pick_rows(arrow_input_t const& input, std::string_view name, std::size_t& column) {
    if (!name.empty()) return arrow_pick_column(input, name, column);
    std::size_t found = 0;
    std::span<ipc_field_t const> const fields = input.schema().fields;
    for (std::size_t position = 0; position != fields.size(); ++position) {
        ipc_field_t const& field = fields[position];
        bool const packed = field.type == ipc_type_t::fixed_size_binary_k &&
                            arrow_metadata_value(field.metadata, ipc_extension_name_key_k) == "numkong.u1x8";
        if (field.type != ipc_type_t::fixed_size_list_k && !packed) continue;
        column = position, ++found;
    }
    if (found == 1) return {};
    return fmt::format("{} holds {} fixed-size columns of rows; name one with --vectors-column", input.name(), found);
}

/** Views the rows of column @p column, named @p name, of the batch @p input last read, through the
 *  converter every caller uses. */
inline std::string arrow_rows_of(arrow_input_t const& input, std::size_t column, usearch_rows_view_t& rows) {
    std::size_t count = 0;
    usearch_status_t const status = usearch_rows_view_from_arrow(input.column_schema(column),
                                                                 input.column_array(column), &rows, &count);
    if (status == usearch_status_success_k) return {};
    return arrow_column_refused(input, input.schema().fields[column].name, status);
}

/**
 *  @brief Views the keys of column @p column of the batch @p input last read: 16-byte keys in place
 *      through the converter, and 64-bit integers widened as row numbers are, into @p widened.
 *
 *  @return Why it cannot, a negative integer naming no key.
 */
inline std::string arrow_keys_of(arrow_input_t const& input, std::size_t column, std::vector<usearch_key_t>& widened,
                                 usearch_keys_view_t& keys) {
    ipc_field_t const& field = input.schema().fields[column];
    if (field.type != ipc_type_t::int_k) {
        std::size_t count = 0;
        usearch_status_t const status = usearch_keys_view_from_arrow(input.column_schema(column),
                                                                     input.column_array(column), &keys, &count);
        return status == usearch_status_success_k ? std::string {} : arrow_column_refused(input, field.name, status);
    }
    if (field.width != 64)
        return fmt::format("{} holds {} as {}-bit integers, not 64-bit", input.name(), field.name, field.width);
    ArrowArray const& integers = *input.column_array(column);
    std::size_t const rows = static_cast<std::size_t>(integers.length);
    if (rows && !integers.buffers[1]) return arrow_column_refused(input, field.name, usearch_status_malformed_k);
    auto const* const bytes = static_cast<byte_t const*>(integers.buffers[1]);
    widened.resize(rows);
    for (std::size_t row = 0; row != rows; ++row) {
        std::uint64_t const integer = ipc_load_little<std::uint64_t>(bytes + row * sizeof(std::uint64_t));
        if (field.signedness == ipc_signedness_t::signed_k && static_cast<std::int64_t>(integer) < 0)
            return fmt::format("{} holds a negative key in {}", input.name(), field.name);
        widened[row] = std::bit_cast<usearch_key_t>(uuid_of(integer));
    }
    keys = {widened.data(), sizeof(usearch_key_t)};
    return {};
}

#pragma endregion Input

#pragma region Import

/** One top-level field of an imported schema as the column it restores, and where its buffers
 *  start in every batch. */
struct arrow_column_t {
    ipc_field_t const* field = nullptr;
    dtype_t dtype = dtype_t::unknown_k;
    column_shape_t shape;

    /** The prefix sums' dtype, @c u32_k or @c u64_k; unknown for a regular column. */
    dtype_t sums_dtype = dtype_t::unknown_k;

    /** Bytes of one row of a regular column, or of one element of a ragged one. */
    std::size_t element_bytes = 0;

    std::size_t first_buffer = 0;

    /** Where the batch's values start among the field's buffers: past the validity bitmap, the
     *  offsets, and the child's own validity bitmap, whichever the type has. */
    std::size_t values_buffer() const noexcept {
        switch (field->type) {
        case ipc_type_t::utf8_k:
        case ipc_type_t::large_utf8_k:
        case ipc_type_t::binary_k:
        case ipc_type_t::large_binary_k:
        case ipc_type_t::fixed_size_list_k: return first_buffer + 2;
        case ipc_type_t::list_k:
        case ipc_type_t::large_list_k: return first_buffer + 3;
        default: return first_buffer + 1;
        }
    }
};

/** Reads @p field as the column it restores into @p column, its metadata first and its Arrow type
 *  second; false where the two disagree or no SafeGraphs column has the layout. */
inline bool arrow_column_of(ipc_field_t const& field, arrow_column_t& column) {
    if (field.children.size() > 1 || (field.children.size() == 1 && !field.children[0].children.empty())) return false;
    ipc_field_t const& element = field.children.empty() ? field : field.children[0];
    column.field = &field;
    column.dtype = arrow_dtype_of(element);
    column.shape = scalar_shape_t {};
    switch (field.type) {
    case ipc_type_t::utf8_k:
    case ipc_type_t::binary_k:
    case ipc_type_t::list_k: column.sums_dtype = dtype_t::u32_k; break;
    case ipc_type_t::large_utf8_k:
    case ipc_type_t::large_binary_k:
    case ipc_type_t::large_list_k: column.sums_dtype = dtype_t::u64_k; break;
    case ipc_type_t::fixed_size_list_k: column.shape = vector_shape_t {static_cast<std::size_t>(field.width)}; break;
    case ipc_type_t::struct_k: return false;
    default: break;
    }
    if (std::string_view const named = arrow_metadata_value(field.metadata, arrow_dtype_key_k); !named.empty())
        column.dtype = dtype_named(named);
    if (std::string_view const spelled = arrow_metadata_value(field.metadata, arrow_shape_key_k); !spelled.empty()) {
        std::optional<column_shape_t> const shape = arrow_shape_of(spelled);
        if (!shape) return false;
        column.shape = *shape;
    }
    if (column.dtype == dtype_t::unknown_k) return false;

    // The Arrow bytes of a row, or of an element, hold the column to the dtype and shape it names.
    std::size_t const arrow_element_bytes = ipc_type_is_fixed(element.type) ? ipc_value_bytes(element) : 1;
    if (column.sums_dtype != dtype_t::unknown_k) {
        column.element_bytes = dtype_bytes(column.dtype);
        return column.element_bytes && column.element_bytes == arrow_element_bytes;
    }
    if (!ipc_type_is_fixed(element.type)) return false;
    std::size_t const arrow_row_bytes = arrow_element_bytes * (field.children.empty() ? 1 : field.width);
    column.element_bytes = safegraphs_column_t {.dtype = column.dtype, .shape = column.shape}.row_bytes();
    return column.element_bytes && column.element_bytes == arrow_row_bytes;
}

/** One record batch read from the file, and the monolith it lays out as. */
struct arrow_part_t {
    std::array<ipc_node_t, ipc_fields_max_k> nodes {};
    std::array<bytes_view_t, ipc_buffers_max_k> buffers {};
    ipc_batch_t batch;

    /** Every column the batch restores, in the schema's order. */
    std::vector<safegraphs_column_t> columns;
    std::vector<safegraphs_named_column_t> extra_columns;
    safegraphs_header_t header;

    /** Bytes of the monolith, and where it starts in the checkpoint. */
    std::size_t bytes = 0;
    std::size_t offset = 0;
};

/**
 *  @brief Lays the Arrow IPC file at @p arrow_path out as a SafeGraphs checkpoint in @p checkpoint:
 *      one monolith per record batch, behind a manifest leading the buffer when there are several.
 *
 *  Every batch is validated against the schema before a byte is copied, then each column takes one
 *  copy of the buffers it restores, so an import also does no work per row.
 *
 *  @param[in] arrow_path The file to read.
 *  @param[in] metric The metric the caller names, which the file must record too, or which a file
 *      recording none takes; an empty one takes whatever the file records.
 *  @param[out] checkpoint The checkpoint's bytes, which every @c load_*_buffer verb takes.
 *  @param[out] summary What the checkpoint holds.
 *  @return Why the import stopped, or nothing.
 */
inline std::string arrow_import(std::string const& arrow_path, std::string_view metric, std::string& checkpoint,
                                arrow_summary_t& summary) {
    auto [mapping, mapping_status] = mapped_file_t::read(arrow_path);
    if (failed(mapping_status)) return fmt::format("cannot read {}: {}", arrow_path, status_to_string(mapping_status));
    bytes_view_t const file = mapping.view();

    std::array<ipc_field_t, ipc_fields_max_k> fields_pool {};
    std::array<ipc_key_value_t, 4 * ipc_fields_max_k> pairs_pool {};
    ipc_schema_t schema;
    std::vector<ipc_block_t> blocks;
    std::size_t blocks_count = 0;
    ipc_status_t status = ipc_read_footer(file, fields_pool, pairs_pool, schema, blocks, blocks_count);
    if (status == ipc_status_t::too_small_k) {
        blocks.resize(blocks_count);
        status = ipc_read_footer(file, fields_pool, pairs_pool, schema, blocks, blocks_count);
    }
    if (status != ipc_status_t::success_k) return fmt::format("{} is no Arrow IPC file this reader takes", arrow_path);
    if (!blocks_count) return fmt::format("{} holds no record batch", arrow_path);

    // The three graph columns are found by name, and every other column is a producer's.
    std::vector<arrow_column_t> columns(schema.fields.size());
    std::array<std::size_t, 3> graph_columns {columns.size(), columns.size(), columns.size()};
    std::size_t first_buffer = 0;
    for (std::size_t index = 0; index != schema.fields.size(); ++index) {
        ipc_field_t const& field = schema.fields[index];
        if (!arrow_column_of(field, columns[index]))
            return fmt::format("{} holds a {} column no SafeGraphs column lays out like", arrow_path, field.name);
        if (field.name.size() >= safegraphs_name_capacity_k)
            return fmt::format("{} names a column {}, longer than a SafeGraphs name", arrow_path, field.name);
        columns[index].first_buffer = first_buffer;
        first_buffer += ipc_buffers_of(field.type) + ipc_buffers_of(field.children);
        if (field.name == "key") graph_columns[0] = index;
        if (field.name == "value") graph_columns[1] = index;
        if (field.name == "neighbors") graph_columns[2] = index;
    }
    for (std::size_t const found : graph_columns)
        if (found == columns.size()) return fmt::format("{} lacks a key, value or neighbors column", arrow_path);

    std::string_view const recorded = arrow_metadata_value(schema.metadata, "metric");
    if (!metric.empty() && !recorded.empty() && metric != recorded)
        return fmt::format("{} records the {} metric, not {}", arrow_path, recorded, metric);
    std::vector<safegraphs_custom_field_t> custom_fields;
    if (recorded.empty() && !metric.empty()) custom_fields.push_back({"metric", metric});
    for (ipc_key_value_t const& pair : schema.metadata)
        if (pair.key != "alignment") custom_fields.push_back({pair.key, pair.value});

    std::vector<arrow_part_t> parts(blocks_count);
    std::size_t const alignment = safegraphs_alignment_default_k;
    for (std::size_t index = 0; index != blocks_count; ++index) {
        arrow_part_t& part = parts[index];
        ipc_block_t const& block = blocks[index];
        bytes_view_t const message_bytes = file.subspan(static_cast<std::size_t>(block.offset));
        std::uint32_t metadata_bytes = 0;
        std::size_t prefix_bytes = 0;
        ipc_message_t message;
        status = ipc_read_prefix(message_bytes, metadata_bytes, prefix_bytes);
        if (status == ipc_status_t::success_k && prefix_bytes + metadata_bytes > message_bytes.size())
            status = ipc_status_t::truncated_k;
        if (status == ipc_status_t::success_k)
            status = ipc_read_message(message_bytes.subspan(prefix_bytes, metadata_bytes), message);
        if (status == ipc_status_t::success_k)
            status = ipc_read_batch(message, message_bytes.subspan(static_cast<std::size_t>(block.metadata_bytes)),
                                    schema, part.nodes, part.buffers, part.batch);
        if (status == ipc_status_t::success_k) status = ipc_validate_batch(schema, part.batch);
        if (status != ipc_status_t::success_k)
            return fmt::format("{} holds a record batch {} that does not validate", arrow_path, index);

        // Each column takes its prefix sums, then its values, every buffer on the alignment.
        std::size_t const rows = static_cast<std::size_t>(part.batch.length);
        std::size_t end = 0;
        auto place = [&](std::size_t bytes) {
            std::size_t const start = round_up_to_multiple(end, alignment);
            return end = start + bytes, iota_view<std::size_t> {start, end};
        };
        for (std::size_t column_index = 0; column_index != columns.size(); ++column_index) {
            arrow_column_t const& column = columns[column_index];
            safegraphs_column_t restored {.dtype = column.dtype, .shape = column.shape};
            if (column.sums_dtype != dtype_t::unknown_k) {
                std::size_t const sum_bytes = dtype_bytes(column.sums_dtype);
                bytes_view_t const sums = part.batch.buffers[column.first_buffer + 1];
                auto sum_at = [&](std::size_t row) -> std::uint64_t {
                    return sum_bytes == 8 ? ipc_load_little<std::uint64_t>(sums.data() + 8 * row)
                                          : ipc_load_little<std::uint32_t>(sums.data() + 4 * row);
                };
                // A sliced producer's offsets start past zero, where SafeGraphs sums always start.
                if (rows && sum_at(0))
                    return fmt::format("{} holds a {} column whose offsets start past zero", arrow_path,
                                       column.field->name);
                std::uint64_t const elements = rows ? sum_at(rows) : 0;
                restored.offsets = prefix_sums_t {column.sums_dtype, place((rows + 1) * sum_bytes)};
                restored.data = place(static_cast<std::size_t>(elements) * column.element_bytes);
            }
            else restored.data = place(rows * column.element_bytes);
            part.columns.push_back(restored);
            if (column_index != graph_columns[0] && column_index != graph_columns[1] &&
                column_index != graph_columns[2])
                part.extra_columns.push_back({safegraphs_name_t {column.field->name}, restored});
        }
        part.header.key = part.columns[graph_columns[0]];
        part.header.value = part.columns[graph_columns[1]];
        part.header.neighbors = part.columns[graph_columns[2]];
        part.header.alignment = alignment;
        part.header.extra_columns = part.extra_columns;
        auto const [data_offset, offset_status] = safegraphs_data_offset(part.header, custom_fields);
        if (failed(offset_status)) return fmt::format("cannot lay out a header: {}", status_to_string(offset_status));
        part.header.data_offset = data_offset;
        part.bytes = data_offset + end;
        summary.nodes += rows;
    }

    // One batch is a lone monolith, and several follow the manifest that leads their buffer.
    std::vector<std::size_t> offsets(parts.size());
    auto part_at = [&](std::size_t part) {
        return safegraphs_part_t {.bytes = parts[part].bytes,
                                  .nodes = static_cast<std::size_t>(parts[part].batch.length)};
    };
    auto const [total, layout_status] = safegraphs_buffer_layout(parts.size(), part_at, alignment, offsets);
    if (failed(layout_status)) return fmt::format("cannot lay out a manifest: {}", status_to_string(layout_status));
    checkpoint.assign(total, '\0');
    bytes_span_t const buffer = std::as_writable_bytes(std::span {checkpoint});
    for (std::size_t index = 0; index != parts.size(); ++index) {
        arrow_part_t& part = parts[index];
        part.offset = offsets[index];
        bytes_span_t const monolith = buffer.subspan(part.offset, part.bytes);
        bytes_span_t const segment = monolith.subspan(part.header.data_offset);
        // An empty batch may carry no offsets at all, and its one zero sum is already in place.
        auto copy = [&](iota_view<std::size_t> range, bytes_view_t source) {
            if (std::size_t const bytes = min_of(range.size(), source.size()))
                std::memcpy(segment.data() + range.first, source.data(), bytes);
        };
        for (std::size_t column_index = 0; column_index != columns.size(); ++column_index) {
            arrow_column_t const& column = columns[column_index];
            safegraphs_column_t const& restored = part.columns[column_index];
            if (restored.offsets) copy(restored.offsets->bytes, part.batch.buffers[column.first_buffer + 1]);
            copy(restored.data, part.batch.buffers[column.values_buffer()]);
        }
        if (status_t const framed = safegraphs_write_header(monolith, part.header, custom_fields); failed(framed))
            return fmt::format("cannot write a header: {}", status_to_string(framed));
    }
    if (status_t const led = safegraphs_commit_buffer(buffer, parts.size(), part_at, offsets); failed(led))
        return fmt::format("cannot write the manifest: {}", status_to_string(led));

    summary.parts = parts.size(), summary.bytes = checkpoint.size();
    summary.metric = metric.empty() ? recorded : metric;
    return {};
}

/** What @ref arrow_matches_read answers: the key of every query, and what each one found. */
struct arrow_matches_t {
    std::vector<uuid128_t> queries;
    copied_matches_t matches;
};

/** Whether @p field holds one key per row, as @ref arrow_matches_write lays them out. */
inline bool arrow_field_holds_keys(ipc_field_t const& field) noexcept {
    return field.type == ipc_type_t::fixed_size_binary_k && field.width == sizeof(uuid128_t);
}

/** Whether @p fields are the columns @ref arrow_matches_write lays out, whoever wrote them. */
inline bool arrow_fields_hold_matches(std::span<ipc_field_t const> fields) noexcept {
    if (fields.size() != 2 || fields[0].name != "query" || !arrow_field_holds_keys(fields[0]) ||
        fields[1].name != "matches" || fields[1].type != ipc_type_t::large_list_k || fields[1].children.size() != 1)
        return false;
    ipc_field_t const& match = fields[1].children[0];
    if (match.type != ipc_type_t::struct_k || match.children.size() != 2) return false;
    ipc_field_t const& key = match.children[0];
    ipc_field_t const& distance = match.children[1];
    return key.name == "key" && arrow_field_holds_keys(key) && distance.name == "distance" &&
           distance.type == ipc_type_t::floating_point_k && distance.width == 64;
}

/** Appends the rows of @p batch, validated against the schema of matches, to @p read. */
inline void arrow_append_matches(ipc_batch_t const& batch, arrow_matches_t& read) {
    bytes_view_t const queries = batch.buffers[1], offsets = batch.buffers[3];
    bytes_view_t const keys = batch.buffers[6], distances = batch.buffers[8];
    std::size_t const rows = static_cast<std::size_t>(batch.length);
    for (std::size_t row = 0; row != rows; ++row) {
        read.queries.emplace_back();
        std::memcpy(read.queries.back().bytes, queries.data() + row * sizeof(uuid128_t), sizeof(uuid128_t));
        std::int64_t const first = ipc_load_little<std::int64_t>(offsets.data() + row * sizeof(std::int64_t));
        std::int64_t const last = ipc_load_little<std::int64_t>(offsets.data() + (row + 1) * sizeof(std::int64_t));
        read.matches.counts.push_back(static_cast<std::size_t>(last - first));
        for (std::size_t match = static_cast<std::size_t>(first); match != static_cast<std::size_t>(last); ++match) {
            read.matches.keys.emplace_back();
            std::memcpy(read.matches.keys.back().bytes, keys.data() + match * sizeof(uuid128_t), sizeof(uuid128_t));
            std::uint64_t const bits = ipc_load_little<std::uint64_t>(distances.data() + match * sizeof(std::uint64_t));
            read.matches.scores.push_back(std::bit_cast<f64_t>(bits));
        }
    }
}

/**
 *  @brief Reads the matches of a search back from @p arrow_path, in either framing.
 *
 *  @param[in] arrow_path The file, which any Arrow writer may have written.
 *  @param[out] read The queries and their matches.
 *  @return Why the read stopped, or nothing.
 */
inline std::string arrow_matches_read(std::string const& arrow_path, arrow_matches_t& read) {
    arrow_input_t input;
    if (std::string failure = input.open(arrow_path); !failure.empty()) return failure;
    if (!arrow_fields_hold_matches(input.schema().fields))
        return fmt::format("{} is no Arrow IPC of a search's matches", arrow_path);
    for (bool more = true;;) {
        if (std::string failure = input.next(more); !failure.empty()) return failure;
        if (!more) return {};
        arrow_append_matches(input.batch(), read);
    }
}

#pragma endregion Import

} // namespace unum::usearch::tools
