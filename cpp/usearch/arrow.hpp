/**
 *  @file cpp/usearch/arrow.hpp
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief Views of Apache Arrow columns as the rows, keys and tapes every call takes, and exports
 *      of what a call wrote into an arena as Arrow columns, all in place and throwing nothing.
 *
 *  A converter reads one column's schema and array and answers the view with its entry count, or
 *  the status explaining why the column has no such view. An exporter fills a caller's schema and
 *  device array, which read the arena's outputs until its next call and release apart.
 *
 *  @code{.cpp}
 *  namespace us = unum::usearch;
 *  auto [column, status] = us::keys_view_from_arrow(keys_schema, keys_array);
 *  std::span<usearch_key_t const> const keys {column.view.data, column.count}; // 16 bytes apart
 *  auto [batch, cursor_status] = index.on(executor).cursor_next(cursor, 1024, arena);
 *  ArrowSchema schema {};
 *  ArrowDeviceArray array {};
 *  us::status_t const exported = us::keys_view_to_arrow(batch.keys, batch.written, arena, schema, array);
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`

#include "usearch/arrow.h"   // `ArrowSchema`, `ArrowArray`, `ArrowDeviceArray`, the converters and exporters
#include "usearch/types.hpp" // `expected`, `status_t`, `converted_`, `arena_t`

namespace unum::usearch {

/** One Arrow column viewed in place as @p view_type_, with the entries it holds. */
template <typename view_type_>
struct arrow_column {

    /** The column's entries as a call takes them. */
    view_type_ view {};

    /** Entries in the column. */
    std::size_t count = 0;
};

/** Views @p array of @p schema through @p converter, or answers why it has no such view. */
template <typename view_type_>
expected<arrow_column<view_type_>> viewed_from_arrow_(usearch_status_t (*converter)(ArrowSchema const*,
                                                                                    ArrowArray const*, view_type_*,
                                                                                    std::size_t*),
                                                      ArrowSchema const& schema, ArrowArray const& array) noexcept {
    arrow_column<view_type_> column {};
    status_t const status = converted_(converter(&schema, &array, &column.view, &column.count),
                                       "The Arrow column has no view of this kind");
    return {column, status};
}

/** Views the 16-byte keys of one @c fixed_size_binary(16) column in place. */
inline expected<arrow_column<usearch_keys_view_t>> keys_view_from_arrow(ArrowSchema const& schema,
                                                                        ArrowArray const& array) noexcept {
    return viewed_from_arrow_(usearch_keys_view_from_arrow, schema, array);
}

/** Views the rows of one @c fixed_size_list column, or of a bit-packed @c fixed_size_binary one,
 *  in place. */
inline expected<arrow_column<usearch_rows_view_t>> rows_view_from_arrow(ArrowSchema const& schema,
                                                                        ArrowArray const& array) noexcept {
    return viewed_from_arrow_(usearch_rows_view_from_arrow, schema, array);
}

/** Views one column of sequences or shapes in place, as a tape of 32- or 64-bit offsets. */
inline expected<arrow_column<usearch_tape_view_t>> tape_view_from_arrow(ArrowSchema const& schema,
                                                                        ArrowArray const& array) noexcept {
    return viewed_from_arrow_(usearch_tape_view_from_arrow, schema, array);
}

/** Exports @p count keys a call wrote into @p arena as one @c fixed_size_binary(16) column. */
inline status_t keys_view_to_arrow(usearch_keys_view_t keys, std::size_t count, arena_t const& arena,
                                   ArrowSchema& schema, ArrowDeviceArray& array) noexcept {
    return converted_(usearch_keys_view_to_arrow(keys, count, arena.handle(), &schema, &array),
                      "The keys could not be exported");
}

/** Exports @p count packed rows a call wrote into @p arena as one @c fixed_size_list column. */
inline status_t rows_view_to_arrow(usearch_rows_view_t rows, std::size_t count, arena_t const& arena,
                                   ArrowSchema& schema, ArrowDeviceArray& array) noexcept {
    return converted_(usearch_rows_view_to_arrow(rows, count, arena.handle(), &schema, &array),
                      "The rows could not be exported");
}

/** Exports a tape of @p count entries a call wrote into @p arena as one binary or list column. */
inline status_t tape_view_to_arrow(usearch_tape_view_t tape, std::size_t count, arena_t const& arena,
                                   ArrowSchema& schema, ArrowDeviceArray& array) noexcept {
    return converted_(usearch_tape_view_to_arrow(tape, count, arena.handle(), &schema, &array),
                      "The tape could not be exported");
}

/** Exports the linked keys of @p count entries a call wrote into @p arena as one @c large_list
 *  column. */
inline status_t keys_tape_view_to_arrow(usearch_keys_tape_view_t keys, std::size_t count, arena_t const& arena,
                                        ArrowSchema& schema, ArrowDeviceArray& array) noexcept {
    return converted_(usearch_keys_tape_view_to_arrow(keys, count, arena.handle(), &schema, &array),
                      "The linked keys could not be exported");
}

/** Exports the ragged matches of @p count queries a call wrote into @p arena as one
 *  @c large_list column. */
inline status_t matches_tape_view_to_arrow(usearch_matches_tape_view_t matches, std::size_t count, arena_t const& arena,
                                           ArrowSchema& schema, ArrowDeviceArray& array) noexcept {
    return converted_(usearch_matches_tape_view_to_arrow(matches, count, arena.handle(), &schema, &array),
                      "The matches could not be exported");
}

} // namespace unum::usearch
