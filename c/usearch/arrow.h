/**
 *  @file c/usearch/arrow.h
 *  @author Ash Vardanian
 *  @date November 6, 2026
 *  @brief The binary interface between Apache Arrow and the views every call takes and answers.
 *
 *  Converters read one Arrow column into a @c usearch_keys_view_t, a @c usearch_rows_view_t or a
 *  @c usearch_tape_view_t in place, and exporters hand the keys, rows, tapes and ragged matches a
 *  call wrote into an arena out as Arrow columns in place. Loops over whole datasets, record
 *  batches and schemas belong to their callers.
 *
 *  Keys are @c fixed_size_binary(16), tagged as the @c arrow.uuid extension or not, whose bytes are
 *  those of @c usearch_key_t. Rows are a @c fixed_size_list of their scalar: @c double, @c float,
 *  @c halffloat, @c int8, @c uint8 or @c int32 as Arrow names them, @c bf16 as @c uint16 elements
 *  tagged @c numkong.bf16, the @c e5m2, @c e4m3, @c e3m2 and @c e2m3 floats as @c uint8 elements
 *  tagged @c numkong.e5m2 and alike, @c i4x2 and @c u4x2 as tagged @c uint8 elements of two columns
 *  each, and @c u1x8 as one @c fixed_size_binary tagged @c numkong.u1x8, most significant bit
 *  first. Geospatial observations are rows of 11 @c double columns, NaN marking what is unknown.
 *  Sequences are @c binary or @c large_binary, with @c utf8 and @c large_utf8 read as the same
 *  layout, and shapes are a @c list or @c large_list of @c fixed_size_list(3) points, by the width
 *  of their offsets. No column a converter reads may hold a null, a dictionary or a view.
 *
 *  A search's ragged matches export as a @c large_list of @c struct{key, distance}, and a cursor's
 *  links as a @c large_list of keys; padded matches export their keys and scores as keys and rows.
 *
 *  The structures of the Arrow C data and device interfaces come from Arrow's own
 *  @c usearch/arrow_abi.h, under the guards Arrow defines, so a program including another copy of
 *  them compiles with either one.
 */

#ifndef UNUM_USEARCH_ARROW_H
#define UNUM_USEARCH_ARROW_H

#include "usearch/arrow_abi.h" // `ArrowSchema`, `ArrowArray`, `ArrowDeviceArray`
#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Views the 16-byte keys of one Arrow @c fixed_size_binary(16) column in place, tagged
 *      @c arrow.uuid or not.
 *  @param[in] schema The column's schema.
 *  @param[in] array The column, whose @c offset is applied; only its lengths and pointers are
 *      read, never its buffers, so the @c array of an @c ArrowDeviceArray converts as well.
 *  @param[out] keys The keys, one every 16 bytes.
 *  @param[out] count The number of keys.
 *  @return @c usearch_status_invalid_argument_k for a missing or released input or one holding a
 *      null, @c usearch_status_unsupported_layout_k for another format, a dictionary or a view,
 *      @c usearch_status_malformed_k for a negative length or offset or a missing buffer, and
 *      @c usearch_status_overflow_k for a length past what @c size_t counts.
 *
 *  A @c null_count of -1 with a validity buffer may hold a null, and is refused as one.
 */
USEARCH_EXPORT usearch_status_t usearch_keys_view_from_arrow(struct ArrowSchema const* schema,
                                                             struct ArrowArray const* array, usearch_keys_view_t* keys,
                                                             size_t* count);

/**
 *  @brief Views the rows of one Arrow @c fixed_size_list column, or of a @c fixed_size_binary one
 *      tagged @c numkong.u1x8, in place, the offsets of the column and its child applied.
 *  @param[in] schema The column's schema.
 *  @param[in] array The column, whose @c offset is applied.
 *  @param[out] rows The rows, packed, with their @c columns and @c scalar.
 *  @param[out] count The number of rows.
 *  @return The statuses of @c usearch_keys_view_from_arrow, refusing a list of @c bool elements as
 *      an unsupported layout too.
 */
USEARCH_EXPORT usearch_status_t usearch_rows_view_from_arrow(struct ArrowSchema const* schema,
                                                             struct ArrowArray const* array, usearch_rows_view_t* rows,
                                                             size_t* count);

/**
 *  @brief Views one Arrow column of sequences or shapes in place: @c binary, @c utf8 or a @c list
 *      of 3-element lists as a tape of 32-bit offsets, and @c large_binary, @c large_utf8 or a
 *      @c large_list of them as one of 64-bit offsets, starting at the column's @c offset.
 *  @param[in] schema The column's schema.
 *  @param[in] array The column, whose @c offset is applied.
 *  @param[out] tape The tape, its @c scalar @c u8 for bytes and the points' otherwise.
 *  @param[out] count The number of entries.
 *  @return The statuses of @c usearch_keys_view_from_arrow.
 */
USEARCH_EXPORT usearch_status_t usearch_tape_view_from_arrow(struct ArrowSchema const* schema,
                                                             struct ArrowArray const* array, usearch_tape_view_t* tape,
                                                             size_t* count);

/**
 *  @brief Exports @p count keys a call wrote into @p arena as one Arrow @c fixed_size_binary(16)
 *      column tagged as @c arrow.uuid, in place.
 *
 *  The array holds an export of @p arena open, as @c usearch_arena_export_open does, until its
 *  @c release, so no call rewinds the arena meanwhile. Arenas on host pages export as
 *  @c ARROW_DEVICE_CPU, and those of a VRAM executor as @c ARROW_DEVICE_CUDA, @c ARROW_DEVICE_ROCM
 *  or @c ARROW_DEVICE_METAL with the vendor's ordinal of their device and no @c sync_event, as the
 *  call that filled one drained its uploads. The schema's @c name is empty, for the caller to point
 *  at its own. Schema and array release apart, each freeing only its own nodes, and a consumer may
 *  move their children out before either release.
 *
 *  @par Statuses:
 *  - @c usearch_status_contended_k A call holds @p arena.
 *  - @c usearch_status_overflow_k The arena already counts @c UINT32_MAX exports.
 *  - @c usearch_status_invalid_argument_k An output is missing, or the keys lie outside @p arena.
 *  - @c usearch_status_unsupported_layout_k The keys are not packed.
 *  - @c usearch_status_out_of_memory_k An allocation failed.
 */
USEARCH_EXPORT usearch_status_t usearch_keys_view_to_arrow(usearch_keys_view_t keys, size_t count,
                                                           usearch_arena_t arena, struct ArrowSchema* schema,
                                                           struct ArrowDeviceArray* array);

/**
 *  @brief Exports @p count packed rows a call wrote into @p arena as one Arrow @c fixed_size_list
 *      column of their scalar, or a @c fixed_size_binary one for @c u1x8, in place.
 *  @return The statuses of @c usearch_keys_view_to_arrow, with a scalar Arrow holds no layout
 *      for, or columns filling no whole element, refused as an unsupported layout.
 */
USEARCH_EXPORT usearch_status_t usearch_rows_view_to_arrow(usearch_rows_view_t rows, size_t count,
                                                           usearch_arena_t arena, struct ArrowSchema* schema,
                                                           struct ArrowDeviceArray* array);

/**
 *  @brief Exports a tape of @p count entries a call wrote into @p arena as one Arrow column in
 *      place, its offsets as they are: @c binary or @c large_binary for @c u8 elements, which a
 *      sequence's are, and a @c list or @c large_list of 3-element lists for any other scalar.
 *  @return The statuses of @c usearch_keys_view_to_arrow, with an unknown layout refused as an
 *      unsupported one, and @c usearch_status_device_k for a device's last offset no driver reads.
 *
 *  A @c list length comes from the tape's last offset, which crosses from a device arena through
 *  its driver. Shape points asked for in @c u8 read as @c binary, whose offsets are bytes, so ask
 *  a shapes get for another scalar before exporting it.
 */
USEARCH_EXPORT usearch_status_t usearch_tape_view_to_arrow(usearch_tape_view_t tape, size_t count,
                                                           usearch_arena_t arena, struct ArrowSchema* schema,
                                                           struct ArrowDeviceArray* array);

/**
 *  @brief Exports a ragged run of @p count entries' keys a call wrote into @p arena as one Arrow
 *      @c large_list of @c arrow.uuid keys, in place.
 *  @return The statuses of @c usearch_tape_view_to_arrow.
 */
USEARCH_EXPORT usearch_status_t usearch_keys_tape_view_to_arrow(usearch_keys_tape_view_t keys, size_t count,
                                                                usearch_arena_t arena, struct ArrowSchema* schema,
                                                                struct ArrowDeviceArray* array);

/**
 *  @brief Exports a ragged run of @p count queries' matches a call wrote into @p arena as one Arrow
 *      @c large_list of @c struct{key, distance}, in place.
 *  @return The statuses of @c usearch_tape_view_to_arrow.
 */
USEARCH_EXPORT usearch_status_t usearch_matches_tape_view_to_arrow(usearch_matches_tape_view_t matches, size_t count,
                                                                   usearch_arena_t arena, struct ArrowSchema* schema,
                                                                   struct ArrowDeviceArray* array);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_ARROW_H
