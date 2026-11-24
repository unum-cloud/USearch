// Apache Arrow columns viewed as inputs, and arena outputs exported as columns, mirroring arrow.h.
//
// File: golang/arrow.go
// Author: Ash Vardanian
// Date: November 18, 2026

package usearch

import "unsafe"

/*
#cgo noescape usearch_keys_view_from_arrow
#cgo nocallback usearch_keys_view_from_arrow
#cgo noescape usearch_rows_view_from_arrow
#cgo nocallback usearch_rows_view_from_arrow
#cgo noescape usearch_tape_view_from_arrow
#cgo nocallback usearch_tape_view_from_arrow
#cgo noescape usearch_keys_view_to_arrow
#cgo nocallback usearch_keys_view_to_arrow
#cgo noescape usearch_rows_view_to_arrow
#cgo nocallback usearch_rows_view_to_arrow
#cgo noescape usearch_tape_view_to_arrow
#cgo nocallback usearch_tape_view_to_arrow
#cgo noescape usearch_keys_tape_view_to_arrow
#cgo nocallback usearch_keys_tape_view_to_arrow
#cgo noescape usearch_matches_tape_view_to_arrow
#cgo nocallback usearch_matches_tape_view_to_arrow

#include "shim.h"
*/
import "C"

// ArrowSchema is the Arrow C data interface's schema of one column, laid out as struct ArrowSchema,
// so any Arrow library reads it through its address.
type ArrowSchema C.struct_ArrowSchema

// ArrowArray is the Arrow C data interface's array of one column, laid out as struct ArrowArray.
type ArrowArray C.struct_ArrowArray

// ArrowDeviceArray is an ArrowArray beside the device its buffers live on, laid out as struct
// ArrowDeviceArray.
type ArrowDeviceArray C.struct_ArrowDeviceArray

// Release hands the schema back to its producer, once; a released schema is safe to release again.
func (schema *ArrowSchema) Release() { C.usearch_go_release_schema((*C.struct_ArrowSchema)(schema)) }

// Release hands the array back to its producer, once; a released array is safe to release again.
func (array *ArrowDeviceArray) Release() {
	C.usearch_go_release_array((*C.struct_ArrowDeviceArray)(array))
}

// Array reports the column itself, which the converters read as they read a host array.
func (array *ArrowDeviceArray) Array() *ArrowArray {
	return (*ArrowArray)(unsafe.Pointer(&array.array))
}

// arrowed turns the status a converter or an exporter answers into a Go error.
func arrowed(status C.usearch_status_t) error {
	if status == C.usearch_status_success_k {
		return nil
	}
	return fail(Status(status), "the Arrow column could not be converted")
}

// KeysViewFromArrow views the 16-byte keys of a fixed_size_binary(16) column in place, tagged
// arrow.uuid or not, viewing the column's buffers.
func KeysViewFromArrow(schema *ArrowSchema, array *ArrowArray) ([]Key, error) {
	var keys C.usearch_keys_view_t
	var count C.size_t
	if err := arrowed(C.usearch_keys_view_from_arrow((*C.struct_ArrowSchema)(schema), (*C.struct_ArrowArray)(array), &keys, &count)); err != nil {
		return nil, err
	}
	return unsafe.Slice((*Key)(unsafe.Pointer(keys.data)), count), nil
}

// RowsViewFromArrow views the rows of a fixed_size_list column, or of a fixed_size_binary one
// tagged numkong.u1x8, in place, viewing the column's buffers.
func RowsViewFromArrow(schema *ArrowSchema, array *ArrowArray) (RowsView, error) {
	var rows C.usearch_rows_view_t
	var count C.size_t
	if err := arrowed(C.usearch_rows_view_from_arrow((*C.struct_ArrowSchema)(schema), (*C.struct_ArrowArray)(array), &rows, &count)); err != nil {
		return RowsView{}, err
	}
	return RowsView{pointer: rows.data, rows: uint(count), dimensions: uint(rows.columns), stride: uint(rows.stride), scalar: Scalar(rows.scalar)}, nil
}

// TapeViewFromArrow views a column of sequences or shapes in place as a Tape: binary, utf8 or a
// list of 3-element lists with 32-bit offsets, and their large variants with 64-bit ones.
func TapeViewFromArrow(schema *ArrowSchema, array *ArrowArray) (Tape, error) {
	var tape C.usearch_tape_view_t
	var count C.size_t
	if err := arrowed(C.usearch_tape_view_from_arrow((*C.struct_ArrowSchema)(schema), (*C.struct_ArrowArray)(array), &tape, &count)); err != nil {
		return Tape{}, err
	}
	return Tape{
		pointer: tape.data,
		offsets: *(*unsafe.Pointer)(unsafe.Pointer(&tape.offsets)),
		entries: uint(count),
		layout:  tape.layout,
		scalar:  Scalar(tape.scalar),
	}, nil
}

// KeysViewToArrow exports keys a call wrote into arena as one fixed_size_binary(16) column tagged
// arrow.uuid, in place, valid until the arena's next call or close; release both outputs once read.
func KeysViewToArrow(keys []Key, arena *Arena, schema *ArrowSchema, array *ArrowDeviceArray) error {
	outputs, err := arena.lower()
	if err != nil {
		return err
	}
	view := C.usearch_keys_view_t{data: (*C.usearch_key_t)(unsafe.Pointer(unsafe.SliceData(keys))), stride: C.size_t(unsafe.Sizeof(Key{}))}
	return arrowed(C.usearch_keys_view_to_arrow(view, C.size_t(len(keys)), outputs, (*C.struct_ArrowSchema)(schema), (*C.struct_ArrowDeviceArray)(array)))
}

// RowsViewToArrow exports the packed rows a call wrote into arena as one fixed_size_list column of
// their scalar, or a fixed_size_binary one for U1x8Scalar, in place.
func RowsViewToArrow(vectors DenseVectors, arena *Arena, schema *ArrowSchema, array *ArrowDeviceArray) error {
	outputs, err := arena.lower()
	if err != nil {
		return err
	}
	return arrowed(C.usearch_rows_view_to_arrow(vectors.raw, C.size_t(vectors.rows), outputs, (*C.struct_ArrowSchema)(schema), (*C.struct_ArrowDeviceArray)(array)))
}

// TapeViewToArrow exports the entries a call wrote into arena as one column in place: binary or
// large binary for a sequence's bytes, and a list or large list of 3-element lists for points.
func TapeViewToArrow(tape TapeView, arena *Arena, schema *ArrowSchema, array *ArrowDeviceArray) error {
	outputs, err := arena.lower()
	if err != nil {
		return err
	}
	return arrowed(C.usearch_tape_view_to_arrow(tape.raw, C.size_t(tape.written), outputs, (*C.struct_ArrowSchema)(schema), (*C.struct_ArrowDeviceArray)(array)))
}

// KeysTapeViewToArrow exports the links a cursor pull wrote into arena as one large list of
// arrow.uuid keys, in place; a batch from NextKeys carries none to export.
func KeysTapeViewToArrow(batch DenseBatch, arena *Arena, schema *ArrowSchema, array *ArrowDeviceArray) error {
	outputs, err := arena.lower()
	if err != nil {
		return err
	}
	return arrowed(C.usearch_keys_tape_view_to_arrow(batch.links, C.size_t(batch.written), outputs, (*C.struct_ArrowSchema)(schema), (*C.struct_ArrowDeviceArray)(array)))
}

// MatchesTapeViewToArrow exports the ragged matches a call wrote into arena as one large list of
// struct{key, distance}, in place.
func MatchesTapeViewToArrow(matches MatchesTape, arena *Arena, schema *ArrowSchema, array *ArrowDeviceArray) error {
	outputs, err := arena.lower()
	if err != nil {
		return err
	}
	return arrowed(C.usearch_matches_tape_view_to_arrow(matches.raw, C.size_t(matches.queries), outputs, (*C.struct_ArrowSchema)(schema), (*C.struct_ArrowDeviceArray)(array)))
}
