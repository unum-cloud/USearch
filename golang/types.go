// Shared vocabulary of every USearch modality, which the dense modality builds on.
//
// File: golang/types.go
// Author: Ash Vardanian
// Date: August 16, 2026

package usearch

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"runtime"
	"runtime/cgo"
	"strings"
	"unsafe"
)

/*
#cgo CFLAGS: -I${SRCDIR}/../c
#cgo LDFLAGS: -lusearch_static
#cgo linux LDFLAGS: -lstdc++ -lm
#cgo darwin LDFLAGS: -lc++
#cgo windows LDFLAGS: -lstdc++

// A function is noescape when C keeps none of its Go pointers past the call, and nocallback when
// its argument structure carries no callback field: an executor runs only the callbacks of the call
// that names them.
#cgo noescape usearch_outcome_status
#cgo nocallback usearch_outcome_status
#cgo noescape usearch_outcome_wait
#cgo nocallback usearch_outcome_wait
#cgo noescape usearch_executor_for_ram
#cgo nocallback usearch_executor_for_ram
#cgo noescape usearch_executor_for_vram
#cgo nocallback usearch_executor_for_vram
#cgo noescape usearch_executor_for_nvme
#cgo nocallback usearch_executor_for_nvme
#cgo noescape usearch_executor_drain
#cgo nocallback usearch_executor_drain
#cgo noescape usearch_arena_init
#cgo nocallback usearch_arena_init
#cgo noescape usearch_arena_describe
#cgo nocallback usearch_arena_describe
#cgo noescape usearch_status_from_name
#cgo nocallback usearch_status_from_name
#cgo noescape usearch_scalar_from_name
#cgo nocallback usearch_scalar_from_name
#cgo noescape usearch_residency_from_name
#cgo nocallback usearch_residency_from_name
#cgo noescape usearch_traversal_from_name
#cgo nocallback usearch_traversal_from_name
#cgo noescape usearch_mood_from_name
#cgo nocallback usearch_mood_from_name
#cgo noescape usearch_refine_verdict_from_name
#cgo nocallback usearch_refine_verdict_from_name
#cgo noescape usearch_phase_from_name
#cgo nocallback usearch_phase_from_name
#cgo noescape usearch_cpu_class_from_name
#cgo nocallback usearch_cpu_class_from_name
#cgo noescape usearch_gpu_class_from_name
#cgo nocallback usearch_gpu_class_from_name
#cgo noescape usearch_nvme_class_from_name
#cgo nocallback usearch_nvme_class_from_name

#include "shim.h"
*/
import "C"

// region: Errors

// Status names what went wrong, in a vocabulary shared by the engine and by this binding.
// Validation performed here before a call crosses reports the status the engine would have
// reported for the same fault.
//
// A Status is an error, so errors.Is(err, usearch.StatusReadOnly) matches any failure holding it.
type Status int32

// The statuses a call ends in. Only Deferred.Status reports StatusDeferred, the one status
// that is not final.
const (
	// StatusSuccess means the call completed and holds no message.
	StatusSuccess Status = C.usearch_status_success_k
	// StatusDeferred means the call was submitted deferred and is not final yet.
	StatusDeferred Status = C.usearch_status_deferred_k
	// StatusOutOfMemory means an allocation or reservation could not be satisfied.
	StatusOutOfMemory Status = C.usearch_status_out_of_memory_k
	// StatusShapeMismatch means dimensionality or row counts disagree.
	StatusShapeMismatch Status = C.usearch_status_shape_mismatch_k
	// StatusUnsupportedLayout means a stride or packing the engine cannot traverse.
	StatusUnsupportedLayout Status = C.usearch_status_unsupported_layout_k
	// StatusUnknownKind means a scalar, metric, or residency tag this build does not know.
	StatusUnknownKind Status = C.usearch_status_unknown_kind_k
	// StatusReadOnly means a mutating verb reached a handle that serves its checkpoint in place.
	StatusReadOnly Status = C.usearch_status_read_only_k
	// StatusIO means a file could not be opened, read, written, or parsed.
	StatusIO Status = C.usearch_status_io_k
	// StatusDevice means a device could not be selected, or its memory could not be claimed.
	StatusDevice Status = C.usearch_status_device_k
	// StatusMalformed means a caller-supplied representation could not be decoded.
	StatusMalformed Status = C.usearch_status_malformed_k
	// StatusInternal means the library reported neither success nor a failure.
	StatusInternal Status = C.usearch_status_internal_k
	// StatusOverflow means a hard design ceiling was reached, like the 32-bit ID space.
	StatusOverflow Status = C.usearch_status_overflow_k
	// StatusUninitialized means the verb needs a prior successful reservation that never happened.
	StatusUninitialized Status = C.usearch_status_uninitialized_k
	// StatusConfigMismatch means the request contradicts an already-frozen configuration.
	StatusConfigMismatch Status = C.usearch_status_config_mismatch_k
	// StatusContended means the call was refused rather than waited for, like a submission to an
	// executor another call holds, or an arena still serving another call.
	StatusContended Status = C.usearch_status_contended_k
	// StatusCancelled means a progress handler asked the operation to stop; it unwound and published
	// nothing, so the handle holds whatever it did before the call.
	StatusCancelled Status = C.usearch_status_cancelled_k
	// StatusUnsupported means the verb is declared for every residency but this one does not
	// implement it; a caller picks another residency.
	StatusUnsupported Status = C.usearch_status_unsupported_k
	// StatusInvalidArgument means an argument breaks the verb's contract, like a required value left
	// nil.
	StatusInvalidArgument Status = C.usearch_status_invalid_argument_k
)

// String renders the status as the library spells it.
func (status Status) String() string {
	return C.GoString(C.usearch_status_name(C.usearch_status_t(status)))
}

// StatusFromName reads a status back from the spelling String gives it,
// failing with StatusInvalidArgument on a name the library does not spell.
func StatusFromName(name string) (Status, error) {
	var named C.usearch_status_t
	if status := C.usearch_status_from_name(view(name), &named); status != C.usearch_status_success_k {
		return 0, fail(Status(status), "no status is spelled \""+name+"\"")
	}
	return Status(named), nil
}

// Error implements the error interface, so a bare Status is an errors.Is target.
func (status Status) Error() string { return status.String() }

// Error is one failed call: a status every caller can branch on, and the detail the status
// cannot hold. Reach the status with errors.Is or errors.As.
type Error struct {
	Status  Status
	Message string
}

// Error implements the error interface.
func (failure *Error) Error() string {
	return fmt.Sprintf("usearch: %s: %s", failure.Status, failure.Message)
}

// Unwrap exposes the status, which is what errors.Is compares.
func (failure *Error) Unwrap() error { return failure.Status }

// fail builds the failure this binding reports for a fault it caught before the boundary.
func fail(status Status, message string) *Error {
	return &Error{Status: status, Message: message}
}

// check turns a final outcome into a Go error. Messages are static library storage and need no
// deallocation, so the copy is the only cost.
func check(outcome C.usearch_outcome_t) error {
	if outcome.status == C.usearch_status_success_k {
		return nil
	}
	return &Error{Status: Status(outcome.status), Message: C.GoString(outcome.message)}
}

// endregion: Errors

// region: Keys & Distances

// Distance is a score between two entries, in the metric of the index that produced it.
type Distance = float64

// Key is a 128-bit entry identity, most significant byte first, rendered as a lowercase
// dashed UUID.
//
// Keys need not be unique: several entries may share one, and DenseIndexOn.Count reports how
// many. Integer identities park in the low eight bytes, so KeyFromUint64(42) prints as
// 00000000-0000-0000-0000-00000000002a.
type Key [16]byte

// NilKey is the all-zero key.
var NilKey Key

// KeyFromUint64 parks an integer identity in the low eight bytes, zeroing the rest.
func KeyFromUint64(value uint64) Key {
	var key Key
	binary.BigEndian.PutUint64(key[8:], value)
	return key
}

// Uint64 reports the integer identity in the low eight bytes, and false when the high eight
// hold anything.
func (key Key) Uint64() (uint64, bool) {
	if binary.BigEndian.Uint64(key[:8]) != 0 {
		return 0, false
	}
	return binary.BigEndian.Uint64(key[8:]), true
}

// String renders the key as a lowercase dashed UUID.
func (key Key) String() string {
	var builder strings.Builder
	builder.Grow(36)
	for position, digit := range key {
		if position == 4 || position == 6 || position == 8 || position == 10 {
			builder.WriteByte('-')
		}
		const hexadecimal = "0123456789abcdef"
		builder.WriteByte(hexadecimal[digit>>4])
		builder.WriteByte(hexadecimal[digit&0x0f])
	}
	return builder.String()
}

// ParseKey reads the rendering String produces. Dashes may sit anywhere, or nowhere.
func ParseKey(text string) (Key, error) {
	var key Key
	digits := strings.ReplaceAll(text, "-", "")
	if len(digits) != 2*len(key) {
		return NilKey, fail(StatusMalformed, "the key is not thirty-two hexadecimal digits")
	}
	if _, err := hex.Decode(key[:], []byte(digits)); err != nil {
		return NilKey, fail(StatusMalformed, "the key is not thirty-two hexadecimal digits")
	}
	return key, nil
}

// MarshalText implements encoding.TextMarshaler, so a key inside a struct round-trips
// through JSON as the dashed UUID String prints.
func (key Key) MarshalText() ([]byte, error) { return []byte(key.String()), nil }

// AppendText implements encoding.TextAppender.
func (key Key) AppendText(buffer []byte) ([]byte, error) { return append(buffer, key.String()...), nil }

// UnmarshalText implements encoding.TextUnmarshaler.
func (key *Key) UnmarshalText(text []byte) error {
	parsed, err := ParseKey(string(text))
	if err != nil {
		return err
	}
	*key = parsed
	return nil
}

// A batch reinterprets a whole []Key as the C array, so a single key must reinterpret the
// same way rather than copying byte by byte - one path, provable by the assertion below.
func (key Key) toRaw() C.usearch_key_t { return *(*C.usearch_key_t)(unsafe.Pointer(&key)) }

func keyFromRaw(raw C.usearch_key_t) Key { return *(*Key)(unsafe.Pointer(&raw)) }

// endregion: Keys & Distances

// region: Scalars

// Scalar names a stored or exchanged scalar type by width and encoding. Sub-byte kinds
// pack several dimensions per byte, as their xN suffixes spell.
type Scalar int32

// The scalar types rows travel in, ordered by shrinking dynamic range. The numbering
// leaves gaps for kinds that do not exist yet.
const (
	// ScalarUnknown tags a type the library could not name.
	ScalarUnknown Scalar = C.usearch_scalar_unknown_k
	// ScalarF64 is 64-bit double precision.
	ScalarF64 Scalar = C.usearch_scalar_f64_k
	// ScalarF32 is 32-bit single precision.
	ScalarF32 Scalar = C.usearch_scalar_f32_k
	// ScalarBF16 is the brain floating-point format.
	ScalarBF16 Scalar = C.usearch_scalar_bf16_k
	// ScalarF16 is IEEE half precision.
	ScalarF16 Scalar = C.usearch_scalar_f16_k
	// ScalarE5M2 is 8-bit floating point with five exponent bits.
	ScalarE5M2 Scalar = C.usearch_scalar_e5m2_k
	// ScalarE4M3 is 8-bit floating point with four exponent bits.
	ScalarE4M3 Scalar = C.usearch_scalar_e4m3_k
	// ScalarE3M2 is 6-bit floating point with three exponent bits.
	ScalarE3M2 Scalar = C.usearch_scalar_e3m2_k
	// ScalarE2M3 is 6-bit floating point with two exponent bits.
	ScalarE2M3 Scalar = C.usearch_scalar_e2m3_k
	// ScalarI8 is an 8-bit signed integer.
	ScalarI8 Scalar = C.usearch_scalar_i8_k
	// ScalarI4x2 packs two 4-bit signed integers per byte.
	ScalarI4x2 Scalar = C.usearch_scalar_i4x2_k
	// ScalarI32 is a 32-bit signed integer, the coordinates of geospatial points.
	ScalarI32 Scalar = C.usearch_scalar_i32_k
	// ScalarU8 is an 8-bit unsigned integer.
	ScalarU8 Scalar = C.usearch_scalar_u8_k
	// ScalarU4x2 packs two 4-bit unsigned integers per byte.
	ScalarU4x2 Scalar = C.usearch_scalar_u4x2_k
	// ScalarU1x8 packs eight single-bit dimensions per byte.
	ScalarU1x8 Scalar = C.usearch_scalar_u1x8_k
)

// String renders the scalar kind as the library spells it.
func (scalar Scalar) String() string {
	return C.GoString(C.usearch_scalar_name(C.usearch_scalar_t(scalar)))
}

// ScalarFromName reads a scalar kind back from the spelling String gives it,
// failing with StatusInvalidArgument on a name the library does not spell.
func ScalarFromName(name string) (Scalar, error) {
	var scalar C.usearch_scalar_t
	if status := C.usearch_scalar_from_name(view(name), &scalar); status != C.usearch_status_success_k {
		return 0, fail(Status(status), "no scalar kind is spelled \""+name+"\"")
	}
	return Scalar(scalar), nil
}

// dimensionsPerValue is how many logical dimensions one stored value holds: one for the
// byte-and-wider kinds, more for the packed ones.
func (scalar Scalar) dimensionsPerValue() uint {
	switch scalar {
	case ScalarI4x2, ScalarU4x2:
		return 2
	case ScalarU1x8:
		return 8
	default:
		return 1
	}
}

// Storage types for the encodings Go has no builtin for. Each is a distinct defined type so
// RowsOf can tell them apart, and each holds the raw bits rather than a converted value.
type (
	// BFloat16 holds the sixteen bits of a brain floating-point value.
	BFloat16 uint16
	// Float16 holds the sixteen bits of an IEEE half-precision value.
	Float16 uint16
	// E5M2Scalar holds the eight bits of a five-exponent minifloat.
	E5M2Scalar uint8
	// E4M3Scalar holds the eight bits of a four-exponent minifloat.
	E4M3Scalar uint8
	// E3M2Scalar holds the six occupied bits of a three-exponent minifloat.
	E3M2Scalar uint8
	// E2M3Scalar holds the six occupied bits of a two-exponent minifloat.
	E2M3Scalar uint8
	// I4x2Scalar holds two 4-bit signed integers.
	I4x2Scalar int8
	// U4x2Scalar holds two 4-bit unsigned integers.
	U4x2Scalar uint8
	// U1x8Scalar holds eight single-bit dimensions.
	U1x8Scalar uint8
)

// Element constrains the element types a RowsView may view, ordered by shrinking dynamic range.
type Element interface {
	float64 | float32 | BFloat16 | Float16 |
		E5M2Scalar | E4M3Scalar | E3M2Scalar | E2M3Scalar |
		int8 | I4x2Scalar | uint8 | U4x2Scalar | U1x8Scalar
}

// scalarOf names the scalar kind of a constrained element type.
func scalarOf[Value Element]() Scalar {
	var probe Value
	switch any(probe).(type) {
	case float64:
		return ScalarF64
	case float32:
		return ScalarF32
	case BFloat16:
		return ScalarBF16
	case Float16:
		return ScalarF16
	case E5M2Scalar:
		return ScalarE5M2
	case E4M3Scalar:
		return ScalarE4M3
	case E3M2Scalar:
		return ScalarE3M2
	case E2M3Scalar:
		return ScalarE2M3
	case int8:
		return ScalarI8
	case I4x2Scalar:
		return ScalarI4x2
	case uint8:
		return ScalarU8
	case U4x2Scalar:
		return ScalarU4x2
	case U1x8Scalar:
		return ScalarU1x8
	default:
		return ScalarUnknown
	}
}

// endregion: Scalars

// region: Row Views

// RowsView views a caller-owned slab as rows of equal width: a base pointer, a row count, and
// a byte pitch between consecutive rows. It is the shape every batch argument takes.
//
// A RowsView only points at memory it never owns. It stays valid only as long as the slice it was
// built from, which the binding pins until each call it crosses is final: the collector traces the
// base pointer, so keeping the view alive keeps the backing array alive.
type RowsView struct {
	pointer    unsafe.Pointer
	rows       uint
	dimensions uint
	stride     uint
	scalar     Scalar
}

// RowsOf views a packed slab as rows of the given width.
//
// Panics on a zero width or a partial trailing row: both name a caller's arithmetic mistake
// rather than a condition the library could report. Sub-byte kinds count logical dimensions,
// so sixteen dimensions of U1x8Scalar occupy two bytes per row.
func RowsOf[Value Element](data []Value, dimensions uint) RowsView {
	if dimensions == 0 {
		panic("usearch: a matrix row cannot be zero dimensions wide")
	}
	scalar := scalarOf[Value]()
	perValue := scalar.dimensionsPerValue()
	if dimensions%perValue != 0 {
		panic(fmt.Sprintf("usearch: %d dimensions do not fill whole %s values", dimensions, scalar))
	}
	valuesPerRow := dimensions / perValue
	if uint(len(data))%valuesPerRow != 0 {
		panic(fmt.Sprintf("usearch: %d values do not divide into rows of %d dimensions", len(data), dimensions))
	}
	var element Value
	return RowsView{
		pointer:    unsafe.Pointer(unsafe.SliceData(data)),
		rows:       uint(len(data)) / valuesPerRow,
		dimensions: dimensions,
		stride:     valuesPerRow * uint(unsafe.Sizeof(element)),
		scalar:     scalar,
	}
}

// StridedRowsOf views a slab whose pitch exceeds its row width - the padded layout a
// memory-mapped dataset arrives in.
//
// Panics when the stride cannot hold a row, or when the rows overrun the slab.
func StridedRowsOf[Value Element](data []Value, rows, dimensions, stride uint) RowsView {
	if dimensions == 0 {
		panic("usearch: a matrix row cannot be zero dimensions wide")
	}
	scalar := scalarOf[Value]()
	perValue := scalar.dimensionsPerValue()
	if dimensions%perValue != 0 {
		panic(fmt.Sprintf("usearch: %d dimensions do not fill whole %s values", dimensions, scalar))
	}
	var element Value
	elementBytes := uint(unsafe.Sizeof(element))
	rowBytes := dimensions / perValue * elementBytes
	if stride < rowBytes {
		panic(fmt.Sprintf("usearch: a stride of %d bytes cannot hold a row of %d", stride, rowBytes))
	}
	if rows != 0 && (rows-1)*stride+rowBytes > uint(len(data))*elementBytes {
		panic(fmt.Sprintf("usearch: %d rows of stride %d overrun a slab of %d values", rows, stride, len(data)))
	}
	return RowsView{
		pointer:    unsafe.Pointer(unsafe.SliceData(data)),
		rows:       rows,
		dimensions: dimensions,
		stride:     stride,
		scalar:     scalar,
	}
}

// Rows reports how many rows the view spans.
func (matrix RowsView) Rows() uint { return matrix.rows }

// Dimensions reports the logical width of one row.
func (matrix RowsView) Dimensions() uint { return matrix.dimensions }

// Stride reports the byte pitch between consecutive row starts.
func (matrix RowsView) Stride() uint { return matrix.stride }

// Scalar reports the scalar type the rows hold.
func (matrix RowsView) Scalar() Scalar { return matrix.scalar }

// rows passes the matrix as the batch view a verb reads; the library checks its width against
// the frozen geometry.
func (call *call) rows(matrix RowsView) C.usearch_rows_view_t {
	call.pinner.Pin(matrix.pointer)
	return C.usearch_rows_view_t{
		data:    matrix.pointer,
		stride:  C.size_t(matrix.stride),
		columns: C.size_t(matrix.dimensions),
		scalar:  C.usearch_scalar_t(matrix.scalar),
	}
}

// endregion: Row Views

// region: Matches

// MatchesView is a ranked run a call wrote into its arena: per query, up to Columns keys and the
// scores positionally paired with them. A search writes distances, nearest first; a late
// interaction writes summed similarities, largest first.
type MatchesView struct {
	raw     C.usearch_matches_view_t
	queries uint
}

// Queries reports how many queries the call resolved.
func (matches MatchesView) Queries() uint { return matches.queries }

// Columns reports the matches the call made room for per query, at most its wanted - the "k" in "kANN".
func (matches MatchesView) Columns() uint { return uint(matches.raw.columns) }

// Query views one query's keys and scores from the arena, trimmed to the count that query
// reported; query must be below Queries.
func (matches MatchesView) Query(query uint) ([]Key, []Distance) {
	columns := matches.Columns()
	count := min(unsafe.Slice((*uint)(unsafe.Pointer(matches.raw.counts)), matches.queries)[query], columns)
	from := query * columns
	keys := unsafe.Slice((*Key)(unsafe.Pointer(matches.raw.keys)), matches.queries*columns)
	scores := unsafe.Slice((*Distance)(unsafe.Pointer(matches.raw.scores)), matches.queries*columns)
	return keys[from : from+count], scores[from : from+count]
}

// MatchesTape is a ragged run of matches a call wrote into its arena, one run per query: every pair
// Multiply's walk scored per left row in traversal order, every key Maxsim scored per query largest
// score first, or every entry Within found per query.
type MatchesTape struct {
	raw      C.usearch_matches_tape_view_t
	queries  uint
	required uint
	written  uint
}

// Queries reports how many left rows the call multiplied, or how many queries it scored.
func (matches MatchesTape) Queries() uint { return matches.queries }

// Required reports the pairs the walks scored across every left row, or for Maxsim the keys scored
// across every query, before the wanted bound cut any; zero for Within.
func (matches MatchesTape) Required() uint { return matches.required }

// Written reports the matches Within wrote across every query; zero for Multiply and Maxsim.
func (matches MatchesTape) Written() uint { return matches.written }

// Query views one query's keys and scores from the arena, a left row's pairs for Multiply; query
// must be below Queries.
func (matches MatchesTape) Query(query uint) ([]Key, []Distance) {
	offsets := unsafe.Slice((*uint)(unsafe.Pointer(matches.raw.offsets)), matches.queries+1)
	keys := unsafe.Slice((*Key)(unsafe.Pointer(matches.raw.keys)), offsets[matches.queries])
	scores := unsafe.Slice((*Distance)(unsafe.Pointer(matches.raw.scores)), offsets[matches.queries])
	return keys[offsets[query]:offsets[query+1]], scores[offsets[query]:offsets[query+1]]
}

// endregion: Matches

// region: Callbacks

// Admitter is the predicate a filtered verb asks about its candidate keys: a [Predicate], a
// [BatchPredicate] or a [CheapBatchPredicate], whose type tells the library what one call costs.
type Admitter interface {
	// admitter passes the handler through handle, which stays in place until the verb is final.
	admitter(handle *cgo.Handle) C.usearch_predicate_t
}

// Predicate admits one candidate key into a filtered search, or passes over it, asked as candidates
// are scored. It runs on the executor's threads together, so it must be safe for concurrent use,
// and it must not panic.
type Predicate func(key Key) bool

// BatchPredicate writes an Admission for every key into admissions, which start as Reject, for
// handlers that pay per call: it is asked once per round a thread settles, about every candidate
// it gathered. Both slices live only until it returns. It runs on the executor's threads together,
// so it must be safe for concurrent use, and it must not panic.
type BatchPredicate func(keys []Key, admissions []Admission)

// CheapBatchPredicate is a [BatchPredicate] asked as candidates are scored, a few keys a call.
type CheapBatchPredicate func(keys []Key, admissions []Admission)

// Admission is whether a predicate admits one candidate key or passes over it.
type Admission int32

const (
	// Reject skips the key; the walk still routes through it.
	Reject Admission = C.usearch_reject_k
	// Admit reports the key among the matches.
	Admit Admission = C.usearch_admit_k
)

// ProgressCallback is called with the units a long call finished so far and the units it
// finishes unless stopped; false stops it wherever the verb can stop. It never runs on two
// threads at once, and it must not panic.
type ProgressCallback func(completed, total uint) bool

//export usearchAdmits
func usearchAdmits(keys *C.usearch_key_t, admissions *C.usearch_admission_t, count C.size_t, state unsafe.Pointer) {
	asked := unsafe.Slice((*Key)(unsafe.Pointer(keys)), count)
	verdicts := unsafe.Slice((*Admission)(unsafe.Pointer(admissions)), count)
	switch predicate := (*cgo.Handle)(state).Value().(type) {
	case Predicate:
		for index, key := range asked {
			if predicate(key) {
				verdicts[index] = Admit
			}
		}
	case BatchPredicate:
		predicate(asked, verdicts)
	case CheapBatchPredicate:
		predicate(asked, verdicts)
	}
}

//export usearchContinues
func usearchContinues(completed, total C.size_t, state unsafe.Pointer) C.usearch_continuation_t {
	if (*cgo.Handle)(state).Value().(ProgressCallback)(uint(completed), uint(total)) {
		return C.usearch_continue_k
	}
	return C.usearch_cancel_k
}

func (predicate Predicate) admitter(handle *cgo.Handle) C.usearch_predicate_t {
	if predicate == nil {
		return C.usearch_predicate_t{}
	}
	return admits(handle, predicate, C.usearch_predicate_cost_cheap_k)
}

func (predicate BatchPredicate) admitter(handle *cgo.Handle) C.usearch_predicate_t {
	if predicate == nil {
		return C.usearch_predicate_t{}
	}
	return admits(handle, predicate, C.usearch_predicate_cost_expensive_k)
}

func (predicate CheapBatchPredicate) admitter(handle *cgo.Handle) C.usearch_predicate_t {
	if predicate == nil {
		return C.usearch_predicate_t{}
	}
	return admits(handle, predicate, C.usearch_predicate_cost_cheap_k)
}

// admits passes handler to the C trampoline through handle, at the cost its type names.
func admits(handle *cgo.Handle, handler any, cost C.usearch_predicate_cost_t) C.usearch_predicate_t {
	*handle = cgo.NewHandle(handler)
	return C.usearch_predicate_t{callback: C.usearch_go_admits(), state: unsafe.Pointer(handle), cost: cost}
}

// predicate passes an optional predicate as the struct a filtering verb takes. It hands the handle
// slot rather than the call through the interface, so the call stays off the heap.
func (call *call) predicate(admitter Admitter) C.usearch_predicate_t {
	if admitter == nil {
		return C.usearch_predicate_t{}
	}
	return admitter.admitter(&call.slots.handles[0])
}

// progress passes an optional handler and its report count as the struct a reporting verb takes.
func (call *call) progress(progress ProgressCallback, steps uint) C.usearch_progress_t {
	if progress == nil {
		return C.usearch_progress_t{}
	}
	return C.usearch_progress_t{
		callback: C.usearch_go_continues(),
		state:    call.callback(1, progress),
		steps:    C.size_t(steps),
	}
}

// callback passes a handler to its C trampoline as an opaque handle in the call's slot rather than
// a Go pointer, since the closure holds Go pointers the cgo rules forbid handing to C.
func (call *call) callback(slot int, handler any) unsafe.Pointer {
	call.slots.handles[slot] = cgo.NewHandle(handler)
	return unsafe.Pointer(&call.slots.handles[slot])
}

// endregion: Callbacks

// region: Executors

// Executor owns the host threads or device streams a fanned-out verb runs on.
//
// An executor owns workers; an index owns entries. Keeping them apart is what lets one pool serve
// several indexes, and what makes the cost of spawning visible at the line that spawns it. Its
// kind is fixed by the constructor that produced it and must match the residency of the handles
// it drives, or the engine reports StatusDevice.
//
// A nil *Executor is the null executor: it owns nothing, and every call handed it runs on the
// calling goroutine - single-threaded on RAM, on the handle's device and default stream on VRAM.
// One call holds an executor at a time, while handles stay freely concurrent - it is the executor
// that serializes. A second call finding it held finishes StatusContended, and nothing queues.
//
// A deferred call on a caller-exclusive pool runs on the pool's thread 0, so polling alone
// finishes it; a single thread or a VRAM executor runs it at submission.
type Executor struct {
	handle C.usearch_executor_t
}

// Residency is where a handle keeps its payload, and where an executor computes and places the
// arenas it makes.
type Residency int32

const (
	// ResidencyRam is volatile host memory, traversed by CPU threads.
	ResidencyRam Residency = C.usearch_residency_ram_k
	// ResidencyVram is accelerator memory, traversed by GPU warps on a chosen device.
	ResidencyVram Residency = C.usearch_residency_vram_k
	// ResidencyNvme is persisted storage, traversed through pluggable I/O engines and sized past RAM.
	ResidencyNvme Residency = C.usearch_residency_nvme_k
)

// String renders the residency as the library spells it.
func (residency Residency) String() string {
	return C.GoString(C.usearch_residency_name(C.usearch_residency_t(residency)))
}

// ResidencyFromName reads a residency back from the spelling String gives it,
// failing with StatusInvalidArgument on a name the library does not spell.
func ResidencyFromName(name string) (Residency, error) {
	var residency C.usearch_residency_t
	if status := C.usearch_residency_from_name(view(name), &residency); status != C.usearch_status_success_k {
		return 0, fail(Status(status), "no residency is spelled \""+name+"\"")
	}
	return Residency(residency), nil
}

// Traversal is how a search reaches the entries it scores.
type Traversal int32

const (
	// TraversalWalk walks the graph, scoring the entries the expansion reaches - approximate, the default.
	TraversalWalk Traversal = C.usearch_traversal_walk_k
	// TraversalScan scores every entry published when the query's scan starts - exact, on RAM and NVMe handles.
	TraversalScan Traversal = C.usearch_traversal_scan_k
)

// String renders the traversal as the library spells it.
func (traversal Traversal) String() string {
	return C.GoString(C.usearch_traversal_name(C.usearch_traversal_t(traversal)))
}

// TraversalFromName reads a traversal back from the spelling String gives it,
// failing with StatusInvalidArgument on a name the library does not spell.
func TraversalFromName(name string) (Traversal, error) {
	var traversal C.usearch_traversal_t
	if status := C.usearch_traversal_from_name(view(name), &traversal); status != C.usearch_status_success_k {
		return 0, fail(Status(status), "no traversal is spelled \""+name+"\"")
	}
	return Traversal(traversal), nil
}

// Mood is what an executor does with its threads or streams once it goes idle.
type Mood int32

const (
	// MoodParked lets idle threads sleep; the next call pays a wake-up.
	MoodParked Mood = C.usearch_mood_parked_k
	// MoodSpinning keeps idle threads hot, trading power for the next call's latency.
	MoodSpinning Mood = C.usearch_mood_spinning_k
)

// String renders the mood as the library spells it.
func (mood Mood) String() string {
	return C.GoString(C.usearch_mood_name(C.usearch_mood_t(mood)))
}

// MoodFromName reads a mood back from the spelling String gives it,
// failing with StatusInvalidArgument on a name the library does not spell.
func MoodFromName(name string) (Mood, error) {
	var mood C.usearch_mood_t
	if status := C.usearch_mood_from_name(view(name), &mood); status != C.usearch_status_success_k {
		return 0, fail(Status(status), "no mood is spelled \""+name+"\"")
	}
	return Mood(mood), nil
}

// LoadPath is how a load from a path read the parts onto the devices.
type LoadPath int32

const (
	// LoadPathNone means the index was never loaded from a path, or lives on RAM or NVMe.
	LoadPathNone LoadPath = C.usearch_load_path_none_k
	// LoadPathHostCopy means the parts passed through host memory, as some file took no cuFile
	// registration.
	LoadPathHostCopy LoadPath = C.usearch_load_path_host_copy_k
	// LoadPathCuFile means every part went straight onto the device through cuFile.
	LoadPathCuFile LoadPath = C.usearch_load_path_cufile_k
)

// String renders the load path as the library spells it.
func (path LoadPath) String() string {
	return C.GoString(C.usearch_load_path_name(C.usearch_load_path_t(path)))
}

// Phase is the kind of work a refine or reshard plan is doing, the same in every residency.
type Phase int32

const (
	// PhaseNone means no plan, or one that has not started.
	PhaseNone Phase = C.usearch_phase_none_k
	// PhaseCopy is a reshard moving keys and values into a target, with the links it keeps.
	PhaseCopy Phase = C.usearch_phase_copy_k
	// PhaseBuild is a reshard on NVMe building blocks of a target's rows in RAM and writing
	// their links.
	PhaseBuild Phase = C.usearch_phase_build_k
	// PhasePartition is a reshard cutting a source into one part per target.
	PhasePartition Phase = C.usearch_phase_partition_k
	// PhaseScatter is a reshard moving each part into its target.
	PhaseScatter Phase = C.usearch_phase_scatter_k
	// PhaseHealth is a round's health check, opening or closing it.
	PhaseHealth Phase = C.usearch_phase_health_k
	// PhaseHeal is a round linking each island with the root's component.
	PhaseHeal Phase = C.usearch_phase_heal_k
	// PhaseHubs is a round drawing the hubs its gather streams entries past.
	PhaseHubs Phase = C.usearch_phase_hubs_k
	// PhaseGather is a round gathering every entry into a neighborhood.
	PhaseGather Phase = C.usearch_phase_gather_k
	// PhaseRestitch is a round re-stitching each neighborhood.
	PhaseRestitch Phase = C.usearch_phase_restitch_k
	// PhaseWrite is a round on NVMe rewriting each tile of links from what its restitch changed.
	PhaseWrite Phase = C.usearch_phase_write_k
	// PhaseDone means the plan ended its round, and nothing more is pending.
	PhaseDone Phase = C.usearch_phase_done_k
)

// String renders the phase as the library spells it.
func (phase Phase) String() string {
	return C.GoString(C.usearch_phase_name(C.usearch_phase_t(phase)))
}

// PhaseFromName reads a phase back from the spelling String gives it,
// failing with StatusInvalidArgument on a name the library does not spell.
func PhaseFromName(name string) (Phase, error) {
	var phase C.usearch_phase_t
	if status := C.usearch_phase_from_name(view(name), &phase); status != C.usearch_status_success_k {
		return 0, fail(Status(status), "no phase is spelled \""+name+"\"")
	}
	return Phase(phase), nil
}

// PhaseProgress is where a plan stands: its phase, that phase's position in the plan, and its own
// count.
type PhaseProgress struct {
	// Phase is the kind of work in hand.
	Phase Phase
	// Ordinal is this phase's position among the ones the plan runs.
	Ordinal uint
	// Completed counts the units of this phase done, entries or islands.
	Completed uint
	// Total counts the units of this phase in all.
	Total uint
}

// CPUClass is a CPU family USearch ships tuned variants for; a RAM or NVMe placement or executor
// may cap dispatch at one, and the zero value sets no ceiling.
type CPUClass int32

const (
	// CPUClassUnknown sets no ceiling, so dispatch may pick any class the machine runs.
	CPUClassUnknown CPUClass = C.usearch_cpu_class_unknown_k
	// CPUClassSerial is the portable floor, with no SIMD assumptions.
	CPUClassSerial CPUClass = C.usearch_cpu_class_serial_k
	// CPUClassMonoversioned is the one class of a monoversioned build.
	CPUClassMonoversioned CPUClass = C.usearch_cpu_class_monoversioned_k
	// CPUClassIntelHaswell is Haswell-class x86: AVX2 and FMA.
	CPUClassIntelHaswell CPUClass = C.usearch_cpu_class_intel_haswell_k
	// CPUClassIntelSierra is Sierra Forest-class x86: AVX-VNNI without AVX-512.
	CPUClassIntelSierra CPUClass = C.usearch_cpu_class_intel_sierra_k
	// CPUClassIntelSkylake is Skylake-class x86: the AVX-512 foundation.
	CPUClassIntelSkylake CPUClass = C.usearch_cpu_class_intel_skylake_k
	// CPUClassIntelIcelake is Icelake-class x86: AVX-512 VNNI and VPOPCNT.
	CPUClassIntelIcelake CPUClass = C.usearch_cpu_class_intel_icelake_k
	// CPUClassAMDGenoa is Genoa-class x86: native bf16 dot products over AVX-512.
	CPUClassAMDGenoa CPUClass = C.usearch_cpu_class_amd_genoa_k
	// CPUClassIntelSapphire is Sapphire Rapids-class x86: AVX-512 FP16 and the AMX tiles.
	CPUClassIntelSapphire CPUClass = C.usearch_cpu_class_intel_sapphire_k
	// CPUClassIntelGranite is Granite Rapids-class x86: the AMX-FP16 tiles.
	CPUClassIntelGranite CPUClass = C.usearch_cpu_class_intel_granite_k
	// CPUClassIntelDiamond is Diamond Rapids-class x86: AVX10.2's FP8 conversions.
	CPUClassIntelDiamond CPUClass = C.usearch_cpu_class_intel_diamond_k
	// CPUClassAWSG2 is Graviton2-class Arm: the bare NEON floor.
	CPUClassAWSG2 CPUClass = C.usearch_cpu_class_aws_g2_k
	// CPUClassAppleM2 is Armv8.6 NEON with dot products and bf16.
	CPUClassAppleM2 CPUClass = C.usearch_cpu_class_apple_m2_k
	// CPUClassAppleM4 is Apple M4-class Arm: the SME outer-product tiles.
	CPUClassAppleM4 CPUClass = C.usearch_cpu_class_apple_m4_k
	// CPUClassAWSG5 is Graviton5-class Arm: Armv9.2.
	CPUClassAWSG5 CPUClass = C.usearch_cpu_class_aws_g5_k
	// CPUClassNVIDIAVera is Vera-class Arm: FP8 dot products.
	CPUClassNVIDIAVera CPUClass = C.usearch_cpu_class_nvidia_vera_k
	// CPUClassAlibabaC930 is Alibaba C930-class RISC-V: RVA23 vectors.
	CPUClassAlibabaC930 CPUClass = C.usearch_cpu_class_alibaba_c930_k
	// CPUClassIBMPower10 is IBM POWER10-class Power: VSX vectors and MMA accumulators.
	CPUClassIBMPower10 CPUClass = C.usearch_cpu_class_ibm_power10_k
	// CPUClassWasm32V128 is WebAssembly with fixed-width simd128.
	CPUClassWasm32V128 CPUClass = C.usearch_cpu_class_wasm32_v128_k
	// CPUClassWasm64Extended is WebAssembly with relaxed SIMD, threads and 64-bit addressing.
	CPUClassWasm64Extended CPUClass = C.usearch_cpu_class_wasm64_extended_k
)

// String renders the class as the library spells it.
func (class CPUClass) String() string {
	return C.GoString(C.usearch_cpu_class_name(C.usearch_cpu_class_t(class)))
}

// CPUClassFromName reads a CPU class back from the spelling String gives it,
// failing with StatusInvalidArgument on a name the library does not spell.
func CPUClassFromName(name string) (CPUClass, error) {
	var class C.usearch_cpu_class_t
	if status := C.usearch_cpu_class_from_name(view(name), &class); status != C.usearch_status_success_k {
		return 0, fail(Status(status), "no CPU class is spelled \""+name+"\"")
	}
	return CPUClass(class), nil
}

// GPUClass is a GPU family USearch ships tuned variants for; a VRAM placement or executor may cap
// dispatch at one, and the zero value sets no ceiling.
type GPUClass int32

const (
	// GPUClassUnknown sets no ceiling, so dispatch may pick any class every named device loads.
	GPUClassUnknown GPUClass = C.usearch_gpu_class_unknown_k
	// GPUClassNVIDIAAmpere is Ampere-class NVIDIA, and every NVIDIA GPU without a class of its own.
	GPUClassNVIDIAAmpere GPUClass = C.usearch_gpu_class_nvidia_ampere_k
	// GPUClassNVIDIAAda is Ada-class NVIDIA: the fp8 conversions.
	GPUClassNVIDIAAda GPUClass = C.usearch_gpu_class_nvidia_ada_k
	// GPUClassNVIDIAHopper is Hopper-class NVIDIA.
	GPUClassNVIDIAHopper GPUClass = C.usearch_gpu_class_nvidia_hopper_k
	// GPUClassNVIDIABlackwell is Blackwell-class NVIDIA.
	GPUClassNVIDIABlackwell GPUClass = C.usearch_gpu_class_nvidia_blackwell_k
	// GPUClassNVIDIABlackwellRTX is Consumer Blackwell NVIDIA.
	GPUClassNVIDIABlackwellRTX GPUClass = C.usearch_gpu_class_nvidia_blackwellrtx_k
	// GPUClassNVIDIARubin is Rubin-class NVIDIA.
	GPUClassNVIDIARubin GPUClass = C.usearch_gpu_class_nvidia_rubin_k
	// GPUClassAMDCDNA3 is CDNA3 AMD: gfx942.
	GPUClassAMDCDNA3 GPUClass = C.usearch_gpu_class_amd_cdna3_k
	// GPUClassAMDCDNA4 is CDNA4 AMD: gfx950.
	GPUClassAMDCDNA4 GPUClass = C.usearch_gpu_class_amd_cdna4_k
	// GPUClassAMDCDNA5 is CDNA5 AMD: gfx1250.
	GPUClassAMDCDNA5 GPUClass = C.usearch_gpu_class_amd_cdna5_k
	// GPUClassAppleMetal7 is Apple7-family GPUs, and every Apple GPU without a class of its own.
	GPUClassAppleMetal7 GPUClass = C.usearch_gpu_class_apple_metal7_k
	// GPUClassAppleMetal9 is Apple9-family GPUs.
	GPUClassAppleMetal9 GPUClass = C.usearch_gpu_class_apple_metal9_k
	// GPUClassAppleMetal10 is Apple10-family GPUs.
	GPUClassAppleMetal10 GPUClass = C.usearch_gpu_class_apple_metal10_k
)

// String renders the class as the library spells it.
func (class GPUClass) String() string {
	return C.GoString(C.usearch_gpu_class_name(C.usearch_gpu_class_t(class)))
}

// GPUClassFromName reads a GPU class back from the spelling String gives it,
// failing with StatusInvalidArgument on a name the library does not spell.
func GPUClassFromName(name string) (GPUClass, error) {
	var class C.usearch_gpu_class_t
	if status := C.usearch_gpu_class_from_name(view(name), &class); status != C.usearch_status_success_k {
		return 0, fail(Status(status), "no GPU class is spelled \""+name+"\"")
	}
	return GPUClass(class), nil
}

// NvmeClass is an I/O engine USearch drives external memory through, newest first; an NVMe executor
// pinned to one tries it and degrades down the ladder the media admit.
type NvmeClass int32

const (
	// NvmeClassUnknown derives the newest engine the host and the media admit.
	NvmeClassUnknown NvmeClass = C.usearch_nvme_class_unknown_k
	// NvmeClassPosix is pread or ReadFile on the calling thread.
	NvmeClassPosix NvmeClass = C.usearch_nvme_class_posix_k
	// NvmeClassLinuxAIO is Linux AIO over io_submit and io_getevents, with O_DIRECT.
	NvmeClassLinuxAIO NvmeClass = C.usearch_nvme_class_linux_aio_k
	// NvmeClassLinuxUring is io_uring reads on files and block devices.
	NvmeClassLinuxUring NvmeClass = C.usearch_nvme_class_linux_uring_k
	// NvmeClassLinuxUringPassthrough is io_uring NVMe passthrough on namespace character devices.
	NvmeClassLinuxUringPassthrough NvmeClass = C.usearch_nvme_class_linux_uring_passthrough_k
	// NvmeClassLinuxSPDK is the SPDK user-space driver, on namespaces named by transport identifiers.
	NvmeClassLinuxSPDK NvmeClass = C.usearch_nvme_class_linux_spdk_k
	// NvmeClassWindowsIORing is I/O Rings, on Windows 11 22H2 and newer.
	NvmeClassWindowsIORing NvmeClass = C.usearch_nvme_class_windows_ioring_k
)

// String renders the class as the library spells it.
func (class NvmeClass) String() string {
	return C.GoString(C.usearch_nvme_class_name(C.usearch_nvme_class_t(class)))
}

// NvmeClassFromName reads an NVMe class back from the spelling String gives it,
// failing with StatusInvalidArgument on a name the library does not spell.
func NvmeClassFromName(name string) (NvmeClass, error) {
	var class C.usearch_nvme_class_t
	if status := C.usearch_nvme_class_from_name(view(name), &class); status != C.usearch_status_success_k {
		return 0, fail(Status(status), "no NVMe class is spelled \""+name+"\"")
	}
	return NvmeClass(class), nil
}

// ExecutorForRam spawns host workers for RAM-resident indexes, interleaved across the compute
// domains local to memoryDomains - the same list the indexes were initialized with, which is what
// keeps compute and memory domains in agreement. An empty list derives them from the machine.
//
// Zero threads takes seven eighths of the cores those domains hold, since full occupancy starves
// the ordered publication path. A cpuClassCeiling intersects with what the machine runs, so a class
// the machine lacks narrows rather than refuses. ScratchLimit caps each thread's scratch, past
// which a call fails with StatusOutOfMemory; zero takes 1 GiB.
func ExecutorForRam(memoryDomains []int32, cpuClassCeiling CPUClass, threads uint, scratchLimit uint) (*Executor, error) {
	var call callWithResult[C.usearch_executor_t]
	arguments := open[C.usearch_executor_for_ram_t](&call)
	*arguments = C.usearch_executor_for_ram_t{
		executor:      call.result,
		outcome:       call.outcome(),
		ram:           call.ram(memoryDomains, cpuClassCeiling),
		threads:       C.size_t(threads),
		scratch_limit: C.size_t(scratchLimit),
	}
	C.usearch_executor_for_ram(arguments)
	return executorOf(block(&call, nil))
}

// ExecutorForNvme spawns host workers for NVMe-resident indexes, one submission ring apiece,
// across the compute domains local to the devices paths names. One ring per thread is what keeps
// submission lock-free: a thread owns its queue outright.
//
// Zero threads derives from those devices' queue capabilities. Every ring submits through
// nvmeClass, opened exactly or refused, and NvmeClassUnknown tries io_uring, then AIO, then
// blocking reads. The first of memoryDomains places the threads and their scratch over the domain
// the devices sit on; an empty list takes the devices' domain. ScratchLimit caps each thread's
// scratch, past which a call fails with StatusOutOfMemory; zero takes 1 GiB. The same threads
// serve RAM-resident indexes too.
func ExecutorForNvme(
	paths []string, memoryDomains []int32, cpuClassCeiling CPUClass, threads uint, nvmeClass NvmeClass, scratchLimit uint,
) (*Executor, error) {
	var call callWithResult[C.usearch_executor_t]
	arguments := open[C.usearch_executor_for_nvme_t](&call)
	*arguments = C.usearch_executor_for_nvme_t{
		executor:      call.result,
		outcome:       call.outcome(),
		nvme:          call.nvme(paths, cpuClassCeiling),
		ram:           call.ram(memoryDomains, CPUClassUnknown),
		threads:       C.size_t(threads),
		nvme_class:    C.usearch_nvme_class_t(nvmeClass),
		scratch_limit: C.size_t(scratchLimit),
	}
	C.usearch_executor_for_nvme(arguments)
	return executorOf(block(&call, nil))
}

// ExecutorForVram opens or adopts one non-blocking stream on every named device, destroyed when the
// executor closes, all of them driven by the one host thread that calls the verbs - which is why it
// takes no thread count. Empty devices drives every listed device of the first vendor with a known
// class.
//
// Streams is empty, or the caller's CUDA or HIP streams, or Metal command queues on Apple, one per
// device in devices order. A CUDA or HIP stream becomes the executor's, destroyed when it closes; a
// Metal queue is retained, the executor releasing its own reference while the caller keeps theirs.
// On an error every stream stays the caller's. Zero scratchLimit lets scratch grow as geometry
// asks, per device.
// A gpuClassCeiling no named device loads is refused with StatusUnsupported.
func ExecutorForVram(
	devices []int32, gpuClassCeiling GPUClass, streams []unsafe.Pointer, scratchLimit uint,
) (*Executor, error) {
	// The library reads one stream per device from any non-nil pointer, so only this side sees a short list.
	if len(streams) != 0 && len(streams) != len(devices) {
		return nil, fail(StatusInvalidArgument, "pass no streams or one per device")
	}
	var call callWithResult[C.usearch_executor_t]
	arguments := open[C.usearch_executor_for_vram_t](&call)
	*arguments = C.usearch_executor_for_vram_t{
		executor:      call.result,
		outcome:       call.outcome(),
		vram:          call.vram(devices, gpuClassCeiling),
		streams:       (*unsafe.Pointer)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(streams)), len(streams))),
		scratch_limit: C.size_t(scratchLimit),
	}
	C.usearch_executor_for_vram(arguments)
	return executorOf(block(&call, nil))
}

// executorOf wraps the handle an executor factory wrote.
func executorOf(handle C.usearch_executor_t, err error) (*Executor, error) {
	if err != nil {
		return nil, err
	}
	return &Executor{handle: handle}, nil
}

// Drain waits until no call holds this executor, then adopts mood until the next call; it only
// waits and never runs a call itself. A failure is the executor's own, such as a lost device,
// never one of a drained call.
func (executor *Executor) Drain(mood Mood) error {
	handle, err := executor.lower()
	if err != nil {
		return err
	}
	var call callWithResult[struct{}]
	arguments := open[C.usearch_executor_drain_t](&call)
	*arguments = C.usearch_executor_drain_t{executor: handle, outcome: call.outcome(), mood: C.usearch_mood_t(mood)}
	C.usearch_executor_drain(arguments)
	_, err = block(&call, nil)
	return err
}

// endregion: Executors

// region: Arenas

// Arena is memory the library writes every output whose length scales with a call into - matches,
// rows, products, monoliths, cursor and stream batches alike - reused so a steady loop allocates once.
//
// Every call given an arena rewinds and refills it, so the views of its previous call and every
// slice they handed out end there, as they do once it is closed; to keep one call's views past the
// next, give that call another arena. An arena serves one call at a time, and a second overlapping
// it fails with StatusContended. Deferred verbs view it as blocking ones do, since Close waits for
// every deferred call naming it. Views over a GPU arena hold device memory, which Describe reports
// and no Go code may read.
type Arena struct {
	handle C.usearch_arena_t
}

// NewArena makes an empty arena where executor computes, host pages or the memory of its GPU,
// decided once; a nil executor makes host pages for calls on the calling goroutine. Nothing is
// allocated until a call writes into it. An arena must not outlive the executor that made it, so
// defer [Executor.Close] before [Arena.Close]; one made with a nil executor lives until it is
// closed. A GPU arena grows and frees on its executor's stream, so it serves only calls on that
// executor, any other or nil finishing StatusDevice.
func NewArena(executor *Executor) (*Arena, error) {
	handle, err := executor.lower()
	if err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_arena_t]
	arguments := open[C.usearch_arena_init_t](&call)
	*arguments = C.usearch_arena_init_t{arena: call.result, executor: handle, outcome: call.outcome()}
	C.usearch_arena_init(arguments)
	arena, err := block(&call, nil)
	if err != nil {
		return nil, err
	}
	return &Arena{handle: arena}, nil
}

// ArenaDescription is where an arena's outputs live, which decides whether Go code may read them.
type ArenaDescription struct {
	// Residency is ResidencyRam for host pages and ResidencyVram for one device's memory, which no Go
	// code may read.
	Residency Residency
	// Device is the ordinal of the GPU holding the outputs on ResidencyVram, and 0 for host pages.
	Device uint
}

// Describe reports where the arena's outputs live, as the executor that made it decided.
func (arena *Arena) Describe() (ArenaDescription, error) {
	handle, err := arena.lower()
	if err != nil {
		return ArenaDescription{}, err
	}
	var call callWithResult[ArenaDescription]
	arguments := open[C.usearch_arena_describe_t](&call)
	*arguments = C.usearch_arena_describe_t{
		arena:       handle,
		outcome:     call.outcome(),
		description: (*C.usearch_arena_description_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_arena_describe(arguments)
	return block(&call, nil)
}

// endregion: Arenas

// region: Calls

// slots is what the library reads and writes for every verb: the outcome, and the handles its
// callbacks reach Go through.
type slots struct {
	outcome C.usearch_outcome_t
	handles [2]cgo.Handle
}

// frame is one call's slots, argument structure and result in a single pinned allocation, whose
// only Go pointers are those the arguments hold, every one of them pinned until the call settles.
type frame[Arguments, Result any] struct {
	slots
	arguments Arguments
	result    Result
}

// call is one verb crossing into the library: the pins that keep every Go pointer its argument
// structure carries in place until the call is final, and the slots it reports through. cgo lets C
// keep a Go pointer only while it is pinned, and pinning is also what keeps every pointee on the
// heap, which the noescape promises rely on.
type call struct {
	pinner runtime.Pinner
	slots  *slots
	// arguments is the frame's argument structure, cleared once the call is final, so a settled
	// frame holds no Go pointer and its outcome may still cross to C unpinned.
	arguments []byte
}

// callWithResult is a call and the slot its result lands in, which the library writes in place.
type callWithResult[Result any] struct {
	call
	result *Result
}

// open pins a fresh frame and hands back its argument structure, which a verb fills and passes to
// C; verbs refuse before opening, so a refusal pins and allocates nothing. The arguments live in
// the frame because cgo moves to the heap whatever a C function that may call back into Go reaches.
func open[Arguments, Result any](opened *callWithResult[Result]) *Arguments {
	frame := new(frame[Arguments, Result])
	opened.pinner.Pin(frame)
	opened.slots, opened.result = &frame.slots, &frame.result
	opened.arguments = unsafe.Slice((*byte)(unsafe.Pointer(&frame.arguments)), unsafe.Sizeof(frame.arguments))
	return &frame.arguments
}

// outcome is where the library reports the call's status.
func (call *call) outcome() *C.usearch_outcome_t { return &call.slots.outcome }

// status reads the outcome word without waiting.
func (call *call) status() Status {
	return Status(C.usearch_outcome_status(call.outcome()))
}

// settle waits until the call is final, releases everything it pinned, and reports its failure, the
// same every time it is asked.
func (call *call) settle() error {
	if call.status() == StatusDeferred {
		C.usearch_outcome_wait(call.outcome())
	}
	for slot, handle := range call.slots.handles {
		if handle != 0 {
			handle.Delete()
			call.slots.handles[slot] = 0
		}
	}
	clear(call.arguments)
	call.pinner.Unpin()
	return check(call.slots.outcome)
}

// block settles the call a verb opened and hands back its result, or the verb's refusal.
func block[Result any](call *callWithResult[Result], refusal error) (Result, error) {
	var none Result
	if refusal != nil {
		return none, refusal
	}
	if err := call.settle(); err != nil {
		return none, err
	}
	return *call.result, nil
}

// configOf reads a nil config as the zero one, by value, so generic verbs allocate nothing for it.
func configOf[Config any](config *Config) Config {
	var options Config
	if config != nil {
		options = *config
	}
	return options
}

// pin keeps a Go object in place until the call is final and hands back its address. It takes no
// type parameter, since a generic call inlined into a generic verb reaches its callee through a
// dictionary, and escape analysis then moves the verb's call to the heap.
func (call *call) pin(pointer unsafe.Pointer) unsafe.Pointer {
	call.pinner.Pin(pointer)
	return pointer
}

// pinSlice pins the backing array of a slice of length elements at data until the call is final,
// and answers nil for an empty slice.
func (call *call) pinSlice(data unsafe.Pointer, length int) unsafe.Pointer {
	if length == 0 {
		return nil
	}
	return call.pin(data)
}

// view passes a string to a call that only reads it before returning.
func view(text string) C.usearch_bytes_view_t {
	view := C.usearch_bytes_view_t{data: (*C.uint8_t)(unsafe.Pointer(unsafe.StringData(text)))}
	// cgo names the length's typedef apart from size_t, though the two are as wide.
	*(*C.size_t)(unsafe.Pointer(&view.length)) = C.size_t(len(text))
	return view
}

// Deferred is a submitted call, which holds its outcome and inputs until it is final, owned by the
// one goroutine that polls or waits on it. Status polls without blocking, Wait blocks until the
// call is final and hands back its result, and Close waits the same way and drops it. A Deferred
// dropped without Wait or Close is released by the collector once the call is final, never
// blocking on it: until then every later collection checks it again.
type Deferred[Result any] struct {
	call    callWithResult[Result]
	cleanup runtime.Cleanup
}

// submitted hands the call a verb opened to its caller, who waits on it or closes it, or drops it
// to the collector; a verb's refusal comes back instead, with nothing submitted.
func submitted[Result any](deferred *Deferred[Result], refusal error) (*Deferred[Result], error) {
	if refusal != nil {
		return nil, refusal
	}
	deferred.cleanup = runtime.AddCleanup(deferred, release, deferred.call.call)
	return deferred, nil
}

// release settles a dropped call once it is final, and otherwise arms itself again for the next
// collection, since the collector's goroutine must not block.
func release(call call) {
	if call.status() == StatusDeferred {
		runtime.AddCleanup(&call, release, call)
		return
	}
	_ = call.settle()
}

// Status reports StatusDeferred until the call is final, then its final status, without blocking.
func (deferred *Deferred[Result]) Status() Status { return deferred.call.status() }

// Wait blocks until the call is final, releases its pins, then hands back its result, the same
// every time it is asked.
func (deferred *Deferred[Result]) Wait() (Result, error) {
	deferred.cleanup.Stop()
	return block(&deferred.call, nil)
}

// Close blocks until the call is final, releases its pins, and reports the call's failure. It is
// idempotent and safe after Wait.
func (deferred *Deferred[Result]) Close() error {
	deferred.cleanup.Stop()
	return deferred.call.settle()
}

// endregion: Calls

// region: Receivers

// receiver is a handle's field, so a Close is seen, and the executor its verbs run on.
type receiver[Handle comparable] struct {
	handle   *Handle
	executor *Executor
}

// lower reports the handles a verb hands across, refusing a closed handle before C dereferences it.
func (receiver receiver[Handle]) lower() (Handle, C.usearch_executor_t, error) {
	var none Handle
	if receiver.handle == nil || *receiver.handle == none {
		return none, nil, fail(StatusUninitialized, "the handle is closed or was never initialized")
	}
	executor, err := receiver.executor.lower()
	return *receiver.handle, executor, err
}

// lowerInto is lower for a verb writing into arena, refusing a closed one too.
func (receiver receiver[Handle]) lowerInto(arena *Arena) (Handle, C.usearch_executor_t, C.usearch_arena_t, error) {
	handle, executor, err := receiver.lower()
	if err != nil {
		return handle, executor, nil, err
	}
	outputs, err := arena.lower()
	return handle, executor, outputs, err
}

// lower reports the handle a verb hands across: nil for the null executor, and a refusal for a
// closed one rather than a silent fall back to the calling goroutine.
func (executor *Executor) lower() (C.usearch_executor_t, error) {
	if executor == nil {
		return nil, nil
	}
	if executor.handle == nil {
		return nil, fail(StatusUninitialized, "the executor is closed or was never spawned")
	}
	return executor.handle, nil
}

// Close waits out the call holding the executor, then releases its threads, rings or streams; every
// arena it made must be closed first. It is idempotent; a closed executor must not be used again.
func (executor *Executor) Close() error {
	if executor == nil || executor.handle == nil {
		return nil
	}
	C.usearch_executor_free(executor.handle)
	executor.handle = nil
	return nil
}

// lower reports the handle a verb hands across: nil for a nil arena, which the call refuses, and a
// refusal for a closed one.
func (arena *Arena) lower() (C.usearch_arena_t, error) {
	if arena == nil {
		return nil, nil
	}
	if arena.handle == nil {
		return nil, fail(StatusUninitialized, "the arena is closed or was never made")
	}
	return arena.handle, nil
}

// Close releases the memory, waiting for every deferred call naming the arena, and ends every view
// over it; it comes before the Close of the executor that made the arena. It is idempotent, and a
// closed arena must not be used again.
func (arena *Arena) Close() error {
	if arena == nil || arena.handle == nil {
		return nil
	}
	C.usearch_arena_free(arena.handle)
	arena.handle = nil
	return nil
}

// endregion: Receivers

// region: Introspection

// Version reports the version of the linked library.
func Version() string { return C.GoString(C.usearch_version_name()) }

// CapabilitiesCompiled lists the hardware capabilities the linked binary was compiled with,
// comma separated.
func CapabilitiesCompiled() string { return C.GoString(C.usearch_capabilities_compiled()) }

// CapabilitiesAvailable lists the hardware capabilities the current machine offers at
// runtime, comma separated.
func CapabilitiesAvailable() string { return C.GoString(C.usearch_capabilities_available()) }

// endregion: Introspection

// region: Helpers

// ints passes a slice of IDs until the call is final.
func (call *call) ints(values []int32) (*C.int32_t, C.size_t) {
	return (*C.int32_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(values)), len(values))), C.size_t(len(values))
}

// ram passes the memory domains of a RAM placement or executor, beside the class ceiling.
func (call *call) ram(memoryDomains []int32, cpuClassCeiling CPUClass) C.usearch_ram_options_t {
	domains, count := call.ints(memoryDomains)
	return C.usearch_ram_options_t{
		memory_domains:       domains,
		memory_domains_count: count,
		cpu_class_ceiling:    C.usearch_cpu_class_t(cpuClassCeiling),
	}
}

// vram passes the devices of a VRAM placement or executor, beside the class ceiling.
func (call *call) vram(devices []int32, gpuClassCeiling GPUClass) C.usearch_vram_options_t {
	ordinals, count := call.ints(devices)
	return C.usearch_vram_options_t{
		devices:           ordinals,
		devices_count:     count,
		gpu_class_ceiling: C.usearch_gpu_class_t(gpuClassCeiling),
	}
}

// ramPlacement passes an optional RAM placement, nil for none, which keeps the handle's own.
func (call *call) ramPlacement(
	memoryDomains []int32, cpuClassCeiling CPUClass, shardCapacity uint,
) *C.usearch_ram_options_t {
	if memoryDomains == nil {
		return nil
	}
	placement := call.ram(memoryDomains, cpuClassCeiling)
	placement.shard_capacity = C.size_t(shardCapacity)
	return (*C.usearch_ram_options_t)(call.pin(unsafe.Pointer(&placement)))
}

// vramPlacement passes an optional VRAM placement, nil for none.
func (call *call) vramPlacement(devices []int32, shardCapacity uint) *C.usearch_vram_options_t {
	if devices == nil {
		return nil
	}
	placement := call.vram(devices, GPUClassUnknown)
	placement.shard_capacity = C.size_t(shardCapacity)
	return (*C.usearch_vram_options_t)(call.pin(unsafe.Pointer(&placement)))
}

// nvmePlacement passes an optional NVMe placement, nil for none.
func (call *call) nvmePlacement(paths []string) *C.usearch_nvme_options_t {
	if paths == nil {
		return nil
	}
	placement := call.nvme(paths, CPUClassUnknown)
	return (*C.usearch_nvme_options_t)(call.pin(unsafe.Pointer(&placement)))
}

// nvme passes a run of device paths, each viewed rather than copied. Go strings are not
// null-terminated and the interface never assumes one, so the bytes cross as they are.
func (call *call) nvme(paths []string, cpuClassCeiling CPUClass) C.usearch_nvme_options_t {
	views := make([]C.usearch_bytes_view_t, len(paths))
	for position, path := range paths {
		views[position] = call.text(path)
	}
	return C.usearch_nvme_options_t{
		paths:             (*C.usearch_bytes_view_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(views)), len(views))),
		paths_count:       C.size_t(len(views)),
		cpu_class_ceiling: C.usearch_cpu_class_t(cpuClassCeiling),
	}
}

// bytes passes a byte slab as the view the C interface reads, with no terminator and no copy.
func (call *call) bytes(bytes []byte) C.usearch_bytes_view_t {
	return C.usearch_bytes_view_t{
		data:   (*C.uint8_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(bytes)), len(bytes))),
		length: C.size_t(len(bytes)),
	}
}

// text views a Go string as the pointer-and-length pair every path argument takes. The string is
// never terminated, by contract, so nothing is copied and nothing is freed.
func (call *call) text(text string) C.usearch_bytes_view_t {
	if len(text) == 0 {
		return C.usearch_bytes_view_t{}
	}
	return C.usearch_bytes_view_t{
		data:   (*C.uint8_t)(call.pin(unsafe.Pointer(unsafe.StringData(text)))),
		length: C.size_t(len(text)),
	}
}

// keys passes a packed key slice as the run the C interface reads, one key per row of the batch
// it accompanies.
func (call *call) keys(keys []Key) C.usearch_keys_view_t {
	return C.usearch_keys_view_t{
		data:   (*C.usearch_key_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(keys)), len(keys))),
		stride: C.size_t(unsafe.Sizeof(Key{})),
	}
}

// points passes places in three dimensions as the f32 rows a sampling verb reads.
func (call *call) points(points [][3]float32) C.usearch_rows_view_t {
	return C.usearch_rows_view_t{
		data:    call.pinSlice(unsafe.Pointer(unsafe.SliceData(points)), len(points)),
		stride:  C.size_t(unsafe.Sizeof([3]float32{})),
		columns: 3,
		scalar:  C.usearch_scalar_t(ScalarF32),
	}
}

// endregion: Helpers

// region: Tapes

// Tape views a ragged batch as Apache Arrow lays one out: elements end to end, and one more
// offset than entries around them, entry i spanning offsets[i] up to offsets[i+1]. An element is
// a byte of a sequence or a point of a shape, three scalars wide.
//
// Like a RowsView, a Tape only points at memory it never owns, and the binding pins both slices
// until each call it crosses is final.
type Tape struct {
	pointer unsafe.Pointer
	offsets unsafe.Pointer
	entries uint
	layout  C.usearch_tape_layout_t
	scalar  Scalar
}

// SequencesTape views bytes end to end around len(offsets) - 1 sequences.
//
// Panics when the offsets overrun the bytes, which names a caller's arithmetic mistake rather
// than a condition the library could report.
func SequencesTape(data []byte, offsets []uint64) Tape {
	return tapeOf(data, offsets, 1)
}

// SequencesOf packs sequences end to end into a fresh Tape, one entry apiece.
func SequencesOf(sequences ...[]byte) Tape {
	offsets := make([]uint64, len(sequences)+1)
	for position, sequence := range sequences {
		offsets[position+1] = offsets[position] + uint64(len(sequence))
	}
	data := make([]byte, 0, offsets[len(sequences)])
	for _, sequence := range sequences {
		data = append(data, sequence...)
	}
	return tapeOf(data, offsets, 1)
}

// ShapesTape views points end to end, x, y and z interleaved, around len(offsets) - 1 shapes
// whose offsets count points.
//
// Panics when the offsets overrun the points.
func ShapesTape[Value Element](data []Value, offsets []uint64) Tape {
	return tapeOf(data, offsets, 3)
}

// ShapesOf packs shapes end to end into a fresh Tape, one entry apiece, each holding its points
// with x, y and z interleaved.
//
// Panics on a shape whose scalars do not divide into whole points.
func ShapesOf[Value Element](shapes ...[]Value) Tape {
	offsets := make([]uint64, len(shapes)+1)
	for position, shape := range shapes {
		if len(shape)%3 != 0 {
			panic(fmt.Sprintf("usearch: %d scalars do not divide into points of three", len(shape)))
		}
		offsets[position+1] = offsets[position] + uint64(len(shape)/3)
	}
	data := make([]Value, 0, offsets[len(shapes)]*3)
	for _, shape := range shapes {
		data = append(data, shape...)
	}
	return tapeOf(data, offsets, 3)
}

// tapeOf views data as elements of width scalars around offsets.
func tapeOf[Value Element](data []Value, offsets []uint64, width uint64) Tape {
	if len(offsets) == 0 {
		panic("usearch: a tape needs one more offset than entries")
	}
	if last := offsets[len(offsets)-1]; last*width > uint64(len(data)) {
		panic(fmt.Sprintf("usearch: offsets ending at %d overrun %d scalars", last, len(data)))
	}
	return Tape{
		pointer: unsafe.Pointer(unsafe.SliceData(data)),
		offsets: unsafe.Pointer(unsafe.SliceData(offsets)),
		entries: uint(len(offsets)) - 1,
		layout:  C.usearch_tape_u64_k,
		scalar:  scalarOf[Value](),
	}
}

// Entries reports how many entries the tape spans.
func (tape Tape) Entries() uint { return tape.entries }

// Scalar reports the scalar type the elements hold.
func (tape Tape) Scalar() Scalar { return tape.scalar }

// tape passes a tape as the batch view a verb reads, with the offsets as wide as it holds them.
func (call *call) tape(tape Tape) C.usearch_tape_view_t {
	view := C.usearch_tape_view_t{data: call.pin(tape.pointer), layout: tape.layout, scalar: C.usearch_scalar_t(tape.scalar)}
	*(*unsafe.Pointer)(unsafe.Pointer(&view.offsets)) = call.pin(tape.offsets)
	return view
}

// TapeView is a ragged batch a call wrote into its arena, readable until the arena serves another
// call: Get's entries sharing a key.
type TapeView struct {
	raw     C.usearch_tape_view_t
	written uint
	width   uint
}

// Written reports how many entries the call wrote; zero when the key is absent.
func (tape TapeView) Written() uint { return tape.written }

// Scalar reports the scalar type the elements were written as.
func (tape TapeView) Scalar() Scalar { return Scalar(tape.raw.scalar) }

// span reports where entry starts and ends, in elements.
func (tape TapeView) span(entry uint) (uint, uint) {
	if tape.raw.layout == C.usearch_tape_u32_k {
		offsets := unsafe.Slice(*(**uint32)(unsafe.Pointer(&tape.raw.offsets)), tape.written+1)
		return uint(offsets[entry]), uint(offsets[entry+1])
	}
	offsets := unsafe.Slice(*(**uint64)(unsafe.Pointer(&tape.raw.offsets)), tape.written+1)
	return uint(offsets[entry]), uint(offsets[entry+1])
}

// TapeEntry views one entry's scalars from the arena - a sequence's bytes, or a shape's points
// with x, y and z interleaved - refused with StatusInvalidArgument when Value is not the scalar
// type they were written as; entry must be below Written.
func TapeEntry[Value Element](tape TapeView, entry uint) ([]Value, error) {
	if scalar := scalarOf[Value](); scalar != tape.Scalar() {
		return nil, fail(StatusInvalidArgument, "elements written as "+tape.Scalar().String()+" cannot be read as "+scalar.String())
	}
	begin, end := tape.span(entry)
	if begin == end {
		return nil, nil
	}
	var element Value
	first := unsafe.Add(tape.raw.data, begin*tape.width*uint(unsafe.Sizeof(element)))
	return unsafe.Slice((*Value)(first), (end-begin)*tape.width), nil
}

// endregion: Tapes

// region: Reports

// TraversalStats is the traversal telemetry one call reports. Every call overwrites it
// rather than accumulating, so a sweep over many batches sums them itself with Accumulate.
type TraversalStats struct {
	// VisitedMembers counts the graph members the call visited.
	VisitedMembers uint
	// ComputedDistances counts the distances the call computed, diversity re-selections
	// included.
	ComputedDistances uint
	// Restarts counts starvation re-seedings - a graph-connectivity health signal rather than
	// a cost counter.
	Restarts uint
	// MissingHints counts hint keys no shard held an entry under; zero unless a search was
	// handed hints.
	MissingHints uint
}

// Accumulate folds another call's telemetry into this one. Binding-only.
func (stats *TraversalStats) Accumulate(other TraversalStats) {
	stats.VisitedMembers += other.VisitedMembers
	stats.ComputedDistances += other.ComputedDistances
	stats.Restarts += other.Restarts
	stats.MissingHints += other.MissingHints
}

// Reachability is one reachability audit: the population it judged, and how much of it a
// search reaches.
type Reachability struct {
	// Audited counts the entries published when the audit started; compare Reachable against
	// this, never a later size.
	Audited uint
	// Reachable counts those a search reaches; the rest no search finds until a refinement
	// re-links them.
	Reachable uint
	// DepthFromRoot counts the depths the walk from the root took to converge.
	DepthFromRoot uint
	// DepthToRoot counts the depths the walk back toward the root took, where a health check
	// walks both ways.
	DepthToRoot uint
	// Unlinked counts the entries no reached entry links to.
	Unlinked uint
	// Underlinked counts the entries linked to by fewer entries than they link to.
	Underlinked uint
	// BeyondMostPopulatedDepth counts the entries deeper than the depth holding the most entries.
	BeyondMostPopulatedDepth uint
	// UnreachedIslands counts the islands no entry outside them links into.
	UnreachedIslands uint
	// StrandedIslands counts the islands with no link out.
	StrandedIslands uint
	// Components counts the components when direction is ignored.
	Components uint
}

// RefineVerdict is what a round concluded, so no caller picks a change rate.
type RefineVerdict int32

const (
	// RefineNone means no round concluded: a report no round filled yet.
	RefineNone RefineVerdict = C.usearch_refine_none_k
	// RefineImproved means the round changed some entry's link set, so another round can help.
	RefineImproved RefineVerdict = C.usearch_refine_improved_k
	// RefineConverged means the round changed no entry's link set.
	RefineConverged RefineVerdict = C.usearch_refine_converged_k
	// RefineStalled means the round lowered nothing while something stands, as full
	// entries leave no slot.
	RefineStalled RefineVerdict = C.usearch_refine_stalled_k
	// RefineStopped means the caller stopped the round.
	RefineStopped RefineVerdict = C.usearch_refine_stopped_k
)

// String renders the verdict as the library spells it.
func (verdict RefineVerdict) String() string {
	return C.GoString(C.usearch_refine_verdict_name(C.usearch_refine_verdict_t(verdict)))
}

// RefineVerdictFromName reads a refine verdict back from the spelling String gives it,
// failing with StatusInvalidArgument on a name the library does not spell.
func RefineVerdictFromName(name string) (RefineVerdict, error) {
	var verdict C.usearch_refine_verdict_t
	if status := C.usearch_refine_verdict_from_name(view(name), &verdict); status != C.usearch_status_success_k {
		return 0, fail(Status(status), "no refine verdict is spelled \""+name+"\"")
	}
	return RefineVerdict(verdict), nil
}

// RefineReport is what one Refine or Reshard did to the graph, so far when its plan stays
// live, and what the whole plan did once its finish lands it.
type RefineReport struct {
	// Reselected counts the entries whose neighbors were selected again, the islands of every
	// heal round included.
	Reselected uint
	// Changed counts those whose neighbor set gained a member it lacked; zero means settled.
	Changed uint
	// IslandsFound counts the entries the first audit found not reached both ways.
	IslandsFound uint
	// IslandsReconnected counts the links written between an island and a reached entry.
	IslandsReconnected uint
	// Neighborhoods counts the neighborhoods the round re-stitched.
	Neighborhoods uint
	// WidestNeighborhood counts the entries of the widest neighborhood.
	WidestNeighborhood uint
	// AddedLinks counts the links the round added.
	AddedLinks uint
	// DroppedLinks counts the links the round dropped.
	DroppedLinks uint
	// Verdict is what the round concluded.
	Verdict RefineVerdict
	// Opening is the health check opening the round.
	Opening Reachability
	// Reachability is the health check closing the heal, as a fresh Health right after would take it.
	Reachability Reachability
}

// stats passes optional telemetry as the pointer the C interface fills.
func (call *call) stats(stats *TraversalStats) *C.usearch_traversal_stats_t {
	if stats == nil {
		return nil
	}
	return (*C.usearch_traversal_stats_t)(call.pin(unsafe.Pointer(stats)))
}

// endregion: Reports

// region: Plans

// livePlan is one long operation kept live across calls: the handle the library opens on the first
// call naming it, and where the last such call stopped. Every modality's refine and reshard plan is
// one, each over its own handle type.
type livePlan[Handle comparable] struct {
	// Progress is where the last call naming the plan stopped, PhaseDone once its rounds ended.
	Progress PhaseProgress
	handle   Handle
}

// lower passes the plan's handle and phase to a call, both nil for no plan, pinned by the call's
// pinner. It takes the pinner rather than being a method of the call, as methods take no type
// parameters and the plan's handle type is one.
func (plan *livePlan[Handle]) lower(pinner *runtime.Pinner) (*Handle, *C.usearch_phase_progress_t) {
	if plan == nil {
		return nil, nil
	}
	pinner.Pin(plan)
	return &plan.handle, (*C.usearch_phase_progress_t)(unsafe.Pointer(&plan.Progress))
}

// take hands the plan's handle to its finish, leaving the plan fresh; a nil plan hands none, which
// the library refuses.
func (plan *livePlan[Handle]) take() Handle {
	var none Handle
	if plan == nil {
		return none
	}
	handle := plan.handle
	plan.handle = none
	return handle
}

// endregion: Plans

// region: Children

// childReceiver is a cursor's or a stream's handle field and the parent handle it was opened on,
// which every pull names, so a Close of either is seen, and the executor a pull runs on, nil for
// the calling goroutine; the child keeps none.
type childReceiver[Child, Parent comparable] struct {
	receiver[Child]
	parent *Parent
}

// lowerInto reports the handles a pull hands across, refusing a closed child, parent, executor or
// arena before C dereferences them.
func (on childReceiver[Child, Parent]) lowerInto(arena *Arena) (Child, Parent, C.usearch_executor_t, C.usearch_arena_t, error) {
	var none Parent
	child, executor, outputs, err := on.receiver.lowerInto(arena)
	if err != nil {
		return child, none, executor, outputs, err
	}
	if *on.parent == none {
		return child, none, executor, outputs, fail(StatusUninitialized, "the handle it was opened on is closed")
	}
	return child, *on.parent, executor, outputs, nil
}

// endregion: Children
