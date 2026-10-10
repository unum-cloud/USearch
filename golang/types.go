// Package usearch provides Go bindings for USearch, consumed through its stable C headers.
//
// The package links the pre-instantiated C library, so no index template is instantiated
// here and nothing but the vocabulary below crosses the boundary. Shared types mirror
// include/usearch/types.h; the dense modality mirrors include/usearch/dense_index.h and
// carries a Dense prefix, leaving room for the modalities that follow.
//
//	options := usearch.DefaultDenseOptions()
//	options.Dimensions = 768
//	options.Quantization = usearch.ScalarI8
//
//	index, err := usearch.DenseInitRam(options, nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer index.Close()
//
//	if err := index.Reserve(1_000_000); err != nil {
//	    log.Fatal(err)
//	}
package usearch

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"runtime"
	"runtime/cgo"
	"strings"
	"sync/atomic"
	"unsafe"
)

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lusearch_static
#cgo linux LDFLAGS: -lstdc++ -lm
#cgo darwin LDFLAGS: -lc++
#cgo windows LDFLAGS: -lstdc++

// Every verb here retains nothing and takes no callback, so each claims both promises. A drain
// runs the deferred calls an executor without threads of its own still holds, whose callbacks
// re-enter Go, so it claims `noescape` alone.
#cgo noescape usearch_executor_for_ram
#cgo nocallback usearch_executor_for_ram
#cgo noescape usearch_executor_on_pool
#cgo nocallback usearch_executor_on_pool
#cgo noescape usearch_executor_for_vram
#cgo nocallback usearch_executor_for_vram
#cgo noescape usearch_executor_for_nvme
#cgo nocallback usearch_executor_for_nvme
#cgo noescape usearch_executor_drain

#include "shim.h"
*/
import "C"

// region: Errors

// Status names what went wrong, in a vocabulary shared by the engine and by this binding.
// Validation performed here before a call crosses reports the status the engine would have
// reported for the same fault.
//
// A Status is an error, so errors.Is(err, usearch.StatusReadOnly) matches any failure carrying it.
type Status int32

// The statuses of include/usearch/types.h, numbered alike. Only Deferred.Status reports
// StatusDeferred, the one status that is not final.
const (
	// StatusSuccess means the call completed and carries no message.
	StatusSuccess Status = C.usearch_status_success_k
	// StatusDeferred means the call was submitted deferred and is not final yet.
	StatusDeferred Status = C.usearch_status_deferred_k
	// StatusOutOfMemory means an allocation or reservation could not be satisfied.
	StatusOutOfMemory Status = C.usearch_status_out_of_memory_k
	// StatusShapeMismatch means dimensionality, row counts, or output capacity disagree.
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
	// StatusContended means the call was refused rather than waited for, like a submission to a
	// full deferred table, or an arena still serving another call.
	StatusContended Status = C.usearch_status_contended_k
	// StatusCancelled means a progress handler asked the operation to stop; it unwound and published
	// nothing, so the handle carries whatever it did before the call.
	StatusCancelled Status = C.usearch_status_cancelled_k
	// StatusUnsupported means the verb is declared for every residency but this one does not
	// implement it; a caller picks another residency.
	StatusUnsupported Status = C.usearch_status_unsupported_k
	// StatusInvalidArgument means an argument breaks the verb's contract, like a required value left
	// nil.
	StatusInvalidArgument Status = C.usearch_status_invalid_argument_k
)

// String renders the status as the library spells it. Calls usearch_status_name.
func (status Status) String() string {
	return C.GoString(C.usearch_status_name(C.usearch_status_t(status)))
}

// Error implements the error interface, so a bare Status is an errors.Is target.
func (status Status) Error() string { return status.String() }

// Error is one failed call: a status every caller can branch on, and the detail the status
// cannot carry. Reach the status with errors.Is or errors.As.
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
// carry anything.
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

// The scalar types of include/usearch/types.h, ordered by shrinking dynamic range. The C
// enum reserves the numbering gaps for kinds that have not landed yet.
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
	// ScalarU8 is an 8-bit unsigned integer.
	ScalarU8 Scalar = C.usearch_scalar_u8_k
	// ScalarU4x2 packs two 4-bit unsigned integers per byte.
	ScalarU4x2 Scalar = C.usearch_scalar_u4x2_k
	// ScalarU1x8 packs eight single-bit dimensions per byte.
	ScalarU1x8 Scalar = C.usearch_scalar_u1x8_k
)

// String renders the scalar kind as the library spells it. Calls usearch_scalar_name.
func (scalar Scalar) String() string {
	return C.GoString(C.usearch_scalar_name(C.usearch_scalar_t(scalar)))
}

// dimensionsPerValue is how many logical dimensions one stored value carries: one for the
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

// RowsView borrows a caller-owned slab as rows of equal width: a base pointer, a row count, and
// a byte pitch between consecutive rows. It is the shape every batch argument and every batch
// takes, read as a view where a verb reads rows and written as a span where it lands them.
//
// A RowsView borrows and never owns. It stays valid only as long as the slice it was built
// from, which the binding pins until each call it crosses is final: the collector traces the
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

// lower lends the rows as the batch view a verb reads; the library checks their width against
// the frozen geometry.
func (matrix RowsView) lower(call *call) C.usearch_rows_view_t {
	call.pinner.Pin(matrix.pointer)
	return C.usearch_rows_view_t{
		data:    matrix.pointer,
		stride:  C.size_t(matrix.stride),
		columns: C.size_t(matrix.dimensions),
		scalar:  C.usearch_scalar_t(matrix.scalar),
	}
}

// lowerSpan lends the rows as the span a verb lands rows in, one per row of the matrix.
func (matrix RowsView) lowerSpan(call *call) C.usearch_rows_span_t {
	call.pinner.Pin(matrix.pointer)
	return C.usearch_rows_span_t{
		data:     matrix.pointer,
		stride:   C.size_t(matrix.stride),
		columns:  C.size_t(matrix.dimensions),
		capacity: C.size_t(matrix.rows),
		scalar:   C.usearch_scalar_t(matrix.scalar),
	}
}

// endregion: Row Views

// region: Matches

// MatchesSpan is where a ranked run lands - three slabs sized together, holding a
// queries-by-columns matrix of keys, the scores positionally paired with them, and one match
// count per query. A search lands distances, nearest first; a late interaction lands summed
// similarities, largest first.
//
// A verb allocates one for its own result, or lands its run in one the caller lends, whose
// width may exceed the call's wanted count to land rows inside a wider matrix.
//
// Mirrors usearch_matches_span_t.
type MatchesSpan struct {
	keys    []Key
	scores  []Distance
	counts  []uint
	columns uint
}

// packedMatches sizes the three slabs for a batch of queries, each reporting up to wanted matches.
func packedMatches(queries, wanted uint) MatchesSpan {
	return MatchesSpan{
		keys:    make([]Key, queries*wanted),
		scores:  make([]Distance, queries*wanted),
		counts:  make([]uint, queries),
		columns: wanted,
	}
}

// MatchesSpanOf lends slabs the caller already owns - the path for a sweep that reuses one
// set of buffers. The row width follows from the lengths, so lengths that cannot be read as one
// shape are refused here.
func MatchesSpanOf(keys []Key, scores []Distance, counts []uint) (MatchesSpan, error) {
	if len(counts) == 0 {
		return MatchesSpan{}, fail(StatusShapeMismatch, "a run must hold at least one query")
	}
	if len(keys) != len(scores) {
		return MatchesSpan{}, fail(StatusShapeMismatch, "the key and score slabs are not the same shape")
	}
	if len(keys)%len(counts) != 0 {
		return MatchesSpan{}, fail(StatusShapeMismatch, "the slabs do not divide into one row per query")
	}
	columns := uint(len(keys) / len(counts))
	return MatchesSpan{keys: keys, scores: scores, counts: counts, columns: columns}, nil
}

// Wanted reports the upper bound on matches per query - the width of every row.
func (matches *MatchesSpan) Wanted() uint { return matches.columns }

// Queries reports how many queries the slabs were sized for.
func (matches *MatchesSpan) Queries() uint { return uint(len(matches.counts)) }

// Counts borrows the per-query match counts.
func (matches *MatchesSpan) Counts() []uint { return matches.counts }

// Row borrows one query's keys and scores, trimmed to the count that query reported.
// Nothing is copied, and an out-of-range query yields two empty slices.
func (matches *MatchesSpan) Row(query uint) ([]Key, []Distance) {
	if query >= uint(len(matches.counts)) {
		return nil, nil
	}
	count := min(matches.counts[query], matches.columns)
	from := query * matches.columns
	return matches.keys[from : from+count], matches.scores[from : from+count]
}

// lower lends the slabs to a call keeping up to wanted matches per query, which checks their
// capacity against its query count. The strides have no length, so only this side sees wanted
// overrun the row.
func (matches *MatchesSpan) lower(call *call, wanted uint) (C.usearch_matches_span_t, *Error) {
	if wanted > matches.columns {
		return C.usearch_matches_span_t{}, fail(StatusShapeMismatch, "the wanted count overruns the rows of the span")
	}
	// A zero stride is refused even when nothing lands, so a zero-wide row still strides by one.
	return C.usearch_matches_span_t{
		keys:          (*C.usearch_key_t)(unsafe.Pointer(lend(call, matches.keys))),
		keys_stride:   C.size_t(max(matches.columns, 1) * uint(unsafe.Sizeof(Key{}))),
		scores:        (*C.usearch_distance_t)(unsafe.Pointer(lend(call, matches.scores))),
		scores_stride: C.size_t(max(matches.columns, 1) * uint(unsafe.Sizeof(Distance(0)))),
		counts:        (*C.size_t)(unsafe.Pointer(lend(call, matches.counts))),
		capacity:      C.size_t(len(matches.counts)),
		wanted:        C.size_t(wanted),
	}, nil
}

// endregion: Matches

// region: Callbacks

// Predicate admits a candidate key into a filtered search, or passes over it. It runs on the
// executor's threads together, so it must be safe for concurrent use, and a panic inside it
// rejects the key.
//
// Mirrors usearch_predicate_t.
type Predicate func(key Key) bool

// ProgressCallback is called with the units a long call finished so far and the units it
// finishes unless stopped; false stops it wherever the verb can stop. It never runs on two
// threads at once, and a panic inside it stops the call.
//
// Mirrors usearch_progress_callback_t.
type ProgressCallback func(completed, total uint) bool

//export usearchAdmits
func usearchAdmits(key C.usearch_key_t, state unsafe.Pointer) (admission C.usearch_admission_t) {
	admission = C.usearch_reject_k
	defer func() { _ = recover() }() // a panic unwinding through C frames would skip their unlocks
	if (*cgo.Handle)(state).Value().(Predicate)(keyFromRaw(key)) {
		admission = C.usearch_admit_k
	}
	return admission
}

//export usearchContinues
func usearchContinues(completed, total C.size_t, state unsafe.Pointer) (continuation C.usearch_continuation_t) {
	continuation = C.usearch_cancel_k
	defer func() { _ = recover() }()
	if (*cgo.Handle)(state).Value().(ProgressCallback)(uint(completed), uint(total)) {
		continuation = C.usearch_continue_k
	}
	return continuation
}

// predicate lends an optional predicate as the callback and state pair a filtering verb takes.
func (call *call) predicate(predicate Predicate) (C.usearch_predicate_t, unsafe.Pointer) {
	if predicate == nil {
		return nil, nil
	}
	return C.usearch_go_admits(), call.callback(predicate)
}

// progress lends an optional handler as the callback and state pair a reporting verb takes.
func (call *call) progress(progress ProgressCallback) (C.usearch_progress_callback_t, unsafe.Pointer) {
	if progress == nil {
		return nil, nil
	}
	return C.usearch_go_continues(), call.callback(progress)
}

// callback lends a Go value to a C callback as an opaque handle rather than a Go pointer, since the
// closure it names holds Go pointers the cgo rules forbid handing to C.
func (call *call) callback(value any) unsafe.Pointer {
	handle := pin(call, new(cgo.Handle))
	*handle = cgo.NewHandle(value)
	call.handles = append(call.handles, *handle)
	return unsafe.Pointer(handle)
}

// endregion: Callbacks

// region: Executors

// Executor owns the host threads or device streams a fanned-out verb runs on.
//
// An executor owns workers; an index owns rows. Keeping them apart is what lets one pool serve
// several indexes, and what makes the cost of spawning visible at the line that spawns it. Its
// kind is fixed by the constructor that produced it and must match the residency of the handles
// it drives, or the engine reports StatusDevice.
//
// A nil *Executor is the C interface's null executor: it owns nothing, and every call handed it
// runs on the calling goroutine - single-threaded on RAM, on the handle's device and default
// stream on VRAM. Calls on one executor run in submission order, while handles stay freely
// concurrent - it is the executor that orders.
//
// Every constructor takes a deferredLimit sizing its table of deferred calls, where zero takes the
// default; a deferred submission finding the table full finishes StatusContended.
//
// Mirrors usearch_executor_t.
type Executor struct {
	handle C.usearch_executor_t
}

// Mood is what an executor does with its threads or streams once it goes idle.
//
// Mirrors usearch_mood_t.
type Mood int32

const (
	// MoodParked lets idle threads sleep; the next call pays a wake-up.
	MoodParked Mood = C.usearch_mood_parked_k
	// MoodSpinning keeps idle threads hot, trading power for the next call's latency.
	MoodSpinning Mood = C.usearch_mood_spinning_k
)

// String renders the mood as the library spells it. Calls usearch_mood_name.
func (mood Mood) String() string {
	return C.GoString(C.usearch_mood_name(C.usearch_mood_t(mood)))
}

// ExecutorForRam spawns host workers for RAM-resident indexes, interleaved across the compute
// domains local to memoryDomains - the same list the indexes were initialized with, which is what
// keeps compute and memory domains in agreement. An empty list derives them from the machine.
//
// Zero threads takes seven eighths of the cores those domains carry, since full occupancy starves
// the ordered publication path. Calls usearch_executor_for_ram.
func ExecutorForRam(memoryDomains []int32, threads, deferredLimit uint) (*Executor, error) {
	lent := open[*Executor]()
	executor := new(Executor)
	C.usearch_executor_for_ram(&C.usearch_executor_for_ram_t{
		executor:       pin(&lent.call, &executor.handle),
		outcome:        lent.outcome,
		ram:            lent.ram(memoryDomains),
		threads:        C.size_t(threads),
		deferred_limit: C.size_t(deferredLimit),
	})
	lent.result = func() *Executor { return executor }
	return block(lent)
}

// ExecutorOnPool creates a RAM executor over a borrowed ForkUnion fu_pool_t, so several
// libraries in one process share one set of threads while each keeps its own scratch. The pool
// must outlive the executor, and keeps its lender's mood. Calls usearch_executor_on_pool.
func ExecutorOnPool(memoryDomains []int32, pool unsafe.Pointer, deferredLimit uint) (*Executor, error) {
	lent := open[*Executor]()
	executor := new(Executor)
	C.usearch_executor_on_pool(&C.usearch_executor_on_pool_t{
		executor:       pin(&lent.call, &executor.handle),
		outcome:        lent.outcome,
		ram:            lent.ram(memoryDomains),
		pool:           pool,
		deferred_limit: C.size_t(deferredLimit),
	})
	lent.result = func() *Executor { return executor }
	return block(lent)
}

// ExecutorForNvme spawns host workers for NVMe-resident indexes, one submission ring apiece,
// across the compute domains local to the devices paths names. One ring per thread is what keeps
// submission lock-free: a thread owns its queue outright.
//
// Zero threads derives from those devices' queue capabilities. Calls usearch_executor_for_nvme.
func ExecutorForNvme(paths []string, threads, deferredLimit uint) (*Executor, error) {
	lent := open[*Executor]()
	executor := new(Executor)
	C.usearch_executor_for_nvme(&C.usearch_executor_for_nvme_t{
		executor:       pin(&lent.call, &executor.handle),
		outcome:        lent.outcome,
		nvme:           lent.nvme(paths),
		threads:        C.size_t(threads),
		deferred_limit: C.size_t(deferredLimit),
	})
	lent.result = func() *Executor { return executor }
	return block(lent)
}

// ExecutorForVram opens one submission stream on every named device, or borrows the caller's,
// all of them driven by the one host thread that calls the verbs - which is why it takes no thread
// count.
//
// Streams is empty, or holds one live CUDA stream per device in devices order, each outliving the
// executor. Zero scratchLimit lets scratch grow as geometry asks. Calls usearch_executor_for_vram.
func ExecutorForVram(devices []int32, streams []unsafe.Pointer, scratchLimit, deferredLimit uint) (*Executor, error) {
	// The library reads one stream per device from any non-nil pointer, so only this side sees a short list.
	if len(streams) != 0 && len(streams) != len(devices) {
		return nil, fail(StatusShapeMismatch, "pass no streams or one per device")
	}
	lent := open[*Executor]()
	executor := new(Executor)
	C.usearch_executor_for_vram(&C.usearch_executor_for_vram_t{
		executor:       pin(&lent.call, &executor.handle),
		outcome:        lent.outcome,
		vram:           lent.vram(devices),
		streams:        lend(&lent.call, streams),
		scratch_limit:  C.size_t(scratchLimit),
		deferred_limit: C.size_t(deferredLimit),
	})
	lent.result = func() *Executor { return executor }
	return block(lent)
}

// Drain waits for every call submitted to this executor, then adopts mood whenever it next goes
// idle. A failure is the executor's own, such as a lost device, never one of a drained call.
// Calls usearch_executor_drain.
func (executor *Executor) Drain(mood Mood) error {
	handle, refusal := executor.lower()
	if refusal != nil {
		return refusal
	}
	lent := open[struct{}]()
	C.usearch_executor_drain(&C.usearch_executor_drain_t{executor: handle, outcome: lent.outcome, mood: C.usearch_mood_t(mood)})
	_, err := block(lent)
	return err
}

// Close drains every thread or stream and releases them, waiting for every deferred call naming
// the executor. It is idempotent, and a closed executor must not be used again. Calls
// usearch_executor_free.
func (executor *Executor) Close() error {
	if executor == nil || executor.handle == nil {
		return nil
	}
	C.usearch_executor_free(executor.handle)
	executor.handle = nil
	return nil
}

// lower reports the handle a verb hands across: nil for the null executor, and a refusal for a
// closed one rather than a silent fall back to the calling goroutine.
func (executor *Executor) lower() (C.usearch_executor_t, *Error) {
	if executor == nil {
		return nil, nil
	}
	if executor.handle == nil {
		return nil, fail(StatusInvalidArgument, "the executor is closed or was never spawned")
	}
	return executor.handle, nil
}

// endregion: Executors

// region: Arenas

// Arena holds the outputs only a call can size - Get's rows, Multiply's products, SaveBuffer's
// bytes - so a steady loop allocates once.
//
// The zero value holds nothing; the first call handed it creates the memory where that call's
// executor computes, and every later call rewinds and refills it. A result landed in an arena
// stays valid until the arena serves another call or is closed, so a caller keeping one result
// past the next call hands that call another arena, or none. An arena serves one call at a time;
// a second concurrent one reports StatusContended, so a zero arena is first handed out by one
// goroutine.
//
// Mirrors usearch_arena_t.
type Arena struct {
	handle C.usearch_arena_t
}

// Close releases the memory, waiting for every deferred call naming the arena. It is idempotent,
// and leaves the arena reusable as a zero value. Calls usearch_arena_free.
func (arena *Arena) Close() error {
	if arena == nil || arena.handle == nil {
		return nil
	}
	C.usearch_arena_free(arena.handle)
	arena.handle = nil
	return nil
}

// landing is where a result's outputs live: the caller's arena, or a fresh one the result owns
// and its Close frees. Slices a result borrows from it are invisible to the collector, so reading
// them after that Close, or after a lent arena serves another call, reads freed memory.
type landing struct {
	owned *Arena
}

// land picks the arena a call writes into, creating one for the result when the caller lent none.
func land(arena *Arena) (*Arena, landing) {
	if arena != nil {
		return arena, landing{}
	}
	owned := new(Arena)
	return owned, landing{owned: owned}
}

// Close frees the arena the call created for this result, and does nothing for a lent one.
func (landing landing) Close() error { return landing.owned.Close() }

// endregion: Arenas

// region: Deferred Calls

// call is one verb crossing into the library: the outcome it reports through, and the pins and
// handles that keep every Go pointer and callback its argument structure carries in place until
// the call is final. cgo lets C keep a Go pointer only while it is pinned, and pinning is also
// what keeps every pointee on the heap, which the noescape promises rely on.
//
// The outcome is its own allocation, since the pinner holds Go pointers of its own and anything
// sharing its object would carry them across.
type call struct {
	outcome *C.usearch_outcome_t
	pinner  runtime.Pinner
	handles []cgo.Handle
	// refusal is a fault this binding caught before the call crossed.
	refusal *Error
	// discard frees what the call produced, once no result will hand it out.
	discard func()
}

// lent is a call and the result it produces once it succeeded.
type lent[Result any] struct {
	call
	result func() Result
}

// open starts a call; a verb fills its argument structure, crosses, and names its result.
func open[Result any]() *lent[Result] {
	opened := new(lent[Result])
	opened.outcome = pin(&opened.call, new(C.usearch_outcome_t))
	return opened
}

// refuse records a failure the binding caught, so the call never crosses.
func (lent *lent[Result]) refuse(refusal *Error) *lent[Result] {
	lent.refusal = refusal
	return lent
}

// settle waits until the call is final, releases everything it lent, and reports its failure.
func (call *call) settle() error {
	C.usearch_outcome_wait(call.outcome)
	call.pinner.Unpin()
	for _, handle := range call.handles {
		handle.Delete()
	}
	call.handles = nil
	if call.refusal != nil {
		return call.refusal
	}
	return check(*call.outcome)
}

// drop frees what the call produced, for a result no one will read.
func (call *call) drop() {
	if call.discard != nil {
		call.discard()
		call.discard = nil
	}
}

// block settles a call and hands back what it produced.
func block[Result any](lent *lent[Result]) (Result, error) {
	var produced Result
	if err := lent.settle(); err != nil {
		lent.drop()
		return produced, err
	}
	lent.discard = nil // the result owns what the call produced
	if lent.result != nil {
		produced = lent.result()
	}
	return produced, nil
}

// Deferred is a submitted call, which owns its outcome, inputs, outputs and arena until it is
// final. Status polls without blocking, Wait blocks until the call is final and hands back what it
// produced, and Close waits the same way and frees what no one took. A Deferred dropped without
// Wait or Close keeps its pins, which the runtime reports as a leaked runtime.Pinner.
//
// An executor without threads of its own may hold a deferred call until its next Drain, blocking
// call or Close, so drain it before a Wait there. A failure the binding catches before the call
// crosses is reported here too, like any other.
//
// Binding-only; Wait and Close call usearch_outcome_wait.
type Deferred[Result any] struct {
	lent *lent[Result]
}

// submitted hands a call to its caller, who waits on it or closes it.
func submitted[Result any](lent *lent[Result]) *Deferred[Result] {
	return &Deferred[Result]{lent: lent}
}

// Status reports StatusDeferred until the call is final, then its final status, without blocking.
func (deferred *Deferred[Result]) Status() Status {
	if deferred.lent.refusal != nil {
		return deferred.lent.refusal.Status
	}
	return Status(atomic.LoadInt32((*int32)(unsafe.Pointer(&deferred.lent.outcome.status))))
}

// Wait blocks until the call is final, releases its pins, then hands back what it produced; one
// goroutine waits at a time. Calls usearch_outcome_wait.
func (deferred *Deferred[Result]) Wait() (Result, error) {
	return block(deferred.lent)
}

// Close blocks until the call is final, releases its pins, frees whatever it produced that Wait
// has not handed out, and reports the call's failure. It is idempotent and safe after Wait. Calls
// usearch_outcome_wait.
func (deferred *Deferred[Result]) Close() error {
	err := deferred.lent.settle()
	deferred.lent.drop()
	return err
}

// endregion: Deferred Calls

// region: Introspection

// Version reports the version of the linked library. Calls usearch_version_name.
func Version() string { return C.GoString(C.usearch_version_name()) }

// CapabilitiesCompiled lists the hardware capabilities the linked binary was compiled with,
// comma separated. Calls usearch_capabilities_compiled.
func CapabilitiesCompiled() string { return C.GoString(C.usearch_capabilities_compiled()) }

// CapabilitiesAvailable lists the hardware capabilities the current machine offers at
// runtime, comma separated. Calls usearch_capabilities_available.
func CapabilitiesAvailable() string { return C.GoString(C.usearch_capabilities_available()) }

// endregion: Introspection

// region: Helpers

// A count crosses as a C size_t, a key is reinterpreted rather than copied, and a status word is
// loaded atomically as 32 bits, so each pair must agree in width on every platform this package
// builds for. A constant overflows, and fails the build, the moment one of them diverges.
const _ = uint(unsafe.Sizeof(C.size_t(0))) - uint(unsafe.Sizeof(uint(0)))
const _ = uint(unsafe.Sizeof(uint(0))) - uint(unsafe.Sizeof(C.size_t(0)))
const _ = uint(unsafe.Sizeof(C.usearch_key_t{})) - uint(unsafe.Sizeof(Key{}))
const _ = uint(unsafe.Sizeof(Key{})) - uint(unsafe.Sizeof(C.usearch_key_t{}))
const _ = uint(unsafe.Sizeof(C.usearch_status_t(0))) - uint(unsafe.Sizeof(int32(0)))
const _ = uint(unsafe.Sizeof(int32(0))) - uint(unsafe.Sizeof(C.usearch_status_t(0)))

// pin keeps a Go object in place until the call is final and hands back its address.
func pin[Value any](call *call, value *Value) *Value {
	call.pinner.Pin(value)
	return value
}

// lend pins a slice's backing array until the call is final, and lends an empty slice as nil.
func lend[Value any](call *call, slice []Value) *Value {
	if len(slice) == 0 {
		return nil
	}
	return pin(call, &slice[0])
}

// ints lends a slice of IDs until the call is final.
func (call *call) ints(values []int32) (*C.int32_t, C.size_t) {
	return (*C.int32_t)(unsafe.Pointer(lend(call, values))), C.size_t(len(values))
}

// ram lends the memory domains of a RAM placement or executor.
func (call *call) ram(memoryDomains []int32) C.usearch_ram_options_t {
	domains, count := call.ints(memoryDomains)
	return C.usearch_ram_options_t{memory_domains: domains, memory_domains_count: count}
}

// vram lends the devices of a VRAM placement or executor.
func (call *call) vram(devices []int32) C.usearch_vram_options_t {
	ordinals, count := call.ints(devices)
	return C.usearch_vram_options_t{devices: ordinals, devices_count: count}
}

// nvme lends a run of device paths, each borrowed rather than copied. Go strings are not
// null-terminated and the interface never assumes one, so the bytes cross as they are.
func (call *call) nvme(paths []string) C.usearch_nvme_options_t {
	views := make([]C.usearch_bytes_view_t, len(paths))
	for position, path := range paths {
		views[position] = call.text(path)
	}
	return C.usearch_nvme_options_t{paths: lend(call, views), paths_count: C.size_t(len(views))}
}

// bytes lends a byte slab as the view the C interface reads, with no terminator and no copy.
func (call *call) bytes(bytes []byte) C.usearch_bytes_view_t {
	return C.usearch_bytes_view_t{
		data:   (*C.uint8_t)(unsafe.Pointer(lend(call, bytes))),
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
		data:   (*C.uint8_t)(unsafe.Pointer(pin(call, unsafe.StringData(text)))),
		length: C.size_t(len(text)),
	}
}

// keys lends a packed key slice as the run the C interface reads, one key per row of the batch
// it accompanies.
func (call *call) keys(keys []Key) C.usearch_keys_view_t {
	return C.usearch_keys_view_t{
		data:   (*C.usearch_key_t)(unsafe.Pointer(lend(call, keys))),
		stride: C.size_t(unsafe.Sizeof(Key{})),
	}
}

// keysSpan lends a packed key slice as the run a verb writes, bounded by its length.
func (call *call) keysSpan(keys []Key) C.usearch_keys_span_t {
	return C.usearch_keys_span_t{
		data:     (*C.usearch_key_t)(unsafe.Pointer(lend(call, keys))),
		stride:   C.size_t(unsafe.Sizeof(Key{})),
		capacity: C.size_t(len(keys)),
	}
}

// endregion: Helpers
