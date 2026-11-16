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
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lusearch_static
#cgo linux LDFLAGS: -lstdc++ -lm
#cgo darwin LDFLAGS: -lc++
#cgo windows LDFLAGS: -lstdc++

// Every verb here retains nothing and takes no callback, so each claims both promises.
#cgo noescape usearch_executor_for_ram
#cgo nocallback usearch_executor_for_ram
#cgo noescape usearch_executor_on_pool
#cgo nocallback usearch_executor_on_pool
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

// The statuses a call ends in. Only Deferred.Status reports StatusDeferred, the one status
// that is not final.
const (
	// StatusSuccess means the call completed and carries no message.
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
	// nothing, so the handle carries whatever it did before the call.
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
// a byte pitch between consecutive rows. It is the shape every batch argument takes.
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

// Query borrows one query's keys and scores from the arena, trimmed to the count that query
// reported; query must be below Queries.
func (matches MatchesView) Query(query uint) ([]Key, []Distance) {
	columns := matches.Columns()
	count := min(unsafe.Slice((*uint)(unsafe.Pointer(matches.raw.counts)), matches.queries)[query], columns)
	from := query * columns
	keys := unsafe.Slice((*Key)(unsafe.Pointer(matches.raw.keys)), matches.queries*columns)
	scores := unsafe.Slice((*Distance)(unsafe.Pointer(matches.raw.scores)), matches.queries*columns)
	return keys[from : from+count], scores[from : from+count]
}

// endregion: Matches

// region: Callbacks

// Predicate admits a candidate key into a filtered search, or passes over it. It runs on the
// executor's threads together, so it must be safe for concurrent use, and it must not panic.
type Predicate func(key Key) bool

// ProgressCallback is called with the units a long call finished so far and the units it
// finishes unless stopped; false stops it wherever the verb can stop. It never runs on two
// threads at once, and it must not panic.
type ProgressCallback func(completed, total uint) bool

//export usearchAdmits
func usearchAdmits(keys *C.usearch_key_t, admissions *C.usearch_admission_t, count C.size_t, state unsafe.Pointer) {
	predicate := (*cgo.Handle)(state).Value().(Predicate)
	verdicts := unsafe.Slice(admissions, count)
	for index, key := range unsafe.Slice(keys, count) {
		verdicts[index] = C.usearch_reject_k
		if predicate(keyFromRaw(key)) {
			verdicts[index] = C.usearch_admit_k
		}
	}
}

//export usearchContinues
func usearchContinues(completed, total C.size_t, state unsafe.Pointer) C.usearch_continuation_t {
	if (*cgo.Handle)(state).Value().(ProgressCallback)(uint(completed), uint(total)) {
		return C.usearch_continue_k
	}
	return C.usearch_cancel_k
}

// predicate lends an optional predicate as the struct a filtering verb takes.
func (call *call) predicate(predicate Predicate) C.usearch_predicate_t {
	if predicate == nil {
		return C.usearch_predicate_t{}
	}
	return C.usearch_predicate_t{callback: C.usearch_go_admits(), state: call.callback(predicate)}
}

// progress lends an optional handler and its report count as the struct a reporting verb takes.
func (call *call) progress(progress ProgressCallback, steps uint) C.usearch_progress_t {
	if progress == nil {
		return C.usearch_progress_t{}
	}
	return C.usearch_progress_t{callback: C.usearch_go_continues(), state: call.callback(progress), steps: C.size_t(steps)}
}

// callback lends a handler to its C trampoline as an opaque handle rather than a Go pointer, since
// the closure holds Go pointers the cgo rules forbid handing to C.
func (call *call) callback(handler any) unsafe.Pointer {
	handle := pin(call, new(cgo.Handle))
	*handle = cgo.NewHandle(handler)
	call.handles = append(call.handles, *handle)
	return unsafe.Pointer(handle)
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

// ExecutorForRam spawns host workers for RAM-resident indexes, interleaved across the compute
// domains local to memoryDomains - the same list the indexes were initialized with, which is what
// keeps compute and memory domains in agreement. An empty list derives them from the machine.
//
// Zero threads takes seven eighths of the cores those domains carry, since full occupancy starves
// the ordered publication path. A cpuClassCeiling intersects with what the machine runs, so a class
// the machine lacks narrows rather than refuses.
func ExecutorForRam(memoryDomains []int32, cpuClassCeiling CPUClass, threads uint) (*Executor, error) {
	lent := open[C.usearch_executor_t]()
	C.usearch_executor_for_ram(&C.usearch_executor_for_ram_t{
		executor: lent.result,
		outcome:  lent.outcome,
		ram:      lent.ram(memoryDomains, cpuClassCeiling),
		threads:  C.size_t(threads),
	})
	return executorOf(block(lent))
}

// ExecutorOnPool creates a RAM executor over a borrowed ForkUnion fu_pool_t, so several
// libraries in one process share one set of threads while each keeps its own scratch. The pool
// must outlive the executor, and keeps its lender's mood.
func ExecutorOnPool(memoryDomains []int32, cpuClassCeiling CPUClass, pool unsafe.Pointer) (*Executor, error) {
	lent := open[C.usearch_executor_t]()
	C.usearch_executor_on_pool(&C.usearch_executor_on_pool_t{
		executor: lent.result,
		outcome:  lent.outcome,
		ram:      lent.ram(memoryDomains, cpuClassCeiling),
		pool:     pool,
	})
	return executorOf(block(lent))
}

// ExecutorForNvme spawns host workers for NVMe-resident indexes, one submission ring apiece,
// across the compute domains local to the devices paths names. One ring per thread is what keeps
// submission lock-free: a thread owns its queue outright.
//
// Zero threads derives from those devices' queue capabilities.
func ExecutorForNvme(paths []string, cpuClassCeiling CPUClass, threads uint) (*Executor, error) {
	lent := open[C.usearch_executor_t]()
	C.usearch_executor_for_nvme(&C.usearch_executor_for_nvme_t{
		executor: lent.result,
		outcome:  lent.outcome,
		nvme:     lent.nvme(paths, cpuClassCeiling),
		threads:  C.size_t(threads),
	})
	return executorOf(block(lent))
}

// ExecutorForVram opens one submission stream on every named device, or borrows the caller's,
// all of them driven by the one host thread that calls the verbs - which is why it takes no thread
// count.
//
// Streams is empty, or holds one live CUDA stream per device in devices order, each outliving the
// executor. Zero scratchLimit lets scratch grow as geometry asks. A gpuClassCeiling no named device
// loads is refused with StatusUnsupported.
func ExecutorForVram(
	devices []int32, gpuClassCeiling GPUClass, streams []unsafe.Pointer, scratchLimit uint,
) (*Executor, error) {
	// The library reads one stream per device from any non-nil pointer, so only this side sees a short list.
	if len(streams) != 0 && len(streams) != len(devices) {
		return nil, fail(StatusShapeMismatch, "pass no streams or one per device")
	}
	lent := open[C.usearch_executor_t]()
	C.usearch_executor_for_vram(&C.usearch_executor_for_vram_t{
		executor:      lent.result,
		outcome:       lent.outcome,
		vram:          lent.vram(devices, gpuClassCeiling),
		streams:       lend(&lent.call, streams),
		scratch_limit: C.size_t(scratchLimit),
	})
	return executorOf(block(lent))
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
	handle, refusal := executor.lower()
	if refusal != nil {
		return refusal
	}
	lent := open[struct{}]()
	C.usearch_executor_drain(&C.usearch_executor_drain_t{executor: handle, outcome: lent.outcome, mood: C.usearch_mood_t(mood)})
	_, err := block(lent)
	return err
}

// Close waits out the call holding the executor, then releases its threads or streams. It is
// idempotent, and a closed executor must not be used again. The host arenas it made stay valid,
// while its GPU arenas must be closed first.
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

// Arena is memory the library writes every output whose length scales with a call into - matches,
// rows, products, monoliths, cursor and stream batches alike - reused so a steady loop allocates once.
//
// Every call given an arena rewinds and refills it, so the views of its previous call and every
// slice they handed out end there, as they do once it is closed; to keep one call's views past
// the next, give that call another arena. An arena serves one call at a time, and a second
// overlapping it fails with StatusContended. Deferred verbs borrow it as blocking ones do,
// since Close waits for every deferred call naming it. Views over a GPU arena hold device
// memory, which Describe reports and no Go code may read.
type Arena struct {
	handle C.usearch_arena_t
}

// NewArena makes an empty arena where executor computes, host pages or the memory of its GPU,
// decided once; a nil executor makes host pages for calls on the calling goroutine. Nothing is
// allocated until a call writes into it, and host pages may outlive their executor. A GPU arena
// grows and frees on its executor's stream, so it serves only calls on that executor, any other or
// nil finishing StatusDevice, and must be closed before it.
func NewArena(executor *Executor) (*Arena, error) {
	handle, refusal := executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_arena_t]()
	C.usearch_arena_init(&C.usearch_arena_init_t{arena: lent.result, executor: handle, outcome: lent.outcome})
	arena, err := block(lent)
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
	lent := open[ArenaDescription]()
	C.usearch_arena_describe(&C.usearch_arena_describe_t{
		arena:       arena.lower(),
		outcome:     lent.outcome,
		description: (*C.usearch_arena_description_t)(unsafe.Pointer(lent.result)),
	})
	return block(lent)
}

// Close releases the memory, waiting for every deferred call naming the arena, and ends every view
// over it. It is idempotent, and a closed arena must not be used again.
func (arena *Arena) Close() error {
	if arena == nil || arena.handle == nil {
		return nil
	}
	C.usearch_arena_free(arena.handle)
	arena.handle = nil
	return nil
}

// lower reports the handle a verb hands across, nil for a nil or closed arena, which the call
// refuses.
func (arena *Arena) lower() C.usearch_arena_t {
	if arena == nil {
		return nil
	}
	return arena.handle
}

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
}

// lent is a call and the slot its result lands in, which the library writes in place.
type lent[Result any] struct {
	call
	result *Result
}

// open starts a call; a verb fills its argument structure, pointing it at the result slot, and
// crosses.
func open[Result any]() *lent[Result] {
	opened := new(lent[Result])
	opened.outcome = pin(&opened.call, new(C.usearch_outcome_t))
	opened.result = pin(&opened.call, new(Result))
	return opened
}

// refuse records a failure the binding caught, so the call never crosses.
func (lent *lent[Result]) refuse(refusal *Error) *lent[Result] {
	lent.refusal = refusal
	return lent
}

// status reads the outcome word without waiting.
func (call *call) status() Status {
	return Status(C.usearch_outcome_status(call.outcome))
}

// settle waits until the call is final, releases everything it lent, and reports its failure.
func (call *call) settle() error {
	if call.status() == StatusDeferred {
		C.usearch_outcome_wait(call.outcome)
	}
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

// block settles a call and hands back its result.
func block[Result any](lent *lent[Result]) (Result, error) {
	if err := lent.settle(); err != nil {
		var empty Result
		return empty, err
	}
	return *lent.result, nil
}

// Deferred is a submitted call, which holds its outcome and inputs until it is final, owned by the
// one goroutine that polls or waits on it. Status polls without blocking, Wait blocks until the
// call is final and hands back its result, and Close waits the same way and drops it. A Deferred
// dropped without Wait or Close is released by the collector once the call is final, never
// blocking on it: until then every later collection checks it again.
//
// A failure the binding catches before the call crosses is reported here too, like any other.
type Deferred[Result any] struct {
	lent    *lent[Result]
	cleanup runtime.Cleanup
}

// submitted hands a call to its caller, who waits on it or closes it, or drops it to the collector.
func submitted[Result any](lent *lent[Result]) *Deferred[Result] {
	deferred := &Deferred[Result]{lent: lent}
	deferred.cleanup = runtime.AddCleanup(deferred, release[Result], lent)
	return deferred
}

// release settles a dropped call once it is final, and otherwise hands it to another dropped
// Deferred for the next collection, since the collector's goroutine must not block.
func release[Result any](lent *lent[Result]) {
	if lent.status() == StatusDeferred {
		submitted(lent)
		return
	}
	_ = lent.settle()
}

// Status reports StatusDeferred until the call is final, then its final status, without blocking.
func (deferred *Deferred[Result]) Status() Status {
	if deferred.lent.refusal != nil {
		return deferred.lent.refusal.Status
	}
	return deferred.lent.status()
}

// Wait blocks until the call is final, releases its pins, then hands back its result, the same
// every time it is asked.
func (deferred *Deferred[Result]) Wait() (Result, error) {
	deferred.cleanup.Stop()
	return block(deferred.lent)
}

// Close blocks until the call is final, releases its pins, and reports the call's failure. It is
// idempotent and safe after Wait.
func (deferred *Deferred[Result]) Close() error {
	deferred.cleanup.Stop()
	return deferred.lent.settle()
}

// endregion: Deferred Calls

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

// ram lends the memory domains of a RAM placement or executor, beside the class ceiling.
func (call *call) ram(memoryDomains []int32, cpuClassCeiling CPUClass) C.usearch_ram_options_t {
	domains, count := call.ints(memoryDomains)
	return C.usearch_ram_options_t{
		memory_domains:       domains,
		memory_domains_count: count,
		cpu_class_ceiling:    C.usearch_cpu_class_t(cpuClassCeiling),
	}
}

// vram lends the devices of a VRAM placement or executor, beside the class ceiling.
func (call *call) vram(devices []int32, gpuClassCeiling GPUClass) C.usearch_vram_options_t {
	ordinals, count := call.ints(devices)
	return C.usearch_vram_options_t{
		devices:           ordinals,
		devices_count:     count,
		gpu_class_ceiling: C.usearch_gpu_class_t(gpuClassCeiling),
	}
}

// nvme lends a run of device paths, each borrowed rather than copied. Go strings are not
// null-terminated and the interface never assumes one, so the bytes cross as they are.
func (call *call) nvme(paths []string, cpuClassCeiling CPUClass) C.usearch_nvme_options_t {
	views := make([]C.usearch_bytes_view_t, len(paths))
	for position, path := range paths {
		views[position] = call.text(path)
	}
	return C.usearch_nvme_options_t{
		paths:             lend(call, views),
		paths_count:       C.size_t(len(views)),
		cpu_class_ceiling: C.usearch_cpu_class_t(cpuClassCeiling),
	}
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

// endregion: Helpers
