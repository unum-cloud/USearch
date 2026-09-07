// Package usearch provides Go bindings for USearch, consumed through its stable C headers.
//
// The package links the pre-instantiated C library, so no index template is instantiated
// here and nothing but the vocabulary below crosses the boundary. Shared types mirror
// include/usearch/types.h; the dense modality mirrors include/usearch/dense_index.h and
// carries a Dense prefix, leaving room for the modalities that follow.
//
//	options := usearch.DefaultDenseOptions()
//	options.Dimensions = 768
//	options.Quantization = usearch.I8
//
//	index, err := usearch.NewDenseIndex(&options)
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
	"errors"
	"fmt"
	"runtime"
	"strings"
	"unsafe"
)

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -L/usr/local/lib -lusearch_c

#include "shim.h"
*/
import "C"

// region: Errors

// Status names what went wrong, in a vocabulary shared by the engine and by this binding.
// Validation performed here before a call crosses reports the status the engine would have
// reported for the same fault.
type Status int32

// The failure identities of include/usearch/types.h, plus the tag for a status this build
// does not know.
const (
	// Success means the call completed and carries no message.
	Success Status = 0
	// OutOfMemory means an allocation or reservation could not be satisfied.
	OutOfMemory Status = 1
	// ShapeMismatch means dimensionality, row counts, or output capacity disagree.
	ShapeMismatch Status = 2
	// UnsupportedLayout means a stride or packing the engine cannot traverse.
	UnsupportedLayout Status = 3
	// UnknownKind means a scalar, metric, or residency tag this build does not know.
	UnknownKind Status = 4
	// ReadOnly means a mutating verb reached a handle that opens its checkpoint in place.
	ReadOnly Status = 5
	// IOFailure means a file could not be opened, read, written, or parsed.
	IOFailure Status = 6
	// DeviceFailure means a device could not be selected, or its memory could not be claimed.
	DeviceFailure Status = 7
	// MalformedInput means a caller-supplied representation could not be decoded.
	MalformedInput Status = 8
	// Internal means the library reported neither success nor a failure.
	Internal Status = 9
	// Overflow means a hard design ceiling was reached, like the 32-bit identity space.
	Overflow Status = 10
	// Uninitialized means the verb needs a prior successful reservation that never happened.
	Uninitialized Status = 11
	// ConfigMismatch means the request contradicts an already-frozen configuration.
	ConfigMismatch Status = 12
	// Contended means a synchronization contract was violated by a concurrent operation;
	// discard any produced output and retry in a quieter phase.
	Contended Status = 13
	// Cancelled means a progress handler asked the operation to stop; it unwound and published
	// nothing, so the handle carries whatever it did before the call.
	Cancelled Status = 14
	// Unrecognized tags a status a newer library reported and this build cannot name. An
	// unknown status is never success.
	Unrecognized Status = -1
)

// String renders the status as its C enumerator's stem.
func (status Status) String() string {
	switch status {
	case Success:
		return "success"
	case OutOfMemory:
		return "out_of_memory"
	case ShapeMismatch:
		return "shape_mismatch"
	case UnsupportedLayout:
		return "unsupported_layout"
	case UnknownKind:
		return "unknown_kind"
	case ReadOnly:
		return "read_only"
	case IOFailure:
		return "io"
	case DeviceFailure:
		return "device"
	case MalformedInput:
		return "malformed_input"
	case Internal:
		return "internal"
	case Overflow:
		return "overflow"
	case Uninitialized:
		return "uninitialized"
	case ConfigMismatch:
		return "config_mismatch"
	case Contended:
		return "contended"
	case Cancelled:
		return "cancelled"
	default:
		return "unrecognized"
	}
}

// Error is one failed call: a status every caller can branch on, and the detail the status
// cannot carry. Reach the status with errors.As.
type Error struct {
	Status  Status
	Message string
}

// Error implements the error interface.
func (failure *Error) Error() string {
	return fmt.Sprintf("usearch: %s: %s", failure.Status, failure.Message)
}

// fail builds the failure this binding reports for a fault it caught before the boundary.
func fail(status Status, message string) error {
	return &Error{Status: status, Message: message}
}

// check turns a populated C error slot into a Go error. Messages are static library storage
// and need no deallocation, so the copy is the only cost.
func check(raw C.usearch_error_t) error {
	if raw.status == C.usearch_success_k {
		return nil
	}
	status := Status(raw.status)
	if status <= Success || status > Cancelled {
		status = Unrecognized
	}
	message := ""
	if raw.message != nil {
		message = C.GoString(raw.message)
	}
	return &Error{Status: status, Message: message}
}

// endregion: Errors

// region: Keys & Distances

// Distance is a score between two entries, in the metric of the index that produced it.
type Distance = float64

// Key is a 128-bit entry identity, most significant byte first, rendered as a lowercase
// dashed UUID.
//
// Keys need not be unique: several entries may share one, and DenseIndex.Count reports how
// many. Integer identities park in the low eight bytes, so KeyFromUint64(42) prints as
// 00000000-0000-0000-0000-00000000002a.
type Key [16]byte

// NilKey is the all-zero key.
var NilKey Key

// KeyFromUint64 parks an integer identity in the low eight bytes, zeroing the rest.
func KeyFromUint64(value uint64) Key {
	var key Key
	for position := 0; position < 8; position++ {
		key[15-position] = byte(value >> (8 * position))
	}
	return key
}

// Uint64 reports the integer identity in the low eight bytes, and false when the high eight
// carry anything.
func (key Key) Uint64() (uint64, bool) {
	for position := 0; position < 8; position++ {
		if key[position] != 0 {
			return 0, false
		}
	}
	var value uint64
	for position := 8; position < 16; position++ {
		value = value<<8 | uint64(key[position])
	}
	return value, true
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
	const malformed = "the key is not thirty-two hexadecimal digits"
	var digits [32]byte
	written := 0
	for _, character := range text {
		if character == '-' {
			continue
		}
		if written == len(digits) {
			return NilKey, fail(MalformedInput, malformed)
		}
		digit, ok := hexadecimalDigit(character)
		if !ok {
			return NilKey, fail(MalformedInput, malformed)
		}
		digits[written] = digit
		written++
	}
	if written != len(digits) {
		return NilKey, fail(MalformedInput, malformed)
	}
	var key Key
	for position := range key {
		key[position] = digits[position*2]<<4 | digits[position*2+1]
	}
	return key, nil
}

func hexadecimalDigit(character rune) (byte, bool) {
	switch {
	case character >= '0' && character <= '9':
		return byte(character - '0'), true
	case character >= 'a' && character <= 'f':
		return byte(character-'a') + 10, true
	case character >= 'A' && character <= 'F':
		return byte(character-'A') + 10, true
	default:
		return 0, false
	}
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

// ScalarKind names a stored or exchanged scalar type by width and encoding. Sub-byte kinds
// pack several dimensions per byte, as their xN suffixes spell.
type ScalarKind int32

// The scalar types of include/usearch/types.h, ordered by shrinking dynamic range. The C
// enum reserves the numbering gaps for kinds that have not landed yet.
const (
	// UnknownScalar tags a type the library could not name.
	UnknownScalar ScalarKind = 0
	// F64 is 64-bit double precision.
	F64 ScalarKind = 1
	// F32 is 32-bit single precision.
	F32 ScalarKind = 2
	// BF16 is the brain floating-point format.
	BF16 ScalarKind = 3
	// F16 is IEEE half precision.
	F16 ScalarKind = 4
	// E5M2 is 8-bit floating point with five exponent bits.
	E5M2 ScalarKind = 5
	// E4M3 is 8-bit floating point with four exponent bits.
	E4M3 ScalarKind = 6
	// E3M2 is 6-bit floating point with three exponent bits.
	E3M2 ScalarKind = 7
	// E2M3 is 6-bit floating point with two exponent bits.
	E2M3 ScalarKind = 8
	// I8 is an 8-bit signed integer.
	I8 ScalarKind = 10
	// I4x2 packs two 4-bit signed integers per byte.
	I4x2 ScalarKind = 11
	// U8 is an 8-bit unsigned integer.
	U8 ScalarKind = 20
	// U4x2 packs two 4-bit unsigned integers per byte.
	U4x2 ScalarKind = 21
	// U1x8 packs eight single-bit dimensions per byte.
	U1x8 ScalarKind = 22
)

// String renders the scalar kind as its C enumerator's stem.
func (kind ScalarKind) String() string {
	switch kind {
	case F64:
		return "f64"
	case F32:
		return "f32"
	case BF16:
		return "bf16"
	case F16:
		return "f16"
	case E5M2:
		return "e5m2"
	case E4M3:
		return "e4m3"
	case E3M2:
		return "e3m2"
	case E2M3:
		return "e2m3"
	case I8:
		return "i8"
	case I4x2:
		return "i4x2"
	case U8:
		return "u8"
	case U4x2:
		return "u4x2"
	case U1x8:
		return "u1x8"
	default:
		return "unknown"
	}
}

// known reports whether this build can name the kind. The C enum reserves numbering gaps for
// kinds that have not landed yet, so an unrecognized tag must never be guessed at.
func (kind ScalarKind) known() bool { return kind.String() != "unknown" }

// dimensionsPerValue is how many logical dimensions one stored value carries: one for the
// byte-and-wider kinds, more for the packed ones.
func (kind ScalarKind) dimensionsPerValue() uint {
	switch kind {
	case I4x2, U4x2:
		return 2
	case U1x8:
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

// Scalar constrains the element types a Matrix may view, ordered by shrinking dynamic range.
type Scalar interface {
	float64 | float32 | BFloat16 | Float16 |
		E5M2Scalar | E4M3Scalar | E3M2Scalar | E2M3Scalar |
		int8 | I4x2Scalar | uint8 | U4x2Scalar | U1x8Scalar
}

// kindOf names the scalar kind of a constrained element type.
func kindOf[Element Scalar]() ScalarKind {
	var probe Element
	switch any(probe).(type) {
	case float64:
		return F64
	case float32:
		return F32
	case BFloat16:
		return BF16
	case Float16:
		return F16
	case E5M2Scalar:
		return E5M2
	case E4M3Scalar:
		return E4M3
	case E3M2Scalar:
		return E3M2
	case E2M3Scalar:
		return E2M3
	case int8:
		return I8
	case I4x2Scalar:
		return I4x2
	case uint8:
		return U8
	case U4x2Scalar:
		return U4x2
	case U1x8Scalar:
		return U1x8
	default:
		return UnknownScalar
	}
}

// endregion: Scalars

// region: Matrix Views

// Matrix borrows a caller-owned slab as rows of equal width: a base pointer, a row count, and
// a byte pitch between consecutive rows. It is the shape every batch argument and every batch
// output takes, and it is written through as readily as read from, so one type serves both
// DenseIndex.Add and DenseIndex.Get.
//
// A Matrix borrows and never owns. It stays valid only as long as the slice it was built
// from, which the binding pins across each call it crosses: the collector traces the base
// pointer, so keeping the view alive keeps the backing array alive.
type Matrix struct {
	pointer    unsafe.Pointer
	rows       uint
	dimensions uint
	stride     uint
	kind       ScalarKind
}

// RowsOf views a packed slab as rows of the given width.
//
// Panics on a zero width or a partial trailing row: both name a caller's arithmetic mistake
// rather than a condition the library could report. Sub-byte kinds count logical dimensions,
// so sixteen dimensions of U1x8Scalar occupy two bytes per row.
func RowsOf[Element Scalar](data []Element, dimensions uint) Matrix {
	if dimensions == 0 {
		panic("usearch: a matrix row cannot be zero dimensions wide")
	}
	kind := kindOf[Element]()
	perValue := kind.dimensionsPerValue()
	if dimensions%perValue != 0 {
		panic(fmt.Sprintf("usearch: %d dimensions do not fill whole %s values", dimensions, kind))
	}
	valuesPerRow := dimensions / perValue
	if uint(len(data))%valuesPerRow != 0 {
		panic(fmt.Sprintf("usearch: %d values do not divide into rows of %d dimensions", len(data), dimensions))
	}
	var element Element
	return Matrix{
		pointer:    unsafe.Pointer(unsafe.SliceData(data)),
		rows:       uint(len(data)) / valuesPerRow,
		dimensions: dimensions,
		stride:     valuesPerRow * uint(unsafe.Sizeof(element)),
		kind:       kind,
	}
}

// StridedRowsOf views a slab whose pitch exceeds its row width - the padded layout a
// memory-mapped dataset arrives in.
//
// Panics when the stride cannot hold a row, or when the rows overrun the slab.
func StridedRowsOf[Element Scalar](data []Element, rows, dimensions, stride uint) Matrix {
	if dimensions == 0 {
		panic("usearch: a matrix row cannot be zero dimensions wide")
	}
	kind := kindOf[Element]()
	perValue := kind.dimensionsPerValue()
	if dimensions%perValue != 0 {
		panic(fmt.Sprintf("usearch: %d dimensions do not fill whole %s values", dimensions, kind))
	}
	var element Element
	elementBytes := uint(unsafe.Sizeof(element))
	rowBytes := dimensions / perValue * elementBytes
	if stride < rowBytes {
		panic(fmt.Sprintf("usearch: a stride of %d bytes cannot hold a row of %d", stride, rowBytes))
	}
	if rows != 0 && (rows-1)*stride+rowBytes > uint(len(data))*elementBytes {
		panic(fmt.Sprintf("usearch: %d rows of stride %d overrun a slab of %d values", rows, stride, len(data)))
	}
	return Matrix{
		pointer:    unsafe.Pointer(unsafe.SliceData(data)),
		rows:       rows,
		dimensions: dimensions,
		stride:     stride,
		kind:       kind,
	}
}

// Rows reports how many rows the view spans.
func (matrix Matrix) Rows() uint { return matrix.rows }

// Dimensions reports the logical width of one row.
func (matrix Matrix) Dimensions() uint { return matrix.dimensions }

// Stride reports the byte pitch between consecutive row starts.
func (matrix Matrix) Stride() uint { return matrix.stride }

// Kind reports the scalar type the rows hold.
func (matrix Matrix) Kind() ScalarKind { return matrix.kind }

// validate checks a view against a frozen geometry before it crosses the boundary. A width of
// zero skips the geometry check, for the calls whose width the caller alone decides.
func (matrix Matrix) validate(dimensions uint) error {
	if matrix.pointer == nil {
		return fail(ShapeMismatch, "the matrix borrows no storage")
	}
	if dimensions != 0 && matrix.dimensions != dimensions {
		return fail(ShapeMismatch, "the matrix rows are not as wide as the index geometry")
	}
	return nil
}

// endregion: Matrix Views

// region: Residency

// Residency says where an index's payload lives and which engine traverses it.
//
// Sealed: RAMResidency, VRAMResidency and NVMeResidency are the only implementations, since the
// method below is unexported. A placement and the residency that reads it are one value, so a
// caller cannot fill an arm the tag does not select - the contradiction the ABI's tagged union
// forbids is unrepresentable here too.
//
// A placement names hardware and nothing else: the workers belong to an Executor, and the
// candidate-pool breadth rides each call. Empty placements derive.
type Residency interface {
	// write stamps the tag and its arm of the options union. Unexported, which is what seals
	// the interface - no other package can add a fourth residency.
	write(options *C.usearch_dense_options_t, pinner *runtime.Pinner) error
	// String names the residency as the C interface spells it.
	String() string
}

// RAMResidency is a volatile-memory index traversed by CPU threads - the concurrent baseline.
type RAMResidency struct {
	// MemoryDomains are the NUMA node ids to place one shard on apiece; empty derives every
	// node the process may use.
	MemoryDomains []int32
}

// VRAMResidency is an accelerator-memory index traversed by GPU warps, sharded across the named
// devices and never replicated.
type VRAMResidency struct {
	// Devices are the accelerator ordinals to shard across; empty derives one free device.
	Devices []int32
}

// NVMeResidency is a persisted index traversed through pluggable I/O engines, sized past RAM.
//
// External-memory graphs are never modified in place, so the handle is read-only and every
// mutating verb fails on it with ReadOnly.
type NVMeResidency struct {
	// Paths are the block devices to span, one shard apiece; empty derives.
	Paths []string
}

func (residency RAMResidency) String() string  { return "ram" }
func (residency VRAMResidency) String() string { return "vram" }
func (residency NVMeResidency) String() string { return "nvme" }

func (residency RAMResidency) write(options *C.usearch_dense_options_t, pinner *runtime.Pinner) error {
	domains, count := pinnedInts(residency.MemoryDomains, pinner)
	C.usearch_go_write_ram(options, C.usearch_ram_options_t{
		memory_domains:       domains,
		memory_domains_count: count,
	})
	return nil
}

func (residency VRAMResidency) write(options *C.usearch_dense_options_t, pinner *runtime.Pinner) error {
	devices, count := pinnedInts(residency.Devices, pinner)
	C.usearch_go_write_vram(options, C.usearch_vram_options_t{
		devices:       devices,
		devices_count: count,
	})
	return nil
}

func (residency NVMeResidency) write(options *C.usearch_dense_options_t, pinner *runtime.Pinner) error {
	paths, count, err := pinnedPaths(residency.Paths, pinner)
	if err != nil {
		return err
	}
	C.usearch_go_write_nvme(options, C.usearch_nvme_options_t{paths: paths, paths_count: count})
	return nil
}

// DefaultResidency is a RAM index across every domain the process may use.
func DefaultResidency() Residency { return RAMResidency{} }

// writeResidency stamps whichever placement the caller named, refusing a nil one where the
// caller can still read which argument was left out.
func writeResidency(residency Residency, options *C.usearch_dense_options_t, pinner *runtime.Pinner) error {
	if residency == nil {
		return fail(MalformedInput, "the residency is nil; name RAMResidency, VRAMResidency, or NVMeResidency")
	}
	return residency.write(options, pinner)
}

// readResidency reports the placement a checkpoint's tag names, with nothing filled in.
//
// A checkpoint records where it was written, never where a reader must keep it, so the
// placement comes back empty rather than reconstructed - the caller names it before init.
func readResidency(options *C.usearch_dense_options_t) (Residency, error) {
	switch options.residency {
	case C.usearch_residency_ram_k:
		return RAMResidency{}, nil
	case C.usearch_residency_vram_k:
		return VRAMResidency{}, nil
	case C.usearch_residency_nvme_k:
		return NVMeResidency{}, nil
	default:
		return nil, fail(UnknownKind, "the checkpoint names an unknown residency")
	}
}

// pinnedInts borrows a slice of ids for the duration of a call, pinning it so the collector
// cannot move it while C reads through the pointer.
func pinnedInts(values []int32, pinner *runtime.Pinner) (*C.int32_t, C.size_t) {
	if len(values) == 0 {
		return nil, 0
	}
	pinner.Pin(&values[0])
	return (*C.int32_t)(unsafe.Pointer(&values[0])), C.size_t(len(values))
}

// pinnedPaths lends a run of device paths, each borrowed rather than copied. Go strings are
// not null-terminated and the interface never assumes one, so the bytes cross as they are.
func pinnedPaths(paths []string, pinner *runtime.Pinner) (*C.usearch_bytes_view_t, C.size_t, error) {
	if len(paths) == 0 {
		return nil, 0, nil
	}
	views := make([]C.usearch_bytes_view_t, len(paths))
	for position, path := range paths {
		if len(path) == 0 {
			return nil, 0, fail(MalformedInput, "a device path must not be empty")
		}
		bytes := unsafe.Slice(unsafe.StringData(path), len(path))
		pinner.Pin(&bytes[0])
		views[position] = borrow(path)
	}
	pinner.Pin(&views[0])
	return &views[0], C.size_t(len(views)), nil
}

// endregion: Residency

// region: Executors

// Executor owns the host threads or device streams a fanned-out verb runs on.
//
// An executor owns workers; an index owns rows. Keeping them apart is what lets one pool serve
// several indexes, and what makes the cost of spawning visible at the line that spawns it. Its
// residency must match the handles it drives - the engine reads a mismatch as a device fault
// rather than deriving one, so there is no nil to pass.
//
// One driver at a time: a second concurrent call on the same executor reports Contended before
// any work begins. Handles stay freely concurrent - it is the executor that serializes.
type Executor struct {
	handle C.usearch_executor_t
}

// NewExecutor spawns the workers one residency needs.
//
// Zero threads derives: seven eighths of the cores for RAM, since full occupancy starves the
// ordered publication path, and the devices' queue capabilities for NVMe. A VRAM executor opens
// one stream per device driven by the calling thread, so it reads no thread count at all.
func NewExecutor(residency Residency, threads uint) (*Executor, error) {
	var pinner runtime.Pinner
	defer pinner.Unpin()

	var failure C.usearch_error_t
	var handle C.usearch_executor_t
	// One factory per residency, as the C interface has them: their arities differ because a
	// VRAM executor drives one stream per device from the calling thread and reads no count.
	switch placement := residency.(type) {
	case RAMResidency:
		domains, count := pinnedInts(placement.MemoryDomains, &pinner)
		options := C.usearch_ram_options_t{memory_domains: domains, memory_domains_count: count}
		handle = C.usearch_executor_for_ram(&options, C.size_t(threads), &failure)
	case VRAMResidency:
		devices, count := pinnedInts(placement.Devices, &pinner)
		options := C.usearch_vram_options_t{devices: devices, devices_count: count}
		handle = C.usearch_executor_for_vram(&options, &failure)
	case NVMeResidency:
		paths, count, err := pinnedPaths(placement.Paths, &pinner)
		if err != nil {
			return nil, err
		}
		options := C.usearch_nvme_options_t{paths: paths, paths_count: count}
		handle = C.usearch_executor_for_nvme(&options, C.size_t(threads), &failure)
	default:
		return nil, fail(MalformedInput, "the residency is nil; name RAMResidency, VRAMResidency, or NVMeResidency")
	}
	if err := check(failure); err != nil {
		return nil, err
	}
	if handle == nil {
		return nil, fail(Internal, "the library returned no handle and no error")
	}
	return &Executor{handle: handle}, nil
}

// Drain waits for every deferred call on this executor to finish.
func (executor *Executor) Drain() error {
	if executor == nil || executor.handle == nil {
		return fail(Uninitialized, "the executor is freed or was never initialized")
	}
	var failure C.usearch_error_t
	C.usearch_executor_drain(executor.handle, &failure)
	return check(failure)
}

// Free drains every thread or stream and releases them. It is idempotent, and a freed executor
// must not be used again.
func (executor *Executor) Free() error {
	if executor == nil || executor.handle == nil {
		return nil
	}
	C.usearch_executor_free(executor.handle)
	executor.handle = nil
	return nil
}

// Close is Free under the name io.Closer names it, so an executor works with defer.
func (executor *Executor) Close() error { return executor.Free() }

// live reports the handle a verb hands across, refusing a missing or freed executor where the
// caller can still read which argument was left out.
func (executor *Executor) live() (C.usearch_executor_t, error) {
	if executor == nil || executor.handle == nil {
		return nil, fail(MalformedInput, "the executor is nil, freed, or was never initialized")
	}
	return executor.handle, nil
}

// endregion: Executors

// region: Introspection

// Version reports the version of the linked library.
func Version() string { return C.GoString(C.usearch_version()) }

// CapabilitiesCompiled lists the hardware capabilities the linked binary was compiled with,
// comma separated.
func CapabilitiesCompiled() string { return C.GoString(C.usearch_capabilities_compiled()) }

// CapabilitiesAvailable lists the hardware capabilities the current machine offers at
// runtime, comma separated.
func CapabilitiesAvailable() string { return C.GoString(C.usearch_capabilities_available()) }

// endregion: Introspection

// region: Helpers

// A count crosses as a C size_t, and a key is reinterpreted rather than copied, so both
// pairs must agree in width on every platform this package builds for. A constant overflows,
// and fails the build, the moment one of them diverges.
const _ = uint(unsafe.Sizeof(C.size_t(0))) - uint(unsafe.Sizeof(uint(0)))
const _ = uint(unsafe.Sizeof(uint(0))) - uint(unsafe.Sizeof(C.size_t(0)))
const _ = uint(unsafe.Sizeof(C.usearch_key_t{})) - uint(unsafe.Sizeof(Key{}))
const _ = uint(unsafe.Sizeof(Key{})) - uint(unsafe.Sizeof(C.usearch_key_t{}))

// borrowBytes lends a byte slab as the view the C interface reads, with no terminator and no
// copy. A byte run and a path are the same shape, since text is bytes that happen to decode.
func borrowBytes(bytes []byte) C.usearch_bytes_view_t {
	if len(bytes) == 0 {
		return C.usearch_bytes_view_t{}
	}
	return C.usearch_bytes_view_t{
		data:   (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(bytes))),
		length: C.size_t(len(bytes)),
	}
}

// lendBytes lends a writable slab as the span a verb fills - the mutable twin of borrowBytes.
func lendBytes(bytes []byte) C.usearch_bytes_span_t {
	if len(bytes) == 0 {
		return C.usearch_bytes_span_t{}
	}
	return C.usearch_bytes_span_t{
		data:   (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(bytes))),
		length: C.size_t(len(bytes)),
	}
}

// borrowKeys lends a key slice as the run the C interface reads, one key per row of the matrix
// it accompanies. It crosses by value, so the pointer never sits in Go memory the engine can
// see and nothing needs pinning.
func borrowKeys(keys []Key) C.usearch_keys_view_t {
	if len(keys) == 0 {
		return C.usearch_keys_view_t{}
	}
	return C.usearch_keys_view_t{
		data:  (*C.usearch_key_t)(unsafe.Pointer(unsafe.SliceData(keys))),
		count: C.size_t(len(keys)),
	}
}

// borrow views a Go string as the pointer-and-length pair every path argument takes. The
// string is never terminated, by contract, so nothing is copied and nothing is freed.
func borrow(text string) C.usearch_bytes_view_t {
	return C.usearch_bytes_view_t{
		data:   (*C.uint8_t)(unsafe.Pointer(unsafe.StringData(text))),
		length: C.size_t(len(text)),
	}
}

// StatusOf reports the status a failure carries, for callers that would rather branch than
// unwrap. A nil error is Success; anything this package did not produce is Unrecognized.
func StatusOf(err error) Status {
	var failure *Error
	if errors.As(err, &failure) {
		return failure.Status
	}
	if err == nil {
		return Success
	}
	return Unrecognized
}

// endregion: Helpers
