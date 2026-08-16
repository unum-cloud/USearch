// Vector Search over equidimensional dense embeddings - the Go twin of
// include/usearch/dense_index.h.
//
// Every verb is a batch over strided matrices, so a memory-mapped dataset reaches the index
// with no reshape and no copy, and a single vector is a one-row matrix. The candidate-pool
// breadth rides each call rather than the handle, which lets a recall sweep run against one
// immutable index with no mutation between points.

package usearch

import (
	"runtime"
	"runtime/cgo"
	"unsafe"
)

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -L/usr/local/lib -lusearch_c

#include "shim.h"
*/
import "C"

// region: Options

// DenseMetric names a distance kind supported out of the box, with SIMD kernels dispatched by
// the machine's capabilities. All of them derive from a single inner-product kernel with
// pre-computed magnitudes; Hamming and Jaccard are bit-level, over the packed sub-byte kinds.
//
// The C layer's custom-callback escape hatch stays unexposed here, as in the Rust binding: a
// caller-supplied metric forfeits the SIMD kernel chains and the magnitude cache both.
type DenseMetric int32

const (
	// UnknownMetric tags a metric the library could not name.
	UnknownMetric DenseMetric = 0
	// Inner is the inner product.
	Inner DenseMetric = 1
	// Angular is the cosine distance.
	Angular DenseMetric = 2
	// Euclidean is the squared euclidean distance.
	Euclidean DenseMetric = 3
	// Hamming counts differing bits.
	Hamming DenseMetric = 4
	// Jaccard is the bit-level Jaccard distance.
	Jaccard DenseMetric = 5
)

// String renders the metric as its C enumerator's stem.
func (metric DenseMetric) String() string {
	switch metric {
	case Inner:
		return "inner"
	case Angular:
		return "angular"
	case Euclidean:
		return "euclidean"
	case Hamming:
		return "hamming"
	case Jaccard:
		return "jaccard"
	default:
		return "unknown"
	}
}

// DimensionPadding says whether stored rows round up to the SIMD step quantum.
type DimensionPadding uint8

const (
	// NaturalDimensions keeps rows at their natural width.
	NaturalDimensions DimensionPadding = iota
	// SimdQuantumDimensions gives cache-line-aligned rows and tail-free unrolled kernels, at
	// slightly larger arenas.
	SimdQuantumDimensions
)

// DenseOptions holds the construction and loading options of a dense index.
type DenseOptions struct {
	// Dimensions is the width of the indexed vectors; zero defers the geometry to a later
	// Load, which reads it off the checkpoint.
	Dimensions uint
	// DimensionPadding is a property of the stored layout, so it applies whichever residency
	// traverses it.
	DimensionPadding DimensionPadding
	// Metric is the distance kind used between vectors.
	Metric DenseMetric
	// Quantization is the scalar type vectors are stored and quantized into.
	Quantization ScalarKind
	// Connectivity bounds the connections per node in the graph; zero derives.
	Connectivity uint
	// Residency says where the payload lives, and what loading a checkpoint means.
	Residency Residency
}

// DefaultDenseOptions is an angular BF16 index in RAM, with every optional field derived.
func DefaultDenseOptions() DenseOptions {
	return DenseOptions{
		Metric:       Angular,
		Quantization: BF16,
		Residency:    DefaultResidency(),
	}
}

// toRaw lowers the options, leaving the custom-metric callback null.
func (options *DenseOptions) toRaw() (C.usearch_dense_options_t, error) {
	var raw C.usearch_dense_options_t
	raw.metric = C.usearch_dense_metric_t(options.Metric)
	raw.quantization = C.usearch_scalar_t(options.Quantization)
	raw.dimensions = C.size_t(options.Dimensions)
	raw.pad_dimensions = C.bool(options.DimensionPadding == SimdQuantumDimensions)
	raw.connectivity = C.size_t(options.Connectivity)
	if err := options.Residency.write(&raw); err != nil {
		return raw, err
	}
	return raw, nil
}

// denseOptionsFromRaw lifts the options a checkpoint's headers describe.
func denseOptionsFromRaw(raw *C.usearch_dense_options_t) (DenseOptions, error) {
	options := DenseOptions{
		Dimensions:   uint(raw.dimensions),
		Metric:       DenseMetric(raw.metric),
		Quantization: ScalarKind(raw.quantization),
		Connectivity: uint(raw.connectivity),
	}
	if bool(raw.pad_dimensions) {
		options.DimensionPadding = SimdQuantumDimensions
	}
	if options.Metric < Inner || options.Metric > Jaccard {
		return options, fail(UnknownKind, "the checkpoint names an unknown metric")
	}
	if !options.Quantization.known() {
		return options, fail(UnknownKind, "the checkpoint names an unknown scalar kind")
	}
	residency, err := readResidency(raw)
	if err != nil {
		return options, err
	}
	options.Residency = residency
	return options, nil
}

// reload lowers the options and lifts them straight back - the marshalling round trip, and the
// only thing that exercises the union accessors in shim.c. A test file cannot name cgo types,
// so the round trip is reachable only from here.
func (options DenseOptions) reload() (DenseOptions, error) {
	raw, err := options.toRaw()
	if err != nil {
		return DenseOptions{}, err
	}
	return denseOptionsFromRaw(&raw)
}

// reloadTagged is reload with the three enum tags overwritten by whatever a newer library
// might have written, so the lift path can be shown to reject them rather than guess.
func (options DenseOptions) reloadTagged(metric, quantization, residency int32) (DenseOptions, error) {
	raw, err := options.toRaw()
	if err != nil {
		return DenseOptions{}, err
	}
	raw.metric = C.usearch_dense_metric_t(metric)
	raw.quantization = C.usearch_scalar_t(quantization)
	raw.residency = C.usearch_residency_t(residency)
	return denseOptionsFromRaw(&raw)
}

// endregion: Options

// region: Per-Call Settings & Telemetry

// DenseStats is the traversal telemetry one call reports. Every call overwrites it rather
// than accumulating, so a sweep over many batches sums them itself with Accumulate.
type DenseStats struct {
	// VisitedMembers counts the graph members the call visited.
	VisitedMembers uint
	// ComputedDistances counts the distances the call computed, diversity re-selections
	// included.
	ComputedDistances uint
	// Restarts counts starvation re-seedings - a graph-connectivity health signal rather than
	// a cost counter.
	Restarts uint
}

// Accumulate folds another call's telemetry into this one.
func (stats *DenseStats) Accumulate(other DenseStats) {
	stats.VisitedMembers += other.VisitedMembers
	stats.ComputedDistances += other.ComputedDistances
	stats.Restarts += other.Restarts
}

// report copies the telemetry out, when the caller asked for any.
func report(sink *DenseStats, raw C.usearch_dense_traversal_stats_t) {
	if sink == nil {
		return
	}
	sink.VisitedMembers = uint(raw.visited_members)
	sink.ComputedDistances = uint(raw.computed_distances)
	sink.Restarts = uint(raw.restarts)
}

// endregion: Per-Call Settings & Telemetry

// region: Results

// DenseOutputs is where a batched search writes its results - three slabs sized together,
// holding a queries-by-wanted matrix of keys, the distances positionally paired with them,
// and one match count per query.
//
// The wanted count is the "k" in kANN and the row width of both matrices, so it is fixed by
// the constructor and cannot be restated afterwards.
type DenseOutputs struct {
	keys      []Key
	distances []Distance
	counts    []uint
	wanted    uint
}

// PackedDenseOutputs sizes the three slabs for a batch of queries, each reporting up to
// wanted matches.
func PackedDenseOutputs(queries, wanted uint) DenseOutputs {
	return DenseOutputs{
		keys:      make([]Key, queries*wanted),
		distances: make([]Distance, queries*wanted),
		counts:    make([]uint, queries),
		wanted:    wanted,
	}
}

// DenseOutputsOf adopts slabs the caller already owns - the allocation-free path for a sweep
// that reuses one set of buffers. The three lengths are checked here, once.
func DenseOutputsOf(keys []Key, distances []Distance, counts []uint, wanted uint) (DenseOutputs, error) {
	if wanted == 0 {
		return DenseOutputs{}, fail(ShapeMismatch, "the outputs reserve no room for a single match")
	}
	if uint(len(keys)) < uint(len(counts))*wanted || uint(len(distances)) < uint(len(counts))*wanted {
		return DenseOutputs{}, fail(ShapeMismatch, "the matrices are narrower than the wanted count")
	}
	return DenseOutputs{keys: keys, distances: distances, counts: counts, wanted: wanted}, nil
}

// Wanted reports the upper bound on matches per query.
func (outputs *DenseOutputs) Wanted() uint { return outputs.wanted }

// Queries reports how many queries the slabs were sized for.
func (outputs *DenseOutputs) Queries() uint { return uint(len(outputs.counts)) }

// Counts borrows the per-query match counts.
func (outputs *DenseOutputs) Counts() []uint { return outputs.counts }

// MatchesOf borrows one query's keys and distances, trimmed to the count that query reported.
// Nothing is copied, and an out-of-range query yields two empty slices.
func (outputs *DenseOutputs) MatchesOf(query uint) ([]Key, []Distance) {
	if query >= uint(len(outputs.counts)) {
		return nil, nil
	}
	// The library reports the true extent, which may exceed the room reserved for it.
	count := min(outputs.counts[query], outputs.wanted)
	from := query * outputs.wanted
	return outputs.keys[from : from+count], outputs.distances[from : from+count]
}

// validate checks the slabs hold a batch of the given size. The constructors already fixed the
// width, so only the query count is open.
func (outputs *DenseOutputs) validate(queries uint) error {
	if outputs.wanted == 0 {
		return fail(ShapeMismatch, "the outputs reserve no room for a single match")
	}
	if uint(len(outputs.counts)) < queries {
		return fail(ShapeMismatch, "the outputs hold fewer rows than there are queries")
	}
	return nil
}

// endregion: Results

// region: Index

// DenseIndex is a Vector Search Index over equidimensional dense representations.
//
// A handle may be shared across threads, so no method here takes a lock. On a RAM handle,
// insertions and searches overlap freely: searches read only entries published behind the
// visibility watermark and never write to the graph, while insertions synchronize through
// per-node locks and publish in order. Serialization is the one exception, demanding a
// mutation-free window, though concurrent readers keep searching throughout.
type DenseIndex struct {
	handle C.usearch_dense_index_t
}

// NewDenseIndex initializes an index for the residency its options select.
func NewDenseIndex(options *DenseOptions) (*DenseIndex, error) {
	raw, err := options.toRaw()
	if err != nil {
		return nil, err
	}
	var failure C.usearch_error_t
	handle := C.usearch_dense_init(&raw, &failure)
	if err := check(failure); err != nil {
		return nil, err
	}
	if handle == nil {
		return nil, fail(Internal, "the library returned no handle and no error")
	}
	return &DenseIndex{handle: handle}, nil
}

// DenseMetadata reads a checkpoint's geometry and metric metadata without loading it - from
// the manifest's summary when the path names one, so no shard is opened.
func DenseMetadata(path string) (DenseOptions, error) {
	var raw C.usearch_dense_options_t
	var failure C.usearch_error_t
	C.usearch_dense_metadata(borrow(path), &raw, &failure)
	runtime.KeepAlive(path)
	if err := check(failure); err != nil {
		return DenseOptions{}, err
	}
	return denseOptionsFromRaw(&raw)
}

// DenseMetadataBuffer reads geometry and metric metadata from a serialized monolith buffer.
func DenseMetadataBuffer(buffer []byte) (DenseOptions, error) {
	if len(buffer) == 0 {
		return DenseOptions{}, fail(MalformedInput, "the buffer is empty")
	}
	var raw C.usearch_dense_options_t
	var failure C.usearch_error_t
	C.usearch_dense_metadata_buffer(unsafe.Pointer(unsafe.SliceData(buffer)), C.size_t(len(buffer)), &raw, &failure)
	runtime.KeepAlive(buffer)
	if err := check(failure); err != nil {
		return DenseOptions{}, err
	}
	return denseOptionsFromRaw(&raw)
}

// Free releases the index. It is idempotent, and a freed handle must not be used again.
func (index *DenseIndex) Free() error {
	if index.handle == nil {
		return nil
	}
	C.usearch_dense_free(index.handle)
	index.handle = nil
	return nil
}

// Close implements io.Closer by releasing the index.
func (index *DenseIndex) Close() error { return index.Free() }

// Reserve claims capacity for a number of incoming vectors ahead of insertion. The capacity
// is the desired total, including the current size.
func (index *DenseIndex) Reserve(capacity uint) error {
	if err := index.live(); err != nil {
		return err
	}
	var failure C.usearch_error_t
	C.usearch_dense_reserve(index.handle, C.size_t(capacity), &failure)
	return check(failure)
}

// Clear erases all entries, keeping the reserved capacity and the frozen geometry.
func (index *DenseIndex) Clear() error {
	if err := index.live(); err != nil {
		return err
	}
	var failure C.usearch_error_t
	C.usearch_dense_clear(index.handle, &failure)
	return check(failure)
}

// Size reports the number of stored entries.
func (index *DenseIndex) Size() (uint, error) {
	if err := index.live(); err != nil {
		return 0, err
	}
	var failure C.usearch_error_t
	return measured(C.usearch_dense_size(index.handle, &failure), failure)
}

// Capacity reports the reserved capacity in entries.
func (index *DenseIndex) Capacity() (uint, error) {
	if err := index.live(); err != nil {
		return 0, err
	}
	var failure C.usearch_error_t
	return measured(C.usearch_dense_capacity(index.handle, &failure), failure)
}

// Dimensions reports the frozen width of the stored vectors.
func (index *DenseIndex) Dimensions() (uint, error) {
	if err := index.live(); err != nil {
		return 0, err
	}
	var failure C.usearch_error_t
	return measured(C.usearch_dense_dimensions(index.handle, &failure), failure)
}

// Connectivity reports the frozen connections-per-node bound of the graph.
func (index *DenseIndex) Connectivity() (uint, error) {
	if err := index.live(); err != nil {
		return 0, err
	}
	var failure C.usearch_error_t
	return measured(C.usearch_dense_connectivity(index.handle, &failure), failure)
}

// MemoryUsage reports the memory the index occupies, in bytes.
func (index *DenseIndex) MemoryUsage() (uint, error) {
	if err := index.live(); err != nil {
		return 0, err
	}
	var failure C.usearch_error_t
	return measured(C.usearch_dense_memory_usage(index.handle, &failure), failure)
}

// SerializedLength reports the expected serialized size in bytes: every shard file, plus the
// manifest when the index spans more than one.
func (index *DenseIndex) SerializedLength() (uint, error) {
	if err := index.live(); err != nil {
		return 0, err
	}
	var failure C.usearch_error_t
	return measured(C.usearch_dense_serialized_length(index.handle, &failure), failure)
}

// Capabilities reports the capability codename the index dispatched to on this machine.
func (index *DenseIndex) Capabilities() (string, error) {
	if err := index.live(); err != nil {
		return "", err
	}
	var failure C.usearch_error_t
	raw := C.usearch_dense_capabilities(index.handle, &failure)
	if err := check(failure); err != nil {
		return "", err
	}
	if raw == nil {
		return "", fail(Internal, "the library reported no capabilities and no error")
	}
	return C.GoString(raw), nil
}

// Add inserts a matrix of vectors under their keys - the native batch operation, where a
// single vector is a one-row matrix. Overlaps safely with concurrent searches on the same
// handle.
//
// The input scalar type is independent of the stored quantization; the library converts on
// ingest. An expansion of zero derives the candidate-pool breadth, and a nil stats skips
// telemetry.
//
// Returns the row count on success and zero on failure: a batch is all-or-nothing, since
// every fallible step runs before the first entry identity is claimed. An accelerator fault
// severe enough to poison the device context is the sole exception, reporting DeviceFailure
// and latching the index unusable rather than partially applied.
func (index *DenseIndex) Add(keys []Key, vectors Matrix, expansion uint, stats *DenseStats) (uint, error) {
	if err := index.live(); err != nil {
		return 0, err
	}
	dimensions, err := index.Dimensions()
	if err != nil {
		return 0, err
	}
	if err := vectors.validate(dimensions); err != nil {
		return 0, err
	}
	if uint(len(keys)) != vectors.Rows() {
		return 0, fail(ShapeMismatch, "the key count does not match the vector row count")
	}
	if len(keys) == 0 {
		return 0, nil
	}

	config := C.usearch_dense_expansion_config_t{expansion_factor: C.size_t(expansion)}
	var rawStats C.usearch_dense_traversal_stats_t
	var failure C.usearch_error_t
	added := C.usearch_dense_add(index.handle,
		(*C.usearch_key_t)(unsafe.Pointer(unsafe.SliceData(keys))),
		vectors.pointer, C.size_t(vectors.rows), C.size_t(vectors.stride),
		C.usearch_scalar_t(vectors.kind),
		&config, &rawStats, &failure)
	runtime.KeepAlive(keys)
	runtime.KeepAlive(vectors)
	if err := check(failure); err != nil {
		return 0, err
	}
	report(stats, rawStats)
	return uint(added), nil
}

// SearchInto resolves a matrix of queries into caller-owned outputs - the allocation-free
// batch path, parallelized internally.
//
// Never writes to the graph, so any number of these may overlap with concurrent insertions.
func (index *DenseIndex) SearchInto(queries Matrix, outputs *DenseOutputs, expansion uint, stats *DenseStats) error {
	return index.searchDispatch(queries, outputs, expansion, stats, nil)
}

// SearchIfInto is the filtered sibling of SearchInto, admitting only the keys admits
// approves. Non-matching entries still route the traversal, so recall does not collapse at
// low selectivity.
//
// The predicate runs on the library's worker threads, so it must be safe for concurrent use.
func (index *DenseIndex) SearchIfInto(queries Matrix, outputs *DenseOutputs, expansion uint, stats *DenseStats,
	admits func(Key) bool) error {
	if admits == nil {
		return fail(MalformedInput, "the admission predicate is nil")
	}
	return index.searchDispatch(queries, outputs, expansion, stats, admits)
}

func (index *DenseIndex) searchDispatch(queries Matrix, outputs *DenseOutputs, expansion uint, stats *DenseStats,
	admits func(Key) bool) error {
	if err := index.live(); err != nil {
		return err
	}
	dimensions, err := index.Dimensions()
	if err != nil {
		return err
	}
	if err := queries.validate(dimensions); err != nil {
		return err
	}
	if err := outputs.validate(queries.Rows()); err != nil {
		return err
	}
	if queries.Rows() == 0 {
		return nil
	}

	config := C.usearch_dense_expansion_config_t{expansion_factor: C.size_t(expansion)}
	var rawStats C.usearch_dense_traversal_stats_t
	var failure C.usearch_error_t
	var key Key
	var distance Distance
	keysStride := C.size_t(outputs.wanted * uint(unsafe.Sizeof(key)))
	distancesStride := C.size_t(outputs.wanted * uint(unsafe.Sizeof(distance)))
	keysPointer := (*C.usearch_key_t)(unsafe.Pointer(unsafe.SliceData(outputs.keys)))
	distancesPointer := (*C.usearch_distance_t)(unsafe.Pointer(unsafe.SliceData(outputs.distances)))
	countsPointer := (*C.size_t)(unsafe.Pointer(unsafe.SliceData(outputs.counts)))

	if admits == nil {
		C.usearch_dense_search(index.handle,
			queries.pointer, C.size_t(queries.rows), C.size_t(queries.stride),
			C.usearch_scalar_t(queries.kind), C.size_t(outputs.wanted),
			&config, &rawStats,
			keysPointer, keysStride,
			distancesPointer, distancesStride,
			countsPointer, &failure)
	} else {
		// The predicate crosses as an opaque handle rather than a Go pointer, since the
		// closure it names holds Go pointers the cgo rules forbid handing to C. The state
		// pointer addresses the handle itself, which is an integer and holds no pointers.
		state := cgo.NewHandle(admits)
		defer state.Delete()
		C.usearch_dense_search_if(index.handle,
			queries.pointer, C.size_t(queries.rows), C.size_t(queries.stride),
			C.usearch_scalar_t(queries.kind), C.size_t(outputs.wanted),
			&config, &rawStats,
			C.usearch_go_admits(), unsafe.Pointer(&state),
			keysPointer, keysStride,
			distancesPointer, distancesStride,
			countsPointer, &failure)
	}
	runtime.KeepAlive(queries)
	runtime.KeepAlive(outputs.keys)
	runtime.KeepAlive(outputs.distances)
	runtime.KeepAlive(outputs.counts)
	if err := check(failure); err != nil {
		return err
	}
	report(stats, rawStats)
	return nil
}

//export usearchDenseAdmits
func usearchDenseAdmits(key C.usearch_key_t, state unsafe.Pointer) C.int {
	admits := (*cgo.Handle)(state).Value().(func(Key) bool)
	if admits(keyFromRaw(key)) {
		return 1
	}
	return 0
}

// Count reports how many entries are stored under a key; zero means the index holds none.
func (index *DenseIndex) Count(key Key) (uint, error) {
	if err := index.live(); err != nil {
		return 0, err
	}
	var failure C.usearch_error_t
	counted := C.usearch_dense_count(index.handle, key.toRaw(), &failure)
	if err := check(failure); err != nil {
		return 0, err
	}
	return uint(counted), nil
}

// Contains reports whether the index holds at least one entry under a key.
func (index *DenseIndex) Contains(key Key) (bool, error) {
	counted, err := index.Count(key)
	return counted != 0, err
}

// Get retrieves the vectors stored under a key into the matrix, one row per entry sharing the
// key, in insertion order.
//
// Returns the true number of entries, never the number written: a count above the matrix's
// row capacity means the copy was truncated, so size for that many and call again. Count is
// the same number without the copy.
func (index *DenseIndex) Get(key Key, vectors Matrix) (uint, error) {
	if err := index.live(); err != nil {
		return 0, err
	}
	dimensions, err := index.Dimensions()
	if err != nil {
		return 0, err
	}
	if err := vectors.validate(dimensions); err != nil {
		return 0, err
	}
	var failure C.usearch_error_t
	entries := C.usearch_dense_get(index.handle, key.toRaw(),
		vectors.pointer, C.size_t(vectors.rows), C.size_t(vectors.stride),
		C.usearch_scalar_t(vectors.kind), &failure)
	runtime.KeepAlive(vectors)
	if err := check(failure); err != nil {
		return 0, err
	}
	return uint(entries), nil
}

// Merge absorbs a donor index - cheaper than re-inserting its vectors one by one, wiring
// cross-links via landmark stitching. Readers may keep searching throughout, since
// unpublished rows hide behind the watermark.
//
// The donor is consumed whether the call succeeds or fails: the C side owns it afterwards, so
// the donor handle is emptied here and its own Close becomes a no-op.
func (index *DenseIndex) Merge(donor *DenseIndex) error {
	if err := index.live(); err != nil {
		return err
	}
	if donor == nil || donor.handle == nil {
		return fail(Uninitialized, "the donor index is not initialized")
	}
	handle := donor.handle
	donor.handle = nil
	var failure C.usearch_error_t
	C.usearch_dense_merge(index.handle, handle, &failure)
	return check(failure)
}

// Save writes the index under a path, in the layout its extension picks - one .safegraphs
// monolith, or a .safegraphs.json manifest beside its shards. Demands a mutation-free window;
// concurrent readers keep searching throughout.
func (index *DenseIndex) Save(path string) error {
	if err := index.live(); err != nil {
		return err
	}
	var failure C.usearch_error_t
	C.usearch_dense_save(index.handle, borrow(path), &failure)
	runtime.KeepAlive(path)
	return check(failure)
}

// Load reads a checkpoint into this handle, shard by shard, with the handle's residency
// deciding the behavior: materialize into RAM, upload to the device, or open in place on
// disk. A handle whose geometry was left zeroed freezes it here, from the checkpoint's own
// headers.
func (index *DenseIndex) Load(path string) error {
	if err := index.live(); err != nil {
		return err
	}
	var failure C.usearch_error_t
	C.usearch_dense_load(index.handle, borrow(path), &failure)
	runtime.KeepAlive(path)
	return check(failure)
}

// SaveBuffer writes the index into a caller-owned buffer as one monolith. Size the buffer
// with SerializedLength.
func (index *DenseIndex) SaveBuffer(buffer []byte) error {
	if err := index.live(); err != nil {
		return err
	}
	if len(buffer) == 0 {
		return fail(ShapeMismatch, "the buffer is empty")
	}
	var failure C.usearch_error_t
	C.usearch_dense_save_buffer(index.handle, unsafe.Pointer(unsafe.SliceData(buffer)), C.size_t(len(buffer)),
		&failure)
	runtime.KeepAlive(buffer)
	return check(failure)
}

// LoadBuffer reads a monolith from a buffer into this handle, with the residency deciding the
// behavior like Load. A residency that opens in place demands the buffer outlive the handle.
func (index *DenseIndex) LoadBuffer(buffer []byte) error {
	if err := index.live(); err != nil {
		return err
	}
	if len(buffer) == 0 {
		return fail(MalformedInput, "the buffer is empty")
	}
	var failure C.usearch_error_t
	C.usearch_dense_load_buffer(index.handle, unsafe.Pointer(unsafe.SliceData(buffer)), C.size_t(len(buffer)),
		&failure)
	runtime.KeepAlive(buffer)
	return check(failure)
}

// endregion: Index

// region: Helpers

// live reports whether the handle still names an index.
func (index *DenseIndex) live() error {
	if index == nil || index.handle == nil {
		return fail(Uninitialized, "the index is freed or was never initialized")
	}
	return nil
}

// measured pairs a probe's reading with its error slot. cgo renders a C function as a Go
// wrapper that cannot be taken as a value, so each probe calls its own symbol and hands the
// two results here rather than passing the function in.
func measured(reading C.size_t, failure C.usearch_error_t) (uint, error) {
	if err := check(failure); err != nil {
		return 0, err
	}
	return uint(reading), nil
}

// endregion: Helpers
