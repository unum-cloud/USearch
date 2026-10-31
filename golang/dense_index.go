// Vector Search over equidimensional dense embeddings.
//
// File: golang/dense_index.go
// Author: Ash Vardanian
// Date: August 16, 2026

// Every verb is a batch over strided matrices, so a memory-mapped dataset reaches the index
// with no reshape and no copy, and a single vector is a one-row matrix. The breadth of the
// candidates rides each call rather than the handle, which lets a recall sweep run against one
// immutable index with no mutation between points.

package usearch

import (
	"runtime/cgo"
	"unsafe"
)

/*
// Two unchecked promises, so each is asserted per verb rather than per group.
//
// `nocallback` says the C function can never re-enter Go, and cgo then skips the machinery that
// makes a Go-to-C-to-Go crossing safe. It is claimed only where the argument structure carries no
// callback and no executor: a verb naming an executor without threads of its own first runs the
// deferred calls it holds, whose callbacks re-enter Go on the calling thread.
//
// `noescape` says the call retains none of its pointer arguments, so Go may leave them on the
// stack. Submission copies the argument structure, and everything the structure names is pinned,
// which also keeps it on the heap until the call is final.
#cgo noescape usearch_dense_init_ram
#cgo nocallback usearch_dense_init_ram
#cgo noescape usearch_dense_init_vram
#cgo nocallback usearch_dense_init_vram
#cgo noescape usearch_dense_view_ram
#cgo nocallback usearch_dense_view_ram
#cgo noescape usearch_dense_cursor_init
#cgo nocallback usearch_dense_cursor_init
#cgo noescape usearch_dense_search_stream_init
#cgo nocallback usearch_dense_search_stream_init
//
// Retains nothing, but runs on an executor or admits a callback, so `noescape` alone.
#cgo noescape usearch_dense_view_nvme
#cgo noescape usearch_dense_reserve
#cgo noescape usearch_dense_clear
#cgo noescape usearch_dense_describe
#cgo noescape usearch_dense_count
#cgo noescape usearch_dense_get
#cgo noescape usearch_dense_health
#cgo noescape usearch_dense_metadata_read
#cgo noescape usearch_dense_metadata_read_buffer
#cgo noescape usearch_dense_load_ram
#cgo noescape usearch_dense_load_vram
#cgo noescape usearch_dense_load_ram_buffer
#cgo noescape usearch_dense_load_vram_buffer
#cgo noescape usearch_dense_insert
#cgo noescape usearch_dense_refine
#cgo noescape usearch_dense_search
#cgo noescape usearch_dense_maxsim
#cgo noescape usearch_dense_multiply
#cgo noescape usearch_dense_attend
#cgo noescape usearch_dense_absorb
#cgo noescape usearch_dense_save
#cgo noescape usearch_dense_save_buffer
#cgo noescape usearch_dense_cursor_next
#cgo noescape usearch_dense_search_stream_next

#include "shim.h"
*/
import "C"

// region: Geometry

// DenseMetric names a distance kind supported out of the box, with SIMD kernels dispatched by
// the machine's capabilities. All of them derive from a single inner-product kernel with
// pre-computed magnitudes; DenseMetricHamming and DenseMetricJaccard are bit-level, over the packed sub-byte kinds.
//
// Arbitrary metrics belong to the C++ baseline tier, which forfeits the SIMD kernel chains and
// the magnitude cache both.
//
// Mirrors usearch_dense_metric_t.
type DenseMetric int32

const (
	// DenseMetricUnknown tags a metric the library could not name.
	DenseMetricUnknown DenseMetric = C.usearch_dense_metric_unknown_k
	// DenseMetricInner is the inner product.
	DenseMetricInner DenseMetric = C.usearch_dense_metric_inner_k
	// DenseMetricAngular is the cosine distance.
	DenseMetricAngular DenseMetric = C.usearch_dense_metric_angular_k
	// DenseMetricEuclidean is the L2 distance.
	DenseMetricEuclidean DenseMetric = C.usearch_dense_metric_euclidean_k
	// DenseMetricHamming counts differing bits.
	DenseMetricHamming DenseMetric = C.usearch_dense_metric_hamming_k
	// DenseMetricJaccard is the bit-level Jaccard distance.
	DenseMetricJaccard DenseMetric = C.usearch_dense_metric_jaccard_k
)

// String renders the metric as the library spells it. Calls usearch_dense_metric_name.
func (metric DenseMetric) String() string {
	return C.GoString(C.usearch_dense_metric_name(C.usearch_dense_metric_t(metric)))
}

// DenseOptions holds the construction options of a dense index, the fields
// usearch_dense_init_ram_t and usearch_dense_init_vram_t share.
type DenseOptions struct {
	// Dimensions is the width of the indexed vectors; zero leaves the geometry unfrozen, so the
	// index admits no entries.
	Dimensions uint
	// Metric is the distance kind used between vectors.
	Metric DenseMetric
	// Quantization is the scalar type vectors are stored and quantized into.
	Quantization Scalar
	// Connectivity bounds the connections per node in the graph; zero derives.
	Connectivity uint
	// ValuesPerKey bounds how many entries may share one key; zero derives to one - unique
	// keys. Above one is what late interaction needs, a document being one key holding its
	// token rows.
	ValuesPerKey uint
	// CPUClass is the class a host index dispatches as; the zero value derives the CPU's own.
	CPUClass CPUClass
	// GPUClass is the class a device index dispatches as; the zero value derives the devices' own.
	GPUClass GPUClass
}

// DefaultDenseOptions is an angular bf16 index, with every optional field derived.
func DefaultDenseOptions() DenseOptions {
	return DenseOptions{Metric: DenseMetricAngular, Quantization: ScalarBF16}
}

// DenseMetadata is the geometry and metric a checkpoint froze - everything a reader learns
// without opening it, and nothing a reader must choose for itself.
//
// ValuesPerKey is absent because it bounds a live handle's mutable state rather than anything
// the serialized form carries, and the placement is absent because a checkpoint records where it
// was written, never where a reader must keep it. Both stay yours to set in the load config.
//
// Mirrors usearch_dense_metadata_t.
type DenseMetadata struct {
	// Dimensions is the scalars per stored vector.
	Dimensions uint
	// Metric is the distance kind the checkpoint scores with.
	Metric DenseMetric
	// Quantization is the scalar type its vectors are stored and quantized into.
	Quantization Scalar
	// Connectivity is the connections per node the graph was built with.
	Connectivity uint
}

func denseMetadataFromRaw(raw *C.usearch_dense_metadata_t) DenseMetadata {
	return DenseMetadata{
		Dimensions:   uint(raw.dimensions),
		Metric:       DenseMetric(raw.metric),
		Quantization: Scalar(raw.quantization),
		Connectivity: uint(raw.connectivity),
	}
}

// DenseIndexDescription is a live index's frozen metadata and current counters, each read once,
// so under concurrent insertions they may come from slightly different moments.
//
// Mirrors usearch_dense_index_description_t.
type DenseIndexDescription struct {
	// Metadata is the frozen metric, quantization, dimensions and connectivity, as a checkpoint
	// reports them.
	Metadata DenseMetadata
	// Size is the number of stored entries.
	Size uint
	// Capacity is the number of entries hosted without growing again.
	Capacity uint
	// MemoryUsage is the bytes the handle holds; traversal scratch is the executor's.
	MemoryUsage uint
	// CPUClass is the CPU class dispatch picked on this machine, as usearch_cpu_class_name spells it.
	CPUClass string
	// GPUClass is the GPU class dispatch picked on the index's devices; "unknown" for a host index.
	GPUClass string
	// ShardsCount is the number of live shards.
	ShardsCount uint
}

// endregion: Geometry

// region: Configs

// DenseTraversalStats is the traversal telemetry one call reports. Every call overwrites it
// rather than accumulating, so a sweep over many batches sums them itself with Accumulate.
//
// Mirrors usearch_traversal_stats_t.
type DenseTraversalStats struct {
	// VisitedMembers counts the graph members the call visited.
	VisitedMembers uint
	// ComputedDistances counts the distances the call computed, diversity re-selections
	// included.
	ComputedDistances uint
	// Restarts counts starvation re-seedings - a graph-connectivity health signal rather than
	// a cost counter.
	Restarts uint
}

// Accumulate folds another call's telemetry into this one. Binding-only.
func (stats *DenseTraversalStats) Accumulate(other DenseTraversalStats) {
	stats.VisitedMembers += other.VisitedMembers
	stats.ComputedDistances += other.ComputedDistances
	stats.Restarts += other.Restarts
}

// DenseReachability is one reachability audit: the population it judged, and how much of it a
// search reaches.
//
// Mirrors usearch_reachability_t.
type DenseReachability struct {
	// Audited counts the entries published when the audit started; compare Reachable against
	// this, never a later size.
	Audited uint
	// Reachable counts those a search reaches; the rest no search finds until a refinement
	// re-links them.
	Reachable uint
	// Sweeps counts the levels the sweep took to converge.
	Sweeps uint
	// BackSweeps counts the levels the sweep back toward the entry point took, where an audit
	// sweeps both ways.
	BackSweeps uint
}

// DenseRefineReport is what one Refine, Absorb or Split did to the graph.
//
// Mirrors usearch_refine_report_t.
type DenseRefineReport struct {
	// Reselected counts the entries whose neighbors were selected again, the islands of every
	// heal round included.
	Reselected uint
	// Changed counts those whose neighbor set gained a member it lacked; zero means settled.
	Changed uint
	// IslandsFound counts the entries the first audit found not reached both ways.
	IslandsFound uint
	// IslandsReconnected counts the links written between an island and a reached entry.
	IslandsReconnected uint
	// Reachability is the audit closing the heal, as a fresh Health right after would take it.
	Reachability DenseReachability
}

// DenseSoftmaxState is the online-softmax carry of one query row, which folds one attention
// call into another over the same queries.
//
// Mirrors usearch_dense_softmax_state_t.
type DenseSoftmaxState struct {
	// MaximumScore is the largest logit seen so far - the recurrence's m.
	MaximumScore float32
	// Partition is the sum of exponentials taken against that maximum - the recurrence's l.
	Partition float32
}

// stats lends optional telemetry as the pointer the C interface fills.
func (call *call) stats(stats *DenseTraversalStats) *C.usearch_traversal_stats_t {
	if stats == nil {
		return nil
	}
	return (*C.usearch_traversal_stats_t)(unsafe.Pointer(pin(call, stats)))
}

// ValueLookup resolves the keys an attention walk scores to their value rows, all of them rows
// of Values, which stays pinned until the call is final.
//
// Values must be as wide as the keys and stored in the index's own quantization, since the call
// reads them as such; a key Row names no row of Values for, or values of another shape, refuse
// the call with StatusShapeMismatch. Row runs on the executor's threads together, so it must be
// safe for concurrent use, and a panic inside it answers no row and becomes the call's error.
//
// Row is a host function, while a VRAM walk resolves its rows through a device one, so Attend
// refuses a VRAM handle with StatusDevice.
//
// Mirrors usearch_value_lookup_t.
type ValueLookup struct {
	Values RowsView
	Row    func(key Key) (row uint, found bool)
}

// DenseLoadRamConfig carries the optional settings of DenseLoadRam.
//
// Mirrors usearch_dense_load_ram_t.
type DenseLoadRamConfig struct {
	// Executor is the one the read fans out over; nil reads on the calling goroutine.
	Executor *Executor
	// MemoryDomains are the NUMA node IDs to place the parts on, part p loading whole into shard p
	// modulo their count; empty derives every node the process may use.
	MemoryDomains []int32
	// CPUClass is the class to dispatch as; the zero value derives the CPU's own.
	CPUClass CPUClass
	// ValuesPerKey bounds how many entries may share one key, which no checkpoint records; zero
	// derives to one.
	ValuesPerKey uint
	// Progress reports the shards placed and stops wherever it says to; a stopped load creates no
	// index.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseLoadVramConfig carries the optional settings of DenseLoadVram.
//
// Mirrors usearch_dense_load_vram_t.
type DenseLoadVramConfig struct {
	// Executor is the one the upload runs on; nil uploads on the default stream of each device.
	Executor *Executor
	// Devices are the CUDA ordinals to place the parts on, part p loading whole onto device p modulo
	// their count; empty derives one free device.
	Devices []int32
	// GPUClass is the class to dispatch as; the zero value derives the devices' own.
	GPUClass GPUClass
	// ValuesPerKey bounds how many entries may share one key; zero derives to one.
	ValuesPerKey uint
	// Progress reports the shards placed and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseLoadRamBufferConfig carries the optional settings of DenseLoadRamBuffer.
//
// Mirrors usearch_dense_load_ram_buffer_t.
type DenseLoadRamBufferConfig struct {
	// Executor is the one the copy fans out over; nil copies on the calling goroutine.
	Executor *Executor
	// MemoryDomains are the NUMA node IDs to place shards on, the monolith loading whole into the
	// first; empty derives every node the process may use.
	MemoryDomains []int32
	// CPUClass is the class to dispatch as; the zero value derives the CPU's own.
	CPUClass CPUClass
	// ValuesPerKey bounds how many entries may share one key; zero derives to one.
	ValuesPerKey uint
	// Progress reports the bytes placed and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseLoadVramBufferConfig carries the optional settings of DenseLoadVramBuffer.
//
// Mirrors usearch_dense_load_vram_buffer_t.
type DenseLoadVramBufferConfig struct {
	// Executor is the one the upload runs on; nil uploads on the default stream of each device.
	Executor *Executor
	// Devices are the CUDA ordinals to place shards on, the monolith loading whole onto the first;
	// empty derives one free device.
	Devices []int32
	// GPUClass is the class to dispatch as; the zero value derives the devices' own.
	GPUClass GPUClass
	// ValuesPerKey bounds how many entries may share one key; zero derives to one.
	ValuesPerKey uint
	// Progress reports the bytes placed and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseViewNvmeConfig carries the optional settings of DenseViewNvme.
//
// Mirrors usearch_dense_view_nvme_t.
type DenseViewNvmeConfig struct {
	// Executor is the NVMe executor whose rings read the checkpoint's headers; nil reads on the
	// calling goroutine.
	Executor *Executor
	// Paths are the devices or directories to serve the parts from; empty serves each where its
	// manifest points.
	Paths []string
	// CPUClass is the class to dispatch as; the zero value derives the CPU's own.
	CPUClass CPUClass
}

// DenseMetadataReadConfig carries the optional settings of DenseMetadataRead.
//
// Mirrors usearch_dense_metadata_read_t.
type DenseMetadataReadConfig struct {
	// Executor is the one the read runs on; nil reads on the calling goroutine.
	Executor *Executor
}

// DenseMetadataReadBufferConfig carries the optional settings of DenseMetadataReadBuffer.
//
// Mirrors usearch_dense_metadata_read_buffer_t.
type DenseMetadataReadBufferConfig struct {
	// Executor is the one the read runs on; nil reads on the calling goroutine.
	Executor *Executor
}

// DenseInsertConfig carries the optional settings of Insert. A zero value derives every setting.
//
// Mirrors usearch_dense_insert_t.
type DenseInsertConfig struct {
	// ExpansionFactor is the breadth of the candidates the build walk explores; zero derives it
	// from the geometry.
	ExpansionFactor uint
	// Progress reports the entries inserted; every entry is claimed before the first report, so a
	// stop is ignored.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
}

// DenseRefineConfig carries the optional settings of Refine. A zero value derives every setting.
//
// Mirrors usearch_dense_refine_t.
type DenseRefineConfig struct {
	// ExpansionFactor is the breadth of the candidates the re-selection walk explores; zero derives
	// it from the geometry.
	ExpansionFactor uint
	// Progress reports the nodes swept and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
}

// DenseSearchConfig carries the optional settings of Search. A zero value derives every setting.
//
// Mirrors usearch_dense_search_t.
type DenseSearchConfig struct {
	// ExpansionFactor is the breadth of the candidates the walk explores; zero derives it from
	// the geometry, and a scan ignores it.
	ExpansionFactor uint
	// Traversal walks the graph by default; TraversalScan scores every entry exactly, on RAM and
	// NVMe handles, and a VRAM handle refuses it with StatusUnsupported.
	Traversal Traversal
	// Predicate admits only the keys it approves; nil admits everything. Non-matching entries
	// still route the traversal, so recall does not collapse at low selectivity. A VRAM handle
	// refuses one with StatusDevice.
	Predicate Predicate
	// Progress reports the queries resolved and stops wherever it says to; a stopped search
	// reports StatusCancelled and returns no matches.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
}

// DenseMaxsimConfig carries the optional settings of Maxsim. A zero value derives every setting.
//
// Mirrors usearch_dense_maxsim_t.
type DenseMaxsimConfig struct {
	// ExpansionFactor is the breadth of the candidates the walk explores; zero derives it from
	// the geometry.
	ExpansionFactor uint
	// Progress reports the queries resolved and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
}

// DenseMultiplyConfig carries the optional settings of Multiply. A zero value derives every
// setting.
//
// Mirrors usearch_dense_multiply_t.
type DenseMultiplyConfig struct {
	// Wanted bounds the pairs kept per left row, the first ones its walk scored; zero keeps every
	// pair.
	Wanted uint
	// ExpansionFactor is the breadth of the candidates, which widens the emitted pattern; zero
	// derives it from the geometry.
	ExpansionFactor uint
	// Progress reports the rows resolved and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
}

// DenseAttendConfig carries the optional settings of Attend. A zero value derives every setting.
//
// Mirrors usearch_dense_attend_t.
type DenseAttendConfig struct {
	// HeadCount is the query heads sharing the index's key-value head, and with it one walk per
	// token: row t * HeadCount + h is head h of token t, so it must divide the rows; zero means one.
	HeadCount uint
	// Carry is the online-softmax state, one entry per query row and lent by the caller, which
	// folds this call into another over the same queries by partial-softmax rescaling.
	Carry []DenseSoftmaxState
	// OutputsScalar is the scalar type the attended rows land as; zero takes the queries' own.
	OutputsScalar Scalar
	// ExpansionFactor is the breadth of the candidates; zero derives it. Wider candidates walk further,
	// folding more of the tail.
	ExpansionFactor uint
	// ScoreScale is the logit scale, conventionally the inverse square root of the head
	// dimensionality; zero derives it from the index's frozen geometry.
	ScoreScale float32
	// Progress reports the queries resolved and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
}

// DenseAbsorbConfig carries the optional settings of Absorb.
//
// Mirrors usearch_dense_absorb_t.
type DenseAbsorbConfig struct {
	// Progress reports the donors absorbed and stops between them wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseSplitConfig carries the optional settings of Split.
//
// Mirrors usearch_dense_split_t.
type DenseSplitConfig struct {
	// Seed picks the part roots: equal seeds over an unchanged index split equally.
	Seed uint64
	// ExpansionFactor is the breadth of the candidates of the closing refines' walks; zero derives.
	ExpansionFactor uint
	// Progress reports the nodes the closing refines re-selected; a stop never undoes a move.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
}

// DenseSampleConfig carries the optional settings of Sample. A zero value draws across the whole
// index with nothing shown.
//
// Mirrors usearch_dense_sample_t.
type DenseSampleConfig struct {
	// Shown holds the keys already shown: never drawn again, and the paths reach toward them.
	Shown []Key
	// ShownCoordinates places each Shown key, one point apiece.
	ShownCoordinates [][3]float32
	// Focus holds the keys to look around; empty looks around the whole index.
	Focus []Key
	// Seed makes the draw repeatable: equal seeds, shown keys and focus over an unchanged index
	// draw equally.
	Seed uint64
	// ExpansionFactor is the breadth of the candidates of the walks behind the draws; zero derives.
	ExpansionFactor uint
	// Progress reports the draws and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
}

// DenseSaveConfig carries the optional settings of Save.
//
// Mirrors usearch_dense_save_t.
type DenseSaveConfig struct {
	// Progress reports the shards written and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseSaveBufferConfig carries the optional settings of SaveBuffer.
//
// Mirrors usearch_dense_save_buffer_t.
type DenseSaveBufferConfig struct {
	// Progress reports the bytes written and stops wherever it says to; a stopped save leaves no
	// monolith.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseSearchStreamInitConfig carries the optional settings of SearchStream.
//
// Mirrors usearch_dense_search_stream_init_t.
type DenseSearchStreamInitConfig struct {
	// ExpansionFactor is the breadth of the candidates the walk explores; zero derives it from the
	// geometry.
	ExpansionFactor uint
}

// endregion: Configs

// region: Results

// DenseVectors is the rows a call wrote packed into its arena: Get's entries sharing a key, or Attend's
// attended rows.
//
// Mirrors usearch_rows_view_t.
type DenseVectors struct {
	exported
	raw  C.usearch_rows_view_t
	rows uint
}

// Rows reports how many rows the call wrote; zero when Get found the key absent.
func (vectors DenseVectors) Rows() uint { return vectors.rows }

// Stride reports the byte pitch between consecutive rows.
func (vectors DenseVectors) Stride() uint { return uint(vectors.raw.stride) }

// Columns reports the values in each row, as the library wrote them.
func (vectors DenseVectors) Columns() uint { return uint(vectors.raw.columns) }

// Scalar reports the scalar type the rows were written as.
func (vectors DenseVectors) Scalar() Scalar { return Scalar(vectors.raw.scalar) }

// DenseValues borrows the rows from their arena as one slab of values, rows end to end, refused with
// StatusInvalidArgument when Value is not the scalar type they were written as.
func DenseValues[Value Element](vectors DenseVectors) ([]Value, error) {
	if err := vectors.host(); err != nil {
		return nil, err
	}
	if vectors.rows == 0 {
		return nil, nil
	}
	if scalar := scalarOf[Value](); scalar != vectors.Scalar() {
		return nil, fail(StatusInvalidArgument, "rows written as "+vectors.Scalar().String()+" cannot be read as "+scalar.String())
	}
	var element Value
	return unsafe.Slice((*Value)(vectors.raw.data), vectors.rows*vectors.Stride()/uint(unsafe.Sizeof(element))), nil
}

// DenseProducts is the ragged sparse product Multiply wrote into its arena: per left row, every pair
// its walk scored, in traversal order. Maxsim lands its keys in the same shape, one run per query,
// largest score first.
//
// Mirrors usearch_matches_tape_view_t.
type DenseProducts struct {
	exported
	raw      C.usearch_matches_tape_view_t
	queries  uint
	required uint
}

// Queries reports how many left rows the call multiplied, or how many queries it scored.
func (products DenseProducts) Queries() uint { return products.queries }

// Required reports the pairs the walks scored across every left row, or for Maxsim the keys scored
// across every query, before the wanted bound cut any.
func (products DenseProducts) Required() uint { return products.required }

// Query borrows one left row's pairs from the arena - a key naming each indexed entry, and its
// product - or one query's keys, refused with StatusInvalidArgument when query is not below Queries.
func (products DenseProducts) Query(query uint) ([]Key, []Distance, error) {
	if err := products.host(); err != nil {
		return nil, nil, err
	}
	if query >= products.queries {
		return nil, nil, fail(StatusInvalidArgument, "the query is not below the call's count")
	}
	offsets := unsafe.Slice((*uint)(unsafe.Pointer(products.raw.offsets)), products.queries+1)
	keys := unsafe.Slice((*Key)(unsafe.Pointer(products.raw.keys)), offsets[products.queries])
	scores := unsafe.Slice((*Distance)(unsafe.Pointer(products.raw.scores)), offsets[products.queries])
	return keys[offsets[query]:offsets[query+1]], scores[offsets[query]:offsets[query+1]], nil
}

// Monolith is the serialized index SaveBuffer wrote into its arena.
type Monolith struct {
	exported
	raw C.usearch_bytes_view_t
}

// Bytes borrows the monolith from its arena.
func (monolith Monolith) Bytes() ([]byte, error) {
	if err := monolith.host(); err != nil {
		return nil, err
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(monolith.raw.data)), monolith.raw.length), nil
}

// DenseBatch is one cursor pull written into its arena: the keys, and beside them the keys each links
// to when NextLinks asked for them; the vectors themselves come from Get.
type DenseBatch struct {
	exported
	keys    C.usearch_keys_view_t
	links   *C.usearch_keys_tape_view_t
	written uint
}

// Written reports the entries the pull wrote; zero means the walk is exhausted.
func (batch DenseBatch) Written() uint { return batch.written }

// Keys borrows the entries' keys from the arena.
func (batch DenseBatch) Keys() ([]Key, error) {
	if err := batch.host(); err != nil {
		return nil, err
	}
	return unsafe.Slice((*Key)(unsafe.Pointer(batch.keys.data)), batch.written), nil
}

// Links borrows the keys one entry links to from the arena, refused with StatusInvalidArgument when the
// pull was NextKeys or entry is not below Written.
//
// Mirrors usearch_keys_tape_view_t.
func (batch DenseBatch) Links(entry uint) ([]Key, error) {
	if err := batch.host(); err != nil {
		return nil, err
	}
	if batch.links == nil || entry >= batch.written {
		return nil, fail(StatusInvalidArgument, "the pull exported no links, or the entry is not below its count")
	}
	if batch.links.offsets == nil {
		return nil, nil
	}
	offsets := unsafe.Slice((*uint)(unsafe.Pointer(batch.links.offsets)), batch.written+1)
	linked := unsafe.Slice((*Key)(unsafe.Pointer(batch.links.keys)), offsets[batch.written])
	return linked[offsets[entry]:offsets[entry+1]], nil
}

// denseSampleOutputs is every output usearch_dense_sample writes, pinned as one object per call.
type denseSampleOutputs struct {
	keys          C.usearch_keys_view_t
	entryOrdinals *C.size_t
	parents       C.usearch_matches_view_t
	coordinates   C.usearch_rows_view_t
	written       C.size_t
}

// DenseDraws is what Sample wrote into its arena, one entry per draw: its key, which of that key's
// entries it is, the shown or earlier draw it hangs from, and where it is placed.
type DenseDraws struct {
	exported
	outputs denseSampleOutputs
}

// Written reports the draws; fewer than wanted only once the region is exhausted.
func (draws DenseDraws) Written() uint { return uint(draws.outputs.written) }

// Keys borrows the drawn keys from the arena.
func (draws DenseDraws) Keys() ([]Key, error) {
	if err := draws.host(); err != nil {
		return nil, err
	}
	return unsafe.Slice((*Key)(unsafe.Pointer(draws.outputs.keys.data)), draws.Written()), nil
}

// EntryOrdinals borrows which of its key's entries each draw is, in the order Get writes them.
func (draws DenseDraws) EntryOrdinals() ([]uint, error) {
	if err := draws.host(); err != nil {
		return nil, err
	}
	return unsafe.Slice((*uint)(unsafe.Pointer(draws.outputs.entryOrdinals)), draws.Written()), nil
}

// Parents views each draw's parent and its distance as one column per draw, a root holding none.
func (draws DenseDraws) Parents() MatchesView {
	return MatchesView{exported: draws.exported, raw: draws.outputs.parents, queries: draws.Written()}
}

// Coordinates views where each draw is placed, one row of three float32 apiece.
func (draws DenseDraws) Coordinates() DenseVectors {
	return DenseVectors{exported: draws.exported, raw: draws.outputs.coordinates, rows: draws.Written()}
}

// endregion: Results

// region: Index

// DenseIndex is a Vector Search Index over equidimensional dense representations.
//
// A handle may be shared across goroutines. On a RAM handle, insertions and searches overlap
// freely: searches read only entries published behind the visibility watermark and never write
// to the graph, while insertions synchronize through per-node locks and publish in order.
// Serialization is the one exception, demanding a mutation-free window, though concurrent
// readers keep searching throughout.
//
// Its verbs come from the embedded DenseIndexOn and run on the calling goroutine; On runs the same
// verbs blocking on an executor, and Deferred submits them there.
//
// Mirrors usearch_dense_index_t.
type DenseIndex struct {
	DenseIndexOn
	handle C.usearch_dense_index_t
	// residency is the constructor's, which decides whether a Go value lookup can serve Attend.
	residency Residency
}

// newDenseIndex is an index whose handle a lifecycle verb of residency is about to fill.
func newDenseIndex(residency Residency) *DenseIndex {
	index := &DenseIndex{residency: residency}
	index.DenseIndexOn = DenseIndexOn{denseReceiver{index: index}}
	return index
}

// On is the receiver whose verbs block on executor, after the calls submitted there before them.
// Binding-only.
func (index *DenseIndex) On(executor *Executor) DenseIndexOn {
	return DenseIndexOn{denseReceiver{index: index, executor: executor}}
}

// Deferred is the receiver whose verbs submit their calls to executor, each answering a Deferred
// from the same arguments. Binding-only.
func (index *DenseIndex) Deferred(executor *Executor) DenseIndexDeferredOn {
	return DenseIndexDeferredOn{denseReceiver{index: index, executor: executor}}
}

// Close releases the index, waiting for every deferred call naming it. It is idempotent, and a
// closed handle must not be used again. Calls usearch_dense_free.
func (index *DenseIndex) Close() error {
	if index == nil || index.handle == nil {
		return nil
	}
	C.usearch_dense_free(index.handle)
	index.handle = nil
	return nil
}

// DenseInitRam creates an empty, mutable index in host memory, one shard per NUMA node
// memoryDomains names; empty derives every node the process may use. Calls usearch_dense_init_ram.
func DenseInitRam(options DenseOptions, memoryDomains []int32) (*DenseIndex, error) {
	lent := open[*DenseIndex]()
	index := newDenseIndex(ResidencyRam)
	C.usearch_dense_init_ram(&C.usearch_dense_init_ram_t{
		index:          pin(&lent.call, &index.handle),
		outcome:        lent.outcome,
		metric:         C.usearch_dense_metric_t(options.Metric),
		quantization:   C.usearch_scalar_t(options.Quantization),
		dimensions:     C.size_t(options.Dimensions),
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		ram:            lent.ram(memoryDomains, options.CPUClass),
	})
	lent.result = func() *DenseIndex { return index }
	return block(lent)
}

// DenseInitVram creates an empty, mutable index in device memory, sharded across the CUDA
// ordinals devices names and never replicated; empty derives one free device. Calls
// usearch_dense_init_vram.
func DenseInitVram(options DenseOptions, devices []int32) (*DenseIndex, error) {
	lent := open[*DenseIndex]()
	index := newDenseIndex(ResidencyVram)
	C.usearch_dense_init_vram(&C.usearch_dense_init_vram_t{
		index:          pin(&lent.call, &index.handle),
		outcome:        lent.outcome,
		metric:         C.usearch_dense_metric_t(options.Metric),
		quantization:   C.usearch_scalar_t(options.Quantization),
		dimensions:     C.size_t(options.Dimensions),
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		vram:           lent.vram(devices, options.GPUClass),
	})
	lent.result = func() *DenseIndex { return index }
	return block(lent)
}

// DenseLoadRam copies a checkpoint into a new, mutable index in host memory. The path names a
// manifest of a sharded checkpoint or a single .safegraphs monolith. Calls usearch_dense_load_ram.
func DenseLoadRam(path string, config *DenseLoadRamConfig) (*DenseIndex, error) {
	if config == nil {
		config = new(DenseLoadRamConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[*DenseIndex]()
	index := newDenseIndex(ResidencyRam)
	C.usearch_dense_load_ram(&C.usearch_dense_load_ram_t{
		index:          pin(&lent.call, &index.handle),
		executor:       executor,
		outcome:        lent.outcome,
		path:           lent.text(path),
		ram:            lent.ram(config.MemoryDomains, config.CPUClass),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	lent.result = func() *DenseIndex { return index }
	return block(lent)
}

// DenseLoadVram copies a checkpoint into a new, mutable index in device memory, with the placement
// rules of DenseLoadRam. Calls usearch_dense_load_vram.
func DenseLoadVram(path string, config *DenseLoadVramConfig) (*DenseIndex, error) {
	if config == nil {
		config = new(DenseLoadVramConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[*DenseIndex]()
	index := newDenseIndex(ResidencyVram)
	C.usearch_dense_load_vram(&C.usearch_dense_load_vram_t{
		index:          pin(&lent.call, &index.handle),
		executor:       executor,
		outcome:        lent.outcome,
		path:           lent.text(path),
		vram:           lent.vram(config.Devices, config.GPUClass),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	lent.result = func() *DenseIndex { return index }
	return block(lent)
}

// DenseLoadRamBuffer copies a monolith from a buffer into a new, mutable index in host memory, as
// DenseLoadRam does a path. Calls usearch_dense_load_ram_buffer.
func DenseLoadRamBuffer(buffer []byte, config *DenseLoadRamBufferConfig) (*DenseIndex, error) {
	if config == nil {
		config = new(DenseLoadRamBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[*DenseIndex]()
	index := newDenseIndex(ResidencyRam)
	C.usearch_dense_load_ram_buffer(&C.usearch_dense_load_ram_buffer_t{
		index:          pin(&lent.call, &index.handle),
		executor:       executor,
		outcome:        lent.outcome,
		buffer:         lent.bytes(buffer),
		ram:            lent.ram(config.MemoryDomains, config.CPUClass),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	lent.result = func() *DenseIndex { return index }
	return block(lent)
}

// DenseLoadVramBuffer copies a monolith from a buffer into a new, mutable index in device memory.
// Calls usearch_dense_load_vram_buffer.
func DenseLoadVramBuffer(buffer []byte, config *DenseLoadVramBufferConfig) (*DenseIndex, error) {
	if config == nil {
		config = new(DenseLoadVramBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[*DenseIndex]()
	index := newDenseIndex(ResidencyVram)
	C.usearch_dense_load_vram_buffer(&C.usearch_dense_load_vram_buffer_t{
		index:          pin(&lent.call, &index.handle),
		executor:       executor,
		outcome:        lent.outcome,
		buffer:         lent.bytes(buffer),
		vram:           lent.vram(config.Devices, config.GPUClass),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	lent.result = func() *DenseIndex { return index }
	return block(lent)
}

// DenseViewRam serves a checkpoint in place through a memory mapping, as a read-only index
// whose mutating verbs report StatusReadOnly; the files must stay unmodified while it lives.
// Calls usearch_dense_view_ram.
func DenseViewRam(path string) (*DenseIndex, error) {
	lent := open[*DenseIndex]()
	index := newDenseIndex(ResidencyRam)
	C.usearch_dense_view_ram(&C.usearch_dense_view_ram_t{
		index:   pin(&lent.call, &index.handle),
		outcome: lent.outcome,
		path:    lent.text(path),
	})
	lent.result = func() *DenseIndex { return index }
	return block(lent)
}

// DenseViewNvme serves a checkpoint in place from NVMe drives, read through the executor's
// submission rings, as a read-only index. Until NVMe checkpoints carry keys, it reports
// StatusUnsupported. Calls usearch_dense_view_nvme.
func DenseViewNvme(path string, config *DenseViewNvmeConfig) (*DenseIndex, error) {
	if config == nil {
		config = new(DenseViewNvmeConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[*DenseIndex]()
	index := newDenseIndex(ResidencyNvme)
	C.usearch_dense_view_nvme(&C.usearch_dense_view_nvme_t{
		index:    pin(&lent.call, &index.handle),
		executor: executor,
		outcome:  lent.outcome,
		path:     lent.text(path),
		nvme:     lent.nvme(config.Paths, config.CPUClass),
	})
	lent.result = func() *DenseIndex { return index }
	return block(lent)
}

// DenseMetadataRead reads a checkpoint's geometry and metric without loading it - from the
// manifest's summary when the path names one, so no shard is opened. Calls
// usearch_dense_metadata_read.
func DenseMetadataRead(path string, config *DenseMetadataReadConfig) (DenseMetadata, error) {
	if config == nil {
		config = new(DenseMetadataReadConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return DenseMetadata{}, refusal
	}
	lent := open[DenseMetadata]()
	raw := pin(&lent.call, new(C.usearch_dense_metadata_t))
	C.usearch_dense_metadata_read(&C.usearch_dense_metadata_read_t{
		executor: executor,
		outcome:  lent.outcome,
		path:     lent.text(path),
		metadata: raw,
	})
	lent.result = func() DenseMetadata { return denseMetadataFromRaw(raw) }
	return block(lent)
}

// DenseMetadataReadBuffer reads geometry and metric out of a serialized monolith buffer. Calls
// usearch_dense_metadata_read_buffer.
func DenseMetadataReadBuffer(buffer []byte, config *DenseMetadataReadBufferConfig) (DenseMetadata, error) {
	if config == nil {
		config = new(DenseMetadataReadBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return DenseMetadata{}, refusal
	}
	lent := open[DenseMetadata]()
	raw := pin(&lent.call, new(C.usearch_dense_metadata_t))
	C.usearch_dense_metadata_read_buffer(&C.usearch_dense_metadata_read_buffer_t{
		executor: executor,
		outcome:  lent.outcome,
		buffer:   lent.bytes(buffer),
		metadata: raw,
	})
	lent.result = func() DenseMetadata { return denseMetadataFromRaw(raw) }
	return block(lent)
}

// endregion: Index

// region: Receivers

// denseReceiver is an index and the executor its verbs run on, nil for the calling goroutine.
type denseReceiver struct {
	index    *DenseIndex
	executor *Executor
}

// lower reports the handles a verb hands across, refusing a closed index before C dereferences it.
func (receiver denseReceiver) lower() (C.usearch_dense_index_t, C.usearch_executor_t, *Error) {
	if receiver.index == nil || receiver.index.handle == nil {
		return nil, nil, fail(StatusUninitialized, "the index is closed or was never initialized")
	}
	executor, refusal := receiver.executor.lower()
	return receiver.index.handle, executor, refusal
}

// DenseIndexOn is the receiver whose verbs block until their call is final: on the calling goroutine
// when embedded in a DenseIndex, and on an executor when DenseIndex.On makes it. Binding-only.
type DenseIndexOn struct {
	denseReceiver
}

// DenseIndexDeferredOn is the receiver whose verbs submit their calls to an executor and answer a
// Deferred, offering every verb of DenseIndexOn with the same arguments but the blocking-only sugar,
// cursors and streams. Binding-only.
//
// The library reads a deferred call's inputs until the call is final, so the slices a verb is
// handed must not change until then; its arena is borrowed, as a blocking verb borrows it.
type DenseIndexDeferredOn struct {
	denseReceiver
}

// Reserve claims capacity for a number of incoming vectors ahead of insertion. The capacity
// is the desired total, including the current size. Calls usearch_dense_reserve.
func (on DenseIndexOn) Reserve(capacity uint) error {
	_, err := block(on.reserve(C.usearch_completion_blocking_k, capacity))
	return err
}

// Reserve submits DenseIndexOn.Reserve. Calls usearch_dense_reserve.
func (on DenseIndexDeferredOn) Reserve(capacity uint) *Deferred[struct{}] {
	return submitted(on.reserve(C.usearch_completion_deferred_k, capacity))
}

// Clear erases all entries, keeping the reserved capacity and the frozen geometry. No other verb
// may overlap it on the same handle. Calls usearch_dense_clear.
func (on DenseIndexOn) Clear() error {
	_, err := block(on.clear(C.usearch_completion_blocking_k))
	return err
}

// Clear submits DenseIndexOn.Clear. Calls usearch_dense_clear.
func (on DenseIndexDeferredOn) Clear() *Deferred[struct{}] {
	return submitted(on.clear(C.usearch_completion_deferred_k))
}

// Describe reports the frozen metadata and the current counters in one call. Calls
// usearch_dense_describe.
func (on DenseIndexOn) Describe() (DenseIndexDescription, error) {
	return block(on.describe(C.usearch_completion_blocking_k))
}

// Describe submits DenseIndexOn.Describe. Calls usearch_dense_describe.
func (on DenseIndexDeferredOn) Describe() *Deferred[DenseIndexDescription] {
	return submitted(on.describe(C.usearch_completion_deferred_k))
}

// Count reports how many entries are stored under a key; zero means the index holds none.
// Calls usearch_dense_count.
func (on DenseIndexOn) Count(key Key) (uint, error) {
	return block(on.count(C.usearch_completion_blocking_k, key))
}

// Count submits DenseIndexOn.Count. Calls usearch_dense_count.
func (on DenseIndexDeferredOn) Count(key Key) *Deferred[uint] {
	return submitted(on.count(C.usearch_completion_deferred_k, key))
}

// Contains reports whether the index holds at least one entry under a key. Binding-only, over
// usearch_dense_count.
func (on DenseIndexOn) Contains(key Key) (bool, error) {
	counted, err := on.Count(key)
	return counted != 0, err
}

// Get exports every vector stored under a key into arena, one row per entry sharing the key, in
// insertion order, converted to the outputsScalar kind - one call, however many there turn out to be.
//
// The call rewinds arena, and the rows stay readable until it serves another call. Calls
// usearch_dense_get.
func (on DenseIndexOn) Get(key Key, outputsScalar Scalar, arena *Arena) (DenseVectors, error) {
	return block(on.get(C.usearch_completion_blocking_k, key, outputsScalar, arena))
}

// Get submits DenseIndexOn.Get. Calls usearch_dense_get.
func (on DenseIndexDeferredOn) Get(key Key, outputsScalar Scalar, arena *Arena) *Deferred[DenseVectors] {
	return submitted(on.get(C.usearch_completion_deferred_k, key, outputsScalar, arena))
}

// Insert adds a matrix of vectors under their keys - the native batch operation, where a
// single vector is a one-row matrix. Overlaps safely with concurrent searches on the same
// handle.
//
// The input scalar type is independent of the stored quantization; the library converts on
// ingest.
//
// A batch is all-or-nothing and becomes visible at once, since every fallible step runs before
// the first entry identity is claimed. A VRAM call that fails after its first launch publishes
// none of its entries, and leaves the handle refusing every later mutation. Calls
// usearch_dense_insert.
func (on DenseIndexOn) Insert(keys []Key, vectors RowsView, config *DenseInsertConfig) error {
	_, err := block(on.insert(C.usearch_completion_blocking_k, keys, vectors, config))
	return err
}

// Insert submits DenseIndexOn.Insert. Calls usearch_dense_insert.
func (on DenseIndexDeferredOn) Insert(keys []Key, vectors RowsView, config *DenseInsertConfig) *Deferred[struct{}] {
	return submitted(on.insert(C.usearch_completion_deferred_k, keys, vectors, config))
}

// Refine re-selects the neighbors of every stored entry against the finished graph, improving
// edges chosen when the graph around them was still incomplete.
//
// A build step rather than a repair: a cheap insertion pass followed by one refinement reaches
// a better graph, in less total time, than a single expensive insertion pass - so a wide
// expansion factor belongs here rather than in Insert. Reports the entries re-selected and
// changed, so a caller loops until Changed falls below whatever rate they consider settled.
//
// On a RAM handle searches may run throughout; insertions may not. A stop is a time budget
// rather than a failure, so it reports no status. Calls usearch_dense_refine.
func (on DenseIndexOn) Refine(config *DenseRefineConfig) (DenseRefineReport, error) {
	return block(on.refine(C.usearch_completion_blocking_k, config))
}

// Refine submits DenseIndexOn.Refine. Calls usearch_dense_refine.
func (on DenseIndexDeferredOn) Refine(config *DenseRefineConfig) *Deferred[DenseRefineReport] {
	return submitted(on.refine(C.usearch_completion_deferred_k, config))
}

// Health audits how much of the graph a search reaches: forward reachability from as many
// landmarks as one search draws, swept level by level. Read-only, so concurrent insertions merely
// leave fresh entries unaudited. Calls usearch_dense_health.
func (on DenseIndexOn) Health() (DenseReachability, error) {
	return block(on.health(C.usearch_completion_blocking_k))
}

// Health submits DenseIndexOn.Health. Calls usearch_dense_health.
func (on DenseIndexDeferredOn) Health() *Deferred[DenseReachability] {
	return submitted(on.health(C.usearch_completion_deferred_k))
}

// Search resolves a matrix of queries into up to wanted nearest matches apiece, written into arena
// - the native batch path, parallelized internally, and filtered when the config carries a predicate.
//
// Never writes to the graph, so any number of these may overlap with concurrent insertions. A
// zero wanted walks nothing and lands only the counts, every one zero. Calls usearch_dense_search.
func (on DenseIndexOn) Search(queries RowsView, wanted uint, arena *Arena, config *DenseSearchConfig) (MatchesView, error) {
	return block(on.search(C.usearch_completion_blocking_k, queries, wanted, arena, config))
}

// Search submits DenseIndexOn.Search. Calls usearch_dense_search.
func (on DenseIndexDeferredOn) Search(queries RowsView, wanted uint, arena *Arena,
	config *DenseSearchConfig) *Deferred[MatchesView] {
	return submitted(on.search(C.usearch_completion_deferred_k, queries, wanted, arena, config))
}

// Maxsim scores multi-row queries against multi-vector documents by late interaction: every
// query row takes its best match within a key, and those maxima sum to that key's score.
//
// A document is one key holding its token rows, which is what ValuesPerKey above one exists
// for. rowsPerQuery is the token count each query occupies, queries holds them end to end, and
// wanted counts documents rather than rows, zero keeping every key a query scored. Each query's
// keys land in arena as one run, largest score first. Runs on an inner-product index alone.
//
// See https://arxiv.org/abs/2004.12832 - ColBERT, which introduced this scoring. Calls
// usearch_dense_maxsim.
func (on DenseIndexOn) Maxsim(queries RowsView, rowsPerQuery, wanted uint, arena *Arena,
	config *DenseMaxsimConfig) (DenseProducts, error) {
	return block(on.maxsim(C.usearch_completion_blocking_k, queries, rowsPerQuery, wanted, arena, config))
}

// Maxsim submits DenseIndexOn.Maxsim. Calls usearch_dense_maxsim.
func (on DenseIndexDeferredOn) Maxsim(queries RowsView, rowsPerQuery, wanted uint, arena *Arena,
	config *DenseMaxsimConfig) *Deferred[DenseProducts] {
	return submitted(on.maxsim(C.usearch_completion_deferred_k, queries, rowsPerQuery, wanted, arena, config))
}

// Multiply is the approximate sparse product of a matrix against the indexed vectors - the linear
// algebra sibling of Search, reporting every pair the traversal scored rather than only the
// nearest. Pairs arrive in traversal order rather than sorted, and the pairs the walk never
// reaches are the low and negative products. Runs on an inner-product index alone.
//
// How many pairs a walk scores is known only once it finishes, so the products, keys and offsets
// land in arena, as Get's rows do. Calls usearch_dense_multiply.
func (on DenseIndexOn) Multiply(left RowsView, arena *Arena, config *DenseMultiplyConfig) (DenseProducts, error) {
	return block(on.multiply(C.usearch_completion_blocking_k, left, arena, config))
}

// Multiply submits DenseIndexOn.Multiply. Calls usearch_dense_multiply.
func (on DenseIndexDeferredOn) Multiply(left RowsView, arena *Arena, config *DenseMultiplyConfig) *Deferred[DenseProducts] {
	return submitted(on.multiply(C.usearch_completion_deferred_k, left, arena, config))
}

// Attend is softmax attention of every query row over every indexed key its walk scores, folded
// as Flash Attention folds key tiles, with no top-k cut, writing one row per query into arena,
// already divided by the partition.
//
// The index holds one layer's keys for one key-value head and no values, so lookup resolves every
// scored key to its value row. Runs on an inner-product index alone, since a softmax needs logits,
// and on RAM and NVMe handles, since a VRAM walk needs a device lookup no Go function can be.
//
// See https://arxiv.org/abs/2205.14135 - Flash Attention, whose recurrence this reuses. Calls
// usearch_dense_attend.
func (on DenseIndexOn) Attend(queries RowsView, lookup ValueLookup, arena *Arena,
	config *DenseAttendConfig) (DenseVectors, error) {
	return block(on.attend(C.usearch_completion_blocking_k, queries, lookup, arena, config))
}

// Attend submits DenseIndexOn.Attend. Calls usearch_dense_attend.
func (on DenseIndexDeferredOn) Attend(queries RowsView, lookup ValueLookup, arena *Arena,
	config *DenseAttendConfig) *Deferred[DenseVectors] {
	return submitted(on.attend(C.usearch_completion_deferred_k, queries, lookup, arena, config))
}

// Absorb moves every donor into this index, one after another, keeping their graphs rather than
// re-inserting their vectors: each donor's entries append shard by shard, then a refinement links
// them across the appended part. Donors must share the index's residency, scalar, metric and
// geometry. Readers of a RAM index keep searching throughout; a VRAM one admits no overlapping
// search.
//
// Each absorbed donor is consumed, and every later verb on it reports StatusUninitialized. A stop
// between donors, or a failure, leaves every donor not yet absorbed a live handle the caller still
// owns and closes; until the call is final every donor reports StatusUninitialized. Calls
// usearch_dense_absorb.
func (on DenseIndexOn) Absorb(donors []*DenseIndex, config *DenseAbsorbConfig) (DenseRefineReport, error) {
	return block(on.absorb(C.usearch_completion_blocking_k, donors, config))
}

// Absorb submits DenseIndexOn.Absorb. Calls usearch_dense_absorb.
func (on DenseIndexDeferredOn) Absorb(donors []*DenseIndex, config *DenseAbsorbConfig) *Deferred[DenseRefineReport] {
	return submitted(on.absorb(C.usearch_completion_deferred_k, donors, config))
}

// Split hands this index to the receivers, the inverse of Absorb: graph-local parts keeping their
// edges, one per receiver, leaving this index empty but live. Receivers must be empty and share
// the index's residency, scalar, class, metric, geometry and shard count, and a refusal moves
// nothing. Calls usearch_dense_split.
func (on DenseIndexOn) Split(receivers []*DenseIndex, config *DenseSplitConfig) (DenseRefineReport, error) {
	return block(on.split(C.usearch_completion_blocking_k, receivers, config))
}

// Split submits DenseIndexOn.Split. Calls usearch_dense_split.
func (on DenseIndexDeferredOn) Split(receivers []*DenseIndex, config *DenseSplitConfig) *Deferred[DenseRefineReport] {
	return submitted(on.split(C.usearch_completion_deferred_k, receivers, config))
}

// Sample draws up to wanted keys around the config's Focus, or across the whole index, each
// hanging from a shown or earlier draw and placed in three dimensions beside the shown ones, so a
// view grows one connected sample as it zooms; a larger wanted draws a superset. Only RAM samples
// yet. Calls usearch_dense_sample.
func (on DenseIndexOn) Sample(wanted uint, arena *Arena, config *DenseSampleConfig) (DenseDraws, error) {
	return block(on.sample(C.usearch_completion_blocking_k, wanted, arena, config))
}

// Sample submits DenseIndexOn.Sample. Calls usearch_dense_sample.
func (on DenseIndexDeferredOn) Sample(wanted uint, arena *Arena, config *DenseSampleConfig) *Deferred[DenseDraws] {
	return submitted(on.sample(C.usearch_completion_deferred_k, wanted, arena, config))
}

// Save writes the index under a path, in the layout its extension picks - one .safegraphs
// monolith, or a .safegraphs.json manifest beside its shards. Demands a mutation-free window;
// concurrent readers keep searching throughout. Calls usearch_dense_save.
func (on DenseIndexOn) Save(path string, config *DenseSaveConfig) error {
	_, err := block(on.save(C.usearch_completion_blocking_k, path, config))
	return err
}

// Save submits DenseIndexOn.Save. Calls usearch_dense_save.
func (on DenseIndexDeferredOn) Save(path string, config *DenseSaveConfig) *Deferred[struct{}] {
	return submitted(on.save(C.usearch_completion_deferred_k, path, config))
}

// SaveBuffer serializes the index as one monolith into arena, a size only the call itself can
// discover. Calls usearch_dense_save_buffer.
func (on DenseIndexOn) SaveBuffer(arena *Arena, config *DenseSaveBufferConfig) (Monolith, error) {
	return block(on.saveBuffer(C.usearch_completion_blocking_k, arena, config))
}

// SaveBuffer submits DenseIndexOn.SaveBuffer. Calls usearch_dense_save_buffer.
func (on DenseIndexDeferredOn) SaveBuffer(arena *Arena, config *DenseSaveBufferConfig) *Deferred[Monolith] {
	return submitted(on.saveBuffer(C.usearch_completion_deferred_k, arena, config))
}

// endregion: Receivers

// region: Calls

func (receiver denseReceiver) reserve(completion C.usearch_completion_t, capacity uint) *lent[struct{}] {
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_dense_reserve(&C.usearch_dense_reserve_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		capacity:   C.size_t(capacity),
	})
	return lent
}

func (receiver denseReceiver) clear(completion C.usearch_completion_t) *lent[struct{}] {
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_dense_clear(&C.usearch_dense_clear_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
	})
	return lent
}

func (receiver denseReceiver) describe(completion C.usearch_completion_t) *lent[DenseIndexDescription] {
	lent := open[DenseIndexDescription]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	raw := pin(&lent.call, new(C.usearch_dense_index_description_t))
	C.usearch_dense_describe(&C.usearch_dense_describe_t{
		index:       index,
		executor:    executor,
		outcome:     lent.outcome,
		completion:  completion,
		description: raw,
	})
	lent.result = func() DenseIndexDescription {
		return DenseIndexDescription{
			Metadata:    denseMetadataFromRaw(&raw.metadata),
			Size:        uint(raw.size),
			Capacity:    uint(raw.capacity),
			MemoryUsage: uint(raw.memory_usage),
			CPUClass:    C.GoString(C.usearch_cpu_class_name(raw.cpu_class)),
			GPUClass:    C.GoString(C.usearch_gpu_class_name(raw.gpu_class)),
			ShardsCount: uint(raw.shards_count),
		}
	}
	return lent
}

func (receiver denseReceiver) count(completion C.usearch_completion_t, key Key) *lent[uint] {
	lent := open[uint]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	entries := pin(&lent.call, new(C.size_t))
	C.usearch_dense_count(&C.usearch_dense_count_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		key:        key.toRaw(),
		entries:    entries,
	})
	lent.result = func() uint { return uint(*entries) }
	return lent
}

func (receiver denseReceiver) get(completion C.usearch_completion_t, key Key, outputsScalar Scalar,
	arena *Arena) *lent[DenseVectors] {
	lent := open[DenseVectors]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	memory, refusal := arena.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	vectors := pin(&lent.call, new(C.usearch_rows_view_t))
	written := pin(&lent.call, new(C.size_t))
	C.usearch_dense_get(&C.usearch_dense_get_t{
		index:          index,
		executor:       executor,
		outcome:        lent.outcome,
		completion:     completion,
		key:            key.toRaw(),
		vectors:        vectors,
		written:        written,
		arena:          memory,
		outputs_scalar: C.usearch_scalar_t(outputsScalar),
	})
	view := lent.rewound(arena)
	lent.result = func() DenseVectors { return DenseVectors{exported: view, raw: *vectors, rows: uint(*written)} }
	return lent
}

func (receiver denseReceiver) insert(completion C.usearch_completion_t, keys []Key, vectors RowsView,
	config *DenseInsertConfig) *lent[struct{}] {
	if config == nil {
		config = new(DenseInsertConfig)
	}
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	// The call has one count, so only this side sees the keys and the rows disagree.
	if uint(len(keys)) != vectors.rows {
		return lent.refuse(fail(StatusShapeMismatch, "the key count does not match the vector row count"))
	}
	C.usearch_dense_insert(&C.usearch_dense_insert_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome,
		completion:       completion,
		count:            C.size_t(vectors.rows),
		keys:             lent.keys(keys),
		vectors:          vectors.lower(&lent.call),
		expansion_factor: C.size_t(config.ExpansionFactor),
		progress:         lent.progress(config.Progress, config.ProgressSteps),
		stats:            lent.stats(config.Stats),
	})
	return lent
}

func (receiver denseReceiver) refine(completion C.usearch_completion_t,
	config *DenseRefineConfig) *lent[DenseRefineReport] {
	if config == nil {
		config = new(DenseRefineConfig)
	}
	lent := open[DenseRefineReport]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	report := pin(&lent.call, new(DenseRefineReport))
	C.usearch_dense_refine(&C.usearch_dense_refine_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome,
		completion:       completion,
		report:           (*C.usearch_refine_report_t)(unsafe.Pointer(report)),
		expansion_factor: C.size_t(config.ExpansionFactor),
		progress:         lent.progress(config.Progress, config.ProgressSteps),
		stats:            lent.stats(config.Stats),
	})
	lent.result = func() DenseRefineReport { return *report }
	return lent
}

func (receiver denseReceiver) health(completion C.usearch_completion_t) *lent[DenseReachability] {
	lent := open[DenseReachability]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	reachability := pin(&lent.call, new(DenseReachability))
	C.usearch_dense_health(&C.usearch_dense_health_t{
		index:        index,
		executor:     executor,
		outcome:      lent.outcome,
		completion:   completion,
		reachability: (*C.usearch_reachability_t)(unsafe.Pointer(reachability)),
	})
	lent.result = func() DenseReachability { return *reachability }
	return lent
}

func (receiver denseReceiver) search(completion C.usearch_completion_t, queries RowsView, wanted uint,
	arena *Arena, config *DenseSearchConfig) *lent[MatchesView] {
	if config == nil {
		config = new(DenseSearchConfig)
	}
	lent := open[MatchesView]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	memory, refusal := arena.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	matches := pin(&lent.call, new(C.usearch_matches_view_t))
	C.usearch_dense_search(&C.usearch_dense_search_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome,
		completion:       completion,
		count:            C.size_t(queries.rows),
		queries:          queries.lower(&lent.call),
		matches:          matches,
		arena:            memory,
		wanted:           C.size_t(wanted),
		expansion_factor: C.size_t(config.ExpansionFactor),
		traversal:        C.usearch_traversal_t(config.Traversal),
		predicate:        lent.predicate(config.Predicate),
		progress:         lent.progress(config.Progress, config.ProgressSteps),
		stats:            lent.stats(config.Stats),
	})
	view := lent.rewound(arena)
	lent.result = func() MatchesView { return MatchesView{exported: view, raw: *matches, queries: queries.rows} }
	return lent
}

func (receiver denseReceiver) maxsim(completion C.usearch_completion_t, queries RowsView, rowsPerQuery, wanted uint,
	arena *Arena, config *DenseMaxsimConfig) *lent[DenseProducts] {
	if config == nil {
		config = new(DenseMaxsimConfig)
	}
	lent := open[DenseProducts]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	// The call counts queries, so only this side sees the matrix's rows fail to divide.
	if rowsPerQuery == 0 || queries.rows%rowsPerQuery != 0 {
		return lent.refuse(fail(StatusShapeMismatch, "the query rows do not divide into whole queries"))
	}
	memory, refusal := arena.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	count := queries.rows / rowsPerQuery
	matches := pin(&lent.call, new(C.usearch_matches_tape_view_t))
	required := pin(&lent.call, new(C.size_t))
	C.usearch_dense_maxsim(&C.usearch_dense_maxsim_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome,
		completion:       completion,
		count:            C.size_t(count),
		queries:          queries.lower(&lent.call),
		rows_per_query:   C.size_t(rowsPerQuery),
		matches:          matches,
		required:         required,
		arena:            memory,
		wanted:           C.size_t(wanted),
		expansion_factor: C.size_t(config.ExpansionFactor),
		progress:         lent.progress(config.Progress, config.ProgressSteps),
		stats:            lent.stats(config.Stats),
	})
	view := lent.rewound(arena)
	lent.result = func() DenseProducts {
		return DenseProducts{exported: view, raw: *matches, queries: count, required: uint(*required)}
	}
	return lent
}

func (receiver denseReceiver) multiply(completion C.usearch_completion_t, left RowsView, arena *Arena,
	config *DenseMultiplyConfig) *lent[DenseProducts] {
	if config == nil {
		config = new(DenseMultiplyConfig)
	}
	lent := open[DenseProducts]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	memory, refusal := arena.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	products := pin(&lent.call, new(C.usearch_matches_tape_view_t))
	required := pin(&lent.call, new(C.size_t))
	C.usearch_dense_multiply(&C.usearch_dense_multiply_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome,
		completion:       completion,
		count:            C.size_t(left.rows),
		left:             left.lower(&lent.call),
		products:         products,
		required:         required,
		arena:            memory,
		wanted:           C.size_t(config.Wanted),
		expansion_factor: C.size_t(config.ExpansionFactor),
		progress:         lent.progress(config.Progress, config.ProgressSteps),
		stats:            lent.stats(config.Stats),
	})
	view := lent.rewound(arena)
	lent.result = func() DenseProducts {
		return DenseProducts{exported: view, raw: *products, queries: left.rows, required: uint(*required)}
	}
	return lent
}

func (receiver denseReceiver) attend(completion C.usearch_completion_t, queries RowsView, lookup ValueLookup,
	arena *Arena, config *DenseAttendConfig) *lent[DenseVectors] {
	if config == nil {
		config = new(DenseAttendConfig)
	}
	lent := open[DenseVectors]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	if lookup.Row == nil {
		return lent.refuse(fail(StatusInvalidArgument, "the value lookup names no row function"))
	}
	// A device walk would call the host trampoline from a kernel.
	if receiver.index.residency == ResidencyVram {
		return lent.refuse(fail(StatusDevice, "a VRAM walk resolves values through a device function, which Go cannot supply"))
	}
	// The carry pointer has no length, so only this side can see it fall short.
	if config.Carry != nil && uint(len(config.Carry)) < queries.rows {
		return lent.refuse(fail(StatusShapeMismatch, "the softmax carry holds fewer entries than there are queries"))
	}
	memory, refusal := arena.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	lent.pinner.Pin(lookup.Values.pointer)
	outputs := pin(&lent.call, new(C.usearch_rows_view_t))
	C.usearch_dense_attend(&C.usearch_dense_attend_t{
		index:              index,
		executor:           executor,
		outcome:            lent.outcome,
		completion:         completion,
		count:              C.size_t(queries.rows),
		queries:            queries.lower(&lent.call),
		head_count:         C.size_t(config.HeadCount),
		value_lookup:       C.usearch_go_looks_up_values(),
		value_lookup_state: lent.callback(lookup),
		carry:              (*C.usearch_dense_softmax_state_t)(unsafe.Pointer(lend(&lent.call, config.Carry))),
		outputs:            outputs,
		arena:              memory,
		outputs_scalar:     C.usearch_scalar_t(config.OutputsScalar),
		expansion_factor:   C.size_t(config.ExpansionFactor),
		score_scale:        C.float(config.ScoreScale),
		progress:           lent.progress(config.Progress, config.ProgressSteps),
		stats:              lent.stats(config.Stats),
	})
	view := lent.rewound(arena)
	lent.result = func() DenseVectors { return DenseVectors{exported: view, raw: *outputs, rows: queries.rows} }
	return lent
}

//export usearchLooksUpValues
func usearchLooksUpValues(state unsafe.Pointer, keys *C.usearch_key_t, count C.size_t, scalar C.usearch_scalar_t,
	columns C.size_t, rows *unsafe.Pointer) {
	upcall := (*cgo.Handle)(state).Value().(upcall)
	lookup := upcall.value.(ValueLookup)
	values := lookup.Values
	found := unsafe.Slice(rows, count)
	defer func() {
		if recovered := recover(); recovered != nil { // a panic unwinding through C frames would skip their unlocks
			clear(found)
			upcall.call.latch(recovered)
		}
	}()
	// The library reads rows of its own shape, so values of another one answer no row at all.
	shaped := Scalar(scalar) == values.scalar && uint(columns) == values.dimensions
	for position, key := range unsafe.Slice((*Key)(unsafe.Pointer(keys)), count) {
		found[position] = nil
		if !shaped {
			continue
		}
		if row, ok := lookup.Row(key); ok && row < values.rows {
			found[position] = unsafe.Add(values.pointer, row*values.stride)
		}
	}
}

func (receiver denseReceiver) absorb(completion C.usearch_completion_t, donors []*DenseIndex,
	config *DenseAbsorbConfig) *lent[DenseRefineReport] {
	if config == nil {
		config = new(DenseAbsorbConfig)
	}
	lent := open[DenseRefineReport]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	for _, donor := range donors {
		if donor == nil || donor.handle == nil {
			return lent.refuse(fail(StatusUninitialized, "a donor is closed or was never initialized"))
		}
	}
	// The library nulls each slot it absorbs, so once a failed call is final the slots still set
	// are the donors it left live, handed back to their owners.
	handles := make([]C.usearch_dense_index_t, len(donors))
	for position, donor := range donors {
		handles[position], donor.handle = donor.handle, nil
	}
	lent.discard = func() {
		for position, handle := range handles {
			if handle != nil {
				donors[position].handle = handle
			}
		}
	}
	report := pin(&lent.call, new(DenseRefineReport))
	C.usearch_dense_absorb(&C.usearch_dense_absorb_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		count:      C.size_t(len(handles)),
		donors:     lend(&lent.call, handles),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(report)),
		progress:   lent.progress(config.Progress, config.ProgressSteps),
	})
	lent.result = func() DenseRefineReport { return *report }
	return lent
}

func (receiver denseReceiver) split(completion C.usearch_completion_t, receivers []*DenseIndex,
	config *DenseSplitConfig) *lent[DenseRefineReport] {
	if config == nil {
		config = new(DenseSplitConfig)
	}
	lent := open[DenseRefineReport]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	handles := make([]C.usearch_dense_index_t, len(receivers))
	for position, target := range receivers {
		if target == nil || target.handle == nil {
			return lent.refuse(fail(StatusUninitialized, "a receiver is closed or was never initialized"))
		}
		handles[position] = target.handle
	}
	report := pin(&lent.call, new(DenseRefineReport))
	C.usearch_dense_split(&C.usearch_dense_split_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome,
		completion:       completion,
		count:            C.size_t(len(handles)),
		receivers:        lend(&lent.call, handles),
		report:           (*C.usearch_refine_report_t)(unsafe.Pointer(report)),
		expansion_factor: C.size_t(config.ExpansionFactor),
		seed:             C.uint64_t(config.Seed),
		progress:         lent.progress(config.Progress, config.ProgressSteps),
		stats:            lent.stats(config.Stats),
	})
	lent.result = func() DenseRefineReport { return *report }
	return lent
}

func (receiver denseReceiver) sample(completion C.usearch_completion_t, wanted uint, arena *Arena,
	config *DenseSampleConfig) *lent[DenseDraws] {
	if config == nil {
		config = new(DenseSampleConfig)
	}
	lent := open[DenseDraws]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	// The call has one count for both, so only this side sees the keys and their places disagree.
	if len(config.Shown) != len(config.ShownCoordinates) {
		return lent.refuse(fail(StatusShapeMismatch, "the shown keys and coordinates disagree in length"))
	}
	memory, refusal := arena.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	outputs := pin(&lent.call, new(denseSampleOutputs))
	C.usearch_dense_sample(&C.usearch_dense_sample_t{
		index:       index,
		executor:    executor,
		outcome:     lent.outcome,
		completion:  completion,
		shown_count: C.size_t(len(config.Shown)),
		shown:       lent.keys(config.Shown),
		shown_coordinates: C.usearch_rows_view_t{
			data:    unsafe.Pointer(lend(&lent.call, config.ShownCoordinates)),
			stride:  C.size_t(unsafe.Sizeof([3]float32{})),
			columns: 3,
			scalar:  C.usearch_scalar_t(ScalarF32),
		},
		focus_count:      C.size_t(len(config.Focus)),
		focus:            lent.keys(config.Focus),
		keys:             &outputs.keys,
		entry_ordinals:   &outputs.entryOrdinals,
		parents:          &outputs.parents,
		coordinates:      &outputs.coordinates,
		written:          &outputs.written,
		arena:            memory,
		wanted:           C.size_t(wanted),
		seed:             C.uint64_t(config.Seed),
		expansion_factor: C.size_t(config.ExpansionFactor),
		progress:         lent.progress(config.Progress, config.ProgressSteps),
		stats:            lent.stats(config.Stats),
	})
	view := lent.rewound(arena)
	lent.result = func() DenseDraws { return DenseDraws{exported: view, outputs: *outputs} }
	return lent
}

func (receiver denseReceiver) save(completion C.usearch_completion_t, path string,
	config *DenseSaveConfig) *lent[struct{}] {
	if config == nil {
		config = new(DenseSaveConfig)
	}
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_dense_save(&C.usearch_dense_save_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		path:       lent.text(path),
		progress:   lent.progress(config.Progress, config.ProgressSteps),
	})
	return lent
}

func (receiver denseReceiver) saveBuffer(completion C.usearch_completion_t, arena *Arena,
	config *DenseSaveBufferConfig) *lent[Monolith] {
	if config == nil {
		config = new(DenseSaveBufferConfig)
	}
	lent := open[Monolith]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	memory, refusal := arena.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	monolith := pin(&lent.call, new(C.usearch_bytes_view_t))
	C.usearch_dense_save_buffer(&C.usearch_dense_save_buffer_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		monolith:   monolith,
		arena:      memory,
		progress:   lent.progress(config.Progress, config.ProgressSteps),
	})
	view := lent.rewound(arena)
	lent.result = func() Monolith { return Monolith{exported: view, raw: *monolith} }
	return lent
}

// endregion: Calls

// region: Cursors & Streams

// DenseCursor is a walk over the index contents, pulled blocking on the executor of the
// receiver that opened it. Each shard's publication watermark is read as the walk enters that
// shard, so an entry once passed is never revisited.
//
// Mirrors usearch_dense_cursor_t.
type DenseCursor struct {
	handle   C.usearch_dense_cursor_t
	receiver denseReceiver
}

// Cursor opens a walk over the current contents. Without starts it yields every entry as stored; with them,
// the start keys' own entries first in order, then the entries those link to breadth-first, each once.
// Calls usearch_dense_cursor_init.
func (on DenseIndexOn) Cursor(starts ...Key) (*DenseCursor, error) {
	index, _, refusal := on.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[*DenseCursor]()
	// cgo inspects the whole object behind a lent pointer, so the handle sits apart from the one
	// the receiver holds.
	handle := pin(&lent.call, new(C.usearch_dense_cursor_t))
	C.usearch_dense_cursor_init(&C.usearch_dense_cursor_init_t{
		index:        index,
		outcome:      lent.outcome,
		starts_count: C.size_t(len(starts)),
		starts:       (*C.usearch_key_t)(unsafe.Pointer(lend(&lent.call, starts))),
		cursor:       handle,
	})
	lent.result = func() *DenseCursor { return &DenseCursor{handle: *handle, receiver: on.denseReceiver} }
	return block(lent)
}

// NextKeys copies up to wanted next keys into arena; a batch of none means the walk is exhausted.
// Calls usearch_dense_cursor_next.
func (cursor *DenseCursor) NextKeys(wanted uint, arena *Arena) (DenseBatch, error) {
	return cursor.next(wanted, nil, arena)
}

// NextLinks copies up to wanted next keys into arena, each with the keys it links to; a batch of none
// means the walk is exhausted. Calls usearch_dense_cursor_next.
func (cursor *DenseCursor) NextLinks(wanted uint, arena *Arena) (DenseBatch, error) {
	return cursor.next(wanted, new(C.usearch_keys_tape_view_t), arena)
}

// next pulls a batch, exporting each entry's links only when links names where their view lands.
func (cursor *DenseCursor) next(wanted uint, links *C.usearch_keys_tape_view_t, arena *Arena) (DenseBatch, error) {
	index, executor, refusal := cursor.receiver.lower()
	if refusal == nil && cursor.handle == nil {
		refusal = fail(StatusInvalidArgument, "the cursor is closed")
	}
	if refusal != nil {
		return DenseBatch{}, refusal
	}
	memory, refusal := arena.lower()
	if refusal != nil {
		return DenseBatch{}, refusal
	}
	lent := open[DenseBatch]()
	keys := pin(&lent.call, new(C.usearch_keys_view_t))
	written := pin(&lent.call, new(C.size_t))
	if links != nil {
		lent.pinner.Pin(links)
	}
	C.usearch_dense_cursor_next(&C.usearch_dense_cursor_next_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: C.usearch_completion_blocking_k,
		cursor:     cursor.handle,
		keys:       keys,
		links:      links,
		written:    written,
		arena:      memory,
		wanted:     C.size_t(wanted),
	})
	view := lent.rewound(arena)
	lent.result = func() DenseBatch {
		return DenseBatch{exported: view, keys: *keys, links: links, written: uint(*written)}
	}
	return block(lent)
}

// Close releases the cursor. It is idempotent. Calls usearch_dense_cursor_free.
func (cursor *DenseCursor) Close() error {
	if cursor == nil || cursor.handle == nil {
		return nil
	}
	C.usearch_dense_cursor_free(cursor.handle)
	cursor.handle = nil
	return nil
}

// DenseSearchStream is a resumable nearest-neighbor walk over one query, pulled blocking on the
// executor of the receiver that opened it. Ordering beyond the expansion horizon is best-effort
// nearest, not globally exact.
//
// Mirrors usearch_dense_search_stream_t.
type DenseSearchStream struct {
	handle   C.usearch_dense_search_stream_t
	receiver denseReceiver
}

// SearchStream opens a resumable search for a one-row query, so callers that post-filter
// aggressively pull until satisfied instead of guessing the wanted count upfront. No residency
// implements the stream yet, so the engine answers StatusUnsupported. Calls
// usearch_dense_search_stream_init.
func (on DenseIndexOn) SearchStream(query RowsView, config *DenseSearchStreamInitConfig) (*DenseSearchStream, error) {
	if config == nil {
		config = new(DenseSearchStreamInitConfig)
	}
	index, _, refusal := on.lower()
	if refusal == nil && query.rows != 1 {
		refusal = fail(StatusShapeMismatch, "a stream walks exactly one query row")
	}
	if refusal != nil {
		return nil, refusal
	}
	lent := open[*DenseSearchStream]()
	handle := pin(&lent.call, new(C.usearch_dense_search_stream_t))
	C.usearch_dense_search_stream_init(&C.usearch_dense_search_stream_init_t{
		stream:           handle,
		index:            index,
		outcome:          lent.outcome,
		query:            query.lower(&lent.call),
		expansion_factor: C.size_t(config.ExpansionFactor),
	})
	lent.result = func() *DenseSearchStream { return &DenseSearchStream{handle: *handle, receiver: on.denseReceiver} }
	return block(lent)
}

// Next pulls up to wanted next-nearest matches into arena, as the one query of a search; a query
// with none means the walk is drained. Calls usearch_dense_search_stream_next.
func (stream *DenseSearchStream) Next(wanted uint, arena *Arena) (MatchesView, error) {
	index, executor, refusal := stream.receiver.lower()
	if refusal == nil && stream.handle == nil {
		refusal = fail(StatusInvalidArgument, "the stream is closed")
	}
	if refusal != nil {
		return MatchesView{}, refusal
	}
	memory, refusal := arena.lower()
	if refusal != nil {
		return MatchesView{}, refusal
	}
	lent := open[MatchesView]()
	matches := pin(&lent.call, new(C.usearch_matches_view_t))
	C.usearch_dense_search_stream_next(&C.usearch_dense_search_stream_next_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: C.usearch_completion_blocking_k,
		stream:     stream.handle,
		matches:    matches,
		arena:      memory,
		wanted:     C.size_t(wanted),
	})
	view := lent.rewound(arena)
	lent.result = func() MatchesView { return MatchesView{exported: view, raw: *matches, queries: 1} }
	return block(lent)
}

// Close releases the stream and the walk state it keeps. It is idempotent. Calls
// usearch_dense_search_stream_free.
func (stream *DenseSearchStream) Close() error {
	if stream == nil || stream.handle == nil {
		return nil
	}
	C.usearch_dense_search_stream_free(stream.handle)
	stream.handle = nil
	return nil
}

// endregion: Cursors & Streams
