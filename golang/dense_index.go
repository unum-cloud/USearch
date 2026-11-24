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
// makes a Go-to-C-to-Go crossing safe. It is claimed only where the argument structure holds no
// callback and no executor.
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
#cgo noescape usearch_dense_load_nvme
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
#cgo noescape usearch_dense_refine_plan_finish
#cgo noescape usearch_dense_search
#cgo noescape usearch_dense_maxsim
#cgo noescape usearch_dense_multiply
#cgo noescape usearch_dense_attend
#cgo noescape usearch_dense_reshard
#cgo noescape usearch_dense_reshard_plan_finish
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

// String renders the metric as the library spells it.
func (metric DenseMetric) String() string {
	return C.GoString(C.usearch_dense_metric_name(C.usearch_dense_metric_t(metric)))
}

// DenseMetricFromName reads a metric back from the spelling String gives it, through
// usearch_dense_metric_from_name; a name the library does not spell fails with
// StatusInvalidArgument.
func DenseMetricFromName(name string) (DenseMetric, error) {
	metric := C.usearch_dense_metric_t(DenseMetricUnknown)
	view := C.usearch_bytes_view_t{data: (*C.uint8_t)(unsafe.Pointer(unsafe.StringData(name))), length: C.size_t(len(name))}
	if status := C.usearch_dense_metric_from_name(view, &metric); status != C.usearch_status_success_k {
		return DenseMetricUnknown, fail(Status(status), "no dense metric is spelled \""+name+"\"")
	}
	return DenseMetric(metric), nil
}

// DenseOptions holds the construction options of a dense index, the fields DenseInitRam and
// DenseInitVram share.
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
	// ValuesPerKey bounds how many entries may share one key; zero is unbounded. Above one is
	// what late interaction needs, a document being one key holding its token rows.
	ValuesPerKey uint
	// CPUClassCeiling is the newest class dispatch may pick, intersected with what the machine runs,
	// so a class the machine lacks narrows rather than refuses; the zero value means no ceiling.
	CPUClassCeiling CPUClass
	// GPUClassCeiling is the newest class dispatch may pick, capped below the best class every named
	// device loads; a ceiling no named device loads is refused as unsupported; the zero value means no
	// ceiling.
	GPUClassCeiling GPUClass
}

// DefaultDenseOptions is an angular bf16 index, with every optional field derived.
func DefaultDenseOptions() DenseOptions {
	return DenseOptions{Metric: DenseMetricAngular, Quantization: ScalarBF16}
}

// DenseMetadata is the geometry and metric a checkpoint froze - everything a reader learns
// without opening it, and nothing a reader must choose for itself.
//
// ValuesPerKey is absent because it bounds a live handle's mutable state rather than anything
// the serialized form holds, and the placement is absent because a checkpoint records where it
// was written, never where a reader must keep it. Both stay yours to set in the load config.
type DenseMetadata struct {
	// Metric is the distance kind the checkpoint scores with.
	Metric DenseMetric
	// Quantization is the scalar type its vectors are stored and quantized into.
	Quantization Scalar
	// Dimensions is the scalars per stored vector.
	Dimensions uint
	// Connectivity is the connections per node the graph was built with.
	Connectivity uint
}

// DenseIndexDescription is a live index's frozen metadata and current counters, each read once,
// so under concurrent insertions they may come from slightly different moments.
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
	// CPUClass is the CPU class whose kits the index dispatched to on RAM and NVMe; CPUClassUnknown
	// on VRAM.
	CPUClass CPUClass
	// GPUClass is the GPU class whose kernels the index dispatched to on VRAM; GPUClassUnknown on RAM
	// and NVMe.
	GPUClass GPUClass
	// ShardsCount is the number of live shards.
	ShardsCount uint
	// LoadPath is how the last load from a path read the parts onto the devices.
	LoadPath LoadPath
}

// endregion: Geometry

// region: Configs

// DenseTraversalStats is the traversal telemetry one call reports. Every call overwrites it
// rather than accumulating, so a sweep over many batches sums them itself with Accumulate.
type DenseTraversalStats struct {
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
func (stats *DenseTraversalStats) Accumulate(other DenseTraversalStats) {
	stats.VisitedMembers += other.VisitedMembers
	stats.ComputedDistances += other.ComputedDistances
	stats.Restarts += other.Restarts
	stats.MissingHints += other.MissingHints
}

// DenseReachability is one reachability audit: the population it judged, and how much of it a
// search reaches.
type DenseReachability struct {
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

// DenseRefineVerdict is what a round concluded, so no caller picks a change rate.
type DenseRefineVerdict int32

const (
	// DenseRefineNone means no round concluded: a report no round filled yet.
	DenseRefineNone DenseRefineVerdict = C.usearch_refine_none_k
	// DenseRefineImproved means the round changed some entry's link set, so another round can help.
	DenseRefineImproved DenseRefineVerdict = C.usearch_refine_improved_k
	// DenseRefineConverged means the round changed no entry's link set.
	DenseRefineConverged DenseRefineVerdict = C.usearch_refine_converged_k
	// DenseRefineStalled means the round lowered nothing while something stands, as full
	// entries leave no slot.
	DenseRefineStalled DenseRefineVerdict = C.usearch_refine_stalled_k
	// DenseRefineStopped means the caller stopped the round.
	DenseRefineStopped DenseRefineVerdict = C.usearch_refine_stopped_k
)

// String renders the verdict as the library spells it.
func (verdict DenseRefineVerdict) String() string {
	return C.GoString(C.usearch_refine_verdict_name(C.usearch_refine_verdict_t(verdict)))
}

// DenseRefineReport is what one Refine or Reshard did to the graph, so far when its plan stays
// live, and what the whole plan did once its finish lands it.
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
	// Neighborhoods counts the neighborhoods the round re-stitched.
	Neighborhoods uint
	// WidestNeighborhood counts the entries of the widest neighborhood.
	WidestNeighborhood uint
	// AddedLinks counts the links the round added.
	AddedLinks uint
	// DroppedLinks counts the links the round dropped.
	DroppedLinks uint
	// Verdict is what the round concluded.
	Verdict DenseRefineVerdict
	// Opening is the health check opening the round.
	Opening DenseReachability
	// Reachability is the health check closing the heal, as a fresh Health right after would take it.
	Reachability DenseReachability
}

// DenseSoftmaxState is the online-softmax carry of one query row, which folds one attention
// call into another over the same queries.
type DenseSoftmaxState struct {
	// MaximumScore is the largest logit seen so far - the recurrence's m.
	MaximumScore float32
	// Partition is the sum of exponentials taken against that maximum - the recurrence's l.
	Partition float32
}

// stats passes optional telemetry as the pointer the C interface fills.
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
// the call with StatusShapeMismatch, and a nil Row refuses it with StatusInvalidArgument. Row
// runs on the executor's threads together, so it must be safe for concurrent use, and it must
// not panic.
//
// Row is a host function, while a VRAM walk resolves its rows through a device one, so Attend
// serves RAM and NVMe handles alone.
type ValueLookup struct {
	Values RowsView
	Row    func(key Key) (row uint, found bool)
}

// DenseLoadRamConfig holds the optional settings of DenseLoadRam.
type DenseLoadRamConfig struct {
	// Executor is the one the read fans out over; nil reads on the calling goroutine.
	Executor *Executor
	// MemoryDomains are the NUMA node IDs to place the parts on, part p loading whole into shard p
	// modulo their count; empty derives every node the process may use.
	MemoryDomains []int32
	// CPUClassCeiling is the newest class dispatch may pick, intersected with what the machine runs,
	// so a class the machine lacks narrows rather than refuses; the zero value means no ceiling.
	CPUClassCeiling CPUClass
	// ValuesPerKey bounds how many entries may share one key, which no checkpoint records; zero
	// is unbounded.
	ValuesPerKey uint
	// Progress reports the shards placed and stops wherever it says to; a stopped load creates no
	// index.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseLoadVramConfig holds the optional settings of DenseLoadVram.
type DenseLoadVramConfig struct {
	// Executor is the one the upload runs on; nil uploads on the default stream of each device.
	Executor *Executor
	// Devices are the CUDA ordinals to place the parts on, part p loading whole onto device p modulo
	// their count; empty derives one free device.
	Devices []int32
	// GPUClassCeiling is the newest class dispatch may pick, capped below the best class every named
	// device loads; a ceiling no named device loads is refused as unsupported; the zero value means no
	// ceiling.
	GPUClassCeiling GPUClass
	// ValuesPerKey bounds how many entries may share one key; zero is unbounded.
	ValuesPerKey uint
	// Progress reports the shards placed and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseLoadRamBufferConfig holds the optional settings of DenseLoadRamBuffer.
type DenseLoadRamBufferConfig struct {
	// Executor is the one the copy fans out over; nil copies on the calling goroutine.
	Executor *Executor
	// MemoryDomains are the NUMA node IDs to place shards on, the monolith loading whole into the
	// first; empty derives every node the process may use.
	MemoryDomains []int32
	// CPUClassCeiling is the newest class dispatch may pick, intersected with what the machine runs,
	// so a class the machine lacks narrows rather than refuses; the zero value means no ceiling.
	CPUClassCeiling CPUClass
	// ValuesPerKey bounds how many entries may share one key; zero is unbounded.
	ValuesPerKey uint
	// Progress reports the bytes placed and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseLoadVramBufferConfig holds the optional settings of DenseLoadVramBuffer.
type DenseLoadVramBufferConfig struct {
	// Executor is the one the upload runs on; nil uploads on the default stream of each device.
	Executor *Executor
	// Devices are the CUDA ordinals to place shards on, the monolith loading whole onto the first;
	// empty derives one free device.
	Devices []int32
	// GPUClassCeiling is the newest class dispatch may pick, capped below the best class every named
	// device loads; a ceiling no named device loads is refused as unsupported; the zero value means no
	// ceiling.
	GPUClassCeiling GPUClass
	// ValuesPerKey bounds how many entries may share one key; zero is unbounded.
	ValuesPerKey uint
	// Progress reports the bytes placed and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseViewNvmeConfig holds the optional settings of DenseViewNvme.
type DenseViewNvmeConfig struct {
	// Executor is the NVMe executor whose rings read the checkpoint's headers; nil reads on the
	// calling goroutine.
	Executor *Executor
	// Paths are the devices or directories to serve the parts from; empty serves each where its
	// manifest points.
	Paths []string
	// CPUClassCeiling is the newest class dispatch may pick, intersected with what the machine runs,
	// so a class the machine lacks narrows rather than refuses; the zero value means no ceiling.
	CPUClassCeiling CPUClass
	// KeyTableDomains are the memory domains of a key table held in RAM, which Count, Get, Sample,
	// MaxSim and hinted searches read; nil holds none, and an empty non-nil slice derives them.
	KeyTableDomains []int32
}

// DenseMetadataReadConfig holds the optional settings of DenseMetadataRead.
type DenseMetadataReadConfig struct {
	// Executor is the one the read runs on; nil reads on the calling goroutine.
	Executor *Executor
}

// DenseMetadataReadBufferConfig holds the optional settings of DenseMetadataReadBuffer.
type DenseMetadataReadBufferConfig struct {
	// Executor is the one the read runs on; nil reads on the calling goroutine.
	Executor *Executor
}

// DenseInsertConfig holds the optional settings of Insert. A zero value derives every setting.
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

// DenseRefineConfig holds the optional settings of Refine. A zero value derives every setting.
type DenseRefineConfig struct {
	// ExpansionFactor is the breadth of the candidates the re-selection walk explores; zero derives
	// it from the geometry.
	ExpansionFactor uint
	// MemoryLimit is the bytes a loaded NVMe handle holds for the one part it rewrites; zero derives
	// what rounds the part in RAM, and RAM and VRAM handles never read it.
	MemoryLimit uint
	// Progress reports each phase's units at its barriers and stops there wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
	// Plan keeps the round live across calls: nil runs one round to its end, a fresh plan opens one
	// that a stop leaves live, and a live one continues where it stopped, or starts the next round.
	Plan *DenseRefinePlan
}

// DenseSearchConfig holds the optional settings of Search. A zero value derives every setting.
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

// DenseMaxsimConfig holds the optional settings of Maxsim. A zero value derives every setting.
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

// DenseMultiplyConfig holds the optional settings of Multiply. A zero value derives every
// setting.
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

// DenseAttendConfig holds the optional settings of Attend. A zero value derives every setting.
type DenseAttendConfig struct {
	// HeadCount is the query heads sharing the index's key-value head, and with it one walk per
	// token: row t * HeadCount + h is head h of token t, so it must divide the rows; zero means one.
	HeadCount uint
	// Carry is the online-softmax state, one entry per query row and passed by the caller, which
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

// DenseReshardConfig holds the optional placements and settings of Reshard. A nil placement keeps
// the handle's own.
type DenseReshardConfig struct {
	// MemoryDomains are the NUMA node IDs of a RAM placement of a handle recut in place, one shard
	// per node at least; nil keeps the placement, and an empty non-nil slice derives them.
	MemoryDomains []int32
	// Devices are the CUDA ordinals of a VRAM placement, one shard per device at least; nil keeps it.
	Devices []int32
	// Paths are the drives of an NVMe placement, one shard per drive at least; nil keeps it.
	Paths []string
	// ShardCapacity is the entries per shard of the RAM or VRAM placement; zero keeps the handle's
	// own bound.
	ShardCapacity uint
	// Seed picks the part roots of a source split among several targets: equal seeds over an
	// unchanged source split equally.
	Seed uint64
	// ExpansionFactor is the breadth of the candidates of the closing rounds; zero derives.
	ExpansionFactor uint
	// MemoryLimit is the bytes a loaded NVMe target holds for the one part it rewrites; zero derives
	// what rounds the part in RAM, and RAM and VRAM targets never read it.
	MemoryLimit uint
	// Progress reports the closing rounds' units at their barriers; a stop never undoes a move.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
	// Plan keeps the reshard live across calls: nil moves and runs to the end, a fresh plan opens one
	// that a stop leaves live, and a live one continues where it stopped.
	Plan *DenseReshardPlan
}

// DenseSampleConfig holds the optional settings of Sample. A zero value draws across the whole
// index with nothing shown.
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

// DenseSaveConfig holds the optional settings of Save.
type DenseSaveConfig struct {
	// Progress reports the shards written and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseSaveBufferConfig holds the optional settings of SaveBuffer.
type DenseSaveBufferConfig struct {
	// Progress reports the bytes written and stops wherever it says to; a stopped save leaves no
	// monolith.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// DenseSearchStreamInitConfig holds the optional settings of SearchStream.
type DenseSearchStreamInitConfig struct {
	// ExpansionFactor is the breadth of the candidates the walk explores; zero derives it from the
	// geometry.
	ExpansionFactor uint
}

// DenseRefinePlan is one refine kept live across calls: the round in hand, and the root every
// round's health checks share. The zero value opens on the first Refine naming it, and
// RefinePlanFinish closes it; while it lives the index admits no insertion and no save, and each
// call names the index and executor it was opened with.
type DenseRefinePlan struct {
	// Phase is where the last call naming the plan stopped, PhaseDone once its round ended.
	Phase  PhaseProgress
	handle C.usearch_dense_refine_plan_t
}

// lower passes the plan's handle and phase to a call, both nil for no plan.
func (plan *DenseRefinePlan) lower(call *call) (*C.usearch_dense_refine_plan_t, *C.usearch_phase_progress_t) {
	if plan == nil {
		return nil, nil
	}
	pin(call, plan)
	return &plan.handle, (*C.usearch_phase_progress_t)(unsafe.Pointer(&plan.Phase))
}

// take hands the plan's handle to its finish, leaving the plan fresh; a nil plan hands none, which
// the library refuses.
func (plan *DenseRefinePlan) take() C.usearch_dense_refine_plan_t {
	if plan == nil {
		return nil
	}
	handle := plan.handle
	plan.handle = nil
	return handle
}

// DenseReshardPlan is one reshard kept live across calls: its own sources and targets, the entries
// still to move, then the rounds of every target that took entries. The zero value opens on the
// first Reshard naming it, and ReshardPlanFinish closes it; while it lives every handle it names
// admits no insertion and no save, and each call names the executor it was opened with.
type DenseReshardPlan struct {
	// Phase is where the last call naming the plan stopped, PhaseDone once its rounds ended.
	Phase  PhaseProgress
	handle C.usearch_dense_reshard_plan_t
}

// lower passes the plan's handle and phase to a call, both nil for no plan.
func (plan *DenseReshardPlan) lower(call *call) (*C.usearch_dense_reshard_plan_t, *C.usearch_phase_progress_t) {
	if plan == nil {
		return nil, nil
	}
	pin(call, plan)
	return &plan.handle, (*C.usearch_phase_progress_t)(unsafe.Pointer(&plan.Phase))
}

// take hands the plan's handle to its finish, leaving the plan fresh; a nil plan hands none, which
// the library refuses.
func (plan *DenseReshardPlan) take() C.usearch_dense_reshard_plan_t {
	if plan == nil {
		return nil
	}
	handle := plan.handle
	plan.handle = nil
	return handle
}

// endregion: Configs

// region: Results

// DenseVectors is the rows a call wrote packed into its arena: Get's entries sharing a key, or Attend's
// attended rows.
type DenseVectors struct {
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

// DenseValues views the rows from their arena as one slab of values, rows end to end, refused with
// StatusInvalidArgument when Value is not the scalar type they were written as.
func DenseValues[Value Element](vectors DenseVectors) ([]Value, error) {
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
type DenseProducts struct {
	raw      C.usearch_matches_tape_view_t
	queries  uint
	required uint
}

// Queries reports how many left rows the call multiplied, or how many queries it scored.
func (products DenseProducts) Queries() uint { return products.queries }

// Required reports the pairs the walks scored across every left row, or for Maxsim the keys scored
// across every query, before the wanted bound cut any.
func (products DenseProducts) Required() uint { return products.required }

// Query views one left row's pairs from the arena - a key naming each indexed entry, and its
// product - or one query's keys; query must be below Queries.
func (products DenseProducts) Query(query uint) ([]Key, []Distance) {
	offsets := unsafe.Slice((*uint)(unsafe.Pointer(products.raw.offsets)), products.queries+1)
	keys := unsafe.Slice((*Key)(unsafe.Pointer(products.raw.keys)), offsets[products.queries])
	scores := unsafe.Slice((*Distance)(unsafe.Pointer(products.raw.scores)), offsets[products.queries])
	return keys[offsets[query]:offsets[query+1]], scores[offsets[query]:offsets[query+1]]
}

// Monolith is the serialized index SaveBuffer wrote into its arena.
type Monolith struct {
	raw C.usearch_bytes_view_t
}

// Bytes views the monolith from its arena.
func (monolith Monolith) Bytes() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(monolith.raw.data)), monolith.raw.length)
}

// DenseBatch is one cursor pull written into its arena: the keys, and beside them the keys each links
// to when NextLinks asked for them; the vectors themselves come from Get.
type DenseBatch struct {
	keys    C.usearch_keys_view_t
	links   C.usearch_keys_tape_view_t
	written uint
}

// Written reports the entries the pull wrote; zero means the walk is exhausted.
func (batch DenseBatch) Written() uint { return batch.written }

// Keys views the entries' keys from the arena.
func (batch DenseBatch) Keys() []Key {
	return unsafe.Slice((*Key)(unsafe.Pointer(batch.keys.data)), batch.written)
}

// Links views the keys one entry links to from the arena, nil after NextKeys; entry must be below
// Written.
func (batch DenseBatch) Links(entry uint) []Key {
	if batch.links.offsets == nil {
		return nil
	}
	offsets := unsafe.Slice((*uint)(unsafe.Pointer(batch.links.offsets)), batch.written+1)
	linked := unsafe.Slice((*Key)(unsafe.Pointer(batch.links.keys)), offsets[batch.written])
	return linked[offsets[entry]:offsets[entry+1]]
}

// DenseDraws is what Sample wrote into its arena, one entry per draw: its key, which of that key's
// entries it is, the shown or earlier draw it hangs from, and where it is placed.
type DenseDraws struct {
	keys          C.usearch_keys_view_t
	entryOrdinals *C.size_t
	parents       C.usearch_matches_view_t
	coordinates   C.usearch_rows_view_t
	written       uint
}

// Written reports the draws; fewer than wanted only once the region is exhausted.
func (draws DenseDraws) Written() uint { return draws.written }

// Keys views the drawn keys from the arena.
func (draws DenseDraws) Keys() []Key {
	return unsafe.Slice((*Key)(unsafe.Pointer(draws.keys.data)), draws.written)
}

// EntryOrdinals views which of its key's entries each draw is, in the order Get writes them.
func (draws DenseDraws) EntryOrdinals() []uint {
	return unsafe.Slice((*uint)(unsafe.Pointer(draws.entryOrdinals)), draws.written)
}

// Parents views each draw's parent and its distance as one column per draw, a root holding none.
func (draws DenseDraws) Parents() MatchesView {
	return MatchesView{raw: draws.parents, queries: draws.written}
}

// Coordinates views where each draw is placed, one row of three float32 apiece.
func (draws DenseDraws) Coordinates() DenseVectors {
	return DenseVectors{raw: draws.coordinates, rows: draws.written}
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
type DenseIndex struct {
	DenseIndexOn
	handle C.usearch_dense_index_t
}

// denseIndexOf wraps the handle a lifecycle verb wrote.
func denseIndexOf(handle C.usearch_dense_index_t, err error) (*DenseIndex, error) {
	if err != nil {
		return nil, err
	}
	index := &DenseIndex{handle: handle}
	index.DenseIndexOn = DenseIndexOn{denseReceiver{index: index}}
	return index, nil
}

// On is the receiver whose verbs block on executor. Binding-only.
func (index *DenseIndex) On(executor *Executor) DenseIndexOn {
	return DenseIndexOn{denseReceiver{index: index, executor: executor}}
}

// Deferred is the receiver whose verbs submit their calls to executor, each answering a Deferred
// from the same arguments. Binding-only.
func (index *DenseIndex) Deferred(executor *Executor) DenseIndexDeferredOn {
	return DenseIndexDeferredOn{denseReceiver{index: index, executor: executor}}
}

// Close releases the index, waiting for every deferred call naming it. It is idempotent, and a
// closed handle must not be used again.
func (index *DenseIndex) Close() error {
	if index == nil || index.handle == nil {
		return nil
	}
	C.usearch_dense_free(index.handle)
	index.handle = nil
	return nil
}

// DenseInitRam creates an empty, mutable index in host memory, one shard per NUMA node
// memoryDomains names; empty derives every node the process may use.
func DenseInitRam(options DenseOptions, memoryDomains []int32) (*DenseIndex, error) {
	call := open[C.usearch_dense_index_t]()
	C.usearch_dense_init_ram(&C.usearch_dense_init_ram_t{
		index:          call.result,
		outcome:        call.outcome,
		metric:         C.usearch_dense_metric_t(options.Metric),
		quantization:   C.usearch_scalar_t(options.Quantization),
		dimensions:     C.size_t(options.Dimensions),
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		ram:            call.ram(memoryDomains, options.CPUClassCeiling),
	})
	return denseIndexOf(block(call))
}

// DenseInitVram creates an empty, mutable index in device memory, sharded across the CUDA
// ordinals devices names and never replicated; empty derives one free device.
func DenseInitVram(options DenseOptions, devices []int32) (*DenseIndex, error) {
	call := open[C.usearch_dense_index_t]()
	C.usearch_dense_init_vram(&C.usearch_dense_init_vram_t{
		index:          call.result,
		outcome:        call.outcome,
		metric:         C.usearch_dense_metric_t(options.Metric),
		quantization:   C.usearch_scalar_t(options.Quantization),
		dimensions:     C.size_t(options.Dimensions),
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		vram:           call.vram(devices, options.GPUClassCeiling),
	})
	return denseIndexOf(block(call))
}

// DenseLoadRam copies a checkpoint into a new, mutable index in host memory. The path names a
// manifest of a sharded checkpoint or a single .safegraphs monolith.
func DenseLoadRam(path string, config *DenseLoadRamConfig) (*DenseIndex, error) {
	if config == nil {
		config = new(DenseLoadRamConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	call := open[C.usearch_dense_index_t]()
	C.usearch_dense_load_ram(&C.usearch_dense_load_ram_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome,
		path:           call.text(path),
		ram:            call.ram(config.MemoryDomains, config.CPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       call.progress(config.Progress, config.ProgressSteps),
	})
	return denseIndexOf(block(call))
}

// DenseLoadVram copies a checkpoint into a new, mutable index in device memory, with the placement
// rules of DenseLoadRam.
func DenseLoadVram(path string, config *DenseLoadVramConfig) (*DenseIndex, error) {
	if config == nil {
		config = new(DenseLoadVramConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	call := open[C.usearch_dense_index_t]()
	C.usearch_dense_load_vram(&C.usearch_dense_load_vram_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome,
		path:           call.text(path),
		vram:           call.vram(config.Devices, config.GPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       call.progress(config.Progress, config.ProgressSteps),
	})
	return denseIndexOf(block(call))
}

// DenseLoadRamBuffer copies a monolith from a buffer into a new, mutable index in host memory, as
// DenseLoadRam does a path.
func DenseLoadRamBuffer(buffer []byte, config *DenseLoadRamBufferConfig) (*DenseIndex, error) {
	if config == nil {
		config = new(DenseLoadRamBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	call := open[C.usearch_dense_index_t]()
	C.usearch_dense_load_ram_buffer(&C.usearch_dense_load_ram_buffer_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome,
		buffer:         call.bytes(buffer),
		ram:            call.ram(config.MemoryDomains, config.CPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       call.progress(config.Progress, config.ProgressSteps),
	})
	return denseIndexOf(block(call))
}

// DenseLoadVramBuffer copies a monolith from a buffer into a new, mutable index in device memory.
func DenseLoadVramBuffer(buffer []byte, config *DenseLoadVramBufferConfig) (*DenseIndex, error) {
	if config == nil {
		config = new(DenseLoadVramBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	call := open[C.usearch_dense_index_t]()
	C.usearch_dense_load_vram_buffer(&C.usearch_dense_load_vram_buffer_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome,
		buffer:         call.bytes(buffer),
		vram:           call.vram(config.Devices, config.GPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       call.progress(config.Progress, config.ProgressSteps),
	})
	return denseIndexOf(block(call))
}

// DenseViewRam serves a checkpoint in place through a memory mapping, as a read-only index
// whose mutating verbs report StatusReadOnly; the files must stay unmodified while it lives.
func DenseViewRam(path string) (*DenseIndex, error) {
	call := open[C.usearch_dense_index_t]()
	C.usearch_dense_view_ram(&C.usearch_dense_view_ram_t{
		index:   call.result,
		outcome: call.outcome,
		path:    call.text(path),
	})
	return denseIndexOf(block(call))
}

// DenseViewNvme serves a checkpoint in place from NVMe drives, read through the executor's
// submission rings, as a read-only index.
func DenseViewNvme(path string, config *DenseViewNvmeConfig) (*DenseIndex, error) {
	if config == nil {
		config = new(DenseViewNvmeConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	call := open[C.usearch_dense_index_t]()
	C.usearch_dense_view_nvme(&C.usearch_dense_view_nvme_t{
		index:    call.result,
		executor: executor,
		outcome:  call.outcome,
		path:     call.text(path),
		nvme:     call.nvme(config.Paths, config.CPUClassCeiling),
		ram:      call.keyTable(config.KeyTableDomains, config.CPUClassCeiling),
	})
	return denseIndexOf(block(call))
}

// DenseLoadNvme opens a checkpoint's manifest from NVMe drives as a writable index: it answers
// every read a view answers, and Refine and Reshard rewrite its parts. A lone monolith has no
// manifest to replace and reports StatusUnsupported, as do Paths naming any.
func DenseLoadNvme(path string, config *DenseViewNvmeConfig) (*DenseIndex, error) {
	if config == nil {
		config = new(DenseViewNvmeConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	call := open[C.usearch_dense_index_t]()
	C.usearch_dense_load_nvme(&C.usearch_dense_load_nvme_t{
		index:    call.result,
		executor: executor,
		outcome:  call.outcome,
		path:     call.text(path),
		nvme:     call.nvme(config.Paths, config.CPUClassCeiling),
		ram:      call.keyTable(config.KeyTableDomains, config.CPUClassCeiling),
	})
	return denseIndexOf(block(call))
}

// keyTable passes the optional RAM placement of an NVMe handle's key table, nil holding none.
func (call *call) keyTable(memoryDomains []int32, cpuClassCeiling CPUClass) *C.usearch_ram_options_t {
	if memoryDomains == nil {
		return nil
	}
	ram := call.ram(memoryDomains, cpuClassCeiling)
	return pin(call, &ram)
}

// DenseMetadataRead reads a checkpoint's geometry and metric without loading it - from the
// manifest's summary when the path names one, so no shard is opened.
func DenseMetadataRead(path string, config *DenseMetadataReadConfig) (DenseMetadata, error) {
	if config == nil {
		config = new(DenseMetadataReadConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return DenseMetadata{}, refusal
	}
	call := open[DenseMetadata]()
	C.usearch_dense_metadata_read(&C.usearch_dense_metadata_read_t{
		executor: executor,
		outcome:  call.outcome,
		path:     call.text(path),
		metadata: (*C.usearch_dense_metadata_t)(unsafe.Pointer(call.result)),
	})
	return block(call)
}

// DenseMetadataReadBuffer reads geometry and metric out of a serialized monolith buffer.
func DenseMetadataReadBuffer(buffer []byte, config *DenseMetadataReadBufferConfig) (DenseMetadata, error) {
	if config == nil {
		config = new(DenseMetadataReadBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return DenseMetadata{}, refusal
	}
	call := open[DenseMetadata]()
	C.usearch_dense_metadata_read_buffer(&C.usearch_dense_metadata_read_buffer_t{
		executor: executor,
		outcome:  call.outcome,
		buffer:   call.bytes(buffer),
		metadata: (*C.usearch_dense_metadata_t)(unsafe.Pointer(call.result)),
	})
	return block(call)
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
// handed must not change until then; its arena is viewed, as a blocking verb views it.
type DenseIndexDeferredOn struct {
	denseReceiver
}

// Reserve claims capacity for a number of incoming vectors ahead of insertion. The capacity
// is the desired total, including the current size.
func (on DenseIndexOn) Reserve(capacity uint) error {
	_, err := block(on.reserve(C.usearch_completion_blocking_k, capacity))
	return err
}

// Reserve submits DenseIndexOn.Reserve.
func (on DenseIndexDeferredOn) Reserve(capacity uint) *Deferred[struct{}] {
	return submitted(on.reserve(C.usearch_completion_deferred_k, capacity))
}

// Clear erases all entries, keeping the reserved capacity and the frozen geometry. No other verb
// may overlap it on the same handle.
func (on DenseIndexOn) Clear() error {
	_, err := block(on.clear(C.usearch_completion_blocking_k))
	return err
}

// Clear submits DenseIndexOn.Clear.
func (on DenseIndexDeferredOn) Clear() *Deferred[struct{}] {
	return submitted(on.clear(C.usearch_completion_deferred_k))
}

// Describe reports the frozen metadata and the current counters in one call.
func (on DenseIndexOn) Describe() (DenseIndexDescription, error) {
	return block(on.describe(C.usearch_completion_blocking_k))
}

// Describe submits DenseIndexOn.Describe.
func (on DenseIndexDeferredOn) Describe() *Deferred[DenseIndexDescription] {
	return submitted(on.describe(C.usearch_completion_deferred_k))
}

// Count reports how many entries are stored under a key; zero means the index holds none.
func (on DenseIndexOn) Count(key Key) (uint, error) {
	return block(on.count(C.usearch_completion_blocking_k, key))
}

// Count submits DenseIndexOn.Count.
func (on DenseIndexDeferredOn) Count(key Key) *Deferred[uint] {
	return submitted(on.count(C.usearch_completion_deferred_k, key))
}

// Contains reports whether the index holds at least one entry under a key, as a nonzero Count.
func (on DenseIndexOn) Contains(key Key) (bool, error) {
	counted, err := on.Count(key)
	return counted != 0, err
}

// Get exports every vector stored under a key into arena, one row per entry sharing the key, in
// insertion order, converted to the outputsScalar kind - one call, however many there turn out to be.
//
// The call rewinds arena, and the rows stay readable until it serves another call.
func (on DenseIndexOn) Get(key Key, outputsScalar Scalar, arena *Arena) (DenseVectors, error) {
	return block(on.get(C.usearch_completion_blocking_k, key, outputsScalar, arena))
}

// Get submits DenseIndexOn.Get.
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
// none of its entries, and leaves the handle refusing every later mutation.
func (on DenseIndexOn) Insert(keys []Key, vectors RowsView, config *DenseInsertConfig) error {
	_, err := block(on.insert(C.usearch_completion_blocking_k, keys, vectors, config))
	return err
}

// Insert submits DenseIndexOn.Insert.
func (on DenseIndexDeferredOn) Insert(keys []Key, vectors RowsView, config *DenseInsertConfig) *Deferred[struct{}] {
	return submitted(on.insert(C.usearch_completion_deferred_k, keys, vectors, config))
}

// Refine re-selects the neighbors of every stored entry against the finished graph, improving
// edges chosen when the graph around them was still incomplete.
//
// A build step rather than a repair: a cheap insertion pass followed by one refinement reaches
// a better graph, in less total time, than a single expensive insertion pass - so a wide
// expansion factor belongs here rather than in Insert. A caller repeats rounds until the report's
// Verdict reads DenseRefineConverged, so no change rate is theirs to pick.
//
// On a RAM handle searches may run throughout; insertions may not. A stop answers StatusCancelled
// and leaves the graph correct and merely less improved, while a config's Plan keeps the round live
// for the next call.
func (on DenseIndexOn) Refine(config *DenseRefineConfig) (DenseRefineReport, error) {
	return block(on.refine(C.usearch_completion_blocking_k, config))
}

// Refine submits DenseIndexOn.Refine.
func (on DenseIndexDeferredOn) Refine(config *DenseRefineConfig) *Deferred[DenseRefineReport] {
	return submitted(on.refine(C.usearch_completion_deferred_k, config))
}

// Health measures how much of the graph a search reaches from the root, swept level by level.
// Read-only, so concurrent insertions merely leave fresh entries uncounted.
func (on DenseIndexOn) Health() (DenseReachability, error) {
	return block(on.health(C.usearch_completion_blocking_k))
}

// Health submits DenseIndexOn.Health.
func (on DenseIndexDeferredOn) Health() *Deferred[DenseReachability] {
	return submitted(on.health(C.usearch_completion_deferred_k))
}

// Search resolves a matrix of queries into up to wanted nearest matches apiece, written into arena
// - the native batch path, parallelized internally, and filtered when the config holds a predicate.
//
// Never writes to the graph, so any number of these may overlap with concurrent insertions. A
// zero wanted walks nothing and lands only the counts, every one zero.
func (on DenseIndexOn) Search(queries RowsView, wanted uint, arena *Arena, config *DenseSearchConfig) (MatchesView, error) {
	return block(on.search(C.usearch_completion_blocking_k, queries, wanted, arena, config))
}

// Search submits DenseIndexOn.Search.
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
// See https://arxiv.org/abs/2004.12832 - ColBERT, which introduced this scoring.
func (on DenseIndexOn) Maxsim(queries RowsView, rowsPerQuery, wanted uint, arena *Arena,
	config *DenseMaxsimConfig) (DenseProducts, error) {
	return block(on.maxsim(C.usearch_completion_blocking_k, queries, rowsPerQuery, wanted, arena, config))
}

// Maxsim submits DenseIndexOn.Maxsim.
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
// land in arena, as Get's rows do.
func (on DenseIndexOn) Multiply(left RowsView, arena *Arena, config *DenseMultiplyConfig) (DenseProducts, error) {
	return block(on.multiply(C.usearch_completion_blocking_k, left, arena, config))
}

// Multiply submits DenseIndexOn.Multiply.
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
// See https://arxiv.org/abs/2205.14135 - Flash Attention, whose recurrence this reuses.
func (on DenseIndexOn) Attend(queries RowsView, lookup ValueLookup, arena *Arena,
	config *DenseAttendConfig) (DenseVectors, error) {
	return block(on.attend(C.usearch_completion_blocking_k, queries, lookup, arena, config))
}

// Attend submits DenseIndexOn.Attend.
func (on DenseIndexDeferredOn) Attend(queries RowsView, lookup ValueLookup, arena *Arena,
	config *DenseAttendConfig) *Deferred[DenseVectors] {
	return submitted(on.attend(C.usearch_completion_deferred_k, queries, lookup, arena, config))
}

// Reshard moves the entries of sources into targets, keeping their links rather than re-inserting
// their vectors, then runs a round on every target that took entries. Several sources gather into
// one target, itself among them; one source cuts into one graph-local part per target; and one
// index as its only source and target recuts its own shards in place into as few as fit the
// placement's ShardCapacity. The receiver contributes only its executor.
//
// Every handle is checked before the first entry moves, so a refusal moves nothing, and every
// handle stays the caller's: a source the targets do not name ends empty and live. Readers of a RAM
// target keep searching throughout; a VRAM one admits no overlapping search.
func (on DenseIndexOn) Reshard(sources, targets []*DenseIndex, config *DenseReshardConfig) (DenseRefineReport, error) {
	return block(on.reshard(C.usearch_completion_blocking_k, sources, targets, config))
}

// Reshard submits DenseIndexOn.Reshard.
func (on DenseIndexDeferredOn) Reshard(sources, targets []*DenseIndex,
	config *DenseReshardConfig) *Deferred[DenseRefineReport] {
	return submitted(on.reshard(C.usearch_completion_deferred_k, sources, targets, config))
}

// RefinePlanFinish closes a live refine plan and frees it whatever the call answers, reporting what
// the whole plan did to the graph; a round a stop left in hand closes as that stop would have.
func (on DenseIndexOn) RefinePlanFinish(plan *DenseRefinePlan) (DenseRefineReport, error) {
	return block(on.refinePlanFinish(C.usearch_completion_blocking_k, plan))
}

// RefinePlanFinish submits DenseIndexOn.RefinePlanFinish.
func (on DenseIndexDeferredOn) RefinePlanFinish(plan *DenseRefinePlan) *Deferred[DenseRefineReport] {
	return submitted(on.refinePlanFinish(C.usearch_completion_deferred_k, plan))
}

// ReshardPlanFinish closes a live reshard plan on this receiver's executor and frees it
// whatever the call answers, reporting what the whole plan did; a plan whose entries have not
// all moved runs no round and reports zeros.
func (on DenseIndexOn) ReshardPlanFinish(plan *DenseReshardPlan) (DenseRefineReport, error) {
	return block(on.reshardPlanFinish(C.usearch_completion_blocking_k, plan))
}

// ReshardPlanFinish submits DenseIndexOn.ReshardPlanFinish.
func (on DenseIndexDeferredOn) ReshardPlanFinish(plan *DenseReshardPlan) *Deferred[DenseRefineReport] {
	return submitted(on.reshardPlanFinish(C.usearch_completion_deferred_k, plan))
}

// Sample draws up to wanted keys around the config's Focus, or across the whole index, each
// hanging from a shown or earlier draw and placed in three dimensions beside the shown ones, so a
// view grows one connected sample as it zooms; a larger wanted draws a superset. Only RAM samples
// yet.
func (on DenseIndexOn) Sample(wanted uint, arena *Arena, config *DenseSampleConfig) (DenseDraws, error) {
	return block(on.sample(C.usearch_completion_blocking_k, wanted, arena, config))
}

// Sample submits DenseIndexOn.Sample.
func (on DenseIndexDeferredOn) Sample(wanted uint, arena *Arena, config *DenseSampleConfig) *Deferred[DenseDraws] {
	return submitted(on.sample(C.usearch_completion_deferred_k, wanted, arena, config))
}

// Save writes the index under a path, in the layout its extension picks - one .safegraphs
// monolith, or a .safegraphs.json manifest beside its shards. Demands a mutation-free window;
// concurrent readers keep searching throughout.
func (on DenseIndexOn) Save(path string, config *DenseSaveConfig) error {
	_, err := block(on.save(C.usearch_completion_blocking_k, path, config))
	return err
}

// Save submits DenseIndexOn.Save.
func (on DenseIndexDeferredOn) Save(path string, config *DenseSaveConfig) *Deferred[struct{}] {
	return submitted(on.save(C.usearch_completion_deferred_k, path, config))
}

// SaveBuffer serializes the index as one monolith into arena, a size only the call itself can
// discover.
func (on DenseIndexOn) SaveBuffer(arena *Arena, config *DenseSaveBufferConfig) (Monolith, error) {
	return block(on.saveBuffer(C.usearch_completion_blocking_k, arena, config))
}

// SaveBuffer submits DenseIndexOn.SaveBuffer.
func (on DenseIndexDeferredOn) SaveBuffer(arena *Arena, config *DenseSaveBufferConfig) *Deferred[Monolith] {
	return submitted(on.saveBuffer(C.usearch_completion_deferred_k, arena, config))
}

// endregion: Receivers

// region: Calls

func (receiver denseReceiver) reserve(completion C.usearch_completion_t, capacity uint) *callWithResult[struct{}] {
	call := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	C.usearch_dense_reserve(&C.usearch_dense_reserve_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome,
		completion: completion,
		capacity:   C.size_t(capacity),
	})
	return call
}

func (receiver denseReceiver) clear(completion C.usearch_completion_t) *callWithResult[struct{}] {
	call := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	C.usearch_dense_clear(&C.usearch_dense_clear_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome,
		completion: completion,
	})
	return call
}

func (receiver denseReceiver) describe(completion C.usearch_completion_t) *callWithResult[DenseIndexDescription] {
	call := open[DenseIndexDescription]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	C.usearch_dense_describe(&C.usearch_dense_describe_t{
		index:       index,
		executor:    executor,
		outcome:     call.outcome,
		completion:  completion,
		description: (*C.usearch_dense_index_description_t)(unsafe.Pointer(call.result)),
	})
	return call
}

func (receiver denseReceiver) count(completion C.usearch_completion_t, key Key) *callWithResult[uint] {
	call := open[uint]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	C.usearch_dense_count(&C.usearch_dense_count_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome,
		completion: completion,
		key:        key.toRaw(),
		entries:    (*C.size_t)(unsafe.Pointer(call.result)),
	})
	return call
}

func (receiver denseReceiver) get(completion C.usearch_completion_t, key Key, outputsScalar Scalar,
	arena *Arena) *callWithResult[DenseVectors] {
	call := open[DenseVectors]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	C.usearch_dense_get(&C.usearch_dense_get_t{
		index:          index,
		executor:       executor,
		outcome:        call.outcome,
		completion:     completion,
		key:            key.toRaw(),
		vectors:        &call.result.raw,
		written:        (*C.size_t)(unsafe.Pointer(&call.result.rows)),
		arena:          arena.lower(),
		outputs_scalar: C.usearch_scalar_t(outputsScalar),
	})
	return call
}

func (receiver denseReceiver) insert(completion C.usearch_completion_t, keys []Key, vectors RowsView,
	config *DenseInsertConfig) *callWithResult[struct{}] {
	if config == nil {
		config = new(DenseInsertConfig)
	}
	call := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	// The call has one count, so only this side sees the keys and the rows disagree.
	if uint(len(keys)) != vectors.rows {
		return call.refuse(fail(StatusShapeMismatch, "the key count does not match the vector row count"))
	}
	C.usearch_dense_insert(&C.usearch_dense_insert_t{
		index:            index,
		executor:         executor,
		outcome:          call.outcome,
		completion:       completion,
		count:            C.size_t(vectors.rows),
		keys:             call.keys(keys),
		vectors:          vectors.lower(&call.call),
		expansion_factor: C.size_t(config.ExpansionFactor),
		progress:         call.progress(config.Progress, config.ProgressSteps),
		stats:            call.stats(config.Stats),
	})
	return call
}

func (receiver denseReceiver) refine(completion C.usearch_completion_t,
	config *DenseRefineConfig) *callWithResult[DenseRefineReport] {
	if config == nil {
		config = new(DenseRefineConfig)
	}
	call := open[DenseRefineReport]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	plan, phase := config.Plan.lower(&call.call)
	C.usearch_dense_refine(&C.usearch_dense_refine_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome,
		completion: completion,
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(call.result)),
		refine: C.usearch_refine_options_t{
			expansion_factor: C.size_t(config.ExpansionFactor),
			memory_limit:     C.size_t(config.MemoryLimit),
		},
		progress: call.progress(config.Progress, config.ProgressSteps),
		stats:    call.stats(config.Stats),
		plan:     plan,
		phase:    phase,
	})
	return call
}

func (receiver denseReceiver) refinePlanFinish(completion C.usearch_completion_t,
	plan *DenseRefinePlan) *callWithResult[DenseRefineReport] {
	call := open[DenseRefineReport]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	C.usearch_dense_refine_plan_finish(&C.usearch_dense_refine_plan_finish_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome,
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(call.result)),
	})
	return call
}

func (receiver denseReceiver) health(completion C.usearch_completion_t) *callWithResult[DenseReachability] {
	call := open[DenseReachability]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	C.usearch_dense_health(&C.usearch_dense_health_t{
		index:        index,
		executor:     executor,
		outcome:      call.outcome,
		completion:   completion,
		reachability: (*C.usearch_reachability_t)(unsafe.Pointer(call.result)),
	})
	return call
}

func (receiver denseReceiver) search(completion C.usearch_completion_t, queries RowsView, wanted uint,
	arena *Arena, config *DenseSearchConfig) *callWithResult[MatchesView] {
	if config == nil {
		config = new(DenseSearchConfig)
	}
	call := open[MatchesView]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	call.result.queries = queries.rows
	C.usearch_dense_search(&C.usearch_dense_search_t{
		index:            index,
		executor:         executor,
		outcome:          call.outcome,
		completion:       completion,
		count:            C.size_t(queries.rows),
		queries:          queries.lower(&call.call),
		matches:          &call.result.raw,
		arena:            arena.lower(),
		wanted:           C.size_t(wanted),
		expansion_factor: C.size_t(config.ExpansionFactor),
		traversal:        C.usearch_traversal_t(config.Traversal),
		predicate:        call.predicate(config.Predicate),
		progress:         call.progress(config.Progress, config.ProgressSteps),
		stats:            call.stats(config.Stats),
	})
	return call
}

func (receiver denseReceiver) maxsim(completion C.usearch_completion_t, queries RowsView, rowsPerQuery, wanted uint,
	arena *Arena, config *DenseMaxsimConfig) *callWithResult[DenseProducts] {
	if config == nil {
		config = new(DenseMaxsimConfig)
	}
	call := open[DenseProducts]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	// The call counts queries, so only this side sees the matrix's rows fail to divide.
	if rowsPerQuery == 0 || queries.rows%rowsPerQuery != 0 {
		return call.refuse(fail(StatusShapeMismatch, "the query rows do not divide into whole queries"))
	}
	call.result.queries = queries.rows / rowsPerQuery
	C.usearch_dense_maxsim(&C.usearch_dense_maxsim_t{
		index:            index,
		executor:         executor,
		outcome:          call.outcome,
		completion:       completion,
		count:            C.size_t(call.result.queries),
		queries:          queries.lower(&call.call),
		rows_per_query:   C.size_t(rowsPerQuery),
		matches:          &call.result.raw,
		required:         (*C.size_t)(unsafe.Pointer(&call.result.required)),
		arena:            arena.lower(),
		wanted:           C.size_t(wanted),
		expansion_factor: C.size_t(config.ExpansionFactor),
		progress:         call.progress(config.Progress, config.ProgressSteps),
		stats:            call.stats(config.Stats),
	})
	return call
}

func (receiver denseReceiver) multiply(completion C.usearch_completion_t, left RowsView, arena *Arena,
	config *DenseMultiplyConfig) *callWithResult[DenseProducts] {
	if config == nil {
		config = new(DenseMultiplyConfig)
	}
	call := open[DenseProducts]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	call.result.queries = left.rows
	C.usearch_dense_multiply(&C.usearch_dense_multiply_t{
		index:            index,
		executor:         executor,
		outcome:          call.outcome,
		completion:       completion,
		count:            C.size_t(left.rows),
		left:             left.lower(&call.call),
		products:         &call.result.raw,
		required:         (*C.size_t)(unsafe.Pointer(&call.result.required)),
		arena:            arena.lower(),
		wanted:           C.size_t(config.Wanted),
		expansion_factor: C.size_t(config.ExpansionFactor),
		progress:         call.progress(config.Progress, config.ProgressSteps),
		stats:            call.stats(config.Stats),
	})
	return call
}

func (receiver denseReceiver) attend(completion C.usearch_completion_t, queries RowsView, lookup ValueLookup,
	arena *Arena, config *DenseAttendConfig) *callWithResult[DenseVectors] {
	if config == nil {
		config = new(DenseAttendConfig)
	}
	call := open[DenseVectors]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	// The carry pointer has no length, so only this side can see it fall short.
	if config.Carry != nil && uint(len(config.Carry)) < queries.rows {
		return call.refuse(fail(StatusShapeMismatch, "the softmax carry holds fewer entries than there are queries"))
	}
	var valueLookup C.usearch_value_lookup_t
	var valueLookupState unsafe.Pointer
	if lookup.Row != nil {
		call.pinner.Pin(lookup.Values.pointer)
		valueLookup, valueLookupState = C.usearch_go_looks_up_values(), call.callback(lookup)
	}
	call.result.rows = queries.rows
	C.usearch_dense_attend(&C.usearch_dense_attend_t{
		index:              index,
		executor:           executor,
		outcome:            call.outcome,
		completion:         completion,
		count:              C.size_t(queries.rows),
		queries:            queries.lower(&call.call),
		head_count:         C.size_t(config.HeadCount),
		value_lookup:       valueLookup,
		value_lookup_state: valueLookupState,
		carry:              (*C.usearch_dense_softmax_state_t)(unsafe.Pointer(pinSlice(&call.call, config.Carry))),
		outputs:            &call.result.raw,
		arena:              arena.lower(),
		outputs_scalar:     C.usearch_scalar_t(config.OutputsScalar),
		expansion_factor:   C.size_t(config.ExpansionFactor),
		score_scale:        C.float(config.ScoreScale),
		progress:           call.progress(config.Progress, config.ProgressSteps),
		stats:              call.stats(config.Stats),
	})
	return call
}

//export usearchLooksUpValues
func usearchLooksUpValues(state unsafe.Pointer, keys *C.usearch_key_t, count C.size_t, scalar C.usearch_scalar_t,
	columns C.size_t, rows *unsafe.Pointer) {
	lookup := (*cgo.Handle)(state).Value().(ValueLookup)
	values := lookup.Values
	found := unsafe.Slice(rows, count)
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

func (receiver denseReceiver) reshard(completion C.usearch_completion_t, sources, targets []*DenseIndex,
	config *DenseReshardConfig) *callWithResult[DenseRefineReport] {
	if config == nil {
		config = new(DenseReshardConfig)
	}
	call := open[DenseRefineReport]()
	// A reshard names its handles in its lists, so the receiver passes only its executor.
	executor, refusal := receiver.executor.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	var ram *C.usearch_ram_options_t
	if config.MemoryDomains != nil {
		placement := call.ram(config.MemoryDomains, 0)
		placement.shard_capacity = C.size_t(config.ShardCapacity)
		ram = pin(&call.call, &placement)
	}
	var vram *C.usearch_vram_options_t
	if config.Devices != nil {
		placement := call.vram(config.Devices, 0)
		placement.shard_capacity = C.size_t(config.ShardCapacity)
		vram = pin(&call.call, &placement)
	}
	var nvme *C.usearch_nvme_options_t
	if config.Paths != nil {
		placement := call.nvme(config.Paths, 0)
		nvme = pin(&call.call, &placement)
	}
	sourceHandles, targetHandles := denseHandles(sources), denseHandles(targets)
	plan, phase := config.Plan.lower(&call.call)
	C.usearch_dense_reshard(&C.usearch_dense_reshard_t{
		sources_count: C.size_t(len(sourceHandles)),
		sources:       pinSlice(&call.call, sourceHandles),
		targets_count: C.size_t(len(targetHandles)),
		targets:       pinSlice(&call.call, targetHandles),
		executor:      executor,
		outcome:       call.outcome,
		completion:    completion,
		ram:           ram,
		vram:          vram,
		nvme:          nvme,
		report:        (*C.usearch_refine_report_t)(unsafe.Pointer(call.result)),
		refine: C.usearch_refine_options_t{
			expansion_factor: C.size_t(config.ExpansionFactor),
			memory_limit:     C.size_t(config.MemoryLimit),
		},
		seed:     C.uint64_t(config.Seed),
		progress: call.progress(config.Progress, config.ProgressSteps),
		stats:    call.stats(config.Stats),
		plan:     plan,
		phase:    phase,
	})
	return call
}

// denseHandles lists the handles of indexes.
func denseHandles(indexes []*DenseIndex) []C.usearch_dense_index_t {
	handles := make([]C.usearch_dense_index_t, len(indexes))
	for position, index := range indexes {
		handles[position] = index.handle
	}
	return handles
}

func (receiver denseReceiver) reshardPlanFinish(completion C.usearch_completion_t,
	plan *DenseReshardPlan) *callWithResult[DenseRefineReport] {
	call := open[DenseRefineReport]()
	// A reshard names its handles in its lists, so the receiver passes only its executor.
	executor, refusal := receiver.executor.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	C.usearch_dense_reshard_plan_finish(&C.usearch_dense_reshard_plan_finish_t{
		executor:   executor,
		outcome:    call.outcome,
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(call.result)),
	})
	return call
}

func (receiver denseReceiver) sample(completion C.usearch_completion_t, wanted uint, arena *Arena,
	config *DenseSampleConfig) *callWithResult[DenseDraws] {
	if config == nil {
		config = new(DenseSampleConfig)
	}
	call := open[DenseDraws]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	// The call has one count for both, so only this side sees the keys and their places disagree.
	if len(config.Shown) != len(config.ShownCoordinates) {
		return call.refuse(fail(StatusShapeMismatch, "the shown keys and coordinates disagree in length"))
	}
	C.usearch_dense_sample(&C.usearch_dense_sample_t{
		index:             index,
		executor:          executor,
		outcome:           call.outcome,
		completion:        completion,
		shown_count:       C.size_t(len(config.Shown)),
		shown:             call.keys(config.Shown),
		shown_coordinates: call.points(config.ShownCoordinates),
		focus_count:       C.size_t(len(config.Focus)),
		focus:             call.keys(config.Focus),
		keys:              &call.result.keys,
		entry_ordinals:    &call.result.entryOrdinals,
		parents:           &call.result.parents,
		coordinates:       &call.result.coordinates,
		written:           (*C.size_t)(unsafe.Pointer(&call.result.written)),
		arena:             arena.lower(),
		wanted:            C.size_t(wanted),
		seed:              C.uint64_t(config.Seed),
		expansion_factor:  C.size_t(config.ExpansionFactor),
		progress:          call.progress(config.Progress, config.ProgressSteps),
		stats:             call.stats(config.Stats),
	})
	return call
}

func (receiver denseReceiver) save(completion C.usearch_completion_t, path string,
	config *DenseSaveConfig) *callWithResult[struct{}] {
	if config == nil {
		config = new(DenseSaveConfig)
	}
	call := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	C.usearch_dense_save(&C.usearch_dense_save_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome,
		completion: completion,
		path:       call.text(path),
		progress:   call.progress(config.Progress, config.ProgressSteps),
	})
	return call
}

func (receiver denseReceiver) saveBuffer(completion C.usearch_completion_t, arena *Arena,
	config *DenseSaveBufferConfig) *callWithResult[Monolith] {
	if config == nil {
		config = new(DenseSaveBufferConfig)
	}
	call := open[Monolith]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return call.refuse(refusal)
	}
	C.usearch_dense_save_buffer(&C.usearch_dense_save_buffer_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome,
		completion: completion,
		monolith:   &call.result.raw,
		arena:      arena.lower(),
		progress:   call.progress(config.Progress, config.ProgressSteps),
	})
	return call
}

// endregion: Calls

// region: Cursors & Streams

// DenseCursor is a walk over the index contents, pulled blocking on the executor of the
// receiver that opened it. Each shard's publication watermark is read as the walk enters that
// shard, so an entry once passed is never revisited.
type DenseCursor struct {
	handle   C.usearch_dense_cursor_t
	receiver denseReceiver
}

// Cursor opens a walk over the current contents. Without starts it yields every entry as stored; with them,
// the start keys' own entries first in order, then the entries those link to breadth-first, each once.
func (on DenseIndexOn) Cursor(starts ...Key) (*DenseCursor, error) {
	index, _, refusal := on.lower()
	if refusal != nil {
		return nil, refusal
	}
	call := open[C.usearch_dense_cursor_t]()
	C.usearch_dense_cursor_init(&C.usearch_dense_cursor_init_t{
		index:        index,
		outcome:      call.outcome,
		starts_count: C.size_t(len(starts)),
		starts:       (*C.usearch_key_t)(unsafe.Pointer(pinSlice(&call.call, starts))),
		cursor:       call.result,
	})
	handle, err := block(call)
	if err != nil {
		return nil, err
	}
	return &DenseCursor{handle: handle, receiver: on.denseReceiver}, nil
}

// NextKeys copies up to wanted next keys into arena; a batch of none means the walk is exhausted.
func (cursor *DenseCursor) NextKeys(wanted uint, arena *Arena) (DenseBatch, error) {
	return cursor.next(wanted, false, arena)
}

// NextLinks copies up to wanted next keys into arena, each with the keys it links to; a batch of none
// means the walk is exhausted.
func (cursor *DenseCursor) NextLinks(wanted uint, arena *Arena) (DenseBatch, error) {
	return cursor.next(wanted, true, arena)
}

// next pulls a batch, exporting each entry's links only when linked asks for them.
func (cursor *DenseCursor) next(wanted uint, linked bool, arena *Arena) (DenseBatch, error) {
	index, executor, refusal := cursor.receiver.lower()
	if refusal != nil {
		return DenseBatch{}, refusal
	}
	call := open[DenseBatch]()
	var links *C.usearch_keys_tape_view_t
	if linked {
		links = &call.result.links
	}
	C.usearch_dense_cursor_next(&C.usearch_dense_cursor_next_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome,
		completion: C.usearch_completion_blocking_k,
		cursor:     cursor.handle,
		keys:       &call.result.keys,
		links:      links,
		written:    (*C.size_t)(unsafe.Pointer(&call.result.written)),
		arena:      arena.lower(),
		wanted:     C.size_t(wanted),
	})
	return block(call)
}

// Close releases the cursor. It is idempotent.
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
type DenseSearchStream struct {
	handle   C.usearch_dense_search_stream_t
	receiver denseReceiver
}

// SearchStream opens a resumable search for a one-row query, so callers that post-filter
// aggressively pull until satisfied instead of guessing the wanted count upfront. No residency
// implements the stream yet, so the engine answers StatusUnsupported.
func (on DenseIndexOn) SearchStream(query RowsView, config *DenseSearchStreamInitConfig) (*DenseSearchStream, error) {
	if config == nil {
		config = new(DenseSearchStreamInitConfig)
	}
	index, _, refusal := on.lower()
	// The call takes one row and no count, so only this side sees a matrix holding more.
	if refusal == nil && query.rows != 1 {
		refusal = fail(StatusShapeMismatch, "a stream walks exactly one query row")
	}
	if refusal != nil {
		return nil, refusal
	}
	call := open[C.usearch_dense_search_stream_t]()
	C.usearch_dense_search_stream_init(&C.usearch_dense_search_stream_init_t{
		stream:           call.result,
		index:            index,
		outcome:          call.outcome,
		query:            query.lower(&call.call),
		expansion_factor: C.size_t(config.ExpansionFactor),
	})
	handle, err := block(call)
	if err != nil {
		return nil, err
	}
	return &DenseSearchStream{handle: handle, receiver: on.denseReceiver}, nil
}

// Next pulls up to wanted next-nearest matches into arena, as the one query of a search; a query
// with none means the walk is drained.
func (stream *DenseSearchStream) Next(wanted uint, arena *Arena) (MatchesView, error) {
	index, executor, refusal := stream.receiver.lower()
	if refusal != nil {
		return MatchesView{}, refusal
	}
	call := open[MatchesView]()
	call.result.queries = 1
	C.usearch_dense_search_stream_next(&C.usearch_dense_search_stream_next_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome,
		completion: C.usearch_completion_blocking_k,
		stream:     stream.handle,
		matches:    &call.result.raw,
		arena:      arena.lower(),
		wanted:     C.size_t(wanted),
	})
	return block(call)
}

// Close releases the stream and the walk state it keeps. It is idempotent.
func (stream *DenseSearchStream) Close() error {
	if stream == nil || stream.handle == nil {
		return nil
	}
	C.usearch_dense_search_stream_free(stream.handle)
	stream.handle = nil
	return nil
}

// endregion: Cursors & Streams
