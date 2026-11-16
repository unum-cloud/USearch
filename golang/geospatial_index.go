// Similarity Search over observations on and above the Earth, under sight and view-cone relations.
//
// File: golang/geospatial_index.go
// Author: Ash Vardanian
// Date: November 16, 2026

// Every verb is a batch over a slice of observations, laid out as the library reads them, so a
// slice crosses with no copy and a single observation is a batch of one. Every score is the
// straight-line chord between two positions, in metres. Reports, traversal stats, cursor batches
// and draws are the dense modality's own types, since the library fills them through the same C
// structures. Terrain is not bound, so a handle answers the visible relation with
// StatusUnsupported.

package usearch

import (
	"math"
	"unsafe"
)

/*
// `nocallback` only where the argument structure carries no callback and no executor; `noescape`
// everywhere, since submission copies the structure and everything it names is pinned.
#cgo noescape usearch_geospatial_init_ram
#cgo nocallback usearch_geospatial_init_ram
#cgo noescape usearch_geospatial_init_vram
#cgo nocallback usearch_geospatial_init_vram
#cgo noescape usearch_geospatial_view_ram
#cgo nocallback usearch_geospatial_view_ram
#cgo noescape usearch_geospatial_cursor_init
#cgo nocallback usearch_geospatial_cursor_init
#cgo noescape usearch_geospatial_view_nvme
#cgo noescape usearch_geospatial_reserve
#cgo noescape usearch_geospatial_clear
#cgo noescape usearch_geospatial_describe
#cgo noescape usearch_geospatial_count
#cgo noescape usearch_geospatial_get
#cgo noescape usearch_geospatial_health
#cgo noescape usearch_geospatial_metadata_read
#cgo noescape usearch_geospatial_metadata_read_buffer
#cgo noescape usearch_geospatial_load_ram
#cgo noescape usearch_geospatial_load_vram
#cgo noescape usearch_geospatial_load_ram_buffer
#cgo noescape usearch_geospatial_load_vram_buffer
#cgo noescape usearch_geospatial_insert
#cgo noescape usearch_geospatial_refine
#cgo noescape usearch_geospatial_refine_plan_finish
#cgo noescape usearch_geospatial_search
#cgo noescape usearch_geospatial_within
#cgo noescape usearch_geospatial_reshard
#cgo noescape usearch_geospatial_reshard_plan_finish
#cgo noescape usearch_geospatial_sample
#cgo noescape usearch_geospatial_save
#cgo noescape usearch_geospatial_save_buffer
#cgo noescape usearch_geospatial_cursor_next

#include "shim.h"
#include "usearch/geospatial_index.h"
*/
import "C"

// region: Geometry

// GeospatialObservation is one observation as devices record it, every field in its natural unit
// and NaN wherever nothing was recorded: an unknown accuracy counts as exact, an unknown view as
// looking everywhere without limit, and an unknown time as always. Its layout is the library's
// own, so a slice of them crosses as it is.
type GeospatialObservation struct {
	// LatitudeDegrees is degrees north on WGS-84, south negative.
	LatitudeDegrees float64
	// LongitudeDegrees is degrees east on WGS-84, west negative.
	LongitudeDegrees float64
	// AltitudeMetres is metres above the WGS-84 ellipsoid.
	AltitudeMetres float64
	// AccuracyMetres is one standard deviation of the horizontal position.
	AccuracyMetres float64
	// HeadingDegrees is the direction the device looked, clockwise from true north.
	HeadingDegrees float64
	// PitchDegrees is how far above the horizon the device looked, below it negative.
	PitchDegrees float64
	// HeadingAccuracyDegrees is one standard deviation of the heading.
	HeadingAccuracyDegrees float64
	// FieldOfViewDegrees is the full field of view.
	FieldOfViewDegrees float64
	// RangeMetres is how far the device saw, along the slant.
	RangeMetres float64
	// BeginSeconds is the first instant the observation covers, in Unix seconds, UTC.
	BeginSeconds float64
	// EndSeconds is the last instant the observation covers, in Unix seconds, UTC.
	EndSeconds float64
}

// GeospatialAt is an observation of a position alone, every other field unrecorded.
func GeospatialAt(latitudeDegrees, longitudeDegrees float64) GeospatialObservation {
	unrecorded := math.NaN()
	return GeospatialObservation{
		LatitudeDegrees: latitudeDegrees, LongitudeDegrees: longitudeDegrees,
		AltitudeMetres: unrecorded, AccuracyMetres: unrecorded, HeadingDegrees: unrecorded,
		PitchDegrees: unrecorded, HeadingAccuracyDegrees: unrecorded, FieldOfViewDegrees: unrecorded,
		RangeMetres: unrecorded, BeginSeconds: unrecorded, EndSeconds: unrecorded,
	}
}

// GeospatialRelation picks which entries count as matches of a query.
type GeospatialRelation int32

const (
	// GeospatialRelationUnknown tags a relation the library could not name.
	GeospatialRelationUnknown GeospatialRelation = C.usearch_geospatial_relation_unknown_k
	// GeospatialRelationNear admits entries whose positions are close, widened by both accuracies.
	GeospatialRelationNear GeospatialRelation = C.usearch_geospatial_relation_near_k
	// GeospatialRelationVisible admits entries with an unobstructed line of sight over the terrain.
	GeospatialRelationVisible GeospatialRelation = C.usearch_geospatial_relation_visible_k
	// GeospatialRelationSees admits entries the query's view cone holds.
	GeospatialRelationSees GeospatialRelation = C.usearch_geospatial_relation_sees_k
	// GeospatialRelationSeen admits entries whose view cone holds the query.
	GeospatialRelationSeen GeospatialRelation = C.usearch_geospatial_relation_seen_k
)

// GeospatialOptions holds the construction options of a geospatial index, the fields
// GeospatialInitRam and GeospatialInitVram share.
type GeospatialOptions struct {
	// Connectivity bounds the connections per node in the graph; zero derives.
	Connectivity uint
	// ValuesPerKey bounds how many entries may share one key; zero is unbounded.
	ValuesPerKey uint
	// CPUClassCeiling is the newest class dispatch may pick, intersected with what the machine runs,
	// so a class the machine lacks narrows rather than refuses; the zero value means no ceiling.
	CPUClassCeiling CPUClass
	// GPUClassCeiling is the newest class dispatch may pick, capped below the best class every named
	// device loads; a ceiling no named device loads is refused as unsupported; the zero value means no
	// ceiling.
	GPUClassCeiling GPUClass
}

// GeospatialMetadata is the connectivity a checkpoint froze.
type GeospatialMetadata struct {
	// Connectivity is the connections per node the graph was built with.
	Connectivity uint
}

// GeospatialIndexDescription is a live index's frozen metadata and current counters, each read
// once, so under concurrent insertions they may come from slightly different moments.
type GeospatialIndexDescription struct {
	// Metadata is the frozen connectivity, as a checkpoint reports it.
	Metadata GeospatialMetadata
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

// GeospatialKeyStats is what Count reports about one key.
type GeospatialKeyStats struct {
	// Entries counts the observations stored under the key; zero means the index holds none.
	Entries uint
}

// GeospatialObservations is the observations Get wrote into its arena, readable until the arena
// serves another call.
type GeospatialObservations struct {
	raw     *C.usearch_geospatial_observation_t
	written uint
}

// Written reports how many observations the call wrote; zero when the key is absent.
func (observations GeospatialObservations) Written() uint { return observations.written }

// Observations borrows the observations from the arena, in insertion order.
func (observations GeospatialObservations) Observations() []GeospatialObservation {
	return unsafe.Slice((*GeospatialObservation)(unsafe.Pointer(observations.raw)), observations.written)
}

// endregion: Geometry

// region: Configs

// GeospatialSearchConfig carries the optional settings of Search. A zero value derives every
// setting.
type GeospatialSearchConfig struct {
	// Metres is the radius around each query; zero ranks by distance alone.
	Metres float64
	// Seconds is the slack on both ends of the query's time interval.
	Seconds float64
	// ExpansionFactor is the breadth of the candidates the walk explores; zero derives, and a
	// selective relation wants it wider.
	ExpansionFactor uint
	// Traversal walks the graph by default; TraversalScan judges every entry exactly, on RAM and
	// NVMe handles.
	Traversal Traversal
	// Predicate admits only the keys it approves; nil admits everything.
	Predicate Predicate
	// Progress reports the queries resolved and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
}

// GeospatialWithinConfig carries the optional settings of Within. A zero value derives every
// setting.
type GeospatialWithinConfig struct {
	// Seconds is the slack on both ends of the query's time interval.
	Seconds float64
	// Traversal walks the graph completely by default; TraversalScan judges every entry exactly, on
	// RAM and NVMe handles.
	Traversal Traversal
	// Predicate admits only the keys it approves; nil admits everything.
	Predicate Predicate
	// Progress reports the queries resolved and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
}

// GeospatialViewNvmeConfig carries the optional settings of GeospatialViewNvme.
type GeospatialViewNvmeConfig struct {
	// Executor is the NVMe executor whose rings read the checkpoint's headers; nil reads on the
	// calling goroutine.
	Executor *Executor
	// Paths are the devices or directories to serve the parts from; empty serves each where its
	// manifest points.
	Paths []string
	// CPUClassCeiling is the newest class dispatch may pick, intersected with what the machine runs,
	// so a class the machine lacks narrows rather than refuses; the zero value means no ceiling.
	CPUClassCeiling CPUClass
}

// GeospatialRefineConfig carries the optional settings of Refine. A zero value derives every
// setting.
type GeospatialRefineConfig struct {
	// ExpansionFactor is the breadth of the candidates the re-selection walk explores; zero derives.
	ExpansionFactor uint
	// MemoryLimit is the bytes a loaded NVMe handle holds for the one part it rewrites; zero derives.
	MemoryLimit uint
	// Progress reports each phase's units at its barriers and stops there wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
	// Plan keeps the round live across calls: nil runs one round to its end, a fresh plan opens one
	// that a stop leaves live, and a live one continues where it stopped, or starts the next round.
	Plan *GeospatialRefinePlan
}

// GeospatialReshardConfig carries the optional settings of Reshard. A zero value derives every
// setting.
type GeospatialReshardConfig struct {
	// Seed picks the part roots of a source split among several targets: equal seeds over an
	// unchanged source split equally.
	Seed uint64
	// ExpansionFactor is the breadth of the candidates of the closing rounds; zero derives.
	ExpansionFactor uint
	// MemoryLimit is the bytes a loaded NVMe target holds for the one part it rewrites; zero derives.
	MemoryLimit uint
	// Progress reports the closing rounds' units at their barriers; a stop never undoes a move.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *DenseTraversalStats
	// Plan keeps the reshard live across calls: nil moves and runs to the end, a fresh plan opens one
	// that a stop leaves live, and a live one continues where it stopped.
	Plan *GeospatialReshardPlan
}

// GeospatialRefinePlan is one refine kept live across calls: the round in hand, and the root every
// round's health checks share. The zero value opens on the first Refine naming it, and
// RefinePlanFinish closes it; while it lives the index admits no insertion and no save, and each
// call names the index and executor it was opened with.
type GeospatialRefinePlan = livePlan[C.usearch_geospatial_refine_plan_t]

// GeospatialReshardPlan is one reshard kept live across calls: its own sources and targets, the
// entries still to move, then the rounds of every target that took entries. The zero value opens
// on the first Reshard naming it, and ReshardPlanFinish closes it; while it lives every handle it
// names admits no insertion and no save, and each call names the executor it was opened with.
type GeospatialReshardPlan = livePlan[C.usearch_geospatial_reshard_plan_t]

// endregion: Configs

// region: Index

// GeospatialIndex is a Similarity Search Index over observations in space and time: photos, video
// frames, drone footage and sensor fixes, each where a device was, how sure it was, which way it
// looked, how far it saw, and when.
//
// A handle may be shared across goroutines. On a RAM handle, insertions and searches overlap
// freely, and serialization demands a mutation-free window while readers keep searching.
//
// Its verbs come from the embedded GeospatialIndexOn and run on the calling goroutine; On runs the
// same verbs blocking on an executor, and Deferred submits them there.
type GeospatialIndex struct {
	GeospatialIndexOn
	handle C.usearch_geospatial_index_t
}

// geospatialIndexOf wraps the handle a lifecycle verb wrote.
func geospatialIndexOf(handle C.usearch_geospatial_index_t, err error) (*GeospatialIndex, error) {
	if err != nil {
		return nil, err
	}
	index := &GeospatialIndex{handle: handle}
	index.GeospatialIndexOn = GeospatialIndexOn{geospatialReceiver{index: index}}
	return index, nil
}

// On is the receiver whose verbs block on executor. Binding-only.
func (index *GeospatialIndex) On(executor *Executor) GeospatialIndexOn {
	return GeospatialIndexOn{geospatialReceiver{index: index, executor: executor}}
}

// Deferred is the receiver whose verbs submit their calls to executor, each answering a Deferred
// from the same arguments. Binding-only.
func (index *GeospatialIndex) Deferred(executor *Executor) GeospatialIndexDeferredOn {
	return GeospatialIndexDeferredOn{geospatialReceiver{index: index, executor: executor}}
}

// Close releases the index, waiting for every deferred call naming it. It is idempotent, and a
// closed handle must not be used again.
func (index *GeospatialIndex) Close() error {
	if index == nil || index.handle == nil {
		return nil
	}
	C.usearch_geospatial_free(index.handle)
	index.handle = nil
	return nil
}

// GeospatialInitRam creates an empty, mutable index in host memory, one shard per NUMA node
// memoryDomains names; empty derives every node the process may use.
func GeospatialInitRam(options GeospatialOptions, memoryDomains []int32) (*GeospatialIndex, error) {
	lent := open[C.usearch_geospatial_index_t]()
	C.usearch_geospatial_init_ram(&C.usearch_geospatial_init_ram_t{
		index:          lent.result,
		outcome:        lent.outcome,
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		ram:            lent.ram(memoryDomains, options.CPUClassCeiling),
	})
	return geospatialIndexOf(block(lent))
}

// GeospatialInitVram creates an empty, mutable index in device memory, sharded across the CUDA
// ordinals devices names and never replicated; empty derives one free device.
func GeospatialInitVram(options GeospatialOptions, devices []int32) (*GeospatialIndex, error) {
	lent := open[C.usearch_geospatial_index_t]()
	C.usearch_geospatial_init_vram(&C.usearch_geospatial_init_vram_t{
		index:          lent.result,
		outcome:        lent.outcome,
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		vram:           lent.vram(devices, options.GPUClassCeiling),
	})
	return geospatialIndexOf(block(lent))
}

// GeospatialLoadRam copies a checkpoint into a new, mutable index in host memory. The path names a
// manifest of a sharded checkpoint or a single .safegraphs monolith.
func GeospatialLoadRam(path string, config *DenseLoadRamConfig) (*GeospatialIndex, error) {
	if config == nil {
		config = new(DenseLoadRamConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_geospatial_index_t]()
	C.usearch_geospatial_load_ram(&C.usearch_geospatial_load_ram_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		path:           lent.text(path),
		ram:            lent.ram(config.MemoryDomains, config.CPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return geospatialIndexOf(block(lent))
}

// GeospatialLoadVram copies a checkpoint into a new, mutable index in device memory, with the
// placement rules of GeospatialLoadRam.
func GeospatialLoadVram(path string, config *DenseLoadVramConfig) (*GeospatialIndex, error) {
	if config == nil {
		config = new(DenseLoadVramConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_geospatial_index_t]()
	C.usearch_geospatial_load_vram(&C.usearch_geospatial_load_vram_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		path:           lent.text(path),
		vram:           lent.vram(config.Devices, config.GPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return geospatialIndexOf(block(lent))
}

// GeospatialLoadRamBuffer copies a monolith from a buffer into a new, mutable index in host memory,
// as GeospatialLoadRam does a path.
func GeospatialLoadRamBuffer(buffer []byte, config *DenseLoadRamBufferConfig) (*GeospatialIndex, error) {
	if config == nil {
		config = new(DenseLoadRamBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_geospatial_index_t]()
	C.usearch_geospatial_load_ram_buffer(&C.usearch_geospatial_load_ram_buffer_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		buffer:         lent.bytes(buffer),
		ram:            lent.ram(config.MemoryDomains, config.CPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return geospatialIndexOf(block(lent))
}

// GeospatialLoadVramBuffer copies a monolith from a buffer into a new, mutable index in device
// memory.
func GeospatialLoadVramBuffer(buffer []byte, config *DenseLoadVramBufferConfig) (*GeospatialIndex, error) {
	if config == nil {
		config = new(DenseLoadVramBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_geospatial_index_t]()
	C.usearch_geospatial_load_vram_buffer(&C.usearch_geospatial_load_vram_buffer_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		buffer:         lent.bytes(buffer),
		vram:           lent.vram(config.Devices, config.GPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return geospatialIndexOf(block(lent))
}

// GeospatialViewRam serves a checkpoint in place through a memory mapping, as a read-only index
// whose mutating verbs report StatusReadOnly; the files must stay unmodified while it lives.
func GeospatialViewRam(path string) (*GeospatialIndex, error) {
	lent := open[C.usearch_geospatial_index_t]()
	C.usearch_geospatial_view_ram(&C.usearch_geospatial_view_ram_t{
		index:   lent.result,
		outcome: lent.outcome,
		path:    lent.text(path),
	})
	return geospatialIndexOf(block(lent))
}

// GeospatialViewNvme serves a checkpoint in place from NVMe drives, read through the executor's
// submission rings, as a read-only index.
func GeospatialViewNvme(path string, config *GeospatialViewNvmeConfig) (*GeospatialIndex, error) {
	if config == nil {
		config = new(GeospatialViewNvmeConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_geospatial_index_t]()
	C.usearch_geospatial_view_nvme(&C.usearch_geospatial_view_nvme_t{
		index:    lent.result,
		executor: executor,
		outcome:  lent.outcome,
		path:     lent.text(path),
		nvme:     lent.nvme(config.Paths, config.CPUClassCeiling),
	})
	return geospatialIndexOf(block(lent))
}

// GeospatialMetadataRead reads a checkpoint's connectivity without loading it.
func GeospatialMetadataRead(path string, config *DenseMetadataReadConfig) (GeospatialMetadata, error) {
	if config == nil {
		config = new(DenseMetadataReadConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return GeospatialMetadata{}, refusal
	}
	lent := open[GeospatialMetadata]()
	C.usearch_geospatial_metadata_read(&C.usearch_geospatial_metadata_read_t{
		executor: executor,
		outcome:  lent.outcome,
		path:     lent.text(path),
		metadata: (*C.usearch_geospatial_metadata_t)(unsafe.Pointer(lent.result)),
	})
	return block(lent)
}

// GeospatialMetadataReadBuffer reads the connectivity out of a serialized monolith buffer.
func GeospatialMetadataReadBuffer(buffer []byte, config *DenseMetadataReadBufferConfig) (GeospatialMetadata, error) {
	if config == nil {
		config = new(DenseMetadataReadBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return GeospatialMetadata{}, refusal
	}
	lent := open[GeospatialMetadata]()
	C.usearch_geospatial_metadata_read_buffer(&C.usearch_geospatial_metadata_read_buffer_t{
		executor: executor,
		outcome:  lent.outcome,
		buffer:   lent.bytes(buffer),
		metadata: (*C.usearch_geospatial_metadata_t)(unsafe.Pointer(lent.result)),
	})
	return block(lent)
}

// endregion: Index

// region: Receivers

// geospatialReceiver is an index and the executor its verbs run on, nil for the calling goroutine.
type geospatialReceiver struct {
	index    *GeospatialIndex
	executor *Executor
}

// lower reports the handles a verb hands across, refusing a closed index before C dereferences it.
func (receiver geospatialReceiver) lower() (C.usearch_geospatial_index_t, C.usearch_executor_t, *Error) {
	if receiver.index == nil || receiver.index.handle == nil {
		return nil, nil, fail(StatusUninitialized, "the index is closed or was never initialized")
	}
	executor, refusal := receiver.executor.lower()
	return receiver.index.handle, executor, refusal
}

// GeospatialIndexOn is the receiver whose verbs block until their call is final: on the calling
// goroutine when embedded in a GeospatialIndex, and on an executor when GeospatialIndex.On makes
// it. Binding-only.
type GeospatialIndexOn struct {
	geospatialReceiver
}

// GeospatialIndexDeferredOn is the receiver whose verbs submit their calls to an executor and
// answer a Deferred, offering every verb of GeospatialIndexOn with the same arguments but the
// blocking-only sugar and cursors. Binding-only.
//
// The library reads a deferred call's inputs until the call is final, so the slices a verb is
// handed must not change until then; its arena is borrowed, as a blocking verb borrows it.
type GeospatialIndexDeferredOn struct {
	geospatialReceiver
}

// Reserve claims capacity for a number of incoming observations ahead of insertion. The capacity is
// the desired total, including what the index holds.
func (on GeospatialIndexOn) Reserve(entries uint) error {
	_, err := block(on.reserve(C.usearch_completion_blocking_k, entries))
	return err
}

// Reserve submits GeospatialIndexOn.Reserve.
func (on GeospatialIndexDeferredOn) Reserve(entries uint) *Deferred[struct{}] {
	return submitted(on.reserve(C.usearch_completion_deferred_k, entries))
}

// Clear erases all entries, keeping the reserved capacity. No other verb may overlap it on the
// same handle.
func (on GeospatialIndexOn) Clear() error {
	_, err := block(on.clear(C.usearch_completion_blocking_k))
	return err
}

// Clear submits GeospatialIndexOn.Clear.
func (on GeospatialIndexDeferredOn) Clear() *Deferred[struct{}] {
	return submitted(on.clear(C.usearch_completion_deferred_k))
}

// Describe reports the frozen metadata and the current counters in one call.
func (on GeospatialIndexOn) Describe() (GeospatialIndexDescription, error) {
	return block(on.describe(C.usearch_completion_blocking_k))
}

// Describe submits GeospatialIndexOn.Describe.
func (on GeospatialIndexDeferredOn) Describe() *Deferred[GeospatialIndexDescription] {
	return submitted(on.describe(C.usearch_completion_deferred_k))
}

// Count reports how many observations are stored under a key.
func (on GeospatialIndexOn) Count(key Key) (GeospatialKeyStats, error) {
	return block(on.count(C.usearch_completion_blocking_k, key))
}

// Count submits GeospatialIndexOn.Count.
func (on GeospatialIndexDeferredOn) Count(key Key) *Deferred[GeospatialKeyStats] {
	return submitted(on.count(C.usearch_completion_deferred_k, key))
}

// Contains reports whether the index holds at least one entry under a key, as a nonzero Count.
func (on GeospatialIndexOn) Contains(key Key) (bool, error) {
	counted, err := on.Count(key)
	return counted.Entries != 0, err
}

// Get exports every observation stored under a key into arena, in insertion order, each within the
// outward rounding the index stores it to.
//
// The call rewinds arena, and the observations stay readable until it serves another call.
func (on GeospatialIndexOn) Get(key Key, arena *Arena) (GeospatialObservations, error) {
	return block(on.get(C.usearch_completion_blocking_k, key, arena))
}

// Get submits GeospatialIndexOn.Get.
func (on GeospatialIndexDeferredOn) Get(key Key, arena *Arena) *Deferred[GeospatialObservations] {
	return submitted(on.get(C.usearch_completion_deferred_k, key, arena))
}

// Insert adds observations under their keys - the native batch operation, where a single
// observation is a batch of one. A batch is all-or-nothing and becomes visible at once.
func (on GeospatialIndexOn) Insert(keys []Key, observations []GeospatialObservation, config *DenseInsertConfig) error {
	_, err := block(on.insert(C.usearch_completion_blocking_k, keys, observations, config))
	return err
}

// Insert submits GeospatialIndexOn.Insert.
func (on GeospatialIndexDeferredOn) Insert(keys []Key, observations []GeospatialObservation,
	config *DenseInsertConfig) *Deferred[struct{}] {
	return submitted(on.insert(C.usearch_completion_deferred_k, keys, observations, config))
}

// Refine re-selects the neighbors of every stored entry against the finished graph, as the dense
// Refine does; a caller repeats rounds until the report's Verdict reads DenseRefineConverged.
func (on GeospatialIndexOn) Refine(config *GeospatialRefineConfig) (DenseRefineReport, error) {
	return block(on.refine(C.usearch_completion_blocking_k, config))
}

// Refine submits GeospatialIndexOn.Refine.
func (on GeospatialIndexDeferredOn) Refine(config *GeospatialRefineConfig) *Deferred[DenseRefineReport] {
	return submitted(on.refine(C.usearch_completion_deferred_k, config))
}

// Health measures how much of the graph a search reaches from the root. Read-only.
func (on GeospatialIndexOn) Health() (DenseReachability, error) {
	return block(on.health(C.usearch_completion_blocking_k))
}

// Health submits GeospatialIndexOn.Health.
func (on GeospatialIndexDeferredOn) Health() *Deferred[DenseReachability] {
	return submitted(on.health(C.usearch_completion_deferred_k))
}

// Search finds up to wanted nearest entries of each query observation that relation admits,
// written into arena, scored in chord metres; the relation, the time window and the predicate
// narrow the candidates, so a query may hold fewer than wanted. Never writes to the graph.
func (on GeospatialIndexOn) Search(queries []GeospatialObservation, relation GeospatialRelation, wanted uint,
	arena *Arena, config *GeospatialSearchConfig) (MatchesView, error) {
	return block(on.search(C.usearch_completion_blocking_k, queries, relation, wanted, arena, config))
}

// Search submits GeospatialIndexOn.Search.
func (on GeospatialIndexDeferredOn) Search(queries []GeospatialObservation, relation GeospatialRelation, wanted uint,
	arena *Arena, config *GeospatialSearchConfig) *Deferred[MatchesView] {
	return submitted(on.search(C.usearch_completion_deferred_k, queries, relation, wanted, arena, config))
}

// Within reports every entry within metres of each query observation that relation admits, with
// its chord metres, as one ragged run per query written into arena; metres must be positive.
func (on GeospatialIndexOn) Within(queries []GeospatialObservation, relation GeospatialRelation, metres float64,
	arena *Arena, config *GeospatialWithinConfig) (DenseProducts, error) {
	return block(on.within(C.usearch_completion_blocking_k, queries, relation, metres, arena, config))
}

// Within submits GeospatialIndexOn.Within.
func (on GeospatialIndexDeferredOn) Within(queries []GeospatialObservation, relation GeospatialRelation,
	metres float64, arena *Arena, config *GeospatialWithinConfig) *Deferred[DenseProducts] {
	return submitted(on.within(C.usearch_completion_deferred_k, queries, relation, metres, arena, config))
}

// Reshard moves the entries of sources into targets, keeping their links rather than re-inserting
// them, then runs a round on every target that took entries: several sources gather into one
// target, and one source cuts into one graph-local part per target. The receiver contributes only
// its executor, and a refusal moves nothing.
func (on GeospatialIndexOn) Reshard(sources, targets []*GeospatialIndex, config *GeospatialReshardConfig) (DenseRefineReport, error) {
	return block(on.reshard(C.usearch_completion_blocking_k, sources, targets, config))
}

// Reshard submits GeospatialIndexOn.Reshard.
func (on GeospatialIndexDeferredOn) Reshard(sources, targets []*GeospatialIndex,
	config *GeospatialReshardConfig) *Deferred[DenseRefineReport] {
	return submitted(on.reshard(C.usearch_completion_deferred_k, sources, targets, config))
}

// RefinePlanFinish closes a live refine plan and frees it whatever the call answers, reporting
// what the whole plan did to the graph.
func (on GeospatialIndexOn) RefinePlanFinish(plan *GeospatialRefinePlan) (DenseRefineReport, error) {
	return block(on.refinePlanFinish(C.usearch_completion_blocking_k, plan))
}

// RefinePlanFinish submits GeospatialIndexOn.RefinePlanFinish.
func (on GeospatialIndexDeferredOn) RefinePlanFinish(plan *GeospatialRefinePlan) *Deferred[DenseRefineReport] {
	return submitted(on.refinePlanFinish(C.usearch_completion_deferred_k, plan))
}

// ReshardPlanFinish closes a live reshard plan on this receiver's executor and frees it whatever
// the call answers, reporting what the whole plan did.
func (on GeospatialIndexOn) ReshardPlanFinish(plan *GeospatialReshardPlan) (DenseRefineReport, error) {
	return block(on.reshardPlanFinish(C.usearch_completion_blocking_k, plan))
}

// ReshardPlanFinish submits GeospatialIndexOn.ReshardPlanFinish.
func (on GeospatialIndexDeferredOn) ReshardPlanFinish(plan *GeospatialReshardPlan) *Deferred[DenseRefineReport] {
	return submitted(on.reshardPlanFinish(C.usearch_completion_deferred_k, plan))
}

// Sample draws up to wanted keys around the config's Focus, or across the whole index, each
// hanging from a shown or earlier draw and placed in three dimensions beside the shown ones.
func (on GeospatialIndexOn) Sample(wanted uint, arena *Arena, config *DenseSampleConfig) (DenseDraws, error) {
	return block(on.sample(C.usearch_completion_blocking_k, wanted, arena, config))
}

// Sample submits GeospatialIndexOn.Sample.
func (on GeospatialIndexDeferredOn) Sample(wanted uint, arena *Arena, config *DenseSampleConfig) *Deferred[DenseDraws] {
	return submitted(on.sample(C.usearch_completion_deferred_k, wanted, arena, config))
}

// Save writes the index under a path, in the layout its extension picks - one .safegraphs
// monolith, or a .safegraphs.json manifest beside its shards.
func (on GeospatialIndexOn) Save(path string, config *DenseSaveConfig) error {
	_, err := block(on.save(C.usearch_completion_blocking_k, path, config))
	return err
}

// Save submits GeospatialIndexOn.Save.
func (on GeospatialIndexDeferredOn) Save(path string, config *DenseSaveConfig) *Deferred[struct{}] {
	return submitted(on.save(C.usearch_completion_deferred_k, path, config))
}

// SaveBuffer serializes the index as one monolith into arena.
func (on GeospatialIndexOn) SaveBuffer(arena *Arena, config *DenseSaveBufferConfig) (Monolith, error) {
	return block(on.saveBuffer(C.usearch_completion_blocking_k, arena, config))
}

// SaveBuffer submits GeospatialIndexOn.SaveBuffer.
func (on GeospatialIndexDeferredOn) SaveBuffer(arena *Arena, config *DenseSaveBufferConfig) *Deferred[Monolith] {
	return submitted(on.saveBuffer(C.usearch_completion_deferred_k, arena, config))
}

// endregion: Receivers

// region: Calls

func (receiver geospatialReceiver) reserve(completion C.usearch_completion_t, entries uint) *lent[struct{}] {
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_geospatial_reserve(&C.usearch_geospatial_reserve_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		entries:    C.size_t(entries),
	})
	return lent
}

func (receiver geospatialReceiver) clear(completion C.usearch_completion_t) *lent[struct{}] {
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_geospatial_clear(&C.usearch_geospatial_clear_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
	})
	return lent
}

func (receiver geospatialReceiver) describe(completion C.usearch_completion_t) *lent[GeospatialIndexDescription] {
	lent := open[GeospatialIndexDescription]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_geospatial_describe(&C.usearch_geospatial_describe_t{
		index:       index,
		executor:    executor,
		outcome:     lent.outcome,
		completion:  completion,
		description: (*C.usearch_geospatial_index_description_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver geospatialReceiver) count(completion C.usearch_completion_t, key Key) *lent[GeospatialKeyStats] {
	lent := open[GeospatialKeyStats]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_geospatial_count(&C.usearch_geospatial_count_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		key:        key.toRaw(),
		key_stats:  (*C.usearch_geospatial_key_stats_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver geospatialReceiver) get(completion C.usearch_completion_t, key Key,
	arena *Arena) *lent[GeospatialObservations] {
	lent := open[GeospatialObservations]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_geospatial_get(&C.usearch_geospatial_get_t{
		index:        index,
		executor:     executor,
		outcome:      lent.outcome,
		completion:   completion,
		key:          key.toRaw(),
		observations: &lent.result.raw,
		written:      (*C.size_t)(unsafe.Pointer(&lent.result.written)),
		arena:        arena.lower(),
	})
	return lent
}

func (receiver geospatialReceiver) insert(completion C.usearch_completion_t, keys []Key,
	observations []GeospatialObservation, config *DenseInsertConfig) *lent[struct{}] {
	if config == nil {
		config = new(DenseInsertConfig)
	}
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	// The call has one count, so only this side sees the keys and the observations disagree.
	if len(keys) != len(observations) {
		return lent.refuse(fail(StatusShapeMismatch, "the key count does not match the observations"))
	}
	C.usearch_geospatial_insert(&C.usearch_geospatial_insert_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome,
		completion:       completion,
		count:            C.size_t(len(keys)),
		keys:             lent.keys(keys),
		observations:     lent.observations(observations),
		expansion_factor: C.size_t(config.ExpansionFactor),
		progress:         lent.progress(config.Progress, config.ProgressSteps),
		stats:            lent.stats(config.Stats),
	})
	return lent
}

// observations lends a slice of observations as the run the library reads, laid out alike.
func (call *call) observations(observations []GeospatialObservation) *C.usearch_geospatial_observation_t {
	return (*C.usearch_geospatial_observation_t)(unsafe.Pointer(lend(call, observations)))
}

func (receiver geospatialReceiver) refine(completion C.usearch_completion_t,
	config *GeospatialRefineConfig) *lent[DenseRefineReport] {
	if config == nil {
		config = new(GeospatialRefineConfig)
	}
	lent := open[DenseRefineReport]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	plan, phase := config.Plan.lower(&lent.call)
	C.usearch_geospatial_refine(&C.usearch_geospatial_refine_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
		refine: C.usearch_refine_options_t{
			expansion_factor: C.size_t(config.ExpansionFactor),
			memory_limit:     C.size_t(config.MemoryLimit),
		},
		progress: lent.progress(config.Progress, config.ProgressSteps),
		stats:    lent.stats(config.Stats),
		plan:     plan,
		phase:    phase,
	})
	return lent
}

func (receiver geospatialReceiver) refinePlanFinish(completion C.usearch_completion_t,
	plan *GeospatialRefinePlan) *lent[DenseRefineReport] {
	lent := open[DenseRefineReport]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_geospatial_refine_plan_finish(&C.usearch_geospatial_refine_plan_finish_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver geospatialReceiver) health(completion C.usearch_completion_t) *lent[DenseReachability] {
	lent := open[DenseReachability]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_geospatial_health(&C.usearch_geospatial_health_t{
		index:        index,
		executor:     executor,
		outcome:      lent.outcome,
		completion:   completion,
		reachability: (*C.usearch_reachability_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver geospatialReceiver) search(completion C.usearch_completion_t, queries []GeospatialObservation,
	relation GeospatialRelation, wanted uint, arena *Arena, config *GeospatialSearchConfig) *lent[MatchesView] {
	if config == nil {
		config = new(GeospatialSearchConfig)
	}
	lent := open[MatchesView]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	lent.result.queries = uint(len(queries))
	C.usearch_geospatial_search(&C.usearch_geospatial_search_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome,
		completion:       completion,
		count:            C.size_t(len(queries)),
		queries:          lent.observations(queries),
		relation:         C.usearch_geospatial_relation_t(relation),
		tolerance:        C.usearch_geospatial_tolerance_t{metres: C.double(config.Metres), seconds: C.double(config.Seconds)},
		matches:          &lent.result.raw,
		arena:            arena.lower(),
		wanted:           C.size_t(wanted),
		expansion_factor: C.size_t(config.ExpansionFactor),
		traversal:        C.usearch_traversal_t(config.Traversal),
		predicate:        lent.predicate(config.Predicate),
		progress:         lent.progress(config.Progress, config.ProgressSteps),
		stats:            lent.stats(config.Stats),
	})
	return lent
}

func (receiver geospatialReceiver) within(completion C.usearch_completion_t, queries []GeospatialObservation,
	relation GeospatialRelation, metres float64, arena *Arena, config *GeospatialWithinConfig) *lent[DenseProducts] {
	if config == nil {
		config = new(GeospatialWithinConfig)
	}
	lent := open[DenseProducts]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	lent.result.queries = uint(len(queries))
	C.usearch_geospatial_within(&C.usearch_geospatial_within_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		count:      C.size_t(len(queries)),
		queries:    lent.observations(queries),
		relation:   C.usearch_geospatial_relation_t(relation),
		tolerance:  C.usearch_geospatial_tolerance_t{metres: C.double(metres), seconds: C.double(config.Seconds)},
		matches:    &lent.result.raw,
		written:    (*C.size_t)(unsafe.Pointer(&lent.result.required)),
		arena:      arena.lower(),
		traversal:  C.usearch_traversal_t(config.Traversal),
		predicate:  lent.predicate(config.Predicate),
		progress:   lent.progress(config.Progress, config.ProgressSteps),
		stats:      lent.stats(config.Stats),
	})
	return lent
}

func (receiver geospatialReceiver) reshard(completion C.usearch_completion_t, sources, targets []*GeospatialIndex,
	config *GeospatialReshardConfig) *lent[DenseRefineReport] {
	if config == nil {
		config = new(GeospatialReshardConfig)
	}
	lent := open[DenseRefineReport]()
	// A reshard names its handles in its lists, so the receiver lends only its executor.
	executor, refusal := receiver.executor.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	sourceHandles, targetHandles := geospatialHandles(sources), geospatialHandles(targets)
	plan, phase := config.Plan.lower(&lent.call)
	C.usearch_geospatial_reshard(&C.usearch_geospatial_reshard_t{
		sources_count: C.size_t(len(sourceHandles)),
		sources:       lend(&lent.call, sourceHandles),
		targets_count: C.size_t(len(targetHandles)),
		targets:       lend(&lent.call, targetHandles),
		executor:      executor,
		outcome:       lent.outcome,
		completion:    completion,
		report:        (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
		refine: C.usearch_refine_options_t{
			expansion_factor: C.size_t(config.ExpansionFactor),
			memory_limit:     C.size_t(config.MemoryLimit),
		},
		seed:     C.uint64_t(config.Seed),
		progress: lent.progress(config.Progress, config.ProgressSteps),
		stats:    lent.stats(config.Stats),
		plan:     plan,
		phase:    phase,
	})
	return lent
}

// geospatialHandles lists the handles of indexes.
func geospatialHandles(indexes []*GeospatialIndex) []C.usearch_geospatial_index_t {
	handles := make([]C.usearch_geospatial_index_t, len(indexes))
	for position, index := range indexes {
		handles[position] = index.handle
	}
	return handles
}

func (receiver geospatialReceiver) reshardPlanFinish(completion C.usearch_completion_t,
	plan *GeospatialReshardPlan) *lent[DenseRefineReport] {
	lent := open[DenseRefineReport]()
	// A reshard names its handles in its lists, so the receiver lends only its executor.
	executor, refusal := receiver.executor.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_geospatial_reshard_plan_finish(&C.usearch_geospatial_reshard_plan_finish_t{
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver geospatialReceiver) sample(completion C.usearch_completion_t, wanted uint, arena *Arena,
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
	C.usearch_geospatial_sample(&C.usearch_geospatial_sample_t{
		index:             index,
		executor:          executor,
		outcome:           lent.outcome,
		completion:        completion,
		shown_count:       C.size_t(len(config.Shown)),
		shown:             lent.keys(config.Shown),
		shown_coordinates: lent.points(config.ShownCoordinates),
		focus_count:       C.size_t(len(config.Focus)),
		focus:             lent.keys(config.Focus),
		keys:              &lent.result.keys,
		entry_ordinals:    &lent.result.entryOrdinals,
		parents:           &lent.result.parents,
		coordinates:       &lent.result.coordinates,
		written:           (*C.size_t)(unsafe.Pointer(&lent.result.written)),
		arena:             arena.lower(),
		wanted:            C.size_t(wanted),
		seed:              C.uint64_t(config.Seed),
		expansion_factor:  C.size_t(config.ExpansionFactor),
		progress:          lent.progress(config.Progress, config.ProgressSteps),
		stats:             lent.stats(config.Stats),
	})
	return lent
}

func (receiver geospatialReceiver) save(completion C.usearch_completion_t, path string,
	config *DenseSaveConfig) *lent[struct{}] {
	if config == nil {
		config = new(DenseSaveConfig)
	}
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_geospatial_save(&C.usearch_geospatial_save_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		path:       lent.text(path),
		progress:   lent.progress(config.Progress, config.ProgressSteps),
	})
	return lent
}

func (receiver geospatialReceiver) saveBuffer(completion C.usearch_completion_t, arena *Arena,
	config *DenseSaveBufferConfig) *lent[Monolith] {
	if config == nil {
		config = new(DenseSaveBufferConfig)
	}
	lent := open[Monolith]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_geospatial_save_buffer(&C.usearch_geospatial_save_buffer_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		monolith:   &lent.result.raw,
		arena:      arena.lower(),
		progress:   lent.progress(config.Progress, config.ProgressSteps),
	})
	return lent
}

// endregion: Calls

// region: Cursors

// GeospatialCursor is a walk over the index contents, pulled blocking on the executor of the
// receiver that opened it; an entry once passed is never revisited.
type GeospatialCursor struct {
	handle   C.usearch_geospatial_cursor_t
	receiver geospatialReceiver
}

// Cursor opens a walk over the current contents. Without starts it yields every entry as stored;
// with them, the start keys' own entries first in order, then the entries those link to
// breadth-first, each once.
func (on GeospatialIndexOn) Cursor(starts ...Key) (*GeospatialCursor, error) {
	index, _, refusal := on.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_geospatial_cursor_t]()
	C.usearch_geospatial_cursor_init(&C.usearch_geospatial_cursor_init_t{
		index:        index,
		outcome:      lent.outcome,
		starts_count: C.size_t(len(starts)),
		starts:       (*C.usearch_key_t)(unsafe.Pointer(lend(&lent.call, starts))),
		cursor:       lent.result,
	})
	handle, err := block(lent)
	if err != nil {
		return nil, err
	}
	return &GeospatialCursor{handle: handle, receiver: on.geospatialReceiver}, nil
}

// NextKeys copies up to wanted next keys into arena; a batch of none means the walk is exhausted.
func (cursor *GeospatialCursor) NextKeys(wanted uint, arena *Arena) (DenseBatch, error) {
	return cursor.next(wanted, false, arena)
}

// NextLinks copies up to wanted next keys into arena, each with the keys it links to; a batch of
// none means the walk is exhausted.
func (cursor *GeospatialCursor) NextLinks(wanted uint, arena *Arena) (DenseBatch, error) {
	return cursor.next(wanted, true, arena)
}

// next pulls a batch, exporting each entry's links only when linked asks for them.
func (cursor *GeospatialCursor) next(wanted uint, linked bool, arena *Arena) (DenseBatch, error) {
	index, executor, refusal := cursor.receiver.lower()
	if refusal != nil {
		return DenseBatch{}, refusal
	}
	lent := open[DenseBatch]()
	var links *C.usearch_keys_tape_view_t
	if linked {
		links = &lent.result.links
	}
	C.usearch_geospatial_cursor_next(&C.usearch_geospatial_cursor_next_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: C.usearch_completion_blocking_k,
		cursor:     cursor.handle,
		keys:       &lent.result.keys,
		links:      links,
		written:    (*C.size_t)(unsafe.Pointer(&lent.result.written)),
		arena:      arena.lower(),
		wanted:     C.size_t(wanted),
	})
	return block(lent)
}

// Close releases the cursor. It is idempotent.
func (cursor *GeospatialCursor) Close() error {
	if cursor == nil || cursor.handle == nil {
		return nil
	}
	C.usearch_geospatial_cursor_free(cursor.handle)
	cursor.handle = nil
	return nil
}

// endregion: Cursors
