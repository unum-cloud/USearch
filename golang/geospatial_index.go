// Similarity Search over observations on and above the Earth, under sight and view-cone relations.
//
// File: golang/geospatial_index.go
// Author: Ash Vardanian
// Date: November 16, 2026

// Every verb is a batch over a slice of observations, laid out as the library reads them, so a
// slice crosses with no copy and a single observation is a batch of one. Every score is the
// straight-line chord between two positions, in metres. Cursor batches and draws are the dense
// modality's own types, since the library fills them through the same C structures. Terrain is not
// bound, so a handle answers the visible relation with StatusUnsupported.

package usearch

import (
	"math"
	"unsafe"
)

/*
// noescape and nocallback follow the rule in types.go.
#cgo noescape usearch_geospatial_init_ram
#cgo noescape usearch_geospatial_init_vram
#cgo noescape usearch_geospatial_view_ram
#cgo noescape usearch_geospatial_cursor_init
#cgo nocallback usearch_geospatial_cursor_init
#cgo noescape usearch_geospatial_view_nvme
#cgo noescape usearch_geospatial_reserve
#cgo nocallback usearch_geospatial_reserve
#cgo noescape usearch_geospatial_clear
#cgo nocallback usearch_geospatial_clear
#cgo noescape usearch_geospatial_describe
#cgo nocallback usearch_geospatial_describe
#cgo noescape usearch_geospatial_count
#cgo nocallback usearch_geospatial_count
#cgo noescape usearch_geospatial_get
#cgo nocallback usearch_geospatial_get
#cgo noescape usearch_geospatial_health
#cgo nocallback usearch_geospatial_health
#cgo noescape usearch_geospatial_metadata_read
#cgo nocallback usearch_geospatial_metadata_read
#cgo noescape usearch_geospatial_metadata_read_buffer
#cgo nocallback usearch_geospatial_metadata_read_buffer
#cgo noescape usearch_geospatial_load_ram
#cgo noescape usearch_geospatial_load_vram
#cgo noescape usearch_geospatial_load_ram_buffer
#cgo noescape usearch_geospatial_load_vram_buffer
#cgo noescape usearch_geospatial_insert
#cgo noescape usearch_geospatial_refine
#cgo noescape usearch_geospatial_refine_plan_finish
#cgo nocallback usearch_geospatial_refine_plan_finish
#cgo noescape usearch_geospatial_search
#cgo noescape usearch_geospatial_within
#cgo noescape usearch_geospatial_reshard
#cgo noescape usearch_geospatial_reshard_plan_finish
#cgo nocallback usearch_geospatial_reshard_plan_finish
#cgo noescape usearch_geospatial_sample
#cgo noescape usearch_geospatial_save
#cgo noescape usearch_geospatial_save_buffer
#cgo noescape usearch_geospatial_cursor_next
#cgo nocallback usearch_geospatial_cursor_next

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

// GeospatialLoadRamConfig carries the optional settings of GeospatialLoadRam.
type GeospatialLoadRamConfig struct {
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

// GeospatialLoadVramConfig carries the optional settings of GeospatialLoadVram.
type GeospatialLoadVramConfig struct {
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

// GeospatialLoadRamBufferConfig carries the optional settings of GeospatialLoadRamBuffer.
type GeospatialLoadRamBufferConfig struct {
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

// GeospatialLoadVramBufferConfig carries the optional settings of GeospatialLoadVramBuffer.
type GeospatialLoadVramBufferConfig struct {
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

// GeospatialMetadataReadConfig carries the optional settings of GeospatialMetadataRead.
type GeospatialMetadataReadConfig struct {
	// Executor is the one the read runs on; nil reads on the calling goroutine.
	Executor *Executor
}

// GeospatialMetadataReadBufferConfig carries the optional settings of GeospatialMetadataReadBuffer.
type GeospatialMetadataReadBufferConfig struct {
	// Executor is the one the read runs on; nil reads on the calling goroutine.
	Executor *Executor
}

// GeospatialInsertConfig carries the optional settings of Insert. A zero value derives them all.
type GeospatialInsertConfig struct {
	// ExpansionFactor is the breadth of the candidates the build walk explores; zero derives it
	// from the geometry.
	ExpansionFactor uint
	// Progress reports the entries inserted; every entry is claimed before the first report, so a
	// stop is ignored.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *TraversalStats
}

// GeospatialSampleConfig carries the optional settings of Sample. A zero value draws across the
// whole index with nothing shown.
type GeospatialSampleConfig struct {
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
	Stats *TraversalStats
}

// GeospatialSaveConfig carries the optional settings of Save.
type GeospatialSaveConfig struct {
	// Progress reports the shards written and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// GeospatialSaveBufferConfig carries the optional settings of SaveBuffer.
type GeospatialSaveBufferConfig struct {
	// Progress reports the bytes written and stops wherever it says to; a stopped save leaves no
	// monolith.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

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
	// Predicate admits only the keys it approves, one or a batch at a time; nil admits everything.
	Predicate Admitter
	// Progress reports the queries resolved and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *TraversalStats
}

// GeospatialWithinConfig carries the optional settings of Within. A zero value derives every
// setting.
type GeospatialWithinConfig struct {
	// Seconds is the slack on both ends of the query's time interval.
	Seconds float64
	// Traversal walks the graph completely by default; TraversalScan judges every entry exactly, on
	// RAM and NVMe handles.
	Traversal Traversal
	// Predicate admits only the keys it approves, one or a batch at a time; nil admits everything.
	Predicate Admitter
	// Progress reports the queries resolved and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *TraversalStats
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
	Stats *TraversalStats
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
	Stats *TraversalStats
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
	index.GeospatialIndexOn = GeospatialIndexOn{geospatialReceiver{receiver[C.usearch_geospatial_index_t]{handle: &index.handle}}}
	return index, nil
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
	var lent lent[C.usearch_geospatial_index_t]
	arguments := open[C.usearch_geospatial_init_ram_t](&lent)
	*arguments = C.usearch_geospatial_init_ram_t{
		index:          lent.result,
		outcome:        lent.outcome(),
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		ram:            lent.ram(memoryDomains, options.CPUClassCeiling),
	}
	C.usearch_geospatial_init_ram(arguments)
	return geospatialIndexOf(block(&lent, nil))
}

// GeospatialInitVram creates an empty, mutable index in device memory, sharded across the CUDA
// ordinals devices names and never replicated; empty derives one free device.
func GeospatialInitVram(options GeospatialOptions, devices []int32) (*GeospatialIndex, error) {
	var lent lent[C.usearch_geospatial_index_t]
	arguments := open[C.usearch_geospatial_init_vram_t](&lent)
	*arguments = C.usearch_geospatial_init_vram_t{
		index:          lent.result,
		outcome:        lent.outcome(),
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		vram:           lent.vram(devices, options.GPUClassCeiling),
	}
	C.usearch_geospatial_init_vram(arguments)
	return geospatialIndexOf(block(&lent, nil))
}

// GeospatialLoadRam copies a checkpoint into a new, mutable index in host memory. The path names a
// manifest of a sharded checkpoint or a single .safegraphs monolith.
func GeospatialLoadRam(path string, config *GeospatialLoadRamConfig) (*GeospatialIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var lent lent[C.usearch_geospatial_index_t]
	arguments := open[C.usearch_geospatial_load_ram_t](&lent)
	*arguments = C.usearch_geospatial_load_ram_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome(),
		path:           lent.text(path),
		ram:            lent.ram(options.MemoryDomains, options.CPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		progress:       lent.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_geospatial_load_ram(arguments)
	return geospatialIndexOf(block(&lent, nil))
}

// GeospatialLoadVram copies a checkpoint into a new, mutable index in device memory, with the
// placement rules of GeospatialLoadRam.
func GeospatialLoadVram(path string, config *GeospatialLoadVramConfig) (*GeospatialIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var lent lent[C.usearch_geospatial_index_t]
	arguments := open[C.usearch_geospatial_load_vram_t](&lent)
	*arguments = C.usearch_geospatial_load_vram_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome(),
		path:           lent.text(path),
		vram:           lent.vram(options.Devices, options.GPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		progress:       lent.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_geospatial_load_vram(arguments)
	return geospatialIndexOf(block(&lent, nil))
}

// GeospatialLoadRamBuffer copies a monolith from a buffer into a new, mutable index in host memory,
// as GeospatialLoadRam does a path.
func GeospatialLoadRamBuffer(buffer []byte, config *GeospatialLoadRamBufferConfig) (*GeospatialIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var lent lent[C.usearch_geospatial_index_t]
	arguments := open[C.usearch_geospatial_load_ram_buffer_t](&lent)
	*arguments = C.usearch_geospatial_load_ram_buffer_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome(),
		buffer:         lent.bytes(buffer),
		ram:            lent.ram(options.MemoryDomains, options.CPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		progress:       lent.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_geospatial_load_ram_buffer(arguments)
	return geospatialIndexOf(block(&lent, nil))
}

// GeospatialLoadVramBuffer copies a monolith from a buffer into a new, mutable index in device
// memory.
func GeospatialLoadVramBuffer(buffer []byte, config *GeospatialLoadVramBufferConfig) (*GeospatialIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var lent lent[C.usearch_geospatial_index_t]
	arguments := open[C.usearch_geospatial_load_vram_buffer_t](&lent)
	*arguments = C.usearch_geospatial_load_vram_buffer_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome(),
		buffer:         lent.bytes(buffer),
		vram:           lent.vram(options.Devices, options.GPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		progress:       lent.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_geospatial_load_vram_buffer(arguments)
	return geospatialIndexOf(block(&lent, nil))
}

// GeospatialViewRam serves a checkpoint in place through a memory mapping, as a read-only index
// whose mutating verbs report StatusReadOnly; the files must stay unmodified while it lives.
func GeospatialViewRam(path string) (*GeospatialIndex, error) {
	var lent lent[C.usearch_geospatial_index_t]
	arguments := open[C.usearch_geospatial_view_ram_t](&lent)
	*arguments = C.usearch_geospatial_view_ram_t{
		index:   lent.result,
		outcome: lent.outcome(),
		path:    lent.text(path),
	}
	C.usearch_geospatial_view_ram(arguments)
	return geospatialIndexOf(block(&lent, nil))
}

// GeospatialViewNvme serves a checkpoint in place from NVMe drives, read through the executor's
// submission rings, as a read-only index.
func GeospatialViewNvme(path string, config *GeospatialViewNvmeConfig) (*GeospatialIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var lent lent[C.usearch_geospatial_index_t]
	arguments := open[C.usearch_geospatial_view_nvme_t](&lent)
	*arguments = C.usearch_geospatial_view_nvme_t{
		index:    lent.result,
		executor: executor,
		outcome:  lent.outcome(),
		path:     lent.text(path),
		nvme:     lent.nvme(options.Paths, options.CPUClassCeiling),
	}
	C.usearch_geospatial_view_nvme(arguments)
	return geospatialIndexOf(block(&lent, nil))
}

// GeospatialMetadataRead reads a checkpoint's connectivity without loading it.
func GeospatialMetadataRead(path string, config *GeospatialMetadataReadConfig) (GeospatialMetadata, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return GeospatialMetadata{}, err
	}
	var lent lent[GeospatialMetadata]
	arguments := open[C.usearch_geospatial_metadata_read_t](&lent)
	*arguments = C.usearch_geospatial_metadata_read_t{
		executor: executor,
		outcome:  lent.outcome(),
		path:     lent.text(path),
		metadata: (*C.usearch_geospatial_metadata_t)(unsafe.Pointer(lent.result)),
	}
	C.usearch_geospatial_metadata_read(arguments)
	return block(&lent, nil)
}

// GeospatialMetadataReadBuffer reads the connectivity out of a serialized monolith buffer.
func GeospatialMetadataReadBuffer(buffer []byte, config *GeospatialMetadataReadBufferConfig) (GeospatialMetadata, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return GeospatialMetadata{}, err
	}
	var lent lent[GeospatialMetadata]
	arguments := open[C.usearch_geospatial_metadata_read_buffer_t](&lent)
	*arguments = C.usearch_geospatial_metadata_read_buffer_t{
		executor: executor,
		outcome:  lent.outcome(),
		buffer:   lent.bytes(buffer),
		metadata: (*C.usearch_geospatial_metadata_t)(unsafe.Pointer(lent.result)),
	}
	C.usearch_geospatial_metadata_read_buffer(arguments)
	return block(&lent, nil)
}

// endregion: Index

// region: Receivers

// geospatialReceiver is an index and the executor its verbs run on, nil for the calling goroutine.
type geospatialReceiver struct {
	receiver[C.usearch_geospatial_index_t]
}

// GeospatialIndexOn is the receiver whose verbs block until their call is final: on the calling
// goroutine when embedded in a GeospatialIndex, and on an executor when GeospatialIndexOn.On
// makes it. Binding-only.
type GeospatialIndexOn struct {
	geospatialReceiver
}

// GeospatialIndexDeferredOn is the receiver whose verbs submit their calls to an executor and
// answer a Deferred, offering every verb of GeospatialIndexOn with the same arguments but the
// blocking-only sugar. Binding-only.
//
// The library reads a deferred call's inputs until the call is final, so the slices a verb is
// handed must not change until then; its arena is borrowed, as a blocking verb borrows it.
type GeospatialIndexDeferredOn struct {
	geospatialReceiver
}

// On is the receiver whose verbs block on executor. Binding-only.
func (on GeospatialIndexOn) On(executor *Executor) GeospatialIndexOn {
	on.executor = executor
	return on
}

// Deferred is the receiver whose verbs submit their calls to executor, each answering a Deferred
// from the same arguments, or the refusal it caught before submitting anything. Binding-only.
func (on GeospatialIndexOn) Deferred(executor *Executor) GeospatialIndexDeferredOn {
	on.executor = executor
	return GeospatialIndexDeferredOn{on.geospatialReceiver}
}

// Reserve claims capacity for a number of incoming observations ahead of insertion. The capacity is
// the desired total, including what the index holds.
func (on GeospatialIndexOn) Reserve(entries uint) error {
	var lent lent[struct{}]
	_, err := block(&lent, on.reserve(&lent, C.usearch_completion_blocking_k, entries))
	return err
}

// Reserve submits GeospatialIndexOn.Reserve.
func (on GeospatialIndexDeferredOn) Reserve(entries uint) (*Deferred[struct{}], error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.reserve(&deferred.lent, C.usearch_completion_deferred_k, entries))
}

// Clear erases all entries, keeping the reserved capacity. No other verb may overlap it on the
// same handle.
func (on GeospatialIndexOn) Clear() error {
	var lent lent[struct{}]
	_, err := block(&lent, on.clear(&lent, C.usearch_completion_blocking_k))
	return err
}

// Clear submits GeospatialIndexOn.Clear.
func (on GeospatialIndexDeferredOn) Clear() (*Deferred[struct{}], error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.clear(&deferred.lent, C.usearch_completion_deferred_k))
}

// Describe reports the frozen metadata and the current counters in one call.
func (on GeospatialIndexOn) Describe() (GeospatialIndexDescription, error) {
	var lent lent[GeospatialIndexDescription]
	return block(&lent, on.describe(&lent, C.usearch_completion_blocking_k))
}

// Describe submits GeospatialIndexOn.Describe.
func (on GeospatialIndexDeferredOn) Describe() (*Deferred[GeospatialIndexDescription], error) {
	deferred := new(Deferred[GeospatialIndexDescription])
	return submitted(deferred, on.describe(&deferred.lent, C.usearch_completion_deferred_k))
}

// Count reports how many observations are stored under a key.
func (on GeospatialIndexOn) Count(key Key) (GeospatialKeyStats, error) {
	var lent lent[GeospatialKeyStats]
	return block(&lent, on.count(&lent, C.usearch_completion_blocking_k, key))
}

// Count submits GeospatialIndexOn.Count.
func (on GeospatialIndexDeferredOn) Count(key Key) (*Deferred[GeospatialKeyStats], error) {
	deferred := new(Deferred[GeospatialKeyStats])
	return submitted(deferred, on.count(&deferred.lent, C.usearch_completion_deferred_k, key))
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
	var lent lent[GeospatialObservations]
	return block(&lent, on.get(&lent, C.usearch_completion_blocking_k, key, arena))
}

// Get submits GeospatialIndexOn.Get.
func (on GeospatialIndexDeferredOn) Get(key Key, arena *Arena) (*Deferred[GeospatialObservations], error) {
	deferred := new(Deferred[GeospatialObservations])
	return submitted(deferred, on.get(&deferred.lent, C.usearch_completion_deferred_k, key, arena))
}

// Insert adds observations under their keys - the native batch operation, where a single
// observation is a batch of one. A batch is all-or-nothing and becomes visible at once.
func (on GeospatialIndexOn) Insert(keys []Key, observations []GeospatialObservation, config *GeospatialInsertConfig) error {
	var lent lent[struct{}]
	_, err := block(&lent, on.insert(&lent, C.usearch_completion_blocking_k, keys, observations, config))
	return err
}

// Insert submits GeospatialIndexOn.Insert.
func (on GeospatialIndexDeferredOn) Insert(keys []Key, observations []GeospatialObservation,
	config *GeospatialInsertConfig) (*Deferred[struct{}], error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.insert(&deferred.lent, C.usearch_completion_deferred_k, keys, observations, config))
}

// Refine re-selects the neighbors of every stored entry against the finished graph, as the dense
// Refine does; a caller repeats rounds until the report's Verdict reads RefineConverged.
func (on GeospatialIndexOn) Refine(config *GeospatialRefineConfig) (RefineReport, error) {
	var lent lent[RefineReport]
	return block(&lent, on.refine(&lent, C.usearch_completion_blocking_k, config))
}

// Refine submits GeospatialIndexOn.Refine.
func (on GeospatialIndexDeferredOn) Refine(config *GeospatialRefineConfig) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.refine(&deferred.lent, C.usearch_completion_deferred_k, config))
}

// Health measures how much of the graph a search reaches from the root. Read-only.
func (on GeospatialIndexOn) Health() (Reachability, error) {
	var lent lent[Reachability]
	return block(&lent, on.health(&lent, C.usearch_completion_blocking_k))
}

// Health submits GeospatialIndexOn.Health.
func (on GeospatialIndexDeferredOn) Health() (*Deferred[Reachability], error) {
	deferred := new(Deferred[Reachability])
	return submitted(deferred, on.health(&deferred.lent, C.usearch_completion_deferred_k))
}

// Search finds up to wanted nearest entries of each query observation that relation admits,
// written into arena, scored in chord metres; the relation, the time window and the predicate
// narrow the candidates, so a query may hold fewer than wanted. Never writes to the graph.
func (on GeospatialIndexOn) Search(queries []GeospatialObservation, relation GeospatialRelation, wanted uint,
	arena *Arena, config *GeospatialSearchConfig) (MatchesView, error) {
	var lent lent[MatchesView]
	return block(&lent, on.search(&lent, C.usearch_completion_blocking_k, queries, relation, wanted, arena, config))
}

// Search submits GeospatialIndexOn.Search.
func (on GeospatialIndexDeferredOn) Search(queries []GeospatialObservation, relation GeospatialRelation, wanted uint,
	arena *Arena, config *GeospatialSearchConfig) (*Deferred[MatchesView], error) {
	deferred := new(Deferred[MatchesView])
	return submitted(deferred, on.search(&deferred.lent, C.usearch_completion_deferred_k, queries, relation, wanted,
		arena, config))
}

// Within reports every entry within metres of each query observation that relation admits, with
// its chord metres, as one ragged run per query written into arena; metres must be positive.
func (on GeospatialIndexOn) Within(queries []GeospatialObservation, relation GeospatialRelation, metres float64,
	arena *Arena, config *GeospatialWithinConfig) (MatchesTape, error) {
	var lent lent[MatchesTape]
	return block(&lent, on.within(&lent, C.usearch_completion_blocking_k, queries, relation, metres, arena, config))
}

// Within submits GeospatialIndexOn.Within.
func (on GeospatialIndexDeferredOn) Within(queries []GeospatialObservation, relation GeospatialRelation, metres float64,
	arena *Arena, config *GeospatialWithinConfig) (*Deferred[MatchesTape], error) {
	deferred := new(Deferred[MatchesTape])
	return submitted(deferred, on.within(&deferred.lent, C.usearch_completion_deferred_k, queries, relation, metres,
		arena, config))
}

// Reshard moves the entries of sources into targets, keeping their links rather than re-inserting
// them, then runs a round on every target that took entries: several sources gather into one
// target, and one source cuts into one graph-local part per target. The receiver contributes only
// its executor, and a refusal moves nothing.
func (on GeospatialIndexOn) Reshard(sources, targets []*GeospatialIndex, config *GeospatialReshardConfig) (RefineReport,
	error) {
	var lent lent[RefineReport]
	return block(&lent, on.reshard(&lent, C.usearch_completion_blocking_k, sources, targets, config))
}

// Reshard submits GeospatialIndexOn.Reshard.
func (on GeospatialIndexDeferredOn) Reshard(sources, targets []*GeospatialIndex,
	config *GeospatialReshardConfig) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.reshard(&deferred.lent, C.usearch_completion_deferred_k, sources, targets, config))
}

// RefinePlanFinish closes a live refine plan and frees it whatever the call answers, reporting
// what the whole plan did to the graph.
func (on GeospatialIndexOn) RefinePlanFinish(plan *GeospatialRefinePlan) (RefineReport, error) {
	var lent lent[RefineReport]
	return block(&lent, on.refinePlanFinish(&lent, C.usearch_completion_blocking_k, plan))
}

// RefinePlanFinish submits GeospatialIndexOn.RefinePlanFinish.
func (on GeospatialIndexDeferredOn) RefinePlanFinish(plan *GeospatialRefinePlan) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.refinePlanFinish(&deferred.lent, C.usearch_completion_deferred_k, plan))
}

// ReshardPlanFinish closes a live reshard plan on this receiver's executor and frees it whatever
// the call answers, reporting what the whole plan did.
func (on GeospatialIndexOn) ReshardPlanFinish(plan *GeospatialReshardPlan) (RefineReport, error) {
	var lent lent[RefineReport]
	return block(&lent, on.reshardPlanFinish(&lent, C.usearch_completion_blocking_k, plan))
}

// ReshardPlanFinish submits GeospatialIndexOn.ReshardPlanFinish.
func (on GeospatialIndexDeferredOn) ReshardPlanFinish(plan *GeospatialReshardPlan) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.reshardPlanFinish(&deferred.lent, C.usearch_completion_deferred_k, plan))
}

// Sample draws up to wanted keys around the config's Focus, or across the whole index, each
// hanging from a shown or earlier draw and placed in three dimensions beside the shown ones.
func (on GeospatialIndexOn) Sample(wanted uint, arena *Arena, config *GeospatialSampleConfig) (DenseDraws, error) {
	var lent lent[DenseDraws]
	return block(&lent, on.sample(&lent, C.usearch_completion_blocking_k, wanted, arena, config))
}

// Sample submits GeospatialIndexOn.Sample.
func (on GeospatialIndexDeferredOn) Sample(wanted uint, arena *Arena, config *GeospatialSampleConfig) (*Deferred[DenseDraws],
	error) {
	deferred := new(Deferred[DenseDraws])
	return submitted(deferred, on.sample(&deferred.lent, C.usearch_completion_deferred_k, wanted, arena, config))
}

// Save writes the index under a path, in the layout its extension picks - one .safegraphs
// monolith, or a .safegraphs.json manifest beside its shards.
func (on GeospatialIndexOn) Save(path string, config *GeospatialSaveConfig) error {
	var lent lent[struct{}]
	_, err := block(&lent, on.save(&lent, C.usearch_completion_blocking_k, path, config))
	return err
}

// Save submits GeospatialIndexOn.Save.
func (on GeospatialIndexDeferredOn) Save(path string, config *GeospatialSaveConfig) (*Deferred[struct{}], error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.save(&deferred.lent, C.usearch_completion_deferred_k, path, config))
}

// SaveBuffer serializes the index as one monolith into arena.
func (on GeospatialIndexOn) SaveBuffer(arena *Arena, config *GeospatialSaveBufferConfig) (Monolith, error) {
	var lent lent[Monolith]
	return block(&lent, on.saveBuffer(&lent, C.usearch_completion_blocking_k, arena, config))
}

// SaveBuffer submits GeospatialIndexOn.SaveBuffer.
func (on GeospatialIndexDeferredOn) SaveBuffer(arena *Arena, config *GeospatialSaveBufferConfig) (*Deferred[Monolith],
	error) {
	deferred := new(Deferred[Monolith])
	return submitted(deferred, on.saveBuffer(&deferred.lent, C.usearch_completion_deferred_k, arena, config))
}

// endregion: Receivers

// region: Calls

func (receiver geospatialReceiver) reserve(lent *lent[struct{}], completion C.usearch_completion_t,
	entries uint) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_reserve_t](lent)
	*arguments = C.usearch_geospatial_reserve_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome(),
		completion: completion,
		entries:    C.size_t(entries),
	}
	C.usearch_geospatial_reserve(arguments)
	return nil
}

func (receiver geospatialReceiver) clear(lent *lent[struct{}], completion C.usearch_completion_t) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_clear_t](lent)
	*arguments = C.usearch_geospatial_clear_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome(),
		completion: completion,
	}
	C.usearch_geospatial_clear(arguments)
	return nil
}

func (receiver geospatialReceiver) describe(lent *lent[GeospatialIndexDescription],
	completion C.usearch_completion_t) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_describe_t](lent)
	*arguments = C.usearch_geospatial_describe_t{
		index:       index,
		executor:    executor,
		outcome:     lent.outcome(),
		completion:  completion,
		description: (*C.usearch_geospatial_index_description_t)(unsafe.Pointer(lent.result)),
	}
	C.usearch_geospatial_describe(arguments)
	return nil
}

func (receiver geospatialReceiver) count(lent *lent[GeospatialKeyStats], completion C.usearch_completion_t,
	key Key) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_count_t](lent)
	*arguments = C.usearch_geospatial_count_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome(),
		completion: completion,
		key:        key.toRaw(),
		key_stats:  (*C.usearch_geospatial_key_stats_t)(unsafe.Pointer(lent.result)),
	}
	C.usearch_geospatial_count(arguments)
	return nil
}

func (receiver geospatialReceiver) get(lent *lent[GeospatialObservations], completion C.usearch_completion_t, key Key,
	arena *Arena) error {
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_get_t](lent)
	*arguments = C.usearch_geospatial_get_t{
		index:        index,
		executor:     executor,
		outcome:      lent.outcome(),
		completion:   completion,
		key:          key.toRaw(),
		observations: &lent.result.raw,
		written:      (*C.size_t)(unsafe.Pointer(&lent.result.written)),
		arena:        outputs,
	}
	C.usearch_geospatial_get(arguments)
	return nil
}

func (receiver geospatialReceiver) insert(lent *lent[struct{}], completion C.usearch_completion_t, keys []Key,
	observations []GeospatialObservation, config *GeospatialInsertConfig) error {
	options := configOf(config)
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	// The call has one count, so only this side sees the keys and the observations disagree.
	if len(keys) != len(observations) {
		return fail(StatusShapeMismatch, "the key count does not match the observations")
	}
	arguments := open[C.usearch_geospatial_insert_t](lent)
	*arguments = C.usearch_geospatial_insert_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome(),
		completion:       completion,
		count:            C.size_t(len(keys)),
		keys:             lent.keys(keys),
		observations:     lent.observations(observations),
		expansion_factor: C.size_t(options.ExpansionFactor),
		progress:         lent.progress(options.Progress, options.ProgressSteps),
		stats:            lent.stats(options.Stats),
	}
	C.usearch_geospatial_insert(arguments)
	return nil
}

// observations lends a slice of observations as the run the library reads, laid out alike.
func (call *call) observations(observations []GeospatialObservation) *C.usearch_geospatial_observation_t {
	return (*C.usearch_geospatial_observation_t)(call.lend(unsafe.Pointer(unsafe.SliceData(observations)), len(observations)))
}

func (receiver geospatialReceiver) refine(lent *lent[RefineReport], completion C.usearch_completion_t,
	config *GeospatialRefineConfig) error {
	options := configOf(config)
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_refine_t](lent)
	plan, phase := options.Plan.lower(&lent.call)
	*arguments = C.usearch_geospatial_refine_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome(),
		completion: completion,
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
		refine: C.usearch_refine_options_t{
			expansion_factor: C.size_t(options.ExpansionFactor),
			memory_limit:     C.size_t(options.MemoryLimit),
		},
		progress: lent.progress(options.Progress, options.ProgressSteps),
		stats:    lent.stats(options.Stats),
		plan:     plan,
		phase:    phase,
	}
	C.usearch_geospatial_refine(arguments)
	return nil
}

func (receiver geospatialReceiver) refinePlanFinish(lent *lent[RefineReport], completion C.usearch_completion_t,
	plan *GeospatialRefinePlan) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_refine_plan_finish_t](lent)
	*arguments = C.usearch_geospatial_refine_plan_finish_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome(),
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
	}
	C.usearch_geospatial_refine_plan_finish(arguments)
	return nil
}

func (receiver geospatialReceiver) health(lent *lent[Reachability], completion C.usearch_completion_t) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_health_t](lent)
	*arguments = C.usearch_geospatial_health_t{
		index:        index,
		executor:     executor,
		outcome:      lent.outcome(),
		completion:   completion,
		reachability: (*C.usearch_reachability_t)(unsafe.Pointer(lent.result)),
	}
	C.usearch_geospatial_health(arguments)
	return nil
}

func (receiver geospatialReceiver) search(lent *lent[MatchesView], completion C.usearch_completion_t,
	queries []GeospatialObservation, relation GeospatialRelation, wanted uint, arena *Arena,
	config *GeospatialSearchConfig) error {
	options := configOf(config)
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_search_t](lent)
	lent.result.queries = uint(len(queries))
	*arguments = C.usearch_geospatial_search_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome(),
		completion: completion,
		count:      C.size_t(len(queries)),
		queries:    lent.observations(queries),
		relation:   C.usearch_geospatial_relation_t(relation),
		tolerance: C.usearch_geospatial_tolerance_t{
			metres:  C.double(options.Metres),
			seconds: C.double(options.Seconds),
		},
		matches:          &lent.result.raw,
		arena:            outputs,
		wanted:           C.size_t(wanted),
		expansion_factor: C.size_t(options.ExpansionFactor),
		traversal:        C.usearch_traversal_t(options.Traversal),
		predicate:        lent.predicate(options.Predicate),
		progress:         lent.progress(options.Progress, options.ProgressSteps),
		stats:            lent.stats(options.Stats),
	}
	C.usearch_geospatial_search(arguments)
	return nil
}

func (receiver geospatialReceiver) within(lent *lent[MatchesTape], completion C.usearch_completion_t,
	queries []GeospatialObservation, relation GeospatialRelation, metres float64, arena *Arena,
	config *GeospatialWithinConfig) error {
	options := configOf(config)
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_within_t](lent)
	lent.result.queries = uint(len(queries))
	*arguments = C.usearch_geospatial_within_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome(),
		completion: completion,
		count:      C.size_t(len(queries)),
		queries:    lent.observations(queries),
		relation:   C.usearch_geospatial_relation_t(relation),
		tolerance: C.usearch_geospatial_tolerance_t{
			metres:  C.double(metres),
			seconds: C.double(options.Seconds),
		},
		matches:   &lent.result.raw,
		written:   (*C.size_t)(unsafe.Pointer(&lent.result.written)),
		arena:     outputs,
		traversal: C.usearch_traversal_t(options.Traversal),
		predicate: lent.predicate(options.Predicate),
		progress:  lent.progress(options.Progress, options.ProgressSteps),
		stats:     lent.stats(options.Stats),
	}
	C.usearch_geospatial_within(arguments)
	return nil
}

func (receiver geospatialReceiver) reshard(lent *lent[RefineReport], completion C.usearch_completion_t, sources,
	targets []*GeospatialIndex, config *GeospatialReshardConfig) error {
	options := configOf(config)
	// A reshard names its handles in its lists, so the receiver lends only its executor.
	executor, err := receiver.executor.lower()
	if err != nil {
		return err
	}
	sourceHandles, targetHandles := geospatialHandles(sources), geospatialHandles(targets)
	arguments := open[C.usearch_geospatial_reshard_t](lent)
	plan, phase := options.Plan.lower(&lent.call)
	*arguments = C.usearch_geospatial_reshard_t{
		sources_count: C.size_t(len(sourceHandles)),
		sources:       (*C.usearch_geospatial_index_t)(lent.lend(unsafe.Pointer(unsafe.SliceData(sourceHandles)), len(sourceHandles))),
		targets_count: C.size_t(len(targetHandles)),
		targets:       (*C.usearch_geospatial_index_t)(lent.lend(unsafe.Pointer(unsafe.SliceData(targetHandles)), len(targetHandles))),
		executor:      executor,
		outcome:       lent.outcome(),
		completion:    completion,
		report:        (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
		refine: C.usearch_refine_options_t{
			expansion_factor: C.size_t(options.ExpansionFactor),
			memory_limit:     C.size_t(options.MemoryLimit),
		},
		seed:     C.uint64_t(options.Seed),
		progress: lent.progress(options.Progress, options.ProgressSteps),
		stats:    lent.stats(options.Stats),
		plan:     plan,
		phase:    phase,
	}
	C.usearch_geospatial_reshard(arguments)
	return nil
}

// geospatialHandles lists the handles of indexes.
func geospatialHandles(indexes []*GeospatialIndex) []C.usearch_geospatial_index_t {
	handles := make([]C.usearch_geospatial_index_t, len(indexes))
	for position, index := range indexes {
		handles[position] = index.handle
	}
	return handles
}

func (receiver geospatialReceiver) reshardPlanFinish(lent *lent[RefineReport], completion C.usearch_completion_t,
	plan *GeospatialReshardPlan) error {
	// A reshard names its handles in its lists, so the receiver lends only its executor.
	executor, err := receiver.executor.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_reshard_plan_finish_t](lent)
	*arguments = C.usearch_geospatial_reshard_plan_finish_t{
		executor:   executor,
		outcome:    lent.outcome(),
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
	}
	C.usearch_geospatial_reshard_plan_finish(arguments)
	return nil
}

func (receiver geospatialReceiver) sample(lent *lent[DenseDraws], completion C.usearch_completion_t, wanted uint,
	arena *Arena, config *GeospatialSampleConfig) error {
	options := configOf(config)
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	// The call has one count for both, so only this side sees the keys and their places disagree.
	if len(options.Shown) != len(options.ShownCoordinates) {
		return fail(StatusShapeMismatch, "the shown keys and coordinates disagree in length")
	}
	arguments := open[C.usearch_geospatial_sample_t](lent)
	*arguments = C.usearch_geospatial_sample_t{
		index:             index,
		executor:          executor,
		outcome:           lent.outcome(),
		completion:        completion,
		shown_count:       C.size_t(len(options.Shown)),
		shown:             lent.keys(options.Shown),
		shown_coordinates: lent.points(options.ShownCoordinates),
		focus_count:       C.size_t(len(options.Focus)),
		focus:             lent.keys(options.Focus),
		keys:              &lent.result.keys,
		entry_ordinals:    &lent.result.entryOrdinals,
		parents:           &lent.result.parents,
		coordinates:       &lent.result.coordinates,
		written:           (*C.size_t)(unsafe.Pointer(&lent.result.written)),
		arena:             outputs,
		wanted:            C.size_t(wanted),
		seed:              C.uint64_t(options.Seed),
		expansion_factor:  C.size_t(options.ExpansionFactor),
		progress:          lent.progress(options.Progress, options.ProgressSteps),
		stats:             lent.stats(options.Stats),
	}
	C.usearch_geospatial_sample(arguments)
	return nil
}

func (receiver geospatialReceiver) save(lent *lent[struct{}], completion C.usearch_completion_t, path string,
	config *GeospatialSaveConfig) error {
	options := configOf(config)
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_save_t](lent)
	*arguments = C.usearch_geospatial_save_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome(),
		completion: completion,
		path:       lent.text(path),
		progress:   lent.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_geospatial_save(arguments)
	return nil
}

func (receiver geospatialReceiver) saveBuffer(lent *lent[Monolith], completion C.usearch_completion_t, arena *Arena,
	config *GeospatialSaveBufferConfig) error {
	options := configOf(config)
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_save_buffer_t](lent)
	*arguments = C.usearch_geospatial_save_buffer_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome(),
		completion: completion,
		monolith:   &lent.result.raw,
		arena:      outputs,
		progress:   lent.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_geospatial_save_buffer(arguments)
	return nil
}

// endregion: Calls

// region: Cursors

// GeospatialCursor is a walk over the index contents; an entry once passed is never revisited.
//
// Its pulls come from the embedded GeospatialCursorOn and run on the calling goroutine; On runs
// them blocking on an executor, named again for every pull.
type GeospatialCursor struct {
	GeospatialCursorOn
	handle C.usearch_geospatial_cursor_t
}

// GeospatialCursorOn is the receiver whose pulls block until their call is final: on the calling
// goroutine when embedded in a GeospatialCursor, and on an executor when GeospatialCursorOn.On
// makes it. Binding-only.
type GeospatialCursorOn struct {
	childReceiver[C.usearch_geospatial_cursor_t, C.usearch_geospatial_index_t]
}

// On is the receiver whose pulls block on executor. Binding-only.
func (on GeospatialCursorOn) On(executor *Executor) GeospatialCursorOn {
	on.executor = executor
	return on
}

// Cursor opens a walk over the current contents. Without starts it yields every entry as stored;
// with them, the start keys' own entries first in order, then the entries those link to
// breadth-first, each once.
func (index *GeospatialIndex) Cursor(starts ...Key) (*GeospatialCursor, error) {
	handle, _, err := index.lower()
	if err != nil {
		return nil, err
	}
	var lent lent[C.usearch_geospatial_cursor_t]
	arguments := open[C.usearch_geospatial_cursor_init_t](&lent)
	*arguments = C.usearch_geospatial_cursor_init_t{
		index:        handle,
		outcome:      lent.outcome(),
		starts_count: C.size_t(len(starts)),
		starts:       (*C.usearch_key_t)(lent.lend(unsafe.Pointer(unsafe.SliceData(starts)), len(starts))),
		cursor:       lent.result,
	}
	C.usearch_geospatial_cursor_init(arguments)
	cursor := new(GeospatialCursor)
	if cursor.handle, err = block(&lent, nil); err != nil {
		return nil, err
	}
	cursor.childReceiver = childReceiver[C.usearch_geospatial_cursor_t, C.usearch_geospatial_index_t]{
		receiver: receiver[C.usearch_geospatial_cursor_t]{handle: &cursor.handle}, parent: &index.handle}
	return cursor, nil
}

// NextKeys copies up to wanted next keys into arena; a batch of none means the walk is exhausted.
func (on GeospatialCursorOn) NextKeys(wanted uint, arena *Arena) (DenseBatch, error) {
	var lent lent[DenseBatch]
	return block(&lent, on.next(&lent, wanted, arena, noLinks))
}

// NextLinks copies up to wanted next keys into arena, each with the keys it links to; a batch of
// none means the walk is exhausted.
func (on GeospatialCursorOn) NextLinks(wanted uint, arena *Arena) (DenseBatch, error) {
	var lent lent[DenseBatch]
	return block(&lent, on.next(&lent, wanted, arena, linksOf))
}

// next pulls a batch, exporting each entry's links into the slot links picks, if any.
func (on GeospatialCursorOn) next(lent *lent[DenseBatch], wanted uint, arena *Arena, links batchLinks) error {
	cursor, index, executor, outputs, err := on.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_geospatial_cursor_next_t](lent)
	*arguments = C.usearch_geospatial_cursor_next_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome(),
		completion: C.usearch_completion_blocking_k,
		cursor:     cursor,
		keys:       &lent.result.keys,
		links:      links(lent.result),
		written:    (*C.size_t)(unsafe.Pointer(&lent.result.written)),
		arena:      outputs,
		wanted:     C.size_t(wanted),
	}
	C.usearch_geospatial_cursor_next(arguments)
	return nil
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
