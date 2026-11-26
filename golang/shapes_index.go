// Similarity Search over 3D point clouds of varying size, with no correspondence between points.
//
// File: golang/shapes_index.go
// Author: Ash Vardanian
// Date: November 16, 2026

// Every verb is a batch over a tape, the Apache Arrow layout of ragged entries, so molecules or
// scans of any point counts share one batch, and a single shape is a one-entry tape. Cursor
// batches and draws are the dense modality's own types, since the library fills them through the
// same C structures.

package usearch

import "unsafe"

/*
// noescape and nocallback follow the rule in types.go.
#cgo noescape usearch_shapes_metric_from_name
#cgo nocallback usearch_shapes_metric_from_name
#cgo noescape usearch_shapes_init_ram
#cgo nocallback usearch_shapes_init_ram
#cgo noescape usearch_shapes_init_vram
#cgo nocallback usearch_shapes_init_vram
#cgo noescape usearch_shapes_view_ram
#cgo nocallback usearch_shapes_view_ram
#cgo noescape usearch_shapes_cursor_init
#cgo nocallback usearch_shapes_cursor_init
#cgo noescape usearch_shapes_view_nvme
#cgo nocallback usearch_shapes_view_nvme
#cgo noescape usearch_shapes_reserve
#cgo nocallback usearch_shapes_reserve
#cgo noescape usearch_shapes_clear
#cgo nocallback usearch_shapes_clear
#cgo noescape usearch_shapes_describe
#cgo nocallback usearch_shapes_describe
#cgo noescape usearch_shapes_count
#cgo nocallback usearch_shapes_count
#cgo noescape usearch_shapes_get
#cgo nocallback usearch_shapes_get
#cgo noescape usearch_shapes_health
#cgo nocallback usearch_shapes_health
#cgo noescape usearch_shapes_metadata_read
#cgo nocallback usearch_shapes_metadata_read
#cgo noescape usearch_shapes_metadata_read_buffer
#cgo nocallback usearch_shapes_metadata_read_buffer
#cgo noescape usearch_shapes_load_ram
#cgo noescape usearch_shapes_load_vram
#cgo noescape usearch_shapes_load_ram_buffer
#cgo noescape usearch_shapes_load_vram_buffer
#cgo noescape usearch_shapes_insert
#cgo noescape usearch_shapes_refine
#cgo noescape usearch_shapes_refine_plan_finish
#cgo nocallback usearch_shapes_refine_plan_finish
#cgo noescape usearch_shapes_search
#cgo noescape usearch_shapes_reshard
#cgo noescape usearch_shapes_reshard_plan_finish
#cgo nocallback usearch_shapes_reshard_plan_finish
#cgo noescape usearch_shapes_sample
#cgo noescape usearch_shapes_save
#cgo noescape usearch_shapes_save_buffer
#cgo noescape usearch_shapes_cursor_next
#cgo nocallback usearch_shapes_cursor_next

#include "shim.h"
#include "usearch/shapes_index.h"
*/
import "C"

// region: Geometry

// ShapesMetric names how two shapes compare, with no correspondence between their points.
type ShapesMetric int32

const (
	// ShapesMetricUnknown tags a metric the library could not name.
	ShapesMetricUnknown ShapesMetric = C.usearch_shapes_metric_unknown_k
	// ShapesMetricChamfer is the mean distance from each point to the other shape's nearest one.
	ShapesMetricChamfer ShapesMetric = C.usearch_shapes_metric_chamfer_k
	// ShapesMetricHausdorff is the largest distance from a point to the other shape's nearest one.
	ShapesMetricHausdorff ShapesMetric = C.usearch_shapes_metric_hausdorff_k
	// ShapesMetricGromovWasserstein compares how each shape's own points sit relative to one
	// another, never the coordinates.
	ShapesMetricGromovWasserstein ShapesMetric = C.usearch_shapes_metric_gromov_wasserstein_k
	// ShapesMetricWassersteinProcrustes pairs the points by an exact transport plan and superposes
	// the shapes by the rotation or reflection that plan implies.
	ShapesMetricWassersteinProcrustes ShapesMetric = C.usearch_shapes_metric_wasserstein_procrustes_k
)

// String renders the metric as the library spells it.
func (metric ShapesMetric) String() string {
	return C.GoString(C.usearch_shapes_metric_name(C.usearch_shapes_metric_t(metric)))
}

// ShapesMetricFromName reads a metric back from the spelling String gives it, through
// usearch_shapes_metric_from_name; a name the library does not spell fails with
// StatusInvalidArgument.
func ShapesMetricFromName(name string) (ShapesMetric, error) {
	metric := C.usearch_shapes_metric_t(ShapesMetricUnknown)
	if status := C.usearch_shapes_metric_from_name(view(name), &metric); status != C.usearch_status_success_k {
		return ShapesMetricUnknown, fail(Status(status), "no shapes metric is spelled \""+name+"\"")
	}
	return ShapesMetric(metric), nil
}

// ShapesOptions holds the construction options of a shapes index, the fields ShapesInitRam and
// ShapesInitVram share.
type ShapesOptions struct {
	// Metric is how two shapes compare.
	Metric ShapesMetric
	// Quantization is the scalar type points are stored in: ScalarF16 or ScalarF32.
	Quantization Scalar
	// Points fixes the points of every shape; zero admits any count per entry.
	Points uint
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

// DefaultShapesOptions is a Chamfer index over f32 points, with every optional field derived.
func DefaultShapesOptions() ShapesOptions {
	return ShapesOptions{Metric: ShapesMetricChamfer, Quantization: ScalarF32}
}

// ShapesMetadata is the metric and geometry a checkpoint froze.
type ShapesMetadata struct {
	// Metric is the distance kind the checkpoint scores with.
	Metric ShapesMetric
	// Quantization is the scalar type its points are stored in.
	Quantization Scalar
	// Points is the points every shape holds, zero when they vary.
	Points uint
	// Connectivity is the connections per node the graph was built with.
	Connectivity uint
}

// ShapesIndexDescription is a live index's frozen metadata and current counters, each read once,
// so under concurrent insertions they may come from slightly different moments.
type ShapesIndexDescription struct {
	// Metadata is the frozen metric, quantization, points and connectivity, as a checkpoint
	// reports them.
	Metadata ShapesMetadata
	// Size is the number of stored entries.
	Size uint
	// Capacity is the number of entries hosted without growing again.
	Capacity uint
	// TotalPoints sums the points of every stored entry.
	TotalPoints uint
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

// ShapesKeyStats is what Count reports about one key: its entries and their extent.
type ShapesKeyStats struct {
	// Entries counts the entries stored under the key; zero means the index holds none.
	Entries uint
	// TotalPoints sums their points, the points a Get of the key writes.
	TotalPoints uint
	// LargestPoints is the most points any of them holds.
	LargestPoints uint
}

// endregion: Geometry

// region: Configs

// ShapesLoadRamConfig carries the optional settings of ShapesLoadRam.
type ShapesLoadRamConfig struct {
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

// ShapesLoadVramConfig carries the optional settings of ShapesLoadVram.
type ShapesLoadVramConfig struct {
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

// ShapesLoadRamBufferConfig carries the optional settings of ShapesLoadRamBuffer.
type ShapesLoadRamBufferConfig struct {
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

// ShapesLoadVramBufferConfig carries the optional settings of ShapesLoadVramBuffer.
type ShapesLoadVramBufferConfig struct {
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

// ShapesMetadataReadConfig carries the optional settings of ShapesMetadataRead.
type ShapesMetadataReadConfig struct {
	// Executor is the one the read runs on; nil reads on the calling goroutine.
	Executor *Executor
}

// ShapesMetadataReadBufferConfig carries the optional settings of ShapesMetadataReadBuffer.
type ShapesMetadataReadBufferConfig struct {
	// Executor is the one the read runs on; nil reads on the calling goroutine.
	Executor *Executor
}

// ShapesInsertConfig carries the optional settings of Insert. A zero value derives every setting.
type ShapesInsertConfig struct {
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

// ShapesSampleConfig carries the optional settings of Sample. A zero value draws across the whole
// index with nothing shown.
type ShapesSampleConfig struct {
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

// ShapesSaveConfig carries the optional settings of Save.
type ShapesSaveConfig struct {
	// Progress reports the shards written and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// ShapesSaveBufferConfig carries the optional settings of SaveBuffer.
type ShapesSaveBufferConfig struct {
	// Progress reports the bytes written and stops wherever it says to; a stopped save leaves no
	// monolith.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// ShapesViewNvmeConfig holds the optional settings of ShapesViewNvme.
type ShapesViewNvmeConfig struct {
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

// ShapesSearchConfig carries the optional settings of Search. A zero value derives every setting.
type ShapesSearchConfig struct {
	// ExpansionFactor is the breadth of the candidates the walk explores; zero derives it, and a scan
	// ignores it.
	ExpansionFactor uint
	// Traversal walks the graph by default; TraversalScan scores every entry exactly, on RAM and
	// NVMe handles.
	Traversal Traversal
	// Predicate admits only the keys it approves, one or a batch at a time; nil admits everything.
	// Non-matching entries still route the traversal.
	Predicate Admitter
	// Progress reports the queries resolved and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
	// Stats is where the traversal telemetry lands, overwritten by the call.
	Stats *TraversalStats
}

// ShapesRefineConfig holds the optional settings of Refine. A zero value derives every setting.
type ShapesRefineConfig struct {
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
	Plan *ShapesRefinePlan
}

// ShapesReshardConfig holds the optional settings of Reshard. A zero value derives every setting.
type ShapesReshardConfig struct {
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
	Plan *ShapesReshardPlan
}

// ShapesRefinePlan is one refine kept live across calls: the round in hand, and the root every
// round's health checks share. The zero value opens on the first Refine naming it, and
// RefinePlanFinish closes it; while it lives the index admits no insertion and no save, and each
// call names the index and executor it was opened with.
type ShapesRefinePlan = livePlan[C.usearch_shapes_refine_plan_t]

// ShapesReshardPlan is one reshard kept live across calls: its own sources and targets, the
// entries still to move, then the rounds of every target that took entries. The zero value opens
// on the first Reshard naming it, and ReshardPlanFinish closes it; while it lives every handle it
// names admits no insertion and no save, and each call names the executor it was opened with.
type ShapesReshardPlan = livePlan[C.usearch_shapes_reshard_plan_t]

// endregion: Configs

// region: Index

// ShapesIndex is a Similarity Search Index over 3D point clouds whose point counts vary per entry:
// molecules, scans, or any shape compared without a correspondence between its points.
//
// A handle may be shared across goroutines. On a RAM handle, insertions and searches overlap
// freely, and serialization demands a mutation-free window while readers keep searching.
//
// Its verbs come from the embedded ShapesIndexOn and run on the calling goroutine; On runs the
// same verbs blocking on an executor, and Deferred submits them there.
type ShapesIndex struct {
	ShapesIndexOn
	handle C.usearch_shapes_index_t
}

// shapesIndexOf wraps the handle a lifecycle verb wrote.
func shapesIndexOf(handle C.usearch_shapes_index_t, err error) (*ShapesIndex, error) {
	if err != nil {
		return nil, err
	}
	index := &ShapesIndex{handle: handle}
	index.ShapesIndexOn = ShapesIndexOn{shapesReceiver{receiver[C.usearch_shapes_index_t]{handle: &index.handle}}}
	return index, nil
}

// Close releases the index, waiting for every deferred call naming it. It is idempotent, and a
// closed handle must not be used again.
func (index *ShapesIndex) Close() error {
	if index == nil || index.handle == nil {
		return nil
	}
	C.usearch_shapes_free(index.handle)
	index.handle = nil
	return nil
}

// ShapesInitRam creates an empty, mutable index in host memory, one shard per NUMA node
// memoryDomains names; empty derives every node the process may use.
func ShapesInitRam(options ShapesOptions, memoryDomains []int32) (*ShapesIndex, error) {
	var call callWithResult[C.usearch_shapes_index_t]
	arguments := open[C.usearch_shapes_init_ram_t](&call)
	*arguments = C.usearch_shapes_init_ram_t{
		index:          call.result,
		outcome:        call.outcome(),
		metric:         C.usearch_shapes_metric_t(options.Metric),
		quantization:   C.usearch_scalar_t(options.Quantization),
		points:         C.size_t(options.Points),
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		ram:            call.ram(memoryDomains, options.CPUClassCeiling),
	}
	C.usearch_shapes_init_ram(arguments)
	return shapesIndexOf(block(&call, nil))
}

// ShapesInitVram creates an empty, mutable index in device memory, sharded across the CUDA
// ordinals devices names and never replicated; empty derives one free device.
func ShapesInitVram(options ShapesOptions, devices []int32) (*ShapesIndex, error) {
	var call callWithResult[C.usearch_shapes_index_t]
	arguments := open[C.usearch_shapes_init_vram_t](&call)
	*arguments = C.usearch_shapes_init_vram_t{
		index:          call.result,
		outcome:        call.outcome(),
		metric:         C.usearch_shapes_metric_t(options.Metric),
		quantization:   C.usearch_scalar_t(options.Quantization),
		points:         C.size_t(options.Points),
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		vram:           call.vram(devices, options.GPUClassCeiling),
	}
	C.usearch_shapes_init_vram(arguments)
	return shapesIndexOf(block(&call, nil))
}

// ShapesLoadRam copies a checkpoint into a new, mutable index in host memory. The path names a
// manifest of a sharded checkpoint or a single .safegraphs monolith.
func ShapesLoadRam(path string, config *ShapesLoadRamConfig) (*ShapesIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_shapes_index_t]
	arguments := open[C.usearch_shapes_load_ram_t](&call)
	*arguments = C.usearch_shapes_load_ram_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome(),
		path:           call.text(path),
		ram:            call.ram(options.MemoryDomains, options.CPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		progress:       call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_shapes_load_ram(arguments)
	return shapesIndexOf(block(&call, nil))
}

// ShapesLoadVram copies a checkpoint into a new, mutable index in device memory, with the
// placement rules of ShapesLoadRam.
func ShapesLoadVram(path string, config *ShapesLoadVramConfig) (*ShapesIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_shapes_index_t]
	arguments := open[C.usearch_shapes_load_vram_t](&call)
	*arguments = C.usearch_shapes_load_vram_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome(),
		path:           call.text(path),
		vram:           call.vram(options.Devices, options.GPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		progress:       call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_shapes_load_vram(arguments)
	return shapesIndexOf(block(&call, nil))
}

// ShapesLoadRamBuffer copies a monolith from a buffer into a new, mutable index in host memory, as
// ShapesLoadRam does a path.
func ShapesLoadRamBuffer(buffer []byte, config *ShapesLoadRamBufferConfig) (*ShapesIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_shapes_index_t]
	arguments := open[C.usearch_shapes_load_ram_buffer_t](&call)
	*arguments = C.usearch_shapes_load_ram_buffer_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome(),
		buffer:         call.bytes(buffer),
		ram:            call.ram(options.MemoryDomains, options.CPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		progress:       call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_shapes_load_ram_buffer(arguments)
	return shapesIndexOf(block(&call, nil))
}

// ShapesLoadVramBuffer copies a monolith from a buffer into a new, mutable index in device memory.
func ShapesLoadVramBuffer(buffer []byte, config *ShapesLoadVramBufferConfig) (*ShapesIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_shapes_index_t]
	arguments := open[C.usearch_shapes_load_vram_buffer_t](&call)
	*arguments = C.usearch_shapes_load_vram_buffer_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome(),
		buffer:         call.bytes(buffer),
		vram:           call.vram(options.Devices, options.GPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		progress:       call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_shapes_load_vram_buffer(arguments)
	return shapesIndexOf(block(&call, nil))
}

// ShapesViewRam serves a checkpoint in place through a memory mapping, as a read-only index whose
// mutating verbs report StatusReadOnly; the files must stay unmodified while it lives.
func ShapesViewRam(path string) (*ShapesIndex, error) {
	var call callWithResult[C.usearch_shapes_index_t]
	arguments := open[C.usearch_shapes_view_ram_t](&call)
	*arguments = C.usearch_shapes_view_ram_t{
		index:   call.result,
		outcome: call.outcome(),
		path:    call.text(path),
	}
	C.usearch_shapes_view_ram(arguments)
	return shapesIndexOf(block(&call, nil))
}

// ShapesViewNvme serves a checkpoint in place from NVMe drives, read through the executor's
// submission rings, as a read-only index.
func ShapesViewNvme(path string, config *ShapesViewNvmeConfig) (*ShapesIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_shapes_index_t]
	arguments := open[C.usearch_shapes_view_nvme_t](&call)
	*arguments = C.usearch_shapes_view_nvme_t{
		index:    call.result,
		executor: executor,
		outcome:  call.outcome(),
		path:     call.text(path),
		nvme:     call.nvme(options.Paths, options.CPUClassCeiling),
	}
	C.usearch_shapes_view_nvme(arguments)
	return shapesIndexOf(block(&call, nil))
}

// ShapesMetadataRead reads a checkpoint's metric and geometry without loading it.
func ShapesMetadataRead(path string, config *ShapesMetadataReadConfig) (ShapesMetadata, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return ShapesMetadata{}, err
	}
	var call callWithResult[ShapesMetadata]
	arguments := open[C.usearch_shapes_metadata_read_t](&call)
	*arguments = C.usearch_shapes_metadata_read_t{
		executor: executor,
		outcome:  call.outcome(),
		path:     call.text(path),
		metadata: (*C.usearch_shapes_metadata_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_shapes_metadata_read(arguments)
	return block(&call, nil)
}

// ShapesMetadataReadBuffer reads the metric and geometry out of a serialized monolith buffer.
func ShapesMetadataReadBuffer(buffer []byte, config *ShapesMetadataReadBufferConfig) (ShapesMetadata, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return ShapesMetadata{}, err
	}
	var call callWithResult[ShapesMetadata]
	arguments := open[C.usearch_shapes_metadata_read_buffer_t](&call)
	*arguments = C.usearch_shapes_metadata_read_buffer_t{
		executor: executor,
		outcome:  call.outcome(),
		buffer:   call.bytes(buffer),
		metadata: (*C.usearch_shapes_metadata_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_shapes_metadata_read_buffer(arguments)
	return block(&call, nil)
}

// endregion: Index

// region: Receivers

// shapesReceiver is an index and the executor its verbs run on, nil for the calling goroutine.
type shapesReceiver struct {
	receiver[C.usearch_shapes_index_t]
}

// ShapesIndexOn is the receiver whose verbs block until their call is final: on the calling
// goroutine when embedded in a ShapesIndex, and on an executor when ShapesIndexOn.On makes it.
// Binding-only.
type ShapesIndexOn struct {
	shapesReceiver
}

// ShapesIndexDeferredOn is the receiver whose verbs submit their calls to an executor and answer
// a Deferred, offering every verb of ShapesIndexOn with the same arguments but the blocking-only
// sugar. Binding-only.
//
// The library reads a deferred call's inputs until the call is final, so the slices a verb is
// handed must not change until then; its arena is viewed, as a blocking verb views it.
type ShapesIndexDeferredOn struct {
	shapesReceiver
}

// On is the receiver whose verbs block on executor. Binding-only.
func (on ShapesIndexOn) On(executor *Executor) ShapesIndexOn {
	on.executor = executor
	return on
}

// Deferred is the receiver whose verbs submit their calls to executor, each answering a Deferred
// from the same arguments, or the refusal it caught before submitting anything. Binding-only.
func (on ShapesIndexOn) Deferred(executor *Executor) ShapesIndexDeferredOn {
	on.executor = executor
	return ShapesIndexDeferredOn{on.shapesReceiver}
}

// Reserve claims capacity for a number of incoming entries and the points they hold, ahead of
// insertion. Both are desired totals, including what the index holds.
func (on ShapesIndexOn) Reserve(entries, points uint) error {
	var call callWithResult[struct{}]
	_, err := block(&call, on.reserve(&call, C.usearch_completion_blocking_k, entries, points))
	return err
}

// Reserve submits ShapesIndexOn.Reserve.
func (on ShapesIndexDeferredOn) Reserve(entries, points uint) (*Deferred[struct{}], error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.reserve(&deferred.call, C.usearch_completion_deferred_k, entries, points))
}

// Clear erases all entries, keeping the reserved capacity. No other verb may overlap it on the
// same handle.
func (on ShapesIndexOn) Clear() error {
	var call callWithResult[struct{}]
	_, err := block(&call, on.clear(&call, C.usearch_completion_blocking_k))
	return err
}

// Clear submits ShapesIndexOn.Clear.
func (on ShapesIndexDeferredOn) Clear() (*Deferred[struct{}], error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.clear(&deferred.call, C.usearch_completion_deferred_k))
}

// Describe reports the frozen metadata and the current counters in one call.
func (on ShapesIndexOn) Describe() (ShapesIndexDescription, error) {
	var call callWithResult[ShapesIndexDescription]
	return block(&call, on.describe(&call, C.usearch_completion_blocking_k))
}

// Describe submits ShapesIndexOn.Describe.
func (on ShapesIndexDeferredOn) Describe() (*Deferred[ShapesIndexDescription], error) {
	deferred := new(Deferred[ShapesIndexDescription])
	return submitted(deferred, on.describe(&deferred.call, C.usearch_completion_deferred_k))
}

// Count reports how many entries are stored under a key, with the extent of their points.
func (on ShapesIndexOn) Count(key Key) (ShapesKeyStats, error) {
	var call callWithResult[ShapesKeyStats]
	return block(&call, on.count(&call, C.usearch_completion_blocking_k, key))
}

// Count submits ShapesIndexOn.Count.
func (on ShapesIndexDeferredOn) Count(key Key) (*Deferred[ShapesKeyStats], error) {
	deferred := new(Deferred[ShapesKeyStats])
	return submitted(deferred, on.count(&deferred.call, C.usearch_completion_deferred_k, key))
}

// Contains reports whether the index holds at least one entry under a key, as a nonzero Count.
func (on ShapesIndexOn) Contains(key Key) (bool, error) {
	counted, err := on.Count(key)
	return counted.Entries != 0, err
}

// Get exports every shape stored under a key into arena, one tape entry per entry sharing the
// key, in insertion order and pose, converted to the outputsScalar kind and read back with
// TapeEntry.
//
// The call rewinds arena, and the tape stays readable until it serves another call.
func (on ShapesIndexOn) Get(key Key, outputsScalar Scalar, arena *Arena) (TapeView, error) {
	var call callWithResult[TapeView]
	return block(&call, on.get(&call, C.usearch_completion_blocking_k, key, outputsScalar, arena))
}

// Get submits ShapesIndexOn.Get.
func (on ShapesIndexDeferredOn) Get(key Key, outputsScalar Scalar, arena *Arena) (*Deferred[TapeView], error) {
	deferred := new(Deferred[TapeView])
	return submitted(deferred, on.get(&deferred.call, C.usearch_completion_deferred_k, key, outputsScalar, arena))
}

// Insert adds a tape of shapes under their keys - the native batch operation, where a single
// shape is a one-entry tape. The points' scalar type is independent of the stored quantization,
// and a batch is all-or-nothing and becomes visible at once.
func (on ShapesIndexOn) Insert(keys []Key, shapes Tape, config *ShapesInsertConfig) error {
	var call callWithResult[struct{}]
	_, err := block(&call, on.insert(&call, C.usearch_completion_blocking_k, keys, shapes, config))
	return err
}

// Insert submits ShapesIndexOn.Insert.
func (on ShapesIndexDeferredOn) Insert(keys []Key, shapes Tape, config *ShapesInsertConfig) (*Deferred[struct{}],
	error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.insert(&deferred.call, C.usearch_completion_deferred_k, keys, shapes, config))
}

// Refine re-selects the neighbors of every stored entry against the finished graph, as the dense
// Refine does; a caller repeats rounds until the report's Verdict reads RefineConverged.
func (on ShapesIndexOn) Refine(config *ShapesRefineConfig) (RefineReport, error) {
	var call callWithResult[RefineReport]
	return block(&call, on.refine(&call, C.usearch_completion_blocking_k, config))
}

// Refine submits ShapesIndexOn.Refine.
func (on ShapesIndexDeferredOn) Refine(config *ShapesRefineConfig) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.refine(&deferred.call, C.usearch_completion_deferred_k, config))
}

// Health measures how much of the graph a search reaches from the root. Read-only.
func (on ShapesIndexOn) Health() (Reachability, error) {
	var call callWithResult[Reachability]
	return block(&call, on.health(&call, C.usearch_completion_blocking_k))
}

// Health submits ShapesIndexOn.Health.
func (on ShapesIndexDeferredOn) Health() (*Deferred[Reachability], error) {
	deferred := new(Deferred[Reachability])
	return submitted(deferred, on.health(&deferred.call, C.usearch_completion_deferred_k))
}

// Search resolves a tape of queries into up to wanted nearest matches apiece, written into arena,
// and filtered when the config holds a predicate. Never writes to the graph.
func (on ShapesIndexOn) Search(queries Tape, wanted uint, arena *Arena, config *ShapesSearchConfig) (MatchesView,
	error) {
	var call callWithResult[MatchesView]
	return block(&call, on.search(&call, C.usearch_completion_blocking_k, queries, wanted, arena, config))
}

// Search submits ShapesIndexOn.Search.
func (on ShapesIndexDeferredOn) Search(queries Tape, wanted uint, arena *Arena,
	config *ShapesSearchConfig) (*Deferred[MatchesView], error) {
	deferred := new(Deferred[MatchesView])
	return submitted(deferred, on.search(&deferred.call, C.usearch_completion_deferred_k, queries, wanted, arena,
		config))
}

// Reshard moves the entries of sources into targets, keeping their links rather than re-inserting
// them, then runs a round on every target that took entries: several sources gather into one
// target, and one source cuts into one graph-local part per target. The receiver contributes only
// its executor, and a refusal moves nothing.
func (on ShapesIndexOn) Reshard(sources, targets []*ShapesIndex, config *ShapesReshardConfig) (RefineReport, error) {
	var call callWithResult[RefineReport]
	return block(&call, on.reshard(&call, C.usearch_completion_blocking_k, sources, targets, config))
}

// Reshard submits ShapesIndexOn.Reshard.
func (on ShapesIndexDeferredOn) Reshard(sources, targets []*ShapesIndex,
	config *ShapesReshardConfig) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.reshard(&deferred.call, C.usearch_completion_deferred_k, sources, targets, config))
}

// RefinePlanFinish closes a live refine plan and frees it whatever the call answers, reporting
// what the whole plan did to the graph.
func (on ShapesIndexOn) RefinePlanFinish(plan *ShapesRefinePlan) (RefineReport, error) {
	var call callWithResult[RefineReport]
	return block(&call, on.refinePlanFinish(&call, C.usearch_completion_blocking_k, plan))
}

// RefinePlanFinish submits ShapesIndexOn.RefinePlanFinish.
func (on ShapesIndexDeferredOn) RefinePlanFinish(plan *ShapesRefinePlan) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.refinePlanFinish(&deferred.call, C.usearch_completion_deferred_k, plan))
}

// ReshardPlanFinish closes a live reshard plan on this receiver's executor and frees it whatever
// the call answers, reporting what the whole plan did.
func (on ShapesIndexOn) ReshardPlanFinish(plan *ShapesReshardPlan) (RefineReport, error) {
	var call callWithResult[RefineReport]
	return block(&call, on.reshardPlanFinish(&call, C.usearch_completion_blocking_k, plan))
}

// ReshardPlanFinish submits ShapesIndexOn.ReshardPlanFinish.
func (on ShapesIndexDeferredOn) ReshardPlanFinish(plan *ShapesReshardPlan) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.reshardPlanFinish(&deferred.call, C.usearch_completion_deferred_k, plan))
}

// Sample draws up to wanted keys around the config's Focus, or across the whole index, each
// hanging from a shown or earlier draw and placed in three dimensions beside the shown ones.
func (on ShapesIndexOn) Sample(wanted uint, arena *Arena, config *ShapesSampleConfig) (DenseDraws, error) {
	var call callWithResult[DenseDraws]
	return block(&call, on.sample(&call, C.usearch_completion_blocking_k, wanted, arena, config))
}

// Sample submits ShapesIndexOn.Sample.
func (on ShapesIndexDeferredOn) Sample(wanted uint, arena *Arena, config *ShapesSampleConfig) (*Deferred[DenseDraws],
	error) {
	deferred := new(Deferred[DenseDraws])
	return submitted(deferred, on.sample(&deferred.call, C.usearch_completion_deferred_k, wanted, arena, config))
}

// Save writes the index under a path, in the layout its extension picks - one .safegraphs
// monolith, or a .safegraphs.json manifest beside its shards.
func (on ShapesIndexOn) Save(path string, config *ShapesSaveConfig) error {
	var call callWithResult[struct{}]
	_, err := block(&call, on.save(&call, C.usearch_completion_blocking_k, path, config))
	return err
}

// Save submits ShapesIndexOn.Save.
func (on ShapesIndexDeferredOn) Save(path string, config *ShapesSaveConfig) (*Deferred[struct{}], error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.save(&deferred.call, C.usearch_completion_deferred_k, path, config))
}

// SaveBuffer serializes the index as one monolith into arena.
func (on ShapesIndexOn) SaveBuffer(arena *Arena, config *ShapesSaveBufferConfig) (Monolith, error) {
	var call callWithResult[Monolith]
	return block(&call, on.saveBuffer(&call, C.usearch_completion_blocking_k, arena, config))
}

// SaveBuffer submits ShapesIndexOn.SaveBuffer.
func (on ShapesIndexDeferredOn) SaveBuffer(arena *Arena, config *ShapesSaveBufferConfig) (*Deferred[Monolith], error) {
	deferred := new(Deferred[Monolith])
	return submitted(deferred, on.saveBuffer(&deferred.call, C.usearch_completion_deferred_k, arena, config))
}

// endregion: Receivers

// region: Calls

func (receiver shapesReceiver) reserve(call *callWithResult[struct{}], completion C.usearch_completion_t, entries,
	points uint) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_reserve_t](call)
	*arguments = C.usearch_shapes_reserve_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		entries:    C.size_t(entries),
		points:     C.size_t(points),
	}
	C.usearch_shapes_reserve(arguments)
	return nil
}

func (receiver shapesReceiver) clear(call *callWithResult[struct{}], completion C.usearch_completion_t) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_clear_t](call)
	*arguments = C.usearch_shapes_clear_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
	}
	C.usearch_shapes_clear(arguments)
	return nil
}

func (receiver shapesReceiver) describe(call *callWithResult[ShapesIndexDescription], completion C.usearch_completion_t) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_describe_t](call)
	*arguments = C.usearch_shapes_describe_t{
		index:       index,
		executor:    executor,
		outcome:     call.outcome(),
		completion:  completion,
		description: (*C.usearch_shapes_index_description_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_shapes_describe(arguments)
	return nil
}

func (receiver shapesReceiver) count(call *callWithResult[ShapesKeyStats], completion C.usearch_completion_t, key Key) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_count_t](call)
	*arguments = C.usearch_shapes_count_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		key:        key.toRaw(),
		key_stats:  (*C.usearch_shapes_key_stats_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_shapes_count(arguments)
	return nil
}

func (receiver shapesReceiver) get(call *callWithResult[TapeView], completion C.usearch_completion_t, key Key,
	outputsScalar Scalar, arena *Arena) error {
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_get_t](call)
	call.result.arena = call.stamp(arena, outputs)
	call.result.width = 3
	*arguments = C.usearch_shapes_get_t{
		index:          index,
		executor:       executor,
		outcome:        call.outcome(),
		completion:     completion,
		key:            key.toRaw(),
		shapes:         &call.result.raw,
		written:        (*C.size_t)(unsafe.Pointer(&call.result.written)),
		arena:          outputs,
		outputs_scalar: C.usearch_scalar_t(outputsScalar),
	}
	C.usearch_shapes_get(arguments)
	return nil
}

func (receiver shapesReceiver) insert(call *callWithResult[struct{}], completion C.usearch_completion_t, keys []Key, shapes Tape,
	config *ShapesInsertConfig) error {
	options := configOf(config)
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	// The call has one count, so only this side sees the keys and the entries disagree.
	if uint(len(keys)) != shapes.Entries() {
		return fail(StatusShapeMismatch, "the key count does not match the tape's entries")
	}
	arguments := open[C.usearch_shapes_insert_t](call)
	*arguments = C.usearch_shapes_insert_t{
		index:            index,
		executor:         executor,
		outcome:          call.outcome(),
		completion:       completion,
		count:            C.size_t(len(keys)),
		keys:             call.keys(keys),
		shapes:           call.tape(shapes),
		expansion_factor: C.size_t(options.ExpansionFactor),
		progress:         call.progress(options.Progress, options.ProgressSteps),
		stats:            call.stats(options.Stats),
	}
	C.usearch_shapes_insert(arguments)
	return nil
}

func (receiver shapesReceiver) refine(call *callWithResult[RefineReport], completion C.usearch_completion_t,
	config *ShapesRefineConfig) error {
	options := configOf(config)
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_refine_t](call)
	plan, phase := options.Plan.lower(&call.pinner)
	*arguments = C.usearch_shapes_refine_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(call.result)),
		refine: C.usearch_refine_options_t{
			expansion_factor: C.size_t(options.ExpansionFactor),
			memory_limit:     C.size_t(options.MemoryLimit),
		},
		progress: call.progress(options.Progress, options.ProgressSteps),
		stats:    call.stats(options.Stats),
		plan:     plan,
		phase:    phase,
	}
	C.usearch_shapes_refine(arguments)
	return nil
}

func (receiver shapesReceiver) refinePlanFinish(call *callWithResult[RefineReport], completion C.usearch_completion_t,
	plan *ShapesRefinePlan) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_refine_plan_finish_t](call)
	*arguments = C.usearch_shapes_refine_plan_finish_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_shapes_refine_plan_finish(arguments)
	return nil
}

func (receiver shapesReceiver) health(call *callWithResult[Reachability], completion C.usearch_completion_t) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_health_t](call)
	*arguments = C.usearch_shapes_health_t{
		index:        index,
		executor:     executor,
		outcome:      call.outcome(),
		completion:   completion,
		reachability: (*C.usearch_reachability_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_shapes_health(arguments)
	return nil
}

func (receiver shapesReceiver) search(call *callWithResult[MatchesView], completion C.usearch_completion_t, queries Tape,
	wanted uint, arena *Arena, config *ShapesSearchConfig) error {
	options := configOf(config)
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_search_t](call)
	call.result.arena = call.stamp(arena, outputs)
	call.result.queries = queries.Entries()
	*arguments = C.usearch_shapes_search_t{
		index:            index,
		executor:         executor,
		outcome:          call.outcome(),
		completion:       completion,
		count:            C.size_t(queries.Entries()),
		queries:          call.tape(queries),
		matches:          &call.result.raw,
		arena:            outputs,
		wanted:           C.size_t(wanted),
		expansion_factor: C.size_t(options.ExpansionFactor),
		traversal:        C.usearch_traversal_t(options.Traversal),
		predicate:        call.predicate(options.Predicate),
		progress:         call.progress(options.Progress, options.ProgressSteps),
		stats:            call.stats(options.Stats),
	}
	C.usearch_shapes_search(arguments)
	return nil
}

func (receiver shapesReceiver) reshard(call *callWithResult[RefineReport], completion C.usearch_completion_t, sources,
	targets []*ShapesIndex, config *ShapesReshardConfig) error {
	options := configOf(config)
	// A reshard names its handles in its lists, so the receiver passes only its executor.
	executor, err := receiver.executor.lower()
	if err != nil {
		return err
	}
	sourceHandles, targetHandles := shapesHandles(sources), shapesHandles(targets)
	arguments := open[C.usearch_shapes_reshard_t](call)
	plan, phase := options.Plan.lower(&call.pinner)
	*arguments = C.usearch_shapes_reshard_t{
		sources_count: C.size_t(len(sourceHandles)),
		sources:       (*C.usearch_shapes_index_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(sourceHandles)), len(sourceHandles))),
		targets_count: C.size_t(len(targetHandles)),
		targets:       (*C.usearch_shapes_index_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(targetHandles)), len(targetHandles))),
		executor:      executor,
		outcome:       call.outcome(),
		completion:    completion,
		report:        (*C.usearch_refine_report_t)(unsafe.Pointer(call.result)),
		refine: C.usearch_refine_options_t{
			expansion_factor: C.size_t(options.ExpansionFactor),
			memory_limit:     C.size_t(options.MemoryLimit),
		},
		seed:     C.uint64_t(options.Seed),
		progress: call.progress(options.Progress, options.ProgressSteps),
		stats:    call.stats(options.Stats),
		plan:     plan,
		phase:    phase,
	}
	C.usearch_shapes_reshard(arguments)
	return nil
}

// shapesHandles lists the handles of indexes.
func shapesHandles(indexes []*ShapesIndex) []C.usearch_shapes_index_t {
	handles := make([]C.usearch_shapes_index_t, len(indexes))
	for position, index := range indexes {
		handles[position] = index.handle
	}
	return handles
}

func (receiver shapesReceiver) reshardPlanFinish(call *callWithResult[RefineReport], completion C.usearch_completion_t,
	plan *ShapesReshardPlan) error {
	// A reshard names its handles in its lists, so the receiver passes only its executor.
	executor, err := receiver.executor.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_reshard_plan_finish_t](call)
	*arguments = C.usearch_shapes_reshard_plan_finish_t{
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_shapes_reshard_plan_finish(arguments)
	return nil
}

func (receiver shapesReceiver) sample(call *callWithResult[DenseDraws], completion C.usearch_completion_t, wanted uint,
	arena *Arena, config *ShapesSampleConfig) error {
	options := configOf(config)
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	// The call has one count for both, so only this side sees the keys and their places disagree.
	if len(options.Shown) != len(options.ShownCoordinates) {
		return fail(StatusShapeMismatch, "the shown keys and coordinates disagree in length")
	}
	arguments := open[C.usearch_shapes_sample_t](call)
	call.result.arena = call.stamp(arena, outputs)
	*arguments = C.usearch_shapes_sample_t{
		index:             index,
		executor:          executor,
		outcome:           call.outcome(),
		completion:        completion,
		shown_count:       C.size_t(len(options.Shown)),
		shown:             call.keys(options.Shown),
		shown_coordinates: call.points(options.ShownCoordinates),
		focus_count:       C.size_t(len(options.Focus)),
		focus:             call.keys(options.Focus),
		keys:              &call.result.keys,
		entry_ordinals:    &call.result.entryOrdinals,
		parents:           &call.result.parents,
		coordinates:       &call.result.coordinates,
		written:           (*C.size_t)(unsafe.Pointer(&call.result.written)),
		arena:             outputs,
		wanted:            C.size_t(wanted),
		seed:              C.uint64_t(options.Seed),
		expansion_factor:  C.size_t(options.ExpansionFactor),
		progress:          call.progress(options.Progress, options.ProgressSteps),
		stats:             call.stats(options.Stats),
	}
	C.usearch_shapes_sample(arguments)
	return nil
}

func (receiver shapesReceiver) save(call *callWithResult[struct{}], completion C.usearch_completion_t, path string,
	config *ShapesSaveConfig) error {
	options := configOf(config)
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_save_t](call)
	*arguments = C.usearch_shapes_save_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		path:       call.text(path),
		progress:   call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_shapes_save(arguments)
	return nil
}

func (receiver shapesReceiver) saveBuffer(call *callWithResult[Monolith], completion C.usearch_completion_t, arena *Arena,
	config *ShapesSaveBufferConfig) error {
	options := configOf(config)
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_save_buffer_t](call)
	call.result.arena = call.stamp(arena, outputs)
	*arguments = C.usearch_shapes_save_buffer_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		monolith:   &call.result.raw,
		arena:      outputs,
		progress:   call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_shapes_save_buffer(arguments)
	return nil
}

// endregion: Calls

// region: Cursors

// ShapesCursor is a walk over the index contents; an entry once passed is never revisited.
//
// Its pulls come from the embedded ShapesCursorOn and run on the calling goroutine; On runs them
// blocking on an executor, named again for every pull.
type ShapesCursor struct {
	ShapesCursorOn
	handle C.usearch_shapes_cursor_t
}

// ShapesCursorOn is the receiver whose pulls block until their call is final: on the calling
// goroutine when embedded in a ShapesCursor, and on an executor when ShapesCursorOn.On makes it.
// Binding-only.
type ShapesCursorOn struct {
	childReceiver[C.usearch_shapes_cursor_t, C.usearch_shapes_index_t]
}

// On is the receiver whose pulls block on executor. Binding-only.
func (on ShapesCursorOn) On(executor *Executor) ShapesCursorOn {
	on.executor = executor
	return on
}

// Cursor opens a walk over the current contents. Without starts it yields every entry as stored;
// with them, the start keys' own entries first in order, then the entries those link to
// breadth-first, each once.
func (index *ShapesIndex) Cursor(starts ...Key) (*ShapesCursor, error) {
	handle, _, err := index.lower()
	if err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_shapes_cursor_t]
	arguments := open[C.usearch_shapes_cursor_init_t](&call)
	*arguments = C.usearch_shapes_cursor_init_t{
		index:        handle,
		outcome:      call.outcome(),
		starts_count: C.size_t(len(starts)),
		starts:       (*C.usearch_key_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(starts)), len(starts))),
		cursor:       call.result,
	}
	C.usearch_shapes_cursor_init(arguments)
	cursor := new(ShapesCursor)
	if cursor.handle, err = block(&call, nil); err != nil {
		return nil, err
	}
	cursor.childReceiver = childReceiver[C.usearch_shapes_cursor_t, C.usearch_shapes_index_t]{
		receiver: receiver[C.usearch_shapes_cursor_t]{handle: &cursor.handle}, parent: &index.handle}
	return cursor, nil
}

// NextKeys copies up to wanted next keys into arena; a batch of none means the walk is exhausted.
func (on ShapesCursorOn) NextKeys(wanted uint, arena *Arena) (DenseBatch, error) {
	var call callWithResult[DenseBatch]
	return block(&call, on.next(&call, wanted, arena, noLinks))
}

// NextLinks copies up to wanted next keys into arena, each with the keys it links to; a batch of
// none means the walk is exhausted.
func (on ShapesCursorOn) NextLinks(wanted uint, arena *Arena) (DenseBatch, error) {
	var call callWithResult[DenseBatch]
	return block(&call, on.next(&call, wanted, arena, linksOf))
}

// next pulls a batch, exporting each entry's links into the slot links picks, if any.
func (on ShapesCursorOn) next(call *callWithResult[DenseBatch], wanted uint, arena *Arena, links batchLinks) error {
	cursor, index, executor, outputs, err := on.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_shapes_cursor_next_t](call)
	call.result.arena = call.stamp(arena, outputs)
	*arguments = C.usearch_shapes_cursor_next_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: C.usearch_completion_blocking_k,
		cursor:     cursor,
		keys:       &call.result.keys,
		links:      links(call.result),
		written:    (*C.size_t)(unsafe.Pointer(&call.result.written)),
		arena:      outputs,
		wanted:     C.size_t(wanted),
	}
	C.usearch_shapes_cursor_next(arguments)
	return nil
}

// Close releases the cursor. It is idempotent.
func (cursor *ShapesCursor) Close() error {
	if cursor == nil || cursor.handle == nil {
		return nil
	}
	C.usearch_shapes_cursor_free(cursor.handle)
	cursor.handle = nil
	return nil
}

// endregion: Cursors
