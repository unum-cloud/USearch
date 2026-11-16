// Similarity Search over 3D point clouds of varying size, with no correspondence between points.
//
// File: golang/shapes_index.go
// Author: Ash Vardanian
// Date: November 16, 2026

// Every verb is a batch over a tape, the Apache Arrow layout of ragged entries, so molecules or
// scans of any point counts share one batch, and a single shape is a one-entry tape. Reports,
// traversal stats, cursor batches and draws are the dense modality's own types, since the library
// fills them through the same C structures.

package usearch

import "unsafe"

/*
// `nocallback` only where the argument structure carries no callback and no executor; `noescape`
// everywhere, since submission copies the structure and everything it names is pinned.
#cgo noescape usearch_shapes_init_ram
#cgo nocallback usearch_shapes_init_ram
#cgo noescape usearch_shapes_init_vram
#cgo nocallback usearch_shapes_init_vram
#cgo noescape usearch_shapes_view_ram
#cgo nocallback usearch_shapes_view_ram
#cgo noescape usearch_shapes_cursor_init
#cgo nocallback usearch_shapes_cursor_init
#cgo noescape usearch_shapes_view_nvme
#cgo noescape usearch_shapes_reserve
#cgo noescape usearch_shapes_clear
#cgo noescape usearch_shapes_describe
#cgo noescape usearch_shapes_count
#cgo noescape usearch_shapes_get
#cgo noescape usearch_shapes_health
#cgo noescape usearch_shapes_metadata_read
#cgo noescape usearch_shapes_metadata_read_buffer
#cgo noescape usearch_shapes_load_ram
#cgo noescape usearch_shapes_load_vram
#cgo noescape usearch_shapes_load_ram_buffer
#cgo noescape usearch_shapes_load_vram_buffer
#cgo noescape usearch_shapes_insert
#cgo noescape usearch_shapes_refine
#cgo noescape usearch_shapes_refine_plan_finish
#cgo noescape usearch_shapes_search
#cgo noescape usearch_shapes_reshard
#cgo noescape usearch_shapes_reshard_plan_finish
#cgo noescape usearch_shapes_sample
#cgo noescape usearch_shapes_save
#cgo noescape usearch_shapes_save_buffer
#cgo noescape usearch_shapes_cursor_next

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
	view := C.usearch_bytes_view_t{data: (*C.uint8_t)(unsafe.Pointer(unsafe.StringData(name))), length: C.size_t(len(name))}
	if status := C.usearch_shapes_metric_from_name(view, &metric); status != C.usearch_status_success_k {
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

// ShapesViewNvmeConfig carries the optional settings of ShapesViewNvme.
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

// ShapesRefineConfig carries the optional settings of Refine. A zero value derives every setting.
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
	Stats *DenseTraversalStats
	// Plan keeps the round live across calls: nil runs one round to its end, a fresh plan opens one
	// that a stop leaves live, and a live one continues where it stopped, or starts the next round.
	Plan *ShapesRefinePlan
}

// ShapesReshardConfig carries the optional settings of Reshard. A zero value derives every setting.
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
	Stats *DenseTraversalStats
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
	index.ShapesIndexOn = ShapesIndexOn{shapesReceiver{index: index}}
	return index, nil
}

// On is the receiver whose verbs block on executor. Binding-only.
func (index *ShapesIndex) On(executor *Executor) ShapesIndexOn {
	return ShapesIndexOn{shapesReceiver{index: index, executor: executor}}
}

// Deferred is the receiver whose verbs submit their calls to executor, each answering a Deferred
// from the same arguments. Binding-only.
func (index *ShapesIndex) Deferred(executor *Executor) ShapesIndexDeferredOn {
	return ShapesIndexDeferredOn{shapesReceiver{index: index, executor: executor}}
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
	lent := open[C.usearch_shapes_index_t]()
	C.usearch_shapes_init_ram(&C.usearch_shapes_init_ram_t{
		index:          lent.result,
		outcome:        lent.outcome,
		metric:         C.usearch_shapes_metric_t(options.Metric),
		quantization:   C.usearch_scalar_t(options.Quantization),
		points:         C.size_t(options.Points),
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		ram:            lent.ram(memoryDomains, options.CPUClassCeiling),
	})
	return shapesIndexOf(block(lent))
}

// ShapesInitVram creates an empty, mutable index in device memory, sharded across the CUDA
// ordinals devices names and never replicated; empty derives one free device.
func ShapesInitVram(options ShapesOptions, devices []int32) (*ShapesIndex, error) {
	lent := open[C.usearch_shapes_index_t]()
	C.usearch_shapes_init_vram(&C.usearch_shapes_init_vram_t{
		index:          lent.result,
		outcome:        lent.outcome,
		metric:         C.usearch_shapes_metric_t(options.Metric),
		quantization:   C.usearch_scalar_t(options.Quantization),
		points:         C.size_t(options.Points),
		connectivity:   C.size_t(options.Connectivity),
		values_per_key: C.size_t(options.ValuesPerKey),
		vram:           lent.vram(devices, options.GPUClassCeiling),
	})
	return shapesIndexOf(block(lent))
}

// ShapesLoadRam copies a checkpoint into a new, mutable index in host memory. The path names a
// manifest of a sharded checkpoint or a single .safegraphs monolith.
func ShapesLoadRam(path string, config *DenseLoadRamConfig) (*ShapesIndex, error) {
	if config == nil {
		config = new(DenseLoadRamConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_shapes_index_t]()
	C.usearch_shapes_load_ram(&C.usearch_shapes_load_ram_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		path:           lent.text(path),
		ram:            lent.ram(config.MemoryDomains, config.CPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return shapesIndexOf(block(lent))
}

// ShapesLoadVram copies a checkpoint into a new, mutable index in device memory, with the
// placement rules of ShapesLoadRam.
func ShapesLoadVram(path string, config *DenseLoadVramConfig) (*ShapesIndex, error) {
	if config == nil {
		config = new(DenseLoadVramConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_shapes_index_t]()
	C.usearch_shapes_load_vram(&C.usearch_shapes_load_vram_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		path:           lent.text(path),
		vram:           lent.vram(config.Devices, config.GPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return shapesIndexOf(block(lent))
}

// ShapesLoadRamBuffer copies a monolith from a buffer into a new, mutable index in host memory, as
// ShapesLoadRam does a path.
func ShapesLoadRamBuffer(buffer []byte, config *DenseLoadRamBufferConfig) (*ShapesIndex, error) {
	if config == nil {
		config = new(DenseLoadRamBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_shapes_index_t]()
	C.usearch_shapes_load_ram_buffer(&C.usearch_shapes_load_ram_buffer_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		buffer:         lent.bytes(buffer),
		ram:            lent.ram(config.MemoryDomains, config.CPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return shapesIndexOf(block(lent))
}

// ShapesLoadVramBuffer copies a monolith from a buffer into a new, mutable index in device memory.
func ShapesLoadVramBuffer(buffer []byte, config *DenseLoadVramBufferConfig) (*ShapesIndex, error) {
	if config == nil {
		config = new(DenseLoadVramBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_shapes_index_t]()
	C.usearch_shapes_load_vram_buffer(&C.usearch_shapes_load_vram_buffer_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		buffer:         lent.bytes(buffer),
		vram:           lent.vram(config.Devices, config.GPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return shapesIndexOf(block(lent))
}

// ShapesViewRam serves a checkpoint in place through a memory mapping, as a read-only index whose
// mutating verbs report StatusReadOnly; the files must stay unmodified while it lives.
func ShapesViewRam(path string) (*ShapesIndex, error) {
	lent := open[C.usearch_shapes_index_t]()
	C.usearch_shapes_view_ram(&C.usearch_shapes_view_ram_t{
		index:   lent.result,
		outcome: lent.outcome,
		path:    lent.text(path),
	})
	return shapesIndexOf(block(lent))
}

// ShapesViewNvme serves a checkpoint in place from NVMe drives, read through the executor's
// submission rings, as a read-only index.
func ShapesViewNvme(path string, config *ShapesViewNvmeConfig) (*ShapesIndex, error) {
	if config == nil {
		config = new(ShapesViewNvmeConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_shapes_index_t]()
	C.usearch_shapes_view_nvme(&C.usearch_shapes_view_nvme_t{
		index:    lent.result,
		executor: executor,
		outcome:  lent.outcome,
		path:     lent.text(path),
		nvme:     lent.nvme(config.Paths, config.CPUClassCeiling),
	})
	return shapesIndexOf(block(lent))
}

// ShapesMetadataRead reads a checkpoint's metric and geometry without loading it.
func ShapesMetadataRead(path string, config *DenseMetadataReadConfig) (ShapesMetadata, error) {
	if config == nil {
		config = new(DenseMetadataReadConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return ShapesMetadata{}, refusal
	}
	lent := open[ShapesMetadata]()
	C.usearch_shapes_metadata_read(&C.usearch_shapes_metadata_read_t{
		executor: executor,
		outcome:  lent.outcome,
		path:     lent.text(path),
		metadata: (*C.usearch_shapes_metadata_t)(unsafe.Pointer(lent.result)),
	})
	return block(lent)
}

// ShapesMetadataReadBuffer reads the metric and geometry out of a serialized monolith buffer.
func ShapesMetadataReadBuffer(buffer []byte, config *DenseMetadataReadBufferConfig) (ShapesMetadata, error) {
	if config == nil {
		config = new(DenseMetadataReadBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return ShapesMetadata{}, refusal
	}
	lent := open[ShapesMetadata]()
	C.usearch_shapes_metadata_read_buffer(&C.usearch_shapes_metadata_read_buffer_t{
		executor: executor,
		outcome:  lent.outcome,
		buffer:   lent.bytes(buffer),
		metadata: (*C.usearch_shapes_metadata_t)(unsafe.Pointer(lent.result)),
	})
	return block(lent)
}

// endregion: Index

// region: Receivers

// shapesReceiver is an index and the executor its verbs run on, nil for the calling goroutine.
type shapesReceiver struct {
	index    *ShapesIndex
	executor *Executor
}

// lower reports the handles a verb hands across, refusing a closed index before C dereferences it.
func (receiver shapesReceiver) lower() (C.usearch_shapes_index_t, C.usearch_executor_t, *Error) {
	if receiver.index == nil || receiver.index.handle == nil {
		return nil, nil, fail(StatusUninitialized, "the index is closed or was never initialized")
	}
	executor, refusal := receiver.executor.lower()
	return receiver.index.handle, executor, refusal
}

// ShapesIndexOn is the receiver whose verbs block until their call is final: on the calling
// goroutine when embedded in a ShapesIndex, and on an executor when ShapesIndex.On makes it.
// Binding-only.
type ShapesIndexOn struct {
	shapesReceiver
}

// ShapesIndexDeferredOn is the receiver whose verbs submit their calls to an executor and answer
// a Deferred, offering every verb of ShapesIndexOn with the same arguments but the blocking-only
// sugar and cursors. Binding-only.
//
// The library reads a deferred call's inputs until the call is final, so the slices a verb is
// handed must not change until then; its arena is borrowed, as a blocking verb borrows it.
type ShapesIndexDeferredOn struct {
	shapesReceiver
}

// Reserve claims capacity for a number of incoming entries and the points they hold, ahead of
// insertion. Both are desired totals, including what the index holds.
func (on ShapesIndexOn) Reserve(entries, points uint) error {
	_, err := block(on.reserve(C.usearch_completion_blocking_k, entries, points))
	return err
}

// Reserve submits ShapesIndexOn.Reserve.
func (on ShapesIndexDeferredOn) Reserve(entries, points uint) *Deferred[struct{}] {
	return submitted(on.reserve(C.usearch_completion_deferred_k, entries, points))
}

// Clear erases all entries, keeping the reserved capacity. No other verb may overlap it on the
// same handle.
func (on ShapesIndexOn) Clear() error {
	_, err := block(on.clear(C.usearch_completion_blocking_k))
	return err
}

// Clear submits ShapesIndexOn.Clear.
func (on ShapesIndexDeferredOn) Clear() *Deferred[struct{}] {
	return submitted(on.clear(C.usearch_completion_deferred_k))
}

// Describe reports the frozen metadata and the current counters in one call.
func (on ShapesIndexOn) Describe() (ShapesIndexDescription, error) {
	return block(on.describe(C.usearch_completion_blocking_k))
}

// Describe submits ShapesIndexOn.Describe.
func (on ShapesIndexDeferredOn) Describe() *Deferred[ShapesIndexDescription] {
	return submitted(on.describe(C.usearch_completion_deferred_k))
}

// Count reports how many entries are stored under a key, with the extent of their points.
func (on ShapesIndexOn) Count(key Key) (ShapesKeyStats, error) {
	return block(on.count(C.usearch_completion_blocking_k, key))
}

// Count submits ShapesIndexOn.Count.
func (on ShapesIndexDeferredOn) Count(key Key) *Deferred[ShapesKeyStats] {
	return submitted(on.count(C.usearch_completion_deferred_k, key))
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
	return block(on.get(C.usearch_completion_blocking_k, key, outputsScalar, arena))
}

// Get submits ShapesIndexOn.Get.
func (on ShapesIndexDeferredOn) Get(key Key, outputsScalar Scalar, arena *Arena) *Deferred[TapeView] {
	return submitted(on.get(C.usearch_completion_deferred_k, key, outputsScalar, arena))
}

// Insert adds a tape of shapes under their keys - the native batch operation, where a single
// shape is a one-entry tape. The points' scalar type is independent of the stored quantization,
// and a batch is all-or-nothing and becomes visible at once.
func (on ShapesIndexOn) Insert(keys []Key, shapes Tape, config *DenseInsertConfig) error {
	_, err := block(on.insert(C.usearch_completion_blocking_k, keys, shapes, config))
	return err
}

// Insert submits ShapesIndexOn.Insert.
func (on ShapesIndexDeferredOn) Insert(keys []Key, shapes Tape, config *DenseInsertConfig) *Deferred[struct{}] {
	return submitted(on.insert(C.usearch_completion_deferred_k, keys, shapes, config))
}

// Refine re-selects the neighbors of every stored entry against the finished graph, as the dense
// Refine does; a caller repeats rounds until the report's Verdict reads DenseRefineConverged.
func (on ShapesIndexOn) Refine(config *ShapesRefineConfig) (DenseRefineReport, error) {
	return block(on.refine(C.usearch_completion_blocking_k, config))
}

// Refine submits ShapesIndexOn.Refine.
func (on ShapesIndexDeferredOn) Refine(config *ShapesRefineConfig) *Deferred[DenseRefineReport] {
	return submitted(on.refine(C.usearch_completion_deferred_k, config))
}

// Health measures how much of the graph a search reaches from the root. Read-only.
func (on ShapesIndexOn) Health() (DenseReachability, error) {
	return block(on.health(C.usearch_completion_blocking_k))
}

// Health submits ShapesIndexOn.Health.
func (on ShapesIndexDeferredOn) Health() *Deferred[DenseReachability] {
	return submitted(on.health(C.usearch_completion_deferred_k))
}

// Search resolves a tape of queries into up to wanted nearest matches apiece, written into arena,
// and filtered when the config carries a predicate. Never writes to the graph.
func (on ShapesIndexOn) Search(queries Tape, wanted uint, arena *Arena, config *DenseSearchConfig) (MatchesView, error) {
	return block(on.search(C.usearch_completion_blocking_k, queries, wanted, arena, config))
}

// Search submits ShapesIndexOn.Search.
func (on ShapesIndexDeferredOn) Search(queries Tape, wanted uint, arena *Arena,
	config *DenseSearchConfig) *Deferred[MatchesView] {
	return submitted(on.search(C.usearch_completion_deferred_k, queries, wanted, arena, config))
}

// Reshard moves the entries of sources into targets, keeping their links rather than re-inserting
// them, then runs a round on every target that took entries: several sources gather into one
// target, and one source cuts into one graph-local part per target. The receiver contributes only
// its executor, and a refusal moves nothing.
func (on ShapesIndexOn) Reshard(sources, targets []*ShapesIndex, config *ShapesReshardConfig) (DenseRefineReport, error) {
	return block(on.reshard(C.usearch_completion_blocking_k, sources, targets, config))
}

// Reshard submits ShapesIndexOn.Reshard.
func (on ShapesIndexDeferredOn) Reshard(sources, targets []*ShapesIndex,
	config *ShapesReshardConfig) *Deferred[DenseRefineReport] {
	return submitted(on.reshard(C.usearch_completion_deferred_k, sources, targets, config))
}

// RefinePlanFinish closes a live refine plan and frees it whatever the call answers, reporting
// what the whole plan did to the graph.
func (on ShapesIndexOn) RefinePlanFinish(plan *ShapesRefinePlan) (DenseRefineReport, error) {
	return block(on.refinePlanFinish(C.usearch_completion_blocking_k, plan))
}

// RefinePlanFinish submits ShapesIndexOn.RefinePlanFinish.
func (on ShapesIndexDeferredOn) RefinePlanFinish(plan *ShapesRefinePlan) *Deferred[DenseRefineReport] {
	return submitted(on.refinePlanFinish(C.usearch_completion_deferred_k, plan))
}

// ReshardPlanFinish closes a live reshard plan on this receiver's executor and frees it whatever
// the call answers, reporting what the whole plan did.
func (on ShapesIndexOn) ReshardPlanFinish(plan *ShapesReshardPlan) (DenseRefineReport, error) {
	return block(on.reshardPlanFinish(C.usearch_completion_blocking_k, plan))
}

// ReshardPlanFinish submits ShapesIndexOn.ReshardPlanFinish.
func (on ShapesIndexDeferredOn) ReshardPlanFinish(plan *ShapesReshardPlan) *Deferred[DenseRefineReport] {
	return submitted(on.reshardPlanFinish(C.usearch_completion_deferred_k, plan))
}

// Sample draws up to wanted keys around the config's Focus, or across the whole index, each
// hanging from a shown or earlier draw and placed in three dimensions beside the shown ones.
func (on ShapesIndexOn) Sample(wanted uint, arena *Arena, config *DenseSampleConfig) (DenseDraws, error) {
	return block(on.sample(C.usearch_completion_blocking_k, wanted, arena, config))
}

// Sample submits ShapesIndexOn.Sample.
func (on ShapesIndexDeferredOn) Sample(wanted uint, arena *Arena, config *DenseSampleConfig) *Deferred[DenseDraws] {
	return submitted(on.sample(C.usearch_completion_deferred_k, wanted, arena, config))
}

// Save writes the index under a path, in the layout its extension picks - one .safegraphs
// monolith, or a .safegraphs.json manifest beside its shards.
func (on ShapesIndexOn) Save(path string, config *DenseSaveConfig) error {
	_, err := block(on.save(C.usearch_completion_blocking_k, path, config))
	return err
}

// Save submits ShapesIndexOn.Save.
func (on ShapesIndexDeferredOn) Save(path string, config *DenseSaveConfig) *Deferred[struct{}] {
	return submitted(on.save(C.usearch_completion_deferred_k, path, config))
}

// SaveBuffer serializes the index as one monolith into arena.
func (on ShapesIndexOn) SaveBuffer(arena *Arena, config *DenseSaveBufferConfig) (Monolith, error) {
	return block(on.saveBuffer(C.usearch_completion_blocking_k, arena, config))
}

// SaveBuffer submits ShapesIndexOn.SaveBuffer.
func (on ShapesIndexDeferredOn) SaveBuffer(arena *Arena, config *DenseSaveBufferConfig) *Deferred[Monolith] {
	return submitted(on.saveBuffer(C.usearch_completion_deferred_k, arena, config))
}

// endregion: Receivers

// region: Calls

func (receiver shapesReceiver) reserve(completion C.usearch_completion_t, entries, points uint) *lent[struct{}] {
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_shapes_reserve(&C.usearch_shapes_reserve_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		entries:    C.size_t(entries),
		points:     C.size_t(points),
	})
	return lent
}

func (receiver shapesReceiver) clear(completion C.usearch_completion_t) *lent[struct{}] {
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_shapes_clear(&C.usearch_shapes_clear_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
	})
	return lent
}

func (receiver shapesReceiver) describe(completion C.usearch_completion_t) *lent[ShapesIndexDescription] {
	lent := open[ShapesIndexDescription]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_shapes_describe(&C.usearch_shapes_describe_t{
		index:       index,
		executor:    executor,
		outcome:     lent.outcome,
		completion:  completion,
		description: (*C.usearch_shapes_index_description_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver shapesReceiver) count(completion C.usearch_completion_t, key Key) *lent[ShapesKeyStats] {
	lent := open[ShapesKeyStats]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_shapes_count(&C.usearch_shapes_count_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		key:        key.toRaw(),
		key_stats:  (*C.usearch_shapes_key_stats_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver shapesReceiver) get(completion C.usearch_completion_t, key Key, outputsScalar Scalar,
	arena *Arena) *lent[TapeView] {
	lent := open[TapeView]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	lent.result.width = 3
	C.usearch_shapes_get(&C.usearch_shapes_get_t{
		index:          index,
		executor:       executor,
		outcome:        lent.outcome,
		completion:     completion,
		key:            key.toRaw(),
		shapes:         &lent.result.raw,
		written:        (*C.size_t)(unsafe.Pointer(&lent.result.written)),
		arena:          arena.lower(),
		outputs_scalar: C.usearch_scalar_t(outputsScalar),
	})
	return lent
}

func (receiver shapesReceiver) insert(completion C.usearch_completion_t, keys []Key, shapes Tape,
	config *DenseInsertConfig) *lent[struct{}] {
	if config == nil {
		config = new(DenseInsertConfig)
	}
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	// The call has one count, so only this side sees the keys and the entries disagree.
	if uint(len(keys)) != shapes.Entries() {
		return lent.refuse(fail(StatusShapeMismatch, "the key count does not match the tape's entries"))
	}
	C.usearch_shapes_insert(&C.usearch_shapes_insert_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome,
		completion:       completion,
		count:            C.size_t(len(keys)),
		keys:             lent.keys(keys),
		shapes:           shapes.lower(&lent.call),
		expansion_factor: C.size_t(config.ExpansionFactor),
		progress:         lent.progress(config.Progress, config.ProgressSteps),
		stats:            lent.stats(config.Stats),
	})
	return lent
}

func (receiver shapesReceiver) refine(completion C.usearch_completion_t,
	config *ShapesRefineConfig) *lent[DenseRefineReport] {
	if config == nil {
		config = new(ShapesRefineConfig)
	}
	lent := open[DenseRefineReport]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	plan, phase := config.Plan.lower(&lent.call)
	C.usearch_shapes_refine(&C.usearch_shapes_refine_t{
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

func (receiver shapesReceiver) refinePlanFinish(completion C.usearch_completion_t,
	plan *ShapesRefinePlan) *lent[DenseRefineReport] {
	lent := open[DenseRefineReport]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_shapes_refine_plan_finish(&C.usearch_shapes_refine_plan_finish_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver shapesReceiver) health(completion C.usearch_completion_t) *lent[DenseReachability] {
	lent := open[DenseReachability]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_shapes_health(&C.usearch_shapes_health_t{
		index:        index,
		executor:     executor,
		outcome:      lent.outcome,
		completion:   completion,
		reachability: (*C.usearch_reachability_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver shapesReceiver) search(completion C.usearch_completion_t, queries Tape, wanted uint,
	arena *Arena, config *DenseSearchConfig) *lent[MatchesView] {
	if config == nil {
		config = new(DenseSearchConfig)
	}
	lent := open[MatchesView]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	lent.result.queries = queries.Entries()
	C.usearch_shapes_search(&C.usearch_shapes_search_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome,
		completion:       completion,
		count:            C.size_t(queries.Entries()),
		queries:          queries.lower(&lent.call),
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

func (receiver shapesReceiver) reshard(completion C.usearch_completion_t, sources, targets []*ShapesIndex,
	config *ShapesReshardConfig) *lent[DenseRefineReport] {
	if config == nil {
		config = new(ShapesReshardConfig)
	}
	lent := open[DenseRefineReport]()
	// A reshard names its handles in its lists, so the receiver lends only its executor.
	executor, refusal := receiver.executor.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	sourceHandles, targetHandles := shapesHandles(sources), shapesHandles(targets)
	plan, phase := config.Plan.lower(&lent.call)
	C.usearch_shapes_reshard(&C.usearch_shapes_reshard_t{
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

// shapesHandles lists the handles of indexes.
func shapesHandles(indexes []*ShapesIndex) []C.usearch_shapes_index_t {
	handles := make([]C.usearch_shapes_index_t, len(indexes))
	for position, index := range indexes {
		handles[position] = index.handle
	}
	return handles
}

func (receiver shapesReceiver) reshardPlanFinish(completion C.usearch_completion_t,
	plan *ShapesReshardPlan) *lent[DenseRefineReport] {
	lent := open[DenseRefineReport]()
	// A reshard names its handles in its lists, so the receiver lends only its executor.
	executor, refusal := receiver.executor.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_shapes_reshard_plan_finish(&C.usearch_shapes_reshard_plan_finish_t{
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver shapesReceiver) sample(completion C.usearch_completion_t, wanted uint, arena *Arena,
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
	C.usearch_shapes_sample(&C.usearch_shapes_sample_t{
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

func (receiver shapesReceiver) save(completion C.usearch_completion_t, path string,
	config *DenseSaveConfig) *lent[struct{}] {
	if config == nil {
		config = new(DenseSaveConfig)
	}
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_shapes_save(&C.usearch_shapes_save_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		path:       lent.text(path),
		progress:   lent.progress(config.Progress, config.ProgressSteps),
	})
	return lent
}

func (receiver shapesReceiver) saveBuffer(completion C.usearch_completion_t, arena *Arena,
	config *DenseSaveBufferConfig) *lent[Monolith] {
	if config == nil {
		config = new(DenseSaveBufferConfig)
	}
	lent := open[Monolith]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_shapes_save_buffer(&C.usearch_shapes_save_buffer_t{
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

// ShapesCursor is a walk over the index contents, pulled blocking on the executor of the
// receiver that opened it; an entry once passed is never revisited.
type ShapesCursor struct {
	handle   C.usearch_shapes_cursor_t
	receiver shapesReceiver
}

// Cursor opens a walk over the current contents. Without starts it yields every entry as stored;
// with them, the start keys' own entries first in order, then the entries those link to
// breadth-first, each once.
func (on ShapesIndexOn) Cursor(starts ...Key) (*ShapesCursor, error) {
	index, _, refusal := on.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_shapes_cursor_t]()
	C.usearch_shapes_cursor_init(&C.usearch_shapes_cursor_init_t{
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
	return &ShapesCursor{handle: handle, receiver: on.shapesReceiver}, nil
}

// NextKeys copies up to wanted next keys into arena; a batch of none means the walk is exhausted.
func (cursor *ShapesCursor) NextKeys(wanted uint, arena *Arena) (DenseBatch, error) {
	return cursor.next(wanted, false, arena)
}

// NextLinks copies up to wanted next keys into arena, each with the keys it links to; a batch of
// none means the walk is exhausted.
func (cursor *ShapesCursor) NextLinks(wanted uint, arena *Arena) (DenseBatch, error) {
	return cursor.next(wanted, true, arena)
}

// next pulls a batch, exporting each entry's links only when linked asks for them.
func (cursor *ShapesCursor) next(wanted uint, linked bool, arena *Arena) (DenseBatch, error) {
	index, executor, refusal := cursor.receiver.lower()
	if refusal != nil {
		return DenseBatch{}, refusal
	}
	lent := open[DenseBatch]()
	var links *C.usearch_keys_tape_view_t
	if linked {
		links = &lent.result.links
	}
	C.usearch_shapes_cursor_next(&C.usearch_shapes_cursor_next_t{
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
func (cursor *ShapesCursor) Close() error {
	if cursor == nil || cursor.handle == nil {
		return nil
	}
	C.usearch_shapes_cursor_free(cursor.handle)
	cursor.handle = nil
	return nil
}

// endregion: Cursors
