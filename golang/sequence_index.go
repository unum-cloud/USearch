// Similarity Search over variable-length byte sequences, under edit distances and alignments.
//
// File: golang/sequence_index.go
// Author: Ash Vardanian
// Date: November 16, 2026

// Every verb is a batch over a tape, the Apache Arrow layout of ragged entries, so a column read
// from Arrow or Parquet reaches the index with no copy, and a single sequence is a one-entry tape.
// Cursor batches and draws are the dense modality's own types, since the library fills them through
// the same C structures.

package usearch

import "unsafe"

/*
// noescape and nocallback follow the rule in types.go.
#cgo noescape usearch_sequence_metric_from_name
#cgo nocallback usearch_sequence_metric_from_name
#cgo noescape usearch_sequence_init_ram
#cgo nocallback usearch_sequence_init_ram
#cgo noescape usearch_sequence_init_vram
#cgo nocallback usearch_sequence_init_vram
#cgo noescape usearch_sequence_view_ram
#cgo nocallback usearch_sequence_view_ram
#cgo noescape usearch_sequence_cursor_init
#cgo nocallback usearch_sequence_cursor_init
#cgo noescape usearch_sequence_view_nvme
#cgo nocallback usearch_sequence_view_nvme
#cgo noescape usearch_sequence_reserve
#cgo nocallback usearch_sequence_reserve
#cgo noescape usearch_sequence_clear
#cgo nocallback usearch_sequence_clear
#cgo noescape usearch_sequence_describe
#cgo nocallback usearch_sequence_describe
#cgo noescape usearch_sequence_count
#cgo nocallback usearch_sequence_count
#cgo noescape usearch_sequence_get
#cgo nocallback usearch_sequence_get
#cgo noescape usearch_sequence_health
#cgo nocallback usearch_sequence_health
#cgo noescape usearch_sequence_metadata_read
#cgo nocallback usearch_sequence_metadata_read
#cgo noescape usearch_sequence_metadata_read_buffer
#cgo nocallback usearch_sequence_metadata_read_buffer
#cgo noescape usearch_sequence_load_ram
#cgo noescape usearch_sequence_load_vram
#cgo noescape usearch_sequence_load_ram_buffer
#cgo noescape usearch_sequence_load_vram_buffer
#cgo noescape usearch_sequence_insert
#cgo noescape usearch_sequence_refine
#cgo noescape usearch_sequence_refine_plan_finish
#cgo nocallback usearch_sequence_refine_plan_finish
#cgo noescape usearch_sequence_search
#cgo noescape usearch_sequence_reshard
#cgo noescape usearch_sequence_reshard_plan_finish
#cgo nocallback usearch_sequence_reshard_plan_finish
#cgo noescape usearch_sequence_sample
#cgo noescape usearch_sequence_save
#cgo noescape usearch_sequence_save_buffer
#cgo noescape usearch_sequence_cursor_next
#cgo nocallback usearch_sequence_cursor_next

#include "shim.h"
#include "usearch/sequence_index.h"
*/
import "C"

// region: Geometry

// SequenceMetric names how two sequences are scored: the uniform edit distances, a weighted
// global alignment, or the overlap of their windows.
type SequenceMetric int32

const (
	// SequenceMetricUnknown tags a metric the library could not name.
	SequenceMetricUnknown SequenceMetric = C.usearch_sequence_metric_unknown_k
	// SequenceMetricLevenshteinBytes counts single-byte insertions, deletions and substitutions.
	SequenceMetricLevenshteinBytes SequenceMetric = C.usearch_sequence_metric_levenshtein_bytes_k
	// SequenceMetricLevenshteinUTF8 counts the same edits over decoded code points.
	SequenceMetricLevenshteinUTF8 SequenceMetric = C.usearch_sequence_metric_levenshtein_utf8_k
	// SequenceMetricNeedlemanWunsch is a global alignment under a substitution matrix and a gap cost.
	SequenceMetricNeedlemanWunsch SequenceMetric = C.usearch_sequence_metric_needleman_wunsch_k
	// SequenceMetricOverlap is the share of the larger window set the other side misses.
	SequenceMetricOverlap SequenceMetric = C.usearch_sequence_metric_overlap_k
)

// String renders the metric as the library spells it.
func (metric SequenceMetric) String() string {
	return C.GoString(C.usearch_sequence_metric_name(C.usearch_sequence_metric_t(metric)))
}

// SequenceMetricFromName reads a metric back from the spelling String gives it, through
// usearch_sequence_metric_from_name; a name the library does not spell fails with
// StatusInvalidArgument.
func SequenceMetricFromName(name string) (SequenceMetric, error) {
	metric := C.usearch_sequence_metric_t(SequenceMetricUnknown)
	if status := C.usearch_sequence_metric_from_name(view(name), &metric); status != C.usearch_status_success_k {
		return SequenceMetricUnknown, fail(Status(status), "no sequence metric is spelled \""+name+"\"")
	}
	return SequenceMetric(metric), nil
}

// SequenceWeights are the substitution matrix and gap cost an alignment scores with; the
// Levenshtein kinds score with unit costs and take none. The library reads them during the call
// that names them alone.
type SequenceWeights struct {
	// SubstitutionMatrix holds 256 × 256 signed scores over byte alphabets, row-major.
	SubstitutionMatrix []int8
	// Gap is the cost of one unaligned symbol.
	Gap int8
}

// SequenceOptions holds the construction options of a sequence index, the fields SequenceInitRam
// and SequenceInitVram share.
type SequenceOptions struct {
	// Metric is how two sequences are scored.
	Metric SequenceMetric
	// Weights are what SequenceMetricNeedlemanWunsch needs, and every other kind refuses; nil for
	// none.
	Weights *SequenceWeights
	// WindowWidths are the window widths in bytes SequenceMetricOverlap needs, and every other kind
	// refuses.
	WindowWidths []uint
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

// DefaultSequenceOptions is a byte-level Levenshtein index, with every optional field derived.
func DefaultSequenceOptions() SequenceOptions {
	return SequenceOptions{Metric: SequenceMetricLevenshteinBytes}
}

// SequenceMetadata is the metric and connectivity a checkpoint froze.
type SequenceMetadata struct {
	// Metric is the scoring kind the checkpoint scores with.
	Metric SequenceMetric
	// Connectivity is the connections per node the graph was built with.
	Connectivity uint
}

// SequenceIndexDescription is a live index's frozen metadata and current counters, each read
// once, so under concurrent insertions they may come from slightly different moments.
type SequenceIndexDescription struct {
	// Metadata is the frozen metric and connectivity, as a checkpoint reports them.
	Metadata SequenceMetadata
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

// SequenceKeyStats is what Count reports about one key: its entries and their extent.
type SequenceKeyStats struct {
	// Entries counts the entries stored under the key; zero means the index holds none.
	Entries uint
	// TotalBytes sums their lengths, the bytes a Get of the key writes.
	TotalBytes uint
	// LargestBytes is the longest of them.
	LargestBytes uint
}

// endregion: Geometry

// region: Configs

// SequenceLoadRamConfig carries the optional settings of SequenceLoadRam.
type SequenceLoadRamConfig struct {
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

// SequenceLoadVramConfig carries the optional settings of SequenceLoadVram.
type SequenceLoadVramConfig struct {
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

// SequenceLoadRamBufferConfig carries the optional settings of SequenceLoadRamBuffer.
type SequenceLoadRamBufferConfig struct {
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

// SequenceLoadVramBufferConfig carries the optional settings of SequenceLoadVramBuffer.
type SequenceLoadVramBufferConfig struct {
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

// SequenceMetadataReadConfig carries the optional settings of SequenceMetadataRead.
type SequenceMetadataReadConfig struct {
	// Executor is the one the read runs on; nil reads on the calling goroutine.
	Executor *Executor
}

// SequenceMetadataReadBufferConfig carries the optional settings of SequenceMetadataReadBuffer.
type SequenceMetadataReadBufferConfig struct {
	// Executor is the one the read runs on; nil reads on the calling goroutine.
	Executor *Executor
}

// SequenceInsertConfig carries the optional settings of Insert. A zero value derives every setting.
type SequenceInsertConfig struct {
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

// SequenceSampleConfig carries the optional settings of Sample. A zero value draws across the whole
// index with nothing shown.
type SequenceSampleConfig struct {
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

// SequenceSaveConfig carries the optional settings of Save.
type SequenceSaveConfig struct {
	// Progress reports the shards written and stops wherever it says to.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// SequenceSaveBufferConfig carries the optional settings of SaveBuffer.
type SequenceSaveBufferConfig struct {
	// Progress reports the bytes written and stops wherever it says to; a stopped save leaves no
	// monolith.
	Progress ProgressCallback
	// ProgressSteps bounds the reports Progress hears, the last at completion; zero derives to ten.
	ProgressSteps uint
}

// SequenceViewNvmeConfig holds the optional settings of SequenceViewNvme.
type SequenceViewNvmeConfig struct {
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

// SequenceSearchConfig carries the optional settings of Search. A zero value derives every setting.
type SequenceSearchConfig struct {
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

// SequenceRefineConfig holds the optional settings of Refine. A zero value derives every setting.
type SequenceRefineConfig struct {
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
	Plan *SequenceRefinePlan
}

// SequenceReshardConfig holds the optional settings of Reshard. A zero value derives every
// setting.
type SequenceReshardConfig struct {
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
	Plan *SequenceReshardPlan
}

// SequenceRefinePlan is one refine kept live across calls: the round in hand, and the root every
// round's health checks share. The zero value opens on the first Refine naming it, and
// RefinePlanFinish closes it; while it lives the index admits no insertion and no save, and each
// call names the index and executor it was opened with.
type SequenceRefinePlan = livePlan[C.usearch_sequence_refine_plan_t]

// SequenceReshardPlan is one reshard kept live across calls: its own sources and targets, the
// entries still to move, then the rounds of every target that took entries. The zero value opens
// on the first Reshard naming it, and ReshardPlanFinish closes it; while it lives every handle it
// names admits no insertion and no save, and each call names the executor it was opened with.
type SequenceReshardPlan = livePlan[C.usearch_sequence_reshard_plan_t]

// check refuses a matrix of another size than the 256 × 256 the library reads, since only this
// side sees its length.
func (weights *SequenceWeights) check() error {
	if weights == nil || weights.SubstitutionMatrix == nil || len(weights.SubstitutionMatrix) == 256*256 {
		return nil
	}
	return fail(StatusShapeMismatch, "a substitution matrix holds 256 × 256 scores")
}

// lower passes optional weights for the call alone, nil for none.
func (weights *SequenceWeights) lower(call *call) *C.usearch_sequence_weights_t {
	if weights == nil {
		return nil
	}
	lowered := &C.usearch_sequence_weights_t{
		substitution_matrix: (*C.int8_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(weights.SubstitutionMatrix)), len(weights.SubstitutionMatrix))),
		gap:                 C.int8_t(weights.Gap),
	}
	return (*C.usearch_sequence_weights_t)(call.pin(unsafe.Pointer(lowered)))
}

// endregion: Configs

// region: Index

// SequenceIndex is a Similarity Search Index over variable-length byte sequences: strings, DNA and
// protein reads, or any bytes a metric can score.
//
// A handle may be shared across goroutines. On a RAM handle, insertions and searches overlap
// freely, and serialization demands a mutation-free window while readers keep searching.
//
// Its verbs come from the embedded SequenceIndexOn and run on the calling goroutine; On runs the
// same verbs blocking on an executor, and Deferred submits them there.
type SequenceIndex struct {
	SequenceIndexOn
	handle C.usearch_sequence_index_t
}

// sequenceIndexOf wraps the handle a lifecycle verb wrote.
func sequenceIndexOf(handle C.usearch_sequence_index_t, err error) (*SequenceIndex, error) {
	if err != nil {
		return nil, err
	}
	index := &SequenceIndex{handle: handle}
	index.SequenceIndexOn = SequenceIndexOn{sequenceReceiver{receiver[C.usearch_sequence_index_t]{handle: &index.handle}}}
	return index, nil
}

// Close releases the index, waiting for every deferred call naming it. It is idempotent, and a
// closed handle must not be used again.
func (index *SequenceIndex) Close() error {
	if index == nil || index.handle == nil {
		return nil
	}
	C.usearch_sequence_free(index.handle)
	index.handle = nil
	return nil
}

// SequenceInitRam creates an empty, mutable index in host memory, one shard per NUMA node
// memoryDomains names; empty derives every node the process may use.
func SequenceInitRam(options SequenceOptions, memoryDomains []int32) (*SequenceIndex, error) {
	if err := options.Weights.check(); err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_sequence_index_t]
	arguments := open[C.usearch_sequence_init_ram_t](&call)
	*arguments = C.usearch_sequence_init_ram_t{
		index:               call.result,
		outcome:             call.outcome(),
		metric:              C.usearch_sequence_metric_t(options.Metric),
		weights:             options.Weights.lower(&call.call),
		window_widths:       (*C.size_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(options.WindowWidths)), len(options.WindowWidths))),
		window_widths_count: C.size_t(len(options.WindowWidths)),
		connectivity:        C.size_t(options.Connectivity),
		values_per_key:      C.size_t(options.ValuesPerKey),
		ram:                 call.ram(memoryDomains, options.CPUClassCeiling),
	}
	C.usearch_sequence_init_ram(arguments)
	return sequenceIndexOf(block(&call, nil))
}

// SequenceInitVram creates an empty, mutable index in device memory, sharded across the CUDA
// ordinals devices names and never replicated; empty derives one free device.
func SequenceInitVram(options SequenceOptions, devices []int32) (*SequenceIndex, error) {
	if err := options.Weights.check(); err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_sequence_index_t]
	arguments := open[C.usearch_sequence_init_vram_t](&call)
	*arguments = C.usearch_sequence_init_vram_t{
		index:               call.result,
		outcome:             call.outcome(),
		metric:              C.usearch_sequence_metric_t(options.Metric),
		weights:             options.Weights.lower(&call.call),
		window_widths:       (*C.size_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(options.WindowWidths)), len(options.WindowWidths))),
		window_widths_count: C.size_t(len(options.WindowWidths)),
		connectivity:        C.size_t(options.Connectivity),
		values_per_key:      C.size_t(options.ValuesPerKey),
		vram:                call.vram(devices, options.GPUClassCeiling),
	}
	C.usearch_sequence_init_vram(arguments)
	return sequenceIndexOf(block(&call, nil))
}

// SequenceLoadRam copies a checkpoint into a new, mutable index in host memory. The path names a
// manifest of a sharded checkpoint or a single .safegraphs monolith, and weights are the ones a
// Needleman-Wunsch checkpoint scores with, which no checkpoint records.
func SequenceLoadRam(path string, weights *SequenceWeights, config *SequenceLoadRamConfig) (*SequenceIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	if err := weights.check(); err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_sequence_index_t]
	arguments := open[C.usearch_sequence_load_ram_t](&call)
	*arguments = C.usearch_sequence_load_ram_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome(),
		path:           call.text(path),
		ram:            call.ram(options.MemoryDomains, options.CPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		weights:        weights.lower(&call.call),
		progress:       call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_sequence_load_ram(arguments)
	return sequenceIndexOf(block(&call, nil))
}

// SequenceLoadVram copies a checkpoint into a new, mutable index in device memory, with the
// placement rules of SequenceLoadRam.
func SequenceLoadVram(path string, weights *SequenceWeights, config *SequenceLoadVramConfig) (*SequenceIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	if err := weights.check(); err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_sequence_index_t]
	arguments := open[C.usearch_sequence_load_vram_t](&call)
	*arguments = C.usearch_sequence_load_vram_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome(),
		path:           call.text(path),
		vram:           call.vram(options.Devices, options.GPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		weights:        weights.lower(&call.call),
		progress:       call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_sequence_load_vram(arguments)
	return sequenceIndexOf(block(&call, nil))
}

// SequenceLoadRamBuffer copies a monolith from a buffer into a new, mutable index in host memory,
// as SequenceLoadRam does a path.
func SequenceLoadRamBuffer(buffer []byte, weights *SequenceWeights,
	config *SequenceLoadRamBufferConfig) (*SequenceIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	if err := weights.check(); err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_sequence_index_t]
	arguments := open[C.usearch_sequence_load_ram_buffer_t](&call)
	*arguments = C.usearch_sequence_load_ram_buffer_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome(),
		buffer:         call.bytes(buffer),
		ram:            call.ram(options.MemoryDomains, options.CPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		weights:        weights.lower(&call.call),
		progress:       call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_sequence_load_ram_buffer(arguments)
	return sequenceIndexOf(block(&call, nil))
}

// SequenceLoadVramBuffer copies a monolith from a buffer into a new, mutable index in device
// memory.
func SequenceLoadVramBuffer(buffer []byte, weights *SequenceWeights,
	config *SequenceLoadVramBufferConfig) (*SequenceIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	if err := weights.check(); err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_sequence_index_t]
	arguments := open[C.usearch_sequence_load_vram_buffer_t](&call)
	*arguments = C.usearch_sequence_load_vram_buffer_t{
		index:          call.result,
		executor:       executor,
		outcome:        call.outcome(),
		buffer:         call.bytes(buffer),
		vram:           call.vram(options.Devices, options.GPUClassCeiling),
		values_per_key: C.size_t(options.ValuesPerKey),
		weights:        weights.lower(&call.call),
		progress:       call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_sequence_load_vram_buffer(arguments)
	return sequenceIndexOf(block(&call, nil))
}

// SequenceViewRam serves a checkpoint in place through a memory mapping, as a read-only index
// whose mutating verbs report StatusReadOnly; the files must stay unmodified while it lives.
func SequenceViewRam(path string, weights *SequenceWeights) (*SequenceIndex, error) {
	if err := weights.check(); err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_sequence_index_t]
	arguments := open[C.usearch_sequence_view_ram_t](&call)
	*arguments = C.usearch_sequence_view_ram_t{
		index:   call.result,
		outcome: call.outcome(),
		path:    call.text(path),
		weights: weights.lower(&call.call),
	}
	C.usearch_sequence_view_ram(arguments)
	return sequenceIndexOf(block(&call, nil))
}

// SequenceViewNvme serves a checkpoint in place from NVMe drives, read through the executor's
// submission rings, as a read-only index.
func SequenceViewNvme(path string, weights *SequenceWeights, config *SequenceViewNvmeConfig) (*SequenceIndex, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return nil, err
	}
	if err := weights.check(); err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_sequence_index_t]
	arguments := open[C.usearch_sequence_view_nvme_t](&call)
	*arguments = C.usearch_sequence_view_nvme_t{
		index:    call.result,
		executor: executor,
		outcome:  call.outcome(),
		path:     call.text(path),
		nvme:     call.nvme(options.Paths, options.CPUClassCeiling),
		weights:  weights.lower(&call.call),
	}
	C.usearch_sequence_view_nvme(arguments)
	return sequenceIndexOf(block(&call, nil))
}

// SequenceMetadataRead reads a checkpoint's metric and connectivity without loading it.
func SequenceMetadataRead(path string, config *SequenceMetadataReadConfig) (SequenceMetadata, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return SequenceMetadata{}, err
	}
	var call callWithResult[SequenceMetadata]
	arguments := open[C.usearch_sequence_metadata_read_t](&call)
	*arguments = C.usearch_sequence_metadata_read_t{
		executor: executor,
		outcome:  call.outcome(),
		path:     call.text(path),
		metadata: (*C.usearch_sequence_metadata_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_sequence_metadata_read(arguments)
	return block(&call, nil)
}

// SequenceMetadataReadBuffer reads the metric and connectivity out of a serialized monolith buffer.
func SequenceMetadataReadBuffer(buffer []byte, config *SequenceMetadataReadBufferConfig) (SequenceMetadata, error) {
	options := configOf(config)
	executor, err := options.Executor.lower()
	if err != nil {
		return SequenceMetadata{}, err
	}
	var call callWithResult[SequenceMetadata]
	arguments := open[C.usearch_sequence_metadata_read_buffer_t](&call)
	*arguments = C.usearch_sequence_metadata_read_buffer_t{
		executor: executor,
		outcome:  call.outcome(),
		buffer:   call.bytes(buffer),
		metadata: (*C.usearch_sequence_metadata_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_sequence_metadata_read_buffer(arguments)
	return block(&call, nil)
}

// endregion: Index

// region: Receivers

// sequenceReceiver is an index and the executor its verbs run on, nil for the calling goroutine.
type sequenceReceiver struct {
	receiver[C.usearch_sequence_index_t]
}

// SequenceIndexOn is the receiver whose verbs block until their call is final: on the calling
// goroutine when embedded in a SequenceIndex, and on an executor when SequenceIndexOn.On makes it.
// Binding-only.
type SequenceIndexOn struct {
	sequenceReceiver
}

// SequenceIndexDeferredOn is the receiver whose verbs submit their calls to an executor and answer
// a Deferred, offering every verb of SequenceIndexOn with the same arguments but the blocking-only
// sugar. Binding-only.
//
// The library reads a deferred call's inputs until the call is final, so the slices a verb is
// handed must not change until then; its arena is viewed, as a blocking verb views it.
type SequenceIndexDeferredOn struct {
	sequenceReceiver
}

// On is the receiver whose verbs block on executor. Binding-only.
func (on SequenceIndexOn) On(executor *Executor) SequenceIndexOn {
	on.executor = executor
	return on
}

// Deferred is the receiver whose verbs submit their calls to executor, each answering a Deferred
// from the same arguments, or the refusal it caught before submitting anything. Binding-only.
func (on SequenceIndexOn) Deferred(executor *Executor) SequenceIndexDeferredOn {
	on.executor = executor
	return SequenceIndexDeferredOn{on.sequenceReceiver}
}

// Reserve claims capacity for a number of incoming entries and the bytes they hold, ahead of
// insertion. Both are desired totals, including what the index holds.
func (on SequenceIndexOn) Reserve(entries, payloadBytes uint) error {
	var call callWithResult[struct{}]
	_, err := block(&call, on.reserve(&call, C.usearch_completion_blocking_k, entries, payloadBytes))
	return err
}

// Reserve submits SequenceIndexOn.Reserve.
func (on SequenceIndexDeferredOn) Reserve(entries, payloadBytes uint) (*Deferred[struct{}], error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.reserve(&deferred.call, C.usearch_completion_deferred_k, entries, payloadBytes))
}

// Clear erases all entries, keeping the reserved capacity. No other verb may overlap it on the
// same handle.
func (on SequenceIndexOn) Clear() error {
	var call callWithResult[struct{}]
	_, err := block(&call, on.clear(&call, C.usearch_completion_blocking_k))
	return err
}

// Clear submits SequenceIndexOn.Clear.
func (on SequenceIndexDeferredOn) Clear() (*Deferred[struct{}], error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.clear(&deferred.call, C.usearch_completion_deferred_k))
}

// Describe reports the frozen metadata and the current counters in one call.
func (on SequenceIndexOn) Describe() (SequenceIndexDescription, error) {
	var call callWithResult[SequenceIndexDescription]
	return block(&call, on.describe(&call, C.usearch_completion_blocking_k))
}

// Describe submits SequenceIndexOn.Describe.
func (on SequenceIndexDeferredOn) Describe() (*Deferred[SequenceIndexDescription], error) {
	deferred := new(Deferred[SequenceIndexDescription])
	return submitted(deferred, on.describe(&deferred.call, C.usearch_completion_deferred_k))
}

// Count reports how many entries are stored under a key, with the extent of their bytes.
func (on SequenceIndexOn) Count(key Key) (SequenceKeyStats, error) {
	var call callWithResult[SequenceKeyStats]
	return block(&call, on.count(&call, C.usearch_completion_blocking_k, key))
}

// Count submits SequenceIndexOn.Count.
func (on SequenceIndexDeferredOn) Count(key Key) (*Deferred[SequenceKeyStats], error) {
	deferred := new(Deferred[SequenceKeyStats])
	return submitted(deferred, on.count(&deferred.call, C.usearch_completion_deferred_k, key))
}

// Contains reports whether the index holds at least one entry under a key, as a nonzero Count.
func (on SequenceIndexOn) Contains(key Key) (bool, error) {
	counted, err := on.Count(key)
	return counted.Entries != 0, err
}

// Get exports every sequence stored under a key into arena, one tape entry per entry sharing the
// key, in insertion order, read back with TapeEntry[byte].
//
// The call rewinds arena, and the tape stays readable until it serves another call.
func (on SequenceIndexOn) Get(key Key, arena *Arena) (TapeView, error) {
	var call callWithResult[TapeView]
	return block(&call, on.get(&call, C.usearch_completion_blocking_k, key, arena))
}

// Get submits SequenceIndexOn.Get.
func (on SequenceIndexDeferredOn) Get(key Key, arena *Arena) (*Deferred[TapeView], error) {
	deferred := new(Deferred[TapeView])
	return submitted(deferred, on.get(&deferred.call, C.usearch_completion_deferred_k, key, arena))
}

// Insert adds a tape of sequences under their keys - the native batch operation, where a single
// sequence is a one-entry tape. A batch is all-or-nothing and becomes visible at once.
func (on SequenceIndexOn) Insert(keys []Key, sequences Tape, config *SequenceInsertConfig) error {
	var call callWithResult[struct{}]
	_, err := block(&call, on.insert(&call, C.usearch_completion_blocking_k, keys, sequences, config))
	return err
}

// Insert submits SequenceIndexOn.Insert.
func (on SequenceIndexDeferredOn) Insert(keys []Key, sequences Tape, config *SequenceInsertConfig) (*Deferred[struct{}],
	error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.insert(&deferred.call, C.usearch_completion_deferred_k, keys, sequences, config))
}

// Refine re-selects the neighbors of every stored entry against the finished graph, as the dense
// Refine does; a caller repeats rounds until the report's Verdict reads RefineConverged.
func (on SequenceIndexOn) Refine(config *SequenceRefineConfig) (RefineReport, error) {
	var call callWithResult[RefineReport]
	return block(&call, on.refine(&call, C.usearch_completion_blocking_k, config))
}

// Refine submits SequenceIndexOn.Refine.
func (on SequenceIndexDeferredOn) Refine(config *SequenceRefineConfig) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.refine(&deferred.call, C.usearch_completion_deferred_k, config))
}

// Health measures how much of the graph a search reaches from the root. Read-only.
func (on SequenceIndexOn) Health() (Reachability, error) {
	var call callWithResult[Reachability]
	return block(&call, on.health(&call, C.usearch_completion_blocking_k))
}

// Health submits SequenceIndexOn.Health.
func (on SequenceIndexDeferredOn) Health() (*Deferred[Reachability], error) {
	deferred := new(Deferred[Reachability])
	return submitted(deferred, on.health(&deferred.call, C.usearch_completion_deferred_k))
}

// Search resolves a tape of queries into up to wanted nearest matches apiece, written into arena,
// and filtered when the config holds a predicate. Never writes to the graph.
func (on SequenceIndexOn) Search(queries Tape, wanted uint, arena *Arena, config *SequenceSearchConfig) (MatchesView,
	error) {
	var call callWithResult[MatchesView]
	return block(&call, on.search(&call, C.usearch_completion_blocking_k, queries, wanted, arena, config))
}

// Search submits SequenceIndexOn.Search.
func (on SequenceIndexDeferredOn) Search(queries Tape, wanted uint, arena *Arena,
	config *SequenceSearchConfig) (*Deferred[MatchesView], error) {
	deferred := new(Deferred[MatchesView])
	return submitted(deferred, on.search(&deferred.call, C.usearch_completion_deferred_k, queries, wanted, arena,
		config))
}

// Reshard moves the entries of sources into targets, keeping their links rather than re-inserting
// them, then runs a round on every target that took entries: several sources gather into one
// target, and one source cuts into one graph-local part per target. The receiver contributes only
// its executor, and a refusal moves nothing.
func (on SequenceIndexOn) Reshard(sources, targets []*SequenceIndex, config *SequenceReshardConfig) (RefineReport,
	error) {
	var call callWithResult[RefineReport]
	return block(&call, on.reshard(&call, C.usearch_completion_blocking_k, sources, targets, config))
}

// Reshard submits SequenceIndexOn.Reshard.
func (on SequenceIndexDeferredOn) Reshard(sources, targets []*SequenceIndex,
	config *SequenceReshardConfig) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.reshard(&deferred.call, C.usearch_completion_deferred_k, sources, targets, config))
}

// RefinePlanFinish closes a live refine plan and frees it whatever the call answers, reporting
// what the whole plan did to the graph.
func (on SequenceIndexOn) RefinePlanFinish(plan *SequenceRefinePlan) (RefineReport, error) {
	var call callWithResult[RefineReport]
	return block(&call, on.refinePlanFinish(&call, C.usearch_completion_blocking_k, plan))
}

// RefinePlanFinish submits SequenceIndexOn.RefinePlanFinish.
func (on SequenceIndexDeferredOn) RefinePlanFinish(plan *SequenceRefinePlan) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.refinePlanFinish(&deferred.call, C.usearch_completion_deferred_k, plan))
}

// ReshardPlanFinish closes a live reshard plan on this receiver's executor and frees it whatever
// the call answers, reporting what the whole plan did.
func (on SequenceIndexOn) ReshardPlanFinish(plan *SequenceReshardPlan) (RefineReport, error) {
	var call callWithResult[RefineReport]
	return block(&call, on.reshardPlanFinish(&call, C.usearch_completion_blocking_k, plan))
}

// ReshardPlanFinish submits SequenceIndexOn.ReshardPlanFinish.
func (on SequenceIndexDeferredOn) ReshardPlanFinish(plan *SequenceReshardPlan) (*Deferred[RefineReport], error) {
	deferred := new(Deferred[RefineReport])
	return submitted(deferred, on.reshardPlanFinish(&deferred.call, C.usearch_completion_deferred_k, plan))
}

// Sample draws up to wanted keys around the config's Focus, or across the whole index, each
// hanging from a shown or earlier draw and placed in three dimensions beside the shown ones.
func (on SequenceIndexOn) Sample(wanted uint, arena *Arena, config *SequenceSampleConfig) (DenseDraws, error) {
	var call callWithResult[DenseDraws]
	return block(&call, on.sample(&call, C.usearch_completion_blocking_k, wanted, arena, config))
}

// Sample submits SequenceIndexOn.Sample.
func (on SequenceIndexDeferredOn) Sample(wanted uint, arena *Arena, config *SequenceSampleConfig) (*Deferred[DenseDraws],
	error) {
	deferred := new(Deferred[DenseDraws])
	return submitted(deferred, on.sample(&deferred.call, C.usearch_completion_deferred_k, wanted, arena, config))
}

// Save writes the index under a path, in the layout its extension picks - one .safegraphs
// monolith, or a .safegraphs.json manifest beside its shards.
func (on SequenceIndexOn) Save(path string, config *SequenceSaveConfig) error {
	var call callWithResult[struct{}]
	_, err := block(&call, on.save(&call, C.usearch_completion_blocking_k, path, config))
	return err
}

// Save submits SequenceIndexOn.Save.
func (on SequenceIndexDeferredOn) Save(path string, config *SequenceSaveConfig) (*Deferred[struct{}], error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.save(&deferred.call, C.usearch_completion_deferred_k, path, config))
}

// SaveBuffer serializes the index as one monolith into arena.
func (on SequenceIndexOn) SaveBuffer(arena *Arena, config *SequenceSaveBufferConfig) (Monolith, error) {
	var call callWithResult[Monolith]
	return block(&call, on.saveBuffer(&call, C.usearch_completion_blocking_k, arena, config))
}

// SaveBuffer submits SequenceIndexOn.SaveBuffer.
func (on SequenceIndexDeferredOn) SaveBuffer(arena *Arena, config *SequenceSaveBufferConfig) (*Deferred[Monolith], error) {
	deferred := new(Deferred[Monolith])
	return submitted(deferred, on.saveBuffer(&deferred.call, C.usearch_completion_deferred_k, arena, config))
}

// endregion: Receivers

// region: Calls

func (receiver sequenceReceiver) reserve(call *callWithResult[struct{}], completion C.usearch_completion_t, entries,
	payloadBytes uint) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_reserve_t](call)
	*arguments = C.usearch_sequence_reserve_t{
		index:         index,
		executor:      executor,
		outcome:       call.outcome(),
		completion:    completion,
		entries:       C.size_t(entries),
		payload_bytes: C.size_t(payloadBytes),
	}
	C.usearch_sequence_reserve(arguments)
	return nil
}

func (receiver sequenceReceiver) clear(call *callWithResult[struct{}], completion C.usearch_completion_t) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_clear_t](call)
	*arguments = C.usearch_sequence_clear_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
	}
	C.usearch_sequence_clear(arguments)
	return nil
}

func (receiver sequenceReceiver) describe(call *callWithResult[SequenceIndexDescription],
	completion C.usearch_completion_t) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_describe_t](call)
	*arguments = C.usearch_sequence_describe_t{
		index:       index,
		executor:    executor,
		outcome:     call.outcome(),
		completion:  completion,
		description: (*C.usearch_sequence_index_description_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_sequence_describe(arguments)
	return nil
}

func (receiver sequenceReceiver) count(call *callWithResult[SequenceKeyStats], completion C.usearch_completion_t, key Key) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_count_t](call)
	*arguments = C.usearch_sequence_count_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		key:        key.toRaw(),
		key_stats:  (*C.usearch_sequence_key_stats_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_sequence_count(arguments)
	return nil
}

func (receiver sequenceReceiver) get(call *callWithResult[TapeView], completion C.usearch_completion_t, key Key,
	arena *Arena) error {
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_get_t](call)
	call.result.width = 1
	*arguments = C.usearch_sequence_get_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		key:        key.toRaw(),
		sequences:  &call.result.raw,
		written:    (*C.size_t)(unsafe.Pointer(&call.result.written)),
		arena:      outputs,
	}
	C.usearch_sequence_get(arguments)
	return nil
}

func (receiver sequenceReceiver) insert(call *callWithResult[struct{}], completion C.usearch_completion_t, keys []Key,
	sequences Tape, config *SequenceInsertConfig) error {
	options := configOf(config)
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	// The call has one count, so only this side sees the keys and the entries disagree.
	if uint(len(keys)) != sequences.Entries() {
		return fail(StatusShapeMismatch, "the key count does not match the tape's entries")
	}
	arguments := open[C.usearch_sequence_insert_t](call)
	*arguments = C.usearch_sequence_insert_t{
		index:            index,
		executor:         executor,
		outcome:          call.outcome(),
		completion:       completion,
		count:            C.size_t(len(keys)),
		keys:             call.keys(keys),
		sequences:        sequences.lower(&call.call),
		expansion_factor: C.size_t(options.ExpansionFactor),
		progress:         call.progress(options.Progress, options.ProgressSteps),
		stats:            call.stats(options.Stats),
	}
	C.usearch_sequence_insert(arguments)
	return nil
}

func (receiver sequenceReceiver) refine(call *callWithResult[RefineReport], completion C.usearch_completion_t,
	config *SequenceRefineConfig) error {
	options := configOf(config)
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_refine_t](call)
	plan, phase := options.Plan.lower(&call.call)
	*arguments = C.usearch_sequence_refine_t{
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
	C.usearch_sequence_refine(arguments)
	return nil
}

func (receiver sequenceReceiver) refinePlanFinish(call *callWithResult[RefineReport], completion C.usearch_completion_t,
	plan *SequenceRefinePlan) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_refine_plan_finish_t](call)
	*arguments = C.usearch_sequence_refine_plan_finish_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_sequence_refine_plan_finish(arguments)
	return nil
}

func (receiver sequenceReceiver) health(call *callWithResult[Reachability], completion C.usearch_completion_t) error {
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_health_t](call)
	*arguments = C.usearch_sequence_health_t{
		index:        index,
		executor:     executor,
		outcome:      call.outcome(),
		completion:   completion,
		reachability: (*C.usearch_reachability_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_sequence_health(arguments)
	return nil
}

func (receiver sequenceReceiver) search(call *callWithResult[MatchesView], completion C.usearch_completion_t, queries Tape,
	wanted uint, arena *Arena, config *SequenceSearchConfig) error {
	options := configOf(config)
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_search_t](call)
	call.result.queries = queries.Entries()
	*arguments = C.usearch_sequence_search_t{
		index:            index,
		executor:         executor,
		outcome:          call.outcome(),
		completion:       completion,
		count:            C.size_t(queries.Entries()),
		queries:          queries.lower(&call.call),
		matches:          &call.result.raw,
		arena:            outputs,
		wanted:           C.size_t(wanted),
		expansion_factor: C.size_t(options.ExpansionFactor),
		traversal:        C.usearch_traversal_t(options.Traversal),
		predicate:        call.predicate(options.Predicate),
		progress:         call.progress(options.Progress, options.ProgressSteps),
		stats:            call.stats(options.Stats),
	}
	C.usearch_sequence_search(arguments)
	return nil
}

func (receiver sequenceReceiver) reshard(call *callWithResult[RefineReport], completion C.usearch_completion_t, sources,
	targets []*SequenceIndex, config *SequenceReshardConfig) error {
	options := configOf(config)
	// A reshard names its handles in its lists, so the receiver passes only its executor.
	executor, err := receiver.executor.lower()
	if err != nil {
		return err
	}
	sourceHandles, targetHandles := sequenceHandles(sources), sequenceHandles(targets)
	arguments := open[C.usearch_sequence_reshard_t](call)
	plan, phase := options.Plan.lower(&call.call)
	*arguments = C.usearch_sequence_reshard_t{
		sources_count: C.size_t(len(sourceHandles)),
		sources:       (*C.usearch_sequence_index_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(sourceHandles)), len(sourceHandles))),
		targets_count: C.size_t(len(targetHandles)),
		targets:       (*C.usearch_sequence_index_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(targetHandles)), len(targetHandles))),
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
	C.usearch_sequence_reshard(arguments)
	return nil
}

// sequenceHandles lists the handles of indexes.
func sequenceHandles(indexes []*SequenceIndex) []C.usearch_sequence_index_t {
	handles := make([]C.usearch_sequence_index_t, len(indexes))
	for position, index := range indexes {
		handles[position] = index.handle
	}
	return handles
}

func (receiver sequenceReceiver) reshardPlanFinish(call *callWithResult[RefineReport], completion C.usearch_completion_t,
	plan *SequenceReshardPlan) error {
	// A reshard names its handles in its lists, so the receiver passes only its executor.
	executor, err := receiver.executor.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_reshard_plan_finish_t](call)
	*arguments = C.usearch_sequence_reshard_plan_finish_t{
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(call.result)),
	}
	C.usearch_sequence_reshard_plan_finish(arguments)
	return nil
}

func (receiver sequenceReceiver) sample(call *callWithResult[DenseDraws], completion C.usearch_completion_t, wanted uint,
	arena *Arena, config *SequenceSampleConfig) error {
	options := configOf(config)
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	// The call has one count for both, so only this side sees the keys and their places disagree.
	if len(options.Shown) != len(options.ShownCoordinates) {
		return fail(StatusShapeMismatch, "the shown keys and coordinates disagree in length")
	}
	arguments := open[C.usearch_sequence_sample_t](call)
	*arguments = C.usearch_sequence_sample_t{
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
	C.usearch_sequence_sample(arguments)
	return nil
}

func (receiver sequenceReceiver) save(call *callWithResult[struct{}], completion C.usearch_completion_t, path string,
	config *SequenceSaveConfig) error {
	options := configOf(config)
	index, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_save_t](call)
	*arguments = C.usearch_sequence_save_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		path:       call.text(path),
		progress:   call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_sequence_save(arguments)
	return nil
}

func (receiver sequenceReceiver) saveBuffer(call *callWithResult[Monolith], completion C.usearch_completion_t, arena *Arena,
	config *SequenceSaveBufferConfig) error {
	options := configOf(config)
	index, executor, outputs, err := receiver.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_save_buffer_t](call)
	*arguments = C.usearch_sequence_save_buffer_t{
		index:      index,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		monolith:   &call.result.raw,
		arena:      outputs,
		progress:   call.progress(options.Progress, options.ProgressSteps),
	}
	C.usearch_sequence_save_buffer(arguments)
	return nil
}

// endregion: Calls

// region: Cursors

// SequenceCursor is a walk over the index contents; an entry once passed is never revisited.
//
// Its pulls come from the embedded SequenceCursorOn and run on the calling goroutine; On runs them
// blocking on an executor, named again for every pull.
type SequenceCursor struct {
	SequenceCursorOn
	handle C.usearch_sequence_cursor_t
}

// SequenceCursorOn is the receiver whose pulls block until their call is final: on the calling
// goroutine when embedded in a SequenceCursor, and on an executor when SequenceCursorOn.On makes
// it. Binding-only.
type SequenceCursorOn struct {
	childReceiver[C.usearch_sequence_cursor_t, C.usearch_sequence_index_t]
}

// On is the receiver whose pulls block on executor. Binding-only.
func (on SequenceCursorOn) On(executor *Executor) SequenceCursorOn {
	on.executor = executor
	return on
}

// Cursor opens a walk over the current contents. Without starts it yields every entry as stored;
// with them, the start keys' own entries first in order, then the entries those link to
// breadth-first, each once.
func (index *SequenceIndex) Cursor(starts ...Key) (*SequenceCursor, error) {
	handle, _, err := index.lower()
	if err != nil {
		return nil, err
	}
	var call callWithResult[C.usearch_sequence_cursor_t]
	arguments := open[C.usearch_sequence_cursor_init_t](&call)
	*arguments = C.usearch_sequence_cursor_init_t{
		index:        handle,
		outcome:      call.outcome(),
		starts_count: C.size_t(len(starts)),
		starts:       (*C.usearch_key_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(starts)), len(starts))),
		cursor:       call.result,
	}
	C.usearch_sequence_cursor_init(arguments)
	cursor := new(SequenceCursor)
	if cursor.handle, err = block(&call, nil); err != nil {
		return nil, err
	}
	cursor.childReceiver = childReceiver[C.usearch_sequence_cursor_t, C.usearch_sequence_index_t]{
		receiver: receiver[C.usearch_sequence_cursor_t]{handle: &cursor.handle}, parent: &index.handle}
	return cursor, nil
}

// NextKeys copies up to wanted next keys into arena; a batch of none means the walk is exhausted.
func (on SequenceCursorOn) NextKeys(wanted uint, arena *Arena) (DenseBatch, error) {
	var call callWithResult[DenseBatch]
	return block(&call, on.next(&call, wanted, arena, noLinks))
}

// NextLinks copies up to wanted next keys into arena, each with the keys it links to; a batch of
// none means the walk is exhausted.
func (on SequenceCursorOn) NextLinks(wanted uint, arena *Arena) (DenseBatch, error) {
	var call callWithResult[DenseBatch]
	return block(&call, on.next(&call, wanted, arena, linksOf))
}

// next pulls a batch, exporting each entry's links into the slot links picks, if any.
func (on SequenceCursorOn) next(call *callWithResult[DenseBatch], wanted uint, arena *Arena, links batchLinks) error {
	cursor, index, executor, outputs, err := on.lowerInto(arena)
	if err != nil {
		return err
	}
	arguments := open[C.usearch_sequence_cursor_next_t](call)
	*arguments = C.usearch_sequence_cursor_next_t{
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
	C.usearch_sequence_cursor_next(arguments)
	return nil
}

// Close releases the cursor. It is idempotent.
func (cursor *SequenceCursor) Close() error {
	if cursor == nil || cursor.handle == nil {
		return nil
	}
	C.usearch_sequence_cursor_free(cursor.handle)
	cursor.handle = nil
	return nil
}

// endregion: Cursors
