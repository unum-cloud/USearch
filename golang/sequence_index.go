// Similarity Search over variable-length byte sequences, under edit distances and alignments.
//
// File: golang/sequence_index.go
// Author: Ash Vardanian
// Date: November 16, 2026

// Every verb is a batch over a tape, the Apache Arrow layout of ragged entries, so a column read
// from Arrow or Parquet reaches the index with no copy, and a single sequence is a one-entry tape.
// Reports, traversal stats, cursor batches and draws are the dense modality's own types, since the
// library fills them through the same C structures.

package usearch

import "unsafe"

/*
// `nocallback` only where the argument structure carries no callback and no executor; `noescape`
// everywhere, since submission copies the structure and everything it names is pinned.
#cgo noescape usearch_sequence_init_ram
#cgo nocallback usearch_sequence_init_ram
#cgo noescape usearch_sequence_init_vram
#cgo nocallback usearch_sequence_init_vram
#cgo noescape usearch_sequence_view_ram
#cgo nocallback usearch_sequence_view_ram
#cgo noescape usearch_sequence_cursor_init
#cgo nocallback usearch_sequence_cursor_init
#cgo noescape usearch_sequence_view_nvme
#cgo noescape usearch_sequence_reserve
#cgo noescape usearch_sequence_clear
#cgo noescape usearch_sequence_describe
#cgo noescape usearch_sequence_count
#cgo noescape usearch_sequence_get
#cgo noescape usearch_sequence_health
#cgo noescape usearch_sequence_metadata_read
#cgo noescape usearch_sequence_metadata_read_buffer
#cgo noescape usearch_sequence_load_ram
#cgo noescape usearch_sequence_load_vram
#cgo noescape usearch_sequence_load_ram_buffer
#cgo noescape usearch_sequence_load_vram_buffer
#cgo noescape usearch_sequence_insert
#cgo noescape usearch_sequence_refine
#cgo noescape usearch_sequence_refine_plan_finish
#cgo noescape usearch_sequence_search
#cgo noescape usearch_sequence_reshard
#cgo noescape usearch_sequence_reshard_plan_finish
#cgo noescape usearch_sequence_sample
#cgo noescape usearch_sequence_save
#cgo noescape usearch_sequence_save_buffer
#cgo noescape usearch_sequence_cursor_next

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
	view := C.usearch_bytes_view_t{data: (*C.uint8_t)(unsafe.Pointer(unsafe.StringData(name))), length: C.size_t(len(name))}
	if status := C.usearch_sequence_metric_from_name(view, &metric); status != C.usearch_status_success_k {
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

// SequenceViewNvmeConfig carries the optional settings of SequenceViewNvme.
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

// SequenceRefineConfig carries the optional settings of Refine. A zero value derives every setting.
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
	Stats *DenseTraversalStats
	// Plan keeps the round live across calls: nil runs one round to its end, a fresh plan opens one
	// that a stop leaves live, and a live one continues where it stopped, or starts the next round.
	Plan *SequenceRefinePlan
}

// SequenceReshardConfig carries the optional settings of Reshard. A zero value derives every
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
	Stats *DenseTraversalStats
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

// lower lends optional weights for the call alone, nil for none; a matrix of another size than
// the 256 × 256 the library reads is refused here, since only this side sees its length.
func (weights *SequenceWeights) lower(call *call) (*C.usearch_sequence_weights_t, *Error) {
	if weights == nil {
		return nil, nil
	}
	if weights.SubstitutionMatrix != nil && len(weights.SubstitutionMatrix) != 256*256 {
		return nil, fail(StatusShapeMismatch, "a substitution matrix holds 256 × 256 scores")
	}
	return pin(call, &C.usearch_sequence_weights_t{
		substitution_matrix: (*C.int8_t)(unsafe.Pointer(lend(call, weights.SubstitutionMatrix))),
		gap:                 C.int8_t(weights.Gap),
	}), nil
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
	index.SequenceIndexOn = SequenceIndexOn{sequenceReceiver{index: index}}
	return index, nil
}

// On is the receiver whose verbs block on executor. Binding-only.
func (index *SequenceIndex) On(executor *Executor) SequenceIndexOn {
	return SequenceIndexOn{sequenceReceiver{index: index, executor: executor}}
}

// Deferred is the receiver whose verbs submit their calls to executor, each answering a Deferred
// from the same arguments. Binding-only.
func (index *SequenceIndex) Deferred(executor *Executor) SequenceIndexDeferredOn {
	return SequenceIndexDeferredOn{sequenceReceiver{index: index, executor: executor}}
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
	lent := open[C.usearch_sequence_index_t]()
	weights, refusal := options.Weights.lower(&lent.call)
	if refusal != nil {
		return sequenceIndexOf(block(lent.refuse(refusal)))
	}
	C.usearch_sequence_init_ram(&C.usearch_sequence_init_ram_t{
		index:               lent.result,
		outcome:             lent.outcome,
		metric:              C.usearch_sequence_metric_t(options.Metric),
		weights:             weights,
		window_widths:       (*C.size_t)(unsafe.Pointer(lend(&lent.call, options.WindowWidths))),
		window_widths_count: C.size_t(len(options.WindowWidths)),
		connectivity:        C.size_t(options.Connectivity),
		values_per_key:      C.size_t(options.ValuesPerKey),
		ram:                 lent.ram(memoryDomains, options.CPUClassCeiling),
	})
	return sequenceIndexOf(block(lent))
}

// SequenceInitVram creates an empty, mutable index in device memory, sharded across the CUDA
// ordinals devices names and never replicated; empty derives one free device.
func SequenceInitVram(options SequenceOptions, devices []int32) (*SequenceIndex, error) {
	lent := open[C.usearch_sequence_index_t]()
	weights, refusal := options.Weights.lower(&lent.call)
	if refusal != nil {
		return sequenceIndexOf(block(lent.refuse(refusal)))
	}
	C.usearch_sequence_init_vram(&C.usearch_sequence_init_vram_t{
		index:               lent.result,
		outcome:             lent.outcome,
		metric:              C.usearch_sequence_metric_t(options.Metric),
		weights:             weights,
		window_widths:       (*C.size_t)(unsafe.Pointer(lend(&lent.call, options.WindowWidths))),
		window_widths_count: C.size_t(len(options.WindowWidths)),
		connectivity:        C.size_t(options.Connectivity),
		values_per_key:      C.size_t(options.ValuesPerKey),
		vram:                lent.vram(devices, options.GPUClassCeiling),
	})
	return sequenceIndexOf(block(lent))
}

// SequenceLoadRam copies a checkpoint into a new, mutable index in host memory. The path names a
// manifest of a sharded checkpoint or a single .safegraphs monolith, and weights are the ones a
// Needleman-Wunsch checkpoint scores with, which no checkpoint records.
func SequenceLoadRam(path string, weights *SequenceWeights, config *DenseLoadRamConfig) (*SequenceIndex, error) {
	if config == nil {
		config = new(DenseLoadRamConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_sequence_index_t]()
	lentWeights, refusal := weights.lower(&lent.call)
	if refusal != nil {
		return sequenceIndexOf(block(lent.refuse(refusal)))
	}
	C.usearch_sequence_load_ram(&C.usearch_sequence_load_ram_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		path:           lent.text(path),
		ram:            lent.ram(config.MemoryDomains, config.CPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		weights:        lentWeights,
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return sequenceIndexOf(block(lent))
}

// SequenceLoadVram copies a checkpoint into a new, mutable index in device memory, with the
// placement rules of SequenceLoadRam.
func SequenceLoadVram(path string, weights *SequenceWeights, config *DenseLoadVramConfig) (*SequenceIndex, error) {
	if config == nil {
		config = new(DenseLoadVramConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_sequence_index_t]()
	lentWeights, refusal := weights.lower(&lent.call)
	if refusal != nil {
		return sequenceIndexOf(block(lent.refuse(refusal)))
	}
	C.usearch_sequence_load_vram(&C.usearch_sequence_load_vram_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		path:           lent.text(path),
		vram:           lent.vram(config.Devices, config.GPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		weights:        lentWeights,
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return sequenceIndexOf(block(lent))
}

// SequenceLoadRamBuffer copies a monolith from a buffer into a new, mutable index in host memory,
// as SequenceLoadRam does a path.
func SequenceLoadRamBuffer(buffer []byte, weights *SequenceWeights,
	config *DenseLoadRamBufferConfig) (*SequenceIndex, error) {
	if config == nil {
		config = new(DenseLoadRamBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_sequence_index_t]()
	lentWeights, refusal := weights.lower(&lent.call)
	if refusal != nil {
		return sequenceIndexOf(block(lent.refuse(refusal)))
	}
	C.usearch_sequence_load_ram_buffer(&C.usearch_sequence_load_ram_buffer_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		buffer:         lent.bytes(buffer),
		ram:            lent.ram(config.MemoryDomains, config.CPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		weights:        lentWeights,
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return sequenceIndexOf(block(lent))
}

// SequenceLoadVramBuffer copies a monolith from a buffer into a new, mutable index in device
// memory.
func SequenceLoadVramBuffer(buffer []byte, weights *SequenceWeights,
	config *DenseLoadVramBufferConfig) (*SequenceIndex, error) {
	if config == nil {
		config = new(DenseLoadVramBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_sequence_index_t]()
	lentWeights, refusal := weights.lower(&lent.call)
	if refusal != nil {
		return sequenceIndexOf(block(lent.refuse(refusal)))
	}
	C.usearch_sequence_load_vram_buffer(&C.usearch_sequence_load_vram_buffer_t{
		index:          lent.result,
		executor:       executor,
		outcome:        lent.outcome,
		buffer:         lent.bytes(buffer),
		vram:           lent.vram(config.Devices, config.GPUClassCeiling),
		values_per_key: C.size_t(config.ValuesPerKey),
		weights:        lentWeights,
		progress:       lent.progress(config.Progress, config.ProgressSteps),
	})
	return sequenceIndexOf(block(lent))
}

// SequenceViewRam serves a checkpoint in place through a memory mapping, as a read-only index
// whose mutating verbs report StatusReadOnly; the files must stay unmodified while it lives.
func SequenceViewRam(path string, weights *SequenceWeights) (*SequenceIndex, error) {
	lent := open[C.usearch_sequence_index_t]()
	lentWeights, refusal := weights.lower(&lent.call)
	if refusal != nil {
		return sequenceIndexOf(block(lent.refuse(refusal)))
	}
	C.usearch_sequence_view_ram(&C.usearch_sequence_view_ram_t{
		index:   lent.result,
		outcome: lent.outcome,
		path:    lent.text(path),
		weights: lentWeights,
	})
	return sequenceIndexOf(block(lent))
}

// SequenceViewNvme serves a checkpoint in place from NVMe drives, read through the executor's
// submission rings, as a read-only index.
func SequenceViewNvme(path string, weights *SequenceWeights, config *SequenceViewNvmeConfig) (*SequenceIndex, error) {
	if config == nil {
		config = new(SequenceViewNvmeConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_sequence_index_t]()
	lentWeights, refusal := weights.lower(&lent.call)
	if refusal != nil {
		return sequenceIndexOf(block(lent.refuse(refusal)))
	}
	C.usearch_sequence_view_nvme(&C.usearch_sequence_view_nvme_t{
		index:    lent.result,
		executor: executor,
		outcome:  lent.outcome,
		path:     lent.text(path),
		nvme:     lent.nvme(config.Paths, config.CPUClassCeiling),
		weights:  lentWeights,
	})
	return sequenceIndexOf(block(lent))
}

// SequenceMetadataRead reads a checkpoint's metric and connectivity without loading it.
func SequenceMetadataRead(path string, config *DenseMetadataReadConfig) (SequenceMetadata, error) {
	if config == nil {
		config = new(DenseMetadataReadConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return SequenceMetadata{}, refusal
	}
	lent := open[SequenceMetadata]()
	C.usearch_sequence_metadata_read(&C.usearch_sequence_metadata_read_t{
		executor: executor,
		outcome:  lent.outcome,
		path:     lent.text(path),
		metadata: (*C.usearch_sequence_metadata_t)(unsafe.Pointer(lent.result)),
	})
	return block(lent)
}

// SequenceMetadataReadBuffer reads the metric and connectivity out of a serialized monolith buffer.
func SequenceMetadataReadBuffer(buffer []byte, config *DenseMetadataReadBufferConfig) (SequenceMetadata, error) {
	if config == nil {
		config = new(DenseMetadataReadBufferConfig)
	}
	executor, refusal := config.Executor.lower()
	if refusal != nil {
		return SequenceMetadata{}, refusal
	}
	lent := open[SequenceMetadata]()
	C.usearch_sequence_metadata_read_buffer(&C.usearch_sequence_metadata_read_buffer_t{
		executor: executor,
		outcome:  lent.outcome,
		buffer:   lent.bytes(buffer),
		metadata: (*C.usearch_sequence_metadata_t)(unsafe.Pointer(lent.result)),
	})
	return block(lent)
}

// endregion: Index

// region: Receivers

// sequenceReceiver is an index and the executor its verbs run on, nil for the calling goroutine.
type sequenceReceiver struct {
	index    *SequenceIndex
	executor *Executor
}

// lower reports the handles a verb hands across, refusing a closed index before C dereferences it.
func (receiver sequenceReceiver) lower() (C.usearch_sequence_index_t, C.usearch_executor_t, *Error) {
	if receiver.index == nil || receiver.index.handle == nil {
		return nil, nil, fail(StatusUninitialized, "the index is closed or was never initialized")
	}
	executor, refusal := receiver.executor.lower()
	return receiver.index.handle, executor, refusal
}

// SequenceIndexOn is the receiver whose verbs block until their call is final: on the calling
// goroutine when embedded in a SequenceIndex, and on an executor when SequenceIndex.On makes it.
// Binding-only.
type SequenceIndexOn struct {
	sequenceReceiver
}

// SequenceIndexDeferredOn is the receiver whose verbs submit their calls to an executor and answer
// a Deferred, offering every verb of SequenceIndexOn with the same arguments but the blocking-only
// sugar and cursors. Binding-only.
//
// The library reads a deferred call's inputs until the call is final, so the slices a verb is
// handed must not change until then; its arena is borrowed, as a blocking verb borrows it.
type SequenceIndexDeferredOn struct {
	sequenceReceiver
}

// Reserve claims capacity for a number of incoming entries and the bytes they hold, ahead of
// insertion. Both are desired totals, including what the index holds.
func (on SequenceIndexOn) Reserve(entries, payloadBytes uint) error {
	_, err := block(on.reserve(C.usearch_completion_blocking_k, entries, payloadBytes))
	return err
}

// Reserve submits SequenceIndexOn.Reserve.
func (on SequenceIndexDeferredOn) Reserve(entries, payloadBytes uint) *Deferred[struct{}] {
	return submitted(on.reserve(C.usearch_completion_deferred_k, entries, payloadBytes))
}

// Clear erases all entries, keeping the reserved capacity. No other verb may overlap it on the
// same handle.
func (on SequenceIndexOn) Clear() error {
	_, err := block(on.clear(C.usearch_completion_blocking_k))
	return err
}

// Clear submits SequenceIndexOn.Clear.
func (on SequenceIndexDeferredOn) Clear() *Deferred[struct{}] {
	return submitted(on.clear(C.usearch_completion_deferred_k))
}

// Describe reports the frozen metadata and the current counters in one call.
func (on SequenceIndexOn) Describe() (SequenceIndexDescription, error) {
	return block(on.describe(C.usearch_completion_blocking_k))
}

// Describe submits SequenceIndexOn.Describe.
func (on SequenceIndexDeferredOn) Describe() *Deferred[SequenceIndexDescription] {
	return submitted(on.describe(C.usearch_completion_deferred_k))
}

// Count reports how many entries are stored under a key, with the extent of their bytes.
func (on SequenceIndexOn) Count(key Key) (SequenceKeyStats, error) {
	return block(on.count(C.usearch_completion_blocking_k, key))
}

// Count submits SequenceIndexOn.Count.
func (on SequenceIndexDeferredOn) Count(key Key) *Deferred[SequenceKeyStats] {
	return submitted(on.count(C.usearch_completion_deferred_k, key))
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
	return block(on.get(C.usearch_completion_blocking_k, key, arena))
}

// Get submits SequenceIndexOn.Get.
func (on SequenceIndexDeferredOn) Get(key Key, arena *Arena) *Deferred[TapeView] {
	return submitted(on.get(C.usearch_completion_deferred_k, key, arena))
}

// Insert adds a tape of sequences under their keys - the native batch operation, where a single
// sequence is a one-entry tape. A batch is all-or-nothing and becomes visible at once.
func (on SequenceIndexOn) Insert(keys []Key, sequences Tape, config *DenseInsertConfig) error {
	_, err := block(on.insert(C.usearch_completion_blocking_k, keys, sequences, config))
	return err
}

// Insert submits SequenceIndexOn.Insert.
func (on SequenceIndexDeferredOn) Insert(keys []Key, sequences Tape, config *DenseInsertConfig) *Deferred[struct{}] {
	return submitted(on.insert(C.usearch_completion_deferred_k, keys, sequences, config))
}

// Refine re-selects the neighbors of every stored entry against the finished graph, as the dense
// Refine does; a caller repeats rounds until the report's Verdict reads DenseRefineConverged.
func (on SequenceIndexOn) Refine(config *SequenceRefineConfig) (DenseRefineReport, error) {
	return block(on.refine(C.usearch_completion_blocking_k, config))
}

// Refine submits SequenceIndexOn.Refine.
func (on SequenceIndexDeferredOn) Refine(config *SequenceRefineConfig) *Deferred[DenseRefineReport] {
	return submitted(on.refine(C.usearch_completion_deferred_k, config))
}

// Health measures how much of the graph a search reaches from the root. Read-only.
func (on SequenceIndexOn) Health() (DenseReachability, error) {
	return block(on.health(C.usearch_completion_blocking_k))
}

// Health submits SequenceIndexOn.Health.
func (on SequenceIndexDeferredOn) Health() *Deferred[DenseReachability] {
	return submitted(on.health(C.usearch_completion_deferred_k))
}

// Search resolves a tape of queries into up to wanted nearest matches apiece, written into arena,
// and filtered when the config carries a predicate. Never writes to the graph.
func (on SequenceIndexOn) Search(queries Tape, wanted uint, arena *Arena, config *DenseSearchConfig) (MatchesView, error) {
	return block(on.search(C.usearch_completion_blocking_k, queries, wanted, arena, config))
}

// Search submits SequenceIndexOn.Search.
func (on SequenceIndexDeferredOn) Search(queries Tape, wanted uint, arena *Arena,
	config *DenseSearchConfig) *Deferred[MatchesView] {
	return submitted(on.search(C.usearch_completion_deferred_k, queries, wanted, arena, config))
}

// Reshard moves the entries of sources into targets, keeping their links rather than re-inserting
// them, then runs a round on every target that took entries: several sources gather into one
// target, and one source cuts into one graph-local part per target. The receiver contributes only
// its executor, and a refusal moves nothing.
func (on SequenceIndexOn) Reshard(sources, targets []*SequenceIndex, config *SequenceReshardConfig) (DenseRefineReport, error) {
	return block(on.reshard(C.usearch_completion_blocking_k, sources, targets, config))
}

// Reshard submits SequenceIndexOn.Reshard.
func (on SequenceIndexDeferredOn) Reshard(sources, targets []*SequenceIndex,
	config *SequenceReshardConfig) *Deferred[DenseRefineReport] {
	return submitted(on.reshard(C.usearch_completion_deferred_k, sources, targets, config))
}

// RefinePlanFinish closes a live refine plan and frees it whatever the call answers, reporting
// what the whole plan did to the graph.
func (on SequenceIndexOn) RefinePlanFinish(plan *SequenceRefinePlan) (DenseRefineReport, error) {
	return block(on.refinePlanFinish(C.usearch_completion_blocking_k, plan))
}

// RefinePlanFinish submits SequenceIndexOn.RefinePlanFinish.
func (on SequenceIndexDeferredOn) RefinePlanFinish(plan *SequenceRefinePlan) *Deferred[DenseRefineReport] {
	return submitted(on.refinePlanFinish(C.usearch_completion_deferred_k, plan))
}

// ReshardPlanFinish closes a live reshard plan on this receiver's executor and frees it whatever
// the call answers, reporting what the whole plan did.
func (on SequenceIndexOn) ReshardPlanFinish(plan *SequenceReshardPlan) (DenseRefineReport, error) {
	return block(on.reshardPlanFinish(C.usearch_completion_blocking_k, plan))
}

// ReshardPlanFinish submits SequenceIndexOn.ReshardPlanFinish.
func (on SequenceIndexDeferredOn) ReshardPlanFinish(plan *SequenceReshardPlan) *Deferred[DenseRefineReport] {
	return submitted(on.reshardPlanFinish(C.usearch_completion_deferred_k, plan))
}

// Sample draws up to wanted keys around the config's Focus, or across the whole index, each
// hanging from a shown or earlier draw and placed in three dimensions beside the shown ones.
func (on SequenceIndexOn) Sample(wanted uint, arena *Arena, config *DenseSampleConfig) (DenseDraws, error) {
	return block(on.sample(C.usearch_completion_blocking_k, wanted, arena, config))
}

// Sample submits SequenceIndexOn.Sample.
func (on SequenceIndexDeferredOn) Sample(wanted uint, arena *Arena, config *DenseSampleConfig) *Deferred[DenseDraws] {
	return submitted(on.sample(C.usearch_completion_deferred_k, wanted, arena, config))
}

// Save writes the index under a path, in the layout its extension picks - one .safegraphs
// monolith, or a .safegraphs.json manifest beside its shards.
func (on SequenceIndexOn) Save(path string, config *DenseSaveConfig) error {
	_, err := block(on.save(C.usearch_completion_blocking_k, path, config))
	return err
}

// Save submits SequenceIndexOn.Save.
func (on SequenceIndexDeferredOn) Save(path string, config *DenseSaveConfig) *Deferred[struct{}] {
	return submitted(on.save(C.usearch_completion_deferred_k, path, config))
}

// SaveBuffer serializes the index as one monolith into arena.
func (on SequenceIndexOn) SaveBuffer(arena *Arena, config *DenseSaveBufferConfig) (Monolith, error) {
	return block(on.saveBuffer(C.usearch_completion_blocking_k, arena, config))
}

// SaveBuffer submits SequenceIndexOn.SaveBuffer.
func (on SequenceIndexDeferredOn) SaveBuffer(arena *Arena, config *DenseSaveBufferConfig) *Deferred[Monolith] {
	return submitted(on.saveBuffer(C.usearch_completion_deferred_k, arena, config))
}

// endregion: Receivers

// region: Calls

func (receiver sequenceReceiver) reserve(completion C.usearch_completion_t, entries, payloadBytes uint) *lent[struct{}] {
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_sequence_reserve(&C.usearch_sequence_reserve_t{
		index:         index,
		executor:      executor,
		outcome:       lent.outcome,
		completion:    completion,
		entries:       C.size_t(entries),
		payload_bytes: C.size_t(payloadBytes),
	})
	return lent
}

func (receiver sequenceReceiver) clear(completion C.usearch_completion_t) *lent[struct{}] {
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_sequence_clear(&C.usearch_sequence_clear_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
	})
	return lent
}

func (receiver sequenceReceiver) describe(completion C.usearch_completion_t) *lent[SequenceIndexDescription] {
	lent := open[SequenceIndexDescription]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_sequence_describe(&C.usearch_sequence_describe_t{
		index:       index,
		executor:    executor,
		outcome:     lent.outcome,
		completion:  completion,
		description: (*C.usearch_sequence_index_description_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver sequenceReceiver) count(completion C.usearch_completion_t, key Key) *lent[SequenceKeyStats] {
	lent := open[SequenceKeyStats]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_sequence_count(&C.usearch_sequence_count_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		key:        key.toRaw(),
		key_stats:  (*C.usearch_sequence_key_stats_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver sequenceReceiver) get(completion C.usearch_completion_t, key Key, arena *Arena) *lent[TapeView] {
	lent := open[TapeView]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	lent.result.width = 1
	C.usearch_sequence_get(&C.usearch_sequence_get_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		key:        key.toRaw(),
		sequences:  &lent.result.raw,
		written:    (*C.size_t)(unsafe.Pointer(&lent.result.written)),
		arena:      arena.lower(),
	})
	return lent
}

func (receiver sequenceReceiver) insert(completion C.usearch_completion_t, keys []Key, sequences Tape,
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
	if uint(len(keys)) != sequences.Entries() {
		return lent.refuse(fail(StatusShapeMismatch, "the key count does not match the tape's entries"))
	}
	C.usearch_sequence_insert(&C.usearch_sequence_insert_t{
		index:            index,
		executor:         executor,
		outcome:          lent.outcome,
		completion:       completion,
		count:            C.size_t(len(keys)),
		keys:             lent.keys(keys),
		sequences:        sequences.lower(&lent.call),
		expansion_factor: C.size_t(config.ExpansionFactor),
		progress:         lent.progress(config.Progress, config.ProgressSteps),
		stats:            lent.stats(config.Stats),
	})
	return lent
}

func (receiver sequenceReceiver) refine(completion C.usearch_completion_t,
	config *SequenceRefineConfig) *lent[DenseRefineReport] {
	if config == nil {
		config = new(SequenceRefineConfig)
	}
	lent := open[DenseRefineReport]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	plan, phase := config.Plan.lower(&lent.call)
	C.usearch_sequence_refine(&C.usearch_sequence_refine_t{
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

func (receiver sequenceReceiver) refinePlanFinish(completion C.usearch_completion_t,
	plan *SequenceRefinePlan) *lent[DenseRefineReport] {
	lent := open[DenseRefineReport]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_sequence_refine_plan_finish(&C.usearch_sequence_refine_plan_finish_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver sequenceReceiver) health(completion C.usearch_completion_t) *lent[DenseReachability] {
	lent := open[DenseReachability]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_sequence_health(&C.usearch_sequence_health_t{
		index:        index,
		executor:     executor,
		outcome:      lent.outcome,
		completion:   completion,
		reachability: (*C.usearch_reachability_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver sequenceReceiver) search(completion C.usearch_completion_t, queries Tape, wanted uint,
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
	C.usearch_sequence_search(&C.usearch_sequence_search_t{
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

func (receiver sequenceReceiver) reshard(completion C.usearch_completion_t, sources, targets []*SequenceIndex,
	config *SequenceReshardConfig) *lent[DenseRefineReport] {
	if config == nil {
		config = new(SequenceReshardConfig)
	}
	lent := open[DenseRefineReport]()
	// A reshard names its handles in its lists, so the receiver lends only its executor.
	executor, refusal := receiver.executor.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	sourceHandles, targetHandles := sequenceHandles(sources), sequenceHandles(targets)
	plan, phase := config.Plan.lower(&lent.call)
	C.usearch_sequence_reshard(&C.usearch_sequence_reshard_t{
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

// sequenceHandles lists the handles of indexes.
func sequenceHandles(indexes []*SequenceIndex) []C.usearch_sequence_index_t {
	handles := make([]C.usearch_sequence_index_t, len(indexes))
	for position, index := range indexes {
		handles[position] = index.handle
	}
	return handles
}

func (receiver sequenceReceiver) reshardPlanFinish(completion C.usearch_completion_t,
	plan *SequenceReshardPlan) *lent[DenseRefineReport] {
	lent := open[DenseRefineReport]()
	// A reshard names its handles in its lists, so the receiver lends only its executor.
	executor, refusal := receiver.executor.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_sequence_reshard_plan_finish(&C.usearch_sequence_reshard_plan_finish_t{
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		plan:       plan.take(),
		report:     (*C.usearch_refine_report_t)(unsafe.Pointer(lent.result)),
	})
	return lent
}

func (receiver sequenceReceiver) sample(completion C.usearch_completion_t, wanted uint, arena *Arena,
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
	C.usearch_sequence_sample(&C.usearch_sequence_sample_t{
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

func (receiver sequenceReceiver) save(completion C.usearch_completion_t, path string,
	config *DenseSaveConfig) *lent[struct{}] {
	if config == nil {
		config = new(DenseSaveConfig)
	}
	lent := open[struct{}]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_sequence_save(&C.usearch_sequence_save_t{
		index:      index,
		executor:   executor,
		outcome:    lent.outcome,
		completion: completion,
		path:       lent.text(path),
		progress:   lent.progress(config.Progress, config.ProgressSteps),
	})
	return lent
}

func (receiver sequenceReceiver) saveBuffer(completion C.usearch_completion_t, arena *Arena,
	config *DenseSaveBufferConfig) *lent[Monolith] {
	if config == nil {
		config = new(DenseSaveBufferConfig)
	}
	lent := open[Monolith]()
	index, executor, refusal := receiver.lower()
	if refusal != nil {
		return lent.refuse(refusal)
	}
	C.usearch_sequence_save_buffer(&C.usearch_sequence_save_buffer_t{
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

// SequenceCursor is a walk over the index contents, pulled blocking on the executor of the
// receiver that opened it; an entry once passed is never revisited.
type SequenceCursor struct {
	handle   C.usearch_sequence_cursor_t
	receiver sequenceReceiver
}

// Cursor opens a walk over the current contents. Without starts it yields every entry as stored;
// with them, the start keys' own entries first in order, then the entries those link to
// breadth-first, each once.
func (on SequenceIndexOn) Cursor(starts ...Key) (*SequenceCursor, error) {
	index, _, refusal := on.lower()
	if refusal != nil {
		return nil, refusal
	}
	lent := open[C.usearch_sequence_cursor_t]()
	C.usearch_sequence_cursor_init(&C.usearch_sequence_cursor_init_t{
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
	return &SequenceCursor{handle: handle, receiver: on.sequenceReceiver}, nil
}

// NextKeys copies up to wanted next keys into arena; a batch of none means the walk is exhausted.
func (cursor *SequenceCursor) NextKeys(wanted uint, arena *Arena) (DenseBatch, error) {
	return cursor.next(wanted, false, arena)
}

// NextLinks copies up to wanted next keys into arena, each with the keys it links to; a batch of
// none means the walk is exhausted.
func (cursor *SequenceCursor) NextLinks(wanted uint, arena *Arena) (DenseBatch, error) {
	return cursor.next(wanted, true, arena)
}

// next pulls a batch, exporting each entry's links only when linked asks for them.
func (cursor *SequenceCursor) next(wanted uint, linked bool, arena *Arena) (DenseBatch, error) {
	index, executor, refusal := cursor.receiver.lower()
	if refusal != nil {
		return DenseBatch{}, refusal
	}
	lent := open[DenseBatch]()
	var links *C.usearch_keys_tape_view_t
	if linked {
		links = &lent.result.links
	}
	C.usearch_sequence_cursor_next(&C.usearch_sequence_cursor_next_t{
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
func (cursor *SequenceCursor) Close() error {
	if cursor == nil || cursor.handle == nil {
		return nil
	}
	C.usearch_sequence_cursor_free(cursor.handle)
	cursor.handle = nil
	return nil
}

// endregion: Cursors
