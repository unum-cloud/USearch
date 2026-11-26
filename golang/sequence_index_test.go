// Tests for the sequence modality.
//
// File: golang/sequence_index_test.go
// Author: Ash Vardanian
// Date: November 16, 2026

// Each modality runs the same cases: a fresh index, both serialized forms, a refine round and its
// plan, and every reshard with its refusals. The cases mirror test/modalities.py, and the shared
// helpers pool, newArena and keysFrom come from dense_index_test.go.

package usearch

import (
	"errors"
	"math/rand"
	"path/filepath"
	"testing"
)

// region: Helpers

// randomSequences draws count lowercase sequences of 8 to 16 bytes.
func randomSequences(count uint, seed int64) Tape {
	generator := rand.New(rand.NewSource(seed))
	sequences := make([][]byte, count)
	for position := range sequences {
		sequence := make([]byte, 8+generator.Intn(9))
		for offset := range sequence {
			sequence[offset] = byte('a' + generator.Intn(26))
		}
		sequences[position] = sequence
	}
	return SequencesOf(sequences...)
}

// endregion: Helpers

// region: Entries

// Every sequence searched for comes back first under its own key, at distance zero, and Get hands
// back its bytes.
func TestSequenceSearchFindsTheInsertedEntry(t *testing.T) {
	t.Parallel()
	options := DefaultSequenceOptions()
	options.Metric = SequenceMetricLevenshteinUTF8
	index, err := SequenceInitRam(options, nil)
	if err != nil {
		t.Fatalf("the index must initialize: %v", err)
	}
	defer index.Close()
	if err := index.Insert(keysFrom(1, 3), SequencesOf([]byte("kitten"), []byte("sitting"), []byte("mitten")), nil); err != nil {
		t.Fatalf("the batch must insert: %v", err)
	}
	if description, err := index.Describe(); err != nil || description.Size != 3 ||
		description.Metadata.Metric != SequenceMetricLevenshteinUTF8 {
		t.Errorf("the index describes %+v: %v", description, err)
	}
	arena := newArena(t, nil)
	matches, err := index.Search(SequencesOf([]byte("sitting")), 1, arena, nil)
	if err != nil {
		t.Fatalf("the search must run: %v", err)
	}
	if keys, distances, _ := matches.Query(0); len(keys) != 1 || keys[0] != KeyFromUint64(2) || distances[0] != 0 {
		t.Errorf("the search found %v at %v", keys, distances)
	}
	if stats, err := index.Count(KeyFromUint64(2)); err != nil || stats.Entries != 1 || stats.TotalBytes != 7 {
		t.Errorf("the key counts %+v: %v", stats, err)
	}
	stored, err := index.Get(KeyFromUint64(2), arena)
	if err != nil || stored.Written() != 1 {
		t.Fatalf("the key must get: %+v %v", stored, err)
	}
	if sequence, err := TapeEntry[byte](stored, 0); err != nil || string(sequence) != "sitting" {
		t.Errorf("the key got %q: %v", sequence, err)
	}
	if err := index.Insert(keysFrom(9, 2), SequencesOf([]byte("one")), nil); !errors.Is(err, StatusShapeMismatch) {
		t.Errorf("two keys for one sequence reported %v", err)
	}
}

// Metrics print as the library spells them.
func TestSequenceMetricsSpellTheLibrarysNames(t *testing.T) {
	t.Parallel()
	if name := SequenceMetricLevenshteinBytes.String(); name == "" || name == "unknown" {
		t.Errorf("the byte-level Levenshtein kind prints %q", name)
	}
	if name := SequenceMetric(99).String(); name != "unrecognized" && name != "unknown" {
		t.Errorf("an unknown sequence kind prints %q", name)
	}
}

// Every metric reads back from its own spelling, and a spelling no metric holds is refused.
func TestSequenceMetricsReadBackFromTheirNames(t *testing.T) {
	t.Parallel()
	for _, metric := range []SequenceMetric{SequenceMetricLevenshteinBytes, SequenceMetricLevenshteinUTF8,
		SequenceMetricNeedlemanWunsch, SequenceMetricOverlap} {
		if read, err := SequenceMetricFromName(metric.String()); err != nil || read != metric {
			t.Errorf("%q reads back as %v: %v", metric.String(), read, err)
		}
	}
	if _, err := SequenceMetricFromName("hamming-ish"); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a misspelled sequence metric reads back with %v", err)
	}
}

// endregion: Entries

// region: Lifecycle

// buildSequence fills a fresh host index with count random entries under consecutive keys from
// first.
func buildSequence(t *testing.T, executor *Executor, count, first uint, seed int64) *SequenceIndex {
	t.Helper()
	index, err := SequenceInitRam(DefaultSequenceOptions(), nil)
	if err != nil {
		t.Fatalf("the index must initialize: %v", err)
	}
	t.Cleanup(func() {
		if err := index.Close(); err != nil {
			t.Errorf("the index must close: %v", err)
		}
	})
	if count != 0 {
		if err := index.On(executor).Insert(keysFrom(first, count), randomSequences(count, seed), nil); err != nil {
			t.Fatalf("the batch must insert: %v", err)
		}
	}
	return index
}

// sizeSequence reports how many entries an index describes.
func sizeSequence(t *testing.T, index *SequenceIndex) uint {
	t.Helper()
	description, err := index.Describe()
	if err != nil {
		t.Fatalf("the index must describe: %v", err)
	}
	return description.Size
}

func TestSequenceInitRamStartsEmpty(t *testing.T) {
	t.Parallel()
	index := buildSequence(t, nil, 0, 0, 0)
	description, err := index.Describe()
	if err != nil || description.Size != 0 || description.CPUClass == CPUClassUnknown ||
		description.GPUClass != GPUClassUnknown {
		t.Errorf("a fresh host index describes %+v: %v", description, err)
	}
}

func TestSequenceInitVramSurfacesTheLibraryStatus(t *testing.T) {
	t.Parallel()
	index, err := SequenceInitVram(DefaultSequenceOptions(), nil)
	if err != nil {
		var refusal *Error
		if !errors.As(err, &refusal) || refusal.Status == StatusSuccess || refusal.Status == StatusDeferred {
			t.Errorf("a VRAM index refused with %v", err)
		}
		return
	}
	defer index.Close()
	if description, err := index.Describe(); err != nil || description.GPUClass == GPUClassUnknown {
		t.Errorf("a VRAM index describes %+v: %v", description, err)
	}
}

// Every entry a cursor walks comes back once, and a closed index refuses its verbs.
func TestSequenceCursorsWalkEveryEntryOnce(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildSequence(t, executor, 16, 0, 1)
	cursor, err := index.Cursor()
	if err != nil {
		t.Fatalf("the cursor must open: %v", err)
	}
	defer cursor.Close()
	arena := newArena(t, executor)
	seen := map[Key]bool{}
	for {
		batch, err := cursor.On(executor).NextLinks(5, arena)
		if err != nil {
			t.Fatalf("the cursor must pull: %v", err)
		}
		if batch.Written() == 0 {
			break
		}
		pulled, err := batch.Keys()
		if err != nil {
			t.Fatalf("the batch must read: %v", err)
		}
		for _, key := range pulled {
			seen[key] = true
		}
	}
	if len(seen) != 16 {
		t.Errorf("the walk saw %d of 16 keys", len(seen))
	}
	closed := buildSequence(t, nil, 0, 0, 0)
	if err := closed.Close(); err != nil {
		t.Fatalf("the index must close: %v", err)
	}
	if _, err := closed.Describe(); !errors.Is(err, StatusUninitialized) {
		t.Errorf("a closed index described with %v", err)
	}
}

// Both serialized forms load back holding every entry, and a view serves the same file in place.
func TestSequenceSavesAndLoads(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildSequence(t, executor, 32, 0, 2)
	path := filepath.Join(t.TempDir(), "index.safegraphs")
	if err := index.On(executor).Save(path, nil); err != nil {
		t.Fatalf("the index must save: %v", err)
	}
	monolith, err := index.On(executor).SaveBuffer(newArena(t, executor), nil)
	if err != nil {
		t.Fatalf("the index must serialize: %v", err)
	}
	if _, err := SequenceMetadataRead(path, nil); err != nil {
		t.Errorf("the checkpoint's metadata must read: %v", err)
	}
	buffer, err := monolith.Bytes()
	if err != nil {
		t.Fatalf("the monolith must read: %v", err)
	}
	if _, err := SequenceMetadataReadBuffer(buffer, nil); err != nil {
		t.Errorf("the monolith's metadata must read: %v", err)
	}
	loaded := []func() (*SequenceIndex, error){
		func() (*SequenceIndex, error) { return SequenceLoadRam(path, nil, nil) },
		func() (*SequenceIndex, error) { return SequenceLoadRamBuffer(buffer, nil, nil) },
		func() (*SequenceIndex, error) { return SequenceViewRam(path, nil) },
	}
	for position, load := range loaded {
		copied, err := load()
		if position == 2 && errors.Is(err, StatusUnsupported) {
			continue // a residency that serves no view in place says so
		}
		if err != nil {
			t.Fatalf("form %d must load: %v", position, err)
		}
		if size := sizeSequence(t, copied); size != 32 {
			t.Errorf("form %d holds %d of 32 entries", position, size)
		}
		if held, err := copied.Contains(KeyFromUint64(17)); err != nil || !held {
			t.Errorf("form %d lost key 17: %v", position, err)
		}
		copied.Close()
	}
}

// A round reports what it did, its closing audit judging every entry, and leaves each entry
// findable.
func TestSequenceRefineReportsARoundOverEveryEntry(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildSequence(t, executor, 64, 0, 0)
	report, err := index.On(executor).Refine(nil)
	if err != nil {
		t.Fatalf("the round must run: %v", err)
	}
	if report.Changed > report.Reselected || report.Reachability.Audited != 64 || report.Verdict == RefineNone {
		t.Errorf("the round reported %+v", report)
	}
	if held, err := index.Contains(KeyFromUint64(17)); err != nil || !held {
		t.Errorf("the round lost key 17: %v", err)
	}
}

// A plan keeps a stopped round live for the next call, and the receiver closes it once.
func TestSequenceRefinePlanResumesAndFinishes(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildSequence(t, executor, 64, 0, 0)
	on := index.On(executor)
	var plan SequenceRefinePlan
	stops := func(uint, uint) bool { return false }
	if _, err := on.Refine(&SequenceRefineConfig{Plan: &plan, Progress: stops}); err != nil && !errors.Is(err, StatusCancelled) {
		t.Fatalf("the stopped round reported %v", err)
	}
	if plan.Progress.Phase == PhaseDone || plan.Progress.Completed > plan.Progress.Total {
		t.Errorf("a stopped round left %+v", plan.Progress)
	}
	if _, err := on.Refine(&SequenceRefineConfig{Plan: &plan}); err != nil || plan.Progress.Phase != PhaseDone {
		t.Fatalf("the resumed round ended in %+v: %v", plan.Progress, err)
	}
	if report, err := on.RefinePlanFinish(&plan); err != nil || report.Reachability.Audited != 64 {
		t.Errorf("the finished plan reported %+v: %v", report, err)
	}
	if _, err := on.RefinePlanFinish(&plan); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a finished plan finished again with %v", err)
	}
}

// A plan lives across calls on one executor, so a call on the calling goroutine cannot open one.
func TestSequenceRefinePlanNeedsAnExecutor(t *testing.T) {
	t.Parallel()
	index := buildSequence(t, nil, 0, 0, 0)
	if _, err := index.Refine(&SequenceRefineConfig{Plan: new(SequenceRefinePlan)}); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a plan on the calling goroutine opened with %v", err)
	}
}

// Several sources into one target gather every entry there, and the other sources end empty
// and live.
func TestSequenceReshardGathersEverySourceIntoTheTarget(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	first := buildSequence(t, executor, 32, 0, 0)
	donors := []*SequenceIndex{buildSequence(t, executor, 32, 32, 1), buildSequence(t, executor, 32, 64, 2)}
	if _, err := first.On(executor).Reshard(append([]*SequenceIndex{first}, donors...), []*SequenceIndex{first}, nil); err != nil {
		t.Fatalf("the gather must run: %v", err)
	}
	if size := sizeSequence(t, first); size != 96 {
		t.Errorf("the target holds %d of 96 entries", size)
	}
	for _, key := range []uint64{40, 80} {
		if held, err := first.Contains(KeyFromUint64(key)); err != nil || !held {
			t.Errorf("the target lacks key %d: %v", key, err)
		}
	}
	for position, donor := range donors {
		if size := sizeSequence(t, donor); size != 0 {
			t.Errorf("donor %d kept %d entries", position, size)
		}
	}
}

// One source into several targets cuts it between them, and the source stays live but empty.
func TestSequenceReshardCutsOneSourceIntoSeveralTargets(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	source := buildSequence(t, executor, 64, 0, 0)
	targets := []*SequenceIndex{buildSequence(t, nil, 0, 0, 0), buildSequence(t, nil, 0, 0, 0)}
	if _, err := source.On(executor).Reshard([]*SequenceIndex{source}, targets, &SequenceReshardConfig{Seed: 7}); err != nil {
		t.Fatalf("the cut must run: %v", err)
	}
	left, right := sizeSequence(t, targets[0]), sizeSequence(t, targets[1])
	if left+right != 64 || left == 0 || right == 0 || sizeSequence(t, source) != 0 {
		t.Errorf("the cut left %d and %d, and %d in the source", left, right, sizeSequence(t, source))
	}
}

// A cut a stop interrupts keeps every entry moved, the next call naming the plan ends its rounds,
// and a deferred receiver closes it.
func TestSequenceReshardCutRunsThroughAPlanAndADeferredFinish(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	source := buildSequence(t, executor, 64, 0, 0)
	targets := []*SequenceIndex{buildSequence(t, nil, 0, 0, 0), buildSequence(t, nil, 0, 0, 0)}
	sources := []*SequenceIndex{source}
	var plan SequenceReshardPlan
	stops := func(uint, uint) bool { return false }
	config := &SequenceReshardConfig{Plan: &plan, Progress: stops}
	if _, err := source.On(executor).Reshard(sources, targets, config); err != nil && !errors.Is(err, StatusCancelled) {
		t.Fatalf("the stopped cut reported %v", err)
	}
	if moved := sizeSequence(t, targets[0]) + sizeSequence(t, targets[1]); moved != 64 || plan.Progress.Phase == PhaseDone {
		t.Errorf("a stop moved %d of 64 entries and left %+v", moved, plan.Progress)
	}
	if _, err := source.On(executor).Reshard(sources, targets, &SequenceReshardConfig{Plan: &plan}); err != nil ||
		plan.Progress.Phase != PhaseDone {
		t.Fatalf("the continued cut ended in %+v: %v", plan.Progress, err)
	}
	finished, err := source.Deferred(executor).ReshardPlanFinish(&plan)
	if err != nil {
		t.Fatalf("the deferred finish must submit: %v", err)
	}
	if _, err := finished.Wait(); err != nil {
		t.Errorf("the deferred finish reported %v", err)
	}
}

// One index as its only source and target has a single shard to recut, which the library refuses.
func TestSequenceReshardRecutOfOneHandleIsUnsupported(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildSequence(t, executor, 16, 0, 0)
	only := []*SequenceIndex{index}
	if _, err := index.On(executor).Reshard(only, only, nil); !errors.Is(err, StatusUnsupported) {
		t.Errorf("a recut of one handle reported %v", err)
	}
	if size := sizeSequence(t, index); size != 16 {
		t.Errorf("the refusal left %d of 16 entries", size)
	}
}

// Empty or repeated lists are malformed and a target holding foreign entries mismatches; every
// refusal leaves the entries where they were.
func TestSequenceReshardRefusesMalformedListsAndMovesNothing(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	first, second := buildSequence(t, executor, 8, 0, 0), buildSequence(t, executor, 8, 8, 1)
	on := first.On(executor)
	if _, err := on.Reshard(nil, []*SequenceIndex{first}, nil); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("no sources reported %v", err)
	}
	if _, err := on.Reshard([]*SequenceIndex{first, first}, []*SequenceIndex{first}, nil); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a repeated source reported %v", err)
	}
	if _, err := on.Reshard([]*SequenceIndex{first}, []*SequenceIndex{second}, nil); !errors.Is(err, StatusConfigMismatch) {
		t.Errorf("a foreign target reported %v", err)
	}
	if sizeSequence(t, first) != 8 || sizeSequence(t, second) != 8 {
		t.Errorf("a refusal moved entries: %d and %d", sizeSequence(t, first), sizeSequence(t, second))
	}
}

// endregion: Lifecycle
