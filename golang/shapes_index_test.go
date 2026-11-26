// Tests for the shapes modality.
//
// File: golang/shapes_index_test.go
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

// randomShapes draws count shapes of 4 to 12 points in the unit cube.
func randomShapes(count uint, seed int64) Tape {
	generator := rand.New(rand.NewSource(seed))
	shapes := make([][]float32, count)
	for position := range shapes {
		shape := make([]float32, 3*(4+generator.Intn(9)))
		for offset := range shape {
			shape[offset] = generator.Float32()
		}
		shapes[position] = shape
	}
	return ShapesOf(shapes...)
}

// endregion: Helpers

// region: Entries

// Shapes of different point counts share one batch, each finds itself first, and Get hands back
// its points.
func TestShapesSearchFindsTheInsertedEntry(t *testing.T) {
	t.Parallel()
	generator := rand.New(rand.NewSource(42))
	shapes := make([][]float32, 3)
	for position, points := range []int{5, 9, 13} {
		shapes[position] = make([]float32, 3*points)
		for offset := range shapes[position] {
			shapes[position][offset] = generator.Float32()
		}
	}
	options := DefaultShapesOptions()
	options.Metric = ShapesMetricHausdorff
	index, err := ShapesInitRam(options, nil)
	if err != nil {
		t.Fatalf("the index must initialize: %v", err)
	}
	defer index.Close()
	if err := index.Insert(keysFrom(1, 3), ShapesOf(shapes...), nil); err != nil {
		t.Fatalf("the batch must insert: %v", err)
	}
	if description, err := index.Describe(); err != nil || description.Size != 3 || description.TotalPoints != 27 ||
		description.Metadata.Metric != ShapesMetricHausdorff {
		t.Errorf("the index describes %+v: %v", description, err)
	}
	arena := newArena(t, nil)
	matches, err := index.Search(ShapesOf(shapes[2]), 1, arena, nil)
	if err != nil {
		t.Fatalf("the search must run: %v", err)
	}
	if keys, _, _ := matches.Query(0); len(keys) != 1 || keys[0] != KeyFromUint64(3) {
		t.Errorf("the search found %v", keys)
	}
	if stats, err := index.Count(KeyFromUint64(3)); err != nil || stats.Entries != 1 || stats.TotalPoints != 13 {
		t.Errorf("the key counts %+v: %v", stats, err)
	}
	stored, err := index.Get(KeyFromUint64(2), ScalarF32, arena)
	if err != nil || stored.Written() != 1 {
		t.Fatalf("the key must get: %+v %v", stored, err)
	}
	if points, err := TapeEntry[float32](stored, 0); err != nil || len(points) != 27 {
		t.Errorf("the key got %d scalars: %v", len(points), err)
	}
	if _, err := TapeEntry[float64](stored, 0); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("reading f32 points as f64 reported %v", err)
	}
}

// Metrics print as the library spells them.
func TestShapesMetricsSpellTheLibrarysNames(t *testing.T) {
	t.Parallel()
	if name := ShapesMetricChamfer.String(); name == "" || name == "unknown" {
		t.Errorf("the Chamfer kind prints %q", name)
	}
}

// Every metric reads back from its own spelling, and an empty spelling is refused.
func TestShapesMetricsReadBackFromTheirNames(t *testing.T) {
	t.Parallel()
	if read, err := ShapesMetricFromName(ShapesMetricChamfer.String()); err != nil || read != ShapesMetricChamfer {
		t.Errorf("%q reads back as %v: %v", ShapesMetricChamfer.String(), read, err)
	}
	if _, err := ShapesMetricFromName(""); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("an empty shapes metric name reads back with %v", err)
	}
}

// endregion: Entries

// region: Lifecycle

// buildShapes fills a fresh host index with count random entries under consecutive keys from first.
func buildShapes(t *testing.T, executor *Executor, count, first uint, seed int64) *ShapesIndex {
	t.Helper()
	index, err := ShapesInitRam(DefaultShapesOptions(), nil)
	if err != nil {
		t.Fatalf("the index must initialize: %v", err)
	}
	t.Cleanup(func() {
		if err := index.Close(); err != nil {
			t.Errorf("the index must close: %v", err)
		}
	})
	if count != 0 {
		if err := index.On(executor).Insert(keysFrom(first, count), randomShapes(count, seed), nil); err != nil {
			t.Fatalf("the batch must insert: %v", err)
		}
	}
	return index
}

// sizeShapes reports how many entries an index describes.
func sizeShapes(t *testing.T, index *ShapesIndex) uint {
	t.Helper()
	description, err := index.Describe()
	if err != nil {
		t.Fatalf("the index must describe: %v", err)
	}
	return description.Size
}

func TestShapesInitRamStartsEmpty(t *testing.T) {
	t.Parallel()
	index := buildShapes(t, nil, 0, 0, 0)
	description, err := index.Describe()
	if err != nil || description.Size != 0 || description.CPUClass == CPUClassUnknown ||
		description.GPUClass != GPUClassUnknown {
		t.Errorf("a fresh host index describes %+v: %v", description, err)
	}
}

func TestShapesInitVramSurfacesTheLibraryStatus(t *testing.T) {
	t.Parallel()
	index, err := ShapesInitVram(DefaultShapesOptions(), nil)
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
func TestShapesCursorsWalkEveryEntryOnce(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildShapes(t, executor, 16, 0, 1)
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
	closed := buildShapes(t, nil, 0, 0, 0)
	if err := closed.Close(); err != nil {
		t.Fatalf("the index must close: %v", err)
	}
	if _, err := closed.Describe(); !errors.Is(err, StatusUninitialized) {
		t.Errorf("a closed index described with %v", err)
	}
}

// Both serialized forms load back holding every entry, and a view serves the same file in place.
func TestShapesSavesAndLoads(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildShapes(t, executor, 32, 0, 2)
	path := filepath.Join(t.TempDir(), "index.safegraphs")
	if err := index.On(executor).Save(path, nil); err != nil {
		t.Fatalf("the index must save: %v", err)
	}
	monolith, err := index.On(executor).SaveBuffer(newArena(t, executor), nil)
	if err != nil {
		t.Fatalf("the index must serialize: %v", err)
	}
	if _, err := ShapesMetadataRead(path, nil); err != nil {
		t.Errorf("the checkpoint's metadata must read: %v", err)
	}
	buffer, err := monolith.Bytes()
	if err != nil {
		t.Fatalf("the monolith must read: %v", err)
	}
	if _, err := ShapesMetadataReadBuffer(buffer, nil); err != nil {
		t.Errorf("the monolith's metadata must read: %v", err)
	}
	loaded := []func() (*ShapesIndex, error){
		func() (*ShapesIndex, error) { return ShapesLoadRam(path, nil) },
		func() (*ShapesIndex, error) { return ShapesLoadRamBuffer(buffer, nil) },
		func() (*ShapesIndex, error) { return ShapesViewRam(path) },
	}
	for position, load := range loaded {
		copied, err := load()
		if position == 2 && errors.Is(err, StatusUnsupported) {
			continue // a residency that serves no view in place says so
		}
		if err != nil {
			t.Fatalf("form %d must load: %v", position, err)
		}
		if size := sizeShapes(t, copied); size != 32 {
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
func TestShapesRefineReportsARoundOverEveryEntry(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildShapes(t, executor, 64, 0, 0)
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
func TestShapesRefinePlanResumesAndFinishes(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildShapes(t, executor, 64, 0, 0)
	on := index.On(executor)
	var plan ShapesRefinePlan
	stops := func(uint, uint) bool { return false }
	if _, err := on.Refine(&ShapesRefineConfig{Plan: &plan, Progress: stops}); err != nil && !errors.Is(err, StatusCancelled) {
		t.Fatalf("the stopped round reported %v", err)
	}
	if plan.Progress.Phase == PhaseDone || plan.Progress.Completed > plan.Progress.Total {
		t.Errorf("a stopped round left %+v", plan.Progress)
	}
	if _, err := on.Refine(&ShapesRefineConfig{Plan: &plan}); err != nil || plan.Progress.Phase != PhaseDone {
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
func TestShapesRefinePlanNeedsAnExecutor(t *testing.T) {
	t.Parallel()
	index := buildShapes(t, nil, 0, 0, 0)
	if _, err := index.Refine(&ShapesRefineConfig{Plan: new(ShapesRefinePlan)}); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a plan on the calling goroutine opened with %v", err)
	}
}

// Several sources into one target gather every entry there, and the other sources end empty
// and live.
func TestShapesReshardGathersEverySourceIntoTheTarget(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	first := buildShapes(t, executor, 32, 0, 0)
	donors := []*ShapesIndex{buildShapes(t, executor, 32, 32, 1), buildShapes(t, executor, 32, 64, 2)}
	if _, err := first.On(executor).Reshard(append([]*ShapesIndex{first}, donors...), []*ShapesIndex{first}, nil); err != nil {
		t.Fatalf("the gather must run: %v", err)
	}
	if size := sizeShapes(t, first); size != 96 {
		t.Errorf("the target holds %d of 96 entries", size)
	}
	for _, key := range []uint64{40, 80} {
		if held, err := first.Contains(KeyFromUint64(key)); err != nil || !held {
			t.Errorf("the target lacks key %d: %v", key, err)
		}
	}
	for position, donor := range donors {
		if size := sizeShapes(t, donor); size != 0 {
			t.Errorf("donor %d kept %d entries", position, size)
		}
	}
}

// One source into several targets cuts it between them, and the source stays live but empty.
func TestShapesReshardCutsOneSourceIntoSeveralTargets(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	source := buildShapes(t, executor, 64, 0, 0)
	targets := []*ShapesIndex{buildShapes(t, nil, 0, 0, 0), buildShapes(t, nil, 0, 0, 0)}
	if _, err := source.On(executor).Reshard([]*ShapesIndex{source}, targets, &ShapesReshardConfig{Seed: 7}); err != nil {
		t.Fatalf("the cut must run: %v", err)
	}
	left, right := sizeShapes(t, targets[0]), sizeShapes(t, targets[1])
	if left+right != 64 || left == 0 || right == 0 || sizeShapes(t, source) != 0 {
		t.Errorf("the cut left %d and %d, and %d in the source", left, right, sizeShapes(t, source))
	}
}

// A cut a stop interrupts keeps every entry moved, the next call naming the plan ends its rounds,
// and a deferred receiver closes it.
func TestShapesReshardCutRunsThroughAPlanAndADeferredFinish(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	source := buildShapes(t, executor, 64, 0, 0)
	targets := []*ShapesIndex{buildShapes(t, nil, 0, 0, 0), buildShapes(t, nil, 0, 0, 0)}
	sources := []*ShapesIndex{source}
	var plan ShapesReshardPlan
	stops := func(uint, uint) bool { return false }
	config := &ShapesReshardConfig{Plan: &plan, Progress: stops}
	if _, err := source.On(executor).Reshard(sources, targets, config); err != nil && !errors.Is(err, StatusCancelled) {
		t.Fatalf("the stopped cut reported %v", err)
	}
	if moved := sizeShapes(t, targets[0]) + sizeShapes(t, targets[1]); moved != 64 || plan.Progress.Phase == PhaseDone {
		t.Errorf("a stop moved %d of 64 entries and left %+v", moved, plan.Progress)
	}
	if _, err := source.On(executor).Reshard(sources, targets, &ShapesReshardConfig{Plan: &plan}); err != nil ||
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
func TestShapesReshardRecutOfOneHandleIsUnsupported(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildShapes(t, executor, 16, 0, 0)
	only := []*ShapesIndex{index}
	if _, err := index.On(executor).Reshard(only, only, nil); !errors.Is(err, StatusUnsupported) {
		t.Errorf("a recut of one handle reported %v", err)
	}
	if size := sizeShapes(t, index); size != 16 {
		t.Errorf("the refusal left %d of 16 entries", size)
	}
}

// Empty or repeated lists are malformed and a target holding foreign entries mismatches; every
// refusal leaves the entries where they were.
func TestShapesReshardRefusesMalformedListsAndMovesNothing(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	first, second := buildShapes(t, executor, 8, 0, 0), buildShapes(t, executor, 8, 8, 1)
	on := first.On(executor)
	if _, err := on.Reshard(nil, []*ShapesIndex{first}, nil); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("no sources reported %v", err)
	}
	if _, err := on.Reshard([]*ShapesIndex{first, first}, []*ShapesIndex{first}, nil); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a repeated source reported %v", err)
	}
	if _, err := on.Reshard([]*ShapesIndex{first}, []*ShapesIndex{second}, nil); !errors.Is(err, StatusConfigMismatch) {
		t.Errorf("a foreign target reported %v", err)
	}
	if sizeShapes(t, first) != 8 || sizeShapes(t, second) != 8 {
		t.Errorf("a refusal moved entries: %d and %d", sizeShapes(t, first), sizeShapes(t, second))
	}
}

// endregion: Lifecycle
