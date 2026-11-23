// Tests for the geospatial modality.
//
// File: golang/geospatial_index_test.go
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

// randomObservations draws count positions between the 60th parallels.
func randomObservations(count uint, seed int64) []GeospatialObservation {
	generator := rand.New(rand.NewSource(seed))
	observations := make([]GeospatialObservation, count)
	for position := range observations {
		observations[position] = GeospatialAt(generator.Float64()*120-60, generator.Float64()*360-180)
	}
	return observations
}

// endregion: Helpers

// region: Entries

// An observation searched for comes back first at zero chord metres, Within reports only the
// entries inside its radius, and Get hands back the position.
func TestGeospatialSearchFindsTheInsertedEntry(t *testing.T) {
	t.Parallel()
	observations := []GeospatialObservation{GeospatialAt(48.8566, 2.3522), GeospatialAt(48.8656, 2.3522),
		GeospatialAt(52.52, 13.405)}
	index, err := GeospatialInitRam(GeospatialOptions{}, nil)
	if err != nil {
		t.Fatalf("the index must initialize: %v", err)
	}
	defer index.Close()
	if err := index.Insert(keysFrom(1, 3), observations, nil); err != nil {
		t.Fatalf("the batch must insert: %v", err)
	}
	arena := newArena(t, nil)
	matches, err := index.Search(observations[2:], GeospatialRelationNear, 1, arena, nil)
	if err != nil {
		t.Fatalf("the search must run: %v", err)
	}
	if keys, distances := matches.Query(0); len(keys) != 1 || keys[0] != KeyFromUint64(3) || distances[0] != 0 {
		t.Errorf("the search found %v at %v", keys, distances)
	}
	within, err := index.Within(observations[:1], GeospatialRelationNear, 2000, arena, nil)
	if err != nil {
		t.Fatalf("the containment must run: %v", err)
	}
	if keys, _ := within.Query(0); len(keys) != 2 || within.Written() != 2 {
		t.Errorf("the radius held %v of %d", keys, within.Written())
	}
	if _, err := index.Within(observations[:1], GeospatialRelationVisible, 2000, arena, nil); !errors.Is(err, StatusUnsupported) {
		t.Errorf("a visible relation without terrain reported %v", err)
	}
	stored, err := index.Get(KeyFromUint64(3), arena)
	if err != nil || stored.Written() != 1 || !nearly(stored.Observations()[0].LatitudeDegrees, 52.52, 1e-6) {
		t.Errorf("the key got %+v: %v", stored.Observations(), err)
	}
	if stats, err := index.Count(KeyFromUint64(3)); err != nil || stats.Entries != 1 {
		t.Errorf("the key counts %+v: %v", stats, err)
	}
}

// endregion: Entries

// region: Lifecycle

// buildGeospatial fills a fresh host index with count random entries under consecutive keys from
// first.
func buildGeospatial(t *testing.T, executor *Executor, count, first uint, seed int64) *GeospatialIndex {
	t.Helper()
	index, err := GeospatialInitRam(GeospatialOptions{}, nil)
	if err != nil {
		t.Fatalf("the index must initialize: %v", err)
	}
	t.Cleanup(func() {
		if err := index.Close(); err != nil {
			t.Errorf("the index must close: %v", err)
		}
	})
	if count != 0 {
		if err := index.On(executor).Insert(keysFrom(first, count), randomObservations(count, seed), nil); err != nil {
			t.Fatalf("the batch must insert: %v", err)
		}
	}
	return index
}

// sizeGeospatial reports how many entries an index describes.
func sizeGeospatial(t *testing.T, index *GeospatialIndex) uint {
	t.Helper()
	description, err := index.Describe()
	if err != nil {
		t.Fatalf("the index must describe: %v", err)
	}
	return description.Size
}

func TestGeospatialInitRamStartsEmpty(t *testing.T) {
	t.Parallel()
	index := buildGeospatial(t, nil, 0, 0, 0)
	description, err := index.Describe()
	if err != nil || description.Size != 0 || description.CPUClass == CPUClassUnknown ||
		description.GPUClass != GPUClassUnknown {
		t.Errorf("a fresh host index describes %+v: %v", description, err)
	}
}

func TestGeospatialInitVramSurfacesTheLibraryStatus(t *testing.T) {
	t.Parallel()
	index, err := GeospatialInitVram(GeospatialOptions{}, nil)
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
func TestGeospatialCursorsWalkEveryEntryOnce(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildGeospatial(t, executor, 16, 0, 1)
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
		for _, key := range batch.Keys() {
			seen[key] = true
		}
	}
	if len(seen) != 16 {
		t.Errorf("the walk saw %d of 16 keys", len(seen))
	}
	closed := buildGeospatial(t, nil, 0, 0, 0)
	if err := closed.Close(); err != nil {
		t.Fatalf("the index must close: %v", err)
	}
	if _, err := closed.Describe(); !errors.Is(err, StatusUninitialized) {
		t.Errorf("a closed index described with %v", err)
	}
}

// Both serialized forms load back holding every entry, and a view serves the same file in place.
func TestGeospatialSavesAndLoads(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildGeospatial(t, executor, 32, 0, 2)
	path := filepath.Join(t.TempDir(), "index.safegraphs")
	if err := index.On(executor).Save(path, nil); err != nil {
		t.Fatalf("the index must save: %v", err)
	}
	monolith, err := index.On(executor).SaveBuffer(newArena(t, executor), nil)
	if err != nil {
		t.Fatalf("the index must serialize: %v", err)
	}
	if _, err := GeospatialMetadataRead(path, nil); err != nil {
		t.Errorf("the checkpoint's metadata must read: %v", err)
	}
	if _, err := GeospatialMetadataReadBuffer(monolith.Bytes(), nil); err != nil {
		t.Errorf("the monolith's metadata must read: %v", err)
	}
	loaded := []func() (*GeospatialIndex, error){
		func() (*GeospatialIndex, error) { return GeospatialLoadRam(path, nil) },
		func() (*GeospatialIndex, error) { return GeospatialLoadRamBuffer(monolith.Bytes(), nil) },
		func() (*GeospatialIndex, error) { return GeospatialViewRam(path) },
	}
	for position, load := range loaded {
		copied, err := load()
		if position == 2 && errors.Is(err, StatusUnsupported) {
			continue // a residency that serves no view in place says so
		}
		if err != nil {
			t.Fatalf("form %d must load: %v", position, err)
		}
		if size := sizeGeospatial(t, copied); size != 32 {
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
func TestGeospatialRefineReportsARoundOverEveryEntry(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildGeospatial(t, executor, 64, 0, 0)
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
func TestGeospatialRefinePlanResumesAndFinishes(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildGeospatial(t, executor, 64, 0, 0)
	on := index.On(executor)
	var plan GeospatialRefinePlan
	stops := func(uint, uint) bool { return false }
	if _, err := on.Refine(&GeospatialRefineConfig{Plan: &plan, Progress: stops}); err != nil && !errors.Is(err, StatusCancelled) {
		t.Fatalf("the stopped round reported %v", err)
	}
	if plan.Phase.Phase == PhaseDone || plan.Phase.Completed > plan.Phase.Total {
		t.Errorf("a stopped round left %+v", plan.Phase)
	}
	if _, err := on.Refine(&GeospatialRefineConfig{Plan: &plan}); err != nil || plan.Phase.Phase != PhaseDone {
		t.Fatalf("the resumed round ended in %+v: %v", plan.Phase, err)
	}
	if report, err := on.RefinePlanFinish(&plan); err != nil || report.Reachability.Audited != 64 {
		t.Errorf("the finished plan reported %+v: %v", report, err)
	}
	if _, err := on.RefinePlanFinish(&plan); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a finished plan finished again with %v", err)
	}
}

// A plan lives across calls on one executor, so a call on the calling goroutine cannot open one.
func TestGeospatialRefinePlanNeedsAnExecutor(t *testing.T) {
	t.Parallel()
	index := buildGeospatial(t, nil, 0, 0, 0)
	if _, err := index.Refine(&GeospatialRefineConfig{Plan: new(GeospatialRefinePlan)}); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a plan on the calling goroutine opened with %v", err)
	}
}

// Several sources into one target gather every entry there, and the other sources end empty
// and live.
func TestGeospatialReshardGathersEverySourceIntoTheTarget(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	first := buildGeospatial(t, executor, 32, 0, 0)
	donors := []*GeospatialIndex{buildGeospatial(t, executor, 32, 32, 1), buildGeospatial(t, executor, 32, 64, 2)}
	if _, err := first.On(executor).Reshard(append([]*GeospatialIndex{first}, donors...), []*GeospatialIndex{first}, nil); err != nil {
		t.Fatalf("the gather must run: %v", err)
	}
	if size := sizeGeospatial(t, first); size != 96 {
		t.Errorf("the target holds %d of 96 entries", size)
	}
	for _, key := range []uint64{40, 80} {
		if held, err := first.Contains(KeyFromUint64(key)); err != nil || !held {
			t.Errorf("the target lacks key %d: %v", key, err)
		}
	}
	for position, donor := range donors {
		if size := sizeGeospatial(t, donor); size != 0 {
			t.Errorf("donor %d kept %d entries", position, size)
		}
	}
}

// One source into several targets cuts it between them, and the source stays live but empty.
func TestGeospatialReshardCutsOneSourceIntoSeveralTargets(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	source := buildGeospatial(t, executor, 64, 0, 0)
	targets := []*GeospatialIndex{buildGeospatial(t, nil, 0, 0, 0), buildGeospatial(t, nil, 0, 0, 0)}
	if _, err := source.On(executor).Reshard([]*GeospatialIndex{source}, targets, &GeospatialReshardConfig{Seed: 7}); err != nil {
		t.Fatalf("the cut must run: %v", err)
	}
	left, right := sizeGeospatial(t, targets[0]), sizeGeospatial(t, targets[1])
	if left+right != 64 || left == 0 || right == 0 || sizeGeospatial(t, source) != 0 {
		t.Errorf("the cut left %d and %d, and %d in the source", left, right, sizeGeospatial(t, source))
	}
}

// A cut a stop interrupts keeps every entry moved, the next call naming the plan ends its rounds,
// and a deferred receiver closes it.
func TestGeospatialReshardCutRunsThroughAPlanAndADeferredFinish(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	source := buildGeospatial(t, executor, 64, 0, 0)
	targets := []*GeospatialIndex{buildGeospatial(t, nil, 0, 0, 0), buildGeospatial(t, nil, 0, 0, 0)}
	sources := []*GeospatialIndex{source}
	var plan GeospatialReshardPlan
	stops := func(uint, uint) bool { return false }
	config := &GeospatialReshardConfig{Plan: &plan, Progress: stops}
	if _, err := source.On(executor).Reshard(sources, targets, config); err != nil && !errors.Is(err, StatusCancelled) {
		t.Fatalf("the stopped cut reported %v", err)
	}
	if moved := sizeGeospatial(t, targets[0]) + sizeGeospatial(t, targets[1]); moved != 64 || plan.Phase.Phase == PhaseDone {
		t.Errorf("a stop moved %d of 64 entries and left %+v", moved, plan.Phase)
	}
	if _, err := source.On(executor).Reshard(sources, targets, &GeospatialReshardConfig{Plan: &plan}); err != nil ||
		plan.Phase.Phase != PhaseDone {
		t.Fatalf("the continued cut ended in %+v: %v", plan.Phase, err)
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
func TestGeospatialReshardRecutOfOneHandleIsUnsupported(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	index := buildGeospatial(t, executor, 16, 0, 0)
	only := []*GeospatialIndex{index}
	if _, err := index.On(executor).Reshard(only, only, nil); !errors.Is(err, StatusUnsupported) {
		t.Errorf("a recut of one handle reported %v", err)
	}
	if size := sizeGeospatial(t, index); size != 16 {
		t.Errorf("the refusal left %d of 16 entries", size)
	}
}

// Empty or repeated lists are malformed and a target holding foreign entries mismatches; every
// refusal leaves the entries where they were.
func TestGeospatialReshardRefusesMalformedListsAndMovesNothing(t *testing.T) {
	t.Parallel()
	executor := pool(t)
	first, second := buildGeospatial(t, executor, 8, 0, 0), buildGeospatial(t, executor, 8, 8, 1)
	on := first.On(executor)
	if _, err := on.Reshard(nil, []*GeospatialIndex{first}, nil); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("no sources reported %v", err)
	}
	if _, err := on.Reshard([]*GeospatialIndex{first, first}, []*GeospatialIndex{first}, nil); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a repeated source reported %v", err)
	}
	if _, err := on.Reshard([]*GeospatialIndex{first}, []*GeospatialIndex{second}, nil); !errors.Is(err, StatusConfigMismatch) {
		t.Errorf("a foreign target reported %v", err)
	}
	if sizeGeospatial(t, first) != 8 || sizeGeospatial(t, second) != 8 {
		t.Errorf("a refusal moved entries: %d and %d", sizeGeospatial(t, first), sizeGeospatial(t, second))
	}
}

// endregion: Lifecycle
