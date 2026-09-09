// Tests for the boundary, not for the engine behind it: struct layouts across an anonymous
// union, sixteen-byte keys reinterpreted in place, stride arithmetic, and handle ownership.
//
// Run under `go test -race` with `GODEBUG=cgocheck=1`, which covers the memory-safety surface
// no assertion here can reach.

package usearch

import (
	"encoding/json"
	"math"
	"math/rand"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

// region: Helpers

const seed = 42

// seededRows returns deterministic unit-length rows, so inner, angular and euclidean rank
// alike and near-parallel rows stop deciding top-1 by float noise.
func seededRows(rows, dimensions uint) []float32 {
	generator := rand.New(rand.NewSource(seed))
	slab := make([]float32, rows*dimensions)
	for row := uint(0); row < rows; row++ {
		values := slab[row*dimensions : (row+1)*dimensions]
		magnitude := float64(0)
		for position := range values {
			values[position] = float32(generator.NormFloat64())
			magnitude += float64(values[position]) * float64(values[position])
		}
		magnitude = math.Sqrt(magnitude)
		for position := range values {
			values[position] /= float32(magnitude)
		}
	}
	return slab
}

// sequentialKeys names rows one through count.
func sequentialKeys(count uint) []Key {
	keys := make([]Key, count)
	for row := uint(0); row < count; row++ {
		keys[row] = KeyFromUint64(uint64(row) + 1)
	}
	return keys
}

// pool spawns the executor every fanned-out verb runs on; the engine derives none.
func pool(t *testing.T) *Executor {
	t.Helper()
	executor, err := NewRAMExecutor(nil, 2)
	if err != nil {
		t.Fatalf("the executor must spawn: %v", err)
	}
	t.Cleanup(func() {
		if err := executor.Close(); err != nil {
			t.Errorf("the executor must close: %v", err)
		}
	})
	return executor
}

// newIndex builds a reserved index and registers its release.
func newIndex(t *testing.T, options DenseOptions, capacity uint) *DenseIndex {
	t.Helper()
	index, err := NewDenseIndex(&options)
	if err != nil {
		t.Fatalf("the index must initialize: %v", err)
	}
	t.Cleanup(func() {
		if err := index.Close(); err != nil {
			t.Errorf("the index must close: %v", err)
		}
	})
	if err := index.Reserve(capacity); err != nil {
		t.Fatalf("the reservation must succeed: %v", err)
	}
	return index
}

// angularF32 is the geometry most tests want.
func angularF32(dimensions uint) DenseOptions {
	options := DefaultDenseOptions()
	options.Dimensions = dimensions
	options.Metric = Angular
	options.Quantization = F32
	return options
}

// populated fills a fresh index with unit rows named one through count.
func populated(t *testing.T, options DenseOptions, count uint) (*DenseIndex, []Key, []float32) {
	t.Helper()
	index := newIndex(t, options, count)
	keys := sequentialKeys(count)
	slab := seededRows(count, options.Dimensions)
	added, err := index.Add(pool(t), keys, RowsOf(slab, options.Dimensions), InsertionOptions{}, nil)
	if err != nil {
		t.Fatalf("the batch must insert: %v", err)
	}
	if added != count {
		t.Fatalf("the batch inserted %d of %d rows", added, count)
	}
	return index, keys, slab
}

// searchOne runs one query and reports how many matches came back.
func searchOne(t *testing.T, index *DenseIndex, query []float32, dimensions uint) uint {
	t.Helper()
	outputs := PackedMatchesSpan(1, 1)
	if err := index.Search(pool(t), RowsOf(query, dimensions), &outputs, SearchOptions{}, nil); err != nil {
		t.Fatalf("the search must run: %v", err)
	}
	keys, _ := outputs.MatchesOf(0)
	return uint(len(keys))
}

func nearly(left, right, tolerance float64) bool { return math.Abs(left-right) <= tolerance }

// endregion: Helpers

// region: Keys

func TestKeysRenderAndParse(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		key      Key
		rendered string
	}{
		{"nil", NilKey, "00000000-0000-0000-0000-000000000000"},
		{"integer", KeyFromUint64(42), "00000000-0000-0000-0000-00000000002a"},
		{"half word", KeyFromUint64(1 << 32), "00000000-0000-0000-0000-000100000000"},
		{"widest integer", KeyFromUint64(^uint64(0)), "00000000-0000-0000-ffff-ffffffffffff"},
		{
			"every byte",
			Key{0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
			"00112233-4455-6677-8899-aabbccddeeff",
		},
	}

	for _, single := range cases {
		t.Run(single.name, func(t *testing.T) {
			t.Parallel()
			if rendered := single.key.String(); rendered != single.rendered {
				t.Fatalf("rendered as %q, wanted %q", rendered, single.rendered)
			}
			if parsed, err := ParseKey(single.rendered); err != nil || parsed != single.key {
				t.Fatalf("%q parsed as %v: %v", single.rendered, parsed, err)
			}
		})
	}
}

// Dashes are cosmetic separators rather than field boundaries, and either case reads the same.
func TestKeysParseLeniently(t *testing.T) {
	t.Parallel()
	for _, text := range []string{
		"0000000000000000000000000000002a",
		"00000000-0000-0000-0000-00000000002A",
		"0-0-00000000000000000000000000002a",
	} {
		t.Run(text, func(t *testing.T) {
			t.Parallel()
			if parsed, err := ParseKey(text); err != nil || parsed != KeyFromUint64(42) {
				t.Fatalf("parsed as %v: %v", parsed, err)
			}
		})
	}
}

func TestKeysRejectMalformedRenderings(t *testing.T) {
	t.Parallel()
	cases := []struct{ name, text string }{
		{"empty", ""},
		{"far too short", "abc"},
		{"one digit short", "00000000-0000-0000-0000-00000000002"},
		{"one digit long", "00000000-0000-0000-0000-00000000002ab"},
		{"not hexadecimal", "00000000-0000-0000-0000-00000000002g"},
	}

	for _, single := range cases {
		t.Run(single.name, func(t *testing.T) {
			t.Parallel()
			if _, err := ParseKey(single.text); StatusOf(err) != MalformedInput {
				t.Errorf("reported %v", StatusOf(err))
			}
		})
	}
}

func TestKeysCarryTheirIntegerIdentity(t *testing.T) {
	t.Parallel()
	for _, value := range []uint64{0, 42, 1 << 32, ^uint64(0)} {
		recovered, integral := KeyFromUint64(value).Uint64()
		if !integral || recovered != value {
			t.Errorf("%d recovered as %d, integral %v", value, recovered, integral)
		}
	}

	var wide Key
	wide[0] = 1
	if _, integral := wide.Uint64(); integral {
		t.Error("a key with high bytes set must report no integer identity")
	}
}

// The text interfaces are what carries a key through a JSON document.
func TestKeysCrossJSONAsTheirRendering(t *testing.T) {
	t.Parallel()
	type record struct {
		Key Key `json:"key"`
	}
	encoded, err := json.Marshal(record{Key: KeyFromUint64(42)})
	if err != nil {
		t.Fatalf("the key must marshal: %v", err)
	}
	if string(encoded) != `{"key":"00000000-0000-0000-0000-00000000002a"}` {
		t.Fatalf("the key marshalled as %s", encoded)
	}
	var decoded record
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("the key must unmarshal: %v", err)
	}
	if decoded.Key != KeyFromUint64(42) {
		t.Fatalf("%s decoded as %v", encoded, decoded.Key)
	}
	if err := json.Unmarshal([]byte(`{"key":"not-a-key"}`), &decoded); StatusOf(err) != MalformedInput {
		t.Errorf("a malformed document reported %v", StatusOf(err))
	}
}

// endregion: Keys

// region: Matrix Views

func TestMatrixViewsDeriveTheirShape(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name       string
		matrix     Matrix
		rows       uint
		dimensions uint
		stride     uint
		kind       ScalarKind
	}{
		{"f32", RowsOf(make([]float32, 6*4), 4), 6, 4, 16, F32},
		{"f64", RowsOf(make([]float64, 6*4), 4), 6, 4, 32, F64},
		{"i8", RowsOf(make([]int8, 6*4), 4), 6, 4, 4, I8},
		{"bf16", RowsOf(make([]BFloat16, 6*4), 4), 6, 4, 8, BF16},
		// Sub-byte kinds count logical dimensions: sixteen bits are two bytes per row.
		{"u1x8", RowsOf(make([]U1x8Scalar, 8), 16), 4, 16, 2, U1x8},
		{"u4x2", RowsOf(make([]U4x2Scalar, 8), 4), 4, 4, 2, U4x2},
		// A pitch wider than the row - the padded layout a mapped dataset arrives in.
		{"padded", StridedRowsOf(make([]float32, 3*8), 3, 4, 8*4), 3, 4, 32, F32},
	}

	for _, single := range cases {
		t.Run(single.name, func(t *testing.T) {
			t.Parallel()
			if single.matrix.Rows() != single.rows {
				t.Errorf("counted %d rows, wanted %d", single.matrix.Rows(), single.rows)
			}
			if single.matrix.Dimensions() != single.dimensions {
				t.Errorf("reported %d dimensions, wanted %d", single.matrix.Dimensions(), single.dimensions)
			}
			if single.matrix.Stride() != single.stride {
				t.Errorf("reported a stride of %d, wanted %d", single.matrix.Stride(), single.stride)
			}
			if single.matrix.Kind() != single.kind {
				t.Errorf("reported the %v kind, wanted %v", single.matrix.Kind(), single.kind)
			}
		})
	}
}

func TestMatrixViewsRejectBadShapes(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		build func()
	}{
		{"zero width", func() { RowsOf(make([]float32, 4), 0) }},
		{"partial row", func() { RowsOf(make([]float32, 5), 4) }},
		{"split value", func() { RowsOf(make([]U1x8Scalar, 4), 4) }},
		{"stride under row", func() { StridedRowsOf(make([]float32, 8), 2, 4, 8) }},
		{"rows overrun", func() { StridedRowsOf(make([]float32, 8), 4, 4, 16) }},
	}

	for _, single := range cases {
		t.Run(single.name, func(t *testing.T) {
			t.Parallel()
			defer func() {
				if recover() == nil {
					t.Fatal("a caller's arithmetic mistake must panic")
				}
			}()
			single.build()
		})
	}
}

// endregion: Matrix Views

// region: Results

// A count above the room reserved for it is the library's way of reporting truncation, so the
// borrow is trimmed to what was actually written rather than to what was found.
func TestOutputsTrimToTheRoomTheyReserved(t *testing.T) {
	t.Parallel()
	const wanted = 2
	counts := []uint{wanted + 3, 1, 0}
	outputs, err := MatchesSpanOf(make([]Key, len(counts)*wanted), make([]Distance, len(counts)*wanted), counts)
	if err != nil {
		t.Fatalf("the slabs must adopt: %v", err)
	}
	if outputs.Queries() != uint(len(counts)) || outputs.Wanted() != wanted {
		t.Fatalf("the outputs describe %d queries of %d", outputs.Queries(), outputs.Wanted())
	}

	for query, count := range counts {
		keys, distances := outputs.MatchesOf(uint(query))
		if expected := min(count, wanted); uint(len(keys)) != expected || uint(len(distances)) != expected {
			t.Errorf("query %d reporting %d borrowed %d matches", query, count, len(keys))
		}
	}
	if keys, distances := outputs.MatchesOf(uint(len(counts))); keys != nil || distances != nil {
		t.Error("a query past the end must borrow nothing")
	}
}

// endregion: Results

// region: Options Round Trip

// The residency arm lives in an anonymous union that cgo cannot address, so it crosses through
// the accessors in shim.c. This is the only thing exercising them.
func TestOptionsSurviveTheRawRoundTrip(t *testing.T) {
	t.Parallel()
	pinned := DenseOptions{
		Dimensions:   768,
		Metric:       Euclidean,
		Quantization: I8,
		Connectivity: 32,
		ValuesPerKey: 4,
		Residency:    RAMResidency{MemoryDomains: []int32{0, 1}},
	}

	options := []DenseOptions{pinned, DefaultDenseOptions()}
	// Ordinal zero is the trap: an empty list derives, a list holding zero names device zero.
	for _, devices := range [][]int32{nil, {0}, {0, 3}} {
		single := angularF32(64)
		single.Residency = VRAMResidency{Devices: devices}
		options = append(options, single)
	}
	for _, paths := range [][]string{nil, {"/dev/nvme0n1"}, {"/dev/nvme0n1", "/dev/nvme1n1"}} {
		single := angularF32(64)
		single.Residency = NVMeResidency{Paths: paths}
		options = append(options, single)
	}

	// A checkpoint records where it was written, never where a reader must keep it, so only
	// the tag survives the trip - the placement comes back empty for the caller to fill.
	for _, single := range options {
		lifted, err := single.reload()
		if err != nil {
			t.Fatalf("%+v must survive the round trip: %v", single.Residency, err)
		}
		if lifted.Residency.String() != single.Residency.String() {
			t.Errorf("the residency came back as %v, not %v", lifted.Residency, single.Residency)
		}
		if lifted.Dimensions != single.Dimensions || lifted.Metric != single.Metric ||
			lifted.Quantization != single.Quantization || lifted.Connectivity != single.Connectivity {
			t.Errorf("%+v came back as %+v", single, lifted)
		}
	}
}

func TestOptionsRejectUnknownTags(t *testing.T) {
	t.Parallel()
	// A fourth residency is unrepresentable - the interface is sealed by an unexported method,
	// so the only bad value left is the absent one, and that is caught on the way down.
	unknown := angularF32(64)
	unknown.Residency = nil
	var pinner runtime.Pinner
	defer pinner.Unpin()
	if _, err := unknown.toRaw(&pinner); StatusOf(err) != MalformedInput {
		t.Errorf("a missing residency must be named, not guessed: %v", StatusOf(err))
	}

	// A tag a newer library wrote is caught on the way up, never passed through. The residency
	// tags are the C interface's, not this package's, since Go names a residency by its type.
	const ramTag = 0
	for _, single := range []struct {
		name                              string
		metric, quantization, residencyOf int32
	}{
		{"metric", 99, int32(F32), ramTag},
		{"quantization", int32(Angular), 77, ramTag},
		{"residency", int32(Angular), int32(F32), 9},
	} {
		t.Run(single.name, func(t *testing.T) {
			t.Parallel()
			_, err := angularF32(64).reloadTagged(single.metric, single.quantization, single.residencyOf)
			if StatusOf(err) != UnknownKind {
				t.Errorf("an unknown %s must not be guessed: %v", single.name, StatusOf(err))
			}
		})
	}
}

// endregion: Options Round Trip

// region: Lifecycle

func TestFreeingTwiceIsHarmless(t *testing.T) {
	t.Parallel()
	options := angularF32(16)
	index, err := NewDenseIndex(&options)
	if err != nil {
		t.Fatalf("the index must initialize: %v", err)
	}

	if err := index.Free(); err != nil {
		t.Fatalf("the first release must succeed: %v", err)
	}
	if err := index.Free(); err != nil {
		t.Fatalf("the second release must be a no-op: %v", err)
	}
	if _, err := index.Size(); StatusOf(err) != Uninitialized {
		t.Errorf("a released handle reported %v", StatusOf(err))
	}

	var absent *DenseIndex
	if _, err := absent.Size(); StatusOf(err) != Uninitialized {
		t.Errorf("a nil handle reported %v", StatusOf(err))
	}
}

// One factory per residency, each reading only what its own hardware has - the VRAM one drives a
// stream per device from the calling thread and takes no thread count.
//
// Whether a build carries the accelerator and block backends decides the outcome, so what is
// asserted is the invariant either way: a factory hands back an executor or a failure, never both
// and never neither.
func TestExecutorFactoriesTakeOnlyWhatTheirResidencyHas(t *testing.T) {
	t.Parallel()
	memory, err := NewRAMExecutor(nil, 2)
	if err != nil {
		t.Fatalf("the RAM executor must spawn: %v", err)
	}
	defer memory.Close()

	devices, err := NewVRAMExecutor(nil)
	if (devices == nil) == (err == nil) {
		t.Errorf("the VRAM factory answered with %v and %v", devices, err)
	}
	if devices != nil {
		devices.Close()
	}

	drives, err := NewNVMeExecutor(nil, 2)
	if (drives == nil) == (err == nil) {
		t.Errorf("the NVMe factory answered with %v and %v", drives, err)
	}
	if drives != nil {
		drives.Close()
	}
}

// An executor outlives no verb that names it, so every door refuses a freed one rather than
// handing the library a dangling handle.
func TestExecutorRefusesUseAfterFree(t *testing.T) {
	t.Parallel()
	index := newIndex(t, angularF32(4), 4)
	executor, err := NewRAMExecutor(nil, 2)
	if err != nil {
		t.Fatalf("the executor must spawn: %v", err)
	}
	if err := executor.Drain(); err != nil {
		t.Fatalf("a live executor must drain: %v", err)
	}

	if err := executor.Free(); err != nil {
		t.Fatalf("the first release must succeed: %v", err)
	}
	if err := executor.Free(); err != nil {
		t.Fatalf("the second release must be a no-op: %v", err)
	}
	if err := executor.Drain(); StatusOf(err) != Uninitialized {
		t.Errorf("a released executor drained with %v", StatusOf(err))
	}

	slab := seededRows(1, 4)
	if _, err := index.Add(executor, sequentialKeys(1), RowsOf(slab, 4), InsertionOptions{}, nil); StatusOf(err) != MalformedInput {
		t.Errorf("a released executor inserted with %v", StatusOf(err))
	}
	var absent *Executor
	if _, err := index.Add(absent, sequentialKeys(1), RowsOf(slab, 4), InsertionOptions{}, nil); StatusOf(err) != MalformedInput {
		t.Errorf("a nil executor inserted with %v", StatusOf(err))
	}
}

// endregion: Lifecycle

// region: Ingest & Retrieval

// The bytes that come back are what proves every stride, scalar kind, and base pointer right;
// a count alone would pass with all three wrong.
func TestAddsAndGetsVectorsIntact(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	index, keys, slab := populated(t, angularF32(dimensions), 4)

	restored := make([]float32, dimensions)
	entries, err := index.Get(keys[1], RowsOf(restored, dimensions))
	if err != nil {
		t.Fatalf("the row must come back: %v", err)
	}
	if entries != 1 {
		t.Fatalf("the key holds %d entries", entries)
	}
	for position, value := range restored {
		if !nearly(float64(value), float64(slab[dimensions+position]), 1e-6) {
			t.Fatalf("dimension %d came back as %v, not %v", position, value, slab[dimensions+position])
		}
	}

	// A padded pitch is the layout a mapped dataset arrives in, and it must cross both ways,
	// leaving the padding of the row it writes through untouched.
	const pitch = 16
	padded := make([]float32, pitch)
	copy(padded, slab[2*dimensions:3*dimensions])
	if _, err := index.Add(pool(t), []Key{KeyFromUint64(500)},
		StridedRowsOf(padded, 1, dimensions, pitch*4), InsertionOptions{}, nil); err != nil {
		t.Fatalf("a padded row must insert: %v", err)
	}
	widened := make([]float32, pitch)
	for position := range widened {
		widened[position] = -1
	}
	if _, err := index.Get(KeyFromUint64(500), StridedRowsOf(widened, 1, dimensions, pitch*4)); err != nil {
		t.Fatalf("a padded row must come back: %v", err)
	}
	for position := 0; position < dimensions; position++ {
		if !nearly(float64(widened[position]), float64(padded[position]), 1e-6) {
			t.Fatalf("padded dimension %d came back as %v", position, widened[position])
		}
	}
	for position := dimensions; position < pitch; position++ {
		if widened[position] != -1 {
			t.Fatalf("the padding at %d was overwritten with %v", position, widened[position])
		}
	}
}

// The truncation contract: Get reports what exists, never what it wrote, so a caller can size
// from the answer and call again.
func TestReportsTheTrueExtentWhenTruncated(t *testing.T) {
	t.Parallel()
	const dimensions = 4
	index := newIndex(t, angularF32(dimensions), 16)
	shared := KeyFromUint64(7)
	slab := seededRows(3, dimensions)

	if _, err := index.Add(pool(t), []Key{shared, shared, shared}, RowsOf(slab, dimensions),
		InsertionOptions{}, nil); err != nil {
		t.Fatalf("three rows may share one key: %v", err)
	}
	if counted, err := index.Count(shared); err != nil || counted != 3 {
		t.Fatalf("the probe counted %d entries: %v", counted, err)
	}
	if held, err := index.Contains(shared); err != nil || !held {
		t.Fatalf("the membership probe reported %v: %v", held, err)
	}
	if held, err := index.Contains(KeyFromUint64(8)); err != nil || held {
		t.Fatalf("an absent key reported %v: %v", held, err)
	}

	cramped := make([]float32, dimensions)
	if entries, err := index.Get(shared, RowsOf(cramped, dimensions)); err != nil || entries != 3 {
		t.Fatalf("a one-row slab reported %d entries: %v", entries, err)
	}

	roomy := make([]float32, 3*dimensions)
	if entries, err := index.Get(shared, RowsOf(roomy, dimensions)); err != nil || entries != 3 {
		t.Fatalf("sized from that answer, the copy reported %d entries: %v", entries, err)
	}
	for position, value := range roomy {
		if !nearly(float64(value), float64(slab[position]), 1e-6) {
			t.Fatalf("value %d came back as %v", position, value)
		}
	}
}

// endregion: Ingest & Retrieval

// region: Shape Rejection

// Every case here is a guard this binding raises before anything crosses into C.
func TestRejectsMalformedShapes(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	index, _, slab := populated(t, angularF32(dimensions), 4)
	executor := pool(t)
	queries := RowsOf(slab[:2*dimensions], dimensions)
	narrow := make([]float32, dimensions/2)

	cases := []struct {
		name string
		run  func() error
	}{
		{"more rows than keys", func() error {
			_, err := index.Add(executor, sequentialKeys(1), RowsOf(slab, dimensions), InsertionOptions{}, nil)
			return err
		}},
		{"rows of the wrong width", func() error {
			_, err := index.Add(executor, sequentialKeys(1), RowsOf(narrow, dimensions/2), InsertionOptions{}, nil)
			return err
		}},
		{"queries of the wrong width", func() error {
			outputs := PackedMatchesSpan(1, 4)
			return index.Search(executor, RowsOf(narrow, dimensions/2), &outputs, SearchOptions{}, nil)
		}},
		{"fewer output rows than queries", func() error {
			outputs := PackedMatchesSpan(1, 4)
			return index.Search(executor, queries, &outputs, SearchOptions{}, nil)
		}},
		{"outputs with no room", func() error {
			_, err := MatchesSpanOf(nil, nil, make([]uint, 1))
			return err
		}},
		{"outputs of unequal shape", func() error {
			_, err := MatchesSpanOf(make([]Key, 4), make([]Distance, 8), make([]uint, 2))
			return err
		}},
		{"outputs that do not divide into rows", func() error {
			_, err := MatchesSpanOf(make([]Key, 5), make([]Distance, 5), make([]uint, 2))
			return err
		}},
		{"outputs holding no query at all", func() error {
			_, err := MatchesSpanOf(make([]Key, 4), make([]Distance, 4), nil)
			return err
		}},
		{"queries that do not divide into documents", func() error {
			outputs := PackedMatchesSpan(2, 4)
			return index.MaxSim(executor, queries, 3, &outputs, SearchOptions{}, nil)
		}},
		{"documents of no rows", func() error {
			outputs := PackedMatchesSpan(2, 4)
			return index.MaxSim(executor, queries, 0, &outputs, SearchOptions{}, nil)
		}},
	}

	for _, single := range cases {
		t.Run(single.name, func(t *testing.T) {
			if status := StatusOf(single.run()); status != ShapeMismatch {
				t.Errorf("reported %v", status)
			}
		})
	}

	outputs := PackedMatchesSpan(2, 4)
	if status := StatusOf(index.SearchIf(executor, queries, &outputs, SearchOptions{}, nil, nil)); status != MalformedInput {
		t.Errorf("a nil predicate reported %v", status)
	}
}

// endregion: Shape Rejection

// region: Search

// A batch is where the output row stride earns its keep: every query's slice must land in its
// own row, and the three positional size_t fields of the telemetry must land in theirs.
func TestBatchSearchFillsEveryOutputRow(t *testing.T) {
	t.Parallel()
	const (
		dimensions = 64
		count      = 256
		queries    = 8
		wanted     = 5
	)
	index, _, slab := populated(t, angularF32(dimensions), count)

	outputs := PackedMatchesSpan(queries, wanted)
	var stats DenseStats
	if err := index.Search(pool(t), RowsOf(slab[:queries*dimensions], dimensions), &outputs, SearchOptions{}, &stats); err != nil {
		t.Fatalf("the batch must resolve: %v", err)
	}
	if outputs.Queries() != queries {
		t.Fatalf("the outputs describe %d of %d queries", outputs.Queries(), queries)
	}
	for query := uint(0); query < queries; query++ {
		if found, distances := outputs.MatchesOf(query); uint(len(found)) != wanted || uint(len(distances)) != wanted {
			t.Fatalf("query %d reported %d of %d matches", query, len(found), wanted)
		}
	}

	// Three positional size_t fields, so a swap in the C struct would go unseen otherwise.
	if stats.VisitedMembers == 0 || stats.ComputedDistances == 0 {
		t.Errorf("the traversal reported %+v", stats)
	}
	total := stats
	total.Accumulate(stats)
	if total.VisitedMembers != 2*stats.VisitedMembers || total.ComputedDistances != 2*stats.ComputedDistances ||
		total.Restarts != 2*stats.Restarts {
		t.Errorf("accumulating %+v gave %+v", stats, total)
	}
}

// The predicate crosses as a cgo handle and runs on the library's worker threads - the most
// dangerous construct in the package.
func TestSearchAdmitsOnlyApprovedKeys(t *testing.T) {
	t.Parallel()
	const dimensions = 32
	index, _, slab := populated(t, angularF32(dimensions), 128)
	queries := RowsOf(slab[:3*dimensions], dimensions)

	admitted := PackedMatchesSpan(3, 10)
	even := func(key Key) bool {
		value, integral := key.Uint64()
		return integral && value%2 == 0
	}
	if err := index.SearchIf(pool(t), queries, &admitted, SearchOptions{ExpansionFactor: 64}, nil, even); err != nil {
		t.Fatalf("the filtered batch must resolve: %v", err)
	}
	seen := 0
	for query := uint(0); query < 3; query++ {
		found, _ := admitted.MatchesOf(query)
		for _, key := range found {
			if !even(key) {
				t.Fatalf("query %d admitted %v", query, key)
			}
			seen++
		}
	}
	if seen == 0 {
		t.Fatal("the filter admitted nothing at all")
	}

	// Admitting nothing is a result, not a failure.
	rejected := PackedMatchesSpan(3, 10)
	if err := index.SearchIf(pool(t), queries, &rejected, SearchOptions{}, nil, func(Key) bool { return false }); err != nil {
		t.Fatalf("a rejecting filter must still resolve: %v", err)
	}
	for query := uint(0); query < 3; query++ {
		if found, _ := rejected.MatchesOf(query); len(found) != 0 {
			t.Fatalf("a rejecting filter admitted %d matches", len(found))
		}
	}
}

// endregion: Search

// region: Metrics & Quantization

// The one place the Go enum values are checked against the C ones: a renumbered constant
// reaches the library as a different kind, and is caught here and nowhere else.
func TestEveryMetricAndQuantizationCrosses(t *testing.T) {
	t.Parallel()
	const dimensions = 64

	scalars := []struct {
		metric       DenseMetric
		quantization ScalarKind
	}{
		{Inner, F32}, {Angular, F64}, {Euclidean, BF16}, {Angular, F16},
		{Euclidean, E5M2}, {Angular, E4M3}, {Inner, I8}, {Angular, U8},
	}
	for _, single := range scalars {
		t.Run(single.metric.String()+"/"+single.quantization.String(), func(t *testing.T) {
			options := angularF32(dimensions)
			options.Metric = single.metric
			options.Quantization = single.quantization
			index, _, slab := populated(t, options, 32)
			if found := searchOne(t, index, slab[:dimensions], dimensions); found != 1 {
				t.Errorf("the walk reported %d matches", found)
			}
		})
	}

	// Bit-level metrics take packed rows, where a row of 64 dimensions is eight bytes.
	for _, metric := range []DenseMetric{Hamming, Jaccard} {
		t.Run(metric.String()+"/u1x8", func(t *testing.T) {
			options := angularF32(dimensions)
			options.Metric = metric
			options.Quantization = U1x8
			index := newIndex(t, options, 32)

			const rows = 16
			generator := rand.New(rand.NewSource(seed))
			packed := make([]U1x8Scalar, rows*dimensions/8)
			for position := range packed {
				packed[position] = U1x8Scalar(generator.Intn(256))
			}
			if _, err := index.Add(pool(t), sequentialKeys(rows), RowsOf(packed, dimensions),
				InsertionOptions{}, nil); err != nil {
				t.Fatalf("packed rows must insert: %v", err)
			}

			outputs := PackedMatchesSpan(1, 4)
			if err := index.Search(pool(t), RowsOf(packed[:dimensions/8], dimensions), &outputs,
				SearchOptions{}, nil); err != nil {
				t.Fatalf("packed rows must search: %v", err)
			}
			if found, _ := outputs.MatchesOf(0); len(found) == 0 {
				t.Error("a packed walk reported no matches")
			}
		})
	}
}

// endregion: Metrics & Quantization

// region: Late Interaction & Refinement

// The outputs are sized over documents rather than rows, which is the one piece of arithmetic
// this door does that the others do not.
func TestMaxSimSizesItsOutputsOverDocuments(t *testing.T) {
	t.Parallel()
	const (
		dimensions = 4
		tokens     = 2
		documents  = 3
		queries    = 2
	)
	options := angularF32(dimensions)
	options.Metric = Inner
	options.ValuesPerKey = tokens
	index := newIndex(t, options, documents*tokens)

	keys := make([]Key, 0, documents*tokens)
	for document := uint(0); document < documents; document++ {
		for token := uint(0); token < tokens; token++ {
			keys = append(keys, KeyFromUint64(uint64(document)))
		}
	}
	slab := seededRows(documents*tokens, dimensions)
	if _, err := index.Add(pool(t), keys, RowsOf(slab, dimensions), InsertionOptions{}, nil); err != nil {
		t.Fatalf("the token rows must insert: %v", err)
	}

	outputs := PackedMatchesSpan(queries, documents)
	var stats DenseStats
	if err := index.MaxSim(pool(t), RowsOf(slab[:queries*tokens*dimensions], dimensions), tokens,
		&outputs, SearchOptions{}, &stats); err != nil {
		t.Fatalf("the late interaction must resolve: %v", err)
	}
	for query := uint(0); query < queries; query++ {
		if found, _ := outputs.MatchesOf(query); len(found) != documents {
			t.Errorf("query %d scored %d of %d documents", query, len(found), documents)
		}
	}
	if stats.ComputedDistances == 0 {
		t.Errorf("the walk reported %+v", stats)
	}
}

// A sweep reports the nodes whose edges changed rather than a status, so a caller loops on the
// number it returns.
func TestRefineReportsItsSweep(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	const count = 32
	index, _, _ := populated(t, angularF32(dimensions), count)

	var stats DenseStats
	changed, err := index.Refine(pool(t), InsertionOptions{ExpansionFactor: 64}, &stats)
	if err != nil {
		t.Fatalf("the sweep must run: %v", err)
	}
	if changed > count {
		t.Errorf("a sweep changed %d of %d nodes", changed, count)
	}
	if stats.VisitedMembers == 0 {
		t.Errorf("the sweep reported %+v", stats)
	}
}

// endregion: Late Interaction & Refinement

// region: Serialization

// Every option field must survive the checkpoint, not just the two a reader usually looks at.
func TestSerializationPreservesEveryOption(t *testing.T) {
	t.Parallel()
	const dimensions = 16
	options := angularF32(dimensions)
	options.Metric = Euclidean
	options.Quantization = I8
	options.Connectivity = 24
	options.ValuesPerKey = 2
	options.Residency = RAMResidency{MemoryDomains: []int32{0}}

	source, _, _ := populated(t, options, 32)
	path := filepath.Join(t.TempDir(), "index.safegraphs")
	if err := source.Save(pool(t), path); err != nil {
		t.Fatalf("the checkpoint must write: %v", err)
	}

	length, err := source.StorageRequirements()
	if err != nil || length == 0 {
		t.Fatalf("the index serializes to %d bytes: %v", length, err)
	}
	buffer := make([]byte, length)
	if err := source.SaveBuffer(pool(t), buffer); err != nil {
		t.Fatalf("the buffer must write: %v", err)
	}

	for _, single := range []struct {
		name     string
		metadata func() (DenseCheckpointMetadata, error)
		restore  func(*DenseIndex) error
	}{
		{"file",
			func() (DenseCheckpointMetadata, error) { return DenseMetadata(path) },
			func(index *DenseIndex) error { return index.Load(pool(t), path) }},
		{"buffer",
			func() (DenseCheckpointMetadata, error) { return DenseMetadataBuffer(buffer) },
			func(index *DenseIndex) error { return index.LoadBuffer(pool(t), buffer) }},
	} {
		t.Run(single.name, func(t *testing.T) {
			metadata, err := single.metadata()
			if err != nil {
				t.Fatalf("the metadata must read without loading: %v", err)
			}
			// The four fields a checkpoint froze, and no more: the placement and ValuesPerKey
			// are the caller's to choose before construction.
			if metadata.Dimensions != dimensions || metadata.Metric != Euclidean ||
				metadata.Quantization != I8 || metadata.Connectivity != 24 {
				t.Errorf("the checkpoint describes %+v", metadata)
			}

			// A zeroed geometry freezes here, from the checkpoint's own headers.
			deferred := options
			deferred.Dimensions = 0
			restored := newIndex(t, deferred, 32)
			if err := single.restore(restored); err != nil {
				t.Fatalf("the checkpoint must load: %v", err)
			}
			if frozen, err := restored.Dimensions(); err != nil || frozen != dimensions {
				t.Fatalf("the load froze %d dimensions: %v", frozen, err)
			}
			if size, err := restored.Size(); err != nil || size != 32 {
				t.Fatalf("the restored index holds %d of 32 entries: %v", size, err)
			}
		})
	}
}

// endregion: Serialization

// region: Concurrency

// This earns its keep under -race and GODEBUG=cgocheck=1, where it checks that Go pointers
// handed to C survive concurrent collection. Unlike Rust, a Go test may report from a
// goroutine, so no counters are needed.
func TestSearchesWhileInserting(t *testing.T) {
	t.Parallel()
	const (
		dimensions = 32
		seeded     = 32
		writers    = 2
		batches    = 5
		batchRows  = 8
	)

	index, _, slab := populated(t, angularF32(dimensions), seeded)
	var writing, reading sync.WaitGroup
	done := make(chan struct{})

	for writer := 0; writer < writers; writer++ {
		writing.Add(1)
		mine := pool(t)
		go func(writer int, mine *Executor) {
			defer writing.Done()
			for batch := 0; batch < batches; batch++ {
				keys := make([]Key, batchRows)
				for row := range keys {
					keys[row] = KeyFromUint64(uint64(1000 + (writer*batches+batch)*batchRows + row))
				}
				rows := seededRows(batchRows, dimensions)
				if added, err := index.Add(mine, keys, RowsOf(rows, dimensions), InsertionOptions{}, nil); err != nil {
					t.Errorf("a concurrent batch failed: %v", err)
					return
				} else if added != batchRows {
					t.Errorf("a concurrent batch published %d of %d rows", added, batchRows)
					return
				}
			}
		}(writer, mine)
	}

	for reader := 0; reader < writers; reader++ {
		reading.Add(1)
		mine := pool(t)
		go func(mine *Executor) {
			defer reading.Done()
			query := RowsOf(slab[:dimensions], dimensions)
			for {
				select {
				case <-done:
					return
				default:
				}
				outputs := PackedMatchesSpan(1, 10)
				if err := index.Search(mine, query, &outputs, SearchOptions{}, nil); err != nil {
					t.Errorf("a concurrent search failed: %v", err)
					return
				}
			}
		}(mine)
	}

	writing.Wait()
	close(done)
	reading.Wait()

	expected := uint(seeded + writers*batches*batchRows)
	if size, err := index.Size(); err != nil || size != expected {
		t.Fatalf("the index holds %d of %d entries: %v", size, expected, err)
	}
}

// endregion: Concurrency
