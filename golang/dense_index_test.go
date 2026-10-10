// Tests for the boundary, not for the engine behind it: struct layouts across an anonymous
// union, sixteen-byte keys reinterpreted in place, stride arithmetic, handle ownership, and the
// pins and handles a deferred call keeps until it is final.
//
// Run under `go test -race` with `GODEBUG=cgocheck=1`, which covers the memory-safety surface
// no assertion here can reach.

package usearch

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
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
	executor, err := ExecutorForRam(nil, 2, 0)
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
	index, err := DenseInitRam(options, nil)
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
	options.Quantization = ScalarF32
	return options
}

// populated fills a fresh index with unit rows named one through count.
func populated(t *testing.T, options DenseOptions, count uint) (*DenseIndex, []Key, []float32) {
	t.Helper()
	index := newIndex(t, options, count)
	keys := sequentialKeys(count)
	slab := seededRows(count, options.Dimensions)
	if err := index.On(pool(t)).Insert(keys, RowsOf(slab, options.Dimensions), nil); err != nil {
		t.Fatalf("the batch must insert: %v", err)
	}
	return index, keys, slab
}

// searchOne runs one query and reports how many matches came back.
func searchOne(t *testing.T, index *DenseIndex, query []float32, dimensions uint) uint {
	t.Helper()
	matches, err := index.On(pool(t)).Search(RowsOf(query, dimensions), 1, nil)
	if err != nil {
		t.Fatalf("the search must run: %v", err)
	}
	keys, _ := matches.Row(0)
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
			if key, err := ParseKey(single.text); !errors.Is(err, StatusMalformed) || key != NilKey {
				t.Errorf("reported %v and %v", err, key)
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
	if err := json.Unmarshal([]byte(`{"key":"not-a-key"}`), &decoded); !errors.Is(err, StatusMalformed) {
		t.Errorf("a malformed document reported %v", err)
	}
}

// endregion: Keys

// region: Enumerations

// The one place the Go enum values are checked against the C name tables: a renumbered constant
// reaches the library as a different kind, and is spelled wrong here and nowhere else.
func TestEnumeratorsSpellTheLibrarysNames(t *testing.T) {
	t.Parallel()
	for _, single := range []struct {
		value    interface{ String() string }
		spelling string
	}{
		{StatusDeferred, "deferred"},
		{StatusShapeMismatch, "shape-mismatch"},
		{StatusInvalidArgument, "invalid-argument"},
		{Status(4242), "unrecognized"},
		{MoodParked, "parked"},
		{MoodSpinning, "spinning"},
		{ScalarBF16, "bf16"},
		{ScalarU1x8, "u1x8"},
		{DenseMetricInner, "inner"},
		{DenseMetricJaccard, "jaccard"},
	} {
		if spelled := single.value.String(); spelled != single.spelling {
			t.Errorf("spelled %q, wanted %q", spelled, single.spelling)
		}
	}

	// A status is the errors.Is target every failure unwraps to.
	failure := error(fail(StatusReadOnly, "a view"))
	if !errors.Is(failure, StatusReadOnly) || errors.Is(failure, StatusIO) {
		t.Errorf("%v matched the wrong status", failure)
	}
}

// endregion: Enumerations

// region: Row Views

func TestMatrixViewsDeriveTheirShape(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name       string
		matrix     RowsView
		rows       uint
		dimensions uint
		stride     uint
		scalar     Scalar
	}{
		{"f32", RowsOf(make([]float32, 6*4), 4), 6, 4, 16, ScalarF32},
		{"f64", RowsOf(make([]float64, 6*4), 4), 6, 4, 32, ScalarF64},
		{"i8", RowsOf(make([]int8, 6*4), 4), 6, 4, 4, ScalarI8},
		{"bf16", RowsOf(make([]BFloat16, 6*4), 4), 6, 4, 8, ScalarBF16},
		// Sub-byte kinds count logical dimensions: sixteen bits are two bytes per row.
		{"u1x8", RowsOf(make([]U1x8Scalar, 8), 16), 4, 16, 2, ScalarU1x8},
		{"u4x2", RowsOf(make([]U4x2Scalar, 8), 4), 4, 4, 2, ScalarU4x2},
		// A pitch wider than the row - the padded layout a mapped dataset arrives in.
		{"padded", StridedRowsOf(make([]float32, 3*8), 3, 4, 8*4), 3, 4, 32, ScalarF32},
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
			if single.matrix.Scalar() != single.scalar {
				t.Errorf("reported the %v kind, wanted %v", single.matrix.Scalar(), single.scalar)
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

// endregion: Row Views

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
		keys, scores := outputs.Row(uint(query))
		if expected := min(count, wanted); uint(len(keys)) != expected || uint(len(scores)) != expected {
			t.Errorf("query %d reporting %d borrowed %d matches", query, count, len(keys))
		}
	}
	if keys, scores := outputs.Row(uint(len(counts))); keys != nil || scores != nil {
		t.Error("a query past the end must borrow nothing")
	}
}

// endregion: Results

// region: Lifecycle

func TestClosingTwiceIsHarmless(t *testing.T) {
	t.Parallel()
	index, err := DenseInitRam(angularF32(16), nil)
	if err != nil {
		t.Fatalf("the index must initialize: %v", err)
	}

	if err := index.Close(); err != nil {
		t.Fatalf("the first release must succeed: %v", err)
	}
	if err := index.Close(); err != nil {
		t.Fatalf("the second release must be a no-op: %v", err)
	}
	if _, err := index.Describe(); !errors.Is(err, StatusUninitialized) {
		t.Errorf("a released handle reported %v", err)
	}
	call := index.Deferred(nil).Describe()
	if call.Status() != StatusUninitialized {
		t.Errorf("a released handle submitted with %v", call.Status())
	}
	if err := call.Close(); !errors.Is(err, StatusUninitialized) {
		t.Errorf("a released handle's call closed with %v", err)
	}

	var absent *DenseIndex
	if err := absent.Close(); err != nil {
		t.Errorf("a nil handle closed with %v", err)
	}
	if _, err := absent.On(nil).Describe(); !errors.Is(err, StatusUninitialized) {
		t.Errorf("a nil handle reported %v", err)
	}
}

// One constructor per residency, each reading only what its own hardware has - the VRAM one drives
// a stream per device from the calling thread and takes no thread count.
//
// Whether a build carries the accelerator and block backends decides the outcome, so what is
// asserted is the invariant either way: a constructor hands back an executor or a failure, never
// both and never neither.
func TestExecutorConstructorsTakeOnlyWhatTheirResidencyHas(t *testing.T) {
	t.Parallel()
	memory, err := ExecutorForRam(nil, 2, 0)
	if err != nil {
		t.Fatalf("the RAM executor must spawn: %v", err)
	}
	defer memory.Close()

	devices, err := ExecutorForVram(nil, nil, 0, 0)
	if (devices == nil) == (err == nil) {
		t.Errorf("the VRAM constructor answered with %v and %v", devices, err)
	}
	devices.Close()

	drives, err := ExecutorForNvme(nil, 2, 0)
	if (drives == nil) == (err == nil) {
		t.Errorf("the NVMe constructor answered with %v and %v", drives, err)
	}
	drives.Close()

	if _, err := ExecutorForVram([]int32{0}, make([]unsafe.Pointer, 2), 0, 0); !errors.Is(err, StatusShapeMismatch) {
		t.Errorf("a stream list of the wrong length reported %v", err)
	}
	if _, err := DenseViewNvme("index.safegraphs", nil); !errors.Is(err, StatusUnsupported) {
		t.Errorf("an NVMe view reported %v", err)
	}
}

// An executor outlives no verb that names it, so every door refuses a closed one rather than
// handing the library a dangling handle, while a nil one runs the call on the calling goroutine.
func TestExecutorRefusesUseAfterClose(t *testing.T) {
	t.Parallel()
	index := newIndex(t, angularF32(4), 4)
	executor, err := ExecutorForRam(nil, 2, 0)
	if err != nil {
		t.Fatalf("the executor must spawn: %v", err)
	}
	if err := executor.Drain(MoodSpinning); err != nil {
		t.Fatalf("a live executor must drain: %v", err)
	}

	if err := executor.Close(); err != nil {
		t.Fatalf("the first release must succeed: %v", err)
	}
	if err := executor.Close(); err != nil {
		t.Fatalf("the second release must be a no-op: %v", err)
	}
	if err := executor.Drain(MoodParked); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a released executor drained with %v", err)
	}

	slab := seededRows(1, 4)
	if err := index.On(executor).Insert(sequentialKeys(1), RowsOf(slab, 4), nil); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a released executor inserted with %v", err)
	}
	if _, err := index.On(executor).Get(KeyFromUint64(1), ScalarF64, nil); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a released executor exported with %v", err)
	}
	if _, err := DenseLoadRam("index.safegraphs", &DenseLoadRamConfig{Executor: executor}); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a released executor loaded with %v", err)
	}
	if err := index.Insert(sequentialKeys(1), RowsOf(slab, 4), nil); err != nil {
		t.Errorf("the index must insert on the calling goroutine: %v", err)
	}
}

// endregion: Lifecycle

// region: Ingest & Retrieval

// The bytes that come back are what proves every stride, scalar kind, and base pointer right;
// a count alone would pass with all three wrong.
func TestInsertsAndGetsVectorsIntact(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	index, keys, slab := populated(t, angularF32(dimensions), 4)

	// A padded pitch is the layout a mapped dataset arrives in; the rows come back packed.
	const pitch = 16
	padded := make([]float32, pitch)
	copy(padded, slab[2*dimensions:3*dimensions])
	if err := index.Insert([]Key{keys[1]}, StridedRowsOf(padded, 1, dimensions, pitch*4), nil); err != nil {
		t.Fatalf("a padded row must insert: %v", err)
	}
	if counted, err := index.Count(keys[1]); err != nil || counted != 2 {
		t.Fatalf("the probe counted %d entries: %v", counted, err)
	}
	if held, err := index.Contains(KeyFromUint64(999)); err != nil || held {
		t.Fatalf("an absent key reported %v: %v", held, err)
	}
	vectors, err := index.On(pool(t)).Get(keys[1], ScalarF32, nil)
	if err != nil {
		t.Fatalf("the rows must come back: %v", err)
	}
	defer vectors.Close()
	if vectors.Rows() != 2 || vectors.Stride() != dimensions*4 || vectors.Columns() != dimensions || vectors.Scalar() != ScalarF32 {
		t.Fatalf("the key exported %d rows of %d at a stride of %d", vectors.Rows(), vectors.Columns(), vectors.Stride())
	}
	restored := DenseValues[float32](vectors)
	expected := append(slices.Clone(slab[dimensions:2*dimensions]), padded[:dimensions]...)
	for position, value := range restored {
		if !nearly(float64(value), float64(expected[position]), 1e-6) {
			t.Fatalf("value %d came back as %v, not %v", position, value, expected[position])
		}
	}

	// Sub-byte kinds export at the stride the library reports, not one this side derives.
	packed, err := index.Get(keys[0], ScalarU1x8, nil)
	if err != nil || packed.Rows() != 1 || packed.Stride() != dimensions/8 || len(DenseValues[U1x8Scalar](packed)) != 1 {
		t.Fatalf("a packed export spans %d rows at a stride of %d: %v", packed.Rows(), packed.Stride(), err)
	}
	packed.Close()

	absent, err := index.Get(KeyFromUint64(999), ScalarF32, nil)
	if err != nil || absent.Rows() != 0 || len(DenseValues[float32](absent)) != 0 {
		t.Fatalf("an absent key exported %d rows: %v", absent.Rows(), err)
	}
}

// A lent arena is rewound by the next call it serves, so one loop allocates once, while a result
// landed without one owns its arena and outlives every later call.
func TestArenasOutliveOrYieldToTheNextCall(t *testing.T) {
	t.Parallel()
	const dimensions = 4
	index, keys, slab := populated(t, angularF32(dimensions), 3)

	owned, err := index.Get(keys[0], ScalarF32, nil)
	if err != nil {
		t.Fatalf("the first row must come back: %v", err)
	}
	defer owned.Close()

	var arena Arena
	defer arena.Close()
	first, err := index.Get(keys[1], ScalarF32, &arena)
	if err != nil {
		t.Fatalf("the second row must land in the arena: %v", err)
	}
	handle := arena.handle
	second, err := index.Get(keys[2], ScalarF32, &arena)
	if err != nil || arena.handle != handle {
		t.Fatalf("the arena must be rewound, not replaced: %v", err)
	}
	if first.data != second.data {
		t.Error("a rewound arena must land the next rows where the last ones were")
	}
	if err := second.Close(); err != nil || arena.handle != handle {
		t.Fatal("closing a result must leave a lent arena to its owner")
	}

	for position, value := range DenseValues[float32](owned) {
		if !nearly(float64(value), float64(slab[position]), 1e-6) {
			t.Fatalf("the owned result changed under later calls at %d: %v", position, value)
		}
	}
}

// Products land ragged, so the offsets the binding allocates are what splits them into rows.
func TestMultiplyLandsEveryRowsPairs(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	options := angularF32(dimensions)
	options.Metric = DenseMetricInner
	index, _, slab := populated(t, options, 16)

	var stats DenseTraversalStats
	products, err := index.On(pool(t)).Multiply(RowsOf(slab[:3*dimensions], dimensions),
		&DenseMultiplyConfig{Wanted: 4, Stats: &stats})
	if err != nil {
		t.Fatalf("the product must resolve: %v", err)
	}
	defer products.Close()
	if products.Rows() != 3 || products.Required() < 3*4 {
		t.Fatalf("the product spans %d rows of %d pairs", products.Rows(), products.Required())
	}
	for row := uint(0); row < products.Rows(); row++ {
		if keys, scores := products.Row(row); len(keys) != 4 || len(scores) != 4 {
			t.Errorf("row %d kept %d pairs", row, len(keys))
		}
	}
	if keys, scores := products.Row(3); keys != nil || scores != nil {
		t.Error("a row past the end must borrow nothing")
	}

	angular, _, _ := populated(t, angularF32(dimensions), 4)
	if _, err := angular.Multiply(RowsOf(slab[:dimensions], dimensions), nil); !errors.Is(err, StatusConfigMismatch) {
		t.Errorf("a product over distances reported %v", err)
	}
}

// A cursor takes the row width from the caller's matrix, so the rows it lands must be the ones
// inserted, in the scalar kind the caller asked for.
func TestCursorsWalkEveryRowOnce(t *testing.T) {
	t.Parallel()
	const dimensions = 4
	const count = 5
	index, keys, slab := populated(t, angularF32(dimensions), count)

	cursor, err := index.On(pool(t)).Cursor()
	if err != nil {
		t.Fatalf("the cursor must open: %v", err)
	}
	defer cursor.Close()
	walked := make([]Key, 0, count)
	rows := make([]float64, 0, count*dimensions)
	batch := make([]Key, 2)
	values := make([]float64, 2*dimensions)
	for {
		written, err := cursor.NextBatch(batch, RowsOf(values, dimensions))
		if err != nil {
			t.Fatalf("the cursor must pull: %v", err)
		}
		if written == 0 {
			break
		}
		walked = append(walked, batch[:written]...)
		rows = append(rows, values[:written*dimensions]...)
	}
	if !slices.Equal(walked, keys) {
		t.Fatalf("the cursor walked %v", walked)
	}
	for position, value := range rows {
		if !nearly(value, float64(slab[position]), 1e-6) {
			t.Fatalf("value %d came back as %v, not %v", position, value, slab[position])
		}
	}

	keysOnly, err := index.Cursor()
	if err != nil {
		t.Fatalf("a keys-only cursor must open: %v", err)
	}
	defer keysOnly.Close()
	if written, err := keysOnly.NextKeys(make([]Key, count+1)); err != nil || written != count {
		t.Errorf("a keys-only pull wrote %d keys: %v", written, err)
	}
	if err := keysOnly.Close(); err != nil {
		t.Fatalf("the cursor must close: %v", err)
	}
	if _, err := keysOnly.NextKeys(batch); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a closed cursor pulled with %v", err)
	}
}

// A stream pulls one row of a search at a time, so its first pull is the nearest match.
func TestSearchStreamsYieldNearestFirst(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	index, keys, slab := populated(t, angularF32(dimensions), 6)

	if _, err := index.SearchStream(RowsOf(slab[:2*dimensions], dimensions), nil); !errors.Is(err, StatusShapeMismatch) {
		t.Errorf("a two-row stream opened with %v", err)
	}
	stream, err := index.SearchStream(RowsOf(slab[:dimensions], dimensions), nil)
	if errors.Is(err, StatusUnsupported) {
		t.Skip("this build streams no residency")
	}
	if err != nil {
		t.Fatalf("the stream must open: %v", err)
	}
	defer stream.Close()
	found := make([]Key, 4)
	distances := make([]Distance, 4)
	total := uint(0)
	for {
		written, err := stream.Next(found, distances)
		if err != nil {
			t.Fatalf("the stream must pull: %v", err)
		}
		if written == 0 {
			break
		}
		if total == 0 && found[0] != keys[0] {
			t.Errorf("the stream opened with %v", found[0])
		}
		total += written
	}
	if total != 6 {
		t.Errorf("the stream yielded %d of 6 entries", total)
	}
	if _, err := stream.Next(found, distances[:1]); !errors.Is(err, StatusShapeMismatch) {
		t.Errorf("unequal slabs pulled with %v", err)
	}
}

// endregion: Ingest & Retrieval

// region: Shape Rejection

// The keys-to-rows, documents-to-rows, wanted-to-span and carry-to-queries guards are this
// binding's, since each call has one count and C pointers carry no lengths; widths and capacities
// cross untouched and come back refused by the library.
func TestRejectsMalformedShapes(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	options := angularF32(dimensions)
	options.Metric = DenseMetricInner
	index, _, slab := populated(t, options, 4)
	on := index.On(pool(t))
	queries := RowsOf(slab[:2*dimensions], dimensions)
	narrow := make([]float32, dimensions/2)

	cases := []struct {
		name string
		run  func() error
	}{
		{"more rows than keys", func() error {
			return on.Insert(sequentialKeys(1), RowsOf(slab, dimensions), nil)
		}},
		{"rows of the wrong width", func() error {
			return on.Insert(sequentialKeys(1), RowsOf(narrow, dimensions/2), nil)
		}},
		{"queries of the wrong width", func() error {
			_, err := on.Search(RowsOf(narrow, dimensions/2), 4, nil)
			return err
		}},
		{"fewer output rows than queries", func() error {
			lent, _ := MatchesSpanOf(make([]Key, 4), make([]Distance, 4), make([]uint, 1))
			_, err := on.Search(queries, 4, &DenseSearchConfig{Matches: &lent})
			return err
		}},
		{"more wanted than the span holds", func() error {
			lent, _ := MatchesSpanOf(make([]Key, 4), make([]Distance, 4), make([]uint, 2))
			_, err := on.Search(queries, 3, &DenseSearchConfig{Matches: &lent})
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
			_, err := on.Maxsim(queries, 3, 4, nil)
			return err
		}},
		{"documents of no rows", func() error {
			_, err := on.Maxsim(queries, 0, 4, nil)
			return err
		}},
		{"a carry shorter than the queries", func() error {
			lookup := ValueLookup{Values: RowsOf(slab, dimensions), Row: func(Key) (uint, bool) { return 0, true }}
			outputs := make([]float32, 2*dimensions)
			return on.Attend(queries, lookup, RowsOf(outputs, dimensions),
				&DenseAttendConfig{Carry: make([]DenseSoftmaxState, 1)})
		}},
	}

	for _, single := range cases {
		t.Run(single.name, func(t *testing.T) {
			if err := single.run(); !errors.Is(err, StatusShapeMismatch) {
				t.Errorf("reported %v", err)
			}
		})
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

	var stats DenseTraversalStats
	outputs, err := index.On(pool(t)).Search(RowsOf(slab[:queries*dimensions], dimensions), wanted,
		&DenseSearchConfig{Stats: &stats})
	if err != nil {
		t.Fatalf("the batch must resolve: %v", err)
	}
	if outputs.Queries() != queries {
		t.Fatalf("the outputs describe %d of %d queries", outputs.Queries(), queries)
	}
	for query := uint(0); query < queries; query++ {
		if found, distances := outputs.Row(query); uint(len(found)) != wanted || uint(len(distances)) != wanted {
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

	// A lent span wider than the wanted count lands each row at the span's own pitch.
	lent, err := MatchesSpanOf(make([]Key, queries*(wanted+2)), make([]Distance, queries*(wanted+2)), make([]uint, queries))
	if err != nil {
		t.Fatalf("the slabs must adopt: %v", err)
	}
	landed, err := index.Search(RowsOf(slab[:queries*dimensions], dimensions), wanted, &DenseSearchConfig{Matches: &lent})
	if err != nil {
		t.Fatalf("the batch must land in the lent span: %v", err)
	}
	for query := uint(0); query < queries; query++ {
		into, _ := landed.Row(query)
		allocated, _ := outputs.Row(query)
		if !slices.Equal(into, allocated) {
			t.Fatalf("query %d landed %v rather than %v", query, into, allocated)
		}
	}
}

// The predicate crosses as a cgo handle and runs on the library's worker threads - the most
// dangerous construct in the package.
func TestSearchAdmitsOnlyApprovedKeys(t *testing.T) {
	t.Parallel()
	const dimensions = 32
	index, _, slab := populated(t, angularF32(dimensions), 128)
	queries := RowsOf(slab[:3*dimensions], dimensions)
	on := index.On(pool(t))

	even := func(key Key) bool {
		value, integral := key.Uint64()
		return integral && value%2 == 0
	}
	admitted, err := on.Search(queries, 10, &DenseSearchConfig{ExpansionFactor: 64, Predicate: even})
	if err != nil {
		t.Fatalf("the filtered batch must resolve: %v", err)
	}
	seen := 0
	for query := uint(0); query < 3; query++ {
		found, _ := admitted.Row(query)
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

	// Admitting nothing is a result, not a failure, and a panic rejects rather than unwinding.
	for name, predicate := range map[string]Predicate{
		"rejecting": func(Key) bool { return false },
		"panicking": func(Key) bool { panic("a caller's bug") },
	} {
		rejected, err := on.Search(queries, 10, &DenseSearchConfig{Predicate: predicate})
		if err != nil {
			t.Fatalf("a %s filter must still resolve: %v", name, err)
		}
		for query := uint(0); query < 3; query++ {
			if found, _ := rejected.Row(query); len(found) != 0 {
				t.Fatalf("a %s filter admitted %d matches", name, len(found))
			}
		}
	}
}

// A handler is called with growing counts and stops the call at the first false, under the status
// a stop reports, and a panic inside it reads as a stop.
func TestProgressHandlersReportAndStop(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	index, _, slab := populated(t, angularF32(dimensions), 32)
	queries := RowsOf(slab[:4*dimensions], dimensions)

	var reports atomic.Uint64
	reporting := func(completed, total uint) bool {
		reports.Add(1)
		return completed <= total
	}
	if _, err := index.Search(queries, 3, &DenseSearchConfig{Progress: reporting}); err != nil {
		t.Fatalf("a reporting search must resolve: %v", err)
	}
	if reports.Load() == 0 {
		t.Error("the handler was never called")
	}
	for name, progress := range map[string]ProgressCallback{
		"stopping":  func(uint, uint) bool { return false },
		"panicking": func(uint, uint) bool { panic("a caller's bug") },
	} {
		if _, err := index.Search(queries, 3, &DenseSearchConfig{Progress: progress}); !errors.Is(err, StatusCancelled) {
			t.Errorf("a %s handler reported %v", name, err)
		}
	}
}

// endregion: Search

// region: Metrics & Quantization

func TestEveryMetricAndQuantizationCrosses(t *testing.T) {
	t.Parallel()
	const dimensions = 64

	scalars := []struct {
		metric       DenseMetric
		quantization Scalar
	}{
		{DenseMetricInner, ScalarF32}, {DenseMetricAngular, ScalarF64}, {DenseMetricEuclidean, ScalarBF16}, {DenseMetricAngular, ScalarF16},
		{DenseMetricEuclidean, ScalarE5M2}, {DenseMetricAngular, ScalarE4M3}, {DenseMetricInner, ScalarI8}, {DenseMetricAngular, ScalarU8},
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
	for _, metric := range []DenseMetric{DenseMetricHamming, DenseMetricJaccard} {
		t.Run(metric.String()+"/u1x8", func(t *testing.T) {
			options := angularF32(dimensions)
			options.Metric = metric
			options.Quantization = ScalarU1x8
			index := newIndex(t, options, 32)

			const rows = 16
			generator := rand.New(rand.NewSource(seed))
			packed := make([]U1x8Scalar, rows*dimensions/8)
			for position := range packed {
				packed[position] = U1x8Scalar(generator.Intn(256))
			}
			if err := index.Insert(sequentialKeys(rows), RowsOf(packed, dimensions), nil); err != nil {
				t.Fatalf("packed rows must insert: %v", err)
			}
			outputs, err := index.Search(RowsOf(packed[:dimensions/8], dimensions), 4, nil)
			if err != nil {
				t.Fatalf("packed rows must search: %v", err)
			}
			if found, _ := outputs.Row(0); len(found) == 0 {
				t.Error("a packed walk reported no matches")
			}
		})
	}
}

// endregion: Metrics & Quantization

// region: Linear Algebra & Refinement

// The outputs are sized over documents rather than rows, which is the one piece of arithmetic
// this door does that the others do not.
func TestMaxsimSizesItsOutputsOverDocuments(t *testing.T) {
	t.Parallel()
	const (
		dimensions = 4
		tokens     = 2
		documents  = 3
		queries    = 2
	)
	options := angularF32(dimensions)
	options.Metric = DenseMetricInner
	options.ValuesPerKey = tokens
	index := newIndex(t, options, documents*tokens)

	keys := make([]Key, 0, documents*tokens)
	for document := uint(0); document < documents; document++ {
		for token := uint(0); token < tokens; token++ {
			keys = append(keys, KeyFromUint64(uint64(document)))
		}
	}
	slab := seededRows(documents*tokens, dimensions)
	if err := index.Insert(keys, RowsOf(slab, dimensions), nil); err != nil {
		t.Fatalf("the token rows must insert: %v", err)
	}

	var stats DenseTraversalStats
	outputs, err := index.On(pool(t)).Maxsim(RowsOf(slab[:queries*tokens*dimensions], dimensions), tokens, documents,
		&DenseMaxsimConfig{Stats: &stats})
	if err != nil {
		t.Fatalf("the late interaction must resolve: %v", err)
	}
	for query := uint(0); query < queries; query++ {
		if found, _ := outputs.Row(query); len(found) != documents {
			t.Errorf("query %d scored %d of %d documents", query, len(found), documents)
		}
	}
	if stats.ComputedDistances == 0 {
		t.Errorf("the walk reported %+v", stats)
	}
}

// The lookup crosses as a cgo handle over a pinned matrix, so its rows must be what the outputs
// average: every value row equal makes every output that row, whatever the softmax weights.
func TestAttendFoldsTheRowsTheLookupNames(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	const count = 16
	options := angularF32(dimensions)
	options.Metric = DenseMetricInner
	index, _, slab := populated(t, options, count)

	value := seededRows(1, dimensions)
	lookup := ValueLookup{Values: RowsOf(value, dimensions), Row: func(Key) (uint, bool) { return 0, true }}
	outputs := make([]float32, 2*dimensions)
	carry := make([]DenseSoftmaxState, 2)
	if err := index.On(pool(t)).Attend(RowsOf(slab[:2*dimensions], dimensions), lookup, RowsOf(outputs, dimensions),
		&DenseAttendConfig{Carry: carry}); err != nil {
		t.Fatalf("the attention must resolve: %v", err)
	}
	for position, attended := range outputs {
		if !nearly(float64(attended), float64(value[position%dimensions]), 1e-4) {
			t.Fatalf("output %d attended to %v, not %v", position, attended, value[position%dimensions])
		}
	}
	if carry[0].Partition <= 0 {
		t.Errorf("the carry reported %+v", carry[0])
	}

	// A missing row, a panicking lookup and values of another width are a wrong softmax, never a smaller one.
	for name, refused := range map[string]ValueLookup{
		"missing":   {Values: lookup.Values, Row: func(Key) (uint, bool) { return 0, false }},
		"past":      {Values: lookup.Values, Row: func(Key) (uint, bool) { return 1, true }},
		"panicking": {Values: lookup.Values, Row: func(Key) (uint, bool) { panic("a caller's bug") }},
		"narrow":    {Values: RowsOf(value[:dimensions/2], dimensions/2), Row: lookup.Row},
	} {
		if err := index.Attend(RowsOf(slab[:dimensions], dimensions), refused, RowsOf(outputs[:dimensions], dimensions), nil); !errors.Is(err, StatusShapeMismatch) {
			t.Errorf("a %s lookup reported %v", name, err)
		}
	}
	if err := index.Attend(RowsOf(slab[:dimensions], dimensions), ValueLookup{Values: lookup.Values},
		RowsOf(outputs[:dimensions], dimensions), nil); !errors.Is(err, StatusInvalidArgument) {
		t.Errorf("a lookup without a row function reported %v", err)
	}
}

// A sweep reports the nodes whose edges changed rather than a status, so a caller loops on the
// number it returns, and an audit reports how much of the result a search reaches.
func TestRefineAndHealthReportTheirSweeps(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	const count = 32
	index, _, _ := populated(t, angularF32(dimensions), count)
	on := index.On(pool(t))

	var stats DenseTraversalStats
	changed, err := on.Refine(&DenseRefineConfig{ExpansionFactor: 64, Stats: &stats})
	if err != nil {
		t.Fatalf("the sweep must run: %v", err)
	}
	if changed > count {
		t.Errorf("a sweep changed %d of %d nodes", changed, count)
	}
	if stats.VisitedMembers == 0 {
		t.Errorf("the sweep reported %+v", stats)
	}

	reachability, err := on.Health()
	if err != nil {
		t.Fatalf("the audit must run: %v", err)
	}
	if reachability.Audited != count || reachability.Reachable > reachability.Audited {
		t.Errorf("the audit reported %+v", reachability)
	}
}

// Every donor is consumed whatever the outcome, so no donor can be closed twice.
func TestMergeConsumesItsDonors(t *testing.T) {
	t.Parallel()
	const dimensions = 4
	index, _, _ := populated(t, angularF32(dimensions), 2)
	donor, err := DenseInitRam(angularF32(dimensions), nil)
	if err != nil {
		t.Fatalf("the donor must initialize: %v", err)
	}
	if err := donor.Insert([]Key{KeyFromUint64(7)}, RowsOf(seededRows(1, dimensions), dimensions), nil); err != nil {
		t.Fatalf("the donor must insert: %v", err)
	}

	err = index.Merge([]*DenseIndex{donor}, nil)
	if err != nil && !errors.Is(err, StatusUnsupported) {
		t.Fatalf("the merge reported %v", err)
	}
	if _, err := donor.Describe(); !errors.Is(err, StatusUninitialized) {
		t.Errorf("a consumed donor described with %v", err)
	}
	if err := donor.Close(); err != nil {
		t.Errorf("a consumed donor closed with %v", err)
	}
	if err := index.Merge([]*DenseIndex{donor}, nil); !errors.Is(err, StatusUninitialized) {
		t.Errorf("a consumed donor merged again with %v", err)
	}
}

// endregion: Linear Algebra & Refinement

// region: Serialization

// Every option field must survive the checkpoint, not just the two a reader usually looks at.
func TestSerializationPreservesEveryOption(t *testing.T) {
	t.Parallel()
	const dimensions = 16
	options := angularF32(dimensions)
	options.Metric = DenseMetricEuclidean
	options.Quantization = ScalarI8
	options.Connectivity = 24
	options.ValuesPerKey = 2

	source, _, _ := populated(t, options, 32)
	on := source.On(pool(t))
	path := filepath.Join(t.TempDir(), "index.safegraphs")
	if err := on.Save(path, nil); err != nil {
		t.Fatalf("the checkpoint must write: %v", err)
	}

	buffer, err := on.SaveBuffer(nil)
	if err != nil || len(buffer.Bytes) == 0 {
		t.Fatalf("the index serializes to %d bytes: %v", len(buffer.Bytes), err)
	}
	defer buffer.Close()

	for _, single := range []struct {
		name     string
		metadata func() (DenseMetadata, error)
		restore  func() (*DenseIndex, error)
	}{
		{"file",
			func() (DenseMetadata, error) { return DenseMetadataRead(path, nil) },
			func() (*DenseIndex, error) {
				return DenseLoadRam(path, &DenseLoadRamConfig{Executor: pool(t), MemoryDomains: []int32{0}, ValuesPerKey: 2})
			}},
		{"buffer",
			func() (DenseMetadata, error) {
				return DenseMetadataReadBuffer(buffer.Bytes, &DenseMetadataReadBufferConfig{Executor: pool(t)})
			},
			func() (*DenseIndex, error) {
				return DenseLoadRamBuffer(buffer.Bytes, &DenseLoadRamBufferConfig{Executor: pool(t), ValuesPerKey: 2})
			}},
		{"view",
			func() (DenseMetadata, error) { return DenseMetadataRead(path, nil) },
			func() (*DenseIndex, error) { return DenseViewRam(path) }},
	} {
		t.Run(single.name, func(t *testing.T) {
			metadata, err := single.metadata()
			if err != nil {
				t.Fatalf("the metadata must read without loading: %v", err)
			}
			// The four fields a checkpoint froze, and no more: the placement and ValuesPerKey
			// are the reader's to choose.
			if metadata.Dimensions != dimensions || metadata.Metric != DenseMetricEuclidean ||
				metadata.Quantization != ScalarI8 || metadata.Connectivity != 24 {
				t.Errorf("the checkpoint describes %+v", metadata)
			}

			restored, err := single.restore()
			if err != nil {
				t.Fatalf("the checkpoint must load: %v", err)
			}
			defer restored.Close()
			description, err := restored.Describe()
			if err != nil || description.Metadata != metadata || description.Size != 32 {
				t.Fatalf("the restored index describes %+v: %v", description, err)
			}
			if found := searchOne(t, restored, seededRows(1, dimensions), dimensions); found != 1 {
				t.Errorf("the restored index reported %d matches", found)
			}
		})
	}

	viewed, err := DenseViewRam(path)
	if err != nil {
		t.Fatalf("the checkpoint must map: %v", err)
	}
	defer viewed.Close()
	if err := viewed.Reserve(64); !errors.Is(err, StatusReadOnly) {
		t.Errorf("a view reserved with %v", err)
	}
}

// endregion: Serialization

// region: Deferred Calls

// A deferred search pins its queries and owns its outputs, and answers what the blocking one
// does once drained and waited on.
func TestDeferredSearchMatchesTheBlockingOne(t *testing.T) {
	t.Parallel()
	const dimensions = 16
	const queries = 4
	const wanted = 5
	index, _, slab := populated(t, angularF32(dimensions), 64)
	executor := pool(t)

	call := index.Deferred(executor).Search(RowsOf(slab[:queries*dimensions], dimensions), wanted, nil)
	if status := call.Status(); status != StatusDeferred {
		t.Fatalf("a submitted call reported %v", status)
	}
	runtime.GC() // the pins, not the variables, keep everything the library reads in place
	if err := executor.Drain(MoodParked); err != nil {
		t.Fatalf("the executor must drain: %v", err)
	}
	if status := call.Status(); status != StatusSuccess {
		t.Fatalf("a drained call reported %v", status)
	}
	deferred, err := call.Wait()
	if err != nil {
		t.Fatalf("the call must succeed: %v", err)
	}

	blocking, err := index.Search(RowsOf(slab[:queries*dimensions], dimensions), wanted, nil)
	if err != nil {
		t.Fatalf("the blocking search must succeed: %v", err)
	}
	for query := uint(0); query < queries; query++ {
		later, _ := deferred.Row(query)
		now, _ := blocking.Row(query)
		if !slices.Equal(later, now) {
			t.Fatalf("query %d found %v rather than %v", query, later, now)
		}
	}
}

// An owned arena and owned outputs travel with the call and come back with its result.
func TestDeferredCallsHandBackWhatTheyOwned(t *testing.T) {
	t.Parallel()
	const dimensions = 4
	index, keys, slab := populated(t, angularF32(dimensions), 4)
	executor := pool(t)
	on := index.Deferred(executor)

	found := on.Get(keys[2], ScalarF32, nil)
	buffer := on.SaveBuffer(nil)
	counted := on.Count(keys[3])
	described := on.Describe()
	if err := executor.Drain(MoodParked); err != nil {
		t.Fatalf("the executor must drain: %v", err)
	}

	vectors, err := found.Wait()
	if err != nil {
		t.Fatalf("the rows must come back: %v", err)
	}
	defer vectors.Close()
	if !slices.Equal(DenseValues[float32](vectors), slab[2*dimensions:3*dimensions]) {
		t.Errorf("the rows came back as %v", DenseValues[float32](vectors))
	}
	monolith, err := buffer.Wait()
	if err != nil {
		t.Fatalf("the monolith must come back: %v", err)
	}
	defer monolith.Close()
	if metadata, err := DenseMetadataReadBuffer(monolith.Bytes, nil); err != nil || metadata.Dimensions != dimensions {
		t.Errorf("the monolith describes %+v: %v", metadata, err)
	}
	if entries, err := counted.Wait(); err != nil || entries != 1 {
		t.Errorf("the count came back as %d: %v", entries, err)
	}
	if description, err := described.Wait(); err != nil || description.Size != 4 {
		t.Errorf("the description came back as %+v: %v", description, err)
	}
}

// A blocking call on an executor without threads of its own first runs the deferred calls it
// holds, so a predicate re-enters Go from inside that blocking call - which is why no verb naming
// an executor claims `nocallback`.
func TestBlockingCallsRunTheCallbacksOfHeldOnes(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	index, keys, slab := populated(t, angularF32(dimensions), 16)
	executor := pool(t)

	var admissions atomic.Uint64
	odd := func(key Key) bool {
		admissions.Add(1)
		value, _ := key.Uint64()
		return value%2 == 1
	}
	call := index.Deferred(executor).Search(RowsOf(slab[:dimensions], dimensions), 4,
		&DenseSearchConfig{Predicate: odd})
	if _, err := index.On(executor).Count(keys[0]); err != nil {
		t.Fatalf("the blocking count must run: %v", err)
	}
	matches, err := call.Wait()
	if err != nil {
		t.Fatalf("the filtered call must succeed: %v", err)
	}
	if admissions.Load() == 0 {
		t.Error("the predicate never ran")
	}
	found, _ := matches.Row(0)
	for _, key := range found {
		if !odd(key) {
			t.Fatalf("the filter admitted %v", key)
		}
	}
}

// Close settles a call no one waits for, which still lands, and leaves alone what Wait handed out.
func TestClosedDeferredCallsStillLand(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	const count = 64
	index := newIndex(t, angularF32(dimensions), count)
	executor := pool(t)
	slab := seededRows(count, dimensions)
	on := index.Deferred(executor)

	inserted := on.Insert(sequentialKeys(count), RowsOf(slab, dimensions), nil)
	discarded := on.Get(KeyFromUint64(1), ScalarF32, nil)
	kept := on.Get(KeyFromUint64(1), ScalarF32, nil)
	if err := executor.Drain(MoodParked); err != nil {
		t.Fatalf("the executor must drain: %v", err)
	}
	for _, call := range []interface{ Close() error }{inserted, discarded, discarded} {
		if err := call.Close(); err != nil {
			t.Fatalf("a closed call reported %v", err)
		}
	}
	vectors, err := kept.Wait()
	if err != nil {
		t.Fatalf("the rows must come back: %v", err)
	}
	defer vectors.Close()
	if err := kept.Close(); err != nil {
		t.Fatalf("closing a waited call reported %v", err)
	}
	if !slices.Equal(DenseValues[float32](vectors), slab[:dimensions]) {
		t.Errorf("the rows came back as %v", DenseValues[float32](vectors))
	}
	if description, err := index.Describe(); err != nil || description.Size != count {
		t.Fatalf("the index holds %d of %d entries: %v", description.Size, count, err)
	}
}

// A submission finding the executor's deferred table full is refused rather than waited for, and
// a refusal the binding catches is reported through the same Status and Wait.
func TestDeferredRefusalsReachStatusAndWait(t *testing.T) {
	t.Parallel()
	const dimensions = 8
	const count = 16
	index := newIndex(t, angularF32(dimensions), count)
	slab := seededRows(count, dimensions)

	executor, err := ExecutorForRam(nil, 1, 1)
	if err != nil {
		t.Fatalf("the executor must spawn: %v", err)
	}
	defer executor.Close()
	on := index.Deferred(executor)
	first := on.Insert(sequentialKeys(count), RowsOf(slab, dimensions), nil)
	second := on.Insert(sequentialKeys(count), RowsOf(slab, dimensions), nil)
	if status := second.Status(); status != StatusContended {
		t.Errorf("a submission past the table reported %v", status)
	}
	if _, err := second.Wait(); !errors.Is(err, StatusContended) {
		t.Errorf("a submission past the table finished with %v", err)
	}
	if err := executor.Drain(MoodParked); err != nil {
		t.Fatalf("the executor must drain: %v", err)
	}
	if _, err := first.Wait(); err != nil {
		t.Fatalf("the first submission must succeed: %v", err)
	}

	refused := on.Insert(sequentialKeys(1), RowsOf(slab, dimensions), nil)
	if status := refused.Status(); status != StatusShapeMismatch {
		t.Errorf("a refused submission reported %v", status)
	}
	if _, err := refused.Wait(); !errors.Is(err, StatusShapeMismatch) {
		t.Errorf("a refused submission finished with %v", err)
	}
	if description, err := index.Describe(); err != nil || description.Size != count {
		t.Fatalf("the index holds %d of %d entries: %v", description.Size, count, err)
	}
}

// endregion: Deferred Calls

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
		mine := index.On(pool(t))
		go func(writer int) {
			defer writing.Done()
			for batch := 0; batch < batches; batch++ {
				keys := make([]Key, batchRows)
				for row := range keys {
					keys[row] = KeyFromUint64(uint64(1000 + (writer*batches+batch)*batchRows + row))
				}
				rows := seededRows(batchRows, dimensions)
				if err := mine.Insert(keys, RowsOf(rows, dimensions), nil); err != nil {
					t.Errorf("a concurrent batch failed: %v", err)
					return
				}
			}
		}(writer)
	}

	for reader := 0; reader < writers; reader++ {
		reading.Add(1)
		mine := index.On(pool(t))
		go func() {
			defer reading.Done()
			query := RowsOf(slab[:dimensions], dimensions)
			for {
				select {
				case <-done:
					return
				default:
				}
				if _, err := mine.Search(query, 10, nil); err != nil {
					t.Errorf("a concurrent search failed: %v", err)
					return
				}
			}
		}()
	}

	writing.Wait()
	close(done)
	reading.Wait()

	expected := uint(seeded + writers*batches*batchRows)
	if description, err := index.Describe(); err != nil || description.Size != expected {
		t.Fatalf("the index holds %d of %d entries: %v", description.Size, expected, err)
	}
}

// endregion: Concurrency
