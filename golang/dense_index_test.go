// Tests for the boundary, not for the engine behind it. Whether approximate search returns
// good neighbors is the C library's own suite's job; what can break here is marshalling -
// struct layouts across an anonymous union, sixteen-byte keys reinterpreted in place, stride
// arithmetic, and handle ownership.
//
// Half the memory-safety surface is checked by tooling rather than by assertions, so run this
// under `go test -race` with `GODEBUG=cgocheck=1`.

package usearch

import (
	"encoding/json"
	"math"
	"math/rand"
	"path/filepath"
	"sync"
	"testing"
)

// region: Helpers

const seed = 42

// seededRows returns deterministic unit-length rows. Normalizing is not cosmetic: on the unit
// sphere inner, angular, and euclidean induce the same ranking, so one assertion covers all
// three, and near-parallel rows stop deciding top-1 by float noise.
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

// sequentialKeys names rows one through count, so a key's integer identity is its row plus one.
func sequentialKeys(count uint) []Key {
	keys := make([]Key, count)
	for row := uint(0); row < count; row++ {
		keys[row] = KeyFromUint64(uint64(row) + 1)
	}
	return keys
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
	added, err := index.Add(keys, RowsOf(slab, options.Dimensions), 0, nil)
	if err != nil {
		t.Fatalf("the batch must insert: %v", err)
	}
	if added != count {
		t.Fatalf("the batch inserted %d of %d rows", added, count)
	}
	return index, keys, slab
}

// topOne collapses the three-buffer dance into the answer most assertions want.
func topOne(t *testing.T, index *DenseIndex, query []float32, dimensions uint) (Key, Distance, bool) {
	t.Helper()
	outputs := PackedDenseOutputs(1, 1)
	if err := index.SearchInto(RowsOf(query, dimensions), &outputs, 0, nil); err != nil {
		t.Fatalf("the search must run: %v", err)
	}
	keys, distances := outputs.MatchesOf(0)
	if len(keys) == 0 {
		return NilKey, 0, false
	}
	return keys[0], distances[0], true
}

func nearly(left, right, tolerance float64) bool { return math.Abs(left-right) <= tolerance }

// endregion: Helpers

// region: Keys

func TestKeysRoundTripThroughTheirRendering(t *testing.T) {
	if rendered := KeyFromUint64(42).String(); rendered != "00000000-0000-0000-0000-00000000002a" {
		t.Fatalf("the integer identity rendered as %q", rendered)
	}

	for _, value := range []uint64{0, 1, 42, 1 << 32, ^uint64(0)} {
		key := KeyFromUint64(value)
		recovered, integral := key.Uint64()
		if !integral || recovered != value {
			t.Fatalf("%d recovered as %d, integral %v", value, recovered, integral)
		}
		parsed, err := ParseKey(key.String())
		if err != nil || parsed != key {
			t.Fatalf("%v round-tripped as %v: %v", key, parsed, err)
		}
	}

	// A non-zero high half has no integer identity, and dashes may sit anywhere or nowhere.
	var wide Key
	wide[0] = 1
	if _, integral := wide.Uint64(); integral {
		t.Fatal("a key with high bytes set must report no integer identity")
	}
	for _, text := range []string{"0000000000000000000000000000002a", "00000000-0000-0000-0000-00000000002A"} {
		if parsed, err := ParseKey(text); err != nil || parsed != KeyFromUint64(42) {
			t.Fatalf("parsing %q gave %v: %v", text, parsed, err)
		}
	}
	for _, text := range []string{"", "abc", "00000000-0000-0000-0000-00000000002ag"} {
		if _, err := ParseKey(text); StatusOf(err) != MalformedInput {
			t.Fatalf("parsing %q reported %v", text, StatusOf(err))
		}
	}

	// The text interfaces are what carries a key through a JSON document.
	type record struct {
		Key Key `json:"key"`
	}
	encoded, err := json.Marshal(record{Key: KeyFromUint64(42)})
	if err != nil {
		t.Fatalf("the key must marshal: %v", err)
	}
	var decoded record
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("the key must unmarshal: %v", err)
	}
	if decoded.Key != KeyFromUint64(42) {
		t.Fatalf("%s decoded as %v", encoded, decoded.Key)
	}
}

// endregion: Keys

// region: Matrix Views

func TestMatrixViewsDeriveTheirShape(t *testing.T) {
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

// region: Options Round Trip

// The residency arm lives in an anonymous union that cgo cannot address, so it crosses through
// the accessors in shim.c. This is the only thing exercising them.
func TestOptionsSurviveTheRawRoundTrip(t *testing.T) {
	pinned := DenseOptions{
		Dimensions:       768,
		DimensionPadding: SimdQuantumDimensions,
		Metric:           Euclidean,
		Quantization:     I8,
		Connectivity:     32,
		Residency:        Residency{Kind: RAM, Threads: 7, HugePages: RequireHugePages},
	}

	options := []DenseOptions{pinned, DefaultDenseOptions()}
	// Ordinal zero is the trap: it must not come back as FirstFreeDevice, which marshals as -1.
	for _, device := range []Device{FirstFreeDevice, 0, 3} {
		for _, addressing := range []Addressing{DeviceOnly, UnifiedAddressing} {
			single := angularF32(64)
			single.Residency = Residency{Kind: VRAM, Device: device, Addressing: addressing}
			options = append(options, single)
		}
	}
	for _, sectors := range []uint32{0, 512, 4096} {
		for _, mode := range []IOMode{DirectIO, PageCachedIO} {
			single := angularF32(64)
			single.Residency = Residency{Kind: NVMe, SectorBytes: sectors, QueueDepth: 128, IOMode: mode}
			options = append(options, single)
		}
	}

	for _, single := range options {
		lifted, err := single.reload()
		if err != nil {
			t.Fatalf("%+v must survive the round trip: %v", single.Residency, err)
		}
		if lifted != single {
			t.Errorf("%+v came back as %+v", single, lifted)
		}
	}
}

func TestOptionsRejectUnknownTags(t *testing.T) {
	// An unknown residency never reaches the union, so it is caught on the way down.
	unknown := angularF32(64)
	unknown.Residency.Kind = ResidencyKind(9)
	if _, err := unknown.toRaw(); StatusOf(err) != UnknownKind {
		t.Errorf("an unknown residency must not be guessed: %v", StatusOf(err))
	}

	// A tag a newer library wrote is caught on the way up, never passed through.
	for _, single := range []struct {
		name                              string
		metric, quantization, residencyOf int32
	}{
		{"metric", 99, int32(F32), int32(RAM)},
		{"quantization", int32(Angular), 77, int32(RAM)},
		{"residency", int32(Angular), int32(F32), 9},
	} {
		t.Run(single.name, func(t *testing.T) {
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

// endregion: Lifecycle

// region: Ingest & Retrieval

// The bytes that come back are what proves every stride, scalar kind, and base pointer right;
// a count alone would pass with all three wrong.
func TestAddsAndGetsVectorsIntact(t *testing.T) {
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

	// A padded pitch is the layout a mapped dataset arrives in, and it must cross both ways.
	const pitch = 16
	padded := make([]float32, pitch)
	copy(padded, slab[2*dimensions:3*dimensions])
	if _, err := index.Add([]Key{KeyFromUint64(500)},
		StridedRowsOf(padded, 1, dimensions, pitch*4), 0, nil); err != nil {
		t.Fatalf("a padded row must insert: %v", err)
	}
	widened := make([]float32, pitch)
	if _, err := index.Get(KeyFromUint64(500), StridedRowsOf(widened, 1, dimensions, pitch*4)); err != nil {
		t.Fatalf("a padded row must come back: %v", err)
	}
	for position := 0; position < dimensions; position++ {
		if !nearly(float64(widened[position]), float64(padded[position]), 1e-6) {
			t.Fatalf("padded dimension %d came back as %v", position, widened[position])
		}
	}
}

// The truncation contract: Get reports what exists, never what it wrote, so a caller can size
// from the answer and call again.
func TestReportsTheTrueExtentWhenTruncated(t *testing.T) {
	const dimensions = 4
	index := newIndex(t, angularF32(dimensions), 16)
	shared := KeyFromUint64(7)
	slab := seededRows(3, dimensions)

	if _, err := index.Add([]Key{shared, shared, shared}, RowsOf(slab, dimensions), 0, nil); err != nil {
		t.Fatalf("three rows may share one key: %v", err)
	}
	if counted, err := index.Count(shared); err != nil || counted != 3 {
		t.Fatalf("the probe counted %d entries: %v", counted, err)
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
	const dimensions = 8
	index, _, slab := populated(t, angularF32(dimensions), 4)
	queries := RowsOf(slab[:2*dimensions], dimensions)
	narrow := make([]float32, dimensions/2)

	cases := []struct {
		name string
		run  func() error
	}{
		{"more rows than keys", func() error {
			_, err := index.Add(sequentialKeys(1), RowsOf(slab, dimensions), 0, nil)
			return err
		}},
		{"rows of the wrong width", func() error {
			_, err := index.Add(sequentialKeys(1), RowsOf(narrow, dimensions/2), 0, nil)
			return err
		}},
		{"queries of the wrong width", func() error {
			outputs := PackedDenseOutputs(1, 4)
			return index.SearchInto(RowsOf(narrow, dimensions/2), &outputs, 0, nil)
		}},
		{"fewer output rows than queries", func() error {
			outputs := PackedDenseOutputs(1, 4)
			return index.SearchInto(queries, &outputs, 0, nil)
		}},
		{"outputs with no room", func() error {
			_, err := DenseOutputsOf(make([]Key, 4), make([]Distance, 4), make([]uint, 1), 0)
			return err
		}},
		{"outputs narrower than wanted", func() error {
			_, err := DenseOutputsOf(make([]Key, 4), make([]Distance, 4), make([]uint, 2), 4)
			return err
		}},
	}

	for _, single := range cases {
		t.Run(single.name, func(t *testing.T) {
			if status := StatusOf(single.run()); status != ShapeMismatch {
				t.Errorf("reported %v", status)
			}
		})
	}

	outputs := PackedDenseOutputs(2, 4)
	if status := StatusOf(index.SearchIfInto(queries, &outputs, 0, nil, nil)); status != MalformedInput {
		t.Errorf("a nil predicate reported %v", status)
	}
	if status := StatusOf(index.Merge(nil)); status != Uninitialized {
		t.Errorf("an absent donor reported %v", status)
	}
}

// endregion: Shape Rejection

// region: Search

// Self-retrieval across a batch is what proves sixteen-byte keys survive in the right order and
// that the output row stride indexes correctly; the recall it implies is the library's business.
func TestSearchFindsEachRowAsItsOwnNearest(t *testing.T) {
	const (
		dimensions = 64
		count      = 256
		queries    = 8
		wanted     = 5
	)
	index, keys, slab := populated(t, angularF32(dimensions), count)

	outputs := PackedDenseOutputs(queries, wanted)
	var stats DenseStats
	if err := index.SearchInto(RowsOf(slab[:queries*dimensions], dimensions), &outputs, 64, &stats); err != nil {
		t.Fatalf("the batch must resolve: %v", err)
	}

	for query := uint(0); query < queries; query++ {
		found, distances := outputs.MatchesOf(query)
		if uint(len(found)) != wanted {
			t.Fatalf("query %d reported %d of %d matches", query, len(found), wanted)
		}
		if found[0] != keys[query] {
			t.Errorf("row %d found %v rather than itself, %v", query, found[0], keys[query])
		}
		if !nearly(distances[0], 0, 1e-6) {
			t.Errorf("row %d scored %v against itself", query, distances[0])
		}
		if distances[0] > distances[1] {
			t.Errorf("row %d reported matches out of order", query)
		}
	}

	// Three positional size_t fields, so a swap in the C struct would go unseen otherwise.
	if stats.VisitedMembers == 0 || stats.ComputedDistances == 0 {
		t.Errorf("the traversal reported %+v", stats)
	}
	total := stats
	total.Accumulate(stats)
	if total.VisitedMembers != 2*stats.VisitedMembers || total.Restarts != 2*stats.Restarts {
		t.Errorf("accumulating %+v gave %+v", stats, total)
	}
}

// The predicate crosses as a cgo handle and runs on the library's worker threads - the most
// dangerous construct in the package.
func TestSearchAdmitsOnlyApprovedKeys(t *testing.T) {
	const dimensions = 32
	index, _, slab := populated(t, angularF32(dimensions), 128)
	queries := RowsOf(slab[:3*dimensions], dimensions)

	admitted := PackedDenseOutputs(3, 10)
	even := func(key Key) bool {
		value, integral := key.Uint64()
		return integral && value%2 == 0
	}
	if err := index.SearchIfInto(queries, &admitted, 64, nil, even); err != nil {
		t.Fatalf("the filtered batch must resolve: %v", err)
	}
	seen := 0
	for query := uint(0); query < 3; query++ {
		found, _ := admitted.MatchesOf(query)
		for _, key := range found {
			value, integral := key.Uint64()
			if !integral || value%2 != 0 {
				t.Fatalf("query %d admitted %v", query, key)
			}
			seen++
		}
	}
	if seen == 0 {
		t.Fatal("the filter admitted nothing at all")
	}

	// Admitting nothing is a result, not a failure.
	rejected := PackedDenseOutputs(3, 10)
	if err := index.SearchIfInto(queries, &rejected, 0, nil, func(Key) bool { return false }); err != nil {
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

// The one place the Go enum values are checked against the C ones: a renumbered constant is
// caught here and nowhere else.
func TestEveryMetricAndQuantizationSearches(t *testing.T) {
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
			index, keys, slab := populated(t, options, 32)

			found, _, ok := topOne(t, index, slab[:dimensions], dimensions)
			if !ok {
				t.Fatal("a match must come back")
			}
			if found != keys[0] {
				t.Errorf("found %v rather than %v", found, keys[0])
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
			vectors := RowsOf(packed, dimensions)
			if vectors.Rows() != rows {
				t.Fatalf("counted %d of %d packed rows", vectors.Rows(), rows)
			}
			if _, err := index.Add(sequentialKeys(rows), vectors, 0, nil); err != nil {
				t.Fatalf("packed rows must insert: %v", err)
			}

			outputs := PackedDenseOutputs(1, 4)
			query := RowsOf(packed[:dimensions/8], dimensions)
			if err := index.SearchInto(query, &outputs, 0, nil); err != nil {
				t.Fatalf("packed rows must search: %v", err)
			}
			found, distances := outputs.MatchesOf(0)
			if len(found) == 0 {
				t.Fatal("a match must come back")
			}
			if !nearly(distances[0], 0, 1e-9) {
				t.Errorf("a row scored %v against itself", distances[0])
			}
		})
	}
}

// endregion: Metrics & Quantization

// region: Serialization

// Every option field must survive the checkpoint, not just the two a reader usually looks at.
func TestSerializationPreservesEveryOption(t *testing.T) {
	const dimensions = 16
	options := angularF32(dimensions)
	options.Metric = Euclidean
	options.Quantization = I8
	options.Connectivity = 24
	options.DimensionPadding = SimdQuantumDimensions
	options.Residency = Residency{Kind: RAM, Threads: 3}

	source, keys, slab := populated(t, options, 32)
	path := filepath.Join(t.TempDir(), "index.safegraphs")
	if err := source.Save(path); err != nil {
		t.Fatalf("the checkpoint must write: %v", err)
	}

	length, err := source.SerializedLength()
	if err != nil || length == 0 {
		t.Fatalf("the index serializes to %d bytes: %v", length, err)
	}
	buffer := make([]byte, length)
	if err := source.SaveBuffer(buffer); err != nil {
		t.Fatalf("the buffer must write: %v", err)
	}

	for _, single := range []struct {
		name     string
		metadata func() (DenseOptions, error)
		restore  func(*DenseIndex) error
	}{
		{"file",
			func() (DenseOptions, error) { return DenseMetadata(path) },
			func(index *DenseIndex) error { return index.Load(path) }},
		{"buffer",
			func() (DenseOptions, error) { return DenseMetadataBuffer(buffer) },
			func(index *DenseIndex) error { return index.LoadBuffer(buffer) }},
	} {
		t.Run(single.name, func(t *testing.T) {
			metadata, err := single.metadata()
			if err != nil {
				t.Fatalf("the metadata must read without loading: %v", err)
			}
			if metadata.Dimensions != dimensions || metadata.Metric != Euclidean ||
				metadata.Quantization != I8 || metadata.Connectivity != 24 ||
				metadata.DimensionPadding != SimdQuantumDimensions {
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
			found, _, ok := topOne(t, restored, slab[:dimensions], dimensions)
			if !ok || found != keys[0] {
				t.Errorf("the restored index found %v rather than %v", found, keys[0])
			}
		})
	}
}

// endregion: Serialization

// region: Merge

// The C side owns the donor afterwards, so the Go handle must be emptied rather than freed.
func TestMergeConsumesTheDonor(t *testing.T) {
	const dimensions = 16
	options := angularF32(dimensions)
	index, keys, slab := populated(t, options, 32)

	donor := newIndex(t, options, 32)
	donorKeys := make([]Key, 32)
	for row := range donorKeys {
		donorKeys[row] = KeyFromUint64(uint64(row) + 1000)
	}
	donorSlab := seededRows(32, dimensions)
	if _, err := donor.Add(donorKeys, RowsOf(donorSlab, dimensions), 0, nil); err != nil {
		t.Fatalf("the donor must fill: %v", err)
	}

	if err := index.Merge(donor); err != nil {
		t.Fatalf("the donor must be absorbed: %v", err)
	}
	if err := donor.Close(); err != nil {
		t.Fatalf("closing a consumed donor must be a no-op: %v", err)
	}
	if _, err := donor.Size(); StatusOf(err) != Uninitialized {
		t.Errorf("a consumed donor reported %v", StatusOf(err))
	}

	if size, err := index.Size(); err != nil || size != 64 {
		t.Fatalf("the merged index holds %d of 64 entries: %v", size, err)
	}
	if found, _, _ := topOne(t, index, slab[:dimensions], dimensions); found != keys[0] {
		t.Errorf("the merge lost %v", keys[0])
	}
	if found, _, _ := topOne(t, index, donorSlab[:dimensions], dimensions); found != donorKeys[0] {
		t.Errorf("the merge lost %v", donorKeys[0])
	}
}

// endregion: Merge

// region: Concurrency

// This earns its keep under -race and GODEBUG=cgocheck=1, where it checks that Go pointers
// handed to C survive concurrent collection. Unlike Rust, a Go test may report from a
// goroutine, so no counters are needed.
func TestSearchesWhileInserting(t *testing.T) {
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
		go func(writer int) {
			defer writing.Done()
			for batch := 0; batch < batches; batch++ {
				keys := make([]Key, batchRows)
				for row := range keys {
					keys[row] = KeyFromUint64(uint64(1000 + (writer*batches+batch)*batchRows + row))
				}
				rows := seededRows(batchRows, dimensions)
				if added, err := index.Add(keys, RowsOf(rows, dimensions), 0, nil); err != nil {
					t.Errorf("a concurrent batch failed: %v", err)
					return
				} else if added != batchRows {
					t.Errorf("a concurrent batch published %d of %d rows", added, batchRows)
					return
				}
			}
		}(writer)
	}

	for reader := 0; reader < writers; reader++ {
		reading.Add(1)
		go func() {
			defer reading.Done()
			query := RowsOf(slab[:dimensions], dimensions)
			for {
				select {
				case <-done:
					return
				default:
				}
				outputs := PackedDenseOutputs(1, 10)
				if err := index.SearchInto(query, &outputs, 0, nil); err != nil {
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
	if size, err := index.Size(); err != nil || size != expected {
		t.Fatalf("the index holds %d of %d entries: %v", size, expected, err)
	}
}

// endregion: Concurrency
