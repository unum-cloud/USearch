// Examples from outside the package, as its users write them, each checked against its output.
//
// File: golang/example_test.go
// Author: Ash Vardanian
// Date: November 18, 2026

package usearch_test

import (
	"fmt"
	"log"

	usearch "github.com/unum-cloud/usearch/golang"
)

// region: Examples

// The Quickstart of golang/README.md, whose main function runs these same lines.
func Example() {
	options := usearch.DefaultDenseOptions()
	options.Dimensions = 4
	options.Metric = usearch.DenseMetricAngular
	options.Quantization = usearch.ScalarI8
	index, err := usearch.DenseInitRam(options, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()
	if err := index.Reserve(1_000); err != nil { // optional capacity hint; inserts grow on demand
		log.Fatal(err)
	}

	keys := []usearch.Key{usearch.KeyFromUint64(42), usearch.KeyFromUint64(43)}
	slab := []float32{0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2}
	if err := index.Insert(keys, usearch.RowsOf(slab, 4), nil); err != nil {
		log.Fatal(err)
	}

	executor, err := usearch.ExecutorForRam(nil, usearch.CPUClassUnknown, 0)
	if err != nil {
		log.Fatal(err)
	}
	defer executor.Close()
	arena, err := usearch.NewArena(executor)
	if err != nil {
		log.Fatal(err)
	}
	defer arena.Close()
	query := []float32{0.2, 0.1, 0.2, 0.1}
	matches, err := index.On(executor).Search(usearch.RowsOf(query, 4), 10, arena, nil) // on an executor, into its arena
	if err != nil {
		log.Fatal(err)
	}
	localArena, err := usearch.NewArena(nil)
	if err != nil {
		log.Fatal(err)
	}
	defer localArena.Close()
	local, err := index.Search(usearch.RowsOf(query, 4), 10, localArena, nil) // the calling goroutine, into host pages
	if err != nil {
		log.Fatal(err)
	}

	found, _ := matches.Query(0)
	foundLocally, _ := local.Query(0)
	fmt.Println(found[0], foundLocally[0])
	// Output: 00000000-0000-0000-0000-00000000002a 00000000-0000-0000-0000-00000000002a
}

func ExampleKey_String() {
	fmt.Println(usearch.KeyFromUint64(42))
	// Output: 00000000-0000-0000-0000-00000000002a
}

func ExampleParseKey() {
	key, err := usearch.ParseKey("00112233-4455-6677-8899-aabbccddeeff")
	if err != nil {
		log.Fatal(err)
	}
	integer, integral := key.Uint64()
	fmt.Println(key, integer, integral)
	// Output: 00112233-4455-6677-8899-aabbccddeeff 0 false
}

func ExampleDenseIndex() {
	options := usearch.DefaultDenseOptions()
	options.Dimensions = 3
	options.Metric = usearch.DenseMetricEuclidean

	index, err := usearch.DenseInitRam(options, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	if err := index.Reserve(3); err != nil {
		log.Fatal(err)
	}
	rows := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
	keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2), usearch.KeyFromUint64(3)}
	if err := index.Insert(keys, usearch.RowsOf(rows, 3), nil); err != nil {
		log.Fatal(err)
	}

	arena, err := usearch.NewArena(nil)
	if err != nil {
		log.Fatal(err)
	}
	defer arena.Close()
	matches, err := index.Search(usearch.RowsOf([]float32{0, 1, 0}, 3), 2, arena, nil)
	if err != nil {
		log.Fatal(err)
	}
	found, _ := matches.Query(0) // valid until the next call given arena
	fmt.Println(found[0])
	// Output: 00000000-0000-0000-0000-000000000002
}

func ExampleDenseIndexOn_Deferred() {
	options := usearch.DefaultDenseOptions()
	options.Dimensions = 3
	options.Metric = usearch.DenseMetricEuclidean

	index, err := usearch.DenseInitRam(options, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()
	executor, err := usearch.ExecutorForRam(nil, usearch.CPUClassUnknown, 0)
	if err != nil {
		log.Fatal(err)
	}
	defer executor.Close()

	if err := index.Reserve(3); err != nil {
		log.Fatal(err)
	}
	rows := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
	keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2), usearch.KeyFromUint64(3)}
	if err := index.On(executor).Insert(keys, usearch.RowsOf(rows, 3), nil); err != nil {
		log.Fatal(err)
	}

	arena, err := usearch.NewArena(executor)
	if err != nil {
		log.Fatal(err)
	}
	defer arena.Close()
	call, err := index.Deferred(executor).Search(usearch.RowsOf([]float32{0, 0, 1}, 3), 1, arena, nil)
	if err != nil {
		log.Fatal(err)
	}
	matches, err := call.Wait() // a view borrowing arena, as a blocking call's does
	if err != nil {
		log.Fatal(err)
	}
	found, _ := matches.Query(0)
	fmt.Println(found[0])
	// Output: 00000000-0000-0000-0000-000000000003
}

// Mirrors the Sequence Indexes snippet of golang/README.md, adding the arena it borrows and the
// checks it skips.
func ExampleSequenceIndex() {
	index, err := usearch.SequenceInitRam(usearch.SequenceOptions{Metric: usearch.SequenceMetricLevenshteinUTF8}, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	words := usearch.SequencesOf([]byte("kitten"), []byte("sitting"), []byte("mitten"))
	keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2), usearch.KeyFromUint64(3)}
	if err := index.Insert(keys, words, nil); err != nil {
		log.Fatal(err)
	}

	arena, err := usearch.NewArena(nil)
	if err != nil {
		log.Fatal(err)
	}
	defer arena.Close()
	matches, err := index.Search(usearch.SequencesOf([]byte("sittin")), 1, arena, nil)
	if err != nil {
		log.Fatal(err)
	}
	found, distances := matches.Query(0)
	fmt.Println(found[0], distances[0])
	// Output: 00000000-0000-0000-0000-000000000002 1
}

// Mirrors the Shape Indexes snippet of golang/README.md, adding the arena it borrows and the checks
// it skips.
func ExampleShapesIndex() {
	index, err := usearch.ShapesInitRam(usearch.DefaultShapesOptions(), nil)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	// A 2-atom and a 3-atom molecule, x, y and z interleaved.
	molecules := usearch.ShapesOf([]float32{0, 0, 0, 1.1, 0, 0}, []float32{0, 0, 0, 1.5, 0, 0, 0, 1.5, 0})
	keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2)}
	if err := index.Insert(keys, molecules, nil); err != nil {
		log.Fatal(err)
	}

	arena, err := usearch.NewArena(nil)
	if err != nil {
		log.Fatal(err)
	}
	defer arena.Close()
	matches, err := index.Search(usearch.ShapesOf([]float32{0, 0, 0, 1.0, 0, 0}), 1, arena, nil)
	if err != nil {
		log.Fatal(err)
	}
	found, _ := matches.Query(0)
	fmt.Println(found[0])
	// Output: 00000000-0000-0000-0000-000000000001
}

// Mirrors the Geospatial Indexes snippet of golang/README.md, adding the arena it borrows and the
// checks it skips.
func ExampleGeospatialIndex() {
	index, err := usearch.GeospatialInitRam(usearch.GeospatialOptions{}, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	stops := []usearch.GeospatialObservation{
		usearch.GeospatialAt(51.5308, -0.1238), // every other field unrecorded, NaN
		usearch.GeospatialAt(51.5320, -0.1233),
		usearch.GeospatialAt(51.5033, -0.1196),
	}
	keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2), usearch.KeyFromUint64(3)}
	if err := index.Insert(keys, stops, nil); err != nil {
		log.Fatal(err)
	}

	arena, err := usearch.NewArena(nil)
	if err != nil {
		log.Fatal(err)
	}
	defer arena.Close()
	nearby, err := index.Within(stops[:1], usearch.GeospatialRelationNear, 500, arena, nil)
	if err != nil {
		log.Fatal(err)
	}
	found, _ := nearby.Query(0) // every entry within 500 metres, with its chord metres
	fmt.Println(len(found))
	// Output: 2
}

// endregion: Examples
