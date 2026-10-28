// The consumer's view: everything here reaches the package through its import path alone, so
// an accidental reliance on an unexported helper breaks this file rather than passing quietly.

package usearch_test

import (
	"errors"
	"fmt"
	"log"
	"testing"

	usearch "github.com/unum-cloud/USearch/golang"
)

// region: Introspection

func TestPublicSurfaceNamesTheLinkedBuild(t *testing.T) {
	t.Parallel()
	if usearch.CapabilitiesCompiled() == "" {
		t.Error("the build must name the kernels it compiled")
	}
	if usearch.CapabilitiesAvailable() == "" {
		t.Error("the machine must name the kernels it offers")
	}
}

// endregion: Introspection

// region: Fuzzing

// Parsing is the one place this package reads text a caller did not write, so the corpus is
// where malformed renderings are cheapest to find.
func FuzzParseKey(f *testing.F) {
	for _, seed := range []string{
		"00000000-0000-0000-0000-00000000002a",
		"0000000000000000000000000000002a",
		"00112233-4455-6677-8899-AABBCCDDEEFF",
		"-------",
		"",
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, text string) {
		key, err := usearch.ParseKey(text)
		if err != nil {
			if !errors.Is(err, usearch.StatusMalformed) {
				t.Fatalf("%q was refused as %v", text, err)
			}
			if key != usearch.NilKey {
				t.Fatalf("%q was refused yet yielded %v", text, key)
			}
			return
		}
		// Whatever parsed renders canonically, and the canonical rendering parses back to it.
		rendered := key.String()
		again, err := usearch.ParseKey(rendered)
		if err != nil || again != key {
			t.Fatalf("%q rendered as %q and parsed back as %v: %v", text, rendered, again, err)
		}
	})
}

// endregion: Fuzzing

// region: Benchmarks

func BenchmarkParseKey(b *testing.B) {
	const rendered = "00112233-4455-6677-8899-aabbccddeeff"
	b.ReportAllocs()
	for b.Loop() {
		if _, err := usearch.ParseKey(rendered); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkKeyString(b *testing.B) {
	key := usearch.KeyFromUint64(42)
	b.ReportAllocs()
	for b.Loop() {
		_ = key.String()
	}
}

// endregion: Benchmarks

// region: Examples

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
	found, _, err := matches.Row(0) // valid until the next call given arena
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(found[0])
	// Output: 00000000-0000-0000-0000-000000000002
}

func ExampleDenseIndex_Deferred() {
	options := usearch.DefaultDenseOptions()
	options.Dimensions = 3
	options.Metric = usearch.DenseMetricEuclidean

	index, err := usearch.DenseInitRam(options, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()
	executor, err := usearch.ExecutorForRam(nil, usearch.CPUClassUnknown, 0, 0)
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
	call := index.Deferred(executor).Search(usearch.RowsOf([]float32{0, 0, 1}, 3), 1, arena, nil)
	matches, err := call.Wait() // a view borrowing arena, as a blocking call's does
	if err != nil {
		log.Fatal(err)
	}
	found, _, err := matches.Row(0)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(found[0])
	// Output: 00000000-0000-0000-0000-000000000003
}

// endregion: Examples
