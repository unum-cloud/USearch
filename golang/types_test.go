// Tests from outside the package, as its users import it.
//
// File: golang/types_test.go
// Author: Ash Vardanian
// Date: May 3, 2023

// The consumer's view: everything here reaches the package through its import path alone, so
// an accidental reliance on an unexported helper breaks this file rather than passing quietly.

package usearch_test

import (
	"errors"
	"runtime"
	"strings"
	"testing"

	usearch "github.com/unum-cloud/usearch/golang"
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
		strings.Repeat("-", 7),
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

// region: Allocations

// A blocking verb allocates its one pinned frame and a refused one only its error, while a deferred
// one adds the Deferred holding the frame and what the runtime spends registering its cleanup;
// nothing else allocates, so a steady loop costs exactly that.
func TestCallsAllocateOnlyTheirFrames(t *testing.T) {
	options := usearch.DefaultDenseOptions()
	options.Dimensions = 4
	index, err := usearch.DenseInitRam(options, nil)
	if err != nil {
		t.Fatalf("the index must initialize: %v", err)
	}
	defer index.Close()
	keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2)}
	rows := usearch.RowsOf([]float32{1, 0, 0, 0, 0, 1, 0, 0}, 4)
	if err := index.Insert(keys, rows, nil); err != nil {
		t.Fatalf("the batch must insert: %v", err)
	}
	executor, err := usearch.ExecutorForRam(nil, usearch.CPUClassUnknown, 1, 0)
	if err != nil {
		t.Fatalf("the executor must spawn: %v", err)
	}
	defer executor.Close()
	arena, err := usearch.NewArena(executor)
	if err != nil {
		t.Fatalf("the arena must be made: %v", err)
	}
	defer arena.Close()
	closed, err := usearch.NewArena(nil)
	if err != nil {
		t.Fatalf("the arena must be made: %v", err)
	}
	closed.Close()
	on, deferred := index.On(executor), index.Deferred(executor)

	for name, verb := range map[string]func(){
		"blocking count":    func() { _, _ = on.Count(keys[0]) },
		"blocking describe": func() { _, _ = on.Describe() },
		"blocking search":   func() { _, _ = on.Search(rows, 2, arena, nil) },
		"refused search":    func() { _, _ = on.Search(rows, 2, closed, nil) },
	} {
		if allocations := testing.AllocsPerRun(100, verb); allocations != 1 {
			t.Errorf("a %s allocated %v times", name, allocations)
		}
	}
	// The holder stands for the Deferred, whose cleanup the runtime registers the same way.
	cleanup := testing.AllocsPerRun(100, func() { runtime.AddCleanup(new([2]uintptr), func(uint) {}, 0) })
	for name, verb := range map[string]func(){
		"deferred count": func() {
			call, _ := deferred.Count(keys[0])
			_, _ = call.Wait()
		},
		"deferred search": func() {
			call, _ := deferred.Search(rows, 2, arena, nil)
			_, _ = call.Wait()
		},
	} {
		if allocations := testing.AllocsPerRun(100, verb); allocations != 1+cleanup {
			t.Errorf("a %s allocated %v times, %v of them registering its cleanup", name, allocations, cleanup)
		}
	}
}

// endregion: Allocations
