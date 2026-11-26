// Arrow columns exported from an arena and viewed back as inputs.
//
// File: golang/arrow_test.go
// Author: Ash Vardanian
// Date: November 18, 2026

package usearch_test

import (
	"errors"
	"slices"
	"testing"

	usearch "github.com/unum-cloud/usearch/golang"
)

func TestArrowColumnsRoundTrip(t *testing.T) {
	options := usearch.DefaultDenseOptions()
	options.Dimensions = 4
	index, err := usearch.DenseInitRam(options, nil)
	if err != nil {
		t.Fatalf("the index must initialize: %v", err)
	}
	defer index.Close()
	keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2)}
	if err := index.Insert(keys, usearch.RowsOf([]float32{1, 0, 0, 0, 0, 1, 0, 0}, 4), nil); err != nil {
		t.Fatalf("the batch must insert: %v", err)
	}
	arena, err := usearch.NewArena(nil)
	if err != nil {
		t.Fatalf("the arena must be made: %v", err)
	}
	defer arena.Close()
	vectors, err := index.Get(keys[1], usearch.ScalarF32, arena)
	if err != nil {
		t.Fatalf("the key must read: %v", err)
	}
	var schema usearch.ArrowSchema
	var array usearch.ArrowDeviceArray
	if err := usearch.RowsViewToArrow(vectors, arena, &schema, &array); err != nil {
		t.Fatalf("the rows must export: %v", err)
	}
	defer schema.Release()
	defer array.Release()
	rows, err := usearch.RowsViewFromArrow(&schema, array.Array())
	if err != nil || rows.Rows() != 1 || rows.Dimensions() != 4 || rows.Scalar() != usearch.ScalarF32 {
		t.Fatalf("viewed %d rows of %d %v, %v", rows.Rows(), rows.Dimensions(), rows.Scalar(), err)
	}
	// The export pins the arena: closing it now frees nothing, and the deferred Close succeeds once
	// the array is released.
	if err := arena.Close(); !errors.Is(err, usearch.StatusContended) {
		t.Fatalf("closing an exported arena must be refused, got %v", err)
	}
	// The rows live in arena, which the search would rewind under them, so it writes into another.
	searched, err := usearch.NewArena(nil)
	if err != nil {
		t.Fatalf("the arena must be made: %v", err)
	}
	defer searched.Close()
	matches, err := index.Search(rows, 1, searched, nil)
	if err != nil {
		t.Fatalf("the exported row must search: %v", err)
	}
	if found, _, _ := matches.Query(0); !slices.Equal(found, keys[1:]) {
		t.Fatalf("the exported row found %v", found)
	}
}
