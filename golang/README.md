# USearch for Go

The Go SDK links the pre-instantiated C library and consumes it through the stable C headers, so nothing but the vocabulary below crosses the boundary.
It exposes the dense modality — Vector Search over equidimensional embeddings — and nothing else.

Every verb is a batch over strided matrices, so a memory-mapped dataset reaches the index with no reshape and no copy, and a single vector is a one-row matrix.

## Installation

### Linux

Download and install the Debian package from the latest release.
Substitute `<release_tag>`, `<arch>`, and `<usearch_version>` with your settings.

```sh
wget https://github.com/unum-cloud/USearch/releases/download/<release_tag>/usearch_linux_<arch>_<usearch_version>.deb
dpkg -i usearch_linux_<arch>_<usearch_version>.deb
```

### Windows

Run the `winlibinstaller.bat` script from the main repository in the folder where you will run `go run`.
This installs the USearch library and its headers into the same folder where the script was run.

```sh
.\usearch\winlibinstaller.bat
```

### macOS

Download and unpack the zip archive from the latest release.
Move the library and the header directory into their respective folders.

```sh
wget https://github.com/unum-cloud/USearch/releases/download/<release_tag>/usearch_macos_<arch>_<usearch_version>.zip
unzip usearch_macos_<arch>_<usearch_version>.zip
sudo mv libusearch_c.dylib /usr/local/lib && sudo mv -t /usr/local/include usearch
```

## Quickstart

1. Create a `go.mod` file:

```
module usearch_example

go 1.24
```

2. Create an `example.go`:

```go
package main

import (
	"fmt"
	"log"

	usearch "github.com/unum-cloud/usearch/golang"
)

func main() {
	const dimensions = 3
	const entries = 100

	options := usearch.DefaultDenseOptions()
	options.Dimensions = dimensions
	options.Metric = usearch.Angular
	options.Quantization = usearch.F32 // or BF16, F16, E5M2, E4M3, E3M2, E2M3, I8, U8, U1x8

	index, err := usearch.NewDenseIndex(&options)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	if err := index.Reserve(entries); err != nil {
		log.Fatal(err)
	}

	// One batch of `entries` rows, packed end to end.
	keys := make([]usearch.Key, entries)
	vectors := make([]float32, entries*dimensions)
	for entry := 0; entry < entries; entry++ {
		keys[entry] = usearch.KeyFromUint64(uint64(entry))
		for dimension := 0; dimension < dimensions; dimension++ {
			vectors[entry*dimensions+dimension] = float32(entry + dimension)
		}
	}
	if _, err := index.Add(keys, usearch.RowsOf(vectors, dimensions), 0, nil); err != nil {
		log.Fatal(err)
	}

	// Search one query for its three nearest neighbors.
	query := []float32{0.0, 1.0, 2.0}
	outputs := usearch.PackedDenseOutputs(1, 3)
	if err := index.SearchInto(usearch.RowsOf(query, dimensions), &outputs, 0, nil); err != nil {
		log.Fatal(err)
	}
	found, distances := outputs.MatchesOf(0)
	for rank := range found {
		fmt.Println(found[rank], distances[rank])
	}
}
```

Always call `Reserve(capacity)` before the first write.

3. Get USearch:

```sh
go get github.com/unum-cloud/usearch/golang
```

4. Run:

```sh
go run example.go
```

## Matrices

A `Matrix` borrows a caller-owned slab as rows of equal width, and it is the shape every batch argument takes.
`RowsOf` derives the row count and the byte pitch from a packed slice; `StridedRowsOf` takes a pitch wider than the row, which is how a memory-mapped dataset with padded rows arrives.

```go
vectors := usearch.RowsOf(embeddings, 768)                    // packed
padded := usearch.StridedRowsOf(mapped, rows, 768, 1024*4)    // pitch of 1024 floats
```

The element type of the slice names the scalar kind, and it is independent of the index's own quantization — the library converts on ingest and on export.
A `[]float32` query searches an `I8` index without a conversion pass of your own.

Sub-byte kinds count logical dimensions rather than bytes, so 128 dimensions of `U1x8Scalar` occupy sixteen bytes per row:

```go
packed := make([]usearch.U1x8Scalar, rows*128/8)
vectors := usearch.RowsOf(packed, 128) // 128 dimensions, 16 bytes per row
```

## Outputs

A search writes into three slabs sized together — a queries-by-`wanted` matrix of keys, the distances paired with them, and one match count per query.
The `wanted` count is the `k` in kANN and the row width of both matrices, so it is fixed by the constructor and cannot be restated afterwards.

```go
// Allocate the slabs for a batch.
outputs := usearch.PackedDenseOutputs(queries.Rows(), 10)

// Or adopt slabs you already own, for a sweep that reuses one set of buffers.
outputs, err := usearch.DenseOutputsOf(keys, distances, counts, 10)
```

`MatchesOf` borrows one query's results, trimmed to the count that query reported, and copies nothing:

```go
for query := uint(0); query < outputs.Queries(); query++ {
	found, distances := outputs.MatchesOf(query)
	for rank := range found {
		fmt.Println(query, found[rank], distances[rank])
	}
}
```

## Keys

A key is 128 bits of opaque identity, rendered as a lowercase dashed UUID.
Keys need not be unique: several entries may share one, and `Count` reports how many.

```go
key := usearch.KeyFromUint64(42)
fmt.Println(key)                 // 00000000-0000-0000-0000-00000000002a
value, integral := key.Uint64()  // 42, true

parsed, err := usearch.ParseKey("00000000-0000-0000-0000-00000000002a")
```

Keys satisfy `encoding.TextMarshaler` and `encoding.TextUnmarshaler`, so they round-trip through JSON as those same strings.

## Errors

Every failure carries a status shared with the C interface, so a fault this binding catches before the boundary reports the identity the engine would have reported for the same fault.

```go
if _, err := index.Add(keys, vectors, 0, nil); err != nil {
	if usearch.StatusOf(err) == usearch.ShapeMismatch {
		// the keys and the rows disagree
	}

	var failure *usearch.Error
	if errors.As(err, &failure) {
		fmt.Println(failure.Status, failure.Message)
	}
}
```

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout.
A `.safegraphs` path is one self-describing monolith; a `.safegraphs.json` path is a manifest cataloging many, with every shard path resolved relative to it.
In-memory buffers hold single monoliths only.

```go
if err := index.Save("index.safegraphs"); err != nil {
	log.Fatal(err)
}

restored, err := usearch.NewDenseIndex(&options)
if err != nil {
	log.Fatal(err)
}
if err := restored.Load("index.safegraphs"); err != nil {
	log.Fatal(err)
}
```

Leaving `Dimensions` at zero defers the geometry to the load, which freezes it from the checkpoint's own headers.
`DenseMetadata` reads that geometry without opening a shard:

```go
metadata, err := usearch.DenseMetadata("index.safegraphs")
fmt.Println(metadata.Dimensions, metadata.Metric, metadata.Quantization)
```

There is no separate view verb.
Opening a checkpoint in place is a residency, and every mutating verb on such a handle fails with `ReadOnly`:

```go
options.Residency = usearch.Residency{Kind: usearch.NVMe, IOMode: usearch.DirectIO}
```

## Residency

Where the payload lives decides which engine traverses it, and what loading a checkpoint means.

```go
// CPU threads over volatile memory - the concurrent baseline.
options.Residency = usearch.Residency{Kind: usearch.RAM, Threads: 0} // zero takes 7/8 of the cores

// GPU warps over accelerator memory.
options.Residency = usearch.Residency{Kind: usearch.VRAM, Device: usearch.FirstFreeDevice}

// Pluggable I/O over a persisted checkpoint, read-only and sized past RAM.
options.Residency = usearch.Residency{Kind: usearch.NVMe, IOMode: usearch.DirectIO}
```

## Index Introspection

```go
dimensions, _ := index.Dimensions()   // the frozen vector width
size, _ := index.Size()               // the number of stored entries
capacity, _ := index.Capacity()       // the reserved capacity
connectivity, _ := index.Connectivity()
usage, _ := index.MemoryUsage()
capabilities, _ := index.Capabilities() // the codename dispatch picked on this machine

found, _ := index.Contains(usearch.KeyFromUint64(42))
count, _ := index.Count(usearch.KeyFromUint64(42)) // entries sharing one key

fmt.Println(usearch.Version(), usearch.CapabilitiesAvailable())
```

## Filtered Search

`SearchIfInto` admits only the keys a predicate approves.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.

```go
admits := func(key usearch.Key) bool {
	value, integral := key.Uint64()
	return integral && value%2 == 0
}

outputs := usearch.PackedDenseOutputs(queries.Rows(), 10)
err := index.SearchIfInto(queries, &outputs, 0, nil, admits)
```

The predicate runs on the library's worker threads, so it must be safe for concurrent use.

## Telemetry

Both insertion and both search forms report traversal telemetry when asked.
Each call overwrites the struct rather than accumulating, so a sweep sums them itself.

```go
var stats, total usearch.DenseStats
for _, batch := range batches {
	if _, err := index.Add(batch.keys, batch.vectors, 0, &stats); err != nil {
		log.Fatal(err)
	}
	total.Accumulate(stats)
}
fmt.Println(total.VisitedMembers, total.ComputedDistances, total.Restarts)
```

The candidate-pool breadth rides each call as the `expansion` argument rather than the handle, so a recall sweep runs against one immutable index with no mutation between points.

## Merging

Growing an index by absorbing another is cheaper than re-inserting its vectors one by one.
The donor is consumed whether the call succeeds or fails, so its handle is emptied and closing it becomes a no-op.

```go
if err := index.Merge(donor); err != nil {
	log.Fatal(err)
}
```

## Concurrency

A `DenseIndex` may be shared across goroutines with no lock of your own.
Searches read only entries published behind the visibility watermark and never write to the graph, while insertions synchronize through per-node locks and publish in order, so the two overlap freely.

```go
var work sync.WaitGroup
for worker := 0; worker < workers; worker++ {
	work.Add(1)
	go func() {
		defer work.Done()
		outputs := usearch.PackedDenseOutputs(queries.Rows(), 10)
		if err := index.SearchInto(queries, &outputs, 0, nil); err != nil {
			log.Print(err)
		}
	}()
}
work.Wait()
```

Internal parallelism is a property of the residency rather than a mutable setting: `Residency{Kind: RAM, Threads: n}` fixes the worker count at construction, and zero takes 7/8 of the logical cores, since full occupancy starves the ordered publication path.

Serialization is the one exception to the free overlap — saving demands a mutation-free window, though concurrent readers keep searching throughout.
