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
sudo mv libusearch_static.a /usr/local/lib && sudo mv -t /usr/local/include usearch
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

	usearch "github.com/unum-cloud/USearch/golang"
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

	// An index owns rows; an executor owns the threads a fanned-out verb runs on. Every batch
	// verb takes one, and its residency must match the index's.
	executor, err := usearch.NewRAMExecutor(nil, 0)
	if err != nil {
		log.Fatal(err)
	}
	defer executor.Close()

	// One batch of `entries` rows, packed end to end.
	keys := make([]usearch.Key, entries)
	vectors := make([]float32, entries*dimensions)
	for entry := 0; entry < entries; entry++ {
		keys[entry] = usearch.KeyFromUint64(uint64(entry))
		for dimension := 0; dimension < dimensions; dimension++ {
			vectors[entry*dimensions+dimension] = float32(entry + dimension)
		}
	}
	if _, err := index.Add(executor, keys, usearch.RowsOf(vectors, dimensions),
		usearch.InsertionOptions{}, nil); err != nil {
		log.Fatal(err)
	}

	// Search one query for its three nearest neighbors.
	query := []float32{0.0, 1.0, 2.0}
	outputs := usearch.PackedMatchesSpan(1, 3)
	if err := index.Search(executor, usearch.RowsOf(query, dimensions), &outputs,
		usearch.SearchOptions{}, nil); err != nil {
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
go get github.com/unum-cloud/USearch/golang
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

A search writes into a `MatchesSpan` — three caller-owned slabs sized together, holding a queries-by-`wanted` matrix of keys, the distances paired with them, and one match count per query.
The `wanted` count is the `k` in kANN and the row width of both matrices, and it follows from the slab lengths rather than being asked for, so the space reserved and the count asked for cannot disagree.

```go
// Allocate the slabs for a batch.
outputs := usearch.PackedMatchesSpan(queries.Rows(), 10)

// Or lend slabs you already own, for a sweep that reuses one set of buffers.
outputs, err := usearch.MatchesSpanOf(keys, distances, counts)
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
if _, err := index.Add(executor, keys, vectors, usearch.InsertionOptions{}, nil); err != nil {
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
if err := index.Save(executor, "index.safegraphs"); err != nil {
	log.Fatal(err)
}

restored, err := usearch.NewDenseIndex(&options)
if err != nil {
	log.Fatal(err)
}
if err := restored.Load(executor, "index.safegraphs"); err != nil {
	log.Fatal(err)
}
```

Leaving `Dimensions` at zero defers the geometry to the load, which freezes it from the checkpoint's own headers.
`DenseMetadata` reads that geometry without opening a shard:

```go
metadata, err := usearch.DenseMetadata("index.safegraphs")
fmt.Println(metadata.Dimensions, metadata.Metric, metadata.Quantization)
```

The result carries only the four fields a checkpoint froze, so `ValuesPerKey` and the residency stay yours to set in `DenseOptions` before constructing a handle.

There is no separate view verb.
Opening a checkpoint in place is a residency, and every mutating verb on such a handle fails with `ReadOnly`:

```go
options.Residency = usearch.NVMeResidency{Paths: []string{"/dev/nvme0n1"}}
```

## Residency

Where the payload lives decides which engine traverses it, and what loading a checkpoint means.

```go
// CPU threads over volatile memory - the concurrent baseline.
options.Residency = usearch.RAMResidency{} // every domain the process may use

// GPU warps over accelerator memory.
options.Residency = usearch.VRAMResidency{} // one free device

// Pluggable I/O over a persisted checkpoint, read-only and sized past RAM.
options.Residency = usearch.NVMeResidency{Paths: []string{"/dev/nvme0n1"}}
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

`Get` copies the entries stored under one key back out, converting into the slab's own scalar kind and reporting how many rows it wrote:

```go
recovered := make([]float32, count*dimensions)
written, _ := index.Get(usearch.KeyFromUint64(42), usearch.RowsOf(recovered, dimensions))
```

## Filtered Search

`SearchIf` admits only the keys a predicate approves.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.

```go
admits := func(key usearch.Key) bool {
	value, integral := key.Uint64()
	return integral && value%2 == 0
}

outputs := usearch.PackedMatchesSpan(queries.Rows(), 10)
err := index.SearchIf(executor, queries, &outputs, usearch.SearchOptions{}, nil, admits)
```

The predicate runs on the library's worker threads, so it must be safe for concurrent use.

## Late Interaction

`MaxSim` scores multi-row queries against multi-row entries: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `ValuesPerKey` above one exists for, and the outputs are sized over documents rather than rows.
It runs on an inner-product index alone, since a sum of maxima needs logits rather than distances.

```go
options.Metric = usearch.Inner
options.ValuesPerKey = 512

documents := usearch.PackedMatchesSpan(queries.Rows()/tokensPerQuery, 10)
err := index.MaxSim(executor, usearch.RowsOf(tokens, dimensions), tokensPerQuery,
	&documents, usearch.SearchOptions{}, nil)
```

The `multiply` and `attend` verbs of the C interface are not bound here.

## Refinement

`Refine` re-runs the neighbor selection over entries already in the graph, returning how many links it rewrote.
A batch built at a low expansion factor can be brought up to the quality of a slower build afterwards, and the count reaching zero is what settled looks like:

```go
for {
	changed, err := index.Refine(executor, usearch.InsertionOptions{ExpansionFactor: 128}, nil)
	if err != nil {
		log.Fatal(err)
	}
	if changed == 0 {
		break
	}
}
```

## Telemetry

Both insertion and both search forms report traversal telemetry when asked.
Each call overwrites the struct rather than accumulating, so a sweep sums them itself.

```go
var stats, total usearch.DenseStats
for _, batch := range batches {
	if _, err := index.Add(executor, batch.keys, batch.vectors, usearch.InsertionOptions{}, &stats); err != nil {
		log.Fatal(err)
	}
	total.Accumulate(stats)
}
fmt.Println(total.VisitedMembers, total.ComputedDistances, total.Restarts)
```

The candidate-pool breadth rides each call as the `options` argument rather than the handle, so a recall sweep runs against one immutable index with no mutation between points.
`Add` and `Refine` take an `InsertionOptions`, the three read-only walks a `SearchOptions`, and the two stay separate types so a knob either side gains reaches that side alone.

## Concurrency

A `DenseIndex` may be shared across goroutines with no lock of your own.
Searches read only entries published behind the visibility watermark and never write to the graph, while insertions synchronize through per-node locks and publish in order, so the two overlap freely.

```go
var work sync.WaitGroup
for worker := 0; worker < workers; worker++ {
	work.Add(1)
	go func() {
		defer work.Done()
		outputs := usearch.PackedMatchesSpan(queries.Rows(), 10)
		if err := index.Search(executor, queries, &outputs, usearch.SearchOptions{}, nil); err != nil {
			log.Print(err)
		}
	}()
}
work.Wait()
```

Internal parallelism belongs to an `Executor` rather than to a residency or a handle: `NewRAMExecutor(nil, n)` fixes the worker count, and zero takes 7/8 of the logical cores, since full occupancy starves the ordered publication path.
There is a factory per residency, each taking only what its own hardware has - `NewRAMExecutor` and `NewNVMeExecutor` take a thread count, while `NewVRAMExecutor` takes none, one host thread driving one stream per device.
A residency names hardware and nothing else, so one pool serves several indexes and its residency must match theirs.

Serialization is the one exception to the free overlap — saving demands a mutation-free window, though concurrent readers keep searching throughout.
