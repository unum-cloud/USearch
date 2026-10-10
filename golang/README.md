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
	options.Metric = usearch.DenseMetricAngular
	options.Quantization = usearch.ScalarF32 // or ScalarBF16, ScalarF16, ScalarE5M2, ScalarE4M3, ScalarE3M2, ScalarE2M3, ScalarI8, ScalarU8, ScalarU1x8

	index, err := usearch.DenseInitRam(options, nil) // nil places a shard on every NUMA node
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	if err := index.Reserve(entries); err != nil {
		log.Fatal(err)
	}

	// One batch of `entries` rows, packed end to end, inserted on the calling goroutine.
	keys := make([]usearch.Key, entries)
	vectors := make([]float32, entries*dimensions)
	for entry := 0; entry < entries; entry++ {
		keys[entry] = usearch.KeyFromUint64(uint64(entry))
		for dimension := 0; dimension < dimensions; dimension++ {
			vectors[entry*dimensions+dimension] = float32(entry + dimension)
		}
	}
	if err := index.Insert(keys, usearch.RowsOf(vectors, dimensions), nil); err != nil {
		log.Fatal(err)
	}

	// An index owns rows; an executor owns the threads a verb runs on. The same verb runs there
	// through `index.On(executor)`, whose residency must match the index's.
	executor, err := usearch.ExecutorForRam(nil, 0, 0)
	if err != nil {
		log.Fatal(err)
	}
	defer executor.Close()

	// Search one query for its three nearest neighbors.
	query := []float32{0.0, 1.0, 2.0}
	matches, err := index.On(executor).Search(usearch.RowsOf(query, dimensions), 3, nil)
	if err != nil {
		log.Fatal(err)
	}
	found, distances := matches.Row(0)
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

## Receivers

The receiver chooses where a verb runs and how it completes, and the verb stays the same.
The index runs it on the calling goroutine, `index.On(executor)` blocks on an executor, and `index.Deferred(executor)` submits it there and answers a `*usearch.Deferred`:

```go
matches, err := index.Search(queries, 10, nil)              // the calling goroutine
matches, err := index.On(executor).Search(queries, 10, nil) // blocking, after every earlier submission
call := index.Deferred(executor).Search(queries, 10, nil)   // *usearch.Deferred[usearch.MatchesSpan]
```

`DenseIndexOn` and `DenseIndexDeferredOn` offer the same verbs with the same arguments; only the blocking ones add `Contains`, `Cursor` and `SearchStream`, whose every pull blocks.
Constructors address no index yet, so no receiver offers them; the ones the C interface lets fan out take an `Executor` in their config, and run on the calling goroutine without one.

## Deferred Calls

A `Deferred` owns everything the library reads or writes until its call is final: the outcome, the pinned inputs, the outputs and the arena.

```go
call := index.Deferred(executor).Search(queries, 10, nil)
if call.Status() == usearch.StatusDeferred {
	// other work
}
matches, err := call.Wait()
```

Go needs nothing more: a goroutine that wants to overlap the wait simply calls `Wait` there, and a `select` waits on a channel that goroutine closes.
`Status` reads the outcome without blocking, and `Wait` blocks until the call is final and hands back what it produced, as many times as it is asked.
`Close` waits the same way and frees whatever `Wait` has not handed out, so a call no one reads is closed rather than waited on; it is idempotent and safe after `Wait`.
A failure this binding catches before the call crosses arrives through the same `Status` and `Wait`, as the engine's own refusals do.
A `Deferred` dropped without `Wait` or `Close` keeps its pins, which the runtime reports as a leaked `runtime.Pinner`.

An executor without threads of its own may hold a deferred call until its next `Drain`, blocking call or `Close`, so drain it before a `Wait` there.
A submission finding the executor's deferred table full, sized by `deferredLimit`, finishes at once with `StatusContended`.
Calls depending on each other are ordered by waiting on the first before submitting the second.
The slices a deferred verb is handed must not change until it is final, since the library reads and writes them until then.

## Matrices

A `RowsView` borrows a caller-owned slab as rows of equal width, and it is the shape every batch argument takes.
`RowsOf` derives the row count and the byte pitch from a packed slice; `StridedRowsOf` takes a pitch wider than the row, which is how a memory-mapped dataset with padded rows arrives.

```go
vectors := usearch.RowsOf(embeddings, 768)                    // packed
padded := usearch.StridedRowsOf(mapped, rows, 768, 1024*4)    // pitch of 1024 floats
```

The element type of the slice names the scalar kind, and it is independent of the index's own quantization — the library converts on ingest and on export.
A `[]float32` query searches an `ScalarI8` index without a conversion pass of your own.

Sub-byte kinds count logical dimensions rather than bytes, so 128 dimensions of `U1x8Scalar` occupy sixteen bytes per row:

```go
packed := make([]usearch.U1x8Scalar, rows*128/8)
vectors := usearch.RowsOf(packed, 128) // 128 dimensions, 16 bytes per row
```

## Configs and Outputs

Every verb with optional knobs ends with a pointer to a config named after its C argument structure - `DenseInsertConfig`, `DenseSearchConfig`, `DenseMultiplyConfig` - where `nil` and every zero field derive what the C interface derives.
So `nil` makes the bare call, and naming a field reaches a knob:

```go
var stats usearch.DenseTraversalStats
matches, err := index.On(executor).Search(queries, 10, &usearch.DenseSearchConfig{
	ExpansionFactor: 128,
	Stats:           &stats,
})
fmt.Println(stats.ComputedDistances, "distances computed")
```

A search returns a `MatchesSpan` - three slabs sized together, holding a queries-by-`wanted` matrix of keys, the distances paired with them, and one match count per query.
`Row` borrows one query's results, trimmed to the count that query reported, and copies nothing:

```go
for query := uint(0); query < matches.Queries(); query++ {
	found, distances := matches.Row(query)
	for rank := range found {
		fmt.Println(query, found[rank], distances[rank])
	}
}
```

A sweep that must not allocate passes a span back as `Matches`, or lends slabs it already owns, whose rows may be wider than `wanted`:

```go
matches, _ := index.Search(batches[0], 10, nil)
for _, batch := range batches[1:] {
	matches, _ = index.Search(batch, 10, &usearch.DenseSearchConfig{Matches: &matches}) // equal batches
}

lent, err := usearch.MatchesSpanOf(keys, distances, counts)
```

The candidate-pool breadth rides each call as `ExpansionFactor` rather than the handle, so a recall sweep runs against one immutable index with no mutation between points.

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
A `Status` is itself an `error`, so `errors.Is` matches any failure carrying it:

```go
if err := index.Insert(keys, vectors, nil); err != nil {
	if errors.Is(err, usearch.StatusShapeMismatch) {
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
if err := index.On(executor).Save("index.safegraphs", nil); err != nil {
	log.Fatal(err)
}

restored, err := usearch.DenseLoadRam("index.safegraphs", &usearch.DenseLoadRamConfig{Executor: executor})
if err != nil {
	log.Fatal(err)
}
defer restored.Close()

monolith, err := index.SaveBuffer(nil) // sized by the call itself, see Arenas
defer monolith.Close()
copied, err := usearch.DenseLoadRamBuffer(monolith.Bytes, nil)
```

Loading creates the index, freezing its geometry from the checkpoint's own headers.
`DenseMetadataRead` reads that geometry without opening a shard:

```go
metadata, err := usearch.DenseMetadataRead("index.safegraphs", nil) // a nil config reads on the calling goroutine
fmt.Println(metadata.Dimensions, metadata.Metric, metadata.Quantization)
```

The result carries only the four fields a checkpoint froze, so `ValuesPerKey` and the placement stay yours to set in the load config.
Serving a checkpoint in place, rather than copying it, yields a read-only handle whose mutating verbs fail with `StatusReadOnly` - through a memory mapping with `DenseViewRam`, or from drives with `DenseViewNvme`:

```go
mapped, err := usearch.DenseViewRam("index.safegraphs")
served, err := usearch.DenseViewNvme("index.safegraphs", &usearch.DenseViewNvmeConfig{
	Executor: drives,
	Paths:    []string{"/dev/nvme0n1"},
})
```

Until NVMe checkpoints carry keys, `DenseViewNvme` answers `StatusUnsupported`.

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each takes only the placement its own hardware has, one shard per entry, and an empty list derives it.

```go
inRam, err := usearch.DenseInitRam(options, []int32{0, 1})        // one graph per socket
inVram, err := usearch.DenseInitVram(options, []int32{0, 1, 2, 3}) // sharded, never replicated
onGpu, err := usearch.DenseLoadVram("index.safegraphs", &usearch.DenseLoadVramConfig{Devices: []int32{0}})
```

An NVMe index is only ever viewed: external-memory graphs are never modified in place.

## Index Introspection

```go
description, _ := index.Describe()
fmt.Println(description.Metadata.Dimensions, description.Size, description.Capacity,
	description.MemoryUsage, description.Capabilities) // the codename dispatch picked on this machine

found, _ := index.Contains(usearch.KeyFromUint64(42))
count, _ := index.Count(usearch.KeyFromUint64(42)) // entries sharing one key

reachability, _ := index.Health()
fmt.Println(reachability.Reachable, "of", reachability.Audited, "entries reachable")

fmt.Println(usearch.Version(), usearch.CapabilitiesAvailable())
```

`Health` audits how much of the graph a search reaches from its landmarks; entries it misses stay unfound until a `Refine` re-links them.

`Get` exports every entry stored under one key, converted to the scalar kind you name, in one call however many there turn out to be:

```go
vectors, _ := index.Get(usearch.KeyFromUint64(42), usearch.ScalarF32, nil)
defer vectors.Close()
values := usearch.DenseValues[float32](vectors) // vectors.Rows() rows of vectors.Columns(), vectors.Stride() bytes apart
```

A cursor walks the contents as a sequential row scan, landing rows in the scalar kind and width of the matrix you hand it:

```go
cursor, _ := index.On(executor).Cursor()
defer cursor.Close()
keys := make([]usearch.Key, 1024)
rows := make([]float32, 1024*768)
for {
	written, err := cursor.NextBatch(keys, usearch.RowsOf(rows, 768))
	if err != nil || written == 0 {
		break
	}
}
```

## Arenas

`Get`, `Multiply` and `SaveBuffer` return outputs whose size only the call discovers, so they land in an `Arena` the library allocates.
Pass none and the result owns a fresh arena until its `Close`, outliving every later call.
Pass an arena of your own and the call rewinds and refills it, so a steady loop allocates once - but each result stays valid only until that arena serves another call:

```go
var arena usearch.Arena // the zero value holds nothing until a call creates it
defer arena.Close()
for _, key := range keys {
	vectors, err := index.Get(key, usearch.ScalarF32, &arena)
	if err != nil {
		log.Fatal(err)
	}
	consume(usearch.DenseValues[float32](vectors)) // valid until the next call that takes &arena
}
```

An arena serves one call at a time; a second call overlapping it fails with `StatusContended`.
The slices a result borrows from an arena are invisible to the collector, so reading them after its `Close` reads freed memory.

## Filtering and Progress

A `Predicate` in the `DenseSearchConfig` admits only the keys it approves.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.

```go
admits := func(key usearch.Key) bool {
	value, integral := key.Uint64()
	return integral && value%2 == 0
}
matches, err := index.On(executor).Search(queries, 10, &usearch.DenseSearchConfig{Predicate: admits})
```

The predicate runs on the library's worker threads, so it must be safe for concurrent use.

Every long verb's config takes a `Progress` handler, called with the units finished so far and the units the call finishes if it runs to the end; answering `false` stops it wherever the verb can stop, which an `Insert` cannot.

```go
deadline := time.Now().Add(30 * time.Second)
changed, err := index.On(executor).Refine(&usearch.DenseRefineConfig{
	ExpansionFactor: 128,
	Progress:        func(completed, total uint) bool { return time.Now().Before(deadline) },
})
```

A handler never runs on two threads at once, so it may carry a counter or a deadline without a lock.
A panic inside a predicate, a handler or a value lookup is recovered where it is raised and read as a rejection, a stop or a missing row, since unwinding it through the library's frames would skip their unlocks.

## Late Interaction

`Maxsim` scores multi-row queries against multi-row entries: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `ValuesPerKey` above one exists for, and the wanted count is over documents rather than rows.
It runs on an inner-product index alone, since a sum of maxima needs logits rather than distances.

```go
options.Metric = usearch.DenseMetricInner
options.ValuesPerKey = 512

documents, err := index.On(executor).Maxsim(usearch.RowsOf(tokens, dimensions), tokensPerQuery, 10, nil)
```

## Sparse Products

`Multiply` reads an approximate sparse product off the graph: every pair the walk scored for each left row, in traversal order, rather than only the nearest.
It runs on an inner-product index alone, and its products land in an arena like `Get`'s rows.

```go
products, err := index.On(executor).Multiply(usearch.RowsOf(left, dimensions), &usearch.DenseMultiplyConfig{Wanted: 64})
defer products.Close()
for row := uint(0); row < products.Rows(); row++ {
	columns, scores := products.Row(row)
	accumulate(row, columns, scores)
}
```

## Attention

`Attend` is Flash Attention folded into the walk: every key it scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, and the outputs come back already divided by the partition.
The index stores no values, so a `ValueLookup` names, for every scored key, its row of a `Values` matrix as wide as the keys and stored in the index's own quantization.
A key without a row, or values of another width or scalar type, refuses the call with `StatusShapeMismatch`, since a missing row is a wrong softmax rather than a smaller one.
With `HeadCount: 8`, row `t * 8 + h` of the queries is head `h` of token `t`, and the heads of one token share its walk.

```go
lookup := usearch.ValueLookup{
	Values: usearch.RowsOf(valueCache, dimensions),
	Row:    func(key usearch.Key) (uint, bool) { row, ok := rowOf[key]; return row, ok },
}
carry := make([]usearch.DenseSoftmaxState, queriesCount)
err := index.On(executor).Attend(usearch.RowsOf(queries, dimensions), lookup, usearch.RowsOf(context, dimensions),
	&usearch.DenseAttendConfig{HeadCount: 8, Carry: carry})
```

A `Carry` of one `DenseSoftmaxState` per query row folds this branch into another over the same queries, such as a recent window kept out of the index.

## Refinement and Merging

`Refine` re-runs the neighbor selection over entries already in the graph, returning how many nodes had their neighbors changed.
A batch built at a low expansion factor can be brought up to the quality of a slower build afterwards, and the count reaching zero is what settled looks like:

```go
for {
	changed, err := index.On(executor).Refine(&usearch.DenseRefineConfig{ExpansionFactor: 128})
	if err != nil {
		log.Fatal(err)
	}
	if changed == 0 {
		break
	}
}
```

Bigger indexes come from merging, never mutating: `Merge` consumes every donor whatever the outcome, and every later verb on one reports `StatusUninitialized`.
No residency absorbs donors yet, so the engine answers `StatusUnsupported`.

```go
err := index.On(executor).Merge([]*usearch.DenseIndex{shard}, nil)
```

## Telemetry

Insertion, refinement and every walk report traversal telemetry when asked.
Each call overwrites the struct rather than accumulating, so a sweep sums them itself.

```go
var stats, total usearch.DenseTraversalStats
for _, batch := range batches {
	if err := index.On(executor).Insert(batch.keys, batch.vectors, &usearch.DenseInsertConfig{Stats: &stats}); err != nil {
		log.Fatal(err)
	}
	total.Accumulate(stats)
}
fmt.Println(total.VisitedMembers, total.ComputedDistances, total.Restarts)
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
		if _, err := index.On(executors[worker]).Search(queries, 10, nil); err != nil {
			log.Print(err)
		}
	}()
}
work.Wait()
```

Internal parallelism belongs to an `Executor` rather than to a residency or a handle: `ExecutorForRam(nil, n, 0)` fixes the worker count, and zero takes 7/8 of the logical cores, since full occupancy starves the ordered publication path.
Calls on one executor start in submission order, so goroutines that genuinely overlap take one apiece.
There is a constructor per C one, each taking every argument its own hardware has, and ending with the `deferredLimit` sizing its table of deferred calls:

```go
usearch.ExecutorForRam(memoryDomains, threads, deferredLimit)
usearch.ExecutorOnPool(memoryDomains, forkUnionPool, deferredLimit)  // borrows a fu_pool_t
usearch.ExecutorForVram(devices, streams, scratchLimit, deferredLimit) // one host thread, a stream per device
usearch.ExecutorForNvme(paths, threads, deferredLimit)                 // one submission ring per thread
```

Zero for any count derives it.
A placement names hardware and nothing else, so one pool serves several indexes and its residency must match theirs.
`Drain(mood)` waits for every submitted call and picks whether the workers park or spin once idle, and `Close` releases them.

Serialization is the one exception to the free overlap — saving demands a mutation-free window, though concurrent readers keep searching throughout.

## Design

The package is a thin layer over the C ABI in `include/usearch/`, and every rule below follows from that.

- __Names follow the header.__
  Every type is its C name without `usearch_` and `_t` - `DenseMetric`, `DenseTraversalStats`, `DenseIndexDescription` - every constructor is a C lifecycle verb such as `DenseInitRam` or `ExecutorForRam`, a verb's argument structure becomes its `Dense*Config`, and every wrapper's documentation names the C symbol it calls or says it is binding-only.
- __Blocking calls stay simple.__
  `index.Search(queries, 10, nil)` takes no executor or completion, and a nil config derives every knob; nothing about asynchrony reaches a caller who never asks for it.
- __Deferred calls are symmetric, through Go's own idioms.__
  The receiver's type alone chooses the completion, so `DenseIndexOn` and `DenseIndexDeferredOn` offer the same verbs with the same arguments, and a `Deferred` is waited on with `Wait`, or released with `Close`, from whichever goroutine wants to block.
- __Nothing costs anything until it is used.__
  A blocking call pins its inputs for the length of the crossing and allocates only the outputs it returns; a deferred one keeps the same pins in its `Deferred` until the call is final.
- __A `Deferred` has no life of its own.__
  `Status` is an atomic load of the outcome word, and `Wait` and `Close` block in `usearch_outcome_wait`; the binding starts no goroutine, counts no calls and queues nothing.
- __Statuses and modes are enums.__
  `Status`, `Mood`, `DenseMetric` and `Scalar` carry the C enumerators' values, and their `String` methods spell the names the C tables own, so nothing is compared as a string.
- __Arenas are reused, never shared.__
  A result owns a fresh `Arena` by default, passing one reuses it, and no index owns an arena or a lock.
