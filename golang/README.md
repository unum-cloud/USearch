# USearch for Go

The Go package binds the C ABI in `c/usearch/` through cgo, linking the pre-instantiated C library, so nothing but the vocabulary below crosses the boundary.
It needs Go 1.24 or newer with cgo enabled and a C toolchain, on Linux, macOS or Windows.
It exposes dense, sequence, shape and geospatial indexes, each with the same receivers, plans and reports.

## Installation

```sh
go get github.com/unum-cloud/USearch/golang
```

The package links `libusearch_static` and includes the `usearch/` headers, so both must sit on the compiler's search paths, such as `/usr/local/lib` and `/usr/local/include`, or be named through `CGO_LDFLAGS` and `CGO_CFLAGS`.
To build them from a checkout of the main repository and point cgo at them for the current shell:

```sh
cmake --preset library -B build_release && cmake --build build_release
export CGO_CFLAGS="-I$PWD/c" CGO_LDFLAGS="-L$PWD/build_release"
```

Every release also holds prebuilt libraries and headers, where `<release_tag>`, `<arch>` and `<usearch_version>` name the release and the machine:

- __Linux__: `dpkg -i usearch_linux_<arch>_<usearch_version>.deb` installs into `/usr/local`.
- __macOS__: unpack `usearch_macos_<arch>_<usearch_version>.zip`, then move the library into `/usr/local/lib` and the `usearch` directory into `/usr/local/include`.
- __Windows__: `.\usearch\winlibinstaller.bat`, run in the folder where you will `go run`, downloads the latest release's library and headers into that folder.

Each archive downloads from `https://github.com/unum-cloud/USearch/releases/download/<release_tag>/`.

## Quickstart

Declare a module in `go.mod`:

```go.mod
module usearch_example

go 1.24
```

Write `example.go`:

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
	// Quantization also accepts ScalarF16, ScalarE5M2, ScalarE4M3, ScalarE3M2,
	// ScalarE2M3, ScalarI8, ScalarU8 and ScalarU1x8.
	options.Quantization = usearch.ScalarBF16

	// A nil list of memory domains places a shard on every NUMA node.
	index, err := usearch.DenseInitRam(options, nil)
	if err != nil {
		log.Fatal(err)
}
	defer index.Close()
	if err := index.Reserve(entries); err != nil {
		log.Fatal(err)
}

	keys := make([]usearch.Key, entries)
	vectors := make([]float32, entries*dimensions)
	for entry := range entries {
		keys[entry] = usearch.KeyFromUint64(uint64(entry))
		for dimension := range dimensions {
			vectors[entry*dimensions+dimension] = float32(entry + dimension)
}
}
	if err := index.Insert(keys, usearch.RowsOf(vectors, dimensions), nil); err != nil {
		log.Fatal(err)
}

	arena, err := usearch.NewArena(nil)
	if err != nil {
		log.Fatal(err)
}
defer arena.Close()
	query := []float32{0.0, 1.0, 2.0}
	matches, err := index.Search(usearch.RowsOf(query, dimensions), 3, arena, nil)
	if err != nil {
		log.Fatal(err)
}
	found, distances := matches.Query(0)
	for rank := range found {
		fmt.Println(found[rank], distances[rank])
}
}
```

Fetch the package and run it:

```sh
go get github.com/unum-cloud/USearch/golang
go run example.go
```

Always call `Reserve(capacity)` before the first write.

## Receivers & Deferred Calls

The receiver chooses where a verb runs and how it completes, and the verb stays the same.
The index runs it on the calling goroutine, `index.On(executor)` blocks on an executor, and `index.Deferred(executor)` submits it there and answers a `*usearch.Deferred`:

```go
matches, err := index.Search(queries, 10, arena, nil)              // the calling goroutine
matches, err = index.On(executor).Search(queries, 10, arena, nil) // blocking, holding the executor until it returns
call := index.Deferred(executor).Search(queries, 10, arena, nil)   // *usearch.Deferred[usearch.MatchesView]
```

`DenseIndexOn` and `DenseIndexDeferredOn` offer the same verbs with the same arguments; only the blocking ones add `Contains`, `Cursor` and `SearchStream`, whose every pull blocks.

A `Deferred` holds what the library reads or writes until its call is final: the outcome and the pinned inputs.

```go
call := index.Deferred(executor).Search(queries, 10, arena, nil)
if call.Status() == usearch.StatusDeferred {
	// other work
}
matches, err := call.Wait() // a view into arena, as a blocking call's is
```

Go needs nothing more: a goroutine that wants to overlap the wait simply calls `Wait` there, and a `select` waits on a channel that goroutine closes.
`Status` reads the outcome without blocking, and `Wait` blocks until the call is final and hands back what it produced, as many times as it is asked.
`Close` waits the same way and drops the result, so a call no one reads is closed rather than waited on; it is idempotent and safe after `Wait`.
A failure this binding catches before the call crosses arrives through the same `Status` and `Wait`, as the engine's own refusals do.
A `Deferred` dropped without `Wait` or `Close` is released by the collector once its call is final, which checks it again at every later collection rather than blocking on it.

One call holds an executor at a time, so a second call finding it held finishes at once with `StatusContended`, and nothing queues.
A deferred call on a caller-exclusive pool runs on the pool's thread 0, so polling `Status` alone finishes it, and its progress handler and predicate run on the pool's threads; a single thread or a VRAM executor runs it at submission.
Calls depending on each other are ordered by waiting on the first before submitting the second.
The slices a deferred verb is handed must not change until it is final, since the library reads them until then.

## Executors

An index owns entries; an executor owns the threads or device streams a verb runs on, so one executor serves several indexes and the cost of spawning shows at the line that spawns it.
There is a constructor per C one, each taking every argument its own hardware has:

```go
usearch.ExecutorForRam(memoryDomains, cpuClassCeiling, threads)
usearch.ExecutorForVram(devices, gpuClassCeiling, streams, scratchLimit) // one host thread, a stream per device
usearch.ExecutorForNvme(paths, cpuClassCeiling, threads)                 // one submission ring per thread
```

Zero for any count derives it, and zero threads in `ExecutorForRam` takes 7/8 of the logical cores, since full occupancy starves the ordered publication path.
A placement names hardware and nothing else, and an executor's residency must match that of the indexes it drives, or the engine reports `StatusDevice`.
A nil `*Executor` owns nothing, and every call handed it runs on the calling goroutine.
One call holds an executor at a time, so goroutines that genuinely overlap take one apiece.
`Drain(mood)` waits until no call holds the executor and picks whether the workers park or spin once idle, and `Close` releases them.

Constructors address no index yet, so no receiver offers them; the ones the C interface lets fan out take an `Executor` in their config, and run on the calling goroutine without one.

### Concurrency

A `DenseIndex` may be shared across goroutines with no lock of your own.
Searches read only entries published behind the visibility watermark and never write to the graph, while insertions synchronize through per-node locks and publish in order, so the two overlap freely.

```go
var work sync.WaitGroup
for worker := range workers {
	work.Add(1)
	go func() {
		defer work.Done()
		if _, err := index.On(executors[worker]).Search(queries, 10, arenas[worker], nil); err != nil {
			log.Print(err)
}
	}()
}
work.Wait()
```

Serialization is the one exception to the free overlap: saving demands a mutation-free window, though concurrent readers keep searching throughout.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in an `Arena` you pass, and the verb answers a view over it without copying.
`NewArena(executor)` places it where that executor computes, decided once, and `NewArena(nil)` makes host pages for the calling goroutine.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
A view and every slice its accessors returned end at the arena's next call, or its `Close`, and nothing checks them after that.
An arena from a VRAM executor holds device memory, which `Describe` reports and no Go code may read.
It grows and frees on that executor's stream, so it serves only calls on that executor, any other finishing `StatusDevice`, and is closed before it; host pages may outlive theirs.

```go
arena, err := usearch.NewArena(executor) // NewArena(nil) makes host pages for the calling goroutine
	if err != nil {
		log.Fatal(err)
}
defer arena.Close()
for _, batch := range batches {
	matches, err := index.On(executor).Search(batch, 10, arena, nil) // rewinds: the last batch's matches are gone
	if err != nil {
		log.Fatal(err)
}
	keys, _ := matches.Query(0)
	kept = append(kept, keys...) // a slice into the arena goes stale unchecked, so copy what outlives the loop
}

found, _ := usearch.NewArena(nil) // two live results, two arenas
defer found.Close()
fetched, _ := usearch.NewArena(nil)
defer fetched.Close()
matches, _ := index.Search(query, 10, found, nil)
neighbors, _ := matches.Query(0)
for _, key := range neighbors {
	vectors, _ := index.Get(key, usearch.ScalarF32, fetched) // rewinds fetched, never found
	rerank(key, vectors)
}

calls := make([]*usearch.Deferred[usearch.MatchesView], len(shards))
for position, shard := range shards {
	calls[position] = shard.Deferred(executors[position]).Search(queries, 10, arenas[position], nil) // one executor and one arena per submitted call
}
for position, call := range calls {
	if results[position], err = call.Wait(); err != nil {
		log.Fatal(err)
}
}
```

A call that finds its arena serving another call finishes with `StatusContended` and rewinds nothing, so the holder's views stay readable.
`Close` waits for every deferred call naming the arena and ends its views as a rewind does, so a dropped `Deferred` never leaves it unsafe.

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

Every failure holds a status shared with the C interface, so a fault this binding catches before the boundary reports the identity the engine would have reported for the same fault.
A `Status` is itself an `error`, so `errors.Is` matches any failure holding it, and an `*usearch.Error` adds its message:

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

`Status`, like `Mood`, `Residency`, `Traversal`, `DenseMetric` and `Scalar`, holds the C enumerators' values, and its `String` spells the name the C tables own.
`StatusUnsupported` marks a verb declared for every residency that this one does not implement yet, and `StatusContended` a call refused rather than waited for.

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each takes only the placement its own hardware has, one shard per entry, and an empty list derives it.

```go
inRam, err := usearch.DenseInitRam(options, []int32{0, 1})        // one graph per socket
inVram, err := usearch.DenseInitVram(options, []int32{0, 1, 2, 3}) // sharded, never replicated
onGpu, err := usearch.DenseLoadVram("index.safegraphs", &usearch.DenseLoadVramConfig{Devices: []int32{0}})
```

An NVMe index is only ever viewed: external-memory graphs are never modified in place.

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

monolith, err := index.SaveBuffer(arena, nil) // sized by the call itself, see Arenas
copied, err := usearch.DenseLoadRamBuffer(monolith.Bytes(), nil)
```

Loading creates the index, freezing its geometry from the checkpoint's own headers.
`DenseMetadataRead` reads that geometry without opening a shard:

```go
metadata, err := usearch.DenseMetadataRead("index.safegraphs", nil) // a nil config reads on the calling goroutine
fmt.Println(metadata.Dimensions, metadata.Metric, metadata.Quantization)
```

The result holds only the four fields a checkpoint froze, so `ValuesPerKey` and the placement stay yours to set in the load config.
Serving a checkpoint in place, rather than copying it, yields a read-only handle whose mutating verbs fail with `StatusReadOnly`, through a memory mapping with `DenseViewRam`, or from drives with `DenseViewNvme`:

```go
mapped, err := usearch.DenseViewRam("index.safegraphs")
served, err := usearch.DenseViewNvme("index.safegraphs", &usearch.DenseViewNvmeConfig{
	Executor:        drives,
	Paths:           []string{"/dev/nvme0n1"},
	KeyTableDomains: []int32{0}, // a key table on NUMA node 0, for Count, Get and Sample
})
```

Without `KeyTableDomains`, an NVMe view holds nothing per entry, so `Count`, `Get`, `Sample`, MaxSim and hinted searches report `StatusUnsupported`.
`DenseLoadNvme` takes the same config and opens a manifest writable, so `Refine` rewrites its parts one at a time, while a lone monolith reports `StatusUnsupported`.

## Dense Indexes

### Rows & Scalars

Every verb is a batch over strided matrices, so a memory-mapped dataset reaches the index with no reshape and no copy, and a single vector is a one-row matrix.
A `RowsView` views a caller-owned slab as rows of equal width, and it is the shape every batch argument takes.
`RowsOf` derives the row count and the byte pitch from a packed slice; `StridedRowsOf` takes a pitch wider than the row, which is how a memory-mapped dataset with padded rows arrives.

```go
vectors := usearch.RowsOf(embeddings, 768)                    // packed
padded := usearch.StridedRowsOf(mapped, rows, 768, 1024*4)    // pitch of 1024 floats
```

The element type of the slice names the scalar kind, and it is independent of the index's own quantization, since the library converts on ingest and on export.
A `[]float32` query searches a `ScalarI8` index without a conversion pass of your own.

Sub-byte kinds count logical dimensions rather than bytes, so 128 dimensions of `U1x8Scalar` occupy sixteen bytes per row:

```go
packed := make([]usearch.U1x8Scalar, rows*128/8)
vectors := usearch.RowsOf(packed, 128) // 128 dimensions, 16 bytes per row
```

### Search

Every verb with optional knobs ends with a pointer to a config named after its C argument structure, such as `DenseInsertConfig`, `DenseSearchConfig` or `DenseMultiplyConfig`, where `nil` and every zero field derive what the C interface derives.
So `nil` makes the bare call, and naming a field reaches a knob:

```go
var stats usearch.DenseTraversalStats
matches, err := index.On(executor).Search(queries, 10, arena, &usearch.DenseSearchConfig{
	ExpansionFactor: 128,
	Stats:           &stats,
})
fmt.Println(stats.ComputedDistances, "distances computed")
```

A search returns a `MatchesView` over its arena, holding up to `wanted` keys per query and the distances paired with them.
`Query` views one query's results, trimmed to the count that query reported, and copies nothing:

```go
for query := range matches.Queries() {
	found, distances := matches.Query(query)
	for rank := range found {
		fmt.Println(query, found[rank], distances[rank])
}
}
```

The breadth of the candidates rides each call as `ExpansionFactor` rather than the handle, so a recall sweep runs against one immutable index with no mutation between points.
`Traversal: usearch.TraversalScan` skips the walk and scores every entry exactly, on RAM and NVMe handles, which gives such a sweep its ground truth.

Insertion, refinement and every walk report traversal telemetry when asked.
Each call overwrites the struct rather than accumulating, so a sweep sums them itself:

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

### Filtered Search

A `Predicate` in the `DenseSearchConfig` admits only the keys it approves.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.

```go
admits := func(key usearch.Key) bool {
	value, integral := key.Uint64()
	return integral && value%2 == 0
}
matches, err := index.On(executor).Search(queries, 10, arena, &usearch.DenseSearchConfig{Predicate: admits})
```

The predicate runs on the library's worker threads, so it must be safe for concurrent use.

Every long verb's config takes a `Progress` handler, called with the units finished so far and the units the call finishes if it runs to the end; answering `false` stops it wherever the verb can stop, which an `Insert` cannot.

```go
deadline := time.Now().Add(30 * time.Second)
report, err := index.On(executor).Refine(&usearch.DenseRefineConfig{
	ExpansionFactor: 128,
	Progress:        func(completed, total uint) bool { return time.Now().Before(deadline) },
})
```

A handler never runs on two threads at once, so it may hold a counter or a deadline without a lock.
A predicate, a handler or a value lookup must not panic: a panic unwinding through the library's frames is a bug, and nothing here recovers it.

### Retrieval and Streaming

`Get` exports every entry stored under one key, converted to the scalar kind you name, in one call however many there turn out to be:

```go
vectors, _ := index.Get(usearch.KeyFromUint64(42), usearch.ScalarF32, arena)
// vectors.Rows() rows of vectors.Columns(), vectors.Stride() bytes apart
values, _ := usearch.DenseValues[float32](vectors)
```

A cursor walks the contents, each pull writing up to `wanted` keys, and the keys each links to for `NextLinks`, into an arena; the vectors themselves come from `Get`.
`NextKeys` pulls the keys alone, and a batch of none means the walk is exhausted.
`Cursor()` yields every entry as stored, and `Cursor(starts...)` the starts' own entries first, then the entries those link to breadth-first, each once.
Cursors answer on RAM and NVMe handles, while VRAM entries are read back by key or by search.

```go
cursor, _ := index.On(executor).Cursor()
defer cursor.Close()
for {
	batch, err := cursor.NextLinks(1024, arena)
	if err != nil || batch.Written() == 0 {
		break
	}
	consume(batch.Keys(), batch.Links(0))
}
```

A search stream resumes one query's walk, so a caller that post-filters aggressively pulls until satisfied instead of guessing the wanted count upfront.
Each `Next` writes up to `wanted` next-nearest matches into an arena as the one query of a search, and a query with none means the walk is drained.
Ordering beyond the expansion horizon is best-effort nearest rather than globally exact.
No residency streams matches yet, so `SearchStream` answers `StatusUnsupported` today.

```go
stream, err := index.On(executor).SearchStream(usearch.RowsOf(query, dimensions), nil)
	if err != nil {
		log.Fatal(err)
}
defer stream.Close()
for accepted := 0; accepted < 10; {
	matches, err := stream.Next(64, arena)
	if err != nil {
		log.Fatal(err)
}
	found, _ := matches.Query(0)
	if len(found) == 0 {
		break // the walk is drained
}
	for _, key := range found {
		if admits(key) {
			accepted++
}
}
}
```

### Refinement

`Refine` re-runs the neighbor selection over entries already in the graph, returning a `DenseRefineReport`: the entries re-selected and changed, the islands found and reconnected, the neighborhoods re-stitched with the links added and dropped, the `Verdict`, and the health checks opening the round and closing the heal.
A batch built at a low expansion factor can be brought up to the quality of a slower build afterwards, and a `Verdict` other than `DenseRefineImproved` is what settled looks like:

```go
for {
	report, err := index.On(executor).Refine(&usearch.DenseRefineConfig{ExpansionFactor: 128})
	if err != nil {
		log.Fatal(err)
	}
	if report.Verdict != usearch.DenseRefineImproved {
		break
	}
}
```

On a loaded NVMe handle `MemoryLimit` bounds the bytes a round holds for the one part it rewrites: zero rounds the part in RAM, and a smaller budget streams its rounds from the drives.

### Resharding

`Reshard` moves the entries of its sources into its targets, keeping their links rather than re-inserting their vectors, then runs a round on every target that took entries; the receiver passes only its executor, so every handle is named.
Several sources gather into one target, itself among them, and each other source ends empty and live, still yours to close.
Every handle must share the first source's residency, scalar, class, metric and geometry, or the call answers `StatusConfigMismatch`, and a refusal moves nothing.

```go
report, err := index.On(executor).Reshard([]*usearch.DenseIndex{index, first, second}, []*usearch.DenseIndex{index}, nil)
```

One source cut into several empty targets hands each a graph-local part, keeping their edges and leaving the source empty but live.

```go
report, err := index.On(executor).Reshard([]*usearch.DenseIndex{index}, []*usearch.DenseIndex{left, right}, &usearch.DenseReshardConfig{Seed: 42})
```

The index as its only source and target recuts its own shards in place into as few as fit `ShardCapacity`, keeping every entry.

```go
self := []*usearch.DenseIndex{index}
report, err := index.On(executor).Reshard(self, self, &usearch.DenseReshardConfig{MemoryDomains: []int32{}, ShardCapacity: 10_000_000})
```

A `Plan` in a `DenseRefineConfig` or `DenseReshardConfig` keeps a stopped call live: the next call naming it continues where the stop came, its `Phase` says where each call stopped, and `RefinePlanFinish` or `ReshardPlanFinish` closes it and lands what the whole plan did.

```go
var plan usearch.DenseRefinePlan
for plan.Phase.Phase != usearch.PhaseDone {
	_, err := index.On(executor).Refine(&usearch.DenseRefineConfig{Plan: &plan, Progress: withinBudget})
	if err != nil && !errors.Is(err, usearch.StatusCancelled) {
		log.Fatal(err)
	}
}
report, err := index.On(executor).RefinePlanFinish(&plan)
```

### Sampling

`Sample` draws keys around `Focus`, or across the whole index, each hanging from a shown or earlier draw and placed in three dimensions beside the `Shown` keys already on screen, so a view grows one connected sample as it zooms.
Equal seeds, shown keys and focus draw equally, and a larger `wanted` draws a superset.

```go
draws, err := index.On(executor).Sample(256, arena, &usearch.DenseSampleConfig{
	Shown: shown, ShownCoordinates: places, Focus: near, Seed: 42,
})
keys := draws.Keys()
entryOrdinals := draws.EntryOrdinals()
parent, distance := draws.Parents().Query(0) // empty for a root
points, _ := usearch.DenseValues[float32](draws.Coordinates())
```

### Sparse Multiplication

`Multiply` reads an approximate sparse product off the graph: every pair the walk scored for each left row, in traversal order, rather than only the nearest.
It runs on an inner-product index alone, and its keys, products and row offsets land in an arena like `Get`'s rows.

```go
products, err := index.On(executor).Multiply(usearch.RowsOf(left, dimensions), arena, &usearch.DenseMultiplyConfig{Wanted: 64})
for row := range products.Queries() {
	columns, scores := products.Query(row)
	accumulate(row, columns, scores)
}
```

### Late Interaction

`Maxsim` scores multi-row queries against multi-vector documents: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `ValuesPerKey` above one exists for, and the wanted count is over documents rather than rows.
It runs on an inner-product index alone, since a sum of maxima needs logits rather than distances.
The keys land as `Multiply`'s pairs do, one `DenseProducts` run per query, largest score first, and a zero wanted count keeps every key a query scored.

```go
options.Metric = usearch.DenseMetricInner
options.ValuesPerKey = 512

documents, err := index.On(executor).Maxsim(usearch.RowsOf(tokens, dimensions), tokensPerQuery, 10, arena, nil)
for query := range documents.Queries() {
	keys, scores := documents.Query(query)
	rerank(query, keys, scores)
}
```

### Attention

`Attend` is Flash Attention folded into the walk: every key it scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, and the outputs land in an arena already divided by the partition.
The index stores no values, so a `ValueLookup` names, for every scored key, its row of a `Values` matrix as wide as the keys and stored in the index's own quantization.
A key without a row, or values of another width or scalar type, refuses the call with `StatusShapeMismatch`, since a missing row is a wrong softmax rather than a smaller one.
A VRAM walk resolves its rows through a device function, which no Go function can be, so `Attend` serves RAM and NVMe indexes alone.
With `HeadCount: 8`, row `t * 8 + h` of the queries is head `h` of token `t`, and the heads of one token share its walk.

```go
lookup := usearch.ValueLookup{
	Values: usearch.RowsOf(valueCache, dimensions),
	Row:    func(key usearch.Key) (uint, bool) { row, ok := rowOf[key]; return row, ok },
}
carry := make([]usearch.DenseSoftmaxState, queriesCount)
attended, err := index.On(executor).Attend(usearch.RowsOf(queries, dimensions), lookup, arena,
	&usearch.DenseAttendConfig{HeadCount: 8, Carry: carry})
context, _ := usearch.DenseValues[float32](attended)
```

The attended rows take the queries' scalar type unless the config's `OutputsScalar` names another.
A `Carry` of one `DenseSoftmaxState` per query row, in the caller's memory rather than in the arena, folds this branch into another over the same queries, such as a recent window kept out of the index.

### Introspection

```go
description, _ := index.Describe()
fmt.Println(description.Metadata.Dimensions, description.Size, description.Capacity,
	description.MemoryUsage, description.CPUClass) // the class dispatch picked on this machine

found, _ := index.Contains(usearch.KeyFromUint64(42))
count, _ := index.Count(usearch.KeyFromUint64(42)) // entries sharing one key

reachability, _ := index.Health()
fmt.Println(reachability.Reachable, "of", reachability.Audited, "entries reachable")

fmt.Println(usearch.Version(), usearch.CapabilitiesCompiled(), usearch.CapabilitiesAvailable())
```

`Health` measures how much of the graph a search reaches from the root; entries it misses stay unfound until a `Refine` re-links them.

## Sequence Indexes

A sequence index searches byte strings, UTF-8 text, DNA and proteins under edit distances and alignments.
Variable-length entries cross as a `Tape`, the Apache Arrow layout: elements end to end and one more offset than entries, which `SequencesOf` packs and `SequencesTape` wraps an Arrow column as it is:

```go
index, _ := usearch.SequenceInitRam(usearch.SequenceOptions{Metric: usearch.SequenceMetricLevenshteinUTF8}, nil)
defer index.Close()

words := usearch.SequencesOf([]byte("kitten"), []byte("sitting"), []byte("mitten"))
keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2), usearch.KeyFromUint64(3)}
_ = index.Insert(keys, words, nil)

matches, _ := index.Search(usearch.SequencesOf([]byte("sittin")), 1, arena, nil)
found, distances := matches.Query(0) // key 2, one edit away
```

`SequenceMetricNeedlemanWunsch` takes `SequenceWeights`, a 256 × 256 substitution matrix with a gap cost, and `SequenceMetricOverlap` takes `WindowWidths`; every load names the weights again, since no checkpoint records them.
`Get` writes a key's sequences into an arena as a `TapeView`, read with `TapeEntry[byte]`, and `Count` reports a key's total and largest byte lengths.
The verbs, receivers, plans and reports are the dense ones: `Refine`, `Reshard` and their `SequenceRefinePlan` and `SequenceReshardPlan`, cursors and `Sample`, all reporting through `DenseRefineReport` and `DenseTraversalStats`.

## Shape Indexes

A shape index searches 3D point clouds of any size, like molecules and conformers, invariant to rotation, reflection, translation and point order.
Its tape holds points end to end, `x`, `y` and `z` interleaved, with offsets that count points:

```go
index, _ := usearch.ShapesInitRam(usearch.DefaultShapesOptions(), nil) // Chamfer over f32 points
defer index.Close()

// A 2-atom and a 3-atom molecule.
molecules := usearch.ShapesOf([]float32{0, 0, 0, 1.1, 0, 0}, []float32{0, 0, 0, 1.5, 0, 0, 0, 1.5, 0})
keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2)}
_ = index.Insert(keys, molecules, nil)

matches, _ := index.Search(usearch.ShapesOf([]float32{0, 0, 0, 1.0, 0, 0}), 1, arena, nil)
found, _ := matches.Query(0) // key 1
```

The metrics are Chamfer, Hausdorff, Gromov-Wasserstein and Wasserstein-Procrustes, over `f16` or `f32` storage, whatever scalar the points arrive in.
A non-zero `Points` freezes every shape's point count.
`Get` converts each shape to the scalar it is asked for and returns it in its inserted pose, and `Count` reports a key's total and largest point counts.

## Geospatial Indexes

A geospatial index searches observations on and above the Earth, like photos, video frames and sensor fixes.
`GeospatialObservation` is the library's own eleven doubles, so a slice of them crosses with no copy; a field no device recorded stays NaN, which only widens the tests that read it, and `GeospatialAt` fills every field but the position that way:

```go
index, _ := usearch.GeospatialInitRam(usearch.GeospatialOptions{}, nil)
defer index.Close()

stops := []usearch.GeospatialObservation{
	usearch.GeospatialAt(51.5308, -0.1238),
	usearch.GeospatialAt(51.5320, -0.1233),
	usearch.GeospatialAt(51.5033, -0.1196),
}
keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2), usearch.KeyFromUint64(3)}
_ = index.Insert(keys, stops, nil)

nearby, _ := index.Within(stops[:1], usearch.GeospatialRelationNear, 500, arena, nil)
found, chords := nearby.Query(0) // both stops within 500 metres, in chord metres
```

`Within` reports every entry the relation admits inside the radius, however many, while `Search` ranks the `wanted` nearest, a zero `Metres` ranking by distance alone.
The relation is an argument of each call: `Near` widens the radius by both positions' accuracies, `Sees` and `Seen` test the view cones, and `Visible` needs a terrain, which this binding does not bind.
`Get` returns a key's observations within the stored rounding.

## What This Binding Does Not Expose

- __Deferred cursor and stream pulls.__ The C interface takes a deferred completion on `usearch_dense_cursor_next` and `usearch_dense_search_stream_next`, while every pull here blocks on the receiver's executor.
- __Name parsers.__ The `usearch_*_from_name` functions are left out, since enums are never read from strings.
- __Terrain.__ `usearch_geospatial_terrain_t` stays unset, so every geospatial handle answers the visible relation with `StatusUnsupported`; its heights would have to outlive the handle, which no call can pin.
- __Class names and the numeric version.__ `usearch_cpu_class_name`, `usearch_gpu_class_name` and `usearch_version` are unbound; `Version` answers the version's name.

## Design

- __A thin layer over the C ABI.__ Every name follows `c/usearch/`: a C verb `usearch_dense_<verb>` is a `<Verb>` method here, and enums mirror C enums, never strings.
- __The receiver chooses, the verb stays.__ The index runs a verb on the calling goroutine, `On(executor)` blocks on an executor, and `Deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__ `Status` polls through `usearch_outcome_status` and `Wait` blocks through `usearch_outcome_wait`; a goroutine calling `Wait` overlaps the wait, and a `select` reaches it through a channel that goroutine closes.
- __No life of its own.__ The binding starts no goroutines and keeps no queues; releasing a `Deferred` rides `Close`, which waits for the call first, or the collector's cleanup, which frees it only once final.
- __Arenas belong to the caller and are reused, never shared.__ Every output whose length scales with the call lands in an arena the caller passes, and its views end at the arena's next call.
- __Nothing costs anything until it is used.__ Submitting a call allocates nothing, and an arena allocates only when a call writes into it.
- __Types and configs are the header's.__ Every type is its C name without `usearch_` and `_t`, a verb's argument structure becomes its `Dense*Config`, which the other modalities share wherever their fields agree, and every wrapper's documentation names the C symbol it calls or says it is binding-only.
