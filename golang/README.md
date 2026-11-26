# USearch for Go

The Go package binds the C ABI in `c/usearch/` through cgo, linking the pre-instantiated C library, so nothing but the vocabulary below crosses the boundary.
It needs Go 1.24 or newer with cgo enabled and a C toolchain, on Linux, macOS or Windows.
It exposes dense, sequence, shape and geospatial indexes, each with the same receivers, plans and reports.

## Installation

The package takes its headers from the module itself and links `libusearch_static`, which you build once from a checkout of this repository and point cgo at, then fetch the module by branch or by commit:

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release
cmake --build build_release --target usearch_static
export CGO_LDFLAGS="-L$PWD/build_release"         # or install the archive where the linker looks
go get github.com/unum-cloud/usearch/golang@main  # or @<commit>
```

```go
import usearch "github.com/unum-cloud/usearch/golang"
```

The module path carries no major version, so `go.sum` locks the commit you fetched as a pseudo-version, and tags play no part.

## Quickstart

Every batch input is a `RowsView` of a base pointer, a row count and a byte stride, so a memory-mapped dataset feeds the index with no reshape and no copy.
An index owns entries, an `Executor` owns the threads or streams a call runs on, and an `Arena` holds what a call writes back.

```go
package main

import (
	"fmt"
	"log"

	usearch "github.com/unum-cloud/usearch/golang"
)

func main() {
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

	executor, err := usearch.ExecutorForRam(nil, usearch.CPUClassUnknown, 0, 0)
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

	found, _, err := matches.Query(0)
	if err != nil {
		log.Fatal(err)
	}
	foundLocally, _, err := local.Query(0)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(found[0], foundLocally[0])
}
```

`go test` runs these same lines as the package's `Example`.
Every verb takes a pointer to its optional config last, and `nil` derives every setting.

## Receivers & Deferred Calls

The receiver decides where a verb runs and how it completes, while the verb and its arguments stay the same:

- `index.Search(..)` runs on the calling goroutine.
- `index.On(executor).Search(..)` blocks on the executor.
- `index.Deferred(executor).Search(..)` submits the call and answers a `*usearch.Deferred`.

`DenseIndexOn` and `DenseIndexDeferredOn` offer the same verbs with the same arguments; only the blocking one adds `Contains`.
A blocking verb returns its result and an `error` directly, so nothing about asynchrony reaches a caller who never asks for it.

```go
call, err := index.Deferred(executor).Search(queries, 10, arena, nil) // *usearch.Deferred[usearch.MatchesView]
if err != nil {
	return err // refused before anything was submitted
}
if call.Status() == usearch.StatusDeferred {
	// other work
}
matches, err := call.Wait() // a view into arena, as a blocking call's is
```

A deferred verb answers `(*Deferred[T], error)`: a fault this binding catches, such as keys and rows of different counts or a closed handle, comes back at once with no `Deferred`, while the engine's own failures arrive through `Status` and `Wait`.
A `Deferred` holds what the library reads or writes until its call is final: the outcome, the result slot and the pinned inputs.
`Status` reads the outcome without blocking, and `Wait` blocks until the call is final and hands back what it produced, as many times as it is asked.
`Close` waits the same way and drops the result, so a call no one reads is closed rather than waited on; it is idempotent and safe after `Wait`.
A goroutine that wants to overlap the wait simply calls `Wait` there, and a `select` waits on a channel that goroutine closes.
A `Deferred` dropped without `Wait` or `Close` is released by the collector once its call is final, which checks it again at every later collection rather than blocking on it.

The library reads a deferred call's inputs after the verb returns, so the slices a deferred verb is handed must not change until it is final.
One call holds an executor at a time, so a second call finding it held finishes at once with `StatusContended`, and nothing queues.
A deferred call on a caller-exclusive pool runs on the pool's thread 0, so polling `Status` alone finishes it, and its progress handler and predicate run on the pool's threads; a single thread or a VRAM executor runs it at submission.
Calls depending on each other are ordered by waiting on the first before submitting the second, since no call carries an ordering field.

## Executors

An executor owns host threads or device streams, and the traversal scratch each of them walks with, grown to a high-water mark.
One executor serves any number of indexes, and its residency must match the one their handles were built for, or the engine reports `StatusDevice`.
There is a constructor per residency, each taking only what its own hardware has, and the placement lists are the ones the indexes were built with, which keeps compute and memory domains in agreement.

```go
threads, err := usearch.ExecutorForRam(nil, usearch.CPUClassUnknown, 8, 0)           // eight workers
domains, err := usearch.ExecutorForRam([]int32{0, 1}, usearch.CPUClassUnknown, 0, 0) // threads across two NUMA nodes
capped, err := usearch.ExecutorForRam(nil, usearch.CPUClassIntelHaswell, 0, 1<<28)   // nothing newer than Haswell, 256 MiB of scratch each
rings, err := usearch.ExecutorForNvme([]string{"/mnt/nvme0"}, nil, usearch.CPUClassUnknown, 4, usearch.NvmeClassUnknown, 0) // one submission ring apiece
streams, err := usearch.ExecutorForVram([]int32{0, 1}, usearch.GPUClassUnknown, nil, 0) // one stream it opens per device

err = threads.Drain(usearch.MoodSpinning) // waits out the holding call, then keeps spinning
```

`ExecutorForRam` and `ExecutorForNvme` take a thread count, where zero derives one: seven eighths of the cores for RAM, since full occupancy starves the ordered publication path, and the drives' queue capabilities for NVMe.
The last argument caps each thread's scratch, past which a call fails with `StatusOutOfMemory`, and zero takes 1 GiB; an NVMe executor's memory domains place its threads over the drives' own domain.
`ExecutorForVram` takes none, since one host thread drives one stream per device, which the executor opens with itself and destroys when it closes; your own CUDA streams in `streams` answer `StatusUnsupported`, since the executor drives only its own.
A nil `*Executor` owns nothing, and every call handed it runs on the calling goroutine.
`Drain(mood)` waits until no call holds the executor and sets the `Mood` idle workers adopt, and `Close` releases them.
Constructors address no index yet, so no receiver offers them; the ones the C interface lets fan out take an `Executor` in their config, and run on the calling goroutine without one.

One call holds an executor at a time, so goroutines that genuinely overlap take one apiece, while index handles stay freely shared with no lock of your own.
Searches read only entries published behind the visibility watermark and never write to the graph, while insertions synchronize through per-node locks and publish in order, so the two overlap freely:

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
A view read after a later call took its arena fails as `Contended`.
A slice or segment an accessor returned is valid only until the arena's next call.
Closing the arena ends every view over it.
An arena from a VRAM executor may hold device memory: `Describe` reports its `HostAccess`, and every view over memory the host may not read refuses with `StatusDevice`, while its Arrow exports stay open.
It grows and frees on that executor's stream, so it serves only calls on that executor, any other finishing `StatusDevice`.
An arena never outlives the executor that made it, so `defer executor.Close()` comes before `defer arena.Close()`, and an arena from `NewArena(nil)` lives until it is closed.

```go
arena, err := usearch.NewArena(executor) // NewArena(nil) makes host pages for the calling goroutine
if err != nil {
	return err
}
defer arena.Close()
for _, batch := range batches {
	matches, err := index.On(executor).Search(batch, 10, arena, nil) // rewinds: the last batch's matches are gone
	if err != nil {
		return err
	}
	keys, _, err := matches.Query(0)
	if err != nil {
		return err
	}
	kept = append(kept, keys...) // a slice into the arena goes stale unchecked, so copy what outlives the loop
}

found, _ := usearch.NewArena(nil) // two live results, two arenas
defer found.Close()
fetched, _ := usearch.NewArena(nil)
defer fetched.Close()
matches, _ := index.Search(query, 10, found, nil)
neighbors, _, _ := matches.Query(0)
for _, key := range neighbors {
	vectors, _ := index.Get(key, usearch.ScalarF32, fetched) // rewinds fetched, never found
	rerank(key, vectors)
}

calls := make([]*usearch.Deferred[usearch.MatchesView], len(shards))
for position, shard := range shards { // one executor and one arena per submitted call
	if calls[position], err = shard.Deferred(executors[position]).Search(queries, 10, arenas[position], nil); err != nil {
		return err
	}
}
for position, call := range calls {
	if results[position], err = call.Wait(); err != nil {
		return err
	}
}
```

A call that finds its arena serving another call finishes with `StatusContended` and rewinds nothing, so the holder's views stay readable.
`Close` waits for every deferred call naming the arena and ends its views as a rewind does, so a dropped `Deferred` never leaves it unsafe.
Fixed-size reports, such as stats, descriptions and `Required`, stay plain values, and so does state you carry between calls, like the attention `Carry`.

## Keys

A `Key` is a 128-bit identity, rendered as a lowercase dashed UUID.
Integer identities park in the low eight bytes, so `KeyFromUint64(42)` prints as `00000000-0000-0000-0000-00000000002a`.

```go
key := usearch.KeyFromUint64(42)
parsed, err := usearch.ParseKey("00000000-0000-0000-0000-00000000002a")
value, integral := parsed.Uint64() // 42, true; false when the high eight bytes are set
```

Keys need not be unique: an index admits up to `ValuesPerKey` entries under one, `Count` reports how many it holds, and `Get` returns them all.
`NilKey` is the all-zero key, and keys satisfy `encoding.TextMarshaler` and `encoding.TextUnmarshaler`, so they round-trip through JSON as those same strings.

## Errors

Every failure holds a status shared with the C interface, so a fault this binding catches before the boundary reports the identity the engine would have reported for the same fault.
A `Status` is itself an `error`, so `errors.Is` matches any failure holding it, and an `*usearch.Error` adds its message:

```go
matches, err := index.On(executor).Search(usearch.RowsOf(query, 4), 10, arena, nil)
var failure *usearch.Error
switch {
case err == nil:
	fmt.Println(matches.Queries(), "queries")
case errors.Is(err, usearch.StatusContended):
	fmt.Println("the arena serves another call")
case errors.As(err, &failure):
	fmt.Println(failure.Status, failure.Message)
}
```

`Status`, like `Mood`, `Residency`, `Traversal`, `Phase`, `RefineVerdict`, `DenseMetric` and `Scalar`, holds the C enumerators' values, and its `String` spells the name the C tables own.
A handle, executor or arena used after its `Close` answers `StatusUninitialized` before anything crosses, rather than handing the library a dangling handle.
`StatusUnsupported` marks a verb declared for every residency that this one does not implement yet, and `StatusContended` a call refused rather than waited for.
A predicate, a progress handler or a value lookup must not panic: a panic unwinding through the library's frames is a bug, and nothing here recovers it.

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each takes only the placement its own hardware has, as memory domains, devices or paths with one shard per entry, and an empty list derives the placement: every NUMA node, one free device, or wherever the manifest points.

```go
inRam, err := usearch.DenseInitRam(options, []int32{0, 1})         // one graph per socket
inVram, err := usearch.DenseInitVram(options, []int32{0, 1, 2, 3}) // sharded, never replicated
onNvme, err := usearch.DenseViewNvme("index.safegraphs", &usearch.DenseViewNvmeConfig{
	Paths:           []string{"/mnt/nvme0"}, // striped across the drives named
	KeyTableDomains: []int32{0},             // keys resolved through RAM on node 0
})
```

An NVMe handle serves a checkpoint in place and is read-only, so every mutating verb fails on it with `StatusReadOnly`.
Without `KeyTableDomains`, it holds nothing per entry, so `Count`, `Get`, `Sample`, `Maxsim` and hinted searches fail with `StatusUnsupported`.
`DenseLoadNvme` takes the same config and opens a manifest writable, so `Refine` rewrites its parts one at a time, while a lone monolith fails with `StatusUnsupported`.
Bigger indexes come from merging, never mutating: shards build in RAM, serialize, and join a manifest.

## Serialization

The path extension picks the layout: a `.safegraphs` path is one monolith, and a `.safegraphs.json` path is a manifest cataloging many.

```go
err := index.On(executor).Save("index.safegraphs", nil)
loaded, err := usearch.DenseLoadRam("index.safegraphs", nil)
gpu, err := usearch.ExecutorForVram([]int32{0}, usearch.GPUClassUnknown, nil, 0)
placed, err := usearch.DenseLoadVram("index.safegraphs", &usearch.DenseLoadVramConfig{
	Executor: gpu,
	Devices:  []int32{0},
})
viewed, err := usearch.DenseViewRam("index.safegraphs")             // read-only, served from the mapping
metadata, err := usearch.DenseMetadataRead("index.safegraphs", nil) // geometry without loading a shard
```

Checkpoints are residency-portable: one file materializes into RAM, uploads to a device, or opens in place on disk, decided by the constructor that reads it.
The metadata carries only the four fields a checkpoint froze, so `ValuesPerKey` and the placement stay yours to set in the load config.
A stopped `Save` leaves leftovers every reader rejects, which are yours to remove.

In-memory buffers hold single monoliths only, through `SaveBuffer`, `DenseLoadRamBuffer` or `DenseLoadVramBuffer`, and `DenseMetadataReadBuffer`.

```go
monolith, err := index.On(executor).SaveBuffer(arena, nil)
serialized, err := monolith.Bytes()
restored, err := usearch.DenseLoadRamBuffer(serialized, nil)
```

## Dense Indexes

A `DenseIndex` holds equidimensional dense vectors, built from `DenseOptions`: `Dimensions`, `Metric`, the stored `Quantization`, and the `Connectivity` and `ValuesPerKey` bounds, where zero derives.
Every metric derives from a single inner-product kernel with pre-computed magnitudes; arbitrary metrics belong to the C++ baseline tier, forfeiting the SIMD kernel chains and the magnitude cache both.

- `DenseMetricInner`: inner product.
- `DenseMetricAngular`: cosine distance, over pre-computed magnitudes.
- `DenseMetricEuclidean`: L2 distance.
- `DenseMetricHamming`: bit-level, over `U1x8Scalar` vectors alone.
- `DenseMetricJaccard`: bit-level, over `U1x8Scalar` vectors alone.

### Rows & Scalars

Every verb is a batch over strided matrices, so a memory-mapped dataset reaches the index with no reshape and no copy, and a single vector is a one-row matrix.
`RowsOf` derives the row count and the byte pitch from a packed slice; `StridedRowsOf` takes a pitch wider than the row, which is how a memory-mapped dataset with padded rows arrives.

```go
vectors := usearch.RowsOf(embeddings, 768)                 // packed
padded := usearch.StridedRowsOf(mapped, rows, 768, 1024*4) // pitch of 1024 floats
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
var stats usearch.TraversalStats
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
	found, distances, err := matches.Query(query)
	if err != nil {
		return err
	}
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
var stats, total usearch.TraversalStats
for _, batch := range batches {
	if err := index.On(executor).Insert(batch.keys, batch.vectors, &usearch.DenseInsertConfig{Stats: &stats}); err != nil {
		return err
	}
	total.Accumulate(stats)
}
fmt.Println(total.VisitedMembers, total.ComputedDistances, total.Restarts)
```

### Filtered Search

The `Predicate` field of a `DenseSearchConfig` admits only the keys it approves.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.
Its type tells the library what one call costs:

- `usearch.Predicate` answers one key at a time, asked as candidates are scored.
- `usearch.BatchPredicate` answers a batch, asked once per round about every candidate a thread gathered, for handlers that pay per call, like a lookup in another database.
- `usearch.CheapBatchPredicate` answers a batch, asked as candidates are scored, a few keys a call.

```go
even := usearch.Predicate(func(key usearch.Key) bool {
	value, integral := key.Uint64()
	return integral && value%2 == 0
})
matches, err := index.On(executor).Search(queries, 10, arena, &usearch.DenseSearchConfig{Predicate: even})

inStock := usearch.BatchPredicate(func(keys []usearch.Key, admissions []usearch.Admission) {
	for position, available := range inventory.Lookup(keys) { // one round trip for the whole batch
		if available {
			admissions[position] = usearch.Admit
		}
	}
})
matches, err = index.On(executor).Search(queries, 10, arena, &usearch.DenseSearchConfig{Predicate: inStock})
```

Admissions start as `usearch.Reject`, and both slices live only until the handler returns.
The predicate may run on several of the executor's threads at once, so it must be safe for concurrent use, as a function shared between goroutines is.

Every long verb's config takes a `Progress` handler, called with the units finished so far and the units the call finishes if it runs to the end; answering `false` stops it wherever the verb can stop, which an `Insert` cannot.

```go
deadline := time.Now().Add(30 * time.Second)
report, err := index.On(executor).Refine(&usearch.DenseRefineConfig{
	ExpansionFactor: 128,
	Progress:        func(completed, total uint) bool { return time.Now().Before(deadline) },
})
```

A handler never runs on two threads at once, so it may hold a counter or a deadline without a lock.

### Retrieval and Streaming

`Get` exports every entry stored under one key, converted to the scalar kind you name, in one call however many there turn out to be:

```go
vectors, err := index.Get(usearch.KeyFromUint64(42), usearch.ScalarF32, arena)
// vectors.Rows() rows of vectors.Columns(), vectors.Stride() bytes apart
values, err := usearch.DenseValues[float32](vectors)
```

A cursor walks the contents, each pull writing up to `wanted` keys, and the keys each links to for `NextLinks`, into an arena; the vectors themselves come from `Get`.
`NextKeys` pulls the keys alone, and a batch of none means the walk is exhausted.
`Cursor()` yields every entry as stored, and `Cursor(starts...)` the starts' own entries first, then the entries those link to breadth-first, each once.
Cursors answer on RAM and NVMe handles, while VRAM entries are read back by key or by search.
A cursor opens on the index and keeps no executor: `cursor.NextLinks(..)` pulls on the calling goroutine, and `cursor.On(executor).NextLinks(..)` blocks on the executor named for that pull.

```go
cursor, err := index.Cursor()
if err != nil {
	return err
}
defer cursor.Close()
for {
	batch, err := cursor.On(executor).NextLinks(1024, arena)
	if err != nil || batch.Written() == 0 {
		break
	}
	keys, err := batch.Keys()
	if err != nil {
		return err
	}
	links, err := batch.Links(0)
	if err != nil {
		return err
	}
	consume(keys, links)
}
```

A search stream resumes one query's walk, so a caller that post-filters aggressively pulls until satisfied instead of guessing the wanted count upfront.
Each `Next` writes up to `wanted` next-nearest matches into an arena as the one query of a search, and a query with none means the walk is drained.
Ordering beyond the expansion horizon is best-effort nearest rather than globally exact.
No residency streams matches yet, so `SearchStream` answers `StatusUnsupported` today.

```go
stream, err := index.SearchStream(usearch.RowsOf(query, dimensions), nil)
if err != nil {
	return err
}
defer stream.Close()
for accepted := 0; accepted < 10; {
	matches, err := stream.On(executor).Next(64, arena)
	if err != nil {
		return err
	}
	found, _, err := matches.Query(0)
	if err != nil {
		return err
	}
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

`Refine` re-runs the neighbor selection over entries already in the graph, returning a `RefineReport`: the entries re-selected and changed, the islands found and reconnected, the neighborhoods re-stitched with the links added and dropped, the `Verdict`, and the health checks opening the round and closing the heal.
A batch built at a low expansion factor can be brought up to the quality of a slower build afterwards, and a `Verdict` other than `RefineImproved` is what settled looks like:

```go
for {
	report, err := index.On(executor).Refine(&usearch.DenseRefineConfig{ExpansionFactor: 128})
	if err != nil {
		return err
	}
	if report.Verdict != usearch.RefineImproved {
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
cut := &usearch.DenseReshardConfig{Seed: 42}
report, err := index.On(executor).Reshard([]*usearch.DenseIndex{index}, []*usearch.DenseIndex{left, right}, cut)
```

The index as its only source and target recuts its own shards in place into as few as fit `ShardCapacity`, keeping every entry.

```go
self := []*usearch.DenseIndex{index}
recut := &usearch.DenseReshardConfig{MemoryDomains: []int32{}, ShardCapacity: 10_000_000}
report, err := index.On(executor).Reshard(self, self, recut)
```

A `Plan` in a `DenseRefineConfig` or `DenseReshardConfig` keeps a stopped call live: the next call naming it continues where the stop came, its `Progress` says where each call stopped, and `RefinePlanFinish` or `ReshardPlanFinish` closes it and lands what the whole plan did.

```go
var plan usearch.DenseRefinePlan
for plan.Progress.Phase != usearch.PhaseDone {
	_, err := index.On(executor).Refine(&usearch.DenseRefineConfig{Plan: &plan, Progress: withinBudget})
	if err != nil && !errors.Is(err, usearch.StatusCancelled) {
		return err
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
keys, err := draws.Keys()
entryOrdinals, err := draws.EntryOrdinals()
parent, distance, err := draws.Parents().Query(0) // empty for a root
points, err := usearch.DenseValues[float32](draws.Coordinates())
```

### Sparse Multiplication

`Multiply` reads an approximate sparse product off the graph: every pair the walk scored for each left row, in traversal order, rather than only the nearest.
It runs on an inner-product index alone, and its keys, products and row offsets land in an arena as one `MatchesTape`, like `Get`'s rows.

```go
products, err := index.On(executor).Multiply(usearch.RowsOf(left, dimensions), arena, &usearch.DenseMultiplyConfig{Wanted: 64})
for row := range products.Queries() {
	columns, scores, err := products.Query(row)
	if err != nil {
		return err
	}
	accumulate(row, columns, scores)
}
```

`Required` reports the pairs the walks scored before `Wanted` cut any.

### Late Interaction

`Maxsim` scores multi-row queries against multi-vector documents: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `ValuesPerKey` above one exists for, and the wanted count is over documents rather than rows.
It runs on an inner-product index alone, since a sum of maxima needs logits rather than distances.
The keys land as `Multiply`'s pairs do, one `MatchesTape` run per query, largest score first, and a zero wanted count keeps every key a query scored.

```go
options.Metric = usearch.DenseMetricInner
options.ValuesPerKey = 512

documents, err := index.On(executor).Maxsim(usearch.RowsOf(tokens, dimensions), tokensPerQuery, 10, arena, nil)
for query := range documents.Queries() {
	keys, scores, err := documents.Query(query)
	if err != nil {
		return err
	}
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
context, err := usearch.DenseValues[float32](attended)
```

The attended rows take the queries' scalar type unless the config's `OutputsScalar` names another.
A `Carry` of one `DenseSoftmaxState` per query row, in the caller's memory rather than in the arena, folds this branch into another over the same queries, such as a recent window kept out of the index.

### Introspection

```go
description, err := index.Describe()
fmt.Println(description.Metadata.Dimensions, description.Size, description.Capacity,
	description.MemoryUsage, description.CPUClass) // the class dispatch picked on this machine

found, err := index.Contains(usearch.KeyFromUint64(42))
count, err := index.Count(usearch.KeyFromUint64(42)) // entries sharing one key

reachability, err := index.Health()
fmt.Println(reachability.Reachable, "of", reachability.Audited, "entries reachable")

fmt.Println(usearch.Version(), usearch.CapabilitiesCompiled(), usearch.CapabilitiesAvailable())
```

`Health` measures how much of the graph a search reaches from the root and answers a `Reachability`; entries it misses stay unfound until a `Refine` re-links them.
Every enumeration reads back from the name its `String` prints, as `DenseMetricFromName`, `ScalarFromName` or `CPUClassFromName` do.

## Sequence Indexes

A `SequenceIndex` searches byte strings, UTF-8 text, DNA, RNA and proteins under edit distances and alignments.
Its sequences cross as one `Tape`, the Apache Arrow layout of elements end to end around one more offset than entries: `SequencesOf` packs separate entries, and `SequencesTape` views the two runs of an Arrow column as they are.

```go
index, err := usearch.SequenceInitRam(usearch.SequenceOptions{Metric: usearch.SequenceMetricLevenshteinUTF8}, nil)
if err != nil {
	return err
}
defer index.Close()
words := usearch.SequencesOf([]byte("kitten"), []byte("sitting"), []byte("mitten"))
keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2), usearch.KeyFromUint64(3)}
if err := index.Insert(keys, words, nil); err != nil {
	return err
}

matches, err := index.Search(usearch.SequencesOf([]byte("sittin")), 1, arena, nil)
found, distances, err := matches.Query(0) // key 2, one edit away
```

`SequenceMetricNeedlemanWunsch` takes `SequenceWeights`, a 256 × 256 substitution matrix with a gap cost, and `SequenceMetricOverlap` takes `WindowWidths`; every load and view names the weights again, since no checkpoint records them.
`Get` writes a key's sequences into an arena as a `TapeView`, read with `TapeEntry[byte]`, and `Count` reports a key's entries with their total and largest byte lengths.
The receivers, reports and plans work as the dense ones do: every verb takes its own `Sequence*Config`, as `Search` takes a `SequenceSearchConfig` and `SequenceLoadRam` a `SequenceLoadRamConfig`, and `Refine` and `Reshard` take a `SequenceRefinePlan` or a `SequenceReshardPlan` and answer a `RefineReport`.

## Shape Indexes

A `ShapesIndex` searches 3D point clouds of any size, like molecules, conformers and protein backbones, invariant to rotation, reflection, translation and point order.
Its tape holds points end to end, `x`, `y` and `z` interleaved, with offsets that count points, as an Arrow `List` column lays them out:

```go
index, err := usearch.ShapesInitRam(usearch.DefaultShapesOptions(), nil) // Chamfer over f32 points
if err != nil {
	return err
}
defer index.Close()
pair := []float32{0, 0, 0, 1.1, 0, 0}
triple := []float32{0, 0, 0, 1.5, 0, 0, 0, 1.5, 0}
keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2)}
if err := index.Insert(keys, usearch.ShapesOf(pair, triple), nil); err != nil {
	return err
}

probe := []float32{0, 0, 0, 1.0, 0, 0}
matches, err := index.Search(usearch.ShapesOf(probe), 1, arena, nil)
found, _, err := matches.Query(0) // key 1
```

The metrics are Chamfer, Hausdorff, Gromov-Wasserstein and Wasserstein-Procrustes, over `f16` or `f32` storage, whatever scalar the points arrive in.
A non-zero `Points` in `ShapesOptions` freezes every shape's point count.
`Get` converts each shape to the scalar it is asked for and returns it in its inserted pose, and `Count` reports a key's total and largest point counts.
Every verb takes its own `Shapes*Config`, as `Search` takes a `ShapesSearchConfig` and `Insert` a `ShapesInsertConfig`, and the rest works as the dense verbs do.

## Geospatial Indexes

A `GeospatialIndex` searches observations on and above the Earth, like photos, video frames, drone footage and sensor fixes.
`GeospatialObservation` is the library's own eleven doubles, so a slice of them crosses with no copy; a field no device recorded stays NaN, which only widens the tests that read it, and `GeospatialAt` leaves every field but the position that way:

```go
index, err := usearch.GeospatialInitRam(usearch.GeospatialOptions{}, nil)
if err != nil {
	return err
}
defer index.Close()
stops := []usearch.GeospatialObservation{
	usearch.GeospatialAt(51.5308, -0.1238), // every other field unrecorded, NaN
	usearch.GeospatialAt(51.5320, -0.1233),
	usearch.GeospatialAt(51.5033, -0.1196),
}
keys := []usearch.Key{usearch.KeyFromUint64(1), usearch.KeyFromUint64(2), usearch.KeyFromUint64(3)}
if err := index.Insert(keys, stops, nil); err != nil {
	return err
}

nearby, err := index.Within(stops[:1], usearch.GeospatialRelationNear, 500, arena, nil)
found, chords, err := nearby.Query(0) // both stops within 500 metres, in chord metres
```

`Within` reports every entry the relation admits inside the radius, however many, as one `MatchesTape` whose `Written` counts them all, while `Search` ranks the `wanted` nearest, a zero `Metres` ranking by distance alone.
The relation is an argument of each call: `Near` widens the radius by both positions' accuracies, `Sees` and `Seen` test the view cones, and `Visible` needs a terrain, which this binding does not bind.
`Get` returns a key's observations within the stored rounding, and every verb takes its own `Geospatial*Config`.

## Apache Arrow

The converters of `c/usearch/arrow.h` view an Arrow column as a verb's input in place, and export what a call wrote into an arena as a column in place. An export pins its arena until the array is released: every call naming the arena answers `StatusContended`, and so does `Close`, which frees nothing and can be retried:

```go
vectors, err := index.Get(usearch.KeyFromUint64(42), usearch.ScalarF32, arena)
var schema usearch.ArrowSchema
var array usearch.ArrowDeviceArray
if err := usearch.RowsViewToArrow(vectors, arena, &schema, &array); err != nil {
	return err
}
defer schema.Release()
defer array.Release()

rows, err := usearch.RowsViewFromArrow(&schema, array.Array()) // the same rows, as a RowsView
matches, err := index.Search(rows, 10, other, nil)
```

`KeysViewFromArrow`, `RowsViewFromArrow` and `TapeViewFromArrow` read keys, rows and sequence or shape tapes, and `KeysViewToArrow`, `RowsViewToArrow`, `TapeViewToArrow`, `KeysTapeViewToArrow` and `MatchesTapeViewToArrow` write keys, rows, tapes, a cursor's links and ragged matches.

## Left Unbound

These C entry points also take a deferred completion, but the package calls them blocking, whatever receiver you hold:

- The load and view constructors and the metadata reads, with each config's `Executor` choosing where they fan out.
- Cursor and search-stream pulls, which block on the calling goroutine or on the executor `On` names for that pull.

These parts of the C interface have no Go counterpart yet:

- `hints` on dense and shape searches.
- Terrain: `usearch_geospatial_terrain_t` stays unset, so every geospatial handle answers the visible relation with `StatusUnsupported`, since its heights would have to outlive the handle, which no call can pin.
- `usearch_gpu_devices`, and the numeric `usearch_version`, whose name `Version` answers.

## Design

- __A thin layer over the C ABI.__
  Every name follows `c/usearch/`: a C verb `usearch_dense_<verb>` is a `<Verb>` method here, each `.go` file mirrors one header, a verb's argument structure becomes the config of its own modality, as `usearch_shapes_load_ram_t` becomes `ShapesLoadRamConfig`, and enums mirror C enums, never strings.
  A C type every modality shares keeps a modality-neutral name, as `TraversalStats`, `Reachability`, `RefineReport` and `MatchesTape` do.
- __The receiver chooses, the verb stays.__
  The index runs a verb on the calling goroutine, `On(executor)` blocks on an executor, and `Deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `Status` polls through `usearch_outcome_status` and `Wait` blocks through `usearch_outcome_wait`; a goroutine calling `Wait` overlaps the wait, and a `select` reaches it through a channel that goroutine closes.
- __No life of its own.__
  The binding starts no goroutines and keeps no queues, locks or counters; releasing a `Deferred` rides `Close`, which waits for the call first, or the collector's cleanup, which frees it only once final.
- __Arenas belong to the caller and are reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes, and its views end at the arena's next call.
- __A fixed allocation budget per call.__
  cgo pins only heap objects, so each call opens one pinned frame holding its outcome, the handles its callbacks reach Go through, its argument structure and its result.

| Path | Go | Rust |
|---|---|---|
| Blocking call | 1 frame | 0 |
| Refusal before the call | 1 `Error` | 1 error |
| Deferred submit and wait | 4: frame, `Deferred`, 2 in `AddCleanup` | 0 |
| Each callback | 1 handle | 0 |

A blocking call allocates its frame, and a refusal only the `*Error` it answers, before anything is pinned.
A deferred call adds the `Deferred` holding the frame and what `runtime.AddCleanup` spends registering its release, two allocations on Go 1.27.
Each predicate, progress handler or value lookup costs one entry in the `runtime/cgo` handle table, and a failure the engine reports copies its message into the `*Error`.
Nothing else allocates, apart from the optional inputs a call must pin, like `Stats`, a plan or NVMe paths, which pinning keeps on the heap.
`TestCallsAllocateOnlyTheirFrames` holds the first three rows, and an arena allocates only when a call writes into it.
