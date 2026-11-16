# USearch for Java

The Java SDK binds the flat C interface in `include/usearch/` through the FFM API, so it compiles no JNI shim and ships no native artifact of its own to cross-compile.
It instantiates no index template: every verb is a downcall into the pre-instantiated `libusearch_shared`, so one compiled engine serves every language binding.
Java 25 or newer is required, and the runtime needs `--enable-native-access` for the module or class path the SDK loads from.
It exposes dense, sequence, shape and geospatial indexes.

## Installation

Gradle is the recommended route, since Maven Central does not carry the native builds; this Groovy script downloads a "fat JAR" carrying builds for Linux, Windows, macOS, and Android:

```groovy
repositories {
    mavenCentral()
    flatDir { dirs 'lib' }
}

task downloadUSearchJar {
    doLast {
        def usearchVersion = '3.0.0'
        def usearchUrl = "https://github.com/unum-cloud/USearch/releases/download/v${usearchVersion}/usearch-${usearchVersion}.jar"
        def usearchFile = file("lib/usearch-${usearchVersion}.jar")
        usearchFile.parentFile.mkdirs()
        if (!usearchFile.exists()) {
            new URL(usearchUrl).withInputStream { source -> usearchFile.withOutputStream { it << source } }
}
}
}

compileJava.dependsOn downloadUSearchJar

dependencies {
    implementation name: 'usearch', version: '3.0.0', ext: 'jar'
}
```

Gradle drives CMake for the engine, so building from source needs nothing else, and `-PusearchPrebuilt` points the suites at an engine CMake already built:

```sh
gradle clean build                                    # engine, SDK, and the Java, Kotlin and Scala suites
gradle -PusearchPrebuilt=$PWD/build_release test      # against a library built beforehand
```

The SDK looks for the shared object on the system library path first, which a locally built tree offers, and otherwise extracts the platform's copy packaged inside the JAR to a temporary directory.

### Source Layout & Imports

One source file mirrors one header.
`Types.java` carries the shared vocabulary: `Key`, `Status`, `Scalar`, `Residency`, `Traversal`, `Executor`, `Arena`, `Mood`, `Deferred`, `USearchException`, `RowsView`, and the arena-backed `Matches` and `Monolith`.
`DenseIndex.java` carries the dense modality: the index itself, its `On` and `DeferredOn` receivers, `DenseOptions`, `DenseMetric`, `DenseMetadata`, `DenseIndexDescription`, `DenseReachability`, `DenseTraversalStats`, the `Dense*Config` knobs, `ValueLookup`, the arena-backed `DenseVectors`, `DenseProducts` and `DenseBatch`, and the `DenseCursor` and `DenseSearchStream` scans.
`DenseCalls.java` writes each verb's body once, and the index, `On` and `DeferredOn` each run it blocking or deferred.
`SequenceIndex.java`, `ShapesIndex.java` and `GeospatialIndex.java` carry the other modalities, `Tape.java` the ragged batch the first two take, and `BasicCalls.java` the verbs all three spell alike.
`Library.java` loads the shared object and links every verb against it.
Both vocabularies are nested types, and a nested type imports directly, so every call site below spells the bare name:

```java
import cloud.unum.usearch.DenseIndex;
import cloud.unum.usearch.DenseIndex.DenseMetric;
import cloud.unum.usearch.DenseIndex.DenseOptions;
import cloud.unum.usearch.DenseIndex.DenseSearchConfig;
import cloud.unum.usearch.Types.Arena;
import cloud.unum.usearch.Types.Deferred;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.RowsView;
import cloud.unum.usearch.Types.Scalar;
```

## Quickstart

```java
public class Main {
    public static void main(String[] arguments) {
        DenseOptions options = new DenseOptions().dimensions(3).metric(DenseMetric.ANGULAR).quantization(Scalar.BF16);
        try (Executor pool = Executor.forRam(new int[0], null, 0);
                Arena arena = new Arena(pool);
                DenseIndex index = DenseIndex.initRam(options, new int[0])) {

            index.reserve(100); // Optional capacity hint; inserts grow on demand
            index.insert(
                new Key[] {Key.of(42), Key.of(43)},
                RowsView.of(new float[] {0.1f, 0.2f, 0.3f, 0.3f, 0.2f, 0.1f}, 3));

            RowsView query = RowsView.of(new float[] {0.1f, 0.2f, 0.3f}, 3);
            Matches matches = index.search(query, 10, arena);            // the calling thread
            matches = index.on(pool).search(query, 10, arena);           // the same verb, on an executor
            for (int rank = 0; rank < matches.count(0); ++rank) {
                System.out.println(matches.key(0, rank) + " at " + matches.score(0, rank));
}
}
}
}
```

## Receivers & Deferred Calls

The receiver chooses how a call completes, and the verb stays the same.
`DenseIndex` runs a verb on the caller's thread, `index.on(executor)` answers an `On` whose verbs block on that executor, and `index.deferred(executor)` answers a `DeferredOn` whose verbs submit there and answer a `Deferred`.
A `Deferred` owns its outcome, its argument structure and its inputs until the call is final; `status` polls without blocking, and `join` blocks, then answers or raises.
It borrows the index, executor and arena it names exactly as a blocking call does, and `join` answers the same view a blocking call would, over that arena.

```java
try (Arena arena = new Arena(pool);
        Deferred<Matches> call = index.deferred(pool).search(queries, 10, arena)) {
    if (call.status() == Status.DEFERRED) {
        // other work
}
    Matches matches = call.join();    // valid until the next call given `arena`
}
```

The first `join` frees the outcome and keeps what it answered or raised, so a later `status` or `join` repeats it; only a closed `Deferred` raises `IllegalStateException`.
Closing an unfinished `Deferred` waits for the call before freeing what it lent, so a native slab lent to one must stay open until then.
Dropping one never blocks the collector: what it lent stays reachable until the call is final, and only then is it freed.
One thread owns a `Deferred` at a time: it takes no lock, so polling, joining and closing one from two threads at once is a race, and handing it over through a queue or a future is safe.
One call holds an executor at a time, so a second call finding it held finishes with `Status.CONTENDED`, and nothing queues.
A deferred call on a caller-exclusive pool runs on the pool's thread 0, so polling `status` alone finishes it, and its progress handler and predicate run on the pool's threads; a single thread or a VRAM executor runs it at submission.
Calls depending on each other are ordered by joining the first before submitting the second, since no call carries an ordering field.

## Executors

An executor owns the host threads or device streams a fanned-out verb runs on; an index owns entries.
Keeping them apart is what lets one pool serve several indexes, and what makes the cost of spawning visible at the line that spawns.
Its residency must match the residency of every handle it drives.
The index itself runs a call on the calling thread instead: single-threaded on RAM, on the handle's device and default stream on VRAM.

There is one factory per C constructor, and each takes every argument its C twin takes.

```java
Executor.forRam(new int[] {0, 1}, null, 16);                        // two NUMA nodes, sixteen workers
Executor.forRam(new int[0], CPUClass.INTEL_HASWELL, 0);             // ceiling at Haswell
Executor.onPool(forkUnionPool, new int[0], null);                   // a thread pool the caller already runs
Executor.forVram(new int[] {0, 1, 2, 3}, null, new MemorySegment[0], 0); // four devices, own streams
Executor.forNvme(new String[] {"/mnt/a", "/mnt/b"}, null, 8);       // two drives, eight submission rings
```

`forVram` takes no thread count: one host thread keeps every device fed, so the width of that pool is the device list itself.
It borrows one CUDA stream per device when given them, and caps scratch bytes per device at a non-zero `scratchLimit`.
Naming no hardware derives it: an empty domain list spreads over every memory domain the process may allocate on, an empty device list opens one free device, and an empty path list leaves every shard where its manifest already points.

```java
try (Executor pool = Executor.forRam(new int[0], null, 16)) {
    DenseIndex.On receiver = index.on(pool);
    receiver.insert(keys, vectors);
    Matches matches = receiver.search(queries, 10, arena);
    receiver.save("index.safegraphs");
    pool.drain(Mood.PARKED);
}
```

One call holds an executor at a time, and handles themselves stay freely concurrent.
`drain` waits until no call holds the executor, then sets what it does once idle: `Mood.PARKED` lets its threads sleep, `Mood.SPINNING` keeps them hot for the next call's latency.

### Concurrency

A handle may be shared across threads.
On a RAM handle, insertions and searches overlap freely: searches read only entries published behind the visibility watermark and never write to the graph, while insertions synchronize through per-node locks and publish in order.
Serialization is the one exception, demanding a mutation-free window, though concurrent readers keep searching throughout.

Every call leases the index it names through the FFM scope its handle lives in, so closing an index while another thread's blocking call still runs on it raises `IllegalStateException` rather than freeing the handle under that call; deferred calls are waited for by the close itself.
A pull leases its cursor or stream in place of the index, and a call leases one scope only, so an executor, or the index under a pull, is closed once no other thread still calls through it.
An index, executor, arena or deferred call the program drops without closing is released once it is unreachable, and never while a call naming it is unfinished.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in an `Arena` you pass, and the verb answers a view over it without copying.
`new Arena(executor)` places it where that executor computes, decided once, and `new Arena()` makes host pages for the calling thread.
Each call rewinds its arena, so a view ends at the arena's next call and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
Reading a view after the arena's next call reads whatever that call left; closing the arena closes every view's segments, so reading one then throws `IllegalStateException`.
`arena.describe()` reports where the outputs live, and an arena from a VRAM executor holds device memory, which is the caller's to hand to device code rather than read from the host.
It grows, fills and frees on that executor's first stream, so it serves only calls on that executor, keeps it reachable, and must be closed before it; a host arena may outlive its executor.

```java
try (Arena arena = new Arena(pool)) {
    for (RowsView batch : batches) {
        Matches matches = index.on(pool).search(batch, 10, arena);    // rewinds: last batch's matches are gone
        total += matches.score(0, 0);
}
}
try (Arena found = new Arena(); Arena fetched = new Arena()) {       // two live results, two arenas
    Matches hits = index.search(query, 10, found);
    for (int rank = 0; rank < hits.count(0); ++rank) {
        rerank(hits.key(0, rank), index.get(hits.key(0, rank), Scalar.F32, fetched));
}
}
List<Deferred<Matches>> calls = new ArrayList<>();
for (int shard = 0; shard < shards.length; ++shard) {
    calls.add(shards[shard].deferred(pools[shard]).search(queries, 10, arenas[shard]));    // one executor and one arena per submitted call
}
for (Deferred<Matches> call : calls) {
    try (call) {
        consume(call.join());
}
}
```

A blocking call holds its arena until it returns, and a deferred one until it is joined, closed or collected.
A call that finds its arena serving another call finishes with `Status.CONTENDED` and rewinds nothing, so the holder's views stay readable.
Closing an arena waits for the deferred calls naming it, and joining one of them afterwards throws `IllegalStateException`.
Fixed-size reports, like `DenseTraversalStats` and `describe`, stay the caller's own objects.
Every output verb takes an arena, so there is one verb per operation with no allocating twin, and no index holds an arena of its own.

## Keys

A key is sixteen opaque bytes rendered as a lowercase dashed UUID, and integer identities park in the low eight bytes.
`Key.of(42)` therefore prints as `00000000-0000-0000-0000-00000000002a`, byte for byte the same rendering every other USearch SDK produces.
Keys need not be unique: several entries may share one, and `count` reports how many.

Those sixteen bytes are a `java.util.UUID`'s two big-endian longs, so `toUUID` and `fromUUID` convert without reordering anything.
`Key.parse` is the wider reader of the two: it takes thirty-two hexadecimal digits with dashes anywhere or nowhere, in either case, which is the spelling every USearch SDK writes and reads.
`Key` also orders its bytes unsigned, where `UUID.compareTo` reads the same two longs as signed.

```java
Key key = Key.of(42);
Key parsed = Key.parse("00000000-0000-0000-0000-00000000002a");
Key adopted = Key.fromUUID(UUID.randomUUID());
long identity = key.toLong();
```

## Errors

Every failure is an unchecked `USearchException`, since nearly every verb can report one, and its `status()` is a `Status` mirroring the C interface's `usearch_status_*_k`.
The SDK checks only what the engine cannot see, such as a key count against a row count, since the C call carries one count; everything else is the engine's status, forwarded, and a code this build cannot name reads as `Status.INVALID_ARGUMENT`, as in C.
`Status.DEFERRED` is the one non-final status, which only a `Deferred` reports before its call is final.
Any verb on a closed handle raises `IllegalStateException` instead.

```java
try {
    served.insert(keys, vectors);
} catch (USearchException failure) {
    if (failure.status() != Status.READ_ONLY) {
        throw failure;
}
}
```

`Status`, `Scalar`, `DenseMetric`, `Residency`, `Traversal` and `Mood` carry their C enumerators' values in `code()`, and their `toString` spells the names the C tables own, so nothing is compared as a string.

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each takes only the placement its own hardware has, one shard per memory domain, device or path, and an empty list derives the placement: every NUMA node, one free device, or wherever the manifest points.

```java
DenseIndex inRam = DenseIndex.initRam(options, new int[] {0, 1});             // one graph per socket
DenseIndex inVram = DenseIndex.initVram(options, new int[] {0, 1, 2, 3});     // sharded, never replicated
DenseIndex copied = DenseIndex.loadRam("index.safegraphs", pool, new int[0], null, 0, null);
DenseIndex uploaded = DenseIndex.loadVram("index.safegraphs", gpu, new int[] {0}, null, 0, null);
DenseIndex mapped = DenseIndex.viewRam("index.safegraphs");
DenseIndex served = DenseIndex.viewNvme("index.safegraphs.json", disks, new String[] {"/mnt/nvme0"}, null, new int[] {0});
```

`initRam` and `initVram` create an empty, mutable index, and the `load` family copies a checkpoint in, each taking the executor its read fans out over, or `null` for the calling thread.
`viewRam` serves a checkpoint in place through a memory mapping, and `viewNvme` from NVMe drives.
A served handle is read-only, so every mutating verb on it fails with `Status.READ_ONLY`.
`viewNvme` holds a key table in RAM on the NUMA nodes its last argument names; without one, `null`, counts, reads, samples, MaxSim and hinted searches answer `Status.UNSUPPORTED`.
`loadNvme` takes the same arguments and opens a manifest writable, so `refine` rewrites its parts one at a time, while a lone monolith answers `Status.UNSUPPORTED`.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout: a `.safegraphs` path is one monolith, a `.safegraphs.json` path is a manifest cataloging many.
A loaded index takes its geometry and metric from the checkpoint's own headers, and only the placement and `valuesPerKey` from its constructor.

```java
index.on(pool).save("index.safegraphs");
DenseIndex loaded = DenseIndex.loadRam("index.safegraphs");

DenseMetadata described = DenseIndex.metadataRead("index.safegraphs", pool);

Monolith monolith = index.saveBuffer(arena);
DenseIndex copied = DenseIndex.loadRamBuffer(monolith.segment());
```

Only the call can size a monolith, so `saveBuffer` lands it in an arena, as [Arenas](#arenas) describes, and `segment` reads it in place.

`DenseMetadata` carries the four fields a checkpoint froze: dimensions, metric, quantization and connectivity.
The placement is absent because a checkpoint records where it was written rather than where a reader must keep it, and `valuesPerKey` is absent because it bounds a live handle's mutable state rather than the serialized form.

## Dense Indexes

`DenseIndex` carries the modality's options, metric, telemetry and per-call settings as nested types.

### Rows & Scalars

Every verb is a batch over strided matrices, so a memory-mapped dataset reaches the index with no reshape and no copy, and a single vector is a one-row matrix.
`RowsView` borrows a `MemorySegment`: a heap one, which the array factories wrap, or a native one for a memory-mapped dataset.
The scalar type of the input is independent of the stored quantization, and the library converts on ingest.

```java
RowsView packed = RowsView.of(new float[] {0, 1, 1, 0}, 2);
RowsView padded = RowsView.strided(MemorySegment.ofArray(slab), rows, 768, 4096, Scalar.I8);
RowsView mapped = RowsView.of(channel.map(MapMode.READ_ONLY, 0, size, java.lang.foreign.Arena.ofAuto()), 768, Scalar.U8);
RowsView direct = RowsView.of(MemorySegment.ofBuffer(buffer), 768, Scalar.F32);
```

A heap segment is copied across the boundary for the duration of one call, since holding the JVM's own storage in place would stall every collector thread for as long as the batch runs.
A native segment is addressed in place with no copy at all, which is the argument to reach for when a batch is large.
A segment over a direct `ByteBuffer` captures the buffer's position and limit once, so moving the buffer later cannot widen what a call reads.

Quantization kinds run from the widest dynamic range down: `BF16`, the default, `F16`, `E5M2`, `E4M3`, `E3M2`, `E2M3`, `I8`, `I4X2`, `U8`, `U4X2`, `U1X8`; `F64` and `F32` cross as inputs and outputs only.
Sub-byte kinds count logical dimensions, so sixteen `U1X8` dimensions occupy two bytes per row.

### Search

Every ranked verb answers its `Matches`, read in place out of the arena: a queries-by-`columns` matrix of keys, the scores positionally paired with them, and one match count per query.
The wanted count is the "k" in kANN and bounds `columns`, the row width of both matrices the call wrote, and a recall sweep reuses one arena across every point, so it allocates once.
`count`, `key` and `score` all address one query by its position, so no call site multiplies out an offset of its own, and slots past a query's count hold memory the engine never wrote and are refused.

The breadth of the candidates rides each call rather than the handle, which lets a recall sweep run against one immutable index with no mutation between points.
Every verb with knobs beyond its inputs has a bare form and one taking its `Dense*Config`, named after the C structure it fills.
`Traversal.WALK`, the default, walks the graph, and `Traversal.SCAN` scores every entry published when the query's scan starts, exactly, on RAM and NVMe handles.
A `DenseTraversalStats` lent to a config is overwritten with what the walk reported, so one reading always belongs to one call.

```java
DenseTraversalStats stats = new DenseTraversalStats();
index.on(pool).insert(keys, vectors, new DenseInsertConfig().expansionFactor(256));
Matches matches = index.on(pool).search(queries, 10, arena);
matches = index.on(pool).search(queries, new DenseSearchConfig(10).expansionFactor(64).stats(stats), arena);
matches = index.on(pool).search(queries, new DenseSearchConfig(10).traversal(Traversal.SCAN), arena);
System.out.println(stats.computedDistances() + " distances computed by the wider search");
```

### Retrieval and Streaming

`get` returns every vector the key holds, however many there turn out to be, at the natural width of the kind asked for, with the `columns` and `strideBytes` the library reported.

```java
DenseVectors every = index.get(Key.of(42), Scalar.F32, arena);
float[] values = every.segment().toArray(ValueLayout.JAVA_FLOAT);
```

A cursor walks the index contents, reading each shard's publication watermark as it enters that shard, so an entry once passed is never revisited.
Without start keys it yields every entry as stored, and `cursor(starts...)` yields the starts' own entries first, then the entries those link to breadth-first, each once.
Each pull writes up to a wanted count of keys into the arena, and `nextLinks` the keys each links to beside them; the vectors themselves come from `get`.

```java
try (DenseCursor cursor = index.on(pool).cursor()) {
    for (DenseBatch batch = cursor.nextLinks(1024, arena); batch.written() != 0;
            batch = cursor.nextLinks(1024, arena)) {
        export(batch);
    }
}
```

A search stream is a resumable best-first traversal of one query row, so a caller that post-filters aggressively pulls until satisfied instead of guessing "k" upfront; no engine residency implements it yet, so the engine answers `Status.UNSUPPORTED`.

```java
try (DenseSearchStream stream = index.on(pool).searchStream(query)) {
    Matches next = stream.next(16, arena);
}
```

### Refinement

Refinement re-selects the neighbors of every stored entry against the finished graph, improving edges that were chosen when the graph around them was still incomplete.
It is a build step rather than a repair: a cheap insertion pass followed by one refinement reaches a better graph, in less total time, than a single expensive insertion pass, so a wide expansion factor belongs here rather than in `insert`.
It answers a `RefineReport` of the nodes re-selected and changed, the islands found and reconnected, the neighborhoods re-stitched with the links added and dropped, the `verdict`, and the `opening` and `reachability` audits around the round.
A caller loops until the `verdict` is no longer `RefineVerdict.IMPROVED`.

```java
DenseRefineConfig wide = new DenseRefineConfig().expansionFactor(256);
while (index.on(pool).refine(wide).verdict() == DenseIndex.RefineVerdict.IMPROVED) {}
```

A `DenseRefinePlan` passed in the config makes a round resumable: a cancel from the progress handler answers `Status.CANCELLED` and leaves the plan open, the next `refine` naming it continues where the cancel came, and `plan.phase()` reports where the last call stopped.
`refinePlanFinish` on the receiver ends it with the whole plan's report, blocking or deferred; a plan dropped open is never finished and keeps the index closed to insertions.
On a loaded NVMe index, `memoryLimit` caps the bytes held for the one part a round rewrites, and a smaller one streams that part's rounds from the drives.

```java
DenseRefinePlan plan = new DenseRefinePlan();
DenseRefineConfig resumable = new DenseRefineConfig().plan(plan).progress((completed, total) -> !paused());
try {
    index.on(pool).refine(resumable);
} catch (USearchException stopped) {
    if (stopped.status() != Status.CANCELLED) throw stopped;
}
RefineReport report = index.on(pool).refinePlanFinish(plan);
```

### Resharding

`reshard` moves the entries of its sources into its targets, keeping their links rather than re-inserting their vectors, then runs a round on every target that took entries.
Several sources into one target among them gather into it, one source into several empty targets cuts it into graph-local parts, and one index as its own only source and target recuts its shards toward the configured placement.
Every index stays yours: a source the targets do not name ends empty and live.
It runs on the receiver's executor, answers a `RefineReport` of its closing rounds, and takes a `DenseReshardPlan` as `refine` takes its plan, which `reshardPlanFinish` ends.

```java
index.on(pool).reshard(new DenseIndex[] {index, shard0, shard1}, new DenseIndex[] {index}, new DenseReshardConfig());
index.on(pool).reshard(new DenseIndex[] {index}, new DenseIndex[] {left, right}, new DenseReshardConfig().seed(7));
index.on(pool).reshard(
        new DenseIndex[] {index}, new DenseIndex[] {index},
        new DenseReshardConfig().memoryDomains(new int[0]).shardCapacity(10_000_000));
```

### Sampling

`sample` draws keys around a focus, each hanging from a shown or earlier draw, and places them in three dimensions beside the keys already shown, so a view grows one connected sample as it zooms.

```java
Sample drawn = index.on(pool).sample(
        256, arena, new DenseSampleConfig().shown(shown, shownCoordinates).focus(near).seed(42));
float x = drawn.coordinate(0, 0);
```

### Sparse Multiplication

`multiply` is the linear-algebra sibling of `search`: it reports every pair the traversal scored rather than only the nearest, as a ragged product with one run of pairs per left row, in traversal order.
It runs on the inner-product kinds alone, and a non-zero wanted count keeps the first pairs each row's walk scored.
The keys, products and per-row offsets all land in the arena, so a steady loop allocates none.

```java
DenseProducts products = index.on(pool).multiply(left, arena);
for (int row = 0; row < products.queries(); ++row) {
    for (long rank = 0; rank < products.count(row); ++rank) {
        System.out.println(products.key(row, rank) + " x " + products.score(row, rank));
}
}
```

### Late Interaction

`maxsim` scores multi-row queries against multi-vector documents: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, so `valuesPerKey` bounds how many rows share a key and the wanted count is the "k" over documents rather than over rows.
Every query carries the same number of rows, so a padded token batch applies as is.
Each query's keys land largest score first in the same ragged `DenseProducts` that `multiply` returns, and a wanted count of zero keeps every key the query scored.

```java
DenseOptions tokens = new DenseOptions().dimensions(128).metric(DenseMetric.INNER).valuesPerKey(512);
try (DenseIndex index = DenseIndex.initRam(tokens, new int[0])) {
    DenseProducts documents = index.on(pool).maxsim(queryTokens, 32, 10, arena);
    for (long rank = 0; rank < documents.count(0); ++rank) {
        System.out.println(documents.key(0, rank) + " scores " + documents.score(0, rank));
    }
}
```

It runs on the inner-product kinds alone, since a maximum over logits is what late interaction sums; a graph frozen under a distance kind reports `Status.CONFIG_MISMATCH`.

### Attention

`attend` is Flash Attention folded into the walk: every key it scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, and the outputs land in the arena already divided by the partition.
The index stores no values, so a `ValueLookup` names the row of a values matrix each scored key resolves to.
It borrows that matrix and builds its upcall once, so every call reuses both: a native matrix is read in place, a heap one is copied at construction.
The library states the scalar and width it reads each row as, and a values matrix of another kind or a row outside it leaves the key without a row, which refuses the call with `Status.SHAPE_MISMATCH`.
The lookup runs on the executor's threads together, so it must tolerate concurrent calls, and it must not throw: an exception escaping it terminates the process.
It is a host function, so it serves RAM handles alone: a VRAM walk resolves rows through a device function, which this binding does not build.

```java
ValueLookup cache = new ValueLookup(RowsView.of(valueRows, 128, Scalar.BF16), rowOf::get);
DenseSoftmaxState[] carry = new DenseSoftmaxState[queries.rows()];
DenseVectors attended = index.on(pool).attend(queries, cache, new DenseAttendConfig().headCount(8).carry(carry), arena);
```

With `headCount(8)`, row `t * 8 + h` of the queries is head `h` of token `t`, and the heads of one token share its walk.
The `carry` holds one `DenseSoftmaxState` per query row, lent by the caller rather than placed in the arena, which folds this branch into another over the same queries, such as a recent window kept out of the index.
`DenseAttendConfig.outputsScalar` picks the scalar the outputs land as, the queries' own by default.

### Introspection

```java
long entries = index.count(Key.of(42));
boolean present = index.contains(Key.of(42));

DenseIndexDescription description = index.describe();
DenseReachability reachability = index.health();
System.out.println(reachability.reachable() + " of " + reachability.audited() + " entries reachable");
System.out.println(Types.version() + " built for " + Types.capabilitiesCompiled() + ", here " + Types.capabilitiesAvailable());
```

`describe` reports the frozen `DenseMetadata` with the size, capacity, memory usage and dispatched capabilities in one call.
`health` audits how much of the graph a search reaches from its landmarks; entries it misses stay unfound until a `refine` re-links them.
`Types.capabilitiesCompiled` lists the hardware capabilities the linked binary was compiled with, and `Types.capabilitiesAvailable` those the current machine offers.

## Sequence Indexes

A sequence index searches byte strings, UTF-8 text, DNA and proteins under edit distances and alignments.
Variable-length entries cross as a `Tape`, the Apache Arrow layout: elements end to end and one more offset than entries, which `Tape.ofStrings` and `Tape.ofBytes` pack and `Tape.of` borrows from a native segment in place:

```java
try (Executor pool = Executor.forRam(new int[0], null, 0);
        SequenceIndex words = SequenceIndex.initRam(
                new SequenceOptions().metric(SequenceMetric.LEVENSHTEIN_UTF8), new int[0]);
        Arena arena = new Arena(pool)) {
    Key[] keys = {Key.of(1), Key.of(2), Key.of(3)};
    words.on(pool).insert(keys, Tape.ofStrings("kitten", "sitting", "mitten"));
    Matches matches = words.on(pool).search(Tape.ofStrings("sitting"), 1, arena);
    SequenceKeyStats stats = words.count(Key.of(2));
    System.out.println(matches.key(0, 0) + " at " + matches.score(0, 0) + ", " + stats.totalBytes() + " bytes");
}
```

`NEEDLEMAN_WUNSCH` takes `SequenceWeights`, a 256 × 256 substitution matrix with a gap cost, and `OVERLAP` takes `windowWidths`; every load names the weights again, since no checkpoint records them.
`get` writes a key's sequences into an arena as a `Tape`, and `count` reports a key's total and largest byte lengths.
`SequenceMetric.fromName` reads back the spelling `toString` gives, refusing any other with `Status.INVALID_ARGUMENT`.
The verbs, receivers, plans and reports are the dense ones: `refine`, `reshard` and their `SequenceRefinePlan` and `SequenceReshardPlan`, cursors and `sample`, all reporting through `RefineReport` and `DenseTraversalStats`.

## Shape Indexes

A shape index searches 3D point clouds of any size, like molecules and conformers, invariant to rotation, reflection, translation and point order.
Its tape holds points end to end, `x`, `y` and `z` interleaved, with offsets that count points:

```java
try (Executor pool = Executor.forRam(new int[0], null, 0);
        ShapesIndex molecules =
                ShapesIndex.initRam(new ShapesOptions().metric(ShapesMetric.HAUSDORFF), new int[0]);
        Arena arena = new Arena(pool)) {
    Tape atoms = Tape.ofPoints(new float[] {0, 0, 0, 1.1f, 0, 0}, new float[] {0, 0, 0, 1.5f, 0, 0, 0, 1.5f, 0});
    molecules.on(pool).insert(new Key[] {Key.of(1), Key.of(2)}, atoms);
    Tape stored = molecules.on(pool).get(Key.of(2), Scalar.F32, arena);
    System.out.println(stored.count() + " entry of " + stored.points(0).length / 3 + " points");
}
```

The metrics are Chamfer, Hausdorff, Gromov-Wasserstein and Wasserstein-Procrustes, over `F16` or `F32` storage, whatever scalar the points arrive in, and `ShapesMetric.fromName` reads one back from its spelling.
A non-zero `points` freezes every shape's point count.
`get` converts each shape to the scalar it is asked for and returns it in its inserted pose, and `count` reports a key's total and largest point counts.

## Geospatial Indexes

A geospatial index searches observations on and above the Earth, like photos, video frames and sensor fixes.
An `Observation` is the library's own eleven doubles; a field no device recorded stays NaN, which only widens the tests that read it, and `Observation.at` fills every field but the position that way:

```java
try (Executor pool = Executor.forRam(new int[0], null, 0);
        GeospatialIndex places = GeospatialIndex.initRam(new GeospatialOptions(), new int[0]);
        Arena arena = new Arena(pool)) {
    Observation[] stops = {
        Observation.at(51.5308, -0.1238), Observation.at(51.5320, -0.1233), Observation.at(51.5033, -0.1196)
    };
    places.on(pool).insert(new Key[] {Key.of(1), Key.of(2), Key.of(3)}, stops);
    Observation[] station = {Observation.at(51.5308, -0.1238)};
    DenseProducts nearby = places.on(pool).within(station, GeospatialRelation.NEAR, new Tolerance(500, 0), arena);
    System.out.println(nearby.count(0) + " stops within 500 m");
}
```

`within` reports every entry the relation admits inside the radius, however many, while `search` ranks the `wanted` nearest, a zero `Tolerance.metres` ranking by distance alone.
The relation is an argument of each call: `NEAR` widens the radius by both positions' accuracies, `SEES` and `SEEN` test the view cones, and `VISIBLE` needs the terrain `GeospatialOptions.terrain` lends, refusing without one.
`get` returns a key's observations within the stored rounding.

## Kotlin & Scala

The published surface is Java, and both other JVM languages consume it directly, so there is no separate artifact for either.

Nullness is part of the contract.
The package is `@NullMarked`, so a Kotlin caller gets real non-null types across the whole boundary rather than platform types, and the compiler refuses a null where the API does not accept one.
`AutoCloseable` is Kotlin's `use`, and a coroutine waits on a `Deferred` with `withContext(Dispatchers.IO)`:

```kotlin
val index: DenseIndex = DenseIndex.initRam(DenseOptions().dimensions(2).quantization(Scalar.BF16), intArrayOf())
index.use {
Arena(pool).use { arena ->
    val matches = withContext(Dispatchers.IO) { index.deferred(pool).search(queries, 10, arena).use { it.join() } }
    println(matches.key(0, 0))          // valid until the next call given `arena`, or its close
}
}
```

Scala reads no nullness annotations, so what it gets instead is that the array-based batch verbs take `Array[Key]` and `Array[Float]` with no conversion, since both are JVM arrays already:

```scala
val keys: Array[Key] = Array(Key.of(1), Key.of(2))
index.on(pool).insert(keys, RowsView.of(Array(1.0f, 0.0f, 0.0f, 1.0f), 2))
```

## Callbacks

A `java.util.function.Predicate<Key>` on the search, multiply, attention and MaxSim configs decides which candidate keys a walk may report, and a `Progress` on every long verb hears its advance and cancels it by answering false, which the call reports as `Status.CANCELLED`.
An insertion is all-or-nothing, so its progress handler cannot cancel it.
Each runs as an upcall straight into your object, the predicate on the executor's threads together and the progress handler never concurrently.
Neither may throw: an exception escaping a callback terminates the process.

```java
DenseSearchConfig filtered = new DenseSearchConfig(10).predicate(key -> allowed.contains(key));
DenseSearchConfig watched = new DenseSearchConfig(10).progress((completed, total) -> !cancelRequested());
index.save("index.safegraphs", (completed, total) -> { bar.set(completed, total); return true; });
```

## What This Binding Does Not Expose

`cursor` and `searchStream` pull one batch at a time, blocking, so a `DeferredOn` lacks them, and constructors address no index yet, so no receiver offers them.

## Design

- __A thin layer over the C ABI.__
  Every name follows `include/usearch/`: a C verb `usearch_dense_<verb>` is a camel-cased `<verb>` here, a C type drops `usearch_` and `_t`, and enums mirror C enums, never strings.
  Every wrapper's documentation names the C symbol it calls, or says it is binding-only.
- __The receiver chooses, the verb stays.__
  The index runs a verb on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls through `usearch_outcome_status`, `join` blocks through `usearch_outcome_wait`, and a `Deferred` is `AutoCloseable`, so Java waits in try-with-resources or on a virtual thread, and Kotlin in `withContext(Dispatchers.IO)`.
- __No life of its own.__
  The binding starts no threads and keeps no queues; releasing a `Deferred` rides try-with-resources, whose `close` waits for the call first, and the collector frees a dropped one only once it is final.
- __Arenas are lent and reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes, and its views end at the arena's next call; closing the arena closes their FFM scope, so a view of a freed arena throws instead of reading freed memory.
- __Nothing costs anything until it is used.__
  Submitting a call allocates nothing, and an arena allocates only when a call writes into it.
- __Blocking calls stay as simple as the language allows.__
  `index.search(queries, 10, arena)` takes no executor, completion or callback and returns its result; nothing about asynchrony reaches a caller who never asks for it.
- __Argument structures live in FFM arenas.__
  A blocking call fills its C argument structure in a confined arena freed on return and takes no lock; a deferred one keeps it in a shared arena until the call is final.
