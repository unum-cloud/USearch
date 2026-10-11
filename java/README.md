# USearch for Java

The Java SDK binds the flat C interface in `include/usearch/` through the FFM API, so it compiles no JNI shim and ships no native artifact of its own to cross-compile.
It instantiates no index template: every verb is a downcall into the pre-instantiated `libusearch_shared`, so one compiled engine serves every language binding.
Java 25 or newer is required, and the runtime needs `--enable-native-access` for the module or class path the SDK loads from.
It exposes dense indexes; sequence, matrix and geospatial indexes follow as the C library ships them.

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
`DenseVerbs.java` writes the blocking verbs once for the index and its `On` receiver.
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
        try (Executor pool = Executor.forRam(new int[0], 0, 0);
                Arena arena = new Arena(pool);
                DenseIndex index = DenseIndex.initRam(options, new int[0])) {

            index.reserve(100);
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
A submission finding the executor's deferred table full, sized by `deferredLimit`, finishes with `Status.CONTENDED`.
Calls depending on each other are ordered by joining the first before submitting the second, since no call carries an ordering field.

## Executors

An executor owns the host threads or device streams a fanned-out verb runs on; an index owns rows.
Keeping them apart is what lets one pool serve several indexes, and what makes the cost of spawning visible at the line that spawns.
Its residency must match the residency of every handle it drives.
The index itself runs a call on the calling thread instead: single-threaded on RAM, on the handle's device and default stream on VRAM.

There is one factory per C constructor, and each takes every argument its C twin takes, the last of them the `deferredLimit` sizing its table of deferred calls, where zero takes the default.

```java
Executor.forRam(new int[] {0, 1}, 16, 0);                        // two NUMA nodes, sixteen workers
Executor.onPool(forkUnionPool, new int[0], 0);                   // a thread pool the caller already runs
Executor.forVram(new int[] {0, 1, 2, 3}, new MemorySegment[0], 0, 0); // four devices, own streams
Executor.forNvme(new String[] {"/mnt/a", "/mnt/b"}, 8, 0);       // two drives, eight submission rings
```

`forVram` takes no thread count: one host thread keeps every device fed, so the width of that pool is the device list itself.
It borrows one CUDA stream per device when given them, and caps scratch bytes per device at a non-zero `scratchLimit`.
Naming no hardware derives it: an empty domain list spreads over every memory domain the process may allocate on, an empty device list opens one free device, and an empty path list leaves every shard where its manifest already points.

```java
try (Executor pool = Executor.forRam(new int[0], 16, 0)) {
    DenseIndex.On receiver = index.on(pool);
    receiver.insert(keys, vectors);
    Matches matches = receiver.search(queries, 10, arena);
    receiver.save("index.safegraphs");
    pool.drain(Mood.PARKED);
}
```

Calls on one executor run in submission order, and handles themselves stay freely concurrent.
`drain` also sets what the executor does once it goes idle: `Mood.PARKED` lets its threads sleep, `Mood.SPINNING` keeps them hot for the next call's latency.

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
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
A stale view's segments are closed, so reading one throws `IllegalStateException`.
An arena from a VRAM executor holds device memory, which host accessors refuse with `Status.DEVICE`.

```java
try (Arena arena = new Arena(pool)) {
    for (RowsView batch : batches) {
        Matches matches = index.on(pool).search(batch, 10, arena);    // rewinds: last batch's matches are gone
        total += matches.score(0, 0);
}
}
try (Arena found = new Arena(); Arena rows = new Arena()) {          // two live results, two arenas
    Matches hits = index.search(query, 10, found);
    for (int rank = 0; rank < hits.count(0); ++rank) {
        rerank(hits.key(0, rank), index.get(hits.key(0, rank), Scalar.F32, rows));
}
}
List<Deferred<Matches>> calls = new ArrayList<>();
for (int shard = 0; shard < shards.length; ++shard) {
    calls.add(shards[shard].deferred(pool).search(queries, 10, arenas[shard]));    // one arena per submitted call
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
A null arena is refused with `Status.INVALID_ARGUMENT`, and fixed-size reports, like `DenseTraversalStats` and `describe`, stay the caller's own objects.
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
Checks the SDK runs before a call crosses report the status the engine would have reported for the same fault, and a code this build cannot name reads as `Status.INVALID_ARGUMENT`, as in C.
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
DenseIndex copied = DenseIndex.loadRam("index.safegraphs", pool, new int[0], 0);
DenseIndex uploaded = DenseIndex.loadVram("index.safegraphs", gpu, new int[] {0}, 0);
DenseIndex mapped = DenseIndex.viewRam("index.safegraphs");
DenseIndex served = DenseIndex.viewNvme("index.safegraphs.json", disks, new String[] {"/mnt/nvme0"});
```

`initRam` and `initVram` create an empty, mutable index, and the `load` family copies a checkpoint in, each taking the executor its read fans out over, or `null` for the calling thread.
`viewRam` serves a checkpoint in place through a memory mapping, and `viewNvme` from NVMe drives.
A served handle is read-only, so every mutating verb on it fails with `Status.READ_ONLY`.
Until NVMe checkpoints carry keys, `viewNvme` answers `Status.UNSUPPORTED`.

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

`DenseIndex` mirrors `include/usearch/dense_index.h`, with the modality's options, metric, telemetry and per-call settings as nested types.

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
`count`, `key` and `score` all address one query's row by its position, so no call site multiplies out an offset of its own, and slots past a query's count hold memory the engine never wrote and are refused.

The candidate-pool breadth rides each call rather than the handle, which lets a recall sweep run against one immutable index with no mutation between points.
Every verb with knobs beyond its inputs has a bare form and one taking its `Dense*Config`, named after the C structure it fills.
`Traversal.WALK`, the default, walks the graph, and `Traversal.SCAN` scores every row published when the query's scan starts, exactly, on RAM and NVMe handles.
A `DenseTraversalStats` lent to a config is overwritten with what the walk reported, so one reading always belongs to one call.

```java
DenseTraversalStats stats = new DenseTraversalStats();
index.on(pool).insert(keys, vectors, new DenseInsertConfig().expansionFactor(256));
Matches matches = index.on(pool).search(queries, 10, arena);
matches = index.on(pool).search(queries, new DenseSearchConfig(10).expansionFactor(64).stats(stats), arena);
matches = index.on(pool).search(queries, new DenseSearchConfig(10).traversal(Traversal.SCAN), arena);
System.out.println(stats.computedDistances() + " distances computed by the wider search");
```

### Retrieval: Get, Cursors & Streams

`get` returns every row the key holds, however many there turn out to be, at the natural width of the kind asked for, with the `columns` and `strideBytes` the library reported.

```java
DenseVectors every = index.get(Key.of(42), Scalar.F32, arena);
float[] values = every.segment().toArray(ValueLayout.JAVA_FLOAT);
```

A cursor walks the index contents as a sequential row scan, reading each shard's publication watermark as it enters that shard, so a row once passed is never revisited.
Each pull writes up to a wanted count of keys into the arena, and their rows beside them in the scalar asked for.

```java
try (DenseCursor cursor = index.on(pool).cursor()) {
    for (DenseBatch batch = cursor.next(1024, Scalar.F32, arena); batch.written() != 0;
            batch = cursor.next(1024, Scalar.F32, arena)) {
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
Re-selecting a settled node changes nothing, so a caller loops until the changed-node count falls below whatever rate it considers settled.

```java
DenseRefineConfig wide = new DenseRefineConfig().expansionFactor(256);
while (index.on(pool).refine(wide) > index.describe().size() / 100) {}
```

### Merging & Splitting

`merge` absorbs donor indexes, consuming them whether or not it succeeds, and no engine residency absorbs donors yet.
Splitting is not in the C library yet.

```java
index.on(pool).merge(shard0, shard1);   // both donors are closed from here on
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

`maxsim` scores multi-row queries against multi-row entries: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, so `valuesPerKey` bounds how many rows share a key and the wanted count is the "k" over documents rather than over rows.
Every query carries the same number of rows, so a padded token batch applies as is.

```java
DenseOptions tokens = new DenseOptions().dimensions(128).metric(DenseMetric.INNER).valuesPerKey(512);
try (DenseIndex index = DenseIndex.initRam(tokens, new int[0])) {
    Matches documents = index.on(pool).maxsim(queryTokens, 32, 10, arena);
}
```

It runs on the inner-product kinds alone, since a maximum over logits is what late interaction sums; a graph frozen under a distance kind reports `Status.CONFIG_MISMATCH`.

### Attention

`attend` is Flash Attention folded into the walk: every key it scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, and the outputs land in the arena already divided by the partition.
The index stores no values, so a `ValueLookup` names the row of a values matrix each scored key resolves to.
It borrows that matrix and builds its upcall once, so every call reuses both: a native matrix is read in place, a heap one is copied at construction.
The library states the scalar and width it reads each row as, and a values matrix of another kind, a row outside it, or a lookup that throws leaves the key without a row, which refuses the call with `Status.SHAPE_MISMATCH`; the first exception the lookup raised outranks that status, and is raised once the call is final.
The lookup runs on the executor's threads together, so it must tolerate concurrent calls.

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

## What This Binding Does Not Expose

The filtered search of the C interface stays unbound.
Its predicate runs on the engine's own worker threads, which are not attached to the JVM, so every candidate would pay an attach, an upcall, and an exception check.
A predicate that threw would also unwind through native frames that cannot handle it.
Filter the results of `search` instead, or narrow the index.
Progress handlers stay unbound for the same reason, which is what makes an insertion batch all-or-nothing here.

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
  Every output whose length scales with the call lands in an arena the caller passes; each call closes the FFM scope the previous call's views read through, so a stale view throws instead of reading reused memory.
- __Nothing costs anything until it is used.__
  Blocking calls take no deferred-table slot, and an arena allocates only when a call writes into it.
- __Blocking calls stay as simple as the language allows.__
  `index.search(queries, 10, arena)` takes no executor, completion or callback and returns its result; nothing about asynchrony reaches a caller who never asks for it.
- __Argument structures live in FFM arenas.__
  A blocking call fills its C argument structure in a confined arena freed on return and takes no lock; a deferred one keeps it in a shared arena until the call is final.
