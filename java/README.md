# USearch for Java

The Java SDK reaches the flat C interface in `include/usearch/types.h` and `include/usearch/dense_index.h` through the FFM API, so it carries no native boundary of its own — no JNI shim to compile, and one fewer artifact to cross-compile.
It instantiates no index template: every verb is a downcall onto the pre-instantiated `libusearch_shared`, so one compiled engine serves every language binding.
Java 25 or newer is required, and the runtime needs `--enable-native-access` for the module or class path the SDK loads from.

One source file mirrors one header.
`Types.java` carries the shared vocabulary — `Key`, `Status`, `Scalar`, `Executor`, `Arena`, `Mood`, `Deferred`, `USearchException`, `RowsView`, `Matches`, `Monolith`.
`DenseIndex.java` carries the dense modality — the index itself, its `On` and `DeferredOn` receivers, `DenseOptions`, `DenseMetric`, `DenseMetadata`, `DenseIndexDescription`, `DenseReachability`, `DenseTraversalStats`, the `Dense*Config` knobs, `ValueLookup`, the arena-backed `Vectors` and `Products`, and the `DenseCursor` and `DenseSearchStream` scans.
`DenseVerbs.java` writes the blocking verbs once for the index and its `On` receiver.
`Library.java` loads the shared object and links every verb against it.
Both vocabularies are nested types, and a nested type imports directly, so every call site below spells the bare name:

```java
import cloud.unum.usearch.DenseIndex;
import cloud.unum.usearch.DenseIndex.DenseMetric;
import cloud.unum.usearch.DenseIndex.DenseOptions;
import cloud.unum.usearch.DenseIndex.DenseSearchConfig;
import cloud.unum.usearch.Types.Deferred;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.RowsView;
import cloud.unum.usearch.Types.Scalar;
```

## Installation

Installation via Maven Central is not supported due to Continuous Delivery complexity and poor support for native builds.
Gradle installation is the recommended approach.
For the most up-to-date version, the following Groovy script downloads a "fat JAR" carrying builds for Linux, Windows, macOS, and Android:

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

## Quickstart

Every verb is a batch over strided matrices, so a memory-mapped dataset reaches the index with no reshape and no copy, and a single vector is a one-row matrix.
Every failure is a `USearchException` whose `Status` mirrors the C interface's `usearch_status_*_k`.
The index runs every verb on the calling thread, and `index.on(executor)` runs the same verb on the threads an executor owns.

```java
public class Main {
    public static void main(String[] arguments) {
        DenseOptions options = new DenseOptions().dimensions(3).metric(DenseMetric.ANGULAR).quantization(Scalar.F32);
        try (Executor pool = Executor.forRam(new int[0], 0, 0);
                DenseIndex index = DenseIndex.initRam(options, new int[0])) {

            index.reserve(100);
            index.insert(
                new Key[] {Key.of(42), Key.of(43)},
                RowsView.of(new float[] {0.1f, 0.2f, 0.3f, 0.3f, 0.2f, 0.1f}, 3));

            RowsView query = RowsView.of(new float[] {0.1f, 0.2f, 0.3f}, 3);
            Matches matches = index.search(query, 10);            // the calling thread
            matches = index.on(pool).search(query, 10);           // the same verb, on an executor
            for (int rank = 0; rank < matches.count(0); ++rank) {
                System.out.println(matches.key(0, rank) + " at " + matches.score(0, rank));
            }
        }
    }
}
```

## Receivers and Deferred Calls

The receiver chooses how a call completes, and the verb stays the same.
`DenseIndex` runs a verb on the caller's thread, `index.on(executor)` answers an `On` whose verbs block on that executor, and `index.deferred(executor)` answers a `DeferredOn` whose verbs submit there and answer a `Deferred`.
A `Deferred` owns its outcome, its argument structure, its inputs, its outputs and its arena until the call is final; `status` polls without blocking, and `join` blocks, then answers or raises.

```kotlin
val matches = withContext(Dispatchers.IO) { call.join() }    // in a Kotlin coroutine
```

```java
try (Deferred<Matches> call = index.deferred(pool).search(queries, 10)) {
    if (call.status() == Status.DEFERRED) {
        // other work
    }
    Matches matches = call.join();
}
```

The first `join` frees the outcome and keeps what it answered or raised, so a later `status` or `join` repeats it; only a closed `Deferred` raises `IllegalStateException`.
Closing an unfinished `Deferred`, or dropping it, waits for the call before freeing what it lent, so a native slab lent to one must stay open until then.
An executor without threads of its own may hold a deferred call until its next `drain`, blocking call or close, so `drain` it before a `join` there.
A submission finding the executor's deferred table full, sized by `deferredLimit`, finishes with `Status.CONTENDED`.
Calls depending on each other are ordered by joining the first before submitting the second, since no call carries an ordering field.
`cursor` and `searchStream` pull one batch at a time, blocking, so a `DeferredOn` lacks them, and constructors address no index yet, so no receiver offers them.

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

## Matrices and Quantization

`RowsView` borrows a `MemorySegment`: a heap one, which the array factories wrap, or a native one for a memory-mapped dataset.
The scalar type of the input is independent of the stored quantization, and the library converts on ingest.

```java
RowsView packed = RowsView.of(new float[] {0, 1, 1, 0}, 2);
RowsView padded = RowsView.strided(MemorySegment.ofArray(slab), rows, 768, 4096, Scalar.I8);
RowsView mapped = RowsView.of(channel.map(MapMode.READ_ONLY, 0, size, arena), 768, Scalar.U8);
RowsView direct = RowsView.of(MemorySegment.ofBuffer(buffer), 768, Scalar.F32);
```

A heap segment is copied across the boundary for the duration of one call, since holding the JVM's own storage in place would stall every collector thread for as long as the batch runs.
A native segment is addressed in place with no copy at all, which is the argument to reach for when a batch is large.
A segment over a direct `ByteBuffer` captures the buffer's position and limit once, so moving the buffer later cannot widen what a call reads.

Quantization kinds run from the widest dynamic range down: `F64`, `F32`, `BF16`, `F16`, `E5M2`, `E4M3`, `E3M2`, `E2M3`, `I8`, `I4X2`, `U8`, `U4X2`, `U1X8`.
Sub-byte kinds count logical dimensions, so sixteen `U1X8` dimensions occupy two bytes per row.

## Ranked Outputs

Every ranked verb answers its `Matches`: three slabs — a queries-by-wanted matrix of keys, the scores positionally paired with them, and one match count per query.
Given a wanted count, a call allocates them; given a `DenseSearchConfig` over `Matches` the caller already holds, it lands in those, so a recall sweep sizes them once and reuses them across every point.

```java
Matches fresh = index.on(pool).search(queries, 10);
Matches sized = Matches.allocate(queries.rows(), 10);
Matches adopted = Matches.of(keySlab, scoreSlab, countSlab);
index.on(pool).search(queries, new DenseSearchConfig(sized));
```

The wanted count is the "k" in kANN and the row width of both matrices, and `Matches.of` reads it off the slab lengths rather than taking it as an argument, so the room reserved and the count asked for cannot disagree.
Slabs that are not the same shape, or that do not divide into one row per query, are refused with `Status.SHAPE_MISMATCH`.
Native slabs are written in place, and heap ones are staged and copied back once the call is final.

`count`, `key` and `score` all address one query's row by its position, so no call site multiplies out an offset of its own, and slots past a query's count hold memory the engine never wrote and are refused.

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each takes only the placement its own hardware has - memory domains, devices, or paths, one shard per entry - and an empty list derives the placement: every NUMA node, one free device, or wherever the manifest points.

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
Naming no hardware derives it — an empty domain list spreads over every memory domain the process may allocate on, an empty device list opens one free device, and an empty path list leaves every shard where its manifest already points.

```java
try (Executor pool = Executor.forRam(new int[0], 16, 0)) {
    DenseIndex.On receiver = index.on(pool);
    receiver.insert(keys, vectors);
    Matches matches = receiver.search(queries, 10);
    receiver.save("index.safegraphs");
    pool.drain(Mood.PARKED);
}
```

Calls on one executor run in submission order, and handles themselves stay freely concurrent.
`drain` also sets what the executor does once it goes idle: `Mood.PARKED` lets its threads sleep, `Mood.SPINNING` keeps them hot for the next call's latency.

## Per-Call Settings and Telemetry

The candidate-pool breadth rides each call rather than the handle, which lets a recall sweep run against one immutable index with no mutation between points.
Every verb with knobs beyond its inputs has a bare form and one taking its `Dense*Config`, named after the C structure it fills.
A `DenseTraversalStats` lent to a config is overwritten with what the walk reported, so one reading always belongs to one call.

```java
DenseTraversalStats stats = new DenseTraversalStats();
index.on(pool).insert(keys, vectors, new DenseInsertConfig().expansionFactor(256));
index.on(pool).search(queries, new DenseSearchConfig(10).expansionFactor(64).stats(stats));
System.out.println(stats.computedDistances() + " distances computed");
```

## Refinement

Refinement re-selects the neighbors of every stored entry against the finished graph, improving edges that were chosen when the graph around them was still incomplete.
It is a build step rather than a repair: a cheap insertion pass followed by one refinement reaches a better graph, in less total time, than a single expensive insertion pass, so a wide expansion factor belongs here rather than in `insert`.
Re-selecting a settled node changes nothing, so a caller loops until the changed-node count falls below whatever rate it considers settled.

```java
DenseRefineConfig wide = new DenseRefineConfig().expansionFactor(256);
while (index.on(pool).refine(wide) > index.describe().size() / 100) {}
```

## Late Interaction

`maxsim` scores multi-row queries against multi-row entries: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, so `valuesPerKey` bounds how many rows share a key and the wanted count is the "k" over documents rather than over rows.
Every query carries the same number of rows, so a padded token batch applies as is.

```java
DenseOptions tokens = new DenseOptions().dimensions(128).metric(DenseMetric.INNER).valuesPerKey(512);
try (DenseIndex index = DenseIndex.initRam(tokens, new int[0])) {
    Matches documents = index.on(pool).maxsim(queryTokens, 32, 10);
}
```

It runs on the inner-product kinds alone, since a maximum over logits is what late interaction sums; a graph frozen under a distance kind reports `Status.CONFIG_MISMATCH`.

## Attention

`attend` is Flash Attention folded into the walk: every key it scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, and the outputs come back already divided by the partition.
The index stores no values, so a `ValueLookup` names the row of a values matrix each scored key resolves to.
The library states the scalar and width it reads each row as, and a values matrix of another kind, a row outside it, or a lookup that throws leaves the key without a row, which refuses the call with `Status.SHAPE_MISMATCH`; an exception the lookup raised outranks that status.
The lookup runs on the executor's threads together, so it must tolerate concurrent calls.

```java
ValueLookup cache = new ValueLookup(RowsView.of(valueRows, 128, Scalar.BF16), rowOf::get);
DenseSoftmaxState[] carry = new DenseSoftmaxState[queries.rows()];
index.on(pool).attend(queries, cache, new DenseAttendConfig(outputs).headCount(8).carry(carry));
```

With `headCount(8)`, row `t * 8 + h` of the queries is head `h` of token `t`, and the heads of one token share its walk.
The `carry` holds one `DenseSoftmaxState` per query row, which folds this branch into another over the same queries, such as a recent window kept out of the index.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout: a `.safegraphs` path is one monolith, a `.safegraphs.json` path is a manifest cataloging many.
A loaded index takes its geometry and metric from the checkpoint's own headers, and only the placement and `valuesPerKey` from its constructor.

```java
index.on(pool).save("index.safegraphs");
DenseIndex loaded = DenseIndex.loadRam("index.safegraphs");

DenseMetadata described = DenseIndex.metadataRead("index.safegraphs", pool);

try (Monolith monolith = index.saveBuffer()) {
    DenseIndex copied = DenseIndex.loadRamBuffer(monolith.segment());
}
```

Only the call can size a monolith, so `saveBuffer` lands it in an arena, as the next section describes, and `segment` reads it in place.

`DenseMetadata` carries the four fields a checkpoint froze - dimensions, metric, quantization and connectivity.
The placement is absent because a checkpoint records where it was written rather than where a reader must keep it, and `valuesPerKey` is absent because it bounds a live handle's mutable state rather than the serialized form.

## Arenas

Outputs only the call can size — the rows of `get`, the products of `multiply`, the monolith of `saveBuffer` — land in an `Arena` the library allocates where the executor computes, and the result reads them in place with no copy.
Each of these verbs takes an optional arena:

- Passing none gives the result a fresh arena of its own, freed when the result is closed, so the result outlives every later call.
- Passing an arena reuses its memory, so a steady loop allocates once, and the result is valid until that arena is handed to another call or closed.

Reading a result after its arena moved on raises `IllegalStateException` rather than touching freed memory, and so does joining a deferred call whose arena a later call already rewound.
An arena serves one call at a time and takes no lock, so a second call submitted while the first runs fails with `Status.CONTENDED`.
One the program drops without closing is freed once neither it nor any result read out of it is reachable.

```java
try (Arena arena = new Arena()) {
    for (Key key : keys) {
        Vectors rows = index.on(pool).get(key, Scalar.F32, arena);
        consume(rows.segment(), rows.rows(), rows.strideBytes()); // valid until the next call takes `arena`
    }
}
```

## Reading Back

```java
long entries = index.count(Key.of(42));
boolean present = index.contains(Key.of(42));

try (Vectors every = index.get(Key.of(42), Scalar.F32)) {
    float[] values = every.segment().toArray(ValueLayout.JAVA_FLOAT);
}

DenseIndexDescription description = index.describe();
DenseReachability reachability = index.health();
System.out.println(reachability.reachable() + " of " + reachability.audited() + " entries reachable");
```

`get` returns every row the key holds, however many there turn out to be, at the natural width of the kind asked for, with the `columns` and `strideBytes` the library reported.
`describe` reports the frozen `DenseMetadata` with the size, capacity, memory usage and dispatched capabilities in one call.
`health` audits how much of the graph a search reaches from its landmarks; entries it misses stay unfound until a `refine` re-links them.

## Sparse Products

`multiply` is the linear-algebra sibling of `search`: it reports every pair the traversal scored rather than only the nearest, as a ragged product with one run of pairs per left row, in traversal order.
It runs on the inner-product kinds alone, and a non-zero wanted count keeps the first pairs each row's walk scored.

```java
try (Products products = index.on(pool).multiply(left)) {
    for (int row = 0; row < products.queries(); ++row) {
        for (long rank = 0; rank < products.count(row); ++rank) {
            System.out.println(products.key(row, rank) + " x " + products.score(row, rank));
        }
    }
}
```

## Scans and Merges

A cursor walks the index contents as a sequential row scan, reading each shard's publication watermark as it enters that shard, so a row once passed is never revisited.
The caller states the batch and the row width and scalar it wants the rows in, through the arrays and the matrix it lends.

```java
try (DenseCursor cursor = index.on(pool).cursor()) {
    Key[] keys = new Key[1024];
    RowsView rows = RowsView.allocate(1024, 768, Scalar.F32);
    for (long written = cursor.next(keys, rows); written != 0; written = cursor.next(keys, rows)) {
        export(keys, rows, written);
    }
}
```

A search stream is a resumable best-first traversal of one query row, so a caller that post-filters aggressively pulls until satisfied instead of guessing "k" upfront; no engine residency implements it yet, so the engine answers `Status.UNSUPPORTED`.
`merge` absorbs donor indexes, consuming them whether or not it succeeds, and no engine residency absorbs donors yet either.

```java
try (DenseSearchStream stream = index.on(pool).searchStream(query)) {
    Matches next = stream.next(16);
}
index.on(pool).merge(shard0, shard1);   // both donors are closed from here on
```

## Concurrency

A handle may be shared across threads.
On a RAM handle, insertions and searches overlap freely: searches read only entries published behind the visibility watermark and never write to the graph, while insertions synchronize through per-node locks and publish in order.
Serialization is the one exception, demanding a mutation-free window, though concurrent readers keep searching throughout.

Every call leases the index it names through the FFM scope its handle lives in, so closing an index while another thread's blocking call still runs on it raises `IllegalStateException` rather than freeing the handle under that call; deferred calls are waited for by the close itself.
A call leases one scope only, so an executor, a cursor or a stream is closed once no other thread still calls through it.
An index, executor, arena or deferred call the program drops without closing is released once it is unreachable.

The filtered search of the C interface stays unbound.
Its predicate runs on the engine's own worker threads, which are not attached to the JVM, so every candidate would pay an attach, an upcall, and an exception check — and a predicate that threw would unwind through native frames that cannot handle it.
Filter the results of `search` instead, or narrow the index.
Progress handlers stay unbound for the same reason, which is what makes an insertion batch all-or-nothing here.

## Kotlin and Scala

The published surface is Java, and both other JVM languages consume it directly — there is no separate artifact for either.

Nullness is part of the contract.
The package is `@NullMarked`, so a Kotlin caller gets real non-null types across the whole boundary rather than platform types, and the compiler refuses a null where the API does not accept one:

```kotlin
val index: DenseIndex = DenseIndex.initRam(DenseOptions().dimensions(2).quantization(Scalar.F32), intArrayOf())
index.use { ... }                       // AutoCloseable is Kotlin's `use`
val matches = withContext(Dispatchers.IO) { index.deferred(pool).search(queries, 10).use { it.join() } }
```

Scala reads no nullness annotations, so what it gets instead is that the array-based batch verbs take `Array[Key]` and `Array[Float]` with no conversion, since both are JVM arrays already:

```scala
val keys: Array[Key] = Array(Key.of(1), Key.of(2))
index.on(pool).insert(keys, RowsView.of(Array(1.0f, 0.0f, 0.0f, 1.0f), 2))
```

## Design

The SDK is a thin layer over the C ABI in `include/usearch/`, and every rule below follows from that.

- __Names follow the header.__
  Every type is its C name without `usearch_` and `_t`, every constructor is a C lifecycle verb taking only what its C structure takes, and every wrapper's documentation names the C symbol it calls, or says it is binding-only.
- __Blocking calls stay as simple as the language allows.__
  `index.search(queries, 10)` takes no executor, completion or callback and returns its result; nothing about asynchrony reaches a caller who never asks for it.
- __The receiver chooses, the verb stays.__
  `DenseVerbs` writes every blocking verb once, for the index on the calling thread and for the `On` that `index.on(executor)` makes, and the `DeferredOn` that `index.deferred(executor)` makes offers the same verbs answering a `Deferred`.
  The receiver's type alone chooses the completion, never a flag or a string.
- __Deferred calls are symmetric through the language's own idioms.__
  A `Deferred` is `AutoCloseable` and joins as a `CompletableFuture` would, so Java waits in a try-with-resources block or a virtual thread, and Kotlin in `withContext(Dispatchers.IO)`.
- __Nothing costs anything until it is used.__
  A blocking call allocates its argument structure in a confined arena freed on return and takes no lock; a deferred one keeps that memory in a shared arena until the call is final.
- __A `Deferred` has no life of its own.__
  `status` is an acquire load of the outcome word, `join` blocks in `usearch_outcome_wait`, and closing or dropping an unfinished one waits the same way before freeing what it lent.
  The SDK starts no thread, counts no calls and keeps no queue: the C executor owns every thread, and every wait belongs to the C library.
- __Statuses and modes are enums.__
  `Status`, `Mood`, `DenseMetric` and `Scalar` carry the C enumerators' values, and their `toString` spells the names the C tables own, so nothing is compared as a string.
- __Arenas are reused, never shared.__
  A result owns a fresh arena by default, a passed one is reused, and no index holds an arena of its own.
