# USearch for Java

The Java SDK is a thin JNI layer over the flat C interface in `include/usearch/types.h` and `include/usearch/dense_index.h`.
It instantiates no index template: `java/lib.c` marshals arguments and links the pre-instantiated library, so one compiled engine serves every language binding.

One source file mirrors one header.
`Types.java` carries the shared vocabulary — `Key`, `Status`, `ScalarKind`, `Residency`, `Executor`, `USearchException`, `Matrix`, `Matches`.
`DenseIndex.java` carries the dense modality — the index itself, `DenseOptions`, `DenseMetric`, `InsertionOptions`, `SearchOptions`, `DenseStats`, `Refinement`.
`Library.java` loads the shared object.
Both vocabularies are nested types, and a nested type imports directly, so every call site below spells the bare name:

```java
import cloud.unum.usearch.DenseIndex;
import cloud.unum.usearch.DenseIndex.DenseMetric;
import cloud.unum.usearch.DenseIndex.DenseOptions;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.Matrix;
import cloud.unum.usearch.Types.ScalarKind;
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
Every fanned-out verb takes an executor, which owns the threads the work runs on.

```java
public class Main {
    public static void main(String[] arguments) {
        try (Executor pool = new Executor();
                DenseIndex index = new DenseOptions()
                        .dimensions(3)
                        .metric(DenseMetric.ANGULAR)
                        .quantization(ScalarKind.F32)
                        .open()) {

            index.reserve(100);
            index.add(
                new Key[] {Key.of(42), Key.of(43)},
                Matrix.of(new float[] {0.1f, 0.2f, 0.3f, 0.3f, 0.2f, 0.1f}, 3),
                pool);

            Matches matches = index.search(Matrix.of(new float[] {0.1f, 0.2f, 0.3f}, 3), 10, pool);
            for (int rank = 0; rank < matches.count(0); ++rank) {
                System.out.println(matches.key(0, rank) + " at " + matches.score(0, rank));
            }
        }
    }
}
```

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

`Matrix` borrows a caller-owned slab: a primitive array, or a direct `ByteBuffer` for a memory-mapped dataset.
The scalar type of the input is independent of the stored quantization, and the library converts on ingest.

```java
Matrix packed = Matrix.of(new float[] {0, 1, 1, 0}, 2);
Matrix padded = Matrix.strided(slab, rows, 768, 4096, ScalarKind.I8);
Matrix mapped = Matrix.strided(mappedBuffer, rows, 768, 768, ScalarKind.U8);
```

A primitive array is copied across the boundary for the duration of one call, since holding the JVM's own storage in place would stall every collector thread for as long as the batch runs.
A direct `ByteBuffer` is addressed in place with no copy at all, which is the argument to reach for when a batch is large.

Quantization kinds run from the widest dynamic range down: `F64`, `F32`, `BF16`, `F16`, `E5M2`, `E4M3`, `E3M2`, `E2M3`, `I8`, `I4X2`, `U8`, `U4X2`, `U1X8`.
Sub-byte kinds count logical dimensions, so sixteen `U1X8` dimensions occupy two bytes per row.

## Residency

Where the payload lives and which engine traverses it is one tagged choice, and it also decides what loading a checkpoint means.
Placement within a kind — the memory domains, the device ordinals, the device paths — derives from the machine.

```java
new DenseOptions().residency(Residency.RAM);
new DenseOptions().residency(Residency.VRAM);
new DenseOptions().residency(Residency.NVME);
```

An NVMe handle opens its checkpoint in place, so every mutating verb on it fails with `Status.READ_ONLY`.

## Executors

An executor owns the host threads or device streams a fanned-out verb runs on; an index owns rows.
Keeping them apart is what lets one pool serve several indexes, and what makes the cost of spawning visible at the line that spawns.
Its residency must match the residency of every handle it drives, and no verb derives one for itself.

```java
try (Executor pool = new Executor(Residency.RAM, 16)) {
    index.add(keys, vectors, pool);
    Matches matches = index.search(queries, 10, pool);
    index.save("index.safegraphs", pool);
    pool.drain();
}
```

`add`, `refine`, `search`, `searchInto`, `maxsim`, `maxsimInto`, `save`, `load`, `saveBuffer` and `loadBuffer` each take one.
An executor admits one driver at a time, reporting `Status.CONTENDED` rather than corrupting a dispatch already in flight; handles themselves stay freely concurrent.

## Per-Call Settings and Telemetry

The candidate-pool breadth rides each call rather than the handle, which lets a recall sweep run against one immutable index with no mutation between points.
Every verb that walks the graph returns the telemetry the walk reported, and `DenseStats.plus` sums a sweep of them.

```java
DenseStats inserted = index.add(keys, vectors, pool, new InsertionOptions(256));
DenseStats searched = index.searchInto(queries, matches, pool, new SearchOptions(64));
DenseStats total = inserted.plus(searched);
```

## Refinement

Refinement re-selects the neighbors of every stored entry against the finished graph, improving edges that were chosen when the graph around them was still incomplete.
It is a build step rather than a repair: a cheap insertion pass followed by one refinement reaches a better graph, in less total time, than a single expensive insertion pass, so a wide expansion factor belongs here rather than in `add`.
The sweep resumes where a previous one stopped, so a caller loops until the changed-node count falls below whatever rate it considers settled.

```java
Refinement pass = index.refine(pool, new InsertionOptions(256));
while (pass.changedNodes() > index.size() / 100) {
    pass = index.refine(pool, new InsertionOptions(256));
}
```

## Late Interaction

`maxsim` scores multi-row queries against multi-row entries: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, so `valuesPerKey` bounds how many rows share a key and the wanted count is the "k" over documents rather than over rows.
Every query carries the same number of rows, so a padded token batch applies as is.

```java
try (DenseIndex index = new DenseOptions()
        .dimensions(128)
        .metric(DenseMetric.INNER)
        .valuesPerKey(512)
        .open()) {
    Matches documents = index.maxsim(tokens, 32, 10, pool);
}
```

It runs on the inner-product kinds alone, since a maximum over logits is what late interaction sums; a graph frozen under a distance kind reports `Status.CONFIG_MISMATCH`.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout: a `.safegraphs` path is one monolith, a `.safegraphs.json` path is a manifest cataloging many.
A handle built with zero dimensions freezes its geometry at the first load, from the checkpoint's own headers.

```java
index.save("index.safegraphs", pool);
index.load("index.safegraphs", pool);

DenseOptions described = DenseIndex.metadata("index.safegraphs");

byte[] slab = new byte[(int) index.storageRequirements()];
index.saveBuffer(slab, pool);
index.loadBuffer(slab, pool);
```

## Reading Back

```java
long entries = index.count(Key.of(42));
boolean present = index.contains(Key.of(42));

Matrix retrieved = Matrix.allocate(entries, 3, ScalarKind.F32);
index.get(Key.of(42), retrieved);
```

`get` and `search` report the __true__ extent, never the number written, so a count above the room reserved for it means the copy was truncated.
`Matches.truncated` asks that question directly, and no accessor will hand back a slot the engine never wrote.

## Concurrency

A handle may be shared across threads.
On a RAM handle, insertions and searches overlap freely: searches read only entries published behind the visibility watermark and never write to the graph, while insertions synchronize through per-node locks and publish in order.
Serialization is the one exception, demanding a mutation-free window, though concurrent readers keep searching throughout.

The filtered search of the C interface stays unbound.
Its predicate runs on the engine's own worker threads, which are not attached to the JVM, so every candidate would pay an attach, an upcall, and an exception check — and a predicate that threw would unwind through native frames that cannot handle it.
Filter the results of `search` instead, or narrow the index.
Progress handlers stay unbound for the same reason, which is what makes an insertion batch all-or-nothing here.

## Building From Source

The JNI layer links `libusearch_c`, which CMake builds from the instantiation grid, and Gradle drives that CMake tree rather than describing a second native build of its own:

```sh
gradle test
```

`build_artifacts` is where Gradle configures and builds by default; `-PusearchLibraryPath=<directory>` points it elsewhere.

CMake builds the JNI library directly too, alongside every other target:

```sh
cmake -B build_artifacts -DCMAKE_BUILD_TYPE=Release -DUSEARCH_BUILD_LIB_C=1 -DUSEARCH_BUILD_LIB_JAVA=1
cmake --build build_artifacts --target usearch_jni
```

The brute-force stand-in defines the same symbols a different way, so the suite runs in seconds against the ABI instead of waiting on a build of the engine:

```sh
cmake -B build_stub -DUSEARCH_BUILD_LIB_C_STUB=1 -DUSEARCH_BUILD_LIB_JAVA=1
cmake --build build_stub --target usearch_jni
javac -d build_stub/classes -cp junit-platform-console-standalone.jar \
    java/cloud/unum/usearch/*.java java/test/cloud/unum/usearch/*.java
LD_LIBRARY_PATH=build_stub/stub java -Djava.library.path=build_stub \
    -jar junit-platform-console-standalone.jar execute \
    --class-path build_stub/classes --select-class cloud.unum.usearch.DenseIndexTest
```
