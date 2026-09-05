# USearch for Java

The Java SDK is a thin JNI layer over the flat C interface in `include/usearch/dense_index.h`.
It instantiates no index template: `java/lib.cpp` marshals arguments and links the pre-instantiated library, so one compiled engine serves every language binding.

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

```java
import cloud.unum.usearch.*;

public class Main {
    public static void main(String[] arguments) {
        try (DenseIndex index = new DenseOptions()
                .dimensions(3)
                .metric(DenseMetric.ANGULAR)
                .quantization(ScalarKind.F32)
                .open()) {

            index.reserve(100);
            index.add(
                new Key[] {Key.of(42), Key.of(43)},
                DenseMatrix.of(new float[] {0.1f, 0.2f, 0.3f, 0.3f, 0.2f, 0.1f}, 3));

            DenseMatches matches = index.search(DenseMatrix.of(new float[] {0.1f, 0.2f, 0.3f}, 3), 10);
            for (int rank = 0; rank < matches.count(0); ++rank) {
                System.out.println(matches.key(0, rank) + " at " + matches.distance(0, rank));
            }
        }
    }
}
```

## Keys

A key is sixteen opaque bytes rendered as a lowercase dashed UUID, and integer identities park in the low eight bytes.
`Key.of(42)` therefore prints as `00000000-0000-0000-0000-00000000002a`, byte for byte the same rendering every other USearch SDK produces.
Keys need not be unique: several entries may share one, and `count` reports how many.

```java
Key key = Key.of(42);
Key parsed = Key.parse("00000000-0000-0000-0000-00000000002a");
long identity = key.toLong();
```

## Matrices and Quantization

`DenseMatrix` borrows a caller-owned slab: a primitive array, or a direct `ByteBuffer` for a memory-mapped dataset.
The scalar type of the input is independent of the stored quantization, and the library converts on ingest.

```java
DenseMatrix packed = DenseMatrix.of(new float[] {0, 1, 1, 0}, 2);
DenseMatrix padded = DenseMatrix.strided(slab, rows, 768, 4096, ScalarKind.I8);
DenseMatrix mapped = DenseMatrix.strided(mappedBuffer, rows, 768, 768, ScalarKind.U8);
```

Quantization kinds run from the widest dynamic range down: `F64`, `F32`, `BF16`, `F16`, `E5M2`, `E4M3`, `E3M2`, `E2M3`, `I8`, `I4X2`, `U8`, `U4X2`, `U1X8`.
Sub-byte kinds count logical dimensions, so sixteen `U1X8` dimensions occupy two bytes per row.

## Residency

Where the payload lives and which engine traverses it is one tagged choice, and it also decides what loading a checkpoint means.

```java
new DenseOptions().residency(Residency.ram(16, Residency.HugePages.REQUIRED));
new DenseOptions().residency(Residency.vram(0, Residency.Addressing.UNIFIED));
new DenseOptions().residency(Residency.nvme(512, 128, Residency.IoMode.DIRECT));
```

Worker threads are a property of the RAM residency and freeze at initialization.
An NVMe handle opens its checkpoint in place, so every mutating verb on it fails with `Status.READ_ONLY`.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout: a `.safegraphs` path is one monolith, a `.safegraphs.json` path is a manifest cataloging many.
A handle built with zero dimensions freezes its geometry at the first load, from the checkpoint's own headers.

```java
index.save("index.safegraphs");
index.load("index.safegraphs");

DenseOptions described = DenseIndex.metadata("index.safegraphs");

byte[] slab = new byte[(int) index.serializedLength()];
index.saveBuffer(slab);
index.loadBuffer(slab);
```

## Reading Back and Merging

```java
long entries = index.count(Key.of(42));
boolean present = index.contains(Key.of(42));

DenseMatrix retrieved = DenseMatrix.allocate(entries, 3, ScalarKind.F32);
index.get(Key.of(42), retrieved);

index.merge(donor);
```

`get` and `search` report the __true__ extent, never the number written, so a count above the room reserved for it means the copy was truncated.
`DenseMatches.truncated` asks that question directly, and no accessor will hand back a slot the engine never wrote.

`merge` consumes the donor: the native side owns it after the call, so the donor handle empties and its own `close` becomes a no-op.
The donor demands exclusive ownership for the whole call, and a concurrent verb on it from another thread is a use-after-free this binding cannot detect.

## Concurrency

A handle may be shared across threads.
On a RAM handle, insertions and searches overlap freely: searches read only entries published behind the visibility watermark and never write to the graph, while insertions synchronize through per-node locks and publish in order.
Serialization is the one exception, demanding a mutation-free window, though concurrent readers keep searching throughout.

The filtered search of the C interface stays unbound.
Its predicate runs on the engine's own worker threads, which are not attached to the JVM, so every candidate would pay an attach, an upcall, and an exception check — and a predicate that threw would unwind through native frames that cannot handle it.
Filter the results of `search` instead, or narrow the index.

## Building From Source

The JNI layer links `libusearch_c`, which CMake builds from the instantiation grid.
Point Gradle at the directory holding it:

```sh
cmake -B build_release -DCMAKE_BUILD_TYPE=Release -DUSEARCH_BUILD_LIB_C=1
cmake --build build_release --target usearch_c
gradle -PusearchLibraryPath=$PWD/build_release test
```

CMake can also build the JNI library itself, alongside every other target:

```sh
cmake -B build_release -DCMAKE_BUILD_TYPE=Release -DUSEARCH_BUILD_LIB_JAVA=1
cmake --build build_release --target usearch_jni
```
