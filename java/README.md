# USearch for Java

## Installation

Installation via Maven Central is not supported due to Continuous Delivery complexity and poor support for native builds.
Gradle installation is the recommended approach.
For the most up-to-date version, the following Groovy script will download a "fat JAR" containing builds for Linux, Windows, macOS, and Android, compatible with all common hardware platforms:

```groovy
repositories {
    mavenCentral()
    
    // Custom repository for USearch JAR
    flatDir {
        dirs 'lib'
    }
}

// Task to download USearch JAR from GitHub releases
task downloadUSearchJar {
    doLast {
        def usearchVersion = '2.24.0'
        def usearchUrl = "https://github.com/unum-cloud/usearch/releases/download/v${usearchVersion}/usearch-${usearchVersion}.jar"
        def usearchFile = file("lib/usearch-${usearchVersion}.jar")
        
        usearchFile.parentFile.mkdirs()
        if (!usearchFile.exists()) {
            new URL(usearchUrl).withInputStream { i ->
                usearchFile.withOutputStream { it << i }
            }
            println "Downloaded USearch JAR: ${usearchFile.name}"
        }
    }
}

// Make compilation depend on downloading USearch
compileJava.dependsOn downloadUSearchJar

dependencies {
    // USearch JAR from local lib directory (downloaded automatically)
    implementation name: 'usearch', version: '2.24.0', ext: 'jar'
}
```

## Quickstart

```java
import cloud.unum.usearch.Index;

public class Main {
    public static void main(String[] args) {
        try (Index index = new Index.Config()
                .metric(Index.Metric.COSINE)              // Or "cos"
                .quantization(Index.Quantization.FLOAT32) // Or "f32"
                .dimensions(3)
                .capacity(100)
                .build()) {
            
            // Add to Index
            float[] vector = {0.1f, 0.2f, 0.3f};
            index.add(42L, vector);

            // Search
            long[] keys = index.search(new float[]{0.1f, 0.2f, 0.3f}, 10);
            for (long key : keys) {
                System.out.println("Found key: " + key);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## Serialization

To save and load the index from disk, use the following methods:

```java
index.save("index.usearch");
index.load("index.usearch");
index.view("index.usearch");
```

## Extracting, Updating, and Removing Values

It is generally not recommended to use HNSW indexes in case of frequent removals or major distribution shifts.
For small updates, you can use the following methods:

```java
float[] vector = index.get(42L);
boolean removed = index.remove(42L);
boolean renamed = index.rename(43L, 42L);
```

To obtain metadata:

```java
long size = index.size();
long capacity = index.capacity();
long dimensions = index.dimensions();
long connectivity = index.connectivity();
```

## Multiple Data Types and Quantization

USearch supports hardware-agnostic `f64`, `f32`, and `i8` quantization for memory efficiency and performance optimization.

```java
// Double precision (f64) for highest accuracy
try (Index doubleIndex = new Index.Config()
        .metric("cos")
        .dimensions(3)
        .quantization("f64")
        .build()) {
    
    double[] vector = {0.1, 0.2, 0.3};
    doubleIndex.add(42L, vector);
    
    double[] buffer = new double[3];
    doubleIndex.getInto(42L, buffer); // Memory-efficient retrieval
}

// Byte precision (i8) for memory efficiency  
try (Index byteIndex = new Index.Config()
        .metric("cos")
        .dimensions(3)
        .quantization("i8")
        .build()) {
    
    byte[] vector = {10, 20, 30};
    byteIndex.add(42L, vector);
    
    byte[] buffer = new byte[3];
    byteIndex.getInto(42L, buffer); // Memory-efficient retrieval
}
```

## Batch Operations

USearch automatically detects batch operations when vector arrays contain multiple concatenated vectors:

```java
try (Index index = new Index.Config()
        .metric("cos")
        .dimensions(2)
        .build()) {
    
    // Batch add: 3 vectors in one call
    float[] batchVectors = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    index.add(100L, batchVectors); // Adds vectors at keys 100, 101, 102
    
    // Verify batch was added correctly
    System.out.println("Index size: " + index.size()); // Output: 3
}
```

## Concurrent Operations

The USearch index is thread-safe and supports high-performance concurrent operations:

```java
import java.util.concurrent.*;

try (Index index = new Index.Config()
        .metric("cos")
        .dimensions(4)
        .capacity(10000)
        .build()) {
    
    ExecutorService executor = Executors.newFixedThreadPool(8);
    
    // Concurrent additions from multiple threads
    CompletableFuture<Void>[] addTasks = new CompletableFuture[4];
    for (int t = 0; t < 4; t++) {
        final int threadId = t;
        addTasks[t] = CompletableFuture.runAsync(() -> {
            for (int i = 0; i < 1000; i++) {
                long key = threadId * 1000L + i;
                float[] vector = generateRandomVector(4);
                index.add(key, vector);
            }
        }, executor);
    }
    
    // Concurrent searches while adding
    CompletableFuture<Void>[] searchTasks = new CompletableFuture[4];
    for (int t = 0; t < 4; t++) {
        searchTasks[t] = CompletableFuture.runAsync(() -> {
            for (int i = 0; i < 100; i++) {
                float[] query = generateRandomVector(4);
                long[] results = index.search(query, 10);
                processResults(results);
            }
        }, executor);
    }
    
    // Wait for all operations to complete
    CompletableFuture.allOf(addTasks).join();
    CompletableFuture.allOf(searchTasks).join();
    executor.shutdown();
    
    System.out.println("Final index size: " + index.size());
}

private static float[] generateRandomVector(int dimensions) {
    float[] vector = new float[dimensions];
    for (int i = 0; i < dimensions; i++) {
        vector[i] = (float) Math.random();
    }
    return vector;
}
```
