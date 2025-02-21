# USearch for Java

## Installation

```xml
<dependency>
  <groupId>cloud.unum</groupId>
  <artifactId>usearch</artifactId>
  <version>2.17.2</version>
</dependency>
```

Add that snippet to your `pom.xml` and hit `mvn install`.

## Quickstart

```java
import cloud.unum.usearch.Index;

public class Main {
    public static void main(String[] args) {
        try (Index index = new Index.Config()
                .metric("cos")
                .quantization("f32")
                .dimensions(3)
                .capacity(100)
                .build()) {
            
            // Add to Index
            float[] vector = {0.1f, 0.2f, 0.3f};
            index.add(42, vector);

            // Search
            int[] keys = index.search(new float[]{0.1f, 0.2f, 0.3f}, 10);
            for (int key : keys) {
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
float[] vector = index.get(42);
boolean removed = index.remove(42);
boolean renamed = index.rename(43, 42);
```

To obtain metadata:

```java
long size = index.size();
long capacity = index.capacity();
long dimensions = index.dimensions();
long connectivity = index.connectivity();
```
