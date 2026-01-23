# USearch for Rust

Detailed symbol list and documentation for USearch Rust SDK can be found on [docs.rs](https://docs.rs/usearch/latest/usearch/struct.Index.html).

## Installation

```sh
cargo add usearch
```

This will add a USearch dependency to your `Cargo.toml` file.

```toml
[dependencies]
usearch = "..."
```

By default, [SimSIMD](https://github.com/ashvardanian/simsimd) is used to provide dynamic dispatch for SIMD operations.
You can, however, override that by specifying custom features in your `Cargo.toml` file.
To disable all features, use the following configuration:

```toml
[dependencies]
usearch = { version = "...", default-features = false }
```

To enable specific features, use the following configuration:

```toml
[dependencies]
usearch = { version = "...", features = ["simsimd", "openmp", "fp16lib"] }
```

OpenMP (`openmp`) will use the OpenMP runtime for parallelism.
It may not be available on all platforms, but on Linux it will lead to better performance and lower latency of small-batch operations on multi-core CPUs.
The `fp16lib` flag will bring in the C-layer `fp16` library to emulate half-precision floating point operations on older CPUs, where it may not be natively supported.

## Quickstart

```rust
use usearch::{Index, IndexOptions, MetricKind, ScalarKind, new_index};

let options = IndexOptions {
    dimensions: 3, // necessary for most metric kinds
    metric: MetricKind::IP, // or ::L2sq, ::Cos ...
    quantization: ScalarKind::BF16, // or ::F32, ::F16, ::I8, ::B1x8 ...
    connectivity: 0, // zero for auto
    expansion_add: 0, // zero for auto
    expansion_search: 0, // zero for auto
};

let index: Index = new_index(&options).unwrap();

assert!(index.reserve(10).is_ok());
assert!(index.capacity() >= 10);
assert!(index.connectivity() != 0);
assert_eq!(index.dimensions(), 3);
assert_eq!(index.size(), 0);

let first: [f32; 3] = [0.2, 0.1, 0.2];
let second: [f32; 3] = [0.2, 0.1, 0.2];

assert!(index.add(42, &first).is_ok());
assert!(index.add(43, &second).is_ok());
assert_eq!(index.size(), 2);

// Read back the tags
let results = index.search(&first, 10).unwrap();
assert_eq!(results.keys.len(), 2);
```

## Serialization

To save and load the index from disk, use the following methods:

```rust
assert!(index.save("index.usearch").is_ok());
assert!(index.load("index.usearch").is_ok());
assert!(index.view("index.usearch").is_ok());
```

Viewing the index does not load the data into memory, but allows you to inspect and traverse the index structure from external memory using memory-mapping.
Similarly, serializing to/from in-memory buffers is supported.
So you can memory-map the index file manually, and later call `view_from_buffer` or one of its siblings.

```rust
assert!(index.save_to_buffer(&mut serialization_buffer).is_ok());
assert!(index.load_from_buffer(&serialization_buffer).is_ok());
assert!(index.view_from_buffer(&serialization_buffer).is_ok());
```

## Metrics

USearch comes pre-packaged with SimSIMD, bringing over 100 SIMD-accelerated distance kernels for x86 and ARM architectures.
That includes:

- `MetricKind::IP` - Inner Product metric, defined as `IP = 1 - sum(a[i] * b[i])`.
- `MetricKind::L2sq` - Squared Euclidean Distance metric, defined as `L2 = sum((a[i] - b[i])^2)`.
- `MetricKind::Cos` - Cosine Similarity metric, defined as `Cos = 1 - sum(a[i] * b[i]) / (sqrt(sum(a[i]^2) * sqrt(sum(b[i]^2)))`.
- `MetricKind::Pearson` - Pearson Correlation metric.
- `MetricKind::Haversine` - Haversine (Great Circle) Distance metric.
- `MetricKind::Divergence` - Jensen Shannon Divergence metric.
- `MetricKind::Hamming` - Bit-level Hamming Distance metric, defined as the number of differing bits.
- `MetricKind::Tanimoto` - Bit-level Tanimoto (Jaccard) metric, defined as the number of intersecting bits divided by the number of union bits.
- `MetricKind::Sorensen` - Bit-level Sorensen metric.

### User-Defined Metrics

Custom metrics allow for the implementation of specific algorithms to measure the distance or similarity between vectors in the index.
To use a custom metric with USearch, define a function that matches the expected signature and pass it to your index on creation, or later with `change_metric`.
Let's say you are implementing a weighted distance function to search through joint embeddings of images and textual descriptions of some products in a catalog, taking some [UForm](https://github.com/unum-cloud/uform) or CLIP-like models.

```rust
use simsimd::SpatialSimilarity;

let image_dimensions: usize = 768;
let text_dimensions: usize = 512;
let image_weights: f32 = 0.7;
let text_weights: f32 = 0.9;

let weighted_distance = Box::new(move |a: *const f32, b: *const f32| unsafe {
    let a_slice = std::slice::from_raw_parts(a, image_dimensions + text_dimensions);
    let b_slice = std::slice::from_raw_parts(b, image_dimensions + text_dimensions);

    let image_similarity = f32::cosine(a_slice[0..image_dimensions], b_slice[0..image_dimensions]);
    let text_similarity = f32::cosine(a_slice[image_dimensions..], b_slice[image_dimensions..]);
    let similarity = image_weights * image_similarity + text_weights * text_similarity / (image_weights + text_weights);
    
    1.0 - similarity
});
index.change_metric(weighted_distance);
```

You can always revert back to one of the native metrics by calling:
    
```rust
index.change_metric_kind(MetricKind::Cos);
```

## Filtering with Predicates

Sometimes you may want to cross-reference search-results against some external database or filter them based on some criteria.
In most engines, you'd have to manually perform paging requests, successively filtering the results.
In USearch you can simply pass a predicate function to the search method, which will be applied directly during graph traversal.

```rust
let is_odd = |key: Key| key % 2 == 1;
let query = vec![0.2, 0.1, 0.2, 0.1, 0.3];
let results = index.filtered_search(&query, 10, is_odd).unwrap();
assert!(
    results.keys.iter().all(|&key| key % 2 == 1),
    "All keys must be odd"
);
```

## Quantization and Custom Scalar Types

USearch supports the Rust-native `f32` and `f64` scalar types, as well as the `i8` for quantized 8-bit scalars.
Going beyond that, USearch supports non-native `f16` and `b1x8` for half-precision floating point and binary vectors, respectively.

### Half Precision Floating Point

Rust has no native support for half-precision floating-point numbers, but USearch provides a `f16` type.
It has no advanced functionality - it is a transparent wrapper around `i16` and can be used with `half` or any other half-precision library.
Assuming USearch uses the IEEE 754 no conversion is needed, you can `unsafe`-cast the outputs of other IEEE-compliant libraries to `usearch::f16`.

```rust
use usearch::f16 as USearchF16;
use half::f16 as HalfF16;

let vector_a: Vec<HalfF16> = ...
let vector_b: Vec<HalfF16> = ...

let buffer_a: &[USearchF16] = unsafe { std::slice::from_raw_parts(a_half.as_ptr() as *const USearchF16, a_half.len()) };
let buffer_b: &[USearchF16] = unsafe { std::slice::from_raw_parts(b_half.as_ptr() as *const USearchF16, b_half.len()) };

index.add(42, buffer_a);
index.add(43, buffer_b);
```

### Binary Vectors

USearch also implements binary distance functions and natively supports bit-vectors.
If you initialize the index with `quantization: ScalarKind::B1x8`, you can add floating-point vectors and they will be quantized mapping positive values to `1` and negative and zero values to `0`.
Alternatively, you can use the `b1x8` type to represent packed binary vectors directly.

```rs
let index = Index::new(&IndexOptions {
    dimensions: 8,
    metric: MetricKind::Hamming,
    quantization: ScalarKind::B1x8,
    ..Default::default()
})
.unwrap();

// Binary vectors represented as `b1x8` slices
let vector42: Vec<b1x8> = vec![b1x8(0b00001111)];
let vector43: Vec<b1x8> = vec![b1x8(0b11110000)];
let query: Vec<b1x8> = vec![b1x8(0b01111000)];

// Adding binary vectors to the index
index.reserve(10).unwrap();
index.add(42, &vector42).unwrap();
index.add(43, &vector43).unwrap();

let results = index.search(&query, 5).unwrap();

// Validate the search results based on Hamming distance
assert_eq!(results.keys.len(), 2);
assert_eq!(results.keys[0], 43);
assert_eq!(results.distances[0], 2.0); // 2 bits differ between query and vector43
assert_eq!(results.keys[1], 42);
assert_eq!(results.distances[1], 6.0); // 6 bits differ between query and vector42
```

## Performance Tuning

To optimize the performance of the index, you can adjust the expansion values used during index creation and search operations.
Higher expansion values will lead to better search accuracy at the cost of slightly increased memory usage, but potentially much higher search times.
Following methods are available to adjust the expansion values:

```rs
println!("Add expansion: {}", index.expansion_add());
println!("Search expansion: {}", index.expansion_search());
index.change_expansion_add(32);
index.change_expansion_search(32);
```

Optimizing hardware utilization, you may want to check the SIMD hardware acceleration capabilities of the index and memory consumption.
The first will print the codename of the most advanced SIMD instruction set supported by the CPU and used by the index.
The second will print the memory usage of the index in bytes.

```rs
println!("Hardware acceleration: {}", index.hardware_acceleration());
println!("Memory usage: {}", index.memory_usage());
```
