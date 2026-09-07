# USearch for Rust

Detailed symbol list and documentation for the USearch Rust SDK can be found on [docs.rs](https://docs.rs/usearch).

## Installation

```sh
cargo add usearch
```

The crate links the pre-instantiated C library and consumes it through its stable C headers, so no index template is instantiated in your build.
[NumKong](https://github.com/ashvardanian/NumKong) is a hard dependency rather than a feature: its matrices, scalar types, and traits _are_ this crate's vocabulary, and it supplies the SIMD kernels.

The crate compiles no C or C++ of its own: CMake builds the engine, and `build.rs` only links it.
Point `USEARCH_LIB_DIR` at the directory holding `libusearch_indexes`; it defaults to `build_release/src`, where an in-tree CMake build leaves it.

## Quickstart

Every batch input and output is a NumKong matrix - a base pointer, a row count, and a byte stride - so a memory-mapped dataset feeds the index with no reshape and no copy.
An index owns rows; an `Executor` owns the threads or streams a fanned-out verb runs on, and every batch verb takes one.

```rust
use usearch::dense::{Metric, Options, Outputs, Stats};
use usearch::{rows, DenseIndex, Distance, Executor, Key, Residency, ScalarKind};

let index = DenseIndex::new(&Options {
    dimensions: 4,
    metric: Metric::Angular,
    quantization: ScalarKind::I8,
    ..Options::default()
})?;
index.reserve(1_000)?;

let executor = Executor::new(&Residency::default(), 0)?;

let keys = [Key::from(42u64), Key::from(43u64)];
let slab: [f32; 8] = [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2];
index.add(&executor, &keys, rows(&slab, 4), 0, None)?;

let query: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
let (mut found, mut scores, mut counts) = ([Key::NIL; 10], [0.0 as Distance; 10], [0usize; 1]);
let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 10);
index.search_into(&executor, rows(&query, 4), 0, &mut outputs, None)?;
```

One executor serves any number of indexes, and its residency must match the one their handles were built for.
It admits a single driver at a time, so two threads that genuinely overlap take one apiece; index handles stay freely shared.

`rows` borrows a packed slab; a dataset whose rows are padded to a wider pitch passes `MatrixView::from_raw_parts` instead, which takes the byte stride directly.
The scalar type of the input is independent of the stored quantization - the `f32` slab above lands in an `i8` index - so nothing needs converting before insertion.

## Search Allocates Nothing

There is one search verb and it writes into buffers you own, so a benchmark loop allocates once and never again.
The width of the keys matrix _is_ the "k" in "kANN", so the wanted count cannot disagree with the space reserved for it.

```rust
let mut keys = vec![Key::NIL; queries_count * 10];
let mut distances = vec![0.0; queries_count * 10];
let mut counts = vec![0usize; queries_count];
let mut outputs = Outputs::packed(&mut keys, &mut distances, &mut counts, 10);
let mut stats = Stats::default();

index.search_into(&executor, rows(&queries, 768), 128, &mut outputs, Some(&mut stats))?;
println!("{} distances computed", stats.computed_distances);
```

The candidate-pool breadth rides the call rather than the handle, so a recall sweep is a loop over `ef` values against one immutable index, with no mutation and no synchronization between points.

## Residency

Where the payload lives is an enum, not a flag, and it decides what loading a checkpoint means: materialize into RAM, upload to a device, or open in place on disk.

A residency also names the hardware its shards are placed across - one shard per memory domain, device, or path - and an empty list derives one from the machine.

```rust
use usearch::Residency;

Residency::Ram { memory_domains: vec![0, 1] };          // one graph per socket
Residency::Vram { devices: vec![0, 1, 2, 3] };          // sharded, never replicated
Residency::Nvme { paths: vec!["/mnt/nvme0".into()] };   // striped across the drives named
```

The same value constructs the executor that drives the handle, which is what keeps compute and memory domains in agreement.

An NVMe handle opens a checkpoint in place and is read-only, so every mutating verb fails on it.
Bigger indexes come from merging, never mutating: shards build in RAM, serialize, and join a manifest.

## Serialization

The path extension picks the layout - a `.safegraphs` path is one monolith, a `.safegraphs.json` path is a manifest cataloging many.

```rust
index.save(&executor, "index.safegraphs")?;
index.load(&executor, "index.safegraphs")?;

let options = DenseIndex::metadata("index.safegraphs")?; // geometry without loading a shard
```

Checkpoints are residency-portable: one file materializes into RAM, uploads to a device, or opens in place on disk, decided by the handle that loads it.
In-memory buffers hold single monoliths only, through `save_buffer` and `load_buffer`.

## Metrics

Every metric derives from a single inner-product kernel with pre-computed magnitudes, which is why this binding does not expose the C layer's custom-callback escape hatch: a caller-supplied metric forfeits the SIMD kernel chains and the magnitude cache both.

- `Metric::Inner` - inner product.
- `Metric::Angular` - cosine distance, over pre-computed magnitudes.
- `Metric::Euclidean` - squared euclidean distance.
- `Metric::Hamming` - bit-level, over the packed sub-byte kinds.
- `Metric::Jaccard` - bit-level, over the packed sub-byte kinds.

## Filtering With Predicates

Cross-referencing an external database during traversal beats paging through results and filtering afterwards.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.

```rust
let is_odd = |key: Key| key.as_u64().is_some_and(|value| value % 2 == 1);
index.search_if_into(&executor, rows(&query, 768), 0, &mut outputs, None, &is_odd)?;
```

## Scalar Types

The scalar types are NumKong's, so nothing needs re-wrapping to move between the two crates - `f16` and `bf16` included, emulated where the CPU has no native support.

| Rust type                            | `ScalarKind`                    |
| ------------------------------------ | ------------------------------- |
| `f64`, `f32`                         | `F64`, `F32`                    |
| `bf16`, `f16`                        | `BF16`, `F16`                   |
| `e5m2`, `e4m3`, `e3m2`, `e2m3`       | `E5M2`, `E4M3`, `E3M2`, `E2M3`  |
| `i8`, `i4x2`                         | `I8`, `I4x2`                    |
| `u8`, `u4x2`, `u1x8`                 | `U8`, `U4x2`, `U1x8`            |

Initializing an index with a packed quantization and adding floating-point vectors quantizes on ingest; the `u1x8` type also lets you pass packed bit-vectors directly.

```rust
let index = DenseIndex::new(&Options {
    dimensions: 8,
    metric: Metric::Hamming,
    quantization: ScalarKind::U1x8,
    ..Options::default()
})?;

let slab = [u1x8(0b0000_1111), u1x8(0b1111_0000)];
index.add(&executor, &keys, rows(&slab, 8), 0, None)?;
```

## Streaming and Iteration

A finder is a resumable best-first traversal, so callers that post-filter aggressively pull until satisfied instead of guessing "k" upfront.

```rust
for found in index.find(&executor, &query)? {
    let found = found?;
    if found.distance > 0.3 { break; }
    println!("{} at {}", found.key, found.distance);
}
```

A cursor walks the index contents as a sequential row scan behind a publication watermark, so concurrent inserts never appear mid-iteration.

```rust
let mut cursor = index.cursor::<f32>()?;
let mut keys = vec![Key::NIL; 1024];
let mut vectors = vec![0.0f32; 1024 * 768];
loop {
    let written = cursor.next_batch(&mut keys, Some(&mut usearch::rows_mut(&mut vectors, 768)))?;
    if written == 0 { break; }
}
```

## Introspection

```rust
println!("{} entries in {} bytes", index.size()?, index.memory_usage()?);
println!("dispatched to {}", index.capabilities()?);
println!("compiled for {}", usearch::capabilities_compiled());
println!("available here {}", usearch::capabilities_available());
```
