# USearch for Rust

Detailed symbol list and documentation for the USearch Rust SDK can be found on [docs.rs](https://docs.rs/usearch).

## Installation

```sh
cargo add usearch
```

CMake builds the engine and the crate links it through its stable C headers, so no index template is instantiated in your build.
[NumKong](https://github.com/ashvardanian/NumKong) is a hard dependency rather than a feature: its matrices, scalar types, and traits _are_ this crate's vocabulary, and it supplies the SIMD kernels.

## Quickstart

Every batch input is a NumKong matrix - a base pointer, a row count, and a byte stride - so a memory-mapped dataset feeds the index with no reshape and no copy.
An index owns rows; an `Executor` owns the threads or streams a verb runs on, and every verb that touches index data takes one.

```rust
use usearch::dense::{Metric, Options};
use usearch::{rows, DenseIndex, Executor, Key, ScalarKind};

let mut options = Options::default();
options.dimensions = 4;
options.metric = Metric::Angular;
options.quantization = ScalarKind::I8;
let index = DenseIndex::new(&options)?;
let executor = Executor::for_ram(&[], 0, 0)?;
index.reserve(&executor, 1_000)?;

let keys = [Key::from(42u64), Key::from(43u64)];
let slab: [f32; 8] = [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2];
index.add(&executor, &keys, &rows(&slab, 4)?).run()?;

let query: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
let matches = index.search(&executor, &rows(&query, 4)?, 10).run()?;
```

One executor serves any number of indexes, and its residency must match the one their handles were built for.
There is a constructor per residency, each taking only what its own hardware has: `for_ram` and `for_nvme` take a thread count where zero derives one, and `for_vram` takes none, since one host thread drives one stream per device, but may borrow your CUDA streams.
Each ends with an `in_flight_limit` sizing its table of deferred calls, where zero takes the default.
`Executor::default()` owns nothing and runs every call handed to it on the calling thread.
Calls on one executor run in submission order, so two threads that genuinely overlap take one apiece; index handles stay freely shared.

`rows` borrows a packed slab; a dataset whose rows are padded to a wider pitch passes `MatrixView::from_raw_parts` instead, which takes the byte stride directly.
The scalar type of the input is independent of the stored quantization - the `f32` slab above lands in an `i8` index - so nothing needs converting before insertion.

## Builders and Outputs

Every verb with optional arguments returns a builder: set what matters by name, then `run` blocks until the call is done.
A search's `run` returns `Matches` in buffers it allocates; to keep a hot loop allocation-free, land them in buffers you own instead.
The row width follows from the buffer lengths, and may exceed the wanted count to land rows inside a wider matrix.

```rust
let mut keys = vec![Key::NIL; queries_count * 10];
let mut distances = vec![0.0; queries_count * 10];
let mut counts = vec![0usize; queries_count];
let mut matches = MatchesSpan::new(&mut keys, &mut distances, &mut counts)?;
let mut stats = Stats::default();

index.search(&executor, &rows(&queries, 768)?, 10)
    .expansion_factor(128)
    .stats(&mut stats)
    .into(&mut matches)?;
println!("{} distances computed", stats.computed_distances);
```

Either output reads back the same way: `row` pairs one query's keys with its distances, trimmed to the cells the call wrote, and `rows` walks every query in order.

```rust
for row in matches.rows() {
    for (key, distance) in row.keys.iter().zip(row.distances) {
        println!("{key} at {distance}");
    }
}
```

Outputs whose size only the call discovers - the rows of `get`, the pairs of `multiply`, the bytes of `save_buffer` - land in an `Arena` you pass in.
It starts empty, is created by the first call, rewound and refilled by every later one, and freed on drop, so a steady loop allocates once.
What a call returns borrows the arena, so reusing it while that output lives fails to compile; pass a second arena to keep both.

```rust
let mut arena = Arena::default();
let rows = index.get::<f32>(&executor, key, &mut arena)?;
```

The candidate-pool breadth rides the call rather than the handle, as `expansion_factor` on every builder that walks the graph, so a recall sweep is a loop over `ef` values against one immutable index, with no mutation and no synchronization between points.

## Residency

Where the payload lives is an enum, not a flag, and it decides what loading a checkpoint means: materialize into RAM, upload to a device, or open in place on disk.

A residency also names the hardware its shards are placed across - one shard per memory domain, device, or path - and an empty list derives one from the machine.

```rust
use usearch::Residency;

Residency::Ram { memory_domains: vec![0, 1] };          // one graph per socket
Residency::Vram { devices: vec![0, 1, 2, 3] };          // sharded, never replicated
Residency::Nvme { paths: vec!["/mnt/nvme0".into()] };   // striped across the drives named
```

The executor is built from the same placement lists, which is what keeps compute and memory domains in agreement.

An NVMe handle opens a checkpoint in place and is read-only, so every mutating verb fails on it.
Bigger indexes come from merging, never mutating: shards build in RAM, serialize, and join a manifest.

## Serialization

The path extension picks the layout - a `.safegraphs` path is one monolith, a `.safegraphs.json` path is a manifest cataloging many.

```rust
index.save(&executor, "index.safegraphs").run()?;
let loaded = DenseIndex::load(&executor, "index.safegraphs").run()?;
let viewed = DenseIndex::view("index.safegraphs")?;       // read-only, served from the mapping

let metadata = DenseIndex::metadata(&executor, "index.safegraphs")?; // geometry without loading a shard
```

The result carries only the four fields a checkpoint froze, so `values_per_key` and the placement stay yours to set on the load builder.

Checkpoints are residency-portable: one file materializes into RAM, uploads to a device, or opens in place on disk, decided by the `residency` the load names.
In-memory buffers hold single monoliths only, through `save_buffer` and `load_buffer`.

## Metrics

Every metric derives from a single inner-product kernel with pre-computed magnitudes; arbitrary metrics belong to the C++ baseline tier, forfeiting the SIMD kernel chains and the magnitude cache both.

- `Metric::Inner` - inner product.
- `Metric::Angular` - cosine distance, over pre-computed magnitudes.
- `Metric::Euclidean` - L2 distance.
- `Metric::Hamming` - bit-level, over the packed sub-byte kinds.
- `Metric::Jaccard` - bit-level, over the packed sub-byte kinds.

## Filtering With Predicates

Cross-referencing an external database during traversal beats paging through results and filtering afterwards.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.

```rust
let is_odd = |key: Key| key.as_u64().is_some_and(|value| value % 2 == 1);
let matches = index.search(&executor, &rows(&query, 768)?, 10).predicate(&is_odd).run()?;
```

## Progress and Cancellation

Every long verb's builder takes a `progress` handler, called with the units finished so far and the units the call finishes if it runs to the end.
Answering `Continuation::Cancel` stops the call at the next increment boundary.

```rust
use std::time::{Duration, Instant};
use usearch::Continuation;

let deadline = Instant::now() + Duration::from_secs(30);
let mut before_the_deadline = move |_completed: usize, _total: usize| match Instant::now() < deadline {
    true => Continuation::Continue,
    false => Continuation::Cancel,
};
index.refine(&executor).expansion_factor(128).progress(&mut before_the_deadline).run()?;
```

The handler is `FnMut` and `Send` where a predicate is `Fn` and `Sync`: it arrives on whichever thread advanced the work and never on two at once, which is what lets it carry a counter or a deadline without a lock of its own.
A panic inside it is caught and read as a cancellation, unwinding out of a C frame being an abort.
Reports are denominated in whatever the verb consumes - rows for an insertion, queries for a search, donors for a merge, shards for a checkpoint - and how often they arrive is the increment's size rather than a promise, so read the counts and never the number of calls.

What a stop leaves behind is each verb's own contract.
An `add` stays all-or-nothing: a stopped batch reports `Status::Cancelled` and publishes no row.
A stopped `refine` is not a failure but the time budget a whole-graph sweep is meant to run under - re-selection claims no identity, so the graph stays correct and merely less improved.
A stopped `merge` keeps the donors it already absorbed, and its count is the only way to learn where the boundary fell; the donors past it are freed for you, having been handed over by value.

## Linear Algebra Over The Graph

Three verbs treat the index as a sparse operand rather than as a kNN store, and each folds its arithmetic into the graph walk instead of materializing a match list.
All three need logits rather than distances, so they run on an inner-product index alone.

`multiply` is a sparse matrix product: every left row is walked against the index, and every pair the walk scored lands in an arena as one ragged run.
`wanted` keeps the first pairs of each row, and `required` counts how many there were.

```rust
let products = index.multiply(&executor, &rows(&left, 768)?).run(&mut arena)?;
for row in products.rows() {
    println!("{} pairs", row.keys.len());
}
```

`maxsim` is late interaction - every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `values_per_key` above one exists for, and the wanted count is over documents rather than rows.

```rust
let documents = index.maxsim(&executor, &rows(&tokens, 768)?, tokens_per_query, 10).run()?;
```

`attend` is softmax-weighted attention, gathering each value row exactly once and returning outputs already divided by their partition.
An entry's key is its row offset into `values`, so a context inserted in position order needs no mapping, and a `carry` of one `SoftmaxState` per query row folds this branch into another over the same queries.

```rust
let mut carry = vec![SoftmaxState::default(); queries_count];
index.attend(&executor, &rows(&queries, 768)?, &rows(&values, 768)?, &mut rows_mut(&mut context, 768)?)
    .carry(&mut carry)
    .run()?;
```

## Refinement

`refine` re-runs the neighbor selection over entries already in the graph and returns how many links it rewrote, so a batch built at a low expansion factor reaches the quality of a slower build afterwards.
The count reaching zero is what settled looks like.

```rust
while index.refine(&executor).expansion_factor(128).run()? != 0 {}
```

## Scalar Types

The scalar types are NumKong's, so nothing needs re-wrapping to move between the two crates - `f16` and `bf16` included, emulated where the CPU has no native support.

| Rust type                      | `ScalarKind`                   |
| :----------------------------- | :----------------------------- |
| `f64`, `f32`                   | `F64`, `F32`                   |
| `bf16`, `f16`                  | `BF16`, `F16`                  |
| `e5m2`, `e4m3`, `e3m2`, `e2m3` | `E5M2`, `E4M3`, `E3M2`, `E2M3` |
| `i8`, `i4x2`                   | `I8`, `I4x2`                   |
| `u8`, `u4x2`, `u1x8`           | `U8`, `U4x2`, `U1x8`           |

Initializing an index with a packed quantization and adding floating-point vectors quantizes on ingest; the `u1x8` type also lets you pass packed bit-vectors directly.

```rust
let mut options = Options::default();
options.dimensions = 8;
options.metric = Metric::Hamming;
options.quantization = ScalarKind::U1x8;
let index = DenseIndex::new(&options)?;

let slab = [u1x8(0b0000_1111), u1x8(0b1111_0000)];
index.add(&executor, &keys, &rows(&slab, 8)?).run()?;
```

## Streaming and Iteration

A search stream is a resumable best-first traversal, so callers that post-filter aggressively pull until satisfied instead of guessing "k" upfront.

```rust
for found in index.search_stream(&executor, &rows(&query, 768)?)? {
    let found = found?;
    if found.distance > 0.3 { break; }
    println!("{} at {}", found.key, found.distance);
}
```

A cursor walks the index contents as a sequential row scan, reading each shard's publication watermark as it enters that shard, so a row once passed is never revisited.

```rust
let mut cursor = index.cursor::<f32>()?;
let mut keys = vec![Key::NIL; 1024];
let mut vectors = vec![0.0f32; 1024 * 768];
loop {
    let written = cursor.next_batch(&mut keys, Some(&mut usearch::rows_mut(&mut vectors, 768)?))?;
    if written == 0 { break; }
}
```

## Introspection

```rust
let description = index.describe(&executor)?;
println!("{} entries in {} bytes", description.size, description.memory_usage);
println!("dispatched to {}", description.capabilities);
println!("compiled for {}", usearch::capabilities_compiled());
println!("available here {}", usearch::capabilities_available());
```
