# USearch for Rust

Detailed symbol list and documentation for the USearch Rust SDK can be found on [docs.rs](https://docs.rs/usearch).

## Installation

```sh
cargo add usearch
```

The crate links the pre-instantiated C library and consumes it through its stable C headers, so no index template is instantiated in your build.
[NumKong](https://github.com/ashvardanian/NumKong) is a hard dependency rather than a feature: its matrices, scalar types, and traits _are_ this crate's vocabulary, and it supplies the SIMD kernels.

The crate compiles no C or C++ of its own: CMake builds the engine and the crate links it.

## Quickstart

Every batch input is a NumKong matrix - a base pointer, a row count, and a byte stride - so a memory-mapped dataset feeds the index with no reshape and no copy, and every ranked output is a `MatchesSpan` over three buffers you own.
An index owns rows; an `Executor` owns the threads or streams a fanned-out verb runs on, and every batch verb takes one.

```rust
use usearch::dense::{InsertionOptions, MatchesSpan, Metric, Options, SearchOptions, Stats};
use usearch::{rows, DenseIndex, Distance, Executor, Key, ScalarKind};

let index = DenseIndex::new(&Options {
    dimensions: 4,
    metric: Metric::Angular,
    quantization: ScalarKind::I8,
    ..Options::default()
})?;
index.reserve(1_000)?;

let executor = Executor::for_ram(&[], 0)?;

let keys = [Key::from(42u64), Key::from(43u64)];
let slab: [f32; 8] = [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2];
index.add(&executor, &keys, rows(&slab, 4), InsertionOptions::default(), None)?;

let query: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
let (mut found, mut scores, mut counts) = ([Key::NIL; 10], [0.0 as Distance; 10], [0usize; 1]);
let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
index.search(&executor, rows(&query, 4), SearchOptions::default(), &mut matches, None)?;
```

One executor serves any number of indexes, and its residency must match the one their handles were built for.
There is a factory per residency, each taking only what its own hardware has: `for_ram` and `for_nvme` take a thread count where zero derives one, and `for_vram` takes none, since one host thread drives one stream per device.
It admits a single driver at a time, so two threads that genuinely overlap take one apiece; index handles stay freely shared.

`rows` borrows a packed slab; a dataset whose rows are padded to a wider pitch passes `MatrixView::from_raw_parts` instead, which takes the byte stride directly.
The scalar type of the input is independent of the stored quantization - the `f32` slab above lands in an `i8` index - so nothing needs converting before insertion.

## Search Allocates Nothing

There is one search verb and it writes into buffers you own, so a benchmark loop allocates once and never again.
The row width follows from the buffer lengths rather than being asked for, so the wanted count cannot disagree with the space reserved for it.

```rust
let mut keys = vec![Key::NIL; queries_count * 10];
let mut distances = vec![0.0; queries_count * 10];
let mut counts = vec![0usize; queries_count];
let mut matches = MatchesSpan::new(&mut keys, &mut distances, &mut counts)?;
let mut stats = Stats::default();

let options = SearchOptions::with_expansion_factor(128);
index.search(&executor, rows(&queries, 768), options, &mut matches, Some(&mut stats))?;
println!("{} distances computed", stats.computed_distances);
```

A `MatchesSpan` borrows the three buffers for as long as it lives, so results are read back through it rather than through the arrays underneath.
`row` pairs one query's keys with its distances, trimmed to the cells the walk wrote, and `rows` walks every query in order.

```rust
for row in matches.rows() {
    println!("{} of {} matches written", row.keys.len(), row.found);
    for (key, distance) in row.keys.iter().zip(row.distances) {
        println!("{key} at {distance}");
    }
}
```

The candidate-pool breadth rides the call rather than the handle, so a recall sweep is a loop over `ef` values against one immutable index, with no mutation and no synchronization between points.
`add` and `refine` take an `InsertionOptions`, the four read-only walks a `SearchOptions`, and the two stay separate types so a knob either side gains reaches that side alone.

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
index.save(&executor, "index.safegraphs")?;
index.load(&executor, "index.safegraphs")?;

let metadata = DenseIndex::metadata("index.safegraphs")?; // geometry without loading a shard
```

The result carries only the four fields a checkpoint froze, so `values_per_key` and the placement stay yours to set in `Options` before constructing a handle.

Checkpoints are residency-portable: one file materializes into RAM, uploads to a device, or opens in place on disk, decided by the handle that loads it.
In-memory buffers hold single monoliths only, through `save_buffer` and `load_buffer`.

## Metrics

Every metric derives from a single inner-product kernel with pre-computed magnitudes, which is why this binding does not expose the C layer's custom-callback escape hatch: a caller-supplied metric forfeits the SIMD kernel chains and the magnitude cache both.

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
index.search_if(&executor, rows(&query, 768), SearchOptions::default(), &mut matches, None, &is_odd)?;
```

## Progress and Cancellation

Every long verb has a reporting sibling - `add_reporting`, `refine_reporting`, `search_reporting`, `merge_reporting`, `save_reporting` and `load_reporting` - taking a handler called with the units finished so far and the units the call finishes if it runs to the end.
Answering `Continuation::Cancel` stops the call at the next increment boundary.

```rust
use std::time::{Duration, Instant};
use usearch::Continuation;

let deadline = Instant::now() + Duration::from_secs(30);
let mut before_the_deadline = |_completed: usize, _total: usize| match Instant::now() < deadline {
    true => Continuation::Continue,
    false => Continuation::Cancel,
};
index.refine_reporting(&executor, InsertionOptions::with_expansion_factor(128), None, &mut before_the_deadline)?;
```

The handler is `FnMut` and `Send` where a predicate is `Fn` and `Sync`: it arrives on whichever thread advanced the work and never on two at once, which is what lets it carry a counter or a deadline without a lock of its own.
A panic inside it is caught and read as a cancellation, unwinding out of a C frame being an abort.
Reports are denominated in whatever the verb consumes - rows for an insertion, queries for a search, donors for a merge, shards for a checkpoint - and how often they arrive is the increment's size rather than a promise, so read the counts and never the number of calls.

Three verbs mean something different once a handler is present.
An `add` batch stops being all-or-nothing: it advances in increments, a stop keeps the rows already published and reports `Status::Cancelled`, and the count the handler last saw is what the index kept.
A stopped `refine` is not a failure but the time budget a whole-graph sweep is meant to run under - re-selection claims no identity, so the graph stays correct and merely less improved, and the next call resumes where this one stopped.
A stopped `merge` keeps the donors it already absorbed, and its count is the only way to learn where the boundary fell; the donors past it are freed for you, having been handed over by value.

## Linear Algebra Over The Graph

Three verbs treat the index as a sparse operand rather than as a kNN store, and each folds its arithmetic into the graph walk instead of materializing a match list.
All three need logits rather than distances, so they run on an inner-product index alone.

`multiply` is a sparse matrix product: every left row is walked against the index, and the products land in the same `MatchesSpan` a search writes.

```rust
index.multiply(&executor, rows(&left, 768), SearchOptions::default(), &mut pairs, None)?;
```

`maxsim` is late interaction - every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `values_per_key` above one exists for, and the outputs are sized over documents rather than rows.

```rust
index.maxsim(&executor, rows(&tokens, 768), tokens_per_query,
             SearchOptions::default(), &mut documents, None)?;
```

`attend` is softmax-weighted attention, gathering each value row exactly once and returning outputs already divided by their partition.
An entry's key is its row offset into `values`, so a context inserted in position order needs no mapping, and a `carry` of one `SoftmaxState` per query row folds this branch into another over the same queries.

```rust
let mut carry = vec![SoftmaxState::default(); queries_count];
index.attend(&executor, rows(&queries, 768), rows(&values, 768),
             AttentionOptions::default(), &mut rows_mut(&mut context, 768),
             Some(&mut carry), None)?;
```

## Refinement

`refine` re-runs the neighbor selection over entries already in the graph and returns how many links it rewrote, so a batch built at a low expansion factor reaches the quality of a slower build afterwards.
The count reaching zero is what settled looks like.

```rust
while index.refine(&executor, InsertionOptions::with_expansion_factor(128), None)? != 0 {}
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
let index = DenseIndex::new(&Options {
    dimensions: 8,
    metric: Metric::Hamming,
    quantization: ScalarKind::U1x8,
    ..Options::default()
})?;

let slab = [u1x8(0b0000_1111), u1x8(0b1111_0000)];
index.add(&executor, &keys, rows(&slab, 8), InsertionOptions::default(), None)?;
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
