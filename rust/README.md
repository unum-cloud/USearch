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
An index owns rows; an `Executor` owns the threads or streams a call runs on; an `Arena` holds what a call writes back.

```rust
use usearch::*;

let options = DenseOptions {
    dimensions: 4,
    metric: DenseMetric::Angular,
    quantization: Scalar::I8,
    ..DenseOptions::default()
};
let index = DenseIndex::init_ram(&options, &[])?;
index.reserve(1_000)?;

let keys = [Key::from(42u64), Key::from(43u64)];
let slab: [f32; 8] = [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2];
index.insert(&keys, &rows(&slab, 4)?)?;

let executor = Executor::for_ram(&[], 0, 0)?;
let mut arena = Arena::new_in(&executor)?;
let query: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
let matches = index.on(&executor).search(&rows(&query, 4)?, 10, &mut arena)?;   // on an executor, into its arena
let mut local_arena = Arena::new()?;
let local = index.search(&rows(&query, 4)?, 10, &mut local_arena)?;             // the calling thread, into host pages
```

Every verb but `clear` comes from the `DenseVerbs` trait, which `use usearch::*` brings into scope.
`clear` takes `&mut DenseIndex`, since no other call on the handle may overlap it.

One executor serves any number of indexes, and its residency must match the one their handles were built for.
There is a constructor per residency, each taking only what its own hardware has: `for_ram` and `for_nvme` take a thread count where zero derives one, and `for_vram` takes none, since one host thread drives one stream per device, but may borrow your CUDA streams.
Each ends with a `deferred_limit` sizing its table of deferred calls, where zero takes the default.
`Executor::default()` owns nothing and runs every call handed to it on the calling thread.
Calls on one executor start in submission order, so two threads that genuinely overlap take one apiece; index handles stay freely shared.

`rows` borrows a packed slab; a dataset whose rows are padded to a wider pitch passes `MatrixView::from_raw_parts` instead, which takes the byte stride directly.
The scalar type of the input is independent of the stored quantization - the `f32` slab above lands in an `i8` index - so nothing needs converting before insertion.

## Deferred Calls

A deferred receiver submits each call and answers a `Deferred`, which owns the inputs the library reads until the call is final and borrows the index and the arena, as a blocking call does.

```rust
use numkong::Matrix;

let queries = Matrix::<f32>::from_slice(&slab, &[2, 4]).expect("8 values fill 2 × 4");
let call = index.deferred(&executor).search(queries, 10, &mut arena)?;   // Deferred<'_, DenseSearch<..>>
if call.status() == Err(Status::Deferred) {
    // other work
}
let matches = call.await?;                                               // or: call.wait()?
```

The waited result is the `Matches` a blocking search answers, borrowing the same arena.
The library never frees an arena under a call: dropping one waits for the calls writing into it, and a call naming a busy arena finishes with `Status::Contended`, so even a leaked `Deferred` leaves it safe.

Inside an `async` block, `.await` polls the call's status until it is final, so other tasks run meanwhile, while `wait` blocks the thread without spinning.

An executor without threads of its own may hold a deferred call until its `wait`, its next `drain` or blocking call, or the drop of what it names, so `drain` it before an `.await` there.
`status` answers `Ok(())` once the call succeeded, `Err(Status::Deferred)` while it is not final, and its failure otherwise.
A submission finding the executor's deferred table full finishes at once with `Status::Contended`.
Calls depending on each other are ordered by waiting on the first before submitting the second, since no call carries an ordering field.

## Configs and Outputs

Each configurable verb's config converts into its `Dense*Config` struct, whose other fields are named and default to what the C interface derives.
So the bare value is the whole call - `search(&queries, 10, &mut arena)`, `insert(&keys, &vectors)`, `save(path)` - and `..DenseSearchConfig::from(10)` reaches every knob.

```rust
let mut arena = Arena::new_in(&executor)?;
let mut stats = DenseTraversalStats::default();
let matches = index.on(&executor).search(&rows(&queries, 768)?, DenseSearchConfig {
    expansion_factor: 128,
    stats: Some(&mut stats),
    ..DenseSearchConfig::from(10)
}, &mut arena)?;
println!("{} distances computed", stats.computed_distances);
```

`row` pairs one query's keys with its distances, trimmed to the cells the call wrote, and `rows` walks every query in order.

```rust
for row in matches.rows()? {
    for (key, distance) in row.keys.iter().zip(row.distances) {
        println!("{key} at {distance}");
    }
}
```

Every output whose size scales with the call - the matches of `search` and `maxsim`, the rows of `get` and `attend`, the pairs of `multiply`, the bytes of `save_buffer`, cursor and stream batches - lands in an `Arena` the call borrows as `&mut Arena`.
The result borrows it in turn until it is dropped, and it stays yours when the call fails.
`Arena::new_in(&executor)` places it where that executor computes and `Arena::new()` makes host pages for calls on the calling thread; it is rewound and refilled by every call and freed on drop, so a steady loop allocates once.

```rust
let mut arena = Arena::new_in(&executor)?;
let rows = index.on(&executor).get::<f32>(key, &mut arena)?;
println!("{:?}", rows.view()?.shape());
let rows = index.on(&executor).get::<f32>(key, &mut arena)?;       // rewinds it, so the rows above are gone
```

An arena made by a VRAM executor holds device memory, so its `residency()` is `Residency::Vram` and every host accessor of what it holds, such as `row`, `rows`, `view` and `keys`, answers `Status::Device` instead of a slice.

The candidate-pool breadth rides the call rather than the handle, as `expansion_factor` on every config that walks the graph, so a recall sweep is a loop over `ef` values against one immutable index, with no mutation and no synchronization between points.

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each takes only the placement its own hardware has - memory domains, devices, or paths, one shard per entry - and an empty list derives the placement: every NUMA node, one free device, or wherever the manifest points.

```rust
use std::path::Path;

let in_ram = DenseIndex::init_ram(&options, &[0, 1])?;               // one graph per socket
let in_vram = DenseIndex::init_vram(&options, &[0, 1, 2, 3])?;       // sharded, never replicated
let on_nvme = DenseIndex::view_nvme(DenseViewNvmeConfig {
    paths: &[Path::new("/mnt/nvme0")],                               // striped across the drives named
    ..DenseViewNvmeConfig::from("index.safegraphs")
})?;
```

The executor is built from the same placement lists, which is what keeps compute and memory domains in agreement.

An NVMe handle opens a checkpoint in place and is read-only, so every mutating verb fails on it.
Until NVMe checkpoints carry keys, `view_nvme` answers `Status::Unsupported`.
Bigger indexes come from merging, never mutating: shards build in RAM, serialize, and join a manifest.

## Serialization

The path extension picks the layout - a `.safegraphs` path is one monolith, a `.safegraphs.json` path is a manifest cataloging many.

```rust
index.on(&executor).save("index.safegraphs")?;
let loaded = DenseIndex::load_ram("index.safegraphs")?;
let gpu = unsafe { Executor::for_vram(&[0], &[], 0, 0)? };
let placed = DenseIndex::load_vram(DenseLoadVramConfig {
    executor: Some(&gpu),
    devices: &[0],
    ..DenseLoadVramConfig::from("index.safegraphs")
})?;
let viewed = DenseIndex::view_ram("index.safegraphs")?;          // read-only, served from the mapping
let metadata = DenseIndex::metadata("index.safegraphs")?;        // geometry without loading a shard
```

The metadata carries only the four fields a checkpoint froze, so `values_per_key` and the placement stay yours to set in the load config.

Checkpoints are residency-portable: one file materializes into RAM, uploads to a device, or opens in place on disk, decided by the constructor that reads it.
In-memory buffers hold single monoliths only, through `save_buffer` and `load_ram_buffer` or `load_vram_buffer`.

```rust
let mut arena = Arena::new_in(&executor)?;
let monolith = index.on(&executor).save_buffer(&mut arena)?;
let restored = DenseIndex::load_ram_buffer(monolith.view()?)?;
```

## Metrics

Every metric derives from a single inner-product kernel with pre-computed magnitudes; arbitrary metrics belong to the C++ baseline tier, forfeiting the SIMD kernel chains and the magnitude cache both.

- `DenseMetric::Inner` - inner product.
- `DenseMetric::Angular` - cosine distance, over pre-computed magnitudes.
- `DenseMetric::Euclidean` - L2 distance.
- `DenseMetric::Hamming` - bit-level, over `u1x8` rows alone.
- `DenseMetric::Jaccard` - bit-level, over `u1x8` rows alone.

## Scalar Types

The scalar types are NumKong's, so nothing needs re-wrapping to move between the two crates - `f16` and `bf16` included, emulated where the CPU has no native support.

| Rust type                      | `Scalar`                       |
| :----------------------------- | :----------------------------- |
| `f64`, `f32`                   | `F64`, `F32`                   |
| `bf16`, `f16`                  | `BF16`, `F16`                  |
| `e5m2`, `e4m3`, `e3m2`, `e2m3` | `E5M2`, `E4M3`, `E3M2`, `E2M3` |
| `i8`, `i4x2`                   | `I8`, `I4x2`                   |
| `u8`, `u4x2`, `u1x8`           | `U8`, `U4x2`, `U1x8`           |

Initializing an index with a packed quantization and inserting floating-point vectors quantizes on ingest; the `u1x8` type also lets you pass packed bit-vectors directly.

```rust
use numkong::u1x8;

let options = DenseOptions {
    dimensions: 8,
    metric: DenseMetric::Hamming,
    quantization: Scalar::U1x8,
    ..DenseOptions::default()
};
let index = DenseIndex::init_ram(&options, &[])?;
let slab = [u1x8(0b0000_1111), u1x8(0b1111_0000)];
index.insert(&keys, &rows(&slab, 8)?)?;
```

## Filtering With Predicates

Cross-referencing an external database during traversal beats paging through results and filtering afterwards.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.

```rust
let mut arena = Arena::new_in(&executor)?;
let is_odd = |key: Key| u64::try_from(key).is_ok_and(|value| value % 2 == 1);
let matches = index.on(&executor).search(&rows(&query, 768)?, DenseSearchConfig {
    predicate: Some(&is_odd),
    ..DenseSearchConfig::from(10)
}, &mut arena)?;
```

A predicate is `Fn` and `Sync`, since the executor's threads call it together, and a panic inside it rejects the key and resurfaces once the call is final.

## Streaming and Iteration

A search stream is a resumable best-first traversal, so callers that post-filter aggressively pull until satisfied instead of guessing "k" upfront.
It walks exactly one query row, and each pull lands up to `wanted` more matches in an arena as the one row of `Matches`, empty once the walk is drained.
No residency implements the stream yet, so the engine answers `Status::Unsupported`.

```rust
let mut arena = Arena::new_in(&executor)?;
let mut stream = index.on(&executor).search_stream(rows(&query, 768)?)?;
'pulls: loop {
    let pulled = stream.next(64, &mut arena)?;
    let row = pulled.row(0)?;
    if row.keys.is_empty() { break; }
    for (key, distance) in row.keys.iter().zip(row.distances) {
        if *distance > 0.3 { break 'pulls; }
        println!("{key} at {distance}");
    }
}
```

A cursor walks the index contents as a sequential row scan, reading each shard's publication watermark as it enters that shard, so a row once passed is never revisited.
`next_keys` walks the key column alone, and `next_batch` copies the rows beside the keys, both into an arena.

```rust
let mut arena = Arena::new_in(&executor)?;
let mut cursor = index.on(&executor).cursor()?;
loop {
    let batch = cursor.next_batch::<f32>(1024, &mut arena)?;
    if batch.written() == 0 { break; }
    println!("{} keys beside {:?} rows", batch.keys()?.len(), batch.view()?.shape());
}
```

## Refinement

`refine` re-runs the neighbor selection over entries already in the graph and returns how many nodes had their neighbors changed, so a batch built at a low expansion factor reaches the quality of a slower build afterwards.
The count reaching zero is what settled looks like.

```rust
let receiver = index.on(&executor);
while receiver.refine(DenseRefineConfig { expansion_factor: 128, ..DenseRefineConfig::default() })? != 0 {}
```

## Progress and Cancellation

Every long verb's config takes a `progress` handler, called with the units finished so far and the units the call finishes if it runs to the end.
Answering `ControlFlow::Break(())` stops the call at the next increment boundary, except an `insert`, which ignores it.

```rust
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

let deadline = Instant::now() + Duration::from_secs(30);
let mut before_the_deadline = move |_finished: usize, _total: usize| match Instant::now() < deadline {
    true => ControlFlow::Continue(()),
    false => ControlFlow::Break(()),
};
index.on(&executor).refine(DenseRefineConfig {
    expansion_factor: 128,
    progress: Some(&mut before_the_deadline),
    ..DenseRefineConfig::default()
})?;
```

The handler is `FnMut` and `Send` where a predicate is `Fn` and `Sync`: it arrives on whichever thread advanced the work and never on two at once, which is what lets it carry a counter or a deadline without a lock of its own.
A panic inside it is caught, read as a stop and re-raised once the call is final, unwinding out of a C frame being an abort.
Reports are denominated in whatever the verb consumes - rows for an insertion, queries for a search, donors for a merge, shards for a checkpoint, bytes for a buffer - and how often they arrive is the increment's size rather than a guarantee, so read the counts and never the number of calls.

What a stop leaves behind is each verb's own contract.
An `insert` cannot be stopped: it claims every row before its first report, ignores a `Break`, and publishes the whole batch at once.
A stopped `refine` is not a failure but the time budget a whole-graph sweep is meant to run under - re-selection claims no identity, so the graph stays correct and merely less improved.
A stopped `merge` keeps the donors it already absorbed, and its count is the only way to learn where the boundary fell; the donors past it are freed for you, having been handed over by value.
No residency absorbs donors yet, so a `merge` answers `Status::Unsupported` and frees the donors.

## Linear Algebra Over The Graph

Three verbs treat the index as a sparse operand rather than as a kNN store, and each folds its arithmetic into the graph walk instead of materializing a match list.
All three need logits rather than distances, so they run on an inner-product index alone.

`multiply` is a sparse matrix product: every left row is walked against the index, and every pair the walk scored lands in an arena as one ragged run.
`wanted` keeps the first pairs of each row, where zero keeps every pair, and `required` counts how many there were.
The run's Apache Arrow offsets land in the arena beside the keys and products.

```rust
let mut arena = Arena::new_in(&executor)?;
let products = index.on(&executor).multiply(&rows(&left, 768)?, 32, &mut arena)?;
for row in products.rows()? {
    println!("{} pairs", row.keys.len());
}
```

`maxsim` is late interaction - every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `values_per_key` above one exists for, and the wanted count is over documents rather than rows.

```rust
let documents = index.on(&executor).maxsim(&rows(&tokens, 768)?, tokens_per_query, 10, &mut arena)?;
```

`attend` is Flash Attention folded into the walk: every key it scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, and the outputs land in an arena already divided by the partition, as rows of the queries' own scalar type.
The index stores no values, so a `ValueLookup` closure resolves every scored key to its value row, as wide as the keys, stored in the index's own quantization, and borrowed from storage outliving the call.
A `None` row, or rows of another scalar type or width than the library reads, refuses the call with `Status::ShapeMismatch`, and a VRAM handle refuses the host lookup with `Status::InvalidArgument`.
It runs on the executor's threads together, hence `Sync`, and a panic inside it refuses the call and resurfaces once the call is final.
With a `head_count` of 8, row `t * 8 + h` of the queries is head `h` of token `t`, and the heads of one token share its walk.
A `carry` of one `DenseSoftmaxState` per query row, lent by you, folds this branch into another over the same queries, such as a recent window kept out of the index.

```rust
let value_lookup: &ValueLookup<f32> = &|key| cache.get(&key).map(Vec::as_slice);
let mut carry = vec![DenseSoftmaxState::default(); queries_count];
let context = index.on(&executor).attend(&rows(&queries, 128)?, value_lookup, DenseAttendConfig {
    carry: Some(&mut carry),
    ..DenseAttendConfig::from(8)
}, &mut arena)?;
println!("{:?}", context.view()?.shape());
```

## Introspection

```rust
let description = index.describe()?;
println!("{} entries in {} bytes", description.size, description.memory_usage);
println!("dispatched to {}", description.capabilities);
println!("compiled for {}", usearch::capabilities_compiled());
println!("available here {}", usearch::capabilities_available());

let reachability = index.health()?;
println!("{} of {} entries reachable", reachability.reachable, reachability.audited);
```

`health` audits how much of the graph a search reaches from its landmarks; entries it misses stay unfound until a `refine` re-links them.

## Design

The crate is a thin layer over the C ABI in `include/usearch/`, and every rule below follows from that.

- __Names follow the header.__
  Each module mirrors one header, every C type keeps its name without `usearch_` and `_t`, a verb's argument struct becomes its `Dense*Config`, and every wrapper's documentation names the C symbol it calls.
- __Blocking verbs return `Result` directly.__
  `index.search(&queries, 10, &mut arena)?` takes no executor, completion or call value, and nothing about asynchrony reaches a caller who never asks for it.
- __Every byte has one owner.__
  Callers lend memory, never an allocator: no allocation callback crosses into the library.
  Payload belongs to the index, placed by its constructor's residency, and scratch belongs to the executor, grown to a high-water mark.
  Every output whose length scales with the call lands in an `Arena`, while fixed-size reports such as stats, descriptions and `required` stay plain values, and so does state you carry between calls, like the attention `carry`.
  Executors make arenas: `Arena::new_in(&executor)` places one where that executor computes, decided once, and `Arena::new()` makes host pages for the calling thread.
  A call rewinds its arena, so the previous call's outputs are gone, and the borrow checker enforces it: a result borrowing the arena must be dropped before the next call takes it, which also keeps a call's inputs out of the arena it rewinds.
  Every call, blocking or deferred, borrows the arena as `&mut Arena`, since freeing an arena already waits for the calls writing into it.
- __The receiver chooses the completion, and the verb stays the same.__
  `DenseVerbs` writes every shared verb once, for `&DenseIndex` on the calling thread and for the `On` that `index.on(&executor)` or `index.deferred(&executor)` makes.
  An `On` borrows the index and the executor, so neither it nor a `Deferred` it submits can outlive them.
- __A deferred receiver refuses borrowed inputs at compile time.__
  The library reads a deferred call's inputs after the verb returns, so lending `&[Key]` or `&MatrixView` fails with "a deferred receiver needs owned inputs".
  A config or view borrowing a local fails as "does not live long enough".
  It takes keys in a `Vec` or an array, rows in a NumKong `Matrix` or a `'static` `MatrixView`, a `PathBuf`, `String` or `'static` path, and `'static` configs, handlers and value lookups.
- __Nothing costs anything until it is used.__
  A blocking call allocates no box and takes no lock; a deferred one allocates one box holding its outcome, argument struct and inputs until the call is final.
  Beyond that box the binding allocates nothing: an arena growing past its high-water mark is the only allocation a steady loop sees.
  The deferred box is the crate's rather than yours, since a `Deferred` leaked without a wait would leave the library writing into storage you had reused.
- __A `Deferred` has no life of its own.__
  `status` is an acquire load of the outcome word, `wait` blocks in `usearch_outcome_wait`, and dropping an unfinished one waits the same way before freeing its box.
  The crate starts no thread: `.await` polls the status word, since the library has no wake-up, and a `Deferred` borrows its index, so waiting off-thread goes through a scoped thread, as `std::thread::scope(|scope| scope.spawn(move || call.wait()).join())` does.
- __Statuses and modes are enums.__
  `Status`, `Mood`, `DenseMetric` and `Scalar` carry the C enumerators' values, leaving success and the unknown kinds to `Result` and `Option`, so nothing is compared as a string.
  Their `Display` and `FromStr` spell the names the C tables own.

Constructors block: `DenseIndex::init_ram` and `init_vram`, the `load_*` and `view_*` families, and `metadata`, with each load config's `executor` choosing where it fans out.
So do cursors and search streams, which pull on the executor of the receiver that opened them.
