# USearch for Rust

The crate links the pre-instantiated C library and calls the stable C ABI in `c/usearch/` through `extern "C"` declarations, so no index template is instantiated in your build.
It needs Rust 1.100 or newer, CMake 3.28 and a C++20 compiler.
It exposes dense, sequence, shape and geospatial indexes.
Detailed symbol documentation lives on [docs.rs](https://docs.rs/usearch).

## Installation

```sh
cargo add usearch
```

CMake builds the engine from the crate's sources and links it as a static archive.
That build compiles the four most common dense storage types, `bf16`, `u1x8`, `i8` and `e4m3`, and an index over any other is refused as `Unsupported`.
[NumKong](https://github.com/ashvardanian/NumKong) is a hard dependency rather than a feature: its matrices, scalar types, and traits _are_ this crate's vocabulary, and it supplies the SIMD kernels.

Two features swap the implementation of the C interface the crate links, and naming both is refused:

- `cuda` compiles the CUDA backends in and needs a toolkit at build time, with `USEARCH_CUDA_ARCH` choosing the GPU code, the release set `all` by default.
- `stub` links a brute-force stand-in that answers the same verbs over the standard library alone, with exact search, seconds to build and no submodules, for testing a caller and never for shipping.

A WASI build reads the SDK from `WASI_SDK_PATH` and needs `-C target-feature=+simd128`.
`USEARCH_LIBRARY_DIR` names a directory already holding `libusearch_static.a`, such as the `lib/` of a prefix `cmake --install build_release --component static` filled, and skips the CMake build.

## Quickstart

Every batch input is a NumKong matrix of a base pointer, a row count and a byte stride, so a memory-mapped dataset feeds the index with no reshape and no copy.
An index owns entries, an `Executor` owns the threads or streams a call runs on, and an `Arena` holds what a call writes back.

```rust
use usearch::*;

let options = DenseOptions {
    dimensions: 4,
    metric: DenseMetric::Angular,
    quantization: Scalar::I8,
    ..DenseOptions::default()
};
let index = DenseIndex::init_ram(&options, &[])?;
index.reserve(1_000)?; // Optional capacity hint; inserts grow on demand

let keys = [Key::from(42u64), Key::from(43u64)];
let slab: [f32; 8] = [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2];
index.insert(&keys, &rows(&slab, 4)?)?;

let executor = Executor::for_ram(&[], None, 0)?;
let mut arena = Arena::new_in(&executor)?;
let query: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
let matches = index.on(&executor).search(&rows(&query, 4)?, 10, &mut arena)?;   // on an executor, into its arena
let mut local_arena = Arena::new()?;
let local = index.search(&rows(&query, 4)?, 10, &mut local_arena)?;             // the calling thread, into host pages
```

Every verb but `clear` comes from the `DenseVerbs` trait, which `use usearch::*` brings into scope.
`clear` takes `&mut DenseIndex`, since no other call on the handle may overlap it.

## Receivers & Deferred Calls

The receiver decides where a verb runs and how it completes, while the verb and its arguments stay the same:

- `index.search(..)` runs on the calling thread, through `&DenseIndex`.
- `index.on(&executor).search(..)` blocks on the executor.
- `index.deferred(&executor).search(..)` submits the call and answers a `Deferred`.

`DenseVerbs` writes every verb once for all three.
A blocking verb returns its `Result` directly, so nothing about asynchrony reaches a caller who never asks for it.
An `On` borrows the index and the executor, so neither it nor a `Deferred` it submits can outlive them.

```rust
use numkong::Matrix;

let queries = Matrix::<f32>::from_slice(&slab, &[2, 4]).expect("8 values fill 2 × 4");
let call = index.deferred(&executor).search(queries, 10, &mut arena)?;   // Deferred<'_, (Matrix<f32>, ..), Matches>
if call.status() == Err(Status::Deferred) {
    // other work
}
let matches = call.await?;                                               // or: call.wait()?
```

A `Deferred` owns the inputs the library reads until the call is final, and borrows the index and the arena, as a blocking call does.
The waited result is the `Matches` a blocking search answers, borrowing the same arena.
`status` answers `Ok(())` once the call succeeded, `Err(Status::Deferred)` while it is not final, and its failure otherwise.
Inside an `async` block, `.await` polls the call's status until it is final, so other tasks run meanwhile, while `wait` blocks the thread without spinning.
The crate starts no thread, so waiting off-thread goes through a scoped thread, as `std::thread::scope(|scope| scope.spawn(move || call.wait()).join())` does.
Dropping an unfinished `Deferred` waits for the call.

The library reads a deferred call's inputs after the verb returns, so a deferred receiver takes owned inputs: keys in a `Vec` or an array, rows in a NumKong `Matrix` or a `'static` `MatrixView`, a `PathBuf`, `String` or `'static` path, and `'static` configs, handlers and value lookups.
Lending `&[Key]` or `&MatrixView` fails to compile with "a deferred receiver needs owned inputs", and a config or view borrowing a local fails as "does not live long enough".
A deferred call allocates one box holding its outcome, inputs and output until it is final.
The box is the crate's rather than yours, since a `Deferred` leaked without a wait would leave the library writing into storage you had reused.

One call holds an executor at a time, so a second call finding it held finishes at once with `Status::Contended`, and nothing queues.
A deferred call on a caller-exclusive pool runs on the pool's thread 0, so polling alone finishes it, and its progress handler and predicate run on the pool's threads; an inclusive pool, a single thread, WebAssembly and a VRAM executor run it at submission.
Calls depending on each other are ordered by waiting on the first before submitting the second, since no call carries an ordering field.

## Executors

An executor owns host threads or device streams, and the traversal scratch each of them walks with, grown to a high-water mark.
One executor serves any number of indexes, and its residency must match the one their handles were built for.
There is a constructor per residency, each taking only what its own hardware has, and the placement lists are the ones the indexes were built with, which keeps compute and memory domains in agreement.

```rust
let threads = Executor::for_ram(&[], None, 8)?;                        // eight workers, where zero takes 7/8 of the cores
let domains = Executor::for_ram(&[0, 1], None, 0)?;                    // threads interleaved across two NUMA nodes
let capped = Executor::for_ram(&[], Some(CPUClass::IntelHaswell), 0)?; // waiters and atomics no newer than Haswell
let rings = Executor::for_nvme(&["/mnt/nvme0"], None, 4)?;             // one submission ring apiece
let streams = Executor::for_vram(&[0, 1], None, 0)?;                   // one stream per device, scratch grown as asked
let local = Executor::default();                                       // owns nothing, runs on the calling thread

threads.drain(Mood::Spinning)?;                                        // waits out the holding call, then keeps spinning
```

`for_ram` and `for_nvme` take a thread count, where zero derives one.
`for_vram` takes none, since one host thread drives each device's per-thread default stream, and `for_vram_on_streams` orders work on your CUDA streams instead, one per device.
`for_vram_on_streams` is `unsafe`, since you destroy those streams yourself, and only after dropping the executor; it creates and destroys no stream.
`Executor::default()` owns nothing and runs every call handed to it on the calling thread.
One call holds an executor at a time, so two threads that genuinely overlap take one apiece; an executor is `Send` but not `Sync`, while index handles stay freely shared.
`drain` waits until no call holds the executor and sets the `Mood` idle workers adopt, and dropping an executor drains it, unless a GPU arena it made still holds it, which then does so on its own drop.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in an `Arena` you pass, and the verb answers a view over it without copying.
`Arena::new_in(&executor)` places it where that executor computes, decided once, and `Arena::new()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
Every call, blocking or deferred, borrows the arena as `&mut Arena` and its result borrows it in turn, so the borrow checker refuses a stale view at compile time and keeps a call's inputs out of the arena it rewinds.
An arena from a VRAM executor holds device memory, which host accessors must not read; `Arena::describe` says where an arena's outputs live.
It grows, fills and frees on that executor's first stream, so it serves only calls on that executor, any other refused with `Status::Device`.
Every arena holds the executor that made it until it drops.

```rust
let mut arena = Arena::new_in(&executor)?;                           // Arena::new() is host pages for the calling thread
for batch in &batches {
    let matches = index.on(&executor).search(batch, 10, &mut arena)?; // rewinds: the last batch's matches are gone
    total += matches.query(0)?.distances.iter().sum::<f64>();
}

let (mut found, mut fetched) = (Arena::new()?, Arena::new()?);       // two live results, two arenas
for &key in index.search(&query, 10, &mut found)?.query(0)?.keys {
    rerank(key, index.get::<f32>(key, &mut fetched)?.view()?);       // rewinds fetched, never found
}

let (mut executors, mut arenas) = (Vec::new(), Vec::new());          // one executor and one arena per submitted call
for _ in &shards {
    executors.push(Executor::for_ram(&[], None, 8)?);
}
for executor in &executors {
    arenas.push(Arena::new_in(executor)?);
}
let mut calls = Vec::new();
for ((shard, executor), arena) in shards.iter().zip(&executors).zip(&mut arenas) {
    calls.push(shard.deferred(executor).search(queries.clone().expect("room for a copy"), 10, arena)?);
}
for call in calls {
    consume(call.await?);                                            // or: call.wait()?
}
```

The library never frees an arena under a call: dropping one waits for the calls writing into it, and a call naming an arena that serves another call finishes with `Status::Contended` and rewinds nothing, so even a leaked `Deferred` leaves it safe.
An arena's `describe()` says where it lives, and on which device.
Fixed-size reports, such as stats, descriptions and `required`, stay plain values, and so does state you carry between calls, like the attention `carry`.

## Keys

A `Key` is a 128-bit identity, rendered as a lowercase dashed UUID.
Integer identities park in the low eight bytes, so `Key::from(42u64)` prints as `00000000-0000-0000-0000-00000000002a`.

```rust
let key = Key::from(42u64);
let wide = Key::from(0x0123_4567_89ab_cdef_0123_4567_89ab_cdefu128);
let parsed: Key = "00000000-0000-0000-0000-00000000002a".parse()?;
assert_eq!(u64::try_from(parsed), Ok(42));                // refused when the high eight bytes are set
assert_eq!(u128::from(wide), 0x0123_4567_89ab_cdef_0123_4567_89ab_cdef);
```

Keys need not be unique: an index admits up to `values_per_key` entries under one, `count` reports how many it holds, and `get` returns them all.
`Key::NIL` is the all-zero key.

## Errors

Every fallible call returns `usearch::Result`, whose `Error` carries a `Status` and a static `message`, so nothing allocates on the failure path.
A failure the binding catches before the call crosses the boundary reports the status the engine would have used.

```rust
match index.on(&executor).search(&rows(&query, 4)?, 10, &mut arena) {
    Ok(matches) => println!("{} queries", matches.queries()),
    Err(Error { status: Status::Contended, .. }) => println!("the arena serves another call"),
    Err(error) => println!("{}: {}", error.status, error.message),
}
```

`Status` carries the C enumerators' values and leaves success to `Result`, so an `Error` never carries `Status::Deferred`, which only `Deferred::status` reports.
`Status`, `Mood`, `Residency`, `Traversal`, `Phase`, `RefineVerdict`, `DenseMetric` and `Scalar` print through `Display` and parse through `FromStr` the names the C tables own, such as `"out-of-memory"` or `"angular"`, so nothing is compared as a string.
A panic inside a predicate, progress handler or value lookup aborts the process, since it cannot unwind through the library, so handlers report through what they capture.

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each takes only the placement its own hardware has, as memory domains, devices or paths with one shard per entry, and an empty list derives the placement: every NUMA node, one free device, or wherever the manifest points.

```rust
use std::path::Path;

let in_ram = DenseIndex::init_ram(&options, &[0, 1])?;               // one graph per socket
let in_vram = DenseIndex::init_vram(&options, &[0, 1, 2, 3])?;       // sharded, never replicated
let on_nvme = DenseIndex::view_nvme(DenseViewNvmeConfig {
    paths: &[Path::new("/mnt/nvme0")],                               // striped across the drives named
    key_table_domains: Some(&[0]),                                   // keys resolved through RAM on node 0
    ..DenseViewNvmeConfig::from("index.safegraphs")
})?;
```

An NVMe handle serves a checkpoint in place and is read-only, so every mutating verb fails on it with `Status::ReadOnly`.
Without `key_table_domains`, it holds nothing per entry, so `count`, `get`, `sample`, MaxSim and hinted searches fail with `Status::Unsupported`.
`load_nvme` takes the same config and opens a manifest writable, so `refine` rewrites its parts one at a time, while a lone monolith fails with `Status::Unsupported`.
Bigger indexes come from merging, never mutating: shards build in RAM, serialize, and join a manifest.

## Serialization

The path extension picks the layout: a `.safegraphs` path is one monolith, and a `.safegraphs.json` path is a manifest cataloging many.

```rust
index.on(&executor).save("index.safegraphs")?;
let loaded = DenseIndex::load_ram("index.safegraphs")?;
let gpu = Executor::for_vram(&[0], None, 0)?;
let placed = DenseIndex::load_vram(DenseLoadVramConfig {
    executor: Some(&gpu),
    devices: &[0],
    ..DenseLoadVramConfig::from("index.safegraphs")
})?;
let viewed = DenseIndex::view_ram("index.safegraphs")?;          // read-only, served from the mapping
let metadata = DenseIndex::metadata("index.safegraphs")?;        // geometry without loading a shard
```

Checkpoints are residency-portable: one file materializes into RAM, uploads to a device, or opens in place on disk, decided by the constructor that reads it.
The metadata carries only the four fields a checkpoint froze, so `values_per_key` and the placement stay yours to set in the load config.
`view_ram` answers `Status::Unsupported` until RAM shards can serve a mapping in place.
A stopped `save` leaves leftovers every reader rejects, which are yours to remove.

In-memory buffers hold single monoliths only, through `save_buffer`, `load_ram_buffer` or `load_vram_buffer`, and `metadata_buffer`.

```rust
let mut arena = Arena::new_in(&executor)?;
let monolith = index.on(&executor).save_buffer(&mut arena)?;
let restored = DenseIndex::load_ram_buffer(monolith.view())?;
```

## Dense Indexes

A `DenseIndex` holds equidimensional dense vectors, built from `DenseOptions`: `dimensions`, `metric`, the stored `quantization`, and the `connectivity` and `values_per_key` bounds, where zero derives.
Every metric derives from a single inner-product kernel with pre-computed magnitudes; arbitrary metrics belong to the C++ baseline tier, forfeiting the SIMD kernel chains and the magnitude cache both.

- `DenseMetric::Inner` - inner product.
- `DenseMetric::Angular` - cosine distance, over pre-computed magnitudes.
- `DenseMetric::Euclidean` - L2 distance.
- `DenseMetric::Hamming` - bit-level, over `u1x8` vectors alone.
- `DenseMetric::Jaccard` - bit-level, over `u1x8` vectors alone.

### Rows & Scalars

`rows` borrows a packed slab; a dataset whose rows are padded to a wider pitch passes `MatrixView::from_raw_parts` instead, which takes the byte stride directly.
The scalar type of the input is independent of the stored quantization, so the `f32` slab of the Quickstart lands in an `i8` index with nothing converted beforehand.
The scalar types are NumKong's, so nothing needs re-wrapping to move between the two crates, `f16` and `bf16` included, emulated where the CPU has no native support.

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

An `insert` batch is all-or-nothing and becomes visible at once, and it overlaps safely with concurrent searches on the same handle.

### Search

Each configurable verb's config converts into its `Dense*Config` struct, whose other fields are named and default to what the C interface derives.
So the bare value is the whole call, as in `search(&queries, 10, &mut arena)`, `insert(&keys, &vectors)` or `save(path)`, and `..DenseSearchConfig::from(10)` reaches every knob.

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

`query` pairs one query's keys with its distances, nearest first and trimmed to the cells the call wrote, and `iter` walks every query in order.

```rust
for query in matches.iter() {
    for (key, distance) in query.keys.iter().zip(query.distances) {
        println!("{key} at {distance}");
}
}
```

The breadth of the candidates rides the call rather than the handle, as `expansion_factor` on every config that walks the graph, so a recall sweep is a loop over `ef` values against one immutable index, with no mutation and no synchronization between points.
`traversal: Traversal::Scan` scores every entry instead of walking, on RAM and NVMe handles, which gives that sweep its exact ground truth; a VRAM handle refuses it with `Status::Unsupported`.
`DenseTraversalStats` reports the members visited, the distances computed and the starvation `restarts`, a graph-connectivity signal rather than a cost, overwritten by every call it is passed to.

### Filtered Search

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

A predicate is `Fn` and `Sync`, since the executor's threads call it together.
A VRAM handle refuses a host predicate with `Status::Device`.

Every long verb's config takes a `progress` handler, called with the units finished so far and the units the call finishes if it runs to the end.
Answering `ControlFlow::Break(())` stops the call at the next increment boundary.

```rust
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

let deadline = Instant::now() + Duration::from_secs(30);
let mut before_the_deadline = move |_finished: usize, _total: usize| match Instant::now() < deadline {
    true => ControlFlow::Continue(()),
    false => ControlFlow::Break(()),
};
let refined = index.on(&executor).refine(DenseRefineConfig {
    expansion_factor: 128,
    progress: Some(&mut before_the_deadline),
    ..DenseRefineConfig::default()
});
if let Err(failure) = refined {
    assert_eq!(failure.status, Status::Cancelled, "only the deadline stops it");
}
```

The handler is `FnMut` and `Send` where a predicate is `Fn` and `Sync`: it arrives on whichever thread advanced the work and never on two at once, which is what lets it carry a counter or a deadline without a lock of its own.
Reports are denominated in whatever the verb consumes: entries for an insertion, queries for a search, a phase's entries or islands for a refine or a reshard, draws for a sample, shards for a checkpoint, and bytes for a buffer.
How often they arrive is the increment's size rather than a guarantee, so read the counts and never the number of calls.

What a stop leaves behind is each verb's own contract.
An `insert` cannot be stopped: it claims every entry before its first report, ignores a `Break`, and publishes the whole batch at once.
A stopped search reports `Status::Cancelled` and leaves no matches.
A stopped `refine` or `reshard` reports `Status::Cancelled` and leaves a graph that saves; without a plan it closes the round where it stopped, and with one it leaves the plan live for the next call.
A stopped load creates no index, and a stopped `save_buffer` leaves no monolith.

### Retrieval and Streaming

`get` lands every vector stored under a key in an arena, converted to the scalar type you name, one row per entry in insertion order; an absent key yields no rows.

```rust
let mut arena = Arena::new_in(&executor)?;
let vectors = index.on(&executor).get::<f32>(key, &mut arena)?;
println!("{:?}", vectors.view()?.shape());
let vectors = index.on(&executor).get::<f32>(key, &mut arena)?;    // rewinds it, so the vectors above are gone
```

A cursor walks the index contents, reading each shard's publication watermark as it enters that shard, so an entry once passed is never revisited.
Without start keys it yields every entry as stored; with them, the starts' own entries first, then the entries those link to breadth-first, each once.
`next_keys` walks the key column alone, and `next_links` copies the keys each entry links to beside it, both into an arena; the vectors themselves come from `get`.

```rust
let mut arena = Arena::new_in(&executor)?;
let mut cursor = index.on(&executor).cursor(&[])?;
loop {
    let batch = cursor.next_links(1024, &mut arena)?;
    if batch.written() == 0 { break; }
    println!("{} keys, the first linking to {:?}", batch.keys().len(), batch.links(0)?);
}
```

A search stream is a resumable best-first traversal, so callers that post-filter aggressively pull until satisfied instead of guessing "k" upfront.
It walks exactly one query row, and each pull lands up to `wanted` more matches in an arena as the one query of `Matches`, empty once the walk is drained.
No residency implements the stream yet, so the engine answers `Status::Unsupported`.

```rust
let mut arena = Arena::new_in(&executor)?;
let mut stream = index.on(&executor).search_stream(rows(&query, 768)?)?;
'pulls: loop {
    let pulled = stream.next(64, &mut arena)?;
    let found = pulled.query(0)?;
    if found.keys.is_empty() { break; }
    for (key, distance) in found.keys.iter().zip(found.distances) {
        if *distance > 0.3 { break 'pulls; }
        println!("{key} at {distance}");
}
}
```

### Refinement

`refine` re-runs the neighbor selection over entries already in the graph and returns a `RefineReport` of the nodes re-selected, those whose neighbors `changed`, the islands found and reconnected, the neighborhoods re-stitched with the links added and dropped, the `verdict`, and the `opening` and `reachability` health checks around the round, so a batch built at a low expansion factor reaches the quality of a slower build afterwards.
A `verdict` other than `RefineVerdict::Improved` is what settled looks like.
On a RAM handle searches may run throughout, on a VRAM one they may not, and insertions never may.

```rust
let receiver = index.on(&executor);
while receiver.refine(DenseRefineConfig { expansion_factor: 128, ..DenseRefineConfig::default() })?.verdict == RefineVerdict::Improved {}
```

On a loaded NVMe handle `memory_limit` bounds the bytes a round holds for the one part it rewrites: zero rounds the part in RAM, and a smaller budget streams its rounds from the drives.

A `DenseRefinePlan` carries a round across calls: a stop leaves it live, the next call holding it continues where it stopped, `phase` reports the `Phase` the call stopped in, and `refine_plan_finish` closes it.

```rust
let (mut plan, mut phase) = (DenseRefinePlan::default(), PhaseProgress::default());
while let Err(failure) = receiver.refine(DenseRefineConfig {
    progress: Some(&mut before_the_deadline),
    plan: Some(&mut plan),
    phase: Some(&mut phase),
    ..DenseRefineConfig::default()
}) {
    assert_eq!(failure.status, Status::Cancelled);
    println!("stopped in {} at {} of {}", phase.phase, phase.completed, phase.total);
}
let report = receiver.refine_plan_finish(plan)?;
```

A deferred call owns the plan it carries rather than borrowing it, and `wait_with_inputs` hands the plan back beside the answer.

```rust
let (answer, config) = index.deferred(&executor).refine(DenseRefineConfig::from(plan))?.wait_with_inputs();
let plan = config.plan.expect("the call hands its plan back");
```

### Resharding

`reshard` moves the entries of its `sources` into its `targets`, keeping their links rather than re-inserting their entries, then runs a round on every target that took entries; the receiver supplies the executor.
Several sources gather into the one target among them, one source cuts into several empty targets, and one handle alone recuts its own shards in place under `shard_capacity`.
Every handle stays yours: a source the targets do not name ends empty and live.
Readers of a RAM target may keep searching throughout, while a VRAM one admits no overlapping search.

```rust
index.on(&executor).reshard(DenseReshardConfig { sources: &[&index, &first_shard, &second_shard], targets: &[&index], ..Default::default() })?;
index.on(&executor).reshard(DenseReshardConfig { sources: &[&index], targets: &[&left, &right], ..Default::default() })?; // two empty targets take a part each
```

A stopped `reshard` cuts only its closing rounds, every entry already moved, and reports `Status::Cancelled`; a `DenseReshardPlan` continues it as a refine plan does, and `reshard_plan_finish` closes it.

### Sampling

`sample` draws keys around a focus, each hanging from a shown or earlier draw, and places them in three dimensions beside the keys already shown, so a view grows one connected sample as it zooms.

```rust
let drawn = index.on(&executor).sample(
    256,
    &mut arena,
    DenseSampleConfig { shown: &shown, shown_coordinates: &xyz, focus: &near, seed: 42, ..Default::default() },
)?;
let (keys, places) = (drawn.keys(), drawn.coordinates());
```

### Sparse Multiplication

`multiply`, `maxsim` and `attend` treat the index as a sparse operand rather than as a kNN store, and each folds its arithmetic into the graph walk instead of materializing a match list.
All three need logits rather than distances, so they run on an inner-product index alone.

`multiply` is a sparse matrix product: every left row is walked against the index, and every pair the walk scored lands in an arena as one ragged run, in traversal order.
`wanted` keeps the first pairs of each row, where zero keeps every pair, and `required` counts how many there were.
The run's Apache Arrow offsets land in the arena beside the keys and products.

```rust
let mut arena = Arena::new_in(&executor)?;
let products = index.on(&executor).multiply(&rows(&left, 768)?, 32, &mut arena)?;
for row in products.iter() {
    println!("{} pairs", row.keys.len());
}
```

### Late Interaction

`maxsim` is late interaction - every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `values_per_key` above one exists for, and the wanted count is over documents rather than rows.
Each query lands its best documents, largest score first, in the same ragged `DenseProducts` run `multiply` returns, and a `wanted` of zero keeps every document the query scored.

```rust
let documents = index.on(&executor).maxsim(&rows(&tokens, 768)?, tokens_per_query, 10, &mut arena)?;
for query in documents.iter() {
    println!("best {:?} of {} documents", query.keys.first(), query.keys.len());
}
```

### Attention

`attend` is Flash Attention folded into the walk: every key it scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, and the outputs land in an arena already divided by the partition, as rows of the queries' own scalar type.
The index stores no values, so a `ValueLookup` closure resolves every scored key to its value row, as wide as the keys, stored in the index's own quantization, and borrowed from storage outliving the call.
A `None` row, or rows of another scalar type or width than the library reads, refuses the call with `Status::ShapeMismatch`.
It runs on the executor's threads together, hence `Sync`.
A VRAM walk resolves values on the device, where no Rust closure runs, so an arena from a VRAM executor is refused with `Status::Device`.
With a `head_count` of 8, row `t * 8 + h` of the queries is head `h` of token `t`, and the heads of one token share its walk.
A `carry` of one `DenseSoftmaxState` per query row, lent by you, folds this branch into another over the same queries, such as a recent window kept out of the index.
`score_scale` is conventionally the inverse square root of the head width, and zero derives it from the frozen geometry.

```rust
let value_lookup: &ValueLookup<bf16> = &|key| cache.get(&key).map(Vec::as_slice);
let mut carry = vec![DenseSoftmaxState::default(); queries_count];
let context = index.on(&executor).attend(&rows(&queries, 128)?, value_lookup, DenseAttendConfig {
    carry: Some(&mut carry),
    ..DenseAttendConfig::from(8)
}, &mut arena)?;
println!("{:?}", context.view()?.shape());
```

### Introspection

```rust
let description = index.describe()?;
println!("{} entries in {} bytes", description.size, description.memory_usage);
println!("dispatched to {} and {}", description.cpu_class(), description.gpu_class());
println!("compiled for {}", usearch::capabilities_compiled());
println!("available here {}", usearch::capabilities_available());

let reachability = index.health()?;
println!("{} of {} entries reachable", reachability.reachable, reachability.audited);

println!("{} entries under {key}, contained: {}", index.count(key)?, index.contains(key)?);
```

`describe` reads the frozen metadata and the counters once each, so under concurrent insertions they may come from slightly different moments, and `memory_usage` excludes the executor's scratch.
`health` measures how much of the graph a search reaches from the root; entries it misses stay unfound until a `refine` re-links them.
`contains` is the blocking shorthand for a nonzero `count`.

## Sequence Indexes

A `SequenceIndex` searches byte strings, UTF-8 text, DNA, RNA and proteins under edit distances and alignments.
Its sequences cross as one `Tape`, the Apache Arrow `Binary` layout of bytes end to end around one more offset than entries: `Tape::of` packs separate entries, and `Tape::new` borrows the two runs of an Arrow column as they are.

```rust
let options = SequenceOptions {
    metric: SequenceMetric::LevenshteinUtf8,
    ..SequenceOptions::default()
};
let index = SequenceIndex::init_ram(&options, &[])?;
let words = Tape::of(["kitten", "sitting", "mitten"]);
index.insert([Key::from(1u64), Key::from(2u64), Key::from(3u64)], words)?;

let mut arena = Arena::new()?;
let matches = index.search(Tape::of(["sittin"]), 1, &mut arena)?;
assert_eq!(matches.query(0)?.keys, [Key::from(2u64)]);
assert_eq!(matches.query(0)?.distances[0], 1.0); // one edit away
```

`SequenceMetric::NeedlemanWunsch` takes `SequenceWeights`, a 256 × 256 substitution matrix with a gap cost, and `SequenceMetric::Overlap` takes `window_widths`; every load and view names the weights again, since no checkpoint records them.
`get` writes a key's sequences into an arena as `TapeEntries`, and `count` reports a key's entries with their total and largest byte lengths.
The verbs come from `SequenceVerbs`, and the receivers, reports and plans work as the dense ones do: `refine` and `reshard` take a `SequenceRefinePlan` or a `SequenceReshardPlan` and answer a `RefineReport`.

## Shape Indexes

A `ShapesIndex` searches 3D point clouds of any size, like molecules, conformers and protein backbones, invariant to rotation, reflection, translation and point order.
Its tape holds `[x, y, z]` points end to end, with offsets that count points, as an Arrow `List` column lays them out:

```rust
let index = ShapesIndex::init_ram(&ShapesOptions::default(), &[])?; // Chamfer over f32 points
let pair: [[f32; 3]; 2] = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]];
let triple: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]];
index.insert([Key::from(1u64), Key::from(2u64)], Tape::of([&pair[..], &triple[..]]))?;

let probe: [[f32; 3]; 2] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
let mut arena = Arena::new()?;
let matches = index.search(Tape::of([&probe[..]]), 1, &mut arena)?;
assert_eq!(matches.query(0)?.keys, [Key::from(1u64)]);
```

The metrics are Chamfer, Hausdorff, Gromov-Wasserstein and Wasserstein-Procrustes, over `f16` or `f32` storage, whatever scalar the points arrive in.
A non-zero `points` in `ShapesOptions` freezes every shape's point count.
`get::<Value>` converts each shape to the scalar it is asked for and returns it in its inserted pose, and `count` reports a key's total and largest point counts.

## Geospatial Indexes

A `GeospatialIndex` searches observations on and above the Earth, like photos, video frames, drone footage and sensor fixes.
`GeospatialObservation` is the library's own eleven doubles, so a slice of them crosses with no copy; a field no device recorded stays NaN, which only widens the tests that read it, and `GeospatialObservation::at` leaves every field but the position that way:

```rust
let index = GeospatialIndex::init_ram(&GeospatialOptions::default(), &[])?;
let stops = [
    GeospatialObservation::at(51.5308, -0.1238), // every other field unrecorded, NaN
    GeospatialObservation::at(51.5320, -0.1233),
    GeospatialObservation::at(51.5033, -0.1196),
];
index.insert([Key::from(1u64), Key::from(2u64), Key::from(3u64)], &stops)?;

let mut arena = Arena::new()?;
let nearby = index.within(&stops[..1], 500.0, &mut arena)?;
assert_eq!(nearby.query(0)?.keys.len(), 2); // both stops within 500 metres, in chord metres
```

`within` reports every entry the relation admits inside the radius, however many, while `search` ranks the `wanted` nearest, a zero `metres` ranking by distance alone.
The relation is a field of each call's config: `Near` widens the radius by both positions' accuracies, `Sees` and `Seen` test the view cones, and `Visible` tests a line of sight over the `GeospatialTerrain` the handle was built with.
`get` returns a key's observations within the stored rounding.

## What This Binding Does Not Expose

These C entry points also take a deferred completion, but the crate calls them blocking, whatever receiver you hold:

- The `load_*` family, `view_nvme`, `metadata` and `metadata_buffer`, with each load config's `executor` choosing where it fans out.
- `clear`, which runs on the calling thread under `&mut DenseIndex`.
- Cursor and search-stream pulls, which run on the executor of the receiver that opened them, and open only from a blocking one.

`init_ram`, `init_vram` and `view_ram` block in C as well.

## Design

- __A thin layer over the C ABI.__
  Every name follows `c/usearch/`: a C verb `usearch_dense_<verb>` is a `<verb>` here, each module mirrors one header, a verb's argument struct becomes its `Dense*Config`, and every wrapper's documentation names the C symbol it calls.
  Enums mirror C enums, never strings.
- __The receiver chooses, the verb stays.__
  `&DenseIndex` runs a verb on the calling thread, `on(&executor)` blocks on an executor, and `deferred(&executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls through `usearch_outcome_status`, `wait` blocks through `usearch_outcome_wait`, and a `Deferred` is `IntoFuture`, so `.await` polls the status word, since the library has no wake-up.
- __No life of its own.__
  The binding starts no threads and keeps no queues; releasing a `Deferred` rides `Drop`, which waits for the call first.
- __Arenas are lent and reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes; the result borrows it, so the borrow checker retires stale views at compile time.
- __Nothing costs anything until it is used.__
  Blocking calls allocate no box, and an arena allocates only when a call writes into it.
- __Callers lend memory, never an allocator.__
  No allocation callback crosses into the library: payload belongs to the index, placed by its constructor's residency, and scratch belongs to the executor.
- __A deferred receiver refuses borrowed inputs at compile time.__
  The `Accepts` bound admits only owned inputs, so a borrow the library would read after the verb returns never compiles.
