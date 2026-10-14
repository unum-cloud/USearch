# USearch for JavaScript

This package binds the C ABI in `include/usearch/` through a Node-API addon written in plain C over `<node_api.h>`, with no C++ wrapper in between.
It needs Node.js 22 or newer, ships prebuilt addons for Linux and macOS on x86-64 and Arm64 and for Windows on x86-64, and loads as CommonJS or an ES module, with TypeScript declarations.
It exposes dense indexes; sequence, matrix and geospatial indexes follow as the C library ships them.

## Installation

```sh
npm install usearch
```

### Building From Source

The addon links against the C interface as a static archive, so `usearch.node` carries the engine inside it rather than beside it.
`cmake-js` compiles it, and `prebuild` stages it where `node-gyp-build` looks:

```sh
npm run prebuild
npm run build-js
npm test
```

The addon carries the dense storage types its CMake configure compiled: `npm run build-native` and the prebuilt addons pass `USEARCH_DENSE_DTYPES=11`, while a plain configure compiles `bf16`, `u1x8`, `i8` and `e4m3` alone, and the suite skips the kinds such a build refuses.

## Quickstart

Create an index, insert a vector, and search it on an executor, landing the matches in an arena:

```js
const assert = require('node:assert');
const usearch = require('usearch');

const index = usearch.DenseIndex.initRam({ metric: usearch.DenseMetric.Euclidean, connectivity: 16, dimensions: 3 });
index.insert(42n, new Float32Array([0.2, 0.6, 0.4]));

const executor = usearch.Executor.forRam(); // Absent counts derive the worker count from the hardware
const arena = new usearch.Arena(executor); // Memory where the executor computes, reused by every call
const matches = index.on(executor).search(new Float32Array([0.2, 0.6, 0.4]), 10, arena); // One query, up to ten matches
const found = matches.get(0);

assert(index.describe().size === 1);
assert.deepEqual(found.keys, usearch.keyFromInteger(42n));
assert(found.scores[0] === 0);
```

## Receivers & Deferred Calls

The receiver chooses where a verb runs and how it completes, and the verb stays the same.
The index runs it on the calling thread, `index.on(executor)` blocks on an executor, and `index.deferred(executor)` submits it there and answers a `usearch.Deferred`:

```js
const matches = index.search(queries, 10, arena);                 // the calling thread
const onPool = index.on(executor).search(queries, 10, arena);     // blocking, after every earlier submission
const call = index.deferred(executor).search(queries, 10, arena); // Deferred<Matches>
```

`DenseIndexOn` and `DenseIndexDeferredOn` offer the same verbs with the same arguments.
Only the blocking receivers add `contains`, `cursor` and `searchStream`, whose every pull blocks, and only the index itself adds `attend`.
Constructors address no index yet, so no receiver offers them; the ones the C interface lets fan out take an `executor` in their config.

A `Deferred` keeps everything the library reads or writes alive until its call is final: the outcome, the inputs, the outputs and every handle it names.
It is awaitable, implementing the thenable protocol, so `await` reaches it like any other asynchronous value:

```js
const call = index.deferred(executor).search(queries, 10, arena);
if (call.status === usearch.Status.Deferred) {
  // other work
}
const matches = await call; // or: call.wait()
```

`await` waits on the libuv pool, so the event loop keeps turning, while `wait()` blocks the calling thread.
Both answer the same cached result, or throw the same failure, however often they are asked, and `status` reads the outcome without blocking.
`using call = ...` waits for the call before its block ends, and one dropped unfinished is handed back to the collector until it is final, so nothing it borrowed is freed under the library.
Refusals this binding catches before the call crosses, like keys that do not match the rows, throw at submission.

A submission finding the executor's deferred table full, sized by `deferredLimit`, finishes at once with `Status.Contended`.
Calls depending on each other are ordered by waiting on the first before submitting the second.
The arrays a deferred verb is handed must not change until it is final, since the library reads them until then.

## Executors

A call that names no executor runs on the calling thread, single-threaded on RAM and on the handle's device and default stream on VRAM, and spawns nothing behind the caller's back.
An `Executor` owns the workers instead, and one can be shared by several indexes.
Each residency has its own factory:

```js
const hostPool = usearch.Executor.forRam({ threads: 2, memoryDomains: [0, 1] });
const devicePool = usearch.Executor.forVram({ devices: [0, 1], scratchLimit: 1 << 30 });
const diskPool = usearch.Executor.forNvme({ threads: 4, paths: ['/mnt/nvme0', '/mnt/nvme1'] });
```

`forRam` spawns host worker threads and spreads them over the named NUMA nodes, absent naming none and spreading them over every node.
`onPool` borrows an existing ForkUnion pool by its `BigInt` address instead of spawning one.
`forVram` opens one CUDA stream per named ordinal, absent taking one free device, or borrows the `streams` given as `BigInt` addresses, one per device.
It takes no thread count because one host thread drives one stream per device, so the device list is what sizes the work rather than a number of workers.
`forNvme` opens one submission ring per worker over the named device or directory paths, absent deriving them.
Each factory's config holds only what its own C structure takes: `memoryDomains` for RAM, `devices` for VRAM, `paths` for NVMe, and a `deferredLimit` for the deferred calls it tracks, absent taking a default.

`drain(mood)` waits for every call submitted so far, then idles the workers in a `usearch.Mood`: `Parked` threads sleep, and `Spinning` ones stay hot for the next call's latency.
The collector frees an executor eventually, and `free()` runs what it holds and joins its workers now; a freed executor is refused by every later call.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in an `Arena` you pass, and the verb answers a view over it without copying.
`new Arena(executor)` places it where that executor computes, decided once, and `new Arena()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
A deferred call borrows its arena as a blocking one does: another call naming it before the first is final finishes with `Status.Contended`, and `free()` waits for it.
A rewind or a `free()` detaches every buffer over the previous outputs, so a stale view throws `Status.Uninitialized` and typed arrays taken from it read as empty.
An arena from a VRAM executor holds device memory, which host accessors refuse with `Status.Device`.
It grows, fills and frees on that executor's first stream, so it serves only calls on that executor and keeps it alive until freed, and the executor's `free()` refuses with `Status.Contended` while such an arena lives; a host arena may outlive its executor.

```js
const { Arena, keyAt } = usearch;
using arena = new Arena(executor); // new Arena() is host pages for the calling thread
for (const batch of batches) {
  const matches = index.on(executor).search(batch, 10, arena); // rewinds: last batch's matches are gone
  total += matches.scores.reduce((sum, score) => sum + score, 0);
}

using found = new Arena(), rows = new Arena(); // two live results, two arenas
const nearest = index.search(query, 10, found).get(0);
for (let position = 0; position < nearest.count; position++)
  rerank(keyAt(nearest.keys, position), index.get(keyAt(nearest.keys, position), rows));

const calls = shards.map((shard, position) => shard.deferred(executor).search(queries, 10, arenas[position]));
const results = [];
for (const call of calls) results.push(await call); // or: call.wait()
```

Only fixed-size reports, like `describe` and the telemetry readings, come back as plain objects.

## Keys

A key is __sixteen opaque bytes__, most significant first, and that is the only form a full 128-bit key survives.
Matches carry them as one `Uint8Array` of sixteen bytes per match, which is byte-for-byte what the engine writes, so a batch of results needs no conversion pass and no object per match.
The canonical rendering is the lowercase dashed UUID the ABI fixes, and that is a rendering rather than a second carrier.

```js
usearch.keyFromInteger(42n);                 // Uint8Array(16), 42 in the trailing byte
usearch.formatKey(usearch.keyFromInteger(42n)); // '00000000-0000-0000-0000-00000000002a'
usearch.parseKey('f81d4fae-7dec-11d0-a765-00a0c91e6bf6'); // Uint8Array(16)
usearch.keyAt(matches.keys, 0);                           // the first key, as a view rather than a copy
usearch.integerFromKey(usearch.keyAt(matches.keys, 0)); // its integer, when it was inserted as one
```

An integer key parks in the trailing eight bytes with the leading eight zeroed, so `42n` prints as `00000000-0000-0000-0000-00000000002a`.
`integerFromKey` throws on a key whose leading bytes carry anything, since no integer spells it.

Keys are accepted in every one of those spellings, plus a `BigUint64Array` of integer keys, which stays the convenient path for services that number their rows:

```js
index.insert(42n, vector);                                     // a BigInt
index.insert(new BigUint64Array([15n, 16n]), vectors);         // a run of integers
index.insert('f81d4fae-7dec-11d0-a765-00a0c91e6bf6', vector);  // a dashed UUID
index.insert(usearch.parseKey('f81d4fae-7dec-11d0-a765-00a0c91e6bf6'), vector); // bytes
```

A single key broadcasts across every row of a batch, and a `Uint8Array` of keys crosses without a copy.

## Errors

Failures throw an `Error` whose `code` is a `usearch.Status`, one member per status of the C interface, spelled as its canonical name such as `'shape-mismatch'`.
Refusals this binding raises before a call crosses carry the same codes, so a caller branches on the fault rather than on which side of the boundary saw it.
The engine checks every row's width against the geometry it froze, so a narrow row surfaces as its `ShapeMismatch`:

```js
try {
  index.insert(43n, new Float32Array([0.2, 0.6]));
} catch (error) {
  assert(error.code === usearch.Status.ShapeMismatch);
}
```

TypeScript callers can narrow a caught value to the exported `USearchError` type, an `Error` with that `code`.

## Residency

An executor and an index are each built through the residency they drive, so a direct `new usearch.Executor()` or `new usearch.DenseIndex()` throws and names the factories instead.
An index runs its verbs only on an executor of its own residency, and an arena reports the residency it was made for as `arena.residency`.

```js
const hostIndex = usearch.DenseIndex.initRam({ dimensions: 128 }, [0, 1]);
const deviceIndex = usearch.DenseIndex.initVram({ dimensions: 128 }, [0, 1]);
const diskIndex = usearch.DenseIndex.viewNvme('index.safegraphs.json', { executor: diskPool, paths: ['/mnt/nvme0', '/mnt/nvme1'] });
```

`initRam` shards over the named NUMA nodes, absent spreading the shards over every node, and `initVram` over the named CUDA ordinals, absent taking one free device.
An NVMe index is never initialized, only served in place from a checkpoint, read-only.
An index's `free()` waits for every deferred call naming it, and a freed index refuses every later call.

## Serialization

Persist and restore an index through the filesystem, or through memory when checkpoints live in an object store.
Loading creates a new index, its geometry read from the checkpoint, so the verb names where it lands:

```js
index.on(executor).save('index.safegraphs');
const loaded = usearch.DenseIndex.loadRam('index.safegraphs', { executor, memoryDomains: [0, 1] });
const onDevice = usearch.DenseIndex.loadVram('index.safegraphs', { executor: devicePool, devices: [0] });
const viewed = usearch.DenseIndex.viewRam('index.safegraphs'); // Read-only, served from a memory mapping

const monolith = index.on(executor).saveBuffer(arena); // A Monolith viewing the arena
const restored = usearch.DenseIndex.loadRamBuffer(monolith, { executor });
```

`valuesPerKey` bounds a live handle rather than the file, so a load takes it among its config.

Read a checkpoint's geometry before deciding how to open it:

```js
const metadata = usearch.DenseIndex.metadataRead('index.safegraphs', { executor });
const fromMemory = usearch.DenseIndex.metadataReadBuffer(monolith, { executor });
```

## Dense Indexes

A dense index holds equidimensional embeddings.
Only `dimensions` is required, and every other field derives from the geometry when it is left out:

```js
const index = usearch.DenseIndex.initRam(
  {
    dimensions: 128,
    metric: usearch.DenseMetric.Inner, // or Angular, the default, Euclidean, Hamming, Jaccard
    quantization: usearch.Scalar.BF16, // the default, or F16, E5M2, E4M3, E3M2, E2M3, I8, I4x2, U8, U4x2, U1x8
    connectivity: 32, // a power of two; absent derives it from the geometry
    valuesPerKey: 2, // absent derives it to unique keys
  },
  [0, 1], // absent spreads the shards over every NUMA node
);
```

### Rows & Scalars

A bare `TypedArray` is one row, and a `RowsView` is a run of them borrowed in place.
It names the array, the row count, the columns, the bytes between rows, and the scalar kind, so a memory-mapped dataset reaches the index with no reshape and no copy.
`RowsView.of` reads a packed array as rows of the width it is given:

```js
const keys = new BigUint64Array([15n, 16n]);
const vectors = usearch.RowsView.of(new Float32Array([10, 20, 10, 25]), 2); // two rows of two
index.insert(keys, vectors);

const padded = new usearch.RowsView(mapped, rows, 768, 1024 * 4, usearch.Scalar.F32); // a pitch of 1024 floats
const matches = index.search(vectors, 2, arena);
```

The engine converts rows on ingest, so what a caller holds is independent of what the index stores.
A `Float64Array`, `Float32Array`, `Float16Array`, `Int8Array` or `Uint8Array` names its own scalar kind; every other kind travels as bytes in a `RowsView` naming what they mean:

```js
const bits = usearch.DenseIndex.initRam({ metric: usearch.DenseMetric.Hamming, dimensions: 64, quantization: usearch.Scalar.U1x8 });
const packed = usearch.RowsView.of(new Uint8Array([0b10101010, 0, 0, 0, 0, 0, 0, 0]), 64, usearch.Scalar.U1x8);
bits.insert(1n, packed);
bits.search(packed, 2, arena);

const halves = usearch.RowsView.of(new Uint16Array([0x3f80, 0x3f80, 0x3f80, 0x3f80]), 4, usearch.Scalar.BF16);
```

A row run claiming more bytes than its array holds is refused before it crosses.

### Search

`search` and `maxsim` land their matches in the arena as `Matches`, three views: the keys, the scores, and how many matches every query actually found.

```js
const matches = index.search(queries, 10, arena);

matches.queries;   // Rows in the run, one per query
matches.columns;   // Columns per row the call wrote, at most its 'k'
matches.counts;    // BigUint64Array, the matches every query found
matches.keys;      // Uint8Array, sixteen bytes per cell, rows laid end to end
matches.scores;    // Float64Array, one distance per cell

const found = matches.get(0); // A MatchesRow, borrowed out of the run rather than copied
found.count;                  // Matches this query found, never above `wanted`
found.keys;                   // Sixteen bytes per match found
found.scores;                 // One distance per match found
```

Every view reads the arena's own memory rather than a copy, so reading a row costs nothing.
Scores come back as a `Float64Array`, which is the width the ABI writes.

The candidate-pool breadth rides every call rather than the construction, so a recall sweep runs against one immutable index with no mutation between points:

```js
index.on(executor).insert(keys, vectors, { expansionFactor: 128 });
index.on(executor).search(queries, 10, arena, { expansionFactor: 64 });
index.on(executor).search(queries, 10, arena, { traversal: usearch.Traversal.Scan }); // exact, on RAM and NVMe handles
```

`insert`, `refine`, `search`, `maxsim`, `multiply` and `attend` each report how much graph they walked, into a `DenseTraversalStats` reading passed through the config:

```js
const stats = new usearch.DenseTraversalStats();
index.on(executor).search(queries, 10, arena, { stats });

stats.visitedMembers;     // entries whose neighborhoods the walk expanded
stats.computedDistances;  // distance evaluations the walk paid for
stats.restarts;           // times the walk fell back to a fresh entry point

const total = new usearch.DenseTraversalStats();  // sweeping, and reading the whole sweep's cost
for (const expansionFactor of [16, 32, 64, 128]) {
  index.on(executor).search(queries, 10, arena, { expansionFactor, stats });
  total.accumulate(stats);
}
```

Telemetry travels through the reading rather than the return value, so `refine` still answers with the nodes it changed and a ranked verb with its matches, and passing none costs nothing.
A reading is a pure output that each call fully overwrites, so one `DenseTraversalStats` handed to several calls holds the last of them, and a deferred call writes it once it settles.
Use `accumulate` to fold readings together: `expansionFactor` buys recall in distance computations, and only the counters say how many, so a sweep reusing one arena allocates once and reads its cost between the points.

### Filtering & Progress

A search stream pulls the next-nearest matches on the caller's own thread, which is how a JavaScript predicate filters without reserving a run wide enough to survive it:

```js
using stream = index.on(executor).searchStream(query);
const admitted = [];
for (let pulled = stream.next(16, arena); pulled.counts[0] && admitted.length < 10; pulled = stream.next(16, arena)) {
  const { keys, count } = pulled.get(0);
  for (let position = 0; position < count; position++) {
    const key = usearch.keyAt(keys, position).slice(); // A copy outlives the next pull, a view does not
    if (isVisibleToTenant(key)) admitted.push(key);
}
}
```

The engine-side predicate and the progress handlers are not exposed, for the reason under [What This Binding Does Not Expose](#what-this-binding-does-not-expose).

### Retrieval: Get, Cursors & Streams

`get` exports the rows stored under one key into the arena, as `f32` unless `scalar` names another kind:

```js
const stored = index.get(42n, arena); // The vectors under a key, as f32 DenseVectors
using bitsArena = new usearch.Arena();
const packed = index.get(42n, bitsArena, { scalar: usearch.Scalar.U1x8 }); // Or as packed bits, in another arena
```

`get` shapes its rows from the width and stride the call reports: `stored.data` holds `stored.rows` rows of `stored.columns` values, `stored.stride` bytes apart, and no rows when the key is absent.

A cursor walks the whole index instead, in storage order; each shard's watermark is read as the walk enters it, so a row once passed is never revisited.
It lands each batch of up to `wanted` entries in the arena, the keys always and the rows beside them in the kind `vectors` names:

```js
using cursor = index.on(executor).cursor();
for (let batch = cursor.next(1024, arena, { vectors: usearch.Scalar.F32 }); batch.rows; batch = cursor.next(1024, arena, { vectors: usearch.Scalar.F32 })) {
  writeOut(batch.keys, batch.vectors.data, batch.rows);
}
```

Cursors and streams keep their walk state rather than a thread: every pull is an ordinary call on the executor of the receiver that opened them, and each keeps its index alive until freed.
No residency implements the stream yet, so the engine answers `Status.Unsupported`, and a stream reports no telemetry, since the ABI gives a resumable walk none.

### Refinement

A cheap insertion pass followed by one refinement reaches a better graph, in less total time, than a single expensive insertion pass:

```js
const pool = index.on(executor);
pool.insert(keys, vectors, { expansionFactor: 32 });
while (pool.refine({ expansionFactor: 128 }) > vectors.rows / 100) {}
```

Refinement is a build step rather than a repair.
It adds and removes nothing, and searches may run throughout.

### Absorbing & Splitting

Bigger indexes come from absorbing, never mutating: `absorb` appends every donor in turn, keeping its graph and linking it in across the appended part, and a donor the library consumed refuses every later verb with `Status.Uninitialized`.
A donor it did not absorb, like one of another scalar refused with `Status.ConfigMismatch`, stays a live index you still own, though its rows may already have landed.
A deferred absorb holds its donors until it is final, so meanwhile they refuse every verb and no other absorb can name them.

```js
index.on(executor).absorb([shard]);
```

### Sparse Multiplication

`multiply` is the sparse sibling of `search`: every left row emits one pair per column the walk scored rather than one ranked match, so the scores are unnormalized logits and the index must hold the `Inner` kind.
How many pairs a walk scores is known only once it ends, so the call answers with a `DenseProducts` run viewing its arena, laid out with Apache Arrow offsets:

```js
const index = usearch.DenseIndex.initRam({ metric: usearch.DenseMetric.Inner, dimensions: 128 });
const products = index.on(executor).multiply(left, arena, { wanted: 64 }); // Up to 64 pairs per left row
for (let row = 0; row < products.rows; row++) {
  const pairs = products.get(row); // A MatchesRow over `offsets[row]` up to `offsets[row + 1]`
  for (let pair = 0; pair < pairs.count; pair++) accumulate(row, usearch.keyAt(pairs.keys, pair), pairs.scores[pair]);
}
products.required; // Pairs scored across every row, before `wanted` cut any
```

Pairs arrive in traversal order rather than sorted, and a `wanted` bound keeps the first ones each walk scored; leaving it out keeps every pair.
The Arrow `offsets`, one entry longer than the left rows, land in the arena beside the pairs.

### Late Interaction

An index whose `valuesPerKey` is above one holds several token rows per document, and `maxsim` scores documents rather than rows:

```js
const index = usearch.DenseIndex.initRam({ metric: usearch.DenseMetric.Inner, dimensions: 128, valuesPerKey: 64 });
index.on(executor).insert(documentKeys, tokenRows);

// 32 padded token rows per query, one run row per query
const documents = index.on(executor).maxsim(queryTokens, 32, 10, arena);
```

### Attention

`attend` is softmax attention over every key a query's graph walk scores, folded as Flash Attention folds key tiles, with no top-k cut: the long tail the walk scores is what brings it toward exact dense attention.
It lands one key-wide row per query in the arena, in the queries' own kind unless `scalar` names another.
The index stores no values: every key a walk scores, 16 bytes apiece, resolves through `rowOf` to a row of `values`, borrowed for the call rather than copied, as wide as the keys and stored in the index's own scalar kind.
Answering with no row of `values`, or lending values of another width or kind, refuses the call with `ShapeMismatch`, since a missing row is a wrong softmax rather than a smaller one.

```js
// One value row per indexed key, stored like the keys
const values = usearch.RowsView.of(valueMatrix, dimensions);
// The row of `values` a 16-byte key resolves to
const rowOf = (key) => tokenRows.get(usearch.integerFromKey(key));
// A largest logit and a partition per query row, lent and updated in place
const carry = new Float32Array(queries.rows * 2);
// Row `t * 8 + h` is head `h` of token `t`
const outputs = index.attend(queries, values, rowOf, arena, { headCount: 8, carry, scoreScale });
// One key-wide row per query, `outputs.stride` bytes apart
outputs.data;
```

One index holds one layer's one key-value head, so the `headCount` query heads of grouped-query attention share its keys and one walk per token.
The first exception `rowOf` throws stops every later lookup and surfaces as itself.
`rowOf` runs on the host, so `attend` serves RAM handles alone: a VRAM walk resolves rows through a device function, which this binding does not build, and a VRAM arena refuses with `Status.Device`.
The carry is what folds one branch into another over the same queries, by the usual partial-softmax rescaling: attention sinks, a dense recent window, or a second index of one context.
Its two floats travel in one row because a maximum without its partition is a number nothing can rescale, and omitting it altogether starts the softmax fresh.
`scoreScale` multiplies every logit and derives from the geometry when left out, and a wider `expansionFactor` walks further, folding more of the tail.

### Introspection

```js
const { metadata, size, capacity, memoryUsage, capabilities } = index.describe();
metadata.dimensions;                                   // The frozen geometry, as a checkpoint reports it
const containsKey = index.contains(42n);               // Whether a key is in the index
const count = index.count(42n);                        // The number of vectors under a key
const { audited, reachable } = index.health();        // How much of the graph a search reaches

usearch.version();
usearch.capabilitiesCompiled();
usearch.capabilitiesAvailable();
```

`health` audits from the search landmarks, and entries it misses stay unfound until a `refine` re-links them; its `sweeps` counts the levels the audit took to converge.

## What This Binding Does Not Expose

A JavaScript callback runs only on the thread owning the JavaScript context, and an executor's threads never are.
A thread-safe function would queue onto a loop that sits blocked inside the verb, so the callback could never run.
That is why `attend` runs on the calling thread alone, where its `rowOf` can, and no executor receiver offers it.
It is also why a search `predicate`, a per-candidate callback the engine calls from the threads it fanned out over, is not exposed at all; pull through `searchStream` instead and filter in JavaScript.
The same reasoning excludes the progress handlers every long verb takes.

## Design

- __A thin layer over the C ABI.__
  Every name follows `include/usearch/`: a C verb `usearch_dense_<verb>` is a camel-cased `<verb>` here, every type is its C name without `usearch_` and `_t`, and a verb's argument structure becomes its `Dense*Config`.
  Enums mirror C enums through the C name tables, so a misspelling is refused rather than read as a default.
- __The receiver chooses, the verb stays.__
  The index runs a verb on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls through `usearch_outcome_status`, `wait()` blocks through `usearch_outcome_wait`, and `await` hands that same wait to the libuv pool through `napi_async_work`, since the library has no wake-up to register.
- __No life of its own.__
  The binding starts no threads and keeps no queues; releasing a `Deferred` rides `using` and `Symbol.dispose`, which waits for the call first.
- __Arenas are lent and reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes; a rewind detaches the buffers under the previous call's views, so a stale one throws rather than reading another call's memory.
- __Nothing costs anything until it is used.__
  Blocking calls take no deferred-table slot, and an arena allocates only when a call writes into it.
- __Every wrapper names its C symbol.__
  Each wrapper's documentation names the C function it calls, or says it is binding-only, as `contains`, `keyAt` and `RowsView.of` are.
