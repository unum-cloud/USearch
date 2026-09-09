# USearch for JavaScript

USearch is a high-performance library for building and querying vector search indexes, optimized for Node.js and WASM environments.

## Installation

For Node.js environments, install USearch using `npm`:

```sh
npm install usearch
```

## Quickstart

Create an index, add vectors, reserve the run the matches land in, and search.
Every batch verb fans out over an `Executor`, which owns the workers and can be shared by several indexes:

```js
const assert = require('node:assert');
const usearch = require('usearch');

const executor = usearch.Executor.forRam(0); // Zero derives the worker count from the hardware
const index = usearch.DenseIndex.forRam({ metric: 'euclidean', connectivity: 16, dimensions: 3 });
index.add(42n, new Float32Array([0.2, 0.6, 0.4]), executor);

const matches = usearch.MatchesSpan.reserve(1, 10); // One query, up to ten matches apiece
index.search(new Float32Array([0.2, 0.6, 0.4]), matches, executor);
const found = matches.get(0);

assert(index.size() === 1);
assert.deepEqual(found.keys, usearch.keyFromInteger(42n));
assert(found.distances[0] === 0);
```

An executor is required rather than derived.
A pool spawned behind the caller's back claims most of the machine's cores invisibly, and nothing could ever free it again.

## The Three Residencies

An executor and an index are each built through the residency they drive, so a direct `new usearch.Executor()` or `new usearch.DenseIndex()` throws and names the three factories instead.

```js
const hostPool = usearch.Executor.forRam(2, { memoryDomains: [0, 1] });
const hostIndex = usearch.DenseIndex.forRam({ dimensions: 128, memoryDomains: [0, 1] });

const devicePool = usearch.Executor.forVram({ devices: [0, 1] });
const deviceIndex = usearch.DenseIndex.forVram({ dimensions: 128, devices: [0, 1] });

const diskPool = usearch.Executor.forNvme(4, { paths: ['/mnt/nvme0', '/mnt/nvme1'] });
const diskIndex = usearch.DenseIndex.forNvme({ dimensions: 128, paths: ['/mnt/nvme0', '/mnt/nvme1'] });
```

`forRam` spawns host worker threads and spreads them over the named NUMA nodes, absent naming none and spreading them over every node.
`forVram` opens one CUDA stream per named ordinal, absent taking every visible device.
`forNvme` opens one submission ring per worker over the named device or directory paths, absent deriving them.

`forVram` takes no thread count because one host thread drives one stream per device, so the device list is what sizes the work rather than a number of workers.
Each arm belongs to one residency alone — `forRam` reads `memoryDomains`, `forVram` reads `devices`, `forNvme` reads `paths` — and naming another residency's arm is refused rather than ignored.

## Ranked Results

`search`, `multiply` and `maxsim` write into a `MatchesSpan` the caller lends, and answer with nothing at all.
A run is three buffers — the identities, the distances, and how many matches every query actually found — and `reserve` allocates all three at once:

```js
const matches = usearch.MatchesSpan.reserve(queryRows, 10);
index.search(queries, matches, executor);

matches.queries;   // Rows in the run, one per query
matches.wanted;    // Columns per row, the 'k' in 'kANN'
matches.counts;    // BigUint64Array, the matches every query found
matches.keys;      // Uint8Array, sixteen bytes per cell, rows laid end to end
matches.distances; // Float64Array, one score per cell

const found = matches.get(0); // A MatchesRow, borrowed out of the run rather than copied
found.count;                  // Matches this query found, never above `wanted`
found.keys;                   // Sixteen bytes per match found
found.distances;              // One score per match found
```

No ranked verb takes a `k`.
The matches wanted per query is derived from the buffer lengths rather than asked for, which is what keeps the space reserved and the width asked for from disagreeing.
Three buffers a caller already holds make a run directly, and lengths that read as no single shape are refused there and then:

```js
const matches = new usearch.MatchesSpan(keys, distances, counts);
```

`get` views the run's own storage rather than copying it, so reading a row costs nothing, and the next call over the same run overwrites what an earlier row still views.
Reusing one run is the point of lending it: a recall sweep allocates once, searches at every breadth, and reads its counters between the points.

## Identities

An identity is __sixteen opaque bytes__, most significant first, and that is the only form a full 128-bit key survives.
Matches carry them as one `Uint8Array` of sixteen bytes per match, which is byte-for-byte what the engine writes — so a batch of results needs no conversion pass and no object per match.
The canonical rendering is the lowercase dashed UUID the ABI fixes, and that is a rendering rather than a second carrier.

```js
usearch.keyFromInteger(42n);                 // Uint8Array(16), 42 in the trailing byte
usearch.formatKey(usearch.keyFromInteger(42n)); // '00000000-0000-0000-0000-00000000002a'
usearch.parseKey('f81d4fae-7dec-11d0-a765-00a0c91e6bf6'); // Uint8Array(16)
usearch.keyAt(matches.keys, 0);              // the first identity, as a view rather than a copy
usearch.integerFromKey(usearch.keyAt(matches.keys, 0)); // its integer, when it was inserted as one
```

An integer identity parks in the trailing eight bytes with the leading eight zeroed, so `42n` prints as `00000000-0000-0000-0000-00000000002a`.
`integerFromKey` throws on an identity whose leading bytes carry anything, since no integer spells it.

Keys are accepted in every one of those spellings, plus a `BigUint64Array` of integer identities, which stays the convenient path for services that number their rows:

```js
index.add(42n, vector, executor);                                     // a BigInt
index.add(new BigUint64Array([15n, 16n]), vectors, executor);         // a run of integers
index.add('f81d4fae-7dec-11d0-a765-00a0c91e6bf6', vector, executor);  // a dashed UUID
index.add(usearch.parseKey('f81d4fae-7dec-11d0-a765-00a0c91e6bf6'), vector, executor); // bytes
```

## Scalar Kinds

The engine converts rows on ingest, so what a caller holds is independent of what the index stores.
A `Float64Array`, `Float32Array`, `Int8Array` or `Uint8Array` names its own scalar kind; every other kind travels as bytes with `scalarKind` naming what they mean:

```js
const index = usearch.DenseIndex.forRam({ metric: 'hamming', dimensions: 64, quantization: 'u1x8' });
const packed = new Uint8Array([0b10101010, 0, 0, 0, 0, 0, 0, 0]);
index.add(1n, packed, executor, { scalarKind: usearch.ScalarKind.U1x8 });

const matches = usearch.MatchesSpan.reserve(1, 2);
index.search(packed, matches, executor, { scalarKind: usearch.ScalarKind.U1x8 });
```

Distances come back as a `Float64Array`, which is the width the ABI writes.

## Serialization

Persist and restore an index through the filesystem, or through memory when checkpoints live in an object store:

```js
index.save('index.safegraphs', executor);
index.load('index.safegraphs', executor);

const monolith = index.saveBuffer(executor); // Uint8Array of `index.storageRequirements()` bytes
restored.loadBuffer(monolith, executor);
```

Read a checkpoint's geometry before deciding how to open it:

```js
const metadata = usearch.DenseIndex.metadata('index.safegraphs');
const fromMemory = usearch.DenseIndex.metadataBuffer(monolith);
```

## Advanced Index Configuration

Only `dimensions` is required, and every other field derives from the geometry when it is left out:

```js
const index = usearch.DenseIndex.forRam({
  dimensions: 128,
  metric: 'inner', // or 'angular', 'euclidean', 'hamming', 'jaccard'
  quantization: 'f32', // or 'f64', 'bf16', 'f16', 'e5m2', 'e4m3', 'e3m2', 'e2m3', 'i8', 'i4x2', 'u8', 'u4x2', 'u1x8'
  connectivity: 10, // 0 derives it from the geometry
  valuesPerKey: 2, // 0 derives it to unique keys
  memoryDomains: [0, 1] // absent spreads the shards over every NUMA node
});
```

The candidate-pool breadth rides every call rather than the construction, so a recall sweep runs against one immutable index with no mutation between points:

```js
index.add(keys, vectors, executor, { expansionFactor: 128 });
index.search(queries, matches, executor, { expansionFactor: 64 });
```

## Batch Operations

Process multiple vectors at once.
A flattened `TypedArray` is preferred over an array of arrays, which is laid end to end before it crosses:

```js
const keys = new BigUint64Array([15n, 16n]);
const vectors = new Float32Array([10, 20, 10, 25]);
index.add(keys, vectors, executor);

const matches = usearch.MatchesSpan.reserve(2, 2);
index.search(vectors, matches, executor);
const firstRow = matches.get(0);
executor.drain();
```

## Refinement

A cheap insertion pass followed by one refinement reaches a better graph, in less total time, than a single expensive insertion pass:

```js
index.add(keys, vectors, executor, { expansionFactor: 32 });
while (index.refine(executor, { expansionFactor: 128 }) > vectors.length / 100) {}
```

Refinement is a build step rather than a repair.
It adds and removes nothing, and searches may run throughout.

## Traversal Telemetry

`add`, `refine`, `search`, `maxsim`, `multiply` and `attend` each report how much graph they walked, into a `Stats` accumulator passed through the options:

```js
const stats = new usearch.Stats();
index.add(keys, vectors, executor, { stats });
index.search(queries, matches, executor, { stats });

stats.visitedMembers;     // entries whose neighborhoods the walk expanded
stats.computedDistances;  // distance evaluations the walk paid for
stats.restarts;           // times the walk fell back to a fresh entry point
```

Telemetry travels through the accumulator rather than the return value, so `add` still answers with the rows it published and a ranked verb still answers with nothing, and passing no accumulator costs nothing.
One `Stats` handed to several calls sums them, which is what makes a recall sweep's cost readable in distance computations rather than in wall-clock.
`accumulate` folds one reading into another when the sums are gathered afterwards instead.

Telemetry is what a breadth sweep is read against: `expansionFactor` buys recall in distance computations, and only the counters say how many.
A stream opened by `find` reports none, since the ABI gives a resumable walk no telemetry.

## Late Interaction

An index whose `valuesPerKey` is above one holds several token rows per document, and `maxsim` scores documents rather than rows:

```js
const index = usearch.DenseIndex.forRam({ metric: 'inner', dimensions: 128, valuesPerKey: 64 });
index.add(documentKeys, tokenRows, executor);

const documents = usearch.MatchesSpan.reserve(queryCount, 10); // One row per query, not per token row
index.maxsim(queryTokens, 32, documents, executor); // 32 padded token rows per query
```

## Linear Algebra

`multiply` is the sparse sibling of `search`: every left row emits one pair per column the walk scored rather than one ranked match, so the scores are unnormalized logits and the index must hold the `inner` kind.

```js
const index = usearch.DenseIndex.forRam({ metric: 'inner', dimensions: 128 });
const products = usearch.MatchesSpan.reserve(leftRows, 64); // Up to 64 pairs per left row
index.multiply(left, products, executor);
for (let row = 0; row < products.queries; row++) {
  const emitted = Math.min(Number(products.counts[row]), products.wanted);
  for (let pair = 0; pair < emitted; pair++)
    accumulate(row, usearch.keyAt(products.keys, row * products.wanted + pair), products.distances[row * products.wanted + pair]);
}
```

Pairs arrive in traversal order rather than sorted, and each count reports the true number of pairs scored rather than the number written — a count above `wanted` means that row was truncated, so reserve a wider run and call again.

`attend` folds Flash Attention's online softmax into the same walk, writing into the `outputs` the caller lends and answering with nothing, so a sweep allocates once.
An entry's key is its row offset into the value table, and the value rows carry their own width, which need not be the index's.

```js
const outputs = new Float32Array(queryRows * valueDimensions);
const carry = new Float32Array(queryRows * 2); // A running maximum and a partition per query row
index.attend(queries, values, valueDimensions, outputs, executor, { carry, scoreScale, massTolerance });
```

The carry is what folds one branch into another over the same queries — a dense recent window, or a second shard of one context — by the usual partial-softmax rescaling.
Its two floats travel in one row because a maximum without its partition is a number nothing can rescale, and omitting it altogether starts the softmax fresh.
`scoreScale` divides every logit before it is exponentiated and `massTolerance` ends a walk once the residual probability mass falls below it; both derive from the geometry when left out.

## Streaming and Filtering

A finder pulls the next-nearest matches on the caller's own thread, which is how a JavaScript predicate filters without reserving a run wide enough to survive it:

```js
const finder = index.find(query, executor);
const admitted = [];
while (admitted.length < 10) {
  const pulled = finder.next(16);
  if (!pulled) break;
  for (let position = 0; position < pulled.count; position++) {
    const key = usearch.keyAt(pulled.keys, position);
    if (isVisibleToTenant(key)) admitted.push(key);
  }
}
finder.free();
```

A live finder pins one of the executor's threads, so free it the moment pulling stops.

A cursor walks the whole index instead, over a snapshot taken when it was opened, so entries inserted afterwards never appear mid-iteration:

```js
const cursor = index.cursor();
for (let batch = cursor.next(1024); batch; batch = cursor.next(1024)) {
  writeOut(batch.keys, batch.vectors);
}
cursor.free();
```

## Index Introspection

```js
const dimensions = index.dimensions();  // The number of dimensions
const containsKey = index.contains(42n); // Whether a key is in the index
const count = index.count(42n);          // The number of vectors under a key
const stored = index.get(42n);           // Those vectors, as f32
const bytes = index.memoryUsage();       // What the index occupies in memory
const kernels = index.capabilities();    // The kernel chain this machine dispatched to

usearch.version();
usearch.capabilitiesCompiled();
usearch.capabilitiesAvailable();
```

A key spelled on its own answers with a scalar; a run of keys answers with one entry apiece, in the order they were given.

## What This Binding Does Not Expose

`searchIf` takes a per-candidate predicate the engine calls from the threads it fanned out over, while the thread owning the JavaScript context sits blocked inside the verb.
A thread-safe function would queue onto that same blocked loop, so the predicate can never run.
Pull through `find` instead and filter in JavaScript.
The same reasoning excludes the progress handlers every long verb takes.

`merge` stays unexposed while its consumption semantics are still moving.

## Building From Source

The addon links against `libusearch_c`, which CMake drops in the build tree:

```sh
npm_config_usearch_library_dir=build/Release npx node-gyp rebuild
npm run build-js
npm test
```

`npm_config_usearch_library_dir` also points at `build_stub/stub`, the brute-force stand-in for the C interface, which runs the suite in seconds without building the engine.
