# USearch for JavaScript

USearch is a high-performance library for building and querying vector search indexes, optimized for Node.js and WASM environments.

## Installation

For Node.js environments, install USearch using `npm`:

```sh
npm install usearch
```

## Quickstart

Create an index, add vectors, and perform searches with ease:

```js
const assert = require('node:assert');
const usearch = require('usearch');
const index = new usearch.Index({ metric: 'l2sq', connectivity: 16, dimensions: 3 });
index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
const results = index.search(new Float32Array([0.2, 0.6, 0.4]), 10);

assert(index.size() === 1);
assert.deepEqual(results.keys, new BigUint64Array([42n]));
assert.deepEqual(results.distances, new Float32Array([0]));

index.remove(42n);
```

Note: keys are 64-bit integers represented as JavaScript `BigInt` (e.g., `42n`).

## Serialization

Persist and restore your index with serialization methods:

```js
index.save('index.usearch'); // Save the index to a file
index.load('index.usearch'); // Load the index from a file
index.view('index.usearch'); // View the index from a file without loading into memory
```

## Advanced Index Configuration

Customize your index with additional configuration options:

```js
const index = new usearch.Index({
  dimensions: 128,
  metric: 'ip',
  quantization: 'f32',
  connectivity: 10,
  expansion_add: 5,
  expansion_search: 3,
  multi: true
});
```

## Batch Operations

Process multiple vectors at once for efficiency.
For performance reasons we prefer a flattened `TypedArray` over an Array of Arrays:

```js
const keys = new BigUint64Array([15n, 16n]);
const vectors = new Float32Array([10, 20, 10, 25]);
index.add(keys, vectors);
```

Retrieve batch search results:

```js
const batchResults = index.search(vectors, 2);
const firstMatch = batchResults.get(0);
```

Multi-threading is supported for batch operations:

```js
const threads_count = 0; // Zero for auto-detection or pass an unsigned integer
index.add(keys, vectors, threads_count);
const batchResults = index.search(vectors, 2, threads_count);
```

## Index Introspection

Inspect and interact with the index:

```js
const dimensions = index.dimensions(); // Get the number of dimensions
const containsKey = index.contains(42n); // Check if a key is in the index
const count = index.count(42n); // Get the count of vectors for a key
```
