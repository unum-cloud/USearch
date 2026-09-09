const nodeTest = require("node:test");
const realTest = nodeTest.test; // the original function

function loggedTest(name, options, fn) {
  // The API has two call signatures:
  //   test(name, fn)
  //   test(name, options, fn)
  if (typeof options === "function") {
    fn = options;
    options = undefined;
  }

  // Wrap the body so we can log before / after
  const wrapped = async (t) => {
    console.log("▶", name);
    try {
      await fn(t); // run the user’s test
      console.log("✓", name);
    } catch (err) {
      console.log("✖", name);
      throw err; // re-throw so the runner records the failure
    }
  };

  // Delegate back to the real test() with the same options
  return options ? realTest(name, options, wrapped) : realTest(name, wrapped);
}

// Replace both the export and the global the runner puts on each module
global.test = loggedTest;
module.exports = loggedTest; // for completeness if this file is `require`d

const assert = require("node:assert");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const usearch = require("./dist/cjs/usearch.js");

function assertAlmostEqual(actual, expected, tolerance = 1e-6) {
  const lowerBound = expected - tolerance;
  const upperBound = expected + tolerance;
  assert(
    actual >= lowerBound && actual <= upperBound,
    `Expected ${actual} to be almost equal to ${expected}`,
  );
}

// The byte run several integer identities spell, laid end to end.
function keysFromIntegers(...integers) {
  const run = new Uint8Array(integers.length * usearch.keyBytes);
  for (let position = 0; position < integers.length; position++)
    run.set(usearch.keyFromInteger(integers[position]), position * usearch.keyBytes);
  return run;
}

const EXECUTOR = usearch.Executor.forRam(2);

test("Residency factories", async (t) => {
  await t.test("a direct construction is refused, and names the factories", () => {
    assert.throws(
      () => new usearch.Executor(),
      /Executor\.forRam/,
      "the message points the caller at the three executor factories",
    );
    assert.throws(
      () => new usearch.DenseIndex(),
      /DenseIndex\.forRam/,
      "the message points the caller at the three index factories",
    );
  });

  await t.test("an arm belonging to another residency is refused", () => {
    assert.throws(() =>
      usearch.DenseIndex.forRam({ dimensions: 3, devices: [0] }),
    );
  });

  await t.test("a thread count that is not a whole number at or above zero is refused", () => {
    assert.throws(() => usearch.Executor.forRam(-1));
    assert.throws(() => usearch.Executor.forRam(NaN));
    assert.throws(() => usearch.Executor.forRam("2"));
  });
});

test("Identities", async (t) => {
  await t.test("an integer parks in the trailing eight bytes, most significant first", () => {
    const key = usearch.keyFromInteger(42n);

    assert.equal(key.length, 16, "a key is sixteen bytes");
    assert.deepEqual(
      key,
      new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42]),
      "42 lands in the last byte, and nowhere else",
    );
    assert.equal(key[15], 42, "the least significant byte is the last one");
    assert.equal(key[0], 0, "the leading bytes stay zero");
  });

  await t.test("a multi-byte integer keeps its byte order", () => {
    const key = usearch.keyFromInteger(0x0102030405060708n);

    assert.deepEqual(
      key.subarray(8),
      new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]),
      "the most significant byte sits at offset 8",
    );
    assert.equal(usearch.integerFromKey(key), 0x0102030405060708n);
  });

  await t.test("keys render and parse as lowercase dashed UUIDs", () => {
    assert.equal(
      usearch.formatKey(usearch.keyFromInteger(42n)),
      "00000000-0000-0000-0000-00000000002a",
    );
    assert.deepEqual(
      usearch.parseKey("00000000-0000-0000-0000-00000000002a"),
      usearch.keyFromInteger(42n),
    );
    assert.deepEqual(
      usearch.parseKey("0000000000000000000000000000002a"),
      usearch.keyFromInteger(42n),
      "dashes may sit nowhere",
    );
    assert.throws(() => usearch.parseKey("not-a-key"));
  });

  await t.test("a 128-bit identity round-trips through the index", () => {
    const index = usearch.DenseIndex.forRam({
      metric: "euclidean",
      connectivity: 16,
      dimensions: 3,
    });
    const key = usearch.parseKey("f81d4fae-7dec-11d0-a765-00a0c91e6bf6");
    index.add(key, new Float32Array([0.2, 0.6, 0.4]), EXECUTOR);

    const matches = usearch.MatchesSpan.reserve(1, 1);
    index.search(new Float32Array([0.2, 0.6, 0.4]), matches, EXECUTOR);
    const found = matches.get(0);

    assert.deepEqual(found.keys, key, "the high bytes survive the round trip");
    assert.equal(
      usearch.formatKey(usearch.keyAt(found.keys, 0)),
      "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
    );
    assert.throws(
      () => usearch.integerFromKey(found.keys),
      "no integer spells an identity with leading bytes set",
    );
    assert.deepEqual(
      index.contains(key),
      [true],
      "a Uint8Array spells a run, so it answers with a run",
    );
  });

  await t.test("an integer key crosses the addon in the ABI's own byte order", () => {
    const index = usearch.DenseIndex.forRam({
      metric: "euclidean",
      connectivity: 16,
      dimensions: 2,
    });
    index.add(new BigUint64Array([42n]), new Float32Array([10, 20]), EXECUTOR);

    const matches = usearch.MatchesSpan.reserve(1, 1);
    index.search(new Float32Array([10, 20]), matches, EXECUTOR);
    const found = matches.get(0);

    assert.deepEqual(
      found.keys,
      new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42]),
      "the engine holds 42 in the trailing byte, not the leading one",
    );
    assert.equal(
      usearch.formatKey(found.keys),
      "00000000-0000-0000-0000-00000000002a",
      "which is the canonical rendering the ABI fixes",
    );
  });

  await t.test("a key given as a UUID string matches the same key given as bytes", () => {
    const index = usearch.DenseIndex.forRam({
      metric: "euclidean",
      connectivity: 16,
      dimensions: 3,
    });
    index.add("f81d4fae-7dec-11d0-a765-00a0c91e6bf6", new Float32Array([0.2, 0.6, 0.4]), EXECUTOR);

    assert.equal(index.count("f81d4fae-7dec-11d0-a765-00a0c91e6bf6"), 1);
    assert.equal(index.contains("f81d4fae-7dec-11d0-a765-00a0c91e6bf6"), true);
  });
});

test("Single-entry operations", async (t) => {
  await t.test("index info", () => {
    const index = usearch.DenseIndex.forRam({
      dimensions: 2,
      metric: "euclidean",
      connectivity: 16,
    });

    assert.equal(index.connectivity(), 16, "connectivity should be 16");
    assert.equal(index.dimensions(), 2, "dimensions should be 2");
    assert.equal(index.size(), 0, "initial size should be 0");
  });

  await t.test("add", () => {
    const index = usearch.DenseIndex.forRam({ dimensions: 2, metric: "euclidean" });

    index.add(15n, new Float32Array([10, 20]), EXECUTOR);
    index.add(16n, new Float32Array([10, 25]), EXECUTOR);

    assert.equal(index.size(), 2, "size after adding elements should be 2");
    assert.equal(
      index.contains(15),
      true,
      "entry must be present after insertion",
    );

    const matches = usearch.MatchesSpan.reserve(1, 2);
    index.search(new Float32Array([13, 14]), matches, EXECUTOR);
    const found = matches.get(0);

    assert.deepEqual(
      found.keys,
      keysFromIntegers(15n, 16n),
      "keys should be 15 and 16",
    );
    assert(found.distances instanceof Float64Array, "distances are doubles");
    assert(
      found.distances[0] < found.distances[1],
      "matches are ranked nearest first",
    );
  });

  await t.test("get", () => {
    const index = usearch.DenseIndex.forRam({
      dimensions: 2,
      metric: "euclidean",
      quantization: "f32",
    });

    index.add(15n, new Float32Array([10, 20]), EXECUTOR);
    const stored = index.get(15n);

    assert.equal(stored.length, 2, "one row of two dimensions");
    assertAlmostEqual(stored[0], 10, 1e-3);
    assertAlmostEqual(stored[1], 20, 1e-3);
    assert.equal(index.get(16n).length, 0, "an absent key exports nothing");
  });
});

test("Batch operations", async (t) => {
  await t.test("add", () => {
    const indexBatch = usearch.DenseIndex.forRam({
      dimensions: 2,
      metric: "euclidean",
    });

    const keys = [15n, 16n];
    const vectors = [new Float32Array([10, 20]), new Float32Array([10, 25])];

    assert.equal(indexBatch.add(keys, vectors, EXECUTOR), 2, "two rows published");
    assert.equal(indexBatch.size(), 2, "size after adding batch should be 2");

    const matches = usearch.MatchesSpan.reserve(1, 2);
    indexBatch.search(new Float32Array([13, 14]), matches, EXECUTOR);
    const found = matches.get(0);

    assert.deepEqual(
      found.keys,
      keysFromIntegers(15n, 16n),
      "keys should be 15 and 16",
    );
    assert(
      found.distances[0] < found.distances[1],
      "matches are ranked nearest first",
    );
  });

  await t.test("search", () => {
    const indexBatch = usearch.DenseIndex.forRam({
      dimensions: 3,
      metric: "euclidean",
    });

    const keys = [15n, 16n];
    const vectors = [
      new Float32Array([10, 20, 30]),
      new Float32Array([10, 25, 35]),
    ];

    indexBatch.add(keys, vectors, EXECUTOR);
    assert.equal(indexBatch.size(), 2, "size after adding batch should be 2");

    const matches = usearch.MatchesSpan.reserve(2, 2);
    indexBatch.search(
      [new Float32Array([13, 14, 15]), new Float32Array([13, 14, 15])],
      matches,
      EXECUTOR,
    );

    assert.equal(matches.wanted, 2, "the run was reserved two wide");

    assert.deepEqual(
      matches.counts,
      new BigUint64Array([2n, 2n]),
      "counts should be 2 for both keys",
    );

    assert.deepEqual(
      matches.keys,
      keysFromIntegers(15n, 16n, 15n, 16n),
      "keys should be 15 and 16",
    );
    for (const queryIndex of [0, 1])
      assert(
        matches.distances[queryIndex * 2] < matches.distances[queryIndex * 2 + 1],
        "every row is ranked nearest first",
      );
  });

  await t.test("a batch row reads back as its own MatchesRow", () => {
    const indexBatch = usearch.DenseIndex.forRam({
      dimensions: 3,
      metric: "euclidean",
    });
    indexBatch.add(
      [15n, 16n],
      [new Float32Array([10, 20, 30]), new Float32Array([10, 25, 35])],
      EXECUTOR,
    );

    const matches = usearch.MatchesSpan.reserve(2, 2);
    indexBatch.search(
      [new Float32Array([13, 14, 15]), new Float32Array([10, 20, 30])],
      matches,
      EXECUTOR,
    );
    const second = matches.get(1);

    assert.equal(second.count, 2, "the row carries both matches");
    assert.deepEqual(second.keys, keysFromIntegers(15n, 16n));
    assertAlmostEqual(second.distances[0], 0, 1e-3);
  });
});

test("A run's width follows from the buffers it lends", async (t) => {
  await t.test("three buffers of one shape derive the width", () => {
    const matches = new usearch.MatchesSpan(
      new Uint8Array(4 * usearch.keyBytes),
      new Float64Array(4),
      new BigUint64Array(2),
    );

    assert.equal(matches.queries, 2, "one entry of `counts` per query");
    assert.equal(matches.wanted, 2, "four cells over two queries is two apiece");
  });

  await t.test("buffers that read as no single shape are refused", () => {
    assert.throws(
      () =>
        new usearch.MatchesSpan(
          new Uint8Array(4 * usearch.keyBytes),
          new Float64Array(3),
          new BigUint64Array(2),
        ),
      /same shape/,
      "keys and distances must cover the same cells",
    );
    assert.throws(
      () =>
        new usearch.MatchesSpan(
          new Uint8Array(5 * usearch.keyBytes),
          new Float64Array(5),
          new BigUint64Array(2),
        ),
      /one row per query/,
      "five cells do not divide into two rows",
    );
    assert.throws(
      () =>
        new usearch.MatchesSpan(
          new Uint8Array(3),
          new Float64Array(3),
          new BigUint64Array(1),
        ),
      /sixteen-byte identities/,
      "a run of keys is a whole number of identities",
    );
  });
});

test("Expected results", () => {
  const index = usearch.DenseIndex.forRam({
    metric: "euclidean",
    connectivity: 16,
    dimensions: 3,
  });
  index.add(42n, new Float32Array([0.2, 0.6, 0.4]), EXECUTOR);
  const matches = usearch.MatchesSpan.reserve(1, 10);
  index.search(new Float32Array([0.2, 0.6, 0.4]), matches, EXECUTOR);
  const found = matches.get(0);

  assert.equal(index.size(), 1);
  assert.deepEqual(found.keys, usearch.keyFromInteger(42n));
  assertAlmostEqual(found.distances[0], 0);
});

test("One index served by two executors returns the same results", () => {
  const index = usearch.DenseIndex.forRam({
    metric: "euclidean",
    connectivity: 16,
    dimensions: 3,
  });
  index.add(42n, new Float32Array([0.2, 0.6, 0.4]), EXECUTOR);
  assert.equal(index.size(), 1);

  const solo = usearch.Executor.forRam(1);
  const fromSolo = usearch.MatchesSpan.reserve(1, 10);
  const fromShared = usearch.MatchesSpan.reserve(1, 10);
  index.search(new Float32Array([0.2, 0.6, 0.4]), fromSolo, solo);
  index.search(new Float32Array([0.2, 0.6, 0.4]), fromShared, EXECUTOR);

  assert.deepEqual(fromSolo.get(0).keys, fromShared.get(0).keys);
  assert.deepEqual(fromSolo.get(0).distances, fromShared.get(0).distances);
});

test("A verb without an executor is refused", () => {
  const index = usearch.DenseIndex.forRam({
    metric: "euclidean",
    connectivity: 16,
    dimensions: 3,
  });
  assert.throws(() => index.add(42n, new Float32Array([0.2, 0.6, 0.4])));
});

test("A row narrower than the index is refused, not read past", () => {
  const index = usearch.DenseIndex.forRam({
    metric: "euclidean",
    connectivity: 16,
    dimensions: 8,
  });
  assert.throws(() => index.add(42n, new Float32Array([0.2, 0.6, 0.4]), EXECUTOR));
  assert.throws(() =>
    index.search(
      new Float32Array([0.2, 0.6, 0.4]),
      usearch.MatchesSpan.reserve(1, 1),
      EXECUTOR,
    ),
  );
});

test("Expected count()", async (t) => {
  const index = usearch.DenseIndex.forRam({
    metric: "euclidean",
    connectivity: 16,
    dimensions: 3,
  });
  index.add(
    [42n, 43n],
    [new Float32Array([0.2, 0.6, 0.4]), new Float32Array([0.2, 0.6, 0.4])],
    EXECUTOR,
  );

  await t.test("Argument is a number", () => {
    assert.equal(1, index.count(43n));
  });
  await t.test("Argument is a number (does not exist)", () => {
    assert.equal(0, index.count(44n));
  });
  await t.test("Argument is an array", () => {
    assert.deepEqual([1, 1, 0], index.count([42n, 43n, 44n]));
  });
  await t.test("A run of keys answers with a run of booleans", () => {
    assert.deepEqual([true, true, false], index.contains([42n, 43n, 44n]));
  });
});

test("Operations with invalid values", () => {
  const indexBatch = usearch.DenseIndex.forRam({
    dimensions: 2,
    metric: "euclidean",
  });

  const keys = [NaN, 16n];
  const vectors = [new Float32Array([10, 30]), new Float32Array([1, 5])];

  // All keys must be positive integers or bigints
  assert.throws(() => indexBatch.add(keys, vectors, EXECUTOR));

  // Vectors must be a TypedArray or an array of arrays
  assert.throws(() =>
    indexBatch.search(NaN, usearch.MatchesSpan.reserve(1, 2), EXECUTOR),
  );
});

test("Several entries under one key", async (t) => {
  const index = usearch.DenseIndex.forRam({
    metric: "euclidean",
    connectivity: 16,
    dimensions: 3,
    valuesPerKey: 3,
  });
  index.add(
    [42n, 42n],
    [new Float32Array([0.1, 0.6, 0.4]), new Float32Array([0.2, 0.6, 0.4])],
    EXECUTOR,
  );

  await t.test("the count is the whole extent", () => {
    assert.equal(index.count(42n), 2);
    assert.equal(index.contains(42n), true);
  });

  await t.test("every entry exports, in insertion order", () => {
    const stored = index.get(42n);
    assert.equal(stored.length, 6, "two rows of three dimensions");
    assertAlmostEqual(stored[0], 0.1, 1e-3);
    assertAlmostEqual(stored[3], 0.2, 1e-3);
  });
});

test("Capacity and clearing", () => {
  const index = usearch.DenseIndex.forRam({
    metric: "euclidean",
    connectivity: 16,
    dimensions: 3,
  });
  index.reserve(128);
  assert(index.capacity() >= 128, "the reservation must hold");

  index.add(42n, new Float32Array([0.2, 0.6, 0.4]), EXECUTOR);
  assert.equal(index.size(), 1);
  assert(index.memoryUsage() > 0, "a populated index occupies memory");

  index.clear();
  assert.equal(index.size(), 0, "clearing erases every entry");
  assert.equal(index.contains(42n), false);
});

test("Refinement leaves the graph searchable", () => {
  const index = usearch.DenseIndex.forRam({
    metric: "euclidean",
    connectivity: 16,
    dimensions: 4,
  });
  const keys = [];
  const rows = [];
  for (let entry = 0; entry < 64; entry++) {
    keys.push(BigInt(entry));
    rows.push(new Float32Array([entry, entry + 1, entry + 2, entry + 3]));
  }
  index.add(keys, rows, EXECUTOR);

  const changed = index.refine(EXECUTOR, { expansionFactor: 64 });
  assert(typeof changed === "number", "refinement reports the nodes it touched");
  assert.equal(index.size(), 64, "refinement adds and removes nothing");

  const matches = usearch.MatchesSpan.reserve(1, 1);
  index.search(new Float32Array([7, 8, 9, 10]), matches, EXECUTOR);
  assert.equal(usearch.integerFromKey(matches.get(0).keys), 7n);
});

test("Late interaction over multi-row entries", () => {
  const index = usearch.DenseIndex.forRam({
    metric: "inner",
    dimensions: 4,
    valuesPerKey: 4,
  });
  index.add(
    [7n, 7n, 8n, 8n],
    [
      new Float32Array([1, 0, 0, 0]),
      new Float32Array([0, 1, 0, 0]),
      new Float32Array([0, 0, 1, 0]),
      new Float32Array([0, 0, 0, 1]),
    ],
    EXECUTOR,
  );

  const matches = usearch.MatchesSpan.reserve(1, 2);
  index.maxsim(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0]), 2, matches, EXECUTOR);

  assert.equal(matches.wanted, 2);
  assert.equal(Number(matches.counts[0]) > 0, true, "at least one key scored");
  assert.equal(usearch.integerFromKey(usearch.keyAt(matches.keys, 0)), 7n);
});

test("Sparse products against the indexed rows", async (t) => {
  const populated = () => {
    const index = usearch.DenseIndex.forRam({ metric: "inner", dimensions: 4 });
    index.add(
      [1n, 2n, 3n],
      [
        new Float32Array([1, 0, 0, 0]),
        new Float32Array([0, 1, 0, 0]),
        new Float32Array([0, 0, 1, 0]),
      ],
      EXECUTOR,
    );
    return index;
  };

  await t.test("every left row emits the pairs the walk scored", () => {
    const products = usearch.MatchesSpan.reserve(2, 2);
    populated().multiply(
      [new Float32Array([1, 0, 0, 0]), new Float32Array([0, 1, 0, 0])],
      products,
      EXECUTOR,
    );

    assert.equal(products.wanted, 2, "the run's own shape fixes the density");
    assert.equal(products.counts.length, 2, "one emitted run per left row");
    assert(products.distances instanceof Float64Array, "products are doubles");

    // Pairs arrive in traversal order, and a count above `wanted` means the row was truncated.
    const emitted = Math.min(Number(products.counts[0]), products.wanted);
    assert(emitted > 0, "the first row emitted at least one pair");
    const columns = [];
    for (let pair = 0; pair < emitted; pair++)
      columns.push(usearch.integerFromKey(usearch.keyAt(products.keys, pair)));
    assert(columns.includes(1n), "the column the first row overlaps is among them");
  });

  await t.test("a graph frozen under a distance kind is refused", () => {
    const index = usearch.DenseIndex.forRam({
      metric: "euclidean",
      dimensions: 4,
    });
    index.add(1n, new Float32Array([1, 0, 0, 0]), EXECUTOR);
    assert.throws(() =>
      index.multiply(
        new Float32Array([1, 0, 0, 0]),
        usearch.MatchesSpan.reserve(1, 1),
        EXECUTOR,
      ),
    );
  });
});

test("Attention writes into the buffer the caller lends", async (t) => {
  // An entry's key is its row offset into the value table, so the keys run from zero.
  const index = usearch.DenseIndex.forRam({ metric: "inner", dimensions: 4 });
  index.add(
    [0n, 1n],
    [new Float32Array([1, 0, 0, 0]), new Float32Array([0, 1, 0, 0])],
    EXECUTOR,
  );
  const queries = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0]);
  const valueDimensions = 2;
  const values = new Float32Array([10, 20, 30, 40]);

  await t.test("an omitted carry starts the softmax fresh", () => {
    const outputs = new Float32Array(4);
    const answer = index.attend(
      queries,
      values,
      valueDimensions,
      outputs,
      EXECUTOR,
    );

    assert.equal(answer, undefined, "the outputs are the answer, not the return");
    assert(
      outputs.some((scalar) => scalar !== 0),
      "the lent buffer was written in place",
    );
    for (const scalar of outputs)
      assert(Number.isFinite(scalar), "every attended scalar is finite");
  });

  await t.test("a carry of two floats per query row is accepted", () => {
    const outputs = new Float32Array(4);
    const carry = new Float32Array(4);
    const stats = new usearch.Stats();
    index.attend(queries, values, valueDimensions, outputs, EXECUTOR, {
      carry,
      stats,
    });

    assert(
      outputs.some((scalar) => scalar !== 0),
      "the lent buffer was written in place",
    );
    for (const scalar of carry)
      assert(Number.isFinite(scalar), "the maximum and the partition are finite");
    assert.equal(
      typeof stats.computedDistances,
      "number",
      "attention reports the walk it paid for",
    );
  });

  await t.test("a carry that holds no maximum and partition per query is refused", () => {
    assert.throws(() =>
      index.attend(queries, values, valueDimensions, new Float32Array(4), EXECUTOR, {
        carry: new Float32Array(3),
      }),
    );
  });

  await t.test("outputs that hold no row of value dimensions per query are refused", () => {
    assert.throws(() =>
      index.attend(queries, values, valueDimensions, new Float32Array(3), EXECUTOR),
    );
  });
});

test("Snapshot cursors walk every entry once", () => {
  const index = usearch.DenseIndex.forRam({
    metric: "euclidean",
    connectivity: 16,
    dimensions: 2,
  });
  index.add(
    [1n, 2n, 3n],
    [
      new Float32Array([1, 1]),
      new Float32Array([2, 2]),
      new Float32Array([3, 3]),
    ],
    EXECUTOR,
  );

  const cursor = index.cursor();
  const seen = [];
  for (;;) {
    const batch = cursor.next(2);
    if (!batch) break;
    const pulled = batch.keys.length / usearch.keyBytes;
    for (let position = 0; position < pulled; position++)
      seen.push(usearch.integerFromKey(usearch.keyAt(batch.keys, position)));
    assert.equal(
      batch.vectors.length,
      pulled * 2,
      "one row of two dimensions per identity",
    );
  }
  cursor.free();
  cursor.free(); // releasing twice is a no-op

  seen.sort((first, second) => Number(first - second));
  assert.deepEqual(seen, [1n, 2n, 3n]);
});

test("Finders pull matches lazily", () => {
  const index = usearch.DenseIndex.forRam({
    metric: "euclidean",
    connectivity: 16,
    dimensions: 2,
  });
  index.add(
    [1n, 2n, 3n],
    [
      new Float32Array([1, 1]),
      new Float32Array([2, 2]),
      new Float32Array([3, 3]),
    ],
    EXECUTOR,
  );

  const finder = index.find(new Float32Array([1, 1]), EXECUTOR);
  const first = finder.next(1);

  assert.equal(first.count, 1, "one match per pull");
  assert.equal(usearch.integerFromKey(first.keys), 1n, "the nearest comes first");
  assert(first.distances instanceof Float64Array, "distances are doubles");
  finder.free();
});

test("Serialization", async (t) => {
  const indexPath = path.join(os.tmpdir(), "usearch.test.safegraphs");

  t.beforeEach(() => {
    const index = usearch.DenseIndex.forRam({
      metric: "euclidean",
      connectivity: 16,
      dimensions: 3,
    });
    index.add(42n, new Float32Array([0.2, 0.6, 0.4]), EXECUTOR);
    index.save(indexPath, EXECUTOR);
  });

  t.afterEach(() => {
    fs.unlinkSync(indexPath);
  });

  await t.test("load", () => {
    const index = usearch.DenseIndex.forRam({
      metric: "euclidean",
      connectivity: 16,
      dimensions: 3,
    });
    index.load(indexPath, EXECUTOR);
    const matches = usearch.MatchesSpan.reserve(1, 10);
    index.search(new Float32Array([0.2, 0.6, 0.4]), matches, EXECUTOR);
    const found = matches.get(0);

    assert.equal(index.size(), 1);
    assert.deepEqual(found.keys, usearch.keyFromInteger(42n));
    assertAlmostEqual(found.distances[0], 0);
  });

  await t.test("metadata", () => {
    const metadata = usearch.DenseIndex.metadata(indexPath);

    assert.equal(metadata.dimensions, 3);
    assert.equal(metadata.metric, "euclidean");
    assert.equal(metadata.connectivity, 16);
  });
});

test("Serialization through memory", async (t) => {
  const index = usearch.DenseIndex.forRam({
    metric: "euclidean",
    connectivity: 16,
    dimensions: 3,
  });
  index.add(42n, new Float32Array([0.2, 0.6, 0.4]), EXECUTOR);

  const buffer = index.saveBuffer(EXECUTOR);
  assert.equal(
    buffer.length,
    index.storageRequirements(),
    "the monolith is exactly as long as the requirement said",
  );

  await t.test("loadBuffer", () => {
    const restored = usearch.DenseIndex.forRam({
      metric: "euclidean",
      connectivity: 16,
      dimensions: 3,
    });
    restored.loadBuffer(buffer, EXECUTOR);

    assert.equal(restored.size(), 1);
    const matches = usearch.MatchesSpan.reserve(1, 1);
    restored.search(new Float32Array([0.2, 0.6, 0.4]), matches, EXECUTOR);
    assert.deepEqual(matches.get(0).keys, usearch.keyFromInteger(42n));
  });

  await t.test("metadataBuffer", () => {
    const metadata = usearch.DenseIndex.metadataBuffer(buffer);

    assert.equal(metadata.dimensions, 3);
    assert.equal(metadata.metric, "euclidean");
  });
});

test("Mini-float quantization types", async (t) => {
  for (const quantization of ["e5m2", "e4m3", "e3m2", "e2m3"]) {
    await t.test(`add and search with ${quantization}`, () => {
      const index = usearch.DenseIndex.forRam({
        metric: "angular",
        dimensions: 64,
        quantization: quantization,
      });
      const vector = new Float32Array(64);
      for (let position = 0; position < 64; position++)
        vector[position] = position * 0.1;
      index.add(1n, vector, EXECUTOR);

      const matches = usearch.MatchesSpan.reserve(1, 1);
      index.search(vector, matches, EXECUTOR);
      assert.equal(usearch.integerFromKey(matches.get(0).keys), 1n);
    });
  }
});

test("Input scalar kinds beyond the four a TypedArray spells", async (t) => {
  await t.test("packed bits reach a Hamming index as u1x8", () => {
    const index = usearch.DenseIndex.forRam({
      metric: "hamming",
      dimensions: 64,
      quantization: "u1x8",
    });
    const first = new Uint8Array([0b10101010, 0, 0, 0, 0, 0, 0, 0]);
    const second = new Uint8Array([0b10101011, 0, 0, 0, 0, 0, 0, 0]);
    index.add([1n, 2n], [first, second], EXECUTOR, {
      scalarKind: usearch.ScalarKind.U1x8,
    });

    const matches = usearch.MatchesSpan.reserve(1, 2);
    index.search(first, matches, EXECUTOR, {
      scalarKind: usearch.ScalarKind.U1x8,
    });
    const found = matches.get(0);
    assert.equal(usearch.integerFromKey(usearch.keyAt(found.keys, 0)), 1n);
    assertAlmostEqual(found.distances[0], 0, 1e-3);
    assertAlmostEqual(found.distances[1], 1, 1e-3);
  });

  await t.test("bf16 rows travel as bytes with their kind named", () => {
    const index = usearch.DenseIndex.forRam({
      metric: "angular",
      dimensions: 4,
      quantization: "bf16",
    });
    // The bf16 encoding of 1.0 is 0x3f80, little-endian in the byte pair.
    const row = new Uint8Array([0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f]);
    index.add(1n, row, EXECUTOR, { scalarKind: usearch.ScalarKind.BF16 });

    const matches = usearch.MatchesSpan.reserve(1, 1);
    index.search(row, matches, EXECUTOR, {
      scalarKind: usearch.ScalarKind.BF16,
    });
    assert.equal(usearch.integerFromKey(matches.get(0).keys), 1n);
  });

  await t.test("an Int16Array without a named kind is refused", () => {
    const index = usearch.DenseIndex.forRam({ metric: "euclidean", dimensions: 4 });
    assert.throws(() => index.add(1n, new Int16Array([1, 2, 3, 4]), EXECUTOR));
  });

  await t.test("f64 rows keep their width on the way in", () => {
    const index = usearch.DenseIndex.forRam({
      metric: "euclidean",
      dimensions: 2,
      quantization: "f64",
    });
    index.add(1n, new Float64Array([10, 20]), EXECUTOR);

    const matches = usearch.MatchesSpan.reserve(1, 1);
    index.search(new Float64Array([10, 20]), matches, EXECUTOR);
    const found = matches.get(0);
    assert.equal(usearch.integerFromKey(found.keys), 1n);
    assertAlmostEqual(found.distances[0], 0, 1e-9);
  });
});

test("Library introspection", () => {
  assert.equal(typeof usearch.version(), "string");
  assert.equal(typeof usearch.capabilitiesCompiled(), "string");
  assert.equal(typeof usearch.capabilitiesAvailable(), "string");
});

test("Traversal telemetry", async (t) => {
  const populated = () => {
    const index = usearch.DenseIndex.forRam({
      metric: "euclidean",
      connectivity: 16,
      dimensions: 4,
    });
    const keys = [];
    const rows = [];
    for (let entry = 0; entry < 64; entry++) {
      keys.push(BigInt(entry));
      rows.push(new Float32Array([entry, entry + 1, entry + 2, entry + 3]));
    }
    index.add(keys, rows, EXECUTOR);
    return index;
  };

  const assertCounters = (stats, note) => {
    for (const counter of ["visitedMembers", "computedDistances", "restarts"]) {
      assert.equal(typeof stats[counter], "number", `${note}: ${counter} is a number`);
      assert(stats[counter] >= 0, `${note}: ${counter} is non-negative`);
      assert(Number.isInteger(stats[counter]), `${note}: ${counter} is whole`);
    }
  };

  await t.test("a fresh accumulator reads zero", () => {
    assertCounters(new usearch.Stats(), "fresh");
    assert.equal(new usearch.Stats().visitedMembers, 0);
  });

  await t.test("every walking verb reports into the accumulator it is given", () => {
    const index = populated();
    for (const [verb, run] of [
      ["add", (stats) => index.add(99n, new Float32Array([1, 2, 3, 4]), EXECUTOR, { stats })],
      ["refine", (stats) => index.refine(EXECUTOR, { expansionFactor: 64, stats })],
      [
        "search",
        (stats) =>
          index.search(
            new Float32Array([7, 8, 9, 10]),
            usearch.MatchesSpan.reserve(1, 4),
            EXECUTOR,
            { stats },
          ),
      ],
    ]) {
      const stats = new usearch.Stats();
      run(stats);
      assertCounters(stats, verb);
    }
  });

  await t.test("maxsim reports into the accumulator it is given", () => {
    const index = usearch.DenseIndex.forRam({
      metric: "inner",
      dimensions: 4,
      valuesPerKey: 2,
    });
    index.add(
      [7n, 7n, 8n, 8n],
      [
        new Float32Array([1, 0, 0, 0]),
        new Float32Array([0, 1, 0, 0]),
        new Float32Array([0, 0, 1, 0]),
        new Float32Array([0, 0, 0, 1]),
      ],
      EXECUTOR,
    );

    const stats = new usearch.Stats();
    const matches = usearch.MatchesSpan.reserve(1, 2);
    index.maxsim(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0]), 2, matches, EXECUTOR, {
      stats,
    });
    assertCounters(stats, "maxsim");
  });

  await t.test("one accumulator sums the calls it is shared across", () => {
    const index = populated();
    const query = new Float32Array([7, 8, 9, 10]);
    const matches = usearch.MatchesSpan.reserve(1, 4);
    const once = new usearch.Stats();
    index.search(query, matches, EXECUTOR, { stats: once });

    const twice = new usearch.Stats();
    index.search(query, matches, EXECUTOR, { stats: twice });
    index.search(query, matches, EXECUTOR, { stats: twice });
    assert.equal(twice.visitedMembers, 2 * once.visitedMembers);
    assert.equal(twice.computedDistances, 2 * once.computedDistances);
    assert.equal(twice.restarts, 2 * once.restarts);
  });

  await t.test("accumulate folds one reading into another", () => {
    const total = new usearch.Stats();
    const reading = new usearch.Stats();
    reading.visitedMembers = 5;
    reading.computedDistances = 11;
    reading.restarts = 1;
    total.accumulate(reading);
    total.accumulate(reading);
    assert.deepEqual(
      { ...total },
      { visitedMembers: 10, computedDistances: 22, restarts: 2 },
    );
  });

  await t.test("an omitted accumulator leaves every return shape alone", () => {
    const index = populated();
    assert.equal(typeof index.add(98n, new Float32Array([2, 3, 4, 5]), EXECUTOR), "number");
    assert.equal(typeof index.refine(EXECUTOR), "number");
    const matches = usearch.MatchesSpan.reserve(1, 1);
    assert.equal(
      index.search(new Float32Array([7, 8, 9, 10]), matches, EXECUTOR),
      undefined,
      "a ranked verb answers with the run it filled, not a return value",
    );
    assert.equal(usearch.integerFromKey(matches.get(0).keys), 7n);
  });
});
