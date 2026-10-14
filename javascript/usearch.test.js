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
const v8 = require("node:v8");
const vm = require("node:vm");
const usearch = require("./dist/cjs/usearch.js");

// A full collection on demand, so the tests can drop a handle and watch what survives it.
v8.setFlagsFromString("--expose-gc");
const collect = vm.runInNewContext("gc");

function assertAlmostEqual(actual, expected, tolerance = 1e-6) {
  const lowerBound = expected - tolerance;
  const upperBound = expected + tolerance;
  assert(actual >= lowerBound && actual <= upperBound, `Expected ${actual} to be almost equal to ${expected}`);
}

// The byte run several integer identities spell, laid end to end.
function keysFromIntegers(...integers) {
  const run = new Uint8Array(integers.length * usearch.keyBytes);
  for (let position = 0; position < integers.length; position++)
    run.set(usearch.keyFromInteger(integers[position]), position * usearch.keyBytes);
  return run;
}

// A packed run of float rows, `columns` wide.
function rows(columns, ...values) {
  return usearch.RowsView.of(new Float32Array(values), columns);
}

// The bf16 words of values a bf16 holds exactly: the high halves of their float32 bits.
function bf16Words(values) {
  return Uint16Array.from(new Uint32Array(Float32Array.from(values).buffer), (bits) => bits >>> 16);
}

// Whether `verb` ran, rather than answering unsupported as the engine does for the verbs it declares before serving.
function served(verb) {
  try {
    verb();
    return true;
  } catch (error) {
    if (error.code !== usearch.Status.Unsupported) throw error;
    return false;
  }
}

// An index of `options`, or a skipped test when this build's USEARCH_DENSE_DTYPES left its quantization out.
function initRamOrSkip(t, options) {
  try {
    return usearch.DenseIndex.initRam(options);
  } catch (error) {
    if (error.code !== usearch.Status.Unsupported) throw error;
    t.skip(`this build leaves ${options.quantization} out`);
    return undefined;
  }
}

// What a ranked run holds, read out of its arena so two runs compare by content.
function contents(matches) {
  return { keys: matches.keys, scores: matches.scores, counts: matches.counts, columns: matches.columns };
}

const { Arena, DenseIndex, DenseMetric, Executor, Matches, RowsView, Scalar, Status } = usearch;
const EXECUTOR = Executor.forRam({ threads: 2 });

test("Residency factories", async (t) => {
  await t.test("a direct construction is refused, and names the factories", () => {
    assert.throws(() => new Executor(), {
      code: Status.InvalidArgument,
      message: "An Executor is made by Executor.forRam, Executor.onPool, Executor.forVram or Executor.forNvme",
    });
    assert.throws(() => new DenseIndex(), {
      code: Status.InvalidArgument,
      message: "A DenseIndex is opened by DenseIndex.initRam, DenseIndex.initVram, or a DenseIndex.load or view verb.",
    });
  });

  await t.test("a count that is not a whole number at or above zero is refused, never cast", () => {
    const refusal = { code: Status.InvalidArgument, message: "The 'threads' argument must be a non-negative integer" };
    assert.throws(() => Executor.forRam({ threads: -1 }), refusal);
    assert.throws(() => Executor.forRam({ threads: 2.5 }), refusal);
    assert.throws(() => Executor.forRam({ threads: Infinity }), refusal);
    assert.throws(() => Executor.forRam({ threads: NaN }), refusal);
    assert.throws(() => Executor.forRam({ threads: "2" }), {
      code: Status.InvalidArgument,
      message: "The 'threads' argument must be a number or a BigInt",
    });
  });

  await t.test("every executor and index is an instance of its class", () => {
    assert(EXECUTOR instanceof Executor);
    assert(DenseIndex.initRam({ dimensions: 2 }) instanceof DenseIndex);
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
    assert.equal(usearch.formatKey(usearch.keyFromInteger(42n)), "00000000-0000-0000-0000-00000000002a");
    assert.deepEqual(usearch.parseKey("00000000-0000-0000-0000-00000000002a"), usearch.keyFromInteger(42n));
    assert.deepEqual(
      usearch.parseKey("0000000000000000000000000000002a"),
      usearch.keyFromInteger(42n),
      "dashes may sit nowhere",
    );
    assert.throws(() => usearch.parseKey("not-a-key"), {
      code: Status.Malformed,
      message: "The key is not thirty-two hexadecimal digits.",
    });
  });

  await t.test("a 128-bit identity round-trips through the index", () => {
    const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 3 });
    const key = usearch.parseKey("f81d4fae-7dec-11d0-a765-00a0c91e6bf6");
    index.reserve(1);
    index.insert(key, new Float32Array([0.2, 0.6, 0.4]));

    const found = index.search(new Float32Array([0.2, 0.6, 0.4]), 1, new Arena()).get(0);

    assert.deepEqual(found.keys, key, "the high bytes survive the round trip");
    assert.equal(usearch.formatKey(usearch.keyAt(found.keys, 0)), "f81d4fae-7dec-11d0-a765-00a0c91e6bf6");
    assert.throws(() => usearch.integerFromKey(found.keys), {
      code: Status.Malformed,
      message: "The key f81d4fae-7dec-11d0-a765-00a0c91e6bf6 carries no integer identity.",
    });
    assert.equal(index.contains(key), true, "sixteen bytes spell one key");
  });

  await t.test("an integer key crosses the addon in the ABI's own byte order", () => {
    const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 2 });
    index.reserve(1);
    index.insert(new BigUint64Array([42n]), new Float32Array([10, 20]));

    const found = index.search(new Float32Array([10, 20]), 1, new Arena()).get(0);

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
    const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 3 });
    index.reserve(1);
    index.insert("f81d4fae-7dec-11d0-a765-00a0c91e6bf6", new Float32Array([0.2, 0.6, 0.4]));

    assert.equal(index.count("f81d4fae-7dec-11d0-a765-00a0c91e6bf6"), 1);
    assert.equal(index.contains(usearch.parseKey("f81d4fae-7dec-11d0-a765-00a0c91e6bf6")), true);
  });
});

test("Single-entry operations", async (t) => {
  await t.test("index info", () => {
    const index = DenseIndex.initRam({ dimensions: 2, metric: DenseMetric.Euclidean, connectivity: 16 });

    assert.deepEqual(index.describe().metadata, {
      metric: DenseMetric.Euclidean,
      quantization: Scalar.BF16,
      dimensions: 2,
      connectivity: 16,
    });
    assert.equal(index.describe().size, 0, "initial size should be 0");
  });

  await t.test("insert", () => {
    const index = DenseIndex.initRam({ dimensions: 2, metric: DenseMetric.Euclidean });

    index.reserve(2);
    index.insert(15n, new Float32Array([10, 20]));
    index.insert(16n, new Float32Array([10, 25]));

    assert.equal(index.describe().size, 2, "size after adding elements should be 2");
    assert.equal(index.contains(15), true, "entry must be present after insertion");

    const found = index.search(new Float32Array([13, 14]), 2, new Arena()).get(0);

    assert.deepEqual(found.keys, keysFromIntegers(15n, 16n), "keys should be 15 and 16");
    assert(found.scores[0] < found.scores[1], "matches are ranked nearest first");
  });

  await t.test("get shapes its rows from the width the call reports", () => {
    const index = DenseIndex.initRam({ dimensions: 2, metric: DenseMetric.Euclidean });

    index.reserve(1);
    index.insert(15n, new Float32Array([10, 20]));
    const stored = index.get(15n, new Arena());

    assert.deepEqual(
      { data: stored.data, rows: stored.rows, columns: stored.columns, stride: stored.stride, scalar: stored.scalar },
      { data: new Float32Array([10, 20]), rows: 1, columns: 2, stride: 8, scalar: Scalar.F32 },
    );
    const absent = index.get(16n, new Arena());
    assert.deepEqual({ length: absent.data.length, rows: absent.rows }, { length: 0, rows: 0 });
  });
});

test("Batch operations", async (t) => {
  await t.test("a run of rows inserts under a run of keys", () => {
    const index = DenseIndex.initRam({ dimensions: 2, metric: DenseMetric.Euclidean });

    index.reserve(2);
    assert.equal(index.insert([15n, 16n], rows(2, 10, 20, 10, 25)), undefined, "an insertion answers with nothing");
    assert.equal(index.describe().size, 2, "size after adding batch should be 2");

    const found = index.search(new Float32Array([13, 14]), 2, new Arena()).get(0);
    assert.deepEqual(found.keys, keysFromIntegers(15n, 16n), "keys should be 15 and 16");
  });

  await t.test("a single key broadcasts across every row", () => {
    const index = DenseIndex.initRam({ dimensions: 2, metric: DenseMetric.Euclidean, valuesPerKey: 2 });
    index.reserve(2);
    index.insert(7n, rows(2, 1, 2, 3, 4));
    assert.equal(index.count(7n), 2);
  });

  await t.test("keys that name neither one key nor one per row are refused", () => {
    const index = DenseIndex.initRam({ dimensions: 2, metric: DenseMetric.Euclidean });
    index.reserve(3);
    assert.throws(() => index.insert([1n, 2n, 3n], rows(2, 1, 2, 3, 4)), {
      code: Status.ShapeMismatch,
      message: "The keys name neither one key nor one per row",
    });
  });

  await t.test("search over a batch fills one row per query", () => {
    const index = DenseIndex.initRam({ dimensions: 3, metric: DenseMetric.Euclidean });
    index.reserve(2);
    index.insert([15n, 16n], rows(3, 10, 20, 30, 10, 25, 35));

    const matches = index.search(rows(3, 13, 14, 15, 10, 20, 30), 2, new Arena());

    assert.equal(matches.columns, 2, "the run was written two wide");
    assert.deepEqual(matches.counts, new BigUint64Array([2n, 2n]), "both queries found both keys");
    assert.deepEqual(matches.keys, keysFromIntegers(15n, 16n, 15n, 16n));
    assertAlmostEqual(matches.get(1).scores[0], 0, 1e-3);

    const none = index.search(rows(3, 13, 14, 15, 10, 20, 30), 0, new Arena());
    assert.deepEqual(none.counts, new BigUint64Array([0n, 0n]), "a zero wanted lands only zeroed counts");
    assert.equal(none.get(1).count, 0, "and no keys behind them");
  });
});

test("Ranked runs land in the arena they are handed", async (t) => {
  const index = DenseIndex.initRam({ dimensions: 2, metric: DenseMetric.Euclidean });
  index.reserve(2);
  index.insert([1n, 2n], rows(2, 1, 1, 2, 2));

  await t.test("a verb handed no arena is refused rather than given one", () => {
    assert.throws(() => index.search(new Float32Array([1, 1]), 1), {
      code: Status.InvalidArgument,
      message: "The 'arena' argument must be an Arena",
    });
  });

  await t.test("the next call rewinds the arena, so the earlier view refuses and its arrays read empty", () => {
    const arena = new Arena();
    const first = index.search(new Float32Array([1, 1]), 2, arena);
    const taken = first.keys;
    const second = index.search(new Float32Array([2, 2]), 2, arena);

    assert(second instanceof Matches);
    assert.throws(() => first.keys, {
      code: Status.Uninitialized,
      message: "The arena was rewound or freed since this view was written.",
    });
    assert.equal(taken.length, 0, "an array taken out earlier reads nothing rather than the next call's matches");
    assert.equal(usearch.integerFromKey(usearch.keyAt(second.keys, 0)), 2n);
  });
});

test("Rows are checked against the array that holds them", async (t) => {
  const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 8 });
  index.reserve(8);

  await t.test("a row narrower than the index is refused, not read past", () => {
    assert.throws(() => index.insert(42n, new Float32Array([0.2, 0.6, 0.4])), { code: Status.ShapeMismatch });
    assert.throws(() => index.search(new Float32Array([0.2, 0.6, 0.4]), 1, new Arena()), {
      code: Status.ShapeMismatch,
    });
  });

  await t.test("a run claiming more rows than its array holds is refused before it crosses", () => {
    const overreaching = new RowsView(new Float32Array(8), 2, 8, 32, Scalar.F32);
    assert.throws(() => index.insert([1n, 2n], overreaching), {
      code: Status.ShapeMismatch,
      message: "The rows reach past the end of their array",
    });
  });

  await t.test("an array that splits into no whole rows is refused", () => {
    assert.throws(() => RowsView.of(new Float32Array(5), 2), {
      code: Status.ShapeMismatch,
      message: "The array does not split into rows of 2 f32 values.",
    });
  });
});

test("Receivers choose where a verb runs, and the verb stays the same", async (t) => {
  const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 3 });
  index.reserve(2);
  index.insert([42n, 43n], rows(3, 0.2, 0.6, 0.4, 0.9, 0.1, 0.1));
  const query = new Float32Array([0.2, 0.6, 0.4]);

  await t.test("the calling thread and two executors return the same matches", () => {
    const solo = Executor.forRam({ threads: 1 });
    const here = contents(index.search(query, 2, new Arena()));
    assert.deepEqual(contents(index.on(solo).search(query, 2, new Arena())), here);
    assert.deepEqual(contents(index.on(EXECUTOR).search(query, 2, new Arena())), here);
    solo.free();
  });

  await t.test("a receiver refuses an executor that is not one", () => {
    assert.throws(() => index.on({}).count(42n), {
      code: Status.InvalidArgument,
      message: "The 'executor' argument must be an Executor",
    });
  });

  await t.test("a freed executor is refused rather than read as the calling thread", () => {
    const executor = Executor.forRam({ threads: 1 });
    assert.equal(index.on(executor).count(42n), 1, "lookups run on the executor they are handed");
    executor.free();
    executor.free(); // freeing twice is a no-op
    assert.throws(() => index.on(executor).get(42n, new Arena()), {
      code: Status.Uninitialized,
      message: "The Executor was already freed",
    });
    assert.throws(() => new Arena(executor), { code: Status.Uninitialized, message: "The Executor was already freed" });
  });
});

test("Deferred calls", async (t) => {
  const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 3 });
  index.reserve(2);
  index.insert([42n, 43n], rows(3, 0.2, 0.6, 0.4, 0.9, 0.1, 0.1));
  const query = new Float32Array([0.2, 0.6, 0.4]);

  await t.test("a submission stays deferred until it is awaited, then answers its result", async () => {
    const executor = Executor.forRam();
    const call = index.deferred(executor).search(query, 2, new Arena());
    assert(call instanceof usearch.Deferred);
    assert.equal(call.status, Status.Deferred);

    const matches = await call;
    assert.equal(call.status, Status.Success);
    assert.deepEqual(
      contents(matches),
      contents(index.search(query, 2, new Arena())),
      "the same matches the calling thread finds",
    );
    assert.equal(call.wait(), matches, "waiting again answers the cached result");
    assert.equal(await call, matches, "and so does awaiting again");
    executor.free();
  });

  await t.test("every verb a deferred receiver offers answers what its blocking twin does", async () => {
    const executor = Executor.forRam();
    const deferred = index.deferred(executor);
    const calls = {
      count: deferred.count(42n),
      describe: deferred.describe(),
      health: deferred.health(),
      get: deferred.get(43n, new Arena()),
      refine: deferred.refine(),
    };
    assert.equal(await calls.count, 1);
    assert.deepEqual(await calls.describe, index.describe());
    const health = await calls.health;
    assert.deepEqual([health.audited, health.reachable], [2, 2]);
    assert(health.sweeps >= 1, "an audit sweeps at least once");
    const stored = (await calls.get).data;
    [0.9, 0.1, 0.1].forEach((expected, column) => assertAlmostEqual(stored[column], expected, 2 ** -7));
    assert.equal(await calls.refine, 0);
    executor.free();
  });

  await t.test("a failure arrives through wait and await alike, as the same error", async () => {
    const executor = Executor.forRam();
    const call = index.deferred(executor).insert(44n, new Float32Array([0.2, 0.6]));
    let thrown;
    try {
      call.wait();
    } catch (error) {
      thrown = error;
    }
    assert.equal(thrown.code, Status.ShapeMismatch);
    assert.equal(call.status, Status.ShapeMismatch);
    await assert.rejects(
      async () => await call,
      (error) => error === thrown,
    );
    executor.free();
  });

  await t.test("a deferred call borrows its arena, and a call naming it meanwhile is contended", () => {
    const executor = Executor.forRam();
    const arena = new Arena();
    const call = index.deferred(executor).search(query, 2, arena);
    assert.throws(() => index.search(query, 2, arena), {
      code: Status.Contended,
      message: "The arena serves another call",
    });
    assert.throws(() => index.deferred(executor).search(query, 1, arena).wait(), { code: Status.Contended });
    const matches = call.wait();
    assert.deepEqual(matches.counts, new BigUint64Array([2n]), "the refused calls left the holder's views readable");
    assert.equal(index.search(query, 1, arena).columns, 1, "the arena serves the next call once the first is final");
    executor.free();
  });

  await t.test("a rewind or a free retires a deferred result nobody read yet", () => {
    const stale = {
      code: Status.Uninitialized,
      message: "The arena was rewound or freed since this call wrote into it",
    };
    const executor = Executor.forRam();
    const arena = new Arena();
    const rewound = index.deferred(executor).search(query, 2, arena);
    executor.drain();
    assert.equal(index.search(query, 1, arena).columns, 1, "the call after it rewinds the arena");
    assert.throws(() => rewound.wait(), stale);
    const freed = index.deferred(executor).search(query, 2, arena);
    arena.free();
    assert.equal(freed.status, Status.Success, "the free waited for the call");
    assert.throws(() => freed.wait(), stale);
    executor.free();
  });

  await t.test("a dropped call keeps its inputs alive until it runs", () => {
    const executor = Executor.forRam();
    const target = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 3 });
    target.reserve(1);
    (() => target.deferred(executor).insert(7n, new Float32Array([1, 2, 3])))();
    collect();
    executor.drain();
    assert.deepEqual(target.get(7n, new Arena()).data, new Float32Array([1, 2, 3]));
    executor.free();
  });

  await t.test("disposing a call waits for it, as `using` does at the end of its block", () => {
    const executor = Executor.forRam();
    const call = index.deferred(executor).count(43n);
    call[Symbol.dispose]();
    assert.equal(call.status, Status.Success);
    assert.equal(call.wait(), 1);
    executor.free();
  });
});

test("Several entries under one key", async (t) => {
  const index = DenseIndex.initRam({
    metric: DenseMetric.Euclidean,
    connectivity: 16,
    dimensions: 3,
    valuesPerKey: 3,
  });
  index.reserve(2);
  index.insert([42n, 42n], rows(3, 0.1, 0.6, 0.4, 0.2, 0.6, 0.4));

  await t.test("the count is the whole extent", () => {
    assert.equal(index.count(42n), 2);
    assert.equal(index.contains(42n), true);
  });

  await t.test("every entry exports, in insertion order", () => {
    const stored = index.get(42n, new Arena());
    assert.equal(stored.rows, 2, "two rows of three dimensions");
    assertAlmostEqual(stored.data[0], 0.1, 1e-3);
    assertAlmostEqual(stored.data[3], 0.2, 1e-3);
  });
});

test("Capacity and clearing", () => {
  const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 3 });
  index.reserve(128);
  assert(index.describe().capacity >= 128, "the reservation must hold");

  index.insert(42n, new Float32Array([0.2, 0.6, 0.4]));
  assert.equal(index.describe().size, 1);
  assert(index.describe().memoryUsage > 0, "a populated index occupies memory");

  index.clear();
  assert.equal(index.describe().size, 0, "clearing erases every entry");
  assert.equal(index.contains(42n), false);
});

test("Refinement leaves the graph searchable", () => {
  const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 4 });
  const keys = [];
  const values = [];
  for (let entry = 0; entry < 64; entry++) {
    keys.push(BigInt(entry));
    values.push(entry, entry + 1, entry + 2, entry + 3);
  }
  index.reserve(64);
  index.on(EXECUTOR).insert(keys, rows(4, ...values));

  const changed = index.on(EXECUTOR).refine({ expansionFactor: 64 });
  assert.equal(typeof changed, "number", "refinement reports the nodes it touched");
  assert.equal(index.describe().size, 64, "refinement adds and removes nothing");
  assert.equal(usearch.integerFromKey(index.search(new Float32Array([7, 8, 9, 10]), 1, new Arena()).get(0).keys), 7n);
});

test("Absorbing consumes only the donors it absorbed", () => {
  const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 2 });
  const donor = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 2 });
  index.reserve(2);
  donor.reserve(1);
  index.insert(1n, new Float32Array([1, 1]));
  donor.insert(2n, new Float32Array([2, 2]));

  index.absorb([donor]);

  assert.equal(index.count(2n), 1, "the donor's entries joined the index");
  assert.throws(() => donor.describe(), {
    code: Status.Uninitialized,
    message: "The index was freed or absorbed into another",
  });
  donor.free(); // a consumed donor holds nothing left to free
  assert.throws(() => index.absorb([donor]), {
    code: Status.Uninitialized,
    message: "A donor was freed or absorbed already",
  });
  assert.throws(() => index.absorb([EXECUTOR]), {
    code: Status.InvalidArgument,
    message: "The 'donors' argument must hold indexes",
  });

  const twice = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 2 });
  assert.throws(() => index.absorb([twice, twice]), {
    code: Status.Uninitialized,
    message: "A donor was freed or absorbed already",
  });
  assert.equal(twice.describe().size, 0, "a refused absorb hands its donors back");

  const other = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 2, quantization: Scalar.I8 });
  other.reserve(1);
  other.insert(3n, new Float32Array([3, 3]));
  try {
    index.absorb([other]);
  } catch (error) {
    // The engine refuses another scalar, which the stand-in absorbs, and hands the donor back live.
    assert.equal(error.code, Status.ConfigMismatch);
    assert.equal(other.count(3n), 1, "a donor the call did not absorb stays live");
  }
});

test("A deferred absorb holds its donors until it is final", () => {
  const executor = Executor.forRam();
  const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 2 });
  const donor = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 2 });
  index.reserve(1);
  donor.reserve(1);
  donor.insert(2n, new Float32Array([2, 2]));

  const call = index.deferred(executor).absorb([donor]);
  assert.throws(() => donor.count(2n), {
    code: Status.Uninitialized,
    message: "The index was freed or absorbed into another",
  });
  assert.throws(() => index.deferred(executor).absorb([donor]), {
    code: Status.Uninitialized,
    message: "A donor was freed or absorbed already",
  });
  assert.equal(call.wait(), undefined);
  assert.equal(index.count(2n), 1, "the donor's entries joined the index");
  executor.free();
});

test("Late interaction over multi-row entries", () => {
  const index = DenseIndex.initRam({ metric: DenseMetric.Inner, dimensions: 4, valuesPerKey: 4 });
  index.reserve(4);
  index.insert([7n, 7n, 8n, 8n], rows(4, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1));

  const matches = index.maxsim(rows(4, 1, 0, 0, 0, 0, 1, 0, 0), 2, 2, new Arena());

  assert.equal(matches.queries, 1, "two token rows make one query");
  assert.equal(matches.columns, 2);
  assert.deepEqual(matches.counts, new BigUint64Array([2n]), "both keys scored");
  assert.equal(usearch.integerFromKey(usearch.keyAt(matches.keys, 0)), 7n);
  assert.throws(() => index.maxsim(rows(4, 1, 0, 0, 0, 0, 1, 0, 0), 3, 2, new Arena()), {
    code: Status.ShapeMismatch,
    message: "The rows do not split into whole queries.",
  });
});

test("Sparse products against the indexed rows", async (t) => {
  const populated = () => {
    const index = DenseIndex.initRam({ metric: DenseMetric.Inner, dimensions: 4 });
    index.reserve(3);
    index.insert([1n, 2n, 3n], rows(4, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0));
    return index;
  };

  await t.test("every left row emits the pairs the walk scored", () => {
    const products = populated().multiply(rows(4, 1, 0, 0, 0, 0, 1, 0, 0), new Arena());

    assert.equal(products.rows, 2, "one emitted run per left row");
    assert.equal(products.offsets.length, 3, "Arrow offsets close the last row");
    assert.equal(products.offsets[2], BigInt(products.scores.length), "the last offset closes every pair");
    assert(products.scores instanceof Float64Array, "products are doubles");
    assert(products.required >= Number(products.offsets[2]), "the walk scored at least what it kept");

    // Pairs arrive in traversal order, so the first row is searched rather than indexed.
    const first = products.get(0);
    assert(first.count > 0, "the first row emitted at least one pair");
    const columns = [];
    for (let pair = 0; pair < first.count; pair++)
      columns.push(usearch.integerFromKey(usearch.keyAt(first.keys, pair)));
    assert(columns.includes(1n), "the column the first row overlaps is among them");
  });

  await t.test("a pair bound keeps the first pairs each walk scored", () => {
    const products = populated()
      .on(EXECUTOR)
      .multiply(new Float32Array([1, 0, 0, 0]), new Arena(), { wanted: 1 });
    assert.equal(products.get(0).count, 1, "one pair kept");
    assert(products.required >= 1, "however many the walk scored");
  });

  await t.test("a graph frozen under a distance kind is refused", () => {
    const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 4 });
    index.reserve(1);
    index.insert(1n, new Float32Array([1, 0, 0, 0]));
    assert.throws(() => index.multiply(new Float32Array([1, 0, 0, 0]), new Arena()), { code: Status.ConfigMismatch });
  });
});

test("Attention folds every key the walk scores", async (t) => {
  const dimensions = 4;
  const index = DenseIndex.initRam({ metric: DenseMetric.Inner, dimensions });
  // Every key, value and query is one a bf16 holds exactly, so the stored rows are these.
  const keys = [
    new Float32Array([1, 0, 0, 0]),
    new Float32Array([0.5, 0.75, 0, 0]),
    new Float32Array([0, 1, 0, 0]),
    new Float32Array([-0.75, 0.5, 0, 0]),
  ];
  index.reserve(4);
  index.insert([0n, 1n, 2n, 3n], RowsView.of(Float32Array.from(keys.flatMap((key) => [...key])), dimensions));
  const values = [
    new Float32Array([10, 20, 0, 1]),
    new Float32Array([30, 40, 1, 0]),
    new Float32Array([50, 60, 0, 1]),
    new Float32Array([70, 80, 1, 0]),
  ];
  const valueRows = RowsView.of(bf16Words(values.flatMap((value) => [...value])), dimensions, Scalar.BF16);
  const rowOf = (key) => Number(usearch.integerFromKey(key));
  // Exact attention of `query` over every key, at the derived scale of one over the root of the width.
  const attention = (query) => {
    const logits = keys.map((key) => key.reduce((sum, scalar, position) => sum + scalar * query[position], 0) / 2);
    const maximum = Math.max(...logits);
    const weights = logits.map((logit) => Math.exp(logit - maximum));
    const partition = weights.reduce((sum, weight) => sum + weight, 0);
    const mixed = values[0].map(
      (_, column) => weights.reduce((sum, weight, key) => sum + weight * values[key][column], 0) / partition,
    );
    return { mixed, maximum, partition };
  };
  const queries = [
    new Float32Array([1, 0, 0, 0]),
    new Float32Array([0, 1, 0, 0]),
    new Float32Array([0.5, 0.75, 0, 0]),
    new Float32Array([-1, 0, 0, 0]),
  ];
  const queryRows = RowsView.of(Float32Array.from(queries.flatMap((query) => [...query])), dimensions);

  for (const headCount of [1, 2]) {
    await t.test(`${headCount} heads per walk weigh every key as exact attention does`, () => {
      const carry = new Float32Array(queries.length * 2);
      const stats = new usearch.DenseTraversalStats();
      const outputs = index.attend(queryRows, valueRows, rowOf, new Arena(), { headCount, carry, stats });

      assert.deepEqual([outputs.rows, outputs.columns, outputs.scalar], [queries.length, dimensions, Scalar.F32]);
      queries.forEach((query, row) => {
        const { mixed, maximum, partition } = attention(query);
        mixed.forEach((scalar, column) => assertAlmostEqual(outputs.data[row * dimensions + column], scalar, 1e-3));
        assertAlmostEqual(carry[row * 2], maximum, 1e-5);
        assertAlmostEqual(carry[row * 2 + 1], partition, 1e-5);
      });
      assert.equal(typeof stats.computedDistances, "number", "attention reports the walk it paid for");
    });
  }

  await t.test("every key the walk scores resolves through the caller's function", () => {
    const asked = new Set();
    index.attend(queryRows, valueRows, (key) => (asked.add(rowOf(key)), rowOf(key)), new Arena());
    assert.deepEqual([...asked].sort(), [0, 1, 2, 3]);
  });

  await t.test("a key resolving to no row of the values refuses the call", () => {
    for (const answer of [-1, values.length, 0.5, undefined])
      assert.throws(() => index.attend(queryRows, valueRows, () => answer, new Arena()), {
        code: Status.ShapeMismatch,
      });
  });

  await t.test("values in another scalar kind refuse the call", () => {
    const widened = RowsView.of(Float64Array.from(valueRows.data), dimensions);
    assert.throws(() => index.attend(queryRows, widened, rowOf, new Arena()), { code: Status.ShapeMismatch });
  });

  await t.test("the resolver's first exception stops every later lookup and outranks the refusal", () => {
    let calls = 0;
    const failing = () => {
      calls++;
      throw new RangeError("no such token");
    };
    assert.throws(() => index.attend(queryRows, valueRows, failing, new Arena()), {
      name: "RangeError",
      message: "no such token",
    });
    assert.equal(calls, 1);
  });

  await t.test("a carry that holds no maximum and partition per query is refused", () => {
    assert.throws(() => index.attend(queryRows, valueRows, rowOf, new Arena(), { carry: new Float32Array(3) }), {
      code: Status.ShapeMismatch,
      message: "The carry must hold a maximum and a partition per query",
    });
  });

  await t.test("the attended rows land in the kind asked for", () => {
    const outputs = index.attend(queryRows, valueRows, rowOf, new Arena(), { scalar: Scalar.F64 });
    assert(outputs.data instanceof Float64Array);
    assertAlmostEqual(outputs.data[0], attention(queries[0]).mixed[0], 1e-3);
  });
});

test("Cursors walk every entry once, and keep their index alive", () => {
  const opened = () => {
    const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 2 });
    index.reserve(3);
    index.insert([1n, 2n, 3n], rows(2, 1, 1, 2, 2, 3, 3));
    return index.on(EXECUTOR).cursor();
  };
  const cursor = opened();
  collect(); // nothing but the cursor refers to the index now

  const arena = new Arena();
  const pull = () => cursor.next(2, arena, { vectors: Scalar.F32 });
  const seen = [];
  for (let batch = pull(); batch.rows; batch = pull()) {
    for (let position = 0; position < batch.rows; position++) {
      const key = usearch.integerFromKey(usearch.keyAt(batch.keys, position));
      seen.push(key);
      assert.deepEqual(
        batch.vectors.data.subarray(position * 2, position * 2 + 2),
        new Float32Array([Number(key), Number(key)]),
      );
    }
  }
  cursor.free();
  cursor.free(); // releasing twice is a no-op
  assert.throws(() => cursor.next(2, arena), { code: Status.Uninitialized, message: "The walk was already released" });

  const keysAlone = opened().next(3, arena);
  assert.deepEqual([keysAlone.rows, keysAlone.keys.length, keysAlone.vectors], [3, 3 * usearch.keyBytes, undefined]);

  seen.sort((first, second) => Number(first - second));
  assert.deepEqual(seen, [1n, 2n, 3n]);
});

test("Search streams pull matches lazily", () => {
  const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 2 });
  index.reserve(3);
  index.insert([1n, 2n, 3n], rows(2, 1, 1, 2, 2, 3, 3));

  let stream;
  if (!served(() => (stream = index.on(EXECUTOR).searchStream(new Float32Array([1, 1]))))) return;
  const arena = new Arena();

  const nearest = stream.next(1, arena);
  assert.deepEqual(nearest.counts, new BigUint64Array([1n]), "one match per pull");
  assert.equal(usearch.integerFromKey(nearest.get(0).keys), 1n, "the nearest comes first");
  assert.deepEqual(
    stream.next(8, arena).counts,
    new BigUint64Array([2n]),
    "the next pull resumes where the first stopped",
  );
  assert.deepEqual(stream.next(8, arena).counts, new BigUint64Array([0n]), "a drained walk writes nothing");
  stream[Symbol.dispose]();
  assert.throws(() => stream.next(8, arena), {
    code: Status.Uninitialized,
    message: "The walk was already released",
  });
});

test("Serialization", async (t) => {
  const indexPath = path.join(os.tmpdir(), "usearch.test.safegraphs");

  t.beforeEach(() => {
    const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 3 });
    index.reserve(1);
    index.insert(42n, new Float32Array([0.2, 0.6, 0.4]));
    index.on(EXECUTOR).save(indexPath);
  });

  t.afterEach(() => {
    fs.unlinkSync(indexPath);
  });

  await t.test("loadRam", () => {
    const index = DenseIndex.loadRam(indexPath, { executor: EXECUTOR });
    const found = index.search(new Float32Array([0.2, 0.6, 0.4]), 10, new Arena()).get(0);

    assert.equal(index.describe().size, 1);
    assert.deepEqual(found.keys, usearch.keyFromInteger(42n));
    assertAlmostEqual(found.scores[0], 0);
  });

  await t.test("viewRam serves the checkpoint read-only", () => {
    let index;
    if (!served(() => (index = DenseIndex.viewRam(indexPath)))) return;

    assert.deepEqual(
      index.search(new Float32Array([0.2, 0.6, 0.4]), 1, new Arena()).get(0).keys,
      usearch.keyFromInteger(42n),
    );
    assert.throws(() => index.insert(43n, new Float32Array([0.2, 0.6, 0.4])), { code: Status.ReadOnly });
  });

  await t.test("metadataRead", () => {
    assert.deepEqual(DenseIndex.metadataRead(indexPath), {
      metric: DenseMetric.Euclidean,
      quantization: Scalar.BF16,
      dimensions: 3,
      connectivity: 16,
    });
  });
});

test("Serialization through memory", async (t) => {
  const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 3 });
  index.reserve(1);
  index.insert(42n, new Float32Array([0.2, 0.6, 0.4]));

  const monolith = index.on(EXECUTOR).saveBuffer(new Arena());
  assert(monolith instanceof usearch.Monolith && monolith.bytes.length > 0, "the monolith views its arena");

  await t.test("loadRamBuffer", () => {
    const restored = DenseIndex.loadRamBuffer(monolith, { executor: EXECUTOR });

    assert.equal(restored.describe().size, 1);
    assert.deepEqual(
      restored.search(new Float32Array([0.2, 0.6, 0.4]), 1, new Arena()).get(0).keys,
      usearch.keyFromInteger(42n),
    );
  });

  await t.test("metadataReadBuffer", () => {
    assert.deepEqual(DenseIndex.metadataReadBuffer(monolith.bytes), {
      metric: DenseMetric.Euclidean,
      quantization: Scalar.BF16,
      dimensions: 3,
      connectivity: 16,
    });
  });
});

test("Every enumerator crosses the C name tables", async (t) => {
  await t.test("metrics", () => {
    for (const metric of Object.values(DenseMetric)) {
      const bits = metric === DenseMetric.Hamming || metric === DenseMetric.Jaccard;
      const index = DenseIndex.initRam({ metric, dimensions: 8, quantization: bits ? Scalar.U1x8 : Scalar.BF16 });
      assert.equal(index.describe().metadata.metric, metric);
    }
  });

  await t.test("scalar kinds", () => {
    const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 8 });
    index.reserve(1);
    index.insert(1n, new Float32Array(8).fill(1));
    for (const scalar of Object.values(Scalar).filter((scalar) => scalar !== Scalar.I32))
      assert.equal(index.get(1n, new Arena(), { scalar }).scalar, scalar);
    assert.throws(() => index.get(1n, new Arena(), { scalar: Scalar.I32 }), { code: Status.UnknownKind });
  });

  await t.test("moods", () => {
    const executor = Executor.forRam({ threads: 1, deferredLimit: 4 });
    for (const mood of Object.values(usearch.Mood)) executor.drain(mood);
    assert.throws(() => executor.drain("restless"), {
      code: Status.InvalidArgument,
      message: "The 'mood' argument names nothing: 'restless'",
    });
    executor.free();
  });

  await t.test("a misspelled metric is refused rather than read as a default", () => {
    assert.throws(() => DenseIndex.initRam({ metric: "cosine", dimensions: 8 }), {
      code: Status.InvalidArgument,
      message: "The 'metric' argument names nothing: 'cosine'",
    });
  });
});

test("Mini-float quantization types", async (t) => {
  for (const quantization of [Scalar.E5M2, Scalar.E4M3, Scalar.E3M2, Scalar.E2M3]) {
    await t.test(`insert and search with ${quantization}`, (t) => {
      const index = initRamOrSkip(t, { metric: DenseMetric.Angular, dimensions: 64, quantization });
      if (!index) return;
      const vector = new Float32Array(64);
      for (let position = 0; position < 64; position++) vector[position] = position * 0.1;
      index.reserve(1);
      index.insert(1n, vector);

      assert.equal(usearch.integerFromKey(index.search(vector, 1, new Arena()).get(0).keys), 1n);
    });
  }
});

test("A scan scores every row exactly, so each row finds itself first", () => {
  const index = DenseIndex.initRam({ dimensions: 4 });
  const stored = rows(4, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0);
  index.reserve(3);
  index.insert([1n, 2n, 3n], stored);
  const matches = index.search(stored, 3, new Arena(), { traversal: usearch.Traversal.Scan });
  for (let query = 0; query < 3; query++)
    assert.equal(usearch.integerFromKey(usearch.keyAt(matches.get(query).keys, 0)), BigInt(query + 1));
  assert.throws(() => index.search(stored, 3, new Arena(), { traversal: "sweep" }), {
    code: Status.InvalidArgument,
    message: "The 'traversal' argument names nothing: 'sweep'",
  });
});

test("Input scalar kinds beyond the four a TypedArray spells", async (t) => {
  await t.test("packed bits reach a Hamming index as u1x8", () => {
    const index = DenseIndex.initRam({ metric: DenseMetric.Hamming, dimensions: 64, quantization: Scalar.U1x8 });
    const packed = new Uint8Array([0b10101010, 0, 0, 0, 0, 0, 0, 0, 0b10101011, 0, 0, 0, 0, 0, 0, 0]);
    index.reserve(2);
    index.insert([1n, 2n], RowsView.of(packed, 64, Scalar.U1x8));

    const found = index.search(RowsView.of(packed.subarray(0, 8), 64, Scalar.U1x8), 2, new Arena()).get(0);
    assert.equal(usearch.integerFromKey(usearch.keyAt(found.keys, 0)), 1n);
    assertAlmostEqual(found.scores[0], 0, 1e-3);
    assertAlmostEqual(found.scores[1], 1, 1e-3);
  });

  await t.test("bf16 rows travel as a Uint16Array with their kind named", () => {
    const index = DenseIndex.initRam({ metric: DenseMetric.Angular, dimensions: 4, quantization: Scalar.BF16 });
    // The bf16 encoding of 1.0 is 0x3f80.
    const row = RowsView.of(new Uint16Array([0x3f80, 0x3f80, 0x3f80, 0x3f80]), 4, Scalar.BF16);
    index.reserve(1);
    index.insert(1n, row);

    assert.equal(usearch.integerFromKey(index.search(row, 1, new Arena()).get(0).keys), 1n);
    assert.deepEqual(
      index.get(1n, new Arena(), { scalar: Scalar.BF16 }).data,
      new Uint16Array([0x3f80, 0x3f80, 0x3f80, 0x3f80]),
    );
  });

  await t.test("a Uint16Array without a named kind is refused", () => {
    const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 4 });
    index.reserve(1);
    assert.throws(() => index.insert(1n, new Uint16Array([1, 2, 3, 4])), {
      code: Status.UnknownKind,
      message: "The rows spell no scalar kind on their own; name one in a RowsView.",
    });
  });

  await t.test("f64 rows keep their width on the way in, though no index stores them so", () => {
    for (const quantization of [Scalar.F64, Scalar.F32])
      assert.throws(() => DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 2, quantization }), {
        code: Status.UnknownKind,
      });
    const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 2 });
    index.reserve(1);
    index.insert(1n, new Float64Array([10, 20]));

    assertAlmostEqual(index.search(new Float64Array([10, 20]), 1, new Arena()).get(0).scores[0], 0, 1e-9);
  });

  await t.test(
    "f16 rows travel as a Float16Array where the runtime has one",
    { skip: typeof Float16Array !== "function" },
    (t) => {
      const index = initRamOrSkip(t, { metric: DenseMetric.Euclidean, dimensions: 2, quantization: Scalar.F16 });
      if (!index) return;
      index.reserve(1);
      index.insert(2n, new Float16Array([30, 40]));

      assert.deepEqual(index.get(2n, new Arena(), { scalar: Scalar.F16 }).data, new Float16Array([30, 40]));
    },
  );
});

test("Arenas hold every output whose length scales with the call", async (t) => {
  const index = DenseIndex.initRam({ metric: DenseMetric.Inner, dimensions: 4 });
  index.reserve(2);
  index.insert([1n, 2n], rows(4, 1, 0, 0, 0, 0, 1, 0, 0));

  await t.test("each arena keeps its own call's outputs, past the collector", () => {
    const first = index.get(1n, new Arena());
    const second = index.on(EXECUTOR).get(2n, new Arena(EXECUTOR));
    collect();
    assert.deepEqual(first.data, new Float32Array([1, 0, 0, 0]));
    assert.deepEqual(second.data, new Float32Array([0, 1, 0, 0]));
  });

  await t.test("a reused arena allocates once, and only its latest view reads", () => {
    const arena = new Arena();
    let previous;
    for (const key of [1n, 2n, 1n]) {
      const current = index.get(key, arena);
      assert.equal(current.rows, 1);
      if (previous) assert.throws(() => previous.data, { code: Status.Uninitialized });
      previous = current;
    }
    assert.deepEqual(previous.data, new Float32Array([1, 0, 0, 0]));
  });

  await t.test("freeing an arena refuses every view of it, and every later call", () => {
    const arena = new Arena();
    const held = index.get(1n, arena);
    const taken = held.data;
    arena.free();
    arena.free(); // freeing twice is a no-op
    assert.throws(() => held.data, { code: Status.Uninitialized });
    assert.equal(taken.length, 0, "an array taken out earlier reads nothing rather than freed memory");
    assert.throws(() => index.get(1n, arena), { code: Status.Uninitialized, message: "The Arena was already freed" });
  });

  await t.test("an arena takes the residency of the executor that made it", () => {
    assert.equal(new Arena().residency, usearch.Residency.Ram);
    assert.equal(new Arena(EXECUTOR).residency, usearch.Residency.Ram);
    assert.throws(() => new Arena({}), {
      code: Status.InvalidArgument,
      message: "The 'executor' argument must be an Executor",
    });
  });

  await t.test("the exported kind decides the element type and the byte count", () => {
    assert(index.get(1n, new Arena(), { scalar: Scalar.F64 }).data instanceof Float64Array);
    assert.deepEqual(
      index.get(1n, new Arena(), { scalar: Scalar.U1x8 }).data,
      new Uint8Array([0x80]),
      "four bits pack into one byte, the first dimension in the most significant bit",
    );
    const arena = new Arena();
    const absent = index.get(3n, arena);
    assert.equal(absent.data.length, 0, "an absent key exports nothing");
    index.get(1n, arena);
    assert.throws(() => absent.data, { code: Status.Uninitialized }, "an empty view goes stale like any other");
  });
});

test("Library introspection", () => {
  assert.equal(usearch.version(), "3.0.0");
  assert.equal(typeof usearch.capabilitiesCompiled(), "string");
  assert.equal(typeof usearch.capabilitiesAvailable(), "string");
});

test("Traversal telemetry", async (t) => {
  const populated = () => {
    const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, connectivity: 16, dimensions: 4 });
    const keys = [];
    const values = [];
    for (let entry = 0; entry < 64; entry++) {
      keys.push(BigInt(entry));
      values.push(entry, entry + 1, entry + 2, entry + 3);
    }
    index.reserve(64);
    index.insert(keys, rows(4, ...values));
    return index;
  };

  const assertCounters = (stats, note) => {
    for (const counter of ["visitedMembers", "computedDistances", "restarts"]) {
      assert(Number.isInteger(stats[counter]) && stats[counter] >= 0, `${note}: ${counter} is a whole count`);
    }
  };

  await t.test("every walking verb reports into the reading it is given", () => {
    const index = populated();
    for (const [verb, run] of [
      ["insert", (stats) => index.insert(99n, new Float32Array([1, 2, 3, 4]), { stats })],
      ["refine", (stats) => index.on(EXECUTOR).refine({ expansionFactor: 64, stats })],
      ["search", (stats) => index.search(new Float32Array([7, 8, 9, 10]), 4, new Arena(), { stats })],
    ]) {
      const stats = new usearch.DenseTraversalStats();
      run(stats);
      assertCounters(stats, verb);
    }
  });

  await t.test("a deferred call reports into its reading once it settles", async () => {
    const executor = Executor.forRam();
    const stats = new usearch.DenseTraversalStats();
    const call = populated()
      .deferred(executor)
      .search(new Float32Array([7, 8, 9, 10]), 4, new Arena(), { stats });
    assert.deepEqual({ ...stats }, { visitedMembers: 0, computedDistances: 0, restarts: 0 });
    await call;
    assert(stats.computedDistances > 0, "the walk it paid for lands with its result");
    executor.free();
  });

  await t.test("one reading holds the last call, not the sum of them", () => {
    const index = populated();
    const query = new Float32Array([7, 8, 9, 10]);
    const once = new usearch.DenseTraversalStats();
    index.search(query, 4, new Arena(), { stats: once });

    const reused = new usearch.DenseTraversalStats();
    index.search(query, 4, new Arena(), { stats: reused });
    index.search(query, 4, new Arena(), { stats: reused });
    assert.deepEqual({ ...reused }, { ...once });
  });

  await t.test("accumulate folds one reading into another", () => {
    const total = new usearch.DenseTraversalStats();
    const reading = new usearch.DenseTraversalStats();
    reading.visitedMembers = 5;
    reading.computedDistances = 11;
    reading.restarts = 1;
    total.accumulate(reading);
    total.accumulate(reading);
    assert.deepEqual({ ...total }, { visitedMembers: 10, computedDistances: 22, restarts: 2 });
  });
});

test("Reachability audits", () => {
  const index = DenseIndex.initRam({ metric: DenseMetric.Euclidean, dimensions: 2 });
  index.reserve(2);
  index.insert([1n, 2n], rows(2, 1, 1, 2, 2));
  const health = index.on(EXECUTOR).health();
  assert.deepEqual([health.audited, health.reachable], [2, 2]);
  assert(health.sweeps >= 1, "an audit sweeps at least once");
});
