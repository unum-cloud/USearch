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

const THREADS = 2;

test("Single-entry operations", async (t) => {
  await t.test("index info", () => {
    const index = new usearch.Index(2, "l2sq");

    assert.equal(index.connectivity(), 16, "connectivity should be 16");
    assert.equal(index.dimensions(), 2, "dimensions should be 2");
    assert.equal(index.size(), 0, "initial size should be 0");
  });

  await t.test("add", () => {
    const index = new usearch.Index(2, "l2sq");

    index.add(15n, new Float32Array([10, 20]));
    index.add(16n, new Float32Array([10, 25]));

    assert.equal(index.size(), 2, "size after adding elements should be 2");
    assert.equal(
      index.contains(15),
      true,
      "entry must be present after insertion",
    );

    const results = index.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(
      results.keys,
      new BigUint64Array([15n, 16n]),
      "keys should be 15 and 16",
    );
    assert.deepEqual(
      results.distances,
      new Float32Array([45, 130]),
      "distances should be 45 and 130",
    );
  });

  await t.test("remove", () => {
    const index = new usearch.Index(2, "l2sq");

    index.add(15n, new Float32Array([10, 20]));
    index.add(16n, new Float32Array([10, 25]));
    index.add(25n, new Float32Array([20, 40]));
    index.add(26n, new Float32Array([20, 45]));

    assert.equal(index.remove(15n), 1);

    assert.equal(index.size(), 3, "size after removing elements should be 3");
    assert.equal(
      index.contains(15),
      false,
      "entry must be absent after insertion",
    );

    const results = index.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(
      results.keys,
      new BigUint64Array([16n, 25n]),
      "keys should not include 15",
    );
  });
});

test("Batch operations", async (t) => {
  await t.test("add", () => {
    const indexBatch = new usearch.Index(2, "l2sq");

    const keys = [15n, 16n];
    const vectors = [new Float32Array([10, 20]), new Float32Array([10, 25])];

    indexBatch.add(keys, vectors);
    assert.equal(indexBatch.size(), 2, "size after adding batch should be 2");

    const results = indexBatch.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(
      results.keys,
      new BigUint64Array([15n, 16n]),
      "keys should be 15 and 16",
    );
    assert.deepEqual(
      results.distances,
      new Float32Array([45, 130]),
      "distances should be 45 and 130",
    );
  });

  await t.test("remove", () => {
    const indexBatch = new usearch.Index(2, "l2sq");

    const keys = [15n, 16n, 25n, 26n];
    const vectors = [
      new Float32Array([10, 20]),
      new Float32Array([10, 25]),
      new Float32Array([20, 40]),
      new Float32Array([20, 45]),
    ];
    indexBatch.add(keys, vectors);

    assert.deepEqual(indexBatch.remove([15n, 25n]), [1, 1]);
    assert.equal(indexBatch.size(), 2, "size after removing batch should be 2");

    const results = indexBatch.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(
      results.keys,
      new BigUint64Array([16n, 26n]),
      "keys should not include 15 and 25",
    );
  });

  await t.test("search", () => {
    const indexBatch = new usearch.Index(3, "l2sq");

    const keys = [15n, 16n];
    const vectors = [
      new Float32Array([10, 20, 30]),
      new Float32Array([10, 25, 35]),
    ];

    indexBatch.add(keys, vectors);
    assert.equal(indexBatch.size(), 2, "size after adding batch should be 2");

    const results = indexBatch.search(
      [new Float32Array([13, 14, 15]), new Float32Array([13, 14, 15])],
      2,
    );

    assert.equal(results.k, 2, "k should be 2");

    assert.deepEqual(
      results.counts,
      new BigUint64Array([2n, 2n]),
      "counts should be 2 for both keys",
    );

    assert.deepEqual(
      results.keys,
      new BigUint64Array([15n, 16n, 15n, 16n]),
      "keys should be 15 and 16",
    );
    assert.deepEqual(
      results.distances,
      new Float32Array([270, 530, 270, 530]),
      "distances should be 270 and 530",
    );
  });
});

test("Malformed vector batches", async (t) => {
  const newIndex = () => {
    const index = new usearch.Index({
      metric: "l2sq",
      connectivity: 16,
      dimensions: 3,
    });
    index.add(1n, new Float32Array([1, 0, 0]));
    index.add(2n, new Float32Array([0, 1, 0]));
    return index;
  };
  const notAnArrayOfArrays = /Vectors must be a TypedArray or an array of arrays/;
  const wrongRowLength = /Every vector must have 3 dimensions/;

  await t.test("a flat array of numbers is rejected by `search`", () => {
    // Used to return an empty `BatchMatches` without any error.
    assert.throws(() => newIndex().search([1, 0, 0], 2), notAnArrayOfArrays);
  });

  await t.test("a flat array of numbers is rejected by `add`", () => {
    // Used to fail with a misleading "length of keys (1) must match number of vectors (0)".
    const index = newIndex();
    assert.throws(() => index.add(3n, [0, 0, 1]), notAnArrayOfArrays);
    assert.equal(index.size(), 2);
  });

  await t.test("rows that are not arrays are rejected", () => {
    const index = newIndex();
    assert.throws(() => index.search(["abc"], 1), notAnArrayOfArrays);
    assert.throws(() => index.search([[1, 0, 0], 1], 1), notAnArrayOfArrays);
    assert.throws(() => index.search([null], 1), notAnArrayOfArrays);
  });

  await t.test("ragged rows are rejected even if the total size fits", () => {
    // 2 + 4 elements used to be accepted as two vectors of 3 dimensions.
    const ragged = [
      [1, 0],
      [0, 0, 0, 1],
    ];
    const index = newIndex();
    assert.throws(() => index.search(ragged, 2), wrongRowLength);
    assert.throws(() => index.add([3n, 4n], ragged), wrongRowLength);
    assert.equal(index.size(), 2);
  });

  await t.test("rows of the wrong length are rejected", () => {
    const index = newIndex();
    assert.throws(() => index.search([[1, 0, 0, 0]], 1), wrongRowLength);
    assert.throws(() => index.search([[1, 0]], 1), wrongRowLength);
    assert.throws(() => index.search([new Float32Array([1, 0])], 1), wrongRowLength);
    // A single row holding several vectors is not a vector either.
    assert.throws(() => index.search([[1, 0, 0, 0, 1, 0]], 1), wrongRowLength);
  });

  await t.test("`exactSearch` rejects malformed datasets and queries", () => {
    const dataset = [new Float32Array([1, 0, 0]), new Float32Array([0, 1, 0])];
    assert.throws(
      () => usearch.exactSearch([1, 0, 0, 0, 1, 0], dataset, 3, 1, "l2sq"),
      notAnArrayOfArrays,
    );
    assert.throws(
      () => usearch.exactSearch(dataset, [1, 0, 0], 3, 1, "l2sq"),
      notAnArrayOfArrays,
    );
    assert.throws(
      () => usearch.exactSearch(dataset, [[1, 0]], 3, 1, "l2sq"),
      wrongRowLength,
    );
  });

  await t.test("well-formed batches are still accepted", () => {
    const index = newIndex();
    const asArrays = index.search(
      [
        [1, 0, 0],
        [0, 1, 0],
      ],
      1,
    );
    assert.deepEqual(asArrays.counts, new BigUint64Array([1n, 1n]));
    assert.deepEqual(asArrays.get(0).keys, new BigUint64Array([1n]));
    assert.deepEqual(asArrays.get(1).keys, new BigUint64Array([2n]));

    const asTypedArrays = index.search(
      [new Float32Array([1, 0, 0]), new Int8Array([0, 1, 0])],
      1,
    );
    assert.deepEqual(asTypedArrays.get(0).keys, new BigUint64Array([1n]));
    assert.deepEqual(asTypedArrays.get(1).keys, new BigUint64Array([2n]));

    // A single query in a nested array and a flat typed array still work.
    assert.deepEqual(index.search([[0, 1, 0]], 1).keys, new BigUint64Array([2n]));
    assert.deepEqual(
      index.search(new Float32Array([1, 0, 0, 0, 1, 0]), 1).get(1).keys,
      new BigUint64Array([2n]),
    );

    index.add([3n, 4n], [
      [0, 0, 1],
      [1, 1, 0],
    ]);
    assert.equal(index.size(), 4);
  });
});

test("Expected results", () => {
  const index = new usearch.Index({
    metric: "l2sq",
    connectivity: 16,
    dimensions: 3,
  });
  index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
  const results = index.search(new Float32Array([0.2, 0.6, 0.4]), 10);

  assert.equal(index.size(), 1);
  assert.deepEqual(results.keys, new BigUint64Array([42n]));
  assertAlmostEqual(results.distances[0], new Float32Array([0]));
});

test("Multithread search returns same results", () => {
  const index = new usearch.Index({
    metric: "l2sq",
    connectivity: 16,
    dimensions: 3,
  });
  index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
  assert.equal(index.size(), 1);

  const results_1 = index.search(new Float32Array([0.2, 0.6, 0.4]), 10, 0);
  const results_2 = index.search(
    new Float32Array([0.2, 0.6, 0.4]),
    10,
    THREADS,
  );

  assert.deepEqual(results_1.keys, results_2.keys);
  assertAlmostEqual(results_1.distances, results_2.distances);
});

test("Exact search", async (t) => {
  const dataset = [
    new Float32Array([0.2, 0.6, 0.4]),
    new Float32Array([0.6, 0.6, 0.4]),
  ];
  const queries = new Float32Array([0.2, 0.6, 0.4]);
  const dimensions = 3;
  const count = 2;
  const metric = "l2sq";

  await t.test("Single core search", () => {
    const result = usearch.exactSearch(
      dataset,
      queries,
      dimensions,
      count,
      metric,
      1, // threads
    );

    assert.deepEqual(result.keys, new BigUint64Array([0n, 1n]));
    assertAlmostEqual(new Float32Array([0]), result.distances[0]);
    assertAlmostEqual(new Float32Array([0.16]), result.distances[1]);
  });

  await t.test("Multithreaded search", () => {
    const result = usearch.exactSearch(
      dataset,
      queries,
      dimensions,
      count,
      metric,
      THREADS, // threads
    );

    assert.deepEqual(result.keys, new BigUint64Array([0n, 1n]));
    assertAlmostEqual(new Float32Array([0]), result.distances[0]);
    assertAlmostEqual(new Float32Array([0.16]), result.distances[1]);
  });
});

test("Expected count()", async (t) => {
  const index = new usearch.Index({
    metric: "l2sq",
    connectivity: 16,
    dimensions: 3,
  });
  index.add(
    [42n, 43n],
    [new Float32Array([0.2, 0.6, 0.4]), new Float32Array([0.2, 0.6, 0.4])],
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
});

test("Operations with invalid values", () => {
  const indexBatch = new usearch.Index(2, "l2sq");

  const keys = [NaN, 16n];
  const vectors = [new Float32Array([10, 30]), new Float32Array([1, 5])];

  // All keys must be positive integers or bigints
  assert.throws(() => indexBatch.add(keys, vectors));

  // Vectors must be a TypedArray or an array of arrays
  assert.throws(() => indexBatch.search(NaN, 2));
});

test("Invalid operations", async (t) => {
  await t.test("Add the same keys", () => {
    const index = new usearch.Index({
      metric: "l2sq",
      connectivity: 16,
      dimensions: 3,
    });
    index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
    assert.throws(() => index.add(42n, new Float32Array([0.2, 0.6, 0.4])));
  });

  await t.test("Batch add containing the same key", () => {
    const index = new usearch.Index({
      metric: "l2sq",
      connectivity: 16,
      dimensions: 3,
    });
    index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
    assert.throws(() => {
      index.add(
        [41n, 42n, 43n],
        [
          [0.1, 0.6, 0.4],
          [0.2, 0.6, 0.4],
          [0.3, 0.6, 0.4],
        ],
      );
    });
  });
});

test("Serialization", async (t) => {
  const indexPath = path.join(os.tmpdir(), "usearch.test.index");

  t.beforeEach(() => {
    const index = new usearch.Index({
      metric: "l2sq",
      connectivity: 16,
      dimensions: 3,
    });
    index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
    index.save(indexPath);
  });

  t.afterEach(() => {
    fs.unlinkSync(indexPath);
  });

  await t.test("load", () => {
    const index = new usearch.Index({
      metric: "l2sq",
      connectivity: 16,
      dimensions: 3,
    });
    index.load(indexPath);
    const results = index.search(
      new Float32Array([0.2, 0.6, 0.4]),
      10,
      THREADS,
    );

    assert.equal(index.size(), 1);
    assert.deepEqual(results.keys, new BigUint64Array([42n]));
    assertAlmostEqual(results.distances[0], new Float32Array([0]));
  });

  // todo: Skip as the test fails only on windows.
  // The following error in afterEach().
  // `error: "EBUSY: resource busy or locked, unlink`
  await t.test(
    "view: Read data",
    { skip: process.platform === "win32" },
    () => {
      const index = new usearch.Index({
        metric: "l2sq",
        connectivity: 16,
        dimensions: 3,
      });
      index.view(indexPath);
      const results = index.search(new Float32Array([0.2, 0.6, 0.4]), 10);

      assert.equal(index.size(), 1);
      assert.deepEqual(results.keys, new BigUint64Array([42n]));
      assertAlmostEqual(results.distances[0], new Float32Array([0]));
    },
  );

  await t.test(
    "view: Invalid operations: add",
    { skip: process.platform === "win32" },
    () => {
      const index = new usearch.Index({
        metric: "l2sq",
        connectivity: 16,
        dimensions: 3,
      });
      index.view(indexPath);

      // Can't add to an immutable index
      assert.throws(() => index.add(43n, new Float32Array([0.2, 0.6, 0.4])));
    },
  );

  await t.test(
    "view: Invalid operations: remove",
    { skip: process.platform === "win32" },
    () => {
      const index = new usearch.Index({
        metric: "l2sq",
        connectivity: 16,
        dimensions: 3,
      });
      index.view(indexPath);

      // Can't remove from an immutable index
      assert.throws(() => index.remove(42n));
    },
  );
});

test("Mini-float quantization types", async (t) => {
  for (const quantization of ["e5m2", "e4m3", "e3m2", "e2m3"]) {
    await t.test(`add and search with ${quantization}`, () => {
      const index = new usearch.Index({
        metric: "cos",
        dimensions: 64,
        quantization: quantization,
      });
      const vector = new Float32Array(64);
      for (let i = 0; i < 64; i++) vector[i] = i * 0.1;
      index.add(1n, vector);

      const results = index.search(vector, 1);
      assert.equal(results.keys[0], 1n);
    });
  }
});
