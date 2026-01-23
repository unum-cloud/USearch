const nodeTest = require('node:test');
const realTest = nodeTest.test; // the original function

function loggedTest(name, options, fn) {
  // The API has two call signatures:
  //   test(name, fn)
  //   test(name, options, fn)
  if (typeof options === 'function') {
    fn = options;
    options = undefined;
  }

  // Wrap the body so we can log before / after
  const wrapped = async (t) => {
    console.log('▶', name);
    try {
      await fn(t); // run the user’s test
      console.log('✓', name);
    } catch (err) {
      console.log('✖', name);
      throw err; // re-throw so the runner records the failure
    }
  };

  // Delegate back to the real test() with the same options
  return options ? realTest(name, options, wrapped) : realTest(name, wrapped);
}

// Replace both the export and the global the runner puts on each module
global.test = loggedTest;
module.exports = loggedTest; // for completeness if this file is `require`d

const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const usearch = require('./dist/cjs/usearch.js');

function assertAlmostEqual(actual, expected, tolerance = 1e-6) {
  const lowerBound = expected - tolerance;
  const upperBound = expected + tolerance;
  assert(
    actual >= lowerBound && actual <= upperBound,
    `Expected ${actual} to be almost equal to ${expected}`
  );
}

const THREADS = 2;

test('Single-entry operations', async (t) => {
  await t.test('index info', () => {
    const index = new usearch.Index(2, 'l2sq');

    assert.equal(index.connectivity(), 16, 'connectivity should be 16');
    assert.equal(index.dimensions(), 2, 'dimensions should be 2');
    assert.equal(index.size(), 0, 'initial size should be 0');
  });

  await t.test('add', () => {
    const index = new usearch.Index(2, 'l2sq');

    index.add(15n, new Float32Array([10, 20]));
    index.add(16n, new Float32Array([10, 25]));

    assert.equal(index.size(), 2, 'size after adding elements should be 2');
    assert.equal(
      index.contains(15),
      true,
      'entry must be present after insertion'
    );

    const results = index.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(
      results.keys,
      new BigUint64Array([15n, 16n]),
      'keys should be 15 and 16'
    );
    assert.deepEqual(
      results.distances,
      new Float32Array([45, 130]),
      'distances should be 45 and 130'
    );
  });

  await t.test('remove', () => {
    const index = new usearch.Index(2, 'l2sq');

    index.add(15n, new Float32Array([10, 20]));
    index.add(16n, new Float32Array([10, 25]));
    index.add(25n, new Float32Array([20, 40]));
    index.add(26n, new Float32Array([20, 45]));

    assert.equal(index.remove(15n), 1);

    assert.equal(index.size(), 3, 'size after removing elements should be 3');
    assert.equal(
      index.contains(15),
      false,
      'entry must be absent after insertion'
    );

    const results = index.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(
      results.keys,
      new BigUint64Array([16n, 25n]),
      'keys should not include 15'
    );
  });
});

test('Batch operations', async (t) => {
  await t.test('add', () => {
    const indexBatch = new usearch.Index(2, 'l2sq');

    const keys = [15n, 16n];
    const vectors = [new Float32Array([10, 20]), new Float32Array([10, 25])];

    indexBatch.add(keys, vectors);
    assert.equal(indexBatch.size(), 2, 'size after adding batch should be 2');

    const results = indexBatch.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(
      results.keys,
      new BigUint64Array([15n, 16n]),
      'keys should be 15 and 16'
    );
    assert.deepEqual(
      results.distances,
      new Float32Array([45, 130]),
      'distances should be 45 and 130'
    );
  });

  await t.test('remove', () => {
    const indexBatch = new usearch.Index(2, 'l2sq');

    const keys = [15n, 16n, 25n, 26n];
    const vectors = [
      new Float32Array([10, 20]),
      new Float32Array([10, 25]),
      new Float32Array([20, 40]),
      new Float32Array([20, 45]),
    ];
    indexBatch.add(keys, vectors);

    assert.deepEqual(indexBatch.remove([15n, 25n]), [1, 1]);
    assert.equal(indexBatch.size(), 2, 'size after removing batch should be 2');

    const results = indexBatch.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(
      results.keys,
      new BigUint64Array([16n, 26n]),
      'keys should not include 15 and 25'
    );
  });

  await t.test('search', () => {
    const indexBatch = new usearch.Index(3, 'l2sq');

    const keys = [15n, 16n];
    const vectors = [new Float32Array([10, 20, 30]), new Float32Array([10, 25, 35])];

    indexBatch.add(keys, vectors);
    assert.equal(indexBatch.size(), 2, 'size after adding batch should be 2');

    const results = indexBatch.search([new Float32Array([13, 14, 15]), new Float32Array([13, 14, 15])], 2);
    
    assert.equal(results.k, 2, 'k should be 2');

    assert.deepEqual(
      results.counts,
      new BigUint64Array([ 2n, 2n ]) ,
      'counts should be 2 for both keys'
    );

    assert.deepEqual(
      results.keys,
      new BigUint64Array([15n, 16n, 15n, 16n]),
      'keys should be 15 and 16'
    );
    assert.deepEqual(
      results.distances,
      new Float32Array( [ 270, 530, 270, 530 ]),
      'distances should be 270 and 530'
    );
  });

});

test('Expected results', () => {
  const index = new usearch.Index({
    metric: 'l2sq',
    connectivity: 16,
    dimensions: 3,
  });
  index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
  const results = index.search(new Float32Array([0.2, 0.6, 0.4]), 10);

  assert.equal(index.size(), 1);
  assert.deepEqual(results.keys, new BigUint64Array([42n]));
  assertAlmostEqual(results.distances[0], new Float32Array([0]));
});

test('Multithread search returns same results', () => {
  const index = new usearch.Index({
    metric: 'l2sq',
    connectivity: 16,
    dimensions: 3,
  });
  index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
  assert.equal(index.size(), 1);

  const results_1 = index.search(new Float32Array([0.2, 0.6, 0.4]), 10, 0);
  const results_2 = index.search(
    new Float32Array([0.2, 0.6, 0.4]),
    10,
    THREADS
  );

  assert.deepEqual(results_1.keys, results_2.keys);
  assertAlmostEqual(results_1.distances, results_2.distances);
});

test('Exact search', async (t) => {
  const dataset = [
    new Float32Array([0.2, 0.6, 0.4]),
    new Float32Array([0.6, 0.6, 0.4]),
  ];
  const queries = new Float32Array([0.2, 0.6, 0.4]);
  const dimensions = 3;
  const count = 2;
  const metric = 'l2sq';

  await t.test('Single core search', () => {
    const result = usearch.exactSearch(
      dataset,
      queries,
      dimensions,
      count,
      metric,
      1 // threads
    );

    assert.deepEqual(result.keys, new BigUint64Array([0n, 1n]));
    assertAlmostEqual(new Float32Array([0]), result.distances[0]);
    assertAlmostEqual(new Float32Array([0.16]), result.distances[1]);
  });

  await t.test('Multithreaded search', () => {
    const result = usearch.exactSearch(
      dataset,
      queries,
      dimensions,
      count,
      metric,
      THREADS // threads
    );

    assert.deepEqual(result.keys, new BigUint64Array([0n, 1n]));
    assertAlmostEqual(new Float32Array([0]), result.distances[0]);
    assertAlmostEqual(new Float32Array([0.16]), result.distances[1]);
  });
});

test('Expected count()', async (t) => {
  const index = new usearch.Index({
    metric: 'l2sq',
    connectivity: 16,
    dimensions: 3,
  });
  index.add(
    [42n, 43n],
    [new Float32Array([0.2, 0.6, 0.4]), new Float32Array([0.2, 0.6, 0.4])]
  );

  await t.test('Argument is a number', () => {
    assert.equal(1, index.count(43n));
  });
  await t.test('Argument is a number (does not exist)', () => {
    assert.equal(0, index.count(44n));
  });
  await t.test('Argument is an array', () => {
    assert.deepEqual([1, 1, 0], index.count([42n, 43n, 44n]));
  });
});

test('Operations with invalid values', () => {
  const indexBatch = new usearch.Index(2, 'l2sq');

  const keys = [NaN, 16n];
  const vectors = [new Float32Array([10, 30]), new Float32Array([1, 5])];

  // All keys must be positive integers or bigints
  assert.throws(() => indexBatch.add(keys, vectors));

  // Vectors must be a TypedArray or an array of arrays
  assert.throws(() => indexBatch.search(NaN, 2));
});

test('Invalid operations', async (t) => {
  await t.test('Add the same keys', () => {
    const index = new usearch.Index({
      metric: 'l2sq',
      connectivity: 16,
      dimensions: 3,
    });
    index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
    assert.throws(() => index.add(42n, new Float32Array([0.2, 0.6, 0.4])));
  });

  await t.test('Batch add containing the same key', () => {
    const index = new usearch.Index({
      metric: 'l2sq',
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
        ]
      );
    });
  });
});

test('Serialization', async (t) => {
  const indexPath = path.join(os.tmpdir(), 'usearch.test.index');

  t.beforeEach(() => {
    const index = new usearch.Index({
      metric: 'l2sq',
      connectivity: 16,
      dimensions: 3,
    });
    index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
    index.save(indexPath);
  });

  t.afterEach(() => {
    fs.unlinkSync(indexPath);
  });

  await t.test('load', () => {
    const index = new usearch.Index({
      metric: 'l2sq',
      connectivity: 16,
      dimensions: 3,
    });
    index.load(indexPath);
    const results = index.search(
      new Float32Array([0.2, 0.6, 0.4]),
      10,
      THREADS
    );

    assert.equal(index.size(), 1);
    assert.deepEqual(results.keys, new BigUint64Array([42n]));
    assertAlmostEqual(results.distances[0], new Float32Array([0]));
  });

  // todo: Skip as the test fails only on windows.
  // The following error in afterEach().
  // `error: "EBUSY: resource busy or locked, unlink`
  await t.test(
    'view: Read data',
    { skip: process.platform === 'win32' },
    () => {
      const index = new usearch.Index({
        metric: 'l2sq',
        connectivity: 16,
        dimensions: 3,
      });
      index.view(indexPath);
      const results = index.search(new Float32Array([0.2, 0.6, 0.4]), 10);

      assert.equal(index.size(), 1);
      assert.deepEqual(results.keys, new BigUint64Array([42n]));
      assertAlmostEqual(results.distances[0], new Float32Array([0]));
    }
  );

  await t.test(
    'view: Invalid operations: add',
    { skip: process.platform === 'win32' },
    () => {
      const index = new usearch.Index({
        metric: 'l2sq',
        connectivity: 16,
        dimensions: 3,
      });
      index.view(indexPath);

      // Can't add to an immutable index
      assert.throws(() => index.add(43n, new Float32Array([0.2, 0.6, 0.4])));
    }
  );

  await t.test(
    'view: Invalid operations: remove',
    { skip: process.platform === 'win32' },
    () => {
      const index = new usearch.Index({
        metric: 'l2sq',
        connectivity: 16,
        dimensions: 3,
      });
      index.view(indexPath);

      // Can't remove from an immutable index
      assert.throws(() => index.remove(42n));
    }
  );
});
