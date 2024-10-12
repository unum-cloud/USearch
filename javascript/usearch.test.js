const test = require('node:test');
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


test('Single-entry operations', () => {
    const index = new usearch.Index(2, 'l2sq');

    assert.equal(index.connectivity(), 16, 'connectivity should be 16');
    assert.equal(index.dimensions(), 2, 'dimensions should be 2');
    assert.equal(index.size(), 0, 'initial size should be 0');

    index.add(15n, new Float32Array([10, 20]));
    index.add(16n, new Float32Array([10, 25]));

    assert.equal(index.size(), 2, 'size after adding elements should be 2');
    assert.equal(index.contains(15), true, 'entry must be present after insertion');

    const results = index.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(results.keys, new BigUint64Array([15n, 16n]), 'keys should be 15 and 16');
    assert.deepEqual(results.distances, new Float32Array([45, 130]), 'distances should be 45 and 130');
});

test('Batch operations', () => {
    const indexBatch = new usearch.Index(2, 'l2sq');

    const keys = [15n, 16n];
    const vectors = [new Float32Array([10, 20]), new Float32Array([10, 25])];

    indexBatch.add(keys, vectors);
    assert.equal(indexBatch.size(), 2, 'size after adding batch should be 2');

    const results = indexBatch.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(results.keys, new BigUint64Array([15n, 16n]), 'keys should be 15 and 16');
    assert.deepEqual(results.distances, new Float32Array([45, 130]), 'distances should be 45 and 130');
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

    assert.throws(
        () => indexBatch.add(keys, vectors),
        {
            name: 'Error',
            message: 'All keys must be positive integers or bigints.'
        }
    );

    assert.throws(
        () => indexBatch.search(NaN, 2),
        {
            name: 'Error',
            message: 'Vectors must be a TypedArray or an array of arrays.'
        }
    );
});

test('Invalid operations', async (t) => {
    await t.test('Add the same keys', () => {
        const index = new usearch.Index({
            metric: "l2sq",
            connectivity: 16,
            dimensions: 3,
        });
        index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
        assert.throws(
            () => index.add(42n, new Float32Array([0.2, 0.6, 0.4])),
            {
                name: 'Error',
                message: 'Duplicate keys not allowed in high-level wrappers'
            }
        );
    });
});


test('Serialization', async (t) => {
    const indexPath = path.join(os.tmpdir(), 'usearch.test.index')

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

    await t.test('load', () => {
        const index = new usearch.Index({
            metric: "l2sq",
            connectivity: 16,
            dimensions: 3,
        });
        index.load(indexPath);
        const results = index.search(new Float32Array([0.2, 0.6, 0.4]), 10);

        assert.equal(index.size(), 1);
        assert.deepEqual(results.keys, new BigUint64Array([42n]));
        assertAlmostEqual(results.distances[0], new Float32Array([0]));
    });

    // todo: Skip as the test fails only on windows.
    // The following error in afterEach().
    // `error: "EBUSY: resource busy or locked, unlink`
    await t.test('view', {skip: process.platform === 'win32'}, () => {
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
    });
});
