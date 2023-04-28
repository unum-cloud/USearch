var assert = require('assert');
var usearch = require('bindings')('usearch');

var index = new usearch.Index({ metric: 'l2_sq', connectivity: 16, dim: 2 })
assert.equal(index.connectivity(), 16)
assert.equal(index.dim(), 2)
assert.equal(index.size(), 0)

index.add(15, new Float32Array([10, 20]))
index.add(16, new Float32Array([10, 25]))
assert.equal(index.size(), 2)

var results = index.search(new Float32Array([13, 14]), 3)
assert.deepEqual(results.labels, new Uint32Array([15, 16]))
assert.deepEqual(results.distances, new Float32Array([45, 130]))

