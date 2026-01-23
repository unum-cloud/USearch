#!/usr/bin/env -S uv run --quiet --script
"""
USearch Sparse Vector Tests

Test suite for sparse vector operations and indexing in USearch,
including sparse data structures and search algorithms.

Usage:
    uv run python/scripts/test_sparse.py

Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "pytest",
#   "numpy",
#   "usearch"
# ]
# ///

import pytest
import numpy as np

import usearch
from usearch.index import (
    Index,
    MetricKind,
    ScalarKind,
)


@pytest.mark.parametrize("bits", [7, 97, 256, 4097])
@pytest.mark.parametrize("metric", [MetricKind.Tanimoto])
@pytest.mark.parametrize("connectivity", [3, 13, 50])
@pytest.mark.parametrize("batch_size", [3, 77])
def test_bitwise_index(
    bits: int,
    metric: MetricKind,
    connectivity: int,
    batch_size: int,
):
    index = Index(ndim=bits, metric=metric, connectivity=connectivity)

    keys = np.arange(batch_size)
    byte_vectors = np.random.randint(2, size=(batch_size, bits))
    bit_vectors = np.packbits(byte_vectors, axis=1)

    index.add(keys, bit_vectors)

    byte_vectors_retrieved = np.vstack(index.get(keys, ScalarKind.B1))
    assert np.all(byte_vectors_retrieved == bit_vectors)

    index.search(bit_vectors, 10)
