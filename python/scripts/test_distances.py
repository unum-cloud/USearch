#!/usr/bin/env -S uv run --quiet --script
"""
USearch Distance Functions Tests

Test suite for various distance metrics and their implementations in USearch,
including cosine, L2, inner product, and custom distance functions.

Usage:
    uv run python/scripts/test_distances.py

Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "pytest",
#   "numpy",
#   "usearch"
# ]
# ///

import numpy as np
import pytest

from usearch.eval import random_vectors
from usearch.index import (
    Index,
    MetricKind,
    ScalarKind,
)


@pytest.mark.parametrize(
    "metric",
    [
        MetricKind.Cos,
        MetricKind.L2sq,
    ],
)
@pytest.mark.parametrize(
    "quantization",
    [
        ScalarKind.F32,
        ScalarKind.BF16,
        ScalarKind.F16,
        ScalarKind.E5M2,
        ScalarKind.E4M3,
        ScalarKind.E3M2,
        ScalarKind.E2M3,
        ScalarKind.I8,
        ScalarKind.U8,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.float16,
        np.int8,
    ],
)
def test_distances_continuous(metric, quantization, dtype):
    ndim = 1024
    try:
        index = Index(ndim=ndim, metric=metric, dtype=quantization)
        vectors = random_vectors(count=2, ndim=ndim, dtype=dtype)
        keys = np.arange(2)
        index.add(keys, vectors)
    except ValueError:
        pytest.skip(f"Unsupported metric `{metric}`, quantization `{quantization}`, dtype `{dtype}`")
        return

    rtol = 1e-2
    atol = 1e-2

    distance_itself_first = index.pairwise_distance([0], [0])
    distance_itself_second = index.pairwise_distance([1], [1])
    distance_different = index.pairwise_distance([0], [1])

    assert not np.allclose(distance_different, 0)
    assert np.allclose(distance_itself_first, 0, rtol=rtol, atol=atol)
    assert np.allclose(distance_itself_second, 0, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "metric",
    [
        MetricKind.Hamming,
        MetricKind.Tanimoto,
        MetricKind.Sorensen,
    ],
)
def test_distances_sparse(metric):
    ndim = 1024
    index = Index(ndim=ndim, metric=metric, dtype=ScalarKind.B1)
    vectors = random_vectors(count=2, ndim=ndim, dtype=ScalarKind.B1)
    keys = np.arange(2)
    index.add(keys, vectors)

    rtol = 1e-2
    atol = 1e-2

    distance_itself_first = index.pairwise_distance([0], [0])
    distance_itself_second = index.pairwise_distance([1], [1])
    distance_different = index.pairwise_distance([0], [1])

    assert not np.allclose(distance_different, 0)
    assert np.allclose(distance_itself_first, 0, rtol=rtol, atol=atol) and np.allclose(
        distance_itself_second, 0, rtol=rtol, atol=atol
    )
