#!/usr/bin/env -S uv run --quiet --script
"""
USearch Tooling Tests

Test suite for USearch utility functions and tooling,
including I/O operations, matrix handling, and helper functions.

Usage:
    uv run python/scripts/test_tooling.py
    
Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "pytest",
#   "numpy",
#   "usearch"
# ]
# ///

import os

import pytest
import numpy as np

import usearch
from usearch.io import load_matrix, save_matrix
from usearch.index import search
from usearch.eval import random_vectors

from usearch.index import Match, Matches, BatchMatches, Index, Indexes, kmeans


dimensions = [3, 97, 256]
batch_sizes = [1, 77, 100]


@pytest.mark.parametrize("rows", batch_sizes)
@pytest.mark.parametrize("cols", dimensions)
def test_serializing_fbin_matrix(rows: int, cols: int):
    """
    Test the serialization of floating point binary matrix.

    :param int rows: The number of rows in the matrix.
    :param int cols: The number of columns in the matrix.
    """
    original = np.random.rand(rows, cols).astype(np.float32)
    save_matrix(original, "tmp.fbin")
    reconstructed = load_matrix("tmp.fbin")
    assert np.allclose(original, reconstructed)
    os.remove("tmp.fbin")


@pytest.mark.parametrize("rows", batch_sizes)
@pytest.mark.parametrize("cols", dimensions)
def test_serializing_ibin_matrix(rows: int, cols: int):
    """
    Test the serialization of integer binary matrix.

    :param int rows: The number of rows in the matrix.
    :param int cols: The number of columns in the matrix.
    """
    original = np.random.randint(0, rows + 1, size=(rows, cols)).astype(np.int32)
    save_matrix(original, "tmp.ibin")
    reconstructed = load_matrix("tmp.ibin")
    assert np.allclose(original, reconstructed)
    os.remove("tmp.ibin")


@pytest.mark.parametrize("rows", batch_sizes)
@pytest.mark.parametrize("cols", dimensions)
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("reordered", [False, True])
def test_exact_search(rows: int, cols: int, k: int, reordered: bool):
    """
    Test exact search.

    :param int rows: The number of rows in the matrix.
    :param int cols: The number of columns in the matrix.
    """
    if cols < 10:
        pytest.skip("In low dimensions collisions are likely.")
    original = np.random.rand(rows, cols)
    keys = np.arange(rows)
    k = min(k, rows)

    if reordered:
        reordered_keys = np.arange(rows)
        np.random.shuffle(reordered_keys)
    else:
        reordered_keys = keys

    matches: BatchMatches = search(original, original[reordered_keys], k, exact=True)
    top_matches = [int(m.keys[0]) for m in matches] if rows > 1 else [int(matches.keys[0])]
    assert top_matches == list(reordered_keys)

    matches: Matches = search(original, original[-1], k, exact=True)
    top_match = int(matches.keys[0])
    assert top_match == keys[-1]


def test_matches_creation_and_methods():
    matches = Matches(
        keys=np.array([1, 2]),
        distances=np.array([0.5, 0.6]),
        visited_members=2,
        computed_distances=2,
    )
    assert len(matches) == 2
    assert matches[0] == Match(key=1, distance=0.5)
    assert matches.to_list() == [(1, 0.5), (2, 0.6)]


def test_batch_matches_creation_and_methods():
    keys = np.array([[1, 2], [3, 4]])
    distances = np.array([[0.5, 0.6], [0.7, 0.8]])
    counts = np.array([2, 2])
    batch_matches = BatchMatches(
        keys=keys,
        distances=distances,
        counts=counts,
        visited_members=2,
        computed_distances=2,
    )

    assert len(batch_matches) == 2
    assert batch_matches[0].keys.tolist() == [1, 2]
    assert batch_matches[0].distances.tolist() == [0.5, 0.6]
    assert batch_matches.to_list() == [(1, 0.5), (2, 0.6), (3, 0.7), (4, 0.8)]


def test_multi_index():
    ndim = 10
    index_a = Index(ndim=ndim)
    index_b = Index(ndim=ndim)

    vectors = random_vectors(count=3, ndim=ndim)
    index_a.add(42, vectors[0])
    index_b.add(43, vectors[1])

    indexes = Indexes([index_a, index_b])

    # Search top 10 for 1
    matches = indexes.search(vectors[2], 10)
    assert len(matches) == 2

    # Search top 1 for 1
    matches = indexes.search(vectors[2], 1)
    assert len(matches) == 1

    # Search top 10 for 3
    matches = indexes.search(vectors, 10)
    assert len(matches) == 3
    assert len(matches[0].keys) == 2


def test_kmeans(count_vectors: int = 100, ndim: int = 10, count_clusters: int = 5):
    X = np.random.rand(count_vectors, ndim)
    assignments, distances, centroids = kmeans(X, count_clusters)
    assert len(assignments) == count_vectors
    assert ((assignments >= 0) & (assignments < count_clusters)).all()
