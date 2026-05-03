#!/usr/bin/env -S uv run --quiet --script
"""
USearch Index Tests

Comprehensive test suite for USearch index functionality including
construction, search, serialization, and various data type operations.

Usage:
    uv run python/scripts/test_index.py

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
from time import time

import pytest
import numpy as np

from usearch.eval import random_vectors, self_recall, SearchStats
from usearch.index import (
    Index,
    MetricKind,
    ScalarKind,
    Match,
    Matches,
    BatchMatches,
    Clustering,
)
from usearch.index import (
    DEFAULT_CONNECTIVITY,
)


ndims = [3, 97, 256]
batch_sizes = [1, 11, 77]
quantizations = [
    ScalarKind.F64,
    ScalarKind.F32,
    ScalarKind.BF16,
    ScalarKind.F16,
    ScalarKind.E5M2,
    ScalarKind.E4M3,
    ScalarKind.I8,
]
dtypes = [np.float32, np.float64, np.float16]
threads = 2

connectivity_options = [3, 13, 50, DEFAULT_CONNECTIVITY]
continuous_metrics = [MetricKind.Cos, MetricKind.L2sq]
hash_metrics = [
    MetricKind.Hamming,
    MetricKind.Tanimoto,
    MetricKind.Sorensen,
]


def reset_randomness():
    np.random.seed(int(time()))


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_initialization_and_addition(ndim, metric, quantization, dtype, batch_size):
    reset_randomness()

    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)
    assert len(index) == batch_size


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.F16, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_retrieval(ndim, metric, quantization, dtype, batch_size):
    reset_randomness()

    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)
    vectors_retrieved = np.vstack(index.get(keys, dtype))
    assert np.allclose(vectors_retrieved, vectors, atol=0.1)

    # Try retrieving all the keys
    keys_retrieved = index.keys
    keys_retrieved = np.array(keys_retrieved)
    assert np.all(np.sort(keys_retrieved) == keys)

    # Try retrieving all of them
    if quantization != ScalarKind.I8:
        # The returned vectors can be in a different order
        vectors_batch_retrieved = index.vectors
        vectors_reordering = np.argsort(keys_retrieved)
        vectors_batch_retrieved = vectors_batch_retrieved[vectors_reordering]
        assert np.allclose(vectors_batch_retrieved, vectors, atol=0.1)

    if quantization != ScalarKind.I8 and batch_size > 1:
        # When dealing with non-continuous data, it's important to check that
        # the native bindings access them with correct strides or normalize
        # similar to `np.ascontiguousarray`:
        index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
        vectors = random_vectors(count=batch_size, ndim=ndim + 1, dtype=dtype)
        # Let's skip the first dimension of each vector:
        vectors = vectors[:, 1:]
        index.add(keys, vectors, threads=threads)
        vectors_retrieved = np.vstack(index.get(keys, dtype))
        assert np.allclose(vectors_retrieved, vectors, atol=0.1)

        # Try a transposed version of the same vectors, that is not C-contiguous
        # and should raise an exception!
        index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
        vectors = random_vectors(count=ndim, ndim=batch_size, dtype=dtype)  #! reversed dims
        assert vectors.strides == (batch_size * dtype().itemsize, dtype().itemsize)
        assert vectors.T.strides == (dtype().itemsize, batch_size * dtype().itemsize)
        with pytest.raises(Exception):
            index.add(keys, vectors.T, threads=threads)


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_search(ndim, metric, quantization, dtype, batch_size):
    reset_randomness()

    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)

    if batch_size == 1:
        matches: Matches = index.search(vectors, 10, threads=threads)
        assert isinstance(matches, Matches)
        assert isinstance(matches[0], Match)
        assert matches.keys.ndim == 1
        assert matches.keys.shape[0] == matches.distances.shape[0]
        assert len(matches) == batch_size
        assert np.all(np.sort(index.keys) == np.sort(keys))

    else:
        matches: BatchMatches = index.search(vectors, 10, threads=threads)
        assert isinstance(matches, BatchMatches)
        assert isinstance(matches[0], Matches)
        assert isinstance(matches[0][0], Match)
        assert matches.keys.ndim == 2
        assert matches.keys.shape[0] == matches.distances.shape[0]
        assert len(matches) == batch_size
        assert np.all(np.sort(index.keys) == np.sort(keys))


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
def test_index_self_recall(ndim: int, batch_size: int):
    """
    Test self-recall evaluation scripts.
    """
    reset_randomness()

    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)

    stats_all: SearchStats = self_recall(index, keys=keys)
    stats_quarter: SearchStats = self_recall(index, sample=0.25, count=10)

    assert stats_all.computed_distances > 0
    assert stats_quarter.computed_distances > 0


@pytest.mark.parametrize("batch_size", [1, 7, 1024])
def test_index_duplicates(batch_size):
    reset_randomness()

    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)
    with pytest.raises(Exception):
        index.add(keys, vectors, threads=threads)

    index = Index(ndim=ndim, multi=True)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)
    index.add(keys, vectors, threads=threads)
    assert len(index) == batch_size * 2

    two_per_key = index.get(keys)
    assert np.vstack(two_per_key).shape == (2 * batch_size, ndim)


@pytest.mark.parametrize("batch_size", [1, 7, 1024])
def test_index_stats(batch_size):
    reset_randomness()

    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)

    assert index.max_level >= 0
    assert index.stats.nodes >= batch_size
    assert index.levels_stats[0].nodes == batch_size
    assert index.level_stats(0).nodes == batch_size

    assert index.levels_stats[index.max_level].nodes > 0


@pytest.mark.parametrize("use_view", [True, False])
def test_index_load_from_buffer(use_view: bool, ndim: int = 3, batch_size: int = 10):
    reset_randomness()

    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)

    buffer = index.save()
    assert isinstance(buffer, bytearray)

    def _test_load(obj):
        index.clear()
        assert len(index) == 0
        index.view(obj) if use_view else index.load(obj)
        assert len(index) == batch_size

    _test_load(bytes(buffer))
    _test_load(bytearray(buffer))
    _test_load(memoryview(buffer))
    _test_load(np.array(buffer))
    with pytest.raises(TypeError):
        _test_load(123)


@pytest.mark.parametrize("ndim", [1, 3, 8, 32, 256, 4096])
@pytest.mark.parametrize("batch_size", [0, 1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
def test_index_save_load_restore_copy(ndim, quantization, batch_size):
    reset_randomness()
    index = Index(ndim=ndim, dtype=quantization, multi=False)

    if batch_size > 0:
        keys = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim)
        index.add(keys, vectors, threads=threads)

    # Try copying the original
    copied_index = index.copy()
    assert len(copied_index) == len(index)
    if batch_size > 0:
        assert np.allclose(np.vstack(copied_index.get(keys)), np.vstack(index.get(keys)))

    index.save("tmp.usearch")
    index.clear()
    assert len(index) == 0
    assert os.path.exists("tmp.usearch")

    index.load("tmp.usearch")
    assert len(index) == batch_size
    if batch_size > 0:
        assert len(index[0].flatten()) == ndim

    index_meta = Index.metadata("tmp.usearch")
    assert index_meta is not None

    index = Index.restore("tmp.usearch", view=False)
    assert len(index) == batch_size
    if batch_size > 0:
        assert len(index[0].flatten()) == ndim

    # Try copying the restored index
    copied_index = index.copy()
    assert len(copied_index) == len(index)
    if batch_size > 0:
        assert np.allclose(np.vstack(copied_index.get(keys)), np.vstack(index.get(keys)))

    # Perform the same operations in RAM, without touching the filesystem
    serialized_index = index.save()
    deserialized_metadata = Index.metadata(serialized_index)
    assert deserialized_metadata is not None

    deserialized_index = Index.restore(serialized_index)
    assert len(deserialized_index) == len(index)
    assert set(np.array(deserialized_index.keys)) == set(np.array(index.keys))
    if batch_size > 0:
        assert np.allclose(np.vstack(deserialized_index.get(keys)), np.vstack(index.get(keys)))

    deserialized_index.reset()
    index.reset()
    os.remove("tmp.usearch")


@pytest.mark.parametrize("ndim", [3, 8, 32, 256, 4096])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("threads", [1, 3, 7, 150])
def test_index_restore_multithread_search(ndim, batch_size, threads):

    reset_randomness()
    quantization = ScalarKind.F32
    index = Index(ndim=ndim, dtype=quantization, multi=False)

    if batch_size > 0:
        keys = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim, dtype=quantization)
        index.add(keys, vectors, threads=threads)

    query = random_vectors(count=batch_size, ndim=ndim, dtype=quantization)
    k = min(batch_size, 10)

    result_original = index.search(query, count=k, threads=threads)
    dumped_index: bytes = index.save()
    dumped_index_view = memoryview(dumped_index)

    # When restoring from disk, search must not fail if using multiple threads.
    index_restored = Index.restore(dumped_index, view=False)
    result_restored = index_restored.search(query, count=k, threads=threads)
    assert np.allclose(result_original.distances, result_restored.distances, atol=0.1)

    index_viewed = Index.restore(dumped_index_view, view=True)
    result_view = index_viewed.search(query, count=k, threads=threads)
    assert np.allclose(result_original.distances, result_view.distances, atol=0.1)


@pytest.mark.parametrize("batch_size", [32])
def test_index_contains_remove_rename(batch_size):
    reset_randomness()
    if batch_size <= 1:
        return

    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)

    index.add(keys, vectors, threads=threads)
    assert np.all(index.contains(keys))
    assert np.all(index.count(keys) == np.ones(batch_size))

    removed_keys = keys[: batch_size // 2]
    remaining_keys = keys[batch_size // 2 :]
    index.remove(removed_keys)
    del index[removed_keys]  # ! This will trigger the `__delitem__` dunder method
    assert len(index) == (len(keys) - len(removed_keys))
    assert np.sum(index.contains(keys)) == len(remaining_keys)
    assert np.sum(index.count(keys)) == len(remaining_keys)
    assert np.sum(index.count(removed_keys)) == 0

    assert keys[0] not in index
    assert keys[-1] in index

    renamed_counts = index.rename(removed_keys, removed_keys)
    assert np.sum(index.count(renamed_counts)) == 0

    renamed_counts = index.rename(remaining_keys, removed_keys)
    assert np.sum(index.count(removed_keys)) == len(index)


@pytest.mark.skip(reason="Not guaranteed")
@pytest.mark.parametrize("batch_size", [3, 17, 33])
@pytest.mark.parametrize("threads", [1, 4])
def test_index_oversubscribed_search(batch_size: int, threads: int):
    reset_randomness()
    if batch_size <= 1:
        return

    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)

    index.add(keys, vectors, threads=threads)
    assert np.all(index.contains(keys))
    assert np.all(index.count(keys) == np.ones(batch_size))

    batch_matches: BatchMatches = index.search(vectors, batch_size * 10, threads=threads)
    for i, match in enumerate(batch_matches):
        assert i == match.keys[0]
        assert len(match.keys) == batch_size


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [500, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_clustering(ndim, metric, quantization, dtype, batch_size):
    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)

    clusters: Clustering = index.cluster(vectors=vectors, threads=threads)
    assert len(clusters.matches.keys) == batch_size

    # If no argument is provided, we cluster the present entries
    clusters: Clustering = index.cluster(threads=threads)
    assert len(clusters.matches.keys) == batch_size

    # If no argument is provided, we cluster the present entries
    clusters: Clustering = index.cluster(keys=keys[:50], threads=threads)
    assert len(clusters.matches.keys) == 50

    # If no argument is provided, we cluster the present entries
    clusters: Clustering = index.cluster(min_count=3, max_count=10, threads=threads)
    unique_clusters = set(clusters.matches.keys.flatten().tolist())
    assert len(unique_clusters) >= 3 and len(unique_clusters) <= 10


def test_index_keys_iteration():
    """Test that iterating over index.keys works without infinite loop."""
    index = Index(ndim=3)
    index.add(keys=[42], vectors=np.array([0.2, 0.3, 0.5]))

    keys_list = list(index.keys)
    assert len(keys_list) == 1
    assert keys_list[0] == 42


@pytest.mark.parametrize("ndim", [16, 64])
@pytest.mark.parametrize("batch_size", [10, 50])
def test_index_join(ndim, batch_size):
    """Semantic join should return a 1-to-1 mapping between two indexes."""
    index_a = Index(ndim=ndim, metric=MetricKind.Cos)
    index_b = Index(ndim=ndim, metric=MetricKind.Cos)

    vectors_a = random_vectors(count=batch_size, ndim=ndim)
    vectors_b = random_vectors(count=batch_size, ndim=ndim)
    keys_a = np.arange(batch_size)
    keys_b = np.arange(batch_size, 2 * batch_size)

    index_a.add(keys_a, vectors_a)
    index_b.add(keys_b, vectors_b)

    mapping = index_a.join(index_b, exact=True)
    assert isinstance(mapping, dict)
    assert len(mapping) > 0
    # All returned keys must be valid
    assert all(k in keys_a for k in mapping.keys())
    assert all(v in keys_b for v in mapping.values())
    # No two a-keys should map to the same b-key (stable marriage property)
    assert len(set(mapping.values())) == len(mapping)


def test_index_ip_metric():
    """Inner product metric should be usable and produce valid searches."""
    ndim = 32
    count = 20
    index = Index(ndim=ndim, metric=MetricKind.IP)
    keys = np.arange(count)
    vectors = random_vectors(count=count, ndim=ndim, metric=MetricKind.IP)
    index.add(keys, vectors)

    matches = index.search(vectors[0], 5)
    assert isinstance(matches, Matches)
    assert len(matches) == 5


def test_index_specs():
    """specs property should return a dict with expected keys."""
    ndim = 16
    index = Index(ndim=ndim, metric=MetricKind.L2sq, dtype=ScalarKind.F32)
    s = index.specs
    assert isinstance(s, dict)
    for key in ("ndim", "multi", "connectivity", "expansion_add", "expansion_search", "dtype"):
        assert key in s
    assert s["ndim"] == ndim


@pytest.mark.parametrize("ndim", [16, 64])
@pytest.mark.parametrize("batch_size", [10, 50])
def test_index_exact_search(ndim, batch_size):
    """Exact search must return the query vector itself as the top match."""
    index = Index(ndim=ndim, metric=MetricKind.L2sq)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors)

    if batch_size == 1:
        matches = index.search(vectors, 1, exact=True)
        assert int(matches.keys[0]) == keys[0]
    else:
        matches: BatchMatches = index.search(vectors, 1, exact=True)
        top_keys = [int(m.keys[0]) for m in matches]
        assert top_keys == list(keys)


@pytest.mark.parametrize("batch_size", [6, 20])
def test_index_pairwise_distance_array(batch_size):
    """pairwise_distance with equal-length key arrays returns element-wise distances."""
    ndim = 16
    index = Index(ndim=ndim, metric=MetricKind.L2sq)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors)

    half = batch_size // 2
    left_keys = keys[:half]
    right_keys = keys[half : 2 * half]
    distances = index.pairwise_distance(left_keys, right_keys)
    assert distances.shape == (half,)
    assert np.all(distances >= 0)


def test_index_pairwise_distance_scalar():
    """pairwise_distance with scalar keys returns a scalar distance."""
    ndim = 16
    index = Index(ndim=ndim, metric=MetricKind.L2sq)
    keys = np.arange(4)
    vectors = random_vectors(count=4, ndim=ndim)
    index.add(keys, vectors)

    dist = index.pairwise_distance(0, 1)
    assert isinstance(dist, float)
    assert dist >= 0
    # Distance from a vector to itself should be ~0
    assert index.pairwise_distance(0, 0) == pytest.approx(0.0, abs=1e-3)


def test_index_memory_usage_grows():
    """Memory usage should increase as vectors are added."""
    ndim = 32
    index = Index(ndim=ndim)
    mem_empty = index.memory_usage

    keys = np.arange(100)
    vectors = random_vectors(count=100, ndim=ndim)
    index.add(keys, vectors)

    assert index.memory_usage > mem_empty


def test_index_copied_memory_usage():
    """Test that copy=False results in lower memory usage than copy=True."""
    reset_randomness()

    ndim = 128
    batch_size = 1000
    dtype = np.float32  # ! Ensure same type for both vectors and index
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    keys = np.arange(batch_size)

    # Create index with `copy=True`
    index_copied = Index(ndim=ndim, metric=MetricKind.Cos, dtype=dtype, multi=False)
    index_copied.add(keys, vectors, copy=True, threads=threads)

    # Create index with `copy=False`
    index_viewing = Index(ndim=ndim, metric=MetricKind.Cos, dtype=dtype, multi=False)
    index_viewing.add(keys, vectors, copy=False, threads=threads)

    # Both should have same number of entries
    assert len(index_copied) == len(index_viewing) == batch_size

    # Memory usage should be larger when `copy=True`
    memory_with_copy = index_copied.memory_usage
    memory_without_copy = index_viewing.memory_usage

    assert memory_with_copy > memory_without_copy, (
        f"Expected default index addition to use more memory than copy=False ({memory_with_copy} vs {memory_without_copy})"
    )
