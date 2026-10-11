"""Measuring what a built index actually retrieves.

The oracle is NumKong: it computes distances in the same metric vocabulary and the same kernel
family the index dispatches to, so a recall number compares like with like. NumPy stays for
generation and for the top-k selection, where NumKong has no `argsort` and precision is moot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from time import perf_counter_ns
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

import numkong as nk
from usearch import Arena, DenseIndex, DenseMetric, DenseTraversalStats, On
from usearch.keys import integers_to_keys, keys_to_integers

__all__: list[str] = [
    "SearchStats",
    "exact_matches",
    "index_keys",
    "integers_to_keys",
    "keys_to_integers",
    "measure_seconds",
    "random_vectors",
    "self_recall",
]
"""`integers_to_keys` and `keys_to_integers` are re-exported so a caller reaching for the recall
harness finds the identity bridge beside it."""

_NUMKONG_METRICS: dict[DenseMetric, Literal["inner", "angular", "euclidean", "hamming", "jaccard"]] = {
    DenseMetric.INNER: "inner",
    DenseMetric.ANGULAR: "angular",
    DenseMetric.EUCLIDEAN: "euclidean",
    DenseMetric.HAMMING: "hamming",
    DenseMetric.JACCARD: "jaccard",
}
"""The NumKong spelling of every metric the index scores, which the two libraries share."""

_BITWISE_METRICS: tuple[DenseMetric, ...] = (DenseMetric.HAMMING, DenseMetric.JACCARD)
"""Metrics over packed bits, where generation produces bit vectors rather than gaussians."""

_DESCENDING_METRICS: tuple[DenseMetric, ...] = (DenseMetric.INNER,)
"""NumKong scores `inner` as the raw dot product while the index scores it as `1 - dot`, so the
two rank a candidate list in opposite directions. Every other metric agrees on direction, and
ranking is all an oracle needs - a monotone difference in magnitude changes no ordering."""


def random_vectors(
    count: int,
    dimensions: int,
    *,
    metric: DenseMetric = DenseMetric.ANGULAR,
    seed: int = 0,
) -> npt.NDArray[np.generic]:
    """Reproducible rows for the given metric: `float32`, or packed `uint8` bits for a bitwise one, hence the
    generic dtype.

    Bitwise metrics get packed bits, where *dimensions* counts bits rather than scalars;
    everything else gets unit-norm gaussians, which sit in the representable range of every
    quantization the index offers. The *seed* fixes the draw, so a failure reproduces.
    """
    generator = np.random.default_rng(seed)
    if metric in _BITWISE_METRICS:
        bits = generator.integers(2, size=(count, dimensions), dtype=np.uint8)
        return np.packbits(bits, axis=1)

    rows = generator.standard_normal((count, dimensions)).astype(np.float32)
    norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-10)
    return (rows / norms).astype(np.float32)


def exact_matches(vectors, queries, wanted: int, *, metric: DenseMetric = DenseMetric.ANGULAR) -> npt.NDArray[np.intp]:
    """Brute-force ground truth, scored by the kernels the index itself dispatches to.

    Scores the ``(Q, dimensions)`` queries against the ``(N, dimensions)`` matrix the index was
    built over, in the index's own metric, and returns a ``(Q, wanted)`` matrix of row offsets
    into *vectors*, nearest first.
    """
    scores = np.asarray(nk.cdist(queries, vectors, _NUMKONG_METRICS[metric], out_dtype="float64"))
    if metric in _DESCENDING_METRICS:
        scores = -scores
    wanted = min(wanted, scores.shape[1])
    candidates = np.argpartition(scores, wanted - 1, axis=1)[:, :wanted]
    ordering = np.take_along_axis(scores, candidates, axis=1).argsort(axis=1)
    return np.take_along_axis(candidates, ordering, axis=1)


@dataclass
class SearchStats:
    """What one or more search runs cost, and how much of the truth they found.

    Recall is the share of queries whose true nearest neighbor came back. Efficiency is the
    share of the index a query did not have to score: highest when one distance sufficed,
    zero when every vector was visited, as in an exact scan.
    """

    index_size: int
    count_queries: int
    count_matches: int
    visited_members: int
    computed_distances: int

    @property
    def mean_efficiency(self) -> float:
        return 1 - float(self.computed_distances) / (self.count_queries * self.index_size)

    @property
    def mean_recall(self) -> float:
        return self.count_matches / self.count_queries


def index_keys(index: DenseIndex | On[DenseIndex], arena: Arena) -> npt.NDArray[np.uint8]:
    """Every identity the index holds, as a (size, 16) matrix, walked through one cursor landing in *arena*."""
    cursor, batches = index.cursor(), [np.zeros((0, 16), dtype=np.uint8)]
    while batch := cursor.next(1024, arena):
        batches.append(np.array(batch.keys))
    return np.concatenate(batches, axis=0)


def self_recall(
    index: DenseIndex | On[DenseIndex],
    arena: Arena,
    sample: float | int = 1.0,
    count: int = 1,
    *,
    keys=None,
    vectors=None,
    expansion_factor: int = 0,
    stats: DenseTraversalStats | None = None,
) -> SearchStats:
    """Queries the index with its own vectors, which must find themselves.

    The weakest recall test there is, and the cheapest: it needs no ground truth. Use
    ``exact_matches`` when the question is how the index ranks against the exact answer.

    The *index* is the index or any blocking receiver - ``index.on(executor)`` runs every call on
    that executor, and every output lands in *arena*, made by that executor or by none for the
    index itself. The *sample* is a share when a float and a count when an integer, and *count*
    is how many matches each query retrieves. The *keys* and *vectors* default to everything the
    index holds, and the *stats* given here is what the returned telemetry reads off.
    """
    size = index.describe().size
    if size == 0:
        return SearchStats(0, 0, 0, 0, 0)

    keys = np.asarray(index_keys(index, arena) if keys is None else keys)
    if sample != 1.0:
        wanted = int(ceil(len(keys) * sample)) if isinstance(sample, float) else sample
        chosen = np.random.default_rng(0).choice(len(keys), wanted, replace=False)
        keys = keys[chosen]

    if vectors is None:
        vectors = np.concatenate([np.array(index.get(row, arena)) for row in keys], axis=0)
    vectors = np.atleast_2d(np.asarray(vectors))

    stats = DenseTraversalStats() if stats is None else stats
    matches = index.search(vectors, count, arena, expansion_factor=expansion_factor, stats=stats)

    expected = keys_to_integers(keys)
    found = keys_to_integers(matches.keys)
    found_counts = np.asarray(matches.counts)

    count_matches = 0
    for query_index in range(len(expected)):
        row = found[query_index, : found_counts[query_index]]
        count_matches += int(np.any(row == expected[query_index]))

    return SearchStats(
        index_size=size,
        count_queries=len(keys),
        count_matches=count_matches,
        visited_members=stats.visited_members,
        computed_distances=stats.computed_distances,
    )


def measure_seconds(callable_to_time: Callable[[], Any]) -> tuple[float, Any]:
    """Times one call with the monotonic clock, returning the seconds and whatever it returned."""
    before = perf_counter_ns()
    result = callable_to_time()
    return (perf_counter_ns() - before) / 1e9, result
