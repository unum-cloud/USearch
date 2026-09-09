"""Shared harness for the Python side of the USearch test suite.

Holds the matrices every module parametrizes over, the oracle it checks against, the recall
floors it asserts, and the gate that skips a kernel family this machine cannot reach. Imported by
name from each test module; only what pytest loads for itself lives in `conftest`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import usearch
from usearch import DenseIndex, Executor, Matches
from usearch.eval import exact_matches, random_vectors
from usearch.keys import integers_to_keys, keys_to_integers

# region Matrices

# Dimension counts bracketing the SIMD register widths, where tail handling is most likely to
# diverge: below one register, straddling one, and several whole ones.
_POSSIBLE_DIMENSIONS = [1, 3, 15, 16, 17, 63, 64, 65, 97, 256]

dense_dimensions: list[int] = (
    [int(value) for value in given.split(",")]
    if (given := os.environ.get("USEARCH_DENSE_DIMENSIONS")) is not None
    else _POSSIBLE_DIMENSIONS
)

# Batch sizes bracketing the internal batching: one row, an odd few, and enough to parallelize.
batch_sizes: list[int] = [1, 7, 1024]

# Every scalar type the index stores, widest dynamic range first - floats descending, then
# signed integers, then unsigned. The order every dtype-indexed list in the project takes.
quantizations: list[str] = [
    "f64",
    "f32",
    "bf16",
    "f16",
    "e5m2",
    "e4m3",
    "e3m2",
    "e2m3",
    "i8",
    "u8",
]

# The metrics over real-valued rows; the bitwise pair takes packed bits and is swept separately.
spatial_metrics: list[str] = ["inner", "angular", "euclidean"]
bitwise_metrics: list[str] = ["hamming", "jaccard"]

# The input scalar types a caller is most likely to hand us, none of which need be the stored one.
input_dtypes = [np.float64, np.float32, np.float16, np.int8, np.uint8]

# endregion Matrices

# region Tolerances

# Recall floors for a self-query at the default expansion over `_RECALL_ROWS` rows. Quantization
# is the only thing that moves them: a stored `e2m3` row keeps three mantissa bits, so ties it
# cannot distinguish are resolved arbitrarily.
_RECALL_FLOORS: dict[str, float] = {
    "f64": 0.99,
    "f32": 0.99,
    "bf16": 0.98,
    "f16": 0.98,
    "e5m2": 0.85,
    "e4m3": 0.90,
    "e3m2": 0.70,
    "e2m3": 0.60,
    "i8": 0.90,
    "u8": 0.85,
    "u1x8": 0.60,
}

_RECALL_ROWS = 512


def recall_floor(quantization: str) -> float:
    """The share of true nearest neighbors a build in this scalar type must still return."""
    return _RECALL_FLOORS.get(quantization, 0.90)


# Absolute and relative tolerances for a distance checked against the oracle, by the scalar type
# the index stored. Integers get an absolute slack and no relative one, since their error is a
# rounding step rather than a fraction.
_DISTANCE_TOLERANCES: dict[str, tuple[float, float]] = {
    "f64": (1e-6, 1e-6),
    "f32": (1e-4, 1e-4),
    "bf16": (1e-1, 1e-1),
    "f16": (1e-2, 1e-2),
    "e5m2": (0.3, 0.3),
    "e4m3": (0.2, 0.2),
    "e3m2": (0.5, 0.5),
    "e2m3": (0.5, 0.5),
    "i8": (0.1, 0.1),
    "u8": (0.1, 0.1),
}


def distance_tolerances(quantization: str) -> tuple[float, float]:
    """The `(atol, rtol)` a distance in this stored scalar type may drift from the oracle."""
    return _DISTANCE_TOLERANCES.get(quantization, (0.5, 0.5))


# endregion Tolerances

# region Gates


def requires_capability(codename: str):
    """Skips when dispatch on this machine would not reach that kernel family."""
    return pytest.mark.skipif(
        codename not in usearch.capabilities_available(), reason=f"{codename} unavailable on this machine"
    )


# endregion Gates

# region Builders


def build_index(
    count: int,
    dimensions: int,
    executor: Executor,
    *,
    metric: str = "angular",
    quantization: str = "f32",
    seed: int = 0,
    **options,
) -> tuple[DenseIndex, np.ndarray]:
    """A reserved index holding *count* reproducible rows under the identities ``0..count``.

    Returns the index and the rows it was built from, so a caller can query with either.
    """
    index = DenseIndex.for_ram(dimensions=dimensions, metric=metric, quantization=quantization, **options)
    index.reserve(count)
    vectors = random_vectors(count, dimensions, metric=metric, seed=seed)
    assert index.add(integers_to_keys(np.arange(count, dtype=np.uint64)), vectors, executor=executor) == count
    return index, vectors


def matches_outputs(queries: int, wanted: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three zeroed outputs every ranked verb fills, for *queries* rows of *wanted* columns."""
    return (
        np.zeros((queries, wanted, 16), dtype=np.uint8),
        np.zeros((queries, wanted), dtype=np.float64),
        np.zeros(queries, dtype=np.uint64),
    )


def searched(index: DenseIndex, queries, wanted: int, **kwargs) -> Matches:
    """Runs `search` into freshly sized outputs and reads the three back as one run."""
    keys, distances, counts = matches_outputs(len(queries), wanted)
    index.search(queries, wanted, keys, distances, counts, **kwargs)
    return Matches(keys, distances, counts)


def measured_recall(index: DenseIndex, vectors, queries, wanted: int, executor: Executor, *, metric: str) -> float:
    """The share of true nearest neighbors the index returned, against the NumKong oracle."""
    truth = exact_matches(vectors, queries, wanted, metric=metric)
    found = keys_to_integers(searched(index, queries, wanted, executor=executor).keys)
    hits = sum(len(set(found[row]) & set(truth[row])) for row in range(len(truth)))
    return hits / (len(truth) * wanted)


# endregion Builders
