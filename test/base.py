"""Shared harness for the Python side of the USearch test suite.

Holds the matrices every module parametrizes over, the oracle it checks against, the recall
floors it asserts, and the gate that skips a kernel family this machine cannot reach. Imported by
name from each test module; only what pytest loads for itself lives in `conftest`.
"""

from __future__ import annotations

import os
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

import usearch
from usearch import DenseIndex, DenseMetric, Executor, Scalar
from usearch.eval import exact_matches, random_vectors
from usearch.keys import integers_to_keys, keys_to_integers

# region Settings


def env_text(name: str) -> str | None:
    """Reads `name`, or `None` when it is unset or empty."""
    return os.environ.get(name) or None


def env_parsed[T](name: str, fallback: T, parse: Callable[[str], T | None], expected: str) -> T:
    """Reads `name` through `parse`, or `fallback` when unset or empty; exits with status 1 if it does not parse."""
    text = env_text(name)
    if text is None:
        return fallback
    try:
        parsed = parse(text)
    except (ValueError, TypeError):
        parsed = None
    if parsed is None:
        raise SystemExit(f'{name}="{text}" does not parse, expected {expected}')
    return parsed


def parse_count(text: str) -> int | None:
    """Parses a positive whole number in ASCII digits, like `128`; zero is `None`."""
    digits = re.fullmatch(r"[0-9]+", text) is not None
    return (int(text) or None) if digits else None


def parse_dims(text: str) -> list[int] | None:
    """Parses one count like `128` or a comma list like `64,128,256,512`."""
    counts = [parse_count(piece) for piece in text.split(",")]
    return None if None in counts else counts


_REQUESTED_SEED = os.environ.get("USEARCH_SEED") or "42"
if _REQUESTED_SEED == "random":
    _RUN_SEED = secrets.randbits(32)
elif _REQUESTED_SEED.isascii() and _REQUESTED_SEED.isdigit() and int(_REQUESTED_SEED) < 2**32:
    _RUN_SEED = int(_REQUESTED_SEED)
else:
    raise SystemExit(f'USEARCH_SEED="{_REQUESTED_SEED}" does not parse, expected an unsigned integer or random')


@dataclass(frozen=True)
class Settings:
    """Every `USEARCH_*` variable the Python suite reads, parsed once at import."""

    seed: int
    filter: str
    filter_pattern: re.Pattern[str] | None
    dims: list[int]

    def selects(self, name: str) -> bool:
        """Whether `USEARCH_FILTER` selects the test `name`, as a regex or else as a substring."""
        return bool(self.filter_pattern.search(name)) if self.filter_pattern else self.filter in name


def read_settings() -> Settings:
    """Reads every `USEARCH_*` variable, exiting with status 1 on the first that does not parse."""
    filter = env_text("USEARCH_FILTER") or ""
    try:
        filter_pattern = re.compile(filter) if filter else None
    except re.error:
        filter_pattern = None
    return Settings(
        seed=_RUN_SEED,
        filter=filter,
        filter_pattern=filter_pattern,
        # Dimension counts bracketing the SIMD register widths, where tail handling is most likely
        # to diverge: below one register, straddling one, and several whole ones.
        dims=env_parsed(
            "USEARCH_DIMS", [1, 3, 15, 16, 17, 63, 64, 65, 97, 256], parse_dims, "positive counts like 64,128"
        ),
    )


SETTINGS = read_settings()

# endregion Settings

# region Matrices

# Batch sizes bracketing the internal batching: one row, an odd few, and enough to parallelize.
batch_sizes: list[int] = [1, 7, 1024]

# Every scalar type the index stores, widest dynamic range first - floats descending, then
# signed integers, then unsigned. The order every dtype-indexed list in the project takes.
quantizations: list[Scalar] = [
    Scalar.F64,
    Scalar.F32,
    Scalar.BF16,
    Scalar.F16,
    Scalar.E5M2,
    Scalar.E4M3,
    Scalar.E3M2,
    Scalar.E2M3,
    Scalar.I8,
    Scalar.U8,
]

# The metrics over real-valued rows; the bitwise pair takes packed bits and is swept separately.
spatial_metrics: list[DenseMetric] = [DenseMetric.INNER, DenseMetric.ANGULAR, DenseMetric.EUCLIDEAN]
bitwise_metrics: list[DenseMetric] = [DenseMetric.HAMMING, DenseMetric.JACCARD]

# The input scalar types a caller is most likely to hand us, none of which need be the stored one.
input_dtypes = [np.float64, np.float32, np.float16, np.int8, np.uint8]

# endregion Matrices

# region Tolerances

# Recall floors for a self-query at the default expansion over `_RECALL_ROWS` rows. Quantization
# is the only thing that moves them: a stored `e2m3` row keeps three mantissa bits, so ties it
# cannot distinguish are resolved arbitrarily.
_RECALL_FLOORS: dict[Scalar, float] = {
    Scalar.F64: 0.99,
    Scalar.F32: 0.99,
    Scalar.BF16: 0.98,
    Scalar.F16: 0.98,
    Scalar.E5M2: 0.85,
    Scalar.E4M3: 0.90,
    Scalar.E3M2: 0.70,
    Scalar.E2M3: 0.60,
    Scalar.I8: 0.90,
    Scalar.U8: 0.85,
    Scalar.U1X8: 0.60,
}

_RECALL_ROWS = 512


def recall_floor(quantization: Scalar) -> float:
    """The share of true nearest neighbors a build in this scalar type must still return."""
    return _RECALL_FLOORS.get(quantization, 0.90)


# Absolute and relative tolerances for a distance checked against the oracle, by the scalar type
# the index stored. Integers get an absolute slack and no relative one, since their error is a
# rounding step rather than a fraction.
_DISTANCE_TOLERANCES: dict[Scalar, tuple[float, float]] = {
    Scalar.F64: (1e-6, 1e-6),
    Scalar.F32: (1e-4, 1e-4),
    Scalar.BF16: (1e-1, 1e-1),
    Scalar.F16: (1e-2, 1e-2),
    Scalar.E5M2: (0.3, 0.3),
    Scalar.E4M3: (0.2, 0.2),
    Scalar.E3M2: (0.5, 0.5),
    Scalar.E2M3: (0.5, 0.5),
    Scalar.I8: (0.1, 0.1),
    Scalar.U8: (0.1, 0.1),
}


def distance_tolerances(quantization: Scalar) -> tuple[float, float]:
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
    metric: DenseMetric = DenseMetric.ANGULAR,
    quantization: Scalar = Scalar.F32,
    seed: int = 0,
    **options,
) -> tuple[DenseIndex, np.ndarray]:
    """A reserved index holding *count* reproducible rows under the identities ``0..count``, inserted on *executor*.

    Returns the index and the rows it was built from, so a caller can query with either.
    """
    index = DenseIndex.init_ram(dimensions=dimensions, metric=metric, quantization=quantization, **options)
    index.reserve(count)
    vectors = random_vectors(count, dimensions, metric=metric, seed=seed)
    index.on(executor).insert(integers_to_keys(np.arange(count, dtype=np.uint64)), vectors)
    assert len(index) == count
    return index, vectors


def matches_outputs(queries: int, wanted: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three zeroed caller-owned buffers a `Matches` reads through, for *queries* rows of *wanted* columns."""
    return (
        np.zeros((queries, wanted, 16), dtype=np.uint8),
        np.zeros((queries, wanted), dtype=np.float64),
        np.zeros(queries, dtype=np.uint64),
    )


def measured_recall(
    index: DenseIndex, vectors, queries, wanted: int, executor: Executor, *, metric: DenseMetric
) -> float:
    """The share of true nearest neighbors the index returned, against the NumKong oracle."""
    truth = exact_matches(vectors, queries, wanted, metric=metric)
    found = keys_to_integers(index.on(executor).search(queries, wanted).keys)
    hits = sum(len(set(found[row]) & set(truth[row])) for row in range(len(truth)))
    return hits / (len(truth) * wanted)


# endregion Builders
