"""Shared harness for the Python side of the USearch test suite.

Holds the settings every module reads, the matrices it parametrizes over, the oracle it checks against, the
recall floors it asserts, and the gates that skip what this machine or build cannot reach. Imported by name from
each test module; only what pytest loads for itself lives in `conftest`.

File: test/base.py
Author: Ash Vardanian
Date: August 16, 2026
"""

import os
import random
import re
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import numpy.typing as npt
import pytest

import usearch
from usearch import Arena, DenseIndex, DenseMetric, Executor, Scalar, UnsupportedError
from usearch.eval import exact_matches, random_vectors
from usearch.keys import integers_to_keys, keys_to_integers

# region Shared Settings
_UINT64_MASK = 2**64 - 1


def env_text(variable: str) -> str | None:
    """The stripped text of the environment `variable`, or None when unset or blank."""
    text = os.environ.get(variable, "").strip()
    return text or None


def exit_unparsed(variable: str, text: str, expected: str) -> NoReturn:
    """Stops the session on a setting that does not parse, naming it as the C and C++ suites do."""
    raise SystemExit(f'{variable}="{text}" does not parse, expected {expected}')


def env_seed(variable: str) -> int:
    """An unsigned 64-bit seed: 42 when unset, drawn afresh for `random`."""
    text = env_text(variable)
    if text is None:
        return 42
    if text == "random":
        return random.SystemRandom().getrandbits(64)
    if not text.isdigit() or int(text) > _UINT64_MASK:
        exit_unparsed(variable, text, "an unsigned integer or random")
    return int(text)


def env_real(variable: str, fallback: float) -> float:
    """A positive real, `fallback` when unset."""
    text = env_text(variable)
    if text is None:
        return fallback
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if not value > 0 or value == float("inf"):
        exit_unparsed(variable, text, "a positive number like 0.1 or 10")
    return value


def env_count(variable: str, fallback: int) -> int:
    """A whole number, `fallback` when unset."""
    text = env_text(variable)
    if text is None:
        return fallback
    if not text.isdigit():
        exit_unparsed(variable, text, "a whole number")
    return int(text)


def env_flag(variable: str, fallback: bool = False) -> bool:
    """A switch, `fallback` when unset: 0, false, off and no turn it off."""
    text = env_text(variable)
    if text is None:
        return fallback
    return text not in {"0", "false", "off", "no"}


@dataclass(frozen=True)
class Filter:
    """Which node ids a run selects: a regex search when the pattern compiles, else a substring, all when empty."""

    pattern: str
    regex: re.Pattern[str] | None

    def selects(self, nodeid: str) -> bool:
        """Whether `nodeid` passes the filter."""
        if not self.pattern:
            return True
        if self.regex is not None:
            return self.regex.search(nodeid) is not None
        return self.pattern in nodeid


def env_filter(variable: str) -> Filter:
    """The filter the environment `variable` spells, compiled once."""
    pattern = env_text(variable) or ""
    try:
        regex = re.compile(pattern) if pattern else None
    except re.error:
        regex = None
    return Filter(pattern, regex)


def mix(value: int) -> int:
    """SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def stream_key(seed: int, name: str, index: int = 0) -> int:
    """The `index`-th stream of the family `name` under `seed`, identical in C, C++, Rust and Mojo."""
    hashed = 0xCBF29CE484222325
    for byte in name.encode():
        hashed = ((hashed ^ byte) * 0x100000001B3) & _UINT64_MASK
    return mix((mix(seed ^ hashed) + index) & _UINT64_MASK)


# endregion Shared Settings


def env_dims() -> list[int]:
    """`USEARCH_DIMS`, one count like `128` or a comma list like `64,128`.

    Unset, the counts bracket the SIMD register widths, where tail handling is most likely to diverge: below one
    register, straddling one, and several whole ones.
    """
    text = env_text("USEARCH_DIMS")
    if text is None:
        return [1, 3, 15, 16, 17, 63, 64, 65, 97, 256]
    pieces = text.split(",")
    if not all(piece.isdigit() and int(piece) > 0 for piece in pieces):
        exit_unparsed("USEARCH_DIMS", text, "positive counts like 64,128")
    return [int(piece) for piece in pieces]


@dataclass(frozen=True)
class Settings:
    """Every `USEARCH_*` variable the Python suite reads, parsed once at import."""

    seed: int
    filter: Filter
    scale: float
    dims: list[int]


SETTINGS = Settings(
    seed=env_seed("USEARCH_SEED"),
    filter=env_filter("USEARCH_FILTER"),
    scale=env_real("USEARCH_SCALE", 1.0),
    dims=env_dims(),
)

BACKEND_AXES: set[str] = set()
"""The axes the seed and the case cap leave out, so every backend draws the same inputs: none, as the Python suite
sweeps no backend axis."""

# region Matrices

BATCH_SIZES: list[int] = [1, 7, 1024]
"""Batch sizes bracketing the internal batching: one row, an odd few, and enough to parallelize."""

QUANTIZATIONS: list[Scalar] = [
    Scalar.BF16,
    Scalar.F16,
    Scalar.E5M2,
    Scalar.E4M3,
    Scalar.E3M2,
    Scalar.E2M3,
    Scalar.I8,
    Scalar.U8,
]
"""Every scalar type the index stores, widest dynamic range first - floats descending, then
signed integers, then unsigned. The order every dtype-indexed list in the project takes."""

SPATIAL_METRICS: list[DenseMetric] = [DenseMetric.INNER, DenseMetric.ANGULAR, DenseMetric.EUCLIDEAN]
"""The metrics over real-valued rows."""

INPUT_DTYPES: list[type[np.generic]] = [np.float64, np.float32, np.float16, np.int8, np.uint8]
"""The input scalar types a caller is most likely to hand us, none of which need be the stored one."""

MAX_CASES_PER_TEST = 60
"""The most cases one test function may expand to, today's largest, so a new axis cannot multiply a matrix past it
unnoticed."""

# endregion Matrices

# region Tolerances

_RECALL_FLOORS: dict[Scalar, float] = {
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
"""Recall floors for a self-query at the default expansion over 512 rows. Quantization is the only
thing that moves them: a stored `e2m3` row keeps three mantissa bits, so ties it cannot distinguish
are resolved arbitrarily."""


def recall_floor(quantization: Scalar) -> float:
    """The share of true nearest neighbors a build in this scalar type must still return."""
    return _RECALL_FLOORS.get(quantization, 0.90)


_DISTANCE_TOLERANCES: dict[Scalar, tuple[float, float]] = {
    Scalar.BF16: (1e-1, 1e-1),
    Scalar.F16: (1e-2, 1e-2),
    Scalar.E5M2: (0.3, 0.3),
    Scalar.E4M3: (0.2, 0.2),
    Scalar.E3M2: (0.5, 0.5),
    Scalar.E2M3: (0.5, 0.5),
    Scalar.I8: (0.1, 0.1),
    Scalar.U8: (0.1, 0.1),
}
"""Absolute and relative tolerances for a distance checked against the oracle, by the scalar type
the index stored. Integers get an absolute slack and no relative one, since their error is a
rounding step rather than a fraction."""


def distance_tolerances(quantization: Scalar) -> tuple[float, float]:
    """The `(atol, rtol)` a distance in this stored scalar type may drift from the oracle."""
    return _DISTANCE_TOLERANCES.get(quantization, (0.5, 0.5))


BF16_EPSILON: float = 2**-7
"""The gap between 1 and the next `bf16`, so a value read back from a `bf16` row moves by less."""


def bf16_words(rows: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """*rows* rounded to the nearest even `bf16`, as the `uint16` words a `Scalar.BF16` buffer holds."""
    bits = np.ascontiguousarray(rows, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_rounded(rows: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """The `float32` values a `bf16` index holds for *rows*."""
    return (bf16_words(rows).astype(np.uint32) << 16).view(np.float32)


# endregion Tolerances

# region Gates


def requires_capability(codename: str) -> pytest.MarkDecorator:
    """Skips when dispatch on this machine would not reach that kernel family."""
    return pytest.mark.skipif(codename not in usearch.capabilities_available().split(","), reason=f"needs {codename}")


# endregion Gates

# region Builders


def init_ram_or_skip(**options) -> DenseIndex:
    """An index of *options*, or a skipped test when this build's `USEARCH_DENSE_DTYPES` left its quantization out."""
    try:
        return DenseIndex.init_ram(**options)
    except UnsupportedError:
        pytest.skip(f"built without {options['quantization'].name.lower()} in USEARCH_DENSE_DTYPES")


def build_index(
    count: int,
    dimensions: int,
    executor: Executor,
    *,
    metric: DenseMetric = DenseMetric.ANGULAR,
    quantization: Scalar = Scalar.BF16,
    seed: int,
    **options,
) -> tuple[DenseIndex, npt.NDArray[np.generic]]:
    """A reserved index holding *count* reproducible rows under the identities ``0..count``, inserted on *executor*.

    Returns the index and the rows it was built from, so a caller can query with either: `float32`
    rows, integer-valued for `i8` and `u8` storage, or packed `uint8` bits for a bitwise *metric*,
    hence the generic dtype. The *seed* is the test's own, from the `seed` fixture. Skips when this
    build's `USEARCH_DENSE_DTYPES` left *quantization* out.
    """
    index = init_ram_or_skip(dimensions=dimensions, metric=metric, quantization=quantization, **options)
    index.reserve(count)
    vectors = random_vectors(count, dimensions, metric=metric, seed=seed)
    # Integral storage is a saturating cast, so its rows must already be integers in range.
    if quantization == Scalar.I8:
        vectors = np.round(vectors * 100)
    elif quantization == Scalar.U8:
        vectors = np.round((vectors + 1) * 100)
    index.on(executor).insert(integers_to_keys(np.arange(count, dtype=np.uint64)), vectors)
    assert len(index) == count
    return index, vectors


def measured_recall(
    index: DenseIndex, vectors, queries, wanted: int, executor: Executor, *, metric: DenseMetric
) -> float:
    """The share of true nearest neighbors the index returned, against the NumKong oracle."""
    truth = exact_matches(vectors, queries, wanted, metric=metric)
    found = keys_to_integers(index.on(executor).search(queries, wanted, Arena(executor)).keys)
    hits = sum(len(set(found[query]) & set(truth[query])) for query in range(len(truth)))
    return hits / (len(truth) * wanted)


# endregion Builders
