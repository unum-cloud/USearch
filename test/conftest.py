"""Session-wide configuration for the Python side of the USearch test suite.

Holds only what pytest itself loads: the report header, the seeding fixture, and the executors
every fanned-out verb runs on, which every module inherits without importing anything. Matrices,
oracle, and gates are plain values and functions, so they live in `test.base` and are imported
by name.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Callable, Iterator

import pytest

import usearch

# Fixed at 42 unless `USEARCH_SEED` names another, or `random` to draw one; the banner logs it, so
# a failure that only appears sometimes still names the draw that produced it.
_REQUESTED_SEED = os.environ.get("USEARCH_SEED", "42")
try:
    _RUN_SEED = int.from_bytes(os.urandom(4), "little") if _REQUESTED_SEED == "random" else int(_REQUESTED_SEED)
except ValueError:
    raise SystemExit(f'USEARCH_SEED="{_REQUESTED_SEED}" does not parse') from None


def pytest_report_header() -> list[str]:
    """What this run is actually exercising, printed where pytest prints its own header.

    A `print` from a session fixture is swallowed by pytest's capture unless `-s` is passed;
    this hook always shows, and survives `-q`, xdist, and report capture.
    """
    return [
        f"platform: {platform.platform()}",
        f"python: {platform.python_version()}",
        f"usearch: {usearch.version()} from {usearch.DenseIndex.__module__}",
        f"capabilities compiled: {usearch.capabilities_compiled()}",
        f"capabilities available: {usearch.capabilities_available()}",
        f"seed: {_RUN_SEED}, pin with USEARCH_SEED",
    ]


@pytest.fixture
def seed(__pytest_repeat_step_number: int | None) -> int:
    """A per-test seed that moves with the repeat step, so repetitions differ but reproduce.

    The step counter carries no default on purpose: pytest injects only the parameters without
    one, so a default would replay the first step's draw on every repeat. pytest-repeat hands
    `None` to a test that is not repeated.
    """
    return _RUN_SEED + (__pytest_repeat_step_number or 0)


@pytest.fixture(scope="session")
def executor() -> Iterator[usearch.Executor]:
    """The workers every fanned-out verb runs on, shared by every test that drives one at a time.

    One per session, because an executor admits a single driver and pytest runs a module's tests
    in sequence; a test that genuinely overlaps calls takes `spawn_executor` instead.
    """
    pool = usearch.Executor.for_ram(threads=2)
    yield pool
    pool.drain()


@pytest.fixture
def spawn_executor() -> Iterator[Callable[[], usearch.Executor]]:
    """Hands out one executor per call, for a test whose threads each need their own driver."""
    spawned: list[usearch.Executor] = []

    def spawn() -> usearch.Executor:
        spawned.append(usearch.Executor.for_ram(threads=1))
        return spawned[-1]

    yield spawn
    for pool in spawned:
        pool.drain()
