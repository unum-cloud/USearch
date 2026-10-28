"""Session-wide configuration for the Python side of the USearch test suite.

Holds only what pytest itself loads: the report header, the filter, the seeding fixtures, and the
executors every fanned-out verb runs on, which every module inherits without importing anything.
Settings, matrices, oracle, and gates are plain values and functions, so they live in `base`
and are imported by name.
"""

import platform
import random
from collections.abc import Callable, Iterator

import pytest
from base import SETTINGS

import usearch


def pytest_report_header() -> list[str]:
    """What this run is actually exercising, printed where pytest prints its own header.

    A `print` from a session fixture is swallowed by pytest's capture unless `-s` is passed;
    this hook always shows, and survives `-q`, xdist, and report capture.
    """
    return [
        f"- Platform: {platform.platform()}",
        f"- USearch: {usearch.version()} from {usearch.lib.__file__}",
        f"- Capabilities compiled: {usearch.capabilities_compiled()}",
        f"- Capabilities available: {usearch.capabilities_available()}",
        f"- Seed: {SETTINGS.seed}",
        f"- Filter: {getattr(SETTINGS.filter, 'pattern', SETTINGS.filter) or 'none'}",
        f"- Dims: {','.join(map(str, SETTINGS.dims))}",
    ]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keeps only the tests whose node id `USEARCH_FILTER` selects, on top of any `-k`."""
    deselected = [item for item in items if not SETTINGS.selects(item.nodeid)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if SETTINGS.selects(item.nodeid)]


@pytest.fixture
def seed(__pytest_repeat_step_number: int | None) -> int:
    """A per-test seed that moves with the repeat step, so repetitions differ but reproduce.

    The step counter carries no default on purpose: pytest injects only the parameters without
    one, so a default would replay the first step's draw on every repeat. pytest-repeat hands
    `None` to a test that is not repeated.
    """
    return SETTINGS.seed + (__pytest_repeat_step_number or 0)


@pytest.fixture
def rng(seed: int) -> random.Random:
    """An RNG private to this test, so a neighbour's draws cannot shift this one's."""
    return random.Random(seed)


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
