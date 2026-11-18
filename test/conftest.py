"""Session-wide configuration for the Python side of the USearch test suite.

Holds only what pytest itself loads: the report header, the case cap, the seeding and unraisable
fixtures, and the executors every fanned-out verb runs on, which every module inherits without importing anything.
Settings, matrices, oracle, and gates are plain values and functions, so they live in `base` and are imported by
name.

File: test/conftest.py
Author: Ash Vardanian
Date: August 16, 2026
"""

import collections
import enum
import platform
import random
import sys
from collections.abc import Callable, Iterator

import pytest
from base import BACKEND_AXES, MAX_CASES_PER_TEST, SETTINGS, stream_key

import usearch


def pytest_report_header() -> list[str]:
    """What this run is actually exercising, printed where pytest prints its own header.

    A `print` from a session fixture is swallowed by pytest's capture unless `-s` is passed;
    this hook always shows, and survives `-q`, xdist, and report capture.
    """
    return [
        f"USearch {usearch.version()} from {usearch.lib.__file__}",
        f"- Compiled for: {usearch.capabilities_compiled()}",
        f"- This machine: {usearch.capabilities_available()} on {platform.platform()}",
        f"- Seed: {SETTINGS.seed}, scale: {SETTINGS.scale:g}",
        f"- Dims: {','.join(map(str, SETTINGS.dims))}",
    ]


# region Shared Pytest
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Refuses a test whose cases outgrow MAX_CASES_PER_TEST."""
    variants: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for item in items:
        callspec = getattr(item, "callspec", None)
        parameters = dict(callspec.params) if callspec is not None else {}
        parameters.pop("__pytest_repeat_step_number", None)
        kept = {key: value for key, value in parameters.items() if key not in BACKEND_AXES}
        variants[(item.module.__name__, getattr(item, "originalname", item.name))].add(repr(kept))
    oversized = [
        f"{module}::{name} has {len(cases)}"
        for (module, name), cases in variants.items()
        if len(cases) > MAX_CASES_PER_TEST
    ]
    if oversized:
        raise pytest.UsageError(f"over MAX_CASES_PER_TEST={MAX_CASES_PER_TEST}: " + "; ".join(oversized))


def pytest_make_parametrize_id(config: pytest.Config, val: object, argname: str) -> str | None:
    """Names enum parameters as the code spells them, `Backend.LINUX_URING618`, so an id greps to its source."""
    return f"{type(val).__name__}.{val.name}" if isinstance(val, enum.Enum) else None


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> int:
    """Share inputs across backends while keeping other parameters and repeat steps distinct."""
    callspec = getattr(request.node, "callspec", None)
    parameters = dict(callspec.params) if callspec is not None else {}
    repeat_step = parameters.pop("__pytest_repeat_step_number", 0)
    name = request.node.originalname or request.node.name
    name += repr({key: value for key, value in parameters.items() if key not in BACKEND_AXES})
    return stream_key(SETTINGS.seed, name, repeat_step)


@pytest.fixture
def rng(seed: int) -> random.Random:
    """A generator private to this test, so a neighbour's draws cannot shift this one's."""
    return random.Random(seed)


@pytest.fixture
def unraisable(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Collects what the interpreter reports as unraisable, like a finalizer's exception, for the test to check."""
    seen: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    return seen


# endregion Shared Pytest


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
