"""GIL-release and progress-callback contract for the Python `Index` binding.

USearch releases the GIL around long C++ operations so that *other* Python
work (NumPy ops, file I/O, or another Python thread doing unrelated work) can
make progress concurrently. The `Index` API itself is single-threaded from
Python's perspective - one Python thread per index at a time.

These tests assert the GIL contract end-to-end by:

* Spawning a background Python thread that increments a counter in a tight
  loop while the main thread runs a long USearch op. If the GIL is actually
  released, the counter advances meaningfully during the op.
* Validating that progress callbacks fire across the GIL-release boundary -
  the callback runs from a C++ worker thread that must reacquire the GIL to
  invoke the Python callable, mutate a list, and return a bool.
* Validating that returning `False` from the progress callback terminates the
  operation cleanly, surfacing as a Python `RuntimeError`.
"""

import threading
import time

import numpy as np
import pytest

from usearch.index import Index


def _background_counter():
    """Returns (start_fn, stop_fn, count_fn) for a tight-loop Python thread."""
    counter = [0]
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            counter[0] += 1

    thread = threading.Thread(target=loop, daemon=True)

    def start():
        thread.start()

    def stop_and_join():
        stop.set()
        thread.join()

    return start, stop_and_join, lambda: counter[0]


def _big_random_batch(n: int, ndim: int, seed: int = 42):
    rng = np.random.default_rng(seed=seed)
    keys = np.arange(n, dtype=np.uint64)
    vectors = rng.standard_normal((n, ndim), dtype=np.float32)
    return keys, vectors


# Lower bound on background-counter ticks during one short USearch op. Modern
# hardware loops the trivial `counter[0] += 1` body well over a million times
# per second; 10k is a conservative floor that distinguishes "GIL released"
# from "GIL held" without making the test slow on tiny inputs.
_GIL_TICK_FLOOR = 10_000


def test_gil_released_during_add():
    start, stop_and_join, count = _background_counter()
    start()

    idx = Index(ndim=64, dtype="f32")
    keys, vectors = _big_random_batch(2_000, 64)

    before = count()
    t0 = time.perf_counter()
    idx.add(keys, vectors, threads=4)
    elapsed = time.perf_counter() - t0
    after = count()
    stop_and_join()

    advancement = after - before
    assert advancement > _GIL_TICK_FLOOR, (
        f"GIL appears held: only {advancement:,} background ticks during a "
        f"{elapsed:.3f}s add. Expected > {_GIL_TICK_FLOOR:,}."
    )


def test_gil_released_during_search():
    idx = Index(ndim=64, dtype="f32")
    keys, vectors = _big_random_batch(1_500, 64)
    idx.add(keys, vectors, threads=4)

    start, stop_and_join, count = _background_counter()
    start()

    _, queries = _big_random_batch(1_000, 64, seed=7)
    before = count()
    t0 = time.perf_counter()
    idx.search(queries, 10, threads=4)
    elapsed = time.perf_counter() - t0
    after = count()
    stop_and_join()

    advancement = after - before
    assert advancement > _GIL_TICK_FLOOR, (
        f"GIL appears held during search: only {advancement:,} background ticks during a {elapsed:.3f}s search."
    )


def test_progress_callback_fires_and_completes():
    """The progress callback runs from a C++ worker thread that must reacquire
    the GIL before invoking the Python callable. It must be able to mutate a
    Python list and return a bool without crashing."""

    idx = Index(ndim=64, dtype="f32")
    keys, vectors = _big_random_batch(2_000, 64)

    invocations = []

    def progress(done: int, total: int) -> bool:
        invocations.append((done, total))
        return True

    idx.add(keys, vectors, threads=4, progress=progress)

    assert invocations, "progress callback was never invoked"
    last_done, last_total = invocations[-1]
    assert last_done == last_total == len(keys), (
        f"final progress {(last_done, last_total)} != ({len(keys)}, {len(keys)})"
    )
    # Done counters should be non-decreasing across the run.
    for (d_prev, _), (d_next, _) in zip(invocations, invocations[1:]):
        assert d_prev <= d_next, f"progress went backwards: {d_prev} -> {d_next}"


def test_progress_callback_can_cancel():
    """Returning `False` from the progress callback terminates the op cleanly
    and surfaces as a Python `RuntimeError` - no segfault, no UB."""

    idx = Index(ndim=64, dtype="f32")
    keys, vectors = _big_random_batch(10_000, 64)

    seen = []

    def progress(done: int, total: int) -> bool:
        seen.append(done)
        # Cancel after a few progress reports so we know the path is exercised.
        return len(seen) < 3

    with pytest.raises(RuntimeError, match="terminated"):
        idx.add(keys, vectors, threads=4, progress=progress)

    # Index may be partially populated; the important property is no crash and
    # that the callback was actually invoked the expected number of times.
    assert len(seen) >= 3
    assert len(idx) <= len(keys)


def test_gil_released_with_progress_callback():
    """Combined: background Python thread runs while the main thread is in
    `add()` with an active progress callback. Both must work simultaneously."""

    start, stop_and_join, count = _background_counter()
    start()

    idx = Index(ndim=64, dtype="f32")
    keys, vectors = _big_random_batch(2_000, 64)

    invocations = []

    def progress(done: int, total: int) -> bool:
        invocations.append((done, total))
        return True

    before = count()
    idx.add(keys, vectors, threads=4, progress=progress)
    after = count()
    stop_and_join()

    assert after - before > _GIL_TICK_FLOOR, (
        "background thread didn't advance during add - GIL likely held while callback was active"
    )
    assert invocations and invocations[-1] == (len(keys), len(keys))


def test_concurrent_access_serializes_safely():
    """The documented contract is one Python thread per index; the binding
    enforces it with an internal mutex so accidental concurrent access from
    multiple Python threads serializes instead of crashing. Mix adds with
    disjoint key ranges, searches, and lock-free getters; assert no thread
    errors and that all keys from all `add` workers landed in the index."""

    idx = Index(ndim=64, dtype="f32")
    per_thread = 500
    barrier = threading.Barrier(6)
    errors: list[str] = []

    def run(target, *args):
        def wrapped():
            try:
                barrier.wait()
                target(*args)
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
        return threading.Thread(target=wrapped)

    def adder(tid: int):
        rng = np.random.default_rng(seed=tid)
        base = tid * per_thread
        keys = np.arange(base, base + per_thread, dtype=np.uint64)
        idx.add(keys, rng.standard_normal((per_thread, 64), dtype=np.float32))

    def searcher(tid: int):
        rng = np.random.default_rng(seed=1000 + tid)
        for _ in range(50):
            idx.search(rng.standard_normal((1, 64), dtype=np.float32), 3)

    def getters():
        for i in range(1000):
            _ = i in idx
            _ = len(idx)

    threads = [run(adder, t) for t in range(3)] + [run(searcher, t) for t in range(2)] + [run(getters)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, "thread errors:\n  " + "\n  ".join(errors)
    assert len(idx) == 3 * per_thread
