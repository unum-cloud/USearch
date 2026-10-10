"""Vector search over equidimensional dense embeddings - the Python twin of `test/dense.cpp`.

Every recall claim is checked against `usearch.eval.exact_matches`, which scores with NumKong's
kernels in the index's own metric vocabulary, so a number here compares like with like.

Run:
    python -m pytest test/dense.py -v
"""

from __future__ import annotations

import asyncio
import os
from itertools import pairwise
from operator import attrgetter
from threading import Thread, Timer

import numpy as np
import pytest
from base import (
    SETTINGS,
    batch_sizes,
    build_index,
    distance_tolerances,
    input_dtypes,
    matches_outputs,
    measured_recall,
    quantizations,
    recall_floor,
    spatial_metrics,
)

import numkong as nk
from usearch import (
    Arena,
    CancelledError,
    CapacityOverflowError,
    ConfigMismatchError,
    ContendedError,
    Deferred,
    DeferredOn,
    DenseIndex,
    DenseIndexDescription,
    DenseMetadata,
    DenseMetric,
    DenseReachability,
    DenseTraversalStats,
    Executor,
    InvalidArgumentError,
    Key,
    Match,
    Matches,
    Monolith,
    Mood,
    On,
    Products,
    ReadOnlyError,
    Residency,
    Scalar,
    ShapeMismatchError,
    Status,
    UninitializedError,
    UnknownKindError,
    Vectors,
)
from usearch.eval import random_vectors, self_recall
from usearch.keys import integers_to_keys, keys_to_integers

# region Construction


@pytest.mark.parametrize("dimensions", SETTINGS.dims)
@pytest.mark.parametrize("metric", spatial_metrics, ids=attrgetter("name"))
def test_construction_reports_its_geometry(dimensions: int, metric: DenseMetric):
    """Every setting the factory took reads back off the handle, the modes as their enums."""
    index = DenseIndex.init_ram(dimensions=dimensions, metric=metric, quantization=Scalar.BF16, connectivity=16)
    assert index.dimensions == dimensions
    assert index.connectivity == 16
    assert len(index) == 0
    assert isinstance(index.capabilities, str)
    assert repr(index) == f"DenseIndex(dimensions={dimensions}, size=0)"
    description = index.describe()
    assert isinstance(description, DenseIndexDescription)
    assert description.metadata == (dimensions, metric, Scalar.BF16, 16)
    assert description.metadata.metric is metric
    assert description.size == 0


@pytest.mark.parametrize(
    ("options", "failure"),
    [
        ({"metric": "angular"}, TypeError),  # Modes are enums, never strings.
        ({"quantization": "bf16"}, TypeError),
        ({"quantization": True}, TypeError),  # A bool would read as an enumerator.
        ({"metric": 99}, UnknownKindError),  # An integer no enumerator carries.
        ({"residency": Scalar.F32}, TypeError),  # The factory chooses the residency; no keyword does.
        ({"devices": [0]}, TypeError),  # An arm belonging to another residency.
        ({"threads": 2}, TypeError),  # An index owns rows; an `Executor` owns workers.
        ({"expansion_add": 64}, TypeError),  # A keyword the surface does not have.
        ({"ndim": 8}, TypeError),  # Renamed to 'dimensions'.
    ],
)
def test_construction_rejects_unknown_options(options: dict, failure: type):
    """A name or a mode the factory does not have is refused, not quietly ignored."""
    with pytest.raises(failure):
        DenseIndex.init_ram(dimensions=8, **options)


def test_construction_takes_three_positionals():
    """The three settings every caller names may be positional, and only once; the rest must be keywords."""
    index = DenseIndex.init_ram(32, DenseMetric.EUCLIDEAN, Scalar.F32)
    assert index.dimensions == 32
    with pytest.raises(TypeError):
        DenseIndex.init_ram(32, DenseMetric.EUCLIDEAN, Scalar.F32, 16)
    with pytest.raises(TypeError):
        DenseIndex.init_ram(32, dimensions=16)


@pytest.mark.parametrize("handle_type", [DenseIndex, Executor, On, DeferredOn])
def test_direct_construction_names_the_factories(handle_type: type):
    """No handle is built by calling its type; the refusal says what to call instead."""
    with pytest.raises(TypeError):
        handle_type()


# endregion Construction

# region Executors


def test_vram_executor_takes_no_thread_count():
    """One host thread drives one stream per device, so the device list is the pool's only width."""
    with pytest.raises(TypeError):
        Executor.for_vram(threads=4)


def test_executor_records_its_residency():
    """An executor names where its calls compute, which is where the outputs they size land."""
    assert Executor.for_ram(threads=1).residency is Residency.RAM


def test_ram_executor_takes_memory_domain_ordinals():
    """The NUMA nodes its workers stay local to arrive as ordinals, and nothing else is accepted."""
    Executor.for_ram(memory_domains=[0]).drain()
    with pytest.raises(TypeError):
        Executor.for_ram(memory_domains=["x"])
    with pytest.raises(ValueError):
        Executor.for_ram(memory_domains=[2**40])


def test_drain_takes_a_mood():
    """The idle mood is a `Mood`, and neither a name nor another integer is guessed at."""
    pool = Executor.for_ram(threads=1, deferred_limit=4)
    pool.drain(Mood.SPINNING)
    pool.drain()
    with pytest.raises(TypeError):
        pool.drain("parked")
    with pytest.raises(ValueError):
        pool.drain(7)


def test_borrowed_pool_must_be_named():
    """A pool arrives by address, and a null one is refused rather than dereferenced."""
    with pytest.raises(InvalidArgumentError):
        Executor.on_pool(0)


def test_executor_closes_with_its_block():
    """Leaving a `with` block waits for the executor's calls and closes it, so a later receiver is refused."""
    index, _ = build_index(8, 8, Executor.for_ram(threads=1))
    with Executor.for_ram(threads=1) as pool:
        assert index.on(pool).count(0) == 1
    with pytest.raises(UninitializedError):
        index.on(pool)


# One entry per verb, called on the index itself, which runs it on the caller's thread.
_CALLER_THREAD_VERBS = [
    ("insert", lambda index, rows: index.insert([9], rows[:1])),
    ("refine", lambda index, rows: index.refine()),
    ("search", lambda index, rows: index.search(rows[:1], 1)),
    ("search_stream", lambda index, rows: next(index.search_stream(rows[0]))),
    ("save_buffer", lambda index, rows: index.save_buffer()),
    ("get", lambda index, rows: index.get(0)),
    ("count", lambda index, rows: index.count(0)),
]


@pytest.mark.parametrize(("verb", "call"), _CALLER_THREAD_VERBS, ids=[name for name, _ in _CALLER_THREAD_VERBS])
def test_verbs_run_on_the_callers_thread(executor, verb: str, call):
    """The index itself is a receiver too: its verbs run on the caller's thread rather than refusing."""
    index, vectors = build_index(8, 8, executor)
    call(index, vectors)


@pytest.mark.parametrize("wrong", [2, "ram", None])
def test_receivers_take_only_an_executor(executor, wrong):
    """A pool is a handle, not a thread count or a name, so anything else is refused where it is named."""
    index, _ = build_index(8, 8, executor)
    with pytest.raises(TypeError):
        index.on(wrong)
    with pytest.raises(TypeError):
        index.deferred(wrong)


def test_executor_is_borrowed_for_the_call(executor):
    """A verb borrows the workers for its call and never retains them, so a temporary one serves."""
    index, vectors = build_index(16, 8, Executor.for_ram())
    assert len(index) == 16
    assert len(index.on(executor).search(vectors[:1], 1)) == 1


def test_one_executor_serves_several_indexes(executor):
    """An index owns rows and an executor owns workers, so one pool drives as many as asked."""
    first, vectors = build_index(16, 8, executor)
    second, _ = build_index(16, 8, executor, seed=1)
    assert len(first) == len(second) == 16
    assert len(first.on(executor).search(vectors[:2], 1)) == 2


def test_concurrent_searches_take_one_executor_apiece(executor, spawn_executor):
    """A handle admits any number of callers, and each thread here blocks on an executor of its own."""
    index, vectors = build_index(128, 16, executor)
    pools = [spawn_executor() for _ in range(4)]
    answers: dict[int, Matches] = {}

    def resolve(thread_index: int):
        answers[thread_index] = index.on(pools[thread_index]).search(vectors[:8], 4)

    threads = [Thread(target=resolve, args=(thread_index,)) for thread_index in range(len(pools))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(answers) == len(pools), "every thread must have resolved its own batch"
    for matches in answers.values():
        assert np.asarray(matches.counts).shape == (8,)


# endregion Executors

# region Receivers


def test_receivers_answer_alike(executor):
    """The receiver chooses where the call runs and the verb stays, so all three answer the same matches."""
    index, vectors = build_index(64, 16, executor)
    on_caller = keys_to_integers(index.search(vectors[:4], 5).keys)
    on_executor = keys_to_integers(index.on(executor).search(vectors[:4], 5).keys)
    call = index.deferred(executor).search(vectors[:4], 5)
    assert isinstance(call, Deferred)
    assert call.status == Status.DEFERRED
    executor.drain()
    deferred = keys_to_integers(call.wait().keys)
    assert np.array_equal(on_caller, on_executor)
    assert np.array_equal(on_caller, deferred)


def test_deferred_call_is_deferred_until_drained(executor):
    """A submitted call reads `DEFERRED` until the executor runs it, then answers what a blocking one would."""
    index = DenseIndex.init_ram(16, DenseMetric.ANGULAR, Scalar.F32)
    index.reserve(8)
    call = index.deferred(executor).insert(np.arange(8), random_vectors(8, 16))
    assert call.status is Status.DEFERRED
    assert repr(call) == "Deferred(status=deferred)"
    executor.drain()
    assert call.status is Status.SUCCESS
    assert call.wait() is None
    assert call.wait() is None, "a final call answers as many times as it is asked"
    assert len(index) == 8


def test_deferred_call_owns_its_inputs(executor):
    """The rows a deferred call borrows stay exported inside its `Deferred`, so the caller may drop its own names."""
    index = DenseIndex.init_ram(16, DenseMetric.ANGULAR, Scalar.F32)
    index.reserve(4)
    rows = random_vectors(4, 16)
    call = index.deferred(executor).insert(list(range(4)), rows.copy())
    assert call.status == Status.DEFERRED
    executor.drain()
    call.wait()
    assert np.allclose(np.asarray(index.get(2)), rows[2:3], atol=1e-6)


def test_deferred_call_raises_at_wait(executor):
    """Arguments the binding refuses raise at the call; the engine's own refusals raise once waited for."""
    index, vectors = build_index(16, 8, executor)
    with pytest.raises(ShapeMismatchError):
        index.deferred(executor).insert([1, 2, 3], vectors[:2])
    call = index.deferred(executor).search(vectors[:1, :4], 1)
    assert call.status == Status.DEFERRED
    executor.drain()
    assert call.status is Status.SHAPE_MISMATCH
    for _ in range(2):
        with pytest.raises(ShapeMismatchError) as raised:
            call.wait()
        assert raised.value.status is Status.SHAPE_MISMATCH


def test_deferred_call_awaits(executor):
    """`await` runs `wait()` on the event loop's own executor, so a coroutine suspends rather than blocks."""
    index, vectors = build_index(32, 8, executor)

    async def search() -> Matches:
        call = index.deferred(executor).search(vectors[:2], 3)
        assert call.status == Status.DEFERRED
        executor.drain()
        return await call

    matches = asyncio.run(search())
    assert keys_to_integers(matches.keys)[:, 0].tolist() == [0, 1]


def test_blocking_call_runs_after_earlier_deferred_ones(executor):
    """Calls on one executor start in submission order, so a blocking one sees what a deferred one wrote."""
    index = DenseIndex.init_ram(8, DenseMetric.ANGULAR, Scalar.F32)
    index.reserve(4)
    call = index.deferred(executor).insert([7], random_vectors(1, 8))
    assert index.on(executor).count(7) == 1
    assert call.status is Status.SUCCESS


def test_full_deferred_table_answers_contended():
    """A submission finding the deferred table full finishes `CONTENDED` at once, and the earlier call still runs."""
    index, vectors = build_index(16, 8, Executor.for_ram(threads=1))
    pool = Executor.for_ram(threads=1, deferred_limit=1)
    first = index.deferred(pool).count(3)
    second = index.deferred(pool).count(4)
    assert second.status is Status.CONTENDED
    with pytest.raises(ContendedError):
        second.wait()
    pool.drain()
    assert first.wait() == 1


def test_dropping_an_unfinished_deferred_waits(executor):
    """Dropping a `Deferred` waits for its call, since the call's structure and buffers live inside it."""
    index = DenseIndex.init_ram(8, DenseMetric.ANGULAR, Scalar.F32)
    index.reserve(4)
    call = index.deferred(executor).insert([1, 2], random_vectors(2, 8))
    drainer = Timer(0.05, executor.drain)
    drainer.start()
    del call
    drainer.join()
    assert len(index) == 2


def test_deferred_call_holds_its_arena(executor):
    """A deferred call claims its arena until it is final, so a second call naming it is refused meanwhile."""
    index, vectors = build_index(4, 8, executor)
    arena = Arena()
    call = index.deferred(executor).get(1, arena=arena)
    assert call.status == Status.DEFERRED
    with pytest.raises(ContendedError):
        index.get(2, arena=arena)
    executor.drain()
    rows = call.wait()
    assert np.allclose(np.asarray(rows), vectors[1:2], atol=1e-6)
    del rows, call  # The deferred call answers its result again on every wait, so it holds it too
    assert np.asarray(index.get(2, arena=arena)).shape == (1, 8)


def test_deferred_receiver_lacks_what_cannot_defer(executor):
    """Iterators block on every step and factories address no index, so the receivers that cannot run them lack them."""
    index, _ = build_index(4, 8, executor)
    assert not hasattr(index.deferred(executor), "cursor")
    assert not hasattr(index.deferred(executor), "search_stream")
    assert not hasattr(index.on(executor), "init_ram")
    assert hasattr(index.on(executor), "cursor")
    assert type(index.on(executor)) is On and type(index.deferred(executor)) is DeferredOn
    assert repr(index.on(executor)) == f"On({index!r}, {executor!r})"


# endregion Receivers

# region Ingest


@pytest.mark.parametrize("count", batch_sizes)
@pytest.mark.parametrize("dimensions", SETTINGS.dims)
def test_insert_publishes_every_row(executor, count: int, dimensions: int):
    """A batch is all-or-nothing, so every row of an accepted batch is visible afterwards."""
    index, _ = build_index(count, dimensions, executor)
    assert len(index) == count
    assert index.capacity >= count


@pytest.mark.parametrize("quantization", quantizations, ids=attrgetter("name"))
def test_insert_accepts_every_quantization(executor, quantization: Scalar):
    """Every stored scalar type takes the same `float32` rows and quantizes them on ingest."""
    index, _ = build_index(64, 96, executor, quantization=quantization)
    assert len(index) == 64


@pytest.mark.parametrize("input_dtype", input_dtypes)
def test_insert_accepts_any_input_dtype(executor, input_dtype):
    """The query type is free to differ from the stored quantization."""
    index = DenseIndex.init_ram(dimensions=32, quantization=Scalar.BF16)
    index.reserve(16)
    vectors = (np.random.default_rng(0).random((16, 32)) * 10).astype(input_dtype)
    index.on(executor).insert(np.arange(16), vectors)
    assert len(index) == 16


def test_insert_accepts_packed_rows_by_logical_width(executor):
    """A `u1x8` row counts the bits it packs, so an eight-byte row is a sixty-four-bit vector."""
    index = DenseIndex.init_ram(dimensions=64, metric=DenseMetric.HAMMING, quantization=Scalar.U1X8)
    index.reserve(16)
    rows = random_vectors(8, 64, metric=DenseMetric.HAMMING)
    assert rows.shape == (8, 8), "sixty-four bits pack into eight bytes"
    index.on(executor).insert(np.arange(8), rows, dtype=Scalar.U1X8)
    assert len(index) == 8

    narrow = random_vectors(8, 32, metric=DenseMetric.HAMMING)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).insert(np.arange(8, 16), narrow, dtype=Scalar.U1X8)


def test_insert_refuses_a_dtype_narrower_than_the_buffer(executor):
    """A named kind reinterprets the buffer's bytes, so one narrower than its items would misread every row."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.F32)
    index.reserve(4)
    with pytest.raises(UnknownKindError):
        index.insert(np.arange(4), random_vectors(4, 8), dtype=Scalar.F16)
    assert len(index) == 0


def test_insert_accepts_numkong_tensors_without_numpy(executor):
    """A NumKong tensor is a `memoryview`, so it reaches the index through PEP 3118 alone.

    This is the claim the README leads with, and the only test that exercises it: every other
    path in this file arrives as a NumPy array.
    """
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.F32)
    index.reserve(8)
    rows = nk.Tensor(random_vectors(8, 16))
    assert memoryview(rows).shape == (8, 16), "NumKong tensors export PEP 3118 directly"
    index.on(executor).insert(np.arange(8), rows)
    assert len(index) == 8


@pytest.mark.parametrize("keys", [[Key(11), 12], np.arange(2), integers_to_keys(np.arange(2))])
def test_insert_accepts_every_key_spelling(executor, keys):
    """Keys arrive as objects, as integers, or as the packed wire rows."""
    index = DenseIndex.init_ram(dimensions=4, quantization=Scalar.F32)
    index.reserve(2)
    index.on(executor).insert(keys, random_vectors(2, 4))
    assert len(index) == 2


def test_insert_rejects_mismatched_key_count(executor):
    """A key per row, or the batch is refused before anything is published."""
    index = DenseIndex.init_ram(dimensions=4, quantization=Scalar.F32)
    index.reserve(4)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).insert([1, 2, 3], random_vectors(2, 4))
    assert len(index) == 0


def test_insert_reports_telemetry(executor):
    """A batch reports what it walked into the `DenseTraversalStats` a caller passes."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.F32)
    index.reserve(128)
    stats = DenseTraversalStats()
    index.on(executor).insert(np.arange(64), random_vectors(64, 16), stats=stats)
    assert stats.visited_members > 0
    assert stats.computed_distances > 0


def test_insert_reports_progress(executor):
    """Reports never go backwards, the last one says the whole batch finished, and a handler answering None carries on."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.F32)
    index.reserve(64)
    seen: list[tuple[int, int]] = []

    def watch(completed: int, total: int) -> None:
        seen.append((completed, total))

    index.on(executor).insert(np.arange(64), random_vectors(64, 16), progress=watch)
    assert len(index) == 64
    assert seen, "a watched batch reports at least once"
    assert all(earlier[0] <= later[0] for earlier, later in pairwise(seen)), "counts never decrease"
    assert {total for _, total in seen} == {64}, "every report names the same total"
    assert seen[-1] == (64, 64)


def test_insert_handler_reports_and_cannot_stop(executor):
    """Every row is claimed before the first report, so a handler answering False still lands the whole batch."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.F32)
    index.reserve(64)
    index.on(executor).insert(np.arange(64), random_vectors(64, 16), progress=lambda done, total: done < 8)
    assert len(index) == 64


def test_insert_surfaces_a_raising_handler(executor):
    """A handler's own exception outranks the cancellation it turned into."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.F32)
    index.reserve(8)

    failure = ValueError()

    def broken(completed: int, total: int) -> bool:
        raise failure

    with pytest.raises(ValueError) as raised:
        index.on(executor).insert(np.arange(8), random_vectors(8, 16), progress=broken)
    assert raised.value is failure
    assert len(index) == 8, "the batch landed before the handler's exception surfaced"


# endregion Ingest

# region Search


@pytest.mark.parametrize("dimensions", SETTINGS.dims)
def test_search_shapes(executor, dimensions: int):
    """The three outputs are shaped by the batch and the "k" that was asked for."""
    index, vectors = build_index(64, dimensions, executor)
    stats = DenseTraversalStats()
    matches: Matches = index.on(executor).search(vectors[:3], 5, stats=stats)

    assert np.asarray(matches.keys).shape == (3, 5, 16)
    assert np.asarray(matches.distances).shape == (3, 5)
    assert np.asarray(matches.counts).shape == (3,)
    assert len(matches) == 3
    assert stats.computed_distances > 0


def test_search_rows_are_matches(executor):
    """A result row is a tuple of scored entries, nearest first."""
    index, vectors = build_index(64, 32, executor)
    row = index.on(executor).search(vectors[:1], 4)[0]
    assert all(isinstance(match, Match) for match in row)
    assert isinstance(row[0].key, Key)
    assert row[0].distance <= row[-1].distance


def test_search_reports_fewer_matches_than_asked(executor):
    """A count bounds its own row, so an index holding less than `wanted` still answers."""
    index, vectors = build_index(3, 8, executor)
    matches = index.on(executor).search(vectors[:1], 10)
    count = int(np.asarray(matches.counts)[0])
    assert count <= 3
    assert len(matches[0]) == count


def test_search_reuses_the_returned_matches(executor):
    """One sizing serves a whole sweep, so a second call lands in what the first returned."""
    index, vectors = build_index(64, 16, executor)
    matches = index.on(executor).search(vectors[:2], 4)
    first = keys_to_integers(matches.keys).copy()
    assert index.on(executor).search(vectors[4:6], 4, out=matches) is matches
    assert not np.array_equal(first, keys_to_integers(matches.keys))
    assert int(np.asarray(matches.counts)[0]) == 4


def test_search_lands_in_the_caller_buffers(executor):
    """A `Matches` over buffers the caller owns is a zero-allocation target that writes through."""
    index, vectors = build_index(64, 16, executor)
    keys, distances, counts = matches_outputs(2, 4)
    index.on(executor).search(vectors[:2], 4, out=Matches(keys, distances, counts))
    assert counts.tolist() == [4, 4]
    assert keys_to_integers(keys)[0, 0] == 0


def test_search_rejects_an_overflowing_request(executor):
    """An output matrix that cannot fit memory is refused rather than silently wrapped."""
    index, vectors = build_index(4, 8, executor)
    with pytest.raises(CapacityOverflowError) as raised:
        index.on(executor).search(vectors[:2], 2**62)
    assert raised.value.status is Status.OVERFLOW


@pytest.mark.parametrize("dimensions", [97, 256])
@pytest.mark.parametrize("metric", spatial_metrics, ids=attrgetter("name"))
def test_recall_against_the_oracle(executor, dimensions: int, metric: DenseMetric):
    """The index must return most of what a brute-force scan would, in every metric."""
    index, vectors = build_index(512, dimensions, executor, metric=metric, quantization=Scalar.F32)
    recall = measured_recall(index, vectors, vectors[:64], 10, executor, metric=metric)
    assert recall >= recall_floor(Scalar.F32), f"{metric} recall collapsed to {recall}"


@pytest.mark.parametrize("quantization", quantizations, ids=attrgetter("name"))
def test_recall_survives_quantization(executor, quantization: Scalar):
    """Every stored scalar type still finds what it should, at its own floor."""
    index, _ = build_index(512, 96, executor, quantization=quantization)
    recall = self_recall(index.on(executor)).mean_recall
    assert recall >= recall_floor(quantization), f"{quantization} recall collapsed to {recall}"


@pytest.mark.parametrize(
    "quantization", [Scalar.F64, Scalar.F32, Scalar.BF16, Scalar.F16, Scalar.I8], ids=attrgetter("name")
)
def test_distances_agree_with_the_oracle(executor, quantization: Scalar):
    """A reported distance matches what NumKong computes for the same pair, within tolerance."""
    index, vectors = build_index(128, 64, executor, quantization=quantization)
    queries = vectors[:8]
    matches = index.on(executor).search(queries, 1)
    found = keys_to_integers(matches.keys)[:, 0]
    reported = np.asarray(matches.distances)[:, 0]
    truth = np.asarray(nk.cdist(queries, vectors, "angular", out_dtype="float64"))
    expected = truth[np.arange(len(queries)), found]
    atol, rtol = distance_tolerances(quantization)
    np.testing.assert_allclose(reported, expected, atol=atol, rtol=rtol)


def test_expansion_rides_the_call(executor):
    """The candidate pool is a per-call setting, so one immutable index serves a whole sweep."""
    index, vectors = build_index(512, 64, executor)
    queries = vectors[:32]
    narrow, wide = DenseTraversalStats(), DenseTraversalStats()
    index.on(executor).search(queries, 10, expansion_factor=16, stats=narrow)
    index.on(executor).search(queries, 10, expansion_factor=256, stats=wide)
    assert wide.computed_distances > narrow.computed_distances


def test_telemetry_is_a_per_call_reading(executor):
    """A call overwrites the `DenseTraversalStats` it takes, and `+=` is how a sweep sums the readings."""
    index, vectors = build_index(128, 16, executor)
    queries = vectors[:8]
    stats, total = DenseTraversalStats(), DenseTraversalStats()

    index.on(executor).search(queries, 5, stats=stats)
    first = stats.computed_distances
    assert first > 0
    total += stats

    index.on(executor).search(queries, 5, stats=stats)
    assert stats.computed_distances == first, "a repeat replaces the reading rather than adding to it"
    total += stats
    assert total.computed_distances == 2 * first


@pytest.mark.parametrize(("queries", "wanted"), [(4, 3), (3, 10), (5, 10)])
def test_search_rejects_a_mismatched_out(executor, queries: int, wanted: int):
    """A reused run must be exactly the shape the call writes, no more and no less."""
    index, vectors = build_index(32, 16, executor)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).search(vectors[:4], 10, out=Matches(*matches_outputs(queries, wanted)))


def test_search_rejects_read_only_outputs(executor):
    """A read-only output is refused before anything is written into it."""
    index, vectors = build_index(32, 16, executor)
    read_only = Matches(bytes(2 * 2 * 16), np.zeros((2, 2)), np.zeros(2, np.uint64))
    with pytest.raises(ValueError):
        index.on(executor).search(vectors[:2], 2, out=read_only)


def test_filtered_search_admits_only_approved_keys(executor):
    """A predicate narrows the answer without collapsing the walk that produced it."""
    index, vectors = build_index(128, 32, executor)
    matches = index.on(executor).search(vectors[:4], 8, predicate=lambda key: int(key) % 2 == 0)
    found = keys_to_integers(matches.keys)
    counts = np.asarray(matches.counts)
    for query_index in range(len(counts)):
        assert np.all(found[query_index, : counts[query_index]] % 2 == 0)


def test_filtered_search_propagates_a_raising_predicate(executor):
    """A predicate that raises fails the search rather than silently narrowing it."""
    index, vectors = build_index(64, 16, executor)

    failure = ValueError()

    def broken(key):
        raise failure

    with pytest.raises(ValueError) as raised:
        index.on(executor).search(vectors[:2], 4, predicate=broken)
    assert raised.value is failure


# endregion Search

# region Retrieval


def test_count_and_contains(executor):
    """Membership and multiplicity are the same question asked two ways."""
    index, _ = build_index(32, 16, executor)
    assert index.count(0) == 1
    assert index.count(10_000) == 0
    assert 0 in index
    assert 10_000 not in index


def test_get_returns_the_stored_rows(executor):
    """An `f32` index hands back `f32` rows unchanged."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.F32)
    index.reserve(2)
    vectors = random_vectors(2, 8)
    index.on(executor).insert([1, 2], vectors)

    rows = index.get(1, outputs_dtype=Scalar.F32)
    assert isinstance(rows, Vectors) and not hasattr(rows, "__cuda_array_interface__")
    retrieved = np.asarray(rows)
    assert retrieved.shape == (1, 8)
    assert np.allclose(retrieved[0], vectors[0], atol=1e-6)


def test_get_answers_none_and_subscript_raises(executor):
    """`get` is a `get`; subscription is a lookup, so the two answer absence differently."""
    index, _ = build_index(4, 8, executor)
    assert index.get(10_000) is None
    with pytest.raises(KeyError):
        index[10_000]


def test_get_leaves_unformattable_kinds_as_bytes(executor):
    """Sub-byte and mini-float kinds have no `memoryview` format, so they leave as raw runs."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.BF16)
    index.reserve(1)
    index.on(executor).insert([1], random_vectors(1, 8))
    view = memoryview(index.get(1, outputs_dtype=Scalar.BF16))
    assert view.format == "B"
    assert view.shape == (1, 16)


def test_arena_refuses_reuse_while_a_view_lives(executor):
    """Reuse rewinds the arena, so it waits until every result over the last outputs is dropped."""
    index, _ = build_index(4, 8, executor, quantization=Scalar.F32)
    arena = Arena()
    first = index.on(executor).get(1, arena=arena)
    with pytest.raises(ContendedError):
        index.on(executor).get(2, arena=arena)
    del first
    assert np.asarray(index.on(executor).get(2, arena=arena)).shape == (1, 8)


def test_duplicate_keys_stack(executor):
    """Keys need not be unique; several entries may share one and `count` reports how many."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.F32)
    index.reserve(3)
    index.on(executor).insert([5, 5, 5], random_vectors(3, 8))
    assert index.count(5) == 3
    assert np.asarray(index[5]).shape == (3, 8)


# endregion Retrieval

# region Iteration


def test_iteration_walks_every_key(executor):
    """A snapshot walk visits each published entry exactly once."""
    index, _ = build_index(300, 16, executor)
    walked = np.concatenate([np.asarray(batch) for batch in index.cursor(64)], axis=0)
    assert sorted(keys_to_integers(walked).tolist()) == list(range(300))


def test_iteration_carries_typed_vectors(executor):
    """A walk asked for a scalar type yields that type, not the bytes underneath it."""
    index, vectors = build_index(40, 12, executor, quantization=Scalar.F32)
    keys, rows = next(iter(index.cursor(16, outputs_dtype=Scalar.F32, columns=12)))
    assert rows.format == "f"
    assert rows.shape == (16, 12)
    assert np.asarray(keys).shape == (16, 16)
    # The binding never asks the index for its width, so the caller names both or neither.
    with pytest.raises(TypeError):
        index.cursor(16, outputs_dtype=Scalar.F32)
    with pytest.raises(TypeError):
        index.cursor(16, columns=12)


def test_search_stream_yields_nearest_first(executor):
    """A search stream is resumable, ordered by what the walk has reached."""
    index, vectors = build_index(128, 24, executor)
    stream = index.on(executor).search_stream(vectors[0])
    streamed = [match.distance for _, match in zip(range(16), stream, strict=False)]
    assert streamed == sorted(streamed)


def test_walks_close_before_they_are_collected(executor):
    """`close()` and a `with` block release a cursor's snapshot or a stream's walk, after which iteration ends."""
    index, vectors = build_index(32, 8, executor)
    with index.cursor(4) as cursor:
        assert len(np.asarray(next(cursor))) == 4
    assert next(cursor, None) is None
    with index.search_stream(vectors[0]) as stream:
        key, distance = next(stream)
        assert int(key) == 0
    assert next(stream, None) is None
    stream = index.search_stream(vectors[1])
    stream.close()
    assert list(stream) == []


# endregion Iteration

# region Composition


def test_merge_absorbs_and_consumes_the_donor(executor):
    """The donor's entries arrive and its handle stops answering, both at once."""
    first = DenseIndex.init_ram(dimensions=16, quantization=Scalar.F32)
    first.reserve(64)
    first.on(executor).insert(np.arange(0, 32), random_vectors(32, 16))

    second = DenseIndex.init_ram(dimensions=16, quantization=Scalar.F32)
    second.reserve(32)
    second.on(executor).insert(np.arange(32, 64), random_vectors(32, 16, seed=1))

    first.on(executor).merge(second)
    assert len(first) == 64
    assert 40 in first

    with pytest.raises(UninitializedError):
        len(second)


def test_merge_consumes_a_donor_still_walked(executor):
    """A cursor keeps no hold on its index, so a merge consumes the donor and the cursor's next
    pull, naming a handle that is gone, is refused."""
    first, _ = build_index(16, 8, executor)
    second, _ = build_index(16, 8, executor, seed=1)
    cursor = second.cursor(4)
    first.on(executor).merge(second)
    assert len(first) == 32
    with pytest.raises(UninitializedError):
        next(cursor)


def test_merge_refuses_itself(executor):
    """An index cannot absorb the handle it already is."""
    index, _ = build_index(8, 8, executor)
    with pytest.raises(InvalidArgumentError):
        index.on(executor).merge(index)


def test_clear_keeps_the_geometry(executor):
    """Clearing drops the entries and nothing else."""
    index, _ = build_index(32, 16, executor)
    capacity = index.capacity
    index.clear()
    assert len(index) == 0
    assert index.capacity == capacity
    assert index.dimensions == 16


# endregion Composition

# region Serialization


def test_save_load_roundtrip(executor, tmp_path):
    """A checkpoint restores what it stored, its geometry read from its own headers."""
    index, _ = build_index(128, 32, executor, quantization=Scalar.F32)
    path = str(tmp_path / "index.safegraphs")
    index.on(executor).save(path)

    metadata = DenseIndex.metadata_read(path, executor=executor)
    assert isinstance(metadata, DenseMetadata)
    assert metadata.dimensions == 32
    assert metadata.metric is DenseMetric.ANGULAR
    assert DenseMetadata.n_fields == 4, "a checkpoint fixes the geometry, never where it is hosted"

    reloaded = DenseIndex.load_ram(path, executor=executor)
    assert len(reloaded) == len(index)
    assert reloaded.dimensions == 32
    assert 7 in reloaded


def test_save_load_through_buffers(executor):
    """`save_buffer` returns the monolith a `load_ram_buffer` reads back."""
    index, _ = build_index(64, 16, executor, quantization=Scalar.F32)
    buffer = index.on(executor).save_buffer()
    assert isinstance(buffer, Monolith) and memoryview(buffer).nbytes > 0
    assert DenseIndex.metadata_read_buffer(buffer).dimensions == 16

    reloaded = DenseIndex.load_ram_buffer(buffer)
    assert len(reloaded) == len(index)


def test_cancelled_load_creates_no_index(executor, tmp_path):
    """A handler that stops a load leaves nothing behind but the exception."""
    index, _ = build_index(8, 8, executor, quantization=Scalar.F32)
    path = str(tmp_path / "cancelled.safegraphs")
    index.on(executor).save(path)
    with pytest.raises(CancelledError):
        DenseIndex.load_ram(path, progress=lambda completed, total: False)


def test_view_serves_in_place_and_refuses_mutation(executor, tmp_path):
    """A viewed checkpoint answers searches and refuses every mutating verb."""
    index, vectors = build_index(32, 16, executor, quantization=Scalar.F32)
    path = str(tmp_path / "viewed.safegraphs")
    index.on(executor).save(path)

    viewed = DenseIndex.view_ram(path)
    assert len(viewed) == 32
    assert len(viewed.on(executor).search(vectors[:2], 3)) == 2
    with pytest.raises(ReadOnlyError) as raised:
        viewed.on(executor).insert([99], vectors[:1])
    assert raised.value.status is Status.READ_ONLY


def test_encoded_path_is_read_as_a_path(executor, tmp_path):
    """`os.fsencode` returns `bytes`, which still names a file."""
    index, _ = build_index(8, 8, executor, quantization=Scalar.F32)
    path = str(tmp_path / "encoded.safegraphs")
    index.on(executor).save(path)
    assert DenseIndex.metadata_read(os.fsencode(path)).dimensions == 8


def test_metadata_roundtrips_through_a_factory(executor, tmp_path):
    """The metadata fields are the geometry every factory takes, so they rebuild the handle."""
    index = DenseIndex.init_ram(dimensions=48, metric=DenseMetric.EUCLIDEAN, quantization=Scalar.I8, connectivity=24)
    index.reserve(8)
    index.on(executor).insert(np.arange(8), random_vectors(8, 48))
    path = str(tmp_path / "typed.safegraphs")
    index.on(executor).save(path)

    metadata = DenseIndex.metadata_read(path)
    rebuilt = DenseIndex.init_ram(
        metadata.dimensions, metadata.metric, metadata.quantization, connectivity=metadata.connectivity
    )
    assert rebuilt.dimensions == 48
    assert rebuilt.connectivity == 24


# endregion Serialization

# region Refinement


def test_refine_reports_a_sweep_and_leaves_the_index_standing(executor):
    """A sweep re-selects edges without publishing or retiring an entry, so the rows survive it."""
    index, vectors = build_index(64, 16, executor)
    changed = index.on(executor).refine(expansion_factor=64)
    assert isinstance(changed, int)
    assert changed <= len(index), "a sweep cannot change more nodes than the graph holds"
    assert len(index) == 64
    assert len(index.on(executor).search(vectors[:4], 5)) == 4


def test_refine_reports_telemetry(executor):
    """The sweep reports into the `DenseTraversalStats` a caller passes, as every other walk does."""
    index, _ = build_index(32, 16, executor)
    stats = DenseTraversalStats()
    index.on(executor).refine(stats=stats)
    assert stats.visited_members > 0


def test_refine_takes_no_positionals(executor):
    """Every setting a sweep has is a keyword, so a bare number cannot be mistaken for one."""
    index, _ = build_index(8, 8, executor)
    with pytest.raises(TypeError):
        index.refine(64)


def test_refine_rejects_unknown_keywords(executor):
    """A misspelled setting is refused rather than silently derived."""
    index, _ = build_index(8, 8, executor)
    with pytest.raises(TypeError):
        index.on(executor).refine(expansion_add=64)


def test_refine_refuses_a_consumed_handle(executor):
    """A donor absorbed by `merge` has no graph left to sweep."""
    first, _ = build_index(8, 8, executor)
    second, _ = build_index(8, 8, executor, seed=1)
    first.on(executor).merge(second)
    with pytest.raises(UninitializedError):
        second.on(executor).refine()


def test_health_reaches_every_entry_after_a_refine(executor):
    """A refine re-links whatever it strands, so the audit after it reaches the whole index."""
    index, _ = build_index(64, 16, executor)
    index.on(executor).refine()
    health = index.on(executor).health()
    assert isinstance(health, DenseReachability)
    assert health.audited == len(index)
    assert health.reachable == health.audited
    assert health.sweeps >= 1


# endregion Refinement

# region Late Interaction

_LATE_INTERACTION_DOCUMENTS = 6
_LATE_INTERACTION_TOKENS = 3


def late_interaction_index(executor, dimensions: int = 16) -> tuple[DenseIndex, np.ndarray]:
    """An inner-product index whose entries are token rows, several sharing one document key."""
    rows = _LATE_INTERACTION_DOCUMENTS * _LATE_INTERACTION_TOKENS
    index = DenseIndex.init_ram(
        dimensions=dimensions,
        metric=DenseMetric.INNER,
        quantization=Scalar.F32,
        values_per_key=_LATE_INTERACTION_TOKENS,
    )
    index.reserve(rows)
    documents = np.repeat(np.arange(_LATE_INTERACTION_DOCUMENTS, dtype=np.uint64), _LATE_INTERACTION_TOKENS)
    tokens = random_vectors(rows, dimensions, seed=5)
    index.on(executor).insert(integers_to_keys(documents), tokens)
    return index, tokens


def scored_documents(index, tokens, queries: int, wanted: int, executor, **kwargs) -> Matches:
    """Runs `maxsim` over *queries* whole queries."""
    return index.on(executor).maxsim(
        tokens[: queries * _LATE_INTERACTION_TOKENS], _LATE_INTERACTION_TOKENS, wanted, **kwargs
    )


def test_maxsim_sizes_its_outputs_over_documents(executor):
    """A query is a run of token rows, so the result has one row per query rather than per token."""
    index, tokens = late_interaction_index(executor)
    queries = 2
    stats = DenseTraversalStats()
    matches = scored_documents(index, tokens, queries, 4, executor, stats=stats)

    assert len(matches) == queries
    assert np.asarray(matches.keys).shape == (queries, 4, 16)
    assert np.asarray(matches.distances).shape == (queries, 4)
    assert np.asarray(matches.counts).shape == (queries,)
    assert stats.computed_distances > 0


def test_maxsim_scores_each_document_once(executor):
    """The reduction groups by key, so a document holding many token rows still lands one entry."""
    index, tokens = late_interaction_index(executor)
    matches = scored_documents(index, tokens, 1, 4, executor)
    count = int(np.asarray(matches.counts)[0])
    found = keys_to_integers(matches.keys)[0, :count]
    assert count > 0
    assert len(set(found.tolist())) == count


def test_maxsim_orders_by_summed_maxima(executor):
    """Summed maxima are a similarity, so a row descends where a search's distances ascend."""
    index, tokens = late_interaction_index(executor)
    matches = scored_documents(index, tokens, 1, 4, executor)
    count = int(np.asarray(matches.counts)[0])
    scored = np.asarray(matches.distances)[0, :count]
    assert list(scored) == sorted(scored, reverse=True)


def test_maxsim_refuses_ragged_queries(executor):
    """The query rows must divide into whole queries, or the grouping is a guess."""
    index, tokens = late_interaction_index(executor)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).maxsim(tokens[: _LATE_INTERACTION_TOKENS + 1], _LATE_INTERACTION_TOKENS, 4)


def test_maxsim_refuses_a_predicate(executor):
    """`maxsim` has no filtered sibling, so the keyword is refused rather than ignored."""
    index, tokens = late_interaction_index(executor)
    with pytest.raises(TypeError):
        index.on(executor).maxsim(
            tokens[:_LATE_INTERACTION_TOKENS],
            _LATE_INTERACTION_TOKENS,
            4,
            predicate=lambda key: True,
        )


def test_maxsim_refuses_a_distance_kind(executor):
    """Late interaction sums maxima over logits, which only the inner-product kinds score."""
    index, vectors = build_index(16, 8, executor, metric=DenseMetric.ANGULAR)
    with pytest.raises(ConfigMismatchError):
        index.on(executor).maxsim(vectors[:2], 2, 4)


def test_maxsim_reports_telemetry(executor):
    """The walk reports into the `DenseTraversalStats` a caller passes, as every other walk does."""
    index, tokens = late_interaction_index(executor)
    stats = DenseTraversalStats()
    scored_documents(index, tokens, 1, 4, executor, stats=stats)
    assert stats.computed_distances > 0


# endregion Late Interaction

# region Linear Algebra


def test_multiply_reports_every_scored_pair(executor):
    """A walk scores many more pairs than a search reports, and this verb hands them all back."""
    index, vectors = build_index(64, 32, executor, metric=DenseMetric.INNER)
    rows = vectors[:4]
    products = index.on(executor).multiply(rows, 0)
    offsets = np.asarray(products.offsets)
    named = keys_to_integers(products.keys)
    scores = np.asarray(products.scores)

    assert isinstance(products, Products) and len(products) == len(rows)
    assert offsets.shape == (len(rows) + 1,)
    assert offsets[0] == 0 and offsets[-1] == len(scores) == products.required
    assert np.all(np.diff(offsets) > 0)
    row_keys, row_scores = products.row(1)
    assert np.array_equal(np.asarray(row_scores), scores[offsets[1] : offsets[2]])
    assert np.array_equal(keys_to_integers(row_keys), named[offsets[1] : offsets[2]])
    assert not hasattr(products.scores, "__cuda_array_interface__")

    # Every product is an exact dot against the row it names, which is the point of the verb.
    for row_index in range(len(rows)):
        for pair in range(int(offsets[row_index]), int(offsets[row_index + 1])):
            expected = float(rows[row_index] @ vectors[named[pair]])
            assert scores[pair] == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_multiply_reports_what_wanted_cut(executor):
    """A bound keeps the first pairs per row, and `required` still counts every pair scored."""
    index, vectors = build_index(64, 16, executor, metric=DenseMetric.INNER)
    products = index.on(executor).multiply(vectors[:2], 4)
    assert np.diff(np.asarray(products.offsets)).tolist() == [4, 4]
    assert products.required > 8, "both rows scored more pairs than were kept"


def test_multiply_refuses_a_predicate(executor):
    """`multiply` has no filtered sibling, so the keyword is refused rather than ignored."""
    index, vectors = build_index(16, 8, executor, metric=DenseMetric.INNER)
    with pytest.raises(TypeError):
        index.on(executor).multiply(vectors[:1], 4, predicate=lambda key: True)


def value_lookup(values: np.ndarray):
    """A value lookup resolving integer keys to their rows of *values*."""
    return lambda keys: values[keys_to_integers(keys)]


@pytest.mark.parametrize("head_count", [1, 2])
def test_attend_over_every_key_is_full_attention(executor, head_count: int):
    """A graph this small walks every key, so each head's output and carry are exact attention's."""
    dimensions, context, tokens = 16, 32, 3
    index, keys_matrix = build_index(context, dimensions, executor, metric=DenseMetric.INNER)
    values = random_vectors(context, dimensions, seed=7)
    queries = random_vectors(tokens * head_count, dimensions, seed=3)
    carry = np.zeros((len(queries), 2), dtype=np.float32)
    outputs = index.on(executor).attend(
        queries,
        value_lookup(values),
        head_count=head_count,
        expansion_factor=2 * context,
        outputs_dtype=Scalar.F32,
        carry=carry,
    )

    logits = queries @ keys_matrix.T / np.sqrt(dimensions)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    np.testing.assert_allclose(
        np.asarray(outputs), weights @ values / weights.sum(axis=1, keepdims=True), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(carry[:, 0], logits.max(axis=1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(carry[:, 1], weights.sum(axis=1), rtol=1e-5)


def test_attend_lands_in_the_caller_buffer(executor):
    """`out` is written through and handed back, so a decode loop allocates nothing."""
    index, keys_matrix = build_index(16, 8, executor, metric=DenseMetric.INNER)
    outputs = np.zeros((2, 8), dtype=np.float32)
    landed = index.on(executor).attend(
        keys_matrix[:2], value_lookup(keys_matrix), outputs_dtype=Scalar.F32, out=outputs
    )
    assert landed is outputs
    assert np.any(outputs != 0)


def test_attend_refuses_a_missing_value_row(executor):
    """A missing row is a wrong softmax rather than a smaller one, and a lookup's own failure outranks it."""
    index, keys_matrix = build_index(8, 8, executor, metric=DenseMetric.INNER)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).attend(keys_matrix[:1], lambda keys: None)
    with pytest.raises(ZeroDivisionError):
        index.on(executor).attend(keys_matrix[:1], lambda keys: 1 / 0)


def test_attend_refuses_a_half_width_carry(executor):
    """A maximum alone is a number nothing can rescale, so the pair is sized exactly."""
    index, keys_matrix = build_index(8, 8, executor, metric=DenseMetric.INNER)
    with pytest.raises(ValueError):
        index.on(executor).attend(keys_matrix[:1], value_lookup(keys_matrix), carry=np.zeros(1, np.float32))


def test_attend_refuses_a_distance_kind(executor):
    """A softmax weighs logits, which only the inner-product kinds score."""
    index, keys_matrix = build_index(8, 8, executor, metric=DenseMetric.ANGULAR)
    with pytest.raises(ConfigMismatchError):
        index.on(executor).attend(keys_matrix[:1], value_lookup(keys_matrix))


# endregion Linear Algebra
