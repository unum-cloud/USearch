"""Vector search over equidimensional dense embeddings - the Python twin of `test/dense.cpp`.

Every recall claim is checked against `usearch.eval.exact_matches`, which scores with NumKong's
kernels in the index's own metric vocabulary, so a number here compares like with like.

Run:
    python -m pytest test/dense.py -v
"""

from __future__ import annotations

import os
from itertools import pairwise
from threading import Thread

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
    DenseIndex,
    Executor,
    InvalidArgumentError,
    Key,
    Match,
    Matches,
    ReadOnlyError,
    ShapeMismatchError,
    Stats,
    UninitializedError,
)
from usearch.eval import random_vectors, self_recall
from usearch.keys import integers_to_keys, keys_to_integers

# region Construction


@pytest.mark.parametrize("dimensions", SETTINGS.dims)
@pytest.mark.parametrize("metric", spatial_metrics)
def test_construction_reports_its_geometry(dimensions: int, metric: str):
    """Every setting the factory took reads back off the handle."""
    index = DenseIndex.for_ram(dimensions=dimensions, metric=metric, quantization="bf16", connectivity=16)
    assert index.dimensions == dimensions
    assert index.connectivity == 16
    assert index.size == 0
    assert len(index) == 0
    assert isinstance(index.capabilities, str)
    assert f"dimensions={dimensions}" in repr(index)
    description = index.describe()
    assert description["dimensions"] == dimensions
    assert description["metric"] == metric
    assert description["quantization"] == "bf16"
    assert description["size"] == 0


@pytest.mark.parametrize(
    ("options", "failure"),
    [
        ({"metric": "manhattan"}, ValueError),
        ({"metric": "l2sq"}, ValueError),  # An alias we deliberately no longer accept.
        ({"quantization": "f128"}, ValueError),
        ({"residency": "vram"}, TypeError),  # The factory chooses the residency; no keyword does.
        ({"devices": [0]}, TypeError),  # An arm belonging to another residency.
        ({"addressing": "device-only"}, TypeError),  # The factory names the union member.
        ({"threads": 2}, TypeError),  # An index owns rows; an `Executor` owns workers.
        ({"expansion_add": 64}, TypeError),  # A v2 keyword the surface no longer has.
        ({"ndim": 8}, TypeError),  # Renamed to 'dimensions'.
    ],
)
def test_construction_rejects_unknown_options(options: dict, failure: type):
    """A name the factory does not have is refused, not quietly ignored."""
    with pytest.raises(failure):
        DenseIndex.for_ram(dimensions=8, **options)


def test_construction_takes_three_positionals():
    """The three settings every caller names may be positional; the rest must be keywords."""
    index = DenseIndex.for_ram(32, "euclidean", "f32")
    assert index.dimensions == 32
    with pytest.raises(TypeError):
        DenseIndex.for_ram(32, "euclidean", "f32", 16)


@pytest.mark.parametrize("handle_type", [DenseIndex, Executor])
def test_direct_construction_names_the_factories(handle_type: type):
    """Neither handle is built by calling its type; the refusal says which factory to call."""
    with pytest.raises(TypeError, match=r"for_ram\(\)"):
        handle_type()


# endregion Construction

# region Executors


def test_vram_executor_takes_no_thread_count():
    """One host thread drives one stream per device, so the device list is the pool's only width."""
    with pytest.raises(TypeError, match="threads"):
        Executor.for_vram(threads=4)


def test_ram_executor_takes_memory_domain_ordinals():
    """The NUMA nodes its workers stay local to arrive as ordinals, and nothing else is accepted."""
    Executor.for_ram(memory_domains=[0]).drain()
    with pytest.raises(TypeError):
        Executor.for_ram(memory_domains=["x"])
    with pytest.raises(ValueError, match="32-bit"):
        Executor.for_ram(memory_domains=[2**40])


def test_drain_takes_a_named_mood():
    """The idle mood is one of two names, and nothing else is guessed at."""
    pool = Executor.for_ram(threads=1, in_flight_limit=4)
    pool.drain(mood="grind")
    pool.drain()
    with pytest.raises(ValueError, match="mood"):
        pool.drain("sleep")


def test_borrowed_pool_must_be_named():
    """A pool arrives by address, and a null one is refused rather than dereferenced."""
    with pytest.raises(InvalidArgumentError):
        Executor.on_pool(0)


# One entry per fanned-out verb, called without its `executor`, which runs it on the caller's thread.
_FANNED_OUT_VERBS = [
    ("add", lambda index, rows: index.add([9], rows[:1])),
    ("refine", lambda index, rows: index.refine()),
    ("search", lambda index, rows: index.search(rows[:1], 1)),
    ("search_stream", lambda index, rows: next(index.search_stream(rows[0]))),
    ("save_buffer", lambda index, rows: index.save_buffer()),
    ("get", lambda index, rows: index.get(0)),
    ("count", lambda index, rows: index.count(0)),
]


@pytest.mark.parametrize(("verb", "call"), _FANNED_OUT_VERBS, ids=[name for name, _ in _FANNED_OUT_VERBS])
def test_fanned_out_verbs_run_without_an_executor(executor, verb: str, call):
    """A verb handed no executor runs on the caller's thread rather than refusing."""
    index, vectors = build_index(8, 8, executor)
    call(index, vectors)


@pytest.mark.parametrize("wrong", [2, "ram"])
def test_executor_keyword_takes_nothing_else(executor, wrong):
    """A pool is a handle, not a thread count, so anything else is refused before the boundary."""
    index, vectors = build_index(8, 8, executor)
    with pytest.raises(TypeError, match="must be an Executor or None"):
        index.search(vectors[:1], 1, executor=wrong)


def test_executor_is_borrowed_for_the_call(executor):
    """A verb borrows the workers for its call and never retains them, so a temporary one serves."""
    index, vectors = build_index(16, 8, Executor.for_ram())
    assert len(index) == 16
    assert len(index.search(vectors[:1], 1, executor=executor)) == 1


def test_one_executor_serves_several_indexes(executor):
    """An index owns rows and an executor owns workers, so one pool drives as many as asked."""
    first, vectors = build_index(16, 8, executor)
    second, _ = build_index(16, 8, executor, seed=1)
    assert len(first) == len(second) == 16
    assert len(first.search(vectors[:2], 1, executor=executor)) == 2


def test_concurrent_searches_take_one_executor_apiece(executor, spawn_executor):
    """A handle admits any number of callers, an executor exactly one, so a thread spawns one."""
    index, vectors = build_index(128, 16, executor)
    pools = [spawn_executor() for _ in range(4)]
    answers: dict[int, Matches] = {}

    def resolve(thread_index: int):
        answers[thread_index] = index.search(vectors[:8], 4, executor=pools[thread_index])

    threads = [Thread(target=resolve, args=(thread_index,)) for thread_index in range(len(pools))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(answers) == len(pools), "every thread must have resolved its own batch"
    for matches in answers.values():
        assert np.asarray(matches.counts).shape == (8,)


# endregion Executors

# region Ingest


@pytest.mark.parametrize("count", batch_sizes)
@pytest.mark.parametrize("dimensions", SETTINGS.dims)
def test_add_publishes_every_row(executor, count: int, dimensions: int):
    """A batch is all-or-nothing, so every row of an accepted batch is visible afterwards."""
    index, _ = build_index(count, dimensions, executor)
    assert len(index) == count
    assert index.capacity >= count


@pytest.mark.parametrize("quantization", quantizations)
def test_add_accepts_every_quantization(executor, quantization: str):
    """Every stored scalar type takes the same `float32` rows and quantizes them on ingest."""
    index, _ = build_index(64, 96, executor, quantization=quantization)
    assert len(index) == 64


@pytest.mark.parametrize("input_dtype", input_dtypes)
def test_add_accepts_any_input_dtype(executor, input_dtype):
    """The query type is free to differ from the stored quantization."""
    index = DenseIndex.for_ram(dimensions=32, quantization="bf16")
    index.reserve(16)
    vectors = (np.random.default_rng(0).random((16, 32)) * 10).astype(input_dtype)
    index.add(np.arange(16), vectors, executor=executor)
    assert len(index) == 16


def test_add_accepts_packed_rows_by_logical_width(executor):
    """A `u1x8` row counts the bits it packs, so an eight-byte row is a sixty-four-bit vector."""
    index = DenseIndex.for_ram(dimensions=64, metric="hamming", quantization="u1x8")
    index.reserve(16)
    rows = random_vectors(8, 64, metric="hamming")
    assert rows.shape == (8, 8), "sixty-four bits pack into eight bytes"
    index.add(np.arange(8), rows, executor=executor, dtype="u1x8")
    assert len(index) == 8

    narrow = random_vectors(8, 32, metric="hamming")
    with pytest.raises(ShapeMismatchError):
        index.add(np.arange(8, 16), narrow, executor=executor, dtype="u1x8")


def test_add_accepts_numkong_tensors_without_numpy(executor):
    """A NumKong tensor is a `memoryview`, so it reaches the index through PEP 3118 alone.

    This is the claim the README leads with, and the only test that exercises it: every other
    path in this file arrives as a NumPy array.
    """
    index = DenseIndex.for_ram(dimensions=16, quantization="f32")
    index.reserve(8)
    rows = nk.Tensor(random_vectors(8, 16))
    assert memoryview(rows).shape == (8, 16), "NumKong tensors export PEP 3118 directly"
    index.add(np.arange(8), rows, executor=executor)
    assert len(index) == 8


@pytest.mark.parametrize("keys", [[Key(11), 12], np.arange(2), integers_to_keys(np.arange(2))])
def test_add_accepts_every_key_spelling(executor, keys):
    """Keys arrive as objects, as integers, or as the packed wire rows."""
    index = DenseIndex.for_ram(dimensions=4, quantization="f32")
    index.reserve(2)
    index.add(keys, random_vectors(2, 4), executor=executor)
    assert len(index) == 2


def test_add_rejects_mismatched_key_count(executor):
    """A key per row, or the batch is refused before anything is published."""
    index = DenseIndex.for_ram(dimensions=4, quantization="f32")
    index.reserve(4)
    with pytest.raises(ShapeMismatchError):
        index.add([1, 2, 3], random_vectors(2, 4), executor=executor)
    assert len(index) == 0


def test_add_reports_telemetry(executor):
    """A batch reports what it walked into the `Stats` a caller passes."""
    index = DenseIndex.for_ram(dimensions=16, quantization="f32")
    index.reserve(128)
    stats = Stats()
    index.add(np.arange(64), random_vectors(64, 16), executor=executor, stats=stats)
    assert stats.visited_members > 0
    assert stats.computed_distances > 0


def test_add_reports_progress(executor):
    """Reports never go backwards, and the last one says the whole batch finished."""
    index = DenseIndex.for_ram(dimensions=16, quantization="f32")
    index.reserve(64)
    seen: list[tuple[int, int]] = []

    def watch(completed: int, total: int) -> bool:
        seen.append((completed, total))
        return True

    index.add(np.arange(64), random_vectors(64, 16), executor=executor, progress=watch)
    assert len(index) == 64
    assert seen, "a watched batch reports at least once"
    assert all(earlier[0] <= later[0] for earlier, later in pairwise(seen)), "counts never decrease"
    assert {total for _, total in seen} == {64}, "every report names the same total"
    assert seen[-1] == (64, 64)


def test_add_cancelled_by_its_handler_publishes_nothing(executor):
    """A batch is all-or-nothing, so a handler stopping it midway leaves no row behind."""
    index = DenseIndex.for_ram(dimensions=16, quantization="f32")
    index.reserve(64)
    with pytest.raises(CancelledError) as raised:
        index.add(np.arange(64), random_vectors(64, 16), executor=executor, progress=lambda done, total: done < 8)
    assert raised.value.status == "cancelled"
    assert len(index) == 0


def test_add_surfaces_a_raising_handler(executor):
    """A handler's own exception outranks the cancellation it turned into."""
    index = DenseIndex.for_ram(dimensions=16, quantization="f32")
    index.reserve(8)

    def broken(completed: int, total: int) -> bool:
        raise ValueError("handler is broken")

    with pytest.raises(ValueError, match="handler is broken"):
        index.add(np.arange(8), random_vectors(8, 16), executor=executor, progress=broken)
    assert len(index) == 0


# endregion Ingest

# region Search


@pytest.mark.parametrize("dimensions", SETTINGS.dims)
def test_search_shapes(executor, dimensions: int):
    """The three outputs are shaped by the batch and the "k" that was asked for."""
    index, vectors = build_index(64, dimensions, executor)
    stats = Stats()
    matches: Matches = index.search(vectors[:3], 5, executor=executor, stats=stats)

    assert np.asarray(matches.keys).shape == (3, 5, 16)
    assert np.asarray(matches.distances).shape == (3, 5)
    assert np.asarray(matches.counts).shape == (3,)
    assert len(matches) == 3
    assert stats.computed_distances > 0


def test_search_rows_are_matches(executor):
    """A result row is a tuple of scored entries, nearest first."""
    index, vectors = build_index(64, 32, executor)
    row = index.search(vectors[:1], 4, executor=executor)[0]
    assert all(isinstance(match, Match) for match in row)
    assert isinstance(row[0].key, Key)
    assert row[0].distance <= row[-1].distance


def test_search_reports_fewer_matches_than_asked(executor):
    """A count bounds its own row, so an index holding less than `wanted` still answers."""
    index, vectors = build_index(3, 8, executor)
    matches = index.search(vectors[:1], 10, executor=executor)
    count = int(np.asarray(matches.counts)[0])
    assert count <= 3
    assert len(matches[0]) == count


def test_search_reuses_the_returned_matches(executor):
    """One sizing serves a whole sweep, so a second call lands in what the first returned."""
    index, vectors = build_index(64, 16, executor)
    matches = index.search(vectors[:2], 4, executor=executor)
    first = keys_to_integers(matches.keys).copy()
    assert index.search(vectors[4:6], 4, executor=executor, out=matches) is matches
    assert not np.array_equal(first, keys_to_integers(matches.keys))
    assert int(np.asarray(matches.counts)[0]) == 4


def test_search_lands_in_the_caller_buffers(executor):
    """A `Matches` over buffers the caller owns is a zero-allocation target that writes through."""
    index, vectors = build_index(64, 16, executor)
    keys, distances, counts = matches_outputs(2, 4)
    index.search(vectors[:2], 4, executor=executor, out=Matches(keys, distances, counts))
    assert counts.tolist() == [4, 4]
    assert keys_to_integers(keys)[0, 0] == 0


def test_search_rejects_an_overflowing_request(executor):
    """An output matrix that cannot fit memory is refused rather than silently wrapped."""
    index, vectors = build_index(4, 8, executor)
    with pytest.raises(CapacityOverflowError) as raised:
        index.search(vectors[:2], 2**62, executor=executor)
    assert raised.value.status == "overflow"


@pytest.mark.parametrize("dimensions", [97, 256])
@pytest.mark.parametrize("metric", spatial_metrics)
def test_recall_against_the_oracle(executor, dimensions: int, metric: str):
    """The index must return most of what a brute-force scan would, in every metric."""
    index, vectors = build_index(512, dimensions, executor, metric=metric, quantization="f32")
    recall = measured_recall(index, vectors, vectors[:64], 10, executor, metric=metric)
    assert recall >= recall_floor("f32"), f"{metric} recall collapsed to {recall}"


@pytest.mark.parametrize("quantization", quantizations)
def test_recall_survives_quantization(executor, quantization: str):
    """Every stored scalar type still finds what it should, at its own floor."""
    index, _ = build_index(512, 96, executor, quantization=quantization)
    recall = self_recall(index, executor=executor).mean_recall
    assert recall >= recall_floor(quantization), f"{quantization} recall collapsed to {recall}"


@pytest.mark.parametrize("quantization", ["f64", "f32", "bf16", "f16", "i8"])
def test_distances_agree_with_the_oracle(executor, quantization: str):
    """A reported distance matches what NumKong computes for the same pair, within tolerance."""
    index, vectors = build_index(128, 64, executor, quantization=quantization)
    queries = vectors[:8]
    matches = index.search(queries, 1, executor=executor)
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
    narrow, wide = Stats(), Stats()
    index.search(queries, 10, executor=executor, expansion_factor=16, stats=narrow)
    index.search(queries, 10, executor=executor, expansion_factor=256, stats=wide)
    assert wide.computed_distances > narrow.computed_distances


def test_telemetry_is_a_per_call_reading(executor):
    """A call overwrites the `Stats` it takes, and `accumulate` is how a sweep sums the readings."""
    index, vectors = build_index(128, 16, executor)
    queries = vectors[:8]
    stats, total = Stats(), Stats()

    index.search(queries, 5, executor=executor, stats=stats)
    first = stats.computed_distances
    assert first > 0
    total.accumulate(stats)

    index.search(queries, 5, executor=executor, stats=stats)
    assert stats.computed_distances == first, "a repeat replaces the reading rather than adding to it"
    total.accumulate(stats)
    assert total.computed_distances == 2 * first


@pytest.mark.parametrize(("queries", "wanted"), [(4, 3), (3, 10), (5, 10)])
def test_search_rejects_a_mismatched_out(executor, queries: int, wanted: int):
    """A reused run must be exactly the shape the call writes, no more and no less."""
    index, vectors = build_index(32, 16, executor)
    with pytest.raises(ShapeMismatchError):
        index.search(vectors[:4], 10, executor=executor, out=Matches(*matches_outputs(queries, wanted)))


def test_search_rejects_read_only_outputs(executor):
    """A read-only output is refused before anything is written into it."""
    index, vectors = build_index(32, 16, executor)
    read_only = Matches(bytes(2 * 2 * 16), np.zeros((2, 2)), np.zeros(2, np.uint64))
    with pytest.raises(ValueError, match="writable"):
        index.search(vectors[:2], 2, executor=executor, out=read_only)


def test_filtered_search_admits_only_approved_keys(executor):
    """A predicate narrows the answer without collapsing the walk that produced it."""
    index, vectors = build_index(128, 32, executor)
    matches = index.search(vectors[:4], 8, executor=executor, predicate=lambda key: int(key) % 2 == 0)
    found = keys_to_integers(matches.keys)
    counts = np.asarray(matches.counts)
    for query_index in range(len(counts)):
        assert np.all(found[query_index, : counts[query_index]] % 2 == 0)


def test_filtered_search_propagates_a_raising_predicate(executor):
    """A predicate that raises fails the search rather than silently narrowing it."""
    index, vectors = build_index(64, 16, executor)

    def broken(key):
        raise ValueError("predicate is broken")

    with pytest.raises(ValueError, match="predicate is broken"):
        index.search(vectors[:2], 4, executor=executor, predicate=broken)


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
    index = DenseIndex.for_ram(dimensions=8, quantization="f32")
    index.reserve(2)
    vectors = random_vectors(2, 8)
    index.add([1, 2], vectors, executor=executor)

    retrieved = np.asarray(index.get(1, dtype="f32"))
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
    index = DenseIndex.for_ram(dimensions=8, quantization="bf16")
    index.reserve(1)
    index.add([1], random_vectors(1, 8), executor=executor)
    view = index.get(1, dtype="bf16")
    assert view.format == "B"
    assert view.shape == (1, 16)


def test_arena_refuses_reuse_while_a_view_lives(executor):
    """Reuse rewinds the arena, so it waits until every view over the last outputs is released."""
    index, _ = build_index(4, 8, executor, quantization="f32")
    arena = Arena()
    first = index.get(1, executor=executor, arena=arena)
    with pytest.raises(ContendedError):
        index.get(2, executor=executor, arena=arena)
    first.release()
    assert np.asarray(index.get(2, executor=executor, arena=arena)).shape == (1, 8)


def test_duplicate_keys_stack(executor):
    """Keys need not be unique; several entries may share one and `count` reports how many."""
    index = DenseIndex.for_ram(dimensions=8, quantization="f32")
    index.reserve(3)
    index.add([5, 5, 5], random_vectors(3, 8), executor=executor)
    assert index.count(5) == 3
    assert np.asarray(index[5]).shape == (3, 8)


# endregion Retrieval

# region Iteration


def test_iteration_walks_every_key(executor):
    """A snapshot walk visits each published entry exactly once."""
    index, _ = build_index(300, 16, executor)
    walked = np.concatenate([np.asarray(batch).copy() for batch in index.cursor(64)], axis=0)
    assert sorted(keys_to_integers(walked).tolist()) == list(range(300))


def test_iteration_carries_typed_vectors(executor):
    """A walk asked for a scalar type yields that type, not the bytes underneath it."""
    index, vectors = build_index(40, 12, executor, quantization="f32")
    keys, rows = next(iter(index.cursor(16, dtype="f32")))
    assert rows.format == "f"
    assert rows.shape == (16, 12)
    assert np.asarray(keys).shape == (16, 16)


def test_search_stream_yields_nearest_first(executor):
    """A search stream is resumable, ordered by what the walk has reached."""
    index, vectors = build_index(128, 24, executor)
    stream = index.search_stream(vectors[0], executor=executor)
    streamed = [match.distance for _, match in zip(range(16), stream, strict=False)]
    assert streamed == sorted(streamed)


# endregion Iteration

# region Composition


def test_merge_absorbs_and_consumes_the_donor(executor):
    """The donor's entries arrive and its handle stops answering, both at once."""
    first = DenseIndex.for_ram(dimensions=16, quantization="f32")
    first.reserve(64)
    first.add(np.arange(0, 32), random_vectors(32, 16), executor=executor)

    second = DenseIndex.for_ram(dimensions=16, quantization="f32")
    second.reserve(32)
    second.add(np.arange(32, 64), random_vectors(32, 16, seed=1), executor=executor)

    first.merge(second, executor=executor)
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
    first.merge(second, executor=executor)
    assert len(first) == 32
    with pytest.raises(InvalidArgumentError):
        next(cursor)


def test_merge_refuses_itself(executor):
    """An index cannot absorb the handle it already is."""
    index, _ = build_index(8, 8, executor)
    with pytest.raises(InvalidArgumentError):
        index.merge(index, executor=executor)


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
    index, _ = build_index(128, 32, executor, quantization="f32")
    path = str(tmp_path / "index.safegraphs")
    index.save(path, executor=executor)

    metadata = DenseIndex.metadata_read(path)
    assert metadata["dimensions"] == 32
    assert metadata["metric"] == "angular"
    assert "residency" not in metadata, "a checkpoint fixes the geometry, never where it is hosted"

    reloaded = DenseIndex.load_ram(path, executor=executor)
    assert len(reloaded) == len(index)
    assert reloaded.dimensions == 32
    assert 7 in reloaded


def test_save_load_through_buffers(executor):
    """`save_buffer` returns the monolith a `load_ram_buffer` reads back."""
    index, _ = build_index(64, 16, executor, quantization="f32")
    buffer = index.save_buffer(executor=executor)
    assert isinstance(buffer, memoryview) and buffer.nbytes > 0
    assert DenseIndex.metadata_read_buffer(buffer)["dimensions"] == 16

    reloaded = DenseIndex.load_ram_buffer(buffer, executor=executor)
    assert len(reloaded) == len(index)


def test_cancelled_load_creates_no_index(executor, tmp_path):
    """A handler that stops a load leaves nothing behind but the exception."""
    index, _ = build_index(8, 8, executor, quantization="f32")
    path = str(tmp_path / "cancelled.safegraphs")
    index.save(path, executor=executor)
    with pytest.raises(CancelledError):
        DenseIndex.load_ram(path, executor=executor, progress=lambda completed, total: False)


def test_view_serves_in_place_and_refuses_mutation(executor, tmp_path):
    """A viewed checkpoint answers searches and refuses every mutating verb."""
    index, vectors = build_index(32, 16, executor, quantization="f32")
    path = str(tmp_path / "viewed.safegraphs")
    index.save(path, executor=executor)

    viewed = DenseIndex.view_ram(path)
    assert len(viewed) == 32
    assert len(viewed.search(vectors[:2], 3, executor=executor)) == 2
    with pytest.raises(ReadOnlyError) as raised:
        viewed.add([99], vectors[:1], executor=executor)
    assert raised.value.status == "read-only"


def test_encoded_path_is_read_as_a_path(executor, tmp_path):
    """`os.fsencode` returns `bytes`, which still names a file."""
    index, _ = build_index(8, 8, executor, quantization="f32")
    path = str(tmp_path / "encoded.safegraphs")
    index.save(path, executor=executor)
    metadata = DenseIndex.metadata_read(os.fsencode(path))
    assert metadata["dimensions"] == 8


def test_metadata_roundtrips_through_a_factory(executor, tmp_path):
    """The metadata keys are the geometry every factory takes, so `**` rebuilds the handle."""
    index = DenseIndex.for_ram(dimensions=48, metric="euclidean", quantization="i8", connectivity=24)
    index.reserve(8)
    index.add(np.arange(8), random_vectors(8, 48), executor=executor)
    path = str(tmp_path / "typed.safegraphs")
    index.save(path, executor=executor)

    rebuilt = DenseIndex.for_ram(**DenseIndex.metadata_read(path))
    assert rebuilt.dimensions == 48
    assert rebuilt.connectivity == 24


# endregion Serialization

# region Refinement


def test_refine_reports_a_sweep_and_leaves_the_index_standing(executor):
    """A sweep re-selects edges without publishing or retiring an entry, so the rows survive it."""
    index, vectors = build_index(64, 16, executor)
    changed = index.refine(executor=executor, expansion_factor=64)
    assert isinstance(changed, int)
    assert changed <= len(index), "a sweep cannot change more nodes than the graph holds"
    assert len(index) == 64
    assert len(index.search(vectors[:4], 5, executor=executor)) == 4


def test_refine_reports_telemetry(executor):
    """The sweep reports into the `Stats` a caller passes, as every other walk does."""
    index, _ = build_index(32, 16, executor)
    stats = Stats()
    index.refine(executor=executor, stats=stats)
    assert stats.visited_members > 0


def test_refine_takes_no_positionals(executor):
    """Every setting a sweep has is a keyword, so a bare number cannot be mistaken for one."""
    index, _ = build_index(8, 8, executor)
    with pytest.raises(TypeError):
        index.refine(64)


def test_refine_rejects_unknown_keywords(executor):
    """A misspelled setting is refused rather than silently derived."""
    index, _ = build_index(8, 8, executor)
    with pytest.raises(TypeError, match="expansion_add"):
        index.refine(executor=executor, expansion_add=64)


def test_refine_refuses_a_consumed_handle(executor):
    """A donor absorbed by `merge` has no graph left to sweep."""
    first, _ = build_index(8, 8, executor)
    second, _ = build_index(8, 8, executor, seed=1)
    first.merge(second, executor=executor)
    with pytest.raises(UninitializedError):
        second.refine(executor=executor)


def test_health_reaches_every_entry_after_a_refine(executor):
    """A refine re-links whatever it strands, so the audit after it reaches the whole index."""
    index, _ = build_index(64, 16, executor)
    index.refine(executor=executor)
    health = index.health(executor=executor)
    assert health["audited"] == len(index)
    assert health["reachable"] == health["audited"]
    assert health["sweeps"] >= 1


# endregion Refinement

# region Late Interaction

_LATE_INTERACTION_DOCUMENTS = 6
_LATE_INTERACTION_TOKENS = 3


def late_interaction_index(executor, dimensions: int = 16) -> tuple[DenseIndex, np.ndarray]:
    """An inner-product index whose entries are token rows, several sharing one document key."""
    rows = _LATE_INTERACTION_DOCUMENTS * _LATE_INTERACTION_TOKENS
    index = DenseIndex.for_ram(
        dimensions=dimensions,
        metric="inner",
        quantization="f32",
        values_per_key=_LATE_INTERACTION_TOKENS,
    )
    index.reserve(rows)
    documents = np.repeat(np.arange(_LATE_INTERACTION_DOCUMENTS, dtype=np.uint64), _LATE_INTERACTION_TOKENS)
    tokens = random_vectors(rows, dimensions, seed=5)
    index.add(integers_to_keys(documents), tokens, executor=executor)
    return index, tokens


def scored_documents(index, tokens, queries: int, wanted: int, executor, **kwargs) -> Matches:
    """Runs `maxsim` over *queries* whole queries."""
    return index.maxsim(
        tokens[: queries * _LATE_INTERACTION_TOKENS], _LATE_INTERACTION_TOKENS, wanted, executor=executor, **kwargs
    )


def test_maxsim_sizes_its_outputs_over_documents(executor):
    """A query is a run of token rows, so the result has one row per query rather than per token."""
    index, tokens = late_interaction_index(executor)
    queries = 2
    stats = Stats()
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
        index.maxsim(tokens[: _LATE_INTERACTION_TOKENS + 1], _LATE_INTERACTION_TOKENS, 4, executor=executor)


def test_maxsim_refuses_a_predicate(executor):
    """`maxsim` has no filtered sibling, so the keyword is refused rather than ignored."""
    index, tokens = late_interaction_index(executor)
    with pytest.raises(TypeError, match="predicate"):
        index.maxsim(
            tokens[:_LATE_INTERACTION_TOKENS],
            _LATE_INTERACTION_TOKENS,
            4,
            executor=executor,
            predicate=lambda key: True,
        )


def test_maxsim_refuses_a_distance_kind(executor):
    """Late interaction sums maxima over logits, which only the inner-product kinds score."""
    index, vectors = build_index(16, 8, executor, metric="angular")
    with pytest.raises(ConfigMismatchError):
        index.maxsim(vectors[:2], 2, 4, executor=executor)


def test_maxsim_reports_telemetry(executor):
    """The walk reports into the `Stats` a caller passes, as every other walk does."""
    index, tokens = late_interaction_index(executor)
    stats = Stats()
    scored_documents(index, tokens, 1, 4, executor, stats=stats)
    assert stats.computed_distances > 0


# endregion Late Interaction

# region Linear Algebra


def test_multiply_reports_every_scored_pair(executor):
    """A walk scores many more pairs than a search reports, and this verb hands them all back."""
    index, vectors = build_index(64, 32, executor, metric="inner")
    rows = vectors[:4]
    products = index.multiply(rows, 0, executor=executor)
    offsets = np.asarray(products["offsets"])
    named = keys_to_integers(products["keys"])
    scores = np.asarray(products["scores"])

    assert offsets.shape == (len(rows) + 1,)
    assert offsets[0] == 0 and offsets[-1] == len(scores) == products["required"]
    assert np.all(np.diff(offsets) > 0)

    # Every product is an exact dot against the row it names, which is the point of the verb.
    for row_index in range(len(rows)):
        for pair in range(int(offsets[row_index]), int(offsets[row_index + 1])):
            expected = float(rows[row_index] @ vectors[named[pair]])
            assert scores[pair] == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_multiply_reports_what_wanted_cut(executor):
    """A bound keeps the first pairs per row, and `required` still counts every pair scored."""
    index, vectors = build_index(64, 16, executor, metric="inner")
    products = index.multiply(vectors[:2], 4, executor=executor)
    assert np.diff(np.asarray(products["offsets"])).tolist() == [4, 4]
    assert products["required"] > 8, "both rows scored more pairs than were kept"


def test_multiply_refuses_a_predicate(executor):
    """`multiply` has no filtered sibling, so the keyword is refused rather than ignored."""
    index, vectors = build_index(16, 8, executor, metric="inner")
    with pytest.raises(TypeError, match="predicate"):
        index.multiply(vectors[:1], 4, executor=executor, predicate=lambda key: True)


def test_attend_produces_a_normalized_mixture(executor):
    """Attention outputs are already divided by their partitions, so they are convex mixtures."""
    dimensions, value_dimensions, context = 16, 8, 32
    index, keys_matrix = build_index(context, dimensions, executor, metric="inner")
    values = random_vectors(context, value_dimensions, seed=7)
    outputs = np.asarray(index.attend(keys_matrix[:4], values, executor=executor, outputs_dtype="f32"))

    # A softmax mixture of rows cannot leave the hull those rows span.
    assert outputs.shape == (4, value_dimensions)
    assert np.all(outputs >= values.min() - 1e-4)
    assert np.all(outputs <= values.max() + 1e-4)
    assert np.isfinite(outputs).all()


def test_attend_lands_in_the_caller_buffer(executor):
    """`out` is written through and handed back, so a decode loop allocates nothing."""
    index, keys_matrix = build_index(16, 8, executor, metric="inner")
    values = random_vectors(16, 4, seed=3)
    outputs = np.zeros((2, 4), dtype=np.float32)
    assert index.attend(keys_matrix[:2], values, executor=executor, outputs_dtype="f32", out=outputs) is outputs
    assert np.any(outputs != 0)


def test_attend_reports_recombinable_state(executor):
    """The running maximum and its partition come back in one row, which is what rescales it."""
    index, keys_matrix = build_index(16, 8, executor, metric="inner")
    values = random_vectors(16, 4, seed=3)
    carry = np.zeros((2, 2), dtype=np.float32)

    index.attend(keys_matrix[:2], values, executor=executor, outputs_dtype="f32", carry=carry)
    assert np.all(carry[:, 1] > 0), "a non-empty context has non-zero softmax mass"
    assert np.isfinite(carry).all()


def test_attend_refuses_a_half_width_carry(executor):
    """A maximum alone is a number nothing can rescale, so the pair is sized exactly."""
    index, keys_matrix = build_index(8, 8, executor, metric="inner")
    values = random_vectors(8, 4, seed=3)
    with pytest.raises(ValueError, match="carry"):
        index.attend(keys_matrix[:1], values, executor=executor, outputs_dtype="f32", carry=np.zeros(1, np.float32))


def test_attend_demands_rank_two_values(executor):
    """A value matrix carries its own width, which the C surface has no way to check."""
    index, keys_matrix = build_index(8, 8, executor, metric="inner")
    with pytest.raises(ShapeMismatchError):
        index.attend(keys_matrix[:1], random_vectors(1, 4)[0], executor=executor)


def test_attend_refuses_a_distance_kind(executor):
    """A softmax weighs logits, which only the inner-product kinds score."""
    index, keys_matrix = build_index(8, 8, executor, metric="angular")
    values = random_vectors(8, 4, seed=3)
    with pytest.raises(ConfigMismatchError):
        index.attend(keys_matrix[:1], values, executor=executor, outputs_dtype="f32")


# endregion Linear Algebra
