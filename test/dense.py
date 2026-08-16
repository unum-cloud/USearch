"""Vector search over equidimensional dense embeddings - the Python twin of `test/dense.cpp`.

Every recall claim is checked against `usearch.eval.exact_matches`, which scores with NumKong's
kernels in the index's own metric vocabulary, so a number here compares like with like.

Run:
    python -m pytest test/dense.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import numkong as nk
from usearch import DenseIndex, Error, Key, Match, Matches, Stats
from usearch.eval import random_vectors, self_recall
from usearch.keys import integers_to_keys, keys_to_integers

from .base import (
    batch_sizes,
    build_index,
    dense_dimensions,
    distance_tolerances,
    input_dtypes,
    measured_recall,
    quantizations,
    recall_floor,
    requires_residency,
    spatial_metrics,
)

# region Construction


@pytest.mark.parametrize("dimensions", dense_dimensions)
@pytest.mark.parametrize("metric", spatial_metrics)
def test_construction_reports_its_geometry(dimensions: int, metric: str):
    """Every setting the constructor took reads back off the handle."""
    index = DenseIndex(dimensions=dimensions, metric=metric, quantization="bf16", connectivity=16)
    assert index.dimensions == dimensions
    assert index.connectivity == 16
    assert index.size == 0
    assert len(index) == 0
    assert isinstance(index.capabilities, str)
    assert f"dimensions={dimensions}" in repr(index)


@pytest.mark.parametrize(
    ("options", "failure"),
    [
        ({"metric": "manhattan"}, ValueError),
        ({"metric": "l2sq"}, ValueError),  # An alias we deliberately no longer accept.
        ({"quantization": "f128"}, ValueError),
        ({"residency": "tape"}, ValueError),
        ({"addressing": "device"}, ValueError),  # Renamed to 'device-only'.
        ({"expansion_add": 64}, TypeError),  # A v2 keyword the surface no longer has.
        ({"ndim": 8}, TypeError),  # Renamed to 'dimensions'.
    ],
)
def test_construction_rejects_unknown_options(options: dict, failure: type):
    """A name the surface does not have is refused, not quietly ignored."""
    with pytest.raises(failure):
        DenseIndex(dimensions=8, **options)


def test_construction_takes_three_positionals():
    """The three settings every caller names may be positional; the rest must be keywords."""
    index = DenseIndex(32, "euclidean", "f32")
    assert index.dimensions == 32
    with pytest.raises(TypeError):
        DenseIndex(32, "euclidean", "f32", 16)


@requires_residency("ram")
def test_ram_residency_takes_its_options():
    """The RAM options are named, not positional, and none of them is a bare flag."""
    index = DenseIndex(dimensions=8, residency="ram", threads=2, pages="base")
    assert index.dimensions == 8


# endregion Construction

# region Ingest


@pytest.mark.parametrize("count", batch_sizes)
@pytest.mark.parametrize("dimensions", dense_dimensions)
def test_add_publishes_every_row(count: int, dimensions: int):
    """A batch is all-or-nothing, so every row of an accepted batch is visible afterwards."""
    index, _ = build_index(count, dimensions)
    assert len(index) == count
    assert index.capacity >= count


@pytest.mark.parametrize("quantization", quantizations)
def test_add_accepts_every_quantization(quantization: str):
    """Every stored scalar type takes the same `float32` rows and quantizes them on ingest."""
    index, _ = build_index(64, 96, quantization=quantization)
    assert len(index) == 64


@pytest.mark.parametrize("input_dtype", input_dtypes)
def test_add_accepts_any_input_dtype(input_dtype):
    """The query type is free to differ from the stored quantization."""
    index = DenseIndex(dimensions=32, quantization="bf16")
    index.reserve(16)
    vectors = (np.random.default_rng(0).random((16, 32)) * 10).astype(input_dtype)
    assert index.add(np.arange(16), vectors) == 16


def test_add_accepts_numkong_tensors_without_numpy():
    """A NumKong tensor is a `memoryview`, so it reaches the index through PEP 3118 alone.

    This is the claim the README leads with, and the only test that exercises it: every other
    path in this file arrives as a NumPy array.
    """
    index = DenseIndex(dimensions=16, quantization="f32")
    index.reserve(8)
    rows = nk.Tensor(random_vectors(8, 16))
    assert memoryview(rows).shape == (8, 16), "NumKong tensors export PEP 3118 directly"
    assert index.add(np.arange(8), rows) == 8
    assert len(index) == 8


@pytest.mark.parametrize("keys", [[Key(11), 12], np.arange(2), integers_to_keys(np.arange(2))])
def test_add_accepts_every_key_spelling(keys):
    """Keys arrive as objects, as integers, or as the packed wire rows."""
    index = DenseIndex(dimensions=4, quantization="f32")
    index.reserve(2)
    assert index.add(keys, random_vectors(2, 4)) == 2
    assert len(index) == 2


def test_add_rejects_mismatched_key_count():
    """A key per row, or the batch is refused before anything is published."""
    index = DenseIndex(dimensions=4, quantization="f32")
    index.reserve(4)
    with pytest.raises(Error) as raised:
        index.add([1, 2, 3], random_vectors(2, 4))
    assert raised.value.status == "shape-mismatch"
    assert len(index) == 0


def test_add_reports_telemetry():
    """Telemetry accumulates into the `Stats` a caller passes, across calls."""
    index = DenseIndex(dimensions=16, quantization="f32")
    index.reserve(128)
    stats = Stats()
    index.add(np.arange(64), random_vectors(64, 16), stats=stats)
    after_first = stats.computed_distances
    index.add(np.arange(64, 128), random_vectors(64, 16, seed=1), stats=stats)
    assert after_first > 0
    assert stats.computed_distances > after_first


def test_add_before_reserve_is_refused():
    """A handle whose geometry is not frozen has nowhere to put a row."""
    index = DenseIndex(quantization="f32")
    with pytest.raises(Error):
        index.add([1], random_vectors(1, 4))


# endregion Ingest

# region Search


@pytest.mark.parametrize("dimensions", dense_dimensions)
def test_search_shapes(dimensions: int):
    """The three result matrices are shaped by the batch and the "k" that was asked for."""
    index, vectors = build_index(64, dimensions)
    matches: Matches = index.search(vectors[:3], 5)

    assert np.asarray(matches.keys).shape == (3, 5, 16)
    assert np.asarray(matches.distances).shape == (3, 5)
    assert np.asarray(matches.counts).shape == (3,)
    assert len(matches) == 3
    assert matches.computed_distances > 0


def test_search_rows_are_matches():
    """A result row is a tuple of scored entries, nearest first."""
    index, vectors = build_index(64, 32)
    row = index.search(vectors[:1], 4)[0]
    assert all(isinstance(match, Match) for match in row)
    assert isinstance(row[0].key, Key)
    assert row[0].distance <= row[-1].distance


def test_search_pads_unfilled_entries_with_zeros():
    """Entries past a count are zeroed, never left as whatever the heap last held."""
    index, vectors = build_index(3, 8)
    matches = index.search(vectors[:1], 10)
    found = np.asarray(matches.keys)
    count = int(np.asarray(matches.counts)[0])
    assert count <= 3
    assert not found[0, count:].any()


def test_search_wants_at_least_one():
    """Asking for no matches is a shape error, not an empty answer."""
    index, vectors = build_index(8, 8)
    with pytest.raises(Error) as raised:
        index.search(vectors[:1], 0)
    assert raised.value.status == "shape-mismatch"


def test_search_rejects_an_overflowing_request():
    """An output matrix that cannot fit memory is refused rather than silently wrapped."""
    index, vectors = build_index(4, 8)
    with pytest.raises(Error) as raised:
        index.search(vectors[:2], 2**62)
    assert raised.value.status == "overflow"


@pytest.mark.parametrize("dimensions", [97, 256])
@pytest.mark.parametrize("metric", spatial_metrics)
def test_recall_against_the_oracle(dimensions: int, metric: str):
    """The index must return most of what a brute-force scan would, in every metric."""
    index, vectors = build_index(512, dimensions, metric=metric, quantization="f32")
    recall = measured_recall(index, vectors, vectors[:64], 10, metric=metric)
    assert recall >= recall_floor("f32"), f"{metric} recall collapsed to {recall}"


@pytest.mark.parametrize("quantization", quantizations)
def test_recall_survives_quantization(quantization: str):
    """Every stored scalar type still finds what it should, at its own floor."""
    index, _ = build_index(512, 96, quantization=quantization)
    recall = self_recall(index).mean_recall
    assert recall >= recall_floor(quantization), f"{quantization} recall collapsed to {recall}"


@pytest.mark.parametrize("quantization", ["f64", "f32", "bf16", "f16", "i8"])
def test_distances_agree_with_the_oracle(quantization: str):
    """A reported distance matches what NumKong computes for the same pair, within tolerance."""
    index, vectors = build_index(128, 64, quantization=quantization)
    queries = vectors[:8]
    matches = index.search(queries, 1)
    found = keys_to_integers(matches.keys)[:, 0]
    reported = np.asarray(matches.distances)[:, 0]
    truth = np.asarray(nk.cdist(queries, vectors, "angular", out_dtype="float64"))
    expected = truth[np.arange(len(queries)), found]
    atol, rtol = distance_tolerances(quantization)
    np.testing.assert_allclose(reported, expected, atol=atol, rtol=rtol)


def test_expansion_rides_the_call():
    """The candidate pool is a per-call setting, so one immutable index serves a whole sweep."""
    index, vectors = build_index(512, 64)
    queries = vectors[:32]
    narrow = index.search(queries, 10, expansion_factor=16)
    wide = index.search(queries, 10, expansion_factor=256)
    assert wide.computed_distances > narrow.computed_distances


def test_search_into_matches_the_allocating_door():
    """Both doors onto one implementation must return the same thing."""
    index, vectors = build_index(256, 48)
    queries = vectors[:8]
    keys = np.zeros((8, 10, 16), dtype=np.uint8)
    distances = np.zeros((8, 10), dtype=np.float64)
    counts = np.zeros(8, dtype=np.uint64)

    index.search_into(queries, 10, keys, distances, counts)
    allocated = index.search(queries, 10)
    assert np.array_equal(np.asarray(allocated.counts), counts)
    assert np.array_equal(np.asarray(allocated.keys), keys)
    assert np.allclose(np.asarray(allocated.distances), distances)


@pytest.mark.parametrize("bad_shape", [(4, 3, 16), (4, 10, 8), (3, 10, 16)])
def test_search_into_rejects_wrong_capacity(bad_shape: tuple):
    """An output buffer must hold exactly what its shape implies, no more and no less."""
    index, vectors = build_index(32, 16)
    with pytest.raises(ValueError):
        index.search_into(vectors[:4], 10, np.zeros(bad_shape, np.uint8), np.zeros((4, 10)), np.zeros(4, np.uint64))


def test_search_into_rejects_read_only_outputs():
    """A read-only output is refused before anything is written into it."""
    index, vectors = build_index(32, 16)
    with pytest.raises(ValueError):
        index.search_into(vectors[:2], 2, bytes(2 * 2 * 16), np.zeros((2, 2)), np.zeros(2, np.uint64))


def test_filtered_search_admits_only_approved_keys():
    """A predicate narrows the answer without collapsing the walk that produced it."""
    index, vectors = build_index(128, 32)
    matches = index.search(vectors[:4], 8, predicate=lambda key: int(key) % 2 == 0)
    found = keys_to_integers(matches.keys)
    counts = np.asarray(matches.counts)
    for query_index in range(len(counts)):
        assert np.all(found[query_index, : counts[query_index]] % 2 == 0)


def test_filtered_search_propagates_a_raising_predicate():
    """A predicate that raises fails the search rather than silently narrowing it."""
    index, vectors = build_index(64, 16)

    def broken(key):
        raise ValueError("predicate is broken")

    with pytest.raises(ValueError, match="predicate is broken"):
        index.search(vectors[:2], 4, predicate=broken)


# endregion Search

# region Retrieval


def test_count_and_contains():
    """Membership and multiplicity are the same question asked two ways."""
    index, _ = build_index(32, 16)
    assert index.count(0) == 1
    assert index.count(10_000) == 0
    assert 0 in index
    assert 10_000 not in index


def test_get_returns_the_stored_rows():
    """An `f32` index hands back `f32` rows unchanged."""
    index = DenseIndex(dimensions=8, quantization="f32")
    index.reserve(2)
    vectors = random_vectors(2, 8)
    index.add([1, 2], vectors)

    retrieved = np.asarray(index.get(1, dtype="f32"))
    assert retrieved.shape == (1, 8)
    assert np.allclose(retrieved[0], vectors[0], atol=1e-6)


def test_get_answers_none_and_subscript_raises():
    """`get` is a `get`; subscription is a lookup, so the two answer absence differently."""
    index, _ = build_index(4, 8)
    assert index.get(10_000) is None
    with pytest.raises(KeyError):
        index[10_000]


def test_get_leaves_unformattable_kinds_as_bytes():
    """Sub-byte and mini-float kinds have no `memoryview` format, so they leave as raw runs."""
    index = DenseIndex(dimensions=8, quantization="bf16")
    index.reserve(1)
    index.add([1], random_vectors(1, 8))
    view = index.get(1, dtype="bf16")
    assert view.format == "B"
    assert view.shape == (1, 16)


def test_duplicate_keys_stack():
    """Keys need not be unique; several entries may share one and `count` reports how many."""
    index = DenseIndex(dimensions=8, quantization="f32")
    index.reserve(3)
    index.add([5, 5, 5], random_vectors(3, 8))
    assert index.count(5) == 3
    assert np.asarray(index[5]).shape == (3, 8)


# endregion Retrieval

# region Iteration


def test_iteration_walks_every_key():
    """A snapshot walk visits each published entry exactly once."""
    index, _ = build_index(300, 16)
    walked = np.concatenate([np.asarray(batch).copy() for batch in index.batches(64)], axis=0)
    assert sorted(keys_to_integers(walked).tolist()) == list(range(300))


def test_iteration_carries_typed_vectors():
    """A walk asked for a scalar type yields that type, not the bytes underneath it."""
    index, vectors = build_index(40, 12, quantization="f32")
    keys, rows = next(iter(index.batches(16, dtype="f32")))
    assert rows.format == "f"
    assert rows.shape == (16, 12)
    assert np.asarray(keys).shape == (16, 16)


def test_finder_streams_nearest_first():
    """A finder is a resumable stream, ordered by what the walk has reached."""
    index, vectors = build_index(128, 24)
    streamed = [match.distance for _, match in zip(range(16), index.find(vectors[0]), strict=False)]
    assert streamed == sorted(streamed)


# endregion Iteration

# region Composition


def test_merge_absorbs_and_consumes_the_donor():
    """The donor's entries arrive and its handle stops answering, both at once."""
    first = DenseIndex(dimensions=16, quantization="f32")
    first.reserve(64)
    first.add(np.arange(0, 32), random_vectors(32, 16))

    second = DenseIndex(dimensions=16, quantization="f32")
    second.reserve(32)
    second.add(np.arange(32, 64), random_vectors(32, 16, seed=1))

    first.merge(second)
    assert len(first) == 64
    assert 40 in first

    with pytest.raises(Error) as raised:
        len(second)
    assert raised.value.status == "uninitialized"


def test_merge_refuses_itself():
    """An index cannot absorb the handle it already is."""
    index, _ = build_index(8, 8)
    with pytest.raises(Error):
        index.merge(index)


def test_clear_keeps_the_geometry():
    """Clearing drops the entries and nothing else."""
    index, _ = build_index(32, 16)
    capacity = index.capacity
    index.clear()
    assert len(index) == 0
    assert index.capacity == capacity
    assert index.dimensions == 16


# endregion Composition

# region Serialization


def test_save_load_roundtrip(tmp_path):
    """A checkpoint restores what it stored, through a handle rebuilt from its own metadata."""
    index, _ = build_index(128, 32, quantization="f32")
    path = str(tmp_path / "index.safegraphs")
    index.save(path)

    metadata = DenseIndex.metadata(path)
    assert metadata["dimensions"] == 32
    assert metadata["metric"] == "angular"

    reloaded = DenseIndex(**metadata)
    reloaded.load(path)
    assert len(reloaded) == len(index)
    assert 7 in reloaded


def test_save_load_through_buffers():
    """A `memoryview` is a checkpoint; a `bytes` path is a path."""
    index, _ = build_index(64, 16, quantization="f32")
    buffer = memoryview(bytearray(index.serialized_length))
    index.save(buffer)

    reloaded = DenseIndex(**DenseIndex.metadata(buffer))
    reloaded.load(buffer)
    assert len(reloaded) == len(index)


def test_encoded_path_is_read_as_a_path(tmp_path):
    """`os.fsencode` returns `bytes`, which must still name a file rather than a blob."""
    index, _ = build_index(8, 8, quantization="f32")
    path = str(tmp_path / "encoded.safegraphs")
    index.save(path)
    metadata = DenseIndex.metadata(os.fsencode(path))
    assert metadata["dimensions"] == 8


def test_metadata_roundtrips_through_the_constructor(tmp_path):
    """The metadata keys are the constructor's own keywords, so `**` rebuilds the handle."""
    index = DenseIndex(dimensions=48, metric="euclidean", quantization="i8", connectivity=24)
    index.reserve(8)
    index.add(np.arange(8), random_vectors(8, 48))
    path = str(tmp_path / "typed.safegraphs")
    index.save(path)

    rebuilt = DenseIndex(**DenseIndex.metadata(path))
    assert rebuilt.dimensions == 48
    assert rebuilt.connectivity == 24


# endregion Serialization

# region Linear Algebra


def test_multiply_reports_every_scored_pair():
    """A walk scores many more pairs than a search reports, and this verb hands them all back."""
    index, vectors = build_index(64, 32)
    rows = vectors[:4]
    wanted = 64
    columns = np.zeros((4, wanted, 16), dtype=np.uint8)
    products = np.zeros((4, wanted), dtype=np.float64)
    counts = np.zeros(4, dtype=np.uint64)

    truncated = index.multiply_into(rows, wanted, columns, products, counts)
    assert truncated == 0
    assert counts.min() > 0

    # Every product is an exact dot against the row it names, which is the point of the verb.
    named = keys_to_integers(columns)
    for row_index in range(len(rows)):
        for pair in range(int(counts[row_index])):
            expected = float(rows[row_index] @ vectors[named[row_index, pair]])
            assert products[row_index, pair] == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_multiply_reports_truncation():
    """A count above the column width means that row was cut, and the return says how many were."""
    index, vectors = build_index(64, 16)
    wanted = 4
    columns = np.zeros((2, wanted, 16), dtype=np.uint8)
    products = np.zeros((2, wanted), dtype=np.float64)
    counts = np.zeros(2, dtype=np.uint64)

    truncated = index.multiply_into(vectors[:2], wanted, columns, products, counts)
    assert truncated == 2, "both rows scored more pairs than the outputs could hold"
    assert counts.max() > wanted, "the count is the true total, not the number written"


def test_multiply_refuses_a_predicate():
    """`multiply` has no filtered sibling, so the keyword is refused rather than ignored."""
    index, vectors = build_index(16, 8)
    with pytest.raises(TypeError, match="predicate"):
        index.multiply_into(
            vectors[:1],
            4,
            np.zeros((1, 4, 16), np.uint8),
            np.zeros((1, 4)),
            np.zeros(1, np.uint64),
            predicate=lambda key: True,
        )


def test_attend_produces_a_normalized_mixture():
    """Attention outputs are already divided by their partitions, so they are convex mixtures."""
    dimensions, value_dimensions, context = 16, 8, 32
    index, keys_matrix = build_index(context, dimensions, quantization="f32")
    values = random_vectors(context, value_dimensions, seed=7)
    queries = keys_matrix[:4]
    outputs = np.zeros((4, value_dimensions), dtype=np.float32)

    index.attend_into(queries, values, outputs, outputs_dtype="f32")

    # A softmax mixture of rows cannot leave the hull those rows span.
    assert np.all(outputs >= values.min() - 1e-4)
    assert np.all(outputs <= values.max() + 1e-4)
    assert np.isfinite(outputs).all()


def test_attend_reports_recombinable_state():
    """The running maximum and its partition come back together, or not at all."""
    index, keys_matrix = build_index(16, 8, quantization="f32")
    values = random_vectors(16, 4, seed=3)
    queries = keys_matrix[:2]
    outputs = np.zeros((2, 4), dtype=np.float32)
    maxima = np.zeros(2, dtype=np.float32)
    partitions = np.zeros(2, dtype=np.float32)

    index.attend_into(queries, values, outputs, outputs_dtype="f32", maximum_scores=maxima, partitions=partitions)
    assert np.all(partitions > 0), "a non-empty context has non-zero softmax mass"
    assert np.isfinite(maxima).all()


def test_attend_demands_both_halves_of_the_state():
    """One of the pair alone is a number nothing can rescale, so it is refused."""
    index, keys_matrix = build_index(8, 8, quantization="f32")
    values = random_vectors(8, 4, seed=3)
    outputs = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(Error) as raised:
        index.attend_into(keys_matrix[:1], values, outputs, outputs_dtype="f32", maximum_scores=np.zeros(1, np.float32))
    assert raised.value.status == "shape-mismatch"


def test_attend_demands_rank_two_values():
    """A value matrix carries its own width, which the C surface has no way to check."""
    index, keys_matrix = build_index(8, 8, quantization="f32")
    with pytest.raises(Error):
        index.attend_into(keys_matrix[:1], random_vectors(1, 4)[0], np.zeros((1, 4), np.float32))


# endregion Linear Algebra
