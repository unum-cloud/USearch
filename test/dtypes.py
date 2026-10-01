"""The vocabulary every modality shares - identities, telemetry, and failures.

Mirrors nothing in C++: `usearch/types.h` has no test unit of its own, because in C++ these are
plain structs. In Python they are types with protocols to honor, so they get one here.

Run:
    python -m pytest test/dtypes.py -v
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from base import batch_sizes, matches_outputs

import usearch
from usearch import Error, Key, Matches, Stats
from usearch.keys import integers_to_keys, keys_to_integers

# region Library


def test_version_is_semantic():
    """The version is a three-part semantic version, and the package agrees with the library."""
    assert usearch.version().count(".") == 2
    assert usearch.__version__ == usearch.version()


def test_capabilities_are_reported():
    """Both capability probes answer, whatever this machine turns out to be."""
    assert isinstance(usearch.capabilities_compiled(), str)
    assert isinstance(usearch.capabilities_available(), str)


def test_extension_names_itself():
    """The extension is `usearch.lib`, and its types are re-exported from the package."""
    assert usearch.lib.__name__ == "usearch.lib"
    assert usearch.DenseIndex.__module__ == "usearch"


@pytest.mark.parametrize("modality", ["SequenceIndex", "MatrixIndex", "GeospatialIndex"])
def test_placeholder_modalities_announce_themselves(modality: str):
    """A modality the library has not instantiated says so rather than failing obscurely."""
    with pytest.raises(NotImplementedError):
        getattr(usearch, modality)()


# endregion Library

# region Keys


@pytest.mark.parametrize("value", [0, 1, 42, 255, 256, 2**63, 2**127 - 1])
def test_key_integer_roundtrip(value: int):
    """An integer identity survives every rendering the type offers."""
    key = Key(value)
    assert int(key) == value
    assert Key(str(key)) == key
    assert Key(key.bytes) == key
    assert len(key.bytes) == 16


def test_key_renders_as_dashed_uuid():
    """Integer identities park in the low eight bytes, so small ones read as mostly zeros."""
    assert str(Key(42)) == "00000000-0000-0000-0000-00000000002a"
    assert repr(Key(42)) == "Key('00000000-0000-0000-0000-00000000002a')"


def test_key_reads_a_uuid():
    """Anything shaped like `uuid.UUID` is already sixteen bytes."""
    identity = uuid.uuid4()
    assert int(Key(identity)) == identity.int


@pytest.mark.parametrize("value", [0, 255, 256, 1000, 2**64, 2**127 - 1])
def test_key_hashes_as_the_integer_it_equals(value: int):
    """Equality and hashing must agree, or a dict keyed by one cannot be read by the other.

    The naive byte-polynomial hash coincides with `hash(int)` below 256 and diverges above it,
    which is exactly the range a shallow test would never reach.
    """
    assert Key(value) == value
    assert hash(Key(value)) == hash(value)
    assert {value: "stored"}[Key(value)] == "stored"
    assert len({Key(value), value}) == 1


def test_key_orders_by_identity():
    """Keys sort, so a caller can put them in a sorted container."""
    assert Key(1) < Key(2)
    assert sorted([Key(3), Key(1), Key(2)]) == [Key(1), Key(2), Key(3)]


@pytest.mark.parametrize(
    ("value", "failure"),
    [("not-a-key", Error), (2**128, OverflowError), (-1, OverflowError), (object(), TypeError)],
)
def test_key_rejects_malformed(value, failure):
    """A key that cannot be read is a failure, never a silent zero."""
    with pytest.raises(failure):
        Key(value)


def test_key_malformed_text_carries_a_status():
    """A binding-side rejection reports the status the engine would for the same fault."""
    with pytest.raises(Error) as raised:
        Key("not-a-key")
    assert raised.value.status == "malformed-input"


@pytest.mark.parametrize("count", batch_sizes)
def test_identity_packing_roundtrip(count: int):
    """Integer identities survive the trip through the sixteen-byte wire rows."""
    original = np.arange(count, dtype=np.uint64)
    packed = integers_to_keys(original)
    assert packed.shape == (count, 16)
    assert not packed[:, :8].any(), "Integer identities park in the low eight bytes"
    assert np.array_equal(keys_to_integers(packed), original)
    assert [int(Key(bytes(row))) for row in packed] == list(range(count))


def test_identities_wider_than_an_integer_are_refused():
    """An identity using its high eight bytes has no integer form, so the run is refused."""
    packed = integers_to_keys(np.arange(4, dtype=np.uint64))
    packed[2, 0] = 1
    with pytest.raises(ValueError):
        keys_to_integers(packed)


# endregion Keys

# region Matches


@pytest.mark.parametrize(("queries", "wanted"), [(1, 1), (3, 5), (7, 16)])
def test_matches_derives_its_width(queries: int, wanted: int):
    """The matches-per-query width follows from the three lengths, never from an argument."""
    matches = Matches(*matches_outputs(queries, wanted))
    assert len(matches) == queries
    assert np.asarray(matches.keys).shape == (queries, wanted, 16)
    assert np.asarray(matches.distances).shape == (queries, wanted)
    assert np.asarray(matches.counts).shape == (queries,)
    assert f"wanted={wanted}" in repr(matches)


def test_matches_reads_through_to_the_buffers():
    """The view borrows rather than copies, so a write lands where the reader is looking."""
    keys, distances, counts = matches_outputs(1, 2)
    matches = Matches(keys, distances, counts)
    keys[0, 0] = integers_to_keys([42])[0]
    distances[0, 0] = 0.25
    counts[0] = 1

    row = matches[0]
    assert len(row) == 1
    assert int(row[0].key) == 42
    assert row[0].distance == 0.25


def test_matches_keeps_its_buffers_alive():
    """The exporters are held for the view's whole life, so the last name may go."""
    matches = Matches(*matches_outputs(2, 3))
    assert int(np.asarray(matches.counts).sum()) == 0
    assert len(matches[1]) == 0


@pytest.mark.parametrize(
    "outputs",
    [
        (np.zeros((1, 2, 16), np.uint8), np.zeros((1, 2)), np.zeros(0, np.uint64)),  # No query at all.
        (np.zeros((1, 2, 16), np.uint8), np.zeros((1, 3)), np.zeros(1, np.uint64)),  # Mismatched widths.
        (np.zeros((1, 3, 16), np.uint8), np.zeros((1, 3)), np.zeros(2, np.uint64)),  # Ragged rows.
        (np.zeros(15, np.uint8), np.zeros(1), np.zeros(1, np.uint64)),  # A partial identity.
    ],
)
def test_matches_refuses_disagreeing_buffers(outputs: tuple):
    """Three lengths that cannot be read as one shape are a shape error, not a guess."""
    with pytest.raises(Error) as raised:
        Matches(*outputs)
    assert raised.value.status == "shape-mismatch"


def test_matches_takes_three_buffers():
    """A run is three outputs, so anything else is refused where it is written."""
    with pytest.raises(TypeError):
        Matches(np.zeros((1, 1, 16), np.uint8), np.zeros((1, 1)))


# endregion Matches

# region Telemetry & Failures


def test_stats_start_empty():
    """A fresh `Stats` has counted nothing."""
    stats = Stats()
    assert stats.visited_members == 0
    assert stats.computed_distances == 0
    assert stats.restarts == 0
    assert "Stats(" in repr(stats)


def test_stats_takes_no_arguments():
    """`Stats` is a reading a call fills, not a value; it is only ever constructed empty."""
    with pytest.raises(TypeError):
        Stats(1)


def test_error_is_a_runtime_error():
    """`Error` subclasses `RuntimeError`, so code that never heard of USearch still catches it."""
    assert issubclass(Error, RuntimeError)


# endregion Telemetry & Failures
