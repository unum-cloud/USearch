"""The vocabulary every modality shares - identities, telemetry, and failures.

Mirrors nothing in C++: `usearch/types.h` has no test unit of its own, because in C++ these are
plain structs. In Python they are types with protocols to honor, so they get one here.

Run:
    python -m pytest test/dtypes.py -v

File: test/dtypes.py
Author: Ash Vardanian
Date: October 1, 2026
"""

import copy
import ctypes
import pickle
import uuid
from enum import IntEnum

import numpy as np
import pytest
from base import BATCH_SIZES

import usearch
from usearch import (
    DenseTraversalStats,
    Error,
    InputOutputError,
    Key,
    MalformedError,
    ShapeMismatchError,
    Status,
)
from usearch.keys import integers_to_keys, keys_to_integers

# region Library


def test_capabilities_are_reported():
    """Both capability probes answer, whatever this machine turns out to be."""
    assert isinstance(usearch.capabilities_compiled(), str)
    assert isinstance(usearch.capabilities_available(), str)


def test_extension_names_itself():
    """The extension is `usearch.lib`, and its types are re-exported from the package."""
    assert usearch.lib.__name__ == "usearch.lib"
    assert usearch.DenseIndex.__module__ == "usearch"


@pytest.mark.parametrize("modality", ["SequenceIndex", "ShapesIndex", "GeospatialIndex"])
def test_placeholder_modalities_announce_themselves(modality: str):
    """A modality the library has not instantiated says so rather than failing obscurely."""
    with pytest.raises(NotImplementedError):
        getattr(usearch, modality)()


@pytest.mark.parametrize("name", ["Status", "Mood", "DenseMetric", "Scalar", "Residency", "Traversal"])
def test_modes_are_int_enums_of_the_package(name: str):
    """Statuses and modes mirror the C enumerators value for value, so nothing compares as a string."""
    enum = getattr(usearch, name)
    assert issubclass(enum, IntEnum)
    assert enum.__module__ == "usearch" and enum.__name__ == name
    assert pickle.loads(pickle.dumps(next(iter(enum)))) is next(iter(enum))


def test_status_values_follow_the_header():
    """`DEFERRED` is the one non-final status, right after `SUCCESS`, and the failures follow it."""
    assert (Status.SUCCESS, Status.DEFERRED) == (0, 1)
    assert min(status for status in Status if status not in (Status.SUCCESS, Status.DEFERRED)) == 2


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
    assert Key(value) == uuid.UUID(int=value) and hash(Key(value)) == hash(uuid.UUID(int=value))


def test_key_never_equals_what_merely_parses_into_it():
    """Text and bytes hash as themselves, so a key equal to them would break every dict holding both."""
    key = Key(5)
    assert key != str(key)
    assert key != key.bytes
    assert len({key, str(key), key.bytes}) == 3


def test_key_pickles_and_copies():
    """A key crosses `multiprocessing` and `copy` as its sixteen bytes."""
    key = Key(2**100 + 7)
    assert pickle.loads(pickle.dumps(key)) == key
    assert copy.deepcopy(key) == key


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
    """A binding-side rejection raises the class the engine would for the same fault."""
    with pytest.raises(MalformedError) as raised:
        Key("not-a-key")
    assert raised.value.status is Status.MALFORMED


@pytest.mark.parametrize("count", BATCH_SIZES)
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

# region Telemetry & Failures


def test_stats_start_empty():
    """A fresh `DenseTraversalStats` has counted nothing."""
    stats = DenseTraversalStats()
    assert stats.visited_members == 0
    assert stats.computed_distances == 0
    assert stats.restarts == 0
    assert repr(stats) == "DenseTraversalStats(visited_members=0, computed_distances=0, restarts=0)"


def test_stats_add_into_a_total():
    """A sweep sums its readings with `+=`, which keeps the total the same object."""
    total, reading = DenseTraversalStats(), DenseTraversalStats()
    alias = total
    total += reading
    assert total is alias and total.visited_members == 0
    with pytest.raises(TypeError):
        total += 1


def test_stats_takes_no_arguments():
    """`DenseTraversalStats` is a reading a call fills, not a value; it is only ever constructed empty."""
    with pytest.raises(TypeError):
        DenseTraversalStats(1)


def test_error_is_a_runtime_error():
    """`Error` subclasses `RuntimeError`, so code that never heard of USearch still catches it."""
    assert issubclass(Error, RuntimeError)


def test_every_failing_status_raises_its_own_class():
    """One `Error` subclass per final failing status of the C ABI, deriving the built-in naming the same fault."""
    subclasses = {
        name
        for name in dir(usearch)
        if isinstance(getattr(usearch, name), type) and issubclass(getattr(usearch, name), Error) and name != "Error"
    }
    assert len(subclasses) == len(Status) - 2, "success and deferred never raise"
    assert issubclass(ShapeMismatchError, ValueError)
    assert issubclass(InputOutputError, OSError)


# endregion Telemetry & Failures

# region Layouts


class _RAMOptions(ctypes.Structure):
    _fields_ = [
        ("memory_domains", ctypes.c_void_p),
        ("memory_domains_count", ctypes.c_size_t),
        ("shard_capacity", ctypes.c_size_t),
        ("cpu_class", ctypes.c_int),
        ("memory_limit", ctypes.c_size_t),
    ]


class _VRAMOptions(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("devices_count", ctypes.c_size_t),
        ("shard_capacity", ctypes.c_size_t),
        ("gpu_class", ctypes.c_int),
    ]


class _NVMeOptions(ctypes.Structure):
    _fields_ = [("paths", ctypes.c_void_p), ("paths_count", ctypes.c_size_t), ("cpu_class", ctypes.c_int)]


class _ExecutorForNVMe(ctypes.Structure):
    _fields_ = [
        ("executor", ctypes.c_void_p),
        ("outcome", ctypes.c_void_p),
        ("nvme", _NVMeOptions),
        ("threads", ctypes.c_size_t),
        ("nvme_class", ctypes.c_int),
    ]


class _TraversalStats(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t) for name in ("visited_members", "computed_distances", "restarts", "missing_hints")
    ]


class _DenseMetadata(ctypes.Structure):
    _fields_ = [
        ("metric", ctypes.c_int),
        ("quantization", ctypes.c_int),
        ("dimensions", ctypes.c_size_t),
        ("connectivity", ctypes.c_size_t),
    ]


class _DenseIndexDescription(ctypes.Structure):
    _fields_ = [
        ("metadata", _DenseMetadata),
        ("size", ctypes.c_size_t),
        ("capacity", ctypes.c_size_t),
        ("memory_usage", ctypes.c_size_t),
        ("cpu_class", ctypes.c_int),
        ("gpu_class", ctypes.c_int),
        ("shards_count", ctypes.c_size_t),
        ("load_path", ctypes.c_int),
    ]


@pytest.mark.parametrize(
    "name, mirror",
    [
        ("usearch_ram_options_t", _RAMOptions),
        ("usearch_vram_options_t", _VRAMOptions),
        ("usearch_nvme_options_t", _NVMeOptions),
        ("usearch_executor_for_nvme_t", _ExecutorForNVMe),
        ("usearch_traversal_stats_t", _TraversalStats),
        ("usearch_dense_index_description_t", _DenseIndexDescription),
    ],
)
def test_struct_layouts(name: str, mirror: type):
    """The argument structs the other bindings lay out by hand match the sizes and offsets the compiler chose."""
    size, offsets = usearch.lib.layouts()[name]
    assert ctypes.sizeof(mirror) == size
    for field, offset in offsets.items():
        assert getattr(mirror, field).offset == offset, field


# endregion Layouts
