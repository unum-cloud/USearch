"""Type stubs for the vocabulary every modality shares."""

import os
import uuid
from collections.abc import Buffer, Sequence
from typing import Literal, NoReturn

# region: Shared Vocabulary

# Every batch verb reads PEP 3118 first and `__array_interface__` second, so NumPy is never
# required - at runtime or here. Frameworks whose stubs predate PEP 688, PyTorch among them,
# support the protocol at runtime but need a `cast` to satisfy a checker.
type Tensor = Buffer

# The scalar kinds the engine stores and exchanges, widest dynamic range first.
type ScalarKind = Literal[
    "f64", "f32", "bf16", "f16", "e5m2", "e4m3", "e3m2", "e2m3", "i8", "i4x2", "u8", "u4x2", "u1x8"
]

# The spellings NumPy and the wider ecosystem hand us for the same kinds, accepted anywhere a
# `ScalarKind` is - these arrive from `array.dtype.str`, so they are not synonyms we invented.
type ScalarAlias = Literal[
    "float64",
    "double",
    "<f8",
    "float32",
    "float",
    "<f4",
    "bfloat16",
    "float16",
    "half",
    "<f2",
    "int8",
    "|i1",
    "int4",
    "i4",
    "uint8",
    "|u1",
    "uint4",
    "u4",
    "uint1",
    "u1",
    "b1",
    "bool",
]
type ScalarName = ScalarKind | ScalarAlias

type KeyLike = Key | int | str | uuid.UUID | Buffer

# What an executor does with its workers once idle: park them, or keep them spinning.
type MoodName = Literal["chill", "grind"]
type PathLike = str | bytes | os.PathLike[str] | os.PathLike[bytes]

class Error(RuntimeError):
    """The base of every failure the library reports, one subclass per status shared with the C API."""

    status: str

class OutOfMemoryError(Error):
    """An allocation or reservation could not be satisfied."""

    status: Literal["out-of-memory"]

class ShapeMismatchError(Error):
    """Dimensionality, row counts, or output capacity disagree."""

    status: Literal["shape-mismatch"]

class UnsupportedLayoutError(Error):
    """A stride or packing the engine cannot traverse."""

    status: Literal["unsupported-layout"]

class UnknownKindError(Error):
    """A scalar, metric, or residency this build does not know."""

    status: Literal["unknown-kind"]

class ReadOnlyError(Error):
    """A mutating verb on a handle that serves its checkpoint in place."""

    status: Literal["read-only"]

class InputOutputError(Error):
    """A file could not be opened, read, written, or parsed."""

    status: Literal["io"]

class DeviceError(Error):
    """A device could not be selected, or its memory could not be claimed."""

    status: Literal["device"]

class MalformedError(Error):
    """A representation could not be decoded - a key, a buffer, a checkpoint."""

    status: Literal["malformed"]

class InternalError(Error):
    """The library reported neither success nor a failure - a defect."""

    status: Literal["internal"]

class CapacityOverflowError(Error):
    """A hard design ceiling was reached; a smaller request cannot help."""

    status: Literal["overflow"]

class UninitializedError(Error):
    """The handle is not ready for the verb - unfrozen, or consumed by a merge."""

    status: Literal["uninitialized"]

class ConfigMismatchError(Error):
    """The request contradicts an already-frozen configuration."""

    status: Literal["config-mismatch"]

class ContendedError(Error):
    """An executor or arena is busy with another call."""

    status: Literal["contended"]

class CancelledError(Error):
    """A progress handler stopped the call, which published nothing."""

    status: Literal["cancelled"]

class UnsupportedError(Error):
    """This residency does not implement the verb; pick another residency."""

    status: Literal["unsupported"]

class InvalidArgumentError(Error):
    """An argument breaks the verb's contract."""

    status: Literal["invalid-argument"]

class Key:
    """A 128-bit entry identity, rendered as a lowercase dashed UUID."""

    def __init__(self, value: KeyLike = 0, /) -> None: ...
    @property
    def bytes(self) -> bytes:
        """The sixteen identity bytes, most significant first."""
        ...

    def __int__(self) -> int: ...
    def __index__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: KeyLike) -> bool: ...
    def __le__(self, other: KeyLike) -> bool: ...
    def __gt__(self, other: KeyLike) -> bool: ...
    def __ge__(self, other: KeyLike) -> bool: ...

class Executor:
    """The workers a fanned-out verb runs on, shared across indexes."""

    def __init__(self, *args: object, **keywords: object) -> NoReturn: ...
    @classmethod
    def for_ram(
        cls, threads: int = 0, *, memory_domains: Sequence[int] | None = None, in_flight_limit: int = 0
    ) -> Executor:
        """Spawns host worker threads for a RAM-resident index."""
        ...

    @classmethod
    def on_pool(
        cls, pool: int, /, *, memory_domains: Sequence[int] | None = None, in_flight_limit: int = 0
    ) -> Executor:
        """Drives RAM-resident indexes on a thread pool the caller owns, passed by address."""
        ...

    @classmethod
    def for_vram(cls, *, devices: Sequence[int] | None = None, in_flight_limit: int = 0) -> Executor:
        """Opens one stream per device for a VRAM-resident index."""
        ...

    @classmethod
    def for_nvme(
        cls, threads: int = 0, *, paths: Sequence[PathLike] | None = None, in_flight_limit: int = 0
    ) -> Executor:
        """Opens one submission ring per worker for an NVMe-resident index."""
        ...

    def drain(self, mood: MoodName = "chill") -> None:
        """Waits for every call submitted before it, then sets what idle workers do."""
        ...

    def __repr__(self) -> str: ...

class Arena:
    """The memory a verb lands the outputs only it can size in, reused by the next call handed it."""

    def __init__(self) -> None: ...

class Stats:
    """Traversal telemetry, fully overwritten by every call it is passed to."""

    def __init__(self) -> None: ...
    @property
    def visited_members(self) -> int: ...
    @property
    def computed_distances(self) -> int: ...
    @property
    def restarts(self) -> int: ...
    def accumulate(self, other: Stats, /) -> None:
        """Folds another reading's counters into this one, summing a sweep's calls."""
        ...

    def __repr__(self) -> str: ...

class Match:
    """One scored entry: the identity found and its distance."""

    @property
    def key(self) -> Key: ...
    @property
    def distance(self) -> float: ...
    def __repr__(self) -> str: ...

class Matches:
    """A ranked run, as `search` and `maxsim` return it, reading through to three buffers."""

    def __init__(self, keys: Buffer, distances: Buffer, counts: Buffer, /) -> None: ...
    @property
    def keys(self) -> memoryview:
        """(queries, wanted, 16) identities, most significant byte first."""
        ...

    @property
    def distances(self) -> memoryview:
        """(queries, wanted) distances in the index's own metric."""
        ...

    @property
    def counts(self) -> memoryview:
        """(queries,) match counts, each bounding its own row."""
        ...

    def __len__(self) -> int: ...
    def __getitem__(self, query_index: int, /) -> tuple[Match, ...]: ...
    def __repr__(self) -> str: ...

# endregion: Shared Vocabulary
