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
type PathLike = str | bytes | os.PathLike[str] | os.PathLike[bytes]

class Error(RuntimeError):
    """Every failure the engine reports, carrying the status shared with the C API."""

    status: str

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
    def for_ram(cls, threads: int = 0, *, memory_domains: Sequence[int] | None = None) -> Executor:
        """Spawns host worker threads for a RAM-resident index."""
        ...

    @classmethod
    def for_vram(cls, *, devices: Sequence[int] | None = None) -> Executor:
        """Opens one stream per device for a VRAM-resident index."""
        ...

    @classmethod
    def for_nvme(cls, threads: int = 0, *, paths: Sequence[PathLike] | None = None) -> Executor:
        """Opens one submission ring per worker for an NVMe-resident index."""
        ...

    def drain(self) -> None: ...
    def __repr__(self) -> str: ...

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
    """A ranked run, read back off the three buffers a search filled."""

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
