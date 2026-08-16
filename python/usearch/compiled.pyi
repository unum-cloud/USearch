"""Type stubs for the `usearch.compiled` extension.

Stub files are never executed, so PEP 695 `type` statements and PEP 688's `Buffer` are read
by any checker regardless of the interpreter running the tests. The package floor is 3.12.
"""

import os
import uuid
from collections.abc import Buffer, Callable, Sequence
from typing import Literal, TypedDict

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

type MetricName = Literal["inner", "angular", "euclidean", "hamming", "jaccard"]
type ResidencyName = Literal["ram", "vram", "nvme"]
type PageName = Literal["base", "huge"]
type AddressingName = Literal["device-only", "unified"]
type IoName = Literal["direct", "page-cached"]

type KeyLike = Key | int | str | uuid.UUID | Buffer
type PathLike = str | bytes | os.PathLike[str] | os.PathLike[bytes]

__version__: str

def version() -> str:
    """The library's semantic version."""
    ...

def capabilities_compiled() -> str:
    """The kernel families this build carries, whatever the machine offers."""
    ...

def capabilities_available() -> str:
    """The kernel families this machine can run, and dispatch picks from."""
    ...

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

class Stats:
    """Traversal telemetry, accumulated across the calls it is passed to."""

    def __init__(self) -> None: ...
    @property
    def visited_members(self) -> int: ...
    @property
    def computed_distances(self) -> int: ...
    @property
    def restarts(self) -> int: ...
    def __repr__(self) -> str: ...

class Match:
    """One scored entry: the identity found and its distance."""

    @property
    def key(self) -> Key: ...
    @property
    def distance(self) -> float: ...
    def __repr__(self) -> str: ...

class Matches:
    """The result of a batch search: three matrices and the telemetry that produced them."""

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
        """(queries,) match counts; entries past a count are zeroed."""
        ...

    @property
    def visited_members(self) -> int: ...
    @property
    def computed_distances(self) -> int: ...
    @property
    def restarts(self) -> int: ...
    def __len__(self) -> int: ...
    def __getitem__(self, query_index: int, /) -> tuple[Match, ...]: ...
    def __repr__(self) -> str: ...

# endregion: Shared Vocabulary

# region: Dense Index

class DenseMetadata(TypedDict):
    """The constructor keywords a checkpoint fixes; residency is never among them."""

    dimensions: int
    metric: MetricName
    quantization: ScalarKind
    connectivity: int
    pad_dimensions: bool

class DenseCursor:
    """A snapshot walk over the entries an index held when the cursor opened."""

    def __iter__(self) -> DenseCursor: ...
    def __next__(self) -> memoryview | tuple[memoryview, memoryview]: ...

class DenseFinder:
    """A resumable best-first stream of matches for one query."""

    def __iter__(self) -> DenseFinder: ...
    def __next__(self) -> Match: ...

class DenseIndex:
    """Vector search over equidimensional dense embeddings."""

    def __init__(
        self,
        dimensions: int = 0,
        metric: MetricName = "angular",
        quantization: ScalarName = "bf16",
        *,
        connectivity: int = 0,
        pad_dimensions: bool = False,
        residency: ResidencyName = "ram",
        threads: int = 0,
        pages: PageName = "base",
        device: int | None = None,
        addressing: AddressingName = "device-only",
        sector_bytes: int = 0,
        queue_depth: int = 0,
        io: IoName = "direct",
    ) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def capacity(self) -> int: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def connectivity(self) -> int: ...
    @property
    def memory_usage(self) -> int: ...
    @property
    def serialized_length(self) -> int: ...
    @property
    def capabilities(self) -> str: ...
    def reserve(self, capacity: int, /) -> None: ...
    def clear(self) -> None: ...
    def add(
        self,
        keys: Sequence[KeyLike] | Tensor,
        vectors: Tensor,
        /,
        *,
        expansion_factor: int = 0,
        dtype: ScalarName | None = None,
        stats: Stats | None = None,
    ) -> int: ...
    def search(
        self,
        queries: Tensor,
        wanted: int,
        /,
        *,
        expansion_factor: int = 0,
        dtype: ScalarName | None = None,
        predicate: Callable[[Key], bool] | None = None,
        stats: Stats | None = None,
    ) -> Matches: ...
    def search_into(
        self,
        queries: Tensor,
        wanted: int,
        keys: Tensor,
        distances: Tensor,
        counts: Tensor,
        /,
        *,
        expansion_factor: int = 0,
        dtype: ScalarName | None = None,
        predicate: Callable[[Key], bool] | None = None,
        stats: Stats | None = None,
    ) -> None: ...
    def multiply_into(
        self,
        rows: Tensor,
        wanted: int,
        columns: Tensor,
        products: Tensor,
        counts: Tensor,
        /,
        *,
        expansion_factor: int = 0,
        dtype: ScalarName | None = None,
        stats: Stats | None = None,
    ) -> int: ...
    def attend_into(
        self,
        queries: Tensor,
        values: Tensor,
        outputs: Tensor,
        /,
        *,
        expansion_factor: int = 0,
        score_scale: float = 0.0,
        mass_tolerance: float = 0.0,
        dtype: ScalarName | None = None,
        values_dtype: ScalarName | None = None,
        outputs_dtype: ScalarName | None = None,
        maximum_scores: Tensor | None = None,
        partitions: Tensor | None = None,
        stats: Stats | None = None,
    ) -> None: ...
    def count(self, key: KeyLike, /) -> int: ...
    def get(self, key: KeyLike, /, *, dtype: ScalarName = "f32") -> memoryview | None: ...
    def merge(self, donor: DenseIndex, /) -> None: ...
    def find(self, query: Tensor, /, *, dtype: ScalarName | None = None) -> DenseFinder: ...
    def batches(self, size: int = 1024, /, *, dtype: ScalarName | None = None) -> DenseCursor: ...
    def save(self, target: PathLike | Tensor, /) -> None: ...
    def load(self, source: PathLike | Tensor, /) -> None: ...
    @staticmethod
    def metadata(source: PathLike | Tensor, /) -> DenseMetadata: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: KeyLike) -> bool: ...
    def __getitem__(self, key: KeyLike) -> memoryview: ...
    def __iter__(self) -> DenseCursor: ...
    def __repr__(self) -> str: ...

# endregion: Dense Index

# region: Other Modalities

class SequenceIndex:
    """Similarity search over variable-length sequences; not yet instantiated in this build."""

    def __init__(self, *args: object, **keywords: object) -> None: ...

class MatrixIndex:
    """Similarity search over rank-2 tensors; not yet instantiated in this build."""

    def __init__(self, *args: object, **keywords: object) -> None: ...

class GeospatialIndex:
    """Similarity search over space and time; not yet instantiated in this build."""

    def __init__(self, *args: object, **keywords: object) -> None: ...

# endregion: Other Modalities
