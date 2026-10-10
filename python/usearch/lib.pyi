"""Type stubs for the `usearch.lib` extension.

Stub files are never executed, so PEP 695 `type` statements and PEP 688's `Buffer` are read
by any checker regardless of the interpreter running the tests. The package floor is 3.12.
"""

import builtins
import os
import uuid
from collections.abc import Buffer, Callable, Generator, Sequence
from enum import IntEnum
from types import GenericAlias, TracebackType
from typing import Any, Final, NoReturn, Self, final, overload, type_check_only

from _typeshed import structseq

# region: Library Probes

__version__: str

def version() -> str:
    """The library's semantic version. Calls `usearch_version_name`."""
    ...

def capabilities_compiled() -> str:
    """The kernel families this build carries. Calls `usearch_capabilities_compiled`."""
    ...

def capabilities_available() -> str:
    """The kernel families this machine can run. Calls `usearch_capabilities_available`."""
    ...

# endregion: Library Probes

# region: Shared Vocabulary

# Every batch verb reads PEP 3118 first and `__array_interface__` second, so NumPy is never
# required - at runtime or here. Frameworks whose stubs predate PEP 688, PyTorch among them,
# support the protocol at runtime but need a `cast` to satisfy a checker.
type _Tensor = Buffer

type _KeyLike = Key | int | str | uuid.UUID | Buffer
type _PathLike = str | bytes | os.PathLike[str] | os.PathLike[bytes]
type _Progress = Callable[[int, int], bool | None]

class Status(IntEnum):
    """What a call finished with, valued as `usearch_status_t`: `Error.status` and `Deferred.status`."""

    SUCCESS = 0
    DEFERRED = 1
    OUT_OF_MEMORY = 2
    SHAPE_MISMATCH = 3
    UNSUPPORTED_LAYOUT = 4
    UNKNOWN_KIND = 5
    READ_ONLY = 6
    IO = 7
    DEVICE = 8
    MALFORMED = 9
    INTERNAL = 10
    OVERFLOW = 11
    UNINITIALIZED = 12
    CONFIG_MISMATCH = 13
    CONTENDED = 14
    CANCELLED = 15
    UNSUPPORTED = 16
    INVALID_ARGUMENT = 17

class Mood(IntEnum):
    """What an executor's idle workers do, valued as `usearch_mood_t`: sleep, or stay hot."""

    PARKED = 0
    SPINNING = 1

class Scalar(IntEnum):
    """A stored or exchanged scalar type, valued as `usearch_scalar_t`, widest dynamic range first."""

    F64 = 1
    F32 = 2
    BF16 = 3
    F16 = 4
    E5M2 = 5
    E4M3 = 6
    E3M2 = 7
    E2M3 = 8
    I8 = 10
    I4X2 = 11
    U8 = 20
    U4X2 = 21
    U1X8 = 22

class Residency(IntEnum):
    """Where a handle's payload lives, valued as `usearch_residency_t`."""

    RAM = 0
    VRAM = 1
    NVME = 2

class Error(RuntimeError):
    """The base of every failure the library reports, one subclass per status shared with the C API."""

    status: Status

class OutOfMemoryError(Error, MemoryError):
    """An allocation or reservation could not be satisfied."""

class ShapeMismatchError(Error, ValueError):
    """Dimensionality, row counts, or output capacity disagree."""

class UnsupportedLayoutError(Error, ValueError):
    """A stride or packing the engine cannot traverse."""

class UnknownKindError(Error, ValueError):
    """A scalar, metric, or residency this build does not know."""

class ReadOnlyError(Error):
    """A mutating verb on a handle that serves its checkpoint in place."""

class InputOutputError(Error, OSError):
    """A file could not be opened, read, written, or parsed."""

class DeviceError(Error):
    """A device could not be selected, or its memory could not be claimed."""

class MalformedError(Error, ValueError):
    """A representation could not be decoded - a key, a buffer, a checkpoint."""

class InternalError(Error):
    """The library reported neither success nor a failure - a defect."""

class CapacityOverflowError(Error, OverflowError):
    """A hard design ceiling was reached; a smaller request cannot help."""

class UninitializedError(Error):
    """The handle is not ready for the verb - unfrozen, consumed by a merge, or closed."""

class ConfigMismatchError(Error, ValueError):
    """The request contradicts an already-frozen configuration."""

class ContendedError(Error):
    """An executor's deferred table, an arena, or a walk is already serving another call."""

class CancelledError(Error):
    """A progress handler stopped the call, which published nothing."""

class UnsupportedError(Error, NotImplementedError):
    """This residency does not implement the verb; pick another residency."""

class InvalidArgumentError(Error, ValueError):
    """An argument breaks the verb's contract."""

@final
class Key:
    """A 128-bit entry identity, rendered as a lowercase dashed UUID."""

    def __new__(cls, value: _KeyLike = 0, /) -> Self: ...
    @property
    def bytes(self) -> bytes:
        """The sixteen identity bytes, most significant first."""
        ...

    def __int__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __lt__(self, other: Key | int | uuid.UUID, /) -> bool: ...
    def __le__(self, other: Key | int | uuid.UUID, /) -> bool: ...
    def __gt__(self, other: Key | int | uuid.UUID, /) -> bool: ...
    def __ge__(self, other: Key | int | uuid.UUID, /) -> bool: ...
    def __reduce__(self) -> tuple[type[Key], tuple[builtins.bytes]]: ...

@final
class Executor:
    """The workers a verb runs on, shared across indexes and named by `on` and `deferred`."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    @classmethod
    def for_ram(
        cls, threads: int = 0, *, memory_domains: Sequence[int] | None = None, deferred_limit: int = 0
    ) -> Executor:
        """Spawns host worker threads for a RAM-resident index. Calls `usearch_executor_for_ram`."""
        ...

    @classmethod
    def on_pool(cls, pool: int, /, *, memory_domains: Sequence[int] | None = None, deferred_limit: int = 0) -> Executor:
        """Drives RAM-resident indexes on a thread pool the caller owns. Calls `usearch_executor_on_pool`."""
        ...

    @classmethod
    def for_vram(cls, *, devices: Sequence[int] | None = None, deferred_limit: int = 0) -> Executor:
        """Opens one stream per device for a VRAM-resident index. Calls `usearch_executor_for_vram`."""
        ...

    @classmethod
    def for_nvme(
        cls, threads: int = 0, *, paths: Sequence[_PathLike] | None = None, deferred_limit: int = 0
    ) -> Executor:
        """Opens one submission ring per worker for an NVMe-resident index. Calls `usearch_executor_for_nvme`."""
        ...

    def drain(self, mood: Mood = ...) -> None:
        """Waits for every call submitted before it, then sets what idle workers do. Calls `usearch_executor_drain`."""
        ...

    @property
    def residency(self) -> Residency:
        """The `Residency` its calls compute in. Python-only."""
        ...

    def __enter__(self) -> Self: ...
    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None, /
    ) -> None:
        """Waits for every call submitted to the executor and closes it. Calls `usearch_executor_free`."""
        ...

    def __repr__(self) -> str: ...

@final
class Arena:
    """The memory a verb lands the outputs only it can size in, reused by the next call handed it."""

    def __init__(self) -> None: ...

# The C base of every result over an arena, which the module does not export by name.
@type_check_only
class _ArenaBuffer:
    """One output a call left in an `Arena`, keeping it alive and blocking it from reuse."""

    def __buffer__(self, flags: int, /) -> memoryview:
        """Host memory only; device memory raises `BufferError`. Python-only."""
        ...

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        """Version 3 of the CUDA Array Interface, present only over device memory. Python-only."""
        ...

@final
class Vectors(_ArenaBuffer):
    """The rows `get` exported into an `Arena`: a (count, dimensions) matrix, or (count, row_bytes) of 'B'."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...

@final
class Products:
    """The ragged sparse product `multiply` exported: per left row, every pair its walk scored."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    @property
    def offsets(self) -> memoryview:
        """(rows + 1,) Apache Arrow offsets, in host memory."""
        ...

    @property
    def keys(self) -> _ArenaBuffer:
        """(pairs, 16) identities of the indexed rows."""
        ...

    @property
    def scores(self) -> _ArenaBuffer:
        """(pairs,) float64 products."""
        ...

    @property
    def required(self) -> int:
        """Pairs scored before the bound cut any."""
        ...

    def row(self, row: int, /) -> tuple[_ArenaBuffer, _ArenaBuffer]:
        """The (keys, scores) of left row `row`, viewing the arena. Python-only."""
        ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class Monolith(_ArenaBuffer):
    """The bytes `save_buffer` exported into an `Arena`, one (length,) run of 'B'."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...

@final
class Match(structseq[Any], tuple[Key, float]):
    """One scored entry: the identity found and its distance, unpacking as `key, distance = match`."""

    __match_args__: Final = ("key", "distance")

    @property
    def key(self) -> Key:
        """The identity of the matched entry."""
        ...

    @property
    def distance(self) -> float:
        """The distance to the query, in the index's own metric."""
        ...

@final
class Matches:
    """A ranked run, as `search` and `maxsim` return it, reading through to three buffers."""

    def __new__(cls, keys: Buffer, distances: Buffer, counts: Buffer, /) -> Self: ...
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

@final
class Deferred[AnswerT]:
    """A call a `DeferredOn` submitted: `await` it, or `wait()` for it; dropping it waits."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        """Names what `wait()` answers, as `Deferred[Matches]`."""
        ...

    @property
    def status(self) -> Status:
        """`Status.DEFERRED` until the call is final, then the final one. Python-only."""
        ...

    def wait(self) -> AnswerT:
        """Blocks until the call is final, then answers or raises. Calls `usearch_outcome_wait`."""
        ...

    def __await__(self) -> Generator[Any, None, AnswerT]:
        """Runs `wait()` on the running event loop's default executor, suspending the task meanwhile."""
        ...

    def __repr__(self) -> str: ...

@final
class On[IndexT]:
    """An index bound to the executor its verbs block on, as `index.on(executor)` answers it."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        """Names the index the receiver addresses, as `On[DenseIndex]`."""
        ...

    def describe(self: On[DenseIndex]) -> DenseIndexDescription: ...
    def reserve(self: On[DenseIndex], capacity: int, /) -> None: ...
    def clear(self: On[DenseIndex]) -> None: ...
    def insert(
        self: On[DenseIndex],
        keys: Sequence[_KeyLike] | _Tensor,
        vectors: _Tensor,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
    ) -> None: ...
    def refine(
        self: On[DenseIndex],
        *,
        expansion_factor: int = 0,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
    ) -> int: ...
    def health(self: On[DenseIndex]) -> DenseReachability: ...
    def search(
        self: On[DenseIndex],
        queries: _Tensor,
        wanted: int,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        predicate: Callable[[Key], bool] | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        out: Matches | None = None,
    ) -> Matches: ...
    def count(self: On[DenseIndex], key: _KeyLike, /) -> int: ...
    def get(
        self: On[DenseIndex], key: _KeyLike, /, *, outputs_dtype: Scalar = ..., arena: Arena | None = None
    ) -> Vectors | None: ...
    def multiply(
        self: On[DenseIndex],
        rows: _Tensor,
        wanted: int,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        arena: Arena | None = None,
    ) -> Products: ...
    def attend[OutputT: _Tensor](
        self: On[DenseIndex],
        queries: _Tensor,
        value_lookup: Callable[[memoryview], _Tensor | None],
        /,
        *,
        head_count: int = 1,
        expansion_factor: int = 0,
        score_scale: float = 0.0,
        dtype: Scalar | None = None,
        outputs_dtype: Scalar | None = None,
        carry: _Tensor | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        out: OutputT | None = None,
    ) -> memoryview | OutputT: ...
    def maxsim(
        self: On[DenseIndex],
        queries: _Tensor,
        rows_per_query: int,
        wanted: int,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        out: Matches | None = None,
    ) -> Matches: ...
    def merge(self: On[DenseIndex], donor: DenseIndex, /, *, progress: _Progress | None = None) -> None: ...
    def save(self: On[DenseIndex], path: _PathLike, /, *, progress: _Progress | None = None) -> None: ...
    def save_buffer(
        self: On[DenseIndex], *, progress: _Progress | None = None, arena: Arena | None = None
    ) -> Monolith: ...
    @overload
    def cursor(
        self: On[DenseIndex], size: int = 1024, *, outputs_dtype: None = None, columns: None = None
    ) -> DenseCursor[memoryview]:
        """Calls `usearch_dense_cursor_init`."""
        ...

    @overload
    def cursor(
        self: On[DenseIndex], size: int = 1024, *, outputs_dtype: Scalar, columns: int
    ) -> DenseCursor[tuple[memoryview, memoryview]]:
        """Calls `usearch_dense_cursor_init`."""
        ...

    def search_stream(
        self: On[DenseIndex], query: _Tensor, /, *, expansion_factor: int = 0, dtype: Scalar | None = None
    ) -> DenseSearchStream: ...
    def __repr__(self) -> str: ...

@final
class DeferredOn[IndexT]:
    """An index bound to the executor its verbs submit to, each answering a `Deferred` at once."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        """Names the index the receiver addresses, as `DeferredOn[DenseIndex]`."""
        ...

    def describe(self: DeferredOn[DenseIndex]) -> Deferred[DenseIndexDescription]: ...
    def reserve(self: DeferredOn[DenseIndex], capacity: int, /) -> Deferred[None]: ...
    def clear(self: DeferredOn[DenseIndex]) -> Deferred[None]: ...
    def insert(
        self: DeferredOn[DenseIndex],
        keys: Sequence[_KeyLike] | _Tensor,
        vectors: _Tensor,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
    ) -> Deferred[None]: ...
    def refine(
        self: DeferredOn[DenseIndex],
        *,
        expansion_factor: int = 0,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
    ) -> Deferred[int]: ...
    def health(self: DeferredOn[DenseIndex]) -> Deferred[DenseReachability]: ...
    def search(
        self: DeferredOn[DenseIndex],
        queries: _Tensor,
        wanted: int,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        predicate: Callable[[Key], bool] | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        out: Matches | None = None,
    ) -> Deferred[Matches]: ...
    def count(self: DeferredOn[DenseIndex], key: _KeyLike, /) -> Deferred[int]: ...
    def get(
        self: DeferredOn[DenseIndex], key: _KeyLike, /, *, outputs_dtype: Scalar = ..., arena: Arena | None = None
    ) -> Deferred[Vectors | None]: ...
    def multiply(
        self: DeferredOn[DenseIndex],
        rows: _Tensor,
        wanted: int,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        arena: Arena | None = None,
    ) -> Deferred[Products]: ...
    def attend[OutputT: _Tensor](
        self: DeferredOn[DenseIndex],
        queries: _Tensor,
        value_lookup: Callable[[memoryview], _Tensor | None],
        /,
        *,
        head_count: int = 1,
        expansion_factor: int = 0,
        score_scale: float = 0.0,
        dtype: Scalar | None = None,
        outputs_dtype: Scalar | None = None,
        carry: _Tensor | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        out: OutputT | None = None,
    ) -> Deferred[memoryview | OutputT]: ...
    def maxsim(
        self: DeferredOn[DenseIndex],
        queries: _Tensor,
        rows_per_query: int,
        wanted: int,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        out: Matches | None = None,
    ) -> Deferred[Matches]: ...
    def merge(
        self: DeferredOn[DenseIndex], donor: DenseIndex, /, *, progress: _Progress | None = None
    ) -> Deferred[None]: ...
    def save(
        self: DeferredOn[DenseIndex], path: _PathLike, /, *, progress: _Progress | None = None
    ) -> Deferred[None]: ...
    def save_buffer(
        self: DeferredOn[DenseIndex], *, progress: _Progress | None = None, arena: Arena | None = None
    ) -> Deferred[Monolith]: ...
    def __repr__(self) -> str: ...

# endregion: Shared Vocabulary

# region: Dense Index

class DenseMetric(IntEnum):
    """How a dense index scores two rows, valued as `usearch_dense_metric_t`."""

    INNER = 1
    ANGULAR = 2
    EUCLIDEAN = 3
    HAMMING = 4
    JACCARD = 5

@final
class DenseMetadata(structseq[Any], tuple[int, DenseMetric, Scalar, int]):
    """The geometry and metric a checkpoint froze, as `usearch_dense_metadata_t` holds them."""

    __match_args__: Final = ("dimensions", "metric", "quantization", "connectivity")

    @property
    def dimensions(self) -> int:
        """Scalars per stored vector."""
        ...

    @property
    def metric(self) -> DenseMetric:
        """The `DenseMetric` the checkpoint scores with."""
        ...

    @property
    def quantization(self) -> Scalar:
        """The `Scalar` its rows are stored and quantized into."""
        ...

    @property
    def connectivity(self) -> int:
        """Connections per node the graph was built with."""
        ...

@final
class DenseIndexDescription(structseq[Any], tuple[DenseMetadata, int, int, int, str]):
    """A live index's frozen metadata and current counters, as `usearch_dense_index_description_t` holds them."""

    __match_args__: Final = ("metadata", "size", "capacity", "memory_usage", "capabilities")

    @property
    def metadata(self) -> DenseMetadata:
        """The frozen `DenseMetadata`, as a checkpoint reports it."""
        ...

    @property
    def size(self) -> int:
        """Stored entries."""
        ...

    @property
    def capacity(self) -> int:
        """Entries hosted without growing again."""
        ...

    @property
    def memory_usage(self) -> int:
        """Bytes the handle holds; traversal scratch is the executor's."""
        ...

    @property
    def capabilities(self) -> str:
        """The codename of the kernels the index dispatched to on this machine."""
        ...

@final
class DenseReachability(structseq[Any], tuple[int, int, int]):
    """One reachability audit, as `usearch_dense_reachability_t` holds it."""

    __match_args__: Final = ("audited", "reachable", "sweeps")

    @property
    def audited(self) -> int:
        """Entries published when the audit started, the population it judged."""
        ...

    @property
    def reachable(self) -> int:
        """Of those, the entries a search reaches; the rest wait for a `refine` to re-link them."""
        ...

    @property
    def sweeps(self) -> int:
        """Levels the sweep took to converge, a diameter proxy."""
        ...

@final
class DenseTraversalStats:
    """Traversal telemetry, fully overwritten by every call it is passed to; `+=` sums a sweep."""

    def __init__(self) -> None: ...
    @property
    def visited_members(self) -> int: ...
    @property
    def computed_distances(self) -> int: ...
    @property
    def restarts(self) -> int: ...
    def __iadd__(self, other: DenseTraversalStats, /) -> Self:
        """Python-only."""
        ...

    def __repr__(self) -> str: ...

@final
class DenseCursor[StepT]:
    """A sequential row scan over the entries an index holds, in batches."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> StepT:
        """Each step calls `usearch_dense_cursor_next`."""
        ...

    def close(self) -> None:
        """Releases the scan before the cursor is collected. Calls `usearch_dense_cursor_free`."""
        ...

    def __enter__(self) -> Self: ...
    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None, /
    ) -> None: ...

@final
class DenseSearchStream:
    """A resumable best-first stream of matches for one query."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Match:
        """Each refill calls `usearch_dense_search_stream_next`."""
        ...

    def close(self) -> None:
        """Releases the walk before the stream is collected. Calls `usearch_dense_search_stream_free`."""
        ...

    def __enter__(self) -> Self: ...
    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None, /
    ) -> None: ...

@final
class DenseIndex:
    """Vector search over equidimensional dense embeddings; its verbs run on the calling thread."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    @classmethod
    def init_ram(
        cls,
        dimensions: int,
        metric: DenseMetric = ...,
        quantization: Scalar = ...,
        *,
        connectivity: int = 0,
        values_per_key: int = 0,
        memory_domains: Sequence[int] | None = None,
    ) -> DenseIndex:
        """Starts an empty, mutable graph in host memory. Calls `usearch_dense_init_ram`."""
        ...

    @classmethod
    def init_vram(
        cls,
        dimensions: int,
        metric: DenseMetric = ...,
        quantization: Scalar = ...,
        *,
        connectivity: int = 0,
        values_per_key: int = 0,
        devices: Sequence[int] | None = None,
    ) -> DenseIndex:
        """Starts an empty, mutable graph on CUDA devices. Calls `usearch_dense_init_vram`."""
        ...

    @classmethod
    def load_ram(
        cls,
        path: _PathLike,
        /,
        *,
        memory_domains: Sequence[int] | None = None,
        executor: Executor | None = None,
        values_per_key: int = 0,
        progress: _Progress | None = None,
    ) -> DenseIndex:
        """Copies a checkpoint into a new, mutable index in host memory. Calls `usearch_dense_load_ram`."""
        ...

    @classmethod
    def load_ram_buffer(
        cls,
        buffer: Buffer,
        /,
        *,
        memory_domains: Sequence[int] | None = None,
        executor: Executor | None = None,
        values_per_key: int = 0,
        progress: _Progress | None = None,
    ) -> DenseIndex:
        """Copies a monolith out of a buffer into host memory. Calls `usearch_dense_load_ram_buffer`."""
        ...

    @classmethod
    def load_vram(
        cls,
        path: _PathLike,
        /,
        *,
        devices: Sequence[int] | None = None,
        executor: Executor | None = None,
        values_per_key: int = 0,
        progress: _Progress | None = None,
    ) -> DenseIndex:
        """Copies a checkpoint into a new, mutable index on CUDA devices. Calls `usearch_dense_load_vram`."""
        ...

    @classmethod
    def load_vram_buffer(
        cls,
        buffer: Buffer,
        /,
        *,
        devices: Sequence[int] | None = None,
        executor: Executor | None = None,
        values_per_key: int = 0,
        progress: _Progress | None = None,
    ) -> DenseIndex:
        """Copies a monolith out of a buffer onto CUDA devices. Calls `usearch_dense_load_vram_buffer`."""
        ...

    @classmethod
    def view_ram(cls, path: _PathLike, /) -> DenseIndex:
        """Serves a checkpoint in place through a memory mapping, read-only. Calls `usearch_dense_view_ram`."""
        ...

    @classmethod
    def view_nvme(
        cls, path: _PathLike, /, *, paths: Sequence[_PathLike] | None = None, executor: Executor | None = None
    ) -> DenseIndex:
        """Serves a checkpoint in place from NVMe drives, read-only. Calls `usearch_dense_view_nvme`."""
        ...

    @staticmethod
    def metadata_read(path: _PathLike, /, *, executor: Executor | None = None) -> DenseMetadata:
        """Reads a checkpoint's geometry without loading it. Calls `usearch_dense_metadata_read`."""
        ...

    @staticmethod
    def metadata_read_buffer(buffer: Buffer, /, *, executor: Executor | None = None) -> DenseMetadata:
        """Reads a monolith's geometry. Calls `usearch_dense_metadata_read_buffer`."""
        ...

    def on(self, executor: Executor, /) -> On[DenseIndex]:
        """A receiver whose verbs block on `executor`. Python-only."""
        ...

    def deferred(self, executor: Executor, /) -> DeferredOn[DenseIndex]:
        """A receiver whose verbs submit to `executor`, each answering a `Deferred`. Python-only."""
        ...

    def describe(self) -> DenseIndexDescription:
        """The frozen metadata and current counters. Calls `usearch_dense_describe`."""
        ...

    def reserve(self, capacity: int, /) -> None:
        """Grows the index to host at least `capacity` entries. Calls `usearch_dense_reserve`."""
        ...

    def clear(self) -> None:
        """Drops every entry, keeping the capacity and geometry. Calls `usearch_dense_clear`."""
        ...

    def insert(
        self,
        keys: Sequence[_KeyLike] | _Tensor,
        vectors: _Tensor,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
    ) -> None:
        """Inserts a batch of vectors under their keys, all or nothing. Calls `usearch_dense_insert`."""
        ...

    def refine(
        self, *, expansion_factor: int = 0, progress: _Progress | None = None, stats: DenseTraversalStats | None = None
    ) -> int:
        """Re-selects every entry's neighbors, answering how many changed. Calls `usearch_dense_refine`."""
        ...

    def health(self) -> DenseReachability:
        """Audits how much of the graph a search reaches. Calls `usearch_dense_health`."""
        ...

    def search(
        self,
        queries: _Tensor,
        wanted: int,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        predicate: Callable[[Key], bool] | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        out: Matches | None = None,
    ) -> Matches:
        """Finds the `wanted` nearest entries for every query row. Calls `usearch_dense_search`."""
        ...

    def count(self, key: _KeyLike, /) -> int:
        """How many entries share `key`. Calls `usearch_dense_count`."""
        ...

    def get(self, key: _KeyLike, /, *, outputs_dtype: Scalar = ..., arena: Arena | None = None) -> Vectors | None:
        """Every vector stored under `key`, or None. Calls `usearch_dense_get`."""
        ...

    def multiply(
        self,
        rows: _Tensor,
        wanted: int,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        arena: Arena | None = None,
    ) -> Products:
        """Every pair the walk scored, an approximate sparse product. Calls `usearch_dense_multiply`."""
        ...

    def attend[OutputT: _Tensor](
        self,
        queries: _Tensor,
        value_lookup: Callable[[memoryview], _Tensor | None],
        /,
        *,
        head_count: int = 1,
        expansion_factor: int = 0,
        score_scale: float = 0.0,
        dtype: Scalar | None = None,
        outputs_dtype: Scalar | None = None,
        carry: _Tensor | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        out: OutputT | None = None,
    ) -> memoryview | OutputT:
        """Softmax attention folded into the walk. Calls `usearch_dense_attend`."""
        ...

    def maxsim(
        self,
        queries: _Tensor,
        rows_per_query: int,
        wanted: int,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        stats: DenseTraversalStats | None = None,
        out: Matches | None = None,
    ) -> Matches:
        """Late-interaction scoring of multi-row queries. Calls `usearch_dense_maxsim`."""
        ...

    def merge(self, donor: DenseIndex, /, *, progress: _Progress | None = None) -> None:
        """Absorbs and consumes `donor`. Calls `usearch_dense_merge`."""
        ...

    def save(self, path: _PathLike, /, *, progress: _Progress | None = None) -> None:
        """Writes a checkpoint in the layout the extension picks. Calls `usearch_dense_save`."""
        ...

    def save_buffer(self, *, progress: _Progress | None = None, arena: Arena | None = None) -> Monolith:
        """Serializes one monolith into an arena. Calls `usearch_dense_save_buffer`."""
        ...

    @overload
    def cursor(self, size: int = 1024, *, outputs_dtype: None = None, columns: None = None) -> DenseCursor[memoryview]:
        """Calls `usearch_dense_cursor_init`."""
        ...

    @overload
    def cursor(
        self, size: int = 1024, *, outputs_dtype: Scalar, columns: int
    ) -> DenseCursor[tuple[memoryview, memoryview]]:
        """Calls `usearch_dense_cursor_init`."""
        ...

    def search_stream(
        self, query: _Tensor, /, *, expansion_factor: int = 0, dtype: Scalar | None = None
    ) -> DenseSearchStream:
        """A resumable stream of matches for one query. Calls `usearch_dense_search_stream_init`."""
        ...

    @property
    def capacity(self) -> int:
        """Entries hosted without growing again. Calls `usearch_dense_describe`."""
        ...

    @property
    def dimensions(self) -> int:
        """Scalars per vector. Calls `usearch_dense_describe`."""
        ...

    @property
    def connectivity(self) -> int:
        """Connections per node. Calls `usearch_dense_describe`."""
        ...

    @property
    def memory_usage(self) -> int:
        """Bytes the handle holds. Calls `usearch_dense_describe`."""
        ...

    @property
    def capabilities(self) -> str:
        """The kernel family dispatch picked. Calls `usearch_dense_describe`."""
        ...

    def __len__(self) -> int:
        """Stored entries. Python-only; calls `usearch_dense_describe`."""
        ...

    def __contains__(self, key: _KeyLike, /) -> bool:
        """Whether any entry holds `key`. Python-only; calls `usearch_dense_count`."""
        ...

    def __getitem__(self, key: _KeyLike, /) -> Vectors:
        """`get` that raises `KeyError` for an absent key. Python-only; calls `usearch_dense_get`."""
        ...

    def __repr__(self) -> str: ...

# endregion: Dense Index

# region: Other Modalities

@final
class SequenceIndex:
    """Similarity search over variable-length sequences."""

    def __init__(self) -> NoReturn: ...

@final
class MatrixIndex:
    """Similarity search over rank-2 tensors."""

    def __init__(self) -> NoReturn: ...

@final
class GeospatialIndex:
    """Similarity search over space and time."""

    def __init__(self) -> NoReturn: ...

# endregion: Other Modalities
