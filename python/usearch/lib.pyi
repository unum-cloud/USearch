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
from typing import Any, Final, NoReturn, Self, final, type_check_only

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

def layouts() -> dict[str, tuple[int, dict[str, int]]]:
    """The size and field offsets of every argument struct the other bindings lay out by hand."""
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

    PARKED = 1
    SPINNING = 2

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
    I32 = 12
    U8 = 20
    U4X2 = 21
    U1X8 = 22

class Residency(IntEnum):
    """Where a handle's payload lives, valued as `usearch_residency_t`."""

    RAM = 1
    VRAM = 2
    NVME = 3

class Traversal(IntEnum):
    """How a search reaches the rows it scores, valued as `usearch_traversal_t`."""

    WALK = 1
    SCAN = 2

class CPUClass(IntEnum):
    """The CPU families dispatch picks from, valued as `usearch_cpu_class_t`; a placement may pin one."""

    SERIAL = 1
    INTEL_HASWELL = 2
    INTEL_SIERRA = 3
    INTEL_SKYLAKE = 4
    INTEL_ICELAKE = 5
    AMD_GENOA = 6
    INTEL_SAPPHIRE = 7
    INTEL_GRANITE = 8
    INTEL_DIAMOND = 9
    AWS_G2 = 10
    APPLE_M2 = 11
    APPLE_M4 = 12
    AWS_G5 = 13
    NVIDIA_VERA = 14
    ALIBABA_C930 = 15
    WASM32_V128 = 16
    WASM64_EXTENDED = 17

class GPUClass(IntEnum):
    """The GPU families dispatch picks from, valued as `usearch_gpu_class_t`; a placement may pin one."""

    NVIDIA_AMPERE = 1
    NVIDIA_ADA = 2
    NVIDIA_HOPPER = 3
    NVIDIA_BLACKWELL = 4
    NVIDIA_BLACKWELLRTX = 5
    NVIDIA_RUBIN = 6
    AMD_CDNA3 = 7
    AMD_CDNA4 = 8
    AMD_CDNA5 = 9
    APPLE_METAL7 = 10
    APPLE_METAL9 = 11
    APPLE_METAL10 = 12

class NVMeClass(IntEnum):
    """The I/O engines NVMe executors submit through, valued as `usearch_nvme_class_t`; an executor may pin one."""

    POSIX = 1
    LINUX_AIO = 2
    LINUX_URING = 3
    LINUX_URING_PASSTHROUGH = 4
    LINUX_SPDK = 5
    WINDOWS_IORING = 6

class Error(RuntimeError):
    """The base of every failure the library reports, one subclass per status shared with the C API."""

    status: Status

class OutOfMemoryError(Error, MemoryError):
    """An allocation or reservation could not be satisfied."""

class ShapeMismatchError(Error, ValueError):
    """Dimensionality or row counts disagree."""

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
    """The handle is not ready for the verb - absorbed as a donor, or closed."""

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
        cls,
        threads: int = 0,
        *,
        paths: Sequence[_PathLike] | None = None,
        deferred_limit: int = 0,
        nvme_class: NVMeClass | None = None,
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
        """Closes it once its calls finish, or leaves that to its last live GPU arena. Calls `usearch_executor_free`."""
        ...

    def __repr__(self) -> str: ...

@final
class Arena:
    """The memory every output whose length scales with a call lands in, rewound by each call handed it."""

    def __new__(cls, executor: Executor | None = None, /) -> Self:
        """Makes one where `executor` computes, or host pages for the calling thread. Calls `usearch_arena_init`."""
        ...

    @property
    def residency(self) -> Residency:
        """The `Residency` its outputs live in. Python-only."""
        ...

# The C base of every result over an arena, which the module does not export by name.
@type_check_only
class _ArenaBuffer:
    """One output a call left in an `Arena`, readable until a later call rewinds it."""

    def __buffer__(self, flags: int, /) -> memoryview:
        """Host memory only; device memory raises `DeviceError`, a rewound arena `UninitializedError`. Python-only."""
        ...

    def __len__(self) -> int:
        """Rows, or bytes of a `Monolith`. Python-only."""
        ...

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        """Version 3 of the CUDA Array Interface, present only over device memory. Python-only."""
        ...

@final
class DenseVectors(_ArenaBuffer):
    """The rows `get` or `attend` wrote into an `Arena`: (count, dimensions), or (count, row_bytes) of 'B'."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...

@final
class DenseProducts:
    """The ragged sparse product `multiply` wrote into an `Arena`: per left row, every pair its walk scored.

    `maxsim` lands its documents alike, one run per query, largest first, `required` counting those it scored.
    """

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    @property
    def offsets(self) -> _ArenaBuffer:
        """(rows + 1,) Apache Arrow offsets."""
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
    """The bytes `save_buffer` wrote into an `Arena`, one (length,) run of 'B'."""

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
    """A ranked run `search` wrote into an `Arena`, reading through to three outputs."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    @property
    def keys(self) -> _ArenaBuffer:
        """(queries, columns, 16) identities, most significant byte first."""
        ...

    @property
    def distances(self) -> _ArenaBuffer:
        """(queries, columns) distances in the index's own metric, columns at most the call's wanted."""
        ...

    @property
    def counts(self) -> _ArenaBuffer:
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
        """`Status.DEFERRED` until the call is final, then the final one. Calls `usearch_outcome_status`."""
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
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> None: ...
    def refine(
        self: On[DenseIndex],
        *,
        expansion_factor: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> int: ...
    def health(self: On[DenseIndex]) -> DenseReachability: ...
    def search(
        self: On[DenseIndex],
        queries: _Tensor,
        wanted: int,
        arena: Arena,
        /,
        *,
        expansion_factor: int = 0,
        traversal: Traversal = ...,
        dtype: Scalar | None = None,
        predicate: Callable[[Key], bool] | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Matches: ...
    def count(self: On[DenseIndex], key: _KeyLike, /) -> int: ...
    def get(self: On[DenseIndex], key: _KeyLike, arena: Arena, /, *, outputs_dtype: Scalar = ...) -> DenseVectors: ...
    def multiply(
        self: On[DenseIndex],
        rows: _Tensor,
        wanted: int,
        arena: Arena,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> DenseProducts: ...
    def attend(
        self: On[DenseIndex],
        queries: _Tensor,
        values: _Tensor,
        row_of: Callable[[Key], int | None],
        arena: Arena,
        /,
        *,
        head_count: int = 1,
        expansion_factor: int = 0,
        score_scale: float = 0.0,
        dtype: Scalar | None = None,
        values_dtype: Scalar | None = None,
        outputs_dtype: Scalar | None = None,
        carry: _Tensor | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> DenseVectors: ...
    def maxsim(
        self: On[DenseIndex],
        queries: _Tensor,
        rows_per_query: int,
        wanted: int,
        arena: Arena,
        /,
        *,
        expansion_factor: int = 0,
        traversal: Traversal = ...,
        dtype: Scalar | None = None,
        predicate: Callable[[Key], bool] | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> DenseProducts: ...
    def absorb(
        self: On[DenseIndex], *donors: DenseIndex, progress: _Progress | None = None, progress_steps: int = 0
    ) -> None: ...
    def save(
        self: On[DenseIndex], path: _PathLike, /, *, progress: _Progress | None = None, progress_steps: int = 0
    ) -> None: ...
    def save_buffer(
        self: On[DenseIndex], arena: Arena, /, *, progress: _Progress | None = None, progress_steps: int = 0
    ) -> Monolith: ...
    def cursor(self: On[DenseIndex], starts: Sequence[_KeyLike] | _Tensor | None = None) -> DenseCursor:
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
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Deferred[None]: ...
    def refine(
        self: DeferredOn[DenseIndex],
        *,
        expansion_factor: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Deferred[int]: ...
    def health(self: DeferredOn[DenseIndex]) -> Deferred[DenseReachability]: ...
    def search(
        self: DeferredOn[DenseIndex],
        queries: _Tensor,
        wanted: int,
        arena: Arena,
        /,
        *,
        expansion_factor: int = 0,
        traversal: Traversal = ...,
        dtype: Scalar | None = None,
        predicate: Callable[[Key], bool] | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Deferred[Matches]: ...
    def count(self: DeferredOn[DenseIndex], key: _KeyLike, /) -> Deferred[int]: ...
    def get(
        self: DeferredOn[DenseIndex], key: _KeyLike, arena: Arena, /, *, outputs_dtype: Scalar = ...
    ) -> Deferred[DenseVectors]: ...
    def multiply(
        self: DeferredOn[DenseIndex],
        rows: _Tensor,
        wanted: int,
        arena: Arena,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Deferred[DenseProducts]: ...
    def attend(
        self: DeferredOn[DenseIndex],
        queries: _Tensor,
        values: _Tensor,
        row_of: Callable[[Key], int | None],
        arena: Arena,
        /,
        *,
        head_count: int = 1,
        expansion_factor: int = 0,
        score_scale: float = 0.0,
        dtype: Scalar | None = None,
        values_dtype: Scalar | None = None,
        outputs_dtype: Scalar | None = None,
        carry: _Tensor | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Deferred[DenseVectors]: ...
    def maxsim(
        self: DeferredOn[DenseIndex],
        queries: _Tensor,
        rows_per_query: int,
        wanted: int,
        arena: Arena,
        /,
        *,
        expansion_factor: int = 0,
        traversal: Traversal = ...,
        dtype: Scalar | None = None,
        predicate: Callable[[Key], bool] | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Deferred[DenseProducts]: ...
    def absorb(
        self: DeferredOn[DenseIndex], *donors: DenseIndex, progress: _Progress | None = None, progress_steps: int = 0
    ) -> Deferred[None]: ...
    def save(
        self: DeferredOn[DenseIndex], path: _PathLike, /, *, progress: _Progress | None = None, progress_steps: int = 0
    ) -> Deferred[None]: ...
    def save_buffer(
        self: DeferredOn[DenseIndex], arena: Arena, /, *, progress: _Progress | None = None, progress_steps: int = 0
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
class DenseIndexDescription(structseq[Any], tuple[DenseMetadata, int, int, int, CPUClass | None, GPUClass | None, int]):
    """A live index's frozen metadata and current counters, as `usearch_dense_index_description_t` holds them."""

    __match_args__: Final = (
        "metadata",
        "size",
        "capacity",
        "memory_usage",
        "cpu_class",
        "gpu_class",
        "shards_count",
    )

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
    def cpu_class(self) -> CPUClass | None:
        """The `CPUClass` dispatch picked on this machine, or None."""
        ...

    @property
    def gpu_class(self) -> GPUClass | None:
        """The `GPUClass` dispatch picked on the index's devices, or None for a host index."""
        ...

    @property
    def shards_count(self) -> int:
        """Live shards."""
        ...

@final
class DenseReachability(structseq[Any], tuple[int, int, int, int]):
    """One reachability audit, as `usearch_reachability_t` holds it."""

    __match_args__: Final = ("audited", "reachable", "sweeps", "back_sweeps")

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
        """Levels the sweep took to converge."""
        ...

    @property
    def back_sweeps(self) -> int:
        """Levels the sweep back toward the entry point took, where an audit sweeps both ways."""
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
class DenseBatch:
    """The rows one cursor step wrote into an `Arena`; empty, so falsy, once the scan is drained."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    @property
    def keys(self) -> _ArenaBuffer:
        """(count, 16) identities, most significant byte first."""
        ...

    @property
    def links(self) -> tuple[_ArenaBuffer, _ArenaBuffer] | None:
        """(offsets, keys) of each row's linked identities, row `i` spanning `offsets[i]` to `offsets[i + 1]`, or None when the step asked for none."""
        ...

    def __len__(self) -> int: ...

@final
class DenseCursor:
    """A sequential row scan over the entries an index holds, pulled in batches by `next`."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def next(self, wanted: int, arena: Arena, /, *, links: bool = False) -> DenseBatch:
        """The next batch of at most `wanted` rows, in `arena`. Calls `usearch_dense_cursor_next`."""
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
    """A resumable best-first stream of matches for one query, pulled by `next`."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def next(self, wanted: int, arena: Arena, /) -> Matches:
        """The next at most `wanted` matches, as one row in `arena`. Calls `usearch_dense_search_stream_next`."""
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
        cpu_class: CPUClass | None = None,
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
        gpu_class: GPUClass | None = None,
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
        progress_steps: int = 0,
        cpu_class: CPUClass | None = None,
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
        progress_steps: int = 0,
        cpu_class: CPUClass | None = None,
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
        progress_steps: int = 0,
        gpu_class: GPUClass | None = None,
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
        progress_steps: int = 0,
        gpu_class: GPUClass | None = None,
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
        """Grows the index to host at least `capacity` entries, a hint, since insertions grow it on demand.

        Calls `usearch_dense_reserve`.
        """
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
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> None:
        """Inserts a batch of vectors under their keys, all or nothing. Calls `usearch_dense_insert`."""
        ...

    def refine(
        self,
        *,
        expansion_factor: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
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
        arena: Arena,
        /,
        *,
        expansion_factor: int = 0,
        traversal: Traversal = ...,
        dtype: Scalar | None = None,
        predicate: Callable[[Key], bool] | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Matches:
        """Finds the `wanted` nearest entries for every query row. Calls `usearch_dense_search`."""
        ...

    def count(self, key: _KeyLike, /) -> int:
        """How many entries share `key`. Calls `usearch_dense_count`."""
        ...

    def get(self, key: _KeyLike, arena: Arena, /, *, outputs_dtype: Scalar = ...) -> DenseVectors:
        """Every vector stored under `key`, empty when it is absent. Calls `usearch_dense_get`."""
        ...

    def multiply(
        self,
        rows: _Tensor,
        wanted: int,
        arena: Arena,
        /,
        *,
        expansion_factor: int = 0,
        dtype: Scalar | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> DenseProducts:
        """Every pair the walk scored, an approximate sparse product. Calls `usearch_dense_multiply`."""
        ...

    def attend(
        self,
        queries: _Tensor,
        values: _Tensor,
        row_of: Callable[[Key], int | None],
        arena: Arena,
        /,
        *,
        head_count: int = 1,
        expansion_factor: int = 0,
        score_scale: float = 0.0,
        dtype: Scalar | None = None,
        values_dtype: Scalar | None = None,
        outputs_dtype: Scalar | None = None,
        carry: _Tensor | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> DenseVectors:
        """Softmax attention folded into the walk, refused on VRAM with `DeviceError`. Calls `usearch_dense_attend`."""
        ...

    def maxsim(
        self,
        queries: _Tensor,
        rows_per_query: int,
        wanted: int,
        arena: Arena,
        /,
        *,
        expansion_factor: int = 0,
        traversal: Traversal = ...,
        dtype: Scalar | None = None,
        predicate: Callable[[Key], bool] | None = None,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> DenseProducts:
        """Late interaction over multi-row queries. Calls `usearch_dense_maxsim`."""
        ...

    def absorb(self, *donors: DenseIndex, progress: _Progress | None = None, progress_steps: int = 0) -> None:
        """Absorbs and consumes every donor, one after another. Calls `usearch_dense_absorb`."""
        ...

    def save(self, path: _PathLike, /, *, progress: _Progress | None = None, progress_steps: int = 0) -> None:
        """Writes a checkpoint in the layout the extension picks. Calls `usearch_dense_save`."""
        ...

    def save_buffer(self, arena: Arena, /, *, progress: _Progress | None = None, progress_steps: int = 0) -> Monolith:
        """Serializes one monolith into an arena. Calls `usearch_dense_save_buffer`."""
        ...

    def cursor(self, starts: Sequence[_KeyLike] | _Tensor | None = None) -> DenseCursor:
        """A walk `DenseCursor.next` pulls in batches, from `starts` when given. Calls `usearch_dense_cursor_init`."""
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
    def cpu_class(self) -> CPUClass | None:
        """The `CPUClass` dispatch picked on this machine, or None. Calls `usearch_dense_describe`."""
        ...

    @property
    def gpu_class(self) -> GPUClass | None:
        """The `GPUClass` dispatch picked on the index's devices, or None. Calls `usearch_dense_describe`."""
        ...

    @property
    def shards_count(self) -> int:
        """Live shards. Calls `usearch_dense_describe`."""
        ...

    def __len__(self) -> int:
        """Stored entries. Python-only; calls `usearch_dense_describe`."""
        ...

    def __contains__(self, key: _KeyLike, /) -> bool:
        """Whether any entry holds `key`. Python-only; calls `usearch_dense_count`."""
        ...

    def __repr__(self) -> str: ...

# endregion: Dense Index

# region: Other Modalities

@final
class SequenceIndex:
    """Similarity search over variable-length sequences."""

    def __init__(self) -> NoReturn: ...

@final
class ShapesIndex:
    """Similarity search over 3D point clouds."""

    def __init__(self) -> NoReturn: ...

@final
class GeospatialIndex:
    """Similarity search over space and time."""

    def __init__(self) -> NoReturn: ...

# endregion: Other Modalities
