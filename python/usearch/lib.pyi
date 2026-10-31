"""Type stubs for the `usearch.lib` extension.

Stub files are never executed, so PEP 695 `type` statements and PEP 688's `Buffer` are read
by any checker regardless of the interpreter running the tests. The package floor is 3.12.

File: python/usearch/lib.pyi
Author: Ash Vardanian
Date: September 8, 2026
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
    """How a search reaches the entries it scores, valued as `usearch_traversal_t`."""

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
    MONOVERSIONED = 18

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
        cls,
        threads: int = 0,
        *,
        memory_domains: Sequence[int] | None = None,
        deferred_limit: int = 0,
        cpu_class: CPUClass | None = None,
    ) -> Executor:
        """Spawns host worker threads for a RAM-resident index. Calls `usearch_executor_for_ram`."""
        ...

    @classmethod
    def on_pool(
        cls,
        pool: int,
        /,
        *,
        memory_domains: Sequence[int] | None = None,
        deferred_limit: int = 0,
        cpu_class: CPUClass | None = None,
    ) -> Executor:
        """Drives RAM-resident indexes on a thread pool the caller owns. Calls `usearch_executor_on_pool`."""
        ...

    @classmethod
    def for_vram(
        cls, *, devices: Sequence[int] | None = None, deferred_limit: int = 0, gpu_class: GPUClass | None = None
    ) -> Executor:
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
        cpu_class: CPUClass | None = None,
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
        """(queries + 1,) Apache Arrow offsets."""
        ...

    @property
    def keys(self) -> _ArenaBuffer:
        """(pairs, 16) identities of the indexed entries."""
        ...

    @property
    def scores(self) -> _ArenaBuffer:
        """(pairs,) float64 products."""
        ...

    @property
    def required(self) -> int:
        """Pairs scored before the bound cut any."""
        ...

    def query(self, query: int, /) -> tuple[_ArenaBuffer, _ArenaBuffer]:
        """The (keys, scores) of left row or query `query`, viewing the arena. Python-only."""
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

    def describe(self: On[DenseIndex]) -> DenseIndexDescription:
        """The frozen metadata and the current counters, read in one call. Each counter is read
        once, so under concurrent insertions they may come from slightly different moments.
        Calls `usearch_dense_describe`.

        Example:
          >>> usearch.DenseIndex.init_ram(256).describe().metadata.dimensions
          256
        """
        ...

    def reserve(self: On[DenseIndex], capacity: int, /) -> None:
        """Grows the index to host at least `capacity` entries, a hint,
        since insertions grow the index on demand.
        Calls `usearch_dense_reserve`.

        Args:
          capacity (int): Entries the index must accept without growing again.
        """
        ...

    def clear(self: On[DenseIndex]) -> None:
        """Drops every entry, keeping the capacity and the frozen geometry.
        Calls `usearch_dense_clear`.
        """
        ...

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
    ) -> None:
        """Inserts a batch of vectors under a batch of keys; one vector is a one-row matrix.
        Keys need not be unique: inserting one twice stores two entries, and `count`
        reports both. A batch is all-or-nothing and becomes visible at once, so a rejected
        batch leaves the index untouched and a concurrent search sees all of it or none.
        Calls `usearch_dense_insert`.

        Args:
          keys (Sequence[Key | int] | buffer): One key per row, or an (N, 16) buffer.
          vectors (buffer): An (N, dimensions) matrix; only the row stride may vary.
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it.
          dtype (Scalar, optional): Names the input scalar type when the buffer cannot; it
            must be as wide as the buffer's items.
          progress (callable, optional): Called with (completed, total) entries. It reports and
            cannot stop the batch, every entry being claimed before the first report; what it
            raises surfaces once the batch has finished. Reports are serialized.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Example:
          >>> from array import array
          >>> index = usearch.DenseIndex.init_ram(dimensions=4)
          >>> index.reserve(1)
          >>> index.insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))
          >>> len(index)
          1
        """
        ...

    def refine(
        self: On[DenseIndex],
        *,
        expansion_factor: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> RefineReport:
        """Re-selects the neighbors of every stored entry against the finished graph, improving
        edges that were chosen when the graph around them was still incomplete.

        A build step rather than a repair: a cheap insertion pass followed by one refinement
        reaches a better graph, in less total time, than one expensive insertion pass - so a
        wide `expansion_factor` belongs here rather than in `insert`. Searches may run
        throughout on a RAM handle but not on a VRAM one, and a caller loops until the
        report's `changed` falls below whatever change rate they consider settled.

        Stopping is safe: re-selecting edges claims no identity, so a stopped sweep still
        re-links its islands, leaves the graph correct and merely less improved, and raises
        `CancelledError`. A handler is therefore how a sweep is put under a time budget.
        Calls `usearch_dense_refine`.

        Args:
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it.
          progress (callable, optional): Called with (completed, total) nodes; returning
            False ends the sweep, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          RefineReport: Entries re-selected and changed, the islands found and reconnected,
            and the audit closing the heal; `changed` at zero means the graph has settled.

        Example:
          >>> from array import array
          >>> index = usearch.DenseIndex.init_ram(dimensions=4)
          >>> index.reserve(1)
          >>> index.insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))
          >>> index.refine().changed
          0
        """
        ...

    def health(self: On[DenseIndex]) -> DenseReachability:
        """Audits how much of the graph a search can reach, sweeping forward reachability from as
        many landmarks as one search draws. Read-only.
        Calls `usearch_dense_health`.

        Example:
          >>> from array import array
          >>> index = usearch.DenseIndex.init_ram(dimensions=4)
          >>> index.reserve(1)
          >>> index.insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))
          >>> health = index.health()
          >>> health.reachable == health.audited == 1
          True
        """
        ...

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
    ) -> Matches:
        """Finds the `wanted` nearest entries for every query row; one query is a one-row matrix.
        Never writes to the graph, so any number of searches may overlap with insertions.
        Calls `usearch_dense_search`.

        Args:
          queries (buffer): A (Q, dimensions) matrix of query rows.
          wanted (int): Upper bound on matches per query - the 'k' in 'kANN'.
          arena (Arena): Where the matches land, rewound by the call.
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it.
          traversal (Traversal, optional): `Traversal.WALK`, the default, walks the graph;
            `Traversal.SCAN` scores every entry exactly, on RAM and NVMe handles.
          dtype (Scalar, optional): Names the query scalar type when the buffer cannot.
          predicate (callable, optional): Admits a Key when it returns truthy. Non-matching
            entries still route the traversal, so recall survives low selectivity. The
            callback re-enters Python, so it runs under the GIL and serializes the walk.
          progress (callable, optional): Called with (completed, total) queries; returning
            False stops the batch and raises `CancelledError`, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          Matches: (Q, columns) runs in the arena, columns at most `wanted`, each count
            bounding its own row.

        Example:
          >>> from array import array
          >>> index = usearch.DenseIndex.init_ram(dimensions=4)
          >>> index.reserve(1)
          >>> query = array('d', [0.1, 0.2, 0.3, 0.4])
          >>> index.insert([7], query)
          >>> key, distance = index.search(query, 1, usearch.Arena())[0][0]
          >>> int(key)
          7
        """
        ...

    def count(self: On[DenseIndex], key: _KeyLike, /) -> int:
        """How many entries share `key`; zero means the index holds none.
        Calls `usearch_dense_count`.

        Args:
          key (Key | int | str): The identity to count.

        Example:
          >>> from array import array
          >>> index = usearch.DenseIndex.init_ram(dimensions=2)
          >>> index.reserve(2)
          >>> row = array('d', [0.1, 0.2])
          >>> index.insert([7], row)
          >>> index.insert([7], row)
          >>> index.count(7)
          2
        """
        ...

    def get(self: On[DenseIndex], key: _KeyLike, arena: Arena, /, *, outputs_dtype: Scalar = ...) -> DenseVectors:
        """Every vector stored under `key`, in insertion order, and none when the key is absent.
        Calls `usearch_dense_get`.

        Args:
          key (Key | int | str): The identity to retrieve.
          arena (Arena): Where the rows land, rewound by the call.
          outputs_dtype (Scalar, optional): The scalar type to export as; converted from storage.

        Returns:
          DenseVectors: A writable (count, dimensions) matrix in the arena, for the kinds
            `memoryview` can format - f64, f32, f16, i8, u8. Sub-byte and mini-float kinds,
            `bf16` among them, leave as a (count, row_bytes) run of 'B' instead. Empty, so falsy,
            when the index holds no entry under `key`.
        """
        ...

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
    ) -> DenseProducts:
        """Approximate sparse product of a matrix against the indexed vectors - the linear algebra
        sibling of `search`, reporting every pair the traversal scored rather than only the
        nearest. Pairs arrive in traversal order rather than sorted, and carry no duplicates.
        How many a walk scores is known only once it ends, so the pairs come back ragged.
        Calls `usearch_dense_multiply`.

        Args:
          rows (buffer): An (R, dimensions) matrix of left-hand rows.
          wanted (int): Upper bound on pairs kept per row, the first ones its walk scored;
            0 keeps every pair.
          arena (Arena): Where the pairs and their offsets land, rewound by the call.
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it. A wider
            factor widens the emitted pattern, being what the walk's breadth decides.
          dtype (Scalar, optional): Names the left-hand scalar type when the buffer cannot.
          progress (callable, optional): Called with (completed, total) rows; returning False
            stops the batch and raises `CancelledError`, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          DenseProducts: `offsets`, (R + 1,) Apache Arrow offsets, left row `i` spanning pairs
            `offsets[i]` up to `offsets[i + 1]`; `keys`, a (pairs, 16) run naming each pair's
            indexed entry; `scores`, the (pairs,) float64 products; and `required`, the pairs
            scored before `wanted` cut any. All of them live in the arena.
        """
        ...

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
    ) -> DenseVectors:
        """Softmax attention of every query row over every indexed key its walk scores, folded
        online as Flash Attention folds key tiles, with no top-k cut. Never writes to the
        graph, so a growing context may stream through this while insertions continue.

        The index holds one layer's keys for one key-value head and no values: `row_of` names
        the row of `values` each scored key reads, so only keys a walk scores are ever resolved.
        Calls `usearch_dense_attend`.

        Args:
          queries (buffer): A (Q, dimensions) matrix; row `t * head_count + h` is head `h`
            of token `t`.
          values (buffer): An (N, dimensions) matrix of value rows, as wide as the keys and
            stored in the index's quantization, borrowed until the call is final.
          row_of (callable): Called with each scored Key, answers its row of `values`. None, or
            a row outside them, refuses the call with `ShapeMismatchError`, as do values of
            another width or scalar type. It re-enters Python from the library's threads, so it
            runs under the GIL and serializes the walk. A VRAM walk resolves values on the
            device, where no Python callable runs, so a VRAM arena raises `DeviceError`.
          arena (Arena): Where the attended rows land, rewound by the call.
          head_count (int, optional): The query heads sharing the index's key-value head, and
            with it one walk per token; it must divide Q.
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it. Wider
            candidates walk further, folding more of the tail.
          score_scale (float, optional): The logit scale, conventionally the inverse square
            root of the head dimensionality; 0.0 derives it from the frozen geometry.
          dtype (Scalar, optional): Names the query scalar type when the buffer cannot.
          values_dtype (Scalar, optional): Names the values' scalar type when the buffer cannot.
          outputs_dtype (Scalar, optional): The scalar type to write the outputs as; defaults
            to the queries' own.
          carry (buffer, optional): A writable (Q, 2) buffer of float32, holding each query's
            largest logit and the softmax denominator taken against it. The pair recombines
            this branch with another over the same queries - attention sinks, a dense recent
            window, or a second index of one context - by partial-softmax rescaling; a maximum
            without its partition is a number nothing can rescale, which is why they travel as
            one row.
          progress (callable, optional): Called with (completed, total) tokens; returning
            False stops the batch and raises `CancelledError`, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          DenseVectors: The (Q, dimensions) attended rows in the arena, already divided by the
            partition; kinds `memoryview` cannot format leave as a (Q, row_bytes) run of 'B'.
        """
        ...

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
    ) -> DenseProducts:
        """Late-interaction scoring of multi-row queries against multi-vector documents: every
        query row takes its best match within a key, and those maxima sum to that key's score.

        A document is one key holding its token rows, which is what `values_per_key` above one
        exists for, and the walk stays over individual tokens under a plain inner product - so
        a document's vectors need no ordering and two documents need not share a vector count.
        The result is sized over documents rather than over rows.
        Calls `usearch_dense_maxsim`.

        Args:
          queries (buffer): The token rows of every query laid end to end, an (Q * R,
            dimensions) matrix.
          rows_per_query (int): Rows every query occupies; uniform, so ColBERT's padding
            applies as is. The query row count must be a whole multiple of it.
          wanted (int): Upper bound on matches per query - the 'k' over documents, not rows;
            0 keeps every document the query scored.
          arena (Arena): Where the scored documents land, rewound by the call.
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it. Widening it
            nominates more documents, each still scored exactly.
          traversal (Traversal, optional): `Traversal.WALK`, the default, nominates documents by
            walking the graph; `Traversal.SCAN` nominates every document, on RAM and NVMe.
          dtype (Scalar, optional): Names the query scalar type when the buffer cannot.
          predicate (callable, optional): Admits a Key when it returns truthy, asked before any
            of its vectors are scored, so a rejected document never scores.
          progress (callable, optional): Called with (completed, total) queries; returning
            False stops the batch and raises `CancelledError`, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          DenseProducts: `offsets`, (Q + 1,) Apache Arrow offsets, query `i` spanning documents
            `offsets[i]` up to `offsets[i + 1]`, largest first; `keys`, a (documents, 16) run of
            identities; `scores`, their (documents,) summed maxima; and `required`, the
            documents scored before `wanted` cut any. All of them live in the arena.
        """
        ...

    def absorb(
        self: On[DenseIndex], *donors: DenseIndex, progress: _Progress | None = None, progress_steps: int = 0
    ) -> RefineReport:
        """Absorbs every donor into this index, one after another, keeping their graphs rather
        than re-inserting their vectors, and linking each in by walks across the appended
        part. Each donor handle is consumed and every verb on it afterwards raises. Readers
        of a RAM index may keep searching throughout, while a VRAM one admits no overlapping
        search.

        Cancellation stops the closing refinement alone: every donor is still consumed, the
        islands still re-linked, and the call raises `CancelledError`. A donor whose absorption
        fails, and every later one, stays a live handle its caller still owns, though the
        entries of the one it struck may already have moved into this index.
        Calls `usearch_dense_absorb`.

        Args:
          donors (DenseIndex): The indexes to absorb.
          progress (callable, optional): Called with (completed, total) nodes the closing
            refinement re-selected; returning False stops it, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.

        Returns:
          RefineReport: What the closing refinement did to the graph.
        """
        ...

    def split(
        self: On[DenseIndex],
        *receivers: DenseIndex,
        expansion_factor: int = 0,
        seed: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> RefineReport:
        """Splits this index into the empty receivers, the inverse of `absorb`: the entries are
        partitioned along the graph, each part lands in one receiver with its edges kept, and
        a closing refinement re-links every receiver. This index is left empty and live.
        Receivers match its residency, scalar, metric, geometry and shard count.
        Calls `usearch_dense_split`.

        Args:
          receivers (DenseIndex): At least two empty indexes, one per part.
          expansion_factor (int, optional): Breadth of the closing walks; 0 derives it.
          seed (int, optional): Picks the part roots; equal seeds split equally.
          progress (callable, optional): Called with (completed, total) nodes the closing
            refinements re-selected; returning False stops them, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          RefineReport: What the closing refinements did, across the receivers.
        """
        ...

    def sample(
        self: On[DenseIndex],
        wanted: int,
        arena: Arena,
        /,
        *,
        shown: Sequence[_KeyLike] | _Tensor | None = None,
        shown_coordinates: _Tensor | None = None,
        focus: Sequence[_KeyLike] | _Tensor | None = None,
        seed: int = 0,
        expansion_factor: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Sample:
        """Draws keys around `focus`, or across the whole index, each hanging from a shown or
        earlier draw and placed in three dimensions beside the shown ones - so a view grows
        one connected sample as it zooms, never scanning the index. Equal seeds, shown keys
        and focus over an unchanged index draw equally, and a larger `wanted` draws a superset.
        Only a RAM index samples yet.
        Calls `usearch_dense_sample`.

        Args:
          wanted (int): The upper bound on draws.
          arena (Arena): Where the draws are written, rewound by the call.
          shown (keys, optional): The keys already shown, never drawn again.
          shown_coordinates (tensor, optional): (len(shown), 3) float32 places of the shown
            keys, the points later draws are placed against.
          focus (keys, optional): The keys to look around; None looks across the index.
          seed (int, optional): The draw; equal seeds draw equally.
          expansion_factor (int, optional): Breadth of the walks behind the draws; 0 derives.
          progress (callable, optional): Called with (written, wanted) draws.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          Sample: The keys, entry ordinals, parents and coordinates of the draws, fewer than
            `wanted` only once the region is exhausted.
        """
        ...

    def save(
        self: On[DenseIndex], path: _PathLike, /, *, progress: _Progress | None = None, progress_steps: int = 0
    ) -> None:
        """Writes the index under `path`, in the layout its extension picks: one `.safegraphs`
        monolith, or a `.safegraphs.json` manifest beside one part per shard. Demands a
        mutation-free window; concurrent readers keep searching throughout.
        Calls `usearch_dense_save`.

        Args:
          path (str | bytes | os.PathLike): Where the checkpoint goes.
          progress (callable, optional): Called with (completed, total) shards; returning
            False stops the write and raises `CancelledError`, leaving a checkpoint no
            reader accepts and whatever parts it already wrote for the caller to remove.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
        """
        ...

    def save_buffer(
        self: On[DenseIndex], arena: Arena, /, *, progress: _Progress | None = None, progress_steps: int = 0
    ) -> Monolith:
        """Serializes the index as one monolith of every shard, the very bytes `save` writes to
        a `.safegraphs` path.
        Calls `usearch_dense_save_buffer`.

        Args:
          arena (Arena): Where the monolith lands, rewound by the call.
          progress (callable, optional): Called with (completed, total) bytes; returning
            False stops the write and raises `CancelledError`, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.

        Returns:
          Monolith: The monolith's bytes, in place in the arena.

        Example:
          >>> index = usearch.DenseIndex.init_ram(dimensions=4)
          >>> index.reserve(1)
          >>> checkpoint = index.save_buffer(usearch.Arena())
          >>> usearch.DenseIndex.metadata_read_buffer(checkpoint).dimensions
          4
        """
        ...

    def cursor(self: On[DenseIndex], starts: Sequence[_KeyLike] | _Tensor | None = None) -> DenseCursor:
        """Calls `usearch_dense_cursor_init`."""
        ...

    def search_stream(
        self: On[DenseIndex], query: _Tensor, /, *, expansion_factor: int = 0, dtype: Scalar | None = None
    ) -> DenseSearchStream:
        """Opens a resumable nearest-neighbor stream for one query, which `next` pulls for as
        long as the caller wants more, each pull into the arena it names. Every pull blocks,
        on the receiver's executor, which the stream keeps alive, or on the calling thread.
        Calls `usearch_dense_search_stream_init`.

        Args:
          query (buffer): One vector of `dimensions` scalars.
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it.
          dtype (Scalar, optional): Names the query scalar type when the buffer cannot.

        Example:
          >>> from array import array
          >>> index = usearch.DenseIndex.init_ram(dimensions=2)
          >>> index.reserve(1)
          >>> index.insert([5], array('d', [0.1, 0.2]))
          >>> stream, arena = index.search_stream(array('d', [0.1, 0.2])), usearch.Arena()
          >>> [int(match.key) for match in stream.next(4, arena)[0]]
          [5]
          >>> len(stream.next(4, arena)[0])
          0
        """
        ...

    def __repr__(self) -> str: ...

@final
class DeferredOn[IndexT]:
    """An index bound to the executor its verbs submit to, each answering a `Deferred` at once."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def __class_getitem__(cls, item: Any, /) -> GenericAlias:
        """Names the index the receiver addresses, as `DeferredOn[DenseIndex]`."""
        ...

    def describe(self: DeferredOn[DenseIndex]) -> Deferred[DenseIndexDescription]:
        """The frozen metadata and the current counters, read in one call. Each counter is read
        once, so under concurrent insertions they may come from slightly different moments.
        Calls `usearch_dense_describe`.

        Example:
          >>> usearch.DenseIndex.init_ram(256).describe().metadata.dimensions
          256
        """
        ...

    def reserve(self: DeferredOn[DenseIndex], capacity: int, /) -> Deferred[None]:
        """Grows the index to host at least `capacity` entries, a hint,
        since insertions grow the index on demand.
        Calls `usearch_dense_reserve`.

        Args:
          capacity (int): Entries the index must accept without growing again.
        """
        ...

    def clear(self: DeferredOn[DenseIndex]) -> Deferred[None]:
        """Drops every entry, keeping the capacity and the frozen geometry.
        Calls `usearch_dense_clear`.
        """
        ...

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
    ) -> Deferred[None]:
        """Inserts a batch of vectors under a batch of keys; one vector is a one-row matrix.
        Keys need not be unique: inserting one twice stores two entries, and `count`
        reports both. A batch is all-or-nothing and becomes visible at once, so a rejected
        batch leaves the index untouched and a concurrent search sees all of it or none.
        Calls `usearch_dense_insert`.

        Args:
          keys (Sequence[Key | int] | buffer): One key per row, or an (N, 16) buffer.
          vectors (buffer): An (N, dimensions) matrix; only the row stride may vary.
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it.
          dtype (Scalar, optional): Names the input scalar type when the buffer cannot; it
            must be as wide as the buffer's items.
          progress (callable, optional): Called with (completed, total) entries. It reports and
            cannot stop the batch, every entry being claimed before the first report; what it
            raises surfaces once the batch has finished. Reports are serialized.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Example:
          >>> from array import array
          >>> index = usearch.DenseIndex.init_ram(dimensions=4)
          >>> index.reserve(1)
          >>> index.insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))
          >>> len(index)
          1
        """
        ...

    def refine(
        self: DeferredOn[DenseIndex],
        *,
        expansion_factor: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Deferred[RefineReport]:
        """Re-selects the neighbors of every stored entry against the finished graph, improving
        edges that were chosen when the graph around them was still incomplete.

        A build step rather than a repair: a cheap insertion pass followed by one refinement
        reaches a better graph, in less total time, than one expensive insertion pass - so a
        wide `expansion_factor` belongs here rather than in `insert`. Searches may run
        throughout on a RAM handle but not on a VRAM one, and a caller loops until the
        report's `changed` falls below whatever change rate they consider settled.

        Stopping is safe: re-selecting edges claims no identity, so a stopped sweep still
        re-links its islands, leaves the graph correct and merely less improved, and raises
        `CancelledError`. A handler is therefore how a sweep is put under a time budget.
        Calls `usearch_dense_refine`.

        Args:
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it.
          progress (callable, optional): Called with (completed, total) nodes; returning
            False ends the sweep, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          RefineReport: Entries re-selected and changed, the islands found and reconnected,
            and the audit closing the heal; `changed` at zero means the graph has settled.

        Example:
          >>> from array import array
          >>> index = usearch.DenseIndex.init_ram(dimensions=4)
          >>> index.reserve(1)
          >>> index.insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))
          >>> index.refine().changed
          0
        """
        ...

    def health(self: DeferredOn[DenseIndex]) -> Deferred[DenseReachability]:
        """Audits how much of the graph a search can reach, sweeping forward reachability from as
        many landmarks as one search draws. Read-only.
        Calls `usearch_dense_health`.

        Example:
          >>> from array import array
          >>> index = usearch.DenseIndex.init_ram(dimensions=4)
          >>> index.reserve(1)
          >>> index.insert([1], array('d', [0.1, 0.2, 0.3, 0.4]))
          >>> health = index.health()
          >>> health.reachable == health.audited == 1
          True
        """
        ...

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
    ) -> Deferred[Matches]:
        """Finds the `wanted` nearest entries for every query row; one query is a one-row matrix.
        Never writes to the graph, so any number of searches may overlap with insertions.
        Calls `usearch_dense_search`.

        Args:
          queries (buffer): A (Q, dimensions) matrix of query rows.
          wanted (int): Upper bound on matches per query - the 'k' in 'kANN'.
          arena (Arena): Where the matches land, rewound by the call.
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it.
          traversal (Traversal, optional): `Traversal.WALK`, the default, walks the graph;
            `Traversal.SCAN` scores every entry exactly, on RAM and NVMe handles.
          dtype (Scalar, optional): Names the query scalar type when the buffer cannot.
          predicate (callable, optional): Admits a Key when it returns truthy. Non-matching
            entries still route the traversal, so recall survives low selectivity. The
            callback re-enters Python, so it runs under the GIL and serializes the walk.
          progress (callable, optional): Called with (completed, total) queries; returning
            False stops the batch and raises `CancelledError`, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          Matches: (Q, columns) runs in the arena, columns at most `wanted`, each count
            bounding its own row.

        Example:
          >>> from array import array
          >>> index = usearch.DenseIndex.init_ram(dimensions=4)
          >>> index.reserve(1)
          >>> query = array('d', [0.1, 0.2, 0.3, 0.4])
          >>> index.insert([7], query)
          >>> key, distance = index.search(query, 1, usearch.Arena())[0][0]
          >>> int(key)
          7
        """
        ...

    def count(self: DeferredOn[DenseIndex], key: _KeyLike, /) -> Deferred[int]:
        """How many entries share `key`; zero means the index holds none.
        Calls `usearch_dense_count`.

        Args:
          key (Key | int | str): The identity to count.

        Example:
          >>> from array import array
          >>> index = usearch.DenseIndex.init_ram(dimensions=2)
          >>> index.reserve(2)
          >>> row = array('d', [0.1, 0.2])
          >>> index.insert([7], row)
          >>> index.insert([7], row)
          >>> index.count(7)
          2
        """
        ...

    def get(
        self: DeferredOn[DenseIndex], key: _KeyLike, arena: Arena, /, *, outputs_dtype: Scalar = ...
    ) -> Deferred[DenseVectors]:
        """Every vector stored under `key`, in insertion order, and none when the key is absent.
        Calls `usearch_dense_get`.

        Args:
          key (Key | int | str): The identity to retrieve.
          arena (Arena): Where the rows land, rewound by the call.
          outputs_dtype (Scalar, optional): The scalar type to export as; converted from storage.

        Returns:
          DenseVectors: A writable (count, dimensions) matrix in the arena, for the kinds
            `memoryview` can format - f64, f32, f16, i8, u8. Sub-byte and mini-float kinds,
            `bf16` among them, leave as a (count, row_bytes) run of 'B' instead. Empty, so falsy,
            when the index holds no entry under `key`.
        """
        ...

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
    ) -> Deferred[DenseProducts]:
        """Approximate sparse product of a matrix against the indexed vectors - the linear algebra
        sibling of `search`, reporting every pair the traversal scored rather than only the
        nearest. Pairs arrive in traversal order rather than sorted, and carry no duplicates.
        How many a walk scores is known only once it ends, so the pairs come back ragged.
        Calls `usearch_dense_multiply`.

        Args:
          rows (buffer): An (R, dimensions) matrix of left-hand rows.
          wanted (int): Upper bound on pairs kept per row, the first ones its walk scored;
            0 keeps every pair.
          arena (Arena): Where the pairs and their offsets land, rewound by the call.
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it. A wider
            factor widens the emitted pattern, being what the walk's breadth decides.
          dtype (Scalar, optional): Names the left-hand scalar type when the buffer cannot.
          progress (callable, optional): Called with (completed, total) rows; returning False
            stops the batch and raises `CancelledError`, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          DenseProducts: `offsets`, (R + 1,) Apache Arrow offsets, left row `i` spanning pairs
            `offsets[i]` up to `offsets[i + 1]`; `keys`, a (pairs, 16) run naming each pair's
            indexed entry; `scores`, the (pairs,) float64 products; and `required`, the pairs
            scored before `wanted` cut any. All of them live in the arena.
        """
        ...

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
    ) -> Deferred[DenseVectors]:
        """Softmax attention of every query row over every indexed key its walk scores, folded
        online as Flash Attention folds key tiles, with no top-k cut. Never writes to the
        graph, so a growing context may stream through this while insertions continue.

        The index holds one layer's keys for one key-value head and no values: `row_of` names
        the row of `values` each scored key reads, so only keys a walk scores are ever resolved.
        Calls `usearch_dense_attend`.

        Args:
          queries (buffer): A (Q, dimensions) matrix; row `t * head_count + h` is head `h`
            of token `t`.
          values (buffer): An (N, dimensions) matrix of value rows, as wide as the keys and
            stored in the index's quantization, borrowed until the call is final.
          row_of (callable): Called with each scored Key, answers its row of `values`. None, or
            a row outside them, refuses the call with `ShapeMismatchError`, as do values of
            another width or scalar type. It re-enters Python from the library's threads, so it
            runs under the GIL and serializes the walk. A VRAM walk resolves values on the
            device, where no Python callable runs, so a VRAM arena raises `DeviceError`.
          arena (Arena): Where the attended rows land, rewound by the call.
          head_count (int, optional): The query heads sharing the index's key-value head, and
            with it one walk per token; it must divide Q.
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it. Wider
            candidates walk further, folding more of the tail.
          score_scale (float, optional): The logit scale, conventionally the inverse square
            root of the head dimensionality; 0.0 derives it from the frozen geometry.
          dtype (Scalar, optional): Names the query scalar type when the buffer cannot.
          values_dtype (Scalar, optional): Names the values' scalar type when the buffer cannot.
          outputs_dtype (Scalar, optional): The scalar type to write the outputs as; defaults
            to the queries' own.
          carry (buffer, optional): A writable (Q, 2) buffer of float32, holding each query's
            largest logit and the softmax denominator taken against it. The pair recombines
            this branch with another over the same queries - attention sinks, a dense recent
            window, or a second index of one context - by partial-softmax rescaling; a maximum
            without its partition is a number nothing can rescale, which is why they travel as
            one row.
          progress (callable, optional): Called with (completed, total) tokens; returning
            False stops the batch and raises `CancelledError`, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          DenseVectors: The (Q, dimensions) attended rows in the arena, already divided by the
            partition; kinds `memoryview` cannot format leave as a (Q, row_bytes) run of 'B'.
        """
        ...

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
    ) -> Deferred[DenseProducts]:
        """Late-interaction scoring of multi-row queries against multi-vector documents: every
        query row takes its best match within a key, and those maxima sum to that key's score.

        A document is one key holding its token rows, which is what `values_per_key` above one
        exists for, and the walk stays over individual tokens under a plain inner product - so
        a document's vectors need no ordering and two documents need not share a vector count.
        The result is sized over documents rather than over rows.
        Calls `usearch_dense_maxsim`.

        Args:
          queries (buffer): The token rows of every query laid end to end, an (Q * R,
            dimensions) matrix.
          rows_per_query (int): Rows every query occupies; uniform, so ColBERT's padding
            applies as is. The query row count must be a whole multiple of it.
          wanted (int): Upper bound on matches per query - the 'k' over documents, not rows;
            0 keeps every document the query scored.
          arena (Arena): Where the scored documents land, rewound by the call.
          expansion_factor (int, optional): Breadth of the candidates; 0 derives it. Widening it
            nominates more documents, each still scored exactly.
          traversal (Traversal, optional): `Traversal.WALK`, the default, nominates documents by
            walking the graph; `Traversal.SCAN` nominates every document, on RAM and NVMe.
          dtype (Scalar, optional): Names the query scalar type when the buffer cannot.
          predicate (callable, optional): Admits a Key when it returns truthy, asked before any
            of its vectors are scored, so a rejected document never scores.
          progress (callable, optional): Called with (completed, total) queries; returning
            False stops the batch and raises `CancelledError`, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          DenseProducts: `offsets`, (Q + 1,) Apache Arrow offsets, query `i` spanning documents
            `offsets[i]` up to `offsets[i + 1]`, largest first; `keys`, a (documents, 16) run of
            identities; `scores`, their (documents,) summed maxima; and `required`, the
            documents scored before `wanted` cut any. All of them live in the arena.
        """
        ...

    def absorb(
        self: DeferredOn[DenseIndex], *donors: DenseIndex, progress: _Progress | None = None, progress_steps: int = 0
    ) -> Deferred[RefineReport]:
        """Absorbs every donor into this index, one after another, keeping their graphs rather
        than re-inserting their vectors, and linking each in by walks across the appended
        part. Each donor handle is consumed and every verb on it afterwards raises. Readers
        of a RAM index may keep searching throughout, while a VRAM one admits no overlapping
        search.

        Cancellation stops the closing refinement alone: every donor is still consumed, the
        islands still re-linked, and the call raises `CancelledError`. A donor whose absorption
        fails, and every later one, stays a live handle its caller still owns, though the
        entries of the one it struck may already have moved into this index.
        Calls `usearch_dense_absorb`.

        Args:
          donors (DenseIndex): The indexes to absorb.
          progress (callable, optional): Called with (completed, total) nodes the closing
            refinement re-selected; returning False stops it, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.

        Returns:
          RefineReport: What the closing refinement did to the graph.
        """
        ...

    def split(
        self: DeferredOn[DenseIndex],
        *receivers: DenseIndex,
        expansion_factor: int = 0,
        seed: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Deferred[RefineReport]:
        """Splits this index into the empty receivers, the inverse of `absorb`: the entries are
        partitioned along the graph, each part lands in one receiver with its edges kept, and
        a closing refinement re-links every receiver. This index is left empty and live.
        Receivers match its residency, scalar, metric, geometry and shard count.
        Calls `usearch_dense_split`.

        Args:
          receivers (DenseIndex): At least two empty indexes, one per part.
          expansion_factor (int, optional): Breadth of the closing walks; 0 derives it.
          seed (int, optional): Picks the part roots; equal seeds split equally.
          progress (callable, optional): Called with (completed, total) nodes the closing
            refinements re-selected; returning False stops them, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          RefineReport: What the closing refinements did, across the receivers.
        """
        ...

    def sample(
        self: DeferredOn[DenseIndex],
        wanted: int,
        arena: Arena,
        /,
        *,
        shown: Sequence[_KeyLike] | _Tensor | None = None,
        shown_coordinates: _Tensor | None = None,
        focus: Sequence[_KeyLike] | _Tensor | None = None,
        seed: int = 0,
        expansion_factor: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Deferred[Sample]:
        """Draws keys around `focus`, or across the whole index, each hanging from a shown or
        earlier draw and placed in three dimensions beside the shown ones - so a view grows
        one connected sample as it zooms, never scanning the index. Equal seeds, shown keys
        and focus over an unchanged index draw equally, and a larger `wanted` draws a superset.
        Only a RAM index samples yet.
        Calls `usearch_dense_sample`.

        Args:
          wanted (int): The upper bound on draws.
          arena (Arena): Where the draws are written, rewound by the call.
          shown (keys, optional): The keys already shown, never drawn again.
          shown_coordinates (tensor, optional): (len(shown), 3) float32 places of the shown
            keys, the points later draws are placed against.
          focus (keys, optional): The keys to look around; None looks across the index.
          seed (int, optional): The draw; equal seeds draw equally.
          expansion_factor (int, optional): Breadth of the walks behind the draws; 0 derives.
          progress (callable, optional): Called with (written, wanted) draws.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
          stats (DenseTraversalStats, optional): Overwritten with this call's telemetry.

        Returns:
          Sample: The keys, entry ordinals, parents and coordinates of the draws, fewer than
            `wanted` only once the region is exhausted.
        """
        ...

    def save(
        self: DeferredOn[DenseIndex], path: _PathLike, /, *, progress: _Progress | None = None, progress_steps: int = 0
    ) -> Deferred[None]:
        """Writes the index under `path`, in the layout its extension picks: one `.safegraphs`
        monolith, or a `.safegraphs.json` manifest beside one part per shard. Demands a
        mutation-free window; concurrent readers keep searching throughout.
        Calls `usearch_dense_save`.

        Args:
          path (str | bytes | os.PathLike): Where the checkpoint goes.
          progress (callable, optional): Called with (completed, total) shards; returning
            False stops the write and raises `CancelledError`, leaving a checkpoint no
            reader accepts and whatever parts it already wrote for the caller to remove.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.
        """
        ...

    def save_buffer(
        self: DeferredOn[DenseIndex], arena: Arena, /, *, progress: _Progress | None = None, progress_steps: int = 0
    ) -> Deferred[Monolith]:
        """Serializes the index as one monolith of every shard, the very bytes `save` writes to
        a `.safegraphs` path.
        Calls `usearch_dense_save_buffer`.

        Args:
          arena (Arena): Where the monolith lands, rewound by the call.
          progress (callable, optional): Called with (completed, total) bytes; returning
            False stops the write and raises `CancelledError`, None carries on.
          progress_steps (int, optional): Reports the call makes at most; 0 derives to ten.

        Returns:
          Monolith: The monolith's bytes, in place in the arena.

        Example:
          >>> index = usearch.DenseIndex.init_ram(dimensions=4)
          >>> index.reserve(1)
          >>> checkpoint = index.save_buffer(usearch.Arena())
          >>> usearch.DenseIndex.metadata_read_buffer(checkpoint).dimensions
          4
        """
        ...

    def __repr__(self) -> str: ...

# endregion: Shared Vocabulary

# region: Dense Index

class DenseMetric(IntEnum):
    """How a dense index scores two vectors, valued as `usearch_dense_metric_t`."""

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
        """The `Scalar` its vectors are stored and quantized into."""
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
class RefineReport(structseq[Any], tuple[int, int, int, int, DenseReachability]):
    """What one refine-family call did to the graph, as `usearch_refine_report_t` counts it."""

    __match_args__: Final = ("reselected", "changed", "islands_found", "islands_reconnected", "reachability")

    @property
    def reselected(self) -> int:
        """Entries whose neighbors were selected again, islands of every heal round included."""
        ...

    @property
    def changed(self) -> int:
        """Of those, the entries whose neighbor set gained a member it lacked; zero means settled."""
        ...

    @property
    def islands_found(self) -> int:
        """Entries the first audit found not reached both ways."""
        ...

    @property
    def islands_reconnected(self) -> int:
        """Links written between an island and an entry reached both ways."""
        ...

    @property
    def reachability(self) -> DenseReachability:
        """The audit closing the heal, as `health` would take it right after."""
        ...

@final
class Sample(structseq[Any], tuple[_ArenaBuffer, _ArenaBuffer, Matches, _ArenaBuffer]):
    """The draws one `sample` call wrote into an `Arena`, viewing it until it rewinds."""

    __match_args__: Final = ("keys", "entry_ordinals", "parents", "coordinates")

    @property
    def keys(self) -> _ArenaBuffer:
        """(written, 16) identities drawn, most significant byte first."""
        ...

    @property
    def entry_ordinals(self) -> _ArenaBuffer:
        """(written,) which of its key's entries each draw is, in the order `get` writes them."""
        ...

    @property
    def parents(self) -> Matches:
        """One column per draw: the shown or earlier draw it hangs from, and its distance."""
        ...

    @property
    def coordinates(self) -> _ArenaBuffer:
        """(written, 3) float32 places of the draws, beside the shown keys."""
        ...

@final
class DenseTraversalStats:
    """Traversal telemetry, fully overwritten by every call it is passed to; `+=` sums a sweep."""

    def __init__(self) -> None: ...
    @property
    def visited_members(self) -> int:
        """Entries whose neighborhoods the walk expanded."""
        ...

    @property
    def computed_distances(self) -> int:
        """Distance evaluations the walk paid for."""
        ...

    @property
    def restarts(self) -> int:
        """Times the walk fell back to a fresh entry point."""
        ...

    def __iadd__(self, other: DenseTraversalStats, /) -> Self:
        """Python-only."""
        ...

    def __repr__(self) -> str: ...

@final
class DenseBatch:
    """The entries one cursor step wrote into an `Arena`; empty, so falsy, once the scan is drained."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    @property
    def keys(self) -> _ArenaBuffer:
        """(count, 16) identities, most significant byte first."""
        ...

    @property
    def links(self) -> tuple[_ArenaBuffer, _ArenaBuffer] | None:
        """(offsets, keys) of each entry's linked identities, entry `i` spanning `offsets[i]` to `offsets[i + 1]`, or None when the step asked for none."""
        ...

    def __len__(self) -> int: ...

@final
class DenseCursor:
    """A sequential scan over the entries an index holds, pulled in batches by `next`."""

    def __new__(cls, *args: object, **keywords: object) -> NoReturn: ...
    def next(self, wanted: int, arena: Arena, /, *, links: bool = False) -> DenseBatch:
        """The next batch of at most `wanted` entries, in `arena`. Calls `usearch_dense_cursor_next`."""
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
        """The next at most `wanted` matches, as one query in `arena`. Calls `usearch_dense_search_stream_next`."""
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
        cls,
        path: _PathLike,
        /,
        *,
        paths: Sequence[_PathLike] | None = None,
        executor: Executor | None = None,
        cpu_class: CPUClass | None = None,
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
    ) -> RefineReport:
        """Re-selects every entry's neighbors, reporting what changed. Calls `usearch_dense_refine`."""
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

    def absorb(self, *donors: DenseIndex, progress: _Progress | None = None, progress_steps: int = 0) -> RefineReport:
        """Absorbs and consumes every donor, one after another. Calls `usearch_dense_absorb`."""
        ...

    def split(
        self,
        *receivers: DenseIndex,
        expansion_factor: int = 0,
        seed: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> RefineReport:
        """Splits the entries into the empty receivers, leaving this index empty. Calls `usearch_dense_split`."""
        ...

    def sample(
        self,
        wanted: int,
        arena: Arena,
        /,
        *,
        shown: Sequence[_KeyLike] | _Tensor | None = None,
        shown_coordinates: _Tensor | None = None,
        focus: Sequence[_KeyLike] | _Tensor | None = None,
        seed: int = 0,
        expansion_factor: int = 0,
        progress: _Progress | None = None,
        progress_steps: int = 0,
        stats: DenseTraversalStats | None = None,
    ) -> Sample:
        """Draws keys around `focus`, placed beside the `shown` keys. Calls `usearch_dense_sample`."""
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
