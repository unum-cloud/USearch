"""Type stubs for the dense index and the walks it hands out."""

from collections.abc import Buffer, Callable, Sequence
from typing import Literal, NoReturn, TypedDict

from usearch.types import Arena, Executor, Key, KeyLike, Match, Matches, PathLike, ScalarKind, ScalarName, Stats, Tensor

# region: Dense Index

type MetricName = Literal["inner", "angular", "euclidean", "hamming", "jaccard"]

class DenseMetadata(TypedDict):
    """The geometry keywords a checkpoint fixes; residency is never among them."""

    dimensions: int
    metric: MetricName
    quantization: ScalarKind
    connectivity: int

class DenseDescription(DenseMetadata):
    """A live handle's frozen geometry beside its current counters."""

    size: int
    capacity: int
    memory_usage: int
    capabilities: str

class DenseProducts(TypedDict):
    """Every pair a `multiply` walk scored, ragged by Apache Arrow offsets."""

    offsets: memoryview
    keys: memoryview
    scores: memoryview
    required: int

class DenseCursor:
    """A snapshot walk over the entries an index held when the cursor opened."""

    def __iter__(self) -> DenseCursor: ...
    def __next__(self) -> memoryview | tuple[memoryview, memoryview]: ...

class DenseSearchStream:
    """A resumable best-first stream of matches for one query."""

    def __iter__(self) -> DenseSearchStream: ...
    def __next__(self) -> Match: ...

class DenseIndex:
    """Vector search over equidimensional dense embeddings."""

    def __init__(self, *args: object, **keywords: object) -> NoReturn: ...
    @classmethod
    def for_ram(
        cls,
        dimensions: int,
        metric: MetricName = "angular",
        quantization: ScalarName = "bf16",
        *,
        connectivity: int = 0,
        values_per_key: int = 0,
        memory_domains: Sequence[int] | None = None,
    ) -> DenseIndex:
        """Starts an empty, mutable graph in host memory, one shard per named NUMA node."""
        ...

    @classmethod
    def for_vram(
        cls,
        dimensions: int,
        metric: MetricName = "angular",
        quantization: ScalarName = "bf16",
        *,
        connectivity: int = 0,
        values_per_key: int = 0,
        devices: Sequence[int] | None = None,
    ) -> DenseIndex:
        """Starts an empty, mutable graph on CUDA devices, one shard per named ordinal."""
        ...

    @classmethod
    def load_ram(
        cls,
        path: PathLike,
        /,
        *,
        executor: Executor | None = None,
        memory_domains: Sequence[int] | None = None,
        values_per_key: int = 0,
        progress: Callable[[int, int], bool] | None = None,
    ) -> DenseIndex:
        """Copies a checkpoint into a new, mutable index in host memory."""
        ...

    @classmethod
    def load_ram_buffer(
        cls,
        buffer: Buffer,
        /,
        *,
        executor: Executor | None = None,
        memory_domains: Sequence[int] | None = None,
        values_per_key: int = 0,
        progress: Callable[[int, int], bool] | None = None,
    ) -> DenseIndex:
        """Copies a monolith out of a buffer into a new, mutable index in host memory."""
        ...

    @classmethod
    def load_vram(
        cls,
        path: PathLike,
        /,
        *,
        executor: Executor | None = None,
        devices: Sequence[int] | None = None,
        values_per_key: int = 0,
        progress: Callable[[int, int], bool] | None = None,
    ) -> DenseIndex:
        """Copies a checkpoint into a new, mutable index on CUDA devices."""
        ...

    @classmethod
    def load_vram_buffer(
        cls,
        buffer: Buffer,
        /,
        *,
        executor: Executor | None = None,
        devices: Sequence[int] | None = None,
        values_per_key: int = 0,
        progress: Callable[[int, int], bool] | None = None,
    ) -> DenseIndex:
        """Copies a monolith out of a buffer into a new, mutable index on CUDA devices."""
        ...

    @classmethod
    def view_ram(cls, path: PathLike, /) -> DenseIndex:
        """Serves a checkpoint in place through a memory mapping, as a read-only index."""
        ...

    @classmethod
    def view_nvme(
        cls,
        path: PathLike,
        /,
        *,
        executor: Executor | None = None,
        paths: Sequence[PathLike] | None = None,
    ) -> DenseIndex:
        """Serves a checkpoint in place from NVMe drives, as a read-only index."""
        ...

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
    def capabilities(self) -> str: ...
    def describe(self, *, executor: Executor | None = None) -> DenseDescription: ...
    def reserve(self, capacity: int, /, *, executor: Executor | None = None) -> None: ...
    def clear(self, *, executor: Executor | None = None) -> None: ...
    def add(
        self,
        keys: Sequence[KeyLike] | Tensor,
        vectors: Tensor,
        /,
        *,
        executor: Executor | None = None,
        expansion_factor: int = 0,
        dtype: ScalarName | None = None,
        progress: Callable[[int, int], bool] | None = None,
        stats: Stats | None = None,
    ) -> None: ...
    def refine(
        self,
        *,
        executor: Executor | None = None,
        expansion_factor: int = 0,
        progress: Callable[[int, int], bool] | None = None,
        stats: Stats | None = None,
    ) -> int: ...
    def health(self, *, executor: Executor | None = None) -> dict[str, int]: ...
    def search(
        self,
        queries: Tensor,
        wanted: int,
        /,
        *,
        executor: Executor | None = None,
        expansion_factor: int = 0,
        dtype: ScalarName | None = None,
        predicate: Callable[[Key], bool] | None = None,
        progress: Callable[[int, int], bool] | None = None,
        stats: Stats | None = None,
        out: Matches | None = None,
    ) -> Matches: ...
    def multiply(
        self,
        rows: Tensor,
        wanted: int,
        /,
        *,
        executor: Executor | None = None,
        expansion_factor: int = 0,
        dtype: ScalarName | None = None,
        progress: Callable[[int, int], bool] | None = None,
        stats: Stats | None = None,
        arena: Arena | None = None,
    ) -> DenseProducts: ...
    def attend[OutputT: Tensor](
        self,
        queries: Tensor,
        values: Tensor,
        /,
        *,
        executor: Executor | None = None,
        expansion_factor: int = 0,
        score_scale: float = 0.0,
        mass_tolerance: float = 0.0,
        dtype: ScalarName | None = None,
        values_dtype: ScalarName | None = None,
        outputs_dtype: ScalarName | None = None,
        carry: Tensor | None = None,
        progress: Callable[[int, int], bool] | None = None,
        stats: Stats | None = None,
        out: OutputT | None = None,
    ) -> memoryview | OutputT: ...
    def maxsim(
        self,
        queries: Tensor,
        rows_per_query: int,
        wanted: int,
        /,
        *,
        executor: Executor | None = None,
        expansion_factor: int = 0,
        dtype: ScalarName | None = None,
        progress: Callable[[int, int], bool] | None = None,
        stats: Stats | None = None,
        out: Matches | None = None,
    ) -> Matches: ...
    def count(self, key: KeyLike, /, *, executor: Executor | None = None) -> int: ...
    def get(
        self,
        key: KeyLike,
        /,
        *,
        executor: Executor | None = None,
        dtype: ScalarName = "f32",
        arena: Arena | None = None,
    ) -> memoryview | None: ...
    def merge(
        self,
        donor: DenseIndex,
        /,
        *,
        executor: Executor | None = None,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None: ...
    def search_stream(
        self,
        query: Tensor,
        /,
        *,
        executor: Executor | None = None,
        expansion_factor: int = 0,
        dtype: ScalarName | None = None,
    ) -> DenseSearchStream: ...
    def cursor(self, size: int = 1024, *, dtype: ScalarName | None = None) -> DenseCursor: ...
    def save(
        self,
        path: PathLike,
        /,
        *,
        executor: Executor | None = None,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None: ...
    def save_buffer(
        self,
        *,
        executor: Executor | None = None,
        progress: Callable[[int, int], bool] | None = None,
        arena: Arena | None = None,
    ) -> memoryview: ...
    @staticmethod
    def metadata_read(path: PathLike, /, *, executor: Executor | None = None) -> DenseMetadata: ...
    @staticmethod
    def metadata_read_buffer(buffer: Buffer, /, *, executor: Executor | None = None) -> DenseMetadata: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: KeyLike) -> bool: ...
    def __getitem__(self, key: KeyLike) -> memoryview: ...
    def __iter__(self) -> DenseCursor: ...
    def __repr__(self) -> str: ...

# endregion: Dense Index
