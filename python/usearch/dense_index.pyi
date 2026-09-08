"""Type stubs for the dense index and the walks it hands out."""

from collections.abc import Callable, Sequence
from typing import Literal, NoReturn, TypedDict

from usearch.types import Executor, Key, KeyLike, Match, Matches, PathLike, ScalarKind, ScalarName, Stats, Tensor

# region: Dense Index

type MetricName = Literal["inner", "angular", "euclidean", "hamming", "jaccard"]

class DenseMetadata(TypedDict):
    """The geometry keywords a checkpoint fixes; residency is never among them."""

    dimensions: int
    metric: MetricName
    quantization: ScalarKind
    connectivity: int
    values_per_key: int

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

    def __init__(self, *args: object, **keywords: object) -> NoReturn: ...
    @classmethod
    def for_ram(
        cls,
        dimensions: int = 0,
        metric: MetricName = "angular",
        quantization: ScalarName = "bf16",
        *,
        connectivity: int = 0,
        values_per_key: int = 0,
        memory_domains: Sequence[int] | None = None,
    ) -> DenseIndex:
        """Places the graph in host memory, one shard per named NUMA node."""
        ...

    @classmethod
    def for_vram(
        cls,
        dimensions: int = 0,
        metric: MetricName = "angular",
        quantization: ScalarName = "bf16",
        *,
        connectivity: int = 0,
        values_per_key: int = 0,
        devices: Sequence[int] | None = None,
    ) -> DenseIndex:
        """Places the graph on CUDA devices, one shard per named ordinal."""
        ...

    @classmethod
    def for_nvme(
        cls,
        dimensions: int = 0,
        metric: MetricName = "angular",
        quantization: ScalarName = "bf16",
        *,
        connectivity: int = 0,
        values_per_key: int = 0,
        paths: Sequence[PathLike] | None = None,
    ) -> DenseIndex:
        """Places the graph on disk, one shard per named path."""
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
    def storage_requirements(self) -> int: ...
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
        executor: Executor,
        expansion_factor: int = 0,
        dtype: ScalarName | None = None,
        stats: Stats | None = None,
    ) -> int: ...
    def refine(
        self,
        *,
        executor: Executor,
        expansion_factor: int = 0,
        stats: Stats | None = None,
    ) -> int: ...
    def search(
        self,
        queries: Tensor,
        wanted: int,
        /,
        *,
        executor: Executor,
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
        executor: Executor,
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
        executor: Executor,
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
        executor: Executor,
        expansion_factor: int = 0,
        score_scale: float = 0.0,
        mass_tolerance: float = 0.0,
        dtype: ScalarName | None = None,
        values_dtype: ScalarName | None = None,
        outputs_dtype: ScalarName | None = None,
        carry: Tensor | None = None,
        stats: Stats | None = None,
    ) -> None: ...
    def maxsim(
        self,
        queries: Tensor,
        rows_per_query: int,
        wanted: int,
        /,
        *,
        executor: Executor,
        expansion_factor: int = 0,
        dtype: ScalarName | None = None,
        stats: Stats | None = None,
    ) -> Matches: ...
    def count(self, key: KeyLike, /) -> int: ...
    def get(self, key: KeyLike, /, *, dtype: ScalarName = "f32") -> memoryview | None: ...
    def merge(self, donor: DenseIndex, /, *, executor: Executor) -> None: ...
    def find(self, query: Tensor, /, *, executor: Executor, dtype: ScalarName | None = None) -> DenseFinder: ...
    def batches(self, size: int = 1024, *, dtype: ScalarName | None = None) -> DenseCursor: ...
    def save(self, target: PathLike | Tensor, /, *, executor: Executor) -> None: ...
    def load(self, source: PathLike | Tensor, /, *, executor: Executor) -> None: ...
    @staticmethod
    def metadata(source: PathLike | Tensor, /) -> DenseMetadata: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: KeyLike) -> bool: ...
    def __getitem__(self, key: KeyLike) -> memoryview: ...
    def __iter__(self) -> DenseCursor: ...
    def __repr__(self) -> str: ...

# endregion: Dense Index
