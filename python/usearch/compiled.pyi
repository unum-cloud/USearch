from __future__ import annotations

from collections.abc import Callable

import numpy

# Module-level constants
DEFAULT_CONNECTIVITY: int
DEFAULT_EXPANSION_ADD: int
DEFAULT_EXPANSION_SEARCH: int
USES_OPENMP: int
USES_NUMKONG: int
USES_NUMKONG_DYNAMIC_DISPATCH: int
USES_SIMSIMD: int
USES_SIMSIMD_DYNAMIC_DISPATCH: int
VERSION_MAJOR: int
VERSION_MINOR: int
VERSION_PATCH: int

class MetricKind(int):
    Unknown: MetricKind
    IP: MetricKind
    Cos: MetricKind
    L2sq: MetricKind
    Haversine: MetricKind
    Divergence: MetricKind
    Pearson: MetricKind
    Jaccard: MetricKind
    Hamming: MetricKind
    Tanimoto: MetricKind
    Sorensen: MetricKind
    Cosine: MetricKind  # alias for Cos
    InnerProduct: MetricKind  # alias for IP

class ScalarKind(int):
    Unknown: ScalarKind
    F64: ScalarKind
    F32: ScalarKind
    BF16: ScalarKind
    F16: ScalarKind
    E5M2: ScalarKind
    E4M3: ScalarKind
    E3M2: ScalarKind
    E2M3: ScalarKind
    I8: ScalarKind
    U8: ScalarKind
    B1: ScalarKind
    U40: ScalarKind
    UUID: ScalarKind
    U64: ScalarKind
    U32: ScalarKind
    U16: ScalarKind
    I64: ScalarKind
    I32: ScalarKind
    I16: ScalarKind

class MetricSignature(int):
    ArrayArray: MetricSignature
    ArrayArraySize: MetricSignature

class IndexStats:
    nodes: int
    edges: int
    max_edges: int
    allocated_bytes: int

class Index:
    def __init__(
        self,
        *,
        ndim: int = 0,
        dtype: ScalarKind = ...,
        connectivity: int = ...,
        expansion_add: int = ...,
        expansion_search: int = ...,
        metric_kind: MetricKind = ...,
        metric_signature: MetricSignature = ...,
        metric_pointer: int = 0,
        multi: bool = False,
        enable_key_lookups: bool = True,
    ) -> None: ...
    def add_many(
        self,
        keys: numpy.ndarray,
        vectors: numpy.ndarray,
        *,
        copy: bool = True,
        threads: int = 0,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None: ...
    def search_many(
        self,
        queries: numpy.ndarray,
        count: int = 10,
        exact: bool = False,
        threads: int = 0,
        progress: Callable[[int, int], bool] | None = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, int, int]: ...
    def cluster_vectors(
        self,
        queries: numpy.ndarray,
        min_count: int = 0,
        max_count: int = 0,
        threads: int = 0,
        progress: Callable[[int, int], bool] | None = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, int, int]: ...
    def cluster_keys(
        self,
        queries: numpy.ndarray,
        min_count: int = 0,
        max_count: int = 0,
        threads: int = 0,
        progress: Callable[[int, int], bool] | None = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, int, int]: ...
    def rename_one_to_one(self, from_: int, to: int) -> bool: ...
    def rename_many_to_many(self, from_: list[int], to: list[int]) -> list[bool]: ...
    def rename_many_to_one(self, from_: list[int], to: int) -> list[bool]: ...
    def remove_one(self, key: int, compact: bool, threads: int) -> bool: ...
    def remove_many(self, key: list[int], compact: bool, threads: int) -> int: ...
    def contains_one(self, key: int) -> bool: ...
    def contains_many(self, keys: numpy.ndarray) -> numpy.ndarray: ...
    def count_one(self, key: int) -> int: ...
    def count_many(self, keys: numpy.ndarray) -> numpy.ndarray: ...
    def get_many(self, keys: numpy.ndarray, dtype: ScalarKind = ...) -> object: ...
    def get_keys_in_slice(self, offset: int = 0, limit: int = ...) -> numpy.ndarray: ...
    def get_keys_at_offsets(self, offsets: numpy.ndarray) -> numpy.ndarray: ...
    def get_key_at_offset(self, offset: int) -> int: ...
    def save_index_to_path(
        self,
        path: str,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None: ...
    def load_index_from_path(
        self,
        path: str,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None: ...
    def view_index_from_path(
        self,
        path: str,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None: ...
    def save_index_to_buffer(
        self,
        progress: Callable[[int, int], bool] | None = None,
    ) -> bytearray: ...
    def load_index_from_buffer(
        self,
        buffer_obj: object,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None: ...
    def view_index_from_buffer(
        self,
        buffer_obj: object,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None: ...
    def reset(self) -> None: ...
    def clear(self) -> None: ...
    def copy(self, *, copy: bool = True) -> Index: ...
    def compact(
        self,
        threads: int,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None: ...
    def join(
        self,
        other: Index,
        max_proposals: int = 0,
        exact: bool = False,
        progress: Callable[[int, int], bool] | None = None,
    ) -> dict[int, int]: ...
    def change_metric(
        self,
        metric_kind: MetricKind = ...,
        metric_signature: MetricSignature = ...,
        metric_pointer: int = 0,
    ) -> None: ...
    def pairwise_distances(
        self,
        left: numpy.ndarray,
        right: numpy.ndarray,
    ) -> numpy.ndarray: ...
    def pairwise_distance(self, left: int, right: int) -> float: ...
    def level_stats(self, level: int) -> IndexStats: ...
    def __len__(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> ScalarKind: ...
    @property
    def connectivity(self) -> int: ...
    @property
    def capacity(self) -> int: ...
    @property
    def multi(self) -> bool: ...
    @property
    def serialized_length(self) -> int: ...
    @property
    def memory_usage(self) -> int: ...
    @property
    def expansion_add(self) -> int: ...
    @expansion_add.setter
    def expansion_add(self, value: int) -> None: ...
    @property
    def expansion_search(self) -> int: ...
    @expansion_search.setter
    def expansion_search(self, value: int) -> None: ...
    @property
    def hardware_acceleration(self) -> str: ...
    @property
    def max_level(self) -> int: ...
    @property
    def stats(self) -> IndexStats: ...
    @property
    def levels_stats(self) -> list[IndexStats]: ...

class Indexes:
    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    def merge(self, shard: Index) -> None: ...
    def merge_paths(
        self,
        paths: list[str],
        view: bool = True,
        threads: int = 0,
    ) -> None: ...
    def search_many(
        self,
        query: numpy.ndarray,
        count: int = 10,
        exact: bool = False,
        threads: int = 0,
        progress: Callable[[int, int], bool] | None = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, int, int]: ...

def exact_search(
    dataset: numpy.ndarray,
    queries: numpy.ndarray,
    count: int = 10,
    *,
    threads: int = 0,
    metric_kind: MetricKind = ...,
    metric_signature: MetricSignature = ...,
    metric_pointer: int = 0,
    progress: Callable[[int, int], bool] | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, int, int]: ...
def kmeans(
    dataset: numpy.ndarray,
    count: int = 10,
    *,
    max_iterations: int = ...,
    inertia_threshold: float = ...,
    max_seconds: float = ...,
    min_shifts: float = ...,
    seed: int = 0,
    threads: int = 0,
    dtype: ScalarKind = ...,
    metric_kind: MetricKind = ...,
    progress: Callable[[int, int], bool] | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: ...
def hardware_acceleration(
    *,
    dtype: ScalarKind = ...,
    ndim: int = 0,
    metric_kind: MetricKind = ...,
) -> str: ...
def hardware_acceleration_compiled() -> str: ...
def hardware_acceleration_available() -> str: ...
def index_dense_metadata_from_path(path: str) -> dict: ...
def index_dense_metadata_from_buffer(buffer: object) -> dict: ...
