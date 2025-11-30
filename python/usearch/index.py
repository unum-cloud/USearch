from __future__ import annotations
from inspect import signature
from collections.abc import Sequence

# The purpose of this file is to provide Pythonic wrapper on top
# the native precompiled CPython module. It improves compatibility
# Python tooling, linters, and static analyzers. It also embeds JIT
# into the primary `Index` class, connecting USearch with Numba.
import os
import sys
import math
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Union,
    NamedTuple,
    List,
    Iterable,
    Tuple,
    Dict,
    Callable,
)

import numpy as np
from tqdm import tqdm

# Precompiled symbols that won't be exposed directly:
from usearch.compiled import (
    Index as _CompiledIndex,
    Indexes as _CompiledIndexes,
    IndexStats as _CompiledIndexStats,
    index_dense_metadata_from_path as _index_dense_metadata_from_path,
    index_dense_metadata_from_buffer as _index_dense_metadata_from_buffer,
    exact_search as _exact_search,
    hardware_acceleration as _hardware_acceleration,
    kmeans as _kmeans,
)

# Precompiled symbols that will be exposed
from usearch.compiled import (
    MetricKind,
    ScalarKind,
    MetricSignature,
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
    USES_OPENMP,
    USES_SIMSIMD,
    USES_FP16LIB,
)

MetricKindBitwise = (
    MetricKind.Hamming,
    MetricKind.Tanimoto,
    MetricKind.Sorensen,
)


class CompiledMetric(NamedTuple):
    pointer: int
    kind: MetricKind
    signature: MetricSignature


# Define TypeAlias for older Python versions
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    TypeAlias = object  # Fallback for older Python versions

Key: TypeAlias = np.uint64

NoneType: TypeAlias = type(None)

KeyOrKeysLike = Union[Key, Iterable[Key], int, Iterable[int], np.ndarray, memoryview]

VectorOrVectorsLike = Union[np.ndarray, Iterable[np.ndarray], memoryview]

DTypeLike = Union[str, ScalarKind]

MetricLike = Union[str, MetricKind, CompiledMetric]

BytesLike = Union[bytes, bytearray, memoryview]

PathOrBuffer = Union[str, os.PathLike, BytesLike]

ProgressCallback = Callable[[int, int], bool]


def _match_signature(func: Callable[[Any], Any], arg_types: List[type], ret_type: type) -> bool:
    assert callable(func), "Not callable"
    sig = signature(func)
    param_types = [param.annotation for param in sig.parameters.values()]
    return param_types == arg_types and sig.return_annotation == ret_type


def _normalize_dtype(
    dtype,
    ndim: int = 0,
    metric: MetricKind = MetricKind.Cos,
) -> ScalarKind:
    if dtype is None or dtype == "":
        if metric in MetricKindBitwise:
            return ScalarKind.B1
        if _hardware_acceleration(dtype=ScalarKind.BF16, ndim=ndim, metric_kind=metric):
            return ScalarKind.BF16
        if _hardware_acceleration(dtype=ScalarKind.F16, ndim=ndim, metric_kind=metric):
            return ScalarKind.F16
        return ScalarKind.F32

    if isinstance(dtype, ScalarKind):
        return dtype

    if isinstance(dtype, str):
        dtype = dtype.lower()

    _normalize = {
        "f64": ScalarKind.F64,
        "f32": ScalarKind.F32,
        "bf16": ScalarKind.BF16,
        "f16": ScalarKind.F16,
        "i8": ScalarKind.I8,
        "b1": ScalarKind.B1,
        "b1x8": ScalarKind.B1,
        "bits": ScalarKind.B1,
        "float64": ScalarKind.F64,
        "float32": ScalarKind.F32,
        "bfloat16": ScalarKind.BF16,
        "float16": ScalarKind.F16,
        "int8": ScalarKind.I8,
        np.float64: ScalarKind.F64,
        np.float32: ScalarKind.F32,
        np.float16: ScalarKind.F16,
        np.int8: ScalarKind.I8,
        np.uint8: ScalarKind.B1,
    }
    return _normalize[dtype]


def _to_numpy_dtype(dtype: ScalarKind):
    if dtype == ScalarKind.BF16:
        return None
    _normalize = {
        ScalarKind.F64: np.float64,
        ScalarKind.F32: np.float32,
        ScalarKind.F16: np.float16,
        ScalarKind.I8: np.int8,
        ScalarKind.B1: np.uint8,
    }
    if dtype in _normalize.values():
        return dtype
    return _normalize[dtype]


def _normalize_metric(metric) -> MetricKind:
    if metric is None:
        return MetricKind.Cos

    if isinstance(metric, str):
        _normalize = {
            "cos": MetricKind.Cos,
            "cosine": MetricKind.Cos,
            "ip": MetricKind.IP,
            "dot": MetricKind.IP,
            "inner_product": MetricKind.IP,
            "l2sq": MetricKind.L2sq,
            "l2_sq": MetricKind.L2sq,
            "haversine": MetricKind.Haversine,
            "divergence": MetricKind.Divergence,
            "pearson": MetricKind.Pearson,
            "hamming": MetricKind.Hamming,
            "tanimoto": MetricKind.Tanimoto,
            "sorensen": MetricKind.Sorensen,
        }
        return _normalize[metric.lower()]

    return metric


def _is_buffer(obj: Any) -> bool:
    """Check if the object is a buffer-like object.
    More portable than `hasattr(obj, "__buffer__")`, which requires Python 3.11+."""
    try:
        memoryview(obj)
        return True
    except TypeError:
        return False


def _search_in_compiled(
    compiled_callable: Callable,
    vectors: np.ndarray,
    *,
    log: Union[str, bool],
    progress: Optional[ProgressCallback],
    **kwargs,
) -> Union[Matches, BatchMatches]:
    #
    assert isinstance(vectors, np.ndarray), "Expects a NumPy array"
    assert vectors.ndim == 1 or vectors.ndim == 2, "Expects a matrix or vector"
    assert not progress or _match_signature(progress, [int, int], bool), "Invalid callback"

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, len(vectors))
    count_vectors = vectors.shape[0]

    def distill_batch(
        batch_matches: BatchMatches,
    ) -> Union[BatchMatches, Matches]:
        return batch_matches[0] if count_vectors == 1 else batch_matches

    progress_callback = progress

    # Create progress bar if needed
    if log:
        name = log if isinstance(log, str) else "Search"
        progress_bar = tqdm(
            desc=name,
            total=count_vectors,
            unit="vector",
        )

        user_progress = progress

        def update_progress_bar(processed: int, total: int) -> bool:
            progress_bar.update(processed - progress_bar.n)
            if user_progress:
                return user_progress(processed, total)
            return True

        progress_callback = update_progress_bar

    if progress_callback:
        tuple_ = compiled_callable(vectors, progress=progress_callback, **kwargs)
    else:
        tuple_ = compiled_callable(vectors, **kwargs)

    if log:
        progress_bar.close()

    return distill_batch(BatchMatches(*tuple_))


def _add_to_compiled(
    compiled,
    *,
    keys,
    vectors,
    copy: bool,
    threads: int,
    log: Union[str, bool],
    progress: Optional[ProgressCallback],
) -> Union[int, np.ndarray]:
    #
    assert isinstance(vectors, np.ndarray), "Expects a NumPy array"
    assert not progress or _match_signature(progress, [int, int], bool), "Invalid callback"
    assert vectors.ndim == 1 or vectors.ndim == 2, "Expects a matrix or vector"
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, len(vectors))

    # Validate or generate the keys
    count_vectors = vectors.shape[0]
    generate_labels = keys is None
    if generate_labels:
        start_id = len(compiled)
        keys = np.arange(start_id, start_id + count_vectors, dtype=Key)
    else:
        if not isinstance(keys, Iterable):
            assert count_vectors == 1, "Each vector must have a key"
            keys = [keys]
        keys = np.array(keys).astype(Key)

    assert len(keys) == count_vectors

    # Create progress bar if needed
    if log:
        name = log if isinstance(log, str) else "Add"
        progress_bar = tqdm(
            desc=name,
            total=count_vectors,
            unit="vector",
        )

        def update_progress_bar(processed: int, total: int) -> bool:
            progress_bar.update(processed - progress_bar.n)
            return progress(processed, total) if progress else True

        compiled.add_many(
            keys,
            vectors,
            copy=copy,
            threads=threads,
            progress=update_progress_bar,
        )
        progress_bar.close()
    else:
        compiled.add_many(keys, vectors, copy=copy, threads=threads, progress=progress)

    return keys


@dataclass
class Match:
    """Single search result with key and distance."""

    key: int
    distance: float

    def to_tuple(self) -> tuple:
        return self.key, self.distance


@dataclass
class Matches:
    """Search results for a single query."""

    keys: np.ndarray
    distances: np.ndarray

    visited_members: int = 0
    computed_distances: int = 0

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Match:
        if isinstance(index, int) and index < len(self):
            return Match(
                key=self.keys[index],
                distance=self.distances[index],
            )
        else:
            raise IndexError(f"`index` must be an integer under {len(self)}")

    def to_list(self) -> List[tuple]:
        """Convert to list of (key, distance) tuples."""
        return [(int(key), float(distance)) for key, distance in zip(self.keys, self.distances)]

    def __repr__(self) -> str:
        return f"usearch.Matches({len(self)})"


@dataclass
class BatchMatches(Sequence):
    """Search results for multiple queries in batch operations.

    Unused positions in arrays contain sentinel values (default keys, max distances).
    Access individual results via indexing: batch_matches[i] returns valid matches only.

    Attributes:
        keys: 2D array of shape (n_queries, k) containing match keys
        distances: 2D array of shape (n_queries, k) containing distances  
        counts: 1D array of shape (n_queries,) with actual number of matches per query
        visited_members: Total graph nodes visited during search
        computed_distances: Total distance computations performed
    """

    keys: np.ndarray
    distances: np.ndarray
    counts: np.ndarray

    visited_members: int = 0
    computed_distances: int = 0

    def __len__(self) -> int:
        return len(self.counts)

    def __getitem__(self, index: int) -> Matches:
        if isinstance(index, int) and index < len(self):
            return Matches(
                keys=self.keys[index, : self.counts[index]],
                distances=self.distances[index, : self.counts[index]],
                visited_members=self.visited_members // len(self),
                computed_distances=self.computed_distances // len(self),
            )
        else:
            raise IndexError(f"`index` must be an integer under {len(self)}")

    def to_list(self) -> List[List[tuple]]:
        """Flatten matches for all queries into a list of `(key, distance)` tuples."""
        list_of_matches = [self.__getitem__(row) for row in range(self.__len__())]
        return [match.to_tuple() for matches in list_of_matches for match in matches]

    def mean_recall(self, expected: np.ndarray, count: Optional[int] = None) -> float:
        """Measures recall [0, 1] as of `Matches` that contain the corresponding
        `expected` entry anywhere among results."""
        return self.count_matches(expected, count=count) / len(expected)

    def count_matches(self, expected: np.ndarray, count: Optional[int] = None) -> int:
        """Measures recall [0, len(expected)] as of `Matches` that contain the corresponding
        `expected` entry anywhere among results.
        """
        assert len(expected) == len(self)
        recall = 0
        if count is None:
            count = self.keys.shape[1]

        if count == 1:
            recall = np.sum(self.keys[:, 0] == expected)
        else:
            for i in range(len(self)):
                recall += expected[i] in self.keys[i, :count]
        return recall

    def __repr__(self) -> str:
        return f"usearch.BatchMatches({np.sum(self.counts)} across {len(self)} queries)"


@dataclass
class Clustering:
    def __init__(
        self,
        index: Index,
        matches: BatchMatches,
        queries: Optional[np.ndarray] = None,
    ) -> None:
        if queries is None:
            queries = index._compiled.get_keys_in_slice()
        self.index = index
        self.queries = queries
        self.matches = matches

    def __repr__(self) -> str:
        return f"usearch.Clustering(for {len(self.queries)} queries)"

    @property
    def centroids_popularity(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.unique(self.matches.keys, return_counts=True)

    def members_of(self, centroid: Key) -> np.ndarray:
        return self.queries[self.matches.keys.flatten() == centroid]

    def subcluster(self, centroid: Key, **clustering_kwargs) -> Clustering:
        sub_keys = self.members_of(centroid)
        return self.index.cluster(keys=sub_keys, **clustering_kwargs)

    def plot_centroids_popularity(self):
        from matplotlib import pyplot as plt

        _, sizes = self.centroids_popularity
        plt.yscale("log")
        plt.plot(sorted(sizes), np.arange(len(sizes)))
        plt.show()

    @property
    def network(self):
        import networkx as nx

        keys, sizes = self.centroids_popularity

        g = nx.Graph()
        for key, size in zip(keys, sizes):
            g.add_node(key, size=size)

        for i, i_key in enumerate(keys):
            for j_key in keys[:i]:
                d = self.index.pairwise_distance(i_key, j_key)
                g.add_edge(i_key, j_key, distance=d)

        return g


class IndexedKeys(Sequence):
    """View of all keys in the index."""

    def __init__(self, index: Index) -> None:
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self,
        offset_offsets_or_slice: Union[int, np.ndarray, slice],
    ) -> Union[Key, np.ndarray]:
        if isinstance(offset_offsets_or_slice, slice):
            start, stop, step = offset_offsets_or_slice.indices(len(self))
            if step != 1:
                raise ValueError("Slicing with a step is not supported")
            return self.index._compiled.get_keys_in_slice(start, stop - start)

        elif isinstance(offset_offsets_or_slice, Iterable):
            offsets = np.array(offset_offsets_or_slice)
            return self.index._compiled.get_keys_at_offsets(offsets)

        else:
            offset = int(offset_offsets_or_slice)
            if offset < 0:
                offset += len(self)
            if offset < 0 or offset >= len(self):
                raise IndexError("Index out of range")
            return self.index._compiled.get_key_at_offset(offset)

    def __array__(self, dtype=None) -> np.ndarray:
        if dtype is None:
            dtype = Key
        return self.index._compiled.get_keys_in_slice().astype(dtype)


class Index:
    """Fast approximate nearest neighbor search for dense vectors.

    Supports various distance metrics (cosine, euclidean, inner product, etc.) 
    and automatic precision optimization. Vector keys must be integers.
    All vectors must have the same dimensionality.

    Example:
        >>> index = Index(ndim=128, metric='cos')
        >>> index.add(key=42, vector=np.random.rand(128))
        >>> matches = index.search(query_vector, count=10)
    """

    def __init__(
        self,
        *,  # All arguments must be named
        ndim: int = 0,
        metric: MetricLike = MetricKind.Cos,
        dtype: Optional[DTypeLike] = None,
        connectivity: Optional[int] = None,
        expansion_add: Optional[int] = None,
        expansion_search: Optional[int] = None,
        multi: bool = False,
        path: Optional[os.PathLike] = None,
        view: bool = False,
        enable_key_lookups: bool = True,
    ) -> None:
        """Construct the index and compiles the functions, if requested (expensive).

        :param ndim: Number of vector dimensions
        :type ndim: int
            Required for some metrics, pre-set for others.
            Haversine, for example, only applies to 2-dimensional latitude/longitude
            coordinates. Angular (Cos) and Euclidean (L2sq), obviously, apply to
            vectors with arbitrary number of dimensions.

        :param metric: Distance function
        :type metric: MetricLike, defaults to MetricKind.Cos
            Kind of the distance function, or the Numba `cfunc` JIT-compiled object.
            Possible `MetricKind` values: IP, Cos, L2sq, Haversine, Pearson,
            Hamming, Tanimoto, Sorensen.

        :param dtype: Scalar type for internal vector storage
        :type dtype: Optional[DTypeLike], defaults to None
            For continuous metrics can be: f16, f32, f64, or i8.
            For bitwise metrics it's implementation-defined, and can't change.
            If nothing is provided, the optimal data type is selected based on the metric
            kind and hardware support.
            Example: you can use the `f16` index with `f32` vectors in Euclidean space,
            which will be automatically downcasted. Moreover, if `dtype=None` is passed,
            and hardware supports `f16` SIMD-instructions, this choice will be done for you.
            You can later double-check the used representation with `index.dtype`.

        :param connectivity: Connections per node in HNSW
        :type connectivity: Optional[int], defaults to None
            Hyper-parameter for the number of Graph connections
            per layer of HNSW. The original paper calls it "M".
            Optional, but can't be changed after construction.

        :param expansion_add: Traversal depth on insertions
        :type expansion_add: Optional[int], defaults to None
            Hyper-parameter for the search depth when inserting new
            vectors. The original paper calls it "efConstruction".
            Can be changed afterwards, as the `.expansion_add`.

        :param expansion_search: Traversal depth on queries
        :type expansion_search: Optional[int], defaults to None
            Hyper-parameter for the search depth when querying
            nearest neighbors. The original paper calls it "ef".
            Can be changed afterwards, as the `.expansion_search`.

        :param multi: Allow multiple vectors with the same key
        :type multi: bool, defaults to True
        :param path: Where to store the index
        :type path: Optional[os.PathLike], defaults to None
        :param view: Are we simply viewing an immutable index
        :type view: bool, defaults to False
        """

        if connectivity is None:
            connectivity = DEFAULT_CONNECTIVITY
        if expansion_add is None:
            expansion_add = DEFAULT_EXPANSION_ADD
        if expansion_search is None:
            expansion_search = DEFAULT_EXPANSION_SEARCH

        assert isinstance(connectivity, int), "Expects integer `connectivity`"
        assert isinstance(expansion_add, int), "Expects integer `expansion_add`"
        assert isinstance(expansion_search, int), "Expects integer `expansion_search`"

        metric = _normalize_metric(metric)
        if isinstance(metric, MetricKind):
            self._metric_kind = metric
            self._metric_jit = None
            self._metric_pointer = 0
            self._metric_signature = MetricSignature.ArrayArraySize
        elif isinstance(metric, CompiledMetric):
            self._metric_jit = metric
            self._metric_kind = metric.kind
            self._metric_pointer = metric.pointer
            self._metric_signature = metric.signature
        else:
            raise ValueError("The `metric` must be a `CompiledMetric` or a `MetricKind`")

        # Validate, that the right scalar type is defined
        dtype = _normalize_dtype(dtype, ndim, self._metric_kind)
        self._compiled = _CompiledIndex(
            ndim=ndim,
            dtype=dtype,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            multi=multi,
            enable_key_lookups=enable_key_lookups,
            metric_kind=self._metric_kind,
            metric_pointer=self._metric_pointer,
            metric_signature=self._metric_signature,
        )

        self.path = path
        if path is not None and os.path.exists(path):
            if view:
                self.view(path)
            else:
                self.load(path)

    @staticmethod
    def metadata(path_or_buffer: PathOrBuffer) -> Optional[dict]:
        try:
            if _is_buffer(path_or_buffer):
                return _index_dense_metadata_from_buffer(path_or_buffer)
            else:
                path_or_buffer = os.fspath(path_or_buffer)
                if not os.path.exists(path_or_buffer):
                    return None
                return _index_dense_metadata_from_path(path_or_buffer)
        except Exception as e:
            raise e

    @staticmethod
    def restore(path_or_buffer: PathOrBuffer, view: bool = False, **kwargs) -> Optional[Index]:
        meta = Index.metadata(path_or_buffer)
        if not meta:
            return None

        index = Index(
            ndim=meta["dimensions"],
            dtype=meta["kind_scalar"],
            metric=meta["kind_metric"],
            **kwargs,
        )

        if view:
            index.view(path_or_buffer)
        else:
            index.load(path_or_buffer)
        return index

    def __len__(self) -> int:
        return self._compiled.__len__()

    def add(
        self,
        keys: KeyOrKeysLike,
        vectors: VectorOrVectorsLike,
        *,
        copy: bool = True,
        threads: int = 0,
        log: Union[str, bool] = False,
        progress: Optional[ProgressCallback] = None,
    ) -> Union[int, np.ndarray]:
        """Inserts one or move vectors into the index.

        For maximal performance the `keys` and `vectors`
        should conform to the Python's "buffer protocol" spec.

        To index a single entry:
            keys: int, vectors: np.ndarray.
        To index many entries:
            keys: np.ndarray, vectors: np.ndarray.

        When working with extremely large indexes, you may want to
        pass `copy=False`, if you can guarantee the lifetime of the
        primary vectors store during the process of construction.

        :param keys: Unique identifier(s) for passed vectors
        :type keys: Optional[KeyOrKeysLike], can be `None`
        :param vectors: Vector or a row-major matrix
        :type vectors: VectorOrVectorsLike
        :param copy: Should the index store a copy of vectors
        :type copy: bool, defaults to True
        :param threads: Optimal number of cores to use
        :type threads: int, defaults to 0
        :param log: Whether to print the progress bar
        :type log: Union[str, bool], defaults to False
        :param progress: Callback to report stats of the progress and control it
        :type progress: Optional[ProgressCallback], defaults to None
        :return: Inserted key or keys
        :type: Union[int, np.ndarray]
        """
        return _add_to_compiled(
            self._compiled,
            keys=keys,
            vectors=vectors,
            copy=copy,
            threads=threads,
            log=log,
            progress=progress,
        )

    def search(
        self,
        vectors: VectorOrVectorsLike,
        count: int = 10,
        radius: float = math.inf,
        *,
        threads: int = 0,
        exact: bool = False,
        log: Union[str, bool] = False,
        progress: Optional[ProgressCallback] = None,
    ) -> Union[Matches, BatchMatches]:
        """Performs approximate nearest neighbors search for one or more queries.
        
        When searching with batch queries, returns BatchMatches that pre-allocates arrays
        for the requested `count` size. If fewer matches exist than requested (e.g., when
        count > index size), use individual query access via batch_matches[i] to get only
        valid results, or check batch_matches.counts to see actual result counts per query.
        
        :param vectors: Query vector or vectors.
        :type vectors: VectorOrVectorsLike
        :param count: Upper count on the number of matches to find
        :type count: int, defaults to 10
            When count > index size, only available vectors will be returned.
            For BatchMatches, unused positions contain sentinel values.
        :param threads: Optimal number of cores to use
        :type threads: int, defaults to 0
        :param exact: Perform exhaustive linear-time exact search
        :type exact: bool, defaults to False
        :param log: Whether to print the progress bar, default to False
        :type log: Union[str, bool], optional
        :param progress: Callback to report stats of the progress and control it
        :type progress: Optional[ProgressCallback], defaults to None
        :return: Matches for one or more queries
        :rtype: Union[Matches, BatchMatches]
            For single queries: Matches with only valid results
            For batch queries: BatchMatches - use indexing for individual results
        """

        return _search_in_compiled(
            self._compiled.search_many,
            vectors,
            # Batch scheduling:
            log=log,
            # Search constraints:
            count=count,
            exact=exact,
            threads=threads,
            progress=progress,
        )

    def contains(self, keys: KeyOrKeysLike) -> Union[bool, np.ndarray]:
        if isinstance(keys, Iterable):
            return self._compiled.contains_many(np.array(keys, dtype=Key))
        else:
            return self._compiled.contains_one(int(keys))

    def __contains__(self, keys: KeyOrKeysLike) -> Union[bool, np.ndarray]:
        return self.contains(keys)

    def count(self, keys: KeyOrKeysLike) -> Union[int, np.ndarray]:
        if isinstance(keys, Iterable):
            return self._compiled.count_many(np.array(keys, dtype=Key))
        else:
            return self._compiled.count_one(int(keys))

    def get(
        self,
        keys: KeyOrKeysLike,
        dtype: Optional[DTypeLike] = None,
    ) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray]]]:
        """Looks up one or more keys from the `Index`, retrieving corresponding vectors.

        Returns `None`, if one key is requested, and its not present.
        Returns a (row) vector, if the key maps into a single vector.
        Returns a (row-major) matrix, if the key maps into a multiple vectors.
        If multiple keys are requested, composes many such responses into a `tuple`.

        :param keys: One or more keys to lookup
        :type keys: KeyOrKeysLike
        :return: One or more keys lookup results
        :rtype: Union[Optional[np.ndarray], Tuple[Optional[np.ndarray]]]
        """
        if not dtype:
            dtype = self.dtype
            view_dtype = _to_numpy_dtype(dtype)
            if view_dtype is None:
                dtype = ScalarKind.F32
                view_dtype = np.float32
        else:
            dtype = _normalize_dtype(dtype)
            view_dtype = _to_numpy_dtype(dtype)
            if view_dtype is None:
                raise NotImplementedError("The requested representation type is not supported by NumPy")

        def cast(result):
            if result is not None:
                return result.view(view_dtype)
            return result

        is_one = not isinstance(keys, Iterable)
        if is_one:
            keys = [keys]
        if not isinstance(keys, np.ndarray):
            keys = np.array(keys, dtype=Key)
        else:
            keys = keys.astype(Key)

        results = self._compiled.get_many(keys, dtype)
        results = cast(results) if isinstance(results, np.ndarray) else [cast(result) for result in results]
        return results[0] if is_one else results

    def __getitem__(self, keys: KeyOrKeysLike) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Looks up one or more keys from the `Index`, retrieving corresponding vectors.

        Returns `None`, if one key is requested, and its not present.
        Returns a (row) vector, if the key maps into a single vector.
        Returns a (row-major) matrix, if the key maps into a multiple vectors.
        If multiple keys are requested, composes many such responses into a `tuple`.

        :param keys: One or more keys to lookup
        :type keys: KeyOrKeysLike
        :return: One or more keys lookup results
        :rtype: Union[Optional[np.ndarray], Tuple[Optional[np.ndarray]]]
        """
        return self.get(keys)

    def remove(
        self,
        keys: KeyOrKeysLike,
        *,
        compact: bool = False,
        threads: int = 0,
    ) -> Union[int, np.ndarray]:
        """Removes one or move vectors from the index.

        When working with extremely large indexes, you may want to
        mark some entries deleted, instead of rebuilding a filtered index.
        In other cases, rebuilding - is the recommended approach.

        :param keys: Unique identifier for passed vectors, optional
        :type keys: KeyOrKeysLike
        :param compact: Removes links to removed nodes (expensive), defaults to False
        :type compact: bool, optional
        :param threads: Optimal number of cores to use, defaults to 0
        :type threads: int, optional
        :return: Array of integers for the number of removed vectors per key
        :type: Union[int, np.ndarray]
        """
        if not isinstance(keys, Iterable):
            return self._compiled.remove_one(keys, compact=compact, threads=threads)
        else:
            keys = np.array(keys, dtype=Key)
            return self._compiled.remove_many(keys, compact=compact, threads=threads)

    def __delitem__(self, keys: KeyOrKeysLike) -> Union[int, np.ndarray]:
        return self.remove(keys)

    def rename(
        self,
        from_: KeyOrKeysLike,
        to: KeyOrKeysLike,
    ) -> Union[int, np.ndarray]:
        """Rename existing member vector or vectors.

        May be used in iterative clustering procedures, where one would iteratively
        relabel every vector with the name of the cluster an entry belongs to, until
        the system converges.

        :param from_: One or more keys to be renamed
        :type from_: KeyOrKeysLike
        :param to: New name or names (of identical length as `from_`)
        :type to: KeyOrKeysLike
        :return: Number of vectors that were found and renamed
        :rtype: int
        """
        if isinstance(from_, Iterable):
            from_ = np.array(from_, dtype=Key)
            if isinstance(to, Iterable):
                to = np.array(to, dtype=Key)
                return self._compiled.rename_many_to_many(from_, to)

            else:
                return self._compiled.rename_many_to_one(from_, int(to))

        else:
            return self._compiled.rename_one_to_one(int(from_), int(to))

    @property
    def jit(self) -> bool:
        """
        :return: True, if the provided `metric` was JIT-ed
        :rtype: bool
        """
        return self._metric_jit is not None

    @property
    def hardware_acceleration(self) -> str:
        """Describes the kind of hardware-acceleration support used in this instance.

        This indicates the type of hardware acceleration that is available and
        being utilized for the current index configuration, including the metric
        kind and number of dimensions.

        :return: "auto" if no hardware acceleration is available, otherwise an ISA subset name.
        :rtype: str
        """
        return self._compiled.hardware_acceleration

    @property
    def size(self) -> int:
        """Returns the number of vectors currently indexed.

        :return: The number of vectors in the index.
        :rtype: int
        """
        return self._compiled.size

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions for vectors in the index.

        :return: The dimensionality of vectors in the index.
        :rtype: int
        """
        return self._compiled.ndim

    @property
    def serialized_length(self) -> int:
        """Returns the length in bytes required to serialize the index.

        :return: The serialized length of the index in bytes.
        :rtype: int
        """
        return self._compiled.serialized_length

    @property
    def metric_kind(self) -> Union[MetricKind, CompiledMetric]:
        """Returns the type of metric used for distance calculations.

        :return: The metric kind used in the index.
        :rtype: Union[MetricKind, CompiledMetric]
        """
        return self._metric_jit.kind if self._metric_jit else self._metric_kind

    @property
    def metric(self) -> Union[MetricKind, CompiledMetric]:
        """Returns the metric object used for distance calculations.

        :return: The metric used in the index.
        :rtype: Union[MetricKind, CompiledMetric]
        """
        return self._metric_jit if self._metric_jit else self._metric_kind

    @metric.setter
    def metric(self, metric: MetricLike):
        """Sets a new metric for the index.

        :param metric: The new metric to be used.
        :type metric: MetricLike
        :raises ValueError: If the metric is not of type `CompiledMetric` or `MetricKind`.
        """
        metric = _normalize_metric(metric)
        if isinstance(metric, MetricKind):
            metric_kind = metric
            metric_pointer = 0
            metric_signature = MetricSignature.ArrayArraySize
        elif isinstance(metric, CompiledMetric):
            metric_kind = metric.kind
            metric_pointer = metric.pointer
            metric_signature = metric.signature
        else:
            raise ValueError("The `metric` must be a `CompiledMetric` or a `MetricKind`")

        return self._compiled.change_metric(
            metric_kind=metric_kind,
            metric_pointer=metric_pointer,
            metric_signature=metric_signature,
        )

    @property
    def dtype(self) -> ScalarKind:
        """Returns the data type of the vectors in the index.

        :return: The data type of the vectors.
        :rtype: ScalarKind
        """
        return self._compiled.dtype

    @property
    def connectivity(self) -> int:
        """Returns the connectivity parameter of the index.

        This parameter controls how many neighbors each node in the graph is connected to.

        :return: The connectivity of the index.
        :rtype: int
        """
        return self._compiled.connectivity

    @property
    def capacity(self) -> int:
        """Returns the current capacity of the index.

        This indicates the maximum number of vectors that can be indexed without reallocation.

        :return: The capacity of the index.
        :rtype: int
        """
        return self._compiled.capacity

    @property
    def memory_usage(self) -> int:
        """Returns the memory usage of the index in bytes.

        :return: The memory usage of the index.
        :rtype: int
        """
        return self._compiled.memory_usage

    @property
    def expansion_add(self) -> int:
        """Returns the expansion parameter used during addition.

        This parameter controls how many candidates are considered when adding new vectors to the index.

        :return: The expansion parameter for additions.
        :rtype: int
        """
        return self._compiled.expansion_add

    @property
    def expansion_search(self) -> int:
        """Returns the expansion parameter used during searches.

        This parameter controls how many candidates are considered when searching in the index.

        :return: The expansion parameter for searches.
        :rtype: int
        """
        return self._compiled.expansion_search

    @expansion_add.setter
    def expansion_add(self, v: int):
        """Sets the expansion parameter used during addition.

        :param v: The new expansion parameter for additions.
        :type v: int
        """
        self._compiled.expansion_add = v

    @expansion_search.setter
    def expansion_search(self, v: int):
        """Sets the expansion parameter used during searches.

        :param v: The new expansion parameter for searches.
        :type v: int
        """
        self._compiled.expansion_search = v

    def save(
        self,
        path_or_buffer: Union[str, os.PathLike, NoneType] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> Optional[bytes]:
        """Saves the index to a file or buffer.

        If `path_or_buffer` is not provided, it defaults to the path stored in `self.path`.

        :param path_or_buffer: The path or buffer where the index will be saved.
        :type path_or_buffer: Union[str, os.PathLike, NoneType], optional
        :param progress: A callback function for progress tracking.
        :type progress: Optional[ProgressCallback], optional
        :return: The index data as bytes if saving to a buffer, otherwise None.
        :rtype: Optional[bytes]
        """
        assert not progress or _match_signature(progress, [int, int], bool), "Invalid callback signature"

        path_or_buffer = path_or_buffer if path_or_buffer is not None else self.path
        if path_or_buffer is None:
            return self._compiled.save_index_to_buffer(progress)
        else:
            self._compiled.save_index_to_path(os.fspath(path_or_buffer), progress)

    def load(
        self,
        path_or_buffer: Union[PathOrBuffer, NoneType] = None,
        progress: Optional[ProgressCallback] = None,
    ):
        """Loads the index from a file or buffer.

        If `path_or_buffer` is not provided, it defaults to the path stored in `self.path`.

        :param path_or_buffer: The path or buffer from which the index will be loaded.
        :type path_or_buffer: Union[str, os.PathLike, BytesLike, NoneType], optional
        :param progress: A callback function for progress tracking.
        :type progress: Optional[ProgressCallback], optional
        :raises Exception: If no source is defined.
        :raises RuntimeError: If the file does not exist.
        """
        assert not progress or _match_signature(progress, [int, int], bool), "Invalid callback signature"

        path_or_buffer = path_or_buffer if path_or_buffer is not None else self.path
        if path_or_buffer is None:
            raise ValueError("path_or_buffer is required")
        if _is_buffer(path_or_buffer):
            self._compiled.load_index_from_buffer(path_or_buffer, progress)
        else:
            path_or_buffer = os.fspath(path_or_buffer)
            if os.path.exists(path_or_buffer):
                self._compiled.load_index_from_path(path_or_buffer, progress)
            else:
                raise FileNotFoundError(f"File not found: {path_or_buffer}")

    def view(
        self,
        path_or_buffer: Union[PathOrBuffer, NoneType] = None,
        progress: Optional[ProgressCallback] = None,
    ):
        """Maps the index from a file or buffer without loading it into memory.

        If `path_or_buffer` is not provided, it defaults to the path stored in `self.path`.

        :param path_or_buffer: The path or buffer to map the index from.
        :type path_or_buffer: Union[str, os.PathLike, bytes, bytearray, NoneType], optional
        :param progress: A callback function for progress tracking.
        :type progress: Optional[ProgressCallback], optional
        :raises Exception: If no source is defined.
        """
        assert not progress or _match_signature(progress, [int, int], bool), "Invalid callback signature"

        path_or_buffer = path_or_buffer if path_or_buffer is not None else self.path
        if path_or_buffer is None:
            raise ValueError("path_or_buffer is required")
        if _is_buffer(path_or_buffer):
            self._compiled.view_index_from_buffer(path_or_buffer, progress)
        else:
            self._compiled.view_index_from_path(os.fspath(path_or_buffer), progress)

    def clear(self):
        """Erases all vectors from the index, preserving the allocated space for future insertions."""
        self._compiled.clear()

    def reset(self):
        """Erases all data from the index, closes any open files, and returns allocated memory to the OS."""
        if not hasattr(self, "_compiled"):
            return
        self._compiled.reset()

    def __del__(self):
        """Destructor method to reset the index when the object is deleted."""
        self.reset()

    def copy(self) -> Index:
        """Creates a copy of the current index.

        :return: A new instance of the Index class with the same configuration and data.
        :rtype: Index
        """
        result = Index(
            ndim=self.ndim,
            metric=self.metric,
            dtype=self.dtype,
            connectivity=self.connectivity,
            expansion_add=self.expansion_add,
            expansion_search=self.expansion_search,
            path=self.path,
        )
        result._compiled = self._compiled.copy()
        return result

    def join(
        self,
        other: Index,
        max_proposals: int = 0,
        exact: bool = False,
        progress: Optional[ProgressCallback] = None,
    ) -> Dict[Key, Key]:
        """Performs "Semantic Join" or pairwise matching between `self` & `other` index.
        Is different from `search`, as no collisions are allowed in resulting pairs.
        Uses the concept of "Stable Marriages" from Combinatorics, famous for the 2012
        Nobel Prize in Economics.

        :param other: Another index.
        :type other: Index
        :param max_proposals: Limit on candidates evaluated per vector, defaults to 0
        :type max_proposals: int, optional
        :param exact: Controls if underlying `search` should be exact, defaults to False
        :type exact: bool, optional
        :param progress: Callback to report stats of the progress and control it
        :type progress: Optional[ProgressCallback], defaults to None
        :return: Mapping from keys of `self` to keys of `other`
        :rtype: Dict[Key, Key]
        """
        assert not progress or _match_signature(progress, [int, int], bool), "Invalid callback signature"

        return self._compiled.join(
            other=other._compiled,
            max_proposals=max_proposals,
            exact=exact,
            progress=progress,
        )

    def cluster(
        self,
        *,
        vectors: Optional[np.ndarray] = None,
        keys: Optional[np.ndarray] = None,
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
        threads: int = 0,
        log: Union[str, bool] = False,
        progress: Optional[ProgressCallback] = None,
    ) -> Clustering:
        """
        Clusters already indexed or provided `vectors`, mapping them to various centroids.

        :param vectors: .
        :type vectors: Optional[VectorOrVectorsLike]
        :param count: Upper bound on the number of clusters to produce
        :type count: Optional[int], defaults to None

        :param threads: Optimal number of cores to use,
        :type threads: int, defaults to 0
        :param log: Whether to print the progress bar
        :type log: Union[str, bool], defaults to False
        :param progress: Callback to report stats of the progress and control it
        :type progress: Optional[ProgressCallback], defaults to None
        :return: Matches for one or more queries
        :rtype: Union[Matches, BatchMatches]
        """
        assert not progress or _match_signature(progress, [int, int], bool), "Invalid callback signature"

        if min_count is None:
            min_count = 0
        if max_count is None:
            max_count = 0

        if vectors is not None:
            assert keys is None, "You can either cluster vectors or member keys"
            results = self._compiled.cluster_vectors(
                vectors,
                min_count=min_count,
                max_count=max_count,
                threads=threads,
                progress=progress,
            )
        else:
            if keys is None:
                keys = self._compiled.get_keys_in_slice()
            if not isinstance(keys, np.ndarray):
                keys = np.array(keys)
            keys = keys.astype(Key)
            results = self._compiled.cluster_keys(
                keys,
                min_count=min_count,
                max_count=max_count,
                threads=threads,
                progress=progress,
            )

        batch_matches = BatchMatches(*results)
        return Clustering(self, batch_matches, keys)

    def pairwise_distance(self, left: KeyOrKeysLike, right: KeyOrKeysLike) -> Union[np.ndarray, float]:
        """Computes the pairwise distance between keys or key arrays.

        If `left` and `right` are single keys, returns the distance between them.
        If `left` and `right` are arrays of keys, returns a matrix of pairwise distances.

        :param left: A single key or an iterable of keys.
        :type left: KeyOrKeysLike
        :param right: A single key or an iterable of keys.
        :type right: KeyOrKeysLike
        :return: Pairwise distance(s) between the provided keys.
        :rtype: Union[np.ndarray, float]
        """
        assert isinstance(left, Iterable) == isinstance(right, Iterable)

        if not isinstance(left, Iterable):
            return self._compiled.pairwise_distance(int(left), int(right))
        else:
            left = np.array(left).astype(Key)
            right = np.array(right).astype(Key)
            return self._compiled.pairwise_distances(left, right)

    @property
    def keys(self) -> IndexedKeys:
        """Returns all keys currently indexed.

        :return: All indexed keys.
        :rtype: IndexedKeys
        """
        return IndexedKeys(self)

    @property
    def vectors(self) -> np.ndarray:
        """Retrieves all vectors associated with the indexed keys.

        :return: Array of vectors.
        :rtype: np.ndarray
        """
        return self.get(self.keys)

    @property
    def max_level(self) -> int:
        """Returns the maximum level in the multi-level graph.

        :return: The maximum level in the graph.
        :rtype: int
        """
        return self._compiled.max_level

    @property
    def nlevels(self) -> int:
        """Returns the number of levels in the multi-level graph.

        :return: Number of levels in the graph.
        :rtype: int
        """
        return self._compiled.max_level + 1

    @property
    def multi(self) -> bool:
        """Indicates whether the index supports multi-value entries.

        :return: True if the index supports multi-value entries, False otherwise.
        :rtype: bool
        """
        return self._compiled.multi

    @property
    def stats(self) -> _CompiledIndexStats:
        """Get the accumulated statistics for the entire multi-level graph.

        :return: Statistics for the entire multi-level graph.
        :rtype: _CompiledIndexStats

        Statistics:
            - `nodes` (int): Number of nodes in the graph.
            - `edges` (int): Number of edges in the graph.
            - `max_edges` (int): Maximum possible number of edges in the graph.
            - `allocated_bytes` (int): Memory allocated for the graph.
        """
        return self._compiled.stats

    @property
    def levels_stats(self) -> List[_CompiledIndexStats]:
        """Get the accumulated statistics for each level of the graph.

        :return: List of statistics for each level of the graph.
        :rtype: List[_CompiledIndexStats]

        Statistics for each level:
            - `nodes` (int): Number of nodes in the level.
            - `edges` (int): Number of edges in the level.
            - `max_edges` (int): Maximum possible number of edges in the level.
            - `allocated_bytes` (int): Memory allocated for the level.
        """
        return self._compiled.levels_stats

    def level_stats(self, level: int) -> _CompiledIndexStats:
        """Get statistics for a specific level of the graph.

        :param level: The level for which to retrieve statistics.
        :type level: int
        :return: Statistics for the specified level.
        :rtype: _CompiledIndexStats

        Statistics:
            - `nodes` (int): Number of nodes in the level.
            - `edges` (int): Number of edges in the level.
            - `max_edges` (int): Maximum possible number of edges in the level.
            - `allocated_bytes` (int): Memory allocated for the level.
        """
        return self._compiled.level_stats(level)

    @property
    def specs(self) -> Dict[str, Union[str, int, bool]]:
        """Returns the specifications of the index.

        :return: Dictionary of index specifications.
        :rtype: Dict[str, Union[str, int, bool]]
        """
        if not hasattr(self, "_compiled"):
            return "usearch.Index(failed)"
        return {
            "type": "usearch.Index",
            "ndim": self.ndim,
            "multi": self.multi,
            "connectivity": self.connectivity,
            "expansion_add": self.expansion_add,
            "expansion_search": self.expansion_search,
            "size": self.size,
            "jit": self.jit,
            "hardware_acceleration": self.hardware_acceleration,
            "metric_kind": self.metric_kind,
            "dtype": self.dtype,
            "path": self.path,
            "compiled_with_openmp": USES_OPENMP,
            "compiled_with_simsimd": USES_SIMSIMD,
            "compiled_with_native_f16": USES_FP16LIB,
        }

    def __repr__(self) -> str:
        """Returns a string representation of the index object.

        :return: String representation of the index.
        :rtype: str
        """
        if not hasattr(self, "_compiled"):
            return "usearch.Index(failed)"
        f = (
            "usearch.Index({} x {}, {}, multi: {}, connectivity: {}, "
            "expansion: {} & {}, {:,} vectors in {} levels, {} hardware acceleration)"
        )
        return f.format(
            self.dtype,
            self.ndim,
            self.metric_kind,
            self.multi,
            self.connectivity,
            self.expansion_add,
            self.expansion_search,
            len(self),
            self.nlevels,
            self.hardware_acceleration,
        )

    def __repr_pretty__(self) -> str:
        """Returns a pretty-printed string representation of the index object.

        :return: Pretty-printed string representation of the index.
        :rtype: str
        """
        if not hasattr(self, "_compiled"):
            return "usearch.Index(failed)"
        level_stats = [f"--- {i}. {self.level_stats(i).nodes:,} nodes" for i in range(self.nlevels)]
        lines = "\n".join(
            [
                "usearch.Index",
                "- config",
                f"-- data type: {self.dtype}",
                f"-- dimensions: {self.ndim}",
                f"-- metric: {self.metric_kind}",
                f"-- multi: {self.multi}",
                f"-- connectivity: {self.connectivity}",
                f"-- expansion on addition :{self.expansion_add} candidates",
                f"-- expansion on search: {self.expansion_search} candidates",
                "- binary",
                f"-- uses OpenMP: {USES_OPENMP}",
                f"-- uses SimSIMD: {USES_SIMSIMD}",
                f"-- supports half-precision: {USES_FP16LIB}",
                f"-- uses hardware acceleration: {self.hardware_acceleration}",
                "- state",
                f"-- size: {self.size:,} vectors",
                f"-- memory usage: {self.memory_usage:,} bytes",
                f"-- max level: {self.max_level}",
                *level_stats,
            ]
        )
        return lines

    def _repr_pretty_(self, printer, cycle):
        """Handles pretty-printing of the object within interactive environments.

        :param printer: The pretty printer instance.
        :type printer: Any
        :param cycle: Cycle flag indicating recursion.
        :type cycle: bool
        """
        printer.text(self.__repr_pretty__())


class Indexes:
    def __init__(
        self,
        indexes: Iterable[Index] = [],
        paths: Iterable[os.PathLike] = [],
        view: bool = False,
        threads: int = 0,
    ) -> None:
        self._compiled = _CompiledIndexes()
        for index in indexes:
            self._compiled.merge(index._compiled)
        self._compiled.merge_paths(paths, view=view, threads=threads)

    def merge(self, index: Index):
        self._compiled.merge(index._compiled)

    def merge_path(self, path: os.PathLike):
        self._compiled.merge_path(os.fspath(path))

    def __len__(self) -> int:
        return self._compiled.__len__()

    def search(
        self,
        vectors,
        count: int = 10,
        *,
        threads: int = 0,
        exact: bool = False,
        progress: Optional[ProgressCallback] = None,
    ):
        return _search_in_compiled(
            self._compiled.search_many,
            vectors,
            # Batch scheduling:
            log=False,
            # Search constraints:
            count=count,
            exact=exact,
            threads=threads,
            progress=progress,
        )


def search(
    dataset: np.ndarray,
    query: np.ndarray,
    count: int = 10,
    metric: MetricLike = MetricKind.Cos,
    *,
    exact: bool = False,
    threads: int = 0,
    log: Union[str, bool] = False,
    progress: Optional[ProgressCallback] = None,
) -> Union[Matches, BatchMatches]:
    """Shortcut for search, that can avoid index construction. Particularly useful for
    tiny datasets, where brute-force exact search works fast enough.

    :param dataset: Row-major matrix.
    :type dataset: np.ndarray
    :param query: Query vector or vectors (also row-major), to find in `dataset`.
    :type query: np.ndarray

    :param count: Upper count on the number of matches to find, defaults to 10
    :type count: int, optional

    :param metric: Distance function
    :type metric: MetricLike, defaults to MetricKind.Cos
        Kind of the distance function, or the Numba `cfunc` JIT-compiled object.
        Possible `MetricKind` values: IP, Cos, L2sq, Haversine, Pearson,
        Hamming, Tanimoto, Sorensen.

    :param threads: Optimal number of cores to use, defaults to 0
    :type threads: int, optional
    :param exact: Perform exhaustive linear-time exact search, defaults to False
    :type exact: bool, optional
    :param log: Whether to print the progress bar, default to False
    :type log: Union[str, bool], optional
    :param progress: Callback to report stats of the progress and control it
    :type progress: Optional[ProgressCallback], defaults to None
    :return: Matches for one or more queries
    :rtype: Union[Matches, BatchMatches]
    """
    assert not progress or _match_signature(progress, [int, int], bool), "Invalid callback signature"
    assert dataset.ndim == 2, "Dataset must be a matrix, with a vector in each row"

    if not exact:
        index = Index(
            ndim=dataset.shape[1],
            metric=metric,
            dtype=dataset.dtype,
        )
        index.add(
            None,
            dataset,
            threads=threads,
            log=log,
            progress=progress,
        )
        return index.search(
            query,
            count,
            threads=threads,
            log=log,
            progress=progress,
        )

    metric = _normalize_metric(metric)
    if isinstance(metric, MetricKind):
        metric_kind = metric
        metric_pointer = 0
        metric_signature = MetricSignature.ArrayArraySize
    elif isinstance(metric, CompiledMetric):
        metric_kind = metric.kind
        metric_pointer = metric.pointer
        metric_signature = metric.signature
    else:
        raise ValueError("The `metric` must be a `CompiledMetric` or a `MetricKind`")

    def search_batch(query, **kwargs):
        assert dataset.shape[1] == query.shape[1], "Number of dimensions differs"
        if dataset.dtype != query.dtype:
            query = query.astype(dataset.dtype)

        return _exact_search(
            dataset,
            query,
            metric_kind=metric_kind,
            metric_signature=metric_signature,
            metric_pointer=metric_pointer,
            **kwargs,
        )

    return _search_in_compiled(
        search_batch,
        query,
        # Batch scheduling:
        log=log,
        # Search constraints:
        count=count,
        threads=threads,
        progress=progress,
    )


def kmeans(
    X,
    k,
    metric: str = "l2sq",
    dtype: str = "bf16",
    max_iterations: int = 300,
    inertia_threshold: float = 1e-4,
    max_seconds: float = 60.0,
    min_shifts: float = 0.01,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs KMeans clustering on a dataset using the USearch library with mixed-precision support.

    This function clusters the given dataset `X` into `k` clusters by iteratively assigning points
    to the nearest centroids and updating the centroids based on the mean of the points assigned to them.
    The algorithm supports mixed-precision types and early termination based on convergence criteria
    like the number of iterations, inertia threshold, maximum runtime, and minimum point shifts.

    Parameters
    ----------
    X : numpy.ndarray
        The input data, where each row represents a data point and each column represents a feature.
    k : int
        The number of clusters to form.
    metric : str, optional
        The distance metric used to calculate the distance between points and centroids.
        Default is "l2sq" (squared Euclidean distance). Cosine "cos" distance is also supported.
    dtype : str, optional
        The data type used for clustering calculations. Default is "bf16" (Brain Float 16).
        Other supported types include "f32" (float32) and "f64" (float64), "f16" (float16),
        "i8" (int8), and b1 (boolean) bit-packed vectors.
    max_iterations : int, optional
        The maximum number of iterations the algorithm should run. Default is 300.
    inertia_threshold : float, optional
        The threshold for inertia (sum of squared distances to centroids) to terminate early.
        When the change in inertia between iterations falls below this value, the algorithm stops.
        Default is 1e-4.
    max_seconds : float, optional
        The maximum allowable runtime for the algorithm in seconds. If exceeded, the algorithm
        terminates early. Default is 60.0 seconds.
    min_shifts : float, optional
        The minimum fraction of points that must change their assigned cluster between iterations
        to continue. If fewer than this fraction of points change clusters, the algorithm terminates.
        Default is 0.01 (1% of the total points).
    seed : int, optional
        The random seed used to initialize the centroids. Default is None.

    Returns
    -------
    assignments : numpy.ndarray
        An array containing the index of the assigned cluster for each point in the dataset.
    distances : numpy.ndarray
        An array containing the distance of each point to its assigned cluster centroid.
    centroids : numpy.ndarray
        The final centroids of the clusters.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid, such as the number of clusters being greater
        than the number of data points.

    Notes
    -----
    This implementation utilizes mixed-precision computation to speed up the clustering process
    while maintaining accuracy. It also incorporates early exit conditions to avoid unnecessary
    computation when the clustering has stabilized, either by reaching a minimal inertia threshold,
    exceeding the maximum runtime, or when very few points are changing clusters between iterations.

    Example
    -------
    >>> X = np.random.rand(100, 10)
    >>> k = 5
    >>> assignments, distances, centroids = usearch.index.kmeans(X, k)
    """
    metric = _normalize_metric(metric)
    dtype = _normalize_dtype(dtype, ndim=X.shape[1], metric=metric)

    # Generating a 64-bit unsigned integer in NumPy may be somewhat tricky.
    seed = np.random.default_rng().integers(0, 2**64, dtype=np.uint64) if seed is None else seed
    assignments, distances, centroids = _kmeans(
        X,
        k,
        metric_kind=metric,
        max_iterations=max_iterations,
        max_seconds=max_seconds,
        min_shifts=min_shifts,
        inertia_threshold=inertia_threshold,
        dtype=dtype,
        seed=seed,
    )
    return assignments, distances, centroids
