from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import ceil
from time import time_ns
from typing import Any

import numpy as np

from usearch.index import (
    BatchMatches,
    Index,
    Key,
    MetricKind,
    MetricKindBitwise,
    ScalarKind,
    _normalize_dtype,
    _normalize_metric,
    _to_numpy_dtype,
)
from usearch.io import load_matrix


def random_vectors(
    count: int,
    metric: MetricKind = MetricKind.IP,
    dtype: ScalarKind = ScalarKind.F32,
    ndim: int | None = None,
    index: Index | None = None,
) -> np.ndarray:
    """Produces a collection of random vectors normalized for the provided `metric`
    and matching wanted `dtype`, which can both be inferred from an existing `index`.
    """

    # Infer default parameters from the `index`, if passed
    if index is not None:
        if not isinstance(index, Index):
            raise ValueError("Unsupported `index` type")

        ndim = index.ndim
        dtype = index.numpy_dtype
        metric = index.metric

    else:
        metric: MetricKind = _normalize_metric(metric)
        dtype: ScalarKind = _normalize_dtype(dtype, ndim=ndim, metric=metric)

    # Produce data
    if metric in MetricKindBitwise or dtype == ScalarKind.B1:
        bit_vectors = np.random.randint(2, size=(count, ndim))
        bit_vectors = np.packbits(bit_vectors, axis=1)
        return bit_vectors

    else:
        x = np.random.rand(count, ndim)
        if _to_numpy_dtype(dtype) == np.int8:
            x = (x * 100).astype(np.int8)
        else:
            x = x.astype(_to_numpy_dtype(dtype))
            if metric == MetricKind.IP:
                return x / np.linalg.norm(x, axis=1, keepdims=True)
        return x


@dataclass
class SearchStats:
    """
    Contains statistics for one or more search runs, including the number of
    internal nodes that were fetched (`visited_members`) and the number
    of times the distance metric was invoked (`computed_distances`).

    Other derivative metrics include the `mean_recall` and `mean_efficiency`.
    Recall is the share of queried vectors, that were successfully found.
    Efficiency describes the number of distances that had to be computed for
    each query, normalized to size of the `index`. Highest efficiency is 0.(9),
    lowest is zero. Highest is achieved, when the distance metric was computed
    just once per query. Lowest happens during exact search, when every distance
    to every present vector had to be computed.
    """

    index_size: int
    count_queries: int
    count_matches: int

    visited_members: int
    computed_distances: int

    @property
    def mean_efficiency(self) -> float:
        return 1 - float(self.computed_distances) / (self.count_queries * self.index_size)

    @property
    def mean_recall(self) -> float:
        return self.count_matches / self.count_queries


def self_recall(index: Index, sample: float | int = 1.0, **kwargs) -> SearchStats:
    """Simplest benchmark for a quality of search, which queries every
    existing member of the index, to make sure approximate search finds
    the point itself.

    :param index: Non-empty pre-constructed index
    :type index: Index
    :param sample: Share (or number) of vectors to search, defaults to 1.0
    :type sample: Union[float, int]
    :return: Evaluation report with key metrics
    :rtype: SearchStats
    """
    if len(index) == 0:
        return 0
    if "count" not in kwargs:
        kwargs["count"] = 1

    if "keys" in kwargs:
        keys = kwargs.pop("keys")
    else:
        keys = np.array(index.keys)

    if sample != 1.0:
        if isinstance(sample, float):
            sample = int(ceil(len(keys) * sample))
        keys = np.random.choice(keys, sample)

    if "vectors" in kwargs:
        vectors = kwargs.pop("vectors")
    else:
        vectors = index.get(keys)

    matches = index.search(vectors, **kwargs)
    count_matches: int = (
        matches.count_matches(keys) if isinstance(matches, BatchMatches) else int(matches.keys[0] == keys[0])
    )
    return SearchStats(
        index_size=len(index),
        count_queries=len(keys),
        count_matches=count_matches,
        visited_members=matches.visited_members,
        computed_distances=matches.computed_distances,
    )


def measure_seconds(f: Callable) -> tuple[float, Any]:
    """Simple function profiling decorator.

    :param f: Function to be profiled
    :type f: Callable
    :return: Time elapsed in seconds and the result of the execution
    :rtype: Tuple[float, Any]
    """
    a = time_ns()
    result = f()
    b = time_ns()
    c = b - a
    secs = c / (10**9)
    return secs, result


def dcg(relevances: np.ndarray, k: int | None = None) -> np.ndarray:
    """Calculate DCG (Discounted Cumulative Gain) up to position k.

    :param relevances: List of true relevance scores (in the order as they are ranked)
    :type relevances: list
    :param k: Position up to which DCG is computed
    :type k: int
    :return: The DCG score at position k
    :rtype: float
    """
    if k:
        relevances = np.asarray(relevances)[:k]

    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.0

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)


def ndcg(relevances: np.ndarray, k: int | None = None) -> np.ndarray:
    """Calculate NDCG (Normalized Discounted Cumulative Gain) at position k.

    :param relevances: List of true relevance scores (in the order as they are ranked)
    :type relevances: list
    :param k: Position up to which NDCG is computed
    :type k: int
    :return: The NDCG score at position k
    :rtype: float
    """
    best_dcg = dcg(sorted(relevances, reverse=True), k)
    if best_dcg == 0:
        return 0.0

    return dcg(relevances, k) / best_dcg


def relevance(expected: np.ndarray, predicted: np.ndarray, k: int | None = None) -> np.ndarray:
    """Calculate relevance scores. Binary relevance scores

    :param expected: ground-truth keys
    :type expected: np.ndarray
    :param predicted: predicted keys
    :type predicted: np.ndarray
    """
    expected = expected[:k]
    predicted = predicted[:k]
    return [1 if i in expected else 0 for i in predicted]


@dataclass
class Dataset:
    keys: np.ndarray
    vectors: np.ndarray
    queries: np.ndarray
    neighbors: np.ndarray

    def crop_neighbors(self, k: int):
        self.neighbors = self.neighbors[:, k]

    @property
    def ndim(self):
        return self.vectors.shape[1]

    @staticmethod
    def build(
        vectors: str | None = None,
        queries: str | None = None,
        neighbors: str | None = None,
        count: int | None = None,
        ndim: int | None = None,
        k: int | None = None,
    ):
        """Either loads an existing dataset from disk, or generates one on the fly.

        :param vectors: _description_, defaults to None
        :type vectors: Optional[str], optional
        :param queries: _description_, defaults to None
        :type queries: Optional[str], optional
        :param neighbors: _description_, defaults to None
        :type neighbors: Optional[str], optional
        :param count: _description_, defaults to None
        :type count: Optional[int], optional
        :param ndim: _description_, defaults to None
        :type ndim: Optional[int], optional
        :param k: _description_, defaults to None
        :type k: Optional[int], optional
        """

        d = Dataset(None, None, None, None)

        if vectors is not None:
            assert ndim is None

            d.vectors = load_matrix(vectors)
            ndim = d.vectors.shape[1]
            count = min(d.vectors.shape[0], count) if count is not None else d.vectors.shape[0]
            d.vectors = d.vectors[:count, :]
            d.keys = np.arange(count, dtype=Key)

            if queries is not None:
                d.queries = load_matrix(queries)
            else:
                d.queries = d.vectors

            if neighbors is not None:
                d.neighbors = load_matrix(neighbors)
                if k is not None:
                    d.neighbors = d.neighbors[:, :k]
            else:
                assert k is None, "Cant override `k`, will retrieve one neighbor"
                d.neighbors = np.reshape(d.keys, (count, 1))

        else:
            assert ndim is not None
            assert count is not None
            assert k is None, "Cant override `k`, will retrieve one neighbor"

            d.vectors = random_vectors(count=count, ndim=ndim)
            d.queries = d.vectors
            d.keys = np.arange(count, dtype=Key)
            d.neighbors = np.reshape(d.keys, (count, 1))

        return d


@dataclass
class TaskResult:
    add_operations: int | None = None
    add_per_second: float | None = None

    search_operations: int | None = None
    search_per_second: float | None = None
    recall_at_one: float | None = None

    def __repr__(self) -> str:
        parts = []
        if self.add_per_second:
            parts.append(f"{self.add_per_second:.2f} add/s")
        if self.search_per_second:
            parts.append(f"{self.search_per_second:.2f} search/s")
        if self.recall_at_one:
            parts.append(f"{self.recall_at_one * 100:.2f}% recall@1")
        return ", ".join(parts)

    @property
    def add_seconds(self) -> float:
        return self.add_operations / self.add_per_second

    @property
    def search_seconds(self) -> float:
        return self.search_operations / self.search_per_second

    def __add__(self, other: TaskResult):
        result = TaskResult()
        if self.add_operations and other.add_operations:
            result.add_operations = self.add_operations + other.add_operations
            result.add_per_second = result.add_operations / (self.add_seconds + other.add_seconds)
        else:
            base = self if self.add_operations else other
            result.add_operations = base.add_operations
            result.add_per_second = base.add_per_second

        if self.search_operations and other.search_operations:
            result.search_operations = self.search_operations + other.search_operations
            result.recall_at_one = (
                self.recall_at_one * self.search_operations + other.recall_at_one * other.search_operations
            ) / (self.search_operations + other.search_operations)
            result.search_per_second = result.search_operations / (self.search_seconds + other.search_seconds)
        else:
            base = self if self.search_operations else other
            result.search_operations = base.search_operations
            result.search_per_second = base.search_per_second
            result.recall_at_one = base.recall_at_one

        return result


@dataclass
class AddTask:
    keys: np.ndarray
    vectors: np.ndarray

    def __call__(self, index: Index) -> TaskResult:
        batch_size: int = self.vectors.shape[0]
        old_size: int = len(index)
        dt, _ = measure_seconds(lambda: index.add(self.keys, self.vectors))

        assert len(index) == old_size + batch_size
        return TaskResult(
            add_operations=batch_size,
            add_per_second=batch_size / dt,
        )

    @property
    def ndim(self):
        return self.vectors.shape[1]

    @property
    def count(self):
        return self.vectors.shape[0]

    def inplace_shuffle(self):
        """Reorders the `vectors` and `keys`. Often used for robustness benchmarks."""

        new_order = np.arange(self.count)
        np.random.shuffle(new_order)
        self.keys = self.keys[new_order]
        self.vectors = self.vectors[new_order, :]

    def slices(self, batch_size: int) -> list[AddTask]:
        """Splits this dataset into smaller chunks."""

        return [
            AddTask(
                keys=self.keys[start_row : start_row + batch_size],
                vectors=self.vectors[start_row : start_row + batch_size, :],
            )
            for start_row in range(0, self.count, batch_size)
        ]

    def clusters(self, number_of_clusters: int) -> list[AddTask]:
        """Splits this dataset into smaller chunks."""

        from sklearn.cluster import KMeans

        clustering = KMeans(
            n_clusters=number_of_clusters,
            random_state=0,
            n_init="auto",
        ).fit(self.vectors)

        partitioning = defaultdict(list)
        for row, cluster in enumerate(clustering.labels_):
            partitioning[cluster].append(row)

        return [
            AddTask(
                keys=self.keys[rows],
                vectors=self.vectors[rows, :],
            )
            for rows in partitioning.values()
        ]


@dataclass
class SearchTask:
    queries: np.ndarray
    neighbors: np.ndarray

    def __call__(self, index: Index) -> TaskResult:
        dt, results = measure_seconds(lambda: index.search(self.queries, self.neighbors.shape[1]))

        return TaskResult(
            search_per_second=self.queries.shape[0] / dt,
            recall_at_one=results.mean_recall(self.neighbors[:, 0].flatten()),
        )

    def slices(self, batch_size: int) -> list[SearchTask]:
        """Splits this dataset into smaller chunks."""

        return [
            SearchTask(
                queries=self.queries[start_row : start_row + batch_size, :],
                neighbors=self.neighbors[start_row : start_row + batch_size, :],
            )
            for start_row in range(0, self.queries.shape[0], batch_size)
        ]


@dataclass
class Evaluation:
    tasks: list[AddTask | SearchTask]
    count: int
    ndim: int

    @staticmethod
    def for_dataset(dataset: Dataset, batch_size: int = 0, clusters: int = 1) -> Evaluation:
        tasks = []
        add = AddTask(vectors=dataset.vectors, keys=dataset.keys)
        search = SearchTask(queries=dataset.queries, neighbors=dataset.neighbors)

        if batch_size:
            tasks.extend(add.slices(batch_size))
            tasks.extend(search.slices(batch_size))
        elif clusters != 1:
            tasks.extend(add.clusters(clusters))
            print(tasks)
            tasks.append(search)
        else:
            tasks.append(add)
            tasks.append(search)

        return Evaluation(
            tasks=tasks,
            count=add.count,
            ndim=add.ndim,
        )

    def __call__(self, index: Index, post_clean: bool = True) -> dict:
        task_result = TaskResult()

        try:
            for task in self.tasks:
                task_result = task_result + task(index)
        except KeyboardInterrupt:
            pass

        if post_clean:
            index.clear()
        return {
            **index.specs,
            **asdict(task_result),
        }


if __name__ == "__main__":
    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Evaluate vector search index for speed and accuracy.")

    # Define expected arguments
    parser.add_argument("--vectors", type=str, required=False, help="Path to the file containing the vectors.")
    parser.add_argument("--queries", type=str, required=False, help="Path to the file containing the query vectors.")
    parser.add_argument("--neighbors", type=str, required=False, help="Path to the file with neighbor arrays.")
    parser.add_argument("--dtype", type=str, required=False, help="Quantization type for internal storage.")
    parser.add_argument("--metric", type=str, required=False, help="Distance function.")
    parser.add_argument("--count", type=int, help="Number of vectors to use.")
    parser.add_argument("--ndim", type=int, help="Number of dimensions for the vectors.")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size for indexing and searching.")
    parser.add_argument("--clusters", type=int, default=1, help="Number of clusters for indexing.")

    # Parse arguments from the command line
    args = parser.parse_args()

    # Load or generate dataset
    dataset = Dataset.build(
        vectors=args.vectors,
        queries=args.queries,
        neighbors=args.neighbors,
        count=args.count,
        ndim=args.ndim,
    )

    # Prepare the evaluation
    evaluation = Evaluation.for_dataset(dataset, batch_size=args.batch_size, clusters=args.clusters)
    index = Index(ndim=dataset.ndim, dtype=args.dtype, metric=args.metric)

    # Perform the evaluation
    results = evaluation(index)

    # Print the evaluation results
    print("Evaluation results:", results)
