#!/usr/bin/env -S uv run --quiet --script
"""
USearch HNSW Index Benchmarking

Benchmarks HNSW approximate nearest neighbor search configurations.
Supports comma-separated --dtype, --metric, and --backend for cross-product sweeps.

Development setup (once):
    uv venv && source .venv/bin/activate
    uv pip install -e . --force-reinstall
    uv pip install faiss-cpu  # optional, for --backend faiss

Development usage (uses local build from .venv):
    python python/scripts/bench_index.py --dtype f32,f16,i8 -n 100000 -k 10
    python python/scripts/bench_index.py --dtype f32,b1 --metric ip,hamming --backend usearch,faiss

Standalone usage (installs usearch from PyPI automatically):
    uv run python/scripts/bench_index.py --dtype f32 -n 100000 -k 10

Examples:
    bench_index.py --dtype f32,f16,i8 -n 100000 --neighbors-count 10
    bench_index.py --dtype f32,fp16 --backend usearch,faiss -n 100000 --neighbors-count 100
    bench_index.py --dtype b1 --metric hamming -n 100000 --dimensions 512
    bench_index.py --dtype i8 --vectors data/base.fbin --queries data/query.fbin
"""
# /// script
# dependencies = [
#   "numpy",
#   "usearch",
#   "tqdm"
# ]
# ///

import argparse
import sys
from dataclasses import dataclass
from time import perf_counter

import numpy as np

import usearch
from usearch.eval import random_vectors
from usearch.index import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
    Index,
    Key,
    _normalize_dtype,
    _normalize_metric,
)
from usearch.io import load_matrix

SPATIAL_METRICS = {"ip", "cos", "l2sq"}
BITWISE_METRICS = {"hamming", "tanimoto", "jaccard", "sorensen"}
ALL_METRICS = SPATIAL_METRICS | BITWISE_METRICS

USEARCH_DENSE_DTYPES = {"f64", "f32", "f16", "bf16", "e5m2", "e4m3", "e3m2", "e2m3", "i8", "u8"}
USEARCH_BINARY_DTYPES = {"b1", "bits"}

# FAISS dtypes imported lazily from index_faiss.py when needed
_FAISS_DENSE_DTYPES = {"f32", "f16", "fp16", "bf16", "i8", "int8", "u8", "uint8"}
_FAISS_BINARY_DTYPES = {"b1", "bits"}


@dataclass
class BenchConfig:
    """Single benchmark configuration."""

    dtype: str
    metric: str
    backend: str
    connectivity: int
    expansion_add: int
    expansion_search: int
    neighbors_count: int


@dataclass
class BenchResult:
    """Single benchmark result row."""

    library: str
    dtype: str
    metric: str
    acceleration: str
    add_per_second: float
    search_per_second: float
    recall: float
    neighbors_count: int


def _is_supported(backend: str, dtype: str, metric: str) -> bool:
    """Check if a (backend, dtype, metric) combination is valid."""
    is_bitwise = metric in BITWISE_METRICS
    if backend == "usearch":
        valid_dtypes = USEARCH_BINARY_DTYPES if is_bitwise else USEARCH_DENSE_DTYPES
    elif backend == "faiss":
        valid_dtypes = _FAISS_BINARY_DTYPES if is_bitwise else _FAISS_DENSE_DTYPES
        if is_bitwise and metric != "hamming":
            return False
    else:
        return False
    return dtype in valid_dtypes


def _data_category(dtype: str, metric: str) -> str:
    """Return a key grouping dtypes that need the same synthetic data distribution."""
    if dtype in ("b1", "bits") or metric in BITWISE_METRICS:
        return "bits"
    if dtype in ("u8", "uint8"):
        return "uint8"
    if dtype in ("i8", "int8"):
        return "int8"
    return "float32"


def _recall_at_k(found_keys: np.ndarray, ground_truth: np.ndarray) -> float:
    """Fraction of queries where any true neighbor appears in the top-k results."""
    hits = 0
    for i in range(found_keys.shape[0]):
        if any(key in ground_truth[i] for key in found_keys[i]):
            hits += 1
    return hits / found_keys.shape[0]


def _parse_csv(value: str) -> list[str]:
    """Split a comma-separated string into a list of stripped tokens."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _create_index(config: BenchConfig, dimensions: int):
    """Create a USearch or FAISS index based on the config. Returns an index with add/search/hardware_acceleration."""
    is_binary = config.metric in BITWISE_METRICS

    if config.backend == "usearch":
        metric_kind = _normalize_metric(config.metric)
        ndim = dimensions * 8 if is_binary else dimensions
        scalar_kind = _normalize_dtype(config.dtype, ndim=ndim, metric=metric_kind)
        return Index(
            ndim=ndim,
            metric=metric_kind,
            dtype=scalar_kind,
            connectivity=config.connectivity,
            expansion_add=config.expansion_add,
            expansion_search=config.expansion_search,
        )

    if config.backend == "faiss":
        from index_faiss import IndexFAISS

        return IndexFAISS(
            dimensions=dimensions,
            metric=config.metric,
            dtype=config.dtype,
            connectivity=config.connectivity,
            expansion_add=config.expansion_add,
            expansion_search=config.expansion_search,
        )

    raise ValueError(f"Unknown backend: {config.backend}")


def _run_benchmark(
    dtype_name: str,
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray | None,
    config: BenchConfig,
) -> BenchResult:
    """Run a single benchmark configuration. Returns a BenchResult."""
    dimensions = vectors.shape[1]
    index = _create_index(config, dimensions)
    keys = np.arange(vectors.shape[0], dtype=Key)

    start = perf_counter()
    index.add(keys, vectors, log=True, dtype=dtype_name)
    add_elapsed = perf_counter() - start

    start = perf_counter()
    matches = index.search(queries, config.neighbors_count, log=True, dtype=dtype_name)
    search_elapsed = perf_counter() - start

    found_keys = matches.keys.reshape(-1, config.neighbors_count)
    recall = _recall_at_k(found_keys, ground_truth) if ground_truth is not None else float("nan")

    return BenchResult(
        library="FAISS" if config.backend == "faiss" else "USearch",
        dtype=config.dtype,
        metric=config.metric,
        acceleration=index.hardware_acceleration,
        add_per_second=vectors.shape[0] / add_elapsed if add_elapsed > 0 else float("inf"),
        search_per_second=queries.shape[0] / search_elapsed if search_elapsed > 0 else float("inf"),
        recall=recall,
        neighbors_count=config.neighbors_count,
    )


def _print_results(results: list[BenchResult]) -> None:
    if not results:
        return
    k = results[0].neighbors_count
    recall_col = f"Recall@{k}"
    header = (
        f"{'Library':<10} {'Quantization':<14} {'Metric':<10} {'Acceleration':<14} "
        f"{'Add/s':>12} {'Search/s':>12} {recall_col:>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for result in results:
        recall_str = f"{result.recall * 100:.1f}%" if not np.isnan(result.recall) else "N/A"
        print(
            f"{result.library:<10} {result.dtype:<14} {result.metric:<10} {result.acceleration:<14} "
            f"{result.add_per_second:>12,.0f} {result.search_per_second:>12,.0f} {recall_str:>10}"
        )


def main():
    all_dtypes = sorted(USEARCH_DENSE_DTYPES | USEARCH_BINARY_DTYPES | _FAISS_DENSE_DTYPES | _FAISS_BINARY_DTYPES)
    parser = argparse.ArgumentParser(
        description="Benchmark HNSW approximate search configurations",
        epilog=(
            "Comma-separated lists supported for --dtype, --metric, --backend.\n"
            "Examples:\n"
            "  bench_index.py --dtype f32,f16,i8 --metric ip -n 100000 --neighbors-count 10\n"
            "  bench_index.py --dtype b1 --metric hamming -n 100000 --dimensions 512\n"
            "  bench_index.py --dtype f32,b1 --metric ip,hamming --backend usearch,faiss"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source
    parser.add_argument("--vectors", type=str, help="Path to base vectors file (.fbin, .hbin, .i8bin, .u8bin)")
    parser.add_argument("--queries", type=str, help="Path to query vectors file")
    parser.add_argument("--neighbors", type=str, help="Path to ground truth neighbors file (.ibin)")

    # Synthetic data parameters
    parser.add_argument("-n", "--count", default=100_000, type=int, help="Number of vectors (default: 100000)")
    parser.add_argument("-d", "--dimensions", default=256, type=int, help="Number of dimensions (default: 256)")
    parser.add_argument("-nq", "--queries-count", default=1000, type=int, help="Number of queries (default: 1000)")

    # Configuration — all accept comma-separated lists
    parser.add_argument(
        "--dtype",
        default="f32",
        help=f"Quantization type(s), comma-separated (default: f32). Choices: {', '.join(all_dtypes)}",
    )
    parser.add_argument(
        "--metric",
        default="ip",
        help=f"Distance metric(s), comma-separated (default: ip). Choices: {', '.join(sorted(ALL_METRICS))}",
    )
    parser.add_argument(
        "--backend",
        default="usearch",
        help="Backend(s), comma-separated (default: usearch). Choices: usearch, faiss",
    )
    parser.add_argument("-k", "--neighbors-count", default=10, type=int, help="Number of neighbors (default: 10)")

    # HNSW parameters
    parser.add_argument("-c", "--connectivity", type=int, default=DEFAULT_CONNECTIVITY, help="HNSW connectivity (M)")
    parser.add_argument("--expansion-add", type=int, default=DEFAULT_EXPANSION_ADD)
    parser.add_argument("--expansion-search", type=int, default=DEFAULT_EXPANSION_SEARCH)

    args = parser.parse_args()

    # Parse comma-separated lists
    dtypes = _parse_csv(args.dtype)
    metrics = _parse_csv(args.metric)
    backends = _parse_csv(args.backend)

    # Validate
    for dtype in dtypes:
        if dtype not in all_dtypes:
            parser.error(f"Unknown dtype '{dtype}'. Choices: {', '.join(all_dtypes)}")
    for metric in metrics:
        if metric not in ALL_METRICS:
            parser.error(f"Unknown metric '{metric}'. Choices: {', '.join(sorted(ALL_METRICS))}")
    for backend in backends:
        if backend not in ("usearch", "faiss"):
            parser.error(f"Unknown backend '{backend}'. Choices: usearch, faiss")

    # Auto-add hamming metric when b1 dtype requested without any bitwise metric
    if any(dtype in ("b1", "bits") for dtype in dtypes) and not any(metric in BITWISE_METRICS for metric in metrics):
        metrics.append("hamming")

    # Library versions and hardware info
    print(f"USearch v{usearch.VERSION_MAJOR}.{usearch.VERSION_MINOR}.{usearch.VERSION_PATCH}")
    print(f"  Compiled ISA: {usearch.hardware_acceleration_compiled()}")
    print(f"  Available ISA: {usearch.hardware_acceleration_available()}")
    if "faiss" in backends:
        try:
            from index_faiss import hardware_acceleration_available as faiss_isa
            from index_faiss import version as faiss_version

            print(f"FAISS v{faiss_version()} (ISA: {faiss_isa()})")
        except ImportError:
            print("FAISS not available. Install: uv pip install faiss-cpu", file=sys.stderr)
            backends = [backend for backend in backends if backend != "faiss"]

    # Build valid (backend, dtype, metric) configurations
    configurations = [
        BenchConfig(
            dtype=dtype,
            metric=metric,
            backend=backend,
            connectivity=args.connectivity,
            expansion_add=args.expansion_add,
            expansion_search=args.expansion_search,
            neighbors_count=args.neighbors_count,
        )
        for backend in backends
        for dtype in dtypes
        for metric in metrics
        if _is_supported(backend, dtype, metric)
    ]

    # Load real data or generate synthetic datasets per dtype category
    if args.vectors:
        print(f"Loading vectors from {args.vectors}")
        base_vectors = load_matrix(args.vectors)
        query_vectors = (
            load_matrix(args.queries)
            if args.queries
            else base_vectors[: min(args.queries_count, base_vectors.shape[0])]
        )
        ground_truth = load_matrix(args.neighbors) if args.neighbors else None
        datasets = None
        print(f"Loaded: {base_vectors.shape[0]:,} x {base_vectors.shape[1]}, k={args.neighbors_count}")
    else:
        queries_count = min(args.queries_count, args.count)
        categories_needed = {_data_category(config.dtype, config.metric) for config in configurations}
        datasets = {}
        print(
            f"Generating synthetic datasets: {args.count:,} vectors, {args.dimensions} dims, k={args.neighbors_count}"
        )
        for category in sorted(categories_needed):
            scalar_kind = _normalize_dtype({"float32": "f32", "uint8": "u8", "int8": "i8", "bits": "b1"}[category])
            metric_kind = _normalize_metric("hamming" if category == "bits" else "ip")
            vectors = random_vectors(
                count=args.count, ndim=args.dimensions, metric=metric_kind, quantization=scalar_kind
            )
            ground_truth_self = np.arange(queries_count, dtype=np.int64).reshape(-1, 1)
            datasets[category] = (vectors, vectors[:queries_count], ground_truth_self)
            print(f"  {category}: {vectors.shape}, {vectors.dtype}")
        base_vectors = query_vectors = ground_truth = None

    # Run benchmarks (Ctrl+C prints partial table)
    results: list[BenchResult] = []
    for config in configurations:
        if datasets is not None:
            category = _data_category(config.dtype, config.metric)
            vectors, queries, ground_truth = datasets[category]
        else:
            category = "float32"
            vectors, queries = base_vectors, query_vectors

        print(f"\n--- {config.backend} / {config.dtype} / {config.metric} ---")
        try:
            results.append(_run_benchmark(category, vectors, queries, ground_truth, config))
        except KeyboardInterrupt:
            print("\n\nInterrupted — printing results collected so far.")
            break
        except Exception as exception:
            print(f"  Failed: {exception}")

    _print_results(results)


if __name__ == "__main__":
    main()
