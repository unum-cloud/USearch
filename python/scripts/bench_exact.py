#!/usr/bin/env -S uv run --quiet --script
"""
USearch Exact Search Benchmarking

Benchmarks exact (brute-force) nearest neighbor search.
Supports comma-separated --dtype, --metric, and --backend for cross-product sweeps.

Development setup (once):
    uv venv && source .venv/bin/activate
    uv pip install -e . --force-reinstall
    uv pip install faiss-cpu numkong  # optional

Usage:
    python python/scripts/bench_exact.py --dtype f32,f16 --backend usearch,faiss -k 10
    python python/scripts/bench_exact.py --dtype f32 --backend usearch,numkong -k 100
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
from usearch import hardware_acceleration
from usearch.index import (
    MetricKind,
    _normalize_dtype,
    _normalize_metric,
    search,
)

SUPPORTED_DTYPES = {"b1", "bits", "i8", "u8", "f16", "bf16", "f32", "f64", "e4m3", "e3m2", "e2m3"}
SUPPORTED_METRICS = {"ip", "cos", "l2sq"}
SUPPORTED_BACKENDS = {"usearch", "faiss", "numkong"}


@dataclass
class ExactBenchResult:
    library: str
    dtype: str
    metric: str
    duration_seconds: float
    queries_per_second: float


def _parse_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _data_category(dtype: str) -> str:
    if dtype in ("b1", "bits"):
        return "bits"
    if dtype in ("u8", "uint8"):
        return "uint8"
    if dtype in ("i8", "int8"):
        return "int8"
    return "float32"


def _generate_vectors(count: int, dimensions: int, category: str) -> np.ndarray:
    if category == "bits":
        bits = np.random.randint(2, size=(count, dimensions))
        return np.packbits(bits, axis=1).astype(np.uint8)
    if category == "uint8":
        return np.random.randint(0, 256, size=(count, dimensions)).astype(np.uint8)
    if category == "int8":
        return np.random.randint(-128, 128, size=(count, dimensions)).astype(np.int8)
    vectors = np.random.randn(count, dimensions).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def _run_usearch(
    haystack: np.ndarray,
    queries: np.ndarray,
    neighbors_count: int,
    metric_kind: MetricKind,
    queries_count: int,
    dtype: str,
    metric: str,
) -> ExactBenchResult:
    start_time = perf_counter()
    _result = search(haystack, queries, neighbors_count, metric=metric_kind, exact=True).keys
    elapsed = perf_counter() - start_time
    return ExactBenchResult(
        library="USearch",
        dtype=dtype,
        metric=metric,
        duration_seconds=elapsed,
        queries_per_second=queries_count / elapsed if elapsed > 0 else float("inf"),
    )


def _run_faiss(
    haystack: np.ndarray,
    queries: np.ndarray,
    neighbors_count: int,
    metric_kind: MetricKind,
    queries_count: int,
    dtype: str,
    metric: str,
) -> ExactBenchResult:
    try:
        from faiss import METRIC_INNER_PRODUCT, METRIC_L2, knn
    except ImportError:
        print("FAISS not available. Install: uv pip install faiss-cpu", file=sys.stderr)
        sys.exit(1)

    if metric_kind not in (MetricKind.L2sq, MetricKind.IP):
        raise ValueError(f"FAISS only supports l2sq and ip metrics, got {metric_kind}")

    faiss_metric = METRIC_L2 if metric_kind == MetricKind.L2sq else METRIC_INNER_PRODUCT
    haystack_float32 = haystack.astype(np.float32) if haystack.dtype != np.float32 else haystack
    queries_float32 = queries.astype(np.float32) if queries.dtype != np.float32 else queries

    start_time = perf_counter()
    _distances, _ids = knn(queries_float32, haystack_float32, neighbors_count, metric=faiss_metric)
    elapsed = perf_counter() - start_time
    return ExactBenchResult(
        library="FAISS",
        dtype=dtype,
        metric=metric,
        duration_seconds=elapsed,
        queries_per_second=queries_count / elapsed if elapsed > 0 else float("inf"),
    )


def _run_numkong(
    haystack: np.ndarray,
    queries: np.ndarray,
    neighbors_count: int,
    metric_kind: MetricKind,
    queries_count: int,
    dtype: str,
    metric: str,
) -> ExactBenchResult:
    try:
        import numkong as nk
    except ImportError:
        print("NumKong not available. Install: uv pip install numkong", file=sys.stderr)
        sys.exit(1)

    nk_metric_map = {
        MetricKind.L2sq: "sqeuclidean",
        MetricKind.IP: "inner",
        MetricKind.Cos: "angular",
    }
    nk_metric = nk_metric_map.get(metric_kind)
    if nk_metric is None:
        raise ValueError(f"NumKong does not support metric {metric_kind}")

    haystack_float32 = haystack.astype(np.float32) if haystack.dtype != np.float32 else haystack
    queries_float32 = queries.astype(np.float32) if queries.dtype != np.float32 else queries

    start_time = perf_counter()
    distance_matrix = nk.cdist(queries_float32, haystack_float32, metric=nk_metric)
    _found_neighbors = np.argpartition(np.asarray(distance_matrix), neighbors_count, axis=1)[:, :neighbors_count]
    elapsed = perf_counter() - start_time
    return ExactBenchResult(
        library="NumKong",
        dtype=dtype,
        metric=metric,
        duration_seconds=elapsed,
        queries_per_second=queries_count / elapsed if elapsed > 0 else float("inf"),
    )


_BACKEND_RUNNERS = {
    "usearch": _run_usearch,
    "faiss": _run_faiss,
    "numkong": _run_numkong,
}


def _print_results(results: list[ExactBenchResult]) -> None:
    if not results:
        return
    header = f"{'Library':<10} {'Dtype':<10} {'Metric':<8} {'Duration':>14} {'Throughput':>18}"
    print(f"\n{header}")
    print("-" * len(header))
    for result in results:
        print(
            f"{result.library:<10} {result.dtype:<10} {result.metric:<8} "
            f"{result.duration_seconds * 1000:>11,.2f} ms {result.queries_per_second:>14,.0f} q/s"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark exact nearest neighbor search",
        epilog=(
            "Comma-separated lists supported for --dtype, --metric, --backend.\n"
            "Examples:\n"
            "  bench_exact.py --dtype f32,f16 --backend usearch,faiss -k 10\n"
            "  bench_exact.py --dtype f32 --backend usearch,numkong -k 100"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-d", "--dimensions", default=256, type=int, help="Number of dimensions (default: 256)")
    parser.add_argument("-n", "--count", default=10**5, type=int, help="Number of vectors (default: 100000)")
    parser.add_argument("-nq", "--queries-count", default=10, type=int, help="Number of queries (default: 10)")
    parser.add_argument("-k", "--neighbors-count", default=100, type=int, help="Number of neighbors (default: 100)")
    parser.add_argument(
        "--dtype",
        default="f32",
        help=f"Data type(s), comma-separated (default: f32). Choices: {', '.join(sorted(SUPPORTED_DTYPES))}",
    )
    parser.add_argument(
        "--metric",
        default="ip",
        help=f"Metric(s), comma-separated (default: ip). Choices: {', '.join(sorted(SUPPORTED_METRICS))}",
    )
    parser.add_argument(
        "--backend",
        default="usearch",
        help=f"Backend(s), comma-separated (default: usearch). Choices: {', '.join(sorted(SUPPORTED_BACKENDS))}",
    )

    args = parser.parse_args()

    dtypes = _parse_csv(args.dtype)
    metrics = _parse_csv(args.metric)
    backends = _parse_csv(args.backend)

    for dtype in dtypes:
        if dtype not in SUPPORTED_DTYPES:
            parser.error(f"Unknown dtype '{dtype}'. Choices: {', '.join(sorted(SUPPORTED_DTYPES))}")
    for metric in metrics:
        if metric not in SUPPORTED_METRICS:
            parser.error(f"Unknown metric '{metric}'. Choices: {', '.join(sorted(SUPPORTED_METRICS))}")
    for backend in backends:
        if backend not in SUPPORTED_BACKENDS:
            parser.error(f"Unknown backend '{backend}'. Choices: {', '.join(sorted(SUPPORTED_BACKENDS))}")

    # Library versions
    print(f"USearch v{usearch.VERSION_MAJOR}.{usearch.VERSION_MINOR}.{usearch.VERSION_PATCH}")
    print(f"  Compiled ISA: {usearch.hardware_acceleration_compiled()}")
    print(f"  Available ISA: {usearch.hardware_acceleration_available()}")

    # Generate datasets per category
    queries_count = min(args.queries_count, args.count)
    categories_needed = {_data_category(dtype) for dtype in dtypes}
    datasets = {}
    print(f"Generating: {args.count:,} vectors, {args.dimensions} dims, k={args.neighbors_count}")
    for category in sorted(categories_needed):
        vectors = _generate_vectors(args.count, args.dimensions, category)
        datasets[category] = (vectors, vectors[:queries_count])
        print(f"  {category}: {vectors.shape}, {vectors.dtype}")

    # Build configurations and run (Ctrl+C prints partial table)
    configurations = [(backend, dtype, metric) for backend in backends for dtype in dtypes for metric in metrics]

    results: list[ExactBenchResult] = []
    for backend, dtype, metric in configurations:
        category = _data_category(dtype)
        haystack, queries = datasets[category]
        metric_kind = _normalize_metric(metric)

        print(f"\n--- {backend} / {dtype} / {metric} ---")
        scalar_kind = _normalize_dtype(dtype, ndim=args.dimensions, metric=metric_kind)
        acceleration = hardware_acceleration(dtype=scalar_kind, ndim=args.dimensions, metric_kind=metric_kind)
        print(f"  Acceleration: {acceleration}")

        runner = _BACKEND_RUNNERS[backend]
        try:
            results.append(runner(haystack, queries, args.neighbors_count, metric_kind, queries_count, dtype, metric))
        except KeyboardInterrupt:
            print("\n\nInterrupted — printing results collected so far.")
            break
        except Exception as exception:
            print(f"  Failed: {exception}")

    _print_results(results)


if __name__ == "__main__":
    main()
