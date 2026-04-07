#!/usr/bin/env -S uv run --quiet --script
"""
USearch Exact Search Benchmarking

Benchmarks exact (brute-force) nearest neighbor search for a single backend.
Run multiple times with different --backend flags to compare.

Usage:
    uv run python/scripts/bench_exact.py --help
    uv run python/scripts/bench_exact.py -k 10 -n 10000 --backend usearch
    uv run python/scripts/bench_exact.py -k 10 -n 10000 --backend faiss
    uv run python/scripts/bench_exact.py -k 10 -n 10000 --backend numkong

Dependencies listed in the script header for uv to resolve automatically.
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
from time import perf_counter

import numpy as np

import usearch
from usearch.compiled import hardware_acceleration
from usearch.eval import random_vectors
from usearch.index import (
    MetricKind,
    ScalarKind,
    _normalize_dtype,
    _normalize_metric,
    search,
)


def _format_duration(duration: float) -> str:
    return f"{duration * 1000:,.2f} ms"


def _format_throughput(duration: float, count: int) -> str:
    if duration > 0:
        return f"{count / duration:,.2f} calls/sec"
    return "inf calls/sec"


def run(
    n: int = 10**5,
    q: int = 10,
    k: int = 100,
    ndim: int = 256,
    dtype: str = "f32",
    metric: str = "ip",
    backend: str = "usearch",
) -> None:
    metric_kind: MetricKind = _normalize_metric(metric)
    scalar_kind: ScalarKind = _normalize_dtype(dtype, ndim=ndim, metric=metric_kind)
    accel = hardware_acceleration(dtype=scalar_kind, ndim=ndim, metric_kind=metric_kind)
    print(f"Hardware acceleration: {accel}")
    print(f"USearch v{usearch.VERSION_MAJOR}.{usearch.VERSION_MINOR}.{usearch.VERSION_PATCH}")

    x = random_vectors(n, ndim=ndim, dtype=scalar_kind)
    queries = x[:q]

    if backend == "usearch":
        t0 = perf_counter()
        _result = search(x, queries, k, metric=metric_kind, exact=True).keys
        elapsed = perf_counter() - t0
        print(f"USearch:  {_format_duration(elapsed)} ({_format_throughput(elapsed, q)})")

    elif backend == "faiss":
        try:
            from faiss import METRIC_INNER_PRODUCT, METRIC_L2, knn
        except ImportError:
            print("FAISS not installed (pip install faiss-cpu)", file=sys.stderr)
            sys.exit(1)

        if metric_kind not in (MetricKind.L2sq, MetricKind.IP):
            print(f"FAISS only supports l2sq and ip metrics, got {metric}", file=sys.stderr)
            sys.exit(1)

        faiss_metric = METRIC_L2 if metric_kind == MetricKind.L2sq else METRIC_INNER_PRODUCT
        x_f32 = x.astype(np.float32) if x.dtype != np.float32 else x
        q_f32 = queries.astype(np.float32) if queries.dtype != np.float32 else queries

        t0 = perf_counter()
        _distances, _ids = knn(q_f32, x_f32, k, metric=faiss_metric)
        elapsed = perf_counter() - t0
        print(f"FAISS:   {_format_duration(elapsed)} ({_format_throughput(elapsed, q)})")

    elif backend == "numkong":
        try:
            import numkong as nk
        except ImportError:
            print("NumKong not installed (pip install numkong)", file=sys.stderr)
            sys.exit(1)

        nk_metric_map = {
            MetricKind.L2sq: "sqeuclidean",
            MetricKind.IP: "inner",
            MetricKind.Cos: "angular",
        }
        nk_metric = nk_metric_map.get(metric_kind)
        if nk_metric is None:
            print(f"NumKong does not support metric '{metric}'", file=sys.stderr)
            sys.exit(1)

        x_f32 = x.astype(np.float32) if x.dtype != np.float32 else x
        q_f32 = queries.astype(np.float32) if queries.dtype != np.float32 else queries

        t0 = perf_counter()
        D = nk.cdist(q_f32, x_f32, metric=nk_metric)
        _neighbors = np.argpartition(np.asarray(D), k, axis=1)[:, :k]
        elapsed = perf_counter() - t0
        print(f"NumKong: {_format_duration(elapsed)} ({_format_throughput(elapsed, q)})")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark exact nearest neighbor search for a single backend",
        epilog="Run multiple times with different --backend to compare.",
    )
    parser.add_argument("--ndim", default=256, type=int, help="Number of dimensions (default: 256)")
    parser.add_argument("-n", default=10**5, type=int, help="Number of vectors (default: 100000)")
    parser.add_argument("-q", default=10, type=int, help="Number of queries (default: 10)")
    parser.add_argument("-k", default=100, type=int, help="Number of neighbors (default: 100)")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["b1", "i8", "u8", "f16", "bf16", "f32", "f64", "e4m3", "e3m2", "e2m3"],
        default="f32",
        help="Data type (default: f32)",
    )
    parser.add_argument("--metric", type=str, choices=["ip", "cos", "l2sq"], default="ip", help="Metric (default: ip)")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["usearch", "faiss", "numkong"],
        default="usearch",
        help="Backend to benchmark (default: usearch)",
    )

    args = parser.parse_args()
    run(
        n=args.n,
        q=args.q,
        k=args.k,
        ndim=args.ndim,
        dtype=args.dtype,
        metric=args.metric,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
