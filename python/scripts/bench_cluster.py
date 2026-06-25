#!/usr/bin/env -S uv run --quiet --script
"""
USearch Clustering Benchmarking

Benchmarks K-means clustering comparing NumPy, FAISS, and USearch.
Supports comma-separated --backend for comparing multiple backends.

Development setup (once):
    uv venv && source .venv/bin/activate
    uv pip install -e . --force-reinstall
    uv pip install faiss-cpu  # optional

Usage:
    python python/scripts/bench_cluster.py --vectors data.fbin --clusters 10 --backend usearch
    python python/scripts/bench_cluster.py --vectors data.fbin --clusters 10 --backend numpy,faiss,usearch
"""
# /// script
# dependencies = [
#   "numpy",
#   "faiss-cpu",
#   "usearch",
#   "tqdm"
# ]
# ///

import argparse
from dataclasses import dataclass
from time import perf_counter

import faiss
import numpy as np
from tqdm import tqdm

import usearch
from usearch.index import kmeans
from usearch.io import load_matrix

SUPPORTED_BACKENDS = {"numpy", "faiss", "usearch"}


@dataclass
class ClusterBenchResult:
    backend: str
    duration_seconds: float
    quality_euclidean: float
    quality_cosine: float
    cluster_sizes_mean: float
    cluster_sizes_std: float


def _parse_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _evaluate_euclidean(vectors, labels, centroids):
    distances = np.linalg.norm(vectors - centroids[labels], axis=1)
    return np.mean(distances)


def _evaluate_cosine(vectors, labels, centroids):
    vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    centroids_normalized = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    cosine_similarities = np.sum(vectors_normalized * centroids_normalized[labels], axis=1)
    return np.mean(1 - cosine_similarities)


def _cluster_numpy(vectors, clusters, max_iterations=100, tolerance=1e-4):
    indices = np.random.choice(vectors.shape[0], clusters, replace=False)
    centroids = vectors[indices]

    for _iteration in tqdm(range(max_iterations), desc="KMeans"):
        distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([vectors[labels == i].mean(axis=0) for i in range(clusters)])
        if np.linalg.norm(new_centroids - centroids) < tolerance:
            break
        centroids = new_centroids

    return labels, centroids


def _cluster_faiss(vectors, clusters, max_iterations=100):
    dimensions = vectors.shape[1]
    kmeans_index = faiss.Kmeans(dimensions, clusters, niter=max_iterations, verbose=False)
    kmeans_index.train(vectors)
    _, assignments = kmeans_index.index.search(vectors, 1)
    return assignments.flatten(), kmeans_index.centroids


def _cluster_usearch(vectors, clusters, max_iterations=100):
    assignments, _, centroids = kmeans(vectors, clusters, max_iterations=max_iterations)
    return assignments, centroids


_BACKEND_RUNNERS = {
    "numpy": _cluster_numpy,
    "faiss": _cluster_faiss,
    "usearch": _cluster_usearch,
}


def _run_cluster(vectors, clusters, max_iterations, backend) -> ClusterBenchResult:
    runner = _BACKEND_RUNNERS[backend]

    start = perf_counter()
    labels, centroids = runner(vectors, clusters, max_iterations=max_iterations)
    elapsed = perf_counter() - start

    quality_euclidean = _evaluate_euclidean(vectors, labels, centroids)
    quality_cosine = _evaluate_cosine(vectors, labels, centroids)
    sizes = np.unique(labels, return_counts=True)[1]

    return ClusterBenchResult(
        backend=backend,
        duration_seconds=elapsed,
        quality_euclidean=quality_euclidean,
        quality_cosine=quality_cosine,
        cluster_sizes_mean=np.mean(sizes),
        cluster_sizes_std=np.std(sizes),
    )


def _print_results(results: list[ClusterBenchResult], max_iterations: int) -> None:
    if not results:
        return
    header = (
        f"{'Backend':<10} {'Time':>10} {'Time/iter':>12} "
        f"{'Quality (L2)':>14} {'Quality (cos)':>14} {'Cluster sizes':>18}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for result in results:
        print(
            f"{result.backend:<10} {result.duration_seconds:>9,.2f}s {result.duration_seconds / max_iterations:>11,.3f}s "
            f"{result.quality_euclidean:>14.4f} {result.quality_cosine:>14.4f} "
            f"{result.cluster_sizes_mean:>8.0f} ± {result.cluster_sizes_std:<6.0f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark K-means clustering algorithms",
        epilog=(
            "Comma-separated --backend supported for comparing multiple backends.\n"
            "Examples:\n"
            "  bench_cluster.py --vectors data.fbin --clusters 10 --backend numpy,faiss,usearch"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vectors", type=str, required=True, help="Path to binary matrix file (.fbin)")
    parser.add_argument("-k", "--clusters", default=10, type=int, help="Number of clusters (default: 10)")
    parser.add_argument("-i", "--iterations", default=100, type=int, help="Maximum iterations (default: 100)")
    parser.add_argument("-n", "--count", type=int, help="Limit number of vectors to use")
    parser.add_argument(
        "--backend",
        default="numpy",
        help=f"Backend(s), comma-separated (default: numpy). Choices: {', '.join(sorted(SUPPORTED_BACKENDS))}",
    )

    args = parser.parse_args()

    backends = _parse_csv(args.backend)
    for backend in backends:
        if backend not in SUPPORTED_BACKENDS:
            parser.error(f"Unknown backend '{backend}'. Choices: {', '.join(sorted(SUPPORTED_BACKENDS))}")

    # Library versions
    print(f"USearch v{usearch.VERSION_MAJOR}.{usearch.VERSION_MINOR}.{usearch.VERSION_PATCH}")
    if "faiss" in backends:
        print(f"FAISS v{faiss.__version__}")

    # Load data
    vectors = load_matrix(args.vectors, count_rows=args.count)
    print(f"Loaded: {vectors.shape[0]:,} x {vectors.shape[1]}, clusters={args.clusters}, iterations={args.iterations}")

    # Run benchmarks
    results: list[ClusterBenchResult] = []
    for backend in backends:
        print(f"\n--- {backend} ---")
        try:
            results.append(_run_cluster(vectors, args.clusters, args.iterations, backend))
        except KeyboardInterrupt:
            print("\n\nInterrupted — printing results collected so far.")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    _print_results(results, args.iterations)

    # Random baseline comparison
    if results:
        random_labels = np.random.randint(0, args.clusters, size=vectors.shape[0])
        last_centroids_result = results[-1]
        # Use last backend's centroids for random baseline
        _, centroids = _BACKEND_RUNNERS[last_centroids_result.backend](vectors, args.clusters, max_iterations=1)
        random_quality = _evaluate_euclidean(vectors, random_labels, centroids)
        random_cosine = _evaluate_cosine(vectors, random_labels, centroids)
        print(f"\nRandom assignment baseline: L2={random_quality:.4f}, cos={random_cosine:.4f}")


if __name__ == "__main__":
    main()
