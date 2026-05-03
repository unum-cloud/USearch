#!/usr/bin/env -S uv run --quiet --script
"""
USearch Index Joining Benchmark

Benchmarks cross-modal join and search on multimodal datasets (e.g., images
and texts). Evaluates self-recall, cross-recall, and bipartite join quality.

Usage:
    uv run python/scripts/join.py --help
    uv run python/scripts/join.py \\
        --vectors-a datasets/cc_3M/texts.fbin \\
        --vectors-b datasets/cc_3M/images.fbin \\
        --metric cos -n 100000

Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "numpy",
#   "numkong",
#   "usearch",
#   "tqdm"
# ]
# ///

import argparse
from time import perf_counter

import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

from usearch.eval import measure_seconds, random_vectors
from usearch.index import CompiledMetric, Index, MetricKind, MetricSignature
from usearch.io import load_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cross-modal index joining and search",
        epilog="If --vectors-a/--vectors-b not provided, generates synthetic data.",
    )
    parser.add_argument("--vectors-a", type=str, help="Path to first dataset (.fbin)")
    parser.add_argument("--vectors-b", type=str, help="Path to second dataset (.fbin)")
    parser.add_argument("--metric", type=str, choices=["ip", "cos", "l2sq"], default="cos", help="Distance metric")
    parser.add_argument("-n", "--count", type=int, default=100_000, help="Max vectors per dataset (default: 100000)")
    parser.add_argument("--ndim", type=int, default=256, help="Dimensions for synthetic data (default: 256)")
    parser.add_argument("-k", type=int, default=10, help="Number of neighbors for recall evaluation (default: 10)")
    parser.add_argument("--dtype", type=str, default="f32", help="Quantization type (default: f32)")
    parser.add_argument("--diagnostics", action="store_true", help="Run self-recall and cross-recall diagnostics")

    args = parser.parse_args()

    # Load or generate data
    if args.vectors_a and args.vectors_b:
        vectors_a = load_matrix(args.vectors_a, count_rows=args.count)
        vectors_b = load_matrix(args.vectors_b, count_rows=args.count)
        print(f"Loaded datasets: A={vectors_a.shape}, B={vectors_b.shape}")
    else:
        print(f"Generating synthetic data: {args.count:,} x {args.ndim}")
        vectors_a = random_vectors(args.count, ndim=args.ndim).astype(np.float32)
        vectors_b = random_vectors(args.count, ndim=args.ndim).astype(np.float32)

    ndim = vectors_a.shape[1]
    min_elements = min(vectors_a.shape[0], vectors_b.shape[0])

    # Build metric
    try:
        from numkong import pointer_to_angular

        metric = CompiledMetric(
            pointer=pointer_to_angular("f32"),
            kind=MetricKind.Cos,
            signature=MetricSignature.ArrayArraySize,
        )
    except ImportError:
        metric = MetricKind.Cos

    # Build indexes
    print("--- Indexing ---")
    index_a = Index(ndim, metric=metric, dtype=args.dtype)
    index_b = Index(ndim, metric=metric, dtype=args.dtype)

    index_a.add(None, vectors_a, log=True)
    index_b.add(None, vectors_b, log=True)
    print(f"Indexed: A={len(index_a):,}, B={len(index_b):,}")

    # Diagnostics
    if args.diagnostics:
        print("\n--- Diagnostics ---")

        # Pairwise similarity
        mean_sim = 0.0
        for i in tqdm(range(min_elements), desc="Pairwise Similarity"):
            a_vec, b_vec = vectors_a[i], vectors_b[i]
            a_norm, b_norm = norm(a_vec), norm(b_vec)
            if a_norm > 0 and b_norm > 0:
                mean_sim += dot(a_vec, b_vec) / (a_norm * b_norm)
        mean_sim /= min_elements
        print(f"Average pairwise cosine similarity: {mean_sim:.4f}")

        search_kwargs = dict(count=args.k, log=True)

        secs, recall_a = measure_seconds(
            lambda: index_a.search(vectors_a, **search_kwargs).recall(np.arange(len(index_a)))
        )
        print(f"Self-recall @{args.k} of A: {recall_a * 100:.2f}% ({secs:.2f}s)")

        secs, recall_b = measure_seconds(
            lambda: index_b.search(vectors_b, **search_kwargs).recall(np.arange(len(index_b)))
        )
        print(f"Self-recall @{args.k} of B: {recall_b * 100:.2f}% ({secs:.2f}s)")

        secs, recall_ab = measure_seconds(
            lambda: index_b.search(vectors_a, **search_kwargs).recall(np.arange(min_elements))
        )
        print(f"Cross-recall @{args.k} A->B: {recall_ab * 100:.2f}% ({secs:.2f}s)")

        secs, recall_ba = measure_seconds(
            lambda: index_a.search(vectors_b, **search_kwargs).recall(np.arange(min_elements))
        )
        print(f"Cross-recall @{args.k} B->A: {recall_ba * 100:.2f}% ({secs:.2f}s)")

    # Join
    print("\n--- Join ---")
    start_time = perf_counter()
    bimapping = index_a.join(index_b, max_proposals=100)
    join_elapsed = perf_counter() - start_time

    recall = sum(1 for i, j in bimapping.items() if i == j)
    recall_pct = recall * 100.0 / min_elements
    print(f"Found {len(bimapping):,} pairings in {join_elapsed:.2f}s, {recall_pct:.2f}% exact matches")


if __name__ == "__main__":
    main()
