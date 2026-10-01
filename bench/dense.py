#!/usr/bin/env -S uv run --quiet --script
"""
USearch Dense Index Benchmark - the Python twin of `bench/dense.cpp`.

Measures ingest and search throughput, and sweeps the per-call expansion factor against
recall - the sweep the new surface makes cheap, since the candidate pool rides the call
rather than the handle, so one immutable index serves every point.

Usage:
    uv run bench/dense.py --count 100000 --dims 256
"""
# /// script
# dependencies = [
#   "numpy",
#   "usearch"
# ]
# ///

import argparse
import functools
import os

import numpy as np

from usearch import DenseIndex
from usearch.eval import integers_to_keys, keys_to_integers, measure_seconds, random_vectors


def parse_seed(text: str) -> int:
    return int.from_bytes(os.urandom(4), "little") if text == "random" else int(text)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the dense index for speed and recall.")
    parser.add_argument("--count", type=int, default=100_000, help="Vectors to index.")
    parser.add_argument("--dims", type=int, default=256, help="Dimensions per vector.")
    parser.add_argument("--queries", type=int, default=1_000, help="Query rows per search.")
    parser.add_argument("--top-k", type=int, default=10, help="Matches to retrieve per query.")
    parser.add_argument("--metric", type=str, default="angular", help="Distance function.")
    parser.add_argument("--quantization", type=str, default="bf16", help="Stored scalar type.")
    parser.add_argument("--dtype", type=str, default="f32", help="Input scalar type.")
    parser.add_argument("--seed", type=parse_seed, default=42, help="Seed of every random draw, or random to draw one.")
    arguments = parser.parse_args()

    print(f"Generating {arguments.count} x {arguments.dims} vectors with seed {arguments.seed}")
    vectors = random_vectors(arguments.count, arguments.dims, metric=arguments.metric, seed=arguments.seed)
    keys = integers_to_keys(np.arange(arguments.count, dtype=np.uint64))

    index = DenseIndex(dimensions=arguments.dims, metric=arguments.metric, quantization=arguments.quantization)
    index.reserve(arguments.count)
    print(f"Dispatching to {index.capabilities}")

    seconds, _ = measure_seconds(lambda: index.add(keys, vectors, dtype=arguments.dtype))
    print(f"Ingest: {arguments.count / seconds:,.0f} vectors/s over {seconds:.2f} s")
    print(f"Memory: {index.memory_usage / (1 << 20):,.1f} MB for {len(index):,} entries")

    sampled = np.random.default_rng(arguments.seed).choice(
        arguments.count, min(arguments.queries, arguments.count), replace=False
    )
    queries = vectors[sampled]
    expected = sampled.astype(np.uint64)

    # The allocating door first, then the same work through caller-owned buffers, so the
    # cost of the three arenas is visible rather than assumed.
    print(f"\n{'expansion':>10} {'queries/s':>12} {'recall@1':>10} {'distances/query':>17}")
    for expansion_factor in (0, 16, 32, 64, 128, 256):
        seconds, matches = measure_seconds(
            functools.partial(
                index.search,
                queries,
                arguments.top_k,
                expansion_factor=expansion_factor,
                dtype=arguments.dtype,
            )
        )
        found = keys_to_integers(matches.keys)[:, 0]
        recall = float(np.mean(found == expected))
        per_query = matches.computed_distances / len(queries)
        print(f"{expansion_factor:>10} {len(queries) / seconds:>12,.0f} {recall:>10.4f} {per_query:>17,.0f}")

    keys_out = np.empty((len(queries), arguments.top_k, 16), dtype=np.uint8)
    distances_out = np.empty((len(queries), arguments.top_k), dtype=np.float64)
    counts_out = np.empty(len(queries), dtype=np.uint64)
    seconds, _ = measure_seconds(
        lambda: index.search_into(queries, arguments.top_k, keys_out, distances_out, counts_out, dtype=arguments.dtype)
    )
    print(f"\nsearch_into, no allocation: {len(queries) / seconds:,.0f} queries/s")


if __name__ == "__main__":
    main()
