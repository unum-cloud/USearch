#!/usr/bin/env -S uv run --quiet --script
"""
USearch Benchmarking Utility

This script provides benchmarking functions for USearch approximate nearest neighbor
search performance evaluation across different configurations and data types.

Usage:
    uv run python/scripts/bench.py

Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "usearch",
#   "tqdm"
# ]
# ///

import itertools
from typing import List
from dataclasses import asdict
import argparse # Added for CLI functionality

import numpy as np
import pandas as pd

import usearch
from usearch.index import Index, Key, MetricKind, ScalarKind, search # Added search import
from usearch.numba import jit as njit
from usearch.eval import Evaluation, AddTask, SearchTask # Adjusted imports as per eval.py
from usearch.index import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
)


def bench_speed(
    eval: Evaluation,
    connectivity: int = DEFAULT_CONNECTIVITY,
    expansion_add: int = DEFAULT_EXPANSION_ADD,
    expansion_search: int = DEFAULT_EXPANSION_SEARCH,
    jit: bool = False,
    train: bool = False,
) -> pd.DataFrame:
    # Build various indexes:
    indexes = []
    jit_options = [False, True] if jit else [False]
    dtype_options = [ScalarKind.F32, ScalarKind.F16, ScalarKind.BF16, ScalarKind.I8]
    for jit, dtype in itertools.product(jit_options, dtype_options):
        metric = MetricKind.IP
        if jit:
            metric = njit(eval.ndim, metric, dtype)
        index = Index(
            ndim=eval.ndim,
            metric=metric,
            dtype=dtype,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            connectivity=connectivity,
            path="USearch" + ["", "+JIT"][jit] + ":" + str(dtype),
        )

        # Skip the cases, where JIT-ing is impossible
        if jit and not index.jit:
            continue
        indexes.append(index)

    # Add FAISS indexes to the mix:
    try:
        from index_faiss import IndexFAISS, IndexQuantizedFAISS

        indexes.append(
            IndexFAISS(
                ndim=eval.ndim,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
                connectivity=connectivity,
                path="FAISS:f32",
            )
        )
        if train:
            indexes.append(
                IndexQuantizedFAISS(
                    train=eval.tasks[0].vectors,
                    expansion_add=expansion_add,
                    expansion_search=expansion_search,
                    connectivity=connectivity,
                    path="FAISS+IVFPQ:f32",
                )
            )
    except (ImportError, ModuleNotFoundError):
        pass

    # Time to evaluate:
    results = [eval(index) for index in indexes]
    return pd.DataFrame(
        {
            "names": [i.path for i in indexes],
            "add_per_second": [x["add_per_second"] for x in results],
            "search_per_second": [x["search_per_second"] for x in results],
            "recall_at_one": [x["recall_at_one"] for x in results],
        }
    )


def bench_params(
    count: int = 1_000_000,
    connectivities: List[int] = [16], # Changed to list for consistency
    dimensions: List[int] = [256],    # Changed to list for consistency
    expansion_add: int = DEFAULT_EXPANSION_ADD,
    expansion_search: int = DEFAULT_EXPANSION_SEARCH,
) -> pd.DataFrame:
    """Measures indexing speed for different dimensionality vectors.

    :param count: Number of vectors, defaults to 1_000_000
    :type count: int, optional
    """

    results = []
    for connectivity, ndim in itertools.product(connectivities, dimensions):
        task = AddTask(
            keys=np.arange(count, dtype=Key),
            vectors=np.random.rand(count, ndim).astype(np.float32),
        )
        eval_obj = Evaluation(tasks=[task], ndim=ndim, count=count) # Pass count to Evaluation
        index = Index(
            ndim=ndim,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )
        # Assuming Evaluation instance has a __call__ method for direct evaluation
        result_dict = eval_obj(index) # Pass index to evaluation
        result_dict["ndim"] = ndim
        result_dict["connectivity"] = connectivity
        results.append(result_dict)

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="USearch Benchmarking Utility",
        epilog="Run `python bench.py --run` to execute a default benchmark."
    )
    parser.add_argument(
        "--run", action="store_true", help="Run a quick default benchmark"
    )
    parser.add_argument(
        "--count", type=int, default=1_000, help="Number of vectors to index (default: 1000)"
    )
    parser.add_argument(
        "--ndim", type=int, default=256, help="Dimensions of vectors (default: 256)"
    )
    parser.add_argument(
        "--connectivity", type=int, default=16, help="HNSW connectivity (default: 16)"
    )
    parser.add_argument(
        "--test_core", action="store_true", help="Run a core usearch functionality test"
    )

    args = parser.parse_args()

    if args.run:
        print(f"Running quick benchmark with count={args.count}, ndim={args.ndim}, connectivity={args.connectivity}")
        
        keys = np.arange(args.count, dtype=Key)
        vectors = np.random.rand(args.count, args.ndim).astype(np.float32)

        queries_count = min(100, args.count)
        queries = vectors[:queries_count] # Use first 100 vectors as queries

        # Compute exact neighbors using usearch.index.search for ground truth
        # MetricKind.IP is assumed as default in bench_speed
        exact_matches = search(
            vectors, 
            queries, # Passed positionally
            1,       # k (count) passed positionally
            metric=MetricKind.IP, 
            exact=True
        )
        # The neighbors array should be 2D, where each row is a list of neighbors for a query
        # We need the keys, not distances, for recall calculation.
        neighbors = exact_matches.keys.reshape(-1, 1).astype(Key)
        
        eval_obj = Evaluation(
            tasks=[
                AddTask(keys=keys, vectors=vectors),
                SearchTask(queries=queries, neighbors=neighbors)
            ],
            ndim=args.ndim,
            count=args.count,
        )

        results_df = bench_speed(
            eval=eval_obj,
            connectivity=args.connectivity,
            expansion_add=DEFAULT_EXPANSION_ADD,
            expansion_search=DEFAULT_EXPANSION_SEARCH,
            jit=False, 
            train=False 
        )
        print("\n--- Benchmark Results ---")
        print(results_df)

    elif args.test_core:
        print("\n--- Running Core USearch Functionality Test ---")
        try:
            index = Index(ndim=3)
            vector = np.array([0.2, 0.6, 0.4])
            key = 42
            index.add(key, vector)
            print(f"Added vector with key {key}: {vector}")
            
            matches = index.search(vector, 1)
            assert matches[0].key == key
            assert matches[0].distance <= 0.001
            print(f"Found match: Key={matches[0].key}, Distance={matches[0].distance}")
            print("Core USearch functionality test PASSED.")
        except Exception as e:
            print(f"Core USearch functionality test FAILED: {e}")
            import traceback
            traceback.print_exc()
        print("--------------------------------------------")

    else:
        parser.print_help()