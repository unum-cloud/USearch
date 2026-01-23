#!/usr/bin/env -S uv run --quiet --script
"""
FAISS Index Wrapper for USearch Benchmarking

Provides FAISS index implementations compatible with USearch benchmarking framework
for performance comparison between USearch and FAISS libraries.

Usage:
    uv run python/scripts/index_faiss.py
    
Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "numpy",
#   "faiss-cpu",
#   "usearch"
# ]
# ///

import os
from typing import Optional

import numpy as np
from faiss import IndexHNSWFlat, IndexIVFPQ, read_index

import usearch
from usearch.index import BatchMatches
from usearch.index import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
)


class IndexFAISS:
    def __init__(
        self,
        index: IndexHNSWFlat = None,
        ndim: int = 0,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
        path: Optional[os.PathLike] = None,
        *args,
        **kwargs,
    ):
        if index is None:
            index = IndexHNSWFlat(ndim, connectivity)
            index.hnsw.efConstruction = expansion_add
            index.hnsw.efSearch = expansion_search

        self._faiss = index
        self._specs = {
            "Class": "usearch.IndexFAISS",
            "Dimensions": ndim,
            "Connectivity": connectivity,
            "Expansion@Add": expansion_add,
            "Expansion@Search": expansion_search,
        }

        self.path = path

    def add(self, keys, vectors):
        # Adding keys isn't supported for most index types
        # self._faiss.add_with_ids(vectors, keys)
        self._faiss.add(vectors)

    def search(self, queries, k: int) -> BatchMatches:
        distances, keys = self._faiss.search(queries, k)
        return BatchMatches(keys, distances, np.array([k] * queries.shape[0]))

    def __len__(self) -> int:
        return self._faiss.ntotal

    def clear(self):
        self._faiss.reset()

    @property
    def specs(self) -> dict:
        self._specs.update(
            {
                "Size": len(self),
            }
        )
        return self._specs

    def load(self, path: os.PathLike):
        self._faiss = read_index(path)


class IndexQuantizedFAISS(IndexFAISS):
    def __init__(
        self,
        train: np.ndarray,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
        *args,
        **kwargs,
    ):
        ndim = train.shape[1]
        super().__init__(
            ndim=ndim,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            *args,
            **kwargs,
        )

        nlist = 10000  # Number of inverted lists (number of partitions).
        nsegment = 16  # Number of segments for PQ (number of subquantizers).
        nbit = 8  # Number of bits to encode each segment.

        self._original_faiss = self._faiss
        self._faiss = IndexIVFPQ(self._faiss, ndim, nlist, nsegment, nbit)

        self._faiss.train(train)
        self._faiss.nprobe = 10
        self._specs["Class"] = "usearch.IndexQuantizedFAISS"
