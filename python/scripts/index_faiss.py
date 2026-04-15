"""
FAISS HNSW Index Wrapper

Provides a USearch-compatible interface around FAISS HNSW indices for
benchmarking. Supports dense (IndexHNSWFlat/IndexHNSWSQ) and binary
(IndexBinaryHNSW) index types with batched tqdm progress bars.
"""

from dataclasses import dataclass

import faiss
import numpy as np
from tqdm import tqdm

# FAISS dtype → ScalarQuantizer type mapping (no learnable quantizers).
_SCALAR_QUANTIZER_MAP: dict[str, int | None] = {
    "f32": None,
    "f16": faiss.ScalarQuantizer.QT_fp16,
    "fp16": faiss.ScalarQuantizer.QT_fp16,
    "bf16": faiss.ScalarQuantizer.QT_bf16,
    "u8": faiss.ScalarQuantizer.QT_8bit_direct,
    "uint8": faiss.ScalarQuantizer.QT_8bit_direct,
    "i8": faiss.ScalarQuantizer.QT_8bit_direct_signed,
    "int8": faiss.ScalarQuantizer.QT_8bit_direct_signed,
}

DENSE_DTYPES = set(_SCALAR_QUANTIZER_MAP.keys())
BINARY_DTYPES = {"b1", "bits"}
ALL_DTYPES = DENSE_DTYPES | BINARY_DTYPES

BITWISE_METRICS = {"hamming", "tanimoto", "jaccard", "sorensen"}


def hardware_acceleration() -> str:
    """Detect which SIMD backend FAISS loaded at import time."""
    if hasattr(faiss, "_swigfaiss_avx512"):
        return "avx512"
    if hasattr(faiss, "_swigfaiss_avx2"):
        return "avx2"
    return "generic"


def hardware_acceleration_available() -> str:
    """Comma-separated list of SIMD capabilities available in this FAISS build."""
    flags = []
    if getattr(faiss, "has_AVX512", False):
        flags.append("avx512")
    if getattr(faiss, "has_AVX512_SPR", False):
        flags.append("avx512-spr")
    if getattr(faiss, "has_AVX2", False):
        flags.append("avx2")
    return ", ".join(flags) if flags else "generic"


def version() -> str:
    """FAISS version string."""
    return getattr(faiss, "__version__", "unknown")


@dataclass
class FaissMatches:
    """Match result compatible with USearch BatchMatches for recall computation."""

    keys: np.ndarray
    distances: np.ndarray


class IndexFAISS:
    """FAISS HNSW wrapper matching the USearch Index interface for benchmarking.

    Keys passed to :meth:`add` are ignored — FAISS assigns sequential IDs
    starting from 0. The :meth:`search` return type mimics USearch's
    ``BatchMatches`` with ``.keys`` and ``.distances`` attributes.
    """

    def __init__(
        self,
        dimensions: int,
        metric: str,
        dtype: str = "f32",
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ):
        self.dimensions = dimensions
        self.metric_name = metric
        self.dtype = dtype
        self._is_binary = metric in BITWISE_METRICS

        if self._is_binary:
            dimensions_bits = dimensions * 8 if dtype in ("b1", "bits") else dimensions
            self._index = faiss.IndexBinaryHNSW(dimensions_bits, connectivity)
        else:
            if dtype not in _SCALAR_QUANTIZER_MAP:
                raise ValueError(
                    f"FAISS does not support dtype '{dtype}'. Supported: {', '.join(sorted(_SCALAR_QUANTIZER_MAP))}"
                )
            faiss_metric = faiss.METRIC_L2 if metric == "l2sq" else faiss.METRIC_INNER_PRODUCT
            sq_type = _SCALAR_QUANTIZER_MAP[dtype]
            if sq_type is None:
                self._index = faiss.IndexHNSWFlat(dimensions, connectivity, faiss_metric)
            else:
                self._index = faiss.IndexHNSWSQ(dimensions, sq_type, connectivity, faiss_metric)

        self._index.hnsw.efConstruction = expansion_add
        self._index.hnsw.efSearch = expansion_search

    def add(self, _keys, vectors: np.ndarray, *, log: bool = False, dtype: str = "") -> None:
        """Add vectors to the index.

        :param _keys: Ignored. FAISS uses sequential IDs internally.
        :param vectors: Row-major matrix of vectors to add.
        :param log: Show tqdm progress bar.
        :param dtype: Ignored. Accepted for USearch API compatibility.
        """
        if self._is_binary:
            data = vectors if vectors.dtype == np.uint8 else vectors.astype(np.uint8)
        else:
            data = vectors if vectors.dtype == np.float32 else vectors.astype(np.float32)

        count = data.shape[0]
        batch_size = max(10_000, count // 20)

        if log:
            with tqdm(total=count, desc="Add", unit="vector") as progress_bar:
                for start in range(0, count, batch_size):
                    end = min(start + batch_size, count)
                    self._index.add(data[start:end])
                    progress_bar.update(end - start)
        else:
            self._index.add(data)

    def search(self, queries: np.ndarray, neighbors_count: int, *, log: bool = False, dtype: str = "") -> FaissMatches:
        """Search the index for nearest neighbors.

        :param queries: Row-major matrix of query vectors.
        :param neighbors_count: Number of neighbors to retrieve per query.
        :param log: Show tqdm progress bar.
        :param dtype: Ignored. Accepted for USearch API compatibility.
        :return: FaissMatches with ``.keys`` and ``.distances`` arrays.
        """
        if self._is_binary:
            data = queries if queries.dtype == np.uint8 else queries.astype(np.uint8)
        else:
            data = queries if queries.dtype == np.float32 else queries.astype(np.float32)

        count = data.shape[0]
        batch_size = max(100, count // 20)

        if log:
            all_distances = []
            all_ids = []
            with tqdm(total=count, desc="Search", unit="vector") as progress_bar:
                for start in range(0, count, batch_size):
                    end = min(start + batch_size, count)
                    distances, ids = self._index.search(data[start:end], neighbors_count)
                    all_distances.append(distances)
                    all_ids.append(ids)
                    progress_bar.update(end - start)
            return FaissMatches(keys=np.vstack(all_ids), distances=np.vstack(all_distances))

        distances, ids = self._index.search(data, neighbors_count)
        return FaissMatches(keys=ids, distances=distances)

    def save(self, path: str) -> None:
        """Save the index to disk."""
        if self._is_binary:
            faiss.write_index_binary(self._index, path)
        else:
            faiss.write_index(self._index, path)

    def load(self, path: str) -> None:
        """Load the index from disk, replacing the current index."""
        if self._is_binary:
            self._index = faiss.read_index_binary(path)
        else:
            self._index = faiss.read_index(path)

    @property
    def hardware_acceleration(self) -> str:
        """SIMD backend FAISS is using (e.g., 'avx512', 'avx2', 'generic')."""
        return hardware_acceleration()

    def __len__(self) -> int:
        return self._index.ntotal
