import numpy as np
from ucall.client import Client

from usearch.index import Matches


def _vector_to_ascii(vector: np.ndarray) -> str | None:
    if vector.dtype != np.int8 and vector.dtype != np.uint8 and vector.dtype != np.byte:
        return None
    if not np.all((vector >= 0) | (vector <= 100)):
        return None

    # Let's map [0, 100] to the range from [23, 123],
    # poking 60 and replacing with the 124.
    vector += 23
    vector[vector == 60] = 124
    ascii_vector = str(vector)
    return ascii_vector


class IndexClient:
    def __init__(self, uri: str = "127.0.0.1", port: int = 8545, use_http: bool = True) -> None:
        self.client = Client(uri=uri, port=port, use_http=use_http)

    def add_one(self, key: int, vector: np.ndarray):
        assert isinstance(key, int)
        assert isinstance(vector, np.ndarray)
        vector = vector.flatten()
        ascii_vector = _vector_to_ascii(vector)
        if ascii_vector:
            self.client.add_ascii(key=key, string=ascii_vector)
        else:
            self.client.add_one(key=key, vectors=vector)

    def add_many(self, keys: np.ndarray, vectors: np.ndarray):
        assert isinstance(keys, int)
        assert isinstance(vectors, np.ndarray)
        assert keys.ndim == 1 and vectors.ndim == 2
        assert keys.shape[0] == vectors.shape[0]
        self.client.add_many(keys=keys, vectors=vectors)

    def add(self, keys: np.ndarray | int, vectors: np.ndarray):
        if isinstance(keys, int) or len(keys) == 1:
            return self.add_one(keys, vectors)
        else:
            return self.add_many(keys, vectors)

    def search_one(self, vector: np.ndarray, count: int) -> Matches:
        matches: list[dict] = []
        vector = vector.flatten()
        ascii_vector = _vector_to_ascii(vector)
        if ascii_vector:
            matches = self.client.search_ascii(string=ascii_vector, count=count)
        else:
            matches = self.client.search_one(vector=vector, count=count)

        print(matches.data)
        matches = matches.json

        keys = np.array((1, count), dtype=np.uint32)
        distances = np.array((1, count), dtype=np.float32)
        counts = np.array((1), dtype=np.uint32)
        for col, result in enumerate(matches):
            keys[0, col] = result["key"]
            distances[0, col] = result["distance"]
        counts[0] = len(matches)

        return keys, distances, counts

    def search_many(self, vectors: np.ndarray, count: int) -> Matches:
        batch_size: int = vectors.shape[0]
        list_of_matches: list[list[dict]] = self.client.search_many(vectors=vectors, count=count)

        keys = np.array((batch_size, count), dtype=np.uint32)
        distances = np.array((batch_size, count), dtype=np.float32)
        counts = np.array((batch_size), dtype=np.uint32)
        for row, matches in enumerate(list_of_matches):
            for col, result in enumerate(matches):
                keys[row, col] = result["key"]
                distances[row, col] = result["distance"]
            counts[row] = len(results)

        return keys, distances, counts

    def search(self, vectors: np.ndarray, count: int) -> Matches:
        if vectors.ndim == 1 or (vectors.ndim == 2 and vectors.shape[0] == 1):
            return self.search_one(vectors, count)
        else:
            return self.search_many(vectors, count)

    def __len__(self):
        return self.client.size().json()

    @property
    def ndim(self):
        return self.client.ndim().json()

    def capacity(self):
        return self.client.capacity().json()

    def connectivity(self):
        return self.client.connectivity().json()

    def load(self, path: str):
        raise NotImplementedError()

    def view(self, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()


if __name__ == "__main__":
    index = IndexClient()
    index.add(42, np.array([0.4] * 256, dtype=np.float32))
    results = index.search(np.array([0.4] * 256, dtype=np.float32), 10)
    print(results)
