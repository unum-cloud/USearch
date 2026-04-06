#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ucall.rich_posix import Server  # type: ignore[import-not-found]

from usearch.index import Index, Key

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from usearch.index import BatchMatches, Matches


def _ascii_to_vector(string: str) -> NDArray[Any]:
    """
    WARNING: A dirty performance hack!
    Assuming the `i8` vectors in our implementations are just integers,
    and generally contain scalars in the [0, 100] range, we can transmit
    them as JSON-embedded strings. The only symbols we must avoid are
    the double-quote '"' (code 22) and backslash '\' (code 60).
    Printable ASCII characters are in [20, 126].
    """
    vector = np.array(string, dtype=np.int8)
    vector[vector == 124] = 60
    vector -= 23
    return vector


def serve(
    ndim_: int,
    metric: str = "ip",
    port: int = 8545,
    threads: int = 1,
    path: str = "index.usearch",
    immutable: bool = False,
) -> None:
    server = Server(port=port)
    index = Index(ndim=ndim_, metric=metric)

    if os.path.exists(path):
        if immutable:
            index.view(path)
        else:
            index.load(path)

    @server  # type: ignore[untyped-decorator]
    def size() -> int:
        return len(index)

    @server  # type: ignore[untyped-decorator]
    def ndim() -> int:
        return index.ndim

    @server  # type: ignore[untyped-decorator]
    def capacity() -> int:
        return index.capacity

    @server  # type: ignore[untyped-decorator]
    def connectivity() -> int:
        return index.connectivity

    @server  # type: ignore[untyped-decorator]
    def add_one(key: int, vector: NDArray[Any]) -> None:
        keys = np.array([key], dtype=Key)
        vectors = vector.flatten().reshape(vector.shape[0], 1)
        index.add(keys, vectors)

    @server  # type: ignore[untyped-decorator]
    def add_many(keys: NDArray[Any], vectors: NDArray[Any]) -> None:
        index.add(keys, vectors, threads=threads)

    @server  # type: ignore[untyped-decorator]
    def search_one(vector: NDArray[Any], count: int) -> list[tuple[int, float]]:
        vectors = vector.reshape(vector.shape[0], 1)
        results: Matches | BatchMatches = index.search(vectors, count)
        return results.to_list()

    @server  # type: ignore[untyped-decorator]
    def search_many(vectors: NDArray[Any], count: int) -> list[tuple[int, float]]:
        results: Matches | BatchMatches = index.search(vectors, count)
        return results.to_list()

    @server  # type: ignore[untyped-decorator]
    def add_ascii(key: int, string: str) -> None:
        add_one(key, _ascii_to_vector(string))

    @server  # type: ignore[untyped-decorator]
    def search_ascii(string: str, count: int) -> list[tuple[int, float]]:
        return cast("list[tuple[int, float]]", search_one(_ascii_to_vector(string), count))

    try:
        server.run()
    except KeyboardInterrupt:
        if not immutable:
            index.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="log server activity")
    parser.add_argument("--ndim", type=int, help="dimensionality of the vectors")
    parser.add_argument("--immutable", type=bool, default=False, help="the index can not be updated")

    parser.add_argument(
        "--metric",
        type=str,
        default="ip",
        choices=["ip", "cos", "l2sq", "haversine"],
        help="distance function to compare vectors",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8545,
        help="port to open for client connections",
    )
    parser.add_argument("-j", "--threads", type=int, default=1, help="number of CPU threads to use")
    parser.add_argument("--path", type=str, default="index.usearch", help="where to store the index")

    args = parser.parse_args()
    assert args.ndim is not None, "Define the number of dimensions!"
    serve(
        ndim_=args.ndim,
        metric=args.metric,
        threads=args.threads,
        port=args.port,
        path=args.path,
        immutable=args.immutable,
    )
