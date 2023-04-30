import datetime
import numpy as np

import faiss
import usearch


print('Welcome to basic kANN benchmark!')


def read_matrix(filename: str, start_row: int = 0, count_rows: int = None):
    """
    Read *.ibin, *.hbin, *.fbin, *.dbin files with matrixes.
    Args:
        :param filename (str): path to the matrix file
        :param start_row (int): start reading vectors from this index
        :param count_rows (int): number of vectors to read. If None, read all vectors
    Returns:
        Parsed matrix (numpy.ndarray)
    """
    dtype = np.float32
    scalar_size = 4
    if filename.endswith('.fbin'):
        dtype = np.float32
        scalar_size = 4
    elif filename.endswith('.dbin'):
        dtype = np.float64
        scalar_size = 8
    elif filename.endswith('.hbin'):
        dtype = np.float16
        scalar_size = 2
    elif filename.endswith('.ibin'):
        dtype = np.int32
        scalar_size = 4
    else:
        raise Exception('Unknown file type')
    with open(filename, 'rb') as f:
        rows, cols = np.fromfile(f, count=2, dtype=np.int32)
        rows = (rows - start_row) if count_rows is None else count_rows
        arr = np.fromfile(
            f, count=rows * cols, dtype=dtype,
            offset=start_row * scalar_size * cols)
    return arr.reshape(rows, cols)


print('Will read datasets!')
xb = read_matrix('datasets/wiki_1M/base.1M.fbin')
xq = read_matrix('datasets/wiki_1M/query.public.100K.fbin')

d = 256  # vector size
M = 16
efSearch = 32  # number of entry points (neighbors) we use on each layer
efConstruction = 128  # number of entry points used on each layer
labels = np.arange(xb.shape[0], dtype=np.longlong)
print('Will benchmark algorithms!')


def measure(f) -> float:
    a = datetime.datetime.now()
    f()
    b = datetime.datetime.now()
    c = b - a
    print(f'Took: {c.seconds} seconds')
    return c.seconds


def construct_both() -> tuple:

    index_meta = faiss.IndexHNSWFlat(d, M)
    index_meta.hnsw.efSearch = efSearch
    index_meta.hnsw.efConstruction = efConstruction
    dt_meta = measure(lambda: index_meta.add(xb))

    index_unum = usearch.make_index(
        expansion_add=efConstruction,
        expansion_search=efSearch,
        connectivity=M,
    )
    dt_unum = measure(lambda: index_unum.add(labels, xb, copy=False))

    return dt_meta, dt_unum


experiments = 10
durations = [construct_both() for _ in range(experiments)]
dt_meta = sum(x[0] for x in durations)
dt_unum = sum(x[1] for x in durations)
print(f'- FAISS vectors per second: {xb.shape[0]*experiments/dt_meta:.2f}')
print(f'- USearch vectors per second: {xb.shape[0]*experiments/dt_unum:.2f}')
