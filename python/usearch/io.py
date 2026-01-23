import os
import struct
import typing

import numpy as np


def numpy_scalar_size(dtype) -> int:
    return {
        np.float64: 8,
        np.int64: 8,
        np.uint64: 8,
        np.float32: 4,
        np.int32: 4,
        np.uint32: 4,
        np.float16: 2,
        np.int16: 2,
        np.uint16: 2,
        np.int8: 1,
        np.uint8: 1,
    }[dtype]


def guess_numpy_dtype_from_filename(filename) -> typing.Optional[type]:
    if filename.endswith(".fbin"):
        return np.float32
    elif filename.endswith(".dbin"):
        return np.float64
    elif filename.endswith(".hbin"):
        return np.float16
    elif filename.endswith(".ibin"):
        return np.int32
    elif filename.endswith(".bbin"):
        return np.uint8
    elif filename.endswith(".i8bin"):
        return np.int8
    elif filename.endswith(".i32bin"):
        return np.int32
    elif filename.endswith(".f32bin"):
        return np.float32
    else:
        return None


def load_matrix(
    filename: str,
    start_row: int = 0,
    count_rows: int = None,
    view: bool = False,
    dtype: typing.Optional[type] = None,
) -> typing.Optional[np.ndarray]:
    """Read *.ibin, *.bbib, *.hbin, *.fbin, *.dbin, *.i8bin, *.i32bin files with matrices.

    :param filename: path to the matrix file
    :param start_row: start reading vectors from this index
    :param count_rows: number of vectors to read. If None, read all vectors
    :param view: set to `True` to memory-map the file instead of loading to RAM

    :return: parsed matrix
    :rtype: numpy.ndarray
    """
    if dtype is None:
        dtype = guess_numpy_dtype_from_filename(filename)
        if dtype is None:
            raise Exception("Unknown file type")
    scalar_size = numpy_scalar_size(dtype)

    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        rows, cols = np.fromfile(f, count=2, dtype=np.int32).astype(np.uint64)
        
        # Validate file size matches expected data size
        f.seek(0, 2)  # Go to end
        file_size = f.tell()
        expected_size = 8 + (rows * cols * scalar_size)  # Header + data
        
        if file_size != expected_size:
            if file_size < expected_size:
                raise ValueError(f"File {filename} is truncated. Expected {expected_size:,} bytes, got {file_size:,} bytes")
            else:
                raise ValueError(f"File {filename} is larger than expected. Expected {expected_size:,} bytes, got {file_size:,} bytes")
        
        f.seek(8)  # Back to start of data
        rows = (rows - start_row) if count_rows is None else count_rows
        row_offset = start_row * scalar_size * cols

        if view:
            return np.memmap(
                f,
                dtype=dtype,
                mode="r",
                offset=8 + row_offset,
                shape=(rows, cols),
            )
        else:
            return np.fromfile(
                f,
                count=rows * cols,
                dtype=dtype,
                offset=row_offset,
            ).reshape(rows, cols)


def save_matrix(vectors: np.ndarray, filename: str):
    """Write *.ibin, *.bbib, *.hbin, *.fbin, *.dbin, *.i8bin, *.i32bin, *.f32bin files with matrices.

    :param vectors: the matrix to serialize
    :type vectors: numpy.ndarray
    :param filename: path to the matrix file
    :type filename: str
    """
    if filename.endswith(".fbin"):
        dtype = np.float32
    elif filename.endswith(".dbin"):
        dtype = np.float64
    elif filename.endswith(".hbin"):
        dtype = np.float16
    elif filename.endswith(".ibin"):
        dtype = np.int32
    elif filename.endswith(".bbin"):
        dtype = np.uint8
    elif filename.endswith(".i8bin"):
        dtype = np.int8
    elif filename.endswith(".i32bin"):
        dtype = np.int32
    elif filename.endswith(".f32bin"):
        dtype = np.float32
    else:
        dtype = vectors.dtype

    assert len(vectors.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        count, dim = vectors.shape
        f.write(struct.pack("<i", count))
        f.write(struct.pack("<i", dim))
        vectors.astype(dtype).flatten().tofile(f)
