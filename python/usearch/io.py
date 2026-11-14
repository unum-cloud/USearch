"""The `fbin`, `ibin` and sibling matrix formats: an int32 row count, an int32 column count, then the rows.

A `.u128bin` holds 128-bit little-endian integers, which NumPy has no scalar for, so its elements read
and write as opaque 16-byte `V16` values; a `[rows, 16]` array of `uint8` bytes saves as one column.

File: python/usearch/io.py
Author: Ash Vardanian
Date: May 4, 2023
"""

from __future__ import annotations

import struct

import numpy as np
import numpy.typing as npt

_EXTENSION_DTYPES: dict[str, npt.DTypeLike] = {
    ".f64bin": np.float64,
    ".dbin": np.float64,
    ".fbin": np.float32,
    ".f32bin": np.float32,
    ".bf16bin": np.uint16,
    ".f16bin": np.float16,
    ".hbin": np.float16,
    ".i8bin": np.int8,
    ".u8bin": np.uint8,
    ".bbin": np.uint8,
    ".u1x8bin": np.uint8,
    ".b1bin": np.uint8,
    ".ibin": np.int32,
    ".i32bin": np.int32,
    ".u64bin": np.uint64,
    ".u128bin": np.dtype("V16"),
}
"""The scalar type each extension names, since the header carries only the shape; ``bf16`` rows read as their ``uint16`` words."""

_BIT_EXTENSIONS: tuple[str, ...] = (".u1x8bin", ".b1bin")
"""The extensions whose header counts bits per row, while their rows read as whole bytes."""


def _dtype_for(filename: str) -> npt.DTypeLike | None:
    """The scalar type the extension of *filename* names, or None when it names none."""
    return next((dtype for suffix, dtype in _EXTENSION_DTYPES.items() if filename.endswith(suffix)), None)


def load_matrix(
    filename: str,
    start_row: int = 0,
    count_rows: int | None = None,
    *,
    view: bool = False,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.generic]:
    """Reads a matrix from an ``.f64bin``, ``.dbin``, ``.fbin``, ``.f32bin``, ``.bf16bin``, ``.f16bin``, ``.hbin``, ``.i8bin``, ``.u8bin``, ``.bbin``, ``.u1x8bin``, ``.b1bin``, ``.ibin``, ``.i32bin``, ``.u64bin`` or ``.u128bin`` file, in the scalar type its extension or *dtype* names, hence the generic dtype; ``.u128bin`` elements come back as ``V16``, and ``.view(np.uint8)`` exposes their bytes.

    Reads *count_rows* rows starting at *start_row*, or every row when *count_rows* is None. Setting
    *view* memory-maps the file rather than loading it into RAM, which is how a dataset larger than
    memory is read. A ``.u1x8bin`` or ``.b1bin`` row comes back as its packed bytes. Raises
    `FileNotFoundError` for a missing file and `ValueError` for an extension naming no scalar type,
    a file whose header is truncated or whose size it contradicts, or a *start_row* or *count_rows*
    reaching outside the file's rows.
    """
    if dtype is None:
        dtype = _dtype_for(filename)
        if dtype is None:
            raise ValueError(f"the extension of {filename} names no scalar type; pass `dtype=`")
    scalar = np.dtype(dtype)

    with open(filename, "rb") as file:
        header = np.fromfile(file, count=2, dtype=np.int32)
        if header.size != 2:
            raise ValueError(f"{filename} holds {header.size * 4} bytes, too few for its 8-byte header")
        rows, columns = (int(extent) for extent in header)
        if rows < 0 or columns < 0:
            raise ValueError(f"{filename} declares a negative shape of {rows} x {columns}")
        if filename.endswith(_BIT_EXTENSIONS):
            columns = (columns + 7) // 8
        file.seek(0, 2)
        expected = 8 + rows * columns * scalar.itemsize
        if file.tell() != expected:
            raise ValueError(f"{filename} holds {file.tell():,} bytes, but its header declares {expected:,}")

        if not 0 <= start_row <= rows:
            raise ValueError(f"start_row={start_row} is outside the {rows} rows of {filename}")
        count = rows - start_row if count_rows is None else count_rows
        if not 0 <= count <= rows - start_row:
            raise ValueError(f"count_rows={count} is outside the {rows - start_row} rows left in {filename}")
        offset = 8 + start_row * columns * scalar.itemsize
        if view:
            return np.memmap(file, dtype=scalar, mode="r", offset=offset, shape=(count, columns))
        file.seek(offset)
        return np.fromfile(file, count=count * columns, dtype=scalar).reshape(count, columns)


def save_matrix(vectors: npt.NDArray[np.generic], filename: str) -> None:
    """Writes a matrix to any file `load_matrix` reads.

    An extension naming a scalar type converts the rows into it; any other keeps the matrix's own,
    so *vectors* may hold any scalar type. A ``.u128bin`` takes ``V16`` elements, or a ``[rows, 16]``
    ``uint8`` array of each row's little-endian bytes, such as ``uuid.int.to_bytes(16, "little")``.
    A ``.u1x8bin`` or ``.b1bin`` takes ``[rows, row_bytes]`` packed bytes, its header counting bits.
    """
    if vectors.ndim != 2:
        raise ValueError(f"expected a matrix, got {vectors.ndim} dimensions")
    dtype = np.dtype(_dtype_for(filename) or vectors.dtype)
    if dtype == np.dtype("V16") and vectors.dtype == np.uint8:
        if vectors.shape[1] != 16:
            raise ValueError(f"expected 16 bytes per 128-bit row, got {vectors.shape[1]}")
        vectors = np.ascontiguousarray(vectors).view(dtype)
    rows, columns = vectors.shape
    with open(filename, "wb") as file:
        file.write(struct.pack("<ii", rows, columns * 8 if filename.endswith(_BIT_EXTENSIONS) else columns))
        vectors.astype(dtype).tofile(file)
