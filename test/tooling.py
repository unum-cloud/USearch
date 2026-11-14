"""Dataset serialization - the `fbin`, `ibin`, and `ivecs` formats `usearch.io` reads.

Run:
    python -m pytest test/tooling.py -v

File: test/tooling.py
Author: Ash Vardanian
Date: August 16, 2026
"""

import os
import struct
import uuid

import numpy as np
import pytest
from base import BATCH_SIZES, SETTINGS

from usearch.io import load_matrix, save_matrix


@pytest.mark.parametrize("rows", BATCH_SIZES)
@pytest.mark.parametrize("columns", SETTINGS.dims)
@pytest.mark.parametrize(("extension", "dtype"), [("fbin", np.float32), ("ibin", np.int32)])
def test_matrix_roundtrip(tmp_path, rows: int, columns: int, extension: str, dtype):
    """A matrix written in either format reads back identical."""
    generator = np.random.default_rng(0)
    original = (
        generator.random((rows, columns)).astype(dtype)
        if dtype is np.float32
        else generator.integers(0, rows + 1, size=(rows, columns)).astype(dtype)
    )
    path = str(tmp_path / f"matrix.{extension}")
    save_matrix(original, path)
    assert np.array_equal(original, load_matrix(path))
    os.remove(path)


@pytest.mark.parametrize("rows", BATCH_SIZES)
def test_key_columns_roundtrip(tmp_path, rows: int):
    """Keys save as one column of 64-bit or 128-bit little-endian integers, the files `--keys` reads."""
    identities = np.random.default_rng(0).integers(0, 2**63, size=(rows, 1), dtype=np.uint64)
    path = str(tmp_path / "keys.u64bin")
    save_matrix(identities, path)
    assert np.array_equal(identities, load_matrix(path))

    uuids = [uuid.UUID(int=int(identity) * 0x9E3779B97F4A7C15 + row) for row, identity in enumerate(identities[:, 0])]
    little_endian = np.frombuffer(b"".join(key.int.to_bytes(16, "little") for key in uuids), dtype=np.uint8)
    path = str(tmp_path / "keys.u128bin")
    save_matrix(little_endian.reshape(rows, 16), path)
    loaded = load_matrix(path)
    assert loaded.shape == (rows, 1)
    assert [uuid.UUID(int=int.from_bytes(row.tobytes(), "little")) for row in loaded[:, 0]] == uuids


def test_packed_bits_header_counts_bits(tmp_path):
    """A `.u1x8bin` header counts bits per row, as the tools read it, and its rows read back as bytes."""
    packed = np.array([[0xF0, 0x0F], [0xFF, 0x00]], dtype=np.uint8)
    path = str(tmp_path / "bits.u1x8bin")
    save_matrix(packed, path)
    with open(path, "rb") as file:
        assert struct.unpack("<ii", file.read(8)) == (2, 16)
    assert np.array_equal(packed, load_matrix(path))


def test_bf16_extension_reads_its_words(tmp_path):
    """A `.bf16bin` keeps its 16-bit words, which no NumPy dtype would decode."""
    words = np.array([[0x3F80, 0xBF80]], dtype=np.uint16)
    path = str(tmp_path / "words.bf16bin")
    save_matrix(words, path)
    assert np.array_equal(words, load_matrix(path))


def test_missing_matrix_raises(tmp_path):
    """A missing file is a `FileNotFoundError`, never a silent None."""
    with pytest.raises(FileNotFoundError):
        load_matrix(str(tmp_path / "absent.fbin"))


@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize(
    ("start_row", "count_rows"),
    [(-1, None), (5, None), (2, 5), (0, -1), (1, 4)],
)
def test_matrix_row_range_out_of_bounds_raises(tmp_path, view: bool, start_row: int, count_rows):
    """A row range reaching outside the file is a `ValueError`, not an `OSError` or a bad reshape."""
    path = str(tmp_path / "matrix.fbin")
    save_matrix(np.arange(12, dtype=np.float32).reshape(4, 3), path)
    with pytest.raises(ValueError, match="rows"):
        load_matrix(path, start_row=start_row, count_rows=count_rows, view=view)


def test_matrix_row_range_edges_are_valid(tmp_path):
    """The empty tail and the full file are both in range."""
    original = np.arange(12, dtype=np.float32).reshape(4, 3)
    path = str(tmp_path / "matrix.fbin")
    save_matrix(original, path)
    assert load_matrix(path, start_row=4).shape == (0, 3)
    assert np.array_equal(load_matrix(path, start_row=1, count_rows=3), original[1:])


def test_matrix_truncated_header_raises(tmp_path):
    """A file shorter than the 8-byte header is a `ValueError` naming the file."""
    path = tmp_path / "short.fbin"
    path.write_bytes(bytes([1, 0]))
    with pytest.raises(ValueError, match="header"):
        load_matrix(str(path))
