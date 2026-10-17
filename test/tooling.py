"""Dataset serialization - the `fbin`, `ibin`, and `ivecs` formats `usearch.io` reads.

Run:
    python -m pytest test/tooling.py -v
"""

from __future__ import annotations

import os
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


def test_missing_matrix_raises(tmp_path):
    """A missing file is a `FileNotFoundError`, never a silent None."""
    with pytest.raises(FileNotFoundError):
        load_matrix(str(tmp_path / "absent.fbin"))
