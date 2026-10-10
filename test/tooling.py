"""Dataset serialization - the `fbin`, `ibin`, and `ivecs` formats `usearch.io` reads.

Run:
    python -m pytest test/tooling.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from base import SETTINGS, batch_sizes

from usearch.io import load_matrix, save_matrix


@pytest.mark.parametrize("rows", batch_sizes)
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


def test_missing_matrix_raises(tmp_path):
    """A missing file is a `FileNotFoundError`, never a silent None."""
    with pytest.raises(FileNotFoundError):
        load_matrix(str(tmp_path / "absent.fbin"))
