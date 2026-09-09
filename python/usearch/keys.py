"""Bridging integer identities to the 128-bit keys every batch verb takes.

No NumPy dtype expresses a 128-bit integer, so batch identities travel as big-endian sixteen-byte
rows in both directions. These two functions are the whole bridge; a scalar key needs neither,
since `usearch.Key` reads an `int`, a `str`, a `uuid.UUID`, or sixteen raw bytes directly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["integers_to_keys", "keys_to_integers"]


def integers_to_keys(values) -> np.ndarray:
    """Packs integer identities into the ``(N, 16)`` big-endian rows every batch verb takes.

    Takes anything NumPy reads as unsigned integers, one identity per row, and returns a
    ``uint8`` matrix ready for ``add``, ``search``, and their siblings.
    """
    packed = np.ascontiguousarray(values, dtype=">u8").reshape(-1, 1)
    rows = np.zeros((packed.shape[0], 16), dtype=np.uint8)
    rows[:, 8:] = packed.view(np.uint8)
    return rows


def keys_to_integers(keys) -> np.ndarray:
    """Unpacks the low eight bytes of each identity, the half integer keys park in.

    Takes an ``(..., 16)`` buffer of identity bytes, such as ``Matches.keys``, and returns a
    ``uint64`` array shaped like it without the trailing axis. An identity using its high eight
    bytes - a ``uuid4``, say - has no 64-bit integer form, so a run holding one is refused
    rather than silently halved.
    """
    raw = np.asarray(keys)
    if raw[..., :8].any():
        raise ValueError("An identity using its high eight bytes has no 64-bit integer form")
    low = np.ascontiguousarray(raw[..., 8:])
    return low.view(">u8").reshape(raw.shape[:-1])
