"""Smaller & Faster Single-File Vector Search Engine from Unum."""

import ctypes
import os
import sys

#! Load NumKong before the USearch extension module
#! We can't just use the `import numkong` as on Linux and Windows (unlike MacOS),
#! the symbols are not automatically loaded into the global namespace.
#! NumKong v7.5+ ships as a regular package (`numkong/__init__.py` plus a
#! `numkong/_numkong.<ext-suffix>` extension). We require v7.5+ in
#! `pyproject.toml`, so we load the compiled submodule directly.
try:
    from numkong import _numkong as _numkong_ext

    _numkong_path = _numkong_ext.__file__
    if _numkong_path is None:
        raise ImportError("Could not locate NumKong compiled extension")

    if sys.platform == "win32":
        # On Windows, register the extension's directory so the OS loader finds
        # any bundled sibling DLLs, then load without `RTLD_GLOBAL`.
        os.add_dll_directory(os.path.dirname(_numkong_path))
        numkong_lib = ctypes.CDLL(_numkong_path)
    else:
        # On Linux/macOS we need `RTLD_GLOBAL` so USearch's extension module can
        # resolve NumKong symbols at its own load time.
        numkong_lib = ctypes.CDLL(_numkong_path, mode=ctypes.RTLD_GLOBAL)

except ImportError:
    pass  # If the user doesn't want NumKong, we assume they know what they're doing


from usearch import lib as _extension
from usearch.lib import *  # noqa: F401,F403

__version__ = _extension.__version__

del _extension
