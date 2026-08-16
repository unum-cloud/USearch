"""Makes the bare test module names a package.

The C++ units here are named after their family - `dense.cpp`, `core.cpp` - and the Python
files mirror them one for one. Without this file `types.py` would shadow the standard library
module of that name for every later import in the process.
"""
