import os
import platform
import sys
from pathlib import Path

from setuptools import Extension, setup

compile_args = []
link_args = []
macros_args = []


# Check the environment variables
is_linux: bool = sys.platform == "linux"
is_macos: bool = sys.platform == "darwin"
is_windows: bool = sys.platform == "win32"
machine: str = platform.machine().lower()


if is_linux:
    compile_args.append("-std=c11")
    compile_args.append("-O3")  # Maximize performance
    compile_args.append("-Wno-unknown-pragmas")
    compile_args.append("-fdiagnostics-color=always")
    compile_args.append("-g1")  # Simplify debugging; the normal `-g` makes builds much longer

    # NumKong symbols are resolved at runtime via ctypes.CDLL in __init__.py, and the CPython
    # ones by the interpreter; narrowing this to `ignore-in-shared-libs` would demand both here.
    link_args.append("-Wl,--unresolved-symbols=ignore-all")
    link_args.append("-static-libstdc++")

if is_macos:
    # MacOS 10.15 or higher is needed for `aligned_alloc` support.
    # https://github.com/unum-cloud/USearch/actions/runs/4975434891/jobs/8902603392
    compile_args.append("-mmacosx-version-min=10.15")
    compile_args.append("-std=c11")
    compile_args.append("-O3")
    compile_args.append("-fcolor-diagnostics")
    compile_args.append("-Wno-unknown-pragmas")
    compile_args.append("-g1")

    # NumKong symbols are resolved at runtime via ctypes.CDLL in __init__.py
    link_args.append("-undefined")
    link_args.append("dynamic_lookup")

if is_windows:
    compile_args.append("/O2")
    compile_args.append("/W1")  # Reduce warnings verbosity

    link_args.append("/FORCE")  # Force linking with missing NumKong symbols


# The CPython layer: module glue first, then the shared ingestion, then one file per modality
# named after the header it mirrors - the same stems `rust/` and `include/usearch/` list.
binding_sources = [
    "python/lib.c",
    "python/types.c",
    "python/dense_index.c",
    "python/sequence_index.c",
    "python/matrix_index.c",
    "python/geospatial_index.c",
]


# Only the public headers and the binding's own: the engine arrives as a library, not as sources.
include_dirs = ["include", "python"]

# Where CMake dropped `libusearch_c`; the engine leaves it under `<build>/c` and the brute-force
# stand-in under `<build>/stub`, so choosing between the two is this one path. Both answer the same
# verbs, so a suite that only exercises the ABI runs against the stand-in in seconds.
library_dir = os.path.abspath(os.environ.get("USEARCH_LIBRARY_DIR", "build_release/c"))

# An editable install imports the extension from the source tree, where no loader path points at
# the build tree; the same directory rides along as an `rpath` so the library is found at import.
if is_linux:
    link_args.append(f"-Wl,-rpath,{library_dir}")
if is_macos:
    link_args.append(f"-Wl,-rpath,{library_dir}")


ext_modules = [
    Extension(
        # Lives under the `usearch` package so `usearch/__init__.py` preloads NumKong first.
        "usearch.lib",
        binding_sources,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros_args,
        include_dirs=include_dirs,
        libraries=["usearch_c"],
        library_dirs=[library_dir],
        language="c++",  # Only the linker; the `.c` files still compile as C.
    ),
]


__version__ = Path("VERSION").read_text().strip()
__lib_name__ = "usearch"

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# NumKong is the one sibling with a runtime half - `python/usearch/__init__.py` resolves its shared
# library through `ctypes` - so it is the only one that has to be installed alongside the wheel.
# Batch inputs arrive through the buffer protocol, so NumPy is a convenience of `usearch.eval`
# and of the test suite, never of the index itself.
install_requires = [
    "numkong",
    "stringzilla",
]


setup(
    name=__lib_name__,
    version=__version__,
    packages=["usearch"],
    package_dir={"usearch": "python/usearch"},
    package_data={"usearch": ["lib.pyi", "types.pyi", "dense_index.pyi", "py.typed"]},
    description="Smaller & Faster Single-File Vector Search Engine from Unum",
    author="Ash Vardanian",
    author_email="info@unum.cloud",
    python_requires=">=3.12",
    url="https://github.com/unum-cloud/USearch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    license_files=["LICENSE"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Java",
        "Programming Language :: JavaScript",
        "Programming Language :: Objective C",
        "Programming Language :: Rust",
        "Programming Language :: Other",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Topic :: System :: Clustering",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
    extras_require={"eval": ["numpy", "tqdm"]},
    install_requires=install_requires,
)
