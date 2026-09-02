import os
import platform
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

compile_args = []
cxx_compile_args = []
link_args = []
macros_args = []


# Check the environment variables
is_linux: bool = sys.platform == "linux"
is_macos: bool = sys.platform == "darwin"
is_windows: bool = sys.platform == "win32"
machine: str = platform.machine().lower()


# Sibling libraries come from the submodules, and every wheel compiles all three in - the switches the
# CMake build exposes have no equivalent here, so the macros are constants.
macros_args.append(("USEARCH_USE_NUMKONG", "1"))
macros_args.append(("USEARCH_USE_FORKUNION", "1"))
macros_args.append(("USEARCH_USE_STRINGZILLA", "1"))


#! NumKong uses dynamic dispatch, and will not build the library as part of `usearch` package.
#! It relies on the fact that NumKong ships its own bindings for most platforms, and the user should
#! install it separately!
macros_args.extend(
    [
        ("NK_RUNTIME_DISPATCH", "1"),
        ("NK_TARGET_NEON", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_NEONBFDOT", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_NEONHALF", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_NEONSDOT", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SVE", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SVEBFDOT", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SVEHALF", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SVESDOT", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SVE2", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_HASWELL", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SKYLAKE", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_ICELAKE", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SAPPHIRE", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_GENOA", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_NEONFHM", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SVE2P1", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_TURIN", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SIERRA", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_ALDER", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SAPPHIREAMX", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_GRANITEAMX", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SME", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SME2", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SME2P1", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SMEF64", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SMEFA64", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SMEHALF", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SMEBF16", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_SMELUT2", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_RVV", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_RVVHALF", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_RVVBF16", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_RVVBB", "0"),  # ? Hide-out all complex intrinsics
        ("NK_TARGET_V128RELAXED", "0"),  # ? Hide-out all complex intrinsics
    ]
)

# The CPython layer is C and the instantiation grid is C++, so the two carry separate flag
# sets and `MixedLanguageBuildExt` below compiles them in separate passes.
if is_linux:
    compile_args.append("-std=c11")
    compile_args.append("-O3")  # Maximize performance
    compile_args.append("-Wno-unknown-pragmas")
    compile_args.append("-fdiagnostics-color=always")
    compile_args.append("-g1")  # Simplify debugging; the normal `-g` makes builds much longer

    cxx_compile_args.append("-std=c++20")
    cxx_compile_args.append("-O3")
    cxx_compile_args.append("-ffast-math")  # Maximize floating-point performance
    cxx_compile_args.append("-Wno-unknown-pragmas")
    cxx_compile_args.append("-fdiagnostics-color=always")
    cxx_compile_args.append("-g1")

    # Linking to NumKong
    link_args.append("-Wl,--unresolved-symbols=ignore-in-shared-libs")
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

    cxx_compile_args.append("-mmacosx-version-min=10.15")
    cxx_compile_args.append("-std=c++20")
    cxx_compile_args.append("-O3")
    cxx_compile_args.append("-ffast-math")
    cxx_compile_args.append("-fcolor-diagnostics")
    cxx_compile_args.append("-Wno-unknown-pragmas")
    cxx_compile_args.append("-g1")

    # NumKong symbols are resolved at runtime via ctypes.CDLL in __init__.py
    link_args.append("-undefined")
    link_args.append("dynamic_lookup")

if is_windows:
    compile_args.append("/O2")
    compile_args.append("/W1")  # Reduce warnings verbosity

    cxx_compile_args.append("/std:c++20")
    cxx_compile_args.append("/O2")
    cxx_compile_args.append("/fp:fast")  # Enable fast math for MSVC
    cxx_compile_args.append("/W1")
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

# The type-erasure boundary plus the per-dtype shards, in the order `src/CMakeLists.txt` lists
# them - add a shard here when one joins the library. Mirrors `build.rs`.
library_sources = [
    "src/types.cpp",
    "src/dense_index.cpp",
    "src/dense/ram_index_bf16.cpp",
    "src/dense/ram_index_f16.cpp",
    "src/dense/ram_index_e5m2.cpp",
    "src/dense/ram_index_e4m3.cpp",
    "src/dense/ram_index_e3m2.cpp",
    "src/dense/ram_index_e2m3.cpp",
    "src/dense/ram_index_i8.cpp",
    "src/dense/ram_index_i4.cpp",
    "src/dense/ram_index_u8.cpp",
    "src/dense/ram_index_u4.cpp",
    "src/dense/ram_index_u1.cpp",
    "src/sequence.cpp",
]

macros_args.append(("USEARCH_BUILD_CUDA", "0"))

ext_modules = [
    Extension(
        # Lives under the `usearch` package so `usearch/__init__.py` preloads NumKong first.
        "usearch.compiled",
        binding_sources + library_sources,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros_args,
        language="c++",  # Only the linker; the `.c` files still compile as C.
    ),
]


class MixedLanguageBuildExt(build_ext):
    """Compiles the C++ instantiation grid apart from the C bindings, then links the two.

    One flag set cannot serve both: `-std=c++20` is a hard error on a `.c` translation unit,
    and the shards need it. Compiling the shards here also caps the job count, since in a
    container `os.cpu_count` reports the host and a dozen parallel shards will OOM a wheel
    builder.
    """

    def initialize_options(self):
        super().initialize_options()
        self.parallel = int(os.environ.get("USEARCH_BUILD_PARALLEL", min(os.cpu_count() or 1, 4)))

    def build_extension(self, extension):
        cxx_sources = [source for source in extension.sources if source.endswith((".cpp", ".cc", ".cu"))]
        if cxx_sources:
            extension.sources = [source for source in extension.sources if source not in cxx_sources]
            objects = self.compiler.compile(
                cxx_sources,
                output_dir=self.build_temp,
                macros=list(extension.define_macros),
                include_dirs=list(extension.include_dirs) + list(self.include_dirs or []),
                debug=self.debug,
                extra_postargs=cxx_compile_args,
                depends=extension.depends,
            )
            extension.extra_objects = list(extension.extra_objects) + objects
        super().build_extension(extension)


__version__ = Path("VERSION").read_text().strip()
__lib_name__ = "usearch"

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Sibling headers come from the submodules, so a wheel can only be built from a recursive clone.
include_dirs = [
    "include",
    "python",
    "src",
    "numkong/include",
    "forkunion/include",
    "stringzilla/include",
]

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
    package_data={"usearch": ["compiled.pyi", "py.typed"]},
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
    include_dirs=include_dirs,
    ext_modules=ext_modules,
    cmdclass={"build_ext": MixedLanguageBuildExt},
    extras_require={"eval": ["numpy", "tqdm"]},
    install_requires=install_requires,
)
