import os
import platform
import subprocess
import sys

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

compile_args = []
link_args = []
macros_args = []


def get_bool_env(name: str, preference: bool) -> bool:
    return os.environ.get(name, "1" if preference else "0") == "1"


def get_bool_env_w_name(name: str, preference: bool) -> tuple:
    return name, "1" if get_bool_env(name, preference) else "0"


# Check the environment variables
is_android: bool = sys.platform == "android"
is_linux: bool = sys.platform == "linux"
is_macos: bool = sys.platform == "darwin"
is_windows: bool = sys.platform == "win32"
machine: str = platform.machine().lower()


is_gcc = False
is_clang = False
if is_linux or is_android:
    cxx = os.environ.get("CXX")
    if cxx:
        try:
            command = "where" if os.name == "nt" else "which"
            full_path = subprocess.check_output([command, cxx], text=True).strip()
            compiler_name = os.path.basename(full_path)
            is_gcc = ("g++" in compiler_name) and ("clang++" not in compiler_name)
            is_clang = ("clang++" in compiler_name) and ("g++" not in compiler_name)
        except subprocess.CalledProcessError:
            pass


# ? Is there a way we can bring back NumKong on Windows?
# ? Using `ctypes.CDLL(numkong.__file__)` breaks the CI
# ? with "Windows fatal exception: access violation".
prefer_numkong: bool = not is_windows
prefer_openmp: bool = (is_linux or is_android) and is_gcc

use_numkong: bool = get_bool_env("USEARCH_USE_NUMKONG", prefer_numkong)
use_openmp: bool = get_bool_env("USEARCH_USE_OPENMP", prefer_openmp)

# Dynamic LOADING of a separate NumKong library often fails at runtime on Android
# due to the Bionic linker's behavior with RTLD_GLOBAL and Python extensions.
# In such cases, we bundle NumKong source files directly into the extension.
use_numkong_bundle: bool = use_numkong and is_android

# Common arguments for all platforms
macros_args.append(("USEARCH_USE_OPENMP", "1" if use_openmp else "0"))
macros_args.append(("USEARCH_USE_NUMKONG", "1" if use_numkong else "0"))


#! NumKong uses dynamic dispatch, and will not build the library as part of `usearch` package.
#! It relies on the fact that NumKong ships its own bindings for most platforms, and the user should
#! install it separately!
macros_args.extend(
    [
        ("NK_DYNAMIC_DISPATCH", "1" if use_numkong else "0"),
        ("NK_TARGET_NEON", "1" if is_android and ("arm" in machine or "aarch64" in machine) else "0"),
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

if is_linux or is_android:
    compile_args.append("-std=c++17")
    compile_args.append("-O3")  # Maximize performance
    compile_args.append("-ffast-math")  # Maximize floating-point performance
    compile_args.append("-Wno-unknown-pragmas")
    compile_args.append("-fdiagnostics-color=always")

    # Simplify debugging, but the normal `-g` may make builds much longer!
    compile_args.append("-g1")

    # Linking to NumKong
    compile_args.append("-Wl,--unresolved-symbols=ignore-in-shared-libs")

    # On Android we don't usually want to link libstdc++ statically
    if is_linux:
        link_args.append("-static-libstdc++")

    if use_openmp:
        compile_args.append("-fopenmp")
        link_args.append("-lgomp")

if is_macos:
    # MacOS 10.15 or higher is needed for `aligned_alloc` support.
    # https://github.com/unum-cloud/USearch/actions/runs/4975434891/jobs/8902603392
    compile_args.append("-mmacosx-version-min=10.15")
    compile_args.append("-std=c++17")
    compile_args.append("-O3")  # Maximize performance
    compile_args.append("-ffast-math")  # Maximize floating-point performance
    compile_args.append("-fcolor-diagnostics")
    compile_args.append("-Wno-unknown-pragmas")

    # Simplify debugging, but the normal `-g` may make builds much longer!
    compile_args.append("-g1")

    # NumKong symbols are resolved at runtime via ctypes.CDLL in __init__.py
    link_args.append("-undefined")
    link_args.append("dynamic_lookup")

if is_windows:
    compile_args.append("/std:c++17")
    compile_args.append("/O2")  # Maximize performance
    compile_args.append("/fp:fast")  # Maximize floating-point performance
    compile_args.append("/W1")  # Reduce warnings verbosity
    link_args.append("/FORCE")  # Force linking with missing NumKong symbols


sources = ["python/lib.cpp"]
if use_numkong_bundle:
    sources.append("python/numkong_bundle.cpp")

ext_modules = [
    Pybind11Extension(
        "usearch.compiled",
        sources,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros_args,
        language="c++",
    ),
]

__version__ = open("VERSION", "r").read().strip()
__lib_name__ = "usearch"

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Depending on the macros, adjust the include directories
include_dirs = [
    "include",
    "python",
    "stringzilla/include",
]
install_requires = [
    "numpy",
    "tqdm",
]
if use_numkong:
    include_dirs.append("numkong/include")
    if use_numkong_bundle:
        include_dirs.append("numkong/c")
    else:
        install_requires.append("numkong")


# With Clang, `setuptools` doesn't properly use the `language="c++"` argument we pass.
# The right thing would be to pass down `-x c++` to the compiler, before specifying the source files.
# This nasty workaround overrides the `CC` environment variable with the `CXX` variable.
cc_compiler_variable = os.environ.get("CC")
cxx_compiler_variable = os.environ.get("CXX")
if is_clang:
    if cxx_compiler_variable:
        os.environ["CC"] = cxx_compiler_variable

setup(
    name=__lib_name__,
    version=__version__,
    packages=["usearch"],
    package_dir={"usearch": "python/usearch"},
    package_data={"usearch": ["compiled.pyi", "py.typed"]},
    description="Smaller & Faster Single-File Vector Search Engine from Unum",
    author="Ash Vardanian",
    author_email="info@unum.cloud",
    python_requires=">=3.10",
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
    install_requires=install_requires,
)

# Reset the CC environment variable, that we overrode earlier.
if is_clang:
    if cxx_compiler_variable:
        os.environ["CC"] = cc_compiler_variable
