import os
import sys
import subprocess
import platform
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension

compile_args = []
link_args = []
macros_args = []


def get_bool_env(name: str, preference: bool) -> bool:
    return os.environ.get(name, "1" if preference else "0") == "1"


def get_bool_env_w_name(name: str, preference: bool) -> tuple:
    return name, "1" if get_bool_env(name, preference) else "0"


# Check the environment variables
is_linux: bool = sys.platform == "linux"
is_macos: bool = sys.platform == "darwin"
is_windows: bool = sys.platform == "win32"
machine: str = platform.machine().lower()


is_gcc = False
is_clang = False
if is_linux:
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


# ? Is there a way we can bring back SimSIMD on Windows?
# ? Using `ctypes.CDLL(simsimd.__file__)` breaks the CI
# ? with "Windows fatal exception: access violation".
prefer_simsimd: bool = not is_windows
prefer_fp16lib: bool = True
prefer_openmp: bool = is_linux and is_gcc

use_simsimd: bool = get_bool_env("USEARCH_USE_SIMSIMD", prefer_simsimd)
use_fp16lib: bool = get_bool_env("USEARCH_USE_FP16LIB", prefer_fp16lib)
use_openmp: bool = get_bool_env("USEARCH_USE_OPENMP", prefer_openmp)

# Common arguments for all platforms
macros_args.append(("USEARCH_USE_OPENMP", "1" if use_openmp else "0"))
macros_args.append(("USEARCH_USE_FP16LIB", "1" if use_fp16lib else "0"))
macros_args.append(("USEARCH_USE_SIMSIMD", "1" if use_simsimd else "0"))


#! Unlike OpenMP and FP16LIB, the SimSIMD is integrated differently.
#! It will anyways use dynamic dispatch, and will not build the library as part of `usearch` package.
#! It relies on the fact that SimSIMD ships it's own bindings for most platforms, and the user should
#! install it separately!
macros_args.extend(
    [
        ("SIMSIMD_DYNAMIC_DISPATCH", "1" if use_simsimd else "0"),
        ("SIMSIMD_TARGET_NEON", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_NEON_BF16", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_NEON_F16", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_NEON_I8", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_SVE", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_SVE_BF16", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_SVE_F16", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_SVE_I8", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_SVE2", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_HASWELL", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_SKYLAKE", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_ICE", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_SAPPHIRE", "0"),  # ? Hide-out all complex intrinsics
        ("SIMSIMD_TARGET_GENOA", "0"),  # ? Hide-out all complex intrinsics
    ]
)

if is_linux:
    compile_args.append("-std=c++17")
    compile_args.append("-O3")  # Maximize performance
    compile_args.append("-ffast-math")  # Maximize floating-point performance
    compile_args.append("-Wpass-failed=transform-warning")
    compile_args.append("-Wno-unknown-pragmas")
    compile_args.append("-fdiagnostics-color=always")

    # Simplify debugging, but the normal `-g` may make builds much longer!
    compile_args.append("-g1")

    # Linking to SimSIMD
    compile_args.append("-Wl,--unresolved-symbols=ignore-in-shared-libs")
    link_args.append("-static-libstdc++")

    if use_openmp:
        compile_args.append("-fopenmp")
        link_args.append("-lgomp")

if is_macos:
    # MacOS 10.15 or higher is needed for `aligned_alloc` support.
    # https://github.com/unum-cloud/usearch/actions/runs/4975434891/jobs/8902603392
    compile_args.append("-mmacosx-version-min=10.15")
    compile_args.append("-std=c++17")
    compile_args.append("-O3")  # Maximize performance
    compile_args.append("-ffast-math")  # Maximize floating-point performance
    compile_args.append("-fcolor-diagnostics")
    compile_args.append("-Wpass-failed=transform-warning")
    compile_args.append("-Wno-unknown-pragmas")

    # Simplify debugging, but the normal `-g` may make builds much longer!
    compile_args.append("-g1")

    # Linking OpenMP requires additional preparation in CIBuildWheel.
    # We must install `brew install llvm` ahead of time.
    # import subprocess as cli
    # llvm_base = cli.check_output(["brew", "--prefix", "llvm"]).strip().decode("utf-8")
    # if len(llvm_base):
    #     compile_args.append(f"-I{llvm_base}/include")
    #     compile_args.append("-Xpreprocessor -fopenmp")
    #     link_args.append(f"-L{llvm_base}/lib")
    #     link_args.append("-lomp")
    #     macros_args.append(("USEARCH_USE_OPENMP", "1"))

if is_windows:
    compile_args.append("/std:c++17")
    compile_args.append("/O2")
    compile_args.append("/fp:fast")  # Enable fast math for MSVC
    compile_args.append("/W1")  # Reduce warnings verbosity
    link_args.append("/FORCE")  # Force linkin with missing SimSIMD symbols


ext_modules = [
    Pybind11Extension(
        "usearch.compiled",
        ["python/lib.cpp"],
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
if use_simsimd:
    include_dirs.append("simsimd/include")
    install_requires.append("simsimd>=6.0.5,<7.0.0")
if use_fp16lib:
    include_dirs.append("fp16/include")


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
    description="Smaller & Faster Single-File Vector Search Engine from Unum",
    author="Ash Vardanian",
    author_email="info@unum.cloud",
    url="https://github.com/unum-cloud/usearch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
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
