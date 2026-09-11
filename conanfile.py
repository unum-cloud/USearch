"""Conan 2 recipe. The engine is CMake's to build, so this drives it rather than restating the
instantiation grid and the per-target flags a second time."""

import os

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy, load

required_conan_version = ">=2.0"


class USearchConan(ConanFile):
    name = "usearch"
    license = "Apache-2.0"
    description = "Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices"
    homepage = "https://github.com/unum-cloud/USearch"
    url = "https://github.com/unum-cloud/USearch"
    topics = (
        "vector-search",
        "vector-database",
        "similarity-search",
        "nearest-neighbors",
        "embeddings",
        "semantic-search",
        "retrieval",
        "clustering",
        "simd",
        "gpu",
    )

    # Compiled rather than header-only: the C interface links an instantiation grid that a consumer
    # would otherwise pay to compile itself.
    package_type = "library"
    settings = "os", "arch", "compiler", "build_type"
    options = {"shared": [True, False], "fPIC": [True, False], "with_cuda": [True, False]}
    default_options = {"shared": False, "fPIC": True, "with_cuda": False}
    implements = ["auto_shared_fpic"]

    # `src/` and `c/` hold the instantiations, `cmake/` the toolchains and the package templates,
    # and `VERSION` is what every other SDK reads - so the recipe reads it too rather than carrying
    # a second copy that drifts.
    exports_sources = "CMakeLists.txt", "VERSION", "LICENSE", "include/*", "src/*", "c/*", "cmake/*"

    def set_version(self):
        self.version = load(self, os.path.join(self.recipe_folder, "VERSION")).strip()

    def validate(self):
        check_min_cppstd(self, 20)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        toolchain = CMakeToolchain(self)
        # Off by default in the tree, so the headers and the CMake package files would be absent.
        toolchain.cache_variables["USEARCH_INSTALL"] = True
        toolchain.cache_variables["USEARCH_BUILD_LIB_C"] = True
        toolchain.cache_variables["USEARCH_BUILD_CUDA"] = self.options.with_cuda
        toolchain.cache_variables["USEARCH_BUILD_TEST_CPP"] = False
        toolchain.cache_variables["USEARCH_BUILD_BENCH_CPP"] = False
        toolchain.cache_variables["USEARCH_BUILD_TEST_C"] = False
        toolchain.generate()
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "LICENSE", self.source_folder, os.path.join(self.package_folder, "licenses"))
        CMake(self).install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "usearch")

        # Three components, because the tree exports three targets and a consumer picks between
        # them: the headers alone, the headers plus the siblings they reach for, or the compiled
        # C interface that spares a consumer the instantiation grid.
        self.cpp_info.components["core"].set_property("cmake_target_name", "usearch::core")
        self.cpp_info.components["core"].bindirs = []
        self.cpp_info.components["core"].libdirs = []

        self.cpp_info.components["header"].set_property("cmake_target_name", "usearch::header")
        self.cpp_info.components["header"].requires = ["core"]
        self.cpp_info.components["header"].bindirs = []
        self.cpp_info.components["header"].libdirs = []

        # Named for what the tree builds rather than for the package, so a consumer picks the archive
        # or the shared object explicitly instead of inheriting whichever this recipe defaulted to.
        linkage = "shared" if self.options.shared else "static"
        self.cpp_info.components[linkage].set_property("cmake_target_name", f"usearch::{linkage}")
        self.cpp_info.components[linkage].set_property("pkg_config_name", "usearch")
        self.cpp_info.components[linkage].libs = [f"usearch_{linkage}"]
        self.cpp_info.components[linkage].requires = ["header"]
        if self.settings.os in ("Linux", "FreeBSD"):
            self.cpp_info.components[linkage].system_libs = ["m", "pthread", "dl"]
