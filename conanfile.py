from conans import ConanFile


required_conan_version = ">=1.53.0"


class USearchConan(ConanFile):

    name = "usearch"
    version = "2.24.0"
    license = "Apache-2.0"
    description = "Smaller & Faster Single-File Vector Search Engine from Unum"
    homepage = "https://github.com/unum-cloud/usearch"
    topics = ("search", "vector", "simd")
    settings = "os", "arch", "compiler", "build_type"
    url = "https://github.com/conan-io/conan-center-index"
    package_type = "header-library"

    # No settings/options are necessary, this is header only
    # Potentially add unit-tests in the future:
    # https://docs.conan.io/1/howtos/header_only.html#with-unit-tests
    exports_sources = "include/*"
    no_copy_source = True

    def package(self):
        self.copy("*.h")
