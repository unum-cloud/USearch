# The pin for what only the suites and tools link: {fmt} for the C++ smoke tests, benchmarks and command-line tools, as
# libstdc++ doesn't ship `std::print` yet on our toolchain. `FIND_PACKAGE_ARGS` lets a system-provided {fmt} satisfy the
# dependency without touching the network, but only from the pinned version up - the copies distributions ship predate
# `fmt::println` and would fail deep in the suites rather than here.
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 12.2.0
    FIND_PACKAGE_ARGS 12.2.0
)
FetchContent_MakeAvailable(fmt)
