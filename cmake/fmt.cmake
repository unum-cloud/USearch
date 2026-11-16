# The pin for what only the suites, benchmarks and tools link: {fmt} for their output, as libstdc++ doesn't ship
# `std::print` yet on our toolchains.
# `FIND_PACKAGE_ARGS` lets a system-provided {fmt} satisfy the dependency without touching the network, but only from
# the pinned version up, since older copies predate `fmt::println` and would fail deep in the suites rather than here.
# {fmt} turns itself into a C++20 module under Ninja with recent Clang or MSVC, which no unit here imports.
include(FetchContent)
set(FMT_MODULE OFF)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 12.2.0
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL SYSTEM FIND_PACKAGE_ARGS 12.2.0
)
FetchContent_MakeAvailable(fmt)
