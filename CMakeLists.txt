# This CMake file is heavily inspired by following `usearch` CMake:
# https://github.com/nlohmann/json/blob/develop/CMakeLists.txt
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(
    usearch
    VERSION 2.25.1
    LANGUAGES C CXX
    DESCRIPTION "Smaller & Faster Single-File Vector Search Engine from Unum"
    HOMEPAGE_URL "https://github.com/unum-cloud/USearch"
)

# Determine if USearch is built as a subproject (using `add_subdirectory`) or if it is the main project
set(USEARCH_IS_MAIN_PROJECT OFF)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(USEARCH_IS_MAIN_PROJECT ON)
endif ()

# Allow CMake 3.13+ to override options when using FetchContent / add_subdirectory
if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

# Probed only to default `USEARCH_BUILD_CUDA`: `check_language` caches the host compiler it paired with, which would
# then outrank a caller's own `CMAKE_CUDA_HOST_COMPILER`.
include(CheckLanguage)
set(USEARCH_CAN_BUILD_CUDA OFF)
if (NOT DEFINED USEARCH_BUILD_CUDA)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        set(USEARCH_CAN_BUILD_CUDA ON)
        message(STATUS "CUDA compiler available: ${CMAKE_CUDA_COMPILER}")
    else ()
        message(STATUS "CUDA compiler not available - the `.cu` units stay out of the build")
    endif ()
endif ()

# Options
option(USEARCH_INSTALL "Install CMake targets" OFF)

option(USEARCH_USE_JEMALLOC "Use JeMalloc for faster memory allocations" OFF)

option(USEARCH_BUILD_TEST_CPP "Compile a native unit test in C++" ${USEARCH_IS_MAIN_PROJECT})
option(USEARCH_BUILD_BENCH_CPP "Compile a native benchmark in C++" ${USEARCH_IS_MAIN_PROJECT})
option(USEARCH_BUILD_LIB_C "Compile a native library for the C 99 interface" OFF)
option(USEARCH_BUILD_TEST_C "Compile a test for the C 99 interface" OFF)
option(USEARCH_BUILD_LIB_C_STUB "Compile a brute-force stand-in for the C 99 interface" OFF)
option(USEARCH_BUILD_LIB_JAVA "Compile a native library for the Java bindings" OFF)
option(USEARCH_BUILD_LIB_WOLFRAM "Compile a native library for the Wolfram Language bindings" OFF)
option(USEARCH_BUILD_CUDA "Compile CUDA index tests & benchmarks" ${USEARCH_CAN_BUILD_CUDA})

# One list feeds every CUDA target, so a fat binary from this tree runs on the whole fleet. Ampere is the floor: its
# cubins stay loadable on every later 8.x part, so "80" alone covers 86 and 89. Hopper and both Blackwell lines are
# architecture-specific ("a" suffix) because the kernels reach for `wgmma` and the tensor-memory instructions, which
# carry no forward-compatible PTX - each needs its own entry. Override with a single architecture, or with "native",
# when iterating on one machine and the extra compile time is not worth paying.
set(USEARCH_CUDA_ARCHITECTURES
    "80;90a;100a;120a"
    CACHE STRING "GPU architectures baked into the CUDA targets: Ampere floor, Hopper, Blackwell datacenter & consumer"
)

# Three-digit "a" entries are only understood from CMake 3.31 onward, while the CPU-only build stays on the 3.28 floor
# that stock Ubuntu 24.04 ships. The floor follows the architectures actually asked for, so a single-architecture build
# on a distribution CMake is never refused for a capability it does not use.
if (USEARCH_BUILD_CUDA AND CMAKE_VERSION VERSION_LESS 3.31)
    string(REGEX MATCH "1[0-9][0-9]a" architectures_needing_cmake_331 "${USEARCH_CUDA_ARCHITECTURES}")
    if (architectures_needing_cmake_331)
        message(
            FATAL_ERROR
                "USEARCH_CUDA_ARCHITECTURES names \"${architectures_needing_cmake_331}\", which needs CMake 3.31 or newer, found ${CMAKE_VERSION} - override with a single architecture or \"native\""
        )
    endif ()
endif ()

# One diagnostic is what an iteration wants; surveying a tree that has drifted wants the whole list at once.
set(USEARCH_MAX_ERRORS
    "1"
    CACHE STRING "Diagnostics a compiler emits before giving up on a translation unit"
)

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any cloned repo
# execute code), so point `core.hooksPath` at `.githooks` here - configuring the build arms the `pre-commit` checks.
# Skipped when built as a subproject or without `.git` (tarball / export builds).
if (USEARCH_IS_MAIN_PROJECT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE _usearch_hooks_status
        OUTPUT_QUIET ERROR_QUIET
    )
    if (_usearch_hooks_status EQUAL 0)
        message(STATUS "usearch: Git hooks enabled (core.hooksPath -> .githooks)")
    endif ()
endif ()

# Includes
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include(ExternalProject)

# Configuration
include(GNUInstallDirs)
set(USEARCH_TARGET_NAME ${PROJECT_NAME})
set(USEARCH_CONFIG_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}"
    CACHE INTERNAL ""
)
set(USEARCH_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
set(USEARCH_TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(USEARCH_CMAKE_CONFIG_TEMPLATE "cmake/config.cmake.in")
set(USEARCH_CMAKE_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(USEARCH_CMAKE_VERSION_CONFIG_FILE "${USEARCH_CMAKE_CONFIG_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(USEARCH_CMAKE_PROJECT_CONFIG_FILE "${USEARCH_CMAKE_CONFIG_DIR}/${PROJECT_NAME}Config.cmake")
set(USEARCH_CMAKE_PROJECT_TARGETS_FILE "${USEARCH_CMAKE_CONFIG_DIR}/${PROJECT_NAME}Targets.cmake")
set(USEARCH_PKGCONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/pkgconfig")

include(CheckCSourceCompiles)
# Check for `__fp16` support
check_c_source_compiles(
    [=[
int
main(int argc, char *argv)
{
  __fp16 a = 1.0;
  __fp16 b = a + a;
  return 0;
}
]=]
    USEARCH_CAN_COMPILE_FP16
)

# Check for `_Float16` support
check_c_source_compiles(
    [=[
int
main(int argc, char *argv)
{
  _Float16 a = 1.0;
  _Float16 b = a + a;
  return 0;
}
]=]
    USEARCH_CAN_COMPILE_FLOAT16
)

# Check for `__bf16` support
check_c_source_compiles(
    [=[
int
main(int argc, char *argv)
{
  __bf16 a = 1.0;
  __bf16 b = a + a;
  return 0;
}
]=]
    USEARCH_CAN_COMPILE_BF16
)

# Check for `_Bfloat16` support
check_c_source_compiles(
    [=[
int
main(int argc, char *argv)
{
  _Bfloat16 a = 1.0;
  _Bfloat16 b = a + a;
  return 0;
}
]=]
    USEARCH_CAN_COMPILE_BFLOAT16
)

# Define our header-only library
add_library(${USEARCH_TARGET_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${USEARCH_TARGET_NAME} ALIAS ${USEARCH_TARGET_NAME})
set(USEARCH_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include/")

# Concepts are the main reason C++20 is the base language level, with `std::atomic_ref` and the `<bit>` helpers close
# behind - and the interface target must say so, or external consumers get an unbuildable contract.
target_compile_features(${USEARCH_TARGET_NAME} INTERFACE cxx_std_20)

# Define which types can be compiled
target_compile_definitions(
    ${USEARCH_TARGET_NAME} INTERFACE "USEARCH_CAN_COMPILE_FP16=$<BOOL:${USEARCH_CAN_COMPILE_FP16}>"
)
target_compile_definitions(
    ${USEARCH_TARGET_NAME} INTERFACE "USEARCH_CAN_COMPILE_FLOAT16=$<BOOL:${USEARCH_CAN_COMPILE_FLOAT16}>"
)
target_compile_definitions(
    ${USEARCH_TARGET_NAME} INTERFACE "USEARCH_CAN_COMPILE_BF16=$<BOOL:${USEARCH_CAN_COMPILE_BF16}>"
)
target_compile_definitions(
    ${USEARCH_TARGET_NAME} INTERFACE "USEARCH_CAN_COMPILE_BFLOAT16=$<BOOL:${USEARCH_CAN_COMPILE_BFLOAT16}>"
)

target_include_directories(
    ${USEARCH_TARGET_NAME} ${USEARCH_SYSTEM_INCLUDE} INTERFACE $<BUILD_INTERFACE:${USEARCH_INCLUDE_BUILD_DIR}>
                                                               $<INSTALL_INTERFACE:include>
)
# Install a pkg-config file, so other tools can find this
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/pkg-config.pc.in" "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc")

# A header-only package fits any architecture, and `ARCH_INDEPENDENT` declares exactly that - no hand-written version
# template needed.
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${USEARCH_CMAKE_VERSION_CONFIG_FILE}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion ARCH_INDEPENDENT
)
configure_file(${USEARCH_CMAKE_CONFIG_TEMPLATE} ${USEARCH_CMAKE_PROJECT_CONFIG_FILE} @ONLY)

if (USEARCH_INSTALL)
    install(DIRECTORY ${USEARCH_INCLUDE_BUILD_DIR} DESTINATION ${USEARCH_INCLUDE_INSTALL_DIR})
    install(FILES ${USEARCH_CMAKE_PROJECT_CONFIG_FILE} ${USEARCH_CMAKE_VERSION_CONFIG_FILE}
            DESTINATION ${USEARCH_CONFIG_INSTALL_DIR}
    )
    export(
        TARGETS ${USEARCH_TARGET_NAME}
        NAMESPACE ${PROJECT_NAME}::
        FILE ${USEARCH_CMAKE_PROJECT_TARGETS_FILE}
    )
    install(
        TARGETS ${USEARCH_TARGET_NAME}
        EXPORT ${USEARCH_TARGETS_EXPORT_NAME}
        INCLUDES
        DESTINATION ${USEARCH_INCLUDE_INSTALL_DIR}
    )
    install(
        EXPORT ${USEARCH_TARGETS_EXPORT_NAME}
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${USEARCH_CONFIG_INSTALL_DIR}
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" DESTINATION ${USEARCH_PKGCONFIG_INSTALL_DIR})
endif ()

# Find required packages
if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif ()

find_package(Threads REQUIRED)

if (USEARCH_USE_JEMALLOC)
    include(jemalloc)
endif ()

# Set default build type to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif ()

# Host tuning for the tests & benches only - never the distributable libraries - and only where the host is the target
# and the compiler has a spelling for it; elsewhere the suite runs on its serial floor and the SIMD cells report
# themselves skipped. Apple Clang's `-march=native` advertises only a subset of the host's features - no FP16 arithmetic
# & FML, no SME - leaving NumKong's f16 kits serial; `-mcpu=native` is the complete AArch64 knob, as NumKong's own build
# concludes.
if (MSVC OR CMAKE_CROSSCOMPILING)
    set(USEARCH_HOST_TUNING "")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(USEARCH_HOST_TUNING -mcpu=native)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(USEARCH_HOST_TUNING -march=native)
else ()
    set(USEARCH_HOST_TUNING "")
endif ()

# Shared wiring for every target this project builds. Optimization and debug level are deliberately absent: they live in
# `CMAKE_<LANG>_FLAGS_<CONFIG>`, where a preset or a command-line switch can move them. Only what the code actually
# depends on - fast math, the diagnostics, the sanitizers - is pinned here.
function (usearch_setup_target TARGET_NAME)
    # Compiler-specific options List of all possible compiler IDs:
    # https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(
            ${TARGET_NAME}
            PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=address
                    -fsanitize=leak
                    -fsanitize=alignment
                    -fsanitize=undefined
                    >
                    -ffast-math
                    -fPIC
                    -Wall
                    -Wextra
                    -Wno-conversion
                    -Wno-unknown-pragmas
                    -fmax-errors=${USEARCH_MAX_ERRORS}
                    -pedantic
                    -fdiagnostics-color=always
        )
        target_link_options(
            ${TARGET_NAME}
            PRIVATE
            $<$<CONFIG:DEBUG>:-g
            -fsanitize=address
            -fsanitize=leak
            -fsanitize=alignment
            -fsanitize=undefined
            >
            -fPIC
        )

        target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)

    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(
            ${TARGET_NAME}
            PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=address
                    -fsanitize=alignment
                    -fsanitize=undefined
                    >
                    -ffast-math
                    -fPIC
                    -pedantic
                    -Wfatal-errors
                    -fcolor-diagnostics
        )
        target_link_options(
            ${TARGET_NAME}
            PRIVATE
            $<$<CONFIG:DEBUG>:-g
            -fsanitize=address
            -fsanitize=alignment
            -fsanitize=undefined
            >
            -fPIC
        )

        # Check if the compiler is AppleClang, and if not, add the leak sanitizer
        if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
            # It's likely AppleClang Adjust options as needed for AppleClang
        else ()
            # It's likely LLVM Clang
            target_compile_options(${TARGET_NAME} PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=leak>)
            target_link_options(${TARGET_NAME} PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=leak>)
        endif ()

        if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
            target_link_libraries(
                ${TARGET_NAME} PRIVATE "-framework CoreFoundation" "-framework Security" Threads::Threads
            )
        else ()
            target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)
        endif ()

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "NVIDIA" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        target_compile_options(${TARGET_NAME} PRIVATE $<$<CONFIG:RELEASE>:-O3>)
        target_compile_definitions(${TARGET_NAME} PRIVATE --expt-relaxed-constexpr --extended-lambda)
        set_property(SOURCE bench.cpp PROPERTY LANGUAGE CUDA)
        set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${USEARCH_CUDA_ARCHITECTURES}")

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(${TARGET_NAME} PRIVATE -w -ferror-limit=${USEARCH_MAX_ERRORS})

    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # MSVC has no O3 optimization level
        target_compile_options(${TARGET_NAME} PRIVATE /wd4068 /wd4996)
        target_compile_options(
            ${TARGET_NAME} PRIVATE $<$<CONFIG:DEBUG>:/DEBUG> $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:RELWITHDEBINFO>:/O2
                                   /DEBUG>
        )
    endif ()

    # One source of truth for header paths: whatever the interface target advertises, including the dependency includes,
    # without inheriting its compile features onto pure-C targets.
    target_include_directories(
        ${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:${USEARCH_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
    )
    set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 20)

    # On Windows, when using multi-configuration CMake builds, the paths will be nested, resulting in
    # "build_debug\Debug" on Windows instead of "build_debug". That's why we override all the configs to use the same
    # output directory.
    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
                   ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
                   LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
                   ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
                   LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
    )
    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
                   ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
                   LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
    )
    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}"
                   ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}"
                   LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}"
    )

    # On Linux and MacOS our primary shared libraries for C are called `libusearch_c.so` and `libusearch_c.dylib`, while
    # on Windows the "lib" prefix might be missing by default, so let's ensure it's always there
    get_target_property(TARGET_TYPE ${TARGET_NAME} TYPE)

    if (TARGET_TYPE STREQUAL "SHARED_LIBRARY" OR TARGET_TYPE STREQUAL "STATIC_LIBRARY")
        set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "lib")
    endif ()

    # Link NumKong's shared library, which answers the capability queries and punned kernels for the host CPU.
    if (TARGET nk_shared)
        target_compile_definitions(${TARGET_NAME} PRIVATE "NK_RUNTIME_DISPATCH=1")
        target_link_libraries(${TARGET_NAME} PRIVATE nk_shared)
    endif ()

endfunction ()

# Sibling libraries build themselves: each ships a CMakeLists.txt handling its own ISA detection and dispatch. They
# arrive as submodules in a recursive clone, and as pinned fetches otherwise - a plain `git clone` of USearch, or a
# `FetchContent` of it from another project, still configures. Pins match the submodule commits.
include(FetchContent)

# The siblings are the engine rather than an option: every header reaches for them unguarded, so there is no build
# without them and no switch pretending otherwise. Include directories are copied onto the interface target, because
# `usearch_setup_target` reads `INTERFACE_INCLUDE_DIRECTORIES` directly - a genex that sees only what this target
# declares, not what its link interface would carry - to keep the C++20 feature requirement off the pure-C targets.
function (usearch_sibling NAME LINK_TARGET REPOSITORY TAG)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/CMakeLists.txt")
        add_subdirectory(${NAME})
    else ()
        message(STATUS "usearch: ${NAME} submodule absent, fetching ${TAG}")
        FetchContent_Declare(
            ${NAME}
            GIT_REPOSITORY ${REPOSITORY}
            GIT_TAG ${TAG}
        )
        FetchContent_MakeAvailable(${NAME})
    endif ()

    target_link_libraries(${USEARCH_TARGET_NAME} INTERFACE ${LINK_TARGET})
    target_include_directories(
        ${USEARCH_TARGET_NAME} ${USEARCH_SYSTEM_INCLUDE}
        INTERFACE $<TARGET_PROPERTY:${LINK_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
    )
endfunction ()

# NumKong wants its shared library built for runtime ISA dispatch, so the flag is set before the subdirectory lands.
set(NK_BUILD_SHARED
    ON
    CACHE BOOL "" FORCE
)

usearch_sibling(
    numkong numkong::numkong https://github.com/ashvardanian/NumKong 35b2b6b037c4987c05ec07fc699f9d2d524c98e3
)
usearch_sibling(
    forkunion forkunion::header https://github.com/ashvardanian/ForkUnion 15e4c36e6f8750ea1d2509a5e840d000cfb334ba
)
usearch_sibling(
    stringzilla stringzilla::stringzilla_header https://github.com/ashvardanian/StringZilla
    a5ac71758f177eb8a27883cd1bbae75698ff2c78
)
usearch_sibling(
    smashtable smashtable::smashtable https://github.com/ashvardanian/SmashTable
    cf6dfc559867d222522d5ec0b15a4218034626ee
)

# The dispatch library exports symbols the pre-loaded metrics resolve against at runtime.
if (TARGET nk_shared)
    set_target_properties(nk_shared PROPERTIES ENABLE_EXPORTS ON)
endif ()

# Only `safegraphs_format.hpp` reaches for yyjson, and only a compiled target ever includes it - so it is acquired under
# the same condition as `src/`, leaving the header-only target consumable without it.
if (USEARCH_BUILD_TEST_CPP
    OR USEARCH_BUILD_BENCH_CPP
    OR USEARCH_BUILD_LIB_C
    OR USEARCH_BUILD_TEST_C
)
    FetchContent_Declare(
        yyjson
        GIT_REPOSITORY https://github.com/ibireme/yyjson.git
        GIT_TAG 0.12.0
    )
    FetchContent_MakeAvailable(yyjson)
endif ()

# Pins for what only the suites link: {fmt} for the C++ smoke tests (libstdc++ doesn't ship `std::print` yet on our
# toolchain) and clipp for the benchmark CLIs. `FIND_PACKAGE_ARGS` lets a system-provided {fmt} satisfy the dependency
# without touching the network, but only from the pinned version up - the copies distributions ship predate
# `fmt::println` and would fail deep in the suites rather than here.
if (USEARCH_BUILD_TEST_CPP OR USEARCH_BUILD_BENCH_CPP)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 12.2.0
        FIND_PACKAGE_ARGS 12.2.0
    )
    FetchContent_Declare(
        clipp
        GIT_REPOSITORY https://github.com/muellan/clipp
        GIT_TAG v1.2.3
    )
    FetchContent_MakeAvailable(fmt clipp)

    # Upstream clipp ships no usable CMake target — just a single header — so wrap it in one.
    if (NOT TARGET clipp)
        add_library(clipp INTERFACE)
        target_include_directories(clipp INTERFACE "$<BUILD_INTERFACE:${clipp_SOURCE_DIR}/include>")
        # clipp still spells `std::result_of`, gone from libc++ in C++20 mode.
        target_compile_definitions(clipp INTERFACE _LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS)
    endif ()
endif ()

# Must be called before "add_subdirectory()", and only when USearch runs the show - a parent project decides its own
# testing story. See https://stackoverflow.com/questions/30250494/ctest-not-detecting-tests.
if (USEARCH_IS_MAIN_PROJECT)
    enable_testing()
endif ()

# Every consumer of the pre-instantiated library needs this, the C interface included - its implementation is
# `src/dense_index.cpp` and the per-dtype shards beside it.
if (${USEARCH_BUILD_TEST_CPP}
    OR ${USEARCH_BUILD_BENCH_CPP}
    OR ${USEARCH_BUILD_LIB_C}
    OR ${USEARCH_BUILD_TEST_C}
)
    add_subdirectory(src)
endif ()

if (${USEARCH_BUILD_LIB_C_STUB})
    add_subdirectory(src_stub)
endif ()

if (${USEARCH_BUILD_TEST_CPP})
    add_subdirectory(test)
endif ()

if (${USEARCH_BUILD_BENCH_CPP})
    add_subdirectory(bench)
endif ()

if (${USEARCH_BUILD_LIB_WOLFRAM})
    add_subdirectory(wolfram)
endif ()

if (${USEARCH_BUILD_LIB_C} OR ${USEARCH_BUILD_TEST_C})
    add_subdirectory(c)
endif ()

if (${USEARCH_BUILD_LIB_JAVA})
    add_subdirectory(java)
endif ()
