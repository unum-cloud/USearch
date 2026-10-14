# This CMake file is heavily inspired by following `usearch` CMake:
# https://github.com/nlohmann/json/blob/develop/CMakeLists.txt
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(
    usearch
    VERSION 2.25.1
    LANGUAGES C CXX
    DESCRIPTION "Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices"
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

# Each GPU vendor is probed only to default its `USEARCH_BUILD_<VENDOR>` switch, only while the caller leaves it unset,
# and only under Clang, which compiles every GPU build's host code. NVIDIA needs NVCC, though `check_language` caches
# the host compiler it paired with, which would then outrank a caller's own `CMAKE_CUDA_HOST_COMPILER`. AMD needs ROCm's
# own Clang as the C++ compiler, as the `rocm_*` presets name it. Apple needs only an Apple silicon target, since Metal
# ships with macOS and its kernels compile at first use.
include(CheckLanguage)
set(USEARCH_CAN_BUILD_CUDA OFF)
if (NOT DEFINED USEARCH_BUILD_CUDA AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        set(USEARCH_CAN_BUILD_CUDA ON)
    endif ()
endif ()
set(USEARCH_CAN_BUILD_ROCM OFF)
if (NOT DEFINED USEARCH_BUILD_ROCM AND CMAKE_CXX_COMPILER MATCHES "/rocm[^/]*/llvm/bin/clang")
    set(USEARCH_CAN_BUILD_ROCM ON)
endif ()
set(USEARCH_CAN_BUILD_METAL OFF)
if (NOT DEFINED USEARCH_BUILD_METAL
    AND APPLE
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64"
    AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"
)
    set(USEARCH_CAN_BUILD_METAL ON)
endif ()

# Options
option(USEARCH_INSTALL "Install CMake targets" OFF)

option(USEARCH_USE_JEMALLOC "Use JeMalloc for faster memory allocations" OFF)

option(USEARCH_BUILD_TEST "Compile test/, and c/test.c alongside USEARCH_BUILD_LIB_C" ${USEARCH_IS_MAIN_PROJECT})
option(USEARCH_BUILD_BENCH "Compile the benchmarks in bench/" ${USEARCH_IS_MAIN_PROJECT})
option(USEARCH_BUILD_LIB_C "Compile a native library for the C 99 interface" OFF)
option(USEARCH_BUILD_LIB_C_STUB "Compile a brute-force stand-in for the C 99 interface" OFF)
option(USEARCH_BUILD_LIB_WOLFRAM "Compile a native library for the Wolfram Language bindings" OFF)
option(USEARCH_BUILD_LIB_PYTHON "Compile the CPython extension; `scikit-build-core` switches it on" OFF)
option(USEARCH_BUILD_LIB_NODE "Compile the Node addon; `cmake-js` switches it on" OFF)
option(USEARCH_BUILD_CUDA "Compile the NVIDIA index arms, tests and benchmarks with NVCC" ${USEARCH_CAN_BUILD_CUDA})
option(USEARCH_BUILD_ROCM "Compile the AMD index arms and tests with ROCm's Clang" ${USEARCH_CAN_BUILD_ROCM})
option(USEARCH_BUILD_METAL "Compile the Apple index arms and tests over Metal" ${USEARCH_CAN_BUILD_METAL})
option(USEARCH_BUILD_SPDK "Compile the SPDK engine into every target, finding SPDK through pkg-config" OFF)

# Clang keeps a unit's host and device halves apart, through the host-only passes of its CUDA and HIP modes, where other
# host compilers trip over mixed units.
if ((USEARCH_BUILD_CUDA
     OR USEARCH_BUILD_ROCM
     OR USEARCH_BUILD_METAL)
    AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"
)
    message(
        FATAL_ERROR
            "Every GPU build compiles its host code with Clang, found ${CMAKE_CXX_COMPILER_ID} - set CMAKE_CXX_COMPILER to clang++, and to ROCm's own clang++ for USEARCH_BUILD_ROCM"
    )
endif ()

if (USEARCH_BUILD_SPDK)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(USEARCH_SPDK REQUIRED spdk_nvme spdk_env_dpdk)
    pkg_check_modules(USEARCH_SPDK_SYSTEM REQUIRED spdk_syslibs)
endif ()

# The library compiles one unit per GPU class, each for its own architecture alone - "80" with its PTX for every GPU
# without a class, "89", "90a", "100f" and "120f" - and ignores this list. The CUDA tests, benchmarks and the dispatch
# unit compile one pass per entry, and a NumKong GPU capability turns on in them only when every entry runs it, so
# iterate on one GPU with its family code, like "100f-real", to test that generation's kits. The "-real" entries skip
# the PTX, which would triple the objects of builds that never meet an unknown GPU.
set(USEARCH_CUDA_ARCHITECTURES
    "80-real;90a-real;100f-real;120f-real"
    CACHE STRING "GPU architectures the CUDA tests, benchmarks & dispatch unit compile for"
)

# The AMD units embed one code object per listed processor of their class, and the CDNA3 floor also for CDNA4, which
# keeps every instruction its kits use; CDNA5 drops `v_dot4_i32_i8` and `v_dot2_f32_f16`, so only its own unit runs it.
# A class with no processor listed embeds none, and its devices refuse to load.
set(USEARCH_ROCM_ARCHITECTURES
    "gfx942;gfx950;gfx1250"
    CACHE STRING "AMD processors the code objects compile for: gfx942 for CDNA3, gfx950 for CDNA4, gfx1250 for CDNA5"
)

# The library, its tests and benchmarks compile the first N storage dtypes of the ladder bf16, u1, i8, e4m3, f16, u8,
# e5m2, i4, u4, e3m2, e2m3, and the C interface refuses the rest with `usearch_status_unsupported_k`. The default four
# cover the common embeddings at a fraction of the build; the prebuilt packages compile all eleven.
set(USEARCH_DENSE_DTYPES
    "4"
    CACHE STRING "How many dense storage dtypes to compile, from 1 to 11"
)
if (NOT USEARCH_DENSE_DTYPES MATCHES "^([1-9]|1[01])$")
    message(FATAL_ERROR "USEARCH_DENSE_DTYPES must be an integer from 1 to 11, found \"${USEARCH_DENSE_DTYPES}\"")
endif ()
set(usearch_dense_dtypes_ladder_
    bf16
    u1
    i8
    e4m3
    f16
    u8
    e5m2
    i4
    u4
    e3m2
    e2m3
)

# Drops from @p target the per-dtype units, named by their `_<dtype>` suffix, of the dtypes the ladder leaves out; the
# f32 and f64 units are the suites' own references and always stay.
function (usearch_keep_dense_dtypes target)
    list(SUBLIST usearch_dense_dtypes_ladder_ 0 ${USEARCH_DENSE_DTYPES} compiled_)
    get_target_property(sources_ ${target} SOURCES)
    set(kept_)
    foreach (source_ IN LISTS sources_)
        get_filename_component(stem_ ${source_} NAME_WE)
        string(REGEX MATCH "[^_]+$" dtype_ ${stem_})
        if (dtype_ IN_LIST compiled_ OR NOT dtype_ IN_LIST usearch_dense_dtypes_ladder_)
            list(APPEND kept_ ${source_})
        endif ()
    endforeach ()
    set_target_properties(${target} PROPERTIES SOURCES "${kept_}")
endfunction ()

# Sets @p output to `capability_macro=1` where every one of the architectures, without CMake's "-real" or "-virtual", is
# one of NumKong's cached `nk_target_<capability>_architectures`, and to `capability_macro=0` otherwise. NumKong's
# one-code units need any of them; a unit here compiles every listed code, and a kit compiled into one pass is compiled
# into all.
function (usearch_cuda_capability capability_name capability_macro architectures output)
    set(enabled_ 1)
    foreach (architecture_ IN LISTS architectures)
        string(REGEX REPLACE "-(real|virtual)$" "" architecture_ "${architecture_}")
        if (NOT architecture_ IN_LIST nk_target_${capability_name}_architectures)
            set(enabled_ 0)
        endif ()
    endforeach ()
    set(${output}
        ${capability_macro}=${enabled_}
        PARENT_SCOPE
    )
endfunction ()

# The NumKong GPU capabilities the CUDA codes run, as the definitions a target compiles with.
function (usearch_cuda_capabilities output architectures)
    usearch_cuda_capability(ampere NUMKONG_TARGET_AMPERE "${architectures}" ampere_)
    usearch_cuda_capability(ada NUMKONG_TARGET_ADA "${architectures}" ada_)
    usearch_cuda_capability(hopper NUMKONG_TARGET_HOPPER "${architectures}" hopper_)
    usearch_cuda_capability(blackwell NUMKONG_TARGET_BLACKWELL "${architectures}" blackwell_)
    usearch_cuda_capability(blackwellrtx NUMKONG_TARGET_BLACKWELLRTX "${architectures}" blackwellrtx_)
    set(${output}
        ${ampere_} ${ada_} ${hopper_} ${blackwell_} ${blackwellrtx_}
        PARENT_SCOPE
    )
endfunction ()

# Three-digit "a" & "f" entries are only understood from CMake 3.31 onward, while the CPU-only build stays on the 3.28
# floor that stock Ubuntu 24.04 ships. The floor follows the architectures actually asked for, so a single-architecture
# build on a distribution CMake is never refused for a capability it does not use.
if (USEARCH_BUILD_CUDA AND CMAKE_VERSION VERSION_LESS 3.31)
    string(REGEX MATCH "1[0-9][0-9][af]" architectures_needing_cmake_331 "${USEARCH_CUDA_ARCHITECTURES}")
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
set(USEARCH_TARGET_NAME usearch_header)
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

# Include roots that add no path component: the C interface in `c/`, the templates in `include/`, and the C++ wrappers
# over the C interface in `cpp/`, which only the compiled libraries carry.
set(USEARCH_INCLUDE_BUILD_DIRS "${PROJECT_SOURCE_DIR}/c/" "${PROJECT_SOURCE_DIR}/include/")

# The core: the basic graph index and the utilities it stands on - six headers on the standard library alone, no sibling
# library in sight. Concepts are the main reason C++20 is the base language level, with `std::atomic_ref` and the
# `<bit>` helpers close behind - and the interface target must say so, or external consumers get an unbuildable
# contract.
add_library(usearch_core INTERFACE)
add_library(${PROJECT_NAME}::core ALIAS usearch_core)
set_target_properties(usearch_core PROPERTIES EXPORT_NAME core)
target_sources(
    usearch_core
    INTERFACE FILE_SET
              HEADERS
              BASE_DIRS
              ${USEARCH_INCLUDE_BUILD_DIRS}
              FILES
              c/usearch/types.h
              include/usearch/utilities/types.hpp
              include/usearch/utilities/sanitizers.hpp
              include/usearch/utilities/ram_buffers.hpp
              include/usearch/utilities/ram_sorts.hpp
              include/usearch/utilities/ram_tables.hpp
              include/usearch/basic/ram_index.hpp
)
target_compile_features(usearch_core INTERFACE cxx_std_20)

# The whole header-only library: every header, the siblings it reaches into, and the type verdicts below.
add_library(${USEARCH_TARGET_NAME} INTERFACE)
add_library(${PROJECT_NAME}::header ALIAS ${USEARCH_TARGET_NAME})
set_target_properties(${USEARCH_TARGET_NAME} PROPERTIES EXPORT_NAME header)
target_link_libraries(${USEARCH_TARGET_NAME} INTERFACE usearch_core)

target_include_directories(
    ${USEARCH_TARGET_NAME} ${USEARCH_SYSTEM_INCLUDE} INTERFACE "$<BUILD_INTERFACE:${USEARCH_INCLUDE_BUILD_DIRS}>"
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
    install(DIRECTORY c/usearch include/usearch cpp/usearch DESTINATION ${USEARCH_INCLUDE_INSTALL_DIR})
    install(FILES ${USEARCH_CMAKE_PROJECT_CONFIG_FILE} ${USEARCH_CMAKE_VERSION_CONFIG_FILE}
            DESTINATION ${USEARCH_CONFIG_INSTALL_DIR}
    )
    install(
        TARGETS usearch_core ${USEARCH_TARGET_NAME}
        EXPORT ${USEARCH_TARGETS_EXPORT_NAME}
        FILE_SET HEADERS
        DESTINATION ${USEARCH_INCLUDE_INSTALL_DIR}
        INCLUDES
        DESTINATION ${USEARCH_INCLUDE_INSTALL_DIR}
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
# & FML, no SME - leaving NumKong's f16 kits serial; `-mcpu=native` is the complete AArch64 knob.
if (MSVC OR CMAKE_CROSSCOMPILING)
    set(USEARCH_HOST_TUNING "")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(USEARCH_HOST_TUNING -mcpu=native)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(USEARCH_HOST_TUNING -march=native)
else ()
    set(USEARCH_HOST_TUNING "")
endif ()

# Debug builds take each sanitizer whose instrumented load the toolchain compiles and links; WASI links no sanitizer
# runtime at all.
set(USEARCH_SANITIZERS "")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    foreach (usearch_sanitizer_ address leak alignment undefined)
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=${usearch_sanitizer_}")
        check_cxx_source_compiles(
            "int main(int argc, char **argv) { return *(int *)(void *)argv[0] + argc; }"
            USEARCH_SANITIZER_${usearch_sanitizer_}_
        )
        if (USEARCH_SANITIZER_${usearch_sanitizer_}_)
            list(APPEND USEARCH_SANITIZERS "-fsanitize=${usearch_sanitizer_}")
        endif ()
    endforeach ()
    cmake_pop_check_state()
endif ()

# Shared wiring for every target this project builds. Optimization and debug level are deliberately absent: they live in
# `CMAKE_<LANG>_FLAGS_<CONFIG>`, where a preset or a command-line switch can move them. Only what the code actually
# depends on - fast math, the diagnostics, the sanitizers - is pinned here, and only for the host languages, so a unit
# the CUDA driver compiles is not handed flags it rejects. `HOST` tunes for the machine at hand and takes no probe
# verdict, since the compiler's own feature macros then say exactly what the host has.
function (usearch_setup_target TARGET_NAME)
    cmake_parse_arguments(usearch_setup "HOST" "" "" ${ARGN})
    # Compiler-specific options List of all possible compiler IDs:
    # https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(
            ${TARGET_NAME}
            PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:DEBUG>:${USEARCH_SANITIZERS}>
                    -ffast-math
                    -fPIC
                    -Wall
                    -Wextra
                    -Wno-conversion
                    -Wno-unknown-pragmas
                    -fmax-errors=${USEARCH_MAX_ERRORS}
                    -pedantic
                    -fdiagnostics-color=always>
        )
        target_link_options(${TARGET_NAME} PRIVATE $<$<CONFIG:DEBUG>:-g ${USEARCH_SANITIZERS}> -fPIC)

        target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)

    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(
            ${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:DEBUG>:${USEARCH_SANITIZERS}> -ffast-math
                                   -fPIC -pedantic -Wfatal-errors -fcolor-diagnostics>
        )
        target_link_options(${TARGET_NAME} PRIVATE $<$<CONFIG:DEBUG>:-g ${USEARCH_SANITIZERS}> -fPIC)

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

    # One source of truth for header paths: whatever the interface target advertises, including what the dependencies
    # carry, without inheriting its compile features onto pure-C targets.
    target_include_directories(
        ${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:${USEARCH_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
    )
    # Host-tuned units read the compiler; every other unit takes each sibling's compiled capabilities and instantiates
    # its kits per CPU class, never by a template's default capability, so the class units and the dispatch unit name
    # the same rungs.
    if (usearch_setup_HOST)
        target_compile_options(${TARGET_NAME} PRIVATE ${USEARCH_HOST_TUNING})
    else ()
        target_link_libraries(
            ${TARGET_NAME}
            PRIVATE $<BUILD_INTERFACE:numkong::cpu_capabilities_compiled>
                    $<BUILD_INTERFACE:stringzilla::cpu_capabilities_compiled>
                    $<BUILD_INTERFACE:forkunion::cpu_capabilities_compiled>
        )
    endif ()
    set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 20)

    # Every library carries the `lib` prefix, Windows included, so one file name serves every platform.
    get_target_property(TARGET_TYPE ${TARGET_NAME} TYPE)

    if (TARGET_TYPE STREQUAL "SHARED_LIBRARY" OR TARGET_TYPE STREQUAL "STATIC_LIBRARY")
        set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "lib")
    endif ()

    # Header-only NumKong: each unit inlines the kernels its CPU class runs, and USearch dispatches by CPU class, so no
    # artifact needs NumKong's library or its dispatch points.
    target_link_libraries(${TARGET_NAME} PRIVATE numkong::header)

    # SPDK's transports, socket layers and DPDK drivers register from static constructors, so their archives link whole.
    if (USEARCH_BUILD_SPDK)
        target_compile_definitions(${TARGET_NAME} PRIVATE USEARCH_BUILD_SPDK=1)
        target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${USEARCH_SPDK_INCLUDE_DIRS})
        target_link_libraries(
            ${TARGET_NAME} PRIVATE "-Wl,--whole-archive" "-Wl,-Bstatic" ${USEARCH_SPDK_LDFLAGS} "-Wl,-Bdynamic"
                                   "-Wl,--no-whole-archive" ${USEARCH_SPDK_SYSTEM_STATIC_LDFLAGS}
        )
    endif ()

endfunction ()

# Sibling libraries build themselves: each ships a CMakeLists.txt handling its own ISA detection and dispatch. They
# arrive as submodules in a recursive clone, and as pinned fetches otherwise - a plain `git clone` of USearch, or a
# `FetchContent` of it from another project, still configures. Pins match the submodule commits.
include(FetchContent)

# The siblings are the engine rather than an option: every header reaches for them unguarded, so there is no build
# without them and no switch pretending otherwise. Include directories are copied onto the interface target, because
# `usearch_setup_target` reads `INTERFACE_INCLUDE_DIRECTORIES` directly - a genex that sees only what this target
# declares, not what its link interface would carry - to keep the C++20 feature requirement off the pure-C targets. So
# each sibling's paths are copied onto the interface here.
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

    # Collected for the export set: their interfaces carry a language level and `Threads::Threads`, which the paths
    # copied below do not.
    get_target_property(SIBLING_TARGET ${LINK_TARGET} ALIASED_TARGET)
    if (NOT SIBLING_TARGET)
        set(SIBLING_TARGET ${LINK_TARGET})
    endif ()
    set(USEARCH_SIBLING_TARGETS
        ${USEARCH_SIBLING_TARGETS} ${SIBLING_TARGET}
        PARENT_SCOPE
    )

    target_include_directories(
        ${USEARCH_TARGET_NAME} ${USEARCH_SYSTEM_INCLUDE}
        INTERFACE $<TARGET_PROPERTY:${LINK_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
    )
endfunction ()

# Every artifact lands at the top of the build tree, however deep the directory that declares it - one `-L`, one path to
# a binary, no guessing which subdirectory a target chose. Multi-configuration generators would nest a second level,
# `build_debug\Debug` on Windows, so each configuration is pinned to the same place as the default. Set before any
# target exists, since these seed a target's properties at creation.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach (USEARCH_CONFIGURATION DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${USEARCH_CONFIGURATION} "${CMAKE_BINARY_DIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${USEARCH_CONFIGURATION} "${CMAKE_BINARY_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${USEARCH_CONFIGURATION} "${CMAKE_BINARY_DIR}")
endforeach ()

# The compiled siblings land as archives inside our shared objects, so every one is position-independent.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

usearch_sibling(
    numkong numkong::header https://github.com/ashvardanian/NumKong 215377db8fc434fe96e3d548d416670cbafc6198
)
usearch_sibling(
    forkunion forkunion::header https://github.com/ashvardanian/ForkUnion 075d14137420f204ff175acd2d29c22e163e241c
)
usearch_sibling(
    stringzilla stringzilla::header https://github.com/ashvardanian/StringZilla
    7826f358a51df3871fba5cc825ebab3e1163174a
)
usearch_sibling(
    smashtable smashtable::header https://github.com/ashvardanian/SmashTable a7df2585228a444755ce85ac4b1a86dc425a3cee
)

# Recorded without headers - each sibling owns its own - so `usearch::header` names nothing a consumer cannot resolve.
if (USEARCH_INSTALL AND USEARCH_SIBLING_TARGETS)
    install(TARGETS ${USEARCH_SIBLING_TARGETS} EXPORT ${USEARCH_TARGETS_EXPORT_NAME})
endif ()

# Both exports read the set as it stands when called, so they trail every target that belongs to it.
if (USEARCH_INSTALL)
    export(
        TARGETS usearch_core ${USEARCH_TARGET_NAME} ${USEARCH_SIBLING_TARGETS}
        NAMESPACE ${PROJECT_NAME}::
        FILE ${USEARCH_CMAKE_PROJECT_TARGETS_FILE}
    )
    install(
        EXPORT ${USEARCH_TARGETS_EXPORT_NAME}
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${USEARCH_CONFIG_INSTALL_DIR}
    )
endif ()

# Only `safegraphs_format.hpp` reaches for yyjson, and only a compiled target ever includes it - so it is acquired under
# the same condition as `src/`, leaving the header-only target consumable without it.
if (USEARCH_BUILD_TEST
    OR USEARCH_BUILD_BENCH
    OR USEARCH_BUILD_LIB_C
)
    # Its objects go inside our own libraries, so a packager installing this tree must not publish yyjson's header,
    # archive and CMake config beside ours.
    set(YYJSON_INSTALL
        OFF
        CACHE BOOL "" FORCE
    )
    FetchContent_Declare(
        yyjson
        GIT_REPOSITORY https://github.com/ibireme/yyjson.git
        GIT_TAG 0.12.0
    )
    FetchContent_MakeAvailable(yyjson)
endif ()

# The pin for what only the suites link: {fmt} for the C++ smoke tests and benchmarks, as libstdc++ doesn't ship
# `std::print` yet on our toolchain. `FIND_PACKAGE_ARGS` lets a system-provided {fmt} satisfy the dependency without
# touching the network, but only from the pinned version up - the copies distributions ship predate `fmt::println` and
# would fail deep in the suites rather than here.
if (USEARCH_BUILD_TEST OR USEARCH_BUILD_BENCH)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 12.2.0
        FIND_PACKAGE_ARGS 12.2.0
    )
    FetchContent_MakeAvailable(fmt)
endif ()

# Must be called before "add_subdirectory()", and only when USearch runs the show - a parent project decides its own
# testing story. See https://stackoverflow.com/questions/30250494/ctest-not-detecting-tests.
if (USEARCH_IS_MAIN_PROJECT)
    enable_testing()
endif ()

# The arms, their dispatch and the suites below all read one ladder length.
add_compile_definitions(USEARCH_DENSE_DTYPES=${USEARCH_DENSE_DTYPES})

# Every consumer of the pre-instantiated library needs this, the C interface included - its implementation is
# `src/dense_index.cpp` and the per-dtype shards beside it.
if (${USEARCH_BUILD_TEST}
    OR ${USEARCH_BUILD_BENCH}
    OR ${USEARCH_BUILD_LIB_C}
)
    add_subdirectory(src)
endif ()

if (${USEARCH_BUILD_LIB_C_STUB})
    add_subdirectory(src_stub)
endif ()

if (${USEARCH_BUILD_TEST})
    add_subdirectory(test)
endif ()

if (${USEARCH_BUILD_BENCH})
    add_subdirectory(bench)
endif ()

if (${USEARCH_BUILD_LIB_WOLFRAM})
    add_subdirectory(wolfram)
endif ()

if (${USEARCH_BUILD_LIB_C})
    add_subdirectory(c)
    add_subdirectory(cpp)
endif ()

if (${USEARCH_BUILD_LIB_PYTHON})
    add_subdirectory(python)
endif ()

if (${USEARCH_BUILD_LIB_NODE})
    add_subdirectory(javascript)
endif ()
