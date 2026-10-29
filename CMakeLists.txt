cmake_minimum_required(VERSION 3.28)
project(
    usearch
    VERSION 3.0.0
    LANGUAGES C CXX
    DESCRIPTION "Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices"
    HOMEPAGE_URL "https://github.com/unum-cloud/USearch"
)

# No target uses C++20 modules, and a Clang installed without `clang-scan-deps` would fail every C++ unit.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# The file runs top to bottom: the options and the settings they validate, the defaults every target is created under,
# the header-only targets, the siblings and the other dependencies, the wiring every compiled target shares, the
# subdirectories, and the install and export rules.

# Allow CMake 3.13+ to override options when using FetchContent / add_subdirectory
if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

# Options
option(USEARCH_INSTALL "Install CMake targets" OFF)

option(USEARCH_WITH_JEMALLOC "Use JeMalloc for faster memory allocations" OFF)

option(USEARCH_BUILD_TEST "Compile test/, and c/test.c alongside USEARCH_BUILD_SHARED" OFF)
option(USEARCH_BUILD_BENCH "Compile the benchmarks in bench/" OFF)
option(USEARCH_BUILD_SHARED "Build and install the shared and static C 99 libraries by default" OFF)
option(USEARCH_BUILD_STUB "Compile a brute-force stand-in for the C 99 interface" OFF)
option(USEARCH_BUILD_WOLFRAM "Compile a native library for the Wolfram Language bindings" OFF)
option(USEARCH_BUILD_PYTHON "Compile the CPython extension; `scikit-build-core` switches it on" OFF)
option(USEARCH_BUILD_NODE "Compile the Node addon; `cmake-js` switches it on" OFF)
option(USEARCH_BUILD_SWIFT "Package the static library for SwiftPM, as an XCFramework or an artifact bundle" OFF)
option(USEARCH_BUILD_TOOLS "Compile the command-line tools in tools/, and their MPI build where MPI is found" OFF)
set(USEARCH_CUDA_ARCH
    ""
    CACHE STRING "NVIDIA code the index variants, tests and benchmarks build for: empty, all, native or one like 100f"
)
set(USEARCH_ROCM_ARCH
    ""
    CACHE STRING "AMD processor the index variants and tests build for: empty, all, native or one like gfx942"
)
set(USEARCH_METAL_ARCH
    ""
    CACHE STRING "Apple GPU family the index variants and tests serve: empty, all or one like apple9"
)
option(USEARCH_WITH_SPDK "Compile the SPDK engine into every target, finding SPDK through pkg-config" OFF)
option(USEARCH_WERROR "Treat compiler warnings as errors" OFF)

# Seeded from the environment at the first configure, so `cargo`, `cmake-js` and `scikit-build-core` honour
# `USEARCH_MARCH=native` and `USEARCH_MULTIVERSIONED=OFF`.
set(USEARCH_MARCH
    "$ENV{USEARCH_MARCH}"
    CACHE STRING
          "The `-march` floor: empty for the architecture floor, an ISA level, native, or a CPU like graniterapids"
)
set(usearch_multiversioned_ ON)
if (DEFINED ENV{USEARCH_MULTIVERSIONED})
    set(usearch_multiversioned_ "$ENV{USEARCH_MULTIVERSIONED}")
endif ()
option(USEARCH_MULTIVERSIONED "Compile every CPU class above the floor, or one monoversioned class for USEARCH_MARCH"
       ${usearch_multiversioned_}
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

# One diagnostic is what an iteration wants; surveying a tree that has drifted wants the whole list at once.
set(USEARCH_MAX_ERRORS
    "1"
    CACHE STRING "Diagnostics a compiler emits before giving up on a translation unit"
)

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any cloned repo
# execute code), so point `core.hooksPath` at `.githooks` here - configuring the build arms the `pre-commit` checks.
# Skipped when built as a subproject or without `.git` (tarball / export builds).
if (PROJECT_IS_TOP_LEVEL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE usearch_hooks_status_
        OUTPUT_QUIET ERROR_QUIET
    )
    if (usearch_hooks_status_ EQUAL 0)
        message(STATUS "usearch: Git hooks enabled (core.hooksPath -> .githooks)")
    endif ()
endif ()

# The build type when the caller names none.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif ()

# Every artifact lands at the top of the build tree, however deep the directory that declares it - one `-L`, one path to
# a binary, no guessing which subdirectory a target chose. Multi-configuration generators would nest a second level,
# `build_debug\Debug` on Windows, so each configuration is pinned to the same place as the default. Set before any
# target exists, since these seed a target's properties at creation.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach (configuration_ DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${configuration_} "${CMAKE_BINARY_DIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${configuration_} "${CMAKE_BINARY_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${configuration_} "${CMAKE_BINARY_DIR}")
endforeach ()

# The compiled siblings land as archives inside our shared objects, so every one is position-independent.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include roots that add no path component: the C interface in `c/`, the templates in `include/`, and the C++ wrappers
# over the C interface in `cpp/`, which only the compiled libraries carry.
set(usearch_include_build_dirs_ "${PROJECT_SOURCE_DIR}/c/" "${PROJECT_SOURCE_DIR}/include/")

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
              ${usearch_include_build_dirs_}
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
add_library(usearch_header INTERFACE)
add_library(${PROJECT_NAME}::header ALIAS usearch_header)
set_target_properties(usearch_header PROPERTIES EXPORT_NAME header)
target_link_libraries(usearch_header INTERFACE usearch_core)
target_include_directories(
    usearch_header INTERFACE "$<BUILD_INTERFACE:${usearch_include_build_dirs_}>" $<INSTALL_INTERFACE:include>
)

include(GNUInstallDirs)

# Dependencies: the siblings through `usearch_sibling`, the rest each acquired by its own `cmake/<dependency>.cmake`,
# and each only for the builds that compile against it.
include(FetchContent)

if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif ()
find_package(Threads REQUIRED)

include(cmake/usearch_siblings.cmake)
include(cmake/yyjson.cmake)
if (USEARCH_BUILD_TEST
    OR USEARCH_BUILD_BENCH
    OR USEARCH_BUILD_TOOLS
)
    include(cmake/fmt.cmake)
endif ()
if (USEARCH_WITH_JEMALLOC)
    include(cmake/jemalloc.cmake)
endif ()
if (USEARCH_WITH_SPDK)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(usearch_spdk REQUIRED spdk_nvme spdk_env_dpdk)
    pkg_check_modules(usearch_spdk_system REQUIRED spdk_syslibs)
endif ()

include(cmake/usearch_gpu.cmake)

include(cmake/usearch_platform.cmake)

include(cmake/usearch_flags.cmake)

# Shared wiring for every target this project builds. Optimization and debug level are deliberately absent: they live in
# `CMAKE_<LANG>_FLAGS_<CONFIG>`, where a preset or a command-line switch can move them. Only what the code actually
# depends on - fast math, the diagnostics, the sanitizers - is pinned here, and only for the host languages, so a unit
# the CUDA driver compiles is not handed flags it rejects. `HOST` tunes for the machine at hand and takes no probe
# verdict, since the compiler's own feature macros then say exactly what the host has.
function (usearch_setup_target target_)
    cmake_parse_arguments(PARSE_ARGV 1 arg "HOST" "" "")
    # Compiler-specific options List of all possible compiler IDs:
    # https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(
            ${target_}
            PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:DEBUG>:${usearch_sanitizers_}> -ffast-math
                    -fno-finite-math-only -fPIC -fmax-errors=${USEARCH_MAX_ERRORS} -fdiagnostics-color=always>
        )
        target_link_options(${target_} PRIVATE $<$<CONFIG:DEBUG>:-g ${usearch_sanitizers_}> -fPIC)

        target_link_libraries(${target_} PRIVATE Threads::Threads)

    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(
            ${target_} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:DEBUG>:${usearch_sanitizers_}> -ffast-math
                               -fno-finite-math-only -fPIC -Wfatal-errors -fcolor-diagnostics>
        )
        target_link_options(${target_} PRIVATE $<$<CONFIG:DEBUG>:-g ${usearch_sanitizers_}> -fPIC)

        if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
            target_link_libraries(${target_} PRIVATE "-framework CoreFoundation" "-framework Security" Threads::Threads)
        else ()
            target_link_libraries(${target_} PRIVATE Threads::Threads)
        endif ()

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(${target_} PRIVATE -w -ferror-limit=${USEARCH_MAX_ERRORS})

    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # MSVC has no O3 optimization level
        target_compile_options(${target_} PRIVATE /wd4068 /wd4996)
        target_compile_options(
            ${target_} PRIVATE $<$<CONFIG:DEBUG>:/DEBUG> $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:RELWITHDEBINFO>:/O2
                               /DEBUG>
        )
    endif ()
    target_link_libraries(${target_} PRIVATE $<BUILD_INTERFACE:usearch_warnings>)

    # One source of truth for header paths: whatever the interface target advertises, including what the dependencies
    # carry, without inheriting its compile features onto pure-C targets.
    target_include_directories(${target_} PRIVATE $<TARGET_PROPERTY:usearch_header,INTERFACE_INCLUDE_DIRECTORIES>)
    target_include_directories(
        ${target_} SYSTEM PRIVATE $<TARGET_PROPERTY:usearch_header,INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>
    )
    # Host-tuned units read the compiler; every other unit takes each sibling's compiled capabilities and instantiates
    # its kits per CPU class, never by a template's default capability, so the class units and the dispatch unit name
    # the same rungs.
    if (arg_HOST AND usearch_host_tuning_)
        target_compile_options(${target_} PRIVATE ${usearch_host_tuning_})
    else ()
        target_compile_options(${target_} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:${usearch_target_arch_flags_}>")
    endif ()
    if (NOT arg_HOST)
        target_link_libraries(
            ${target_}
            PRIVATE $<BUILD_INTERFACE:numkong::cpu_capabilities_compiled>
                    $<BUILD_INTERFACE:stringzilla::cpu_capabilities_compiled>
                    $<BUILD_INTERFACE:forkunion::cpu_capabilities_compiled>
        )
    endif ()
    set_target_properties(${target_} PROPERTIES CXX_STANDARD 20)

    # Every library carries the `lib` prefix, Windows included, so one file name serves every platform.
    get_target_property(target_type_ ${target_} TYPE)
    if (target_type_ STREQUAL "SHARED_LIBRARY" OR target_type_ STREQUAL "STATIC_LIBRARY")
        set_target_properties(${target_} PROPERTIES PREFIX "lib")
    endif ()

    # Header-only NumKong: each unit inlines the kernels its CPU class runs, and USearch dispatches by CPU class, so no
    # artifact needs NumKong's library or its dispatch points.
    target_link_libraries(${target_} PRIVATE $<BUILD_INTERFACE:numkong::header>)

    # SPDK's transports, socket layers and DPDK drivers register from static constructors, so their archives link whole.
    if (USEARCH_WITH_SPDK)
        target_compile_definitions(${target_} PRIVATE USEARCH_WITH_SPDK=1)
        target_include_directories(${target_} SYSTEM PRIVATE ${usearch_spdk_INCLUDE_DIRS})
        # An object library hands its link items to every consumer, which links the archives whole once already.
        if (NOT target_type_ STREQUAL "OBJECT_LIBRARY")
            target_link_libraries(
                ${target_} PRIVATE "-Wl,--whole-archive" "-Wl,-Bstatic" ${usearch_spdk_LDFLAGS} "-Wl,-Bdynamic"
                                   "-Wl,--no-whole-archive" ${usearch_spdk_system_STATIC_LDFLAGS}
            )
        endif ()
    endif ()

    # JeMalloc serves the processes this project starts, its executables; the C library links it in `c/`.
    if (USEARCH_WITH_JEMALLOC AND target_type_ STREQUAL "EXECUTABLE")
        target_link_libraries(${target_} PRIVATE ${USEARCH_JEMALLOC_LIBRARY})
    endif ()
endfunction ()

# Adds to @p target the per-dtype units listed after each dtype's keyword, for the dtypes the ladder compiles:
# `usearch_dense_dtype_sources(target BF16 a_bf16.cpp F16 a_f16.cpp ...)`. The f32 and f64 units are the suites' own
# references, outside the ladder, so their targets list them directly.
set(usearch_dense_dtypes_ladder_
    BF16
    U1
    I8
    E4M3
    F16
    U8
    E5M2
    I4
    U4
    E3M2
    E2M3
)
function (usearch_dense_dtype_sources target_)
    set(dtype_)
    foreach (argument_ IN LISTS ARGN)
        if (argument_ IN_LIST usearch_dense_dtypes_ladder_)
            set(dtype_ ${argument_})
        elseif (dtype_)
            list(APPEND sources_${dtype_}_ ${argument_})
        else ()
            message(FATAL_ERROR "${target_} lists \"${argument_}\" under no dense dtype of the ladder")
        endif ()
    endforeach ()
    set(position_ 0)
    foreach (dtype_ IN LISTS usearch_dense_dtypes_ladder_)
        if (position_ LESS USEARCH_DENSE_DTYPES)
            target_sources(${target_} PRIVATE ${sources_${dtype_}_})
        endif ()
        math(EXPR position_ "${position_} + 1")
    endforeach ()
endfunction ()

# Must be called before "add_subdirectory()", and only when USearch runs the show - a parent project decides its own
# testing story. See https://stackoverflow.com/questions/30250494/ctest-not-detecting-tests.
if (PROJECT_IS_TOP_LEVEL)
    enable_testing()
endif ()

# The variants, their dispatch and the suites below all read one ladder length.
add_compile_definitions(USEARCH_DENSE_DTYPES=${USEARCH_DENSE_DTYPES})

# The punned objects are the C interface's implementation, built only for the targets that link them.
add_subdirectory(src EXCLUDE_FROM_ALL)
if (USEARCH_BUILD_STUB)
    add_subdirectory(src_stub)
endif ()
if (USEARCH_BUILD_TEST)
    add_subdirectory(test)
endif ()
if (USEARCH_BUILD_BENCH)
    add_subdirectory(bench)
endif ()
add_subdirectory(c)
if (USEARCH_BUILD_SHARED)
    add_subdirectory(cpp)
endif ()
if (USEARCH_BUILD_WOLFRAM)
    add_subdirectory(wolfram)
endif ()
if (USEARCH_BUILD_TOOLS)
    add_subdirectory(tools)
endif ()
if (USEARCH_BUILD_PYTHON)
    add_subdirectory(python)
endif ()
if (USEARCH_BUILD_NODE)
    add_subdirectory(javascript)
endif ()

include(cmake/usearch_install.cmake)
