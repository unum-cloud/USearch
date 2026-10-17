cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(
    usearch
    VERSION 3.0.0
    LANGUAGES C CXX
    DESCRIPTION "Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices"
    HOMEPAGE_URL "https://github.com/unum-cloud/USearch"
)

# The file runs top to bottom: the options and the settings they validate, the defaults every target is created under,
# the header-only targets, the dependencies, one `cmake/<dependency>.cmake` each, the install and export rules, the
# wiring every compiled target shares, and the subdirectories.

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
option(USEARCH_BUILD_DISTRIBUTED "Compile the MPI drivers in distributed/" OFF)
option(USEARCH_BUILD_CUDA "Compile the NVIDIA index variants, tests and benchmarks with NVCC" ${USEARCH_CAN_BUILD_CUDA})
option(USEARCH_BUILD_ROCM "Compile the AMD index variants and tests with ROCm's Clang" ${USEARCH_CAN_BUILD_ROCM})
option(USEARCH_BUILD_METAL "Compile the Apple index variants and tests over Metal" ${USEARCH_CAN_BUILD_METAL})
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

# The library compiles one unit per GPU class, each for its own architecture alone - "80" with its PTX for every GPU
# without a class, "89", "90a", "100f" and "120f" - and only the classes this list names, with or without "-real", so
# "100f-real" builds the Blackwell kernels alone and "native" builds every class. The CUDA tests, benchmarks and the
# dispatch unit compile one pass per entry, and a NumKong GPU capability turns on in them only when every entry runs it,
# so iterate on one GPU with its family code, like "100f-real", to test that generation's kits. The "-real" entries skip
# the PTX, which would triple the objects of builds that never meet an unknown GPU.
set(USEARCH_CUDA_ARCHITECTURES
    "80-real;89-real;90a-real;100f-real;120f-real"
    CACHE STRING "GPU architectures the CUDA tests, benchmarks & dispatch unit compile for"
)

# Three-digit "a" & "f" entries are only understood from CMake 3.31 onward, while the CPU-only build stays on the 3.28
# floor that stock Ubuntu 24.04 ships. The floor follows the architectures actually asked for, so a single-architecture
# build on a distribution CMake is never refused for a capability it does not use.
if (USEARCH_BUILD_CUDA
    AND CMAKE_VERSION VERSION_LESS 3.31
    AND USEARCH_CUDA_ARCHITECTURES MATCHES "1[0-9][0-9][af]"
)
    message(
        FATAL_ERROR
            "USEARCH_CUDA_ARCHITECTURES \"${USEARCH_CUDA_ARCHITECTURES}\" names a three-digit \"a\" or \"f\" architecture, which needs CMake 3.31 or newer, found ${CMAKE_VERSION} - override with a single architecture or \"native\""
    )
endif ()

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

# One diagnostic is what an iteration wants; surveying a tree that has drifted wants the whole list at once.
set(USEARCH_MAX_ERRORS
    "1"
    CACHE STRING "Diagnostics a compiler emits before giving up on a translation unit"
)

# Every consumer of the pre-instantiated library builds `src/`, the C interface included - its implementation is
# `src/dense_index.cpp` and the per-dtype shards beside it - and acquires yyjson for it.
set(usearch_build_punned_ OFF)
if (USEARCH_BUILD_TEST
    OR USEARCH_BUILD_BENCH
    OR USEARCH_BUILD_LIB_C
    OR USEARCH_BUILD_DISTRIBUTED
)
    set(usearch_build_punned_ ON)
endif ()

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any cloned repo
# execute code), so point `core.hooksPath` at `.githooks` here - configuring the build arms the `pre-commit` checks.
# Skipped when built as a subproject or without `.git` (tarball / export builds).
if (USEARCH_IS_MAIN_PROJECT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
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
add_library(usearch_header INTERFACE)
add_library(${PROJECT_NAME}::header ALIAS usearch_header)
set_target_properties(usearch_header PROPERTIES EXPORT_NAME header)
target_link_libraries(usearch_header INTERFACE usearch_core)
target_include_directories(
    usearch_header INTERFACE "$<BUILD_INTERFACE:${USEARCH_INCLUDE_BUILD_DIRS}>" $<INSTALL_INTERFACE:include>
)

# The package files: a pkg-config file, so other tools can find this, and the CMake config with its version file. A
# header-only package fits any architecture, and `ARCH_INDEPENDENT` declares exactly that - no hand-written version
# template needed.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(USEARCH_CONFIG_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}"
    CACHE INTERNAL ""
)
set(USEARCH_TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(USEARCH_CMAKE_VERSION_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(USEARCH_CMAKE_PROJECT_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/pkg-config.pc.in" "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc")
write_basic_package_version_file(
    ${USEARCH_CMAKE_VERSION_CONFIG_FILE}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion ARCH_INDEPENDENT
)
configure_file(cmake/config.cmake.in ${USEARCH_CMAKE_PROJECT_CONFIG_FILE} @ONLY)

# Dependencies, each acquired by its own `cmake/<dependency>.cmake`, and each only for the builds that compile against
# it.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include(FetchContent)

if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif ()
find_package(Threads REQUIRED)

# Sibling libraries build themselves: each ships a CMakeLists.txt handling its own ISA detection and dispatch. They
# arrive as submodules in a recursive clone, and as pinned fetches otherwise - a plain `git clone` of USearch, or a
# `FetchContent` of it from another project, still configures. Pins match the submodule commits. The siblings are the
# engine rather than an option: every header reaches for them unguarded, so there is no build without them and no switch
# pretending otherwise. Include directories are copied onto the interface target, because `usearch_setup_target` reads
# `INTERFACE_INCLUDE_DIRECTORIES` directly - a genex that sees only what this target declares, not what its link
# interface would carry - to keep the C++20 feature requirement off the pure-C targets.
function (usearch_sibling name link_target repository tag)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name}/CMakeLists.txt")
        add_subdirectory(${name})
    else ()
        message(STATUS "usearch: ${name} submodule absent, fetching ${tag}")
        FetchContent_Declare(
            ${name}
            GIT_REPOSITORY ${repository}
            GIT_TAG ${tag}
        )
        FetchContent_MakeAvailable(${name})
    endif ()

    target_link_libraries(usearch_header INTERFACE ${link_target})

    # Collected for the export set: their interfaces carry a language level and `Threads::Threads`, which the paths
    # copied below do not.
    get_target_property(sibling_target_ ${link_target} ALIASED_TARGET)
    if (NOT sibling_target_)
        set(sibling_target_ ${link_target})
    endif ()
    set(usearch_sibling_targets_
        ${usearch_sibling_targets_} ${sibling_target_}
        PARENT_SCOPE
    )

    target_include_directories(usearch_header INTERFACE $<TARGET_PROPERTY:${link_target},INTERFACE_INCLUDE_DIRECTORIES>)
endfunction ()

set(usearch_sibling_targets_)
include(numkong)
include(forkunion)
include(stringzilla)
include(smashtable)
if (usearch_build_punned_)
    include(yyjson)
endif ()
if (USEARCH_BUILD_TEST
    OR USEARCH_BUILD_BENCH
    OR USEARCH_BUILD_DISTRIBUTED
)
    include(fmt)
endif ()
if (USEARCH_BUILD_DISTRIBUTED)
    include(mpi)
endif ()
if (USEARCH_USE_JEMALLOC)
    include(jemalloc)
endif ()
if (USEARCH_BUILD_SPDK)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(USEARCH_SPDK REQUIRED spdk_nvme spdk_env_dpdk)
    pkg_check_modules(USEARCH_SPDK_SYSTEM REQUIRED spdk_syslibs)
endif ()

# NVCC is only a device compiler here, so the toolkit is found for its paths alone, and every NVIDIA and AMD unit links
# its device bundle through an assembler stub. Both serve every directory below, which inherits them.
if (USEARCH_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif ()
if (USEARCH_BUILD_CUDA OR USEARCH_BUILD_ROCM)
    enable_language(ASM)
endif ()

# The headers, the package files and the export set, which records the siblings without headers - each owns its own - so
# `usearch::header` names nothing a consumer cannot resolve. Both exports read the set as it stands when called, so they
# trail every target that belongs to it.
if (USEARCH_INSTALL)
    install(DIRECTORY c/usearch include/usearch cpp/usearch DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${USEARCH_CMAKE_PROJECT_CONFIG_FILE} ${USEARCH_CMAKE_VERSION_CONFIG_FILE}
            DESTINATION ${USEARCH_CONFIG_INSTALL_DIR}
    )
    install(
        TARGETS usearch_core usearch_header
        EXPORT ${USEARCH_TARGETS_EXPORT_NAME}
        FILE_SET HEADERS
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" DESTINATION "${CMAKE_INSTALL_DATADIR}/pkgconfig")
    if (usearch_sibling_targets_)
        install(TARGETS ${usearch_sibling_targets_} EXPORT ${USEARCH_TARGETS_EXPORT_NAME})
    endif ()
    export(
        TARGETS usearch_core usearch_header ${usearch_sibling_targets_}
        NAMESPACE ${PROJECT_NAME}::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
    )
    install(
        EXPORT ${USEARCH_TARGETS_EXPORT_NAME}
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${USEARCH_CONFIG_INSTALL_DIR}
    )
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
function (usearch_setup_target target)
    cmake_parse_arguments(arg "HOST" "" "" ${ARGN})
    # Compiler-specific options List of all possible compiler IDs:
    # https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(
            ${target}
            PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:DEBUG>:${USEARCH_SANITIZERS}>
                    -ffast-math
                    -fno-finite-math-only
                    -fPIC
                    -Wall
                    -Wextra
                    -Wno-conversion
                    -Wno-unknown-pragmas
                    -fmax-errors=${USEARCH_MAX_ERRORS}
                    -pedantic
                    -fdiagnostics-color=always>
        )
        target_link_options(${target} PRIVATE $<$<CONFIG:DEBUG>:-g ${USEARCH_SANITIZERS}> -fPIC)

        target_link_libraries(${target} PRIVATE Threads::Threads)

    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(
            ${target}
            PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:DEBUG>:${USEARCH_SANITIZERS}>
                    -ffast-math
                    -fno-finite-math-only
                    -fPIC
                    -pedantic
                    -Wfatal-errors
                    -fcolor-diagnostics>
        )
        target_link_options(${target} PRIVATE $<$<CONFIG:DEBUG>:-g ${USEARCH_SANITIZERS}> -fPIC)

        if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
            target_link_libraries(${target} PRIVATE "-framework CoreFoundation" "-framework Security" Threads::Threads)
        else ()
            target_link_libraries(${target} PRIVATE Threads::Threads)
        endif ()

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(${target} PRIVATE -w -ferror-limit=${USEARCH_MAX_ERRORS})

    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # MSVC has no O3 optimization level
        target_compile_options(${target} PRIVATE /wd4068 /wd4996)
        target_compile_options(
            ${target} PRIVATE $<$<CONFIG:DEBUG>:/DEBUG> $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:RELWITHDEBINFO>:/O2 /DEBUG>
        )
    endif ()

    # One source of truth for header paths: whatever the interface target advertises, including what the dependencies
    # carry, without inheriting its compile features onto pure-C targets.
    target_include_directories(${target} PRIVATE $<TARGET_PROPERTY:usearch_header,INTERFACE_INCLUDE_DIRECTORIES>)
    # Host-tuned units read the compiler; every other unit takes each sibling's compiled capabilities and instantiates
    # its kits per CPU class, never by a template's default capability, so the class units and the dispatch unit name
    # the same rungs.
    if (arg_HOST)
        target_compile_options(${target} PRIVATE ${USEARCH_HOST_TUNING})
    else ()
        target_link_libraries(
            ${target}
            PRIVATE $<BUILD_INTERFACE:numkong::cpu_capabilities_compiled>
                    $<BUILD_INTERFACE:stringzilla::cpu_capabilities_compiled>
                    $<BUILD_INTERFACE:forkunion::cpu_capabilities_compiled>
        )
    endif ()
    set_target_properties(${target} PROPERTIES CXX_STANDARD 20)

    # Every library carries the `lib` prefix, Windows included, so one file name serves every platform.
    get_target_property(target_type_ ${target} TYPE)
    if (target_type_ STREQUAL "SHARED_LIBRARY" OR target_type_ STREQUAL "STATIC_LIBRARY")
        set_target_properties(${target} PROPERTIES PREFIX "lib")
    endif ()

    # Header-only NumKong: each unit inlines the kernels its CPU class runs, and USearch dispatches by CPU class, so no
    # artifact needs NumKong's library or its dispatch points.
    target_link_libraries(${target} PRIVATE numkong::header)

    # SPDK's transports, socket layers and DPDK drivers register from static constructors, so their archives link whole.
    if (USEARCH_BUILD_SPDK)
        target_compile_definitions(${target} PRIVATE USEARCH_BUILD_SPDK=1)
        target_include_directories(${target} SYSTEM PRIVATE ${USEARCH_SPDK_INCLUDE_DIRS})
        target_link_libraries(
            ${target} PRIVATE "-Wl,--whole-archive" "-Wl,-Bstatic" ${USEARCH_SPDK_LDFLAGS} "-Wl,-Bdynamic"
                              "-Wl,--no-whole-archive" ${USEARCH_SPDK_SYSTEM_STATIC_LDFLAGS}
        )
    endif ()

    # JeMalloc serves the processes this project starts, its executables; the C library links it in `c/`.
    if (USEARCH_USE_JEMALLOC AND target_type_ STREQUAL "EXECUTABLE")
        target_link_libraries(${target} PRIVATE ${JEMALLOC_LIBRARIES})
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
function (usearch_dense_dtype_sources target)
    cmake_parse_arguments(PARSE_ARGV 1 shards "" "" "${usearch_dense_dtypes_ladder_}")
    if (shards_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "${target} lists \"${shards_UNPARSED_ARGUMENTS}\" under no dense dtype of the ladder")
    endif ()
    set(position_ 0)
    foreach (dtype_ IN LISTS usearch_dense_dtypes_ladder_)
        if (position_ LESS USEARCH_DENSE_DTYPES)
            target_sources(${target} PRIVATE ${shards_${dtype_}})
        endif ()
        math(EXPR position_ "${position_} + 1")
    endforeach ()
endfunction ()

# Must be called before "add_subdirectory()", and only when USearch runs the show - a parent project decides its own
# testing story. See https://stackoverflow.com/questions/30250494/ctest-not-detecting-tests.
if (USEARCH_IS_MAIN_PROJECT)
    enable_testing()
endif ()

# The variants, their dispatch and the suites below all read one ladder length.
add_compile_definitions(USEARCH_DENSE_DTYPES=${USEARCH_DENSE_DTYPES})

if (usearch_build_punned_)
    add_subdirectory(src)
endif ()
if (USEARCH_BUILD_LIB_C_STUB)
    add_subdirectory(src_stub)
endif ()
if (USEARCH_BUILD_TEST)
    add_subdirectory(test)
endif ()
if (USEARCH_BUILD_BENCH)
    add_subdirectory(bench)
endif ()
if (USEARCH_BUILD_LIB_WOLFRAM)
    add_subdirectory(wolfram)
endif ()
if (USEARCH_BUILD_LIB_C)
    add_subdirectory(c)
    add_subdirectory(cpp)
endif ()
if (USEARCH_BUILD_DISTRIBUTED)
    add_subdirectory(distributed)
endif ()
if (USEARCH_BUILD_LIB_PYTHON)
    add_subdirectory(python)
endif ()
if (USEARCH_BUILD_LIB_NODE)
    add_subdirectory(javascript)
endif ()
