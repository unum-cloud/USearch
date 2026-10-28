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
# the header-only targets, the siblings and the other dependencies, the install and export rules, the wiring every
# compiled target shares, and the subdirectories.

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
option(USEARCH_BUILD_CUDA "Compile the NVIDIA index variants, tests and benchmarks with NVCC" OFF)
option(USEARCH_BUILD_ROCM "Compile the AMD index variants and tests with ROCm's Clang" OFF)
option(USEARCH_BUILD_METAL "Compile the Apple index variants and tests over Metal" OFF)
option(USEARCH_WITH_SPDK "Compile the SPDK engine into every target, finding SPDK through pkg-config" OFF)
option(USEARCH_WERROR "Treat compiler warnings as errors" OFF)

# Seeded from the environment at the first configure, so `cargo`, `cmake-js` and `scikit-build-core` honour
# `USEARCH_TARGET_ARCH=native`.
set(USEARCH_TARGET_ARCH
    "$ENV{USEARCH_TARGET_ARCH}"
    CACHE STRING
          "CPU the own units are tuned for: empty for the architecture floor, native, or a CPU name like graniterapids"
)
set(USEARCH_TARGET_CLASSES
    ""
    CACHE STRING "CPU classes compiled beside serial: empty for every class, a list like intel_granite, or native"
)
if (USEARCH_TARGET_ARCH STREQUAL "native" AND CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "USEARCH_TARGET_ARCH=native needs the host as the target; name a CPU instead")
endif ()
# Only a native build asks the siblings which capabilities this host runs.
if (USEARCH_TARGET_CLASSES STREQUAL "native" AND NOT USEARCH_TARGET_ARCH STREQUAL "native")
    message(FATAL_ERROR "USEARCH_TARGET_CLASSES=native needs USEARCH_TARGET_ARCH=native")
endif ()

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
# "100f-real" builds the Blackwell kernels alone. The CUDA tests, benchmarks and the dispatch unit compile one pass per
# entry, and a NumKong GPU capability turns on in them only when every entry runs it, so iterate on one GPU with its
# family code, like "100f-real", to test that generation's kits. The "-real" entries skip the PTX, which would triple
# the objects of builds that never meet an unknown GPU. The kernels start at Ampere, and Turing's "75" would turn every
# NumKong GPU capability off in the default tests.
set(USEARCH_CUDA_ARCHITECTURES
    "80-real;89-real;90a-real;100f-real;120f-real;80-virtual"
    CACHE STRING
          "GPU architectures the CUDA tests, benchmarks & dispatch unit compile for, or native for the visible GPUs"
)

# "native" maps each GPU `nvidia-smi` lists to the code its class needs, family suffix included, where a bare compute
# capability would drop it.
set(usearch_cuda_architectures_)
if (USEARCH_BUILD_CUDA)
    set(usearch_cuda_native_codes_
        75-real=75-real
        80-real=80-real
        86-real=86-real
        89-real=89-real
        90-real=90a-real
        100-real=100f-real
        103-real=103f-real
        110-real=110f-real
        120-real=120f-real
        121-real=121f-real
    )
    set(usearch_cuda_visible_)
    if ("native" IN_LIST USEARCH_CUDA_ARCHITECTURES)
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE usearch_cuda_visible_
            ERROR_QUIET
        )
        separate_arguments(usearch_cuda_visible_ UNIX_COMMAND "${usearch_cuda_visible_}")
        if (NOT usearch_cuda_visible_)
            message(
                FATAL_ERROR "USEARCH_CUDA_ARCHITECTURES=native found no GPU through nvidia-smi; name its code instead"
            )
        endif ()
    endif ()
    foreach (entry_ IN LISTS USEARCH_CUDA_ARCHITECTURES)
        if (NOT entry_ STREQUAL "native")
            list(APPEND usearch_cuda_architectures_ ${entry_})
        endif ()
    endforeach ()
    foreach (visible_ IN LISTS usearch_cuda_visible_)
        set(native_code_)
        if (visible_ MATCHES "^([0-9]+)\\.([0-9])$")
            set(visible_ "${CMAKE_MATCH_1}${CMAKE_MATCH_2}-real")
            foreach (pair_ IN LISTS usearch_cuda_native_codes_)
                if (pair_ MATCHES "^${visible_}=(.+)$")
                    set(native_code_ ${CMAKE_MATCH_1})
                endif ()
            endforeach ()
        endif ()
        if (NOT native_code_)
            message(FATAL_ERROR "USEARCH_CUDA_ARCHITECTURES=native has no code for the visible ${visible_} GPU; "
                                "name its code instead"
            )
        endif ()
        if (NOT native_code_ IN_LIST usearch_cuda_architectures_)
            list(APPEND usearch_cuda_architectures_ ${native_code_})
        endif ()
    endforeach ()
endif ()

# Three-digit "a" & "f" entries are only understood from CMake 3.31 onward, while the CPU-only build stays on the 3.28
# floor that stock Ubuntu 24.04 ships. The floor follows the architectures actually asked for, so a single-architecture
# build on a distribution CMake is never refused for a capability it does not use.
if (CMAKE_VERSION VERSION_LESS 3.31 AND usearch_cuda_architectures_ MATCHES "1[0-9][0-9][af]")
    message(
        FATAL_ERROR
            "USEARCH_CUDA_ARCHITECTURES \"${usearch_cuda_architectures_}\" names a three-digit \"a\" or \"f\" architecture, which needs CMake 3.31 or newer, found ${CMAKE_VERSION} - override with a single architecture"
    )
endif ()

# The AMD units embed one code object per listed processor of their class, and the CDNA3 floor also for CDNA4, which
# keeps every instruction its kits use; CDNA5 drops `v_dot4_i32_i8` and `v_dot2_f32_f16`, so only its own unit runs it.
# A class with no processor listed embeds none, and its devices refuse to load.
set(USEARCH_ROCM_ARCHITECTURES
    "gfx942;gfx950;gfx1250;gfx1251"
    CACHE STRING "AMD processors the code objects compile for: gfx942 for CDNA3, gfx950 for CDNA4, gfx125x for CDNA5"
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

# The package files: a pkg-config file, so other tools can find this, and the CMake config with its version file. A
# header-only package fits any architecture, and `ARCH_INDEPENDENT` declares exactly that - no hand-written version
# template needed.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(usearch_config_install_dir_ "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}")
set(usearch_targets_export_name_ "${PROJECT_NAME}Targets")
set(usearch_cmake_version_config_file_ "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(usearch_cmake_project_config_file_ "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/pkg-config.pc.in" "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc")
write_basic_package_version_file(
    ${usearch_cmake_version_config_file_}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion ARCH_INDEPENDENT
)
configure_file(cmake/config.cmake.in ${usearch_cmake_project_config_file_} @ONLY)

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

# Sibling libraries build themselves: each ships a CMakeLists.txt handling its own ISA detection and dispatch. They
# arrive as submodules in a recursive clone, and as pinned fetches otherwise - a plain `git clone` of USearch, or a
# `FetchContent` of it from another project, still configures. Pins match the submodule commits. The siblings are the
# engine rather than an option: every header reaches for them unguarded, so there is no build without them and no switch
# pretending otherwise. Include directories are copied onto the interface target, because `usearch_setup_target` reads
# `INTERFACE_INCLUDE_DIRECTORIES` directly - a genex that sees only what this target declares, not what its link
# interface would carry - to keep the C++20 feature requirement off the pure-C targets. Every sibling is a system
# dependency, so its own warnings never fail a USearch build under `USEARCH_WERROR`, and builds only what a USearch
# target links. A defined @p override_, the sibling's `FETCHCONTENT_SOURCE_DIR_<NAME>`, outranks the submodule.
function (usearch_sibling name_ override_ link_target_ repository_ tag_)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name_}/CMakeLists.txt" AND NOT DEFINED ${override_})
        add_subdirectory(${name_} EXCLUDE_FROM_ALL SYSTEM)
    else ()
        message(STATUS "usearch: ${name_} submodule absent, fetching ${tag_}")
        FetchContent_Declare(
            ${name_}
            GIT_REPOSITORY ${repository_}
            GIT_TAG ${tag_}
            EXCLUDE_FROM_ALL SYSTEM
        )
        FetchContent_MakeAvailable(${name_})
    endif ()

    target_link_libraries(usearch_header INTERFACE ${link_target_})

    # Collected for the export set: their interfaces carry a language level and `Threads::Threads`, which the paths
    # copied below do not.
    get_target_property(sibling_target_ ${link_target_} ALIASED_TARGET)
    if (NOT sibling_target_)
        set(sibling_target_ ${link_target_})
    endif ()
    set(usearch_sibling_targets_
        ${usearch_sibling_targets_} ${sibling_target_}
        PARENT_SCOPE
    )

    target_include_directories(
        usearch_header SYSTEM INTERFACE $<TARGET_PROPERTY:${link_target_},INTERFACE_INCLUDE_DIRECTORIES>
    )
endfunction ()

# The siblings tune their own units for the CPU USearch is tuned for.
if (USEARCH_TARGET_ARCH)
    set(NUMKONG_TARGET_ARCH "${USEARCH_TARGET_ARCH}")
    set(STRINGZILLA_TARGET_ARCH "${USEARCH_TARGET_ARCH}")
    set(FORKUNION_TARGET_ARCH "${USEARCH_TARGET_ARCH}")
endif ()

# NumKong: the distance kits and tensors, header-only, as every unit inlines the kernels its CPU or GPU class runs.
# ForkUnion: the thread pools, the memory topology, and the ISA-specific waiters and atomic references. StringZilla: the
# string kits behind the sequence modality, from the Levenshtein steps to the overlaps. SmashTable: the Elias-Fano row
# directories of the NVMe indexes.
set(usearch_sibling_targets_)
usearch_sibling(
    numkong FETCHCONTENT_SOURCE_DIR_NUMKONG numkong::header https://github.com/ashvardanian/NumKong
    3faee602cdbcc3ee1ca8880594e4c1b2d38c8f91
)
usearch_sibling(
    forkunion FETCHCONTENT_SOURCE_DIR_FORKUNION forkunion::header https://github.com/ashvardanian/ForkUnion
    12a0a2706534452830868b7e33a4697549c898d5
)
usearch_sibling(
    stringzilla FETCHCONTENT_SOURCE_DIR_STRINGZILLA stringzilla::header https://github.com/ashvardanian/StringZilla
    a4e9f5b0ac91481a5c30957e05d2310f3928c1d8
)
usearch_sibling(
    smashtable FETCHCONTENT_SOURCE_DIR_SMASHTABLE smashtable::header https://github.com/ashvardanian/SmashTable
    e0db741da4e7b9cf6bc13aa493eff264aff3cfb8
)
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

# NVCC is only a device compiler here, so the toolkit is found for its paths alone, and every NVIDIA and AMD unit links
# its device bundle through an assembler stub. Both serve every directory below, which inherits them.
if (USEARCH_BUILD_CUDA)
    find_package(CUDAToolkit 13 REQUIRED)
endif ()
if (USEARCH_BUILD_CUDA OR USEARCH_BUILD_ROCM)
    enable_language(ASM)
endif ()

# The headers, the package files and the export set, which records the siblings without headers - each owns its own - so
# `usearch::header` names nothing a consumer cannot resolve. Both exports read the set as it stands when called, so they
# trail every target that belongs to it.
if (USEARCH_INSTALL)
    install(DIRECTORY c/usearch include/usearch cpp/usearch DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${usearch_cmake_project_config_file_} ${usearch_cmake_version_config_file_}
            DESTINATION ${usearch_config_install_dir_}
    )
    install(
        TARGETS usearch_core usearch_header
        EXPORT ${usearch_targets_export_name_}
        FILE_SET HEADERS
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" DESTINATION "${CMAKE_INSTALL_DATADIR}/pkgconfig")
    if (usearch_sibling_targets_)
        install(TARGETS ${usearch_sibling_targets_} EXPORT ${usearch_targets_export_name_})
    endif ()
    export(
        TARGETS usearch_core usearch_header ${usearch_sibling_targets_}
        NAMESPACE ${PROJECT_NAME}::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
    )
    install(
        EXPORT ${usearch_targets_export_name_}
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${usearch_config_install_dir_}
    )
endif ()

# USEARCH_TARGET_ARCH sets the floor of USearch's own C and C++ units, and each class unit adds its class's flags over
# it: empty keeps the architecture's floor, whatever a distribution compiler defaults to, `native` tunes for this host,
# and a CPU name like `graniterapids` for that CPU, also when cross-compiling. AArch64 names CPUs through `-mcpu`, as
# `-march` takes only architecture versions.
set(usearch_target_arch_flags_)
if (CMAKE_SYSTEM_NAME MATCHES "Emscripten|WASI")
    # A WebAssembly module's one SIMD capability comes from its toolchain file.
elseif (MSVC AND USEARCH_TARGET_ARCH STREQUAL "native")
    # MSVC has no `/arch:native`, so it takes the `/arch:` of the widest capability this host runs.
    foreach (capability_ IN ITEMS diamond skylake haswell neonbfdot neonsdot neonhalf)
        if (numkong_target_${capability_}_runs AND NOT usearch_target_arch_flags_)
            set(usearch_target_arch_flags_ ${numkong_target_${capability_}_flags})
        endif ()
    endforeach ()
elseif (MSVC AND USEARCH_TARGET_ARCH)
    message(
        FATAL_ERROR
            "MSVC takes USEARCH_TARGET_ARCH=native only; pass an /arch: value such as AVX2, AVX512 or AVX10.2 through "
            "CMAKE_CXX_FLAGS instead of `${USEARCH_TARGET_ARCH}`"
    )
elseif (USEARCH_TARGET_ARCH AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(usearch_target_arch_flags_ -mcpu=${USEARCH_TARGET_ARCH})
elseif (USEARCH_TARGET_ARCH)
    set(usearch_target_arch_flags_ -march=${USEARCH_TARGET_ARCH})
elseif (MSVC OR APPLE)
    # MSVC's defaults are the floor, and on Apple `-arch` pins it per slice of a universal build.
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(usearch_target_arch_flags_ -march=armv8-a)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(usearch_target_arch_flags_ -march=x86-64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set(usearch_target_arch_flags_ -march=rv64gc)
endif ()

# Host tuning for the tests & benches only - never the distributable libraries - and only while USEARCH_TARGET_ARCH is
# empty, the host is the target and the compiler has a spelling for it; elsewhere the suite runs on the switch's floor
# and the SIMD variants report themselves skipped. Apple Clang's `-march=native` advertises only a subset of the host's
# features - no FP16 arithmetic & FML, no SME - leaving NumKong's f16 kits serial; `-mcpu=native` is the complete
# AArch64 knob.
if (USEARCH_TARGET_ARCH
    OR MSVC
    OR CMAKE_CROSSCOMPILING
)
    set(usearch_host_tuning_ "")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(usearch_host_tuning_ -mcpu=native)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(usearch_host_tuning_ -march=native)
else ()
    set(usearch_host_tuning_ "")
endif ()

# Warnings apply to USearch's own targets only, so fetched dependencies build with their own flags. The C interface
# takes named arguments as partially designated structs, leaving the rest zeroed. CUDA sources take none, since NVCC
# rejects the host spellings.
add_library(usearch_warnings INTERFACE)
if (MSVC)
    if (USEARCH_WERROR)
        target_compile_options(usearch_warnings INTERFACE /WX)
    endif ()
else ()
    target_compile_options(
        usearch_warnings
        INTERFACE
            "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Wpedantic;-Wno-conversion;-Wno-unknown-pragmas;-Wno-missing-field-initializers>"
            "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CXX_COMPILER_ID:Clang,AppleClang>>:-Wdocumentation>"
    )
    if (USEARCH_WERROR)
        target_compile_options(usearch_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>")
    endif ()
endif ()

# Debug builds take each sanitizer whose instrumented load the toolchain compiles and links; WASI links no sanitizer
# runtime at all.
set(usearch_sanitizers_ "")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    foreach (usearch_sanitizer_ address leak alignment undefined)
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=${usearch_sanitizer_}")
        check_cxx_source_compiles(
            "int main(int argc, char **argv) { return *(int *)(void *)argv[0] + argc; }"
            usearch_sanitizer_${usearch_sanitizer_}_compiles_
        )
        if (usearch_sanitizer_${usearch_sanitizer_}_compiles_)
            list(APPEND usearch_sanitizers_ "-fsanitize=${usearch_sanitizer_}")
        endif ()
    endforeach ()
    cmake_pop_check_state()
endif ()

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
    target_link_libraries(${target_} PRIVATE usearch_warnings)

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
    target_link_libraries(${target_} PRIVATE numkong::header)

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
# The C library as SwiftPM links it: an XCFramework on Apple platforms and an SE-0482 artifact bundle elsewhere, packed
# by NumKong's script beside the C headers and their module map. Each build adds its target to the artifact in
# USEARCH_SWIFT_DIRECTORY, so a release fills one artifact from a build per target.
if (USEARCH_BUILD_SWIFT)
    set(USEARCH_SWIFT_DIRECTORY
        "${PROJECT_BINARY_DIR}"
        CACHE PATH "Where the Swift artifact lands"
    )
    set(USEARCH_SWIFT_SDK
        ""
        CACHE STRING "The Apple SDK the archive targets, like `macosx` or `iphonesimulator`"
    )
    set(USEARCH_SWIFT_TRIPLES
        "${CMAKE_SYSTEM_PROCESSOR}-unknown-linux-gnu"
        CACHE STRING "Swift target triples the artifact bundle serves"
    )
    if (APPLE)
        set(usearch_swift_target_ -D "NUMKONG_SWIFT_SDK=${USEARCH_SWIFT_SDK}" -D
                                  "NUMKONG_SWIFT_ARCHITECTURE=${CMAKE_OSX_ARCHITECTURES}"
        )
    else ()
        set(usearch_swift_target_ -D "NUMKONG_SWIFT_TRIPLES=$<JOIN:${USEARCH_SWIFT_TRIPLES},,>" -D
                                  "NUMKONG_SWIFT_VERSION=${PROJECT_VERSION}"
        )
    endif ()
    set(usearch_swift_headers_ "${PROJECT_BINARY_DIR}/swift_headers")
    add_custom_target(
        usearch_swift ALL
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${PROJECT_SOURCE_DIR}/c/usearch"
                "${usearch_swift_headers_}/usearch"
        COMMAND "${CMAKE_COMMAND}" -E copy "${PROJECT_SOURCE_DIR}/include/module.modulemap" "${usearch_swift_headers_}"
        COMMAND
            "${CMAKE_COMMAND}" -D NUMKONG_SWIFT_MODULE=USearchC -D "NUMKONG_SWIFT_ARCHIVE=$<TARGET_FILE:usearch_static>"
            -D "NUMKONG_SWIFT_HEADERS=${usearch_swift_headers_}" -D "NUMKONG_SWIFT_DIRECTORY=${USEARCH_SWIFT_DIRECTORY}"
            ${usearch_swift_target_} -P "${numkong_SOURCE_DIR}/cmake/numkong_swift_artifact.cmake"
        DEPENDS usearch_static
        VERBATIM
    )
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
