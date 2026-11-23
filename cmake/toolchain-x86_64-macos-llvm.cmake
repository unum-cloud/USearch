# macOS x86-64, compiled with the host's Clang against a macOS SDK's headers, as far as the static library: linking
# needs Apple's linker and libraries, so executables build on a Mac.
#
# ~~~
#   cmake -B build_macos -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-x86_64-macos-llvm.cmake \
#       -D USEARCH_MACOS_SDK=/path/to/MacOSX.sdk
#   cmake --build build_macos --target usearch_static
# ~~~
set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(USEARCH_MACOS_SDK
    "$ENV{USEARCH_MACOS_SDK}"
    CACHE PATH "A macOS SDK, or a tree holding its `usr/include`"
)
# Compiler checks re-read this file in a project of their own, which needs the path forwarded.
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES USEARCH_MACOS_SDK)
if (NOT IS_DIRECTORY "${USEARCH_MACOS_SDK}/usr/include")
    message(FATAL_ERROR "USEARCH_MACOS_SDK must name a macOS SDK, not '${USEARCH_MACOS_SDK}'")
endif ()
set(CMAKE_OSX_SYSROOT "${USEARCH_MACOS_SDK}")
set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)

find_program(CMAKE_C_COMPILER clang REQUIRED)
find_program(CMAKE_CXX_COMPILER clang++ REQUIRED)
find_program(CMAKE_AR llvm-ar REQUIRED)
find_program(CMAKE_RANLIB llvm-ranlib REQUIRED)

set(CMAKE_C_COMPILER_TARGET x86_64-apple-macos26)
set(CMAKE_CXX_COMPILER_TARGET x86_64-apple-macos26)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
