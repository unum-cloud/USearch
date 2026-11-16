# FreeBSD x86-64, cross-compiled with the host's Clang and LLD against a FreeBSD sysroot.
#
# ~~~
#   curl -O https://download.freebsd.org/releases/amd64/14.5-RELEASE/base.txz
#   mkdir freebsd-sysroot && tar -xJf base.txz -C freebsd-sysroot ./usr/include ./usr/lib ./lib
#   cmake -B build_freebsd -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-x86_64-freebsd-llvm.cmake -D USEARCH_BUILD_TEST=ON \
#       -D USEARCH_FREEBSD_SYSROOT=$PWD/freebsd-sysroot
#   cmake --build build_freebsd
# ~~~
set(CMAKE_SYSTEM_NAME FreeBSD)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(USEARCH_FREEBSD_SYSROOT
    "$ENV{USEARCH_FREEBSD_SYSROOT}"
    CACHE PATH "An unpacked FreeBSD `base.txz`"
)
# Compiler checks re-read this file in a project of their own, which needs the path forwarded.
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES USEARCH_FREEBSD_SYSROOT)
if (NOT IS_DIRECTORY "${USEARCH_FREEBSD_SYSROOT}/usr/include")
    message(FATAL_ERROR "USEARCH_FREEBSD_SYSROOT must name an unpacked FreeBSD base, not '${USEARCH_FREEBSD_SYSROOT}'")
endif ()
set(CMAKE_SYSROOT "${USEARCH_FREEBSD_SYSROOT}")

find_program(CMAKE_C_COMPILER clang REQUIRED)
find_program(CMAKE_CXX_COMPILER clang++ REQUIRED)

set(CMAKE_C_COMPILER_TARGET x86_64-unknown-freebsd14.5)
set(CMAKE_CXX_COMPILER_TARGET x86_64-unknown-freebsd14.5)
# The sysroot's libc++ headers, not the host Clang's newer ones, which name symbols FreeBSD's libc++ lacks.
set(CMAKE_CXX_FLAGS_INIT "-nostdinc++ -isystem ${USEARCH_FREEBSD_SYSROOT}/usr/include/c++/v1")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
