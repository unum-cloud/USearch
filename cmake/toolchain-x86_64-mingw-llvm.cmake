# Windows x64, cross-compiled with the host's Clang and LLD against MinGW-w64's headers, CRT and libstdc++.
#
# ~~~
#   sudo apt install clang lld g++-mingw-w64-x86-64-posix
#   cmake -B build_mingw -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-x86_64-mingw-llvm.cmake -D USEARCH_BUILD_TEST=ON
#   cmake --build build_mingw
# ~~~
#
# The binaries link statically, so a Windows host runs them without MinGW's DLLs.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

find_program(CMAKE_C_COMPILER clang REQUIRED)
find_program(CMAKE_CXX_COMPILER clang++ REQUIRED)
# A host archiver for another architecture writes no index of COFF members, which LLD then cannot find.
find_program(CMAKE_AR x86_64-w64-mingw32-ar REQUIRED)
find_program(CMAKE_RANLIB x86_64-w64-mingw32-ranlib REQUIRED)

set(CMAKE_C_COMPILER_TARGET x86_64-w64-windows-gnu)
set(CMAKE_CXX_COMPILER_TARGET x86_64-w64-windows-gnu)
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld -static")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld")

set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
