# AArch64 Linux, cross-compiled with the GNU toolchain.
#
# ~~~
#   sudo apt install crossbuild-essential-arm64 qemu-user
#   cmake -B build_aarch64_gnu -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-gnu.cmake
#   cmake --build build_aarch64_gnu --target usearch_shared
# ~~~
#
# No compiler version is named anywhere: `find_program` takes whatever the prefix resolves to. GCC's own driver supplies
# the SME runtime routines the streaming kernels call, which is why this file carries no link flags where the LLVM
# sibling does.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

find_program(CMAKE_C_COMPILER aarch64-linux-gnu-gcc REQUIRED)
find_program(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++ REQUIRED)

find_program(usearch_qemu_aarch64_ qemu-aarch64)
if (usearch_qemu_aarch64_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${usearch_qemu_aarch64_};-L;/usr/aarch64-linux-gnu;-cpu;max")
endif ()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
