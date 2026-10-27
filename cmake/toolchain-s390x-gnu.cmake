# IBM Z Linux, cross-compiled with the GNU toolchain.
#
# ~~~
#   sudo apt install crossbuild-essential-s390x qemu-user
#   cmake -B build_s390x_gnu -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-s390x-gnu.cmake
#   cmake --build build_s390x_gnu --target usearch_shared
# ~~~
#
# The only big-endian target, so it is what exercises the checkpoint's byte-order conversion rather than leaving it dead
# - keep it in any sweep that touches serialization.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR s390x)

find_program(CMAKE_C_COMPILER s390x-linux-gnu-gcc REQUIRED)
find_program(CMAKE_CXX_COMPILER s390x-linux-gnu-g++ REQUIRED)

find_program(usearch_qemu_s390x_ qemu-s390x)
if (usearch_qemu_s390x_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${usearch_qemu_s390x_};-L;/usr/s390x-linux-gnu")
endif ()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
