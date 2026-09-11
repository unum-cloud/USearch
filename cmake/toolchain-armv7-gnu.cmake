# 32-bit Arm Linux, hard-float, cross-compiled with the GNU toolchain.
#
# ~~~
#   sudo apt install crossbuild-essential-armhf qemu-user
#   cmake -B build_armv7_gnu -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-armv7-gnu.cmake -D USEARCH_BUILD_LIB_C=1
#   cmake --build build_armv7_gnu --target usearch_shared
# ~~~
#
# The one 32-bit target here, so it is what catches a `std::size_t` assumed to be eight bytes.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7)

find_program(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc REQUIRED)
find_program(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++ REQUIRED)

find_program(USEARCH_QEMU_ARM_ qemu-arm)
if (USEARCH_QEMU_ARM_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${USEARCH_QEMU_ARM_};-L;/usr/arm-linux-gnueabihf")
endif ()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
