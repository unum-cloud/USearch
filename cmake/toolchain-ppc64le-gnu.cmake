# PowerPC 64 little-endian Linux, cross-compiled with GCC 15 or newer, and run under QEMU.
#
# ~~~
#   sudo apt install crossbuild-essential-ppc64el qemu-user
#   cmake --preset linux_ppc64le
#   cmake --build --preset linux_ppc64le
#   ctest --preset linux_ppc64le
# ~~~
#
# GCC 15 is the floor, as StringZilla's POWER9 kernels open their half of `altivec.h` per function only from there. QEMU
# emulates a POWER10, which runs the `ibm_power10` class: VSX kernels, `pause_short` waits and ISA 3.0 store atomics.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)

# The `g++-15-powerpc64le-linux-gnu` package installs only the versioned name, so it is looked up first.
find_program(CMAKE_C_COMPILER NAMES powerpc64le-linux-gnu-gcc-15 powerpc64le-linux-gnu-gcc REQUIRED)
find_program(CMAKE_CXX_COMPILER NAMES powerpc64le-linux-gnu-g++-15 powerpc64le-linux-gnu-g++ REQUIRED)

find_program(usearch_qemu_ppc64le_ qemu-ppc64le)
if (usearch_qemu_ppc64le_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${usearch_qemu_ppc64le_};-L;/usr/powerpc64le-linux-gnu;-cpu;power10")
endif ()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
