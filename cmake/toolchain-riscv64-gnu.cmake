# RISC-V 64 Linux, cross-compiled with GCC 16, and run under QEMU.
#
# ~~~
#   sudo apt install gcc-16-riscv64-linux-gnu g++-16-riscv64-linux-gnu libc6-dev-riscv64-cross qemu-user
#   cmake --preset linux_riscv64
#   cmake --build --preset linux_riscv64
#   ctest --preset linux_riscv64
# ~~~
#
# GCC 16 is the floor, as GCC 15 crashes compiling StringZilla's RVV kernels and the class above serial goes missing.
# QEMU emulates an RVA23 core with Zacas and 256-bit vectors, which runs the `alibaba_c930` class and the RVA23 builds
# of the distribution's C library.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

find_program(CMAKE_C_COMPILER riscv64-linux-gnu-gcc-16 REQUIRED)
find_program(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++-16 REQUIRED)

find_program(usearch_qemu_riscv64_ qemu-riscv64)
if (usearch_qemu_riscv64_)
    set(CMAKE_CROSSCOMPILING_EMULATOR
        "${usearch_qemu_riscv64_};-L;/usr/riscv64-linux-gnu;-cpu;rva23s64,zacas=true,vlen=256"
    )
endif ()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
