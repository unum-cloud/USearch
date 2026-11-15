# RISC-V 64 Linux, cross-compiled with the GNU toolchain.
#
# ~~~
#   sudo apt install crossbuild-essential-riscv64 qemu-user
#   cmake --preset linux_riscv64
#   cmake --build --preset linux_riscv64 --target usearch_shared
# ~~~
#
# Units compile at the `rv64gc` floor, and the RVV kits pick their extensions per function.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

find_program(CMAKE_C_COMPILER riscv64-linux-gnu-gcc REQUIRED)
find_program(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++ REQUIRED)

find_program(usearch_qemu_riscv64_ qemu-riscv64)
if (usearch_qemu_riscv64_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${usearch_qemu_riscv64_};-L;/usr/riscv64-linux-gnu;-cpu;max")
endif ()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
