# AArch64 Linux, cross-compiled with whatever Clang and LLD the host holds, and run under QEMU.
#
# ~~~
#   sudo apt install clang lld crossbuild-essential-arm64 qemu-user
#   cmake --preset linux_arm64
#   cmake --build --preset linux_arm64
#   ctest --preset linux_arm64
# ~~~
#
# QEMU's `-cpu max` holds LSE, RCPC and SVE2, so every Arm CPU class the library compiles runs, `aws_g2` and `aws_g5`
# included.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

find_program(CMAKE_C_COMPILER clang REQUIRED)
find_program(CMAKE_CXX_COMPILER clang++ REQUIRED)

set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)

# NumKong's SME kernels hold `__arm_new("za")`, whose lazy saves call `__arm_tpidr2_save` and `__arm_tpidr2_restore`.
# Only GCC's `libgcc_eh.a` defines them, and the references are hidden, so `-u` pulls those members in statically; LLD
# resolves LTO symbols before the archive, where `ld.bfd` would drop the members first.
set(usearch_aarch64_link_flags_ "-fuse-ld=lld -Wl,-u,__arm_tpidr2_save -Wl,-u,__arm_tpidr2_restore -lgcc_eh")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${usearch_aarch64_link_flags_}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${usearch_aarch64_link_flags_}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${usearch_aarch64_link_flags_}")

find_program(usearch_qemu_aarch64_ qemu-aarch64)
if (usearch_qemu_aarch64_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${usearch_qemu_aarch64_};-L;/usr/aarch64-linux-gnu;-cpu;max")
endif ()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
