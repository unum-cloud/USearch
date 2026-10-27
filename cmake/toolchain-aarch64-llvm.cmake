# AArch64 Linux, cross-compiled with whatever Clang and LLD the host carries.
#
# ~~~
#   sudo apt install clang lld crossbuild-essential-arm64
#   cmake -B build_aarch64_llvm -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-llvm.cmake
#   cmake --build build_aarch64_llvm --target usearch_shared
# ~~~
#
# The archiver, the ranlib and the linker are left to CMake and to the driver, so a distribution shipping `llvm-ar` only
# as `llvm-ar-21` still works and CI pins no version.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

find_program(CMAKE_C_COMPILER clang REQUIRED)
find_program(CMAKE_CXX_COMPILER clang++ REQUIRED)

set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)

# LLD rather than `ld.bfd`. NumKong's SME kernels carry `__arm_new("za")`, whose lazy-save calls `__arm_tpidr2_save` and
# `__arm_tpidr2_restore` are defined only in GCC's `libgcc_eh.a` - and BFD resolves that archive before LTO materializes
# the references, dropping the members that answer them. LLD resolves LTO symbols first, so the archive is still on the
# table when they appear.
#
# The routines themselves are `libgcc_eh.a`'s, and NumKong compiles `-fvisibility=hidden`, so the references are hidden
# and only a static definition can answer them - `-u` forces the archive members in rather than letting either linker
# decide it has no reason to look.
set(usearch_aarch64_link_flags_ "-fuse-ld=lld -Wl,-u,__arm_tpidr2_save -Wl,-u,__arm_tpidr2_restore -lgcc_eh")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${usearch_aarch64_link_flags_}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${usearch_aarch64_link_flags_}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${usearch_aarch64_link_flags_}")

find_program(usearch_qemu_aarch64_ qemu-aarch64)
if (usearch_qemu_aarch64_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${usearch_qemu_aarch64_};-L;/usr/aarch64-linux-gnu;-cpu;max")
endif ()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
