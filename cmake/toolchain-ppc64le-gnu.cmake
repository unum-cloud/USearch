# PowerPC 64 little-endian Linux, cross-compiled with the GNU toolchain.
#
# ~~~
#   sudo apt install crossbuild-essential-ppc64el qemu-user
#   cmake -B build_ppc64le_gnu -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-ppc64le-gnu.cmake -D USEARCH_BUILD_LIB_C=1
#   cmake --build build_ppc64le_gnu --target usearch_shared
# ~~~
#
# The ISA probes settle on VSX here, which is what a Power CPU class would build on.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)

find_program(CMAKE_C_COMPILER powerpc64le-linux-gnu-gcc REQUIRED)
find_program(CMAKE_CXX_COMPILER powerpc64le-linux-gnu-g++ REQUIRED)

find_program(USEARCH_QEMU_PPC64LE_ qemu-ppc64le)
if (USEARCH_QEMU_PPC64LE_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${USEARCH_QEMU_PPC64LE_};-L;/usr/powerpc64le-linux-gnu")
endif ()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
