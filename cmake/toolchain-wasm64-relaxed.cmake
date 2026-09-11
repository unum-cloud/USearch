# WebAssembly via Emscripten - the `wasm64_relaxed_k` CPU class.
#
# ~~~
#   source /path/to/emsdk/emsdk_env.sh
#   emcmake cmake -B build_wasm64_relaxed -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm64-relaxed.cmake -D USEARCH_BUILD_LIB_C=1
#   cmake --build build_wasm64_relaxed --target usearch_shared
# ~~~
#
# Shared memory so the parks are real futexes, relaxed SIMD, and the 64-bit addressing that alone carries one module
# past 4 GiB. `emcmake` already points CMake at Emscripten's toolchain, so this chains to it.
if (NOT DEFINED ENV{EMSDK})
    message(FATAL_ERROR "EMSDK is unset; source the SDK's `emsdk_env.sh` before configuring.")
endif ()
include("$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")

set(CMAKE_C_FLAGS_INIT "-pthread -msimd128 -mrelaxed-simd")
set(CMAKE_CXX_FLAGS_INIT "-pthread -msimd128 -mrelaxed-simd")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-pthread -sMEMORY64 -sALLOW_MEMORY_GROWTH")

find_program(USEARCH_NODE_ node)
if (USEARCH_NODE_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${USEARCH_NODE_}")
endif ()
