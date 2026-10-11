# WebAssembly via Emscripten - the `wasm64_extended_k` CPU class.
#
# ~~~
#   source /path/to/emsdk/emsdk_env.sh
#   emcmake cmake -B build_wasm64_emscripten_release -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm64-emscripten.cmake -DUSEARCH_BUILD_LIB_C=1
#   cmake --build build_wasm64_emscripten_release --target usearch_static
# ~~~
#
# Relaxed SIMD, shared memory and 64-bit addressing are whole-module choices, so one file fixes all three: the pools
# spawn Web Workers, and the memory passes 4 GB. A page loads it only when cross-origin isolated; Node runs it from
# 24.21, newer than emsdk's bundled 24.19. `emcmake` recognises the toolchain file only as the single argument
# `-DCMAKE_TOOLCHAIN_FILE=…`; written with a space it appends its own file, which then wins. Emscripten's own file reads
# the width off `CMAKE_C_FLAGS` as it loads, so the flags precede the include.
if (NOT DEFINED ENV{EMSDK})
    message(FATAL_ERROR "EMSDK is unset; source the SDK's `emsdk_env.sh` before configuring.")
endif ()

set(CMAKE_C_FLAGS
    "-pthread -sMEMORY64 -msimd128 -mrelaxed-simd"
    CACHE STRING "Flags used by the C compiler during all build types."
)
set(CMAKE_CXX_FLAGS
    "-pthread -sMEMORY64 -msimd128 -mrelaxed-simd"
    CACHE STRING "Flags used by the CXX compiler during all build types."
)
set(EMSCRIPTEN_SYSTEM_PROCESSOR wasm64)
include("$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")

set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-pthread -sMEMORY64 -sPROXY_TO_PTHREAD -sALLOW_MEMORY_GROWTH -sEXIT_RUNTIME=1 -sNODE_HOST_ENV"
)
