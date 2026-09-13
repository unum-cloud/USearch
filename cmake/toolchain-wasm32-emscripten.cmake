# WebAssembly via Emscripten - the `wasm32_v128_k` CPU class.
#
# ~~~
#   source /path/to/emsdk/emsdk_env.sh
#   emcmake cmake -B build_wasm32_emscripten -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-emscripten.cmake -DUSEARCH_BUILD_LIB_C=1
#   cmake --build build_wasm32_emscripten --target usearch_static
# ~~~
#
# The module every engine loads: 32-bit, one thread, the fixed-width `simd128` tier, so it runs in any page - served
# with cross-origin isolation or not - and on WebKit, which ships no relaxed SIMD. `emcmake` recognises the toolchain
# file only as the single argument `-DCMAKE_TOOLCHAIN_FILE=…`; written with a space it appends its own file, which then
# wins. Emscripten's own file reads the width off `CMAKE_C_FLAGS` as it loads, so the flags precede the include.
if (NOT DEFINED ENV{EMSDK})
    message(FATAL_ERROR "EMSDK is unset; source the SDK's `emsdk_env.sh` before configuring.")
endif ()

set(CMAKE_C_FLAGS
    "-msimd128"
    CACHE STRING "Flags used by the C compiler during all build types."
)
set(CMAKE_CXX_FLAGS
    "-msimd128"
    CACHE STRING "Flags used by the CXX compiler during all build types."
)
set(EMSCRIPTEN_SYSTEM_PROCESSOR wasm32)
include("$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-sALLOW_MEMORY_GROWTH")
