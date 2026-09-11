# WebAssembly via Emscripten - the `wasm32_base_k` CPU class.
#
# ~~~
#   source /path/to/emsdk/emsdk_env.sh
#   emcmake cmake -B build_wasm32_base -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-base.cmake -D USEARCH_BUILD_LIB_C=1
#   cmake --build build_wasm32_base --target usearch_shared
# ~~~
#
# The compatibility floor: one thread, no vector extension, so a module runs in any runtime and in any page,
# cross-origin isolated or not. `emcmake` already points CMake at Emscripten's toolchain, so this chains to it.
if (NOT DEFINED ENV{EMSDK})
    message(FATAL_ERROR "EMSDK is unset; source the SDK's `emsdk_env.sh` before configuring.")
endif ()
include("$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")

set(CMAKE_C_FLAGS_INIT "")
set(CMAKE_CXX_FLAGS_INIT "")
set(CMAKE_EXE_LINKER_FLAGS_INIT "")

find_program(USEARCH_NODE_ node)
if (USEARCH_NODE_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${USEARCH_NODE_}")
endif ()
