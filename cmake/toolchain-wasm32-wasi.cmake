# WebAssembly via the WASI SDK - the `wasm32_v128_k` CPU class for standalone runtimes and the Rust crate.
#
# ~~~
#   export WASI_SDK_PATH=/opt/wasi-sdk
#   cmake -B build_wasm32_wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi.cmake -DUSEARCH_BUILD_LIB_C=1
#   cmake --build build_wasm32_wasi --target usearch_static
# ~~~
#
# `wasm32-wasip1`, one thread, the fixed-width `simd128` tier, self-contained memory: the same module runs under
# Wasmtime, Wasmer and Node. The SDK's own file names the compilers and the sysroot; this one adds the tier and the
# runtime `ctest` drives. Exceptions stay off, which is the `noeh` multilib `build.rs` links against.
if (NOT DEFINED WASI_SDK_PATH)
    if (DEFINED ENV{WASI_SDK_PATH})
        set(WASI_SDK_PATH "$ENV{WASI_SDK_PATH}")
    else ()
        message(FATAL_ERROR "WASI_SDK_PATH is unset; point it at an unpacked wasi-sdk release before configuring.")
    endif ()
endif ()

# Nested try-compile projects reread this file and inherit the environment, but not this cache variable.
set(ENV{WASI_SDK_PATH} "${WASI_SDK_PATH}")

set(WASI_SDK_PREFIX "${WASI_SDK_PATH}")
include("${WASI_SDK_PATH}/share/cmake/wasi-sdk-p1.cmake")

# The C tests install crash handlers through `<signal.h>`, which WASI provides only as an emulation library.
set(CMAKE_C_FLAGS_INIT "-msimd128 -D_WASI_EMULATED_SIGNAL")
set(CMAKE_CXX_FLAGS_INIT "-msimd128 -fno-exceptions -D_WASI_EMULATED_SIGNAL")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-lwasi-emulated-signal")

find_program(USEARCH_WASMTIME_ wasmtime PATHS "$ENV{HOME}/.wasmtime/bin")
if (USEARCH_WASMTIME_)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${USEARCH_WASMTIME_};run;-S;inherit-env=y;--dir;.")
endif ()
