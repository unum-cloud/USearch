# Target checks, the floor of USearch's own units, and the host tuning of the tests and benchmarks.

if (USEARCH_TARGET_ARCH STREQUAL "native" AND CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "USEARCH_TARGET_ARCH=native needs the host as the target; name a CPU instead")
endif ()
# Only a native build asks the siblings which capabilities this host runs.
if (USEARCH_TARGET_CLASSES STREQUAL "native" AND NOT USEARCH_TARGET_ARCH STREQUAL "native")
    message(FATAL_ERROR "USEARCH_TARGET_CLASSES=native needs USEARCH_TARGET_ARCH=native")
endif ()

# USEARCH_TARGET_ARCH sets the floor of USearch's own C and C++ units, and each class unit adds its class's flags over
# it: empty keeps the architecture's floor, whatever a distribution compiler defaults to, `native` tunes for this host,
# and a CPU name like `graniterapids` for that CPU, also when cross-compiling. AArch64 names CPUs through `-mcpu`, as
# `-march` takes only architecture versions.
set(usearch_target_arch_flags_)
if (CMAKE_SYSTEM_NAME MATCHES "Emscripten|WASI")
    # A WebAssembly module's one SIMD capability comes from its toolchain file.
elseif (MSVC AND USEARCH_TARGET_ARCH STREQUAL "native")
    # MSVC has no `/arch:native`, so it takes the `/arch:` of the widest capability this host runs.
    foreach (capability_ IN ITEMS diamond skylake haswell neonbfdot neonsdot neonhalf)
        if (numkong_target_${capability_}_runs AND NOT usearch_target_arch_flags_)
            set(usearch_target_arch_flags_ ${numkong_target_${capability_}_flags})
        endif ()
    endforeach ()
elseif (MSVC AND USEARCH_TARGET_ARCH)
    message(
        FATAL_ERROR
            "MSVC takes USEARCH_TARGET_ARCH=native only; pass an /arch: value such as AVX2, AVX512 or AVX10.2 through "
            "CMAKE_CXX_FLAGS instead of `${USEARCH_TARGET_ARCH}`"
    )
elseif (USEARCH_TARGET_ARCH AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(usearch_target_arch_flags_ -mcpu=${USEARCH_TARGET_ARCH})
elseif (USEARCH_TARGET_ARCH)
    set(usearch_target_arch_flags_ -march=${USEARCH_TARGET_ARCH})
elseif (MSVC OR APPLE)
    # MSVC's defaults are the floor, and on Apple `-arch` pins it per slice of a universal build.
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(usearch_target_arch_flags_ -march=armv8-a)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(usearch_target_arch_flags_ -march=x86-64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set(usearch_target_arch_flags_ -march=rv64gc)
endif ()

# Host tuning for the tests & benches only - never the distributable libraries - and only while USEARCH_TARGET_ARCH is
# empty, the host is the target and the compiler has a spelling for it; elsewhere the suite runs on the switch's floor
# and the SIMD variants report themselves skipped. Apple Clang's `-march=native` advertises only a subset of the host's
# features - no FP16 arithmetic & FML, no SME - leaving NumKong's f16 kits serial; `-mcpu=native` is the complete
# AArch64 knob.
if (USEARCH_TARGET_ARCH
    OR MSVC
    OR CMAKE_CROSSCOMPILING
)
    set(usearch_host_tuning_ "")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(usearch_host_tuning_ -mcpu=native)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(usearch_host_tuning_ -march=native)
else ()
    set(usearch_host_tuning_ "")
endif ()
