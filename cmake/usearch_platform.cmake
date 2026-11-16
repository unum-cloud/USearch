# The CPU USearch's own units are tuned for, or a WebAssembly module's one SIMD capability.
#
# The root sets these before including this file, after `usearch_flags.cmake` and the siblings:
#
# - `usearch_native_probe_library_`: the sibling whose probes MSVC's `native` reads, like `stringzilla`;
# - `usearch_native_probe_capabilities_`: its capabilities, widest first, like `icelake skylake haswell neon`;
# - `usearch_arch_per_target_`: ON where each target applies `usearch_target_arch_flags_` itself.
#
# Without the last one, every unit linking `usearch_warnings` takes the floor.
# The file leaves `usearch_target_arch_flags_`, the floor, and `usearch_host_tuning_` for tests and benchmarks.

if (USEARCH_MARCH STREQUAL "native" AND CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "USEARCH_MARCH=native needs the host as the target; name a CPU instead")
endif ()

# USEARCH_MARCH sets the floor of USearch's own C and C++ units, which class units raise with their own flags.
# Empty keeps the architecture's floor, whatever a distribution compiler defaults to.
# An ISA level like `x86-64-v3` raises it, `native` tunes for this host, and a CPU name like `graniterapids` tunes for
# that CPU, also when cross-compiling.
# AArch64 names CPUs through `-mcpu`, as `-march` takes only architecture versions.
# Power compilers have no `-march` at all.
set(usearch_target_arch_flags_ "")
if (CMAKE_SYSTEM_NAME MATCHES "^(Emscripten|WASI)$")
    # A WebAssembly module starts from its toolchain file's flags, which `v128` and `v128relaxed` extend.
    if (USEARCH_MARCH STREQUAL "v128relaxed")
        set(usearch_target_arch_flags_ -msimd128 -mrelaxed-simd)
    elseif (USEARCH_MARCH STREQUAL "v128")
        set(usearch_target_arch_flags_ -msimd128)
    elseif (USEARCH_MARCH AND NOT USEARCH_MARCH STREQUAL "serial")
        message(FATAL_ERROR "USEARCH_MARCH on WebAssembly: serial, v128 or v128relaxed, not `${USEARCH_MARCH}`")
    endif ()
elseif (MSVC AND USEARCH_MARCH STREQUAL "native")
    # MSVC has no `/arch:native`, so it takes the `/arch:` of the widest capability this host runs.
    # The probing sibling caches its findings, so the flags are read from the cache by name.
    foreach (capability_ IN LISTS usearch_native_probe_capabilities_)
        if (${usearch_native_probe_library_}_target_${capability_}_runs AND NOT usearch_target_arch_flags_)
            get_property(
                usearch_target_arch_flags_
                CACHE ${usearch_native_probe_library_}_target_${capability_}_flags
                PROPERTY VALUE
            )
        endif ()
    endforeach ()
elseif (MSVC AND USEARCH_MARCH)
    message(FATAL_ERROR "MSVC takes USEARCH_MARCH=native only; pass an /arch: value such as AVX2, AVX512 or AVX10.2 "
                        "through CMAKE_CXX_FLAGS instead of `${USEARCH_MARCH}`"
    )
elseif (USEARCH_MARCH)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|AARCH64|arm64|ARM64|ppc64|ppc64le|powerpc64|powerpc64le)$")
        set(usearch_target_arch_flags_ -mcpu=${USEARCH_MARCH})
    else ()
        set(usearch_target_arch_flags_ -march=${USEARCH_MARCH})
    endif ()
elseif (MSVC OR APPLE)
    # MSVC's defaults are the floor, and on Apple `-arch` pins it per slice of a universal build.
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|AARCH64|arm64|ARM64)$")
    set(usearch_target_arch_flags_ -march=armv8-a)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(usearch_target_arch_flags_ -march=x86-64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64$")
    set(usearch_target_arch_flags_ -march=rv64gc)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64|ppc64le|powerpc64|powerpc64le)$")
    # POWER8 is the floor of the little-endian ELFv2 ABI.
    set(usearch_target_arch_flags_ -mcpu=power8)
endif ()
if (NOT usearch_arch_per_target_)
    target_compile_options(usearch_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:${usearch_target_arch_flags_}>")
endif ()

# Host tuning is for the tests and benchmarks that opt in, never the distributable libraries.
# It applies only while USEARCH_MARCH is empty, the host is the target and the compiler has a spelling for it; elsewhere
# the suites run on the floor and the SIMD variants report themselves skipped.
# Apple Clang's `-march=native` advertises only a subset of the host's features, while `-mcpu=native` is the complete
# AArch64 knob.
set(usearch_host_tuning_ "")
if (USEARCH_MARCH
    OR MSVC
    OR CMAKE_CROSSCOMPILING
)
    # The floor already says what the suites run on.
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|AARCH64|arm64|ARM64)$")
    set(usearch_host_tuning_ -mcpu=native)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(usearch_host_tuning_ -march=native)
endif ()
