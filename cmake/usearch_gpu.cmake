# The GPU host compiler, the CUDA and ROCm codes, and the toolkit and assembler GPU units link.

# Clang keeps a unit's host and device halves apart, through the host-only passes of its CUDA and HIP modes, where other
# host compilers trip over mixed units.
if ((USEARCH_CUDA_ARCH
     OR USEARCH_ROCM_ARCH
     OR USEARCH_METAL_ARCH)
    AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"
)
    message(
        FATAL_ERROR
            "Every GPU build compiles its host code with Clang, found ${CMAKE_CXX_COMPILER_ID} - set CMAKE_CXX_COMPILER to clang++, and to ROCm's own clang++ for USEARCH_ROCM_ARCH"
    )
endif ()

# The library compiles one unit per GPU class, each for its own codes alone, and only the classes a resolved code names.
# `all` is the release set, of which each class compiles the lowest code its kernels need, never below CUDA 13's Turing
# floor: PTX at a plain code where one suffices, so it runs on every later GPU, and the `100f` cubin where datacenter
# Blackwell's `tcgen05` needs a family code. `native` maps each GPU `nvidia-smi` lists to the code its class needs,
# family suffix included, where a bare compute capability would drop it, and a native or named code compiles as SASS
# alone. The CUDA tests and the dispatch unit read the same codes, and a NumKong GPU capability turns on in the tests
# only when every code runs it.
set(usearch_cuda_architectures_)
if (USEARCH_CUDA_ARCH STREQUAL "all")
    set(usearch_cuda_architectures_ 80-virtual 89-virtual 90-virtual 100f-real 120-virtual)
elseif (USEARCH_CUDA_ARCH STREQUAL "native")
    set(usearch_cuda_native_codes_
        75-real=75-real
        80-real=80-real
        86-real=86-real
        89-real=89-real
        90-real=90a-real
        100-real=100f-real
        103-real=103f-real
        110-real=110f-real
        120-real=120f-real
        121-real=121f-real
    )
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE usearch_cuda_visible_
        ERROR_QUIET
    )
    separate_arguments(usearch_cuda_visible_ UNIX_COMMAND "${usearch_cuda_visible_}")
    if (NOT usearch_cuda_visible_)
        message(FATAL_ERROR "USEARCH_CUDA_ARCH=native found no GPU through nvidia-smi; name its code instead")
    endif ()
    foreach (visible_ IN LISTS usearch_cuda_visible_)
        set(native_code_)
        if (visible_ MATCHES "^([0-9]+)\\.([0-9])$")
            set(visible_ "${CMAKE_MATCH_1}${CMAKE_MATCH_2}-real")
            foreach (pair_ IN LISTS usearch_cuda_native_codes_)
                if (pair_ MATCHES "^${visible_}=(.+)$")
                    set(native_code_ ${CMAKE_MATCH_1})
                endif ()
            endforeach ()
        endif ()
        if (NOT native_code_)
            message(
                FATAL_ERROR
                    "USEARCH_CUDA_ARCH=native has no code for the visible ${visible_} GPU; name its code instead"
            )
        endif ()
        if (NOT native_code_ IN_LIST usearch_cuda_architectures_)
            list(APPEND usearch_cuda_architectures_ ${native_code_})
        endif ()
    endforeach ()
elseif (USEARCH_CUDA_ARCH MATCHES "^[0-9]+[af]?$")
    set(usearch_cuda_architectures_ ${USEARCH_CUDA_ARCH}-real)
elseif (USEARCH_CUDA_ARCH)
    message(FATAL_ERROR "USEARCH_CUDA_ARCH takes all, native or one code like 100f, not `${USEARCH_CUDA_ARCH}`")
endif ()

# Three-digit "a" & "f" entries are only understood from CMake 3.31 onward, while the CPU-only build stays on the 3.28
# floor that stock Ubuntu 24.04 ships. The floor follows the codes actually asked for, so a single-code build on a
# distribution CMake is never refused for a capability it does not use.
if (CMAKE_VERSION VERSION_LESS 3.31 AND usearch_cuda_architectures_ MATCHES "1[0-9][0-9][af]")
    message(
        FATAL_ERROR
            "USEARCH_CUDA_ARCH resolves to \"${usearch_cuda_architectures_}\", with a three-digit \"a\" or \"f\" code, which needs CMake 3.31 or newer, found ${CMAKE_VERSION} - name a plain code instead"
    )
endif ()

# The AMD units embed one code object per resolved processor of their class, and the CDNA3 floor also for CDNA4, which
# keeps every instruction its kits use; CDNA5 drops `v_dot4_i32_i8` and `v_dot2_f32_f16`, so only its own unit runs it.
# `all` is gfx942 for CDNA3, gfx950 for CDNA4 and gfx1250 and gfx1251 for CDNA5, and `native` the processors
# `rocm_agent_enumerator` lists. A class with no processor resolved does not exist.
set(usearch_rocm_architectures_)
if (USEARCH_ROCM_ARCH STREQUAL "all")
    set(usearch_rocm_architectures_ gfx942 gfx950 gfx1250 gfx1251)
elseif (USEARCH_ROCM_ARCH STREQUAL "native")
    execute_process(
        COMMAND rocm_agent_enumerator
        OUTPUT_VARIABLE usearch_rocm_visible_
        ERROR_QUIET
    )
    separate_arguments(usearch_rocm_visible_ UNIX_COMMAND "${usearch_rocm_visible_}")
    foreach (visible_ IN LISTS usearch_rocm_visible_)
        if (visible_ MATCHES "^gfx[1-9]" AND NOT visible_ IN_LIST usearch_rocm_architectures_)
            list(APPEND usearch_rocm_architectures_ ${visible_})
        endif ()
    endforeach ()
    if (NOT usearch_rocm_architectures_)
        message(FATAL_ERROR "USEARCH_ROCM_ARCH=native found no GPU through rocm_agent_enumerator; name its processor")
    endif ()
elseif (USEARCH_ROCM_ARCH)
    set(usearch_rocm_architectures_ ${USEARCH_ROCM_ARCH})
endif ()

# Metal compiles its kernels on the device, so a family only names the Apple class it serves.
if (USEARCH_METAL_ARCH STREQUAL "native")
    message(FATAL_ERROR "USEARCH_METAL_ARCH takes all or one family like apple9, as Metal compiles on the device")
endif ()

# NVCC is only a device compiler here, so the toolkit is found for its paths alone, and every NVIDIA and AMD unit links
# its device bundle through an assembler stub. Both serve every directory below, which inherits them.
if (USEARCH_CUDA_ARCH)
    find_package(CUDAToolkit 13 REQUIRED)
endif ()
if (USEARCH_CUDA_ARCH OR USEARCH_ROCM_ARCH)
    enable_language(ASM)
endif ()
