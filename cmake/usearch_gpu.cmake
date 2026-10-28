# The GPU host compiler, the CUDA and ROCm architectures, and the toolkit and assembler GPU units link.

# Clang keeps a unit's host and device halves apart, through the host-only passes of its CUDA and HIP modes, where other
# host compilers trip over mixed units.
if ((USEARCH_BUILD_CUDA
     OR USEARCH_BUILD_ROCM
     OR USEARCH_BUILD_METAL)
    AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"
)
    message(
        FATAL_ERROR
            "Every GPU build compiles its host code with Clang, found ${CMAKE_CXX_COMPILER_ID} - set CMAKE_CXX_COMPILER to clang++, and to ROCm's own clang++ for USEARCH_BUILD_ROCM"
    )
endif ()

# The library compiles one unit per GPU class, each for its own architecture alone - "80" with its PTX for every GPU
# without a class, "89", "90a", "100f" and "120f" - and only the classes this list names, with or without "-real", so
# "100f-real" builds the Blackwell kernels alone. The CUDA tests, benchmarks and the dispatch unit compile one pass per
# entry, and a NumKong GPU capability turns on in them only when every entry runs it, so iterate on one GPU with its
# family code, like "100f-real", to test that generation's kits. The "-real" entries skip the PTX, which would triple
# the objects of builds that never meet an unknown GPU. The kernels start at Ampere, and Turing's "75" would turn every
# NumKong GPU capability off in the default tests.
set(USEARCH_CUDA_ARCHITECTURES
    "80-real;89-real;90a-real;100f-real;120f-real;80-virtual"
    CACHE STRING
          "GPU architectures the CUDA tests, benchmarks & dispatch unit compile for, or native for the visible GPUs"
)

# "native" maps each GPU `nvidia-smi` lists to the code its class needs, family suffix included, where a bare compute
# capability would drop it.
set(usearch_cuda_architectures_)
if (USEARCH_BUILD_CUDA)
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
    set(usearch_cuda_visible_)
    if ("native" IN_LIST USEARCH_CUDA_ARCHITECTURES)
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE usearch_cuda_visible_
            ERROR_QUIET
        )
        separate_arguments(usearch_cuda_visible_ UNIX_COMMAND "${usearch_cuda_visible_}")
        if (NOT usearch_cuda_visible_)
            message(
                FATAL_ERROR "USEARCH_CUDA_ARCHITECTURES=native found no GPU through nvidia-smi; name its code instead"
            )
        endif ()
    endif ()
    foreach (entry_ IN LISTS USEARCH_CUDA_ARCHITECTURES)
        if (NOT entry_ STREQUAL "native")
            list(APPEND usearch_cuda_architectures_ ${entry_})
        endif ()
    endforeach ()
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
            message(FATAL_ERROR "USEARCH_CUDA_ARCHITECTURES=native has no code for the visible ${visible_} GPU; "
                                "name its code instead"
            )
        endif ()
        if (NOT native_code_ IN_LIST usearch_cuda_architectures_)
            list(APPEND usearch_cuda_architectures_ ${native_code_})
        endif ()
    endforeach ()
endif ()

# Three-digit "a" & "f" entries are only understood from CMake 3.31 onward, while the CPU-only build stays on the 3.28
# floor that stock Ubuntu 24.04 ships. The floor follows the architectures actually asked for, so a single-architecture
# build on a distribution CMake is never refused for a capability it does not use.
if (CMAKE_VERSION VERSION_LESS 3.31 AND usearch_cuda_architectures_ MATCHES "1[0-9][0-9][af]")
    message(
        FATAL_ERROR
            "USEARCH_CUDA_ARCHITECTURES \"${usearch_cuda_architectures_}\" names a three-digit \"a\" or \"f\" architecture, which needs CMake 3.31 or newer, found ${CMAKE_VERSION} - override with a single architecture"
    )
endif ()

# The AMD units embed one code object per listed processor of their class, and the CDNA3 floor also for CDNA4, which
# keeps every instruction its kits use; CDNA5 drops `v_dot4_i32_i8` and `v_dot2_f32_f16`, so only its own unit runs it.
# A class with no processor listed embeds none, and its devices refuse to load.
set(USEARCH_ROCM_ARCHITECTURES
    "gfx942;gfx950;gfx1250;gfx1251"
    CACHE STRING "AMD processors the code objects compile for: gfx942 for CDNA3, gfx950 for CDNA4, gfx125x for CDNA5"
)

# NVCC is only a device compiler here, so the toolkit is found for its paths alone, and every NVIDIA and AMD unit links
# its device bundle through an assembler stub. Both serve every directory below, which inherits them.
if (USEARCH_BUILD_CUDA)
    find_package(CUDAToolkit 13 REQUIRED)
endif ()
if (USEARCH_BUILD_CUDA OR USEARCH_BUILD_ROCM)
    enable_language(ASM)
endif ()
