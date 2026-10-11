# The RAM arms compile per CPU class. An arm inlines its class's kits and runs on nothing older, so its unit takes the
# class's instruction set outright and the compiler may use it anywhere inside. A class is its parent plus a delta - the
# kits and the ForkUnion bits `parent_class_of`, `numkong_capabilities_of`, `waiter_for_cpu_class` and
# `atomic_ref_for_cpu_class` add over the parent - restated here row for row. A unit inherits its parent's flags, adds
# the flags of each delta kit that compiled, and exists only when the parent does and some delta entry compiled: the
# rung oracle at unit granularity, and the same gates the arm lists carry.
add_library(usearch_dense_ram_variants OBJECT dense/ram_variants_serial.cpp)
usearch_setup_target(usearch_dense_ram_variants)
target_include_directories(usearch_dense_ram_variants PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(usearch_dense_ram_variants PRIVATE "USEARCH_BUILD_CUDA=0")
target_link_libraries(usearch_dense_ram_variants PRIVATE yyjson forkunion::static)
set(usearch_variant_serial_exists_ TRUE)
set(usearch_variant_serial_flags_)

# Unions the kits' flags: x86 `-m<feature>` flags collect, an Arm or RISC-V `-march=` string merges to the newest base
# with every extension once, and MSVC's `/arch:` keeps the last, which the lineage lists highest.
function (usearch_union_isa_flags out_)
    set(collected_)
    set(arm_base_)
    set(arm_extensions_)
    set(riscv_base_)
    set(riscv_extensions_)
    set(msvc_arch_)
    foreach (flag_ IN LISTS ARGN)
        if (flag_ MATCHES "^-march=(armv[0-9.]+-a)(.*)$")
            if (NOT arm_base_ OR CMAKE_MATCH_1 STRGREATER arm_base_)
                set(arm_base_ ${CMAKE_MATCH_1})
            endif ()
            string(REPLACE "+" ";" parts_ "${CMAKE_MATCH_2}")
            list(APPEND arm_extensions_ ${parts_})
        elseif (flag_ MATCHES "^-march=(rv[0-9]+[a-z]+)(.*)$")
            if (NOT riscv_base_ OR CMAKE_MATCH_1 STRGREATER riscv_base_)
                set(riscv_base_ ${CMAKE_MATCH_1})
            endif ()
            string(REPLACE "_" ";" parts_ "${CMAKE_MATCH_2}")
            list(APPEND riscv_extensions_ ${parts_})
        elseif (flag_ MATCHES "^/arch:")
            set(msvc_arch_ ${flag_})
        else ()
            list(APPEND collected_ ${flag_})
        endif ()
    endforeach ()
    if (arm_base_)
        list(REMOVE_DUPLICATES arm_extensions_)
        list(JOIN arm_extensions_ "+" joined_)
        list(APPEND collected_ "-march=${arm_base_}+${joined_}")
    endif ()
    if (riscv_base_)
        list(REMOVE_DUPLICATES riscv_extensions_)
        list(JOIN riscv_extensions_ "_" joined_)
        list(APPEND collected_ "-march=${riscv_base_}_${joined_}")
    endif ()
    list(APPEND collected_ ${msvc_arch_})
    list(REMOVE_DUPLICATES collected_)
    set(${out_}
        "${collected_}"
        PARENT_SCOPE
    )
endfunction ()

function (usearch_dense_ram_variant class)
    cmake_parse_arguments(arg "" "OVER" "" ${ARGN})
    if (NOT usearch_variant_${arg_OVER}_exists_)
        return()
    endif ()
    set(flags_ ${usearch_variant_${arg_OVER}_flags_})
    set(advances_ FALSE)
    foreach (entry_ IN LISTS arg_UNPARSED_ARGUMENTS)
        if (nk_target_${entry_}_compiles)
            set(advances_ TRUE)
            list(APPEND flags_ ${nk_target_${entry_}_flags})
        elseif (fu_target_${entry_}_compiles)
            set(advances_ TRUE)
        endif ()
    endforeach ()
    set(usearch_variant_${class}_exists_
        ${advances_}
        PARENT_SCOPE
    )
    if (NOT advances_)
        return()
    endif ()
    usearch_union_isa_flags(flags_ ${flags_})
    set(usearch_variant_${class}_flags_
        "${flags_}"
        PARENT_SCOPE
    )
    target_sources(usearch_dense_ram_variants PRIVATE dense/ram_variants_${class}.cpp)
    set_source_files_properties(dense/ram_variants_${class}.cpp PROPERTIES COMPILE_OPTIONS "${flags_}")
endfunction ()

usearch_dense_ram_variant(intel_haswell OVER serial haswell)
usearch_dense_ram_variant(
    intel_sierra
    OVER
    intel_haswell
    alder
    sierra
    x86_tpause
    x86_cmpccxadd
)
usearch_dense_ram_variant(intel_skylake OVER intel_haswell skylake)
usearch_dense_ram_variant(intel_icelake OVER intel_skylake icelake)
usearch_dense_ram_variant(amd_genoa OVER intel_icelake genoa)
usearch_dense_ram_variant(intel_sapphire OVER amd_genoa sapphire sapphireamx)
usearch_dense_ram_variant(intel_granite OVER intel_sapphire graniteamx)
usearch_dense_ram_variant(
    intel_diamond
    OVER
    intel_granite
    alder
    sierra
    diamond
    x86_tpause
    x86_cmpccxadd
)
usearch_dense_ram_variant(aws_g2 OVER serial neon)
usearch_dense_ram_variant(
    apple_m2
    OVER
    aws_g2
    neonsdot
    neonbfdot
    neonfhm
    arm64_lse
    arm64_rcpc
)
usearch_dense_ram_variant(apple_m4 OVER apple_m2 sme smebi32)
usearch_dense_ram_variant(aws_g5 OVER apple_m2 arm64_wfet)
usearch_dense_ram_variant(nvidia_vera OVER aws_g5 neonfp8)
usearch_dense_ram_variant(
    alibaba_c930
    OVER
    serial
    rvv
    rvvbb
    risc5_wrs
    risc5_zacas
)
# A WebAssembly module carries the one SIMD capability its toolchain declares, so its verdicts read the compiler's own
# macros rather than a probe that would pass the capability's flags itself.
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR CMAKE_SYSTEM_NAME STREQUAL "WASI")
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles(
        "#if !defined(__wasm_simd128__)\n#error\n#endif\nint main() { return 0; }" nk_target_v128_compiles
    )
    check_cxx_source_compiles(
        "#if !defined(__wasm_relaxed_simd__)\n#error\n#endif\nint main() { return 0; }" nk_target_v128relaxed_compiles
    )
endif ()
usearch_dense_ram_variant(wasm32_v128 OVER serial v128)
usearch_dense_ram_variant(wasm64_extended OVER wasm32_v128 v128relaxed)

# The type-erased implementations behind the C interface: the dispatch unit visits the arms through their explicitly
# instantiated views and compiles at the baseline, since it inlines no kernel. Under CUDA the dispatch unit is the `.cu`
# twin and the device arms join, so the C libraries carry them too.
add_library(usearch_punned OBJECT types.cpp sequence.cpp)
usearch_setup_target(usearch_punned)
target_include_directories(usearch_punned PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usearch_punned PRIVATE yyjson forkunion::static stringzilla::header)
set(USEARCH_PUNNED_LIBRARIES usearch_punned usearch_dense_ram_variants)

if (USEARCH_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # One unit per GPU class, compiled for its own architecture alone - the device twin of the CPU classes' per-unit ISA
    # flags - so no cubin carries another generation's kernels. Every unit sees the same NumKong GPU capabilities, as
    # the CPU units see the same probe verdicts, so the arms' types agree across units, and each class's ceiling keeps
    # its arms within its generation. The codes reach NVCC directly, since CMake spells family-specific ones only from
    # 3.31.
    set(usearch_cuda_capabilities_ NUMKONG_TARGET_AMPERE=1 NUMKONG_TARGET_ADA=1 NUMKONG_TARGET_HOPPER=1
                                   NUMKONG_TARGET_BLACKWELL=1 NUMKONG_TARGET_BLACKWELLRTX=1
    )
    function (usearch_dense_vram_variant class)
        set(target_ usearch_dense_vram_variants_${class})
        add_library(${target_} OBJECT dense/vram_variants_${class}.cu)
        target_include_directories(
            ${target_} PRIVATE $<TARGET_PROPERTY:${USEARCH_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
                               ${CMAKE_CURRENT_SOURCE_DIR}
        )
        # Skips `usearch_setup_target`, whose host-compiler flags the CUDA driver rejects, so it links NumKong's and
        # ForkUnion's compilable capabilities itself.
        target_compile_definitions(${target_} PRIVATE "USEARCH_BUILD_CUDA=1" ${usearch_cuda_capabilities_})
        set_target_properties(
            ${target_}
            PROPERTIES CXX_STANDARD 20
                       CUDA_STANDARD 20
                       CUDA_ARCHITECTURES OFF
                       POSITION_INDEPENDENT_CODE ON
        )
        target_compile_options(${target_} PRIVATE --expt-relaxed-constexpr ${ARGN})
        target_link_libraries(
            ${target_}
            PRIVATE yyjson
                    CUDA::cuda_driver
                    forkunion::static
                    numkong::header
                    numkong::cpu_capabilities_compiled
                    stringzilla::cpu_capabilities_compiled
                    forkunion::cpu_capabilities_compiled
        )
        set(USEARCH_PUNNED_LIBRARIES
            ${USEARCH_PUNNED_LIBRARIES} ${target_}
            PARENT_SCOPE
        )
    endfunction ()

    usearch_dense_vram_variant(
        nvidia_ampere --generate-code=arch=compute_80,code=sm_80 --generate-code=arch=compute_80,code=compute_80
    )
    usearch_dense_vram_variant(nvidia_ada --generate-code=arch=compute_89,code=sm_89)
    usearch_dense_vram_variant(nvidia_hopper --generate-code=arch=compute_90a,code=sm_90a)
    usearch_dense_vram_variant(nvidia_blackwell --generate-code=arch=compute_100f,code=sm_100f)
    usearch_dense_vram_variant(nvidia_blackwellrtx --generate-code=arch=compute_120f,code=sm_120f)

    target_sources(usearch_punned PRIVATE dense_index.cu)
    target_compile_definitions(usearch_punned PRIVATE "USEARCH_BUILD_CUDA=1" ${usearch_cuda_capabilities_})
    set_target_properties(
        usearch_punned
        PROPERTIES CUDA_STANDARD 20
                   CUDA_ARCHITECTURES "${USEARCH_CUDA_ARCHITECTURES}"
                   POSITION_INDEPENDENT_CODE ON
    )
    # The variant over every arm nests one union per alternative, past the front end's default of 200 in flight.
    target_compile_options(
        usearch_punned PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                               "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --pending_instantiations=1024>"
    )
    target_link_libraries(usearch_punned PRIVATE CUDA::cuda_driver)
else ()
    target_sources(usearch_punned PRIVATE dense_index.cpp)
    target_compile_definitions(usearch_punned PRIVATE "USEARCH_BUILD_CUDA=0")
endif ()

# Objects never travel through another object library, so every consumer names the whole set: executables and shared
# libraries link the targets and inherit their dependencies, an archive takes the objects themselves.
set(USEARCH_PUNNED_OBJECTS)
foreach (library_ IN LISTS USEARCH_PUNNED_LIBRARIES)
    list(APPEND USEARCH_PUNNED_OBJECTS $<TARGET_OBJECTS:${library_}>)
endforeach ()
set(USEARCH_PUNNED_LIBRARIES
    ${USEARCH_PUNNED_LIBRARIES}
    PARENT_SCOPE
)
set(USEARCH_PUNNED_OBJECTS
    ${USEARCH_PUNNED_OBJECTS}
    PARENT_SCOPE
)
