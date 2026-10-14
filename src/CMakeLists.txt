# The RAM arms compile per CPU class. An arm inlines its class's kits and runs on nothing older, so its unit takes the
# class's instruction set outright and the compiler may use it anywhere inside. A class is its parent plus a delta - the
# kits and the ForkUnion bits `parent_class_of`, `numkong_capabilities_of`, `waiter_for_cpu_class` and
# `atomic_ref_for_cpu_class` add over the parent - restated here row for row. A unit inherits its parent's flags, adds
# the flags of each delta kit that compiled, and exists only when the parent does and some delta entry compiled: the
# rung oracle at unit granularity, and the same gates the arm lists carry.
add_library(usearch_dense_ram_variants OBJECT dense/ram_variants_serial.cpp)
usearch_setup_target(usearch_dense_ram_variants)
target_include_directories(usearch_dense_ram_variants PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(usearch_dense_ram_variants PRIVATE "USEARCH_ARCH_CUDA_=0")
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

# The NVMe arms compile per CPU class with the RAM unit's flags, for the classes whose 4-way or tiles kits a storage
# walk or scan tells apart: it carries no waiter or atomic reference, so aws_g5 and alibaba_c930, which advance only by
# those, take none, and a WebAssembly module reads no storage directly.
add_library(usearch_dense_nvme_variants OBJECT dense/nvme_variants_serial.cpp)
usearch_setup_target(usearch_dense_nvme_variants)
target_include_directories(usearch_dense_nvme_variants PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(usearch_dense_nvme_variants PRIVATE "USEARCH_ARCH_CUDA_=0")
target_link_libraries(usearch_dense_nvme_variants PRIVATE yyjson forkunion::static)
foreach (
    class_
    intel_haswell
    intel_sierra
    intel_skylake
    intel_icelake
    amd_genoa
    intel_sapphire
    intel_granite
    intel_diamond
    aws_g2
    apple_m2
    apple_m4
    nvidia_vera
)
    if (usearch_variant_${class_}_exists_)
        target_sources(usearch_dense_nvme_variants PRIVATE dense/nvme_variants_${class_}.cpp)
        set_source_files_properties(
            dense/nvme_variants_${class_}.cpp PROPERTIES COMPILE_OPTIONS "${usearch_variant_${class_}_flags_}"
        )
    endif ()
endforeach ()

# The type-erased implementations behind the C interface: the dispatch unit visits the arms through their explicitly
# instantiated views and compiles at the baseline, since it inlines no kernel. Under CUDA the device arms join, so the C
# libraries carry them too.
add_library(usearch_punned OBJECT types.cpp sequence.cpp dense_index.cpp)
usearch_setup_target(usearch_punned)
target_include_directories(usearch_punned PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usearch_punned PRIVATE yyjson forkunion::static stringzilla::header)
set(USEARCH_PUNNED_LIBRARIES usearch_punned usearch_dense_ram_variants usearch_dense_nvme_variants)

# Every NVIDIA or AMD unit with kernels compiles twice. Host-only, through Clang's CUDA or HIP mode, it names its
# kernels from device code alone, so nothing registers with or links a GPU runtime, which opens at first use.
# Device-only, NVCC for NVIDIA and Clang for AMD, it compiles into a bundle of the codes the remaining arguments name, a
# CUDA fatbin or a HIP offload bundle, with @p target's includes and definitions; the bundle links into @p target
# page-aligned in a read-only section through an assembler stub, under `<name>_bundle_start` and `<name>_bundle_end`, so
# it travels into archives too. With no codes the bundle is empty, and its devices refuse to load it. NVCC pairs with
# `CMAKE_CUDA_HOST_COMPILER` when set, and `CMAKE_CUDA_FLAGS` reach its pass. The tests and benchmarks bundle theirs
# through it too.
function (usearch_gpu_bundle target vendor name source)
    set(bundle_ ${CMAKE_CURRENT_BINARY_DIR}/bundles/${target}/${name}.bundle)
    if (vendor STREQUAL "nvidia")
        set(host_compiler_ ${CMAKE_CXX_COMPILER})
        if (CMAKE_CUDA_HOST_COMPILER)
            set(host_compiler_ ${CMAKE_CUDA_HOST_COMPILER})
        endif ()
        set(device_only_
            "${CUDAToolkit_NVCC_EXECUTABLE};-x;cu;--fatbin;-ccbin;${host_compiler_};--expt-relaxed-constexpr"
        )
        separate_arguments(device_flags_ UNIX_COMMAND "${CMAKE_CUDA_FLAGS}")
        set(host_only_ "-xcuda;--cuda-host-only;-nocudalib;--cuda-path=${CUDAToolkit_LIBRARY_ROOT}")
        # Clang's CUDA mode adds the toolkit's `include` alone, while CUDA 13 keeps `cuda/atomic` in CCCL's own.
        target_include_directories(${target} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    else ()
        set(device_only_ "${CMAKE_CXX_COMPILER};-x;hip;--cuda-device-only;-c")
        set(host_only_ "-xhip;--cuda-host-only;-nogpulib")
    endif ()
    set_source_files_properties(${source} PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "${host_only_}")
    if (ARGN)
        add_custom_command(
            OUTPUT ${bundle_}
            COMMAND
                ${device_only_} -std=c++20 -O3 ${ARGN} "-I$<JOIN:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>,;-I>"
                "-D$<JOIN:$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>,;-D>" ${device_flags_} -MD -MF ${bundle_}.d
                ${source} -o ${bundle_}
            DEPENDS ${source}
            DEPFILE ${bundle_}.d
            COMMENT "Compiling the ${name} kernels"
            COMMAND_EXPAND_LISTS VERBATIM
        )
    else ()
        file(WRITE ${bundle_} "")
    endif ()
    set(stub_ ${CMAKE_CURRENT_BINARY_DIR}/bundles/${target}/${name}_bundle.S)
    file(
        WRITE ${stub_}
        ".section .rodata\n.balign 4096\n.globl ${name}_bundle_start\n${name}_bundle_start:\n"
        ".incbin \"${bundle_}\"\n.globl ${name}_bundle_end\n${name}_bundle_end:\n"
        ".section .note.GNU-stack,\"\",@progbits\n"
    )
    set_source_files_properties(${stub_} PROPERTIES LANGUAGE ASM OBJECT_DEPENDS ${bundle_})
    target_sources(${target} PRIVATE ${stub_})
endfunction ()

# NVCC is only a device compiler here, and NVIDIA's units, tests and benchmarks bundle their kernels as above.
if (USEARCH_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(ASM)

    # Sets @p output to NVCC's codes for CMake-style @p architectures: a "-real" entry keeps its cubin alone, a
    # "-virtual" one its PTX alone, a bare one both, and "native" the visible GPUs'.
    function (usearch_cuda_codes output architectures)
        set(codes_)
        foreach (architecture_ IN LISTS architectures)
            if (architecture_ STREQUAL "native")
                list(APPEND codes_ -arch=native)
            elseif (architecture_ MATCHES "^([0-9]+[af]?)-real$")
                list(APPEND codes_ --generate-code=arch=compute_${CMAKE_MATCH_1},code=sm_${CMAKE_MATCH_1})
            elseif (architecture_ MATCHES "^([0-9]+[af]?)-virtual$")
                list(APPEND codes_ --generate-code=arch=compute_${CMAKE_MATCH_1},code=compute_${CMAKE_MATCH_1})
            else ()
                list(APPEND codes_ --generate-code=arch=compute_${architecture_},code=sm_${architecture_}
                     --generate-code=arch=compute_${architecture_},code=compute_${architecture_}
                )
            endif ()
        endforeach ()
        set(${output}
            "${codes_}"
            PARENT_SCOPE
        )
    endfunction ()

    # One unit per GPU class, its fatbin compiled for its own architectures alone - the device twin of the CPU classes'
    # per-unit ISA flags - so no cubin carries another generation's kernels. Each unit sees only its own NumKong GPU
    # target, as an AMD unit does, since NumKong's own kernels compile under every target a unit names.
    function (usearch_dense_vram_nvidia_variant class target_macro)
        set(target_ usearch_dense_vram_variants_${class})
        set(source_ ${CMAKE_CURRENT_SOURCE_DIR}/dense/vram_variants_${class}.cu)
        add_library(${target_} OBJECT ${source_})
        usearch_setup_target(${target_})
        target_include_directories(${target_} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(${target_} PRIVATE "USEARCH_ARCH_CUDA_=1" "NUMKONG_TARGET_${target_macro}=1")
        target_link_libraries(
            ${target_}
            PRIVATE yyjson forkunion::static
            PUBLIC ${CMAKE_DL_LIBS}
        )
        usearch_cuda_codes(codes_ "${ARGN}")
        usearch_gpu_bundle(${target_} nvidia dense_${class} ${source_} ${codes_})
        set(USEARCH_PUNNED_LIBRARIES
            ${USEARCH_PUNNED_LIBRARIES} ${target_}
            PARENT_SCOPE
        )
    endfunction ()
    usearch_dense_vram_nvidia_variant(nvidia_ampere AMPERE 80)
    usearch_dense_vram_nvidia_variant(nvidia_ada ADA 89-real)
    usearch_dense_vram_nvidia_variant(nvidia_hopper HOPPER 90a-real)
    usearch_dense_vram_nvidia_variant(nvidia_blackwell BLACKWELL 100f-real)
    usearch_dense_vram_nvidia_variant(nvidia_blackwellrtx BLACKWELLRTX 120f-real)

    target_compile_definitions(usearch_punned PRIVATE "USEARCH_ARCH_CUDA_=1")
else ()
    target_compile_definitions(usearch_punned PRIVATE "USEARCH_ARCH_CUDA_=0")
endif ()

# The AMD arms join beside the NVIDIA ones, so one dispatch unit carries both vendors. Each class unit bundles its
# kernels as above, through ROCm's Clang, never through CMake's HIP language, whose fat binaries would register with
# `libamdhip64`, for the class's processors `USEARCH_ROCM_ARCHITECTURES` lists.
if (USEARCH_BUILD_ROCM)
    enable_language(ASM)
    function (usearch_dense_vram_amd_variant class target_macro)
        set(processors_)
        foreach (processor_ IN LISTS ARGN)
            if (processor_ IN_LIST USEARCH_ROCM_ARCHITECTURES)
                list(APPEND processors_ --offload-arch=${processor_})
            endif ()
        endforeach ()
        set(target_ usearch_dense_vram_variants_${class})
        set(source_ ${CMAKE_CURRENT_SOURCE_DIR}/dense/vram_variants_${class}.hip)
        add_library(${target_} OBJECT ${source_})
        usearch_setup_target(${target_})
        target_include_directories(${target_} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(${target_} PRIVATE "USEARCH_ARCH_ROCM_=1" "NUMKONG_TARGET_${target_macro}=1")
        target_link_libraries(
            ${target_}
            PRIVATE yyjson forkunion::static
            PUBLIC ${CMAKE_DL_LIBS}
        )
        # A class none of whose processors is listed still links its arms, over an empty code object, so the dispatch
        # unit lists every class whatever the toolchain compiles for.
        usearch_gpu_bundle(${target_} amd dense_${class} ${source_} ${processors_})
        set(USEARCH_PUNNED_LIBRARIES
            ${USEARCH_PUNNED_LIBRARIES} ${target_}
            PARENT_SCOPE
        )
    endfunction ()
    usearch_dense_vram_amd_variant(amd_cdna3 CDNA3 gfx942 gfx950)
    usearch_dense_vram_amd_variant(amd_cdna4 CDNA4 gfx950)
    usearch_dense_vram_amd_variant(amd_cdna5 CDNA5 gfx1250 gfx1251)
    target_compile_definitions(usearch_punned PRIVATE "USEARCH_ARCH_ROCM_=1")
endif ()

# The Apple arms: one plain C++ unit serving every Apple class until a family's kernels differ, whose kernels compile
# from embedded source at first use, so no Metal toolchain is needed to build it. The dispatch unit and the Apple unit
# see `USEARCH_ARCH_METAL_=1`, and nothing links Metal, Foundation or `libobjc`, which `metal_api` opens at first use.
if (USEARCH_BUILD_METAL)
    add_library(usearch_dense_vram_variants_apple OBJECT dense/vram_variants_apple.cpp)
    usearch_setup_target(usearch_dense_vram_variants_apple)
    target_include_directories(usearch_dense_vram_variants_apple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(usearch_dense_vram_variants_apple PRIVATE "USEARCH_ARCH_METAL_=1")
    target_link_libraries(usearch_dense_vram_variants_apple PRIVATE yyjson forkunion::static)
    set(USEARCH_PUNNED_LIBRARIES ${USEARCH_PUNNED_LIBRARIES} usearch_dense_vram_variants_apple)
    target_compile_definitions(usearch_punned PRIVATE "USEARCH_ARCH_METAL_=1")
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
