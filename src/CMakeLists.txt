# The RAM variants compile per CPU class. A variant inlines its class's kits and runs on nothing older, so its unit
# takes the class's instruction set outright and the compiler may use it anywhere inside. A class is its parent plus a
# delta - the kits and the ForkUnion bits `parent_class_of`, `numkong_capabilities_of`, `waiter_for_cpu_class` and
# `atomic_ref_for_cpu_class` add over the parent - restated here row for row. A unit inherits its parent's flags, adds
# the flags of each delta kit that compiled, and exists only when the parent does and some delta entry compiled: the
# rung oracle at unit granularity, which the dispatch units read as `USEARCH_TARGET_<CLASS>`, the one gate every variant
# list carries per class.
add_library(usearch_dense_ram_variants OBJECT dense/ram_variants_serial.cpp)
usearch_setup_target(usearch_dense_ram_variants)
target_include_directories(usearch_dense_ram_variants PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(usearch_dense_ram_variants PRIVATE "USEARCH_ARCH_CUDA_=0")
target_link_libraries(usearch_dense_ram_variants PRIVATE yyjson forkunion::static)
set(usearch_variant_serial_exists_ TRUE)
set(usearch_variant_serial_flags_)
set(usearch_variant_targets_)

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

function (usearch_dense_ram_variant class target_macro)
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
        # StringZilla's kernels of the same generation inlined into the class may need more, like Haswell's `lzcnt`.
        if (sz_target_${entry_}_compiles)
            list(APPEND flags_ ${sz_target_${entry_}_flags})
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
    set(usearch_variant_targets_
        ${usearch_variant_targets_} "USEARCH_TARGET_${target_macro}=1"
        PARENT_SCOPE
    )
    target_sources(usearch_dense_ram_variants PRIVATE dense/ram_variants_${class}.cpp)
    set_source_files_properties(dense/ram_variants_${class}.cpp PROPERTIES COMPILE_OPTIONS "${flags_}")
endfunction ()

usearch_dense_ram_variant(intel_haswell INTEL_HASWELL OVER serial haswell)
usearch_dense_ram_variant(
    intel_sierra
    INTEL_SIERRA
    OVER
    intel_haswell
    alder
    sierra
    x86_tpause
    x86_cmpccxadd
)
usearch_dense_ram_variant(intel_skylake INTEL_SKYLAKE OVER intel_haswell skylake)
usearch_dense_ram_variant(intel_icelake INTEL_ICELAKE OVER intel_skylake icelake)
usearch_dense_ram_variant(amd_genoa AMD_GENOA OVER intel_icelake genoa)
usearch_dense_ram_variant(intel_sapphire INTEL_SAPPHIRE OVER amd_genoa sapphire sapphireamx)
usearch_dense_ram_variant(intel_granite INTEL_GRANITE OVER intel_sapphire graniteamx)
usearch_dense_ram_variant(
    intel_diamond
    INTEL_DIAMOND
    OVER
    intel_granite
    alder
    sierra
    diamond
    x86_tpause
    x86_cmpccxadd
)
usearch_dense_ram_variant(aws_g2 AWS_G2 OVER serial neon)
usearch_dense_ram_variant(
    apple_m2
    APPLE_M2
    OVER
    aws_g2
    neonsdot
    neonbfdot
    neonfhm
    arm64_lse
    arm64_rcpc
)
usearch_dense_ram_variant(apple_m4 APPLE_M4 OVER apple_m2 sme smebi32)
usearch_dense_ram_variant(aws_g5 AWS_G5 OVER apple_m2 arm64_wfet)
usearch_dense_ram_variant(nvidia_vera NVIDIA_VERA OVER aws_g5 neonfp8)
usearch_dense_ram_variant(
    alibaba_c930
    ALIBABA_C930
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
usearch_dense_ram_variant(wasm32_v128 WASM32_V128 OVER serial v128)
usearch_dense_ram_variant(wasm64_extended WASM64_EXTENDED OVER wasm32_v128 v128relaxed)

# The NVMe variants compile per CPU class with the RAM unit's flags, for the classes whose 4-way or tiles kits a storage
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

# The sequence, shapes and geospatial variants compile per CPU class with the flags the dense rows above computed, one
# unit set per basic modality, over the classes where any of their StringZilla or NumKong kits advance. Each unit
# instantiates its own modality's variants, and a modality whose kits stay put at the class falls back down its kit
# chain.
set(usearch_basic_ram_variants_)
foreach (modality_ sequence shapes geospatial)
    set(target_ usearch_${modality_}_ram_variants)
    add_library(${target_} OBJECT ${modality_}/ram_variants_serial.cpp)
    usearch_setup_target(${target_})
    target_include_directories(${target_} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(${target_} PRIVATE "USEARCH_ARCH_CUDA_=0")
    target_link_libraries(${target_} PRIVATE yyjson forkunion::static stringzilla::header)
    foreach (class_ intel_haswell intel_skylake aws_g2 apple_m2 alibaba_c930 wasm64_extended)
        if (usearch_variant_${class_}_exists_)
            target_sources(${target_} PRIVATE ${modality_}/ram_variants_${class_}.cpp)
            set_source_files_properties(
                ${modality_}/ram_variants_${class_}.cpp PROPERTIES COMPILE_OPTIONS
                                                                   "${usearch_variant_${class_}_flags_}"
            )
        endif ()
    endforeach ()
    list(APPEND usearch_basic_ram_variants_ ${target_})
endforeach ()

# The type-erased implementations behind the C interface: the dispatch units visit the variants through their explicitly
# instantiated views and compile at the baseline, since they inline no kernel. Under CUDA the device variants join, so
# the C libraries carry them too.
add_library(usearch_punned OBJECT types.cpp dense_index.cpp sequence_index.cpp shapes_index.cpp geospatial_index.cpp)
usearch_setup_target(usearch_punned)
target_include_directories(usearch_punned PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usearch_punned PRIVATE yyjson forkunion::static stringzilla::header)
target_compile_definitions(usearch_punned PRIVATE ${usearch_variant_targets_})
set(USEARCH_PUNNED_LIBRARIES usearch_punned usearch_dense_ram_variants usearch_dense_nvme_variants
                             ${usearch_basic_ram_variants_}
)

# Every NVIDIA or AMD unit with kernels compiles twice. Host-only, through Clang's CUDA or HIP mode, it names its
# kernels from device code alone, so nothing registers with or links a GPU runtime, which opens at first use.
# Device-only, NVCC for NVIDIA and Clang for AMD, it compiles into a bundle of the codes the remaining arguments name, a
# CUDA fatbin or a HIP offload bundle, with @p target's includes and definitions; the bundle links into @p target
# page-aligned in a read-only section through an assembler stub, under `<name>_bundle_start` and `<name>_bundle_end`, so
# it travels into archives too. With no codes the bundle is empty, and its devices refuse to load it. NVCC pairs with
# `CMAKE_CUDA_HOST_COMPILER` when set, splits its optimizations across every CPU, and `CMAKE_CUDA_FLAGS` reach its pass.
# The device pass compiles @p source, or the `DEVICE` source when the unit keeps its kernels apart, as an AMD unit must:
# HIP-Clang refuses a kernel's address in device code, so a host function takes it, which the host pass may never
# compile, or it would emit launch stubs and a fat-binary registration. The tests and benchmarks bundle theirs through
# it too. Device passes run one at a time, each spreading its own work across every CPU: NVCC's split assemblers take
# about 9 GB on the largest units, so two at once would exhaust a laptop.
set_property(GLOBAL APPEND PROPERTY JOB_POOLS usearch_gpu_device_pass=1)
function (usearch_gpu_bundle target vendor name source)
    cmake_parse_arguments(arg "" "DEVICE" "" ${ARGN})
    set(device_source_ ${source})
    if (arg_DEVICE)
        set(device_source_ ${arg_DEVICE})
    endif ()
    set(bundle_ ${CMAKE_CURRENT_BINARY_DIR}/bundles/${target}/${name}.bundle)
    if (vendor STREQUAL "nvidia")
        set(host_compiler_ ${CMAKE_CXX_COMPILER})
        if (CMAKE_CUDA_HOST_COMPILER)
            set(host_compiler_ ${CMAKE_CUDA_HOST_COMPILER})
        endif ()
        set(device_only_
            "${CUDAToolkit_NVCC_EXECUTABLE};-x;cu;--fatbin;-ccbin;${host_compiler_};--expt-relaxed-constexpr"
            -split-compile=0
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
    # Debug trees compile device code unoptimized, as their host code: NVCC's `-O` reaches only its host compiler, so
    # its debug pass also takes its device compiler and assembler down to `-O0`, without `-G`'s device debug
    # information.
    if (vendor STREQUAL "nvidia")
        set(optimization_ "$<$<CONFIG:Debug>:-O0$<SEMICOLON>-Xcicc$<SEMICOLON>-O0$<SEMICOLON>-Xptxas$<SEMICOLON>-O0>"
                          "$<$<NOT:$<CONFIG:Debug>>:-O3>"
        )
    else ()
        set(optimization_ "$<$<CONFIG:Debug>:-O0>" "$<$<NOT:$<CONFIG:Debug>>:-O3>")
    endif ()
    if (arg_UNPARSED_ARGUMENTS)
        add_custom_command(
            OUTPUT ${bundle_}
            COMMAND
                ${device_only_} -std=c++20 ${optimization_} ${arg_UNPARSED_ARGUMENTS}
                "-I$<JOIN:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>,;-I>"
                "-D$<JOIN:$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>,;-D>" ${device_flags_} -MD -MF ${bundle_}.d
                ${device_source_} -o ${bundle_}
            DEPENDS ${device_source_}
            DEPFILE ${bundle_}.d
            COMMENT "Compiling the ${name} kernels"
            JOB_POOL usearch_gpu_device_pass
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

# Puts the host-only flags `usearch_gpu_bundle` gave the GPU unit @p entry onto every C++ unit of @p target, but for its
# assembler stub, so the unit's sibling shards see the vendor's headers and targets as the entry does.
function (usearch_gpu_host_only target entry)
    get_source_file_property(host_only_ ${entry} COMPILE_OPTIONS)
    foreach (flag_ IN LISTS host_only_)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${flag_}>")
    endforeach ()
    set_source_files_properties(${entry} PROPERTIES COMPILE_OPTIONS "")
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
    # per-unit ISA flags - so no cubin carries another generation's kernels, and only once `USEARCH_CUDA_ARCHITECTURES`
    # names one of its @p selectors or "native", which keeps a one-generation tree to that generation's kernels. Each
    # unit sees only its own NumKong GPU target, as an AMD unit does, since NumKong's own kernels compile under every
    # target a unit names.
    function (usearch_dense_vram_nvidia_variant class target_macro selectors)
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
        set(codes_)
        foreach (selector_ IN ITEMS native ${selectors})
            if (selector_ IN_LIST USEARCH_CUDA_ARCHITECTURES)
                usearch_cuda_codes(codes_ "${ARGN}")
            endif ()
        endforeach ()
        usearch_gpu_bundle(${target_} nvidia dense_${class} ${source_} ${codes_})
        set(USEARCH_PUNNED_LIBRARIES
            ${USEARCH_PUNNED_LIBRARIES} ${target_}
            PARENT_SCOPE
        )
    endfunction ()
    usearch_dense_vram_nvidia_variant(nvidia_ampere AMPERE "80;80-real" 80)
    usearch_dense_vram_nvidia_variant(nvidia_ada ADA "89;89-real" 89-real)
    usearch_dense_vram_nvidia_variant(nvidia_hopper HOPPER "90a;90a-real" 90a-real)
    usearch_dense_vram_nvidia_variant(nvidia_blackwell BLACKWELL "100f;100f-real" 100f-real)
    usearch_dense_vram_nvidia_variant(nvidia_blackwellrtx BLACKWELLRTX "120f;120f-real" 120f-real)

    # Sets @p output to whether `USEARCH_CUDA_ARCHITECTURES` names "native" or any of the remaining selectors.
    function (usearch_cuda_names output)
        set(named_ FALSE)
        foreach (selector_ IN ITEMS native ${ARGN})
            if (selector_ IN_LIST USEARCH_CUDA_ARCHITECTURES)
                set(named_ TRUE)
            endif ()
        endforeach ()
        set(${output}
            ${named_}
            PARENT_SCOPE
        )
    endfunction ()

    # The modality kits need nothing past the Ampere floor, so each modality's unit carries a cubin for each class
    # `USEARCH_CUDA_ARCHITECTURES` names, every one under "native", and the floor's PTX with Ampere's. Ada runs the
    # Ampere cubin, so an Ada build bundles the floor too.
    function (usearch_basic_vram_nvidia_variant modality)
        cmake_parse_arguments(arg "" "" "DEFINITIONS;LIBRARIES" ${ARGN})
        set(target_ usearch_${modality}_vram_variants_nvidia)
        set(source_ ${CMAKE_CURRENT_SOURCE_DIR}/${modality}/vram_variants_nvidia.cu)
        add_library(${target_} OBJECT ${source_})
        usearch_setup_target(${target_})
        target_include_directories(${target_} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(
            ${target_} PRIVATE "USEARCH_ARCH_CUDA_=1" "NUMKONG_TARGET_AMPERE=1" ${arg_DEFINITIONS}
        )
        target_link_libraries(
            ${target_}
            PRIVATE yyjson forkunion::static ${arg_LIBRARIES}
            PUBLIC ${CMAKE_DL_LIBS}
        )
        set(architectures_)
        usearch_cuda_names(ampere_ 80 80-real 89 89-real)
        usearch_cuda_names(hopper_ 90a 90a-real)
        usearch_cuda_names(blackwell_ 100f 100f-real)
        usearch_cuda_names(blackwellrtx_ 120f 120f-real)
        if (ampere_)
            list(APPEND architectures_ 80)
        endif ()
        if (hopper_)
            list(APPEND architectures_ 90-real)
        endif ()
        if (blackwell_)
            list(APPEND architectures_ 100f-real)
        endif ()
        if (blackwellrtx_)
            list(APPEND architectures_ 120f-real)
        endif ()
        usearch_cuda_codes(codes_ "${architectures_}")
        usearch_gpu_bundle(${target_} nvidia ${modality}_nvidia ${source_} ${codes_})
        set(USEARCH_PUNNED_LIBRARIES
            ${USEARCH_PUNNED_LIBRARIES} ${target_}
            PARENT_SCOPE
        )
    endfunction ()
    usearch_basic_vram_nvidia_variant(sequence DEFINITIONS "STRINGZILLA_HEADER_ONLY=1" LIBRARIES stringzilla::header)
    usearch_basic_vram_nvidia_variant(shapes)
    usearch_basic_vram_nvidia_variant(geospatial)

    target_compile_definitions(usearch_punned PRIVATE "USEARCH_ARCH_CUDA_=1")
else ()
    target_compile_definitions(usearch_punned PRIVATE "USEARCH_ARCH_CUDA_=0")
endif ()

# The AMD variants join beside the NVIDIA ones, so one dispatch unit carries both vendors. Each class unit bundles its
# kernels as above, through ROCm's Clang, never through CMake's HIP language, whose fat binaries would register with
# `libamdhip64`, for the class's processors `USEARCH_ROCM_ARCHITECTURES` lists. The kernels sit in a device entry beside
# the class's host unit, which names none of them.
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
        set(kernels_ ${CMAKE_CURRENT_SOURCE_DIR}/dense/vram_kernels_${class}.hip)
        add_library(${target_} OBJECT ${source_})
        usearch_setup_target(${target_})
        target_include_directories(${target_} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(${target_} PRIVATE "USEARCH_ARCH_ROCM_=1" "NUMKONG_TARGET_${target_macro}=1")
        target_link_libraries(
            ${target_}
            PRIVATE yyjson forkunion::static
            PUBLIC ${CMAKE_DL_LIBS}
        )
        # A class none of whose processors is listed still links its variants, over an empty code object, so the
        # dispatch unit lists every class whatever the toolchain compiles for.
        usearch_gpu_bundle(
            ${target_}
            amd
            dense_${class}
            ${source_}
            ${processors_}
            DEVICE
            ${kernels_}
        )
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

# The Apple variants: one plain C++ unit per Apple class, as NVIDIA and AMD have one per class, whose kernels compile
# from embedded source at first use, one library per dtype and class, so no Metal toolchain is needed to build them. The
# dispatch unit and the Apple unit see `USEARCH_ARCH_METAL_=1`, and nothing links Metal, Foundation or `libobjc`, which
# `metal_api` opens at first use.
if (USEARCH_BUILD_METAL)
    add_library(
        usearch_dense_vram_variants_apple OBJECT
        dense/vram_variants_apple_metal7.cpp dense/vram_variants_apple_metal9.cpp dense/vram_variants_apple_metal10.cpp
    )
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
