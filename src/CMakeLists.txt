# The file runs top to bottom: the classes of every backend - the CPU classes, then one block per GPU vendor - the
# variant libraries, one row per modality, residency and backend, and the dispatch unit over all of them. Every class
# leaves the same three variables, read by the rows: `usearch_class_<class>_exists_`, `usearch_class_<class>_flags_`,
# the ISA flags of a CPU class or the device codes of a GPU class, and `usearch_class_<class>_definitions_`. Every
# backend leaves `usearch_<backend>_classes_`, the classes it declared, and `usearch_<backend>_definitions_`, which its
# libraries compile with.

# The RAM variants compile per CPU class. A variant inlines its class's kits and runs on nothing older, so its unit
# takes the class's instruction set outright and the compiler may use it anywhere inside. A class is its parent plus a
# delta - the kits and the ForkUnion bits `parent_class_of`, `numkong_capabilities_of`, `waiter_for_cpu_class` and
# `atomic_ref_for_cpu_class` add over the parent - restated here row for row. A class inherits its parent's flags, adds
# the flags of each delta kit, and exists only when the parent does and every delta entry is enabled, after any `-D`
# override of its library. `USEARCH_TARGET_CLASSES` then keeps it or not, and the kept classes are the rung oracle at
# unit granularity, which the dispatch unit reads as `USEARCH_TARGET_<CLASS>`, 1 or 0, the one gate every variant list
# carries per class, collected in `usearch_cpu_gates_`. The `serial` class is the floor every CPU runs.
set(usearch_class_serial_exists_ TRUE)
set(usearch_class_serial_runs_ TRUE)
set(usearch_cpu_classes_ serial)
set(usearch_cpu_gates_)
set(usearch_cpu_definitions_ "USEARCH_ARCH_CUDA_=0")

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

# Declares the CPU class @p class_, gated as `USEARCH_TARGET_<target_macro>`, over the class named after `OVER` and the
# delta entries after it, and records the parent the basic and NVMe rows fall back to.
function (usearch_cpu_class class_ target_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "OVER" "")
    set(exists_ ${usearch_class_${arg_OVER}_exists_})
    set(runs_ ${usearch_class_${arg_OVER}_runs_})
    set(flags_ ${usearch_class_${arg_OVER}_flags_})
    foreach (entry_ IN LISTS arg_UNPARSED_ARGUMENTS)
        if (numkong_target_${entry_}_enabled)
            list(APPEND flags_ ${numkong_target_${entry_}_flags})
        elseif (NOT forkunion_target_${entry_}_enabled AND NOT usearch_target_${entry_}_compiles_)
            set(exists_ FALSE)
        endif ()
        if (NOT numkong_target_${entry_}_runs AND NOT forkunion_target_${entry_}_runs)
            set(runs_ FALSE)
        endif ()
        # StringZilla's kernels of the same generation inlined into the class may need more, like Haswell's `lzcnt`.
        if (stringzilla_target_${entry_}_enabled)
            list(APPEND flags_ ${stringzilla_target_${entry_}_flags})
        endif ()
    endforeach ()
    # A targeted build keeps serial and the named classes, or under "native" the classes this host runs in full.
    if (USEARCH_TARGET_CLASSES STREQUAL "native")
        set(kept_ ${runs_})
    elseif (USEARCH_TARGET_CLASSES AND NOT class_ IN_LIST USEARCH_TARGET_CLASSES)
        set(kept_ FALSE)
    else ()
        set(kept_ TRUE)
    endif ()
    if (exists_ AND kept_)
        list(APPEND usearch_cpu_classes_ ${class_})
        list(APPEND usearch_cpu_gates_ "USEARCH_TARGET_${target_macro_}=1")
    else ()
        list(APPEND usearch_cpu_gates_ "USEARCH_TARGET_${target_macro_}=0")
    endif ()
    usearch_union_isa_flags(usearch_class_${class_}_flags_ ${flags_})
    set(usearch_class_${class_}_exists_ ${exists_})
    set(usearch_class_${class_}_runs_ ${runs_})
    set(usearch_class_${class_}_parent_ ${arg_OVER})
    set(usearch_class_${class_}_macro_ ${target_macro_})
    return(
        PROPAGATE
        usearch_class_${class_}_exists_
        usearch_class_${class_}_runs_
        usearch_class_${class_}_flags_
        usearch_class_${class_}_parent_
        usearch_class_${class_}_macro_
        usearch_cpu_classes_
        usearch_cpu_gates_
    )
endfunction ()

# A WebAssembly module carries the one SIMD capability its toolchain declares, so its verdicts read the compiler's own
# macros rather than a probe that would pass the capability's flags itself.
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR CMAKE_SYSTEM_NAME STREQUAL "WASI")
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles(
        "#if !defined(__wasm_simd128__)\n#error\n#endif\nint main() { return 0; }" usearch_target_v128_compiles_
    )
    check_cxx_source_compiles(
        "#if !defined(__wasm_relaxed_simd__)\n#error\n#endif\nint main() { return 0; }"
        usearch_target_v128relaxed_compiles_
    )
endif ()

usearch_cpu_class(intel_haswell INTEL_HASWELL OVER serial haswell)
usearch_cpu_class(
    intel_sierra
    INTEL_SIERRA
    OVER
    intel_haswell
    alder
    sierra
    x86_tpause
    x86_cmpccxadd
)
usearch_cpu_class(intel_skylake INTEL_SKYLAKE OVER intel_haswell skylake)
usearch_cpu_class(intel_icelake INTEL_ICELAKE OVER intel_skylake icelake)
usearch_cpu_class(amd_genoa AMD_GENOA OVER intel_icelake genoa)
usearch_cpu_class(intel_sapphire INTEL_SAPPHIRE OVER amd_genoa sapphire sapphireamx)
usearch_cpu_class(intel_granite INTEL_GRANITE OVER intel_sapphire graniteamx)
usearch_cpu_class(
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
usearch_cpu_class(aws_g2 AWS_G2 OVER serial neon)
usearch_cpu_class(
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
usearch_cpu_class(apple_m4 APPLE_M4 OVER apple_m2 sme smebi32)
usearch_cpu_class(aws_g5 AWS_G5 OVER apple_m2 arm64_wfet)
usearch_cpu_class(nvidia_vera NVIDIA_VERA OVER aws_g5 neonfp8)
usearch_cpu_class(
    alibaba_c930
    ALIBABA_C930
    OVER
    serial
    rvv
    rvvbb
    riscv64_wrs
    riscv64_zacas
)
usearch_cpu_class(wasm32_v128 WASM32_V128 OVER serial v128)
usearch_cpu_class(wasm64_extended WASM64_EXTENDED OVER wasm32_v128 v128relaxed)
foreach (class_ IN LISTS USEARCH_TARGET_CLASSES)
    if (NOT class_ STREQUAL "native" AND NOT usearch_class_${class_}_exists_)
        message(FATAL_ERROR "USEARCH_TARGET_CLASSES names ${class_}, which is no CPU class this toolchain compiles")
    endif ()
endforeach ()

# Every NVIDIA or AMD unit with kernels compiles twice. Host-only, through Clang's CUDA or HIP mode, it names its
# kernels from device code alone, so nothing registers with or links a GPU runtime, which opens at first use.
# Device-only, NVCC for NVIDIA and Clang for AMD, it compiles into a bundle of the codes the remaining arguments name, a
# CUDA fatbin or a HIP offload bundle, with @p target_'s includes and definitions; the bundle links into @p target_
# page-aligned in a read-only section through an assembler stub, under `<name>_bundle_start` and `<name>_bundle_end`, so
# it travels into archives too. With no codes the bundle is empty, and its devices refuse to load it. NVCC pairs with
# `CMAKE_CUDA_HOST_COMPILER` when set, splits its optimizations across four threads, and `CMAKE_CUDA_FLAGS` reach its
# pass. The device pass compiles @p source_, or the `DEVICE` source when the unit keeps its kernels apart, as a dense
# unit does per dtype and an AMD unit must: HIP-Clang refuses a kernel's address in device code, so a host function
# takes it, which the host pass may never compile, or it would emit launch stubs and a fat-binary registration. A device
# pass split across every CPU grew one assembler past 10 GB, so each splits four ways and eight run at once. The GPU
# helpers here serve the tests and benchmarks too, which bundle their own kernels through them.
set_property(GLOBAL APPEND PROPERTY JOB_POOLS usearch_gpu_device_pass=8)
function (usearch_gpu_bundle target_ vendor_ name_ source_)
    cmake_parse_arguments(PARSE_ARGV 4 arg "" "DEVICE" "")
    set(device_source_ ${source_})
    if (arg_DEVICE)
        set(device_source_ ${arg_DEVICE})
    endif ()
    set(bundle_ ${CMAKE_CURRENT_BINARY_DIR}/bundles/${target_}/${name_}.bundle)
    # Debug trees compile device code unoptimized, as their host code: NVCC's `-O` reaches only its host compiler, so
    # its debug pass also takes its device compiler and assembler down to `-O0`, without `-G`'s device debug
    # information.
    if (vendor_ STREQUAL "nvidia")
        set(host_compiler_ ${CMAKE_CXX_COMPILER})
        if (CMAKE_CUDA_HOST_COMPILER)
            set(host_compiler_ ${CMAKE_CUDA_HOST_COMPILER})
        endif ()
        # NVCC's default leaves Turing and Ampere cubins uncompressed, ten times their compressed size.
        set(device_only_ "${CUDAToolkit_NVCC_EXECUTABLE};-x;cu;--fatbin;--compress-mode=size;-ccbin;${host_compiler_}"
                         --expt-relaxed-constexpr -split-compile=4
        )
        separate_arguments(device_flags_ UNIX_COMMAND "${CMAKE_CUDA_FLAGS}")
        set(optimization_ "$<$<CONFIG:Debug>:-O0$<SEMICOLON>-Xcicc$<SEMICOLON>-O0$<SEMICOLON>-Xptxas$<SEMICOLON>-O0>"
                          "$<$<NOT:$<CONFIG:Debug>>:-O3>"
        )
        set(host_only_ "-xcuda;--cuda-host-only;-nocudalib;--cuda-path=${CUDAToolkit_LIBRARY_ROOT}")
        # Clang's CUDA mode adds the toolkit's `include` alone, while CUDA 13 keeps `cuda/atomic` in CCCL's own.
        target_include_directories(${target_} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    else ()
        set(device_only_ "${CMAKE_CXX_COMPILER};-x;hip;--cuda-device-only;-c")
        set(optimization_ "$<$<CONFIG:Debug>:-O0>" "$<$<NOT:$<CONFIG:Debug>>:-O3>")
        set(host_only_ "-xhip;--cuda-host-only;-nogpulib")
    endif ()
    set_source_files_properties(${source_} PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "${host_only_}")
    if (arg_UNPARSED_ARGUMENTS)
        add_custom_command(
            OUTPUT ${bundle_}
            COMMAND
                ${device_only_} -std=c++20 ${optimization_} ${arg_UNPARSED_ARGUMENTS}
                "-I$<JOIN:$<TARGET_PROPERTY:${target_},INCLUDE_DIRECTORIES>,;-I>"
                "-D$<JOIN:$<TARGET_PROPERTY:${target_},COMPILE_DEFINITIONS>,;-D>" ${device_flags_} -MD -MF ${bundle_}.d
                ${device_source_} -o ${bundle_}
            DEPENDS ${device_source_}
            DEPFILE ${bundle_}.d
            COMMENT "Compiling the ${name_} kernels"
            JOB_POOL usearch_gpu_device_pass
            COMMAND_EXPAND_LISTS VERBATIM
        )
    else ()
        file(WRITE ${bundle_} "")
    endif ()
    set(stub_ ${CMAKE_CURRENT_BINARY_DIR}/bundles/${target_}/${name_}_bundle.S)
    file(
        WRITE ${stub_}
        ".section .rodata\n.balign 4096\n.globl ${name_}_bundle_start\n${name_}_bundle_start:\n"
        ".incbin \"${bundle_}\"\n.globl ${name_}_bundle_end\n${name_}_bundle_end:\n"
        ".section .note.GNU-stack,\"\",@progbits\n"
    )
    set_source_files_properties(${stub_} PROPERTIES LANGUAGE ASM OBJECT_DEPENDS ${bundle_})
    target_sources(${target_} PRIVATE ${stub_})
endfunction ()

# Puts the host-only flags `usearch_gpu_bundle` gave the GPU unit @p entry_ onto every C++ unit of @p target_, but for
# its assembler stub, so the unit's sibling shards see the vendor's headers and targets as the entry does.
function (usearch_gpu_host_only target_ entry_)
    get_source_file_property(host_only_ ${entry_} COMPILE_OPTIONS)
    foreach (flag_ IN LISTS host_only_)
        target_compile_options(${target_} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${flag_}>")
    endforeach ()
    set_source_files_properties(${entry_} PROPERTIES COMPILE_OPTIONS "")
endfunction ()

# Sets @p output_ to NVCC's codes for CMake-style @p architectures_: a "-real" entry keeps its cubin alone, a "-virtual"
# one its PTX alone, and a bare one both.
function (usearch_cuda_codes output_ architectures_)
    set(codes_)
    foreach (architecture_ IN LISTS architectures_)
        if (architecture_ MATCHES "^([0-9]+[af]?)-real$")
            list(APPEND codes_ --generate-code=arch=compute_${CMAKE_MATCH_1},code=sm_${CMAKE_MATCH_1})
        elseif (architecture_ MATCHES "^([0-9]+[af]?)-virtual$")
            list(APPEND codes_ --generate-code=arch=compute_${CMAKE_MATCH_1},code=compute_${CMAKE_MATCH_1})
        else ()
            list(APPEND codes_ --generate-code=arch=compute_${architecture_},code=sm_${architecture_}
                 --generate-code=arch=compute_${architecture_},code=compute_${architecture_}
            )
        endif ()
    endforeach ()
    set(${output_}
        "${codes_}"
        PARENT_SCOPE
    )
endfunction ()

# Sets @p output_ to whether the resolved `USEARCH_CUDA_ARCHITECTURES` name any of the codes after it, with or without
# CMake's "-real" or "-virtual".
function (usearch_cuda_names output_)
    set(named_ FALSE)
    foreach (architecture_ IN LISTS usearch_cuda_architectures_)
        if (architecture_ MATCHES "^([0-9]+[af]?)(-real|-virtual)?$")
            if (CMAKE_MATCH_1 IN_LIST ARGN)
                set(named_ TRUE)
            endif ()
        endif ()
    endforeach ()
    set(${output_}
        ${named_}
        PARENT_SCOPE
    )
endfunction ()

# Sets @p output_ to `capability_macro_=1` where every one of the architectures is one of NumKong's cached
# `numkong_target_<capability>_architectures`, with or without CMake's "-real" or "-virtual", and to
# `capability_macro_=0` otherwise. NumKong's one-code units need any of them; a unit here compiles every listed code,
# and a kit compiled into one pass is compiled into all.
function (usearch_cuda_capability capability_name_ capability_macro_ architectures_ output_)
    set(enabled_ 1)
    foreach (architecture_ IN LISTS architectures_)
        set(supported_ 0)
        foreach (supported_architecture_ IN LISTS numkong_target_${capability_name_}_architectures)
            if (architecture_ MATCHES "^${supported_architecture_}(-real|-virtual)?$")
                set(supported_ 1)
            endif ()
        endforeach ()
        if (NOT supported_)
            set(enabled_ 0)
        endif ()
    endforeach ()
    set(${output_}
        ${capability_macro_}=${enabled_}
        PARENT_SCOPE
    )
endfunction ()

# The NumKong GPU capabilities the CUDA codes run, as the definitions a target compiles with.
function (usearch_cuda_capabilities output_ architectures_)
    usearch_cuda_capability(ampere NUMKONG_TARGET_AMPERE "${architectures_}" ampere_)
    usearch_cuda_capability(ada NUMKONG_TARGET_ADA "${architectures_}" ada_)
    usearch_cuda_capability(hopper NUMKONG_TARGET_HOPPER "${architectures_}" hopper_)
    usearch_cuda_capability(blackwell NUMKONG_TARGET_BLACKWELL "${architectures_}" blackwell_)
    usearch_cuda_capability(blackwellrtx NUMKONG_TARGET_BLACKWELLRTX "${architectures_}" blackwellrtx_)
    set(${output_}
        ${ampere_} ${ada_} ${hopper_} ${blackwell_} ${blackwellrtx_}
        PARENT_SCOPE
    )
endfunction ()

# Sets @p output_ to the `--offload-arch` codes of the processors after it that `USEARCH_ROCM_ARCHITECTURES` lists.
function (usearch_rocm_codes output_)
    set(codes_)
    foreach (processor_ IN LISTS ARGN)
        if (processor_ IN_LIST USEARCH_ROCM_ARCHITECTURES)
            list(APPEND codes_ --offload-arch=${processor_})
        endif ()
    endforeach ()
    set(${output_}
        "${codes_}"
        PARENT_SCOPE
    )
endfunction ()

# The NVIDIA classes: one per GPU generation, its fatbin compiled for its own architectures alone - the device twin of
# the CPU classes' per-unit ISA flags - so no cubin carries another generation's kernels, and only once
# `USEARCH_CUDA_ARCHITECTURES` names one of its @p selectors_, which keeps a one-generation tree to that generation's
# kernels; an unnamed class still exists, over an empty bundle, but only a named one joins the dispatch units' ladders,
# through its gate `USEARCH_TARGET_NVIDIA_<CLASS>` in `usearch_gpu_gates_`, 1 or 0. Each class sees only its own NumKong
# GPU target, as an AMD class does, since NumKong's own kernels compile under every target a unit names. NVCC is only a
# device compiler here; the dispatch unit sees `USEARCH_ARCH_CUDA_` either way.
set(usearch_gpu_gates_)
if (USEARCH_BUILD_CUDA)
    set(usearch_nvidia_classes_)
    set(usearch_nvidia_definitions_ "USEARCH_ARCH_CUDA_=1")
    function (usearch_nvidia_class class_ target_macro_ selectors_)
        usearch_cuda_names(named_ ${selectors_})
        set(codes_)
        if (named_)
            usearch_cuda_codes(codes_ "${ARGN}")
        endif ()
        set(usearch_class_${class_}_exists_
            TRUE
            PARENT_SCOPE
        )
        set(usearch_class_${class_}_named_
            ${named_}
            PARENT_SCOPE
        )
        set(usearch_class_${class_}_flags_
            "${codes_}"
            PARENT_SCOPE
        )
        set(usearch_class_${class_}_definitions_
            "NUMKONG_TARGET_${target_macro_}=1"
            PARENT_SCOPE
        )
        if (named_)
            list(APPEND usearch_gpu_gates_ "USEARCH_TARGET_NVIDIA_${target_macro_}=1")
        else ()
            list(APPEND usearch_gpu_gates_ "USEARCH_TARGET_NVIDIA_${target_macro_}=0")
        endif ()
        set(usearch_nvidia_classes_
            ${usearch_nvidia_classes_} ${class_}
            PARENT_SCOPE
        )
        set(usearch_gpu_gates_
            ${usearch_gpu_gates_}
            PARENT_SCOPE
        )
    endfunction ()
    usearch_nvidia_class(nvidia_ampere AMPERE "80;86" 80)
    usearch_nvidia_class(nvidia_ada ADA "89" 89-real)
    usearch_nvidia_class(nvidia_hopper HOPPER "90a" 90a-real)
    usearch_nvidia_class(nvidia_blackwell BLACKWELL "100;100a;100f;103a;103f" 100f-real)
    usearch_nvidia_class(nvidia_blackwellrtx BLACKWELLRTX "120f;121f" 120f-real)

    # The vendor-wide class `nvidia`, for kits that need nothing past the Ampere floor: one unit carrying a cubin for
    # each generation named above and the floor's PTX with Ampere's. Ada runs the Ampere cubin, so an Ada build bundles
    # the floor too. It gates its variants through its own `USEARCH_TARGET_NVIDIA` whenever it carries any generation,
    # so a build without the Ampere class never picks an empty Ampere rung.
    set(usearch_nvidia_architectures_)
    if (usearch_class_nvidia_ampere_named_ OR usearch_class_nvidia_ada_named_)
        list(APPEND usearch_nvidia_architectures_ 80)
    endif ()
    if (usearch_class_nvidia_hopper_named_)
        list(APPEND usearch_nvidia_architectures_ 90-real)
    endif ()
    if (usearch_class_nvidia_blackwell_named_)
        list(APPEND usearch_nvidia_architectures_ 100f-real)
    endif ()
    if (usearch_class_nvidia_blackwellrtx_named_)
        list(APPEND usearch_nvidia_architectures_ 120f-real)
    endif ()
    set(usearch_class_nvidia_exists_ TRUE)
    usearch_cuda_codes(usearch_class_nvidia_flags_ "${usearch_nvidia_architectures_}")
    if (usearch_nvidia_architectures_)
        list(APPEND usearch_gpu_gates_ "USEARCH_TARGET_NVIDIA=1")
    else ()
        list(APPEND usearch_gpu_gates_ "USEARCH_TARGET_NVIDIA=0")
    endif ()
    set(usearch_class_nvidia_definitions_ "NUMKONG_TARGET_AMPERE=1")
else ()
    set(usearch_nvidia_definitions_ "USEARCH_ARCH_CUDA_=0")
    list(
        APPEND
        usearch_gpu_gates_
        USEARCH_TARGET_NVIDIA=0
        USEARCH_TARGET_NVIDIA_AMPERE=0
        USEARCH_TARGET_NVIDIA_ADA=0
        USEARCH_TARGET_NVIDIA_HOPPER=0
        USEARCH_TARGET_NVIDIA_BLACKWELL=0
        USEARCH_TARGET_NVIDIA_BLACKWELLRTX=0
    )
endif ()

# The AMD classes, each embedding a code object for each of its processors `USEARCH_ROCM_ARCHITECTURES` lists, through
# ROCm's Clang, never through CMake's HIP language, whose fat binaries would register with `libamdhip64`. A class none
# of whose processors is listed still exists, over an empty code object, so every class builds whatever the toolchain
# compiles for, while only one with a listed processor joins the dispatch units' ladders through
# `USEARCH_TARGET_AMD_<CLASS>`.
if (USEARCH_BUILD_ROCM)
    set(usearch_amd_classes_)
    set(usearch_amd_definitions_ "USEARCH_ARCH_ROCM_=1")
    function (usearch_amd_class class_ target_macro_)
        usearch_rocm_codes(codes_ ${ARGN})
        set(usearch_class_${class_}_exists_
            TRUE
            PARENT_SCOPE
        )
        set(usearch_class_${class_}_flags_
            "${codes_}"
            PARENT_SCOPE
        )
        set(usearch_class_${class_}_definitions_
            "NUMKONG_TARGET_${target_macro_}=1"
            PARENT_SCOPE
        )
        if (codes_)
            list(APPEND usearch_gpu_gates_ "USEARCH_TARGET_AMD_${target_macro_}=1")
        else ()
            list(APPEND usearch_gpu_gates_ "USEARCH_TARGET_AMD_${target_macro_}=0")
        endif ()
        set(usearch_amd_classes_
            ${usearch_amd_classes_} ${class_}
            PARENT_SCOPE
        )
        set(usearch_gpu_gates_
            ${usearch_gpu_gates_}
            PARENT_SCOPE
        )
    endfunction ()
    usearch_amd_class(amd_cdna3 CDNA3 gfx942 gfx950)
    usearch_amd_class(amd_cdna4 CDNA4 gfx950)
    usearch_amd_class(amd_cdna5 CDNA5 gfx1250 gfx1251)
else ()
    list(APPEND usearch_gpu_gates_ USEARCH_TARGET_AMD_CDNA3=0 USEARCH_TARGET_AMD_CDNA4=0 USEARCH_TARGET_AMD_CDNA5=0)
endif ()

# The Apple classes, one per Metal generation, whose kernels compile from embedded source at first use, one library per
# dtype and class, so no Metal toolchain is needed to build them and a class carries no flags. Nothing links Metal,
# Foundation or `libobjc`, which `metal_api` opens at first use.
if (USEARCH_BUILD_METAL)
    set(usearch_apple_classes_)
    set(usearch_apple_definitions_ "USEARCH_ARCH_METAL_=1")
    function (usearch_apple_class class_ target_macro_)
        set(usearch_class_${class_}_exists_
            TRUE
            PARENT_SCOPE
        )
        set(usearch_apple_classes_
            ${usearch_apple_classes_} ${class_}
            PARENT_SCOPE
        )
        set(usearch_gpu_gates_
            ${usearch_gpu_gates_} "USEARCH_TARGET_APPLE_${target_macro_}=1"
            PARENT_SCOPE
        )
    endfunction ()
    usearch_apple_class(apple_metal7 METAL7)
    usearch_apple_class(apple_metal9 METAL9)
    usearch_apple_class(apple_metal10 METAL10)
else ()
    list(APPEND usearch_gpu_gates_ USEARCH_TARGET_APPLE_METAL7=0 USEARCH_TARGET_APPLE_METAL9=0
         USEARCH_TARGET_APPLE_METAL10=0
    )
endif ()

# Declares the object library @p target_ of the units after `SOURCES`, wired as every library here is: the shared setup,
# this directory's includes, yyjson and ForkUnion, and the `DEFINITIONS` and `LIBRARIES` it adds.
function (usearch_object_library target_)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;DEFINITIONS;LIBRARIES")
    add_library(${target_} OBJECT ${arg_SOURCES})
    usearch_setup_target(${target_})
    # A shared library or Python module built of these objects exports only the `USEARCH_EXPORT` entry points, which
    # `USEARCH_EXPORTING` turns from imports into exports on Windows.
    set_target_properties(
        ${target_}
        PROPERTIES C_VISIBILITY_PRESET hidden
                   CXX_VISIBILITY_PRESET hidden
                   VISIBILITY_INLINES_HIDDEN ON
    )
    target_include_directories(${target_} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(${target_} PRIVATE USEARCH_EXPORTING ${arg_DEFINITIONS})
    target_link_libraries(${target_} PRIVATE yyjson forkunion::static ${arg_LIBRARIES})
endfunction ()

# Declares the @p modality_ variants resident in @p residency_ on @p backend_, over the classes after it that exist, and
# adds the libraries to `usearch_<backend>_variants_`. Each class's unit is `<modality>/<residency>_variants_<class>`. A
# CPU or Apple backend builds one library, `usearch_<modality>_<residency>_variants`, with `_apple` for Apple, of a unit
# per class compiled with its class's flags. An NVIDIA or AMD backend builds one library per class,
# `usearch_<modality>_<residency>_variants_<class>`, since each bundles its own device codes under its own NumKong
# target, and links the dynamic loader its runtime opens through. A modality naming its `KERNELS`, the dtypes or metrics
# its kernels specialize on, keeps the host variants in the class unit, `.cpp` on NVIDIA and `.hip` on AMD, and compiles
# each from `<modality>/<residency>_kernels_<backend>_<kernel>` into a bundle of its own, once per class, dense taking
# only its first `USEARCH_DENSE_DTYPES`; the others compile their one `.cu` class unit both ways. A CPU row naming its
# `GATE` tells its dispatch unit `<modality>_index.cpp` which listed classes it compiled, as
# `USEARCH_TARGET_<GATE>_<CLASS>`, 1 or 0, since an ancestor it compiles for a kept class may be no kept class itself.
function (usearch_variants modality_ residency_ backend_)
    cmake_parse_arguments(PARSE_ARGV 3 arg "" "GATE" "DEFINITIONS;LIBRARIES;KERNELS")
    set(library_ usearch_${modality_}_${residency_}_variants)
    set(unit_ ${modality_}/${residency_}_variants)
    set(classes_)
    if (backend_ STREQUAL "cpu")
        # Each kept CPU class compiles as itself where the row lists it, else as the nearest ancestor the row lists.
        foreach (kept_ IN LISTS usearch_cpu_classes_)
            set(ancestor_ ${kept_})
            while (ancestor_ AND NOT ancestor_ IN_LIST arg_UNPARSED_ARGUMENTS)
                set(ancestor_ ${usearch_class_${ancestor_}_parent_})
            endwhile ()
            if (ancestor_ AND NOT ancestor_ IN_LIST classes_)
                list(APPEND classes_ ${ancestor_})
            endif ()
        endforeach ()
        if (arg_GATE)
            set(gates_)
            foreach (class_ IN LISTS arg_UNPARSED_ARGUMENTS)
                if (class_ STREQUAL "serial")
                    continue()
                elseif (class_ IN_LIST classes_)
                    list(APPEND gates_ "USEARCH_TARGET_${arg_GATE}_${usearch_class_${class_}_macro_}=1")
                else ()
                    list(APPEND gates_ "USEARCH_TARGET_${arg_GATE}_${usearch_class_${class_}_macro_}=0")
                endif ()
            endforeach ()
            set_property(
                SOURCE ${modality_}_index.cpp
                APPEND
                PROPERTY COMPILE_DEFINITIONS ${gates_}
            )
        endif ()
    else ()
        foreach (class_ IN LISTS arg_UNPARSED_ARGUMENTS)
            if (usearch_class_${class_}_exists_)
                list(APPEND classes_ ${class_})
            endif ()
        endforeach ()
    endif ()
    if (NOT classes_)
        return()
    endif ()
    set(libraries_)
    if (backend_ STREQUAL "cpu" OR backend_ STREQUAL "apple")
        if (backend_ STREQUAL "apple")
            set(library_ ${library_}_apple)
        endif ()
        usearch_object_library(
            ${library_}
            DEFINITIONS ${usearch_${backend_}_definitions_} ${arg_DEFINITIONS}
            LIBRARIES ${arg_LIBRARIES}
        )
        foreach (class_ IN LISTS classes_)
            target_sources(${library_} PRIVATE ${unit_}_${class_}.cpp)
            set_source_files_properties(
                ${unit_}_${class_}.cpp PROPERTIES COMPILE_OPTIONS "${usearch_class_${class_}_flags_}"
            )
        endforeach ()
        list(APPEND libraries_ ${library_})
    else ()
        foreach (class_ IN LISTS classes_)
            set(target_ ${library_}_${class_})
            if (NOT arg_KERNELS)
                set(source_ ${CMAKE_CURRENT_SOURCE_DIR}/${unit_}_${class_}.cu)
            elseif (backend_ STREQUAL "nvidia")
                set(source_ ${CMAKE_CURRENT_SOURCE_DIR}/${unit_}_${class_}.cpp)
                set(extension_ cu)
            else ()
                set(source_ ${CMAKE_CURRENT_SOURCE_DIR}/${unit_}_${class_}.hip)
                set(extension_ hip)
            endif ()
            usearch_object_library(
                ${target_}
                SOURCES ${source_}
                DEFINITIONS ${usearch_${backend_}_definitions_} ${usearch_class_${class_}_definitions_}
                            ${arg_DEFINITIONS}
                LIBRARIES ${arg_LIBRARIES}
            )
            target_link_libraries(${target_} PUBLIC ${CMAKE_DL_LIBS})
            if (NOT arg_KERNELS)
                usearch_gpu_bundle(
                    ${target_} ${backend_} ${modality_}_${class_} ${source_} ${usearch_class_${class_}_flags_}
                )
            endif ()
            set(position_ 0)
            set(rows_)
            foreach (kernel_ IN LISTS arg_KERNELS)
                if (NOT modality_ STREQUAL "dense" OR position_ LESS USEARCH_DENSE_DTYPES)
                    usearch_gpu_bundle(
                        ${target_}
                        ${backend_}
                        ${modality_}_${class_}_${kernel_}
                        ${source_}
                        ${usearch_class_${class_}_flags_}
                        DEVICE
                        ${CMAKE_CURRENT_SOURCE_DIR}/${modality_}/${residency_}_kernels_${backend_}_${kernel_}.${extension_}
                    )
                    string(
                        APPEND
                        rows_
                        ".quad ${modality_}_${class_}_${kernel_}_bundle_start, ${modality_}_${class_}_${kernel_}_bundle_end\n"
                    )
                endif ()
                math(EXPR position_ "${position_} + 1")
            endforeach ()
            # The dense class unit finds each dtype's bundle in one table of two addresses per dtype, in
            # `dense_dtypes_t` order, under `<modality>_<class>_bundles`.
            if (modality_ STREQUAL "dense")
                set(table_ ${CMAKE_CURRENT_BINARY_DIR}/bundles/${target_}/${modality_}_${class_}_bundles.S)
                file(WRITE ${table_}
                     ".section .data.rel.ro\n.balign 8\n.globl ${modality_}_${class_}_bundles\n"
                     "${modality_}_${class_}_bundles:\n${rows_}.section .note.GNU-stack,\"\",@progbits\n"
                )
                set_source_files_properties(${table_} PROPERTIES LANGUAGE ASM)
                target_sources(${target_} PRIVATE ${table_})
            endif ()
            list(APPEND libraries_ ${target_})
        endforeach ()
    endif ()
    set(usearch_${backend_}_variants_
        ${usearch_${backend_}_variants_} ${libraries_}
        PARENT_SCOPE
    )
endfunction ()

# The dense NVMe variants compile for the CPU classes whose 4-way or tiles kits a storage walk or scan tells apart: they
# carry no waiter or atomic reference, so aws_g5 and alibaba_c930, which advance only by those, take none, and a
# WebAssembly module reads no storage directly. The sequence, shapes and geospatial variants compile for the CPU classes
# where any of their StringZilla or NumKong kits advance, and a modality whose kits stay put at a class falls back down
# its kit chain; on NVIDIA they need nothing past the Ampere floor, so they compile for the vendor-wide class, and
# shapes adds datacenter Blackwell for its packed f16 kit.
set(usearch_dense_nvme_classes_
    serial
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
set(usearch_basic_cpu_classes_
    serial
    intel_haswell
    intel_skylake
    aws_g2
    apple_m2
    alibaba_c930
    wasm64_extended
)

# The variant libraries: one row per modality, residency and backend. A backend a modality lacks is a missing row, and a
# row over the classes of a vendor this tree does not build declares nothing.
set(usearch_cpu_variants_)
set(usearch_nvidia_variants_)
set(usearch_amd_variants_)
set(usearch_apple_variants_)
usearch_variants(dense ram cpu ${usearch_cpu_classes_})
usearch_variants(dense nvme cpu ${usearch_dense_nvme_classes_} GATE NVME)
usearch_variants(
    dense vram nvidia ${usearch_nvidia_classes_}
    KERNELS bf16
            u1
            i8
            e4m3
            f16
            u8
            e5m2
            i4
            u4
            e3m2
            e2m3
)
usearch_variants(
    dense vram amd ${usearch_amd_classes_}
    KERNELS bf16
            u1
            i8
            e4m3
            f16
            u8
            e5m2
            i4
            u4
            e3m2
            e2m3
)
usearch_variants(dense vram apple ${usearch_apple_classes_})
usearch_variants(
    sequence cpu cpu ${usearch_basic_cpu_classes_} GATE BASIC
    DEFINITIONS "STRINGZILLA_HEADER_ONLY=1"
    LIBRARIES stringzilla::header
)
usearch_variants(
    sequence vram nvidia nvidia
    KERNELS levenshtein_bytes levenshtein_utf8 overlap needleman_wunsch
    DEFINITIONS "STRINGZILLA_HEADER_ONLY=1"
    LIBRARIES stringzilla::header
)
usearch_variants(shapes cpu cpu ${usearch_basic_cpu_classes_} GATE BASIC LIBRARIES stringzilla::header)
usearch_variants(shapes vram nvidia nvidia KERNELS f16 f32)
usearch_variants(shapes vram nvidia nvidia_blackwell KERNELS f16)
usearch_variants(geospatial cpu cpu ${usearch_basic_cpu_classes_} GATE BASIC LIBRARIES stringzilla::header)
usearch_variants(geospatial vram nvidia nvidia)
usearch_variants(basic vram nvidia nvidia)

# The type-erased implementations behind the C interface: the dispatch units visit the variants through their explicitly
# instantiated views and compile at the baseline, since they inline no kernel, gated per CPU and GPU class and told
# which GPU vendors joined.
usearch_object_library(
    usearch_punned
    SOURCES types.cpp dense_index.cpp sequence_index.cpp shapes_index.cpp geospatial_index.cpp
    DEFINITIONS ${usearch_cpu_gates_} ${usearch_gpu_gates_} ${usearch_nvidia_definitions_} ${usearch_amd_definitions_}
                ${usearch_apple_definitions_}
    LIBRARIES stringzilla::header
)

# Objects never travel through another object library, so every consumer names the whole set: executables and shared
# libraries link the targets and inherit their dependencies, an archive takes the objects themselves.
set(USEARCH_PUNNED_LIBRARIES usearch_punned ${usearch_cpu_variants_} ${usearch_nvidia_variants_}
                             ${usearch_amd_variants_} ${usearch_apple_variants_}
)
set(USEARCH_PUNNED_OBJECTS)
foreach (library_ IN LISTS USEARCH_PUNNED_LIBRARIES)
    list(APPEND USEARCH_PUNNED_OBJECTS $<TARGET_OBJECTS:${library_}>)
endforeach ()
# The behavior tests compile their class cells with the classes' own flags, so every class's variables reach the parent.
set(usearch_class_variables_)
foreach (
    class_ IN
    LISTS usearch_cpu_classes_ usearch_nvidia_classes_
    ITEMS nvidia
)
    list(APPEND usearch_class_variables_ usearch_class_${class_}_exists_ usearch_class_${class_}_flags_
         usearch_class_${class_}_definitions_
    )
endforeach ()
return(PROPAGATE USEARCH_PUNNED_LIBRARIES USEARCH_PUNNED_OBJECTS usearch_cpu_classes_ usearch_nvidia_classes_
       ${usearch_class_variables_}
)
