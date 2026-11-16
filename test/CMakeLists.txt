# The header-only suite: the basic graph index and the utilities under it. ForkUnion, NumKong and yyjson appear as the
# tests' own tools - the ISA-specific atomic references, the tensors, the checkpoint files - never through the library.
add_executable(
    usearch_test_header_only header_only.cpp utilities/ram_buffers.cpp utilities/ram_tables.cpp basic/sentences.cpp
                             basic/bioinformatics.cpp basic/meshes.cpp
)
usearch_setup_target(usearch_test_header_only HOST)
target_include_directories(usearch_test_header_only PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools)
target_link_libraries(
    usearch_test_header_only PRIVATE fmt::fmt yyjson forkunion::header numkong::header stringzilla::header
)
add_test(NAME usearch_test_header_only COMMAND usearch_test_header_only)

# The header-only tier's witness: everything under `utilities/`, `basic/` & `geospatial/` compiles & runs from headers
# alone - its own and the header-only siblings', with no compiled library or yyjson behind it.
add_executable(usearch_test_header_only_witness header_only_witness.cpp)
usearch_setup_target(usearch_test_header_only_witness HOST)
target_link_libraries(usearch_test_header_only_witness PRIVATE forkunion::header numkong::header)
add_test(NAME usearch_test_header_only_witness COMMAND usearch_test_header_only_witness)

# The dependency-free core: `basic/ram_index.hpp` and its `utilities/` suppliers compile with no sibling on the include
# path, so a stray include fails here instead of hiding behind the suites' links.
add_library(usearch_test_header_core OBJECT header_core.cpp)
target_compile_features(usearch_test_header_core PRIVATE cxx_std_20)
target_include_directories(usearch_test_header_core PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/c)

# The Linux kernel ABI mirrors match the kernel's own spelling, checked at compile time against the build's headers.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(usearch_test_kernel_abi OBJECT kernel_abi.cpp)
    target_compile_features(usearch_test_kernel_abi PRIVATE cxx_std_20)
    target_include_directories(usearch_test_kernel_abi PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/c)
    target_link_libraries(usearch_test_kernel_abi PRIVATE forkunion::header numkong::header)
endif ()

# The VRAM cells compile once per vendor, so each unit sees the SDK facts of the one vendor it composes over: the dense
# rung cells and, for the vendors that have them, the modalities' units after `SOURCES`, each instantiated for the
# executor of @p vendor_ alone, whose `USEARCH_ARCH_<VENDOR>_` is @p architecture_macro_ and the others' are zero, with
# the `DEFINITIONS` the vendor's kits read. The vendor's `ENTRY` registers with the runner, which names none of them.
# NVIDIA and AMD units compile host-only through Clang's CUDA or HIP mode, as the vendor's entry does. NVIDIA's cells
# run the library's class fatbins; AMD's `DEVICE` entry compiles once more device-only into the suite's own code object
# of the `CODES`. Apple's kernels compile from embedded source at first use, so its entry is a plain C++ unit.
function (usearch_test_vram_shards vendor_ architecture_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "ENTRY;DEVICE" "SOURCES;DEFINITIONS;CODES")
    set(target_ usearch_test_vram_shards_${vendor_})
    set(entry_ ${CMAKE_CURRENT_SOURCE_DIR}/${arg_ENTRY})
    add_library(${target_} OBJECT dense/vram_indexes.cpp ${arg_SOURCES} ${entry_})
    usearch_setup_target(${target_} HOST)
    target_include_directories(
        ${target_} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools ${PROJECT_SOURCE_DIR}/src
    )
    foreach (macro_ IN ITEMS USEARCH_ARCH_CUDA_ USEARCH_ARCH_ROCM_ USEARCH_ARCH_METAL_)
        if (macro_ STREQUAL architecture_macro_)
            target_compile_definitions(${target_} PRIVATE ${macro_}=1)
        else ()
            target_compile_definitions(${target_} PRIVATE ${macro_}=0)
        endif ()
    endforeach ()
    target_compile_definitions(${target_} PRIVATE ${arg_DEFINITIONS})
    target_link_libraries(${target_} PRIVATE fmt::fmt yyjson forkunion::static stringzilla::header)
    if (vendor_ STREQUAL "nvidia" OR vendor_ STREQUAL "amd")
        set(device_)
        if (arg_DEVICE)
            set(device_ DEVICE ${CMAKE_CURRENT_SOURCE_DIR}/${arg_DEVICE})
        endif ()
        usearch_gpu_bundle(${target_} ${vendor_} test_vram_${vendor_} ${entry_} ${arg_CODES} ${device_})
        usearch_gpu_host_only(${target_} ${entry_})
    endif ()
    set(usearch_test_vram_shards_
        ${usearch_test_vram_shards_} ${target_}
        PARENT_SCOPE
    )
endfunction ()

# One row per vendor: NVIDIA's cells run the newest class the build keeps, on the devices the library picks it for, and
# AMD compiles for the listed processors of the CDNA3 class, which CDNA3 and CDNA4 run. Neither defines a NumKong or
# StringZilla kernel, as the library's class units define none.
set(usearch_test_vram_shards_)
if (USEARCH_CUDA_ARCH)
    usearch_test_vram_shards(
        nvidia USEARCH_ARCH_CUDA_
        ENTRY dense/vram_nvidia.cpp
        SOURCES sequence/vram_index.cpp shapes/vram_index.cpp geospatial/vram_index.cpp
        DEFINITIONS ${usearch_gpu_gates_} "NUMKONG_TARGET_CUDA=0" "STRINGZILLA_TARGET_CUDA=0"
    )
endif ()
if (USEARCH_ROCM_ARCH)
    usearch_rocm_codes(usearch_test_rocm_codes_ gfx942 gfx950)
    usearch_test_vram_shards(
        amd USEARCH_ARCH_ROCM_
        ENTRY dense/vram_amd.hip
        DEVICE dense/vram_kernels_amd.hip
        DEFINITIONS "NUMKONG_TARGET_ROCM=0" "STRINGZILLA_TARGET_ROCM=0"
        CODES ${usearch_test_rocm_codes_}
    )
endif ()
if (USEARCH_METAL_ARCH)
    usearch_test_vram_shards(
        apple USEARCH_ARCH_METAL_
        ENTRY dense/vram_apple.cpp
        SOURCES sequence/vram_index.cpp shapes/vram_index.cpp geospatial/vram_index.cpp
        DEFINITIONS ${usearch_gpu_gates_} "STRINGZILLA_ARCH_METAL_=1" "NUMKONG_ARCH_METAL_=1"
    )
    # The faces splice StringZilla's and NumKong's Metal sources, which those headers embed by their include paths.
    target_compile_options(
        usearch_test_vram_shards_apple PRIVATE "--embed-dir=${stringzilla_SOURCE_DIR}/include"
                                               "--embed-dir=${numkong_SOURCE_DIR}/include"
    )
endif ()

# The engine suite: the dense behavior cells, one per mechanism, with each storage dtype's cells in a unit of its own
# that compiles only when `USEARCH_DENSE_DTYPES` keeps the dtype and registers itself with the runner; the sequence,
# shapes and geospatial engines; the suites over the C interface; and every vendor's VRAM cells the build carries. It
# links the library, which the suites over the C interface call and the device listing names its classes through.
# Optimization level comes from the configuration: measured on these units, -O1 costs as much as -O3 and only slows the
# suite down.
add_executable(
    usearch_test
    main.cpp
    dense.cpp
    utilities/safegraphs_format.cpp
    dense/ram_metric.cpp
    dense/nvme_indexes.cpp
    sequence/ram_index.cpp
    shapes/ram_index.cpp
    geospatial/ram_index.cpp
    abi/dense.cpp
    abi/sequence.cpp
    abi/shapes.cpp
    abi/geospatial.cpp
)
usearch_dense_dtype_sources(
    usearch_test
    BF16 dense/ram_bf16.cpp
    U1 dense/ram_u1.cpp
    I8 dense/ram_i8.cpp
    E4M3 dense/ram_e4m3.cpp
    F16 dense/ram_f16.cpp
    U8 dense/ram_u8.cpp
    E5M2 dense/ram_e5m2.cpp
    I4 dense/ram_i4.cpp
    U4 dense/ram_u4.cpp
    E3M2 dense/ram_e3m2.cpp
    E2M3 dense/ram_e2m3.cpp
)
usearch_setup_target(usearch_test HOST)
target_include_directories(
    usearch_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(
    usearch_test
    PRIVATE ${usearch_test_vram_shards_}
            fmt::fmt
            yyjson
            forkunion::static
            stringzilla::header
            ${USEARCH_PUNNED_LIBRARIES}
            ${CMAKE_DL_LIBS}
)
add_test(NAME usearch_test COMMAND usearch_test)

# The class bundles hold USearch's kernels alone: no NumKong or StringZilla kernel or device table rides along.
if (USEARCH_CUDA_ARCH OR USEARCH_ROCM_ARCH)
    if (USEARCH_CUDA_ARCH)
        find_program(usearch_test_cuobjdump_ cuobjdump HINTS ${CUDAToolkit_BIN_DIR} REQUIRED)
    endif ()
    if (USEARCH_ROCM_ARCH)
        find_program(usearch_test_offload_bundler_ clang-offload-bundler HINTS /opt/rocm/llvm/bin REQUIRED)
        find_program(usearch_test_readelf_ llvm-readelf HINTS /opt/rocm/llvm/bin REQUIRED)
    endif ()
    add_test(
        NAME usearch_test_gpu_bundles
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bundles.sh ${PROJECT_BINARY_DIR}/src/bundles "${usearch_test_cuobjdump_}"
                "${usearch_test_offload_bundler_}" "${usearch_test_readelf_}" ${usearch_rocm_architectures_}
    )
endif ()

# Placement over emulated machines: listings built from machine profiles, with no GPU or driver behind them.
add_executable(usearch_test_placement placement.cpp)
usearch_setup_target(usearch_test_placement HOST)
target_include_directories(
    usearch_test_placement PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(usearch_test_placement PRIVATE fmt::fmt yyjson forkunion::static stringzilla::header)
add_test(NAME usearch_test_placement COMMAND usearch_test_placement)

# The command-line tools' suite: the Arrow export and import behind `usearch export` and `usearch import`,
# round-tripping a checkpoint of every modality through the C interface, and the slots of a file standing as a raw
# drive.
add_executable(usearch_test_tools tools.cpp)
usearch_setup_target(usearch_test_tools HOST)
target_include_directories(usearch_test_tools PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools)
target_link_libraries(
    usearch_test_tools PRIVATE fmt::fmt yyjson forkunion::static stringzilla::header ${USEARCH_PUNNED_LIBRARIES}
)
add_test(NAME usearch_test_tools COMMAND usearch_test_tools)
