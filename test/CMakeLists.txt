# The header-only suite: the basic graph index and the utilities under it. ForkUnion, NumKong and yyjson appear as the
# tests' own tools - the ISA-specific atomic references, the tensors, the checkpoint files - never through the library.
add_executable(
    usearch_test_header_only header_only.cpp utilities/ram_buffers.cpp utilities/ram_tables.cpp basic/sentences.cpp
                             basic/bioinformatics.cpp basic/meshes.cpp
)
usearch_setup_target(usearch_test_header_only HOST)
target_include_directories(usearch_test_header_only PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usearch_test_header_only PRIVATE fmt::fmt yyjson forkunion::header numkong::header)
add_test(NAME usearch_test_header_only COMMAND usearch_test_header_only)

# The engine suite lists the dense per-family test units, sharded per dtype so the instantiation grids build in
# parallel, and keeps the shards of the dtypes `USEARCH_DENSE_DTYPES` compiles; the sequence and shapes engines add one
# unit each. Optimization level comes from the configuration: measured on these shards, -O1 costs as much as -O3 and
# only slows the suite down.
add_executable(
    usearch_test
    main.cpp
    dense.cpp
    utilities/safegraphs_format.cpp
    dense/ram_metric_f64.cpp
    dense/ram_metric_f32.cpp
    dense/ram_indexes.cpp
    dense/ram_index_f32.cpp
    dense/nvme_indexes.cpp
    sequence/ram_index.cpp
    shapes/ram_index.cpp
    geospatial/ram_index.cpp
)
usearch_dense_dtype_sources(
    usearch_test
    BF16 dense/ram_metric_bf16.cpp dense/ram_index_bf16.cpp
    F16 dense/ram_metric_f16.cpp dense/ram_index_f16.cpp
    E5M2 dense/ram_metric_e5m2.cpp dense/ram_index_e5m2.cpp
    E4M3 dense/ram_metric_e4m3.cpp
    E3M2 dense/ram_metric_e3m2.cpp
    E2M3 dense/ram_metric_e2m3.cpp
    I8 dense/ram_metric_i8.cpp dense/ram_index_i8.cpp
    I4 dense/ram_metric_i4.cpp dense/ram_index_i4.cpp
    U8 dense/ram_metric_u8.cpp dense/ram_index_u8.cpp
    U4 dense/ram_metric_u4.cpp
    U1 dense/ram_metric_u1.cpp dense/ram_index_u1.cpp
)
usearch_setup_target(usearch_test HOST)
# The capability names come from the C interface's own probes, compiled in rather than linking the whole library.
target_sources(usearch_test PRIVATE ${PROJECT_SOURCE_DIR}/src/types.cpp)
target_include_directories(usearch_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usearch_test PRIVATE fmt::fmt yyjson forkunion::static stringzilla::header)
add_test(NAME usearch_test COMMAND usearch_test)

# The header-only tier's witness: everything under `utilities/`, `basic/` & `geospatial/` compiles & runs from headers
# alone - its own and the header-only siblings', with no compiled library or yyjson behind it.
add_executable(usearch_test_header_only_witness header_only_witness.cpp)
usearch_setup_target(usearch_test_header_only_witness HOST)
target_link_libraries(usearch_test_header_only_witness PRIVATE forkunion::header numkong::header)
add_test(NAME usearch_test_header_only_witness COMMAND usearch_test_header_only_witness)

# The VRAM test shards compile once per vendor, so each unit sees the SDK facts of the one vendor it composes over: the
# per-dtype shards, the sharded suite and, for the vendors that have them, the modalities' units after `SOURCES`, each
# instantiated for the executor of @p vendor alone, whose `USEARCH_ARCH_<VENDOR>_` is @p architecture_macro and the
# others' are zero, with the `DEFINITIONS` the vendor's kits read. The vendor's `ENTRY` registers with the runner, which
# names none of them. NVIDIA and AMD shards compile host-only through Clang's CUDA or HIP mode, as the vendor's entry
# does, so the vendor's headers define the index members and NumKong keeps the vendor's `NUMKONG_TARGET_*`; the entry,
# or its `DEVICE` entry, compiles once more device-only into the suite's own bundle of the `CODES`, holding the shards'
# kernels and the attends over its device value lookups. Apple's kernels compile from embedded source at first use, so
# its entry is a plain C++ unit.
function (usearch_test_vram_shards vendor architecture_macro)
    cmake_parse_arguments(arg "" "ENTRY;DEVICE" "SOURCES;DEFINITIONS;CODES" ${ARGN})
    set(target_ usearch_test_vram_shards_${vendor})
    set(entry_ ${CMAKE_CURRENT_SOURCE_DIR}/${arg_ENTRY})
    add_library(${target_} OBJECT dense/vram_indexes.cpp)
    usearch_dense_dtype_sources(
        ${target_}
        BF16 dense/vram_index_bf16.cpp
        F16 dense/vram_index_f16.cpp
        E5M2 dense/vram_index_e5m2.cpp
        E4M3 dense/vram_index_e4m3.cpp
        E3M2 dense/vram_index_e3m2.cpp
        E2M3 dense/vram_index_e2m3.cpp
        I8 dense/vram_index_i8.cpp
        I4 dense/vram_index_i4.cpp
        U8 dense/vram_index_u8.cpp
        U4 dense/vram_index_u4.cpp
        U1 dense/vram_index_u1.cpp
    )
    target_sources(${target_} PRIVATE ${arg_SOURCES} ${entry_})
    usearch_setup_target(${target_} HOST)
    target_include_directories(${target_} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    foreach (macro_ IN ITEMS USEARCH_ARCH_CUDA_ USEARCH_ARCH_ROCM_ USEARCH_ARCH_METAL_)
        if (macro_ STREQUAL architecture_macro)
            target_compile_definitions(${target_} PRIVATE ${macro_}=1)
        else ()
            target_compile_definitions(${target_} PRIVATE ${macro_}=0)
        endif ()
    endforeach ()
    target_compile_definitions(${target_} PRIVATE ${arg_DEFINITIONS})
    target_link_libraries(${target_} PRIVATE fmt::fmt yyjson forkunion::static stringzilla::header)
    if (vendor STREQUAL "nvidia" OR vendor STREQUAL "amd")
        set(device_)
        if (arg_DEVICE)
            set(device_ DEVICE ${CMAKE_CURRENT_SOURCE_DIR}/${arg_DEVICE})
        endif ()
        usearch_gpu_bundle(${target_} ${vendor} test_vram_${vendor} ${entry_} ${arg_CODES} ${device_})
        usearch_gpu_host_only(${target_} ${entry_})
    endif ()
    set(usearch_test_vram_shards_
        ${usearch_test_vram_shards_} ${target_}
        PARENT_SCOPE
    )
endfunction ()

# One row per vendor: NVIDIA tests a generation's kits only when every listed architecture runs them, and AMD compiles
# for the listed processors of the CDNA3 class, which CDNA3 and CDNA4 run.
set(usearch_test_vram_shards_)
if (USEARCH_BUILD_CUDA)
    usearch_cuda_capabilities(usearch_test_cuda_capabilities_ "${USEARCH_CUDA_ARCHITECTURES}")
    usearch_cuda_codes(usearch_test_cuda_codes_ "${USEARCH_CUDA_ARCHITECTURES}")
    usearch_test_vram_shards(
        nvidia USEARCH_ARCH_CUDA_
        ENTRY dense/vram_nvidia.cu
        SOURCES sequence/vram_index.cpp shapes/vram_index.cpp geospatial/vram_index.cpp
        DEFINITIONS ${usearch_test_cuda_capabilities_}
        CODES ${usearch_test_cuda_codes_}
    )
endif ()
if (USEARCH_BUILD_ROCM)
    usearch_rocm_codes(usearch_test_rocm_codes_ gfx942 gfx950)
    usearch_test_vram_shards(
        amd USEARCH_ARCH_ROCM_
        ENTRY dense/vram_amd.hip
        DEVICE dense/vram_kernels_amd.hip
        DEFINITIONS "NUMKONG_TARGET_CDNA3=1"
        CODES ${usearch_test_rocm_codes_}
    )
endif ()
if (USEARCH_BUILD_METAL)
    usearch_test_vram_shards(apple USEARCH_ARCH_METAL_ ENTRY dense/vram_apple.cpp)
endif ()

# The one VRAM runner links the library too, so it reaches every class variant it compiled through the C interface.
add_executable(usearch_test_vram dense_vram.cpp)
usearch_setup_target(usearch_test_vram HOST)
target_link_libraries(
    usearch_test_vram PRIVATE ${usearch_test_vram_shards_} fmt::fmt yyjson forkunion::static
                              ${USEARCH_PUNNED_LIBRARIES} ${CMAKE_DL_LIBS}
)
add_test(NAME usearch_test_vram COMMAND usearch_test_vram)
