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
# per-dtype shards, the sharded suite and, for the vendors that have them, the modalities' units, each instantiated for
# the executor of @p vendor alone, whose `USEARCH_ARCH_<VENDOR>_` is @p architecture_macro and the others' are zero.
# NVIDIA and AMD shards compile host-only through Clang's CUDA or HIP mode, as the vendor's entry does, so the vendor's
# headers define the index members and NumKong keeps the vendor's `NUMKONG_TARGET_*`; the caller adds the entry and the
# flags. Each carried vendor's entry registers with the runner, which names none of them.
function (usearch_test_vram_shards vendor architecture_macro)
    set(target_ usearch_test_vram_shards_${vendor})
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
    usearch_setup_target(${target_} HOST)
    target_include_directories(${target_} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    foreach (macro_ IN ITEMS USEARCH_ARCH_CUDA_ USEARCH_ARCH_ROCM_ USEARCH_ARCH_METAL_)
        if (macro_ STREQUAL architecture_macro)
            target_compile_definitions(${target_} PRIVATE ${macro_}=1)
        else ()
            target_compile_definitions(${target_} PRIVATE ${macro_}=0)
        endif ()
    endforeach ()
    target_link_libraries(${target_} PRIVATE fmt::fmt yyjson forkunion::static stringzilla::header)
endfunction ()

set(usearch_test_vram_shards_)

# NVIDIA's entry compiles once more with NVCC device-only, into the suite's own fatbin of the shards' kernels and the
# attends over its device value lookups, testing a generation's kits only when every listed architecture runs them.
if (USEARCH_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(ASM)
    set(usearch_test_vram_nvidia_entry_ ${CMAKE_CURRENT_SOURCE_DIR}/dense/vram_nvidia.cu)
    usearch_test_vram_shards(nvidia USEARCH_ARCH_CUDA_)
    target_sources(
        usearch_test_vram_shards_nvidia PRIVATE sequence/vram_index.cpp shapes/vram_index.cpp geospatial/vram_index.cpp
                                                ${usearch_test_vram_nvidia_entry_}
    )
    usearch_cuda_capabilities(test_cuda_capabilities_ "${USEARCH_CUDA_ARCHITECTURES}")
    target_compile_definitions(usearch_test_vram_shards_nvidia PRIVATE ${test_cuda_capabilities_})
    usearch_cuda_codes(test_cuda_codes_ "${USEARCH_CUDA_ARCHITECTURES}")
    usearch_gpu_bundle(
        usearch_test_vram_shards_nvidia nvidia test_vram_nvidia ${usearch_test_vram_nvidia_entry_} ${test_cuda_codes_}
    )
    usearch_gpu_host_only(usearch_test_vram_shards_nvidia ${usearch_test_vram_nvidia_entry_})
    list(APPEND usearch_test_vram_shards_ usearch_test_vram_shards_nvidia)
endif ()

# AMD's entry compiles host-only, and its device entry, which includes it, with ROCm's Clang device-only, into the
# suite's own code object of the shards' kernels and the attends over its device value lookups, for the listed
# processors of the CDNA3 class, which CDNA3 and CDNA4 run. The Apple kernels compile from embedded source at first use.
if (USEARCH_BUILD_ROCM)
    enable_language(ASM)
    set(usearch_test_vram_amd_entry_ ${CMAKE_CURRENT_SOURCE_DIR}/dense/vram_amd.hip)
    usearch_test_vram_shards(amd USEARCH_ARCH_ROCM_)
    target_sources(usearch_test_vram_shards_amd PRIVATE ${usearch_test_vram_amd_entry_})
    target_compile_definitions(usearch_test_vram_shards_amd PRIVATE "NUMKONG_TARGET_CDNA3=1")
    set(test_rocm_processors_)
    foreach (processor_ IN ITEMS gfx942 gfx950)
        if (processor_ IN_LIST USEARCH_ROCM_ARCHITECTURES)
            list(APPEND test_rocm_processors_ --offload-arch=${processor_})
        endif ()
    endforeach ()
    usearch_gpu_bundle(
        usearch_test_vram_shards_amd
        amd
        test_vram_amd
        ${usearch_test_vram_amd_entry_}
        ${test_rocm_processors_}
        DEVICE
        ${CMAKE_CURRENT_SOURCE_DIR}/dense/vram_kernels_amd.hip
    )
    usearch_gpu_host_only(usearch_test_vram_shards_amd ${usearch_test_vram_amd_entry_})
    list(APPEND usearch_test_vram_shards_ usearch_test_vram_shards_amd)
endif ()

if (USEARCH_BUILD_METAL)
    usearch_test_vram_shards(apple USEARCH_ARCH_METAL_)
    target_sources(usearch_test_vram_shards_apple PRIVATE dense/vram_apple.cpp)
    list(APPEND usearch_test_vram_shards_ usearch_test_vram_shards_apple)
endif ()

# The one VRAM runner links the library too, so it reaches every class variant it compiled through the C interface.
add_executable(usearch_test_vram dense_vram.cpp)
usearch_setup_target(usearch_test_vram HOST)
target_link_libraries(
    usearch_test_vram PRIVATE ${usearch_test_vram_shards_} fmt::fmt yyjson forkunion::static
                              ${USEARCH_PUNNED_LIBRARIES} ${CMAKE_DL_LIBS}
)
add_test(NAME usearch_test_vram COMMAND usearch_test_vram)
