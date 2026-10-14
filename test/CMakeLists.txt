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

# The one VRAM runner: one object library compiles the per-dtype test shards once, each instantiating every vendor its
# `USEARCH_ARCH_<VENDOR>_` carries, and each carried vendor's entry registers with the runner, which names none of them.
# The library links in too, so the runner reaches every class arm it compiled through the C interface.
add_library(usearch_test_vram_shards OBJECT dense/vram_indexes.cpp)
usearch_dense_dtype_sources(
    usearch_test_vram_shards
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
usearch_setup_target(usearch_test_vram_shards HOST)
target_include_directories(usearch_test_vram_shards PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(
    usearch_test_vram_shards
    PRIVATE "USEARCH_ARCH_CUDA_=$<BOOL:${USEARCH_BUILD_CUDA}>" "USEARCH_ARCH_ROCM_=$<BOOL:${USEARCH_BUILD_ROCM}>"
            "USEARCH_ARCH_METAL_=$<BOOL:${USEARCH_BUILD_METAL}>"
)
target_link_libraries(usearch_test_vram_shards PRIVATE fmt::fmt yyjson forkunion::static)
add_executable(usearch_test_vram dense_vram.cpp)
usearch_setup_target(usearch_test_vram HOST)
target_link_libraries(
    usearch_test_vram PRIVATE usearch_test_vram_shards fmt::fmt yyjson forkunion::static ${USEARCH_PUNNED_LIBRARIES}
                              ${CMAKE_DL_LIBS}
)
add_test(NAME usearch_test_vram COMMAND usearch_test_vram)

# NVIDIA's entry compiles once more with NVCC device-only, into the suite's own fatbin of the shards' kernels and the
# attends over its device value lookups, testing a generation's kits only when every listed architecture runs them.
if (USEARCH_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(ASM)
    set(usearch_test_vram_nvidia_entry_ ${CMAKE_CURRENT_SOURCE_DIR}/dense/vram_nvidia.cu)
    target_sources(usearch_test_vram_shards PRIVATE ${usearch_test_vram_nvidia_entry_})
    usearch_cuda_capabilities(test_cuda_capabilities_ "${USEARCH_CUDA_ARCHITECTURES}")
    target_compile_definitions(usearch_test_vram_shards PRIVATE ${test_cuda_capabilities_})
    usearch_cuda_codes(test_cuda_codes_ "${USEARCH_CUDA_ARCHITECTURES}")
    usearch_gpu_bundle(
        usearch_test_vram_shards nvidia test_vram_nvidia ${usearch_test_vram_nvidia_entry_} ${test_cuda_codes_}
    )
endif ()

# The AMD shards run over the library's CDNA3 floor unit, whose code object CDNA3 and CDNA4 run, and the Apple kernels
# compile from embedded source at first use.
if (USEARCH_BUILD_ROCM)
    target_sources(usearch_test_vram_shards PRIVATE dense/vram_amd.cpp)
endif ()
if (USEARCH_BUILD_METAL)
    target_sources(usearch_test_vram_shards PRIVATE dense/vram_apple.cpp)
endif ()
