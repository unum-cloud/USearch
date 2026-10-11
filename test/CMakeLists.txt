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

# The dense suite lists the per-family test units, sharded per dtype so the instantiation grids build in parallel, and
# keeps the shards of the dtypes `USEARCH_DENSE_DTYPES` compiles. Optimization level comes from the configuration:
# measured on these shards, -O1 costs as much as -O3 and only slows the suite down.
add_executable(
    usearch_test
    main.cpp
    dense.cpp
    utilities/safegraphs_format.cpp
    dense/ram_metric_f64.cpp
    dense/ram_metric_f32.cpp
    dense/ram_metric_bf16.cpp
    dense/ram_metric_f16.cpp
    dense/ram_metric_e5m2.cpp
    dense/ram_metric_e4m3.cpp
    dense/ram_metric_e3m2.cpp
    dense/ram_metric_e2m3.cpp
    dense/ram_metric_i8.cpp
    dense/ram_metric_i4.cpp
    dense/ram_metric_u8.cpp
    dense/ram_metric_u4.cpp
    dense/ram_metric_u1.cpp
    dense/ram_indexes.cpp
    dense/ram_index_f32.cpp
    dense/ram_index_bf16.cpp
    dense/ram_index_f16.cpp
    dense/ram_index_e5m2.cpp
    dense/ram_index_i8.cpp
    dense/ram_index_i4.cpp
    dense/ram_index_u8.cpp
    dense/ram_index_u1.cpp
    dense/nvme_indexes.cpp
)
usearch_keep_dense_dtypes(usearch_test)
usearch_setup_target(usearch_test HOST)
# The capability names come from the C interface's own probes, compiled in rather than linking the whole library.
target_sources(usearch_test PRIVATE ${PROJECT_SOURCE_DIR}/src/types.cpp)
target_include_directories(usearch_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usearch_test PRIVATE fmt::fmt yyjson forkunion::static)
add_test(NAME usearch_test COMMAND usearch_test)

# The header-only tier's witness: everything under `utilities/` & `basic/` compiles & runs with the standard library
# alone - only -Iinclude, no submodule paths, no third-party targets.
add_executable(usearch_test_header_only_witness header_only_witness.cpp)
target_include_directories(usearch_test_header_only_witness PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(usearch_test_header_only_witness PROPERTIES CXX_STANDARD 20)
add_test(NAME usearch_test_header_only_witness COMMAND usearch_test_header_only_witness)

# The one VRAM runner: every vendor compiles the vendor-neutral residency shards into an object library of its own, with
# its `USEARCH_ARCH_<VENDOR>_` naming the cell they compose over, and registers its entry with the runner, which links
# whichever vendors this build carries and names none of them.
set(usearch_test_vram_shards_
    dense/vram_index_bf16.cpp
    dense/vram_index_f16.cpp
    dense/vram_index_e5m2.cpp
    dense/vram_index_e4m3.cpp
    dense/vram_index_e3m2.cpp
    dense/vram_index_e2m3.cpp
    dense/vram_index_i8.cpp
    dense/vram_index_i4.cpp
    dense/vram_index_u8.cpp
    dense/vram_index_u4.cpp
    dense/vram_index_u1.cpp
    dense/vram_indexes.cpp
)
add_executable(usearch_test_vram dense_vram.cpp ${PROJECT_SOURCE_DIR}/src/types.cpp)
usearch_keep_dense_dtypes(usearch_test_vram)
usearch_setup_target(usearch_test_vram HOST)
target_link_libraries(usearch_test_vram PRIVATE fmt::fmt yyjson forkunion::static)
add_test(NAME usearch_test_vram COMMAND usearch_test_vram)

# The NVIDIA object compiles the shards with the host compiler over the plain-C++ faces, and its entry with NVCC, since
# device attention looks values up on the device and the cells' kernels instantiate beside it.
if (USEARCH_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(usearch_test_vram_nvidia OBJECT ${usearch_test_vram_shards_} dense/vram_nvidia.cu)
    usearch_keep_dense_dtypes(usearch_test_vram_nvidia)
    usearch_setup_target(usearch_test_vram_nvidia HOST)
    target_include_directories(usearch_test_vram_nvidia PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(
        usearch_test_vram_nvidia
        PROPERTIES CXX_STANDARD 20
                   CUDA_STANDARD 20
                   CUDA_ARCHITECTURES "${USEARCH_CUDA_ARCHITECTURES}"
    )
    # The device units test a generation's kits only when every listed architecture runs them.
    usearch_cuda_capabilities(test_cuda_capabilities_ "${USEARCH_CUDA_ARCHITECTURES}")
    target_compile_definitions(usearch_test_vram_nvidia PRIVATE "USEARCH_ARCH_CUDA_=1" ${test_cuda_capabilities_})
    target_compile_options(usearch_test_vram_nvidia PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries(
        usearch_test_vram_nvidia PRIVATE fmt::fmt yyjson forkunion::static numkong::header CUDA::cudart_static
    )
    target_compile_definitions(usearch_test_vram PRIVATE "USEARCH_ARCH_CUDA_=1")
    target_link_libraries(usearch_test_vram PRIVATE usearch_test_vram_nvidia CUDA::cudart_static)
endif ()

# The Apple object compiles everything with the host compiler; its kernels compile from embedded source at first use.
if (USEARCH_BUILD_METAL)
    add_library(usearch_test_vram_apple OBJECT ${usearch_test_vram_shards_} dense/vram_apple.cpp)
    usearch_keep_dense_dtypes(usearch_test_vram_apple)
    usearch_setup_target(usearch_test_vram_apple HOST)
    target_include_directories(usearch_test_vram_apple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(usearch_test_vram_apple PRIVATE "USEARCH_ARCH_METAL_=1")
    target_link_libraries(usearch_test_vram_apple PRIVATE fmt::fmt yyjson forkunion::static)
    target_compile_definitions(usearch_test_vram PRIVATE "USEARCH_ARCH_METAL_=1")
    target_link_libraries(
        usearch_test_vram PRIVATE usearch_test_vram_apple "-framework Metal" "-framework Foundation" objc
    )
endif ()
