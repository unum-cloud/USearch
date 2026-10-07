# Every benchmark is named after the index type it drives, and its dtype shards, ordered by shrinking dynamic range,
# each carry one stored dtype's instantiation grid so the grids build in parallel.
add_executable(
    usearch_bench_dense_ram_index
    dense_ram_index.cpp
    dense/ram_index_f32.cpp
    dense/ram_index_bf16.cpp
    dense/ram_index_f16.cpp
    dense/ram_index_e5m2.cpp
    dense/ram_index_e4m3.cpp
    dense/ram_index_e3m2.cpp
    dense/ram_index_e2m3.cpp
    dense/ram_index_i8.cpp
    dense/ram_index_i4.cpp
    dense/ram_index_u8.cpp
    dense/ram_index_u4.cpp
    dense/ram_index_u1.cpp
)
usearch_setup_target(usearch_bench_dense_ram_index HOST)
target_include_directories(usearch_bench_dense_ram_index PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usearch_bench_dense_ram_index PRIVATE fmt::fmt yyjson forkunion::static stringzilla::static)

add_executable(
    usearch_bench_dense_ram_indexes
    dense_ram_indexes.cpp
    dense/ram_indexes_f32.cpp
    dense/ram_indexes_bf16.cpp
    dense/ram_indexes_f16.cpp
    dense/ram_indexes_e5m2.cpp
    dense/ram_indexes_e4m3.cpp
    dense/ram_indexes_e3m2.cpp
    dense/ram_indexes_e2m3.cpp
    dense/ram_indexes_i8.cpp
    dense/ram_indexes_i4.cpp
    dense/ram_indexes_u8.cpp
    dense/ram_indexes_u4.cpp
    dense/ram_indexes_u1.cpp
)
usearch_setup_target(usearch_bench_dense_ram_indexes HOST)
target_include_directories(usearch_bench_dense_ram_indexes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usearch_bench_dense_ram_indexes PRIVATE fmt::fmt yyjson forkunion::static stringzilla::static)

# The kernel microbench times the distance kits alone, one row per kernel, dtype, capability tier and shape.
add_executable(usearch_bench_dense_ram_metric dense/ram_metric.cpp)
usearch_setup_target(usearch_bench_dense_ram_metric HOST)
target_include_directories(usearch_bench_dense_ram_metric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usearch_bench_dense_ram_metric PRIVATE fmt::fmt)

# The CUDA bench skips `usearch_setup_target`: its host-compiler flags don't survive the CUDA driver, so the target
# wires the interface includes directly.
if (USEARCH_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(
        usearch_bench_dense_vram_index
        dense_vram_index.cu
        dense/vram_index_bf16.cu
        dense/vram_index_f16.cu
        dense/vram_index_e5m2.cu
        dense/vram_index_e4m3.cu
        dense/vram_index_e3m2.cu
        dense/vram_index_e2m3.cu
        dense/vram_index_i8.cu
        dense/vram_index_i4.cu
        dense/vram_index_u8.cu
        dense/vram_index_u4.cu
        dense/vram_index_u1.cu
    )
    target_include_directories(
        usearch_bench_dense_vram_index PRIVATE $<TARGET_PROPERTY:${USEARCH_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
                                               ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(
        usearch_bench_dense_vram_index
        PROPERTIES CXX_STANDARD 20
                   CUDA_STANDARD 20
                   CUDA_ARCHITECTURES "${USEARCH_CUDA_ARCHITECTURES}"
    )
    target_compile_options(usearch_bench_dense_vram_index PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    # A generation's kits join the bench only when every listed architecture runs them.
    usearch_cuda_capabilities(bench_cuda_capabilities_ "${USEARCH_CUDA_ARCHITECTURES}")
    target_compile_definitions(usearch_bench_dense_vram_index PRIVATE ${bench_cuda_capabilities_})
    target_link_libraries(
        usearch_bench_dense_vram_index PRIVATE fmt::fmt yyjson CUDA::cuda_driver forkunion::static stringzilla::static
    )
endif ()
