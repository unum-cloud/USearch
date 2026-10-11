# Every benchmark is named after the residency, the arity and the family it drives, and its dtype shards, ordered by
# shrinking dynamic range, each carry one stored dtype's instantiation grid so the grids build in parallel. The
# `_algebra` binaries measure graph products and graph attention over the same builds, so they take the dot kernel's
# dtypes alone.
add_executable(
    usearch_bench_dense_ram_index_search
    dense_ram_index_search.cpp
    dense/ram_index_search_f32.cpp
    dense/ram_index_search_bf16.cpp
    dense/ram_index_search_f16.cpp
    dense/ram_index_search_e5m2.cpp
    dense/ram_index_search_e4m3.cpp
    dense/ram_index_search_e3m2.cpp
    dense/ram_index_search_e2m3.cpp
    dense/ram_index_search_i8.cpp
    dense/ram_index_search_i4.cpp
    dense/ram_index_search_u8.cpp
    dense/ram_index_search_u4.cpp
    dense/ram_index_search_u1.cpp
)
add_executable(
    usearch_bench_dense_ram_indexes_search
    dense_ram_indexes_search.cpp
    dense/ram_indexes_search_f32.cpp
    dense/ram_indexes_search_bf16.cpp
    dense/ram_indexes_search_f16.cpp
    dense/ram_indexes_search_e5m2.cpp
    dense/ram_indexes_search_e4m3.cpp
    dense/ram_indexes_search_e3m2.cpp
    dense/ram_indexes_search_e2m3.cpp
    dense/ram_indexes_search_i8.cpp
    dense/ram_indexes_search_i4.cpp
    dense/ram_indexes_search_u8.cpp
    dense/ram_indexes_search_u4.cpp
    dense/ram_indexes_search_u1.cpp
)
add_executable(
    usearch_bench_dense_ram_index_algebra
    dense_ram_index_algebra.cpp
    dense/ram_index_algebra_bf16.cpp
    dense/ram_index_algebra_f16.cpp
    dense/ram_index_algebra_e5m2.cpp
    dense/ram_index_algebra_e4m3.cpp
    dense/ram_index_algebra_e3m2.cpp
    dense/ram_index_algebra_e2m3.cpp
    dense/ram_index_algebra_i8.cpp
    dense/ram_index_algebra_i4.cpp
    dense/ram_index_algebra_u8.cpp
    dense/ram_index_algebra_u4.cpp
)
add_executable(
    usearch_bench_dense_ram_indexes_algebra
    dense_ram_indexes_algebra.cpp
    dense/ram_indexes_algebra_bf16.cpp
    dense/ram_indexes_algebra_f16.cpp
    dense/ram_indexes_algebra_e5m2.cpp
    dense/ram_indexes_algebra_e4m3.cpp
    dense/ram_indexes_algebra_e3m2.cpp
    dense/ram_indexes_algebra_e2m3.cpp
    dense/ram_indexes_algebra_i8.cpp
    dense/ram_indexes_algebra_i4.cpp
    dense/ram_indexes_algebra_u8.cpp
    dense/ram_indexes_algebra_u4.cpp
)
# The NVMe benches carry no f32 shard, since f32 is never a stored dtype.
add_executable(
    usearch_bench_dense_nvme_indexes_search
    dense_nvme_indexes_search.cpp
    dense/nvme_indexes_search_bf16.cpp
    dense/nvme_indexes_search_f16.cpp
    dense/nvme_indexes_search_e5m2.cpp
    dense/nvme_indexes_search_e4m3.cpp
    dense/nvme_indexes_search_e3m2.cpp
    dense/nvme_indexes_search_e2m3.cpp
    dense/nvme_indexes_search_i8.cpp
    dense/nvme_indexes_search_i4.cpp
    dense/nvme_indexes_search_u8.cpp
    dense/nvme_indexes_search_u4.cpp
    dense/nvme_indexes_search_u1.cpp
)
add_executable(
    usearch_bench_dense_nvme_indexes_algebra
    dense_nvme_indexes_algebra.cpp
    dense/nvme_indexes_algebra_bf16.cpp
    dense/nvme_indexes_algebra_f16.cpp
    dense/nvme_indexes_algebra_e5m2.cpp
    dense/nvme_indexes_algebra_e4m3.cpp
    dense/nvme_indexes_algebra_e3m2.cpp
    dense/nvme_indexes_algebra_e2m3.cpp
    dense/nvme_indexes_algebra_i8.cpp
    dense/nvme_indexes_algebra_i4.cpp
    dense/nvme_indexes_algebra_u8.cpp
    dense/nvme_indexes_algebra_u4.cpp
)
foreach (
    bench_target_ IN
    ITEMS usearch_bench_dense_ram_index_search usearch_bench_dense_ram_indexes_search
          usearch_bench_dense_ram_index_algebra usearch_bench_dense_ram_indexes_algebra
          usearch_bench_dense_nvme_indexes_search usearch_bench_dense_nvme_indexes_algebra
)
    usearch_keep_dense_dtypes(${bench_target_})
    usearch_setup_target(${bench_target_} HOST)
    target_include_directories(${bench_target_} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${bench_target_} PRIVATE fmt::fmt yyjson forkunion::static stringzilla::static)
endforeach ()

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
        usearch_bench_dense_vram_index_search
        dense_vram_index_search.cu
        dense/vram_index_search_bf16.cu
        dense/vram_index_search_f16.cu
        dense/vram_index_search_e5m2.cu
        dense/vram_index_search_e4m3.cu
        dense/vram_index_search_e3m2.cu
        dense/vram_index_search_e2m3.cu
        dense/vram_index_search_i8.cu
        dense/vram_index_search_i4.cu
        dense/vram_index_search_u8.cu
        dense/vram_index_search_u4.cu
        dense/vram_index_search_u1.cu
    )
    add_executable(
        usearch_bench_dense_vram_indexes_search
        dense_vram_indexes_search.cu
        dense/vram_indexes_search_bf16.cu
        dense/vram_indexes_search_f16.cu
        dense/vram_indexes_search_e5m2.cu
        dense/vram_indexes_search_e4m3.cu
        dense/vram_indexes_search_e3m2.cu
        dense/vram_indexes_search_e2m3.cu
        dense/vram_indexes_search_i8.cu
        dense/vram_indexes_search_i4.cu
        dense/vram_indexes_search_u8.cu
        dense/vram_indexes_search_u4.cu
        dense/vram_indexes_search_u1.cu
    )
    add_executable(
        usearch_bench_dense_vram_index_algebra
        dense_vram_index_algebra.cu
        dense/vram_index_algebra_bf16.cu
        dense/vram_index_algebra_f16.cu
        dense/vram_index_algebra_e5m2.cu
        dense/vram_index_algebra_e4m3.cu
        dense/vram_index_algebra_e3m2.cu
        dense/vram_index_algebra_e2m3.cu
        dense/vram_index_algebra_i8.cu
        dense/vram_index_algebra_i4.cu
        dense/vram_index_algebra_u8.cu
        dense/vram_index_algebra_u4.cu
    )
    add_executable(
        usearch_bench_dense_vram_indexes_algebra
        dense_vram_indexes_algebra.cu
        dense/vram_indexes_algebra_bf16.cu
        dense/vram_indexes_algebra_f16.cu
        dense/vram_indexes_algebra_e5m2.cu
        dense/vram_indexes_algebra_e4m3.cu
        dense/vram_indexes_algebra_e3m2.cu
        dense/vram_indexes_algebra_e2m3.cu
        dense/vram_indexes_algebra_i8.cu
        dense/vram_indexes_algebra_i4.cu
        dense/vram_indexes_algebra_u8.cu
        dense/vram_indexes_algebra_u4.cu
    )
    # A generation's kits join the bench only when every listed architecture runs them.
    usearch_cuda_capabilities(bench_cuda_capabilities_ "${USEARCH_CUDA_ARCHITECTURES}")
    foreach (bench_target_ IN ITEMS usearch_bench_dense_vram_index_search usearch_bench_dense_vram_indexes_search
                                    usearch_bench_dense_vram_index_algebra usearch_bench_dense_vram_indexes_algebra
    )
        usearch_keep_dense_dtypes(${bench_target_})
        target_include_directories(
            ${bench_target_} PRIVATE $<TARGET_PROPERTY:${USEARCH_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
                                     ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_target_properties(
            ${bench_target_}
            PROPERTIES CXX_STANDARD 20
                       CUDA_STANDARD 20
                       CUDA_ARCHITECTURES "${USEARCH_CUDA_ARCHITECTURES}"
        )
        target_compile_options(${bench_target_} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        target_compile_definitions(${bench_target_} PRIVATE ${bench_cuda_capabilities_})
        target_link_libraries(
            ${bench_target_} PRIVATE fmt::fmt yyjson CUDA::cuda_driver forkunion::static stringzilla::static
                                     numkong::header
        )
    endforeach ()
endif ()
