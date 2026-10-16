# Every benchmark is named after the residency, the arity and the family it drives, and its dtype shards, ordered by
# shrinking dynamic range, each carry one stored dtype's instantiation grid so the grids build in parallel. The
# `_algebra` binaries measure graph products and graph attention over the same builds, so they take the dot kernel's
# dtypes alone, and the NVMe benches carry no f32 shard, since f32 is never a stored dtype.

# A dense bench, @p target on @p backend, from the units before the first dtype keyword, the entry first, and the dtype
# shards after them, as `usearch_dense_dtype_sources` takes them. A CPU bench inlines the kits its host flags pick. An
# NVIDIA bench compiles host-only through Clang's CUDA mode, shards and entry unit alike, and the entry once more with
# NVCC device-only, into the binary's own fatbin of every listed architecture's kernels; a generation's kits join it
# only when every listed architecture runs them, and its device listing names its classes through the C interface's own
# probes, compiled in as the tests do.
function (usearch_dense_bench target backend)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "ENTRY" "UNITS;SHARDS")
    add_executable(${target} ${arg_ENTRY} ${arg_UNITS})
    usearch_dense_dtype_sources(${target} ${arg_SHARDS})
    usearch_setup_target(${target} HOST)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE fmt::fmt yyjson forkunion::static stringzilla::static)
    if (backend STREQUAL "nvidia")
        usearch_cuda_capabilities(capabilities_ "${USEARCH_CUDA_ARCHITECTURES}")
        usearch_cuda_codes(codes_ "${USEARCH_CUDA_ARCHITECTURES}")
        target_sources(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src/types.cpp)
        target_compile_definitions(${target} PRIVATE "USEARCH_ARCH_CUDA_=1" ${capabilities_})
        target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
        usearch_gpu_bundle(${target} nvidia bench_vram_nvidia ${CMAKE_CURRENT_SOURCE_DIR}/${arg_ENTRY} ${codes_})
        usearch_gpu_host_only(${target} ${CMAKE_CURRENT_SOURCE_DIR}/${arg_ENTRY})
    endif ()
endfunction ()

usearch_dense_bench(
    usearch_bench_dense_ram_index_search cpu
    ENTRY dense_ram_index_search.cpp
    UNITS dense/ram_index_search_f32.cpp
    SHARDS BF16 dense/ram_index_search_bf16.cpp
           F16 dense/ram_index_search_f16.cpp
           E5M2 dense/ram_index_search_e5m2.cpp
           E4M3 dense/ram_index_search_e4m3.cpp
           E3M2 dense/ram_index_search_e3m2.cpp
           E2M3 dense/ram_index_search_e2m3.cpp
           I8 dense/ram_index_search_i8.cpp
           I4 dense/ram_index_search_i4.cpp
           U8 dense/ram_index_search_u8.cpp
           U4 dense/ram_index_search_u4.cpp
           U1 dense/ram_index_search_u1.cpp
)
usearch_dense_bench(
    usearch_bench_dense_ram_indexes_search cpu
    ENTRY dense_ram_indexes_search.cpp
    UNITS dense/ram_indexes_search_f32.cpp
    SHARDS BF16 dense/ram_indexes_search_bf16.cpp
           F16 dense/ram_indexes_search_f16.cpp
           E5M2 dense/ram_indexes_search_e5m2.cpp
           E4M3 dense/ram_indexes_search_e4m3.cpp
           E3M2 dense/ram_indexes_search_e3m2.cpp
           E2M3 dense/ram_indexes_search_e2m3.cpp
           I8 dense/ram_indexes_search_i8.cpp
           I4 dense/ram_indexes_search_i4.cpp
           U8 dense/ram_indexes_search_u8.cpp
           U4 dense/ram_indexes_search_u4.cpp
           U1 dense/ram_indexes_search_u1.cpp
)
usearch_dense_bench(
    usearch_bench_dense_ram_index_algebra cpu
    ENTRY dense_ram_index_algebra.cpp
    SHARDS BF16 dense/ram_index_algebra_bf16.cpp
           F16 dense/ram_index_algebra_f16.cpp
           E5M2 dense/ram_index_algebra_e5m2.cpp
           E4M3 dense/ram_index_algebra_e4m3.cpp
           E3M2 dense/ram_index_algebra_e3m2.cpp
           E2M3 dense/ram_index_algebra_e2m3.cpp
           I8 dense/ram_index_algebra_i8.cpp
           I4 dense/ram_index_algebra_i4.cpp
           U8 dense/ram_index_algebra_u8.cpp
           U4 dense/ram_index_algebra_u4.cpp
           U1 dense/ram_index_algebra_u1.cpp
)
usearch_dense_bench(
    usearch_bench_dense_ram_indexes_algebra cpu
    ENTRY dense_ram_indexes_algebra.cpp
    SHARDS BF16 dense/ram_indexes_algebra_bf16.cpp
           F16 dense/ram_indexes_algebra_f16.cpp
           E5M2 dense/ram_indexes_algebra_e5m2.cpp
           E4M3 dense/ram_indexes_algebra_e4m3.cpp
           E3M2 dense/ram_indexes_algebra_e3m2.cpp
           E2M3 dense/ram_indexes_algebra_e2m3.cpp
           I8 dense/ram_indexes_algebra_i8.cpp
           I4 dense/ram_indexes_algebra_i4.cpp
           U8 dense/ram_indexes_algebra_u8.cpp
           U4 dense/ram_indexes_algebra_u4.cpp
           U1 dense/ram_indexes_algebra_u1.cpp
)
usearch_dense_bench(
    usearch_bench_dense_nvme_indexes_search cpu
    ENTRY dense_nvme_indexes_search.cpp
    SHARDS BF16 dense/nvme_indexes_search_bf16.cpp
           F16 dense/nvme_indexes_search_f16.cpp
           E5M2 dense/nvme_indexes_search_e5m2.cpp
           E4M3 dense/nvme_indexes_search_e4m3.cpp
           E3M2 dense/nvme_indexes_search_e3m2.cpp
           E2M3 dense/nvme_indexes_search_e2m3.cpp
           I8 dense/nvme_indexes_search_i8.cpp
           I4 dense/nvme_indexes_search_i4.cpp
           U8 dense/nvme_indexes_search_u8.cpp
           U4 dense/nvme_indexes_search_u4.cpp
           U1 dense/nvme_indexes_search_u1.cpp
)
usearch_dense_bench(
    usearch_bench_dense_nvme_indexes_algebra cpu
    ENTRY dense_nvme_indexes_algebra.cpp
    SHARDS BF16 dense/nvme_indexes_algebra_bf16.cpp
           F16 dense/nvme_indexes_algebra_f16.cpp
           E5M2 dense/nvme_indexes_algebra_e5m2.cpp
           E4M3 dense/nvme_indexes_algebra_e4m3.cpp
           E3M2 dense/nvme_indexes_algebra_e3m2.cpp
           E2M3 dense/nvme_indexes_algebra_e2m3.cpp
           I8 dense/nvme_indexes_algebra_i8.cpp
           I4 dense/nvme_indexes_algebra_i4.cpp
           U8 dense/nvme_indexes_algebra_u8.cpp
           U4 dense/nvme_indexes_algebra_u4.cpp
           U1 dense/nvme_indexes_algebra_u1.cpp
)
if (USEARCH_BUILD_CUDA)
    usearch_dense_bench(
        usearch_bench_dense_vram_index_search nvidia
        ENTRY dense_vram_index_search.cu
        SHARDS BF16 dense/vram_index_search_bf16.cpp
               F16 dense/vram_index_search_f16.cpp
               E5M2 dense/vram_index_search_e5m2.cpp
               E4M3 dense/vram_index_search_e4m3.cpp
               E3M2 dense/vram_index_search_e3m2.cpp
               E2M3 dense/vram_index_search_e2m3.cpp
               I8 dense/vram_index_search_i8.cpp
               I4 dense/vram_index_search_i4.cpp
               U8 dense/vram_index_search_u8.cpp
               U4 dense/vram_index_search_u4.cpp
               U1 dense/vram_index_search_u1.cpp
    )
    usearch_dense_bench(
        usearch_bench_dense_vram_indexes_search nvidia
        ENTRY dense_vram_indexes_search.cu
        SHARDS BF16 dense/vram_indexes_search_bf16.cpp
               F16 dense/vram_indexes_search_f16.cpp
               E5M2 dense/vram_indexes_search_e5m2.cpp
               E4M3 dense/vram_indexes_search_e4m3.cpp
               E3M2 dense/vram_indexes_search_e3m2.cpp
               E2M3 dense/vram_indexes_search_e2m3.cpp
               I8 dense/vram_indexes_search_i8.cpp
               I4 dense/vram_indexes_search_i4.cpp
               U8 dense/vram_indexes_search_u8.cpp
               U4 dense/vram_indexes_search_u4.cpp
               U1 dense/vram_indexes_search_u1.cpp
    )
    usearch_dense_bench(
        usearch_bench_dense_vram_index_algebra nvidia
        ENTRY dense_vram_index_algebra.cu
        SHARDS BF16 dense/vram_index_algebra_bf16.cpp
               F16 dense/vram_index_algebra_f16.cpp
               E5M2 dense/vram_index_algebra_e5m2.cpp
               E4M3 dense/vram_index_algebra_e4m3.cpp
               E3M2 dense/vram_index_algebra_e3m2.cpp
               E2M3 dense/vram_index_algebra_e2m3.cpp
               I8 dense/vram_index_algebra_i8.cpp
               I4 dense/vram_index_algebra_i4.cpp
               U8 dense/vram_index_algebra_u8.cpp
               U4 dense/vram_index_algebra_u4.cpp
               U1 dense/vram_index_algebra_u1.cpp
    )
    usearch_dense_bench(
        usearch_bench_dense_vram_indexes_algebra nvidia
        ENTRY dense_vram_indexes_algebra.cu
        SHARDS BF16 dense/vram_indexes_algebra_bf16.cpp
               F16 dense/vram_indexes_algebra_f16.cpp
               E5M2 dense/vram_indexes_algebra_e5m2.cpp
               E4M3 dense/vram_indexes_algebra_e4m3.cpp
               E3M2 dense/vram_indexes_algebra_e3m2.cpp
               E2M3 dense/vram_indexes_algebra_e2m3.cpp
               I8 dense/vram_indexes_algebra_i8.cpp
               I4 dense/vram_indexes_algebra_i4.cpp
               U8 dense/vram_indexes_algebra_u8.cpp
               U4 dense/vram_indexes_algebra_u4.cpp
               U1 dense/vram_indexes_algebra_u1.cpp
    )
endif ()

# A bench over the C interface, @p target from its one unit @p source, linking the library as users do; only its exact
# oracles inline kits, which the host flags pick.
function (usearch_c_interface_bench target source)
    add_executable(${target} ${source})
    usearch_setup_target(${target} HOST)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(
        ${target} PRIVATE fmt::fmt yyjson forkunion::static stringzilla::header ${USEARCH_PUNNED_LIBRARIES}
                          ${CMAKE_DL_LIBS}
    )
endfunction ()
usearch_c_interface_bench(usearch_bench_sequence_index_search sequence_index_search.cpp)
usearch_c_interface_bench(usearch_bench_shapes_index_search shapes_index_search.cpp)
usearch_c_interface_bench(usearch_bench_geospatial_index_search geospatial_index_search.cpp)

# The kernel microbench times the distance kits alone, one row per kernel, dtype, capability tier and shape.
add_executable(usearch_bench_dense_ram_metric dense/ram_metric.cpp)
usearch_setup_target(usearch_bench_dense_ram_metric HOST)
target_include_directories(usearch_bench_dense_ram_metric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usearch_bench_dense_ram_metric PRIVATE fmt::fmt)
