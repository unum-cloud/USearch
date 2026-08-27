# The library every binding links: one type-erasure boundary - `dense.cuh`, compiled here as
# `dense.cpp` - beside the per-dtype shards, each carrying its slice of the instantiation grid
# in a parallel LLVM backend. The CUDA sibling compiles the same shim through nvcc, so the VRAM
# arms join the variant without the portable library ever seeing the CUDA SDK.
add_library(
    usearch_indexes STATIC
    types.cpp
    dense_index.cpp
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
    sequence.cpp
)
setup_target(usearch_indexes)
target_include_directories(usearch_indexes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(usearch_indexes PRIVATE "USEARCH_BUILD_CUDA=0")
target_link_libraries(usearch_indexes PRIVATE ${USEARCH_FORKUNION_LIBRARY} ${USEARCH_STRINGZILLA_LIBRARY})

# The CUDA library skips `setup_target`: its host-compiler flags don't survive the CUDA driver,
# so the target wires the interface includes directly.
if (USEARCH_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(usearch_indexes_cuda STATIC dense_index.cu)
    target_include_directories(
        usearch_indexes_cuda PRIVATE $<TARGET_PROPERTY:${USEARCH_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
                                     ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_definitions(usearch_indexes_cuda PRIVATE "USEARCH_BUILD_CUDA=1")
    set_target_properties(
        usearch_indexes_cuda
        PROPERTIES CXX_STANDARD 20
                   CUDA_STANDARD 20
                   CUDA_ARCHITECTURES "${USEARCH_CUDA_ARCHITECTURES}"
                   POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(usearch_indexes_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries(usearch_indexes_cuda PRIVATE CUDA::cuda_driver)
endif ()
